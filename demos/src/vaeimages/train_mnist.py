# Offline training of the MNIST digit VAE for the "Digits" tab of the vaegen demo.
# NOT part of the browser bundle, and the MNIST data is NOT stored in this repository.
#
# Re-run (Python 3 with torch and numpy; a GPU is used if present, a CPU takes a few minutes):
#   1. download the four MNIST files into some folder DATA (outside the repo), e.g.
#        for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
#          curl -sSfLO https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz; done
#   2. python3 demos/src/vaegen/train_mnist.py DATA demos/src/vaegen/mnist_weights.ts
#
# Data: MNIST handwritten digits (Y. LeCun, C. Cortes, C. J. C. Burges), 60000 training and
# 10000 test images of 28×28 grey pixels. The Keras documentation states that LeCun and
# Cortes hold the copyright and that MNIST is available under CC BY-SA 3.0.
#
# Model (as in Kingma & Welling 2014): encoder 784 → 192 ReLU → 2 means + 2 log-variances;
# decoder 2 → 192 ReLU → 784 sigmoid (pixel "ink" probabilities); loss = binary cross-entropy
# + KL to a standard normal. Fixed seed. The exported weights are int8 with one scale per
# output unit (row), base64-encoded; biases stay float. The script prints the checks
# (test reconstruction error, kNN accuracy in 2D, float vs int8 decoder difference).
import gzip, sys, json, base64
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DATA, OUT = sys.argv[1], sys.argv[2]
H, Z, EPOCHS, BATCH, SEED = 192, 2, 60, 128, 1
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def idx(name):
    with gzip.open(f"{DATA}/{name}.gz") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


xtr = torch.tensor(idx("train-images-idx3-ubyte").reshape(-1, 784) / 255.0, dtype=torch.float32)
ytr = idx("train-labels-idx1-ubyte")
xte = torch.tensor(idx("t10k-images-idx3-ubyte").reshape(-1, 784) / 255.0, dtype=torch.float32)
yte = idx("t10k-labels-idx1-ubyte")


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.em, self.ev = nn.Linear(784, H), nn.Linear(H, Z), nn.Linear(H, Z)
        self.d1, self.d2 = nn.Linear(Z, H), nn.Linear(H, 784)

    def enc(self, x):
        h = F.relu(self.e1(x))
        return self.em(h), self.ev(h)

    def dec_logits(self, z):
        return self.d2(F.relu(self.d1(z)))


net = VAE().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.3)
X = xtr.to(dev)
g = torch.Generator(device="cpu").manual_seed(SEED)
for ep in range(EPOCHS):
    perm = torch.randperm(len(X), generator=g)
    tot = 0.0
    for b in range(0, len(X), BATCH):
        x = X[perm[b:b + BATCH]]
        mu, lv = net.enc(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        bce = F.binary_cross_entropy_with_logits(net.dec_logits(z), x, reduction="sum")
        kl = 0.5 * torch.sum(mu ** 2 + lv.exp() - lv - 1)
        loss = (bce + kl) / len(x)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x)
    sched.step()
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print(f"epoch {ep}: loss per image {tot / len(X):.1f}", file=sys.stderr)

net = net.cpu().eval()
P = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}


def q8(W):
    """int8 per row: W ≈ q * s[:, None]."""
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
    return q, s


Q = {k: q8(P[k + ".weight"]) for k in ["e1", "em", "ev", "d1", "d2"]}
deq = {k: Q[k][0].astype(np.float64) * Q[k][1][:, None] for k in Q}


def np_enc(x, Wd):
    h = np.maximum(0, x @ Wd["e1"].T + P["e1.bias"])
    return h @ Wd["em"].T + P["em.bias"], h @ Wd["ev"].T + P["ev.bias"]


def np_dec(z, Wd):
    h = np.maximum(0, z @ Wd["d1"].T + P["d1.bias"])
    return 1 / (1 + np.exp(-(h @ Wd["d2"].T + P["d2.bias"])))


Wf = {k: P[k + ".weight"] for k in Q}
xt = xte.numpy().astype(np.float64)
mu_f, lv_f = np_enc(xt, Wf)
mu_q, lv_q = np_enc(xt, deq)
rec_f, rec_q = np_dec(mu_f, Wf), np_dec(mu_q, deq)
eps = 1e-7
bce = lambda r: -(xt * np.log(r + eps) + (1 - xt) * np.log(1 - r + eps)).sum(1).mean()
print(f"test reconstruction (z = encoder mean): BCE per image {bce(rec_f):.1f} (float), {bce(rec_q):.1f} (int8); "
      f"mean squared pixel error {((rec_f - xt) ** 2).mean():.4f}; baseline 'mean image' BCE "
      f"{bce(np.tile(xtr.numpy().mean(0), (len(xt), 1))):.1f}", file=sys.stderr)
print(f"int8 vs float decoder on the test means: max |pixel diff| {np.abs(rec_f - rec_q).max():.4f}, "
      f"mean {np.abs(rec_f - rec_q).mean():.5f}; latent mean shift {np.abs(mu_f - mu_q).max():.4f}", file=sys.stderr)
# kNN (k = 15) accuracy in the 2D latent space: training-set encoder means as reference
mu_tr, _ = np_enc(xtr.numpy().astype(np.float64), deq)
A, B = torch.tensor(mu_q), torch.tensor(mu_tr)
pred = []
for b in range(0, len(A), 1000):
    d = torch.cdist(A[b:b + 1000], B)
    nn_idx = d.topk(15, largest=False).indices.numpy()
    pred += [np.bincount(ytr[r], minlength=10).argmax() for r in nn_idx]
print(f"15-NN accuracy of test digits in the 2D latent space: {(np.array(pred) == yte).mean():.3f}; "
      f"mean encoder sd {np.exp(0.5 * lv_q).mean():.3f}", file=sys.stderr)

# 1000 test digits (100 per class, the first of each) as the faint background scatter
pick = np.concatenate([np.where(yte == c)[0][:100] for c in range(10)])
pts = [[round(float(mu_q[i, 0]), 2), round(float(mu_q[i, 1]), 2), int(yte[i])] for i in pick]
b64 = lambda a: base64.b64encode(a.tobytes()).decode()
f32 = lambda a: [float(f"{v:.5g}") for v in np.ravel(a)]
out = {"H": H, "Z": Z}
for k in Q:
    out[k] = {"q": b64(Q[k][0]), "s": f32(Q[k][1]), "b": f32(P[k + ".bias"]), "nIn": int(Q[k][0].shape[1]), "nOut": int(Q[k][0].shape[0])}
with open(OUT, "w") as f:
    f.write("// Generated by train_mnist.py (see the command at its top). Do not edit by hand.\n")
    f.write("// MNIST digit VAE, 784-%d-2 / 2-%d-784; int8 weights (base64) with a float scale per row.\n" % (H, H))
    f.write("// Data: MNIST (LeCun, Cortes & Burges), CC BY-SA 3.0 as stated in the Keras documentation.\n")
    f.write("export interface Q8Layer { q: string; s: number[]; b: number[]; nIn: number; nOut: number }\n")
    f.write("export const MNIST_VAE: { H: number; Z: number; e1: Q8Layer; em: Q8Layer; ev: Q8Layer; d1: Q8Layer; d2: Q8Layer } = ")
    f.write(json.dumps(out, separators=(",", ":")) + ";\n")
    f.write("/** 1000 test digits: [z1, z2, digit] at the encoder mean. */\n")
    f.write("export const MNIST_POINTS: [number, number, number][] = " + json.dumps(pts, separators=(",", ":")) + ";\n")
print("wrote", OUT, file=sys.stderr)

if len(sys.argv) > 3:   # optional: PNG comparing float and int8 decoders on a 15×15 latent grid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gz = np.linspace(-2.5, 2.5, 15)
    zz = np.array([[a, b] for b in gz[::-1] for a in gz])
    tile = lambda r: r.reshape(15, 15, 28, 28).transpose(0, 2, 1, 3).reshape(420, 420)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, t in zip(ax, [tile(np_dec(zz, Wf)), tile(np_dec(zz, deq)), np.abs(tile(np_dec(zz, Wf)) - tile(np_dec(zz, deq)))],
                         ["float32", "int8", "|difference|"]):
        a.imshow(img, cmap="gray_r", vmin=0, vmax=1); a.set_title(t); a.axis("off")
    fig.savefig(sys.argv[3], dpi=80, bbox_inches="tight")
