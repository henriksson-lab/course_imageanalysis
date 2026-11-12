# Offline self-supervised training of the tiny encoder of the embedprobe demo (modern deep
# learning day, unit attention/). NOT part of the browser bundle.
#
# Re-run from the repository root (Python 3 with torch, numpy and Pillow; a GPU is used if
# present, a CPU takes about ten minutes):
#   1. node_modules/.bin/esbuild demos/src/embedprobe/make_data.ts --bundle --platform=node --log-level=warning | node - DATA
#   2. python3 demos/src/embedprobe/train_ssl.py DATA demos/src/embedprobe/ssl_weights.ts
# Data: SYNTHETIC images drawn by the demo's own generator (scene.ts; seeds 1..1400 for
# training, 1401..1500 for the probe evaluation) and random crops of the course micrograph
# demos/src/fourier2d/cell256.png (R. M. Hunt, CC BY 3.0, Wikimedia Commons), shown at
# 128 × 128 as in the demo. The masks are NEVER used to train the encoder.
#
# Encoder (input: grey levels / 255, 128 × 128; output 16-D per pixel of a 64 × 64 grid):
#   c1 1→12 3×3 stride 2 · c2 12→16 3×3 · c3 16→16 3×3 dilation 2 · c4 16→16 3×3 dilation 4
#   (all "same" zero padding, ReLU) · c5 16→16 1×1 · L2 normalisation. 7.2 k weights.
# Dense contrastive training (InfoNCE, temperature 0.1): two views of the same 64 × 64 crop
# shifted by up to ±4 pixels, with independent photometric augmentations (gain 0.35–1.6, offset, gamma 0.7–1.4, blur
# σ 0–1.5, noise sd 0–12 grey levels; the same random flip / 90° rotation for both views).
# For 64 random grid positions per crop, the embedding in view 1 must pick out the SAME
# image point in view 2 among all positions of the batch, except points of the same image
# closer than 12 pixels (they need not be told apart: this makes the features smooth). No hue / colour augmentation (the
# images are grey). Adam lr 2e-3, batch 32, 8000 steps, seed 1. Int8 export as lib/q8.ts.
# Also writes: the probe accuracy (logistic regression on frozen embeddings vs raw 9 × 9
# patches), the label-efficiency curve (labelled pixels → Dice on the evaluation images;
# frozen probe vs the same network trained from scratch; mean of 5 seeds) and reference
# embeddings of the demo image for demos/test/moderndl-test.ts.
import sys, json, base64
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

DATA, OUT = sys.argv[1], sys.argv[2]
SEED, STEPS, BATCH, CROP, NPOS, TAU = 1, 8000, 32, 64, 64, 0.1
torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
X = np.fromfile(f"{DATA}/ssl_x.u8", np.uint8).reshape(-1, 128, 128).astype(np.float32) / 255
Y = np.fromfile(f"{DATA}/ssl_y.u8", np.uint8).reshape(-1, 128, 128)
REF = np.fromfile(f"{DATA}/ssl_ref.u8", np.uint8).reshape(128, 128).astype(np.float32) / 255
Xtr, Xev, Yev = X[:1400], X[1400:], Y[1400:]
cell = np.asarray(Image.open("demos/src/fourier2d/cell256.png").convert("L").resize((128, 128), Image.BOX), np.float32) / 255
NAMES = ["c1", "c2", "c3", "c4", "c5"]


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(12, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 16, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(16, 16, 3, padding=4, dilation=4)
        self.c5 = nn.Conv2d(16, 16, 1)

    def forward(self, x):
        r = F.relu
        z = self.c5(r(self.c4(r(self.c3(r(self.c2(r(self.c1(x)))))))))
        return F.normalize(z, dim=1)


JIT = 4   # views are shifted against each other by up to ±JIT pixels (even numbers)


def crops(gen):
    """Two crops of the same image, the second shifted by (dy, dx); returns both and the shifts (grid units)."""
    o1 = np.empty((BATCH, 1, CROP, CROP), np.float32); o2 = np.empty_like(o1); sh = np.zeros((BATCH, 2), int)
    for b in range(BATCH):
        src = cell if gen.random() < 0.15 else Xtr[gen.integers(len(Xtr))]
        k, fl = gen.integers(4), gen.random() < 0.5
        src = np.rot90(src, k)
        if fl: src = src[:, ::-1]
        dy, dx = 2 * gen.integers(-JIT // 2, JIT // 2 + 1, 2)
        y, x = gen.integers(JIT, 128 - CROP - JIT + 1, 2)
        o1[b, 0] = src[y:y + CROP, x:x + CROP]
        o2[b, 0] = src[y + dy:y + dy + CROP, x + dx:x + dx + CROP]
        sh[b] = (dy // 2, dx // 2)
    return torch.tensor(o1), torch.tensor(o2), sh


def blur(x, s):
    if s < 0.3: return x
    k = torch.arange(-3, 4, dtype=torch.float32, device=x.device)
    k = torch.exp(-k ** 2 / (2 * s * s)); k = k / k.sum()
    x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode="replicate"), k.view(1, 1, 1, 7))
    return F.conv2d(F.pad(x, (0, 0, 3, 3), mode="replicate"), k.view(1, 1, 7, 1))


def augment(x, g):
    B = x.shape[0]
    u = lambda lo, hi: torch.tensor(lo + (hi - lo) * g.random(B), dtype=torch.float32, device=x.device).view(B, 1, 1, 1)
    gain = torch.exp(u(np.log(0.35), np.log(1.6))); off = u(-0.08, 0.08); gam = torch.exp(u(np.log(0.7), np.log(1.4)))
    y = (x.clamp(0, 1) ** gam) * gain + off
    y = blur(y, 1.5 * g.random())
    return y + u(0, 12 / 255) * torch.randn_like(y)


def train():
    g = np.random.default_rng(SEED)
    net = Enc().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    G = CROP // 2
    for step in range(STEPS):
        x1, x2, sh = crops(g)
        z1, z2 = net(augment(x1.to(dev), g)), net(augment(x2.to(dev), g))     # B × 16 × 32 × 32
        # positions in view 1 (away from the border) and the same image points in view 2
        m = JIT // 2
        py = g.integers(m, G - m, (BATCH, NPOS)); px = g.integers(m, G - m, (BATCH, NPOS))
        i1 = torch.tensor(py * G + px, device=dev)
        i2 = torch.tensor((py - sh[:, :1]) * G + (px - sh[:, 1:]), device=dev)
        f = lambda z, idx: torch.gather(z.flatten(2), 2, idx.unsqueeze(1).expand(-1, 16, -1)).permute(0, 2, 1).reshape(-1, 16)
        a, b = f(z1, i1), f(z2, i2)
        logits = a @ b.T / TAU
        # do not ask the network to tell apart points of the same image closer than 6 grid cells
        img = torch.arange(BATCH, device=dev).repeat_interleave(NPOS)
        Py = torch.tensor(py.reshape(-1), device=dev); Px = torch.tensor(px.reshape(-1), device=dev)
        near = (img[:, None] == img[None, :]) & ((Py[:, None] - Py[None, :]) ** 2 + (Px[:, None] - Px[None, :]) ** 2 < 36)
        near.fill_diagonal_(False)
        logits = logits.masked_fill(near, -1e4)
        t = torch.arange(len(a), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 499: print(f"  step {step + 1}: loss {loss.item():.3f}", file=sys.stderr)
    return net.eval()


def q8(W):
    W = W.reshape(W.shape[0], -1).astype(np.float64)
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8), s


def export(net):
    sd, o = net.state_dict(), {}
    for n in NAMES:
        q, s = q8(sd[n + ".weight"].cpu().numpy())
        o[n] = {"q": base64.b64encode(q.tobytes()).decode(), "s": [float(f"{v:.6g}") for v in s],
                "b": [float(f"{v:.6g}") for v in sd[n + ".bias"].cpu().numpy()], "nIn": int(q.shape[1]), "nOut": int(q.shape[0])}
    return o


def quantised(e):
    net = Enc(); st = {}
    for n in NAMES:
        q = np.frombuffer(base64.b64decode(e[n]["q"]), np.int8).astype(np.float64)
        W = q.reshape(e[n]["nOut"], e[n]["nIn"]) * np.array(e[n]["s"])[:, None]
        st[n + ".weight"] = torch.tensor(W.reshape(net.state_dict()[n + ".weight"].shape), dtype=torch.float32)
        st[n + ".bias"] = torch.tensor(e[n]["b"], dtype=torch.float32)
    net.load_state_dict(st)
    return net.eval()


def embed(net, imgs):
    with torch.no_grad():
        return net(torch.tensor(imgs[:, None])).numpy().transpose(0, 2, 3, 1)   # n × 64 × 64 × 16


def raw_patches(imgs):
    p = np.pad(imgs, ((0, 0), (4, 4), (4, 4)), mode="edge")
    out = np.stack([p[:, 4 + dy:4 + dy + 128:2, 4 + dx:4 + dx + 128:2] for dy in range(-4, 5) for dx in range(-4, 5)], -1)
    return out                                                                    # n × 64 × 64 × 81


def logreg(Z, y, iters=400, lr=0.5, l2=1e-3):
    Zb = np.concatenate([Z, np.ones((len(Z), 1))], 1); w = np.zeros(Zb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Zb @ w)); w -= lr * (Zb.T @ (p - y) / len(y) + l2 * w)
    return w


def dice(pred, truth):
    return float(2 * (pred & truth).sum() / max(1, pred.sum() + truth.sum()))


def probe_dice(Ztr, ytr, Zev, yev):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    w = logreg((Ztr - mu) / sd, ytr)
    Zb = np.concatenate([(Zev - mu) / sd, np.ones((len(Zev), 1))], 1)
    return dice(Zb @ w > 0, yev.astype(bool))


print("training", file=sys.stderr)
net = train().cpu()
e = export(net)
qn = quantised(e)

# --- probes: one training image, evaluation on 100 held-out images ---
ytr_full = Y[0, ::2, ::2].reshape(-1)
yev_full = Yev[:, ::2, ::2].reshape(-1)
Es_tr, Es_ev = embed(qn, X[:1]).reshape(-1, 16), embed(qn, Xev).reshape(-1, 16)
Er_tr, Er_ev = raw_patches(X[:1]).reshape(-1, 81), raw_patches(Xev).reshape(-1, 81)
info = {"probe Dice, learned features (all pixels of one image)": round(probe_dice(Es_tr, ytr_full, Es_ev, yev_full), 3),
        "probe Dice, raw 9x9 patches (all pixels of one image)": round(probe_dice(Er_tr, ytr_full, Er_ev, yev_full), 3)}
print(info, file=sys.stderr)

# --- label-efficiency curve: n labelled pixels (half nucleus, half background) ---
def scratch_dice(pix, lab, seed):
    torch.manual_seed(seed)
    m = Enc(); head = nn.Conv2d(16, 1, 1)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), lr=3e-3)
    x = torch.tensor(X[:1, None]); t = torch.tensor(lab, dtype=torch.float32)
    for _ in range(300):
        lo = head(m(x)).flatten()[torch.tensor(pix)]
        l = F.binary_cross_entropy_with_logits(lo, t); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        p = head(m(torch.tensor(Xev[:, None]))).numpy().reshape(-1) > 0
    return dice(p, yev_full.astype(bool))

NS = [4, 10, 30, 100, 300]
curve = {"n": NS, "learned (frozen, probe)": [], "raw patches (probe)": [], "from scratch (same network)": []}
for n in NS:
    acc = {k: [] for k in curve if k != "n"}
    for s in range(5):
        gs = np.random.default_rng(100 + s)
        pos, neg = np.flatnonzero(ytr_full == 1), np.flatnonzero(ytr_full == 0)
        pix = np.concatenate([gs.choice(pos, n // 2, replace=False), gs.choice(neg, n - n // 2, replace=False)])
        lab = ytr_full[pix]
        acc["learned (frozen, probe)"].append(probe_dice(Es_tr[pix], lab, Es_ev, yev_full))
        acc["raw patches (probe)"].append(probe_dice(Er_tr[pix], lab, Er_ev, yev_full))
        acc["from scratch (same network)"].append(scratch_dice(pix, lab, s))
    for k, v in acc.items(): curve[k].append(round(float(np.mean(v)), 3))
    print(n, {k: curve[k][-1] for k in acc}, file=sys.stderr)

# --- reference embeddings of the demo image ---
zr = embed(qn, REF[None])[0]
ref = {"pos": [[8, 8], [32, 32], [50, 20], [20, 50]], "z": [[round(float(v), 5) for v in zr[y, x]] for y, x in [[8, 8], [32, 32], [50, 20], [20, 50]]]}

with open(OUT, "w") as f:
    f.write("// Generated by train_ssl.py (see the command at its top). Do not edit by hand.\n")
    f.write("// Tiny self-supervised encoder (7.2 k weights), int8 (base64), one scale per output channel.\n")
    f.write("import type { Q8Layer } from \"../lib/q8\";\n")
    f.write("export const SSL_LAYERS: Record<string, Q8Layer> = " + json.dumps(e, separators=(",", ":")) + ";\n")
    f.write("/** Probe quality on 100 held-out synthetic images, printed by the training script. */\n")
    f.write("export const SSL_INFO: Record<string, number> = " + json.dumps(info) + ";\n")
    f.write("/** Label efficiency: Dice on held-out images vs number of labelled pixels (mean of 5 seeds). */\n")
    f.write("export const SSL_CURVE: Record<string, number[]> = " + json.dumps(curve) + ";\n")
    f.write("/** Embeddings of the demo image (seed 900001) at grid positions [y, x], for the tests. */\n")
    f.write("export const SSL_REF = " + json.dumps(ref) + ";\n")
print("wrote", OUT, file=sys.stderr)
