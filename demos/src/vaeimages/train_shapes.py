# Offline training of the cell-shape VAE for the "Cell shapes" tab of the vaeimages demo.
# NOT part of the browser bundle.
#
# Re-run from the repository root (Python 3 with torch and numpy; a GPU is used if present,
# a CPU takes a few minutes):
#   python3 demos/src/vaeimages/train_shapes.py demos/src/vaeimages/shape_weights.ts [TESTDIR]
# With TESTDIR, the 2000 test masks and labels are also written there as raw bytes
# (shapes_test_x.u8, shapes_test_y.u8) for the checks in demos/test/vae-check.ts.
#
# Data: SYNTHETIC binary cell masks, 28 × 28 pixels, made here with a fixed seed (no external
# data). Four kinds, equally many of each:
#   0 round       an almost circular cell (axis ratio 1–1.25), radius 5–9 px
#   1 elongated   an ellipse with axis ratio 1.7–2.8, any orientation
#   2 dividing    two overlapping discs (a dumbbell), any orientation
#   3 protrusion  a round body with one thin protrusion, any direction
# Every outline gets a small random wobble (Fourier modes 2–5 of the radius). Masks are drawn
# with 4 × 4 supersampling (grey edge pixels) and shifted so the centre of mass is at (14, 14),
# as for MNIST. Three "anomalies" (shapes of none of these kinds: a ring, two separate cells,
# a square) are exported as examples only, never trained on.
#
# Model: identical to train_mnist.py (784 → 192 ReLU → 2 means + 2 log-variances; decoder
# 2 → 192 ReLU → 784 sigmoid; loss = binary cross-entropy + KL to N(0, I)); int8 weights.
import sys, json, base64
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

OUT = sys.argv[1]
TESTDIR = sys.argv[2] if len(sys.argv) > 2 else None
H, Z, EPOCHS, BATCH, SEED = 192, 2, 40, 128, 1
N_TRAIN, N_TEST = 32000, 2000
KINDS = ["round", "elongated", "dividing", "protrusion"]
EX_LABEL = ["a round cell", "an elongated cell", "a dividing cell", "a cell with a protrusion"]
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

SS = 4
g = (np.arange(28 * SS) + 0.5) / SS            # supersampled pixel centres, 0..28
GX, GY = np.meshgrid(g, g)                       # GX = column (x), GY = row (y)


def wobble(rng, th):
    out = np.ones_like(th)
    for k in range(2, 6):
        out += rng.normal(0, 0.03) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    return out


def blob(rng, cx, cy, a, b, phi):
    """Inside of a wobbly ellipse with semi-axes a, b, rotated by phi, centred at (cx, cy)."""
    dx, dy = GX - cx, GY - cy
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    th = np.arctan2(v, u)
    r_ell = 1 / np.sqrt((np.cos(th) / a) ** 2 + (np.sin(th) / b) ** 2)
    return np.hypot(u, v) < r_ell * wobble(rng, th)


def capsule(cx, cy, phi, length, half_w):
    """A thin bar from (cx, cy) in direction phi, with rounded end."""
    dx, dy = GX - cx, GY - cy
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    uc = np.clip(u, 0, length)
    return np.hypot(u - uc, v) < half_w


def shape(rng, kind):
    phi = rng.uniform(0, np.pi)
    if kind == 0:
        R, q = rng.uniform(5, 9), rng.uniform(1, 1.25)
        m = blob(rng, 14, 14, R * np.sqrt(q), R / np.sqrt(q), phi)
    elif kind == 1:
        R, q = rng.uniform(4.5, 7), rng.uniform(1.7, 2.8)
        m = blob(rng, 14, 14, R * np.sqrt(q), R / np.sqrt(q), phi)
    elif kind == 2:
        r = rng.uniform(4, 6.5); d = rng.uniform(1.2, 1.9) * r
        c, s = np.cos(phi) * d / 2, np.sin(phi) * d / 2
        m = blob(rng, 14 + c, 14 + s, r, r * rng.uniform(0.9, 1), phi) | blob(rng, 14 - c, 14 - s, r, r * rng.uniform(0.9, 1), phi)
    else:
        R, L, hw = rng.uniform(5, 7.5), rng.uniform(3, 5.5), rng.uniform(0.9, 1.4)
        phi = rng.uniform(0, 2 * np.pi)
        cx, cy = 14 - np.cos(phi) * L / 3, 14 - np.sin(phi) * L / 3
        m = blob(rng, cx, cy, R, R * rng.uniform(0.9, 1), phi) | capsule(cx, cy, phi, R + L, hw)
    return m


def finish(m):
    img = m.reshape(28, SS, 28, SS).mean(axis=(1, 3))
    s = img.sum()
    cy = (img.sum(1) * np.arange(28)).sum() / s
    cx = (img.sum(0) * np.arange(28)).sum() / s
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    return np.roll(np.roll(img, dy, 0), dx, 1)


def make(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 784), np.float32); y = np.arange(n) % 4
    for i in range(n):
        X[i] = finish(shape(rng, y[i])).ravel()
    return X, y


xtr, ytr = make(N_TRAIN, 11)
xte, yte = make(N_TEST, 12)
print(f"made {N_TRAIN} training and {N_TEST} test masks; mean ink {xtr.mean():.3f}", file=sys.stderr)


def anomalies():
    out = []
    ring = (np.hypot(GX - 14, GY - 14) < 9) & (np.hypot(GX - 14, GY - 14) > 5)
    out.append(("anomaly: ring", finish(ring)))
    two = (np.hypot(GX - 7, GY - 9) < 4.5) | (np.hypot(GX - 21, GY - 19) < 4.5)
    out.append(("anomaly: two cells", finish(two)))
    sq = (np.abs(GX - 14) < 7) & (np.abs(GY - 14) < 7)
    out.append(("anomaly: square", finish(sq)))
    return out


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
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.3)
X = torch.tensor(xtr).to(dev)
gen = torch.Generator(device="cpu").manual_seed(SEED)
for ep in range(EPOCHS):
    perm = torch.randperm(len(X), generator=gen)
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


xt = xte.astype(np.float64)
mu_q, lv_q = np_enc(xt, deq)
rec = np_dec(mu_q, deq)
print(f"test: mean squared pixel error {((rec - xt) ** 2).mean():.4f}; mean encoder sd {np.exp(0.5 * lv_q).mean():.3f}", file=sys.stderr)

pick = np.concatenate([np.where(yte == c)[0][:250] for c in range(4)])
pts = [[round(float(mu_q[i, 0]), 2), round(float(mu_q[i, 1]), 2), int(yte[i])] for i in pick]
u8 = lambda img: base64.b64encode(np.clip(np.round(np.asarray(img).ravel() * 255), 0, 255).astype(np.uint8).tobytes()).decode()
examples = []
for c in range(4):
    for i in np.where(yte == c)[0][300:303]:          # three per kind, not among the scatter points
        examples.append({"label": EX_LABEL[c], "img": u8(xte[i])})
for lab, img in anomalies():
    examples.append({"label": lab, "img": u8(img)})

b64 = lambda a: base64.b64encode(a.tobytes()).decode()
f32 = lambda a: [float(f"{v:.5g}") for v in np.ravel(a)]
out = {"H": H, "Z": Z}
for k in Q:
    out[k] = {"q": b64(Q[k][0]), "s": f32(Q[k][1]), "b": f32(P[k + ".bias"]), "nIn": int(Q[k][0].shape[1]), "nOut": int(Q[k][0].shape[0])}
with open(OUT, "w") as f:
    f.write("// Generated by train_shapes.py (see the command at its top). Do not edit by hand.\n")
    f.write("// Cell-shape VAE, 784-%d-2 / 2-%d-784, trained on SYNTHETIC cell masks; int8 weights (base64).\n" % (H, H))
    f.write("import type { Q8Layer } from \"./mnist_weights\";\n")
    f.write("export const SHAPE_KINDS = " + json.dumps(KINDS) + ";\n")
    f.write("export const SHAPE_VAE: { H: number; Z: number; e1: Q8Layer; em: Q8Layer; ev: Q8Layer; d1: Q8Layer; d2: Q8Layer } = ")
    f.write(json.dumps(out, separators=(",", ":")) + ";\n")
    f.write("/** 1000 test masks (250 per kind): [z1, z2, kind] at the encoder mean. */\n")
    f.write("export const SHAPE_POINTS: [number, number, number][] = " + json.dumps(pts, separators=(",", ":")) + ";\n")
    f.write("/** Example masks (28 × 28 grey bytes, base64): three test masks per kind, then three anomalies. */\n")
    f.write("export const SHAPE_EXAMPLES: { label: string; img: string }[] = " + json.dumps(examples, separators=(",", ":")) + ";\n")
print("wrote", OUT, file=sys.stderr)

if TESTDIR:
    (np.clip(np.round(xte * 255), 0, 255).astype(np.uint8)).tofile(f"{TESTDIR}/shapes_test_x.u8")
    yte.astype(np.uint8).tofile(f"{TESTDIR}/shapes_test_y.u8")
    print("wrote test masks to", TESTDIR, file=sys.stderr)
