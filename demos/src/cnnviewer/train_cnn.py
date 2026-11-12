# Offline training of the small CNN of the cnnviewer demo (neural-networks day, unit cnn/).
# NOT part of the browser bundle, and the MNIST data is NOT stored in this repository.
#
# Re-run from the repository root (Python 3 with torch and numpy; a GPU is used if present,
# a CPU takes some minutes):
#   python3 demos/src/cnnviewer/train_cnn.py MNIST.npz demos/src/cnnviewer/cnn_weights.ts
# MNIST.npz is the Keras copy of MNIST (keys x_train, y_train, x_test, y_test).
#
# Data: MNIST handwritten digits (Y. LeCun, C. Cortes, C. J. C. Burges; CC BY-SA 3.0 as stated
# in the Keras documentation), and SYNTHETIC cell masks (round / elongated / dividing /
# protrusion), made with the generator of demos/src/vaeimages/train_shapes.py (copied below,
# same seeds), 28 × 28.
#
# Network (about 9k weights): conv 3×3, 1→8, ReLU, 2×2 max pool → conv 3×3, 8→16, ReLU,
# 2×2 max pool → flatten (16 × 7 × 7 = 784, channel-major) → dense 784→classes → softmax.
# "same" zero padding. Loss: cross-entropy; Adam, lr 1e-3, batch 128, fixed seed.
#
# Weight sets written (int8, one scale per output channel, see demos/src/lib/q8.ts):
#   digits      MNIST, 10 epochs, no augmentation
#   digitsAug   MNIST, 10 epochs, every batch randomly rotated by up to ±45°
#   digitsE0/E1/E3   snapshots of "digits" after 0 (initial), 1 and 3 epochs
#   shapes      cell masks, 10 epochs (4 classes)
# plus, for "digits": the 9 test patches (8 × 8 = the receptive field of a layer-2 unit) that
# excite each layer-2 channel most, 8 example cell masks, and reference logits for
# demos/test/neuralnets-test.ts. The script prints test accuracies (also on rotated digits).
import sys, json, base64, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

NPZ, OUT = sys.argv[1], sys.argv[2]
SEED, EPOCHS, BATCH = 1, 10, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(NPZ)
xtr = torch.tensor(d["x_train"] / 255.0, dtype=torch.float32)[:, None]
ytr = torch.tensor(d["y_train"], dtype=torch.long)
xte = torch.tensor(d["x_test"] / 255.0, dtype=torch.float32)[:, None]
yte = torch.tensor(d["y_test"], dtype=torch.long)

# ---- synthetic cell masks (copied from vaeimages/train_shapes.py) ----
SS = 4
g = (np.arange(28 * SS) + 0.5) / SS
GX, GY = np.meshgrid(g, g)


def wobble(rng, th):
    out = np.ones_like(th)
    for k in range(2, 6):
        out += rng.normal(0, 0.03) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    return out


def blob(rng, cx, cy, a, b, phi):
    dx, dy = GX - cx, GY - cy
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    th = np.arctan2(v, u)
    r_ell = 1 / np.sqrt((np.cos(th) / a) ** 2 + (np.sin(th) / b) ** 2)
    return np.hypot(u, v) < r_ell * wobble(rng, th)


def capsule(cx, cy, phi, length, half_w):
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
        r = rng.uniform(4, 6.5); dd = rng.uniform(1.2, 1.9) * r
        c, s = np.cos(phi) * dd / 2, np.sin(phi) * dd / 2
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


sx, sy = make(32000, 11)
stx, sty = make(2000, 12)
sx = torch.tensor(sx).view(-1, 1, 28, 28); sy = torch.tensor(sy, dtype=torch.long)
stx = torch.tensor(stx).view(-1, 1, 28, 28); sty = torch.tensor(sty, dtype=torch.long)


class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.c1, self.c2, self.fc = nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1), nn.Linear(784, k)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def rotate(x, deg):
    """Rotate a batch by per-image angles (degrees, counter-clockwise on screen), zero fill."""
    a = torch.as_tensor(deg, dtype=torch.float32, device=x.device) * math.pi / 180
    c, s = torch.cos(a), torch.sin(a)
    theta = torch.stack([torch.stack([c, s, torch.zeros_like(c)], 1), torch.stack([-s, c, torch.zeros_like(c)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def accuracy(net, X, Y, deg=0.0):
    net.eval(); ok = 0
    nd = next(net.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), 1000):
            xb = X[i:i + 1000].to(nd)
            if deg: xb = rotate(xb, torch.full((len(xb),), deg))
            ok += (net(xb).argmax(1).cpu() == Y[i:i + 1000]).sum().item()
    net.train()
    return ok / len(X)


def train(X, Y, k, aug, snaps=()):
    torch.manual_seed(SEED)
    net = Net(k).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = {}
    if 0 in snaps: out[0] = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            if aug: xb = rotate(xb, (torch.rand(len(xb), generator=gen) * 2 - 1) * aug)
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in snaps: out[ep] = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
        print(f"  epoch {ep}: loss {loss.item():.4f}", file=sys.stderr)
    return net.cpu().eval(), out


def q8(W):
    W = W.reshape(W.shape[0], -1).astype(np.float64)
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def export(sd):
    o = {}
    for name in ["c1", "c2", "fc"]:
        W = sd[name + ".weight"].numpy()
        q, s = q8(W)
        o[name] = {"q": base64.b64encode(q.tobytes()).decode(), "s": [float(f"{v:.6g}") for v in s],
                   "b": [float(f"{v:.6g}") for v in sd[name + ".bias"].numpy()], "nIn": int(q.shape[1]), "nOut": int(q.shape[0])}
    return o


def quantised_net(sd, k):
    """The network with the int8-dequantised weights the browser uses (biases rounded as exported)."""
    net = Net(k)
    e = export(sd)
    st = {}
    for name in ["c1", "c2", "fc"]:
        q = np.frombuffer(base64.b64decode(e[name]["q"]), np.int8).astype(np.float64)
        W = q.reshape(e[name]["nOut"], e[name]["nIn"]) * np.array(e[name]["s"])[:, None]
        st[name + ".weight"] = torch.tensor(W.reshape(sd[name + ".weight"].shape), dtype=torch.float32)
        st[name + ".bias"] = torch.tensor(e[name]["b"], dtype=torch.float32)
    net.load_state_dict(st)
    return net.eval()


sets, info = {}, {}
print("digits, no augmentation", file=sys.stderr)
net, snaps = train(xtr, ytr, 10, 0, snaps=(0, 1, 3))
sets["digits"] = export(net.state_dict())
for e in (0, 1, 3): sets[f"digitsE{e}"] = export(snaps[e])
qn = quantised_net(net.state_dict(), 10)
for deg in (0, 30, 45, 90, 180):
    info[f"digits acc rot {deg}"] = accuracy(qn, xte, yte, deg)
for e in (0, 1, 3):
    info[f"digitsE{e} acc"] = accuracy(quantised_net(snaps[e], 10), xte, yte)
print("digits, rotation augmentation ±45°", file=sys.stderr)
neta, _ = train(xtr, ytr, 10, 45)
sets["digitsAug"] = export(neta.state_dict())
qa = quantised_net(neta.state_dict(), 10)
for deg in (0, 30, 45, 90, 180):
    info[f"digitsAug acc rot {deg}"] = accuracy(qa, xte, yte, deg)
print("cell masks", file=sys.stderr)
nets, _ = train(sx, sy, 4, 0)
sets["shapes"] = export(nets.state_dict())
info["shapes acc"] = accuracy(quantised_net(nets.state_dict(), 4), stx, sty)

# top-9 patches per layer-2 channel (digits net, test set): activation after ReLU, before pooling
with torch.no_grad():
    a2 = F.relu(qn.c2(F.max_pool2d(F.relu(qn.c1(xte)), 2)))     # N × 16 × 14 × 14
patches = []
pad = F.pad(xte, (2, 6, 2, 6))                                    # unit (i, j) sees input rows 2i−2 .. 2i+5
for c in range(16):
    v = a2[:, c].reshape(-1)
    top, used = [], set()
    for idx in torch.argsort(v, descending=True).tolist():
        n, rest = divmod(idx, 196); i, j = divmod(rest, 14)
        if n in used: continue
        used.add(n)
        p = pad[n, 0, 2 * i: 2 * i + 8, 2 * j: 2 * j + 8]
        top.append(base64.b64encode((p.numpy() * 255).round().astype(np.uint8).tobytes()).decode())
        if len(top) == 9: break
    patches.append(top)

# 8 example cell masks (two per kind) and reference logits
ex_shapes = [base64.b64encode((stx[i, 0].numpy() * 255).round().astype(np.uint8).tobytes()).decode() for i in range(8)]
ex_labels = [int(sty[i]) for i in range(8)]
with torch.no_grad():
    ref_digits = qn(xte[:3]).numpy().round(5).tolist()
    ref_shapes = quantised_net(nets.state_dict(), 4)(stx[:2]).numpy().round(5).tolist()
ref = {"digitsTestImages": [base64.b64encode((xte[i, 0].numpy() * 255).round().astype(np.uint8).tobytes()).decode() for i in range(3)],
       "digitsLogits": ref_digits, "shapesLogits": ref_shapes}

for k, v in info.items(): print(f"{k}: {v:.4f}", file=sys.stderr)
with open(OUT, "w") as f:
    f.write("// Generated by train_cnn.py (see the command at its top). Do not edit by hand.\n")
    f.write("// Small CNN (conv 3x3x8, pool, conv 3x3x16, pool, dense), int8 weights (base64), one scale per output channel.\n")
    f.write("// Data: MNIST (LeCun, Cortes & Burges; CC BY-SA 3.0 as stated in the Keras documentation) and synthetic cell masks.\n")
    f.write("import type { Q8Layer } from \"../lib/q8\";\n")
    f.write("export interface CnnWeights { c1: Q8Layer; c2: Q8Layer; fc: Q8Layer }\n")
    f.write("export const CNN_SETS: Record<string, CnnWeights> = " + json.dumps(sets, separators=(",", ":")) + ";\n")
    f.write("/** Test accuracies printed by the training script (int8 weights). */\n")
    f.write("export const CNN_INFO: Record<string, number> = " + json.dumps({k: round(v, 4) for k, v in info.items()}) + ";\n")
    f.write("/** For each layer-2 channel of \"digits\": the 9 most exciting 8 × 8 test patches (grey bytes, base64). */\n")
    f.write("export const TOP_PATCHES: string[][] = " + json.dumps(patches, separators=(",", ":")) + ";\n")
    f.write("/** Synthetic test cell masks (28 × 28 grey bytes, base64) with their class 0..3. */\n")
    f.write("export const SHAPE_EXAMPLES: { img: string; c: number }[] = " + json.dumps([{"img": a, "c": b} for a, b in zip(ex_shapes, ex_labels)], separators=(",", ":")) + ";\n")
    f.write("/** Reference outputs of the int8 networks, checked by demos/test/neuralnets-test.ts. */\n")
    f.write("export const CNN_REF = " + json.dumps(ref, separators=(",", ":")) + ";\n")
print("wrote", OUT, file=sys.stderr)
