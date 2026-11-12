# Offline training of the tiny U-Net of the unetseg demo (neural-networks day, unit unet/).
# NOT part of the browser bundle.
#
# Re-run from the repository root (Python 3 with torch and numpy; a GPU is used if present,
# a CPU takes a while):
#   1. node_modules/.bin/esbuild demos/src/unetseg/make_data.ts --bundle --platform=node --log-level=warning | node - DATA
#   2. python3 demos/src/unetseg/train_unet.py DATA demos/src/unetseg/unet_weights.ts
# Step 1 draws the SYNTHETIC training images with the demo's own generator (data.ts →
# lib/synth.ts), so the demo's "new image" comes from exactly the training distribution.
#
# Network (input: grey levels / 255, 64 × 64; all convolutions 3 × 3, "same" zero padding, ReLU):
#   e1a 1→8, e1b 8→8                    64 × 64     ─────────────── skip ──────────┐
#   pool; e2a 8→16, e2b 16→16           32 × 32     ──── skip ────┐                 │
#   pool; b1 16→32, b2 32→32            16 × 16 (bottleneck)      │                 │
#   upsample ×2 (nearest), concat ─────────────────────────────── ┘ → d2a 48→16, d2b 16→16
#   upsample ×2 (nearest), concat ────────────────────────────────────────────────── ┘ → d1a 24→8, d1b 8→8
#   out 1×1, 8→1: logit of "nucleus"; probability = sigmoid(logit).
# Weight sets:  unet   (per-pixel binary cross-entropy)
#               noskip (same, but no skip connections: d2a 32→16, d1a 16→8)
#               dice   (soft Dice loss 1 − 2Σpg / (Σp + Σg), per image)
# Training: Adam lr 2e-3 then 5e-4, batch 32, 40 epochs, random flips and 90° rotations, seed 1.
# Int8 export as demos/src/lib/q8.ts. Prints validation Dice at threshold 0.5.
import sys, json, base64
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DATA, OUT = sys.argv[1], sys.argv[2]
EPOCHS, BATCH, SEED = 40, 32, 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
load = lambda n, t: np.fromfile(f"{DATA}/{n}", t)
Xtr = torch.tensor(load("unet_train_x.f32", np.float32).reshape(-1, 1, 64, 64) / 255.0)
Ytr = torch.tensor(load("unet_train_y.u8", np.uint8).reshape(-1, 1, 64, 64).astype(np.float32))
Xva = torch.tensor(load("unet_val_x.f32", np.float32).reshape(-1, 1, 64, 64) / 255.0)
Yva = torch.tensor(load("unet_val_y.u8", np.uint8).reshape(-1, 1, 64, 64).astype(np.float32))
Xref = torch.tensor(load("unet_ref_x.f32", np.float32).reshape(1, 1, 64, 64) / 255.0)
NAMES = ["e1a", "e1b", "e2a", "e2b", "b1", "b2", "d2a", "d2b", "d1a", "d1b", "out"]


class UNet(nn.Module):
    def __init__(self, skip=True):
        super().__init__()
        self.skip = skip
        c = lambda i, o, k=3: nn.Conv2d(i, o, k, padding=k // 2)
        self.e1a, self.e1b, self.e2a, self.e2b = c(1, 8), c(8, 8), c(8, 16), c(16, 16)
        self.b1, self.b2 = c(16, 32), c(32, 32)
        self.d2a, self.d2b = c(48 if skip else 32, 16), c(16, 16)
        self.d1a, self.d1b = c(24 if skip else 16, 8), c(8, 8)
        self.out = c(8, 1, 1)

    def forward(self, x):
        r = F.relu
        e1 = r(self.e1b(r(self.e1a(x))))
        e2 = r(self.e2b(r(self.e2a(F.max_pool2d(e1, 2)))))
        b = r(self.b2(r(self.b1(F.max_pool2d(e2, 2)))))
        u2 = F.interpolate(b, scale_factor=2, mode="nearest")
        d2 = r(self.d2b(r(self.d2a(torch.cat([u2, e2], 1) if self.skip else u2))))
        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = r(self.d1b(r(self.d1a(torch.cat([u1, e1], 1) if self.skip else u1))))
        return self.out(d1)


def soft_dice_loss(logit, y):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((1, 2, 3)) + 1
    den = p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1
    return (1 - num / den).mean()


def augment(x, y, gen):
    k = int(torch.randint(4, (1,), generator=gen))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if torch.rand(1, generator=gen) < 0.5: x, y = x.flip(3), y.flip(3)
    return x, y


def dice_at(net, X, Y, t=0.5):
    with torch.no_grad():
        p = torch.sigmoid(net(X.to(next(net.parameters()).device))).cpu() > t
    tp = (p * Y).sum(); return float(2 * tp / (p.sum() + Y.sum()))


def train(skip, loss):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    net = UNet(skip).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.25)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(Xtr[idx], Ytr[idx], gen)
            xb, yb = xb.to(dev), yb.to(dev)
            lo = net(xb)
            l = F.binary_cross_entropy_with_logits(lo, yb) if loss == "ce" else soft_dice_loss(lo, yb)
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        if ep % 10 == 9: print(f"  epoch {ep + 1}: loss {l.item():.4f}  val Dice {dice_at(net, Xva, Yva):.4f}", file=sys.stderr)
    return net.cpu().eval()


def q8(W):
    W = W.reshape(W.shape[0], -1).astype(np.float64)
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def export(net):
    sd, o = net.state_dict(), {}
    for n in NAMES:
        q, s = q8(sd[n + ".weight"].numpy())
        o[n] = {"q": base64.b64encode(q.tobytes()).decode(), "s": [float(f"{v:.6g}") for v in s],
                "b": [float(f"{v:.6g}") for v in sd[n + ".bias"].numpy()], "nIn": int(q.shape[1]), "nOut": int(q.shape[0])}
    return o


def quantised(net, e):
    st = {}
    for n in NAMES:
        q = np.frombuffer(base64.b64decode(e[n]["q"]), np.int8).astype(np.float64)
        W = q.reshape(e[n]["nOut"], e[n]["nIn"]) * np.array(e[n]["s"])[:, None]
        st[n + ".weight"] = torch.tensor(W.reshape(net.state_dict()[n + ".weight"].shape), dtype=torch.float32)
        st[n + ".bias"] = torch.tensor(e[n]["b"], dtype=torch.float32)
    qn = UNet(net.skip); qn.load_state_dict(st)
    return qn.eval()


sets, info, ref = {}, {}, {}
for name, skip, loss in [("unet", True, "ce"), ("noskip", False, "ce"), ("dice", True, "dice")]:
    print(name, file=sys.stderr)
    net = train(skip, loss)
    e = export(net)
    sets[name] = {"skip": skip, "layers": e}
    qn = quantised(net, e)
    info[name + " val Dice"] = round(dice_at(qn, Xva, Yva), 4)
    with torch.no_grad():
        p = torch.sigmoid(qn(Xref))[0, 0].numpy()
    ref[name] = {"mean": round(float(p.mean()), 5), "p": [round(float(p[y, x]), 5) for y, x in [(10, 10), (32, 32), (50, 20), (20, 50)]]}
    print(f"  {name}: val Dice (int8) {info[name + ' val Dice']}", file=sys.stderr)

with open(OUT, "w") as f:
    f.write("// Generated by train_unet.py (see the command at its top). Do not edit by hand.\n")
    f.write("// Tiny U-Net (8-16-32 channels), int8 weights (base64), one scale per output channel. Synthetic training data.\n")
    f.write("import type { Q8Layer } from \"../lib/q8\";\n")
    f.write("export interface UnetWeights { skip: boolean; layers: Record<string, Q8Layer> }\n")
    f.write("export const UNET_SETS: Record<string, UnetWeights> = " + json.dumps(sets, separators=(",", ":")) + ";\n")
    f.write("/** Validation Dice at threshold 0.5 (400 images, int8 weights), printed by the training script. */\n")
    f.write("export const UNET_INFO: Record<string, number> = " + json.dumps(info) + ";\n")
    f.write("/** Probabilities on the demo's first image (seed 200001), checked by demos/test/neuralnets-test.ts. */\n")
    f.write("export const UNET_REF: Record<string, { mean: number; p: number[] }> = " + json.dumps(ref) + ";\n")
print("wrote", OUT, file=sys.stderr)
