"""By-hand numbers of the cellpose/ unit (exercise B1-B10, lecture worked examples).
Run: python3 -I tools/figs/cellpose/numbers.py   (output kept in numbers.txt)
Derived from plan/cellnets-day.md's numbers.py (Cellpose part), extended."""
import math

def sect(t): print("\n=== " + t)

def heat1d(n, steps, src=None):
    """'add 1 at the source, then average with the in-cell neighbours' on n pixels."""
    src = n // 2 if src is None else src
    h = [0.0] * n
    out = []
    for _ in range(steps):
        h[src] += 1
        h = [sum(h[j] for j in (i - 1, i, i + 1) if 0 <= j < n) / len([j for j in (i - 1, i, i + 1) if 0 <= j < n]) for i in range(n)]
        out.append(h[:])
    return out

def arrows(h):
    n = len(h); H = lambda i: h[i] if 0 <= i < n else 0.0
    return [(lambda d: 1 if d > 0 else -1 if d < 0 else 0)(H(i + 1) - H(i - 1)) for i in range(n)]

sect("B1: 5 pixels, source in the middle, three steps")
for k, h in enumerate(heat1d(5, 3), 1): print("step", k, " ".join("%.4f" % v for v in h))
print("arrows after 3 steps:", arrows(heat1d(5, 3)[-1]))
sect("B1: 11 pixels, three steps (the ends are not reached yet)")
h11 = heat1d(11, 3)[-1]
print(" ".join("%.3f" % v for v in h11)); print("arrows:", arrows(h11))
for k in range(1, 8):
    h = heat1d(11, k)[-1]
    if all(v > 0 for v in h): print("all 11 pixels have heat from step", k); break

sect("B2: 5x5 field")
D = {"→": (0, 1), "←": (0, -1), "↑": (-1, 0), "↓": (1, 0), "↗": (-1, 1), "↖": (-1, -1), "↘": (1, 1), "↙": (1, -1), "·": (0, 0)}
F = ["→←↓←←", "→↘↓↙←", "→→·←←", "↗↑↑↘↓", "↗↑↑→·"]
def follow(F, p):
    path = [p]
    while True:
        dy, dx = D[F[p[0]][p[1]]]
        q = (min(4, max(0, p[0] + dy)), min(4, max(0, p[1] + dx)))
        if q in path: return path, path[path.index(q):]
        path.append(q); p = q
for s in [(0, 0), (4, 0), (3, 3)]:
    pth, cyc = follow(F, s); print(s, "->", pth, "cycle", cyc)
def groups(F):
    g = {}
    for i in range(5):
        for j in range(5):
            g.setdefault(tuple(sorted(follow(F, (i, j))[1])), []).append((i, j))
    return g
for k, v in groups(F).items(): print("ends in", k, ":", len(v), "cells")
for (r, c, a) in [(0, 1, "→"), (0, 0, "↘"), (0, 1, "↓")]:
    G = [list(x) for x in F]; G[r][c] = a; G = ["".join(x) for x in G]
    print(f"fix ({r},{c}) = {a}:", sorted(len(v) for v in groups(G).values()))

sect("B3: flow check")
mask = [(1, 0), (1, 0), (0, 1), (1, 0)]
for pred in ([(1, 0), (0.8, 0.6), (0, 1), (-1, 0)], [(1, 0), (0.8, 0.6), (0, 1), (0.6, 0.8)]):
    sq = [(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(pred, mask)]
    print("squared differences", [round(x, 3) for x in sq], "mean", round(sum(sq) / 4, 4))

sect("B4: diameter and rescaling (steps = 200 x diameter / training diameter)")
for train, diam, W, H in [(30, 45, 600, 450), (30, 15, 600, 450), (17, 34, 696, 520)]:
    s = train / diam
    print(f"train {train}, typed {diam}: factor {s:.3f} -> {round(W * s)} x {round(H * s)} pixels ({s * s:.2f} x the pixels), steps {round(200 * diam / train)}")
print("true cells 45 px, typed 15: the network sees cells of", 45 * 2, "px")

sect("B5: 1D flow with one wrong arrow (pixels 1..10), 6 steps")
f = [+1, +1, -1, +1, +1, -1, -1, -1, -1, -1]
for start in range(1, 11):
    x = start; traj = [x]
    for _ in range(6): x = x + f[x - 1]; traj.append(x)
    print(start, traj)
f2 = f[:]; f2[2] = +1
print("correct arrow at 3:", [ (lambda s: [s := s + f2[s - 1] for _ in range(8)][-1])(st) for st in range(1, 11)])

sect("B9: two opposite arrows, bilinear v(x) = 1 - 2x on [0, 1]")
for eta in (1, 0.5, 0.25, 0.75):
    x = 0.0; xs = [x]
    for _ in range(6): x = x + eta * (1 - 2 * x); xs.append(round(x, 4))
    print("step", eta, xs)
print("divergence dv/dx = -2 (a sink); e_{j+1} = (1 - 2 eta) e_j")

sect("B10: longest paths")
for d in (17, 30, 45):
    print(f"disc of diameter {d}: longest path ~ d/2 = {d / 2:.1f} steps; default steps 200 x d / 30 = {200 * d / 30:.0f}")
L = 60
print(f"C-shape with arc length L = {L}, source in the middle of the arc: longest path ~ L/2 = {L / 2:.0f} (+ the half width)")
r_in, r_out, opening = 9, 18, 2 * 55
mid = (r_in + r_out) / 2; arc = mid * math.radians(360 - opening)
print(f"demo C (radii {r_in}-{r_out}, opening {opening} deg): centreline arc {arc:.1f} px, half {arc / 2:.1f}")
