"""By-hand numbers of the yolo/ unit (lecture and exercise). Run: python3 -I tools/figs/yolo/numbers.py
Re-derived from the day plan's script (plan/cellnets-day.md, "Numbers"); the anchor and heat-map
questions moved from detection/ keep the numbers computed there."""
import math

def sect(t): print("\n=== " + t)
def biou(a, c):
    iw = max(0, min(a[2], c[2]) - max(a[0], c[0])); ih = max(0, min(a[3], c[3]) - max(a[1], c[1]))
    i = iw * ih; return i / ((a[2]-a[0])*(a[3]-a[1]) + (c[2]-c[0])*(c[3]-c[1]) - i)
def giou(a, c):
    iw = max(0, min(a[2], c[2]) - max(a[0], c[0])); ih = max(0, min(a[3], c[3]) - max(a[1], c[1]))
    i = iw * ih; u = (a[2]-a[0])*(a[3]-a[1]) + (c[2]-c[0])*(c[3]-c[1]) - i
    C = (max(a[2], c[2]) - min(a[0], c[0])) * (max(a[3], c[3]) - min(a[1], c[1]))
    return i / u - (C - u) / C
sig = lambda x: 1 / (1 + math.exp(-x))

sect("B2: output cells and numbers")
for size in (256, 512, 640):
    g = [size // s for s in (8, 16, 32)]; n = sum(x * x for x in g)
    print(size, g, n, "cells;", n * 144, "numbers with 80 classes (x 144);", n * 66, "with 2 classes")
print("reach of 15 bins:", [15 * s for s in (8, 16, 32)], "px; a box 400 wide centred on a cell needs 200 per side:",
      [s for s in (8, 16, 32) if 15 * s >= 200])

sect("B1: decode one cell")
E = lambda p: sum(k * v for k, v in p.items())
d = [E({2: .5, 3: .5}), E({1: 1}), E({1: .5, 2: .5}), E({2: 1})]
s, cx, cy = 16, 7.5, 5.5
box = ((cx - d[0]) * s, (cy - d[1]) * s, (cx + d[2]) * s, (cy + d[3]) * s)
print("centre", cx * s, cy * s, "d", d, "box", box, "IoU with (84,70,150,122): %.4f" % biou(box, (84, 70, 150, 122)))
for lg in (1.2, -0.5, -3.0): print("sigmoid(%g) = %.3f" % (lg, sig(lg)))
print("two peaks 0.6@2 + 0.4@9 =", 2 * .6 + 9 * .4)

sect("B3: NMS with classes")
B = {"b1": ((10, 10, 50, 50), .90, "cell"), "b2": ((14, 12, 54, 52), .80, "cell"), "b3": ((40, 10, 80, 50), .75, "cell"),
     "b4": ((12, 10, 52, 50), .60, "debris"), "b5": ((44, 14, 84, 54), .50, "cell"), "b6": ((100, 100, 120, 120), .20, "cell")}
names = list(B)
for i in range(6):
    for j in range(i + 1, 6):
        v = biou(B[names[i]][0], B[names[j]][0])
        if v > 0: print(names[i], names[j], round(v, 3))
def greedy(t, agn, conf=0.25):
    alive = [n for n in sorted(names, key=lambda n: -B[n][1]) if B[n][1] > conf]; kept = []
    while alive:
        k = alive.pop(0); kept.append(k)
        alive = [n for n in alive if not ((agn or B[n][2] == B[k][2]) and biou(B[n][0], B[k][0]) > t)]
    return kept
for t in (0.5, 0.2): print("t", t, "aware", greedy(t, False), "agnostic", greedy(t, True))

sect("B6: two cells 8 px wide, centres 6 px apart (side by side); all grid offsets")
def report(st, n=64):
    both = one = none = 0
    for ox in range(n):
        for oy in range(n):
            x0, y0 = ox * st / n, oy * st / n            # left/top edge of the first cell's box
            boxes = [(x0, y0, x0 + 8, y0 + 8), (x0 + 6, y0, x0 + 14, y0 + 8)]
            cs = [[(i, j) for i in range(-1, 3) for j in range(-1, 3)
                   if b[0] < (i + .5) * st < b[2] and b[1] < (j + .5) * st < b[3]] for b in boxes]
            # each grid cell answers for one object only: distinct cells needed
            if cs[0] and cs[1] and (len(set(cs[0]) | set(cs[1])) >= 2): both += 1
            elif cs[0] or cs[1]: one += 1
            else: none += 1
    t = n * n
    print("stride", st, ": both reportable %.0f %%, one %.0f %%, none %.0f %%" % (100 * both / t, 100 * one / t, 100 * none / t))
for st in (8, 32): report(st)

sect("B7: distribution loss, target 2.3")
best = min((-(0.7 * math.log(p) + 0.3 * math.log(1 - p)), p) for p in [k / 1000 for k in range(1, 1000)])
print("minimiser p2 = %.3f, mean %.2f; one-hot rounding error for 2.3: %.1f" % (best[1], 2 * best[1] + 3 * (1 - best[1]), 0.3))

sect("B8: GIoU")
A = (0, 0, 4, 4)
for x in (8, 7, 6, 5, 4, 3):
    print("B at x =", x, "IoU %.3f  GIoU %.3f" % (biou(A, (x, 0, x + 4, 4)), giou(A, (x, 0, x + 4, 4))))

sect("B10: counterexample")
C = {"c": ((0, 0, 40, 40), .9, "debris"), "a": ((10, 0, 50, 40), .8, "cell"), "b": ((20, 0, 60, 40), .7, "cell")}
print("IoU c-a %.3f a-b %.3f c-b %.3f" % (biou(C["c"][0], C["a"][0]), biou(C["a"][0], C["b"][0]), biou(C["c"][0], C["b"][0])))

sect("Speed (measured in measured_yolonet.json): images per hour at 0.85 s and 0.22 s")
for t in (0.85, 0.22): print(t, "s ->", round(3600 / t), "images/hour")
