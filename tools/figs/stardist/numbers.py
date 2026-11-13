"""Numbers of the by-hand questions in stardist/ (lecture, exercise, quiz). Run: python3 -I numbers.py
(The StarDist part of plan/cellnets-day.md's numbers.py, plus the extra checks of this unit.)"""
import math


def sect(t): print("\n=== " + t)


def poly_area(r):
    n = len(r); a = 2 * math.pi / n
    return 0.5 * math.sin(a) * sum(r[k] * r[(k + 1) % n] for k in range(n))


def rect_rays(hx, hy, n):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n; c, s = abs(math.cos(t)), abs(math.sin(t))
        out.append(min(hx / c if c > 1e-12 else math.inf, hy / s if s > 1e-12 else math.inf))
    return out


def ell_rays(a, b, n):
    return [a * b / math.sqrt((b * math.cos(2 * math.pi * k / n)) ** 2 + (a * math.sin(2 * math.pi * k / n)) ** 2) for k in range(n)]


sect("B1 / slide: 8 x 4 rectangle from its centre")
r = rect_rays(4, 2, 8)
print("rays", [round(x, 3) for x in r])
A = poly_area(r); print("8 rays: polygon area %.3f of 32, IoU %.3f" % (A, A / 32))
A32 = poly_area(rect_rays(4, 2, 32)); print("32 rays: area %.3f IoU %.3f" % (A32, A32 / 32))
A16 = poly_area(rect_rays(4, 2, 16)); print("16 rays: area %.3f IoU %.3f" % (A16, A16 / 32))
print("corner triangles lost with 8 rays: 4 x %.1f" % ((32 - A) / 4))

sect("B2: rays 5,4,3,4,5,4,3,4")
r = [5, 4, 3, 4, 5, 4, 3, 4]
print("area %.3f  (sum of products %d, factor %.4f)" % (poly_area(r), sum(r[k] * r[(k + 1) % 8] for k in range(8)), 0.5 * math.sin(math.pi / 4)))
r2 = r[:]; r2[0] = 6
print("ray 0 (right) grows to 6: area %.3f" % poly_area(r2))

sect("B6: disc seen from its centre (n-gon coverage)")
for n in (4, 8, 16, 32, 64):
    print(n, "coverage %.4f" % (n / (2 * math.pi) * math.sin(2 * math.pi / n)))

sect("B8: ellipse a=6, b=3 from its centre")
for n in (8, 16, 32):
    A = poly_area(ell_rays(6, 3, n)); print(n, "rays", [round(x, 3) for x in ell_rays(6, 3, n)] if n == 8 else "", "area %.3f frac %.4f" % (A, A / (math.pi * 18)))
print("true area %.3f" % (math.pi * 18))

sect("Quiz: four equal rays r")
for rr in (1, 3):
    print(rr, poly_area([rr] * 4), "= 2 r^2 =", 2 * rr * rr)

sect("B7: five-pointed star (tips R=26, notches r=11) seen from its centre")
R, rn = 26, 11
star = [R if i % 2 == 0 else rn for i in range(10)]
print("10 rays (tips + notches): exact star area %.1f" % poly_area(star))
print("5 rays on the tips: pentagon area %.1f -> sticks out by %.1f" % (poly_area([R] * 5), poly_area([R] * 5) - poly_area(star)))

sect("B4: polygon NMS by hand (IoU table given)")
probs = {"P1": 0.92, "P2": 0.85, "P3": 0.80, "P4": 0.62, "P5": 0.55, "P6": 0.35}
iou = {("P1", "P2"): 0.71, ("P1", "P3"): 0.18, ("P1", "P4"): 0.05, ("P1", "P5"): 0.33,
       ("P2", "P3"): 0.22, ("P2", "P5"): 0.41, ("P3", "P4"): 0.46, ("P3", "P6"): 0.12, ("P4", "P6"): 0.58}


def I(a, b): return iou.get((a, b), iou.get((b, a), 0))


def greedy(names, scores, thr, minscore):
    order = sorted([n for n in names if scores[n] >= minscore], key=lambda n: -scores[n])
    kept, log = [], []
    while order:
        b = order.pop(0); kept.append(b)
        rem = [o for o in order if I(b, o) > thr]
        log.append((b, rem)); order = [o for o in order if o not in rem]
    return kept, log


for ms, thr in ((0.5, 0.3), (0.5, 0.5), (0.3, 0.7)):
    print("prob >= %.1f, NMS %.1f:" % (ms, thr), greedy(list(probs), probs, thr, ms))

sect("Cost of NMS: pairs for m candidates")
for m in (239, 4737, 9661):
    print(m, "candidates: all pairs %d" % (m * (m - 1) // 2))
