"""Numbers of the cellnets-compare/ exercise and lecture (by-hand questions and the results table).

Run from the repository root after the measurement script (measure.ts):
  python3 -I tools/figs/cellnets-compare/numbers.py

B1, B3: from plan/cellnets-day.md (compare part of its numbers.py, copied here).
B2: a 4 x 5 IoU table from the classical result on a crop of a real image (labels written by
measure.ts to .cache/cellnets-compare/). B6: bootstrap over the seven images of results.json.
Results table of the lecture: printed as Markdown from results.json.
"""
import json
import os
import numpy as np
from scipy import ndimage as ndi

R = json.load(open(os.environ.get("CC_RESULTS", "tools/figs/cellnets-compare/results.json")))
CACHE = os.environ.get("CC_OUT", ".cache/cellnets-compare").rstrip("/") + "/"
IDS = ["sparse", "dim", "dense", "u2os", "u2os2", "histology", "brightfield"]
NAMES = {"sparse": "sparse", "dim": "dim", "dense": "dense", "u2os": "U2OS 1", "u2os2": "U2OS 2",
         "histology": "histology", "brightfield": "bright field"}
M = [("classical", "classical"), ("tuned", "classical, tuned on the image"), ("stardist", "StarDist"),
     ("cellpose", "Cellpose"), ("sam", "SAM automatic")]


def sect(t):
    print("\n=== " + t)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


sect("B1: worked F1 (20 nuclei)")
for lab, tp, np_, ng in [("A .5", 18, 21, 20), ("A .7", 12, 21, 20), ("B .5", 16, 17, 20), ("B .7", 15, 17, 20)]:
    print(lab, "TP", tp, "FP", np_ - tp, "FN", ng - tp, "F1 %.3f" % f1(tp, np_ - tp, ng - tp))

sect("B3: cost (seconds per 512 x 512 image, demos/wasm READMEs)")
for name, t in [("StarDist", 2.13), ("Cellpose (4 workers)", 7.7), ("SAM automatic 8x8", 48.3)]:
    print(f"{name}: {3600 / t:.0f} images/hour; 10 000 images {10000 * t / 3600:.1f} h")
print("StarDist params ~ %.2f M (5.5 MB f32)" % (5.5e6 / 4 / 1e6))
print("SAM 2.1 tiny params ~ %.1f M (77 980 836 B f16)" % (77980836 / 2 / 1e6))

sect("Results table (F1 at 0.5 / 0.7, splits/merges)")
print("| method | " + " | ".join(NAMES[i] for i in IDS) + " |")
for m, name in M:
    cells = []
    for i in IDS:
        r = R["runs"][i]["methods"][m]
        cells.append(f"{r['f1'][0]:.2f} / {r['f1'][4]:.2f}")
    print(f"| {name} | " + " | ".join(cells) + " |")
for i in IDS + ["histology-inv", "brightfield-inv"]:
    run = R["runs"][i]
    print(i, "truth", run["nTrue"], " ".join(f"{m}: n={r['nPred']} F1 {r['f1'][0]:.3f}/{r['f1'][4]:.3f} TP/FP/FN {r['tp5']}/{r['fp5']}/{r['fn5']} "
                                         f"split {r['splits']} merge {r['merges']} {r['ms']} ms" for m, r in run["methods"].items()))
    if "tuned" in run["methods"]:
        print("   tuned params", run["methods"]["tuned"].get("params"))

sect("Mean F1 at 0.5 over the 5 fluorescence images and over all 7")
for m, name in M:
    a = [R["runs"][i]["methods"][m]["f1"][0] for i in IDS]
    print(f"{name}: fluorescence {np.mean(a[:5]):.3f}, all {np.mean(a):.3f}")

sect("B6: bootstrap over images of the mean F1 difference (10 000 resamples, seed 1)")
for a, b in [("stardist", "classical"), ("cellpose", "classical"), ("stardist", "cellpose")]:
    d = np.array([R["runs"][i]["methods"][a]["f1"][0] - R["runs"][i]["methods"][b]["f1"][0] for i in IDS])
    rng = np.random.default_rng(1)
    boot = np.array([d[rng.integers(0, 7, 7)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{a} - {b}: per image {np.round(d, 2).tolist()}, mean {d.mean():.3f}, 95 % interval [{lo:.3f}, {hi:.3f}], "
          f"fraction of resamples <= 0: {(boot <= 0).mean():.3f}")


sect("B6 (c): fluorescence images only, and without the dense image")
d = np.array([R["runs"][i]["methods"]["stardist"]["f1"][0] - R["runs"][i]["methods"]["classical"]["f1"][0] for i in IDS])
for lab_, sel in [("5 fluorescence", [0, 1, 2, 3, 4]), ("6 without dense", [0, 1, 3, 4, 5, 6])]:
    x = d[sel]
    rng = np.random.default_rng(1)
    boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{lab_}: mean {x.mean():.3f}, 95 % interval [{lo:.3f}, {hi:.3f}], fraction <= 0: {(boot <= 0).mean():.3f}")
print("per-image F1 (3 decimals): StarDist", [round(R["runs"][i]["methods"]["stardist"]["f1"][0], 3) for i in IDS],
      "classical", [round(R["runs"][i]["methods"]["classical"]["f1"][0], 3) for i in IDS])

sect("B2: IoU table of a crop of the classical result")
def lab(key, m):
    run = R["runs"][key]
    return np.fromfile(f"{CACHE}{key}-{m}.bin", dtype=np.int32).reshape(run["h"], run["w"])


def table(t, p):
    ts = [v for v in np.unique(t) if v]
    ps = [v for v in np.unique(p) if v]
    T = np.zeros((len(ts), len(ps)))
    for a, i in enumerate(ts):
        for b, j in enumerate(ps):
            inter = np.sum((t == i) & (p == j))
            T[a, b] = inter / (np.sum(t == i) + np.sum(p == j) - inter)
    return ts, ps, T


# A closed group of the classical result on the inverted histology image: 4 annotated nuclei and
# the predictions overlapping them (and only them), with one merge; plus the nearest prediction
# that overlaps no nucleus at all (an extra object), so the table has 4 x 5 entries.
best = None
for key in ("dense", "histology-inv"):
    t_all, p_all = lab(key, "truth"), lab(key, "classical")
    nt, np_ = t_all.max(), p_all.max()
    inter = np.zeros((nt + 1, np_ + 1))
    idx = (t_all > 0) & (p_all > 0)
    np.add.at(inter, (t_all[idx], p_all[idx]), 1)
    at = np.bincount(t_all.ravel(), minlength=nt + 1)
    ap = np.bincount(p_all.ravel(), minlength=np_ + 1)
    for j in range(1, np_ + 1):
        if (inter[1:, j] >= 0.5 * at[1:]).sum() != 2:
            continue
        T, P = set(np.nonzero(inter[1:, j])[0] + 1), {j}
        for _ in range(6):
            P |= set(np.nonzero(inter[sorted(T)][:, 1:].sum(0))[0] + 1)
            T |= set(np.nonzero(inter[1:, sorted(P)].sum(1))[0] + 1)
        if len(T) > 4 or len(P) > 5:
            continue
        # add the nearest whole true objects matched one-to-one (closed pairs) until 4 true objects,
        # then the nearest pure-background predictions until 5 predictions
        cy, cx = np.argwhere(np.isin(t_all, sorted(T))).mean(0)
        cen_t = ndi.center_of_mass(np.ones_like(t_all), t_all, range(nt + 1))
        cen_p = ndi.center_of_mass(np.ones_like(p_all), p_all, range(np_ + 1))
        pairs = []
        for i in range(1, nt + 1):
            if i in T or at[i] == 0:
                continue
            ov = np.nonzero(inter[i, 1:])[0] + 1
            if len(ov) == 1 and np.count_nonzero(inter[1:, ov[0]]) == 1 and len(P) < 5:
                pairs.append((np.hypot(cen_t[i][0] - cy, cen_t[i][1] - cx), i, ov[0]))
        pairs.sort()
        T, P = set(T), set(P)
        for d, i, q in pairs:
            if len(T) >= 4 or len(P) >= 5:
                break
            T.add(i); P.add(q)
        pure = sorted((np.hypot(cen_p[q][0] - cy, cen_p[q][1] - cx), q) for q in range(1, np_ + 1) if ap[q] and inter[1:, q].sum() == 0)
        for d, q in pure:
            if len(P) >= 5:
                break
            P.add(q)
        if len(T) != 4 or len(P) != 5:
            continue
        Ts, Ps = sorted(T), sorted(P)
        Tm = np.array([[inter[i, q] / (at[i] + ap[q] - inter[i, q]) for q in Ps] for i in Ts])
        nm = int((Tm >= 0.5).sum())
        spread = np.ptp(np.argwhere(np.isin(t_all, Ts) | np.isin(p_all, Ps)), axis=0).max()
        score = (nm == 2) * 10 - spread / 100
        if best is None or score > best[0]:
            best = (score, key, Ts, Ps, j, t_all, p_all, inter, at)
score, key, ts, ps, jm, t_all, p_all, inter, at = best
# order: by column of the centroid (left to right), truths and predictions separately
ts = sorted(ts, key=lambda v: np.argwhere(t_all == v)[:, 1].mean())
ps = sorted(ps, key=lambda v: np.argwhere(p_all == v)[:, 1].mean())
T = np.array([[inter[i, j] / (at[i] + (p_all == j).sum() - inter[i, j]) for j in ps] for i in ts])
print("image", key, "true labels", ts, "predicted", ps, "merge = predicted column", ps.index(jm) + 1)
print("areas true", [int(at[i]) for i in ts], "predicted", [int((p_all == j).sum()) for j in ps])
print("fraction of each true object inside each prediction:")
for a, i in enumerate(ts):
    print("  T%d " % (a + 1) + " ".join(f"{inter[i, j] / at[i]:.2f}" for j in ps))
print("IoU (rows T1..T4, columns P1..P5):")
for a in range(T.shape[0]):
    print("  T%d " % (a + 1) + " ".join(f"{v:.2f}" for v in T[a]))
m = T >= 0.5
tp = int(m.sum())
print(f"matches at 0.5: {[(a + 1, b + 1) for a, b in zip(*np.nonzero(m))]}; TP {tp}, FP {T.shape[1] - tp}, FN {T.shape[0] - tp}, F1 {f1(tp, T.shape[1] - tp, T.shape[0] - tp):.3f}")
