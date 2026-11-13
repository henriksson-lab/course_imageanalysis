"""Figures for topics/segmentation.qmd (the segmentation topic page).

Everything is synthetic (fixed seeds); every number quoted on the page is printed here and
written to the JSON file given as the first argument (default: no JSON). Run from anywhere:
   python3 -I tools/figs/topics-segmentation/figs.py [numbers.json]
PNG: figure size in inches = display size / 100 px, dpi 200, fonts >= 12 pt.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage as ndi
from skimage import filters, segmentation, measure, morphology, feature
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "topics")
plt.rcParams.update({"font.size": 13, "svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, SKY, PURPLE, YELLOW, GREY = "#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#ffd400", "#777777"
NUM = {}


def save(fig, name, **kw):
    if name.endswith(".png"):
        kw.setdefault("dpi", 200)
    fig.savefig(os.path.join(OUT, name), facecolor="white", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig); print("wrote", name)


def show(ax, img, vmin=0, vmax=1, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


PAL = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178],
                [213, 94, 0], [204, 121, 167], [255, 255, 255], [150, 220, 120], [255, 150, 150]]) / 255


def outlines(ax, img, lab, vmax=1):
    """Grey image with each labelled object outlined in its own colour."""
    rgb = np.dstack([np.clip(img / vmax, 0, 1)] * 3)
    b = segmentation.find_boundaries(lab, mode="inner")
    ids = lab[b]
    rgb[b] = PAL[(ids - 1) % len(PAL)]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ the scene
N = 160
yy, xx = np.mgrid[0:N, 0:N].astype(float)
# (cy, cx, a, b, angle_deg, brightness): two touching pairs, one dim nucleus, one elongated
NUCLEI = [(30, 30, 13, 10, 20, 1.0), (32, 52, 12, 10, -10, 0.95),          # touching pair
          (90, 28, 14, 11, 70, 0.9), (130, 40, 11, 11, 0, 1.0),
          (40, 105, 13, 9, 35, 0.9), (75, 75, 11, 11, 0, 0.45),            # dim one
          (110, 100, 13, 10, -30, 1.0), (122, 121, 12, 10, 40, 0.95),      # touching pair
          (70, 135, 16, 7, 80, 0.9), (25, 140, 10, 10, 0, 1.0)]


def ellipse(cy, cx, a, b, ang):
    t = np.deg2rad(ang)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


GT = np.zeros((N, N), int)
OBJ = np.zeros((N, N))
for i, (cy, cx, a, b, ang, br) in enumerate(NUCLEI):
    m = ellipse(cy, cx, a, b, ang) & (GT == 0)
    GT[m] = i + 1
    OBJ[m] = br
rng = np.random.default_rng(7)
BG = 0.10 + 0.40 * xx / N                     # illumination ramp left -> right
IMG = ndi.gaussian_filter(BG + 0.45 * OBJ, 1.2) + rng.normal(0, 0.07, (N, N))
SEM = GT > 0
NUM["scene"] = {"size": N, "objects": len(NUCLEI), "noise_sd": 0.07, "ramp": [0.10, 0.50],
                "contrast_bright": 0.45, "contrast_dim": round(0.45 * 0.45, 3)}


# ------------------------------------------------------------------ scores
def dice(a, b):
    return 2 * np.sum(a & b) / (a.sum() + b.sum())


def object_f1(lab, gt, tau=0.5):
    """One-to-one matching at IoU > tau (above 0.5 the matching is unique)."""
    tp = 0
    for g in range(1, gt.max() + 1):
        gm = gt == g
        ids, cnt = np.unique(lab[gm], return_counts=True)
        best = 0
        for i, c in zip(ids, cnt):
            if i == 0:
                continue
            pm = lab == i
            iou = c / np.sum(gm | pm)
            best = max(best, iou)
        tp += best > tau
    npred = len(np.unique(lab)) - (1 if 0 in lab else 0)
    ngt = gt.max()
    f1 = 2 * tp / (npred + ngt) if npred + ngt else 0
    return tp, npred, ngt, f1


def score(name, lab):
    lab = measure.label(lab > 0) if lab.max() <= 1 else lab
    d = dice(lab > 0, SEM)
    tp, npred, ngt, f1 = object_f1(lab, GT)
    NUM.setdefault("gallery", {})[name] = {"dice": round(float(d), 3), "objects": int(npred),
                                           "matched": int(tp), "f1": round(float(f1), 3)}
    print(f"{name:28s} Dice {d:.3f}  objects {npred:3d}  matched {tp}/{ngt}  F1 {f1:.3f}")
    return lab, d, tp, npred, f1


def remove_small(mask, n=15):
    return morphology.remove_small_objects(mask, n)


def graphcut(img, mu0, mu1, sd, lam, beta=None):
    """Two labels by one minimum cut (scipy max-flow, capacities scaled to integers).
    Data = squared distance to the class mean (Gaussian, equal sd); smoothness = lam per pair of
    4-neighbours with different labels (times exp(-beta*dI^2) if beta is given)."""
    h, w = img.shape
    n = h * w
    S, T = n, n + 1
    idx = np.arange(n).reshape(h, w)
    D0 = ((img - mu0) ** 2 / (2 * sd ** 2)).ravel()   # cost of "background"
    D1 = ((img - mu1) ** 2 / (2 * sd ** 2)).ravel()   # cost of "object"
    rows, cols, caps = [], [], []
    # source side = object: cutting S->p means p is background (pay D0); p->T cut: p object (pay D1)
    rows += [np.full(n, S), np.arange(n)]; cols += [np.arange(n), np.full(n, T)]; caps += [D0, D1]
    for a, b, d in ((idx[:, :-1], idx[:, 1:], img[:, 1:] - img[:, :-1]),
                    (idx[:-1, :], idx[1:, :], img[1:, :] - img[:-1, :])):
        wgt = lam * (np.exp(-beta * d ** 2) if beta is not None else np.ones_like(d))
        rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; caps += [wgt.ravel(), wgt.ravel()]
    r = np.concatenate(rows); c = np.concatenate(cols)
    cap = np.round(np.concatenate(caps) * 1000).astype(np.int32)
    G = csr_matrix((cap, (r, c)), shape=(n + 2, n + 2))
    G.sum_duplicates()
    res = maximum_flow(G, S, T, method="dinic")
    resid = G - res.flow
    resid.data = np.where(resid.data > 0, 1, 0).astype(np.int32)
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, S, directed=True, return_predecessors=False)
    seg = np.zeros(n + 2, bool); seg[reach] = True
    return seg[:n].reshape(h, w)


def dist_watershed(mask, min_distance=9):
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(mask), 1.5)
    pk = feature.peak_local_max(dist, min_distance=min_distance, labels=measure.label(mask),
                                exclude_border=False)
    mk = np.zeros(mask.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    return segmentation.watershed(-dist, mk, mask=mask)


def feature_stack(img):
    fs = [img]
    for s in (1, 2, 4):
        fs += [ndi.gaussian_filter(img, s), ndi.gaussian_gradient_magnitude(img, s),
               ndi.gaussian_laplace(img, s)]
    fs += [ndi.gaussian_filter(img, 8)]
    return np.stack(fs, -1)


def paint_strokes(objects):
    """Simulated painting: small disks inside the listed objects, short lines of background."""
    lab = np.zeros((N, N), int)
    for i in objects:
        cy, cx = NUCLEI[i][:2]
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = 2
    for (y0, x0, y1, x1) in [(5, 5, 5, 60), (150, 10, 150, 150), (60, 50, 100, 55), (5, 75, 50, 80),
                             (90, 150, 150, 150), (55, 90, 55, 120)]:
        n = 80
        for t in np.linspace(0, 1, n):
            y, x = int(round(y0 + t * (y1 - y0))), int(round(x0 + t * (x1 - x0)))
            if not SEM[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
                lab[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 1
    return lab


def forest(img, strokes, seed=0):
    F = feature_stack(img)
    m = strokes > 0
    clf = RandomForestClassifier(n_estimators=50, random_state=seed, min_samples_leaf=2)
    clf.fit(F[m], strokes[m])
    return clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(img.shape)


# ------------------------------------------------------------------ figure 1: the question
def fig_question():
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.95))
    show(axs[0], IMG, -0.05, 1.0)
    axs[0].set_title("the image")
    show(axs[1], SEM, 0, 1)
    axs[1].set_title("which pixels? (semantic)")
    rgb = np.ones((N, N, 3)) * 0.0
    for i in range(1, GT.max() + 1):
        rgb[GT == i] = PAL[(i - 1) % len(PAL)]
    axs[2].imshow(rgb, interpolation="nearest"); axs[2].set_xticks([]); axs[2].set_yticks([])
    for s in axs[2].spines.values():
        s.set_visible(False)
    axs[2].set_title("which object? (instances)")
    fig.tight_layout(pad=0.3); save(fig, "fig-seg-question.png")


# ------------------------------------------------------------------ figure 2: one image, eight methods
def fig_gallery():
    res = []
    t = filters.threshold_otsu(IMG)
    NUM["otsu_t"] = round(float(t), 3)
    m_otsu = remove_small(IMG > t)
    res.append(("global threshold (Otsu)", score("global threshold (Otsu)", m_otsu)[0]))

    loc = filters.threshold_local(IMG, block_size=41, method="gaussian", offset=-0.06)
    m_loc = remove_small(ndi.binary_fill_holes(ndi.binary_opening(IMG > loc, iterations=1)), 30)
    res.append(("local threshold", score("local threshold", m_loc)[0]))

    ws = dist_watershed(m_loc)
    res.append(("local threshold + watershed", score("local threshold + watershed", ws)[0]))

    # k-means / GMM idea on grey levels = Otsu-like; skip. Chan-Vese (region-based level set):
    cv = segmentation.chan_vese(IMG, mu=0.15, lambda1=1, lambda2=1, max_num_iter=300, init_level_set="checkerboard")
    if cv[SEM].mean() < 0.5:
        cv = ~cv
    res.append(("Chan–Vese (regions)", score("Chan–Vese (regions)", remove_small(cv))[0]))

    # graph cut with the Otsu class means: removes specks, keeps the ramp problem
    mu0, mu1 = IMG[IMG <= t].mean(), IMG[IMG > t].mean()
    sd = np.sqrt(0.5 * (IMG[IMG <= t].var() + IMG[IMG > t].var()))
    gc = graphcut(IMG, mu0, mu1, sd, lam=2.0)
    res.append(("graph cut (intensity model)", score("graph cut (intensity model)", gc)[0]))

    # seeded random walker: one seed per nucleus (a click) + background strokes -> instances
    seeds = np.zeros((N, N), int)
    for i, (cy, cx, *_r) in enumerate(NUCLEI):
        seeds[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4] = i + 2
    bgs = paint_strokes([]) == 1
    seeds[bgs] = 1
    rw = segmentation.random_walker(ndi.gaussian_filter(IMG, 1), seeds, beta=2000, mode="cg_j")
    rw = np.where(rw == 1, 0, rw - 1)
    res.append(("random walker (clicks)", score("random walker (clicks)", rw)[0]))

    # pixel classifier: a forest trained on painted strokes in 3 nuclei
    strokes = paint_strokes([0, 4, 7])
    p = forest(IMG, strokes)
    m_rf = remove_small(p > 0.5)
    res.append(("forest (painted pixels)", score("forest (painted pixels)", m_rf)[0]))
    NUM["forest_strokes"] = {"object_pixels": int((strokes == 2).sum()), "background_pixels": int((strokes == 1).sum())}

    fig, axs = plt.subplots(2, 4, figsize=(12.4, 7.0))
    axs = axs.ravel()
    show(axs[0], IMG, -0.05, 1.0)
    for (cy, cx, *_r) in NUCLEI:
        pass
    axs[0].set_title("image", fontsize=13)
    axs[0].text(2, N - 4, "light: darker left,\nbrighter right", color="white", fontsize=11, va="bottom")
    for ax, (name, lab) in zip(axs[1:], res):
        outlines(ax, IMG, lab, vmax=1.0)
        g = NUM["gallery"][name]
        ax.set_title(f"{name}\nDice {g['dice']:.2f} · {g['matched']} of 10 found · {g['objects']} objects",
                     fontsize=11.5)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.3)
    save(fig, "fig-seg-gallery.png")


# ------------------------------------------------------------------ figure 3: touching objects
def fig_touching():
    n = 70
    y, x = np.mgrid[0:n, 0:n].astype(float)
    m = ((y - 35) ** 2 + (x - 23) ** 2 <= 14 ** 2) | ((y - 35) ** 2 + (x - 47) ** 2 <= 13 ** 2)
    rg = np.random.default_rng(3)
    img = ndi.gaussian_filter(0.15 + 0.6 * m, 1.0) + rg.normal(0, 0.04, (n, n))
    mask = ndi.binary_opening(img > filters.threshold_otsu(img), iterations=1)
    nlab = measure.label(mask).max()
    dist = ndi.distance_transform_edt(mask)
    ws = dist_watershed(mask, min_distance=8)
    NUM["touching"] = {"labelled_objects": int(nlab), "watershed_objects": int(ws.max()),
                       "max_distance": round(float(dist.max()), 1)}
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.85))
    show(axs[0], img, 0, 1); axs[0].set_title("image")
    outlines(axs[1], mask * 0.85, measure.label(mask)); axs[1].set_title(f"threshold + label:\n{nlab} object")
    show(axs[2], dist, 0, dist.max(), cmap="magma"); axs[2].set_title("distance to the\nbackground")
    outlines(axs[3], mask * 0.85, ws); axs[3].set_title(f"watershed of the\ndistance: {ws.max()} objects")
    fig.tight_layout(pad=0.3); save(fig, "fig-seg-touching.png")


# ------------------------------------------------------------------ figure 4: one failure per family
def fig_failures():
    rg = np.random.default_rng(11)
    n = 90
    y, x = np.mgrid[0:n, 0:n].astype(float)
    fig, axs = plt.subplots(1, 4, figsize=(11.6, 3.45))

    # (a) watershed of the gradient without markers: one region per minimum
    disc = (y - 45) ** 2 + (x - 45) ** 2 <= 25 ** 2
    img = ndi.gaussian_filter(0.2 + 0.5 * disc, 1.0) + rg.normal(0, 0.05, (n, n))
    grad = filters.sobel(ndi.gaussian_filter(img, 1.0))
    ws = segmentation.watershed(grad)
    NUM.setdefault("failures", {})["watershed_regions"] = int(ws.max())
    outlines(axs[0], img, ws)
    axs[0].set_title(f"watershed, no markers:\n{ws.max()} regions for 2", fontsize=12)

    # (b) graph cut: thin arm lost when λ is large (shrinking bias)
    obj = ((y - 50) ** 2 + (x - 35) ** 2 <= 16 ** 2) | ((np.abs(y - 50) <= 1) & (x >= 35) & (x <= 85))
    img = 0.2 + 0.5 * obj + rg.normal(0, 0.15, (n, n))
    out = {}
    for lam in (0.5, 12.0):
        seg = graphcut(img, 0.2, 0.7, 0.15, lam)
        arm = obj & (x > 55)
        out[lam] = (seg, float(seg[arm].mean()))
    NUM["failures"]["graphcut_arm_kept"] = {str(k): round(v[1], 2) for k, v in out.items()}
    outlines(axs[1], np.clip(img, 0, 1), measure.label(out[12.0][0]))
    axs[1].set_title(f"graph cut, large λ: the thin\narm is lost ({100 * out[12.0][1]:.0f} % kept)", fontsize=12)

    # (c) an edge-based contour (geodesic, with a balloon) leaks through a gap in the edge
    r2 = (y - 45) ** 2 + (x - 40) ** 2
    disc = r2 <= 22 ** 2
    ring = disc & ~(r2 <= 19 ** 2)
    res = {}
    for name, gw in (("gap", 4), ("closed", -1)):
        gap = ring & (x > 50) & (np.abs(y - 45) <= gw)
        wall = ring & ~gap
        img = ndi.gaussian_filter(0.55 - 0.4 * wall + np.random.default_rng(11).normal(0, 0.04, (n, n)), 0.7)
        g = segmentation.inverse_gaussian_gradient(img, alpha=100, sigma=1.5)
        ls = segmentation.morphological_geodesic_active_contour(
            g, 300, init_level_set=(r2 <= 25).astype(np.int8), smoothing=1, balloon=1, threshold=0.6).astype(bool)
        res[name] = (img, ls, int(np.sum(ls & ~disc)))
    NUM["failures"]["geodesic_outside_px"] = {k: v[2] for k, v in res.items()}
    NUM["failures"]["geodesic_image_px"] = n * n
    img, ls, leak = res["gap"]
    rgb = np.dstack([np.clip(img / 0.8, 0, 1)] * 3)
    rgb[segmentation.find_boundaries(ls, mode="inner")] = PAL[0]
    rgb[r2 <= 25] = PAL[2]
    axs[2].imshow(rgb, interpolation="nearest"); axs[2].set_xticks([]); axs[2].set_yticks([])
    for sp in axs[2].spines.values():
        sp.set_visible(False)
    axs[2].set_title(f"edge-based contour leaks\nthrough a gap ({leak} px outside)", fontsize=12)

    # (d) a classifier knows only what it was shown: trained on bright nuclei, misses the dim one
    strokes = paint_strokes([0, 4, 7])
    p = forest(IMG, strokes)
    m = p > 0.5
    dimfrac = float(m[GT == 6].mean())
    NUM["failures"]["forest_dim_found"] = round(dimfrac, 2)
    NUM["failures"]["forest_dim_meanprob"] = round(float(p[GT == 6].mean()), 2)
    outlines(axs[3], IMG, measure.label(remove_small(m)))
    cy, cx = NUCLEI[5][:2]
    axs[3].add_patch(plt.Circle((cx, cy), 16, fill=False, ec=RED, lw=2, ls="--"))
    axs[3].set_title(f"forest trained on bright nuclei:\nthe dim one {100 * dimfrac:.0f} % found", fontsize=12)
    fig.tight_layout(pad=0.3); save(fig, "fig-seg-failures.png")


# ------------------------------------------------------------------ figure 5: a map of the methods
def fig_map():
    fam = {"each pixel alone": BLUE, "regions grown on a graph": GREEN, "energy with neighbours": ORANGE,
           "a moving boundary": PURPLE, "learned from examples": RED}
    # (x: what you supply, y: what one decision looks at)
    pts = [("global threshold,\nOtsu", 0.2, 0.25, "each pixel alone"),
           ("k-means, mixtures", 0.55, 0.62, "each pixel alone"),
           ("local threshold", 0.9, 1.25, "each pixel alone"),
           ("watershed\n(markers)", 2.0, 2.15, "regions grown on a graph"),
           ("FH, superpixels,\nmerge trees", 0.7, 2.0, "regions grown on a graph"),
           ("random walker", 2.05, 2.95, "regions grown on a graph"),
           ("graph cut, MRF", 1.55, 1.5, "energy with neighbours"),
           ("normalized cuts", 0.4, 3.05, "energy with neighbours"),
           ("snakes, geodesic\ncontours", 2.4, 1.85, "a moving boundary"),
           ("Chan–Vese", 2.0, 2.55, "a moving boundary"),
           ("live-wire,\npolar DP", 2.85, 1.3, "a moving boundary"),
           ("active shape\nmodel", 3.2, 3.15, "a moving boundary"),
           ("forest, SVM on\nfeature stacks", 3.35, 1.0, "learned from examples"),
           ("U-Net", 4.35, 2.0, "learned from examples"),
           ("StarDist, Cellpose", 4.3, 3.1, "learned from examples"),
           ("SAM (+ a click)", 4.55, 2.6, "learned from examples")]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for name, x, y, f in pts:
        ax.plot(x, y, "o", ms=11, color=fam[f], mec="black", mew=0.8, zorder=3)
        ax.text(x + 0.09, y, name, va="center", fontsize=12, zorder=4,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xlim(0, 5.6); ax.set_ylim(-0.1, 3.5)
    ax.set_xticks([0.25, 1.25, 2.4, 3.4, 4.5])
    ax.set_xticklabels(["nothing", "a few\nparameters", "seeds or a\nstart curve", "painted pixels,\nshapes", "many labelled\nimages"])
    ax.set_yticks([0.25, 1.2, 2.2, 3.1])
    ax.set_yticklabels(["one pixel", "a neighbourhood", "a region or\na boundary", "the whole object\nor image"])
    ax.set_xlabel("what you must supply →")
    ax.set_ylabel("what one decision looks at →")
    for f, c in fam.items():
        ax.plot([], [], "o", ms=10, color=c, mec="black", mew=0.8, label=f)
    ax.legend(loc="lower right", fontsize=11, frameon=False, ncol=1)
    ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.3); save(fig, "fig-seg-map.svg")


if __name__ == "__main__":
    fig_question()
    fig_gallery()
    fig_touching()
    fig_failures()
    fig_map()
    print(json.dumps(NUM, indent=1))
    if len(sys.argv) > 1:
        json.dump(NUM, open(sys.argv[1], "w"), indent=1)
