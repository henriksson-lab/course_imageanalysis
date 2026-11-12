# Plan: a day on graph cut algorithms

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. write binary segmentation as **minimising an energy** with a data term and a smoothness term, and explain what the weight $\lambda$ trades off;
2. compute a **minimum s–t cut** of a small graph by hand, using augmenting paths, and state the **max-flow / min-cut** theorem;
3. **build the graph** for a small image (t-links from the data term, n-links from the smoothness term) and check that the cost of a cut equals the energy of the labelling;
4. use **seeds** for interactive segmentation, and recognise the typical failure modes (shrinking bias, blocky boundaries from a 4-neighbourhood, a bad intensity model);
5. relate cuts to linear algebra: the **graph Laplacian**, and **normalized cuts** via its second eigenvector.

Why it fits this audience: it needs nothing beyond graphs, sums and eigenvectors, and
(5) is a direct payoff of their linear algebra. Edge weights reuse image gradients from
*Filtering & convolution*.

## Prerequisite gap

- **Thresholding** (histogram, global threshold, Otsu / minimum-error, why noise breaks
  it) is taught on the thresholding day (`plan/thresholding-day.md`). Unit 1 recalls it
  in one line.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `seg-energy/`: Segmentation as energy minimisation

- Recall (one line; taught on the thresholding day, `plan/thresholding-day.md`): thresholding a noisy image gives isolated wrong pixels. Pixels should agree with their neighbours.
- Labelling $L_p \in \{0, 1\}$. Energy
  $E(L) = \sum_p D_p(L_p) + \lambda \sum_{(p,q)} w_{pq}\,[L_p \ne L_q]$.
- Data term from an intensity model: $D_p(\ell) = -\log P(I_p \mid \ell)$ (two Gaussians: with $\lambda = 0$ this is the minimum-error threshold of the thresholding day, unit 2 `otsu/`; back-reference, not re-derived).
- Smoothness term: counts (weighted) boundary length. $\lambda = 0$ is thresholding; large $\lambda$ gives one blob or nothing.
- $2^n$ labellings: brute force is hopeless for an image. On a **1D chain** dynamic programming solves it exactly (the demo uses this); on a 2D grid we need something else → graph cuts.
- **Demo `segchain`** (new): a noisy 1D intensity profile (e.g. a line across a cell), two intensity levels to set, a $\lambda$ slider. Shows the thresholded labels, the optimal labels (exact, by dynamic programming), and the energy split into data + $\lambda\cdot$boundaries. Presets: "λ = 0 is thresholding", "too smooth", "a thin real structure disappears".
- By hand: energies of 3–4 given labellings of a 1×5 image; which is optimal for $\lambda = 1$ and $\lambda = 5$?

### 2. `maxflow/`: Graphs, cuts and maximum flow

- Directed graph with capacities, source $s$, sink $t$. An **s–t cut** splits the nodes into $S \ni s$ and $T \ni t$; its cost is the sum of capacities from $S$ to $T$.
- Flow: capacity and conservation constraints. **Max-flow = min-cut** (state; sketch why "≤" is obvious and the residual graph gives "=").
- Ford–Fulkerson with BFS (Edmonds–Karp): augment along shortest paths in the residual graph until $t$ is unreachable; the nodes still reachable from $s$ form the minimum cut.
- Mention: Boykov–Kolmogorov (what image libraries use) and push–relabel; complexity only in words.
- **Demo `maxflow`** (new): a graph of ~8 nodes drawn in SVG, capacities editable (click an edge, ±). Buttons: *next augmenting path* (path highlighted, bottleneck shown), *run to end*, *reset*. Edges show flow/capacity; toggle *residual graph*. At the end: the reachable set shaded and the cut edges marked, with "flow = cut = …". Presets: a textbook example, a graph where a greedy path choice needs a back-edge (shows why residual reverse edges matter), a graph with two equal minimum cuts.
- By hand: max flow / min cut of a 5–6 node graph; find the cut from the final residual graph.

### 3. `graphcut-seg/`: Graph-cut segmentation

- The construction (Boykov & Jolly 2001): one node per pixel + $s$ (object) + $t$ (background).
  **t-links** carry the data term, **n-links** carry $\lambda w_{pq}$ with
  $w_{pq} = \exp\!\big(-(I_p - I_q)^2 / 2\sigma^2\big)$: cutting across an edge in the image is cheap.
- Check by hand that cut cost = $E(L)$ for a 1×3 image. Binary energies with this form (submodular) are solved **exactly**.
- **Seeds**: hard constraints via infinite t-links; the intensity model can be estimated from the seeds.
- Failure modes: shrinking bias (short boundaries are cheap: thin structures get cut off), metrication (4- vs 8-neighbourhood gives blocky boundaries), overlapping intensity distributions.
- Pointer, not taught in depth: GrabCut (iterate: fit colour models, cut again).
- **Demo `pixelgraph`** (new, small): a 6×6 image drawn as its graph: pixels as nodes, n-link thickness = weight, t-links as coloured bars. Sliders $\lambda$, $\sigma$. Shows the minimum cut and the energy terms. Edit pixel values by clicking. This is the bridge between the by-hand construction and the real image.
- **Demo `graphcut`** (new, main demo): a 128×128 image (the micrograph crop already used in Part 1, plus a synthetic test image with noise and a thin protrusion). Paint object/background seeds; sliders $\lambda$, $\sigma$, noise; data term from seed histograms or from two fixed levels; 4- or 8-neighbourhood. Live result overlay (max-flow at this size runs in tens of ms). Side panel: thresholded result for comparison, energy split. Presets reproduce each failure mode above.
- By hand: build the graph for a 1×3 / 2×2 image, list all cuts, find the minimum, compare with the labelling energies.

### 4. `ncut/`: Spectral graph partitioning and normalized cuts

- Minimum cut without seeds prefers to cut off **one** node: it needs balancing.
- Graph Laplacian $L = D - W$; $\mathbf{x}^\top L\mathbf{x} = \tfrac12\sum_{pq} w_{pq}(x_p - x_q)^2$; for $\mathbf{x} \in \{\pm1\}^n$ this is $4\times$ the cut.
- Relax to real $\mathbf{x}$: the **Fiedler vector** (second-smallest eigenvector). Normalized cut (Shi & Malik 2000) → generalized eigenproblem $L\mathbf{x} = \mu D\mathbf{x}$. Threshold the eigenvector to get a partition.
- Contrast: graph cuts = exact on a restricted energy with seeds; spectral = approximate, no seeds, global.
- **Demo `ncut`** (new): (a) a point cloud of two or three blobs (draggable points), graph by Gaussian affinity with a $\sigma$ slider; (b) a small (32×32) image. Shows the Fiedler vector as colour, a threshold slider, the resulting cut, and min-cut vs normalized-cut side by side. Eigenvector by Lanczos / inverse power iteration in TS (n ≤ 1024).
- By hand: Laplacian of a path of 4 nodes, verify an eigenvector, and the Laplacian quadratic form for a ±1 vector.

### 5. (optional / further reading) `alpha-expansion/`: more than two labels

- Multi-label Potts energy is NP-hard; α-expansion solves a binary graph cut per label move and is within a factor 2 of optimal.
- **Demo**: reuse `graphcut` with `data-labels="3"` (seeds in three colours), showing each expansion move in turn.

## Shared engineering

- `demos/src/lib/maxflow.ts`: one max-flow implementation (BK or push–relabel) for images, plus an instrumented Edmonds–Karp that records each augmenting path for the step-through demo.
- `demos/src/lib/graph.ts`: grid-graph construction (4/8-neighbourhood, edge weights from intensities), shared by `pixelgraph`, `graphcut`, `ncut`.
- Test (new; the framework has none yet): a small node script comparing max-flow against brute-force enumeration of all cuts on random graphs of ≤ 12 nodes, run by `make test`.
- Image: reuse `demos/src/fourier2d/cell256.png`, downsampled to 128×128.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560 on a slide.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`segchain`) |
| 10:45–12:15 | Unit 2 lecture + by-hand max-flow, then `maxflow` demo |
| 13:15–14:45 | Unit 3 lecture, build-the-graph by hand, `pixelgraph`, `graphcut` |
| 15:00–16:30 | Unit 4 lecture + exercise (`ncut`); quizzes as wrap-up |

## Open questions

- How long is a "day" in this course (hours, and lecture vs exercise time)? Four units may be too much for one day; unit 4 could move to the next day or to further reading.
- Normalized cuts: core or optional? It is the strongest link to linear algebra, so my recommendation is core.
- Should the day include GrabCut / colour images, or stay greyscale?
- Programming language for the code parts is still open (see README).

## References

- Boykov & Jolly (2001), *Interactive graph cuts for optimal boundary & region segmentation*, ICCV.
- Boykov & Kolmogorov (2004), *An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision*, PAMI.
- Kolmogorov & Zabih (2004), *What energy functions can be minimized via graph cuts?*, PAMI.
- Boykov, Veksler & Zabih (2001), *Fast approximate energy minimization via graph cuts*, PAMI (α-expansion).
- Shi & Malik (2000), *Normalized cuts and image segmentation*, PAMI.
- Rother, Kolmogorov & Blake (2004), *GrabCut*, SIGGRAPH.
