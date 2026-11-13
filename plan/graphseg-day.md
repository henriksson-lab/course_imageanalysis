# Plan: a day on graph-based segmentation beyond graph cuts

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.

**Course map (2026-11-13, "Queue, second batch"):** elective, group "Curves and energies";
**entry after core day 13 (graph cuts)**. Hard prerequisites: day 13 (`seg-energy/`,
`graphcut-seg/`, `ncut/`: pixel graphs, contrast weights, the graph Laplacian), day 10
(watershed: `ws-landscape/`, `ws-flooding/`, `ws-gradient/`), day 9 (`label-connectivity/`:
union–find). Soft prerequisites: `gmrf/` (Ising elective: the same Laplacian linear system with a
data term), `kmeans/` (Gaussian mixtures elective), `morph-reconstruct/` (dynamics, h-minima).
Independent of snakes, level sets, dynamic programming and splines. The day **owns**: the random
walker, minimum spanning trees in segmentation (Kruskal, single linkage, Felzenszwalb–Huttenlocher),
watershed as a minimum spanning forest, hierarchical segmentation (merge trees, ultrametric contour
maps, region adjacency graphs), and superpixels (SLIC). See "Proposed reorganisation" for the
ownership moves this needs.

## Goal of the day

After the day a student can:

1. state the **random walker** in three equivalent ways (the probability that a walker started at
   a pixel first reaches an object seed; the voltage in a resistor network with seeds clamped to
   1 and 0; the smoothest function on the graph that matches the seeds) and compute it **by hand**
   on a chain, where it is "voltage drops in proportion to resistance";
2. explain why the random walker gives **probabilities**, never islands without a seed (maximum
   principle), and where it fails (results depend on seed position; far from all seeds the
   probabilities fade towards an even split); compare it with graph cuts on the same seeds;
3. build a **minimum spanning tree** with Kruskal's algorithm (union–find recalled from day 9),
   explain why removing its heaviest edges is **single-linkage** segmentation, and why that
   **chains** through one weak link;
4. run **Felzenszwalb–Huttenlocher** by hand on a 1D signal: merge two components only if the
   edge between them is no heavier than each one's **internal difference** plus $k/|C|$; predict
   what $k$ does (scale, not a minimum size) and why a textured region survives;
5. see seeded **watershed as a minimum spanning forest**: Meyer flooding is Prim's algorithm run from
   all markers at once, and each pixel goes to the seed it reaches with the **lowest highest pass**
   (minimax path); compare seeded segmentation three ways: **sum** (graph cut), **squares**
   (random walker), **maximum** (watershed);
6. build a **hierarchy** of segmentations from a merge order (a dendrogram/merge tree), cut it at
   a level, read an **ultrametric contour map**, and predict how the hierarchy changes when regions
   are merged by **altitude, dynamics or area**, or by a region criterion on a **region adjacency
   graph**;
7. explain **SLIC superpixels** as k-means on (colour, $x$, $y$) with a **local search window**,
   predict what the compactness $m$ and the number $K$ do, measure superpixels (boundary recall,
   under-segmentation error, achievable accuracy), and use them as the **nodes of a smaller graph**;
8. *(math)* derive the random walker system $L_U\mathbf x_U = -B^\top\mathbf x_M$ from the
   Dirichlet energy, prove the maximum principle and that multi-label probabilities sum to 1; the
   cut property and the minimax-path property of an MST; that a seeded MSF assigns each pixel to its
   minimax-nearest seed; the ultrametric inequality of a hierarchy; why SLIC's monotonicity proof
   breaks.

Why it fits this audience: everything is graphs, sorting, averages and one sparse linear system
(the graph Laplacian they met as a quadratic form in `ncut/`). Day 13 left them with one
algorithm (max-flow) and one eigenvector; this day shows the other three classic graph tools
(linear systems, spanning trees, hierarchies) and closes the loop with the watershed of day 10:
graph cuts, random walker and watershed become one family that differs only in how it adds up
edge weights. Kirchhoff's laws and "a harmonic function is the average of its neighbours" are
physics/calculus they know.

## Prerequisite gap

What the built units already teach (checked), and what this day recalls in one line:

- **`ncut/`** (built, day 13): pixel graph with Gaussian affinities, $L = D - W$,
  $\mathbf x^\top L\mathbf x = \tfrac12\sum w_{pq}(x_p - x_q)^2$ (math slide "the graph Laplacian"),
  Fiedler vector, min-cut vs normalized cut table (row "typical use: … superpixels"). Unit 1 starts
  from that quadratic form: "`ncut/` minimised it without seeds and needed an eigenvector; with
  seeds it becomes a linear system". Its one-line superpixel pointer is repointed here.
- **`graphcut-seg/`** (built): t-links, n-links, `contrastWeights` $w_{pq} = \exp(-(I_p - I_q)^2/2\sigma^2)$
  (this day uses the **same** weights and the same σ, so the methods can be compared on one
  graph), seeds as hard constraints, the seeds slide "markers on the watershed day, seeds here",
  failure modes (shrinking bias, metrication). Recalled in units 1 and 3.
- **`gmrf/`** (built, Ising elective, **not** a hard prerequisite): "quadratic smoothing = one
  linear system with the graph Laplacian", $(I + \lambda L)\mathbf x = \mathbf y$. Unit 1 says in
  one line: "same matrix; the walker replaces the data term by clamped seeds" and gives the
  random walker *with priors* as that link. Students without `gmrf/` lose nothing.
- **`label-connectivity/`** (built, day 9): union–find with the `unionfind` demo and its math slide
  (nearly linear). Unit 2 recalls it in one slide: Kruskal *is* labelling with edges sorted by
  weight.
- **`ws-flooding/`** (built): Meyer's priority-queue flooding, FIFO on plateaus, math slide
  "flooding is a shortest-path search". Unit 3 recalls it and re-reads it on edges.
- **`ws-gradient/`** (built): the slide "Region merging: a hierarchy" (merge tree of six basins,
  cutting the tree; h-minima as one cut) and the leakage slide. Unit 4 starts from that figure and
  generalises it; unit 3 explains leakage as "the max of a path ignores everything but one edge".
- **`morph-reconstruct/`** (built): **dynamics** of an extremum (math slide) and h-extrema. Recalled
  in unit 4 as the "dynamics" attribute.
- **k-means**: isodata = 1D 2-means in `otsu/` (core day 6); Lloyd's algorithm, objective and
  k-means++ in `kmeans/` (Gaussian mixtures elective, may not have been taken). Unit 5 therefore
  re-states the two steps of k-means in **one recall slide** (assign to the nearest centre, move
  each centre to its mean), with pointers to both, and does not depend on k-means++ or the proof.
- **CIELAB and ΔE** (`colour-perceptual/`, core day 7): SLIC's colour distance is a Lab distance
  (one line).
- **`gmm-images/`** (built): slide "Relatives: mean shift and superpixels" with the SLIC figure on
  the micrograph (243 superpixels). Unit 5 owns SLIC now; that slide stays as the short preview and
  gets a pointer here (see "Proposed reorganisation").

No prerequisite gap that needs new material elsewhere.

## Overlaps / who owns what

| Plan / unit | What this day takes | What stays there |
|---|---|---|
| `ncut/` (day 13) | Nothing taught; the Laplacian quadratic form is recalled | Graph Laplacian, Fiedler vector, normalized cuts (owner). Its superpixel pointer line → `superpixels/` |
| `graphcut-seg/` (day 13) | The **three-way** seeded comparison (sum / squares / max) and power watershed live in `ws-msf/` | The pairwise watershed vs graph-cut comparison on the seeds slide stays as is (day 13 is core and comes first) |
| `plan/watershed-day.md` optional unit 5 `ws-graphs/` (**not built**) and the `floodgrid` `msf` mode (not built, `plan/build-issues.md`) | The whole unit: watershed as MSF, Meyer = Prim from all markers, watershed cuts (Cousty et al.), power watershed, the side-by-side with graph cuts. Built here as core unit `ws-msf/` with the `graphgrid` demo | Day 10 keeps everything it has; `ws-flooding/`'s shortest-path math slide gets a one-line forward pointer |
| `ws-gradient/` "Region merging: a hierarchy" | Hierarchies as a topic: merge trees, attributes, UCM, RAG merging (`merge-trees/`) | The slide stays as the preview (h-minima as one cut of a hierarchy), plus a pointer here |
| `morph-reconstruct/` | Nothing taught; dynamics recalled as one attribute | Dynamics, h-extrema (owner) |
| `label-connectivity/` | Nothing taught; union–find reused for Kruskal and FH | Union–find (owner) |
| `kmeans/` (GMM elective) | Nothing taught; SLIC recalls the two steps in one slide | k-means as an algorithm (owner) |
| `gmm-images/` slide "Relatives: mean shift and superpixels" | SLIC as a method (owner now) | Mean shift (one slide, stays there); the SLIC bullet becomes a preview with a pointer |
| `gmrf/` (Ising elective) | Nothing; one-line link "same Laplacian system" | Quadratic smoothing, TV (owner) |
| `livewire/`, `fastmarching/` (DP elective) | Nothing; one line in `ws-msf/`: Prim adds the cheapest **edge**, Dijkstra the cheapest **path sum** | Dijkstra, live-wire, fast marching (owner) |
| `object-eval/`, `agreement/` (day 16) | Superpixel metrics (boundary recall, under-segmentation error, achievable segmentation accuracy) are defined in `superpixels/` | Object matching, F1/AP, boundary distances (owner); pointer for evaluating a whole hierarchy |
| `instances/` (day 18) | Nothing taught; optional `multicut/` points to "network predicts boundaries → watershed/agglomeration" | Instance segmentation (owner) |
| `pixelclass/` (day 14) | Nothing; one line in `superpixels/`: features per superpixel instead of per pixel | Feature stack, pixel classification (owner) |

Notation used throughout the day (stated on the first slide of each unit, consistent with day 13):
**affinity** $w_{pq}$ (large = similar; graph cuts, random walker) and **dissimilarity**
$d_{pq}$ (large = different; MST, FH, watershed). Image weights: $w_{pq} = \exp(-(I_p - I_q)^2/2\sigma^2)$,
$d_{pq} = |I_p - I_q|$ (or the gradient landscape $\max(f_p, f_q)$ for watershed). λ is not used
on this day except in the random walker with priors (weight of the prior, as in the course
convention).

## Units (one folder each: lec, quiz, exercise with #explore)

Five core units of about 75–80 min each including the exercise (they lean on day 13 and day 10, so
each starts further along than a fresh topic), plus one optional unit. Order: the random walker
first (closest to day 13: same graph, same seeds, same Laplacian), then spanning trees (a new tool),
then watershed as a spanning forest (needs both: it closes the "three ways" comparison), then
hierarchies (built from the merge order of units 2–3), then superpixels (uses units 2–4 as
"other superpixel methods" and as the graph they feed). Every lecture's "About this lecture" slide
ends with the math-layer line from README.md.

### 1. `random-walker/`: The random walker: seeds, probabilities and a linear system

Concept layer (≈ 18 content slides, 4 math):

- Recall (one slide, figure): the pixel graph and contrast weights of `graphcut-seg/`, and the
  Laplacian quadratic form of `ncut/`: "small when neighbours with a strong link have similar
  values". `ncut/` minimised it **without** seeds and needed an eigenvector. Today: **with** seeds.
- **The question the walker answers**: drop a walker on an unlabelled pixel; at each step it moves
  to a neighbour, preferring strong links (probability $w_{pq}/d_p$). It rarely crosses an edge in
  the image. The pixel's value is the **probability that it reaches an object seed before a
  background seed**. Picture: 20 seeded walks from one pixel, 17 end on the object seed: $\approx 0.85$.
- **Nobody simulates it**: the probability at a pixel is the **weighted average** of its
  neighbours' probabilities (first step: "where do I go next?"). Seeds are fixed at 1 and 0. A
  function that is the average of its neighbours everywhere except at the seeds is **harmonic**;
  finding it is **one sparse linear system** (one equation per unlabelled pixel).
- **The electrical picture**: every edge is a resistor with conductance $w_{pq}$ (resistance
  $1/w_{pq}$); object seeds at 1 V, background seeds at 0 V; the probability is the voltage. On a
  chain the voltage drops in proportion to resistance: a weak link (an image edge) takes most of
  the drop. Worked on the 1×5 chain (by-hand example (a)): voltages 1, 6/7, 5/7, 1/7, 0.
- **From probabilities to labels**: label = the seed class with the highest probability (two
  labels: threshold at 0.5). The probability map is **uncertainty for free**: values near 0.5
  show where more seeds help.
- **K labels**: one system per label (K − 1 needed: the probabilities sum to 1 at every pixel).
- **What the walker guarantees**: no isolated islands (every object region touches an object
  seed; maximum principle in the math layer), a unique answer (no local minima, unlike snakes),
  smooth boundaries that bridge small gaps in an edge instead of leaking through them.
- **Failure modes** (each with a demo preset and a figure): (1) **seed position matters**: on a
  chain with two weak edges the 0.5 crossing sits where half the total resistance is reached, not
  necessarily at the weakest edge (by-hand (c)); (2) **far from all seeds** the probabilities flatten
  towards an even split, so a large unmarked region is decided weakly; (3) **edges only**: no
  intensity model, so two regions of different brightness with no edge between them are not
  separated (fix: priors, below); (4) σ: too small and noise edges trap the walker; too large and
  real edges do not.
- **Random walker with priors** (one slide): add a per-pixel pull towards each class from an
  intensity model (the data term of `seg-energy/`), weighted by λ. Same linear system with an
  extra diagonal; the link to `gmrf/`'s $(I + \lambda L)\mathbf x = \mathbf y$ in one line (for those who took
  the Ising elective).
- **Random walker vs graph cut** on the same seeds (table + figure): both use $w_{pq}$; graph cut
  minimises the **sum of cut weights** over 0/1 labels (exact, hard labels, shrinking bias);
  the walker minimises the **sum of squared differences** over real values (exact, soft labels, no
  islands, depends on seed placement). Full three-way comparison in unit 3.
- **Cost**: a sparse symmetric positive-definite system with $n$ unknowns and ≈ 5 non-zeros per
  row (4-neighbourhood); solved iteratively (conjugate gradients) in a few hundred cheap sweeps,
  warm-started when the seeds change. Pointer: image libraries ship it as "random walker
  segmentation".
- Where this leads (one line each): `ws-msf/` (the same seeds with a max instead of squares),
  3D stacks (the same system with 6 neighbours), seeds from a detector instead of clicks.

Math layer (`## The math: … {.math}`):

- *The Dirichlet problem on a graph*: minimise $E(\mathbf x) = \tfrac12\sum_{(p,q)} w_{pq}(x_p - x_q)^2 = \tfrac12\mathbf x^\top L\mathbf x$
  with $\mathbf x_M$ fixed at the seeds. Order the nodes as marked/unmarked,
  $L = \begin{pmatrix} L_M & B \\ B^\top & L_U\end{pmatrix}$; $\nabla_{\mathbf x_U}E = 0$ gives
  $L_U\mathbf x_U = -B^\top\mathbf x_M$. Row $p$ reads $d_p x_p = \sum_q w_{pq}x_q$: the
  neighbour average.
- *Hitting probabilities solve the same equations*: first-step analysis,
  $P_p = \sum_q \tfrac{w_{pq}}{d_p}P_q$, $P = 1$ on object seeds, $0$ on background seeds; uniqueness
  ⇒ same solution. Kirchhoff's current law gives it a third time.
- *Maximum principle*: a harmonic function has no interior strict extremum, so $0 \le x_p \le 1$
  and every connected set $\{x > 0.5\}$ touches an object seed. $L_U$ is positive definite iff
  every connected component of the graph contains a seed (else it is singular: the constant on a
  seedless component is free).
- *K labels*: $L_U X = -B^\top M$ with one column per label; the columns sum to the all-ones
  vector by linearity and uniqueness.
- *Priors*: $(L_U + \lambda\sum_k\Lambda_k)\,\mathbf x^k_U = \lambda\boldsymbol\lambda^k - B^\top\mathbf m^k$
  (Grady 2005) with $\Lambda_k$ the diagonal of class likelihoods; for one class without seeds this is
  `gmrf/`'s system.
- *Conjugate gradients* in words: each step needs one multiplication by $L_U$ (a 5-point stencil);
  the number of steps grows with the contrast of the weights (condition number), which is why the
  demo floors $w_{pq}$ at a small ε.

Figures (≈ 10 of 22 slides carry an image; scripts in `tools/figs/random-walker/`):
pixel graph with weights and seeds (recall, SVG); 20 seeded random walks on a 24×24 two-region
image with their endpoints; the chain as a resistor ladder with voltages (SVG); the probability map
of the micrograph from two clicks with the 0.5 contour; three-label probability maps as RGB; seed
position on the two-weak-edges chain (bar chart of voltages with both crossings marked); a large
unseeded region fading to 0.5; two regions with no edge (walker fails) next to walker with priors
(succeeds); graph cut vs walker on a thin protrusion and on an edge with a gap (two columns); CG
residual vs iteration (plot).

**Demo `walker1d`** (new, small, the bridge to the by-hand work): a chain of 5–12 nodes drawn as a
resistor ladder; each edge's conductance set by dragging its thickness, or derived from a 1D
intensity profile with a σ slider. Seeds: the end nodes (object 1, background 0), or click any node
to toggle object/background/free. Bars show the voltages (probabilities) with the 0.5 line and the
labels; under each edge its share of the total voltage drop. Buttons *averaging sweep* (one
Gauss–Seidel sweep: every free node takes its neighbours' weighted average; converges visibly),
*solve*, *reset*. Toggle *graph cut* marks the minimum cut on the same chain (the weakest edge
between the seeds) next to the walker's 0.5 crossing. *Walkers*: release 200 seeded walks from a
clicked node, show the fraction that ends at the object seed next to the computed voltage.
Presets: "by hand (a): one weak edge", "equal weights: a straight ramp", "two weak edges: walker and
cut disagree" (by-hand (c)), "move a seed inwards", "averaging converges", "walkers estimate the
voltage". Pure SVG, instant.

**Demo `randwalk`** (new, main demo; also serves unit 3 with `data-mode="compare"`): 128×128
images: a seeded synthetic scene (a bright object with a thin protrusion, a second object touching
it with a weak boundary, an edge with a 3-px gap, two flat regions of different brightness without
an edge between them, noise slider) and `cell256.png` downsampled (`lib/cell.ts`, credit shown).
Paint seeds with `LabelPainter` (`lib/paint.ts`; 2 or 3 labels). Panels: image with seeds and the
label boundary; probability map of the selected label (diverging map, 0.5 = mid-grey) or, for 3
labels, the probabilities as RGB; an *uncertainty* view ($1 - \max_k x^k$). Controls: σ (log
slider), 4/8-neighbourhood, *priors* toggle with λ (intensity model from the seeds' histograms, as in
`graphcut`). Click a pixel: its probability, and an inset with 30 seeded walks from it.
Readouts: CG iterations, residual, time. Presets: "two clicks", "three labels", "weak boundary:
walker bridges the gap", "thin protrusion", "seed far away: fading to 0.5", "no edge between two
brightnesses: fails", "… priors fix it", "σ too small", "micrograph". In `compare` mode (unit 3):
three result panels for the same seeds: graph cut with seeds only (`lib/graph.ts` `solveSeg`, n-links
only, infinite t-links at seeds), random walker, seeded watershed / MSF (`lib/mst.ts`), plus the
power-watershed toggle; presets "leak through a gap", "shrinking on a thin structure", "seed
position", "all three agree on a clean edge".

By hand (concept): (a) chain of 5 nodes, seeds 1 (object) and 0 (background) at the ends,
weights 1, 1, 0.25, 1: resistances 1, 1, 4, 1, total 7, voltages $1, 6/7, 5/7, 1/7, 0$; label at 0.5;
where would a graph cut on the same weights cut? (the 0.25 edge: they agree). (b) one free node
joined to an object seed by $w = 3$ and to two background seeds by $w = 1$ and $w = 1$:
$x = 3/5$; change the background weights to 2 and 2: $x = 3/7$; compare with the graph-cut rule
(cheaper of 3 vs the sum of the other two; with a single free node the two methods always agree,
why?). (c) chain of 7 nodes, weights 0.2, 1, 1, 1, 1, 0.25 (resistances 5, 1, 1, 1, 1, 4, total 13):
voltages $1, 8/13, 7/13, 6/13, 5/13, 4/13, 0$; the walker's boundary is between nodes 3 and 4 (a
strong edge), while the graph cut (and, unit 3, the watershed) cuts the 0.2 edge and keeps only
node 1 as object: discuss which bias each shows. (d) move the object seed from node 1 to node 3 in
(c): nodes 1–2 hang off the seed and get voltage 1; between the seed and the background seed the
resistances are 1, 1, 1, 4 (total 7), so the voltages of nodes 4–6 are $6/7, 5/7, 4/7$ and the
0.5 crossing jumps into the weak 0.25 edge, where the graph cut also cuts now: the walker's boundary
depends on where the seeds are. (e) a 3×3 grid,
uniform weights, object seed top-left, background seed bottom-right: by symmetry, which pixels
have exactly 0.5? (the anti-diagonal). *(math)*: (f) derive $L_U\mathbf x_U = -B^\top\mathbf x_M$
from $E(\mathbf x)$; (g) show that hitting probabilities satisfy the same equations; (h) prove the
maximum principle on a connected graph; (i) show the K label probabilities sum to 1; (j) $L_U$ is
singular iff some component has no seed.

Builders: every chain value above is checked in `demos/test/graphseg-test.ts` before it goes into
the exercise.

### 2. `mst-seg/`: Spanning trees: single linkage and Felzenszwalb–Huttenlocher

Concept layer (≈ 18 content slides, 4 math):

- Switch of view (one slide): edge weights as **dissimilarities** $d_{pq} = |I_p - I_q|$ (small =
  same region). Segmentation as **grouping**: join pixels across cheap edges, never across expensive
  ones. No seeds, no energy minimum: a greedy rule.
- **Minimum spanning tree**: the cheapest set of edges that connects all pixels without cycles.
  **Kruskal**: sort edges by weight, add each edge unless it closes a cycle. "Does it close a
  cycle?" = "are both ends already in the same component?" = **union–find** (recall
  `label-connectivity/`: labelling is Kruskal with all edges of weight 0 or ∞). Worked on a 6-node
  graph (figure). Prim (grow one tree by its cheapest outgoing edge) in one line, for unit 3.
- **Single linkage**: stop Kruskal at a threshold $t$ (or delete the MST edges heavier than $t$):
  the components are the regions. Same as "connected components of the graph with only the edges
  $d < t$": thresholding the **edges**, not the pixels.
- **Chaining**: a gradual ramp or one low-contrast spot in a boundary links two regions: single
  linkage only looks at the **weakest link** between regions. And one global $t$ cannot serve both
  a textured region (needs large $t$ to hold together) and a flat region next to a faint boundary
  (needs small $t$). By-hand (b) shows it.
- **Felzenszwalb–Huttenlocher (FH)**: Kruskal with an **adaptive** merge test. For a component
  $C$, its **internal difference** $\mathrm{Int}(C)$ = the heaviest edge in its spanning tree (how
  much variation it already tolerates). Merge $C_1$ and $C_2$ across edge $d$ only if
  $d \le \min\big(\mathrm{Int}(C_1) + k/|C_1|,\ \mathrm{Int}(C_2) + k/|C_2|\big)$. A textured
  region tolerates large differences inside, a flat one only small ones; $k/|C|$ lets small
  components merge easily.
- **What $k$ does**: larger $k$ → larger components; it sets a **scale**, it is **not** a minimum
  size (a strong edge still separates small regions). A separate *minimum size* post-pass merges
  leftovers with their cheapest neighbour.
- Pre-smoothing (Gaussian σ ≈ 0.8 px, recall day 2's `canny/`) and colour (Lab distance, recall
  `colour-perceptual/`) in one slide.
- **What FH guarantees**: the result is neither **too fine** (no two neighbouring regions should
  have been merged by the test) nor **too coarse** (no region could be split by it), stated; the
  run time is dominated by the sort, $O(m\log m)$ for $m$ edges (≈ 4 edges per pixel).
- **Failure modes** (figures): leaks through one weak link remain (the test still uses the
  cheapest edge between components); irregular, stringy regions; texture boundaries weaker than
  the texture itself are not found (nothing local can find them, pointer: features, `pixelclass/`);
  sensitivity to $k$ and to noise.
- Where FH is used today: fast **over-segmentation** into irregular superpixels (unit 5 compares it
  with SLIC), object-proposal pipelines (one line).

Math layer:

- *Cut property*: for any split of the nodes into two sets, the cheapest edge across the split is
  in some MST; proof by exchange. Kruskal correctness follows.
- *Minimax paths*: in an MST, the unique path between $p$ and $q$ minimises the **largest** edge
  weight over all paths between them (proof from the cut property). So single linkage at $t$ joins
  $p$ and $q$ iff their minimax distance is below $t$.
- *Single linkage = MST cut*: deleting all MST edges heavier than $t$ gives the components of
  $\{d < t\}$.
- *FH predicate and its guarantees*: $D(C_1, C_2) = [\min_{e \in \partial(C_1, C_2)} d_e > \mathrm{MInt}(C_1, C_2)]$;
  sketch why processing edges in increasing order gives "not too fine" and "not too coarse"
  (Felzenszwalb & Huttenlocher 2004, Thm. 1–2).
- *Union–find cost*: near-linear (recall `label-connectivity/` math slide), so the sort dominates.

Figures (≈ 9 of 20; scripts in `tools/figs/mst-seg/`): pixel grid with dissimilarities as edge
labels; Kruskal on a 6-node graph in four frames; the MST of a 16×16 crop drawn over the image;
single linkage at three thresholds on the micrograph; chaining through a ramp (1D plot + 2D
result); the textured-vs-flat 1D example as a bar chart with components coloured (single linkage vs
FH); FH on the micrograph at $k$ = 50, 300, 1500; "k is not a minimum size" (small high-contrast
spots survive large $k$); failure: a leak through one weak boundary pixel.

**Demo `graphgrid`** (new, small; serves units 2 and 3; built on `gridStepper` from
`lib/gridui.ts`): a 6×6 to 10×10 grid of integers (click ± to edit), 4-neighbour edges drawn
between cells with their weights (dissimilarity $|f_p - f_q|$ or, in unit 3, altitude
$\max(f_p, f_q)$; menu). Algorithms (each a `GridAlgorithm` producing frames): **Kruskal** (the
sorted edge list next to the grid, current edge highlighted, verdict "join" / "same component:
skip", components coloured, union–find parent arrows optional); **single linkage at $t$** (slider);
**FH with $k$** (the two components' Int, size and threshold printed for the current edge, verdict
merge/keep); **Prim from markers** (unit 3: click cells to place markers; the queue of boundary
edges shown sorted; the edge that would join two trees is marked as a cut edge); *minimax path*:
click two cells to draw the MST path between them with its heaviest edge marked. Presets: "the
1D example of the exercise" (a 1×12 grid), "chaining: a ramp", "texture next to a flat region",
"FH with small and large k", "Prim from two markers" (unit 3), "a plateau: ties" (unit 3).

**Demo `fhseg`** (new, main demo of unit 2): 256×256 images: `cell256.png`, a synthetic scene (a
textured disc, two flat regions with a faint step between them, a linear ramp, a boundary with a
one-pixel low-contrast gap, noise slider) and the colour scene of the colour/GMM days (colour
mode, Lab distances). Panels: result with random colours or mean colours; boundaries over the
image; a size histogram of the components. Controls: method *FH / single linkage*; $k$ (log slider,
FH) or $t$ (single linkage); pre-smoothing σ; minimum size; 4/8-neighbourhood. Click a region: its
Int, size and threshold $\mathrm{Int} + k/|C|$; click a boundary: the cheapest edge between the two
regions and why they did not merge. Readouts: number of regions, time (sort once per image/σ,
then the merge pass is linear: live on the $k$ slider). Presets: "single linkage chains through
the ramp", "single linkage: texture or faint step, not both", "FH: both", "k sets scale, not
size", "minimum size cleanup", "leak through a one-pixel gap", "micrograph", "colour scene".

By hand (concept): (a) Kruskal on a given 6-node weighted graph: list the edges in order, accepted
or skipped, total weight; (b) the signal $[0, 6, 0, 6, 20, 20, 20, 20, 24, 24, 24, 24]$ on a
chain, $d$ = absolute difference (edges 6, 6, 6, 14, 0, 0, 0, 4, 0, 0, 0): show that no single-linkage
threshold gives three regions (keeping the textured part together needs $t > 6$, which merges
across the step of 4); run FH with $k = 8$ (zeros first: two flat components of size 4, threshold
2; the step 4 > 2: keep; the 6-edges merge the texture: thresholds 8, then $6 + 4 = 10$, then
$6 + 8/3$; the 14-edge: thresholds 8 and 2: keep): three regions. (c) the signal
$[10, 11, 13, 12, 22, 23, 21, 24]$, $k = 4$: run FH (sorted edges 1, 1, 1, 2, 2, 3, 10) to get
$\{1..4\}, \{5..8\}$; find the smallest $k$ that gives one region ($k \ge 32$: needs
$\mathrm{Int} + k/4 \ge 10$ for the component with $\mathrm{Int} = 2$). (d) predict: what happens
to the regions in (c) if every value is doubled? If noise of ±1 is added? *(math)*: (e) prove the
cut property; (f) show the MST path is a minimax path; (g) show single linkage at $t$ = components
of $\{d < t\}$ = MST minus edges $> t$; (h) show FH's result in (b) is not too fine.

### 3. `ws-msf/`: Watershed as a minimum spanning forest

Concept layer (≈ 17 content slides, 4 math):

- Recall (one slide, `ws-flooding/` figure): Meyer's flooding from markers with a priority queue.
- **Flooding on edges**: give each edge the height of the pass between its pixels,
  $d_{pq} = \max(f_p, f_q)$ (or a gradient between them). Then Meyer's algorithm is **Prim's
  algorithm started from all markers at once**: always add the cheapest edge leaving the labelled
  set. The trees grown from the markers form a **minimum spanning forest** (MSF): the cheapest set
  of edges in which every pixel is connected to exactly one marker. The edges never added between
  two trees are the **watershed cut**.
- Equivalent picture: add a virtual node joined to all markers by free edges, run Kruskal (unit 2),
  delete the virtual node: the MST falls apart into one tree per marker.
- **Minimax paths, again**: each pixel goes to the marker it can reach with the **lowest highest
  pass** (unit 2's minimax distance). That is the flooding picture: water from that marker reaches
  the pixel first. One sentence on ties (plateaus, FIFO) and why they matter (recall).
- **Drop of water**: unseeded watershed = markers at the regional minima; watershed cuts satisfy the
  drop-of-water principle (a drop on either side of a cut edge runs down to different minima),
  stated (Cousty et al.).
- **Why watershed leaks**: the path cost is the **maximum** of the edges, so only the single
  lowest pass matters; a one-pixel gap in a boundary is enough (recall `ws-gradient/` leakage
  slide, now explained). And why it does **not** shrink: there is no cost for boundary length.
- **Seeded segmentation three ways** (core slide, table + figure with the same seeds and
  weights): graph cut (`graphcut-seg/`) adds up cut weights (**sum**): short boundaries, shrinking
  bias; random walker (unit 1) adds up **squares** of differences: soft, smooth, seed position
  matters; watershed takes the **maximum** along a path: fast, no shrinking, leaks. One weight per
  edge, one choice of how to combine them.
- **Power watershed** (one slide, stated): the energy $\sum w_{pq}^{\,p}|x_p - x_q|^{\,q}$ contains
  the family: $q = 1$ gives graph cuts, $q = 2$ the random walker; as $p \to \infty$ only the
  order of the weights matters and the minimiser becomes an MSF cut (a watershed); with $q = 2$ the
  random walker then decides the ties on plateaus. (Exact table to be checked against Couprie et
  al. 2011, open question 3.)
- **Speed**: MSF/watershed is near-linear (a sort or a bucket queue), the walker a linear system,
  graph cuts a max-flow: the reason watershed is the default first step on large 3D data.
- Prim vs Dijkstra in one line: Prim adds the cheapest **edge**, Dijkstra (DP elective,
  `livewire/`) the cheapest **path sum**; replacing the sum by a max turns Dijkstra into the
  watershed (math slide of `ws-flooding/`).

Math layer:

- *Seeded MSF*: with a virtual root $r$ joined to all markers by weight $-\infty$, an MST of the
  extended graph restricted to the original nodes is an MSF rooted at the markers; Prim from $r$
  is Meyer flooding.
- *Minimax assignment*: in a seeded MSF each node lies in the tree of a marker $m$ with
  $\pi(p, m) = \min_m \min_{\text{paths}}\max_{e} d_e$ (ties aside); proof from the cut property.
- *Watershed cuts* (Cousty et al. 2009): edge-weighted definition, the drop-of-water principle, and
  the theorem "watershed cuts = MSF cuts relative to the minima", stated with a picture.
- *Power watershed*: the limit $p \to \infty$ of $\arg\min\sum w^p|x_p - x_q|^q$, stated with the
  conditions; the q = 1, 2 cases at finite p identified with unit 1 and `graphcut-seg/`.

Figures (≈ 9 of 21; scripts in `tools/figs/ws-msf/`): the 1D landscape of `ws-landscape/` with
pass heights as edge weights; Prim from two markers on a 4×4 grid in four frames; the virtual-root
construction; minimax paths from one pixel to two markers with their highest passes; a watershed
cut on a small edge-weighted graph; leak through a gap (path with its single highest edge marked);
the three-way comparison on the synthetic scene (three panels, same seeds); the 7-node chain of
unit 1 (c) under all three methods; a timing bar chart (MSF, walker, max-flow on 128², 256²).

**Demos**: `graphgrid` in *Prim from markers* mode (above); `randwalk` with `data-mode="compare"`
(above). The `floodgrid` `msf` mode planned by the watershed day is replaced by `graphgrid`.

By hand (concept): (a) a chain of 7 pixels a–g with markers A at a and B at g, edge weights
$2, 5, 1, 3, 6, 2$: run Prim from both markers (pops: a–b 2 → A; f–g 2 → B; b–c 5 → A; c–d 1 → A;
d–e 3 → A; then e–f 6 joins two trees: the cut); the minimax distance of d to A (5) and to B (6).
(b) the same chain read as a random walker and a graph cut with affinities $w = 1/d$
(resistances = the weights $d$): where does each put the boundary? (graph cut: the smallest
affinity, the 6-edge, like the watershed; the walker: total resistance 19, voltages of d and e
$11/19$ and $8/19$, so it crosses 0.5 on the 3-edge between d and e: two methods agree, one does not). (c) a 4×4 image with two markers and
one gap in a ridge (given): predict where the watershed leaks, then trace Prim to check. (d) fill
in the comparison table (cost, soft/hard, shrinks?, leaks?, needs seeds?) for the three methods.
*(math)*: (e) prove that Prim from a virtual root gives a seeded MSF; (f) prove the minimax
assignment; (g) show that any increasing transformation of the dissimilarities (adding a constant,
squaring) leaves the watershed unchanged (only the order of the weights matters), and find a small
graph where the graph cut changes when a constant is added to every cut weight.

Builders: the 4×4 grid in (c) is chosen without unintended ties, and its trace checked by the test
suite against `lib/watershed.ts` flooding (same labels).

### 4. `merge-trees/`: Hierarchical segmentation and merge trees

Concept layer (≈ 18 content slides, 3 math):

- Recall (one slide, the `ws-gradient/` merge-tree figure): merging basins across their lowest
  pass, one at a time, gives a tree; h-minima is one cut of it.
- **Why a hierarchy**: there is rarely one right scale (a nucleus, the cell, the cluster of cells).
  Compute **all** segmentations at once as a nested family, decide later (by a threshold, by
  clicking, by a measurement).
- **The merge tree (dendrogram)**: leaves = initial regions (pixels, basins or superpixels);
  every merge creates a parent node at the **level** where it happens. Cutting the tree
  horizontally at level $t$ gives a partition; raising $t$ only merges, never splits.
- **The MST already is a hierarchy**: Kruskal's merge order (unit 2) gives the single-linkage
  tree; for watershed basins it is the hierarchy by **altitude** (pass height).
- **Changing the order: attributes**. Re-weight each merge by a property of the regions it would
  join: **dynamics** (depth of the shallower basin: recall `morph-reconstruct/`), **area** (small
  regions go first, whatever the contrast), **volume** (area × depth). Same graph, same MST,
  different trees: worked on a 1D landscape of five basins (by-hand (a)).
- **The ultrametric contour map (UCM)**: give each boundary the level at which it disappears.
  One grey image holds the whole hierarchy: thresholding it at $t$ gives the partition at $t$.
  Strong, lasting boundaries are bright. Pointer: the best classical contour detectors of the
  2010s output a UCM.
- **Region adjacency graph (RAG)**: one node per region, one edge per touching pair, carrying
  statistics (mean intensities, boundary length, mean gradient along the boundary). **Greedy
  merging**: always merge the most similar pair, update its edges, repeat (a priority queue with
  lazy updates, recall `pqueue`). With "difference of means" instead of "cheapest edge" a single
  weak pixel no longer causes a leak. Statistical region merging and binary partition trees named
  in one line each.
- **Another tree: the component tree** (one slide): threshold the image at every level and nest the
  connected components (recall `label-connectivity/`): the tree of upper (or lower) level sets.
  Region merging builds trees from **edges**, the component tree from **thresholds**; the minima
  of one are the basins of the other (stated).
- **Choosing a cut**: one global level; a different level per branch (the cut that maximises a
  score, e.g. regions closest to an expected size or shape); interactive (click to merge/split).
  Evaluation of a hierarchy over all its levels (boundary precision–recall, recall `object-eval/`)
  in one line.
- **Failure modes**: greedy merging is final (an early wrong merge is never undone); attributes
  need a meaningful scale (area in px² vs µm²); a hierarchy of bad initial regions stays bad
  (garbage leaves).
- Where this leads: superpixels as leaves (unit 5), learned boundary probabilities as the weights
  of the RAG (`instances/`, optional `multicut/`), 3D supervoxel hierarchies.

Math layer:

- *Ultrametric*: the level $u(p, q)$ at which $p$ and $q$ first share a region satisfies
  $u(p, r) \le \max(u(p, q), u(q, r))$; conversely every ultrametric defines a hierarchy. For the
  single-linkage tree $u$ is the minimax distance (unit 2).
- *UCM ⇔ hierarchy*: the thresholds of a UCM are nested; a contour map is a UCM iff thresholding
  never leaves a dangling boundary piece (closed contours at every level), stated.
- *Attribute re-weighting* (Najman & Schmitt 1996; Najman 2011): extinction values; the hierarchy
  by dynamics equals the sequence of h-minima for increasing $h$.

Figures (≈ 10 of 21; scripts in `tools/figs/merge-trees/`): nucleus / cell / cluster at three
levels of one hierarchy; a dendrogram with a horizontal cut and the matching partition; the
five-basin 1D landscape with three trees (altitude, dynamics, area) side by side; the UCM of the
micrograph and two thresholds of it; a RAG drawn over an over-segmentation; greedy merging in four
frames; leak by cheapest edge vs no leak by mean difference; the component tree of a small 1D signal;
a per-branch cut vs a horizontal cut.

**Demo `mergetree`** (new, main demo): 256×256 `cell256.png` and the synthetic scene of `fhseg`.
Initial regions: watershed basins of the gradient (`lib/watershed.ts`, all minima), FH at small
$k$, or SLIC (unit 5). Hierarchy: *altitude* (single linkage on pass heights), *dynamics*, *area*,
*volume*, *RAG: difference of means*. Panels: partition at the current level (mean colours or
boundaries); the UCM as an image; the dendrogram (`lib/plot.ts`; leaves ordered so branches do
not cross, thinned above ~300 leaves) with a draggable horizontal cut line. Click a region: its
node highlights in the dendrogram, and vice versa; *merge* / *split* buttons for a per-branch
cut. Readouts: number of regions at the level; with the synthetic scene, regions matched to the
truth (IoU ≥ 0.5, `lib/eval.ts`). Presets: "altitude: the watershed hierarchy", "dynamics vs
altitude", "area: small regions first", "UCM thresholds", "cheapest edge leaks, mean difference does
not", "per-branch cut: one nucleus, one cluster", "micrograph".

By hand (concept): (a) a 1D landscape with basins a–e (minima 1, 3, 0, 4, 2; passes a|b 5, b|c 4,
c|d 7, d|e 6; widths 10, 2, 8, 3, 6): build the merge tree by altitude (b+c at 4, then a at 5, d+e
at 6, all at 7), by dynamics (dynamics b 1, d 2, a 4, e 5; c is the global minimum) and by area
(b into c, d into e, then {d, e} into {b, c}, then a); write the three-region partition of each
(altitude $\{a, b, c\}, \{d\}, \{e\}$; dynamics and area $\{a\}, \{b, c\}, \{d, e\}$). (b) the UCM of
the dynamics hierarchy on the four boundaries (b|c 1, d|e 2, a|b 4, c|d 5): threshold it at 3. (c)
a 3×3 RAG with given region means: run greedy mean-difference merging for three steps. (d) predict
which attribute keeps a small bright spot longest. *(math)*: (e) prove the ultrametric inequality
for any merge tree; (f) show that thresholding the dynamics UCM at $h$ gives the h-minima
segmentation; (g) show that single-linkage $u$ equals the minimax distance.

Builders: (a)'s dynamics and area orders are checked in the test suite against `lib/hierarchy.ts`.

### 5. `superpixels/`: Superpixels: SLIC and smaller graphs

Concept layer (≈ 17 content slides, 3 math):

- **Why superpixels**: a 256×256 image is 65 536 nodes and ≈ 260 000 edges; 500 superpixels are
  500 nodes and ≈ 1 500 edges. Superpixels keep the boundaries (if they are good), give regions
  to compute features on (mean, histogram, texture), and are the natural leaves of unit 4's
  hierarchy. The price: a boundary missed by the superpixels is missed by everything after them.
- **What we want**: (1) adhere to boundaries, (2) regular size and shape (compact, about the same
  area), (3) a chosen number $K$, (4) fast. FH (unit 2) and watershed (unit 3) give (1) and (4) but
  not (2) or (3).
- Recall (one slide): k-means in two steps (assign each point to the nearest centre, move each
  centre to its points' mean), from `otsu/` (isodata) and `kmeans/`.
- **SLIC**: k-means on 5-D points (Lab colour, $x$, $y$) with three changes: (1) centres start on
  a **regular grid** with spacing $S = \sqrt{N/K}$ (moved to the lowest gradient in a 3×3
  neighbourhood, so none starts on an edge); (2) the distance mixes colour and position,
  $D^2 = d_c^2 + (m\,d_s/S)^2$, with **compactness** $m$; (3) each centre only looks at pixels in a
  $2S \times 2S$ window around it. ≈ 10 iterations. Worked example (by-hand (a)): one pixel, two
  centres, the winner switches at $m \approx 30$.
- **Why the window**: plain k-means compares every pixel with every centre ($NK$ distances per
  iteration); SLIC compares each pixel with the few centres whose windows cover it (≈ $4N$): for
  $N = 65\,536$ and $K = 256$, 16.8 M vs 262 k distances.
- **Compactness $m$**: small $m$ → colour wins: superpixels follow edges, irregular shapes; large
  $m$ → position wins: a grid of near-squares that ignores edges. Grey images: $d_c$ on intensity,
  $m$ in grey-level units.
- **Connectivity**: k-means does not know that a superpixel should be one piece; small orphan
  fragments are relabelled to a neighbour afterwards (connected components and union–find
  recalled).
- **Measuring superpixels** (figure + numbers): **boundary recall** (fraction of true boundary
  pixels within 2 px of a superpixel boundary), **under-segmentation error** (how much superpixels
  spill over true regions), **achievable segmentation accuracy** (the best Dice/accuracy any method
  that labels whole superpixels could reach: an upper bound for everything after). Trade-off curves
  against $K$ for SLIC, FH and watershed superpixels.
- **Using them**: build the RAG (unit 4) and run graph cut, the random walker, ncut or a
  hierarchy on it; classify superpixels with region features (pointer `pixelclass/`). Same seeds,
  pixel graph vs superpixel graph: similar result, a fraction of the time; but a thin structure
  narrower than a superpixel is lost.
- Variants (one line each): compact watershed, SEEDS / energy-driven boundary moves, supervoxels
  in 3D and video (the same algorithm with $z$ or $t$ added).
- Pointer: image libraries ship SLIC, FH and compact watershed under those names.

Math layer:

- *SLIC objective*: $\sum_k\sum_{p \in C_k} D^2(p, \mathbf c_k)$; the update step is the mean in
  5-D (minimises the sum of squared $D$ because $D^2$ is a weighted Euclidean norm with weights
  $1$ and $m^2/S^2$).
- *Why monotonicity can break*: k-means' proof needs every pixel to be compared with every centre;
  with windows that move with the centres, a pixel's candidate set changes between iterations, so
  the objective can rise (stated; in practice ≈ 10 iterations suffice).
- *Cost*: $O(N)$ per iteration independent of $K$ (each pixel is covered by ≈ 4 windows).
- *Achievable segmentation accuracy* as an upper bound: the best labelling constant on each
  superpixel is the majority truth label of each superpixel; its accuracy bounds every
  superpixel-level method.

Figures (≈ 9 of 20; scripts in `tools/figs/superpixels/`): the micrograph as 65 536 pixels vs 300
superpixels with their RAG; SLIC grid initialisation and one centre's $2S$ window; SLIC
iterations 0, 1, 3, 10; compactness $m$ = 5, 20, 80 side by side; orphan fragments before and after
the connectivity step; SLIC vs FH vs watershed superpixels at the same $K$; boundary recall and
under-segmentation error against $K$ (plot, three methods); graph cut on pixels vs on superpixels
(two columns, with run times); a thin filament lost by large superpixels.

**Demo `slic`** (new, main demo): 256×256 images: `cell256.png` (grey), the colour scene of the
colour/GMM days (with ground-truth regions), an H&E crop if a licensed one is found (open question
6). Panels: superpixel boundaries over the image; mean-colour rendering; for a selected centre,
its window and the pixels it won. Controls: $K$ (25–1000, log), $m$, iterations (step / play /
reset, showing the centres move), *connectivity* toggle (orphans highlighted when off), method
tabs *SLIC / FH / watershed* (the latter two from `lib/mst.ts`, `lib/watershed.ts`, with $K$
matched approximately). Readouts on the synthetic scene: boundary recall, under-segmentation
error, achievable accuracy, run time; a small plot of these against $K$ for the three methods
(computed once). *Use them* tab: paint seeds (`LabelPainter`) and run graph cut or the random
walker on the pixel graph and on the superpixel RAG side by side, with times. Presets: "grid
start", "m small: follows edges", "m large: squares", "orphans without connectivity", "same K,
three methods", "graph cut on superpixels", "thin filament lost", "micrograph".

By hand (concept): (a) a grey pixel $p$ with $I = 50$ at $(4, 4)$, centres A ($I = 52$ at
$(0, 0)$) and B ($I = 70$ at $(6, 4)$), $S = 8$: compute $D^2$ for $m = 10$ (A: 54, B: 406.25 → A)
and $m = 40$ (A: 804, B: 500 → B); find the $m$ where $p$ switches ($m^2(32 - 4)/64 = 396$,
$m \approx 30$). (b) distances per iteration for plain k-means and SLIC at $N = 65\,536$,
$K = 256$ (16.8 M vs ≈ 262 k). (c) superpixel metrics on a printed 8×8 truth and a 4-superpixel
partition: boundary recall, achievable accuracy. (d) predict: what do SLIC superpixels look like on
a flat region? On a textured one? With $K$ too small for the objects? (e) RAG sizes and graph-cut
cost reduction for 500 superpixels. *(math)*: (f) show the update step minimises the weighted
squared distance; (g) construct a 1D example where a moving window raises the objective; (h) show
that achievable accuracy is an upper bound.

### 6. (optional) `multicut/`: Attraction and repulsion: multicut and the mutex watershed

- Signed edges: some pairs should be together (attractive), some apart (repulsive), e.g. from a
  network that predicts boundary probabilities (pointer `instances/`). The **multicut**
  (correlation clustering) objective: cut the repulsive edges, keep the attractive ones, number
  of regions not fixed; NP-hard in general.
- Greedy solutions as Kruskal variants: greedy additive edge contraction; the **mutex watershed**
  (Wolf et al. 2018): process all edges by |weight|; an attractive edge merges unless a repulsive
  "mutex" constraint between the two clusters forbids it; a repulsive edge adds such a constraint.
  No seeds and no threshold.
- Connection back: with only attractive edges and a threshold it is single linkage (unit 2); the
  "seeds" of watershed are a special case of mutex constraints.
- **Demo**: `graphgrid` with `data-mode="mutex"` (signed weights shown as colours) and a 128×128
  synthetic boundary-probability image in `mergetree` (`data-mode="mutex"`).

## Shared engineering

New library files owned by this day (names checked: no collision with `demos/src/lib/`):

- **`demos/src/lib/randomwalker.ts`** (new): Laplacian assembly from a grid and affinities
  (reuses `gridEdges`, `contrastWeights` from `lib/graph.ts`, with an ε floor), seeded system
  split ($L_U$, $B$) as a matrix-free 5/9-point operator, **Jacobi-preconditioned conjugate
  gradients** `cgSolve(apply, b, x0, {tol, maxIter, onIter})` (generic; `lib/deconv.ts` has an
  internal CG for TV: point it here if the deconvolution owner agrees), `randomWalker(img, w, h,
  seeds, {sigma, conn, labels, lambda, priors, x0})` → per-label probabilities, iterations,
  residual; `gaussSeidelSweep` for `walker1d`; `simulateWalks(graph, start, seeds, n, rng)` (seeded,
  `lib/random.ts`); chain helpers (voltages by resistances) for the 1D demo and tests.
- **`demos/src/lib/mst.ts`** (new): edge lists with dissimilarities (`|ΔI|`, Lab ΔE via
  `lib/colour.ts`, altitude $\max(f_p, f_q)$), `kruskal` (with a recorded trace for
  `graphgrid`), `singleLinkage(t)`, `felzenszwalb(img, {k, sigma, minSize, conn, colour})` (sort
  once, cache, re-run the linear merge pass on a new $k$), `primForest(markers)` = seeded MSF with
  FIFO tie-break (`lib/pqueue.ts`), `minimaxPath(p, q)` on a tree, `virtualRootMSF`. Union–find
  from `lib/label.ts` (`UnionFind`; extend it there with `size` and a per-set `maxEdge` if needed,
  through its owner, the labelling day).
- **`demos/src/lib/hierarchy.ts`** (new): merge tree from a Kruskal/Prim order (parent array,
  levels), `cutAt(level)`, `ucm(tree, labels)` (boundary saliency image), attribute re-weighting
  (dynamics, area, volume; extinction values), `rag(labels, img)` (adjacency, boundary lengths,
  region statistics), greedy RAG merging with a lazy priority queue (`lib/pqueue.ts` `Heap`),
  `componentTree(img)` (small images, for the one-slide figure), dendrogram layout (leaf order,
  coordinates) for `lib/plot.ts`.
- **`demos/src/lib/superpixels.ts`** (new): `slic(img, {K, m, iters, colour, trace})` with grid
  initialisation and gradient perturbation, windowed assignment, mean update (reusing the distance
  conventions of `lib/kmeans.ts`; plain k-means stays there), `enforceConnectivity` (via
  `lib/label.ts`), metrics `boundaryRecall(sp, truth, tol)`, `undersegmentationError`,
  `achievableAccuracy` (truth matching via `lib/metrics.ts`; boundary masks via `lib/eval.ts`
  `boundaryMask`), `superpixelGraph(sp, img)` → RAG for `lib/graph.ts`'s `solveSeg` and
  `randomwalker.ts`.
- **Extensions of existing files (through their owners):** `lib/graph.ts` (graph-cut day): a
  `solveSeg` variant with seeds only (infinite t-links, no data term) and on an arbitrary edge list
  (for the superpixel RAG), if `solveSeg` does not already accept one; `lib/gridui.ts`: an edge-
  weight overlay for `GridView` (numbers between cells, highlight one edge) if `CellArrow` is not
  enough; `lib/synth.ts`: the synthetic scene of `randwalk`/`fhseg`/`mergetree` (with ground-truth
  regions), and the colour scene of `colourgmm` moved there (already listed in
  `plan/build-issues.md`).
- **Demos**: `walker1d`, `randwalk` (+ `compare`), `graphgrid` (+ `mutex`), `fhseg`, `mergetree`,
  `slic`. All in the gallery `demos.qmd`.
- **Speed** (TypeScript first, measured): CG on 128² (16 k unknowns, 9-point stencil) needs a few
  hundred iterations at low σ: estimate 30–80 ms, so warm-start from the previous solution on every
  seed stroke, solve at 64² while painting and at 128² on release; if a full solve on release
  still exceeds ~30 ms *and* feels sluggish, move `cgSolve` to Rust/Wasm (`demos/wasm/randomwalker/`)
  per `demos/README.md`, keeping the TS reference. FH on 256² (≈ 260 k edges): sort once per
  image/σ (≈ 20 ms), merge pass ≈ 5 ms per $k$ → live. SLIC 256², 10 iterations: a few ms per
  iteration. Hierarchy and UCM from an existing over-segmentation: a few ms. No WebGL.
- **Tests** (`demos/test/graphseg-test.ts`, run by `make test`): random walker on chains equals the
  resistor formula; probabilities in [0, 1] and summing to 1 for 3 labels; CG agrees with
  `lib/linalg.ts` `solve` on small grids; simulated walks within 3 standard errors of the solved
  probability (fixed seed); Kruskal total weight equals Prim's on random graphs and equals brute
  force on ≤ 8 nodes; single linkage at $t$ equals `label2d` of the thresholded edge graph; FH
  reproduces the by-hand examples of unit 2 ((b): three regions; (c): two regions at $k = 4$, one at
  $k = 32$, two at $k = 31.9$); seeded MSF labels equal `lib/watershed.ts` flooding on edge
  altitude for random integer images without ties, and every pixel's label is a minimax-nearest
  marker; merge tree levels form an ultrametric; dynamics hierarchy cut at $h$ equals
  `hMinimaMarkers` + flooding in 1D; unit 4 (a) orders; SLIC: every superpixel connected after
  `enforceConnectivity`, the by-hand distances of unit 5 (a); achievable accuracy ≥ any
  superpixel-constant labelling on random cases; all by-hand numbers in this plan.
- **Images**: `cell256.png` via `lib/cell.ts` (credit shown in every demo using it); synthetic
  scenes labelled "synthetic" in captions; an H&E crop only with a verified licence.
- **Lessons from `plan/build-issues.md`**: prefix every per-demo CSS rule with
  `.qb-demo[data-demo="<name>"]` (demo.css loads late); use `![](fig.svg){fig-alt="..."}` (bracket
  text becomes a visible caption on slides); add `.nostretch` to lone slide images; use
  `lib/plot.ts` (its clip-path ids are fixed) rather than local plot code; labels ≥ 18 px on
  slides (the dendrogram and `graphgrid` edge numbers need checking at 1150×560); keep static
  figures as the PDF fallback; static figure scripts live in `tools/figs/<unit>/`, not in the
  scratchpad.
- Follow `demos/README.md`: TypeScript, `make typecheck`, no network, fixed seeds, presets +
  reset, `slider`/`button`/`checkbox` helpers (keyboard stays with reveal.js), `data-*` settings,
  fits 1150×560 on a slide (`randwalk` compare mode and `slic` use two panels + readout on slides,
  all panels with `data-layout="full"` on the exercise page).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:15 | Unit 1 `random-walker/`: lecture (`walker1d`, `randwalk`), chains by hand |
| 10:30–11:45 | Unit 2 `mst-seg/`: lecture (`graphgrid` Kruskal/FH), FH by hand, `fhseg` |
| 12:45–14:00 | Unit 3 `ws-msf/`: lecture (`graphgrid` Prim), Prim by hand, `randwalk` compare |
| 14:15–15:30 | Unit 4 `merge-trees/`: lecture, three trees by hand, `mergetree` |
| 15:45–17:00 | Unit 5 `superpixels/`: lecture, SLIC by hand, `slic`; quizzes as wrap-up |

## Proposed reorganisation

1. **Move the watershed day's optional `ws-graphs/` here as core unit `ws-msf/`.** It was never
   built (`plan/build-issues.md`, together with the `floodgrid` `msf` mode). Its content
   (watershed as MSF, Meyer = Prim, watershed cuts, the graph-cut comparison, power watershed) is
   exactly this day's unit 3, and it is much stronger after the random walker and spanning trees
   than as an optional tail of day 10. `plan/watershed-day.md` unit 5 then becomes a one-line
   pointer, and the `floodgrid` `msf` mode is dropped in favour of `graphgrid` (Prim from
   markers). Built-unit edit (later, not here): a forward pointer line in `ws-flooding/` (math slide
   "flooding is a shortest-path search") and in `ws-gradient/` (region-merging and leakage slides).
2. **Ownership changes for `plan/coursemap.md`** (owners table): random walker → this day;
   minimum spanning trees / single linkage / FH → this day (union–find stays with labelling);
   hierarchical segmentation (merge trees, UCM, RAG merging) → this day (`ws-gradient/` keeps its
   preview slide, `morph-reconstruct/` keeps dynamics); superpixels (SLIC) → this day
   (`gmm-images/` keeps its "Relatives" slide as a preview, and mean shift; `kmeans/` keeps k-means).
   Built-unit edits (later): `ncut/` pointer line ("superpixels, SLIC, mean shift") → points to
   `superpixels/`; `gmm-images/` SLIC bullet → pointer; `graphcut-seg/` seeds slide → one line
   "a third way, the random walker, and all three side by side: graph-based segmentation elective".
3. **Sidebar placement**: in the group "Curves and energies" (as the course map says), **after
   Ising/MRF**: it needs only day 13, but students who did `gmrf/` recognise the Laplacian system,
   and it is not part of the snakes → level sets → DP chain. Optionally rename the group to
   "Curves, graphs and energies", since this day has no curves. (Alternative: a group of its own,
   "Graphs", with this day only; not proposed, since one-day groups make the sidebar longer.)
4. **Topic page "Segmentation"** (planned in the course map): add the random walker, FH, SLIC and
   hierarchies; the three-way seeded comparison (sum / squares / max) is a natural centre for that
   page. **Prerequisite database**: hard = day 13 (`graphcut-seg/`, `ncut/`), day 10
   (`ws-flooding/`, `ws-gradient/`), `label-connectivity/`; soft = `gmrf/`, `kmeans/`,
   `morph-reconstruct/`, `colour-perceptual/`.

## Open questions

1. **Five core units**: the day is long (5 × 75 min). Fallbacks in order: make `superpixels/` a
   shorter unit (drop the metrics math and the "use them" tab from the lecture, keep it in the
   exercise); merge the RAG material of `merge-trees/` into `superpixels/`; or make
   `merge-trees/` optional. Proposal: keep five, since each unit is shorter than a fresh topic.
2. **Order of units 1 and 2**: random walker first (proposal: continuity with day 13) or spanning
   trees first (Kruskal is simpler; then unit 3 directly after unit 2)? With the proposal, unit 3
   needs both and sits well after them.
3. **Power watershed table**: check the exact $(p, q)$ table of Couprie et al. 2011 (which limits
   give graph cuts, the random walker, MSF cuts and shortest-path forests) before writing the slide;
   if in doubt, keep the slide to "q = 1 graph cuts, q = 2 random walker, p → ∞ watershed with ties
   broken by q" and cite the paper.
4. **Random walker vs graph cut claims**: "bridges weak boundaries / less shrinking" is the
   literature's claim (Grady 2006); the demo presets must show what actually happens on our
   synthetic scene and the slides report it honestly (as several units already do).
5. **Component tree**: one slide in `merge-trees/` (proposal), or leave it to a pointer? It is not
   taught on the morphology day; if that day later adds attribute filters, this slide becomes a
   recall.
6. **Real images**: a licensed colour image with annotated regions (for superpixel metrics on a
   real image; the BSDS-style datasets have licence restrictions) and an H&E crop (the colour and
   unmixing days have the same need). Until then: synthetic scenes with truth, and the micrograph
   without metrics.
7. **Optional `multicut/`**: worth building, or a pointer slide at the end of `merge-trees/`? It is
   the bridge to modern instance segmentation (`instances/`); proposal: optional unit, built after
   the core.
8. **Wasm for CG**: decide after measuring `randwalk` in the browser (see Shared engineering).

## References

- Grady (2006), *Random walks for image segmentation*, PAMI.
- Grady (2005), *Multilabel random walker image segmentation using prior models*, CVPR.
- Doyle & Snell (1984), *Random Walks and Electric Networks*, MAA (teacher background: harmonic
  functions, hitting probabilities, effective resistance).
- Kruskal (1956), *On the shortest spanning subtree of a graph and the traveling salesman problem*,
  Proc. AMS; Prim (1957), *Shortest connection networks and some generalizations*, Bell Syst. Tech. J.
- Zahn (1971), *Graph-theoretical methods for detecting and describing gestalt clusters*, IEEE
  Trans. Computers (MST clustering).
- Felzenszwalb & Huttenlocher (2004), *Efficient graph-based image segmentation*, IJCV.
- Meyer (1994), *Topographic distance and watershed lines*, Signal Processing.
- Cousty, Bertrand, Najman & Couprie (2009), *Watershed cuts: minimum spanning forests and the drop
  of water principle*, PAMI.
- Allène, Audibert, Couprie & Keriven (2010), *Some links between extremum spanning forests,
  watersheds and min-cuts*, Image Vis. Comput.
- Couprie, Grady, Najman & Talbot (2011), *Power watershed: a unifying graph-based optimization
  framework*, PAMI.
- Najman & Schmitt (1996), *Geodesic saliency of watershed contours and hierarchical
  segmentation*, PAMI; Najman (2011), *On the equivalence between hierarchical segmentations and
  ultrametric watersheds*, JMIV.
- Salembier & Garrido (2000), *Binary partition tree as an efficient representation for image
  processing, segmentation and information retrieval*, IEEE TIP.
- Nock & Nielsen (2004), *Statistical region merging*, PAMI.
- Arbeláez, Maire, Fowlkes & Malik (2011), *Contour detection and hierarchical image
  segmentation*, PAMI (UCM).
- Achanta, Shaji, Smith, Lucchi, Fua & Süsstrunk (2012), *SLIC superpixels compared to
  state-of-the-art superpixel methods*, PAMI.
- Stutz, Hermans & Leibe (2018), *Superpixels: an evaluation of the state-of-the-art*, CVIU
  (boundary recall, under-segmentation error, achievable segmentation accuracy).
- Neubert & Protzel (2014), *Compact watershed and preemptive SLIC*, ICPR.
- Andres, Kappes, Beier, Köthe & Hamprecht (2011), *Probabilistic image segmentation with
  closedness constraints*, ICCV (multicut); Wolf et al. (2018), *The mutex watershed*, ECCV.

## Build notes (2026-11-13)

Built: the five core units (`random-walker/`, `mst-seg/`, `ws-msf/`, `merge-trees/`, `superpixels/`),
demos `walker1d`, `randwalk` (+ `data-mode="compare"`), `graphgrid`, `fhseg`, `mergetree`, `slic`,
new libs `lib/randomwalker.ts`, `lib/mst.ts`, `lib/hierarchy.ts`, `lib/superpixels.ts`, shared
synthetic scenes in `demos/src/fhseg/scenes.ts` (candidates for `lib/synth.ts`), test
`demos/test/graphseg-test.ts`, figure scripts in `tools/figs/<unit>/` (compute.ts with the course
libs → JSON → figs.py). Optional `multicut/` not built.

Deviations and honest results:
- Random walker speed: Jacobi-preconditioned CG needs 400–1300 iterations at 128² (0.3–1 s in
  node); the demo solves progressively per animation frame with warm starts. A quick aggregation
  multigrid test gave only ~2×; Wasm alone would not fix the iteration count. Open: multigrid or
  a direct sparse solver.
- On clean synthetic edges the graph cut (seeds only) and the walker give identical results;
  differences appear with weak edges, small seeds and noise (reported as such). In the gap scene
  the walker bulges slightly through the gap while the graph cut closes it; the watershed leaks
  only with strong noise.
- The power-watershed toggle of `randwalk` compare mode was not built (stated on a slide only).
- `graphgrid` draws its own SVG (no `gridStepper`: its GridView has no edge-weight overlay).
- Merge-tree "area" attribute is defined as the area of the smaller region at the moment of the
  merge (extinction-style), which reproduces the by-hand orders of unit 4 (a).
- SLIC on the grey test scene adheres less to boundaries than FH and the watershed at equal K
  (reported in `superpixels/`); the colour scene is easy for all three.
- The SLIC connectivity clean-up gives each orphan the label of its left/upper neighbour (as in
  the original SLIC), which can cross true boundaries at small m.
