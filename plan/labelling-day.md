# Plan: a day on connected-component labelling and region properties

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
Decided 2026-11-12: labelling is its own day, directly after the morphology day.

## Goal of the day

After the day a student can:

1. define **4-, 8-connectivity** (and 6/18/26 in 3D), paths and **connected components**, and explain the **digital topology paradox**: why foreground and background must use complementary connectivities (8/4 or 4/8);
2. label a small binary image by hand with **flood fill** (BFS or DFS) and with the **two-pass algorithm** using a **union–find** structure, and state why path compression and union by rank make it near-linear;
3. compute **region properties** of a labelled object by hand: area, centroid, bounding box, perimeter (and the bias of each estimator), **Euler number**;
4. compute the **second-moment (covariance) matrix** of a region and read **orientation**, **elongation / eccentricity** and the equivalent ellipse from its eigenvalues and eigenvectors;
5. build a measurement pipeline (threshold → clean with morphology → label → measure → **filter by properties** → **count**), know when a **label image** is needed instead of a binary mask, and recognise the counting errors from touching and border objects.

Why it fits this audience: labelling is a graph problem with a classic data structure
(union–find) that can be traced completely by hand, and the region properties are
integrals turned into sums. The second-moment matrix is a symmetric 2×2 matrix whose
eigenvectors are the principal axes of the object: the spectral theorem and PCA from
their linear algebra, applied to a blob they just drew. The day is also the hinge of the
course: every segmentation method (threshold, watershed, graph cuts, level sets) ends in
a label image, and this day says what to do with it.

## Prerequisite gap

- **Thresholding and binary masks**: taught on the thresholding day
  (`plan/thresholding-day.md`), hysteresis included; one-line recall.
- **Morphology** (previous day): opening, fill holes, clear border and reconstruction are
  used as known. The link "the component of $p$ is $R_A(\{p\})$" is stated there in one
  line and developed here in unit 1.
- **Queue and stack**: assumed from programming; one slide with a picture of each.
  Union–find is new and gets the slow treatment (picture, by hand, then cost).
- **Covariance matrix, eigen-decomposition of a symmetric matrix**: known from linear
  algebra and statistics; recalled in one slide in unit 3 (real eigenvalues, orthogonal
  eigenvectors, $C = Q \Lambda Q^\top$).
- **Distance transform** is not needed.

## Links to other days

- **Morphology** (previous day): component = reconstruction from a single seed pixel;
  flood fill = geodesic dilation with the cross (4) or square (8) SE, one ring per BFS
  level. Area opening (keep components with area ≥ $a$) as the *connected* counterpart of
  an opening: it removes small objects without changing the shape of the others, like
  opening by reconstruction. The connectivity conventions (foreground 8, background 4)
  are the ones used there.
- **Watershed**: regional minima are the components of plateaus (computed with this day's
  code); the watershed output is a label image measured with unit 3–4 tools. Watershed
  unit 1 recalls connected components in one line with a pointer here. Touching nuclei counted as one object (unit 4 here) are the
  motivation for that day.
- **Graph cuts**: a binary cut must still be labelled to count objects; α-expansion and
  normalized cuts return label images directly. The union–find structure reappears in
  Kruskal's minimum spanning tree (watershed on graphs, unit 5 there).
- **Skeletons** (`plan/skeleton-day.md`): thinning must preserve the number of
  components and holes, i.e. the Euler number from unit 3; skeleton length and branch
  counts are further per-object properties.
- **Level sets**: the front may split or merge; the number of components of
  $\{\phi < 0\}$ is counted with this day's labelling.
- **Marching cubes**: one surface per label; the 3D connectivity pair (26/6) is the
  voxel-side version of the marching-cubes ambiguity; Euler number of a voxel set vs
  Euler characteristic $V - E + F$ of the extracted mesh.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise.

### 1. `label-connectivity/`: Connectivity, components and flood fill

- Motivation: a thresholded image of nuclei. "How many are there, and how big is each?"
  The binary mask does not know which pixel belongs to which object.
- Neighbourhoods: $N_4$, $N_8$ (and the diagonal $N_D$); a **path**, a **connected
  set**, a **component** = maximal connected set. 4 vs 8 gives different counts on the
  same mask (two diagonal-touching pixels).
- **The digital topology paradox**: a closed 8-connected diagonal curve. With 8 for both
  foreground and background, the curve is closed but the inside leaks out diagonally (it
  separates and does not separate); with 4 for both, a closed 4-curve can leave the
  background in pieces that touch only diagonally. Digital Jordan curve theorem
  (Rosenfeld): it holds for the pairs (8, 4) and (4, 8). Convention for the course:
  foreground 8, background 4. The hexagonal grid as the "no paradox" picture (one slide).
- 3D: $N_6$ (faces), $N_{18}$ (+ edges), $N_{26}$ (+ corners); valid pairs (26, 6) and
  (6, 26) (18/6 also used). One picture of the 3×3×3 cube in each mode.
- **Flood fill**: from a seed, BFS with a queue (grows ring by ring, like geodesic
  dilation from the morphology day) or DFS with a stack (grows in long arms). Recursive
  DFS overflows the call stack on a 1000×1000 blob: use an explicit stack. Scan-line
  flood fill (fill a whole run, push the runs above and below) as the paint-bucket
  algorithm.
- **Labelling by repeated flood fill**: scan the image; at every unlabelled foreground
  pixel start a new label and flood. Every pixel is visited a bounded number of times:
  $O(n)$. Random memory access is its practical weakness (motivates unit 2).
- **Demo `morphgrid`** with `data-mode="label"` (moved here from the morphology plan;
  reuses that demo's grid painting and *step* / *step row* / *run* / *reset* controls):
  a 16×16 grid (up to 24×24). Sub-modes *flood BFS* / *flood DFS*: the queue or stack is
  shown beside the grid as a list of cells, the current pixel outlined, each new label
  in its own colour with the number written in the cell. Toggle foreground 4/8 (the
  background connectivity follows as the complement). Presets: "diagonal neighbours: one
  or two objects?", "BFS rings vs DFS arms", "a spiral", "count with 4 and with 8".
- **Demo `connparadox`** (new, small): a 12×12 binary grid with a diagonal closed curve
  preset. Separate selectors for foreground and background connectivity (4/8 each).
  Shows foreground components and background components in two colour families, and a
  verdict line: "1 curve, inside and outside separate" / "inside leaks out" / "the curve
  is broken". A 3D tab: a 5×5×5 voxel set (WebGL via `lib/gl3d.ts`, rotate by
  dragging) with a voxel ring that touches only along edges or corners, selectors
  6/18/26 for foreground and background. Presets: "8/8: closed but leaky", "4/4: no
  closed curve", "8/4 works", "3D: corner-touching voxels".
- By hand: (a) a 8×8 mask: count components with 4 and with 8, and count the background
  components (holes) with the complementary connectivity. (b) BFS flood fill from a
  given seed: write the queue contents for the first eight steps.

### 2. `label-unionfind/`: Two-pass labelling and union–find

- **Two-pass algorithm** (Rosenfeld & Pfaltz 1966): raster scan; for each foreground
  pixel look only at the already-visited neighbours (4: up, left; 8: up-left, up,
  up-right, left). None labelled → new provisional label; one label → copy it; several
  labels → copy the smallest and record that they are **equivalent**. Second pass:
  replace each provisional label by its class representative, renumbered 1, 2, 3…
- Why equivalences arise: a U-shape gets two labels on its arms that meet at the bottom.
  A comb or spiral gives long chains of equivalences.
- **Union–find** (disjoint-set forest): `parent` array, $\text{find}(x)$ follows parents to
  the root, $\text{union}(a, b)$ links one root under the other. Without care, trees become
  chains and find costs $O(n)$. **Union by rank** (or size): attach the shallower tree
  under the deeper one → depth $\le \log_2 n$. **Path compression** (or path halving):
  point every visited node to the root during find. Both together: amortised
  $O(\alpha(n))$ per operation, with $\alpha$ the inverse Ackermann function ($\le 4$ for any
  image that fits in the universe): "near-linear" total cost. Stated with intuition,
  not proved.
- **Run-length / scan-line variants**: label runs (maximal horizontal segments) instead of
  pixels. A run in row $y$ joins every run in row $y - 1$ that overlaps it (4) or overlaps
  it after widening by one pixel (8). Fewer union operations, sequential memory access;
  run-length encoding is also a compact mask format. Block-based labelling (Grana et al.:
  2×2 blocks, decision trees) named as what fast libraries do.
- **3D**: the same two passes over a volume; the already-visited neighbours are 3 for
  $N_6$ and 13 for $N_{26}$. Cost and memory: a 512³ stack has 134 M voxels, so the label
  type (uint16 overflows at 65 535 objects; use uint32) and run-based labelling matter.
- Complexity summary table: flood fill $O(n)$ random access; two-pass $O(n\,\alpha(n))$
  sequential; run-based $O(n + r\,\alpha(r))$ for $r$ runs.
- **Demo `morphgrid`** with `data-mode="label"`, sub-mode *two-pass*: first pass stepped
  pixel by pixel with the visited neighbours highlighted and the provisional label
  written in each cell; the equivalence table beside the grid; then the second pass
  recolouring the cells. Sub-mode *runs*: runs drawn as bars, unions drawn as links
  between rows. Presets: "a U needs one equivalence", "a comb: many", "a spiral",
  "4 vs 8 on the same image".
- **Demo `unionfind`** (new): the union–find forest drawn in SVG as trees of numbered
  nodes (up to ~30), driven either by the equivalences recorded from a `morphgrid`
  first pass (same preset images) or by a typed/clicked list of union operations.
  Toggles: *union by rank* on/off, *path compression* off / full / halving. Each find
  animates the path to the root and then the compression rewiring. Readout: tree
  heights, total pointer steps so far. Presets: "naive unions build a chain",
  "union by rank keeps trees flat", "compression flattens on the way", "equivalences
  from the comb image".
- By hand: (a) two-pass labelling of a 6×6 image with a U-shaped object and a second
  object, with 4 and with 8: provisional labels, equivalence table, final labels.
  (b) A sequence of 8 unions on 8 elements: draw the forest without and with union by
  rank, then do one find with path compression.

### 3. `regionprops/`: Region properties and moments

- One row per label: what can be measured from the set of pixels $\Omega_k$ of object
  $k$. Everything is a sum over the pixels (or over the boundary), computed in one scan
  for all labels at once.
- **Area** $m_{00} = |\Omega|$ (times pixel area for physical units). **Centroid**
  $\bar x = m_{10}/m_{00}$, $\bar y = m_{01}/m_{00}$ with raw moments
  $m_{pq} = \sum_{(x,y) \in \Omega} x^p y^q$. **Bounding box**; extent = area / box area.
- **Perimeter** and its bias: (i) count pixel edges between object and background:
  for a digitised disc this tends to $8r$, not $2\pi r$, an overestimate of $4/\pi \approx
  27\,\%$ that does **not** go away at higher resolution; (ii) the 8-chain code with
  steps 1 and $\sqrt 2$: a few per cent over, still orientation-dependent; (iii) better
  weights (e.g. 0.948 / 1.343, Vossepoel & Smeulders) or the Cauchy–Crofton formula
  (count crossings with lines in several directions) or the marching-squares contour
  length (marching-cubes day). Consequence: **circularity** $4\pi A / P^2$ is biased and
  can exceed 1 for small objects.
- **Central moments** $\mu_{pq} = \sum (x - \bar x)^p (y - \bar y)^q$ (translation
  invariant). The **second-moment matrix**
  $C = \frac{1}{m_{00}}\begin{pmatrix}\mu_{20} & \mu_{11}\\ \mu_{11} & \mu_{02}\end{pmatrix}$
  is the covariance matrix of the pixel coordinates. Symmetric, so real eigenvalues
  $\lambda_1 \ge \lambda_2 \ge 0$ and orthogonal eigenvectors: the **principal axes**.
  Orientation $\theta = \tfrac12 \operatorname{atan2}(2\mu_{11}, \mu_{20} - \mu_{02})$ (derived
  from the eigenvector of a 2×2 matrix, by hand). The **equivalent ellipse** has the same
  second moments: semi-axes $a = 2\sqrt{\lambda_1}$, $b = 2\sqrt{\lambda_2}$ (a solid
  ellipse has variance $a^2/4$ along its axis; check for a disc). **Eccentricity**
  $\sqrt{1 - \lambda_2/\lambda_1}$, elongation $\sqrt{\lambda_1/\lambda_2}$.
- Link to linear algebra named explicitly: this is PCA of the point cloud of pixels, and
  the spectral theorem $C = Q\Lambda Q^\top$; rotating the object rotates $Q$ and leaves
  $\Lambda$ unchanged. Degenerate case: $\lambda_1 = \lambda_2$ (disc, but also a square
  or a regular polygon): orientation is undefined, so it should not be reported.
- Invariants in one slide: central moments (translation), normalised moments (scale),
  eigenvalues and Hu's seven invariants (rotation), named not derived.
- **Euler number** $E = \#\text{components} - \#\text{holes}$ (with the 8/4 pair).
  Holes = background components not touching the border (labelling the complement, or
  via fill holes from the morphology day). Surprise: $E$ can be computed **locally** by
  counting 2×2 patterns (Gray's bit-quads) in one scan, without labelling. Thinning
  (skeleton day) preserves $E$; in 3D the Euler number becomes components − tunnels +
  cavities (one sentence, link to $V - E + F$ on the marching-cubes day).
- **Demo `moments`** (new, main demo): a 160×160 canvas with one or two objects, drawn
  with a brush or placed from shapes (ellipse, rectangle, L, ring, crescent) that can be
  dragged (centre handle), rotated (rim handle) and scaled. Overlays: centroid,
  bounding box, the equivalent ellipse, the two eigenvectors as arrows of length
  $2\sqrt{\lambda_i}$; side panel with $m_{00}$, $\mu_{20}, \mu_{11}, \mu_{02}$, the matrix
  $C$, $\lambda_{1,2}$, $\theta$, eccentricity, Euler number, and three perimeter estimates
  (edge count, chain code, Crofton) against the exact value for analytic shapes. A
  *resolution* slider re-digitises the shape and a small plot shows perimeter estimate /
  true value versus resolution. Presets: "rotate the ellipse: the eigenvectors follow,
  the eigenvalues do not", "a square: equal eigenvalues", "the L-shape: the ellipse is
  not the shape", "a ring: Euler number 0", "a digital disc: 8r not 2πr",
  "circularity above 1".
- By hand: (a) a 5×5 object given as a pixel list: area, centroid, $\mu_{20}, \mu_{11},
  \mu_{02}$, the matrix $C$, its eigenvalues and eigenvectors, orientation. (b) Perimeter
  of a digitised disc of radius 4 by edge count and by chain code, compared with $2\pi
  r$. (c) Euler number of a small image with two objects, one with a hole, by counting
  components and holes; optionally check with bit-quads.

### 4. `label-measure/`: Working with label images: filter, count, measure

- **Label image vs binary mask**: two touching objects are one blob in a mask but two
  labels in a label image (the boundary between them exists only there). 0 = background,
  integers per object; mask of object $k$ is $L = k$. Data type (uint16/uint32), relabel
  sequentially after deletion, display with a random colour map with neighbouring labels
  kept distinct (the shared `labelColors` in `lib/imageops.ts`). Sources: this day's
  labelling, watershed, graph cuts, level sets, and modern networks (Cellpose/StarDist)
  all output label images.
- **The measurement table**: one row per object, one column per property; a scatter
  plot of two properties (area vs eccentricity) separates populations (debris, round
  nuclei, rod-shaped bacteria) that a single threshold cannot.
- **Filtering objects by properties**: remove small objects (= area opening, a connected
  filter: shapes of the others unchanged, unlike a morphological opening), keep round
  ones (eccentricity, circularity), remove border objects (bounding box touches the
  frame, or clear border from the morphology day). Hysteresis thresholding (the idea
  is taught on the thresholding day, `plan/thresholding-day.md`, unit 3): here only the
  implementation, reconstruction of the low mask from the high mask, or label the low
  mask and keep labels containing a high pixel (one slide).
- **Counting** and its errors: touching objects merge (undercount → watershed day),
  fragmented objects split (overcount → closing before labelling), noise specks
  (→ area filter). **Border objects**: discarding all of them biases against large
  objects; the unbiased counting frame (Gundersen: count objects touching the top/left
  edges, exclude bottom/right) in one slide.
- The full pipeline on one real image: threshold → open → fill holes → label → measure →
  filter → count, with the effect of each step on the count.
- **Demo `regionfilter`** (new): 256×256, three linked panels. Left: image with labels
  coloured, a selected object outlined; middle: the measurement table (sortable,
  click a row to highlight the object); right: scatter plot of two chosen properties,
  brush a rectangle to keep only objects inside it (the others greyed out). Inputs: a
  seeded synthetic image of round nuclei, elongated rods, specks and a few touching
  pairs; the thresholded synthetic nuclei of the watershed day; a thresholded
  `cell256.png`. Controls: threshold, pre-cleaning (open radius, fill holes on/off),
  connectivity 4/8, border rule (keep / remove / counting frame). Readout: count, mean
  area. If `lib/watershed.ts` exists: a *split touching objects* toggle that swaps the
  labelling for a distance-transform watershed. Presets: "debris: filter by area",
  "rods vs discs: eccentricity", "two touching nuclei count as one", "border objects:
  counting frame", "connectivity changes the count".
- By hand: (a) a 10×10 label image and a property table: which labels survive "area ≥ 5
  and eccentricity < 0.8"? Relabel sequentially. (b) Count objects in a frame using the
  unbiased counting rule. (c) Write down the pipeline (as a list of steps, not code) for
  a given task and image.

## Shared engineering

- `demos/src/lib/label.ts` (**owned by this day**; the morphology, watershed and
  skeleton days call it):
  - Neighbourhood offsets for 4/8 and 6/18/26 and the "already visited" half for raster
    scans (coordinate with `lib/graph.ts` from the graph-cut plan, which builds grid
    neighbourhoods: one source for the offset tables).
  - Flood fill (BFS / DFS / scan-line) with an instrumented version that records the
    queue or stack per step for `morphgrid`.
  - Two-pass labelling with union–find (union by rank, path compression or halving,
    counters for pointer steps), instrumented version that records provisional labels,
    equivalences and the forest after each union for `morphgrid` and `unionfind`.
  - Run-based labelling (also returns the runs), 3D labelling (6/18/26), relabel
    sequential, background components and holes.
  - Region properties in one scan for all labels: area, centroid, bounding box, raw and
    central moments to order 2, the 2×2 covariance with closed-form eigenvalues and
    eigenvectors, orientation, eccentricity, three perimeter estimators, circularity,
    Euler number (bit-quads), touches-border flags.
  - Filters: by predicate on properties, area opening, unbiased counting frame.
- Reused, not duplicated: `lib/morph.ts` (morphology day: opening, fill holes, clear
  border, reconstruction for hysteresis), `lib/imageops.ts` (loading `cell256.png`,
  seeded synthetic nuclei, label-to-colour map; add a "mixed population" generator of
  discs, rods and specks), `lib/gl3d.ts` (marching-cubes day) for the 3D tab of
  `connparadox`, `lib/watershed.ts` and `lib/distance.ts` (optional toggle in
  `regionfilter`), and the SVG plot helper (`lib/plot.ts`) for the scatter plot and the
  perimeter convergence plot.
- `morphgrid` gets the `label` mode here; its grid and step controls stay in the
  morphology day's demo folder. If the mode makes `morphgrid` too large, split the grid
  UI into `lib/gridui.ts` and make `labelgrid` a separate bundle (open question).
- **Language / speed** (`demos/README.md`): **TypeScript** throughout. Two-pass
  labelling of 256×256 takes well under 5 ms and region properties are one more scan;
  a 128³ volume (2 M voxels) labels in tens of ms. No Rust/Wasm expected; measure first.
  WebGL only for the small 3D tab.
- Tests (`make test`, as in the other plans): on random 2D and 3D binary images, flood
  fill = two-pass = run-based (equal up to a permutation of labels) for every
  connectivity; union–find with and without the heuristics gives the same partition;
  bit-quad Euler number = components − holes from labelling; moments of a digitised
  ellipse give its orientation and axes within a tolerance; perimeter of large digital
  discs by edge count / $2\pi r$ → $4/\pi$; area opening = brute-force removal.
- Images: `demos/src/fourier2d/cell256.png` (credit as in `fourier2d`); synthetic images
  generated in TS with fixed seeds (no files).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only (no modifier keys
  or hover), keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide,
  `data-*` settings (`data-mode`, `data-image`, `data-preset`), static figures as print
  fallbacks. Grid cells ≥ 24 px; label numbers in cells readable on a projector.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, counting by hand with 4 and 8, `connparadox`, flood fill in `morphgrid` |
| 10:45–12:15 | Unit 2 lecture, two-pass by hand, `morphgrid` two-pass, `unionfind` |
| 13:15–14:45 | Unit 3 lecture, moments by hand, `moments` |
| 15:00–16:30 | Unit 4 lecture + exercise (`regionfilter`); quizzes as wrap-up |

## Open questions

- How long is a "day" (same question as for the other days)? Units 1 and 2 are the
  lighter ones and could merge if the day is short; unit 4 could be shortened to the
  `regionfilter` exercise.
- `morphgrid` with a `label` mode (proposal, reuses the grid) or a separate `labelgrid`
  demo sharing `lib/gridui.ts`?
- 3D: only the small `connparadox` tab and one slide (proposal), or labelling a
  marching-cubes volume (e.g. the nuclei stack) with one colour per component?
- How far into perimeter estimators: edge count and chain code with their bias (core),
  Crofton and optimal weights as one slide (proposal)?
- Hu invariants and higher moments: named only (proposal), or a short exercise?
- A real image with many separate objects of different shapes (nuclei plus debris, or
  bacteria) would make unit 4 stronger; same licence search as the watershed and
  morphology plans.
- Programming language for any code parts is still open (see README).

## References

- Rosenfeld & Pfaltz (1966), *Sequential operations in digital picture processing*, JACM (two-pass labelling).
- Rosenfeld (1970), *Connectivity in digital pictures*, JACM (digital topology, 4/8 pairs).
- Kong & Rosenfeld (1989), *Digital topology: introduction and survey*, CVGIP.
- Tarjan (1975), *Efficiency of a good but not linear set union algorithm*, JACM (union–find, inverse Ackermann).
- He, Ren, Gao, Zhao, Yao & Chao (2017), *The connected-component labeling problem: a review of state-of-the-art algorithms*, Pattern Recognition.
- Grana, Borghesani & Cucchiara (2010), *Optimized block-based connected components labeling with decision trees*, IEEE Trans. Image Processing.
- Gray (1971), *Local properties of binary images in two dimensions*, IEEE Trans. Computers (bit-quads, Euler number).
- Vossepoel & Smeulders (1982), *Vector code probability and metrication error in the representation of straight lines of finite length*, CGIP (perimeter weights).
- Hu (1962), *Visual pattern recognition by moment invariants*, IRE Trans. Information Theory.
- Gundersen (1977), *Notes on the estimation of the numerical density of arbitrary profiles: the edge effect*, J. Microscopy (unbiased counting frame).
- Gonzalez & Woods, *Digital Image Processing*, chapters on connectivity and representation & description.
