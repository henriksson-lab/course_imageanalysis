# Plan: a day on connected-component labelling and region properties

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
Decided 2026-11-12: labelling is its own day, directly after the morphology day.

**Course map (2026-11-12):** core day 8 of 17, Part II "From pixels to objects". Order:
day 7 morphology → this → day 9 watershed. Entry point of the elective group "Objects"
(skeletons, moments). Edits from the three reviews applied 2026-11-12: units 1–2 merged;
`shape-features/` moved here from the measurements day (day 10); PCA recalled from `pca/`
(day 1) instead of introduced here; object-level evaluation is the evaluation day's (day 15).

## Goal of the day

After the day a student can:

1. define **4-, 8-connectivity** (and 6/18/26 in 3D), paths and **connected components**, and explain the **digital topology paradox**: why foreground and background must use complementary connectivities (8/4 or 4/8);
2. label a small binary image by hand with **flood fill** (BFS or DFS) and with the **two-pass algorithm** using a **union–find** structure, and state why path compression and union by rank make it near-linear;
3. compute **region properties** of a labelled object by hand: area, centroid, bounding box, perimeter (and the bias of each estimator), **Euler number**;
4. compute the **second-moment (covariance) matrix** of a region and read **orientation**, **elongation / eccentricity** and the equivalent ellipse from its eigenvalues and eigenvectors, recognising it as the PCA of day 1 applied to pixel coordinates;
5. compute the **shape descriptors beyond moments** by hand on a small object: convex hull and its area (shoelace formula), solidity, form factor, equivalent diameter, **minimum and maximum Feret diameter** (rotating calipers), maximum inscribed radius (distance transform), and say how each depends on resolution and the perimeter estimator;
6. build a measurement pipeline (threshold → clean with morphology → label → measure → **filter by properties** → **count**), know when a **label image** is needed instead of a binary mask, and recognise the counting errors from touching and border objects.

Why it fits this audience: labelling is a graph problem with a classic data structure
(union–find) that can be traced completely by hand, and the region properties are
integrals turned into sums. The second-moment matrix is a symmetric 2×2 matrix whose
eigenvectors are the principal axes of the object: the PCA of day 1 (`pca/`), applied
to the cloud of pixel coordinates of a blob they just drew. The shape descriptors add
geometry (convex hull, calipers) to the integrals. The day is also the hinge of the
course: every segmentation method (threshold, watershed, graph cuts, level sets) ends in
a label image, and this day says what to do with it.

## Prerequisite gap

- **Thresholding and binary masks**: taught on day 5 (`plan/thresholding-day.md`),
  hysteresis included; one-line recall.
- **Morphology** (day 7, the previous day): opening, fill holes, clear border,
  reconstruction and the **distance transform** (owned there) are used as known. The link "the component of $p$ is $R_A(\{p\})$" is stated there in one
  line and developed here in unit 1.
- **Queue and stack**: assumed from programming; one slide with a picture of each.
  Union–find is new and gets the slow treatment (picture, by hand, then cost).
- **Covariance matrix, eigenvectors, PCA**: taught in `pca/` (day 1, the owner); recalled
  in one slide in unit 2 (real eigenvalues, orthogonal eigenvectors, $S = Q \Lambda Q^\top$,
  eigenvalues in descending order).
- **Distance transform** (day 7, morphology owns it): used for the inscribed radius in
  unit 3; one-line recall.

## Links to other days

- **Morphology** (previous day): component = reconstruction from a single seed pixel;
  flood fill = geodesic dilation with the cross (4) or square (8) SE, one ring per BFS
  level. Area opening (keep components with area ≥ $a$) as the *connected* counterpart of
  an opening: it removes small objects without changing the shape of the others, like
  opening by reconstruction. The connectivity conventions (foreground 8, background 4)
  are the ones used there.
- **Watershed** (day 9, next day): regional minima are the components of plateaus
  (computed with this day's code); the watershed output is a label image measured with
  unit 2–4 tools. Watershed unit 1 recalls connected components in one line with a
  pointer here. Touching nuclei counted as one object (unit 4 here) are the motivation
  for that day.
- **Measurements** (day 10): continues the catalogue with intensity, texture, relations
  and colocalisation; it recalls this day's region properties and shape features as rows.
- **Graph cuts** (day 12): a binary cut must still be labelled to count objects;
  α-expansion and normalized cuts return label images directly. The union–find structure
  reappears in Kruskal's minimum spanning tree (watershed on graphs, unit 5 there).
- **Evaluation and statistics** (day 15, `plan/evaluation-day.md`): owns object-level
  evaluation (matching predicted to true objects by IoU, object F1/AP, split/merge
  errors). Here only one forward pointer in unit 4: "a count can be right while the
  objects are wrong".
- **Skeletons** (elective, `plan/skeleton-day.md`): thinning must preserve the number of
  components and holes, i.e. the Euler number from unit 2; skeleton length and branch
  counts are further per-object properties.
- **Moments** (elective, `plan/moments-day.md`): recalls orders 0–2 from `regionprops/`
  and the shoelace formula from `shape-features/`, and adds higher orders and invariants.
- **Level sets** (elective): the front may split or merge; the number of components of
  $\{\phi < 0\}$ is counted with this day's labelling.
- **Marching cubes** (elective; owns `volumes3d/`, the course's 3D-images unit): one
  surface per label; the 3D connectivity pair (26/6) is the voxel-side version of the
  marching-cubes ambiguity; Euler number of a voxel set vs Euler characteristic
  $V - E + F$ of the extracted mesh.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise. Unit 1 merges the former units
1 (`label-connectivity/`) and 2 (`label-unionfind/`), which this plan already called the
lighter ones (review-organisation §2.6); unit 3 is the `shape-features/` unit moved here
from the measurements day.

### 1. `label-connectivity/`: Connectivity, components, flood fill and union–find

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
  (6, 26) (18/6 also used). One picture of the 3×3×3 cube in each mode; stacks and voxel
  spacing are a pointer to `volumes3d/` (3D elective).
- **Flood fill**: from a seed, BFS with a queue (grows ring by ring, like geodesic
  dilation from day 7) or DFS with a stack (grows in long arms). Recursive DFS overflows
  the call stack on a 1000×1000 blob: use an explicit stack. Labelling by repeated flood
  fill is $O(n)$, but with random memory access. Scan-line flood fill named in one line.
- **Two-pass algorithm** (Rosenfeld & Pfaltz 1966): raster scan; for each foreground
  pixel look only at the already-visited neighbours (4: up, left; 8: up-left, up,
  up-right, left). None labelled → new provisional label; one label → copy it; several
  labels → copy the smallest and record that they are **equivalent**. Second pass:
  replace each provisional label by its class representative, renumbered 1, 2, 3…
  A U-shape gets two labels on its arms that meet at the bottom.
- **Union–find** (disjoint-set forest): `parent` array, find follows parents to the root,
  union links one root under the other. Without care, trees become chains. **Union by
  rank** keeps depth $\le \log_2 n$; **path compression** points visited nodes to the
  root. Together: amortised $O(\alpha(n))$ per operation ("near-linear"), stated with
  intuition, not proved (math slide).
- Run-based and block-based labelling, and 3D two-pass (3 / 13 visited neighbours; label
  type uint32 for large stacks), in one slide with the complexity table: flood fill $O(n)$
  random access; two-pass $O(n\,\alpha(n))$ sequential; run-based $O(n + r\,\alpha(r))$.
- **Demo `morphgrid`** with `data-mode="label"` (the morphology day's grid on
  `lib/gridui.ts`, step controls): a 16×16 grid (up to 24×24). Sub-modes *flood BFS* /
  *flood DFS* (queue or stack shown beside the grid), *two-pass* (visited neighbours
  highlighted, provisional labels in the cells, equivalence table, then the second-pass
  recolouring) and *runs*. Toggle foreground 4/8 (background follows as the complement).
  Presets: "diagonal neighbours: one or two objects?", "BFS rings vs DFS arms", "a U
  needs one equivalence", "a comb: many", "count with 4 and with 8".
- **Demo `connparadox`** (new, small): a 12×12 binary grid with a diagonal closed curve
  preset. Separate selectors for foreground and background connectivity (4/8 each);
  verdict line ("inside and outside separate" / "inside leaks out" / "the curve is
  broken"). A 3D tab: a 5×5×5 voxel set (WebGL via `lib/gl3d.ts`) with selectors
  6/18/26. Presets: "8/8: closed but leaky", "4/4: no closed curve", "8/4 works", "3D:
  corner-touching voxels".
- **Demo `unionfind`** (new): the union–find forest drawn in SVG (up to ~30 nodes), driven
  by the equivalences of a `morphgrid` two-pass run or by clicked union operations.
  Toggles: union by rank, path compression off / full / halving. Presets: "naive unions
  build a chain", "union by rank keeps trees flat", "compression flattens on the way".
- By hand: (a) a 8×8 mask: count components with 4 and with 8, and the background
  components (holes) with the complementary connectivity. (b) Two-pass labelling of a 6×6
  image with a U-shaped object, with 8-connectivity: provisional labels, equivalence
  table, final labels. (c) A sequence of 6 unions: draw the forest without and with union
  by rank, then one find with path compression.

### 2. `regionprops/`: Region properties and moments

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
  length (marching-cubes elective). Consequence: **circularity** $4\pi A / P^2$ is biased
  and can exceed 1 for small objects (developed in unit 3).
- **Central moments** $\mu_{pq} = \sum (x - \bar x)^p (y - \bar y)^q$ (translation
  invariant). The **second-moment matrix**
  $S = \frac{1}{m_{00}}\begin{pmatrix}\mu_{20} & \mu_{11}\\ \mu_{11} & \mu_{02}\end{pmatrix}$
  is the sample covariance matrix of the pixel coordinates. **Recall slide (`pca/`, day 1):**
  "this is PCA of the pixel coordinates": symmetric, so real eigenvalues
  $\lambda_1 \ge \lambda_2 \ge 0$ (descending, as in `pca/`) and orthogonal eigenvectors: the
  **principal axes**. Orientation $\theta = \tfrac12 \operatorname{atan2}(2\mu_{11}, \mu_{20} - \mu_{02})$
  (derived from the eigenvector of a 2×2 matrix, by hand; math slide). The **equivalent
  ellipse** has the same second moments: semi-axes $a = 2\sqrt{\lambda_1}$, $b = 2\sqrt{\lambda_2}$
  (a solid ellipse has variance $a^2/4$ along its axis; check for a disc).
  **Eccentricity** $\sqrt{1 - \lambda_2/\lambda_1}$, elongation $\sqrt{\lambda_1/\lambda_2}$.
- Rotating the object rotates $Q$ in $S = Q\Lambda Q^\top$ and leaves $\Lambda$ unchanged.
  Degenerate case: $\lambda_1 = \lambda_2$ (disc, but also a square or a regular polygon):
  orientation is undefined, so it should not be reported.
- Invariants in one slide: central moments (translation), normalised moments (scale),
  eigenvalues and Hu's seven invariants (rotation), named not derived (moments elective).
- **Euler number** $E = \#\text{components} - \#\text{holes}$ (with the 8/4 pair).
  Holes = background components not touching the border (labelling the complement, or
  via fill holes from day 7). Surprise: $E$ can be computed **locally** by counting 2×2
  patterns (Gray's bit-quads) in one scan, without labelling. Thinning (skeleton
  elective) preserves $E$; in 3D the Euler number becomes components − tunnels +
  cavities (one sentence).
- **Demo `moments`** (new, main demo; the course's shape-editing shell on
  `lib/shapecanvas.ts`): a 160×160 canvas with one or two objects, drawn with a brush or
  placed from shapes (ellipse, rectangle, L, ring, crescent) that can be dragged, rotated
  and scaled. Overlays: centroid, bounding box, the equivalent ellipse, the two
  eigenvectors as arrows of length $2\sqrt{\lambda_i}$; side panel with $m_{00}$,
  $\mu_{20}, \mu_{11}, \mu_{02}$, the matrix $S$, $\lambda_{1,2}$, $\theta$, eccentricity,
  Euler number, and three perimeter estimates (edge count, chain code, Crofton) against
  the exact value for analytic shapes. A *resolution* slider re-digitises the shape and a
  small plot shows perimeter estimate / true value versus resolution. Presets: "rotate
  the ellipse: the eigenvectors follow, the eigenvalues do not", "a square: equal
  eigenvalues", "the L-shape: the ellipse is not the shape", "a ring: Euler number 0",
  "a digital disc: 8r not 2πr".
- By hand: (a) a 5×5 object given as a pixel list: area, centroid, $\mu_{20}, \mu_{11},
  \mu_{02}$, the matrix $S$, its eigenvalues and eigenvectors, orientation. (b) Perimeter
  of a digitised disc of radius 4 by edge count and by chain code, compared with $2\pi
  r$. (c) Euler number of a small image with two objects, one with a hole, by counting
  components and holes; optionally check with bit-quads.

### 3. `shape-features/`: Size and shape beyond moments (moved from the measurements day)

- Picture first: four objects with nearly equal area (disc, ellipse, star, crescent):
  which numbers tell them apart?
- Recalled from unit 2 in one line each: area, perimeter (edge count overestimates a disc
  by $4/\pi$), extent, axis lengths $4\sqrt{\lambda_{1,2}}$, eccentricity. **Equivalent
  diameter** $d = \sqrt{4A/\pi}$ (2D) vs the equivalent-sphere diameter $\sqrt[3]{6V/\pi}$
  (3D): same name, different formula in different tools.
- **Convex hull**: smallest convex set containing the object. For a digital object take
  the hull of the pixel **corners** (or of the pixel centres: a convention that changes
  small objects a lot; with centres a 1-pixel-wide line has hull area 0). Algorithm in
  words: Andrew's monotone chain on boundary points, $O(n \log n)$. **Area by the
  shoelace formula** (the course's home of the shoelace formula; Green's theorem is its
  math layer, recalled by the moments and marching-cubes electives). **Solidity** =
  $A / A_{\text{hull}}$ (1 for convex, low for stars, crescents, clumps of two touching
  nuclei: a classic "should I split this?" feature, and the bridge to day 9).
- **Form factor / circularity** $4\pi A / P^2$: 1 for a disc. Inherits the perimeter bias
  of unit 2: with edge-count perimeter a digital disc gives $\pi/4 \approx 0.79$; with a
  corrected estimator small discs give values **above 1**. Rule: compare form factors
  only within one tool and one resolution. **Compactness** is a name collision (second
  moments $2\pi\,\overline{r^2}/A$, or $P^2/(4\pi A)$, or a 3D $S^3/(36\pi V^2)$): show all
  on the same shapes in one slide.
- **Feret diameters** (caliper diameters): the width $w(\theta)$ between two parallel
  support lines at angle $\theta$. **Max Feret** = the diameter of the point set; **min
  Feret** = the smallest width. Both are attained on the convex hull; **rotating
  calipers** find them in $O(h)$ for $h$ hull vertices (min width has one caliper flush
  with a hull edge). Small worked example by hand on a 5-vertex polygon.
- **Radii from the distance transform** (recall from day 7, morphology owns
  $\mathrm{DT}$): **maximum inscribed radius** $\max \mathrm{DT}$ (thickness, insensitive to
  protrusions); mean / median radius over the object. A rod of width $w$ has max radius
  $\approx w/2$ whatever its length; contrast with min Feret. Pointer: the largest disc
  is a point of the medial axis (skeleton elective). Optional row: minimum enclosing
  circle.
- **Contour descriptors** (one slide, review-gaps gap 13): the boundary as a complex
  signal $x(t) + i\,y(t)$, its Fourier coefficients (recall `fourier/`, day 1);
  keeping $k$ coefficients and reconstructing gives a smoothed outline; dropping the DC
  term and normalising by $|c_1|$ gives translation and scale invariance. Demo mode
  below.
- Pitfalls slide: units (µm vs px: area scales with the square, ratios do not);
  **anisotropic pixels/voxels** (compute in physical coordinates, spacing
  $(\Delta x, \Delta y, \Delta z)$); objects touching the **border** are truncated (drop
  them or use the counting frame of unit 4); **resolution**: a 10-pixel object has no
  meaningful form factor; **segmentation error**: a one-pixel dilation changes area by
  $\approx P/A = 2/r$ for a disc.
- **Demo `shapefeat`** = `moments` shell with `data-mode="features"` (decided: one
  shape-editing UI on `lib/shapecanvas.ts`, answering the measurements plan's open
  question). Shape presets (disc, ellipse, rectangle, star, crescent, two touching discs,
  rod, L), drag/rotate/scale or polygon editing. Overlays: convex hull (corner or centre
  convention), the calipers at a draggable $\theta$ with a plot of $w(\theta)$, min and max
  Feret, the maximum inscribed circle, minimum enclosing circle, equal-area circle; a
  *Fourier outline* overlay with a $k$ slider. Side panel: live feature table with a
  *units* toggle (px / µm, anisotropic option) and exact values for analytic shapes;
  *resolution* and *mask error* (±2 px) sliders. Presets: "disc: form factor 0.79 with
  edge-count perimeter", "small disc: form factor above 1", "star vs disc: same area,
  solidity tells", "two touching nuclei: solidity drops", "rod: min Feret ≈ 2 × inscribed
  radius", "anisotropic pixels: the circle that is not", "k Fourier coefficients".
- By hand: (a) for a 6×6 object (pixel list), the convex hull of the pixel corners, its
  area (shoelace), solidity and extent; (b) min and max Feret of a 4–5 vertex polygon by
  checking the calipers flush with each edge; (c) max inscribed radius and mean radius
  from a given distance transform; (d) form factor of a 4×4 square with the edge-count
  perimeter vs the true value $\pi/4$.

### 4. `label-measure/`: Working with label images: filter, count, measure

- **Label image vs binary mask**: two touching objects are one blob in a mask but two
  labels in a label image (the boundary between them exists only there). 0 = background,
  integers per object; mask of object $k$ is $L = k$. Data type (uint16/uint32), relabel
  sequentially after deletion, display with a random colour map with neighbouring labels
  kept distinct (the label palette in `lib/colour.ts`). Sources: this day's labelling,
  watershed (day 9), graph cuts (day 12), level sets (elective), and instance-segmentation
  networks (day 16) all output label images.
- **The measurement table**: one row per object, one column per property; a scatter
  plot of two properties (area vs solidity) separates populations (debris, round nuclei,
  rod-shaped bacteria) that a single threshold cannot.
- **Filtering objects by properties**: remove small objects (= area opening, a connected
  filter: shapes of the others unchanged, unlike a morphological opening), keep round
  ones (eccentricity, form factor, solidity), remove border objects (bounding box touches
  the frame, or clear border from day 7). Hysteresis thresholding (the idea is taught on
  day 5, thresholding unit 3): here only the implementation, reconstruction of the low
  mask from the high mask, or label the low mask and keep labels containing a high pixel
  (one slide).
- **Counting** and its errors: touching objects merge (undercount → day 9, watershed),
  fragmented objects split (overcount → closing before labelling), noise specks
  (→ area filter). **Border objects**: discarding all of them biases against large
  objects; the unbiased counting frame (Gundersen: count objects touching the top/left
  edges, exclude bottom/right) in one slide. Forward pointer (one line): a right count
  with wrong objects is caught only by object-level matching, day 15.
- The full pipeline on one real image: threshold → open → fill holes → label → measure →
  filter → count, with the effect of each step on the count.
- **Demo `regionfilter`** (new; the course's feature-table + scatter shell, reused by
  measurements `featurespace`-style views, RF `featimp` and moments `momentclass`):
  256×256, three linked panels. Left: image with labels coloured, a selected object
  outlined; middle: the measurement table (sortable, click a row to highlight the
  object); right: scatter plot of two chosen properties, brush a rectangle to keep only
  objects inside it. Inputs: a seeded synthetic image of round nuclei, elongated rods,
  specks and a few touching pairs (`lib/synth.ts`); a thresholded `cell256.png`.
  Controls: threshold, pre-cleaning (open radius, fill holes on/off), connectivity 4/8,
  border rule (keep / remove / counting frame). Readout: count, mean area. Presets:
  "debris: filter by area", "rods vs discs: eccentricity", "two touching nuclei count as
  one (solidity flags them)", "border objects: counting frame", "connectivity changes the
  count". (The *split touching objects* toggle is dropped here: watershed is the next
  day and can reuse this demo with its own mode.)
- By hand: (a) a 10×10 label image and a property table: which labels survive "area ≥ 5
  and solidity > 0.9"? Relabel sequentially. (b) Count objects in a frame using the
  unbiased counting rule. (c) Write down the pipeline (as a list of steps, not code) for
  a given task and image.

## Shared engineering

- `demos/src/lib/label.ts` (**owned by this day**; the morphology, watershed and
  skeleton days call it):
  - Neighbourhood offsets for 4/8 and 6/18/26 and the "already visited" half for raster
    scans: imported from the "grid" section of `lib/graph.ts` (owner: graph-cut day), one
    source for the offset tables.
  - Flood fill (BFS / DFS / scan-line) with an instrumented version that records the
    queue or stack per step for `morphgrid`.
  - Two-pass labelling with union–find (union by rank, path compression or halving,
    counters for pointer steps), instrumented version that records provisional labels,
    equivalences and the forest after each union for `morphgrid` and `unionfind`.
  - Run-based labelling (also returns the runs), 3D labelling (6/18/26), relabel
    sequential, background components and holes.
  - Region properties in one scan for all labels: area, centroid, bounding box, raw and
    central moments to order 2, the 2×2 covariance $S$ with eigenvalues and eigenvectors
    from the closed form in `lib/linalg.ts` (not in `imageops.ts`), orientation,
    eccentricity, three perimeter estimators, circularity, Euler number (bit-quads),
    touches-border flags.
  - Filters: by predicate on properties, area opening, unbiased counting frame.
- `demos/src/lib/convexhull.ts` (**owned by this day**, moved with `shape-features/`):
  monotone-chain hull of pixel corners or centres, shoelace area, rotating calipers (min
  width, max diameter, $w(	heta)$ for plotting), minimum enclosing circle (Welzl, seeded
  shuffle). Used by measurements, skeleton and snakes.
- `demos/src/lib/shapecanvas.ts` (**owned by this day**): the shape-editing canvas behind
  the `moments` shell (brush, shape presets, drag/rotate/scale, polygon vertices,
  re-digitise at a resolution). Modes: `moments` (unit 2), `features` (unit 3, the former
  `shapefeat`), and the moments elective's `higher` / `invariants` / `momentclass` modes.
  Shape features (equivalent diameter, form factor, compactness variants, solidity,
  Feret, inscribed radius, Fourier outline) are computed in `lib/features.ts` (measurements
  day), first written here for unit 3.
- Reused, not duplicated: `lib/morph.ts` (day 7: opening, fill holes, clear border,
  reconstruction for hysteresis), `lib/distance.ts` (day 7: inscribed radius),
  `lib/imageops.ts` (`loadGrey()` for `cell256.png`), `lib/synth.ts` (seeded synthetic
  nuclei; add a "mixed population" generator of discs, rods and specks), `lib/colour.ts`
  (label palette), `lib/fft.ts` (Fourier outline), `lib/gl3d.ts` (owner: 3D graphics
  elective) for the 3D tab of `connparadox`, and `lib/plot.ts` for the scatter plot and
  the perimeter convergence plot.
- `morphgrid` gets the `label` mode here; its grid UI is `lib/gridui.ts` (owner:
  morphology day), so the mode stays in the morphology day's demo folder (decided; no
  separate `labelgrid` bundle).
- **Language / speed** (`demos/README.md`): **TypeScript** throughout. Two-pass
  labelling of 256×256 takes well under 5 ms and region properties are one more scan;
  a 128³ volume (2 M voxels) labels in tens of ms. No Rust/Wasm expected; measure first.
  WebGL only for the small 3D tab.
- Tests (`make test` over `demos/test/*-test.ts`): on random 2D and 3D binary images, flood
  fill = two-pass = run-based (equal up to a permutation of labels) for every
  connectivity; union–find with and without the heuristics gives the same partition;
  bit-quad Euler number = components − holes from labelling; moments of a digitised
  ellipse give its orientation and axes within a tolerance; perimeter of large digital
  discs by edge count / $2\pi r$ → $4/\pi$; area opening = brute-force removal; convex hull
  vs brute force on random point sets; rotating-calipers min width and max diameter vs
  brute force; shoelace area of a digitised disc's corner hull → $\pi r^2$; form factor of
  an analytic disc polygon = 1.
- Images: `demos/src/fourier2d/cell256.png` (credit as in `fourier2d`); synthetic images
  generated in TS with fixed seeds (no files).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only (no modifier keys
  or hover), keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide,
  `data-*` settings (`data-mode`, `data-image`, `data-preset`), static figures as print
  fallbacks. Grid cells ≥ 24 px; label numbers in cells readable on a projector.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, counting and two-pass by hand, `connparadox`, `morphgrid` (label), `unionfind` |
| 10:45–12:15 | Unit 2 lecture, moments by hand, `moments` |
| 13:15–14:45 | Unit 3 lecture, hull and calipers by hand, `moments` (features mode) |
| 15:00–16:30 | Unit 4 lecture + exercise (`regionfilter`); quizzes as wrap-up |

## Open questions

- **Decided (course map / reviews):** units 1 and 2 merged; `shape-features/` moved here
  from the measurements day; PCA is recalled from `pca/` (day 1), not introduced here
  (review-ordering C3 proposed labelling as the PCA owner; the course map gives it to
  `pca/`); object-level matching and F1/AP go to the evaluation day (day 15), not an
  `object-eval/` unit here (review-gaps gap 4 proposed this day; the course map decides).
  Unit 1 is now the densest; if it overruns, run-based labelling and the `unionfind`
  demo's halving mode go to the exercise only.
- **Decided:** `morphgrid` gets a `label` mode on `lib/gridui.ts`; no separate `labelgrid`.
- **Decided:** 3D here is only the small `connparadox` tab and one slide; 3D images as such
  are `volumes3d/` (marching-cubes elective).
- How far into perimeter estimators: edge count and chain code with their bias (core),
  Crofton and optimal weights as one slide (proposal)?
- **Decided:** Hu invariants and higher moments are named only; the moments elective
  teaches them.
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
- Andrew (1979), *Another efficient algorithm for convex hulls in two dimensions*, IPL; Toussaint (1983), *Solving geometric problems with the rotating calipers*, IEEE MELECON; Welzl (1991), *Smallest enclosing disks (balls and ellipsoids)* (moved with `shape-features/`).
- Zahn & Roskies (1972), *Fourier descriptors for plane closed curves*, IEEE Trans. Computers.
