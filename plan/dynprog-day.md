# Plan: a day on dynamic programming in image analysis

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 32 of 51, Part VI "Segmentation as optimisation". Hard prerequisites: Graph cuts (day 27). Soft (see also): Snakes and active shape models (day 30), Morphology (day 13), Watershed (day 15). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VI (snakes → level sets → dynamic programming; Ising/MRF); entry after day 27 (graph cuts). Last day of its chain: after snakes and level sets. Unit order: `dp-paths/`, `polar-boundary/`, `livewire/`, `fastmarching/` (moved from the level set day); `seamcarve/` and `dtw/` optional.

## Goal of the day

After the day a student can:

1. state the **principle of optimality**: an optimal path is made of optimal sub-paths. Write the **Bellman recursion** for a minimal path, fill the **cost table** by hand on a small grid, recover the path by **backtracking**, and compare the cost ($O(nmk)$) with brute force ($k^{n}$ paths);
2. find a **closed boundary** around a cell or nucleus by unwrapping the image to **polar coordinates** and running DP with a smoothness constraint. Explain why closing the path needs extra care, and when the star-shape assumption fails;
3. explain why a plain DP order no longer works on a general graph, and use **Dijkstra's algorithm** (DP where the order of the nodes is found as the algorithm runs) for **live-wire / intelligent scissors** segmentation;
4. explain **fast marching** as Dijkstra with an upwind update, compute one update by hand, and say why its arrival time approximates the Euclidean distance where Dijkstra on a grid does not;
5. (optional) use a minimal monotone path through an energy image for **seam carving**; align two 1D signals with **dynamic time warping**, and see scanline stereo as the same table.

Why it fits this audience: the method needs only sums, minima and a table. The by-hand
part is literally filling in a matrix. Energies come from the gradient magnitude in
*Filtering & convolution*, and the polar unwrapping is the inverse mapping with bilinear
interpolation from *Images & geometric transforms*. The day shows what makes a problem
**exactly** solvable: a chain or an ordering of the nodes. This is the counterpart of
"submodular" on the graph-cut day.

## Prerequisite gap

- **Algorithmic thinking and big-O**: assumed only informally ("simple scripts").
  Unit 1 counts operations explicitly ($3^{n}$ vs $3nm$ with numbers: a 256×256 image
  has about $10^{124}$ monotone paths, more than the ~$10^{80}$ atoms in the universe) rather than relying on
  complexity notation.
- **Graphs** (nodes, edges, weights): defined on the graph-cut day (day 27). Unit 3
  recalls the definition in two lines.
- **Gradient magnitude** (Sobel) and **bilinear interpolation**: covered in `kernels/`
  and `images/` (day 1). They are recalled in one slide each where used.
- **Level set equation and snakes**: the two previous days of this chain of days;
  unit 4 (fast marching) and unit 2 (DP snake) recall them.
- No probability is needed, except in the optional Viterbi remark (pointer only).

## Links to other days

- **Graph cuts** (day 27, `plan/graphcut-day.md`): its unit 1 demo `segchain` already solves a
  two-label 1D chain energy exactly by DP (Viterbi, `demos/src/segchain/dp.ts`). This
  day **reuses** it as the "states per column" example in unit 1 (embedded as is, one
  slide, no new demo) and names the general idea: DP is exact on chains and trees, and
  graph cuts are exact on 2D grids for binary submodular energies. The polar boundary
  in unit 2 is exactly a chain (one radius per angle) and makes the contrast concrete.
- **Distance transform** (owned by morphology, day 13; watershed day 15 recalls it):
  the two-pass chamfer transform is a DP. Each pass is a Bellman recursion over a scan
  order, and one pass is not enough because the shortest path may need to go "against"
  the scan. One slide in unit 1 and a bridge to unit 3 (Dijkstra drops the fixed
  order). It reuses `lib/distance.ts`, and the algorithm is not taught again here.
- **Watershed** (day 15): Dijkstra and flooding use the same priority
  queue (`lib/pqueue.ts`). Live-wire minimises the **sum** of costs along a path, while
  watershed minimises the **maximum** (one slide in unit 3, same picture).
- **Level sets** (`plan/levelsets-day.md`, previous in this chain): fast marching moved
  from there to this day (unit 4) and recalls the level set equation. Live-wire on a
  grid is the discrete minimal path, and geodesic active contours the continuous local
  one.
- **Snakes / active shape models** (`plan/snakes-day.md`, first day of this chain): that
  day owns the parametric snake energy (Kass et al.). This day shows only **DP
  optimisation of a snake** (Amini et al.) as an application in unit 2. The snake
  energy is quoted, not derived, with a pointer to that day.
- **Skeletons** (skeleton day, Part III): length along a centreline between
  two points is a minimal path. The live-wire demo has a "measure along a neurite"
  preset that uses the cost "low on the skeleton" (the synthetic neurite tree from
  `lib/imageops.ts`).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units (`seamcarve/`, `dtw/`).

### 1. `dp-paths/`: The DP principle: minimal paths on a grid

- Picture first: a small cost image (dark = cheap) and the question "the cheapest
  path from the top row to the bottom row, moving one row down at a time, one step
  left, straight or right". Try to guess. Count the paths: $m\cdot 3^{n-1}$.
- **Optimal substructure**: if the best path to the bottom passes through pixel $p$,
  then its part up to $p$ is the best path to $p$. Hence
  $M(i, j) = C(i, j) + \min_{d \in \{-1, 0, 1\}} M(i-1, j+d)$, with $M(0, j) = C(0, j)$.
  Store **where the minimum came from** (a pointer per cell), take the smallest
  $M$ in the last row, and follow the pointers back (**backtracking**).
- Cost: $3nm$ additions and comparisons, against $m\cdot 3^{n-1}$ paths. Memory:
  the table, or only one row if only the cost (not the path) is needed.
- The general recipe: (1) stages and states, (2) the recursion, (3) an order in which
  every right-hand side is already known, (4) backtracking. The **order** is the
  condition that will matter in unit 3.
- Same recursion, other states: `segchain` from the graph-cut day (stage = pixel, state
  = label 0/1, transition cost $\lambda[L_p \ne L_{p+1}]$). This is the Viterbi algorithm.
  The demo is embedded unchanged on one slide.
- Same recursion, other order: the forward pass of the two-pass distance transform is
  $D(i, j) = \min(D(i, j),\ D(i-1, j) + 1,\ D(i, j-1) + 1)$. Why a second pass is
  needed (a path that goes up or left) is left as a cliffhanger for unit 3.
- **Demo `dpgrid`** (new): a grid of 5×6 up to 8×10 integer costs drawn as numbers on
  grey squares. Click a cell to change its cost (±). Buttons: *step* (fills the next
  cell, highlighting its three candidates in the row above, the chosen minimum and the
  pointer arrow), *fill row*, *fill all*, *backtrack* (animates the path from the
  best cell in the last row), *reset*. Toggle *show pointers*. Readout: cells filled,
  operations so far vs the number of paths (exact integer, e.g. "1 458 vs 11 664").
  Moves selectable: $\{\swarrow, \downarrow, \searrow\}$ or $\{\rightarrow, \downarrow\}$
  (top-left to bottom-right; the DTW table of unit 6). Presets: "by-hand exercise grid"
  (identical to the exercise), "greedy fails" (the locally cheapest first step leads
  into an expensive region), "two equal optima" (tie rule shown), "a detour a
  monotone path cannot take" (motivates unit 3). A second mode,
  `data-mode="dijkstra"`, is used in unit 3.
- By hand: (a) fill the 5×5 table for a given cost grid, backtrack, and check that the
  greedy top-down path is worse. (b) Count the paths and the table operations for
  5×5 and for 100×100. (c) Write the recursion for `segchain` with two labels and fill
  it for a 1×5 signal (the graph-cut day's exercise numbers, so the answers can be
  checked against each other).

### 2. `polar-boundary/`: Closed boundaries: DP in polar coordinates

- Picture: a nucleus with a blurred, partly weak boundary and a neighbour touching it.
  A threshold leaks, while level sets and snakes need a starting curve and can get
  stuck.
- Assumption: the object is **star-shaped** around a centre $c$ (every ray from $c$
  crosses the boundary once). Then a boundary is a function $r(\theta)$.
- **Unwrap**: sample the image along $K$ rays at $R$ radii,
  $P(k, r) = I(c + r(\cos\theta_k, \sin\theta_k))$, by bilinear interpolation (inverse
  mapping, as in *Images*). The closed curve becomes a path from left to right through
  the $K \times R$ polar image, one radius per angle.
- Cost: low on the boundary, e.g. minus the **radial derivative** $-\partial_r P$
  (dark inside to bright outside, or its negative). The sign selects the edge
  polarity, which a plain gradient magnitude cannot do.
- Smoothness: hard constraint $|r_{k+1} - r_k| \le \delta$ and/or a soft penalty
  $\alpha\,|r_{k+1} - r_k|$. Recursion
  $M(k, r) = C(k, r) + \min_{|d| \le \delta} \big(M(k-1, r+d) + \alpha |d|\big)$:
  unit 1 turned sideways.
- **Closure**: the path must also satisfy $|r_0 - r_{K-1}| \le \delta$. Three
  options, compared in the demo:
  (a) ignore it: a visible jump at $\theta = 0$;
  (b) exact: fix the start radius $r_0$ and run DP once for each of the $R$ starts,
      keeping the best closed path ($R$ times the work, still a few ms);
  (c) the common heuristic: unroll the angles two or three times, run once, and keep
      the middle copy (usually closes; not guaranteed optimal).
- Where it fails: non-star-shaped objects (a kidney shape seen from a bad centre), a
  wrong centre, and a brighter neighbouring edge on the same ray. Remedies: centre from
  a seed click or from the distance-transform peak (day 13), an annulus limit
  $r_{\min} \le r \le r_{\max}$, or iterate (new centre = centroid of the result).
- Contrast: graph cuts optimise a 2D labelling, level sets descend locally and change
  topology, while polar DP is a **global optimum** within a restricted shape family.
- Application: **DP optimisation of a snake** (Amini, Weymouth & Jain 1990). Each
  snake point may move within a small window. With first-order (elastic) terms the
  energy is a closed chain, so DP gives the best move for all points at once, and this
  is iterated. Polar DP is the special case "move along the ray". The snake energy
  itself is quoted from the snakes day (one slide, pointer only). Second-order
  (bending) terms need pairs of positions as states: name it, $O(n m^3)$.
- **Demo `polarseg`** (new): left, the image with a draggable centre handle, the rays
  (thinned for display) and the resulting contour; middle, the unwrapped polar image
  with the cost as an overlay option and the optimal path; right (or toggle), the
  cumulative table $M$. Sliders: $\delta$ (0–4), $\alpha$, $r_{\min}$/$r_{\max}$
  (shown as an annulus), number of angles $K$. Menus: edge polarity *dark→bright /
  bright→dark / either*; closure *open / unroll ×3 / exact*. Readout: the closure gap
  $|r_0 - r_{K-1}|$ and the path cost. Images: a seeded synthetic nucleus (an ellipse
  with a weak arc, noise, and a touching neighbour; from the nuclei generator in
  `lib/imageops.ts`) and the micrograph `cell256.png` (its nucleus). Presets: "nucleus:
  done", "no smoothness: the path jumps to the neighbour", "open path: jump at 0°",
  "bad centre: not star-shaped", "wrong polarity", "micrograph nucleus". Optional
  toggle *Amini iteration*: points move in a window, with a step button.
- By hand: a 4-angle × 4-radius polar cost table with $\delta = 1$: (a) the best open
  path; (b) the best closed path by trying all four start radii; (c) show that (a)
  does not close. Then: map two given polar samples back to $(x, y)$ (the bilinear
  sampling done once).

### 3. `livewire/`: When there is no order: Dijkstra and live-wire

- Picture: the "detour" preset of `dpgrid`. The best path to a pixel goes up and
  around, so no row-by-row order exists. The same is true of the distance-transform
  second pass from unit 1.
- Graph: pixels are nodes and neighbours (4 or 8) are edges with a cost. The Bellman
  equations $d(v) = \min_{u \sim v} \big(d(u) + w(u, v)\big)$, $d(s) = 0$, still
  hold. The problem is that we do not know in which order to evaluate them.
- **Dijkstra** (1959): with non-negative weights, the unfinished node with the smallest
  tentative $d$ is final. A priority queue gives the order, so this is DP with the
  order discovered on the fly. The result is a shortest-path **tree** from the seed:
  every pixel stores its predecessor (backtracking again).
  Cost $O(N \log N)$ for an image with $N$ pixels.
- Why negative weights break it (one picture). Monotone-path DP was the special case
  where the order is known in advance (a DAG).
- **Live-wire / intelligent scissors** (Mortensen & Barrett 1995; Falcão et al. 1998):
  local cost from (i) gradient magnitude (low on edges), (ii) Laplacian zero crossing,
  (iii) gradient direction (penalise crossing an edge), with weights. Click a seed,
  compute the tree once, and the path to the cursor is then just backtracking, so it
  follows the mouse in real time. Click again to fix an anchor and start a new tree.
  Close the contour on the first anchor.
- Behaviour: short-cutting across weak edges (sums favour short paths), and the
  "path cooling" idea of fixing anchors automatically (named).
- Same tree, other uses: minimal-path **length along a neurite** between two points
  (skeleton link). With cost 1 per step the tree gives the distance transform from a
  point (exact for the graph metric). Watershed minimises the maximum instead of the
  sum (one slide, recall of day 15), and fast marching is the continuous version (unit 4).
- **Demo `dpgrid`** in `data-mode="dijkstra"`: the same grid with 4-neighbour moves in
  every direction. *Step* finalises the smallest node in the queue, which is shown as a
  sorted list beside the grid, and updates its neighbours. Finalised cells are shaded
  in order, and the pointers form the tree. Presets: "the detour" (the same grid as in
  unit 1, now solved), "by-hand exercise grid", "a wall with one gap".
- **Demo `livewire`** (new, main demo): 256×256, the micrograph and the synthetic nuclei
  image, plus the neurite tree. Click to place a seed or anchor. The path to the current
  pointer is drawn live. A draggable **free-end handle** does the same without hover,
  for projectors and touch. *Close*, *undo anchor* and *reset* buttons. Sliders:
  smoothing $\sigma$, weights of the gradient / zero-crossing / direction terms.
  View toggles: *image / cost / distance from seed* (colour map with iso-lines, which
  shows the tree "flooding" outward). Readouts: path length in pixels and the cost.
  Presets: "trace the nucleus with three anchors", "weak edge: the wire short-cuts",
  "no smoothing: noise wins", "the neighbour steals the path", "measure along a
  neurite" (cost low on the skeleton; length readout with 1 / $\sqrt2$ steps, as on
  the skeleton day).
- By hand: Dijkstra on a 4×4 grid with given costs (vertex weights) from a corner:
  the order in which nodes are finalised, the final distances and the tree. Then
  change one cost so that the best path makes a detour, and explain why one
  row-by-row pass gets it wrong.

### 4. `fastmarching/`: From Dijkstra to fast marching: continuous minimal paths

Moved here from the level set day (was its optional unit 5; review-organisation §2.7).
It completes the day's arc: grid DP → polar DP → Dijkstra/live-wire → fast marching.

- Picture: Dijkstra's distance from a seed on a grid is "blocky" (the 4-neighbour
  distance is the city-block metric; 8 neighbours give octagons). We want the true
  Euclidean arrival time of a front, i.e. a circle around the seed on a flat cost.
- **Arrival time** $T(x, y)$ of a front moving outward with speed $F > 0$:
  $|\nabla T|\,F = 1$ (the Eikonal equation), or with cost $P = 1/F$:
  $|\nabla T| = P$. With $P = 1$, $T$ is the Euclidean distance transform (owned by
  morphology, day 13). Recall from the level set day (previous in this chain): the same
  front as $\phi_t + F|\nabla\phi| = 0$ when $F$ never changes sign.
- **Fast marching** (Sethian 1996) = Dijkstra with an **upwind update**: a pixel's value
  is computed from its already-final neighbours by solving a small quadratic
  $(T - T_x)^2 + (T - T_y)^2 = P^2$ (taking the smaller neighbour per axis), instead of
  $\min(d(u) + w)$. Same priority queue, same "smallest tentative is final" argument.
  The quadratic and its one-neighbour fallback go on a math slide.
- Minimal paths: backtrack by **gradient descent on $T$** (continuous, sub-pixel) instead
  of following pointers. Geodesic active contours (level set day) are the local version
  of this global minimal path; live-wire (unit 3) is the discrete one.
- Flooding from seeds with the "maximum" instead of the "sum" is watershed (day 15):
  one slide, recall only.
- **Demo `fastmarch`** (new; replaces the level set plan's `curveflow`
  `data-mode="arrival"`, now built here on `lib/shortestpath.ts`): the same images as
  `livewire`. Click seeds; panels: arrival time as a colour map with iso-lines
  (successive fronts), and the Dijkstra (4/8-neighbour) result for comparison. Cost:
  constant or from the image ($P = 1/g$ with the edge-stopping $g$). Click a target to
  draw the minimal path (gradient descent on $T$) next to the live-wire path. Presets:
  "flat cost: circles, not diamonds", "Dijkstra 4-nbr vs fast marching", "two seeds:
  where fronts meet", "minimal path along a neurite", "edge cost: front slows at the
  nucleus".
- By hand: on a 3×3 neighbourhood with known $T$ above and left of a pixel and
  $P = 1$, solve the upwind quadratic and compare with the Dijkstra value
  $\min(T_x, T_y) + 1$; the case where only one neighbour is known.

### 5. (optional / further reading) `seamcarve/`: Seam carving: content-aware resizing

Optional since the course map (review-organisation §2.7; review-gaps: seam carving is image
editing, not analysis); kept complete so it can be taught as an extra.

- Picture: shrinking an image to 60 % width by scaling (everything gets thin), by
  cropping (content lost at the edges), and by seam carving (the background shrinks,
  the objects keep their shape).
- **Energy** $e(i, j) = |\partial_x I| + |\partial_y I|$ (or the Sobel magnitude from
  *Filtering & convolution*). A **vertical seam** is an 8-connected path with one pixel
  per row. Removing it reduces the width by one and keeps the image rectangular.
- The cheapest seam is exactly the unit 1 problem with $C = e$. Remove it, recompute
  the energy near the seam, and repeat. Horizontal seams: the same on the transposed
  image.
- Why greedy per-row removal fails (jagged rows). Why removing the $k$ cheapest
  **columns** fails (straight cuts through objects). Seams bend around objects.
- Control by the user: **protect** (add a large cost on a painted mask) and **remove**
  an object (a large negative cost, then carve until the mask is gone).
- Limits: when the background is used up, seams must cross objects. Faces and straight
  lines bend. **Forward energy** (Rubinstein et al. 2008) counts the new edges that
  removal creates (one slide with the picture, named, not derived).
- Enlarging (insert the $k$ cheapest seams, duplicating them): one sentence.
- **Demo `seamcarve`** (new, main demo): an image of about 256×192 with three views:
  the image, the energy, and the cumulative cost table $M$ as a colour map with the
  pointers implied by colour. Width slider from 100 % down to 50 %. The removed seams
  are shown on the original, coloured by removal order (a "seam map"), and the next
  seam is overlaid in red. Side-by-side comparison panel: *scale / crop / carve*.
  Brushes: *protect* and *remove*. Toggles: *vertical / horizontal*, energy
  *gradient / Sobel / forward*. Removal order is precomputed once per image/energy/mask
  (stored as an index map), so the slider is instant. Presets: "background shrinks,
  object stays", "too far: the seams cut the object", "protect it", "remove an object",
  "columns instead of seams", "backward vs forward energy". Images: the micrograph
  `cell256.png` (organelles in a cytoplasm background) and a synthetic scene (discs and
  a straight bar on a textured background, seeded). See open questions for a photograph.
- By hand: a 4×5 energy table: find the minimal seam by DP, remove it, write down the
  4×4 result. Then a second table where the cheapest column differs from the cheapest
  seam.

### 6. (optional / further reading) `dtw/`: Aligning signals: dynamic time warping

- Two 1D signals with the same events at different speeds: intensity along a
  **kymograph** row or column (a moving particle or a contraction wave), two intensity
  profiles across similar cells, or two growth curves. The pointwise (lockstep)
  distance is large, although the signals are "the same".
- **Warping path** through the $n \times m$ table of $|a_i - b_j|$, monotone with
  steps $\rightarrow, \downarrow, \searrow$:
  $D(i, j) = |a_i - b_j| + \min\big(D(i-1, j), D(i, j-1), D(i-1, j-1)\big)$.
  This is unit 1 with another step set (`dpgrid` with the $\{\rightarrow, \downarrow\}$
  moves plus the diagonal).
- Constraints: a **Sakoe–Chiba band** $|i - j| \le w$ (speed and robustness against
  pathological warping), and slope limits. DTW is not a metric (one example).
- **Scanline stereo** (Ohta & Kanade 1985; one or two slides): match a left and a right
  scanline with the same table. A diagonal step is a match, a horizontal or vertical
  step is an occlusion with a fixed cost. Disparity = $i - j$ along the path.
- **Demo `dtw`** (new): two signals (synthetic, seeded: a pulse train and a time-warped,
  noisy copy; or two rows/columns picked from a synthetic kymograph shown as an image,
  where clicking selects the rows). Panels: both signals with matching lines between
  aligned samples, and the cost table as a heat map with the warping path and the band.
  Sliders: band width $w$, noise, warp strength. Readout: lockstep distance vs DTW
  distance. `data-mode="stereo"`: the left and right scanlines of a seeded
  random-dot stereogram (a raised square), occlusion-cost slider, recovered disparity
  vs truth below. Presets: "pure shift", "local stretch", "band too narrow", "noise is
  over-warped", "stereo: occlusion at the step".
- By hand: DTW table for $a = (0, 2, 2, 0)$, $b = (0, 0, 2, 0)$; the path and the
  distance; compare with the lockstep distance.

## Shared engineering

- `demos/src/lib/dp.ts` (new): one **monotone-path DP** over a cost matrix with a
  configurable step set (offsets plus per-step penalty, e.g. $\{-\delta..\delta\}$ with
  $\alpha|d|$, or $\{\rightarrow, \downarrow, \searrow\}$), optional band, and closure
  mode (*open / unroll / exact over all starts*). It returns the cumulative table,
  the pointers and the path. An instrumented variant records each cell's candidates and
  choice for `dpgrid`. Used by `dpgrid`, `seamcarve`, `polarseg`, `dtw`. This day owns
  `lib/dp.ts`, which also holds chain Viterbi / forward–backward (used by the Ising
  day; no forward–backward in `lib/smooth.ts`). `segchain/dp.ts` may keep its file.
- `demos/src/lib/shortestpath.ts` (new): Dijkstra on an implicit grid graph (4/8
  neighbours from `lib/graph.ts` `gridEdges`, or computed on the fly for 256×256),
  with vertex or edge costs, returning distances and the predecessor array. An
  instrumented variant (pop order, queue snapshots) is used for `dpgrid` dijkstra mode.
  It uses `lib/pqueue.ts` (owner: watershed day; heap + bucket + FIFO), not a new
  queue. Fast marching (unit 4) adds the upwind update as a variant in the same file.
- `demos/src/lib/polar.ts` (owner: this day): polar / log-polar unwrap and
  back-projection, used by radon, FFT registration, moments and measurements.
- `demos/src/lib/imageops.ts` (filters owned by `kernels/`): Gaussian, Sobel, Laplacian
  (zero crossings for live-wire), bilinear sampling, `loadGrey()` for `cell256.png`.
  Synthetic generators (nuclei, seam-carving scene, kymograph) go in `lib/synth.ts`
  (owner: thresholding day); the neurite tree in `lib/trees.ts`.
- `lib/plot.ts` (owner: morphology day) for the `dtw` signal panel. Sequential colour
  map for the cumulative tables from `lib/colour.ts` (owner: colour day). Grid
  step-through UI for `dpgrid` from `lib/gridui.ts` (owner: morphology day).
- Speed (TypeScript first; per `demos/README.md` rule of 30 ms):
  - polar DP: $K = 360$, $R = 100$, $\delta = 3$ gives about 0.25 M operations, and
    exact closure ×100 about 25 M, both a few ms;
  - live-wire: one Dijkstra on 65 k pixels with a heap, estimated 10–20 ms per click,
    while mouse moves only backtrack;
  - seam carving: removing 128 seams from 256×192, each a DP plus a local energy update,
    is estimated at 50–150 ms. This is precomputed once on image/mask/energy change
    (behind a short "computing…" note), so the slider only reads an index map. Measure.
    Only if it exceeds ~300 ms on a laptop, move the seam DP to Rust/Wasm per the README
    (TS reference kept and tested).
- Tests (`demos/test/dynprog-test.ts`, run by the shared `make test`): DP path cost
  equals brute-force enumeration on random grids up to 5×6 for each step set. Exact
  closure equals brute force over all closed paths on 5×4. Dijkstra equals Bellman–Ford
  on random small grids, and equals the monotone DP when only monotone moves are
  allowed. Fast marching on a flat cost is within 1 px of the exact Euclidean distance
  at radius 50. Each removed seam is 8-connected with one pixel per row. DTW against brute
  force on sequences of length ≤ 6.
- Follow `demos/README.md`: fixed seeds, presets + reset, nothing hover-only
  (the free-end handle in `livewire`), keyboard back to reveal.js, fits 1150×560 on a
  slide, image credits shown (the micrograph credit as in `fourier2d`).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, DP table by hand, `dpgrid`, `segchain` recap |
| 10:45–12:15 | Unit 2 lecture, closed polar path by hand, `polarseg` |
| 13:15–14:45 | Unit 3 lecture, Dijkstra by hand, `dpgrid` (dijkstra), `livewire` |
| 15:00–16:30 | Unit 4 lecture, fast-marching update by hand, `fastmarch`; quizzes as wrap-up |

Units 5 (seam carving) and 6 (DTW / stereo) are further reading with their demos in the
exercise; DTW is the better swap-in for an analysis cohort (kymographs).

## Open questions

- **Decided (course map / review-organisation §2.7):** four core units ending with fast
  marching; seam carving and DTW optional.
- Seam carving on a micrograph is less convincing than on a photograph (people,
  landscape). Should we add a freely licensed photograph (CC0 / CC BY, credited), or
  keep to microscopy and synthetic scenes? (Seam carving is now optional, so this only
  matters if it is taught.)
- Polar boundary: is the exact closure (all starts) worth teaching, or is the unroll
  heuristic plus "it usually works" enough? Recommendation: teach both, exact in the
  by-hand exercise because it is just a loop.
- Amini DP snakes: one slide in unit 2 (current plan), or a full exercise? The snakes
  day (first in this chain) has covered the energy, so a short exercise is possible.
- A real kymograph (and its licence) for `dtw`, or synthetic only?
- Scanline stereo: keep it in the optional unit, or drop it (it needs no new image, so
  it is cheap)?
- Programming language for the code parts is still open (see README).

## References

- Bellman (1957), *Dynamic Programming*, Princeton University Press.
- Felzenszwalb & Zabih (2011), *Dynamic programming and graph algorithms in computer vision*, PAMI (survey; the best single reading for the day).
- Avidan & Shamir (2007), *Seam carving for content-aware image resizing*, SIGGRAPH.
- Rubinstein, Shamir & Avidan (2008), *Improved seam carving for video retargeting*, SIGGRAPH (forward energy).
- Amini, Weymouth & Jain (1990), *Using dynamic programming for solving variational problems in vision*, PAMI.
- Appleton & Sun (2003), *Circular shortest paths by branch and bound*, Pattern Recognition (closure in polar DP).
- Baggett et al. (2005), *Whole cell segmentation in solid tissue sections*, Cytometry A (DP boundary finding for cells).
- Dijkstra (1959), *A note on two problems in connexion with graphs*, Numerische Mathematik.
- Mortensen & Barrett (1995), *Intelligent scissors for image composition*, SIGGRAPH.
- Falcão, Udupa, Samarasekera, Sharma, Hirsch & Lotufo (1998), *User-steered image segmentation paradigms: live wire and live lane*, Graphical Models and Image Processing.
- Sakoe & Chiba (1978), *Dynamic programming algorithm optimization for spoken word recognition*, IEEE Trans. ASSP.
- Ohta & Kanade (1985), *Stereo by intra- and inter-scanline search using dynamic programming*, PAMI.
- Viterbi (1967), *Error bounds for convolutional codes and an asymptotically optimum decoding algorithm*, IEEE Trans. Information Theory.
- Borgefors (1986), *Distance transformations in digital images*, CVGIP (the two-pass DP).
- Sethian (1996), *A fast marching level set method for monotonically advancing fronts*, PNAS.
- Sethian (1999), *Level Set Methods and Fast Marching Methods*, Cambridge University Press.
