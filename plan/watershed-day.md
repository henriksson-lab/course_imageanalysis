# Plan: a day on watershed segmentation

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. read a greyscale image as a **landscape**, and define regional minima, **catchment basins** and **watershed lines** by flooding (and by a drop of water running downhill);
2. run **priority-queue flooding** (Meyer's algorithm) by hand on a small grid from given markers, including the tie-breaking on plateaus, and state its cost;
3. choose the **landscape** for a task: gradient magnitude for boundaries between regions, the inverted **distance transform** for touching blobs;
4. explain **oversegmentation** and fix it with smoothing, **h-minima** (minima shallower than $h$ are filled) and **marker-controlled** watershed;
5. compute a **distance transform** by hand (two-pass chamfer) and predict when two touching discs are split, from the depth of the neck between their peaks.

Why it fits this audience: the core is one algorithm on a grid plus one picture (a
landscape flooded with water), so it can be traced fully by hand. The calculus link is
explicit: minima, saddles and steepest descent of a height function; gradient
magnitude comes straight from *Filtering & convolution* (Sobel); the distance transform
is a small optimisation problem with a geometric answer. Watershed is still the default
way to split touching nuclei in microscopy, so the payoff is concrete.

## Prerequisite gap

- **Gradient magnitude** (Sobel) and **Gaussian smoothing** are known from
  `kernels/`. Recall them in one slide each; nothing new.
- **Thresholding** (threshold → binary mask) is taught on the thresholding day
  (`plan/thresholding-day.md`); **connected components** with 4- vs 8-connectivity
  (needed for regional minima and for counting objects) on the labelling day
  (`plan/labelling-day.md`). Unit 1 recalls each in one line.
- **Mathematical morphology** gets its own day (`plan/morphology-day.md`; decided
  2026-11-12). h-minima is still introduced here by its effect ("fill every basin up by
  $h$; basins that overflow merge"), with a pointer to morphological reconstruction on
  the morphology day for the definition.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `ws-landscape/`: Images as landscapes, flooding in 1D

- Recall (one line each): threshold → binary mask (thresholding day, `plan/thresholding-day.md`); connected components, 4/8 (labelling day, `plan/labelling-day.md`). Where it fails: two touching cells become one component; the brightness varies over the image so no single threshold works (remedies: thresholding day, unit 3 `local-threshold/`).
- An image as a height map $f(x, y)$. **Regional minimum**: a connected plateau whose neighbours are all higher. **Catchment basin**: the points from which a drop of water runs downhill (steepest descent, $-\nabla f$) into that minimum. **Watershed line**: where the drop could go either way (ridges, saddles).
- The equivalent **flooding** picture: pierce a hole at every minimum and lower the landscape into water. Where water from two basins would meet, build a dam. The dams are the watershed lines; each basin is one region.
- What to flood: intensity itself (dark nuclei on a bright background are already basins), or the **gradient magnitude** $\|\nabla f\|$: flat regions are valleys, edges are ridges, so the dams sit on the edges. 1D example: a step profile, its $|f'|$, two valleys, one ridge at the edge.
- Every regional minimum makes a basin: noise adds minima, so noise adds regions (**oversegmentation**), the main problem of the day. Two cures previewed: **h-minima** and **markers**.
- **Demo `flood1d`** (new): a 1D profile of ~60 samples drawn as a landscape in SVG, water in blue per basin (one colour per basin). A *water level* slider floods from below; dams appear as vertical bars where basins meet, with their height. Profile from presets or edited by dragging control points; option *flood $|f'|$ instead of $f$* draws the derivative below the input and floods that. Slider $h$: shallow minima are filled (shown as a hatched fill-in) and their basins merge. Toggle *markers*: click samples to make them the only sources; other minima are then just "lakes" that get drowned. Readout: number of basins. Presets: "two cells" (a profile across two dark nuclei), "noisy: one basin per wiggle", "h removes the noise", "edges: flood the derivative", "plateau" (a flat valley: where does the dam go?).
- By hand: a 1D signal of 12 integers. Mark the regional minima, flood level by level, write down the basin of each sample and the dam positions. Repeat with $h = 2$: which minima survive?

### 2. `ws-flooding/`: The flooding algorithm on a grid

- From 1D to a pixel grid: neighbours (4 or 8), a plateau is a connected set of equal pixels, a regional minimum is found by connected components of plateaus.
- **Markers**: labelled seed pixels (by default one label per regional minimum).
- **Meyer's algorithm** (priority-queue flooding):
  1. put every unlabelled neighbour of a marker into a priority queue with priority $f(p)$;
  2. pop the lowest pixel $p$ (ties: first in, first out);
  3. if all labelled neighbours of $p$ carry the same label, give $p$ that label; otherwise $p$ is on the watershed line (or, in the variant without lines, takes the label of the neighbour that reached it first);
  4. push the unlabelled, not yet queued neighbours of $p$; repeat until the queue is empty.
- Why it floods correctly: the queue always releases the lowest pixel on the current shoreline, so the water rises evenly in all basins at once. The FIFO rule spreads water across a plateau like a wavefront, so a dam on a plateau falls in its middle.
- Cost: $n$ pushes and pops, $O(n \log n)$ with a binary heap, $O(n)$ with a bucket queue for 8-bit images (256 buckets). The 1991 alternative (Vincent & Soille: sort all pixels, flood level by level) in one slide.
- Two flavours of output: a one-pixel **watershed line** vs **every pixel labelled** (what most libraries return by default). 4- vs 8-connectivity: a 4-connected line needs an 8-connected background to be closed and vice versa.
- **Demo `floodgrid`** (new): an 8×8 grid (up to 12×12) of integers drawn as cells with the number in each, shaded as heights. Click a cell to set its value (± buttons) or to place/remove a marker (label colour chosen from a small palette); *all regional minima as markers* button. *Step* pops one pixel: the popped cell is outlined, its labelled neighbours highlighted, the decision written out ("all neighbours are A → A" / "A and B meet → line"). The **queue** is shown next to the grid as a sorted list of (value, arrival order, cell), so students can check their own trace. Buttons *step*, *step to next level*, *run to end*, *reset*. Options: 4/8-neighbours, watershed line on/off, tie-breaking FIFO vs LIFO (to show why FIFO matters on plateaus). Presets: "two basins", "plateau between two markers", "every minimum is a marker (oversegmentation in miniature)", "a marker that is not a minimum".
- By hand: a 5×5 grid with two markers. Trace the queue for the first eight pops (value, arrival order), then finish the labelling and mark the line pixels. Compare 4- with 8-neighbours on the same grid.

### 3. `ws-gradient/`: Watershed segmentation of a real image

- Landscape = $\|\nabla (G_\sigma * f)\|$ (Sobel after a Gaussian, from `kernels/`). Basins are regions of nearly constant intensity; dams sit on edges.
- On the raw gradient every speck of noise is a minimum: hundreds of regions. Three fixes, each with its trade-off:
  - **smooth more** (larger $\sigma$): fewer minima, but edges move and thin structures blur away;
  - **h-minima**: fill every basin up by $h$; minima with **depth** (dynamic) below $h$ disappear. One parameter with a clear meaning in grey levels;
  - **marker-controlled watershed**: decide the markers first (user clicks, a threshold, peaks of another image) and impose them as the only minima. The flooding then only decides *where* the boundaries go, not *how many* regions there are.
- Region merging after oversegmentation (merge basins across low dams) in one slide, as the general idea behind hierarchical watersheds.
- Leakage: a gap in an edge lets water through: where watershed fails and graph cuts or level sets do better (links only).
- **Demo `watershed`** (new, main demo): 256×256 input (the micrograph `cell256.png`, plus a synthetic seeded test image of a few blobs with noise and one edge with a gap). Three panels: input; landscape (choice: gradient magnitude, intensity, inverted intensity) with minima as dots; result as coloured regions or boundaries over the input. Controls: $\sigma$, $h$, noise; markers: *all minima* / *h-minima* / *clicked* (click to add a marker, shift-click to give it a new label, a *background marker* brush). A *flood level* slider replays the flooding (pixels with $f \le$ level coloured) so the basins visibly grow and meet. Readout: number of regions, number of minima, run time. Presets: "raw gradient: oversegmentation", "smoothing alone", "h-minima", "two clicks: marker-controlled", "edge with a gap: leakage".
- By hand: a 6×6 image of two flat regions with a noisy pixel. Compute the gradient magnitude with a simple difference kernel, find its minima, then apply $h = 1$ and say how many regions remain.

### 4. `ws-distance/`: Splitting touching objects with the distance transform

- The classic microscopy problem: thresholding nuclei gives one blob for two touching nuclei. The intensity has no edge between them, so the gradient does not help.
- **Distance transform** $D(p) = \min_{q \notin \text{object}} \|p - q\|$: zero outside, highest in the middle of each object. Two touching discs → two peaks with a **saddle** at the neck. Watershed of $-D$, restricted to the mask, cuts at the neck.
- Computing $D$ by hand: city-block distance with a **two-pass** algorithm (forward pass from the top-left with the upper/left neighbours, backward pass from the bottom-right). Chessboard and **chamfer 3-4** weights as better approximations; exact Euclidean in linear time (Felzenszwalb & Huttenlocher) named, not derived. Comparison picture: a disc under the three metrics.
- When are two discs split? Discs of radius $r$ with centres $d$ apart: peaks of height $r$, the neck at height $\sqrt{r^2 - d^2/4}$ (the distance from the midpoint to where the circles cross). With h-maxima markers they split iff the neck depth $r - \sqrt{r^2 - d^2/4} > h$. Worked example: $r = 10$, $d = 12$ gives depth 2.
- Failure modes: an elongated or bean-shaped cell has a ridge with several small peaks (oversplitting, fixed by $h$); strongly overlapping cells have too shallow a neck (undersplitting); holes in the mask create false peaks (fill holes first). Combining both landscapes: markers from peaks of $D$, flooding on the gradient.
- **Demo `distws`** (new): a binary mask built from discs and ellipses that can be dragged (centre handle) and resized (rim handle), or from the thresholded synthetic nuclei image (seeded, ~15 touching nuclei with noise, threshold slider). Panels: mask; distance transform as colour with iso-lines and the peaks marked; result with each object in its own colour. Controls: metric (city-block, chessboard, chamfer 3-4, Euclidean), $h$, *flood $-D$ / flood the gradient with markers from $D$*. With exactly two discs, a profile along the line between the centres shows the two peaks, the neck and the depth versus $h$, next to the formula. Presets: "two discs: split", "move them together until they merge", "an ellipse is oversplit", "three in a row", "noisy mask with holes", "nuclei image".
- By hand: city-block distance transform of a 7×7 mask of two overlapping squares, both passes. Then for $r = 10$ and $h = 1$, the smallest $d$ that still gives two objects.

### 5. (optional / further reading) `ws-graphs/`: Watershed on graphs and links to other days

- The pixel grid as a graph with edge weights (e.g. $\max(f_p, f_q)$ or $|f_p - f_q|$). Seeded watershed = a **minimum spanning forest** rooted at the markers; Meyer flooding is Prim's algorithm run from all markers at once (watershed cuts, Cousty et al.).
- Same graph, same seeds as the graph-cut day: graph cuts minimise the sum of cut weights (smooth, shrinking bias); watershed minimises the largest weight along each path (fast, no shrinking, but leaks through a single weak spot). Power watershed unifies both in one sentence.
- Links only: **level sets** (a front that grows from seeds with a speed set by the image; fast marching uses the same priority queue as Meyer's algorithm, on arrival times instead of heights); **marching cubes** (watershed labels of a 3D nuclei stack turned into one surface mesh per nucleus; iso-lines of $D$ in `distws` are marching squares).
- Modern pipelines: networks that predict a distance map or boundary probability, followed by a watershed (StarDist/Cellpose-style post-processing) in one slide.
- **Demo**: reuse `floodgrid` with `data-mode="msf"`, drawing edge weights between cells and growing the forest one edge at a time; and if the graph-cut day is built, reuse its `graphcut` demo with the same seeds as `watershed` for a side-by-side.

## Shared engineering

- `demos/src/lib/pqueue.ts`: a binary heap with a FIFO tie-break (priority, then insertion counter), plus a bucket queue for integer heights. Also usable by a fast-marching demo on the level-set day.
- `demos/src/lib/watershed.ts`: neighbour iteration (4/8), regional minima by connected components of plateaus, h-minima (reconstruction by erosion), Meyer flooding with markers, with/without watershed line. One instrumented version that records every pop (pixel, value, queue snapshot, decision) for `floodgrid` and `flood1d`, and one fast version for 256×256 images. The 1D demo calls the same code on a 1×n image.
- `demos/src/lib/distance.ts`: two-pass chamfer (city-block, chessboard, 3-4) and exact Euclidean (Felzenszwalb–Huttenlocher).
- `demos/src/lib/imageops.ts`: Gaussian blur and Sobel magnitude on Float32 arrays, loading `cell256.png` into a grey array, a seeded synthetic "touching nuclei" generator, and a label-to-colour map (seeded palette; neighbouring labels kept distinct). Move the filtering code from `kernel3x3/conv.ts` here rather than duplicating it, if the kernel demo can import it unchanged.
- Coordinate with the graph-cut plan's `lib/graph.ts` (grid neighbourhoods), so `ws-graphs` can share it.
- Tests (same `make test` as proposed in the graph-cut plan): on random small integer images, check that every region contains exactly one marker and is connected, that the result does not depend on the scan order except on plateaus, that h-minima equals a brute-force "fill by h" in 1D, and that the distance transforms match brute force.
- Images: `demos/src/fourier2d/cell256.png` (credit as in `fourier2d`); synthetic images generated in TS with fixed seeds (no files).
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide. 256×256 Meyer flooding runs in a few ms, so every slider can update live.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, 1D flooding by hand, `flood1d` |
| 10:45–12:15 | Unit 2 lecture, queue trace by hand, then `floodgrid` |
| 13:15–14:45 | Unit 3 lecture + exercise (`watershed`) |
| 15:00–16:30 | Unit 4 lecture, distance transform by hand, `distws`; quizzes as wrap-up |

## Open questions

- How long is a "day" (same question as for the graph-cut day)? Units 1 and 2 could be merged into one unit if time is short, keeping the 1D demo only in the lecture.
- Which day comes first: graph cuts or watershed? It decides in which day the "seeded segmentation on the pixel graph" comparison lives. (Thresholding and connected components are now recalled here in any case: they have their own days.)
- A real image of many touching nuclei would make unit 4 much stronger than `cell256.png` (a single cell in TEM). Candidates: a fluorescence nuclei image from the Broad Bioimage Benchmark Collection; licence and credit to be checked. Otherwise unit 4 uses only the synthetic nuclei.
- ~~Morphology as its own unit or a black box?~~ Decided: its own day (see above).
- Watershed line or full labelling as the default output in the demos? Proposal: full labelling with boundaries drawn, the line as an option in `floodgrid` only.
- Programming language for the code parts is still open (see README).

## References

- Beucher & Lantuéjoul (1979), *Use of watersheds in contour detection*, Workshop on image processing.
- Vincent & Soille (1991), *Watersheds in digital spaces: an efficient algorithm based on immersion simulations*, PAMI.
- Meyer (1994), *Topographic distance and watershed lines*, Signal Processing.
- Beucher & Meyer (1993), *The morphological approach to segmentation: the watershed transformation*, in Mathematical Morphology in Image Processing.
- Soille (2003), *Morphological Image Analysis*, Springer (h-minima, reconstruction, distance transforms).
- Borgefors (1986), *Distance transformations in digital images*, CVGIP (chamfer).
- Felzenszwalb & Huttenlocher (2012), *Distance transforms of sampled functions*, Theory of Computing.
- Cousty, Bertrand, Najman & Couprie (2009), *Watershed cuts: minimum spanning forests and the drop of water principle*, PAMI.
- Couprie, Grady, Najman & Talbot (2011), *Power watershed: a unifying graph-based optimization framework*, PAMI.
- Malpica et al. (1997), *Applying watershed algorithms to the segmentation of clustered nuclei*, Cytometry.
