# Plan: a day on the Hough transform (lines, circles, generalised shapes)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** elective, group "Detection" (Hough, integral images); entry after core day 2 (edges, blobs and ridges). Canny has moved to day 2 (`canny/`); this day starts from its edge points.

## Goal of the day

After the day a student can:

1. *(recalled from day 2)* start from Canny's **edge points with a direction** and say why the Hough transform wants thin edges with a direction;
2. explain the **duality** behind the Hough transform: a point in the image is a curve in parameter space, and the curves of collinear points meet in one point; find the common $(\rho, \theta)$ of three points by hand;
3. explain why slope–intercept fails and the **normal parameterisation** $\rho = x\cos\theta + y\sin\theta$ works, and run **voting and peak finding** in an accumulator, choosing its resolution and threshold and predicting the effect of noise and clutter;
4. detect **circles** (3D parameter space $(a, b, r)$, 2D with known radius) and use the **gradient direction** to cut the number of votes;
5. extend the idea to arbitrary shapes with the **generalised Hough transform** (R-table; rotation and scale as extra dimensions);
6. fit a model with outliers by **RANSAC**, compute the number of iterations needed, and compare voting with hypothesise-and-verify.

Why it fits this audience: everything is a change of variables, a sum of votes, and a
maximum; the duality is a line in one space becoming a point in another, which is
linear algebra they already know. The step from lines to circles to arbitrary shapes
is the same idea with more parameters, and makes the cost of each extra dimension
concrete. RANSAC needs only basic probability.

## Prerequisite gap

- **Canny, gradient direction, non-maximum suppression, hysteresis** are owned by the edges day
  (day 2, `canny/`; decided in `plan/coursemap.md`). Unit 1 opens with a **one-slide recall**
  ("smooth, gradient, NMS, hysteresis → a list of edge points $(x_i, y_i, \theta_i)$") and
  embeds day 2's `canny` demo as the edge source. Review-ordering proposed keeping Canny here
  (no core day needed it); the course map overrules this, since the new edges day owns it. The
  freed hour splits the former dense unit 3 into `hough-circles/` and `ght/`, as this plan
  proposed.
- **Hysteresis** is introduced on the edges day (day 2, inside Canny) and generalised as a
  segmentation tool on the thresholding day (day 6); here it only appears inside the Canny recall.
- **Least squares** (normal equations for a line fit) is assumed from linear algebra; recalled in
  one line in unit 4, where it refits the RANSAC inliers.

## Overlaps with other plans and units

- **Edges day** (day 2): owns Sobel/gradient direction, Canny, NMS and scale space. This day uses
  them; nothing re-derived. NMS in parameter space (unit 1) is named as "the same idea as day 2's
  NMS across the edge".
- **Thresholding** (day 6): hysteresis as a general segmentation tool (first met on day 2 in
  Canny). Mentioned here only inside the Canny recall.
- **Morphology** (day 8): regional maxima and h-maxima. Peak finding in the accumulator is the
  same problem (local maxima above a threshold, in a window); one line in unit 1 names the link.
- **`fourier/`** (day 1): the spectrum shows orientation as an angle. The one-slide Radon link
  (section 5) relates the Hough transform for lines to the **Radon transform**; the Radon
  elective owns it.
- **Moments** (elective, Objects): orientation from second moments (labelling day 9,
  `regionprops/`) is the alternative for deskewing one object; moment-based template matching is
  the alternative to the generalised Hough transform. One slide in unit 3 names both.
- **Snakes / active shape models** (elective, Curves and energies): *local* refinements that need
  an initial pose; the generalised Hough transform is a *global* detector that can supply it.
  Pose uses the same similarity transform as Procrustes there. One slide in unit 3.
- **Dynamic programming** (elective): polar DP and live-wire find *free-form* boundaries given a
  centre or anchors; Hough finds *parametric* shapes anywhere. A circle Hough detection can give
  polar DP its centre (one preset in `houghcircle` exports a centre; one slide in unit 2).
- **Labelling / measurements** (days 9, 11): equivalent diameter, circularity and the enclosing /
  inscribed circles are measured *after* segmentation; the Hough radius is a *detection*
  estimate. The least-squares circle fit used for the RANSAC refit (unit 4) is taught here.
- **Watershed** (day 10) and **morphology** (day 8, distance transform): distance-transform
  peaks give centres of touching discs from a binary mask; circle Hough gives them from edges.
  Named as alternatives on one slide (unit 2).
- **Skeletons** (elective, Objects) and **edges day `hessian/`**: vesselness finds curved
  filaments; Hough lines find straight ones (microtubules, stress fibres). One line each way.
- **Graphics 3D** (elective): homogeneous coordinates. The point–line duality of the projective
  plane is the cleanest statement of the Hough duality; math-layer slide in unit 1. The 3D circle
  accumulator can optionally be shown with `lib/gl3d.ts`.
- **Random forests** (day 14) and **modern deep learning** (day 18): Hough forests (learned
  patches vote for an object centre) and centre-voting / centre-heatmap detectors are a one-line
  "where this leads" in unit 3. Detection metrics (IoU matching, AP) belong to the evaluation day
  (day 16).
- **Registration**: RANSAC's main application (fitting a transform to matched points) belongs to
  the planned Motion elective (after day 12); one pointer line in unit 4. RANSAC stays core here.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus a one-slide link to the Radon
transform. (Unit 1 `canny/` moved to the edges day; the former unit 3 `hough-shapes/` is split
into `hough-circles/` and `ght/`.)

### 1. `hough-lines/`: Duality, the accumulator and line detection

- **Recall (one slide, day 2 `canny/`)**: smooth, gradient, NMS, hysteresis turn an image into a
  list of thin edge points $(x_i, y_i)$, each with a direction $\theta_i$ and a strength. The
  `canny` demo of day 2 is the edge source for every demo of this day.
- Concept layer, picture first: through one point pass infinitely many lines; each is described by two numbers, so the point "is" a curve in the space of line parameters. Two points: two curves, they cross at the one line through both. Three collinear points: three curves through a common point. **Detecting a line = finding where many curves meet.** Demo `houghdual` before any formula.
- **Slope–intercept fails**: $y = mx + c$ gives the nice picture (a point becomes a straight line in $(m, c)$) but vertical lines have $m = \infty$ and $m$ is unbounded, so it cannot be a finite grid. Shown in the demo by dragging three points towards a vertical line.
- **Normal parameterisation**: a line is the set of points whose projection onto the unit normal $(\cos\theta, \sin\theta)$ equals $\rho$: $\rho = x\cos\theta + y\sin\theta$, $\theta \in [0, \pi)$, $|\rho| \le$ the image diagonal. Bounded. A point becomes a **sinusoid** in $(\theta, \rho)$. Small worked example: points $(2,0), (1,1), (0,2)$ at $\theta = 0°, 45°, 90°, 135°$; only $\theta = 45°$ gives the same $\rho = \sqrt2$ for all three.
- **The accumulator**: a grid of bins $(\theta_j, \rho_k)$, every edge point adds one vote to each bin its sinusoid passes through; lines are **peaks**. Gaps in a line do not matter (votes just add), which is the strength of the method.
- **Peak finding**: threshold on the vote count (≈ minimum line length in pixels) plus **non-maximum suppression in parameter space** (keep local maxima in a window; the same idea as NMS across the edge on day 2, and as regional maxima on day 8). Without NMS one line gives a cluster of detections.
- **Resolution vs precision**: bins too coarse merge nearby lines and give a poor angle; too fine, votes of one noisy line spread over many bins and the peak vanishes. Rule of thumb in pictures: $\Delta\rho \approx 1$ px, $\Delta\theta$ such that the far end of the line moves about one pixel. Noise in the edge positions widens the peak; clutter (texture, other objects) raises the floor of the accumulator.
- **Use the gradient direction**: the gradient *is* the normal of the edge, so $\theta_i$ is known from Canny (day 2). Vote only near $\theta_i$ (one bin, or a small window for its uncertainty). Votes per point: from $N_\theta$ (e.g. 180) to about 1–5; clutter votes no longer pile up by chance.
- **Lines vs segments**: a peak is an infinite line. Endpoints: walk along the line, collect the edge points within a pixel of it, split at gaps longer than $g$, keep pieces longer than $L_\text{min}$.
- Applications on one slide each, with the demo preset: lane / road edges and building edges in a photograph; **deskewing** a scanned page or a tilted micrograph (dominant $\theta$ = the peak of the column sums of the accumulator); **filaments** (microtubules, stress fibres: straight segments in clutter).
- **The math**: the Hough transform as a sum $H(\theta, \rho) = \sum_i [\,|x_i\cos\theta + y_i\sin\theta - \rho| < \Delta\rho/2\,]$; the sinusoid $\rho = A\cos(\theta - \varphi)$ with $A = \sqrt{x^2 + y^2}$, $\varphi = \operatorname{atan2}(y, x)$ (amplitude and phase from the point's polar coordinates); two points' sinusoids intersect at the normal of the line through them (solve the $2\times2$ system); cost $O(N_\text{edge} N_\theta)$ vs $O(N_\text{edge})$ with the gradient; angle precision of a segment of length $L$ with endpoint noise $\sigma$ is about $\sigma\sqrt{2}/L$ radians, which sets a useful $\Delta\theta$; projective duality: in homogeneous coordinates a line $\mathbf{l}$ and a point $\mathbf{x}$ satisfy $\mathbf{l}^\top\mathbf{x} = 0$, symmetric in both (pointer to the 3D graphics elective).
- **Demo `houghdual`** (new): left, an image plane with up to 8 draggable points (click to add, right-click or a button to remove); right, the $(\theta, \rho)$ plane with one coloured sinusoid per point. Hovering is not needed: click a point in parameter space to draw that line in the image; the intersection of all curves is marked when the points are collinear. Toggle *parameterisation: slope–intercept / normal* (in slope–intercept the curves are straight lines and run off the panel as the points approach a vertical line). Toggle *accumulator* overlays the binned votes with a $\Delta\theta, \Delta\rho$ slider. Presets: "three collinear points", "two points: one line", "towards vertical: slope–intercept breaks", "four points, two lines: two crossings", "noisy collinear: the curves almost meet", "coarse vs fine bins".
- **Demo `houghlines`** (new, main demo of the unit): three panels: image with edge points (from day 2's Canny via `lib/edges.ts`, with its $\sigma$ and thresholds collapsed into one *edges* setting), accumulator (heat map, detected peaks circled), detected lines or segments drawn on the image. Sliders: $\Delta\theta$, $\Delta\rho$, vote threshold, NMS window, noise, clutter density; toggles *gradient voting* (with window width), *segments* (gap $g$, $L_\text{min}$). Readouts: number of votes cast, time, detections. Images: synthetic (a few lines, a grid, a road with lane markings, a rotated striped "page", seeded straight filaments in clutter) and `cell256.png`. Presets: "clean lines", "no NMS: clusters", "bins too fine: peak dissolves", "bins too coarse: lines merge", "clutter raises the floor", "gradient voting: 100× fewer votes", "segments with gaps", "deskew the page", "filaments".
- By hand: (a) the three points above: sinusoid values at four angles, fill a 4×5 accumulator, read the line; (b) predict: a dashed line with half its pixels missing — what happens to the peak height and position? (c) two parallel lines 3 px apart with $\Delta\rho = 4$: what is detected? *(math)*: (d) derive the intersection of the sinusoids of $(x_1, y_1)$ and $(x_2, y_2)$; (e) amplitude–phase form of the sinusoid; (f) votes and memory for a 512×512 image with 10 000 edge points, $\Delta\theta = 0.5°$, with and without gradient voting.

### 2. `hough-circles/`: Circles (and ellipses briefly)

- Concept layer. **Circles**: $(x - a)^2 + (y - b)^2 = r^2$, three parameters. With a **known radius**, every edge point votes for all centres at distance $r$ from it: a circle of votes in the $(a, b)$ plane, and the circles of votes of all points on a true circle meet at its centre. A 2D accumulator that looks like the image. Picture and demo first.
- **Unknown radius**: one $(a, b)$ slice per radius, a 3D accumulator (a cone of votes per point). Memory and time grow with every parameter: e.g. 256×256×40 radii ≈ 2.6 M bins.
- **Gradient direction** again: the centre lies along the gradient, at distance $r$ in front of or behind the point (dark-on-bright vs bright-on-dark). Two votes per radius instead of $\approx 2\pi r$; with known polarity, one. Collapsed variant: vote along the gradient ray for all radii into a 2D centre map, then estimate $r$ per centre from a radius histogram.
- Applications (each a preset): nuclei and round cells; bacterial / yeast **colonies** on a plate; **wells** of a multi-well plate (known radius, plus a regular grid); vesicles; **powder diffraction rings** (concentric circles with a known centre: the accumulator reduces to a 1D histogram of radius).
- Failure modes: touching or overlapping circles (shared votes, false centres between them), non-circular cells (a smeared peak), a wrong radius range, strong straight edges nearby. Alternatives on one slide: distance-transform peaks (day 8) and markers for watershed (day 10); LoG blob detection at scale $\sigma \approx r/\sqrt2$ (day 2) as the region-based cousin; a circle centre as the start of polar DP (elective).
- **Ellipses** briefly: five parameters; the full accumulator is impractical. Two ideas named: use pairs of points with parallel gradients (their midpoints lie on a line through the centre) to find the centre first, then the rest; or hypothesise-and-verify (unit 4).
- **The math**: the circle accumulator as a sum over edge points of a ring kernel, i.e. a **convolution** of the edge map with a ring of radius $r$ (link to `kernels/`; and with gradient voting the convolution of $\nabla I$ with a vector field); bins per dimension and the exponential growth $N^d$ of the accumulator; ellipse: the conic $Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0$ has 5 degrees of freedom.
- **Demo `houghcircle`** (new): left, image with edge points and detections; middle, the accumulator slice at the radius on the slider (heat map, peaks marked); right, the maximum over all radii (or, toggle, a 3D view of the $(a, b, r)$ accumulator as stacked slices via `lib/gl3d.ts`). Sliders: radius (slice shown), radius range $[r_\text{min}, r_\text{max}]$, vote threshold, NMS window, noise; toggles *gradient voting*, *polarity: dark-on-bright / bright-on-dark / either*. Click on an edge point to see its votes (a ring, or two dots per radius with gradient voting). Images: seeded synthetic nuclei (discs of several radii, touching pairs, one ellipse, partial occlusion; from the nuclei generator in `lib/synth.ts`), a synthetic multi-well plate, a synthetic powder-diffraction ring pattern, and `cell256.png`. Readouts: votes cast, time. Presets: "one radius: a sharp peak", "wrong radius: a ring, no peak", "scan the radius", "gradient voting", "touching cells: a false centre", "ellipse: smeared peak", "plate wells", "diffraction rings", "micrograph nuclei". TypeScript first; measure the 3D case without gradient voting at 256×256 (roughly 5 000 edge points × 40 radii × $2\pi r$ ≈ 10⁷ votes): only if it exceeds ~30 ms after optimisation does the inner loop move to Wasm.
- By hand: (a) radius 5, edge point $(3, 4)$ with unit gradient $(0.6, 0.8)$: the two candidate centres; then three edge points of a circle with given gradients: vote and find the centre; (b) predict: how many accumulator bins and votes for 300×300, radii 5–40, with and without gradient voting? (c) predict where the false centre appears for two touching discs. *(math)*: (d) show that the known-radius circle Hough is the convolution of the edge map with a ring; (e) degrees of freedom of a conic and why the ellipse accumulator is 5D.

### 3. `ght/`: The generalised Hough transform

- Concept layer. **Generalised Hough transform** (Ballard 1981): an arbitrary shape has no formula, so we store it as a table. Pick a reference point inside the template. For every edge point of the template, record the vector to the reference point, indexed by the edge direction: the **R-table**. In a new image, every edge point looks up the vectors listed for its direction and votes for the reference points they give. A peak = the shape is there. Worked on a small polygon with 4 directions.
- The circle with gradient voting (unit 2) *is* a GHT whose R-table holds one vector per direction (length $r$, along the gradient): the general method contains the special case.
- **Rotation and scale** as extra dimensions: rotate/scale the R-table vectors and the indexing direction, one accumulator slice per $(\phi, s)$; 2D becomes 4D. Cost and the remedy of coarse-to-fine search (a Gaussian pyramid, day 2).
- Occlusion and clutter: votes from the visible part still peak; similar clutter shapes give false positives; the threshold as a minimum visible fraction.
- Comparison slide: GHT (global, finds all copies, tolerates occlusion, fixed shape) vs active shape models (local, needs initial pose, shape varies; snakes elective) vs moment invariants (moments elective) vs polar DP (free-form star-shaped boundary from a centre; DP elective) vs template matching by NCC (FFT registration, day 12: intensities instead of edge votes). Where this leads: **Hough forests** learn which local patches vote for which object centre (trees from day 14); deep centre-voting detectors (day 18). One line each.
- **The math**: GHT with similarity transform $\mathbf{c} = \mathbf{x} - sR(\phi)\,\mathbf{v}$, $\phi$-shift of the table index; GHT as **matched filtering**: with all directions merged it is correlation of the edge map with the template's edge map (the NCC of day 12 on edge maps); cost of the $(\phi, s)$ search.
- **Demo `ght`** (new): left, the template (presets: a key, a leaf, a letter "R", a gear; editable as a polygon with draggable vertices) with its reference point (draggable) and the R-table drawn as arrows per direction bin, plus the table itself (direction bin → list of vectors); right, a scene with copies of the template (translated, optionally rotated and scaled), partial occlusion, clutter shapes and noise, the accumulator and detections. Sliders: number of direction bins, rotation range and step, scale range and step, threshold, noise, occlusion. Click an edge point in the scene to see its votes. Presets: "translation only", "occluded copy still found", "rotation without rotation search: missed", "rotation search: found, 36× the work", "similar clutter shape: false positive", "circle template = circle Hough".
- By hand: (a) build the R-table of a small square (4 directions) with the reference at its centre, and vote with the edges of a translated square; (b) the same square rotated by 90°: which table entries change, and does the unrotated table still find it? (c) predict: number of accumulator slices and votes for 36 rotations × 5 scales. *(math)*: (d) how the R-table changes under rotation by $\phi$ and scaling by $s$; (e) show that with one direction bin the GHT is the correlation of the two edge maps.

### 4. `ransac/`: Hypothesise and verify: RANSAC and randomised Hough

- Concept layer. Picture: a least-squares line through points with 30 % outliers is pulled away; the Hough transform solves it by voting, but needs a grid in parameter space. Other idea: **guess and check**.
- **RANSAC** (Fischler & Bolles 1981): repeat — pick a **minimal sample** (2 points for a line, 3 for a circle); fit the model exactly; count the points within a distance $\varepsilon$ (the **inliers**); keep the model with most inliers. At the end, refit by least squares on its inliers only.
- Parameters and failure modes: $\varepsilon$ too small (true points rejected), too large (outliers accepted, two nearby lines merged); not enough iterations; several models in the data (sequential RANSAC: remove inliers, run again) — Hough finds all at once.
- **Randomised Hough** (Xu, Oja & Kultanen 1990) as the bridge: pick a minimal sample, fit, and cast *one* vote for that parameter value (sparse accumulator). **Probabilistic Hough** (Kiryati et al. 1991; progressive variant Matas et al. 2000): vote with a random subset of edge points and stop once a peak is clear. Pointer: image libraries' "probabilistic Hough" returns segments.
- **Comparison** (table): voting (Hough) vs hypothesise-and-verify (RANSAC): number of parameters (Hough suffers $N^d$, RANSAC does not), many instances (Hough natural, RANSAC sequential), outlier fraction (Hough copes with very high fractions, RANSAC's iterations explode), precision (Hough limited by bins, RANSAC + refit is continuous), determinism (Hough deterministic, RANSAC random: fixed seeds in the demo). Where RANSAC is standard: fitting geometric transforms between images from matched points (registration, panoramas): one-line pointer to the Motion elective (planned, after day 12).
- **The math**: with inlier fraction $w$ and sample size $s$, one sample is all-inlier with probability $w^s$; after $N$ samples at least one is with probability $1 - (1 - w^s)^N$; so $N = \log(1 - p)/\log(1 - w^s)$. Worked: $w = 0.5$, $s = 2$, $p = 0.99$ gives $N = 17$; $s = 3$ gives 35; $w = 0.2$, $s = 2$ gives 113. Least-squares refit: total least squares for a line (the normal is the eigenvector of the scatter matrix $S$ with the smaller eigenvalue (PCA of the points, `pca/` day 1): unit 1's $(\theta, \rho)$ again), algebraic circle fit (linear in $(a, b, c)$ with $c = r^2 - a^2 - b^2$); MSAC / robust loss in one line.
- **Demo `ransac`** (new): a point set (draggable; add / remove points) with inliers on a line or a circle and seeded outliers; model *line / circle*. Buttons *one sample* (the two or three chosen points highlighted, the model drawn with its $\varepsilon$ band, inliers counted), *run N*, *reset*. Sliders $\varepsilon$, outlier fraction, iterations, seed. Panels: best-so-far model and inlier count vs iteration; least squares on all points vs RANSAC + refit; the curve $1 - (1 - w^s)^N$ against $N$ with the current run marked; side toggle *same points in the Hough accumulator* (reusing `lib/hough.ts`). Presets: "least squares fails", "RANSAC: 17 iterations suffice", "80 % outliers: Hough still works, RANSAC needs hundreds", "ε too large: merges two lines", "two lines: sequential RANSAC", "circle".
- By hand: (a) 8 given points, two given samples: fit the line through each, count inliers with $\varepsilon = 0.5$, choose; (b) compute $N$ for the three worked cases above and for a circle with $w = 0.3$; (c) predict: which is better for a line in 90 % clutter, and which for a 6-parameter model with 40 % outliers? *(math)*: (d) derive the formula for $N$; (e) show that the total-least-squares line normal is an eigenvector of the scatter matrix; (f) write the algebraic circle fit as a linear least-squares problem.

### 5. Link to the Radon transform (one slide, no unit)

- The Radon transform has its own elective day (`plan/radon-day.md`, Transforms group), which owns projections, sinograms, the Fourier slice theorem and reconstruction (it absorbs the `radon` demo idea as its `radonproj` demo). This day keeps one link slide: the Hough transform of a binary edge image is a discretised Radon transform; voting, accumulator peaks and line detection stay here.
- Shared conventions with the Radon day: $\theta \in [0^\circ, 180^\circ)$, signed $\rho$, the same sinogram/accumulator layout; `lib/hough.ts` may call `lib/radon.ts`.

## Shared engineering

- `demos/src/lib/hough.ts` (new): line accumulator over $(\theta, \rho)$ with optional gradient-direction window; circle accumulator for one radius or a radius range (full or gradient voting, with polarity); R-table build and GHT voting with rotation/scale steps; peak finding (threshold + NMS window, on 2D and 3D arrays); segment extraction along a detected line (gap, minimum length); an instrumented mode that records the votes of one chosen point for the "click to see its votes" overlays. Flat `Float32Array` accumulators.
- `demos/src/lib/ransac.ts` (new): generic RANSAC over a model interface (`minimal sample size`, `fit`, `distance`, `refit`), with line and circle models, seeded via `rng(seed)` from `lib/random.ts`, a step recorder for the step-through, the iteration formula, and randomised Hough voting into a sparse map.
- Edge points: Canny (Gaussian derivatives, NMS, hysteresis) from `lib/edges.ts` (owner: edges day, day 2), which returns the typed edge list `{x, y, theta, strength}` used by every Hough demo; Gaussian and Sobel from `lib/imageops.ts`; no edge code here.
- Plots (accumulator heat maps, inlier count vs iteration, probability curve, profiles): `lib/plot.ts` (owner morphology day) plus a heat-map helper shared with the `pixelgraph`/`ncut` demos; diverging/sequential maps from `lib/colour.ts`.
- Synthetic images (seeded, with known ground truth parameters for testing): lines and grid scene, road with lanes, rotated striped page, straight filaments in clutter, nuclei (generator in `lib/synth.ts`), multi-well plate, powder-diffraction rings, GHT template scenes. Real: `demos/src/fourier2d/cell256.png` (credited as in Part 1). A photograph for lane/building lines only if a free-licence image is found; otherwise synthetic.
- Speed: TypeScript first. Candidates to measure for Wasm (`demos/wasm/hough/`): the full 3D circle accumulator without gradient voting at 256×256, and GHT with rotation × scale search. Keep the TS reference and an agreement test if Wasm is used.
- Tests (`demos/test/hough-test.ts`, run by `make test`): line and circle detection on synthetic images recovers the generating parameters within one bin; GHT finds a translated / rotated template; RANSAC with a fixed seed finds the inlier set; the iteration formula against simulation; Radon of a point is a sinusoid.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560 on a slide, static fallback figures.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1: Canny recall (day 2), lecture + by-hand sinusoids, `houghdual`, `houghlines` |
| 10:45–12:15 | Unit 2 lecture + by-hand circle votes, `houghcircle` |
| 13:15–14:45 | Unit 3 lecture + R-table by hand, `ght` |
| 15:00–16:30 | Unit 4 lecture + exercise (`ransac`); Radon link slide; quizzes as wrap-up |

## Open questions

- **Decided (course map):** Canny belongs to the edges day (day 2, `canny/`); unit 1 here is a recall, and the former unit 3 is split into `hough-circles/` and `ght/`. (Review-ordering's edit 15, "keep Canny here", is overruled by the map.)
- **RANSAC**: core here. If the Motion elective (registration with feature matching) is written, it recalls RANSAC from this day and keeps the transform-fitting application.
- **Decided:** the Radon link stays one slide; the Radon elective owns the slice theorem.
- Real images: is there a free-licence photograph (road, building) and a real micrograph with straight filaments or a plate image? Otherwise all applications except `cell256.png` are synthetic.
- **Decided (course map):** a day is four units of about 1.5 h.

## References

- Hough (1962), *Method and means for recognizing complex patterns*, US Patent 3,069,654.
- Duda & Hart (1972), *Use of the Hough transformation to detect lines and curves in pictures*, Comm. ACM (normal parameterisation).
- Canny (1986), *A computational approach to edge detection*, PAMI.
- Kimme, Ballard & Sklansky (1975), *Finding circles by an array of accumulators*, Comm. ACM.
- Ballard (1981), *Generalizing the Hough transform to detect arbitrary shapes*, Pattern Recognition.
- Illingworth & Kittler (1988), *A survey of the Hough transform*, CVGIP.
- Xu, Oja & Kultanen (1990), *A new curve detection method: randomized Hough transform*, Pattern Recognition Letters.
- Kiryati, Eldar & Bruckstein (1991), *A probabilistic Hough transform*, Pattern Recognition; Matas, Galambos & Kittler (2000), *Robust detection of lines using the progressive probabilistic Hough transform*, CVIU.
- Fischler & Bolles (1981), *Random sample consensus*, Comm. ACM.
- Gall & Lempitsky (2009), *Class-specific Hough forests for object detection*, CVPR.
- Deans (1983), *The Radon Transform and Some of Its Applications*, Wiley; Kak & Slaney (1988), *Principles of Computerized Tomographic Imaging*, IEEE Press (Fourier slice theorem).
