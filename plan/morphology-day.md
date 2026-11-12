# Plan: a day on mathematical morphology

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
**Course map (2026-11-12):** core day 8 of 18, Part II "From pixels to objects". Order: day 7 (colour) → this → day 9 (labelling). **Owner of the distance transform** (course map; moved here from the watershed plan, review-ordering C2).

## Goal of the day

After the day a student can:

1. read a binary image as a **set**, choose a **structuring element** (SE) and compute **erosion** and **dilation** by hand, both as "where does the SE fit / hit" and as a **min / max over a neighbourhood**; state why these are not convolutions (nonlinear, but shift-invariant);
2. compute **opening** and **closing**, state their properties (increasing, idempotent, anti-extensive / extensive, duality), and use them to remove noise and to **filter objects by size**, including a **granulometry** (size distribution);
3. apply the same operators to **grey-level** images: remove an uneven background with a **top-hat**, find edges with the **morphological gradient**, and predict from the SE size which bright or dark structures survive;
4. compute a **geodesic dilation** and a **morphological reconstruction** by hand, and use it to **fill holes**, **remove objects touching the border**, compute **h-maxima / h-minima** and **impose markers** (what the watershed day needs);
5. define the **distance transform** $\mathrm{DT}(p)$, compute a city-block DT **by hand**, *(math)* state the two-pass chamfer algorithm, and relate erosion by a disc to **thresholding the distance transform** and dilation by a disc to a front moving at constant speed.

Why it fits this audience: the definitions are set operations and min/max, so everything
can be done by hand on a 7×7 grid, and the formal side is light. The properties have a
linear-algebra counterpart worth naming: an opening is idempotent like a projection
($P^2 = P$), dilation distributes over union as a linear map distributes over sums
(max-plus algebra in one sentence), and erosion/dilation are a dual pair like a map and
its adjoint. The sliding window, the border problem and the "separable kernel" trick
come straight from `kernels/` (day 1), so morphology is introduced as "the same
window, but take the min or max instead of the weighted sum".

## Prerequisite gap

- **Sliding-window filtering, border handling, separability**: known from `kernels/` (day 1).
  Recall in one slide; the border question returns with a twist (erosion needs
  "outside = foreground" or "outside = background" to be chosen, and the choice
  visibly matters).
- **Binary images / thresholding** are taught on day 6 (`plan/thresholding-day.md`).
  Unit 1 opens with a one-line recall (threshold → 0/1 mask, foreground = 1), with noise
  specks and gaps as the motivation for the day. Top-hat background correction is owned
  here (unit 3); day 6's unit 3 had a forward pointer to it. The **median filter** was
  met on day 6 (one slide there); unit 3 here places it among the rank filters.
- **Connected components** (4- vs 8-connectivity) are needed here only as a picture: in
  unit 1 (cross vs square SE) and in unit 4 (reconstruction floods one component). Labelling as an algorithm, and
  everything done with labels, is its own day, directly after this one
  (`plan/labelling-day.md`; decided 2026-11-12). Keep the conventions (foreground
  8-connected, background 4-connected) identical here, on the labelling day and on the
  watershed day.
- **Distance transform**: **defined here** (unit 2): $\mathrm{DT}(p) = \min_{q \notin A}\|p - q\|$,
  a picture, the city-block DT by hand, the two-pass chamfer algorithm in the math layer,
  exact Euclidean DT named. The watershed day (9), skeleton elective, measurements (day 11),
  dynprog elective ("chamfer is DP") and level sets (SDF) recall it. Symbol $\mathrm{DT}$
  (course convention; $D_p$ is the data term on day 13).
- **Set notation** (union, intersection, complement, translate $A_x$, reflection
  $\check B = \{-b\}$): one slide, with a pixel-grid picture for each.

## Links to other days

- **Labelling** (day 9, `plan/labelling-day.md`, the next day): a connected component is the
  reconstruction of the mask from a single pixel (one line in unit 4, developed there).
  The labelling day uses this day's opening, fill holes and clear border to clean masks
  before measuring.
- **Skeletons** (elective "Objects", `plan/skeleton-day.md`, entry after day 9): hit-or-miss,
  thinning, medial axis and skeleton analysis all live there. It builds on this day's
  erosion, opening (Lantuéjoul's formula) and the distance-transform link of unit 2;
  one pointer slide at the end of unit 2 ("the centres of the largest discs that fit
  form the skeleton: see the skeleton elective").
- **Watershed** (day 10, `plan/watershed-day.md`) relies on this day for the *definitions* of
  **morphological reconstruction**, **h-minima** (fill every basin by $h$ = reconstruction
  by erosion of $f + h$ above $f$), **regional minima**, **minima imposition** for
  marker-controlled watershed, **fill holes** (applied to a mask before the
  distance transform in `ws-distance/`), and the **distance transform** itself (unit 2;
  `ws-distance/` recalls it and spends its time on the splitting). All are in unit 4 here, with the same 1D
  integer-signal exercise style as the watershed day's unit 1, so the two by-hand
  answers can be checked against each other. The `recon1d` demo is designed to be
  embedded on the watershed day (`data-mode="hmin"`), and the watershed library calls
  this day's reconstruction code rather than re-implementing it.
- **Distance transform**: erosion/dilation by a Euclidean disc of radius $r$ = threshold
  of $\mathrm{DT}$ (unit 2). This day owns `lib/distance.ts`; the watershed day uses it.
- **Level sets** (elective): dilation by a disc of radius $r$ = the front moved outwards by
  $F = 1$ for time $r$ (the zero level of the SDF shifted by $r$). One slide in unit 2.
- **Graph cuts** (day 13): opening/closing of a noisy threshold as the cheap, local alternative
  to a smoothness term; same test image so results can be compared (unit 2, one forward-pointer slide).
- **Marching cubes** (3D elective; 3D images in its `volumes3d/` unit): all operators carry over to 3D with a ball as SE; a closing of a
  segmented stack before surface extraction removes tiny handles and cavities (one
  sentence in unit 1).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise. (Skeletons, hit-or-miss and
thinning moved to their own day, `plan/skeleton-day.md`; connected-component labelling
to `plan/labelling-day.md`.)

### 1. `morph-binary/`: Sets, structuring elements, erosion and dilation

- Motivation: a thresholded micrograph with specks, holes and gaps. Linear blurring
  smears the mask into grey; we want an operation that keeps it binary and acts on shape.
- Binary image = set $A \subset \mathbb{Z}^2$ of foreground pixels. Structuring element
  $B$: a small set with a marked **origin** (cross, 3×3 square, disc, line, any shape).
- **Erosion** $A \ominus B = \{x : B_x \subseteq A\}$: where the SE **fits**.
  **Dilation** $A \oplus B = \{a + b : a \in A, b \in B\} = \{x : \check B_x \cap A \ne \emptyset\}$:
  where the reflected SE **hits** (the Minkowski sum). Equivalent forms
  $A \ominus B = \bigcap_{b \in B} A_{-b}$, $A \oplus B = \bigcup_{b \in B} A_b$: "union of shifted copies".
- As grey values: $(A \ominus B)(x) = \min_{b \in B} A(x + b)$,
  $(A \oplus B)(x) = \max_{b \in B} A(x - b)$. Same sliding window as convolution;
  min/max instead of weighted sum. Not linear ($\max(f + g) \ne \max f + \max g$), but
  shift-invariant. Why the reflection appears in dilation (the same flip as in
  convolution vs correlation).
- Properties: **increasing** ($A \subseteq A' \Rightarrow$ same for results);
  **extensive / anti-extensive** if the origin is in $B$; **duality**
  $(A \ominus B)^c = A^c \oplus \check B$ (eroding the object = dilating the background);
  dilation is commutative and associative, so $A \oplus (B_1 \oplus B_2) = (A \oplus B_1) \oplus B_2$:
  a 15×15 square is a horizontal then a vertical line (the separable-kernel trick
  again); a disc is approximated by lines in a few directions. Cost $O(n|B|)$ naive,
  $O(n)$ per line with van Herk / Gil–Werman (named, one picture).
- **Boundary extraction**: $A \setminus (A \ominus B)$ (inner boundary), with the cross
  vs the square giving an 8- vs 4-connected contour.
- Border convention: outside = 0 makes erosion eat in from the image edge; outside = 1
  does not. Shown in the demo.
- **Demo `morphgrid`** (new, main demo of units 1–2): a 16×16 binary grid (up to 24×24),
  paint and erase by click/drag. SE editor: a 5×5 (or 7×7) grid, click to toggle cells,
  a *set origin* button then click (no hover, no modifier keys needed); SE presets
  cross / square / disc / line / L-shape (asymmetric). Operation: erosion, dilation,
  opening, closing, inner boundary, $A^c$. Output overlaid on the input: kept pixels
  dark, removed pixels red, added pixels green. *Step* moves the SE over the image in
  scan order: the SE is drawn at the current pixel, its cells green where they hit
  foreground and red where they miss, and the decision written out ("all 5 inside → 1",
  "one hit → 1"). Buttons *step*, *step row*, *run*, *reset*; border: outside = 0 / 1.
  Toggle *show duality*: erosion of $A$ next to the complement of the dilation of $A^c$
  by $\check B$. Presets: "erosion: where it fits", "dilation adds a layer", "the L-shape:
  why reflect", "duality", "two lines make a square", "border convention".
- By hand: a 7×7 binary image and a 3-pixel L-shaped SE with the origin in the corner:
  erosion and dilation; check the duality on the same example; then the inner boundary
  with the cross and with the square.

### 2. `morph-openclose/`: Opening, closing and size filtering

- **Opening** $A \circ B = (A \ominus B) \oplus B$ = the union of all translates of $B$
  that fit inside $A$ ("roll the SE around inside; keep what it reaches"). **Closing**
  $A \bullet B = (A \oplus B) \ominus B$ = the same for the background.
- Properties: increasing; opening **anti-extensive**, closing **extensive**; both
  **idempotent** ($(A \circ B) \circ B = A \circ B$: a projection); duality
  $(A \circ B)^c = A^c \bullet \check B$. Proof of idempotence sketched with the
  "union of fitting translates" picture, not with algebra.
- Noise removal: opening removes specks smaller than $B$, closing fills holes and gaps
  smaller than $B$. Open-then-close ≠ close-then-open (alternating sequential filters
  named). What else is lost: corners rounded, thin necks cut, thin bridges kept.
- **Size filtering and granulometry**: openings with discs of growing radius $r$ form a
  decreasing family. The area left $a(r) = |A \circ D_r|$ is a cumulative size
  distribution; the **pattern spectrum** $a(r) - a(r + 1)$ is a histogram of object
  sizes, without segmenting individual objects. Size = "the largest disc that fits",
  so a long thin object counts as small.
- **The distance transform** (owned here): $\mathrm{DT}(p)$ = distance from a foreground
  pixel to the nearest background pixel. Picture first (a blob and its DT as a "mountain";
  ridges in the middle). The metric matters: city-block (4-steps), chessboard (8-steps),
  chamfer (3-4 weights), Euclidean. By hand on an 8×8 mask with the city-block metric.
  Algorithm: two raster passes (forward with the upper-left neighbours, backward with the
  lower-right), each a "min over neighbours + step": the math slide; exact Euclidean DT
  named (separable, linear time), not derived.
- **Link to the distance transform**: with a Euclidean disc $D_r = \{b : \|b\| \le r\}$,
  $A \ominus D_r = \{x : \mathrm{DT}(x) > r\}$ and $A \oplus D_r = \{x : \mathrm{DT}_{A^c}(x) \le r\}$ (distance
  to $A$). Erosion by any disc is one threshold of one image; an opening is two distance
  transforms. Level-set link: the dilated boundary is the front moved out at unit speed.
  Digital caveat: chessboard distance ↔ square SE, city-block ↔ diamond (cross iterated).
- Pointer (one slide): the centres of the largest discs that fit inside $A$ form its
  skeleton, the ridges of $\mathrm{DT}$; hit-or-miss, thinning and skeletons are the skeleton
  elective (`plan/skeleton-day.md`).
- Forward pointer to graph cuts (day 13): opening/closing of a noisy threshold vs an energy with a
  smoothness term on the same image (one slide, side by side).
- **Demo**: reuse `morphgrid` with `data-mode="openclose"` (presets: "opening removes
  the speck", "closing fills the gap", "open∘close ≠ close∘open", "apply twice: nothing
  changes").
- **Demo `granulo`** (new): a 256×256 binary image of seeded discs from two or three
  size populations plus specks (and a thresholded mask of the synthetic nuclei image).
  Slider $r$; the opening shown with removed pixels coloured by the radius at which they
  disappeared (a "size map"). Right: the curve $a(r)$ and the pattern spectrum as bars,
  current $r$ marked. SE: disc / square / diamond / horizontal line (orientation filter).
  Toggle *via distance transform*: shows $\mathrm{DT}$ with the iso-line $\mathrm{DT} = r$ on top of the
  eroded set, so the coincidence is visible. Presets: "two sizes, two peaks",
  "touching discs count as one bigger object?", "lines: find the horizontal ones",
  "square SE on discs".
- By hand: open and close a 7×7 image with the 3×3 square; check idempotence by doing it
  twice. Then a 1D binary signal with runs of length 1, 2, 2, 4, 5: the pattern spectrum
  for line SEs of length 1–5. Finally **compute** the city-block distance transform of an
  8×8 mask (concept: by counting steps; *(math)*: with the two-pass algorithm), read off
  the erosion by the diamond of radius 2 and check two pixels directly.

### 3. `morph-grey/`: Grey-level morphology, top-hat and gradient

- Flat SE on a grey image: erosion = **min filter**, dilation = **max filter**; the
  formulas of unit 1 unchanged. Picture: the graph of $f$ as a landscape (the umbra);
  the 1D SE as a horizontal bar.
- **Opening** = push the bar up from below as far as it goes; the top of the bar traces
  $f \circ B$. Bright peaks narrower than $B$ are cut off, wider ones kept. **Closing** =
  push down from above: dark valleys narrower than $B$ are filled.
- **Top-hat** $f - f \circ B$ (white) and $f \bullet B - f$ (black): the small bright (dark)
  structures on a slowly varying background. Classic use: uneven illumination, then a
  single threshold works. Choosing $B$: larger than the objects, smaller than the
  background variation. Rolling ball (ImageJ's background subtraction) = opening with a
  non-flat, ball-shaped SE (one slide; non-flat SEs: $\max_b f(x - b) + s(b)$).
- **Morphological gradient** $(f \oplus B) - (f \ominus B)$; inner and outer halves.
  Compared with Sobel from `kernels/`: no sign, thicker, but no direction preference and
  no ringing.
- Rank filters as the family between: min, median, max over the same window (the median,
  met on day 6, removes salt-and-pepper noise that the min or max alone cannot). Link to
  `kernels/`: nonlinear filters with the same window.
- Pointer: the max-plus algebra (dilation is a "convolution" with max for sum and $+$
  for product); no convolution theorem, but the slope transform as further reading.
- **Demo `morph1d`** (new): a 1D profile (~120 samples) in SVG. The flat SE drawn as a
  bar of width $w$ (slider) that **rolls** under (opening) or over (closing) the signal:
  a *position* slider moves it, *sweep* animates it, and the trace left behind is drawn.
  Selectable output traces: erosion, dilation, opening, closing, white/black top-hat,
  gradient, median; SE shape flat / parabola (non-flat). Profiles: presets and drag
  control points; one profile is a line across `cell256.png`. Presets: "narrow peak goes,
  wide peak stays", "peaks on a slope: top-hat", "noise: min, median, max",
  "gradient vs derivative".
- **Demo `tophat`** (new): 256×256, three panels: input, operation result, threshold
  overlay. Inputs: a seeded synthetic image of small bright spots and dark specks on a
  sloping, vignetted background with noise; `cell256.png` (dark granules and
  membranes: black top-hat); the synthetic nuclei. Controls: SE disc radius, operation
  (erosion, dilation, opening, closing, white / black top-hat, morphological gradient,
  Sobel for comparison), threshold slider applied to the input vs to the top-hat with a
  readout of detected spots (true/false for the synthetic image). A draggable profile
  line shows the 1D profile under it, linking to `morph1d`. Presets: "one threshold
  fails", "top-hat flattens the background", "SE too small: spots shrink", "SE too large:
  background leaks in", "granules in the cell", "morphological gradient vs Sobel".
- By hand: a 1D signal of 12 integers: erosion, dilation, opening, white top-hat with a
  flat SE of width 3; which peaks survive with width 5? Then the morphological gradient
  of a 1D step and of a ramp.

### 4. `morph-reconstruct/`: Geodesic operations, reconstruction and h-extrema

- Problem: an opening removes small objects but also changes the shapes of the
  remaining ones. Wanted: "keep every object that contains a seed, unchanged".
- **Marker** $m$ and **mask** $g$ with $m \le g$. **Geodesic dilation**
  $\delta_g(m) = (m \oplus B) \wedge g$: dilate, then clip under the mask. Iterate until
  nothing changes: **reconstruction by dilation** $R_g(m)$. Binary picture: flood fill
  from the seeds inside the mask. Dual: geodesic erosion $(m \ominus B) \vee g$ with
  $m \ge g$ → reconstruction by erosion $R^\varepsilon_g(m)$.
- Binary applications: **opening by reconstruction** $R_A(A \ominus B)$ (small objects go,
  large ones return intact); **remove border objects** $A \setminus R_A(A \cap \text{frame})$;
  **fill holes** $\big(R_{A^c}(A^c \cap \text{frame})\big)^c$ (the background that cannot be
  reached from the image edge is a hole).
- One line: the connected component containing $p$ is $R_A(\{p\})$; labelling all
  components efficiently is the next day (day 9, `plan/labelling-day.md`).
- Grey-level reconstruction: same definition with min/max. **h-maxima**
  $\mathrm{HMAX}_h(f) = R_f(f - h)$: every peak lowered by $h$, peaks of height (dynamic)
  $< h$ flattened away; the domes $f - \mathrm{HMAX}_h(f)$; **regional maxima** = plateaus
  with all neighbours lower; **extended maxima** = regional maxima of $\mathrm{HMAX}_h$
  (markers for the nuclei). **h-minima** dual: $R^\varepsilon_f(f + h)$, basins
  shallower than $h$ filled; this is exactly what the watershed day (day 10) uses.
- **Minima imposition**: given a marker image, $f_m = 0$ on markers and $\max f$
  elsewhere; $R^\varepsilon_{(f + 1) \wedge f_m}(f_m)$ keeps $f$'s shape but has regional
  minima only at the markers. This is how "marker-controlled watershed" is computed;
  named here, used there.
- Algorithm: iterate geodesic dilations (slow, but this is the definition and what the
  step-through shows); the fast version with a FIFO queue (Vincent 1993) in one slide:
  only pixels on the changing front are revisited.
- **Demo `recon1d`** (new, small, reused by the watershed day): a 1D integer profile
  (~40 samples, editable by dragging, values shown) as bars. Marker drawn in a second
  colour under the mask. Modes (`data-mode`): *reconstruct* (marker = $f - h$ or clicked
  samples), *hmax*, *hmin*, *impose* (click samples to make markers; the imposed profile
  appears). *Step* applies one geodesic dilation (the dilated marker shown dashed, the
  clipped part hatched), *run to stability* with an iteration counter. Slider $h$;
  readouts: number of regional minima/maxima before and after. Presets: "a single
  seed floods its hill", "h removes the shallow peaks", "h-minima: basins fill",
  "two markers: imposition", and the same 12-sample signal as the watershed day's
  by-hand exercise.
- **Demo `reconstruct`** (new): 256×256, binary and grey tabs. Binary: a seeded mask of
  blobs (some touching the frame, some with holes, specks); buttons *fill holes*,
  *clear border*, *opening* vs *opening by reconstruction* (SE radius slider); click a
  blob to use that pixel as the marker (its reconstruction is exactly that blob: the
  one-line link to labelling). No *label* button: colour-by-component lives in the
  labelling day's demos.
  Grey: synthetic nuclei or `cell256.png`; $h$ slider with regional maxima / minima as
  dots and their count; domes and h-maxima as images; *impose* mode with clicked
  markers, showing the imposed landscape. A *replay* slider shows the reconstruction
  growing iteration by iteration. Presets: "fill holes", "clear border", "opening breaks
  shapes, reconstruction does not", "one dot per nucleus with h = …", "noise: hundreds
  of maxima".
- By hand: (a) the 12-integer 1D signal from the watershed day: compute $f - h$ for
  $h = 2$, iterate geodesic dilations to stability, mark the h-maxima; then the h-minima
  and check against the watershed answer. (b) A 7×7 mask with a hole and a blob on the
  frame: fill holes and clear border by reconstruction.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, erosion/dilation by hand, `morphgrid` |
| 10:45–12:15 | Unit 2 lecture, opening/closing, 1D pattern spectrum and city-block DT by hand, `granulo` |
| 13:15–14:45 | Unit 3 lecture + exercise (`morph1d`, `tophat`) |
| 15:00–16:30 | Unit 4 lecture, 1D reconstruction by hand, `recon1d`, `reconstruct`; quizzes as wrap-up |

## Shared engineering

- `demos/src/lib/morph.ts` (new): one implementation for all demos of the day and for the
  watershed day.
  - SE type: list of offsets with an origin; constructors (cross, square, diamond,
    Euclidean disc, line at an angle, from a grid), reflection, decomposition of squares
    into lines.
  - Binary (`Uint8Array`) and grey (`Float32Array`/`Uint8Array`) erosion, dilation,
    opening, closing, top-hats, gradient, rank/median; border mode outside = 0 / 1 /
    replicate; non-flat SE (parabola/ball). Naive version (for step-through and tests)
    plus van Herk / Gil–Werman for lines and rectangles. A 1D signal is a 1×n image.
  - Geodesic dilation/erosion, reconstruction (instrumented iterate-to-stability version
    that records each step, and a fast FIFO-queue version), fill holes, clear border,
    opening by reconstruction, regional extrema, h-maxima/h-minima, extended extrema,
    minima imposition.
  - Erosion/dilation by an arbitrary SE (used by the skeleton day's hit-or-miss and
    Lantuéjoul skeleton, which live in that day's `lib/skeleton.ts`).
- `demos/src/lib/label.ts` is **owned by the labelling day** (`plan/labelling-day.md`),
  which specifies it. This day may call it (e.g. counting regional maxima as plateau
  components in `reconstruct`, counting objects in `granulo`) but adds nothing to it.
  `morph.ts` should call `label.ts` for "regional extrema = components of plateaus" so
  that `lib/watershed.ts` can call either and nothing is implemented twice.
- **Owner of `demos/src/lib/distance.ts`**: city-block, chessboard and chamfer two-pass DT
  (instrumented for step-through), exact Euclidean DT; used by `granulo`, the watershed
  day's `ws-distance/`, skeleton, measurements and dynprog.
- Reused, not duplicated: `lib/imageops.ts` (owner `kernels/`: `loadGrey()` for
  `cell256.png`, Gaussian/Sobel for the gradient comparison); synthetic images in
  `lib/synth.ts` (owner thresholding: the seeded nuclei generator, "spots on an uneven
  background", "discs of several sizes", "blobs with holes and frame contact" are added
  there); label colours from `lib/colour.ts` (owner colour). The FIFO for fast
  reconstruction comes from `lib/pqueue.ts` (owner watershed: heap + bucket + FIFO);
  until that exists, a ring buffer in `morph.ts` that moves there.
- **Owner of `lib/plot.ts`** (never in `lib/ui.ts`): the SVG line/bar/scatter/heat-map
  helper for `granulo`, `morph1d`, `recon1d`, and for ~20 demos elsewhere (the 1D
  step-through demos `flood1d`, `segchain`, `mrf1d`, `xcorr1d`, `dtw` share its line
  plot plus step controls).
- **Owner of `lib/gridui.ts`**: the grid step-through UI (cells ≥ 24 px, step/back/play,
  value overlay) under `morphgrid`; reused by labelling (`label` mode, `floodgrid`),
  skeleton (`hitmiss`), watershed, dynprog (`dpgrid`), integral (`satstep`), graphics3d
  (`raster`).
- **Language / speed** (`demos/README.md`): all of this stays in **TypeScript**. At
  256×256 a naive disc erosion with $r \le 10$ (~300 offsets) is ~20 M min-operations,
  around 20–40 ms in TS; decomposition, van Herk lines or the distance-transform route
  bring it to a few ms, and reconstruction with a FIFO queue is linear. Measure first.
  Candidates for Rust/Wasm only if measurement shows > 30 ms: `granulo` recomputing the
  whole size map for all radii on a slider change (precompute once instead: the size map
  is one image, so this should not be needed), and a non-flat ball SE on 256×256 if it
  is made live. No WebGL (all demos are 2D).
- Tests (`demos/test/morphology-test.ts`, run by `make test`): duality
  $(A \ominus B)^c = A^c \oplus \check B$ on random images and SEs; adjunction
  $A \oplus B \subseteq C \iff A \subseteq C \ominus B$; opening idempotent and
  $A \circ B \subseteq A \subseteq A \bullet B$; van Herk = naive; Euclidean-disc erosion =
  threshold of the exact distance transform; two-pass chamfer = brute-force distances; fast reconstruction = iterated geodesic
  dilation; h-minima in 1D = a brute-force "fill by $h$" (the
  same check the watershed plan lists).
- Images: `demos/src/fourier2d/cell256.png` via `loadGrey()` (credit as in `fourier2d`); synthetic images
  generated in TS with fixed seeds (no files).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only (no modifier keys or
  hover; keyboard back to reveal.js), labels ≥ 18 px, fits 1150×560 on a slide, `data-*`
  settings (`data-mode`, `data-image`, `data-preset`), static figures as print fallbacks.
  The 16×16 `morphgrid` must keep cells ≥ 24 px on a slide.
  Two other days add modes to `morphgrid`, specified in their plans: `data-mode="label"`
  (`plan/labelling-day.md`) and `data-mode="hitmiss"` with thinning (`plan/skeleton-day.md`).

## Open questions

- **Decided (course map):** the distance transform is owned here (unit 2, which is now
  denser; if it overruns, `granulo`'s line-SE presets go to the exercise).
- How long is a "day" (same question as for the other days)? If four units are too many,
  units 1 and 2 merge into one (keeping `morphgrid` and dropping `granulo` to the
  exercise), or the grey-level unit is shortened to top-hat only. Moving labelling out
  of unit 4 already frees about 20 min there.
- ~~Skeletons and hit-or-miss: optional or core?~~ Decided 2026-11-12: **their own day**
  (`plan/skeleton-day.md`, now an elective in group "Objects"), including hit-or-miss;
  this day keeps one pointer slide.
- How much algebra: properties only (proposal), or also adjunction and the lattice view
  (openings as the general "increasing, anti-extensive, idempotent" operators)?
- A real image for top-hat and granulometry: `cell256.png` (TEM, single cell) has
  granules but no clear uneven background. Candidates: a fluorescence image with
  vignetting and many spots/nuclei (e.g. from the Broad Bioimage Benchmark Collection,
  licence to be checked, same search as the watershed plan), or a photo taken by the
  course team (e.g. grains on a table with side lighting) with a clear licence.
- ~~Where is connected-components labelling taught as an algorithm?~~ Decided
  2026-11-12: its **own day, directly after this one** (`plan/labelling-day.md`); here
  only the one-line link "component = reconstruction from one pixel".
- 3D morphology: one sentence (proposal), with a pointer to `volumes3d/` (3D elective).
- Programming language for any code parts is still open (see README).

## References

- Serra (1982), *Image Analysis and Mathematical Morphology*, Academic Press.
- Soille (2003), *Morphological Image Analysis: Principles and Applications*, 2nd ed., Springer (reconstruction, h-extrema, minima imposition, granulometries).
- Gonzalez & Woods, *Digital Image Processing*, chapter on morphological image processing (textbook-level treatment, by-hand examples).
- Haralick, Sternberg & Zhuang (1987), *Image analysis using mathematical morphology*, PAMI.
- Vincent (1993), *Morphological grayscale reconstruction in image analysis: applications and efficient algorithms*, IEEE Trans. Image Processing.
- van Herk (1992), *A fast algorithm for local minimum and maximum filters on rectangular and octagonal kernels*, Pattern Recognition Letters; Gil & Werman (1993), PAMI.
- Maragos (1989), *Pattern spectrum and multiscale shape representation*, PAMI.
- Sternberg (1986), *Grayscale morphology*, CVGIP (rolling ball).
- Rosenfeld & Pfaltz (1966), *Sequential operations in digital picture processing*, JACM (distance transform; their two-pass labelling is on the labelling day).
