# Plan: a day on optical flow (and non-rigid registration)

Internal working doc (plan/ is not rendered). Status: core units 1–4 built 2026-11-13 (see "Build notes" at the end); optional unit 5 not built.
**Course map (2026-11-13, self-study restructure):** day 25 of 51, Part V "Aligning images and motion". Hard prerequisites: Edges, blobs and ridges (day 2), FFT registration (day 23). Soft (see also): Deconvolution (day 5), Splines and NURBS (day 6). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part V (should-have), entry after **FFT
registration (day 23)**. First day of the group; the tracking day (same group, not yet planned)
comes after labelling and the point-detection day and may follow this one. This day also holds
**non-rigid registration** as one unit (the course map left "related unit or day" open: unit 4
here; see open questions).

**Notation** (course conventions): λ is the weight of the smoothness/regularisation term (Horn
and Schunck's α² is written λ); the flow is $(u, v)$ in pixels per frame; the concept layer says
"frame 1 / frame 2" (FFT registration said "image A / image B"; for registration in unit 4 the
"fixed" and the "moving" image); $I_x, I_y, I_t$ are the derivatives in $x$, $y$ and time;
"window" is the local neighbourhood of Lucas–Kanade (the edges day's integration scale ρ), and
"taper" stays reserved for Hann weighting (day 23).

## Goal of the day

After the day a student can:

1. describe a **flow field** (a vector per pixel: where this pixel's content went), read it in
   the standard visualisations (arrows, the **colour wheel**: hue = direction, saturation =
   speed), and say what the **brightness constancy** assumption means and when biology breaks it
   (bleaching, blinking, focus drift, objects entering the plane);
2. derive (concept: linearise; math: Taylor) the **optical flow constraint** $I_x u + I_y v + I_t = 0$:
   one equation, two unknowns; explain the **aperture problem** (along an edge only the
   **normal** component of motion is visible) and draw the **constraint line** in velocity space;
3. read motion as **orientation in space–time**: in a **kymograph** (an $x$–$t$ image) a moving
   feature is a line whose slope is its speed; 1D flow = the orientation of that line;
4. compute **Lucas–Kanade** flow in a window by hand: least squares over the window's
   constraints, whose 2×2 matrix is **the structure tensor** of the edges day; read reliability
   from its eigenvalues (flat: both small; edge: one small = aperture problem; corner/texture:
   both large); explain why it fails for **large motion** and how a **coarse-to-fine pyramid**
   with warping fixes it;
5. explain **Horn–Schunck** flow as a **regularised energy** (data term + λ·smoothness, the
   pattern of the deconvolution day), predict what λ does (noise and holes vs over-smoothed
   motion boundaries), and see **filling-in**: the flow from edges and corners spreads into flat
   regions over the iterations;
6. **evaluate** a flow field: **endpoint error** (and angular error) against ground truth from a
   synthetic warp; without ground truth, the **warping residual** (warp frame 2 back and compare);
7. describe **non-rigid registration** as finding a smooth **deformation field**, parametrise it
   by a **B-spline free-form deformation** (control grid, local support), regularise it
   (bending energy, no folding: Jacobian determinant > 0), and choose a **similarity measure**:
   SSD (same contrast), NCC (gain/offset), **mutual information** (any consistent relation
   between intensities, e.g. two stains or two modalities); compute MI from a small joint
   histogram by hand;
8. *(math)* derive the Lucas–Kanade normal equations, the Horn–Schunck per-pixel update
   ("average the neighbours, then step towards the constraint line"), the cubic B-spline values
   $\tfrac16, \tfrac46, \tfrac16$ at the knots, and $\mathrm{MI} = H(A) + H(B) - H(A, B)$.

Why it fits this audience: optical flow is three things they already have, put together: a
Taylor expansion (calculus), a 2×2 least-squares problem whose matrix is the structure tensor
(edges day, day 2), and a data + λ·penalty energy (deconvolution day, day 5). Every method has a
1D version small enough to do by hand. The applications (cell migration, tissue flow, cytoplasmic
streaming, kymographs of transport) are exactly what the biology-oriented students image.

## Prerequisite gap

All needed days come before day 23, so the day needs nothing new from Parts IV–V.

- **Gradient as a difference of neighbours, Gaussian derivatives** (`canny/`, day 2): recalled
  in one line in unit 1.
- **The structure tensor** $J = G_\rho * (\nabla I\,\nabla I^\top)$, its eigenvalues, coherence,
  Harris/Shi–Tomasi corners (`structure-tensor/`, day 2): unit 2 recalls it in **one slide** and
  then shows that Lucas–Kanade's matrix **is** $J$ (the built unit's "Interest points" slide
  already says "the optical flow and tracking days builds on this").
- **Gaussian pyramid** (`scale-space/`, day 2; owner): recalled in one line in unit 2 (blur, then
  downsample by 2). Not re-taught.
- **Warping with inverse mapping and bilinear interpolation** (`images/`, day 1): recalled in
  unit 2 (warp frame 2 towards frame 1 by the current flow estimate) and unit 4 (apply a
  deformation field). One line each.
- **Least squares / normal equations** (linear algebra; used in `stitching/` math, day 23):
  recalled in the unit 2 math slide.
- **Regularisation: data + λ·penalty, choosing λ, early stopping** (`regularise/`, day 5; owner
  of regularisation with continuous unknowns): unit 3 recalls "the pattern" slide and the
  roughness penalty row of its table; Horn–Schunck is that pattern with a vector field as the
  unknown. Quadratic vs total-variation smoothing of a field: `gmrf/` (Ising day) as a
  pointer only, since that day may not have been taken.
- **Rigid shifts, phase correlation, block-wise motion estimation** (`phasecorr/`,
  `stitching/`, day 23): this day starts where `stitching/`'s "Limits: rigid only" slide stops.
  Unit 1 recalls "one shift for the whole image"; PIV (optional unit 5) is phase correlation per
  window.
- **Gradient descent** ("walk downhill"): owned by `backprop/` (day 36), which may come later
  than this day. Unit 4 explains it self-contained in one concept slide (a ball on the
  similarity landscape) with a forward pointer, as `snakes/` does.
- **Entropy** of a histogram (`entropy/`, compression day; may not have been taken) and the
  **two-channel scatter plot / joint histogram** (`coloc/`, day 16): unit 4 defines entropy in
  one slide self-contained ("how many yes/no questions to guess the value") and recalls the
  scatter plot from `coloc/`.
- **B-splines** (splines day, requested, not yet planned; course map "Ownership changes":
  the splines day owns spline theory and non-rigid registration recalls it): unit 4 shows the
  cubic B-spline only as "a smooth bump per control point" with a pointer; nothing depends on
  the splines day.

## Overlaps with other days (who owns what)

| Topic | Owner | Here |
|---|---|---|
| Brightness constancy, optical flow constraint, aperture problem, Lucas–Kanade, Horn–Schunck, coarse-to-fine flow, flow colour coding, endpoint/angular error, warping residual | **this day** (proposed for the owners table) | units 1–3 |
| Structure tensor, Harris/Shi–Tomasi corners, orientation and coherence | edges day, `structure-tensor/` | unit 2 recall slide; the LK matrix is $J$, the "good features to track" are its min-eigenvalue maxima: one slide, no re-derivation. 1D flow as kymograph orientation (unit 1) uses the space–time structure tensor in words, pointer to the doubled-angle slide |
| Gaussian pyramid, scale space | edges day, `scale-space/` | unit 2 recall, used for coarse-to-fine |
| Warping, bilinear interpolation, affine maps, homogeneous coordinates | `images/` | units 2 and 4 recall |
| Rigid translation by correlation; NCC; template matching; sub-pixel peaks; drift correction; one-line PIV mention | FFT registration, `matching/`, `phasecorr/`, `stitching/` | NCC as a similarity measure is **recalled** in unit 4 (owner day 23); rigid pre-alignment before non-rigid = day 23's tools; optional unit 5's PIV is windowed phase correlation (recall `phasecorr/`, sub-pixel from `stitching/`) |
| Regularisation of inverse problems (data + λ·penalty, λ choice, early stopping) | deconvolution, `regularise/` | unit 3: Horn–Schunck as that pattern; unit 4: bending energy as the penalty. Recall, not re-derived |
| Energy minimisation over labels; data + λ·smoothness | graph cuts, `seg-energy/` | one-line pointer in unit 3 (same structure, labels vs continuous vectors) |
| Quadratic vs TV smoothing (GMRF), robust penalties | Ising day, `gmrf/` | unit 3: robust penalty for sharp motion boundaries as a pointer + demo toggle, not taught |
| Spline theory: B-spline basis, Cox–de Boor, knots | splines day (not yet planned) | unit 4 recalls the cubic B-spline as a bump with values ⅙, ⅔, ⅙; pointer. When the splines day is planned it may take the FFD demo as an application (course map: "free-form-deformation non-rigid registration" listed under its image uses) |
| Landmark registration (Procrustes), thin-plate splines | snakes day, `shape-models/` (Procrustes) | unit 4 contrast slide: landmark-based vs intensity-based non-rigid; thin-plate splines as a pointer (splines day) |
| Mutual information as a similarity measure; joint histogram for registration | **this day** (proposed; no current owner) | unit 4. Entropy itself: `entropy/` (compression day) recalled if taken; self-contained definition here |
| Joint histogram / scatter plot of two channels | `coloc/` (day 16; moves to the colocalisation day per "Ownership changes") | unit 4 recall: "the coloc scatter plot, now used to *align*, not to measure overlap" |
| Gradient descent | `backprop/` (day 36) | unit 4: one self-contained slide, forward pointer |
| Feature tracking (KLT: track Shi–Tomasi corners by LK), linking detections, Kalman, lineages, MSD | tracking day (Motion group, not yet planned) | unit 2 ends with one pointer slide: "LK at corners = tracking points; linking objects over time is the tracking day" |
| Learned flow (correlation volumes + recurrent refinement), learned registration | modern deep learning day (day 39) | one pointer line in unit 3 summary ("networks trained on synthetic flow; same evaluation by EPE") |
| Topic page "Registration and alignment" (planned) | topic pages | this day supplies the non-rigid and flow entries |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit. Every lecture
follows README.md "Marking the math layer": concept slides with the demo first, then
`## The math: … {.math}` slides; the "About this lecture" slide ends with the standard line.
Exercises: concept questions first with `Solutions`, then `### Math questions` tagged *(math)*
with `Solutions (math)`. Folder names checked against the repo: none exist yet.

### 1. `flow-constraint/`: Motion between frames: brightness constancy and the aperture problem

- Motivation with pictures (time-lapse, honest captions): a migrating cell sheet in a wound
  assay, cytoplasmic streaming, a beating heart tube, a drifting-and-deforming tissue. Day 23
  found **one** shift for the whole image; here every pixel gets its own vector: a **flow field**.
- **Showing a flow field**: arrows (subsampled; length = speed), the **colour wheel** (hue =
  direction, saturation/brightness = speed, with the wheel as legend), the speed map. One slide
  of reading exercises: translation, rotation, expansion, shear, a vortex, two cells moving apart.
  Colour-wheel caveat: saturation is relative to the maximum speed shown (state the scale).
- **Brightness constancy**: a moving point keeps its grey value. When it holds (fixed
  illumination, in focus, short frame interval) and when it does not (bleaching, blinking
  fluorophores, objects leaving the focal plane, shading changes on a rotating 3D object).
- **1D first**: a profile moving right by $u$ px per frame. At a pixel on a slope, the grey value
  drops by slope × $u$ from frame 1 to frame 2, so $u = -I_t / I_x$. On a flat stretch ($I_x = 0$):
  no information. On a peak: same. A window of pixels, combined by least squares, averages the
  good ones. (By-hand example below gives exactly 1.0 from five pixels where single pixels say
  0.5, undefined and 1.5.)
- **Kymograph**: stack the profile over time into an $x$–$t$ image; a moving feature draws a
  **line**, its slope is the speed, crossing lines are features passing each other, a curved line
  is acceleration. 1D flow is the **orientation** of the lines in this image (pointer to
  `structure-tensor/`: orientation from the structure tensor, now in $x$–$t$). Biology: kymographs
  of vesicle transport along an axon, of a moving cell edge (protrusion/retraction).
- **The math: the optical flow constraint** {.math}: $I(x + u\,\delta t, y + v\,\delta t, t + \delta t) = I(x, y, t)$;
  first-order Taylor ⇒ $I_x u + I_y v + I_t = 0$, i.e. $\nabla I \cdot (u, v) = -I_t$. Valid when
  the motion is small compared with the scale over which the gradient changes (why large motion
  fails: unit 2). Normal flow $\mathbf u_\perp = -I_t\,\nabla I / |\nabla I|^2$. In 1D: $u = -I_t / I_x$;
  the window least-squares estimate $u = -\sum I_x I_t / \sum I_x^2$ is the orientation of the
  space–time structure tensor $\begin{pmatrix}\sum I_x^2 & \sum I_x I_t\\ \sum I_x I_t & \sum I_t^2\end{pmatrix}$.
- **Aperture problem** (2D): one equation, two unknowns. Looking at a straight edge through a
  small hole, you can only see the motion **across** the edge; motion along it is invisible.
  Picture: the barber pole; a bar moving diagonally seen through a circle. In velocity space
  $(u, v)$ each pixel's constraint is a **line**; all velocities on it explain that pixel. Two
  pixels with different edge orientations: two lines, one intersection = the true motion
  (intersection of constraints). Flat region: no line at all.
- **The math: the constraint line** {.math}: the set $\{(u, v) : I_x u + I_y v = -I_t\}$ is a line
  with normal $\nabla I$ at distance $|I_t|/|\nabla I|$ from the origin; its closest point to the
  origin is the normal flow. Two pixels: a 2×2 linear system, solvable iff the two gradients are
  not parallel (determinant ≠ 0): the aperture problem as rank deficiency.
- Where the next units go: combine constraints **locally** (Lucas–Kanade, unit 2) or **globally**
  with a smoothness prior (Horn–Schunck, unit 3).
- **Demo `aperture`** (new): a pattern (bar, grating, corner, textured blob, a cell crop) moves
  with a velocity set by dragging an arrow; seen through a circular hole whose radius is a slider.
  Left: the hole and the perceived (normal) motion arrows per sampled pixel; right: velocity
  space $(u, v)$ with each sampled pixel's constraint line, the true velocity as a dot and the
  least-squares intersection. Toggle "remove the hole". Presets: "bar through a hole: only the
  normal component", "the barber pole", "a corner: the lines cross", "texture: many lines agree",
  "flat patch: no lines", "noise: lines scatter around the truth". Labels ≥ 18 px; fits 1150×560.
- **Demo `kymoflow`** (new): a 1D profile (line through the micrograph, a bump, a ramp, several
  vesicle-like spots) moving over time; panels: frames 1 and 2 with $I_x$ and $I_t$ marked at a
  draggable pixel, the per-pixel estimate $-I_t/I_x$ (with holes where $|I_x|$ is small), the
  window least-squares estimate (window slider), and the **kymograph** with a draggable line to
  read a slope. Sliders: speed, noise, window, bleaching per frame, frame interval. Presets: "a
  ramp: exact everywhere", "a bump: single pixels disagree, the window is right" (the by-hand
  example), "too fast: the estimate collapses" (motion > half the bump width), "bleaching breaks
  constancy" (a bias in $u$), "two spots, two speeds: crossing lines", "transport: read the
  speeds from the slopes". `data-mode="kymo"` shows only the kymograph part (optional unit 5).
- **Demo `flowviz`** (new, small): synthetic flow fields (translation, rotation, expansion,
  shear, vortex, saddle, two cells moving apart, a wound closing) shown as arrows, colour wheel and
  speed map side by side; sliders for the field's parameters and arrow spacing; a "which field is
  this?" quiz mode (shows only the colour coding). `data-mode="divcurl"` adds divergence and
  curl maps (optional unit 5).
- By hand (concept): (a) the bump example: frame 1 $(0, 0, 1, 4, 1, 0, 0)$, frame 2
  $(0, 0, 0, 1, 4, 1, 0)$ (true $u = 1$); central differences in frame 1 for $I_x$, frame
  difference for $I_t$, at $x = 1 … 5$: $I_x = (0.5, 2, 0, -2, -0.5)$, $I_t = (0, -1, -3, 3, 1)$;
  per pixel $u = (0, 0.5, \text{undefined}, 1.5, 2)$; window least squares
  $u = -\sum I_xI_t/\sum I_x^2 = 8.5/8.5 = 1.0$. (b) Constraint lines: pixel A has
  $(I_x, I_y, I_t) = (1, 0, -2)$, pixel B $(0, 1, -1)$: draw both lines, read off $(u, v) = (2, 1)$.
  (c) Read five colour-wheel images (which field?). (d) A kymograph with lines of slope
  3 px/frame and −1 px/frame: speeds and directions; with a 2 s frame interval and 0.1 µm pixels,
  the speeds in µm/s (0.15 and 0.05).
- Math questions: derive the constraint by Taylor and state the neglected term; show the normal
  flow is the closest point of the constraint line to the origin; show the 1D window estimate
  minimises $\sum (I_x u + I_t)^2$.

### 2. `lucas-kanade/`: Lucas–Kanade: local least squares, the structure tensor, and pyramids

- Idea: assume the flow is **constant in a small window**; every pixel in the window gives one
  constraint line; take the velocity closest to all lines (least squares). Picture: a fan of
  constraint lines in velocity space and the best point.
- **Recall slide (`structure-tensor/`)**: $J$ = the window's covariance of gradients; ellipse
  picture; flat / edge / corner from the eigenvalues.
- **The LK matrix is the structure tensor.** The least-squares problem's 2×2 matrix is exactly
  $J$ (window = ρ, gradient scale = σ). So the edges day's map already says where flow can be
  measured: **corner/texture** (both eigenvalues large): reliable; **edge** (one small): only the
  normal component (aperture problem); **flat** (both small): nothing. Use the smaller eigenvalue
  as a confidence and mask where it is small (Shi–Tomasi "good features to track").
- **The math: Lucas–Kanade** {.math}: minimise $\sum_{\mathbf x \in W} w(\mathbf x)\,(I_x u + I_y v + I_t)^2$;
  normal equations
  $\begin{pmatrix}\sum wI_x^2 & \sum wI_xI_y\\ \sum wI_xI_y & \sum wI_y^2\end{pmatrix}\begin{pmatrix}u\\v\end{pmatrix} = -\begin{pmatrix}\sum wI_xI_t\\ \sum wI_yI_t\end{pmatrix}$,
  i.e. $J\,\mathbf u = -\mathbf b$; with Gaussian $w$ the sums are $G_\rho *$ products (all
  pixels at once by five blurs); solvable iff $\mu_2 > 0$; error amplification ∝ $1/\mu_2$ (the
  condition number of $J$). Weighted by $w$: a Gaussian window.
- Window size: small = noisy and more aperture problems; large = mixes different motions (motion
  boundaries, a cell crawling over a static background). The same σ/ρ trade-off as the edges day.
- **Large motion fails.** The linearisation holds only while the motion is small compared with
  the image's smoothness (rule of thumb: about one pixel at the scale of the gradients). Two cures,
  usually together: (a) **iterate with warping**: estimate, warp frame 2 back by the estimate
  (recall `images/`: inverse mapping + bilinear), estimate the remaining small motion, add;
  (b) **coarse-to-fine**: on the Gaussian pyramid (recall `scale-space/`) an 8 px motion is 1 px
  at level 3; estimate there, double the flow and warp at level 2, refine, …, down to full
  resolution. Picture: the pyramid with the flow at each level.
- **The math: iterative and pyramidal LK** {.math}: Gauss–Newton on
  $\sum_W (I_2(\mathbf x + \mathbf u) - I_1(\mathbf x))^2$: linearise around the current
  $\mathbf u_k$, solve $J\,\Delta\mathbf u = -\mathbf b(\mathbf u_k)$, update; at level $\ell$
  the motion is $\mathbf u / 2^\ell$, initialise level $\ell - 1$ with $2\mathbf u_\ell$. Failure:
  coarse levels blur away small objects moving differently from their surroundings.
- **Block matching** for comparison (one slide): try shifts per window and keep the best score
  (day 23's NCC/phase correlation per window, = PIV, optional unit 5): robust to large motion,
  integer unless sub-pixel fitted, slow; LK: sub-pixel, fast, small motion only.
- Pointer slide: tracking sparse points with LK at Shi–Tomasi corners (feature tracking); linking
  objects over time and lineages: the tracking day.
- **Demo `lkflow`** (new, main demo): frame 1 = a 128² crop of `cell256.png` (or a synthetic
  texture/nuclei scene from `lib/synth.ts`), frame 2 = frame 1 warped by a known synthetic flow
  (translation, rotation, zoom, shear, swirl, two regions moving differently) + seeded noise.
  Panels: frames (flicker toggle), estimated flow (colour wheel or arrows), confidence map
  ($\mu_2$), endpoint-error map against the truth with mean EPE. Click a pixel: its window's
  gradient cloud (as in `structtensor`), the constraint lines in velocity space, the LS point,
  the true point and the $J$ ellipse. Sliders: σ, window ρ, noise, motion size, confidence
  threshold, iterations (warping), pyramid levels 1–4 (side strip shows each level and its flow).
  Presets: "small shift: works", "8 px shift: fails", "the pyramid rescues it", "rotation",
  "membranes: aperture problem, low confidence", "flat background: nothing to see", "a motion
  boundary: the window mixes two motions", "noise: raise the window". Readouts ≥ 18 px.
- By hand (concept): (a) four pixels with $(I_x, I_y, I_t)$ = $(1, 0, -2), (1, 0, -2), (0, 1, -1), (0, 1, -1)$:
  $J = \mathrm{diag}(2, 2)$, $-\mathbf b = (4, 2)$, $(u, v) = (2, 1)$. (b) All four $(1, 0, -2)$:
  $J = \mathrm{diag}(4, 0)$, singular; $u = 2$, $v$ undetermined: which structure is this?
  (c) All four $(1, 1, -3)$ (a 45° edge): eigenvalues 8 and 0; the normal flow $(1.5, 1.5)$.
  (d) A 10 px motion with a pyramid of 4 levels: motion at each level (10, 5, 2.5, 1.25 px):
  at which level does LK start to work? (e) Which of five marked spots in a micrograph would you
  trust (nucleus interior, membrane, cell corner, background, textured cytoplasm)?
- Math questions: derive the normal equations; show that rotating the image rotates $J$ and the
  flow consistently; show that for a pure translation $\mathbf d$ of a quadratic image one LK step is
  exact; count the cost per pixel (five blurs + a 2×2 solve).

### 3. `horn-schunck/`: Horn–Schunck: global smoothness, and how to evaluate a flow field

- Idea: instead of a window, ask that the **whole field** fits the constraints **and** is smooth.
  Recall `regularise/` "the pattern": misfit + λ·penalty. Data term: each pixel's constraint
  (squared); penalty: how much neighbouring vectors differ (the roughness row of the
  `regularise/` table, now for $u$ and $v$). λ small: trusts the constraints (noisy, holes in flat
  regions); λ large: a smooth field that blurs across motion boundaries. Same structure as
  `seg-energy/` (labels there, vectors here): one-line pointer.
- **Filling-in**: in a flat region there are no constraints, but smoothness pulls the flow in
  from the edges and corners around it. Horn–Schunck gives a **dense** field, LK a field with
  holes and a confidence. The demo animates the iterations: the flow spreads like heat.
- **The update in words**: each iteration, every pixel (1) takes the **average** of its
  neighbours' flow, then (2) moves it **towards its own constraint line**, by a step that is large
  where the gradient is strong and small where it is weak or λ is large. Picture in velocity space.
- **The math: Horn–Schunck** {.math}: discrete energy
  $E(u, v) = \sum_p (I_x u_p + I_y v_p + I_t)^2 + \lambda \sum_{(p, q)} \big[(u_p - u_q)^2 + (v_p - v_q)^2\big]$
  over 4-neighbour pairs. Setting $\partial E / \partial u_p = \partial E / \partial v_p = 0$ with
  neighbour averages $\bar u_p, \bar v_p$ gives
  $(I_x^2 + 4\lambda)u + I_xI_y v = 4\lambda\bar u - I_xI_t$ (and the $v$ twin), solved per pixel by
  $u = \bar u - I_x\,\dfrac{I_x\bar u + I_y\bar v + I_t}{4\lambda + I_x^2 + I_y^2}$,
  $v = \bar v - I_y\,\dfrac{I_x\bar u + I_y\bar v + I_t}{4\lambda + I_x^2 + I_y^2}$:
  Jacobi iteration of a sparse linear system (continuous form: Euler–Lagrange,
  $I_x(I_xu + I_yv + I_t) - \lambda\nabla^2 u = 0$). Convex, unique minimum; convergence slow
  (information travels one pixel per iteration: hence filling-in takes many iterations; multigrid
  / coarse-to-fine as the fix).
- **Sharp motion boundaries**: the squared penalty hates jumps, so it blurs the boundary between
  a moving cell and the static background. Robust penalties (TV / Charbonnier: pay for the size of
  the jump, not its square) keep boundaries (pointer: `regularise/` table row "total variation",
  `gmrf/` on the Ising day); a robust data term ignores pixels where brightness constancy
  fails. Modern variational flow = robust data + robust smoothness + coarse-to-fine + warping;
  learned flow networks (pointer, day 39) are trained on synthetic sequences and evaluated the
  same way.
- **LK vs HS** table: local vs global; sparse + confidence vs dense; window size vs λ; fast
  closed form vs iterations; both need small motion (pyramids for both).
- **Evaluating a flow field**: with ground truth (synthetic warp of a real image, or a simulated
  sequence): **endpoint error** EPE $= |\mathbf u_{\text{est}} - \mathbf u_{\text{true}}|$ per
  pixel, its mean, and its map (errors concentrate at motion boundaries and in flat regions);
  **angular error** (angle between $(u, v, 1)$ vectors, the older measure; small for slow wrong
  flow, which is why EPE is preferred). Without ground truth (real data): the **warping
  residual**: warp frame 2 back by the flow, subtract frame 1 (small residual is necessary, not
  sufficient: a wrong flow in a flat region leaves no residual). Honest note: choosing λ by the
  EPE on a synthetic sequence that resembles the data; the discrepancy idea of `regularise/` as a
  recall.
- **The math: error measures** {.math}: $\mathrm{EPE} = \sqrt{(u - u_{gt})^2 + (v - v_{gt})^2}$;
  $\mathrm{AE} = \arccos\dfrac{u u_{gt} + v v_{gt} + 1}{\sqrt{u^2 + v^2 + 1}\sqrt{u_{gt}^2 + v_{gt}^2 + 1}}$;
  why AE penalises errors at slow pixels more.
- **Demo `hsflow`** (new): the same frame pairs and synthetic flows as `lkflow` (shared scenes
  from `lib/flow.ts`), plus a scene with a large flat region inside a textured ring (filling-in)
  and a cell moving over a static background (motion boundary). Controls: λ slider, play / step /
  reset of the iterations (animated, a few iterations per frame), quadratic vs robust smoothness
  toggle, pyramid on/off. Panels: flow (colour wheel / arrows), EPE map, LK result side by side
  (toggle), and a plot of mean EPE against λ (computed on a button, the `regpath`-style λ curve,
  recall `regularise/`) with the current λ marked; warping-residual map in `data-mode="evaluate"`.
  Presets: "λ small: noisy, holes", "λ large: boundaries smeared", "filling-in: watch it spread",
  "motion boundary: quadratic blurs, robust keeps", "LK vs HS", "choose λ by EPE", "residual
  without truth: the flat-region trap".
- By hand (concept): (a) one HS update at a pixel with $(I_x, I_y, I_t) = (2, 0, -4)$,
  neighbour average $(\bar u, \bar v) = (0, 0)$, $\lambda = 1$: $u = 0 - 2 \cdot (-4)/(4 + 4) = 1$,
  $v = 0$ (the constraint alone says $u = 2$; smoothness halves it); with $\lambda = 0$: $u = 2$;
  with $(\bar u, \bar v) = (2, 1)$: $u = 2$, $v = 1$ (already on the line: unchanged). (b) Predict
  the flow in the middle of a uniform disc that moves right, after 1, 10 and 1000 iterations.
  (c) EPE and AE for estimate $(2, 1)$ vs truth $(2.5, 1)$ (EPE 0.5) and for $(0.1, 0)$ vs
  $(0.6, 0)$ (EPE 0.5; AE much larger): which pixel looks worse in each measure? (d) LK or HS for:
  dense tissue flow in an embryo; sparse bright particles on black; a cell crawling over a
  textured static substrate.
- Math questions: derive the per-pixel update from the discrete energy; show the update leaves a
  field unchanged iff it is a stationary point; show $E$ is convex (a sum of squares of linear
  functions); continuous Euler–Lagrange equations.

### 4. `nonrigid-reg/`: Non-rigid registration: deformation fields, B-spline FFD and mutual information

- From flow to registration: optical flow assumes **small** motion between **similar** frames;
  registration aligns images that may differ a lot (two time points far apart, two animals to an
  atlas, a live image to a fixed-and-stained one, two stains on serial sections). Same output: a
  **deformation field** (a vector per pixel), but found by optimisation of a similarity score
  over a smooth family of deformations. Pipeline: rigid/affine first (day 23, `images/`), then
  non-rigid.
- **A smooth family: free-form deformation.** Put a coarse **grid of control points** over the
  image; each control point carries a displacement; the displacement of any pixel is a smooth
  blend of the nearby control points (each control point = a smooth bump, the cubic B-spline, 4
  control points wide in each direction). Grid spacing = how local the deformation can be;
  moving one control point changes only a local patch. Pointer: B-splines in depth on the splines
  day; landmarks + thin-plate splines as the landmark-based alternative (Procrustes for the
  rigid part: `shape-models/`).
- **The math: B-spline FFD** {.math}: $\mathbf d(\mathbf x) = \sum_{k,l} \mathbf c_{kl}\,\beta^3\!\big(\tfrac{x}{\Delta} - k\big)\,\beta^3\!\big(\tfrac{y}{\Delta} - l\big)$,
  $\beta^3(t) = \tfrac23 - t^2 + \tfrac12|t|^3$ for $|t| < 1$, $\tfrac16(2 - |t|)^3$ for
  $1 \le |t| < 2$, 0 beyond; at the knots $\tfrac16, \tfrac46, \tfrac16$, partition of unity
  (a constant shift of all $\mathbf c$ is a translation); linear in the coefficients, so the
  gradient of the cost with respect to $\mathbf c$ is a blur of per-pixel gradients.
- **Keeping it physical**: no folding (the map $\mathbf x \mapsto \mathbf x + \mathbf d(\mathbf x)$
  must not turn space inside out). The **Jacobian determinant** map: > 1 local expansion, < 1
  compression, ≤ 0 folding (forbidden). Regularise with a **bending energy** (penalise second
  derivatives of $\mathbf d$, λ weight; recall `regularise/`). Mention, pointer-level:
  diffeomorphic methods guarantee no folding by composing many small deformations; the demons
  algorithm (smooth the flow-like force each iteration) is Horn–Schunck-like.
- **Similarity measures** (what "aligned" means), from the picture of the **joint histogram**
  (recall the `coloc/` scatter plot): for each pixel, plot (value in fixed, value in moving).
  Aligned images give a **tight** pattern; misaligned a **cloud**.
  - **SSD**: the pattern should be the diagonal (same contrast; e.g. two frames of a time-lapse).
  - **NCC** (owner day 23, recall): the pattern should be a straight line (gain and offset).
  - **Mutual information**: the pattern should be **any** tight pattern (inverted contrast, two
    stains, two modalities, a non-monotone mapping). MI measures how much knowing one image's
    value tells you about the other's. Entropy in one slide: the number of yes/no questions
    needed to guess a value (recall `entropy/` if the compression day was taken).
  - Pitfalls of MI: needs enough pixels per histogram bin (bin count trade-off), sensitive to the
    overlap size (normalised MI as the fix: pointer), local optima.
- **The math: mutual information** {.math}: $p(a, b)$ from the normalised joint histogram,
  $H(A) = -\sum_a p(a)\log_2 p(a)$, $H(A, B) = -\sum_{a,b} p(a, b)\log_2 p(a, b)$,
  $\mathrm{MI} = H(A) + H(B) - H(A, B) = \sum p(a, b)\log_2\frac{p(a, b)}{p(a)p(b)} \ge 0$, with
  equality iff independent; $\mathrm{MI} \le \min(H(A), H(B))$, with equality when one image
  determines the other. Smooth histograms (Parzen/linear binning) make MI differentiable in the
  deformation.
- **Optimisation**: gradient descent on similarity + λ·bending energy (one self-contained slide:
  a ball rolling downhill on the cost landscape; forward pointer to `backprop/`); coarse-to-fine
  in **both** the image (pyramid) and the grid (start coarse, refine control points); local
  optima and capture range (start from a good rigid alignment).
- Applications (one slide): aligning an embryo time-lapse with tissue deformation, registering
  live imaging to fixed immunostaining, serial sections (3D reconstruction from slices), atlas
  registration of brains, correcting breathing/heartbeat motion in intravital imaging. Honest
  limits: registration hides real biology if the deformation is too free (it can "explain away" a
  growing structure): check with the Jacobian map and a held-out landmark error.
- **Demo `ffdreg`** (new, main demo): fixed image = a 128² crop of `cell256.png`; moving image =
  the fixed one deformed by a seeded smooth random field (and optionally re-mapped in intensity:
  identity, inverted, non-monotone "second stain"), with noise. Overlaid control grid (spacing
  slider: 4×4 … 16×16 knots) with **draggable control points** for a manual warp; button
  *optimise* runs gradient descent (animated) on SSD / NCC / MI + λ·bending energy, with grid
  refinement (coarse-to-fine) as a toggle. Panels: moving after warp, difference / checkerboard
  overlay, deformed grid, Jacobian determinant map (folding in a distinct colour), cost vs
  iteration; readout: mean displacement error against the known field. Presets: "drag one
  control point: a local bump", "optimise with SSD", "fine grid, λ = 0: folding", "inverted
  contrast: SSD fails, MI works", "coarse to fine", "too free: it explains away a growing blob".
- **Demo `jointhist`** (new, small): two images (crop of the micrograph and its intensity-mapped
  copy), a translation (and rotation) slider; panels: the overlay, the joint histogram (heat map),
  and SSD / NCC / MI curves against shift with the current shift marked; selectors: intensity
  mapping (identity, gain+offset, inverted, non-monotone), number of bins, noise. Presets:
  "aligned: a thin line", "misaligned: a cloud", "inverted: NCC −1, MI high, SSD worst",
  "non-monotone: only MI peaks at the truth", "too many bins: MI noisy", "tiny overlap: MI
  misleads".
- By hand (concept): (a) 1D FFD: control points every 10 px at $x = 0, 10, …, 40$ with
  displacements $(0, 0, 6, 0, 0)$: displacement at $x = 20$ is $6 \cdot \tfrac46 = 4$, at $x = 10$
  is $1$, at $x = 15$ is $6\,\beta^3(0.5) = 6 \cdot \tfrac{23}{48} \approx 2.9$. (b) Folding: the
  map $x + d(x)$ folds where $d'(x) < -1$; the steepest slope of this bump is
  $6/10 \cdot \tfrac23 = 0.4$: no folding; from which coefficient on does it fold (15)? (c) Joint
  histograms of 8 pixels: $A = (0,0,0,0,1,1,1,1)$, $B = 1 - A$: $p = \begin{pmatrix}0 & \tfrac12\\ \tfrac12 & 0\end{pmatrix}$,
  $H(A) = H(B) = H(A, B) = 1$ bit, MI = 1 bit, NCC = −1, SSD maximal; $B$ shifted so that the
  joint counts are $\begin{pmatrix}2&2\\2&2\end{pmatrix}$: $H(A, B) = 2$, MI = 0. (d) Which
  measure for: two frames of a time-lapse; a DAPI image to a brightfield image; H&E to an
  immunofluorescence section?
- Math questions: verify $\beta^3$ at the knots and partition of unity; show
  $\mathrm{MI} = H(A) + H(B) - H(A, B)$ equals the KL form; show MI is invariant to any
  one-to-one remapping of $B$'s grey values (with exact histograms) and NCC is not; derivative of
  the SSD cost with respect to one control coefficient (chain rule: image gradient × B-spline
  weight).

### 5. (optional / further reading) `flow-applications/`: Flow in biology: PIV, kymographs, divergence and curl

- **PIV-style analysis** (particle image velocimetry, borrowed from fluid mechanics, used for
  cytoplasmic streaming, collective cell migration, tissue flows): cut both frames into
  interrogation windows, find each window's shift by (phase) correlation (recall `phasecorr/`),
  sub-pixel by the peak fit (recall `stitching/`), validate (peak ratio, median test against the
  neighbours), interpolate the rejected vectors. Window size vs resolution vs reliability.
  PIV vs LK/HS: PIV for textures/particles with larger motion, differential methods for smooth
  images and sub-pixel motion.
- **Reading a flow field**: speed maps and their statistics; **divergence** (sources/sinks:
  expansion, contraction, tissue growth) and **curl** (rotation: vortices in streaming,
  rotating embryos); strain rate as a pointer; averaging over time (time-averaged flow and its
  variability); flow along a cell edge for protrusion/retraction.
- **Kymographs from a time-lapse**: draw a line (or a curved path along an axon), sample it in
  every frame (bilinear, `images/`), stack: speeds from slopes; the orientation measured
  automatically with the structure tensor in $x$–$t$ (recall unit 1). Pitfall: motion not along
  the line is invisible (the aperture problem again).
- **Integrating flow**: following a point through a sequence by chaining flows accumulates
  error (as day 23's "register to the previous frame"); pathlines vs streamlines (one line).
  Pointer: the tracking day for objects.
- **The math: divergence and curl** {.math}: $\nabla\cdot\mathbf u = u_x + v_y$,
  $\omega = v_x - u_y$; for a rotation $\mathbf u = \Omega(-y, x)$: div 0, curl $2\Omega$; for an
  expansion $\mathbf u = k(x, y)$: div $2k$, curl 0; computed by Gaussian derivatives
  (`canny/`).
- **Demo `piv`** (new): seeded particle images (spots from `lib/synth.ts`) advected by a vortex,
  shear or two-vortex flow (true field known), or two micrograph frames; window size and overlap
  sliders, method (phase correlation / NCC / LK per window), validation on/off; panels: vector
  field over the image, rejected vectors marked, EPE map. Presets: "small windows: noisy",
  "large windows: smooth but blurred vortex", "validation removes outliers", "PIV vs LK on a
  smooth image".
- Reuse: `flowviz` with `data-mode="divcurl"`; `kymoflow` with `data-mode="kymo"` (several
  spots at different speeds, drawn path).
- By hand: div and curl of three small fields given as 3×3 grids of vectors; a kymograph slope
  to µm/s; window size choice for particles 3 px apart moving 5 px per frame.

## Shared engineering

- **Reuse** (no local copies; demos/src/lib/README.md):
  - `lib/imageops.ts`: `gaussianDerivative`, `gradient`, `gaussianBlur`, `sampleBilinear`,
    `warp` (with a per-pixel map from the flow), `gaussianPyramid`, `pyrDown`/`pyrUp`, `resize`,
    `toRGBA`/`drawGrey`.
  - `lib/edges.ts` (owner edges day): `structureTensor(img, σ, ρ)` and `minEig` for the LK
    matrix and confidence (LK's right-hand side needs $G_\rho*(I_xI_t)$, $G_\rho*(I_yI_t)$:
    computed in `lib/flow.ts` with `gaussianBlur`, not by changing `structureTensor`).
  - `lib/linalg.ts`: `eigSym2`, `inv2`/`det2` for per-pixel 2×2 solves and the ellipse.
  - `lib/registration.ts` (owner day 23): `correlate2d`, `findPeaks`, `peakRatio`,
    `parabolaPeak2d`, `hann`/`taper2d` for PIV windows; `fourierShift2d` for exact sub-pixel
    test translations.
  - `lib/histogram.ts`: `histogram2d` for the joint histogram; `lib/entropy.ts`:
    `entropyFromCounts` for MI.
  - `lib/colour.ts`: `hsvToRgb` for the flow colour wheel, `diverging` for divergence/curl and
    Jacobian maps, `viridis` for EPE.
  - `lib/plot.ts`: `Plot` with `arrow` (velocity space, quivers on small grids), `line` (EPE vs
    λ, cost vs iteration, score curves), `heatmap` (joint histogram), `colorbar`.
  - `lib/cell.ts` (`loadCell`, show `CELL_CREDIT`), `lib/synth.ts` (`spots` for particles,
    `nuclei`, `checkerboard`, `addGaussianNoise`), `lib/random.ts` (`rng`, fixed seeds),
    `lib/ui.ts` (sliders, presets + reset, `autoMount`), `lib/gridui.ts` not needed.
- **New: `demos/src/lib/flow.ts`** (owned by this day; no DOM; runs under node): synthetic flow
  fields with ground truth (translation, rotation, zoom, shear, swirl/vortex, saddle, two regions,
  smooth random field from a seed); `warpByFlow(img, u, v)` (backward, bilinear); derivatives
  $I_x, I_y, I_t$ (Gaussian or Horn–Schunck's 2×2×2 averages); `lucasKanade(I1, I2, σ, ρ)` →
  $(u, v, \mu_2)$; `lkIterative` and `lkPyramid(levels, iters)`; `hornSchunckStep` /
  `hornSchunck(λ, iters, robust?)` (Jacobi; robust = Charbonnier by reweighting);
  `endpointError`, `angularError`, `warpResidual`; `flowToRGBA` (colour wheel, with
  `colourWheelLegend`), `quiver(u, v, step)` (arrow list for `Plot`); `divergence`, `curl`;
  `kymograph(frames, path)` (bilinear sampling along a polyline); `pivField(I1, I2, win, step,
  method)` with median-test validation (calls `lib/registration.ts`); 1D helpers for `kymoflow`
  (`flow1d`, window LS).
- **New: `demos/src/lib/deform.ts`** (owned by this day; no DOM): cubic B-spline `bspline3(t)`
  and derivative (a comment: move to the splines day's library when that day is planned, and
  re-export from here); `ffdDisplacement(coeffs, gridW, gridH, spacing, w, h)`;
  `jacobianDeterminant`; `bendingEnergy` + gradient; similarity measures with gradients wrt the
  coefficients: `ssd`, `ncc`, `mutualInformation(a, b, bins)` (linear-binned joint histogram via
  `histogram2d`-style binning, so it is differentiable); `registerFFD(fixed, moving, {measure,
  λ, spacing, levels, iters})` as a generator yielding each iteration (for animation);
  `randomSmoothField(seed)`; intensity remaps (inverted, non-monotone) for the "second stain".
  Names checked: no `flow.ts`, `deform.ts` or `spline.ts` exists in `demos/src/lib/`.
- **New demos** (names checked against `demos/src/`, none exist): `aperture`, `kymoflow`,
  `flowviz`, `lkflow`, `hsflow`, `ffdreg`, `jointhist`, optional `piv`.
- **CSS** (build-issues): `demo.css` loads late; prefix per-demo selectors with
  `.qb-demo[data-demo="lkflow"]` etc.; slide text ≥ 18 px including readouts and the colour-wheel
  legend; every demo fits 1150×560 on a slide (two image panels of 256 px + one plot, or the
  click-pixel velocity-space panel replacing a plot on slides via `data-panels`).
- **Performance** (measure in the browser; TypeScript first): LK at 128² is five Gaussian blurs
  + a 2×2 solve per pixel (≈ 5–10 ms); pyramidal LK with 3 levels × 3 warps ≈ 20–30 ms
  (borderline: recompute on slider release, not on every move, if measured slow). Horn–Schunck:
  one Jacobi sweep at 128² ≈ 0.3 ms; animate 10–20 sweeps per frame; the EPE-vs-λ curve is a
  button (≈ 15 λ values × 300 sweeps, < 2 s). FFD with MI at 128² and an 8×8 grid: one gradient
  step ≈ 10–20 ms (estimate); animate one step per frame. Move a kernel to Rust/Wasm only if a
  live recompute measures > ~30 ms (candidates: `hornSchunck` at 256², MI gradient). No WebGL.
- **Tests** (`demos/test/opticalflow-test.ts`, run by `make test`): the 1D bump example gives
  $u = 1.0$; the four-pixel LK examples give $(2, 1)$, singular, normal flow $(1.5, 1.5)$;
  `lucasKanade` recovers a Fourier-shifted sub-pixel translation (0.3, −0.6) of the smoothed
  micrograph to < 0.05 px in the interior; `lkPyramid` recovers an 8 px shift where single-level
  LK fails; the HS update reproduces the by-hand numbers and leaves an exact solution unchanged;
  HS energy decreases monotonically; EPE/AE by-hand numbers; `bspline3` knot values and partition
  of unity; FFD by-hand displacements (4, 1, 2.875); Jacobian determinant of the identity = 1;
  MI of $B = 1 - A$ = 1 bit, of the independent case = 0; `registerFFD` with SSD reduces the
  displacement error on the seeded field; divergence/curl of rotation and expansion. The
  numbers in exercise solutions come from these functions.
- **Images**: `cell256.png` (credit as everywhere); synthetic sequences with known flow from the
  micrograph and `lib/synth.ts`. Real time-lapses would help (open question).
- Follow `demos/README.md`: no network, fixed seeds, presets + reset, keyboard back to reveal.js,
  static figure in each deck/exercise as the print fallback; every demo on `demos.qmd` and the
  sidebar (a new section "Motion" → "Optical flow").

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `flow-constraint/`: lecture, the bump by hand, `kymoflow`, `aperture`, `flowviz` |
| 10:45–12:15 | Unit 2 `lucas-kanade/`: lecture, the four-pixel LS by hand, `lkflow` |
| 13:15–14:45 | Unit 3 `horn-schunck/`: lecture, one update by hand, `hsflow`, evaluation |
| 15:00–16:30 | Unit 4 `nonrigid-reg/`: lecture, FFD and MI by hand, `jointhist`, `ffdreg`; quizzes as wrap-up |
| (self-study) | Optional unit 5 `flow-applications/` (`piv`, divergence/curl, kymographs) |

## Open questions

- **Non-rigid registration: unit or day?** Planned here as one core unit (unit 4). If the
  Motion group grows, it could become its own day (FFD + demons + diffeomorphic + landmark/TPS +
  evaluation of registration), with this unit as its first. Decide with the splines day plan,
  which lists FFD registration among its image uses: proposal: the splines day keeps the spline
  theory and points here for the registration application (as the course map's "Ownership
  changes" says).
- **Ownership** (add to the coursemap owners table): optical flow (constraint, aperture, LK,
  HS, EPE, flow colour coding) and **mutual information** as a similarity measure → this day.
  Divergence/curl of a vector field: this day (optional unit 5) unless another day claims it.
- **Applications core or optional?** Kymographs (unit 1) and block matching (unit 2) are in the
  core; PIV, divergence/curl and chaining flows are optional unit 5. Should PIV be core for a
  biology-heavy cohort (swap with the robust-penalty material in unit 3)?
- Unit 3 is dense (HS + robust penalties + evaluation). If it overruns, split evaluation into
  the optional unit or shorten the robust-penalty part to one pointer line.
- **Pending edits to built units** (cross-links): `structure-tensor/` "Interest points" and
  "Uses" slides → point to `lucas-kanade/`; `stitching/` "Limits: rigid only" slide → point to
  `flow-constraint/` and `nonrigid-reg/`; `scale-space/` pyramid slide → "used for coarse-to-fine
  flow" (one line); `regularise/` "The pattern" slide's "returns … in the later days" → name
  Horn–Schunck; `coloc/` scatter plot → forward pointer to `jointhist`.
- **Real data**: a licensed time-lapse (migrating cell sheet or cytoplasmic streaming) and a
  pair for multimodal registration (e.g. live/fixed or two stains) would make `lkflow`/`piv` and
  `ffdreg` convincing; until then synthetic warps of `cell256.png` (honest captions).
- The tracking day (same group) should reuse `lib/flow.ts` for LK point tracking and decide
  whether KLT feature tracking is taught there or as a slide here.
- Programming language for the code parts is still open (see README).
- Student material names no software; flow toolboxes and registration packages stay in this
  plan's teacher notes only.

## References

- Horn & Schunck (1981), *Determining optical flow*, Artificial Intelligence.
- Lucas & Kanade (1981), *An iterative image registration technique with an application to stereo vision*, IJCAI.
- Shi & Tomasi (1994), *Good features to track*, CVPR.
- Bouguet (2000), *Pyramidal implementation of the Lucas Kanade feature tracker*, Intel technical report.
- Barron, Fleet & Beauchemin (1994), *Performance of optical flow techniques*, IJCV (angular error).
- Black & Anandan (1996), *The robust estimation of multiple motions*, CVIU.
- Brox, Bruhn, Papenberg & Weickert (2004), *High accuracy optical flow estimation based on a theory for warping*, ECCV.
- Sun, Roth & Black (2010), *Secrets of optical flow estimation and their principles*, CVPR.
- Baker, Scharstein, Lewis, Roth, Black & Szeliski (2011), *A database and evaluation methodology for optical flow*, IJCV (EPE, colour coding).
- Teed & Deng (2020), *RAFT: recurrent all-pairs field transforms for optical flow*, ECCV (pointer: learned flow).
- Rueckert, Sonoda, Hayes, Hill, Leach & Hawkes (1999), *Nonrigid registration using free-form deformations*, IEEE TMI.
- Viola & Wells (1997), *Alignment by maximization of mutual information*, IJCV.
- Maes, Collignon, Vandermeulen, Marchal & Suetens (1997), *Multimodality image registration by maximization of mutual information*, IEEE TMI.
- Pluim, Maintz & Viergever (2003), *Mutual-information-based registration of medical images: a survey*, IEEE TMI.
- Thirion (1998), *Image matching as a diffusion process: an analogy with Maxwell's demons*, Medical Image Analysis.
- Sotiras, Davatzikos & Paragios (2013), *Deformable medical image registration: a survey*, IEEE TMI.
- Raffel, Willert, Scarano, Kähler, Wereley & Kompenhans (2018), *Particle Image Velocimetry: A Practical Guide*, Springer.
- Vig, Hamby & Wolgemuth (2016), *On the quantification of cellular velocity fields*, Biophysical Journal.
- Unser (1999), *Splines: a perfect fit for signal and image processing*, IEEE Signal Processing Magazine (B-spline background).

## Build notes (2026-11-13)

Built: `flow-constraint/`, `lucas-kanade/`, `horn-schunck/`, `nonrigid-reg/` (lec, quiz, exercise each);
demos `aperture`, `kymoflow`, `flowviz`, `lkflow`, `hsflow`, `jointhist`, `ffdreg`; libs `lib/flow.ts`,
`lib/deform.ts`; test `demos/test/opticalflow-test.ts` (68 checks); figures by
`tools/figs/opticalflow/figs.py`. Not built: optional unit 5 `flow-applications/` and demo `piv`.
Implemented but not embedded: `kymoflow` `data-mode="kymo"` (hides the profile plots) and `flowviz`
`data-mode="divcurl"` (`divergence`, `curlScreen` in lib/flow.ts), for unit 5.

Deviations from the plan:

- **Horn–Schunck solver**: red–black (checkerboard) Gauss–Seidel instead of plain Jacobi, so every
  half-sweep is an exact minimisation and the energy never increases (tested); the robust option is
  Charbonnier smoothness by reweighting (majorise–minimise, also monotone). The math slide says so.
- **LK derivatives**: $I_x, I_y$ from the mean of the two frames (Gaussian derivative σ), $I_t$ = blurred
  frame difference; LK's window blur is cut at 3ρ. `lib/flow.ts` has a monomorphic local copy of the
  separable filter (`sepFilter`, `blur3`) for speed; `imageops.separable` was ~2× slower here.
- **lkflow by-hand membrane pixel** is (85, 114) (nuclear envelope; μ₂ = 2 % of μ₁), not a free choice.
- The plan's "flat background" spot on the micrograph is not flat (TEM background has texture), so the
  "flat" cases use a synthetic ring scene (`ringScene`) and the LK windows figure uses a synthetic
  rectangle.
- **FFD API**: `ffdDisplacement(grid, w, h)` with an `FFDGrid` object (not separate coeff arrays);
  `FFDRegistration` is a class with `iterate()` (not a generator); grid refinement is exact B-spline
  subdivision (`refineGrid`, tested). Bending energy = squared second differences of the coefficients
  / (Δ² · area). Similarity costs are normalised (SSD / var(F), 1 − NCC, −MI / H(F)) so one λ scale
  serves all three.
- **Folding preset**: folding needs a 6 px grid and noise 0.25 with λ = 0 (57 folded pixels); a single
  dragged control point on a 16 px grid folds only beyond 36 px (more than the 30 px drag limit).
- **Findings reported honestly in the units**: noise in the gradients biases least squares towards
  zero (aperture demo, noise 6: (0.84, 0.78) for truth (1, 1), not cured by a larger hole); NCC
  fails for inverted contrast as a cost (wants +1); coarse-to-fine made no difference on the demo's
  2.4 px field (both 0.08 px); MI with 128 bins peaks at the smallest overlap.
- Performance (node): one LK solve at 128² ≈ 18 ms, 4-level × 3-iteration pyramid ≈ 80 ms (coalesced
  per animation frame; Wasm candidate if it feels slow); one HS sweep ≈ 0.7 ms; one FFD step ≈ 17–20 ms.
- Pending cross-links in other days' units (not edited, ownership): see "Pending edits to built units"
  above; also `coloc/` → `jointhist`.
