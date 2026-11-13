# Plan: a day on active contours (snakes) and active shape models ("smart snakes")

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 30 of 51, Part VI "Segmentation as optimisation". Hard prerequisites: Graph cuts (day 27), Edges, blobs and ridges (day 2). Soft (see also): Splines and NURBS (day 6). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VI (snakes → level sets → dynamic programming; Ising/MRF); entry after day 27 (graph cuts). First day of its chain: level sets follows.

## Goal of the day

After the day a student can:

1. write the **snake energy** of Kass, Witkin & Terzopoulos (elasticity $\alpha$, rigidity $\beta$, image term) for a closed polygon, compute the internal force at one point by hand, and predict what changing $\alpha$ and $\beta$ does to the contour;
2. derive the force from the energy in two ways: by differentiating the **discrete** energy with respect to one point, and (gently) by the **Euler–Lagrange equation** $-\alpha\mathbf{x}'' + \beta\mathbf{x}'''' + \nabla E_\text{ext} = 0$, and see that both give the same stencil;
3. write the internal force as a **cyclic pentadiagonal matrix** $A$, take one **semi-implicit step** $(A + \gamma I)\mathbf{x}_{t+1} = \gamma\mathbf{x}_t + \mathbf{f}(\mathbf{x}_t)$, and explain via the eigenvalues of $A$ why it is stable where plain gradient descent is not;
4. explain the **capture range** problem and choose between a larger smoothing scale, a **balloon force** and **gradient vector flow** (GVF); name the limits of explicit snakes (initialisation, fixed topology, point bunching);
5. build a **point distribution model**: align landmark shapes with **Procrustes analysis**, compute the mean shape and the modes of variation by **PCA** (recalled from day 1, `pca/`), and generate plausible shapes $\mathbf{x} = \bar{\mathbf{x}} + \Phi\mathbf{b}$ with $|b_k| \le 3\sqrt{\lambda_k}$;
6. describe the **active shape model search** (profile match along normals, fit pose and shape parameters, clamp, repeat, coarse to fine), and say why it is more robust than a snake when the object class is known.

Why it fits this audience: almost every step is linear algebra they already have. The snake
matrix is a circulant matrix (eigenvectors are sampled cosines and sines, a direct link to *The
Fourier transform* as a change of basis); its stencil is the second-difference kernel from
*Filtering & convolution* and that kernel convolved with itself; Procrustes is a least-squares
similarity transform (the matrices of *Images and coordinates*, demo `matrix2x2`); a shape model is
a change of basis to the eigenvectors of a covariance matrix. The calculus of variations is the
one genuinely new tool, and it is introduced as "the discrete gradient, with the sum replaced by an
integral".

## Prerequisite gap

- **Calculus of variations** is new. Present the discrete energy first (a function of $2n$ numbers; its gradient is ordinary partial derivatives), then the continuous Euler–Lagrange equation as the limit, with one worked case ($\int \tfrac12 \alpha |\mathbf{x}'|^2\,ds \Rightarrow -\alpha\mathbf{x}''$, via integration by parts on one slide). No function spaces, no boundary-term discussion beyond "the curve is closed, so they vanish".
- **Covariance matrix and PCA**: taught on day 1 (`pca/`: 2D cloud, covariance $S$, eigenvectors); the second-moment matrix of a region came back on day 14 (`regionprops/`). Unit 3 recalls it in one slide and goes straight to $2n$ dimensions.
- **SVD** may not have been covered. In 2D the Procrustes rotation has a closed form (a $2\times 2$ least-squares problem, or complex numbers); SVD is named as the general-dimension tool (3D shapes) on one slide, with the $2\times2$ case computed in the demo.
- **Segmentation**: known from Part II (thresholding, day 12) and from graph cuts (day 27). Unit 1 opens with one slide "find the boundary of this nucleus": a threshold gives a ragged, leaky outline; we want a smooth closed curve that sticks to edges.
- **Image gradients and Gaussian smoothing** come from day 1 (`kernels/`) and day 2 (Gaussian derivatives at scale σ, `scale-space/`); one recall slide on $|\nabla(G_\sigma * I)|$ as an edge map.
- **Energy minimisation** (data + λ·smoothness, MAP view) is owned by day 27 (`seg-energy/`); this day recalls it (unit 1) and never re-derives it.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `snakes/`: Active contours as energy minimisation

- Picture first: a closed rubber band of beads, pulled by springs to its neighbours and attracted to edges. Demo before formulas.
- **Recall slide (day 27, `seg-energy/`):** same structure as the graph-cut energy: an image term pulls the curve to edges, an internal term keeps it smooth; here the unknown is a curve, not a labelling, and the weights are $\alpha, \beta$ instead of one λ (β is the course's only other use of the letter: snake rigidity).
- Contour as a polygon $\mathbf{x}_i = (x_i, y_i)$, $i = 0..n-1$, indices mod $n$ (closed). Energy
  $$E = \sum_i \tfrac12\alpha\,|\mathbf{x}_{i+1} - \mathbf{x}_i|^2 + \tfrac12\beta\,|\mathbf{x}_{i-1} - 2\mathbf{x}_i + \mathbf{x}_{i+1}|^2 + E_\text{ext}(\mathbf{x}_i).$$
  $\alpha$: elasticity (short is cheap; a snake alone shrinks to a point). $\beta$: rigidity (straight is cheap; corners cost). Image term $E_\text{ext} = -|\nabla(G_\sigma * I)|^2$: low on edges.
- Force on one point = minus the partial derivative: $\alpha(\mathbf{x}_{i-1} - 2\mathbf{x}_i + \mathbf{x}_{i+1})$ (pull towards the neighbours' midpoint) and $-\beta(\mathbf{x}_{i-2} - 4\mathbf{x}_{i-1} + 6\mathbf{x}_i - 4\mathbf{x}_{i+1} + \mathbf{x}_{i+2})$ (straighten). Recognise $[1, -2, 1]$ from *Filtering & convolution*, and $[1,-4,6,-4,1] = [1,-2,1]*[1,-2,1]$.
- Continuous version: $\mathbf{x}(s)$, $E = \int \tfrac12(\alpha|\mathbf{x}_s|^2 + \beta|\mathbf{x}_{ss}|^2) + E_\text{ext}(\mathbf{x})\,ds$; Euler–Lagrange $-\alpha\mathbf{x}_{ss} + \beta\mathbf{x}_{ssss} + \nabla E_\text{ext} = 0$: a **force balance**, internal vs external. Derive the $\alpha$ term by integration by parts; state the $\beta$ term by analogy.
- Gradient descent: $\mathbf{x} \leftarrow \mathbf{x} - \tau\,\partial E/\partial\mathbf{x}$ with the external force sampled bilinearly at each point. Show in the demo that it works for small $\tau$ and explodes (zig-zag) for slightly larger $\tau$: motivates unit 2.
- Behaviour to read from the demo: $\alpha$ too large → shrinks off the edge; $\beta$ too large → cannot follow corners; $\sigma$ small → stuck on noise; snake far from the object → does not move at all (capture range, unit 2).
- Pointers, not taught here: snake points are not spread evenly along the curve; the demo resamples every $k$ steps (toggle). Discrete search alternatives: the greedy snake (Williams & Shah) and the **dynamic-programming snake** (Amini et al.), which is optimised exactly over a window of candidate positions per point (covered on the dynamic programming day, later in this chain of days; one slide pointer here).
- **Demo `snake`** (new, main demo of the day): image panel 128×128 (synthetic: a noisy disk, a rounded square, a "U" with a deep concavity, two nearby blobs, a weak-edge blob; real: a crop of the cell micrograph). Click to place an initial circle (radius by drag) or draw a polygon. Sliders $\alpha$, $\beta$, step $\tau$ (or $\gamma = 1/\tau$), $\sigma$, number of points $n$; update rule *explicit / semi-implicit* (unit 2); external force *edge gradient / GVF* and balloon $\kappa$ with sign (unit 2); resample every $k$ steps. Overlays: external force arrows on a coarse grid, per-point internal (blue) and external (red) force arrows, energy split versus iteration. Play / step / reset. Side panel (toggle): the matrix $A$ (or $A + \gamma I$) as a heat map for the current $n$, with the cyclic corner entries highlighted. Presets: "fits the disk", "α too large: shrinks away", "β too large: rounds the square", "σ too small: stuck on noise", "too far: does not move", "τ too large: explodes", "concavity: cannot enter the U", "two blobs: one snake bridges them".
- By hand: a square of 4 points (and an 8-point octagon) with $\alpha = 1$, $\beta = 0$: internal force at each point, one gradient step with $\tau = 0.1$, check the square shrinks and stays a square; add $\beta = 1$ and recompute at one corner. Euler–Lagrange for a circle $\mathbf{x}(s) = r(\cos s, \sin s)$: the elastic force points inward with magnitude $\alpha r$.

### 2. `snake-matrix/`: The snake as a linear system, and better external forces

- Stack the $x$ coordinates into $\mathbf{x} \in \mathbb{R}^n$ (same for $y$, solved separately with the same matrix). Internal force $= -A\mathbf{x}$ with $A$ **cyclic pentadiagonal**, rows $[\beta,\, -\alpha - 4\beta,\, 2\alpha + 6\beta,\, -\alpha - 4\beta,\, \beta]$ wrapping around the corners. Write it out for $n = 6$.
- Properties visible by hand: $A\mathbf{1} = 0$ (translating the snake costs nothing), $A$ symmetric positive semidefinite. $A$ is **circulant**: a circular convolution, so its eigenvectors are the sampled cosines/sines $\cos(2\pi k i / n)$ (the DFT basis) with eigenvalues $\lambda_k = 4\alpha\sin^2(\pi k/n) + 16\beta\sin^4(\pi k/n)$. Low $k$ (overall shape) is barely penalised, high $k$ (wiggles) strongly; $\beta$ penalises high frequencies more steeply than $\alpha$.
- Explicit step $\mathbf{x}_{t+1} = (I - \tau A)\mathbf{x}_t + \tau\mathbf{f}$: mode $k$ is multiplied by $1 - \tau\lambda_k$; unstable once $\tau\lambda_\text{max} > 2$, $\lambda_\text{max} = 4\alpha + 16\beta$. That is the zig-zag explosion seen in unit 1.
- **Semi-implicit** (Kass et al.): internal force at the new position, external at the old: $(A + \gamma I)\mathbf{x}_{t+1} = \gamma\mathbf{x}_t + \mathbf{f}(\mathbf{x}_t)$, $\gamma = 1/\tau$. Mode $k$ multiplied by $\gamma/(\gamma + \lambda_k) < 1$ for any $\gamma > 0$: unconditionally stable for the internal part. $A + \gamma I$ is fixed while $\alpha, \beta, \gamma, n$ are fixed: factor (or invert) once, reuse every step; cost in words (banded/cyclic solve is $O(n)$).
- External forces and **capture range**: $-\nabla E_\text{ext}$ is only non-zero within a few $\sigma$ of an edge. Options:
  1. larger $\sigma$ (larger range, blurred and shifted edges; coarse-to-fine schedule);
  2. **balloon force** (Cohen 1991): $\kappa\,\mathbf{n}_i$ along the polygon normal, inflates through flat regions, can push through weak edges if too strong; normalised edge force so the edge can stop it;
  3. **gradient vector flow** (Xu & Prince 1998): diffuse the edge-force field into flat regions by minimising $\iint \mu|\nabla\mathbf{v}|^2 + |\nabla f|^2\,|\mathbf{v} - \nabla f|^2$: near edges $\mathbf{v} \approx \nabla f$, far away $\mathbf{v}$ is smooth. Iterate $\mathbf{v} \leftarrow \mathbf{v} + \Delta t\,[\mu\nabla^2\mathbf{v} - |\nabla f|^2(\mathbf{v} - \nabla f)]$ (the Laplacian kernel again). Pictures: arrows pointing into the "U" concavity, which the plain edge force never does.
- **Limitations** and where they lead: needs initialisation near the object; fixed number of points and fixed topology (two objects need two snakes; one snake cannot split); self-intersection; parameters per image. The implicit representation that removes the topology problem, and the edge-based (geodesic active contour) and region-based (Chan–Vese) level set models, are treated on the level set day (next in this chain of days); this unit shows one slide with the comparison and a pointer, not the methods. The other limitation, "any shape is allowed", leads to units 3–4.
- **Demo `snakespectrum`** (new, small): no image, internal forces only. A closed polygon of $n = 8$–$32$ points (draggable; presets "noisy circle", "square", "star"). Sliders $\alpha$, $\beta$, $\tau$; rule *explicit / semi-implicit*; step / play / reset. Panels: the matrix $A$ with numbers (for $n \le 12$) or as a heat map; bar plot of the eigenvalues $\lambda_k$ and of the per-step amplification $|1 - \tau\lambda_k|$ (explicit) or $\gamma/(\gamma + \lambda_k)$ (semi-implicit), with the line at 1; the contour's own Fourier coefficients per mode, so the student sees high modes die first. Presets: "α only: shrinks to a point", "β only: smooths but keeps size longer", "explicit at the limit", "explicit just beyond: explodes", "semi-implicit with huge τ: still stable".
- The main `snake` demo is reused here with the balloon / GVF toggles and the matrix side panel enabled (`data-forces="all"`, `data-matrix="on"`); presets "U: GVF enters the concavity", "balloon from inside", "balloon leaks through the weak edge", "far start: GVF captures it".
- By hand: write $A$ for $n = 6$, $\alpha = 1$, $\beta = 0$; check $A\mathbf{1} = 0$ and that $(1, -1, 1, -1, 1, -1)$ is an eigenvector with eigenvalue $4$; largest stable $\tau$ for the explicit step; amplification of that mode for the semi-implicit step with $\gamma = 1$. 1D GVF on 7 samples: one or two diffusion iterations by hand, see the force spread.

### 3. `shape-models/`: Landmarks, Procrustes alignment and point distribution models

- Motivation: a snake knows only "smooth"; a hand, a leaf, a vertebra or a left ventricle has a characteristic shape. Learn it from examples (Cootes & Taylor: "smart snakes").
- **Landmarks**: $n$ corresponding points on every training shape (anatomical points + evenly spaced points in between). A shape is a vector $\mathbf{x} = (x_1, y_1, \dots, x_n, y_n) \in \mathbb{R}^{2n}$. Correspondence is the hard, manual part; mention automatic methods only by name.
- **Shape** = what is left after removing translation, scale and rotation (Kendall). **Procrustes**: find $s, R, \mathbf{t}$ minimising $\sum_i |sR\mathbf{x}_i + \mathbf{t} - \mathbf{y}_i|^2$.
  - translation: centre both at the centroid;
  - 2D closed form: with $a = \sum (\mathbf{x}_i\cdot\mathbf{y}_i)/|\mathbf{x}|^2$, $b = \sum (x_i y'_i - y_i x'_i)/|\mathbf{x}|^2$, the best $sR = \begin{pmatrix} a & -b \\ b & a\end{pmatrix}$: a **linear** least-squares problem in $(a, b)$ (link: the similarity matrices of *Images and coordinates*);
  - general dimension: $R = UV^\top$ from the SVD of the $2\times2$ (or $3\times3$) cross-covariance, with a sign fix against reflections (Umeyama). One slide.
- **Generalised Procrustes** for a training set: align all to one shape, compute the mean, normalise it, re-align all to the mean, repeat until the mean stops changing (usually 2–3 iterations).
- **PCA, recalled** (one slide, day 1 `pca/`): a 2D cloud, its covariance $S$, eigenvectors = principal axes, eigenvalues (descending) = variances. Then straight to $\mathbb{R}^{2n}$: $S = \frac{1}{s-1}\sum_j(\mathbf{x}_j - \bar{\mathbf{x}})(\mathbf{x}_j - \bar{\mathbf{x}})^\top$, eigenvectors $\boldsymbol{\phi}_k$ = **modes of variation**, each one a whole-shape deformation where all landmarks move together.
- **Point distribution model**: $\mathbf{x} \approx \bar{\mathbf{x}} + \Phi\mathbf{b}$, keep $k$ modes explaining e.g. 95 % of the variance; $\mathbf{b} = \Phi^\top(\mathbf{x} - \bar{\mathbf{x}})$ (orthonormal basis: projection = change of basis, as for the DFT). Plausible shapes: $|b_k| \le 3\sqrt{\lambda_k}$ (or a Mahalanobis bound on $\mathbf{b}$). With fewer shapes than $2n$, at most $s - 1$ non-zero eigenvalues (mention the small Gram-matrix trick).
- Failure modes to show (the time freed by recalling PCA goes here and to Procrustes): wrong correspondence (one shape's landmarks shifted by one) produces a spurious mode; skipping alignment makes the first modes "rotation" and "size"; too few training shapes; a linear model straightening a bending finger (curved motion needs several linear modes).
- **Demo `procrustes`** (new): two landmark sets (triangle, a 6-point "L", a 12-point leaf outline), each point draggable; the second set can be rotated / scaled / moved as a whole by handles. Buttons step through *centre → scale → rotate* with animation, showing the current $a, b$ (scale $= \sqrt{a^2+b^2}$, angle $=\operatorname{atan2}(b, a)$), the residual sum of squares, and the $2\times2$ matrix. Toggles: allow reflection; scale off (rigid only). Mode *generalised*: 10 training shapes drawn in grey, iterate GPA step by step, mean shape in bold. Presets: "pure rotation recovers exactly", "noise: best fit is not exact", "mirror image: reflection or not", "one landmark mislabelled", "GPA on a set".
- **Demo `shapepca`** (new): a synthetic training set (shape family menu: nucleus/kidney outline, leaf, hand; 20–60 shapes from `lib/shapes.ts`, fixed seed, generated from a few hidden parameters + landmark noise + random pose). Panels: raw set, aligned set with per-landmark scatter, eigenvalue bar chart with cumulative variance; sliders $b_1..b_5$ limited to $\pm3\sqrt{\lambda_k}$ (toggle "unlimited" to see implausible shapes); animate one mode back and forth. Mode *project*: drag landmarks of a shape freely, see its closest model shape $\bar{\mathbf{x}} + \Phi\mathbf{b}$ (and the clamped one) update live: the projection step of ASM. Controls: number of training shapes, landmark noise, alignment on/off, "corrupt correspondence". Presets: "first mode of the hand: finger spread", "without alignment: rotation is a mode", "3 training shapes", "bad correspondence", "drag a finger: the model follows".
- By hand: align triangle $\{(0,0),(2,0),(0,1)\}$ to a rotated, scaled copy using the 2D formula; three tiny training "shapes" of 2 landmarks in 1D ($\mathbb{R}^2$): covariance, eigenvectors, the one mode; project a new shape onto the mode and clamp $b$.

### 4. `asm/`: Active shape model search

- Idea (Cootes, Taylor, Cooper & Graham 1995): like a snake, move each point towards a better image position; unlike a snake, then **constrain the whole point set** to an allowed shape.
- **Profile model**: at each landmark of each training image, sample the image along the normal ($k$ pixels each side), take the derivative and normalise; per landmark, the mean profile $\bar{\mathbf{g}}$ and covariance $S_g$. Fit quality of a candidate profile: Mahalanobis distance $(\mathbf{g} - \bar{\mathbf{g}})^\top S_g^{-1}(\mathbf{g} - \bar{\mathbf{g}})$ (the simplest version: sum of squared differences, or "strongest edge along the normal", shown first).
- **Search loop**:
  1. start with the mean shape at a rough pose;
  2. for each landmark, sample a longer profile ($m > k$ each side) and move to the best-matching offset;
  3. fit pose $(s, R, \mathbf{t})$ by Procrustes to the suggested points (unit 3);
  4. $\mathbf{b} = \Phi^\top(T^{-1}(\mathbf{y}) - \bar{\mathbf{x}})$, clamp each $|b_k| \le 3\sqrt{\lambda_k}$;
  5. repeat until few points move.
  Step 2 alone is a "snake without internal energy"; steps 3–4 replace $\alpha$/$\beta$ by learnt shape knowledge.
- **Multi-resolution**: a Gaussian pyramid (recalled from day 2, `scale-space/`), profile models per level; search coarse to fine. Larger capture range and faster, the same idea as large $\sigma$ for snakes.
- Comparison slide: snake vs ASM on the same image (occlusion, clutter, a missing edge segment: ASM fills it in from the model; an unusual shape outside the training set: ASM cannot follow it, the snake can). Applications: hand outlines, bones in radiographs, cardiac ventricles, faces, nuclei outlines in microscopy.
- **Demo `asm`** (new): a synthetic image rendered from a shape of the trained model with unseen parameters (from `lib/shapes.ts`: filled shape, blur, noise, optional clutter blobs and an occluding bar). Place the initial mean shape by dragging (move / scale / rotate handles). Step through the loop with one button per sub-step (*profiles → suggested points → pose → project → clamp*), or play. Overlays: normals with sampled profiles at a selected landmark (plot of the profile, the model mean and the cost per offset beside the image), suggested points, current model shape, $\mathbf{b}$ as bars against their $\pm3\sqrt{\lambda}$ limits. Controls: noise, number of modes $k$, profile length $k$ and search length $m$, match *strongest edge / Mahalanobis*, pyramid levels 1–3, constraint on/off (off = point-wise snake-like moves). Toggle "run the snake from the same start" (reuses `lib/snake.ts`) for side-by-side comparison. Presets: "clean image, good start", "noise: constraint saves it", "occlusion: model fills in", "far start: single level fails, pyramid works", "no shape constraint: points scatter", "shape outside the training set".
- By hand: a 9-sample profile along one normal and a 3-sample mean profile; slide, compute SSD per offset, pick the best; given $\Phi$ (orthonormal, 2 modes) and $\lambda$, project a suggested shape and clamp.

### 5. (optional / further reading) `aam/`: active appearance models and beyond

- Active appearance models (Cootes, Edwards & Taylor 1998): warp each training image to the mean shape (piecewise affine on a triangulation: link to *Images and coordinates*), PCA on the shape-free texture, a combined shape + texture model; fitting by a learnt linear map from texture residual to parameter update. Synthesised faces/organs as the picture.
- Pointers: 3D shape models of organs (meshes instead of polygons; marching cubes produces such meshes), automatic correspondence (minimum description length), shape priors inside level set and graph cut segmentation, and today's CNN landmark detectors that are often still regularised by a PDM.
- **Demo**: reuse `shapepca` with `data-mode="appearance"` on a synthetic family with texture (a cell with nucleus and graded cytoplasm): sliders for shape modes and texture modes, the synthesised image beside the shape.

## Shared engineering

- `demos/src/lib/snake.ts` (new): closed polygon type; resampling to equal arc length; cyclic pentadiagonal $A$ for given $\alpha, \beta, n$; solver for $(A + \gamma I)$ (precomputed dense inverse is fine for $n \le 200$; or cyclic banded LU); explicit and semi-implicit steps; outward normals; balloon force; bilinear sampling of a force field; GVF iteration on a grid. Used by `snake`, `snakespectrum` and the snake comparison in `asm`.
- `demos/src/lib/shapes.ts` (new): seeded synthetic shape families with built-in landmark correspondence (nucleus/kidney outline, leaf, hand with 5 fingers; each from a few hidden parameters, plus landmark noise and random similarity pose); rasterise a shape to a grey image with blur, noise, clutter and occluder (for `asm`, `snake` presets); training-set generator with a fixed seed so lecture and exercise show the same set.
- `demos/src/lib/linalg.ts` (owner: graph-cut day, which moves `jacobiEigen` there from `ncut/spectral.ts`): this day adds $2\times2$ SVD / closed-form Procrustes and generalised Procrustes; it uses the shared symmetric eigendecomposition (Jacobi, up to ~120×120; Gram trick when $s < 2n$) and mean/covariance. No second eigen routine.
- Gaussian blur, Sobel, Gaussian pyramid: the shared `demos/src/lib/imageops.ts` (filters owned by `kernels/`, Gaussian derivatives and pyramid by the edges day). Edge map $|\nabla(G_\sigma * I)|^2$ and its gradient computed there.
- Bar plots (eigenvalues, amplification factors, $\mathbf{b}$ against limits) and the profile plot: the shared `lib/plot.ts` (owner: morphology day; never in `ui.ts`). Matrix heat map with optional numbers: a small helper, also useful for the `pixelgraph`/`ncut` demos.
- Performance: a step of a 100-point snake is microseconds; GVF on 128×128 for ~100 iterations is a few ms, recomputed only when $\sigma$ or $\mu$ changes; PCA on 60 shapes × 2·40 coordinates is instantaneous. TypeScript throughout; no Wasm expected (measure against the ~30 ms rule). Animation pauses when the slide is not visible.
- Tests (`demos/test/snakes-test.ts`, run by the shared `make test`): $A\mathbf{1} = 0$ and eigenvalues match the closed form; semi-implicit step with internal force only shrinks a circle and never increases the internal energy; Procrustes recovers a known similarity transform exactly (and the reflection guard works); PCA of shapes generated from 2 hidden parameters has 2 dominant eigenvalues; ASM converges on a noise-free synthetic image from a 10 px / 10° offset start.
- Image: the cell micrograph `demos/src/fourier2d/cell256.png` (credit R. M. Hunt, CC BY 3.0, as in `make-image.R`), crop around one organelle/nucleus for `snake`. Check early that a snake gives a sensible outline on it; if not, use a fluorescence nuclei image with a clear licence (same open question as the level set plan; one shared choice). ASM uses synthetic images only (no licensed annotated training set needed).
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js (all dragging with the pointer), fits 1150×560 on a slide, labels ≥ 18 px, static fallback figures for print.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, internal forces by hand, `snake` |
| 10:45–12:15 | Unit 2 lecture, the matrix by hand, `snakespectrum`, `snake` with balloon/GVF |
| 13:15–14:45 | Unit 3 lecture, Procrustes and PCA by hand, `procrustes`, `shapepca` |
| 15:00–16:30 | Unit 4 lecture + exercise (`asm`); quizzes as wrap-up |

## Open questions

- How long is a "day"? Unit 2 combines the linear algebra of the update with external forces; if too dense, GVF could move to the optional unit (keep the balloon).
- How far into the calculus of variations? Proposal: discrete gradient first, Euler–Lagrange for the $\alpha$ term derived, $\beta$ term stated. Enough, or should the continuous derivation be dropped in favour of the discrete one only?
- The circulant eigenvalue analysis is the strongest link to linear algebra and to the Fourier unit, but it is a detour. Core or a "Where this leads" box?
- **Decided (course map):** PCA is owned by day 1 `pca/` (2D point cloud → covariance → eigenvectors); unit 3 recalls it in one slide and reuses its figure.
- Synthetic training sets only, or also a small real annotated set (e.g. hand outlines or nuclei with a clear licence)? Real data shows correspondence is hard; synthetic keeps the demo controlled.
- Should the ASM unit include a snake-vs-ASM comparison exercise, or keep the two halves of the day independent?
- Cross-references to keep consistent in both directions: the snakes-vs-level-sets slide (level set day: topology, GAC), the DP snake (dynamic programming day), shape priors (graph cut day 27 and level set day). **Decided (course map):** order inside the day is snakes → level sets → dynamic programming, so all three are forward pointers from here.
- Programming language for the code parts is still open (see README).

## References

- Kass, Witkin & Terzopoulos (1988), *Snakes: active contour models*, IJCV.
- Cohen (1991), *On active contour models and balloons*, CVGIP: Image Understanding.
- Amini, Weymouth & Jain (1990), *Using dynamic programming for solving variational problems in vision*, PAMI.
- Williams & Shah (1992), *A fast algorithm for active contours and curvature estimation*, CVGIP: Image Understanding.
- Xu & Prince (1998), *Snakes, shapes, and gradient vector flow*, IEEE Trans. Image Processing.
- McInerney & Terzopoulos (1996), *Deformable models in medical image analysis: a survey*, Medical Image Analysis.
- Blake & Isard (1998), *Active Contours*, Springer.
- Cootes & Taylor (1992), *Active shape models: "smart snakes"*, BMVC.
- Cootes, Taylor, Cooper & Graham (1995), *Active shape models: their training and application*, Computer Vision and Image Understanding.
- Cootes, Taylor & Lanitis (1994), *Active shape models: evaluation of a multi-resolution method for improving image search*, BMVC.
- Cootes, Edwards & Taylor (1998/2001), *Active appearance models*, ECCV / PAMI.
- Gower (1975), *Generalized Procrustes analysis*, Psychometrika.
- Umeyama (1991), *Least-squares estimation of transformation parameters between two point patterns*, PAMI.
- Dryden & Mardia (1998), *Statistical Shape Analysis*, Wiley.
- Davies, Twining, Cootes, Waterton & Taylor (2002), *A minimum description length approach to statistical shape modeling*, IEEE TMI.
