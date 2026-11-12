# Plan: a day on edges, blobs and ridges (Part I, day 2)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
Follows `plan/coursemap.md` (this day **owns** Gaussian derivatives, LoG/DoG, the Hessian,
the structure tensor, Canny and scale space) and `plan/review-organisation.md` §2.1
(the units `canny/`, `scale-space/`, `hessian/`, `structure-tensor/`).

## Goal of the day

After the day a student can:

1. smooth an image with a **Gaussian of width σ**, explain why it is applied as two 1D passes
   (**separable**), and why we **differentiate after smoothing** (derivatives amplify noise;
   smoothing then differentiating is one kernel, the **derivative of a Gaussian**);
2. compute the **gradient magnitude and direction** at a pixel, and run **Canny** by hand on a
   small patch: smooth, gradient, **non-maximum suppression**, **hysteresis**; predict what σ and
   the two thresholds do;
3. explain **scale space**: a structure exists only at a range of scales; blurring with a
   Gaussian removes structure but never invents it (1D); the image pyramid as a cheap scale
   space;
4. detect **blobs** with the **Laplacian of Gaussian** (or its cheap twin, the **difference of
   Gaussians**) over scales, explain why the response must be **scale-normalised** (× σ²), and
   read off the size: detected radius ≈ √2·σ; locate edges as **zero crossings** of the LoG;
5. form the **Hessian** per pixel and classify the pixel from its two eigenvalues as **blob,
   ridge/valley, saddle or flat**; explain what a **vesselness** measure (Frangi) rewards and
   suppresses;
6. form the **structure tensor** from gradients, read **orientation** and **coherence** from its
   eigenvectors and eigenvalues, and tell **edge, corner and flat** apart (Harris/Förstner as the
   pointer to interest points);
7. *(math)* derive $\partial_x(G_\sigma * I) = (\partial_x G_\sigma) * I$, the σ² normalisation and
   the √2·σ radius, the closed-form eigenvalues of a symmetric 2×2 matrix, and the structure
   tensor as the covariance of gradients.

Why it fits this audience: it is calculus (Taylor, partial derivatives, the second-derivative
test, the gradient) and symmetric 2×2 eigenproblems, applied per pixel. The students know all
the mathematics; what is new is that the image is a *noisy, sampled* function, so every
derivative has to come with a scale. It is also the day that gives Part I its first
"detection" results (edges, spots, lines), right after the purely linear material of day 1.

## Prerequisite gap

What day 1 already teaches (checked in the built units), and what this day recalls in one line:

- **`kernels/`** (built): 3×3 kernels, correlation vs convolution, linearity, finite differences
  from Taylor (slope $\tfrac12[-1\;0\;1]$, curvature $[1\;-2\;1]$), Sobel as a separable outer
  product (in the notes), gradient "points across the edge" and edge strength $\|\nabla I\|$,
  the Laplacian, the 1D **zero crossing** of the curvature (B9), "smooth first, then
  differentiate" and the LoG as one bullet each (slide "Noise and edges"), border rules. This
  day starts from that slide and makes it quantitative. Nothing is re-derived.
- **`fourier/`** (built): low-pass/high-pass, the Gaussian's frequency response is a Gaussian
  (math slide), separability of the 2D DFT, aliasing and "blur before downsampling". Recalled for
  the pyramid (unit 2) and once for "differentiating = multiplying by frequency, so it boosts
  the noise" (unit 1). `fourier/exercise.qmd` promises "scale space and the LoG come back": this
  day keeps that promise.
- **`pca/`** (new, day 1, after `fourier/`): covariance matrix of a 2D point cloud, its
  eigenvectors as the principal directions, the covariance ellipse. Unit 4 leans on it heavily:
  **the structure tensor is the covariance matrix of the gradient vectors in a window.** Unit 3
  recalls "symmetric 2×2 ⇒ real eigenvalues, orthogonal eigenvectors" from it and `images/`
  (`matrix2x2` demo).
- **Hysteresis thresholding: an ordering problem.** The Hough plan and the brief call it a
  "recall from the thresholding day", but the thresholding day is now **day 5**, three days
  *after* this one. Proposal: this day **introduces** hysteresis in its original context (it is
  Canny's invention, Canny 1986) as a two-slide concept ("strong pixels seed, weak pixels
  survive only if connected to a seed"; connectivity as a picture, not an algorithm), and the
  thresholding day's unit 3 **recalls** it and generalises it to any image (faint filaments),
  keeping its `hysteresis` demo and the 4/8-connectivity discussion. The flood fill itself stays
  owned by morphology/labelling. Needs a one-line change in `plan/thresholding-day.md`
  (open question 1).
- **Thresholding a response map** (unit 2: "keep LoG responses above t") is used only as "keep
  values above a number"; no histogram method needed.

## Overlaps with other plans and units (who takes what)

| Plan | What moves here | What stays there |
|---|---|---|
| `hough-day.md` unit 1 `canny/` | The whole unit: four steps, Canny's criteria (math), NMS with sub-pixel interpolation (math), the `canny` demo and its presets, the by-hand NMS patch | A **recall slide** + the edge-list output `{x, y, θ, strength}` that units 2–3 vote with; the freed hour splits Hough unit 3 into `hough-circles/` + `ght/` as that plan proposes |
| `skeleton-day.md` unit 4 | Hessian of the smoothed image, eigenvalue picture (tube vs blob vs background), the by-hand Hessian of $-x^2$ and $-(x^2+y^2)$, Frangi formula and its β, c, multiscale σ²-normalised maximum (basics, single worked filter) | Vesselness as an **application**: the `vesselness` demo on vessel trees (`lib/trees.ts`), junction failure modes, the pipeline vesselness → threshold → skeleton → graph, neurons/vessel measurements, 3D Frangi |
| `wavelets-day.md` unit 5 `cwt/` | LoG blob detection, scale-normalised LoG across scales, blob radius ≈ √2·σ (Lindeberg); its open question "who owns the LoG?" is answered | Mexican hat as a wavelet and the CWT/scalogram (recall: "the Mexican hat is the 1D LoG of `scale-space/`"); **à trous** spot detection (Olivo-Marin) and the `atrous` demo, whose "LoG at one σ" comparison mode reuses `lib/edges.ts` |
| `thresholding-day.md` unit 3 | Hysteresis is *introduced* here (see prerequisite gap) | Hysteresis generalised to faint structures, the `hysteresis` demo, connectivity |
| `randomforest-day.md` unit 4 / `svm-day.md` unit 4 / `integral-day.md` | Teaching of every Gaussian feature family (smoothed intensity, gradient magnitude, LoG/DoG, Hessian eigenvalues, structure-tensor eigenvalues). RF's "structure tensor: no owner, concept + math slide here" is resolved | The **feature stack** (`featureStack()`), its browser and the paint–train loop (RF owns); each family recalled as one picture with a pointer here. Integral: box filters as the fast approximation of Gaussians and DoG (centre–surround), SURF's box Hessian: one-line pointer back to unit 3 |
| `morphology-day.md` unit 3 | nothing | Morphological gradient (dilation − erosion). Unit 1 names it in one line as the non-linear contrast to the derivative ("edge strength without derivatives; comes on the morphology day") |
| `snakes-day.md` (ASM pyramid), `graphics3d-day.md` (mipmaps) | The **Gaussian pyramid** is owned by unit 2 | Both recall it in one line (`snakes`: "Gaussian pyramid (from *Filtering & convolution*)" should be repointed to `scale-space/`) |
| `fftregistration-day.md` / review-gaps gap 5 | Corners as interest points (unit 4, pointer) | Feature-based registration, if a motion/registration elective is written |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise. Order: edges first (unit 1 builds
the Gaussian derivative tools and gives the most visual payoff), then scale (unit 2 needs σ from
unit 1), then the two 2×2-matrix units (Hessian = second derivatives, structure tensor = products
of first derivatives).

### 1. `canny/`: Edges — Gaussian derivatives and Canny

Concept layer:

- Recall (one slide, from `kernels/` "Noise and edges"): derivatives amplify noise; smooth
  first. Picture: a noisy step profile, its finite difference (all noise), and the difference of
  the smoothed profile (one clean bump).
- **The Gaussian of width σ**: the bell curve as a kernel; σ in pixels; the kernel is cut at
  about 3σ (so σ = 2 needs 13 taps). Larger σ: less noise, but edges widen, corners round off and
  close edges merge. The 3×3 Gaussian of `kernels/` is σ ≈ 0.85.
- **Separability**: a 2D Gaussian = a 1D Gaussian along rows, then along columns. Cost per pixel
  $2k$ instead of $k^2$ multiplications (σ = 4: 50 vs 625). Recall that Sobel was already a
  smoothing column times a derivative row.
- **Derivative of a Gaussian**: smoothing then differentiating is one kernel (convolutions
  combine, recall). Its shape: a positive and a negative lobe. In 2D: $G_x$ is a Gaussian
  derivative along $x$ times a plain Gaussian along $y$, i.e. **Sobel's big brother with a
  tunable σ**. Picture of the kernels at σ = 1, 2, 4.
- **Gradient magnitude and direction** per pixel: $\|\nabla L\|$ and the angle
  $\theta = \operatorname{atan2}(L_y, L_x)$, shown as a colour wheel; the direction is
  perpendicular to the edge. Convention from `images/`: $y$ down, angles counter-clockwise as
  seen on screen. Worked example with two numbers ($L_x = 3$, $L_y = 4$: strength 5, 53°).
- Why a gradient image is not yet "the edges": a thick ridge of magnitude, noise everywhere.
- **Canny** in four steps, each with a picture and a one-line reason:
  1. smooth with $G_\sigma$; 2. gradient (magnitude and direction); 3. **non-maximum
  suppression**: keep a pixel only if it is at least as large as its two neighbours *along the
  gradient direction* (rounded to 0°, 45°, 90°, 135°): the ridge becomes one pixel wide; worked
  on a 5×5 patch with numbers; 4. **hysteresis** (introduced here, see prerequisite gap):
  $t_\text{high}$ seeds, $t_\text{low}$ keeps what is connected to a seed: faint parts of real
  edges survive, isolated noise does not.
- Choosing σ and the thresholds: σ too small → noise edges; σ too large → corners cut, close
  edges merge, T-junctions break; $t_\text{high}$ too high → gaps; $t_\text{low}$ too low →
  noise chains attached to real edges. The ratio $t_\text{high}/t_\text{low}$ of about 2–3 is a
  rule of thumb.
- Where this leads (one line each): Hough voting with the edge list and its directions (Hough
  elective); the morphological gradient as a non-linear edge strength (morphology day); edge
  maps as the external force for snakes (snakes elective).
- Pointer: image libraries call the pipeline "Canny".

Math layer (`## The math: … {.math}` slides):

- *Derivative of a Gaussian*: $\partial_x (G_\sigma * I) = (\partial_x G_\sigma) * I$
  (differentiate under the integral); $\partial_x G_\sigma(x) = -\tfrac{x}{\sigma^2}G_\sigma(x)$;
  separability $G_\sigma(x, y) = g_\sigma(x)\,g_\sigma(y)$, hence
  $\partial_x G_\sigma = g'_\sigma(x)\,g_\sigma(y)$.
- *Noise*: white noise of variance $s^2$ through a kernel $w$ has variance $s^2\sum w_k^2$;
  for $\tfrac12[-1\;0\;1]$ that is $s^2/2$ while the signal slope is unchanged, and the
  derivative of a Gaussian reduces it as $\propto s^2/\sigma^3$. In frequency: differentiation
  multiplies by $i\omega$ (boosts high frequencies), the Gaussian multiplies by
  $e^{-\sigma^2\omega^2/2}$ (recall `fourier/`), so the product is a band-pass.
- *Gradient direction*: the directional derivative $\partial_\mathbf{u} L = \nabla L\cdot\mathbf u$ is
  largest for $\mathbf u \parallel \nabla L$; hence NMS looks along the gradient.
- *Canny's three criteria* in words (good detection, good localisation, one response per edge)
  and the result that the optimal step-edge detector is close to the derivative of a Gaussian.
- *Sub-pixel NMS*: compare with the magnitude interpolated (bilinear, `images/`) at
  $\mathbf p \pm \nabla L/\|\nabla L\|$ instead of rounding the direction.

**Demo `gaussderiv`** (new, small, 1D): a profile of 256 samples with a step, a ramp, a thin
line and two close steps, plus seeded noise (presets also: a line profile across the
micrograph). Rows: signal; smoothed signal $G_\sigma * f$; finite difference of the raw signal;
derivative-of-Gaussian response; the kernel itself drawn to scale on the same x-axis. Sliders σ
(0.5–16, log) and noise. Toggle *order* 0 / 1 / 2 (smoothing, slope, curvature; order 2 shows
the zero crossing at the edge, as a forward pointer to unit 2). Readout: kernel taps, and
"multiplications per pixel: 2D direct vs separable" for the current σ. Presets: "finite
difference on noise", "σ = 2 cleans it", "σ = 8: the two close steps merge", "a thin line gives
two lobes", "curvature: zero crossing at the edge".

**Demo `canny`** (taken over from the Hough plan, extended with a step-through): a 128×128 image
(synthetic: a rectangle, a disc, two close parallel edges, a T-junction, a faint low-contrast
edge, seeded noise; real: `demos/src/fourier2d/cell256.png` downsampled, and the full 256×256 on
the exercise page). Stage tabs *input / smoothed / magnitude / direction (colour wheel) / after
NMS / after hysteresis*, plus *overlay on input*. Sliders σ, noise, $t_\text{low}$,
$t_\text{high}$ (both thresholds also drawn on a small histogram of the post-NMS magnitudes).
Click a pixel: inset with its 3×3 magnitude neighbourhood, the gradient arrow, the two neighbours
compared by NMS (highlighted), and the verdict kept/removed; in the hysteresis tab the inset says
seed / connected weak / isolated weak. Button *grow hysteresis step by step* floods from the seeds
one ring per click (or *play*). Toggle *sub-pixel NMS*. Presets: "σ too small: noise edges",
"σ too large: close edges merge, corner rounds", "NMS thins the ridge", "one threshold: gaps or
noise", "hysteresis closes the gap", "micrograph". Output (for the Hough demos): the edge list.

By hand (concept): (a) the 1D Gaussian for σ = 1 sampled at −2…2 (0.05, 0.24, 0.40, 0.24, 0.05)
applied to a noisy step, then the central difference, vs the central difference of the raw
signal; (b) gradient magnitude and direction (degrees) from $L_x, L_y$ at three pixels; (c) NMS on
a given 5×5 magnitude + direction patch; (d) hysteresis on a 6×6 grid with given thresholds;
(e) predict: two edges 4 px apart, σ = 1 vs σ = 4. *(math)*: (f) show
$\partial_x(G_\sigma * I) = (\partial_x G_\sigma) * I$ and write $\partial_x G_\sigma$; (g) noise
variance after $\tfrac12[-1\;0\;1]$ and after Sobel x for unit white noise; (h) sub-pixel NMS at
one pixel by bilinear interpolation.

### 2. `scale-space/`: Scale space, blobs and pyramids

Concept layer:

- **Structures exist at a scale**: a nucleus is a blob at σ ≈ 10 px and texture at σ ≈ 1 px; a
  leaf is an edge at one scale and a line at a coarser one. There is no "right" σ for a whole
  image. Picture: the micrograph at σ = 1, 2, 4, 8, 16.
- **Scale space**: the stack $L(x, y; \sigma)$ of all blurred versions. A 1D signal gives a 2D
  picture ($x$ against σ) in which edges and bumps trace lines that end where the structure
  merges into a coarser one.
- **Why the Gaussian** (intuition only): blurring should only *remove* structure, never create it.
  In 1D, Gaussian blurring never creates a new extremum or zero crossing as σ grows; blurring by
  σ₁ then σ₂ equals one blur by $\sqrt{\sigma_1^2 + \sigma_2^2}$ (the scales form a chain). Among
  smoothing kernels the Gaussian is the only one with both properties (stated; references for
  the curious). Demo toggle: the same signal blurred with a box kernel grows a spurious extremum.
- **Second derivatives and zero crossings**: the LoG $\nabla^2 L$ (recall `kernels/`) is
  positive on the dark side of an edge, negative on the bright side; its **zero crossings** mark
  edges (Marr–Hildreth). Compared with Canny: closed contours but more spurious ones and poor
  corners. One slide.
- **Blob detection**: a bright blob gives a strong negative LoG at its centre; the strongest
  response comes when the kernel's negative centre fits the blob. Use $-\nabla^2 L$ so bright
  blobs are positive (the "Mexican hat", pointer to the wavelets elective).
- **Why scale-normalise**: blurring flattens the image, so raw derivatives shrink as σ grows and
  the unnormalised LoG always prefers the smallest σ. Multiplying by σ² (σ per derivative) makes
  responses at different scales comparable: measure distances in units of σ. Then each blob's
  response, plotted against σ, has a maximum at its own scale: **automatic scale selection**.
- **Size from scale**: for a disc of radius $r$ the normalised LoG peaks at $\sigma = r/\sqrt2$,
  so a detection at scale σ is drawn as a circle of radius **√2·σ** (2D). Worked with numbers.
- **Detection = local maxima in $(x, y, \sigma)$** above a threshold (non-maximum suppression in
  3D: 26 neighbours), drawn as circles. Failure modes: touching blobs (one big or two small),
  elongated blobs (respond at the width, not the length), edges giving weak responses (the
  Hessian, unit 3, rejects them), uneven background (the LoG ignores constant and linear
  background: one advantage over thresholding).
- **Difference of Gaussians**: $G_{k\sigma} - G_\sigma \approx (k-1)\,\sigma^2\nabla^2 G$:
  already scale-normalised, and the blurs are needed anyway. Neighbouring levels of a blur stack
  give the DoG for free.
- **Image pyramid**: after blurring by σ the image has no detail finer than ~σ, so it can be
  **downsampled** (recall `fourier/`: blur before downsampling, or alias). **Gaussian pyramid**:
  blur, take every second pixel, repeat; total size 1 + ¼ + 1/16 + … = 4/3 of the image. **Laplacian
  pyramid**: the difference between a level and the upsampled next level (a DoG); the levels add
  back up to the image exactly. Uses: coarse-to-fine search (snakes/ASM, registration), fast
  large-scale filters, blending, mipmaps (graphics elective).
- Pointer: scale-invariant keypoints (SIFT) are DoG extrema in a pyramid plus an orientation
  from the gradient (one line).

Math layer:

- *Scale space as diffusion*: $L = G_\sigma * I$ solves the heat equation
  $\partial_t L = \tfrac12\nabla^2 L$ with $t = \sigma^2$; semigroup
  $G_{\sigma_1} * G_{\sigma_2} = G_{\sqrt{\sigma_1^2 + \sigma_2^2}}$ (product of Fourier
  Gaussians). Causality in words (Koenderink, Lindeberg), and the honest note that in 2D
  new local extrema *can* appear, so the uniqueness statement is about level sets / a weaker
  causality.
- *DoG ≈ LoG*: $\partial_\sigma G = \sigma\nabla^2 G$, so
  $G_{k\sigma} - G_\sigma \approx (k-1)\sigma\,\partial_\sigma G = (k-1)\sigma^2\nabla^2 G$.
- *Normalisation and size*: 1D box of half-width $a$: $\sigma^2 L_{xx}(0) \propto
  -\tfrac{a}{\sigma}e^{-a^2/2\sigma^2}$, maximal at $\sigma = a$; 2D disc of radius $r$ (flux of
  $\nabla G$ through the circle): $\sigma^2\nabla^2 L(0) = -\tfrac{r^2}{\sigma^2}e^{-r^2/2\sigma^2}$,
  maximal at $r^2/\sigma^2 = 2$, i.e. $r = \sqrt2\,\sigma$; a Gaussian blob of width $s$:
  maximal at $\sigma = s$. Why exactly $\sigma^{2}$ (γ-normalisation, $\sigma^{\gamma m}$ for
  derivative order $m$; γ = 1 for blobs): stated.
- *Pyramid*: Laplacian pyramid reconstruction is exact by construction (telescoping sum).

**Demo `scale1d`** (new, 1D scale space): a 1D signal (presets: bumps of widths 2, 6, 18 px; a
step and a ramp; a noisy line profile across the micrograph). Top: the signal and its blurred
version at the current σ; middle: the **scale-space image** (x horizontal, log σ vertical, 64
levels) of the chosen quantity: $L$, $\sigma L_x$, or $-\sigma^2 L_{xx}$, diverging colours, a
horizontal line at the current σ; extrema / zero crossings traced as curves. Click a position:
bottom panel plots the response against σ (normalised and raw on the same axes) with its
maximum marked and the predicted σ = half-width. Toggles: *normalise*, *kernel: Gaussian / box*
(the box grows a new extremum in the "box creates structure" preset; the example signal is found
and tested offline). Presets: "three bumps, three scales", "raw LoG prefers small σ",
"normalised: each bump picks its own σ", "edges and bumps as curves in scale space", "box kernel
creates structure", "noise lives at small σ".

**Demo `blobscale`** (new, main demo): a 256×256 image: synthetic seeded spots (Gaussian and
flat discs of radii 2–20 px, varying brightness, a few touching pairs, an elongated one, on an
uneven background with Poisson-like noise; ground-truth radii known) and `cell256.png`
(nuclei/nucleoli as blobs). Panels: image with detected circles (radius √2·σ, colour by scale);
the response map $-\sigma^2\nabla^2 L$ at the current σ; a small σ-plot for the clicked pixel
(raw vs normalised). Controls: σ slider (log), *normalise* toggle, *LoG / DoG* (ratio $k$
slider), *detect over all scales* (NMS in $(x, y, \sigma)$) with threshold, noise level, *show
pyramid* (side strip with the Gaussian pyramid levels and the Laplacian pyramid, and a
*reconstruct* button showing the zero error). Readouts: number of detections, and with
synthetic images hits / misses / false alarms against the truth. Presets: "one σ finds one
size", "raw LoG: only the small spots", "all scales, normalised", "radius = √2·σ (overlay
truth)", "touching blobs", "elongated blob", "DoG matches LoG", "uneven background does not
matter", "nuclei in the micrograph", "the pyramid".

By hand (concept): (a) a 1D signal with a box of width 6, normalised response values at σ = 1, 2,
3, 4, 6 read from a printed table/plot: which σ wins, what half-width does it give? (b) a blob
detected at σ = 5 px: radius? Diameter in µm at 0.2 µm/px? (c) DoG from two given blurred rows;
compare with the LoG row ×(k−1)σ²; (d) pyramid sizes and total cost for a 1024² image, and why
the blur comes before the downsampling; (e) predict what happens to the detections when the image
is enlarged ×2. *(math)*: (f) derive σ = a for the 1D box and $r = \sqrt2\,\sigma$ for the disc;
(g) $G_{\sigma_1} * G_{\sigma_2}$ via Fourier; (h) DoG ≈ $(k-1)\sigma^2\nabla^2 G$ from
$\partial_\sigma G = \sigma\nabla^2 G$.

### 3. `hessian/`: The Hessian — blobs, ridges and saddles

Concept layer:

- Recall the second-derivative test from calculus: at a point, the curvature in every direction
  is described by one symmetric 2×2 matrix. In an image: per pixel, at scale σ,
  $H = \begin{pmatrix} L_{xx} & L_{xy} \\ L_{xy} & L_{yy}\end{pmatrix}$, three second derivatives of
  a Gaussian (separable kernels, pictures at σ = 2).
- **Eigenvalues = the curvatures along the two principal directions; eigenvectors = those
  directions** (orthogonal, because $H$ is symmetric; recall `pca/`). Order
  $|\lambda_1| \le |\lambda_2|$. The landscape picture: the image as a height map, a hilltop, a
  ridge, a pass, a plain.
- **Classification** (bright structures on dark; signs flip for dark ones), as regions of the
  $(\lambda_1, \lambda_2)$ plane:
  both strongly negative → **bright blob**; $\lambda_2 \ll 0$, $\lambda_1 \approx 0$ → **bright
  ridge (line)**, eigenvector of $\lambda_1$ points **along** it; opposite signs → **saddle**;
  both ≈ 0 → **flat**; an edge gives a weak, mixed response (the Hessian is not an edge detector).
- Two numbers that do not depend on rotation: the **trace** $\lambda_1 + \lambda_2$ is the LoG of
  unit 2 (it cannot tell a blob from a ridge); the **determinant** $\lambda_1\lambda_2$ is large
  only for blobs (the "determinant of Hessian" blob detector; pointer: the box-filter version on
  the integral-image elective).
- **Vesselness** (Frangi, basics): reward pixels where $\lambda_2$ is strongly negative
  (structure present) and $|\lambda_1/\lambda_2|$ is small (line, not blob); two knobs: β (how
  strictly blobs are rejected) and c (what counts as noise). **Multi-scale**: compute at several
  σ, normalise by σ² (unit 2), take the maximum; the winning σ grows with the line width.
  Failure in one picture: junctions look blob-like and fade. Applications (vessel and neuron
  networks, skeletonising them, measuring) are the skeleton elective: one pointer slide.
- Ridges in practice: membranes in the micrograph, filaments, roads, fingerprints' ridges.

Math layer:

- *Taylor*: $L(\mathbf x + \mathbf h) \approx L + \nabla L^\top\mathbf h + \tfrac12\mathbf h^\top H\mathbf h$;
  the curvature in direction $\mathbf u$ is $\mathbf u^\top H\mathbf u$, extreme along the
  eigenvectors (Rayleigh quotient), with values $\lambda_1, \lambda_2$.
- *Closed form*: $\lambda_{1,2} = \tfrac12\big(L_{xx} + L_{yy} \mp
  \sqrt{(L_{xx} - L_{yy})^2 + 4L_{xy}^2}\big)$; orientation
  $\tfrac12\operatorname{atan2}(2L_{xy}, L_{xx} - L_{yy})$; trace and determinant are invariants.
- *Frangi (2D)*: $\mathcal V = 0$ if $\lambda_2 > 0$, else
  $\exp(-R_B^2/2\beta^2)\,\big(1 - \exp(-S^2/2c^2)\big)$, $R_B = \lambda_1/\lambda_2$,
  $S = \sqrt{\lambda_1^2 + \lambda_2^2}$; scale normalisation $\sigma^2 H$; for a Gaussian line
  profile of width $s$ the normalised $\lambda_2$ peaks at $\sigma = s$ (exercise).
- *3D* in one line: 3×3 Hessian, three eigenvalues: blob / tube / sheet.

**Demo `hessian`** (new): images: synthetic quadratic patches (editable $f = ax^2 + bxy + cy^2$
with three sliders, shown as a heat map with contours), a synthetic scene with a bright blob, a
bright line, a dark line, a saddle (two blobs and two gaps), an edge and noise; a seeded vessel
tree from `lib/trees.ts` (skeleton plan) if it exists, else a simple curved-line generator;
`cell256.png` (membranes as ridges). Panels: image; classification map (colours for bright blob /
dark blob / bright ridge / dark ridge / saddle / flat, with a threshold on $|\lambda|$) or
selected scalar map (λ₁, λ₂, trace = LoG, determinant, vesselness); the
$(\lambda_1, \lambda_2)$ plane with the class regions drawn and a scatter of all pixels (thinned).
Click a pixel: a glyph at that pixel (eigenvectors as two axes, lengths ∝ |λ|, colour by sign),
the matrix with numbers, the eigenvalues, the verdict, and its point highlighted in the scatter.
Controls: σ, *multi-scale* (σ range, shows the best-scale map), β and c (vesselness mode),
bright/dark. Presets: "hilltop: both negative", "ridge: one eigenvalue", "saddle", "an edge is
not a ridge", "the LoG cannot tell blob from ridge; the determinant can", "vesselness: blob
suppressed", "one σ misses the thick line", "multi-scale", "membranes in the micrograph".
The skeleton day's `vesselness` demo becomes a preset/`data-mode="vessel"` of this demo (open
question 4).

By hand (concept): (a) Hessian, eigenvalues, eigenvectors and verdict for $f = -x^2$,
$-(x^2 + y^2)$, $x^2 - y^2$, $-(x + y)^2$; (b) second derivatives from a 3×3 patch with
$[1\;-2\;1]$ and the cross kernel $\tfrac14[\,1\;0\;-1;\;0\;0\;0;\;-1\;0\;1\,]$, then the
eigenvalues and the class; (c) a numeric 2×2 Hessian: Frangi value for given β, c; (d) predict
the class at the centre, the rim and outside a bright disc. *(math)*: (e) the closed-form
eigenvalues; (f) $\mathbf u^\top H\mathbf u$ is extreme along the eigenvectors; (g) the scale
at which $\sigma^2\lambda_2$ peaks for a Gaussian line of width $s$.

### 4. `structure-tensor/`: Orientation, coherence and corners

Concept layer:

- Question: what is the dominant **orientation** in a small window (fibres, wood grain,
  fingerprints, muscle, collagen), and how sure are we?
- First try: average the gradients in the window. It **fails**: on a bright line the gradients
  on the two sides point in opposite directions and cancel. Fix: average something that does not
  care about the sign: $\nabla L\,\nabla L^\top$ (the gradient direction counted as an axis, not
  an arrow). Shown in the demo with the arrows and their sum.
- **The structure tensor** $J = \text{(local Gaussian average, width ρ) of }
  \begin{pmatrix} L_x^2 & L_xL_y \\ L_xL_y & L_y^2\end{pmatrix}$: two scales, σ for the
  derivatives (noise), ρ for the window (how large a region we summarise).
- **It is the covariance matrix of the gradient vectors in the window** (recall `pca/`): plot
  all gradients of the window as a point cloud; its ellipse is $J$. Elongated cloud → one
  dominant direction; round cloud → none or many; tiny cloud → flat.
- Reading it: the eigenvector of the larger eigenvalue $\mu_1$ is the dominant gradient direction
  (across the stripes); the orientation of the structure is perpendicular to it. **Coherence**
  $\big(\tfrac{\mu_1 - \mu_2}{\mu_1 + \mu_2}\big)^2 \in [0, 1]$: 1 = perfectly oriented,
  0 = isotropic. Display: hue = orientation, saturation = coherence, brightness = energy.
- **Edge, corner, flat**: one large eigenvalue = edge or oriented texture; **two large = corner**
  (gradients in two directions in the same window); both small = flat. Corner measures:
  Harris $\det J - k\,(\operatorname{tr}J)^2$, Förstner/Shi–Tomasi $\mu_2$ (the smaller
  eigenvalue). Corners are **interest points**: they can be found again in a shifted or rotated
  image, which is what feature-based registration and tracking match (pointer to the motion
  elective; SIFT/ORB-style descriptors named in one line).
- Contrast with the Hessian (one slide, a table): Hessian = second derivatives, signed, finds
  blobs and lines at their centre; structure tensor = squared first derivatives, unsigned,
  finds orientation and corners, needs a window ρ. Both are symmetric 2×2 matrices per pixel,
  drawn as ellipses.
- Uses: orientation maps of fibres, coherence-enhancing diffusion and anisotropic filtering
  (pointer), corners for registration, a feature channel for pixel classifiers (pointer to the
  random-forest day).

Math layer:

- *Why the outer product*: the squared directional derivative
  $(\nabla L\cdot\mathbf u)^2 = \mathbf u^\top\nabla L\nabla L^\top\mathbf u$; averaged over the
  window, $E(\mathbf u) = \mathbf u^\top J\mathbf u$ is the energy of change in direction
  $\mathbf u$; maximal along the first eigenvector (Rayleigh quotient), minimal along the second.
  Same algebra as PCA (largest variance direction).
- *Doubled angle*: with $\nabla L = r(\cos\phi, \sin\phi)$,
  $J_{xx} - J_{yy} = \langle r^2\cos2\phi\rangle$, $2J_{xy} = \langle r^2\sin2\phi\rangle$: the
  dominant orientation is $\tfrac12\operatorname{atan2}(2J_{xy}, J_{xx} - J_{yy})$, and opposite
  gradients add instead of cancelling.
- *Harris from the auto-correlation*: shifting a window by $\mathbf h$ changes it by
  $\sum (I(\mathbf x + \mathbf h) - I(\mathbf x))^2 \approx \mathbf h^\top J\mathbf h$; a corner is
  where this is large for every $\mathbf h$, i.e. $\mu_2$ large. $\det J = \mu_1\mu_2$,
  $\operatorname{tr}J = \mu_1 + \mu_2$.
- *Rank*: without averaging (ρ = 0), $J = \nabla L\nabla L^\top$ has rank 1: every pixel is an
  "edge"; the window is what makes corners visible.

**Demo `structtensor`** (new, main demo): images: synthetic textures (stripes with a slider for
angle and frequency; stripes whose angle rotates across the image; a concentric "fingerprint";
two textures with different orientations meeting at a boundary; a checkerboard and polygons for
corners; isotropic noise), `cell256.png`, and a real oriented texture if a licensed one is found
(open question 6). Panels: image with an **ellipse grid** overlay (one ellipse per 16 px cell,
axes from $J$); the orientation/coherence colour map; corner response (Harris or $\mu_2$) with
detected corners (local maxima above a threshold) as crosses. Click a pixel: the window drawn on
the image, its gradient vectors as arrows (sum arrow shown: it cancels on lines), the gradient
**scatter plot with its covariance ellipse** (the `pca/` picture), $J$ with numbers, $\mu_1$,
$\mu_2$, orientation, coherence, verdict edge/corner/flat. Sliders σ and ρ, Harris $k$, corner
threshold, noise; toggle *average gradients instead* (the naive method). Presets: "edge: one big
eigenvalue", "corner: two", "flat: none", "averaging gradients cancels on a line", "ρ = 0: every
pixel an edge", "ρ too small: noisy orientation; too large: boundaries blur", "rotating stripes:
the hue follows", "corners of a checkerboard", "micrograph: membranes oriented, nuclei
isotropic".

By hand (concept): (a) $J$ of a window whose four gradients are (1, 0), (−1, 0), (1, 0), (−1, 0):
average gradient vs $J$, eigenvalues, verdict; (b) the same for (1, 0), (0, 1), (−1, 0), (0, −1)
and for all zeros; (c) orientation and coherence from a given $J$ with numbers; (d) Harris
response for three given tensors (edge, corner, flat) with $k = 0.05$; (e) predict the ellipse
glyph on a corner, an edge, inside a nucleus. *(math)*: (f) show
$\mathbf u^\top J\mathbf u = \langle(\nabla L\cdot\mathbf u)^2\rangle$ and that its maximiser is
the first eigenvector; (g) the doubled-angle formula for the orientation; (h) the
auto-correlation approximation $\approx \mathbf h^\top J\mathbf h$.

## Shared engineering

Names follow the shared-infrastructure table in `plan/review-organisation.md` §3.

- **`demos/src/lib/imageops.ts`** (shared owner of linear filters; created by whichever plan is
  built first, this day if it comes first): `loadGrey()` for `cell256.png` (with downsampling);
  `gaussKernel1D(sigma, order: 0 | 1 | 2)` (truncation at ⌈3.5σ⌉, normalised so order 0 sums to
  1, order 1 has unit response to a ramp, order 2 to a parabola); separable convolution on
  `Float32Array` with border modes (replicate / zero / reflect; replicate default, consistent with
  `kernel3x3/conv.ts`); `gaussian(img, σ)`, `gaussDeriv(img, σ, ox, oy)` giving
  $L_x, L_y, L_{xx}, L_{xy}, L_{yy}$; `log(img, σ, normalised)`, `dog(img, σ, k)`; 1D versions of
  the same for `gaussderiv` and `scale1d`; `gaussianPyramid`, `laplacianPyramid`, `reconstruct`
  (`graphics3d` mipmaps and `snakes` ASM reuse them). The RF day's `featureStack()` calls these.
- **`demos/src/lib/edges.ts`** (new, this day; the table's "Hessian/structure tensor/NMS in
  `lib/edges.ts`"): gradient magnitude/direction; NMS (4-direction and sub-pixel bilinear);
  edge list `{x, y, theta, strength}` (used by the Hough demos); zero crossings of a response;
  `hessian(img, σ)` → per-pixel $(L_{xx}, L_{xy}, L_{yy})$ + eigenvalues/eigenvectors;
  `classifyHessian`; `frangi(img, sigmas, β, c, bright)` returning response and best-scale map;
  `structureTensor(img, σ, ρ)` → $(J_{xx}, J_{xy}, J_{yy})$, orientation, coherence;
  `harris(J, k)`, `minEig(J)`; `scaleSpaceStack(img, sigmas, fn)` and
  `extrema3D(stack, threshold)` for blob detection; local maxima in 2D for corners.
- **`demos/src/lib/linalg.ts`**: closed-form eigenvalues/eigenvectors of a symmetric 2×2
  (the table puts it here, **not** in `imageops.ts`; replaces `matrix2x2/geom.ts`
  `eigenDirections` for the symmetric case). If not yet created, this day adds the function.
- **Hysteresis**: `hysteresis(mag, tLow, tHigh, conn)` in `lib/threshold.ts` (thresholding
  plan's file); its flood comes from `lib/morph.ts` reconstruction or `lib/label.ts` when they
  exist, otherwise a BFS in `threshold.ts` (as that plan says). The `canny` step-through needs an
  **instrumented** variant that records the flood ring by ring.
- **Drawing**: ellipse glyphs, arrows and the $(\lambda_1, \lambda_2)$ scatter via `lib/plot.ts`
  (with the covariance-ellipse helper the `pca/` unit and GMM need); colour wheel for directions,
  diverging map for LoG/second derivatives, orientation–coherence HSV map in `lib/colour.ts`
  (`colormaps` section). Signed responses always with 0 = mid-grey (as in `kernel3x3`).
- **Synthetic images** (seeded, with ground truth): spots/discs of known radii on an uneven
  background, edges/corners/T-junction scene, quadratic patches, stripes/rotating stripes/
  fingerprint textures → `lib/synth.ts`. Vessel trees from `lib/trees.ts` (skeleton plan) when
  it exists.
- **Speed** (TypeScript first, measured): a separable Gaussian on 256×256 at σ = 8 (57 taps) is
  ~7 M multiply-adds, a few ms. The scale-space stack for `blobscale` (≈ 16 levels, DoG from
  consecutive blurs, incremental blurring by the semigroup) is computed once per image/noise
  change (target < 150 ms, with a "computing…" state), then the σ slider only indexes. Structure
  tensor = 2 derivative passes + 3 ρ-blurs. No Wasm expected; revisit only if the multi-scale
  Hessian at 256×256 exceeds ~30 ms per slider move (then cache per σ).
- **Tests** (`demos/test/edges-test.ts`, run by `make test`): order-0 kernel sums to 1; the
  order-1 kernel on a ramp of slope 1 gives 1 and order 2 on $x^2/2$ gives 1 (away from
  borders); separable = direct 2D convolution on random images; DoG ≈ (k−1)·LoG within a
  few %; scale-normalised LoG on a synthetic disc of radius r peaks within one scale step of
  $r/\sqrt2$; NMS on a smoothed disc leaves a 1-px ring (no 4-connected thick runs); Hessian
  eigenvalues of sampled quadratics match the closed form; structure-tensor orientation of
  rotated stripes within 1°; Harris maxima at the corners of a square; Laplacian pyramid
  reconstructs exactly.
- **Images**: `demos/src/fourier2d/cell256.png` (Wikimedia Commons, credit as in
  `make-image.R`) for all four units; synthetic images labelled as synthetic in captions.
- Follow `demos/README.md`: TypeScript, `make typecheck`, no network, fixed seeds, presets +
  reset, the `slider`/`button`/`checkbox` helpers, labels ≥ 18 px, 1150×560 on a slide
  (`blobscale` and `structtensor` use two large panels + a small side panel on slides, and all
  panels on the exercise page via `data-layout="full"`), static figures as PDF fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `canny/`: lecture (`gaussderiv`, `canny`), NMS + hysteresis by hand |
| 10:45–12:15 | Unit 2 `scale-space/`: lecture (`scale1d`), blob size by hand, `blobscale` |
| 13:15–14:45 | Unit 3 `hessian/`: lecture, Hessian of quadratics by hand, `hessian` |
| 15:00–16:30 | Unit 4 `structure-tensor/`: lecture, $J$ by hand, `structtensor`; quizzes as wrap-up |

## Open questions

1. **Hysteresis ownership**: the thresholding day is now day 5, after this day. Proposal above:
   introduce hysteresis here (Canny context), thresholding day recalls and generalises it. Needs
   agreement and a one-line edit of `plan/thresholding-day.md` (and of the Hough plan's
   "recall from the thresholding day").
2. **Is four units too much for day 2**, coming straight after day 1 (which grows to four units
   with `pca/`)? Fallback order of cuts: move the pyramid to a short optional slide; shorten
   Frangi to one slide (the skeleton elective has the full treatment); make Harris a pointer.
   The review's fallback (one merged `gaussian-derivatives/` unit after `kernels/`, Canny on the
   Hough day) is no longer needed now that the day exists.
3. **Unit 1 name**: `canny/` (as in the review and the Hough plan, keeps the `canny` demo name)
   or `edges/` (it now also owns Gaussian derivatives, separability and the gradient)? Proposal:
   folder `canny/`, deck title "Edges: Gaussian derivatives and Canny".
4. **One Hessian demo or two**: should the skeleton day's planned `vesselness` demo become a
   `data-mode="vessel"` of `hessian` (one shell, tree image, best-scale overlay), with the
   skeleton day keeping only `arbor`? Proposal: yes; needs agreement from the skeleton plan.
5. **Zero crossings / Marr–Hildreth**: one slide in unit 2 (proposal) or a preset in `canny`
   (zero-crossing edges vs Canny side by side)? Both are cheap; the preset may be the clearer
   comparison.
6. **Real images**: a licence-compatible oriented texture (collagen/SHG, muscle, wood, a
   fingerprint) for `structtensor`, and a real spot image (FISH / single-molecule) for
   `blobscale`; the wavelets plan has the same need, so one search for both.
7. **Median filter**: review-ordering C1 wanted the median (first non-linear filter) in a Part 1
   scale unit. It does not fit this day's theme; owner to be decided (`cameranoise/` on day 3,
   or morphology's rank filters). Not planned here.
8. **Evaluation of edge detectors** (precision/recall of edge pixels with a tolerance, Pratt's
   figure of merit): leave to the evaluation day as a pointer, or one exercise question here?
9. Edits elsewhere once this plan is accepted (not done here): `kernels/` "Noise and edges" and
   `fourier/exercise.qmd` forward pointers to this day; Hough unit 1 → recall; skeleton unit 4 →
   application; wavelets `cwt/` → pointer; RF/SVM/integral recalls; snakes' pyramid source.

## References

- Canny (1986), *A computational approach to edge detection*, PAMI.
- Marr & Hildreth (1980), *Theory of edge detection*, Proc. R. Soc. Lond. B.
- Koenderink (1984), *The structure of images*, Biol. Cybern. (scale space, causality).
- Witkin (1983), *Scale-space filtering*, IJCAI.
- Lindeberg (1994), *Scale-Space Theory in Computer Vision*, Kluwer; Lindeberg (1998), *Feature
  detection with automatic scale selection*, IJCV (scale-normalised derivatives, √2·σ).
- Babaud, Witkin, Baudin & Duda (1986), *Uniqueness of the Gaussian kernel for scale-space
  filtering*, PAMI.
- Burt & Adelson (1983), *The Laplacian pyramid as a compact image code*, IEEE Trans. Commun.
- Frangi, Niessen, Vincken & Viergever (1998), *Multiscale vessel enhancement filtering*, MICCAI.
- Harris & Stephens (1988), *A combined corner and edge detector*, Alvey Vision Conf.
- Förstner & Gülch (1987), *A fast operator for detection and precise location of distinct
  points, corners and centres of circular features*, ISPRS.
- Bigün & Granlund (1987), *Optimal orientation detection of linear symmetry*, ICCV (structure
  tensor, orientation).
- Lowe (2004), *Distinctive image features from scale-invariant keypoints*, IJCV (DoG pyramid;
  teacher background, one-line pointer).
