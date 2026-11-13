# Plan: a day on sparse representations

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.
Follows `plan/coursemap.md`, section "Queue, second batch" (**Sparse representations**: sparsity in
a basis, L1 vs L2, compressed sensing, sparse coding and dictionary learning, uses in
denoising/inpainting/deconvolution and fast MRI). Elective, group **Transforms**, entry after
**Wavelets**. This day **owns** (proposed, see "Proposed reorganisation"): the ℓ0/ℓ1/ℓ2 penalties
and the geometry of the lasso, soft thresholding as the **proximal step** of ℓ1 in a general
(non-orthonormal) problem, **ISTA/FISTA** and **greedy pursuit (OMP)**, **compressed sensing**
(incoherence, recovery guarantees in words, the phase transition), **CS-MRI**, and **sparse coding
/ dictionary learning** (MOD, K-SVD, patch-based restoration).

## Goal of the day

After the day a student can:

1. say what it means for a signal to be **sparse** or **compressible in a basis** (few large
   coefficients; sorted magnitudes that fall fast), pick a basis that fits a signal (spikes, Fourier/DCT,
   wavelets), and count how many coefficients carry 99 % of the energy from a sorted list;
2. compare the **ℓ2 penalty** (shrinks every coefficient, never to zero), the **ℓ1 penalty**
   (shrinks and sets small ones exactly to zero) and the **ℓ0 count** (what we want, but a
   combinatorial search), explain the **geometry of the lasso** (a diamond touches the error
   ellipse at a corner), and apply **soft thresholding** by hand; derive the lasso optimality
   conditions and λ_max = ‖Aᵀy‖_∞ *(math)*;
3. run one step of **ISTA** by hand (a gradient step on the data term, then soft thresholding),
   explain why it is "Landweber + shrink", what FISTA's momentum adds, and how **orthogonal
   matching pursuit** differs (greedy ℓ0, no shrinkage bias); write deconvolution and inpainting
   with a sparsity prior as one problem with different forward operators; derive ISTA as
   minimising a quadratic upper bound and state its rate *(math)*;
4. explain **compressed sensing**: a K-sparse signal of length N can be recovered from M ≪ N
   **incoherent** measurements (M of the order of K log(N/K)) by ℓ1 minimisation; tell **coherent
   aliasing** (regular undersampling: copies) from **incoherent aliasing** (random undersampling:
   noise-like) and say why only the second can be removed by thresholding; read a
   **phase-transition** diagram; state the null-space property / RIP in words *(math)*;
5. explain **fast MRI by compressed sensing**: MRI measures Fourier coefficients (k-space), scan
   time grows with the number of k-space lines, **variable-density random** undersampling plus a
   wavelet/TV-sparse reconstruction gives 2–8× shorter scans; predict the artefacts of regular,
   random and centre-only sampling; compute the fold-over of 2× regular undersampling *(math:
   aliasing as the IDFT of the sampling mask)*;
6. explain **sparse coding with a dictionary** of image patches, how a dictionary is **learned**
   (alternate sparse coding and atom update; MOD and K-SVD; k-means as the one-atom special case;
   PCA as the orthogonal, non-sparse cousin), and how overlapping patch codes **denoise** and
   **inpaint** an image; compare a learned dictionary with a fixed wavelet/DCT basis and with
   non-local means; derive the K-SVD atom update as a rank-1 SVD *(math)*;
7. say what a sparsity prior **destroys** (low-contrast texture, small faint objects, the contrast
   of edges through shrinkage) and what that means for measurements.

Why it fits this audience: the day is linear algebra with one non-smooth term. Sparsity is
change of basis (wavelets, DCT, PCA, all known); the lasso picture is a level-set ellipse meeting
a norm ball in 2D, which they can draw; ISTA is gradient descent (known from the neural network
and deconvolution days) plus the scalar soft threshold they derived on the wavelets day;
compressed sensing is an underdetermined system $A\mathbf x = \mathbf y$ (null spaces, known from
`illposed/` and `algebraic-recon/`) with a prior that picks one solution; K-SVD is an SVD. It is
also the day that ties together the "data + λ·penalty" thread of the course (deconvolution,
graph cuts, Ising/TV, tomography) under one non-quadratic penalty, and it is the classical
ancestor of the learned priors of the diffusion elective.

## Prerequisite gap

What earlier units already teach (checked in the built units), and what this day recalls in one
line:

- **`haar/`, `dwt2d/`, `shrinkage/` (Wavelets, the entry day)**: orthonormal wavelet bases, the
  "keep the K largest" curve and the heavy-tailed coefficient histogram (`dwt2d/` slides "How
  sparse is a real image?", "Keep the K largest"), hard vs soft thresholding, the universal
  threshold, and the math slide "soft thresholding minimises an ℓ1 energy" (orthonormal $W$, one
  scalar problem per coefficient), plus "A third view: a prior on the coefficients". **This day
  starts exactly there** and removes the word *orthonormal*: what if $A$ blurs, masks, undersamples,
  or the dictionary is redundant? The scalar soft-threshold derivation is recalled, not repeated.
- **`dct/` (Compression)**: the DCT basis, energy compaction, the KLT/PCA as the optimal *linear*
  basis. Unit 1 recalls it in one line (a second sparsifying basis); unit 5 uses the 8×8 DCT as the
  fixed starting dictionary and contrasts "best linear basis (KLT)" with "best sparse dictionary".
  Not a hard prerequisite (the compression day is optional within the group).
- **`fourier/` (day 1)**: the DFT as a change of basis, aliasing, the `fourier2d` "keep largest"
  mode. Units 3–4 recall aliasing in one slide each. MRI needs no physics beyond one concept slide
  "the scanner measures Fourier coefficients of the object, one line of k-space at a time".
- **`illposed/`, `deconv/`, `regularise/` (core days 4–5)**: forward model, null space,
  data + λ·penalty, Landweber iterations and early stopping, filter factors, MAP reading (Laplace
  prior = ℓ1, one line on the `regularise/` MAP slide). This day **recalls** the pattern in one slide
  and adds a non-quadratic penalty, a new solver and a new kind of forward operator
  (undersampling). ISTA's gradient step **is** the Landweber step of `regularise/`.
- **`gmrf/` (Ising elective, owns TV)**: TV keeps steps, staircases ramps. Here TV is recalled in
  one slide as **analysis sparsity** (ℓ1 of the gradient) next to **synthesis sparsity** (ℓ1 of
  coefficients). Not a prerequisite; the slide stands alone.
- **`algebraic-recon/` (Radon elective)**: few-view CT with non-negativity + TV (30 views, RMSE
  0.081 FBP vs 0.020 TV). Recalled in unit 3 as "you have already seen compressed sensing work"
  (soft prerequisite).
- **`pca/` (day 1)** and **`kmeans/` (Gaussian-mixture elective)**: unit 5 recalls PCA in one line
  (orthogonal atoms, all used) and **teaches k-means in one concept slide** as "a dictionary where
  each patch uses exactly one atom", because `kmeans/` is an elective and may not have been taken.
- **`denoise/` (day 5)**: Gaussian, median, bilateral, **non-local means** (patch similarity),
  method noise. Unit 5's comparison reuses these as baselines (`lib/denoise.ts`).
- **`cameranoise/` (day 3)**: noise σ and PSNR. Recalled in one line.
- **`neuralnet/` (day 15)**: gradient descent and step size. Recalled in one line in unit 2 (not a
  prerequisite; the elective can be taken after Part I). Learned ISTA and convolutional sparse
  coding are pointers only.

Nothing needs to be taught here that a prerequisite day should own.

## Overlaps with other plans and units (who takes what)

| Plan / unit | What moves here | What stays there |
|---|---|---|
| Wavelets `shrinkage/` (built) | The ℓ1 penalty **in general** (non-orthonormal $A$, redundant dictionaries), the lasso geometry, ℓ0/ℓ1/ℓ2 comparison, the proximal reading of soft thresholding, iterative solvers | Wavelet shrinkage as **denoising in an orthonormal basis** (hard/soft, universal threshold, MAD σ̂, cycle spinning) and its one ℓ1 math slide; it gets a one-line forward pointer to `sparsity/` |
| Wavelets `dwt2d/` (built) | nothing | "How sparse is a real image?" and "Keep the K largest": recalled as one slide (a figure reused) in unit 1 |
| Compression `dct/` (built) | nothing | DCT and KLT; recalled in unit 1 (basis) and unit 5 (DCT dictionary, KLT vs learned dictionary) |
| Deconvolution `regularise/` (built) | Sparsity-regularised deblurring as a worked application (unit 2), with its own 1D/2D presets | Regularisation in general, choosing λ (discrepancy, L-curve), early stopping, MAP; the `regpath` demo. Unit 2 recalls "data + λ·penalty", Landweber, and that choosing λ works the same way (no re-teaching) |
| Deconvolution `denoise/` (built) | nothing | The classical denoiser comparison and NL-means; unit 5 uses `lib/denoise.ts` NL-means as the baseline |
| Ising `gmrf/` (built, owns TV) | nothing | TV; recalled here as analysis-ℓ1 in one slide, and as the TV option in `csmri` and `sparserestore` (via `lib/smooth.ts` `tv2d`) |
| Radon `algebraic-recon/` (built) | nothing | Few-view CT with TV; recalled in unit 3 as an instance of compressed sensing |
| Diffusion elective `diffusion-inverse/` (built) | nothing | Learned priors for inverse problems, inpainting by diffusion, hallucination. Unit 4 and 5 end with a pointer: "sparsity is a hand-made prior; a diffusion model is a learned one" |
| Queued **Edge-preserving smoothing and PDEs** day (not planned) | Inpainting **as a sparse inverse problem** (masked forward operator; fixed basis in unit 2, patch dictionary in unit 5) | Diffusion-based and exemplar-based inpainting, TV flow, bilateral/NL-means in depth. See "Proposed reorganisation" |
| Spectral unmixing `endmembers/` (built) | nothing | Non-negative unmixing; sparse unmixing named in one line in unit 1's "where sparsity appears" slide |
| Neural networks `cnn/` (built) | nothing | Convolutional filters; unit 5 pointer: learned dictionary atoms look like first-layer CNN filters and Gabor/edge detectors (Olshausen & Field) |

## Units (one folder each: lec, quiz, exercise with #explore)

Five core units (units 1, 2, 3, 5 about 1.5 h each; unit 4 about 1 h) plus one optional unit.
Order: the penalty (unit 1) → how to solve with it (unit 2) → what it allows: fewer measurements
(unit 3) → the flagship application (unit 4) → learn the basis from data (unit 5).

Every lecture starts with "About this lecture" ending in *Slides titled **The math** give the
formulas and proofs; the ideas are complete without them.* Math slides are titled
`## The math: … {.math}`; concept slides and the summary stand without them. Figures: in each
unit below, slides marked **[fig]** carry an image next to text (target: about half of all
slides, counting demo slides); figure scripts in `tools/figs/<unit>/`, hand-written SVGs in the
unit folder.

### 1. `sparsity/`: Sparsity and the ℓ1 penalty

Concept layer (about 18 content slides, 4 of them math):

- **[fig] Recall** (one slide, `dwt2d/` + `dct/`): the cell image, its wavelet coefficients and
  the "keep the K largest" curve: 5 % of the coefficients keep most of the image. Figure reused
  from `tools/figs/dwt2d/`.
- **[fig] Sparse vs compressible**: exactly sparse (K nonzeros: spikes, a piecewise-constant
  signal in Haar) vs compressible (sorted magnitudes fall like a power law: real images). Figure:
  sorted coefficient magnitudes on log–log axes for a spike train, a step signal and the cell
  image, in three bases.
- **[fig] The right basis depends on the signal**: spikes are sparse in the pixel basis, sines in
  Fourier/DCT, steps in Haar, smooth-with-edges in Daubechies wavelets; nothing is sparse in
  every basis. Figure: a 3×3 grid (signal × basis), coefficients as stem plots.
- **Where sparsity appears in the course** (text slide, one line each with pointers): JPEG/JPEG
  2000 (`jpeg/`, `compress-science/`), wavelet shrinkage (`shrinkage/`), few-view CT (`algebraic-recon/`),
  spot images (a few bright pixels: `spot-filters/`), unmixing with few fluorophores per pixel
  (`endmembers/`).
- **[fig] Denoising as a question about a penalty** (recall `shrinkage/`): noisy coefficients $w$;
  find $c$ close to $w$ that is "simple". Three ways to say simple: few nonzeros (ℓ0), small
  sum of magnitudes (ℓ1), small sum of squares (ℓ2). Figure: the three scalar transfer curves
  $c(w)$: ℓ2 = a line of slope $1/(1+λ)$, ℓ1 = soft threshold, ℓ0 = hard threshold.
- **Numbers**: $w = (4, -0.5, 1.2, -2.5, 0.3, 3)$, λ = 1. ℓ2 (halving): $(2, -0.25, 0.6, -1.25,
  0.15, 1.5)$, no zeros; ℓ1 (soft, $t = 1$): $(3, 0, 0.2, -1.5, 0, 2)$, two exact zeros. ℓ2
  shrinks big coefficients most (in absolute terms); ℓ1 shrinks all by the same amount and
  kills the small ones.
- **[fig] Why ℓ1 makes zeros: the geometry** (the central picture of the day). Two unknowns,
  the data misfit as ellipses around the least-squares point; the penalty as a ball that grows
  (or, equivalently, a budget ‖x‖ ≤ s). The ℓ2 ball is round: it touches the ellipse anywhere,
  generically off the axes. The ℓ1 ball is a **diamond with corners on the axes**: the ellipse
  usually touches a corner, where one coordinate is exactly zero. Figure: side-by-side ℓ2 and ℓ1
  pictures with the same ellipses.
- **By hand in the picture**: least-squares point (3, 1), round contours, budget 2: the ℓ2
  answer is (1.90, 0.63); the ℓ1 answer is the corner (2, 0).
- **[fig] Try it: `lassogeom`** (demo slide).
- **ℓ0, and why not just count?** Counting nonzeros is what we mean by sparse, but minimising it
  means trying every support: $\binom{256}{8} \approx 4\cdot10^{14}$ supports for 8 nonzeros out
  of 256. ℓ1 is the closest **convex** penalty: one valley, no local minima, solvable by
  iterations (unit 2). ℓp with p < 1 is even pointier and sparser but not convex (one line;
  picture in the demo).
- **[fig] The lasso and its path**: with a general matrix $A$ (columns not orthogonal), minimise
  ½‖Ax − y‖² + λ‖x‖₁ ("lasso", statistics name, one line). As λ falls from large to small,
  coefficients **enter one at a time**; above λ_max = the largest correlation $|a_j^\top y|$ all
  are zero. Figure: the regularisation path (coefficients vs λ) for a 6-column problem, ℓ1 vs ℓ2
  (ridge = Tikhonov: all enter at once and only reach zero at λ = ∞).
- **Shrinkage bias** (one slide): ℓ1 pulls the surviving coefficients towards 0 by λ: edges lose
  contrast (recall `shrinkage/` "soft loses contrast"). Fixes in one line each: refit the chosen
  support by least squares ("debiasing"), or reweighted ℓ1.
- **[fig] A probabilistic reading** (recall `regularise/` MAP): ℓ2 = Gaussian prior, ℓ1 = Laplace
  prior (peaked at 0, heavy tails), which is the shape of the wavelet coefficient histogram from
  `dwt2d/`. Figure: the `dwt2d` histogram with a fitted Gaussian and a fitted Laplace density
  overlaid, log scale.
- **Analysis vs synthesis** (one slide, text + small figure **[fig]**): synthesis: the image *is*
  a few atoms, $x = Wc$ with $c$ sparse; analysis: some transform *of* the image is sparse, e.g.
  its gradient: ℓ1 of the gradient is **total variation** (owned by `gmrf/`): TV is sparsity of
  differences, which is why it makes piecewise-constant images. Figure: a step-and-ramp profile,
  its gradient (sparse for the step, not for the ramp).

Math layer:

- *The math: the three scalar problems* (recall the `shrinkage/` derivation for ℓ1 in one line;
  ℓ2: $c = w/(1+λ)$; ℓ0 with weight μ: hard threshold at $\sqrt{2μ}$).
- *The math: the lasso optimality conditions*: $f(x) = \tfrac12\|Ax - y\|^2 + λ\|x\|_1$ is convex;
  $x^\star$ is optimal iff $a_j^\top(y - Ax^\star) = λ\,\operatorname{sign}(x_j^\star)$ for
  $x_j^\star \ne 0$ and $|a_j^\top(y - Ax^\star)| \le λ$ for $x_j^\star = 0$ (subgradient of $|t|$
  is $[-1, 1]$ at 0). Hence $x^\star = 0$ iff $λ \ge λ_{\max} = \|A^\top y\|_\infty$.
- *The math: ℓ1 as the convex envelope of ℓ0* on the box $\|x\|_\infty \le 1$ (stated, with the 1D
  picture); ridge closed form $(A^\top A + λI)^{-1}A^\top y$ and why it has no exact zeros
  (generic case).
- *The math: MAP with a Laplace prior*: $p(c) \propto e^{-|c|/b}$ ⇒ $-\log p(x\mid y) =
  \frac1{2σ^2}\|Ax - y\|^2 + \frac1b\|x\|_1 + \text{const}$, so λ = σ²/b (same pattern as
  `regularise/`).

**Demo `lassogeom`** (new, main demo): left, the 2D coefficient plane: the least-squares point as
a large drag handle, misfit ellipses (their shape from a 2×2 $A^\top A$ set by a "correlation of
the columns" slider, so round = orthonormal), the penalty ball growing with λ (or the budget s),
the solution as a dot with its path traced as λ goes from 0 to λ_max. Penalty menu ℓ2 / ℓ1 / ℓ0.5 /
ℓ0 (ℓ0 shows the two axes plus the origin as the feasible "ball"). Right: the scalar transfer
curves (tab 1) or the **regularisation path** for a 6-column problem with a fixed seed (tab 2,
coefficients vs log λ, ℓ1 vs ℓ2 side by side, vertical line at the current λ, λ_max marked).
Readouts: the solution, number of zeros, misfit, penalty. Presets: "round contours: ℓ1 = soft
threshold", "the corner wins", "correlated columns: ℓ1 picks one", "ℓ2 never reaches the axis",
"λ ≥ λ_max: all zero", "p < 1: pointier, but two valleys", "the path: one at a time".
Computation: 2D lasso by exact enumeration of supports (4 cases), 6D path by coordinate descent
on a log grid of 100 λ values (well under 1 ms).

Exercise (by hand, concept): (a) ℓ2, ℓ1 (soft) and ℓ0 (hard) shrinkage of
$w = (4, -0.5, 1.2, -2.5, 0.3, 3)$ with λ = 1 (ℓ2: $c = w/2$; hard at $t = 1$); count zeros;
(b) sorted magnitudes 10, 6, 3, 2, 1, 0.5, 0.3, 0.2: how many coefficients for 95 % and 99 % of the
energy? (3 give 96.4 %, 4 give 99.1 %); (c) the budget picture: least-squares point (3, 1) with
round contours; the ℓ2 answer with budget 2 is (1.90, 0.63), find the ℓ1 answer (2, 0) by
drawing; (d) λ_max for $A = I$, $y = (3, 1)$; (e) match four transfer curves / four restored
profiles to ℓ0, ℓ1, ℓ2, TV; (f) which basis makes each of four sketched signals sparse? *(math)*:
(g) derive λ_max = ‖Aᵀy‖_∞ from the optimality conditions; (h) show that ridge gives $x_j = 0$
only if $(A^\top A + λI)^{-1}A^\top y$ happens to vanish there; (i) MAP with a Laplace prior gives
the lasso, and λ = σ²/b.

### 2. `ista/`: Solving with ℓ1: thresholding iterations and pursuit

Concept layer (about 18 content slides, 4 math):

- **[fig] When the basis is not orthonormal there is no one-line answer**: blur, a mask (missing
  pixels), undersampling, a redundant dictionary (two bases at once). Problem: minimise
  ½‖A x − y‖² + λ‖x‖₁ (synthesis form: $A = $ forward operator × synthesis basis $W$). Figure: four
  small panels, one per forward operator, with the same sparse truth.
- **Recall** (one slide, `regularise/`): the data + λ·penalty pattern, and the Landweber step
  $x \leftarrow x + τA^\top(y - Ax)$ (a gradient step on the misfit).
- **[fig] ISTA in one sentence**: take a Landweber step, then soft-threshold with $τλ$; repeat.
  "Fit a bit better, then make it simpler." Figure: a 1D signal after the gradient step (noisy,
  dense) and after thresholding (sparse), two iterations.
- **By hand**: $A = \begin{pmatrix}1 & 0.5\\ 0.5 & 1\end{pmatrix}$, $y = (2, 1)$, λ = 0.5,
  step τ = 0.4, start at 0: gradient step → (1.0, 0.8), threshold 0.2 → (0.8, 0.6); next
  (0.96, 0.58); converges to (1.6, 0). Ridge with the same λ: (1.15, 0.48). The truth was (2, 0):
  ℓ1 finds the support, and shrinks the value (bias, unit 1).
- **Step size**: τ must be below 1/L, L = the largest gain of $A^\top A$ (for a blur: the
  largest $|H|^2$ = 1). Too large: diverges (as for gradient descent, one line pointer to
  `backprop/`).
- **[fig] Soft thresholding is the "proximal step" of ℓ1**: the gradient step ignores the
  penalty; the threshold then solves the penalty part exactly. The same split works for any
  penalty whose own step is easy (non-negativity: clip at 0; TV: a TV denoiser, recall
  `algebraic-recon/`'s "alternate a data step with a TV step"). Figure: a 2D contour picture with
  the zigzag of iterates, gradient arrow then a snap to the axis.
- **[fig] FISTA: momentum** (one slide): extrapolate from the last two iterates before the step;
  same cost per iteration, far fewer iterations. Figure: objective vs iteration on log axes, ISTA
  vs FISTA (from the demo).
- **[fig] Greedy alternative: orthogonal matching pursuit (OMP)**: pick the atom most correlated
  with the residual, refit all chosen atoms by least squares, update the residual; stop at K
  atoms or when the residual is at the noise level. On the by-hand example: correlations 2.5 and
  2 → pick atom 1, refit → (2, 0), residual 0: exact, **no shrinkage**. When OMP fails: two very
  similar atoms (it commits early). Figure: three OMP iterations on a 1D spike signal (residual
  shrinking).
- **[fig] Try it: `ista1d`** (demo slide).
- **Deconvolution with a sparsity prior** **[fig]**: recall the `deconv/`/`regularise/` result
  (Tikhonov blurs edges, rings); with a wavelet ℓ1 prior: sharp steps, less ringing, but small
  low-contrast features vanish. Same λ choice rules as `regularise/` (discrepancy principle).
  Figure: a blurred noisy profile restored by Tikhonov, TV and wavelet-ℓ1 (same RMSE scale).
- **[fig] Inpainting as an inverse problem**: the forward operator **keeps** the known pixels and
  drops the rest ($A$ = mask). Least squares has no opinion about the hole (the hole is the null
  space, recall `illposed/`); a sparsity prior fills it with the sparsest continuation. Works for
  thin scratches and scattered missing pixels, fails for holes larger than the atoms (unit 5:
  patches; queued PDE day: diffusion/exemplar inpainting; `diffusion-inverse/`: a learned prior).
  Figure: a cell crop with 50 % random pixels missing and a scratch, restored with DCT/wavelet ℓ1.
- **[fig] Try it: `sparserestore`** (demo slide, modes deblur and inpaint).
- **Redundant dictionaries** (one slide): use two bases at once, spikes + DCT, or wavelets +
  DCT: a signal with both a few spikes and a few sinusoids is sparse in neither basis alone but
  in the union. Separation of spots from smooth background as the image version (pointer to
  optional unit 6).

Math layer:

- *The math: the proximal operator*: $\operatorname{prox}_{τg}(v) = \arg\min_x \tfrac12\|x - v\|^2 +
  τg(x)$; for $g = λ\|\cdot\|_1$ it is the soft threshold at τλ (from `shrinkage/`); for the
  indicator of $x \ge 0$ it is clipping.
- *The math: ISTA as majorise–minimise*: with $L \ge \|A\|^2$,
  $\tfrac12\|Ax - y\|^2 \le \tfrac12\|Ax_k - y\|^2 + \nabla^\top(x - x_k) + \tfrac L2\|x - x_k\|^2$;
  minimising the bound + λ‖x‖₁ is one ISTA step, so the objective never increases; a fixed point
  satisfies the lasso optimality conditions of unit 1.
- *The math: rates*: ISTA $f(x_k) - f^\star \le L\|x_0 - x^\star\|^2/(2k)$; FISTA with
  $t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2$, $z = x_k + \frac{t_k - 1}{t_{k+1}}(x_k - x_{k-1})$:
  $O(1/k^2)$ (stated).
- *The math: OMP*: residual orthogonal to the chosen atoms after each refit (normal equations on
  the support); exact recovery if the mutual coherence $μ < 1/(2K - 1)$ (stated; links to unit 3).

**Demo `ista1d`** (new): a 1D signal of length 128 or 256 that is sparse in a chosen basis
(spikes / Haar / db4 / DCT; seeded random support with K slider, or the Donoho–Johnstone test
signals from `lib/wavelet.ts`), forward operator (identity / Gaussian blur σ / random mask with
missing fraction / spikes + DCT dictionary), noise σ. Solver: ISTA, FISTA, OMP; λ (log slider,
with λ_max marked), iterations (log slider 1–1000) and a *step* button that shows the gradient
step and then the threshold as two separate frames. Panels: truth, data, estimate; the
coefficients as a stem plot with the ±τλ band and true support marked (hits, misses, false
atoms coloured); objective vs iteration (log–log) for all three solvers; error vs λ curve
(computed once per setting). Toggle *debias* (least-squares refit on the found support).
Presets: "identity: ISTA = one soft threshold", "blur: ISTA needs many steps", "FISTA vs ISTA",
"OMP is exact here", "OMP commits to the wrong atom" (two similar atoms), "step size too large:
diverges", "inpaint a sparse-in-DCT signal", "spikes + sines need both bases", "shrinkage bias
and debiasing".

**Demo `sparserestore`** (new, 2D; also used in unit 5): image (cell crop 128×128 on slides,
256×256 on the exercise page; synthetic piecewise-constant shapes; a texture patch), task
(denoise / deblur with Gaussian σ / inpaint with random missing fraction or a scratch mask /
text overlay to remove), prior menu set per embed by `data-priors` (unit 2: none (least squares
or Tikhonov), wavelet ℓ1 (Haar/db4, with cycle-spinning toggle), TV (from `lib/smooth.ts`); unit 5
adds 8×8 DCT patch dictionary, learned dictionary from `patchdict`, NL-means baseline). Solver
FISTA with progressive iterations (image updates every 10 iterations). λ slider, iteration count.
Panels: observed, restored, error vs truth (diverging), and PSNR vs iteration. Presets: "deblur:
Tikhonov rings, wavelet-ℓ1 does not", "TV: cartoon", "λ too large: texture gone", "50 % of the
pixels missing", "a scratch is fine, a large hole is not", "remove the text".

Exercise (by hand, concept): (a) the two ISTA iterations of the by-hand example and the limit
(1.6, 0) read from a given table; check that the ridge answer has no zero; (b) OMP on the same
example; why is OMP's answer (2, 0) and ISTA's (1.6, 0)? (c) step size: for a blur with
$\max|H| = 1$, which τ are safe? (d) given an inpainting mask on an 8-sample signal, which
coefficients of a constant + one cosine can be recovered (count unknowns vs known samples);
(e) predict restored profiles for Tikhonov, TV and wavelet-ℓ1 on a blurred step-and-ramp-and-spike
profile (four sketches to match). *(math)*: (f) show that the soft threshold is
$\operatorname{prox}_{τλ|\cdot|}$; (g) derive the ISTA step from the quadratic upper bound;
(h) show that a fixed point of ISTA satisfies the lasso optimality conditions.

### 3. `compressed-sensing/`: Compressed sensing: fewer measurements than unknowns

Concept layer (about 20 content slides, 4 math):

- **[fig] The surprise**: Nyquist (recall `fourier/`, `resolution/`): to capture a band-limited
  signal, sample at twice its bandwidth. But a signal with K nonzero entries out of N has only K
  numbers of information (plus where they are: $\log_2\binom{N}{K}$ bits; 48.5 bits for 8 of 256).
  Can we measure ~K numbers instead of N? Figure: a 256-sample spike signal with 8 spikes and
  its 256 Fourier coefficients; 40 random ones highlighted.
- **[fig] Underdetermined, so add a prior**: $y = Ax$ with M < N rows: infinitely many solutions
  (the null space, recall `illposed/`, `algebraic-recon/`). Least energy (ℓ2) picks the one that
  spreads energy everywhere; ℓ1 picks the sparse one. Figure: 3D picture, a random line (the
  solution set) meeting the ℓ1 octahedron at a vertex and the ℓ2 sphere anywhere.
- **By hand, 4 unknowns, 2 measurements**: $A = \begin{pmatrix}1&1&1&1\\1&-1&2&0\end{pmatrix}$,
  $y = (3, 6)$. Minimum energy: $x = (1.2, -0.6, 2.1, 0.3)$, ℓ1 = 4.2, ℓ2 = 2.51. Sparsest:
  $(0, 0, 3, 0)$ (only column 3 is proportional to $y$), ℓ1 = 3, ℓ2 = 3. ℓ2 prefers the spread
  answer, ℓ1 the true one.
- **[fig] Coherent vs incoherent aliasing** (the key picture): take every 4th Fourier
  coefficient of the spike signal and the zero-filled reconstruction shows **4 copies** of each
  spike (recall aliasing); take the same number **at random** and it shows the spikes plus
  **noise-like** low-level junk spread everywhere. Copies cannot be told from signal; noise-like
  junk can be thresholded away. Figure: the two zero-filled reconstructions side by side.
- **[fig] Recovery by iterative thresholding** (ISTA from unit 2): threshold the strong spikes,
  re-predict the data, the junk they caused disappears, weaker spikes emerge. Figure: four
  iterations.
- **Incoherence**: measurements must "see" every atom a little: random Fourier samples of a
  spike signal (each sample sees all spikes) — good; random **pixels** of a spike signal (most
  samples see nothing) — useless. Pixel basis vs Fourier is maximally incoherent; random
  matrices are incoherent with every fixed basis. Rule of thumb: the sensing basis must be
  spread out in the basis where the signal is sparse.
- **How many measurements?** M of the order of K·log(N/K), a few times K in practice; recovery
  is **stable**: with noise, the error is proportional to the noise (use the ℓ1 problem with a
  data term, as in unit 2), and compressible (not exactly sparse) signals are recovered as well
  as their best K-term approximation allows (recall `dwt2d/`).
- **[fig] Phase transition** (Donoho–Tanner): axes δ = M/N (how undersampled) and ρ = K/M (how
  sparse relative to measurements); each cell = fraction of random trials recovered. A sharp
  boundary: below it almost always success, above almost always failure. OMP's boundary lies
  below ℓ1's. Figure: the diagram computed by the demo (static copy for print).
- **[fig] Try it: `csrecover`** and **[fig] Try it: `phasetransition`** (two demo slides).
- **[fig] You have seen it already**: few-view CT with TV (recall `algebraic-recon/`: 30 views,
  FBP 0.081 vs TV 0.020): the gradient of the phantom is sparse, the projections are incoherent
  with it. Figure reused from `tools/figs/algebraic-recon/`.
- **The single-pixel camera** (one line + **[fig]** schematic): one detector, random mirror
  patterns, M ≪ N patterns. Why it matters only where detectors are expensive (infrared, THz).
- **What compressed sensing does not do** (text slide): it does not beat Nyquist for signals that
  are not sparse; it needs the sparsifying basis to be right; with too few measurements it fails
  abruptly (phase transition), and failure can look plausible (a sparse, wrong image); it trades
  measurement time for computation.

Math layer:

- *The math: null space property*: ℓ1 minimisation recovers every K-sparse $x$ from $Ax$ iff for
  every nonzero $h$ in the null space and every support S of size K, $\|h_S\|_1 < \|h_{S^c}\|_1$
  (proof sketch in four lines: compare $\|x + h\|_1$ with $\|x\|_1$).
- *The math: coherence and RIP*: mutual coherence $μ(A) = \max_{i\ne j}|a_i^\top a_j|$ (unit
  columns); recovery if $K < (1 + 1/μ)/2$. Restricted isometry: $(1-δ_K)\|x\|^2 \le \|Ax\|^2 \le
  (1+δ_K)\|x\|^2$ for all K-sparse $x$; Gaussian matrices satisfy it with high probability when
  $M \ge C K\log(N/K)$ (stated, Candès–Tao, Donoho 2006).
- *The math: zero filling is $A^\top y$*: for partial Fourier sampling with mask P, the zero-filled
  reconstruction is $F^{-1}PFx = x * \mathrm{psf}_P$, where $\mathrm{psf}_P$ is the inverse DFT of
  the mask: regular masks have a few large side peaks (copies), random masks many small ones
  (noise-like). The sidelobe level measures incoherence.
- *The math: the noisy problem*: $\min\|x\|_1$ s.t. $\|Ax - y\|_2 \le ε$ ⇔ the lasso for some λ;
  error bound $\|x^\star - x\|_2 \le C_0\,\|x - x_K\|_1/\sqrt K + C_1 ε$ (stated).

**Demo `csrecover`** (new, 1D, main demo): signal length N = 256, K-sparse in spikes / DCT /
Haar (K slider, seeded positions and signs/amplitudes, or a compressible signal); measurement
type: random Fourier samples / regular every-r-th Fourier sample / random Gaussian matrix /
random pixels; M slider (8–256), noise σ. Panels: truth; measurements highlighted on the
spectrum (or the sampled pixels); zero-filled (min-norm) reconstruction with the true spikes
overlaid; ℓ1 recovery (FISTA, 500 iterations, with an *animate* button that shows the
thresholding iterations) and OMP; the sampling PSF (inverse DFT of the mask) in a small panel.
Readouts: relative error, support recovered yes/no, M/N, K/M. Presets: "regular: copies",
"random: noise-like junk", "ℓ1 recovers from 25 %", "too few: abrupt failure", "random pixels
see nothing", "Gaussian matrix works for any basis", "compressible signal: best-K error",
"noise: stable, not exact".

**Demo `phasetransition`** (new): N = 64 (slides) or 128 (exercise page), grid of δ × ρ (20 × 20),
Gaussian or partial-Fourier sensing, solver ℓ1 (FISTA to tolerance) or OMP, trials per cell
(2–10, fixed seeds). The diagram fills in progressively (cells computed in animation-frame
chunks; a *stop* button); click a cell to see one trial in a small `csrecover`-style panel.
Overlay the theoretical Donoho–Tanner ℓ1 curve (a tabulated array computed offline, cited) as a
dashed line. Presets: "ℓ1, Gaussian", "OMP: lower boundary", "partial Fourier", "with noise:
the edge blurs". Static PNG of a precomputed run for print.

Exercise (by hand, concept): (a) the 4-unknown example: verify both solutions satisfy $Ax = y$,
compare their ℓ1 and ℓ2 norms; why is (0, 0, 3, 0) the only 1-sparse solution? (b) bits to name
8 positions out of 256 (48.5) vs 256 samples; (c) predict the zero-filled reconstruction of a
single spike from every 2nd and every 4th Fourier sample (copies at N/2, N/4) and from random
samples; (d) which measurement is useless for a spike signal: random pixels or random Fourier
coefficients? for a sinusoid? (e) read success/failure for four (δ, ρ) points off the printed
diagram; how does doubling N at fixed δ, ρ change the picture? *(math)*: (f) prove the
null-space property is sufficient (four lines); (g) mutual coherence of the 2×4 matrix above
(normalise columns); (h) show that keeping the even DFT coefficients of a length-N signal and
zero-filling gives $\tfrac12(x_n + x_{n+N/2})$.

### 4. `cs-mri/`: Fast MRI with compressed sensing

Concept layer (about 14 content slides, 3 math; about 1 h):

- **[fig] MRI measures Fourier coefficients** (one concept slide, no physics): magnetic field
  gradients make each position precess at its own frequency; the received signal at one time is
  one Fourier coefficient of the object. The scanner fills **k-space** line by line; the image is
  the inverse 2D DFT (recall `fourier/`). Figure: a head phantom (`sheppLogan` from `lib/synth.ts`)
  and its k-space (log magnitude), one line highlighted.
- **Time = lines**: each phase-encoding line costs a fixed time; a 3D scan with 256 × 128 lines at
  10 ms per line takes 5.5 min; patients move, children cannot hold still, the heart beats.
  Fewer lines = shorter scans.
- **[fig] Three ways to skip lines**: (i) keep only the centre: a blurred image (recall low-pass);
  (ii) keep every 2nd line: the image **folds** (two copies half a field of view apart; recall
  aliasing); (iii) random lines, denser at the centre (**variable density**: the energy is at low
  frequencies): noise-like artefacts. Figure: masks and their zero-filled images, 3 × 2 grid.
- **[fig] The reconstruction**: sparse in wavelets (and/or TV): FISTA with the partial-Fourier
  forward operator; data consistency on the measured lines, sparsity elsewhere. Figure: zero-filled
  vs CS from 25 % of the lines.
- **[fig] Try it: `csmri`** (demo slide).
- **In practice** (text slide): accelerations of about 2–8× are used clinically since about 2017,
  usually combined with multiple receiver coils (parallel imaging, one line: each coil sees the
  object through its own sensitivity, which adds equations); radial and spiral trajectories as a
  pointer (radial lines in k-space = the Fourier slice theorem of `backprojection/`); dynamic MRI
  (time is very sparse: frames differ little).
- **[fig] What can go wrong**: small, low-contrast lesions shrink or vanish (the ℓ1 shrinkage of
  unit 1); TV makes a "cartoon"; too much acceleration fails abruptly (unit 3). Unlike a learned
  prior, the sparsity prior does not invent plausible anatomy, but it can remove real detail.
  Figure: a phantom with a small faint disc, recovered at 4× and 8×.
- **Learned reconstruction** (one slide, pointer): unrolling ISTA's iterations and learning the
  thresholds and filters gives a network (learned ISTA; pointer to `cnn/`); diffusion priors for
  the same problem: `diffusion-inverse/`.

Math layer:

- *The math: k-space sampling and the PSF of a mask*: $y = PFx + n$; zero-filled $=
  F^{-1}P^\top y$; point spread function of the mask $=F^{-1}(\text{mask})$; regular undersampling
  by $r$ gives $r$ copies shifted by FOV/r.
- *The math: the CS-MRI problem*: $\min_x \tfrac12\|PFx - y\|^2 + λ_1\|Wx\|_1 + λ_2\,\mathrm{TV}(x)$
  (Lustig, Donoho & Pauly 2007); one FISTA step costs two FFTs and one wavelet transform pair;
  $PF$ has orthonormal rows, so L = 1 and τ = 1 works.
- *The math: variable density*: sampling probability $p(k) \propto (1 - |k|/k_{\max})^q$ with
  the centre fully sampled (one line); why: coherence is worst at low frequencies for wavelet
  atoms.

**Demo `csmri`** (new, 2D): phantom (`sheppLogan` 128² on slides / 256² on the exercise page; a
phantom with small faint lesions; the cell image as a non-MRI test), k-space mask: centre only /
regular every r-th line (+ centre) / random lines uniform / random variable density (exponent q) /
radial spokes (pointer), acceleration slider (1–10×), noise σ. Reconstruction: zero-filled
(instant) / CS with wavelet ℓ1 (Haar/db4) / + TV (`lib/smooth.ts` `tv2d` as the TV prox), λ
sliders, iterations progressive (FISTA, image refresh every 5 iterations). Panels: mask, k-space
(log magnitude with the mask), zero-filled, CS result, error vs truth; the mask's PSF as an
inset. Readouts: acceleration, PSNR/SSIM (`lib/metrics.ts`), scan-time readout "x min → y min"
for a given time per line. Presets: "centre only: blur", "every 2nd line: fold", "random: noise-like",
"variable density + wavelets: 4×", "8×: lesions vanish", "TV cartoon", "radial spokes".

Exercise (by hand, concept): (a) scan time: 256 × 128 lines at 10 ms, then at 4× acceleration
(5.5 min → 82 s); (b) 1D fold: $x = (0, 0, 5, 0, 0, 0, 0, 0)$, keep the even DFT coefficients,
zero-filled result $(0, 0, 2.5, 0, 0, 0, 2.5, 0)$; where is the ghost for every 4th line?
(c) why must the centre of k-space be sampled densely? (use a phantom's spectrum); (d) predict
the images for three given masks (match pictures); (e) a radiologist asks: "will CS hide a
3-pixel faint lesion?" answer using unit 1's shrinkage. *(math)*: (f) the PSF of a regular mask
with spacing r; (g) show that $PF$ with orthonormal $F$ has $\|PF\| = 1$, so τ = 1 is a valid ISTA
step.

### 5. `dictionaries/`: Sparse coding and dictionary learning

Concept layer (about 22 content slides, 4 math):

- **[fig] Patches as vectors**: an 8 × 8 patch is a point in 64 dimensions (recall `pca/`'s "two
  pixels, one point"). A **dictionary** is a set of atoms (64-vectors); **sparse coding**: write each
  patch as a few atoms. Figure: a patch = 0.8 × atom 12 + 0.3 × atom 40 − 0.2 × atom 3.
- **[fig] Fixed dictionaries**: the 64 DCT basis images (recall `dct/`), an overcomplete DCT (256
  atoms: more frequencies than needed), wavelets. Figure: the overcomplete DCT atom grid.
- **Overcomplete = more atoms than pixels**: more choice, sparser codes; coding is no longer a
  change of basis but a lasso/OMP problem (unit 2) per patch.
- **[fig] Learn the atoms from the image itself**: alternate (1) **sparse coding**: OMP for every
  patch with the current atoms; (2) **atom update**: change each atom (and its coefficients) to
  fit the patches that use it. Figure: atoms after 0, 5 and 20 iterations on the cell image.
- **[fig] k-means is the one-atom special case** (taught in one slide here): each patch uses exactly
  one atom with coefficient 1 → assignment = coding, cluster mean = atom update (recall
  `kmeans/` if taken). K-SVD generalises this to "a few atoms with any coefficients". Figure: 2D
  point cloud with k-means centres vs K-SVD atoms as directions (lines through 0).
- **K-SVD atom update** (concept): for one atom, collect the patches that use it, remove the
  other atoms' contributions, and take the **best single direction** through those residuals: the
  first principal direction (no centring; recall `pca/`). **MOD** (one line): update all atoms at
  once by least squares.
- **By hand**: three 2-pixel residuals (2, 1), (4, 2), (−1, −1) use the same atom; the best
  direction is the top eigenvector of $\sum r r^\top = \begin{pmatrix}21 & 11\\11 & 6\end{pmatrix}$:
  (0.884, 0.467); new coefficients 2.24, 4.47, −1.35.
- **[fig] PCA vs learned dictionary**: PCA (= KLT, `dct/`) gives orthogonal atoms, all used by every
  patch, ordered by variance: optimal for **linear** approximation with k terms. A learned
  dictionary gives overcomplete, non-orthogonal atoms, each patch using a **different** few:
  better for **sparse** (non-linear) approximation. Figure: PCA eigen-patches vs K-SVD atoms of
  the same image, and the error-vs-k curves.
- **[fig] What learned atoms look like**: oriented edges, bars and blobs at several positions —
  like Gabor functions, the receptive fields of V1 neurons (Olshausen & Field 1996) and the first
  layer of a CNN (pointer to `cnn/`). Figure: atoms learned from natural-image patches and from
  the micrograph.
- **[fig] Denoising with patches** (K-SVD denoising): code every **overlapping** patch with OMP,
  stopping when the residual reaches the noise level (≈ 64 · (1.15σ)² for 8 × 8, one line);
  average the overlapping reconstructions (each pixel is covered by 64 patches) with the noisy
  image. Figure: noisy, wavelet shrinkage, NL-means (`denoise/`), DCT dictionary, learned
  dictionary, with PSNR.
- **Why it works**: images repeat local structure (the same idea as NL-means, recall `denoise/`);
  the dictionary stores the recurring patterns, the sparse code chooses among them, noise is not
  sparse in any atom.
- **[fig] Inpainting with patches**: code each patch using only its known pixels (mask the atoms
  the same way), then fill the unknown pixels from the code. Figure: 50 % missing pixels and a
  text overlay, restored.
- **[fig] Try it: `patchdict`** and **[fig] Try it: `sparserestore`** (unit 5 priors).
- **Deconvolution with a patch prior** (one slide): alternate a data step (Landweber, unit 2) with
  a patch-denoising step: the same split as ISTA with a learned denoiser in place of the
  threshold ("plug-and-play", pointer to `diffusion-inverse/`, where the denoiser is a network).
- **Costs and limits** (text slide): learning is slow (OMP on tens of thousands of patches per
  iteration); patch size limits the structures (larger holes than a patch cannot be filled); the
  dictionary learnt on one image type may hallucinate its structures into another (a mild version
  of the learned-prior warning of the diffusion elective); texture with no repetition is lost.
- **Summary of the day** slide (concept only): one table "penalty / solver / what it buys / what it
  destroys" for ℓ2, ℓ1 in a fixed basis, TV, ℓ1 in a learned dictionary.

Math layer:

- *The math: sparse coding and dictionary learning as one problem*:
  $\min_{D, C}\ \|Y - DC\|_F^2$ s.t. $\|c_i\|_0 \le K$ for every column, atoms of unit norm;
  non-convex, solved by alternation; each step does not increase the objective.
- *The math: the K-SVD update*: $\|Y - DC\|_F^2 = \|E_k - d_k c^k\|_F^2$ with $E_k = Y - \sum_{j\ne
  k} d_j c^j$ restricted to the columns that use atom k; the minimiser over $d_k, c^k$ is the
  leading singular pair of $E_k$ (Eckart–Young, recall `pca/` "the error is the sum of the dropped
  eigenvalues"). MOD: $D = YC^\top(CC^\top)^{-1}$.
- *The math: k-means as K = 1 with coefficients 1* (one line each for the two steps).
- *The math: averaging overlapping patches*: $\hat x = (λI + \sum_p R_p^\top R_p)^{-1}(λy +
  \sum_p R_p^\top D c_p)$ with $R_p$ the patch extraction; $\sum_p R_p^\top R_p$ is diagonal
  (64 for interior pixels), so it is a per-pixel weighted average.

**Demo `patchdict`** (new, main demo): image (cell 128², a texture, synthetic shapes, a natural
photo shared with the colour/compression days if licensed), patch size 6 or 8, number of atoms
(64 / 128 / 256), sparsity K (1–8), initial dictionary (random patches / overcomplete DCT), method
(K-SVD / MOD / k-means). Buttons *one iteration*, *run 20*, *reset*; training on a seeded subset of
4000 patches. Panels: the atom grid (sorted by usage, each atom shown with its usage count), the
image with a movable patch selector, the selected patch, its sparse code (stem plot of the chosen
atoms with coefficients, atoms highlighted in the grid) and reconstruction; representation error vs
iteration; error vs K for PCA (first K components), DCT and the learned dictionary. Presets: "DCT
start", "learned: edges appear", "k-means atoms: blurry averages", "K = 1 vs K = 4", "PCA vs
learned at K = 4", "texture image: stripes in the atoms". The learned dictionary is exported to
`sparserestore` on the same page through the shared `patches.ts` cache (no network; same-page
only), otherwise `sparserestore` learns its own at startup (progressively).

Exercise (by hand, concept): (a) code a 4-pixel patch with OMP over a given 6-atom dictionary for
K = 1 and K = 2; (b) the K-SVD update of the by-hand example: eigenvector, new coefficients;
(c) k-means vs K-SVD: which atoms does each learn from patches that are all ±(one edge pattern)?
(k-means needs two atoms, K-SVD one); (d) counting: overlapping 8 × 8 patches of a 256 × 256 image
(249² = 62 001), each pixel in 64 patches; cost of OMP with 256 atoms and K = 4 per patch;
(e) OMP stopping threshold for σ = 10 grey levels and 8 × 8 patches; (f) rank four methods on a
given PSNR table and say which you would use on a fluorescence image with repeated nuclei and on a
single-instance texture. *(math)*: (g) show the K-SVD update is a rank-1 approximation and solve
it with the SVD; (h) derive the patch-averaging formula and show $\sum_p R_p^\top R_p$ is diagonal;
(i) show the k-means steps are the K = 1, coefficient-1 case of the alternation.

### 6. (optional / further reading) `sparse-separation/`: Two dictionaries: separating spots, edges and texture

- **Morphological component analysis** (Starck, Elad & Donoho 2005): an image = cartoon (sparse
  in wavelets or TV) + texture (sparse in local DCT) or spots (sparse in pixels) + smooth background
  (sparse in coarse wavelets); solve one ℓ1 problem over the union and read each part off its own
  coefficients. Image uses: separating puncta from a diffuse cytoplasmic signal, removing
  stripes/periodic texture, cartoon–texture decomposition.
- Reweighted ℓ1 and ℓ0-like penalties (one slide each), **convolutional sparse coding** (atoms as
  small kernels sliding over the image: a link between dictionaries and convolution layers), and
  **learned ISTA** (unrolled iterations as a network).
- **Demo**: `sparserestore` with `data-mode="separate"` (two-dictionary ISTA; panels: the input and
  the two or three components). Presets: "spots + background", "cartoon + texture", "stripes".
- By hand: a 6-sample signal = one spike + one cosine; one ISTA iteration over the union basis
  (given correlations).

## Shared engineering

Use the shared libraries (`demos/src/lib/README.md`); new files this day owns are
`demos/src/lib/sparse.ts` and `demos/src/lib/patches.ts` (neither name exists in `demos/src/lib/`).

- **`demos/src/lib/sparse.ts`** (new, owner: this day; no DOM, runs under node):
  - penalties and proximal operators: `soft`/`hard` **re-exported from `lib/wavelet.ts`** (no
    second copy), vector versions `softVec(v, t)`, `hardVec`, `proxNonneg`, `proxL2(v, t)`;
    `lassoScalar(w, λ, p)` for p = 0, 1, 2 (transfer curves);
  - `LinOp` = the `Operator` interface of `lib/deconv.ts` (`forward`, `adjoint`) plus an optional
    `norm2` estimate; builders `identityOp`, `maskOp(mask)`, `blurOp1d/2d` (wrapping
    `directOperator1d`/`fftOperator2d` from `lib/deconv.ts`), `partialFourierOp1d/2d(mask)` (via
    `lib/fft.ts`; orthonormal scaling), `gaussianMatrixOp(M, N, rng)`, `synthesisOp(W)` composing
    with a wavelet (`lib/wavelet.ts` `dwt`/`idwt`, `dwt2`/`idwt2`) or DCT basis (`lib/dct.ts`), and
    `unionOp(W1, W2)` for two dictionaries; `powerIteration(op)` for L;
  - solvers with an iteration callback (snapshots, objective history): `ista`, `fista` (generic prox
    argument, so TV from `lib/smooth.ts` `tv2d` and clipping plug in), `omp` (dense, small N; QR
    update of the support via `lib/linalg.ts` `lstsq`), `debias(support)`;
  - `lassoPath(A, y, lambdas)` by coordinate descent with warm starts (for `lassogeom`),
    `lasso2dExact` (enumerate supports), `lambdaMax`;
  - compressed sensing: `randomSupport(N, K, rng)`, `sparseSignal`, sampling masks
    `randomMask`, `regularMask(r, centre)`, `variableDensityMask2d(accel, q, centre, rng)`,
    `radialMask2d(spokes)`, `maskPsf`; `recoveryTrial(…)` and `phaseCell(…)` for the diagram;
    `DONOHO_TANNER_L1` (tabulated (δ, ρ) curve, computed offline, source cited in the file);
    `mutualCoherence(A)`.
- **`demos/src/lib/patches.ts`** (new, owner: this day; no DOM): `extractPatches(img, w, h, p,
  stride)`, `aggregatePatches(…, weights)` (overlap averaging with the noisy image, unit 5 math),
  masked patches for inpainting; dictionaries `dctDictionary(p, atoms)` (overcomplete DCT, built from
  `lib/dct.ts`), `randomPatchDictionary`; `batchOmp(D, Y, K | errTarget)` with a precomputed Gram
  matrix (Rubinstein et al. 2008); `ksvdStep`, `modStep`, k-means atoms via `lib/kmeans.ts`
  (not reimplemented); `ksvdDenoise`, `patchInpaint`; a same-page dictionary cache keyed by image
  and settings.
- **Reused, not duplicated**: `lib/wavelet.ts` (bases, `soft`/`hard`, `denoise2`, cycle-spinning
  `shift2`, `testSignal`), `lib/dct.ts` (DCT matrices, 2D DCT, block DCT), `lib/fft.ts` (partial
  Fourier, mask PSFs), `lib/deconv.ts` (`Operator`, blur operators, `gaussian1d`, `fftOperator2d`),
  `lib/smooth.ts` (`tv1d`, `tv2d` as TV proximal steps), `lib/denoise.ts` (NL-means baseline,
  `methodNoise`), `lib/linalg.ts` (`lstsq`, `svd`, `jacobiEigen`), `lib/kmeans.ts`, `lib/metrics.ts`
  (PSNR with explicit peak, SSIM), `lib/synth.ts` (`sheppLogan`, test shapes; add the
  faint-lesion phantom and a spikes-plus-sines signal **there**, not locally), `lib/random.ts`
  (`rng`, `gauss`, `normalArray`, `sampleIndices`), `lib/imageops.ts` (`loadGrey`), `lib/cell.ts`
  (`loadCell`, `CELL_CREDIT`), `lib/plot.ts` (stem plots via `line`/`scatter`, log axes, `heatmap`
  for the phase diagram and k-space, `colorbar`), `lib/colour.ts` (diverging map for errors and
  coefficients, `viridis` for k-space).
- **Speed** (TypeScript first; measure, `demos/README.md` rule: Wasm only if a live recompute
  measures > ~30 ms):
  - `lassogeom`: < 1 ms per drag (2D exact, 6D path by coordinate descent).
  - `ista1d`, `csrecover`: N = 256, dense M × N products or FFTs; 500 FISTA iterations ≈ 10–30 ms:
    live on slider release, animation optional.
  - `phasetransition`: 400 cells × trials × FISTA at N = 64–128: seconds in total, by design
    **progressive** in animation-frame chunks (not a live recompute), so no Wasm.
  - `csmri` and `sparserestore`: one FISTA iteration at 256² = 2 FFTs + one 2D DWT pair
    ≈ 20–25 ms; 100 iterations are progressive (refresh every 5–10 iterations); 128² on slides
    (≈ 5 ms per iteration). Write `demos/bench/sparse-bench.ts` (node) timing one iteration at 128²
    and 256².
  - `patchdict` / dictionary restoration: Batch-OMP over 14 641 overlapping 8 × 8 patches of a
    128² image with 128–256 atoms, K ≤ 4: estimated 0.3–1 s per pass; learning on a 4000-patch
    subset ≈ 0.1–0.3 s per iteration. Progressive with a "computing…" state. If a slider-release
    recompute of the patch-denoising step exceeds ~1 s at 128² after Batch-OMP, move `batchOmp`
    to Rust/Wasm (`demos/wasm/sparse/`, flat `Float32Array` in/out, TypeScript reference kept and
    tested against it).
- **Tests** (`demos/test/sparse-test.ts`, run by `make test`): `softVec` equals `wavelet.soft`
  elementwise; the by-hand ISTA example reproduces (0.8, 0.6), (0.96, 0.58) and converges to
  (1.6, 0); ridge gives (1.152, 0.485); OMP returns (2, 0); `lambdaMax` makes the lasso solution
  zero and λ_max·0.99 does not; lasso path by coordinate descent matches `lasso2dExact` in 2D;
  FISTA objective ≤ ISTA objective after 100 iterations on a random problem; ISTA objective is
  non-increasing; the 2×4 CS example recovers (0, 0, 3, 0) by ℓ1 and the min-norm solution is
  (1.2, −0.6, 2.1, 0.3); regular even-index Fourier masking gives ½(x_n + x_{n+N/2}); partial
  Fourier operators are adjoint-consistent (⟨Ax, y⟩ = ⟨x, Aᵀy⟩) for all operators; recovery of a
  K = 5, N = 128 signal from M = 50 Gaussian measurements succeeds for a fixed seed; `extractPatches`
  + `aggregatePatches` with an identity code returns the image; Batch-OMP equals plain OMP; K-SVD
  objective non-increasing over 5 iterations; the by-hand K-SVD update gives (0.884, 0.467).
- **Demo conventions** (`demos/README.md`, `plan/build-issues.md` lessons): TypeScript,
  `make typecheck`, no network, fixed seeds, presets + reset, the `slider`/`button`/`checkbox`
  helpers (never bind keys), labels ≥ 18 px, fits 1150 × 560 on a slide (`csmri`, `sparserestore`
  and `patchdict` show two large panels + a side strip on slides, all panels with
  `data-layout="full"` on the exercise page); per-demo CSS prefixed with
  `.qb-demo[data-demo="<name>"]` (the shared `demo.css` loads late); `data-*` settings
  (`data-mode`, `data-priors`, `data-n`) so one demo serves several units; static figures in the
  decks as the print fallback.
- **Figures in decks**: `![](fig.svg){fig-alt="..."}` (bracket text would show as a caption on
  slides), `.nostretch` on lone slide images; figure scripts in `tools/figs/<unit>/` (in the repo,
  not the scratchpad; see build-issues "figure scripts live only in the scratchpad").
- **Images**: `cell256.png` (credit via `CELL_CREDIT`), Shepp–Logan phantom (synthetic, labelled as
  such; it is a head phantom, good for the MRI unit), synthetic signals and shapes; real MRI
  k-space data only if a licence-compatible set is found (open question 3).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `sparsity/`: lecture (`lassogeom`), shrinkage and the budget picture by hand |
| 10:45–12:15 | Unit 2 `ista/`: lecture (`ista1d`), two ISTA steps and OMP by hand, `sparserestore` (deblur, inpaint) |
| 13:00–14:30 | Unit 3 `compressed-sensing/`: lecture (`csrecover`, `phasetransition`), the 2×4 example and fold-over by hand |
| 14:30–15:30 | Unit 4 `cs-mri/`: lecture (`csmri`), scan time and the 1D fold by hand |
| 15:45–17:15 | Unit 5 `dictionaries/`: lecture (`patchdict`, `sparserestore` with dictionaries), K-SVD update by hand; quizzes as wrap-up |

If the day must be shorter: merge unit 4 into unit 3 as three slides + the `csmri` demo on the
exercise page, and move the 2D restoration part of unit 2 (`sparserestore`) into unit 5.

## Proposed reorganisation

1. **Ownership of sparsity (coursemap edit, owners table and "Decided after the deconvolution
   plan").** Today the course map says "sparsity with wavelets". Proposal: add an owners-table row
   *"Sparsity priors in general: ℓ0/ℓ1/ℓ2, lasso, proximal/ISTA, compressed sensing, dictionary
   learning → Sparse representations"*; the wavelets day keeps **wavelet shrinkage** (orthonormal
   basis, its own ℓ1 math slide) as the first, special case. Effect on built units: one forward
   pointer line in `shrinkage/` ("A third view" slide), `regularise/` ("What the penalty can say":
   "few non-zero coefficients in a basis → sparse representations elective"), `denoise/` (the
   ℓ1-vs-ℓ2 line) and `algebraic-recon/` (priors slide: "this is compressed sensing"). No content
   moves out of built units.
2. **Sidebar position inside "Transforms".** Entry stays "after Wavelets" (hard prerequisites:
   `haar/`, `dwt2d/`, `shrinkage/`, plus core `regularise/`). Proposed order in the group:
   Wavelets → Image compression → Radon → **Sparse representations**, because the day recalls
   `dct/` and `algebraic-recon/` (soft prerequisites) and closes the group by unifying it
   ("transforms make images sparse; sparsity lets us measure less"). Students who skip compression
   or Radon lose one recall slide each, nothing more.
3. **Inpainting split with the queued "Edge-preserving smoothing and PDEs" day.** That day's queue
   entry lists inpainting (diffusion- and exemplar-based). Proposal: inpainting **as an inverse
   problem with a mask operator** is introduced here (unit 2, one concept slide + presets; unit 5
   patch inpainting); the PDE day owns diffusion-based and exemplar-based inpainting and recalls
   the mask formulation in one line. Also add this day to the **Denoising** topic page and propose
   a small **Inpainting** section there (sparse, PDE, diffusion-model inpainting from
   `diffusion-inverse/`).
4. **K-means as a dependency.** Unit 5 teaches k-means in one slide rather than requiring the
   Gaussian-mixture elective (`kmeans/`), so the prerequisite map lists `kmeans/` as soft only.

## Open questions

1. **Five units or four?** Unit 4 (CS-MRI) is short; the fallback above folds it into unit 3.
   Recommendation: keep it separate (the scope names fast MRI as *the* application, and the 2D
   demo needs room).
2. **How much optimisation theory?** Proximal operators and FISTA rates are on math slides only;
   ADMM/primal–dual (needed for exact TV) stays a one-line pointer, and the TV prox reuses
   `lib/smooth.ts`. Agree?
3. **Real MRI data**: a licence-compatible fully sampled k-space or magnitude image (several public
   MRI sets have research-only licences, which may not allow redistribution in a course site).
   Fallback: Shepp–Logan and a synthetic brain-like phantom (both synthetic, labelled).
4. **Phase-transition reference curve**: tabulate the Donoho–Tanner ℓ1 weak threshold offline
   (cite the source table; at δ = 0.5 the curve is near ρ ≈ 0.38, to be checked against the
   table when building) or show only the empirical diagram?
5. **Where does the patch-prior "plug-and-play" slide belong**: here (unit 5, proposed: one slide)
   or in `diffusion-inverse/`, which already has the learned-denoiser view?
6. **Colour/natural photo** for `patchdict` (natural-image atoms are the classic picture): reuse
   the colour/compression days' photo once licensed (open there too).
7. **Lasso naming**: use "lasso" on slides (statistics name, students may meet it) or only "ℓ1
   regularisation"? Proposal: say both once, then ℓ1.
8. Edits elsewhere once accepted (not done here): coursemap owners row and Transforms order;
   forward pointers in `shrinkage/`, `regularise/`, `denoise/`, `algebraic-recon/`; the Denoising
   topic page; the queued PDE day's plan to recall the mask formulation.

## References

- R. Tibshirani (1996), Regression shrinkage and selection via the lasso, *J. R. Stat. Soc. B* 58.
- S. S. Chen, D. L. Donoho and M. A. Saunders (1998), Atomic decomposition by basis pursuit, *SIAM
  J. Sci. Comput.* 20.
- D. L. Donoho and J. M. Johnstone (1994), Ideal spatial adaptation by wavelet shrinkage,
  *Biometrika* 81 (recall from the wavelets day).
- I. Daubechies, M. Defrise and C. De Mol (2004), An iterative thresholding algorithm for linear
  inverse problems with a sparsity constraint, *Comm. Pure Appl. Math.* 57 (ISTA).
- A. Beck and M. Teboulle (2009), A fast iterative shrinkage-thresholding algorithm for linear
  inverse problems, *SIAM J. Imaging Sci.* 2 (FISTA).
- Y. C. Pati, R. Rezaiifar and P. S. Krishnaprasad (1993), Orthogonal matching pursuit, *Asilomar
  Conf.*; J. A. Tropp (2004), Greed is good: algorithmic results for sparse approximation, *IEEE
  Trans. Inf. Theory* 50.
- E. J. Candès, J. Romberg and T. Tao (2006), Robust uncertainty principles: exact signal
  reconstruction from highly incomplete frequency information, *IEEE Trans. Inf. Theory* 52;
  D. L. Donoho (2006), Compressed sensing, *IEEE Trans. Inf. Theory* 52.
- E. J. Candès and M. B. Wakin (2008), An introduction to compressive sampling, *IEEE Signal
  Processing Magazine* 25(2) (the most readable overview; teacher background).
- D. L. Donoho and J. Tanner (2009), Observed universality of phase transitions in high-dimensional
  geometry, *Phil. Trans. R. Soc. A* 367.
- M. F. Duarte et al. (2008), Single-pixel imaging via compressive sampling, *IEEE Signal
  Processing Magazine* 25(2).
- M. Lustig, D. Donoho and J. M. Pauly (2007), Sparse MRI: the application of compressed sensing for
  rapid MR imaging, *Magn. Reson. Med.* 58; M. Lustig et al. (2008), Compressed sensing MRI, *IEEE
  Signal Processing Magazine* 25(2).
- B. A. Olshausen and D. J. Field (1996), Emergence of simple-cell receptive field properties by
  learning a sparse code for natural images, *Nature* 381.
- K. Engan, S. O. Aase and J. H. Husøy (1999), Method of optimal directions for frame design,
  *ICASSP* (MOD).
- M. Aharon, M. Elad and A. Bruckstein (2006), K-SVD: an algorithm for designing overcomplete
  dictionaries for sparse representation, *IEEE Trans. Signal Process.* 54.
- M. Elad and M. Aharon (2006), Image denoising via sparse and redundant representations over
  learned dictionaries, *IEEE Trans. Image Process.* 15.
- J. Mairal, M. Elad and G. Sapiro (2008), Sparse representation for color image restoration, *IEEE
  Trans. Image Process.* 17 (inpainting).
- R. Rubinstein, M. Zibulevsky and M. Elad (2008), Efficient implementation of the K-SVD algorithm
  using batch orthogonal matching pursuit, Technion tech. report CS-2008-08.
- J.-L. Starck, M. Elad and D. L. Donoho (2005), Image decomposition via the combination of sparse
  representations and a variational approach, *IEEE Trans. Image Process.* 14 (MCA, optional unit).
- K. Gregor and Y. LeCun (2010), Learning fast approximations of sparse coding, *ICML* (learned
  ISTA, pointer).
- M. Elad (2010), *Sparse and Redundant Representations*, Springer; S. Foucart and H. Rauhut (2013),
  *A Mathematical Introduction to Compressive Sensing*, Birkhäuser; T. Hastie, R. Tibshirani and
  M. Wainwright (2015), *Statistical Learning with Sparsity*, CRC (teacher background).

## Build status (2026-11-13)

Built: units `sparsity/`, `ista/`, `compressed-sensing/`, `cs-mri/`, `dictionaries/` (lec, quiz, exercise
with #explore); demos `lassogeom`, `ista1d`, `sparserestore`, `csrecover`, `phasetransition`, `csmri`,
`patchdict`; libraries `demos/src/lib/sparse.ts`, `demos/src/lib/patches.ts`; tests
`demos/test/sparse-test.ts` (add to `TESTS` in the Makefile); figure scripts `tools/figs/{sparsity,ista,
compressed-sensing,cs-mri,dictionaries}/` (Python + node `data.ts` that use the demos' code).
Not built: optional unit 6 `sparse-separation/` (and the `data-mode="separate"` of `sparserestore`),
`demos/bench/sparse-bench.ts` (outside this build's file ownership; timings below were measured with a
scratch script).

Deviations from the plan, found while building:
- The faint-lesion phantom and the spikes-plus-sines signal live in `lib/sparse.ts` (`LESION_PHANTOM`,
  `spikesPlusSines`), not in `lib/synth.ts` (existing lib files were not to be edited). The synthetic
  shapes/texture images and masks of the restoration demos are in `demos/src/sparserestore/images.ts`.
- `DONOHO_TANNER_L1` is tabulated from the state-evolution formula of Donoho, Maleki & Montanari (2009),
  which coincides with the Donoho–Tanner weak threshold: ρ(0.5) = 0.386 (plan: "≈ 0.38" ✓).
- ISTA with τL = 2.2 on the blur problem does **not** diverge in the demo (the threshold kills the
  growing constant component); FISTA does. The "diverges" preset uses FISTA, and the slide says
  "unstable" rather than "diverges".
- OMP picks the atom between two close blurred spikes (preset "OMP commits to the wrong atom": spikes at
  126 and 130 under blur σ = 3, OMP picks 128).
- 1D deblurring comparison (step, ramp, spike, bump; blur σ = 3): RMSE Tikhonov 0.111, wavelet ℓ1 0.108,
  TV 0.089; the plan's "wavelet ℓ1: sharp steps, less ringing" holds only weakly on this profile, and the
  slide reports the ranking honestly.
- CS-MRI on the 256² Shepp–Logan at 4× (variable density): zero-filled 21.3 dB, CS (Haar ℓ1) 27.1 dB;
  but centre-only zero filling at the same 4× already gives 26.7 dB (PSNR hides the lost edges; noted).
  The five faint lesions vanish already at 4× (plan said "at 8×"); the 8× preset keeps its name.
- Patch inpainting: better than wavelet ℓ1 for 50 % random missing pixels (24.1 vs 22.4 dB) but worse
  for a 16 × 16 hole (it invents a dark blob: 25.3 vs 31.1 dB with the scratches). Used on the slide as
  the "learned prior invents structure" example. `patchInpaint` fills holes by onion peeling.
- Denoising on the micrograph crop (σ = 20): all methods within 0.4 dB (wavelet 24.3, NL-means 24.0,
  DCT dictionary 24.4, learned 24.0, noisy 22.1): the micrograph's texture limits every method.
- Timings (node, 128²/256²): partial Fourier AᵀA 5/23 ms, Haar prox 4/8 ms, db4 prox 4/14 ms, db4 prox
  with 4 shifts 17/57 ms, TV prox (25 Chambolle iterations) 54/115 ms. All 2D solvers run progressively
  (refresh every ~30 ms), so no Wasm. The phase diagram costs about 15 ms per ℓ1 trial at N = 64
  (16 × 16 cells × 2 trials ≈ 8 s, progressive).
