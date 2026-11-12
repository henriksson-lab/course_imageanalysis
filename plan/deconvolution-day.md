# Plan: a day on deconvolution and restoration

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
**Course map (2026-11-12):** core day 5 of 18, Part I, directly after
day 4 *Image formation* (`plan/imageformation-day.md`) and before Part II (thresholding).
Built from units 3–4 of the earlier combined image formation plan, extended.

Owner of **restoration**: inverse filtering as a method, truncated inverse, Wiener,
Tikhonov, **regularisation of inverse problems in general** (data term + λ·penalty, choosing
λ, early stopping as regularisation; see open questions), Richardson–Lucy, 3D deconvolution
in practice, blind deconvolution, PSF mismatch, restoration artefacts, the classical
**denoiser comparison** and **what may be measured after restoration**. It does not own the
forward model, the PSF/OTF or the diagnosis "inversion is ill-posed" (image formation), noise
models (`cameranoise/`), TV and MRF priors (Ising elective, `gmrf/`), sparsity and wavelet
shrinkage (wavelets elective), or learned and generative restoration (modern deep learning,
`genrestore`).

## Goal of the day

After the day a student can:

1. restore a blurred, noisy image with a **truncated inverse**, a **Wiener** filter and
   **Tikhonov** regularisation, predict from the filter's gain curve what the result will
   look like, and say what each assumes; derive the Wiener filter and the Tikhonov normal
   equations *(math)*;
2. write a restoration as **data term + λ·penalty**, explain what the penalty adds (what we
   know about the object), and choose λ by the **discrepancy principle**, the **L-curve** or a
   test image with known truth; read Tikhonov and truncation as **filter factors** on the
   singular values *(math)*;
3. name **priors beyond Tikhonov** (non-negativity, total variation, sparsity in a basis) by
   what they keep and what they destroy (steps, ramps, fine texture);
4. run **Richardson–Lucy** by hand for one step, explain why it suits photon-counting data
   (non-negative, flux-preserving), and explain **semi-convergence**: the number of
   iterations is a regularisation parameter; derive RL from the Poisson likelihood
   *(math)*;
5. explain what 3D deconvolution does for **widefield** and **confocal** stacks, and why it
   cannot fill the **missing cone**;
6. predict the effect of a **wrong PSF** (too narrow: leftover blur; too wide: ringing and
   halos), and say what **blind deconvolution** tries and why it needs constraints;
7. recognise the **artefacts** of restoration (ringing, speckles, halos, border wrap) and
   compare simple **denoisers** (Gaussian, median, bilateral, non-local means) by what they
   destroy, using the **method noise**;
8. treat a restored image as an **estimate**: say which measurements it changes (peaks,
   sizes and separations near the resolution limit, object counts) and which it keeps
   (total flux over a region larger than the PSF), and what to report.

Why it fits this audience: it is the payoff of Part I. The forward model of day 4 is a
linear map that the Fourier transform diagonalises; restoration is solving a badly
conditioned linear system, and regularisation is the textbook fix, readable through the
SVD they know. Richardson–Lucy adds one piece of probability (the Poisson likelihood of
`cameranoise/`). The data + λ·penalty pattern met here returns on the graph-cut day and in
four electives.

## Prerequisites and overlaps

Recall, don't re-teach: each item below gets at most one recall slide with a pointer.

| Concept | Owner | Here |
|---|---|---|
| Forward model $g = \text{sample}(f\ast h) + n$, PSF, OTF/MTF, cut-off, Nyquist, 3D PSF and missing cone (picture), bead-measured PSF | Image formation (day 4, `plan/imageformation-day.md`) | Unit 1 opens with one "yesterday" slide. Unit 3 reuses the missing-cone picture and adds what deconvolution does about it. |
| Inverse problem, ill-posed, invisible objects, blur as a circulant matrix, condition number; naive inverse explodes | Image formation, unit 3 `illposed/` | Unit 1 recalls the explosion in one slide (same `deconvlab` presets) and starts from "what must we add?". The $(1,2,1)/4$ example and the $2\times2$ blur continue here with λ. |
| Poisson shot noise, read noise, offset, SNR, **PSNR**, flat-field | `cameranoise/` (day 3) | Unit 1: SNR per frequency for Wiener; the noise level σ for the discrepancy principle (unit 2); unit 3 uses the Poisson likelihood and needs an offset-corrected, non-negative image; PSNR in all comparisons (peak explicit). Variance stabilisation: see open questions. |
| Convolution, Laplacian kernel, border rules | `kernels/` (day 1) | Laplacian as the roughness penalty (unit 1); "wrap" border rule as the cause of FFT border ringing (one line) |
| SVD, least squares, normal equations | linear algebra (audience.md) | Unit 2 recalls in one line; filter factors are the new part |
| PCA | `pca/` (day 1) | Not needed; no recall |
| Energy = data term + λ·smoothness for **labels** | Graph cuts `seg-energy/` (day 13) | This day is the course's first data + λ·penalty (continuous unknowns). `seg-energy/` recalls it: "the same form, but each pixel takes one of two labels". λ is the regularisation weight (course convention). |
| Total variation, MRF priors | Ising/MRF elective (`gmrf/`, owns TV) | Unit 2: one concept slide (TV keeps steps, staircases ramps) and a 1D demo option; no algorithm. Pointer to the elective. |
| Sparsity prior, wavelet shrinkage | Wavelets elective (`shrinkage/`) | Unit 2: one line ("a penalty on coefficients in a basis") in the priors slide; unit 4 one line in the denoiser comparison |
| Iterative reconstruction, SIRT semi-convergence, MLEM | Radon elective | Radon unit 4 recalls units 2–3 here (early stopping, MLEM = RL with the projector). Here: one forward-pointer line. |
| Linear unmixing, least squares with a mixing matrix | Spectral unmixing elective | Not needed. Stain "colour deconvolution" is unrelated despite the name: one sentence in unit 1's notes. |
| Median / rank filters | Morphology (rank filters as the family between min and max) | Unit 4 introduces the median as a **denoiser** (one slide); morphology places it among the rank filters, thresholding recalls it. See open questions. |
| Learned denoising, generative restoration, hallucination | Modern deep learning (`genrestore`, optional `learned-restoration/`) | Unit 4: one pointer slide, no names ("networks trained on noisy/clean pairs now beat all of these; they can invent structure"). Wiener from unit 1 is the honest baseline in `genrestore`. |
| Object-level evaluation, statistics of measurements | Evaluation and statistics (day 16) | Unit 4's "what may be measured" is about restoration only; one forward pointer |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.
Every lecture starts with the "About this lecture" slide ending in the standard line on
**The math** slides. Math slides are titled `## The math: … {.math}`; concept slides and the
summary stand without them.

### 1. `deconv/`: Linear deconvolution: inverse, Wiener, Tikhonov

Concept layer:

- Recall (one slide, day 4): dividing by $H$ is perfect without noise and explodes with it;
  beyond the cut-off there is nothing to divide by. "What must we add?" Today's answer,
  step by step: first **give up** where the data are hopeless, then **trust each frequency
  as much as its SNR allows**, then state what we believe about the object as a **penalty**.
- **Truncated inverse**: divide only where $|H| > \varepsilon$, set the rest to 0. The
  simplest regularisation. Price: the sharp cut in frequency gives **ringing** next to edges
  (recall: a sharp cut in frequency is a sinc in space, `fourier/`). ε small: noisy; ε large:
  blurred.
- **Wiener filter**: per frequency, trust the data as much as its SNR allows. Where signal
  ≫ noise, invert; where noise ≫ signal, damp to zero; in between, a smooth roll-off (less
  ringing than truncation). Gain $\dfrac{H^*}{|H|^2 + 1/\mathrm{SNR}(f)}$. Numbers: $H = 0.5$,
  SNR 100 → gain 1.92 (≈ inverse, 2); $H = 0.05$ → gain 4 instead of 20. It needs the
  **signal and noise power**: noise from `cameranoise/`; natural images have power falling
  roughly like $1/f^2$, so the SNR falls with frequency even where $H$ does not.
- **Tikhonov**: choose the image that fits the data and is not too wild: minimise
  misfit + λ·penalty. Penalty = total energy (prefer small values) or roughness (the
  Laplacian of `kernels/`: prefer smooth images). λ small: noisy; λ large: blurred. With
  the energy penalty it **is** Wiener with a flat signal spectrum ($\lambda = 1/\mathrm{SNR}$);
  with the roughness penalty the damping grows with frequency, like Wiener with $1/f^2$.
  Choosing λ is unit 2.
- Three ways to read the same result: as a **gain curve** (how much of each frequency is
  kept), as the **effective transfer** $|W H|$ (what the whole chain, optics + restoration,
  now passes: a wider band than the optics, never past the cut-off), and as an image.
  Students predict the image from the curve before the demo shows it.
- **Practicalities** (one slide each): the PSF must be correct (a 20% wrong width gives
  ringing or leftover blur; unit 3 treats this fully), sampled at least at Nyquist (day 4),
  normalised to sum 1. FFT methods treat the image as **periodic**: a bright object at the
  right edge leaks to the left, and the jump at the border rings. Fix: pad and taper the
  borders (pointer to the "wrap" border rule of `kernels/`).
- Teacher note (not on a slide): simulating and inverting with the same discrete PSF is the
  "inverse crime" and flatters every method; the demo simulates the blur at twice the
  resolution and with a slightly different PSF in the "realistic" presets. Also in the
  notes: stain "colour deconvolution" (unmixing elective) is unrelated despite the name.
- **Demo `deconvlab`** (new, main demo; also embedded on day 4 with
  `data-mode="inverse"`): object (synthetic cell phantom as truth, cell micrograph,
  Siemens star, beads + edges), true PSF (Gaussian σ / Airy NA / motion, from `lib/psf.ts`),
  noise (Gaussian σ or photon count, seeded), method (inverse / truncated inverse with ε /
  Wiener with SNR slider and "flat" or "$1/f^2$" signal model / Tikhonov with λ slider,
  penalty energy or roughness), a separate "PSF used for deconvolution" width slider
  (mismatch), border handling (periodic / padded and tapered). Panels: observed, restored,
  error vs truth (diverging map), and a radial frequency plot: $|H|$, filter gain $|W|$,
  effective transfer $|WH|$, noise floor. Plot of error vs regularisation parameter (sweep
  of ~30 values, current value marked). Readouts RMSE and PSNR (`lib/metrics.ts`, peak
  explicit as in `cameranoise/`), fraction of negative pixels. Presets: "0.1% noise:
  inverse explodes" (recall), "truncated inverse: ringing", "Wiener at the right SNR",
  "Wiener with the wrong SNR", "$1/f^2$ beats flat", "too much regularisation: blur",
  "Tikhonov = Wiener with a flat spectrum", "roughness penalty", "wrong PSF width",
  "periodic borders ring", "motion blur: Wiener bridges the zeros".
- By hand: inverse and Wiener gains for given $H$, SNR; truncation with ε = 0.1 on a gain
  table; Tikhonov for the 2-pixel blur of day 4,
  $A = \begin{bmatrix}0.75 & 0.25\\0.25 & 0.75\end{bmatrix}$ (gains 1 and 0.5 on the sum and
  the difference), data $\mathbf y = (1, 0.5)$, energy penalty: λ = 0 gives
  $\hat{\mathbf x} = (1.25, 0.25)$, λ = 0.1 gives $(1.04, 0.33)$, λ = 1 gives
  $(0.48, 0.28)$; predict the restored image from a given gain curve (four sketches to
  match).

Math layer: *The math: the Wiener filter* (minimise $E|F - WG|^2$ per frequency,
independent signal and noise → $W = H^*S_f/(|H|^2S_f + S_n)$; the remaining error per
frequency $S_fS_n/(|H|^2S_f + S_n)$); *The math: Tikhonov* ($\min\|A\mathbf x - \mathbf y\|^2
+ \lambda\|D\mathbf x\|^2$ → $(A^\top A + \lambda D^\top D)\mathbf x = A^\top\mathbf y$, diagonal
in Fourier space for circulant $A$, $D$: $\hat x = H^*G/(|H|^2 + \lambda|\hat D|^2)$; equals
Wiener when $\lambda|\hat D|^2 = S_n/S_f$; gain at $f = 0$ is $1/(1+\lambda)$ for the energy
penalty but exactly 1 for the roughness penalty, since $\hat D(0) = 0$: the roughness
penalty keeps the mean).

### 2. `regularise/`: Regularisation: what we add, and how much

The general lesson of unit 1, taught once for the course (see open questions on ownership).

Concept layer:

- **The pattern**: every restoration so far is "fit the data" + λ·"look like what we
  expect". The data term says how the image was made (the forward model, the noise); the
  penalty says what we know about the object; λ says how much we trust each. The same
  pattern returns on the graph-cut day (labels instead of intensities), in tomography, in
  unmixing and in the electives on MRFs and wavelets.
- **What the penalty can say** (one slide, a 1D step-and-ramp profile restored with each):
  small values (energy); smooth (roughness, Tikhonov); **never negative** (a constraint, not
  a penalty: photons are not negative); **mostly piecewise constant** (total variation:
  penalise the size of jumps, not their square, so a sharp step costs no more than a ramp;
  keeps edges, turns ramps into staircases, shrinks small bright spots; owner: Ising
  elective); **few non-zero coefficients in a basis** (sparsity, wavelets elective). The
  prior is a modelling choice: choose it from what the object is, and expect it to impose
  itself where the data are weak.
- **Choosing λ** (the question every student asks):
  - **by eye**: common and not reproducible; fine for display, not for measurement;
  - **discrepancy principle**: the residual (data − blurred estimate) should be as large as
    the noise, not smaller. Too small: the estimate fits the noise. Needs the noise level
    σ, which `cameranoise/` showed how to measure (offset frames, flat regions). Numbers:
    256×256 pixels, σ = 2 grey levels: choose λ so the RMS residual is ≈ 2;
  - **L-curve**: plot size of the residual against size of the penalty for many λ, both on
    log axes. Too little regularisation: the penalty explodes for a small gain in fit; too
    much: the fit worsens for a small gain in smoothness. Pick the **corner**. Needs no σ,
    but the corner can be vague;
  - **known truth** on a test image (phantom or bead slide) with the same microscope and
    noise: the honest benchmark; transfers only if the test resembles the real data.
  - One line: generalised cross-validation exists (leave a pixel out, predict it); not
    taught.
- **Filter factors** (concept version, a picture): every linear method keeps a fraction of
  each frequency (each singular direction): the inverse keeps all (1, and divides), the
  truncated inverse keeps 1 or 0, Tikhonov rolls off smoothly around $|H|^2 \approx λ$.
  Regularisation = choosing this curve.
- **Early stopping as regularisation**: an iterative method that starts from a flat image
  and improves the fit step by step picks up the strong (low) frequencies first and the
  weak (high, noisy) ones last. Stopping after $k$ steps is a filter whose roll-off moves
  with $k$: **the number of iterations plays the role of λ**. Error to the truth: first
  falls, then rises (**semi-convergence**); the data fit keeps improving. Prepares unit 3
  (Richardson–Lucy) and is recalled by the Radon elective (SIRT).
- **A probabilistic reading** (one concept slide): data term = "how likely are these data
  if this were the object" (noise model); penalty = "how likely is this object before
  looking" (prior). The regularised solution is the **most probable object** given both.
  Forward pointer: a learned prior is the modern deep learning day's (`genrestore`).
- **Demo `regpath`** (new, 1D so that every λ is instant and the curves are exact): a 1D
  profile (synthetic: a step, a ramp, two close peaks, a narrow peak; or a line across the
  cell phantom), Gaussian blur width, noise σ (seeded). Penalty menu: energy / roughness /
  TV (smoothed) / roughness + non-negativity. λ on a log slider. Tab *iterate* replaces λ
  by an iteration count $k$ (Landweber steps, with optional non-negativity) on a log slider.
  Panels: truth, data and estimate; the residual (should look like noise: a running
  standard deviation band ±σ drawn); filter factors vs frequency (quadratic penalties and
  Landweber); **L-curve** with the current λ as a moving dot and the corner marked;
  **error vs λ** (or vs $k$) with the discrepancy-principle λ and the L-corner λ marked.
  Presets: "λ too small: fits the noise", "λ too large: blurred", "discrepancy principle",
  "L-curve corner", "the two rules disagree" (correlated noise), "roughness keeps the mean,
  energy shrinks it", "TV keeps the step, staircases the ramp", "TV shrinks the narrow
  peak", "non-negativity removes the dips", "iterations as λ: semi-convergence".
- By hand: filter factors $s^2/(s^2 + \lambda)$ for $s = 1, 0.5, 0.1$ and λ = 0.01, 0.1;
  Landweber factors $1 - (1 - s^2)^k$: for $s = 0.5$, $k = 1, 4, 16$ give 0.25, 0.68,
  0.99; for $s = 0.1$, $k = 16$ gives 0.15 and $k = 100$ gives 0.63 (weak frequencies need
  many iterations: that is why stopping early regularises); read λ off a given L-curve and a
  given residual-vs-λ table with known σ; match four restored step-and-ramp profiles to
  energy / roughness / TV / non-negativity.

Math layer: *The math: filter factors* (SVD $A = U\Sigma V^\top$; any linear restoration is
$\hat{\mathbf x} = \sum_i \phi_i\,\frac{\mathbf u_i^\top\mathbf y}{s_i}\mathbf v_i$; inverse
$\phi_i = 1$, truncated SVD $\phi_i \in \{0, 1\}$, Tikhonov $\phi_i = s_i^2/(s_i^2 + \lambda)$;
for a circulant blur $s_i = |H(f)|$ and the $\mathbf v_i$ are Fourier waves; the noise in
each term is $\phi_i n_i/s_i$); *The math: choosing λ* (discrepancy: $\|A\hat{\mathbf x}_\lambda
- \mathbf y\|^2 = N\sigma^2$, monotone in λ so one root; L-curve: $(\log\|A\hat{\mathbf
x}_\lambda - \mathbf y\|, \log\|D\hat{\mathbf x}_\lambda\|)$, corner of maximal curvature);
*The math: early stopping* (Landweber $\mathbf x_{k+1} = \mathbf x_k + \omega A^\top(\mathbf y
- A\mathbf x_k)$ from $\mathbf x_0 = 0$ has $\phi_i = 1 - (1 - \omega s_i^2)^k$; for small
$\omega s_i^2$ roughly Tikhonov with $\lambda \approx 1/(\omega k)$); *The math: MAP* (Gaussian
noise and a Gaussian prior: $-\log p(\mathbf x\mid\mathbf y) = \frac{1}{2\sigma^2}\|A\mathbf x
- \mathbf y\|^2 + \frac{1}{2\tau^2}\|D\mathbf x\|^2 + \text{const}$, so $\lambda =
\sigma^2/\tau^2$; TV and $\ell^1$ penalties as Laplace priors, one line).

### 3. `richardson-lucy/`: Richardson–Lucy, 3D, and the PSF you use

Concept layer:

- Linear filters can give **negative** intensities and do not know that photons are
  counted. **Richardson–Lucy**: start from a flat guess; blur it; compare with the data
  (ratio data / blurred guess); spread the ratio back with the PSF; multiply the guess by it.
  Repeat. The guess stays non-negative and keeps the total number of photons.
- By hand (1D, circular, 4 pixels): PSF $(0.25, 0.5, 0.25)$, data $(1, 2, 1, 0)$, start
  $(1,1,1,1)$: one step gives $(1, 1.5, 1, 0.5)$, a second $(0.9, 1.95, 0.9, 0.25)$. Sum 4 =
  sum of the data each time; already sharper.
- **Iterations as regularisation** (recall unit 2): early iterations recover the strong low
  frequencies, later ones the weak high frequencies, and then the noise. Semi-convergence:
  stop early, or add a prior (TV-regularised RL: one line). Fewer photons → stop earlier.
- Needs an **offset-corrected**, non-negative image (recall `cameranoise/`: subtract the
  camera offset, keep the photon scale) and a **background** term (a constant added in the
  model), otherwise RL sharpens the background into speckles.
- **3D** (three slides): widefield stacks are where deconvolution is most used. Recall the
  elongated 3D PSF and the missing cone (day 4). 3D RL with the 3D PSF puts out-of-focus
  light back where it came from: contrast and axial resolution improve a lot. But the
  missing cone is **not filled**: a thin horizontal sheet (all its frequencies along the
  axis) stays smeared in $z$; vertical structures are restored better than horizontal ones.
  Confocal: the pinhole already removed most out-of-focus light and its OTF has no missing
  cone, so deconvolution gains less but works more reliably (better SNR per frequency
  where it matters). The 3D PSF must match the depth: refractive index mismatch makes it
  asymmetric along $z$ and depth-dependent (one picture).
- **The PSF you use** (two slides): a PSF that is **too narrow** leaves blur (under-
  restored, looks safe); **too wide** gives over-sharpening, dark rims and **halos** around
  bright objects, ringing, objects that look smaller than they are. Measured PSF (from beads,
  day 4) includes the real aberrations but is noisy; computed PSF is clean but ideal. A
  common compromise: fit a model PSF to the measured beads.
- **Blind deconvolution** (one slide): estimate PSF and object together, alternating RL
  steps on each. Very under-determined (the trivial answer: PSF = point, object = the
  blurred image), so it needs constraints: PSF non-negative, sum 1, limited size, smooth or
  of a parametric shape. Use it when no PSF can be measured (motion, unknown aberration);
  prefer a measured or computed PSF otherwise.
- Artefacts (catalogue, one slide with thumbnails): **ringing** next to sharp edges and
  bright objects, **speckling** (noise turned into bright dots) after many iterations,
  **halos** from a wrong PSF, **border** artefacts, axial **smearing** from the missing cone.
- Forward pointer (one line): for emission tomography the same algorithm with the projector
  instead of the PSF is MLEM (Radon elective).
- **Demo `rlsteps`** (new, main demo), modes via `data-mode`:
  - `2d` (default): object (beads of brightness 1:4 next to an edge, Siemens star, cell
    phantom), PSF (Gaussian / Airy), photons (Poisson, seeded), background, PSF used:
    true / too narrow / too wide (width slider) / blind (starts from a Gaussian guess and
    alternates). Iteration slider (log scale, 0–1000). Panels: observed, RL estimate, error
    vs truth, a line profile through the bead pair and the edge (ringing visible); for
    blind, the current PSF estimate next to the true one. Plot vs iteration (log x): error
    vs truth (U-shaped with noise) and data misfit (falling); the best Wiener error from
    unit 1 as a dashed line. Readout: peak ratio and integrated ratio of the two beads
    (truth 4.0), total flux. Presets: "no noise: keeps improving", "noise: semi-
    convergence", "fewer photons: stop earlier", "1000 iterations: speckles", "ringing at
    the edge", "forgot the background", "PSF too wide: halos", "PSF too narrow: still
    blurred", "blind: PSF collapses to a point without constraints", "blind with a size
    limit", "RL vs best Wiener".
  - `xz`: a small 3D volume (64×64×32, synthetic: beads, a vertical filament, a horizontal
    sheet, a hollow sphere), widefield or confocal 3D PSF (from `lib/psf.ts`), photons,
    iterations. Panels: xz and xy slices of truth, observed and RL estimate; the xz OTF
    support with the missing cone outlined; axial profile through a bead and through the
    sheet. Presets: "widefield: blur from above and below", "RL restores the bead in
    $z$", "the sheet stays smeared: missing cone", "confocal: less to gain, nothing
    missing", "z-steps too large".
  - Computation: snapshots at ~30 log-spaced iterations are computed progressively in
    animation-frame chunks after any change, and the slider scrubs through them (see
    engineering).
- By hand: one and two RL steps (above), and check that the sum is kept; given error-vs-
  iteration curves for 100 and 1000 photons, which needs fewer iterations and why; which
  way a 20% too wide PSF biases the measured FWHM of a bead; why blind deconvolution
  without constraints returns the input image.

Math layer: *The math: Richardson–Lucy from the Poisson likelihood* (maximise
$\sum_i y_i\log(Ax)_i - (Ax)_i$; the fixed-point condition $A^\top\big(y/(Ax)\big) = A^\top
\mathbf 1$ gives the multiplicative update $x \leftarrow x \cdot A^\top\big(y/Ax\big)$ for a
PSF summing to 1; it is an EM algorithm, so the likelihood never decreases; flux and
non-negativity are preserved; with background $b$: $Ax + b$ in the ratio); *The math: why
early stopping regularises here too* (linearised around a flat image, the $k$-th iterate
damps frequency $f$ by about $1 - (1 - |H(f)|^2)^k$: Landweber's factors from unit 2);
*The math: the missing cone* (widefield OTF support in $(k_r, k_z)$: a doughnut whose
cross-section is two lobes touching at the origin, zero inside a cone around the $k_z$ axis; a horizontal sheet has $k_r = 0$, so only $k_z = 0$
survives; confocal OTF = convolution of two such supports, which fills the cone).

### 4. `denoise/`: Denoising, artefacts, and what you may measure

Concept layer:

- **When there is nothing to un-blur**: with PSF = point, Wiener and Tikhonov become
  **denoisers** (unit 1, with $H = 1$): low-pass filters chosen by the SNR. The comparison,
  each method by what it destroys:
  - **Gaussian** (linear): blurs edges and noise alike;
  - **median** (rank filter: sort the window, take the middle): removes outliers
    (salt-and-pepper, hot pixels), keeps steps, rounds corners and removes lines thinner
    than half the window; one slide, owned here as a denoiser (morphology places it among
    the rank filters);
  - **bilateral**: average only neighbours that are close *and* similar in value: keeps
    edges, makes cartoons, fails when noise ≈ edge height;
  - **non-local means**: average pixels whose whole *patch* looks similar, wherever they
    are: best on repetitive structure, can smear unique fine detail;
  - **TV and wavelet shrinkage** (one line each, recall unit 2; pointers to the Ising and
    wavelets electives).
- The **method noise** check: noisy − denoised should look like pure noise; structure in it
  (an edge, a filament) means the filter removed signal. Poisson noise depends on the
  signal: either use a method that knows it (RL, Poisson data terms) or stabilise the
  variance first (pointer to `cameranoise/`, see open questions).
- **Learned restoration** (one pointer slide, no names): networks trained on noisy/clean
  pairs, or even on single noisy images, beat all of these; trained to "look right" they
  can **invent** structure. Modern deep learning day (`genrestore`) shows when, with
  Wiener from unit 1 as the honest baseline.
- **Honesty: a restored image is an estimate** (three slides). It depends on the PSF, the
  method, λ or the stopping point. It improves contrast and separation, but cannot recover
  what lies beyond the cut-off (day 4: invisible objects stay invisible). What happens to
  measurements:
  - **total flux** over a region larger than the PSF: kept (RL exactly; roughness-penalised
    Tikhonov exactly; Wiener and energy-penalised Tikhonov slightly shrink it);
  - **peak intensities**: rise non-linearly, depend on iterations: do not compare;
  - **sizes** (FWHM) of objects near the resolution limit: shrink with iterations towards a
    floor set by the method, not by the object;
  - **separations** of close objects: biased when the dip between them is weak;
  - **counts**: speckles and split objects can add false objects; ringing can add rims.
- So: segment or detect on the restored image if it helps, **measure intensities on the raw
  (offset- and flat-field-corrected) image** within the masks found, validate any
  measurement taken on restored data with a known-truth test (beads, phantom), and **report**
  method, PSF (measured or computed), λ or iterations, and show raw data next to the result.
  Never compare images restored with different settings. Forward pointer: statistics of
  measurements is the evaluation day's.
- **Demo `denoisers`** (new): image (cell crop 128², synthetic: steps, a ramp, a thin line,
  a repeated texture), noise type (Gaussian σ / Poisson photons / salt-and-pepper), method
  (Gaussian σ / median radius / bilateral $\sigma_s$, $\sigma_r$ / non-local means patch,
  search window, $h$ / Wiener with PSF = point / TV λ, if `lib/smooth.ts` from the Ising
  elective exists). Panels: noisy, denoised, method noise (noisy − denoised), error vs
  truth; a line profile across the step and the thin line; readout PSNR. Two-up mode to
  compare two methods on the same noise. Presets: "Gaussian blurs the step", "salt and
  pepper: median wins", "median eats the thin line", "bilateral keeps edges", "bilateral at
  high noise: cartoon", "non-local means on texture", "non-local means smears the unique
  spot", "method noise shows the removed edge", "best PSNR, wrong look" (PSNR and the eye
  disagree).
- **Demo `rlsteps`, `data-mode="measure"`**: a field of synthetic objects with known truth
  (bead pairs at 0.8–1.5× the Rayleigh distance, spots of known flux 1:2:4, disks of known
  diameter), Poisson noise; method (Wiener / Tikhonov / RL with iteration slider). A table
  updates live: peak ratio, integrated ratio, FWHM, measured separation, number of detected
  spots (a fixed threshold), each as raw / restored / truth, cells coloured by error.
  Presets: "integrals hold", "peaks lie", "sizes shrink to a floor", "close pairs: the
  separation is biased", "1000 iterations: extra spots".
- By hand: median vs mean of a 1D row with an outlier and a step
  ($(10, 10, 10, 200, 10, 10, 50, 50, 50)$, window 3); bilateral weights for a 3-pixel
  neighbourhood across a step (values 10, 10, 50, $\sigma_r = 10$: the far side gets weight
  $e^{-8} \approx 0.0003$); which denoiser for salt-and-pepper, for a thin filament, for a
  repetitive lattice; a trust table (peak / integral / FWHM / separation / count after RL):
  fill in "kept / changed / unknown".

Math layer: *The math: bilateral and non-local means* (weights
$w_{pq} = e^{-\|p-q\|^2/2\sigma_s^2}e^{-(I_p - I_q)^2/2\sigma_r^2}$; NLM weights from the patch
distance $e^{-\|P_p - P_q\|^2/h^2}$, normalised); *The math: flux preservation* (RL: sum the
update, using $A^\top\mathbf 1 = \mathbf 1$; linear filters: total flux is the $f = 0$
coefficient, multiplied by the gain at $f = 0$); *The math: the median minimises absolute
deviation* ($\arg\min_m\sum|x_i - m|$; why outliers do not pull it).

### 5. (optional / further reading) `deconv-practice/`: Deconvolution on real data

For groups with their own widefield or confocal data. Workflow as concept, no tools: check
sampling (day 4 Nyquist table), acquire beads with the same objective and medium, correct
offset and flat-field, choose a PSF (measured, computed, fitted), pad the stack, choose the
stopping point on a bead slide by semi-convergence, apply to the sample, run the method
noise check, keep the raw data. Depth-varying PSF (one slide: split the stack into slabs
with their own PSF). Exercise on a real stack + bead stack once licensed data exist (see
open questions); until then the `rlsteps` `xz` mode with a "realistic" preset (PSF simulated
at a different depth than assumed).

## Exercises: concept first, then math

Each `exercise.qmd`: concept questions (predict, explain, numbers) with a `Solutions`
callout; `### Math questions` with items tagged *(math)* and a `Solutions (math)` callout;
then `## Explore {#explore}` with the demo and 4–6 concrete questions. Examples:

- Unit 1 concept: Wiener gains; "the deconvolved image is full of fine dots: what went
  wrong?"; "the result has dark rims around bright cells: what went wrong?"; Tikhonov on
  the 2-pixel blur for three λ. Math: derive the Wiener filter; show Tikhonov with $D = I$
  equals Wiener with flat spectra; show the roughness penalty keeps the mean.
- Unit 2 concept: filter factors; read λ from an L-curve and from the discrepancy rule;
  "doubling the noise: does the best λ grow or shrink?"; match profiles to priors. Math:
  derive the Tikhonov filter factors from the SVD; Landweber filter factors; λ =
  σ²/τ² from the MAP.
- Unit 3 concept: one and two RL steps; error-vs-iteration curves for two photon counts;
  "which structures in a widefield stack does deconvolution restore least, and why?";
  PSF too wide vs too narrow. Math: derive the RL fixed point; prove flux preservation;
  show the RL update never makes a pixel negative.
- Unit 4 concept: trust table; which denoiser for salt-and-pepper, a thin filament, a
  lattice; median vs mean on a row; read a method-noise image. Math: the median minimises
  the sum of absolute deviations; bilateral weights.
- Explore examples: "Find the SNR setting at which Wiener's error is smallest and compare it
  with the true SNR (`deconvlab`)"; "Where do the discrepancy principle and the L-curve
  disagree, and which is closer to the best λ? (`regpath`)"; "At which iteration is the
  error smallest for 100 and 1000 photons? Explain (`rlsteps`)"; "Find a horizontal
  structure that RL cannot restore (`rlsteps`, xz)"; "Which measurement survives 1000
  iterations? (`rlsteps`, measure)"; "Find a noise level where bilateral is worse than
  Gaussian (`denoisers`)".

## Shared engineering

- **FFT**: `lib/fft.ts` (owner `fourier/`; moved there by whichever of day 4 / this day is
  built first). Needed here in addition to day 4's helpers: FFT-based circular correlation
  (RL's back-projection), a **3D FFT** (separable, power-of-two sizes) for the `xz` mode,
  padding with a cosine taper.
- `lib/psf.ts` (owner: image formation): 2D and 3D PSFs, widefield and confocal, OTF.
- `demos/src/lib/deconv.ts` (new, owner: this day): inverse, truncated inverse, Wiener
  (flat / $1/f^2$), Tikhonov (energy / roughness), all as per-frequency gains on a shared FFT
  of the data; Landweber with optional non-negativity; Richardson–Lucy (2D and 3D, with
  background) with an iteration callback (for snapshots and error curves); blind RL
  alternation with PSF constraints (sum 1, non-negative, support radius); 1D Tikhonov and
  TV deblurring for `regpath` (TV by iteratively reweighted least squares with conjugate
  gradients on 256 samples); L-curve corner (maximal curvature on the sampled curve) and
  the discrepancy λ (bisection).
- `demos/src/lib/denoise.ts` (new, owner: this day): Gaussian (from `lib/imageops.ts`),
  median (histogram-based for small integer radii), bilateral, non-local means (per-offset
  formulation with box sums from `lib/integral.ts`, so the cost is search window × pixels,
  not × patch size). 2D TV denoising is taken from `lib/smooth.ts` (owner: Ising elective)
  if it exists; otherwise the TV option is hidden.
- Reuse: `lib/imageops.ts` `loadGrey()`; `lib/random.ts` (`rng`, `gauss`, `poisson`);
  `lib/metrics.ts` (RMSE, PSNR with explicit peak); `lib/plot.ts` (line plots, log axes,
  the L-curve); `lib/colour.ts` (diverging map for error and method-noise images);
  `lib/synth.ts` for the Siemens star, beads, bead pairs, step/ramp/line test image, the
  synthetic cell phantom and the 3D test volume (add there, not locally).
- **Speed and Wasm** (measure first, `demos/README.md`):
  - `deconvlab`: the restoration is one product + one inverse FFT of 256² (a few ms): live
    on every slider move. The error-vs-parameter sweep (~30 inverse FFTs, ~100 ms) is
    computed in animation-frame chunks after the slider stops.
  - `regpath`: 1D, 256 samples; the quadratic penalties are closed form per λ (40 λ values
    for the L-curve: under 1 ms); TV by IRLS (20 outer × ≤ 50 CG steps) per λ: the TV
    L-curve is computed progressively.
  - `rlsteps` 2D: two FFT convolutions per iteration on 256² (~15–25 ms in TypeScript):
    1000 iterations do not fit live, so snapshots at log-spaced iterations are computed
    progressively in chunks (UI stays responsive, snapshots appear as they finish); 128²
    on slides if needed.
  - `rlsteps` xz: 64×64×32 = 131k voxels, two 3D FFT convolutions per iteration (estimated
    20–40 ms in TypeScript); 200 iterations progressively take several seconds. This is the
    first **Wasm** candidate.
  - `denoisers`: non-local means on 128² with an 11×11 search window is ~2–4 M box-sum
    operations: fine.
  - Write `demos/bench/deconvolution-bench.ts` (node) timing one RL iteration at 128²/256²,
    one 3D RL iteration at 64×64×32, and NLM. Move RL (FFT + multiply loop) to
    `demos/wasm/deconv/` only if an iteration exceeds ~30 ms at 256² (2D) or makes the 3D
    snapshots take more than ~5 s on a typical laptop, keeping the TypeScript version as
    the reference and testing that both agree.
- Tests: `demos/test/deconvolution-test.ts` (`make test`): Wiener with noise → 0 equals the
  inverse filter; Tikhonov gain matches the normal equations solved directly for an 8-pixel
  1D case and the by-hand 2-pixel values; roughness-penalised Tikhonov keeps the mean;
  Landweber iterates match the filter-factor formula; the discrepancy λ gives the target
  residual; RL keeps the sum and non-negativity, reproduces the by-hand steps
  $(1, 1.5, 1, 0.5)$ and $(0.9, 1.95, 0.9, 0.25)$, and its likelihood never decreases; 3D RL
  equals 2D RL on a volume constant along $z$ with a matching PSF; median/bilateral/NLM match
  brute-force versions on 8×8.
- Images: a **synthetic cell phantom** (seeded, from `lib/synth.ts`) is the ground truth
  for all error curves; `demos/src/fourier2d/cell256.png` (credited) is offered for looks,
  with the caption that it is itself already blurred. Every demo: presets + reset, labels
  ≥ 18 px, fits 1150×560, `data-*` settings, keyboard back to reveal.js, static fallback
  figure in each deck, no runtime network, fixed seeds.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture (recall slide from day 4), Wiener gains and the 2-pixel Tikhonov by hand, `deconvlab` |
| 10:45–12:15 | Unit 2 lecture, filter factors by hand, `regpath` (λ, L-curve, discrepancy, iterations) |
| 13:15–14:45 | Unit 3 lecture, one RL step by hand, `rlsteps` (2D, then xz) |
| 15:00–16:30 | Unit 4 lecture, `denoisers`, trust table with `rlsteps` measure mode; quizzes as wrap-up |

## Open questions

- **Regularisation ownership (coursemap open decision).** Recommendation: **yes**, this
  day owns "ill-posed inverse problems and regularisation": data term + λ·penalty with
  continuous unknowns, quadratic penalties, filter factors, choosing λ (discrepancy,
  L-curve, known truth), early stopping as regularisation, and the MAP reading. Image
  formation keeps only the diagnosis (ill-posed, condition number). Recalled, not
  re-taught, by `seg-energy/` ("the same form with labels; min cut instead of a linear
  solve"; `seg-energy/` keeps ownership of energy minimisation for **labelling**), Radon
  (SIRT semi-convergence, TV-regularised reconstruction), unmixing (conditioning of the
  mixing matrix), Ising `gmrf/` ($(I + \lambda L)\mathbf x = \mathbf y$ is Tikhonov
  denoising with a roughness penalty) and wavelets (shrinkage as the $\ell^1$ case). TV stays
  owned by Ising, sparsity by wavelets; this day names both in one concept slide. Suggested
  coursemap rows: "Regularisation of inverse problems (data + λ·penalty, choosing λ, early
  stopping) → Deconvolution"; "Energy minimisation for labelling (data + λ·smoothness) →
  Graph cuts (`seg-energy/`)".
- **Median filter (coursemap open decision).** Recommendation: the deconvolution day
  introduces the median **as a denoiser** (one concept slide in unit 4 plus the
  `denoisers` demo, which shows outliers removed, steps kept, thin lines eaten). It comes
  before thresholding and morphology, so it is the first place the median is needed with a
  demo. Morphology owns **rank filters** as a family (min / median / max) and places the
  median there; thresholding's current concept slide shrinks to a one-line recall. If the
  course is shortened and this day is merged into image formation without unit 4,
  thresholding keeps its slide.
- Unit 3 is full (RL, 3D, PSF mismatch, blind, artefacts: ~18 slides, a two-mode demo). If
  it overflows, 3D moves into the optional unit 5 (principle 6: split, don't cut), keeping
  one slide on the missing cone in unit 3.
- Is unit 2 too abstract for the weaker students? Its concept layer is pictures and one
  1D demo; the L-curve and filter factors could be the first slides to go to the math layer
  if testing shows it.
- Variance stabilisation (Anscombe) for Poisson noise: does `cameranoise/` teach it? If not,
  one line in unit 4.
- Real data: a widefield stack with a measured bead stack from the same objective would make
  units 3–5 more convincing than simulated data. Which data, under which licence? Until then
  all demos are simulated and say so.
- Other plans to update (not changed here): `plan/coursemap.md` (the two decisions above;
  the "(new, to be planned)" mark); `plan/moderndl-day.md` refers to Wiener/Tikhonov as
  "day 4": now this day; `plan/thresholding-day.md` (median slide, see above);
  `plan/bitdepth-day.md` says this day's position is "not yet set": it is directly after
  image formation.
- Learned restoration stays on the modern deep learning day; this day's pointer slide
  should use the same words ("posterior mean" vs "invented detail") once that unit is
  written.

## References

- N. Wiener (1949), *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*,
  MIT Press.
- A. N. Tikhonov and V. Y. Arsenin (1977), *Solutions of Ill-Posed Problems*, Winston.
- M. Bertero and P. Boccacci (1998), *Introduction to Inverse Problems in Imaging*, IOP
  Publishing (close to this day's level; Wiener, Tikhonov, Landweber, RL).
- P. C. Hansen (2010), *Discrete Inverse Problems: Insight and Algorithms*, SIAM (filter
  factors, L-curve, semi-convergence; shared with the image formation and Radon plans).
- P. C. Hansen (1992), Analysis of discrete ill-posed problems by means of the L-curve,
  *SIAM Review* 34(4).
- V. A. Morozov (1966), On the solution of functional equations by the method of
  regularization, *Soviet Math. Dokl.* 7 (discrepancy principle).
- W. H. Richardson (1972), Bayesian-based iterative method of image restoration, *JOSA*
  62(1); L. B. Lucy (1974), An iterative technique for the rectification of observed
  distributions, *Astron. J.* 79.
- L. A. Shepp and Y. Vardi (1982), Maximum likelihood reconstruction for emission
  tomography, *IEEE Trans. Med. Imaging* 1(2) (RL = EM; MLEM).
- T. J. Holmes (1992), Blind deconvolution of quantum-limited incoherent imagery:
  maximum-likelihood approach, *JOSA A* 9(7); D. A. Fish et al. (1995), Blind deconvolution
  by means of the Richardson–Lucy algorithm, *JOSA A* 12(1).
- P. Sarder and A. Nehorai (2006), Deconvolution methods for 3-D fluorescence microscopy
  images, *IEEE Signal Processing Magazine* 23(3) (widefield vs confocal, missing cone).
- N. Dey et al. (2006), Richardson–Lucy algorithm with total variation regularization for 3D
  confocal microscope deconvolution, *Microsc. Res. Tech.* 69(4).
- D. Sage et al. (2017), DeconvolutionLab2: an open-source software for deconvolution
  microscopy, *Methods* 115 (teacher reference only; no tool names in student material).
- L. I. Rudin, S. Osher and E. Fatemi (1992), Nonlinear total variation based noise removal
  algorithms, *Physica D* 60 (pointer; owned by the Ising plan).
- C. Tomasi and R. Manduchi (1998), Bilateral filtering for gray and color images, *ICCV*.
- A. Buades, B. Coll and J.-M. Morel (2005), A review of image denoising algorithms, with a
  new one, *Multiscale Model. Simul.* 4(2) (non-local means, method noise).
