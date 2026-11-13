# Plan: a day on diffusion models

Internal working doc (plan/ is not rendered). Status: core units built (2026-11-13); see "Build notes" at the end.
**Course map:** Part VII (requested 2026-11-12; this plan is
its first day). Entry after day 39 *Modern deep learning*, whose `diffusion/` unit is
the introduction this day builds on. The day also leans on day 38 (VAEs) and day 5
(deconvolution). Pending coursemap edit: the "Generative models" row changes from
"requested, not yet planned" to "planned (`plan/diffusion-day.md`)", and the owners table
gains the rows listed under *Overlaps*.

**Course map (2026-11-13, self-study restructure):** day 41 of 51, Part VII "Learning from examples". Hard prerequisites: Modern deep learning (day 39), Variational autoencoders (day 38), Deconvolution (day 5). Soft (see also): Ising model and Markov random fields (day 28), Gaussian mixtures (day 35). Source: `days.yml`.

## Goal of the day

After the day a student can:

1. describe the forward process by its **signal-to-noise ratio**: what a noise level
   destroys first (fine detail: high frequencies) and why generation therefore goes from coarse
   to fine; compare two **schedules** on a log-SNR axis;
2. explain why training a network to **denoise** at every noise level (**denoising score
   matching**) gives the **score** of the noised data, without ever knowing the data density;
   convert between noise, clean-image and score predictions;
3. run **ancestral (stochastic)** and **deterministic** sampling steps by hand, explain that
   both reach the same distributions by different paths, and predict what too few steps do
   (samples pulled to averages, too little spread);
4. **evaluate** generated images as a distribution: precision (are samples realistic?),
   recall (is the data covered?), a Fréchet distance between fitted Gaussians and what it
   cannot see, and a **memorisation** check against the training set;
5. explain **conditioning** and **classifier-free guidance** (one network, label dropped at
   random during training), and predict how the guidance weight trades diversity for
   typicality, up to samples more extreme than any real example;
6. explain **latent diffusion**: diffusion in an autoencoder's latent space (the VAE day),
   why it is cheaper, and that the finest detail then comes from the decoder;
7. use a diffusion model as a **prior for an inverse problem** (denoising, deblurring,
   super-resolution, inpainting, virtual staining) and contrast posterior samples with the
   **regularised estimate** of the deconvolution day; locate where generated detail comes
   from (the **null space** of the forward model and the parts under the noise), and check a
   restoration with data consistency, sample disagreement and coverage;
8. *(math)* prove the denoising score matching identity; derive Tweedie for a Gaussian prior
   and show the optimal denoiser is the Wiener filter; solve the probability-flow ODE for
   Gaussian data and the Euler error; derive the guided density $p(\mathbf x\mid y)^{1+w}p(\mathbf x)^{-w}$
   for Gaussians; show the posterior null-space component equals the prior's; state what the
   tractable likelihood approximation used for posterior sampling drops.

Why it fits this audience: every ingredient is already in the course. The forward process is
"add Gaussian noise" (day 3); the best denoiser is a conditional mean, and for Gaussian data it
is the Wiener filter (day 5); the score is a gradient (calculus); sampling is following a vector
field; inverse problems and the null space were set up on days 4–5. Analytic toy models
(Gaussian mixtures, exact scores) let every demo show the exact answer next to the
approximation. No architecture, library or product is taught as a tool.

## Prerequisite gap

- **`diffusion/`** (day 39, built): forward formula $\mathbf x_t = \sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\varepsilon$,
  linear and cosine schedules, "the best denoiser is an average", Tweedie (math slide), one
  DDPM and one DDIM step (math slide), memorising vs generalising, conditioning by Bayes and
  the guidance weight $w$, posterior mean vs sample (twice the error), hallucination, the five
  rules for what may be measured. **This day recalls each of these in at most one slide** and
  does not re-derive them; its demos `diffusion2d` and `genrestore` are embedded unchanged
  (existing presets) on recall slides only.
- **VAE day** (day 38, built: `autoencoder/`, `vae/`, `vae-gen/`): latent space, encoder
  means and spreads, the decoder outputs a mean (blur), the 2D cell-shape latent space of
  `vaeimages`. Unit 3 recalls them; its demo reuses the shape VAE's weights.
- **Deconvolution day** (day 5): Wiener/Tikhonov (`deconv/`), data + λ·penalty, the
  discrepancy principle and MAP (`regularise/` "A probabilistic reading"), denoisers and
  "what may be measured" (`denoise/`). **Image formation** (day 4): OTF cut-off
  (`resolution/`), invisible objects and the null space (`illposed/`). Unit 4 recalls them.
- **Fourier** (day 1, `fourier/`): power spectrum, low-pass. Unit 1 uses a radially
  averaged spectrum (recall in one line).
- **Neural networks** (day 36): loss, training, `unet/` (the usual denoiser shape; one line).
  **Evaluation** (day 37): `leakage/` (never test on training images) is recalled for the
  memorisation check.
- **Not assumed:** the GMM day (the mixture is restated as in `diffusion/`), the Ising
  day (`mcmc/` is a "see also" for Langevin steps), stochastic or ordinary differential
  equations (introduced as "follow the arrows, with or without jitter"; the SDE/ODE language
  is math layer only).

## Overlaps / who owns what

Ownership change from coursemap "Ownership changes" applied: **this day owns diffusion models
in depth**; `diffusion/` stays on day 39 as the **introduction** (decided here, not moved),
and only its "Where this leads" pointer is edited (see pending edits). Proposed new owners-table
rows: *score matching, samplers, guidance, latent diffusion, diffusion posterior sampling* →
this day; *evaluation of generated images (precision/recall of a distribution, Fréchet
distance, memorisation check)* → this day (unit 2).

| Day / unit | Owns, this day recalls | This day adds |
|---|---|---|
| Modern DL `diffusion/` | forward formula, schedules, Tweedie statement, one DDPM/DDIM step, conditioning by Bayes, $w$, mean vs sample (2×), memorisation (exact toy), hallucination examples, measurement rules | SNR/frequency view, DSM proof and parameterisations, ODE vs SDE, step count and spacing, inversion, generative evaluation, label-dropout guidance and over-guidance, latent diffusion, posterior sampling algorithms, null-space view, coverage |
| Modern DL `diffusion2d`, `genrestore` | the 2D sampler with exact score; exact posterior hallucination on 32×32 cells | Embedded unchanged on recall slides; **no new modes needed** (new demos below ask different questions) |
| Modern DL optional `learned-restoration/` (not built) | supervised restoration, blind-spot denoising | Unit 4 points to it for self-supervised denoising; not repeated |
| VAE day | latent space, ELBO, decoder blur, `vaeimages` shape VAE | Latent diffusion on that 2D latent (unit 3); math aside "diffusion is a VAE with a fixed encoder" |
| Deconvolution day | Wiener, Tikhonov, MAP, discrepancy principle, denoisers, measuring after restoration | "Wiener is the optimal denoiser for a Gaussian prior" (unit 1); posterior sampling vs regularised estimate (unit 4) |
| Image formation `illposed/`, `resolution/` | null space, cut-off, invisible objects | "generated detail lives in the null space" (unit 4) |
| Camera noise `cameranoise/` | SNR, PSNR | log-SNR of a schedule (unit 1) |
| Evaluation day | object matching, leakage, statistics | distribution-level metrics for generated images (unit 2); per-object metrics only recalled |
| Wavelets `shrinkage/`, deconvolution `denoise/` | classical denoisers | one line in unit 4: "any good denoiser can serve as a prior" |
| Ising `mcmc/` | MCMC, detailed balance | Langevin corrector as "see also" (unit 2) |
| Integrity unit (`integrity/`, not placed) | images as evidence | unit 4 points to it if it exists, as `diffusion/` does |

Topic page "Denoising" (coursemap): add this day's unit 1 (denoiser = score) and unit 4.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each, plus one optional unit. Folder names checked against the
repo: none exist.

### 1. `score-matching/`: Noise levels, denoisers and the score

*Guiding question: what exactly does a network learn when we teach it to remove noise?*

Concept layer:

- Recall `diffusion/` in one slide (forward strip, "the best denoiser is an average"), with
  `diffusion2d` embedded at `data-preset="forward"`.
- **Noise level as SNR**: $\text{SNR} = \bar\alpha/(1-\bar\alpha)$ (recall `cameranoise/`).
  Schedules compared on a log-SNR axis: they differ mainly in how much time they spend at
  each SNR. Equivalent view: $\mathbf x_t/\sqrt{\bar\alpha}$ = clean image + noise of size
  $\sigma = 1/\sqrt{\text{SNR}}$ ("noise level σ"), used from here on in concept slides.
- **What noise destroys first**: image power falls with frequency (recall `fourier/`); white
  noise is flat. At each noise level, frequencies whose power is below the noise floor are
  gone. So the forward process erases fine detail first and coarse structure last, and
  generation (the reverse) paints coarse layout first and fine detail last. Numbers: power
  $1000/f^2$, crossover at $f^\* = \sqrt{1000\,\text{SNR}}$: SNR 1 → 31.6, 0.1 → 10, 0.01 → 3.2
  cycles per image.
- **The best denoiser at one noise level**: for a Gaussian image model (only a power
  spectrum) it is exactly the **Wiener filter** of the deconvolution day with no blur: keep
  frequencies above the noise floor, shrink those below. For real images it is the
  conditional mean: sharper than Wiener where the data say what is there, an average where
  they do not.
- **Denoising score matching**, intuition first: make training pairs (clean image, noisy
  version, the noise we added). Regress the clean image (or the noise) on the noisy one with
  squared error. A squared-error learner converges to the **average target for each input**,
  i.e. the conditional mean, and by Tweedie that average is a step along the **score** (the
  direction in which the noisy data become more likely). We never need the density of
  images; the noise we added is the label. Picture: a scatter of (noisy, clean) pairs in 1D
  and the running average through it.
- **One network for all noise levels**: the noise level is an input. Three equivalent
  outputs: the noise $\hat{\boldsymbol\varepsilon}$, the clean image $\hat{\mathbf x}_0$, or the score; converting is
  arithmetic (worked example with the numbers of `diffusion/`'s forward step). Which output
  is trained changes how errors at different noise levels are weighted (one sentence; math
  slide).
- **Where learned scores fail**: far from the data and at low noise levels the training
  pairs are sparse, so the learned score is poor there; rare modes get little weight (recall
  the rare blob of `diffusion2d`).
- Pointer: the usual denoiser network is a U-Net (`unet/`) with the noise level fed in.

Math layer (`## The math: … {.math}`):

- Conditional score of the forward kernel: $\nabla_{\mathbf x_t}\log p(\mathbf x_t\mid\mathbf x_0) = -(\mathbf x_t - \sqrt{\bar\alpha}\mathbf x_0)/(1-\bar\alpha) = -\boldsymbol\varepsilon/\sqrt{1-\bar\alpha}$.
- **DSM identity** (Vincent 2011): $\mathbb E\|\mathbf s_\theta(\mathbf x_t) - \nabla\log p(\mathbf x_t\mid\mathbf x_0)\|^2 = \mathbb E\|\mathbf s_\theta(\mathbf x_t) - \nabla\log p_t(\mathbf x_t)\|^2 + C$ (expand the square; the cross term uses $\nabla p_t = \int \nabla p(\mathbf x_t\mid\mathbf x_0)p(\mathbf x_0)\,d\mathbf x_0$).
- Gaussian prior per frequency, $x_0\sim\mathcal N(0, P)$: $\mathbb E[x_0\mid x_0+\sigma\varepsilon] = \frac{P}{P+\sigma^2}(x_0+\sigma\varepsilon)$: the Wiener gain; Tweedie gives the same.
- Parameterisations: $\hat{\mathbf x}_0 = (\mathbf x_t - \sqrt{1-\bar\alpha}\hat{\boldsymbol\varepsilon})/\sqrt{\bar\alpha}$, $\mathbf s = -\hat{\boldsymbol\varepsilon}/\sqrt{1-\bar\alpha}$, $\mathbf v = \sqrt{\bar\alpha}\boldsymbol\varepsilon - \sqrt{1-\bar\alpha}\mathbf x_0$; $\|\boldsymbol\varepsilon - \hat{\boldsymbol\varepsilon}\|^2 = \text{SNR}\cdot\|\mathbf x_0 - \hat{\mathbf x}_0\|^2$ (so ε-training weights low-noise levels heavily).
- Aside (optional slide): the training loss is a weighted sum of denoising errors over SNR, which is the ELBO of a VAE whose encoder is the fixed noising process (link to `vae/`).

**Demo `scorematch1d`** (new; analytic + tiny precomputed model). 1D data: a mixture of 2–3
Gaussians (presets: two narrow modes, a wide + a narrow mode, a rare mode at 5 %). Panels on
one x-axis: (top) data density and the noised density $p_t$; (middle) a scatter of 400
seeded training pairs (noisy $x_t$ on x, clean $x_0$ on y) with the **exact** denoiser
$\mathbb E[x_0\mid x_t]$ drawn through it, a binned average of the scatter, and the **learned**
denoiser; (bottom) the score: exact, and the learned one via Tweedie. Slider: noise level
(log-SNR); toggle what the net predicts (clean / noise / score) with the same curve shown in
that unit. Learned denoiser: a tiny MLP $(x_t, \log\text{SNR}) \to \hat\varepsilon$ (2 → 32 → 32 → 1, tanh,
~1.2 k weights), three precomputed snapshots (200, 2 000, 20 000 steps) plus a *train 500 more
steps* button that runs live from the current snapshot (fixed seed; lib/nn.ts; a few ms per
100 steps). Readouts: mean squared denoising error of exact vs learned, the region where
training pairs are sparse (shaded). Presets: "the binned average is the denoiser", "high noise:
the denoiser points at the mean", "low noise: a staircase between modes", "under-trained: wrong
far from the data", "rare mode: the learned score ignores it", "predicting noise vs clean:
same curve".

**Demo `noisespectrum`** (new; analytic, image-sized). A 128×128 crop of `cell256` (credit
shown) and a synthetic nuclei image (`lib/synth.ts nuclei`). Slider log-SNR (and schedule
menu, a marker moving along a small ā(τ) plot). Panels: noisy image $\mathbf x_t/\sqrt{\bar\alpha}$; the
**Wiener denoiser** at that noise level (signal power from the image's own radially averaged
spectrum, the "Gaussian image model"); the radially averaged power spectrum with the noise
floor as a horizontal line and the crossover frequency marked; a **sample from the Gaussian
image model** (*resample* button: random phases with that spectrum). Presets: "fine detail
goes first", "SNR 1: what is left", "the Wiener denoiser at high noise is a blur", "a Gaussian
prior knows the spectrum, not the cells" (the sample looks like clouds). Cost: two 128² FFTs
per update (~5 ms).

By hand (concept): (a) SNR and σ for ā = 0.8, 0.5, 0.2 (4, 1, 0.25; σ = 0.5, 1, 2);
(b) crossover frequency for $P = 1000/f^2$ at SNR 1, 0.1, 0.01, and what a tenfold drop in SNR
does (÷√10); (c) Gaussian data with variance 4, noise level σ = 1: the best denoiser multiplies by
$4/5 = 0.8$; what happens to the multiplier as σ grows?; (d) a table of eight (noisy, clean)
pairs: average the clean values in two bins and compare with the denoiser curve;
(e) convert $\hat\varepsilon = -1$ at $x_t = 1$, $\sqrt{\bar\alpha} = 0.8$ into $\hat x_0 = 2.0$, score $1.67$ and $v = -2.0$.
*(math)*: (f) prove the DSM identity; (g) derive the conditional score of the forward kernel;
(h) Wiener gain as the optimal denoiser for a Gaussian prior, and check it with Tweedie;
(i) show $\|\boldsymbol\varepsilon-\hat{\boldsymbol\varepsilon}\|^2 = \text{SNR}\,\|\mathbf x_0-\hat{\mathbf x}_0\|^2$.

### 2. `diffusion-sampling/`: Samplers, steps, and judging the samples

*Guiding question: how do we get from noise to an image, how many steps does it take, and how
do we know the samples are any good?*

Concept layer:

- Recall in one slide: sampling = repeated denoising (`diffusion2d`, `data-preset="few"`).
- **Two ways down**, same destinations: a **stochastic** path (denoise, then add back a little
  fresh noise: ancestral sampling) and a **deterministic** path (follow the arrows of a smooth
  flow, no fresh noise). At every noise level both have the same distribution of points;
  single paths differ (jittery vs smooth). An interpolating knob (η: 0 = deterministic, 1 =
  ancestral).
- **Why steps matter**: each step follows the arrows in a straight line, but the true paths
  bend, most strongly near the data at low noise. Too few steps cut corners: samples land on
  averages and have too little spread (worked numbers below). Where to put the steps matters
  as much as how many: more steps near the data.
- **Stochastic steps forget**: fresh noise lets later steps correct earlier errors, at the
  price of needing more steps; deterministic steps carry errors to the end but need fewer.
  Extra correction steps at a fixed noise level (Langevin) are a random walk that keeps the
  distribution (see also `mcmc/` if taken).
- **Deterministic sampling is a map**: noise ↔ image, invertible. Run it backwards to find
  the noise that gives a real image ("inversion"), then edit or interpolate there. Contrast
  with the VAE encoder: no network learned for the way back.
- Cost is counted in **network evaluations**; current practice reaches a few to tens of steps
  by better step placement, higher-order steps, or training a student network to jump
  (distillation; one line).
- **Judging generated images as a distribution** (this day owns it): no single image is right
  or wrong; compare the cloud of samples with the cloud of real images.
  - **Precision**: what fraction of samples lie where real images lie (realism);
    **recall**: what fraction of real images have samples nearby (coverage). Neighbourhoods
    from k-nearest-neighbour radii.
  - **Fréchet distance**: fit one Gaussian to each cloud and compare means and covariances.
    Cheap and standard (image papers compute it on features from a pretrained network and call
    it FID: one-line pointer); blind to everything a Gaussian cannot see: a generator that puts
    all samples between two modes can score perfectly.
  - **Memorisation check**: distance from each sample to its nearest training image (recall
    `leakage/`: training images are not evidence of generalisation).
  - For science, evaluate **the measurement you will make**: compare histograms of nucleus
    area, counts, intensities between generated and held-out real images.

Math layer:

- Noise-level form $\mathbf x = \mathbf x_0 + \sigma\boldsymbol\varepsilon$: probability-flow ODE $d\mathbf x/d\sigma = -\sigma\nabla\log p_\sigma(\mathbf x)$; reverse SDE stated; both share the marginals $p_\sigma$ (Fokker–Planck stated, not derived).
- Gaussian data $\mathcal N(0, s^2)$: $d x/d\sigma = \sigma x/(s^2+\sigma^2)$, solution $x \propto \sqrt{s^2+\sigma^2}$; Euler steps shrink the spread (exercise numbers).
- DDIM with η (the family between deterministic and ancestral); recall the two step formulas of `diffusion/` and show they are discretisations of the ODE/SDE.
- Fréchet distance $\|\mu_1-\mu_2\|^2 + \operatorname{tr}(\Sigma_1+\Sigma_2-2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})$; in 1D $(\mu_1-\mu_2)^2 + (s_1-s_2)^2$. k-NN precision and recall definitions.

**Demo `samplerpaths`** (new; exact score, 1D data so time can be an axis). Background: the
noised density $p_\sigma(x)$ as a heat map over (noise level on x, position on y), exact for
a 1D mixture (presets: two modes, three unequal modes, Gaussian). Overlay: 30 highlighted
trajectories and 2 000 endpoint samples as a histogram against the true density at the right
edge. Controls: steps (1–200, log), η (0–1), step spacing (uniform in time / uniform in
log-SNR / dense near the data), solver (Euler / two-stage), Langevin corrections per step (0–3),
*seed*, *reset*. Toggle *invert*: drag a point on the data edge and see the noise it maps to
(deterministic only). Readouts: network evaluations, endpoint mean/sd vs truth, fraction between
modes, Fréchet distance and 1D Wasserstein distance to the truth. Presets: "same marginals,
different paths", "one Euler step: everything at the mean", "5 steps: too narrow", "spacing
fixes it", "stochastic forgets a bad start", "inversion and back". Cost: 2 000 paths × 200 steps ×
≤ 8 components ≈ 3 M updates (TS, a few ms); the heat map 300 × 200 × K once per preset.

**Demo `genmetrics`** (new; 2D, exact). The data clouds of `diffusion2d` (same mixtures, via
the shared definition, see engineering), 1 000 real points, 1 000 generated points from a
*generator* menu: exact sampler with N steps (slider) and η; "between the modes" (one wide
Gaussian matched in mean and covariance); "mode dropping" (one blob missing); "memoriser"
(copies of 50 training points with tiny jitter); "too narrow" (samples shrunk towards their
mode). Panels: the two clouds; the fitted Gaussian ellipses (`lib/plot.ts ellipse`); the
k-NN balls toggle; a bar chart of precision, recall, Fréchet distance and median
nearest-training distance; a small curve "metric vs steps". `data-mode="guidance"` (used by
unit 3): generator = exact sampler with class guidance, and the curve becomes precision/recall
vs $w$. Presets: "between the modes: perfect Fréchet distance, low precision", "mode dropping:
recall drops", "memoriser: perfect on every metric except nearest-training distance", "more
steps: all metrics improve, then flatten".

By hand (concept): (a) one deterministic step: $x_i = 1$, $\bar\alpha_i = 0.36$, $\hat\varepsilon = 0.5$, to
$\bar\alpha_{i-1} = 0.64$: $\hat x_0 = 1.0$, $x_{i-1} = 1.1$; (b) the stochastic step from the same
numbers: mean 0.969, variance 0.246 (sd 0.50); which one is more spread out over many draws?;
(c) Gaussian data sd 1, start at noise level σ = 3: one Euler step to σ = 0 gives spread 0.32,
two steps (3 → 1.5 → 0) give 0.54, the exact answer is 1; predict ten steps;
(d) Fréchet distance in 1D for these three sample spreads (0.47, 0.22, 0); (e) data at ±2 with sd
0.5, generator $\mathcal N(0, 4.25)$: Fréchet distance 0, yet about 37 % of the samples fall within ±1
of the centre where there is almost no data: which metric catches it?; (f) four generated images
with given nearest-training distances: which are suspicious? *(math)*: (g) solve the
probability-flow ODE for Gaussian data and derive the Euler factors; (h) show the DDIM step is
an Euler step of the ODE in the variables ($\mathbf x/\sqrt{\bar\alpha}$, σ); (i) derive the 1D
Fréchet formula from the general one.

### 3. `guidance/`: Conditioning, guidance and latent diffusion

*Guiding question: how do we ask for a particular kind of image, what does pushing harder do,
and why do most image generators work in a compressed space?*

Concept layer:

- Recall (one slide): conditioning by Bayes and the weight $w$ (`diffusion/`), with
  `diffusion2d` at `data-preset="guide"`.
- **Conditional denoisers**: the label (class, a treatment, a text, an image) is an extra
  input. Two older/newer routes to guidance: a separate **classifier** on noisy images whose
  gradient is added (needs a second network that works on noise), and **classifier-free
  guidance**: train one network, and drop the label at random (say 10 % of the time) so it
  learns both the conditional and the unconditional denoiser; at sampling time extrapolate
  from unconditional towards conditional by $w$.
- **What $w$ does**: it sharpens the class towards points where the label is *most certain*,
  not most typical. Moderate $w$: cleaner, more typical samples, less diversity (precision up,
  recall down: `genmetrics` guidance mode). Large $w$: samples more extreme than any real
  example (beyond the class mean, away from the other classes), saturated contrast. Worked
  numbers below. Guided diffusion does not even sample the sharpened density exactly (the demo
  shows both).
- **Conditioning on an image** (the bridge to unit 4): the condition is a measured image
  (a blurred, noisy, or other-channel image); trained on pairs. The same rules apply: strong
  guidance gives confident, typical, possibly wrong detail.
- **Latent diffusion**: an image of 256 × 256 × 3 = 196 608 numbers vs a latent of
  32 × 32 × 4 = 4 096 (48× fewer). Train an autoencoder first (the VAE day), then run diffusion
  on the latent codes, and decode at the end. Cheaper per step, and the diffusion model spends
  its capacity on layout and content instead of imperceptible pixel detail.
  - The **decoder** draws the finest detail: it is a mean (VAE day), so whatever the
    latent cannot represent is filled in by the decoder's habits, not by any measurement.
  - The latent codes of the training set are a cloud; a diffusion model learns that cloud
    instead of the VAE's "round Gaussian" prior, so samples land where real codes are (fixing
    the VAE's "nonsense far from the data", `vae-gen/` limits slide).
- Text conditioning uses attention between image patches and text tokens (one line, recall
  `attention/`).

Math layer:

- Classifier-free combination $\hat{\boldsymbol\varepsilon} = (1+w)\hat{\boldsymbol\varepsilon}(\mathbf x_t, y) - w\,\hat{\boldsymbol\varepsilon}(\mathbf x_t, \varnothing)$, equivalent (recall) to the score of $p(\mathbf x\mid y)^{1+w}p(\mathbf x)^{-w} \propto p(\mathbf x\mid y)\,p(y\mid\mathbf x)^{w}$.
- Gaussians: $p(x\mid y) = \mathcal N(m, s^2)$, $p(x) = \mathcal N(0, 1)$: the guided density is Gaussian with precision $(1+w)/s^2 - w$ and mean $\frac{(1+w)m/s^2}{(1+w)/s^2 - w}$; normalisable only if $s^2 < (1+w)/w$. Applying the combination at every noise level is not the same as sampling this density (the noised versions of a sharpened density are not the sharpened noised densities): one sentence + the demo.
- Latent: aggregate posterior $q(\mathbf z) = \frac1N\sum_i\mathcal N(\mathbf z; \boldsymbol\mu_i, \operatorname{diag}\boldsymbol\sigma_i^2)$ from the encoder; its noised score is a mixture score (exact in the demo).

**Demo `guidance1d`** (new; exact, 1D). Two or three classes, each a 1D Gaussian mixture,
overlapping (presets: two overlapping Gaussians at ±1, a class with two sub-modes, a rare
class). Panels: class densities, the **sharpened density** $p(x\mid y)p(y\mid x)^w$ (exact,
normalised numerically on a grid), and a histogram of 2 000 samples from **guided diffusion**
(classifier-free combination of exact conditional and unconditional mixture scores at every
step, `lib/diffusion.ts`). Slider $w$ (0–10), class menu, *label dropout* illustration (the
unconditional score drawn as the same curve with the label removed). Readouts: sample mean and
sd of the chosen class vs unguided, fraction of samples beyond the class mean, fraction beyond
the most extreme 1 % of real class members. Presets: "w = 0: the class as it is", "w = 3:
narrower, pushed away from the other class", "w = 8: more extreme than real data", "guided
sampling ≠ sharpened density", "the sub-mode near the other class disappears".

**Demo `latentdiff`** (new; exact score in a real VAE latent). The 2D cell-shape latent of the
VAE day (`vaeimages` shape VAE, decoder in `lib/q8.ts` format, imported unchanged). Data:
precomputed encoder means and spreads of 800 synthetic training shapes with their kind
(round, elongated, dividing, protrusion), bundled (~20 KB). Left: the latent plane with the
codes coloured by kind, the reverse-diffusion paths of 16 samples (exact score of the noised
aggregate posterior, deterministic or stochastic, steps slider), the VAE prior circle for
contrast. Right: a 4 × 4 grid of decoded samples; a bar chart of the kind proportions among 400
samples (nearest-code kind) vs the training set. Controls: kind (none / one of four) and
guidance $w$; *spread*: encoder spreads vs tiny spreads (memorising codes: decoded samples are
reconstructions of training shapes, with the nearest-training distance shown); *source*:
latent diffusion vs "VAE prior" (decode $\mathcal N(0, I)$ draws). Presets: "VAE prior: some
nonsense shapes", "latent diffusion: shapes where real ones are", "dividing cells on demand",
"strong guidance: caricature dumbbells", "tiny spreads: it only copies", "the decoder draws the
outline (zoom on an edge)". Cost: 400 samples × 100 steps × 800 components ≈ 32 M operations
(~30–60 ms, run on button press, not on every slider tick; or subsample to 200 components
for live paths; measure before considering Wasm).

By hand (concept): (a) the classifier-free combination with $\hat\varepsilon_\text{uncond} = 0.2$,
$\hat\varepsilon_\text{cond} = 0.5$, $w = 3$: 1.4; (b) classes $\mathcal N(\pm1, 1)$ with equal
weights, $p(A\mid x) = 1/(1+e^{-2x})$: compare the guided density of class A at $x = 2$ and
$x = 1$ for $w = 0, 4, 8$ (ratios 0.61, 0.94, 1.45): from which $w$ is a point beyond the class
mean more likely than the class mean?; (c) latent compression factor 48 and what it means for
the cost per step; (d) a generated cell from a latent model shows a crisp 1-pixel membrane:
where did that membrane come from?; (e) predict the precision/recall curve against $w$.
*(math)*: (f) derive the classifier-free combination from Bayes; (g) the guided Gaussian for
$m = 1$, $s^2 = 0.5$, $w = 1$: mean 4/3, variance 1/3; when is it not normalisable?;
(h) write the score of the noised aggregate posterior.

### 4. `diffusion-inverse/`: Diffusion priors for inverse problems, and where detail comes from

*Guiding question: if a diffusion model restores, deblurs or "virtually stains" my image, which
parts of the result were measured and which were imagined?*

Concept layer:

- Recall (one slide each): the regularised estimate of the deconvolution day (data term +
  λ·penalty = MAP with a hand-made prior, `regularise/`), and `diffusion/`'s mean vs sample
  with `genrestore` at `data-preset="meanvs"`.
- **Swap the hand-made prior for a learned one**: the same Bayes, but the prior is "images
  like the training images". Three answers to return: the most probable image (MAP), the
  **posterior mean** (cautious, blurry where unsure), **posterior samples** (sharp, varied).
  For a Gaussian prior all three coincide with Wiener/Tikhonov; with a realistic prior they
  separate.
- **How to sample the posterior with an unconditional model**: at every step add a push
  towards agreeing with the measurement. The exact push needs "how well would this noisy
  intermediate explain the data", which is unknown; the usual trick asks how well its **current
  clean estimate** explains the data (cheap, approximate, over-confident early on). For holes
  (inpainting) a simpler trick pastes the re-noised measured pixels at every step. Or train a
  **conditional** model on pairs (unit 3) when pairs exist. Any good denoiser can serve as the
  prior in an iterative restoration (one line; the classical denoisers of `denoise/` and
  `shrinkage/` included).
- **Where generated detail lives** (the key slide): split every image into the part the
  measurement sees and the part it cannot (the **null space**: `illposed/` invisible objects,
  frequencies beyond the cut-off of `resolution/`). The measured part is pinned by the data up to
  noise; the null-space part comes **entirely from the prior**. Hence:
  - **denoising**: no null space, but detail under the noise level comes from the prior;
  - **deblurring / super-resolution**: everything beyond the cut-off (e.g. structure finer than
    186 nm at NA 1.4, green light) is prior;
  - **inpainting**: the hole is prior;
  - **virtual staining** (predict a fluorescence channel from a transmitted-light image): the
    whole target channel is unmeasured; only what is **correlated** with the input in the
    training data can be predicted; anything else (a marker whose position is not visible in
    the input, an intensity change caused by a treatment) is painted as the training average.
- **Checks** (extending `diffusion/`'s rules, not repeating them):
  1. **data consistency**: re-apply the forward model; the residual should look like noise of
     the right size (the discrepancy principle of `regularise/`). Necessary, **not
     sufficient**: every null-space hallucination passes it;
  2. **disagreement** between samples locates the prior's contribution;
  3. **coverage**: on images with known truth, does the 90 % band of the samples contain the
     truth 90 % of the time? With a mismatched prior it does not (over-confident bands);
  4. a prior trained without the phenotype you look for (recall `genrestore`'s
     "prior without dividing cells") turns it into the normal case.
- Scientific uses, honestly: denoising low-light images (good for display and segmentation
  help, measure on validated readouts), super-resolution (beyond the cut-off is prior),
  virtual staining (saves a channel, cannot discover what is not in the input), and
  restoration for data reduction. Recall `diffusion/`'s rule "restoration for display is not
  restoration as data"; pointer to the integrity unit if placed.

Math layer:

- Posterior score $\nabla\log p_t(\mathbf x_t\mid\mathbf y) = \nabla\log p_t(\mathbf x_t) + \nabla\log p_t(\mathbf y\mid\mathbf x_t)$; for a Gaussian prior the second term is exactly $\mathcal N(\mathbf y; A\,\mathbb E[\mathbf x_0\mid\mathbf x_t], \sigma_n^2 I + A\,\mathrm{Cov}[\mathbf x_0\mid\mathbf x_t]A^\top)$; the common approximation keeps the mean and drops the covariance term (hence over-confident at high noise).
- Gaussian prior $\mathcal N(0, s^2)$, $y = x + n$: MAP = mean = Wiener = Tikhonov with $\lambda = \sigma_n^2/s^2$ (recall `regularise/` MAP slide).
- Null space: with $A = $ projection and a prior that factorises over range/null space, the posterior's null-space component **is** the prior's (conditionally on the range part, for a correlated prior: the prior's conditional). Data consistency $\|A\hat{\mathbf x} - \mathbf y\|$ is unchanged by adding any null-space vector.
- Replacement for inpainting: the measured coordinates of $\mathbf x_t$ given noise-free $\mathbf y$ are $\sqrt{\bar\alpha}\mathbf y + \sqrt{1-\bar\alpha}\boldsymbol\varepsilon$ exactly; the approximation is in the coupling to the unmeasured ones.
- Coverage: definition and why it holds by construction for the exact posterior under the true prior.

**Demo `posterior1d`** (new; exact posterior plus approximate samplers, 1D so the band and
the truth are curves). Signals of 64 samples: a few Gaussian spots on a background (one spot,
two spots at a random separation). Training set: 400 such signals (seeded); prior = narrow
Gaussians around them, written in the DCT basis so a mirror-boundary Gaussian blur and an
ideal low-pass (explicit cut-off) are diagonal (as `genrestore`, `lib/mixture.ts
posteriorDiag`). Measurement menu: blur (σ), **ideal low-pass** (cut-off slider: the null
space is explicit), noise only, missing samples (paint a gap). Panels: truth (toggle),
measurement; **Tikhonov** (`lib/deconv.ts tikhonov1d`, λ by the discrepancy principle);
posterior mean; 5 posterior samples overlaid with the 90 % band; a strip showing each sample's
**null-space part** ("what came from the prior"). *Sampler* menu: exact posterior / guided
reverse diffusion with the clean-estimate approximation (exact prior score, N steps,
strength) / replacement (gap mode). *Prior* menu: matched / "never two close spots" / "spots
wider than the truth". Readouts: spot count per sample, residual vs expected noise
(consistency), *coverage* button: 200 seeded repeats, fraction of positions where the truth is
inside the 90 % band (exact sampler only; ~50 ms). Presets: "two spots resolved: all agree",
"two spots below the cut-off: samples split 1 vs 2", "prior without close pairs: always one",
"consistent and wrong: the residual check passes", "approximate sampler: too narrow a band",
"coverage 90 % with the right prior, 60 % with the wrong one", "gap: the prior fills it".
Cost: exact posterior 400 × 64 per measurement (trivial); approximate sampler 16 samples ×
200 steps × 400 components × 64 ≈ 80 M operations (~100 ms, on a button).

**Demo `virtualstain`** (new; exact, two-channel 32×32 synthetic cells). Training set: 300
pairs (transmitted-light-like channel: cell outline, nucleus edge, texture; fluorescence
channel: nucleus, plus a *marker*). Prior: narrow Gaussians around the 2 048-pixel pairs; the
input channel is measured with noise, the target channel not at all ($a_i = 0$ for its pixels),
so the posterior over the target is exact (`posteriorDiag`). Panels: input, truth target
(toggle), posterior mean, four samples, disagreement map, and the top training pairs' weights.
*Marker* menu: "nucleus" (visible in the input: well predicted), "random spots" (not visible:
samples disagree, the mean is a uniform haze), "treated cells" (truth from a treatment that
doubles the marker intensity, prior trained on untreated: the prediction shows untreated
intensity and the treatment effect vanishes). Readouts per panel: nucleus area, spot count,
mean marker intensity, each against truth. Presets: "nucleus: virtual staining works", "spots:
it paints the average", "a treatment effect disappears", "noisier input: the nucleus too
becomes uncertain".

By hand (concept): (a) a two-pixel camera that measures only the mean, $y = (x_1 + x_2)/2 =
0.5$, prior over three training images (1, 0), (0, 1), (0.5, 0.5) with weights 0.4, 0.4, 0.2:
posterior weights, posterior mean (0.5, 0.5), the chance a sample shows a "resolved" pair (0.8),
the Tikhonov answer with a difference penalty (0.5, 0.5); which of these pass the data
consistency check (all); (b) NA 1.4, λ = 520 nm: cut-off period 186 nm; a restored image shows
120 nm spacing: where does it come from?; (c) a 90 % band contains the truth 120 times out of
200: what does that say about the prior?; (d) virtual staining: for four readouts (nucleus area,
marker spot count, marker intensity change under treatment, cell outline length) decide which may
be reported, and what validation would be needed; (e) given a residual map and a sample
disagreement map, mark which regions are trustworthy. *(math)*: (f) MAP = mean = Wiener =
Tikhonov for the scalar Gaussian case with $\lambda = \sigma_n^2/s^2$; (g) show that adding a
null-space vector leaves $\|A\mathbf x - \mathbf y\|$ unchanged, and that the posterior null-space
component equals the prior's for a factorising prior; (h) for a Gaussian prior, write
$p_t(\mathbf y\mid\mathbf x_t)$ exactly and identify what the common approximation drops; (i) why
replacement is exact for the measured coordinates.

### 5. (optional / further reading) `flow-matching/`: Straight paths: flow matching

- The deterministic sampler follows curved paths; choose instead the **straight line** between a
  noise sample and a data sample as the training path and learn its velocity (regression again,
  free labels again). Paths of the learned flow are straighter, so fewer steps suffice;
  re-straightening by retraining on its own pairs (one slide). Same Gaussian-noise endpoints;
  diffusion's ODE is one particular choice of path.
- Math: the linear interpolant $\mathbf x_\tau = (1-\tau)\mathbf x_0 + \tau\boldsymbol\varepsilon$, the conditional velocity $\boldsymbol\varepsilon - \mathbf x_0$, the regression to its conditional mean; relation to the v-prediction of unit 1.
- **Demo**: `samplerpaths` with `data-path="linear"` (new mode, default unchanged): the same heat
  map and trajectories for the straight-line path (exact velocity for a mixture), steps slider;
  preset "one step is almost enough for a single mode, not for two".
- By hand: velocity of the straight path for $x_0 = 2$, $\varepsilon = -1$; why the exact
  marginal velocity still bends between two modes.

## Shared engineering

- **No runtime network, no model zoo.** Every demo is analytic (exact mixture scores and
  posteriors) except `scorematch1d`'s tiny MLP (precomputed snapshots + live continuation) and
  `latentdiff`'s use of the existing shape-VAE decoder. Training/precompute scripts live next
  to the demo with a header comment (command, seed, data, architecture), like
  `diffusion2d/train_score.py` and `vaeimages/train_shapes.py`:
  - `scorematch1d/train_dsm.ts`: node script (esbuild-bundled like the tests) using
    `lib/nn.ts` (tanh, Adam, `trainBatch` exist): writes `dsm_weights.ts` (three snapshots per
    preset, float32 base64, < 50 KB). The live *train more* button uses the same code path, so
    the snapshot at 2 000 steps is reproducible bit-for-bit in node (tested).
  - `latentdiff/make_codes.py`: regenerates 800 shapes with `vaeimages/train_shapes.py`'s
    generator (imported, same seed rules), runs the int8 encoder exactly as `lib/q8.ts`
    dequantises it, writes `latent_codes.ts` (means, spreads, kind). A test checks the TS
    encoder (`shapeVae.encode`) on 10 regenerated masks against the stored codes.
- **New lib files owned by this day** (names checked: no collision in `demos/src/lib/`):
  - `lib/diffusion.ts` (no DOM): generic d-dimensional score interface
    `ScoreFn(x, abar, out)`; time grids (uniform τ, uniform log-SNR, dense near the data);
    `logSnr`, `abarFromLogSnr` (schedules themselves stay in `lib/mixture.ts`:
    `alphaBarAt`, `alphaBar`, reused); steps `ddimStep(η)`, `ddpmStep`, `eulerOdeStep`,
    `twoStageOdeStep`, `langevinStep`; `sampleRun` with optional trajectory recording;
    `invertOde`; `cfgCombine`; `posteriorGuidedStep` (clean-estimate approximation) and
    `replacementStep`; straight-path (flow-matching) velocity for a diagonal mixture. Moves
    the 2D sampler loop out of `diffusion2d/model.ts`, which then re-exports it
    (**regression: `diffusion2d` presets give identical readouts**, checked against the numbers
    in `diffusion/lec.qmd` notes) and the mixture presets `dataSet()` so `genmetrics` uses the
    same clouds.
  - `lib/genmetrics.ts` (no DOM): `frechetGaussian(mu1, S1, mu2, S2, d)` (2D closed form; d-dim via
    `linalg.ts jacobiEigen`), `knnRadii`, `precisionRecall(real, fake, k)`,
    `nearestDistances(samples, train)`, `wasserstein1d`, `intervalCoverage`. Kept separate from
    `metrics.ts` (per-image) and `eval.ts` (per-object).
- **Reused shared libraries:** `lib/mixture.ts` (`makeDiag`, `scoreDiag`, `noisedMixture`,
  `restrictMixture`, `sampleDiag`, `posteriorDiag`, `posteriorMean`, `posteriorSample`;
  owner modern DL day: no changes needed, extensions go through the owner); `lib/nn.ts`;
  `lib/q8.ts` and `vaeimages/imagevae.ts` (`shapeVae`, imported, not copied); `lib/fft.ts`,
  `lib/psf.ts radialAverage`, `lib/deconv.ts` (`tikhonov1d`, `discrepancyLambda`,
  `gaussian1d`); `lib/dct.ts` (`dct1`, `dct2`, `idct2`) for the DCT-diagonal priors;
  `lib/synth.ts` (`nuclei`, `addGaussianNoise`), `lib/cell.ts` (`loadCell`, `CELL_CREDIT` shown);
  `lib/random.ts` (`rng`, `gauss`); `lib/plot.ts` (line, band, histogram, heatmap, scatter,
  ellipse, bars; clip ids are fixed per build-issues); `lib/colour.ts` (`viridis`,
  `OKABE_ITO` for classes); `lib/ui.ts` sliders/presets/buttons; `lib/stats.ts` (`quantile`).
- **Local files** (candidates for `lib/synth.ts` later, per build-issues): `posterior1d/spots.ts`
  (1D spot signals), `virtualstain/pairs.ts` (two-channel cells; built on `genrestore`'s cell
  generator once that moves to `lib/synth.ts`, otherwise a local generator, not a copy).
- **Lessons from the first build** (`plan/build-issues.md`): prefix every per-demo CSS
  selector with `.qb-demo[data-demo="…"]` (demo.css loads late); labels and readouts ≥ 18 px on
  slides (the existing `diffusion2d` slide CSS uses 14–17 px; do not copy it); fits 1150 × 560
  (`virtualstain` and `latentdiff` are the tight ones: image grids as 2 × 2 blocks); figure
  scripts for the print fallbacks committed in the repo (`demos/test/diffusionday-figs.ts`,
  like `vae-figs.ts`), not left in a scratchpad; unit decks ≤ 18 content slides (unit 4 is the
  risk: keep the virtual-staining part to three slides + demo).
- **Performance** (TypeScript first; measure in the browser before Wasm): everything above is
  ≤ 5 ms per live update except the button-triggered runs (`latentdiff` 400 samples,
  `posterior1d` approximate sampler and coverage), which are 30–100 ms and therefore run on
  press, not live. No Wasm expected.
- **Tests** (`demos/test/diffusionday-test.ts`, added to `make test`): `lib/diffusion.ts`
  samplers with the exact score and 1 000 steps reproduce mixture mean/variance (1D and 2D);
  ODE inversion round trip; `diffusion2d` regression; every by-hand number (Euler spreads 0.316 /
  0.535, DDIM 1.1, DDPM mean 0.969 / variance 0.246, Fréchet 0.468 / 0.216 / 0, 37 % between
  modes, CFG 1.4, guided Gaussian 4/3 and 1/3, density ratios 0.61 / 0.94 / 1.45, two-pixel
  posterior weights) computed by the same code; `frechetGaussian` vs a brute-force matrix square
  root; k-NN precision/recall on hand-made sets; DSM snapshot reproducibility; `posterior1d`
  exact posterior vs importance sampling; coverage ≈ 0.9 with the matched prior; `virtualstain`
  posterior weights sum to 1 and the nucleus readout matches truth on the "nucleus" preset;
  `latentdiff` stored codes vs TS encoder.
- Site: new sidebar group "Generative models" (after "More learning") with
  the day "Diffusion models" and the five units; every demo in `demos.qmd`. Follow
  `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, static figure
  fallbacks for PDF/print.

**Pending edits outside this day (to apply when built; not done by this plan):**

- `diffusion/exercise.qmd`, "Where this leads": point to the four units by name
  (`score-matching/`, `diffusion-sampling/`, `guidance/`, `diffusion-inverse/`) instead of the
  generic "a day on diffusion models". Nothing else in `diffusion/` changes.
- `plan/coursemap.md`: Generative models row → planned; owners-table rows (see Overlaps).
- `vae-gen/exercise.qmd` "Where this leads" (diffusion bullet) could also point to `guidance/`
  (latent diffusion); optional, owner VAE day.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`scorematch1d`, `noisespectrum`) |
| 10:45–12:15 | Unit 2 lecture, by-hand steps and Euler, `samplerpaths`, `genmetrics` |
| 13:15–14:45 | Unit 3 lecture + exercise (`guidance1d`, `genmetrics` guidance mode, `latentdiff`) |
| 15:00–16:30 | Unit 4 lecture, two-pixel camera by hand, `posterior1d`, `virtualstain`; quizzes as wrap-up |

## Open questions

1. **`diffusion/` stays or moves?** Decided here: it stays on day 39 as the introduction (the
   learning part needs it for the hallucination rules); this day only recalls it. Confirm in the
   coursemap.
2. Evaluation of generated images placed in unit 2 (it is needed to say what "better" means for
   steps) and reused in unit 3 for guidance. Alternative: a fifth core unit `gen-eval/` if the
   day is lengthened.
3. Latent diffusion shares unit 3 with guidance. If unit 3 runs long, latent diffusion becomes
   optional unit 6 `latent-diffusion/` with `latentdiff` unchanged.
4. Real data: a licensed paired transmitted-light / fluorescence set would make
   `virtualstain` honest about texture; synthetic only for now (as all of day 39).
5. `posterior1d` uses the Jacobian-free clean-estimate approximation; is the version with the
   Jacobian (exact for the mixture, more code) worth showing as a third sampler?
6. Should a second day in the group (e.g. GANs/normalising flows, or generative models
   for 3D) be requested? Not planned.
7. The integrity unit (`integrity/`) is still unplaced; unit 4 points to it only if it exists.

## References

- Ho, Jain & Abbeel (2020), *Denoising diffusion probabilistic models*, NeurIPS.
- Song & Ermon (2019), *Generative modeling by estimating gradients of the data distribution*, NeurIPS.
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), *Score-based generative modeling through stochastic differential equations*, ICLR.
- Hyvärinen (2005), *Estimation of non-normalized statistical models by score matching*, JMLR; Vincent (2011), *A connection between score matching and denoising autoencoders*, Neural Computation.
- Efron (2011), *Tweedie's formula and selection bias*, JASA.
- Kingma, Salimans, Poole & Ho (2021), *Variational diffusion models*, NeurIPS (log-SNR view, ELBO).
- Song, Meng & Ermon (2021), *Denoising diffusion implicit models*, ICLR.
- Karras, Aittala, Aila & Laine (2022), *Elucidating the design space of diffusion-based generative models*, NeurIPS (step spacing, second-order steps).
- Rissanen, Heinonen & Solin (2023), *Generative modelling with inverse heat dissipation*, ICLR; Dieleman (2024), *Diffusion is spectral autoregression* (blog post; coarse-to-fine view).
- Dhariwal & Nichol (2021), *Diffusion models beat GANs on image synthesis*, NeurIPS (classifier guidance); Ho & Salimans (2022), *Classifier-free diffusion guidance*, arXiv:2207.12598.
- Rombach, Blattmann, Lorenz, Esser & Ommer (2022), *High-resolution image synthesis with latent diffusion models*, CVPR.
- Heusel, Ramsauer, Unterthiner, Nessler & Hochreiter (2017), *GANs trained by a two time-scale update rule converge to a local Nash equilibrium*, NeurIPS (Fréchet distance); Kynkäänniemi, Karras, Laine, Lehtinen & Aila (2019), *Improved precision and recall metric for assessing generative models*, NeurIPS.
- Carlini et al. (2023), *Extracting training data from diffusion models*, USENIX Security (memorisation).
- Chung, Kim, McCann, Klasky & Ye (2023), *Diffusion posterior sampling for general noisy inverse problems*, ICLR; Lugmayr et al. (2022), *RePaint: inpainting using denoising diffusion probabilistic models*, CVPR (replacement); Kadkhodaie & Simoncelli (2021), *Stochastic solutions for linear inverse problems using the prior implicit in a denoiser*, NeurIPS.
- Blau & Michaeli (2018), *The perception–distortion tradeoff*, CVPR.
- Lipman, Chen, Ben-Hamu, Nickel & Le (2023), *Flow matching for generative modeling*, ICLR; Liu, Gong & Liu (2023), *Flow straight and fast: learning to generate and transfer data with rectified flow*, ICLR.
- Christiansen et al. (2018), *In silico labeling: predicting fluorescent labels in unlabeled images*, Cell; Ounkomol et al. (2018), *Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy*, Nature Methods (virtual staining).
- Belthangady & Royer (2019), *Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction*, Nature Methods (hallucination in microscopy).

## Build notes (2026-11-13, deviations from the plan above)

Built: the four core units (`score-matching/`, `diffusion-sampling/`, `guidance/`, `diffusion-inverse/`), the eight demos, `lib/diffusion.ts`, `lib/genmetrics.ts`, the test `demos/test/diffusion-test.ts` (not `diffusionday-test.ts`), figures by `tools/figs/diffusion/figs.ts`. Not built: the optional unit `flow-matching/` and the `samplerpaths` `data-path="linear"` mode (`lib/diffusion.ts` has `straightVelocity1d` ready). `diffusion/exercise.qmd` "Where this leads" now links the four units.

- **`lib/diffusion.ts`** works in the noise-level form x = x₀ + σε with a *denoiser* D(x, σ) (score = (D − x)/σ²); the deterministic step is then exactly Euler of the probability-flow ODE, η interpolates to the ancestral step. The VP one-step formulas of `diffusion/` are kept as `ddimStepVP`/`ddpmStepVP` (by-hand numbers). `diffusion2d/model.ts` was **not** refactored to use it (other day's demo; not allowed to change); `genmetrics` imports `dataSet` from `diffusion2d/model.ts` unchanged.
- **Two-stage step** = midpoint in log σ (DPM-Solver-2 form), not Heun: Heun in σ explodes for the large first steps of few-step samplers (sd 50 for 2 steps).
- **`samplerpaths`**: "spacing fixes it" became "spacing matters": with these narrow modes the cosine schedule (uniform in time) is the good spacing and uniform-in-log-SNR or ρ = 7 ("dense near data") are worse at 5 steps. "Stochastic forgets a bad start" uses a start 5× too narrow; a persistent denoiser error is *not* forgotten (checked: ancestral sampling amplifies it), and the lecture says so. Inversion: slider (and click on the plot), not drag.
- **`genmetrics`**: in guidance mode precision stays flat (≈ 0.95–0.97, narrow blobs), only recall drops (0.97 → 0.32 at w = 4); presets and units say so. Mode dropping drops the first quarter of the components (on "rare blob" that is the big blob). The memoriser also lowers k-NN recall (50 distinct points).
- **`latentdiff`**: the decoder is evaluated with `lib/q8.ts` from `vaeimages/shape_weights.ts` (importing `vaeimages/imagevae.ts` would bundle the 440 KB MNIST weights); the test checks it equals `shapeVae.decode`. Codes: `make_codes.py` (numpy; executes the mask-generator functions of `train_shapes.py` via `ast`, runs the int8 encoder), 800 masks, seed 41. 200 samples × 40 steps (~0.3 s, debounced) instead of 400 × 100. Caricature preset at w = 4 (w = 8 leaves the plot). No "zoom on an edge" preset.
- **`posterior1d`**: spot centres limited to 28–36 and prior sd 0.05 so that 400 training signals give a posterior with several candidates (with centres 16–48 it collapsed onto one training signal). "Prior without close pairs" gives a confident wrong *separation* (2 spots, farther apart), not "always one"; "consistent and wrong" uses the wide-spot prior (30/32 samples one spot, residual 1.02). The clean-estimate sampler (Jacobian-free, ζ = 1) fits the noise (residual 0.00) and is only slightly narrower than the exact posterior. Coverage 88.4 % matched, 74.5 % wide, 82.4 % no close pairs. Local generator in `model.ts` (no separate `spots.ts`).
- **`virtualstain`**: the cell outline and texture are fixed and the training set is 75 nucleus geometries × 4 independent marker draws (cells that look identical in the input), otherwise the 2 048-dim exact posterior collapses onto one training pair. Local generator (not `genrestore`'s).
- **Local helper** `demos/src/scorematch1d/dview.ts` (`injectStyle`, `choiceRow`, `imgPanel`, `helperCss`, `fmt`) is used by all eight demos: candidate for `lib/ui.ts`.
- Slide plots use 23-unit labels (`host.closest(".reveal")`) so tick labels are ≈ 18 px after scaling.
