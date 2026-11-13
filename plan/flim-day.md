# Plan: a day on fluorescence lifetime imaging (FLIM)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.
**Course map (2026-11-12):** elective, group "Fluorescence quantification" (colocalisation, FRET,
FLIM); entry after core day 7 (colour). What it needs from the core: Part I (`fourier/`,
`cameranoise/`, `psf/`, `illposed/`, `deconv/`, `richardson-lucy/`, `denoise/`) and day 7's
`colour-display/`. The spectral unmixing elective helps (units 2 and 4 recall it), but it is not
required. Per the course map's "Ownership changes", **this day owns phasors**: the unmixing
day's optional `phasor/` unit (not built) becomes a recall of this day.

## Goal of the day

After the day a student can:

1. explain what a **fluorescence lifetime** is (the mean time a molecule stays excited, an
   exponential waiting time), why it **does not depend on concentration, intensity or
   bleed-through** to first order, and what changes it (quenching, FRET, binding, pH, viscosity);
   compute the fraction of photons that arrive after a given time;
2. describe **time-correlated single-photon counting**: one photon timed per pulse at most, a
   histogram of arrival times, the **instrument response function (IRF)** acting as a
   **convolution** (the forward model of `psf/` in time), wrap-around from the previous pulse,
   pile-up at high count rates; estimate τ by hand from the **mean arrival time** and from
   **two gates**;
3. **fit** mono- and bi-exponential decays by **reconvolution** (blur the model, not the data,
   recalling why deconvolution is ill-posed), compare **least squares** with **Poisson maximum
   likelihood** when there are few photons, and predict the precision: about $1/\sqrt N$ relative
   error with $N$ photons for one lifetime, far worse for two; tell **amplitude-weighted** from
   **intensity-weighted** mean lifetimes; explain **global analysis** (shared lifetimes, per-pixel
   fractions) as linear unmixing in time;
4. explain **frequency-domain** measurement: modulated excitation gives an emission wave that is
   **delayed (phase φ)** and **demodulated (m)**; compute the **phase and modulation lifetimes**
   and use $\tau_\phi < \tau_m$ to recognise a non-single-exponential decay;
5. compute and read **phasors**: the normalised first Fourier coefficient of the decay (the same
   two numbers as $m\cos\phi$, $m\sin\phi$); single exponentials lie on the **universal
   semicircle**; mixtures lie on the **straight line** between their components at the **photon
   fraction**; three components fill a triangle; calibrate with a reference (dividing by the
   IRF's phasor is the safe, one-frequency deconvolution); select regions on the phasor plot and
   map them back to the image;
6. measure **FRET by lifetime** ($E = 1 - \tau_{DA}/\tau_D$), separate the interacting fraction
   from the efficiency (bi-exponential fit or the phasor line), and interpret
   **autofluorescence/metabolic imaging** (free vs bound NAD(P)H) on the phasor plot;
7. *(math)* derive the phasor of an exponential and the semicircle, the linearity of phasors in
   photon fractions, "convolution = product of phasors", $\tan\phi = \omega\tau$ and
   $m = 1/\sqrt{1 + \omega^2\tau^2}$, the Poisson log-likelihood and its maximum (the mean arrival
   time for an infinite window), and the Cramér–Rao bound $\operatorname{Var}\hat\tau \ge \tau^2/N$.

Why it fits this audience: the whole day is calculus (exponentials, integrals, a derivative of the
log-likelihood) and one complex number per pixel. Two Part I ideas pay off strongly: **convolution
with an instrument response** (the PSF in time) and **the Fourier coefficient as a projection onto
a wave** (here a single frequency, the laser's repetition rate). The phasor plot is one of the most
visual results in the course: a cell becomes a cloud of points, and mixtures become straight lines.

Notation (course conventions): τ for lifetime (never λ; λ stays the regularisation weight and is not
used on this day except in the deconvolution recall); ω = 2πf for the angular modulation/repetition
frequency; $N$ photons; $(g, s)$ for phasor coordinates; IRF; photon (intensity) fraction $f_i$
vs amplitude fraction $\alpha_i$. Times in ns, frequencies in MHz.

## Prerequisite gap

What the core already teaches (checked in the built units) and what this day recalls in one line:

- **Exponentials, integrals, means of a distribution**: calculus (audience.md). The exponential
  distribution of waiting times is new; taught in unit 1 with a picture (a constant chance per
  nanosecond of emitting), no probability theory beyond the mean.
- **Poisson counts, variance = mean** (`cameranoise/`, day 3; "The math: Poisson counts"): recalled
  in unit 2 for the histogram bins.
- **Convolution, the forward model** (`kernels/`, `psf/` "The forward model in one line"): the IRF
  is the PSF of the time axis. Recall slide in unit 1.
- **Ill-posedness and why we do not divide by the transfer function** (`illposed/` "Divide by H",
  `deconv/`): recalled in unit 2 to justify reconvolution fitting, and in unit 3 to explain why
  dividing by the IRF's phasor at *one low* frequency is safe.
- **The Poisson log-likelihood** (`richardson-lucy/`, "The math: RL from the Poisson likelihood"):
  unit 2 reuses exactly that likelihood, now with a few parameters instead of a whole image.
  Maximum likelihood as an idea is taught on the GMM elective (`gaussian-mixture/`) but this day
  does not depend on it: one concept slide ("the parameters that make the observed counts most
  probable") is enough.
- **Fourier coefficient as a projection; convolution theorem; waves are eigenvectors of
  convolution** (`fourier/`, "Everything is a sum of waves" and "The math: the convolution
  theorem"): units 3 and 4 lean on these. Note the sign: `fourier/` defines
  $c_k = \int f\,e^{-2\pi i k t}dt$, so a decay's first coefficient is $g - is$; the phasor plot
  conventionally plots $s$ upwards (state once in unit 3's math slide).
- **Nonlinear least squares** (Gauss–Newton / Levenberg–Marquardt): not owned by any built unit.
  Unit 2 teaches it in words (start, linearise, step, repeat; local minima) with the details in the
  math layer. See open question 3 (ownership, since point detection and localisation will fit
  Gaussians too).
- **Linear least squares / NNLS / simplex** (`lsunmix/`, `endmembers/`, unmixing elective): used
  for global analysis (unit 2) and the three-component triangle (unit 4); one recall slide each,
  self-contained for students who did not take the unmixing day (least squares is linear algebra).
- **Colour maps and merges** (`colour-display/`): lifetime images are shown as hue = lifetime,
  brightness = intensity, with a perceptual map rather than a rainbow. One-line recall.
- **Denoising and binning** (`denoise/`, `kernels/`): binning pixels to gain photons and median
  filtering of phasor coordinates; one-line recalls.

## Overlaps / who owns what

Course map: this day **owns** fluorescence lifetime, time-domain (TCSPC) and frequency-domain
measurement, decay fitting with Poisson statistics, the **phasor approach** (lifetime phasors and
the general "phasor of a sampled curve"), and FLIM-FRET. Others recall it, and this day recalls
others:

| Day / unit | Relation |
|---|---|
| Spectral unmixing (`endmembers/`, the unbuilt optional `phasor/`) | **Ownership change applied:** this day teaches phasors (unit 3). The unmixing day's optional spectral-phasor unit becomes a one-slide recall in `endmembers/` ("the spectral phasor is the phasor of the FLIM day applied to a spectrum: mixtures on lines, three dyes in a triangle"), or, if it is ever built, a short optional unit named **`spectral-phasor/`** (the folder name `phasor/` is taken here; open question 1) that imports `lib/phasor.ts`. Here: global analysis is "unmixing in time" (unit 2), the phasor triangle is the simplex of `endmembers/` (unit 4); both are recalls with pointers. |
| FRET (same elective group; `plan/fret-day.md`, planned in parallel) | Agreed split with that plan: the **FRET day owns Förster theory** ($E(r)$, $R_0$, the $r^{-6}$ law, κ²) and intensity methods (sensitised emission with bleed-through/cross-excitation correction recalling `crosstalk/`, acceptor photobleaching, ratiometric sensors), and puts the rate picture $E = 1 - \tau_{DA}/\tau_D$ in its unit 1 math layer as the bridge; its unit 3 points here. **This day owns lifetime-based FRET** (unit 4): it recalls $E(r)$ and $R_0$ in one concept slide (a recall, not re-taught; enough for a student who takes FLIM without FRET) and reuses `lib/fret.ts` (open question 2). |
| Colocalisation (same group; takes over `coloc/`) | One bullet in unit 4: colocalisation says "in the same diffraction-limited pixel", FRET/FLIM says "within ~10 nm". The `coloc/` notes already point forward to this group. |
| Deconvolution (day 5) | Owns ill-posed inverse problems and the Poisson likelihood of RL. This day recalls both; it does not re-derive regularisation. "Fit through the forward model" (reconvolution) is the time-axis version of the same lesson. |
| Fourier (`fourier/`, day 1) | Owns the DFT, Fourier series, convolution theorem. Recalled; the phasor is $c_1$ of a periodic decay, normalised by $c_0$. |
| Camera noise (`cameranoise/`, day 3) | Owns Poisson noise and SNR. Recalled. Single-photon counting detectors (no read noise, dead time) are new here, one slide. |
| Colour display (`colour-display/`) | Owns colour maps and merges; recalled for lifetime–intensity images. |
| GMM elective (`gaussian-mixture/`, `em/`) | Owns maximum likelihood and EM in general. Pointer only: fitting a bi-exponential to photon arrival times is a two-component mixture of exponential distributions, and EM would work (one bullet in unit 2's math layer). |
| Evaluation and statistics (day 16) | Owns replicates and statistics of measurements: per-cell mean lifetimes are compared like any other measurement (one bullet, unit 4). |
| Measurements (`intensity-features/`) | Per-object mean lifetime / fraction bound becomes a column of the feature table (one bullet). |
| Point detection (`plan/pointdetection-day.md`, planned in parallel; takes over `localisation/`) | It teaches Gaussian fitting by least squares and Poisson ML with Gauss–Newton/LM in its math layer, and the Cramér–Rao bound for positions. Same ideas, different model: whichever day comes first teaches nonlinear fitting in full, the other recalls it in one slide. The generic optimiser should be one file, `lib/nlfit.ts` (owned here), with that plan's `lib/spotfit.ts` calling it (open question 3). |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus one optional unit. Folder names
checked against the built site (`_quarto.yml`): `lifetime/`, `decayfit/`, `phasor/`,
`flim-fret/`; optional `flim-more/`. Sidebar: a new elective section "Fluorescence
quantification" → "FLIM" with these four units, after the colocalisation and FRET days when they
exist.

### 1. `lifetime/`: Lifetimes, decays and time-correlated photon counting

Concept layer:

- Picture first: two cells, one twice as bright as the other. Is it more dye, or brighter dye?
  Intensity cannot tell; the **lifetime** can. A FLIM image is intensity plus "how long did the
  light take to come out" in every pixel.
- **What a lifetime is**: after excitation, a molecule leaves the excited state at a constant rate
  per nanosecond (emitting a photon, or losing the energy without light). Many molecules → the
  emission decays **exponentially**, $I(t) = A\,e^{-t/\tau}$; τ is the **mean** waiting time
  (typical 0.3–5 ns). After one τ, 37 % of the photons are still to come; after 3τ, 5 %.
  Plot on a log axis: a straight line with slope $-1/\tau$.
- **What changes τ and what does not**: concentration, excitation power, detector gain, out-of-focus
  dilution and bleed-through from another channel change intensity, **not τ** (first order). τ
  shortens when a new way to lose the energy appears: a quencher, binding, an acceptor nearby
  (FRET, unit 4), pH or viscosity for sensor dyes. Quantum yield and lifetime fall together
  (one-line math). Example lifetimes as shapes, not product names: "a green fluorescent protein
  ≈ 2.5 ns", "fluorescein in base ≈ 4 ns", "free NADH ≈ 0.4 ns".
- **Measuring nanoseconds**: a pulsed laser (pulses every 12.5 ns at 80 MHz); for each pulse a fast
  detector may record **at most one photon** and its delay after the pulse. Repeating millions of
  times builds a **histogram of arrival times** (e.g. 256 bins over 12.5 ns): the decay curve,
  with Poisson noise in every bin. In a scanning microscope, one histogram per pixel: the data are
  a 3D stack $x \times y \times t$ (same shape as the lambda stack of `lsunmix/`, with time bins
  instead of wavelength bins).
- **The IRF is a convolution**: the laser pulse has a width and the detector's timing jitters, so
  the measured histogram is the true decay **convolved** with the instrument response function
  (IRF, ~0.1–0.3 ns wide) plus a flat background (dark counts, room light). Recall `psf/`: the PSF
  blurs space, the IRF blurs time; same forward model $y = h * x + b$. Recorded from a scattering
  sample (zero lifetime).
- **Two side effects of the pulse train**: (a) **wrap-around**: when τ is not ≪ the pulse period,
  the tail of the previous pulse's decay is still arriving (the histogram is a periodic sum);
  (b) **pile-up**: the electronics time only the first photon per pulse, so at high count rates
  early photons are over-represented and τ looks shorter. Rule of thumb: detect photons in only a
  few percent of pulses, which is why FLIM is slow (and why every photon counts in unit 2).
- **Quick estimates without fitting**: (a) the **mean arrival time** minus the IRF's mean equals τ
  for a single exponential (means add under convolution); (b) **two gates** of equal width Δt:
  $\tau = \Delta t / \ln(D_1/D_2)$. Both are fast and simple; both break with background, a short
  window or two lifetimes (unit 2 does better).
- **Time-gated cameras** in one bullet (a wide-field image per delay gate, the two-gate formula per
  pixel), as the camera-based alternative; frequency-domain in unit 3.
- Display: lifetime as hue, intensity as brightness (recall `colour-display/`), with a perceptual
  map; mask out pixels with too few photons.

Math layer (`{.math}` slides):

- *The exponential law*: rate equation $dn/dt = -(k_r + k_{nr})\,n$ ⇒ $n(t) = n_0 e^{-t/\tau}$,
  $\tau = 1/(k_r + k_{nr})$; quantum yield $Q = k_r\tau$; adding a quenching rate $k_q$ shortens τ
  and lowers Q by the same factor (the Stern–Volmer picture $\tau_0/\tau = 1 + k_q\tau_0[Q]$, one
  line). Arrival-time density $p(t) = \tau^{-1}e^{-t/\tau}$, mean τ, variance τ².
- *The measured histogram*: expected counts in bin $j$:
  $\mu_j = N\int_{t_j}^{t_{j+1}}\big(h * p_T\big)(t)\,dt + b$, with the periodic decay
  $p_T(t) = \tau^{-1}e^{-t/\tau}/(1 - e^{-T/\tau})$ on $[0, T)$ (geometric series over previous
  pulses); counts $y_j \sim \text{Poisson}(\mu_j)$.
- *Means add under convolution*: $\langle t\rangle_{h*p} = \langle t\rangle_h + \langle t\rangle_p$
  (first moment of a convolution; one line from $\int\!\!\int (u+v)h(u)p(v)\,du\,dv$).
- *Two-gate estimator*: $D_1/D_2 = e^{\Delta t/\tau}$ for adjacent equal gates on a pure
  exponential.

**Demo `tcspc`** (new): top, a timeline of laser pulses with individual photons dropping into a
histogram as they are detected (animated, speed slider; "fast-forward to N photons"); bottom, the
histogram on linear or **log** axis (toggle), with the true decay, the IRF and the IRF-convolved
model overlaid. Sliders: τ (0.2–8 ns), IRF width, IRF delay, background, repetition period
(12.5–100 ns), **count rate** (percent of pulses with a detected photon; pile-up simulated by
keeping only the first photon per pulse), photons $N$ (10–10⁶, log). Readouts: mean-arrival
estimate (raw and IRF-corrected), two-gate estimate (gates drawn as draggable shaded bands),
fraction wrapped from previous pulses. Presets: "one photon at a time", "the log plot is a
straight line", "the IRF blurs the start", "τ = 8 ns at 80 MHz: the tail wraps around",
"too bright: pile-up shortens τ", "background flattens the tail", "100 photons: what can we say?".

**Demo `flimcell`** (new, used by all four units through `data-mode` and `data-scene`): a seeded
synthetic cell (`cellScene` from `lib/unmix.ts`, so unmixing students recognise it) rendered as an
$x \times y \times t$ photon-count stack (128×128 × 64 or 128 time bins, Poisson, IRF,
background). Panels: intensity image, lifetime image (hue = τ, brightness = intensity), and a
side panel that depends on the mode. Mode `estimate` (unit 1): click a pixel or drag a box to see
its histogram (log), estimator choice *mean arrival* / *two gates*, a binning slider (1×1 … 9×9),
photons-per-pixel slider. Scenes: `concentration` (one dye, τ constant, concentration varies 10×:
the intensity image has structure, the lifetime image is flat), `twodyes` (same brightness, two
lifetimes: the intensity image is flat, the lifetime image is not), `quencher` (a compartment
where τ is shortened). Presets: "intensity lies, lifetime does not", "same brightness, different
lifetime", "too few photons per pixel: bin", "background pixels: mask them".

By hand (concept): (a) τ = 2 ns, 1000 photons: how many arrive after 2 ns, after 4 ns, after
6 ns? (368, 135, 50); (b) two gates 0–2 ns and 2–4 ns hold 632 and 233 photons: estimate τ
(2/ln 2.71 ≈ 2.0 ns); (c) the measured mean arrival time is 3.1 ns and the IRF's mean is 0.6 ns:
τ ≈ 2.5 ns, and why the estimate fails with a flat background; (d) 80 MHz, τ = 4 ns: which
fraction of a decay is still to come when the next pulse arrives? ($e^{-3.125} \approx 4.4\,\%$)
and at τ = 1 ns?; (e) predict: a cell is twice as bright but has the same τ; another has the same
brightness but τ dropped from 2.5 to 1.5 ns. Which one points to a molecular change?
*(math)*: (f) derive the periodic decay $p_T$ by summing over previous pulses; (g) prove that
means add under convolution; (h) derive the two-gate formula and show that a background $b$ per
gate biases it upwards.

### 2. `decayfit/`: Fitting decays with few photons

Concept layer:

- The question: from one pixel's histogram (often 100–1000 photons), find τ, or τ₁, τ₂ and their
  fractions, and say how sure we are.
- **Fit through the forward model (reconvolution)**: do not deconvolve the IRF out of the
  histogram (recall `illposed/`: dividing by the transfer function amplifies noise). Instead
  convolve the *model* with the measured IRF and compare with the data. The parameters: amplitude,
  τ (or two of each), background, a small IRF shift. The tail-fit shortcut (fit only after the
  peak) as one bullet and why it wastes photons.
- **Nonlinear fitting in words**: τ enters non-linearly, so there is no closed form. Start from a
  guess (the mean arrival time of unit 1), linearise around it, take the least-squares step,
  repeat; damp the step when it overshoots (Levenberg–Marquardt as one name). Picture: the cost
  as a landscape over (τ, A) with a single bowl for one lifetime and a long curved valley for two.
- **Which cost? Least squares vs Poisson likelihood**: least squares treats a bin with 2 counts as
  just as reliable as one with 2000; weighting by $1/y_j$ (the classic χ²) breaks for bins with 0
  or 1 counts (they get infinite or huge weight), and pulls the fit towards low tails. The **Poisson
  likelihood** (recall `richardson-lucy/`: the same formula, now with three parameters) uses the
  counts as what they are. With thousands of photons per pixel all three agree; with hundreds they
  do not. The demo measures the bias and spread rather than us claiming them.
- **How many photons?** For one lifetime, the best possible relative error is about $1/\sqrt N$:
  100 photons → 10 %, 10 000 → 1 %. Real instruments do a bit worse (IRF, window, background).
  For **two** lifetimes the cost valley is long and flat: fits need tens of thousands of photons,
  and two exponentials with lifetimes within a factor ~2 can be traded for each other almost
  without changing the curve (sums of exponentials are badly conditioned; picture: two very
  different parameter sets whose curves overlap within the noise). **Binning** pixels trades
  resolution for photons (recall `denoise/`): the main practical lever.
- **Reporting two lifetimes in one number**: amplitude-weighted mean
  $\bar\tau_a = \sum\alpha_i\tau_i$ (tracks quenching of the whole population) vs
  intensity-weighted mean $\bar\tau_f = \sum f_i\tau_i$ (what a one-exponential fit and the mean
  arrival time roughly report); they differ, so say which.
- **Global analysis**: in a cell the same few species appear in every pixel, only in different
  proportions. Fit the lifetimes **once** for the whole image (or a region), then each pixel's
  amplitudes are a **linear** problem: the decay shapes (convolved with the IRF) are the columns of
  a matrix, exactly **unmixing with time bins as channels** (recall `lsunmix/`; NNLS from
  `endmembers/` keeps fractions ≥ 0). Few photons per pixel suffice, because the hard non-linear
  part used the photons of all pixels.
- Checking a fit: residuals weighted by $\sqrt{\mu_j}$ should look like noise without structure; a
  wavy residual means a missing component (the residual map of `lsunmix/` in time).
- Pointer: learned estimators (a network trained on simulated decays) exist and are fast; they
  inherit the photon limit and the biases of their training data (one line, link to day 15).

Math layer:

- *Poisson log-likelihood* $\ell(\theta) = \sum_j y_j\log\mu_j(\theta) - \mu_j(\theta)$ (constant
  terms dropped); the **deviance** $2\sum_j[\,y_j\log(y_j/\mu_j) - (y_j - \mu_j)]$ as the
  "Poisson χ²", minimised like a least-squares cost (0·log 0 = 0, so empty bins are harmless).
  Compare with Neyman's $\sum (y_j - \mu_j)^2/y_j$ and Pearson's $\sum (y_j - \mu_j)^2/\mu_j$.
- *Mono-exponential, no IRF, infinite window*: the ML estimate is the **mean arrival time**
  (derivative of $\sum_i[-\log\tau - t_i/\tau]$); its variance is $\tau^2/N$; Fisher information
  $N/\tau^2$ and the Cramér–Rao bound; the "F-value" $F = (\sigma_\tau/\tau)\sqrt N \ge 1$ as the
  figure of merit of an instrument.
- *Gauss–Newton / Levenberg–Marquardt*: linearise $\mu(\theta + \delta) \approx \mu(\theta) +
  J\delta$, solve $(J^\top WJ + \mu_\text{LM}\,\mathrm{diag}(J^\top WJ))\,\delta = J^\top W(y -
  \mu)$; for Poisson ML the same step with $W = \mathrm{diag}(1/\mu_j)$ re-evaluated each
  iteration (iteratively reweighted least squares). Parameter covariance $\approx (J^\top WJ)^{-1}$
  at the optimum; its condition number for two near lifetimes (recall the condition number of
  `lsunmix/`).
- *Global analysis* as variable projection: for fixed lifetimes the amplitudes are the linear
  least-squares (or NNLS) solution; only the few lifetimes are fitted non-linearly.
- *Mixture view* (pointer): a bi-exponential arrival-time density is a two-component mixture;
  EM from the GMM elective would fit it photon by photon.

**Demo `decayfit`** (new, main demo of the unit): left, one simulated histogram (log axis) with
the fitted model and the IRF, and below it the weighted residuals; right, the cost landscape as a
heat map over (τ, A) for one lifetime, or (τ₁, τ₂) for two (amplitudes profiled out), with the
optimiser's path drawn step by step. Controls: true τ₁, τ₂, fraction, background, photons $N$ (log
slider, 30–10⁶), model (mono / bi / bi with fixed τ₁), cost (*least squares* / *χ² with 1/y
weights* / *Poisson likelihood*), *reconvolution* on/off (off = pretend the IRF is a spike), buttons
*new noise* (seeded sequence), *step*, *fit*. Readouts: estimates ± the covariance-based error,
reduced χ²/deviance, $\bar\tau_a$ and $\bar\tau_f$. Presets: "10 000 photons: all costs agree",
"200 photons: χ² with 1/y goes wrong", "ignore the IRF: τ too long", "two lifetimes, 1 000 photons:
the long valley", "two lifetimes, 100 000 photons: now resolvable", "fit one exponential to two:
the residual is wavy", "background left out".

**Demo `photonbudget`** (new, small): a Monte Carlo of the estimators. For the current settings it
simulates 500 seeded histograms and shows, per method (mean arrival time, two gates, LS, χ² with
1/y, Poisson ML), the histogram of estimated τ with the true value marked, plus a plot of relative
error and bias against $N$ (log–log, with the $1/\sqrt N$ line). Toggle mono / bi (for bi: the
spread of τ₁ and τ₂ and their correlation as a scatter). Presets: "the 1/√N line", "few photons:
who is biased?", "background hurts the mean arrival time", "two lifetimes need ~100× more photons"
(the builder replaces "100×" by the measured factor). Precomputed per preset if a live run exceeds
~30 ms (it is 500 small fits).

By hand (concept): (a) a bin expects 4 counts: probability of 0 counts ($e^{-4} \approx 1.8\,\%$);
in a 256-bin tail with means of 0.5–4 counts, how many empty bins to expect? Why does weighting by
$1/y_j$ fail there?; (b) relative precision of τ with 100, 400 and 10 000 photons (ideal
instrument), and the photons needed for ±5 %; (c) a bi-exponential with $\alpha = (0.5, 0.5)$,
τ = (0.5, 3) ns: $\bar\tau_a = 1.75$ ns, photon fractions $f = (0.14, 0.86)$,
$\bar\tau_f \approx 2.64$ ns; which one would a one-exponential fit roughly report?; (d) global
analysis: lifetimes known (1 and 3 ns), a pixel's counts in three gates given: set up the 3×2
linear system and solve it by least squares; (e) a residual plot with a wave: what is missing?
*(math)*: (f) derive the ML estimate for one lifetime from photon arrival times and show it is the
mean; (g) compute the Fisher information and the Cramér–Rao bound; (h) write the Poisson deviance
and show that its weights in a Gauss–Newton step are $1/\mu_j$; (i) show that for fixed τ's the
amplitudes solve a linear least-squares problem.

### 3. `phasor/`: Frequency domain and phasors

Concept layer:

- **Frequency-domain FLIM** first (it is where the two numbers come from): excite with light whose
  intensity is a wave at frequency f (tens of MHz). The emission is a wave of the same frequency,
  **delayed** (phase φ) and **flattened** (modulation m < 1), because the decay smears it.
  Measured by a detector or camera whose gain is modulated, at a few phase steps: a few samples
  of one wave (recall `fourier/`: projecting onto a wave).
- **Why the output is still a wave**: emission = excitation convolved with the decay, and waves
  are the eigenvectors of convolution (recall the `fourier/` math slide): the decay can only
  multiply the wave by one complex number. That number is its **Fourier coefficient at f**:
  its angle is φ, its length is m.
- **Phase and modulation lifetimes**: for one exponential, $\tan\phi = \omega\tau$ and
  $m = 1/\sqrt{1 + \omega^2\tau^2}$: two independent ways to read τ ($\tau_\phi$, $\tau_m$). They
  agree for a single exponential; for a mixture $\tau_\phi < \tau_m$: a free test for "more than
  one lifetime". Best sensitivity when $\omega\tau \approx 1$ (80 MHz for τ ≈ 2 ns).
- **Same numbers from a TCSPC histogram**: the histogram covers one pulse period T; its first
  Fourier coefficient at $\omega = 2\pi/T$, divided by the total count, is the same complex number.
  So time and frequency domain meet in one plot. Wrap-around (unit 1) does not hurt: the
  periodic sum is exactly what a Fourier coefficient at the repetition frequency expects.
- **The phasor plot**: per pixel, $g = m\cos\phi$, $s = m\sin\phi$, i.e. the normalised
  cosine and sine projections of the decay. No fitting, no model: one multiply-add per time bin.
  Every single exponential lands on the **universal semicircle** (centre (½, 0), radius ½): long
  lifetimes near the origin, short near (1, 0). The lifetime can be read off the angle.
- **Mixtures lie on lines**: the phasor is linear in the decay and normalised by intensity, so a
  pixel containing two species lies on the **chord** between their two semicircle points, at a
  position given by the **photon fraction** (lever rule). Points **inside** the semicircle are
  mixtures; the line through a cloud, extended to the semicircle, gives both lifetimes without
  any fit (when the two species are the same everywhere).
- **Reciprocity**: select a region on the phasor plot (a cursor), and the pixels that fall in it
  light up in the image; paint a region in the image, and its pixels light up on the plot. The
  phasor histogram of a whole image (2D, log scale) is the typical view.
- **Calibration and the IRF**: convolution with the IRF multiplies the phasor by the IRF's phasor
  (a rotation plus a shrink). Measure a reference of known lifetime (or the IRF itself), and
  divide it out: one complex division for the whole image. This *is* deconvolution, but at a
  single low frequency where the IRF's response is close to 1, so nothing explodes (contrast with
  `illposed/`, where high frequencies have response ≈ 0). Same for coarse time bins (a known
  shrink and shift).
- **Noise**: with $N$ photons the phasor cloud of a pure species has a spread of roughly
  $1/\sqrt N$; dim pixels scatter, flat background pulls points towards the origin. Smoothing the
  $g$ and $s$ images (median, recall `denoise/`) tightens the cloud without blurring the
  intensity image. Higher harmonics (2ω) separate species whose first-harmonic phasors coincide
  (optional unit).

Math layer:

- *Phasor of an exponential*: $P = \dfrac{\int_0^\infty e^{-t/\tau}e^{i\omega t}dt}{\int_0^\infty
  e^{-t/\tau}dt} = \dfrac{1}{1 - i\omega\tau} = \dfrac{1 + i\omega\tau}{1 + \omega^2\tau^2}$, so
  $g = 1/(1 + \omega^2\tau^2)$, $s = \omega\tau/(1 + \omega^2\tau^2)$, hence
  $\tan\phi = \omega\tau$, $m = 1/\sqrt{1 + \omega^2\tau^2}$, and $|P - \tfrac12| = \tfrac12$.
  Sign note: `fourier/`'s $c_1$ uses $e^{-i\omega t}$, so $P = \bar c_1/c_0$.
- *Linearity*: for $I = \sum_k I_k$, $P = \sum_k f_k P_k$ with photon fractions
  $f_k = \int I_k / \int I$ (a convex combination: segment, triangle). Photon fraction vs amplitude
  fraction: $f_k = \alpha_k\tau_k/\sum\alpha_j\tau_j$.
- *Convolution*: $P_{h*I} = P_h\,P_I$ (convolution theorem at one frequency), hence calibration
  $P_\text{true} = P_\text{meas}\cdot P_\text{ref,true}/P_\text{ref,meas}$.
- *Discrete version*: $g = \sum_j y_j\cos(\omega t_j)/\sum_j y_j$ (likewise $s$), the DFT bin 1 of
  the histogram; binning with width Δt multiplies $P$ by $\mathrm{sinc}(\omega\Delta t/2)$ and a
  half-bin phase (removed by calibration).
- *τ_φ ≤ τ_m* for any mixture (Jensen-type argument for two components; or shown numerically in
  the exercise).
- *Noise*: the variance of $g$, $s$ for $N$ photons from a pure exponential, $\approx
  (\tfrac12 - \text{second-harmonic terms})/N$ (state; the demo shows the $1/\sqrt N$ spread).

**Demo `modulation`** (new, small): excitation wave and emission wave over two periods, the decay
inset; sliders τ₁, τ₂, fraction, frequency f (10–200 MHz, log). Readouts φ, m, $\tau_\phi$,
$\tau_m$ and a mini phasor plot with the current point. Presets: "one lifetime: τ_φ = τ_m",
"two lifetimes: τ_φ < τ_m", "frequency too low: everything looks the same", "frequency too high:
the wave is flattened away", "ωτ ≈ 1: best".

**Demo `phasorplot`** (new, main concept demo): left, the phasor plane with the universal semicircle
(labelled lifetime ticks for the current frequency), points for up to three species with draggable
lifetimes along the semicircle (or free points for non-exponential species), the chord or triangle
between them, and a draggable mixture point that shows its photon fractions (lever rule); right,
the corresponding decay on a log axis (sum and components) and, on demand, a cloud of simulated
pixels with $N$ photons each. Controls: frequency, photons $N$, IRF width with *calibrate* toggle
(uncalibrated: the whole semicircle rotates and shrinks), number of time bins, harmonic 1/2.
Presets: "single lifetimes live on the semicircle", "two species: a straight line", "photon
fraction is not amplitude fraction", "three species: a triangle", "uncalibrated IRF: everything
rotates", "100 photons: a cloud", "coarse time bins shrink the phasor".

**Demo `flimcell`**, mode `phasor`: the cell's phasor histogram next to the intensity and lifetime
images; a circular cursor (drag, resize) on the phasor plot highlights its pixels in the image;
a brush on the image highlights its pixels on the plot; the *median-filter g, s* toggle; a line
tool to fit a line through the cloud and read the two lifetimes where it meets the semicircle.
Scenes: `twodyes`, `mixture` (two species in varying proportions: a cloud stretched along a
chord), `background`. Presets: "a cloud on a line", "cursor ↔ image", "median-filtered phasors",
"background pulls to the origin".

By hand (concept): (a) at 80 MHz ($\omega = 0.503\ \text{rad/ns}$), τ = 2 ns: φ ≈ 45°,
m ≈ 0.71, $(g, s) \approx (0.50, 0.50)$; (b) a pixel at $(g, s) = (0.8, 0.4)$: is it on the
semicircle? Its lifetime ($s/g = \omega\tau$ → ≈ 1.0 ns); (c) a 50/50 photon mixture of 0.4 ns
and 2.5 ns at 80 MHz: phasors (0.96, 0.19) and (0.39, 0.49), midpoint (0.67, 0.34), inside the
semicircle; $\tau_\phi \approx 1.0$ ns and $\tau_m \approx 1.7$ ns; (d) the phasor of a 4-bin
histogram (8, 4, 2, 1) over one period: (0.40, 0.20); is it on the semicircle? Why not (coarse
bins)?; (e) calibration: a 4.0 ns reference should give φ = 63.6°, m = 0.445 but gives φ = 70°,
m = 0.40; a sample measures φ = 50°, m = 0.60: correct it (43.6°, 0.668) and compute
$\tau_\phi \approx 1.9$ ns, $\tau_m \approx 2.2$ ns: one lifetime or more?
*(math)*: (f) derive the phasor of an exponential and show it lies on the circle
$|P - \tfrac12| = \tfrac12$; (g) show that the phasor of a sum is the photon-fraction-weighted sum;
(h) show that the IRF multiplies phasors and derive the calibration formula; (i) show that two
different mono-exponentials can never give the same phasor, but two different bi-exponentials can
(first harmonic only).

### 4. `flim-fret/`: FLIM-FRET and metabolic imaging

Concept layer:

- **FRET in one slide** (a recall of the FRET day, not re-taught; enough to follow if it was not taken): an acceptor
  within a few nanometres takes energy from the excited donor without light; the efficiency falls
  steeply with distance (half at the Förster distance $R_0 \approx 5$ nm). Another way to leave
  the excited state → the donor's lifetime **shortens**.
- **FRET efficiency from lifetimes**: $E = 1 - \tau_{DA}/\tau_D$, with $\tau_D$ from a donor-only
  control. Why lifetime beats intensity here: no need to know concentrations, and acceptor
  bleed-through and cross-excitation (the corrections of intensity FRET, recalling `crosstalk/`)
  do not enter when only the donor's decay is recorded.
- **Efficiency vs interacting fraction**: a pixel usually holds donors that interact and donors
  that do not. A bi-exponential fit with $\tau_D$ fixed gives the efficiency of the interacting
  state and its fraction (global analysis of unit 2 makes this work at FLIM photon counts). On the
  phasor plot, pixels lie on the **line** from the donor-only point towards the FRET-state point;
  position along it = **photon** fraction (which underweights the quenched, dimmer state).
  The **FRET trajectory** (the curve that the phasor traces as E goes from 0 to 1) and where a
  real FRET state sits on it, with background and unquenched donors pulling it back.
- **Pitfalls**: the donor lifetime depends on the environment too (always a donor-only control in
  the same cells); autofluorescence adds a third species; acceptor emission leaking into the donor
  channel adds a delayed (rising) component that falls **outside** the semicircle; few photons
  (binning, global analysis).
- Colocalisation vs FRET in one bullet: colocalisation means "within one diffraction-limited
  pixel" (~200 nm, `coloc/`), FRET "within ~10 nm".
- **Autofluorescence and metabolic imaging (brief)**: NADH and NADPH (indistinguishable optically)
  fluoresce without labels; **free** ≈ 0.4 ns, **protein-bound** ≈ 2–3 ns. On the phasor plot a
  cell's cloud lies on the line between free and bound; its position = **fraction bound**, an
  indicator of metabolic state (more free in glycolytic cells, more bound with oxidative
  phosphorylation, as a tendency, not a rule). FAD in one bullet (the opposite behaviour; redox
  ratio). Typically excited with two-photon infrared light (one bullet, no instrument names).
- **Three components**: a pixel with free NADH, bound NADH and a label (or a second bound form)
  lies inside the **triangle** of the three phasors; the fractions are its barycentric
  coordinates (recall the simplex of `endmembers/`: the same picture in a fixed 2D projection);
  four or more components need more harmonics or spectral channels (optional unit).
- From pixels to a result: segment cells (thresholding/watershed of the intensity image), report
  per cell the mean lifetime or fraction bound with the photon count, compare cells and
  experiments with the statistics of day 16 (pointer), and show the phasor histogram per
  condition.

Math layer:

- *FRET rates* (recall of the FRET day's unit 1 math slide, stated not derived again):
  $k_T = \tau_D^{-1}(R_0/r)^6$, $\tau_{DA} = (\tau_D^{-1} + k_T)^{-1}$ ⇒ $E = 1 - \tau_{DA}/\tau_D$;
  distance $r = R_0(1/E - 1)^{1/6}$ with the κ² caveat in one line.
- *Mixed population*: amplitudes $\alpha$ (interacting) and $1 - \alpha$; photon fraction of the
  FRET state $f = \alpha\tau_{DA}/(\alpha\tau_{DA} + (1 - \alpha)\tau_D)$; the phasor point
  $P = (1 - f)P_D + fP_{DA}$.
- *FRET trajectory*: $P(E) = P_{\tau_D(1 - E)}$ combined with background and unquenched-donor
  fractions (state).
- *Three-component fractions*: solve the 3×3 system $[P_1\ P_2\ P_3;\ 1\ 1\ 1]\,\mathbf f =
  [P;\ 1]$ (barycentric coordinates).
- *Molecule vs photon fractions for NADH*: $\alpha_k \propto f_k/\tau_k$ if both forms have the
  same radiative rate (stated as an assumption).

**Demo `fretphasor`** (new): left, the phasor plane with donor-only point, the FRET trajectory
(E from 0 to 1, ticks every 0.1), the line from donor-only to the current FRET state, and a
simulated cloud of pixels; right, $E(r)$ with $R_0$ and the current distance (draggable), and the
donor decay (log) with its two components. Sliders: $\tau_D$, efficiency E (or distance r),
interacting fraction α, photons per pixel, background, autofluorescence fraction (third species),
acceptor bleed-through (a rising component). Readouts: $\tau_{DA}$, photon fraction f, apparent
one-exponential lifetime, estimates of E and α by (a) a bi-exponential fit with $\tau_D$ fixed and
(b) the phasor line. Presets: "no FRET: on the donor point", "all donors interact, E = 0.5",
"30 % interact: on the line, closer to the donor than you think", "E vs fraction: same mean
lifetime, different story", "autofluorescence: a third corner", "acceptor leak: outside the
semicircle", "too few photons".

**Demo `flimcell`**, scenes `fret` and `nadh`: `fret` is a biosensor cell with FRET efficiency
higher in one compartment (lifetime image and phasor cloud along the donor line; per-region
readout of E and fraction); `nadh` has cytoplasm and mitochondria-like regions with different
bound fractions plus a few bright label spots (a third species); a three-cursor triangle tool
reads fractions per pixel (maps of fraction bound). Presets: "FRET compartment", "fraction-bound
map", "a third species", "per-cell values" (cells from a threshold of the intensity image).

By hand (concept): (a) $\tau_D = 2.5$ ns, $\tau_{DA} = 1.5$ ns: E = 0.4 and, with $R_0 = 5$ nm,
r ≈ 5.35 nm; (b) 30 % of donors interact with E = 0.6 (τ = 1.0 ns): amplitude fraction 0.3, photon
fraction ≈ 0.15; which is the position on the phasor line?; (c) a one-exponential fit to (b)
gives ≈ 2.3 ns: what "efficiency" would you wrongly report?; (d) NADH at 80 MHz: free (0.96, 0.19),
bound (0.39, 0.49), a cell at (0.73, 0.31): photon fraction bound ≈ 0.40 (lever rule); molecule
fraction bound ≈ 0.10 if both forms are equally bright per unit lifetime; (e) a pixel inside a
given triangle of three phasors: estimate its fractions by eye, then check the barycentric
coordinates. *(math)*: (f) derive $E = 1 - \tau_{DA}/\tau_D$ from the rates; (g) derive the photon
fraction from the amplitude fraction; (h) solve the 3×3 barycentric system for (e).

### 5. (optional / further reading) `flim-more/`: Beyond the first harmonic

- **Higher harmonics and multiple frequencies**: two species with the same first-harmonic phasor
  separate at 2ω; with $H$ harmonics up to $2H + 1$ components can be resolved in principle
  (counting equations); noise grows with the harmonic.
- **Rising components and excited-state reactions**: a negative amplitude (acceptor emission, a
  solvent relaxation) puts the phasor outside the semicircle; how to recognise it.
- **Spectral + lifetime phasors combined**: a spectral phasor per pixel (recall: the same
  transform on a spectrum, the unmixing day) next to the lifetime phasor; more components
  separable when both are used.
- **Detector effects in more detail**: dead time and pile-up correction, the count-rate limit,
  wide-field time-gated and modulated cameras vs scanning single-photon counting, as concepts
  (no instrument or product names).
- **Learned lifetime estimation**: networks trained on simulated decays (and their failure
  outside the training range), pointer to days 15 and 18.
- **Demo**: `phasorplot` with `data-harmonics="2"` (two phasor planes side by side) and
  `fretphasor` with the rising-component preset; no new demo.

## Shared engineering

Reuse (check `demos/src/lib/README.md`; do not copy):

- `lib/random.ts`: `rng`, `exponential` (arrival times), `poisson` (bin counts and photons per
  pixel), `gauss` (IRF jitter), `choice` (species per photon).
- `lib/plot.ts`: decays on `ylog` axes (`line`, `step` mid for histograms), residual plots, the
  cost landscape with `heatmap` + `colorbar`, the phasor plane with `equal: true`, `fn` for the
  semicircle, `scatter` for points/clouds, `arrow`/`text` for lever-rule labels, `band` for gates.
  The clip-path id fix is in, so several plots per page are fine (no local `uniqueClip`).
- `lib/histogram.ts`: `histogram2d` for the phasor histogram (log display), `histogram` for the
  Monte-Carlo estimate distributions.
- `lib/fft.ts`: `dft1d` to show "the phasor is DFT bin 1" in tests and in the `phasorplot` bins
  option; `fft1d` for periodic IRF convolution when the bin count is a power of two (otherwise a
  direct circular convolution of ≤ 256 bins, which is cheap).
- `lib/linalg.ts`: `cholesky`/`choleskySolve` or `solve` for the LM normal equations (p ≤ 6),
  `inverse` for parameter covariances, `cond` for the bi-exponential conditioning readout, `lstsq`
  for global amplitudes.
- `lib/unmix.ts`: `cellScene(n, seed)` for the cell geometry (same cell as the unmixing day),
  `nnls` for non-negative global amplitudes, `triangleArea`/barycentric helpers where they fit.
- `lib/colour.ts`: `colormapLUT`/`viridis` (or a cyclic-free perceptual map) for τ, `hsvToRgb`
  for the lifetime–intensity merge, `categoryColor`/`OKABE_ITO` for species on phasor plots.
- `lib/imageops.ts`: `Grey`, `drawGrey`, `boxFilter` (binning), `toRGBA`; `lib/morph.ts`
  `median` for median-filtered g/s maps; `lib/threshold.ts` (Otsu) for the per-cell mask in
  `flimcell`; `lib/label.ts` for per-cell readouts; `lib/stats.ts` (`mean`, `sd`, `quantile`) for
  the Monte-Carlo summaries.
- `lib/ui.ts`: `slider`, `button`, `checkbox`, `presets`, `autoMount`, `draggable`, `svgPoint`.
- Planned in parallel by sibling days (use them if built first, otherwise add the functions to
  their files through the owner): `lib/fret.ts` (FRET day: `efficiency`, `distanceFromE`,
  `lifetimeWithAcceptor`); `lib/spotfit.ts` (point-detection day) should call `lib/nlfit.ts`
  rather than carry its own LM. The colocalisation day's `lib/twochannel.ts` is not needed here
  (FLIM records only the donor channel).

New library files owned by this day (names checked against `demos/src/lib/`: none exist):

- **`lib/decay.ts`** (no DOM): multi-exponential models on a periodic window (closed-form periodic
  sum), IRF models (Gaussian with delay, or a measured array) and periodic convolution of model
  and IRF, expected bin counts $\mu_j$ (bin-integrated, plus background), TCSPC simulation
  photon by photon (seeded; IRF jitter, background, wrap-around, first-photon-only pile-up) and a
  fast binned version for images (Poisson per pixel total, then a multinomial over bins),
  estimators (mean arrival with IRF correction, two-gate), frequency-domain helpers (φ, m,
  $\tau_\phi$, $\tau_m$ for any decay), amplitude/photon fraction conversions, $\bar\tau_a$,
  $\bar\tau_f$, the FRET trajectory on the phasor plot (built on `lib/fret.ts`'s `efficiency`,
  `distanceFromE` and `lifetimeWithAcceptor`, owned by the FRET day; no copies here), and
  `flimScene(kind, opts)` producing $x \times y \times t$ stacks (time-major Float32Array, the
  layout convention of `lib/unmix.ts`) for the scenes `concentration`, `twodyes`, `quencher`,
  `mixture`, `background`, `fret`, `nadh`, with ground-truth maps.
- **`lib/nlfit.ts`** (no DOM, generic, written for later reuse by point detection / localisation):
  Levenberg–Marquardt for small parameter vectors with user model + analytic or numeric Jacobian,
  costs `ls`, `weighted` (fixed weights), `neyman` (1/y), `poisson` (deviance, IRLS weights
  1/μ), parameter transforms for positivity (log τ) and bounds, fixed parameters, iteration trace
  (for drawing the path), covariance estimate at the optimum, `costGrid(model, data, p1range,
  p2range, profile)` for landscapes with linear parameters profiled out, and variable projection
  for global analysis (non-linear lifetimes shared, linear amplitudes per pixel via `lstsq`/NNLS).
- **`lib/phasor.ts`** (no DOM): `phasor(curve, harmonic, period)` for any sampled periodic curve
  (time bins or wavelength bins, so the unmixing day's spectral phasor imports it),
  `phasorStack(stack, nBins, P, harmonic)` → g, s maps, `semicircle(ω, τ)` and
  `lifetimeFromPhasor` (τ_φ, τ_m), `calibrate(gs, refPhasorMeasured, τref, ω)` (complex division),
  `binningCorrection(nBins)`, `fractionOnSegment` (lever rule by projection), `barycentric3`,
  `lineThroughCloud` (principal axis of a cloud and its intersections with the semicircle),
  `phasorHistogram` (wraps `histogram2d`), `cursorMask(g, s, centre, radius)`.

Demos and speed:

- Demos: `tcspc`, `flimcell` (modes `estimate`, `fit`, `phasor`; scenes as above), `decayfit`,
  `photonbudget`, `modulation`, `phasorplot`, `fretphasor`. None of these names exist in
  `demos/src/`.
- `flimcell` stack: 128×128 × 128 bins = 2.1 M values; simulation with the binned version is a
  one-off per scene/photon setting (target < 200 ms with a "computing…" state). Phasor maps are
  2.1 M multiply-adds (a few ms): **live**. Per-pixel LM fits of 16 k pixels are not live in
  TypeScript (estimate several hundred ms): run on a *fit image* button with progress, or on the
  binned image; only if a live binning slider is wanted and the measured fit exceeds ~30 ms, move
  the per-pixel fit loop to Rust/Wasm (`demos/wasm/flimfit/`, plain arrays in/out, with the
  TypeScript `lib/nlfit.ts` as the reference). The contrast "phasors are instant, fits are slow"
  is itself a teaching point.
- `photonbudget`: 500 fits × 5 methods per update; precompute per preset, recompute on button.
- Per-demo CSS: prefix every selector with `.qb-demo[data-demo="<name>"]` (demo.css loads late,
  `plan/build-issues.md`); slide text and axis labels ≥ 18 px; 1150×560 on slides (two panels plus
  a compact control column; `data-layout="full"` for all panels on exercise pages); nothing
  hover-only; static figure fallbacks for every embed.
- Tests (`demos/test/flim-test.ts`, run by `make test`): the periodic model integrates to $N$ per
  period; IRF convolution of a spike returns the model; the phasor of a sampled exponential lies on
  the semicircle within the binning correction; a two-species mixture lies on the segment at the
  photon fraction; `calibrate` recovers a reference exactly; $P_{h*I} = P_hP_I$ numerically;
  `phasor` equals the conjugate of `dft1d` bin 1 divided by bin 0; $\tau_\phi = \tau_m$ for one
  exponential and $\tau_\phi < \tau_m$ for random mixtures; LM recovers parameters on noise-free
  data for every cost; Poisson ML for one lifetime without IRF over a long window equals the mean
  arrival time; Monte-Carlo relative spread of the ML τ is within 10 % of $1/\sqrt N$ at
  $N = 10^4$; the two-gate formula is exact on a noise-free exponential; global analysis recovers
  amplitudes exactly for known lifetimes; `barycentric3` sums to 1.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `lifetime/`: lecture (`tcspc`, `flimcell` estimate), photon fractions and gates by hand |
| 10:45–12:15 | Unit 2 `decayfit/`: lecture (`decayfit`), Poisson and photon budget by hand, `photonbudget` |
| 13:15–14:45 | Unit 3 `phasor/`: lecture (`modulation`, `phasorplot`), phasors by hand, `flimcell` phasor |
| 15:00–16:30 | Unit 4 `flim-fret/`: lecture (`fretphasor`), FRET and NADH by hand, `flimcell` fret/nadh; quizzes as wrap-up |

## Open questions

1. **Folder name `phasor/`**: this day takes it (owner of phasors, coursemap "Ownership changes").
   The unmixing plan's optional `phasor/` (not built) should become a recall slide in
   `endmembers/` or an optional `spectral-phasor/` unit; `plan/unmixing-day.md` needs that edit
   (not done here).
2. **FRET ownership split**: consistent with `plan/fret-day.md` (FRET day: Förster theory and
   intensity methods, group order colocalisation → FRET → FLIM; this day: lifetime-based FRET,
   phasors, a one-slide recall of $E(r)$). If FLIM is ever taught before FRET, that recall slide
   must carry the minimum alone. Add rows to the coursemap's owners table once agreed.
3. **Nonlinear fitting ownership**: no built unit teaches Gauss–Newton/Levenberg–Marquardt; both
   this day and `plan/pointdetection-day.md` (unit 3, Gaussian spot fitting) teach it with Poisson
   ML and Cramér–Rao. Proposal: one owner row in the coursemap ("nonlinear least squares and
   Poisson ML fitting"), given to point detection (entry after day 2, so usually earlier) with this
   day recalling it in one slide when taken after; the shared code `lib/nlfit.ts` stays here and
   `lib/spotfit.ts` calls it. Until decided, unit 2 carries the concept slide and math layer.
4. **Is unit 3 too dense** (frequency domain + phasor + mixtures + calibration + noise)? Fallback:
   move calibration details and the binning correction to `flim-more/`, keep frequency domain to
   two slides plus the `modulation` demo.
5. **Order within the elective group**: colocalisation → FRET → FLIM is natural (FLIM-FRET builds
   on FRET), but FLIM must stand alone after day 7. The plan assumes nothing from the other two
   days.
6. **Real data** (principle 4): a real TCSPC FLIM image (e.g. a FRET biosensor or label-free NADH)
   with a clear licence, plus its IRF or a reference measurement; public FLIM datasets exist, the
   licence must be checked (teacher notes only for any source/tool names). Until then the
   synthetic `flimScene` carries the demos (with ground truth).
7. Should global analysis get its own optional unit (more on variable projection, image-wide
   lifetime fitting, spatial regularisation of fraction maps) instead of a slide pair in unit 2?
8. Edits elsewhere once accepted (not done here): `endmembers/` recall slide for the spectral
   phasor; `coloc/` notes pointer (already generic); `fourier/exercise.qmd` "Where this leads" could
   name the phasor as an application of a single Fourier coefficient; sidebar section and
   `demos.qmd` gallery entries.

## References

- Lakowicz (2006), *Principles of Fluorescence Spectroscopy*, 3rd ed., Springer (lifetimes,
  frequency domain, FRET, quenching).
- Becker (2005), *Advanced Time-Correlated Single Photon Counting Techniques*, Springer (TCSPC,
  pile-up, IRF); teacher background only.
- O'Connor & Phillips (1984), *Time-Correlated Single Photon Counting*, Academic Press.
- Köllner & Wolfrum (1992), *How many photons are necessary for fluorescence-lifetime
  measurements?*, Chem. Phys. Lett. (F-value, $1/\sqrt N$).
- Hall & Selinger (1981), *Better estimates of exponential decay parameters*, J. Phys. Chem.
  (Poisson ML vs least squares).
- Maus et al. (2001), *An experimental comparison of the maximum likelihood estimation and
  nonlinear least-squares fluorescence lifetime analysis of single molecules*, Anal. Chem.
- Turton, Reid & Beddard (2003), *Accurate analysis of fluorescence decays from single molecules
  in photon counting experiments*, Anal. Chem. (bias of χ² at low counts).
- Istratov & Vyvenko (1999), *Exponential analysis in physical phenomena*, Rev. Sci. Instrum.
  (ill-conditioning of multi-exponential fits).
- Verveer, Squire & Bastiaens (2000), *Global analysis of fluorescence lifetime imaging microscopy
  data*, Biophys. J.
- Golub & Pereyra (2003), *Separable nonlinear least squares: the variable projection method and
  its applications*, Inverse Problems.
- Jameson, Gratton & Hall (1984), *The measurement and analysis of heterogeneous emissions by
  multifrequency phase and modulation fluorometry*, Appl. Spectrosc. Rev.
- Digman, Caiolfa, Zamai & Gratton (2008), *The phasor approach to fluorescence lifetime imaging
  analysis*, Biophys. J.
- Redford & Clegg (2005), *Polar plot representation for frequency-domain analysis of fluorescence
  lifetimes*, J. Fluoresc.
- Stringari et al. (2011), *Phasor approach to fluorescence lifetime microscopy distinguishes
  different metabolic states of germ cells in a live tissue*, PNAS (NADH free/bound).
- Ranjit, Malacrida, Jameson & Gratton (2018), *Fit-free analysis of fluorescence lifetime imaging
  data using the phasor approach*, Nat. Protoc.
- Wallrabe & Periasamy (2005), *Imaging protein molecules using FRET and FLIM microscopy*, Curr.
  Opin. Biotechnol.
- Datta, Heaster, Sharick, Gillette & Skala (2020), *Fluorescence lifetime imaging microscopy:
  fundamentals and advances in instrumentation, analysis, and applications*, J. Biomed. Opt.
- Fereidouni, Bader & Gerritsen (2012), *Spectral phasor analysis allows rapid and reliable
  unmixing of fluorescence microscopy spectral images*, Optics Express (spectral phasor, the
  unmixing day's recall).

## Build notes (2026-11-13, first build)

Built: the four core units `lifetime/`, `decayfit/`, `phasor/`, `flim-fret/` (lec, quiz, exercise),
seven demos (`tcspc`, `flimcell`, `decayfit`, `photonbudget`, `modulation`, `phasorplot`,
`fretphasor`), `lib/decay.ts`, `lib/phasor.ts`, `lib/nlfit.ts`, tests `demos/test/flim-test.ts`,
figures from `tools/figs/flim/figs.py`. Not built: the optional `flim-more/`. Every number in the
units was computed by script or read from the demos' own readouts (fixed seeds).

Deviations from the plan above:

- **Model.** For a Gaussian IRF, `lib/decay.ts` uses the exact model (exponentially modified
  Gaussian, wrapped round the period, integrated over each bin); for a measured IRF array, the
  circular convolution of bins (which shifts by half a bin; tested). The FRET rate functions come
  from `lib/fret.ts` (FRET day), re-exported as `fretEfficiency`, `fretDistance`,
  `lifetimeWithAcceptor`.
- **Two gates** are snapped to whole bins (unequal bin counts per gate biased the estimate).
- **`nlfit.ts`**: Levenberg–Marquardt with costs `ls`, `weighted`, `neyman` (empty bins weight 1),
  `poisson`; log transform for positive parameters, bounds with an active set; no separate
  `costGrid` (the `decayfit` demo builds its landscape with `costValue` and `linearAmplitudes`).
- **`flimcell`** modes: `estimate`, `fit` (global analysis), `phasor`, `fret` (FRET and NADH
  scenes in one group). Simplified: no box selection (click a pixel; binning gives the box), the
  line tool is a checkbox (principal axis of the intensity-weighted cloud), the NADH triangle
  has fixed vertices (free 0.4, bound 2.5, label 4.5 ns), no per-cell segmentation (cell mean and
  brush mean instead; the "per-cell values" preset was dropped).
- **`phasorplot`**: no free (non-exponential) species; IRF delay fixed at 0.4 ns. The preset
  "coarse time bins shrink the phasor" became "coarse time bins **distort** the phasor": with 4
  bins the half-bin phase and aliasing of higher harmonics dominate the sinc shrink (a 2.5 ns
  decay lands outside the semicircle at (0.267, 0.581); calibration on the same bins repairs it).
  Phasor by-hand (d) accordingly uses bin centres: (8, 4, 2, 1) → (0.141, 0.424), outside the
  semicircle, not the plan's (0.40, 0.20).
- **Phasor noise**: the measured spread per coordinate is about ½/√N (0.052 at 100 photons), not
  1/√N; the units say ½/√N.
- **`photonbudget`**: 400 histograms of 64 bins, computed asynchronously in slices (no
  precomputed tables). Measured: one lifetime reaches 10 % at about 100 photons; two lifetimes
  (0.8 and 3 ns, 60 % of photons in the long one) between 3 000 and 10 000: "30 to 100 times";
  the preset is "two lifetimes need far more photons".
- **`decayfit`**: a one-exponential fit to two lifetimes lands between τ̄_a and τ̄_f (1.58 ns for
  1.43 / 2.12), not "roughly τ̄_f"; the unit says so. Least squares with a free background is a
  little low at 10 000 photons (2.42 ± 0.05 vs 2.50).
- **`fretphasor`**: acceptor leak pulls the mixture outside the semicircle only when large (the
  preset uses 70 %); the leak component's own phasor is drawn. Autofluorescence 0.4 ns,
  acceptor 3 ns.
- **Unit 2 global analysis by hand** uses counts (612, 203, 148) in gates 0–2, 2–4, 4–8 ns
  (solution 315 and 695 photons).
- **Recalls**: unit 4 links the FRET day's `fret-physics/` (being built in parallel); the
  `endmembers/` recall slide for the spectral phasor is not done (another day's file).
- **Shared UI**: the demos use `canny/imgview.ts` (`ImgView`, `injectStyle`) and a local
  `phasorplot/plane.ts` (phasor plane with τ ticks, drag handle, log-slider map), candidates for
  `lib/`.
