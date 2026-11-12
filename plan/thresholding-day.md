# Plan: a day on thresholding

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. compute and read a **histogram** (bins, normalised histogram as a probability distribution $p_i$, cumulative histogram), and turn a threshold into a binary mask;
2. derive **Otsu's method**: show that total variance = within-class + between-class variance, that $\sigma_B^2(t) = \omega_0\omega_1(\mu_0 - \mu_1)^2$, and compute the best threshold **by hand** on a small histogram and in one pass over cumulative sums;
3. relate Otsu, **isodata** (Ridler–Calvard, 1D two-means) and **minimum-error** (Kittler–Illingworth, two Gaussians, Bayes decision boundary) thresholds, and predict from the shape of the histogram which one is biased and in which direction;
4. recognise when **one global threshold cannot work** (uneven illumination, unimodal histogram, small objects, noise) and choose a remedy: smoothing first, background correction, **local thresholds** (mean − C, Niblack, Sauvola) with a sensible window, or **hysteresis**;
5. **evaluate** a segmentation against ground truth: confusion matrix, precision/recall, Dice/Jaccard, and the **ROC** and precision–recall curves traced as the threshold sweeps.

Why it fits this audience: a histogram is a probability distribution, Otsu is an
exercise in means and variances (the variance decomposition they know from statistics),
minimum-error thresholding is "where do two normal densities cross" (calculus, a
quadratic equation), and the local methods are local means and standard deviations
computed with box filters from `kernels/`. Everything is 1D on the histogram except
units 3 and 4, so the by-hand examples stay small.

## Prerequisite gap

- **Images as arrays, grey levels, bit depth** (8 vs 16 bit): known from `images/`. One-line recall; the 16-bit case matters for bin choice (unit 1).
- **Box and Gaussian filters, median filter**: known from `kernels/`. Used as preprocessing (unit 1) and to compute local means (unit 3); recall in one slide each, nothing new.
- **Normal distribution, mean, variance**: assumed (audience.md). Bayes' rule and "posterior ∝ likelihood × prior" are not assumed: one slide with a two-class picture before the minimum-error threshold (unit 2).
- **Connected components** (4- vs 8-connectivity): needed only for hysteresis (unit 3), as a picture ("keep the pieces of the low mask that touch a seed"). Labelling as an algorithm belongs to the labelling day (`plan/labelling-day.md`); use the same convention (foreground 8-connected) as the morphology, labelling and watershed plans.
- **Segmentation**: none in Part 1 yet. This day introduces the words *segmentation*, *binary mask*, *foreground/background* and *ground truth*, which the other segmentation days can then use without definition.

## Links to other days (what they can now drop)

This day owns thresholding. The plans below each carry a short thresholding recap
because Part 1 had none; with this day in place each can shrink to a one-line recall
(their files are not edited here; this is a note for whoever revises them):

- **Graph cuts** (`plan/graphcut-day.md`, unit 1 `seg-energy/`): the opening "a histogram, a threshold, why noise breaks it" becomes one line. Its data term $-\log P(I_p \mid \ell)$ with two Gaussians "the threshold is where they cross" is exactly this day's minimum-error threshold (unit 2): a back-reference instead of a new derivation. Graph cuts' "$\lambda = 0$ is thresholding" is the bridge.
- **Ising / Bayesian MRF segmentation** (plan being written in parallel): the same two-Gaussian likelihood plus a prior that makes neighbours agree. Unit 2 here ends with the pixel-wise Bayes decision; the Ising day and the graph-cut day are where "**pixels should agree with their neighbours**" is added. Unit 3 points to both as the principled answer to the noise failure mode.
- **Watershed** (`plan/watershed-day.md`, unit 1): "threshold → binary mask" becomes a one-line recall; the connected-components part of that recap stays (or moves to the labelling day). Its "the brightness varies so no single threshold works" can point to unit 3 here.
- **Morphology** (`plan/morphology-day.md`, unit 1): the two opening slides (histogram, threshold, 0/1 mask) become a one-line recall. **Top-hat background correction stays on the morphology day**; unit 3 here cross-references it as one remedy for uneven illumination (one slide with a picture, no re-teaching), and the `tophat` demo's "threshold the input vs the top-hat" is the natural follow-up.
- **Labelling** (`plan/labelling-day.md`): already assumes thresholding as known. Hysteresis thresholding is taught here (unit 3, as an idea, with a demo); the labelling day's one slide can become "hysteresis = reconstruction + labelling: the implementation of what you saw on the thresholding day".
- **Level sets** (`plan/levelsets-day.md`, unit 4): the thresholding recap becomes a back-reference, and "Chan–Vese with $\mu = 0$ is the threshold at $(c_1 + c_2)/2$, i.e. two-means" is literally the **isodata** fixed point of unit 2.
- **Marching cubes** (`plan/marchingcubes-day.md`, unit 4 "choosing $c$"): "Otsu / histogram methods if the thresholding unit exists" can now point to unit 2. Its `isocontour` demo can reuse `lib/histogram.ts`.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `histogram/`: Histograms and a global threshold

- Motivation: a micrograph (`cell256.png`) and a synthetic image of bright blobs on a dark background. "Which pixels are object?" The simplest answer: those brighter than $t$. The mask $B(x) = [I(x) > t]$ is the first **segmentation**.
- The **histogram** $h_i$ = number of pixels with grey level $i$ (or in bin $i$). Picture first: the image, then its pixels sorted into piles. **Normalised** histogram $p_i = h_i / N$: a probability distribution over grey levels ($\sum p_i = 1$), so mean $\mu = \sum i\,p_i$ and variance $\sigma^2 = \sum (i - \mu)^2 p_i$ are the familiar ones. The histogram forgets *where* pixels are: shuffle the pixels and it is unchanged (one slide with a shuffled image; this is why every method in units 1–2 ignores neighbours).
- **Cumulative** histogram $P(t) = \sum_{i \le t} p_i$: the fraction of pixels at or below $t$, i.e. the background fraction the threshold produces. Read off "keep the brightest 10 %" as $t = P^{-1}(0.9)$ (the **percentile** threshold, a robust baseline when the object fraction is known). Also the first moment $M(t) = \sum_{i \le t} i\,p_i$: both are running sums, which makes unit 2 one pass.
- **Bins**: 8-bit gives 256 natural bins; 16-bit microscopy needs a choice. Too few bins merge peaks, too many give a spiky histogram. One slide with the same data at 8, 32, 256 bins.
- What makes a **good** threshold? A bimodal histogram with a deep valley: any $t$ in the valley gives nearly the same mask (robust). A shallow valley: the mask changes a lot with $t$ (the area vs $t$ curve is steep). Show the mask area as a function of $t$ next to the histogram. "Good" is made precise twice today: as a criterion on the histogram (unit 2) and against ground truth (unit 4).
- **Noise** widens the peaks so they overlap: isolated wrong pixels. **Smoothing first** (Gaussian or median from `kernels/`) averages the noise, so each peak narrows (variance $\sigma^2 / n$ for an $n$-pixel average) and the valley reappears. The price: edges blur and small objects lose contrast (partial-volume pixels fill the valley). Picture: histogram before and after a 3×3 mean.
- **Demo `histthresh`** (new, main demo of the unit): image (left) with the mask overlaid in colour, histogram (right) with a draggable threshold line, the cumulative histogram as a second curve (toggle), and the mask area vs $t$ curve. Controls: threshold, number of bins, noise level, Gaussian blur $\sigma$, median 3×3 on/off, linear/log histogram scale. Inputs: synthetic blobs (seeded; contrast and noise adjustable), `cell256.png`, a low-contrast image. A hidden-by-default criterion curve (Otsu's $\sigma_B^2(t)$, `data-criterion="otsu"`) so the same demo serves unit 2. Presets: "clean bimodal: any t in the valley works", "noise fills the valley", "blur restores it", "too few bins", "the micrograph: where is the valley?", "percentile threshold".
- By hand: a 5×5 image with values 0–7: draw the histogram, the normalised and the cumulative histogram; threshold at $t = 3$ and at $t = 4$; which pixels change and why is the change small or large?

### 2. `otsu/`: Choosing the threshold from the histogram

- Picture: a two-peak histogram split at $t$ into class 0 (background, $i \le t$) and class 1. Class weights $\omega_0(t) = P(t)$, $\omega_1 = 1 - \omega_0$; class means $\mu_0, \mu_1$; class variances $\sigma_0^2, \sigma_1^2$.
- **Otsu** (1979): a good split makes each class tight. Minimise the within-class variance $\sigma_W^2 = \omega_0\sigma_0^2 + \omega_1\sigma_1^2$.
  - Derive the decomposition $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ with $\sigma_B^2 = \omega_0(\mu_0 - \mu)^2 + \omega_1(\mu_1 - \mu)^2$ (add and subtract the class mean inside the square; the cross term vanishes). $\sigma_T^2$ does not depend on $t$, so minimising $\sigma_W^2$ = **maximising the between-class variance**.
  - Using $\mu = \omega_0\mu_0 + \omega_1\mu_1$: $\sigma_B^2(t) = \omega_0\omega_1(\mu_0 - \mu_1)^2$. Reading: large when the classes are far apart *and* both are substantial.
  - One pass: with the running sums of unit 1, $\sigma_B^2(t) = \dfrac{(\mu\,P(t) - M(t))^2}{P(t)\,(1 - P(t))}$. $O(L)$ for $L$ grey levels after the $O(N)$ histogram.
  - **Worked example** (used in the by-hand exercise and in the demo's preset): levels 0–7 with counts $h = (2, 5, 3, 1, 1, 2, 4, 2)$, $N = 20$, $\mu = 3.3$, $\sigma_T^2 = 5.81$. Table over $t = 0..6$ of $\omega_0, \mu_0, \mu_1, \sigma_B^2$: $\sigma_B^2 = 1.21, 3.60, 4.84, \mathbf{5.02}, 4.86, 3.94, 1.52$, so $t^* = 3$ ($\omega_0 = 0.55$, $\mu_0 = 1.27$, $\mu_1 = 5.78$).
- **Isodata / Ridler–Calvard** (1978): start anywhere, set $t \leftarrow (\mu_0(t) + \mu_1(t))/2$, repeat. This is **k-means with $k = 2$ in 1D** (assign to the nearer mean, recompute means). On the example from $t = 5.5$: $4.17 \to 3.75 \to 3.53$, the same split as Otsu. Otsu's optimum is always a fixed point of isodata (set $d\sigma_W^2/dt = 0$ in the continuous version), but isodata can stop at another fixed point: a local vs global optimum, shown in the demo.
- **Minimum-error** (Kittler–Illingworth 1986): model the histogram as $\omega_0 \mathcal{N}(\mu_0, \sigma_0^2) + \omega_1 \mathcal{N}(\mu_1, \sigma_1^2)$.
  - Bayes picture: classify $x$ as object where $\omega_1 f_1(x) > \omega_0 f_0(x)$; the error is the area of the two tails on the wrong side; the error is smallest where the weighted densities **cross** (differentiate the error w.r.t. $t$).
  - Equal variances: take logs, the quadratic terms cancel: $t = \tfrac{\mu_0 + \mu_1}{2} + \tfrac{\sigma^2}{\mu_1 - \mu_0}\ln\tfrac{\omega_0}{\omega_1}$. Equal weights too: the midpoint (isodata). Unequal variances: a quadratic with **two** crossings (the wider class wins in both tails).
  - Kittler–Illingworth fit the two Gaussians to the split at each $t$ and minimise $J(t) = 1 + 2(\omega_0\ln\sigma_0 + \omega_1\ln\sigma_1) - 2(\omega_0\ln\omega_0 + \omega_1\ln\omega_1)$ (one slide; derivation in the notes as the average negative log-likelihood).
  - Consequence to remember: Otsu and isodata implicitly assume **equal variances and similar weights**. With a small object (say 5 % of the pixels) or a much wider background peak, Otsu moves into the larger/wider class; minimum-error follows the Bayes boundary as long as the Gaussian model fits.
- Briefly (one slide each, demo only):
  - **Triangle** (Zack et al. 1977): line from the histogram peak to the far end; threshold at the bin farthest from the line. Designed for a **unimodal** histogram with a long tail of a few object pixels (fluorescent spots), where Otsu fails.
  - **Maximum entropy** (Kapur et al. 1985): maximise $H_0(t) + H_1(t)$, the entropies of the two class distributions $p_i/\omega_k$. Another criterion, another answer: no method is "the" threshold.
  - **Multi-level Otsu**: $\sigma_B^2 = \sum_k \omega_k(\mu_k - \mu)^2$ over $K$ classes; for three classes, search all pairs $(t_1, t_2)$ ($O(L^2)$, fine for 256 levels). Example: nucleus / cytoplasm / background in `cell256.png`.
- **Demo `otsubars`** (new, small): the 8-level histogram of the worked example as draggable bars (heights 0–9). A threshold slider; live table of $\omega_0, \mu_0, \mu_1, \sigma_B^2$ for every $t$, the current row highlighted; a stacked bar showing $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ (total constant while $t$ moves); buttons *isodata step* (draws the two means and the midpoint) and *reset*. Presets: "the worked example", "two equal peaks", "a tiny object class", "isodata stuck at a local fixed point". This is the bridge from the by-hand table to the image demos.
- **Demo `mixture`** (new, main demo): two Gaussians with sliders $\mu_0, \mu_1, \sigma_0, \sigma_1, \omega_1$; the weighted densities, their sum, and a sampled histogram (seeded, sample size slider: smaller samples give a noisier histogram). Vertical markers for the thresholds of Otsu, isodata, minimum-error, triangle, Kapur, and the true Bayes boundary (both crossings when they exist), each with its error rate (shaded tails for the selected one) and the criterion curve ($\sigma_B^2$, $J$, entropy) below on a shared $t$ axis. A small synthetic image drawn from the same mixture (discs with area fraction $\omega_1$) with the mask of the selected method. Presets: "equal and well separated: all agree", "unequal weights: Otsu drifts", "unequal variances: two crossings", "small object 3 %: unimodal-looking, triangle wins", "heavy overlap: every threshold is bad (→ neighbours, graph cuts / Ising)".
- `histthresh` with `data-criterion="otsu"` (and multi-level Otsu toggle) on `cell256.png` closes the unit on a real image.
- By hand: (a) the Otsu table for the worked example (fill $\omega_0, \mu_0, \mu_1, \sigma_B^2$ for three given $t$; the rest is given); (b) two isodata steps from $t = 5.5$; (c) the equal-variance Bayes threshold for $\mu_0 = 50$, $\mu_1 = 150$, $\sigma = 20$, $\omega_1 = 0.1$, and which way it moves from the midpoint; (d) verify $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ at $t^* = 3$.

### 3. `local-threshold/`: When one threshold is not enough

- Failure gallery (pictures, then the demo): **uneven illumination** (vignetting, a gradient: the background on the right is brighter than objects on the left; the histogram peaks smear into one), **unimodal** histogram (few small objects), **small or faint objects** (partial volume), **noise** (isolated wrong pixels: no threshold on single pixels fixes it).
- Remedy 1, correct the background first: divide by or subtract a smooth background estimate (a heavily blurred image, a fitted plane/quadratic by least squares, or the **top-hat** from the morphology day: one cross-reference slide, not re-taught). Then a global method works again.
- Remedy 2, **local (adaptive) thresholds** $T(x)$ computed in a window $W$ around each pixel:
  - **mean − C**: $T(x) = m_W(x) - C$ (the local mean is a box filter from `kernels/`).
  - **Niblack** (1986): $T = m + k\,s$ with the local standard deviation $s$ ($k \approx -0.2$). Fails in empty background: there $s$ is noise and half the noise is above $T$.
  - **Sauvola** (2000): $T = m\,(1 + k(s/R - 1))$, $R$ = dynamic range of $s$ (128 for 8-bit), $k \approx 0.2$–$0.5$: in flat regions $s \ll R$ and $T$ drops below $m$, so background stays background.
  - **Window size** is the real parameter: larger than the objects (otherwise the window sits inside an object and splits it), smaller than the scale of the illumination change. Same trade-off as the top-hat structuring element.
  - Cost: local mean and variance in $O(1)$ per pixel with the **integral image** (summed-area table): the 2D version of the cumulative histogram, $\sum_W I$ = four lookups. Variance from $\sum I$ and $\sum I^2$.
- Remedy 3, **hysteresis** (Canny 1986): two thresholds $t_{\text{low}} < t_{\text{high}}$. Pixels above $t_{\text{high}}$ are sure seeds; pixels above $t_{\text{low}}$ are kept only if **connected** to a seed. Faint parts of real objects survive, isolated noise above $t_{\text{low}}$ does not. Picture first on a 1D profile, then on a grid with 8-connectivity. The implementation (reconstruction, labelling) belongs to the morphology and labelling days.
- Pointer: when noise is the problem, the principled fix is to make **pixels agree with their neighbours**: the Ising / Bayesian MRF day and the graph-cut day (thresholding is the $\lambda = 0$ case of the graph-cut energy).
- **Demo `localthresh`** (new): a seeded synthetic image with known ground truth (discs of several sizes, a text-like pattern option) under a linear gradient + vignette + noise, and a real image (see open questions). Panels: input with mask overlay; threshold surface $T(x)$ as an image; a draggable profile line showing $I$ and $T$ along it (the 1D picture of "the threshold follows the background"). Method: global Otsu, background-divided Otsu (blur $\sigma$ slider), mean − C, Niblack, Sauvola; sliders window size, $k$, $C$, illumination strength, noise. Readout: Dice against ground truth (links to unit 4). Presets: "global Otsu fails on a gradient", "divide by the background", "window too small: discs hollow out", "window too large: back to global", "Niblack in empty background", "Sauvola fixes it".
- **Demo `hysteresis`** (new, small): a 1D profile (top) and a 128×128 image (bottom): thin, faint curvilinear structures (synthetic vessels/filaments with a fading intensity) plus noise; real option: Sobel magnitude of `cell256.png`. Sliders $t_{\text{low}}$, $t_{\text{high}}$ (both also drawn on the histogram), 4/8-connectivity. Colours: seeds ($> t_{\text{high}}$), kept ($> t_{\text{low}}$, connected), rejected ($> t_{\text{low}}$, not connected). Side by side: single threshold at $t_{\text{low}}$ and at $t_{\text{high}}$. Presets: "high alone breaks the filament", "low alone keeps noise", "hysteresis keeps both ends", "4- vs 8-connectivity".
- By hand: (a) mean − C and Niblack thresholds along a 1×10 profile with a ramp background, window 3, borders replicated; which pixels flip relative to a global threshold? (b) hysteresis on a 6×6 grid with given $t_{\text{low}}, t_{\text{high}}$, 4- and 8-connectivity; (c) a 4×4 integral image and one window sum from four lookups.

### 4. `seg-eval/`: Is the segmentation good? Evaluation against ground truth

- **Ground truth**: a synthetic image where the truth is known by construction, or a manual annotation (which has its own uncertainty, mostly at boundaries; two annotators disagree). Pixel-wise comparison of a predicted mask with the truth.
- **Confusion matrix**: TP, FP, FN, TN; picture: the mask overlay coloured by the four cases. Derived rates: precision $= TP/(TP + FP)$, recall (sensitivity, TPR) $= TP/(TP + FN)$, specificity, FPR $= FP/(FP + TN)$, accuracy.
- Why **accuracy misleads**: objects are 3 % of the pixels; "all background" scores 97 %. **Dice** $= 2TP/(2TP + FP + FN)$ (= F1, the harmonic mean of precision and recall) and **Jaccard** / IoU $= TP/(TP + FP + FN)$; $J = D/(2 - D)$. Neither uses TN.
- **Sweeping the threshold**: every $t$ gives one confusion matrix. The **ROC curve** (TPR vs FPR) and the **precision–recall curve** are traced as $t$ moves. Computed in one pass from the histograms of the object and background pixels: TPR$(t)$ and FPR$(t)$ are $1 -$ the cumulative histograms of the two classes (unit 1 again; with the two-Gaussian model they are the tail areas of unit 2). Area under the ROC curve = probability that a random object pixel is brighter than a random background pixel.
- Where on the curve? Max Dice, max Youden's $J = \text{TPR} - \text{FPR}$, or a fixed FPR; mark the automatic thresholds of unit 2 on the curves. ROC looks good under heavy class imbalance while PR does not: show both.
- Pixel metrics vs **object** metrics: one merged pair of touching nuclei has a near-perfect Dice but a wrong count. Pointer to the labelling day (object-level matching and counting) and the watershed day (splitting).
- Boundary-sensitive measures (Hausdorff distance) as a one-slide pointer.
- **Demo `threshroc`** (new): synthetic image with ground truth (adjustable object fraction, contrast, noise, illumination gradient) and a real option with a manual annotation (see open questions). Left: image with TP/FP/FN/TN overlay; middle: the confusion matrix as a 2×2 table with counts and colours, precision, recall, Dice, Jaccard, accuracy; right: ROC and PR curves with the current threshold as a moving dot, and the Otsu / minimum-error / max-Dice thresholds marked. A threshold slider, plus a toggle to evaluate the `localthresh` result instead (one point, no curve: a local method has no single $t$). Presets: "balanced classes", "3 % objects: accuracy lies", "Otsu vs max Dice", "uneven illumination: no t is good (ROC flattens)", "noise: smooth first and watch the curve move up".
- By hand: (a) a 5×5 prediction and ground truth: confusion matrix, precision, recall, Dice, Jaccard, accuracy; (b) the same image thresholded at three values of $t$: three ROC points and the curve through them; (c) show $J = D/(2 - D)$.

### 5. (optional / further reading) `threshold-more/`: More than one feature, more than two classes

- Multi-level Otsu in depth (three classes on `cell256.png`; why the search grows as $L^{K-1}$; dynamic programming as the fix, pointer to the dynamic-programming day).
- 2D histograms: intensity vs local mean (a "spatial" Otsu); colour thresholding (per channel, or in a hue–saturation space). Leads naturally to clustering (k-means in more dimensions: isodata generalised) and to pixel classifiers (the step to machine learning, one slide).
- **Demo**: reuse `histthresh` with `data-levels="3"` (multi-level Otsu, two threshold lines), and `mixture` with three components.

## Shared engineering

- `demos/src/lib/histogram.ts` (new): histogram with given bins and range (8-bit, 16-bit, float images), normalisation, cumulative sums $P(t)$, $M(t)$ and second moments, class statistics for a split; an SVG histogram drawer (bars, log scale, draggable threshold lines) on top of the shared plot helper (`lib/plot.ts`, proposed by the morphology plan). Reused by the marching-cubes `isocontour` demo and any demo with a threshold slider (`tophat`, `distws`, `regionfilter`, `chanvese`'s $c_1, c_2$ markers).
- `demos/src/lib/threshold.ts` (new): global methods from a histogram (percentile, Otsu, multi-level Otsu, isodata with a recorded trace, Kittler–Illingworth, triangle, Kapur), each returning the threshold *and* its criterion curve for plotting; local methods (mean − C, Niblack, Sauvola) via an integral image; hysteresis; evaluation (confusion matrix, Dice, Jaccard, precision/recall, ROC and PR curves from two class histograms).
- Hysteresis needs a connected flood from seeds: call the reconstruction in `lib/morph.ts` (morphology plan) or the labelling in `lib/label.ts` (labelling plan) rather than writing a third flood fill; if neither exists yet, a BFS in `threshold.ts` that is later replaced.
- Integral image, box mean/variance, Gaussian and median: in `lib/imageops.ts` (the agreed home for linear filters across plans).
- **Synthetic images with known ground truth**: generators in `lib/imageops.ts` (the morphology plan already proposes "spots on an uneven background" and "discs of several sizes" there) that return `{image, truth}`: discs/blobs with given object fraction, contrast and blur (partial volume), additive Gaussian noise, illumination gradient and vignette, faint filaments for hysteresis. All seeded with `rng(seed)`.
- Image: reuse `demos/src/fourier2d/cell256.png` (256×256, 8-bit), at full size; everything here is $O(N)$ or $O(N + L)$ and runs in a few ms in TypeScript, so no Wasm.
- Tests (extending `demos/test/`): Otsu against brute force (compute $\sigma_W^2$ directly for every $t$); the identity $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ on random histograms; the worked example's table and $t^* = 3$; Otsu's threshold is an isodata fixed point; Kittler–Illingworth on a densely sampled equal-variance mixture lands near the analytic Bayes threshold; integral-image local mean/variance against a naive window; ROC from histograms against ROC from sorting the pixels.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560 on a slide (image 256 px + histogram/curves panels), labels ≥ 18 px.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`histthresh`) |
| 10:45–12:15 | Unit 2 lecture, Otsu table by hand, `otsubars`, `mixture` |
| 13:15–14:45 | Unit 3 lecture + exercise (`localthresh`, `hysteresis`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`threshroc`); quizzes as wrap-up |

## Open questions

- How long is a "day" (hours, lecture vs exercise)? Unit 2 is the densest; if time is short, Kittler–Illingworth's $J(t)$ can stay in the notes and the lecture keeps only "where the Gaussians cross".
- Evaluation as its own unit (proposal) or spread over the units? It is needed by every segmentation day; owning it here gives the others a shared vocabulary and the same `threshroc` metrics code.
- Hysteresis here or on the labelling day? Proposal: the idea and the demo here, the algorithm (reconstruction + labelling) there.
- A real image with uneven illumination and a manual ground truth for units 3–4: `cell256.png` has no clear illumination gradient and no annotation. Candidates: a fluorescence nuclei image with vignetting and annotations (e.g. Broad Bioimage Benchmark Collection; licence to be checked, same search as the watershed and morphology plans), or a document photo with a shadow (classic for Sauvola; course team's own photo for a clear licence). Otherwise annotate `cell256.png` ourselves.
- Colour images: stay greyscale (proposal), with colour only in the optional unit?
- Programming language for the code parts is still open (see README).

## References

- Otsu (1979), *A threshold selection method from gray-level histograms*, IEEE Trans. SMC.
- Ridler & Calvard (1978), *Picture thresholding using an iterative selection method*, IEEE Trans. SMC.
- Kittler & Illingworth (1986), *Minimum error thresholding*, Pattern Recognition.
- Kapur, Sahoo & Wong (1985), *A new method for gray-level picture thresholding using the entropy of the histogram*, CVGIP.
- Zack, Rogers & Latt (1977), *Automatic measurement of sister chromatid exchange frequency*, J. Histochem. Cytochem. (triangle method).
- Niblack (1986), *An Introduction to Digital Image Processing*, Prentice Hall.
- Sauvola & Pietikäinen (2000), *Adaptive document image binarization*, Pattern Recognition.
- Canny (1986), *A computational approach to edge detection*, PAMI (hysteresis).
- Viola & Jones (2001), *Rapid object detection using a boosted cascade of simple features*, CVPR (integral image; also Crow 1984, summed-area tables, SIGGRAPH).
- Sezgin & Sankur (2004), *Survey over image thresholding techniques and quantitative performance evaluation*, J. Electronic Imaging.
- Fawcett (2006), *An introduction to ROC analysis*, Pattern Recognition Letters.
- Taha & Hanbury (2015), *Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool*, BMC Medical Imaging.
