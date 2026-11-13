# Plan: a day on point (spot) detection

Internal working doc (plan/ is not rendered). Status: draft 2026-11-12; **core units 1–4 built 2026-11-13** (see "Build notes" at the end); optional unit 5 not built.
**Course map (2026-11-13, self-study restructure):** day 20 of 51, Part IV "Detecting points, lines and features". Hard prerequisites: Edges, blobs and ridges (day 2). Soft (see also): Bit depth and noise (day 3), Image formation (day 4), Morphology (day 13), Evaluation and statistics (day 37). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part IV (with Hough and integral images),
entry after **day 2, Edges, blobs and ridges**. In practice it works best after day 4 (image
formation), because units 2 and 3 lean on `cameranoise/` (day 3) and `psf/` (day 4); see
open question 1. It is a prerequisite of the planned **tracking** day (Motion group: "tracking
after labelling and the point-detection day"), and the anchor of the planned **topic page
"Point / spot detection"**.

## Goal of the day

After the day a student can:

1. explain spot detection as **filter, then find local maxima, then threshold**, and why the
   best linear filter for a spot of known shape in white noise is the spot shape itself (the
   **matched filter**: a Gaussian of the PSF width, or its zero-mean version, which is a
   LoG/DoG-like centre–surround);
2. run **non-maximum suppression** for points by hand (a pixel is a spot candidate if it equals
   the maximum of its window), choose the window radius from the minimum spot separation, and
   relate it to NMS elsewhere in the course (Canny across the edge, Hough accumulators,
   phase-correlation surfaces, boxes in deep-learning detectors);
3. compare the **Gaussian/LoG/DoG** detectors (recalled from day 2) with the **à trous
   (multiscale product) detector**, and say what each assumes about spot size and background;
4. set a detection threshold **from the noise**, not by eye: estimate the noise (MAD), compute
   the noise of the filtered image, and predict the number of **false positives per image** for a
   threshold of $k$ noise standard deviations; explain why the naive Gaussian tail
   **underestimates** it (maxima of noise are not ordinary pixels) and how Poisson noise changes
   the threshold with the local background;
5. localise a spot with **sub-pixel** precision by centroid, by a three-point Gaussian
   (log-parabola) fit and by **least-squares Gaussian fitting**; predict the **bias** of each
   (background, window, pixel locking) and the **precision**, which falls as
   $\sigma_\text{PSF}/\sqrt N$ with the photon count $N$ (Cramér–Rao intuition);
6. say when two spots **merge** into one maximum (separation below $2\sqrt{\sigma_\text{PSF}^2 +
   \sigma_\text{filter}^2}$, Sparrow recalled), and how multi-spot fitting with a penalty for extra
   spots splits them;
7. **evaluate** a spot detector against ground truth: one-to-one matching within a distance gate,
   TP/FP/FN, precision, recall, F1, the localisation error (RMSE) of matched pairs, and the
   precision–recall curve as the threshold sweeps;
8. *(math)* derive the matched filter (Cauchy–Schwarz), the noise of a filtered image
   ($\sigma_n^2\sum w_k^2$), the expected number of noise maxima above a threshold (Euler
   characteristic heuristic), the three-point Gaussian fit, the least-squares normal equations for
   a Gaussian spot (Gauss–Newton), and the Cramér–Rao bound $s^2/N$ for a Gaussian PSF.

Why it fits this audience: it is filtering and maxima (day 2), the normal distribution and its
tail, least squares, and one inequality (Cauchy–Schwarz). It turns three earlier ideas (blob
detection, the noise model, the PSF) into one complete measurement with an error bar, and it is
the method most microscopy students will use first (counting mRNA spots, vesicles, beads,
single molecules, particle tracking).

## Prerequisite gap

- **Gaussian smoothing, LoG/DoG, scale-normalisation, detection as maxima in $(x, y, \sigma)$,
  "a Gaussian spot of width $s$ peaks at σ = s"** (day 2, `scale-space/`, built; the deck already
  says "Spot detection in fluorescence images: LoG/DoG at a known spot size, then local maxima"
  in *Where this leads*). Recalled in one slide in unit 1; nothing re-derived. The `blobscale`
  demo is the reference picture.
- **NMS** (day 2, `canny/`): NMS *across an edge*; unit 1 generalises to "maximum in a window".
- **Noise model, shot noise, Poisson counts, SNR, variance stabilisation as a pointer**
  (day 3, `cameranoise/`, built). Unit 2 recalls the "usual noise model" slide (Poisson + read
  noise) in one slide. If a group takes this day straight after day 2, that slide must stand
  alone (it is written to): "photons arrive at random; the variance equals the mean; plus a
  constant read noise".
- **PSF and its Gaussian approximation** $\sigma = 0.21\lambda/\mathrm{NA}$ (day 4, `psf/`,
  built: "spot detectors should be tuned to it"), **Sparrow $d = 2\sigma$ for a Gaussian PSF**
  (`resolution/`, built). Units 1, 3 and 4 recall them in one line each; if day 4 has not been
  taken, the Gaussian spot is simply stated as the model ("a point of light is imaged as a small
  Gaussian of width σ").
- **Least squares** (linear algebra; `stitching/`'s global solve and `lsunmix/` are the nearest
  course pictures): unit 3 uses normal equations and one Gauss–Newton step; the concept layer
  needs only "adjust the parameters until the model looks like the data".
- **Matching, F1, AP** (day 37, `object-eval/`): if taken after day 37 this is a recall; if
  taken before, unit 4 has two self-contained concept slides on one-to-one matching (see
  overlaps). **Multiple testing** (day 37, `replicates/`): one-line pointer in unit 2 ("many
  maxima = many tests").
- **Morphological maximum filter / h-maxima** (day 13, `morph-grey/`, `morph-reconstruct/`): used
  as named pictures only ("the maximum filter"; "h-maxima: keep peaks by height, not by window");
  works before day 13 because the concept is a window maximum.

## Overlaps / who owns what

This day **owns spot detection as a task**: the detection pipeline (filter → maxima →
threshold), the matched-filter argument, detection thresholds from noise statistics and false
positives per image, sub-pixel localisation by fitting, crowded spots, and the evaluation of
point detections by distance-gated matching. Everything below is recalled in at most one slide.

| Plan / unit | What this day takes | What stays there |
|---|---|---|
| Edges day `scale-space/` (owner of LoG/DoG, scale space, $(x,y,\sigma)$ maxima) | Nothing re-taught. LoG/DoG are *used* as spot filters; unit 1 adds why a centre–surround filter is the matched filter with the background removed | All of LoG/DoG, normalisation, √2·σ, `blobscale`. `blobscale`'s "hits/misses/false alarms" readout stays as is; this day's `spotdetect` is the spot-specific (one known size, many spots) version |
| Wavelets `cwt/` (optional, **not built**) | **À trous spot detection** (Olivo-Marin multiscale product, MAD per scale, $k\hat\sigma_j$ thresholds) and its demo idea `atrous` move here, into unit 1 and the `spotdetect` demo (filter = "à trous product"). The à trous transform is taught self-contained as "repeated B3-spline blurring with holes; differences of consecutive blurs", i.e. a DoG stack without downsampling, so no wavelet theory is needed | The à trous transform *as a wavelet* (undecimated DWT, perfect reconstruction, shift invariance), CWT and scalogram; `cwt/` recalls spot detection in one slide with a pointer here. **Proposed ownership change** (open question 2) |
| Image formation `localisation/` (optional, **not built**) | The fitting part: centroid vs Gaussian least squares vs maximum likelihood, precision ≈ $\sigma/\sqrt N$, the effect of background and pixel size, and the demo idea `localise` (becomes the *precision* tab of `spotfit`, unit 3). Localisation microscopy (switching molecules on a few at a time) becomes this day's optional unit 5 `smlm/` | One slide in `resolution/` or `illposed/`: "a single point can be *localised* far better than it can be *resolved*; see the point-detection day". SIM stays an image-formation pointer. **Proposed ownership change** (open question 2) |
| FFT registration `stitching/` (built: parabolic peak, centroid, upsampled DFT, pixel locking, drift in localisation microscopy) | Nothing re-taught. Unit 3 recalls the parabola and pixel locking in one slide and adds what is new for spots: the log-parabola is exact for a Gaussian, and fitting the model to all pixels beats any three-point rule | Sub-pixel peaks of correlation surfaces, the `subpixel` demo, drift correction. `smlm/` recalls `drift` |
| Hough (`hough-lines/`, `hough-circles/`): accumulator peaks, NMS in parameter space | Nothing. Unit 1's "same operation everywhere" slide shows an accumulator next to a spot image | Voting, accumulators, their peak finding |
| Evaluation `object-eval/` (owner: object matching, assignment, F1, AP, naming trap) | **Matching points by distance** (a gate radius instead of an IoU threshold) and the **localisation error** of matched pairs are new here; one-to-one matching, greedy vs optimal, F1 and AP are recalled (one slide) or, before day 37, stated in two concept slides without the math layer, which stays in `object-eval/` | IoU, the assignment math, uniqueness above 0.5, AP definitions, the naming trap |
| Modern deep learning `detection/` (built: boxes, greedy box NMS, heat-map/centre-point detection) | Nothing. One line: "a network can learn the filter (a heat map), but the maxima, NMS, threshold and matching are this day's" | Boxes, anchors, box NMS, learned detectors |
| Morphology `morph-reconstruct/` (h-maxima), `morph-grey/` (top-hat) | Nothing. h-maxima named as the prominence-based alternative to the window rule; white top-hat as a background remover | Both operators |
| Deconvolution (owner of regularised inverse problems) | Nothing. One line in unit 4: "spots as a sparse sum of PSFs" is a regularised inverse problem with a sparsity penalty (pointer to deconvolution and wavelet sparsity) | Regularisation |
| Integral images (same group): box-filter DoG / determinant of Hessian, boosting | Nothing. One line in unit 1: a box-filter centre–surround is the fast approximation | Box filters |
| Tracking (planned, Motion group) | Detections with positions and uncertainties are its input | Linking, gating over time, Kalman, MSD |

**Concept owners table:** this day would add the row "Spot/point detection (matched filter,
detection thresholds from noise, false positives per image, sub-pixel localisation by fitting,
Cramér–Rao, distance-gated point matching) | Point detection" to `plan/coursemap.md` (not edited
here).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus one optional unit. Order:
find candidates (unit 1), decide which are real (unit 2), measure where they are (unit 3), then
the hard cases and how to score the whole detector (unit 4).

### 1. `spot-filters/`: Finding spots — filter, then local maxima

Concept layer:

- **The task** in pictures: mRNA spots (FISH), vesicles, beads, single molecules, stars. Output
  is a **list of points** (x, y, brightness, score), not a mask. Three steps: **filter** (make a
  spot stand out from noise and background), **find local maxima** (one candidate per spot),
  **threshold** (unit 2), then **refine** the position (unit 3).
- **A spot is a copy of the PSF** (recall `psf/` in one line: Gaussian with $\sigma =
  0.21\lambda/\mathrm{NA}$, 78 nm at NA 1.4; in pixels typically σ ≈ 1–2 px). So unlike day 2's
  blobs, spots have **one known size**: no search over scale needed.
- **Why filter**: single pixels are noisy; a spot is spread over ~$\pi\sigma^2$ pixels, so
  averaging over the spot collects its signal. **The matched filter**: to detect a known shape in
  white noise, correlate with that shape. Worked with numbers: amplitude 20, noise sd 10, σ = 1.5:
  single-pixel SNR 2; matched-filter SNR $20\sqrt{\pi\cdot 1.5^2}/10 \approx 5.3$. Smoothing wider
  than the spot does not help (it averages in pure noise); narrower leaves noise.
- **Background**: a smooth background (uneven illumination, cell autofluorescence) shifts every
  pixel. Remove it by making the filter **zero-mean**: spot template minus its local mean = a
  centre–surround filter, which is (almost) day 2's **DoG/LoG** at σ ≈ spot σ (recall
  `scale-space/` in one slide: "a Gaussian spot of width s peaks at σ = s"). Alternatives on one
  slide: subtract a heavily blurred image, or a white top-hat (morphology day, pointer).
- **À trous detector** (Olivo-Marin): blur repeatedly with a small B3-spline kernel
  $(1, 4, 6, 4, 1)/16$, each time with the taps spread twice as far apart ("with holes": no
  downsampling, every level keeps the full image size); the differences of consecutive blurs are
  band-pass images $w_1, w_2, \dots$ (a DoG stack, recall). Noise lives mainly in $w_1$, spots of
  1–2 px in $w_2$–$w_3$, background in the last blur. Threshold each $w_j$ at $k\hat\sigma_j$ and
  **multiply** the kept scales: spots persist across scales, noise does not. Robust to uneven
  background by construction. The wavelet view is a pointer to the wavelets day.
- **Local maxima**: a pixel is a candidate if it is **the largest in its window** (radius $r$).
  Implementation as a picture: compare the image with its **maximum filter** (grey dilation,
  morphology day) and keep the pixels where they are equal. Worked on a 5×5 patch.
- **Choosing the window**: $r$ ≈ the smallest separation you expect between real spots. Too small:
  noise ripples on one spot give several candidates; too large: a dim spot next to a bright one is
  suppressed. **Plateaus** (equal neighbours after rounding to integers, bit depth): keep one pixel
  per plateau. Alternative: **h-maxima** (morphology day): keep peaks by their height above the
  surrounding saddle, independent of distance.
- **Same operation everywhere** (one slide, the topic-page picture): NMS across an edge (Canny,
  day 2), maxima in $(x, y, \sigma)$ (blobs, day 2), peaks of a Hough accumulator, of a
  phase-correlation surface (day 23), greedy NMS of boxes (deep-learning detectors). Only the
  neighbourhood and the score differ.
- Failure modes, each a demo preset: uneven background with a plain Gaussian filter; filter too
  narrow (noise maxima) or too wide (close spots merge, dim spots drown); window too small
  (double detections) or too large (lost neighbours); bright out-of-focus blobs and debris
  (bigger than a spot: a scale check or the Hessian determinant rejects them, pointer to
  `hessian/`).
- Pointer (one line): image libraries call these "peak local max", "LoG/DoG blob detection" or
  "wavelet spot detection".

Math layer (`## The math: … {.math}` slides):

- *Matched filter*: response $r = \sum_k w_k (A p_k + n_k)$ with white noise of variance
  $\sigma_n^2$; signal $A\,\mathbf w^\top\mathbf p$, noise sd $\sigma_n\|\mathbf w\|$;
  $\mathrm{SNR} = A\,\mathbf w^\top\mathbf p/(\sigma_n\|\mathbf w\|) \le A\|\mathbf p\|/\sigma_n$ by
  Cauchy–Schwarz, with equality iff $\mathbf w \propto \mathbf p$. For a Gaussian spot of peak 1,
  $\|\mathbf p\|^2 \approx \pi\sigma^2$. With an unknown constant background, maximise over $\mathbf w$
  with $\sum w_k = 0$: $\mathbf w \propto \mathbf p - \bar p$ (projection onto the zero-sum
  subspace).
- *Noise after filtering*: variance $\sigma_n^2\sum_k w_k^2$ (recall `canny/` math); for a
  normalised 2D Gaussian of width $\sigma_f$, $\sum w_k^2 \approx 1/(4\pi\sigma_f^2)$.
- *À trous*: $a_j = h_j * a_{j-1}$ with $h_j$ the B3 kernel dilated by $2^{j-1}$; $w_j = a_{j-1} -
  a_j$; $a_0 = a_J + \sum_j w_j$ (telescoping, as the Laplacian pyramid on day 2); the multiscale
  product $P = \prod_{j \in S} \tilde w_j$ of thresholded scales.
- *Local maxima*: $M = \{p : f(p) = (f \oplus B_r)(p)\}$ (dilation by a disk), plus a tie rule.

**Demo `peaks1d`** (new, small, 1D): a 256-sample profile with Gaussian spots of σ = 1.5 at chosen
positions and amplitudes (two spots 3 px apart, a dim spot next to a bright one, a spot on a
ramp), seeded noise. Rows: raw profile; filtered profile (Gaussian / zero-mean Gaussian / DoG /
à trous level product), with the filter kernel drawn to scale; candidates as ticks (window
radius $r$ shown as a bracket around the selected maximum); optional h-maxima mode with the
height $h$ shown as a bar from the peak to its saddle. Sliders: filter σ, $r$, noise, $h$;
toggle *quantise to integers* (plateaus). Readouts: number of candidates, single-pixel SNR vs
filtered SNR of the selected spot. Presets: "raw maxima: one per noise wiggle", "matched filter",
"filter too wide: the close pair merges", "window too small: double peaks", "window too large:
the dim neighbour is lost", "ramp: zero-mean filter", "plateau", "h-maxima instead of a window".

**Demo `spotdetect`** (new, main demo): a 256×256 synthetic fluorescence image from
`lib/synth.ts` `spots()` (40–150 spots, σ = 1.5 px, amplitudes 20–120 above background, uneven
background with slope and bump, Gaussian or Poisson noise; truth known) and, when licensed, a
real spot image (open question 6). Panels: image with detections (circles: green = matched to
truth, red = false, yellow = missed truth; matching from unit 4, radius 2 px); the filtered image
(signed maps with 0 = mid-grey); for the à trous filter, a thumbnail strip $w_1 \dots w_4$, the
last blur and the product. Controls: filter (*none / Gaussian / zero-mean Gaussian / LoG / DoG /
top-hat + Gaussian / à trous product*), filter σ (or scales used for à trous), maxima window $r$,
threshold (in units of the estimated noise, from unit 2; *raw value* toggle), noise level,
background on/off, spot density. Readouts: candidates, kept, TP / FP / FN, F1. Click a detection:
inset with its 9×9 neighbourhood and the window. Presets: "Gaussian filter, flat background",
"uneven background: Gaussian fails, LoG does not", "à trous product", "à trous: only $w_1$ =
noise", "filter σ = 0.5 vs 1.5 vs 4", "window too small", "dense spots", "LoG at one σ is
enough for spots (compare `blobscale`)". `data-presets` to show a subset on slides.

By hand (concept): (a) 5×5 patch: list the local maxima with $r = 1$ (3×3 window) and $r = 2$;
(b) SNR of a spot of amplitude 30 in noise sd 12 at one pixel and after the matched filter at
σ = 1 ($\sqrt{\pi} \approx 1.77$; 2.5 → 4.4); (c) à trous level 1 of the 9-sample spike
$(0,0,0,0,16,0,0,0,0)$: $a_1 = (0,0,1,4,6,4,1,0,0)$, $w_1 = (0,0,-1,-4,10,-4,-1,0,0)$; check
$a_1 + w_1$ = input; compute $a_2$ at the centre with the dilated kernel (44/16 = 2.75); (d) two
spots 4 px apart, spot σ = 1.5: which window radii keep both? (e) predict what a constant
background of 100 does to the response of a Gaussian filter and of a LoG. *(math)*: (f) the
matched filter by Cauchy–Schwarz; (g) the zero-mean matched filter under an unknown constant
background; (h) $\sum w_k^2$ for a sampled Gaussian: show $\approx 1/(4\pi\sigma^2)$ in 2D.

### 2. `spot-threshold/`: How many false spots? Thresholds from noise statistics

Concept layer:

- The question every user asks: "which threshold?" Answer: decide **how many false spots per
  image you accept**, then compute the threshold from the noise. A threshold on raw grey values
  does not transfer between images, exposures or microscopes; a threshold in **units of noise**
  does.
- **Estimating the noise** from the image itself: the median absolute deviation of a high-pass
  image (or of $w_1$ of à trous), $\hat\sigma = \mathrm{MAD}/0.6745$: robust because spots are a
  small fraction of pixels. Worked example with 9 numbers, one of them a spot.
- **Noise after the filter** (recall unit 1): the filtered image has noise sd
  $\sigma_n\sqrt{\sum w_k^2}$ (≈ 0.19 σ_n for a normalised Gaussian at σ = 1.5). Threshold the
  filtered image at $t = k\cdot$that, so $k$ means the same for every filter.
- **False positives per image**: in pure noise every pixel above $k$ sd has probability
  $P(Z > k)$ (0.13 % for $k = 3$). But we threshold **local maxima**, and there are many: about one
  per 20–25 σ_f² pixels (≈ 4 800 in a 512² image at σ_f = 1.5, measured). And a maximum is the
  largest of its neighbourhood, so it is **higher than a typical pixel**: the naive estimate
  4 800 × 0.13 % ≈ 6 false spots is far too optimistic; the measured count is **≈ 100 at k = 3,
  ≈ 3 at k = 4, ≈ 0.05 at k = 5** (20 seeded 512² noise images; the builder recomputes these
  numbers with the shared library). Lesson: $k = 3$ is not "safe" for spot detection; choose $k$
  from the false-positive count you can live with.
- **Detection limit**: with $k$ fixed, the dimmest detectable spot has matched-filter SNR ≈ $k$
  (plus a margin for 50 % detection probability). Converting back: minimum amplitude ≈ $k\,\sigma_n /
  \sqrt{\pi\sigma^2}$. Halving the noise (4× the photons, recall `cameranoise/`) halves it.
- **Poisson noise**: the noise sd grows with brightness ($\sqrt{\text{mean}}$, recall
  `cameranoise/`), so one global $k\hat\sigma$ gives too many false spots on bright background and
  misses dim spots on dark background. Remedies: a **local** noise estimate (from the local
  background level: $\sigma^2 \approx$ gain·background + read noise²), or a variance-stabilising
  transform first ($2\sqrt{x + 3/8}$, the Anscombe transform: the pointer in `cameranoise/` made
  concrete). The demo shows the false spots concentrating on the bright patch.
- **A test per maximum** (one line, pointer to the evaluation day's multiple testing): thousands
  of maxima = thousands of tests; "false spots per image" is the family-wise view, a false
  discovery rate the alternative.
- **Choosing $k$ in practice** (one slide): from a target false-positive count (blank image or
  noise model), from a precision–recall curve against annotated spots (unit 4), or the à trous
  rule $k \approx 3$ per scale (the product of scales lowers the false rate further).
- Failure modes: correlated noise (camera patterns, smoothed images: MAD on the raw image then
  underestimates the filtered noise unless $\sum w_k^2$ is used), structured background (cell
  edges, autofluorescent granules: not noise, the noise model cannot help; a scale or shape test
  can), saturated spots (bit depth).

Math layer:

- *MAD*: for Gaussian noise $\mathrm{median}|x - \mathrm{median}(x)| = 0.6745\,\sigma$.
- *Detection as a test*: with known σ_n and unknown background, the generalised likelihood ratio
  for "spot of shape $\mathbf p$ here" vs "background only" in a window is
  $\big((\mathbf p - \bar p)^\top \mathbf I\big)^2 / \big(\sigma_n^2\|\mathbf p - \bar p\|^2\big)$:
  the squared zero-mean matched-filter response in units of its noise sd. Thresholding it at $k^2$
  is unit 1's detector; the matched filter is not a heuristic but the optimal test.
- *Expected number of maxima above $u$* (Euler characteristic heuristic for smooth Gaussian
  fields, Adler; Worsley): for white noise smoothed by a Gaussian of width $\sigma_f$, area $A$
  pixels, threshold $u$ sd,
  $\mathbb E[\#] \approx A\,\frac{1}{2\sigma_f^2}\,\frac{u\,e^{-u^2/2}}{(2\pi)^{3/2}}$
  (σ_f = 1.5, $A = 512^2$: ≈ 120 at $u = 3$, ≈ 5 at $u = 4$; the measured counts are ≈ 100 and
  ≈ 3, so the formula is a slightly conservative guide). Compare with the naive
  $(\#\text{maxima})\cdot P(Z > u)$, which misses the factor $u^2$-ish growth from taking maxima.
- *Poisson*: $\mathrm{Var} = \text{mean}$; Anscombe $2\sqrt{x + 3/8}$ has variance ≈ 1 for means
  above ≈ 4 (delta method: $\mathrm{Var}\,g(X) \approx g'(\mu)^2\mu$).

**Demo `falsespots`** (new): a 256×256 image in two modes: *noise only* and *spots + noise*;
noise Gaussian or Poisson (with a bright background patch); the filter from unit 1 (Gaussian or
LoG, σ slider). Panels: the filtered image with the maxima above threshold marked; the
**histogram of local-maximum values** (in noise sd units) with the standard normal tail drawn
over it (the shift to the right is visible); a plot of **false spots per image vs $k$**:
measured (averaged over 16 seeded noise images, computed once per σ), naive tail estimate, Euler
characteristic estimate; on the spots image also the recall vs $k$, so the trade-off is one
picture. Controls: $k$, filter σ, noise type, background level, *local noise estimate* toggle,
*Anscombe* toggle, image size (to show that false spots scale with area). Readouts: $\hat\sigma$
(MAD) vs true σ, noise of the filtered image, false spots, missed spots. Presets: "k = 3 is not
safe", "maxima are higher than pixels", "the formula vs the count", "bigger image, more false
spots", "Poisson: false spots on the bright patch", "local noise estimate fixes it", "Anscombe
fixes it", "dim spots: the detection limit".

By hand (concept): (a) MAD of $(3, -1, 0, 2, -2, 1, 25, -1, 0)$: median 0, absolute deviations
sorted → MAD = 1, $\hat\sigma \approx 1.48$ (the spot value 25 does not matter; the sd would);
(b) noise of the filtered image for σ_n = 10 and a Gaussian of σ_f = 1.5 (≈ 1.9); threshold for
$k = 4$; (c) a 2048² image at σ_f = 1.5: how many noise maxima, and roughly how many false spots
at $k = 3$, 4, 5 (scale the 512² numbers by 16)? (d) the dimmest spot detectable at $k = 4$ with
σ_n = 10, σ = 1.5; (e) Poisson: background 100 vs 400 photons: noise sd 10 vs 20; with one global
threshold set for the dark region, what happens in the bright one? *(math)*: (f) derive
$0.6745$ for the MAD; (g) the GLR statistic of a known-shape spot with unknown amplitude and
background; (h) evaluate the Euler-characteristic formula for given $A$, σ_f, $u$; (i) the
Anscombe variance by the delta method.

### 3. `spot-fit/`: Sub-pixel localisation — centroids, Gaussian fits and the Cramér–Rao bound

Concept layer:

- The maximum pixel gives the position to ±0.5 px (100 nm at 200 nm pixels). The spot covers
  many pixels, and all of them carry information about the centre: we can do **much** better.
  Recall `stitching/`: sub-pixel peaks of a correlation surface (parabola, pixel locking).
- **Centroid** (centre of mass) in a window: simple, fast. Biased **towards the window centre**
  when there is background (the background's own centroid is the window centre), and noisy pixels
  at the window edge add a lot of variance. Fixes: subtract the background first, use a small
  window, iterate the window onto the centroid. Worked example below.
- **Three-point Gaussian fit** per axis: the logarithm of a Gaussian is a parabola, so a parabola
  through the logs of the peak and its two neighbours gives the centre **exactly** for a noise-free
  Gaussian (unlike the parabola on raw values, which is biased: pixel locking). Worked numbers:
  Gaussian σ = 1, true offset 0.30 px: centroid of 3 pixels 0.16, parabola 0.25, log-parabola
  0.30; with a background of 0.5 added, the centroid drops to 0.10.
- **Fit the model to all pixels** (least squares): spot model = background + amplitude ×
  Gaussian(x − x₀, y − y₀; σ), parameters $(x_0, y_0, A, B, \sigma)$; adjust them until the model
  image matches the data (sum of squared differences smallest). Picture: data, model, residual.
  For fixed $(x_0, y_0, \sigma)$ the best $A, B$ are a linear least-squares problem (two
  unknowns); only the position and width need iteration (a few Gauss–Newton steps from the
  centroid). Weighted least squares or **maximum likelihood** for Poisson counts (weight each pixel
  by 1/its variance) is better at low photon counts.
- **How precise can it get?** Intuition: each detected photon lands at a random place drawn from
  the PSF (width $s$). Estimating the centre from $N$ photons is estimating a mean from $N$ samples:
  standard error $s/\sqrt N$. With $s$ = 100 nm: 1 000 photons → 3 nm; 100 photons → 10 nm. The
  diffraction limit (≈ 250 nm) is about **resolving two** points, not **localising one**
  (`resolution/` recalled). This $s/\sqrt N$ is the best any unbiased estimator can do (the
  **Cramér–Rao bound**, named in the concept layer, formula in the math layer).
- **What makes it worse**: pixels (the photon's position is only known to its pixel: adds
  $a^2/12$ to $s^2$), background photons (they carry no position information but add noise; this
  dominates for dim spots), the wrong model (an Airy PSF fitted with a Gaussian: small bias; an
  out-of-focus or tilted dipole: larger), and **pixel locking** of crude estimators.
  Least-squares fitting has about 16/9 of the bound's variance (≈ 1.33× its sd); maximum
  likelihood reaches the bound.
- Uncertainty as an output: every detection gets a position **and** an error bar (from the
  bound with the fitted $N$ and background, or from the fit's curvature). The tracking and
  localisation-microscopy uses need it.
- Pointer: image libraries call this "Gaussian fitting", "subpixel refinement" or "radial
  symmetry centre" (a fast non-iterative alternative, one line).

Math layer:

- *Centroid bias*: window of half-width $m$, spot amplitude sum $S$ with centre $\delta$,
  background $b$ per pixel over $(2m+1)$ pixels: $\hat x = S\delta/(S + (2m+1)b)$ (1D): shrinks
  towards 0.
- *Three-point Gaussian*: $\delta = \dfrac{\ln c_{-1} - \ln c_{+1}}{2(\ln c_{-1} - 2\ln c_0 + \ln
  c_{+1})}$; exact for a sampled Gaussian (point-sampled; with pixel integration, nearly exact).
- *Least squares*: minimise $\sum_k (I_k - B - A\,g_k(x_0, y_0, \sigma))^2$; normal equations for
  $(A, B)$ given the rest; Gauss–Newton update $\Delta\boldsymbol\theta = (J^\top J)^{-1}J^\top\mathbf r$
  with the Jacobian of the model; Levenberg–Marquardt damping in one line. Pixel-integrated
  Gaussian model with error functions (one line). Poisson maximum likelihood: minimise $\sum_k
  (\mu_k - I_k\ln\mu_k)$ (recall the Poisson log-likelihood of `richardson-lucy/`).
- *Cramér–Rao*: for $N$ i.i.d. positions from $\mathcal N(x_0, s^2)$, Fisher information $N/s^2$,
  so $\mathrm{Var}(\hat x_0) \ge s^2/N$. With pixels of size $a$ and background $b$ photons/pixel
  (Thompson, Larson & Webb 2002): $\langle(\Delta x)^2\rangle \approx \frac{s^2 + a^2/12}{N} +
  \frac{8\pi s^4 b}{a^2N^2}$ (background variance $b$ for Poisson); Mortensen et al. (2010) correct
  the least-squares constant (×16/9) and show maximum likelihood reaches the bound.

**Demo `spotfit`** (new, main demo; absorbs image formation's planned `localise`). Three tabs:
- *one spot*: a 9×9 pixel grid with one simulated spot (pixel-integrated Gaussian or Airy PSF,
  photon count $N$, background $b$, pixel size, seeded Poisson noise); the true centre is a
  draggable cross within the central pixel. Overlays: estimates of centroid (window size
  slider), three-point parabola, three-point log-parabola, least-squares Gaussian, Poisson
  maximum likelihood; the residual image of the fit; a step-through of Gauss–Newton iterations
  (button *step*, the model contour moving onto the spot).
- *bias*: sweep the true offset 0…1 px; estimated − true for each estimator (noise-free, then
  averaged with noise): centroid with background flattens towards the centre, parabola shows the
  pixel-locking S-curve, log-parabola and the fit are flat.
- *precision*: 500 seeded repeats; scatter of estimates around the truth (with the pixel grid in
  the background) and a log–log plot of sd vs $N$ for each estimator against the Cramér–Rao /
  Thompson curve, current $N$ marked; background and pixel-size sliders move the bound.
Presets: "pixel peak: ±0.5 px", "centroid with background: pulled to the centre", "parabola:
pixel locking", "log-parabola is exact for a Gaussian", "fit all pixels", "1 000 photons: a few
nm", "background dominates dim spots", "pixel size: too big and too small", "Airy spot, Gaussian
model: small bias", "maximum likelihood vs least squares at 50 photons".

By hand (concept): (a) the worked example: samples $(0.430, 0.956, 0.783)$ at $-1, 0, 1$: centroid
(0.16), parabola (0.25), log-parabola (0.30); with 0.5 added to each, centroid (0.10) and
log-parabola (biased now: subtract the background first); (b) precision for $s$ = 100 nm and
N = 100, 400, 2 500 photons (10, 5, 2 nm); (c) with 100 nm pixels: $s_a^2 = s^2 + a^2/12$, how much
worse? (≈ 4 %); with 400 nm pixels? (d) predict: doubling the exposure (photons ×2) vs halving
the background: which helps a dim spot more? (e) a centroid window of 3 vs 9 pixels with noise:
which is noisier, and why? *(math)*: (f) derive the three-point Gaussian formula; (g) the linear
least-squares solution for $A, B$ given the position; one Gauss–Newton step in 1D for $x_0$ with
given numbers; (h) Fisher information of $N$ Gaussian samples and the bound $s^2/N$; (i) the
centroid bias formula with a constant background.

### 4. `spot-crowd/`: Crowded spots and evaluating a detector

Concept layer:

- **When do two spots become one?** The sum of two Gaussians of width σ has **two maxima only if
  the separation $d > 2\sigma$** (Sparrow for a Gaussian PSF, recalled from `resolution/`). The
  detection filter widens the spots: $\sigma_\text{eff}^2 = \sigma_\text{PSF}^2 +
  \sigma_\text{filter}^2$, so a matched filter (σ_filter = σ_PSF) merges pairs closer than
  $2\sqrt2\,\sigma_\text{PSF}$. **The central trade-off of the day**: more smoothing → fewer false
  spots (unit 2) but more merged neighbours.
- **Density** in one picture: as spots per µm² rise, recall drops (merges) and localisation error
  rises (a neighbour's tail biases the fit) long before the spots "look" crowded. A rule of thumb
  to read off the demo: spacing below ~4 σ_PSF already costs accuracy.
- **Splitting by fitting**: fit one spot; if the residual is much larger than the noise or the
  width comes out too large, fit two (or $n + 1$) and keep the extra spot only if it improves the
  fit by more than a penalty (an extra spot always reduces the residual: the penalty prevents
  fitting noise; same idea as the regularisation weight elsewhere). Iterative "add a spot where
  the residual is largest". Pictures: one-Gaussian fit to a pair (wide, elongated residual), the
  two-Gaussian fit (flat residual).
- Other routes (one slide, pointers): deconvolve first (sharper spots, recall deconvolution; noise
  amplification), spots as a **sparse sum of PSFs** (a regularised inverse problem with a sparsity
  penalty: deconvolution and wavelets days), learned detectors that output a heat map
  (`detection/`); localisation microscopy avoids the problem physically by switching on few
  molecules per frame (optional unit 5).
- **Evaluating a detector**: compare the detected points with ground truth (simulation, or an
  annotated image). **One-to-one matching within a gate radius** $r_g$ (e.g. 2 px or ~σ_PSF):
  each true spot may claim at most one detection and vice versa. Matched = **TP**, unmatched
  detections = **FP**, unmatched truths = **FN**; precision, recall, F1 (the same formulas as for
  pixels and objects: recall `seg-eval/`, `object-eval/`). Points have no overlap, so the
  **distance** replaces IoU as the matching criterion.
- **Greedy vs optimal matching** (one slide; worked example below): taking the closest pair first
  can lose a match; the optimal assignment (most matches, then smallest total distance) fixes it.
  Image libraries call the solver the "Hungarian algorithm" (one line; the math is in
  `object-eval/`).
- **Two kinds of error, reported separately**: **detection** (F1 at the gate) and
  **localisation** (RMSE of the matched pairs' distances, also its bias as a mean offset). The
  gate couples them: too large a gate turns bad localisations into TPs, too small a gate turns
  them into an FP + FN pair. Plot F1 against the gate radius and choose a gate in the flat part.
  (Localisation-microscopy challenges report the "Jaccard index" TP/(TP+FP+FN) and RMSE; the
  evaluation day calls the former matching accuracy.)
- **The threshold sweep**: rank detections by score (filter response or fitted amplitude/SNR);
  as the threshold falls, recall rises and precision falls: the **precision–recall curve** and AP
  (recall `object-eval/`). Choose the threshold at the F1 maximum or at a required precision.
  Report also **F1 vs density** for the method: a method is "good" only up to a density.
- Honest evaluation: simulated truth is exact but may not look like real data (PSF model, noise,
  background); annotated real data are realistic but annotators miss dim spots (recall
  `agreement/` if taken). Pointer to the evaluation day for leakage (do not tune $k$ and the
  gate on the test images).

Math layer:

- *Merging*: $f(x) = g(x - d/2) + g(x + d/2)$; $f''(0) = 2g''(d/2) \propto (d^2/4\sigma^2 - 1)$, so
  the centre is a maximum (one spot) iff $d < 2\sigma$; Gaussians convolve with variances adding.
- *Model selection*: residual sum of squares $\mathrm{RSS}_n$ with $n$ spots, each adding 3
  parameters; keep $n + 1$ if $\mathrm{RSS}_n - \mathrm{RSS}_{n+1} > 3\cdot c\,\sigma_n^2$ (a
  likelihood-ratio / information-criterion rule, $c \approx 2$ for AIC, $\ln(\#\text{pixels})$ for
  BIC). Stated, with the warning that the penalty is a choice.
- *Matching*: assignment over the distance matrix with $D_{ij} = \infty$ beyond the gate;
  maximise the number of matches, break ties by total distance (cost $1 - \epsilon D_{ij}$ form,
  recall `object-eval/`'s linear-assignment slide).
- *Bias of the localisation error under a gate*: the RMSE of matched pairs is truncated at $r_g$,
  so it underestimates the true error when $r_g$ is within ~3× the RMSE (one line, with the
  Rayleigh-distribution picture: 2D errors of sd σ_loc have mean distance $\sigma_\text{loc}\sqrt{\pi/2}$).

**Demo `crowdfit`** (new, small): two Gaussian spots (σ_PSF slider) with a separation slider
$d$ (0–6 σ) and an amplitude ratio slider, seeded noise. Panels: 1D profile through both (with
the matched-filtered profile), the 2D image with candidates (maxima after the filter) and the
fitted spots; the residual image of the one-spot and of the two-spot fit; a bar of RSS₁, RSS₂
and the penalty, with the verdict "1 spot / 2 spots". A small plot: fraction of seeded repeats
split correctly vs $d$, for *maxima only* and *maxima + 2-spot fitting*, with $d = 2\sigma$ and
$2\sqrt2\sigma$ marked. Presets: "d = 3σ: two maxima", "d = 2σ: one maximum", "the filter merges
them earlier", "the fit splits what the maxima cannot", "unequal pair: the dim one is lost", "a
penalty too small: noise gets a spot".

**Demo `spotmatch`** (new, main demo): truth and detections from the `spotdetect` pipeline on a
256×256 synthetic image with a **density slider** (10–400 spots), or a small hand-editable scene
(click to add/move truth and detection points) for the greedy-vs-optimal case. Panels: image with
truth (rings) and detections (dots), matched pairs joined by lines coloured by distance, gate
circles on hover-free click; **F1 vs gate radius**; **precision–recall** curve as the threshold
sweeps (current threshold as a dot, AP and the best-F1 threshold reported); **F1 and RMSE vs
density** (computed once per method over seeded images). Controls: gate radius, threshold $k$,
*greedy / optimal* toggle, detector (Gaussian / LoG / à trous, with or without fitting from unit
3), density. Readouts: TP, FP, FN, precision, recall, F1, Jaccard, RMSE, mean offset. Presets:
"greedy loses a match", "gate too small: good spots counted twice wrong", "gate too large:
everything matches", "PR curve: pick the threshold", "density: recall collapses", "fitting
rescues localisation, not merges", "a biased detector (offset 0.5 px): RMSE vs bias".

By hand (concept): (a) two spots of σ = 1.5 px: do they give two maxima at $d$ = 4 px raw? After
a matched filter? ($2\sigma = 3$: yes; $2\sqrt2\sigma \approx 4.24$: no); (b) the matching
example: truths $T_1 = (0, 0)$, $T_2 = (2.5, 0)$; detections $D_1 = (1, 0)$, $D_2 = (-2, 0)$; gate
2.5. Greedy: $D_1$–$T_1$ (distance 1) first, then nothing: TP 1, FP 1, FN 1, F1 = 0.5. Optimal:
$D_1$–$T_2$ (1.5), $D_2$–$T_1$ (2.0): TP 2, F1 = 1, RMSE = $\sqrt{(1.5^2 + 2^2)/2} \approx 1.77$;
(c) a table of 6 detections ranked by score with TP/FP labels and 5 truths: PR points, F1 at each
threshold, the best threshold; (d) predict: what happens to F1 and RMSE when the gate grows from
1 to 5 px with a detector of localisation sd 1 px? *(math)*: (e) derive the 2σ merging rule and
the effective σ after filtering; (f) write the matching of (b) as an assignment problem with the
gate; (g) the mean distance of 2D Gaussian errors (Rayleigh) and why a gate truncates the RMSE;
(h) when does the 2-spot model pay its penalty, for given RSS values.

### 5. (optional / further reading) `smlm/`: Localisation microscopy — an image from detections

Takes over the localisation-microscopy story of image formation's unbuilt optional
`localisation/` unit (open question 2).

- The idea: if only a few molecules emit at a time (photoswitching, blinking, binding events),
  each is an isolated spot that units 1–3 can detect and fit to ~10–20 nm; collect thousands of
  frames, and **render the list of positions** as an image. Resolution is now set by localisation
  precision **and** by labelling density (Nyquist for labels: you need two localisations per
  resolution length in each direction; recall `resolution/`).
- The pipeline as a chain of this day's units: detect (unit 1–2), fit with uncertainty (unit 3),
  reject crowded/merged spots (width and residual checks, unit 4), correct drift (recall
  `stitching/`'s drift slide and the `drift` demo), merge repeated blinks of one molecule (a
  nearest-neighbour link over consecutive frames: a pointer to the tracking day), render
  (histogram of positions, or a Gaussian per localisation with its own uncertainty).
- Artefacts: too dense per frame (merged spots fitted as one: phantom molecules between two real
  ones), too few frames (a dotted structure), drift (smear), repeated blinks (clusters that are
  one molecule). Evaluation: the challenge metrics (Jaccard and RMSE, unit 4); a resolution
  measure from the data (Fourier ring correlation named in one line).
- 3D in one slide: astigmatic PSF, ellipticity encodes z (pointer).
- **Demo `smlmrender`** (new): a seeded synthetic structure (a ring, two lines 40 nm apart, a
  grid), frames with a chosen number of active emitters per frame (density slider), photons,
  background, drift; play frames → localisations accumulate → render. Toggles: *drift
  correction*, *merge blinks*, *reject wide spots*. Presets: "sparse: a sharp image", "too dense:
  phantoms between the lines", "few frames: dots", "drift smears, correction restores", "blinks
  make clusters". Images: synthetic only (labelled as such).
- By hand: precision for N = 1 000 photons, $s$ = 110 nm, background 10 photons/pixel, 100 nm
  pixels (Thompson formula); number of localisations needed for 20 nm Nyquist resolution on a
  1 µm line; the phantom-molecule position for two equal emitters 150 nm apart fitted as one.

## Shared engineering

Reuse first (see `demos/src/lib/README.md`); nothing here re-implements filters, labelling,
matching or plotting.

- **Filters**: `lib/imageops.ts` (`gaussianBlur`, `gaussianKernel`, `logFilter`, `dogFilter`,
  `correlate2d`, `boxFilter`), `lib/edges.ts` (`normLoGStack`, `localMaxima2D`, `extrema3D`,
  `gaussian1d` for `peaks1d`), `lib/morph.ts` (`dilate` for the maximum filter picture,
  `whiteTopHat`, `hMaxima`, `regionalMaxima`).
- **Noise and images**: `lib/synth.ts` `spots()` (truth list, background, Gaussian/Poisson
  noise), `addPoissonNoise`, `addGaussianNoise`; `lib/random.ts` (`rng`, `poisson`, `gauss`);
  `lib/wavelet.ts` `madSigma`; `lib/stats.ts` (`normCdf`, `normQuantile`, `median`, `mad`);
  `lib/psf.ts` (`gaussianPSF`, `airyPSF`, `airySigma`) for the PSF models in `spotfit`.
- **Sub-pixel baselines**: `lib/registration.ts` `parabolaOffset`, `centroidOffset`,
  `parabolaPeak2d` (the same three-point rules, so `spotfit` and `subpixel` agree).
- **Fitting**: `lib/linalg.ts` (`solve`, `lstsq`, `cholesky`, `inv2`/`inv3`) for Gauss–Newton
  normal equations; `lib/deconv.ts` `poissonLogLik` for the maximum-likelihood objective.
- **Matching and curves**: `lib/eval.ts` (`hungarian`, `prFromScores`, `averagePrecision`,
  `matchingAccuracy`, `pointSetDistances`).
- **Drawing/UI**: `lib/plot.ts` (heat maps, scatter, `line`, `band`, `histogram`, `fn` for the
  normal tail and the Cramér–Rao curve, log axes), `lib/colour.ts` (`diverging` with 0 = mid-grey
  for signed filter responses), `lib/ui.ts` (`slider`, `button`, `checkbox`, `presets`,
  `autoMount`), `lib/gridui.ts` `GridView` for the 9×9 pixel grid of `spotfit` and the 5×5
  by-hand patches. Real-image loading via `loadGrey()` when a licensed spot image exists.
- **New lib files owned by this day** (names checked against `demos/src/lib/`: no collisions):
  - `demos/src/lib/spotdetect.ts` (no DOM): `matchedKernel(σ, zeroMean)`, `filterResponseNoise(w)`
    ($\sqrt{\sum w^2}$), `detectSpots(img, {filter, sigma, radius, k, noise: "mad" | "local" |
    number, anscombe})` → `{x, y, score, snr}[]` (calls `localMaxima2D` / the maximum filter;
    no new local-maximum code), `localNoiseMap(img, background, gain, readNoise)`, `anscombe`,
    `eulerExpectedMaxima(area, σf, u)`, `noiseMaximaCount` (seeded simulation used by
    `falsespots` and the tests), `atrousProduct(img, scales, k)` (see next item).
  - `demos/src/lib/spotfit.ts` (no DOM): `centroid(img, x, y, m, background?)`,
    `gauss3Point(cm, c0, cp)` (log-parabola), `pixelGaussian(x0, y0, σ, A, B, size)` (pixel-
    integrated with erf, Airy option via `lib/psf.ts`), `fitGaussianLSQ(patch, init, {fixSigma,
    weights})` with an iteration record for the step-through, `fitGaussianMLE`, `thompsonVar(s, a,
    b, N)`, `crlbGaussian(s, N)`, `fitTwoSpots`, `selectSpotCount(RSS, params, σn, penalty)`,
    `renderLocalisations` (for `smlmrender`).
  - **Additions to files owned by other days** (through their owners): à trous decomposition and
    reconstruction (`atrous(img, J)`, `atrousRecon`) in `lib/wavelet.ts` (owner wavelets day; the
    wavelets plan listed it but the built file has no à trous; this day adds it if the owner
    agrees, else it lives in `spotdetect.ts` and `wavelet.ts` re-exports it);
    `matchPoints(truth, det, gate, "greedy" | "optimal")` → `{pairs, tp, fp, fn, rmse, bias}` and
    `f1VsGate` in `lib/eval.ts` (owner evaluation day; built on its `hungarian`); options in
    `lib/synth.ts` `spots()` for pixel-integrated spots, a fixed position list, a density mode and
    pairs at given separations (owner foundation; today's `spots()` point-samples the Gaussian
    and enforces a minimum distance of 4σ, which hides exactly the crowding of unit 4).
- **Peak finders already duplicated** (consolidation note for `build-issues.md`, not done here):
  `edges.localMaxima2D`, `boxes.localMaxima`, `registration.findPeaks`, `hough.peaks2`/`peaks3`,
  `morph.regionalMaxima`. This day uses `localMaxima2D` and adds none.
- **Speed** (TypeScript, measured before promising "live"): Gaussian/LoG at 256² is a few ms;
  maxima ≈ 1 ms; à trous with 4 scales ≈ 4 separable passes; fitting 150 spots × 7×7 patch × ≤ 10
  Gauss–Newton iterations ≈ 0.5 M model evaluations (< 20 ms). The `falsespots` statistics
  (16 noise images per σ) and the `spotmatch` density curves are computed once per setting with a
  "computing…" state, then sliders only re-threshold. `spotfit` precision tab: 500 fits of 9×9 in
  < 30 ms. No Wasm expected.
- **Tests** (`demos/test/pointdetection-test.ts`, run by `make test`): matched filter SNR on a
  synthetic spot matches $A\sqrt{\pi\sigma^2}/\sigma_n$ within 5 %; `filterResponseNoise` matches
  the measured sd of filtered white noise; MAD estimate within 5 % on noise with 2 % spots; à trous
  scales sum to the input (< 1e-9); measured false spots per 512² at σ_f = 1.5 reproduce the plan's
  ≈ 100 / 3 / 0.05 at $k$ = 3 / 4 / 5 (seeded; numbers in the exercise solutions come from this
  test) and the Euler formula is within a factor 2; log-parabola exact on a point-sampled Gaussian
  (< 1e-9); worked example numbers (0.16, 0.25, 0.30, 0.10); LSQ fit sd within 1.5× of the Thompson
  bound at N = 1 000; MLE within 1.15×; two Gaussians give one maximum below $2\sigma$ and two above;
  `matchPoints` reproduces the greedy/optimal example (TP 1 vs 2) and equals brute-force
  enumeration on random sets of ≤ 7 points.
- Follow `demos/README.md` and `plan/build-issues.md`: TypeScript, `make typecheck`, no network,
  fixed seeds, presets + reset, keyboard back to reveal.js, ≥ 18 px labels on slides (the
  readouts too; earlier builds slipped to 13–17 px), fits 1150×560 on a slide (`spotdetect`,
  `falsespots`, `spotfit`, `spotmatch`: two panels + controls on slides via `data-presets` and a
  compact layout, all panels on the exercise page with `data-layout="full"`), per-demo CSS
  selectors prefixed `.qb-demo[data-demo="…"]` (demo.css loads late), `lib/plot.ts` for every plot
  (clip ids are fixed; no local `uniqueClip()` copies), static figures as print fallbacks, every
  demo added to `demos.qmd` and the sidebar.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `spot-filters/`: lecture (`peaks1d`, `spotdetect`), maxima and à trous by hand |
| 10:45–12:15 | Unit 2 `spot-threshold/`: lecture (`falsespots`), MAD and false-spot counts by hand |
| 13:15–14:45 | Unit 3 `spot-fit/`: lecture (`spotfit`), the three estimators and the bound by hand |
| 15:00–16:30 | Unit 4 `spot-crowd/`: lecture (`crowdfit`, `spotmatch`), matching by hand; quizzes as wrap-up |
| (optional) | Unit 5 `smlm/` (`smlmrender`) as self-study or an extra session |

## Open questions

1. **Entry point**: the course map says "after Edges, blobs and ridges (day 2)". Units 2 and 3
   lean on `cameranoise/` (day 3) and `psf/`/`resolution/` (day 4); the plan makes those recalls
   self-contained, but the day is better after day 4. Proposal: keep day 2 as the formal entry,
   add "recommended after day 4" to the later days table.
2. **Ownership changes** (to add to `plan/coursemap.md`, "Ownership changes", when accepted):
   (a) à trous spot detection moves from wavelets' unbuilt optional `cwt/` to unit 1 here; `cwt/`
   keeps the transform and the scalogram and recalls spot detection; (b) image formation's unbuilt
   optional `localisation/` is replaced by this day's unit 3 (fitting, precision) and optional
   unit 5 `smlm/`; image formation keeps one "localise vs resolve" slide with a pointer; the
   `localise` demo becomes `spotfit`'s *precision* tab; (c) the owners table gains a row for spot
   detection (see overlaps); (d) `object-eval/` keeps matching; point matching by distance is
   taught here.
3. **Matching before day 37**: is a two-slide self-contained matching story in unit 4 acceptable
   (the owner rule allows a recall slide, not teaching)? Proposal: yes, because the day can
   come before day 37; the math stays in `object-eval/`.
4. **Unit 4 density**: crowded spots and evaluation in one unit is full. Fallbacks in order:
   move model selection (RSS penalty) to the math layer only; drop F1-vs-density from the lecture
   (keep it in the demo); or split into `spot-crowd/` + `spot-eval/` (the day then has five core
   units; split rather than cut, principle 6).
5. **3D spots** (z-stacks, anisotropic PSF σ_z ≈ 2–3 σ_xy): one slide in unit 1 and a `data-dim="3"`
   option of `spotdetect` (three MIPs, no WebGL), or leave to the tracking day? Proposal: one slide
   only.
6. **Real data**: a licence-compatible real spot image (smFISH or single-molecule, with or without
   annotation) for `spotdetect` and `spotmatch`; the same need is listed for the edges and wavelets
   days in `plan/build-issues.md` ("real spot images"), so one search for all three. Until then all
   images are synthetic and labelled as such.
7. **Topic page "Point / spot detection"** (coursemap, first requested topic page): this day is its
   backbone; the topic page should link `scale-space/` (`blobscale`), `spot-filters/`,
   `spot-threshold/`, `spot-fit/`, `spot-crowd/`, `stitching/` (`subpixel`), `hough-circles/`,
   `object-eval/`, `detection/`. Build it after this day.
8. **Pending edits elsewhere once accepted** (not done here): `scale-space/lec.qmd` *Where this
   leads* (spot detection → this day, wavelets bullet → "à trous: point-detection day");
   `psf/lec.qmd` "spot detectors (edges day) should be tuned to it" → add "and the point-detection
   day"; `stitching/` "Localisation microscopy and other uses" → pointer to `spot-fit/` / `smlm/`;
   `plan/wavelets-day.md` unit 5, `plan/imageformation-day.md` unit 4, `plan/evaluation-day.md`
   overlaps; the tracking entry in the coursemap's Motion row.
9. **Naming**: "spot" (microscopy) vs "point" (general) in titles. Proposal: day title "Point
   detection", unit titles say "spots", the first slide says "spots, particles, points of light".

## References

- Olivo-Marin (2002), *Extraction of spots in biological images using multiscale products*,
  Pattern Recognition 35:1989–1996 (à trous detector).
- Starck, Murtagh & Bijaoui (1998), *Image Processing and Data Analysis: The Multiscale Approach*,
  CUP (à trous, MAD per scale).
- Smal, Loog, Niessen & Meijering (2010), *Quantitative comparison of spot detection methods in
  fluorescence microscopy*, IEEE TMI 29:282–301.
- Lindeberg (1998), *Feature detection with automatic scale selection*, IJCV (recall).
- Turin (1960), *An introduction to matched filters*, IRE Trans. Inf. Theory.
- Serge, Bertaux, Rigneault & Marguet (2008), *Dynamic multiple-target tracing to probe
  spatiotemporal cartography of cell membranes*, Nature Methods (GLRT detection, multi-target).
- Adler (1981), *The Geometry of Random Fields*, Wiley; Worsley et al. (1996), *A unified statistical
  approach for determining significant signals in images of cerebral activation*, Human Brain
  Mapping (Euler characteristic, expected number of maxima above a threshold).
- Anscombe (1948), *The transformation of Poisson, binomial and negative-binomial data*, Biometrika.
- Cheezum, Walker & Guilford (2001), *Quantitative comparison of algorithms for tracking single
  fluorescent particles*, Biophys. J. (centroid vs Gaussian fit bias).
- Thompson, Larson & Webb (2002), *Precise nanometer localization analysis for individual
  fluorescent probes*, Biophys. J. 82:2775–2783.
- Ober, Ram & Ward (2004), *Localization accuracy in single-molecule microscopy*, Biophys. J.
  (Fisher information, Cramér–Rao).
- Mortensen, Churchman, Spudich & Flyvbjerg (2010), *Optimized localization analysis for
  single-molecule tracking and super-resolution microscopy*, Nature Methods 7:377–381.
- Parthasarathy (2012), *Rapid, accurate particle tracking by calculation of radial symmetry
  centers*, Nature Methods (one-line pointer).
- Holden, Uphoff & Kapanidis (2011), *DAOSTORM: an algorithm for high-density super-resolution
  microscopy*, Nature Methods (multi-emitter fitting).
- Betzig et al. (2006), *Imaging intracellular fluorescent proteins at nanometer resolution*,
  Science; Rust, Bates & Zhuang (2006), *Sub-diffraction-limit imaging by stochastic optical
  reconstruction microscopy*, Nature Methods (localisation microscopy, optional unit).
- Sage et al. (2015), *Quantitative evaluation of software packages for single-molecule
  localization microscopy*, Nature Methods; Sage et al. (2019), *Super-resolution fight club*,
  Nature Methods (Jaccard + RMSE evaluation; teacher background, no tool names in student material).
- Chenouard et al. (2014), *Objective comparison of particle tracking methods*, Nature Methods
  (detection/matching evaluation; bridge to the tracking day).

## Build notes (2026-11-13)

Built: `spot-filters/`, `spot-threshold/`, `spot-fit/`, `spot-crowd/` (lec, quiz, exercise each); demos
`peaks1d`, `spotdetect`, `falsespots`, `spotfit`, `crowdfit`, `spotmatch`; libraries
`demos/src/lib/spotdetect.ts`, `demos/src/lib/spotfit.ts`; test `demos/test/pointdetection-test.ts`
(983 checks); figure script `tools/figs/pointdetection/figs.ts` (8 SVGs in the unit folders).
Local helpers: `demos/src/spotdetect/common.ts` (log slider, tab row, colours; re-exports the edges
day's `canny/imgview.ts` image panel), `demos/src/crowdfit/model.ts` (the crowdfit pipeline without
DOM, used by the demo and the number scripts). Not built: optional unit 5 `smlm/` (`smlmrender`),
the 3D option of `spotdetect`.

Deviations from the plan above:

- **Numbers.** Measured false spots per 512² noise image (σ_f = 1.5, 3 × 3 maxima, 20 seeded
  images): **99.4 / 4.3 / 0.10** at k = 3 / 4 / 5 (plan: ≈ 100 / 3 / 0.05); 4 862 maxima per image;
  Euler formula 123 / 5.0 / 0.07. The decks and exercises quote the measured values.
- **Additions to other owners' libraries not made** (the files belong to other days): the à trous
  transform (`atrous`, `atrousRecon`, 1D versions), `matchPoints`/`f1VsGate` (built on
  `eval.hungarian`, optimal per connected component of the gate graph) and a crowding-capable,
  pixel-integrated spot generator (`spotImage`, no minimum distance) all live in
  `lib/spotdetect.ts`. Candidates for `lib/wavelet.ts`, `lib/eval.ts` and `lib/synth.ts`.
- **Plateaus.** `edges.localMaxima2D` keeps the first pixel of an equal-valued run even when the
  run is a step on a slope; `spotdetect.localMaxima1d` (1D, this day) adds the "plateau must end
  lower" rule. The 2D function was not changed (owner: edges day).
- **spotdetect.** No "raw value" threshold toggle and no real image (open question 6). The plain
  Gaussian and "none" filters subtract the global median (one background level), so the
  uneven-background failure is visible. "Filter σ = 0.5 vs 1.5 vs 4" became two presets (LoG). The
  "window too small" preset uses no filter (with a filter, the window radius hardly matters in 2D:
  checked, F1 0.95–0.97 for r = 1–3 with the zero-mean Gaussian).
- **falsespots.** The Gaussian filter is "Gaussian minus a wide blur (σ = 8)" so that a background
  does not count; the Poisson presets use the LoG (the wide-blur difference leaves a curvature bias
  on the bright patch that looked like Poisson false spots). The bright region is a broad bump,
  not a square patch (a sharp step creates edge responses). The measured curve uses 16 seeded
  noise images (4 at 512²) per filter/σ/size; the local noise estimate is √(blurred image), only in
  Poisson mode.
- **spotfit.** The PSF width is known and fixed in the fits (no σ fit); maximum likelihood by Fisher
  scoring (iteratively reweighted least squares) with Levenberg–Marquardt damping; s = 100 nm,
  pixel size 50–400 nm; RMS-vs-N curve from 120 repeats per N. The Airy option shows no bias of
  the Gaussian fit (symmetry) but a small precision loss (4.6 vs 4.3 nm at 1 000 photons).
- **crowdfit.** Second spot initialised at the largest residual, the first mirrored about the
  one-spot fit; noise sd default 6 (amplitude 60) so that the curves are informative; 12
  separations × 20 seeded pairs.
- **spotmatch.** Detectors: zero-mean Gaussian and LoG (no à trous, whose per-level threshold does
  not fit the single k slider); density up to 1 000 spots (at 400 the recall drop is too small to
  see with σ = 1.5); amplitudes 8–80 so that the PR curve is not trivial; the by-hand tab is
  drag-only (no adding points).
- **Exercises.** The 5 × 5 maxima patch of the exercise differs from the deck's example (the deck
  shows its answer). Pairs in `peaks1d` are 7 px apart (5 px merged already with the matched
  filter in noise).
- **Pending edits elsewhere** (open question 8) and the coursemap rows (open question 2) are not
  done here: the integrator adds the day to the sidebar, gallery and `make test`.
