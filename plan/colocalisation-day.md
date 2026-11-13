# Plan: a day on colocalisation

Internal working doc (plan/ is not rendered). Status: plan, 2026-11-13; not built, except
unit 1 `coloc/` (built for the measurements day, taken over and adapted here).
**Course map (2026-11-13, self-study restructure):** day 44 of 51, Part VIII "Colour and fluorescence". Hard prerequisites: Colour (day 42). Soft (see also): Image formation (day 4), Labelling (day 14), Measurements (day 16), FFT registration (day 23), Spectral unmixing (day 43). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VIII (with FRET and
FLIM, both still unplanned), entry after Colour (day 42). This is the group's first day; FRET
and FLIM build on it (bleed-through, two-channel forward model). **Ownership change applied by
this plan** (coursemap, "Ownership changes"): this day takes over `coloc/` (folder, quiz,
exercise, demo `coloc`) from the measurements day (day 16), which keeps a one-slide recall.

## Goal of the day

After the day a student can:

1. read the **scatter plot** (2D histogram) of two channels and say what colocalised,
   independent, excluding and nested structures look like in it; explain why "white in the
   merge" is not evidence (display range, eye, background);
2. compute **Pearson's r**, the **overlap coefficient** (with k1, k2), **Manders' M1 and M2**
   with thresholds, and say which of them change under an offset, a gain, noise, the choice of
   region, and the threshold; *(math)* derive their invariances;
3. explain **Costes' automatic threshold** (walk down the orthogonal regression line until the
   pixels below are uncorrelated) and when it is undefined;
4. test whether a coefficient is **more than chance**: shift curves (r against a shift of one
   channel), **block randomisation** with blocks about the size of the PSF, a permutation
   p-value; say why shuffling single pixels gives false significance; compute **Li's ICQ** and
   read the product-of-differences image; *(math)* the effective number of independent pixels
   and the null spread of r;
5. do **object-based** colocalisation: detect spots/objects in both channels, compute the
   overlap fraction, the centroid-in-object fraction and nearest-neighbour **distances between
   channels**; compare with the **chance level** for randomly placed objects
   ($1 - e^{-\rho\pi d^2}$) and with a randomisation envelope; explain why centroids can measure
   distances **below the resolution limit** while pixel coefficients cannot, and the bias of a
   measured distance; *(math)* derive the chance level and the distance bias;
6. name the **confounds** and their fixes, in order: background/offset (subtract), bleed-through
   (compensate, recall the unmixing crosstalk unit), chromatic shift (register with beads,
   recall FFT registration), resolution (the PSF: "colocalised" means "closer than about
   0.7 FWHM", never "in contact"); predict the size of each effect with the Gaussian-spot
   formulas *(math)*.

Why it fits this audience: every pixel coefficient is a correlation or a ratio of sums they
know from statistics; the tests are permutation tests (simulation, no distribution theory
needed); the object part is nearest-neighbour geometry; the confounds are the forward model of
image formation (PSF, shift, mixing matrix, offset, noise) run on two channels at once. The new
idea is that **pixels are not independent samples**, which is what makes naive tests wrong.

## Prerequisite gap

The course map lets the group start after day 42, but units 3–4 lean on later days. What
each unit needs, and how it copes if the student comes straight from day 42:

- **Colour** (day 42, `colour-display/`, slide "Fluorescence: channels are grey images"): the
  green/magenta merge and per-channel display ranges. Unit 1's first slide recalls it.
- **PCA** (day 1, `pca/`): the orthogonal regression line is the first principal component
  (already used by the built `coloc/` math slide). One-line recall.
- **Thresholding** (day 12, `histogram/`, `otsu/`): a threshold on a 1D histogram; Costes is
  "a threshold chosen on a 2D histogram". One-line recall.
- **Image formation** (day 4, `psf/`, `resolution/`, demos `psfotf`, `resolve`): PSF, Airy
  pattern, FWHM ≈ 0.51 λ/NA, Rayleigh 0.61 λ/NA; `lib/psf.ts` `airySigma` (σ ≈ 0.21 λ/NA).
  Unit 4 recalls it in one slide with the numbers (NA 1.4, 520 nm: FWHM ≈ 190 nm).
- **Camera noise** (day 3, `cameranoise/`): shot noise, offset, saturation. One line in unit 4.
- **Morphology** (day 13, `morph-grey/`): background subtraction by top-hat. One line in unit 4.
- **Labelling** (day 14, `label-connectivity/`): connected components. Unit 3 needs it to turn a
  thresholded channel into objects; a student who has not done day 14 gets a one-slide recall
  ("one label per connected blob") — enough for the concept layer.
- **Measurements** (day 16, `relations/`): parent–child by overlap or centroid, nearest-neighbour
  distances, Clark–Evans and the Poisson nearest-neighbour law (math slide "The Poisson nearest
  neighbour"). Unit 3 is the **two-channel** version of these and recalls them; it re-states
  the law in its own math slide for students who skipped day 16 (it is two lines).
- **FFT registration** (day 23, `matching/`, `phasecorr/`, `stitching/` demo `subpixel`):
  unit 2's **shift curve** is brute-force cross-correlation (the "slide and score" of
  `matching/`), so the idea of finding a shift by maximising a score is taught here in its own
  terms; unit 4's bead registration recalls phase correlation and sub-pixel peaks in one slide.
- **Spectral unmixing** (`crosstalk/`, demo `bleedthrough`): owns bleed-through and
  2×2 compensation. Unit 4 recalls it in one concept slide + one math slide (the 2×2 inverse),
  self-contained for students who did not take that day.
- **Evaluation and statistics** (day 37, `object-eval/`, `replicates/`): one-to-one matching
  (Hungarian) and the statistics of measurements across cells/replicates. Unit 3 uses matching
  as a one-line recall (concept: "each spot may be used once"); unit 2 ends with a forward
  pointer: a per-image p-value is not a comparison between conditions.

**Proposal (open question 1):** move the group's entry point to "Measurements (day 16)" in the
course map (FFT registration helps). Colocalisation now leaves the measurements day, so a student who skips this day
meets it only as the recall slide in `relations/`; that is acceptable for a specialised topic.

## Overlaps / who owns what

This day **owns** colocalisation: pixel coefficients (Pearson, overlap, k1/k2, Manders M1/M2,
Spearman in one line, ICQ/PDM), thresholds for them (Costes), significance tests for them
(shift curve, block randomisation, Costes' test), object-based colocalisation (overlap,
centroid-in-object, cross-channel nearest-neighbour distances, chance level, cross-K), and the
**catalogue of colocalisation confounds**. Others recall:

| Day / unit | Owns, recalled here | What this day adds / takes |
|---|---|---|
| Measurements (day 16): `coloc/` | — (moves here) | The whole unit, adapted as unit 1. Day 16 keeps **one recall slide** in `relations/` and pointer lines in `intensity-features/`, `texture-features/`, `relations/` (edits listed below) |
| Measurements: `relations/` | Parent–child by overlap/centroid, nearest neighbours, Poisson NN law, Clark–Evans | Unit 3 does the same between **two channels**, adds the chance level and randomisation envelopes |
| Measurements: `texture-features/` | GLCM as a 2D histogram | Unit 1's scatter plot: "same object, second channel instead of shifted copy" (kept from built `coloc/`) |
| Image formation: `psf/`, `resolution/` | PSF, FWHM, Rayleigh/Sparrow, wavelength dependence | Unit 4: what the PSF does to coefficients (Gaussian-spot formulas), "colocalised = closer than ~0.7 FWHM" |
| Unmixing: `crosstalk/` | Bleed-through, single-stained controls, 2×2 compensation, noise amplification | Unit 4 recalls; its slide "Bleed-through fakes colocalisation" points here (pending edit) |
| FFT registration: `phasecorr/`, `stitching/` | Phase correlation, sub-pixel peaks | Unit 4: chromatic shift measured on two-colour beads; unit 2's shift curve is the brute-force version |
| Thresholding: `otsu/` | 1D thresholds | Costes as a 2D-histogram threshold (one line) |
| Morphology: `morph-grey/` | Top-hat background subtraction | Unit 4 one line |
| Evaluation: `object-eval/`, `replicates/` | Hungarian matching, unit of replication | Unit 3 recall; unit 2 forward pointer |
| Planned point-detection day | Spot detection, sub-pixel localisation by fitting, localisation precision | Unit 3 uses LoG + local maxima + weighted centroid with a one-slide recall/pointer; the optional `coloc-points/` uses point lists |
| FRET / FLIM (same group, unplanned) | Proximity below ~10 nm by energy transfer | Forward pointer from units 3–4: "colocalisation stops at the resolution; FRET measures 1–10 nm". FRET reuses `lib/twochannel.ts` and the bleed-through recall |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus one optional unit. Order:
coefficients (unit 1, built) → is it real? (unit 2) → objects instead of pixels (unit 3) →
what the microscope does to all of it (unit 4, the synthesis).

### 1. `coloc/`: Colocalisation of two channels: scatter plot, Pearson and Manders (taken over, adapted)

Built for day 16; keep the deck, quiz, exercise and demo `coloc`, with these changes:

- Notes: "Fourth unit of the measurements day" → "First unit of the colocalisation day
  (Part VIII)"; drop "A planned fluorescence-quantification
  day will take colocalisation further"; "the parent–child relation of the last unit" →
  "the parent–child relation of the measurements day (`relations/`)".
- Slide "Controls" moves to unit 2 (it becomes that unit's opening); slide "Pitfalls" shrinks
  to a two-line preview ("background, bleed-through, chromatic shift and the resolution limit:
  unit 4"); slide "Objects instead of pixels" moves to unit 3. Keep "Thresholds: Costes'
  method" and its math slide here (the threshold is part of M1/M2); unit 2 adds Costes' *test*.
- Add one concept slide **"Rank and robustness"**: one bright pixel pair can carry $r$; the
  **Spearman** coefficient is Pearson of the ranks (one line); it ignores the shape of the
  intensity relation but not the region or the background.
- Add to the math slide "correlation and overlap": Manders' **k1, k2** split
  ($k_1 = \sum AB/\sum A^2$, $k_2 = \sum AB/\sum B^2$, overlap$^2 = k_1k_2$); $r$ under noise
  $r = 1/\sqrt{1 + \sigma^2/(a^2\operatorname{var}A)}$ (now only in the quiz).
- The catalogue slide stays (it is the day's reference table); add rows for ICQ, shift-curve
  width, object fraction at distance $d$ with chance level, nearest-neighbour distance
  (pointing to units 2–3).

Concept layer (as built): merge picture → scatter plot → Pearson (region, noise, bleed-through)
→ M1/M2 asymmetric, need thresholds → Costes walks down the regression line → summary.
Math layer (as built + the additions above): $r$ and overlap as cosines of centred/uncentred
vectors, invariances, M1/M2 formulas and the variant without the own-channel threshold, the
orthogonal regression slope as the top eigenvector.

**Demo `coloc`** (exists; adapt): imports move from `lib/neighbours.ts` to `lib/coloc.ts`, the
scene generator from `coloc/model.ts` to `lib/twochannel.ts` (seed-identical: the numbers in
the deck and exercise must not change; regression test). Add a *Spearman* line to the readout
and a *Manders definition* toggle (with / without the own-channel threshold in the
denominator). Fix per `build-issues.md`: per-demo CSS selectors prefixed
`.qb-demo[data-demo="coloc"]` (the non-slide rules currently lack the prefix); on slides
buttons are 14 px, readout 17 px and figcaption 16 px → all ≥ 18 px (drop the side histogram on
slides if it no longer fits 1150×560; it stays on the page).

**Demo `colocgrid`** (new, small; bridge from the by-hand 3×3 to the image): two editable grids
A and B (6×6, `GridView` from `lib/gridui.ts`, plus/minus editing with drag painting), the
scatter plot of the 36 pairs next to them (`lib/plot.ts` scatter), two threshold sliders drawn
as lines on the scatter plot, and a live readout: $r$, Spearman, overlap, k1, k2, M1, M2, with
the pixels counted in M1/M2 outlined on the grids. Presets: "exercise B1" (the 3×3 values in
the top-left, the rest background 0), "add an offset to B" (B3), "A inside B" (B4), "one bright
outlier carries r", "anti-correlated: r < 0", "no thresholds: M1 = M2 = 1".

By hand (concept, as built): B1–B4 (Pearson, overlap, M1/M2 on 3×3; offset and gain; nested
structures), plus new: **B5′** Spearman of the B1 channels (ranks given with ties averaged) and
the effect of changing the 9 into 90; **B6′** k1, k2 for B1 and their product. Built B5
(three ways to fake colocalisation) moves to unit 4. *(math)*: B6–B8 as built (invariances,
B = aA gives M1 = M2 = 1 on matched thresholds, orthogonal regression slope), plus **B9**
*(math)* overlap$^2 = k_1k_2$.

Explore: built C1, C3, C4, C5, C8 stay; C2's randomisation part and C7 (shift) move to unit 2,
C6 (bleed-through) to unit 4; add two `colocgrid` questions before the `coloc` ones.

### 2. `coloc-null/`: Is it more than chance? Shift curves, randomisation and the ICQ

Concept layer:

- Opening (from `coloc/`'s "Controls" slide): a coefficient means nothing without a reference
  value for "no colocalisation". For independent channels $r$ is not exactly 0 and M1/M2 sit at
  about the area fraction of the other channel (built demo: ≈ 0.2).
- **Shift curve** (van Steensel 1996): shift B against A by $\Delta x = -20 \dots 20$ px and plot
  $r(\Delta x)$. Colocalised small structures: a narrow peak at 0, as wide as the structures
  plus the PSF. Large-scale correlation (cell outline, background): a broad hump that survives
  big shifts. A peak **off zero** is a shift between the channels (forward pointer to unit 4's
  chromatic shift). This is "slide and score" from `matching/` (day 23) with Pearson as score.
- **Randomisation**: destroy the spatial relation but keep each channel's own statistics.
  Shuffle **blocks** of B (Costes et al. 2004), recompute $r$, many times; p-value = share of
  shuffles at least as large as the observed $r$.
- **Why blocks, not pixels**: neighbouring pixels are not independent (the PSF spreads every
  point over a few pixels). Shuffling single pixels produces a null distribution that is far
  too narrow, so almost any $r$ looks "significant". Blocks about the PSF FWHM (or the structure
  size) keep the correlations within a channel. Picture: the two null histograms side by side.
- **What the test does not protect against**: bleed-through and a shared uneven background are
  real overlap *in the images*; randomisation calls them significant (built C6). The test asks
  "more than chance for these images", not "same molecules".
- **Li's ICQ** (Li et al. 2004): per pixel the **product of the differences from the mean**
  (PDM) $(A - \bar A)(B - \bar B)$, shown as a signed image: positive where both are above or both
  below their means (dependent), negative where one is up and one down (segregated). ICQ =
  share of positive PDM pixels − 0.5, in $[-0.5, 0.5]$: ≈ 0 random, > 0 dependent, < 0
  segregated. The PDM image says **where** the correlation comes from, which $r$ cannot.
- **Costes' threshold revisited**: the stopping rule "pixels below have $r \le 0$" is itself a
  null idea: below the thresholds is noise. Failure: weak correlation or a big shared
  background makes the walk stop too low or never start (one picture of the trace).
- Where this leads: a p-value per image is not a comparison of conditions; comparing
  coefficients between treatments needs replicates (day 37, `replicates/`).

Math layer:

- *Permutation p-value*: $p = (1 + \#\{r^* \ge r_\text{obs}\})/(N + 1)$; why the +1; the smallest
  reachable p is $1/(N+1)$ (with 50 shuffles: 0.02, as in the built quiz Q6).
- *Null spread of r*: for $n$ independent pixel pairs, $E[r] = 0$, $\operatorname{var} r \approx
  1/(n-1)$. With a PSF of FWHM $w$ px the image has roughly $n_\text{eff} \approx n/w^2$
  independent samples (order of magnitude; the exact factor depends on the PSF shape), so the
  honest null sd is about $w$ times larger. Numbers: 128² pixels, $w = 3$: sd 0.008 (pixel
  shuffle) vs ≈ 0.023 (blocks).
- *Shift curve of a Gaussian structure*: two equal Gaussian spots of width $\sigma$ shifted by
  $\delta$ have normalised overlap $e^{-\delta^2/(4\sigma^2)}$, so the peak half-width is
  $2\sigma\sqrt{\ln 2} \approx 0.71\,$FWHM; derivation in one line (product of Gaussians).
- *ICQ and Pearson*: $r = \sum \text{PDM} / (n\,s_As_B)$ (sum of PDM values, scaled), ICQ counts
  their signs; under independence each sign is + with probability ½, so $N_+ \sim
  \operatorname{Bin}(N, \tfrac12)$ and a sign test gives a p-value — with the same caveat about
  dependent pixels.
- *Sub-pixel peak of the shift curve*: parabola through the three values around the maximum
  (recall `stitching/`, `parabolaOffset`).

**Demo `colocnull`** (new): the `coloc` scene (via `lib/twochannel.ts`, 128×128). Tabs (one
question each): **shift curve** — $r(\Delta x)$ and $r(\Delta y)$ for −20…20 px, PSF FWHM marked,
peak position with sub-pixel parabola readout; **randomisation** — block-size slider (1, 2, 4,
8, 16 px), *run 200 shuffles* (seeded), histogram of $r^*$ with the observed $r$ as a vertical
line, p-value readout, and the "1-px shuffle" null overlaid in grey for comparison; **PDM/ICQ**
— the signed PDM image (diverging map, 0 = mid-grey), ICQ, sign-test p, and the merge for
orientation; **Costes trace** — $r_\text{below}$ against $t_A$ (from the `trace` that
`costes()` already records), with the stop marked. Sliders shared across tabs: shared fraction
$f$, structure size, blur (PSF σ), shift of B, bleed-through, shared uneven background, noise.
Presets: "colocalised: narrow peak at 0", "large-scale correlation: broad hump", "pixel shuffle:
everything is significant", "blocks the size of the PSF: honest null", "bleed-through passes
the test", "a shifted channel: peak at +2.4 px", "segregated: ICQ < 0", "weak correlation:
Costes stops too low". Timing: one shift curve = 41 Pearson evaluations of 16 k pixels per axis
(< 5 ms); 200 block shuffles ≈ 20–40 ms on a button press (not live; fine).

By hand (concept): (a) 19 given shuffled $r^*$ values and an observed $r$: the p-value, and the
smallest p possible; (b) PDM signs and ICQ for the B1 channels of unit 1 (means given); (c) a
shift curve given at −3…3 px: where is the peak, how wide, is the correlation from small
structures or from the cell outline? (d) predict: block size 1 px vs 8 px for spots blurred to
FWHM 3 px — which null is wider and which test is honest? (e) bleed-through 30 %: what does the
randomisation test say, and why is that not wrong? (f) a Costes trace table ($t_A$, $r_\text{below}$):
pick the threshold. *(math)*: (g) show $r = \sum\text{PDM}/(n s_A s_B)$ (with the $1/n$
convention stated); (h) the p-value formula and the minimum p for $N$ shuffles; (i) the
Gaussian overlap $e^{-\delta^2/4\sigma^2}$ and the half-width of the shift curve; (j) sub-pixel
peak from three values by a parabola.

### 3. `coloc-objects/`: Objects instead of pixels: overlap, distances and the chance level

Concept layer:

- Pixel coefficients mix all structures into one number and cannot say **which** structures
  overlap or **how far apart** they are. Object-based colocalisation asks per object.
- Recall (one slide): spots → objects. Detect spots (LoG + local maxima above a noise-based
  threshold, the edges day / point-detection pointer) or threshold + label larger objects (days
  6, 9). Locate each spot by its intensity-weighted centroid. Every later number inherits the
  detection errors (missed and false spots).
- **Overlap measures** (recall `relations/` parent–child): share of A objects that overlap a B
  object (by any pixel, or by > 50 % of their area), share whose **centroid** lies in a B object.
  Different definitions, different answers for large objects; identical for small spots.
- **Distance measures**: for each A spot, the distance to the nearest B spot; the histogram of
  these distances; "colocalised" = nearest B within $d_\text{max}$. One-to-one pairing (each B
  used once; recall matching on the evaluation day) vs nearest neighbour (one B can serve many
  A).
- **The chance level**: with many B spots, every A spot has a B neighbour nearby by chance.
  For randomly placed B spots at density $\rho_B$, the chance that one lies within $d$ is
  $1 - e^{-\rho_B\pi d^2}$ (sparse: 2.5 %; dense: 22 % at the same $d$). Report the observed
  fraction **next to** the chance fraction, or the corrected
  $(f_\text{obs} - f_\text{chance})/(1 - f_\text{chance})$.
- **Randomisation for objects**: move the B spots to random positions **inside the cell mask**
  (not the whole image), recompute; repeat → an envelope. Ripley's cross-K/L function in one
  slide: the count of B within $r$ of A, against $r$, compared with the envelope: above it =
  attraction, below = exclusion, at every distance at once.
- **Below the resolution**: two spots 60 nm apart blur into one diffraction spot, so pixel
  coefficients call them "colocalised". Their **centroids** are still 60 nm apart, and with
  enough photons and registered channels that distance can be measured. The price: each
  centroid has an error (localisation precision), so measured distances are **biased upward**:
  true 0 nm with 20 nm precision per channel reads ≈ 35 nm on average.
- Where this leads: FRET (same group) for 1–10 nm; localisation microscopy (optional
  `coloc-points/`).

Math layer:

- *Chance level*: B spots as a Poisson process of density $\rho_B$; $P(\text{no B within }
  d) = e^{-\rho_B\pi d^2}$, so $P(D \le d) = 1 - e^{-\rho_B\pi d^2}$ and the density
  $2\pi\rho_B d\,e^{-\rho_B\pi d^2}$ (the Poisson nearest-neighbour law of `relations/`, now
  from A to B). Edge effects and the cell mask: why simulation in the mask beats the formula.
- *Cross-K*: $\hat K_{AB}(r) = \frac{|W|}{n_An_B}\sum_{i,j} [d_{ij} \le r]$; independence gives
  $\pi r^2$; $L(r) = \sqrt{K/\pi}$, so $L(r) - r = 0$ under independence.
- *Distance bias*: with independent Gaussian localisation errors of sd $\sigma_A$, $\sigma_B$
  per axis, the measured separation vector is the true one plus an error of variance
  $\sigma_A^2 + \sigma_B^2$ per axis: $E[d_\text{meas}^2] = d^2 + 2(\sigma_A^2 + \sigma_B^2)$; for
  $d = 0$ the measured distance is Rayleigh-distributed with mean $\sqrt{\sigma_A^2+\sigma_B^2}\sqrt{\pi/2}$
  (20 nm each: 35 nm). Localisation precision itself ($\approx s/\sqrt N$ for $N$ photons) is
  owned by the point-detection day (one-line recall).
- *One-to-one pairing* as an assignment problem (the Hungarian algorithm of the evaluation day,
  one line).

**Demo `colocobjects`** (new): a two-channel spot scene from `lib/twochannel.ts` at physical
scale (pixel 65 nm, NA 1.4, λ 520/670 nm, cells as masks): $n_A$ spots, a fraction $p$ of them
with a B partner at true distance $d$ (nm, random direction), extra independent B spots,
photons per spot, chromatic shift. Panels: merged image with detected spots (green/magenta
circles) and the pair links; histogram of A→B nearest distances with the chance density
overlaid; a $d_\text{max}$ slider (drawn on the histogram) with readout: fraction observed,
fraction by chance, corrected fraction, and the pixel Pearson $r$ for contrast; toggle
*nearest neighbour / one-to-one*; tab **cross-L** with the envelope from 39 random relocations
inside the mask (button, seeded); tab **objects** with larger blobs, thresholded and labelled,
comparing overlap-by-any-pixel, overlap > 50 % and centroid-in-object; tab **distances below
the resolution**: histogram of measured pair distances against the true $d$ (vertical line) and
the predicted mean $\sqrt{d^2 + \dots}$. Presets: "pairs at 0 nm", "independent, sparse: chance
2.5 %", "independent, dense: chance 22 %", "pairs at 60 nm: pixels say yes, centroids say 60",
"few photons: the distance bias", "chromatic shift 100 nm moves the histogram", "big objects:
overlap or centroid?", "missed detections lower the fraction". Sizes: ≤ 300 spots, brute-force
distances, Hungarian on ≤ 300×300 only on toggle (measure; fall back to greedy with a note if
> 30 ms).

By hand (concept): (a) coordinates of 5 A spots and 6 B spots: nearest distances, fraction
within 2 px, nearest-neighbour vs one-to-one where two A spots share the same nearest B;
(b) chance fraction at $d = 2$ px for $\rho_B = 0.002$ and $0.02$ px⁻², and the corrected
fraction for an observed 30 %; (c) two small label images: overlap by any pixel, > 50 %, and
centroid-in-object for each A object; (d) predict: true distance 0, precision 20 nm per channel
— average measured distance (given the factor $\sqrt{\pi/2} \approx 1.25$); (e) why random
relocation must stay inside the cells. *(math)*: (f) derive $1 - e^{-\rho\pi d^2}$; (g)
$E[d_\text{meas}^2] = d^2 + 2(\sigma_A^2 + \sigma_B^2)$; (h) $K_{AB}(r) = \pi r^2$ under
independence (expected count in a disc).

### 4. `coloc-confounds/`: What the microscope does to colocalisation

The synthesis unit: one forward model, four confounds, the order of corrections.

Concept layer:

- One picture of the two-channel forward model: true positions → per-channel PSF (width
  ∝ λ) → chromatic shift → bleed-through (A's light in B's channel) → background (offset +
  uneven, shared) → noise. Each step changes the coefficients in a known direction.
- **Background**: a constant offset leaves $r$ alone but breaks the overlap coefficient and
  thresholds; an **uneven background shared by both channels** (illumination, autofluorescence)
  adds correlation like the cell outline did in unit 1. Fix: subtract (top-hat of the
  morphology day, or a measured background), then choose the region.
- **Bleed-through** (recall `crosstalk/`): dye A seen in channel B correlates the channels by
  itself; measure the leak with a single-stained control, compensate (undo the 2×2 matrix), or
  image sequentially. Over-compensation creates **anti**-colocalisation.
- **Chromatic shift**: different colours focus at slightly different places (laterally tens of
  nm, more at the field edge; axially 100–300 nm). On small structures a one-pixel shift costs a
  lot of $r$ (built demo: 0.99 → 0.94 → 0.81 → 0.63 for 1–3 px). Measure it on **two-colour
  beads** (the shift-curve peak of unit 2, or phase correlation with a sub-pixel peak from the
  FFT registration day), then shift one channel back.
- **Resolution**: the PSF makes everything within about one FWHM overlap. Two single spots
  reach $r = 0.5$ at about **0.7 FWHM** apart (≈ 130 nm at NA 1.4, 520 nm). "Colocalised" means
  "closer than the resolution", never "bound" or "in contact"; the red channel's wider PSF alone
  costs little (≈ 0.97 for 488 vs 640 nm). Below resolution: centroids (unit 3) or FRET.
- **Order of corrections**: offset/background → compensation (linear, needs offset-free data,
  and the leaked light sits at A's position) → registration → region → coefficients with
  thresholds → controls (unit 2). Saturated pixels: exclude or the cloud is clipped.
- A **checklist** slide for reporting (region, thresholds and their method, background
  treatment, compensation with controls, registration with beads, PSF/pixel size, the test and
  its block size, object definition and chance level).

Math layer:

- *Bleed-through*: $B' = B + sA$; for independent $A$, $B$ with equal variances,
  $r(A, B') = s/\sqrt{1 + s^2}$ (s = 0.4: 0.37; the noisy built demo shows 0.32).
  Compensation $\hat B = B' - \hat s A$ and the error $(s - \hat s)A$ (recall the 2×2 inverse
  of `crosstalk/`).
- *Shared background*: $A = S_A + G$, $B = S_B + kG$ with independent $S_A, S_B, G$:
  $r = k\operatorname{var}G/\sqrt{(\operatorname{var}S_A + \operatorname{var}G)(\operatorname{var}S_B + k^2\operatorname{var}G)}$.
- *Gaussian spot pairs*: normalised overlap of two 2D Gaussians of width $\sigma$ at distance
  $\delta$: $e^{-\delta^2/4\sigma^2}$ (shift and resolution are the same formula); equal centres,
  widths $\sigma_1, \sigma_2$: $2\sigma_1\sigma_2/(\sigma_1^2 + \sigma_2^2)$; PSF σ ≈ 0.21 λ/NA
  (recall `psf/`).
- *Registration*: the shift that maximises the bead cross-correlation; sub-pixel by parabola or
  upsampled peak (recall `stitching/`); a residual registration error $\epsilon$ acts like an
  extra shift in the Gaussian formula.

**Demo `colocpipeline`** (new): the full forward model of `lib/twochannel.ts` on a 128×128 cell
scene (physical units: pixel 65 nm, NA, two wavelengths) plus a two-colour **bead** image from
the same optics. Left: truth (positions, coloured by "paired / not paired"); middle: the
measured merge; right: the corrected merge. Confound sliders: true pair distance (nm), uneven
shared background, offset, bleed-through $s$, lateral chromatic shift (x, y in nm), photons.
Correction toggles in pipeline order: *subtract background* (top-hat radius), *compensate*
(with an estimated $\hat s$ slider from a single-stained control: under / right / over),
*register* (shift estimated from the beads by phase correlation + sub-pixel peak, shown as a
readout against the true shift). Readout table: $r$, M1/M2 (Costes), ICQ, object fraction at
$d_\text{max}$ (with chance level) for *truth / measured / corrected*. Tab **resolution sweep**:
$r$ and object-based fraction against the true pair distance 0–500 nm (analytic Gaussian curve
+ simulated points), FWHM marked. Presets: "clean", "red PSF is wider: barely matters",
"100 nm chromatic shift on small spots", "register with beads", "30 % bleed-through:
compensate", "over-compensation: anti-colocalisation", "uneven background shared by both",
"everything at once, corrected step by step", "0 nm and 150 nm look the same in pixels".
Timing target < 30 ms per slider move at 128² (one blur per channel, one FFT registration
pair cached per shift setting, top-hat r ≤ 8); the sweep tab is computed on entry. No Wasm
expected; measure first.

Also linked (not duplicated): `bleedthrough` (`crosstalk/`), `resolve` (`resolution/`),
`subpixel` (`stitching/`).

By hand (concept): (a) two 1×6 channels with $s = 0.3$ leakage: $B'$, $r$ before/after
compensation; estimate $s$ from a single-stained control (slope through the origin, recall
`crosstalk/`); (b) FWHM for NA 1.4 at 520 nm, and the normalised overlap of two spots 100 nm
and 300 nm apart; (c) a 1-px chromatic shift at 65 nm pixels on spots of σ = 80 nm; (d) which
coefficients does a shared uneven background inflate, and which does a constant offset break?
(e) put the five corrections in order and justify each position; (f) built B5 (three ways to
fake colocalisation) moves here. *(math)*: (g) $r = s/\sqrt{1+s^2}$; (h) the two Gaussian
overlap formulas; (i) the shared-background $r$.

### 5. (optional) `coloc-points/`: Colocalisation of point lists

For localisation (super-resolution) microscopy, where the data are coordinates with
uncertainties, not images: coordinate-based colocalisation per localisation (Malkusch et al.
2012), cross-K/L/pair-correlation with edge correction, cluster detection in one line, 3D and
the larger axial chromatic shift. Needs the point-detection day or image formation's
optional `localisation/` (both unbuilt). **Demo**: `colocobjects` with `data-mode="points"`
(no image, point lists with precision ellipses).

## Edits to the measurements day (allowed: only these references)

- `relations/lec.qmd`: new slide **"Two channels: colocalisation"** after "Overlap or centroid?"
  (before the catalogue): the merge figure (`../coloc/fig-merge.png`), "white in the merge is
  not evidence", one line each on the scatter plot, Pearson/Manders needing a region and
  thresholds, object-based colocalisation = the parent–child relation of this unit; pointer to
  Part VIII (colour and fluorescence) (`coloc/` …). Line 159 ("the next unit does
  colocalisation pixel by pixel") → "the colocalisation day does it pixel by pixel".
- `relations/exercise.qmd` line 132 ("the next unit pairs pixels of two channels") → "the
  colocalisation day pairs …".
- `texture-features/lec.qmd` lines 34 (notes) and 103: "the colocalisation unit" → "the
  colocalisation day (`coloc/`)".
- `texture-features/exercise.qmd` line 147: "The unit after next replaces …" → "The
  colocalisation day replaces …"; keep "the next unit measures relations".
- `intensity-features/lec.qmd` line 33 (notes): "first of four units … colocalisation between
  channels" → "first of three units … (colocalisation moved to the Fluorescence quantification
  day; recall slide in `relations/`)".

**Integration edits outside this day's files** (for the integrating step, not the builder):
`_quarto.yml` (move the "Colocalisation of two channels" section out of Day 16 into a new
sidebar group "Fluorescence quantification" → "Colocalisation", with the four units);
`demos.qmd` (unit label of `coloc`, new gallery entries); `crosstalk/lec.qmd` line 181
("Colocalisation itself is taught on the measurements day (`coloc/`)" → the colocalisation
day); `plan/measurements-day.md` (goal 5 and unit 4 → recall; timetable);
`plan/coursemap.md` (group row: colocalisation planned; ownership change applied).

## Shared engineering

- **New `demos/src/lib/coloc.ts`** (owner: this day; no DOM): re-exports `pearsonMasked`,
  `overlapCoefficient`, `manders`, `orthoRegression`, `costes`, `CostesResult`, `blockScramble`
  from `lib/neighbours.ts` (physical move there is a second-pass consolidation, open question 3),
  and adds: `mandersNoOwnThreshold`, `kCoefficients`, `spearmanMasked` (average ranks for
  ties), `pdm`, `icq` (+ sign-test p via `normCdf` of `lib/stats.ts`), `shiftCurve(a, b, w, h,
  maxShift, mask, axis)`, `randomisationTest(a, b, w, h, {block, n, seed, mask})` → `{null,
  p}`; object-based: `crossNearest(ax, ay, bx, by)`, `fractionWithin`, `chanceFraction(rhoB,
  d)`, `pairOneToOne` (uses `hungarian` from `lib/eval.ts`, gated by $d_\text{max}$),
  `crossK(…, radii, area)`, `relocateInMask(n, mask, w, h, rng)`, `envelope(statFn, mask, n,
  seed)`; `gaussianOverlap(delta, s1, s2)` (closed forms for units 2 and 4).
- **New `demos/src/lib/twochannel.ts`** (owner: this day; FRET/FLIM extend it; no DOM): the
  two-channel forward model. `pixelScene(opts)` (moved from `coloc/model.ts` `makeChannels`,
  bit-identical for the same seed), `cellMask`, `spotPairScene({nA, pairFraction, distanceNm,
  extraB, mask, seed})` (true coordinates), `render(points, {pixelNm, NA, lambdaNm, shiftNm,
  photons, offset, background, bleed, seed})` with a Gaussian PSF of σ = `airySigma` from
  `lib/psf.ts` and Poisson noise from `lib/random.ts`, `beadImage`, `unevenBackground`,
  corrections `compensate(a, b, s)`, `subtractBackground` (`whiteTopHat` from `lib/morph.ts`),
  `registerShift` (`correlate2d` + `upsampledPeak2d`/`parabolaPeak2d` + `fourierShift2d` from
  `lib/registration.ts`), and the display helper `mergeGreenMagenta(a, b, rangeA, rangeB)` →
  RGBA (DOM-free; candidate for `lib/colour.ts` later).
- **Reused**: `lib/plot.ts` (scatter, heatmap for the 2D histogram, histogram, line, band for
  envelopes, vline), `lib/histogram.ts` `histogram2d`, `lib/gridui.ts` `GridView` (`colocgrid`),
  `lib/stats.ts` (`quantile`, `pearson`, `normCdf`), `lib/random.ts` (`rng`, `gauss`, `poisson`,
  `shuffle`), `lib/imageops.ts` (`gaussianBlur`, `logFilter`), `lib/morph.ts`
  (`regionalMaxima`, `whiteTopHat`), `lib/label.ts` (`label2d`, `centroids`), `lib/features.ts`
  (`weightedCentroid`), `lib/neighbours.ts` (`assignParents`, `poissonNNDensity`),
  `lib/eval.ts` (`hungarian`), `lib/psf.ts` (`airySigma`, `fwhm`), `lib/colour.ts` (diverging
  map for the PDM image, `inferno` for the 2D histogram), `lib/ui.ts`.
- **Demo files**: `demos/src/coloc/` (adapted; `model.ts` shrinks to the demo's readout glue),
  new `demos/src/colocgrid/`, `demos/src/colocnull/`, `demos/src/colocobjects/`,
  `demos/src/colocpipeline/` (no name collisions with existing demo folders).
- **CSS**: every per-demo selector prefixed `.qb-demo[data-demo="<name>"]` (demo.css loads late);
  slide text ≥ 18 px, including buttons and readouts (`build-issues.md`).
- **Tests** (`demos/test/colocalisation-test.ts`, run by `make test`): Pearson of $aA + b$ is 1;
  M1 = M2 = 1 when $A = B$ and on matched thresholds for $B = aA$; overlap$^2 = k_1k_2$; ICQ of
  $A = B$ is 0.5 and of independent noise ≈ 0; `pixelScene` reproduces the built numbers
  (perfect: $r$ = 0.988; independent inside cells: −0.08; whole image with diffuse: 0.51);
  1-px-shuffle null sd < 8-px-block null sd on a blurred scene; shift curve recovers an imposed
  shift within 0.1 px; `chanceFraction` agrees with simulated Poisson points within 2 sd;
  cross-K of independent points ≈ $\pi r^2$; `gaussianOverlap` matches numeric integration;
  `compensate` with the true $s$ recovers B; `registerShift` recovers an imposed sub-pixel shift
  within 0.1 px.
- **Images**: all synthetic and seeded (labelled "synthetic" in captions). A real two-channel
  fluorescence image and a two-colour bead image with a licence that allows redistribution are
  still needed (already on the `build-issues.md` data list for measurements).
- Follow `demos/README.md`: TypeScript, `make typecheck`, no network, fixed seeds, presets +
  reset, `slider`/`button`/`checkbox` helpers (keyboard stays with reveal.js), fits 1150×560 on
  a slide (`colocobjects` and `colocpipeline` show one tab at a time on slides, all panels on the
  exercise page via `data-layout="full"`), static figures as print fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `coloc/`: lecture, coefficients by hand, `colocgrid`, `coloc` |
| 10:45–12:15 | Unit 2 `coloc-null/`: lecture, p-value and ICQ by hand, `colocnull` |
| 13:15–14:45 | Unit 3 `coloc-objects/`: lecture, distances and chance level by hand, `colocobjects` |
| 15:00–16:30 | Unit 4 `coloc-confounds/`: lecture, confound numbers by hand, `colocpipeline`; quizzes as wrap-up |

## Open questions

1. **Entry point**: the course map says "after Colour (day 42)"; units 3–4 lean on labelling
   (day 14), `relations/` (day 16) and FFT registration (day 23). Proposal: entry after
   Measurements (day 16), FFT registration recommended. The units are written to survive with
   recalls either way.
2. **Measurements day after the hand-over**: day 16 drops to three units. Proposal: give the
   freed slot to the long `intensity-features/` deck (build-issues: "decks run long") or to
   exercises; decide in `plan/measurements-day.md`. Students who skip this day see colocalisation only
   in the recall slide; is that enough, given how often two-channel images occur in biology?
   (Alternative: keep `coloc/` core and make this day units 2–4 + optional; rejected here because
   the course map assigns the unit to this day.)
3. **Where the coloc functions live**: they sit in `lib/neighbours.ts` (measurements day). This
   plan re-exports them from `lib/coloc.ts`; moving the code (and pointing `relations`' demo at
   the slimmer `neighbours.ts`) is a second-pass consolidation.
4. **Spearman, Kendall, Manders' "true" M with ROI-based thresholds, rank-weighted
   colocalisation (RWC)**: Spearman gets one slide; the rest are teacher notes only? Proposal:
   yes; RWC as a catalogue row.
5. **How much statistics in unit 2**: the $n_\text{eff} \approx n/w^2$ argument is an order of
   magnitude; exact formulas (effective sample size from the autocorrelation) are left out.
   Agree with the evaluation day's owner that the per-image p-value vs between-condition test
   split is right.
6. **3D colocalisation** (anisotropic voxels, axial chromatic shift): a slide in unit 4
   (proposal) and the optional unit, or more?
7. **Real data**: a two-channel image with a known positive and negative pair, and two-colour
   beads (e.g. multi-colour bead slides from an open data set); licence check.
8. **FRET/FLIM coordination**: confirm `lib/twochannel.ts` as the group's forward-model file
   (FRET adds a third, sensitised-emission channel and cross-excitation) when those days are
   planned.

## References

- Manders, Verbeek & Aten (1993), *Measurement of co-localization of objects in dual-colour
  confocal images*, J. Microscopy.
- Costes et al. (2004), *Automatic and quantitative measurement of protein-protein
  colocalization in live cells*, Biophys. J. (threshold and randomisation test).
- van Steensel et al. (1996), *Partial colocalization of glucocorticoid and mineralocorticoid
  receptors in discrete compartments in nuclei of rat hippocampus neurons*, J. Cell Sci. (shift
  curve / cross-correlation function).
- Li et al. (2004), *A syntaxin 1, Gαo, and N-type calcium channel complex at a presynaptic
  nerve terminal: analysis by quantitative immunocolocalization*, J. Neurosci. (ICQ, PDM).
- Bolte & Cordelières (2006), *A guided tour into subcellular colocalization analysis in light
  microscopy*, J. Microscopy (pixel and object-based methods, confounds).
- Dunn, Kamocka & McDonald (2011), *A practical guide to evaluating colocalization in biological
  microscopy*, Am. J. Physiol. Cell Physiol.
- Adler & Parmryd (2010), *Quantifying colocalization by correlation: the Pearson correlation
  coefficient is superior to the Mander's overlap coefficient*, Cytometry A.
- Aaron, Taylor & Chew (2018), *Image co-localization – co-occurrence versus correlation*,
  J. Cell Sci.
- Lachmanovich et al. (2003), *Co-localization analysis of complex formation among membrane
  proteins by computerized fluorescence microscopy*, J. Microscopy (object-based, distances).
- Lagache et al. (2015), *Statistical analysis of molecule colocalization in bioimaging*,
  Cytometry A (Ripley-based object colocalisation).
- Malkusch et al. (2012), *Coordinate-based colocalization analysis of single-molecule
  localization microscopy data*, Histochem. Cell Biol. (optional unit).
- Ripley (1977), *Modelling spatial patterns*, J. R. Stat. Soc. B.
- Teacher-only naming (no tool names in student material): common implementations are Fiji's
  Coloc 2 and JACoP, and CellProfiler's MeasureColocalization (see the naming table in
  `plan/measurements-day.md`).

## Build notes (2026-11-13): what was built, deviations from this plan

Built: the four core units `coloc/` (adapted), `coloc-null/`, `coloc-objects/`, `coloc-confounds/`
(lec, quiz, exercise each); demos `coloc` (adapted), `colocgrid`, `colocnull`, `colocobjects`,
`colocpipeline`; libraries `lib/coloc.ts`, `lib/twochannel.ts`; test
`demos/test/colocalisation-test.ts` (162 checks); figure script
`tools/figs/colocalisation/figs.ts` (4 SVGs in `coloc-null/`, `coloc-objects/`,
`coloc-confounds/`). Optional `coloc-points/` not built. Measurements-day pointers edited as
listed above (`relations/` got the recall slide "Two channels: colocalisation").

Deviations and findings:

- **Unit 1 numbering**: B5′/B6′ became B5 (Spearman) and B6 (k1, k2); math B7–B9 are the built
  B6–B8, new B10 (overlap² = k1k2). Built B5 (three ways to fake) is B6 of `coloc-confounds/`.
  Explore: D1–D3 on `colocgrid` (`#explore-colocgrid`), E1–E6 on `coloc` (built C1, C2 without
  the randomisation part, C3, C4, C5 + Manders definition, C8). Built C7 (shift) became
  `coloc-null/` E1–E2, built C6 (bleed-through) `coloc-confounds/` E4. Built quiz Q6, Q7 moved to
  `coloc-null/` (Q5, Q2); new Q6 (outlier: Pearson vs Spearman) and Q7 (Manders definitions).
- **Spearman** on the demo scene is low even for perfect colocalisation (0.81 vs Pearson 0.99):
  half the pixels in the cells are background with random ranks. Taught as such.
- **Randomisation within the region**: `blockScrambleMasked` (lib/coloc.ts) permutes the blocks
  that touch the region. Shuffling over the whole image imports background blocks into the
  cells (ICQ null biased to 0.13); keeping edge blocks fixed leaves real correlation in the
  null (centred at 0.3–0.5 for colocalised channels). The `coloc` demo's own "randomise B ×50"
  keeps the built whole-image behaviour.
- **ICQ is not ≈ 0 for independent sparse spots**: 0.060 (sign test p ≈ 3·10⁻²⁰), block-shuffled
  ICQ 0.100 ± 0.021. The unit teaches "compare the ICQ with shuffled data, not with 0", and the
  sign test as wrong twice (p ≠ ½ for skewed channels; dependent pixels).
- **Null spread numbers** are those of the demo (inside the cells, n = 5978): 1-px shuffle sd
  0.013 = 1/√n, 8-px blocks 0.064 (w ≈ 5 px = FWHM of the blurred structures), instead of the
  plan's 128², w = 3 example.
- **Costes failure presets**: "weak correlation" stops *high* (t_A = 99, 2 % of the pixels above
  both), not low; a second preset "shared background" walks to the bottom (75 % "signal").
- **Shift curve**: the Gaussian prediction e^{−δ²/4σ²} with σ = 2.15 px matches the demo
  (0.94, 0.81, 0.63, 0.16 at 1, 2, 3, 6 px).
- **Registration** (`registerShift`) uses plain cross-correlation (p = 0) with an upsampled
  peak: phase correlation was 0.1–0.2 px off on noisy bead images.
- **colocobjects**: spot detection is a local helper `detectSpots` in lib/coloc.ts (LoG + local
  maxima + weighted centroid); the point-detection day's `lib/spotdetect.ts` appeared during
  the build and should replace it later. The predicted mean distance uses the localisation
  errors measured against the known truth (robust sd) and the Rice mean. One-to-one pairing
  runs the Hungarian method per connected component of the "within d_max" graph (exact, a few
  ms), so no greedy fallback was needed. Dense B spots merge (80 of 120 detected), so the
  chance level from the detected density is 15.5 % instead of 22 %; the preset label says
  "≈ 20 %".
- **colocpipeline**: one full readout (Costes + two spot detections) ≈ 35 ms, the table of three
  columns ≈ 110 ms in node; images and r update live and the table 120 ms after the last
  slider move (no Wasm). The resolution sweep (12 distances) is computed on entering the tab
  (≈ 260 ms). Compensation presets subtract the background first (the order taught);
  over-compensation uses unrelated channels (r = −0.47). The plan's preset "0 nm and 150 nm
  look the same in pixels" is not true (r 0.91 vs 0.48); replaced by "0 nm and 50 nm look the
  same" (r 0.91/0.84, object fraction 94/96 %). The demo's Gaussian PSF (σ = 0.21 λ/NA) has
  FWHM 184 nm at 520 nm; the text quotes ≈ 190 nm (0.51 λ/NA) for the real PSF.
- `coloc/model.ts` still exports `makeChannels` (= `pixelScene`), because
  `demos/test/measurements-test.ts` imports it.
- The ideal column of `colocpipeline` uses the same Poisson noise as the measurement, so a clean
  scene reads identically in both columns.
