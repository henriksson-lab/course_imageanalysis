# Plan: a unit on contrast and intensity transforms (Part I, day 3, after `highbit/`)

Internal working doc (plan/ is not rendered). Status: **built 2026-11-13** (see "Build notes" at the end); placement still to be applied by the integrator.
Follows `plan/coursemap.md`, section "Queue, second batch" (**Contrast and intensity
transforms**: a unit, to be placed in Part I or on the thresholding day). Resolves
`plan/review-gaps.md` gap 10 ("point operations, histogram equalisation, contrast") and the
open item in `plan/thresholding-day.md` ("histogram equalisation: one slide in `histogram/`
or in `highbit/`?"). It also covers the "display vs data" half of gap 7 (the unplaced
`integrity/` unit) and takes over its proposed demo name `displayvsdata`.

One unit, folder **`contrast/`** (lec, quiz, exercise with `#explore` sections). Deck title
**"Contrast and intensity transforms"**.

## Placement (decision)

**Part I, day 3 "Bit depth and noise", as the fourth unit: directly after `highbit/`,
before `floatingpoint/`.** The integrator places it in the sidebar.

Why day 3 and not day 6 (before `histogram/`):

1. **It continues the last slides of `highbit/`.** That unit ends with "Showing 12–16 bits on
   an 8-bit screen: windows", the math slide "A LUT generalises this: any function from raw value
   to colour, stored as a table. The window is the linear LUT", and "Colour maps, HDR and tone
   mapping". This unit is exactly the next step: non-linear LUTs, then LUTs chosen by the data.
   On day 6 the window/LUT story would be three days old.
2. **The message belongs where display is introduced.** `highbit/` already says "the window
   changes only the display; measure on the raw values". This unit turns that sentence into a
   table of what each transform keeps and destroys, *before* the students measure anything
   (thresholds on day 6, measurements on day 11) and before `denoise/` (day 5) asks the same
   question for restored images (it can then recall this unit's table).
3. **Day 6 is already about segmentation and has larger decks** (`histogram/` 15, `otsu/` 19,
   `local-threshold/` 17, `seg-eval/` 15 slide headings); day 3's decks are smaller
   (`quantisation/` 9, `cameranoise/` 11, `highbit/` 15, `floatingpoint/` 13), so a fifth unit
   fits there with less strain (timetable below).
4. **Earlier users benefit**: the log display of spectra (`fourier/`, recalled), gamma before
   `rgb-linear/` (day 7) and `colour-display/`, CLAHE as a preprocessing step that pixel
   classifiers and point detection mention, equalised patches in `texture-features/`.

The cost: equalisation needs the **cumulative histogram**, which `histogram/` (day 6, owner of
histograms) introduces today. This unit therefore introduces the cumulative histogram *as a
mapping* (one concept slide), and `histogram/` recalls it in one line and keeps the percentile
threshold. See "Proposed reorganisation". The histogram itself is used only as "counts per
value", which `highbit/` (the `window16` histogram) and `cameranoise/` already use.

Fallback if day 3 is judged too full: day 6 as the second unit, directly **after**
`histogram/` (not before it: before it, this unit would have to teach the histogram, which
`histogram/` owns). Everything below works there too, with the cumulative histogram recalled
instead of introduced.

## Goal of the unit

After the unit a student can:

1. define a **point operation** (the new value depends only on the old value, not on the
   position or the neighbours), implement it as a **lookup table** (one entry per possible
   value: 256 for 8 bits, 4096 for 12 bits), and recognise the window of `highbit/`, the
   negative, contrast stretching, clipping and a threshold as point operations;
2. read a **transfer curve**: where its slope is > 1 it stretches contrast (and noise), where it
   is < 1 it compresses; predict what a curve does to the histogram;
3. use **gamma** ($v^\gamma$) and **log** transforms, say which one brightens the darks, and why
   the log turns equal ratios into equal steps (spectra, high dynamic range);
4. explain why a point operation on **8-bit data** leaves gaps (a "comb") in the histogram and
   cannot create levels, and why contrast work is done on high-bit or float data, display last;
5. **equalise** a histogram by hand: the **cumulative histogram is the mapping**; explain why
   the result of a discrete image is only **approximately flat** (whole levels move together;
   levels can merge, never split);
6. explain why global equalisation fails (one big background peak → noise amplified; small
   regions ignored) and how **CLAHE** fixes it: **local** mappings per tile, **interpolated**
   between tiles, with the histogram **clipped** to cap the contrast gain;
7. **match** one image's histogram to another's (or to a target shape), and say what that
   erases;
8. state for each transform what it does to **measurements** (order, differences, ratios,
   means, comparisons within an image and between images) and apply the rule: **transform for
   looking and for detection; measure on the raw (linear) data; never measure on an equalised
   image**;
9. *(math)* derive the effect of a point operation on the histogram (change of variables), show
   that $F(X)$ is uniform for a continuous $X$ (probability integral transform) and why the
   discrete version is not, write matching as $G^{-1}\circ F$, bound CLAHE's slope by the clip
   limit, and show with Jensen's inequality why the transform of a mean is not the mean of the
   transform.

Why it fits this audience: a point operation is a function of one variable; its slope, its
inverse and its effect on a density are first-year calculus and probability. The equalisation
mapping is the cumulative distribution function, which they know from statistics. Everything
can be worked by hand on an 8-level image.

## Prerequisite gap

What the built units already teach (checked), and what this unit recalls in one line:

- **`highbit/`** (built, day 3): window/level, auto-contrast by percentiles, hi-lo LUT, "a LUT
  generalises this … the window is the linear LUT" (math slide), "Tone mapping … good for
  looking, bad for measuring" (bullet), and "Displays, LUTs and tone mapping are for looking;
  analysis uses the raw numbers" (summary). This unit **starts from** the window slide and the
  `window16` demo's histogram view; it does not re-teach windows (one recall slide with the
  window drawn as a curve).
- **`quantisation/`** (built, day 3): levels, banding, rounding. Recalled for the comb in the
  histogram after stretching 8-bit data, and "a LUT cannot create levels".
- **`cameranoise/`** (built, day 3): noise models, and the pointer "a variance-stabilising
  transform (e.g. 2√x, the Anscombe transform)". Recalled once: the square root is a gamma of
  ½, and it makes Poisson noise roughly constant; this unit gives the one-line reason in the math
  layer (slope of √x is 1/(2√x)), still a pointer, not a unit on noise.
- **`fourier/`** (built, day 1): spectra shown as $\log(1 + |X|)$. Recalled as the textbook use
  of the log transform.
- **The histogram**: used only as "counts per value" (as in `highbit/` and `cameranoise/`).
  The **cumulative histogram** is introduced here (one concept slide, one math slide), see
  "Proposed reorganisation".
- **Not needed**: neighbourhoods and filters (`kernels/`) only for "CLAHE is not a filter: each
  pixel still goes through a curve, but the curve depends on where the pixel is".

## Overlaps with other plans and units (who takes what)

| Unit / plan | What this unit takes or teaches | What stays there |
|---|---|---|
| `highbit/` (built) | Point operations and LUTs in general, gamma/log as **contrast** curves, the display-vs-analysis table | Window/level, CT windows, auto-contrast percentiles, hi-lo LUT, HDR and tone mapping (its bullet gets a forward pointer: "log and gamma curves: next unit") |
| `histogram/` (built, day 6) | The **cumulative histogram** is introduced here as the equalisation mapping | Histograms as distributions, bins, the percentile threshold (it recalls the cumulative histogram in one line and gains the link "a percentile threshold is a fixed threshold on the equalised image") |
| `rgb-linear/` (built, day 7; owner of the sRGB transfer function) | The power law $v^\gamma$ as a **display/contrast** point operation, and "gamma changes ratios, so do not measure on it" | Gamma **encoding** for storage (sRGB curve with its linear knee), linear light, why blur/average in linear light; its "Gamma encoding" slide recalls "a power-law point operation (`contrast/`)" in one line |
| `colour-display/` (built, day 7) | nothing new | Colour maps; its slide "Displaying a grey image is already a choice" (which recalls LUT and window from the bit-depth day) can add `contrast/` to the recall |
| `box-filters/` (built, integral elective) | CLAHE as the **histogram-based** local contrast method | Local contrast normalisation $(I - m)/(s + \varepsilon)$ with box filters; one-line pointer each way ("two ways to make contrast local: per-tile curves (CLAHE) or local mean and SD") |
| `local-threshold/` (built, day 6) | nothing | Background correction (divide/subtract) and local thresholds. CLAHE is **not** a background correction: one line here, so students do not confuse "make it look even" with "make it measurably even" |
| `denoise/` (built, day 5) | The first "what may you measure" table (point operations) | The table for restorations (flux, peaks, sizes); it can recall this unit in one line ("as with contrast transforms: …") |
| `intensity-features/` (built, day 11) | Rule: never measure intensities on an equalised/CLAHE/matched image | Normalisation of **measurements** (plates, batches, robust z): the right place to normalise is the numbers, not the pixels. Pointer from here |
| `texture-features/` (built, day 11) | Exact (rank) equalisation is defined here (math/optional) | Uses histogram-equalised patches for GLCM (a legitimate use: texture features should not depend on contrast); can point back here |
| `stain-unmixing/` (built, unmixing elective) | Histogram matching of grey images | Stain normalisation between scanners (by stain vectors): one-line pointer "for colour slides, normalise stains, not histograms" |
| `coloc/` (built, colocalisation elective) | "Monotone transforms keep ranks": Spearman unchanged, Pearson changed (one exercise) | Pearson/Spearman as colocalisation measures |
| `review-gaps.md` gap 7 (`integrity/`, unplaced) | The display-vs-data part and the demo name `displayvsdata` | Provenance, duplication detection, ethics, bias: still unplaced (open question 4) |

## The unit: `contrast/`

About 1.5 h including the exercise. Order inside the deck: point operations → shaped curves
(gamma, log) → curves chosen by the histogram (equalisation) → local curves (CLAHE) → curves
chosen by another image (matching) → what all of this does to measurements.

### Concept layer

- **Recall** (one slide, from `highbit/`): the window maps raw values to display levels by a
  straight line, clipped. Draw it as a **curve** (raw on x, display on y) above the histogram.
  Every transform in this unit is another curve.
- **Point operations**: $s = T(v)$, the same function for every pixel. Position and neighbours
  do not matter, so the result is fully described by the curve. Examples as a gallery of curves
  with their results on the micrograph: identity, **negative** ($s = L - 1 - v$), **contrast
  stretch** (a window), **clipping**, a **threshold** (a step: the most extreme point operation,
  forward pointer to day 6), a **sigmoid** (S-curve: more contrast in the middle, less at the
  ends).
- **Lookup tables**: an integer image has only $L$ possible values, so compute $T$ once per
  value and look the result up: 256 entries for 8 bits, 4096 for 12 bits, for any number of
  pixels. Changing the display changes the table, never the image. Two LUTs in a row are one
  LUT (composition).
- **Reading a curve**: the **slope** is the contrast gain. Where the slope is > 1, neighbouring
  values move apart (more contrast, and the noise is amplified by the same factor); where it is
  < 1 they move together; where it is 0 (clipping) they merge. Flat parts lose information,
  steep parts on 8-bit data leave gaps. A curve that **rises everywhere** (monotone) keeps the
  order of the grey values: brighter stays brighter.
- **Gamma** $s = v^\gamma$ on values scaled to [0, 1]: γ < 1 brightens the darks (stretches
  them, compresses the brights), γ > 1 does the reverse; γ = 1 is the identity; black and white
  stay fixed. One knob, a smooth curve. Numbers: with γ = 0.5, 0.04 → 0.2 and 0.81 → 0.9: the
  dark values are stretched 2.5×, the bright ones compressed to about 0.56× (the slopes).
  Pointer: screens and image files use a gamma curve for storage (`rgb-linear/`, day 7); that is
  a different reason, and the colour day owns it.
- **Log** $s = c\,\log(1 + v)$: **equal ratios become equal steps**. A 12-bit image shown with
  $c = 255/\ln 4096 \approx 30.7$: 10 → 74, 100 → 141, 1000 → 212, 4095 → 255, so every decade
  gets about 70 display levels. Uses: Fourier spectra (recall `fourier/`), images spanning
  several orders of magnitude (dim cells next to bright spots, astronomy). Too much log makes
  the noise in the background as prominent as the objects.
- **Quantisation strikes back** (recall `quantisation/`): stretching 8-bit data by a factor 4
  uses only every 4th display level: the histogram becomes a **comb**, smooth gradients show
  bands. A LUT can merge levels but never split them. Rule: do contrast work on the 12/16-bit
  or float data and convert to 8 bits **last**, for display.
- **Let the data choose the curve: equalisation.** Give contrast where the pixels are: the slope
  of the curve should be proportional to how many pixels have that value. The curve whose slope
  is the histogram is the **cumulative histogram** $P(v)$ = fraction of pixels with value ≤ v
  (introduced here). Mapping: $s = \operatorname{round}\big((L - 1)\,P(v)\big)$. Result: about
  the same number of pixels at every output level, the histogram is "spread out".
- **Worked example (8 levels, 64 pixels)**: counts 6, 22, 18, 8, 5, 3, 1, 1 → cumulative 6, 28,
  46, 54, 59, 62, 63, 64 → $7P$ = 0.66, 3.06, 5.03, 5.91, 6.45, 6.78, 6.89, 7 → levels 1, 3, 5,
  6, 6, 7, 7, 7. Output counts at levels 0…7: 0, 6, 0, 22, 0, 18, 13, 5. A flat histogram would
  be 8 each.
- **Why only approximately flat**: all pixels with the same input value get the same output
  value, so a level with 22 pixels stays a level with 22 pixels: it can only be moved, not
  split. Levels can merge (5 and 6 here, rounding), and output levels stay empty (gaps). The
  output has **at most as many distinct values as the input**. The *cumulative* histogram of
  the output is close to a straight line; the histogram itself is not flat. (Exact flatness
  needs breaking ties between equal pixels, e.g. at random or by the local mean: a made-up
  ordering. Pointer, math layer.)
- **What equalisation depends on**: only the **ranks** of the pixels. Any rising curve applied
  before equalisation gives the same result (gamma then equalise = equalise). An equalised value
  says "this pixel is brighter than 73 % of the image": a statement about the image, not about
  the light. Link: thresholding the equalised image at 0.9 keeps the brightest 10 %, the
  percentile threshold of `histogram/` (forward pointer).
- **Where global equalisation fails**: a large, flat background is the biggest peak, so it gets
  the steepest part of the curve: its **noise is amplified** (numbers: 40 % of the pixels in 20
  of 256 levels → slope ≈ 5, noise SD 3 → 15). Small objects with few pixels get little contrast.
  Bright and dark regions of one image compete for the same curve.
- **Adaptive equalisation**: one curve per **tile** (e.g. 8 × 8 tiles), from the tile's own
  histogram. Applied tile by tile it gives **seams** at the tile borders, so each pixel's value
  is a blend of the curves of the four nearest tile centres, weighted by distance (bilinear
  interpolation, recall `images/`). Still a curve per pixel, but the curve depends on where the
  pixel is: the same raw value can become different display values in different places.
- **CLAHE: clip the histogram first.** In a flat tile the histogram is one narrow peak and the
  curve is nearly a step: noise blows up. Clip every bin at a limit (a multiple β of the average
  count, e.g. β = 2–4), spread the clipped excess evenly over all bins, then accumulate. Since
  the slope is the histogram, **the clip limit is the maximum contrast gain**. β = 1: no change
  (identity curve); β → ∞: unclipped adaptive equalisation. Two knobs: tile size (what counts as
  "local", larger than the objects) and clip limit (how much contrast, and noise, may be added).
  Uses: retinal and X-ray images, underwater and fog photographs, preprocessing before detection
  or segmentation; pointer: image libraries call it "CLAHE".
- **Histogram matching** (specification): make an image's histogram look like a **target**
  histogram (another image, a reference, or a shape). Two cumulative histograms: for each input
  value, read its fraction $F(v)$, and find the target value with the same fraction. Equalisation
  is matching to a flat target. Uses: making a tile mosaic or a time series look consistent for
  display, preprocessing for a model trained on another instrument. It also **erases real
  differences**: matching a treated image to a control makes the treatment invisible.
- **What these transforms do to measurements** (the slide every student must remember), as a
  table with rows *window (linear) / gamma, log / global equalisation / CLAHE / matching* and
  columns *order of grey values kept? / differences and ratios kept? / same raw value → same
  output everywhere in the image? / comparable between images?*:
  - linear window: order yes; differences and ratios after removing the offset yes (up to
    clipping); same mapping everywhere yes; between images yes if the same window is used;
  - gamma, log: order yes; differences and ratios no (log: ratios become differences); same
    mapping everywhere yes; between images yes with the same curve;
  - equalisation: order yes; ratios no; same mapping within the image yes, **but the curve
    depends on the image**, so not comparable between images;
  - CLAHE: order only **locally**; the same raw value maps to different outputs in different
    places, so even within one image two cells cannot be compared;
  - matching: order yes; the **differences between images are removed by construction**.
  Worked: two cells of raw mean 300 and 600 on a background of 100 (12-bit): background-
  subtracted ratio 2.5; after γ = 0.5 (on v/4095) the same computation gives 1.98. And: the
  square root of a mean is not the mean of the square roots (pixels 100 and 900: √500 = 22.4,
  but (10 + 30)/2 = 20).
- **What to do**: (1) segment or detect on whatever transform helps, then **measure on the raw
  (linear) data inside the masks**; (2) show figures with the same display settings for images
  you compare, and say so in the caption; (3) report non-linear display adjustments (gamma,
  equalisation, CLAHE) in the figure legend; (4) normalise **measurements** (per well, against
  controls: `intensity-features/`), not pixels. One line: journal image-integrity guidelines say
  the same (no names). Forward pointer: `denoise/` (day 5) asks the same question for restored
  images.

### Math layer (`## The math: … {.math}` slides)

- *Point operations and the histogram*: for a rising, differentiable $T$ and a continuous
  intensity with density $p_X$, $p_S(s) = p_X(v)/T'(v)$ at $s = T(v)$ (change of variables):
  slope > 1 spreads the density thin, slope < 1 piles it up. Composition of LUTs:
  $(T_2\circ T_1)$ is again a table. A point operation is invertible exactly when it is strictly
  monotone on the values that occur.
- *Gamma and log*: $\frac{d}{dv}v^\gamma = \gamma v^{\gamma - 1}$ (→ ∞ at 0 for γ < 1: why
  gamma curves amplify dark noise most); $\log(a v) = \log a + \log v$, so a multiplicative
  factor (exposure, illumination) becomes an additive shift; noise after a point operation
  (delta method): $\operatorname{SD}(T(X)) \approx |T'(\mu)|\,\operatorname{SD}(X)$. For Poisson
  $X$ with mean μ and $T = 2\sqrt{x}$: $\operatorname{SD} \approx \mu^{-1/2}\sqrt\mu = 1$,
  constant: the variance-stabilising transform pointed to in `cameranoise/`.
- *Equalisation*: if $X$ is continuous with cumulative distribution $F$, then $Y = F(X)$ is
  uniform on [0, 1]: $P(Y \le y) = P(X \le F^{-1}(y)) = F(F^{-1}(y)) = y$ (the probability
  integral transform). Discrete: $s_k = \operatorname{round}\big((L - 1)F(k)\big)$; the output
  pmf is a sum of input pmf values, so any level with $h(k) > N/L$ stays above $N/L$; the number
  of used output levels ≤ number of used input levels; what *is* close to linear is the output's
  cumulative histogram (within $1/(2(L-1))$ from rounding, plus the largest single bin). Variant
  $s_k = \operatorname{round}\big((L - 1)\frac{c_k - c_{\min}}{N - c_{\min}}\big)$ (so the
  darkest occurring value maps to 0); stated. Invariance: for a strictly rising $g$,
  $F_{g(X)}(g(v)) = F_X(v)$, so equalising $g(X)$ gives the same image.
- *Exact equalisation* (one bullet): order all $N$ pixels by value, break ties (random, or by a
  local mean), assign output levels by rank in blocks of $N/L$: exactly flat, at the price of an
  invented order among equal pixels.
- *Matching*: $T = G^{-1}\circ F$ with the generalised inverse
  $G^{-1}(u) = \min\{v' : G(v') \ge u\}$; continuous case: $T(X)$ has distribution $G$ (apply
  the probability integral transform twice). Equalisation is $G$ = uniform.
- *CLAHE*: the tile's mapping is $T(v) = \frac{L - 1}{N}\sum_{k \le v} \tilde h(k)$ with the
  clipped histogram $\tilde h(k) = \min(h(k), \beta N/L) + E/L$, $E$ = the clipped excess. Its
  slope is $\frac{L - 1}{N}\tilde h(v) \le \frac{L - 1}{L}(\beta + E/N) \approx \beta$: the clip
  limit bounds the contrast gain (exactly, if the redistribution is repeated until no bin
  exceeds the limit). Interpolation: $T(v;\mathbf x) = \sum_{i=1}^4 w_i(\mathbf x)\,T_i(v)$ with
  bilinear weights; each $T_i$ rises in $v$, so the blend rises in $v$ (order kept at a fixed
  position), but nothing relates $T(\cdot;\mathbf x)$ to $T(\cdot;\mathbf y)$.
- *Measurements*: only affine $T(v) = a v + b$ commutes with averaging; for a concave $T$
  (γ < 1, log) Jensen gives $\overline{T(v)} \le T(\bar v)$. Ratios after offset removal are kept
  only by $T(v) = a(v - b)$. Rank statistics commute with any rising $T$:
  $\operatorname{median}(T(X)) = T(\operatorname{median}X)$, percentiles, Spearman correlation
  (pointer `coloc/`), and any threshold *chosen by a percentile*. Otsu's threshold does **not**
  simply move with $T$ (it depends on means and variances): exercise.

### Slides and figures (about half carry an image)

Target 26 content slides including 7 math slides; 16 carry a figure or a demo (●).
Hand-written SVGs or figures from `tools/figs/contrast/figs.py` (data dumped by a small
`tools/figs/contrast/data.ts` from `lib/contrast.ts`, as the other units do). All images are
the course micrograph `cell256.png` (credit) and the synthetic 12-bit fluorescence image of
`window16` (labelled synthetic).

1. About this lecture.
2. ● Recall: a window is a curve — `fig-window-curve.svg`: histogram of the 12-bit image (log
   counts) with the window drawn as a curve on top, raw → display.
3. ● Point operations — `fig-pointop.svg`: a 3 × 3 patch of numbers → curve → 3 × 3 patch; the
   neighbours are greyed out ("not used").
4. ● A gallery of curves — `fig-curves.png`: six small panels (negative, stretch, clip,
   threshold, sigmoid, identity), each curve above its result on the micrograph.
5. Lookup tables (text + a small table; composition).
6. The math: a curve reshapes the histogram {.math} — ● `fig-density.svg`: a histogram mapped
   through a curve, bins drawn as strips that widen where the slope is > 1.
7. ● Gamma — `fig-gamma.png`: curves γ = 0.4, 1, 2.5 and the 12-bit image at the three gammas.
8. ● Log: equal ratios, equal steps — `fig-log.svg`: log curve for 12 → 8 bit with the decades
   10, 100, 1000 marked; inset: a spectrum shown linearly and as log(1 + |X|).
9. The math: gamma, log and noise {.math}.
10. ● Try it: tone curves (`tonecurve`).
11. ● Stretching 8 bits leaves a comb — `fig-comb.svg`: histogram before/after ×4 stretch of
    8-bit data vs of 12-bit data.
12. ● Let the histogram choose the curve — `fig-idea.svg`: histogram and the curve whose slope
    follows it (steep at the peak, flat at the tails).
13. ● The cumulative histogram is the mapping — `fig-cdf-map.svg`: cumulative histogram used as
    the curve; two arrows from input values to output levels.
14. Worked example: 8 levels by hand (the table above).
15. ● Why only approximately flat — `fig-flat.svg`: output histogram of the worked example
    (tall bars, empty levels) next to its nearly straight cumulative histogram; and the
    micrograph's equalised histogram.
16. The math: equalisation {.math} (probability integral transform; discrete case; invariance;
    exact equalisation in one line).
17. ● Try it: equalisation (`equalise`).
18. ● Where global equalisation fails — `fig-global-fail.png`: the 12-bit image equalised: the
    background noise is loud, the dim cell barely gains; the curve with its steep part at the
    background peak.
19. ● One curve per tile — `fig-tiles.svg`: a 4 × 4 tile grid on the image, four tile curves,
    a pixel with its four nearest tile centres and weights; inset with seams (no
    interpolation).
20. ● Clip the histogram — `fig-clip.svg`: a tile histogram with the clip line, the excess
    spread over all bins, the two resulting curves (slopes capped).
21. The math: CLAHE {.math}.
22. ● Try it: CLAHE (`clahe`).
23. ● Histogram matching — `fig-match.svg`: two cumulative histograms side by side, an arrow
    from $F(v)$ across to the target value with the same fraction.
24. The math: matching {.math}.
25. What each transform does to measurements (the table; the 300/600 example below it).
26. ● Try it: display vs data (`displayvsdata`).
27. The math: measurements under a point operation {.math} (Jensen, ranks, Otsu).
28. What to do (the four rules) — text only.
29. Summary.

That is 28 headings with the opening and the summary (26 content slides); if the deck runs long,
slides 5 and 3 merge, and slide 11 can move to the exercise page.

### Demos

All TypeScript on the shared libraries; every demo has presets, a reset, `data-*` settings,
labels ≥ 18 px on slides, fits 1150 × 560, fixed seeds, no network. Images: `cell256.png` via
`lib/cell.ts` (8-bit, credit shown), the 12-bit synthetic fluorescence image (moved from
`window16/data.ts` into `lib/synth.ts` as `fluorescence12`, see shared engineering), and two
small synthetic scenes (a low-contrast "foggy" scene; a control/treated pair).

**Demo `tonecurve`** (point operations): left the image, right the **curve editor** (raw on x,
display on y) above the input histogram, with the output histogram below. Curve modes: window
(two handles), gamma (slider 0.2–5, log), log (slider c or "fit to range"), sigmoid (centre,
steepness), **free curve** (5 draggable control points, monotone piecewise-linear; a toggle
allows non-monotone curves to show order reversal), negative, threshold. Toggle source
**8-bit / 12-bit** (the comb appears only with 8-bit). Readouts: slope at the pointer, number of
distinct output levels used, fraction clipped low/high. Click a pixel: its raw value, its path
through the curve (dotted lines), its display value. Presets: "the window is a straight line",
"γ = 0.5 lifts the darks", "γ = 2 darkens", "log: every decade gets 70 levels", "stretch 8 bits ×4:
the comb", "the same stretch on 12 bits", "S-curve", "threshold: the steepest curve", "a falling
curve reverses the order". Timing: LUT build ≤ 4096 entries, apply 65 k pixels: < 2 ms.

**Demo `equalise`** (global equalisation and matching): panels: input image; output image; input
histogram with its cumulative histogram overlaid (second axis); output histogram with its
cumulative histogram; the **mapping curve** (which *is* the scaled cumulative histogram).
Controls: image (micrograph, 12-bit fluorescence, foggy scene, the 8-level 8 × 8 example drawn
as a pixel grid with numbers), output levels L (8 / 16 / 64 / 256), mode **equalise / exact
(rank, random ties, seed) / match to …** (target: flat, Gaussian, another image), *apply a
gamma first* slider (to show that equalisation ignores it), noise. Readouts: number of used
output levels, largest output bin vs the flat value N/L, max deviation of the output cumulative
histogram from a straight line. Hover-free: click a bin to highlight its pixels in both images.
Presets: "8 levels by hand" (the worked example, numbers shown), "the cumulative histogram is
the curve", "not flat: bins move, not split", "exact equalisation: flat, but invented order",
"gamma first, same result", "the background peak gets the contrast", "match to another
image", "match to a Gaussian".

**Demo `clahe`** (local equalisation): panels: input with the tile grid overlaid; output; for the
clicked pixel: the four nearest tile centres highlighted, their four curves and the blended
curve (with weights), and the clicked tile's histogram with the clip line and the
redistributed excess. Controls: tiles per side (2–16), clip limit β (1–16, and ∞ = plain
adaptive equalisation), *interpolate between tiles* (off shows the seams), image (12-bit
fluorescence, micrograph, foggy scene, flat noisy background with one faint cell). Readouts:
largest slope of the clicked tile's curve (≤ β), background noise SD before/after in a marked
box. Presets: "global equalisation (1 tile)", "adaptive, no clip: the noise explodes",
"CLAHE β = 3", "tiles without interpolation: seams", "tiles too small", "tiles too large =
global", "β = 1: nothing happens", "the same raw value, two outputs" (two marked pixels with the
same raw value and different outputs). Timing: 8 × 8 tiles with 256-bin histograms and
interpolation on 256²: a few ms; for 12-bit data the tile histograms use 256 bins over the
data range (stated in the caption).

**Demo `displayvsdata`** (what transforms do to measurements; the review's proposed name):
a synthetic **control / treated** pair (12-bit, seeded, the same cells; the treated nuclei are
1.5× brighter, background 100, Poisson noise; ground-truth masks). Top: both images under the
current transform; bottom: a measurement table for both images and their ratio: mean nucleus
intensity minus background, total intensity, nucleus area from a fixed threshold, area from a
percentile threshold, median, and a rank-based statistic (fraction of nucleus pixels above the
image median). Transform (applied to **both** images): none / same window / separate
auto-windows / gamma / log / equalise each image / CLAHE / match treated to control. Toggle
*measure on: displayed values / raw values inside the masks found on the displayed image*.
Readout: "true treated/control ratio 1.50" vs the measured one, coloured green within ±2 %.
Presets: "raw: 1.50", "same window: still 1.50", "separate auto-windows: the figure lies, the
data do not", "gamma: ratio changes", "equalise each: treatment gone", "match to control:
treatment erased", "CLAHE: cells within one image no longer comparable", "segment on CLAHE,
measure raw: 1.50 again".

### Exercise (`exercise.qmd`)

Concept questions (with a `Solutions` callout):

- **B1.** Point operation or not? negative; Gaussian blur; window; median filter; threshold;
  CLAHE; adding a constant; subtracting a background image. For the point operations, sketch the
  curve.
- **B2.** Gamma by hand: values 0.01, 0.04, 0.25, 0.81 (on [0, 1]) with γ = 0.5 and γ = 2. Which
  γ brightens the darks? Slope of γ = 0.5 at 0.04 and at 0.81, and what it does to noise of SD
  0.01 there. (Answers: 0.1, 0.2, 0.5, 0.9; slopes 2.5 and 0.56; SD 0.025 and 0.0056.)
- **B3.** Log display of a 12-bit image to 8 bits with $s = c\ln(1 + v)$: find $c$ so that 4095
  → 255, then map 10, 100, 1000. How many display levels per decade? (30.7; 74, 141, 212; about
  71.)
- **B4.** An 8-bit image uses values 60–123. You stretch it linearly to 0–255. How many distinct
  values does the result have, and what does its histogram look like? Same for a 12-bit image
  using 960–1980 shown on 8 bits.
- **B5.** Equalise the 8-level, 64-pixel image (counts 6, 22, 18, 8, 5, 3, 1, 1) by hand: the
  cumulative histogram, the mapping, the output histogram. Is it flat? Which input levels
  merged, which output levels are empty, and why can the level with 22 pixels not be split?
- **B6.** Histogram matching by hand: the same image to the target counts 0, 0, 8, 8, 16, 16, 8,
  8 (mapping: 2, 4, 5, 6, 7, 7, 7, 7).
- **B7.** Predict: (a) you apply γ = 0.5 and then equalise; (b) you equalise and then equalise
  again; (c) you equalise an image with a constant value. (Same as equalising directly; nothing
  changes, up to rounding; a single output level.)
- **B8.** A background peak holds 40 % of the pixels within 20 of 256 levels, noise SD 3 grey
  levels. Approximate slope of the equalisation curve there and the noise SD afterwards. With
  CLAHE at β = 3, at most?
- **B9.** A CLAHE tile of 16 pixels, 4 levels, counts 1, 10, 4, 1, clip limit 1.5 × the mean
  count (6 pixels): clip, redistribute, accumulate. Contrast gain $L\,h(v)/N$ of the most
  common level before and after (2.5 → 1.75). Why can a bin end above the limit after one
  redistribution?
- **B10.** Two cells: raw means 300 and 600, background 100 (12-bit). Ratio after background
  subtraction; the same after γ = 0.5 applied to $v/4095$ (2.5 vs 1.98); after equalising each
  image separately (cannot be computed: why not?).
- **B11.** Which of these are unchanged by any rising curve: mean, median, 90th percentile
  position *as a fraction of pixels*, Pearson correlation, Spearman correlation, the mask of a
  "brightest 10 %" threshold, the mask of an Otsu threshold?
- **B12.** A colleague equalises every image of a time series "so the cells are easier to see",
  then plots the mean cell intensity over time. What goes wrong, and what should they do?

Math questions (`### Math questions`, each tagged *(math)*, with `Solutions (math)`):

- **B13.** *(math)* Change of variables: show $p_S(s) = p_X(v)/T'(v)$ for a rising $T$, and use
  it to find the histogram shape after $T(v) = \sqrt v$ of a uniform input on [0, 1].
- **B14.** *(math)* Prove that $F(X)$ is uniform for continuous $X$; show that for discrete $X$
  the output pmf at a used level is a sum of input pmf values, hence a level with
  $h(k) > N/L$ stays above $N/L$.
- **B15.** *(math)* Show that equalisation is invariant to a strictly rising $g$ applied first.
- **B16.** *(math)* Show that $T = G^{-1}\circ F$ makes $T(X)$ have distribution $G$ (continuous
  case); what goes wrong when $G$ has flat parts?
- **B17.** *(math)* CLAHE slope bound: with clip limit $\beta N/L$ and the excess spread evenly,
  show that the slope of the tile curve is at most $\frac{L - 1}{L}(\beta + E/N)$.
- **B18.** *(math)* Delta method: show $\operatorname{SD}(T(X)) \approx |T'(\mu)|\operatorname{SD}(X)$
  and that $2\sqrt X$ has SD ≈ 1 for Poisson $X$ with large mean.
- **B19.** *(math)* Jensen: for concave $T$, $\mathbb E[T(X)] \le T(\mathbb E[X])$; give the
  numbers for pixels 100 and 900 and $T = \sqrt{\cdot}$. Which $T$ keep the ratio
  $(\bar a - b)/(\bar c - b)$ of background-subtracted means?

Explore sections (each with E-questions and solutions):

- `## Explore {#explore}`: `tonecurve` (E1 window as a line; E2 γ and the slope readout; E3
  8-bit vs 12-bit comb; E4 a falling curve).
- `### Equalisation {#explore-equalise}`: `equalise` (E5 the 8-level preset vs B5; E6 count used
  levels at L = 256 on the micrograph; E7 gamma first; E8 match to another image).
- `### CLAHE {#explore-clahe}`: `clahe` (E9 no clip vs β = 3: background SD; E10 seams; E11 tile
  size; E12 two equal raw values, two outputs).
- `### Display vs data {#explore-measure}`: `displayvsdata` (E13 go through the presets, fill
  the ratio column; E14 which transforms are safe to *segment* on, and why "segment on CLAHE,
  measure raw" recovers 1.50).

### Quiz (`quiz.qmd`)

About 8 questions, mostly concept: identify a curve from a before/after histogram; which γ for
dark cells; is a threshold a point operation; why the equalised histogram has gaps; what the
clip limit controls; matching a treated image to a control; may you report the mean intensity
of an equalised image (no); one *(math)* question: the slope of the equalisation curve equals
the scaled histogram.

## Optional units

None. If the deck is too long, the fallback is to move histogram matching (slides 23–24) to a
collapsed callout on the exercise page and keep the `equalise` demo's matching mode. A separate
`integrity/` unit (provenance, duplication, ethics) remains a gap-7 question, not part of this
plan.

## Shared engineering

- **New lib file `demos/src/lib/contrast.ts`** (owned by this unit; no existing file of that
  name). No DOM. Works on `Grey` (`lib/imageops.ts`) and on integer arrays with a known number of
  levels:
  - `Lut` (Float32Array of length L), `applyLut(img, lut, vmin)`, `composeLut(a, b)`,
    `lutFromFunction(f, L)`, `isMonotone(lut)`, `slopeOf(lut, v)`;
  - curves: `windowCurve(lo, hi)`, `gammaCurve(γ, lo, hi)`, `logCurve(c | "fit", lo, hi)`,
    `sigmoidCurve(centre, width)`, `piecewiseLinear(points)` (monotone option);
  - `equaliseLut(hist, L, variant: "plain" | "cmin")` using `cumulative()` from
    `lib/histogram.ts`; `equaliseExact(img, L, tieBreak: "random" | "localMean", seed)`;
  - `matchLut(srcHist, targetHist)` with the generalised inverse; target generators
    `flatTarget`, `gaussianTarget`;
  - `clipHistogram(hist, limit, iterate)` → {clipped, excess}; `clahe(img, {tilesX, tilesY,
    clip, bins, range, interpolate})` → {out, tileLuts, tileCentres}; `claheWeights(x, y, …)`
    for the demo's four-tile inspector.
  The window formula already in `window16/data.ts` (`windowLevel`) moves into `windowCurve`;
  `window16` may keep its local copy (no edit required by this plan).
- **`lib/synth.ts`** (extend through its owner): `fluorescence12(seed)` (the generator in
  `window16/data.ts`, moved so that this unit does not import from another demo's folder),
  `foggyScene(seed)` (low-contrast shapes on a smooth gradient), `controlTreated(seed, factor)`
  (two 12-bit images with shared masks). If the owner prefers, these live in
  `demos/src/displayvsdata/scene.ts` and `demos/src/equalise/scenes.ts` first and move later
  (the build-issues list shows that local scenes tend to stay local: prefer the lib).
- Histograms: `intHistogram`, `histogram`, `cumulative`, `histPercentile` from
  `lib/histogram.ts`; plotting with `lib/plot.ts` (`histogram`, `step`, `line`, second axis by a
  second `Plot` overlaid; `vline` for the clip limit); images with `drawGrey`/`toRGBA`
  (`lib/imageops.ts`); thresholds in `displayvsdata` from `lib/threshold.ts`
  (`percentileThreshold`, `otsu`, `applyThreshold`); `rng` from `lib/random.ts`.
- **CSS**: every per-demo rule prefixed `.qb-demo[data-demo="tonecurve"]` (etc.), because
  `demo.css` loads late and wins otherwise (build lesson).
- **Figures**: `![](fig-x.svg){fig-alt="…"}` (no bracket caption text on slides: Quarto shows it
  as a caption), `.nostretch` on slides with a lone image; figure scripts in
  `tools/figs/contrast/` (`figs.py` + `data.ts`), committed with the unit, not left in the
  scratchpad.
- **Tests** `demos/test/contrast-test.ts` (run by `make test`): identity LUT is the identity;
  `composeLut` equals applying twice; the window curve matches `highbit/`'s formula at 130 and
  3176; the 8-level worked example gives mapping 1, 3, 5, 6, 6, 7, 7, 7 and output counts 0, 6,
  0, 22, 0, 18, 13, 5; the matching example gives 2, 4, 5, 6, 7, 7, 7, 7; exact equalisation of
  65 536 pixels to 256 levels gives exactly 256 per level; equalising after a rising gamma gives
  the same image; output distinct values ≤ input distinct values; CLAHE with one tile and no clip
  equals global equalisation; with β = 1 (iterated) it is the identity up to rounding; every
  tile LUT slope ≤ β (iterated clipping); interpolated CLAHE has no jump larger than the noise
  across tile borders on a constant-gradient image; the B10 numbers (2.5, 1.98).
- **Speed**: all live recomputes are LUT builds plus one pass over 256² pixels (< 5 ms in
  TypeScript, to be measured in the browser); no Wasm.
- **Gallery and sidebar**: four entries in `demos.qmd`; sidebar entries Lecture, Quiz,
  Exercise, "Demo: tone curves" (`#explore`), "Demo: equalisation" (`#explore-equalise`),
  "Demo: CLAHE" (`#explore-clahe`), "Demo: display vs data" (`#explore-measure`).

## Suggested timetable (day 3 with five units)

| Time | |
|---|---|
| 09:00–10:15 | `quantisation/` (9 slides) lecture + exercise |
| 10:30–11:45 | `cameranoise/` lecture + exercise |
| 12:45–14:00 | `highbit/` lecture + exercise (`window16`) |
| 14:15–15:45 | **`contrast/`**: lecture (`tonecurve`, `equalise`, `clahe`), B5 by hand, `displayvsdata` |
| 16:00–17:00 | `floatingpoint/` lecture + exercise; quizzes as wrap-up |

If the day is too long: `floatingpoint/` becomes self-study (it is the most self-contained unit
of the day), or this unit's matching slides move to the exercise page.

## Proposed reorganisation

1. **The cumulative histogram is introduced in `contrast/` (day 3), not in `histogram/`
   (day 6).** Owners table in `plan/coursemap.md`: "Histograms, thresholds, Dice/ROC for
   pixels → Thresholding" stays; add a row "Point operations and LUTs (beyond the window),
   gamma/log as contrast curves, histogram equalisation, CLAHE, histogram matching, display vs
   analysis for intensity transforms → `contrast/` (day 3)", with the note "the cumulative
   histogram is first introduced in `contrast/` as the equalisation mapping; `histogram/`
   recalls it". Edit to `histogram/lec.qmd`, slide "The cumulative histogram and the percentile
   threshold": first bullet becomes a one-line recall ("the cumulative histogram, the
   equalisation curve of `contrast/`"), and add one line: "a percentile threshold is a fixed
   threshold on the equalised image". Why it improves the course: the cumulative histogram is
   met first in its most visual use (a curve you can see acting on an image), and the percentile
   threshold gets a second meaning for free.
2. **`highbit/` keeps windows; its "Colour maps, HDR and tone mapping" slide gains a forward
   pointer** ("log, gamma and equalisation: next unit, `contrast/`"); its summary line "Displays,
   LUTs and tone mapping are for looking" stays as the bridge.
3. **Gamma split between `contrast/` and `rgb-linear/`**: `contrast/` owns the power law as a
   point operation and its effect on measurements; `rgb-linear/` keeps the sRGB transfer
   function, gamma *encoding* and linear light, and recalls the power-law curve in one line.
   Add a line to the owners table.
4. **Gap 7 (`integrity/`)**: this unit takes its "display vs data" part and its demo name
   `displayvsdata`; `plan/bitdepth-day.md`'s open item and the diffusion/moderndl/evaluation
   plans' pointers "to the integrity unit if it exists" can point to `contrast/` for display
   adjustments (one line each, when those units are next edited). The rest of gap 7 stays open.
5. **One-line recalls elsewhere** (when those units are next edited; not required for this
   unit to work): `colour-display/` ("Displaying a grey image is already a choice"),
   `box-filters/` (local contrast normalisation ↔ CLAHE), `denoise/` ("What happens to
   measurements": "as for contrast transforms"), `intensity-features/` (normalise measurements,
   not pixels), `texture-features/` (its equalised patches: "exact equalisation, `contrast/`").
6. `plan/thresholding-day.md` open item ("histogram equalisation: one slide in `histogram/` or
   in `highbit/`?") and `plan/review-gaps.md` gap 10 are resolved by this unit.

## Open questions

1. **Placement**: day 3 after `highbit/` (proposed) or day 6 after `histogram/` (fallback)? The
   decision changes only who introduces the cumulative histogram and the recall lines.
2. **Day 3 length** with five units: accept the tighter timetable, or make `floatingpoint/`
   self-study?
3. **Exact equalisation**: keep it in the math layer and the `equalise` demo (proposed), or drop
   it? It explains `texture-features/`'s "256 equally filled levels" but adds one more idea.
4. **The rest of gap 7** (provenance, duplicated regions, ethics, bias) still has no home; a
   closing course unit is the review's other option.
5. **Real images**: a real low-contrast image (retinal fundus, chest X-ray, or a dim
   fluorescence field) with a compatible licence would show CLAHE's typical use better than the
   synthetic foggy scene; the colour and thresholding days have similar open requests.
6. **Colour**: equalising a colour image (per channel shifts hues; equalise lightness only, e.g.
   L* from `colour-perceptual/`) as one line in this unit, or as a recall on the colour day?
   Proposed: one line here with a pointer, nothing on the colour day.

## References

- Gonzalez & Woods, *Digital Image Processing* (4th ed., 2018), ch. 3 (intensity transformations,
  histogram equalisation and specification).
- Pizer, Amburn, Austin, Cromartie, Geselowitz, Greer, ter Haar Romeny, Zimmerman & Zuiderveld
  (1987), *Adaptive histogram equalization and its variations*, Computer Vision, Graphics, and
  Image Processing 39.
- Zuiderveld (1994), *Contrast limited adaptive histogram equalization*, in Graphics Gems IV,
  Academic Press (CLAHE with clipping and bilinear interpolation between tiles).
- Coltuc, Bolon & Chassery (2006), *Exact histogram specification*, IEEE Trans. Image Processing
  (tie-breaking by local means).
- Anscombe (1948), *The transformation of Poisson, binomial and negative-binomial data*,
  Biometrika (variance stabilisation; teacher background).
- Rossner & Yamada (2004), *What's in a picture? The temptation of image manipulation*, J. Cell
  Biol. (display vs data; image-integrity guidelines; teacher background, not named on slides).
- Cromey (2010), *Avoiding twisted pixels: ethical guidelines for the appropriate use and
  manipulation of scientific digital images*, Sci. Eng. Ethics.

## Build notes (2026-11-13)

Built: `contrast/` (lec 30 headings: 28 content slides incl. 6 math slides, 18 with a figure
or demo; quiz 9 questions; exercise B1–B19, E1–E15), demos `tonecurve`, `equalise`, `clahe`,
`displayvsdata`, lib `demos/src/lib/contrast.ts` (new), tests `demos/test/contrast-test.ts`
(46 checks), figures from `tools/figs/contrast/data.ts` + `figs.py` (14 figures).

Deviations from the plan above:

- **Scenes live in `lib/contrast.ts`**, not in `lib/synth.ts` (an existing file this build
  could not edit): `fluorescence12` (same generator and seed as `window16/data.ts`; the test
  checks its 0.5/99.5 % percentiles are 130 and 3176), `foggyScene`, `faintCellScene`,
  `controlTreated`. Move them to `synth.ts` when its owner next edits it. `lib/README.md`
  needs an entry for `contrast.ts` (also not editable here).
- `displayvsdata`: the table has five rows (mean, total and median in nuclei minus
  background; Otsu area on each display; area above the control's threshold); the
  rank-based row was dropped (it is ≈ 1 for both images and teaches nothing). Nuclei are
  segmented with Otsu on each displayed image; the control's nucleus signals are 600–1300
  ADU so that Otsu separates all nuclei from the background on the raw data. CLAHE there
  uses 6 × 6 tiles, β = 6 (brightest/dimmest nucleus 2.17 → 1.52; 4 × 4, β = 3 gave only 1.86).
- `equalise`: "exact" uses random tie-breaking (the lib also has local-mean ties); the noise
  slider is there; "match to another image" uses the micrograph (or the foggy scene when the
  source is the micrograph).
- `tonecurve`: the output histogram is drawn with √counts so the comb stays visible next to
  the background peak; the slope readout is the contrast gain relative to showing the full
  range linearly.
- Figures: `fig-log`, `fig-tiles`, `fig-global-fail`, `fig-gamma`, `fig-curves` are PNG (they
  hold images); `fig-idea` uses the foggy scene (the micrograph's broad histogram made the
  "steep/flat" point weak).
- Demo label sizes on slides: 18 px base, 16 px buttons, 15–17 px captions and readouts
  (the common compromise of the other dense demos).

Integration (for the integrator; files this build did not touch):

- Sidebar: a section "Contrast and intensity transforms" in Day 3 between `highbit/` and
  `floatingpoint/` (Lecture, Quiz, Exercise, and the four demo anchors `#explore`,
  `#explore-equalise`, `#explore-clahe`, `#explore-measure`).
- Gallery (`demos.qmd`): four entries after the `window16` entry.
- `Makefile`: add `demos/test/contrast-test.ts` to `TESTS`.
- The one-line recalls and pointers of "Proposed reorganisation" above.
