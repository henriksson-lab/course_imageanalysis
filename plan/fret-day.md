# Plan: a day on FRET (Förster resonance energy transfer)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
**Course map (2026-11-13, self-study restructure):** day 45 of 51, Part VIII "Colour and fluorescence". Hard prerequisites: Colour (day 42). Soft (see also): Spectral unmixing (day 43), Measurements (day 16), FFT registration (day 23). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VIII (colocalisation,
**FRET**, FLIM); entry after day 42 (colour); the spectral unmixing day
(`crosstalk/` in particular) helps but is not required. Ownership change from the course map
applied: **FRET bleed-through correction builds on the unmixing day's crosstalk unit (recall)**.
The FLIM day (planned in parallel) owns lifetimes, phasors and FLIM-FRET; this day points to it.

## Goal of the day

After the day a student can:

1. explain FRET in pictures: an excited **donor** hands its energy to a nearby **acceptor**
   without a photon; the donor gets **dimmer** (quenched) and the acceptor **lights up**
   (sensitised emission); state what is needed (spectral overlap, distance of a few nm,
   orientation) and why FRET is a **molecular ruler** below the resolution limit;
2. compute the **FRET efficiency** $E = 1/(1 + (r/R_0)^6)$ for given $r$ and $R_0$, read the
   **sensitive window** (about $0.5R_0$–$1.5R_0$) off the curve, and convert an efficiency back
   to an **apparent distance**; explain what the **Förster radius** depends on (overlap,
   donor brightness, orientation factor $\kappa^2$, refractive index);
3. correct a **sensitised-emission** (three-image, "three-cube") measurement by hand:
   estimate the **bleed-through** (donor into the FRET channel) and **cross-excitation**
   (acceptor directly excited) coefficients from donor-only and acceptor-only controls,
   subtract them, and turn the corrected signal into an efficiency with a calibration factor
   $G$; recognise false FRET from a wrong coefficient;
4. compute $E$ from **acceptor photobleaching** (donor before and after), and correct it for
   incomplete bleaching, donor bleaching and motion; explain why it is an end-point method;
5. explain why a measured efficiency is an **apparent** one (efficiency × fraction of donors
   with an acceptor), why averaging $E$ is not averaging $r$, and why $\kappa^2$ makes absolute
   distances uncertain; know that lifetime measurements (FLIM day) separate efficiency from
   fraction;
6. use **ratiometric imaging** with a single-chain biosensor: why the ratio cancels
   concentration, thickness and illumination; subtract background **before** dividing; mask,
   display (intensity-modulated ratio), calibrate a ratio to a concentration and see why
   values near $R_\text{max}$ are unreliable;
7. predict the **noise of a ratio** and of a corrected difference: $\mathrm{CV}^2 \approx
   1/N_A + 1/N_D$ for photon counts, bias and heavy tails when the denominator is small,
   ratio of sums vs mean of ratios, binning or smoothing before dividing; *(math)* derive the
   first-order (delta-method) variance and bias of a ratio, the variance of the three-cube
   $F_c$, and $E = 1 - \tau_{DA}/\tau_D$ from the rate picture.

Why it fits this audience: it is one sixth-power law, a $3\times3$ (lower-triangular) linear
model that is exactly the crosstalk compensation of the unmixing day, and first-order error
propagation (a Taylor expansion, which they know). The new parts are physical (energy transfer,
quenching) and statistical (ratios of noisy counts), both with strong pictures. The day is also
the place where "a number from an image" is most obviously a chain of corrections, each with
its own control.

Notation (course conventions): $\lambda$ is **wavelength** on this day (no prior weight
appears; as on the unmixing day). $E$ FRET efficiency, $r$ donor–acceptor distance, $R_0$
Förster radius, $\kappa^2$ orientation factor, $\tau$ lifetime. Image names: $I_{DD}$ (donor
excitation, donor emission), $I_{DA}$ (donor excitation, acceptor emission: the "FRET
channel"), $I_{AA}$ (acceptor excitation, acceptor emission). Bleed-through coefficient $d$,
cross-excitation coefficient $a$ (the literature also writes $\alpha, \beta$; we avoid $\beta$,
which the course reserves). Ratio $R = I_A/I_D$; noise standard deviation $\sigma$; photon
counts $N$.

## Prerequisite gap

- **Fluorescence basics** (excitation and emission spectra, Stokes shift, filters, bands): the
  colour day's `colour-vision/` ("from a spectrum to three numbers": a channel integrates a
  spectrum through a band) and, if taken, `crosstalk/`. Unit 1 adds one slide on the
  **excitation (absorption) spectrum** next to the emission spectrum (the unmixing day used
  emission only) and on quantum yield as "the fraction of excited molecules that emit".
- **Bleed-through, the mixing matrix, single-stained controls, slope through the origin,
  under/over-compensation**: owned by `crosstalk/` (spectral unmixing day). Unit 2 recalls
  it in **one** slide plus a two-line "if you have not taken the unmixing day" box (a channel
  sees the band; a control measures a column; the slope through the origin estimates it), so
  the day stands alone after day 42.
- **Least squares, condition number** (`lsunmix/`): recalled in one bullet for spectral FRET
  (optional unit) and for "subtraction amplifies noise".
- **Poisson shot noise, read noise, offset, SNR**: `cameranoise/` (day 3), recalled in one line
  each; the ratio-noise material builds on "variance = mean" for photon counts.
- **Background subtraction before a ratio**: `intensity-features/` (measurements day 16) has the
  slide "a ratio of two means cancels gain but not offset". Day 16 may not have been taken
  (entry is after day 42), so unit 4 restates it in one slide with a back-reference.
- **Colour maps, lightness, diverging maps**: `colour-display/` (day 42); used for ratio and
  efficiency maps and for the intensity-modulated display (hue = ratio, lightness = intensity).
- **Image alignment** (channel shift, a cell that moved between two images): owned by FFT
  registration (day 23, `phasecorr/`); a pointer here ("align before dividing"), since day 23
  may come later.
- **Denoising before measuring** (`denoise/`, "what may be measured"): smoothing/binning before
  a ratio is allowed because both channels get the same linear filter; one-line recall.
- **Resolution** (`resolution/`, day 4): one line, "two dyes in one diffraction-limited spot can
  be 200 nm apart; FRET needs < 10 nm".

## Overlaps / who owns what

| Topic | Owner | Here |
|---|---|---|
| Linear mixing, bleed-through, compensation, single-stained controls | Spectral unmixing, `crosstalk/` | Recall (one slide); the three-cube correction is taught **as** compensation with a fixed, lower-triangular structure (course map ownership change) |
| Least squares unmixing, conditioning | Spectral unmixing, `lsunmix/` | Recall; spectral FRET as an application (optional unit 5) |
| Non-linear mixing (FRET bends the cloud) | `endmembers/` has one bullet | This day explains why: donor emission is lost, acceptor emission gained |
| Lifetimes, TCSPC/frequency domain, exponential fitting, phasors, **FLIM-FRET** | FLIM day (parallel plan) | Pointer: one slide in unit 3 ("$E = 1 - \tau_{DA}/\tau_D$; lifetimes separate efficiency from fraction bound"), plus the rate picture in unit 1's math layer as the bridge. The unmixing day's optional `phasor/` already became a recall of the FLIM day |
| Colocalisation (Pearson, Manders, Costes, objects) | Colocalisation day (takes over `coloc/`) | One contrast slide in unit 1: colocalisation = same diffraction-limited spot (~200 nm); FRET = within ~10 nm. Bleed-through fakes both |
| Noise models, SNR | `cameranoise/` | Used. **This day owns noise propagation through ratios and differences of images** (delta method, ratio bias, heavy tails); the evaluation day may recall it |
| Offset/gain and ratios of intensities | `intensity-features/` | Restated in one slide (background before ratio) with a back-reference |
| Image alignment | FFT registration (`phasecorr/`) | Pointer |
| Colour maps, display | `colour-display/` | Used for ratio maps; the intensity-modulated display is taught here (one slide) |
| Thresholding a mask | Thresholding (day 12) | Used for the ratio mask (one line) |
| Photobleaching kinetics | (no owner) | Taught here as much as needed: an exponential loss per exposure; a control region measures it |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`fret-physics/`, `sensitised-emission/`, `acceptor-bleach/`, `ratiometric/`; optional
`fret-stoichiometry/`. (Folder names checked against the built folders: no collisions.)

### 1. `fret-physics/`: FRET in pictures: a ruler for nanometres

Concept layer:

- Motivation (picture first): two proteins that "colocalise" in a merge may be 200 nm apart or
  bound to each other; the microscope cannot tell (recall `resolution/` in one line). We want a
  signal that switches on only when two molecules **touch**.
- **The cartoon**: donor absorbs blue light, normally emits cyan; with an acceptor within a few
  nm, the energy jumps across **without a photon** (near-field dipole coupling, in words) and
  the acceptor emits yellow. Two observable consequences: **donor quenching** (donor dimmer,
  shorter lifetime) and **sensitised emission** (acceptor emission under donor excitation).
  Picture: the two spectra pairs (donor excitation/emission, acceptor excitation/emission) on one
  wavelength axis; the **overlap** of donor emission and acceptor excitation shaded.
- What FRET needs: (1) overlap of donor emission with acceptor absorption; (2) distance
  $\lesssim 10$ nm; (3) dipoles not perpendicular. Excitation spectra appear for the first time in
  the course (the unmixing day only needed emission): one slide, with quantum yield as "the
  fraction of excited molecules that emit a photon".
- **The efficiency curve** $E(r) = 1/(1 + (r/R_0)^6)$: half at $r = R_0$, almost all at
  $0.5R_0$ (0.985), almost nothing at $2R_0$ (0.015). The **sensitive window**
  $0.5R_0$–$1.5R_0$ (0.98 → 0.08): outside it, $E$ says only "close" or "far". Typical $R_0$:
  4–7 nm for fluorescent-protein pairs. Picture: the curve with the window shaded, a protein
  drawn to scale (≈ 4 nm across) above the $r$ axis.
- **Why the sixth power** (in words): dipole–dipole coupling falls as $1/r^3$; transfer rate
  goes as the coupling squared. The steepness is what makes FRET a switch (bound/unbound) more
  than a ruler.
- **What sets $R_0$**: more overlap, brighter donor (quantum yield) → larger $R_0$; orientation
  $\kappa^2$ (0 to 4; 2/3 when the dyes tumble freely); refractive index of the medium. Only the
  sixth root enters, so a factor 2 in overlap moves $R_0$ by 12 %. But $\kappa^2$ is usually
  unknown: from 2/3 to 4 moves $R_0$ by 35 %, to 0.1 by −27 %. Absolute distances are therefore
  soft; changes in $E$ are hard.
- **From efficiency to apparent distance**: $r = R_0 (1/E - 1)^{1/6}$. Worked example: $R_0 = 5$
  nm, $E = 0.30 \pm 0.05$ → $r = 5.76$ nm (5.54–6.00): $r$ is insensitive to errors in $E$ inside
  the window, extremely sensitive outside it.
- **A mixture of distances**: the average of $E$ over a population is not $E$ of the average
  distance (the curve is non-linear). Example: half at $0.5R_0$, half at $R_0$ → mean $E$ = 0.74
  → apparent $r$ = $0.84R_0$, while the mean distance is $0.75R_0$. (Jensen in words; links to
  "averages of non-linear functions" warnings elsewhere in the course.)
- **What FRET is used for in imaging** (one slide): binding of two proteins (intermolecular),
  conformational change of one protein (intramolecular, biosensors: unit 4), cleavage
  (FRET lost). Three ways to measure it: sensitised emission (unit 2), donor recovery after
  acceptor photobleaching (unit 3), donor lifetime (FLIM day). Each slide of the day returns to
  this map.
- **Colocalisation vs FRET** (one slide, forward pointer to the colocalisation day): same
  diffraction-limited spot vs nanometre contact; bleed-through fakes both.
- Pointer: one line on homo-FRET (identical donor and acceptor, seen only in polarisation).

Math layer (`## The math: … {.math}` slides):

- *Rates*: donor decays with rate $k_D = 1/\tau_D$; transfer adds $k_T = k_D (R_0/r)^6$.
  $E = k_T/(k_T + k_D) = 1/(1 + (r/R_0)^6)$. Donor lifetime with acceptor
  $\tau_{DA} = 1/(k_D + k_T)$, hence $E = 1 - \tau_{DA}/\tau_D = 1 - F_{DA}/F_D$ (the donor
  intensity is proportional to its lifetime at fixed excitation). This is the bridge to the
  FLIM day.
- *Förster radius*: $R_0^6 = \dfrac{9\ln 10\;\kappa^2 Q_D J}{128\pi^5 n^4 N_A}$ with the overlap
  integral $J = \int F_D(\lambda)\,\varepsilon_A(\lambda)\,\lambda^4\,d\lambda$ ($F_D$ normalised
  to unit area); in practical units $R_0 = 0.0211\,(\kappa^2 n^{-4} Q_D J)^{1/6}$ nm with $J$ in
  M⁻¹cm⁻¹nm⁴. The $\lambda^4$ weight favours red overlap.
- *Orientation*: $\kappa^2 = (\hat{\boldsymbol\mu}_D\cdot\hat{\boldsymbol\mu}_A -
  3(\hat{\boldsymbol\mu}_D\cdot\hat{\mathbf r})(\hat{\boldsymbol\mu}_A\cdot\hat{\mathbf r}))^2$:
  4 for collinear dipoles along $\mathbf r$, 1 for parallel dipoles side by side, 0 possible;
  isotropic average 2/3.
- *Sensitivity*: $dE/dr = -\tfrac{6}{r}E(1-E)$, largest at $r = R_0$ (slope $-1.5/R_0$); inverse
  $dr/dE = -\tfrac{r}{6E(1-E)}$, which explodes near $E = 0$ and $E = 1$ (the window).

**Demo `fretruler`** (new): left, two molecules on a 2D canvas (donor cyan, acceptor yellow,
drawn to scale with a 1 nm grid; drag the acceptor; their dipole arrows rotatable by dragging
the arrow tips); top right, the four parametric spectra (donor excitation/emission, acceptor
excitation/emission; shapes from `lib/unmix.ts`'s split Gaussian, colours from
`wavelengthToSrgb`), overlap shaded, acceptor spectra draggable along λ; bottom right, the
$E(r)$ curve with the current point and the sensitive window, and two bars: donor emission
(quenched) and acceptor emission (sensitised) under donor excitation. Readouts: $r$, $R_0$
(from overlap, $Q_D$, $\kappa^2$), $\kappa^2$, $E$, apparent $r$ assuming $\kappa^2 = 2/3$.
Toggle *population*: 200 seeded molecules with a distance distribution (mean, width sliders)
and random or fixed orientations; shows the histogram of $r$, the mean $E$ and the apparent
distance vs the mean distance. Presets: "r = R₀: half", "twice R₀: almost nothing", "half R₀:
almost everything", "the sensitive window", "poor overlap: a smaller R₀", "perpendicular
dipoles: κ² = 0", "κ² = 4: the same distance, more FRET", "a mixture of two distances: mean E ≠
E(mean r)". Fits 1150×560 (canvas 420 px + two stacked plots); labels ≥ 18 px on slides.

By hand (concept): (a) $E$ at $r/R_0$ = 0.5, 0.8, 1, 1.2, 1.5, 2 (0.985, 0.79, 0.5, 0.25,
0.08, 0.015); (b) apparent distance for $E$ = 0.30 and its range for $\pm 0.05$ with $R_0 = 5$
nm; (c) how much $R_0$ changes if the overlap doubles, if $\kappa^2$ is 4 instead of 2/3;
(d) a population half at $0.5R_0$, half at $R_0$: mean $E$, apparent and mean distance;
(e) predict from the spectra plot which of two acceptor candidates gives the larger $R_0$.
*(math)*: (f) derive $E = 1 - \tau_{DA}/\tau_D$ from the rates; (g) $dE/dr$ and where it peaks;
(h) $\kappa^2$ for three given dipole geometries; (i) show the isotropic average of $\kappa^2$ is
2/3 (integral over the sphere; optional).

### 2. `sensitised-emission/`: Three images, two controls: correcting sensitised emission

Concept layer:

- Picture first: the raw "FRET channel" $I_{DA}$ (donor excitation, acceptor filter) of a field
  with a donor-only cell, an acceptor-only cell, a cell with free donor and acceptor (no FRET)
  and a cell with a donor–acceptor fusion (FRET): **all four light up**. Two reasons, one of
  them FRET.
- **Recall `crosstalk/`** (one slide): a channel integrates a band; a dye's tail bleeds into
  the neighbour's channel; a single-stained control measures a column of the mixing matrix as
  a slope through the origin. Box for students who skipped the unmixing day.
- **The three images**: $I_{DD}$ (donor excitation, donor emission), $I_{DA}$ (donor excitation,
  acceptor emission), $I_{AA}$ (acceptor excitation, acceptor emission). Picture: the 3 ×
  excitation/emission grid with the spectra, each cell of the grid showing what it collects.
- **Two contaminations of $I_{DA}$**: donor emission tail in the acceptor filter
  (**bleed-through**, $d \cdot I_{DD}$) and acceptor excited directly by the donor laser
  (**cross-excitation**, $a \cdot I_{AA}$). What remains is the sensitised emission
  $F_c = I_{DA} - d\,I_{DD} - a\,I_{AA}$.
- **The controls**: a donor-only sample gives $d$ as the slope of $I_{DA}$ against $I_{DD}$; an
  acceptor-only sample gives $a$ as the slope of $I_{DA}$ against $I_{AA}$ (both through the
  origin after background subtraction, as on the crosstalk day). Same microscope, same settings
  as the experiment: coefficients belong to an instrument configuration, not to a dye pair.
- **This is compensation with a structure**: written as a matrix, the three images are a
  lower-triangular mix of three unknowns (donor signal, FRET signal, directly excited acceptor
  signal); solving it is the $M^{-1}$ of `crosstalk/` with most entries known to be zero. The
  same check without ground truth: the **no-FRET control** (free donor + acceptor, or a mixture
  of the two single-labelled samples) must give $F_c \approx 0$ for every expression level.
- **Wrong coefficients make false FRET**: too small a $d$ leaves a residual proportional to the
  donor brightness; picture: $F_c$ vs $I_{DD}$ scatter in the no-FRET cell tilts (under/over as
  on the crosstalk day).
- **From $F_c$ to a number we can compare**: $F_c$ grows with how much donor and acceptor are
  there. Normalisations (one slide, honestly): divide by $I_{AA}$ (per acceptor), by
  $\sqrt{I_{DD} I_{AA}}$ (a symmetric index), or convert to an efficiency
  $E_\text{app} = F_c / (F_c + G\,I_{DD})$ with an instrument factor $G$ = "how much sensitised
  acceptor signal one unit of lost donor signal produces". Only the last one is an efficiency;
  the indices change with the donor:acceptor ratio even at fixed binding (demo preset). How to
  get $G$: from two constructs of known, different efficiency, or from acceptor photobleaching
  on the same cells (unit 3) (details in optional unit 5).
- **Noise: subtraction of two large numbers** (concept, with numbers): $F_c$ is a small
  difference of bright images, so its relative noise is large when $E$ is small or the
  coefficients are large. Forward pointer to unit 4's noise slides; the same message as
  `lsunmix/`'s "similar spectra amplify noise".
- Practical pitfalls (one slide each, short): background and offset first (an offset is a
  fake component); saturation breaks the linear model; acceptor bleaching during the three
  acquisitions; the three images must be aligned (a chromatic shift produces edges in $F_c$,
  pointer to FFT registration); expression level matters (overexpression gives FRET by
  crowding, "bystander FRET").
- Pointer: with a spectral detector, the same separation is done by least squares on the
  emission spectrum under donor excitation (spectral FRET, optional unit 5; `lsunmix/`).

Math layer:

- *The linear model*: with donor amount $N_D$, acceptor amount $N_A$, efficiency $E$ for the
  paired fraction and instrument constants $c_{\cdot}$:
  $I_{DD} = c_{DD} N_D (1 - E)$, $I_{AA} = c_{AA} N_A$,
  $I_{DA} = d\,I_{DD} + a\,I_{AA} + F_c$ with $F_c = G\,c_{DD} N_D E$. Hence
  $E = F_c/(F_c + G I_{DD})$ and the donor signal without FRET is $I_{DD} + F_c/G$.
- *As a matrix*: $(I_{DD}, I_{DA}, I_{AA})^\top = L\,(u, F_c, v)^\top$ with $L$ lower-triangular
  (rows: $(1,0,0)$, $(d,1,a)$, $(0,0,1)$); forward substitution is the inverse.
- *Noise*: independent Poisson images, so
  $\operatorname{Var}(F_c) \approx \operatorname{Var}(I_{DA}) + d^2\operatorname{Var}(I_{DD}) +
  a^2\operatorname{Var}(I_{AA})$ (plus the uncertainty of $d$ and $a$, which multiplies the
  bright images: a systematic error, not averaging out).
- *Normalised indices* and why they depend on stoichiometry: at fixed $E$ and paired fraction,
  $F_c/I_{AA} \propto N_D/N_A$ and $F_c/\sqrt{I_{DD}I_{AA}} \propto \sqrt{N_D/N_A}$.

**Demo `threecube`** (new, main demo of the day): a seeded field of 8–12 cells (shapes from
`lib/synth.ts` `nuclei`, used as whole cells, with smooth intracellular variation) in four
classes: donor only, acceptor only, free donor + acceptor (no FRET), donor–acceptor fusion
(true $E$ slider), and optionally "binding" cells (a fraction paired). Expression levels vary
per cell; Poisson noise from a photons slider; background offset. Panels: the three raw images
($I_{DD}$, $I_{DA}$, $I_{AA}$) as small tiles; the corrected image (choice: $F_c$, $F_c/I_{AA}$,
$F_c/\sqrt{I_{DD}I_{AA}}$, $E_\text{app}$; diverging map, 0 = mid-grey); a scatter (density-shaded)
of the current control: $I_{DA}$ vs $I_{DD}$ for donor-only pixels, or $I_{DA}$ vs $I_{AA}$ for
acceptor-only pixels, with the user's slope line; a per-cell table (class, mean $E_\text{app}$
or index, true value). Sliders $d$, $a$, $G$, background subtracted; button *measure from the
controls* (slope through the origin, `slopeThroughOrigin` from `lib/unmix.ts`). Toggles: subtract
bleed-through, subtract cross-excitation, misregister $I_{DA}$ by 1 px. Presets: "the raw FRET
channel: everything lights up", "remove bleed-through only", "remove cross-excitation too",
"measured from the controls", "the no-FRET control is zero", "d too small: false FRET follows
the donor", "low E, few photons: the difference is noise", "indices change with the
donor:acceptor ratio, E does not", "one pixel shift: edges", "forgot the background". Fits
1150×560 (three 150 px tiles + 300 px result + 300 px scatter; the table on the exercise page
via `data-layout="full"`).

By hand (concept): (a) estimate $d$ from three donor-only pixels $(I_{DD}, I_{DA})$ = (200, 98),
(400, 205), (800, 398) → 0.50, and $a$ from three acceptor-only pixels $(I_{AA}, I_{DA})$ =
(500, 52), (1000, 99), (1500, 149) → 0.10; (b) a pixel with $I_{DD} = 400$, $I_{DA} = 500$,
$I_{AA} = 1000$: $F_c = 200$, $E_\text{app} = 0.20$ for $G = 2$, the index
$F_c/\sqrt{I_{DD}I_{AA}} = 0.32$; (c) the same pixel with $d = 0.4$ by mistake: $E_\text{app} =
0.23$; a no-FRET pixel with $I_{DD} = 1000$ then shows false $F_c = 100$; (d) Poisson noise of
$F_c$ in (b): $\sqrt{500 + 0.25\cdot400 + 0.01\cdot1000} \approx 24.7$ (12 %), and for a pixel
with $F_c = 20$ at the same brightness; (e) predict which normalisation changes when the
acceptor is expressed at twice the level. *(math)*: (f) derive $E = F_c/(F_c + G I_{DD})$ from
the linear model; (g) write $L$ and $L^{-1}$; (h) show the index dependence on $N_D/N_A$;
(i) derive $\operatorname{Var}(F_c)$ and the extra error from an uncertain $d$.

### 3. `acceptor-bleach/`: Acceptor photobleaching, and what an efficiency means

Concept layer:

- **The idea**: destroy the acceptor in a region with strong acceptor-only light; the donor no
  longer transfers energy and gets **brighter** (dequenching). $E = 1 - I_D^\text{pre}/
  I_D^\text{post}$. One sample, no bleed-through coefficients, self-calibrating: the
  attraction of the method. Picture: donor image before and after, the bleached rectangle
  brighter, a time course of donor (up) and acceptor (down) in the region.
- Worked example: donor 600 before, 800 after → $E = 0.25$.
- **Corrections**, each with its own control:
  - **incomplete bleaching** (acceptor only down to 20 %): $E$ underestimated; measured from the
    acceptor image; correction formula (math layer; example 0.25 → 0.29);
  - **donor bleaching** during imaging and by the bleach light: a non-bleached region of the same
    cell or another cell drifts down; divide by its loss (example: control −5 % → $E$ = 0.29);
  - **motion and drift** between before and after: edges of the cell light up in the $E$ map
    (pointer to FFT registration: align first);
  - **acceptor photoconversion** into a donor-like species (one line: some acceptors become
    donor-coloured when bleached, mimicking dequenching; control: acceptor-only cells must show
    no donor increase).
- **End-point method**: one measurement per cell, the acceptor is gone; slow for live cells
  (bleaching takes seconds, molecules diffuse into the region: the unbleached fraction comes
  back). Comparison table of the three methods (sensitised emission / acceptor bleaching /
  lifetime): what each needs, live or fixed, speed, what it is sensitive to.
- **Pixel-wise vs region-wise $E$**: $1 - \text{pre}/\text{post}$ per pixel is a ratio of two
  noisy numbers (forward pointer to unit 4): the $E$ map is noisy and biased in dim pixels;
  averaging pre and post over the region first is better (ratio of sums). Example: 600 and 800
  photons per pixel → $\sigma_E \approx 0.04$ per pixel.
- **What an efficiency means: apparent $E$** (the second half of the unit, the pitfall slides
  of the day): every intensity method measures $E_\text{app} = f_{DA}\,E$, the efficiency of the
  paired donors times the **fraction of donors that have an acceptor**. 50 % bound at $E = 0.4$
  looks like 100 % bound at $E = 0.2$, and the distance computed from it is wrong ($1.26R_0$
  instead of $1.07R_0$). Lifetime methods separate the two (two donor populations: two
  lifetimes; **FLIM day**, one slide: $E = 1 - \tau_{DA}/\tau_D$, phasors for mixtures).
- **Checklist for a distance claim** (one slide): single well-defined pair, $\kappa^2$
  justified, $f_{DA}$ known, $E$ inside the window, corrections and controls listed. Usually
  the honest statement is a change in $E$, not a distance.

Math layer:

- *Incomplete bleaching*: if a fraction $b$ of acceptors is bleached (all donors paired), the
  donor after is $\propto 1 - (1-b)E$, so $E_\text{obs} = bE/(1 - (1-b)E)$ and
  $E = E_\text{obs}/(b + (1-b)E_\text{obs})$.
- *Fraction bound*: donors are a mixture, intensity $\propto (1 - f_{DA}) + f_{DA}(1 - E)$, so
  $E_\text{app} = f_{DA}E$ exactly for intensity methods; the lifetime decay is a sum of two
  exponentials (pointer).
- *Noise of $E = 1 - P/Q$*: $\sigma_E \approx (P/Q)\sqrt{1/P + 1/Q}$ for Poisson counts
  (delta method, derived in unit 4).

**Demo `apbleach`** (new): a seeded cell (or two: a FRET cell and an acceptor-only control
cell) with donor and acceptor channels; drag a rectangle as the bleach region; *bleach* button
advances bleach pulses (acceptor in the region decays exponentially per pulse, donor
dequenches; a donor bleaching rate applies to the whole field). Panels: donor before, donor
after, $E$ map (pixel-wise, with a smoothing slider), time course plot of donor and acceptor
in the bleached region and in a control region. Sliders: true $E$, fraction paired $f_{DA}$,
pulses (→ bleach completeness), donor bleaching per frame, drift between before and after
(px), photons. Toggles: correct for incomplete bleaching, correct for donor bleaching, align
(integer shift by phase correlation from `lib/registration.ts`). Readouts: region $E$
(ratio of sums) vs mean of pixel $E$, true $E$ and $f_{DA}E$, acceptor remaining. Presets:
"ideal", "stop early: incomplete bleaching", "donor bleaching fakes less FRET", "the cell
moved: edges", "few photons: noisy map, the region mean is fine", "half the donors paired",
"acceptor-only control shows nothing". Fits 1150×560.

By hand (concept): (a) $E$ from pre/post 600/800; (b) correct for 80 % acceptor bleaching
(→ 0.294); (c) correct for 5 % donor loss in the control (→ 0.287); (d) the per-pixel
$\sigma_E$ at 600/800 photons and at 60/80; (e) a population with $E_\text{app} = 0.2$: two
interpretations and their apparent distances for $R_0 = 5$ nm; (f) fill in the comparison table
for a live-cell time-lapse question. *(math)*: (g) derive the incomplete-bleaching correction;
(h) show $E_\text{app} = f_{DA}E$ for intensity methods; (i) derive $\sigma_E$.

### 4. `ratiometric/`: Ratio imaging, biosensors and the noise of a ratio

Concept layer:

- **Biosensors** (picture first): one protein chain carrying donor and acceptor; binding of an
  ion or a phosphorylation changes the shape, the distance, and so $E$. Because donor and
  acceptor come in a fixed 1:1 ratio, the simple ratio $R = I_{DA}/I_{DD}$ (acceptor over donor
  under donor excitation) reports $E$ without the full three-image correction. Only true for
  single-chain sensors; two-molecule FRET needs unit 2. One slide also on single-dye ratio
  sensors (two excitation or emission bands of one dye that shift with pH or ion binding): the
  same arithmetic.
- **Why ratio**: the intensity of either channel depends on sensor concentration, cell
  thickness, illumination and focus; the ratio cancels all of them (they multiply both channels).
  Picture: a flat cell with a thick nucleus region looks "active" in intensity, uniform in ratio.
- **Background before dividing** (restated from `intensity-features/`): an offset does not
  cancel; $(A + b)/(D + b)$ moves towards 1. Example: $A = 100$, $D = 400$, $b = 50$ → 0.33
  instead of 0.25. Outside the cell both channels are background: the ratio of two noise values
  is meaningless (and explodes) → **mask** with a threshold on the sum (day 12, one line).
- **Bleed-through floor**: donor bleed-through adds $d$ to the ratio at zero FRET; it does not
  change the shape of the response but shifts $R_\text{min}$ (one line; calibration absorbs it).
- **Display**: a ratio map in a colour map, with the dim (unreliable) pixels shown dark:
  **intensity-modulated display** (hue = ratio, lightness = intensity; `colour-display/` for
  lightness-monotone maps). Picture: plain ratio map (noisy, speckled background) vs IMD.
- **Calibration**: $R$ between $R_\text{min}$ (no ligand) and $R_\text{max}$ (saturated) maps to a
  concentration through the binding curve with dissociation constant $K_d$
  ($c = K_d\,(R - R_\text{min})/(R_\text{max} - R)$, the simplified form with equal brightness
  at the denominator wavelength). Picture: the curve; near $R_\text{max}$ a small ratio error is a
  huge concentration error (example with $K_d = 200$ nM, $R_\text{min} = 0.5$, $R_\text{max} =
  2.5$: $R = 1.5$ → 200 nM, 2.3 → 1800 nM, 2.4 → 3800 nM). Dynamic range $R_\text{max}/
  R_\text{min}$ as the sensor's quality.
- **Time-lapse pitfalls**: the two fluorophores bleach at different rates → the ratio drifts
  (correct with a no-stimulus control or a fitted drift); channels taken one after the other
  while the cell moves → edge artefacts (simultaneous two-camera or aligned images; pointer to
  FFT registration); pH and temperature change the dyes.
- **The noise of a ratio** (the statistical heart of the unit):
  - with photon counts, $\mathrm{CV}(R)^2 \approx 1/N_A + 1/N_D$: the **dimmer channel
    dominates**. Example: 100 and 400 photons → 11 % per pixel; binning 4×4 → 2.8 %;
  - a ratio is **biased** upwards when the denominator is noisy ($E[A/D] \approx (A/D)(1 +
    \mathrm{CV}_D^2)$), and has **heavy tails** when the denominator can be near zero (histogram
    of pixel ratios: a long right tail);
  - **ratio of sums vs mean of ratios** (recall `crosstalk/`'s exercise: the slope through the
    origin vs the mean of ratios): for a region, divide the summed channels; averaging pixel
    ratios gives dim pixels the same weight as bright ones and inherits the bias;
  - smooth or bin **both** channels with the same linear filter before dividing (allowed:
    `denoise/`'s "what may be measured"), never the ratio image with a median after the fact
    without saying so;
  - $\log R$ is symmetric (a doubling and a halving are equally far from 1) and its noise is
    closer to Gaussian: plot log ratios.
  - The same propagation explains unit 2 ($F_c$ is a difference: variances add) and unit 3 ($E =
    1 - P/Q$).
- **Pitfall summary for the day** (one slide, the list a reviewer would check): background,
  saturation, bleed-through and cross-excitation coefficients from the same settings, controls
  (donor-only, acceptor-only, no-FRET, positive construct), alignment, bleaching, expression
  level / bystander FRET, apparent $E$ vs fraction bound, $\kappa^2$ for distances, ratio of sums,
  masks.

Math layer:

- *Delta method*: for $R = A/D$ with means $\mu_A, \mu_D$:
  $\operatorname{Var}(R) \approx (\mu_A/\mu_D)^2\big[\sigma_A^2/\mu_A^2 + \sigma_D^2/\mu_D^2 -
  2\operatorname{Cov}(A,D)/(\mu_A\mu_D)\big]$; bias
  $E[R] \approx (\mu_A/\mu_D)(1 + \sigma_D^2/\mu_D^2 - \operatorname{Cov}(A,D)/(\mu_A\mu_D))$
  (second-order Taylor expansion of $A/D$). Poisson: $\sigma^2 = \mu$, giving
  $\mathrm{CV}^2 = 1/\mu_A + 1/\mu_D$. Positive correlation (e.g. from shared illumination
  fluctuations) **reduces** ratio noise: why ratioing removes flicker.
- *Heavy tails*: the ratio of two independent zero-mean Gaussians is Cauchy (no mean, no
  variance): stated, as the extreme case of a denominator near zero.
- *Calibration*: derive $c = K_d (R - R_\text{min})/(R_\text{max} - R)$ from a single binding site
  (fraction bound $c/(c + K_d)$) and $R$ linear in the fraction bound (equal-brightness
  assumption; the general form has an extra factor for the brightness change of the
  denominator channel); its derivative $dc/dR = K_d (R_\text{max} - R_\text{min})/(R_\text{max} -
  R)^2$ shows the explosion near $R_\text{max}$.
- *Binning*: summing $m$ pixels multiplies counts by $m$ and divides $\mathrm{CV}^2$ by $m$.

**Demo `ratioimage`** (new): a seeded time-lapse (40 frames, 128×128) of two or three cells
expressing a single-chain sensor, with varying thickness and expression (so intensities lie),
and a signal (a wave crossing one cell, a steady gradient in another, nothing in a control
cell). Two channel tiles, the ratio map (plain or intensity-modulated), a time plot of the
ratio in two clickable regions (ratio of sums and mean of pixel ratios, both drawn), a
calibration panel ($R$ → concentration curve with $R_\text{min}$, $R_\text{max}$, $K_d$ handles
and the current region's value with an error bar). Controls: frame slider + play, photons,
background offset and *subtract background* toggle, mask threshold, binning/smoothing before
the ratio, bleed-through, channel shift (px), acceptor bleaching rate, display mode. Presets:
"intensity lies: thick cells look active", "the ratio cancels thickness", "forgot the
background", "no mask: the background explodes", "intensity-modulated display", "few photons:
bin first", "one channel shifted by a pixel", "acceptor bleaches: the ratio drifts",
"calibrate: near R_max the error explodes". Fits 1150×560 (two 180 px tiles + 280 px ratio map +
side plots stacked).

**Demo `rationoise`** (new, small, numeric): Monte Carlo of a ratio of two Poisson (or Gaussian)
counts with seeded draws. Sliders: mean counts $N_A$, $N_D$ (log scales), correlation (shared
illumination fluctuation), pixels per region, binning. Panels: histogram of pixel ratios (or of
$\log R$) with the true ratio, the delta-method prediction (normal curve) and the observed mean;
bar readouts: true ratio, mean of ratios, ratio of sums, CV observed vs predicted. Tab *difference*
for the three-cube $F_c$: histogram of $F_c$ for given $I_{DA}$, $I_{DD}$, $I_{AA}$, $d$, $a$, with
the variance formula. Tab *bleach*: $E = 1 - P/Q$. Presets: "bright: the formula works", "dim
denominator: bias and a long tail", "log ratio is symmetric", "ratio of sums beats mean of
ratios", "shared flicker cancels", "low FRET: the corrected signal is mostly noise". Pure TS,
10⁵ draws per update is a few ms.

By hand (concept): (a) $R$ with and without background subtraction (100, 400, offset 50);
(b) CV of $R$ at 100/400 photons, after 4×4 binning, and which channel to make brighter;
(c) bias of $A/D$ when $D$ has 10 photons on average; (d) a region with four pixels given as
$(A, D)$: ratio of sums vs mean of ratios, and which is closer to the truth; (e) calibrate $R$ =
1.5, 2.3, 2.4 with the numbers above and the concentration error for $\pm 0.05$ in $R$ at each;
(f) predict what a 1-pixel shift between channels does to the ratio at a cell edge.
*(math)*: (g) derive the delta-method variance and bias of $A/D$; (h) derive the calibration
equation and $dc/dR$; (i) show that a common multiplicative fluctuation cancels in $R$ and
appears as positive covariance in the formula.

### 5. (optional / further reading) `fret-stoichiometry/`: Calibrating $G$, stoichiometry and spectral FRET

- **Measuring $G$**: two constructs with different, fixed efficiency (or the same cells before and
  after acceptor bleaching): $G = \Delta F_c / \Delta I_{DD}$ (sensitised signal gained per donor
  signal lost); a straight-line fit through several constructs.
- **Stoichiometry**: with an acceptor-excitation factor $k$, the donor:acceptor ratio
  $S = (G I_{DD} + F_c)/(G I_{DD} + F_c + I_{AA}/k)$; the **$E$–$S$ histogram** separates
  donor-only, acceptor-only and doubly labelled pixels or cells (the cloud picture of
  `endmembers/` with physics on the axes); alternating excitation as the idea behind it (no
  instrument names).
- **Binding curves** (two-molecule FRET): $E_\text{app}$ against the acceptor:donor ratio
  saturates at the efficiency of the complex; fitting it gives $E$ and an apparent affinity.
  Bystander FRET as the straight-line component at high concentration.
- **Spectral FRET**: with a spectral detector, unmix the emission under donor excitation into
  donor and acceptor contributions by least squares (`lsunmix/` recall: $M$ has two columns,
  donor and acceptor emission shapes); the acceptor part minus its direct excitation (from a
  second excitation) is $F_c$; bleed-through is handled by the unmixing, not by a coefficient.
- **Demo**: `threecube` with `data-mode="es"` (the $E$–$S$ scatter of cells/pixels, $G$ and $k$
  sliders, a set of constructs to calibrate $G$ from), and `data-mode="spectral"` reusing the
  `lsunmix` solver from `lib/unmix.ts`.
- By hand: $G$ from two constructs; $S$ for three given pixels; a 3-channel × 2-component spectral
  FRET unmixing.

## Shared engineering

- **Reuse** (no local copies; build-issues lesson):
  - `lib/unmix.ts` (owner: unmixing day): `Dye`/`emission` (split-Gaussian spectra), `Band`,
    `bandFraction`, `channelMatrix` for the filter/excitation grid; `slopeThroughOrigin`,
    `pearson`, `median`, `blurMap`; `lsPixel`/`pseudoInverse` for spectral FRET (optional unit);
  - `lib/spectra.ts` (owner: colour day): `wavelengthToSrgb` for spectrum colours;
  - `lib/random.ts`: `rng`, `poisson`, `gauss`, `exponential` (fixed seeds);
  - `lib/imageops.ts`: `Grey`, `gaussianBlur`, `boxDown2` (binning), `translate` (channel shift),
    display helpers; `lib/synth.ts`: `nuclei` for cell shapes, `addPoissonNoise`;
  - `lib/registration.ts`: `correlate2d` + `findPeaks` for the *align* toggle in `apbleach` and
    `ratioimage`;
  - `lib/plot.ts`: $E(r)$ curve, spectra (`line`, `band` for the overlap), scatter, histograms,
    time plots, `colorbar`; `lib/histogram.ts`: `histogram2d` for density-shaded scatters;
  - `lib/colour.ts`: `diverging`/`divergingT` (signed $F_c$), `viridis`, `hsvToRgb` (intensity-
    modulated display), `scalarToRGBA`, `drawScalarImage`;
  - `lib/stats.ts`: `mean`, `sd`, `quantile`, `linfit` (calibration lines, $G$ fit);
  - `lib/label.ts`: per-cell statistics in `threecube`; `lib/threshold.ts`: Otsu for the ratio
    mask; `lib/ui.ts`: sliders, presets, `autoMount`.
- **New lib file owned by this day: `demos/src/lib/fret.ts`** (no collision with existing
  files; no DOM). Sections:
  - physics: `efficiency(r, R0)`, `distanceFromE(E, R0)`, `dEdr`, `kappa2(muD, muA, rhat)`,
    `forsterRadius(QD, kappa2, n, J)`, `overlapIntegral(donorEm, accAbs)` on a wavelength grid,
    a parametric **absorption** shape for a `Dye` (mirror-image rule, so `lib/unmix.ts` stays
    unchanged), rate form (`lifetimeWithAcceptor(tauD, E)`) shared with the FLIM day;
  - sensitised emission: forward model `threeCube({ND, NA, E, fPaired, d, a, G, cDD, cAA})`
    → $(I_{DD}, I_{DA}, I_{AA})$; `correctFc`, `eFret(Fc, IDD, G)`, `nfretIndex`,
    `perAcceptorIndex`, `stoichiometry`, `fcVariance`;
  - acceptor photobleaching: `bleachSeries(...)` (exponential acceptor loss per pulse, donor
    dequenching, donor bleaching), `eFromBleach(pre, post)`, `correctIncomplete(Eobs, b)`,
    `correctDonorBleach`;
  - ratios: `ratioOfSums`, `meanOfRatios`, `ratioCVPredicted(muA, muD, cov)`,
    `ratioBiasPredicted`, `calibrate(R, Rmin, Rmax, Kd)`, `calibrateInverse`, `dcdR`;
  - seeded scenes: `fretField(opts)` (cells of the four classes for `threecube`),
    `sensorTimeLapse(opts)` (thickness, expression, signal wave, bleaching for `ratioimage`),
    `bleachCell(opts)` for `apbleach`. Cell outlines come from `synth.nuclei`; if the
    second-pass consolidation moves scene generators into `lib/synth.ts`, these move with
    them.
- Speed: everything is per-pixel arithmetic on ≤ 256² images (sub-ms to a few ms); the time-lapse
  (40 × 128² × 2 channels) is generated once per preset change (~10 ms). Monte Carlo 10⁵ draws
  per update. No Wasm.
- **Tests** (`demos/test/fret-test.ts`, run by `make test`): $E(R_0) = 0.5$ and
  `distanceFromE(efficiency(r)) = r`; $\kappa^2$ = 4, 1, 0 for the three textbook geometries and
  its Monte Carlo isotropic average ≈ 2/3; the three-cube forward model followed by `correctFc`
  and `eFret` with the true $d$, $a$, $G$ returns the true $E$ exactly (noise-free), and the
  no-FRET class gives $F_c = 0$; slopes from noise-free controls equal $d$ and $a$; the
  incomplete-bleaching correction inverts its forward model; Monte Carlo CV and bias of a ratio
  of Poisson counts agree with the delta method within 5 % at ≥ 100 counts; `calibrate` and
  `calibrateInverse` are inverses; ratio is invariant to a common gain.
- **CSS and slides** (build-issues): prefix every per-demo selector with
  `.qb-demo[data-demo="…"]` (demo.css loads late); slide labels and readouts ≥ 18 px (no 16 px
  readouts on slides; use `data-compact` to drop panels instead of shrinking text); 1150×560;
  nothing hover-only; presets + reset; keyboard back to reveal.js (pointer-only drags for the
  dipoles, rectangles and calibration handles); static figure fallback per demo; plots via
  `lib/plot.ts` (clip ids are fixed, no local `uniqueClip`).
- Data: all synthetic with ground truth (parametric spectra: decided on the unmixing day, also
  here). Fluorophore names may label shapes that resemble them ("cyan donor, yellow acceptor") as
  examples only. A real three-image FRET data set and a real ratiometric time-lapse with a clear
  licence would make units 2 and 4 honest (principle 4): open question.
- Every demo also goes on `demos.qmd`; the sidebar gets a "Fluorescence quantification"
  section with "FRET" under it (and the colocalisation and FLIM days beside it).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `fret-physics/`: lecture (`fretruler`), $E(r)$ and $R_0$ by hand |
| 10:45–12:15 | Unit 2 `sensitised-emission/`: lecture, coefficients and $F_c$ by hand, `threecube` |
| 13:15–14:45 | Unit 3 `acceptor-bleach/`: lecture, corrections by hand, `apbleach`; apparent $E$ and the method comparison |
| 15:00–16:30 | Unit 4 `ratiometric/`: lecture, ratio noise and calibration by hand, `ratioimage`, `rationoise`; pitfall checklist and quizzes as wrap-up |

## Open questions

1. **Order within the fluorescence-quantification group**: colocalisation → FRET → FLIM
   (proposed: FRET's unit 3 then points forward to FLIM-FRET) or FLIM before FRET (then FLIM-FRET
   would recall this day's physics)? This plan assumes FRET before FLIM and puts the rate
   picture ($E = 1 - \tau_{DA}/\tau_D$) in unit 1's math layer as the bridge; the FLIM plan must
   not re-teach $E(r)$ and $R_0$.
2. **Noise propagation of ratios: owner?** Proposed here (no owner today). The evaluation day
   ("statistics of measurements") could own the delta method in general instead; then unit 4
   keeps the ratio-specific consequences (bias, tails, ratio of sums) and recalls the formula.
3. **Is unit 4 too full?** Biosensors + calibration + display + ratio noise. Fallback: move
   `rationoise`'s *difference* and *bleach* tabs to the exercise page only, and the calibration
   math to a callout.
4. **Without the unmixing day**: the one-slide recall plus a box may be thin for students who
   skipped `crosstalk/`. Alternative: make `crosstalk/` a hard prerequisite of unit 2 in the
   prerequisite database (`days.yml`), and keep this day's entry point as "Colour; spectral
   unmixing unit 1 required for unit 2".
5. **Pending edits in built units** (not done here): `endmembers/` non-linear mixing bullet →
   pointer to this day; `crosstalk/` "must compensate first (… ratios, FRET)" → link to
   `sensitised-emission/`; `coloc/` "a planned fluorescence-quantification day" → link once
   the group exists; `intensity-features/` ratio slide → forward pointer to `ratiometric/`.
6. **Real data**: a three-image FRET set with donor-only, acceptor-only and fusion controls, and a
   ratiometric time-lapse, both with licences (same search as the unmixing day's lambda stack).
7. Homo-FRET / anisotropy: pointer only (proposed), or an optional unit?

## References

- Förster (1948), *Zwischenmolekulare Energiewanderung und Fluoreszenz*, Ann. Physik.
- Lakowicz (2006), *Principles of Fluorescence Spectroscopy*, 3rd ed., Springer (ch. 13–15: FRET,
  $R_0$, $\kappa^2$, distance distributions).
- Stryer & Haugland (1967), *Energy transfer: a spectroscopic ruler*, PNAS.
- Jares-Erijman & Jovin (2003), *FRET imaging*, Nature Biotechnology.
- Youvan et al. (1997), *Calibration of fluorescence resonance energy transfer in microscopy*,
  Biotechnology et alia (three-cube correction, $F_c$).
- Gordon, Berry, Liang, Levine & Herman (1998), *Quantitative fluorescence resonance energy
  transfer measurements using fluorescence microscopy*, Biophys. J.
- Xia & Liu (2001), *Reliable and global measurement of fluorescence resonance energy transfer
  using fluorescence microscopes*, Biophys. J. (normalised index).
- Zal & Gascoigne (2004), *Photobleaching-corrected FRET efficiency imaging of live cells*,
  Biophys. J. ($E$-FRET, $G$ factor).
- Chen, Mills, Murphy & Periasamy (2006), *Measurement of FRET efficiency and ratio of donor to
  acceptor concentration in living cells*, Biophys. J. (stoichiometry).
- Kenworthy (2001), *Imaging protein–protein interactions using fluorescence resonance energy
  transfer microscopy*, Methods (acceptor photobleaching).
- Karpova et al. (2003), *Fluorescence resonance energy transfer from cyan to yellow fluorescent
  protein detected by acceptor photobleaching*, J. Microscopy.
- Vogel, Thaler & Koushik (2006), *Fanciful FRET*, Science's STKE (pitfalls, apparent $E$).
- Grynkiewicz, Poenie & Tsien (1985), *A new generation of Ca²⁺ indicators with greatly improved
  fluorescence properties*, J. Biol. Chem. (ratio calibration).
- Miyawaki et al. (1997), *Fluorescent indicators for Ca²⁺ based on green fluorescent proteins and
  calmodulin*, Nature (single-chain FRET sensor).
- Thaler et al. (2005), *Quantitative multiphoton spectral imaging and its use for measuring
  resonance energy transfer*, Biophys. J. (spectral FRET).
- Lee et al. (2005), *Accurate FRET measurements within single diffusing biomolecules using
  alternating-laser excitation*, Biophys. J. ($E$–$S$ histograms; idea only).
- Kendall & Stuart, or any statistics text, for the delta method; Marsaglia (2006), *Ratios of
  normal variables*, J. Stat. Software (ratio distributions, heavy tails).

## Build notes (2026-11-13, first build)

Built: the four core units `fret-physics/`, `sensitised-emission/`, `acceptor-bleach/`,
`ratiometric/` (lec, quiz, exercise), demos `fretruler`, `threecube`, `apbleach`,
`ratioimage`, `rationoise`, the new `demos/src/lib/fret.ts`, tests `demos/test/fret-test.ts`,
figures from `tools/figs/fret/figs.ts`. Optional unit 5 (`fret-stoichiometry/`) not built.

Deviations and corrections:
- **Steepest point of E(r)**: the plan says |dE/dr| is largest at r = R0. It is largest at
  (5/7)^(1/6) R0 = 0.945 R0 (slope −0.308/nm for R0 = 5 nm); at R0 it is −1.5/R0. Unit 1's math
  slide and B8 say so.
- **Example pair**: parametric spectra (lib/fret.ts `CYAN_DONOR`, `YELLOW_ACCEPTOR`; absorption is
  a mirrored split Gaussian, emission from lib/unmix.ts) give J = 2.41×10¹⁵ M⁻¹cm⁻¹nm⁴ and
  R0 = 4.95 nm (Q_D 0.40, n 1.4, κ² 2/3). Spectrum colours are fixed cyan/yellow, not
  `wavelengthToSrgb` (the emission peaks would render blue and green, contradicting "cyan
  donor, yellow acceptor").
- `fretruler`: acceptor spectra shift with a slider instead of dragging along λ; population
  orientations use their own seed so toggling "random static orientations" keeps the distances.
  Static random κ² lowers the mean E (0.501 → 0.392 at r = R0), used in E6.
- `threecube`: binding cells (half paired at E = 0.4) always present; "extra free acceptor"
  slider for the D:A preset; per-cell values are printed on the cells; the control masks use
  the known control cells. The E–S and spectral modes (optional unit) are not built.
- `apbleach`: the acceptor-only control cell is bleached as a whole (so photoconversion shows);
  a photoconversion slider added (`bleachScene({photoconvert})`); alignment by
  `lib/registration` phase correlation with the sign chosen by SSD.
- `ratioimage`: the wave peaks at 2580 nM (R = 2.36) so the calibration preset shows the
  explosion near R_max; calibration handles not built (fixed Rmin, Rmax, Kd; region-1 value
  with error bar). Poisson draws above λ = 10 use a normal approximation (`fastPoisson`) to keep
  40 frames at ~170 ms. Default region at frame 20 shows c(mean R) = 2179 nM against a true mean
  of 2457 nM (convex calibration curve: the Jensen effect again), discussed in E6.
- Slides embed demos with `data-presets` (subset of preset buttons) and `data-compact` (hides
  secondary sliders) so every demo fits 1150×560 at ≥ 18 px; plots use smaller viewBoxes on
  slides instead of larger fonts.
- Local DOM helper `demos/src/threecube/view.ts` (image tile with overlay, style injection,
  `filterPresets`, `hideControl`), shared by the five demos; candidate for a shared lib, like
  `canny/imgview.ts`.
- FLIM day: no unit folder existed at build time, so the pointers are plain text ("FLIM day"),
  no links. Add links when the FLIM day is built.
