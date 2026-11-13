# Plan: a day on image formation

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12;
revised the same day after the course map made deconvolution its own day.

**Course map (2026-11-13, self-study restructure):** day 4 of 51, Part I "Images as numbers". Hard prerequisites: Images, filters and frequencies (day 1). Soft (see also): Bit depth and noise (day 3), Edges, blobs and ridges (day 2). Source: `days.yml`.

Day 4 of Part I (`plan/coursemap.md`), after *Images, filters, frequencies*
(`images/`, `kernels/`, `fourier/`, `pca/`), *Edges, blobs and ridges* and *Bit depth and
noise* (`quantisation/`, `cameranoise/`, `highbit/`, `floatingpoint/`). It is followed
directly by the **deconvolution day** (`plan/deconvolution-day.md`), which owns every
restoration method. Together the two days close gap 1 of `plan/review-gaps.md`. This day is
the course owner of **PSF, OTF, sampling/Nyquist in optics and the forward model**,
including the forward model as a linear map and the diagnosis that inverting it is
**ill-posed**. It does **not** own noise models (that is `cameranoise/`) or any cure
(Wiener, Tikhonov, regularisation, Richardson–Lucy, blind deconvolution, denoisers: the
deconvolution day).

## Goal of the day

After the day a student can:

1. describe the **imaging chain** (object → optics → sampling → noise → digital image) as a
   **forward model**, and say at which step blur, aliasing and noise enter;
2. explain the **point spread function**: the image of a point; image = object ∗ PSF (+ noise);
   recognise diffraction, defocus and motion blur as different PSFs;
3. read the **OTF/MTF** (the PSF's Fourier transform) as "how much contrast survives at each
   frequency", explain the **band limit** of a lens, and compute the **Abbe/Rayleigh**
   resolution of a microscope with numbers; derive the cut-off $2\mathrm{NA}/\lambda$ and the
   Airy pattern from the pupil *(math)*;
4. choose a **pixel size** for a microscope from Nyquist ($\Delta x \le \lambda/4\mathrm{NA}$),
   say what goes wrong under- and over-sampled, and explain why the 3D PSF is **elongated
   along the axis**;
5. **measure a PSF from beads** and say what limits the measurement (bead size, noise,
   sampling, background);
6. explain why running the forward model backwards is an **ill-posed** problem: naive
   **inverse filtering** divides by small OTF values and amplifies noise, and beyond the
   cut-off there is nothing to divide by (objects finer than the cut-off are **invisible**);
   write blur as a circulant matrix and relate its condition number to the ratio of largest
   to smallest OTF value *(math)*;
7. *(optional unit)* explain why a single isolated molecule can be **located** far more
   precisely than the PSF is wide.

Why it fits this audience: blur is a convolution, i.e. a (circulant) linear map; the
convolution theorem diagonalises it; inverting it is solving a badly conditioned linear
system. Every piece is linear algebra, Fourier and basic probability they have just seen in
Part 1, now with a physical reason. The day ends on the question the deconvolution day
answers: what must we add to make the inversion possible?

## Prerequisites and overlaps

Recall, don't re-teach: each item below gets at most one recall slide with a pointer.

| Concept | Owner | Here |
|---|---|---|
| Convolution, Gaussian blur, separable kernels | `kernels/` | Unit 1: "a blur kernel *is* a PSF" (one line) |
| Convolution theorem, low-/high-pass, 2D spectrum display, Nyquist $1/2\Delta$, aliasing | `fourier/` (built; slides "Filtering = turning frequencies up and down", "Sampling and aliasing" and its math slide; that slide already says "the optics of a microscope should match the pixel size") | Unit 1 recalls the convolution theorem in one slide; unit 2 recalls "two samples per cycle" in one slide, then **applies** it to the OTF cut-off. No re-derivation. |
| Gaussian scale $\sigma$, LoG, blobs | Edges, blobs and ridges | Unit 2: the Gaussian approximation of the PSF ($\sigma \approx 0.21\,\lambda/\mathrm{NA}$) is "the scale of the optics"; spot detectors should be tuned to it (one line) |
| Poisson shot noise, read noise, gain/offset, SNR, **PSNR** | `cameranoise/` (split from `quantisation/`, `review-organisation.md` §2.2) | Unit 1: noise is the last step of the forward model (one recall slide, no new definitions). Unit 3 uses the noise level to show the explosion of the inverse filter. Offset subtraction and flat-field are `cameranoise/`'s. |
| Condition number, least squares, SVD | linear algebra (audience.md) | Unit 3 recalls in one line; the blur matrix is circulant, so its singular values are the OTF magnitudes |
| Restoration: inverse filtering as a method, truncated inverse, Wiener, Tikhonov and regularisation in general, Richardson–Lucy, blind deconvolution, denoisers (incl. the median), what may be measured afterwards | **Deconvolution day** (`plan/deconvolution-day.md`, the next day) | Nothing taught here. Unit 3 shows only that the naive inverse fails and why, and ends with a forward pointer. See "Hand-over to the deconvolution day". |
| Tomography as a forward model | Radon day (Part II) | Unit 3 ends with a one-line forward pointer ("CT is another forward model: line integrals instead of a PSF"). The Radon plan recalls this day's ill-posedness slide; its cures (windowed ramp, semi-convergence) are recalled from the deconvolution day. |

## Units (one folder each: lec, quiz, exercise with #explore)

Two core units of about 1.5 h each, one short closing unit (about 45 min including its
exercise) and one optional unit. The time freed by moving restoration to its own day goes to
the exercises of units 1–2 (by-hand Nyquist numbers, the bead measurement), which were
squeezed when the day also held deconvolution. Every lecture starts with the "About this
lecture" slide ending in the standard line on **The math** slides. Math slides are titled
`## The math: … {.math}`; concept slides and the summary stand without them.

### 1. `psf/`: Image formation and the point spread function

Concept layer:

- The **imaging chain** as one picture: object (a fluorophore density, or reflectance) →
  optics (blur) → sampling on the pixel grid → noise (photons, read-out; recall
  `cameranoise/`) → quantisation (recall `quantisation/`) → digital image. This is the
  **forward model**: what the instrument does to the truth. Restoration tries to run it
  backwards (unit 3 shows why that is hard; the deconvolution day does it).
- A lens cannot focus light to a point: the image of a point is a small blob, the **PSF**.
  Fluorescence is incoherent, so images of different points add in intensity: the image is a
  sum of shifted, weighted PSFs, i.e. **image = object ∗ PSF**, the convolution of `kernels/`
  with a physical kernel. Small example: three point sources on a 1D line, PSF $(1,2,1)/4$,
  add the copies by hand.
- Recall (one slide, pointer to `fourier/`): convolution = multiplication of spectra. So
  the PSF's Fourier transform, the **OTF**, says how much of each frequency survives; its
  magnitude, the **MTF**, is the contrast transfer (1 at frequency 0, falling to 0).
- **Diffraction**: the lens collects only a cone of light (half-angle $\alpha$,
  $\mathrm{NA} = n\sin\alpha$). The OTF is exactly zero beyond a **cut-off frequency**
  $2\mathrm{NA}/\lambda$: the microscope is a **low-pass filter with a hard edge**. Detail finer
  than the cut-off is not blurred, it is **gone** from the image. The PSF of a perfect lens
  is the **Airy pattern** (central disc + faint rings).
- Numbers for a microscope (green light, $\lambda = 520$ nm): oil objective $\mathrm{NA} =
  1.4$: Rayleigh $0.61\lambda/\mathrm{NA} = 227$ nm, Abbe $\lambda/2\mathrm{NA} = 186$ nm,
  cut-off $5.4$ cycles/µm; dry $\mathrm{NA} = 0.3$: 1.06 µm. Blue vs red light: 450 vs 650
  nm changes the resolution by 44%.
- **Other PSFs**: **defocus** (a wider PSF; for large defocus a ring; its OTF goes
  *negative* at some frequencies, so stripes there appear with **reversed contrast**),
  **motion blur** (a line segment; its OTF is a sinc with **exact zeros**), a Gaussian as the
  convenient approximation of the in-focus PSF. Scattering in tissue, camera pixel area and
  atmospheric turbulence (one line) are blurs too.
- The forward model in one line: $\text{image} = \text{sample}\big(\text{object} \ast
  \text{PSF}\big) + \text{noise}$. It assumes the PSF is the same everywhere (shift
  invariance); one slide on when that fails (field edges, depth in thick tissue, refractive
  index mismatch).
- **Demo `psfotf`** (new, main demo): sliders NA (0.1–1.45), wavelength (400–700 nm,
  slider coloured), defocus (µm), PSF type menu (diffraction-limited / + defocus / motion
  with length and angle / Gaussian approximation). Panels: PSF (linear or log display, to see
  the rings), OTF magnitude as an image with the cut-off circle (signed OTF option, to see
  the negative lobes of defocus), radial MTF profile plot with $2\mathrm{NA}/\lambda$ marked,
  object and blurred image side by side. Objects: the micrograph `cell256.png` (treated as
  a "true object" with an assumed pixel size of 40 nm, so a 10 µm field; stated in the
  caption), a Siemens star, a field of point beads, line pairs. Readouts: Rayleigh and Abbe
  distance in nm, cut-off in cycles/µm, PSF FWHM. Presets: "oil objective, NA 1.4", "dry
  objective, NA 0.3", "blue vs red", "defocus: contrast reversal on the star", "motion
  blur: zeros in the OTF", "Gaussian approximation is close". Computation: the PSF is
  $|\mathcal F(\text{pupil}\cdot e^{i\,\text{defocus phase}})|^2$ on a 256² grid (one FFT),
  the blur one forward + one inverse FFT of the image; well under 30 ms in TypeScript.
- By hand: blur a 1D row of three point sources with $(1,2,1)/4$; Rayleigh distance for
  given $\lambda$, NA; which of two objectives resolves 200 nm?; from an MTF plot, the contrast
  of a grating at a given period; why a pinhole-free widefield image of a bead looks the same
  at all NA but at different sizes.

Math layer (`{.math}` slides):

- *The math: convolution and the OTF*: $g = f \ast h + n$, $G = F\,H + N$; $H(0) = 1$ for a
  PSF normalised to sum 1; MTF $= |H|$.
- *The math: the pupil and the Airy pattern*: coherent amplitude PSF = Fourier transform of
  the pupil $P$; incoherent PSF $h = |\mathcal F P|^2$; so the OTF is the **autocorrelation of
  the pupil**, which is zero once the shift exceeds the pupil diameter: cut-off $2\mathrm{NA}/\lambda$.
  Circular pupil: $h(r) \propto [2J_1(v)/v]^2$, $v = 2\pi\mathrm{NA}\,r/\lambda$; first zero of
  $J_1$ at 3.83 gives $r = 0.61\lambda/\mathrm{NA}$. OTF $= \tfrac{2}{\pi}\big(\arccos s -
  s\sqrt{1 - s^2}\big)$, $s = f/f_c$.
- *The math: defocus and motion*: defocus as a quadratic pupil phase; motion blur
  $h = \tfrac1L\mathbf 1_{[0,L]}$, $H(f) = \mathrm{sinc}(Lf)$, zeros at $f = k/L$.

### 2. `resolution/`: Resolution, sampling and measuring the PSF

Concept layer:

- **Two points**: move them closer; the dip between the peaks disappears. **Rayleigh**:
  peak of one on the first zero of the other ($0.61\lambda/\mathrm{NA}$, dip ≈ 26%);
  **Sparrow**: dip just vanishes (≈ $0.47\lambda/\mathrm{NA}$). Resolution is a convention,
  and **noise** decides in practice: a 26% dip in 20 photons is invisible.
- Recall (one slide, pointer to `fourier/`): a grid needs two samples per cycle. The optics
  deliver nothing above $2\mathrm{NA}/\lambda$, so the pixel (in the sample plane) should be
  $\Delta x \le \lambda/4\mathrm{NA}$: **Nyquist for a microscope**. Pixel in the sample plane
  = camera pixel ÷ magnification.
- Numbers (6.5 µm camera pixels, 520 nm): $100\times/1.4$ → 65 nm (Nyquist 93 nm: fine);
  $60\times/1.4$ → 108 nm (slightly undersampled); $20\times/0.75$ → 325 nm vs 173 nm
  (undersampled by 2: aliasing of fine periodic structure, resolution set by the camera, not
  the lens); $10\times/0.3$ → 650 vs 433 nm.
- **Oversampling** is not free: the same photons spread over more pixels, so each pixel is
  noisier (read noise per pixel, recall `cameranoise/`), files grow, bleaching per useful
  pixel grows. 2–3 pixels across the Rayleigh distance is the usual compromise.
  Deconvolution (next day) needs at least Nyquist sampling.
- **3D**: the PSF is elongated along the optical axis: lateral $\approx 0.61\lambda/\mathrm{NA}$,
  axial $\approx 2n\lambda/\mathrm{NA}^2$ (NA 1.4, oil: 0.23 µm vs 0.8 µm, about 3.5×).
  The widefield OTF has a **missing cone**: out-of-focus light is blurred, not removed, so
  widefield stacks lack optical sectioning; a confocal pinhole removes it (one picture,
  widefield vs confocal xz PSF). Axial Nyquist needs smaller z-steps than people usually
  take, and voxels are anisotropic (pointer: 3D images, marching-cubes/graphics days).
- **Measuring a PSF from beads**: sub-resolution fluorescent beads (100 nm or less), many of
  them, background-subtracted, centred with sub-pixel accuracy, averaged, normalised to sum 1.
  What limits it: bead size (a 200 nm bead widens the measured PSF), noise (averaging $N$
  beads reduces it by $\sqrt N$), sampling (pixel size), aberrations (a real PSF is often
  asymmetric along $z$: refractive index mismatch). Measured vs theoretical PSF: the
  measured one includes the real aberrations; the theoretical one is noise-free. Report the
  FWHM.
- **Demo `resolve`** (new), three modes via `data-mode`:
  - `twopoints` (default): two point sources, separation slider (in nm and in units of
    $\lambda/\mathrm{NA}$), NA and wavelength, pixel size slider with the Nyquist value marked,
    photon count (seeded Poisson noise; noise model from `cameranoise/`, recalled). Panels:
    2D image on the pixel grid, line profile through both points (continuous and sampled),
    dip in %, Rayleigh and Sparrow marks. A second object, a line grating / Siemens star,
    shows **aliasing** when the pixel is above Nyquist. Presets: "Rayleigh", "Sparrow", "just
    resolved but 20 photons", "Nyquist sampled", "undersampled: aliasing on the grating",
    "oversampled: noisier pixels".
  - `axial`: xz slice of the widefield 3D PSF (pupil with defocus phase per $z$ plane,
    64 planes of 128² FFTs; 2D canvas, no WebGL needed), NA slider, z-step slider with the
    axial Nyquist mark, toggle widefield / confocal (product of excitation and detection
    PSFs); readouts lateral and axial FWHM and their ratio. Presets: "NA 1.4", "NA 0.5",
    "z-step too large", "confocal".
  - `beads`: a simulated field of 30 beads (random positions, fixed seed, bead diameter
    slider, photons, background); click "average $N$ beads" (1, 5, 30); panels: one raw bead,
    the averaged PSF, true PSF, profile; readouts measured vs true FWHM. Presets: "one bead,
    noisy", "30 beads", "beads too large", "forgot the background".
- By hand: pixel size in the sample plane for three objective/camera pairs and whether each
  meets Nyquist; how many pixels across the Rayleigh distance; axial vs lateral FWHM for
  NA 0.8 and 1.4; average of three noisy bead profiles; bead-size correction
  ($\text{FWHM}_\text{meas}^2 \approx \text{FWHM}_\text{PSF}^2 + \text{FWHM}_\text{bead}^2$, as
  an approximation).

Math layer: *The math: Rayleigh and Sparrow* (sum of two Airy/Gaussian profiles, dip at the
midpoint, second derivative zero for Sparrow); *The math: Nyquist from the OTF support*
(band-limited to $f_c$ → $\Delta x \le 1/2f_c$; axial $\Delta z \le \lambda/\big(2n(1 - \cos\alpha)\big)$
from the OTF's axial extent); *The math: averaging beads* (variance $\sigma^2/N$; convolution
of PSF with the bead's projected sphere).

### 3. `illposed/`: Running the model backwards (short closing unit)

About eight concept slides, two math slides and a 30-minute exercise. It does **not** teach
any restoration method; it shows why one is needed, and hands over.

Concept layer:

- The question: we know the forward model and the PSF; can we get the object back from the
  image? Running a model backwards is an **inverse problem**. The forward direction is
  well behaved (one object gives one image); the backward direction need not be.
- In frequency space blur is multiplication by $H$, so divide by $H$: **inverse filtering**.
  Without noise and where $H \ne 0$, it is perfect (demo).
- With noise: $G/H = F + N/H$. Where $H$ is small (high frequencies, motion-blur zeros) the
  noise is multiplied by a huge number. **Small example**: 1D blur $(1,2,1)/4$ multiplies the
  three test waves of `fourier/` by 1, 0.5, 0; the inverse multiplies them by 1, 2, ∞. A
  noise of 1% at the fastest wave becomes infinite; at the middle one it doubles.
- **Invisible objects**: a grating finer than the cut-off gives exactly the same image as no
  grating at all. So adding it to any object changes nothing in the image: the data cannot
  tell these objects apart. No method recovers what the optics never transmitted.
- **Ill-posed** (in words): an answer may not exist (noise puts energy where $H = 0$), is not
  unique (invisible objects), and is not stable (tiny changes of the data, huge changes of
  the answer). Deblurring is ill-posed in all three ways.
- The lesson, as the bridge to the next day (one slide with three result thumbnails, no
  method details): restoration only works if we **add what we know** about the object and the
  noise (it is smooth, mostly piecewise constant, never negative; the noise has this size).
  Tomorrow: how to add it, and how to choose how much.
- Forward pointer (one line): tomography (Radon day) is another forward model, with
  line integrals instead of a PSF; the same diagnosis applies.
- **Demo `deconvlab`, `data-mode="inverse"`** (the demo is owned and specified by the
  deconvolution day; this mode shows only the naive inverse): object (Siemens star, beads +
  edges, cell micrograph), PSF (Gaussian σ / Airy NA / motion with length), noise (Gaussian
  σ or photon count, seeded), the "invisible grating" overlay toggle. Panels: observed,
  inverse-filtered, error vs truth, radial plot of $|H|$, $1/|H|$ and the noise floor.
  Presets: "no noise: inverse is perfect", "0.1% noise: inverse explodes", "motion blur:
  zeros", "an invisible grating". Readout: RMSE and the amplification factor
  $\max 1/|H|$ over the frequencies shown.
- By hand: inverse gains $1/H$ for $H = 1, 0.5, 0.1, 0.01$ and the resulting noise for a
  noise level of 1%; which frequency's noise dominates the restored image; sketch a pattern
  the NA 1.4 microscope of unit 1 cannot see (period below $\lambda/2\mathrm{NA} = 186$ nm);
  the $2\times2$ blur $A = \begin{bmatrix}0.75 & 0.25\\0.25 & 0.75\end{bmatrix}$ (gains 1 and
  0.5 on the sum and the difference) versus $\begin{bmatrix}0.55 & 0.45\\0.45 & 0.55\end{bmatrix}$
  (gains 1 and 0.1): how much does an error of 0.01 in the difference grow in each?

Math layer: *The math: blur is a circulant matrix* (eigenvectors = Fourier waves,
eigenvalues = $H(f)$; singular values $|H(f)|$; condition number $\max|H|/\min|H|$, infinite
past the cut-off); *The math: ill-posedness* (Hadamard's three conditions; the null space of
the blur = the frequencies past the cut-off; the bound $\|\delta\mathbf x\|/\|\mathbf x\| \le
\kappa\,\|\delta\mathbf y\|/\|\mathbf y\|$).

### 4. (optional / further reading) `localisation/`: Beating the diffraction limit with statistics

From gap 9 of `review-gaps.md`. Kept on this day because it is a **forward-model** idea: the
PSF is known, so a single isolated molecule's image is fitted with the model, nothing is
inverted. One molecule's position can be estimated far more precisely than the PSF width:
precision ≈ $\sigma_\text{PSF}/\sqrt N$ photons (background and pixel size make it worse;
one line). Centroid vs Gaussian least squares vs maximum likelihood; localisation microscopy
as "switch molecules on a few at a time"; SIM as a Fourier picture (moiré shifts high
frequencies inside the OTF support; one slide recalling unit 1). **Demo `localise`**:
photon-count slider, background slider, repeated seeded fits, scatter of estimates and the
precision vs $N$ against the $\sigma/\sqrt N$ line. Can be taken right after unit 2. Core
days elsewhere do not depend on it.

## Exercises: concept first, then math

Each `exercise.qmd`: concept questions (predict, explain, numbers) with a `Solutions`
callout; `### Math questions` with items tagged *(math)* and a `Solutions (math)` callout;
then `## Explore {#explore}` with the demo and 4–6 concrete questions. Examples:

- Unit 1 concept: "Halving the wavelength or doubling the NA: which improves resolution
  more?"; "A stripe pattern looks inverted in a defocused image. Why?". Math: derive the
  OTF cut-off from the pupil autocorrelation; show $H(0) = 1$ for a normalised PSF.
- Unit 2 concept: Nyquist pixel size for four objectives; "a 3D stack with 1 µm steps at
  NA 1.4: what is lost?"; a Nyquist table for the objectives and camera of a real (named
  by type, not brand) lab microscope; the bead-size correction for 100 and 200 nm beads.
  Math: Sparrow distance for a Gaussian PSF ($2\sigma$).
- Unit 3 concept: inverse gains and noise amplification; "two different objects give the
  same image: construct one pair"; "why does more photons help the inverse filter only a
  little?". Math: eigenvalues of a $4\times4$ circulant blur; condition number of the two
  $2\times2$ blurs.
- Explore examples: "Find a pixel size where the grating aliases (`resolve`)", "How many
  beads must be averaged before the measured FWHM is within 5% of the true one?
  (`resolve`, beads)", "Find the noise level at which the inverse filter first becomes
  useless (`deconvlab`, inverse mode)".

## Hand-over to the deconvolution day

What `plan/deconvolution-day.md` may assume (recall in one "yesterday" slide, never
re-teach):

- the forward model $g = \text{sample}(f \ast h) + n$, shift invariance and when it fails;
  PSF normalised to sum 1, $H(0) = 1$; OTF, MTF, the cut-off $2\mathrm{NA}/\lambda$;
- Nyquist sampling in the sample plane (deconvolution needs at least Nyquist; that sentence
  moves there as a requirement, its reason stays here);
- the 3D PSF, its axial elongation and the **missing cone** of widefield (one picture here;
  what deconvolution can and cannot do about it is the deconvolution day's);
- the measured (bead-averaged) PSF vs the computed one, and what limits the measurement;
- the blur as a circulant matrix, its condition number, the words **inverse problem**,
  **ill-posed**, **invisible object**; the $(1,2,1)/4$ example and the two $2\times2$ blur
  matrices, which the deconvolution day continues with λ.

What this day does not do: it names no cure. "Add what you know" is stated as the
question; Wiener, Tikhonov, regularisation, choosing λ, Richardson–Lucy, blind
deconvolution, denoisers, restoration artefacts and "what may be measured afterwards" are
all the deconvolution day's.

Shared code and demos: `lib/psf.ts` is owned here and used by the deconvolution demos;
`deconvlab` is owned by the deconvolution day and embedded here in `data-mode="inverse"`;
the averaged PSF of `resolve` (beads mode) is offered there as the "measured PSF" option.
Whichever day is built first writes the shared library code.

## Shared engineering

- **FFT**: reuse `demos/src/fourier2d/fft.ts` through its proposed new home `lib/fft.ts`
  (`review-organisation.md`: owner `fourier/`, `rmse` moves to `lib/metrics.ts`). If the move
  has not happened when this day is built, this day does it (one import change in
  `fourier2d`). Needed additions: complex multiply/divide helpers, `fftshift`, FFT-based
  circular convolution, zero padding to a power of two, a radial-average helper for MTF
  plots.
- `demos/src/lib/psf.ts` (new, owner: this day): pupil-based PSF with defocus phase
  ($|\mathcal F(P e^{i\phi})|^2$), Airy and analytical OTF (for tests), Gaussian
  approximation, motion line, 3D xz slice (widefield, confocal = product), OTF from PSF, FWHM
  measurement, Bessel $J_1$. Physical units: pixel size in nm passed explicitly.
- `demos/src/lib/deconv.ts` is owned by the deconvolution day; unit 3 here uses only its
  inverse-filter gain (if this day is built first, it writes that one function there).
- Reuse: `lib/imageops.ts` `loadGrey()` and Gaussian blur; `lib/random.ts` (`rng`, `gauss`,
  `poisson`: moved there on the bit-depth day); `lib/metrics.ts` (RMSE, PSNR with explicit
  peak); `lib/plot.ts` (line plots, log axes); `lib/colour.ts` (diverging map for signed OTF
  and error images); `lib/synth.ts` for the Siemens star, beads, line pairs (add there, not
  locally).
- **Speed** (measure first, `demos/README.md`): a 256² complex FFT in TypeScript is a few
  ms, so `psfotf` (one FFT for the PSF, two for the blur) and `resolve` stay in TypeScript;
  `resolve` axial mode is 64 FFTs of 128², computed once per NA change (tens of ms, not per
  frame). No Wasm expected on this day. `demos/bench/imageformation-bench.ts` (node) times the
  axial mode only.
- Tests: `demos/test/imageformation-test.ts` (run like the bit-depth test, `make test`):
  pupil-based in-focus PSF matches the Airy formula and its OTF the analytical OTF; PSF sums
  to 1, $H(0) = 1$; FFT convolution = direct convolution on small images; the inverse filter
  recovers the object exactly without noise where $H \ne 0$; the averaged bead PSF converges
  to the true PSF on a noise-free field.
- Images: `demos/src/fourier2d/cell256.png` (credited) as the "true object" with an assumed
  pixel size (stated in captions: it is itself already a blurred image); synthetic objects
  with fixed seeds. Every demo: presets + reset, labels ≥ 18 px, fits 1150×560, `data-*`
  settings, keyboard back to reveal.js, static fallback figure in each deck.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`psfotf`) |
| 10:45–12:15 | Unit 2 lecture + two points, by-hand Nyquist numbers (`resolve`, two points) |
| 13:15–14:30 | Unit 2 exercise continued: `resolve` axial and beads, bead-size correction, Nyquist table |
| 14:45–15:45 | Unit 3 (short): inverse gains by hand, `deconvlab` inverse mode, "what must we add?" |
| 15:45–16:30 | Quizzes as wrap-up; optional unit 4 (`localise`) for groups that want it |

## Open questions

- Is the day now too light? Units 1–2 were dense in the four-unit version and get the freed
  time; if it still ends early, the optional `localisation/` unit fills the last slot rather
  than pulling restoration back.
- Flat-field and dark-frame correction (open in `plan/bitdepth-day.md`): here, as part of
  the forward model ("the camera multiplies by a flat field and adds an offset"), or in
  `cameranoise/`? Recommendation: `cameranoise/` owns it (it is about the detector); unit 1
  recalls it in the forward-model picture.
- Coursemap: if the course is shortened, image formation and deconvolution merge into one
  day. Recommendation for the merged day: unit 1 here (condensed), unit 3 here, and the
  deconvolution day's `deconv/` and `richardson-lucy/` condensed; `resolution/` numbers,
  regularisation choice and denoisers become reading.
- Confocal: one picture in unit 2 only, or more (pinhole size, optical sectioning)? Kept to
  one picture; light-sheet and two-photon not mentioned.
- Real data: a measured bead stack would make unit 2 more convincing than the simulated
  one (and the deconvolution day needs a real widefield image + PSF pair). Which data, under
  which licence? Until then all demos are simulated and say so.
- `cell256.png` as "true object": assumed pixel size (40 nm proposed) and the caveat that it
  is already a blurred image. Acceptable for display here; for error-vs-truth plots the
  deconvolution day may prefer a synthetic phantom.
- Is the optional `localisation/` unit worth building for this cohort (gap 9 rates it
  should-have for a bio cohort)?

## References

- M. Born and E. Wolf (1999), *Principles of Optics*, 7th ed., Cambridge University Press
  (Airy pattern, Rayleigh criterion).
- J. W. Goodman (2005), *Introduction to Fourier Optics*, 3rd ed., Roberts & Co. (pupil, OTF,
  incoherent imaging).
- J. B. Pawley, ed. (2006), *Handbook of Biological Confocal Microscopy*, 3rd ed., Springer
  (sampling, 3D PSF, Nyquist for microscopes).
- P. C. Hansen (2010), *Discrete Inverse Problems: Insight and Algorithms*, SIAM (ill-posedness,
  condition; shared with the deconvolution and Radon plans).
- R. E. Thompson, D. R. Larson and W. W. Webb (2002), Precise nanometer localization analysis
  for individual fluorescent probes, *Biophys. J.* 82(5) (optional unit 4).
- Restoration references (Wiener, Tikhonov, Richardson–Lucy, denoisers): see
  `plan/deconvolution-day.md`.
