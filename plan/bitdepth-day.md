# Plan: a day on bit depth, noise and how pixel values are stored

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
**Course map (2026-11-13, self-study restructure):** day 3 of 51, Part I "Images as numbers". Hard prerequisites: Images, filters and frequencies (day 1). Soft (see also): none. Source: `days.yml`.

**Earlier placement (2026-11-12):** day 3 of 51, Part I "Images as numbers". Order: day 2 (edges, blobs and ridges) → this → day 4 (image formation and restoration).
Revised after the reviews: `quantisation/` split into `quantisation/` + `cameranoise/` (course owner of noise models, SNR and PSNR); `gpuformats/` moved to the 3D-graphics day (`plan/graphics3d-day.md`), keeping its stride/endianness slides in `highbit/`.

## Goal of the day

After the day a student can:

1. distinguish **sampling** (where) from **quantisation** (which values), compute the number of levels $2^b$ and the step $\Delta$, and **derive** that rounding error is uniform with variance $\Delta^2/12$, so each extra bit buys about 6 dB of SNR;
2. explain **banding** and why **dithering** (adding noise before rounding, or diffusing the error) trades banding for fine-grained noise;
3. state the **noise models** the course uses (additive Gaussian with std $\sigma_n$, Poisson shot noise $\sqrt N$, read noise), compute **SNR and PSNR in dB**, and decide **how many bits are meaningful** for a camera: full-well capacity, read noise, gain in e⁻/ADU, dynamic range in dB and in bits;
4. handle **12- and 16-bit images** in practice: 12 bits in a 16-bit container (LSB- vs MSB-aligned, packed 12-bit), signed vs unsigned, **wrap-around vs saturation** when subtracting, safe type conversion, and why analysis happens in a wider or floating-point type;
5. **display** a high-bit-depth image on an 8-bit screen: window/level (CT Hounsfield units), contrast stretching, LUTs; know what tone mapping is for;
6. describe how an image is laid out **in memory**: interleaved vs planar, channel order, row stride and padding, endianness, the byte address of pixel $(x, y, c)$ (texture formats, premultiplied alpha and mipmaps: `gpuformats/` on the 3D-graphics day);
7. read and write **IEEE 754** numbers (sign, exponent, mantissa) in float32, float16, bfloat16; relate spacing to relative precision and machine epsilon; recognise **catastrophic cancellation** and **summation error** (naive vs pairwise vs Kahan); know NaN/Inf/subnormals and which integers are exact (up to $2^{24}$ in float32, $2^{11}$ in float16).

Why it fits this audience: quantisation error is a uniform random variable (mean and
variance by a one-line integral), shot noise is a Poisson variance, SNR is a ratio of
variances, and floating point is base-2 scientific notation. Everything can be worked by
hand on a few numbers. The day is practical: every student will meet a black 12-bit image
and a wrapped-around subtraction in their first real project.

## Prerequisite gap / overlaps

- **Images as arrays** (`images/`, day 1): one slide there mentions 8 vs 16 bit. Unit 1 starts from that sentence; nothing is repeated.
- **Colour** (`plan/colour-day.md`, day 42, later): owns colour spaces, gamma, the sRGB transfer function, perception and colour maps. sRGB storage codes appear here only as a forward pointer in `highbit/` (one line); the GPU sRGB formats moved with `gpuformats/` to the graphics day.
- **Thresholding** (`plan/thresholding-day.md`, day 12, later): owns histograms and thresholds. Here the histogram is only used as a display aid (where the data lie, what a window cuts off); unit 3 says "a histogram counts pixels per value" in one line, as a forward pointer. The thresholding day's note on bin choice for 16-bit images recalls `highbit/`.
- **Noise models: owned here** (`cameranoise/`). Additive Gaussian noise with std $\sigma_n$, Poisson shot noise, read noise, SNR and PSNR in dB, and the peak value of an 8/12/16-bit image are defined once in `cameranoise/`; Poisson variance = mean is stated, not derived. Every later day recalls them in one line: thresholding (noise breaks thresholds), image formation (day 4: the forward model, PSF and why inversion is ill-posed, built on this noise model) and deconvolution (`plan/deconvolution-day.md`: Wiener, Richardson–Lucy, denoisers), Ising, wavelets (`shrinkage/`), radon, unmixing (Poisson, weighted least squares), compression (keeps SSIM and "metrics disagree").
- **Image formation** (day 4, `plan/imageformation-day.md`): owns the forward model (PSF, OTF, Nyquist sampling in optics) and motivates the ill-posed inverse problem; deconvolution itself is a separate day (`plan/deconvolution-day.md`, position not yet set). `cameranoise/` ends with a one-line forward pointer to both.
- **Edges day** (day 2): no overlap.
- Order inside the day: unit 1 needs only `images/`; unit 2 needs unit 1; units 3–4 need unit 1.

## Units (one folder each: lec, quiz, exercise with #explore)

Four units, about 1.5 h each including the exercise: `quantisation/`, `cameranoise/`,
`highbit/`, `floatingpoint/`. (Built as `quantisation/`, `highbit/`, `gpuformats/`,
`floatingpoint/`; the split of `quantisation/` and the move of `gpuformats/` are pending
edits to the built units.)

### 1. `quantisation/`: Quantisation and bit depth

- Sampling (positions) vs quantisation (values). $b$ bits → $2^b$ levels, step $\Delta$.
- Uniform rounding error on $[-\Delta/2, \Delta/2]$: mean 0, variance $\Delta^2/12$ (derive). SNR of a full-range signal ≈ $6.02\,b$ dB (+1.76 for a sine).
- Banding / posterisation in smooth regions; **dithering**: random (noise before rounding, error becomes independent of the signal) and error diffusion (Floyd–Steinberg).
- **Demo `quantise`** (built): gradient, smooth synthetic, or the micrograph; bits 1–8; dither none / random / error diffusion; quantised image, error image, error histogram with the uniform density; readout: levels, $\Delta$, measured RMS error vs $\Delta/\sqrt{12}$, SNR in dB.
- By hand: levels and $\Delta$; quantise a short row with and without a given dither; variance of the uniform error.

### 2. `cameranoise/`: Where the numbers come from: camera noise, SNR and PSNR

Split from the second half of the built `quantisation/` deck ("Where the numbers come from /
Light itself is noisy / How many bits are meaningful / adding the noise sources"). Course
owner of noise models.

- The signal is noisy anyway: photons → electrons (full well), **shot noise** $\sqrt N$ (Poisson: variance = mean), read noise, ADC gain (e⁻/ADU), offset. Dynamic range = full well / read noise, in dB and bits. When $g/\sqrt{12} \ll \sigma_\text{read}$ the quantisation is harmless; extra ADC bits below the noise are wasted.
- **The usual model** (new slide): additive Gaussian noise with std $\sigma_n$ as the course-wide default, and when it is wrong (low counts: Poisson; saturation; hot pixels). Notation fixed here: $\sigma_n$ for noise, $\sigma$ for a Gaussian filter scale.
- **SNR and PSNR in dB** (new slide): SNR as a ratio of variances; MSE and PSNR $= 10\log_{10}(\text{peak}^2/\text{MSE})$, with the peak stated explicitly for 8-, 12- and 16-bit data (physical units of intensities defined here). Later days (wavelets, radon, compression, Ising, image formation) recall this slide.
- One-line forward pointer: blur (PSF) and the forward model are day 4; undoing them (deconvolution, denoisers) is the deconvolution day (`plan/deconvolution-day.md`).
- **Demo `camerabits`** (built): simulated camera (full well, read noise, ADC bits, exposure) on the micrograph, with a photon-transfer plot (noise vs signal on log axes: shot, read, quantisation, total, and measured points from simulated flat patches). Presets: scientific sCMOS, 8-bit ADC on a good sensor, 16-bit ADC on a noisy sensor, low light.
- By hand: shot noise and SNR for a given photon count; dynamic range and meaningful bits of a given sensor; PSNR of a given MSE for an 8-bit and a 12-bit image.

### 3. `highbit/`: 12- and 16-bit images in practice

- Where they come from: scientific CMOS/CCD (12, 14, 16 bit), CT (12-bit HU, stored signed), digital X-ray, raw photo files.
- Containers: 12 bits in uint16 (LSB-aligned: 0–4095, black when shown as 0–65535; MSB-aligned: multiples of 16), **packed 12-bit** (two pixels in three bytes).
- Integer types: unsigned/signed range, two's complement, **wrap-around** (modular) vs **saturating** arithmetic vs float; background subtraction and averaging as the classic traps. Type conversion: rescale or clip, never silently.
- **An image is a row of bytes** (2–3 slides kept from `gpuformats/`): interleaved (RGBRGB…) vs planar, channel order, row stride and padding, little-endian multi-byte values, byte address of pixel $(x, y, c)$.
- Display: the screen has 8 bits per channel. Window/level (**display window**: linear map + clip), CT windows in HU, auto-contrast from percentiles, LUTs (grey, hi-lo). The built slide "Colour maps, HDR and tone mapping" becomes a one-line pointer to colour-map design on day 42 (`colour-display/`) and keeps tone mapping (no other owner). sRGB storage codes: one line, "not proportional to light; why: day 42".
- **Demo `window16`** (new): synthetic 12-bit fluorescence image (seeded Poisson noise, offset, a few saturated pixels) and a synthetic CT phantom in HU; histogram with the window marked; window min/max sliders; presets (show as 0–65535, 0–4095, auto min–max, percentile, CT lung/soft tissue/bone); LSB/MSB container; background subtraction in uint16 wrap / uint16 saturate / float; LUT grey or hi-lo.
- By hand: ranges of int types; wrap vs saturate for given values; pack/unpack two 12-bit values; byte offsets with stride; little-endian bytes of a uint16; HU window → grey level; display level for a 12-bit value under a given window.

### (moved) `gpuformats/`: How GPUs store colour

Moved to the 3D-graphics day (`plan/graphics3d-day.md`) as its textures unit (texture
formats, hardware filtering, sRGB8_ALPHA8, premultiplied alpha and compositing, mipmaps and
memory, demo `texformats`). Its memory-layout slides stay in `highbit/` (above).

### 4. `floatingpoint/`: Floating point

- IEEE 754: $(-1)^s\,2^{e-\text{bias}}\,(1.f)_2$; float32, float16, bfloat16 field widths; special values (±0, ±Inf, NaN, subnormals).
- Spacing: ulp doubles at every power of two; relative precision $2^{-p}$; machine epsilon; rounding to nearest even; integers exact up to $2^{p+1}$ (float32 $2^{24}$, float16 $2^{11}$, bfloat16 $2^8$): a 16-bit image in half floats loses its low bits, and 65535 overflows to Inf.
- Catastrophic cancellation (variance as $E[x^2]-E[x]^2$; differences of nearly equal pixels); summation of many pixels: naive error grows with $n$, pairwise with $\log n$, Kahan compensated ≈ constant.
- **Demo `floatbits`** (new), two views: (a) *bits*: click the bits of a float16/bfloat16/float32, type a value, ±1 ulp, special-value presets, spacing-vs-value plot for all three formats; (b) *sum*: sum $n$ simulated 16-bit pixel values naively in float32/float16, pairwise, Kahan, against the exact sum; one- vs two-pass variance.
- By hand: decode/encode a few float16 bit patterns; spacing at 1000 and 3000; does 2049 survive float16?; a cancellation example; naive summation stalls.

## Shared engineering

- Pure numeric modules next to each demo (`quantise/quant.ts`, `camerabits/sensor.ts`, `window16/data.ts`, `floatbits/fp.ts`), so the same code feeds the demos, the tests and the numbers quoted in exercise solutions. `texformats/pack.ts` goes with `gpuformats/` to the graphics day.
- **Owner of `lib/random.ts`** (`rng`, `gauss`, `poisson`): move `poisson` from `camerabits/sensor.ts` and `rng` from `lib/ui.ts` (which re-exports it); add `gauss()`. Used by Ising, GMM, radon, unmixing, thresholding's synthetic images.
- `lib/metrics.ts`, image-quality section: MSE/PSNR (with an explicit peak) owned by `cameranoise/`; SSIM added by the compression day; `rmse` moves here from `fourier2d/fft.ts`.
- Synthetic generators `ramp`, `blobs` (`quantise/quant.ts`) and `fluorescence`, `ctPhantom` (`window16/data.ts`) move to `lib/synth.ts` (owner: thresholding day).
- Test: `demos/test/bitdepth-test.ts` (float16/bfloat16 encode/decode round trips against Python/numpy reference values, packing functions, quantiser error variance, wrap/saturate arithmetic), run by `make test` over `demos/test/*-test.ts`.
- Images: `demos/src/fourier2d/cell256.png` via `loadGrey()` in `lib/imageops.ts` (credited), synthetic images generated with fixed seeds.
- 2D canvas + SVG only (WebGL2 left with `texformats`).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `quantisation/` lecture + exercise (`quantise`) |
| 10:45–12:15 | Unit 2 `cameranoise/` lecture + exercise (`camerabits`) |
| 13:15–14:45 | Unit 3 `highbit/` lecture + exercise (`window16`) |
| 15:00–16:30 | Unit 4 `floatingpoint/` lecture + exercise (`floatbits`); quizzes as wrap-up |

## Open questions

- **Decided (course map):** `gpuformats/` is core only on the 3D-graphics day; layout slides stay in `highbit/`.
- **Decided (course map):** the camera part becomes `cameranoise/` on this day and owns noise models, SNR and PSNR; the PSF and forward model are on day 4 (image formation), deconvolution on its own day (`plan/deconvolution-day.md`). Where flat-field/dark-frame correction goes is open (day 4 or `cameranoise/`).
- **Decided (course map):** colour is day 42, after this day; sRGB storage is a one-line forward pointer.
- **Stale:** compressed formats are covered by the compression day (`compress-science/`).
- Open: an `integrity/` unit (which display and data adjustments are allowed on scientific images, provenance and metadata; review-gaps gap 7) would fit after `highbit/`, but the course map does not place it. Undecided; it would need a fifth unit or a closing course unit.
- Open: if time is short, `floatingpoint/` could share a unit with `highbit/` (review-gaps, mild).

## References

- B. Widrow and I. Kollár (2008), *Quantization Noise*, Cambridge University Press.
- J. R. Janesick (2007), *Photon Transfer*, SPIE Press.
- R. W. Floyd and L. Steinberg (1976), An adaptive algorithm for spatial greyscale, *Proc. SID* 17(2).
- D. Goldberg (1991), What every computer scientist should know about floating-point arithmetic, *ACM Computing Surveys* 23(1).
- N. J. Higham (1993), The accuracy of floating point summation, *SIAM J. Sci. Comput.* 14(4).
- IEEE Std 754-2019, *Standard for Floating-Point Arithmetic*.
- (Khronos texture-format tables and Porter & Duff compositing moved with `gpuformats/` to `plan/graphics3d-day.md`.)
