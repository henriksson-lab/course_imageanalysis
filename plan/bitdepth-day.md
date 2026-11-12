# Plan: a day on bit depth and how pixel values are stored

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. distinguish **sampling** (where) from **quantisation** (which values), compute the number of levels $2^b$ and the step $\Delta$, and **derive** that rounding error is uniform with variance $\Delta^2/12$, so each extra bit buys about 6 dB of SNR;
2. explain **banding** and why **dithering** (adding noise before rounding, or diffusing the error) trades banding for fine-grained noise;
3. decide **how many bits are meaningful** for a camera: full-well capacity, read noise, photon shot noise $\sqrt N$, gain in e⁻/ADU, dynamic range in dB and in bits;
4. handle **12- and 16-bit images** in practice: 12 bits in a 16-bit container (LSB- vs MSB-aligned, packed 12-bit), signed vs unsigned, **wrap-around vs saturation** when subtracting, safe type conversion, and why analysis happens in a wider or floating-point type;
5. **display** a high-bit-depth image on an 8-bit screen: window/level (CT Hounsfield units), contrast stretching, LUTs; know what tone mapping is for;
6. describe how a **GPU** stores an image: interleaved vs planar, channel order, row stride and padding, endianness; texture formats (normalised, integer, float; RGBA8, sRGB8_ALPHA8, RGB565, RGB10_A2, R16F, R32F, R16UI), which ones the hardware can filter, premultiplied alpha, mipmaps, memory per format;
7. read and write **IEEE 754** numbers (sign, exponent, mantissa) in float32, float16, bfloat16; relate spacing to relative precision and machine epsilon; recognise **catastrophic cancellation** and **summation error** (naive vs pairwise vs Kahan); know NaN/Inf/subnormals and which integers are exact (up to $2^{24}$ in float32, $2^{11}$ in float16).

Why it fits this audience: quantisation error is a uniform random variable (mean and
variance by a one-line integral), shot noise is a Poisson variance, SNR is a ratio of
variances, and floating point is base-2 scientific notation. Everything can be worked by
hand on a few numbers. The day is practical: every student will meet a black 12-bit image
and a wrapped-around subtraction in their first real project.

## Prerequisite gap / overlaps

- **Images as arrays** (`images/`): one slide there mentions 8 vs 16 bit. Unit 1 starts from that sentence; nothing is repeated.
- **Colour** (`plan/colour-day.md`, being written in parallel): owns colour spaces, gamma, the sRGB transfer function and perception. Here sRGB appears only as a **storage encoding**: "an 8-bit sRGB value is not proportional to light; the GPU's sRGB formats decode it before filtering; why: see the colour day". One slide in unit 3, one quiz question; no curves, no derivation.
- **Thresholding** (`plan/thresholding-day.md`): owns histograms and thresholds. Here the histogram is only used as a display aid (where the data lie, what a window cuts off); unit 2 recalls "a histogram counts pixels per value" in one line. The thresholding day's note on bin choice for 16-bit images can point to unit 2.
- **Noise / Poisson**: Poisson variance = mean is stated, not derived (basic probability). Shot noise is used again by any denoising day.
- No order of days is assumed: unit 1 needs only `images/`; units 2–4 need unit 1.

## Units (one folder each: lec, quiz, exercise with #explore)

Four units, about 1.5 h each including the exercise.

### 1. `quantisation/`: Quantisation and bit depth

- Sampling (positions) vs quantisation (values). $b$ bits → $2^b$ levels, step $\Delta$.
- Uniform rounding error on $[-\Delta/2, \Delta/2]$: mean 0, variance $\Delta^2/12$ (derive). SNR of a full-range signal ≈ $6.02\,b$ dB (+1.76 for a sine).
- Banding / posterisation in smooth regions; **dithering**: random (noise before rounding, error becomes independent of the signal) and error diffusion (Floyd–Steinberg).
- The signal is noisy anyway: photons → electrons (full well), **shot noise** $\sqrt N$, read noise, ADC gain (e⁻/ADU). Dynamic range = full well / read noise, in dB and bits. When $g/\sqrt{12} \ll \sigma_\text{read}$ the quantisation is harmless; extra ADC bits below the noise are wasted.
- **Demo `quantise`** (new): gradient, smooth synthetic, or the micrograph; bits 1–8; dither none / random / error diffusion; quantised image, error image, error histogram with the uniform density; readout: levels, $\Delta$, measured RMS error vs $\Delta/\sqrt{12}$, SNR in dB.
- **Demo `camerabits`** (new): simulated camera (full well, read noise, ADC bits, exposure) on the micrograph, with a photon-transfer plot (noise vs signal on log axes: shot, read, quantisation, total, and measured points from simulated flat patches). Presets: scientific sCMOS, 8-bit ADC on a good sensor, 16-bit ADC on a noisy sensor, low light.
- By hand: levels and $\Delta$; quantise a short row with and without a given dither; variance of the uniform error; dynamic range and meaningful bits of a given sensor.

### 2. `highbit/`: 12- and 16-bit images in practice

- Where they come from: scientific CMOS/CCD (12, 14, 16 bit), CT (12-bit HU, stored signed), digital X-ray, raw photo files.
- Containers: 12 bits in uint16 (LSB-aligned: 0–4095, black when shown as 0–65535; MSB-aligned: multiples of 16), **packed 12-bit** (two pixels in three bytes).
- Integer types: unsigned/signed range, two's complement, **wrap-around** (modular) vs **saturating** arithmetic vs float; background subtraction and averaging as the classic traps. Type conversion: rescale or clip, never silently.
- Display: the screen has 8 bits per channel. Window/level (linear map + clip), CT windows in HU, auto-contrast from percentiles, LUTs (grey, hi-lo, a perceptual colour map pointer to the colour day). HDR and tone mapping: one slide.
- **Demo `window16`** (new): synthetic 12-bit fluorescence image (seeded Poisson noise, offset, a few saturated pixels) and a synthetic CT phantom in HU; histogram with the window marked; window min/max sliders; presets (show as 0–65535, 0–4095, auto min–max, percentile, CT lung/soft tissue/bone); LSB/MSB container; background subtraction in uint16 wrap / uint16 saturate / float; LUT grey or hi-lo.
- By hand: ranges of int types; wrap vs saturate for given values; pack/unpack two 12-bit values; HU window → grey level; display level for a 12-bit value under a given window.

### 3. `gpuformats/`: How GPUs store colour

- Memory layout: interleaved (RGBRGB…) vs planar (RRR…GGG…), channel order RGBA/BGRA, row stride and padding (alignment 4), little-endian multi-byte values, address of pixel $(x, y, c)$.
- Texture formats: normalised (UNORM: $k/(2^b-1)$), integer (R16UI: no filtering), float (R16F, R32F), packed (RGB565, RGB10_A2), sRGB8_ALPHA8 (decoded by the hardware before filtering and blending: storage only here, the why is on the colour day).
- Hardware filtering: nearest vs bilinear (link to `images/`), which formats are filterable (integer never; R32F only with an extension).
- Premultiplied alpha (why filtering/blending needs it), mipmaps (+1/3 memory), memory per format, compressed formats in one line.
- **Demo `texformats`** (new, WebGL2): the same float data (a low-contrast ramp, smooth blobs, a tiny 4×4 image) uploaded as RGBA8, sRGB8_ALPHA8, RGB565, RGB10_A2, R16F, R32F, R16UI; two formats side by side; contrast stretch to make banding visible; nearest vs linear filtering; readout with bytes per texel, memory, and the stored bits of a clicked texel. Shows a clear message when WebGL2 or a format/extension is unavailable.
- By hand: byte offsets with stride; pack a colour into RGB565 and RGB10_A2; little-endian bytes of a uint16; premultiplied alpha blending; memory of a mipmapped RGBA16F texture.

### 4. `floatingpoint/`: Floating point

- IEEE 754: $(-1)^s\,2^{e-\text{bias}}\,(1.f)_2$; float32, float16, bfloat16 field widths; special values (±0, ±Inf, NaN, subnormals).
- Spacing: ulp doubles at every power of two; relative precision $2^{-p}$; machine epsilon; rounding to nearest even; integers exact up to $2^{p+1}$ (float32 $2^{24}$, float16 $2^{11}$, bfloat16 $2^8$): a 16-bit image in half floats loses its low bits, and 65535 overflows to Inf.
- Catastrophic cancellation (variance as $E[x^2]-E[x]^2$; differences of nearly equal pixels); summation of many pixels: naive error grows with $n$, pairwise with $\log n$, Kahan compensated ≈ constant.
- **Demo `floatbits`** (new), two views: (a) *bits*: click the bits of a float16/bfloat16/float32, type a value, ±1 ulp, special-value presets, spacing-vs-value plot for all three formats; (b) *sum*: sum $n$ simulated 16-bit pixel values naively in float32/float16, pairwise, Kahan, against the exact sum; one- vs two-pass variance.
- By hand: decode/encode a few float16 bit patterns; spacing at 1000 and 3000; does 2049 survive float16?; a cancellation example; naive summation stalls.

## Shared engineering

- Pure numeric modules next to each demo (`quantise/quant.ts`, `camerabits/sensor.ts`, `window16/data.ts`, `floatbits/fp.ts`, `texformats/pack.ts`), so the same code feeds the demos, the tests and the numbers quoted in exercise solutions.
- Test: `demos/test/bitdepth-test.ts` (float16/bfloat16 encode/decode round trips against Python/numpy reference values, packing functions, quantiser error variance, wrap/saturate arithmetic). Run: `node_modules/.bin/esbuild demos/test/bitdepth-test.ts --bundle --platform=node | node`.
- Images: `demos/src/fourier2d/cell256.png` (credited), synthetic images generated with fixed seeds.
- WebGL2 only in `texformats`; the rest is 2D canvas + SVG.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`quantise`, `camerabits`) |
| 10:45–12:15 | Unit 2 lecture + exercise (`window16`) |
| 13:15–14:45 | Unit 3 lecture + exercise (`texformats`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`floatbits`); quizzes as wrap-up |

## Open questions

- Is the GPU unit core for this course, or optional? It is the most "engineering" of the four; the layout part (stride, interleaved/planar, endianness) is useful to everyone, the texture-format part mostly to those who will write GPU code.
- Should unit 1's camera part move to a separate imaging/noise day if one is planned (shot noise, read noise, photon transfer)?
- Colour day ordering: if it comes first, unit 3's sRGB slide can be a one-line recall; if after, it is a forward pointer. Written as a pointer for now.
- Should compressed formats (JPEG artefacts, lossless PNG/TIFF, BC/ASTC on the GPU) get their own unit?

## References

- B. Widrow and I. Kollár (2008), *Quantization Noise*, Cambridge University Press.
- J. R. Janesick (2007), *Photon Transfer*, SPIE Press.
- R. W. Floyd and L. Steinberg (1976), An adaptive algorithm for spatial greyscale, *Proc. SID* 17(2).
- D. Goldberg (1991), What every computer scientist should know about floating-point arithmetic, *ACM Computing Surveys* 23(1).
- N. J. Higham (1993), The accuracy of floating point summation, *SIAM J. Sci. Comput.* 14(4).
- IEEE Std 754-2019, *Standard for Floating-Point Arithmetic*.
- Khronos, *OpenGL ES 3.0 / WebGL 2.0 specification*: texture formats and filterability tables.
- T. Porter and T. Duff (1984), Compositing digital images, *SIGGRAPH* (premultiplied alpha).
