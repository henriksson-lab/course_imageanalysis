# Plan: a day on wavelets

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. explain why the Fourier basis is a poor fit for signals with **localised** features (a spike or a step spreads over all frequencies), and state the **time–frequency trade-off** (a window cannot be narrow in both time and frequency);
2. compute a full **1D Haar transform** of an 8-sample signal by hand (averages and differences, level by level), invert it, and show that the normalised Haar matrix is **orthogonal**;
3. describe a **multiresolution analysis** (scaling function $\varphi$, wavelet $\psi$, nested approximation spaces, $\psi_{j,k}$ by dilation and translation) and compute the DWT as a **two-channel filter bank** (convolution + downsampling by 2) with perfect reconstruction and $\mathcal{O}(N)$ cost;
4. explain what **vanishing moments** buy (Daubechies wavelets: smooth parts of a signal give near-zero detail coefficients) and why that makes the coefficients **sparse**;
5. compute and read the **2D separable DWT** (LL / LH / HL / HH subbands and their edge orientations) of an image;
6. denoise an image by **wavelet shrinkage** (hard vs soft thresholding, the universal threshold $\sigma\sqrt{2\ln N}$, $\sigma$ estimated from the finest subband), and compare "keep the largest coefficients" with Fourier at equal coefficient count.

Why it fits this audience: the whole day is **change of basis** with an orthogonal matrix
(their linear algebra) and inner products of functions (their calculus). The filter bank is
two convolutions from *Filtering & convolution* followed by keeping every second sample, and
the motivation is the one thing the Fourier unit cannot do: say **where** a frequency occurs.

## Links to other units and plans

- **`fourier/` (existing)**: DFT as a change of basis, orthogonality, Parseval, 2D separability, the convolution theorem, aliasing and "low-pass before downsampling", the `fourier2d` demo with its "keep largest" mode. This day **recalls** these in one line each and builds on them: the wavelet basis is another orthogonal basis, chosen to be localised in both position and frequency. The "keep largest" comparison in unit 3 deliberately reuses the `fourier2d` idea and image so the two can be put side by side.
- **`kernels/` (existing)**: convolution, border rules (zeros / replicate / mirror / wrap), Gaussian, derivative kernels. Filter banks are **convolutions + downsampling**; the boundary-handling slide (unit 2) points back to the kernels border slide instead of re-teaching it. The Haar high-pass $(1, -1)/\sqrt2$ is a scaled finite difference, which is the link to the derivative kernels.
- **Compression day (`plan/compression-day.md`, written in parallel)**: split agreed in the brief. **This day owns** wavelet theory and the transforms (Haar, Daubechies, MRA, filter banks, 2D DWT, lifting). **The compression day owns** JPEG 2000 as an application (quantisation, bit-plane / EBCOT coding, rate control), entropy coding, and the JPEG/DCT codec. Here, "keep the $K$ largest coefficients" appears only as a measure of **sparsity** (unit 3) and integer lifting only as the reason lossless JPEG 2000 is possible (unit 6); both end with a pointer to the compression day. If the compression day provides `lib/dct.ts`, unit 3 uses it for the block-DCT comparison (see Shared engineering).
- **Denoising elsewhere**: `plan/ising-day.md` (Ising/MRF prior, unit `mrf-denoise/`; quadratic and **total variation** priors, unit `gmrf/`) and `plan/levelsets-day.md` (curvature flow, a PDE that smooths) treat denoising as **energy minimisation with a prior on neighbouring pixels**. Wavelet shrinkage (unit 4) is a third view: **a sparsity prior in a basis**. Unit 4 ends with one slide comparing the three on the same step-and-ramp profile (TV keeps steps, staircases ramps; shrinkage keeps steps with small ringing, follows ramps). The connection is stated, not derived: soft thresholding is the exact minimiser of $\tfrac12\|y - x\|^2 + \lambda\|Wx\|_1$ for orthogonal $W$, the same form as the TV energy with a different penalty.
- **Scale space / Laplacian of Gaussian**: `fourier/exercise.qmd` promises that "the Laplacian of Gaussian comes back when we detect objects", but no plan owns LoG blob detection yet. Optional unit 5 (Mexican hat = $-\nabla^2 G$) can serve as that place or point to it (open question).
- **Gaussian pyramid**: `plan/snakes-day.md` uses one (via `lib/imageops.ts`) for coarse-to-fine search. Unit 2 mentions the Gaussian / Laplacian pyramid (Burt & Adelson) as the redundant cousin of the DWT: same idea, no downsampled orthogonal basis.

## Prerequisite gap

- Everything needed is in `fourier/` and `kernels/`. Two items get one recall slide each: **inner product of functions** $\langle f, g\rangle = \int f g\,dt$ and orthonormality (from calculus, and the Fourier lecture's projection slide), and **downsampling causes aliasing** (the Fourier lecture's sampling slide), which the filter bank must cancel.
- **Noise model and PSNR** are not defined in Part 1. Unit 4 defines additive Gaussian noise with standard deviation $\sigma$ and $\mathrm{PSNR} = 10\log_{10}(\mathrm{peak}^2/\mathrm{MSE})$ in one slide (the Ising plan defines the same noise model; agree notation).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units.

### 1. `haar/`: Localised basis functions and the Haar transform

- Recall (one line, `fourier/`): the DFT is an orthogonal change of basis to waves that extend over the **whole** signal.
- Picture first: a spike and a step, with their Fourier spectra. A spike at one position has $|X_k| = 1$ for **every** $k$; a step decays only as $1/k$. Where the feature is lives in the **phases**, smeared over all coefficients. Keep 10% of the Fourier coefficients and the step rings everywhere (Gibbs).
- Fix 1: **short-time Fourier transform**: multiply by a window, then DFT; slide the window. Picture of the time–frequency tiling: every tile has the same shape. **Uncertainty**: $\Delta t\,\Delta\omega \ge \tfrac12$ (stated; Gaussian window attains it, Gabor 1946). A narrow window resolves the step but not low frequencies; a wide one the reverse. One window size cannot serve both.
- Fix 2, the wavelet idea: **short windows for high frequencies, long windows for low frequencies**. Tiling picture: tall-thin tiles at high frequency, short-wide at low.
- The **Haar transform**: pairwise averages $s = (a+b)/2$ and differences $d = (a-b)/2$; keep the details, repeat on the averages. For $N = 8$: three levels, result = 1 overall average + $1 + 2 + 4$ details. Worked on the board: $x = (4, 6, 10, 12, 8, 6, 5, 5)$ → $(7 \mid 1 \mid -3, 1 \mid -1, -1, 1, 0)$.
- Inverse: $a = s + d$, $b = s - d$. Nothing is lost: it is a change of basis.
- The basis vectors (rows of the Haar matrix): a constant, one ±1 square wave, two half-length ones, four quarter-length ones. With $1/\sqrt2$ instead of $1/2$ per step, the rows are **orthonormal**: $H H^\top = I$, so Parseval holds just as for the DFT.
- Payoff: a spike touches only $\log_2 N + 1$ Haar coefficients; a piecewise-constant signal has zero details everywhere except at its jumps → **sparse**.
- **Demo `haarstep`** (new): an editable 8- or 16-sample signal (drag bar heights; `data-n="8|16"`). Step button runs one level at a time: pairs highlighted, averages drawn as wide bars, differences as a separate row, coloured by level; a coefficient tree below (Mallat layout). Toggle normalisation $1/2$ vs $1/\sqrt2$ (shows energy is preserved only with $1/\sqrt2$). Click a coefficient to zero it or "keep $K$ largest" slider: the reconstruction and the basis vector of each kept coefficient are drawn; readout of squared error = sum of the dropped squared coefficients (orthonormal case). Toggle "show basis" draws the 8×8 Haar matrix as an image with ±colours. Presets: the board example, a spike, a step, a ramp (details do *not* vanish: motivates unit 2), constant.
- **Demo `wavesparse`** (new; reused in unit 2): a 1D signal of length 256 or 512 (presets: spike, step, two steps, piecewise-smooth "Blocks/Bumps/HeaviSine/Doppler" test signals of Donoho–Johnstone, a chirp, a sine). Two panels side by side: Fourier coefficients (magnitudes, sorted or in natural order) and wavelet coefficients (laid out by level as rows, a scalogram-like strip). One slider "keep $K$ largest" drives both reconstructions; plot of relative error vs $K$ for both bases on log axes. In unit 1 the wavelet menu is Haar only (`data-wavelets="haar"`). Presets: "spike: Fourier needs all", "sine: Fourier needs two", "step: Haar wins", "Doppler: both struggle, wavelets less".
- By hand: the full Haar transform of the 8-sample board signal (or a second one in the exercise), its inverse; reconstruct from the 3 largest coefficients and compute the error; the 2-sample and 4-sample Haar matrices, check $HH^\top = I$; how many Haar coefficients are nonzero for a spike at position 5 of 8?

### 2. `filterbank/`: Multiresolution, filter banks and Daubechies wavelets

- From vectors to functions: the Haar **scaling function** $\varphi = \mathbf 1_{[0,1)}$ and **wavelet** $\psi = \mathbf 1_{[0,1/2)} - \mathbf 1_{[1/2,1)}$. Dilations and translations $\psi_{j,k}(t) = 2^{j/2}\psi(2^j t - k)$: orthonormal in $L^2$ (check $\langle\psi_{1,0}, \psi_{1,1}\rangle = 0$ and $\langle \psi_{0,0}, \psi_{1,0}\rangle = 0$ by drawing).
- **Multiresolution analysis**: nested spaces $V_0 \subset V_1 \subset V_2 \subset \dots$ (piecewise constant at resolution $2^{-j}$); $V_{j+1} = V_j \oplus W_j$ (approximation + detail); $\{\varphi(\cdot - k)\}$ spans $V_0$. Picture: a signal projected onto $V_3, V_2, V_1, V_0$: a coarse-to-fine staircase.
- **Two-scale relation**: $\varphi(t) = \sqrt2\sum_n h_n\varphi(2t - n)$, $\psi(t) = \sqrt2\sum_n g_n\varphi(2t - n)$. The numbers $h_n$, $g_n$ are all we ever compute with. Haar: $h = (1, 1)/\sqrt2$, $g = (1, -1)/\sqrt2$.
- The **DWT as a filter bank**: convolve with the low-pass $h$ and the high-pass $g$ (time-reversed), keep every second sample. Block diagram: analysis (filter, ↓2) → synthesis (↑2, filter, add). Unit 1's averages and differences are exactly this.
- Why downsampling is allowed: each branch alone aliases (recall `fourier/` sampling slide), but for an orthogonal pair with $g_n = (-1)^n h_{1-n}$ (quadrature mirror) the aliasing of the two branches **cancels** on reconstruction. State the conditions, show it in the demo; no z-transform derivation.
- **The cascade / pyramid**: repeat on the low-pass branch only. Work $N + N/2 + N/4 + \dots < 2N$: **$\mathcal{O}(N)$**, faster than the FFT. Contrast with the Gaussian / Laplacian pyramid (Burt & Adelson 1983): same coarse-to-fine idea, but redundant (more numbers than pixels) and not an orthogonal basis.
- **Vanishing moments**: $\psi$ has $p$ vanishing moments if $\int t^m\psi(t)\,dt = 0$ for $m < p$; for the filter, $\sum_n n^m g_n = 0$. Then any polynomial of degree $< p$ gives **zero** detail coefficients, and a smooth signal (Taylor!) gives tiny ones. Haar has $p = 1$: only constants vanish, which is why the ramp in unit 1 left details everywhere.
- **Daubechies wavelets**: the shortest orthogonal filters with $p$ vanishing moments, $2p$ taps. db2: $h = \big(1+\sqrt3,\ 3+\sqrt3,\ 3-\sqrt3,\ 1-\sqrt3\big)/(4\sqrt2)$. Pictures of $\varphi$ and $\psi$ for db1 (= Haar), db2, db4, db8, computed by the cascade algorithm: smoother and wider as $p$ grows. Trade-off: more vanishing moments = sparser for smooth parts, but longer support = more coefficients touched by each edge. Mention symlets (nearly symmetric) and that orthogonal + symmetric is impossible except Haar, so JPEG 2000 uses **biorthogonal** wavelets (CDF 9/7, LeGall 5/3; pointer to unit 6 and the compression day).
- **Boundaries**: the signal is finite; the filter runs off the end. Same choices as the kernels border slide: periodic (wrap; simple, exact orthogonality, but a jump at the border creates large details), symmetric extension (fewer artefacts, the usual image choice), zero padding (adds coefficients). Shown in the demo, not derived.
- **Demo `filterbank`** (new): a 1D signal (length 64–256, presets from `wavesparse` plus a polynomial of chosen degree) through a drawn block diagram: low-pass and high-pass outputs before and after ↓2, after ↑2, after synthesis filters, and their sum against the input (difference shown, ≈ 1e-15). Wavelet menu: Haar, db2, db3, db4, db8, sym4; levels slider 1–5 (cascade view stacks the levels). Checkboxes to break it: "drop the high-pass branch" (shows the aliasing that the high-pass was cancelling), "wrong synthesis filter", boundary mode periodic / symmetric / zero. Side tab "basis functions": $\varphi$ and $\psi$ by the cascade algorithm with the iteration count as a slider (watch the shape converge). Presets: "Haar = unit 1", "perfect reconstruction", "remove one branch: aliasing", "quadratic signal: db3 details vanish, Haar's don't", "periodic boundary: false edge at the end".
- **Demo `wavesparse`** again with `data-wavelets="haar db2 db4 db8"`: preset "HeaviSine: db4 needs far fewer coefficients than Haar".
- By hand: run $x = (4, 6, 10, 12)$ through the Haar filter bank (convolve, keep even samples) and compare with unit 1; check $\sum g_n = 0$ and $\sum n\,g_n = 0$ for db2 using $g = (h_3, -h_2, h_1, -h_0)$ (just arithmetic with $\sqrt3$); show that a linear ramp gives zero db2 details (away from the border); count the operations of a 3-level DWT of length 1024 vs an FFT.

### 3. `dwt2d/`: The 2D wavelet transform and sparsity of images

- Separable, as for the 2D DFT (recall `fourier/`): 1D DWT on every row, then on every column. One level gives four quarter-size subbands: **LL** (approximation), and three detail bands responding to horizontal edges, vertical edges and diagonals. Name the convention used (LH vs HL differ between books) and stick to it in the demo labels.
- By hand on a $2\times2$ block $\begin{pmatrix}a & b\\ c & d\end{pmatrix}$ with Haar: $\tfrac12(a+b+c+d)$, $\tfrac12(a-b+c-d)$, $\tfrac12(a+b-c-d)$, $\tfrac12(a-b-c+d)$. A vertical edge lights only one band.
- Repeat on LL: the **subband pyramid** (Mallat layout). Displaying it: each band rescaled separately, or magnitudes on a log scale; LL looks like a thumbnail.
- **Sparsity of a real image**: the histogram of detail coefficients is sharply peaked at 0 with heavy tails (Laplacian-like). Most of the energy sits in LL and a few large details along edges.
- **Keep the $K$ largest**: reconstruction quality vs $K$ for Haar, db4, global Fourier (the `fourier2d` "keep largest" mode) and $8\times 8$ block DCT, at equal coefficient count. Typical result: wavelets and block DCT beat global Fourier; Haar shows blocks, db4 smooth blur with small ringing near edges; DCT shows $8\times8$ blocks at low $K$. Coefficient count ≠ bits: positions and quantisation cost bits too, which is the compression day's subject (pointer: JPEG 2000 = 2D CDF 9/7 DWT + quantisation + bit-plane coding).
- Shift variance: shift the image by one pixel and the decimated coefficients change a lot (picture). Motivates the undecimated transform in units 4 and 5.
- **Demo `dwt2d`** (new, main demo): the micrograph `demos/src/fourier2d/cell256.png` (256×256, already a power of two) and synthetic images (a disc, a square, oriented bars, a checkerboard, a smooth gradient). Panels: image, subband pyramid (Haar / db2 / db4 / db8 / CDF 9/7 menu, levels 1–6, display: per-band normalised or log magnitude with sign colour), reconstruction from the kept coefficients, error image. "Keep $K$ largest" slider (as % of pixels) with readouts PSNR and % kept; a second mode "zero a subband" (click a band to remove it, e.g. drop all HH or all of level 1). Toggle "compare": the same $K$ for global Fourier and block DCT side by side, with a PSNR-vs-$K$ plot for all bases. Toggle "shift by 1 px". Presets: "one level: three orientations" (oriented bars), "pyramid of the cell", "keep 3%: Haar blocks vs db4", "wavelets vs Fourier vs DCT at 5%", "drop the finest level = blur", "shift variance".
- By hand: the 4 Haar subband values of two $2\times2$ blocks (flat, vertical edge); one level of 2D Haar on a $4\times4$ image with a horizontal edge, which bands are nonzero; how many coefficients does an $L$-level transform of an $N\times N$ image have (exactly $N^2$)? Given a list of PSNR values from the demo, which basis would you choose and why.

### 4. `shrinkage/`: Denoising by wavelet shrinkage

- Picture first: noisy image → wavelet coefficients. Signal: few large coefficients. White Gaussian noise: an orthonormal transform maps it to white Gaussian noise of the **same** $\sigma$ in every coefficient (rotation of an isotropic Gaussian; one line of linear algebra). So: **small coefficients are mostly noise, large ones mostly signal**.
- **Hard thresholding** $\eta_H(w) = w\,[|w| > t]$ (keep or kill) vs **soft thresholding** $\eta_S(w) = \operatorname{sign}(w)\max(|w| - t, 0)$ (kill and shrink). Plot both transfer curves. Hard: lower bias, but isolated surviving noise coefficients become visible blips; soft: smoother, but edges lose contrast.
- How large is $t$? The maximum of $N$ independent $\mathcal N(0, \sigma^2)$ values is about $\sigma\sqrt{2\ln N}$: the **universal threshold** (VisuShrink, Donoho & Johnstone 1994) kills pure noise with high probability. For $256^2$ pixels: $\approx 4.7\sigma$. Tends to over-smooth; mention level-dependent thresholds (SureShrink, BayesShrink) in words only.
- $\sigma$ is unknown: estimate it from the finest diagonal subband, $\hat\sigma = \operatorname{median}(|w|)/0.6745$ (robust because the few signal coefficients barely move the median).
- Leave LL alone; threshold the detail bands only.
- Artefacts and a fix: decimated Haar/Daubechies denoising leaves blocky or ringing artefacts tied to the grid. **Cycle spinning** (Coifman & Donoho 1995): shift, denoise, shift back, average; equivalent to the undecimated transform. Clear improvement in the demo.
- Third view of denoising (cross-reference, one slide): MRF/TV priors (`plan/ising-day.md`) penalise differences of **neighbouring pixels**; shrinkage penalises the size of **coefficients in a basis**. Soft thresholding is the exact minimiser of $\tfrac12\|y - x\|^2 + t\|Wx\|_1$ for orthonormal $W$, the same "data + prior" energy with an $\ell^1$ prior. Same step-and-ramp profile shown with TV and with shrinkage.
- **Demo `wavedenoise`** (new): image (cell crop 256×256, synthetic piecewise-constant shapes) plus seeded Gaussian noise, slider $\sigma$. Wavelet menu (Haar, db2, db4, db8), levels, hard / soft, threshold slider $t$ in units of $\hat\sigma$ with a mark at the universal threshold, cycle-spinning toggle (number of shifts). Panels: noisy, denoised, removed part (noisy − denoised; should look like noise), error vs truth. Plot: PSNR vs $t$ for hard and soft (computed by sweeping, ~20 values), current $t$ marked; histogram of the finest detail coefficients with $\pm t$ lines. Readouts: $\sigma$ true vs $\hat\sigma$, PSNR in/out, % coefficients kept. A 1D tab (`data-mode="1d"`) with the Donoho–Johnstone test signals for the board. Presets: "universal threshold, soft", "hard vs soft at the same $t$", "threshold too low: noise blips", "too high: blur", "Haar blocks → cycle spinning", "σ estimated from HH".
- By hand: apply hard and soft thresholding with $t = 2$ to $(5, -1.5, 0.4, -3, 2.1, -0.2)$; compute $\sqrt{2\ln N}$ for $N = 256^2$ (with $\ln 2 \approx 0.693$); estimate $\hat\sigma$ from a list of 9 given detail coefficients; explain why soft thresholding lowers the contrast of an edge by about $t$.

### 5. (optional / further reading) `cwt/`: The continuous wavelet transform, scalograms and spot detection

- **CWT**: $W(a, b) = \int f(t)\,\tfrac1{\sqrt a}\psi\!\big(\tfrac{t-b}{a}\big)dt$, a convolution with a scaled wavelet at **every** scale and shift (redundant, not a basis). Computed via FFT (convolution theorem from `fourier/`). **Scalogram** $|W(a,b)|$: a picture of where each frequency occurs. **Morlet** wavelet (Gaussian × wave) for oscillations; **Mexican hat** $\propto -\tfrac{d^2}{dt^2}G_\sigma$, the 1D Laplacian of Gaussian.
- Link to blob detection: in 2D, $-\nabla^2 G_\sigma$ responds most to a blob of radius $\approx\sqrt2\,\sigma$; scale-normalised LoG across scales = a Mexican-hat CWT (Lindeberg). Pointer to wherever LoG blob detection is taught.
- **À trous** (undecimated, "with holes") transform: smooth with the B3-spline kernel $(1, 4, 6, 4, 1)/16$ dilated by $2^{j-1}$ (insert zeros), detail $w_j = a_{j-1} - a_j$; the image is exactly $a_J + \sum_j w_j$. Shift-invariant, same size at every scale.
- **Spot detection in fluorescence microscopy** (Olivo-Marin 2002): threshold each $w_j$ at $k\hat\sigma_j$ ($k \approx 3$, $\hat\sigma_j$ by MAD), multiply the thresholded scales (spots persist across scales, noise does not), label connected components. Robust to an uneven background, because the background goes into $a_J$.
- **Demo `scalogram`** (new): a 1D signal (chirp, two tones in sequence, spikes on a sine, a line profile across the micrograph). Panels: signal, STFT spectrogram with a window-width slider, CWT scalogram (Morlet / Mexican hat), same colour scale. Presets: "chirp: STFT window too short / too long / wavelet adapts", "spikes on a sine", "Mexican hat finds bumps of each width".
- **Demo `atrous`** (new): a synthetic fluorescence image (256×256: Gaussian spots of varying brightness and size on a smooth uneven background, Poisson + Gaussian noise, fixed seed) and, if available, a real spot image (open question). Panels: the scales $w_1 \dots w_5$ and $a_J$ as thumbnails, the multiscale product, detections as circles with true positions (synthetic) and counts of hits / misses / false alarms. Sliders $k$, scales used, noise level. Comparison mode: global threshold on the raw image, LoG at one $\sigma$. Presets: "uneven background: global threshold fails", "k = 3", "dim spots lost at high noise", "only scale 1 = noise".
- By hand: à trous level 1 of a 9-sample signal with one spike (convolve with $(1,4,6,4,1)/16$, subtract); check that the sum of the scales gives the input back; the Mexican hat response to a 1D box of width 1, 3, 9 at $\sigma = 1$ (read off a printed plot).

### 6. (optional / further reading) `lifting/`: The lifting scheme and integer wavelets

- Any wavelet filter bank factors into **lifting steps** (Sweldens 1996): split into even and odd samples, **predict** the odd from the even (the detail is the prediction error), **update** the even to keep the average.
- Haar by lifting: $d = x_{2k+1} - x_{2k}$, $s = x_{2k} + d/2$. Invert by running the steps backwards with the signs flipped: invertible **whatever** the predict and update functions are.
- **Integer wavelets**: round inside each step, $s = x_{2k} + \lfloor d/2 \rfloor$, and the transform maps integers to integers and still inverts **exactly**. LeGall 5/3 = linear prediction + update in two lifting steps. This is the reversible path of JPEG 2000 (lossless mode); pointer to the compression day for the codec.
- Also: in-place computation, about half the operations of the filter-bank form.
- **Demo**: extend `haarstep` with `data-mode="lifting"`: the split / predict / update steps drawn as arrows on the 8 or 16 samples, an "integer" toggle showing the rounded values and an exact round trip, and a 5/3 option.
- By hand: integer Haar by lifting of $(4, 6, 10, 12, 8, 6, 5, 5)$ for one level, then invert; show that rounding the plain averages and half-differences, $\lfloor(a+b)/2\rfloor$ and $\lfloor(a-b)/2\rfloor$, is not invertible ($(4, 5)$ and $(3, 5)$ both give $(4, -1)$), while the lifted version is.

## Shared engineering

- `demos/src/lib/wavelet.ts` (new): filter coefficients (Haar, db2–db8, sym4, biorthogonal CDF 9/7 and LeGall 5/3 analysis/synthesis pairs); one analysis/synthesis step on a strided `Float64Array` with boundary mode (periodic, symmetric, zero); multi-level 1D DWT and inverse; 2D separable DWT and inverse with the Mallat layout and a subband index helper (level, orientation → rectangle) for drawing; lifting for Haar and 5/3 with an integer mode; à trous (B3 spline) decomposition and reconstruction; cascade algorithm for $\varphi$, $\psi$; thresholding helpers (hard, soft, MAD $\hat\sigma$, universal threshold); cycle spinning; Donoho–Johnstone test signals; an instrumented Haar that records each level for `haarstep`. No DOM, so it runs under node.
- FFT: reuse `demos/src/fourier2d/fft.ts` (1D and 2D radix-2) for the Fourier side of `wavesparse` and `dwt2d`, the STFT and the FFT-based CWT. Suggest moving it to `demos/src/lib/fft.ts` when another plan needs it too (not done here).
- DCT: the $8\times8$ block DCT for the `dwt2d` comparison should come from one place. If the compression day adds `lib/dct.ts`, use it; otherwise write a small `lib/dct.ts` here and hand ownership to the compression day. Coordinate.
- Image loading, Gaussian blur, LoG: the shared `demos/src/lib/imageops.ts` (one home for linear filters across plans); line/bar/log plots (error vs $K$, PSNR vs $t$, histograms): the shared `lib/plot.ts`; connected components for `atrous` detections: `lib/label.ts` (labelling plan).
- Image: `demos/src/fourier2d/cell256.png` at full 256×256 (credit R. M. Hunt, CC BY 3.0, as in the other plans). Synthetic images and noise from a fixed seed (`rng(seed)` in `lib/ui.ts`).
- Speed: a 6-level db8 2D DWT of 256×256 is ~1 M multiply-adds, a few ms in TypeScript; cycle spinning with 16 shifts and the PSNR-vs-$t$ sweep are the heaviest (tens of ms; compute the sweep once per noise/wavelet change, not per slider tick). No Wasm expected; measure first as `demos/README.md` says.
- Display: coefficient images use a diverging colour map (sign) with per-subband or global normalisation, plus thin lines between subbands; labels ≥ 18 px; everything fits 1150×560 (`dwt2d` with four panels at 256 px uses a 2×2 grid or tabs on slides).
- Tests (in `demos/test/`, next to the existing `test.ts`): perfect reconstruction (1D and 2D, every wavelet and boundary mode, error < 1e-10); orthonormal wavelets preserve energy; db filters satisfy $\sum h = \sqrt2$, $\sum g = 0$ and the vanishing-moment sums; integer lifting round-trips exactly on random integers; Haar of the board example equals $(7, 1, -3, 1, -1, -1, 1, 0)$; à trous scales sum to the input; a few reference coefficient vectors computed offline (e.g. PyWavelets) and pasted in as constants.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, `data-*` settings (`data-wavelets`, `data-mode`, `data-n`) so one demo serves several units.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, Haar by hand, `haarstep`, `wavesparse` (Haar only) |
| 10:45–12:15 | Unit 2 lecture, filter bank and vanishing moments by hand, `filterbank` |
| 13:15–14:45 | Unit 3 lecture, 2D Haar by hand, `dwt2d` |
| 15:00–16:30 | Unit 4 lecture, thresholding by hand, `wavedenoise`; quizzes as wrap-up |

## Open questions

- How long is a "day" (hours, lecture vs exercise time)? Unit 2 is the densest (MRA + filter bank + Daubechies + boundaries); if it runs over, Daubechies and vanishing moments could become their own unit and unit 4 move to optional.
- How much MRA theory: full nested-spaces treatment with $L^2$ and the two-scale relation, or stay with vectors and filters (Strang & Nguyen style) and show $\varphi$, $\psi$ only as pictures? Recommendation: vectors and filters first, function view on two slides.
- Is optional unit 5 the place for the Laplacian of Gaussian / scale-space blob detection that `fourier/exercise.qmd` promises, or does another day own it? Scalogram (1D signals) is less central for image analysis than à trous spot detection; if time is short, keep `atrous` and drop `scalogram`.
- A real fluorescence image with spots (licence-compatible, e.g. a public-domain single-molecule or FISH image) for `atrous`; the current micrograph has no spots.
- Should lifting (unit 6) be taught here or on the compression day? The split puts the transform here; the compression day might prefer to present the 5/3 integer path together with lossless coding. Agree with that plan's author.
- Colour images: stay greyscale (recommended), or show the 2D DWT per channel / in a luminance–chrominance space as a bridge to the colour day?
- 2D subband naming (LH vs HL): pick one convention course-wide, consistent with the compression day.
- Programming language for the code parts is still open (see README).

## References

- Gabor (1946), *Theory of communication*, J. IEE (time–frequency, uncertainty).
- Haar (1910), *Zur Theorie der orthogonalen Funktionensysteme*, Math. Ann.
- Mallat (1989), *A theory for multiresolution signal decomposition: the wavelet representation*, PAMI.
- Daubechies (1988), *Orthonormal bases of compactly supported wavelets*, Comm. Pure Appl. Math.; Daubechies (1992), *Ten Lectures on Wavelets*, SIAM.
- Strang & Nguyen (1996), *Wavelets and Filter Banks*, Wellesley-Cambridge (the linear-algebra view used here).
- Mallat (2009), *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd ed., Academic Press.
- Burt & Adelson (1983), *The Laplacian pyramid as a compact image code*, IEEE Trans. Commun.
- Donoho & Johnstone (1994), *Ideal spatial adaptation by wavelet shrinkage*, Biometrika (VisuShrink, test signals); Donoho (1995), *De-noising by soft-thresholding*, IEEE Trans. Inf. Theory.
- Chang, Yu & Vetterli (2000), *Adaptive wavelet thresholding for image denoising and compression*, IEEE TIP (BayesShrink).
- Coifman & Donoho (1995), *Translation-invariant de-noising*, in *Wavelets and Statistics*, Springer (cycle spinning).
- Sweldens (1996), *The lifting scheme: a custom-design construction of biorthogonal wavelets*, Appl. Comput. Harmon. Anal.; Calderbank, Daubechies, Sweldens & Yeo (1998), *Wavelet transforms that map integers to integers*, ACHA.
- Torrence & Compo (1998), *A practical guide to wavelet analysis*, Bull. Am. Meteorol. Soc. (CWT, scalograms).
- Starck, Murtagh & Bijaoui (1998), *Image Processing and Data Analysis: The Multiscale Approach*, CUP (à trous).
- Olivo-Marin (2002), *Extraction of spots in biological images using multiscale products*, Pattern Recognition 35:1989–1996.
- Lindeberg (1998), *Feature detection with automatic scale selection*, IJCV (scale-normalised LoG).
- Taubman & Marcellin (2002), *JPEG2000: Image Compression Fundamentals, Standards and Practice*, Kluwer (pointer only; the codec belongs to the compression day).
