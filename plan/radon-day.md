# Plan: a day on the Radon transform and tomographic reconstruction

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** elective, group "Transforms" (wavelets → compression; Radon); entry after Part I (core day 4). Independent of the wavelets chain; if the Hough elective (group "Detection") is also taken, run it first so the $(\rho, \theta)$ pictures are back-references.

## Goal of the day

After the day a student can:

1. explain why an X-ray measurement is a **line integral** of the attenuation coefficient
   (Beer–Lambert: $-\ln(I/I_0) = \int \mu\,ds$), and name other modalities that measure line
   integrals (electron tomography, PET/SPECT, optical projection tomography);
2. compute **projections** of a small object by hand at a few angles, parameterise lines by
   $(\rho, \theta)$, and read a **sinogram**: a point becomes a sinusoid, a shift moves the
   sinusoid, the second half of the sinogram mirrors the first;
3. explain **back-projection** and why plain back-projection blurs (the star of a point,
   falling off as $1/r$);
4. state the **Fourier slice theorem** as a picture (the 1D spectrum of a projection is a
   central line of the 2D spectrum) and use it to explain direct Fourier reconstruction and
   its interpolation problem; prove it *(math)*;
5. explain **filtered back-projection**: what the **ramp filter** $|\omega|$ undoes, why it
   amplifies noise, and what a tapered ramp (Shepp–Logan, Hann) trades; derive the ramp
   from the polar Jacobian *(math)*;
6. predict the **artefacts of undersampling**: streaks from too few angles, the **missing
   wedge** of limited-angle tomography, noise from low dose; state the angle rule
   $N_\theta \approx \tfrac{\pi}{2} N$;
7. write tomography as a linear system $A\mathbf x = \mathbf b$, build $A$ for a 2×2 image by
   hand, recognise $A^\top$ as back-projection, run **Kaczmarz/ART** and **SIRT** steps by
   hand, and explain why priors (non-negativity, **total variation**) help with few views or
   low dose, as an instance of the ill-posed inverse problem met on day 4.

Why it fits this audience: every step is linear algebra or calculus they already have.
Projection is a line integral; a sinusoid is $r\cos(\theta - \varphi)$; the Fourier slice
theorem is the 2D Fourier transform of *The Fourier transform* evaluated on a line;
reconstruction is $A\mathbf x = \mathbf b$, least squares and $A^\top$. Poisson noise is recalled
from `cameranoise/` (day 3) as intuition only ("few photons, noisy counts").

## Prerequisite gap

- **2D DFT, convolution theorem, sampling**: taught in `fourier/` (day 1). Units 2–3 recall the 2D spectrum (centre = DC, angle = orientation) and
  "filtering = multiplying the spectrum" in one slide each. The **1D** continuous Fourier
  transform is used in the math layer; it is owned by a `fourier/` math slide (to be added
  there per review-organisation edit 11) and recalled here in one line.
- **Bilinear interpolation and inverse mapping**: `images/` (*Images as arrays*). Recalled
  in one line where a projection samples the image along a ray.
- **Line integrals, polar coordinates, the Jacobian $r$**: calculus. Recalled in one line
  in the math layer of units 2 and 3 (the Jacobian is exactly the ramp).
- **Least squares, normal equations, transpose, null space**: linear algebra; recalled in
  one line in unit 4.
- **Poisson / shot noise** ($\sqrt N$), SNR, PSNR: owned by `cameranoise/` (day 3). Here
  only "fewer photons → noisier line integrals, worst where few photons get through"; no
  derivation.
- **Forward model and ill-posedness**: day 4 (image formation) shows that measuring is a
  linear forward operator and that inverting it amplifies noise. Unit 4 recalls that idea in
  one slide: tomography is the same story with $A$ = the projector. Wiener/Tikhonov/
  Richardson–Lucy deconvolution is the deconvolution day's (`plan/deconvolution-day.md`);
  pointer only.
- **Total variation**: owned by the Ising/MRF elective (`gmrf/`, 1D). Unit 4 states it in
  one line for 2D images, because the Ising elective may not have been taken (different
  group).

## Links to other days

| Topic | Owner | Elsewhere |
|---|---|---|
| Normal parameterisation of a line $(\rho, \theta)$, $\rho = x\cos\theta + y\sin\theta$ | both, **same convention** | Hough: voting, accumulator, peaks; this day: integrals, sinogram, inversion |
| Radon transform, sinogram, back-projection, inversion | **this day** | Hough day: one optional slide "Hough of an edge map ≈ Radon transform of the edge map", pointing here |
| Line detection by peaks in $(\rho,\theta)$ | **Hough day** | this day, optional unit 6: one slide, pointer to Hough |
| Fourier slice theorem | **this day**, unit 2 | `fourier/`: none; FFT registration (day 12): may quote it for "rotation rotates the spectrum" (pointer) |
| 2D DFT, convolution theorem, continuous FT (math), `lib/fft.ts` | `fourier/` (day 1) | this day: recall + reuse of code |
| Polar / log-polar resampling of an image or spectrum | shared code (`lib/polar.ts`, owner: dynprog elective) | FFT registration: Cartesian → (log-)polar; dynprog: polar unwrap; this day: polar → Cartesian **gridding** in direct Fourier reconstruction |
| Beer–Lambert, absorption along a ray (line integral of attenuation) | **this day** (unit 1) | graphics3d `volrender/` generalises to emission–absorption and transmittance $\mathcal T$; stain optical density (unmixing elective, colour day) recalls it in one line |
| Volume rendering, MIP/average projection | **graphics3d** `volrender/` | this day: "volume rendering computes projections to look at, tomography inverts projections to get the volume" |
| Hounsfield units, display window/level | **`highbit/`** (day 3, `window16`) | this day: HU defined as the output scale of a reconstruction, one line, recall for display |
| Energy = data + λ·prior | **graph cuts** `seg-energy/` (day 13) | this day unit 4: the same form with $\|A\mathbf x - \mathbf b\|^2$ as the data term |
| TV and quadratic priors | **ising** (`gmrf/`) | this day unit 4: one line for 2D TV |
| Forward model, ill-posed inversion | **day 4** (image formation) | this day unit 4: recall |
| Deconvolution (Wiener, Tikhonov, Richardson–Lucy), denoiser comparison | **deconvolution day** (`plan/deconvolution-day.md`) | this day: pointer only (MLEM ≈ Richardson–Lucy for emission data) |
| Denoising by wavelet shrinkage | **wavelets** (elective) | this day: pointer only (denoise sinogram or reconstruction) |
| Shot noise, Poisson, SNR/PSNR | **`cameranoise/`** (day 3) | this day: intuition only |

**Hough split (agreed with `plan/hough-day.md`; course convention: origin at the centre, $\theta \in [0°, 180°)$, signed $\rho$).**
The Hough day keeps the *voting* view: each edge pixel votes for all lines through it, which
draws a sinusoid in the accumulator; peaks are lines. This day owns the *integral* view: the
Radon transform sums grey values along every line, and inverting it is the main subject.
Both use the same $(\rho, \theta)$ convention ($\theta \in [0°, 180°)$, $\rho$ signed, origin at
the image centre) and the same sinogram display (θ horizontal, ρ vertical), so the
pictures match. One slide on each side names the link: "the Hough accumulator of a binary
edge map is (a discretised) Radon transform of that map". `lib/radon.ts` is written here; the
Hough demos may call its forward projector for that slide but keep their own voting code.

**Graphics3d split.** This day owns Beer–Lambert (absorption along a ray, two lines, own
math slide). Graphics3d `volrender/` (3D elective) recalls it and generalises to the
emission–absorption ODE (transmittance $\mathcal T$), with an "average = an X-ray" mode in
`volview`; this day points to that mode for the 3D picture. Parallel-beam 3D tomography is a stack of 2D
slices: one slide, no 3D reconstruction demo (`volview` can show a reconstructed stack if a
3D example is wanted later; open question).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units.

### 1. `radon/`: Projections and the sinogram

- Picture first: an X-ray of a hand, a CT slice (credited, CC licence) and the question "the
  X-ray overlays everything; how does CT get a slice?". The answer of the day: take many
  X-rays from all around, then undo the overlaying.
- **Physics in one slide**: a beam loses a fixed fraction of its photons per centimetre of
  material, $\mu$ (attenuation coefficient, higher for bone than soft tissue). Through 20 cm
  of water-like tissue with $\mu \approx 0.2\ \text{cm}^{-1}$, $I/I_0 = e^{-4} \approx 0.018$.
  Taking $-\ln(I/I_0)$ turns the product of attenuations into a **sum of $\mu$ along the
  ray**: a line integral. Other modalities, one line each: electron tomography (tilt series
  of a thin specimen), PET/SPECT (emitted counts summed along a line; no logarithm),
  optical projection tomography (cleared specimens, visible light).
- **The math: Beer–Lambert** {.math}: $dI/ds = -\mu(s) I$ ⇒ $I = I_0 \exp(-\int \mu\,ds)$.
  Monochromatic beam assumed (beam hardening breaks this; optional unit 5).
- **Parameterising lines**: a line is all points with $x\cos\theta + y\sin\theta = \rho$
  ($\theta$ = direction of the normal, $\rho$ = signed distance from the centre). Same
  parameterisation as the Hough elective (a recall there if taken first; otherwise this slide
  defines it). A **projection** at angle
  $\theta$ is the profile $p_\theta(\rho)$ of line integrals of all parallel lines; stacking
  the profiles for $\theta \in [0°, 180°)$ gives the **sinogram**.
- **A point becomes a sinusoid**: a point at radius $r$ and angle $\varphi$ lands at
  $\rho = r\cos(\theta - \varphi)$. Amplitude = distance from the centre, phase = its
  direction. The centre of rotation is the only point that stays put. Hence the name.
- Properties, each as a picture with the demo: **linearity** (the sinogram of two objects is
  the sum: overlapping sinusoids add); **shift** (moving the object moves every sinusoid by
  $\mathbf t\cdot(\cos\theta, \sin\theta)$: a sinusoidal shift); **symmetry**
  $p_{\theta + 180°}(\rho) = p_\theta(-\rho)$ (why 180° suffice for parallel beams);
  **mass**: every projection has the same total.
- **The math: the Radon transform** {.math}: $\mathcal R f(\rho, \theta) = \iint f(x, y)\,
  \delta(x\cos\theta + y\sin\theta - \rho)\,dx\,dy$; linearity, shift theorem
  $f(\mathbf x - \mathbf t) \mapsto \mathcal R f(\rho - \mathbf t\cdot\mathbf n_\theta, \theta)$,
  symmetry, $\int \mathcal R f\,d\rho = \iint f$. Projection of a uniform disc of radius $R$:
  $2\mu\sqrt{R^2 - \rho^2}$ (chord length); of an ellipse: closed form (why the Shepp–Logan
  phantom is built from ellipses: its sinogram is exact).
- Recall: on a pixel grid a projection samples the image along each ray with **bilinear
  interpolation** (`images/`), one line.
- **Demo `radonproj`** (new, main demo of the unit): left, the image with the rotating
  detector (a bar with $N$ bins) and a few of its rays drawn; middle, the current
  projection profile $p_\theta(\rho)$; right, the sinogram filling in column by column as
  $\theta$ advances (play/pause, angle slider, step). **Click in the image** to drop point
  masses: each one's sinusoid is drawn over the sinogram in its own colour, and the current
  $\rho$ of each is marked on the profile. Image menu: point masses only, disc, square,
  modified Shepp–Logan (generated from ellipses in code), the micrograph `cell256.png`
  (credited), "two discs" (linearity). Toggle *shift object* (drag the whole object; watch
  the sinusoids move). Presets: "three points by hand" (the exercise below, at 0°, 45°,
  90°), "the centre does not move", "180°–360° mirrors 0°–180°", "every column has the
  same sum", "Shepp–Logan sinogram", "a real image". 128×128, 180 angles: forward projection
  of the full image is ~3 M bilinear samples, a few ms in TS.
- By hand: three point masses $A = (2, 0)$ mass 1, $B = (0, 1)$ mass 2, $C = (-1, -1)$ mass 1.
  Projections at 0°, 45°, 90° (as bar plots on a $\rho$ axis): at 0°, bars at $\rho = 2, 0, -1$
  with heights 1, 2, 1; at 90°, at $0, 1, -1$; at 45°, at $\approx 1.41, 0.71, -1.41$.
  Check: each projection sums to 4. Sketch the three sinusoids and mark where they cross
  (angles at which two points hide behind each other). Then: projections of a 3×3 binary
  image at 0° and 90° (row and column sums), and the question "can you tell which image it
  was?" (no: the 2×2 swap ambiguity, used again in unit 4).

### 2. `backprojection/`: Back-projection and the Fourier slice theorem

- **Back-projection**: smear each projection back across the image along its own rays,
  and add over all angles. Picture: a single point. One angle gives a stripe; four angles
  a star; 180 angles a bright centre with a halo that fades with distance.
- **Why it blurs**: with $N$ angles, every line passes through the point; at distance $r$
  the $2N$ crossings of a circle of circumference $2\pi r$ are spread out, so the halo
  density falls like $1/r$. Plain back-projection = the true image blurred by a $1/r$ kernel.
  Not a bad image of a point, but every object gets a glow and loses contrast. It is
  the **transpose** of projection, not its inverse (said here, used in unit 4).
- **The math: back-projection** {.math}: $\mathcal B g(\mathbf x) = \int_0^\pi
  g(\mathbf x\cdot\mathbf n_\theta, \theta)\,d\theta$; $\mathcal B\mathcal R f = f * \tfrac{1}{|\mathbf x|}$
  (sketch: back-projecting the projections of a point and changing to polar coordinates,
  $dx\,dy = r\,dr\,d\theta$). The 2D Fourier transform of $1/|\mathbf x|$ is $1/|\boldsymbol\omega|$:
  low frequencies over-weighted.
- **The Fourier slice theorem** as a picture first: the 1D spectrum of the projection at
  angle $\theta$ equals the 2D spectrum of the image along the **central line** through the
  origin at the same angle. By hand with stripes (below). Consequence: projections at all
  angles fill the 2D spectrum spoke by spoke. Reconstruction in principle: FFT every
  projection, place it on its spoke, inverse 2D FFT.
- Recall from `fourier/` (one slide): the 2D spectrum, centre = slow waves, angle =
  orientation; projecting along $y$ averages away every wave that varies in $y$.
- **The math: the Fourier slice theorem** {.math}: proof in three lines (for $\theta = 0$:
  $\int p_0(x) e^{-2\pi i u x}dx = \iint f(x, y) e^{-2\pi i (ux + 0\cdot y)}dx\,dy = F(u, 0)$;
  general $\theta$ by rotating coordinates, since the 2D Fourier transform commutes with
  rotations). Continuous Fourier transform stated as the limit of the DFT, one line.
  Discrete version: the DFT of a projection sampled with $N$ bins equals the 2D DFT on the
  central line only approximately (the spoke falls between grid points except at 0° and 90°).
- **Direct Fourier reconstruction and its problem**: the spokes are dense near the centre
  and sparse at the edge; the inverse FFT needs a Cartesian grid. Interpolating
  (gridding) from polar to Cartesian samples in the frequency domain gives errors that
  spread over the **whole** image (each frequency is a wave over the whole image). Why
  practical scanners mostly use back-projection-based methods; same polar resampling as the
  log-polar step of FFT registration, in the opposite direction (named).
- **Demo `fourierslice`** (new): left, the image with a draggable angle handle; middle, its
  2D log-spectrum with the central line at that angle; right, two curves on one plot: the
  1D DFT magnitude of the projection and the 2D DFT sampled along the line (bilinear), which
  coincide (exactly at 0° and 90°, within interpolation error otherwise; a readout gives
  the difference). Toggle real/imaginary/magnitude. Tab *direct reconstruction*: number of
  angles slider, spokes drawn into an empty spectrum, gridding nearest / bilinear, inverse
  FFT shown with its error image. Images: stripes at a chosen angle (from `fft.ts`
  `stripe`), disc, Shepp–Logan, `cell256.png`. Presets: "stripes: one angle sees them, the
  other doesn't", "projection ↔ slice match", "spokes are sparse at the edge", "nearest
  gridding: ripples everywhere". 256×256 with `fft.ts` (`fft2d` once per image, `fft1d` per
  projection): well under 30 ms.
- **Demo `backproj`** with `data-filter="none"` (introduced here, completed in unit 3): the
  sinogram on the left with the current column highlighted; the reconstruction building up
  angle by angle on the right; step / play / number of angles. Preset "one point → star".
- By hand: (a) back-project the three-point projections of unit 1 at 0° and 90° on a 5×5
  grid (add row and column values): the true points are bright, but so are "ghost"
  crossings; (b) vertical stripes with period 8 in a 64×64 image: the projection onto $x$
  is a cosine with 8 cycles over 64 pixels, its 1D DFT has peaks at $\pm 8$; the projection
  onto $y$ is constant (only DC). Mark both on the 2D spectrum (two dots at $(\pm 8, 0)$)
  and check that each projection's spectrum is the slice at its angle.

### 3. `fbp/`: Filtered back-projection and sampling

- The fix for the $1/r$ blur: before smearing back, **filter each projection** so that the
  overlap of all the smears cancels the halo. Picture: the filtered projection of a point
  has a positive peak and **negative sidelobes**; the negative parts of all the smears
  cancel the glow, leaving a point.
- Why a **ramp**: in the Fourier picture, spokes crowd at the centre; each spoke's weight
  should grow with distance from the centre, in proportion to the gap between spokes,
  i.e. $|\omega|$. Low frequencies (over-counted) are turned down, high ones up. Filtering
  is done in 1D, on each projection: cheap.
- **The math: deriving FBP** {.math}: write the 2D inverse Fourier transform in polar
  coordinates; the Jacobian $|\omega|$ appears;
  $f(\mathbf x) = \int_0^\pi \int P_\theta(\omega)\,|\omega|\,e^{2\pi i \omega\,\mathbf x\cdot\mathbf n_\theta}d\omega\,d\theta$
  = back-projection of ramp-filtered projections. Discrete ramp (Ram–Lak) kernel with
  bin spacing 1: $h[0] = \tfrac14$, $h[n] = -\tfrac{1}{\pi^2 n^2}$ for odd $n$, $0$ for even
  $n \ne 0$. Filtering via FFT needs zero padding to $2N$ (otherwise circular
  convolution wraps one edge into the other: cupping); one line linking to the
  convolution theorem.
- **Noise**: the ramp boosts the highest frequencies, where there is mostly noise. Few
  photons (low dose) → noisy projections → grainy reconstruction. **Tapered ramps**:
  multiply the ramp by a taper that rolls off at high frequency (recall `fourier/` tapers): Shepp–Logan (sinc), cosine,
  Hann; and a cutoff frequency. Trade: less noise, less resolution. Same trade as
  low-pass filtering in `fourier/`.
- **The math: noise in FBP** {.math}: for white noise of variance $\sigma_n^2$ per bin, the
  variance of the reconstruction scales as $\sigma_n^2 \int |\omega|^2 W(\omega)^2 d\omega$; with a
  hard cutoff $\omega_c$ that is $\propto \omega_c^3$: halving the pixel size at equal dose
  costs much more than a factor 2 in noise. *(Stated, the scaling checked in the demo.)*
- **How many angles?** The Fourier picture again: at the outer ring (radius $N/2$) the gap
  between neighbouring spokes is $\pi/N_\theta \cdot N/2$; for it to be at most one frequency
  step, $N_\theta \gtrsim \tfrac{\pi}{2} N$. For 256 bins about 400 angles over 180°.
  Too few angles: **streaks** radiating from high-contrast edges (each back-projected
  smear is no longer cancelled by its neighbours). Too few bins: blur.
- **Limited angle**: in electron tomography the specimen holder only tilts to about ±60–70°.
  The missing angles leave a **missing wedge** in the 2D spectrum (no spokes there):
  objects are elongated along the beam direction, edges parallel to the missing
  directions vanish. Picture: a disc becomes a lemon. Can't be fixed by filtering;
  priors (unit 4) help.
- **Demo `backproj`** (new, main demo): sinogram (from the same projector as `radonproj`) →
  filtered sinogram → reconstruction. Filter menu: none (unit 2), ramp, Shepp–Logan, cosine,
  Hann; cutoff slider; a plot of the filter's frequency response and of the filtered
  projection of the current angle (negative sidelobes visible). Step / play through angles
  so the cancellation is seen happening. Noise: a dose slider (seeded Poisson counts
  $I_0 e^{-p}$, then $-\ln$). Readouts: RMSE and PSNR vs the phantom (`lib/metrics.ts`),
  noise std in a flat region. Toggle *no zero padding* (shows the wrap-around cupping).
  Presets: "plain: glow", "ramp: sharp", "ramp, low dose: grain", "Hann: smoother, softer",
  "one point: ramp kernel cancels the star", "forgot to pad". 128×128 at 180 angles: FFT
  filtering + back-projection ≈ 3 M samples, ~10 ms in TS; 256×256 on demand.
- **Demo `tomosampling`** (new, small): one phantom, sliders for number of angles, angular
  range (180° down to ±45°), detector bins, dose; FBP with Hann. Three panels:
  reconstruction, error vs truth, and the 2D spectrum with the measured spokes overlaid
  (the missing wedge appears as an empty sector). A marker on the angle slider at
  $\tfrac{\pi}{2}N$. Presets: "enough angles", "18 angles: streaks", "±60°: missing wedge",
  "few bins: blur", "low dose vs few angles at equal total photons" (same number of photons
  spread over many noisy views or few clean ones: a classic design question).
- By hand: (a) filter the box projection $(0, 0, 1, 1, 1, 0, 0)$ with the truncated kernel
  $(-0.1, 0.25, -0.1)$: result $(0, -0.1, 0.15, 0.05, 0.15, -0.1, 0)$: edges raised, negative
  just outside; explain how the negative parts cancel the glow. (b) For a 512-bin detector,
  how many angles over 180°? (≈ 800). (c) A tilt series ±60° in 2° steps: how many images,
  and which sector of the spectrum is empty? (61 images; the 60° wedge around the beam
  axis.) (d) Sketch what a horizontal and a vertical line look like after reconstruction
  from ±60° data when the beam is vertical at 0° tilt.

### 4. `algebraic-recon/`: Reconstruction as a linear system

- Discretise: the unknown image is a vector $\mathbf x$ (one value per pixel), each
  measurement is a weighted sum of pixels along its ray, so $\mathbf b = A\mathbf x$. Row $i$
  of $A$ = which pixels ray $i$ passes through (weights = length inside the pixel, or the
  bilinear weights of unit 1). $A$ is huge but **sparse**: a 256² image with 400 angles ×
  256 bins is a 100 000 × 65 536 matrix, but each row touches only a few hundred pixels
  (under 1 %).
- By hand (2×2): $\mathbf x = (a, b, c, d)$; row sums and column sums give a 4×4 $A$ of rank 3.
  The images $\begin{psmallmatrix}1&2\\3&4\end{psmallmatrix}$ and
  $\begin{psmallmatrix}2&1\\2&5\end{psmallmatrix}$ give the same data $(3, 7, 4, 6)$: their
  difference $(1, -1, -1, 1)$ is in the **null space**. One diagonal ray ($a + d$) tells them
  apart. More angles = more rows = smaller null space: the linear-algebra version of "too
  few angles".
- $A^\top$ applied to the data = **back-projection** (each measurement is added back to the
  pixels on its ray, with the same weights): the smear of unit 2 is just the transpose.
  The inverse is not available (non-square, rank-deficient, noisy), so: **least squares**,
  solved iteratively.
- **Kaczmarz / ART** (algebraic reconstruction technique): take one ray at a time; change
  the image as little as possible so that this ray's sum is right (spread the error evenly
  over its pixels). Sweep over all rays, repeat. Picture with **two pixels**: each
  measurement is a line in the $(x_1, x_2)$ plane; ART zig-zags between the lines toward
  their intersection; nearly parallel lines (similar rays) = slow; with noise the lines
  do not meet in one point and ART circles around (relaxation $\omega < 1$ calms it; $\omega$, not λ, which is reserved for the prior weight).
- **SIRT**: compute all ray errors first, back-project them all at once (averaged), update.
  Smoother and more robust to noise than ART, slower to converge. Its **first step from a
  zero image is plain back-projection** (normalised): iteration fixes the blur step by step.
- **Recall (one slide, day 4):** inverting a forward model amplifies noise where the model
  loses information; tomography is that ill-posed inverse problem with $A$ = the projector.
  (Deconvolution methods themselves: deconvolution day, pointer.)
- **Semi-convergence**: with noisy data the error to the truth first falls, then rises as
  the iteration starts fitting the noise. Stopping early is a form of regularisation.
- **Priors**: what we know about the object that the data don't say. Non-negativity
  (attenuation $\ge 0$: clip after each step) and **total variation** (the image is mostly
  piecewise constant; TV is owned by the Ising/MRF elective): alternate "agree with the data" (a SIRT step)
  and "be piecewise constant" (a TV denoising step). With 20–30 views or a missing wedge,
  TV gives clean reconstructions where FBP streaks; the price: lost fine texture and
  staircased ramps (same failure as in 1D). One slide: iterative methods with priors are
  why low-dose CT is possible; also standard in electron tomography.
- **The math: Kaczmarz and SIRT** {.math}: Kaczmarz update
  $\mathbf x \leftarrow \mathbf x + \omega\,\dfrac{b_i - \mathbf a_i^\top\mathbf x}{\|\mathbf a_i\|^2}\,\mathbf a_i$
  (orthogonal projection onto the hyperplane $\mathbf a_i^\top\mathbf x = b_i$); from $\mathbf x_0 = 0$ on
  a consistent system it converges to the **minimum-norm** solution (it never leaves the
  row space). SIRT: $\mathbf x \leftarrow \mathbf x + \omega\,W_c A^\top W_r(\mathbf b - A\mathbf x)$ with
  $W_r$, $W_c$ the inverse row and column sums (not $C$/$R$: $C$ is avoided course-wide): gradient descent on a weighted
  $\|A\mathbf x - \mathbf b\|^2$ (Landweber with $W_c = W_r = I$); normal equations
  $A^\top A\mathbf x = A^\top\mathbf b$. Semi-convergence via the SVD: the filter factors
  of early stopping (one slide, optional).
- **The math: regularised reconstruction** {.math}: $\min_{\mathbf x \ge 0}\
  \tfrac12\|A\mathbf x - \mathbf b\|^2 + \lambda\,\mathrm{TV}(\mathbf x)$,
  $\mathrm{TV}(\mathbf x) = \sum_p \|\nabla x_p\|$ (smoothed by $\sqrt{\|\nabla x_p\|^2 + \varepsilon^2}$
  for gradient steps); the same "data + λ·prior" form as `seg-energy/` (day 13) and the Ising elective, with $A \ne I$. One
  line: for emission data (PET) the data term is the Poisson log-likelihood and the
  iteration becomes multiplicative (MLEM, $\mathbf x \leftarrow \tfrac{\mathbf x}{A^\top \mathbf 1}\,
  A^\top\tfrac{\mathbf b}{A\mathbf x}$), stated only; it is Richardson–Lucy with $A$ = the projector
  (pointer to the deconvolution day).
- **Demo `artsmall`** (new, small, the by-hand exercise made interactive): mode *2 pixels*:
  the $(x_1, x_2)$ plane with 2–4 measurement lines (draggable), the ART zig-zag step by
  step, relaxation slider, noise toggle (inconsistent lines → limit cycle), SIRT path for
  comparison. Mode *2×2 / 3×3 image*: the tiny image with its rays drawn, the matrix $A$ as a
  grid of 0/1 (or lengths) with the current row highlighted, data vector, current $\mathbf x$,
  residual; *next ray* shows the correction spread along the ray; angle set menu (0°+90°,
  +45°, +135°) showing the rank and a null-space image. Presets: "2×2: two images, same
  data", "add a diagonal ray", "ART converges in one sweep", "nearly parallel rays: slow",
  "noisy: ART circles, ω = 0.3 settles".
- **Demo `iterrecon`** (new, main demo): 128×128 phantom; sliders for number of angles,
  angular range, dose; method menu: FBP (Hann), ART, SIRT, SIRT + non-negativity, SIRT + TV
  ($\lambda$ slider); iteration play / step / slider. Panels: reconstruction, error image,
  plot of error vs iteration (and data residual vs iteration, so semi-convergence is seen
  as the two curves disagreeing). Presets: "SIRT step 1 = back-projection", "SIRT
  converging", "noisy: stop early", "30 views: FBP streaks vs TV clean", "±60° wedge: TV
  fills in", "TV too strong: cartoon". Cost: one SIRT iteration = one projection + one
  back-projection ≈ 2 × 3 M samples at 128² × 180 angles ≈ 20–40 ms in TS: play at a few
  iterations per second. Measure; move the projector pair to Rust/Wasm only if 256² or
  live dragging is needed.
- By hand: (a) the 2×2 example above: write $A$, find the null-space vector, check both
  images; (b) one Kaczmarz sweep from $\mathbf x = 0$ over the four rays in the order
  row 1, row 2, column 1, column 2: $(1.5, 1.5, 0, 0) \to (1.5, 1.5, 3.5, 3.5) \to (1, 1.5, 3, 3.5)
  \to (1, 2, 3, 4)$: it returns $\begin{psmallmatrix}1&2\\3&4\end{psmallmatrix}$ even if the
  truth was $\begin{psmallmatrix}2&1\\2&5\end{psmallmatrix}$: the data cannot tell, and ART
  picks the smallest image; (c) one SIRT step from 0 with averaging: $(1.75, 2.25, 2.75, 3.25)$:
  the right ordering, but blurred toward the mean, i.e. plain back-projection; (d) compute
  $A^\top \mathbf b$ and compare with (c).

### 5. (optional) `ct-artefacts/`: What goes wrong in real scanners

- Each artefact as "what happens in the sinogram → what it does to the image":
  - **Dead or miscalibrated detector bin**: a horizontal line in the sinogram (same $\rho$ at
    every angle) → a **ring** around the centre. Fix: interpolate or filter the sinogram.
  - **Metal**: almost no photons get through → $-\ln$ of a near-zero, very noisy count →
    bright and dark **streaks** between metal objects. Fix: treat those rays as missing
    and inpaint the sinogram, or iterate with a prior.
  - **Beam hardening**: a real beam has many energies, soft ones are absorbed first, so
    $-\ln(I/I_0)$ grows *less* than linearly with thickness → **cupping** (centre of a uniform
    object too dark) and dark bands between dense objects. Math layer: $-\ln\big(\sum_E w_E
    e^{-\mu_E L}\big)$ is concave in $L$.
  - **Motion** between angles: inconsistent projections → doubled edges and streaks.
  - **Wrong centre of rotation**: each point becomes a small ring/arc ("tuning-fork"
    artefact); estimate the centre from the 0°/180° mirror symmetry of unit 1.
  - Truncation (object wider than the detector): bright rim.
- Other geometries in words and one picture each: fan beam (rebinning to parallel), helical
  and cone-beam CT (3D), the stack of 2D slices in parallel-beam 3D; PET/SPECT with
  attenuation correction.
- **Demo `ctartefacts`** (new, small): a phantom with optional metal inserts; toggles that
  corrupt the sinogram (dead bin, metal photon starvation, polychromatic beam with a
  thickness slider, motion at one angle range, centre-of-rotation offset); shows corrupted
  sinogram, FBP reconstruction, and one simple fix per artefact (sinogram ring filter,
  metal-ray inpainting by linear interpolation along $\rho$, linearisation by a lookup
  table, centre search by 0°/180° correlation). Presets, one per artefact.
- By hand: given a sinogram sketch with a horizontal bright line at $\rho = 10$, predict the
  artefact and its radius; given $I/I_0$ for 1, 2, 4 cm of a material with a two-energy beam,
  compute $-\ln(I/I_0)$ and show it is not proportional to thickness.

### 6. (optional) `radon-analysis/`: The Radon transform as an image-analysis tool

- No tomography: the Radon transform of an ordinary image as a feature.
- **Dominant orientation**: a pattern of lines at angle $\alpha$ gives projections whose
  **variance** is largest at $\theta = \alpha$ (lines line up with the rays). Variance of each
  sinogram column vs $\theta$: the peak is the orientation. Uses: document skew, fibre and
  texture direction in micrographs, stripe artefacts. Compare (one slide) with the
  structure tensor (day 2, `structure-tensor/`) and the 2D spectrum orientation (`fourier/`).
- **Line detection**: bright lines are peaks in the sinogram; for a binary edge map this is
  the Hough transform (one slide; pointer to the Hough elective, which owns peak finding and
  voting).
- **Demo**: `radonproj` with `data-mode="orient"`: adds a plot of column variance vs
  $\theta$ and the detected orientation drawn on the image; images: stripes, a fibrous texture
  (synthetic, seeded), a skewed text-like block pattern. No new demo.
- By hand: the projections of a 4×4 image of two diagonal lines at 0°, 45°, 90°, 135°;
  which angle has the largest variance?

## Shared engineering

- **`demos/src/lib/radon.ts`** (new): parallel-beam **projector / back-projector pair** on
  flat `Float32Array`s: ray-driven projection with bilinear weights (sampling step ≤ 0.5 px),
  and back-projection as its **exact transpose** (scatter the same weights), so that
  $\langle A\mathbf x, \mathbf y\rangle = \langle \mathbf x, A^\top\mathbf y\rangle$ holds to rounding (needed for
  "back-projection = $A^\top$" and for SIRT to converge). Also: angle sets (full,
  limited range), sinogram ↔ image coordinate conventions ($\rho$ origin at the image centre,
  $\theta \in [0°, 180°)$, same as Hough), ramp and tapered filters with zero-padded FFT
  filtering, FBP, ART/SIRT iterations (instrumented: per-iteration error and residual),
  explicit sparse $A$ (CSR) for images up to 16×16 for `artsmall`, analytic projections of
  ellipses (Shepp–Logan) for tests. One implementation shared by `radonproj`, `fourierslice`,
  `backproj`, `tomosampling`, `iterrecon`, `ctartefacts`. Owner: this day; the Hough elective imports its forward projector for its one Radon slide.
- **Phantoms** in `lib/synth.ts` (owner: thresholding day; Shepp–Logan and `window16`'s
  `ctPhantom` move there): modified Shepp–Logan from its ellipse table (the analytic sinogram
  stays in `lib/radon.ts`), disc, square, point masses, metal inserts, fibrous texture;
  seeded with `rng(seed)` from `lib/random.ts`.
- **Reuse**: `lib/fft.ts` (moved from `fourier2d/fft.ts`, owner `fourier/`; `fft1d` for
  ramp filtering, `fft2d` for `fourierslice` and the spectrum panel of `tomosampling`);
  bilinear sampling from `lib/imageops.ts`; polar ↔ Cartesian resampling in `lib/polar.ts`
  (owner: dynprog elective); `lib/plot.ts` (owner: morphology day; profiles, filter
  responses, error vs iteration); `lib/metrics.ts` (RMSE/PSNR, image-quality section owned
  by `cameranoise/`); colour maps from `lib/colour.ts` (owner: colour day; sequential for
  sinograms, diverging for filtered projections and error images); 2D TV step from
  `lib/smooth.ts` (owner: Ising elective; no private TV in `lib/radon.ts`).
- **Poisson noise**: `poisson()` from `lib/random.ts` (owner: bit-depth day, moved from
  `camerabits/sensor.ts`): counts $\sim \mathrm{Poisson}(I_0 e^{-p})$, then
  $-\ln(\text{count}/I_0)$ with a floor at 1 count (and a visible "photon starvation" flag).
- **Performance**: start in TS at 128×128 and 180 angles (≈ 3 M bilinear samples per
  projection or back-projection, ~10–20 ms). Measure `iterrecon` at 256²; only if one
  SIRT+TV iteration exceeds ~30 ms move the projector pair to `demos/wasm/radon/` (Rust,
  plain functions over `&[f32]`), keeping `lib/radon.ts` as the reference implementation
  and testing that the two agree.
- **Tests** (`demos/test/radon-test.ts`, run by the shared `make test`): adjoint test $\langle A\mathbf x, \mathbf y\rangle =
  \langle \mathbf x, A^\top \mathbf y\rangle$ on random inputs; projections of the Shepp–Logan phantom vs
  the analytic sinogram (small RMSE); FBP of the analytic sinogram reconstructs the phantom
  within a tolerance; Fourier slice: 1D DFT of the 0° projection equals the $v = 0$ row of the
  2D DFT exactly; ART on the 2×2 example reproduces the by-hand numbers.
- Images: `cell256.png` (credited as in `fourier/`); a real CT slice or sinogram only with a
  clear open licence (see open questions). Follow `demos/README.md`: fixed seeds, presets +
  reset, keyboard back to reveal.js, fits 1150×560 on a slide, static figures as fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, three points by hand, `radonproj` |
| 10:45–12:15 | Unit 2 lecture, back-projection and stripes by hand, `backproj` (plain), `fourierslice` |
| 13:15–14:45 | Unit 3 lecture, ramp kernel by hand, `backproj`, `tomosampling` |
| 15:00–16:30 | Unit 4 lecture, 2×2 Kaczmarz by hand, `artsmall`, `iterrecon`; quizzes as wrap-up |

## Open questions

- Length of a day: four dense units. If too much, direct Fourier reconstruction (gridding)
  in unit 2 shrinks to one slide, and the TV part of unit 4 moves to the optional units.
- **Decided:** unit 4 (algebraic reconstruction) stays core. review-gaps made this
  conditional on the image-formation day not carrying the inverse-problem idea; day 4 now
  only motivates ill-posedness and deconvolution has its own day, so unit 4 recalls day 4
  and remains the course's main worked inverse problem in a linear-algebra setting.
- Fan-beam geometry: words only (as planned) or a demo mode? Parallel beam keeps the math
  clean; fan beam is what clinical CT uses.
- 3D: a cone-beam or stacked-slice reconstruction shown in `volview` (graphics3d), or stay
  2D? Recommended 2D, with one pointer slide.
- Real data: a real sinogram (e.g. an openly licensed micro-CT dataset with a
  CC BY licence) would make principle 4 concrete; needs a licence check and downsampling to
  a few hundred kB. Otherwise phantom + `cell256.png` only.
- **Decided (course conventions):** Hough and Radon both use $\theta \in [0°, 180°)$, signed
  $\rho$ from the image centre, the same sinogram orientation; Hough imports `lib/radon.ts`
  for its one slide.
- **Decided:** `fft.ts` moves to `lib/fft.ts` (owner `fourier/`).
- **Decided (review-organisation):** the continuous Fourier transform is added as a math
  slide to `fourier/`; this day recalls it.

## References

- Radon (1917), *Über die Bestimmung von Funktionen durch ihre Integralwerte längs gewisser
  Mannigfaltigkeiten*, Ber. Verh. Sächs. Akad. Wiss. Leipzig (English translation in IEEE
  Trans. Med. Imaging 1986).
- Bracewell (1956), *Strip integration in radio astronomy*, Aust. J. Phys. (projection-slice).
- Cormack (1963), J. Appl. Phys.; Hounsfield (1973), Br. J. Radiol. (the first CT).
- Shepp & Logan (1974), *The Fourier reconstruction of a head section*, IEEE Trans. Nucl. Sci.
- Gordon, Bender & Herman (1970), *Algebraic reconstruction techniques (ART)…*, J. Theor.
  Biol.; Kaczmarz (1937); Gilbert (1972), SIRT; Andersen & Kak (1984), SART, Ultrasonic Imaging.
- Kak & Slaney (1988), *Principles of Computerized Tomographic Imaging*, IEEE Press (SIAM
  reprint 2001); the standard readable text, chapters 3 and 7.
- Natterer (1986), *The Mathematics of Computerized Tomography*, Wiley (SIAM reprint 2001).
- Hansen, Jørgensen & Lionheart (2021), *Computed Tomography: Algorithms, Insight, and Just
  Enough Theory*, SIAM (semi-convergence, regularisation; close to this day's level).
- Buzug (2008), *Computed Tomography: From Photon Statistics to Modern Cone-Beam CT*, Springer
  (artefacts, beam hardening).
- Sidky & Pan (2008), *Image reconstruction in circular cone-beam CT by constrained,
  total-variation minimization*, Phys. Med. Biol.
- Shepp & Vardi (1982), *Maximum likelihood reconstruction for emission tomography*, IEEE
  Trans. Med. Imaging (MLEM).
- Frank (ed.) (2006), *Electron Tomography*, Springer (missing wedge).
- Deans (1983), *The Radon Transform and Some of Its Applications*, Wiley.
