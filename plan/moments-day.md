# Plan: a day on moment-based methods

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. define the **moment of order $p+q$** of an image (continuous and discrete) as the projection of $f$ onto the monomial $x^p y^q$, read the low orders as mass, mean, covariance, skewness and kurtosis of $f$ treated as a 2D distribution, and compute a small table of moments by hand;
2. make moments **invariant** to translation (central moments), scale (normalised moments $\eta_{pq}$) and rotation (Hu's invariants, complex moments), derive the first two Hu invariants by hand, and explain why $\phi_7$ changes sign under mirroring;
3. **normalise the pose** of a shape (centroid, scale, principal axes) and say when this fails (equal eigenvalues, 180° ambiguity);
4. explain why monomials are a **poor basis** (non-orthogonal, ill-conditioned Gram matrix, redundant information), orthogonalise them by **Gram–Schmidt** into **Legendre polynomials**, and **reconstruct** an image from its moments up to order $N$ as an orthogonal projection, like truncating a Fourier series;
5. compute **Zernike moments** on the unit disc, show that $|Z_{nm}|$ is rotation invariant, and name the discretisation errors (square pixels on a round domain, numerically unstable radial polynomials at high order);
6. use moment invariants as a **feature vector** to tell shapes apart (nuclei, leaves, digits), and judge their robustness to noise and low resolution.

Why it fits this audience: moments are integrals of polynomials, the projection and
Gram–Schmidt arguments are linear algebra they already have, and Zernike moments are a
Fourier series in the angle times a polynomial in the radius, in polar coordinates they know
from calculus. The day reuses one idea from *The Fourier transform* almost verbatim: a
coefficient is an inner product with a basis function, and an orthogonal basis makes
truncation the best approximation. The invariants are a clean example of designing a
quantity so that a group of transformations (the matrices of *Images and coordinates*) leaves
it unchanged.

## Prerequisite gap

- **Moments of order 0–2 as region properties** (area, centroid, central moments,
  second-moment matrix, orientation, equivalent ellipse, eccentricity) are taught on the
  labelling day (`regionprops/`, demo `moments`). This day assumes that material and recalls
  it in two slides (the formulas and one screenshot of the equivalent ellipse); the
  orientation derivation is not repeated. If a group meets moments here first, the
  `regionprops/` slides on central moments and the second-moment matrix can be shown as a
  prelude (about 20 minutes); the rest of that unit (perimeter, Euler number) is not needed.
- **Binary masks / segmentation**: shape moments need a mask of one object. One slide:
  threshold and label (labelling and thresholding days); the demos generate their own shapes,
  so no segmentation code is needed.
- **Orthogonal functions and projection**: known from linear algebra (inner product,
  orthogonal projection, Gram–Schmidt on vectors) and from *The Fourier transform*. The step
  from vectors to functions (inner product = integral) is new to some and gets one slide
  with the 1D picture.
- **Polar coordinates and $e^{i\theta}$**: known from calculus and *The Fourier transform*;
  one-line recall that multiplying by $e^{i\alpha}$ rotates by $\alpha$.
- **Classification**: no machine learning is assumed. Unit 5 uses only nearest neighbour and
  nearest class mean in a feature space, explained with a scatter plot.

## Links to other days

- **Labelling** (`plan/labelling-day.md`): owns orders 0–2 (region properties). This day owns
  general order, normalisation, invariants, orthogonal moments and reconstruction. The
  labelling day's `moments` demo is extended here with a new mode rather than duplicated; its
  `regionfilter` measurement table can gain Hu and Zernike columns (unit 5 here), which makes
  its scatter plot a moment-feature space.
- **Thresholding** (`plan/thresholding-day.md`, being written): Tsai's moment-preserving
  threshold chooses the threshold so that the binary image has the same first three grey-level
  moments as the histogram. It is a 1D use of the same definitions; one slide pointer in unit
  1 here, or a full treatment there, whichever day covers thresholding methods in depth.
- **Snakes / shape models** (`plan/snakes-day.md`): Procrustes aligns *landmark* shapes and
  needs point correspondence; moment-based pose normalisation (unit 2) aligns *regions* without
  any correspondence, at the cost of being less precise and failing for symmetric shapes. The
  shape families in `lib/shapes.ts` (nucleus/kidney, leaf) are reused for classification.
  Point distribution models and moment invariants are two answers to "describe a shape with a
  few numbers".
- **Skeletons** (`plan/skeleton-day.md`): moments describe mass distribution and are blind to
  topology and thin branches (a neuron and a blob of the same second moments look alike);
  skeleton features (branch count, length) are the complementary descriptors. One slide in
  unit 5.
- **Marching cubes / 3D** (`plan/marchingcubes-day.md`): 3D moments of a labelled volume give
  the $3\times3$ inertia tensor and principal axes (optional unit 6); moments of a closed mesh
  follow from the divergence theorem, the 3D counterpart of the Green's theorem slide in unit 1.
- **Dynamic programming** (`plan/dynprog-day.md`): its polar unwrapping (`lib/polar.ts`, if it
  exists) is the same resampling that turns Zernike moments into an angular Fourier series.
- **Fourier** (`fourier/`): moments are the Taylor coefficients of the Fourier transform at
  zero frequency, so low-order moments describe the coarse shape (unit 1); reconstruction from
  orthogonal moments is a truncated expansion like low-pass filtering in `fourier2d` (unit 3).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units.

### 1. `moments-basis/`: Moments as projections onto monomials

- Picture first: the same shape drawn as a binary mask and as a grey image; "which few numbers
  summarise it?" Recall from the labelling day: $m_{00}$, centroid, $C$, equivalent ellipse.
  The ellipse misses everything that is not an ellipse (the L-shape, a pear): we need higher
  orders.
- Definition, continuous: $m_{pq} = \iint x^p y^q f(x, y)\,dx\,dy$, i.e. the inner product
  $\langle f, x^p y^q\rangle$; discrete: $m_{pq} = \sum_x \sum_y x^p y^q f(x, y)$. Order $p+q$;
  there are $(N+1)(N+2)/2$ moments up to order $N$. Binary masks and grey images use the same
  formula ($f$ = mask or intensity).
- **Probability view**: $f/m_{00}$ is a 2D density. Order 1: mean; order 2: covariance (the
  labelling day's $C$); order 3: **skewness** (which way the shape leans: a pear vs its
  mirror, the sign of $\mu_{30}$); order 4: **kurtosis** (flat vs peaked, a square vs a
  disc). Marginal moments $\mu_{p0}$ are moments of the projection of $f$ onto the $x$-axis.
- **Uniqueness**: a bounded image with compact support is determined by the full infinite
  set of its moments (stated, not proved; Hu 1962). A finite set is not enough, which is the
  question of unit 3.
- **Link to Fourier**: expanding $e^{-2\pi i(ux+vy)}$ in a Taylor series gives
  $F(u, v) = \sum_{p,q} \frac{(-2\pi i)^{p+q}}{p!\,q!}\,m_{pq}\,u^p v^q$. Moments are the
  derivatives of the spectrum at zero frequency: low orders are coarse shape, high orders
  need high frequencies.
- **Computing them**: separable, $M = V_x^\top F V_y$ with Vandermonde matrices
  $(V_x)_{xp} = x^p$: one matrix sandwich, cost $O(N\cdot WH)$. Map coordinates to $[-1, 1]$
  first: $255^{10} \approx 10^{24}$ in pixel units ruins float precision. Pixel as a point
  vs pixel as a unit square: integrating over the square adds exactly $m_{00}/12$ to
  $\mu_{20}$ (the variance of a uniform distribution on a unit interval), a first taste of
  discretisation error. For a polygon (snake, marching-squares contour), Green's theorem turns
  the area integral into a sum over the edges (one slide, formula for $m_{00}$ and $m_{10}$).
- Pointer: Tsai's moment-preserving thresholding (1D moments of the histogram; see the
  thresholding day).
- **Demo `moments`** (labelling day's demo) with `data-mode="higher"`: same canvas, shapes and
  handles (drag, rotate, scale) plus a *mirror* button and a grey-image option (a blurred or
  textured shape). Side panel: a triangular table of $m_{pq}$ (or $\mu_{pq}$, toggle) up to
  order 4, coloured by sign and magnitude; a *pixel = point / square* toggle with the
  difference shown. Overlay: the marginal projections along $x$ and $y$ as small bar plots,
  with mean, $\pm$ std and skewness arrow. Presets: "a pear and its mirror: the sign of
  $\mu_{30}$", "square vs disc: same $C$, different order 4", "L-shape: the ellipse is not
  enough", "grey vs binary".
- By hand: (a) the continuous moments of the rectangle $[0,a]\times[0,b]$,
  $m_{pq} = a^{p+1}b^{q+1}/((p+1)(q+1))$, then its central moments up to order 2. (b) A 4×4
  grey image: all $m_{pq}$ with $p+q \le 2$ and $\mu_{30}$; which way is it skewed? (c) Check
  $\mu_{20}$ of one pixel as a point (0) and as a square (1/12).

### 2. `moment-invariants/`: Normalisation and invariants

- **Translation**: central moments $\mu_{pq}$ (recall, one line). Expressed in raw moments by
  the binomial theorem (e.g. $\mu_{30} = m_{30} - 3\bar x m_{20} + 2\bar x^2 m_{10}$), so no
  second pass over the image is needed.
- **Scale**: $f(x/s, y/s)$ has $\mu'_{pq} = s^{p+q+2}\mu_{pq}$ (substitute in the integral, by
  hand). Normalised moments $\eta_{pq} = \mu_{pq} / \mu_{00}^{1 + (p+q)/2}$. For grey images
  a contrast change $af$ also cancels in $\eta_{pq}$.
- **Rotation, small example first**: rotate the coordinates in $\mu_{20}, \mu_{11}, \mu_{02}$
  by hand; $\phi_1 = \eta_{20} + \eta_{02}$ (the trace of $C$, scaled) and
  $\phi_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2$ (the squared eigenvalue difference) do
  not change. Linear algebra names both: trace and eigenvalues of a symmetric matrix are
  invariant under $Q^\top C Q$. Fact for the quiz: the disc has the smallest $\phi_1$ of all
  shapes, $1/(2\pi) \approx 0.159$ (the square has $1/6$).
- **Complex moments** make the general case systematic:
  $c_{pq} = \iint (x + iy)^p (x - iy)^q f\,dx\,dy$. Rotation by $\alpha$ multiplies $c_{pq}$ by
  $e^{i(p-q)\alpha}$. Invariants: $|c_{pq}|$, and any product whose phases cancel
  ($c_{30}c_{12}^3$, $c_{20}c_{12}^2$). **Hu's seven invariants** follow (one slide with all
  seven in complex form: $\phi_1 = c_{11}$, $\phi_2 = |c_{20}|^2$, $\phi_3 = |c_{30}|^2$,
  $\phi_4 = |c_{21}|^2$, $\phi_5 = \operatorname{Re}(c_{30}c_{12}^3)$,
  $\phi_6 = \operatorname{Re}(c_{20}c_{12}^2)$, $\phi_7 = \operatorname{Im}(c_{30}c_{12}^3)$,
  with central normalised moments); the long real-valued formulas are shown once, not
  derived.
- **Mirror**: reflection $y \to -y$ conjugates $c_{pq}$, so $\phi_1..\phi_6$ stay and
  $\phi_7$ changes sign: a "skew invariant" that tells a shape from its mirror image (or must
  be dropped / used as $|\phi_7|$ if mirrors should match).
- **Practical problems**: the values span many orders of magnitude (order-3 invariants are
  powers of small numbers), so features are compared as $-\operatorname{sign}(\phi)\log_{10}|\phi|$;
  high orders amplify noise at the boundary; Hu's set is neither independent ($\phi_3$ follows
  from $\phi_4, \phi_5, \phi_7$) nor complete (Flusser 2000 gives a basis). Discretisation:
  rotating a 20-pixel-wide shape changes $\phi$ by several per cent; invariance holds for the
  continuous image only. Affine invariants (Flusser & Suk) named for slanted views.
- **Pose normalisation** as the alternative to invariants: translate to the centroid, scale to
  $m_{00} = 1$, rotate the principal axis onto $x$ (orientation from the labelling day).
  Remaining 180° ambiguity: fix the sign of $\mu_{30}$. Fails when $\lambda_1 \approx \lambda_2$
  (square, star, gear): then use the phase of a higher complex moment $c_{k0}$ for $k$-fold
  symmetry. After normalisation shapes can be compared pixel by pixel (link to Procrustes on
  the snakes day: same goal, no landmarks needed).
- **Demo `moments`** with `data-mode="invariants"`: the shape canvas as in unit 1, plus a
  reference copy frozen by a *set reference* button. Panel: $\mu_{pq}$, $\eta_{pq}$ and
  $\phi_1..\phi_7$ (log scale bars) for the current shape and the reference, with relative
  change per row coloured green (< 1 %) to red. A *resolution* slider re-digitises the shape;
  a *pose-normalise* toggle draws the normalised shape in an inset. Presets: "translate: $m$
  changes, $\mu$ does not", "scale: $\mu$ changes, $\eta$ does not", "rotate: $\eta$ changes,
  $\phi$ does not", "mirror: only $\phi_7$ flips", "small shape: discretisation breaks
  invariance", "square: pose normalisation is ambiguous".
- **Demo `momentphase`** (new, small): the complex plane with arrows for $c_{20}, c_{11},
  c_{30}, c_{21}, c_{12}$ of a draggable/rotatable shape (shared shape canvas). Rotating the
  shape by $\alpha$ spins each arrow by $(p-q)\alpha$ (shown as a trail), so $c_{11}$ stays,
  $c_{20}$ spins twice as fast as $c_{21}$; a mirror button reflects all arrows in the real
  axis. A product $c_{30}c_{12}^3$ is drawn too and stays put. Presets: "rotate slowly",
  "mirror", "a 3-fold symmetric shape: $c_{20} = 0$".
- By hand: (a) scale a rectangle by $s$ and verify $\eta_{20}$ is unchanged. (b) Rotate the
  coordinates by $\alpha$ in $\mu_{20}, \mu_{11}, \mu_{02}$ and show that $\phi_1$ and $\phi_2$
  are invariant. (c) $c_{20}$ and $c_{11}$ of a small point set before and after a 90° rotation
  and a mirror. (d) $\phi_1$ for a disc and a square.

### 3. `orthogonal-moments/`: Better bases: Legendre moments and reconstruction

- Question from unit 1: can we get $f$ back from $m_{pq}$, $p + q \le N$? With an orthogonal
  basis it is a projection; with monomials it is a linear system with the **Gram matrix**
  $G_{pq} = \langle x^p, x^q\rangle$, on $[0,1]$ the Hilbert matrix $1/(p+q+1)$:
  condition number $\approx 10^{13}$ at $N = 10$. Picture: $x^8$ and $x^{10}$ on $[-1, 1]$ are
  almost the same curve, so their moments carry almost the same information (redundancy).
- **Gram–Schmidt** on $1, x, x^2, \dots$ with $\langle g, h\rangle = \int_{-1}^1 g h\,dx$ gives
  the **Legendre polynomials** $P_0 = 1$, $P_1 = x$, $P_2 = (3x^2 - 1)/2$, … (by hand to
  $P_2$); three-term recurrence $(n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}$ for the rest;
  $\int P_n^2 = 2/(2n+1)$.
- **Legendre moments** on the square $[-1,1]^2$:
  $\lambda_{pq} = \frac{(2p+1)(2q+1)}{4}\iint P_p(x) P_q(y) f\,dx\,dy$. Each is a fixed linear
  combination of geometric moments of order $\le p+q$: same information, better coordinates
  (triangular change of basis).
- **Reconstruction** $\hat f_N = \sum_{p+q \le N} \lambda_{pq} P_p(x)P_q(y)$: the orthogonal
  projection onto polynomials of degree $\le N$, so the best approximation in the
  least-squares sense, and adding an order never changes the earlier coefficients. Same
  argument as truncating a Fourier series (`fourier2d` low-pass); the error looks different
  (polynomial ringing piles up at the square's corners and edges, Fourier ringing is
  periodic). Moments as image compression: a 128×128 image vs 231 numbers at $N = 20$.
- **Discretisation**: sampled $P_p$ are not exactly orthogonal on a pixel grid, and the error
  grows with order; remedies: integrate the polynomials over each pixel, or use discrete
  orthogonal polynomials (Tchebichef, Krawtchouk; Mukundan 2001) that are exactly orthogonal
  on $\{0..W-1\}$ (named, one slide).
- **Demo `polybasis`** (new, 1D): the interval $[-1,1]$; left plot shows the monomials up to
  degree $N$ (slider) or the Legendre polynomials (toggle), right plot their Gram matrix as a
  heat map with the condition number. A *Gram–Schmidt step* button orthogonalises one more
  function, showing the projection being subtracted. Lower panel: a 1D signal (step, $|x|$,
  a bump, a profile across a nucleus) approximated with $N$ terms by both bases, with the
  approximation error; in float64 the monomial fit visibly breaks down at $N \approx 12$–15
  while Legendre keeps improving. Presets: "monomials look alike", "Gram–Schmidt to $P_3$",
  "fit a step", "monomials break down".
- **Demo `momentrecon`** (new, main demo of units 3–4): a 64×64 or 128×128 input (binary
  letter, binary nucleus shape from `lib/shapes.ts`, the micrograph crop from `cell256.png`,
  a drawn shape). Three panels rebuilt from moments up to order $N$ (slider 0–40): monomials
  (least-squares via the Gram system), Legendre, Zernike (unit 4), optionally a fourth panel
  with a Fourier low-pass using the same number of coefficients. Under each: the number of
  coefficients and the relative error; a small plot of error vs $N$ for all methods.
  Toggles: pixel-point vs pixel-integrated basis, noise added to the image. Presets: "low
  orders = blurry ellipse", "monomials explode", "Legendre vs Fourier ringing", "Zernike
  outside the disc", "noise in high orders".
- By hand: (a) Gram–Schmidt on $1, x, x^2$ over $[-1, 1]$ to get $P_0, P_1, P_2$ (up to
  scale). (b) The $3\times3$ Gram matrix of $1, x, x^2$ on $[-1, 1]$; why is it not diagonal?
  (c) Approximate $f(x) = |x|$ by $P_0, P_1, P_2$ (coefficients by integrals) and sketch the
  result.

### 4. `zernike/`: Zernike moments on the unit disc

- Why a disc: rotation is natural in polar coordinates, and a square domain is not rotation
  symmetric (Legendre moments of a rotated image mix in a complicated way).
- **Zernike polynomials** $V_{nm}(\rho, \theta) = R_{nm}(\rho)\,e^{im\theta}$, $n \ge 0$,
  $|m| \le n$, $n - |m|$ even. Small ones by hand: $R_{00} = 1$, $R_{11} = \rho$,
  $R_{20} = 2\rho^2 - 1$, $R_{22} = \rho^2$, $R_{31} = 3\rho^3 - 2\rho$,
  $R_{40} = 6\rho^4 - 6\rho^2 + 1$; general formula as a finite sum shown once. Orthogonal on
  the disc: $\iint V_{nm}^* V_{n'm'} = \frac{\pi}{n+1}\delta_{nn'}\delta_{mm'}$ (the angular
  part is the Fourier basis; the radial part is orthogonal with weight $\rho$).
- **Zernike moments** $Z_{nm} = \frac{n+1}{\pi}\iint_{\rho \le 1} f\,V_{nm}^*\,\rho\,d\rho\,d\theta$;
  $Z_{n,-m} = Z_{nm}^*$ for real $f$, so about $(N+2)^2/4$ distinct moments up to order $N$
  (121 at $N = 20$). Reconstruction exactly as in unit 3 (shown in `momentrecon`).
- **Rotation invariance**: rotating $f$ by $\alpha$ gives $Z_{nm}e^{-im\alpha}$, so $|Z_{nm}|$
  is invariant (by hand in one line). Each $Z_{nm}$ is a combination of complex moments
  $c_{pq}$ with $p - q = m$: the same invariants as unit 2, in an orthogonal basis.
  Translation and scale: map the disc to the shape's centroid and a radius from $m_{00}$ (or
  the farthest pixel) before computing.
- **Information and noise**: magnitudes lose the phase (a shape and its rotated copy are
  identical; so are 6 and 9). Low orders: robust, coarse; high orders: detail, but noise
  sensitive. Orthogonality means each moment carries new information (Teh & Chin 1988
  compare noise sensitivity of the moment families).
- **Discretisation errors**: (i) geometric: square pixels approximate the disc boundary, and
  pixels outside the disc are ignored (or the disc is circumscribed and contains empty
  corners); (ii) numerical: the factorial formula for $R_{nm}$ suffers cancellation at
  $n \gtrsim 30$, use a recurrence (Prata & Rusch 1989; Kintner); (iii) sampling the
  polynomial at pixel centres vs integrating. Visible in the demo as reconstruction error
  that stops falling and then rises with $N$.
- Pointer: Zernike polynomials come from optics (Zernike 1934, phase contrast) and are the
  standard description of wavefront aberrations (defocus $= R_{20}$, astigmatism, coma) in
  microscopes and telescopes.
- **Demo `zernikebasis`** (new): a gallery of $V_{nm}$ for $n \le 8$ arranged as a pyramid
  (rows $n$, columns $m$), real part, imaginary part or magnitude (toggle), on a diverging
  colour map. Clicking one shows it large with the radial profile $R_{nm}(\rho)$ as a line
  plot and an optional 3D surface (`lib/gl3d.ts`). Rotation tab: an input image and its
  rotated copy (angle slider), with bar plots of $|Z_{nm}|$ (identical) and arrows for the
  phase of selected $Z_{nm}$ (turning by $m\alpha$). Presets: "defocus and astigmatism",
  "rotate: magnitudes stay", "6 vs 9", "same $|Z|$, different shape".
- **Demo `momentrecon`**: Zernike panel (see unit 3); presets for this unit.
- By hand: (a) verify $\int_0^1 R_{20}R_{00}\,\rho\,d\rho = 0$ and compute
  $\int_0^1 R_{20}^2\,\rho\,d\rho$. (b) Show $|Z_{nm}|$ is unchanged by rotation, starting from
  the substitution $\theta \to \theta - \alpha$. (c) Which $(n, m)$ exist up to $n = 4$; how
  many distinct moments for a real image?

### 5. (optional / further reading) `moment-features/`: Shape descriptors and classification

- A shape becomes a point in a feature space: $(\log\phi_1, \dots, \log\phi_7)$ or
  $(|Z_{nm}|)$. Scaling features (z-scores) so that no coordinate dominates. Classify by
  **nearest class mean** or **nearest neighbour** (Euclidean distance); no further machine
  learning. Leave-one-out error as an honest check.
- Which invariants separate which classes: round vs kidney-shaped vs lobed nuclei, leaf
  species, digits/letters. What invariance costs: 6 and 9, M and W, b and d/p/q collapse
  under rotation invariance (and with $|\phi_7|$, under mirroring too). Choose the invariance
  the task needs, not the most.
- Robustness: boundary noise, salt-and-pepper, low resolution, partial occlusion (moments are
  global: one missing piece changes every moment, unlike local features). Moments vs
  skeleton features vs shape-model parameters (snakes day) in one comparison table.
- Moment-based **template matching**: compute invariants in a sliding window (one summed-area
  table per $x^p y^q f$ makes each window $O(1)$), compare to the template's vector: rotation
  invariant matching without trying all angles; or pose-normalise and correlate.
- **Demo `momentclass`** (new): a seeded training set of 3–4 shape classes (choice: nuclei
  from `lib/shapes.ts`; leaves from `lib/shapes.ts`; a small set of digits and letters
  defined as stroke polylines in TS, rendered with random pose, thickness and noise; no
  fonts, so the result is the same on every machine). Scatter plot of two chosen features
  (axes menu: any Hu invariant or $|Z_{nm}|$), class colours, class means; a test shape
  that can be drawn or picked, classified by nearest mean / nearest neighbour with the
  decision regions shaded. Sliders: noise, resolution, number of training shapes. Readout:
  leave-one-out error. Presets: "nuclei: $\phi_1$ vs $\phi_2$ is enough", "6 vs 9",
  "noise moves high orders first", "low resolution", "occlusion".
- By hand: (a) given a table of $(\phi_1, \phi_2)$ for 6 training shapes in 2 classes,
  classify two test shapes by nearest mean and nearest neighbour. (b) For a list of letters,
  which pairs become indistinguishable under rotation invariance? under mirroring?

### 6. (optional / further reading) `moments-3d/`: Moments of volumes

- $m_{pqr} = \iiint x^p y^q z^r f$; the $3\times3$ covariance (inertia) tensor, principal
  axes and the equivalent ellipsoid of a labelled 3D object (nucleus orientation in a
  confocal stack); its eigenvalues as rotation invariants; 3D Zernike moments and spherical
  harmonics named as the 3D analogue of unit 4. Moments of a mesh via the divergence theorem
  (marching-cubes day).
- **Demo**: reuse `zernikebasis`'s WebGL view with `data-mode="ellipsoid"`: a voxel object
  (from `lib/volume.ts` if it exists) with its principal axes and equivalent ellipsoid, rotate
  by dragging.

## Shared engineering

- `demos/src/lib/moments.ts` (new, owned by this day):
  - Raw, central and normalised moments to arbitrary order for a mask or grey image (separable
    Vandermonde products, coordinates mapped to $[-1, 1]$, float64), pixel-as-point and
    pixel-integrated variants; central from raw by the binomial formula; polygon moments by
    Green's theorem.
  - Complex moments, Hu's seven invariants, Flusser's independent set, the log transform;
    pose normalisation (centroid, scale, principal axis, $\mu_{30}$ sign).
  - Legendre polynomials by recurrence (sampled or integrated per pixel), discrete Tchebichef
    as an option; Legendre moments and reconstruction.
  - Zernike radial polynomials by recurrence, $V_{nm}$ on an $N\times N$ grid (unit disc
    inscribed or circumscribed), Zernike moments and reconstruction; monomial reconstruction
    via the Gram system (deliberately ill-conditioned, for the demo).
- **Split with the labelling day**: `lib/label.ts` keeps its one-scan per-label accumulation of
  moments to order 2 (fast, for all labels at once). `lib/moments.ts` works on one object at a
  time and to any order; its order-2 results must agree with `label.ts` (test). The closed-form
  eigen-decomposition of a symmetric $2\times2$ matrix stays in one place (`lib/linalg.ts` or
  `lib/imageops.ts`, as agreed in the labelling and skeleton plans).
- **`moments` demo**: owned by the labelling day; this day adds `data-mode="higher"` and
  `data-mode="invariants"`. The shape canvas with drag/rotate/scale/mirror handles is factored
  out (`lib/shapecanvas.ts` or inside `lib/shapes.ts`) so that `momentphase` and `momentclass`
  reuse it.
- Reused: `lib/shapes.ts` (snakes day: nucleus/kidney and leaf families, rasterisation; add a
  digit/letter stroke family), `lib/plot.ts` (bar plots, error vs $N$, scatter plot), the matrix
  heat map helper proposed in the snakes plan (Gram matrices), `lib/gl3d.ts` (3D surface and
  ellipsoid views), `lib/imageops.ts` (loading `cell256.png`, noise), `lib/ui.ts` (`rng(seed)`).
- **Language / speed**: TypeScript throughout. Estimated costs: Zernike moments up to $N = 20$
  on 128×128 are ~121 sums over ~13 k disc pixels (≈ 2 M complex multiply-adds, a few ms);
  reconstruction is the same again. $N = 40$ on 256×256 is ≈ 30 M operations, around the 30 ms
  limit: recompute only when $N$ or the image changes (not on every slider tick of other
  controls), cache radial polynomial tables per size, and measure before considering Rust/Wasm
  for that one kernel (with a TS reference and an agreement test, per `demos/README.md`).
- Tests (`make test`, as in the other plans): moments of an analytic rectangle and disc match
  the closed forms; central-from-raw = direct central; order-2 results agree with
  `lib/label.ts`; Hu invariants of a large digitised ellipse are constant under rotation within
  a tolerance and $\phi_7$ flips sign under mirroring; complex moments rotate by
  $e^{i(p-q)\alpha}$; the discrete Legendre and Zernike bases are orthogonal up to a documented
  tolerance (exact for Tchebichef); reconstruction of an image that is a sum of a few basis
  functions is exact; Legendre moments from geometric moments = direct; Zernike recurrence =
  factorial formula for $n \le 20$.
- Images: `demos/src/fourier2d/cell256.png` (crops, credit as in `fourier2d`); synthetic
  shapes generated in TS with fixed seeds (no files, no fonts).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only (no hover, no modifier
  keys), keyboard back to reveal.js, labels ≥ 18 px (moment tables: few rows on slides, more
  in the exercise via `data-order`), fits 1150×560 on a slide, `data-*` settings
  (`data-mode`, `data-order`, `data-basis`, `data-image`, `data-preset`), static figures as
  print fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, moments by hand, `moments` (higher) |
| 10:45–12:15 | Unit 2 lecture, invariants by hand, `moments` (invariants), `momentphase` |
| 13:15–14:45 | Unit 3 lecture, Gram–Schmidt by hand, `polybasis`, `momentrecon` |
| 15:00–16:30 | Unit 4 lecture + exercise (`zernikebasis`, `momentrecon`); quizzes as wrap-up |

## Open questions

- How long is a "day" (same question as for the other days)? Units 1 and 2 could merge into
  one if the labelling day's moment material is fresh; then unit 5 (classification) could
  become core, which gives the day an applied ending. Proposal: keep 1–4 core and put a short
  classification task from `momentclass` into the unit 4 exercise.
- Depth of Hu invariants: derive $\phi_1, \phi_2$ and show the complex-moment construction
  (proposal), or also derive $\phi_7$'s mirror behaviour formally?
- Monomial reconstruction in `momentrecon`: keep it (shows the ill-conditioning vividly) or
  show only the Gram matrix condition number in `polybasis`?
- Discrete orthogonal polynomials (Tchebichef, Krawtchouk): one slide (proposal) or a demo
  toggle?
- Tsai's moment-preserving threshold: here as a pointer (proposal) or on the thresholding day?
- Digits/letters for classification: hand-made stroke glyphs (proposal, reproducible) or a
  small bundled licensed dataset (e.g. a few MNIST digits; licence and size to check)? A real
  image with differently shaped nuclei or leaves would also help (same licence search as other
  plans).
- 3D moments: optional unit (proposal) or a single slide in unit 2?
- Boundary-based descriptors (Fourier descriptors of a contour) are not covered by any plan;
  they would fit next to this day as a comparison (region moments vs contour descriptors).
- Programming language for any code parts is still open (see README).

## References

- Hu (1962), *Visual pattern recognition by moment invariants*, IRE Trans. Information Theory.
- Reiss (1991), *The revised fundamental theorem of moment invariants*, PAMI.
- Teague (1980), *Image analysis via the general theory of moments*, J. Optical Society of America.
- Teh & Chin (1988), *On image analysis by the methods of moments*, PAMI.
- Khotanzad & Hong (1990), *Invariant image recognition by Zernike moments*, PAMI.
- Abu-Mostafa & Psaltis (1984), *Recognitive aspects of moment invariants*, PAMI.
- Flusser (2000), *On the independence of rotation moment invariants*, Pattern Recognition.
- Flusser, Suk & Zitová (2016), *2D and 3D Image Analysis by Moments*, Wiley.
- Liao & Pawlak (1996), *On image analysis by moments*, PAMI (discretisation errors).
- Mukundan, Ong & Lee (2001), *Image analysis by Tchebichef moments*, IEEE Trans. Image Processing.
- Prata & Rusch (1989), *Algorithm for computation of Zernike polynomials expansion coefficients*, Applied Optics.
- Zernike (1934), *Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der Phasenkontrastmethode*, Physica.
- Tsai (1985), *Moment-preserving thresholding: a new approach*, Computer Vision, Graphics, and Image Processing.
- Gonzalez & Woods, *Digital Image Processing*, chapter on representation and description (moment invariants).
