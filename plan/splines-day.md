# Plan: a day on splines and NURBS

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.

**Course map (2026-11-13, self-study restructure):** day 6 of 51, Part I "Images as numbers". Hard prerequisites: Images, filters and frequencies (day 1). Soft (see also): Deconvolution (day 5), Labelling (day 14). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VI (the splines row, requested
2026-11-12); **entry after Part I** (needs `images/`: interpolation, homogeneous coordinates;
`kernels/`: convolution; `fourier/`: frequency response; `regularise/`: data + λ·penalty).
Independent of the snakes → level sets → dynamic programming chain, but best taken before
snakes (spline snakes recall this day). Per "Ownership changes" in `plan/coursemap.md`: **this
day owns spline theory**; snakes (spline snakes), non-rigid registration (B-spline free-form
deformation), image interpolation and outline fitting **recall** it.

## Goal of the day

After the day a student can:

1. explain why a **single polynomial through many points wiggles** (Runge), and why a chain of
   **low-degree pieces** does not; name the continuity of a join (**C⁰, C¹, C², G¹**) from a
   picture of a curve and its curvature comb;
2. evaluate a **Bézier curve** by hand with **de Casteljau** (repeated linear interpolation),
   read its properties from the control polygon (end points, end tangents, convex hull, affine
   invariance), and place control points so that two pieces join with C¹ or G¹;
3. describe **B-splines as a basis**: bumps on a **knot vector**, **local control** (moving one
   control point changes only k + 1 spans), **partition of unity**; evaluate a uniform cubic
   B-spline at a knot ($\tfrac16, \tfrac46, \tfrac16$); see nearest neighbour, linear and cubic
   interpolation as B-splines of degree 0, 1, 3;
4. explain what **weights** add (**NURBS**): a rational curve is a B-spline in **homogeneous
   coordinates** followed by a division; verify by hand that a quadratic with weights
   $(1, \tfrac{1}{\sqrt2}, 1)$ gives an **exact quarter circle**, and why no polynomial curve can;
5. tell **interpolating**, **approximating** (least squares, fewer control points) and
   **smoothing** splines (data + λ·bending) apart, predict what λ does, and recognise the
   natural cubic spline as the **minimum-bending** interpolant;
6. fit a closed smoothing spline to a pixel outline and read **length** and **curvature** from
   it, and explain why this beats counting boundary pixels;
7. explain **cubic B-spline image interpolation** (why it needs a **prefilter**, why it is
   sharper than bilinear and smoother than nearest), and describe a **free-form deformation**
   as a tensor-product spline of displacements;
8. *(math)* derive Bernstein polynomials from de Casteljau, the Cox–de Boor recursion, the
   derivative of a B-spline, the natural cubic spline system, the penalised least-squares
   normal equations $(B^\top B + \lambda P)\mathbf{c} = B^\top\mathbf{y}$, the curvature formula
   for a parametric curve, and the B-spline prefilter as a recursive filter.

Why it fits this audience: it is linear algebra in its friendliest form (a curve is a linear
combination of basis functions; fitting is least squares; the interpolation conditions are a
banded linear system) plus calculus (derivatives, curvature, an integral of $f''^2$). It reuses
three ideas of Part I in a new light: interpolation (`images/`), convolution (`kernels/`: the
cubic B-spline is four boxes convolved) and regularisation (`regularise/`: smoothing splines
are data + λ·penalty). No image analysis beyond Part I is needed.

## Prerequisite gap

What the built units already teach (checked), and what this day recalls in one line:

- **`images/`** (day 1, built): inverse mapping, nearest and **bilinear** interpolation (lecture
  slide + math slide, exercise B4/B8), and **homogeneous coordinates** as $3\times3$ matrices
  with the extra coordinate 1. Recalled: unit 1 (piecewise linear = bilinear in 1D; de
  Casteljau is "lerp of lerps", like bilinear), unit 2 (homogeneous coordinates with a $w$ that
  is *not* 1, divide at the end), unit 4 (beyond bilinear). No projective transforms are taught
  in `images/`; the division by $w$ is introduced here from scratch (one slide), with a pointer
  to `pipeline3d/` (3D graphics day: "the same division as the perspective divide").
- **`kernels/`** (day 1): convolution of kernels combines them. Unit 2: box * box = triangle,
  * box * box = cubic B-spline (picture, no derivation needed).
- **`fourier/`** (day 1): low-pass, aliasing, the frequency response of a kernel. Unit 4
  compares the frequency responses of the nearest/linear/cubic interpolators (one picture).
- **`regularise/`** (day 5, built): "data + λ·penalty", choosing λ (discrepancy, L-curve),
  MAP reading. Unit 3 recalls it in one slide; the smoothing spline is a new instance, not a
  re-derivation.
- **Least squares / normal equations**: linear algebra (audience.md); recalled in one line.
- **Curvature of a plane curve** may be rusty: unit 1 gives it as "1/radius of the best
  touching circle" (concept) and the formula in the math layer.
- Not needed but nice: `shape-features/` (labelling day, Part II) for the perimeter bias of
  pixel counting; unit 3 states the bias in one slide with numbers, so a student who skipped
  Part II is not lost.

## Overlaps / who owns what

This day **owns**: polynomial interpolation and Runge's phenomenon, piecewise cubics and
continuity, Bézier curves and de Casteljau, B-spline bases (knots, Cox–de Boor, local control,
partition of unity), NURBS, interpolating/approximating/smoothing splines, tensor-product
spline surfaces, cubic B-spline image interpolation (prefilter), the free-form deformation
*representation*. Others recall, they do not re-derive.

| Day / unit | What stays there | What this day does |
|---|---|---|
| `images/` (day 1) | nearest, bilinear, inverse mapping, affine homogeneous coordinates | unit 4 goes beyond bilinear; **pending edit** to `images/lec.qmd` (interpolation slide): one line "cubic and B-spline interpolation: splines day" |
| `regularise/` (day 5), owner of continuous regularisation | data + λ·penalty, choosing λ | unit 3 recalls it; the smoothing spline is the 1D curve-fitting instance; λ is the weight of the penalty (convention) |
| `gmrf/` (Ising day) | quadratic first-difference smoothing on a grid, TV | unit 3 one line: "a second-difference penalty is the discrete bending energy; first differences (`gmrf/`) penalise slope, not bending" |
| `orthogonal-moments/` (moments day) | ill-conditioning of monomial bases (Gram matrix) | unit 1 pointer: "the other problem of high-degree polynomials: rounding (moments day)"; Runge here is about the interpolant itself, even in exact arithmetic |
| `shape-features/` (labelling day) | perimeter estimators and their bias, form factor, **Fourier descriptors** (a smooth outline from low frequencies) | unit 3 recalls the bias; compares a smoothing spline (local) with truncated Fourier descriptors (global) in one slide; **pending edit**: pointer in its "Contour descriptors" slide |
| `snakes/`, `snake-matrix/` (snakes day) | the snake energy, the pentadiagonal matrix, forces | **spline snakes** stay a pointer in unit 4 (a snake whose unknowns are a few B-spline control points; smoothness built in). **Pending edit**: one bullet in `snakes/lec.qmd` "Not covered here, but close" |
| Optical flow day (`plan/opticalflow-day.md`, being planned), unit 4 `nonrigid-reg/` | the registration energy (similarity incl. mutual information + λ·bending of the deformation), its optimisation, multiresolution, the `ffdreg` demo and `lib/deform.ts` (`ffdDisplacement`, `jacobianDeterminant`, `registerFFD`) | unit 4 teaches the **B-spline FFD representation** (control grid, displacement field, locality, partition of unity, folding) with the hand-driven `ffd` demo; `nonrigid-reg/` keeps its short FFD recall slide and points here for B-splines in depth. `bspline3` moves from `lib/deform.ts` into `lib/spline.ts` (`deform.ts` re-exports it, as that plan already proposes). Thin-plate splines (pointed to from `nonrigid-reg/`): optional unit 6 here |
| `pipeline3d/` (3D graphics day) | $4\times4$ homogeneous matrices, the perspective divide | unit 2 points to it for "dividing by $w$"; NURBS surfaces (CAD) and subdivision surfaces get a one-line pointer from the optional unit |
| `scale-space/` (edges day) | Gaussian pyramid | unit 4: one line: "B-spline pyramids exist (Unser); the Gaussian one is the course's" |
| Dynamic programming, level sets | curves as optimal paths / zero levels | no overlap beyond a summary line ("three ways to represent a curve: parametric spline, implicit φ, pixel path") |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus two optional units. Order:
the problem and the simplest piece (unit 1), the basis (unit 2), fitting (unit 3), images
(unit 4). Units 1–2 are curve geometry; units 3–4 are where the image analysis is.

### 1. `bezier/`: Polynomial wiggles, piecewise cubics and Bézier curves

Concept layer:

- The problem: we have points (clicks on a cell outline, samples of a profile, pixel centres)
  and want a smooth curve. Three variants: through the points, near the points, controlled by
  the points (picture of each; the rest of the day returns to them).
- **One polynomial through all points**: $n$ points fix a polynomial of degree $n - 1$. For 3–5
  points fine; for 11 equally spaced samples of a smooth bump the curve swings wildly near the
  ends (**Runge**). By hand with numbers: the degree-4 polynomial through
  $(-2,0), (-1,0), (0,1), (1,0), (2,0)$ is $(x^2-1)(x^2-4)/4$ and is $-0.55$ at $x = 1.5$,
  where every sample is 0.
- Fixes that do not work: more points (worse). A fix that works but is not ours: bunch the
  points at the ends (Chebyshev nodes; one line). Our fix: **piecewise low degree**.
- **Piecewise linear** = 1D bilinear (recall `images/`): continuous, but kinks at every point.
- **Piecewise cubic**: each piece has 4 numbers; spend them on matching values *and* slopes at
  the ends (cubic Hermite). Continuity vocabulary with pictures: **C⁰** (no gap), **C¹** (same
  tangent vector), **G¹** (same direction, different speed), **C²** (same curvature). Curvature
  as "1/radius of the touching circle", shown as a **curvature comb**: a jump in the comb is
  what the eye sees as a "bump" in a C¹ join.
- **Parametric curves** $(x(t), y(t))$: outlines fold back and close, so $y = f(x)$ is not
  enough. Everything per coordinate.
- **Bézier curve**: four control points; the curve starts at the first, ends at the last,
  leaves along the first leg and arrives along the last; the middle points pull but are not
  passed through.
- **De Casteljau**: lerp along each leg of the control polygon at fraction $t$, then lerp the
  new points, until one point is left. Worked with numbers ($P = (0,0), (1,2), (3,2), (4,0)$,
  $t = \tfrac12$: $(0.5, 1), (2, 2), (3.5, 1)$ → $(1.25, 1.5), (2.75, 1.5)$ → $(2, 1.5)$). It
  is "linear interpolation of linear interpolations", the same trick as bilinear.
- Properties read off the control polygon: end points and end tangents, **convex hull** (the
  curve never leaves it), **affine invariance** (transform 4 control points instead of 1000
  curve points), **subdivision** (de Casteljau at $t$ splits the curve into two Bézier curves:
  the construction's outer points).
- **Joining Bézier pieces**: C⁰ shares the end point; C¹ puts the next piece's first inner
  point at the mirror image of the last inner point; G¹ anywhere on that ray. This is how
  vector drawing programs and fonts store outlines (one-line pointer, no product names).
- The limitation that leads to unit 2: each piece is controlled separately, the continuity has
  to be maintained by hand, and a single high-degree Bézier is global (every control point
  moves the whole curve) → we want a basis with smoothness and locality built in.

Math layer (`## The math: … {.math}`):

- *Interpolation polynomial*: Vandermonde system or Lagrange basis $L_i(x) = \prod_{j\ne i}
  \frac{x - x_j}{x_i - x_j}$; error $f(x) - p(x) = \frac{f^{(n)}(\xi)}{n!}\prod_i (x - x_i)$;
  the product is huge near the ends for equispaced nodes (Runge), minimised by Chebyshev nodes.
- *Bernstein form*: de Casteljau unrolled gives $\mathbf{B}(t) = \sum_i \binom{n}{i}t^i(1-t)^{n-i}
  \mathbf{P}_i$; partition of unity from the binomial theorem; non-negativity ⇒ convex hull.
- *Derivative*: $\mathbf{B}'(t) = n\sum_i b_{i,n-1}(t)(\mathbf{P}_{i+1} - \mathbf{P}_i)$, so
  $\mathbf{B}'(0) = n(\mathbf{P}_1 - \mathbf{P}_0)$; the C¹ join condition follows.
- *Curvature* of a parametric curve: $\kappa = \frac{x'y'' - y'x''}{(x'^2 + y'^2)^{3/2}}$; for
  the worked curve at $t = 0$: $\mathbf{B}' = (3, 6)$, $\mathbf{B}'' = 6(\mathbf{P}_2 -
  2\mathbf{P}_1 + \mathbf{P}_0) = (6, -12)$, $\kappa = -72/45^{3/2} \approx -0.24$.
- *Cubic Hermite ↔ Bézier*: $\mathbf{P}_1 = \mathbf{P}_0 + \mathbf{T}_0/3$,
  $\mathbf{P}_2 = \mathbf{P}_3 - \mathbf{T}_3/3$.

Demos:

- **`runge`** (new, small): 1D. Samples of a function (bump $1/(1 + 25x^2)$, step, a noisy
  profile across a cell from `lib/cell.ts`) at $n$ points (slider 3–21), equispaced or
  Chebyshev; overlay: the interpolating polynomial, piecewise linear, natural cubic spline
  (from unit 3, labelled "coming in unit 3"). Points draggable vertically. Readout: max error
  against the true function. Presets: "5 points are fine", "11 points: Runge", "Chebyshev
  fixes the ends", "the step: everything rings, the spline least", "one point moved: the
  polynomial changes everywhere".
- **`casteljau`** (new, main demo): control polygon of a cubic (drag 4 points; optional degree
  2–6), slider $t$ showing the de Casteljau construction (levels coloured), the Bernstein basis
  plot below with a vertical line at $t$ and the weights as numbers (≥ 18 px), toggles convex
  hull / curvature comb / tangent at $t$. A second piece can be attached with a join mode
  menu: free, C⁰, G¹, C¹ (the constrained handle follows), with the comb showing the jump.
  Button "transform": rotate/scale the control points and show that the curve follows exactly.
  Presets: "the worked example", "S-curve", "loop", "cusp (inner points crossed)", "C⁰ join:
  a kink", "G¹ but not C¹: the comb jumps", "C¹ join".

By hand (concept): (1) evaluate the degree-4 polynomial above at $x = 1.5$ and compare with
piecewise linear; (2) de Casteljau at $t = \tfrac12$ and $t = \tfrac14$ for the worked control
points, and read the two halves' control points from the construction; (3) where must the next
piece's second control point go for C¹ after $\mathbf{P}_3 = (4, 0)$? (answer $(5, -2)$); for
G¹? (4) which control-point picture gives a loop, which a cusp?
*(math)*: derive the Bernstein form from de Casteljau for $n = 2$; prove partition of unity;
derive $\mathbf{B}'(0)$ and the C¹ condition; compute $\kappa(0)$ for the worked curve; show
the Runge error product is largest near the ends for 5 equispaced nodes.

### 2. `bsplines/`: B-splines and NURBS

Concept layer:

- From unit 1: we want smoothness and locality for free. Idea: write the curve as
  $\sum_i \mathbf{P}_i\,N_i(t)$ with **bump-shaped basis functions** that overlap only with their
  neighbours; the control points are the coefficients (recall: basis and coordinates, as the
  DFT in `fourier/`).
- **B-splines by convolution** (concept route): degree 0 = box (nearest neighbour), degree 1 =
  box * box = hat (linear interpolation), degree 3 = four boxes = a smooth bell (recall
  `kernels/`: convolution combines kernels). Each convolution adds one order of smoothness.
  Uniform cubic B-spline values at the knots: $\tfrac16, \tfrac46, \tfrac16$.
- **Knot vector**: where the pieces join. Uniform knots = shifted copies of one bump. Every
  point of a uniform cubic B-spline curve is a weighted average of 4 control points; at a knot
  $(\mathbf{P}_{i-1} + 4\mathbf{P}_i + \mathbf{P}_{i+1})/6$: the curve **cuts the corners**
  (by hand on a square).
- **Partition of unity** and non-negativity: weights sum to 1 ⇒ convex hull of 4 neighbouring
  points; translation invariance.
- **Local control**: moving one control point changes only 4 spans of a cubic (picture: the
  changed part highlighted), against the global effect in a single Bézier.
- **Knot multiplicity**: repeating a knot reduces the continuity there (triple knot in a cubic:
  a corner, C⁰); **clamped** knots (end knots repeated) make the curve start and end at the end
  control points, like Bézier; **closed (periodic)** B-splines for outlines. A clamped cubic
  with 4 control points is exactly the cubic Bézier.
- **Cox–de Boor** as a picture: a degree-$p$ basis function is a blend of two degree-$(p - 1)$
  ones, ramps times bumps (a tower diagram, degree 0 → 3).
- **What polynomials cannot do**: no polynomial curve is an exact circle (the best cubic
  Bézier quarter is off by 0.03 %; fine for drawing, wrong for "is this a circle?" or for CAD).
- **NURBS: weights**. Give each control point a weight $w_i$: a larger weight pulls the curve
  towards that point. Mechanism: **lift** each control point to $(w x, w y, w)$, run the
  ordinary B-spline in 3D, **divide** by the last coordinate. Homogeneous coordinates as in
  `images/`, but now $w \ne 1$ and the division matters (pointer: the same division is the
  perspective divide in `pipeline3d/`).
- **Exact quarter circle** by hand: control points $(1,0), (1,1), (0,1)$, weights
  $1, \tfrac{1}{\sqrt2}, 1$; at $t = \tfrac12$ the point is $(0.7071, 0.7071)$, radius 1. With
  all weights 1 (plain quadratic Bézier) it is $(0.75, 0.75)$, radius 1.061. Full circle: 9
  control points or 7, standard pictures. Other conics: weight $< 1$ ellipse arc, $= 1$
  parabola, $> 1$ hyperbola.
- What NURBS are used for (one slide): exact geometry in engineering design and 3D modelling,
  invariance under perspective projection; in image analysis rarely needed (most data are
  noisy samples, unit 3), but the homogeneous trick is worth knowing.
- Spline surfaces, briefly: **tensor product** $\sum_{ij}\mathbf{P}_{ij}N_i(u)N_j(v)$, a grid of
  control points; one picture of a bicubic patch (rendered with `lib/gl3d.ts` as a static
  figure). Unit 4 uses exactly this with an image as the control grid.

Math layer:

- *Cox–de Boor*: $N_{i,0}(t) = [t_i \le t < t_{i+1}]$,
  $N_{i,p}(t) = \frac{t - t_i}{t_{i+p} - t_i}N_{i,p-1}(t) + \frac{t_{i+p+1} - t}{t_{i+p+1} -
  t_{i+1}}N_{i+1,p-1}(t)$ (0/0 := 0); support $[t_i, t_{i+p+1})$; partition of unity by
  induction.
- *Uniform B-splines by convolution*: $\beta^p = \beta^0 * \beta^{p-1}$; closed form of the
  cubic $\beta^3(x) = \tfrac23 - x^2 + \tfrac{|x|^3}{2}$ for $|x| \le 1$,
  $\tfrac{(2 - |x|)^3}{6}$ for $1 \le |x| \le 2$; Fourier transform $\operatorname{sinc}^{p+1}$
  (forward reference to unit 4).
- *Continuity*: $C^{p-m}$ at a knot of multiplicity $m$; derivative of a B-spline curve is a
  B-spline of degree $p - 1$ with control points $p\frac{\mathbf{P}_{i+1} - \mathbf{P}_i}{t_{i+p+1}
  - t_{i+1}}$.
- *De Boor's algorithm*: de Casteljau with knot-dependent fractions (stated, one step shown).
- *NURBS*: $\mathbf{C}(t) = \frac{\sum_i w_i\mathbf{P}_iN_i(t)}{\sum_i w_iN_i(t)}$ = projection of
  a polynomial B-spline in homogeneous coordinates; verification of the quarter circle
  ($x^2 + y^2 = 1$ for all $t$ in a few lines); projective invariance (apply a $3\times3$
  homogeneous matrix to the lifted control points).

Demos:

- **`bsplinebasis`** (new): the basis functions $N_{i,p}$ on a knot line; degree 0–3 buttons;
  knots draggable (and doubled/tripled by dropping onto each other); the sum drawn as a dashed
  line (always 1 inside the valid range); step-through of the Cox–de Boor tower for one basis
  function; toggle "as repeated convolution of boxes" (uniform knots). Presets: "uniform
  cubic", "degree 0 → 3", "clamped ends", "double knot: kink in the slope", "triple knot:
  corner", "uneven knots".
- **`bsplinecurve`** (new, main demo): one control polygon (drag, add, delete points), drawn
  as B-spline (degree 1–5, open/clamped/closed) and as a single Bézier through the same points
  side by side; dragging a point highlights the changed spans in both; per-point **weight**
  slider when a point is selected (NURBS); toggles curvature comb, knot markers, convex hulls
  of each span, homogeneous view (the lifted control points and the cone of division, a small
  3D view via `lib/gl3d.ts`, optional, 2D fallback). Presets: "local vs global", "corner
  cutting on a square", "closed outline", "triple knot corner", "quarter circle: weight 1/√2",
  "weight 1: not a circle" (readout: max radius error), "full circle from 9 points", "conics:
  weight < 1, = 1, > 1".

By hand (concept): (1) corner cutting: closed uniform cubic on the square
$(0,0), (4,0), (4,4), (0,4)$: the curve point at the knot of $(4,0)$ is $(10/3, 2/3)$; is it
inside the square? (2) which spans change when $\mathbf{P}_5$ of a cubic with 10 control points
moves? (3) the quarter circle at $t = \tfrac12$ with weights $(1, 1/\sqrt2, 1)$ and with all
weights 1: compute both radii; (4) nearest, linear and cubic interpolation as B-splines: which
degree is which, and which one is not interpolating? (prepares unit 4).
*(math)*: Cox–de Boor by hand on uniform knots $0, 1, 2, 3$: $N_{0,2}(1) = 0.5$,
$N_{0,2}(1.5) = 0.75$; prove partition of unity for degree 1 and 2; derive $\beta^1 = \beta^0 *
\beta^0$; show the rational quarter circle satisfies $x^2 + y^2 = 1$; show a clamped cubic
with 4 control points equals the Bézier curve.

### 3. `spline-fit/`: Interpolating, approximating and smoothing splines

Concept layer:

- The three tasks again, now with data: **interpolate** (through every point: clean data, few
  points), **approximate** (fewer control points than data, least squares), **smooth** (one
  control point per datum, but penalise bending). Picture of the same noisy data fitted three
  ways.
- **Interpolating spline**: choose control points so the curve passes through the data; one
  equation per point, each involving 3 neighbours: a **banded linear system** (tridiagonal),
  solved in $O(n)$. The **natural cubic spline** (zero curvature at the ends).
- **Minimum bending**: among all smooth curves through the points, the natural cubic spline
  has the least $\int f''^2$: the draftsman's thin wooden strip held by weights (the origin of
  the word, picture). By hand: through $(0,0), (1,1), (2,0)$ the spline is $1.5x - 0.5x^3$ on
  $[0,1]$ (bending 6) while the parabola $2x - x^2$ has bending 8.
- **Approximating (regression) spline**: $m < n$ control points, least squares; $m$ controls
  the smoothness in steps; knots placement matters (too few: underfit; too many: wiggles from
  noise). Recall least squares in one line.
- **Smoothing spline**: minimise **data misfit + λ·bending**. Recall `regularise/` (day 5): the
  same pattern, λ the weight of the penalty. λ = 0: interpolation (noise followed); λ → ∞: the
  least-squares straight line (bending 0 costs nothing); in between a smooth curve. Discrete
  version with numbers: three data $(0, 3, 0)$, penalty λ·(second difference)²: λ = 1 gives
  $(0.857, 1.286, 0.857)$, λ → ∞ gives $(1, 1, 1)$; the mean is kept.
- **Choosing λ**: by eye; by the noise level (discrepancy, recall); by **leave-one-out
  cross-validation** (hide each point, predict it, average the error): a U-shaped curve with a
  minimum. The effective number of parameters goes from $n$ (λ = 0) to 2 (λ → ∞).
- A smoothing spline is a **low-pass filter** in disguise: on uniform data its effect is a
  convolution with a fixed bell-shaped kernel whose width grows with $λ^{1/4}$ (picture;
  recall `fourier/`).
- **Image use: smooth outlines.** Boundary pixels of a segmented object (staircase), fitted
  with a **closed** smoothing spline in $x(t), y(t)$. From the fitted curve, **length** is an
  integral of the speed and **curvature** is a formula of the derivatives: both exact for the
  curve, no pixel counting. Numbers: a digital disc of radius 10: edge count 80, true
  perimeter 62.8, smoothing spline within about 1 % (check in the build). Curvature along the
  outline as a colour/comb: bumps and necks of a cell appear as curvature extrema.
- Smoothing spline (local) vs truncated Fourier descriptors (global, `shape-features/`): a
  sharp corner rings in Fourier, stays local in the spline (one slide, side by side).
- Where this leads (one line each): spline snakes (unit 4 pointer, snakes day), kymograph
  edges and growth curves (smooth then differentiate: velocity and acceleration of a boundary).

Math layer:

- *Natural cubic spline system*: with second derivatives $M_i$ at uniform nodes $h$:
  $M_{i-1} + 4M_i + M_{i+1} = \tfrac{6}{h^2}(y_{i-1} - 2y_i + y_{i+1})$, $M_0 = M_n = 0$; the
  worked example gives $M_1 = -3$. The B-spline-coefficient form $(c_{i-1} + 4c_i + c_{i+1})/6 =
  y_i$ is the same tridiagonal shape (used in unit 4).
- *Minimum bending* (Holladay): for any $g$ interpolating the data,
  $\int g''^2 = \int s''^2 + \int (g'' - s'')^2$ (integrate by parts, natural boundary
  conditions kill the boundary terms).
- *Penalised least squares*: $f = B\mathbf{c}$, minimise $\|\mathbf{y} - B\mathbf{c}\|^2 +
  \lambda\mathbf{c}^\top P\mathbf{c}$ ⇒ $(B^\top B + \lambda P)\mathbf{c} = B^\top\mathbf{y}$, with
  $P$ the bending Gram matrix or the discrete second-difference penalty $D^\top D$ (P-splines);
  banded, solved in $O(n)$. The hat matrix $H(\lambda)$, effective degrees of freedom
  $\operatorname{tr}H$, leave-one-out shortcut $\frac{1}{n}\sum\big(\frac{y_i - \hat f_i}{1 -
  H_{ii}}\big)^2$ (and GCV).
- *Equivalent kernel*: on a uniform infinite grid the discrete smoother has frequency response
  $1/(1 + \lambda(2 - 2\cos\omega)^2)$ (compare the first-difference version in `gmrf/`).
- *Arc length and curvature*: $L = \int\|\mathbf{C}'(t)\|\,dt$ (Gauss–Legendre per span);
  curvature formula from unit 1; for a closed curve $\oint\kappa\,ds = 2\pi$ (a check the demo
  displays).

Demos:

- **`splinefit`** (new, main demo): 1D noisy samples of a known curve (seeded noise slider;
  data from a profile across the cell image or synthetic: bump, step, sine with a kink). Mode:
  interpolate / approximate ($m$ control points, slider; knots shown) / smooth (λ on a log
  slider). Shows the fit, the true curve (toggle), residuals as sticks, and a second panel
  with the leave-one-out error and the effective degrees of freedom against λ (or $m$), with a
  marker at the current value and at the minimum. Data points draggable; "add outlier"
  button. Presets: "λ = 0 interpolates the noise", "λ huge: a straight line", "the CV
  minimum", "too few control points", "too many control points", "an outlier", "a step:
  every smoother rounds it" (forward pointer: TV in `gmrf/` keeps steps).
- **`splineoutline`** (new): a synthetic object (seeded shapes from `lib/shapes.ts` or
  `lib/shapecanvas.ts` presets: disc, ellipse, kidney, star, cell with a neck; plus the cell
  micrograph thresholded) rasterised to pixels; its boundary pixels; a closed smoothing spline
  (λ slider, number of control points). Readouts (≥ 18 px): length by edge count, by chain
  code, by the spline, and the true length (synthetic shapes); total curvature $\oint\kappa\,ds$.
  Curvature shown as a comb or colour along the curve; a curvature-vs-arc-length plot below.
  Toggle: truncated Fourier descriptors with $K$ terms on the same outline for comparison.
  Presets: "disc: three perimeters", "small disc (radius 3)", "star: corners vs λ", "neck
  between two cells: the curvature minimum", "Fourier rings at a corner", "too little
  smoothing: pixel staircase curvature".

By hand (concept): (1) three-point discrete smoothing with λ = 0, 1, ∞ (numbers above); what
happens to the mean? (2) the natural spline through $(0,0), (1,1), (2,0)$ at $x = 0.5$ (0.6875)
vs the parabola (0.75): which bends less? (3) predict: doubling the noise, should the best λ go
up or down? (4) perimeter of a 10-pixel-radius digital disc by edge count vs spline; why the
edge count is 4/π too long for every disc. (5) Fourier descriptors vs spline at a corner:
which result changes far from the corner when you smooth more?
*(math)*: set up and solve the natural spline system for the 3-point example; derive the
penalised normal equations; prove Holladay's identity for the given example numerically
($6 \le 8$) and in general; derive the frequency response of the second-difference smoother;
show $\oint\kappa\,ds = 2\pi$ for a circle of radius $r$.

### 4. `spline-images/`: Splines on images: interpolation and free-form deformation

Concept layer:

- An image is samples of a surface; interpolating it is choosing a surface through the
  samples. Recall nearest and bilinear (`images/`): bilinear is the tensor-product spline of
  degree 1 (unit 2). Its faults: kinks at every pixel (derivatives jump), blur on rotation.
- **Two cubic ideas**: (a) **cubic convolution** (a fixed 4×4 kernel with negative lobes,
  interpolating, what most "bicubic" options do; one-line pointer: "image libraries call this
  bicubic"); (b) **cubic B-spline**: the smooth bump of unit 2, but it is *not* interpolating
  ($\tfrac16, \tfrac46, \tfrac16$ blurs), so first compute **coefficients** such that the
  spline passes through the samples: the unit 3 interpolation system, here applied along rows
  and columns. This **prefilter** is a cheap sharpening filter (two passes per row); after it,
  the cubic B-spline interpolates exactly and is C² (smooth derivatives for free).
- Picture/numbers: an impulse 0, 0, 1, 0, 0: B-spline without prefilter peaks at 0.67; with
  prefilter it passes through 1 and has small negative lobes; bilinear gives 0.5 halfway,
  cubic B-spline about 0.60.
- **Quality in frequency** (recall `fourier/`): every interpolator is a low-pass kernel;
  nearest lets aliases through, bilinear blurs high frequencies, cubic B-spline is closest to
  ideal; the ideal (sinc) is infinite and rings. One plot of the four frequency responses.
- **The repeated-rotation test**: rotate by 36° ten times: nearest shreds, bilinear blurs
  visibly, cubic B-spline barely changes (demo). Also: sub-pixel shifts (registration recalls
  this), smooth zoom, and exact derivatives of the interpolated image (gradients at sub-pixel
  positions, e.g. for snakes and registration).
- **Labels and masks still need nearest** (recall `images/` quiz Q6); cubic interpolation
  overshoots (values below 0 or above 255 near edges): clip or not, say which.
- **Free-form deformation (FFD)**: a coarse grid of control points (every 16–32 pixels), each
  with a displacement vector; the displacement of every pixel is the tensor-product cubic
  B-spline of those vectors; warp the image with it (inverse mapping, recall). Properties from
  unit 2: smooth (C²), **local** (one control point moves a 4×4-cell neighbourhood), all
  displacements equal = a translation (partition of unity). Numbers: drag one control point by
  9 px: the pixel at that control point moves 4 px, a pixel at the next control point 1 px,
  diagonally 0.25 px. Too large displacements **fold** the grid (the map stops being one-to-one;
  the determinant of the Jacobian becomes negative): a safe bound is about 0.4–0.5 of the grid
  spacing (check the 2D constant in the build).
- **Pointer: non-rigid registration** (optical flow day, unit `nonrigid-reg/`, demo `ffdreg`):
  registration = find the control-point displacements that make two images agree, with a
  smoothness penalty; coarse-to-fine by refining the control grid. Representation here,
  optimisation there.
- **Pointer: spline snakes** (snakes day): a snake whose unknowns are a few B-spline
  control points instead of 100 beads: smoothness is built in (elasticity/rigidity terms can be
  dropped or reduced), fewer unknowns, curvature and normals exact. One picture.
- Spline surfaces in 3D (one line): the same tensor product in 3D gives smooth volume
  interpolation and 3D FFD; NURBS surfaces are how engineering shapes are stored (pointer to
  the 3D graphics day: meshes are the other representation).
- Summary of the day: one table: degree / interpolating? / continuity / local? / what it is
  used for (nearest, bilinear, cubic convolution, cubic B-spline, smoothing spline, NURBS).

Math layer:

- *Tensor product*: $f(x, y) = \sum_{k,l}c_{kl}\beta^3(x - k)\beta^3(y - l)$; separable, 16
  terms per pixel.
- *Prefilter*: $(c_{k-1} + 4c_k + c_{k+1})/6 = f_k$ is a convolution with $\tfrac16[1\;4\;1]$;
  its inverse factors into a causal and an anti-causal first-order recursive filter with pole
  $z_1 = \sqrt3 - 2 \approx -0.268$ (Unser, Aldroubi & Eden 1991); the impulse's coefficients
  are $\sqrt3\,z_1^{|k|}$; mirror boundary conditions.
- *Frequency responses*: nearest $\operatorname{sinc}$, linear $\operatorname{sinc}^2$, cubic
  B-spline interpolator (cardinal spline) $\frac{\operatorname{sinc}^4(\omega/2\pi)}{(2 +
  \cos\omega)/3}$; the cardinal spline tends to the sinc as the degree grows.
- *Cubic convolution kernel* (Keys, $a = -\tfrac12$) stated piecewise; it reproduces quadratics,
  is only C¹.
- *FFD*: $\mathbf{u}(\mathbf{x}) = \sum_{kl}\mathbf{d}_{kl}\beta^3(x/h - k)\beta^3(y/h - l)$;
  the worked values $\beta^3(0) = \tfrac23$, $\beta^3(1) = \tfrac16$ give $\tfrac49, \tfrac19,
  \tfrac1{36}$ of the displacement; Jacobian $\det(I + \nabla\mathbf{u}) > 0$ as the no-folding
  condition; the displacement field is linear in the $\mathbf{d}_{kl}$ (so its gradient with
  respect to them, needed by registration, is just the basis values).

Demos:

- **`splinezoom`** (new, main demo): the cell micrograph (`lib/cell.ts`, credit shown) and
  synthetic test patterns (`lib/synth.ts`: zone plate, Siemens star, a step edge, an
  impulse). Interpolator menu: nearest / bilinear / cubic convolution / cubic B-spline
  (prefilter on/off). Operations: zoom into a 16×16 patch ×8 (surface view toggle: the patch
  as a height field via `lib/gl3d.ts`), sub-pixel shift, **rotate by 36° n times** (n slider,
  1–20) with the difference to the original and its RMS error. A 1D panel shows a profile
  through the result with the samples. Side plot: the four frequency responses (`lib/plot.ts`).
  Presets: "impulse: who interpolates?", "prefilter off: blur", "rotate 10×: nearest vs
  bilinear vs B-spline", "zone plate: aliasing", "edge: overshoot of the cubics", "label image:
  why nearest".
  Cost: prefilter of 256² is O(n), ~1–2 ms; a 256² warp with 16 taps per pixel ~10–20 ms; ten
  rotations recomputed only on release of the slider. Measure; Wasm only if > 30 ms live.
- **`ffd`** (new): an image (cell micrograph or a checkerboard from `lib/synth.ts`) with a
  control grid overlay (spacing slider 16/32/64 px); drag control points; the warped image,
  the displacement field as arrows, and the Jacobian determinant as a diverging colour map
  (`lib/colour.ts` `diverging`), negative regions (folds) marked. Toggles: degree 1 vs cubic
  (bilinear FFD shows kinks along grid lines), "move all by the same vector" (pure
  translation). Presets: "one point, 9 px" (readouts at the three neighbours: 4, 1, 0.25 px),
  "a smooth bend", "a local bulge", "too far: the grid folds", "translation from partition of
  unity", "degree 1 vs 3".

By hand (concept): (1) the impulse 0, 0, 1, 0, 0 through the uniform cubic B-spline without
prefilter: the values at the samples ($\tfrac16, \tfrac46, \tfrac16$); why does this blur? (2)
bilinear vs cubic halfway between two pixels of a step 0, 0, 100, 100: which value can be
negative or above 100? (3) FFD: one control point moves by $(9, 0)$; displacement at the
control point, one grid spacing away, diagonally; (4) why the FFD of a label image must still
use nearest lookup in the warp; (5) when would you prefer cubic convolution (no prefilter, local
4×4) over the B-spline (prefilter over the whole row)?
*(math)*: show that the impulse coefficients $\sqrt3\,z_1^{|k|}$ satisfy the prefilter equation;
derive the frequency response of $\tfrac16[1\;4\;1]$ and why its inverse exists; compute the
Jacobian of a 1D FFD $x + d\,\beta^3(x/h)$ and the largest $d$ without folding; show the FFD
displacement is linear in the control displacements.

### 5. (optional) `subdivision/`: Corner cutting, knot insertion and subdivision

- **Chaikin's corner cutting**: replace each edge by points at ¼ and ¾; repeat: the limit is
  the quadratic B-spline (by hand twice on a square). The cubic B-spline as the limit of a
  different rule (½, ⅛-¾-⅛). Knot insertion (Boehm) as the algorithm underneath: add control
  points without changing the curve; render a spline by subdividing until the polygon is
  flat.
- Interpolating subdivision (four-point scheme) vs approximating; pointer: subdivision
  surfaces in 3D (Catmull–Clark, Loop) and the 3D graphics day's meshes.
- **Demo `chaikin`** (new, small): a control polygon, buttons "cut once" / "reset", rule menu
  (Chaikin, cubic B-spline, four-point), overlay of the limit curve from `lib/spline.ts`.
- By hand: two Chaikin steps on a square; show the points after one step lie on the
  quadratic B-spline's control polygon after knot insertion.

### 6. (optional) `thin-plate/`: Thin-plate splines and landmark warps

- The 2D analogue of the natural cubic spline: the surface (or warp) through given values at
  scattered points with the least **bending energy** $\iint f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2$; a
  thin metal plate pinned at the landmarks. Solution: an affine part plus radial bumps
  $U(r) = r^2\log r$ around each landmark; a $(n + 3)\times(n + 3)$ linear system.
- Landmark-based warping (pairs of corresponding points → a smooth warp), the contrast with
  the grid-based FFD of unit 4 (global vs local support, scattered vs grid control points), and
  the smoothing version (data + λ·bending, as in unit 3). Pointed to from `nonrigid-reg/`
  (optical flow day) and `shape-models/` (snakes day: Procrustes is the rigid part).
- **Demo `tpswarp`** (new, small): the cell image with draggable landmark pairs; the warped
  image and the deformed grid; λ slider (exact vs smoothed); presets "affine only (3 points)",
  "one landmark moved", "crossing landmarks: folding".
- By hand: with 3 landmarks the warp is affine (the radial part vanishes); *(math)* the
  bending energy of $r^2\log r$ and the side conditions on the radial weights.

## Shared engineering

**Reuse** (do not copy):

- `lib/ui.ts` (`slider`, `button`, `presets`, `checkbox`, `draggable`, `svgPoint`, `rng`,
  `autoMount`); `lib/plot.ts` (`Plot` with `line`, `fn`, `scatter`, `band`, `vline`, `text`,
  `legend`; log-x for λ; clip ids are fixed per build-issues); `lib/colour.ts` (`diverging`,
  `viridis`, `seriesColor` for curvature colour); `lib/imageops.ts` (`Grey`, `warp`,
  `sampleNearest`, `sampleBilinear`, `rotate`, `resize`, `borderIndex` with mirror,
  `toImageData`/`drawGrey`); `lib/cell.ts` (`loadCell`, `CELL_CREDIT`); `lib/synth.ts`
  (`zonePlate`, `siemensStar`, `checkerboard`, `stepEdge`, `addGaussianNoise`);
  `lib/linalg.ts` (`solve`, `cholesky`/`choleskySolve`, `lstsq` for the least-squares fits;
  `inv3` for homogeneous checks); `lib/smooth.ts` (`thomas`: tridiagonal solve for the natural
  spline and the prefilter check); `lib/shapes.ts` (seeded shape families) and
  `lib/shapecanvas.ts` (`rasterise`, `ellipsePoly`, `starPoly`, `polygonArea`) for
  `splineoutline`; `lib/label.ts` + a boundary trace for outlines of thresholded objects;
  `lib/gl3d.ts` (`heightFieldMesh`, `GLView`) for the patch surface and the optional
  homogeneous 3D view; `lib/random.ts` (`gauss`) for seeded noise; `lib/snake.ts`
  (`Contour`, `resample`, `perimeter`, `normals`) for closed outlines, so spline outlines and
  snakes share one contour type; `lib/fft.ts` for the Fourier-descriptor comparison.
- `lib/moments.ts` is not needed (Legendre is the moments day's; `runge` uses Chebyshev
  nodes only).

**New lib files this day owns** (names checked against `demos/src/lib/`: no collisions):

- `demos/src/lib/spline.ts` (no DOM): Bernstein basis, de Casteljau (with all intermediate
  levels for drawing), Bézier split and derivative; knot vectors (uniform, clamped, periodic);
  Cox–de Boor basis values (with the tower of intermediate values for `bsplinebasis`), de
  Boor evaluation, derivatives, knot insertion; rational (NURBS) evaluation via homogeneous
  lift and divide, `nurbsCircle()`/`quarterCircle()`; curvature and curvature comb, arc length
  (Gauss–Legendre per span), $\oint\kappa\,ds$; polynomial interpolation (Lagrange/barycentric
  form, equispaced/Chebyshev nodes); natural and periodic cubic interpolating splines (via
  `thomas` / a cyclic tridiagonal solve); least-squares regression spline; penalised
  (P-spline) smoothing with second-difference penalty, hat-matrix trace and leave-one-out/GCV
  scores. Also the thin-plate radial basis and its linear system (optional unit 6). Shared by `runge`,
  `casteljau`, `bsplinebasis`, `bsplinecurve`, `splinefit`, `splineoutline`, `chaikin`, `tpswarp`; later recalled by a spline snake (snakes day) if built.
- `demos/src/lib/splineimage.ts` (no DOM): uniform B-spline kernels $\beta^0..\beta^3$, the
  cubic B-spline prefilter (causal + anti-causal recursion, mirror boundary), 2D coefficient
  computation, cubic B-spline sampling and its exact gradient at sub-pixel positions, the
  cubic-convolution (Keys) kernel and sampler, frequency responses of the interpolators;
  a degree-1 FFD variant for the "degree 1 vs 3" comparison, and warp via `imageops.warp`
  with a chosen sampler. The cubic FFD displacement field and its Jacobian determinant are
  **imported from `lib/deform.ts`** (owner: optical flow day), not duplicated; `bspline3` and
  its derivative live in `lib/spline.ts` and `deform.ts` re-exports them. Shared by
  `splinezoom`, `ffd`.
- `demos/src/lib/ctrlpoly.ts` (DOM, SVG): a control-polygon editor: draggable points with
  large handles, add/delete, open/closed, optional per-point weight, selection callback,
  highlight of affected spans; built on `ui.ts` `draggable`/`svgPoint`. Shared by
  `casteljau`, `bsplinecurve`, `chaikin`, `splineoutline` (manual edit mode).
- **Request to the `imageops.ts` owner** (not done here): add `"cubic"` and `"bspline"` to the
  `Interp` union so `warp`/`rotate`/`resize` can take them, implemented by calling
  `splineimage.ts`; until then demos call `splineimage.ts` directly.
- Entries for the three new files in `demos/src/lib/README.md` when built.

**Demo rules** (`demos/README.md`, `plan/build-issues.md`): TypeScript only (no Wasm expected;
the heaviest live step is a 256² cubic warp, measure first); fixed seeds for all noisy data;
presets + reset in every demo; pointer-only dragging, keyboard left to reveal.js; fits
1150×560 on a slide (two panels side by side, readouts under them); labels and readouts
≥ 18 px on slides (the Bernstein weights, the perimeter readouts and the FFD readouts are the
risks); per-demo CSS selectors prefixed `.qb-demo[data-demo="…"]` because `demo.css` loads
late; options via `data-*` (`data-preset`, `data-degree`, `data-mode`, `data-image`), so
`bsplinecurve` serves units 2 and 4 (spline snake picture) and `splinefit` serves unit 3 and
the outline intro; static fallback figures in each deck.

**Tests** (`demos/test/splines-test.ts`, run by `make test`): partition of unity of Bernstein
and B-spline bases (random knots, degrees 0–5); de Casteljau = Bernstein sum; de Boor = Cox–de
Boor sum; clamped cubic with 4 points = Bézier; NURBS quarter and full circle have radius 1 to
1e-12; natural spline reproduces linear data exactly and the 3-point example ($M_1 = -3$,
$S(0.5) = 0.6875$); smoothing spline with λ → 0 interpolates and λ → ∞ gives the least-squares
line, and the 3-point discrete example gives $(6/7, 9/7, 6/7)$; prefilter + B-spline sampling
reproduces integer samples to 1e-6 and the impulse coefficients are $\sqrt3\,z_1^{|k|}$; FFD
with zero displacements is the identity, with equal displacements a translation, and the
one-point values are $\tfrac49, \tfrac19, \tfrac1{36}$; curvature of a circle = 1/r and
$\oint\kappa\,ds = 2\pi$; arc length of a smoothing spline on a radius-10 digital disc within
2 % of $20\pi$ (record the actual value for the lecture).

**Images**: the course micrograph via `lib/cell.ts` (credit shown); everything else synthetic.
No new data or licences needed.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `bezier/`: lecture (`runge`, `casteljau`), de Casteljau and joins by hand |
| 10:45–12:15 | Unit 2 `bsplines/`: lecture (`bsplinebasis`, `bsplinecurve`), corner cutting and the quarter circle by hand |
| 13:15–14:45 | Unit 3 `spline-fit/`: lecture (`splinefit`, `splineoutline`), 3-point smoothing and natural spline by hand |
| 15:00–16:30 | Unit 4 `spline-images/`: lecture (`splinezoom`, `ffd`), prefilter and FFD numbers by hand; quizzes as wrap-up |

Optional `subdivision/` and `thin-plate/` as self-study or a fifth slot.

## Open questions

1. **Placement of NURBS**: in `bsplines/` (proposed: the homogeneous lift is two slides once
   B-splines exist, and keeps the day at four units), or its own unit with surfaces (would be
   a fifth core unit; split if `bsplines/` exceeds ~18 slides in the build, principle 6).
2. **How much numerics** of the smoothing spline: proposed discrete second-difference penalty
   (P-spline) in the demo and concept layer, the continuous $\int f''^2$ (Reinsch) only in the
   math layer. Fine, or show both?
3. **Unit 3 depends on Part II only lightly** (`shape-features/` perimeter bias): stated
   self-contained here; if taken after labelling, the slide becomes a recall.
4. **FFD folding bound**: confirm the constant for 2D cubic B-spline FFD (literature gives
   ≈ 0.40 of the spacing for 3D; the 2D value is larger) and phrase it as "keep displacements
   below about 0.4 of the grid spacing" if unsure.
5. **Split with the optical flow day** (`plan/opticalflow-day.md`, drafted in parallel): FFD
   representation here (`ffd`, hand-driven), registration energy and optimisation there
   (`ffdreg`, `lib/deform.ts`). Needed there: `nonrigid-reg/` points to `spline-images/`; its
   `bspline3` comes from `lib/spline.ts`; its by-hand FFD numbers and ours (4, 1, 0.25 px for a
   9 px drag) should use the same convention. If only one FFD demo is wanted, `ffd` could be
   `ffdreg` with `data-mode="manual"`; proposed: keep both (different questions).
6. **Pending edits to built units** (cross-day pointers, one line each): `images/lec.qmd`
   interpolation slide → "cubic and B-spline: splines day"; `snakes/lec.qmd` "Not covered
   here, but close" → spline snakes; `shape-features/lec.qmd` "Contour descriptors" →
   smoothing-spline outlines; `regularise/lec.qmd` → "smoothing splines are the same pattern for
   curves"; `orthogonal-moments/lec.qmd` → Runge pointer; `pipeline3d/lec.qmd` → NURBS use the
   same division (optional).
7. **Sidebar and coursemap**: add a "Splines and NURBS" sub-section under "Curves and energies"
   in `_quarto.yml` (after "Ising model and Markov random fields", or first, since its entry is
   Part I), and mark the coursemap row as planned.

## References

- de Boor, C. (1978/2001), *A Practical Guide to Splines*, Springer (Cox–de Boor, de Boor's
  algorithm, natural splines).
- Farin, G. (2002), *Curves and Surfaces for CAGD: A Practical Guide*, 5th ed., Morgan
  Kaufmann (Bézier, de Casteljau, continuity, rational curves).
- Piegl, L. & Tiller, W. (1997), *The NURBS Book*, 2nd ed., Springer (NURBS, circles, knot
  insertion).
- Runge, C. (1901), *Über empirische Funktionen und die Interpolation zwischen äquidistanten
  Ordinaten*, Z. Math. Phys.
- Reinsch, C. H. (1967), *Smoothing by spline functions*, Numer. Math.
- Holladay, J. C. (1957), *A smoothest curve approximation*, Math. Tables Aids Comput.
- Wahba, G. (1990), *Spline Models for Observational Data*, SIAM (smoothing splines, GCV).
- Eilers, P. H. C. & Marx, B. D. (1996), *Flexible smoothing with B-splines and penalties*,
  Statistical Science (P-splines).
- Unser, M., Aldroubi, A. & Eden, M. (1991), *Fast B-spline transforms for continuous image
  representation and interpolation*, IEEE PAMI; Unser, M. (1999), *Splines: a perfect fit for
  signal and image processing*, IEEE Signal Processing Magazine.
- Keys, R. (1981), *Cubic convolution interpolation for digital image processing*, IEEE TASSP.
- Thévenaz, P., Blu, T. & Unser, M. (2000), *Interpolation revisited*, IEEE TMI.
- Rueckert, D. et al. (1999), *Nonrigid registration using free-form deformations: application
  to breast MR images*, IEEE TMI.
- Choi, Y. & Lee, S. (2000), *Injectivity conditions of 2D and 3D uniform cubic B-spline
  functions*, Graphical Models (FFD folding bound).
- Chaikin, G. M. (1974), *An algorithm for high-speed curve generation*, Computer Graphics and
  Image Processing.
- Bookstein, F. L. (1989), *Principal warps: thin-plate splines and the decomposition of
  deformations*, IEEE PAMI.
- Brigger, P., Hoeg, J. & Unser, M. (2000), *B-spline snakes: a flexible tool for parametric
  contour detection*, IEEE TIP.

## Build notes (2026-11-13): what was built, deviations

Built: the four core units `bezier/`, `bsplines/`, `spline-fit/`, `spline-images/` (lec, quiz,
exercise each), demos `runge`, `casteljau`, `bsplinebasis`, `bsplinecurve`, `splinefit`,
`splineoutline`, `splinezoom`, `ffd`, libraries `lib/spline.ts`, `lib/splineimage.ts`,
`lib/ctrlpoly.ts`, tests `demos/test/splines-test.ts`, figures by `tools/figs/splines/figs.ts`.
Optional units 5 (`subdivision/`, `chaikin`) and 6 (`thin-plate/`, `tpswarp`) not built.

Deviations and corrected numbers:

- **Smoothing spline** (open question 2): `splinefit` uses the exact cubic smoothing spline
  (Reinsch: data + λ∫f''², hat matrix, exact LOO); the concept slides use the discrete
  second-difference example; outlines use a closed P-spline (second differences of the control
  points). λ values of the two are not comparable (said in the speaker notes).
- **NURBS stays in `bsplines/`** (open question 1): 20 slides including 4 math slides; not split.
- **FFD**: `lib/deform.ts` (optical flow day) did not exist when this day's libraries were
  written; the FFD (grid, displacement, field, Jacobian, warp) lives in `lib/splineimage.ts`,
  degree 1 and 3. `bspline3`/`bspline3d` are in `lib/spline.ts`. Consolidation with
  `lib/deform.ts` (it should re-export `bspline3` from `lib/spline.ts`) is left to its owner.
  The `ffd` demo deforms forward (content at x moves to x + u(x)), rendered by a fixed-point
  inverse; handles sit at control point + displacement.
- **FFD folding bound** (open question 4): one control point moved along x folds at 2.25 h (2D;
  1.5 h in 1D); two neighbours moved in opposite directions at about 1.2 h each; the general
  sufficient bound quoted is Choi & Lee's ≈ 0.49 h in 2D (0.40 h in 3D).
- **Perimeter of the radius-10 disc**: edge count 80, chain code 64.3, smoothing spline 62.9
  (0.2 % above 20π = 62.83; m = 40, λ = 10^0.5). The radius-3 disc shows the other side: the same
  λ gives 17.9 against 18.85 (smoothing shrinks small objects), used as an exercise.
- **Rotation test** (10 × 36°, micrograph 128², RMS inside the central disc): nearest 28.1,
  bilinear 21.6, B-spline without prefilter 25.2, cubic convolution 14.8, cubic B-spline 11.3:
  "half of bilinear's change", not "barely changes".
- **Impulse**: the 2D B-spline without prefilter peaks at (2/3)² = 0.44 of the impulse (1D: 0.67).
- **C¹ join** in `casteljau` for a first piece of degree n: Q1 = P_n + (n/3)(P_n − P_{n−1}).
- Not built from the demo plans: the homogeneous 3D view in `bsplinecurve`, the height-field
  surface view and the sub-pixel shift operation in `splinezoom`, the thresholded micrograph and
  the manual edit mode in `splineoutline` (the nucleus does not threshold cleanly), a
  tensor-product surface figure in `bsplines/` (text only). The micrograph appears as a profile
  in `runge` and `splinefit` and as an image in `splinezoom` and `ffd`.
- `splineoutline/boundary.ts` imports `traceContour` from `demos/src/moments/regionprops.ts`
  (labelling day) for the chain-code length.
- Pending cross-day edits (one-line pointers in `images/`, `snakes/`, `shape-features/`,
  `regularise/`, `orthogonal-moments/`, `pipeline3d/`) and the request to add "cubic"/"bspline"
  to `imageops.ts`'s `Interp` were not done (other owners).
