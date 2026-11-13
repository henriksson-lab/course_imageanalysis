# Plan: a day on edge-preserving smoothing and PDEs (Part II)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.
From `plan/coursemap.md`, section "Queue, second batch": **Edge-preserving smoothing and PDEs**
(heat equation = Gaussian blur, anisotropic diffusion, total variation flow, bilateral and
non-local means in depth, inpainting, stability of explicit schemes). Part II
(transforms and reconstruction), **entry after Deconvolution (day 5)**. This day **owns** nonlinear diffusion
(Perona–Malik and its relatives), TV *flow* (TV as a time evolution), stability of explicit
diffusion schemes, bilateral and non-local means **in depth**, and classical (diffusion and
exemplar) inpainting. It **recalls** scale space (edges day), TV as a prior (Ising day,
`gmrf/`), the denoisers (deconvolution day, `denoise/`) and the CFL condition (level-set
day, `levelset-eq/`). See "Proposed reorganisation" for the ownership edits this implies.

**Course map (2026-11-13, self-study restructure):** day 10 of 51, Part II "Transforms and reconstruction". Hard prerequisites: Deconvolution (day 5), Edges, blobs and ridges (day 2). Soft (see also): Ising model and Markov random fields (day 28), Level sets (day 31). Source: `days.yml`.

## Goal of the day

After the day a student can:

1. explain why **repeated local averaging is diffusion**, and why running the **heat equation**
   for a time $t$ is exactly a **Gaussian blur** with $\sigma = \sqrt{2t}$ (the scale space of the
   edges day, now as a process in time); take one explicit time step by hand on a small grid;
2. choose a **stable time step** for an explicit diffusion scheme ($\Delta t \le \Delta x^2/4$ in
   2D), recognise the **checkerboard explosion** of an unstable step, contrast this limit with
   the CFL rule for moving fronts (level sets: $\Delta t \propto \Delta x$, here $\propto \Delta x^2$),
   and say what an **implicit** step buys (any $\Delta t$; one implicit step is the quadratic
   smoother of the Ising day);
3. run **Perona–Malik diffusion**: an **edge-stopping function** $g(|\nabla u|)$ with a contrast
   parameter $K$; predict which edges survive; explain **backward diffusion** (gradients above
   about $K$ get *steeper*), **staircasing** of ramps, why the stopping time is a regularisation
   parameter, and why a little pre-smoothing (Catté) makes the method well behaved;
4. describe **total variation flow** as "each level line moves by its curvature": flat regions
   lose contrast at a rate perimeter/area (a disc of radius $r$: $2/r$), so small objects vanish
   first; connect flow time with the λ of TV denoising; compare heat, TV and Perona–Malik as
   **gradient descent on three energies**;
5. compute **bilateral** and **non-local means** weights by hand, read a weight map, predict
   failure (noise ≈ edge height; unique detail with no copies; $h$ vs noise level), explain the
   **effective number of averaged pixels** and why NLM is a diffusion on a graph of patches;
6. **inpaint** a hole by diffusion (harmonic: the hole solves Laplace's equation with the rim as
   boundary), by TV (keeps edges; connects a broken bar only if the gap is shorter than the bar
   is wide) and by **exemplar copying** (fill order, best-matching patch); say what may never be
   measured in an inpainted region;
7. compare **all smoothers on the same image** (Gaussian, median, Perona–Malik, TV, bilateral,
   non-local means, wavelet shrinkage, Wiener) by what each **assumes about the image** and what
   it destroys, using PSNR/SSIM, method noise and line profiles; and
8. *(math)* derive $G_{\sqrt{2t}} * f$ as the heat-equation solution, the von Neumann
   amplification factor $1 - 4r\sin^2(\omega/2)$ and its limit, the Euler–Lagrange equations of
   $\int|\nabla u|^2$, $\int|\nabla u|$ and $\int\rho(|\nabla u|)$, the backward-diffusion
   condition $\phi'(s) < 0$ for the flux $\phi(s) = s\,g(s)$, the $2/r$ contrast rate, and the
   discrete Laplace system of harmonic inpainting.

Why it fits this audience: it is calculus (partial derivatives, divergence, the chain rule,
Taylor), one new idea (a PDE as "change each pixel a little, repeat") and linear algebra
(the update as a matrix, its eigenvalues decide stability; implicit steps are sparse linear
systems). Everything is shown in a demo first; the PDE language sits in the math layer.

## Prerequisites and recalls

Entry is after day 5, so the days 1–5 are assumed. Two of the recalled units sit in the
"Curves and energies" group (now Part VI), whose entry is after day 27, so a student may arrive here
without them: their recalls must be **self-contained** (one concept slide each, with a pointer
"in depth: …").

- **`scale-space/`** (day 2): the math slide "scale space as diffusion"
  ($\partial_t L = \tfrac12\nabla^2 L$, $t = \sigma^2$) and causality. Unit 1 turns this math
  slide into the concept: the heat equation is the day's starting point. Convention: this day
  writes $u_t = \Delta u$ (the PDE literature's form), so $\sigma = \sqrt{2t}$; one line says it
  is the same equation as in `scale-space/` with time measured differently.
- **`kernels/`** (day 1): the Laplacian kernel $[0\,1\,0;\,1\,{-4}\,1;\,0\,1\,0]$ and border
  rules (mirror border = no heat leaves the image). Recalled, not re-taught.
- **`canny/`** (day 2): gradient magnitude at scale σ (the edge-stopping function reads
  it; Catté pre-smoothing is "use the Canny σ").
- **`structure-tensor/`** (day 2): orientation and coherence; only for the optional
  `tensor-diffusion/` unit (its slide "coherence-enhancing diffusion (pointer)" points here).
- **`denoise/`** (day 5): one slide each on bilateral and non-local means (formulas in its
  math slide), the `denoisers` demo, method noise, "a restored image is an estimate". Unit 4
  starts from these slides and goes deeper; unit 6 extends the comparison.
- **`regularise/`** (day 5): data + λ·penalty, "stopping early regularises". Units 2–3
  recall it: the diffusion time plays the role of λ.
- **`gmrf/`** (Ising): quadratic smoothing = sparse system $(I + \lambda L)x = y$,
  TV denoising (Rudin–Osher–Fatemi), staircasing, contrast loss. Unit 1 uses the system as the
  implicit step; unit 3 recalls TV in one slide and then treats it as a *flow*.
- **`levelset-eq/`** (level sets): forward Euler, upwind differences, the CFL rule
  ("the front may not move more than one pixel per step"), curvature flow of a curve. Unit 1
  recalls CFL in one slide; unit 3 uses "a curve moving by its curvature" for the level lines of
  TV flow (stated with a picture, so it needs no level-set background).
- **`shrinkage/`** (wavelets): appears only as one method in the unit-6 comparison
  (one line, pointer).
- **Divergence** may be rusty: one slide recalling $\nabla\cdot\mathbf v$ as "net outflow"
  (same picture as `levelset-eq/`), drawn as flux arrows between pixels.

## Overlaps with other plans and units (who takes what)

| Plan / unit | What this day takes | What stays there |
|---|---|---|
| `edges-day.md`, `scale-space/` | The heat equation as the subject of a unit: explicit steps, cost ∝ σ², stability, nonlinear diffusivities | Scale space, causality, the semigroup, LoG/DoG; this day recalls them in one slide |
| `deconvolution-day.md`, `denoise/` | Bilateral and NLM **in depth**: weight maps, failure analysis, effective sample size, iterated/joint bilateral, NLM as a patch-graph diffusion, parameter choice vs σn; the multi-method comparison with PDE smoothers (unit 6) | The first introduction (one slide each), the median as a denoiser, method noise, the measurement-honesty slides, the `denoisers` demo. `denoise/` "Beyond these four" gets a pointer here |
| `ising-day.md`, `gmrf/` (owns TV) | TV **flow** (time evolution, level lines by curvature, the $2/r$ rate), TV inpainting, TV next to PM in the energy picture | TV as a prior, ROF as a MAP estimate, Condat 1D, the `mrf1d` demo, prior samples; `lib/smooth.ts` (`tv2d`, `tv1d`, `thomas`) is reused, not copied |
| `levelsets-day.md`, `levelset-eq/` | Stability of **diffusion-type** (parabolic) explicit schemes, $\Delta t \le \Delta x^2/4$, von Neumann analysis, implicit/semi-implicit steps | Upwind, CFL for fronts, curvature flow of curves, reinitialisation. Its line "curvature terms need $\Delta t \lesssim \Delta x^2/4$: just mention it" becomes a pointer here |
| `levelsets-day.md`, `geodesic/` | nothing | The edge-stopping function $g(|\nabla I|)$ for *speeds*; unit 2 names the shared idea in one line ("the same $g$ stops a front there and stops heat here") |
| `diffusion-inverse/` (diffusion models) | nothing | Inpainting with a learned prior (replacement); unit 5 points to it ("a generative model fills a hole with plausible content, possibly invented") |
| Queued **Sparse representations** day | Harmonic, TV and exemplar inpainting | Sparse/dictionary inpainting; that day recalls unit 5 rather than re-teaching inpainting |
| `structure-tensor/` | Coherence- and edge-enhancing (tensor) diffusion in the optional `tensor-diffusion/` | The tensor, orientation, coherence |
| `horn-schunck/`, `nonrigid-reg/`, `spline-fit/` | nothing | One "diffusion is everywhere" line in unit 1: HS's Jacobi iterations and demons-style regularisation smooth a flow field by diffusion; a smoothing spline penalises curvature |
| Topic page "Denoising" (planned) | Unit 6's comparison table and test card are the natural content of that page | The topic page links here instead of duplicating |

## Units (one folder each: lec, quiz, exercise with #explore)

Six units: five lecture units of about 1–1.25 h (lecture + exercise) and a shorter closing
comparison unit (45 min). Order: linear diffusion and numerics first (everything later is a
diffusion with a twist), then the two nonlinear PDEs, then the non-PDE neighbourhood filters
(which turn out to be diffusions on a different graph), then inpainting (diffusion *into* a
hole), then the comparison. Folder names checked against existing folders and demos (no
collisions; `diffusion/` and `diffusion2d` belong to the generative-model days, so this day
avoids "diffusion" in names).

Figures: every unit lists its figures; target about half the content slides with an image
beside text (design principle 3b). Scripts in `tools/figs/<unit>/` (TypeScript using the
shared libs, like `tools/figs/gmrf/figs.ts`). Images with `![](fig.svg){fig-alt="..."}` (no
bracket alt text: it becomes a slide caption), `.nostretch` on lone slide images, per-demo CSS
prefixed `.qb-demo[data-demo="…"]`.

### 1. `heat-equation/`: Smoothing as heat flow: the heat equation and explicit schemes

Concept layer:

- **Average your neighbours, repeat.** One step: each pixel moves a little towards the mean of
  its four neighbours: $u \leftarrow u + \Delta t\,(\text{neighbour sum} - 4u)$, i.e.
  $u \leftarrow u + \Delta t\,\nabla^2 u$ with the Laplacian kernel of `kernels/`. Heat flows
  from warm to cold; the total is kept (mirror borders: nothing leaves).
- **The heat equation = Gaussian blur.** Repeating the step is the heat equation
  $u_t = \Delta u$; after time $t$ the image equals $G_{\sigma} * f$ with
  $\sigma = \sqrt{2t}$ (recall `scale-space/`: the same scale space, now built by a process).
  Picture: 10, 40, 160 small steps beside Gaussians σ = 0.9, 1.8, 3.6.
- **So why bother with a PDE?** For a linear blur, never: convolving is faster (the explicit
  scheme needs about $\sigma^2$ steps: σ = 10 → 200 steps). The PDE view pays off the moment the
  amount of smoothing **depends on the image** (units 2–3) or only part of the image is updated
  (unit 5). This is the day's motivating slide: "a diffusivity $c(x)$ per pixel".
- **One step by hand** on a 1D row (0, 0, 4, 0, 0), $r = \Delta t/\Delta x^2 = 1/4$:
  (0, 1, 2, 1, 0); total stays 4.
- **Too large a step explodes.** Same row with $r = 1/2$: (0, 2, 0, 2, 0) (the centre
  overshoots to 0); $r = 1$: (0, 4, −4, 4, 0) and growing: a **checkerboard** pattern that
  doubles in sign every step. Rule: $\Delta t \le \Delta x^2/2$ in 1D, $\Delta x^2/4$ in 2D,
  $\Delta x^2/6$ in 3D ("no pixel may give away more than it has").
- **Recall CFL** (one slide, self-contained; in depth in `levelset-eq/`): moving fronts must not
  move more than one pixel per step, $\Delta t \le \Delta x/|F|$. Diffusion is stricter:
  $\Delta t \propto \Delta x^2$, so halving the pixel size means 4× more steps on 4× more pixels
  (16× the work in 2D). Table: front vs diffusion, 1D/2D/3D limits.
- **Implicit steps**: solve for the new image instead of computing it: $(I + \Delta t\,L)\,u^{new}
  = u^{old}$. Stable for any Δt (but a huge step is just a big blur, inaccurate in time). This is
  **exactly the quadratic smoother of `gmrf/`** with λ = Δt (recall in one line, self-contained
  for students without the Ising day): "one implicit heat step = one GMRF denoising".
- **Diffusion is everywhere** (one slide, pointers only): Horn–Schunck's flow smoothing
  (`horn-schunck/`), regularising a deformation field (`nonrigid-reg/`), smoothing splines
  (`spline-fit/`), heat-kernel ideas behind the diffusion models (different "diffusion": noise is
  added, not averaged; one line to avoid the name clash).

Math layer:

- *The Gaussian solves the heat equation*: $\partial_t G_{\sqrt{2t}} = \Delta G_{\sqrt{2t}}$ by
  differentiation; in Fourier, $\hat u_t = -|\omega|^2\hat u$ so
  $\hat u(\omega, t) = e^{-|\omega|^2 t}\hat f(\omega)$: a Gaussian low-pass of width
  $\sigma^2 = 2t$ (recall `fourier/`).
- *Explicit scheme as a matrix*: $u^{n+1} = (I - \Delta t\,L)\,u^n$ with the graph Laplacian $L$;
  eigenvalues of $L$ on a grid lie in $[0, 4d/\Delta x^2]$; stability ⇔ all eigenvalues of
  $I - \Delta t L$ in $[-1, 1]$ ⇔ $\Delta t \le \Delta x^2/(2d)$.
- *Von Neumann*: insert $u_j^n = A^n e^{ij\omega}$: amplification $A = 1 - 4r\sin^2(\omega/2)$;
  worst at $\omega = \pi$ (the checkerboard), $|A| \le 1$ ⇔ $r \le 1/2$.
- *Maximum principle*: for $r \le 1/2$ the update is a convex combination of old values (weights
  $1 - 2r, r, r$): no new extrema (the discrete causality of `scale-space/`).
- *Implicit/semi-implicit*: $A = 1/(1 + 4r\sin^2(\omega/2)) \in (0, 1]$ for all $r$; with
  Neumann borders $L$ is diagonalised by the DCT (recall `dct/` in one line if taken), so one
  implicit 2D step costs two DCTs.

**Demo `heatflow`** (new): tabs *1D*, *2D*, *by hand*.
*1D*: a 128-sample profile (step, narrow peak, ramp, noise; preset: a line across `cell256`).
Play/step/reset; Δt as a multiple of the limit (0.1–1.5); scheme explicit / implicit (Thomas,
`lib/smooth.ts`); border mirror / zero. Overlays: the Gaussian blur at $\sigma = \sqrt{2t}$
(dotted; coincides for stable steps), mass readout, and a small plot of the amplification
factor $A(\omega)$ for the current $r$ with the band $|A| \le 1$ (the checkerboard frequency
marked). *2D*: the micrograph or a synthetic test card under explicit steps next to
`gaussianBlur` at $\sigma = \sqrt{2t}$ and their difference (mid-grey = 0); readouts: steps,
time, σ, ms. *By hand*: `gridStepper` (`lib/gridui.ts`) on a 7×7 grid with numbers, one
explicit step per click with the 5-point stencil highlighted. Presets: "heat = Gaussian",
"σ = 10 takes 200 steps", "at the limit", "1.2 × the limit: the checkerboard grows",
"implicit: any Δt is stable", "zero border: heat leaks out", "mirror border: the mean is
kept".

Figures (≈ 18 content slides, ≥ 9 with an image): `fig-average-repeat.svg` (a 1D row, three
steps, arrows of heat flow); `fig-heat-vs-gauss.png` (micrograph after 10/40/160 steps vs
Gaussian, difference maps); `fig-steps-sigma.svg` (steps needed vs σ, quadratic); `fig-handstep.svg`
(the (0,0,4,0,0) example at r = 1/4, 1/2, 1); `fig-checkerboard.png` (2D explosion after 5, 10,
15 steps at 1.2× the limit); `fig-amplification.svg` ($A(\omega)$ for r = 0.25, 0.5, 0.6);
`fig-cfl-table.svg` (front vs diffusion, the two limits as pixel pictures); `fig-implicit.svg`
(explicit vs implicit at Δt = 2 on the step); `fig-flux-divergence.svg` (flux arrows between
pixels, net outflow); demo slide.

By hand (concept): (a) one explicit step on (0, 0, 4, 0, 0) at r = 1/4, 1/2, 1, and the total
each time; (b) one 2D step at the centre of a given 3×3 patch at Δt = 0.2 and 0.3: which is
allowed? (c) σ after 50 steps at Δt = 0.25 (σ = √25 = 5); how many steps for σ = 8? (d) a
microscope image resampled to half the pixel size: how much longer does a diffusion to the
same physical σ take? (e) one implicit step for y = (0, 3, 0), Δt = 1 (the `gmrf/` answer
(0.75, 1.5, 0.75); verify by plugging in). *(math)*: (f) show $G_{\sqrt{2t}}$ solves
$u_t = \Delta u$; (g) derive $A(\omega)$ and the limit $r \le 1/2$, then $r \le 1/4$ in 2D;
(h) show the implicit scheme's $A$ is in $(0, 1]$.

### 2. `perona-malik/`: Edge-stopping diffusion: Perona–Malik

Concept layer:

- **The wish**: smooth inside regions, not across edges. Let the "conductivity" depend on the
  local gradient: $u_t = \nabla\cdot\big(g(|\nabla u|)\,\nabla u\big)$; $g \approx 1$ where the
  image is flat (heat flows), $g \approx 0$ at strong edges (insulation).
- **Edge-stopping functions** with a contrast parameter $K$: $g_1 = e^{-(s/K)^2}$,
  $g_2 = 1/(1 + (s/K)^2)$. Plots. $K$ separates "noise gradients" from "edge gradients": a
  choice like a threshold (link: the hysteresis thresholds of `canny/`).
- **Discrete form**: between each pair of neighbours a flux $g(|d|)\,d$ with $d$ the difference.
  By hand: row (10, 12, 10, 50, 50), $K = 5$, $g_2$, Δt = 0.25: heat gives (10.5, 11.0, 20.5,
  40.0, 50.0), Perona–Malik gives (10.43, 11.14, 10.58, 49.85, 50.0): the noise is smoothed,
  the step is almost untouched ($g_2(40) = 0.015$).
- **Backward diffusion: edges sharpen.** The flux $\phi(s) = s\,g(s)$ rises up to $s = K$
  ($g_2$; $K/\sqrt2$ for $g_1$) and then *falls*: a steeper edge passes *less* heat. Across a
  soft edge steeper than $K$ heat flows uphill and the edge **steepens**. By hand: (0, 2, 6, 10,
  12) with $K = 2$ becomes (0.25, 1.95, 6, 10.05, 11.75): shoulders flatten, middle steepens.
  Picture: φ(s) with the forward and backward ranges shaded, the gradient histogram beneath.
- **Price 1: staircasing.** A ramp steeper than $K$ breaks into terraces; a noisy ramp becomes
  steps at random places. Picture: profile after 0/50/500 steps.
- **Price 2: ill-posed.** Backward diffusion amplifies some noise: an isolated spike with a
  gradient above $K$ is kept forever. Cure (Catté et al.): compute $g$ from a **slightly
  smoothed** gradient, $g(|\nabla G_\sigma * u|)$ (the Canny σ): noise no longer counts as edges,
  real edges still do.
- **When to stop?** Run forever and every region merges into the mean (insulation is never
  perfect). The **stopping time** is a regularisation parameter like λ or the iteration count
  of `regularise/` ("stopping early regularises"). Demo: PSNR vs time has a maximum.
- **Isotropic or anisotropic?** Honest naming slide: Perona–Malik calls it "anisotropic", but
  the conductivity is a number per pixel (the flux is still along the gradient); true
  anisotropy (smooth *along* an edge, not across) needs a tensor: optional `tensor-diffusion/`,
  structure-tensor link. One line: the same $g$ slows a front at edges in geodesic active
  contours (`geodesic/`, level-set day).
- **Time step**: since $g \le 1$, the heat limit still holds ($\Delta t \le 1/4$ in 2D).
- Pointer: image libraries call it "anisotropic diffusion".

Math layer:

- *Divergence form and flux*: in 1D $u_t = \partial_x(\phi(u_x))$, $\phi(s) = s\,g(s)$; expanding,
  $u_t = \phi'(u_x)\,u_{xx}$: the local "diffusion coefficient" is $\phi'$, negative where
  $\phi$ decreases ⇒ backward heat equation, which is ill-posed (high frequencies grow like
  $e^{|\phi'|\omega^2 t}$).
- *Where the switch happens*: $g_2$: $\phi'(s) = (1 - s^2/K^2)/(1 + s^2/K^2)^2 < 0$ for $s > K$;
  $g_1$: $\phi'(s) = (1 - 2s^2/K^2)e^{-s^2/K^2} < 0$ for $s > K/\sqrt2$.
- *In 2D*: decomposing along the gradient direction η and the level line ξ:
  $u_t = g(|\nabla u|)\,u_{\xi\xi} + \phi'(|\nabla u|)\,u_{\eta\eta}$ — along the edge it always
  smooths; across the edge it smooths or sharpens.
- *Discrete maximum principle*: with $0 \le g \le 1$ and $\Delta t \le 1/4$ each update is a
  convex combination of neighbours: no new extrema in the discrete scheme (even though the
  continuous PDE is ill-posed: the grid regularises it, which is why results depend on the
  grid).
- *Catté regularisation*: $g(|\nabla G_\sigma * u|)$ makes the problem well-posed (stated).

**Demo `peronamalik`** (new, main demo of the morning): tabs *1D* and *2D*. *1D*: a profile
(step, soft step, ramp, narrow spike, noise); the flux plot φ(s) with the profile's gradient
histogram underneath, $K$ draggable on it, backward range shaded; play/step; the profile with
truth and the heat result for comparison. *2D*: test card (`synth.testCard`, see shared
engineering) or `cell256` with seeded noise; panels: current image, conductivity map
$g(|\nabla u|)$ (white = flows, black = insulates), line profile across step + ramp + thin line;
controls: $g_1/g_2$/linear, $K$, pre-smoothing σ (0 = classical), Δt (as multiple of the limit),
play/step/reset, noise level. Readout: time, PSNR vs truth, plus a small PSNR-vs-time curve
with its maximum marked ("best stopping time"). Presets: "linear: plain blur", "edges stop
the heat", "K too small: noise kept", "K too large: linear again", "soft edge sharpens
(backward)", "ramp becomes a staircase", "a spike survives forever", "pre-smoothing fixes the
spike", "run forever: everything flattens", "nuclei in the micrograph".

Figures (≈ 19 content slides, ≥ 10 with an image): `fig-conductivity.png` (image and its $g$
map); `fig-g-functions.svg` ($g_1$, $g_2$, linear); `fig-pm-vs-heat.png` (same noisy test card,
heat vs PM at equal time); `fig-pm-byhand.svg` (the (10,12,10,50,50) row, both results);
`fig-flux.svg` (φ(s) with forward/backward shading and the gradient histogram);
`fig-sharpen.svg` (the (0,2,6,10,12) example, before/after); `fig-staircase.svg` (ramp profile
after 0/50/500 steps); `fig-spike.png` (spike kept, then removed with pre-smoothing);
`fig-psnr-time.svg` (PSNR vs time, maximum); `fig-xi-eta.svg` (level line ξ and gradient η at an
edge pixel, math slide); demo slide.

By hand (concept): (a) $g_2$ at $s$ = 1, 5, 20 for $K = 5$ (0.96, 0.5, 0.06): which differences
are noise? (b) one PM step on (10, 12, 10, 50, 50) and the same step for heat; (c) the
sharpening row (0, 2, 6, 10, 12), $K = 2$: compute the fluxes and the update, say which values
moved apart; (d) predict: what happens to a 1-pixel bright dot of height 30 on a flat
background with $K = 5$, without and with pre-smoothing σ = 1? (e) the stopping time: with
noise σn = 5 and edges of 40, suggest $K$ and say why. *(math)*: (f) derive $\phi'$ for $g_1$ and
$g_2$ and the switch points; (g) show the 1D discrete update is a convex combination for
$\Delta t \le 1/2$ when $g \le 1$; (h) derive the ξ/η decomposition.

### 3. `tv-flow/`: Total variation flow: smoothing as energy descent

Concept layer:

- **Recall TV** (one slide, self-contained; in depth in `gmrf/`): the total variation sums
  $|\nabla u|$; an edge costs its height times its length, not the square; TV denoising keeps
  steps, staircases ramps, lowers the contrast of small plateaus.
- **Three smoothers, three energies**, and each PDE is "go downhill on the energy":
  heat ↔ $\int|\nabla u|^2$ (big gradients very expensive ⇒ blur them away); TV ↔
  $\int|\nabla u|$ (all gradient sizes cost the same per unit ⇒ sharp edges are cheap);
  Perona–Malik ↔ $\int\rho(|\nabla u|)$ with a $\rho$ that **flattens out** (large gradients
  almost free ⇒ edges kept; non-convex ⇒ the result depends on the start and the stopping
  time). Picture: the three penalty curves side by side (the "robust statistics" view, Black et
  al.). Link to `regularise/`: same data + penalty pattern.
- **TV flow**: $u_t = \nabla\cdot(\nabla u/|\nabla u|)$. Concept picture: every **level line**
  (iso-contour) of the image moves by its **curvature** (the curve-shortening of `levelset-eq/`,
  shown as an animation, no level-set background needed). Consequences: straight edges do not
  move; corners round off; **flat regions lose contrast at a rate perimeter/area**.
- **Small things go first**: a bright disc of radius $r$ and contrast $c$ keeps its shape and
  sharp edge but its contrast drops at rate $2/r$ and it disappears at $t = c\,r/2$. TV flow is
  a **scale space of its own** where scale = area/perimeter, not σ. Picture: discs of radius
  4, 8, 16 at the same contrast, flowing.
- **Flow time ↔ λ**: TV denoising with weight λ (`gmrf/`) and TV flow for time $t \approx \lambda$
  behave alike (exactly equal in 1D; close in 2D). By hand in 1D: y = (0, 0, 0, 6, 6, 0, 0, 0),
  λ = 2: the middle plateau (width 2, two jumps) drops by $2\cdot2/2 = 2$ to 4, the outer
  plateaus (width 3, one jump each) rise by 2/3; the steps stay sharp; the mean is kept.
- **Numerics honestly**: $1/|\nabla u|$ is infinite on flat regions; one uses
  $|\nabla u|_\varepsilon = \sqrt{|\nabla u|^2 + \varepsilon^2}$, then the conductivity is up to
  $1/\varepsilon$ and the explicit step must shrink to $\Delta t \le \varepsilon/4$ ("stiff"): small ε
  = thousands of steps. That is why TV denoising is solved by other algorithms (a dual
  projection, words only; the demos use `lib/smooth.ts`), and why unit 1's stability matters
  in practice.
- **Compare with Perona–Malik** (table): edges kept (both); staircasing (both, for different
  reasons); contrast loss (TV: predictable, ∝ perimeter/area; PM: almost none at strong edges);
  well-posed (TV yes, convex; PM no); parameters (TV: λ or time; PM: $K$, time, pre-smoothing).

Math layer:

- *Euler–Lagrange*: for $E(u) = \int\Psi(|\nabla u|)$, the gradient (first variation) is
  $-\nabla\cdot\big(\Psi'(|\nabla u|)\,\nabla u/|\nabla u|\big)$; gradient descent
  $u_t = \nabla\cdot(g\nabla u)$ with $g = \Psi'(s)/s$: $\Psi = s^2/2$ → heat; $\Psi = s$ → TV;
  $\Psi = \tfrac{K^2}2\log(1 + s^2/K^2)$ → Perona–Malik $g_2$ (non-convex for $s > K$: the
  backward range of unit 2).
- *Curvature of level lines*: $\nabla\cdot(\nabla u/|\nabla u|) = \kappa$, the curvature of the
  level line through the point (recall $\kappa$ from `implicit/`, or define it here as
  $1/\text{radius}$).
- *The $2/r$ rate*: integrate $u_t$ over the disc: $\frac{d}{dt}\int_D u = \oint_{\partial D}
  \nabla u/|\nabla u|\cdot\mathbf n = -|\partial D|$ (unit normal flux out of a bright plateau),
  so the height falls at $|\partial D|/|D| = 2\pi r/\pi r^2 = 2/r$.
- *ROF and the flow*: optimality of ROF, $u - y = \lambda\,\nabla\cdot(\nabla u/|\nabla u|)$, is one
  implicit step of TV flow with $\Delta t = \lambda$; in 1D (and for radial data) the two coincide.
- *Stiffness*: $g = 1/|\nabla u|_\varepsilon \le 1/\varepsilon$ ⇒ $\Delta t \le \varepsilon\,\Delta x^2/4$.

**Demo `tvflow`** (new): tabs *1D*, *discs*, *image*. *1D*: profile (plateaus of widths 2, 4, 8,
a ramp, noise); flow mode (ε, explicit) and ROF mode (λ, `tv1d` from `lib/smooth.ts`)
overlaid, heat for comparison; readout: each plateau's height vs time with the predicted
straight line (slope = jumps/width). *Discs*: discs of radius 3–20 and squares/stars on a flat
background; play TV flow (ROF with increasing λ, `tv2d`, for speed) and watch small discs fade
first and corners round; a plot of contrast vs time per disc with the $c - 2t/r$ lines;
optional overlay of level lines. *Image*: test card or `cell256` with noise; TV vs PM vs heat at
a matched amount of smoothing (equal PSNR or equal flat-region noise); line profile; method
noise. Presets: "plateaus sink at rate jumps/width", "small discs vanish first", "corners
round, edges stay", "ramp staircases", "flow time ≈ λ", "small ε: thousands of steps",
"TV vs PM vs heat on the micrograph".

Figures (≈ 17 content slides, ≥ 9 with an image): `fig-three-penalties.svg` (Ψ for heat, TV,
PM, and their $g$); `fig-energy-descent.svg` (a 1D row descending, energy vs iteration);
`fig-level-lines.png` (iso-contours of a test image before/after TV flow); `fig-discs-flow.png`
(three discs flowing, contrast plot); `fig-tv-byhand.svg` (the (0,0,0,6,6,0,0,0) example);
`fig-corner-round.png` (square under TV flow); `fig-stiff.svg` (steps needed vs ε);
`fig-tv-pm-heat.png` (three results + profiles); comparison table slide (no image); demo slide.

By hand (concept): (a) the ROF/TV result for y = (0, 0, 0, 6, 6, 0, 0, 0), λ = 2, and for
λ = 6 (the middle plateau would drop by 6 and the outer rise by 2: they meet — the bump is
gone, all values 1.5); (b) discs of radius 5 and 20 with contrast 10: when does each vanish
under TV flow? (c) rank three penalties by how much they charge one step of height 10 vs ten
steps of height 1 (heat ½s²: 50 vs 5; TV: 10 vs 10; PM with $K = 2$: $2\ln 26 \approx 6.5$ vs
$10\cdot 2\ln 1.25 \approx 4.5$, so one big step costs barely more than ten small ones);
(d) predict TV flow on a bright square: shape after a while? *(math)*: (e) derive the
Euler–Lagrange equations of the three energies; (f) derive the $2/r$ rate; (g) show ROF's
optimality condition is an implicit TV-flow step.

### 4. `bilateral-nlm/`: Neighbourhood filters in depth: bilateral and non-local means

Concept layer:

- **Recall** (one slide, `denoise/`): bilateral = close *and* similar; NLM = similar *patches*
  anywhere in a search window; both are weighted means. This unit opens the box.
- **The weight map**: for one pixel, draw the weights of all neighbours as an image. Bilateral at
  a flat spot = a Gaussian blob (it *is* a Gaussian blur there); at an edge = a **half-moon**
  (only the own side); at a corner = a quarter. NLM: bright wherever the patch repeats, even far
  away. Pictures of the four cases.
- **Effective number of averaged pixels** $N_{\text{eff}} = (\sum w)^2/\sum w^2$: noise falls by
  about $\sqrt{N_{\text{eff}}}$. At an edge the bilateral loses half its pixels (less denoising
  near edges: visible as a noisy rim); NLM on a lattice finds hundreds; on a unique spot only
  itself.
- **Choosing the range parameter**: bilateral σr vs noise σn and edge height Δ: σr ≪ Δ keeps the
  edge; σr ≪ σn averages almost nothing (noise looks like edges). Rule of thumb σr ≈ 2σn when
  Δ ≫ σn; it fails when Δ ≈ σn (recall `denoise/`). NLM: even identical patches differ by
  $d^2 \approx 2\sigma_n^2$, so $h$ ~ σn; too small: only itself; too large: blur.
- **Self-weight problem** (NLM): the pixel's own patch always matches perfectly ($d = 0$) and
  dominates when good matches are rare; fixes: give it the weight of the best other match.
- **By hand, 1D NLM**: x = (11, 9, 30, 10, 12, 29, 9, 11), patch 3, pixel 3 (value 10, patch
  (30, 10, 12)), $h = 10$ on the mean squared patch difference: the patch at position 6
  ((29, 9, 11), $d^2 = 1$) gets weight 0.99, the others ≈ 0.06–0.10; the estimate is 10.9,
  where a 3-pixel mean would give 17.3.
- **Bilateral is one step of an edge-stopping diffusion**: its range weight plays the role of
  $g$ in Perona–Malik (unit 2), applied to differences of values rather than gradients; iterate
  it and you get the same **cartoon** and staircases; it can also make **gradient reversal** /
  halo artefacts near soft edges. One slide, picture of 1, 3, 10 iterations.
- **NLM is a diffusion on a graph of patches**: pixels are nodes, NLM weights are edges; one
  NLM step = one step of heat flow on this graph (row-normalised weights = a random walk).
  "Non-local" = the graph connects pixels that look alike, not pixels that are close. Link
  (one line) to `ncut/` (weights between pixels as a graph) and to unit 1.
- **Joint (cross) bilateral**: take the range weights from a *different*, cleaner image — a
  bright-field or a second fluorescence channel guides the smoothing of a noisy channel. Risk:
  structure from the guide is copied into the result. One slide with an example.
- **Cost**: bilateral $O(\text{pixels} \times \text{window})$; NLM $O(\text{pixels} \times
  \text{search window})$ thanks to box sums of squared differences (recall the integral image in
  one line; `denoise/` math slide). Patch-group methods that also *collaborate* across matches
  (one line, no names) and learned denoisers are the next step (pointer to `denoise/`'s learned
  restoration slide).

Math layer:

- *Weights and normalisation*: $\hat u_p = \sum_q w_{pq}u_q/\sum_q w_{pq}$; bilateral and NLM
  weights (recall the `denoise/` math slide); $N_{\text{eff}}$ and the variance of a weighted
  mean of independent noise: $\sigma_n^2\sum w^2/(\sum w)^2$.
- *Expected patch distance*: $\mathbb E\,d^2(P_p, P_q) = d^2(\text{clean}) + 2\sigma_n^2$, hence
  the subtraction $\max(d^2 - 2\sigma_n^2, 0)$ variant and $h \sim \sigma_n$.
- *Bilateral as a robust estimator*: one bilateral step is one iteration of reweighted least
  squares for $\min_m \sum_q w^s_{pq}\,\rho(u_q - m)$ with $\rho$ such that $\rho'(d)/d$ is the
  Gaussian range weight (Elad 2002; Durand & Dorsey) — the same robust-penalty picture as
  unit 3.
- *NLM as graph diffusion*: $W$ the weight matrix, $D$ its row sums; one step
  $u \leftarrow D^{-1}Wu = u - D^{-1}(D - W)u$: an explicit step with the random-walk graph
  Laplacian and Δt = 1.

**Demo `patchweights`** (new): image (test card: step, soft step, corner, lattice of dots,
one unique spot, fine texture; or `cell256`) with seeded noise. Click a pixel → panels:
bilateral weight map split into spatial factor, range factor and product; NLM weight map over
the search window (with the matched patch outlines for the top 5 matches drawn on the image);
readouts: $N_{\text{eff}}$, estimate, truth. Sliders: σs, σr, patch size (3/5/7), search
window (7–31), $h$, noise; toggles: NLM self-weight fix, *iterate bilateral* (1/3/10 times),
*joint bilateral* (guide = noise-free test card or a second synthetic channel with a feature the
noisy one lacks). Presets: "bilateral on a flat region = Gaussian", "at an edge: a half-moon",
"corner: a quarter", "noise ≈ edge height: random weights", "NLM on the lattice: all copies",
"NLM on the unique spot: only itself", "h too small: self-weight", "search window too small",
"iterated bilateral: cartoon", "joint bilateral copies the guide's edge".

Figures (≈ 18 content slides, ≥ 9 with an image): `fig-weightmaps.png` (flat / edge / corner /
lattice, bilateral and NLM); `fig-neff.svg` ($N_{\text{eff}}$ map over the test card, low at
edges and on the unique spot); `fig-sigmar.png` (σr = 0.5σn, 2σn, 8σn); `fig-nlm-byhand.svg` (the
1D example with patches drawn and weights); `fig-selfweight.png`; `fig-bilateral-iter.png` (1, 3,
10 iterations; cartoon, halo); `fig-patch-graph.svg` (pixels as nodes, NLM edges to look-alikes
far away); `fig-joint.png` (noisy channel, guide, joint result, copied-structure warning);
demo slide.

By hand (concept): (a) bilateral at the centre of (12, 10, 50) with σs large, σr = 10 (weights
$e^{-0.02} \approx 0.98$, 1, $e^{-8}$): the estimate; (b) the 1D NLM example above: compute the
six patch distances and weights, then the estimate; (c) $N_{\text{eff}}$ for weights (1, 1, 1, 1)
and (1, 0.1, 0.1, 0.1); how much noise reduction? (d) predict the bilateral weight map at a
T-junction; (e) which filter for a crystal lattice in EM, for a single vesicle, for a noisy
channel with a clean bright-field partner? *(math)*: (f) variance of a weighted mean;
(g) $\mathbb E\,d^2 = d^2_{\text{clean}} + 2\sigma_n^2$; (h) write one NLM step as an explicit step
of the random-walk Laplacian.

### 5. `inpainting/`: Filling holes: diffusion and exemplar inpainting

Concept layer:

- **Why fill holes in scientific images?** Dead/hot pixels and columns, saturated spots,
  scratches and dust, seams between tiles, burnt-in scale bars and text, a removed object.
  First slide sets the rule for the whole unit: **inpainted pixels are invented** — fine for
  display, for a filter that needs a full image (an FFT, a registration), or for a segmentation
  that must not trip over the hole; **never measured**. Keep the mask and exclude it.
- **Harmonic (diffusion) inpainting**: let heat flow *into* the hole from its rim and wait for
  equilibrium: every hole pixel becomes the mean of its four neighbours (Laplace's equation;
  the rim values are fixed). In 1D it is linear interpolation: known 2 and 10 around three
  missing pixels → 4, 6, 8. By hand in 2D: two missing pixels a, b side by side, a's other
  neighbours 10, b's other neighbours 30: $4a = 30 + b$, $4b = 90 + a$ → a = 14, b = 26.
- **Iterating is slow**: Jacobi/Gauss–Seidel = the heat equation of unit 1 run inside the hole;
  information crosses a hole of width $w$ in about $w^2$ iterations (unit 1's σ² law). A direct
  sparse solve is faster (one line).
- **Diffusion blurs**: fine for thin holes (a scratch, a dead column), bad for large holes: the
  fill is smooth, edges entering the hole fade out.
- **TV inpainting**: minimise TV inside the hole: edges entering the hole are continued
  **sharply**. The "connectivity" rule: TV pays edge length, so a dark bar of width $w$ broken
  by a gap of length $d$ is reconnected if $d < w$ (two edges of length $d$ are cheaper than two
  end caps of length $w$), otherwise it is cut. Picture of both cases. Curvature-based and
  transport methods (continue level lines smoothly; Bertalmío et al.): one line.
- **Exemplar (patch) inpainting**: for texture, no PDE works: copy. Fill from the boundary
  inwards, one patch at a time: pick the boundary pixel with the highest **priority** (many known
  pixels around it, and a strong edge running into the hole), find the **best-matching patch**
  elsewhere in the image (sum of squared differences over the known pixels only, as in NLM),
  copy its missing pixels. Edges are continued first, texture fills in after. Picture: fill
  order over time.
- **Exemplar failure**: it copies whatever matches, including an object that was never there;
  a repeated nucleus in a hole is a **fabrication**. Pointer: a generative model does the same
  with a learned prior (`diffusion-inverse/`), more convincingly — and the same rule applies.
- **Which method** (decision table): thin defect → harmonic; edges crossing a medium hole →
  TV; texture or large hole → exemplar; any measurement → none, mask it.
- Forward pointer (one line): inpainting with a sparsity prior is on the sparse-representations
  day (when built).

Math layer:

- *Harmonic inpainting*: minimise $\int_\Omega|\nabla u|^2$ with $u = f$ on $\partial\Omega$:
  Euler–Lagrange $\Delta u = 0$ in Ω (unit 3's machinery); discretely a sparse linear system
  $L_{\Omega\Omega}u_\Omega = -L_{\Omega\partial}f_\partial$; mean-value property; Jacobi =
  explicit heat step with Δt = 1/4.
- *TV inpainting*: minimise $\int_\Omega|\nabla u|$ with the same boundary condition (or with a
  data term $\lambda\int_{\text{known}}(u - f)^2$ outside Ω); the connectivity rule as a
  comparison of two candidate solutions' TV ($2d\,c$ vs $2w\,c$ for contrast $c$).
- *Exemplar priority* (Criminisi et al.): $P(p) = C(p)\,D(p)$, confidence $C(p)$ = mean
  confidence of the known pixels in the patch, data term $D(p) = |\nabla I_p^\perp\cdot\mathbf n_p|/\alpha$
  (isophote strength along the front normal); masked SSD for matching.

**Demo `inpaint`** (new): image (test card with edges and a textured region; `cell256`;
a synthetic dot lattice; a bar for the connectivity rule). Mask: presets (scratch, dead
column, text, small and large holes, broken bar with gap slider, hole on texture) or drawn
with `LabelPainter` (`lib/paint.ts`, brush + undo). Method: harmonic (play Jacobi / Gauss–Seidel
iterations with a counter, or *solve*), TV, exemplar (play patch by patch: fill front, the
chosen target patch and the arrow from its source patch; patch size and search radius sliders).
Panels: result, error inside the hole vs truth (synthetic images), and for exemplar the fill
order map. Presets: "dead column: diffusion is enough", "large hole: diffusion blurs",
"Jacobi needs ~w² iterations", "TV continues edges", "bar: gap < width connects", "bar: gap >
width breaks", "texture: exemplar wins", "exemplar invents a nucleus", "text over the
micrograph".

Figures (≈ 17 content slides, ≥ 9 with an image): `fig-defects.png` (dead column, saturated
spot, scratch, seam on synthetic/micrograph data); `fig-harmonic-1d.svg` (the 2…10 example);
`fig-two-pixel.svg` (the a/b example with the stencil); `fig-jacobi.png` (hole after 10/100/1000
iterations); `fig-harmonic-vs-tv.png` (an edge entering a hole); `fig-connectivity.svg` (bar,
gap < w vs gap > w, the two TV costs); `fig-fill-order.png` (exemplar front over time);
`fig-fabrication.png` (exemplar copies a nucleus into the hole); decision-table slide (no
image); demo slide.

By hand (concept): (a) 1D: known 2 and 10 around three missing pixels (4, 6, 8); start Jacobi
from zeros and do two sweeps — how far off? (b) the 2D two-pixel hole (14, 26); (c) a bar of
width 6 px with a gap of 4 px and one of 10 px: which does TV reconnect? (d) masked SSD: a target
patch with 5 known of 9 pixels, three candidate patches given: which wins? (e) a scale bar
inpainted before measuring mean intensity in a region it crossed: what goes wrong, what to do
instead? *(math)*: (f) derive $\Delta u = 0$ from the Dirichlet energy and write the 2D
two-pixel system; (g) show Jacobi for the hole is the explicit heat step with Δt = 1/4;
(h) TV costs of the connected vs broken bar.

### 6. `smoothers/`: All smoothers on one image

Short closing unit (≈ 12 content slides, 45 min with the demo), designed to also feed the
planned "Denoising" topic page.

Concept layer:

- **One test card, eight methods**: a synthetic image with known truth (step, soft step, ramp,
  1-px line, small discs of several radii/contrasts, a dot lattice, one unique spot, smooth
  background, fine texture) plus the micrograph, the same seeded noise (Gaussian, Poisson,
  salt-and-pepper). Methods: Gaussian (= heat), median (`denoise/`), Perona–Malik, TV, bilateral,
  NLM, wavelet shrinkage (`shrinkage/`, if taken: one line), Wiener with PSF = point (`deconv/`).
- **Every smoother is a prior** (the key slide, a table): Gaussian — the image is smooth;
  median — outliers are rare and structures are wider than the window; Perona–Malik — piecewise
  smooth, edges higher than $K$; TV — piecewise constant, small total edge length; bilateral —
  nearby similar values belong together; NLM — the image repeats itself; wavelet shrinkage —
  few large coefficients; Wiener — a known power spectrum. The method wins where its belief is
  true and fails where it is not.
- **Scores disagree**: PSNR, SSIM, method noise, and **feature scores** read off profiles: edge
  width (10–90 %), line contrast kept, small-disc contrast kept, ramp staircasing, texture kept;
  plus cost. No method wins every column. Tuning each method for best PSNR (pre-computed) vs
  tuning by eye gives different winners ("best PSNR, wrong look" recalled from `denoise/`).
- **Smoothing before segmentation**: edge-preserving smoothing before a threshold or watershed
  gives cleaner masks than a Gaussian of the same strength (picture); intensities are still
  measured on the raw image within those masks (recall `denoise/`).
- **3D and anisotropic voxels** (one line each): all of these extend to stacks; use physical
  distances (pixel size in z) for the spatial parts; cost grows with the window volume.

Math layer: *The math: what each minimises* (one slide: the energies of units 1–4 and
`shrinkage/`/`deconv/` side by side, each as data + λ·penalty or as a weighted mean;
which are linear, convex, local).

**Demo `smoothercompare`** (new): test card or micrograph, noise type and level. *Grid* tab:
the eight results as thumbnails with a synchronised zoom lens (drag on any) and per-method
parameter sliders that start at the pre-computed best-PSNR setting; *method noise* tab: the
eight method-noise images; *profiles* tab: line profiles across step, ramp, line and a small
disc, truth dotted; *scores* tab: a table (PSNR, SSIM via `lib/metrics.ts`, edge width, line
contrast, disc contrast, staircase index, ms), best per column highlighted. Heavy methods
(NLM, TV, PM to convergence) computed on slider release with a "computing…" state and cached
per setting. Presets: "Gaussian noise, moderate", "salt and pepper", "Poisson, few photons",
"strong noise: everyone fails differently", "texture: NLM", "piecewise constant: TV",
"best PSNR vs best look", "smoothing before thresholding" (overlays a fixed threshold on each
result). Alternative (open question 4): extend the existing `denoisers` demo with these
methods via `data-mode="all"` instead of a new demo.

Figures (≈ 12 content slides, ≥ 6 with an image): `fig-testcard.png` (truth with feature
labels); `fig-eight.png` (the 8 results, one noise level); `fig-methodnoise8.png`;
`fig-profiles8.svg`; `fig-prior-table.svg` (each method's belief with a tiny pictogram);
`fig-scores.svg` (scores as a heat-map table); `fig-seg-after.png` (threshold after Gaussian vs
PM vs TV); demo slide.

By hand (concept): (a) match each of six degraded profiles (printed) to the method that made
it; (b) for each of four images (crystal lattice, cartoon-like segmentation map, tissue with
smooth gradients, image with hot pixels) choose a method and name the belief that justifies
it; (c) two methods have equal PSNR: which scores would you look at for counting small spots?
*(math)*: (d) classify the eight methods as linear/non-linear, convex energy/non-convex/no
energy, local/non-local, with a one-line reason each.

## Optional units

### 7. (optional) `tensor-diffusion/`: Smoothing along structures: edge- and coherence-enhancing diffusion

For groups who liked units 2 and the structure tensor. Replace the scalar $g$ by a **2×2
diffusion tensor** built from the structure tensor $J_\rho$ (`structure-tensor/`): same
eigenvectors, eigenvalues chosen so that heat flows **along** edges/fibres but not across.
Edge-enhancing diffusion (Weickert): across-edge conductivity $g(\mu_1)$, along-edge 1.
Coherence-enhancing diffusion: along the orientation, strength rising with coherence: closes
gaps in fibres, fingerprints, membranes. Math: $u_t = \nabla\cdot(D\nabla u)$, $D = R\,
\mathrm{diag}(\lambda_1, \lambda_2)R^\top$; discretisation with mixed derivatives (stated) and
its stability caveat. Demo: `peronamalik` `data-mode="tensor"` with ellipse glyphs of $D$ (via
`lib/plot.ts` ellipses) on fibres/fingerprint textures from `lib/synth.ts`. Figures: diffusion
ellipses on a fingerprint; CED closing gaps in a fibre image.

### 8. (optional) `fast-filters/`: Making edge-preserving filters fast

Semi-implicit/AOS schemes (Weickert) so PM runs with large Δt; the bilateral grid (lift to
(x, y, intensity), blur, slice) as a linear filter in a higher dimension; the guided filter
(local linear model, box sums: recall `integral-image/`) as a fast edge-preserving alternative;
GPU mapping (stencils are ideal, recall the GPU day if taken). Demo: a timing bench (`heatflow`
`data-mode="bench"`) comparing explicit, semi-implicit and convolution.

## Exercises: concept first, then math

Each unit's exercise follows the by-hand lists above (concept questions with a `Solutions`
callout, then `### Math questions`, each tagged *(math)*, with `Solutions (math)`), then
`## Explore {#explore}` with questions on the unit's demo, for example:

- `heatflow`: "Find the largest Δt that does not explode in 1D and in 2D. Is it what the rule
  predicts?"; "Run 50 steps at Δt = 0.25 and compare with the Gaussian: which σ matches?"
- `peronamalik`: "Find the K at which the soft edge starts to sharpen"; "Find the stopping time
  with the best PSNR for two noise levels: does it move?"
- `tvflow`: "Check the 2/r law on three discs"; "Which survives longer: a large faint disc or a
  small bright one?"
- `patchweights`: "Find a pixel where N_eff < 5 for NLM; why there?"; "Make the bilateral fail
  by raising the noise".
- `inpaint`: "Find the gap length at which TV stops reconnecting the bar; compare with the
  width"; "Draw a hole over a nucleus: which method invents what?"
- `smoothercompare`: "Which method keeps the 1-px line best at PSNR ≥ 30 dB?"

## Shared engineering

- **New lib files owned by this day** (names checked against `demos/src/lib/`; note that
  `diffusion.ts` is the diffusion-*models* library, and `smooth.ts`/`denoise.ts`/`levelset.ts`
  have other owners):
  - **`demos/src/lib/pde.ts`** (new, no DOM): explicit 5-point heat step (`heatStep(u, w, h, dt,
    border)`), 1D implicit step (via `thomas` from `lib/smooth.ts`), 2D implicit step with
    Neumann borders by DCT diagonalisation (`dct2`/`idct2` from `lib/dct.ts`), `stabilityLimit(dim,
    gmax)`, `amplification(r, omega, scheme)`; `diffusivity(kind, s, K)` and `flux(kind, s, K)` for
    linear / $g_1$ / $g_2$ (named to avoid confusion with `levelset.ts`'s `edgeStop`);
    `nonlinearStep(u, w, h, dt, kind, K, preSigma)` (Catté pre-smoothing via `gaussianBlur`);
    `tvFlowStep(u, w, h, dt, eps)`; energies `dirichletEnergy`, `tvValue` (re-export of
    `tv2dValue`), `pmEnergy`; profile scores `edgeWidth1090`, `staircaseIndex`; for the optional
    unit `tensorDiffusionStep` (from $J_\rho$ via `lib/edges.ts` `structureTensor`).
  - **`demos/src/lib/inpaint.ts`** (new, no DOM): `harmonicJacobi`, `harmonicGaussSeidel` (one
    sweep per call, for animation), `harmonicSolve` (conjugate gradient on the masked Laplacian),
    `tvInpaint(img, mask, w, h, iters)` (a masked Chambolle-type iteration, sharing code with
    `tv2d` through its owner if possible), `exemplarInpaint(img, mask, w, h, patch, search,
    record?)` with confidence/data-term priority and masked SSD (records each fill step for the
    demo), `fillFront(mask)`, mask presets.
- **Extensions through owners** (proposed to the owners, not copied locally):
  - `lib/denoise.ts` (deconvolution day): `bilateralWeights(img, w, h, x, y, σs, σr)` (returns the
    three factor maps), `nlmWeights(img, w, h, x, y, patch, search, h, selfFix)`, `effectiveN`,
    `jointBilateral(img, guide, …)`, `iterateBilateral`.
  - `lib/synth.ts`: `testCard(n, seed)` with truth and named feature regions (step, soft step,
    ramp, 1-px line, discs of radii 3–16 and several contrasts, dot lattice, unique spot, fine
    texture), reused by units 2–6; fibre/fingerprint textures for the optional unit if not yet
    there (the edges plan asked for them too).
  - `lib/smooth.ts` (Ising day): reuse `tv1d`, `tv2d`, `thomas`, `gmrf1d` as they are; if TV
    inpainting needs a masked data term, add a `weights` argument to `tv2d` there.
  - `lib/metrics.ts`: `psnr`, `ssim` reused; line/disc contrast-kept scores may live in `pde.ts`
    to keep `metrics.ts` generic.
- **UI**: `lib/plot.ts` (profiles, φ(s) with histogram, PSNR-vs-time, amplification factor,
  contrast-vs-time), `lib/gridui.ts` (`gridStepper` for the by-hand stencil), `lib/paint.ts`
  (`LabelPainter` for masks), `drawGrey`/`toRGBA` with `DisplayMode` "mid" for difference and
  method-noise images, diverging maps from `lib/colour.ts`. Per-demo CSS prefixed
  `.qb-demo[data-demo="…"]`; labels ≥ 18 px; layouts fit 1150×560 on slides with two image
  panels + one plot; full panel sets on exercise pages via `data-layout="full"`.
- **Speed** (TypeScript first, measure in the browser): an explicit heat/PM step on 256² is
  ~65 k × ~20 flops (< 1 ms), so `play` runs 5–20 steps per animation frame; the pre-smoothed
  gradient costs a Gaussian per step (~5–15 ms at 256², so 128² on slides). TV flow with small
  ε is stiff: the *discs* and *image* tabs use ROF via `tv2d` (200 iterations at 128² ≈ tens of
  ms, measure) with increasing λ as "time". NLM at 256² with a 15×15 search ≈ 225 offset passes
  with box sums (~100–200 ms): computed on slider release, 128² on slides. Exemplar inpainting:
  masked SSD over a bounded search region (e.g. 48×48 around the hole, 9×9 patches); a 1000-pixel
  hole ≈ 10⁸ operations, animated in steps of a few patches per frame. Wasm only if a
  recompute that should feel live measures above ~30 ms after these choices (candidates: NLM
  in `smoothercompare`, exemplar search); keep the TypeScript reference and an agreement test
  if so.
- **Tests** (`demos/test/pdesmooth-test.ts`, run by `make test`): explicit heat for time $t$
  matches `gaussianBlur` at $\sigma = \sqrt{2t}$ within 1 % away from borders; mirror border keeps
  the mean; 1D explicit step at $r = 0.5 + \epsilon$ grows the checkerboard mode, at $r = 0.5$
  does not; the 1D implicit step equals `gmrf1d` with λ = Δt; 2D DCT implicit step equals a
  CG solve; PM with $K \to \infty$ equals heat; the by-hand rows (10, 12, 10, 50, 50) → (10.43,
  11.14, 10.58, 49.85, 50.0) and (0, 2, 6, 10, 12) → (0.25, 1.95, 6, 10.05, 11.75); no new
  extrema for PM with Δt ≤ 1/4; ROF on (0, 0, 0, 6, 6, 0, 0, 0), λ = 2 gives (2/3, 2/3, 2/3, 4,
  4, 2/3, 2/3, 2/3) via `tv1d`; disc contrast under ROF falls by ≈ 2λ/r (within 10 % for r ≥ 6);
  harmonic two-pixel hole gives (14, 26) and the solve has a residual < 1e-8; Jacobi converges
  to the solve; exemplar inpainting reproduces a hole cut from an exactly periodic texture;
  bilateral with σr → ∞ equals a Gaussian; NLM with $h \to \infty$ equals the box mean over the
  search window; the 1D NLM by-hand estimate is 10.93.
- **Images**: `cell256.png` via `lib/cell.ts` (credit shown), synthetic test card and textures
  labelled as synthetic. Static figures in `tools/figs/<unit>/figs.ts` (as `tools/figs/gmrf/`),
  committed with the units, as PDF/print fallbacks for every demo.

## Suggested timetable

Dense; see open question 1.

| Time | |
|---|---|
| 09:00–10:15 | Unit 1 `heat-equation/`: lecture (`heatflow`), explicit step and stability by hand |
| 10:30–11:45 | Unit 2 `perona-malik/`: lecture, PM step and sharpening by hand, `peronamalik` |
| 11:45–12:30 | Unit 3 `tv-flow/` part 1: energies and TV flow (lecture) |
| 13:30–14:15 | Unit 3 part 2: `tvflow`, the 1D TV and disc questions |
| 14:15–15:15 | Unit 4 `bilateral-nlm/`: lecture, NLM by hand, `patchweights` |
| 15:30–16:30 | Unit 5 `inpainting/`: lecture, two-pixel hole and connectivity, `inpaint` |
| 16:30–17:15 | Unit 6 `smoothers/`: `smoothercompare` in pairs, prior table; quizzes as wrap-up |

## Proposed reorganisation

Changes to other plans and units that this plan implies (not done here):

1. **Ownership table in `plan/coursemap.md`**: add rows "Nonlinear diffusion (Perona–Malik,
   tensor diffusion), TV flow, stability of explicit diffusion schemes, bilateral/NLM in depth,
   classical inpainting → Edge-preserving smoothing and PDEs". `gmrf/` keeps TV as a prior
   (owner of TV); `denoise/` keeps the first introduction of all denoisers; `levelset-eq/` keeps
   CFL for fronts. The heat equation as a *concept* moves from a math slide in `scale-space/`
   to this day's unit 1 (scale-space keeps its math slide and gains a pointer).
2. **Group placement**: the day is not a transform. Proposal: a new day group
   **"Restoration"** (entry after Deconvolution, day 5) holding this day and, later, the queued
   **Sparse representations** day (which additionally needs wavelets). If a new group is not
   wanted, keep "Transforms" but list this day after Radon so the wavelets → compression chain
   stays intact.
3. **Soft prerequisites across groups**: this day's entry (day 5) is earlier than the
   "Curves and energies" entry (day 27), so `gmrf/` and `levelset-eq/` are soft prerequisites
   (recalled self-contained), not hard ones; record it this way in the planned `days.yml`.
   Conversely, `levelset-eq/` and `gmrf/` can list this day as a soft "see also".
4. **Pointer edits** once accepted: `levelset-eq/` "How large may a time step be?" (diffusion
   limit → `heat-equation/`); `levelsets-day.md` unit 2 line "curvature terms … just mention
   it" → pointer; `structure-tensor/` "coherence-enhancing diffusion (pointer)" →
   `tensor-diffusion/`; `denoise/` "Beyond these four" → `bilateral-nlm/` and `smoothers/`;
   `gmrf/` "TV elsewhere in the course" → `tv-flow/` and `inpainting/`; `diffusion-inverse/`
   inpainting slide → `inpainting/` as the classical baseline; `scale-space/` math slide →
   `heat-equation/`.
5. **Sparse representations day** (queue): it should recall harmonic/TV/exemplar inpainting from
   unit 5 and own only sparse/dictionary inpainting and compressed sensing.
6. **Topic page "Denoising"**: build it from unit 6's prior table and test card instead of a
   separate comparison.

## Open questions

1. **Six units in one day is dense.** Fallbacks in order: shorten unit 3 to one slot when most
   students took the Ising day (TV recall is then truly one slide); move exemplar
   inpainting to a short optional section; make unit 6 a take-home exercise and topic page.
   Alternative: split into two half-days ("PDE smoothing": units 1–3; "Neighbourhood filters and
   inpainting": units 4–6).
2. **Notation**: $u_t = \Delta u$ ($\sigma = \sqrt{2t}$) here vs $\partial_t L = \tfrac12\nabla^2L$
   ($t = \sigma^2$) in `scale-space/`. Proposal: keep the PDE literature's form here with one
   explicit conversion line; alternatively adopt the scale-space form everywhere (then the
   explicit limit reads $\Delta t \le \Delta x^2/2$ in 2D, which is easy to confuse).
3. **How much numerics?** The von Neumann factor is in the math layer only; the concept layer
   relies on the (0, 0, 4, 0, 0) example and the demo. Is the implicit/DCT step worth a math
   slide, or should it go to the optional `fast-filters/`?
4. **New `smoothercompare` or an extended `denoisers`?** One shell serving both days via
   `data-mode` follows "one demo, several units"; a new demo keeps `denoisers` simple for day 5.
   Proposal: new demo that imports the same lib functions, so the two cannot drift apart.
5. **TV inpainting algorithm**: a masked data term in `tv2d` (owner: Ising day) vs a separate
   routine in `lib/inpaint.ts`. Proposal: ask the owner to add a per-pixel weight argument.
6. **Real images**: a micrograph with real dead pixels/columns or a tile seam (inpainting), and
   a real oriented fibre image (optional unit; same need as the edges day's `structtensor`).
   Until licensed data exist, synthetic defects on `cell256`.
7. **Naming**: "anisotropic diffusion" for Perona–Malik is standard but not accurate; the deck
   says so on one slide. Agree on the unit title ("Edge-stopping diffusion: Perona–Malik").

## References

- Perona & Malik (1990), *Scale-space and edge detection using anisotropic diffusion*, IEEE PAMI.
- Catté, Lions, Morel & Coll (1992), *Image selective smoothing and edge detection by nonlinear
  diffusion*, SIAM J. Numer. Anal.
- Weickert (1998), *Anisotropic Diffusion in Image Processing*, Teubner (edge- and
  coherence-enhancing diffusion, AOS schemes, stability).
- Black, Sapiro, Marimont & Heeger (1998), *Robust anisotropic diffusion*, IEEE TIP.
- You, Xu, Tannenbaum & Kaveh (1996), *Behavioral analysis of anisotropic diffusion in image
  processing*, IEEE TIP (staircasing).
- Rudin, Osher & Fatemi (1992), *Nonlinear total variation based noise removal algorithms*,
  Physica D.
- Andreu, Ballester, Caselles & Mazón (2001), *Minimizing total variation flow*, Differential and
  Integral Equations; Bellettini, Caselles & Novaga (2002), *The total variation flow in R^N*,
  J. Diff. Eq. (discs keep their shape, contrast rate perimeter/area).
- Chambolle (2004), *An algorithm for total variation minimization and applications*, JMIV.
- Tomasi & Manduchi (1998), *Bilateral filtering for gray and color images*, ICCV.
- Elad (2002), *On the origin of the bilateral filter and ways to improve it*, IEEE TIP;
  Durand & Dorsey (2002), *Fast bilateral filtering for the display of HDR images*, SIGGRAPH.
- Petschnigg et al. (2004) and Eisemann & Durand (2004) (joint/cross bilateral).
- Buades, Coll & Morel (2005), *A review of image denoising algorithms, with a new one*,
  Multiscale Model. Simul. (NLM, method noise).
- Gilboa & Osher (2008), *Nonlocal operators with applications to image processing*, Multiscale
  Model. Simul. (NLM as graph diffusion).
- Bertalmío, Sapiro, Caselles & Ballester (2000), *Image inpainting*, SIGGRAPH.
- Chan & Shen (2001), *Mathematical models for local nontexture inpaintings*, SIAM J. Appl. Math.
  (TV inpainting, connectivity).
- Efros & Leung (1999), *Texture synthesis by non-parametric sampling*, ICCV; Criminisi, Pérez &
  Toyama (2004), *Region filling and object removal by exemplar-based image inpainting*, IEEE TIP.
- He, Sun & Tang (2013), *Guided image filtering*, IEEE PAMI; Chen, Paris & Durand (2007),
  *Real-time edge-aware image processing with the bilateral grid*, SIGGRAPH (optional unit 8).
- Courant, Friedrichs & Lewy (1928) (CFL), and any numerical PDE text for von Neumann analysis
  (teacher background).

## Build status (2026-11-13)

Built: units `heat-equation/`, `perona-malik/`, `tv-flow/`, `bilateral-nlm/`, `inpainting/`, `smoothers/`
(lec, quiz, exercise); demos `heatflow`, `peronamalik`, `tvflow`, `patchweights`, `inpaint`,
`smoothercompare`; libs `lib/pde.ts`, `lib/inpaint.ts`; tests `demos/test/pdesmooth-test.ts`
(not yet in the Makefile TESTS list); figure scripts in `tools/figs/<unit>/` (data.ts with the
course libs + figs.py; `tools/figs/smoothers/best.ts` regenerates the BEST table). Optional units 7–8
not built. Deviations from the plan, found while computing:
- The test card and the bilateral/NLM weight helpers live in `lib/pde.ts` (to move to `lib/synth.ts`
  and `lib/denoise.ts` through their owners). TV inpainting is a primal–dual routine in `lib/inpaint.ts`.
- A soft edge sharpens only for K well below its steepest slope (11 vs 19.3); a "frozen" spike melts
  once its surroundings rise; heat never improves PSNR on the test card at Δt = 0.225.
- tvflow's image tab matches methods by the discrepancy principle (SD(noisy − result) = σn) instead
  of flat-region noise (the micrograph has no flat region).
- Small digital discs vanish slightly faster than c·r/2 under TV flow (pixel outlines are longer).
- Smoothing-before-thresholding uses equal flat-region noise settings (TV λ 45: 105 wrong pixels vs
  Gaussian σ 2: 151), not best-PSNR settings.
