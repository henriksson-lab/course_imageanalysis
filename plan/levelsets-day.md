# Plan: a day on level set methods

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. describe a curve **implicitly** as the zero level of a function $\phi(x, y)$, read inside/outside from its sign, and compute the normal $\nabla\phi/|\nabla\phi|$ and the curvature $\kappa = \nabla\cdot(\nabla\phi/|\nabla\phi|)$ for simple $\phi$ (circle, line);
2. explain what a **signed distance function** is, why $|\nabla\phi| = 1$ is convenient, and why it must be restored now and then (**reinitialisation**);
3. derive the **level set equation** $\phi_t + F|\nabla\phi| = 0$ from the chain rule, and take one time step on a small grid by hand with **upwind** differences, choosing $\Delta t$ from the **CFL condition**;
4. explain why the implicit description handles **merging and splitting** for free, where an explicit polygon (a snake) needs special-case surgery;
5. design a **speed function** for segmentation: edge-based (geodesic active contours, $F$ from an edge-stopping function $g(|\nabla I|)$) versus region-based (**Chan–Vese**, $F$ from the mean intensities inside and outside), and predict their typical failure modes (leaking through weak edges, local minima, sensitivity to the initial curve).

Why it fits this audience: everything rests on partial derivatives, the gradient, the chain rule and Taylor expansions, which they have. The new ideas (a PDE, a time step, stability) are introduced as "move the surface a little, repeat" and are always shown in a demo first. Finite differences and the gradient via Sobel come straight from *Filtering & convolution*; Gaussian pre-smoothing from the same unit.

## Prerequisite gap

- **No PDEs or numerical methods.** The level set equation is presented as "a rule for how each pixel value of $\phi$ changes per time step", i.e. forward Euler = a first-order Taylor step (link to the Taylor slide in *Filtering & convolution*). Stability (CFL) is shown empirically in the demo before the condition is stated. No existence/viscosity-solution theory.
- **Divergence** may be rusty: one slide recalling $\nabla\cdot\mathbf{v} = \partial_x v_1 + \partial_y v_2$, with the picture "how much the normals spread out" (curvature of a circle $= 1/r$).
- **Segmentation**: thresholding is taught on the thresholding day (`plan/thresholding-day.md`); unit 4 back-references it in one line instead of a recap. Graph cuts (`seg-energy/`, `plan/graphcut-day.md`): Chan–Vese is the continuous cousin of the same data + boundary-length energy.
- **Dirac delta / Heaviside**: needed for Chan–Vese only in a smoothed form ($H_\varepsilon$, $\delta_\varepsilon$); introduce as "a soft step and its derivative", plotted.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `implicit/`: Curves as level sets

- Two ways to store a curve: **explicit** (a list of points, a parametric $\mathbf{C}(s)$) versus **implicit** ($\{(x, y) : \phi(x, y) = 0\}$). Contour lines on a map as the picture: the coast is the zero level of the height.
- Examples by formula: a line $\phi = ax + by - c$, a circle $\phi = \sqrt{x^2+y^2} - r$, an ellipse. Sign convention: $\phi < 0$ inside.
- Set operations for free: union $\min(\phi_1, \phi_2)$, intersection $\max$, complement $-\phi$. Two circles drawn as one $\phi$.
- **Topology is a property of the slice, not of $\phi$**: raise the cutting plane $\phi = c$ through a two-peaked surface and watch one contour become two. This is the central picture of the day.
- Geometry from $\phi$: normal $\mathbf{n} = \nabla\phi/|\nabla\phi|$; curvature $\kappa = \nabla\cdot\mathbf{n}$. Worked for the circle: $\kappa = 1/r$.
- **Signed distance function** (SDF): $|\phi|$ = distance to the curve. Then $|\nabla\phi| = 1$, $\mathbf{n} = \nabla\phi$, $\kappa = \nabla^2\phi$ (the Laplacian kernel they already know). Not every $\phi$ is an SDF: $\phi = x^2 + y^2 - r^2$ has the same zero set but not the same slope.
- On a pixel grid: $\phi$ is an image. Extracting the contour = **marching squares** (sign changes along cell edges, linear interpolation for the crossing). Name the link to marching cubes (3D surfaces) and note that the 16-case table and its ambiguous saddle case are covered there; here one cell is enough.
- **Demo `levelsurface`** (new): left, the surface $z = \phi(x, y)$ as a rotatable shaded mesh (≈ 48×48, WebGL via the shared `lib/gl3d.ts`, drag to rotate) with a translucent horizontal plane at height $c$; right, the top view: $\phi$ as a diverging colour map with the contour $\phi = c$ in black and the zero contour in white. Slider $c$; shape menu (circle, ellipse, two circles via min, square, "egg crate" $\sin x \sin y$); toggle "true SDF / squared distance / SDF × 3" (same contour, different surface); click a point to show $\mathbf{n}$ and $\kappa$ there. Presets: "one becomes two" (two circles, sweep $c$), "same curve, different φ", "corner: where κ blows up".
- By hand: on a 5×5 grid sample $\phi = \sqrt{x^2+y^2} - 1.5$; mark signs; on one grid cell interpolate the zero crossing (marching squares by hand); compute $\nabla\phi$ and $\kappa$ by central differences at one pixel and compare with $1/r$.

### 2. `levelset-eq/`: Moving a curve: the level set equation

- The task: move each point of the curve along its normal with speed $F$ (outward if $F > 0$). Explicit way: move the polygon points. Show it fail: points bunch up or spread out, a non-convex shape under inward motion forms a **swallowtail** (self-intersection), and two curves touching need surgery.
- Implicit way: let $\phi$ change with time so that its zero level moves. Derivation with the chain rule: a point on the curve satisfies $\phi(\mathbf{x}(t), t) = 0$ for all $t$; differentiate: $\phi_t + \nabla\phi\cdot\mathbf{x}'(t) = 0$; with $\mathbf{x}' = F\,\mathbf{n} = F\,\nabla\phi/|\nabla\phi|$ this gives
  $$\phi_t + F\,|\nabla\phi| = 0.$$
  Only the normal component of the motion matters; the tangential one just slides points along the curve.
- 1D picture first: $\phi(x) = |x| - 1$ (an "interval" from $-1$ to $1$); with $F = 1$ the V shifts down and the interval grows. Then 2D.
- Discretisation:
  - time: forward Euler, $\phi^{n+1} = \phi^n - \Delta t\,F\,|\nabla\phi^n|$ (= a first-order Taylor step);
  - space: **upwind** differences: use the one-sided difference on the side the information comes from (Osher–Sethian / Godunov scheme, written out with $\max/\min$ of $D^{-}_x\phi$ and $D^{+}_x\phi$). Show in 1D why central differences fail at the kink of $|x|$ (gradient 0 at the minimum, so the curve stops);
  - **CFL condition**: $\Delta t \le \Delta x / \max|F|$ in words: "the front may not move more than one pixel per step". Shown in the demo first (a factor-1.5 time step makes checkerboard noise explode), stated afterwards.
- Speeds: constant $F$ (expand/shrink, "grass fire"); **curvature flow** $F = -\kappa$ (smooths, shrinks; a circle shrinks as $r(t) = \sqrt{r_0^2 - 2t}$, an ODE they can solve); combinations $F = c - \varepsilon\kappa$. Curvature terms use central differences (diffusion-like, not upwind) and a stricter time step $\Delta t \lesssim \Delta x^2/4$: just mention it.
- **Topology for free**: two expanding blobs merge; a dumbbell under constant shrinking splits. (Trap to name: pure curvature flow of a planar curve never splits; it becomes convex and vanishes, Grayson's theorem. The demo has a preset that shows this, because students will expect the dumbbell to pinch.)
- **Reinitialisation**: after many steps $\phi$ gets steep or flat away from the front; restore an SDF without moving the zero contour. Two methods, both by picture: (a) recompute distances to the marching-squares contour (exact, slow for big grids, fine at 128²); (b) iterate $\phi_\tau = \operatorname{sign}(\phi_0)(1 - |\nabla\phi|)$. Narrow band: only update pixels near the front (mention).
- **Demo `curveflow`** (new, main demo of the morning): a 128×128 grid. Draw closed curves with the mouse (or pick shapes); the drawn polygon is converted to an SDF. Speed menu: constant $F$ (slider, ±), curvature $-\varepsilon\kappa$, both. Play / step / reset; slider $\Delta t$ as a multiple of the CFL limit (0.2–1.5); scheme upwind / central; reinitialise every $k$ steps (0 = never). Views: $\phi$ colour map with zero contour, or the $\phi$ surface (reuse the `levelsurface` renderer). Optional overlay: the explicit polygon moved with the same speed, side by side, to show swallowtails and crossing. Presets: "two blobs merge", "dumbbell splits (constant speed)", "dumbbell under curvature flow does not split", "star: curvature rounds corners", "swallowtail (explicit vs implicit)", "Δt too large", "central differences stall", "no reinitialisation: φ degrades".
- By hand: 1D grid $x = -3..3$, $\phi = |x| - 1$, $F = 1$, $\Delta t = 0.5$: one upwind step, check the new zero crossing is at $\pm 1.5$; try $\Delta t = 2$ and see the values oscillate. Circle under curvature flow: solve $r' = -1/r$ and find the vanishing time.

### 3. `geodesic/`: Edge-based segmentation: active contours

- Turn the image into a speed: stop the front where the image has edges. **Edge-stopping function** $g = 1/(1 + |\nabla (G_\sigma * I)|^2/k^2)$ (Gaussian smoothing and Sobel from *Filtering & convolution*): $g \approx 1$ in flat regions, $g \approx 0$ on edges.
- Snakes first, briefly (Kass, Witkin & Terzopoulos 1988): an explicit curve minimising internal (length, bending) + external ($-|\nabla I|^2$) energy. Works, but needs a good start, fixed topology, re-sampling of points. One slide with pictures; it motivates the implicit version.
- Level set version, three stages:
  1. $\phi_t + g\,(c - \varepsilon\kappa)\,|\nabla\phi| = 0$ (Caselles et al. 1993, Malladi–Sethian–Vemuri 1995): a balloon that slows down at edges;
  2. **geodesic active contours** (Caselles, Kimmel & Sapiro 1997): minimise the edge-weighted length $\int g\,ds$; gives an extra advection term $\nabla g\cdot\nabla\phi$ that pulls the front into the edge valley instead of only braking. Present the term and its picture (arrows of $-\nabla g$ pointing to edges), not the full Euler–Lagrange derivation;
  3. the balloon $c$ is needed to pass through flat regions and to enter concavities; too large $c$ makes the front **leak** through weak edges.
- Inside-out or outside-in: one large initial curve shrinking onto several objects splits automatically.
- **Demo `gac`** (new): image panel (synthetic: three blobs with one weak edge and a concave "C" shape, plus noise; real: the cell micrograph crop, 128×128) with an alternative view of $g$ and $-\nabla g$ arrows. Click to place initial circles, or "big box around everything". Sliders $\sigma$, $k$, balloon $c$ (with sign), $\varepsilon$, advection on/off. Play / step / reset; iteration counter. Presets: "shrink-wrap three blobs" (one contour splits into three), "leak through the weak edge", "C shape: stuck without balloon", "noise without smoothing: stops everywhere".
- By hand: a 1D step edge with values 0,0,0,10,10,10 smoothed by [1 2 1]/4; compute $|\nabla I|$ and $g$ per pixel for $k = 2$; where does a front moving with speed $g$ at $c = 1$ slow below 0.1? What is the advection direction on each side of the edge?

### 4. `chanvese/`: Region-based segmentation: Chan–Vese

- Motivation, with a back-reference to thresholding (thresholding day, `plan/thresholding-day.md`) and to `seg-energy/` on the graph-cut day: thresholding a noisy image fails. Edges are weak or blurred in many micrographs, but the **means** inside and outside differ.
- Chan–Vese energy (piecewise-constant Mumford–Shah):
  $$E(\phi, c_1, c_2) = \mu\,\text{Length} + \nu\,\text{Area}_{\text{in}} + \lambda_1\!\int_{\text{in}} (I - c_1)^2 + \lambda_2\!\int_{\text{out}} (I - c_2)^2.$$
  Small 1D example first: 8 pixels, a given split, compute $c_1$, $c_2$ and $E$; move the boundary one pixel and recompute.
- Alternate: for fixed curve, $c_1$, $c_2$ are the **means** inside and outside (one line of calculus). For fixed $c$'s, gradient descent on $\phi$ gives
  $$\phi_t = \delta_\varepsilon(\phi)\big[\mu\kappa - \nu - \lambda_1 (I - c_1)^2 + \lambda_2 (I - c_2)^2\big]$$
  (sign depends on the inside convention; fix it once on the slides). Read the bracket as a speed: a pixel joins the side whose mean it is closer to; $\mu\kappa$ keeps the curve short. Write it with smoothed $H_\varepsilon$, $\delta_\varepsilon$, plotted.
- Consequences: no image gradient needed (robust to noise and blur); with $\mu = 0$ the result is the threshold at $(c_1 + c_2)/2$, i.e. two-means clustering of intensities: literally the **isodata** fixed point of the thresholding day (unit 2 `otsu/`); with a wide $\delta_\varepsilon$ interior holes appear far from the initial curve (new topology, again for free); local minima, so initialisation still matters (a grid of small circles is the classic robust start).
- **Connection to graph cuts**: the same energy (data term + boundary length) is what `graphcut-seg/` minimises on a pixel graph. Graph cuts: global optimum for fixed $c_1, c_2$, discrete, metrication artefacts. Level sets: continuous, sub-pixel, smooth boundaries, any differentiable energy, but local minimum. Whichever day comes first, the second should show this slide.
- Pointers, not taught: multiphase Chan–Vese (two functions → four regions), vector-valued images, shape priors.
- **Demo `chanvese`** (new, main demo of the afternoon): image (synthetic: noisy two-level shapes with a hole, a blurred edge, and a gradient background; real: cell micrograph; noise slider for both). Initialisation: one circle, a grid of circles, or drawn. Sliders $\mu$, $\nu$, $\lambda_1/\lambda_2$ ratio, $\varepsilon$ of $\delta_\varepsilon$. Live evolution with play / step / reset; side panels: $\phi$ map, current $c_1$, $c_2$ shown on the histogram, energy versus iteration (should decrease), and thresholding at $(c_1 + c_2)/2$ for comparison. Toggle "edge-based (`gac` speed) vs Chan–Vese" on the same image, so the morning and afternoon demos meet. Presets: "noise: edges fail, regions work", "μ = 0 is thresholding", "large μ: smooth but loses detail", "hole appears", "gradient background: a single mean is wrong", "bad start: local minimum".
- By hand: the 8-pixel 1D example above for two splits and $\mu = 0$ / $\mu = 1$ (length = number of boundary points); sign of the speed bracket at a given pixel for given $c_1$, $c_2$.

### 5. (optional / further reading) `fastmarching/`: fronts that only grow

- If $F > 0$ always, store the **arrival time** $T(x, y)$ instead: $|\nabla T|\,F = 1$ (Eikonal equation). Fast marching (Sethian 1996) solves it in one sweep, like Dijkstra with an upwind update.
- With $F = 1$, $T$ is the distance transform, i.e. how to build an SDF (links back to reinitialisation). With $F = g$ from the image: minimal paths / livewire-like segmentation; and "flooding from seeds" is the picture behind **watershed** (being planned separately: name the link, do not duplicate).
- 3D: the same equations for surfaces in volumes (CT, confocal stacks); the result is extracted with **marching cubes** (planned separately).
- **Demo**: reuse `curveflow` with `data-mode="arrival"` (show $T$ as colour and its contour lines as successive fronts), seeds by click, speed constant or $g$.

## Shared engineering

- `demos/src/lib/levelset.ts`: grid $\phi$ as `Float64Array` $n \times n$; SDF construction from shapes and from a drawn polygon (exact point-to-segment distance, sign by even-odd rule); upwind (Godunov) $|\nabla\phi|$ for $F$ of either sign; curvature by central differences with a small $\epsilon$ in $|\nabla\phi|$; CFL-limited step; reinitialisation (both methods); smoothed $H_\varepsilon$/$\delta_\varepsilon$; optional narrow band. Shared by `curveflow`, `gac`, `chanvese`, `levelsurface` (for $\kappa$, $\mathbf{n}$).
- `demos/src/lib/contour.ts`: marching squares (with linear interpolation) returning polylines for SVG/canvas. **Coordinate with the marching cubes plan**, which needs the same 2D case as a warm-up; one implementation in lib.
- `demos/src/lib/surface.ts`: height-field mesh builder (diverging colour map, horizontal cut plane) on top of the shared WebGL renderer `lib/gl3d.ts` from the marching cubes plan (all 3D demos use WebGL; decided 2026-11-12). Used by `levelsurface` and `curveflow`.
- Gaussian and Sobel: use the shared `demos/src/lib/imageops.ts` (one home for linear image filters across all plans; decided 2026-11-12), for $g$ in `gac`.
- Diverging colour map + legend helper in `lib/ui.ts` (or `lib/colour.ts`): blue inside, red outside, white at 0; must read on a projector and in greyscale print (add contour lines).
- Performance: 128×128 grid, 2–10 steps per animation frame; measured target ≥ 30 fps on a teaching laptop. Animation must pause when the slide is not visible (reveal.js `slidechanged`) and when the page tab is hidden.
- Tests (node script, `make test`, shared with the graph-cut plan): circle under curvature flow follows $r^2 = r_0^2 - 2t$ within 1 px; constant $F$ moves a circle by $F t$; reinitialisation moves the zero contour less than 0.5 px and gives $|\nabla\phi| \approx 1$ near the front; Chan–Vese energy is non-increasing on a synthetic image; marching squares on an SDF circle gives the right length.
- Images: reuse `demos/src/fourier2d/cell256.png`, downsampled to 128×128 (credit: R. M. Hunt, CC BY 3.0, as in `make-image.R`). Synthetic images generated in TS from a fixed seed. Check early that Chan–Vese and GAC give a sensible result on this TEM crop (dark membranes on a mid-grey cytoplasm are not obviously two-phase); if not, add a fluorescence nuclei image with a clear licence.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js (drawing with the pointer only), fits 1150×560 on a slide, static fallback figures for print.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`levelsurface`, marching squares by hand) |
| 10:45–12:15 | Unit 2 lecture, upwind step by hand, `curveflow` |
| 13:15–14:45 | Unit 3 lecture + exercise (`gac`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`chanvese`); quizzes as wrap-up |

## Open questions

- How long is a "day" (hours, lecture vs exercise)? Unit 2 is the densest (derivation, upwind, CFL, reinitialisation); it could be split into `levelset-eq/` (equation, speeds, topology) and `levelset-num/` (upwind, CFL, reinitialisation), with unit 3 shortened to make room.
- How much numerics? Proposal: upwind and CFL shown in the demo and done once by hand in 1D, no proofs; the curvature time-step limit only mentioned. Is that enough, or too much for this course?
- Geodesic active contours: include the advection term $\nabla g\cdot\nabla\phi$ (needs one more idea), or stop at the "braking balloon" $g(c - \varepsilon\kappa)$? And are explicit snakes worth more than one slide?
- Order relative to the graph-cut day and to watershed / marching cubes: the plan assumes none; the Chan–Vese ↔ graph-cut slide and the marching-squares slide should be written in both directions.
- Is `cell256.png` suitable for Chan–Vese, or should a fluorescence image (clear two-phase) be added?
- Should the optional unit be fast marching, or 3D surfaces (level sets in volumes + marching cubes), depending on what the parallel plans cover?
- Programming language for the code parts is still open (see README).

## References

- Osher & Sethian (1988), *Fronts propagating with curvature-dependent speed: algorithms based on Hamilton–Jacobi formulations*, J. Comput. Phys.
- Sethian (1999), *Level Set Methods and Fast Marching Methods*, Cambridge University Press (2nd ed.).
- Osher & Fedkiw (2003), *Level Set Methods and Dynamic Implicit Surfaces*, Springer (chapters 1–7 for the numerics).
- Kass, Witkin & Terzopoulos (1988), *Snakes: active contour models*, IJCV.
- Caselles, Catté, Coll & Dibos (1993), *A geometric model for active contours in image processing*, Numer. Math.
- Malladi, Sethian & Vemuri (1995), *Shape modeling with front propagation: a level set approach*, PAMI.
- Caselles, Kimmel & Sapiro (1997), *Geodesic active contours*, IJCV.
- Chan & Vese (2001), *Active contours without edges*, IEEE Trans. Image Processing.
- Mumford & Shah (1989), *Optimal approximations by piecewise smooth functions and associated variational problems*, Comm. Pure Appl. Math.
- Sussman, Smereka & Osher (1994), *A level set approach for computing solutions to incompressible two-phase flow*, J. Comput. Phys. (reinitialisation PDE).
- Sethian (1996), *A fast marching level set method for monotonically advancing fronts*, PNAS.
- Grayson (1987), *The heat equation shrinks embedded plane curves to round points*, J. Differential Geom.
- Lorensen & Cline (1987), *Marching cubes*, SIGGRAPH.
