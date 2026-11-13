# Plan: a day on ray tracing

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 50 of 51, Part IX "3D and computing". Hard prerequisites: 3D graphics (day 48). Soft (see also): Colour (day 42), Deconvolution (day 5), How GPUs work (day 51). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part IX (row "3D: ray tracing", requested
2026-11-12); entry after the **3D graphics** day (`plan/graphics3d-day.md`: `pipeline3d/`,
`rasterise/`, `gpuformats/`, `volrender/`), which itself comes after marching cubes. Runs in
parallel with the planned **GPU** day (Part IX) and the planned **geometry from
images** day; both are linked, neither is required.

**Notation** (fixed with the built 3D units): rays $\mathbf r(t) = \mathbf o + t\,\mathbf d$
with $t$ = distance when $\lVert\mathbf d\rVert = 1$ (as in `volrender/`); unit normal
$\mathbf n$ (bold); refractive indices always with subscripts, $n_1, n_2$ (plain $n$ is the
near plane in `pipeline3d/` and noise in `psf/`); barycentric coordinates $(b_a, b_b, b_c)$,
not $\lambda$ (as in `rasterise/`; λ is the prior weight, and wavelength in `psf/`, so
wavelength is written in words here except in the optional optics unit, which follows
`psf/`); radiance $L$; transmittance $\mathcal T$ (as in `volrender/`); BRDF $f_r$, albedo
$\rho$; a probability density is $p(\cdot)$; samples per pixel is $N$. OpenGL/WebGL
conventions (camera looks down $-z$), as on the 3D graphics day.

## Goal of the day

After the day a student can:

1. build the **ray through a pixel** from the camera (the unit `pipeline3d/` chain run
   backwards) and explain ray casting as **the rasteriser's two loops swapped** (for each
   pixel, for each object);
2. **intersect a ray** with a sphere (a quadratic), a plane (one division) and a triangle
   (a $3\times3$ linear system whose solution *is* the barycentric coordinates), pick the
   nearest hit, and get the normal there; *(math)* derive each, including a numerically
   stable quadratic;
3. add **shadow rays**, **mirror reflection** and **refraction** (Snell's law, total internal
   reflection, the **Fresnel** split between reflected and transmitted light), and draw the
   **ray tree** of Whitted-style recursion with a depth limit; explain **shadow acne** as a
   floating-point problem;
4. explain why naive tracing costs **O(pixels × objects)**, and how **bounding-volume
   hierarchies** and **uniform grids** bring a ray down to roughly O(log objects); run the
   **slab test** and a BVH traversal by hand;
5. compare **rasterisation and ray tracing**: what each loops over, what each is good at
   (primary visibility vs shadows, reflections, refraction, global light), and what
   **hardware ray tracing** on GPUs accelerates;
6. state the **rendering equation** in words (light out = light emitted + light reflected
   from all incoming directions), estimate it by **Monte Carlo** (path tracing), predict
   that noise falls as **$1/\sqrt N$** in the samples per pixel, explain **importance
   sampling** (cosine-weighted, light sampling) and why a bad sampling density gives
   fireflies; and recognise **denoising** a path-traced image as the deconvolution day's
   denoisers trading noise for bias;
7. *(optional unit)* trace rays through a **lens**: Snell at two curved surfaces, the
   thin-lens equation in the paraxial limit, ray-transfer ($2\times2$) matrices, aberrations,
   the **focal shift** from a refractive-index mismatch, and where ray optics stops and
   diffraction (`psf/`) takes over.

Why it fits this audience: it is vectors, dot and cross products, a quadratic, a $3\times3$
linear solve (Cramer's rule as ratios of determinants = volumes), recursion, trees, and one
integral estimated by averaging random samples (mean and variance, which they know). It
closes a loop in the 3D group: `volrender/` already shoots one ray per pixel through a
volume; this day shoots them at surfaces and lets them bounce. The image-analysis payoff:
path tracing is a *simulated camera* (its noise is the shot noise of `cameranoise/`), its
denoising is the deconvolution day's problem, and the lens unit links the camera model to
the microscope of image formation.

## Prerequisite gap

- **Camera, matrices, rays from pixels**: `pipeline3d/` (slides "The camera as a change of
  basis", "Perspective by similar triangles", **"From pixels back to rays"**). Unit 1 opens
  with a one-slide recall; nothing about projection matrices is re-taught.
- **Barycentric coordinates, edge functions, z-buffer, Lambert / Blinn–Phong, normals**:
  `rasterise/`. Recalled in one slide each (unit 1: barycentrics and Lambert; unit 3: the
  z-buffer for the rasterisation-vs-rays comparison).
- **One ray per pixel through a volume, compositing, transmittance, `rayBox`**:
  `volrender/` ("One ray per pixel", "Light along a ray: emission and absorption"). Unit 1
  points back ("same rays, now hitting surfaces"); unit 3 teaches the slab test that
  `volrender/` used without derivation.
- **Linear light**: `rgb-linear/` (colour day): light adds in linear values; a path tracer
  accumulates linear radiance and converts to sRGB only for display (one line; tone mapping
  named only).
- **Floating point**: `floatingpoint/` (bit-depth day: "Cancellation", "Summing many
  pixels"). Unit 1's stable quadratic and unit 2's shadow acne recall it; unit 4's
  accumulation of thousands of samples in float32 recalls "Summing many pixels".
- **Noise and $1/\sqrt N$**: `cameranoise/` (shot noise $\sqrt N$, SNR, PSNR, owner of noise
  models) and `resolution/` (averaging $N$ beads gives $\sigma/\sqrt N$). Unit 4 recalls
  both; mean and variance are basic statistics (audience.md).
- **Denoisers**: `denoise/` (deconvolution day: Gaussian, median, bilateral, non-local
  means, method noise, "a restored image is an estimate"). Unit 4 recalls them in one slide
  and reuses the `denoisers` code.
- **PSF, NA**: `psf/` ("Diffraction: a hard edge in frequency", NA = $n\sin\alpha$, the
  refractive-index-mismatch bullet in "The forward model in one line"). Optional unit 5
  recalls it.
- **Not assumed**: any optics beyond NA (Snell's law is taught here from scratch), any
  probability beyond mean/variance/densities (Monte Carlo integration is taught here), any
  GPU knowledge (hardware ray tracing is one slide; the GPU day owns the hardware).

## Overlaps with other days (who owns what)

| Topic | Owner | Elsewhere |
|---|---|---|
| Camera, view/projection matrices, "a pixel is a ray" | 3D graphics `pipeline3d/` | unit 1 here: recall |
| Barycentric coordinates, edge functions, z-buffer, Lambert/Phong/Blinn–Phong, inverse-transpose normals | 3D graphics `rasterise/` | units 1 and 3 here: recall; unit 1 adds that ray-cast barycentrics are automatically the perspective-correct ones (math slide) |
| Ray marching through volumes, emission–absorption, compositing, MIP | 3D graphics `volrender/` | unit 1: pointer ("rays through a volume vs rays at surfaces"); unit 4: the volume-rendering integral is the rendering equation without scattering (one line) |
| **Ray–primitive intersection** (sphere, plane, triangle, box/slab test) | **this day**, units 1 and 3 | `volrender/` uses `rayBox`: pointer here; 3D graphics optional `voxelise/` (ray parity) and marching cubes `surface-measure/`: pointer |
| **Whitted ray tracing**: shadow rays, reflection, refraction, Snell, total internal reflection, Fresnel | **this day**, unit 2 | — |
| **Geometric (ray) optics**: thin lens, ray-transfer matrices, aberrations, focal shift from index mismatch | **this day**, optional unit 5 (proposed owner; see open questions) | image formation `psf/`: one-line pointer from its index-mismatch bullet; wave optics, PSF, OTF stay with `psf/` |
| **Acceleration structures** (BVH, uniform grids, SAH) | **this day**, unit 3 | 3D graphics `voxelise/` ("octrees … bounding-volume hierarchies, one slide") becomes a pointer here; GPU day: pointer |
| **Monte Carlo integration**, importance sampling, $1/\sqrt N$ for estimators | **this day**, unit 4 (proposed owner; no current owner) | Ising `mcmc/` owns Markov chain sampling (contrast slide: here samples are independent); VAE (`vae/`, ELBO estimates) and diffusion: pointer if they need it |
| Shot noise, SNR, PSNR | bit-depth day `cameranoise/` | unit 4: recall ("a path tracer counts simulated photons") |
| Denoisers, method noise, bias of a restored image | deconvolution `denoise/` | unit 4: recall and reuse; learned denoisers: modern deep learning (pointer) |
| SIMT, warps, branch divergence, memory coherence | GPU day (planned in parallel) | unit 3 "hardware ray tracing" slide: one-line recall if taken, self-contained otherwise |
| Hardware ray tracing (BVH traversal and triangle tests in fixed-function units) | **this day**, unit 3 (one slide) | GPU day: pointer here |
| Rendering for view synthesis (NeRF, Gaussian splatting) | geometry-from-images day (planned) | unit 3 summary / unit 4: one-line pointer (NeRF renders with `volrender/`'s compositing, not with surface hits) |
| Mesh data, winding, manifoldness | 3D graphics `pipeline3d/`, marching cubes `marchingcubes/` | unit 3 traces a marching-cubes mesh: recall |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`raycast/`, `whitted/`, `bvh/`, `pathtrace/`; optional `rayoptics/`. (Folder names
checked against the repo: none exists.)

### 1. `raycast/`: Rays from the camera, and where they hit

Concept layer:

- **Recall** (`pipeline3d/` "From pixels back to rays"): a pixel is a ray from the eye.
  Rasterisation (`rasterise/`) asks "which pixels does this triangle cover?"; ray casting
  asks "which object does this pixel see?". **The same two loops, swapped**: for each object,
  for each pixel vs for each pixel, for each object. Primary visibility is identical (same
  nearest surface); what differs is what else each can ask cheaply (units 2–4).
- **Camera rays**: eye $\mathbf o$, pixel centre $(i + \tfrac12, j + \tfrac12)$ → a point on an
  image plane at distance 1 → direction $\mathbf d$, normalised. Field of view and aspect from
  `pipeline3d/` (by hand with numbers). Orthographic camera = parallel rays (`volrender/`).
- **The ray** $\mathbf r(t) = \mathbf o + t\mathbf d$, $t > 0$: points in front of the camera.
  A hit is a value of $t$; the **nearest hit** is the smallest positive $t$ over all objects
  (ray casting's z-buffer).
- **Two ways to describe a surface**: implicit, $f(\mathbf x) = 0$ (sphere, plane: substitute
  the ray, solve for $t$) or parametric (triangle: $\mathbf a + u(\mathbf b - \mathbf a) +
  v(\mathbf c - \mathbf a)$: set equal to the ray, solve for $t, u, v$). Pointer: the level-set
  day's implicit curves (`implicit/`) are the same idea.
- **Sphere**: substituting gives a quadratic in $t$; the discriminant says miss / graze / two
  hits; take the smaller positive root (the larger if the eye is inside, e.g. inside glass).
  Picture: a line through a circle.
- **Plane**: one dot product and one division; $\mathbf d\cdot\mathbf n = 0$ = parallel.
- **Triangle**: the ray meets the triangle's plane at one point; inside iff its three
  barycentric coordinates are $\ge 0$ (recall `rasterise/`: weights from areas, summing to 1).
  The same barycentrics interpolate normals and colours at the hit.
- **Normals and shading at the hit**: sphere normal $(\mathbf p - \mathbf c)/r$, plane normal,
  interpolated vertex normals for triangles; Lambert and Blinn–Phong recalled from
  `rasterise/` in one slide (no new shading model here; physically based light is unit 4).
- **Cost**: every pixel tests every object: a 1920×1080 image of a $10^6$-triangle mesh is
  $2\times10^{12}$ tests per frame. Unit 3 fixes this.
- Pointer (one line): **sphere tracing** of signed distance fields (level sets, `voxelise/`)
  is a third way to find the hit, by marching like `volrender/`.

The math (each on its own `{.math}` slide after its concept slide):

- *The math: the sphere quadratic.* $\lVert\mathbf o + t\mathbf d - \mathbf c\rVert^2 = r^2$ →
  $t^2 + 2t\,\mathbf d\cdot(\mathbf o - \mathbf c) + \lVert\mathbf o - \mathbf c\rVert^2 - r^2
  = 0$ for unit $\mathbf d$; with $b' = \mathbf d\cdot(\mathbf o - \mathbf c)$,
  $t = -b' \pm \sqrt{b'^2 - (\lVert\mathbf o - \mathbf c\rVert^2 - r^2)}$. **Stable form**: when
  $b'^2 \gg$ the constant, one root suffers cancellation (recall `floatingpoint/`
  "Cancellation"); compute the large-magnitude root first, the other as $c/(t_1)$ (Vieta).
- *The math: the plane.* $(\mathbf o + t\mathbf d - \mathbf p_0)\cdot\mathbf n = 0 \Rightarrow t
  = (\mathbf p_0 - \mathbf o)\cdot\mathbf n / (\mathbf d\cdot\mathbf n)$.
- *The math: ray–triangle as a $3\times3$ system* (Möller–Trumbore 1997):
  $\begin{bmatrix}-\mathbf d & \mathbf e_1 & \mathbf e_2\end{bmatrix}
  \begin{bmatrix}t\\u\\v\end{bmatrix} = \mathbf o - \mathbf a$ with $\mathbf e_1 = \mathbf b -
  \mathbf a$, $\mathbf e_2 = \mathbf c - \mathbf a$; Cramer's rule with scalar triple products
  (determinants = signed volumes, recall `linalg` / `rasterise/`'s areas); $(b_a, b_b, b_c) =
  (1-u-v, u, v)$; determinant $\approx 0$ = ray parallel to the plane; its sign = which side
  was hit (back-face test).
- *The math: ray-cast barycentrics are perspective-correct.* The ray hits the 3D point, so
  $(u, v)$ are barycentrics in 3D, which is exactly what `rasterise/`'s $1/w$ correction
  recovers; ray casting never needs the correction.

Demo and exercises:

- **Demo `raystep`** (new, canvas + SVG, CPU, `lib/raytrace.ts`): left (≈ 600 px), the image on
  a coarse pixel grid (16×12, 32×24, 64×48) filling in as rays are traced; right (≈ 480 px), a
  **top view** (x–z plane) of the scene: camera, image plane with the pixel row, objects
  (2–3 spheres, a ground plane, one triangle; drawn as their x–z cross-sections/footprints),
  and the current ray with all its candidate hits marked. Buttons *next pixel*, *next row*,
  *all*, *reset*; click a pixel to inspect it. Readout per object for the current ray: the
  quadratic's discriminant and roots (sphere), $t$ (plane), $t, u, v$ and inside/outside
  (triangle); nearest hit, normal, Lambert term. Counters: rays, intersection tests. Views:
  shaded / depth $t$ / normal as RGB / object id / barycentrics as RGB (triangle). Controls:
  field of view, camera position (drag in the top view), sphere radius. `data-depth` (0 here;
  units 2 uses 1–6). Presets: "one sphere", "a ray that misses (negative discriminant)", "a
  sphere behind the camera ($t < 0$)", "eye inside the sphere", "the triangle and its
  barycentrics", "orthographic vs perspective", "count the tests".
- By hand (concept): ray for pixel (1, 0) of a 4×2 image, field of view 90°, eye at the
  origin; does the ray $\mathbf o = \mathbf 0$, $\mathbf d = (0, 0, -1)$ hit the sphere centred
  $(0.5, 0, -5)$ with $r = 1$, and where; normal at the hit; same ray against the plane
  $y = -1$ (parallel: why no hit) and $\mathbf d = (0, -0.6, -0.8)$ against it; which of three
  given points of a triangle's plane are inside (barycentrics given as numbers); how many tests
  for a 640×480 image of 5000 triangles.
- *(math)*: derive the sphere quadratic and show when cancellation hurts (numbers: $b' =
  10^4$, constant $= 1$ in float32); derive the triangle system and solve it by Cramer's rule;
  show the ray-cast barycentrics equal `rasterise/`'s perspective-correct ones; ray–cylinder
  (infinite, axis $z$) as a quadratic.

### 2. `whitted/`: Shadows, mirrors and glass: recursive ray tracing

Concept layer:

- **Shadow rays**: from the hit, one ray to each light; if anything is hit before the light,
  the point is in shadow (only the light's direct term drops). Rasterisation needs a separate
  pass for this (shadow maps, one line); a ray tracer just asks.
- **Shadow acne**: the shadow ray starts *on* the surface and, in floating point, hits it
  again at $t \approx 10^{-7}$; dark speckles. Fix: start at $\mathbf p + \varepsilon\mathbf n$
  or ignore $t < \varepsilon$; too large an $\varepsilon$ lets light leak ("Peter Panning").
  Recall `floatingpoint/` (spacing of floats grows with magnitude, so a fixed $\varepsilon$
  fails far from the origin).
- **Mirror reflection**: $\mathbf r = \mathbf d - 2(\mathbf d\cdot\mathbf n)\mathbf n$ (the same
  formula as `rasterise/`'s Phong reflection, with the incoming direction): trace the reflected
  ray and add what it sees, times a reflectance.
- **Refraction** (taught here from scratch): light slows down in glass/water by the factor
  $n$ (refractive index); at an interface it bends: **Snell's law** $n_1\sin\theta_1 =
  n_2\sin\theta_2$. Picture: a wavefront entering slower medium (marching soldiers). Towards
  the normal when entering glass, away when leaving.
- **Total internal reflection**: from glass to air beyond the critical angle
  $\sin\theta_c = n_2/n_1$ (41.8° for $n_1 = 1.5$) there is no refracted ray: all light
  reflects (light guides, fibres, the silver look of a water surface seen from below; TIRF
  microscopy, one line).
- **Fresnel**: at a real interface light **splits**: a fraction $R$ reflects, $1 - R$ refracts;
  $R$ is small head-on (4 % for glass) and goes to 1 at grazing angles (why a lake is a mirror
  at sunset, why glass edges look bright). Schlick's approximation as the practical formula.
- **Whitted recursion** (1980): at each hit, colour = local shading (with shadow rays) +
  $R\cdot$trace(reflected) + $(1 - R)\cdot$trace(refracted). The **ray tree**: each glass hit
  doubles the rays; a **depth limit** (and a cut-off when the weight is tiny) stops it;
  depth 0 = unit 1's ray casting. Picture of the tree for one pixel through a glass ball.
- What Whitted ray tracing **cannot** do: soft shadows, light bouncing off a diffuse wall
  onto another (colour bleeding), caustics (light focused *through* glass onto a diffuse
  floor): it only follows the mirror and refraction directions and point lights. Unit 4.
- **Rays in 2D**: everything above happens in the plane of incidence, so a 2D sandbox shows it
  exactly: the `ray2d` demo; two presets go to **lenses** (a ball focuses parallel light, but
  not to one point: spherical aberration; a coverslip/oil vs water mismatch shifts the focus):
  the bridge to optional unit 5 and to `psf/`.

The math:

- *The math: reflection.* Decompose $\mathbf d = \mathbf d_\parallel + \mathbf d_\perp$ along
  $\mathbf n$; flip the normal part; $\lVert\mathbf r\rVert = \lVert\mathbf d\rVert$.
- *The math: Snell's law in vector form.* With $\eta = n_1/n_2$, $c_1 = -\mathbf d\cdot\mathbf n$:
  $\mathbf t = \eta\mathbf d + (\eta c_1 - \sqrt{1 - \eta^2(1 - c_1^2)})\,\mathbf n$; the square
  root's argument $< 0$ is total internal reflection. Derivation from conserving the
  tangential component scaled by $\eta$ and $\lVert\mathbf t\rVert = 1$. Fermat's principle
  (least time) as the reason, one line.
- *The math: Fresnel.* The two polarisation reflectances $R_s, R_p$ (stated, not derived: that
  needs Maxwell's equations), unpolarised $R = (R_s + R_p)/2$; head-on
  $R_0 = \big((n_1 - n_2)/(n_1 + n_2)\big)^2$; Schlick $R(\theta) \approx R_0 + (1 - R_0)(1 -
  \cos\theta)^5$. Brewster's angle ($R_p = 0$) as a remark.
- *The math: the size of the ray tree.* Full binary tree of depth $D$: $2^{D+1} - 1$ rays per
  pixel plus shadow rays; a weight cut-off makes it much smaller in practice.

Demos and exercises:

- **Demo `raystep`** with `data-depth="3"` (same demo as unit 1): toggles *shadows*,
  *reflection*, *refraction*, *Fresnel*, $\varepsilon$ slider (0 shows acne), depth slider
  0–6; the inspected pixel shows its **ray tree** in the top view (colour by type: primary,
  shadow, reflected, refracted) and as an indented list with weights. Counter: rays per
  pixel. Presets: "shadow acne ($\varepsilon = 0$)", "light leaks (huge $\varepsilon$)",
  "mirror ball", "two facing mirrors (depth matters)", "glass ball: refraction and total
  internal reflection", "depth 0 vs 5", "Fresnel on/off on a glass floor".
- **Demo `ray2d`** (new, SVG, CPU, `lib/rays2d.ts`): a 2D ray/reflection/refraction sandbox
  (≈ 1100×540). A source (single ray / point source / parallel beam, draggable, number of
  rays slider), and objects to drag, rotate and set: mirror segment, flat interface, glass
  slab, ball (circle), biconvex lens, prism, half-disc, a "coverslip + sample" stack. Each
  medium has a refractive index slider (1.0–2.4, with labelled presets air, water 1.33, glass
  1.52, immersion oil 1.518, diamond 2.42). Rays are traced with reflection and refraction;
  with *Fresnel* on, every split is drawn with opacity ∝ carried power (and a power readout),
  off = refraction only. Click a hit: incidence and refraction angles, $n_1\sin\theta_1$ vs
  $n_2\sin\theta_2$, $R$. Toggle *dispersion* (index depends on wavelength by a two-term
  Cauchy formula; three colours). Depth cap 12, power cut-off 0.1 %. Presets: "Snell at a flat
  surface", "the critical angle", "a light guide", "Fresnel: glass at grazing angles is a
  mirror", "a ball focuses (spherical aberration)", "a thin lens: parallel beam to a focus",
  "prism: dispersion", "oil vs water under a coverslip: the focus moves". Labels ≥ 18 px on
  slides.
- By hand (concept): reflect $\mathbf d = (1, -1, 0)/\sqrt2$ in a floor with $\mathbf n = (0, 1,
  0)$; air → glass ($1.5$) at 30°: refraction angle (19.5°); critical angle for glass → air and
  for water → air; $R_0$ for air–glass and water–glass (why a glass bead in water is nearly
  invisible); Schlick at 60° and 85°; draw the ray tree for a pixel that sees a glass ball in
  front of a mirror, depth 3; count rays; predict what happens to a shadow behind a glass
  ball in Whitted ray tracing (it is black: no caustics).
- *(math)*: derive the vector reflection formula and show it preserves length; derive the
  vector form of Snell's law and find the TIR condition; show $R_0 \to 0$ when $n_1 = n_2$ and
  why that explains index matching; size of the ray tree with and without a weight cut-off
  (a geometric series when each glass hit keeps $\le 0.96$ of the weight in one branch).

### 3. `bvh/`: Tracing fast: acceleration structures, hardware, and rays vs rasterisation

Concept layer:

- **The problem**: O(pixels × objects × rays per pixel). Most tests miss. Idea: test a cheap
  **bounding volume** first and skip everything inside it if the ray misses.
- **The slab test** for an axis-aligned box: a box is the intersection of three slabs;
  intersect the ray's $t$-intervals; empty = miss. This is `volrender/`'s `rayBox` (used
  there, explained here).
- **Bounding-volume hierarchy**: boxes inside boxes, a binary tree over the objects; traversal
  visits a child only if the ray hits its box, nearest child first, and stops a branch when
  its box is farther than the best hit so far. Building: split the objects at the median of
  the longest axis (simple) or by the **surface area heuristic** (cost ∝ probability a random
  ray hits a box ∝ its surface area × objects inside). A good tree gives about O(log objects)
  tests per ray; a bad one (all objects in one leaf, very overlapping boxes) gives O(objects).
- **Uniform grids**: cut space into cells, list objects per cell, walk the ray through the
  cells in order (3D DDA, the line-drawing idea of `rasterise/`/Bresenham); stop at the first
  cell with a hit. Fast for evenly spread scenes, bad for "a teapot in a stadium" (most cells
  empty, a few crowded). Octrees and k-d trees named as the adaptive versions.
- **Object split vs space split**: a BVH partitions objects (boxes may overlap, each object
  once); a grid/k-d tree partitions space (no overlap, objects may be listed in several cells).
- **Dynamic scenes**: rebuild vs refit the boxes (one line).
- **Tracing a real mesh**: the marching-cubes surface of the `nucleiStack` volume (marching
  cubes day, `lib/marching.ts`): ~10⁵ triangles, brute force vs BVH tests per ray.
- **Rasterisation vs ray tracing**, one comparison slide: loops (objects outer vs pixels
  outer); memory access (one triangle at a time, streaming vs random access into the whole
  scene); primary visibility (same result; rasterisation is cheaper per pixel); shadows,
  reflections, refraction, global light (natural for rays; tricks for rasterisation: shadow
  maps, environment maps, screen-space reflections); cost growth (rasterisation ~linear in
  triangles; ray tracing ~logarithmic after a linear-time build). **Hybrid**: real-time
  engines rasterise primary visibility and trace secondary rays.
- **Hardware ray tracing**: GPUs now have fixed-function units for BVH box tests and
  ray–triangle tests; the programmer supplies the scene and what happens at a hit. Rays that
  bounce in different directions access memory incoherently and diverge (recall the GPU day's
  warps and divergence if taken; one line otherwise). Real time = few samples per pixel plus a
  denoiser (unit 4).

The math:

- *The math: the slab test.* Per axis $t_{k,\pm} = (\text{min}_k / \text{max}_k - o_k)/d_k$;
  $t_\text{near} = \max_k \min(t_{k,-}, t_{k,+})$, $t_\text{far} = \min_k \max(\cdot)$; hit iff
  $t_\text{near} \le t_\text{far}$ and $t_\text{far} > 0$; $d_k = 0$ handled by IEEE infinities
  (recall `floatingpoint/` "Special values").
- *The math: the surface area heuristic.* For a convex box $B'$ inside $B$, the probability that
  a random ray hitting $B$ also hits $B'$ is $A(B')/A(B)$ (Cauchy's formula, stated); split
  cost $C = C_\text{trav} + \frac{A_L}{A}N_L C_\text{isect} + \frac{A_R}{A}N_R C_\text{isect}$;
  choose the split minimising $C$; stop splitting when $C \ge N C_\text{isect}$.
- *The math: why log.* A balanced binary tree over $N$ objects has depth $\lceil\log_2 N\rceil$;
  a ray that hits few leaves visits O(log N) nodes (an idealisation; overlap and long rays add
  to it).

Demo and exercises:

- **Demo `bvhstep`** (new, SVG + canvas, CPU, `lib/bvh.ts`): a 2D scene (the idea is
  dimension-free and readable in 2D) of N discs and segments (N slider 8–5000, seeded;
  layouts *uniform*, *clustered*, *teapot in a stadium*, *one long wall*). Structure: *none* /
  *BVH (median)* / *BVH (SAH)* / *uniform grid* (cells slider). Draw the boxes (a depth slider
  shows level by level) or the grid. Drag a ray; visited nodes/cells highlighted in traversal
  order (*step* / *all* / *reset*), tested primitives outlined, the hit marked. Counters:
  box tests, primitive tests, vs brute force. A small log-x plot (`lib/plot.ts`): average
  tests per ray against N for brute force, grid and BVH (1000 random rays per point, computed
  on change), with O(N) and O(log N) reference lines. A tab *3D mesh*: the `nucleiStack`
  marching-cubes mesh in `gl3d.ts` with its BVH boxes at a chosen level and the counters for a
  ray picked with `pointerRay`. Presets: "brute force", "median BVH", "SAH vs median on a
  clustered scene", "grid on a uniform scene", "teapot in a stadium: the grid fails", "a ray
  that misses everything costs 1 test", "near-first traversal stops early".
- By hand (concept): slab test for the box $[0, 2]\times[0, 1]$ and the ray $\mathbf o = (-1,
  0.5)$, $\mathbf d = (1, 0.25)$ (2D); build a median BVH over 8 given 1D intervals and trace a
  query point/ray, counting tests vs brute force; SAH cost of two candidate splits with given
  areas and counts; which cells a ray visits in a 4×4 grid; for 1920×1080 pixels, 4 rays per
  pixel and $10^6$ triangles: tests with brute force vs ~$2\log_2 N$ per ray; one
  rasterisation-vs-ray-tracing judgement per scenario (CAD wireframe, a mirror room, a
  CT isosurface for measuring, a game at 60 frames per second).
- *(math)*: derive the slab test and show it handles $d_k = 0$ with infinities (and where NaN
  appears: $0\cdot\infty$ when the origin lies on a slab plane); show the SAH probability for
  nested 2D boxes by a perimeter argument (Cauchy–Crofton in 2D: random lines hit a convex set
  with probability ∝ perimeter) on a numerical example; worst case of BVH traversal.

### 4. `pathtrace/`: Path tracing: the rendering equation, Monte Carlo and noise

Concept layer:

- **What Whitted misses** (unit 2's list): light arrives at a point from **every** direction,
  not only from lights and mirrors. Picture: a white wall next to a red wall turns pinkish
  (colour bleeding); soft shadows from a big lamp; caustics.
- **Radiance** $L$ (light per direction; constant along a ray in empty space) and the **BRDF**
  $f_r$ (how much light from one direction a surface sends into another; Lambertian = the
  same in all directions, $\rho/\pi$; a mirror = all in one direction).
- **The rendering equation** (Kajiya 1986) in words: light leaving a point towards the eye =
  light it emits + the sum over all incoming directions of (incoming light × BRDF × cosine).
  The incoming light is itself the outgoing light of another point: an integral equation, with
  the whole scene inside it. Recall: `volrender/`'s emission–absorption integral is the same
  bookkeeping along a ray, without bouncing.
- **Monte Carlo**: estimate an integral by averaging the integrand at random points (taught
  here from scratch, with a 1D example: area under a curve by random samples; π by darts).
  Unbiased: right on average. Its error is a **standard deviation that falls as
  $1/\sqrt N$**: 4× the samples halve the noise; 100× give one more decimal digit. The same law
  as averaging beads (`resolution/`) and as shot noise (`cameranoise/`): a path tracer is a
  simulated camera counting simulated photons.
- **Path tracing**: per pixel, shoot $N$ paths; at each hit pick **one** random new direction,
  continue, and multiply the throughput by BRDF × cosine / pdf; stop at a light or after a
  depth / by **Russian roulette**; average. One path per sample (not a tree, as in Whitted):
  the work goes into more samples instead of more branches.
- **Progressive rendering**: keep a running mean per pixel; the image sharpens while you
  watch; precision of a long float32 sum recalled from `floatingpoint/`.
- **Importance sampling**: choose directions where the integrand is large, and divide by the
  pdf so the average stays right. Cosine-weighted directions for diffuse surfaces (the cosine
  cancels); **sampling the lights directly** (next-event estimation) for small lights: without
  it, a tiny lamp is found by chance and gives rare, huge samples: **fireflies**. A pdf that is
  zero where light exists gives a wrong (biased) image, not just a noisy one. Multiple
  importance sampling named.
- **Denoising**: at 16 samples per pixel the image is noisy; reaching the noise of 1024
  samples by brute force costs 64× the time. Instead denoise: recall the deconvolution day's
  denoisers (`denoise/`: Gaussian, bilateral, non-local means). Renderer bonus: the tracer
  knows each pixel's normal, depth and albedo without noise, so a **joint (cross) bilateral**
  filter can smooth within a surface and stop at geometric edges. Recall `denoise/` "A
  restored image is an estimate": the denoised image has less variance but is **biased**
  (blurred shadows, lost texture); learned denoisers (modern deep learning day) are what
  real-time ray tracing uses (pointer).
- **Where this connects**: Monte Carlo simulation of light in tissue (photon transport with
  scattering, used to model light in thick samples) is this method with a participating
  medium (one line); NeRF-style view synthesis renders with `volrender/`'s compositing
  (pointer to the geometry-from-images day).

The math:

- *The math: the rendering equation.*
  $L_o(\mathbf x, \boldsymbol\omega_o) = L_e(\mathbf x, \boldsymbol\omega_o) + \int_\Omega
  f_r(\mathbf x, \boldsymbol\omega_i, \boldsymbol\omega_o)\,L_i(\mathbf x, \boldsymbol\omega_i)
  \cos\theta_i\,d\boldsymbol\omega_i$, with $L_i(\mathbf x, \boldsymbol\omega) = L_o(\text{hit}(\mathbf x,
  \boldsymbol\omega), -\boldsymbol\omega)$. Expanding it gives a sum over path lengths (a Neumann
  series): direct light, one bounce, two bounces, …
- *The math: Monte Carlo is unbiased, with variance $\sigma^2/N$.* $\hat I = \frac1N\sum f(X_k)/p(X_k)$,
  $X_k \sim p$: $\mathbb E[\hat I] = \int f$ (one line), $\mathrm{Var}[\hat I] = \frac1N
  \mathrm{Var}[f(X)/p(X)]$ (independent samples), so RMS error $\propto 1/\sqrt N$; in PSNR
  (`cameranoise/`): +3 dB per doubling of $N$. Contrast with `mcmc/`: there the samples are
  correlated, which inflates the variance.
- *The math: the Lambertian BRDF and cosine sampling.* $\int_\Omega\cos\theta\,d\omega = \pi$, so
  energy conservation gives $f_r = \rho/\pi$; with $p(\omega) = \cos\theta/\pi$ the estimator is
  $\rho\,L_i$ (the cosine and π cancel); sampling it: a uniform point on the unit disc lifted
  to the hemisphere (Malley's method).
- *The math: zero-variance sampling and Russian roulette.* $p \propto f$ makes $f/p$ constant:
  zero variance (needs the answer; motivates "close to $f$"); continuing a path with
  probability $q$ and dividing by $q$ keeps the estimate unbiased.
- *The math: light sampling.* Change of variables from solid angle to light area:
  $d\omega = \cos\theta_\ell\,dA / \lVert\mathbf x - \mathbf y\rVert^2$, plus a shadow ray.

Demos and exercises:

- **Demo `mcpixel`** (new, small, SVG, `lib/montecarlo.ts` + `lib/plot.ts`): one pixel's
  integral in 2D: a surface point with a half-circle of incoming directions; an incoming-light
  profile $L_i(\theta)$ (presets: overcast sky, a window, a tiny bright lamp, two lamps)
  times $\cos\theta$, drawn as a curve. Draw $N$ samples (slider 1–4096) from a chosen pdf
  (*uniform*, *cosine*, *towards the lamp*, *a bad pdf that misses the lamp*); samples shown as
  rays with their weighted values, the estimate vs the true integral. *Repeat 500×*: a
  histogram of the estimates (`lib/histogram.ts`) and a log–log plot of RMS error vs $N$
  with the $1/\sqrt N$ line. Presets: "uniform, 16 samples", "4× samples halves the error",
  "cosine sampling", "tiny lamp: fireflies", "sample the lamp", "a pdf that misses the lamp:
  biased".
- **Demo `pathtracer`** (new, WebGL2 via `lib/glpathtrace.ts`; CPU fallback in TypeScript from
  `lib/raytrace.ts`): a closed-box scene (white floor/ceiling/back, red and green side walls,
  an area light in the ceiling) with a diffuse sphere, a mirror sphere and a glass sphere
  (switchable); drag to orbit (`OrbitCamera`); about 512×384 on slides. Progressive: *play /
  pause / reset*, samples per pixel shown live and a **target samples-per-pixel slider**
  (1, 4, 16, … 4096, stops there). Controls: max bounces (1 = direct light only, 2 = one
  bounce, …, 8), Russian roulette on/off, sampling *uniform hemisphere / cosine-weighted /
  cosine + light sampling*, light size slider (smaller = more fireflies without light
  sampling), *Whitted mode* (unit 2's model on the same scene, for comparison). Views: image /
  per-pixel standard deviation / error against a reference / normal / albedo. *Compute
  reference* renders 16k spp in the background (a few seconds on the GPU; cached for the
  session; seeded so everyone gets the same image). Plot: RMS error vs spp, log–log, with the
  $1/\sqrt N$ line. **Denoise** panel (CPU, after a read-back): *none / Gaussian / bilateral /
  non-local means / joint bilateral (normal + albedo guides)*, strength slider, side-by-side
  with the reference, RMSE and PSNR of both (`lib/metrics.ts`). Presets: "1 sample per
  pixel", "16 → 64 → 256: noise halves per 4×", "direct light only vs global light (colour
  bleeding)", "Whitted vs path traced", "uniform vs cosine sampling at the same spp", "tiny
  light: fireflies", "light sampling removes them", "glass ball caustic (slow to converge)",
  "denoised 16 spp vs 1024 spp: noise vs bias".
- By hand (concept): estimate $\int_0^1 x^2\,dx$ from 4 given uniform numbers (and with a
  pdf $p(x) = 2x$, using given samples); how many samples to halve / tenth the noise; a
  16-spp render has PSNR 22 dB against the reference: predict 256 spp (+12 dB); a pixel
  that sees a lamp of area $10^{-4}$ of the hemisphere's: how often does uniform sampling hit
  it in 64 samples, and what does that do to the image; which denoiser keeps the shadow edge
  sharp and why (joint bilateral uses depth/normal edges, the shadow edge has none: it
  blurs); Whitted vs path tracing: which effects in a given photo each can reproduce.
- *(math)*: prove $\mathbb E[f(X)/p(X)] = \int f$ and $\mathrm{Var}[\hat I] = \sigma^2/N$; compute
  $\int_\Omega\cos\theta\,d\omega = \pi$ in spherical coordinates and derive $\rho/\pi$; show
  that cosine-weighted sampling of a Lambertian surface under a constant sky has zero
  variance; show Russian roulette is unbiased and increases variance; derive the solid-angle
  to area change of variables.

### 5. (optional / further reading) `rayoptics/`: Rays through lenses and microscopes

- **A lens is two refractions**: unit 2's Snell at two spherical surfaces. Parallel rays near
  the axis meet at the **focal point**; rays far from the axis focus closer (**spherical
  aberration**); different colours focus at different distances (**chromatic aberration**,
  dispersion from unit 2).
- **Paraxial (small-angle) optics**: $\sin\theta \approx \theta$ makes everything linear: a ray
  is a vector (height, angle) and every element is a **$2\times2$ matrix** (free space, a
  refracting surface, a thin lens); a system is a product of matrices (the linear-algebra
  link, as `pipeline3d/`'s $4\times4$ chain). The **thin-lens equation**
  $1/s + 1/s' = 1/f$ and magnification $-s'/s$ fall out.
- **The microscope**: objective and tube lens (infinity-corrected), magnification as a ratio
  of focal lengths; the cone of accepted rays and NA (recall `psf/`: NA $= n\sin\alpha$);
  conjugate planes (where the pinhole of a confocal sits, one line).
- **Refractive-index mismatch**: focusing an oil objective into a watery sample bends the
  marginal rays, so the focus lies *shallower* than the stage moved (axial scaling roughly
  $n_\text{sample}/n_\text{immersion}$ in the paraxial limit, worse for high NA) and the
  focus smears axially (aberration). Consequence for image analysis: z-distances in a stack
  are wrong unless corrected (pointer to `volumes3d/` anisotropy and `measurements`); the PSF
  widens with depth (`psf/` bullet).
- **Where rays stop**: ray optics says a perfect lens focuses to a point; waves say the Airy
  disc of `psf/`. Rays tell you *where* light goes and the aberrations; diffraction sets the
  size of the spot. Wave-optics PSF computation stays with `psf/`.
- **Demo**: `ray2d` with `data-mode="optics"`: lens presets (thin lens, thick lens,
  objective + tube lens, coverslip and immersion stack) plus an **ABCD panel** that shows the
  matrix product of the selected elements and the paraxial prediction (focus, magnification)
  next to the exactly traced rays; a "paraxial / exact" toggle; NA cone readout.
- By hand: thin lens with $f = 50$ mm, object at 75 mm: image distance and magnification;
  matrix product of "free space $d$, then thin lens $f$" applied to a parallel ray; focal shift
  for a 20 µm nominal depth into water with an oil objective (paraxial estimate); *(math)*:
  derive the refraction matrix from Snell's law in the paraxial limit and the thin-lens matrix
  from two surfaces (lensmaker's equation).

## Shared engineering

Reuse (all exist in `demos/src/lib/`):

- `ui.ts` (`slider`, `button`, `checkbox`, `presets`, `autoMount`, `rng`) and `random.ts`
  (`rng`, `uniform`, `choice`) for every demo; fixed seeds everywhere (also in the GPU path
  tracer: a per-pixel hash of (pixel, sample index, seed), so a recording matches).
- `mat4.ts` (`lookAt`, `perspective`, `invert`, vector helpers `add`, `sub`, `scale3`, `dot`,
  `cross`, `normalize`) and `gl3d.ts` (`OrbitCamera`, `GLView` for the 3D mesh tab of
  `bvhstep`, `pointerRay`, `webgl2Available`, `sphereMesh`, `cubeMesh`).
- `volrender.ts`: `pixelRay` (camera ray from $(PV)^{-1}$) and `rayBox` (the slab test; unit 3
  teaches it). Do not copy them; import, and add the general-box version to `lib/bvh.ts`
  that `rayBox` can then call (owner of `volrender.ts`: 3D graphics day).
- `raster.ts`: `barycentric`, `edge` (for the test that ray-cast and rasterised barycentrics
  agree, and for the *rasterise* comparison in `raystep`).
- `linalg.ts`: `det3`, `cross3`, `solve` (reference triangle intersection by Cramer, test
  oracle for Möller–Trumbore).
- `marching.ts` + `volume.ts` (`nucleiStack`): the real mesh for `bvhstep`'s 3D tab;
  `mesh.ts` (`meshTopology`, `faceNormals`).
- `colour.ts`: `linearToSrgb`, `linearToSrgb8` for display (accumulate linear radiance,
  convert once), `inferno` for the standard-deviation and error views.
- `denoise.ts` (owner: deconvolution day): `gaussianDenoise`, `bilateral`, `nonLocalMeans`
  for the `pathtracer` denoise panel. **Request to the owner**: add `jointBilateral(img,
  guides, …)` (range weights from guide images: normal, albedo, depth) there, not locally.
- `metrics.ts` (`rmse`, `psnr`), `histogram.ts`, `plot.ts` (log–log error plots with
  reference lines; clip ids fixed per build-issues).

New library files owned by this day (names checked against `demos/src/lib/`: no collision):

- **`demos/src/lib/raytrace.ts`** (no DOM): `Ray`, scene types (sphere, plane, triangle,
  triangle mesh via `lib/bvh.ts`, materials diffuse/mirror/glass/emissive), `hitSphere` (stable
  quadratic), `hitPlane`, `hitTriangle` (Möller–Trumbore, returns $t, u, v$), `nearestHit`,
  `reflect`, `refract` (returns null on TIR), `fresnel` (exact unpolarised) and `schlick`,
  `cameraRay(i, j, …)`; an **instrumented Whitted tracer** that returns the ray tree (each
  node: type, origin, direction, $t$, weight) for `raystep`; a small **CPU path tracer**
  (`tracePath`, `renderProgressive(target, spp)`) that is the reference implementation and
  the WebGL fallback, plus auxiliary buffers (normal, albedo, depth) for denoising.
- **`demos/src/lib/rays2d.ts`** (no DOM): 2D primitives (segment, circle, arc-bounded lens,
  polygon prism) with media and refractive indices, a 2D tracer with Fresnel splitting and
  power/depth cut-offs, a two-term Cauchy dispersion model, and paraxial $2\times2$ ray-transfer
  matrices for the optional unit (`ray2d`).
- **`demos/src/lib/bvh.ts`** (no DOM): axis-aligned boxes (`slab` test with counters), BVH
  build (median, binned SAH) over boxes of arbitrary primitives in 2D and 3D, traversal
  (nearest-first, early exit) with an optional trace of visited nodes, refit; uniform grid
  with 2D/3D DDA traversal and visited-cell trace. Used by `bvhstep`, by `raytrace.ts` for
  meshes, and available to the 3D graphics optional `voxelise/` (ray parity) and to the GPU
  day.
- **`demos/src/lib/montecarlo.ts`** (no DOM): estimators with running mean/variance
  (Welford), sampling `uniformHemisphere`, `cosineHemisphere` (Malley), `sampleDisc`,
  1D sampling from a tabulated pdf (inverse CDF), stratified samples, Russian roulette helper;
  the 2D-hemisphere ("half-circle") versions for `mcpixel`. The course's home for Monte
  Carlo, if the ownership proposal below is accepted.
- **`demos/src/lib/glpathtrace.ts`** (WebGL2): a progressive path tracer as one fragment
  program on a full-screen triangle, scene in uniforms (≤ 16 spheres/quads; no mesh BVH on
  the GPU: meshes stay in the CPU tab), ping-pong accumulation into two `RGBA32F` targets
  (needs `EXT_color_buffer_float`; fall back to `RGBA16F`, then to the CPU tracer at
  128×96, with a message), a display pass doing linear → sRGB, aux passes for normal/albedo,
  `readPixels` for the denoise panel and the error plot, context only while visible (as
  `gl3d.ts`), no keyboard. A generic "full-screen pass + ping-pong float targets" helper is
  likely also needed by the GPU day's CPU-vs-GPU filter demo: whichever day builds first puts
  it in its own file (proposed name `lib/glpass.ts`) and the other imports it; coordinate
  before building (open question).

Tests (`demos/test/raytracing-test.ts`, run by `make test`):

- sphere/plane/triangle hits against brute-force sampling along the ray; stable vs naive
  quadratic on a near-tangent / far-away case; Möller–Trumbore against `linalg.solve`;
  ray-cast barycentrics equal `raster.ts`'s perspective-correct ones for a projected triangle;
- `reflect` preserves length; `refract` satisfies Snell and returns null exactly beyond the
  critical angle; `fresnel(0) = R_0`, `fresnel(90°) = 1`, Schlick within 1 % of exact for
  glass at < 60°; energy $R + T = 1$ at every split in `rays2d.ts`;
- the slab test against a sampled reference; BVH and grid return the same nearest hit as
  brute force on 1000 random rays over random scenes (2D and 3D, including the
  marching-cubes mesh); SAH cost never worse than "no split" at the root;
- Monte Carlo: uniform and cosine estimators of known integrals converge, RMS error over 500
  repetitions within 10 % of $\sigma/\sqrt N$; cosine sampling of a Lambertian surface under a
  constant sky has zero variance; Russian roulette leaves the mean unchanged;
- the CPU path tracer at 64×48 vs the GPU path tracer: mean difference within the expected
  Monte Carlo error at 1024 spp (a headless-browser check if available, else a manual check
  listed in the build notes).

Performance: `raystep`, `ray2d`, `mcpixel` and `bvhstep` are small and stay in TypeScript.
The path tracer is on the GPU (one 512×384 sample of a 6-object scene is far below a frame).
The CPU fallback and the 3D mesh tab spread work over animation frames (progressive), so no
single update blocks; measure the CPU fallback in the browser: if one sample at 128×96 takes
more than ~30 ms *and* the fallback matters (WebGL2 float targets missing on a machine the
course uses), move `tracePath` to Rust/Wasm per `demos/README.md`; not expected.

UI and CSS (from `plan/build-issues.md`): prefix every per-demo selector with
`.qb-demo[data-demo="<name>"]` (demo.css loads late); slide text and labels ≥ 18 px, including
readouts and the ray-tree list (the 15 px readouts of `raster` are the pattern *not* to copy);
fit 1150×560; presets + reset in every demo; no keyboard bindings; `data-*` settings
(`data-depth`, `data-mode`, `data-preset`, `data-scene`, `data-tab`); a static figure per demo
in the deck and exercise as the print fallback. Everything synthetic; no images or meshes to
license.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, a sphere and a triangle by hand, `raystep` (depth 0) |
| 10:45–12:15 | Unit 2 lecture, Snell and Fresnel by hand, `ray2d`, `raystep` (ray trees) |
| 13:15–14:45 | Unit 3 lecture, slab test and a BVH by hand, `bvhstep` |
| 15:00–16:30 | Unit 4 lecture, Monte Carlo by hand, `mcpixel`, `pathtracer`; quizzes as wrap-up |

Optional unit 5 (`rayoptics/`) is self-study or replaces part of unit 3's slot for a
microscopy-heavy cohort.

## Open questions

- **Ownership proposals for `plan/coursemap.md`** ("Ownership changes to apply when these days
  are planned"; not applied here, this plan edits no other file):
  (a) **Monte Carlo integration and importance sampling**: this day (`pathtrace/`); Ising's
  `mcmc/` keeps Markov chain sampling and recalls it; (b) **geometric optics** (Snell, thin
  lens, ray-transfer matrices, index-mismatch focal shift): this day (`whitted/`, `rayoptics/`);
  `psf/` keeps wave optics and adds a one-line pointer; (c) **acceleration structures (BVH,
  grids) and ray–primitive intersection**: this day; 3D graphics `voxelise/`'s acceleration
  slide becomes a pointer; (d) **hardware ray tracing**: this day (one slide), GPU day points
  here and owns divergence/coherence.
- Should the lens material be core rather than optional (a microscopy cohort may value
  `rayoptics/` over `bvh/`)? Proposal: keep `bvh/` core (it carries rasterisation vs ray
  tracing and hardware), keep two lens presets in core unit 2, and leave `rayoptics/`
  optional.
- Shared full-screen-pass WebGL2 helper with the GPU day (`lib/glpass.ts`?): agree on the
  name and owner before either build starts.
- `EXT_color_buffer_float` availability on lecture-room machines: test; the CPU fallback is
  slow but correct.
- `jointBilateral` in `lib/denoise.ts` (owner: deconvolution day): confirm.
- Is a learned denoiser worth a pointer demo (reuse modern deep learning's restoration
  network if `learned-restoration/` is ever built), or is the joint bilateral enough?
- Spectral rendering (dispersion in the 3D path tracer) and participating media (fog,
  tissue scattering) are left out of the demos; one line each. Revisit if the
  fluorescence/tissue days want Monte Carlo photon transport.
- Programming exercises (write a 100-line ray tracer) are not in this plan; the language for
  code examples is still open (README). Nothing here depends on them.

## References

- Whitted (1980), *An improved illumination model for shaded display*, CACM.
- Kajiya (1986), *The rendering equation*, SIGGRAPH.
- Cook, Porter & Carpenter (1984), *Distributed ray tracing*, SIGGRAPH.
- Möller & Trumbore (1997), *Fast, minimum storage ray–triangle intersection*, J. Graphics Tools.
- Kay & Kajiya (1986), *Ray tracing complex scenes*, SIGGRAPH (slab test, bounding volumes).
- Goldsmith & Salmon (1987), *Automatic creation of object hierarchies for ray tracing*, IEEE CG&A; MacDonald & Booth (1990), *Heuristics for ray tracing using space subdivision*, The Visual Computer (surface area heuristic).
- Amanatides & Woo (1987), *A fast voxel traversal algorithm for ray tracing*, Eurographics (grid DDA).
- Wald, Boulos & Shirley (2007), *Ray tracing deformable scenes using dynamic bounding volume hierarchies*, ACM TOG (refitting).
- Veach (1997), *Robust Monte Carlo methods for light transport simulation*, PhD thesis, Stanford (importance sampling, multiple importance sampling).
- Pharr, Jakob & Humphreys (2023), *Physically Based Rendering: From Theory to Implementation*, 4th ed., MIT Press (free online).
- Shirley, *Ray Tracing in One Weekend* series (free online; teaching order).
- Schlick (1994), *An inexpensive BRDF model for physically-based rendering*, Computer Graphics Forum (Fresnel approximation).
- Dammertz, Sewtz, Hanika & Lensch (2010), *Edge-avoiding À-trous wavelet transform for fast global illumination filtering*, HPG; Schied et al. (2017), *Spatiotemporal variance-guided filtering*, HPG (feature-guided denoising).
- Hecht (2017), *Optics*, 5th ed., Pearson (Snell, Fresnel, paraxial matrices, aberrations).
- Hell, Reiner, Cremer & Stelzer (1993), *Aberrations in confocal fluorescence microscopy induced by mismatches in refractive index*, J. Microscopy (focal shift).
- Wang, Jacques & Zheng (1995), *MCML — Monte Carlo modeling of light transport in multi-layered tissues*, Comput. Methods Programs Biomed. (pointer).

## Build notes (2026-11-13): deviations from this plan

Built: core units `raycast/`, `whitted/`, `bvh/`, `pathtrace/`; demos `raystep` (data-depth 0 / 1–6),
`ray2d`, `bvhstep`, `mcpixel`, `pathtracer`; libs `lib/raytrace.ts`, `lib/rays2d.ts`, `lib/bvh.ts`,
`lib/montecarlo.ts`, `lib/glpathtrace.ts`; test `demos/test/raytracing-test.ts`; figures by
`tools/figs/raytracing/figs.ts`. Optional `rayoptics/` and the `ray2d` optics mode / ABCD panel not built.

- **Schlick**: the plan's "within 1 % of exact for glass below 60°" is wrong: the absolute error is
  0.019 at 60° (0.070 vs 0.089). Tests check ≤ 0.01 below 45° and ≤ 0.04 at every angle.
- `raystep`: hit points are rounded to float32 (as a GPU would compute o + t d) so that ε = 0 shows
  acne (100 of 1328 floor pixels dark vs 46 real shadow pixels at 64 × 48). Camera dragged in the
  top view keeps looking down −z (no yaw). Leak preset uses a low tile (y = 0.12) that loses its
  shadow completely at ε = 0.2. No "rasterise" comparison mode.
- `ray2d`: objects come from presets (no add/remove menu); presets added: "water objective" next to
  "oil objective, watery sample". Converging-beam source added for the coverslip stack.
- `bvhstep`: the 3D mesh tab (nucleiStack marching-cubes mesh) is not in the demo; the mesh numbers
  (6728 triangles, ≈13.8 tests/ray) are computed by script, and the test file checks BVH vs brute
  force on that mesh. Grid in the plot uses ≈√(N/2) cells per side.
- `pathtracer`: 400 × 300 (not 512 × 384); reference 4096 spp (not 16k: in a software-GL headless
  browser one pass takes ~115 ms, on a real GPU a few ms); `data-refspp`, `data-cpu` options.
  Denoise panel works on the displayed sRGB values; `jointBilateral` is local
  (`demos/src/pathtracer/denoise.ts`) — request to the deconvolution day to move it into `lib/denoise.ts`.
  No `lib/glpass.ts` (GPU day coordination): the full-screen-pass + ping-pong code lives in
  `lib/glpathtrace.ts`. CPU fallback: ~59 ms per sample at 128 × 96, spread over frames by rows (no Wasm).
- CPU and GPU path tracers use the same PCG stream per (pixel, sample, seed); a manual browser check
  gave mean radiance 0.04429 (CPU) vs 0.04439 (GPU) over 4000 pixels at 65 spp.
- Ownership proposals (Monte Carlo, geometric optics, acceleration structures, hardware RT) are not
  applied to coursemap.md (not this builder's file).
