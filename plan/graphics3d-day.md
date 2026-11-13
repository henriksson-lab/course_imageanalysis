# Plan: a day on how 3D graphics work (polygons and voxels)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 48 of 51, Part IX "3D and computing". Hard prerequisites: Marching cubes (day 47). Soft (see also): Colour (day 42), Bit depth and noise (day 3). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part IX (marching cubes → 3D graphics); entry after
Part I (day 4), and after the marching-cubes day inside the group. Edits from the three
reviews applied 2026-11-12: `rasterise/` and `shading/` merged into one unit; the built
`gpuformats/` unit moved here from the bit-depth day as unit 3 (textures, filtering,
mipmaps, premultiplied alpha); "volumes as data" (world ↔ voxel affine, oblique slices)
moved to the shared `volumes3d/` unit (marching-cubes day, unit 1), which `volrender/`
recalls.

**Notation:** transmittance is $\mathcal T$ (plain $T$ is temperature on the Ising day);
graphics uses the OpenGL/WebGL conventions (right-handed view space looking down $-z$, NDC
$z \in [-1, 1]$), forced by WebGL, stated once in unit 1.

## Goal of the day

After the day a student can:

1. describe a **polygon mesh** (vertices, indexed triangle list, normals, winding, manifoldness) and a **volume** (3D array + a voxel-to-world transform), and estimate what each costs in memory;
2. follow a vertex through the **transformation pipeline** model → world → view → clip → NDC → screen with $4\times4$ homogeneous matrices, **derive the perspective projection matrix** from similar triangles, and explain the **perspective divide**, orthographic vs perspective, and the camera **look-at** as a change of basis;
3. **rasterise** a triangle by hand: edge functions, **barycentric coordinates**, the **z-buffer**, back-face culling, and explain why attributes need **perspective-correct** interpolation and why depth precision is uneven;
4. **shade** a surface (Lambert, Phong, Blinn–Phong; flat, Gouraud, per-pixel) and **derive why normals transform with the inverse transpose**;
5. explain how a GPU **stores and samples textures**: texture formats and packed formats, sRGB storage, filtering in hardware (bilinear, **mipmaps** and their memory cost of one third), and **premultiplied alpha** with the compositing math, as the interpolation and sampling of Part 1;
6. view a volume (recalling the world ↔ voxel affine and oblique slices of `volumes3d/`): **maximum / minimum / average intensity projection**, and **direct volume rendering** by ray marching with a **transfer function** and **front-to-back alpha compositing**, derived from the emission–absorption model; contrast with isosurface ray casting and with extracting a mesh.

Why it fits this audience: almost everything is linear algebra they know, used in earnest:
homogeneous coordinates and $4\times4$ matrices, change of basis (the camera), inverse and
transpose (normals), barycentric coordinates (a $2\times2$ / $3\times3$ linear solve, or ratios of
determinants = signed areas), and one first-order ODE (light along a ray). It extends the
existing unit *Images as arrays & geometric transforms* (`images/`) from 2D/$3\times3$ to
3D/$4\times4$ and adds the one genuinely new matrix, the perspective projection.
Image-analysis payoff: every 3D dataset in the course (CT, MRI, confocal stacks, meshes from
marching cubes) is looked at through one of these two pipelines.

## Prerequisite gap

- **2D homogeneous coordinates, affine transforms, inverse mapping, nearest/bilinear
  interpolation** are taught in `images/` (lecture "A matrix moves the coordinates",
  "Computing the output: go backwards", "Adding translation"). Unit 1 recalls them in one
  slide and goes to $4\times4$; unit 4 reuses inverse mapping for reslicing.
- **3D images** (stacks, voxel spacing and anisotropy, world ↔ voxel affine, trilinear
  interpolation, orthogonal and oblique slices, a first MIP) are taught in `volumes3d/`,
  unit 1 of the marching-cubes day, which comes first in the 3D group (decided). Unit 4 here
  opens with a one-slide recall.
- **Colour** is used as RGB triples in $[0, 1]$; colour spaces, gamma and colour maps are
  owned by the colour day (day 42). Bit depth, integer types and floating point are
  owned by the bit-depth day (day 3: `quantisation/`, `cameranoise/`, `highbit/`,
  `floatingpoint/`); stride and endianness stay in `highbit/` and are recalled in unit 3.
- **Sampling and aliasing** (Part 1, Fourier unit) is assumed for the texture-minification
  and ray-step-size points; one-line recall.
- Nothing about GPUs or graphics APIs is assumed. The GPU pipeline is taught conceptually
  (stages and what runs per vertex / per fragment); shaders appear at most as pseudo-math.

## Overlaps with other days (who owns what)

| Topic | Owner | Elsewhere |
|---|---|---|
| Mesh data structure, winding, face normals by cross product | **this day**, unit 1 | marching cubes: recall + pointer |
| 4×4 matrices, camera, projection, perspective divide | **this day**, unit 1 | marching cubes `marchingcubes/`: one-line pointer |
| Rasterisation, z-buffer, culling, clipping | **this day**, unit 2 | — |
| Lambert / Phong / Blinn–Phong, flat vs Gouraud vs per-pixel, inverse-transpose normals | **this day**, unit 2 | marching cubes: recall + pointer |
| Texture formats, hardware filtering, mipmaps and their memory, premultiplied alpha and compositing | **this day**, unit 3 (`gpuformats/`, built, moved from the bit-depth day) | `volrender/` compositing recalls unit 3; integral images day (mipmaps vs summed-area tables): pointer |
| Row stride, endianness, "an image is a row of bytes" | bit-depth day `highbit/` (day 3) | unit 3: recall |
| Normals **from the image gradient** $-\nabla I/\lVert\nabla I\rVert$, anisotropic spacing in the gradient | marching cubes `marchingcubes/` | this day unit 4 (isosurface ray casting) points there |
| Isosurface extraction (marching squares/cubes/tetrahedra), measurement | marching cubes | this day unit 4: pointer |
| 3D images: stacks, spacing, world ↔ voxel affine, trilinear, orthogonal and oblique slices, first MIP | marching cubes `volumes3d/` (unit 1, shared) | this day unit 4: recall; MIP/MinIP/AIP as rendering and DVR are owned here |
| Colour spaces, gamma, sRGB, colour maps | colour day (day 42) | units 3 and 4: pointer |
| Floating point, bit depth | bit-depth day (day 3) | units 2 and 3: pointers (depth precision, float textures) |
| Line integral of attenuation, Beer–Lambert | radon day | unit 4 generalises it to emission–absorption |
| 2D affine, inverse mapping, bilinear | `images/` (day 1) | this day: recall, extend to 3D |

**Decided (review-organisation §2.3–2.4):** the marching-cubes day folds its former
`isosurface-render/` unit into `marchingcubes/` (gradient normals, 2–3 slides) and
`surface-measure/`; generic projection and shading there are a one-slide pointer to units 1
and 2 here. There is one volume renderer (`volview`, `lib/volrender.ts`); the marching-cubes
day embeds it for MIP views instead of adding its own MIP mode.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`pipeline3d/`, `rasterise/` (merged with the former `shading/`), `gpuformats/` (built,
moved here), `volrender/`; optional `voxelise/`.

### 1. `pipeline3d/`: Meshes, coordinate spaces and projection

- Two ways to represent a 3D object: a **surface** (polygons) or a **volume** (voxels). Today: how
  each becomes an image on the screen. This unit and the next: polygons; unit 3: how the GPU
  stores and samples images (textures); unit 4: voxels.
- **Meshes**: a vertex array (positions, plus per-vertex attributes: normal, colour, texture
  coordinate) and an **indexed triangle list** (three indices per face). Triangle soup vs indexed
  (a closed mesh has about twice as many triangles as vertices). Why triangles: always planar,
  always convex, interpolation is unique. **Winding**: the order $a, b, c$ defines the face normal
  $(\mathbf b-\mathbf a)\times(\mathbf c-\mathbf a)$ and which side is "front"; consistent
  orientation = every interior edge used once in each direction. **Manifold**: every edge in
  exactly two faces, every vertex has a disc-shaped neighbourhood; examples of non-manifold
  (three faces on an edge, two cubes touching at a corner) and why tools care (normals,
  inside/outside, the volume formula of the marching-cubes day).
- Recall `images/`: in 2D, $\mathbf x' = A\mathbf x + \mathbf t$ is one $3\times3$ matrix in
  homogeneous coordinates. In 3D the same trick gives $4\times4$ matrices
  $\begin{bmatrix} A & \mathbf t\\ \mathbf 0^\top & 1\end{bmatrix}$; points have $w = 1$,
  directions $w = 0$ (translation does not move a direction).
- **Spaces** and the matrices between them: model → world (**model matrix** $M$: place the object),
  world → view (**view matrix** $V$: the camera), view → clip (**projection** $P$), clip → NDC
  (**divide by $w$**), NDC → screen (**viewport**: scale and shift to pixels). Order of products,
  $P\,V\,M\,\mathbf x$, and why it is one matrix per object per frame.
- **The camera as a change of basis** (look-at): from eye $\mathbf e$, target $\mathbf g$ and up
  $\mathbf u$ build an orthonormal basis $\mathbf f = (\mathbf g - \mathbf e)/\lVert\cdot\rVert$,
  $\mathbf r = \mathbf f\times\mathbf u/\lVert\cdot\rVert$, $\mathbf u' = \mathbf r\times\mathbf f$;
  $V = \begin{bmatrix} R & -R\mathbf e\\ \mathbf 0^\top & 1\end{bmatrix}$ with rows of $R$ =
  $\mathbf r, \mathbf u', -\mathbf f$. It is the inverse of the matrix that places the camera in
  the world; $R^{-1} = R^\top$ because the basis is orthonormal. Orbiting = rotating $\mathbf e$
  about $\mathbf g$.
- **Perspective by similar triangles**: a point $(x, y, z)$ in view space (camera looks down
  $-z$) projects onto the image plane at distance $n$ as $x_s = n x/(-z)$. Division by $z$ is not
  linear, so the matrix puts $-z$ into $w$ and the hardware divides afterwards: the **perspective
  divide**. Depth: we still need a depth for the z-buffer; with a matrix the only option is
  $z_\text{ndc} = A + B/z$; choose $A, B$ so that $z = -n \mapsto -1$ and $z = -f \mapsto +1$.
  Assemble the full matrix with field of view and aspect ratio. Clip space is where
  $-w \le x, y, z \le w$ (the view **frustum** becomes the NDC cube).
- **Orthographic** projection: a scale and shift of a box to the NDC cube, $w$ stays 1; parallel
  lines stay parallel, size does not depend on distance (what microscopy viewers often use, and
  what MIP images usually are). Dolly zoom: changing distance and field of view together keeps
  the subject the same size but changes the perspective.
- Pointer: quaternions / rotation interpolation are not covered (one line: rotation matrices do
  not interpolate linearly).
- Optional slide (review-gaps gap 12), **from pixels back to rays**: the projection matrix run
  backwards gives a ray, not a point; one camera cannot measure depth, two can (stereo,
  epipolar geometry named), which is the inverse problem of computer vision (pointer only).
- **Demo `pipeline3d`** (new, WebGL via `lib/gl3d.ts`): left (≈ 700 px), a "god's eye" view of the
  world with the object, the camera drawn as a pyramid and its view frustum as lines; right
  (≈ 420 px), the selected space: *model*, *world*, *view*, *clip/NDC* (the frustum as a cube,
  object visibly squashed toward the far plane) or *screen* (the final image on a coarse pixel
  grid). Under it the matrices $M$, $V$, $P$ with live numbers (3 decimals). Controls: object
  rotation/scale/translation, camera eye drag (orbit) and distance, field of view, near, far,
  *perspective / orthographic*. Click a vertex: its coordinates in every space in a small table,
  including $w$ before the divide. Presets: "orthographic vs perspective", "dolly zoom",
  "near plane too close", "a vertex behind the camera ($w < 0$)", "non-uniform scale", "the
  same object, three cameras".
- By hand: a $4\times4$ for "rotate 90° about $z$, then translate by $(1, 0, 0)$" vs the reverse
  order; apply both to a point and a direction. Derive $x_s = nx/(-z)$ from a sketch. With
  $n = 1, f = 3$: solve for $A, B$ in $z_\text{ndc} = A + B/z$, then project
  $(1, 1, -2)$ through a given $P$ and divide. Build $V$ for $\mathbf e = (0, 0, 5)$,
  $\mathbf g = \mathbf 0$, $\mathbf u = (0, 1, 0)$ and check $V\mathbf e = \mathbf 0$. Which faces of
  a given cube (indices listed) have the wrong winding?

### 2. `rasterise/`: From triangles to shaded pixels: rasterisation, z-buffer and shading

Merged from the former `rasterise/` and `shading/` (review-organisation §2.3; this plan's
own recommendation): the derivations stay core (edge functions and barycentrics, the
z-buffer, perspective-correct interpolation, the inverse transpose), breadth (clipping,
depth formats, the GPU pipeline) shrinks to one slide each, and texturing moves to unit 3.

- After the viewport transform a triangle is three points in pixel coordinates with a depth.
  **Rasterisation**: which pixel centres lie inside, and with which interpolated values.
- **Edge functions**: $E_{ab}(\mathbf p) = (\mathbf b-\mathbf a)\times(\mathbf p-\mathbf a)$
  (the $z$ component, a $2\times2$ determinant) is twice the signed area of triangle $a, b, \mathbf p$;
  inside = all three edge functions have the same sign. Linear in $\mathbf p$, so it updates by
  one addition per pixel step (Pineda 1988), and every pixel is independent (why GPUs are fast).
  Tie-breaking (top-left rule: shared edges drawn once) in one line.
- **Barycentric coordinates** $(b_a, b_b, b_c)$ (not $\lambda$: reserved for eigenvalues and the
  prior weight) $= $ ratios of edge functions, summing to 1; any vertex attribute
  interpolates as $b_a f_a + b_b f_b + b_c f_c$. **Link to bilinear interpolation**
  (`images/`): both are weights from areas that sum to one (a square split into two triangles
  gives a different interpolant: show the crease).
- **Back-face culling** by the sign of the screen-space area (wrong for inconsistently wound
  meshes, link to unit 1). **The z-buffer** (Catmull 1974): one depth per pixel, draw a
  fragment only if nearer; order-independent for opaque surfaces, transparency breaks it
  (compositing, units 3–4); z-fighting.
- **Depth precision** in one slide: $z_\text{ndc} = A + B/z$ is hyperbolic, so moving $n$
  matters far more than $f$; stored as 24-bit integer or 32-bit float (pointer to day 3,
  `floatingpoint/`).
- **Perspective-correct interpolation**: screen-space linear quantities are $1/w$ and $f/w$,
  so $f = \sum b_i f_i/w_i \,/\, \sum b_i/w_i$. The classic wrong result: a receding
  checkerboard bends along the diagonal. Orthographic needs no correction.
- Clipping against the near plane before the divide, and edge aliasing / multisampling: one
  slide together.
- **Shading**: **Lambert** $k_d\max(0, \mathbf n\cdot\mathbf l)$ (projected area, the tilted
  roof); **Phong** specular with $\mathbf r = 2(\mathbf n\cdot\mathbf l)\mathbf n - \mathbf l$;
  **Blinn–Phong** with the half vector; ambient as a crude stand-in; physically based shading
  as a pointer. **Where** to evaluate: flat, Gouraud (misses highlights inside a triangle),
  per-pixel; vertex normals as area-weighted averages of face normals.
- **Transforming normals**: under a non-uniform scale $M\mathbf n$ is no longer perpendicular;
  from $\mathbf n'^\top M\mathbf t = 0$ the right normal is $M^{-\top}\mathbf n$ (math slide);
  picture of a circle squashed into an ellipse.
- **The GPU pipeline, conceptually** (one diagram): vertex stage ($PVM\mathbf x$, normals) →
  clipping, divide, viewport, culling, rasterisation → fragment stage (shading, texture reads,
  unit 3) → depth test and blending. No shader code.
- **Demo `raster`** (new, canvas, CPU software rasteriser in `lib/raster.ts`): a coarse pixel
  grid (16×12, 32×24, 64×48) with two overlapping triangles whose vertices are draggable, each
  with a depth slider. *Next pixel*, *next row*, *all*, *reset*: the current pixel shows its
  edge-function values, barycentric weights (numbers and RGB colour), interpolated depth and
  the depth test. Views: colour buffer, depth buffer, barycentrics, edge function of edge ab.
  Toggles: back-face culling, top-left rule, perspective-correct. Presets: "one triangle",
  "two triangles intersect", "a shared edge", "flipped winding is culled", "affine vs
  perspective-correct". (The former *depth precision* tab becomes one preset "near plane
  0.01 vs 1" with the $z_\text{ndc}(z)$ curve.)
- **Demo `shading`** (new, WebGL via `lib/gl3d.ts`): icosphere (3 levels), torus, low-poly
  crease blob; drag to rotate, light direction on a sphere gizmo. Controls: flat / Gouraud /
  per-pixel, Lambert / Phong / Blinn–Phong, $k_a, k_d, k_s$, shininess; show normals;
  non-uniform scale with normals by $M$ / by $M^{-\top}$. Presets: "Gouraud misses the
  highlight", "facets vs smooth", "squashed sphere: wrong normals", "Phong vs Blinn–Phong at
  grazing light". (Its texture tab is used in unit 3.)
- By hand: for $\mathbf a = (1, 1)$, $\mathbf b = (7, 2)$, $\mathbf c = (3, 6)$, decide for 4
  pixel centres whether they are inside, and compute barycentrics and depth for one. Fill a 4×4
  z-buffer by drawing two triangles in both orders. Perspective-correct midpoint of an edge
  from $w = 1$ ($u = 0$) to $w = 3$ ($u = 1$). Lambert and Blinn–Phong at a point; for
  $M = \mathrm{diag}(2, 1, 1)$ and the plane $x + y = 1$, transform a tangent and the normal
  both ways and check perpendicularity.

### 3. `gpuformats/`: How GPUs store and sample textures (built; moved from the bit-depth day)

The built unit moves here unchanged in substance (review-organisation §2.2–2.3; course map:
"3D graphics gets `gpuformats/`"). Its first three slides (a row of bytes, stride,
endianness, "where is a pixel?") stay as a short section of `highbit/` on the bit-depth day
(day 3) and become a one-slide recall here. Sidebar: the `gpuformats/` block moves from the
bit-depth part to this day.

- Recall (day 3, `highbit/`): an image is a row of bytes; stride; endianness; integer vs
  normalised vs float storage.
- **Texture formats**: what a stored number means (normalised integers, plain integers, half
  and single floats), **packed formats** (bits shared out, e.g. 10-10-10-2), and the built
  **`texformats`** demo ("the same data in different formats").
- **sRGB storage** (more codes for the dark; the GPU decodes before filtering): storage only
  here, the reason for the curve is the colour day's (day 42), one-line recall.
- **Texture mapping and filtering in hardware**: per-vertex $(u, v)$ interpolated
  perspective-correctly (unit 2), then sampled nearest or **bilinear** (`images/`).
  Magnification blocks or blurs; **minification aliases** (recall `fourier/` sampling).
  Integer formats cannot be filtered (interpolating labels 3 and 7 to 5 would be wrong anyway).
- **Mipmaps** (Williams 1983): a pyramid of pre-filtered half-size levels, level chosen from
  the pixel footprint; trilinear = bilinear in two levels + linear between them; anisotropic
  filtering as a mention. **Memory**: width × height × bytes per texel, and the math slide
  "mipmaps cost a third" ($\sum 4^{-k} = 4/3$). Compressed block formats in one line.
  Pointer: the Gaussian pyramid of the edges day (day 2) is the same idea with a better
  filter; summed-area tables (integral-images day) are the alternative to mipmaps.
- **Premultiplied alpha** and the "over" operator (Porter–Duff), with the math slide
  "compositing and filtering with alpha": filtering straight alpha leaks colour from
  invisible pixels. This is the compositing that `volrender/` (unit 4) uses along a ray.
- **Demos**: `texformats` (built); `shading` with `data-tab="texture"`: a checkerboard / text
  texture on a receding plane with nearest / bilinear / mipmap nearest / trilinear, *colour the
  mip level*, and a zoom on the far region showing moiré. Presets: "minification aliasing",
  "mipmaps fix it".
- By hand: as built (memory of a texture with and without mipmaps; premultiplied average of two
  texels); plus: which mip level for a pixel covering 8×8 texels?

### 4. `volrender/`: Seeing voxels: slices, projections and volume rendering

- **Recall slide (`volumes3d/`, marching-cubes day unit 1)**: a volume is a 3D array plus a
  voxel → world $4\times4$ affine $A$ (spacing $(\Delta x, \Delta y, \Delta z)$ as its diagonal
  special case, what medical headers store); world → voxel is $A^{-1}$; oblique slices are
  inverse mapping + trilinear interpolation; a first MIP. Here the same inverse mapping is
  used per ray sample.
- **Projections along rays**: for each screen pixel a ray through the volume (orthographic: parallel
  rays; perspective: through the eye, built with the unit 1 matrices inverted). Sample at steps
  $\Delta s$ with trilinear interpolation. **MIP** = maximum along the ray (vessels, bright spots,
  standard in microscopy and angiography; loses depth, needs rotation to read); **MinIP** (airways);
  **average / sum** ≈ an X-ray (Beer–Lambert, owned by the radon day; the link back to how CT data are
  acquired).
- **Direct volume rendering**. A **transfer function** maps a value to colour $c(v)$ and opacity
  (extinction) $\tau(v)$, which is where the user's choice "what to see" lives. Gentle physics: along
  the ray, light travelling toward the eye is emitted and absorbed: $\dfrac{dL}{ds} = -\tau L + \tau c$
  (a linear first-order ODE). Solved with $t$ = distance from the eye and the **transmittance**
  $\mathcal T(t) = \exp\!\big(-\int_0^t\tau\big)$ (the fraction of light from depth $t$ that reaches the eye):
  $L = \int_0^D \mathcal T(t)\,\tau(t)\,c(t)\,dt$. What is in front attenuates what is behind.
  Discretise in steps: opacity of a segment $\alpha_i = 1 - e^{-\tau_i\Delta s}$, giving
  **front-to-back compositing** $C \leftarrow C + \mathcal T\,\alpha_i c_i$, $\mathcal T \leftarrow \mathcal T(1-\alpha_i)$
  (Porter–Duff "over", recalled from unit 3's premultiplied alpha). Back-to-front gives the same result (by hand). **Early ray termination**:
  stop when $\mathcal T < \varepsilon$. **Opacity correction** when the step changes:
  $\alpha' = 1-(1-\alpha)^{\Delta s'/\Delta s}$, otherwise the image darkens with more samples.
  Step size vs aliasing (wood-grain artefacts; recall sampling). Pre- vs post-classification
  (interpolate values then apply the transfer function, or the reverse): picture only.
  Shading in DVR with the gradient as normal: pointer to the marching-cubes day
  (`marchingcubes/`, gradient normals). Absorption along a ray is the radon day's
  Beer–Lambert line integral (owner there); emission–absorption generalises it.
- **Isosurfaces two ways**: **ray casting** (march until the sample crosses $c$, refine the
  crossing by linear interpolation between the two samples: the same $t$ as marching squares,
  shade with the gradient normal) vs **extracting a mesh** with marching cubes
  (`plan/marchingcubes-day.md`) and rasterising it (units 1–3). Trade-offs: ray casting is
  per frame and per pixel, needs the volume on the GPU, gives no mesh to measure or export;
  extraction is once per $c$, gives a mesh to measure, but its size grows with the surface.
- **Polygons vs voxels**: memory (a 512³ 16-bit CT is 256 MiB; a mesh of $10^6$ triangles with shared
  vertices is about 24 MB), what each represents (surface only vs interior), what each makes easy
  (rendering, measuring area vs measuring volume, Boolean operations, segmentation).
- **Demo `volview`** (new, WebGL2, one canvas with several viewports): a synthetic 64³ volume with
  anisotropic voxels by default (a 3D head-like phantom of nested ellipsoids with different values,
  a thin bright tube tree for MIP, a dark cavity for MinIP, seeded noise), plus the real stack
  of the marching-cubes plan if it is adopted. Left: three orthogonal slices and an oblique slice
  (drag on the 3D view to tilt the plane; the same `volview` mode is embedded in `volumes3d/`,
  where it is first used). Right: 3D view with mode
  *slice planes / MIP / MinIP / average / DVR / isosurface*, drag to rotate,
  *orthographic / perspective*. Transfer-function editor under a histogram of the volume: draggable
  opacity control points (piecewise linear) and a colour-map choice. Controls: step size,
  *opacity correction* on/off, *early termination* on/off with a *samples per pixel* heat map,
  *interpolation nearest / trilinear*, *spacing correct / ignored* (the volume stretches),
  iso-value $c$ for the isosurface mode. Shows the voxel → world matrix. Presets: "MIP finds the
  vessels", "MIP has no depth: rotate", "average = an X-ray", "a transfer function that shows
  only the skull", "semi-transparent soft tissue", "step too large: wood grain", "no opacity
  correction", "early termination saves 70 % of samples", "ignored z spacing", "isosurface by
  ray casting".
- **Demo `raycomposite`** (new, small, canvas/SVG, CPU): one ray through a 2D slice of the phantom
  (drag the ray's end points). Below, plotted along the ray: sampled value, $\alpha_i$ and colour
  from the transfer function (the same editor as `volview`), the accumulated colour $C$ and the
  transmittance $\mathcal T$. Buttons *next sample*, *all*, *reset*; toggles *front-to-back / back-to-front*
  (same final colour, different partial sums), *MIP* (running max instead), *early termination*
  (marks where it would stop), step size. Presets: "opaque wall hides what is behind", "fog",
  "two layers", "MIP vs DVR on the same ray". This is the by-hand exercise made interactive.
- By hand (the voxel → world and trilinear exercise is in `volumes3d/`): a ray with 5 samples: values given, transfer function
  given as a small table: compute MIP, average, and front-to-back $C$ and $\mathcal T$ step by step; repeat
  back-to-front and compare; where can early termination stop with $\varepsilon = 0.05$? Compute
  $\alpha$ for $\tau = 0.5$, $\Delta s = 1$ and for $\Delta s = 0.5$ (opacity correction).

### 5. (optional / further reading) `voxelise/`: Between polygons and voxels

- **Voxelising a mesh** (the inverse of marching cubes): surface voxelisation (which voxels does a
  triangle touch) and solid voxelisation by **ray parity** (along each row, count crossings with
  the surface; odd = inside), which needs a closed, manifold mesh (unit 1). 2D first: a polygon
  to pixels by scan-line parity. Conservative vs centre-sample rules; partial-volume coverage as
  a grey value. Round trip mesh → voxels → marching cubes → mesh: what is lost.
- Signed distance fields as a third representation (pointer to the level-set and marching-cubes
  days), and sphere tracing as an aside.
- **Point clouds**: samples on a surface without connectivity (laser scanners, depth cameras,
  single-molecule localisation microscopy); render as splats; to get a mesh or a volume:
  voxelise (histogram into a grid, then the rest of the day applies) or surface reconstruction
  (Poisson, mention only).
- Acceleration: octrees / sparse voxel grids and bounding-volume hierarchies, one slide.
- **Demo**: `voxelise` (new, small): a 2D tab (draggable polygon, pixel grid, scan-line parity
  step-through, coverage as grey) and a 3D tab (a mesh from `lib/mesh.ts` voxelised at a resolution
  slider, shown as voxel cubes via `lib/gl3d.ts`, readouts: voxel count, mesh vs voxel memory,
  volume error), plus a *round trip* toggle that runs marching cubes on the voxelisation. Reuses
  `lib/marching.ts`.

## Shared engineering

- **`demos/src/lib/gl3d.ts` + `mat4.ts`** (**owned by this day**: rendering is its subject;
  first written for the marching-cubes day, which comes first in the 3D group; one renderer for
  the course). Point clouds and height fields are `gl3d.ts` primitives (no separate
  `lib/surface.ts`), so the **3D point-cloud viewer** (colour `colourcloud`, unmixing
  `unmixcloud`, SVM `svmlift`) and the level-set and FFT-registration surfaces use it:
  - target **WebGL2** (decided for all 3D demos; 3D textures, single-channel
    float/`R16F` textures and `texelFetch` need WebGL2, which all current browsers support;
    decide once, see open questions);
  - `mat4.ts`: `lookAt`, `perspective`, `ortho`, `inverse`, `transpose`, `normalMatrix`
    ($M^{-\top}$, 3×3), and a formatter that prints a matrix with fixed decimals for display;
  - a **camera object** exposing $M$, $V$, $P$ separately (not only the product), orbit around a
    target by pointer drag, field of view, near/far, ortho/perspective, used by every 3D demo;
  - shading **modes** as shader variants: flat / Gouraud / per-pixel, Lambert / Phong /
    Blinn–Phong, *normal by $M$ or $M^{-\top}$* (the "wrong" variant for `shading`), normal lines,
    texture with selectable filtering and a "colour by mip level" mode;
  - **line / gizmo drawing** (frustum, camera pyramid, axes, light direction) and **multiple
    viewports in one canvas** (scissor), so `pipeline3d` and `volview` each use one WebGL
    context (the context limit noted in the marching-cubes plan);
  - unchanged from that plan: pointer coordinates via `getBoundingClientRect` (reveal.js scales
    slides), `devicePixelRatio`, context created only while visible (IntersectionObserver),
    static figure if WebGL is unavailable, no keyboard (arrow keys belong to reveal.js).
- **`demos/src/lib/volrender.ts`** (new, **owned by this day**, on top of `gl3d.ts`): upload a `Volume` as a 3D texture,
  slice quad for an arbitrary plane, a ray-marching fragment program with modes MIP / MinIP /
  average / DVR / first-hit isosurface, transfer function as a 256×1 RGBA texture, step size,
  opacity correction, early termination, a samples-per-pixel output. Used by `volview`
  (**owned by this day**; embedded in `volumes3d/` on the marching-cubes day, and the target
  for 3D tabs of FFT registration and radon), so there is one volume renderer.
- **`demos/src/lib/volume.ts`** (owner: marching-cubes day; `voxelToWorld` $4\times4$ and CPU
  oblique reslicing are written there for `volumes3d/`): this day adds a CPU ray marcher for one ray (instrumented: returns every
  sample, $\alpha_i$, $C$, $\mathcal T$), and the phantom generator (nested ellipsoids, tube tree,
  cavity, seeded noise).
- **`demos/src/lib/raster.ts`** (new): CPU software rasteriser on a small grid: edge functions,
  barycentrics, top-left rule, z-buffer, perspective-correct vs affine attributes, near-plane
  clipping; an instrumented step mode that records per-pixel values for `raster`. Also a test
  oracle for `gl3d.ts` matrices (project a point both ways).
- **`demos/src/lib/mesh.ts`** (owner: marching-cubes day): add primitives (cube with correct
  winding, icosphere, torus, low-poly crease blob), vertex-normal averaging with crease angle,
  winding/manifold checks, and voxelisation (optional unit).
- **`gpuformats/` code** (built: `texformats` demo) moves with the unit; nothing to rewrite.
  The checkerboard/text test textures go to `lib/synth.ts`; the mipmap pyramid builder (2×2 box
  downsample) sits next to the Gaussian pyramid of the edges day in `lib/imageops.ts`.
- **Tests** (`make test` over `demos/test/*-test.ts`): $P$ maps the near/far corners of
  the frustum to the NDC cube corners; `lookAt` gives an orthonormal $R$ and $V\mathbf e = \mathbf 0$;
  barycentrics sum to 1 and reproduce the vertices; the rasteriser covers each pixel of a quad
  split into two triangles exactly once (top-left rule); perspective-correct interpolation
  matches the 3D intersection point; $M^{-\top}\mathbf n$ stays perpendicular to transformed
  tangents; front-to-back = back-to-front compositing; opacity correction makes the result
  (nearly) independent of step size on a homogeneous slab, matching $1 - e^{-\tau D}$;
  mipmap memory = 4/3 of level 0 (to rounding). (Reslicing tests live with `volumes3d/`.)
- Performance: DVR runs on the GPU (64³–128³ is trivial for WebGL2). Everything on the CPU is
  small (one ray, a 64×48 grid, a 64³ voxelisation) and stays in TypeScript; no Wasm expected.
- Data: synthetic volume and meshes generated in code. Real data: share the marching-cubes
  plan's small stack (PNG slice atlas, credit and licence shown). Follow `demos/README.md`:
  fixed seeds, presets + reset, labels ≥ 18 px, fits 1150×560 on a slide, `data-*` settings
  (e.g. `data-mode`, `data-volume`, `data-mesh`, `data-tab`), static figures as print fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, matrices and the perspective derivation by hand, `pipeline3d` |
| 10:45–12:15 | Unit 2 lecture, rasterise and shade by hand, `raster`, `shading` |
| 13:15–14:45 | Unit 3 lecture (`gpuformats/`), texture memory and alpha by hand, `texformats`, `shading` (texture tab) |
| 15:00–16:30 | Unit 4 lecture, one ray by hand, `raycomposite`, `volview`; quizzes as wrap-up |

## Open questions

- **Decided (course map, review-organisation §2.3–2.4):** "volumes as data" (affine,
  reslicing) moved to the shared `volumes3d/` (marching-cubes day, unit 1); units 2 and 3
  merged into `rasterise/` (derivations kept, breadth cut, as recommended here); `gpuformats/`
  joined as unit 3, so the day has two polygon units, one texture unit and one voxel unit.
  Review-gaps also suggested reducing units 1–3 to one overview unit; not applied, since the
  course map makes this whole day a day.
- **Decided:** the marching-cubes rendering unit is folded into its `marchingcubes/` and
  `surface-measure/` units (no separate "Normals from the image" unit).
- `gl3d.ts`: WebGL2 for all 3D demos. Decided 2026-11-12; the marching-cubes plan is updated.
- **Decided (conventions table):** OpenGL/WebGL conventions throughout (right-handed view
  space looking down $-z$, NDC $z \in [-1, 1]$); $z \in [0, 1]$ named in one line.
- A real mesh (e.g. the Stanford bunny, licence permits research use; check for teaching) or
  procedural meshes only? A real CT/MRI volume with a real header affine would make the
  NIfTI/DICOM slide concrete: which dataset, which licence?
- Point clouds: optional unit only, or a slide in unit 4?
- Programming exercises (e.g. loading a NIfTI volume and making a MIP, or writing a tiny
  rasteriser) are not in this plan; whether to add them, and in which language, is open (see
  README). The demos and by-hand exercises do not depend on it.

## References

- Hughes, van Dam, McGuire, Sklar, Foley, Feiner & Akeley (2013), *Computer Graphics: Principles and Practice*, 3rd ed., Addison-Wesley.
- Marschner & Shirley (2021), *Fundamentals of Computer Graphics*, 5th ed., CRC Press (pipeline, rasterisation, barycentrics, normal transform).
- Akenine-Möller, Haines, Hoffman et al. (2018), *Real-Time Rendering*, 4th ed., CRC Press (GPU pipeline, depth precision, texturing).
- Pineda (1988), *A parallel algorithm for polygon rasterization*, SIGGRAPH (edge functions).
- Catmull (1974), *A subdivision algorithm for computer display of curved surfaces*, PhD thesis, Univ. of Utah (z-buffer, texture mapping).
- Sutherland & Hodgman (1974), *Reentrant polygon clipping*, CACM.
- Heckbert & Moreton (1991), *Interpolation for polygon texture mapping and shading*, in *State of the Art in Computer Graphics* (perspective-correct interpolation).
- Gouraud (1971), *Continuous shading of curved surfaces*, IEEE Trans. Computers; Phong (1975), *Illumination for computer generated pictures*, CACM; Blinn (1977), *Models of light reflection for computer synthesized pictures*, SIGGRAPH.
- Williams (1983), *Pyramidal parametrics*, SIGGRAPH (mipmaps).
- Porter & Duff (1984), *Compositing digital images*, SIGGRAPH.
- Levoy (1988), *Display of surfaces from volume data*, IEEE CG&A; Drebin, Carpenter & Hanrahan (1988), *Volume rendering*, SIGGRAPH.
- Max (1995), *Optical models for direct volume rendering*, IEEE TVCG (emission–absorption).
- Engel, Hadwiger, Kniss, Rezk-Salama & Weiskopf (2006), *Real-Time Volume Graphics*, A K Peters (ray marching on the GPU, transfer functions, opacity correction).
- Kniss, Kindlmann & Hansen (2002), *Multidimensional transfer functions for interactive volume rendering*, IEEE TVCG.
- Kaufman, Cohen & Yagel (1993), *Volume graphics*, IEEE Computer (voxelisation).
- NIfTI-1 data format documentation (`sform`/`qform`), NIfTI DFWG; DICOM PS3.3 Image Plane Module (pixel spacing, orientation, position).
- Lorensen & Cline (1987), *Marching cubes*, SIGGRAPH (pointer; see `plan/marchingcubes-day.md`).
