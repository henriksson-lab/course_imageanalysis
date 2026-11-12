# Plan: a day on how 3D graphics work (polygons and voxels)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. describe a **polygon mesh** (vertices, indexed triangle list, normals, winding, manifoldness) and a **volume** (3D array + a voxel-to-world transform), and estimate what each costs in memory;
2. follow a vertex through the **transformation pipeline** model → world → view → clip → NDC → screen with $4\times4$ homogeneous matrices, **derive the perspective projection matrix** from similar triangles, and explain the **perspective divide**, orthographic vs perspective, and the camera **look-at** as a change of basis;
3. **rasterise** a triangle by hand: edge functions, **barycentric coordinates**, the **z-buffer**, back-face culling, and explain why attributes need **perspective-correct** interpolation and why depth precision is uneven;
4. **shade** a surface (Lambert, Phong, Blinn–Phong; flat, Gouraud, per-pixel), **derive why normals transform with the inverse transpose**, and explain texture mapping and texture filtering (bilinear, mipmaps) as the interpolation and sampling of Part 1;
5. view a volume: orthogonal and **oblique slices** by trilinear interpolation through the **world ↔ voxel affine**, **maximum / minimum / average intensity projection**, and **direct volume rendering** by ray marching with a **transfer function** and **front-to-back alpha compositing**, derived from the emission–absorption model; contrast with isosurface ray casting and with extracting a mesh.

Why it fits this audience: almost everything is linear algebra they know, used in earnest:
homogeneous coordinates and $4\times4$ matrices, change of basis (the camera), inverse and
transpose (normals), barycentric coordinates (a $2\times2$ / $3\times3$ linear solve, or ratios of
determinants = signed areas), and one first-order ODE (light along a ray). It extends the
existing unit *Images as arrays & geometric transforms* (`images/`) from 2D/$3\times3$ to
3D/$4\times4$ and adds the one genuinely new matrix, the perspective projection. Oblique
reslicing is exactly the inverse mapping + interpolation of that unit, one dimension up.
Image-analysis payoff: every 3D dataset in the course (CT, MRI, confocal stacks, meshes from
marching cubes) is looked at through one of these two pipelines.

## Prerequisite gap

- **2D homogeneous coordinates, affine transforms, inverse mapping, nearest/bilinear
  interpolation** are taught in `images/` (lecture "A matrix moves the coordinates",
  "Computing the output: go backwards", "Adding translation"). Unit 1 recalls them in one
  slide and goes to $4\times4$; unit 4 reuses inverse mapping for reslicing.
- **3D images** (stacks, voxel spacing, anisotropy, slice viewing, a first MIP) are
  introduced in `plan/marchingcubes-day.md` unit 1 `isocontour/`. This day owns the general
  treatment (world ↔ voxel affine, oblique reslicing, the projections, DVR); see the overlap
  section. Unit 4 is written so it works without the marching-cubes day: it opens with a
  three-slide "a volume is a 3D array plus a transform".
- **Trilinear interpolation**: one line from bilinear (interpolate in $x$, $y$, then $z$);
  also introduced in the marching-cubes day. Here it is stated as a weighted sum of 8 corners.
- **Colour** is used as RGB triples in $[0, 1]$; colour spaces and gamma are owned by
  `plan/colour-day.md` (in progress), how GPUs store colour, depth and texture formats
  (8-bit unorm, half/single float, 24-bit depth) and floating-point behaviour by
  `plan/bitdepth-day.md` (being written). This day points to both and does not re-teach them.
- **Sampling and aliasing** (Part 1, Fourier unit) is assumed for the texture-minification
  and ray-step-size points; one-line recall.
- Nothing about GPUs or graphics APIs is assumed. The GPU pipeline is taught conceptually
  (stages and what runs per vertex / per fragment); shaders appear at most as pseudo-math.

## Overlaps with other days (who owns what)

| Topic | Owner | Elsewhere |
|---|---|---|
| Mesh data structure, winding, face normals by cross product | **this day**, unit 1 | marching cubes: recall + pointer |
| 4×4 matrices, camera, projection, perspective divide | **this day**, unit 1 | marching cubes unit 3: one-line pointer |
| Rasterisation, z-buffer, culling, clipping | **this day**, unit 2 | — |
| Lambert / Phong / Blinn–Phong, flat vs Gouraud vs per-pixel, inverse-transpose normals | **this day**, unit 3 | marching cubes unit 3: recall + pointer |
| Normals **from the image gradient** $-\nabla I/\lVert\nabla I\rVert$, anisotropic spacing in the gradient | marching cubes unit 3 | this day unit 4 (isosurface ray casting) points there |
| Isosurface extraction (marching squares/cubes/tetrahedra), measurement | marching cubes | this day unit 4: pointer |
| 3D images: stacks, spacing, first look (slices, MIP) | marching cubes unit 1 (intro) | this day unit 4 (full: affine, oblique, MIP/MinIP/AIP, DVR) |
| Colour spaces, gamma, sRGB textures | colour day | this day unit 3: pointer |
| GPU colour/depth/texture formats, floats, depth-buffer bit depth | bitdepth day | this day units 2 and 3: pointers |
| 2D affine, inverse mapping, bilinear | `images/` | this day: recall, extend to 3D |

**Proposal for the marching-cubes rendering unit (`isosurface-render/`).** With this day in
place, its generic part (projection as a $4\times4$ matrix, Lambert/specular, face normals and
winding) is replaced by a one-slide recall pointing to units 1 and 3 here. What stays is the
*application*, which is specific to isosurfaces and is a good unit on its own: normals from the
image gradient (trilinearly interpolated to vertices), anisotropic spacing in positions *and*
gradients, failure cases (noise, flat regions, binary inputs), slice planes with the isocontour,
and the `marchcubes` demo. Recommended: keep it as a shorter application unit (≈ 1 h), titled
e.g. "Normals from the image". If the marching-cubes day is short of time, the alternative is to
fold the gradient-normal slides into its unit 2 and the `marchcubes` demo into unit 4, and drop
the unit. Its optional unit 5 mention of "direct volume rendering, MIP" becomes a pointer to
unit 4 here, and its proposed MIP view mode in `marchcubes` is replaced by embedding `volview`
(below), so there is one volume renderer.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `pipeline3d/`: Meshes, coordinate spaces and projection

- Two ways to represent a 3D object: a **surface** (polygons) or a **volume** (voxels). Today: how
  each becomes an image on the screen. This unit and the next two: polygons; unit 4: voxels.
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

### 2. `rasterise/`: From triangles to pixels: rasterisation and the z-buffer

- After the viewport transform a triangle is three points in pixel coordinates with a depth.
  **Rasterisation**: which pixel centres lie inside, and with which interpolated values.
- **Edge functions**: $E_{ab}(\mathbf p) = (\mathbf b-\mathbf a)\times(\mathbf p-\mathbf a)$
  (the $z$ component, a $2\times2$ determinant) is twice the signed area of triangle $a, b, \mathbf p$;
  inside = all three edge functions have the same sign. Linear in $\mathbf p$, so it updates by
  one addition per pixel step (Pineda 1988), and every pixel is independent (why GPUs are fast).
  Bounding box, tie-breaking for pixels exactly on an edge (top-left rule: shared edges are
  drawn once, no gaps, no double blending).
- **Barycentric coordinates**: $\lambda_a = E_{bc}(\mathbf p)/E_{bc}(\mathbf a)$ etc.: ratios of
  areas, $\lambda_a + \lambda_b + \lambda_c = 1$, $\mathbf p = \lambda_a\mathbf a + \lambda_b\mathbf b
  + \lambda_c\mathbf c$. Any vertex attribute interpolates as
  $\lambda_a f_a + \lambda_b f_b + \lambda_c f_c$. **Link to bilinear interpolation** (`images/`):
  both are weights from areas that sum to one; barycentric is the exact linear interpolant on a
  triangle, bilinear the product-of-linears on a square (a square split into two triangles gives
  a different interpolant: show the crease).
- **Back-face culling**: in screen space the winding of a back-facing triangle flips, so the sign
  of its area decides; halves the work for closed meshes, wrong for open or inconsistently wound
  meshes (link to unit 1).
- **The z-buffer** (Catmull 1974): one depth per pixel, initialised to "far"; draw a fragment only
  if it is nearer. Order-independent for opaque surfaces; transparency breaks it (needs sorting,
  or volume-rendering-style compositing, unit 4). Equal depths → **z-fighting**.
- **Depth precision**: $z_\text{ndc} = A + B/z$ is hyperbolic: most of the range is spent near
  the near plane; moving $n$ from 0.01 to 1 helps far more than moving $f$. Stored as a 24-bit
  integer or a 32-bit float (formats and float spacing: pointer to `plan/bitdepth-day.md`;
  "reversed z" as a one-line pointer there).
- **Perspective-correct interpolation**: barycentric weights computed in screen space are *not*
  the weights in 3D, because the divide is nonlinear. Screen-space linear: $1/w$ and $f/w$;
  so interpolate those and divide: $f = \dfrac{\sum\lambda_i f_i/w_i}{\sum\lambda_i/w_i}$. Show
  the classic wrong result: a checkerboard on a receding floor bends along the diagonal.
  Orthographic projection needs no correction ($w = 1$).
- **Clipping**: triangles crossing the near plane must be cut *before* the divide (a point with
  $w \le 0$ is behind the eye and would flip through infinity). Clip against $-w \le z$ in clip
  space; a triangle cut by one plane becomes one or two triangles (Sutherland–Hodgman on one
  plane, by picture). Side planes can be handled by a guard band (mention).
- Aliasing at edges (a pixel is a point sample; recall Part 1 sampling): multisampling as a
  one-slide pointer.
- **Demo `raster`** (new, canvas, CPU software rasteriser in `lib/raster.ts`): a coarse pixel grid
  (selectable 16×12, 32×24, 64×48) with two overlapping triangles whose vertices are draggable,
  each with a depth slider. Buttons *next pixel*, *next row*, *all*, *reset*: the current pixel
  shows its three edge-function values and signs, its barycentric weights (as numbers and as
  an RGB colour), its interpolated depth, and whether the depth test passes. Views: *colour
  buffer*, *depth buffer* (grey), *barycentrics*, *edge function of edge ab* (signed heat map).
  Toggles: *back-face culling*, *top-left rule* (off: a shared edge drawn twice, shown in red),
  *perspective-correct* (a textured quad receding in depth: checkerboard bends when off).
  Second tab *depth precision*: the curve $z_\text{ndc}(z)$ for given $n, f$, its quantisation
  to $b$ bits drawn as steps, and two nearly coplanar quads in a small WebGL view that z-fight as
  $n$ shrinks. Presets: "one triangle", "two triangles intersect" (the z-buffer resolves it per
  pixel), "a shared edge", "flipped winding is culled", "affine vs perspective-correct",
  "near plane 0.01 vs 1".
- By hand: for $\mathbf a = (1, 1)$, $\mathbf b = (7, 2)$, $\mathbf c = (3, 6)$, decide for 4 given
  pixel centres whether they are inside (edge functions), and compute the barycentric weights
  and interpolated depth for one of them. Fill a 4×4 z-buffer by drawing two given triangles
  in both orders: same result? Perspective-correct: an edge from $w = 1$ (texture $u = 0$) to
  $w = 3$ ($u = 1$); what $u$ is shown at the screen-space midpoint, affine vs correct?

### 3. `shading/`: Light, normals and textures

- Without shading a mesh is a flat silhouette. Local illumination with one light: **Lambert**
  (diffuse) $k_d\,\max(0, \mathbf n\cdot\mathbf l)$, from the projected area of a surface patch
  (the same $\cos\theta$ as sunlight on a tilted roof). **Phong** specular
  $k_s\,\max(0, \mathbf r\cdot\mathbf v)^s$ with the reflection
  $\mathbf r = 2(\mathbf n\cdot\mathbf l)\mathbf n - \mathbf l$; **Blinn–Phong** with the half vector
  $\mathbf h = (\mathbf l+\mathbf v)/\lVert\cdot\rVert$, $\max(0, \mathbf n\cdot\mathbf h)^{s'}$ (cheaper,
  better at grazing angles). Ambient term as a crude stand-in for indirect light. Not physical:
  one-slide pointer to physically based shading and global illumination (shadows, inter-reflection).
- **Where** to evaluate: **flat** (one normal per face: facets), **Gouraud** (light at vertices,
  colours interpolated with barycentrics: misses highlights inside a triangle), **per-pixel /
  Phong shading** (interpolate the normal, renormalise, light per pixel). Vertex normals as
  (area-weighted) averages of face normals; hard edges need split vertices.
- **Transforming normals**: under a non-uniform scale, $M\mathbf n$ is no longer perpendicular to the
  surface. Derivation: a tangent $\mathbf t$ transforms as $M\mathbf t$; we need
  $\mathbf n'^\top M\mathbf t = 0$ whenever $\mathbf n^\top\mathbf t = 0$, which holds for
  $\mathbf n' = M^{-\top}\mathbf n$. For rotations $M^{-\top} = M$ (orthogonal), for uniform scale it
  differs only by length. Picture: a circle squashed into an ellipse with tangents and normals.
- **Textures**: per-vertex texture coordinates $(u, v)$, interpolated perspective-correctly
  (unit 2), then the texture (an image) is **sampled**: nearest or **bilinear**, exactly the
  interpolation of `images/`. **Magnification** blurs or blocks; **minification** aliases (many
  texels per pixel: Part 1 sampling). **Mipmaps** (Williams 1983): a pyramid of pre-filtered
  half-size images, level chosen from the pixel footprint $\approx \log_2$ of texels per pixel;
  trilinear = bilinear in two levels + linear between them. Anisotropic filtering: mention.
  Texture formats, bit depth, float textures: pointer to `plan/bitdepth-day.md`; sRGB textures
  and gamma: pointer to `plan/colour-day.md`.
- **The GPU pipeline, conceptually**: vertex stage (per vertex: $P V M\mathbf x$, transform normals;
  programmable: "vertex shader") → primitive assembly, clipping, divide, viewport, culling,
  rasterisation (fixed) → fragment stage (per fragment: shading, texture lookups; programmable:
  "fragment shader") → depth test and blending (fixed, configurable). Written as a diagram plus
  the formulas each stage evaluates; no shader code. Thousands of vertices/fragments in parallel;
  the same independence argument as edge functions (and as marching cubes).
- **Demo `shading`** (new, WebGL via `lib/gl3d.ts`): a mesh (icosphere at 3 subdivision levels,
  torus, a low-poly "blob" with sharp creases), drag to rotate the object; the light direction
  set by dragging a point on a small sphere gizmo (pointer only, no keys). Controls: *flat /
  Gouraud / per-pixel*, *Lambert / Phong / Blinn–Phong*, $k_a, k_d, k_s$, shininess; toggles
  *show normals* (short line per vertex or per face), *show light vector, $\mathbf h$, $\mathbf r$
  at a clicked point*, *non-uniform scale* with *normals by $M$ / by $M^{-\top}$* (wrong shading
  visibly lights the dark side). A readout at a clicked point: $\mathbf n\cdot\mathbf l$ and the terms.
  Tab *texture*: a checkerboard / text texture on a receding plane with *nearest / bilinear /
  mipmap nearest / trilinear*, *colour the mip level*, and a zoom of the far region showing
  moiré. Presets: "Gouraud misses the highlight" (coarse sphere, small shininess spot),
  "facets vs smooth", "squashed sphere: wrong normals", "Phong vs Blinn–Phong at grazing light",
  "minification aliasing", "mipmaps fix it".
- By hand: Lambert and Blinn–Phong at a point for given $\mathbf n$, $\mathbf l$, $\mathbf v$; Gouraud
  vs per-pixel at the centroid of a triangle whose vertices are all unlit but whose centre faces
  the light; for $M = \mathrm{diag}(2, 1, 1)$ and the plane $x + y = 1$ (normal $(1, 1, 0)$),
  transform a tangent and the normal both ways and check perpendicularity; which mip level for a
  pixel covering 8×8 texels?

### 4. `volrender/`: Seeing voxels: slices, projections and volume rendering

- A **volume** is a 3D array $I[i, j, k]$ plus a **voxel → world** transform. In general a
  $4\times4$ affine $\mathbf x = A\,[i, j, k, 1]^\top$ whose columns are the voxel axes scaled by
  the spacing and the position of voxel $(0,0,0)$; what medical headers store (NIfTI `sform`,
  DICOM pixel spacing + orientation + position; one slide, with the warning that axis conventions
  differ between formats). World → voxel: $A^{-1}$. Spacing alone is the diagonal special case;
  anisotropic $\Delta z$ (microscopy, thick CT slices) and what goes wrong when it is ignored.
  Recall (if the marching-cubes day has been seen, it is a recall): a stack, a slice, a MIP.
- **Orthogonal slices** are array indexing. **Oblique reslicing**: a plane with origin $\mathbf p_0$
  and in-plane axes $\mathbf s, \mathbf t$ in world space; for each output pixel $(u, v)$:
  $\mathbf x = \mathbf p_0 + u\mathbf s + v\mathbf t$ → voxel index $A^{-1}\mathbf x$ (not an integer) →
  **trilinear interpolation** of the 8 neighbours. This is inverse mapping from `images/`, one
  dimension up; the same procedure gives the general **resampling** of a volume to a new grid
  (registration, making voxels isotropic).
- **Projections along rays**: for each screen pixel a ray through the volume (orthographic: parallel
  rays; perspective: through the eye, built with the unit 1 matrices inverted). Sample at steps
  $\Delta s$ with trilinear interpolation. **MIP** = maximum along the ray (vessels, bright spots,
  standard in microscopy and angiography; loses depth, needs rotation to read); **MinIP** (airways);
  **average / sum** ≈ an X-ray (Beer–Lambert; the link back to how CT data are acquired).
- **Direct volume rendering**. A **transfer function** maps a value to colour $c(v)$ and opacity
  (extinction) $\tau(v)$, which is where the user's choice "what to see" lives. Gentle physics: along
  the ray, light travelling toward the eye is emitted and absorbed: $\dfrac{dL}{ds} = -\tau L + \tau c$
  (a linear first-order ODE). Solved with $t$ = distance from the eye and the **transmittance**
  $T(t) = \exp\!\big(-\int_0^t\tau\big)$ (the fraction of light from depth $t$ that reaches the eye):
  $L = \int_0^D T(t)\,\tau(t)\,c(t)\,dt$. What is in front attenuates what is behind.
  Discretise in steps: opacity of a segment $\alpha_i = 1 - e^{-\tau_i\Delta s}$, giving
  **front-to-back compositing** $C \leftarrow C + T\,\alpha_i c_i$, $T \leftarrow T(1-\alpha_i)$
  (Porter–Duff "over"). Back-to-front gives the same result (by hand). **Early ray termination**:
  stop when $T < \varepsilon$. **Opacity correction** when the step changes:
  $\alpha' = 1-(1-\alpha)^{\Delta s'/\Delta s}$, otherwise the image darkens with more samples.
  Step size vs aliasing (wood-grain artefacts; recall sampling). Pre- vs post-classification
  (interpolate values then apply the transfer function, or the reverse): picture only.
  Shading in DVR with the gradient as normal: pointer to the marching-cubes day unit 3.
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
  (drag on the 3D view to tilt the plane; plane drawn in the 3D view). Right: 3D view with mode
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
  transmittance $T$. Buttons *next sample*, *all*, *reset*; toggles *front-to-back / back-to-front*
  (same final colour, different partial sums), *MIP* (running max instead), *early termination*
  (marks where it would stop), step size. Presets: "opaque wall hides what is behind", "fog",
  "two layers", "MIP vs DVR on the same ray". This is the by-hand exercise made interactive.
- By hand: given a $4\times4$ voxel → world matrix (spacing 0.5, 0.5, 2 mm and an origin), find the
  world position of voxel $(2, 3, 1)$ and the voxel index of a world point, then trilinearly
  interpolate there from 8 given values. A ray with 5 samples: values given, transfer function
  given as a small table: compute MIP, average, and front-to-back $C$ and $T$ step by step; repeat
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

- **`demos/src/lib/gl3d.ts` + `mat4.ts`** (proposed in `plan/marchingcubes-day.md`; one renderer for
  the course). This day extends rather than duplicates it:
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
- **`demos/src/lib/volrender.ts`** (new, on top of `gl3d.ts`): upload a `Volume` as a 3D texture,
  slice quad for an arbitrary plane, a ray-marching fragment program with modes MIP / MinIP /
  average / DVR / first-hit isosurface, transfer function as a 256×1 RGBA texture, step size,
  opacity correction, early termination, a samples-per-pixel output. Used by `volview`, and by
  the marching-cubes day if it embeds a volume view.
- **`demos/src/lib/volume.ts`** (from the marching-cubes plan): add a `voxelToWorld` $4\times4$
  (spacing becomes its diagonal special case), oblique reslicing on the CPU (reference for
  tests and for `raycomposite`), a CPU ray marcher for one ray (instrumented: returns every
  sample, $\alpha_i$, $C$, $T$), and the phantom generator (nested ellipsoids, tube tree,
  cavity, seeded noise).
- **`demos/src/lib/raster.ts`** (new): CPU software rasteriser on a small grid: edge functions,
  barycentrics, top-left rule, z-buffer, perspective-correct vs affine attributes, near-plane
  clipping; an instrumented step mode that records per-pixel values for `raster`. Also a test
  oracle for `gl3d.ts` matrices (project a point both ways).
- **`demos/src/lib/mesh.ts`** (from the marching-cubes plan): add primitives (cube with correct
  winding, icosphere, torus, low-poly crease blob), vertex-normal averaging with crease angle,
  winding/manifold checks, and voxelisation (optional unit).
- **`lib/imageops.ts`**: the checkerboard/text test textures and the mipmap pyramid builder (a
  2×2 box downsample; or reuse the Gaussian pyramid if the snakes/skeleton plans put one there).
- **Tests** (node script, `make test`, as in the other plans): $P$ maps the near/far corners of
  the frustum to the NDC cube corners; `lookAt` gives an orthonormal $R$ and $V\mathbf e = \mathbf 0$;
  barycentrics sum to 1 and reproduce the vertices; the rasteriser covers each pixel of a quad
  split into two triangles exactly once (top-left rule); perspective-correct interpolation
  matches the 3D intersection point; $M^{-\top}\mathbf n$ stays perpendicular to transformed
  tangents; front-to-back = back-to-front compositing; opacity correction makes the result
  (nearly) independent of step size on a homogeneous slab, matching $1 - e^{-\tau D}$;
  reslicing along an axis-aligned plane equals array indexing.
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
| 10:45–12:15 | Unit 2 lecture, rasterise by hand, `raster` |
| 13:15–14:45 | Unit 3 lecture, shading and normal transform by hand, `shading` |
| 15:00–16:30 | Unit 4 lecture, one ray by hand, `raycomposite`, `volview`; quizzes as wrap-up |

## Open questions

- How long is a "day"? Four dense units; unit 4 carries both "volumes as data" (affine,
  reslicing, projections) and DVR. If too much, split it into `volumes3d/` and `volrender/` and
  move the texture half of unit 3 into further reading.
- Balance for an image-analysis course: is the polygon side (units 1–3) worth three units, or
  should units 2 and 3 merge (rasterisation + shading at overview depth) to give voxels two units?
  My recommendation: keep the derivations (perspective matrix, barycentrics, inverse transpose,
  compositing) as core, since they are the linear-algebra payoff; cut breadth (clipping,
  mipmaps, GPU pipeline) first.
- The marching-cubes rendering unit: keep as a short application unit "Normals from the image"
  (recommended) or fold into its units 2 and 4 (see the overlap section). Needs agreement with
  that plan's author.
- `gl3d.ts`: WebGL2 for all 3D demos. Decided 2026-11-12; the marching-cubes plan is updated.
- Coordinate conventions: OpenGL-style right-handed view space looking down $-z$ and NDC
  $z \in [-1, 1]$ (matches WebGL), or $z \in [0, 1]$ (Direct3D/Vulkan, simpler depth formula)?
  The slides should use one convention throughout; WebGL forces the first for the demos.
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
