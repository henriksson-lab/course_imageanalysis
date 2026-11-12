# Plan: a day on marching cubes (isosurfaces from 3D images)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** elective, group "3D" (marching cubes → 3D graphics); entry after
Part I (core day 4). First day of the group, so it owns the course's **3D-images unit**
`volumes3d/` (unit 1), which the 3D-graphics day and the 3D pointers of labelling, morphology,
skeleton, measurements, FFT registration and moments recall. Edits from the three reviews
applied 2026-11-12: new unit 1 `volumes3d/`; `isocontour/` keeps marching squares, the
asymptotic decider and the shoelace recall; the former `isosurface-render/` is folded into
`marchingcubes/` (gradient normals) and `surface-measure/` (anisotropic spacing, slice planes).

## Goal of the day

After the day a student can:

1. describe a **3D image** (CT, MRI, a confocal or light-sheet stack) as samples of a function $I(x, y, z)$ on a grid with **voxel spacing** $(\Delta x, \Delta y, \Delta z)$, often anisotropic, map between voxel indices and world coordinates with a $4\times4$ affine, resample by **trilinear interpolation** (orthogonal and oblique slices), and say what a **maximum-intensity projection** hides;
2. describe an object boundary as the **isosurface** (level set) $\{I = c\}$, and run **marching squares** by hand on a small grid: classify each cell (16 cases), place the crossing points by **linear interpolation** along edges, and resolve the two **saddle cases** with the **asymptotic decider**;
3. explain **marching cubes**: the 256 cases, why they reduce to 15 (and to 23 without the complement symmetry), what the triangle table stores, why naive ambiguity handling gives **holes**, and how **marching tetrahedra** avoids ambiguity; compute **vertex normals from the image gradient**;
4. **measure** surface area and enclosed volume from the mesh, explain why counting voxel faces overestimates area, and reason about the choice of **iso-value**, **smoothing before extraction**, anisotropic spacing, and mesh size / **decimation**.

Why it fits this audience: every step is linear interpolation, bilinear/trilinear
interpolation, a gradient, a cross product or a determinant. Area and volume of a mesh
are a direct payoff of their linear algebra (a $3\times3$ determinant is the signed
volume of a tetrahedron) and calculus (the divergence theorem). It reuses from day 1
explicitly: **inverse mapping and bilinear interpolation** (`images/`) and **gradients by
central differences / Sobel** (`kernels/`).

## Prerequisite gap

- Part I (days 1–4) is entirely 2D. Unit 1 `volumes3d/` is the course's first look at **3D
  images**: stacks of slices, voxel spacing, why $\Delta z$ is usually larger than $\Delta x$
  in microscopy (the elongated PSF of day 4, one-line recall), the voxel ↔ world affine, and
  how to look at a volume (slices, MIP) before we surface it.
- Segmentation is not needed: an isosurface of the grey-level image *is* a threshold (day 6
  if taken, otherwise one slide) with sub-voxel boundaries. Binary or label images from
  other days (thresholding day 6, watershed day 10, graph cuts day 13, level sets elective)
  are covered in unit 4 as "surfacing a segmentation"; this elective can run before them, so
  unit 4 phrases them as pointers.
- **Trilinear** interpolation is one line from bilinear (interpolate in $x$, $y$, then $z$),
  taught in unit 1. The **divergence theorem** may be rusty: the 2D version is the shoelace
  formula (owned by the labelling day, day 9, `shape-features/`; one-line recall, with the
  formula restated in unit 2 for groups taking this elective before day 9), then the 3D
  version in unit 4.

**Connections to other days.** *Level sets* (elective, Curves group): the level-set
function $\varphi$ is evolved on a grid and the segmentation is its zero contour, drawn and
measured with exactly marching squares/cubes at $c = 0$ (`lib/marching.ts` is shared; the
level-set day draws contours with it and does not teach marching squares by hand); a signed
distance function is the ideal input for marching cubes (unit 4). *Watershed* (day 10) and
*graph cuts* (day 13) produce label images; turning a label into a surface is unit 4.
*3D graphics* (next in this group): projection, rasterisation and shading in full; here they
are one-slide pointers.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`volumes3d/`, `isocontour/`, `marchingcubes/`, `surface-measure/`; optional `surfacenets/`.

### 1. `volumes3d/`: 3D images (shared unit; recalled by the 3D-graphics day and by 3D pointers elsewhere)

- 3D images: a CT scan (isotropic, ~0.5 mm), a confocal stack (e.g. 0.1 µm in $x,y$, 0.5 µm
  in $z$), light-sheet, EM. A volume is a 3D array $I[z, y, x]$ plus a spacing
  $(\Delta x, \Delta y, \Delta z)$; forgetting the spacing squashes everything. Why
  $\Delta z > \Delta x$ in microscopy (axial PSF elongation, day 4) in one line.
- **Voxel ↔ world**: in general a $4\times4$ affine $\mathbf x = A\,[i, j, k, 1]^\top$ whose
  columns are the voxel axes scaled by the spacing and the position of voxel $(0,0,0)$
  (moved here from the 3D-graphics day); what medical headers store (one slide, with the
  warning that axis conventions differ); world → voxel is $A^{-1}$; spacing alone is the
  diagonal special case. Image axes as in `images/`: $x$ right, $y$ down, $z$ along the stack.
- **Trilinear interpolation** (one line from bilinear: a weighted sum of 8 corners).
  **Orthogonal slices** (xy, xz, yz) are array indexing; **oblique reslicing**: a plane with
  origin and in-plane axes; for each output pixel, world point → $A^{-1}$ → trilinear. This is
  inverse mapping from `images/`, one dimension up, and also how a volume is **resampled to
  isotropic voxels** (and how 3D registration would resample).
- **Anisotropy in practice**: an isotropic Gaussian in index space is anisotropic in the
  world; use $\sigma$ per axis in physical units. Connectivity in 3D (6/18/26) is recalled
  from day 9.
- **A first MIP** (maximum along $z$): what it hides: overlaps (two nuclei at different depths
  merge), wrong areas and counts, no depth. Rule: measure in 3D, not on the MIP; 2D
  segmentation slice by slice vs 3D segmentation (one slide). Rendering projections and
  volume rendering in full: 3D-graphics day `volrender/`.
- **Demo `volview`** with `data-mode="volumes3d"` (the course's one volume viewer, owned by
  the 3D-graphics day, first built here): a synthetic nuclei stack with anisotropic voxels
  (two nuclei touching in $xy$ at different depths, one elongated cell), three orthogonal
  slices + an oblique slice with a draggable plane, a MIP panel, a *z spacing: correct /
  ignored* toggle and a $\Delta z$ slider, a *resample to isotropic* button, a voxel → world
  matrix readout, and a click-a-voxel readout of world coordinates. Presets: "two nuclei
  merge in the MIP but not in 3D", "ignored z spacing squashes the cell", "an oblique slice
  through a tilted cell", "resampling to isotropic".
- By hand: given a $4\times4$ voxel → world matrix (spacing 0.5, 0.5, 2 µm and an origin), the
  world position of voxel $(2, 3, 1)$ and the voxel index of a world point; trilinearly
  interpolate there from 8 given values. From three given slices of a small stack, draw the
  MIP and count objects in it and in 3D.

### 2. `isocontour/`: Level sets and marching squares

- The boundary of a bright object is where $I$ crosses a value $c$: the **level set** $\{x : I(x) = c\}$. In 2D a set of curves (an **isocontour**, like a height map's contour lines), in 3D a surface. Thresholding gives pixels; the level set gives a boundary between them.
- We only know $I$ at grid points, so we decide on an interpolant. Per cell (2×2 pixels): mark each corner above (1) or below (0) $c$ → a 4-bit **case index**, 16 cases. Each edge with a sign change is crossed once.
- Where on the edge: linear interpolation, $t = (c - I_a)/(I_b - I_a)$. Compare with "midpoint" (blocky, the contour of a thresholded image).
- Neighbouring cells share an edge, hence the same crossing point: the contour is closed and consistent without any global step. "Marching" = visit the cells independently (embarrassingly parallel).
- **Saddle cases** (5 and 10: diagonal corners above): two valid contours. The bilinear interpolant (from *Images as arrays*) inside the cell has hyperbolic level sets; its saddle value is $f^\ast = (I_{00}I_{11} - I_{10}I_{01})/(I_{00} + I_{11} - I_{10} - I_{01})$. **Asymptotic decider** (Nielson & Hamann 1991): if $f^\ast > c$ the above-$c$ corners are connected. Fixed rule ("always separate") is also consistent in 2D, but may give the wrong topology.
- Measure from the contour: length (sum of segment lengths) and enclosed area by the **shoelace formula** $A = \tfrac12\sum_i (x_i y_{i+1} - x_{i+1} y_i)$ (recalled from day 9 `shape-features/`, its owner; a sum of 2×2 determinants; the 2D preview of unit 4's divergence theorem).
- **Demo `marchsquares`** (new, small): an 8×8 grid of corner values shown as numbers/grey squares. Click a corner to toggle it above/below, or drag up/down to change its value; slider for $c$. Each cell shows its case index and its line segments; toggles: *midpoint / linear interpolation*, saddle rule *separate / join / asymptotic decider*, *show bilinear interpolant* (the cell background shaded with the true interpolant and its exact hyperbolic level curve, so the line segment visibly approximates it). Clicking a cell shows the arithmetic: bits, case, $t$ on each crossed edge, and for a saddle $f^\ast$ vs $c$. Presets: "one bump", "all 16 cases", "saddle: the decider joins", "saddle: the decider separates", "midpoint vs linear".
- **Demo `isocontour`** (new; uses the histogram + threshold slider shell of the thresholding day): the 128×128 micrograph crop and a synthetic image (blobs with noise and a thin neck). Slider for $c$ (shown on a histogram), Gaussian blur $\sigma$, noise; marching-squares contours overlaid on the image; readouts: number of contours, total length, enclosed area (shoelace) vs pixel count above $c$. Presets: "noise gives many tiny contours", "blur first", "two blobs merge as $c$ drops".
- By hand: a 4×4 grid of values with $c = 5$: classify the 9 cells, interpolate the crossing points, draw the contour; one cell is a saddle: compute $f^\ast$ and decide. Derive $t$ from the straight line through $(0, I_a)$ and $(1, I_b)$.

### 3. `marchingcubes/`: From squares to cubes, and normals from the image

- The same idea one dimension up: a cell is a cube of 8 voxels, $2^8 = 256$ cases, 12 edges, up to 5 triangles per cube. Vertices by linear interpolation on edges; shared edges give a closed (watertight) mesh, and each edge vertex is computed once and shared (an index per grid edge).
- **Symmetry**: the 24 rotations of the cube reduce 256 to **23** classes (count it with Burnside's lemma as an aside: $(256 + 6\cdot4 + 3\cdot16 + 8\cdot16 + 6\cdot16)/24 = 23$). Lorensen & Cline (1987) also used complement (swap inside/outside) to get **15**.
- **The triangle table**: for each of the 256 cases a list of edge triples. Key insight for the lecture: the surface meets each face of the cube in a marching-squares contour, so the table can be **generated**: run marching squares on the 6 faces, chain the segments into closed loops, triangulate each loop as a fan. We generate it in code this way rather than copying a table (see shared engineering).
- **Ambiguity and holes**: six classes contain an ambiguous face (a 2D saddle). Two neighbouring cubes must make the same choice on their shared face, or the mesh has a **crack**. The complement trick silently breaks this (Dürst 1988). Fixes: a consistent face rule (23 classes, no complement), the asymptotic decider per face (Nielson & Hamann 1991), and interior ambiguities / tunnels (MC33, Chernyaev 1995; Lewiner et al. 2003). Lecture depth: faces in full, interior ambiguity as a picture only.
- **Marching tetrahedra**: split each cube into 5 or 6 tetrahedra; 16 cases per tetrahedron reduce to 3 (nothing, one corner off → 1 triangle, two corners off → a quad = 2 triangles). Never ambiguous (the interpolant is linear on a tetrahedron), but more triangles and a directional bias from the choice of split.
- Complexity: one pass over the voxels, $O(n^3)$, and the output size grows with the surface area, $O(n^2)$.
- **Normals from the image** (folded in from the former `isosurface-render/`, 2–3 slides): the surface is a level set, so the normal is parallel to $\nabla I$; $\mathbf{n} = -\nabla I/\lVert\nabla I\rVert$ for a bright object, by central differences (or Sobel, `kernels/`) at grid points, trilinearly interpolated to the vertex: smooth shading from a coarse mesh, for free. Face normals by cross product and Lambert shading are a one-slide pointer to the 3D-graphics day (`pipeline3d/`, `rasterise/`). Where gradient normals fail: noise (blur first), flat regions where $\lVert\nabla I\rVert \approx 0$, binary inputs (gradient only along 6 directions).
- **Demo `mccube`** (new): one cube, rotatable by dragging (WebGL via `lib/gl3d.ts`, like every 3D demo). Click the 8 corners to toggle in/out; shows the 8-bit index, the case number 0–255, its class (1 of 15 / 1 of 23) and the rotation that maps it to the class representative (a button animates that rotation). Mode *values*: each corner gets a value (slider for the selected corner, slider for $c$), so the vertices slide along the edges. Ambiguous faces are highlighted; a *face rule* toggle (separate / join / decider) shows the alternative triangulations. Toggle *marching tetrahedra* (5 or 6 tetrahedra) shows the decomposition and its triangles for the same corners. Optional: *neighbour cube* toggle draws a second cube sharing the ambiguous face, with mismatched rules exposing the crack in red. Presets: "one corner", "an edge (two corners)", "the 15 classes" (step through), "an ambiguous face", "a crack", "tetrahedra instead".
- **Demo `marchcubes`** (new, main demo; moved here from the former unit 3): a 48³–64³ volume, marching cubes in TS on every change (a few ms at this size), rendered with `lib/gl3d.ts`, rotated by dragging. Volumes: two spheres that merge as $c$ drops (metaballs), a torus (genus 1), a noisy sphere, an ellipsoid sampled with anisotropic voxels, and a small real stack (see open questions). Controls: $c$ (with histogram), blur $\sigma$, noise, *normals: face / gradient*, *wireframe*, *z spacing* (correct / ignored), *algorithm: MC / MT*, *face rule* (with boundary edges, i.e. cracks, drawn in red), *slice plane* (an xy slice with its marching-squares contour in a side panel). Readouts: vertices, triangles, area, volume, number of components, Euler characteristic. Presets: "two blobs merge", "torus has a hole", "noise: thousands of tiny surfaces", "blur first", "faceted vs gradient normals", "wrong z spacing", "cracks from inconsistent faces".
- By hand: given corner values and $c$: compute the case index (with the course's corner-numbering convention printed on the sheet), list the cut edges, interpolate the vertices, sketch the triangles. Show that two given cases are rotations of each other. For a tetrahedron, list all 16 cases and group them into 3 classes. Gradient at a grid point of a given 3×3×3 neighbourhood by central differences, normalised; which way does $\mathbf{n}$ point if the object is dark on a bright background?

### 4. `surface-measure/`: Measuring surfaces, choosing $c$, and mesh size

- **Area** $= \sum_\text{triangles} \tfrac12\lVert(\mathbf{b}-\mathbf{a})\times(\mathbf{c}-\mathbf{a})\rVert$. **Volume** by the divergence theorem: $V = \sum \tfrac16\,\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})$, a sum of signed tetrahedron volumes ($\tfrac16\det[\mathbf{a}\,\mathbf{b}\,\mathbf{c}]$) — the 3D shoelace formula. Needs a closed, consistently oriented mesh: pad the volume with background so surfaces touching the border close.
- Voxel-based alternatives: volume = voxel count × voxel volume (fine); area = count of exposed voxel faces (**wrong**: about 1.5× too large for a sphere, for any resolution; in 2D the pixel-edge perimeter of a disk is $4/\pi \approx 1.27\times$ too long). Marching cubes area converges to the true area as the voxels shrink.
- **Anisotropic spacing** (folded in from the former `isosurface-render/`): vertex positions *and* the gradient must use $\Delta z$ (recall `volumes3d/`); forgetting it in the gradient tilts the normals, forgetting it in positions biases area and volume. **Slice planes** through the volume with the 2D contour drawn on them: a slice of an isosurface is an isocontour (link back to unit 2).
- **Topology check**: Euler characteristic $\chi = V - E + F = 2 - 2g$ per closed component (sphere 2, torus 0); boundary edges = cracks.
- **Choosing $c$**: a blurred step edge between levels $a$ and $b$ crosses $(a+b)/2$ at the true edge (symmetric blur), so the half-way value is the natural choice; on curved, small objects the blur still biases size (the partial-volume effect). Volume vs $c$ curve: flat plateau = robust, steep = the measurement is mostly a choice of $c$. Otsu / minimum-error from the histogram: taught on day 6 (`plan/thresholding-day.md`, unit 2 `otsu/`); one-line pointer.
- **Smoothing before extraction**: Gaussian blur of the volume (removes noise, rounds corners, shrinks thin structures) vs smoothing the mesh afterwards (Laplacian smoothing shrinks; Taubin's two-step smoothing does not, mention only).
- **Surfacing a segmentation** (connection to watershed day 10, graph cuts day 13, level sets elective): a 0/1 mask at $c = 0.5$ puts every vertex at an edge midpoint: terraces and staircase normals. Remedies: blur the mask slightly before extraction, or use a signed distance function (what level-set methods carry around; its zero level set is sub-voxel accurate). Label images: one surface per label; neighbouring labels give two surfaces with a gap or overlap (multi-label methods: pointer only).
- **Mesh size**: number of triangles ∝ area / voxel area; a 512³ CT bone surface is millions of triangles. Shared vertices (about 2 triangles per vertex) vs triangle soup (3 vertices per triangle). **Decimation**: vertex clustering (snap vertices to a coarser grid, merge, drop degenerate triangles; simple, implemented in the demo) and edge collapse with quadric error metrics (Garland & Heckbert 1997, the standard; mention only). Watch area and volume as the triangle count drops.
- **Demo `meshmeasure`** (new): left, a 2D panel: a disk/ellipse digitised at a resolution slider, with pixel-edge perimeter, marching-squares perimeter and the true value; right, the same for a sphere in 3D (voxel-face area, MC area, true area; voxel-count volume, MC volume), and a small convergence plot (relative error vs voxel size, log–log) that fills in as the resolution changes. A second tab, *iso-value*: volume vs $c$ curve for a blurred sphere and for the real stack, with the current $c$ marked. A third tab, *decimate*: the `marchcubes` view with a vertex-clustering cell-size slider and triangle count / area / volume readouts; input *grey-level* vs *binary mask* vs *blurred mask*. Presets: "voxel faces overestimate area", "MC converges", "a plateau in volume vs c", "terraces from a binary mask", "decimate 10×". (If this is too much for one demo, split off `decimate` and reuse the `marchcubes` renderer with `data-mode="decimate"`.)
- By hand: one bright voxel (value 1) in a zero background, $c = 0.5$: marching cubes gives an **octahedron** with vertices at distance 0.5 from the centre. Compute its area ($\sqrt3 \approx 1.73$) and volume ($1/6$, by the determinant formula) and compare with the voxel count (1) and the voxel faces (6). What $c$ makes the octahedron volume equal 1? Then the perimeter of a digitised disk of radius 3 by pixel edges vs the true $2\pi r$.

### 5. (optional / further reading) `surfacenets/`: dual methods and volume rendering

- **Dual methods**: surface nets / dual contouring put one vertex *per cell* (at the mean of the edge crossings, or a least-squares fit using normals) and connect cells across each sign-changing edge. No case table, no ambiguity cracks, better triangles, and they handle label images naturally. Link to unit 3: the table method is "primal".
- **Direct volume rendering** as the alternative to surfaces: pointer to the 3D-graphics day (`volrender/`); no iso-value to choose, but nothing to measure.
- **Demo**: reuse `marchcubes` with `data-algorithm="surfacenets"` (side by side with MC on the same volume); MIP views come from embedding `volview`; no new demo.

## Shared engineering

- `demos/src/lib/volume.ts` (**owned by this day**): a `Volume` type (`Float32Array` stored $(z, y, x)$, dims, spacing $(\Delta x, \Delta y, \Delta z)$, a `voxelToWorld` $4\times4$ with spacing as its diagonal special case); synthetic generators (spheres/metaballs, torus, ellipsoid, the anisotropic nuclei stack for `volumes3d/`, noise with `rng`/`gauss` from `lib/random.ts`); separable 3D Gaussian blur with per-axis $\sigma$ in physical units; central-difference gradient; trilinear sampling; oblique reslicing and resampling to isotropic on the CPU (reference for tests); MIP along an axis; padding with background; loading a stack from a PNG slice atlas.
- `demos/src/lib/marching.ts` (**owned by this day**; the only marching-squares implementation in the course, also used by the level-set elective, so there is no `lib/contour.ts`): marching squares (with saddle rule option), marching cubes, marching tetrahedra (5/6 split), and the **generated** triangle table: built at load time from the face rule (marching squares on the 6 faces, chain loops, fan-triangulate), with a class index (case → representative + rotation) for `mccube`. Edge-indexed vertex sharing so meshes are indexed. Optional surface nets for unit 5. One implementation used by `marchsquares`, `isocontour`, `mccube`, `marchcubes`, `meshmeasure`, and the level-set demos.
- `demos/src/lib/mesh.ts` (**owned by this day**): face/vertex normals, area, signed volume, boundary edges, connected components, Euler characteristic, vertex-clustering decimation, Laplacian smoothing.
- `demos/src/lib/gl3d.ts` (+ a tiny `mat4.ts`; **owned by the 3D-graphics day**, whose subject is rendering; first written here because this day comes first in the group, to that day's spec): minimal WebGL2 renderer (WebGL2 for all 3D demos; decided 2026-11-12) written in the repo (one shader: Lambert + Blinn–Phong, flat or smooth normals, wireframe overlay via barycentric coordinates, a slice-plane quad with a texture), orbit rotation by pointer drag and buttons for standard views (no keyboard: arrow keys belong to reveal.js). All 3D demos in the course use this renderer (decided 2026-11-12); if WebGL is unavailable, show a static figure and a short note instead of a canvas fallback. Pointer coordinates via `getBoundingClientRect` because reveal.js scales slides with a CSS transform; `devicePixelRatio`-aware canvas. Browsers cap live WebGL contexts (~16): create the context only while the slide/section is visible (IntersectionObserver) and release it on leave.
- `volview` and `lib/volrender.ts` (owned by the 3D-graphics day): unit 1 embeds `volview` with `data-mode="volumes3d"` (slices, oblique plane, MIP); first built here if this day is built first.
- `isocontour` uses `lib/histogram.ts` (thresholding day) for its histogram + threshold slider.
- Tests (`make test` over `demos/test/*-test.ts`): every one of the 256 cases maps to exactly one of the 15/23 classes; meshes from random seeded volumes are closed and consistently oriented (every edge in exactly two triangles, opposite directions) for MC with a consistent face rule and for MT; the sphere's MC area and volume converge to $4\pi r^2$ and $\tfrac43\pi r^3$; the torus has $\chi = 0$; the single-voxel octahedron gives area $\sqrt3$ and volume $1/6$; `voxelToWorld` and its inverse round-trip; reslicing along an axis-aligned plane equals array indexing; resampling a constant volume stays constant.
- Data: synthetic volumes are generated in code (no files). The real stack is a small (about 64×64×32, 8-bit) crop stored as a PNG slice atlas and inlined by esbuild, with a generating script next to it (`make-volume.R`, like `fourier2d/make-image.R`), credit and licence shown in the demo. 2D image: reuse `demos/src/fourier2d/cell256.png`, downsampled to 128×128.
- Follow `demos/README.md`: no CDN (so no three.js), fixed seeds, presets + reset, labels ≥ 18 px, fits 1150×560 on a slide, `data-*` settings (e.g. `data-volume`, `data-mode`, `data-algorithm`), static figures as print fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, voxel ↔ world and the MIP by hand, `volview` (volumes3d) |
| 10:45–12:15 | Unit 2 lecture, marching squares by hand, `marchsquares`, `isocontour` |
| 13:15–14:45 | Unit 3 lecture, one cube and a gradient normal by hand, `mccube`, `marchcubes` |
| 15:00–16:30 | Unit 4 lecture, the octahedron by hand, `meshmeasure`; quizzes as wrap-up |

## Open questions

- **Decided (course map, review-organisation §2.4):** the former rendering unit `isosurface-render/` is folded into `marchingcubes/` (gradient normals) and `surface-measure/` (anisotropic spacing, slice planes); generic projection and shading are the 3D-graphics day's. If four units are still too many, the decimation part of unit 4 moves to further reading.
- The real 3D stack: which data, and under which licence? Candidates: a confocal nuclei stack (e.g. the `cells3d` sample shipped with scikit-image, licence to be checked), a CC-licensed CT from a public archive. It should have anisotropic voxels, to make that point with real data. Or stay synthetic only?
- How deep into ambiguity: face ambiguity and the asymptotic decider as core (my recommendation), MC33 / interior ambiguity as a picture only?
- Generate the triangle table in code (recommended: it teaches the face-rule idea and avoids a 256-row copied table with its own licence/provenance), or use a published table?
- ~~3D rendering: WebGL vs canvas?~~ Decided: WebGL (`lib/gl3d.ts`) for all 3D demos.
- **Decided:** the course's "3D images" unit is `volumes3d/`, unit 1 of this day (review-gaps gap 8 proposed `volumes/` "after `images/` or at the start of the marching-cubes day"; the course map keeps 3D as an elective, so it lives here and core days point to it).
- **Decided (course map):** this elective may run any time after Part I, so watershed (day 10), graph cuts (day 13) and the level-set elective may still be ahead; unit 4 phrases them as pointers. Otsu is day 6's.
- Programming exercises (e.g. running marching cubes on a stack and measuring it in R/Python) are left out of this plan for now; whether to add them, and in which language, is open (see README). The demos and by-hand exercises do not depend on it.

## References

- Lorensen & Cline (1987), *Marching cubes: a high resolution 3D surface construction algorithm*, SIGGRAPH.
- Dürst (1988), *Re: additional reference to "marching cubes"*, Computer Graphics 22(2) (the ambiguity/hole problem).
- Nielson & Hamann (1991), *The asymptotic decider: resolving the ambiguity in marching cubes*, IEEE Visualization.
- Doi & Koide (1991), *An efficient method of triangulating equi-valued surfaces by using tetrahedral cells*, IEICE Trans. (marching tetrahedra).
- Chernyaev (1995), *Marching cubes 33: construction of topologically correct isosurfaces*, CERN report CN/95-17.
- Lewiner, Lopes, Vieira & Tavares (2003), *Efficient implementation of marching cubes' cases with topological guarantees*, J. Graphics Tools.
- Newman & Yi (2006), *A survey of the marching cubes algorithm*, Computers & Graphics.
- Gibson (1998), *Constrained elastic surface nets*, MICCAI; Ju, Losasso, Schaefer & Warren (2002), *Dual contouring of Hermite data*, SIGGRAPH.
- Garland & Heckbert (1997), *Surface simplification using quadric error metrics*, SIGGRAPH.
- Taubin (1995), *A signal processing approach to fair surface design*, SIGGRAPH.
- Lindblad (2005), *Surface area estimation of digitized 3D objects using weighted local configurations*, Image and Vision Computing (why voxel-face area is biased).
