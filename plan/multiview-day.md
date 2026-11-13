# Plan: a day on geometry from multiple images (cameras, stereo, reconstruction)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.

**Course map (2026-11-12):** elective, group "3D", the row "3D: geometry from images"
(requested 2026-11-12). Entry after the 3D graphics day (`plan/graphics3d-day.md`), which
itself comes after marching cubes and Part I. **Ownership change applied here (course map,
"Ownership changes"):** this day owns **stereo**; the dynamic programming day's optional
scanline stereo (`dtw/`, not built) becomes a one-slide application pointer to unit 3 here.

**Notation (stated once in unit 1):** intrinsics $K$, rotation $R$, translation $\mathbf t$,
camera matrix $P = K[R \mid \mathbf t]$ ($3\times4$; the graphics day's $4\times4$ projection is
also called $P$: the unit 1 comparison slide names both). Vision convention: the camera looks
down $+z$, image $y$ points down, pixel centres at integers (as `lib/imageops.ts`); the
graphics day's view space looks down $-z$, and the two differ by $\mathrm{diag}(1, -1, -1)$.
"Equal up to scale" is written $\simeq$ (not with $\lambda$, which is the prior weight).
Essential matrix $E$ (the graph-cuts day's energy is also $E$; never on the same slide),
fundamental matrix $F$, disparity $d$, baseline $B$, focal length $f$ **in pixels**, depth $Z$.
Levenberg–Marquardt damping $\mu$ (not $\lambda$). Transmittance $\mathcal T$ and opacity
$\alpha$ as in `volrender/`.

## Goal of the day

After the day a student can:

1. describe the **pinhole camera** as $P = K[R \mid \mathbf t]$: what each of the 11 numbers
   means (focal length in pixels, principal point, skew/aspect; rotation and position), project
   a point by hand, turn a pixel back into a **ray**, and recognise **lens distortion**;
2. **calibrate** a camera from a known object (direct linear transform, then refine by
   minimising the **reprojection error**), say how many points are needed and which
   configurations are degenerate;
3. explain **epipolar geometry**: why a point in one image constrains its match to a **line**
   in the other, what the **essential** and **fundamental** matrices encode, estimate $F$ from
   point matches with the (normalised) **eight-point algorithm** inside **RANSAC**, and
   **triangulate** a point from two rays, predicting how the error grows with depth;
4. compute **depth from disparity** ($Z = fB/d$) for a **rectified** stereo pair, choose a
   **matching cost** and window, and explain the typical failures (textureless regions,
   repetitive patterns, occlusions, slanted surfaces) and what global methods (scanline
   dynamic programming, semi-global matching, graph cuts) fix;
5. outline **structure from motion** (initial pair, adding cameras, triangulating, repeating)
   and state **bundle adjustment** as one nonlinear least-squares problem over all cameras and
   points; take a Gauss–Newton step by hand and explain the **gauge freedom** (scale!) and the
   sparsity that makes it tractable;
6. explain **dense** reconstruction from many calibrated views: **multi-view stereo** by
   sweeping depths and testing photo-consistency, **neural radiance fields** (the 3D graphics
   day's emission–absorption rendering, fitted to photographs), and **3D Gaussian splatting**
   (sorted Gaussians composited front-to-back, the same "over" as volume rendering);
7. *(math)* derive $\mathbf x'^\top E\,\mathbf x = 0$ from coplanarity, $F = K'^{-\top}EK^{-1}$,
   the DLT and eight-point systems as null-space problems solved by SVD, the rank-2
   projection, $\delta Z \approx Z^2\delta d/(fB)$, the Gauss–Newton/LM normal equations and
   the Schur complement, and the projected covariance $J W \Sigma W^\top J^\top$ of a splat.

Why it fits this audience: it is linear algebra used in earnest. Homogeneous coordinates
(`images/`) and $4\times4$ cameras (`pipeline3d/`) are the starting point; the estimation
problems are null spaces (SVD), cross products as matrices, least squares and its
nonlinear extension. The one genuinely new idea is that **geometry comes from
correspondences**: once a point is matched in two views, its depth is a triangle. The
image-analysis payoff for a science cohort: height maps from SEM tilt pairs, tilt-series
alignment with fiducials (a bundle adjustment), extended depth of field and height from focus
stacks, photogrammetry of specimens and plants, and the camera pose estimates behind every
modern 3D capture.

## Prerequisite gap

The 3D group is entered after Part I, so the core days after Part I (thresholding onwards)
may not have been taken. Everything used from them is recalled self-contained in one slide.

- **Homogeneous coordinates, affine transforms, inverse mapping, bilinear interpolation**:
  `images/` (day 1). Recalled in unit 1 (one line) and unit 3 (rectification is an inverse
  mapping, `images/` "Computing the output: go backwards").
- **The camera chain, look-at as a change of basis, perspective by similar triangles,
  "from pixels back to rays"**: `pipeline3d/` (3D graphics day, unit 1). Unit 1 here opens
  from its last content slide ("From pixels back to rays": one camera loses depth, two do
  not) and re-reads the chain as the vision camera $K[R \mid \mathbf t]$.
- **Emission–absorption, transmittance, front-to-back compositing, early termination**:
  `volrender/` (3D graphics day, unit 4) and premultiplied "over" (`gpuformats/`). Unit 5
  recalls them in one slide each and does not re-derive them.
- **Interest points (Harris/Förstner corners), DoG blobs over scales**: `structure-tensor/`
  and `scale-space/` (edges day, core day 2). Unit 2 recalls "corners can be found again".
- **NCC** (normalised cross-correlation): owned by FFT registration (`matching/`, core day
  12), which may come later. Units 2 and 3 give a one-slide self-contained definition with the
  pointer.
- **RANSAC**: owned by the Hough day (`ransac/`, elective "Detection"), which may not have
  been taken. Unit 2 has a two-slide recall (guess and check; $N = \log(1-p)/\log(1-w^s)$)
  and embeds the built `ransac` demo with `data-preset="seventeen"`.
- **Least squares** (normal equations): linear algebra (audience.md). The built
  `stitching/` math slide "least squares on a graph" (tile positions from pairwise shifts) is
  the linear little sibling of bundle adjustment; unit 4 points to it if day 12 was taken.
- **Gradient descent, a small neural network**: neural networks day (`neuralnet/`,
  `backprop/`, core day 15), possibly not yet taken. Unit 5 says in one slide what it needs
  ("a flexible function with many parameters, fitted by walking downhill on the error") and
  the flatland demo uses a voxel grid instead of a network, so nothing depends on day 15.
- **Defocus PSF**: `psf/` (image formation, core day 4), slide "Other blurs, other PSFs".
  The optional `depth-focus/` unit recalls it.
- **Dynamic programming on a path**: `dp-paths/` (elective "Curves and energies"), maybe not
  taken. Unit 3's scanline slide explains the table in three lines and points there.

## Overlaps / who owns what

| Topic | Owner | Here |
|---|---|---|
| 2D homogeneous coordinates, affine, inverse mapping | `images/` (day 1) | recall |
| $4\times4$ pipeline, look-at, perspective matrix, rays from pixels | `pipeline3d/` (3D graphics) | recall; unit 1 adds the vision form $K[R\mid\mathbf t]$ and the comparison |
| Pinhole intrinsics/extrinsics, lens distortion, calibration (DLT, planar calibration), reprojection error | **this day**, unit 1 | — |
| Homography (the image of a plane; 4 points; DLT) | **this day**, unit 1 (needed for planar calibration, rectification and plane sweep) | Motion elective (planned; panoramas, feature-based registration) recalls it; `ransac/` "Where RANSAC is standard" points here |
| Interest points | `structure-tensor/`, `scale-space/` (edges day) | recall |
| **Matching interest points between images** (patch descriptors, nearest neighbour, ratio test, cross-check) | **this day**, unit 2 (two slides; proposed, see open questions) | Motion elective recalls; `structure-tensor/` "Interest points" names descriptors in one line |
| NCC, template matching | FFT registration `matching/` (day 12) | one-slide self-contained recall |
| RANSAC (algorithm, iteration count, vs Hough) | Hough day `ransac/` | recall; the application to $F$ is here |
| Epipolar geometry, $E$, $F$, eight-point, triangulation | **this day**, unit 2 | — |
| **Stereo**: rectification, disparity → depth, matching costs, cost aggregation, scanline DP and semi-global matching | **this day**, unit 3 (course map ownership change) | dynprog `dtw/`: its "scanline stereo" slides and the `dtw` `data-mode="stereo"` become a one-slide pointer here; graph cuts `graphcut-seg/` "same construction in stereo" line points here |
| DP on a monotone path, `lib/dp.ts` | dynprog (`dp-paths/`) | used by unit 3's scanline aggregation |
| Graph cuts, α-expansion | graph cuts (day 13) | unit 3: one-line pointer (stereo as multi-label energy) |
| Least squares on a graph (global tile alignment) | FFT registration `stitching/` | unit 4: recall as the linear special case |
| Structure from motion, bundle adjustment, gauge freedom | **this day**, unit 4 | Motion elective (non-rigid/feature registration) and the planned tracking day may recall |
| Multi-view stereo (plane sweep, photo-consistency) | **this day**, unit 4 (concept) and unit 5 (`flatland` sweep mode) | — |
| Emission–absorption, compositing, opacity correction | `volrender/` (3D graphics) | unit 5 recalls: NeRF = DVR with a learned field; splatting = sorted "over" |
| Neural radiance fields, 3D Gaussian splatting | **this day**, unit 5 | modern deep learning (day 18) may point here; ray tracing elective (planned) contrasts rasterisation/splatting/ray marching |
| Gradient descent, MLPs | neural networks (day 15) | unit 5: one-slide self-contained version |
| Defocus PSF | image formation `psf/` | optional `depth-focus/` recalls |
| Focus measures (local variance of the Laplacian), extended depth of field | **this day**, optional `depth-focus/` | point-detection and deconvolution days may point here |
| Laplacian, local variance | `kernels/` (day 1), integral images elective (`localStats`) | used in `depth-focus/` |

**Pending edits to other plans and built units** (not made by this plan; listed for the
integration pass):

- `plan/dynprog-day.md` unit 6 `dtw/`: replace the "Scanline stereo" bullet and the
  `data-mode="stereo"` demo mode by a one-slide application pointer to `stereo/` here
  (open question there is thereby decided: keep one slide, no demo mode).
- `pipeline3d/lec.qmd` "From pixels back to rays": "an elective of its own" → link to
  `cameras/`.
- `ransac/lec.qmd` "Where RANSAC is standard": the two-view bullet links to `two-view/`; the
  homography bullet mentions that the homography is taught in `cameras/`.
- `structure-tensor/lec.qmd` "Interest points": add "matched between views in `two-view/`
  (3D group)" next to the motion pointer.
- `graphcut-seg/lec.qmd` (line "same graph construction appears in stereo matching") → link
  to `stereo/`.
- `volrender/lec.qmd` summary: one "where this leads" line to `radiance-fields/`.
- `psf/lec.qmd` "Other blurs, other PSFs": optional pointer to `depth-focus/`.
- `plan/coursemap.md`: move the "3D: geometry from images" row from "requested" to the 3D
  group (marching cubes → 3D graphics → geometry from images), and add owner rows
  (homography; matching interest points; stereo; bundle adjustment) if accepted.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units:
`cameras/`, `two-view/`, `stereo/`, `sfm/`; optional `radiance-fields/` (recommended, see
open questions) and `depth-focus/`.

### 1. `cameras/`: The pinhole camera and calibration

Concept layer:

- Recall (`pipeline3d/`, last slide): a pixel is a ray; one camera loses depth. Today's day:
  get it back from several images. This unit: the camera itself, exactly enough to measure
  with it.
- **The pinhole camera**: light through one point onto a sensor; similar triangles give
  $u = f X/Z$. In **pixels**: $f$ in pixels = focal length / pixel pitch (8 mm and 10 µm →
  800 px); the **principal point** $(c_x, c_y)$ where the optical axis meets the sensor
  (near, not at, the image centre); aspect ratio and skew (≈ 1 and 0 for modern sensors; one
  line). These five numbers are the **intrinsics** $K$: properties of camera + lens, fixed
  when the zoom and focus are fixed.
- **Extrinsics**: where the camera is and where it looks: a rotation $R$ and translation
  $\mathbf t$ from world to camera coordinates (the look-at of `pipeline3d/`, now unknown
  rather than chosen). Camera centre $\mathbf c = -R^\top\mathbf t$.
- **Vision vs graphics cameras** (one comparison slide): graphics builds $P_\text{GL} V M$
  ($4\times4$, keeps a depth for the z-buffer, maps to NDC); vision writes
  $P = K[R \mid \mathbf t]$ ($3\times4$, straight to pixels, depth divided away). Same
  similar triangles; axes flipped ($+z$ forward, $y$ down).
- **Pixel → ray**: direction $R^\top K^{-1}(u, v, 1)^\top$ from $\mathbf c$: the inverse
  question, and the starting point of units 2–5.
- **Lens distortion**: real lenses bend straight lines (barrel, pincushion); a radial model
  $r_d = r(1 + k_1r^2 + k_2r^4)$ in normalised coordinates, applied after projection and
  before $K$. Undistorting is an inverse mapping with interpolation (`images/`). Picture: a
  checkerboard with curved lines, and the residual pattern a missing $k_1$ leaves.
- **Calibration**: photograph an object with known geometry (a checkerboard, a grid of dots,
  or beads/fiducials in microscopy); find its points in the image; solve for $K$, $R$,
  $\mathbf t$, $k_1, k_2$.
  - **From a 3D object (DLT)**: each point gives two linear equations in the 12 entries of
    $P$; 11 unknowns (scale) → at least 6 points, not all on one plane. Then split $P$ into
    $K$ and $R$.
  - **From a flat checkerboard in several poses** (Zhang 2000): one plane gives a
    **homography** (the image of a plane is a $3\times3$ map: 8 numbers, 4 points); each
    view's homography gives two constraints on $K$; three or more views fix it.
  - Linear first, then **refine**: minimise the **reprojection error** (pixel distance between
    observed and predicted points) over all parameters, distortion included. Reported RMS of
    a good calibration: a few tenths of a pixel. Residual arrows that form a pattern (all
    pointing outward at the corners) mean a missing model term, not noise.
- Degenerate and weak setups: all 3D points on one plane (DLT fails), all checkerboard views
  parallel (focal length and distance trade off), points only in the centre (distortion
  unconstrained). Pointer: in microscopy the "camera" is often close to orthographic
  (telecentric); SEM tilt series and light-sheet multi-view use the same ideas with an affine
  camera (one line).

The math (`{.math}` slides after the corresponding concept slides):

- *The camera matrix*: $\mathbf x \simeq K[R \mid \mathbf t]\,\mathbf X$ with
  $K = \begin{bmatrix} f_x & s & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix}$; 11 degrees of
  freedom (5 + 3 + 3). Relation to the graphics matrices: $V = [R \mid \mathbf t]$ with the
  axis flip; $P_\text{GL}$'s first two rows are $K$'s up to the NDC scale and shift.
- *DLT*: from $\mathbf x_i \times P\mathbf X_i = 0$, two independent rows per point:
  $A\mathbf p = 0$, $A \in \mathbb R^{2n\times12}$; minimise $\lVert A\mathbf p\rVert$ subject
  to $\lVert\mathbf p\rVert = 1$ → the right singular vector of the smallest singular value
  (`lib/linalg.ts` `svd`, `nullSpace`). **Normalisation** (Hartley): translate and scale the
  points to mean 0, RMS distance $\sqrt2$ first, or the condition number explodes (pixel
  coordinates ~1000 next to 1). $P = K[R \mid \mathbf t]$ back by RQ decomposition of the
  left $3\times3$.
- *Homography and planar calibration*: for $Z = 0$, $\mathbf x \simeq K[\mathbf r_1\;\mathbf r_2\;\mathbf t]\,(X, Y, 1)^\top = H(X, Y, 1)^\top$;
  orthonormality of $\mathbf r_1, \mathbf r_2$ gives
  $\mathbf h_1^\top\omega\mathbf h_2 = 0$, $\mathbf h_1^\top\omega\mathbf h_1 = \mathbf h_2^\top\omega\mathbf h_2$
  with $\omega = K^{-\top}K^{-1}$ (two linear equations per view in the 6 entries of $\omega$;
  Cholesky gives $K$).
- *Reprojection error*: $\min \sum_i \lVert\mathbf x_i - \pi(K, R, \mathbf t, k; \mathbf X_i)\rVert^2$,
  nonlinear least squares: solved like bundle adjustment (unit 4), forward pointer.

Demos:

- **Demo `pinhole`** (new; one WebGL viewport via `lib/gl3d.ts` + an SVG image panel, about
  560 + 560 px): left, a 3D scene (a checkerboard, a box, a few posts) with the camera drawn as
  a pyramid and one selected ray; right, the image in **pixels** with the sensor frame and the
  principal point. Sliders: focal length (px), $c_x, c_y$, skew, camera position (orbit drag)
  and roll, distortion $k_1, k_2$. Readouts: $K$, $[R \mid \mathbf t]$, $P$ (3 decimals) and,
  for a clicked 3D point, its camera coordinates and pixel; click a pixel → its ray drawn in
  3D. Toggle *vision / graphics convention* (shows the axis flip and $P_\text{GL}$ for the same
  camera, reusing `lib/mat4.ts`). Presets: "focal length vs moving closer (dolly zoom
  recall)", "principal point off centre", "barrel distortion", "pincushion", "a point behind
  the camera", "telecentric (very long focal length, far away)".
- **Demo `dltcalib`** (new; SVG, CPU): a synthetic calibration object (two orthogonal
  checkerboard planes, or one plane in several poses) imaged by a hidden camera with seeded
  pixel noise and optional distortion. Students pick points (click to add/remove from the
  set) or use *all*. Methods: *DLT unnormalised*, *DLT normalised*, *+ refine (LM, step)*;
  for the planar mode *homographies → K*. Shows estimated vs true $K$ (table with
  differences), the reprojection residuals as arrows magnified ×50, the RMS error, and a
  bar of singular values of $A$ (the null space visible as one tiny value). Presets: "6 points
  exactly", "noise 0.5 px, 40 points", "unnormalised DLT", "all points on one plane:
  degenerate", "distortion ignored: the residual pattern", "planar: 1, 2, 3 views",
  "parallel views: focal length undetermined".

By hand (exercise; concept questions first):

- $K$ with $f = 800$, $(c_x, c_y) = (320, 240)$; $R = I$, $\mathbf t = 0$: project
  $(0.5, -0.25, 5)$ → $(400, 200)$; the same point at $Z = 10$ → $(360, 220)$. Which pixels
  do not move when $Z$ changes? (Only the principal point.)
- The ray of pixel $(400, 200)$: $K^{-1}(400, 200, 1)^\top = (0.1, -0.05, 1)$.
- Horizontal field of view of a 640-px-wide sensor with $f = 800$: $2\arctan(320/800) = 43.6°$.
- Lens: 8 mm focal length, 3.45 µm pixels → $f$ in pixels (2319). What happens to $f$ (px)
  with 2×2 binning?
- Radial distortion $k_1 = -0.1$ at normalised radius 0.4: the point moves to
  $0.4(1 - 0.016) = 0.3936$; barrel or pincushion?
- Count: unknowns and equations for DLT with 5 and with 6 points; why 6 coplanar points fail.
- Read a residual-arrow picture: noise or a missing model term?
- **B7.** *(math)* Derive the two DLT rows from $\mathbf x \times P\mathbf X = 0$.
- **B8.** *(math)* Show $\mathbf c = -R^\top\mathbf t$ is the null vector of $P$.
- **B9.** *(math)* Derive the two planar-calibration constraints on $\omega$ from
  $\mathbf r_1^\top\mathbf r_2 = 0$ and $\lVert\mathbf r_1\rVert = \lVert\mathbf r_2\rVert$.

### 2. `two-view/`: Two views: matches, epipolar geometry and triangulation

Concept layer:

- **Correspondences first.** Geometry needs the same point in two images. Interest points
  (recall `structure-tensor/`: corners can be found again) plus a **description** of the
  window around each: a normalised patch compared by NCC (one-slide self-contained recall of
  `matching/`), or a histogram of gradient orientations, which survives small rotations and
  lighting changes (named descriptors in one line, no tool names). **Matching**: nearest
  neighbour in descriptor space; the **ratio test** (best clearly better than second best)
  and the **cross-check** (each is the other's best) throw out ambiguous matches.
  Repetitive texture still produces confident wrong matches → outliers.
- **Epipolar geometry**: the two camera centres and a 3D point span a plane; it cuts each image
  in a line. So the match of a point in image 1 lies on a known **epipolar line** in image 2:
  a 2D search becomes 1D. All epipolar lines pass through the **epipole** (the image of the other
  camera). Pictures for three motions: sideways (lines parallel, epipoles at infinity), forward
  (lines radiate from the **focus of expansion**), general.
- **Essential and fundamental matrices**: the constraint is one $3\times3$ matrix:
  $\mathbf x'^\top F\mathbf x = 0$. $E$ for calibrated cameras (depends only on $R$, $\mathbf t$;
  5 degrees of freedom), $F$ in pixels (7 degrees of freedom; includes both $K$s). $F\mathbf x$
  is the epipolar line of $\mathbf x$.
- **Estimating $F$** from 8 or more matches (the eight-point algorithm: linear, then force rank
  2); normalise coordinates first (same reason as DLT). With outliers: **RANSAC** (two-slide
  recall from `ransac/` + the `ransac` demo): sample 8, fit, count matches within a distance of
  their epipolar line. With $w = 0.5$ inliers, $p = 0.99$: $s = 8$ needs 1177 samples, the
  seven-point algorithm 588, the five-point algorithm for $E$ (calibrated) 146: why minimal
  solvers matter.
- **Degenerate cases**: pure rotation (no baseline: no depth, no epipolar geometry, the views
  are related by a homography: panoramas), all points on one plane (also a homography; $F$ is
  not determined).
- **From $E$ to the cameras**: $E$ gives $R$ and the **direction** of $\mathbf t$ (four
  candidate solutions; the right one puts the points in front of both cameras). The **scale is
  lost**: from images alone, a scene twice as large and twice as far looks identical. Scale
  comes from a known length (a scale bar, a baseline, a calibration object).
- **Triangulation**: two rays from matched pixels should meet; with noise they miss. Midpoint
  of the shortest segment (simple, biased), linear DLT, or the point whose reprojections are
  nearest the observations (optimal). **Uncertainty**: a pixel of error is a wedge; the wedges
  intersect in a long thin quadrilateral; error grows with **depth²** and shrinks with baseline;
  wide baseline → better depth but harder matching. A third view shrinks the region.

The math:

- *Coplanarity*: $\mathbf x' \cdot (\mathbf t \times R\mathbf x) = 0 \Rightarrow \mathbf x'^\top[\mathbf t]_\times R\,\mathbf x = 0$,
  $E = [\mathbf t]_\times R$ (normalised coordinates $K^{-1}\mathbf x$); $F = K'^{-\top}EK^{-1}$;
  $\det F = 0$; epipoles $F\mathbf e = 0$, $F^\top\mathbf e' = 0$. $E$ has two equal singular values
  and one zero.
- *Eight-point*: each match gives one linear equation $\mathbf a_i^\top\mathbf f = 0$ in the 9
  entries; SVD null vector; set the smallest singular value of $F$ to 0 (closest rank-2 matrix
  in Frobenius norm); undo the normalisation $F = T'^\top\hat FT$. Sampson distance as the
  first-order geometric error used for inlier counting (one formula).
- *Decomposing $E$*: $E = U\,\mathrm{diag}(1, 1, 0)V^\top$; $R = UWV^\top$ or $UW^\top V^\top$,
  $\mathbf t = \pm\mathbf u_3$; cheirality picks one.
- *Triangulation*: linear DLT from $\mathbf x \times P\mathbf X = 0$, $\mathbf x' \times P'\mathbf X = 0$
  (4 equations, null vector); midpoint formula; error propagation for the parallel case leading
  to unit 3's $\delta Z$.
- *RANSAC iterations*: recall $N = \log(1-p)/\log(1-w^s)$ with the numbers above.

Demos:

- **Demo `epipolar`** (new; two SVG image panels + a small 3D overview via `lib/gl3d.ts`):
  a synthetic scene (points on a house shape, a ground plane, a few posts) seen by two cameras.
  Click a point in the left image → its epipolar line in the right image, the true match on it,
  and the epipolar plane in the 3D view; epipoles marked (with an arrow when at infinity).
  Sliders: baseline, rotation between the cameras, focal lengths. **Estimate mode**: matches
  with seeded noise $\sigma$ (px) and outlier fraction; method *8-point unnormalised /
  8-point normalised / RANSAC + 8-point*; estimated epipolar lines drawn over the true ones,
  inliers/outliers coloured, residual histogram, $F$ and its singular values. Presets:
  "sideways: parallel lines", "forward: focus of expansion", "general motion", "pure rotation:
  no geometry", "planar scene: degenerate", "no normalisation", "20 % outliers break
  least squares", "RANSAC repairs it".
- **Demo `triangulate`** (new; SVG, flatland view: a top-down 2D world with 1D cameras, from
  `lib/flatland.ts`): two cameras (drag position and heading), one point (drag), each camera's
  pixel row with the observed pixel and a ±$\delta$ px wedge; the uncertainty quadrilateral
  shaded; methods *midpoint / optimal* with seeded noise repeated 200 times as a cloud.
  Side plot: depth error vs depth (with the $Z^2$ curve) and vs baseline. Toggle a third
  camera. Presets: "short baseline: a long thin region", "wide baseline", "point far away",
  "nearly parallel rays", "a third view".

By hand:

- Sideways motion: $R = I$, $\mathbf t = (1, 0, 0)$: write $[\mathbf t]_\times$ and show
  $\mathbf x'^\top E\mathbf x = y - y' = 0$: epipolar lines are rows. Forward motion
  $\mathbf t = (0, 0, 1)$: $-x'y + y'x = 0$: lines through the origin.
- Given $F$ (integers) and a point, compute its epipolar line and check which of three
  candidate points is a possible match.
- RANSAC iterations for $s = 8, 7, 5$ at $w = 0.5$ and $w = 0.8$, $p = 0.99$.
- Ratio test by hand: descriptor distances 0.30/0.32 vs 0.20/0.60 with threshold 0.8.
- Two parallel cameras, $f = 800$ px, $B = 0.1$ m: a point seen at $x = 420$ and $x' = 400$
  ($c_x = 320$) is at $Z = 800 \cdot 0.1/20 = 4$ m and $X = 0.5$ m; with a 1 px error: 3.81–4.21 m.
- A scene and the same scene scaled ×2 and moved ×2 farther: same images? What would fix the
  scale?
- **B7.** *(math)* Derive the epipolar constraint from coplanarity.
- **B8.** *(math)* Show that the epipole is the null vector of $F$ and that every epipolar line
  passes through it.
- **B9.** *(math)* Why the closest rank-2 matrix is obtained by zeroing the smallest singular
  value (Eckart–Young, statement and use).

### 3. `stereo/`: Stereo: from disparity to depth

Concept layer:

- **Rectification**: warp both images so that epipolar lines become the same image rows; then
  matching is a 1D search along a row (each image is resampled by a homography: inverse
  mapping + bilinear, `images/`). Calibrated stereo rigs are nearly rectified by construction.
- **Disparity → depth**: for rectified cameras $d = x - x'$ and $Z = fB/d$. Near things shift
  a lot, far things little; $d = 0$ is infinitely far. **Depth resolution**:
  $\delta Z \approx Z^2\delta d/(fB)$; sub-pixel disparity (parabola through three costs; recall
  `stitching/` "Sub-pixel shifts" if taken) is what makes stereo useful at a distance.
- **Matching costs** in a window: SAD/SSD (fast, fooled by a gain difference between cameras),
  NCC (gain and offset invariant), **census** (compares only the order of pixels with the
  centre: robust to any monotone brightness change, the standard in practice). The **cost
  volume** $C(x, y, d)$: one slice per disparity.
- **Window size**: small windows are noisy; large windows blur depth edges (**foreground
  fattening**: the window straddles a depth edge and the textured foreground wins) and fail on
  slanted surfaces. Winner-take-all per pixel.
- **Failures**: textureless regions (all disparities equally good), repetitive patterns
  (several equally good), **occlusions** (a pixel seen in one image only, at every depth
  edge), specular highlights. **Left–right check**: match both ways, keep pixels whose
  disparities agree; the rest is marked invalid (holes to fill, never "depth 0").
- **Global methods**: add smoothness, the familiar data + λ·smoothness energy (recall
  `seg-energy/` in one line, with disparity as a multi-label labelling).
  - **Scanline dynamic programming**: per row, the optimal monotone path through the
    $x \times d$ cost slice; occlusions as steps with a fixed cost (Ohta & Kanade 1985, the
    former dynprog slide; recall `dp-paths/`'s table in three lines). Rows are solved
    independently → horizontal **streaks**.
  - **Semi-global matching** (Hirschmüller 2008): the same 1D DP along 8 (or 4) directions,
    costs summed; fast, few streaks, the standard in practice.
  - Graph cuts with α-expansion: one-line pointer to the graph-cuts day.
  - Learned matching costs and end-to-end networks: one line (they still build a cost volume).
- **Active stereo and structured light**: project a pattern to give texture to textureless
  surfaces (one slide; depth cameras). **Microscopy**: SEM tilt pairs (the parallax of a tilt
  gives height), stereo endoscopy and stereo microscopes; one slide with the parallax formula
  for a tilt (as an affine camera).
- One-slide pointer: depth from focus and defocus (optional `depth-focus/`).

The math:

- *Rectified geometry*: $x = fX/Z$, $x' = f(X - B)/Z$ ⇒ $d = fB/Z$; $\partial Z/\partial d = -fB/d^2 = -Z^2/(fB)$.
- *Rectifying homographies*: rotate both cameras to a common orientation with the $x$ axis
  along the baseline: $H = K_\text{new}R_\text{rect}R^\top K^{-1}$ (pure rotation of a camera is
  a homography, from unit 2's degenerate case).
- *Costs*: $\mathrm{SAD}$, $\mathrm{NCC}$, census as Hamming distance of bit strings; window sums
  in $O(1)$ per pixel with integral images (pointer to the integral-images elective, or one line).
- *Scanline DP / SGM*: $L_r(\mathbf p, d) = C(\mathbf p, d) + \min\big(L_r(\mathbf p - \mathbf r, d),\, L_r(\mathbf p - \mathbf r, d \pm 1) + P_1,\, \min_k L_r(\mathbf p - \mathbf r, k) + P_2\big) - \min_k L_r(\mathbf p - \mathbf r, k)$;
  $S = \sum_r L_r$; winner-take-all on $S$. $P_1$, $P_2$ play the role of λ (small and large
  jumps).
- *Sub-pixel*: parabola vertex $\delta = (C_{-} - C_{+})/(2(C_{-} - 2C_0 + C_{+}))$.

Demos:

- **Demo `stereomatch`** (new, main demo; canvas): a rectified pair, default 192×144 with up
  to 48 disparities. Scenes (seeded, with ground-truth disparity, generated in
  `lib/stereo.ts` by rendering textured planes): *random-dot stereogram (a raised square)*,
  *slanted plane + textureless wall + striped fence + occluding post*, *gain change between the
  cameras*, and a real pair if one is licensed (open questions). Panels: left image (hover
  shows the window), right image (with the matched window and the epipolar row), disparity map
  (colour map from `lib/colour.ts`, invalid in grey), error map vs truth with the fraction of
  bad pixels (> 1 px). Controls: cost *SAD / NCC / census*, window size, max disparity,
  aggregation *none (WTA) / scanline DP / SGM 4 / SGM 8*, $P_1$, $P_2$, *left–right check*,
  *sub-pixel*, noise. A click on a pixel shows its cost curve $C(d)$ (with WTA minimum, the
  true $d$, and the parabola). Second tab **`data-tab="slice"`**: one row's $x \times d$ cost slice as a
  heat map with the WTA path and the DP path (step-through via the `dpgrid` style of
  `lib/gridui.ts`, from `lib/dp.ts`), occlusion cost slider. Presets: "window 3 vs 15",
  "textureless wall", "striped fence: repeated", "occlusion: left–right check", "gain
  change: SAD fails, census copes", "scanline DP: streaks", "SGM", "sub-pixel on the slanted
  plane".
- Static figure: the disparity-to-depth curve $Z(d)$ with the ±1 px bands (print fallback).

By hand:

- 1D SAD: left row $(1, 1, 5, 9, 5, 1, 1, 1)$, right row shifted by 2; window 3 at the 9: costs
  for $d = 0, 1, 2$ (12, 12, 0). Then a constant row: why every $d$ ties.
- Census transform of a 3×3 window; Hamming distance between two given windows; change all
  values by ×2 + 10 and recompute (unchanged).
- Depth from disparity with $f = 1000$ px, $B = 0.12$ m for $d = 60, 6, 5$ px; the depth step
  between $d = 6$ and $d = 5$ (20 → 24 m) vs between 60 and 59.
- Why does the post look fatter in the disparity map with a 15×15 window? Sketch.
- A 4×5 $x \times d$ cost slice: winner-take-all per column vs the DP path with occlusion cost 2.
- **B7.** *(math)* Derive $\delta Z \approx Z^2\delta d/(fB)$; for a fixed depth error, how must
  $B$ grow with $Z$?
- **B8.** *(math)* Show census is invariant under any strictly increasing intensity map.
- **B9.** *(math)* Show the SGM recursion with $P_1 = P_2 = 0$ reduces to winner-take-all on
  summed costs, and with one direction to scanline DP.

### 4. `sfm/`: Many views: structure from motion and bundle adjustment

Concept layer:

- **The problem**: a set of photographs, no calibration object, no known poses. Find the cameras
  **and** the 3D points at the same time. (Applications: photogrammetry of objects and
  landscapes, plant phenotyping, the camera poses behind unit 5's methods; in microscopy,
  **tilt-series alignment** with fiducial beads is the same problem with an affine camera.)
- **Incremental pipeline**: match all pairs (unit 2) → pick a good initial pair (many matches,
  real baseline) → $E$, two cameras, triangulate → **add a camera** whose matches see already
  reconstructed points: **resection** (the unit 1 DLT again, now with the triangulated points as
  the "calibration object"; pose from 3+ points for a calibrated camera, named) → triangulate new
  points → **bundle adjust** → repeat. Global methods (all rotations first) named in one line.
- **Bundle adjustment**: one least-squares problem: minimise the total reprojection error over all
  camera parameters and all points. "Bundle": the bundle of rays from each point to the cameras,
  adjusted until they meet the observations. It is the nonlinear big sibling of the global tile
  alignment in `stitching/` (pairwise shifts → positions, one linear solve), and of unit 1's
  refinement.
- **Gauss–Newton and Levenberg–Marquardt**, in pictures: linearise the residuals, solve the
  linear least squares, step, repeat; LM adds damping $\mu$: large $\mu$ = small careful
  gradient-descent step, small $\mu$ = full Gauss–Newton step; $\mu$ adapts.
- **Gauge freedom**: the whole reconstruction can be moved, rotated and scaled without changing
  any image: 7 directions with zero cost (6 for calibrated stereo with a known baseline). Fix
  them (first camera, one distance) or the solver wanders. Same idea as fixing one tile in
  `stitching/` ($L\mathbf 1 = 0$).
- **Sparsity**: each observation involves one camera and one point; the Jacobian is almost
  empty; eliminating the points first leaves a small system over the cameras (the Schur
  complement). Why it scales to thousands of images.
- **Robustness**: a few wrong matches dominate a squared error; robust losses (Huber) down-weight
  them (one line, link to RANSAC).
- **Drift and loop closure**: chaining poses accumulates error, exactly the chain-of-tiles
  picture of `stitching/`; seeing an old part of the scene again closes the loop.
- **Uncertainty**: after BA, the inverse of the (gauge-fixed) normal matrix gives covariances:
  points far from all cameras get long ellipses along the viewing direction (unit 2's wedges).
- **Dense reconstruction (multi-view stereo)**, two slides: with the cameras known, every pixel
  can be matched: **plane sweep** (for each depth hypothesis, warp the other images onto the
  reference by the plane's homography, score photo-consistency, keep the best depth per pixel)
  → depth maps → fused into a point cloud or a volume → a mesh (marching cubes, recalled from
  the 3D group). Unit 5 continues this thread with radiance fields.

The math:

- *The objective*: $\min_{\{P_j\},\{\mathbf X_i\}}\sum_{(i,j)\in\mathcal O}\rho\big(\lVert\mathbf x_{ij} - \pi(P_j, \mathbf X_i)\rVert^2\big)$;
  camera parameters: rotation (axis-angle, 3), translation (3), optionally $f$, $k_1$, $k_2$.
- *Gauss–Newton*: $\mathbf r(\boldsymbol\theta + \boldsymbol\delta) \approx \mathbf r + J\boldsymbol\delta$
  ⇒ $J^\top J\,\boldsymbol\delta = -J^\top\mathbf r$; LM: $(J^\top J + \mu D)\boldsymbol\delta = -J^\top\mathbf r$;
  accept/reject and adapt $\mu$.
- *Schur complement*: $\begin{bmatrix}U & W\\ W^\top & V\end{bmatrix}\begin{bmatrix}\boldsymbol\delta_c\\ \boldsymbol\delta_p\end{bmatrix} = \begin{bmatrix}\mathbf g_c\\ \mathbf g_p\end{bmatrix}$,
  $V$ block-diagonal (3×3 per point): $(U - WV^{-1}W^\top)\boldsymbol\delta_c = \mathbf g_c - WV^{-1}\mathbf g_p$.
- *Gauge*: the null space of $J$ (a similarity has 7 parameters); covariance
  $\Sigma = \sigma^2(J^\top J)^{+}$ or with fixed gauge.
- *Plane-induced homography*: $H = K'(R - \mathbf t\mathbf n^\top/d)K^{-1}$ for the plane
  $\mathbf n^\top\mathbf X = d$.

Demos:

- **Demo `bundleadjust`** (new; SVG, flatland: a 2D world with 1D cameras, from
  `lib/flatland.ts`, so every point, camera, ray and residual is visible at once): 5–8 cameras
  around 30–60 points (an outline of an object, seeded). Initial guess = truth + seeded
  perturbation (slider), or the result of the **incremental** mode (step: initial pair →
  triangulate → resect next camera → … , each step shown). Buttons *GN step*, *LM step*, *run*,
  *reset*; damping $\mu$ readout; plot of RMS reprojection error per iteration (log scale);
  residuals drawn in each camera's 1D image; gauge options *fix camera 1 + scale / fix nothing*
  (the solution drifts and $J^\top J$'s smallest eigenvalues are shown as ~0); outlier matches
  toggle with *squared / Huber* loss; covariance ellipses of the points after convergence;
  sparsity pattern of $J$ and $J^\top J$ as a small heat map (`lib/plot.ts` `heatmap`).
  Presets: "small perturbation: converges in 3 steps", "large perturbation: GN diverges, LM
  copes", "no gauge fixed", "an outlier with squared loss", "Huber", "points far away: long
  ellipses", "incremental reconstruction", "loop closure: drift".
- **Demo `flatland`** with `data-mode="sweep"` (shared with unit 5): plane (line) sweep in
  flatland: a reference camera's pixel, depth hypotheses along its ray, the other cameras' pixels
  for each hypothesis, photo-consistency (variance across views) vs depth; the depth map for all
  pixels; presets "textured object", "textureless face", "occluded in one view".

By hand:

- Count unknowns and equations: 10 cameras (6 parameters each), 1000 points, each seen in 4
  images; nonzeros per Jacobian row; size of the reduced camera system (60×60).
- Gauss–Newton in one unknown: a point at $X = 1$ observed at normalised $x = 0.25$, unknown
  depth $Z$, residual $r(Z) = 0.25 - 1/Z$; from $Z_0 = 3$: $Z_1 = 3.75$, $Z_2 \approx 3.98$
  (truth 4).
- Which of these do not change any image: rotate everything, scale everything, move one camera?
- Order an incremental reconstruction of 5 views from a table of match counts.
- Plane sweep for one pixel by hand: three hypotheses, the other image's values given, pick
  the best.
- **B7.** *(math)* Derive the Gauss–Newton step from the linearised residual.
- **B8.** *(math)* Derive the Schur complement system and count its cost vs the full system.
- **B9.** *(math)* Show that the BA objective is invariant under a similarity transform of
  all points and cameras; hence 7 zero eigenvalues of $J^\top J$.

### 5. (optional, recommended) `radiance-fields/`: Seeing from new viewpoints: radiance fields and Gaussian splatting

- **The question changes**: not "where are the surfaces?" but "what would a new photograph
  look like?" (novel view synthesis). Input: many photographs + their cameras from unit 4.
- Recall (`volrender/`, one slide): along a ray, $C = \sum_i \mathcal T_i\alpha_i c_i$,
  $\mathcal T_{i+1} = \mathcal T_i(1 - \alpha_i)$: front-to-back "over".
- **Neural radiance fields** (Mildenhall et al. 2020): represent the scene as a **density**
  $\sigma(\mathbf x)$ and a colour $c(\mathbf x, \text{direction})$ (view-dependent: shine);
  render each pixel by exactly the emission–absorption ray marching of `volrender/`; compare to
  the photograph; change the field to reduce the difference, for all pixels of all photos.
  The field is stored in a small neural network (one slide: a flexible function fitted by
  gradient descent; pointer to `neuralnet/` and `backprop/`) or in a voxel grid (the same idea,
  no network, what the demo does). Fourier-feature inputs (positional encoding) let a network
  represent fine detail (recall `fourier/` frequencies; one line). Rendering is slow: many
  samples per ray.
- **Why it works and fails**: consistency across many views carves the density; few views or
  wrong camera poses → floaters and blur; the depth is a by-product (expected ray termination).
- **3D Gaussian splatting** (Kerbl et al. 2023;
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/): the scene as a few hundred
  thousand 3D **Gaussian blobs**, each with a position, a covariance (an ellipsoid: scale +
  rotation), an opacity and a view-dependent colour. Render by **projecting** each Gaussian to a
  2D ellipse on the screen, **sorting** by depth, and **compositing front-to-back** with
  $\alpha_i = o_i\exp(-\tfrac12\mathbf d^\top\Sigma_i'^{-1}\mathbf d)$: the same "over" as volume
  rendering, but over a sorted list of blobs instead of samples along a ray. Rasterised, so
  real-time. Started from the sparse points of structure from motion (unit 4), optimised by
  gradient descent on the photo error, with **densification** (split big blobs, clone in
  under-reconstructed areas) and pruning of transparent ones.
- **Comparison table**: MVS mesh / NeRF / Gaussian splats: representation, rendering cost,
  editability, measurable geometry (splats and NeRFs give pictures, not watertight surfaces:
  measuring needs extraction, e.g. marching cubes on the density), failure modes.
- Honest caveat for scientists: these methods produce plausible images of unseen viewpoints;
  where views are missing they fill in, so they are not measurements (link to "hallucinated
  detail" on the modern deep learning day, one line).
- Splats in other places: "splatting" in volume rendering (Westover 1990; footprints instead of
  rays) and point-cloud rendering (`voxelise/` optional unit of 3D graphics): one line.

The math:

- *Volume rendering integral* (recall): $C = \int \mathcal T(t)\sigma(t)c(t)\,dt$,
  discretised with $\alpha_i = 1 - e^{-\sigma_i\delta_i}$; the loss
  $\sum_{\text{pixels}}\lVert C - C_\text{photo}\rVert^2$; its gradient w.r.t. $\sigma_i$, $c_i$
  by the chain rule through the compositing (one sample written out).
- *Projected covariance* (EWA, Zwicker et al. 2001): $\Sigma' = J W\Sigma W^\top J^\top$, with $W$
  the world→camera rotation and $J$ the Jacobian of the perspective map at the centre,
  $J = \begin{bmatrix} f/Z & 0 & -fX/Z^2\\ 0 & f/Z & -fY/Z^2\end{bmatrix}$; $\Sigma = RSS^\top R^\top$
  keeps it positive definite during optimisation.
- *Sorting vs ray order*: compositing by centre depth is exact only when blobs do not overlap
  in depth (popping artefacts when the order flips).

Demos:

- **Demo `flatland`** (new; SVG/canvas, CPU, the main demo of the unit): a 2D world seen by
  8–16 1D cameras arranged around it; the "photographs" are 1D pixel rows shown as colour
  strips beside each camera. Modes: `sweep` (unit 4), **`grid`** (a radiance field on a 64×64
  grid: density + colour per cell, fitted by gradient descent through the emission–absorption
  renderer; shows the field, the rendered vs observed rows, the loss curve and a **novel view**
  from a draggable camera) and **`splat`** (a few dozen 2D Gaussians with opacity and colour,
  initialised from sparse points, projected to 1D, sorted, composited front-to-back; fitted by
  gradient descent with optional densification). *Step / run / reset*, learning-rate and
  number-of-cameras sliders, *view-dependent colour* toggle. Click a pixel of any camera: its
  ray, with $\alpha_i$, $\mathcal T$ and contributions listed (the `raycomposite` picture in
  flatland). Presets: "16 views: converges", "3 views: floaters", "wrong camera pose: blur",
  "splats: densification", "unsorted splats: wrong order", "novel view between the cameras".
- **Demo `splatview`** (new; WebGL2 via a splat primitive in `lib/gl3d.ts`, CPU fallback):
  a precomputed procedural 3D splat scene (a few thousand Gaussians: a cell-like object with
  nucleus and organelles, and a test scene with three overlapping ellipsoids). Drag to orbit.
  Toggles: *sorted / unsorted*, *ellipse outlines*, *centres only (the SfM point cloud)*,
  opacity scale, size scale; click a pixel → the sorted list of splats covering it with
  $\alpha_i$, $\mathcal T$ and contributions (CPU reference). Presets: "points only", "splats",
  "sorting off", "shrink the splats: holes", "one pixel's list".

By hand:

- Composite three splats at one pixel: $\alpha = 0.5, 0.4, 0.8$, colours given: weights 0.5,
  0.2, 0.24, final $\mathcal T = 0.06$; reverse the order.
- A splat's $\alpha$ at a pixel 1 σ and 2 σ from its centre ($o = 0.8$): 0.485, 0.108.
- A spherical Gaussian with σ = 0.1 at depth $Z = 10$, $f = 100$: footprint σ = 1 px; at $Z = 5$?
- Grid field: one ray with 4 samples (σ, c given): rendered colour; which sample's density
  should increase if the photo is brighter?
- **B7.** *(math)* Derive $\partial C/\partial c_i$ and $\partial C/\partial\alpha_i$ for front-to-back
  compositing.
- **B8.** *(math)* Derive $J$ and $\Sigma'$ for a point on the optical axis.

### 6. (optional / further reading) `depth-focus/`: Depth from focus and defocus

- A microscope (or any lens) sharp only at one depth: a **focus stack** (z-stack) holds depth
  information. **Depth from focus**: per pixel, the slice where a **focus measure** (local
  variance of the Laplacian, or of the gradient magnitude, in a window; `kernels/`) is largest;
  parabola around the peak for sub-slice depth. By-product: the **all-in-focus image**
  (extended depth of field), picking each pixel from its sharpest slice.
- Failures: textureless regions (no measure), windows straddling a depth step, noise.
- **Depth from defocus**: from two images with different focus, the blur size gives the
  distance (recall the defocus PSF of `psf/`); one slide, with the thin-lens relation in the math
  layer.
- **Demo `focusstack`** (new; canvas): a synthetic textured height map (seeded) imaged into a
  stack of 24 slices with a depth-dependent Gaussian blur; panels: one slice (slider), focus
  measure at a clicked pixel vs $z$ with the peak, the estimated height map vs truth, the
  all-in-focus image. Controls: window, measure, noise, slice spacing. Presets: "good texture",
  "a textureless patch", "a step", "few slices: sub-slice parabola".
- By hand: focus-measure values for 5 slices, pick the depth, parabola refinement.

## Shared engineering

Reuse (no local copies; extend through the owners):

- `lib/linalg.ts`: `svd`, `nullSpace`, `lstsq`, `solve`, `cholesky`, `cross3`, `inv3`, `det3`,
  `jacobiEigen` (DLT, eight-point, rank-2 projection, $E$ decomposition, Schur solves,
  gauge eigenvalues). Add via its owner (graph-cuts day) if missing: `rq3` (RQ of a 3×3 for
  $K$, $R$ from $P$) and `skew3` ($[\mathbf t]_\times$).
- `lib/mat4.ts`: `lookAt`, `rotation(axis, angle)`, `perspective`, `transformPoint` (graphics
  convention comparison in `pinhole`).
- `lib/gl3d.ts`: `GLView`, `OrbitCamera`, `addPoints`, `addLines` (camera pyramids, rays,
  epipolar plane), `addMesh`, `pointerRay`; one WebGL context per demo (scissored viewports).
  **Request to its owner (3D graphics day):** an `addSplats` primitive (instanced screen
  quads with a per-instance 2D conic, colour and opacity; premultiplied blending; draw order
  given by the caller) for `splatview`.
- `lib/volrender.ts`: `frontToBack`, `backToFront`, `segmentAlpha`, `correctAlpha` (the
  compositing in `flatland` and `splatview` uses these, so the two days share one definition).
- `lib/ransac.ts` (owner: Hough day): generic `ransac()`; **request**: make `Model<M>` generic
  in the datum type (`Model<M, D = P2>`) so the fundamental-matrix model (datum = a match pair,
  $s = 8$) plugs in; `iterationsNeeded` for the slide numbers.
- `lib/dp.ts` (owner: dynprog): `pathDP` with `cornerSteps(true, occlusion)` for the scanline
  tab of `stereomatch`; `lib/gridui.ts` for its step-through.
- `lib/integral.ts`: `integralImage`, `rectSum`, `boxMean`, `localStats` (window costs in
  $O(1)$; focus measures).
- `lib/imageops.ts`: `warp`, `sampleBilinear` (rectification, plane sweep), `laplacian`,
  `gaussianBlur`, `gradient`; `lib/synth.ts` for textures (`addGaussianNoise`, `checkerboard`);
  `lib/random.ts` `rng(seed)`, `gauss`; `lib/plot.ts` (cost curves, error plots, Jacobian
  sparsity `heatmap`, loss curves); `lib/colour.ts` (disparity colour map, `viridis`);
  `lib/registration.ts` `parabolaOffset` (sub-pixel disparity and focus peak).
- `lib/ui.ts` helpers (sliders, buttons, presets, `autoMount`).

New library files owned by this day (names checked against `demos/src/lib/`; none exist):

- **`lib/camera.ts`** (no DOM): `Camera {K, R, t, k1, k2}`, `project`, `projectDistorted`,
  `undistortPoint` (fixed-point iteration), `pixelRay`, `cameraCentre`, `axisAngleToR`/`RToAxisAngle`
  (Rodrigues), `normalise2d`/`normalise3d` (Hartley), `dltCamera`, `decomposeP` (RQ),
  `homographyDLT`, `zhangK` (from homographies), `visionToGL`/`glToVision` (the axis flip and
  $K$ ↔ $P_\text{GL}$), synthetic calibration targets.
- **`lib/twoview.ts`** (no DOM): `skew`, `essentialFromRt`, `fundamentalFromE`, `eightPoint`
  (normalised/unnormalised), `enforceRank2`, `epipolarLine`, `epipoles`, `sampsonDistance`,
  `decomposeE` (four candidates + cheirality), `triangulateDLT`, `triangulateMidpoint`,
  `triangulateOptimal2` (iterative), `rectifyUncalibrated`/`rectifyCalibrated` (homography
  pairs), a `Model` adapter for `lib/ransac.ts`, descriptor helpers (`patchDescriptor`,
  `matchRatio`, `crossCheck`).
- **`lib/stereo.ts`** (no DOM): scene generator with ground-truth disparity (textured planes,
  random dots, gain change), `costVolume(L, R, maxD, "sad"|"ssd"|"ncc"|"census", r)` (window
  sums by `lib/integral.ts`), `census`, `wta`, `leftRightCheck`, `subpixel`, `scanlineDP` (via
  `lib/dp.ts`), `sgm(costs, dirs 4|8, P1, P2)`, `badPixelRate`. Measure `sgm` at 192×144×48×8
  in the browser; if a live update exceeds ~30 ms, move `sgm` (and only it) to
  `demos/wasm/stereo/` (Rust, flat `&[f32]`), keeping this TypeScript version as the reference
  and testing that they agree.
- **`lib/bundle.ts`** (no DOM): generic Levenberg–Marquardt over a residual function with
  block-sparse Jacobian (camera blocks + point blocks), Schur complement solve (dense for the
  reduced camera system: small sizes only), Huber weighting, gauge fixing, covariance of points,
  an iteration recorder; problem builders for flatland (camera: x, y, heading, f) and 3D
  (axis-angle, t, optional f, k1).
- **`lib/flatland.ts`** (no DOM): the 2D world: scenes (polygons with textured edges,
  seeded), 1D pinhole cameras (position, heading, $f$, $c$, pixel count), `render1d` (ray casting
  against segments), point projection and back-projection, `triangulate2d`, plane-sweep scoring;
  a 2D **grid radiance field** (density + colour, `renderRay`, analytic gradients) and **2D
  Gaussian splats** (`projectSplat` to 1D, sort, composite, analytic gradients, densify/prune);
  Adam or plain gradient steps with a fixed seed. Shared by `triangulate`, `bundleadjust`,
  `flatland`.
- **`lib/splat.ts`** (no DOM): 3D Gaussians (`position, scale, rotation, opacity, colour`),
  `covariance3d`, `projectCovariance` (EWA), `conic`, `sortByDepth`, CPU `compositePixel`
  (reference and the "one pixel's list"), CPU `renderSplats` fallback for small images,
  procedural scenes (seeded).
- **`lib/focus.ts`** (no DOM): `focusStack(heightMap, nSlices, blurPerUnit, noise, seed)`,
  `focusMeasure(img, "lapvar"|"gradvar", r)`, `depthFromFocus` (+ parabola),
  `allInFocus`.

Tests (`demos/test/multiview-test.ts`, run by `make test`): project then back-project a pixel
gives a ray through the point; DLT recovers $P$ exactly (up to scale) from 6 noiseless
non-coplanar points and fails (rank) for coplanar ones; normalised DLT has a much smaller
condition number; `zhangK` recovers $K$ from 3 synthetic views; `eightPoint` satisfies
$\mathbf x'^\top F\mathbf x \approx 0$ on noiseless data and gives rank 2; `decomposeE` + cheirality
returns the generating $R$, $\hat{\mathbf t}$; triangulation is exact without noise; rectified
pairs have equal $y$ for matches; `costVolume` on a shifted random-dot image finds the shift
everywhere textured; census invariant under $2I + 10$; `sgm` with one direction equals
`scanlineDP`; LM converges on a perturbed flatland problem and $J^\top J$ has 4 zero eigenvalues
in 2D (similarity in the plane) without gauge fixing, 7 in 3D; `projectCovariance` matches a
Monte-Carlo projection of samples; `compositePixel` front-to-back = back-to-front; flatland
gradients match finite differences (`finiteDiff` from `lib/autodiff.ts`); `depthFromFocus` finds
the height of a textured plane within a slice.

Follow `demos/README.md` and `plan/build-issues.md`: fixed seeds, presets + reset, labels ≥ 18
px on slides (cost curves and readouts included), fits 1150×560 on a slide, `data-*` settings
(`data-mode`, `data-tab`, `data-preset`, `data-scene`), no keyboard capture, static figures as
print fallbacks; prefix per-demo CSS selectors with `.qb-demo[data-demo="…"]` (demo.css loads
late); unique plot clip ids come from the fixed `lib/plot.ts`; WebGL contexts only while
visible. Every demo goes on `demos.qmd`; the sidebar gets a "Geometry from images" section
under "3D" after "3D graphics". Performance: everything except `sgm` and the flatland
optimisations is tiny; the flatland fits run a fixed number of gradient steps per animation
frame (budget ≈ 10 ms), so they never block. Data: all synthetic by default.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, project and calibrate by hand, `pinhole`, `dltcalib` |
| 10:45–12:15 | Unit 2 lecture, epipolar lines and triangulation by hand, `epipolar`, `triangulate` |
| 13:15–14:45 | Unit 3 lecture, costs and disparity by hand, `stereomatch` (both tabs) |
| 15:00–16:30 | Unit 4 lecture, a Gauss–Newton step by hand, `bundleadjust`, `flatland` (sweep); quizzes as wrap-up |

Variant for a cohort that came for view synthesis: unit 1 shortened to 60 min (calibration by
DLT only; planar calibration as reading), unit 3 to 60 min (costs, WTA, SGM in concept only),
and `radiance-fields/` taught 15:45–16:45 after a 45-min unit 4.

## Open questions

- **Radiance fields core or optional?** The request names Gaussian splatting explicitly, and
  it is the most modern topic of the day; but camera, two-view geometry, stereo and bundle
  adjustment are the foundations it stands on. Proposed: optional but recommended, with the
  timetable variant above; alternatively a fifth half-day.
- **Ownership of matching interest points (descriptors, ratio test) and of the homography**:
  proposed for this day (the Motion elective is not planned yet, and this day needs both).
  Add both to the course map's owners table, or give them to the Motion day and recall here?
- **Feature detection beyond corners**: the edges day names scale-invariant keypoints in one
  line. Do we need a "keypoints across scale and rotation" slide here, or is the corner + patch
  version enough for the demos (which use synthetic scenes with known matches plus noise)?
- **Real data**: a licensed rectified stereo pair with ground truth (benchmark datasets are
  typically free for research; teaching use and redistribution must be checked), a real
  SEM tilt pair or a microscope focus stack, and a small photo set for SfM. Or synthetic only
  (as most of the course)?
- **3D bundle adjustment demo**: flatland only (proposed: everything visible, small), or also a
  3D version in `lib/gl3d.ts`? `lib/bundle.ts` supports both; the 3D one could be a preset of
  `epipolar` (two-view refinement) later.
- **Affine cameras** (orthographic microscopes, SEM, tilt series): one slide in units 1 and 4
  (proposed), or a separate optional unit with the tilt-series alignment worked through?
- `splatview` needs an `addSplats` primitive in `lib/gl3d.ts` (owner: 3D graphics day); until
  then the CPU fallback at ≤ 320×240 is enough for a few thousand splats.
- Programming language for code parts is still open (see README); nothing here depends on it.

## References

- Hartley & Zisserman (2004), *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press (camera, DLT, epipolar geometry, $F$/$E$, triangulation, bundle adjustment).
- Szeliski (2022), *Computer Vision: Algorithms and Applications*, 2nd ed., Springer (the best single reading for the day; stereo, SfM, MVS, view synthesis).
- Zhang (2000), *A flexible new technique for camera calibration*, PAMI.
- Abdel-Aziz & Karara (1971), *Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry*, ASP Symposium.
- Longuet-Higgins (1981), *A computer algorithm for reconstructing a scene from two projections*, Nature.
- Hartley (1997), *In defense of the eight-point algorithm*, PAMI.
- Nistér (2004), *An efficient solution to the five-point relative pose problem*, PAMI.
- Fischler & Bolles (1981), *Random sample consensus*, CACM (recall; owned by the Hough day).
- Lowe (2004), *Distinctive image features from scale-invariant keypoints*, IJCV (ratio test).
- Scharstein & Szeliski (2002), *A taxonomy and evaluation of dense two-frame stereo correspondence algorithms*, IJCV.
- Zabih & Woodfill (1994), *Non-parametric local transforms for computing visual correspondence*, ECCV (census).
- Ohta & Kanade (1985), *Stereo by intra- and inter-scanline search using dynamic programming*, PAMI.
- Hirschmüller (2008), *Stereo processing by semiglobal matching and mutual information*, PAMI.
- Triggs, McLauchlan, Hartley & Fitzgibbon (2000), *Bundle adjustment — a modern synthesis*, Vision Algorithms: Theory and Practice, Springer.
- Snavely, Seitz & Szeliski (2006), *Photo tourism: exploring photo collections in 3D*, SIGGRAPH (incremental SfM).
- Schönberger & Frahm (2016), *Structure-from-motion revisited*, CVPR.
- Collins (1996), *A space-sweep approach to true multi-image matching*, CVPR (plane sweep).
- Furukawa & Hernández (2015), *Multi-view stereo: a tutorial*, Foundations and Trends in Computer Graphics and Vision.
- Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi & Ng (2020), *NeRF: representing scenes as neural radiance fields for view synthesis*, ECCV.
- Fridovich-Keil, Yu, Tancik, Chen, Recht & Kanazawa (2022), *Plenoxels: radiance fields without neural networks*, CVPR (the grid version in `flatland`).
- Kerbl, Kopanas, Leimkühler & Drettakis (2023), *3D Gaussian splatting for real-time radiance field rendering*, ACM TOG (SIGGRAPH); project page https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
- Zwicker, Pfister, van Baar & Gross (2001), *EWA volume splatting*, IEEE Visualization.
- Westover (1990), *Footprint evaluation for volume rendering*, SIGGRAPH.
- Max (1995), *Optical models for direct volume rendering*, IEEE TVCG (recall).
- Nayar & Nakagawa (1994), *Shape from focus*, PAMI.
- Pentland (1987), *A new sense for depth of field*, PAMI (depth from defocus).
- Mastronarde & Held (2017), *Automated tilt series alignment and tomographic reconstruction in IMOD*, J. Struct. Biol. (fiducial alignment as bundle adjustment; teacher reference only, no tool names in student material).

## Build notes (2026-11-13, first build)

Built: the four core units `cameras/`, `two-view/`, `stereo/`, `sfm/` and the demos `pinhole`,
`dltcalib`, `epipolar`, `triangulate`, `stereomatch` (tabs images/slice), `bundleadjust`,
`flatland` (`data-mode="sweep"` only). New libraries: `lib/camera.ts`, `lib/twoview.ts`,
`lib/stereo.ts`, `lib/bundle.ts`, `lib/flatland.ts`; tests `demos/test/multiview-test.ts`;
figures from `tools/figs/multiview/figs.py`. Not built: optional `radiance-fields/`,
`depth-focus/`, the `flatland` grid/splat modes, `splatview`, `lib/splat.ts`, `lib/focus.ts`.

Deviations from the plan:
- `sgm` stays TypeScript (no Wasm yet): SGM 8 at 192 × 144 × 40 takes ≈ 175 ms in node on the
  (loaded) build machine; the demo coalesces updates into one animation frame. Wasm candidate.
- `costVolume` uses a local running-sum box filter (`boxMeanFast`, same result as
  `lib/integral.ts` `boxMean`, ~10× faster here) instead of `boxMean`.
- `lib/ransac.ts` is P2-only, so `lib/twoview.ts` has its own `ransacF` loop (uses `random.ts`
  `sampleIndices`). No `rq3`/`skew3` were added to `lib/linalg.ts`: RQ lives in `camera.ts`
  (`decomposeP`), `skew` in `twoview.ts`.
- Scanline DP is the exact Viterbi with P1/P2 jump costs (O(D) per pixel), tested against
  `lib/dp.ts` `viterbi`; the slice tab draws WTA and the DP path (no gridui step-through).
  The plan's "SGM with one direction equals scanline DP" holds only at the last pixel of a row
  (no backtracking); the test and B9 say exactly that.
- Descriptor helpers (`patchDescriptor`, `matchRatio`, `crossCheck`) not written: the demos use
  synthetic scenes with known matches; descriptors and the ratio test are concept slides only.
- Stereo scenes: the textureless wall has a textured frame, and the fence has textured rails
  (otherwise no aggregation can recover them; recorded honestly in the notes). Gain scene uses
  smooth texture (binary random dots kept SAD working under gain).
- Flatland BA: 10 cameras (f = 130 px), 40 points, only points seen ≥ 3 times, σ = 0.2 px; the
  gauge fixes camera 1 and the coordinate of the opposite camera along the longer baseline
  component (fixing camera 2's x left a near-gauge mode, eigenvalue 0.03). Loop-closure preset
  not built; incremental mode resects by 1D DLT with LM refinement (and a fallback of starts).
- `epipolar`'s 3D view uses gl3d mesh/lines; no residual histogram (mean distances instead).
- Unit 1 numbers by hand: as planned. Unit 3 B5 slice table changed (the plan's table tied).
