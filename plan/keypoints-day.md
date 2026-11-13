# Plan: a day on keypoints and descriptors (finding, describing and matching points)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.

**Course map (2026-11-13, "Queue, second batch"):** elective day, group "Detection" (Hough,
integral images, point detection, **keypoints**). Hard entry: Edges, blobs and ridges (day 2).
Soft prerequisites: the Hough day's `ransac/` and FFT registration (day 12). Both are recalled
so that this day stands alone. **Ownership:** this day becomes the owner of **feature
matching**: keypoints across scale and rotation, local descriptors (gradient histograms and
binary), matching (distances, ratio test, mutual nearest neighbours), and fitting a 2D
transform to the matches with RANSAC. The built `two-view/` unit (geometry from multiple
images) currently claims "matching interest points between images" in two slides. That claim
moves here (see "Proposed reorganisation").

Notation (stated once in unit 1): image $I$, Gaussian scale space $L(x, y; \sigma)$, DoG
$D(x, y; \sigma) = L(\cdot; k\sigma) - L(\cdot; \sigma)$, structure tensor $J$ with eigenvalues
$\mu_1 \ge \mu_2$ (as in `structure-tensor/`), Hessian $H$ (as in `hessian/`). A keypoint is
$(x, y, \sigma, \theta)$. A descriptor is a vector $\mathbf d$, or a bit string $\mathbf b$. A
match is a pair $(i, j)$. A 2D transform is $T$; a homography is $H$ only where `cameras/` uses
it, and **never on the same slide as the Hessian** (the Hessian is written $\mathcal H$ in
unit 1 when both appear in one deck). The inlier fraction is $w$, the minimal sample size $s$
and the RANSAC tolerance $\varepsilon$, all as in `ransac/`. λ is not used.

## Goal of the day

After the day a student can:

1. say what makes a good **interest point**: it can be found again (**repeatable**) in another
   view, and it is **distinctive** enough to be told apart from the others. Explain why a
   corner works and a point on a straight edge does not (recall `structure-tensor/`);
2. explain why a single-scale corner detector fails when the image is **zoomed**, and find
   keypoints with a **scale**: extrema of the **difference of Gaussians** in $(x, y, \sigma)$
   over an octave pyramid (recall `scale-space/`), or a scale-normalised Harris measure. Refine
   the position and scale to sub-pixel accuracy, and reject edge responses with the Hessian
   ratio test;
3. give each keypoint an **orientation** from a histogram of gradient directions, and explain
   how position, scale and orientation make the description **invariant** to translation,
   zoom and rotation (and what remains: tilt, blur, lighting);
4. build a **gradient-histogram descriptor** by hand on a small grid of cells (128 numbers =
   4 × 4 cells × 8 directions), and say why it tolerates small shifts, contrast changes and
   lighting. Build a **binary descriptor** from pairwise intensity comparisons, and compare two
   bit strings by the **Hamming distance**;
5. **match** descriptors by nearest neighbour under a suitable distance. Apply the **ratio
   test** and the **mutual nearest-neighbour** (cross-) check, predict which matches survive,
   and measure matching quality by precision and recall against a known transform;
6. choose a **2D transform model** (translation, rigid, similarity, affine, homography) from
   what the scene and camera allow, count its degrees of freedom and the number of point pairs
   it needs, and fit it robustly with **RANSAC** (recall). Compute the number of iterations,
   then refit on the inliers;
7. use matches for **stitching** (tiles and panoramas), for **registration** (rotation, scale and
   large changes where phase correlation fails), and for **geometry from multiple images**
   (pointer to `two-view/`), and compare feature-based with intensity-based registration;
8. *(math)* derive the sub-pixel offset $\hat{\mathbf x} = -\mathcal H^{-1}\nabla D$ and the edge
   criterion $\operatorname{tr}^2/\det < (r+1)^2/r$. Show the invariance of the normalised
   descriptor to an affine change of intensity. Relate the L2 distance of unit vectors to the
   cosine, and the Hamming distance to XOR and bit counting. Write the linear least-squares
   systems for affine and homography (DLT) fits, and derive $N = \log(1-p)/\log(1-w^s)$ for
   the models of the day.

Why it fits this audience: the day reuses the matrices the students already know (the
structure tensor, the Hessian, Taylor expansions) and turns them into a working pipeline that
every vision system uses. The new ideas are combinatorial and statistical rather than
analytical: **invariance by normalisation** (measure the scale and the angle, then undo them),
**distinctiveness in a high-dimensional space** (why "nearest" is not enough and the ratio
test helps), and **robustness** (most of the matches can be wrong and the transform is still
found). Linear algebra (least squares, null spaces) and basic probability are enough for all
of it. For a science cohort it is also the method behind aligning serial sections, slide-scanner
tiles with rotation, correlative light/EM overviews and photographs of specimens.

## Prerequisite gap

The day is entered after core day 2. Everything else is recalled, self-contained, in at most
one or two slides each.

- **Harris/Förstner corners, the structure tensor, "interest points can be found again"**:
  `structure-tensor/` (day 2, slides "Edge, corner, flat: Harris and Förstner", "The math:
  Harris from shifting a window", "Interest points"). Unit 1 opens with a one-slide recall and
  the `structtensor` demo with `data-preset` "corners of a checkerboard". Nothing is re-derived.
- **Scale space, scale-normalised LoG, DoG ≈ LoG, maxima in $(x, y, \sigma)$, the Gaussian
  pyramid**: `scale-space/` (day 2). Its "Where this leads" line already points to SIFT-style
  keypoints ("DoG extrema in a pyramid, plus an orientation"); unit 1 keeps that promise. One
  recall slide with the `blobscale` figure; the octave bookkeeping is new here.
- **The Hessian, trace and determinant**: `hessian/` (day 2). Recalled in one line for the
  edge-rejection test (a ridge has one large and one small curvature).
- **Local maxima, sub-pixel peaks by a parabola**: `spot-filters/` and `spot-fit/` (the
  point-detection elective, slide "Three-point rules"), possibly not taken. Unit 1 gives the 1D
  three-point parabola in one slide (self-contained) and generalises it to 3D in the math layer.
- **Moments, the intensity centroid**: `regionprops/` (labelling, core day 9, may come later).
  Unit 2 needs only "the brightness-weighted centre of a patch" for the orientation of binary
  descriptors. One line, self-contained, with a pointer.
- **NCC** (normalised cross-correlation): owned by `matching/` (FFT registration, core day 12,
  may come later). Unit 2 defines it in one slide as the simplest descriptor (the normalised
  patch) with a pointer, as `two-view/` already does.
- **The Hungarian algorithm / one-to-one assignment**: `object-eval/` (evaluation, day 16) and
  `track-linking/` (tracking elective). Unit 3 uses it in one line ("if every point may be used
  only once, solve an assignment problem") with a pointer. It is not taught here.
- **RANSAC**: owned by `ransac/` (Hough elective, same group, maybe not taken). Unit 4 has a
  two-slide recall (minimal sample → exact fit → count inliers → keep the best → refit;
  $N = \log(1-p)/\log(1-w^s)$) and embeds the built `ransac` demo with `data-preset="seventeen"`.
  The application to transforms between images, which `ransac/` "Where RANSAC is standard"
  says "belongs to the planned Motion elective", is taught here.
- **Affine maps in homogeneous coordinates, inverse mapping, bilinear interpolation**:
  `images/` (day 1). Recalled in one line for warping image 2 onto image 1.
- **The homography** (image of a plane, rotating camera; DLT from 4 pairs): owned by `cameras/`
  (3D group, probably not taken). Unit 4 gives a self-contained concept slide (a 3 × 3 matrix,
  8 numbers, straight lines stay straight, parallel lines may meet) and one math slide with the
  DLT rows, pointing to `cameras/` for the derivation and to `two-view/` for the "pure rotation
  → homography" argument.
- **Procrustes** (closed-form similarity from point pairs): owned by `shape-models/` (snakes
  elective). Unit 4 states the result in one line (centre, rotate by SVD, scale) and points
  there.
- **Global alignment of a tile graph by least squares**: `stitching/` (day 12). Unit 4 recalls
  it as "the same graph, now with rotations"; bundle adjustment: pointer to `sfm/`.
- **k-means** (optional unit only): `kmeans/` (Gaussian-mixtures elective). One-slide
  self-contained version in the optional unit.

## Overlaps with other plans and units (who owns what)

| Topic | Owner | Here |
|---|---|---|
| Structure tensor, Harris/Förstner/Shi–Tomasi corners | `structure-tensor/` (day 2) | recall (one slide); unit 1 adds the **multi-scale** Harris (two scales, normalisation) |
| Scale space, normalised LoG, DoG ≈ LoG, pyramid, blob detection in $(x, y, \sigma)$ | `scale-space/` (day 2) | recall (one slide); unit 1 adds **octaves**, sub-pixel/sub-scale refinement, contrast and edge rejection, orientation assignment |
| Hessian, trace, determinant | `hessian/` (day 2) | the curvature-ratio edge test (one slide + math) |
| Determinant-of-Hessian with box filters (SURF-style) | integral images `box-filters/` | one-line pointer in unit 1 "other detectors" |
| Spot detection, local maxima, three-point parabola | point detection `spot-filters/`, `spot-fit/` | self-contained 1D parabola; spots are blobs **without** a descriptor (one comparison line) |
| Intensity centroid (first moments) | `regionprops/` (day 9) | one line (orientation of binary descriptors) |
| NCC, template matching | `matching/` (day 12) | one-slide self-contained recall ("the normalised patch is the simplest descriptor") |
| **Keypoints across scale and rotation, orientation assignment, repeatability** | **this day**, unit 1 | — |
| **Local descriptors: gradient-orientation histograms, binary descriptors, invariance by normalisation, descriptor distances** | **this day**, unit 2 | `two-view/` recalls in one slide; SVM day's unbuilt `detect-window/` (HOG) recalls the cell histogram from here |
| **Matching: nearest neighbour, ratio test, mutual NN, one-to-one, match precision/recall, fast search** | **this day**, unit 3 (moves from `two-view/`) | `two-view/`, `sfm/` recall; tracking's `track-linking/` stays the owner of linking over time |
| One-to-one assignment (Hungarian) | `object-eval/` (evaluation) | one-line use with a pointer |
| RANSAC (algorithm, iteration count, vs Hough) | `ransac/` (Hough elective) | two-slide recall + demo; **application to 2D transforms taught here** |
| 2D transforms (affine, homogeneous coordinates, warping) | `images/` (day 1) | recall; the **hierarchy of models and their degrees of freedom** is taught here |
| Homography: definition, DLT, plane/rotation geometry | `cameras/` (3D group) | self-contained concept + math slide, pointer |
| Similarity from point pairs (Procrustes) | `shape-models/` (snakes elective) | one-line statement, pointer |
| Translation by phase correlation, global tile alignment, drift | FFT registration (`phasecorr/`, `stitching/`) | comparison slide (intensity vs features), recall of global least squares |
| Fundamental matrix, epipolar geometry, RANSAC on $F$, triangulation | `two-view/` (3D group) | pointer only ("matches → $F$ → depth") |
| Bundle adjustment | `sfm/` | pointer only (global panorama refinement) |
| Laplacian pyramid | `scale-space/` | used for multi-band blending (one slide in unit 4) |
| Optical flow, feature tracking (Shi–Tomasi "good features to track") | `lucas-kanade/` | pointer: small motions → flow; large motions → match keypoints |
| Non-rigid registration (B-spline FFD), landmark validation | `nonrigid-reg/` | pointer: features give the rigid/affine start, FFD refines |
| Learned detectors/descriptors, attention-based matchers | modern deep learning (`attention/`) | optional unit 5: one slide, pointer |
| MSER (regions stable over many thresholds) | watershed/thresholding (`ws-flooding/`, `histogram/`) | one line in unit 1 "other detectors" |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus one optional unit. The order
follows the pipeline: **detect → describe → match → fit**. Folder names checked against the
repo: `keypoints/`, `descriptors/`, `feature-matching/`, `feature-alignment/`,
`feature-search/` do not exist (`matching/`, `stitching/`, `ransac/` do and are not reused).
Deck titles: "Keypoints: corners and blobs across scale", "Describing a keypoint: gradient
histograms and bits", "Matching descriptors: nearest, ratio and mutual", "From matches to a
transform: RANSAC, stitching and registration".

Each deck: "About this lecture" (ends with the math-layer line from README.md), about 18–24
content slides including the `.math` slides, a summary, and about half of the slides with a
figure or demo next to text (counts per unit below).

### 1. `keypoints/`: Keypoints — corners and blobs across scale

Concept layer:

- **The task** (picture: two photographs of the same scene, zoomed and rotated, with a few
  corresponding points joined by lines): find a few hundred points that we can **find again**
  in the other image, and later **recognise**. Two requirements: **repeatability** (the detector
  fires at the same physical point) and **distinctiveness** (the neighbourhood looks unlike the
  others). Uses on one strip of thumbnails: panorama, slide tiles, serial sections, 3D from
  photographs.
- **Recall: corners** (one slide, from `structure-tensor/`): a window at a corner changes under
  every shift; Harris $\det J - k(\operatorname{tr}J)^2$, Förstner/Shi–Tomasi $\mu_2$. They are
  repeatable under shift and rotation (the eigenvalues do not change when the image rotates).
- **Zoom breaks it**: the same corner seen 2× larger needs a 2× larger window; at a fixed σ the
  rounded corner of a zoomed image looks like an edge, and a fine-scale texture corner
  disappears when zoomed out. Picture: Harris at σ = 1.5 on an image and its 2.5× zoom, with
  the repeated points coloured. Repeatability falls to about a third.
- **Give each point a scale** (recall `scale-space/`): the scale-normalised response of a blob
  peaks at its own σ (radius √2·σ), so a detection at $(x, y, \sigma)$ in image 1 appears at
  $(sx, sy, s\sigma)$ in an image zoomed by $s$. Detection = **local extrema in $(x, y, \sigma)$**
  of the DoG (26 neighbours). The scale is not a nuisance: it is **measured**, and it fixes the
  size of the window that will be described.
- **The octave pyramid** (new bookkeeping, one picture): blur in steps of $k = 2^{1/S}$ (e.g.
  $S = 3$ steps per octave), subtract neighbours to get DoGs, and after each doubling of σ
  downsample by 2 (recall: blur before downsampling). Cost: 4/3 of one image per level. The first
  octave is often computed on a 2× upsampled image to catch the smallest blobs.
- **Multi-scale corners** (one slide): Harris with derivative scale σ_D and window ρ = 1.5–2 σ_D,
  normalised by σ_D², over a range of σ_D; keep a corner at the σ where the normalised LoG peaks
  (Harris–Laplace, named only). Corners localise better; DoG blobs are more numerous and more
  scale-stable. Both are used.
- **Refining a candidate** (picture: a 3×3×3 neighbourhood and a parabola through three
  samples): fit a parabola through the samples around the extremum along x, y and σ. The true
  peak lies between samples; if the offset is more than half a sample, move to the neighbour
  and repeat. Gives sub-pixel position **and** sub-level scale. The 1D three-point rule is the
  same as in `spot-fit/` (pointer).
- **Throw away weak and edge-like candidates**:
  - low contrast: $|D(\hat{\mathbf x})|$ below a threshold (noise extrema; the threshold scales
    with the noise, as in `spot-filters/` "How many false spots?");
  - **edges**: the DoG also responds along edges, where the position along the edge is
    undefined (the aperture problem again: `structure-tensor/`, `lucas-kanade/`). The Hessian at
    the point has one large and one small curvature. Reject if the ratio of curvatures exceeds
    $r$ (typically 10). Picture: candidates on a rectangle before/after the test, survivors only
    at the corners.
- **Orientation** (picture: a keypoint, its gradient arrows in a disc of radius ∝ σ, and the
  36-bin orientation histogram with its peak): the dominant gradient direction, weighted by
  magnitude and a Gaussian window. A second peak above 80 % of the highest gives a **second
  keypoint** at the same place (about 15 % of points). Rotating the image rotates θ by the same
  angle, so describing the window **relative to θ** removes rotation.
- **What a keypoint is now**: a circle with a radius line, $(x, y, \sigma, \theta)$: a small
  coordinate frame that moves with the image under translation, zoom and rotation. Picture: the
  same frames drawn on both views of the scene.
- **Measuring a detector: repeatability** (picture: repeatability against zoom factor and
  against rotation angle for single-scale Harris, multi-scale Harris and DoG, from the demo):
  the fraction of points in image 1 whose transformed position (known transform) has a
  detection in image 2 within ε px and at a matching scale. Single-scale Harris collapses with
  zoom; DoG stays flat up to about 2.5× and then degrades. Rotation does not hurt any of them
  much.
- **What is not covered**: strong **tilt** (perspective foreshortening squashes circles into
  ellipses; affine-adapted detectors estimate the ellipse from the structure tensor, one line),
  heavy blur, non-rigid deformation, and textureless images (no keypoints: phase correlation or
  intensity registration instead).
- **Other detectors** (one slide, one line each with a thumbnail): **FAST** (a ring of 16 pixels,
  9 contiguous ones all brighter or all darker: very cheap, no scale, used with pyramids);
  determinant of Hessian with box filters (pointer to `box-filters/`); **MSER**, regions that
  stay the same over many thresholds (pointer to `ws-flooding/`); learned detectors (pointer to
  the optional unit). Pointer: image libraries call the DoG detector with orientation "SIFT
  keypoints".

Math layer (`## The math: … {.math}`):

- *Scale covariance*: if $I_2(\mathbf x) = I_1(\mathbf x/s)$ then
  $L_2(\mathbf x; \sigma) = L_1(\mathbf x/s; \sigma/s)$, and the γ-normalised derivatives satisfy
  $\sigma^2\nabla^2L_2(\mathbf x;\sigma) = (\sigma/s)^2\nabla^2L_1(\mathbf x/s;\sigma/s)$: extrema
  move from $(\mathbf x, \sigma)$ to $(s\mathbf x, s\sigma)$. The unnormalised version picks the
  wrong scale (recall `scale-space/` "Why scale-normalise").
- *Octaves*: levels $\sigma_i = \sigma_0k^i$, $k = 2^{1/S}$; $S + 3$ blurred images per octave give
  $S + 2$ DoGs and extrema on $S$ interior levels; incremental blur
  $\sqrt{\sigma_{i+1}^2 - \sigma_i^2}$ (semigroup, recall); DoG ≈ $(k-1)\sigma^2\nabla^2G$ (recall).
- *Sub-pixel refinement*: $D(\mathbf x_0 + \mathbf h) \approx D + \nabla D^\top\mathbf h +
  \tfrac12\mathbf h^\top\mathcal H_D\mathbf h$ with $\mathbf h = (x, y, \sigma)$; setting the gradient
  to zero gives $\hat{\mathbf h} = -\mathcal H_D^{-1}\nabla D$ and the peak value
  $D(\hat{\mathbf h}) = D + \tfrac12\nabla D^\top\hat{\mathbf h}$; derivatives by central differences
  in the 3×3×3 cube. 1D special case: $\hat h = \tfrac{D_{-1} - D_{+1}}{2(D_{-1} - 2D_0 + D_{+1})}$.
- *Edge test*: for the 2×2 spatial Hessian with eigenvalues $\alpha = r\beta$,
  $\operatorname{tr}^2/\det = (\alpha + \beta)^2/(\alpha\beta) = (r + 1)^2/r$, increasing in $r \ge 1$;
  keep if $\operatorname{tr}^2/\det < (r+1)^2/r$ (12.1 for $r = 10$) and $\det > 0$. No
  eigenvalues needed.
- *Orientation*: histogram over 36 bins of $\theta = \operatorname{atan2}(L_y, L_x)$, weights
  $\|\nabla L\|\,e^{-\|\mathbf x - \mathbf x_0\|^2/2(1.5\sigma)^2}$; parabola through the peak bin
  and its neighbours (the same three-point rule, now on a circle).
- *Repeatability*: $\mathrm{rep} = |\{i : \exists j,\ \|T\mathbf x_i - \mathbf y_j\| < \varepsilon,\
  |\log(s\sigma_i/\sigma_j)| < \delta\}| / \min(n_1, n_2)$, counted only in the common region.

**Demo `dogpyramid`** (new, step-through, small): a 192×192 image (synthetic: discs and squares
of three sizes on a gentle ramp, a long bar, seeded noise; or a crop of `cell256.png`). Left: the
octave pyramid as a grid of thumbnails, blurred images in one column and DoGs (diverging, 0 =
mid-grey) in the next, one row per octave (sizes halve). Right: the selected DoG level large, with
candidates marked: extremum candidates (all), after the contrast threshold, after the edge test,
with colour per stage. Click a candidate: inset with its 3×3×3 neighbourhood as three small 3×3
grids with numbers (the centre highlighted, the larger/smaller neighbours counted), the
sub-pixel offset vector and the refined $(x, y, \sigma)$, the 2×2 Hessian with
$\operatorname{tr}^2/\det$ against the threshold, and the verdict. Controls: steps per octave $S$
(1–5), contrast threshold, edge ratio $r$, *upsample first octave* toggle, noise. Buttons *step*
(build level by level), *all*. Presets: "one octave, three levels", "the bar gives edge
candidates", "edge test removes them", "contrast threshold vs noise", "S = 1: blobs between
levels are missed", "refinement moves a point by 0.4 px". Readout: candidates per stage.

**Demo `repeatability`** (new, main demo): two views side by side (each 256×256 on the page,
2 × 200×200 plus a plot on slides). View 1: a synthetic scene (`featureScene`, see shared
engineering: corners, blobs at several scales, a repetitive grid, textured patches) or
`cell256.png`. View 2: view 1 warped by a known similarity (rotation slider −180…180°, zoom
slider 0.4–3, log) plus optional perspective tilt, brightness gain/offset, gamma, blur and noise.
Detector: *Harris (one σ) / Harris multi-scale / DoG / FAST*. Keypoints drawn as circles with
radius ∝ σ and an orientation tick; points of view 1 that are repeated in view 2 are filled
green, unrepeated red. Plot below: repeatability against zoom (or against angle, tab), computed
by sweeping the slider's range once per detector change (cached), with the current value
marked. Click a keypoint in view 1: its predicted position in view 2 is drawn, with the
nearest detection, and in a small inset its σ-response curve in both views (the peak shifts by
$\log s$). Controls also: number of keypoints (keep the strongest $n$), ε. Presets: "rotation:
all detectors fine", "zoom 2×: Harris (one σ) loses the points", "multi-scale and DoG survive",
"tilt 50°: everything degrades", "gain and offset: no effect", "blur", "the micrograph",
"FAST is fast" (with a timing readout per detector).

Figures (static, `tools/figs/keypoints/`): two-views-with-correspondences (synthetic photo-like
scene); thumbnails of uses; Harris at one σ on an image and its zoom (repeated points
coloured); scale covariance sketch (the same blob response curve against log σ, shifted by
log s); octave pyramid diagram (blur and DoG stacks, downsampling arrows, sizes); a 3×3×3
neighbourhood; 1D parabola refinement through three samples; candidates on a rectangle before
and after the edge test; orientation histogram with 80 % line and two peaks; keypoint frames
on both views; repeatability-vs-zoom curves (from the demo's functions); detector thumbnails
(FAST ring, MSER regions). About 12 of 22 slides with a figure or demo.

By hand (concept):

- **A1.** Harris responses for three given 2×2 tensors (recall) and the verdict; then the same
  scene zoomed 3×: which of the three windows (given) still contains the corner?
- **A2.** A blob is detected at σ = 2.4 px in image 1 and at σ = 6.0 px in image 2. What is the
  zoom factor? With $S = 3$ steps per octave starting at σ₀ = 1.6, in which octave and level
  is each found?
- **A3.** Octave bookkeeping: for $S = 3$, list the σ of the six blurred images of octave 0 and
  the five DoGs; which levels can hold extrema? What is the total number of pixels for 4
  octaves of a 512² image (with and without the 2× upsampled first octave)?
- **A4.** 1D refinement: DoG samples 0.6, 1.0, 0.8 at x = −1, 0, 1. Offset (+0.17 px) and the
  refined peak value (≈ 1.008).
- **A5.** Edge test with $r = 10$ (threshold 12.1): Hessians $\mathrm{diag}(-4, -3)$ (49/12 = 4.1,
  keep), $\mathrm{diag}(-10, -0.5)$ (110.25/5 = 22.1, reject), and one with $\det < 0$ (a saddle:
  reject). Sketch where each could lie in an image.
- **A6.** Orientation: eight gradients (magnitude, angle) around a keypoint; fill a 36-bin
  histogram (10° bins) and find the orientation(s); add 30° to all angles: what happens?
- **A7.** Predict: keypoints on (a) a uniformly grey wall, (b) a brick wall (repetitive), (c) a
  defocused image of (b). Which requirement fails in each case?

*(math)*:

- **B8.** *(math)* Show that scale-normalised LoG extrema move from $(\mathbf x, \sigma)$ to
  $(s\mathbf x, s\sigma)$ under a zoom by $s$, and that the raw LoG does not have this property.
- **B9.** *(math)* Derive $\hat{\mathbf h} = -\mathcal H^{-1}\nabla D$ and the peak value
  $D + \tfrac12\nabla D^\top\hat{\mathbf h}$ from the second-order Taylor expansion.
- **B10.** *(math)* Show that $(\alpha + \beta)^2/(\alpha\beta)$ depends only on $r = \alpha/\beta$
  and is increasing for $r > 1$.
- **B11.** *(math)* Why $S + 3$ blurred images per octave, and the incremental blur between
  levels from the semigroup property.

### 2. `descriptors/`: Describing a keypoint — gradient histograms and bits

Concept layer:

- **The job**: turn the window around each keypoint into a vector, so that the same physical
  point gives nearby vectors in both images and different points give distant ones. Picture:
  three keypoint windows from view 1, the true partner and two impostors from view 2, and their
  descriptors as bar plots.
- **Invariance by normalisation** (one picture, the central idea of the day): the keypoint
  frame $(x, y, \sigma, \theta)$ is measured, so we sample the window **in that frame**: centred at
  $(x, y)$, size ∝ σ, rotated by θ. Translation, zoom and rotation are gone before anything is
  computed. What remains is handled by the description itself: **lighting**, small errors in the
  frame, small deformations, noise.
- **The simplest descriptor: the normalised patch** (recall `matching/`, self-contained):
  resample a 16×16 patch in the keypoint frame, subtract its mean, divide by its standard
  deviation, compare by NCC. Invariant to gain and offset. But it is fragile: a one-pixel shift
  or a slight deformation changes every number (picture: NCC of a patch with itself shifted by
  0, 1, 2 px drops quickly on texture).
- **Gradient histograms** (the SIFT-style descriptor, step by step with pictures):
  1. gradients in the window (in the keypoint frame), magnitude-weighted by a Gaussian (centre
     counts more, frame errors at the rim matter less);
  2. divide the window into **4 × 4 cells**; in each cell a histogram of gradient directions in
     **8 bins** (45° each); each gradient's vote is shared between neighbouring bins and cells
     (soft binning) so that small shifts and rotations change the histogram smoothly;
  3. concatenate: 4 × 4 × 8 = **128 numbers**;
  4. **normalise** to unit length (a gain change multiplies all gradients: gone; an offset does
     not change gradients at all), **clip** large entries (0.2) and renormalise: one glinting
     highlight cannot dominate.
  Why it works: within a cell, **where** exactly a gradient sits does not matter (tolerates
  small shifts and deformations), but **which way** it points does (keeps the shape). The
  same idea as the complex cells of the visual cortex (one line).
- **Worked example on the slide**: one cell with four gradients → its 8-bin histogram; the image
  rotated by 90° → the histogram shifts by two bins; measured relative to θ → identical.
- **Binary descriptors** (picture: a smoothed patch with 30 random point pairs drawn as line
  segments, each giving one bit): compare intensities at pairs of points, $b_i = [I(\mathbf p_i) <
  I(\mathbf q_i)]$, 256 pairs → **256 bits = 32 bytes**. Fixed pairs chosen once (seeded random,
  or learned to be uncorrelated). Invariant to **any increasing change of intensity** (only
  the order matters). Rotation: rotate the pair pattern by θ, with θ from the
  **intensity centroid** (the vector from the keypoint to the brightness-weighted centre of the
  patch: first moments, pointer to `regionprops/`). Smooth first: single pixels are noisy.
- **Comparing descriptors: distances** (one slide with a table): L2 (Euclidean) for
  histograms; L1 or the χ²-distance for histograms where small bins matter; **Hamming** (number
  of differing bits) for binary descriptors, computed with an exclusive-or and a bit count: tens
  of times faster than 128 floating-point differences. Worked: two 16-bit strings, Hamming
  distance 4.
- **Trade-offs** (table, one slide): gradient histogram vs binary: size (128 numbers vs 32 bytes),
  speed of computing and comparing, distinctiveness (histograms better under blur and viewpoint
  change), invariance to lighting (both; binary to any monotone change). Uses: histograms for
  accuracy (3D reconstruction, microscopy registration), binary for speed (video, robots, phones).
- **Where descriptors fail** (picture row): repetitive texture (many identical windows: the
  descriptor is distinctive locally but not globally), strong viewpoint change (the window
  content itself changes), changed modality (fluorescence vs brightfield, light vs EM: the
  gradients may even reverse sign; one line: orientation-only histograms ignoring the sign, or
  mutual information on intensities, pointer to `nonrigid-reg/`).
- Pointers: the same cell histogram, computed densely over a window, is the HOG feature of
  window detectors (SVM elective, `svm-day` plan) and was the input to many classifiers before
  neural networks; learned descriptors (optional unit 5). Image libraries call the gradient
  histogram "SIFT descriptors" and the steered binary one "ORB" (one line, no other names).

Math layer:

- *Invariance to affine intensity changes*: $I' = aI + b$ ($a > 0$) gives $\nabla I' = a\nabla I$;
  every histogram entry scales by $a$; unit-length normalisation removes $a$; $b$ never
  appears. Clipping makes it approximately invariant to non-linear changes that affect a few
  large gradients.
- *Soft (trilinear) binning*: a gradient at continuous cell coordinates $(u, v)$ and angle
  bin coordinate $o$ adds $m\,(1-|u - u_i|)(1-|v - v_j|)(1-|o - o_k|)$ to the 8 surrounding
  bins; the descriptor is then a continuous (piecewise linear) function of the frame.
- *Rotation by θ*: sample at $\mathbf x_0 + \sigma R(\theta)\mathbf u$ and subtract θ from every
  gradient angle; rotating the image by φ changes θ by φ and leaves the descriptor unchanged
  (up to resampling).
- *Distances*: for unit vectors $\|\mathbf a - \mathbf b\|^2 = 2 - 2\,\mathbf a\cdot\mathbf b$ (L2 ↔
  cosine); the **root descriptor** (L1-normalise, then take square roots) turns the Euclidean
  distance into the Hellinger distance between histograms, which weights small bins more
  (one line, a measurable gain). Hamming: $d_H(\mathbf a, \mathbf b) = \operatorname{popcount}(\mathbf
  a \oplus \mathbf b)$; for two unrelated descriptors with balanced bits it is
  Binomial$(n, \tfrac12)$: mean $n/2$, sd $\sqrt n/2$ (128 ± 8 for 256 bits), so a true match at
  distance 40 is about 11 sd away.
- *Intensity centroid*: $\theta = \operatorname{atan2}(m_{01}, m_{10})$ with
  $m_{pq} = \sum x^py^qI(x, y)$ over a disc centred at the keypoint.
- *High dimensions* (one slide): for random vectors in 128 dimensions, the distances to the
  nearest and to the farthest neighbour are close together (concentration of distances); this
  is why an absolute distance threshold works badly and why unit 3 uses a ratio.

**Demo `descriptorlab`** (new, main demo): left, a 256×256 view (the `featureScene` or the
micrograph) with keypoints from `lib/keypoints.ts`; click one to select. Middle, the keypoint
window resampled in its frame (large, 16×16 samples shown as pixels) with the gradient arrows,
the 4×4 cell grid and the Gaussian weight as a faint disc; or, in *binary* mode, the smoothed
patch with the point pairs as segments coloured by their bit. Right: the descriptor as a
4×4 grid of small polar "roses" (8 petals each) plus the 128-bar plot, or the bit string as a
16×16 grid of black/white squares. Bottom: a strip of **distances from the selected descriptor to
every descriptor in view 2**, sorted, with the true partner marked (known transform) and the
first and second nearest labelled. Controls on view 2: rotation, zoom, gain, offset, gamma,
blur, noise (as `repeatability`); toggles *rotate to θ* (off: rotation breaks the descriptor),
*normalise*, *clip 0.2*, *soft binning*, *root descriptor*; descriptor *patch (NCC) / histogram /
binary*; distance *L2 / L1 / χ² / Hamming* (Hamming only for binary). Presets: "the true
partner is nearest", "rotate 40° without θ: lost", "with θ: found again", "gain ×3: normalising
fixes it", "gamma 0.5: clipping helps", "the normalised patch fails under a 1-px shift",
"binary: 256 bits, Hamming", "repetitive grid: three equally near impostors", "blur".

Figures (`tools/figs/descriptors/`): three windows with partner and impostors; the keypoint
frame resampling sketch (circle, radius, rotated grid); NCC of a patch with its 0/1/2-px shifts
(curve); the four steps of the gradient-histogram descriptor (four small panels); the worked
cell histogram before/after a 90° rotation; binary pairs over a patch; the intensity centroid
arrow; the Hamming worked example (two bit rows, XOR row); the trade-off table needs no image;
failure strip (repetitive texture, viewpoint change, modality change). About 11 of 21 slides
with a figure or demo.

By hand (concept):

- **A1.** One cell: gradients (magnitude, angle) = (2, 10°), (1, 50°), (3, 100°), (2, 355°).
  Fill the 8-bin histogram with nearest-bin voting (bins centred at 0°, 45°, …):
  [4, 1, 3, 0, 0, 0, 0, 0]. The image rotated by 90°: [0, 0, 4, 1, 3, 0, 0, 0]. Measured
  relative to the dominant orientation: identical.
- **A2.** The same cell in an image twice as bright, and with 50 grey levels added: the
  histogram before and after normalisation to unit length.
- **A3.** Hamming distance between 1011 0010 1110 0001 and 1001 0110 1100 0011 (XOR:
  0010 0100 0010 0010, distance 4). How many bits would two unrelated 256-bit descriptors
  typically differ in?
- **A4.** Binary bits from a 3×3 smoothed patch and four given point pairs; then the same patch
  after the intensity mapping $I \mapsto I^2$ (unchanged bits) and after adding noise to one
  pixel (which bits can flip?).
- **A5.** Storage and speed: 2000 keypoints per image, 128 floats (4 bytes) vs 256 bits: memory per
  image, and the number of operations to compare all pairs between two images.
- **A6.** Predict which descriptor (patch/NCC, gradient histogram, binary) handles: a 1-px
  frame error; a gamma change; a 30° rotation without orientation assignment; light vs electron
  microscopy.

*(math)*:

- **B7.** *(math)* Prove that the unit-normalised gradient-histogram descriptor is invariant to
  $I \mapsto aI + b$, $a > 0$. What happens for $a < 0$ (contrast inversion)?
- **B8.** *(math)* Show $\|\mathbf a - \mathbf b\|^2 = 2 - 2\mathbf a\cdot\mathbf b$ for unit vectors,
  and that for the root descriptor $\|\sqrt{\mathbf a} - \sqrt{\mathbf b}\|^2 = 2 - 2\sum\sqrt{a_ib_i}$
  (Hellinger) when $\mathbf a, \mathbf b$ are L1-normalised.
- **B9.** *(math)* Mean and variance of the Hamming distance between two independent random
  256-bit strings with $P(1) = \tfrac12$; how many standard deviations below the mean is a
  distance of 60?
- **B10.** *(math)* Show that the intensity-centroid angle rotates with the image (rotate the
  coordinates in $m_{10}, m_{01}$).

### 3. `feature-matching/`: Matching descriptors — nearest, ratio and mutual

Concept layer:

- **From two lists to pairs**: $n_1$ descriptors in image 1, $n_2$ in image 2. For each in image 1,
  the **nearest** in image 2 (smallest distance). Picture: two views with all nearest-neighbour
  matches as lines: a mess of crossing lines (many wrong).
- **Why a distance threshold is not enough** (picture: histograms of the nearest-neighbour
  distance for correct and for wrong matches, from the demo; they overlap heavily): some
  features are just less distinctive than others, and in high dimensions all distances look
  alike (recall unit 2).
- **The ratio test** (Lowe; picture: the ratio $d_1/d_2$ histograms for correct and wrong
  matches, well separated): accept a match only if the best is clearly better than the second
  best, $d_1/d_2 < 0.8$. A correct match stands out from the crowd; an ambiguous one (repetitive
  texture, a common-looking corner) has a close runner-up. Numbers: distances 0.20 and 0.60 →
  0.33, accept; 0.30 and 0.32 → 0.94, reject. At 0.8 the original study kept about 95 % of the
  correct matches and removed about 90 % of the wrong ones (stated with source).
- **Mutual nearest neighbours** (cross-check; picture: a bipartite drawing with arrows both
  ways): accept $(i, j)$ only if $j$ is $i$'s nearest **and** $i$ is $j$'s nearest. Removes the
  "many points in image 1 all match the same point in image 2" failure. Cheap: one more search
  in the other direction. Combined with the ratio test: a common default.
- **One-to-one assignment** (one slide): if each point may be used at most once and we want the
  smallest total distance, it is an assignment problem (the Hungarian algorithm: pointer to
  `object-eval/` and `track-linking/`). Rarely needed here: mutual NN plus RANSAC (unit 4) is
  usually enough, and the cost grows as $n^3$.
- **How good are the matches?** (picture: precision–recall curve as the ratio threshold moves,
  with the 0.8 point marked; recall `object-eval/` vocabulary): with a known transform (a
  synthetic second view, or a hand-checked pair), a match is correct if the transformed point
  lands within ε px. Precision = correct / accepted, recall = correct / possible. Lower
  thresholds give cleaner but fewer matches; unit 4 needs **enough** correct matches with a
  **tolerable** outlier fraction, not perfection.
- **What still goes wrong** (picture strip): repetitive texture (windows, tiles, a regular
  array of wells or cells: the ratio test rejects most, the rest are confidently wrong),
  symmetric objects (a match to the mirror partner), changes in the scene (moved objects,
  divided cells: correct descriptors, wrong for the transform). The cure is geometry: unit 4.
- **Speed** (one slide): brute force compares all pairs: $n_1n_2$ distances of length 128 (2000 ×
  2000 × 128 ≈ 5·10⁸ operations: a second in a browser, a few ms on a GPU, pointer to the GPU
  elective). Faster: **search trees** (k-d trees split the space by coordinates; exact search
  degrades in high dimensions, so libraries use **approximate** nearest neighbours: a few
  percent of wrong neighbours for 100× speed), and for binary descriptors **hashing** on subsets
  of bits. Details in the optional unit.
- **Guided matching** (one line): once a rough transform is known, search only near the
  predicted position (the same idea as epipolar search in `two-view/` and gating in
  `track-linking/`).
- Summary picture: the same pair of views at each stage (all NN → ratio → mutual), with counts
  and precision.

Math layer:

- *Why the ratio works* (a simple model): the distance to the correct partner comes from a
  narrow "noise" distribution; the distances to the $n_2 - 1$ wrong ones from a broad
  "background" distribution. The second-nearest distance $d_2$ estimates the lower edge of the
  background **for this feature**, so $d_1/d_2$ compares the match to the local density of
  impostors, a per-feature threshold. Under the background alone, $d_1$ and $d_2$ are the two
  smallest of $n_2$ samples and are close together (ratio near 1).
- *Mutual NN as a constraint*: the set of mutual pairs is a matching (each point at most once);
  it is the fixed point of "nearest in both directions"; compared with the assignment problem
  $\min\sum C_{ij}x_{ij}$ (recall the linear program from `track-linking/` math, one line).
- *Precision and recall of matches*: definitions with the transform tolerance ε; the area
  under the PR curve as a single number (recall `object-eval/` AP).
- *k-d tree*: depth $\log_2 n$, exact search visits a number of leaves that grows quickly with
  dimension; best-bin-first with a fixed budget of leaves as an approximate search (stated).

**Demo `featurematch`** (new, main demo): two views side by side (view 2 = view 1 under a
known similarity/affine/homography with lighting changes and noise, or a cropped and rotated
second tile of the micrograph), keypoints and descriptors from units 1–2 (descriptor type
selectable: *histogram / binary / patch*). Matches drawn as lines, green = correct, red =
wrong (by the known transform within ε). Panels on the page (tabs on slides): (a) the views with
matches; (b) histograms of $d_1$ and of $d_1/d_2$ for correct vs wrong matches with the current
threshold as a draggable line; (c) the precision–recall curve as the ratio threshold sweeps,
current point marked, curves for *NN only*, *ratio*, *ratio + mutual*. Controls: ratio threshold
(0.5–1.0), *mutual* toggle, *one-to-one (assignment)* toggle, distance metric, number of
keypoints, view-2 transform and lighting (as `repeatability`), ε. Readouts: matches accepted,
correct, precision, recall, time for the search (brute force; a *k-d tree (approximate)*
option with a leaf budget, showing the fraction of exact neighbours found). Click a keypoint:
its nearest and second-nearest in view 2 highlighted with $d_1$, $d_2$ and the ratio. Presets:
"nearest only: a mess", "distance threshold: can't separate", "ratio 0.8", "ratio + mutual",
"repetitive grid", "binary vs histogram under blur", "approximate search: fast and almost
right", "the micrograph tiles". The output (accepted matches with the ground-truth flag) feeds
`featurealign` through `lib/featurematch.ts`.

Figures (`tools/figs/feature-matching/`): all-NN matches between two views; $d_1$ histograms
(correct vs wrong); $d_1/d_2$ histograms with 0.8 marked; bipartite cross-check sketch; PR curve;
failure strip (repetitive texture, mirror symmetry, moved object); k-d tree partition of a 2D
point set with a query; stage-by-stage summary (counts). About 10 of 18 slides with a figure or
demo.

By hand (concept):

- **A1.** A table of distances (4 points in image 1 × 5 in image 2). Find each nearest neighbour,
  apply the ratio test at 0.8, then the mutual check; which pairs survive each step?
- **A2.** Ratio test by hand: distance pairs (0.20, 0.60), (0.30, 0.32), (0.45, 0.50),
  (0.10, 0.11): accept or reject at 0.8 and at 0.6. Which ones are probably on a repetitive
  texture?
- **A3.** Precision and recall from counts: 400 possible correspondences, 250 accepted, 210 of
  them correct. What changes if the threshold is lowered and 150 are accepted with 145 correct?
  Which setting is better for unit 4, given that it needs inliers but tolerates outliers?
- **A4.** Brute-force cost for 2000 × 2000 histogram descriptors vs binary descriptors (bit
  operations on 64-bit words).
- **A5.** Predict: matching a cell-culture image to the same field after one hour: which matches
  are correct for the stage drift but wrong for the "transform" (moving cells)? What would the
  ratio test do with a regular array of wells?

*(math)*:

- **B6.** *(math)* Show that the mutual-NN pairs form a matching (no point in two pairs) and give
  a small example where mutual NN keeps fewer pairs than the optimal assignment.
- **B7.** *(math)* For $n$ background distances drawn uniformly on $[a, b]$, compute the
  expected ratio of the smallest to the second smallest when the first is shifted down by a
  true match at distance $c < a$; how does it depend on $c$?
- **B8.** *(math)* Number of distance evaluations of brute force vs a balanced k-d tree in 2
  dimensions for $n = 10^6$ (exact search visits $O(\log n)$ leaves); why the argument breaks in
  128 dimensions.

### 4. `feature-alignment/`: From matches to a transform — RANSAC, stitching and registration

Concept layer:

- **The goal**: from the matched pairs, find the transform that maps image 2 onto image 1, and
  ignore the wrong matches. Picture: matches with 40 % outliers; the transform found; image 2
  warped onto image 1 as a checkerboard overlay (seams vanish).
- **Which transform?** (picture: a square under each model, the "transform zoo"): **translation**
  (2 numbers, 1 pair), **rigid** (rotation + translation, 3 numbers, 2 pairs), **similarity**
  (+ zoom, 4, 2 pairs), **affine** (6, 3 pairs: parallel lines stay parallel), **homography**
  (8, 4 pairs: straight lines stay straight, parallel lines may meet). Rule: as simple as the
  physics allows. A microscope stage: translation (plus rotation if the sample was remounted);
  serial sections: rigid or similarity, then non-rigid (`nonrigid-reg/`); a flat object
  photographed from an angle or a camera turning about its centre: homography. More freedom
  fits more noise and needs more pairs per RANSAC sample.
- **The homography** (self-contained, one slide; `cameras/` owns it): a 3×3 matrix acting on
  homogeneous coordinates (recall `images/`), defined up to scale: 8 numbers. It maps the image
  of a **plane** seen by one camera to its image in another, and it relates **any** two images
  taken by a camera **rotating about its centre** (no parallax), which is why panoramas work
  (`two-view/` "Degenerate cases": pure rotation).
- **Fitting to clean pairs** (one slide, the least-squares idea): every pair gives two
  equations (x and y). Translation: the mean of the differences. Similarity: closed form
  (centre both point sets, best rotation from an SVD, scale from the spread; Procrustes,
  `shape-models/`). Affine: 6 unknowns, linear least squares. Homography: also linear after
  multiplying out the division (the DLT), with normalised coordinates. One wrong match pulls
  them all (picture: least-squares affine fit with 3 outliers: visibly skewed).
- **RANSAC recall** (two slides; owned by `ransac/`): pick a **minimal sample** (s pairs), fit
  exactly, map all points of image 2 and count those that land within ε px of their partner
  (**inliers**), keep the best, **refit** by least squares on its inliers. Number of iterations
  for success probability $p$: $N = \log(1-p)/\log(1-w^s)$. The built `ransac` demo (lines) as
  the reminder, then the same loop on matches in `featurealign`.
- **The numbers that matter** (table on the slide, $p = 0.99$):

  | inlier fraction $w$ | translation $s=1$ | similarity $s=2$ | affine $s=3$ | homography $s=4$ |
  |---|---|---|---|---|
  | 0.5 | 7 | 17 | 35 | 72 |
  | 0.3 | 13 | 49 | 169 | 567 |

  Simple models need few samples: one more reason to choose the simplest model. A good ratio
  test raises $w$, which matters more than anything else.
- **Choosing ε** and checking the result: ε a few pixels (larger than the keypoint precision,
  smaller than the scene's structure); ε too large accepts near-misses, too small rejects
  correct matches at the borders. After the refit: report the **number of inliers** and the
  **residual** (RMS distance of inliers); too few inliers (say < 10, or < 4 % of matches) means
  "no reliable alignment", not "a bad one". Degenerate samples (collinear points for an affine
  map, three collinear of four for a homography) are skipped.
- **Stitching** (pictures: a 3-tile mosaic with seams, then blended; and a 3-photo panorama):
  for tiles with rotation or large unknown offsets, match each overlapping pair, fit by RANSAC,
  then make the transforms consistent: chaining pairwise transforms accumulates error (recall
  `stitching/`: the tile graph and global least squares; with rotations or homographies the same
  idea becomes a nonlinear least squares, bundle adjustment, pointer to `sfm/`). **Blending**:
  feathering (weights falling to zero at tile edges) and multi-band blending (blend each level of
  the Laplacian pyramid with a different width: recall `scale-space/`); exposure differences
  are corrected per tile (gain), not by the blend.
- **Features vs intensities** (comparison table, one slide; recall `phasecorr/` and
  `matching/`): phase correlation needs no keypoints, uses all pixels, finds a translation
  precisely, and fails for rotation/zoom (unless log-polar, further reading) and small overlap;
  features handle rotation, zoom, large offsets, small overlap and partial occlusion, need
  texture, and are less precise per point. Common practice: features for the rough transform,
  then intensity-based refinement (`matching/`, `nonrigid-reg/`). Repetitive texture defeats
  both in different ways.
- **Registration in practice** (one slide, three pictures): serial sections (rigid/similarity,
  then non-rigid); a time-lapse with drift and rotation (each frame to a key frame; recall
  `stitching/` drift); correlative overviews (light/EM: few reliable keypoints, so manual
  landmarks + the same least-squares fits; one line).
- **Geometry from multiple images** (one slide, pointer): matches between two views of a 3D
  scene do **not** obey a homography (parallax): instead each match lies on an epipolar line,
  and RANSAC fits the fundamental matrix (8 or 7 pairs per sample) → `two-view/`; many views →
  `sfm/`; dense depth → `stereo/`. A homography check flags the degenerate cases.
- Summary picture: the full pipeline detect → describe → match → fit → warp, with the numbers
  of points at each stage for the demo scene.
- Pointer: image libraries call the whole pipeline "feature-based registration" or
  "panorama stitching" (one line).

Math layer:

- *The transform hierarchy in homogeneous form*: translation $\begin{psmallmatrix}I & \mathbf t\\ 0 &
  1\end{psmallmatrix}$, similarity $\begin{psmallmatrix}sR & \mathbf t\\ 0 & 1\end{psmallmatrix}$,
  affine $\begin{psmallmatrix}A & \mathbf t\\ 0 & 1\end{psmallmatrix}$, homography a general $3\times3$
  up to scale; invariants (lengths, angles, ratios of lengths, parallelism, collinearity and the
  cross-ratio).
- *Affine least squares*: each pair gives $x' = a_{11}x + a_{12}y + t_x$, $y' = a_{21}x + a_{22}y +
  t_y$; stack as $M\mathbf a = \mathbf b$ ($2n \times 6$), solve the normal equations; the x and y
  parts decouple into two $n\times3$ problems with the same matrix.
- *Similarity in closed form* (recall Procrustes): centre, $C = \sum \mathbf y_i\mathbf x_i^\top =
  U\Sigma V^\top$, $R = U\,\mathrm{diag}(1, \det UV^\top)V^\top$, $s = \operatorname{tr}(\Sigma\,\mathrm{diag}(1, \det UV^\top)) /
  \sum\|\mathbf x_i\|^2$ (centred points), $\mathbf t = \bar{\mathbf y} - sR\bar{\mathbf x}$.
- *Homography DLT* (recall `cameras/`): $\mathbf x' \times H\mathbf x = 0$ gives two independent
  rows per pair in the 9 unknowns of $\mathbf h$; stack to $A\mathbf h = 0$ ($2n\times9$); $\mathbf h$ =
  the right singular vector of the smallest singular value; normalise both point sets first
  (centroid at 0, mean distance √2) and undo it, $H = T'^{-1}\hat HT$.
- *Errors*: transfer error $\|\mathbf x' - T(\mathbf x)\|$ (used for inliers); symmetric transfer
  error for homographies; the algebraic DLT error is not geometric, so refit by Gauss–Newton
  on the transfer error (one line, recall `spot-fit/` or `sfm/`).
- *RANSAC numbers*: $1 - (1 - w^s)^N \ge p \Rightarrow N \ge \log(1-p)/\log(1-w^s)$; the table
  above; adaptive stopping (update $w$ from the best inlier count so far).
- *A rotating camera gives a homography*: $\mathbf x' \simeq KRK^{-1}\mathbf x$ (pointer to
  `cameras/`, `two-view/`).
- *Multi-band blending*: blend Laplacian-pyramid level $\ell$ with a mask blurred at that level;
  collapse (recall `scale-space/` "The math: the Laplacian pyramid is exact").

**Demo `transformzoo`** (new, small, SVG): a reference square with a grid and a few landmarks, and
up to 6 draggable target points. Model *translation / rigid / similarity / affine / homography*:
the least-squares fit from the current pairs is drawn as the warped grid, residual arrows at each
landmark, degrees of freedom and "pairs needed" shown, RMS residual readout. Too few pairs: the
model is greyed with "needs n pairs". Toggle *add an outlier* (one target far off): the fit skews.
Presets: "one pair: a translation", "two pairs: similarity", "three: affine (parallel lines stay
parallel)", "four: homography (lines meet)", "more pairs than needed: least squares",
"one outlier ruins it". Pure TypeScript, instant.

**Demo `featurealign`** (new, main demo): `data-mode="pair"` (default) and `data-mode="mosaic"`.
*Pair*: the two views from `featurematch` (same scenes and presets) with the accepted matches;
model selector as in `transformzoo`; buttons *one sample* (the s chosen pairs highlighted, the
exact fit drawn as the warped outline of image 2 on image 1, inliers within ε coloured),
*run N*, *reset*; sliders ε, iterations, seed, outlier fraction (adds seeded wrong matches on
top of the real ones), ratio threshold (re-filters the matches: shows how a better $w$ cuts
$N$). Panels: best inlier count against iteration, with the $N(w, s, p)$ prediction; the result
as a **checkerboard overlay** or **difference image** of view 1 and warped view 2; residual
readout (RMS, and the true error against the known transform); *least squares on all* vs
*RANSAC + refit* toggle. *Mosaic*: 3–4 overlapping views of a larger synthetic scene (or the
micrograph upsampled to 512², honest caption) taken with rotations and zooms (or, preset
"panorama", by a virtual camera turning about its centre: homographies); buttons *pairwise
chain*, *global refinement* (least squares on all inlier matches, via `lib/bundle.ts` for the
nonlinear case), blend *none / feather / multi-band*; readout: corner error of each tile vs
truth. Presets: "least squares on all matches fails", "RANSAC similarity: a few iterations",
"homography needs hundreds at w = 0.3", "a tighter ratio test → fewer iterations", "ε too
large", "affine on a rotating-camera pair: residual at the edges, homography fits",
"no texture: too few inliers, report failure", "mosaic: chain drifts, global fixes",
"multi-band blending". Uses `lib/ransac.ts` (generic datum, see shared engineering).

Figures (`tools/figs/feature-alignment/`): checkerboard overlay before/after; the transform zoo
(a square under five models); least-squares affine skewed by outliers; the RANSAC table needs no
image; an ε sketch (inlier band around predicted positions); 3-tile mosaic with seams, chained
vs global errors, blended result; feather vs multi-band blend detail; features-vs-intensities
table with two thumbnails; serial-section and drift thumbnails; two-view parallax picture
(matches that no homography fits, epipolar lines); pipeline summary with counts. About 12 of 24
slides with a figure or demo.

By hand (concept):

- **A1.** Name the simplest model and the minimal number of pairs: a slide-scanner tile grid;
  the same slide after remounting; two photographs of a poster from different angles; a
  panorama from a tripod; two photographs of a plant from different positions (trick: none of the
  2D models; why?).
- **A2.** Fit a translation to three pairs, (0, 0)→(3, 1), (4, 0)→(7, 1), (0, 4)→(3, 6): the
  least-squares translation (3, 4/3) and the residuals (0, −1/3), (0, −1/3), (0, 2/3); which
  pair is suspicious? Then a similarity from the first two pairs only.
- **A3.** RANSAC iterations for $p = 0.99$: complete the table above for $w = 0.7$ (s = 1…4: 4,
  7, 11, 17); explain why a ratio test that removes half of the wrong matches but 5 % of the
  right ones is worth it for a homography.
- **A4.** Given an affine transform $A, \mathbf t$ (integers) and four matches, which are inliers
  for ε = 2 px?
- **A5.** Chaining: four tiles in a row, each pairwise rotation estimate off by 0.2°: the angle
  error of tile 4 relative to tile 1, and the position error at its far edge (tile width 1000 px).
  What does a global fit do with the redundant pairs if the tiles also overlap diagonally?
- **A6.** Predict for each case whether phase correlation or features is the better first
  choice: tiles with 10 % overlap and a known stage grid; sections rotated by up to 180°; an
  almost empty tile; two time points with a 3× zoom change.

*(math)*:

- **B7.** *(math)* Write the $2n\times6$ least-squares system for the affine fit and show that it
  splits into two $n\times3$ systems with the same matrix.
- **B8.** *(math)* Derive the two DLT rows of one pair from $\mathbf x' \times H\mathbf x = 0$ and
  show that the third row is a combination of the two.
- **B9.** *(math)* Derive $N = \log(1-p)/\log(1-w^s)$ and the approximation $N \approx
  -\log(1-p)/w^s$ for small $w^s$.
- **B10.** *(math)* Show that a camera rotating about its centre gives $\mathbf x' \simeq
  KRK^{-1}\mathbf x$, a homography (one step from $\mathbf x \simeq K\mathbf X$).

### 5. (optional / further reading) `feature-search/`: Many images — fast search, visual words and learned features

- **Approximate nearest neighbours** in depth: k-d trees with a leaf budget, randomised forests
  of trees, hashing for binary descriptors; accuracy vs speed curves.
- **Visual words** (bag of features): cluster descriptors with k-means (self-contained recall,
  pointer to `kmeans/`), describe an image by the histogram of its words, compare images by
  histogram similarity with term weighting: **image retrieval** in large collections, finding
  overlapping pairs before matching (which tiles/photos overlap at all?), loop closure in
  reconstruction (`sfm/`).
- **Learned keypoints and descriptors** (one or two slides): a network trained on many image
  pairs outputs keypoints and descriptors, or matches directly with attention between the two
  sets (pointer to `attention/`); more robust to lighting and viewpoint; same downstream steps
  (ratio test, RANSAC).
- **Demo `visualwords`** (small): 40 synthetic "scenes" (seeded) and the micrograph tiles; build
  a vocabulary (k slider), show the word histogram of a query and the ranked retrieved images;
  presets "k too small: everything looks alike", "k = 200", "a rotated query is still found".
- By hand: a 5-word vocabulary, word histograms of three images, cosine similarities, ranking.

## Proposed reorganisation

Moves that clearly improve the course (other plans and built units are **not** edited here;
listed for the integration pass):

1. **Feature matching moves from `two-view/` to this day.** The built `two-view/` notes say
   "This unit owns matching interest points between images (descriptors, ratio test,
   cross-check)", and `plan/multiview-day.md` left the question open ("give them to the Motion
   day and recall here?"); its descriptor helpers were never written. Proposal: `two-view/`
   condenses the slides "Correspondences first" and "Matching, and why it still goes wrong" into
   **one recall slide** (keeping the ratio numbers 0.20/0.60 and 0.30/0.32, which agree with
   unit 3 here) with links to `descriptors/` and `feature-matching/`; its notes line on
   ownership changes accordingly. `plan/multiview-day.md`'s planned helpers (`patchDescriptor`,
   `matchRatio`, `crossCheck`) are provided by `lib/descriptors.ts` and `lib/featurematch.ts`.
   A later optional preset of `epipolar` could run on real matches from these libraries.
2. **The application of RANSAC to transforms between images moves from the "planned Motion
   elective" to this day** (unit 4). The Motion group was built without it (optical flow,
   tracking). `ransac/lec.qmd` "Where RANSAC is standard", first bullet: "This application
   belongs to the planned Motion elective…" → link to `feature-alignment/`. RANSAC itself stays
   owned by `ransac/`.
3. **Course map, owners table**: add the row "Keypoints across scale and rotation, local
   descriptors (gradient histograms, binary), feature matching (ratio test, mutual NN), fitting
   2D transforms to matches → Keypoints day (`keypoints/`, `descriptors/`,
   `feature-matching/`, `feature-alignment/`)". Homography stays with `cameras/`, Procrustes with
   `shape-models/`.
4. **Pointer edits** (one line each): `structure-tensor/` "Interest points" (keypoint detectors and
   descriptors → `keypoints/`, `descriptors/`); `scale-space/` "Where this leads" (the SIFT line →
   `keypoints/`); `stitching/` "Limits of a shift", bullet "Features instead of pixels" →
   `feature-alignment/`; `matching/` after the intensity-vs-landmark table ("Feature points and
   how to match them: the edges day … and the Motion elective" → this day); `lucas-kanade/`
   "tracking points as a pointer" may add "large motions: match keypoints (`feature-matching/`)".
5. **Entry and sidebar**: keep the hard entry "after day 2" (all real prerequisites are on day 2),
   but list `ransac/` (Hough) and the FFT registration day as **soft** prerequisites in the
   planned prerequisite database. In the sidebar, place the day **last in the Detection group**
   (after Point detection), titled "Keypoints and descriptors". The multiview day gains this
   day as a soft prerequisite for `two-view/` and `sfm/`.
6. **HOG**: the SVM plan's unbuilt `detect-window/` (HOG detector) should recall the cell
   histogram of gradient directions from `descriptors/` instead of teaching it (one slide there).

Not proposed: moving RANSAC itself here. It fits the Hough day's voting-vs-verification theme,
it is built, and its demo is reused here.

## Shared engineering

Reuse (no local copies; extend through the owners):

- `lib/imageops.ts`: `gaussianBlur`, `gaussianDerivative`, `gradient`, `dogFilter`, `pyrDown`,
  `gaussianPyramid`, `laplacianPyramid`, `collapsePyramid` (multi-band blending), `warp`,
  `sampleBilinear` (resampling windows in the keypoint frame, warping view 2), `resize`,
  `rotate`.
- `lib/edges.ts` (edges day): `structureTensor`, `harris`, `minEig`, `localMaxima2D`,
  `scaleSpaceStack`, `extrema3D`, `hessEig`. The octave DoG pyramid is new (below) but must
  agree with `extrema3D` on a single octave (test).
- `lib/ransac.ts` (owner: Hough day): generic `ransac()`, `iterationsNeeded`, the step recorder.
  **Request (made backwards compatible by this day if the owner agrees):** make the model generic
  in the datum type, `Model<M, D = P2>` (and `ransac<M, D>`, `inliersOf<M, D>`), so that a match
  pair `{a: P2, b: P2}` is a datum. The line/circle models and the `ransac` demo keep compiling
  unchanged. `lib/twoview.ts`'s private `ransacF` loop can then move onto it later (not part of
  this day).
- `lib/camera.ts` (multiview day): `homographyDLT`, `applyH`, `normalise2d` (do not duplicate).
- `lib/shapes.ts`: `procrustes` (similarity fit; the plan once wanted it in `linalg.ts`, see
  build-issues).
- `lib/linalg.ts`: `svd`, `lstsq`, `solve`, `eigSym2`, `inv3`, `det3`.
- `lib/linking.ts`: `assignDense` (one-to-one assignment option). Note: `matchPoints` in
  `lib/linking.ts` and `lib/spotdetect.ts` are **spatial** matchers; this day's matchers are in
  descriptor space and get distinct names (`matchDescriptors`, below).
- `lib/registration.ts` (FFT registration day): `parabolaOffset` (1D three-point refinement, the
  orientation peak), `solveGlobal`/`maxSpanningTree` (the translation part of the mosaic's
  global step, as a comparison).
- `lib/bundle.ts` (multiview day): Levenberg–Marquardt for the mosaic's global refinement with
  similarity/homography parameters (small problems).
- `lib/random.ts` (`rng`, `sampleIndices`, `shuffle`), `lib/plot.ts` (histograms, PR curves,
  inlier curves, polar roses via `PolarPlot`), `lib/colour.ts` (diverging maps for DoG,
  `categoryColor` for correct/wrong), `lib/histogram.ts`, `lib/metrics.ts` (`precision`,
  `recall`), `lib/kmeans.ts` (optional unit), `lib/cell.ts` (`loadCell`, `CELL_CREDIT`),
  `lib/ui.ts` helpers.
- **`lib/synth.ts`** (shared foundation; additive change): `featureScene(opts)` — a seeded,
  photo-like synthetic scene (polygons with corners, discs at several radii, a repetitive grid
  patch, textured regions, gentle shading), sized to 512² so that views and mosaics can be cut
  from it; `viewOf(scene, T, opts)` is in `lib/transforms2d.ts` (below).

New library files owned by this day (names checked against `demos/src/lib/`; none exist):

- **`lib/keypoints.ts`** (no DOM): `Keypoint {x, y, sigma, theta, response, octave, level}`;
  `dogPyramid(img, {sigma0, S, octaves, upsample})` → blurred and DoG levels per octave
  (incremental blur); `dogCandidates`, `refineExtremum` (3D quadratic, iterate to the neighbour
  when the offset exceeds 0.5, returns the offset and the peak value), `edgeRatioTest(r)`,
  `contrastTest`; `detectDoG(img, opts)` (all stages, with a stage recorder for `dogpyramid`);
  `detectHarris(img, σ, k)` and `detectHarrisMultiScale(img, σs)` (normalised, Laplacian scale
  selection); `detectFAST(img, t, n = 9)`; `assignOrientations(img or pyramid, kps, {bins: 36,
  peakRatio: 0.8})`; `repeatability(kps1, kps2, T, {eps, scaleTol, region})`; `keepStrongest`.
- **`lib/descriptors.ts`** (no DOM): `sampleFrame(img, kp, size)` (window resampled in the
  keypoint frame); `patchDescriptor` (normalised patch, NCC); `gradHistDescriptor(img, kp,
  {cells: 4, bins: 8, soft: true, clip: 0.2, root: false})` → Float32Array(128);
  `binaryPattern(seed, n = 256, patchSize)` (seeded pairs), `binaryDescriptor(img, kp, pattern,
  {steer: true})` → Uint32Array(8); `intensityCentroidAngle`; distances `l2`, `l1`, `chi2`,
  `hamming` (XOR + popcount on 32-bit words), `distanceFn(kind)`.
- **`lib/featurematch.ts`** (no DOM): `matchDescriptors(D1, D2, {metric, k: 2})` → for each
  query the two nearest with distances (brute force, typed arrays); `ratioTest`, `mutualCheck`,
  `oneToOne` (via `lib/linking.ts` `assignDense`); `evaluateMatches(matches, kps1, kps2, T, eps)`
  → correct flags, precision, recall; `prCurve` over the ratio threshold; `KdTree` (build,
  exact search, best-bin-first with a leaf budget) for the speed preset; `Match {i, j, d1, d2}`.
- **`lib/transforms2d.ts`** (no DOM): `Transform2D` (3×3 row-major `Float64Array`) with
  `translationFit`, `rigidFit`, `similarityFit` (via `procrustes`), `affineFit` (least squares),
  `homographyFit` (via `camera.ts` `homographyDLT`), `minimalSize(kind)`, `dof(kind)`;
  `apply`, `invert`, `compose`, `transferError`, `symmetricTransferError`; RANSAC models for
  `lib/ransac.ts` with datum = match pair, degeneracy checks (collinear samples); `refineGN`
  (Gauss–Newton on transfer error for the homography); `viewOf(scene, T, {gain, offset, gamma,
  blur, noise, seed, size})` → view + ground-truth transform; `warpInto(mosaic, img, T)`,
  `featherWeights`, `multibandBlend` (Laplacian pyramid from `imageops.ts`); mosaic generator
  (tiles with rotations/zooms; virtual rotating camera via $KRK^{-1}$).

Speed (TypeScript first, measured): a DoG pyramid for 256² (4 octaves, S = 3, 6 blurs per
octave, incremental σ) is roughly 20–40 ms; orientation + 128-D descriptors for 300 keypoints
a few ms; brute-force matching 300 × 300 × 128 ≈ 1.2·10⁷ multiply-adds ≈ 10–20 ms; binary
Hamming much less; RANSAC iterations microseconds each. View 1's features are cached; a
transform slider recomputes view 2 (detect + describe + match) in an estimated 40–70 ms, which
exceeds the ~30 ms rule. Plan: coalesce updates to one per animation frame with a "computing…"
state, recompute on slider release for the repeatability sweep (cached per detector), measure in
the browser first. Only if the drag still feels slow, move `dogPyramid` + `gradHistDescriptor`
(and only them) to Rust → Wasm in `demos/wasm/keypoints/` (flat `&[f32]`), keeping the TypeScript
versions as reference with an agreement test. The mosaic (4 × 256² views) computes once per
preset.

Tests (`demos/test/keypoints-test.ts`, run by `make test`):

- `dogPyramid` level σ's follow $\sigma_0k^i$; DoG on one octave matches `scaleSpaceStack` +
  `extrema3D` positions; a synthetic Gaussian blob of width $s$ is detected at σ within one
  level of $s$ and with sub-pixel error < 0.1 px after `refineExtremum`; the edge test rejects
  candidates along a straight bar and keeps the corners of a square.
- Repeatability of DoG on a scene and its 2× zoom > 0.6 and > repeatability of single-scale Harris
  (the numbers quoted on the slides come from this test); rotation by 90° gives repeatability
  close to 1 (exact resampling).
- Orientation assignment rotates with the image (error < 3° for 30° and 120° rotations).
- Gradient-histogram descriptor: unit length; identical under $I \mapsto 3I + 20$; the A1 cell
  example reproduces [4, 1, 3, 0, 0, 0, 0, 0]; the descriptor of a rotated view (with θ) is
  nearer to the original than to any other descriptor for > 90 % of keypoints.
- Binary descriptor unchanged under a strictly increasing intensity map; `hamming` against a
  bit-by-bit reference; the A3 example gives 4.
- `ratioTest` and `mutualCheck` on the A1 distance table of unit 3; `KdTree` exact search equals
  brute force; `evaluateMatches` counts against a known transform.
- `transforms2d`: each fit recovers its generating transform exactly from noiseless minimal
  samples; `homographyFit` agrees with `camera.ts` `homographyDLT`; RANSAC with a fixed seed on
  matches with 50 % outliers recovers a similarity to < 0.5 px; `iterationsNeeded` reproduces the
  slide table (7, 17, 35, 72; 13, 49, 169, 567; 4, 7, 11, 17).
- The generic `lib/ransac.ts` change: the existing `hough-test.ts` RANSAC tests pass unchanged.

Follow `demos/README.md` and `plan/build-issues.md`: TypeScript, `make typecheck`, no network,
fixed seeds, presets + reset, the `slider`/`button`/`checkbox` helpers, no keyboard capture,
labels ≥ 18 px on slides (including histogram axes, PR curves and readouts), fit 1150×560 on a
slide (`repeatability`, `featurematch`, `featurealign` show two views + one plot on slides and
all panels with `data-layout="full"` on the exercise page), `data-*` settings (`data-mode`,
`data-preset`, `data-scene`, `data-tab`). **Prefix every per-demo CSS selector with
`.qb-demo[data-demo="…"]`** (demo.css loads late). In the decks, write static figures as
`![](fig.svg){fig-alt="..."}` (bracket text becomes a visible caption on slides) and add
`.nostretch` to a lone slide image. Static figures as print fallbacks. Every demo goes on
`demos.qmd`; the sidebar gets "Keypoints and descriptors" in the Detection group (see
reorganisation 5).

Images: `cell256.png` (credit as in `lib/cell.ts`, `CELL_CREDIT`) and synthetic scenes labelled as
synthetic. A real overlapping photo pair or tile set would help (open question 3).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `keypoints/`: lecture (`structtensor` recall, `dogpyramid`), octaves, refinement and edge test by hand, `repeatability` |
| 10:45–12:15 | Unit 2 `descriptors/`: lecture, cell histogram and Hamming by hand, `descriptorlab` |
| 13:15–14:45 | Unit 3 `feature-matching/`: lecture, ratio and mutual check by hand, `featurematch` |
| 15:00–16:30 | Unit 4 `feature-alignment/`: lecture (`ransac` recall, `transformzoo`), RANSAC numbers and fits by hand, `featurealign` (pair, then mosaic); quizzes as wrap-up |

## Open questions

1. **Ownership** (reorganisation 1–3): does this day take feature matching from `two-view/`, and
   the transform-fitting application of RANSAC from the never-written Motion elective? Proposed:
   yes, with recalls there.
2. **Group and order**: Detection (as the course map says; detection of points is the first half)
   or Part III "Aligning images" / the Motion group (the second half is registration)? Proposed:
   Detection, last in the group, with `feature-alignment/` cross-linked from the FFT registration
   day and the Motion group.
3. **Real images**: a licence-compatible pair or triple of overlapping photographs (a panorama)
   and overlapping microscopy tiles with rotation (the FFT registration day needs tiles too: one
   search for both). Otherwise synthetic scenes plus the micrograph, honestly captioned.
4. **Binary descriptor depth**: one concept slide + math (proposed), or also the learned pair
   selection (choosing uncorrelated, high-variance tests) as a math slide?
5. **Affine-covariant detectors** (ellipses from the second-moment matrix, MSER) and viewpoint
   change: one line each (proposed) or an optional unit?
6. **Wasm**: decide after measuring the view-2 recompute in the browser (see speed).
7. **Mosaic global refinement**: reuse `lib/bundle.ts` (multiview) for similarity/homography
   tiles, or keep the mosaic to translations + rotations solved linearly (small-angle)? Proposed:
   `bundle.ts`, small problem, already tested.
8. **Numbers from Lowe (2004)** (95 % / 90 % at ratio 0.8): quote with the source and show the
   demo's own numbers next to them (they will differ on synthetic scenes).
9. **Optional unit 5**: build it, or keep it as further reading only, with the demo deferred?

## References

- Lowe (2004), *Distinctive image features from scale-invariant keypoints*, IJCV (DoG keypoints,
  refinement, edge test, orientation, gradient-histogram descriptor, ratio test).
- Lowe (1999), *Object recognition from local scale-invariant features*, ICCV.
- Harris & Stephens (1988), *A combined corner and edge detector*, Alvey Vision Conf. (recall).
- Shi & Tomasi (1994), *Good features to track*, CVPR (recall).
- Lindeberg (1998), *Feature detection with automatic scale selection*, IJCV (recall).
- Mikolajczyk & Schmid (2004), *Scale & affine invariant interest point detectors*, IJCV
  (Harris–Laplace, affine adaptation).
- Mikolajczyk et al. (2005), *A comparison of affine region detectors*, IJCV (repeatability).
- Mikolajczyk & Schmid (2005), *A performance evaluation of local descriptors*, PAMI.
- Matas, Chum, Urban & Pajdla (2002), *Robust wide baseline stereo from maximally stable
  extremal regions*, BMVC (MSER).
- Rosten & Drummond (2006), *Machine learning for high-speed corner detection*, ECCV (FAST).
- Bay, Ess, Tuytelaars & Van Gool (2008), *Speeded-up robust features (SURF)*, CVIU (pointer).
- Calonder, Lepetit, Strecha & Fua (2010), *BRIEF: binary robust independent elementary
  features*, ECCV.
- Rublee, Rabaud, Konolige & Bradski (2011), *ORB: an efficient alternative to SIFT or SURF*,
  ICCV (steered binary tests, intensity centroid).
- Rosin (1999), *Measuring corner properties*, CVIU (intensity centroid).
- Arandjelović & Zisserman (2012), *Three things everyone should know to improve object
  retrieval*, CVPR (root descriptor / Hellinger).
- Muja & Lowe (2014), *Scalable nearest neighbor algorithms for high dimensional data*, PAMI
  (approximate search).
- Beis & Lowe (1997), *Shape indexing using approximate nearest-neighbour search in
  high-dimensional spaces*, CVPR (best-bin-first).
- Sivic & Zisserman (2003), *Video Google: a text retrieval approach to object matching in
  videos*, ICCV (visual words; optional unit).
- Fischler & Bolles (1981), *Random sample consensus*, CACM (recall; owned by the Hough day).
- Hartley & Zisserman (2004), *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge
  (homography, DLT, normalisation, transfer errors).
- Brown & Lowe (2007), *Automatic panoramic image stitching using invariant features*, IJCV
  (matching, RANSAC, bundle adjustment, multi-band blending).
- Burt & Adelson (1983), *A multiresolution spline with application to image mosaics*, ACM TOG
  (multi-band blending).
- Szeliski (2006), *Image alignment and stitching: a tutorial*, Foundations and Trends in
  Computer Graphics and Vision; Szeliski (2022), *Computer Vision: Algorithms and
  Applications*, 2nd ed., ch. 7–8 (the best single reading for the day).
- Saalfeld, Fetter, Cardona & Tomancak (2012), *Elastic volume reconstruction from series of
  ultra-thin microscopy sections*, Nature Methods (features + RANSAC for serial sections; teacher
  background, no tool names in student material).
- DeTone, Malisiewicz & Rabinovich (2018), *SuperPoint: self-supervised interest point detection
  and description*, CVPRW; Sarlin et al. (2020), *SuperGlue: learning feature matching with graph
  neural networks*, CVPR (learned features; optional unit, teacher background).

## Build status (2026-11-13)

Built: the four core units `keypoints/`, `descriptors/`, `feature-matching/`, `feature-alignment/`
(lec, quiz, exercise with #explore), six demos (`dogpyramid`, `repeatability`, `descriptorlab`,
`featurematch`, `transformzoo`, `featurealign` with `data-mode="pair|mosaic"`), the libraries
`lib/keypoints.ts`, `lib/descriptors.ts`, `lib/featurematch.ts`, `lib/transforms2d.ts`, the test
`demos/test/keypoints-test.ts` (86 checks), figure scripts in `tools/figs/keypoints/` (export.ts for
the data with the course libraries, kplib.py, figs.py) and `tools/figs/{descriptors,feature-matching,
feature-alignment}/figs.py`. Shared demo helpers for the four image demos:
`demos/src/repeatability/kpcommon.ts` (scenes, views, detect/describe with caching, PairView).
Not built: the optional unit 5 `feature-search/` and its `visualwords` demo.

Deviations from the plan (for the integrator):

- `featureScene` and `gridScene` live in `lib/transforms2d.ts`, not in `lib/synth.ts` (an existing
  file this day may not edit). Move them when `synth.ts` is next edited.
- `lib/ransac.ts` was not made generic in the datum type (existing file). `transforms2d.ts` has its
  own RANSAC loop over pairs (`ransacTransform`) that reuses `drawSample` and `iterationsNeeded`.
  The proposed `Model<M, D = P2>` change is still open.
- `keypoints.ts` has its own fast separable Gaussian (`fastGauss`) and Harris response; the
  imageops/edges versions were too slow for the multi-scale detectors (measured: DoG pyramid of 256²
  ≈ 90 ms in node on the build machine; multi-scale Harris 1.4 s → 0.26 s by computing large scales
  on downsampled images). No Wasm: recomputation is coalesced to one per frame with a "computing…"
  readout; the repeatability sweep runs in the background, one zoom per timer tick.
- Repeatability counts one-to-one (each point of image 2 used once); without that, single-scale
  detectors scored > 100 % under tilt.
- Plan claims corrected by measurement: (1) a large square gives no DoG keypoints at its corners
  (its side candidates fail the edge test and it is kept once, as a blob at its centre) — the test
  checks that instead; (2) the normalised patch is only slightly more fragile than the gradient
  histogram under a frame error in this demo (2 px: 86 % vs 90 % nearest = partner), because the
  window is 12σ wide and sampled from the blurred level — the deck says so; (3) second orientation
  peaks: 45 % extra keypoints on the synthetic scene (Lowe: about 15 % on photographs);
  (4) multi-scale Harris (our simple Harris–Laplace) is weaker than DoG on the synthetic scene;
  (5) "tilt 50°" degrades DoG (48 %) much more than single-scale Harris (80 %) and FAST (85 %).
- `featurealign` mosaic: similarity tiles only (global step linear, no `bundle.ts`); the
  "panorama" (rotating camera) mosaic preset is not built; the rotating-camera case is shown in
  pair mode (preset "camera"). Gain compensation is described, not implemented.
- `repeatability`: the click shows predicted position and nearest detection in a readout instead
  of the plan's σ-response inset.
- Real images: synthetic scenes plus the course micrograph (upsampled 2× to 512², captioned).

Placement: Detection group, last (after Point detection), section "Keypoints and descriptors";
the reorganisation proposals above are unchanged.
