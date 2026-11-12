# Review: is key content missing?

Internal working doc (plan/ is not rendered). Status: review, 2026-11-12.

**Inputs read:** all 28 day plans in `plan/` (there are 28, not 33), the built units in
`_quarto.yml` (Parts 1–4: `images`, `kernels`, `fourier`, the graph-cut, VAE and bit-depth
units), `audience.md` and `designprinciples.md`. The yardstick: standard image
processing/analysis textbooks (Gonzalez & Woods, Sonka–Hlavac–Boyle, Szeliski,
Jähne), bioimage-analysis syllabi (NEUBIAS, EMBL, Bankhead's *Analyzing fluorescence
microscopy images*), and what a practitioner meets in a first real project.

## Summary verdict

The plans are deep and consistent, and they do the demo-first, theory-not-tools approach
well. But the coverage is uneven. **Segmentation by energy and contours has about eight
days** (graph cuts, Ising, level sets, snakes/ASM, dynamic programming, watershed,
thresholding, GMM). The material on **where an image comes from** (optics, PSF, noise)
and how to **undo** it (deconvolution, denoising) has almost none. Nothing covers
**motion** (non-rigid registration, optical flow, tracking). **Modern deep learning**
gets one draft day plus the VAE part. The **low-level features** every later day assumes
(scale space, LoG blobs, structure tensor, Canny) have no owner: each plan points to
another plan or covers them as a recall. Several plans' own open questions flag these
gaps. They are verified below.

Meta-gaps found along the way:

- **No course map.** Nothing says which of the ~28 days plus Part 1 are core and which
  are electives, or in what order they come. Every plan asks "how long is a day?" and
  "which day comes first?". That count is 2–3 semesters of material. A one-page
  `plan/course-map.md` (core track, electives, prerequisite graph, owner of each shared
  concept) would settle most of the open questions about ownership.
- **Part 1 is thin** (3 units) and has no plan file. Many days assume Part 1 material
  that it does not contain (see gap 2).

## Gaps flagged in the plans' open questions, and whether they hold

| Flagged in | Gap | Verified? | Where it goes below |
|---|---|---|---|
| `bitdepth` | Separate imaging/noise day (shot/read noise, photon transfer) | **Yes**: shot noise is only in `quantisation/`. No plan has a PSF, noise model or deconvolution. | Gap 1 |
| `fftregistration` | Non-rigid registration / optical flow: "no other plan covers it" | **Yes**: no plan mentions it beyond that line | Gap 5 |
| `hough` | Canny / edge detection: "no plan owns a full pipeline" | **Yes**: it exists only as the compact unit 1 of the Hough day | Gap 2 |
| `wavelets` | LoG / scale-space blob detection, promised by `fourier/exercise.qmd`, has no owner | **Yes**: optional `cwt/` and a feature-stack slide in RF/SVM only | Gap 2 |
| `randomforest` | Structure tensor: "no plan owns it" | **Yes**: one slide in RF unit 4, one mention in `radon` | Gap 2 / 5 |
| `moments` | Contour descriptors (Fourier descriptors): no plan covers them | **Yes** | Gap 13 |
| `thresholding` → `labelling` | Object-level matching and counting metrics: `seg-eval/` points to the labelling day | **No owner**: the labelling day counts objects but has no matching or object F1/AP | Gap 4 |
| `neuralnets` | Transformers: one pointer slide or a unit? | Pointer only, and so are self-supervised learning and foundation models | Gap 3 |
| `vae` | GANs / diffusion: pointers only | **Yes** | Gap 3 |
| `marchingcubes`, `measurements`, `morphology`, `moments` | A "3D images" unit (anisotropy, 3D filtering, 3D measurement) | **Yes**: each plan has a slide, no plan owns it | Gap 8 |
| `snakes`, `compression`, `gmm`, `vae` | Who owns PCA? | **Partly resolved**: the built `autoencoder/` unit teaches PCA, but it sits inside the VAE part. A shared owner is still needed. | Coordination |
| `integral`, `randomforest` | Who owns boosting / trees? | **Resolved** by the RF plan's proposal (stumps/AdaBoost on the integral day, trees/forests on the RF day) | – |
| `colour` | JPEG DCT "has no home" | **Stale**: the compression day owns it | – |
| `colour` | Hyperspectral / spectral unmixing | **Stale**: the unmixing day exists | – |
| `bitdepth` | Compressed formats as a unit | **Stale**: the compression day covers them | – |
| `compression` | Video compression | Agree: out of scope; a pointer is enough | – |
| `integral` | Bias/fairness slide for face detection | Real, but too narrow: it belongs in a course-level unit | Gap 7 |

## Ranked gap list

Importance: **must-have** (a graduate of a modern image analysis course is expected to
know it, and practitioners need it), **should-have** (standard in good syllabi, often
needed), **nice-to-have** (a pointer is enough).

### 1. Image formation, noise and restoration: **must-have** → new day

**Why it matters.** Every analysis starts from an image that is *object ∗ PSF, sampled,
plus Poisson–Gaussian noise*. Without this model, students cannot say why a nucleus edge
is blurred, which pixel size to use (Nyquist for a microscope), why the deconvolution
button makes things worse, or why the noise is not Gaussian at low counts. It is
also the natural home of a central linear-algebra/Fourier payoff, the ill-posed inverse
problem. The Radon day's algebraic reconstruction and the Ising day's TV prior already
build half of it. Today the course has `fourier/` (convolution theorem, aliasing),
shot noise in `quantisation/`, wavelet shrinkage, an MRF prior and VAE denoising. These
are scattered pieces with no forward model to tie them together.

**Smallest coverage:** one new day, "From light to pixels and back". It fits after Part 1
and the bit-depth day.

1. `psf/`: **Image formation as convolution.** Point sources → PSF; the OTF as the PSF's
   Fourier transform; the diffraction limit (Abbe/Rayleigh: $d \approx 0.61\lambda/\mathrm{NA}$)
   as an OTF cut-off; pixel size and Nyquist sampling for a microscope; widefield vs
   confocal PSF in 3D as one picture (the axial elongation).
2. `noise-models/`: **Camera noise model.** Poisson shot noise plus Gaussian read noise,
   gain and offset; the photon-transfer curve (variance vs mean is a line, and its slope
   is the gain); signal-dependent noise, the Anscombe transform; flat-field and dark-frame
   correction as the per-pixel inverse of the model. This takes over the camera part of
   `quantisation/`, as the bit-depth plan's open question proposes.
3. `deconv/`: **Deconvolution as an inverse problem.** Inverse filtering blows up noise
   where the OTF is small; Wiener filtering (per-frequency SNR weighting) and Tikhonov
   regularisation; the link to least squares and the condition number (unmixing and Radon
   days), and to TV (Ising day).
4. `richardson-lucy/`: **Iterative restoration.** Richardson–Lucy as the maximum-likelihood
   estimate under Poisson noise (the derivation is a good math layer); the number of
   iterations acts as regularisation; ringing; one slide comparing denoisers (median,
   bilateral, non-local means, wavelet shrinkage, self-supervised networks of the
   Noise2Void kind, as "learned denoisers" with no names) and when each one changes your
   measurements. Optional extension: `localisation/` (gap 9).

**Demo ideas.**
- `psfotf`: sliders for NA, wavelength and pixel size. Panels show the Airy PSF, the OTF
  with its cut-off circle, and two point sources moving apart: the Rayleigh criterion
  becomes visible, and undersampling aliases.
- `deconvlab`: a blurred, noisy synthetic image with ground truth. Methods: inverse, Wiener
  (SNR slider), Tikhonov (λ), RL (iteration slider). Panels show the image, its spectrum and
  the error vs truth. Presets: "inverse filter: noise explodes", "RL too long: ringing and
  noise", "wrong PSF".

### 2. Scale space, edges and blob/spot detection: **must-have** → two units in Part 1

**Why it matters.** Spot and blob detection (FISH, vesicles, nuclei centres) is among the
most common bioimage tasks. Scale is the most important idea in low-level vision. And at
least six plans *use* LoG/DoG, the Hessian, the structure tensor or Canny (Hough, RF, SVM,
skeleton/vesselness, wavelets, snakes) while each assumes another plan teaches them.
`kernels/` gives LoG and "scale" one bullet each. This gap is about ownership, not
importance, and it is cheap to close.

**Smallest coverage:** two new Part 1 units after `fourier/`.

- `scalespace/`: **Scale space and blob detection.** The Gaussian scale space
  $L_\sigma$ (the heat equation as a math layer); why there are no new extrema; the
  scale-normalised LoG and DoG; automatic scale selection (blob radius $\approx \sqrt2\sigma$);
  the Hessian and its eigenvalues (blob/ridge/saddle), which the skeleton day's
  vesselness can recall; Gaussian pyramids. Demo `blobscale`: a spot image with a σ slider,
  the LoG response, and a scale-normalised response curve for a clicked spot whose maximum
  gives its size; a "detect all" mode with NMS in $(x, y, \sigma)$.
- `edges/`: **Edges, corners and orientation.** Canny moved here from the Hough day
  (Hough unit 1 becomes a recall, as its open question suggests); the structure tensor
  (local orientation and coherence, Harris/Shi–Tomasi corners as eigenvalues). It is
  pure linear algebra (eigenvalues of a 2×2 symmetric matrix) and serves RF, registration
  (gap 5) and optical flow. Demo `structtensor`: ellipses drawn over the image with
  σ and ρ sliders and a corner-response overlay; presets "edge: one big eigenvalue",
  "corner: two", "flat: none".

### 3. Modern deep learning for images: **must-have** (core) + **should-have** (frontier) → make the NN day core, add one day

**Why it matters.** Practitioners now segment cells with generalist pretrained networks,
detect objects with deep detectors, and increasingly restore images with learned models.
The built Part 3 teaches VAEs *before* the course has a CNN or U-Net. The neural-network
day (`plan/neuralnets-day.md`) is still a draft. Its optional `deep-practice/` holds
shortcut learning, domain shift and transfer learning, which are what practitioners most
need. Instance segmentation, detection, self-supervised and foundation models, and
diffusion are pointers only. For this audience the ideas are accessible (a layer is a
matrix, a loss is a function to descend), so theory-first does not excuse leaving them out.

**Smallest coverage:**
- **Make `plan/neuralnets-day.md` core and place it before the VAE part.** Promote
  `deep-practice/` to core, or at least its shortcut-learning and domain-shift half.
- **New day "Deep learning in practice: instances, pretraining, generation":**
  1. `instance-seg/`: from a semantic mask to instances. Boundary class, distance
     map + watershed, star-convex polygons (rays from the centre), flow fields to the
     centre followed to a sink. These are the ideas behind the generalist cell
     segmenters, named in one line only. Strong links to the watershed and labelling
     days.
  2. `detection/`: boxes, IoU, anchors/heatmaps (centre-point detection), non-maximum
     suppression, precision–recall and average precision (shared with gap 4).
  3. `pretrained/`: transfer learning, self-supervised pretraining (contrastive and
     masked-image ideas as pictures), embeddings as features (links to measurements,
     RF and SVM days), promptable "segment anything"-style models (a point or box prompt
     → mask), and what they still get wrong on microscopy. The ViT "image as patches"
     slide goes here.
  4. `diffusion/`: generative models beyond the VAE. Forward noising, a learned denoiser,
     sampling by repeated denoising; the score of a Gaussian mixture as the toy exact case
     (reuses the GMM day). Content-aware restoration and **hallucination**: why a
     generated detail is not a measurement (ties to gaps 1 and 7).

**Demo ideas.**
- `flowinstances`: a synthetic image of touching nuclei with its "predicted" centre-flow
  field (computed from ground truth plus a noise slider). Particles follow the flow and
  the sinks become instances. A star-convex mode with n rays per pixel; presets "two
  touching nuclei separate", "too much noise: over-splitting", "non-convex cell: rays
  fail".
- `diffusion2d`: a 2D point cloud (the GMM day's datasets) with forward noising over a time
  slider and reverse sampling with the exact mixture score. A coarser "learned" score
  shows mode dropping and samples between modes.
- Alternative: `embedmap`, with precomputed patch embeddings of the micrograph. Click a
  patch to see a similarity map. This shows "pretrained features already know what a
  nucleus is" without training.

### 4. Validation, object-level evaluation and statistics of measurements: **must-have** → one unit + promote one

**Why it matters.** The thresholding day's `seg-eval/` is good on pixel metrics, and it
points to the labelling day for **object-level matching**. That day only counts objects:
nothing teaches matching at an IoU threshold, object F1, or AP, which are the metrics of
every instance-segmentation and detection paper. Also missing or only optional:
**data leakage** in train/test splits (the RF plan has one good line about correlated
pixels and OOB, but splitting by image/field/patient is never stated as a rule);
**inter-annotator variability** as the ceiling of any metric; and the step from
measurements to a scientific claim. That step covers the unit of replication
(cells are not independent replicates: pseudoreplication), effect sizes, multiple
comparisons over hundreds of features, and measurement bias from resolution and
partial volume. The measurements day's `features-use/` (normalisation, batch effects) is
the right place but is optional.

**Smallest coverage:**
- Add `object-eval/` to the **labelling day** (or extend `seg-eval/`). Matching predicted
  to true objects by IoU, TP/FP/FN objects, F1 vs IoU threshold, AP, split/merge errors;
  annotator disagreement as a band; how to split data for honest evaluation (by image, by
  experiment). Demo `objmatch`: two label images (truth, prediction with merges and splits)
  and an IoU-threshold slider; matched pairs coloured, F1 and AP curves; preset "Dice 0.95
  but count wrong".
- **Make `features-use/` core** on the measurements day and add a "from measurements to
  conclusions" half: biological vs technical replicates, cells-per-replicate vs replicates
  as $n$ (a SuperPlot-style picture), effect size and confidence intervals, multiple testing
  across features; measurement bias (small objects: perimeter and area bias from pixel size
  and PSF, linking to gap 1). Demo `replicates`: simulated cells nested in replicates with
  a replicate-effect slider; the p-value with $n$ = cells vs $n$ = replicates; presets "1000
  cells, no real effect, p < 0.001".

### 5. Registration beyond translation, and optical flow: **should-have (high)** → new day

**Why it matters.** The FFT registration day handles translation, plus rotation/scale as
an optional unit. Real tasks need more: aligning two modalities (EM–light, CT–MRI),
correcting sample deformation, matching to an atlas, and estimating motion fields in live
imaging. The plan flags non-rigid registration and optical flow as unowned. Feature-based
registration (interest points, descriptors, RANSAC fit of an affine transform or
homography) is standard in every vision syllabus. RANSAC already sits on the Hough day
without its main application. The theory fits the audience well: least squares,
Procrustes (snakes day), gradient descent, and the structure tensor.

**Smallest coverage:** one new day, "Aligning images: features, intensities and flow",
after the FFT registration day.

1. `keypoints/`: interest points (recall `edges/`, `scalespace/`), the idea of a local
   descriptor (gradient-orientation histograms; recall HOG from the SVM day), matching
   with the ratio test.
2. `robust-fit/`: estimate a similarity/affine/homography from matches. Least squares,
   then RANSAC (recalled from the Hough day, or moved here as its open question offers);
   Procrustes recalled from the snakes day.
3. `intensity-reg/`: registration as optimisation over transform parameters. SSD, NCC, and
   **mutual information** for multimodal images (joint histograms); the cost landscape,
   local minima, coarse-to-fine pyramids; a non-rigid deformation field with a smoothness
   penalty (B-spline or demons, at concept level).
4. `opticalflow/`: brightness constancy, the aperture problem, Lucas–Kanade as structure-
   tensor least squares, Horn–Schunck as the regularised version (links to Ising/gmrf and
   snakes).

**Demo ideas.**
- `regcost`: a fixed image and a moving image (the same modality, or inverted contrast as a
  second modality) with rotation and shift sliders. Side panels show the cost surface
  (SSD, NCC, MI), the joint histogram sharpening at alignment, and a gradient-descent path
  from a draggable start. Preset "SSD fails across modalities, MI works".
- `aperture`: a moving bar or grating seen through a window, with the Lucas–Kanade
  estimate and the structure-tensor ellipse. Enlarging the window until a corner enters
  resolves the flow.

### 6. Tracking over time: **should-have** → new short day (or join with gap 5)

**Why it matters.** Time-lapse microscopy, cell lineages, particle tracking and video are
everyday data. The plans touch time only through drift correction (FFT registration),
kymographs (DP optional) and background subtraction (GMM optional). Tracking also has
clean theory for this audience: linear assignment, Kalman filtering (linear algebra and
Gaussians), and diffusion statistics.

**Smallest coverage:** three units.
1. `linking/`: tracking by detection. Frame-to-frame linking, greedy vs global
   assignment (the Hungarian algorithm / linear assignment as a min-cost problem,
   which links to the max-flow day), gating, gap closing, divisions and merges.
2. `kalman/`: predict–update with a motion model, the Kalman filter as a weighted average of
   Gaussians, then its use in linking.
3. `trajectories/`: mean squared displacement (diffusive, confined, directed), track-level
   evaluation (switches, fragmentation), lineage trees as a pointer.

**Demo ideas.** `linker`: particles with a density/speed slider. Greedy and optimal
assignment are shown side by side, and identity switches are counted. `kalman2d`: a noisy
2D track, the estimate and the covariance ellipse, with process and measurement noise
sliders.

If only two days are possible for gaps 5 and 6 together, make one "Motion" day:
`intensity-reg/`, `opticalflow/`, `linking/`, `kalman/`. Keypoints and RANSAC
then go to the Hough day's `ransac/` unit as an application slide.

### 7. Images as evidence: integrity, reproducibility, ethics and bias: **should-have** → one unit

**Why it matters.** Students will make figures and measurements that others rely on.
The plans have pieces: Cromey's guidelines are cited in the compression references,
window/level appears in `highbit/`, shortcut learning is in optional `deep-practice/`,
a bias slide is proposed for faces on the integral day, and OME/Zarr are in optional
`compress-science/`. None is core. Missing as a whole: which adjustments are allowed on
scientific images (linear vs non-linear, whole image vs region, disclosure); duplication
and manipulation, and how they are detected; provenance and metadata (pixel size, channel,
acquisition settings) as part of the data; FAIR data; consent and privacy for medical
images and faces; dataset bias and who is under-represented; generated or restored images
that look like data (ties to gap 3 `diffusion/`).

**Smallest coverage:** one unit, `integrity/`, as the last unit of the bit-depth day
(it already owns display, windows and LUTs) or as a closing course unit. Demo `displayvsdata`:
change display (window, gamma, LUT) vs change data, with the histogram showing clipped
values and a measurement that changes only in the second case. Optional second mode:
duplicated-region detection with the phase correlation of the FFT registration day.

### 8. 3D image analysis basics: **should-have** → one unit

**Why it matters.** Confocal, light-sheet, CT and EM data are 3D (and often 3D+t). The
plans mention 3D in many places: 3D connectivity (labelling), anisotropic voxels
(marching cubes), MIP and volume rendering (graphics3d), 3D thinning (skeleton), 3D
moments (optional). But four plans ask for a shared "3D images" unit that does not
exist. Common practitioner errors: measuring on a MIP, isotropic filters on anisotropic
voxels, 2D segmentation slice by slice.

**Smallest coverage:** one unit, `volumes/`, owned early (after `images/` or at the start of
the marching-cubes day). Stacks and voxel spacing; anisotropic Gaussian filtering and
resampling to isotropic; 6/18/26-connectivity; what a MIP hides (overlaps, wrong areas
and counts); 2D-per-slice vs 3D segmentation; volume and surface measures (pointer to the
marching-cubes day). Demo: reuse `lib/volume.ts`/`gl3d.ts`. Two touching nuclei at different
depths merge in the MIP but not in 3D, and a z-spacing slider shows anisotropy.

### 9. Localisation and super-resolution microscopy: **nice-to-have** (should-have for a bio cohort) → optional unit in the gap 1 day

**Why it matters.** It is a showcase for "beat the diffraction limit with statistics".
Fitting a PSF model to photons gives a precision of about $\sigma/\sqrt N$, a clean
estimation-theory result for this audience. SIM is a direct application of the Fourier
unit (moiré moves high frequencies into the OTF). The FFT registration plan already
mentions SMLM drift.

**Smallest coverage:** optional `localisation/` on the image-formation day: centroid vs
Gaussian least-squares vs ML fit, precision vs photon count, rendering a localisation
image, and SIM as a Fourier picture (one slide). Demo: photon-count slider, repeated fits
with a scatter of estimates, and the precision curve against the $\sigma/\sqrt N$ bound.

### 10. Point operations, histogram equalisation, contrast and illumination: **nice-to-have** → slides

Histogram equalisation/CLAHE and gamma/contrast stretching are textbook staples. They
appear only on the colour and integral days. Flat-field correction appears only as a
colour-day slide. Add an equalisation slide plus the "equalisation is not for
measurement" warning to `highbit/` or the thresholding day's `histogram/`. Flat-field
correction belongs to `noise-models/` (gap 1).

### 11. Interpolation, resampling and pyramids: **nice-to-have** → slides

`images/` has bilinear interpolation and `fourier/` has aliasing. Nothing connects them:
interpolation kernels as low-pass filters (nearest, linear, cubic, sinc), and
anti-aliasing before downsampling. Two slides in `scalespace/` (gap 2), which also
introduces the Gaussian pyramid used by snakes and registration.

### 12. Camera geometry and multi-view 3D: **nice-to-have** → pointer

Pinhole camera, calibration, epipolar geometry and stereo are core in computer vision
courses but secondary for this course's leaning. The graphics3d day already derives the
projection matrix. Add one "inverse problem: from pixels back to rays" slide there.
Scanline stereo is already in the optional DP unit.

### 13. Contour descriptors: **nice-to-have** → slide

Fourier descriptors (flagged by the moments plan) are a cheap, demo-friendly link to
`fourier/`. Add one slide plus a demo mode to the measurements day's `shape-features/`
(keep $k$ coefficients of the boundary, reconstruct).

### 14. Clustering beyond k-means / superpixels: **nice-to-have** → slide

Mean shift and SLIC superpixels as a slide on the GMM day (`gmm-images/`). Superpixels
also shrink graph-cut and ncut graphs; add a pointer there.

### Coordination fixes (no new content)

- **One owner per shared concept**, recorded in the course map: PCA (`autoencoder/` is
  built; move it or a PCA unit earlier if snakes/compression/GMM come before Part 3);
  LoG/Hessian/structure tensor/Canny → the new Part 1 units (gap 2); RANSAC → Hough or
  registration; object metrics → labelling (gap 4); image formation/noise → gap 1 day.
- Remove the stale open questions (colour: JPEG DCT, hyperspectral; bit depth:
  compressed formats).

## Over-weighted topics

These have more depth than their importance warrants for a theory-first image *analysis*
course. Each suggestion frees time for the gaps above.

- **Computer graphics (`plan/graphics3d-day.md`).** Three of four units are polygon
  rendering (projection, rasterisation, z-buffer, shading, textures, mipmaps). It is a
  good linear-algebra payoff, but it is graphics, not analysis. Proposal: merge
  units 1–3 into one overview unit (meshes + projection matrix; rasterisation and shading
  as pictures) and keep `volrender/`, the part analysts use. Or make it an elective day.
- **GPU texture formats (`gpuformats/`, built).** Packed formats, premultiplied alpha
  and mipmap memory matter only to those who write GPU code (the plan says so). Keep
  stride and endianness as one slide in `highbit/`, and make the rest optional.
- **Integral images as a whole day.** Summed-area tables are one elegant idea (one unit:
  table + box filters + local statistics). Rectangle features and the Viola–Jones
  cascade are historical face detection with consent and bias issues. Proposal: one
  unit on the kernels/thresholding side, Viola–Jones as an optional unit.
- **Moments as a whole day after labelling and measurements.** Second-order moments are
  taught on the labelling day, and Hu invariants are named there. Legendre/Zernike
  orthogonal moments and reconstruction are a niche in modern practice. Proposal: one
  optional unit (`zernike/` with `orthogonal-moments/` folded in).
- **Classical contour models.** Snakes with active shape models, level sets, DP (with
  seam carving), graph cuts (4 units plus ncut), watershed and Ising: six-plus days of
  energy-based segmentation against one day of deep learning. They are worth keeping
  (they teach energies, which also underpin DL losses), but as a core of about three days
  (graph cuts, watershed, one of level sets/snakes) with the rest as electives. In
  particular: **seam carving** is image editing, not analysis (the DP plan asks the
  same; prefer DTW/kymographs); the **ASM half** of the snakes day is largely superseded
  (keep the PCA shape model as one slide); the Ising day's **MCMC/phase transition** is
  physics more than image analysis (make `mcmc/` optional, keep `mrf-denoise/` and
  `gmrf/`).
- **Classical classifiers: three days (SVM, random forest, GMM) next to one NN day.** The
  SVM dual/kernel trick and SMO are heavy for a course whose students will mostly use
  forests (pixel classification) and networks. Proposal: shrink the SVM day to two units
  (max margin + kernels at concept level, dual in the math layer only), and make the RF
  day's pixel classification the core "trainable segmentation" unit.
- **Compression internals.** Entropy coding, LZ77/LZW and arithmetic coding get more room
  than the part that matters to analysts: what lossy compression does to measurements and
  which format to store. Agree with the plan's own open question: make `compress-science/`
  core and cut unit 2 to a slide.
- **Floating point summation (`floatingpoint/`, built).** Kahan/pairwise summation is a
  fine demo, but for this audience the take-home is three rules of thumb. It could share
  a unit with `highbit/` if time is short. This one is mild.
- **Radon day: six units.** Justified for a medical-imaging cohort, heavy for others.
  Keep units 1–3 core and make algebraic reconstruction core only if the image-formation
  day (gap 1) does not already carry the inverse-problem idea. Otherwise make it an
  elective track.

## Suggested priority if only a few days can be added

1. Two Part 1 units: `scalespace/`, `edges/` (gap 2): cheap, and they unblock six plans.
2. Image formation and restoration day (gap 1).
3. Neural-network day core and before the VAEs, plus `object-eval/` and a core
   `features-use/` (gaps 3 and 4, mostly existing plans).
4. Deep learning in practice day (gap 3).
5. Motion day: registration + tracking (gaps 5 and 6).
6. `integrity/` and `volumes/` units (gaps 7 and 8).

Pay for these by making the over-weighted days above electives.
