# Course map (decisions, 2026-11-12)

> **Decided 2026-11-13: this is a self-study course with no cap on the number of days. There
> are no electives and no core/elective distinction.** Every day belongs to one sequence, sorted
> into themed parts by prerequisites (see "Self-study restructure" at the end). The tables below
> still use the older core/elective wording until that restructure is applied.

Single source of truth for the structure of the course. Plans and units follow this
file; where a plan disagrees, this file wins and the plan should be updated. Based on
the three reviews (`review-ordering.md`, `review-gaps.md`, `review-organisation.md`).

## Core track

Order within a part is the teaching order. Days marked *(new)* have no plan yet.

| Part | Theme | Days |
|---|---|---|
| I | Images as numbers | 1. Images, filters, frequencies (`images/`, `kernels/`, `fourier/`, `pca/`) · 2. Edges, blobs and ridges · 3. Bit depth and noise (`quantisation/`, `cameranoise/`, `highbit/`, `contrast/`, `floatingpoint/`) · 4. Image formation · 5. Deconvolution |
| II | From pixels to objects | 6. Thresholding · 7. Colour · 8. Morphology · 9. Labelling · 10. Watershed · 11. Measurements |
| III | Aligning images | 12. FFT registration |
| IV | Segmentation as optimisation | 13. Graph cuts (built: `seg-energy/`, `maxflow/`, `graphcut-seg/`, `ncut/`) |
| V | Learning from examples | 14. Random forests (owns classification basics) · 15. Neural networks (built `neuralnet/` is unit 1) · 16. Evaluation and statistics · 17. VAEs (built: `autoencoder/`, `vae/`, `vae-gen/`) · 18. Modern deep learning (its diffusion unit builds on the VAE) |

18 core days (renumbered 2026-11-12 after adding deconvolution and putting VAEs before
modern deep learning). If the course must be shorter, the first candidates to move to
electives are: VAEs (17), FFT registration (12), colour (7), and image formation +
deconvolution (4–5, merge into one day).

## Electives

Each group can be taken after the core day named.

| Group | Days | Entry after |
|---|---|---|
| Objects | Skeletons, moments | Labelling |
| Curves and energies | Snakes/ASM → level sets → dynamic programming; Ising/MRF | Graph cuts |
| Curves and energies *(splines: built 2026-11-13, `plan/splines-day.md`; first in the group's sidebar)* | **Splines and NURBS**: polynomial interpolation and its wiggles, piecewise cubics and continuity (C⁰/C¹/C²), Bézier curves and de Casteljau, B-splines as a basis (knots, local control, partition of unity, Cox–de Boor), interpolating vs approximating vs smoothing splines (a regularised fit: data + λ·curvature), **NURBS** (rational weights, exact circles/conics, homogeneous coordinates from the images unit), spline surfaces briefly; image uses: B-spline/cubic image interpolation (beyond the bilinear of the images unit), spline-based active contours, free-form-deformation (B-spline) non-rigid registration, smooth outlines and length/curvature from fitted splines | Part I (images unit: interpolation, homogeneous coordinates) |
| Detection | Hough, integral images | Edges, blobs and ridges |
| More learning | SVM, Gaussian mixtures | Random forests |
| Generative models *(built 2026-11-13, `plan/diffusion-day.md`)* | **Diffusion models** as their own day: forward noising and noise schedules, denoising as the core operation, score functions and denoising score matching (intuition first), sampling (ancestral/DDPM-style and deterministic/DDIM-style, number of steps), conditioning and classifier-free guidance, latent diffusion (links to the VAE day), diffusion priors for inverse problems (links to deconvolution: posterior sampling vs the regularised estimate), evaluation of generated images, scientific uses (denoising, super-resolution, virtual staining) and the risk of hallucinated detail | Modern deep learning (day 18) |
| Transforms | Wavelets → compression; Radon | Part I |
| Spectral | Spectral unmixing | Colour |
| 3D | Marching cubes → 3D graphics (gets `gpuformats/`) | Part I |
| Computing *(built 2026-11-13, `plan/gpus-day.md`)* | **How GPUs work, and what they are good and bad at**: many simple cores vs few complex ones (throughput vs latency), SIMT/warps and branch divergence, the memory hierarchy (registers, shared memory, global memory, caches) and why memory bandwidth usually limits image code, data parallelism (map, stencil/convolution, reduction, prefix scan, histogram with atomics), transfers between CPU and GPU as a bottleneck, occupancy and arithmetic intensity (a roofline picture), floating-point precision on GPUs (float32/float16/bfloat16, tensor cores; recall the bit depth day), which image methods map well (filters, FFT, element-wise ops, neural networks, ray marching) and which do not (sequential/data-dependent algorithms: flooding/watershed priority queues, union–find labelling, max-flow, dynamic programming with dependencies, small problems dominated by launch/transfer overhead). Demos simulated in the browser (warps and divergence, memory coalescing, a roofline calculator, CPU-vs-GPU timing of the same filter via WebGL2 where available). Links: the 3D graphics day (the graphics pipeline that GPUs come from; gpuformats), integral images (parallel scan), neural networks | Bit depth and noise; 3D graphics helps |
| 3D *(ray tracing: built 2026-11-13, `plan/raytracing-day.md`)* | **Ray tracing**: rays from the camera through pixels (camera model recalled from the 3D graphics day), ray–object intersection (sphere, plane, triangle; barycentric coordinates), shading, shadows, mirror reflection and refraction (Snell, Fresnel) — Whitted-style recursion; acceleration structures (bounding-volume hierarchies, grids) and why naive tracing is O(pixels × objects); **path tracing**: the rendering equation as an integral, Monte Carlo estimation, noise that falls as 1/√samples, importance sampling, and denoising (links to the deconvolution day's denoisers); rasterisation vs ray tracing (what each is good at); ray optics in microscopes and lenses (link to image formation); hardware ray tracing on GPUs (link to the GPU day). Demos: a step-through tracer on a coarse pixel grid, a 2D ray/reflection/refraction sandbox, a progressive path tracer in the browser (WebGL2 or Wasm if measured slow) with a samples-per-pixel slider | 3D graphics |
| 3D *(geometry from multiple images: built 2026-11-13, `plan/multiview-day.md`)* | **Reconstructing geometry from multiple images**: the pinhole camera (intrinsics, extrinsics, calibration), two-view geometry (epipolar lines, the fundamental/essential matrix), stereo (rectification, disparity → depth, matching costs), triangulation, structure from motion and bundle adjustment (least squares over cameras and points), multi-view stereo, depth from focus/defocus briefly, and modern view synthesis: neural radiance fields and **3D Gaussian splatting** (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Links: camera matrices from the 3D graphics day, feature matching + RANSAC from the Hough day, scanline stereo from the dynamic programming day (becomes an application pointer), volume rendering/compositing from the 3D graphics day (same compositing as splatting) | 3D graphics |
| Motion *(built 2026-11-13: optical flow `plan/opticalflow-day.md`, tracking `plan/tracking-day.md`)* | **Optical flow** as its own day (requested 2026-11-12): brightness constancy and the aperture problem, Lucas–Kanade (local least squares, the structure tensor link to the edges day), Horn–Schunck (global smoothness, a regularised energy), coarse-to-fine pyramids for large motion, flow visualisation and evaluation, applications (cell/tissue flow, kymographs, PIV-style analysis); non-rigid registration as a related unit or day; **Tracking** as its own day (requested 2026-11-12): linking detections over time (nearest neighbour, global assignment / Hungarian, gating), motion models and prediction (Kalman filter), gaps and missed detections, **dividing objects** (splits/merges, lineage trees), appearing/disappearing objects, track evaluation, and track analysis (MSD, velocities). Inspiration for methods: the user's tool blockflow (/home/mahogny/github/claude/blockflow) has several tracking methods — read it when planning, but no tool names in student material | FFT registration; tracking after labelling (needs objects) and the point-detection day |
| Detection *(point detection: built 2026-11-13, `plan/pointdetection-day.md`)* | Spot/point detection: local maxima, LoG/DoG and à trous detectors, thresholds on detection statistics, sub-pixel localisation by fitting, evaluation by matching | Edges, blobs and ridges |
| Fluorescence quantification *(built 2026-11-13: `plan/colocalisation-day.md`, `plan/fret-day.md`, `plan/flim-day.md`)* | **Colocalisation** (Pearson, Manders with thresholds, Costes, object-based colocalisation, bleed-through and resolution as confounds); **FRET** (sensitised emission with bleed-through/cross-excitation correction, acceptor photobleaching, FRET efficiency and distance, ratiometric imaging); **FLIM** (lifetimes, time-domain/TCSPC and frequency-domain, exponential fitting, phasor analysis, FLIM-FRET) | Colour; spectral unmixing helps for FRET |
| Objects *(spatial statistics: built 2026-11-13, `plan/spatialstats-day.md`; after Moments in the sidebar)* | **Spatial statistics of objects**: point patterns and complete spatial randomness (quadrats, intensity), nearest-neighbour and empty-space functions (G, F, J) with edge correction, Ripley's K/L and the pair correlation, simulation envelopes and global tests, two cell types (cross-K, mark connection) and inhomogeneous density (`point-patterns/`, `nn-functions/`, `ripley-k/`, `envelopes/`, `marked-patterns/`) | Measurements (day 11); point detection and labelling help |
| Curves and energies *(graph-based segmentation: built 2026-11-13, `plan/graphseg-day.md`; after Ising/MRF in the sidebar)* | **Graph-based segmentation beyond graph cuts**: the random walker, spanning trees (single linkage, Felzenszwalb–Huttenlocher), watershed as a minimum spanning forest, hierarchical segmentation and merge trees, superpixels (SLIC) (`random-walker/`, `mst-seg/`, `ws-msf/`, `merge-trees/`, `superpixels/`) | Graph cuts (day 13); watershed (day 10) |
| Detection *(keypoints: built 2026-11-13, `plan/keypoints-day.md`; after Point detection in the sidebar)* | **Keypoints and descriptors**: corners and blobs across scale (DoG pyramid, repeatability), gradient-histogram and binary descriptors, matching (ratio test, mutual check, k-d trees), fitting transforms to matches with RANSAC, stitching and registration (`keypoints/`, `descriptors/`, `feature-matching/`, `feature-alignment/`) | Edges, blobs and ridges; Hough (RANSAC) and FFT registration help |
| Transforms *(edge-preserving smoothing: built 2026-11-13, `plan/pdesmooth-day.md`; after Radon in the sidebar)* | **Edge-preserving smoothing and PDEs**: the heat equation and explicit schemes, Perona–Malik, total variation flow, bilateral filter and non-local means in depth, diffusion and exemplar inpainting, all smoothers compared (`heat-equation/`, `perona-malik/`, `tv-flow/`, `bilateral-nlm/`, `inpainting/`, `smoothers/`) | Deconvolution (day 5); the plan proposes a new group "Restoration" (not applied) |
| Transforms *(sparse representations: built 2026-11-13, `plan/sparse-day.md`; last in the group)* | **Sparse representations**: sparsity and the ℓ1 penalty, ISTA/FISTA and pursuit, compressed sensing, fast MRI with compressed sensing, sparse coding and dictionary learning (`sparsity/`, `ista/`, `compressed-sensing/`, `cs-mri/`, `dictionaries/`) | Wavelets; deconvolution (`regularise/`) |

## Owners of shared concepts

A concept is **taught** only by its owner; every other day **recalls** it in at most one
slide with a pointer.

| Concept | Owner |
|---|---|
| PCA | `pca/` unit in day 1 (2D picture, covariance, eigenvectors); `autoencoder/` keeps "PCA = linear autoencoder" |
| Gaussian derivatives, LoG/DoG, Hessian, structure tensor, Canny, scale space | Edges, blobs and ridges (takes Canny from the Hough plan and the Hessian/vesselness basics from the skeleton plan) |
| Noise models, SNR, PSNR | `cameranoise/` (day 3) |
| PSF, OTF, sampling/Nyquist in optics, the forward model | Image formation |
| Deconvolution (inverse filtering, Wiener, Tikhonov, Richardson–Lucy, blind), regularisation of inverse problems, median as denoiser, restoration artefacts | Deconvolution (own day, decided 2026-11-12; directly after image formation) |
| Hysteresis thresholding | Introduced on the edges day (inside Canny, day 2); generalised as a segmentation tool on the thresholding day (day 6) |
| Histograms, thresholds, Dice/ROC for pixels | Thresholding |
| Object-level evaluation (matching, F1/AP), leakage, statistics of measurements | Evaluation and statistics |
| Distance transform | Morphology |
| Connected components | Labelling |
| Train/test, overfitting, cross-validation, the feature stack | Random forests |
| Boosting | Integral images |
| Energy minimisation (data + λ·smoothness) | Graph cuts (`seg-energy/`) |
| Integral image | Integral images |
| NCC, template matching | FFT registration |
| Radon transform | Radon |
| Feature matching (descriptors, ratio test, cross-check, RANSAC on matches) | Keypoints and descriptors (`two-view/` recalls it) |
| Point-pattern statistics (CSR, G/F/J, K/L/g, edge corrections, envelopes, cross/mark functions) | Spatial statistics of objects (`relations/` keeps the first Poisson nearest-neighbour law; `coloc-objects/` keeps cross-K as an application) |
| Random walker, minimum spanning trees / FH segmentation, hierarchical segmentation, superpixels, watershed as MSF | Graph-based segmentation (`ws-msf/`; union–find stays with labelling; k-means with `kmeans/`) |
| Sparsity priors in general (ℓ0/ℓ1, lasso, ISTA, compressed sensing, dictionaries) | Sparse representations (wavelet shrinkage stays on the wavelets day as the first special case) |
| Nonlinear diffusion, TV flow, explicit-scheme stability for diffusion, bilateral/NLM in depth, classical inpainting | Edge-preserving smoothing and PDEs (`gmrf/` owns TV as a prior; `denoise/` first introduces the denoisers) |
| Point operations, gamma/log, histogram equalisation, CLAHE, histogram matching; the cumulative histogram | `contrast/` (day 3); `histogram/` recalls the cumulative histogram |

## Conventions

- λ is the weight of the prior/smoothness term everywhere; β only for snake rigidity
  and the β-VAE.
- "Kernel" alone means a convolution kernel; the SVM one is "kernel function".
- Always "bit depth" in full.
- Math layer markup: see README.md, "Marking the math layer".

## Decided after the deconvolution plan (2026-11-12)

- Deconvolution owns ill-posed inverse problems and regularisation with continuous
  unknowns (data + λ·penalty, choosing λ, early stopping). `seg-energy/` keeps energy
  minimisation over labels and recalls deconvolution; TV stays with Ising, sparsity with
  wavelets.
- Median filter: introduced as a denoiser on the deconvolution day (one slide plus the
  `denoisers` demo); morphology owns rank filters as a family; thresholding recalls it in
  one line.

## Open decisions

- None at the moment. (VAEs before modern deep learning: decided yes, 2026-11-12.)

## Ownership changes to apply when these days are planned

Applied ones are marked **done (2026-11-13)**.


- **Done (2026-11-13).** Colocalisation: the colocalisation day takes over the `coloc/` unit of the measurements
  day (measurements keeps a one-slide recall, in `relations/lec.qmd`). Day 11 now has three units.
- **Done (2026-11-13).** Phasor analysis: the FLIM day owns phasors (the `phasor/` folder is now the FLIM unit); the unmixing day's optional
  phasor unit becomes a recall (not built; `plan/unmixing-day.md` still names the folder `phasor/`).
- **Done (2026-11-13).** FRET bleed-through correction builds on the unmixing day's crosstalk unit (recall in `sensitised-emission/`).
- Splines: the splines day owns spline theory; snakes (spline snakes), non-rigid registration
  (B-spline free-form deformation), image interpolation and outline fitting recall it.
- **Done (2026-11-13).** Diffusion: the diffusion day owns diffusion models in depth and builds on the modern deep learning day's
  `diffusion/` unit, which stays there as the introduction and points to the four diffusion units.
- **Done (2026-11-13), pointer pending.** Stereo: the geometry-from-images day owns stereo (`stereo/`); the dynamic programming day's optional
  scanline stereo becomes an application of its DP with a pointer (the pointer in the DP units is not yet added).

## Topic pages (planned 2026-11-12, build after the current day build)

Some topics are approached from many angles across days. Each gets a **topic page**
(a "metapage") that explains the question in plain language, compares the approaches
side by side (what each assumes, when it fails), and links to the relevant units and
demos. Topic pages are part of the site navigation (a "Topics" section), not days.

Candidates (first one requested: point detection):
- Point / spot detection: edges (LoG/DoG blobs), wavelets (à trous), image formation
  (localisation microscopy), FFT registration (sub-pixel peaks), Hough (accumulator
  peaks), evaluation (matching detections), plus the planned point-detection day.
- Denoising: kernels, deconvolution (Wiener, median, NL-means), wavelets (shrinkage),
  Ising (MRF/TV), modern deep learning (learned restoration, hallucination).
- Segmentation: thresholding, watershed, graph cuts, level sets, snakes, random forests,
  neural networks.
- Registration and alignment: images (affine), FFT registration, Procrustes (snakes),
  motion (planned).
- Eigenvectors in image analysis: PCA, structure tensor/Hessian, normalized cuts,
  shape models, moments.
- Energy minimisation: graph cuts, Ising, snakes, level sets, dynamic programming,
  deconvolution (regularisation).

## Applied after the second batch (2026-11-13)

The planners' "Proposed reorganisation" sections were accepted (owners rows above). The
spatial-statistics day's entry point is after Measurements (day 11). The watershed day's optional
`ws-graphs/` unit is superseded by `ws-msf/` on the graph-based segmentation day. Edits to
**built units** implied by these decisions are listed in `plan/crosslink-todo.md` for the
cross-link pass.

## Prerequisite map (planned 2026-11-12, build after the current day build)

For learners who want one topic and only what it needs.

- **One database** of the course structure, the single source for navigation: e.g.
  `days.yml` at the repo root (machine-readable; this coursemap stays the human
  rationale). Per day: id, title, part or elective group, number, units (folder, title),
  **hard prerequisites** (days or units that must come first), **soft prerequisites**
  (nice-to-have back-references), concepts it owns (from the owners table). Units may
  list their own prerequisites at unit level (finer than day level). Seed it from
  `review-ordering.md`'s dependency table and the plans' "prerequisite gap" sections.
- **Overview page** (`overview.qmd`, linked from the home page and navbar): all days as
  an interactive dependency graph (a TypeScript demo reading the database: parts as
  columns, electives grouped, hard edges solid, soft edges dashed). Click a day or
  unit → highlight its transitive hard prerequisites and show an ordered **learning
  path** (topological order of just those prerequisites). Also a plain list view for
  print/accessibility.
- **Per-page prerequisite box**: every lec.qmd/exercise.qmd shows "Before this unit:"
  with links to its direct hard prerequisites (and "see also" for soft ones), generated
  from the database at render time (a Quarto pre-render script or Lua filter, so no unit
  hand-maintains the list). Reverse links ("Used later in: …") can come from the same data.
- A check in `make test` (or a separate `make check-prereqs`): every unit in the sidebar is
  in the database, every prerequisite exists, the graph has no cycles, and the sidebar
  order respects hard prerequisites.
- The sidebar in `_quarto.yml` could later be generated from the same database.

## Queue, second batch (requested 2026-11-13; all built 2026-11-13)

All six items are planned and built (marked below). To be planned and built after the current batch. Days and content may be reorganised when
planning (the user allowed reorganising days and content).

- **Built 2026-11-13** (`plan/keypoints-day.md`; elective Detection, after Point detection). **Keypoints and descriptors** (day): interest-point detectors (Harris/Förstner recall from the
  edges day, DoG/scale-space extrema), orientation and scale invariance, SIFT-style gradient
  histograms and binary descriptors, matching (ratio test, mutual nearest neighbours), robust
  fitting with RANSAC (recall the Hough day), uses: stitching, registration, geometry from images.
  Becomes the owner of feature matching that several days currently assume.
- **Built 2026-11-13** (`plan/spatialstats-day.md`; elective Objects, after Moments, entry after Measurements). **Spatial statistics of objects** (day): point patterns (complete spatial randomness as the
  null model), nearest-neighbour distance distributions, Ripley's K/L and pair correlation,
  clustering vs regularity, edge corrections, marked patterns (types of cells), simulation
  envelopes; inputs from point detection and labelling; link to the evaluation day's statistics.
- **Built 2026-11-13** (`plan/contrast-unit.md`; core day 3, unit `contrast/` between `highbit/` and `floatingpoint/`). **Contrast and intensity transforms** (unit, place when planning: Part I or the thresholding
  day): point operations, gamma/log, histogram equalisation (the cumulative histogram as the
  mapping), CLAHE, what these do to measurements (display vs analysis).
- **Built 2026-11-13** (`plan/graphseg-day.md`; elective Curves and energies, after Ising/MRF). **Graph-based segmentation beyond graph cuts** (day or units): the random walker (a linear
  system on the graph Laplacian; recall ncut), minimum-spanning-tree segmentation
  (Felzenszwalb–Huttenlocher), superpixels (SLIC; recall k-means), watershed as a minimum
  spanning forest (recall the watershed day), hierarchical segmentation.
- **Built 2026-11-13** (`plan/sparse-day.md`; elective Transforms, last, after Edge-preserving smoothing). **Sparse representations** (day): sparsity in a basis (recall wavelets, DCT), L1 vs L2
  (geometry of the lasso), compressed sensing (incoherent sampling, recovery), sparse coding
  and dictionary learning, uses in denoising/inpainting/deconvolution and fast MRI.
- **Built 2026-11-13** (`plan/pdesmooth-day.md`; elective Transforms, after Radon; the proposed new group "Restoration" is not applied). **Edge-preserving smoothing and PDEs** (day): heat equation = Gaussian blur (scale space
  recall), anisotropic diffusion (Perona–Malik), total variation flow (recall Ising/TV),
  bilateral filter and non-local means in depth (recall deconvolution denoisers), inpainting
  (diffusion-based and exemplar-based), stability of explicit schemes (CFL recall from level sets).


## Self-study restructure (decided 2026-11-13, to do)

The course is self-study and has no cap on days, so the core/elective split goes away. All days
form one sequence, grouped into themed parts and ordered so that hard prerequisites come first.
Learners pick a topic and follow its prerequisites (the prerequisite map makes that easy).

To do:
1. **New part structure** (proposal, to be refined when applied): fold every former elective into
   the part where its topic belongs, after its prerequisites, e.g.
   - Part I Images as numbers: + contrast, splines, GPUs, compression, wavelets, sparse
     representations, edge-preserving smoothing/PDEs, Radon (or a separate "Transforms and
     reconstruction" part);
   - Part II From pixels to objects: + skeletons, moments, spatial statistics, point detection,
     keypoints, Hough, integral images;
   - Part III Aligning images and motion: FFT registration, optical flow, tracking, geometry from
     multiple images;
   - Part IV Segmentation as optimisation: graph cuts, graph-based segmentation, snakes, level
     sets, dynamic programming, Ising/MRF;
   - Part V Learning from examples: random forests, SVM, Gaussian mixtures, neural networks,
     evaluation, VAEs, modern deep learning, networks for cells and objects, diffusion;
   - Part VI Fluorescence and spectra: colour (or keep in II), unmixing, colocalisation, FRET, FLIM;
   - Part VII 3D: marching cubes, 3D graphics, ray tracing.
   Day numbers become global (1…N) in that order.
2. Derive the order from the prerequisite database (`days.yml`, see "Prerequisite map") so that
   it is checked, not hand-maintained; the sidebar is generated from it.
3. Remove "elective"/"core" wording from units, plans, the home page and the course map (about
   127 .qmd files and most plans mention it); replace "the X elective" with "the X day".
4. Update day numbers in plans and units ("day N" references) after renumbering.
