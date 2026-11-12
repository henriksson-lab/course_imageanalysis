# Course map (decisions, 2026-11-12)

Single source of truth for the structure of the course. Plans and units follow this
file; where a plan disagrees, this file wins and the plan should be updated. Based on
the three reviews (`review-ordering.md`, `review-gaps.md`, `review-organisation.md`).

## Core track

Order within a part is the teaching order. Days marked *(new)* have no plan yet.

| Part | Theme | Days |
|---|---|---|
| I | Images as numbers | 1. Images, filters, frequencies (`images/`, `kernels/`, `fourier/`, `pca/`) · 2. Edges, blobs and ridges · 3. Bit depth and noise (`quantisation/`, `cameranoise/`, `highbit/`, `floatingpoint/`) · 4. Image formation · 5. Deconvolution |
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
| Curves and energies: splines *(requested 2026-11-12, not yet planned)* | **Splines and NURBS**: polynomial interpolation and its wiggles, piecewise cubics and continuity (C⁰/C¹/C²), Bézier curves and de Casteljau, B-splines as a basis (knots, local control, partition of unity, Cox–de Boor), interpolating vs approximating vs smoothing splines (a regularised fit: data + λ·curvature), **NURBS** (rational weights, exact circles/conics, homogeneous coordinates from the images unit), spline surfaces briefly; image uses: B-spline/cubic image interpolation (beyond the bilinear of the images unit), spline-based active contours, free-form-deformation (B-spline) non-rigid registration, smooth outlines and length/curvature from fitted splines | Part I (images unit: interpolation, homogeneous coordinates) |
| Detection | Hough, integral images | Edges, blobs and ridges |
| More learning | SVM, Gaussian mixtures | Random forests |
| Generative models *(requested 2026-11-12, not yet planned)* | **Diffusion models** as their own day: forward noising and noise schedules, denoising as the core operation, score functions and denoising score matching (intuition first), sampling (ancestral/DDPM-style and deterministic/DDIM-style, number of steps), conditioning and classifier-free guidance, latent diffusion (links to the VAE day), diffusion priors for inverse problems (links to deconvolution: posterior sampling vs the regularised estimate), evaluation of generated images, scientific uses (denoising, super-resolution, virtual staining) and the risk of hallucinated detail | Modern deep learning (day 18) |
| Transforms | Wavelets → compression; Radon | Part I |
| Spectral | Spectral unmixing | Colour |
| 3D | Marching cubes → 3D graphics (gets `gpuformats/`) | Part I |
| 3D: geometry from images *(requested 2026-11-12, not yet planned)* | **Reconstructing geometry from multiple images**: the pinhole camera (intrinsics, extrinsics, calibration), two-view geometry (epipolar lines, the fundamental/essential matrix), stereo (rectification, disparity → depth, matching costs), triangulation, structure from motion and bundle adjustment (least squares over cameras and points), multi-view stereo, depth from focus/defocus briefly, and modern view synthesis: neural radiance fields and **3D Gaussian splatting** (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Links: camera matrices from the 3D graphics day, feature matching + RANSAC from the Hough day, scanline stereo from the dynamic programming day (becomes an application pointer), volume rendering/compositing from the 3D graphics day (same compositing as splatting) | 3D graphics |
| Motion *(should-have, not yet planned)* | **Optical flow** as its own day (requested 2026-11-12): brightness constancy and the aperture problem, Lucas–Kanade (local least squares, the structure tensor link to the edges day), Horn–Schunck (global smoothness, a regularised energy), coarse-to-fine pyramids for large motion, flow visualisation and evaluation, applications (cell/tissue flow, kymographs, PIV-style analysis); non-rigid registration as a related unit or day; **Tracking** as its own day (requested 2026-11-12): linking detections over time (nearest neighbour, global assignment / Hungarian, gating), motion models and prediction (Kalman filter), gaps and missed detections, **dividing objects** (splits/merges, lineage trees), appearing/disappearing objects, track evaluation, and track analysis (MSD, velocities). Inspiration for methods: the user's tool blockflow (/home/mahogny/github/claude/blockflow) has several tracking methods — read it when planning, but no tool names in student material | FFT registration; tracking after labelling (needs objects) and the point-detection day |
| Point detection *(requested 2026-11-12, not yet planned — plan after the current build)* | Spot/point detection: local maxima, LoG/DoG and à trous detectors, thresholds on detection statistics, sub-pixel localisation by fitting, evaluation by matching | Edges, blobs and ridges |
| Fluorescence quantification *(requested 2026-11-12, not yet planned — plan after the current build)* | **Colocalisation** (Pearson, Manders with thresholds, Costes, object-based colocalisation, bleed-through and resolution as confounds); **FRET** (sensitised emission with bleed-through/cross-excitation correction, acceptor photobleaching, FRET efficiency and distance, ratiometric imaging); **FLIM** (lifetimes, time-domain/TCSPC and frequency-domain, exponential fitting, phasor analysis, FLIM-FRET) | Colour; spectral unmixing helps for FRET |

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

- Colocalisation: the colocalisation day takes over the `coloc/` unit of the measurements
  day (measurements keeps a one-slide recall).
- Phasor analysis: the FLIM day owns phasors; the unmixing day's optional `phasor/` unit
  becomes a recall.
- FRET bleed-through correction builds on the unmixing day's crosstalk unit (recall).
- Splines: the splines day owns spline theory; snakes (spline snakes), non-rigid registration
  (B-spline free-form deformation), image interpolation and outline fitting recall it.
- Diffusion: the diffusion day owns diffusion models in depth; the modern deep learning day's
  `diffusion/` unit (being built) becomes a short introduction with a pointer, or moves to
  the diffusion day as its first unit — decide when planning.
- Stereo: the geometry-from-images day owns stereo; the dynamic programming day's optional
  scanline stereo becomes an application of its DP with a pointer.

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
