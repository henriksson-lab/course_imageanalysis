# Course map (decisions, 2026-11-12)

> **Decided 2026-11-13: this is a self-study course with no cap on the number of days. There
> are no electives.** Every day belongs to one sequence of 51 days, sorted into nine themed parts
> by prerequisites (applied 2026-11-13, see "Self-study restructure" at the end). The machine-readable
> structure is `days.yml` at the repo root; this file keeps the rationale.

The rationale for the structure of the course (the order itself is `days.yml`). Plans and units follow this
file; where a plan disagrees, this file wins and the plan should be updated. Based on
the three reviews (`review-ordering.md`, `review-gaps.md`, `review-organisation.md`).

## Parts and days

The single source of the order is `days.yml` (parts → days → units, with hard and soft
prerequisites and the concepts each day owns); `make check-prereqs` checks that every hard
prerequisite comes earlier, and `make sidebar` generates the sidebar from it. Order within a part
is the teaching order; day numbers are global (1–51). Each day's plan is named in its `plan:` field.

| Part | Theme | Days |
|---|---|---|
| I | Images as numbers | 1. Images, filters and frequencies (`images/` `kernels/` `fourier/` `pca/`) · 2. Edges, blobs and ridges (`canny/` `scale-space/` `hessian/` `structure-tensor/`) · 3. Bit depth and noise (`quantisation/` `cameranoise/` `highbit/` `contrast/` `floatingpoint/`) · 4. Image formation (`psf/` `resolution/` `illposed/`) · 5. Deconvolution (`deconv/` `regularise/` `richardson-lucy/` `denoise/`) · 6. Splines and NURBS (`bezier/` `bsplines/` `spline-fit/` `spline-images/`) |
| II | Transforms and reconstruction | 7. Wavelets (`haar/` `filterbank/` `dwt2d/` `shrinkage/`) · 8. Image compression (`entropy/` `lossless/` `dct/` `jpeg/` `compress-science/`) · 9. Radon transform and tomography (`radon/` `backprojection/` `fbp/` `algebraic-recon/`) · 10. Edge-preserving smoothing and PDEs (`heat-equation/` `perona-malik/` `tv-flow/` `bilateral-nlm/` `inpainting/` `smoothers/`) · 11. Sparse representations (`sparsity/` `ista/` `compressed-sensing/` `cs-mri/` `dictionaries/`) |
| III | From pixels to objects | 12. Thresholding (`histogram/` `otsu/` `local-threshold/` `seg-eval/`) · 13. Morphology (`morph-binary/` `morph-openclose/` `morph-grey/` `morph-reconstruct/`) · 14. Labelling (`label-connectivity/` `regionprops/` `shape-features/` `label-measure/`) · 15. Watershed (`ws-landscape/` `ws-flooding/` `ws-gradient/` `ws-distance/`) · 16. Measurements (`intensity-features/` `texture-features/` `relations/`) · 17. Skeletons (`skel-thinning/` `skel-medialaxis/` `skel-graph/` `skel-neurovessel/`) · 18. Moments (`moments-basis/` `moment-invariants/` `orthogonal-moments/` `zernike/`) · 19. Spatial statistics of objects (`point-patterns/` `nn-functions/` `ripley-k/` `envelopes/` `marked-patterns/`) |
| IV | Detecting points, lines and features | 20. Point detection (`spot-filters/` `spot-threshold/` `spot-fit/` `spot-crowd/`) · 21. Hough transform (`hough-lines/` `hough-circles/` `ght/` `ransac/`) · 22. Integral images (`integral-image/` `box-filters/` `rect-features/` `viola-jones/`) |
| V | Aligning images and motion | 23. FFT registration (`matching/` `xcorr-fft/` `phasecorr/` `stitching/`) · 24. Keypoints and descriptors (`keypoints/` `descriptors/` `feature-matching/` `feature-alignment/`) · 25. Optical flow (`flow-constraint/` `lucas-kanade/` `horn-schunck/` `nonrigid-reg/`) · 26. Tracking (`track-linking/` `track-predict/` `track-events/` `track-analysis/`) |
| VI | Segmentation as optimisation | 27. Graph cuts (`seg-energy/` `maxflow/` `graphcut-seg/` `ncut/`) · 28. Ising model and Markov random fields (`ising/` `mcmc/` `mrf-denoise/` `gmrf/`) · 29. Graph-based segmentation (`random-walker/` `mst-seg/` `ws-msf/` `merge-trees/` `superpixels/`) · 30. Snakes and active shape models (`snakes/` `snake-matrix/` `shape-models/` `asm/`) · 31. Level sets (`implicit/` `levelset-eq/` `geodesic/` `chanvese/`) · 32. Dynamic programming (`dp-paths/` `polar-boundary/` `livewire/` `fastmarching/`) |
| VII | Learning from examples | 33. Random forests (`dtree/` `forest/` `forest-objects/` `pixelclass/`) · 34. Support vector machines (`classify/` `maxmargin/` `kernel-trick/` `pixel-classify/`) · 35. Gaussian mixtures (`kmeans/` `gaussian-mixture/` `em/` `gmm-images/`) · 36. Neural networks (`neuralnet/` `backprop/` `cnn/` `unet/`) · 37. Evaluation and statistics (`object-eval/` `agreement/` `leakage/` `replicates/`) · 38. Variational autoencoders (`autoencoder/` `vae/` `vae-gen/`) · 39. Modern deep learning (`instances/` `detection/` `attention/` `diffusion/`) · 40. Networks for cells and objects *(planned)* (`stardist/` `cellpose/` `yolo/` `sam/` `cellnets-compare/`) · 41. Diffusion models (`score-matching/` `diffusion-sampling/` `guidance/` `diffusion-inverse/`) |
| VIII | Colour and fluorescence | 42. Colour (`colour-vision/` `rgb-linear/` `colour-perceptual/` `colour-display/`) · 43. Spectral unmixing (`crosstalk/` `lsunmix/` `endmembers/` `stain-unmixing/`) · 44. Colocalisation (`coloc/` `coloc-null/` `coloc-objects/` `coloc-confounds/`) · 45. FRET (`fret-physics/` `sensitised-emission/` `acceptor-bleach/` `ratiometric/`) · 46. Fluorescence lifetime imaging (FLIM) (`lifetime/` `decayfit/` `phasor/` `flim-fret/`) |
| IX | 3D and computing | 47. Marching cubes (`volumes3d/` `isocontour/` `marchingcubes/` `surface-measure/`) · 48. 3D graphics (`pipeline3d/` `rasterise/` `gpuformats/` `volrender/`) · 49. Geometry from multiple images (`cameras/` `two-view/` `stereo/` `sfm/`) · 50. Ray tracing (`raycast/` `whitted/` `bvh/` `pathtrace/`) · 51. How GPUs work (`gpu-parallel/` `gpu-memory/` `gpu-patterns/` `gpu-fit/`) |

Why this order (refined from the proposal in "Self-study restructure"):

- **Part I** keeps the foundations (old days 1–5) and adds splines, which need only `images/`
  (interpolation, homogeneous coordinates) and the regularisation pattern of deconvolution.
- **Part II** collects the transforms (wavelets → compression, Radon, edge-preserving smoothing,
  sparse representations) as their own part directly after Part I: all of them hang off
  `fourier/` and the deconvolution day; sparse representations come last (they build on wavelets
  and deconvolution and recall Radon's few-view reconstruction).
- **Part III** is the object pipeline (thresholding → morphology → labelling → watershed →
  measurements, a hard chain) followed by the days that need labelled objects: skeletons,
  moments, spatial statistics (entry after measurements).
- **Part IV** (point detection, Hough, integral images) needs only the edges day.
- **Part V** puts FFT registration first (keypoints recall NCC; optical flow builds on phase
  correlation and stitching); tracking comes after point detection and labelling. Geometry from
  multiple images is **not** here, because it needs the 3D graphics day (camera chain): it moved
  to Part IX.
- **Part VI** starts with graph cuts (owner of the energy vocabulary); Ising/MRF and graph-based
  segmentation follow directly, then the snakes → level sets → dynamic programming chain.
- **Part VII**: random forests (owner of the classification basics) → SVM, Gaussian mixtures →
  neural networks → evaluation → VAEs → modern deep learning → networks for cells and objects
  (planned, `plan/segmodels-plan.md`) → diffusion models.
- **Part VIII** groups colour with the fluorescence days that depend on it (unmixing,
  colocalisation, FRET, FLIM).
- **Part IX**: marching cubes (owner of 3D images) → 3D graphics → geometry from multiple images,
  ray tracing; the GPU day closes the course (it needs only bit depth and floating point, and its
  examples become back-references to the whole course).

Old numbers (18-day sequence of 2026-11-12) → new: 1→1, 2→2, 3→3, 4→4, 5→5, 6→12, 7→42, 8→13, 9→14, 10→15, 11→16, 12→23, 13→27, 14→33, 15→36, 16→37, 17→38, 18→39; the planned "day 19" of `plan/segmodels-plan.md` is day 40.

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
| Hysteresis thresholding | Introduced on the edges day (inside Canny, day 2); generalised as a segmentation tool on the thresholding day (day 12) |
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
  day (measurements keeps a one-slide recall, in `relations/lec.qmd`). Day 16 now has three units.
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
spatial-statistics day's entry point is after Measurements (day 16). The watershed day's optional
`ws-graphs/` unit is superseded by `ws-msf/` on the graph-based segmentation day. Edits to
**built units** implied by these decisions are listed in `plan/crosslink-todo.md` for the
cross-link pass.

## Prerequisite map (planned 2026-11-12; database, check, sidebar and per-page box built 2026-11-13)

For learners who want one topic and only what it needs.

- **Done (2026-11-13).** `days.yml`, `tools/days.py` (`make check-prereqs`, `make sidebar`),
  the generated sidebar and part colours, and the per-page box (`tools/prereqs.lua`, see
  README.md). Still to build: the overview page.
- **One database** of the course structure, the single source for navigation: e.g.
  `days.yml` at the repo root (machine-readable; this coursemap stays the human
  rationale). Per day: id, title, part, number, units (folder, title),
  **hard prerequisites** (days or units that must come first), **soft prerequisites**
  (nice-to-have back-references), concepts it owns (from the owners table). Units may
  list their own prerequisites at unit level (finer than day level). Seed it from
  `review-ordering.md`'s dependency table and the plans' "prerequisite gap" sections.
- **Overview page** (`overview.qmd`, linked from the home page and navbar): all days as
  an interactive dependency graph (a TypeScript demo reading the database: parts as
  columns, hard edges solid, soft edges dashed). Click a day or
  unit → highlight its transitive hard prerequisites and show an ordered **learning
  path** (topological order of just those prerequisites). Also a plain list view for
  print/accessibility.
- **Per-page prerequisite box**: every quiz.qmd/exercise.qmd shows "Before this unit:"
  with links to its direct hard prerequisites (and "see also" for soft ones), generated
  from the database at render time (a Quarto pre-render script or Lua filter, so no unit
  hand-maintains the list); every lec.qmd gets one line "Builds on: …" on its "About this
  lecture" slide. Reverse links ("Used later in: …") can come from the same data.
- A check in `make test` (or a separate `make check-prereqs`): every unit in the sidebar is
  in the database, every prerequisite exists, the graph has no cycles, and the sidebar
  order respects hard prerequisites.
- The sidebar in `_quarto.yml` is generated from the same database (`make sidebar`).

## Queue, second batch (requested 2026-11-13; all built 2026-11-13)

All six items are planned and built (marked below). To be planned and built after the current batch. Days and content may be reorganised when
planning (the user allowed reorganising days and content).

- **Built 2026-11-13** (`plan/keypoints-day.md`; now day 24, Part V). **Keypoints and descriptors** (day): interest-point detectors (Harris/Förstner recall from the
  edges day, DoG/scale-space extrema), orientation and scale invariance, SIFT-style gradient
  histograms and binary descriptors, matching (ratio test, mutual nearest neighbours), robust
  fitting with RANSAC (recall the Hough day), uses: stitching, registration, geometry from images.
  Becomes the owner of feature matching that several days currently assume.
- **Built 2026-11-13** (`plan/spatialstats-day.md`; now day 19, Part III, entry after Measurements). **Spatial statistics of objects** (day): point patterns (complete spatial randomness as the
  null model), nearest-neighbour distance distributions, Ripley's K/L and pair correlation,
  clustering vs regularity, edge corrections, marked patterns (types of cells), simulation
  envelopes; inputs from point detection and labelling; link to the evaluation day's statistics.
- **Built 2026-11-13** (`plan/contrast-unit.md`; day 3, unit `contrast/` between `highbit/` and `floatingpoint/`). **Contrast and intensity transforms** (unit, place when planning: Part I or the thresholding
  day): point operations, gamma/log, histogram equalisation (the cumulative histogram as the
  mapping), CLAHE, what these do to measurements (display vs analysis).
- **Built 2026-11-13** (`plan/graphseg-day.md`; now day 29, Part VI, after Ising/MRF). **Graph-based segmentation beyond graph cuts** (day or units): the random walker (a linear
  system on the graph Laplacian; recall ncut), minimum-spanning-tree segmentation
  (Felzenszwalb–Huttenlocher), superpixels (SLIC; recall k-means), watershed as a minimum
  spanning forest (recall the watershed day), hierarchical segmentation.
- **Built 2026-11-13** (`plan/sparse-day.md`; now day 11, last day of Part II). **Sparse representations** (day): sparsity in a basis (recall wavelets, DCT), L1 vs L2
  (geometry of the lasso), compressed sensing (incoherent sampling, recovery), sparse coding
  and dictionary learning, uses in denoising/inpainting/deconvolution and fast MRI.
- **Built 2026-11-13** (`plan/pdesmooth-day.md`; now day 10, Part II, after Radon; the proposed new group "Restoration" is not applied). **Edge-preserving smoothing and PDEs** (day): heat equation = Gaussian blur (scale space
  recall), anisotropic diffusion (Perona–Malik), total variation flow (recall Ising/TV),
  bilateral filter and non-local means in depth (recall deconvolution denoisers), inpainting
  (diffusion-based and exemplar-based), stability of explicit schemes (CFL recall from level sets).


## Self-study restructure (decided 2026-11-13, applied 2026-11-13)

The course is self-study and has no cap on days, so the earlier split into a fixed track and
optional days is gone. All days form one sequence, grouped into themed parts and ordered so
that hard prerequisites come first. Learners pick a topic and follow its prerequisites (the
prerequisite map makes that easy).

Applied:
1. **Part structure**: nine parts, 51 days ("Parts and days" above). Refinements of the first
   proposal: the transforms form their own Part II ("Transforms and reconstruction") right after
   Part I; detection (point detection, Hough, integral images) is its own Part IV; keypoints
   moved to "Aligning images and motion" (matching and registration); geometry from multiple
   images moved to the 3D part (it needs the 3D graphics day); colour opens Part VIII "Colour and
   fluorescence"; the GPU day closes Part IX "3D and computing"; Ising/MRF and graph-based
   segmentation directly follow graph cuts.
2. The order is in `days.yml`, checked by `make check-prereqs` (folders exist, prerequisites
   exist, no cycles, every hard prerequisite earlier, sidebar current); the sidebar and the
   per-part colours in `styles.css` are generated by `make sidebar`.
3. The old wording was removed from units, plans, the home page and this file ("the X day";
   group names became part numbers). The review files (`review-*.md`) are left as written:
   they are historical and use their own numbering.
4. "Day N" references in units and plans use the global numbers (old → new mapping above).

## Future work (queued 2026-11-13, not yet planned)

- **Fine-tuning strategies for neural network models** (day, placed after "Networks for cells
  and objects"). Currently only one slide in `attention/` (frozen features vs fine-tuning) and
  scattered mentions in the networks day. Scope: when to fine-tune vs use as is vs train from
  scratch; frozen features + a small head (linear probe) vs full fine-tuning vs partial (freeze
  early layers, layer-wise/discriminative learning rates, warm-up); parameter-efficient methods
  (adapters, low-rank updates such as LoRA, prompt/decoder-only tuning, e.g. tuning only SAM's
  mask decoder); how many annotations are needed, augmentation, class imbalance; human-in-the-loop
  and active learning (annotate the model's mistakes, retrain, repeat); domain shift and
  normalisation; catastrophic forgetting; evaluating on held-out images/experiments (recall the
  evaluation day). Demos: fine-tune a small network in the browser (TypeScript or Rust/Wasm —
  YOLOv11-rs supports training) and watch train/validation curves for frozen vs partial vs full;
  a learning-curve demo (accuracy vs number of annotated objects); a low-rank update visualiser.
