# Course map (decisions, 2026-11-12)

Single source of truth for the structure of the course. Plans and units follow this
file; where a plan disagrees, this file wins and the plan should be updated. Based on
the three reviews (`review-ordering.md`, `review-gaps.md`, `review-organisation.md`).

## Core track

Order within a part is the teaching order. Days marked *(new)* have no plan yet.

| Part | Theme | Days |
|---|---|---|
| I | Images as numbers | 1. Images, filters, frequencies (`images/`, `kernels/`, `fourier/`, plus a new `pca/` unit after `fourier/`) · 2. Edges, blobs and ridges *(new)* · 3. Bit depth and noise (`quantisation/`, new `cameranoise/`, `highbit/`, `floatingpoint/`) · 4. Image formation *(new)* · 5′. Deconvolution *(new, to be planned)* |
| II | From pixels to objects | 5. Thresholding · 6. Colour · 7. Morphology · 8. Labelling · 9. Watershed · 10. Measurements |
| III | Aligning images | 11. FFT registration |
| IV | Segmentation as optimisation | 12. Graph cuts (built: `seg-energy/`, `maxflow/`, `graphcut-seg/`, `ncut/`) |
| V | Learning from examples | 13. Random forests (owns classification basics) · 14. Neural networks (absorbs built `neuralnet/` as unit 1) · 15. Evaluation and statistics *(new)* · 16. Modern deep learning *(new)* · 17. VAEs (built: `autoencoder/`, `vae/`, `vae-gen/`) |

18 core days (deconvolution added 2026-11-12; numbering after day 4 shifts by one when the map is next renumbered). If the course must be shorter, the first candidates to move to electives
are: VAEs (17), FFT registration (11), colour (6), image formation + deconvolution (4–5, keep one combined day).

## Electives

Each group can be taken after the core day named.

| Group | Days | Entry after |
|---|---|---|
| Objects | Skeletons, moments | Labelling |
| Curves and energies | Snakes/ASM → level sets → dynamic programming; Ising/MRF | Graph cuts |
| Detection | Hough, integral images | Edges, blobs and ridges |
| More learning | SVM, Gaussian mixtures | Random forests |
| Transforms | Wavelets → compression; Radon | Part I |
| Spectral | Spectral unmixing | Colour |
| 3D | Marching cubes → 3D graphics (gets `gpuformats/`) | Part I |
| Motion *(should-have, not yet planned)* | Non-rigid registration, optical flow, tracking | FFT registration |

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
| Hysteresis thresholding | Introduced on the edges day (inside Canny, day 2); generalised as a segmentation tool on the thresholding day (day 5) |
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

- Should VAEs (day 17) come before modern deep learning (day 16), so diffusion builds on
  the VAE? Recommended: yes.
