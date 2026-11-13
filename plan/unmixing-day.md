# Plan: a day on spectral unmixing

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
**Course map (2026-11-13, self-study restructure):** day 43 of 51, Part VIII "Colour and fluorescence". Hard prerequisites: Colour (day 42). Soft (see also): Bit depth and noise (day 3), Measurements (day 16). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VIII (spectral unmixing); entry after day 42 (colour).
Revised after the reviews: `unmixgeom/` merged into `endmembers/`; stain deconvolution moved here from the colour day as unit 4 `stain-unmixing/`.

## Goal of the day

After the day a student can:

1. explain why fluorophores (stains, materials) **bleed into each other's channels**: broad emission spectra seen through finite filter bands; compute a channel signal from a sampled spectrum and a filter, and write a multi-channel pixel as $\mathbf{y} = M\mathbf{a} + \text{noise}$ with the component spectra as the **columns** of $M$;
2. **compensate** the square case (2 dyes × 2 channels) by hand with $M^{-1}$, estimate a column of $M$ from a **single-stained control**, and recognise under- and over-compensation (tilted populations, negative abundances) in a scatter plot;
3. **unmix by least squares** when there are more channels than components: the normal equations, the pseudo-inverse, the geometric picture (projection onto the column space), and use the **residual** to detect a missing component such as autofluorescence;
4. predict **noise amplification** from spectral similarity: compute the condition number for two near-parallel spectra by hand, and explain why more channels help but cannot rescue nearly identical dyes; *(math)* derive $\operatorname{Cov}(\hat{\mathbf{a}}) = \sigma_n^2 (M^\top M)^{-1}$ and weighted least squares for Poisson noise;
5. describe the **geometry** of mixed pixels (a cone spanned by the spectra; a simplex when abundances sum to one) and what the constraints **non-negativity** (NNLS) and **sum-to-one** do to a noisy pixel outside the cone;
6. **find endmembers** when the spectra are unknown: count components from the singular values (PCA), pick pure pixels as the vertices of the simplex, or factorise with **NMF**, and explain why NMF's answer is not unique;
7. **separate histology stains** in brightfield (colour deconvolution): Beer–Lambert, optical density, the 3×3 stain matrix and its inverse, and estimate stain vectors from the image (Macenko as two-component vertex finding).

Why it fits this audience: the whole day is $A\mathbf{x} = \mathbf{b}$ with real data. Inversion,
least squares, the pseudo-inverse, condition number, SVD and eigenvectors are all in their
linear algebra; here each acquires a physical meaning (a column is a spectrum, a small singular
value is two dyes that look alike, the column space is the set of explainable pixels). The new
parts are the constraints (cones, simplices) and the factorisation without known spectra, both
of which have strong pictures.

Notation (course conventions): $\lambda$ is wavelength on this day (no prior weight
appears); noise std $\sigma_n$ (from `cameranoise/`); singular values $\sigma_i$ always with an index.

## Prerequisite gap

- **Least squares, pseudo-inverse, condition number, SVD**: from linear algebra (audience.md). Recall in one slide each, picture first (projection onto a plane in 3D; a thin parallelogram for a bad condition number). The SVD is used only through "singular values = how much a matrix stretches along its principal directions".
- **PCA** (covariance of a point cloud, principal axes): owned by `pca/` (day 1). Unit 3 uses it as "the SVD of the pixel matrix" with one recall slide.
- **Noise model** (Poisson shot noise, variance = mean; read noise, offset, saturation): recalled in one line from `cameranoise/` (day 3), used for weighted least squares and for the demos' noise model.
- **Linear light, white balance, the colour cloud, sRGB → linear**: day 42 (colour), recalled in unit 4.
- **Optical density / Beer–Lambert**: taught in unit 4 (moved here with stain deconvolution); the line-integral derivation is owned by the radon day, recalled in one line if taken.
- **Fourier series**: only for the optional phasor unit; a back-reference to `fourier/` (the first Fourier coefficient of a sampled curve).

## Links to other days

- **Colour** (day 42, `plan/colour-day.md`): same linear model, different physics. **Decided (review-organisation §2.8, course map):** this day is the one owner of linear unmixing: the general theory (N channels × K components, compensation as the square case, least squares and the pseudo-inverse, conditioning, non-negativity and sum-to-one, cone/simplex geometry, endmember discovery) **and** Ruifrok–Johnston stain deconvolution with `stainsep` and Macenko (unit 4, moved from the colour day's old unit 4). The colour day keeps white balance (in `rgb-linear/`) and a one-slide forward pointer to unit 4. Back-references to day 42: "a camera is a linear map from a spectrum to three numbers" (= "a filter set maps an emission spectrum to N channel values"), linear light, `colourcloud`. Shared code: `stainsep` calls the solver in `lib/unmix.ts` (square inverse = LS with N = K); spectrum plotting and wavelength-to-colour from `lib/colour.ts`.
- **Measurements** (day 16, `plan/measurements-day.md`, `coloc/`): bleed-through is a colocalisation confound there ("bleed-through fakes colocalisation" preset of `coloc`, with a pointer to unit 1 here, marked day). This day shows the converse in one slide: Pearson's $r$ between two channels before and after compensation on an image with no true colocalisation. Pearson / Manders are not re-derived.
- **Bit depth and noise** (day 3, `cameranoise/`): owns the detector noise model (shot noise, read noise, gain, offset, saturation). Here: an offset must be subtracted before unmixing (an offset is a fake component with a flat spectrum), and saturated pixels break linearity (one bullet each, back-reference).
- **PCA** (`pca/`, day 1): used, not re-taught. Clustering of pixel colours (k-means slide on day 42; GMM day) is a different model (each pixel belongs to one class) from unmixing (each pixel is a mixture); one contrast slide in unit 3: "clustering = which component; unmixing = how much of each".
- **Thresholding** (day 12): the abundance maps are what one thresholds afterwards (one bullet; e.g. "positive cells" from an unmixed channel).
- **Ising / MRF** (`plan/ising-day.md`): spatial priors on abundance maps (neighbouring pixels have similar abundances) as a "where this leads" pointer in unit 3; not taught.
- **Images as arrays & geometric transforms** (`images/`, day 1): "the columns of a matrix are where the axes land" and the `matrix2x2` demo are the picture behind compensation (one-line back-reference).
- **Fourier** (`fourier/`, day 1): the optional phasor unit is the first Fourier coefficient of a spectrum.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`crosstalk/`, `lsunmix/`, `endmembers/` (with the cone/simplex geometry of the former
`unmixgeom/`), `stain-unmixing/` (moved from the colour day); optional `phasor/`.

### 1. `crosstalk/`: Bleed-through, the mixing model and compensation

Adapted from the earlier course's `compensation/` unit (quantbio-3MB00A), reframed from flow cytometry to imaging: the cell becomes a pixel, the cytometer's detectors become image channels.

- Picture first: a two-colour fluorescence image of a cell (nucleus green, membrane orange) where the "orange" channel shows a ghost of the nucleus. Why: emission spectra are broad with long red tails; a filter collects a band; the green dye's tail reaches the orange band. Excitation crosstalk in one bullet (a laser excites both dyes).
- **Channel signal** = spectrum × filter transmission, summed over wavelength (a sampled integral, five or six bins in the by-hand case). One dye at unit amount gives one vector of channel values: a **column** of the mixing matrix $M$. Fluorescence **adds**: $\mathbf{y} = M\mathbf{a}$, rows = channels, columns = components. Each pixel is a vector; the image is $Y = MA$ (channels × pixels).
- Brightfield aside (one slide): absorbing stains multiply, so take optical density first; then the same model. Forward pointer to unit 4.
- **Single-stained controls**: an image with only dye 1 gives column 1, as the slope of channel 2 vs channel 1 in the 2D pixel scatter (regression through the origin, after offset subtraction).
- **Compensation** (N = K = 2): $\hat{\mathbf{a}} = M^{-1}\mathbf{y}$; the inverse has negative off-diagonals ("subtract 25 % of the green channel from the orange channel"). Under- and over-compensation: single-stained populations tilted up or down; over-compensation gives **negative** abundances, which no dye can have: the check without ground truth.
- Similar dyes: $\det M \to 0$, the inverse is large and amplifies noise (picture: the thin parallelogram). Teaser for unit 2.
- Consequence for measurements: bleed-through makes two independent channels correlate (Pearson $r$ before/after on the demo image; back-reference to day 16, `coloc/`). Physical remedies in one bullet: narrower filters (cost: fewer photons), sequential excitation.
- Flow cytometry as the motivating aside (one slide, kept from the old unit with its cytometer photo, CC BY 3.0): the same compensation with one "pixel" per cell; spectral cytometers unmix 20+ dyes with the least squares of unit 2. No instrument or software names in the student material beyond that one line.
- The math (`{.math}` slides): channel value as $y_c = \int s_k(\lambda)\,f_c(\lambda)\,d\lambda \approx \sum_i s_k(\lambda_i) f_c(\lambda_i)\Delta\lambda$; the 2×2 inverse formula; the regression-through-origin slope $\sum xy/\sum x^2$ as the least-squares estimate of a spillover coefficient.
- **Demo `spectramix`** (new): top, emission spectra of 2–4 dyes on a 400–750 nm axis (parametric: skewed Gaussian / log-normal shapes, peak and width draggable, drawn in their wavelength colour from `lib/colour.ts`); filter bands as draggable/resizable shaded rectangles (one per channel, 2–8 channels; toggle *spectral detector*: 8–32 contiguous bins). Right: the mixing matrix $M$ as numbers and a heat map, columns normalised to their own channel, bleed-through percentages, $\det M$ (square) or condition number (general). Presets: "well separated", "green into orange: the classic tail", "narrow filters: less bleed-through, fewer photons" (a photon count readout per dye), "two dyes 20 nm apart", "8-bin spectral detector".
- **Demo `bleedthrough`** (adapted from the old `compensation` demo: `demos/src/compensation/main.ts`, `math.ts`): left, a synthetic two-channel cell image (seeded; nucleus with dye 1, membrane with dye 2, some double-labelled vesicles, Poisson noise) shown as two channels and a merge; middle, the 2D pixel scatter (density-shaded) before compensation; right, after compensation with the user's guess. Sliders $s_{21}$, $s_{12}$; readouts $M$, $M^{-1}$, slopes and medians of the single-labelled populations (same "flat / under / over" verdict logic as the old demo), Pearson $r$ of the compensated channels, percentage of negative pixels. Presets as in the old demo: "no compensation", "under", "over", "reveal true $M$", plus "similar dyes: noise explodes". `data-mode="cells"` keeps the original four-population cytometry version for the aside slide.
- By hand (concept): (a) channel values for one dye from a 5-bin spectrum and two filters; (b) the old exercise's B1–B4 (numbers kept: $S = \begin{pmatrix}1 & 0.1\\0.2 & 1\end{pmatrix}$, compensate (2100, 1400), similar dyes with $\det = 0.19$ and a 10-unit error), reworded for pixels; (c) estimate a spillover coefficient from 4 control pixels. *(math)*: show that compensating with a wrong spillover leaves a residual slope of $s_\text{true} - s_\text{guess}$ (to first order).

### 2. `lsunmix/`: More channels than components: least squares and conditioning

- Picture: a **lambda stack** (one image per 10 nm band, 8–32 channels) of a cell with four labels; each pixel is a measured spectrum; the job is to explain it as a sum of four reference spectra. Applications in one slide: spectral confocal microscopy, multiplexed immunofluorescence, hyperspectral remote sensing (a land pixel = soil + vegetation + water), hyperspectral histology, Raman imaging.
- N > K: no exact solution with noise; take the abundances whose mixture is **closest** to the measured spectrum. Geometric picture (3 channels, 2 dyes): the explainable pixels form a plane; least squares drops the pixel perpendicularly onto it; the perpendicular part is the **residual**.
- Normal equations $M^\top M\hat{\mathbf{a}} = M^\top\mathbf{y}$; pseudo-inverse $M^+ = (M^\top M)^{-1}M^\top$ computed **once** and applied to every pixel (unmixing is one matrix product for the whole image). Square case = $M^{-1}$ (unit 1 is a special case).
- **Residual map**: where the model is wrong. A missing component (autofluorescence, an unexpected dye) lights up; adding its spectrum as an extra column removes it. Background/offset as a flat-spectrum component.
- **Conditioning**: two near-parallel spectra → small singular value → large noise in the difference of their abundances, while their sum is well determined (picture: the error ellipse elongated along the "dye 1 minus dye 2" direction). Condition number $\kappa = \sigma_\text{max}/\sigma_\text{min}$. More channels add photons and average noise, but do not separate identical shapes. **Choosing channels/filters** to minimise $\kappa$ (or the noise on the abundance you care about), linked back to `spectramix`.
- **Noise model** (recall `cameranoise/`, day 3): photon counts are Poisson, so bright channels are noisier in absolute terms; **weighted least squares** with weights $1/\text{expected count}$. Ordinary LS is fine when counts are high; in photon-starved images WLS (or Poisson maximum likelihood, pointer) helps.
- Unconstrained LS can give negative abundances in dim pixels: the motivation for the constraints in unit 3.
- The math: derivation of the normal equations (gradient of $\|\mathbf{y} - M\mathbf{a}\|^2$, or orthogonality of the residual to the columns); $P = M M^+$ as a projection; SVD form $M^+ = V\Sigma^{-1}U^\top$ and why $1/\sigma_\text{min}$ amplifies noise; $\operatorname{Cov}(\hat{\mathbf{a}}) = \sigma_n^2(M^\top M)^{-1}$ for i.i.d. noise; WLS $\hat{\mathbf{a}} = (M^\top W M)^{-1}M^\top W\mathbf{y}$ as maximum likelihood under the Gaussian approximation to Poisson.
- **Demo `lsunmix`** (new, main demo of the day): a synthetic cell (seeded, 128×128) with four components (nucleus, mitochondria-like network, membrane, diffuse autofluorescence with a broad spectrum), rendered into an N-channel lambda stack with Poisson noise. Controls: number of channels (3–32), spectral separation slider (moves two of the dyes' peaks towards each other), photons per pixel, include/exclude autofluorescence from $M$, method (*one channel per dye* / LS / WLS; NNLS appears in unit 3 via `data-methods`). Panels: true composite and unmixed composite (pseudocolour merge), the four abundance maps with true-vs-unmixed toggle, residual map, the spectra plot with the channel bins; click a pixel to see its measured spectrum, the fitted mixture and the per-component contributions stacked. Readouts: $\kappa(M)$, per-component RMSE against the truth, percentage of negative abundances. Presets: "one channel per dye: bleed-through", "least squares removes it", "two dyes 10 nm apart: noise explodes", "more channels help (a bit)", "forgot the autofluorescence: the residual lights up", "add it as a component", "photon-starved: WLS vs LS".
- By hand (concept): (a) 3 channels × 2 dyes: form $M^\top M$ and $M^\top\mathbf{y}$, solve the 2×2 system, compute the residual and check it is orthogonal to both columns; (b) two spectra $(1, 0.9, 0.5)$ and $(0.9, 1, 0.5)$-style near-parallel columns: eigenvalues of $M^\top M$ (2×2) and $\kappa$; how much a unit error in one channel moves each abundance; compare with two well-separated spectra; (c) predict: which of three given filter sets separates two given dyes best? *(math)*: derive the normal equations; show $P^2 = P$; derive $\operatorname{Cov}(\hat{\mathbf{a}})$.

### 3. `endmembers/`: The geometry of mixtures, and finding the spectra when they are unknown

Merges the former `unmixgeom/` (cones, simplices, NNLS) into `endmembers/` (review-organisation
§2.8, offered by this plan's own open question). The geometry is needed for vertex finding anyway;
NNLS keeps its picture and the active-set idea in words, the KKT details stay in the math layer.

- Pixels as a **point cloud in channel space** (3 channels so it can be seen; N channels via projection). With non-negative abundances and no noise, every pixel lies in the **cone** spanned by the component spectra; pure pixels on its edges, the dark pixels near the origin. Normalise each pixel by its total intensity (or require abundances to sum to one): the cone becomes a **simplex** (a triangle for three components). Same picture as mixtures on a straight line in the chromaticity diagram of day 42 (back-reference).
- Noise pushes dim pixels **outside** the cone; unconstrained LS then reports negative abundances. **NNLS**: the closest point inside the cone; the projection lands on a face or an edge, and the components "behind" that face get exactly zero (active-set idea in words, Lawson–Hanson). **Fully constrained** (non-negative + sum-to-one): closest point in the simplex. Honest costs: a **positive bias** in dim regions; a wrong endmember puts real pixels outside the simplex and the constraint hides the error.
- Contrast with clustering: "which component?" vs "how much of each?" (one slide). Non-linear mixing (quenching, FRET, inner filter, saturation) in one slide: the cloud bends. Spatial priors: pointer to the Ising/MRF day.
- Best case first: **reference spectra** from single-stained controls (unit 1) or a library. Why it is not always possible: spectra shift in the cell's environment, autofluorescence is unknown, a remote-sensing scene's materials are unknown.
- **How many components?** The SVD / PCA of the pixel matrix $Y$ (recall `pca/`, day 1): K singular values stand out of a noise floor (scree plot). PCA axes are **not** the spectra: they are orthogonal and have negative entries (picture: principal axes of a triangle-shaped cloud vs its corners).
- **Pure-pixel methods**: the spectra are the **vertices** of the simplex. N-FINDR (largest-volume simplex, a determinant) and VCA-style (most extreme pixel along a direction, project out, repeat). Concepts only, with the vertex picture. Macenko's stain estimate (unit 4) is the two-component case.
- **NMF**: $Y \approx WH$ with $W, H \ge 0$, no pure pixels needed. Multiplicative updates from a seeded positive start; the error never increases. **Non-uniqueness**: any cone that contains the data and lies inside the positive orthant fits equally well. Remedies in words: pure pixels, minimum volume, sparsity, some known spectra. Autofluorescence as the typical extra component: fix the known spectra and find one more.
- Validation without ground truth: residual, non-negativity, agreement with a control.
- The math: NNLS as a quadratic programme with KKT conditions; sum-to-one by a Lagrange multiplier or a heavily weighted row of ones; truncated SVD as the best rank-K approximation (Eckart–Young, stated); simplex volume $\propto |\det|$; NMF objective $\|Y - WH\|_F^2$ and the multiplicative update $H \leftarrow H \odot (W^\top Y) \oslash (W^\top W H)$ as gradient descent with a per-entry step; KL-NMF as Poisson maximum likelihood (pointer); the formal non-uniqueness $WR^{-1}, RH$.
- **Demo `unmixcloud`** (new, WebGL via `lib/gl3d.ts`, the shared 3D point-cloud viewer with day 42's `colourcloud`): a 3-channel synthetic image and its pixels as a rotatable cloud, endmember spectra as arrows, the cone (translucent) or, with *normalise*, the simplex; endmembers dragged in a 2D normalised-triangle panel. Click a pixel: its LS, NNLS and fully-constrained projections and abundance bars. Presets: "noise-free: everything inside the cone", "noise: LS goes negative", "NNLS snaps to the face", "normalise: the cone becomes a triangle", "a wrong endmember", "an unmodelled component: a fourth corner". SVG fallback `data-mode="2d"`.
- **Demo `endmembers`** (new): synthetic scene (3 or 4 components, with or without pure-pixel regions, noise, optional autofluorescence). Panels: scree plot; the pixel cloud projected to 2D (K = 3: the triangle in SVG; K = 4 via `unmixcloud`); true vs found spectra; found abundance maps. Methods: *PCA axes* (the wrong answer), *vertex finding* (step button), *NMF* (step / run, seed selector, error curve). Presets: "count the components", "pure pixels: the vertices are the spectra", "no pure pixels: vertex finding undershoots", "NMF from two seeds: two answers, same error", "fix three known spectra, find the autofluorescence".
- By hand (concept): (a) a 2-channel pixel just outside the 2D cone of two given spectra: LS gives a negative abundance; find the NNLS solution by projecting onto the nearer ray; (b) barycentric abundances of a pixel for three given endmembers; (c) from 6 given 2D pixel positions, the triangle of largest area (N-FINDR by hand); (d) decide K from a list of singular values. *(math)*: check the KKT conditions for a given NNLS solution; one multiplicative update of $H$ on a 2×2 example; two valid NMF factorisations of the same $Y$.

### 4. `stain-unmixing/`: Brightfield stains: colour deconvolution

Moved from the colour day's old unit 4 `colour-unmixing/` (review-organisation §2.8). Folder
name `stain-unmixing/`; if the colour day's `colour-unmixing/` folder was already built, keep
that folder name (the sidebar rule: never rename a built folder) and move it here.

- Recall day 42 in one slide: linear light, the colour cloud; white balance and brightfield white reference (flat-field) are there.
- **Colour deconvolution** (Ruifrok & Johnston 2001): brightfield stains **absorb**. Beer–Lambert per channel: $I_c = I_{0,c}\,10^{-\sum_k a_{kc} C_k}$ for stain concentrations $C_k$. **Optical density** $\mathrm{OD}_c = -\log_{10}(I_c / I_{0,c})$: then $\mathbf{od} = \sum_k C_k\,\mathbf{m}_k = M^\top \mathbf{C}$, **linear** in the concentrations: exactly the model of units 1–2 with N = K = 3. Picture: in RGB the pixels of a two-stain slide lie on a curved surface; in OD space they fill a flat wedge (the cone of unit 3) spanned by the two stain vectors (`colourcloud` with an OD option).
  - Stain vectors $\mathbf{m}_k$ (unit rows of $M$), e.g. Ruifrok's hematoxylin ≈ (0.65, 0.70, 0.29), eosin ≈ (0.07, 0.99, 0.11), DAB ≈ (0.27, 0.57, 0.78).
  - $\mathbf{C} = (M^\top)^{-1}\,\mathbf{od}$ per pixel: three concentration images (H, E, residual; or H, DAB, residual). With two stains, complete $M$ with the cross product as a third ("residual") vector.
  - What goes wrong: wrong stain vectors → **crosstalk** (unit 1 again); nearly parallel stains → ill-conditioned $M$ (unit 2's condition number); dark saturated pixels ($I = 0$, log undefined, clip); wrong $I_0$; scattering or co-localising stains break the linear model. DAB is not stoichiometric: its "concentration" is a relative measure (one slide of honesty for quantification).
  - **Estimating stain vectors** from the image: Macenko et al. (2009): the plane of the OD cloud by PCA and two extreme angles in it = **two-component vertex finding** (unit 3). Stain normalisation between scanners as one bullet.
- **Demo `stainsep`** (moved from the colour day): a synthetic brightfield image generated by Beer–Lambert from known concentration maps (seeded: nuclei with hematoxylin, cytoplasm and stroma with eosin; mode H-DAB: a subset of nuclei DAB-positive), with detector noise and an $I_0$ slider; plus a real H&E / H-DAB image when licensed. Panels: RGB image; OD point cloud with the stain vectors as arrows; the three concentration images; for the synthetic image, the true maps and a crosstalk readout. The stain matrix is **editable** (drag the stain vectors' tips on the OD unit sphere shown as a triangle, or type entries); $M$, $(M^\top)^{-1}$ and $\kappa$ shown live. Presets: "Ruifrok defaults", "the true vectors: perfect separation", "wrong eosin vector: crosstalk", "two stains, residual from the cross product", "nearly parallel stains: noise explodes", "wrong I₀ / saturated pixels", "estimate from the image (Macenko)", "H-DAB: count positive nuclei by thresholding the DAB channel" (Otsu, back-reference to day 12).
- By hand: (a) a pixel with $I = (80, 120, 200)$, $I_0 = 255$: its OD; (b) a two-stain matrix restricted to two channels: solve for the two concentrations; verify that mixing is additive in OD and multiplicative in RGB; (c) the residual vector as a cross product and the inverse of a small $3 \times 3$ $M$; (d) the condition number of two given stain matrices, and which separation is noisier.

### 5. (optional / further reading) `phasor/`: Spectral phasors: unmixing without a fit

- Each pixel's spectrum → two numbers: the real and imaginary parts of its **first Fourier coefficient**, normalised by the total intensity: $g = \sum_i I_i\cos(2\pi i/N)/\sum_i I_i$, $s$ likewise with sine. Back-reference to `fourier/`.
- Because the transform is linear and the normalisation is by intensity, a mixture of two components lies **on the line segment** between their phasors, at a position given by the intensity fractions; three components fill a triangle (the simplex of unit 3 again, in a fixed, data-independent 2D projection). Fit-free: select a region on the phasor plot and see the pixels light up in the image.
- FLIM in two bullets: the same mathematics on the fluorescence decay in time (pure single-exponential lifetimes on the universal semicircle); spectral + lifetime unmixing combined. Limits: higher harmonics needed when phasors coincide; noise spreads the cloud.
- **Demo `phasor`** (new): the `lsunmix` lambda stack (or a separate seeded scene) and its phasor histogram (2D, log scale, `lib/histogram.ts`); harmonic $n = 1, 2$; the pure components marked; brush a region on the phasor plot ↔ highlight pixels; a draggable cursor on the segment between two components reads off the fraction. Presets: "two dyes: a line", "three dyes: a triangle", "autofluorescence is a separate cloud", "similar dyes: overlapping phasors, try n = 2".
- By hand: the phasor of a 4-bin spectrum; of a 50/50 mixture of two given spectra, and check it is the midpoint (intensity-weighted).

## Adapting the earlier course's material

From `/home/mahogny/github/course/quantbio-3MB00A/`:

- `compensation/lec.qmd` → `crosstalk/lec.qmd`: keep the slide sequence (two dyes two channels → spillover matrix → before → estimate from a single-stained control → compensation → try it → why it works → similar dyes → summary), with "cell" replaced by "pixel" and the scatter coming from an image. Drop all R code and the flowCore slide (principle 0); the cytometer slides shrink to one aside slide. The speaker notes are mostly reusable.
- `compensation/exercise.qmd` → `crosstalk/exercise.qmd`: keep B1–B4 and E1–E4 (numbers, solutions) reworded for images; drop the "In R" section (R1–R5). R6 (4 detectors × 2 dyes, least squares) becomes a by-hand concept question in `lsunmix/exercise.qmd`. Restructure to the concept / `### Math questions` layout.
- Figures: `fig02-spectra.png` (schematic spectra and filter bands), `fig07-under-over.png`, `fig09-similar-dyes.png` are adaptable in content but drawn from cell populations; regenerate as image-based versions (static fallbacks of `bleedthrough`). `fig01-flow-cytometer.jpg` (CC BY 3.0, credit kept) and `fig01b-cytometer-schematic.png` for the aside. `make-comp-figs.R` as the template for a teacher-side figure script next to the demo (like `fourier2d/make-image.R`); the `.fcs` file and `fig03-real-data.png` are not needed (no real cytometry data in this course; keep the reference in teacher notes only).
- `demos/src/compensation/main.ts`, `math.ts` → `demos/src/bleedthrough/`: the UI uses only helpers that exist in this repo's `lib/ui.ts` (`svgEl`, `htmlEl`, `slider`, `presets`, `scale`, `numList`, `autoMount`); generalise `simulate()` from four lognormal populations to pixels of a synthetic two-channel image, keep `inv2`, `lsLine`, `median`, `summary` and the verdict logic; population mode kept as `data-mode="cells"`.
- New in this course: everything from N > K onward (least squares, residual, conditioning in full, WLS), the geometry and constraints, endmember discovery, phasors, all the image-based demos, and the spectra/filter demo.

## Shared engineering

- `demos/src/lib/unmix.ts` (new): forward model (spectra × abundances → N-channel stack, Poisson noise via `rng(seed)`, offset); parametric emission spectra and filter bands (channel integration); LS via a precomputed pseudo-inverse (Cholesky of $M^\top M$, K ≤ 8), WLS per pixel; NNLS (Lawson–Hanson active set) and fully constrained LS; residual and per-component RMSE; SVD/PCA of the pixel matrix (via the $N\times N$ covariance, N ≤ 32); vertex finding (VCA-style and a small N-FINDR); NMF with multiplicative updates (Frobenius and KL), seeded initialisation, error trace; spectral phasor. NNLS per pixel at 128×128 × K = 4 should take a few ms in TS; NMF at 16k pixels × 32 channels × K = 4 is ~2M multiply-adds per iteration, fine for live stepping. Measure before considering Wasm.
- `lib/linalg.ts` (owner: graph-cut day; colour adds 3×3 helpers): reuse the symmetric eigendecomposition (singular values of $M$ via $M^\top M$, PCA), solve, condition number; add Cholesky and a small dense $N\times K$ product if not there.
- `lib/colour.ts` (owner colour day): wavelength → display colour for spectra and filter bands, the pseudocolour merge of K abundance maps (one hue per component, additive), colour maps for residual (diverging, for signed values) and abundances (sequential). This day adds optical density and stain matrices (Ruifrok defaults) there.
- `lib/gl3d.ts` (owner 3D-graphics day): the point-cloud primitive, line segments (arrows) and translucent triangles (cone/simplex faces). `unmixcloud`, `colourcloud` (day 42) and SVM `svmlift` share one 3D point-cloud viewer.
- `lib/plot.ts` (owner morphology): spectrum curves, scree/bar plots, error curves; `lib/histogram.ts` (owner thresholding, 2D histogram added by colour): density-shaded scatter in `bleedthrough` and the phasor histogram.
- Noise: `poisson` and `rng` from `lib/random.ts` (owner bit-depth day).
- Synthetic images in `lib/synth.ts` (owner thresholding), including the Beer–Lambert H&E / H-DAB generator moved with `stainsep`; shapes from `lib/shapes.ts` and blur from `lib/imageops.ts`; one seeded generator for the multi-component cell used by `bleedthrough`, `lsunmix`, `unmixcloud`, `endmembers` and `phasor`, so students recognise the same cell across the day.
- Tests (`demos/test/unmixing-test.ts`, run by `make test`): LS recovers abundances exactly on noise-free data; $M^+ M = I$; residual orthogonal to the columns; NNLS satisfies KKT and matches brute-force enumeration of active sets for K ≤ 4; fully constrained solution sums to one; $\kappa$ of an orthonormal $M$ is 1; NMF error is non-increasing; vertex finding returns the true spectra on pure-pixel data; the phasor of a two-component mixture lies on the segment; the ported `bleedthrough` reproduces the old demo's numbers in `cells` mode; colour deconvolution recovers the concentration maps exactly on a noise-free synthetic image with the true $M$.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js (3D orbit by pointer only), fits 1150×560 on a slide (image 256 px + scatter/cloud + compact panel), labels ≥ 18 px, nothing hover-only, static figure fallback for each demo.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + by-hand compensation, `spectramix`, `bleedthrough` |
| 10:45–12:15 | Unit 2 lecture + by-hand normal equations and condition number, `lsunmix` |
| 13:15–14:45 | Unit 3 `endmembers/` lecture + by-hand cone/simplex and N-FINDR, `unmixcloud`, `endmembers` |
| 15:00–16:30 | Unit 4 `stain-unmixing/` lecture, OD and stain inverse by hand, `stainsep`; quizzes as wrap-up |

## Open questions

- **Decided (review-organisation §2.8):** the NNLS/simplex material merged into `endmembers/`; unit 3 is now the densest unit (geometry + NNLS + endmember discovery): if it overruns, NMF's math layer moves to the notes.
- **Decided (review-organisation §2.8, course map):** this day owns all linear unmixing, including stain deconvolution, the condition number and Macenko (as two-component vertex finding); the colour day keeps only a forward pointer.
- ~~Spectra in the demos: synthetic or tabulated real dyes?~~ Decided 2026-11-12: **parametric synthetic spectra** throughout (no licence issues, freely draggable). Dye names (FITC, PE, …) may label shapes that resemble them, as examples only.
- **A real lambda stack** (spectral confocal, 4+ labels, with single-stained controls) and/or a small hyperspectral remote-sensing crop would make units 2 and 4 honest (principle 4). Needs a source with a clear licence; until then the synthetic data carry the demos (and have ground truth).
- Phasor: optional (proposed) or a short core slot instead of part of unit 3's NMF? It is the strongest link to the Fourier unit and very visual.
- Should spatial regularisation of abundance maps (MRF / total variation) be more than a pointer?

## References

- Zimmermann (2005), *Spectral imaging and linear unmixing in light microscopy*, Adv. Biochem. Eng. Biotechnol.
- Dickinson, Bearman, Tille, Lansford & Fraser (2001), *Multi-spectral imaging and linear unmixing add a whole new dimension to laser scanning fluorescence microscopy*, BioTechniques.
- Roederer (2001), *Spectral compensation for flow cytometry: visualization artifacts, limitations, and caveats*, Cytometry.
- Keshava & Mustard (2002), *Spectral unmixing*, IEEE Signal Processing Magazine.
- Bioucas-Dias et al. (2012), *Hyperspectral unmixing overview: geometrical, statistical, and sparse regression-based approaches*, IEEE JSTARS.
- Lawson & Hanson (1974), *Solving Least Squares Problems* (NNLS).
- Heinz & Chang (2001), *Fully constrained least squares linear spectral mixture analysis*, IEEE TGRS.
- Winter (1999), *N-FINDR: an algorithm for fast autonomous spectral end-member determination*, Proc. SPIE.
- Nascimento & Bioucas-Dias (2005), *Vertex component analysis*, IEEE TGRS.
- Lee & Seung (1999), *Learning the parts of objects by non-negative matrix factorization*, Nature; Lee & Seung (2001), *Algorithms for non-negative matrix factorization*, NIPS.
- Donoho & Stodden (2004), *When does non-negative matrix factorization give a correct decomposition into parts?*, NIPS.
- Fereidouni, Bader & Gerritsen (2012), *Spectral phasor analysis allows rapid and reliable unmixing of fluorescence microscopy spectral images*, Optics Express; Digman et al. (2008), *The phasor approach to fluorescence lifetime imaging analysis*, Biophys. J.
- Ruifrok & Johnston (2001), *Quantification of histochemical staining by color deconvolution*, Analytical and Quantitative Cytology and Histology (unit 4); Macenko et al. (2009), *A method for normalizing histology slides for quantitative analysis*, ISBI.
- Bolte & Cordelières (2006), *A guided tour into subcellular colocalization analysis in light microscopy*, J. Microscopy (bleed-through as a confound).
