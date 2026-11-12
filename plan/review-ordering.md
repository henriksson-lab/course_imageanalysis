# Review: an order for the days

Internal working doc (plan/ is not rendered). Status: review, 2026-11-12. Read-only review:
no other plan was edited. The edits it implies are listed at the end.

**Inputs.** The 28 day plans in `plan/` (the brief said about 33; these are the files that
exist today), the built units in `_quarto.yml` (Part 1 `images/ kernels/ fourier/`, which has
no plan file and is treated as one day, "Part 1"; Part 2 = graph-cut day; Part 3 = VAE day;
Part 4 = bit-depth day), `audience.md` and `designprinciples.md`. That is **29 days** of
about 4 × 1.5 h each. The order in which the plans were written is ignored.

**Method.** For every plan I read the prerequisite, overlap and "links" sections and the unit
list, and checked each "known from X" claim against the built units. A dependency is
**hard** when a unit cannot be taught without the material (a demo, a definition or a
by-hand exercise is built on it). It is **soft** when it is a back-reference or a one-slide
recall that would also work as a forward pointer.

---

## 1. Dependency table

"P1" = Part 1 (`images/`, `kernels/`, `fourier/`). In column 4, **bold** means the plan
claims to be the owner, and the concept is used by other days. ⚠ marks a concept that
another day also introduces as new (see section 4).

| Day | Hard prerequisites | Soft back-references (recall or pointer) | Introduces first (used elsewhere) |
|---|---|---|---|
| **Part 1** (built) | — | — | image = matrix, affine/homogeneous coords, inverse mapping, bilinear; convolution, border rules, box/3×3 Gaussian, Sobel, Laplacian, "smooth then differentiate" (LoG mentioned in one bullet); DFT as change of basis, 2D spectrum, convolution theorem, shift theorem (exercise), sampling/aliasing |
| **Bit depth** (built Part 4) | P1 `images/` | colour (sRGB storage slide), thresholding (histogram as display) | **quantisation error Δ²/12, SNR ~6 dB/bit, shot noise / noise sources**, integer types, wrap-around vs saturation, windowing / Hounsfield, sRGB storage, mipmaps, IEEE float, cancellation / Kahan |
| **Thresholding** | P1 (`kernels/` box/Gaussian) | bit depth (16-bit bins); labelling (CC as a picture for hysteresis); integral (local sums); morphology (top-hat) | **histogram, segmentation vocabulary (mask, foreground, ground truth)**, percentile threshold, **Otsu, isodata = 2-means in 1D, Bayes rule for two classes, minimum-error threshold**, local thresholds, **hysteresis**, **evaluation: confusion matrix, Dice/Jaccard, ROC/PR** |
| **Colour** | P1 (`images/`, interpolation, Gaussian), thresholding (Otsu on a derived channel, unit 3) | `fourier/` sampling (chroma subsampling); PCA (optional unit, Macenko); graph cuts (GrabCut); unmixing | **colour spaces, gamma / linear light, CIELAB, ΔE, YCbCr + 4:2:0, white balance, Beer–Lambert / optical density, colour deconvolution, colour maps**, colour segmentation (⚠ k-means in Lab) |
| **Morphology** | thresholding (binary masks), `kernels/` (sliding window, borders) | labelling (CC as picture); watershed (⚠ distance transform "owned" there); level sets; graph cuts | **SE, erosion/dilation, opening/closing, granulometry, grey morphology, top-hat, rank/median filters (⚠), reconstruction, regional minima, h-minima, fill holes, clear border, connectivity conventions**; ⚠ distance transform (used in unit 2 with a by-hand city-block DT) |
| **Labelling** | thresholding; morphology ("previous day": reconstruction, fill holes, clear border) | watershed, skeleton, level sets, marching cubes | **flood fill, two-pass labelling, union–find, region properties, second-moment (covariance) matrix → principal axes (names PCA ⚠), perimeter bias, Euler number, label images and measurement tables** |
| **Watershed** | `kernels/` gradient; labelling (CC, regional minima); morphology (h-minima, reconstruction, fill holes) | thresholding; graph cuts (seeded segmentation); level sets (fast marching) | landscape view, flooding, **priority queue (Meyer)** ⚠ with dynprog, markers, **distance transform + two-pass chamfer** ⚠ |
| **Skeleton** | morphology (erosion, opening, reconstruction); distance transform; labelling (connectivity, Euler number) | watershed, level sets (grassfire), graph cuts, marching cubes | hit-or-miss, thinning, medial axis, skeleton graph, **Hessian of a Gaussian, vesselness** |
| **Measurements** | labelling (regionprops, tables) | morphology (granulometry, edge band), thresholding (evaluation), bit depth (rescaling, saturation), colour / unmixing (channels), moments / skeleton (catalogue rows), distance transform | **shape-feature catalogue**, shoelace area, Feret, intensity features, **Haralick / co-occurrence texture** (⚠ entropy), colocalisation (Pearson, Manders), feature tables, z-scores |
| **Moments** | labelling (moments of order 0–2; or a 20-min prelude) | `fourier/` (orthogonal functions, $e^{i\theta}$), snakes (Procrustes), skeleton, dynprog (polar resampling), marching cubes (3D) | general moments, **Hu invariants, Legendre/Zernike, orthogonal-basis truncation**, Green's theorem (⚠ shoelace), nearest-mean / 1-NN classification (⚠ ML vocabulary) |
| **Graph cuts** (built Part 2) | thresholding (histogram, min-error threshold) | — (built units are self-contained) | **energy = data + λ·smoothness, MAP view, DP on a chain (math slide), graphs, s–t cut, max-flow/min-cut, seeds, Gaussian affinity, graph Laplacian, Fiedler vector, normalized cut** |
| **Ising / MRF** | P1; graph cuts' energy vocabulary (translation slide $\lambda = 2J$) | thresholding (Bayes), graph cuts `ncut/` (Laplacian), level sets (length term) | **Gibbs/Boltzmann, Markov chains (stationary = eigenvector), MCMC, MAP vs MPM, Gaussian MRF, total variation** (⚠ Gaussian noise model), Potts, ICM |
| **Snakes / shape models** | P1 (gradient, Gaussian σ, ⚠ Gaussian pyramid "from kernels") | dynprog (DP snake), level sets (topology), graph cuts (**not referenced; should be**) | **calculus of variations / Euler–Lagrange**, snake energy, GVF, ⚠ **PCA** (claims "first time"), **Procrustes**, point distribution model, ASM |
| **Level sets** | P1 (Taylor step) | thresholding (isodata), graph cuts (same energy), snakes (GAC vs snakes), marching squares, dynprog (fast marching ↔ Dijkstra) | implicit curves, signed distance function, curvature, level-set equation, CFL, geodesic active contours, Chan–Vese, Eikonal / fast marching |
| **Dynamic programming** | P1 (gradient, bilinear) | graph cuts (`segchain`, graphs), distance transform (chamfer = DP), watershed (priority queue), snakes (energy), level sets | **Bellman principle**, seam carving, polar unwrapping (`lib/polar.ts`), **Dijkstra** / live-wire, DTW |
| **Hough** | `kernels/` gradient; thresholding (hysteresis) | morphology (regional maxima), moments, snakes (pose), radon (same $(\rho,\theta)$), dynprog | ⚠ **Canny with non-maximum suppression** (no owner), gradient direction as a number, accumulator / voting, **RANSAC**, least-squares circle fit |
| **Integral image** | `kernels/` (box filter, borders) | thresholding (local thresholds), bit depth (overflow), dynprog (table filling), wavelets (Haar), moments, graphics3d (mipmaps) | **summed-area table**, constant-cost box filters, Haar-like features, ⚠ feature stack (box group), **boosting / AdaBoost**, ⚠ stumps, cascade |
| **FFT registration** | `fourier/` (DFT, FFT, convolution and shift theorem); `kernels/` (correlation vs convolution) | `images/` (interpolation; "registration" teaser), integral (fast NCC denominators), compression (leakage), snakes (Procrustes), graphics3d (3D), graph cuts (Laplacian for stitching) | **NCC, template matching, FFT cross-correlation, phase correlation, Hann window, sub-pixel peak, stitching / drift**, log-polar (Fourier–Mellin) |
| **Radon** | `fourier/` (2D DFT, convolution theorem, sampling) | `images/`, bit depth (shot noise), Ising (TV), Hough (same convention), graphics3d (transmittance), wavelets | Radon / sinogram, back-projection, **Fourier slice theorem**, ramp filter / FBP, algebraic reconstruction (ART), CT artefacts |
| **Wavelets** | `fourier/` and `kernels/` | compression (JPEG 2000), Ising and level sets (denoising), snakes (pyramid) | **Haar, MRA, filter banks, DWT, sparsity**, lifting, shrinkage denoising, ⚠ PSNR + Gaussian noise model, optional CWT / ⚠ LoG blob detection |
| **Compression** | `fourier/` (DFT as basis, "keep largest") | bit depth (Δ²/12), colour (YCbCr, 4:2:0), wavelets, PCA, moments | **information and entropy** (⚠ used by forest and Haralick), Huffman/arithmetic, prediction, **DCT, KLT**, JPEG, ⚠ **MSE/PSNR, SSIM**, file formats for science |
| **Unmixing** | linear algebra only | colour (same model, Beer–Lambert), bit depth (shot noise, offset), PCA, measurements (colocalisation), Ising | N×K mixing model, pseudo-inverse and conditioning, non-negativity, simplex geometry, endmembers, NMF, phasors |
| **GMM** | thresholding (Bayes, isodata, min-error) | PCA / covariance, colour (Lab), graph cuts (data term), Ising (HMRF), VAE (ELBO), measurements, SVM | **k-means as an algorithm**, feature-space clustering, **multivariate Gaussian**, **maximum likelihood**, **EM**, BIC/AIC, per-pixel background models |
| **SVM** | linear algebra; ⚠ ML vocabulary (claims to introduce it) | neural nets, GMM, thresholding (ROC), measurements, moments, graph cuts (RBF = Gaussian affinity), skeleton (Hessian), wavelets (LoG) | linear classifier, margin, hinge loss, dual, kernel trick / RBF, standardisation, ⚠ feature stack, ⚠ train/test/CV |
| **Random forest** | `kernels/` (Gaussian, derivatives) | thresholding (evaluation), measurements (object tables), moments (1-NN), integral (boosting), neural nets (`nnfit` hinges vs steps), GMM, graph cuts (data term); ⚠ LoG (wavelets), Hessian (skeleton) | decision trees, Gini / ⚠ entropy, bagging, OOB, **feature importance**, ⚠ **feature stack and paint–train–correct**, ⚠ train/test/overfitting, structure tensor (no other owner) |
| **Neural nets** | `kernels/` (convolution) | thresholding (Dice, Otsu baseline), measurements (learned vs designed features), graph cuts, watershed / labelling (instance splitting), PCA, Ising (Hopfield) | **neuron, losses, gradient descent, backprop, softmax/cross-entropy, Adam**, ⚠ overfitting + regularisation, **CNN, U-Net, Dice loss**; ⚠ absorbs the built `neuralnet/` |
| **VAE** (built Part 3) | its own `neuralnet/` unit (or the neural-net day) | snakes (PDM) | ⚠ "matrix, bend, matrix" (`neuralnet/`), ⚠ **PCA in many dimensions** (`autoencoder/`, "recap or introduction"), autoencoder, **KL, ELBO**, reparameterisation, latent space |
| **Marching cubes** | `images/` (bilinear → trilinear) | thresholding (choosing $c$), level sets, labelling, watershed, graph cuts, graphics3d, morphology | **3D images: stacks, voxel spacing, first MIP** (⚠ with graphics3d), marching squares / cubes, ⚠ shoelace / divergence theorem, normals from the gradient, surface measures |
| **Graphics 3D** | `images/` (homogeneous coordinates, inverse mapping) | `fourier/` (aliasing), colour, bit depth (formats), marching cubes (3D images) | **4×4 transforms, projection, meshes, rasterisation, z-buffer, shading, volume rendering / transmittance**, oblique reslicing, MIP/AIP |

### The shape of the graph

The hard edges form a shallow DAG with one long chain:

```
Part 1 ──┬── thresholding ── morphology ── labelling ──┬── watershed ── skeleton
         │        │                                    ├── measurements
         │        │                                    └── moments
         │        ├── graph cuts ──┬── Ising
         │        │                ├── (snakes) ── level sets
         │        │                └── dynprog
         │        ├── colour ── unmixing
         │        ├── GMM
         │        └── hough (hysteresis)
         ├── bit depth
         ├── FFT registration, radon, wavelets ── compression        (all off fourier/)
         ├── random forest ── neural nets ── VAE ;  forest ── SVM, integral
         └── marching cubes ── graphics3d
```

Three observations drive the ordering:

1. **Thresholding is the root of almost everything after Part 1.** Thirteen plans recall it
   (histogram, mask, Bayes two-class decision, evaluation). It must be the first day after
   the foundations, and its evaluation unit (Dice, ROC) is reused by every learning day.
2. **Morphology → labelling → watershed is a real chain** (each day's demos and by-hand
   exercises use the previous day's operators). Skeleton and measurements hang off its end.
3. **Most other days hang directly off Part 1** (the Fourier family, the 3D pair, the
   learning days) and can float. Their position is a pedagogical choice, not a constraint.

---

## 2. Who should introduce each shared concept

This resolves every "first time" claim against the proposed core order (section 3). Where
the owner is an elective, the core day that uses the concept must stand alone.

| Concept | Introduce on | Then recalled by |
|---|---|---|
| Gaussian of width σ, separability, **gradient direction**, **LoG/DoG blob detection**, **Gaussian pyramid**, **median filter** | **new Part 1 unit `scale/`** (see conflict C1) | thresholding (median), hough, snakes (pyramid), forest/SVM (LoG), wavelets (CWT), skeleton (Gaussian derivatives), gpuformats (mipmaps) |
| Noise model (shot + read + Gaussian), SNR, **PSNR** | bit depth (`quantisation/` already has SNR in dB; add PSNR in one line) | Ising, wavelets, radon, compression (keeps SSIM and "metrics disagree") |
| Histogram, mask, Bayes two-class rule, isodata = 2-means | thresholding | everything segmentation; GMM generalises |
| Segmentation evaluation (confusion, Dice, ROC/PR) | thresholding `seg-eval/` | forest, neural nets (Dice loss), SVM, measurements |
| Hysteresis | thresholding (idea and demo) | labelling (implementation = reconstruction + labelling), hough (Canny), skeleton |
| Connectivity (4/8) | thresholding and morphology as a picture; labelling as an algorithm | watershed, skeleton, marching cubes |
| **Distance transform** (definition, by hand, two-pass chamfer) | **morphology** unit 2 (moved from watershed; see C2) | watershed (uses it), skeleton, measurements, dynprog (chamfer = DP), level sets (SDF) |
| Reconstruction, regional minima, h-minima | morphology | labelling, watershed, skeleton, hough |
| Covariance matrix of a 2D point cloud, eigenvectors = principal axes, the name **PCA** | **labelling** `regionprops/` (already says "spectral theorem and PCA") | VAE, colour, GMM, snakes, compression, unmixing |
| PCA in many dimensions (eigen-images, explained variance, reconstruction error) | **VAE** `autoencoder/` (built) | snakes (PDM = PCA of shapes), compression (KLT), unmixing, colour (decorrelation) |
| Energy = data + smoothness, λ, MAP view | graph cuts `seg-energy/` (built) | Ising, snakes, level sets, dynprog, radon (ART + TV), wavelets (shrinkage) |
| Graph, graph Laplacian | graph cuts `maxflow/`, `ncut/` | Ising (`gmrf/`), dynprog, FFT registration (stitching), SVM (Gram = affinity) |
| Classifier, feature space, train/test, overfitting, validation, **feature stack**, paint–train–correct | **random forest** (first learning day) | neural nets (adds validation curves, regularisation), SVM, GMM, integral, moments, measurements |
| Decision stump | random forest | integral (boosting) |
| Neuron, loss, gradient descent, backprop | neural nets | VAE (`neuralnet/` shrinks to recall), SVM (hinge = ReLU) |
| Entropy / information | compression owns; random forest uses Gini as main criterion and states entropy in one line | measurements (Haralick entropy) |
| Multivariate Gaussian, maximum likelihood, EM | GMM | graph cuts (GrabCut pointer), Ising (HMRF), VAE (KL already uses 1D/diagonal) |
| Hessian of a Gaussian | random forest (concept slide, core) | skeleton (vesselness, full treatment), SVM |
| 3D images (stacks, spacing, MIP) | marching cubes unit 1 | graphics3d (full treatment), FFT registration, moments 3D |
| Shoelace / Green's theorem | measurements (shoelace, concept) | moments (Green, math layer), marching cubes (divergence theorem) |
| Canny / non-maximum suppression | hough unit 1 (elective; no core day needs it) | — |
| Linear mixing model, condition number | colour unit 4 (core, as application) | unmixing (general theory) |

---

## 3. Proposed ordering

### Assumptions on size

The course length is not stated anywhere in the repo. With 29 days of about 6 h, all days
cannot be core in a single course of normal size (a 7.5 ECTS course holds roughly 12–15
teaching days). The proposal is therefore a **core track of 13 days** plus **electives
grouped by theme**. Each elective group has an entry point after a specific core day. A
**lean core of 10 days** is given for a shorter course.

### Core track (13 days)

| # | Day | Part / theme | Why here |
|---|---|---|---|
| 1 | **Part 1** (`images/`, `kernels/`, `fourier/` + new `scale/`) | **I. Images as numbers** | Everything rests on it. With `scale/` added it is a full day of four units and owns the filters that eight plans assume. |
| 2 | **Bit depth** (built) | I | Concrete and needs only `images/`. It answers the microscopy student's first real question ("why is my 12-bit image black?"), defines the noise model, SNR and saturation before any day measures intensities, and supports thresholding's 16-bit bin choice. |
| 3 | **Thresholding** | **II. From pixels to objects** | Root of all segmentation. It introduces the vocabulary (mask, ground truth), the Bayes two-class rule and evaluation, which later days only recall. |
| 4 | **Colour** | II | It extends day 3 directly: thresholding in a better space (hue, $a^*b^*$). Colour deconvolution produces the stain images that day 8 measures. Placing it here keeps the morphology → labelling → watershed chain unbroken. |
| 5 | **Morphology** | II | It cleans day 3's masks and owns the distance transform and reconstruction for days 6–7 (and the skeleton elective). |
| 6 | **Labelling** | II | Hard prerequisite: morphology. It also introduces the covariance matrix / principal axes concretely: a cloud of pixel coordinates you can see. |
| 7 | **Watershed** | II | Needs days 5–6. Its motivation (touching nuclei counted as one) is labelling's last unit. |
| 8 | **Measurements** | II | The end of the pipeline: a label image becomes a table. It closes Part II with the "so what" and feeds the object-classification examples of day 11. |
| 9 | **FFT registration** | **III. Comparing and aligning images** | It brings Fourier back after a week (spaced retrieval). It is a concrete payoff of the shift theorem, it changes pace between two segmentation blocks, and stitching and drift correction are daily practice in microscopy. |
| 10 | **Graph cuts** (built) | **IV. Segmentation as optimisation** | The first "global" segmentation: it answers day 3's noise failure ("neighbours should agree") and sets the energy/graph vocabulary for every elective in this part. It comes after watershed, which settles watershed's open question: the seeded-segmentation comparison lives here, as a back-reference to markers. |
| 11 | **Random forest** | **V. Learning from examples** | The first learning day should be the most concrete: a tree is nested `if`s. Pixel classification is "segmentation by painting", so it ties straight back to Parts II and IV (a forest probability is a graph-cut data term). It owns the ML vocabulary and the feature stack built from Part 1's `scale/` filters. |
| 12 | **Neural networks** | V | The motivation comes from day 11: "the feature stack was hand-made; let the network learn it". CNN and U-Net, with Dice loss reusing day 3's Dice. |
| 13 | **VAE** (built) | V | Needs day 12 (its `neuralnet/` unit shrinks to a recall) and day 6's 2D covariance. `autoencoder/` lifts PCA to many dimensions. Generative models close the course. |

Pedagogical arguments in one place:

- **Concrete before abstract.** Each part opens with the most tangible day: bit depth
  (numbers in a file), thresholding before energies, trees before
  networks, a 2D point cloud of a nucleus (labelling) before eigen-images (VAE).
- **Motivation chains.** Every core day starts from a failure of an earlier day: noise breaks
  thresholds (3 → 5, 3 → 10); touching objects merge (6 → 7); hand-made features limit the
  forest (11 → 12); a plain autoencoder cannot generate (13, internal).
- **Spacing and interleaving.** Fourier returns on day 9 (and in the electives radon,
  wavelets, compression). Probability returns as Bayes (3) → MAP energy (10) → class
  probabilities (11–12) → ELBO (13). Eigenvectors return as principal axes (6) → Fiedler
  vector (10) → PCA in many dimensions (13).
- **One first introduction per concept** (section 2). Every later day uses a one-line recall.

### Electives, grouped by theme

Each group lists its entry point (the earliest core day after which it can run) and its
internal order. The groups are independent of each other unless stated.

| Group (theme) | Days, in order | Entry after core day | Notes |
|---|---|---|---|
| A. **Objects, deeper** | skeleton; moments | 8 (skeleton needs 5–7; moments needs 6) | Highest priority for a bio-image audience (neurons, vessels; shape descriptors). The measurements catalogue rows then turn from forward pointers into back-references. |
| B. **Curves and energies** | snakes → level sets → dynamic programming; Ising (any time after 10) | 10 | Snakes before level sets: level sets answer snakes' topology problem. Dynprog last: it reuses `segchain`, the DP snake and the chamfer-as-DP picture. Ising can also follow graph cuts directly; it is the probabilistic reading of day 10. |
| C. **Detection** | hough; integral image | hough after 3 (needs hysteresis and `scale/`); integral after 11 (stumps, feature stack) | Viola–Jones reads best right after the forest day (stump → boosting). |
| D. **More learning** | SVM; GMM | 11 (SVM); GMM after 11, ideally after 13 (ELBO link) | Both recall the forest day's vocabulary. GMM generalises day 3's isodata and min-error threshold, and day 4's k-means in Lab. |
| E. **Transforms and imaging physics** | wavelets → compression; radon | 9 (any time after Part 1, but best after day 9 refreshes Fourier) | Wavelets before compression (JPEG 2000 is an application). Radon after Hough if both run (same $(\rho,\theta)$ pictures). |
| F. **Several channels** | unmixing | 4 (colour) and 2 (shot noise) | Best directly before or after measurements' colocalisation unit (bleed-through). |
| G. **3D** | marching cubes → graphics3d | 3 (marching cubes needs thresholding for "choosing $c$", and is better after 6 for label surfaces) | Marching cubes introduces 3D images; graphics3d unit 4 then recalls them. |

Recommended first electives, if 1–3 extra days are available: **skeleton** (group A), then
**level sets** (with a short snakes recap) or **Ising**, then **unmixing** or **wavelets**
depending on the student mix.

### Alternative orderings

- **Lean core (10 days)**: 1 Part 1, 2 thresholding, 3 morphology, 4 labelling,
  5 watershed, 6 measurements, 7 graph cuts, 8 random forest, 9 neural nets, 10 bit depth or
  VAE. Colour, FFT registration and the other of bit depth/VAE become electives. Cost:
  labelling must then introduce PCA alone (it already does in 2D), and measurements'
  intensity slides must state the saturation and rescaling facts themselves.
- **Imaging-physics flavour (13 days)**: keep days 1–3, 5–8 and 10 of the core track. Replace
  4, 9 and 11–13 with colour, unmixing, FFT registration, wavelets, compression and radon,
  in that order. This suits a medical-imaging or physics cohort. The learning block becomes
  one elective day (random forest), and neural nets/VAE move to electives. No hard
  dependency breaks, because the Fourier family hangs only off Part 1.
- **Early learning variant**: run Part V right after day 8 and graph cuts after it. This is
  valid (no hard edge from 10 to 11–13), but graph cuts then loses the "classifier
  probability as data term" link as a back-reference, and every learning day's graph-cut
  slide becomes a forward pointer. Not recommended.

---

## 4. Conflicts and cycles found, and how to break them

Hard cycles (two days each needing the other first) are rare: plans written "no order
assumed" mostly fall back to "recap if not seen". The real problems are **double first
introductions** and **false "known from X" claims**.

**C1. Missing owner for scale and non-linear filters (false claims about `kernels/`).**
Several plans assume material from `kernels/` that the built unit does not contain. The
built unit has only 3×3 kernels, plus one bullet on LoG with σ.
- thresholding: "box, Gaussian **and median** filter known from `kernels/`" (no median there);
- snakes: "a **Gaussian pyramid** (from *Filtering & convolution*)" (absent);
- hough: "gradient direction stated only in words", and LoG "taught in `kernels/`" (one bullet);
- forest and SVM point **LoG/DoG** to the wavelets day's *optional* `cwt/`; wavelets itself
  asks who owns LoG blob detection, which `fourier/exercise.qmd` promises.
**Fix:** add a fourth Part 1 unit, `scale/` ("Scale: Gaussians, blobs and pyramids"). It
covers the Gaussian with σ and separability, gradient magnitude **and direction**,
LoG/DoG at a scale (blob detection), the Gaussian pyramid, and the median as the first
non-linear filter (with a pointer to morphology's rank filters). Part 1 then becomes a
four-unit day like the others.

**C2. Distance transform: morphology ↔ watershed ↔ skeleton.** Morphology says the DT is
"defined and computed on the watershed day", yet morphology unit 2 uses it ("erosion by a
disc = threshold of $D$") and has a by-hand city-block DT on an 8×8 grid. Watershed (later
in any sensible order, since it needs morphology) owns the definition and the two-pass
chamfer. Skeleton depends on it and says "watershed day". The result is a forward
reference from the earlier day, and a second "first" by-hand DT.
**Fix:** morphology owns the DT (definition, picture, city-block by hand, two-pass algorithm
in the math layer). Watershed unit 4 recalls it and spends the time on the splitting
itself. `lib/distance.ts` stays shared. Dynprog's "chamfer is DP" points to morphology.

**C3. PCA has three "first introductions".** Snakes unit 3 says PCA "appears here for the
first time". The built `autoencoder/` is written as "recap or introduction". The VAE plan
says "whoever comes first owns it". Labelling unit 3 computes the covariance matrix of the
pixel coordinates and names "the spectral theorem and PCA". Colour, GMM, compression and
unmixing each cite snakes and labelling as the source.
**Fix:** a two-step ownership. **Labelling** (core, day 6) introduces the covariance of a 2D
cloud and its eigenvectors as principal axes, and says "this is PCA". **`autoencoder/`**
(core, day 13) owns PCA in many dimensions. Snakes recalls both: the point distribution
model is "PCA of landmark vectors". Its open question about a shared PCA slide set is
answered by labelling's slide plus `autoencoder/`'s two-pixel figure.

**C4. Two neural-network introductions.** The built `neuralnet/` (VAE day, unit 1) and the
neural-net day's unit 1 teach the same material with the same numbers. **Fix:** with the
neural-net day in the core before the VAE, `neuralnet/` shrinks to the 15-minute recall the
neural-net plan proposes. The VAE day gains an hour (real microscopy data or β-VAE weights,
its own open questions). If the neural-net day is cut (lean core), `neuralnet/` stays
as built.

**C5. ML vocabulary claimed by three days.** SVM ("introduces *feature, training set, test
set, overfitting, cross-validation* so other days can use them"), random forest ("introduces
training/test, overfitting and validation itself") and neural nets (overfitting, train /
validation / test). Moments (`moment-features/`) and measurements (`features-use/`) also
introduce classification "briefly". **Fix:** the first learning day, random forest, owns the
vocabulary. Neural nets adds learning curves and regularisation. SVM recalls everything and
uses the time for margins and kernels. The moments and measurements optional units use the
words with a pointer.

**C6. Feature stack claimed by three days.** Random forest unit 4, SVM unit 4 and integral
unit 3 all introduce a per-pixel multiscale feature stack. The forest plan already proposes
the fix: **forest owns** the stack and the paint–train–correct loop. SVM keeps
standardisation and CV by stroke. Integral keeps "large boxes are free" and the box group.
There is one demo shell (`pixelclass`). Adopt it. With `scale/` in Part 1 (C1), the stack's
channels are all recalls except the Hessian and the structure tensor (next item).

**C7. Hessian owned by an elective but used by a core day.** Skeleton (elective) owns
Hessian eigenvalues and vesselness. Forest (core) uses them as a stack channel with a
pointer to skeleton. **Fix:** forest gives the Hessian as one concept slide ("tube vs blob
from two eigenvalues") and owns the structure tensor (no other owner). Skeleton recalls both
and does the full vesselness.

**C8. Stumps vs boosting.** Integral wants stumps "introduced once"; forest proposes to own
the stump, and integral owns boosting. In the proposed order forest comes first, so it works
as written. Integral's unit 4 opens with "a stump is a depth-1 tree (random-forest day)".

**C9. Thresholding re-taught by the built Part 2.** `seg-energy/lec.qmd` has a full
"Thresholding" slide (histogram, valley, noise specks) written before the thresholding day
existed. The morphology and watershed plans also carry recaps. With thresholding as day 3,
these become one-line recalls, as the thresholding plan lists.

**C10. Energy minimisation introduced three times.** `seg-energy/` (built) introduces
data + λ·smoothness. Ising unit 1 builds its own energy (with a translation slide to
`seg-energy/`). Snakes unit 1 "Active contours as energy minimisation" does **not reference
graph cuts at all**. **Fix:** graph cuts owns the vocabulary (it is core and comes first).
Snakes unit 1 adds one recall slide ("same structure: a data term pulls to edges, an
internal term keeps it smooth; here the unknown is a curve, not a labelling"). Ising's
translation slide becomes a back-reference. Level sets already says it right.

**C11. Noise model and PSNR defined in four places.** Wavelets says "noise model and PSNR
are not defined in Part 1" and defines them. Ising defines Gaussian noise. Compression owns
MSE/PSNR/SSIM in `lib/metrics.ts`. Radon reuses PSNR. The built `quantisation/` already
defines the noise sources and SNR in dB. **Fix:** bit depth (day 2) owns noise and SNR; add
PSNR as one line on the existing SNR slide. Compression keeps SSIM and the "same MSE,
different images" demo. Wavelets and Ising recall.

**C12. Integral image taught twice.** Thresholding unit 3 computes local means "via the
integral image" and has a by-hand item (c) "a 4×4 integral image". Integral owns it. Since
integral is an elective, **thresholding should compute local means as a box filter from
`kernels/`** and only name the integral image as the fast trick ("see the integral-image
day"). Drop by-hand (c) there, or keep it as optional. Integral unit 1 is then the first
real treatment in any order.

**C13. Hysteresis pointers disagree.** Thresholding owns the idea and demo; Hough recalls
it from thresholding; skeleton says "hysteresis: labelling day". Labelling has one slide
"hysteresis = reconstruction + labelling". **Fix:** the skeleton pointer goes to
thresholding (with labelling for the implementation).

**C14. Watershed vs graph cuts: "which comes first?"** (watershed open question). **Fix:**
watershed first (day 7, inside the objects chain), graph cuts later (day 10). The
"seeded segmentation on the pixel graph" comparison lives on graph cuts (`graphcut-seg/`
seeds slide: "markers on the watershed day, seeds here"), and watershed's optional
`ws-graphs/` points forward.

**C15. Labelling vs skeleton order** (skeleton open question). Labelling first (core); this
removes skeleton's two-slide recap of connectivity.

**C16. k-means.** Thresholding presents isodata as "k-means with $k = 2$ in 1D". Colour
unit 3 runs k-means in Lab assuming that day. GMM owns k-means as an algorithm. With colour
(core) before GMM (elective), colour says "isodata in 3D = k-means" in one slide with the
`colourseg` demo, and GMM recalls it and adds the objective, convergence and initialisation.
There is no cycle; only the "owner" wording in GMM needs adjusting.

**C17. Entropy.** Compression owns information and entropy. Forest uses entropy /
information gain, and measurements uses entropy in Haralick features; both are earlier than
compression in this order. **Fix:** forest uses Gini as its main criterion and gives entropy
one line ("expected surprise; full story on the compression day"). Measurements does the
same.

**C18. 3D images introduced twice.** Marching cubes unit 1 introduces stacks, spacing and
MIP. Graphics3d unit 4 owns the "full" treatment and opens with a three-slide intro so it
stands alone. **Fix:** marching cubes before graphics3d inside elective group G. Graphics3d
unit 4's intro becomes a recall when both run.

**C19. Shoelace / Green.** Shoelace area appears on measurements (core, convex hull),
marching cubes (unit 1, "2D divergence theorem") and moments (Green's theorem). **Fix:**
measurements states the shoelace formula (concept). Moments and marching cubes recall it and
own the Green / divergence-theorem math layer.

**C20. Colour ↔ bit depth (sRGB).** The built `gpuformats/` points forward to the colour day
for gamma. With bit depth on day 2 and colour on day 4 this stays a forward pointer, which
is acceptable (one slide, labelled). Colour unit 2 then recalls "you saw sRGB storage codes;
here is why they exist". The alternative (colour before bit depth) breaks thresholding's
16-bit bin choice and is not worth it.

**C21. Mutual pointers that are not real cycles.** Dynprog ↔ level sets (Dijkstra / fast
marching), Hough ↔ radon (same $(\rho,\theta)$), GMM ↔ VAE (ELBO), colour ↔ unmixing (shared
linear model), FFT registration ↔ integral (NCC denominators). Each plan already phrases
these as "names the link, neither re-teaches it". Keep them as they are; within an elective
group the order given in section 3 makes them back-references.

---

## 5. Concrete edits the ordering implies

Edits to plans (and, where noted, to built units). None were made in this review.

1. **New Part 1 unit `scale/`** (plan it, or extend `kernels/`): Gaussian σ and separability,
   gradient magnitude and direction, LoG/DoG blob detection at a scale, Gaussian pyramid,
   median filter. Add it to `_quarto.yml` Part 1. Update `fourier/exercise.qmd`'s "LoG comes
   back" promise to point there. (C1)
2. **`plan/morphology-day.md`**: take ownership of the distance transform (definition,
   by-hand city-block, two-pass chamfer in the math layer). Change "defined and computed on
   the watershed day" to "defined here". (C2)
3. **`plan/watershed-day.md`**: unit 4 recalls the DT from morphology and drops its by-hand
   two-pass exercise (or keeps a short one as a recall). Unit 1's thresholding, CC and
   h-minima recaps become back-references. Answer the open question: watershed before graph
   cuts. (C2, C14)
4. **`plan/labelling-day.md`**: unit 3 states explicitly that it is the course's PCA
   introduction (2D cloud, covariance, eigenvectors = axes, the word PCA), with the slide
   designed for reuse. (C3)
5. **`plan/snakes-day.md`**: unit 3 recalls PCA (labelling, `autoencoder/`) instead of
   introducing it. Answer its open question with that. Unit 1 adds the energy back-reference
   to `seg-energy/`. Change the Gaussian pyramid source to `scale/`. (C1, C3, C10)
6. **`plan/vae-day.md`** and the built `neuralnet/`: shrink `neuralnet/` to the 15-minute
   recall when the neural-net day is taught. `autoencoder/` keeps PCA as "many dimensions;
   2D on the labelling day". Use the freed hour as its open questions suggest. (C3, C4)
7. **`plan/svm-day.md`**: drop "introduces the vocabulary so other days can use it". Unit 1
   recalls the forest day's vocabulary and feature stack. Repoint LoG to `scale/`. (C5, C6)
8. **`plan/randomforest-day.md`**: confirm ownership of the ML vocabulary, feature stack and
   stump. Add a Hessian concept slide (instead of a pointer to the elective skeleton day).
   Repoint LoG/DoG to `scale/`. Make Gini the main split criterion, with entropy as one line
   pointing to compression. (C5–C8, C17)
9. **`plan/neuralnets-day.md`**: unit 2 recalls train/test and overfitting from the forest
   day and adds learning curves and regularisation. Unit 3 opens from "the forest's
   hand-made stack". (C5)
10. **`plan/integral-day.md`**: unit 4 recalls the stump from the forest day. Unit 3
    recalls the feature stack and adds the box group only. (C6, C8)
11. **`plan/thresholding-day.md`**: unit 3 computes local means as box filters (from
    `kernels/`), names the integral image as a pointer and makes by-hand (c) optional. The
    median comes from `scale/`, not `kernels/`. (C1, C12)
12. **Built `seg-energy/lec.qmd`**: reduce the "Thresholding" slide to a one-line recall of
    the thresholding day. Keep the noise-specks picture as the motivation. (C9)
13. **Built `quantisation/lec.qmd`** and **`plan/bitdepth-day.md`**: add PSNR as one line
    next to SNR, and claim the noise model. Then **`plan/wavelets-day.md`** (unit 4),
    **`plan/ising-day.md`** (unit 3/4) and **`plan/compression-day.md`** (unit 4: keep
    SSIM) recall it. (C11)
14. **`plan/skeleton-day.md`**: hysteresis pointer → thresholding. The DT pointer →
    morphology. Drop the "if labelling comes later" recap and answer its open question. (C2,
    C13, C15)
15. **`plan/hough-day.md`**: unit 1 recalls gradient direction and LoG from `scale/`. It
    keeps Canny/NMS (no core day needs it), so its open question about a separate
    edge-detection day can be closed as "not needed". (C1)
16. **`plan/gmm-day.md`**: say k-means was met as isodata (thresholding) and in Lab
    (colour); this day owns the algorithm. Its prerequisite list cites labelling and
    `autoencoder/` for PCA. (C3, C16)
17. **`plan/graphics3d-day.md`**: unit 4's three-slide 3D intro becomes conditional on
    whether marching cubes has been taught first. (C18)
18. **`plan/moments-day.md`** and **`plan/marchingcubes-day.md`**: recall the shoelace
    formula from measurements. (C19)
19. **`plan/measurements-day.md`**: the catalogue rows for moments, skeleton and unmixing are
    forward pointers to electives. Mark them "(elective day)" so the concept layer never
    depends on them. Entropy in Haralick gets one line. (C17)
20. **`plan/graphcut-day.md`** (and `graphcut-seg/` seeds slide): add the back-reference
    "markers on the watershed day, seeds here". Answer its open question on day length: the
    core order needs all four units, so `ncut/` stays core (it is the Laplacian's home for
    Ising, dynprog and registration). (C14)
21. **`_quarto.yml`**: reorder the parts to match section 3 when the new days are built:
    I Images as numbers (Part 1 + bit depth); II From pixels to objects; III Comparing and
    aligning; IV Segmentation as optimisation; V Learning from examples. Electives go in a
    separate sidebar section.
