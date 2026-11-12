# Review: what goes where (organisation of content across days)

Internal working doc (plan/ is not rendered). Status: review for discussion, 2026-11-12.
Scope: **which day/unit owns which content**, duplication, misplaced and overloaded units,
shared demo/lib infrastructure and notation. The **order of days is out of scope** (another
reviewer); where an ownership choice depends on order, this is said explicitly.

Inputs read: all 28 `plan/*-day.md` (there are 28, not ~33), the 15 built units in
`_quarto.yml` (headings of every `lec.qmd`, plus `kernels/`, `fourier/`, `images/`,
`seg-energy/`, `autoencoder/` in more detail), `audience.md`, `designprinciples.md`,
`demos/README.md`, and the existing code in `demos/src/` (`lib/*.ts` and per-demo modules).

---

## 0. Summary

The plans already negotiate most boundaries pairwise ("one-line recall, pointer"), and many
of those agreements are consistent. The real problems are where **three or more plans** each
claim to *introduce* something "so the day stands alone", and where **no plan** owns a topic
that several need:

1. **PCA** is taught in full in `autoencoder/` (built) and `snakes-day` unit 3, and each of
   six other plans says "introduced on another day". Make it **one standalone unit `pca/`**.
2. **Classification basics** (feature space, train/test, overfitting, cross-validation) are
   introduced from zero by the SVM, random-forest and neural-network plans, and partly by
   moments and measurements. Make **`classify/`** (SVM unit 1) the day-neutral owner.
3. **Gaussian derivatives, LoG/DoG, Hessian, structure tensor, Canny/NMS, scale space** have
   no owner: they sit in Hough unit 1, skeleton unit 4, wavelets optional unit 5, RF unit 4
   and SVM unit 4, and `fourier/` promises "the LoG comes back". Propose a **new day "Edges,
   blobs and ridges"**, which also relieves the two most overloaded units (Hough unit 3,
   skeleton unit 4).
4. **`neuralnet/` should move** to the neural-networks day as its unit 1 (keep the folder
   name), and the VAE part should start from `autoencoder/` with a recall.
5. The **bit-depth part** is coherent, but `gpuformats/` overlaps the 3D-graphics day
   (textures, mipmaps, alpha compositing) and the camera-noise half of `quantisation/` is the
   only place that defines a noise model, which five other plans need. Move `gpuformats/` to
   the graphics day; split `quantisation/` into `quantisation/` + `cameranoise/`.
   Content-wise the bit-depth part belongs with the Part 1 foundations, not after the VAEs.
6. Three-way overlaps to settle by moves: **3D images** (marching cubes unit 1 vs graphics
   unit 4) → a shared `volumes3d/`; **shape measurements** (labelling, measurements,
   moments) → move `shape-features/` to the labelling day and split measurements unit 4;
   **linear unmixing** (colour unit 4 vs unmixing day) → stain deconvolution goes to the
   unmixing day, and colour maps become core on the colour day.
7. Infrastructure: about 15 helpers are proposed under different names by several plans, and
   **three of them already exist twice or three times in the code** (eigen-solver, 2D point
   sets, image loading). One owner and one file name each, below.

Net effect on the number of days: +1 (the new edges day). If the calendar is fixed, the
first candidates to be compressed are the labelling and watershed days, whose own plans say
their units 1–2 can merge (§2.6).

---

## 1. Duplication table

"Teaches" means *introduces with its own slides/by-hand work*, not a one-line recall.
Recommended owner in **bold**; everyone else recalls in one line + pointer, and the recall
slide must be written so it also works as a forward pointer (order is not fixed).

| Concept | Days/units that teach it now | Recommended owner | Notes |
|---|---|---|---|
| **PCA** (covariance of a cloud, eigenvectors, scores, variance explained) | `autoencoder/` (built: two-pixel PCA, by hand, math slide); snakes `shape-models/` (2D cloud → $\mathbb R^{2n}$); GMM `gaussian-mixture/` (covariance ellipse); compression `dct/` (KLT); unmixing `endmembers/`; colour optional unit; labelling `regionprops/` (second-moment matrix) | **new standalone unit `pca/`** "Covariance and PCA: the shape of a point cloud", built from the `autoencoder/` PCA slides + `linae` demo and the snakes 2D-cloud slides. It is attached to whichever of {labelling, snakes, GMM, VAE} the order reviewer puts first | Every other plan already assumes "introduced elsewhere"; the snakes plan (open question) and compression plan ask for exactly this shared slide set. `autoencoder/` then keeps "PCA is a linear autoencoder" and the curve argument |
| Second-moment matrix of a region, orientation, eccentricity | labelling `regionprops/`; moments `moments-basis/`, `moment-invariants/` (recall); measurements `shape-features/` (recall) | **labelling `regionprops/`** | consistent in the plans; keep. One slide in `regionprops/`: "this is PCA of the pixel coordinates" (pointer to `pca/`) |
| **Distance transform** (definition, chamfer, exact EDT) | watershed `ws-distance/` (by hand); morphology unit 2 (definition + erosion link); skeleton `skel-medialaxis/` (definition); dynprog unit 1 (DP view); measurements (inscribed radius) | **watershed `ws-distance/`** | already agreed; morphology and skeleton must stay at "definition + picture", dynprog at one slide. No change needed beyond checking the wording |
| **Histogram**, normalised histogram as $p_i$ | thresholding `histogram/`; compression `entropy/` (histogram → entropy); `highbit/` (built, display aid); GMM (weighted points) | **thresholding `histogram/`** | `highbit/` and `entropy/` recall. Code: one `lib/histogram.ts` (compression's `lib/entropy.ts` must not keep its own histogram) |
| Contrast stretching, LUTs, colour maps for display | `highbit/` (built: slide "Colour maps, HDR and tone mapping"); colour optional `colour-display/`; levelsets/dynprog/radon each propose a map | **colour `colour-display/` (made core)** for colour-map design; **`highbit/`** for window/level of high-bit data | `highbit/`'s colour-map slide becomes a pointer |
| Gaussian blur, Sobel, Laplacian, LoG named | `kernels/` (built) | **`kernels/`** | consistent |
| **Gaussian derivatives at scale σ, LoG/DoG blob detection, scale space, Gaussian pyramid** | wavelets optional `cwt/` (Mexican hat, à trous spots); snakes (pyramid for ASM); SVM/RF unit 4 (as pictures); graphics3d/`gpuformats/` (mipmaps); `fourier/exercise.qmd` promises it | **new `scale-space/`** (new edges day, §2.1) | no current owner; three plans flag it as an open question |
| **Hessian**, eigenvalues for tube/blob, vesselness | skeleton `skel-neurovessel/` (core); RF unit 4, SVM unit 4 (feature pictures) | **new `hessian/`** (edges day); skeleton keeps vesselness as an *application* | relieves skeleton unit 4, which its plan calls the densest |
| **Structure tensor** (orientation, coherence, corners) | RF unit 4 ("no owner found: concept + math slide here") | **new `structure-tensor/`** (edges day), with Harris corners | also gives FFT registration/RANSAC a pointer for feature points |
| **Canny, non-maximum suppression** | Hough `canny/` ("compact, no other owner") | **new `canny/`** (edges day, moved from Hough) | Hough plan explicitly asks for this |
| **Hysteresis thresholding** | thresholding `local-threshold/` (idea + demo); labelling (algorithm = reconstruction + labelling); Hough unit 1 (recall) | **thresholding** (idea), labelling (implementation note) | consistent |
| **Connected components**, 4/8, flood fill, union–find | labelling units 1–2 | **labelling** | morphology, watershed, thresholding, skeleton recall; consistent |
| Morphological reconstruction, regional minima, **h-minima** | morphology `morph-reconstruct/`; watershed `ws-landscape/` ("by its effect") | **morphology** | consistent; the shared `recon1d` demo with `data-mode="hmin"` is the right mechanism |
| **Energy = data + smoothness**, the weight λ, "λ = 0 is thresholding" | `seg-energy/` (built); Ising `mrf-denoise/` (Bayesian posterior energy); levelsets `chanvese/`; snakes `snakes/`; radon `algebraic-recon/` (TV); wavelets `shrinkage/` (L1 view) | **`seg-energy/`** for the labelling energy; **Ising `ising/`+`mrf-denoise/`** for the probabilistic view (Gibbs, MAP vs MPM, sampling) | `seg-energy/` already has a "probabilistic view" math slide: Ising must recall it, not re-derive. Notation: §4 |
| Multi-label MRF / Potts / hidden MRF + EM | Ising optional `potts/`; GMM optional `hmrf-em/`; graph-cut optional `alpha-expansion/`; `graphcut-seg/` "Beyond: GrabCut and more labels" slide | **Ising `potts/`** (model + EM-style re-estimation); graph-cut `alpha-expansion/` (optimiser only) | merge GMM `hmrf-em/` into `potts/` (GMM plan asks "here, on the Ising day, or both?"); one demo |
| Dynamic programming on a chain (Viterbi) | `seg-energy/` (built, math slide + `segchain`); dynprog `dp-paths/` | **dynprog `dp-paths/`** | `seg-energy/` math slide stays as the forward pointer; dynprog reuses `segchain` (already planned) |
| Dijkstra / priority queue / fast marching | dynprog `livewire/` (Dijkstra); watershed `ws-flooding/` (PQ flooding); levelsets optional `fastmarching/` | dynprog for Dijkstra, watershed for flooding; **move `fastmarching/` to the dynprog day** (§2.7) | |
| **Graph Laplacian**, Fiedler vector | `ncut/` (built); Ising `gmrf/` ("defines it from scratch on a path"); FFT registration `stitching/` | **`ncut/`** | Ising `gmrf/` shrinks to a two-line recall |
| **Marching squares** | marching cubes `isocontour/` (by hand); level sets `implicit/` (by hand in the exercise per its timetable) | **marching cubes `isocontour/`** | level sets draws the zero contour with it and has one slide; drop the by-hand marching squares from its exercise |
| **3D images**: stacks, voxel spacing, world↔voxel affine, trilinear, slices, MIP | marching cubes `isocontour/` (intro); graphics3d `volrender/` (full) | **new shared `volumes3d/`** (§2.4) | graphics3d plan proposes the same split in its open questions |
| Shading (Lambert/Phong), normals, projection matrices | graphics3d units 1, 3; marching cubes `isosurface-render/` | **graphics3d** | marching cubes keeps only "normals from the image gradient" (graphics3d plan proposes this) |
| **Mipmaps**, texture filtering, premultiplied alpha / compositing | `gpuformats/` (built: mipmaps, memory, premultiplied alpha, compositing math); graphics3d `shading/` (texture filtering, mipmaps), `volrender/` (front-to-back compositing); integral optional `integral-more/` (mipmaps vs SAT) | **`gpuformats/` moved into the graphics day** (§2.3) | currently the same mipmap and alpha-compositing material is planned twice |
| **Spectral leakage / periodic extension / tapering windows** | compression `dct/` ("why not the DFT"); FFT registration `phasecorr/` (Hann); wavelets `haar/` (STFT windows) | **`fourier/`**: add one concept slide + one math slide | all three plans re-explain the periodic assumption because `fourier/` does not; one slide there makes them recalls |
| Continuous Fourier transform | radon `backprojection/` math slide ("`fourier/` has no continuous FT"); wavelets (inner product of functions) | **`fourier/`** math layer | radon plan asks exactly this |
| **Noise model** (Gaussian additive, Poisson shot, read noise), **SNR / PSNR in dB** | `quantisation/` (built: shot noise, read noise, ~6 dB/bit); wavelets `shrinkage/` ("PSNR not defined in Part 1", defines noise + PSNR); compression `jpeg/` (MSE/PSNR/SSIM); radon (Poisson sampler, RMSE); Ising (Gaussian noise); unmixing (Poisson, WLS) | **new `cameranoise/`** (split from `quantisation/`, §2.2) for noise models and SNR/PSNR; **compression `jpeg/`** for SSIM and "metrics can disagree" | wavelets and radon recall |
| **Segmentation/classification evaluation** (confusion matrix, Dice, Jaccard, ROC/PR, imbalance) | thresholding `seg-eval/`; NN `unet/` (Dice loss, recall); SVM, RF (reuse) | **thresholding `seg-eval/`** | consistent across plans |
| **Classification basics**: feature space, linear classifier (a threshold is one), nearest class mean, train/validation/test, overfitting, cross-validation | SVM `classify/` + `pixel-classify/`; RF `dtree/` ("introduces training/test data, overfitting and validation itself"); NN `backprop/`; moments `moment-features/` (LOO); measurements `features-use/`; `neuralnet/` (built, informally) | **SVM `classify/`, rewritten day-neutral** ("Classification in image analysis") | the RF, NN and SVM plans each say "if another ML day comes first, unit 1 shrinks to a recall": settle it now. Regulariser-specific material stays per day (weight decay/dropout: NN; OOB: RF; C: SVM) |
| **Pixel feature stack** + paint–train–look–correct loop | RF `pixelclass/`; SVM `pixel-classify/`; integral `rect-features/` | **RF `pixelclass/`** | RF plan proposes it; SVM keeps standardisation + CV by stroke; integral keeps "why large boxes are free" |
| Decision stumps / boosting | RF `dtree/` (stump); integral `viola-jones/` (AdaBoost) | stump: **RF**; AdaBoost/cascade: **integral** | consistent; only the code needs one split search (§3) |
| **k-means** (objective, Lloyd, k-means++) | GMM `kmeans/`; colour `colour-perceptual/` (k-means in Lab); thresholding isodata | **GMM `kmeans/`** | colour unit 3 drops its k-means part (time goes to colour maps) |
| Bayes decision for two classes, minimum-error threshold | thresholding `otsu/`; GMM (recall); Ising (recall); `seg-energy/` (built, "where the Gaussians cross") | **thresholding `otsu/`** | consistent |
| **Entropy** (information) | compression `entropy/`; RF `dtree/` (information gain); measurements (Haralick entropy); NN (cross-entropy) | **compression `entropy/`** | RF teaches Gini as the main criterion and entropy as a one-liner |
| **Beer–Lambert / line integral of attenuation** | radon `radon/` (derived); colour `colour-unmixing/` (optical density); graphics3d `volrender/` (emission–absorption ODE); unmixing (pointer) | **radon `radon/`** (absorption along a ray); graphics3d generalises; stain OD recalls | three derivations of the same two lines today |
| **Least squares, pseudo-inverse, condition number** | unmixing `lsunmix/` (full); colour unit 4 (condition-number slide); moments `orthogonal-moments/` (Gram matrix) | **unmixing `lsunmix/`** | recall elsewhere |
| Total variation | Ising `gmrf/` (1D); radon `algebraic-recon/` (2D, one line); wavelets (comparison) | **Ising `gmrf/`** | consistent |
| Procrustes | snakes `shape-models/` | **snakes** | consistent |
| NCC, template matching | FFT registration `matching/`; integral (fast denominators) | **FFT registration**; integral owns the summed-area trick | consistent |
| Polar / log-polar resampling | dynprog `polar-boundary/` (first use); FFT registration `fourier-mellin/`; radon (gridding); moments `zernike/`; measurements (radial bins) | concept: **`images/`** (inverse mapping), first worked use **dynprog `polar-boundary/`** | code: one `lib/polar.ts` |
| Bilinear / trilinear interpolation | `images/` (bilinear); marching cubes + graphics3d (trilinear) | `images/` / **`volumes3d/`** | |
| Local mean/variance via integral image | thresholding `local-threshold/` (by-hand item (c)); integral `integral-image/`, `box-filters/` | **integral** | thresholding keeps the recall slide (its plan already agrees) |
| Lifting / integer wavelets | wavelets optional `lifting/`; compression optional `compress-science/` (5/3 in JPEG 2000) | **wavelets `lifting/`** | compression one slide |
| DCT | compression `dct/`; wavelets (comparison) | **compression** | consistent |
| Radon ↔ Hough link | Hough optional unit 5 (slide); radon optional `radon-analysis/` | **radon `radon-analysis/`**; Hough one slide | consistent |
| Window/level (CT, HU) | `highbit/` (built); radon (one line) | **`highbit/`** | consistent |

---

## 2. Proposed moves, merges and splits

Ordered by impact. Each says what moves, from where to where, and why.

### 2.1 New day: "Edges, blobs and ridges" (differential image features)

- **Units:** `canny/` (moved from the Hough day, unchanged), `scale-space/` (Gaussian
  derivatives at scale σ, "derivative of Gaussian = Gaussian of derivative", LoG/DoG and
  scale-normalised blob detection, Gaussian pyramid; spot detection from wavelets `cwt/`
  goes here, à trous stays on the wavelets day as a pointer), `hessian/` (Hessian as second
  derivatives at scale σ, eigenvalues → blob / ridge / background, Frangi vesselness as the
  worked filter), `structure-tensor/` (gradient outer product, orientation and coherence,
  Harris/Förstner corners).
- **Why:** none of these has an owner; five plans use them and each recalls them "with a
  pointer to the day that owns them", which does not exist (RF: "structure tensor, no owner
  found"; SVM: "pointer to wavelets `atrous`/LoG, skeleton second derivatives"; Hough: "no
  plan owns a full edge-detection pipeline"; wavelets: "is unit 5 the place for the LoG
  that `fourier/exercise.qmd` promises?"). It is pure calculus + symmetric 2×2 eigenvalues,
  which fits the audience; it is the natural home of the feature-stack channels.
- **Knock-on:** Hough day: unit 1 becomes a recall; split dense unit 3 into
  `hough-circles/` and `ght/` (its plan proposes exactly this if `canny/` leaves).
  Skeleton day: unit 4 keeps vesselness as an application and the neuron/vessel
  measurements (its plan says unit 4 is the densest; this is the split it asks for, without
  losing the applications). RF/SVM: feature channels recalled as pictures.
- **Fallback if no new day is possible:** put `scale-space/` + `hessian/` + `structure-tensor/`
  as one unit `gaussian-derivatives/` right after `kernels/` (it is a follow-up of
  `kernels/`'s last slide "Laplacian of Gaussian … scale of the structures"), and leave
  `canny/` on the Hough day.

### 2.2 Bit depth (built, Part 4): fits as a foundations topic; two moves

- **Grouping.** Thematically this part is a **foundation** (how pixel values are stored and
  shown), and later plans depend on it: thresholding (16-bit bins), integral (overflow, float
  sums), measurements (intensity units, saturation), radon (HU), FFT registration (float,
  ε), unmixing (offset, saturation, Poisson). It belongs in the same group as `images/`,
  `kernels/`, `fourier/`, not as a late part after the VAEs. (Where exactly is the order
  reviewer's call; the dependency is one-directional: it needs only `images/`.)
- **Split `quantisation/` into `quantisation/` + `cameranoise/`** (principle 6: split, don't
  cut). The built deck already has two halves: rounding/dithering, then "Where the numbers
  come from / Light itself is noisy / How many bits are meaningful / adding the noise
  sources" with the `camerabits` demo. The second half becomes `cameranoise/`, extended
  with one slide on additive Gaussian noise as the usual model and SNR/PSNR in dB, and is
  the **course owner of noise models** (the bit-depth plan's own open question 2 asks for
  this). Wavelets `shrinkage/`, radon, Ising, unmixing then recall instead of defining.
- **Move `gpuformats/` to the 3D-graphics day** as its "textures" unit. Its slides on
  sRGB storage, filtering in hardware, premultiplied alpha + compositing math, and mipmaps +
  memory are exactly what graphics3d `shading/` (texture filtering, mipmaps) and
  `volrender/` (alpha compositing) plan to teach again. The row/stride/endianness part is
  generally useful; keep 2–3 of those slides as a short section in `highbit/` ("an image is
  a row of bytes") so the bit-depth day still has it. The bit-depth plan already asks
  whether `gpuformats/` is core; on this course it is core only for the graphics day.
- Result: bit-depth day = `quantisation/`, `cameranoise/`, `highbit/`, `floatingpoint/`.
- `highbit/` slide "Colour maps, HDR and tone mapping" → one-line pointer to colour
  `colour-display/` (keep tone mapping, which no other plan owns).

### 2.3 3D graphics day: merge 2+3, take in `gpuformats/`, give away "volumes as data"

- Merge `rasterise/` and `shading/` into one unit (rasterisation, z-buffer, Lambert/Phong,
  inverse-transpose normals; its own plan recommends keeping these derivations and cutting
  breadth). Texture filtering/mipmaps move to `gpuformats/` (above).
- Unit 4: keep `volrender/` (MIP/MinIP/average as rendering, DVR, emission–absorption,
  compositing); the "volume = array + affine, oblique reslicing" part goes to `volumes3d/`.
- Result: `pipeline3d/`, `rasterise/` (merged), `gpuformats/`, `volrender/`; optional
  `voxelise/`. The polygon side drops from three units to two, which answers the plan's
  own "is the polygon side worth three units?".

### 2.4 Shared `volumes3d/` unit; marching cubes day shrinks its rendering unit

- **`volumes3d/`** "3D images": stacks, voxel spacing and anisotropy, world↔voxel affine,
  trilinear interpolation, orthogonal and oblique slices, a first MIP. Owner: the **marching
  cubes day** (as its unit 1, since it is the image-analysis day of the pair and the one that
  first needs volumes); graphics3d `volrender/` opens with a recall. Also the pointer target
  for 3D in labelling, morphology, skeleton, FFT registration and measurements.
- Marching cubes `isocontour/` keeps marching squares, the asymptotic decider and shoelace
  area only.
- Fold `isosurface-render/` (marching cubes unit 3) into `marchingcubes/` (gradient normals,
  2–3 slides) and `surface-measure/` (anisotropic spacing); both plans propose this.
- Result: `volumes3d/`, `isocontour/`, `marchingcubes/`, `surface-measure/`; optional
  `surfacenets/`.

### 2.5 `neuralnet/` moves to the neural-networks day; VAE part starts at `autoencoder/`

- The NN plan's unit 1 `neurons/` is planned to **absorb** `neuralnet/` (by-hand pass,
  collapse, hinges figure, `nnfit`). Do it the other way round: **keep the built folder
  `neuralnet/` as NN unit 1** (the sidebar rule says never rename a folder) and extend it
  with the neuron as a linear classifier, XOR and `nnplayground`. Drop `neurons/` as a folder
  name.
- VAE part: `autoencoder/` gets a 2–3 slide recall at the start ("a layer: matrix, then
  bend; training = walk downhill; see *Neural networks*"). The freed hour goes to the
  VAE plan's open items (β = 4 weight set, real nucleus crops).
- Sidebar: the `neuralnet/` block moves out of "Part 3: Variational autoencoders" into the
  neural-networks part when that part is created.
- Both units share the same one-weight gradient-step numbers (already proposed).

### 2.6 Labelling / measurements / moments: re-cut the three "measure an object" days

- **Labelling:** merge units 1–2 into one (`label-connectivity/` absorbs union–find; its own
  plan calls them the lighter ones and offers the merge). Take in **measurements
  `shape-features/`** (convex hull, solidity, Feret by rotating calipers, form factor,
  equivalent diameter, inscribed radius), which sits naturally next to `regionprops/`
  (perimeter bias → form-factor bias is then taught in one place). Result: `label-connectivity/`,
  `regionprops/`, `shape-features/`, `label-measure/`.
- **Measurements:** split its overloaded unit 4 (its plan: "bundles neighbours and
  colocalisation and could split") into `relations/` (neighbours, parent–child) and
  `coloc/` (Pearson, Manders, Costes, bleed-through pointer to unmixing). Dissolve optional
  `features-use/`: units/normalisation/aggregation become the closing section of
  `intensity-features/`; classification of feature tables is RF `forest-objects/`.
  Result: `intensity-features/`, `texture-features/`, `relations/`, `coloc/`. The CellProfiler
  vs blockflow table stays teacher-only (decided).
- **Moments:** unchanged; units 1–2 may merge if `regionprops/` was seen (its plan says so).
  `moments-basis/` must recall orders 0–2 from `regionprops/`, not re-derive them.

### 2.7 Dynamic programming and level sets

- Make **seam carving optional** on the dynprog day (its plan doubts it for an analysis
  course) and **move level sets' optional `fastmarching/` to the dynprog day** as its fourth
  core unit "From Dijkstra to fast marching: continuous minimal paths" (both plans already
  say "fast marching is Dijkstra with an upwind update"). This gives the dynprog day a
  coherent "minimal paths" arc (grid DP → polar DP → Dijkstra/live-wire → fast marching)
  and removes an optional unit from the level-set day, whose own plan wonders whether the
  optional slot should be 3D instead.
- Level sets: drop marching squares by hand from `implicit/` (owned by marching cubes);
  if unit 2 stays too dense, split it into `levelset-eq/` and `levelset-num/` as its plan
  proposes, and shorten `geodesic/`.

### 2.8 Colour and unmixing: one owner of linear unmixing

- Move colour unit 4's **stain deconvolution** (Beer–Lambert OD, 3×3 stain matrix, `stainsep`,
  Macenko) to the **unmixing day** as an applied unit (brightfield = log, then the same
  model; Macenko is vertex finding, already that day's unit 4 topic). To make room, merge
  unmixing `unmixgeom/` into `endmembers/` (its plan offers this).
  Result: `crosstalk/`, `lsunmix/`, `endmembers/` (with cone/simplex), `stain-unmixing/`
  (moved, keeps the `colour-unmixing/` folder name if it is built first); optional `phasor/`.
- Colour day: white balance moves into `rgb-linear/` (it is a diagonal matrix in linear RGB;
  colour plan proposes it); `colour-perceptual/` drops k-means (GMM owns it) and keeps
  thresholds on hue/$a^*b^*$, ΔE balls; **`colour-display/` becomes core** (every demo in
  the course uses its colour maps, and the colour plan itself says the "optional" label is
  wrong). Result: `colour-vision/`, `rgb-linear/`, `colour-perceptual/`, `colour-display/`.
- If the two plan authors prefer to keep stains on the colour day, the minimum is: the
  condition-number and least-squares material is taught only on the unmixing day.

### 2.9 Smaller moves

- **GMM optional `hmrf-em/`** → merged into Ising `potts/` (one demo, one owner).
- **Compression optional `compress-science/`** → core (its plan: "for microscopy/medical
  students the most important message of the day"); trim `lossless/` (LZ77/LZW to one slide,
  drop `lzstep`), as the plan itself suggests.
- **`seg-energy/` (built)** slide "Thresholding" → one-line recall once the thresholding day
  exists (thresholding plan lists this).
- **Ising `gmrf/`**: graph Laplacian as recall of `ncut/`.
- **FFT registration `stitching/`**: dense; if needed split off `subpixel/` (its plan proposes
  this) rather than cutting drift/mosaic.
- **SVM `pixel-classify/`**: becomes SVM-specific (standardisation, CV by stroke, multi-class)
  on the shared `pixelclass` shell; this frees time, so optional `detect-window/` (HOG) can
  become core if wanted.
- **Snakes day** is two topics (explicit snakes; statistical shape models). Keep it as one
  day "deformable models", but with `pca/` taught elsewhere, `shape-models/` loses its PCA
  introduction and gains time for Procrustes and failure modes.
- **Integral day** is also two topics (summed-area tables; boosted detection). Keep;
  `viola-jones/` stays concept-level and recalls the stump from RF `dtree/`.

### 2.10 Optional ↔ core, summary

| Unit | Now | Proposed |
|---|---|---|
| colour `colour-display/` | optional | **core** |
| compression `compress-science/` | optional | **core** (trim `lossless/`) |
| dynprog `seamcarve/` | core | optional |
| levelsets `fastmarching/` | optional (levelsets) | **core on dynprog day** |
| GMM `hmrf-em/` | optional | merged into Ising `potts/` |
| measurements `features-use/` | optional | dissolved (§2.6) |
| `gpuformats/` | core (bit depth) | core on graphics day |
| wavelets `cwt/` | optional | LoG/spot detection → `scale-space/`; scalogram stays optional |
| graphcut `ncut/` | core (built) | core (keep) |
| Ising `gmrf/` vs `potts/` | gmrf core, potts optional | keep (Ising plan's recommendation) |

### 2.11 Gaps noticed (no plan owns them; for the course owner to decide, not moved here)

Edge detection beyond Canny (now §2.1); feature points/descriptors and non-rigid
registration/optical flow (FFT registration plan: "no other plan covers it"); contour
(Fourier) descriptors (moments plan); deconvolution / PSF (no plan, though microscopy is the
main application domain); classical non-linear denoising (median, bilateral, non-local means;
only the median is named in `kernels/`).

---

## 3. Shared infrastructure: one name, one owner

Owner = the day that writes it first and keeps its tests; others import and extend through
that owner. "Exists" marks code already in `demos/src/`.

### 3.1 Library modules

| Thing | Names used in plans / code now | Proposed file | Owner | Users |
|---|---|---|---|---|
| Symmetric eigen-solver (Jacobi), 2×2 closed form, SVD 2×2, Cholesky, solve, condition number, null space | **exists 3×**: `ncut/spectral.ts` `jacobiEigen`, `matrix2x2/geom.ts` `eigenDirections`, `linae/data.ts` (PCA angle); planned `lib/linalg.ts` (snakes), "2×2 eigen in `lib/linalg.ts` or `lib/imageops.ts`" (labelling, skeleton, moments, RF) | `lib/linalg.ts` (2×2 closed form here too, **not** in `imageops.ts`) | graph-cut day (move `jacobiEigen` now) | snakes, colour, GMM, unmixing, compression (64×64 KLT), labelling, skeleton, edges day, SVM, moments, VAE |
| FFT 1D/2D (+ rectangular, 3D, real helpers) | **exists**: `fourier2d/fft.ts` (also holds `rmse`, `makeMask`); four plans propose `lib/fft.ts` | `lib/fft.ts` (move `rmse` to `lib/metrics.ts`, masks stay in `fourier2d/`) | `fourier/` | compression (`lib/dct.ts`), FFT registration, radon, wavelets, moments |
| Image loading (`cell256.png` → grey array), display helpers | **exists 6×**: `kernel3x3`, `camerabits`, `ncut`, `fourier2d`, `quantise`, `graphcut` each load it | `lib/imageops.ts` `loadGrey()` | first plan built (watershed/morphology) | all |
| Linear filters (Gaussian, Sobel, Laplacian, LoG/DoG, Gaussian derivatives, Hessian, structure tensor), bilinear sampling | **exists**: `kernel3x3/conv.ts` (3×3 only); `lib/imageops.ts` "decided 2026-11-12" | `lib/imageops.ts`; Hessian/structure tensor/NMS in `lib/edges.ts` if it grows | `kernels/` (filters), edges day (derivatives) | nearly all |
| Pixel feature stack | `featureStack()` in `imageops.ts` (SVM, RF); `lib/pixelfeatures.ts` rejected | `featureStack(img, groups, sigmas)` in `lib/imageops.ts` | RF day | SVM, integral (box group), NN (comparison) |
| Seeded synthetic generators with ground truth | `imageops.ts` (nuclei, spots on uneven background, discs, mixed population, phantoms, kymograph, seam scene, cartoon faces), `lib/shapes.ts`, `lib/trees.ts`, `lib/volume.ts`, `window16/data.ts` (`fluorescence`, `ctPhantom` **exist**), `quantise/quant.ts` (`ramp`, `blobs` **exist**) | `lib/synth.ts` for 2D image generators (incl. Shepp–Logan and the 12-bit generator from `window16`); keep `lib/shapes.ts` (landmark shapes), `lib/trees.ts` (branching graphs), `lib/volume.ts` (3D) | thresholding (first big user) | all segmentation and measurement days |
| Random numbers | `rng()` in `lib/ui.ts` (**exists**); `gauss()` (GMM), Poisson sampler (radon; **exists** in `camerabits/sensor.ts`), unmixing Poisson | `lib/random.ts` (`rng`, `gauss`, `poisson`); `ui.ts` re-exports `rng` | bit-depth day (move `poisson`) | Ising (speed!), GMM, radon, unmixing, cameranoise |
| Plotting (lines, bars, scatter, histograms, heat map with numbers, covariance ellipse, polar plot) | `lib/plot.ts` (morphology) "or in `lib/ui.ts`"; heat map (measurements, Hough, snakes "matrix heat map"); ellipse helper (GMM) | `lib/plot.ts` (never `ui.ts`); histogram drawer in `lib/histogram.ts` on top of it | morphology day | ~20 demos |
| Colour maps (sequential, diverging, qualitative, label palette) | `lib/colour.ts` (colour); diverging map "in `lib/ui.ts` or `lib/colour.ts`" (levelsets); sequential (dynprog); `labelColors` in `imageops.ts` (labelling, watershed) | all maps incl. label palette in `lib/colour.ts` (`colormaps` section) | colour day (`colour-display/`) | levelsets, dynprog, radon, wavelets, integral, unmixing, labelling |
| Histogram | `lib/histogram.ts` (thresholding); `lib/entropy.ts` "histogram of integer arrays incl. negative and 16-bit" (compression) | `lib/histogram.ts` (support signed/16-bit); `entropy.ts` imports it | thresholding | compression, GMM, colour (2D histogram), measurements, marching cubes |
| Metrics | `lib/metrics.ts` (compression: MSE/PSNR/SSIM); evaluation in `lib/threshold.ts` (Dice, ROC); `rmse` in `fft.ts` (**exists**) | `lib/metrics.ts` with two sections: image quality, segmentation/classification; `threshold.ts` keeps thresholds only | thresholding (evaluation) + compression (quality) | NN, SVM, RF, radon, wavelets, colour |
| Integral image, box mean/variance | `lib/integral.ts` (integral) vs "in `lib/imageops.ts`" (thresholding) | `lib/integral.ts`; `imageops.ts` re-exports box mean/variance | integral day | thresholding, FFT registration (NCC), RF/SVM stack |
| k-means / isodata | `lib/kmeans.ts` (GMM); isodata in `threshold.ts`; k-means in Lab via `threshold.ts` (colour) | `lib/kmeans.ts` (weighted points) | GMM | thresholding, colour |
| Grid neighbourhoods 4/8, 6/18/26 | **exists**: `lib/graph.ts` `gridEdges`; planned offsets in `label.ts`, neighbour iteration in `watershed.ts`, `neighbours.ts` (measurements) | offset tables in `lib/graph.ts` (rename section "grid"), imported by all | graph-cut day | labelling, watershed, morphology, dynprog, measurements |
| Priority queue / FIFO | `lib/pqueue.ts` (watershed); FIFO ring in `morph.ts` | `lib/pqueue.ts` (heap + bucket + FIFO) | watershed | dynprog, fast marching, morphology |
| Polar / log-polar resampling | `lib/polar.ts` "or in `imageops.ts`" (dynprog); radon, FFT registration, moments, measurements | `lib/polar.ts` | dynprog | radon, FFT registration, moments, measurements |
| Marching squares | `lib/contour.ts` (levelsets) **and** `lib/marching.ts` (marching cubes) | `lib/marching.ts` only | marching cubes | levelsets (zero contour), any demo that draws iso-lines |
| DP / chains | `segchain/dp.ts` (**exists**), `lib/dp.ts` (dynprog), forward–backward in `lib/smooth1d.ts` (Ising) | `lib/dp.ts` (monotone paths + chain Viterbi/forward–backward); `segchain` may keep its file | dynprog | graph-cut, Ising |
| Smoothing priors (tridiagonal GMRF, 1D/2D TV, Huber) | `lib/smooth1d.ts` (Ising); "2D TV routine if Ising/levelsets add one" (radon) | `lib/smooth.ts` (1D and 2D) | Ising | radon, wavelets comparison |
| Classifiers | `lib/classifier.ts` interface (RF); `svm.ts`, `svmkernels.ts`, `forest.ts`, `boost.ts`, `nn.ts` (**exists**), `cnn.ts`, `autodiff.ts`, `q8.ts` | as listed; **one** weighted split search in `forest.ts` used by `boost.ts` | RF (interface) | SVM, NN, integral, moments (`momentclass`) |
| 2D point-cloud datasets | **exists**: `ncut/data.ts` `POINT_PRESETS`; `lib/datasets2d.ts` (GMM); `lib/playground.ts` (RF) | `lib/datasets2d.ts` (move `ncut` presets in) + `lib/playground.ts` | `classify/` (SVM unit 1, the owner of feature spaces) | GMM, RF, SVM, NN, ncut |
| WebGL renderer | `lib/gl3d.ts` + `mat4.ts` (marching cubes), camera/shading modes/multi-viewport (graphics3d), `lib/volrender.ts` (graphics3d), `lib/surface.ts` (levelsets), point-cloud mode (colour, unmixing), height map (FFT registration) | `lib/gl3d.ts`, `lib/mat4.ts`, `lib/volrender.ts`; point clouds and height fields as `gl3d.ts` primitives (no separate `surface.ts`) | **graphics3d day** (rendering is its subject); marching cubes owns `marching.ts`, `mesh.ts`, `volume.ts` | colour, unmixing, SVM (`svmlift`), levelsets, moments, Hough |
| Label painting on an image | **exists**: seed painting in `graphcut/main.ts`; `lib/paint.ts` (RF) | `lib/paint.ts` (extract from `graphcut`) | RF | graph cut, pixelclass, (U-Net demo if interactive) |
| Shape editing canvas | `lib/shapecanvas.ts` (moments); `moments` demo (labelling); `shapefeat` (measurements); `procrustes`/`snake` dragging | `lib/shapecanvas.ts` | labelling (owner of `moments` demo) | moments, measurements, snakes |
| Grid step-through UI | `morphgrid` (+`label`, `hitmiss` modes), `lib/gridui.ts` (labelling, open question), `floodgrid`, `dpgrid`, `satstep`, `raster`, `pixelgraph` | `lib/gridui.ts` (cells ≥ 24 px, step/back/play, value overlay) | morphology | labelling, skeleton, watershed, dynprog, integral, graphics3d |
| Convex hull, calipers | `lib/convexhull.ts` (measurements) | same | labelling (if `shape-features/` moves) | skeleton, snakes |
| Radon projector pair | `lib/radon.ts` | same | radon | Hough (one slide) |
| Tests | `demos/test/bitdepth-test.ts`, `vae-check.ts`, `nn-test.ts`, `compression-test.ts`, "`make test`" in every plan, `demos/test/test.ts` | one runner `make test` over `demos/test/*-test.ts` | graph-cut day (first proposed) | all |

### 3.2 Demo shells proposed under different names (merge into one shell + `data-mode`)

| One shell | Replaces / serves | Owner |
|---|---|---|
| `pixelclass` (`data-classifier`, `data-stack`) | RF `pixelclass`, SVM `pixelsvm` | RF |
| 2D point playground (`lib/playground.ts`) | `svm2d`, `linclass`, `kmeans2d`, `gauss2d`, `em2d`, `treeplay`, `forestplay`, `nnplayground`, ncut point mode | `classify/` |
| `moments` + `shapecanvas` | labelling `moments`, moments modes `higher`/`invariants`, measurements `shapefeat`, `momentphase`, `momentclass` | labelling |
| `morphgrid` on `lib/gridui.ts` | morphology, `label` mode, `hitmiss` mode; same UI for `floodgrid`, `dpgrid` | morphology |
| 3D point cloud viewer | colour `colourcloud`, unmixing `unmixcloud`, SVM `svmlift` | graphics3d (`gl3d.ts`) |
| `volview` | graphics3d, marching cubes MIP mode (dropped), FFT registration 3D tab, radon 3D (if any) | graphics3d |
| Feature table + scatter | labelling `regionfilter`, measurements `featurespace`, RF `featimp`, moments `momentclass` | labelling (`regionfilter`) |
| Histogram + threshold slider | `histthresh`, `isocontour`, `tophat`, `distws`, `regionfilter`, `chanvese` markers, `histgmm`, `mixture` | thresholding (`lib/histogram.ts` drawer) |
| 1D signal step-through | `segchain`, `flood1d`, `recon1d`, `morph1d`, `mrf1d`, `xcorr1d`, `haarstep`, `dtw` | shared `lib/plot.ts` line + step controls; not one demo, one helper |

---

## 4. Conventions and notation (agree once, course-wide)

| Item | Current usage | Proposal |
|---|---|---|
| Weight of the prior/smoothness/regularisation term vs the data term | `seg-energy/` (built) $\lambda$; Ising $J$ (and "$\lambda = 2J$"); Chan–Vese $\mu$ (length), $\lambda_1,\lambda_2$ (data); snakes $\alpha,\beta$ (elasticity, rigidity); wavelets $\lambda$ (L1); radon TV weight; NN weight decay; SVM $C$; VAE $\beta$; Geman & Geman $\beta$ | **$\lambda$ everywhere** for "weight of the prior relative to the data term". Ising keeps $J$ on physics slides with the bridge slide $\lambda = 2J$. Chan–Vese: $\lambda$ for length, data weights 1 in the concept layer, literature $\mu$ named once. SVM: $C$, with "$C \sim 1/\lambda$" said once. **$\beta$ only** for snake rigidity and β-VAE; never for prior strength or inverse temperature (Ising plan agrees) |
| "kernel" | `kernels/` convolution kernel; SVM kernel function; "compute kernel" in `demos/README.md` (Wasm) | convolution: **kernel**; SVM: **kernel function $k(\mathbf x, \mathbf z)$**, folder `kernel-trick/`, file `lib/svmkernels.ts`; README: "computation"/"inner loop" in student-facing text |
| "hinge" | `neuralnet/`: ReLU unit; SVM: hinge loss | keep both and say "the same bend" (SVM plan) |
| "window" | `highbit/` display window/level; FFT registration Hann window; wavelets STFT window; local threshold sliding window; `fourier/` "cycles per window" | **display window** (window/level), **taper** (Hann, Tukey), **sliding window**; `fourier/` keeps "window" for the signal interval |
| "depth" | bit depth; depth buffer / NDC z; stack depth; tree depth; network depth | always **bit depth** in full; $z$ for volumes; "depth" alone only for trees/networks |
| Wavelet subband names | wavelets: "pick LH vs HL course-wide"; compression and integral refer to subbands | concept layer: **LL, H (horizontal detail), V (vertical detail), D (diagonal)** with a picture; math layer defines letters once (first letter = filter along $x$). Same in compression and integral |
| Image axes and angles | `images/` $I[i,j]$, row $i$, column $j$; measurements: "orientation in degrees, image $y$ down"; Hough/radon: origin at centre, $\theta \in [0°, 180°)$, signed $\rho$; graphics3d: right-handed view space looking down $-z$, NDC $z \in [-1, 1]$ | one slide in `images/` (and a constant in `lib/imageops.ts`): pixel $(i, j)$ = (row, column), $x$ right, $y$ **down**, angles counter-clockwise **as seen on screen**; volumes stored $(z, y, x)$ with spacing $(\Delta x, \Delta y, \Delta z)$; Hough/radon convention as agreed; graphics uses OpenGL/WebGL conventions (forced by WebGL) |
| Images vs functions | most: $I$; moments, morphology: $f$ | $I$ for images; $f$ only for a generic function in math layers |
| Covariance matrix | `autoencoder/` (built) $S$; snakes $S$; labelling $C$; GMM $\Sigma$; measurements "matrix $C$"; `neuralnet/` uses $C$ for the second weight matrix | **$S$** for a sample covariance (data, region, landmarks), **$\Sigma$** for a model parameter (GMM, prior); avoid $C$ (clashes with `neuralnet/` and SVM $C$) |
| Eigenvalue order | PCA descending $\lambda_1 \ge \lambda_2$ (`autoencoder/`); Laplacian ascending (Fiedler = 2nd smallest, `ncut/`); Hessian by magnitude $|\lambda_1| \le |\lambda_2|$ (Frangi) | keep all three, but **state the order on the slide** each time; PCA/covariance always descending |
| $\lambda$ overload | eigenvalue; prior weight; wavelength (colour, unmixing); Poisson mean (`camerabits` code) | eigenvalues $\lambda_i$ (with index), prior weight $\lambda$ (no index), wavelength $\lambda$ only on colour/unmixing days (which have no prior weight); Poisson mean $N$ or $\mu$ on slides |
| Labels and spins | `seg-energy/` $L_p \in \{0,1\}$, $\ell$ generic label, $D_p(\ell)$ data term, $w_{pq}$; Ising $s_p \in \{-1,+1\}$; SVM $y \in \{-1,+1\}$; NN/RF $y \in \{0..K{-}1\}$ | keep `seg-energy/`'s $L_p$, $\ell$, $D_p$, $w_{pq}$ for all segmentation days (Ising, levelsets, GMM data term, NN/RF "data term" slides); $s_p$ only on Ising slides; $y \in \{-1,+1\}$ only on the SVM day |
| Distance transform symbol | morphology $D(p)$; watershed, skeleton, measurements | **$\mathrm{DT}(p)$** or $d(p)$: $D_p$ is the data term |
| Laplacian symbol | `ncut/` $L$; `seg-energy/` labels $L$; $\nabla^2$ in `kernels/` | graph Laplacian $L$ only on ncut/Ising/stitching slides (with $\mathbf L$ never used for labels there); image Laplacian $\nabla^2$ |
| Loss / likelihood | VAE/NN "loss", $L(w)$ in `neuralnet/`; GMM "$\log L$" | loss $\mathcal L$, log-likelihood $\log p(\cdot)$ |
| Noise and scale | `seg-energy/` noise std $s$; elsewhere $\sigma$ for noise, Gaussian scale, mixture width, sigmoid | $\sigma$ = Gaussian (filter) scale; $\sigma_n$ = noise std (`cameranoise/` defines); keep `seg-energy/`'s $s$ = class width (built); activation $\varphi$, never $\sigma$ (NN plan's open question) |
| $T$, $t$ | Ising temperature $T$; graphics3d transmittance $T$; threshold $t$; level-set time $t$; snakes "keep $t$ modes" | $T$ temperature; $\mathcal T$ (or "transmittance" spelled out) in graphics; $t$ threshold; time $t$ only on level-set slides; snakes: **$k$ modes** (as in `autoencoder/`) |
| Neural network notation | `neuralnet/` $\mathbf h = \mathrm{ReLU}(A\mathbf x + \mathbf b)$, $C\mathbf h + d$; NN day $W^{(l)}$, $\varphi$ | NN day uses $W^{(l)}$; one bridge line "$A = W^{(1)}, C = W^{(2)}$" in the moved `neuralnet/` (already proposed) |
| Mean shape / mean image | $\bar{\mathbf x}$ (snakes, `autoencoder/`) | keep (already agreed by both plans) |
| Connectivity | foreground 8, background 4 (morphology, labelling, watershed, thresholding) | keep (already agreed) |
| Fourier normalisation and display | `fourier/`: unnormalised forward DFT ($X_2 = 4$ for 8 samples), DC in the centre; compression: orthonormal DCT | state on one slide in `fourier/` and once in compression `dct/` ("orthonormal so Parseval holds") |
| Gaussian affinity / RBF | graph-cut `contrastWeights(sigma)` $e^{-(I_p-I_q)^2/2\sigma^2}$; SVM $\gamma$ | one RBF function; slides say $\gamma = 1/(2\sigma^2)$ once |
| Voxel spacing | measurements $(s_x, s_y)$; marching cubes $(\Delta x, \Delta y, \Delta z)$ | $(\Delta x, \Delta y, \Delta z)$ |
| Physical units of intensities, PSNR peak | compression: peak argument for 8/12/16 bit | defined once in `cameranoise/`; `lib/metrics.ts` takes the peak explicitly |

Proposal: keep this table as the single reference (e.g. a future `plan/conventions.md`, not
created here) and add a "Notation" line to each plan that deviates.

---

## 5. Prioritised list of concrete edits

**P1: decide first (they change unit lists of several plans)**

1. `plan/neuralnets-day.md`: unit 1 becomes the existing `neuralnet/` (extended), drop the
   `neurons/` folder name. `plan/vae-day.md`: mark `neuralnet/` as moved; add the 2–3 recall
   slides to `autoencoder/lec.qmd` and keep only the parameter-count question in
   `autoencoder/exercise.qmd`. `_quarto.yml`: move the `neuralnet/` sidebar block when the NN
   part exists.
2. New plan `plan/edges-day.md` (`canny/`, `scale-space/`, `hessian/`, `structure-tensor/`);
   then `plan/hough-day.md` (unit 1 → recall; split unit 3 into `hough-circles/` + `ght/`),
   `plan/skeleton-day.md` (unit 4: vesselness as application, Hessian recalled),
   `plan/wavelets-day.md` (`cwt/`: LoG spot detection → pointer), `plan/randomforest-day.md`
   and `plan/svm-day.md` (structure tensor / Hessian / LoG as recalls).
3. New standalone `pca/` unit (source: `autoencoder/` PCA slides + `linae`, snakes unit 3's
   2D-cloud part). Edit `plan/snakes-day.md` unit 3, `plan/gmm-day.md` unit 2,
   `plan/compression-day.md`, `plan/unmixing-day.md`, `plan/colour-day.md`,
   `plan/labelling-day.md` (`regionprops/` pointer) to recall it; `autoencoder/lec.qmd` "About
   this lecture" note updated to point to `pca/`.
4. `plan/svm-day.md` unit 1 `classify/` rewritten as the day-neutral owner of feature space,
   train/validation/test, overfitting, cross-validation; `plan/randomforest-day.md` unit 1,
   `plan/neuralnets-day.md` unit 2, `plan/moments-day.md` unit 5,
   `plan/measurements-day.md` unit 5 recall it.
5. Bit depth: split `quantisation/` into `quantisation/` + `cameranoise/` (built deck,
   exercise, quiz, `camerabits` embed; add Gaussian noise model and PSNR slide);
   `plan/bitdepth-day.md` updated; move `gpuformats/` to `plan/graphics3d-day.md`
   (and its sidebar block); move 2–3 stride/endianness slides into `highbit/`.
   `plan/wavelets-day.md`, `plan/radon-day.md`, `plan/ising-day.md`,
   `plan/unmixing-day.md`: noise model and PSNR become recalls of `cameranoise/`.
6. 3D: `plan/marchingcubes-day.md` gets `volumes3d/` as unit 1, `isocontour/` shrinks,
   `isosurface-render/` is folded; `plan/graphics3d-day.md` merges `rasterise/`+`shading/`,
   takes `gpuformats/`, and `volrender/` recalls `volumes3d/`.

**P2: moves between two plans**

7. `plan/labelling-day.md`: merge units 1–2; add `shape-features/` (from
   `plan/measurements-day.md`); measurements: split unit 4 into `relations/` + `coloc/`,
   dissolve `features-use/`.
8. `plan/colour-day.md` / `plan/unmixing-day.md`: stain deconvolution → unmixing day;
   white balance → `rgb-linear/`; `colour-display/` core; k-means dropped from
   `colour-perceptual/`; unmixing `unmixgeom/` merged into `endmembers/`.
9. `plan/levelsets-day.md` / `plan/dynprog-day.md`: `fastmarching/` → dynprog core;
   `seamcarve/` → optional; marching squares by hand removed from level sets.
10. `plan/gmm-day.md` `hmrf-em/` → merged into `plan/ising-day.md` `potts/`.
11. `fourier/lec.qmd`: add "the DFT assumes the image repeats: leakage and tapers" (concept +
    math slide) and a continuous-FT math slide; `plan/compression-day.md`,
    `plan/fftregistration-day.md`, `plan/wavelets-day.md`, `plan/radon-day.md` recall them.
12. `plan/compression-day.md`: `compress-science/` core, `lossless/` trimmed.
13. `highbit/lec.qmd`: "Colour maps, HDR and tone mapping" → pointer to `colour-display/`
    (keep tone mapping). `seg-energy/lec.qmd`: "Thresholding" slide → one-line recall once
    the thresholding day exists.
14. `plan/graphcut-day.md`: add an "Overlaps / who owns what" section (it is the only plan
    without one): owns the labelling energy and $\lambda$, `maxflow`, graph Laplacian; points to
    Ising (probabilistic view, Potts), dynprog (chains), thresholding (data term = min-error).
15. `plan/thresholding-day.md`: integral image by-hand item (c) → recall; local statistics
    via `lib/integral.ts`.

**P3: infrastructure refactors (existing code first, then plan wording)**

16. Move `jacobiEigen` from `demos/src/ncut/spectral.ts` into `lib/linalg.ts`; point
    `matrix2x2/geom.ts` and `linae/data.ts` at it (regression tests: `ncut`, `linae` unchanged).
17. Move `demos/src/fourier2d/fft.ts` → `lib/fft.ts` (`rmse` → `lib/metrics.ts`).
18. One `loadGrey()` for `cell256.png` in `lib/imageops.ts`, used by the six demos that load it.
19. Move `poisson` from `camerabits/sensor.ts` and `rng` into `lib/random.ts` (`ui.ts`
    re-exports `rng`); add `gauss()`.
20. Move `ncut/data.ts` `POINT_PRESETS` into `lib/datasets2d.ts`; extract graph-cut seed
    painting into `lib/paint.ts`; move `window16` and `quantise` synthetic generators into
    `lib/synth.ts`.
21. Update every plan's "Shared engineering" to the file names in §3.1 (notably: no
    `lib/contour.ts`, no 2×2 eigen in `imageops.ts`, no colour map in `ui.ts`, no histogram in
    `entropy.ts`, evaluation in `lib/metrics.ts`, `gl3d.ts` owned by graphics3d).
22. Adopt the conventions table (§4); first fixes in plans not yet built: SVM
    `lib/svmkernels.ts` and "kernel function"; snakes "keep $k$ modes"; labelling/measurements
    $S$ instead of $C$; morphology/watershed $\mathrm{DT}$ instead of $D$; wavelets/compression/integral
    subband names.
