# Plan: a day on measuring objects

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** core day 11 of 18, Part II "From pixels to objects" (its last
day). Order: day 10 watershed → this → day 12 FFT registration. Edits from the three
reviews applied 2026-11-12: `shape-features/` moved to the labelling day (day 9);
`relations-coloc/` split into `relations/` and `coloc/`; optional `features-use/`
dissolved (units/normalisation/aggregation → end of `intensity-features/`; classifying
feature tables → random forests day 14, `forest-objects/`; statistics of measurements →
evaluation day 16).

## Goal of the day

After the day a student can:

1. read a per-object **measurement table** as produced by common tools, say for each common column what it measures, in which **units**, and what it is sensitive to (resolution, perimeter estimator, segmentation error, background, border); the size and shape columns are recalled from the labelling day (day 9, `regionprops/`, `shape-features/`);
2. compute **intensity** descriptors of an object (integrated, mean, median, quartiles, MAD, edge intensity, **mass displacement**, a **radial distribution** in normalised rings) and say which ones change under a background offset or a one-pixel mask error;
3. build a **grey-level co-occurrence matrix** for a tiny patch and compute the main **Haralick** features from it, and explain the choices of distance, angle and number of grey levels;
4. compute **neighbour** and **parent–child** measurements (number of neighbours, percent touching, nearest-neighbour distance, children per parent) and recognise how the definition of "neighbour" changes the answer;
5. compute **colocalisation** coefficients of two channels (Pearson, Manders M1/M2 with thresholds), and recognise the classic pitfalls (background, missing thresholds, bleed-through, resolution);
6. turn per-object features into something usable: physical units, normalisation and per-image / per-well aggregation (classification of feature tables: day 14; statistics of measurements and replicates: day 16).

Why it fits this audience: every feature is a sum, a ratio, a maximum or a small
matrix computed over a set of pixels; the statistics (mean, variance, median, Pearson
correlation) are ones they already know, now applied to pixels. The new parts are
**combinatorial** (the co-occurrence matrix, the neighbour graph) and statistical
(colocalisation as a 2D histogram). The geometric descriptors (convex hull, calipers)
were taught on day 9. The day is deliberately a practical
**catalogue**: theory of individual measures lives on other days and is recalled here
in one line with a pointer.

## Prerequisite gap / overlaps

This day owns the practical **catalogue of common per-object measures** and how to use
them (definitions as the tools implement them, naming, units, pitfalls, aggregation).
It does not re-derive measures whose theory another day owns:

- **Labelling** (day 9, `plan/labelling-day.md`): owns, in `regionprops/`, area, centroid,
  bounding box and extent, the **perimeter estimators and their bias**, central moments
  and the second-moment matrix $S$ → orientation, eccentricity, equivalent ellipse axes,
  and the **Euler number**; in `shape-features/` (moved there from this day) convex hull,
  shoelace area, solidity, form factor and compactness variants, Feret diameters by
  rotating calipers, inscribed radius and Fourier outlines; and in `label-measure/` the
  measurement table as a data structure, filtering by properties, counting, border
  objects and the unbiased counting frame. Here all of these are **recalled in one
  catalogue slide** at the start of unit 1 (with the tool naming of the teacher table
  below); no re-derivation. The `regionfilter` demo there is the "filter by a feature"
  demo; this day does not duplicate it.
- **Moments** (elective, `plan/moments-day.md`): owns Hu invariants, Zernike moments and
  moment features. One catalogue row here ("rotation-invariant shape codes; moments
  elective"), no derivation; the concept layer never depends on it.
- **Skeletons** (elective, `plan/skeleton-day.md`): owns skeleton length, branch counts,
  Sholl analysis, tortuosity and radius along a centre line. Catalogue row only, marked
  "(elective)".
- **Morphology** (day 8, `plan/morphology-day.md`): owns granulometry and the pattern spectrum
  (`morph-openclose/`, demo `granulo`). CellProfiler's `MeasureGranularity` and
  blockflow's granularity rows (survival/loss fraction per radius) are the per-image /
  per-object version: a catalogue row in unit 2 next to texture, with a pointer. The
  edge band for edge intensity (unit 1) is an erosion: one-line recall. Also owns the
  **distance transform** $\mathrm{DT}$ (`lib/distance.ts`), used here for radial bins and
  "expand until adjacent".
- **Thresholding** (day 6, `plan/thresholding-day.md`): owns pixel-level segmentation
  **evaluation** (Dice, Jaccard, ROC). Here only the converse question is asked: how much
  does a feature change when the segmentation is slightly wrong (unit 1 presets). Its
  histogram helper is reused for intensity distributions. Manders thresholds and
  Costes' automatic threshold (unit 4) are thresholds on a 2D scatter plot; one-line
  back-reference to Otsu, not re-derived.
- **Evaluation and statistics** (day 16, `plan/evaluation-day.md`): owns object-level
  evaluation and the **statistics of measurements** (unit of replication, cells vs
  replicates as $n$, effect sizes, multiple testing over many features, measurement bias).
  This day ends with a one-line forward pointer there.
- **Random forests** (day 14): classification of feature tables (`forest-objects/`),
  train/test, overfitting and feature selection. One-line forward pointer.
- **Bit depth and noise** (day 3, `plan/bitdepth-day.md`): owns how intensities are stored
  (12/16-bit, rescaling, saturation, offset). Here: integrated and mean intensity are in
  **whatever units the file stored**, CellProfiler rescales to $[0, 1]$ by the container
  range (so a 12-bit image in uint16 gives "mean intensity 0.03"), and saturated pixels
  bias max, mean and integrated intensity. One slide in unit 2 with a pointer.
- **Colour** (day 7, `plan/colour-day.md`): owns colour spaces. Colocalisation here works
  on two **fluorescence channels** (or two stain concentration images); spectral
  **bleed-through** is named as a colocalisation artefact with a pointer to the spectral
  unmixing elective (`plan/unmixing-day.md`, which also owns stain deconvolution).
- **Image formation** (day 4): the PSF sets the resolution limit used in unit 4 ("same
  diffraction spot") and the small-object bias of size measures; one-line recall.
- **PCA** (`pca/`, day 1): one line where redundant features are discussed.
- **Entropy** (compression elective owns it): the Haralick entropy gets one line
  ("expected surprise of a pixel pair").
- **3D**: stacks, voxel spacing $(\Delta x, \Delta y, \Delta z)$, anisotropy: pointer to
  `volumes3d/` (marching-cubes elective).

Links to other days: every segmentation day (thresholding day 6, watershed day 10, graph
cuts day 13, networks days 15 and 18, level sets as an elective) ends in a label image,
and this day is what is done with it next.

## Units (one folder each: lec, quiz, exercise with #explore)

Student material names measures by what they compute (e.g. "solidity: area / convex
hull area"), never by a tool's column name; see the teacher-only naming table below.

Four core units, about 1.5 h each including the exercise (no optional unit). Each
unit's lecture ends with a **catalogue slide**: a table of the unit's features with
formula, unit (px, px², µm, intensity units, dimensionless), range, and the column name
in CellProfiler and in blockflow (see the naming table below).

### 1. `intensity-features/`: Intensity inside an object

- **Opening recall** (one catalogue slide, pointer to day 9 `regionprops/` and
  `shape-features/`): the size and shape rows of the table (area, perimeter and its bias,
  extent, axis lengths $4\sqrt{\lambda_{1,2}}$ of $S$, eccentricity, Euler number, convex area,
  solidity, form factor, the compactness name collision, Feret, inscribed radius), with
  their tool names; moments (Hu/Zernike) and skeleton rows marked "(elective)". Then the
  question of this day: what else is in the table?
- Picture first: five cells with the same mask and the same total signal but different
  distributions (uniform, bright rim, one bright spot off-centre, punctate, gradient).
  Which numbers separate them?
- **Moments of the intensity values** in the object $\Omega$: integrated (sum), mean,
  std, min, max; robust versions: median, lower/upper **quartiles**, **MAD**
  (median absolute deviation; $1.4826\cdot$MAD estimates $\sigma$ for normal data).
  Integrated intensity ∝ amount of fluorophore (DNA content: G1 vs G2 nuclei have a 1:2
  ratio of integrated DAPI); mean intensity ∝ concentration.
- **Which features survive what** (the core message of the unit, as a table): a constant
  background offset $b$ adds $b|\Omega|$ to the integral and $b$ to mean/median/quartiles,
  leaves std and MAD unchanged; a gain $g$ scales everything except ratios; a mask that
  is one pixel too large includes the blurred edge and lowers the mean; saturation caps
  max and biases mean and integral. Local background: measure a **ring** around the
  object (gap, then band) and subtract (blockflow composes this from ring labels).
  Pointer to day 3 (bit depth) for stored units and rescaling.
- **Edge intensity**: the same statistics on the object's boundary band (object minus its
  erosion by one pixel; recall from day 8). Ratio edge/mean distinguishes rim
  staining (membrane) from filled staining.
- **Intensity-weighted centroid** $\bar{\mathbf x}_I = \sum I(\mathbf x)\mathbf x / \sum I$
  vs geometric centroid; **mass displacement** = their distance (in µm). Large when the
  signal is polarised (one bright spot off-centre); 0 by symmetry for uniform and
  rim-stained discs. One line: it is the first intensity moment, i.e. the intensity
  version of the region moments of day 9.
- **Radial distribution** (CellProfiler `MeasureObjectIntensityDistribution`): divide
  the object into $n$ rings by **normalised distance** $\rho = d_c/(d_c + d_e)$ from the
  centre ($d_c$ = distance to the centre point, $d_e$ = distance to the edge, from the
  distance transform $\mathrm{DT}$, day 8), so rings follow the object's shape instead of being circles.
  Per ring $k$: **FracAtD** = fraction of the total intensity in ring $k$; **MeanFrac** =
  FracAtD divided by the fraction of the object's pixels in ring $k$ (1 = uniform, >1 =
  enriched); **RadialCV** = coefficient of variation of the ring's intensity across
  angular sectors (asymmetry within a ring). Choice of centre (centroid, intensity max, a
  parent object such as the nucleus) matters; with a nucleus as centre this measures
  nuclear vs peripheral localisation (translocation assays).
- Per-object vs per-image measures: the same statistics over the whole image or a
  region; image-level saturation fraction and focus as quality-control columns (one
  slide).
- **Closing section: from a feature table to numbers you can compare** (from the
  dissolved `features-use/`): **physical units** and calibration in one table (px → µm,
  px² → µm², voxel anisotropy; dimensionless ratios stay dimensionless); **aggregation**
  per object → per image / per well by median (robust) or mean, plus counts and
  fractions ("% cells with > 3 spots"); **normalisation** as z-scores against control
  wells, robust z with median and MAD; plate and batch effects (row/column gradients) in
  one picture; **redundancy** (area, perimeter, equivalent diameter, convex area and max
  Feret are strongly correlated: pick one per block, or PCA, recall `pca/` day 1).
  Forward pointers, one line each: classifying objects from the table is day 14
  (`forest-objects/`); turning the numbers into a claim (replicates, effect sizes,
  multiple testing) is day 16.
- **Demo `intensityfeat`** (new): one synthetic cell (seeded, 160×160, a mask with an
  editable outline from `lib/shapes.ts` or a disc/ellipse), intensity pattern chosen from
  uniform / rim / off-centre spot / punctate / gradient, sliders for background offset,
  gain, noise, saturation level, mask error (erode/dilate ±2 px), number of radial bins,
  centre (centroid / intensity maximum). Panels: (left) the image with mask, edge band,
  radial rings in alternating tints, the geometric centroid and the intensity-weighted
  centroid joined by the mass-displacement arrow; (middle) the histogram of intensities
  inside the mask with mean, median, quartiles and MAD marked (thresholding day's
  histogram drawer); (right) bar plot of FracAtD and MeanFrac per ring, and a live table
  of all features with a "Δ vs reference" column after each slider change. A second tab
  shows a population of 20 cells of two types and a scatter of two chosen features. A
  *plate* tab (from the dissolved `featurespace` demo): a 4×6 well plate with two seeded
  phenotypes and a plate gradient, one aggregated feature before and after robust
  z-score normalisation.
  Presets: "background offset: mean moves, std does not", "same integral, different
  pattern", "rim vs filled: edge/mean ratio", "polarised cell: mass displacement",
  "nuclear vs cytoplasmic: radial bins", "dilated mask dilutes the mean", "saturation",
  "plate gradient hides the phenotype", "robust z fixes it".
- By hand: (a) a 5×5 object with given intensities: integrated, mean, median, quartiles,
  MAD, edge-band mean (after one erosion with the cross); (b) intensity-weighted centroid
  and mass displacement; (c) normalised radial position of three marked pixels from
  given $d_c$, $d_e$, assign them to 3 rings, compute FracAtD and MeanFrac for a tiny
  example. (d) z-score and robust z-score of a feature for 5 wells given controls.

### 2. `texture-features/`: Texture: co-occurrence matrices and Haralick features

- Picture first: two patches with the **same histogram** (same mean, std, quartiles) but
  different textures (fine vs coarse, stripes vs blobs; made by permuting pixels).
  Histogram features cannot tell them apart; texture needs **pairs** of pixels.
- **Grey-level co-occurrence matrix** (Haralick 1973): quantise to $L$ levels; for an
  offset $(d, \theta)$ count pairs $(i, j)$ = (level at $\mathbf x$, level at
  $\mathbf x + \mathbf d$) inside the object; symmetrise ($P + P^\top$) and normalise to
  a joint probability $p(i, j)$. Link: it is the **2D histogram** of a pixel against its
  shifted copy, the same object as the colocalisation scatter plot of unit 4 with the
  second channel replaced by the image itself shifted by $\mathbf d$.
- Features as statistics of $p$ (each computed by hand in the exercise):
  **contrast** $\sum (i-j)^2 p$ (mass away from the diagonal), **homogeneity / inverse
  difference moment** $\sum p/(1 + (i-j)^2)$, **ASM** $\sum p^2$ and **energy** $\sqrt{\text{ASM}}$
  (uniformity), **entropy** $-\sum p\log p$ (one line: expected surprise of a pixel pair; entropy is owned by the compression elective), **correlation**
  $\sum (i-\mu_i)(j-\mu_j)p / (\sigma_i\sigma_j)$ (Pearson correlation of a pixel with its
  neighbour: the autocorrelation at lag $\mathbf d$, recall `fourier/`, day 1). Name the rest
  of Haralick's 13–14 (sum/difference averages, variances and entropies, information
  measures of correlation) without formulas; many are strongly correlated with the five.
- **Choices that change the numbers**: distance $d$ (scale of the texture: CellProfiler's
  "texture scale"), angle (0°, 45°, 90°, 135° in 2D, 13 directions in 3D; report per
  angle or average for rotation invariance), number of grey levels $L$ (8–256; too many
  levels on a small object gives a sparse matrix and noisy features), quantisation range
  (image min–max vs fixed range: per-object rescaling destroys comparability across
  objects), symmetric or not, which pixels count (pairs with both ends inside the object).
  Naming trap: "homogeneity", "IDM" and "inverse difference" are three slightly
  different formulas across tools; "energy" is ASM in some and $\sqrt{\text{ASM}}$ in
  others.
- Other texture descriptors in one slide each: **Gabor** filter bank energies (orientation
  and frequency selective; link to the Fourier day: a Gaussian window times a complex
  exponential), **local binary patterns** (8 neighbours thresholded at the centre → an
  8-bit code, histogram of codes, rotation-invariant uniform patterns), and
  **granularity** (opening spectrum; owned by day 8, CellProfiler
  `MeasureGranularity`). Forward pointer: network embeddings as learned texture features (days 15 and 18).
- **Demo `glcm`** (new): left, a texture patch (synthetic, seeded: oriented stripes with
  angle and period sliders, checkerboard, white noise, smoothed noise with a correlation
  length slider, blobs, the "same histogram" permuted pair; real: crops of `cell256.png`
  and a fluorescence image, see open questions) shown zoomed with the current offset
  drawn as an arrow between a few sample pixel pairs; middle, the GLCM as a heat map
  ($L \times L$, log colour scale option, diagonal marked); right, the five main features
  as bars, and a polar plot of contrast (or correlation) versus angle for the current
  $d$. Controls: $d$ (1–16), angle (0/45/90/135 or a free angle with rounding shown),
  levels $L$ (2–64), quantisation range (patch min–max / fixed), symmetric toggle, an
  object mask option (disc) so only pairs inside count. A *compare* mode shows two
  patches side by side with feature differences. Presets: "same histogram, different
  texture", "stripes: contrast depends on angle", "period = d: contrast vanishes",
  "fine vs coarse: contrast vs d", "too many levels on a small patch", "min–max
  rescaling hides a brightness difference".
- By hand: (a) the symmetric normalised GLCM of a 4×4 image with 3 grey levels for
  offset (1, 0°), then contrast, ASM, homogeneity, entropy and correlation; (b) the same
  image at 90°: which features change and why; (c) LBP code of one 3×3 neighbourhood
  and its rotation-invariant version.

### 3. `relations/`: Measurements between objects: neighbours, parents and children

- Two kinds of "between" measurements: between **objects** (this unit) and between
  **channels** (unit 4). Both need a definition of *which* pixels or objects are paired,
  and both are easy to over-interpret.
- **Neighbours**. Three definitions, three answers: **adjacent** (labels share a
  boundary, 4/8-connectivity), **within distance $r$** (boundaries or centroids closer
  than $r$), **expand until adjacent** (grow every object into the background up to the
  midline between objects, i.e. a Voronoi tessellation of the objects via the distance
  transform $\mathrm{DT}$, day 8; the same picture as watershed, day 10). Per object:
  **number of neighbours**, **percent touching** (fraction of the boundary pixels adjacent to another object), **first and
  second closest** object and their distances, **angle between** the two closest
  neighbours. The **neighbour graph** (objects as nodes; link to day 9's region
  adjacency, and forward to the graphs of day 13). Population summaries: the
  nearest-neighbour-distance histogram compared with a random (Poisson) pattern of the
  same density (Clark–Evans ratio $R = \bar d_{\text{obs}} / (0.5/\sqrt{\rho})$: < 1
  clustered, > 1 regular). Edge effect: objects near the image border have fewer
  neighbours (counting frame again).
- **Parents and children**: nuclei inside cells, spots inside nuclei, cytoplasm = cell
  minus nucleus. A child's parent = the label with the largest overlap (or the label at
  the child's centroid); per parent: **child count**, mean child intensity, distance of
  each child to the parent's centroid or edge (normalised, as in radial bins). Children
  outside any parent and children overlapping two parents are explicit cases. Per-parent
  aggregation of child features (mean, sum, count) is how "spots per nucleus" columns
  are made. Overlap as a relation: overlap fraction of A and of B, Jaccard (pointer to
  day 6's evaluation; object matching by IoU is day 16).
- **Demo `neighbours`** (new): a seeded synthetic population of 30–80 cells (nuclei as
  ellipses from `lib/shapes.ts` with density and clustering sliders: Poisson, clustered,
  regular), cell bodies by expansion from the nuclei, and spots inside some nuclei.
  Overlay: the neighbour graph for the chosen definition (adjacent / within $r$ /
  expand until adjacent, $r$ slider), click an object to see its row (number of
  neighbours, percent touching, first/second closest, angle, child count). Side panel:
  histogram of number of neighbours and of nearest-neighbour distance with the Poisson
  expectation overlaid, Clark–Evans $R$; a border rule toggle (all / exclude border /
  counting frame). Mode *parents*: nuclei → spots, with parent assignment by overlap or
  centroid and the ambiguous cases highlighted. Presets: "adjacent vs expanded: counts
  differ", "within r: count grows with r", "clustered vs regular: R < 1 and > 1",
  "border objects have fewer neighbours", "spots per nucleus", "a spot on the border
  between two nuclei".
- By hand: (a) adjacency list and number of neighbours from a 6×6 label image (4- and
  8-connectivity); (b) nearest-neighbour distances of 5 given centroids and the
  Clark–Evans ratio; (c) which spots belong to which nucleus in a small label pair, by
  overlap and by centroid.

### 4. `coloc/`: Measurements between channels: colocalisation

- Picture first: two channels merged in green/magenta: "yellow means colocalised" is
  the claim this unit takes apart.
- **Colocalisation** (two channels $A$, $B$ in a region: whole image, a cell, or each
  object). The picture is the **scatter plot** (2D histogram) of $(A(\mathbf x),
  B(\mathbf x))$.
  - **Pearson** $r$ = correlation of the two channels over the region's pixels
    (statistics they know). Background pixels pull $r$ up (both dark together), so the
    ROI matters; $r$ is invariant to offset and gain but not to bleed-through or noise.
  - **Overlap coefficient** (Manders 1993) $\sum AB / \sqrt{\sum A^2 \sum B^2}$
    (uncentred Pearson; sensitive to offsets).
  - **Manders M1** = fraction of $A$'s total intensity located where $B > t_B$; **M2**
    the converse. Asymmetric by design: a small structure fully inside a large one gives
    M1 ≈ 1, M2 small. Without thresholds ($t = 0$) noise makes every pixel "positive" and
    M1 ≈ M2 ≈ 1, which is why the **thresholds** are the whole method: Costes'
    automatic threshold (lower $t$ along the regression line until the pixels below have
    $r \le 0$) in one slide; manual thresholds as a percentage of the maximum
    (CellProfiler) in one line.
  - Controls: shift one channel by a few pixels (or scramble blocks: Costes'
    randomisation test) and see $r$ drop to the chance level; chromatic misalignment,
    bleed-through (pointer to the spectral unmixing elective), resolution (two proteins in
    the same 200 nm diffraction spot "colocalise" at any distance below it; recall the PSF
    from day 4). Pointer:
    object-based colocalisation (fraction of spots in A whose centroid lies in an object
    of B) reuses the parent–child relation of unit 3.
- **Demo `coloc`** (new): two synthetic channels (seeded): a set of structures in $A$; $B$
  = a mixture of the same structures (fraction $f$ slider), independent structures, a
  constant background offset, noise, bleed-through of $A$ into $B$ (slider), and a lateral
  shift of $B$ (slider, pixels). Panels: merged false-colour image (green/magenta) with an
  ROI choice (whole image / inside cells mask); the 2D histogram with the regression line,
  draggable threshold lines $t_A$, $t_B$ dividing it into quadrants; readouts $r$
  (all pixels and above thresholds), overlap coefficient, M1, M2, and a *Costes* button
  that animates the threshold search; a *randomise* button (block scramble of $B$, 50
  repeats) showing the distribution of $r$ under no colocalisation. Presets: "perfect
  colocalisation", "independent channels: r ≈ 0", "background inflates r", "M1 ≈ M2 ≈ 1
  without thresholds", "A inside B: M1 ≫ M2", "bleed-through fakes colocalisation",
  "shift by 3 px: r drops".
- By hand: (a) Pearson, overlap coefficient, M1 and M2 (with given thresholds) for two
  3×3 channels; (b) the same with a constant offset added to one channel: which change?
  (c) M1 and M2 for a small structure fully inside a large one.

## CellProfiler vs blockflow: naming of the common measures

> **Teacher reference only** (decided 2026-11-12): the course is about the ideas, not
> specific tools. Students learn each measure by its definition and behaviour; tool
> names (CellProfiler, blockflow, …) do not appear in student material beyond at most a
> one-line pointer. This table is for the teachers, e.g. for later code examples.

For the catalogue slides and a one-page handout. Blockflow names checked in its source
(`src/ops/measure.rs`, `src/ops/intensity_statistics.rs`, `src/ops/cleanup.rs`, `FERET.md`);
CellProfiler names from its module documentation. Blockflow is **3D-first**: most shape
rows are defined for voxels (a 2D image is a one-slice volume), and families are opt-in
per measurement.

| Measure | CellProfiler (module: column) | blockflow (row family: column) | Note for the course |
|---|---|---|---|
| Area / volume | SizeShape: `Area` | shape: `count` (voxels); `physical_volume` | blockflow counts voxels; physical via spacing |
| Centroid | `Location_Center_X/Y` | shape: `centroid_x/y/z` | |
| Bounding box, extent | `BoundingBoxMinimum_X…`, `BoundingBoxArea`, `Extent` | shape: `bbox_min_*`, `bbox_volume`, `bbox_fill_fraction` | extent = bbox fill fraction |
| Perimeter / surface | `Perimeter` | boundary: `surface_area` from exposed voxel **faces** | face count = edge-count estimator in 2D: the biased one ($4/\pi$ on discs) |
| Form factor | `FormFactor` $4\pi A/P^2$ | `form_factor` $36\pi V^2/S^3$ (3D) | same idea, 3D formula; inherits the face-count bias |
| Compactness | `Compactness` (from second moments) | `compactness` $S^3/(36\pi V^2)$ (= 1/form factor) | **same name, different measure** |
| Convex hull, solidity | `ConvexArea`, `Solidity` | projected-convex: `convex_hull_area`, `solidity`; 3D hull: `volume`, `surface_area` | blockflow's 2D solidity is on the projection |
| Axis lengths, orientation, eccentricity | `MajorAxisLength`, `MinorAxisLength`, `Orientation` (degrees), `Eccentricity` | shape: `principal_axis_length_0/1/2`, `orientation_yx`, `eccentricity` (scikit-image 2D definition, Y/X plane) | indices instead of major/minor; check angle units and sign (image y down) |
| Equivalent diameter | `EquivalentDiameter` (circle, 2D) | `equivalent_sphere_diameter`, `equivalent_sphere_radius` | sphere vs circle: differ on 2D data |
| Feret | `MaxFeretDiameter`, `MinFeretDiameter` | `max_hull_feret_diameter`, `max_projected_feret_diameter`, `max_voxel_feret_diameter`, directional Feret (`max_directional_feret_diameter`) | blockflow names the **method** in the column; no min Feret yet (planned via directional widths and rotating calipers, `FERET.md`) |
| Inscribed / mean radius | `MaximumRadius`, `MeanRadius`, `MedianRadius` | radius rows: `max`, `mean`, `median` (exact per-label distance field, or a shared approximation) | |
| Enclosing circle | — | `min_enclosing_circle_radius`, enclosing sphere `radius` | blockflow only |
| Euler number | `EulerNumber` | topology: `euler_number`, `components`, `tunnels`, `cavities` | 3D: components − tunnels + cavities |
| Hu / Zernike | `Zernike_n_m` | Hu, weighted Hu, Zernike, 3D Zernike rows | moments elective |
| Integrated, mean, min, max | Intensity: `IntegratedIntensity`, `MeanIntensity`, `MinIntensity`, `MaxIntensity` | intensity: `sum`, `mean`, `min`, `max` | "integrated" = sum |
| Std, quartiles, median, MAD | `StdIntensity`, `LowerQuartileIntensity`, `MedianIntensity`, `UpperQuartileIntensity`, `MADIntensity` | distribution: `quartile_1`, `median`, `quartile_3`, `std_dev`, `mad`, percentiles; statistics: `std_dev`, `coefficient_of_variation` | |
| Edge intensity | `IntegratedIntensityEdge`, `MeanIntensityEdge`, … | not a named family (composable: measure a boundary/ring label image, `ring_labels_into`) | |
| Weighted centroid, mass displacement | `Location_CenterMassIntensity_X/Y`, `MassDisplacement` | `weighted_centroid_*`, statistics: `mass_displacement` | |
| Radial distribution | IntensityDistribution: `FracAtD`, `MeanFrac`, `RadialCV` | not found | |
| Fraction above a level | — | statistics: `count_ge_<t>`, `fraction_ge_<t>` | saturation check |
| Texture | Texture: 13 Haralick features per scale and direction (`AngularSecondMoment`, `Contrast`, `Correlation`, `InverseDifferenceMoment`, `Entropy`, `InfoMeas1/2`, …), grey levels setting | GLCM: 25 features incl. `angular_second_moment` **and** `energy`, `homogeneity` **and** `inverse_difference_moment` (+ normalised variants), `information_measure_correlation_1/2`, `maximal_correlation_coefficient`; directed offsets, fused across offsets on request; explicit quantisation (levels, min, max) | the naming traps of unit 3 are visible here |
| Granularity | `MeasureGranularity` spectrum | granularity: survival / loss fractions per radius, AUC, peak loss radius | morphology day |
| Neighbours | ObjectNeighbors: `NumberOfNeighbors`, `PercentTouching`, `FirstClosestDistance`, `SecondClosestDistance`, `AngleBetweenNeighbors` (adjacent / within distance / expand until adjacent) | `touching_neighbors` + `contact_faces`; centroid: `within_distance_neighbors`, `closest_label`, `closest_distance`, `second_closest_*`, `angle_between_closest`; boundary-distance and expansion-until-adjacent relations | no percent touching as such; contact face counts instead |
| Parent / child | RelateObjects: `Children_<X>_Count`, `Parent_<Y>` | overlap: `child_count`, `fraction_of_label`, maximum-overlap assignment; overlap `jaccard` | |
| Colocalisation | Colocalization: `Correlation`, `Slope`, `Overlap`, `K1/K2`, `Manders`, `RWC`, `Costes` | `pearson`, `slope_b_on_a`, `overlap_coefficient`, `manders_m1/m2` (threshold 0), Costes (`threshold_a/b`, `manders_m1/m2`, `below_threshold_pearson`), rank-weighted | blockflow's basic Manders uses **zero** as threshold: the "M1 ≈ M2 ≈ 1" preset |

## Shared engineering

- `demos/src/lib/features.ts` (new, owned by this day; the shape rows are first written
  for the labelling day's `shape-features/`): one function per catalogue row over a label
  image + optional intensity image(s) + pixel spacing $(\Delta x, \Delta y)$, returning
  a table (one row per label) with the column definitions and units used on the catalogue
  slides. Builds on `lib/label.ts` (labelling day: area, centroid, bbox, moments to order
  2, perimeter estimators, Euler number, border flags) and `lib/convexhull.ts`
  (labelling day) rather than recomputing them; shape rows (equivalent diameter, form
  factor, compactness variants, solidity, Feret, radii from `lib/distance.ts`), intensity
  statistics (exact
  median/quartiles/MAD by sorting per object), edge band (erosion from `lib/morph.ts`),
  weighted centroid and mass displacement, radial bins (normalised distance from two
  distance transforms), local background ring, robust z-scores and per-well aggregation.
  Zernike/Hu columns optional via `lib/moments.ts` (moments elective).
- `demos/src/lib/glcm.ts` (new): quantisation (range, levels), GLCM for an offset within
  a mask, symmetrise/normalise, the Haralick features (all 13 for the table, with the
  alternative definitions flagged), LBP codes and histogram.
- `demos/src/lib/neighbours.ts` (new, small): adjacency pairs and boundary-contact
  counts from a label image (neighbourhood offset tables from the "grid" section of
  `lib/graph.ts`, owner: graph-cut day), expansion to the midline via
  `lib/distance.ts` (feature transform / nearest label), centroid nearest neighbours
  (brute force is fine for ≤ 200 objects), parent assignment by overlap or centroid;
  colocalisation coefficients and Costes threshold search (or `lib/coloc.ts` if it grows).
- Reused, not duplicated: `lib/label.ts`, `lib/convexhull.ts`, `lib/moments.ts`,
  `lib/distance.ts`, `lib/morph.ts`, `lib/imageops.ts` (`loadGrey()`),
  `lib/synth.ts` (seeded synthetic populations), `lib/colour.ts` (label palette,
  green/magenta merge), `lib/shapes.ts` (nucleus outlines), `lib/histogram.ts`
  (thresholding day: histogram drawer for `intensityfeat`), `lib/threshold.ts` (Otsu for
  Costes' comparison), `lib/plot.ts` (bars, polar plot, scatter, heat map: add a heat-map
  helper there if missing), `rng`/`gauss`/`poisson` from `lib/random.ts`.
- **Language / speed**: TypeScript throughout. Estimates: all features of ~80 objects in a
  256×256 image in a few ms (one labelling scan, two distance transforms, sorting per
  object); a GLCM of a 128×128 patch is one pass; Costes' search is ~50 Pearson
  evaluations; the randomisation test (50 block scrambles of 256×256) a few tens of ms,
  run on button press. No Rust/Wasm expected; measure first.
- Tests (`make test`; hull, calipers and shape-row tests live with the labelling day):
  radial bins: FracAtD sums to 1, MeanFrac = 1 for a
  uniform object; GLCM of a constant image has ASM = 1, contrast = 0, and a symmetric
  GLCM equals its transpose; Pearson of $B = aA + b$ is 1; M1 = M2 = 1 when $A = B$;
  `features.ts` order-2 columns agree with `lib/label.ts` and `lib/moments.ts`; robust z
  of the control wells has median 0.
- Images: synthetic, seeded, generated in TS (no files) for all presets;
  `demos/src/fourier2d/cell256.png` for texture crops; a real two-channel fluorescence
  image and a real cell population (see open questions; credit and licence in the demo).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only, keyboard back to
  reveal.js, labels ≥ 18 px, fits 1150×560 on a slide (the feature tables scroll inside
  their panel and show a preset-chosen subset of rows on slides via `data-rows`),
  `data-*` settings (`data-mode`, `data-preset`, `data-rows`), static figures as print
  fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`intensityfeat`, incl. the plate tab) |
| 10:45–12:15 | Unit 2 lecture, GLCM by hand, `glcm` |
| 13:15–14:45 | Unit 3 lecture + exercise (`neighbours`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`coloc`); quizzes as wrap-up |

## Open questions

- **Decided (course map / review-organisation §2.6):** `shape-features/` moved to the
  labelling day; `relations-coloc/` split into `relations/` and `coloc/` (colocalisation
  stays core, as recommended here); `features-use/` dissolved. Review-gaps gap 4 proposed
  making `features-use/` core with a "from measurements to conclusions" half; the course
  map gives the statistics of measurements to the evaluation day (day 16), so only units,
  normalisation and aggregation stay here. If texture is too long, shorten it to GLCM
  only.
- **Decided:** `shapefeat` is `data-mode="features"` of the labelling day's `moments`
  shell (`lib/shapecanvas.ts`).
- How tool-specific should the material be? Proposal: teach definitions tool-neutrally
  and give the CellProfiler/blockflow (and scikit-image) naming table as a handout. Is
  blockflow shown to students by name?
- Definitions to settle in `lib/features.ts` and say on the slides: hull of pixel corners
  vs centres; which perimeter estimator the form factor uses; axis lengths as
  $4\sqrt{\lambda}$ (scikit-image / CellProfiler; to check against blockflow's
  `principal_axis_length_*`); orientation in degrees with the image y-axis pointing down.
- Real data: a two-channel colocalisation image and a cell population with nuclei and
  cytoplasm under a licence that allows redistribution (Broad Bioimage Benchmark
  Collection sets such as BBBC022 Cell Painting are candidates; check licence per set).
- **Decided:** no 3D unit here. 3D images (stacks, spacing, anisotropy) are `volumes3d/`
  and surface measures are `surface-measure/`, both on the marching-cubes elective; this
  day has a pointer only.
- Programming language for the exercise code is still open (see README); units contain
  no code sections.

## References

- Carpenter et al. (2006), *CellProfiler: image analysis software for identifying and quantifying cell phenotypes*, Genome Biology; Stirling et al. (2021), *CellProfiler 4*, BMC Bioinformatics; CellProfiler module documentation (MeasureObjectSizeShape, MeasureObjectIntensity, MeasureObjectIntensityDistribution, MeasureTexture, MeasureObjectNeighbors, MeasureColocalization, RelateObjects, MeasureGranularity).
- van der Walt et al. (2014), *scikit-image: image processing in Python*, PeerJ (`regionprops` definitions).
- Haralick, Shanmugam & Dinstein (1973), *Textural features for image classification*, IEEE SMC.
- Ojala, Pietikäinen & Mäenpää (2002), *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*, PAMI.
- Andrew (1979), *Another efficient algorithm for convex hulls in two dimensions*, IPL; Toussaint (1983), *Solving geometric problems with the rotating calipers*, IEEE MELECON; Welzl (1991), *Smallest enclosing disks (balls and ellipsoids)*.
- Manders, Verbeek & Aten (1993), *Measurement of co-localization of objects in dual-colour confocal images*, J. Microscopy.
- Costes et al. (2004), *Automatic and quantitative measurement of protein-protein colocalization in live cells*, Biophys. J.
- Bolte & Cordelières (2006), *A guided tour into subcellular colocalization analysis in light microscopy*, J. Microscopy; Dunn, Kamocka & McDonald (2011), *A practical guide to evaluating colocalization in biological microscopy*, Am. J. Physiol. Cell Physiol.
- Clark & Evans (1954), *Distance to nearest neighbor as a measure of spatial relationships in populations*, Ecology.
- Caicedo et al. (2017), *Data-analysis strategies for image-based cell profiling*, Nature Methods; Bray et al. (2016), *Cell Painting*, Nature Protocols.
- Ljosa, Sokolnicki & Carpenter (2012), *Annotated high-throughput microscopy image sets for validation*, Nature Methods (BBBC).
