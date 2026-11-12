# Plan: a day on measuring objects

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. read a per-object **measurement table** as produced by CellProfiler, scikit-image or blockflow, say for each common column what it measures, in which **units**, and what it is sensitive to (resolution, perimeter estimator, segmentation error, background, border);
2. compute the **size and shape** descriptors that go beyond moments by hand on a small object: convex hull area, solidity, extent, form factor, equivalent diameter, **minimum and maximum Feret diameter** (rotating calipers), maximum inscribed radius (distance transform);
3. compute **intensity** descriptors of an object (integrated, mean, median, quartiles, MAD, edge intensity, **mass displacement**, a **radial distribution** in normalised rings) and say which ones change under a background offset or a one-pixel mask error;
4. build a **grey-level co-occurrence matrix** for a tiny patch and compute the main **Haralick** features from it, and explain the choices of distance, angle and number of grey levels;
5. compute **neighbour** and **parent–child** measurements (number of neighbours, percent touching, nearest-neighbour distance, children per parent) and **colocalisation** coefficients of two channels (Pearson, Manders M1/M2 with thresholds), and recognise the classic pitfalls of each;
6. turn many features into something usable: physical units, normalisation and per-image / per-well aggregation, redundancy between features, and downstream filtering, PCA or classification (briefly).

Why it fits this audience: every feature is a sum, a ratio, a maximum or a small
matrix computed over a set of pixels; the statistics (mean, variance, median, Pearson
correlation) are ones they already know, now applied to pixels. The new parts are
**geometric** (convex hull, calipers, distance transform) and **combinatorial** (the
co-occurrence matrix, the neighbour graph). The day is deliberately a practical
**catalogue**: theory of individual measures lives on other days and is recalled here
in one line with a pointer.

## Prerequisite gap / overlaps

This day owns the practical **catalogue of common per-object measures** and how to use
them (definitions as the tools implement them, naming, units, pitfalls, aggregation).
It does not re-derive measures whose theory another day owns:

- **Labelling** (`plan/labelling-day.md`): owns, in `regionprops/`, area, centroid,
  bounding box and extent, the **perimeter estimators and their bias**, central moments
  and the second-moment matrix → orientation, eccentricity, equivalent ellipse axes, and
  the **Euler number**; and in `label-measure/` the measurement table as a data
  structure, filtering by properties, counting, border objects and the unbiased counting
  frame. Here these appear as **rows of the catalogue**, each with a one-line recall and
  a pointer; the new content is how they combine into derived ratios (form factor uses
  the perimeter, so it inherits its bias; major/minor axis lengths are $4\sqrt{\lambda_i}$
  of that day's matrix $C$) and how the tools name them. The `regionfilter` demo there is
  the downstream "filter by a feature" demo; unit 1 here does not duplicate it.
- **Moments** (`plan/moments-day.md`): owns Hu invariants, Zernike moments and moment
  features for classification (`moment-features/`). CellProfiler's
  `MeasureObjectSizeShape` reports `Zernike_n_m` columns and blockflow has Hu, weighted Hu
  and 2D/3D Zernike rows: one catalogue row here ("rotation-invariant shape codes, see the
  moments day"), no derivation.
- **Skeletons** (`plan/skeleton-day.md`): owns skeleton length, branch counts, Sholl
  analysis, tortuosity and radius along a centre line. Catalogue row only. The maximum
  inscribed radius here is the maximum of the distance transform inside the object, i.e.
  the largest disc of that day's medial axis; one sentence pointer.
- **Morphology** (`plan/morphology-day.md`): owns granulometry and the pattern spectrum
  (`morph-openclose/`, demo `granulo`). CellProfiler's `MeasureGranularity` and
  blockflow's granularity rows (survival/loss fraction per radius) are the per-image /
  per-object version: a catalogue row in unit 3 next to texture, with a pointer. The
  edge band for edge intensity (unit 2) is an erosion: one-line recall.
- **Thresholding** (`plan/thresholding-day.md`): owns segmentation **evaluation**
  (Dice, Jaccard, ROC). Here only the converse question is asked: how much does a
  feature change when the segmentation is slightly wrong (unit 1 and 2 presets). Its
  histogram helper is reused for intensity distributions. Manders thresholds and
  Costes' automatic threshold (unit 4) are thresholds on a 2D scatter plot; one-line
  back-reference to Otsu, not re-derived.
- **Bit depth** (`plan/bitdepth-day.md`, in progress): owns how intensities are stored
  (12/16-bit, rescaling, saturation, offset). Here: integrated and mean intensity are in
  **whatever units the file stored**, CellProfiler rescales to $[0, 1]$ by the container
  range (so a 12-bit image in uint16 gives "mean intensity 0.03"), and saturated pixels
  bias max, mean and integrated intensity. One slide in unit 2 with a pointer.
- **Colour** (`plan/colour-day.md`, in progress): owns colour spaces and stain
  separation. Colocalisation here works on two **fluorescence channels** (or two stain
  concentration images after that day's colour deconvolution); spectral bleed-through is
  named as a colocalisation artefact with a pointer to its unmixing slide.
- **Distance transform**: defined on the watershed and morphology days
  (`lib/distance.ts`); used here for inscribed radius, mean radius, radial bins and
  "expand until adjacent".
- **PCA**: introduced on the labelling and snakes days; unit 5 uses it in one line.

Links to other days, described neutrally: every segmentation day (thresholding,
watershed, graph cuts, level sets, deep-learning segmenters) ends in a label image, and
this day is what is done with it next.

## Units (one folder each: lec, quiz, exercise with #explore)

Student material names measures by what they compute (e.g. "solidity: area / convex
hull area"), never by a tool's column name; see the teacher-only naming table below.

Four core units, about 1.5 h each including the exercise, plus one optional unit. Each
unit's lecture ends with a **catalogue slide**: a table of the unit's features with
formula, unit (px, px², µm, intensity units, dimensionless), range, and the column name
in CellProfiler and in blockflow (see the naming table below).

### 1. `shape-features/`: Size and shape beyond moments

- Picture first: four objects with nearly equal area (disc, ellipse, star, crescent):
  which numbers tell them apart? The measurement table as the answer, recalled from
  `label-measure/` in one line.
- **Recalled rows** (one line each, pointer to `regionprops/`): area (pixel count ×
  pixel area), perimeter (three estimators, edge count overestimates a disc by $4/\pi$),
  bounding box and **extent** = area / box area, major/minor axis length $4\sqrt{\lambda_{1,2}}$,
  orientation, eccentricity, Euler number. **Equivalent diameter** $d = \sqrt{4A/\pi}$
  (2D) vs the equivalent-sphere diameter $\sqrt[3]{6V/\pi}$ (3D): same name, different
  formula in different tools.
- **Convex hull**: smallest convex set containing the object. For a digital object take
  the hull of the pixel **corners** (or of the pixel centres: a convention that changes
  small objects a lot; with centres a 1-pixel-wide line has hull area 0). Algorithm in
  words: Andrew's monotone chain on boundary points, $O(n \log n)$. **Convex area**,
  **solidity** = $A / A_{\text{hull}}$ (1 for convex, low for stars, crescents, clumps
  of two touching nuclei: a classic "should I split this?" feature).
- **Form factor / circularity** $4\pi A / P^2$: 1 for a disc, smaller for elongated or
  ragged boundaries. Inherits the perimeter bias: with edge-count perimeter a digital disc
  gives $\pi/4 \approx 0.79$; with a corrected estimator small discs give values **above
  1**. Rule: compare form factors only within one tool and one resolution. **Compactness**:
  name collision. CellProfiler's is $2\pi\,\overline{r^2}/A$ (mean squared distance of
  pixels from the centroid, 1 for a disc, from second moments, no perimeter); others use
  $P^2/(4\pi A)$ (= 1/form factor); blockflow's 3D `compactness` is $S^3/(36\pi V^2)$,
  the 3D inverse form factor. Show all three on the same shapes.
- **Feret diameters** (caliper diameters): the width $w(\theta)$ of the object between
  two parallel support lines at angle $\theta$. **Max Feret** = the diameter of the point
  set; **min Feret** = the smallest width (what a sieve or a flow cytometer "sees").
  Both are attained on the convex hull; **rotating calipers** (Shamos, Toussaint) find
  them in $O(h)$ for $h$ hull vertices: min width is attained with one caliper flush with
  a hull edge, so only $h$ angles need checking. Small worked example by hand on a
  5-vertex polygon. Caveat: pixel-centre vs pixel-corner hull again; brute force over all
  pixel pairs is $O(n^2)$ (why tools restrict to the hull).
- **Radii from the distance transform** (recall from the watershed/morphology day):
  $D(x)$ = distance to the background. **Maximum inscribed radius** $\max D$ (largest
  disc that fits: thickness of the object, insensitive to protrusions); **mean / median
  radius** = mean / median of $D$ over the object. A rod of width $w$ has max radius
  $\approx w/2$ whatever its length; contrast with min Feret. Pointer: the largest disc
  is a point of the medial axis (skeleton day). Optional row: **minimum enclosing circle**
  (blockflow reports it): the smallest disc containing the object.
- Pitfalls slide: units ($\mu$m vs px: area scales with the square, ratios do not);
  **anisotropic pixels/voxels** (z-step ≠ xy pixel size: compute moments and distances
  in physical coordinates, not in index space); objects touching the **border** are
  truncated (area, Feret, shape all wrong; drop them or use the counting frame from
  `label-measure/`); **resolution**: a 10-pixel object has no meaningful form factor;
  **segmentation error**: a one-pixel dilation adds about $P$ pixels, i.e. a relative area
  change $\approx P/A = 2/r$ for a disc, large for small objects.
- **Demo `shapefeat`** (new, main demo): a 200×200 canvas with one object, made from a
  shape preset (disc, ellipse, rectangle, star, crescent, two touching discs, rod, L)
  that can be dragged, rotated and scaled, or drawn with a brush / edited by dragging the
  vertices of a polygon that is then digitised. Overlays (toggles): convex hull (corner or
  centre convention), the two calipers at the current angle with a draggable $\theta$
  handle and a small plot of $w(\theta)$ with min and max marked, the max and min Feret
  segments, the maximum inscribed circle, the minimum enclosing circle, the equivalent
  ellipse and the equal-area circle. Side panel: a live feature table (area, perimeter
  by the chosen estimator, convex area, solidity, extent, form factor, the three
  compactness variants, major/minor axis, eccentricity, equivalent diameter, min/max
  Feret, max/mean inscribed radius, Euler number) with a *units* toggle (px / µm with a
  pixel-size field, anisotropic option: separate x and y pixel size) and a column
  showing the exact value for analytic shapes. A *resolution* slider re-digitises the
  shape; a *mask error* slider (−2…+2 px of erosion/dilation) shows how every row moves.
  Presets: "disc: form factor 0.79 with edge-count perimeter", "small disc: form factor
  above 1", "star vs disc: same area, solidity tells", "two touching nuclei: solidity
  drops", "rod: min Feret ≈ 2 × inscribed radius, max Feret ≈ length", "crescent: hull
  vs object", "anisotropic pixels: the circle that is not", "1-pixel dilation on a small
  object".
- By hand: (a) for a given 6×6 object (pixel list), the convex hull of the pixel corners,
  its area (shoelace formula), solidity and extent; (b) min and max Feret of a 4–5 vertex
  polygon by checking the calipers flush with each edge; (c) the city-block / chessboard
  distance transform of a 5×7 object, max inscribed radius and mean radius; (d) form
  factor of a 4×4 square with the edge-count perimeter vs the true value $\pi/4$.

### 2. `intensity-features/`: Intensity inside an object

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
  Pointer to the bit-depth day for stored units and rescaling.
- **Edge intensity**: the same statistics on the object's boundary band (object minus its
  erosion by one pixel; recall from morphology). Ratio edge/mean distinguishes rim
  staining (membrane) from filled staining.
- **Intensity-weighted centroid** $\bar{\mathbf x}_I = \sum I(\mathbf x)\mathbf x / \sum I$
  vs geometric centroid; **mass displacement** = their distance (in µm). Large when the
  signal is polarised (one bright spot off-centre); 0 by symmetry for uniform and
  rim-stained discs. One line: it is the first intensity moment, i.e. the intensity
  version of the moments day.
- **Radial distribution** (CellProfiler `MeasureObjectIntensityDistribution`): divide
  the object into $n$ rings by **normalised distance** $\rho = d_c/(d_c + d_e)$ from the
  centre ($d_c$ = distance to the centre point, $d_e$ = distance to the edge, from the
  distance transform), so rings follow the object's shape instead of being circles.
  Per ring $k$: **FracAtD** = fraction of the total intensity in ring $k$; **MeanFrac** =
  FracAtD divided by the fraction of the object's pixels in ring $k$ (1 = uniform, >1 =
  enriched); **RadialCV** = coefficient of variation of the ring's intensity across
  angular sectors (asymmetry within a ring). Choice of centre (centroid, intensity max, a
  parent object such as the nucleus) matters; with a nucleus as centre this measures
  nuclear vs peripheral localisation (translocation assays).
- Per-object vs per-image measures: the same statistics over the whole image or a
  region; image-level saturation fraction and focus as quality-control columns (one
  slide).
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
  shows a population of 20 cells of two types and a scatter of two chosen features.
  Presets: "background offset: mean moves, std does not", "same integral, different
  pattern", "rim vs filled: edge/mean ratio", "polarised cell: mass displacement",
  "nuclear vs cytoplasmic: radial bins", "dilated mask dilutes the mean", "saturation".
- By hand: (a) a 5×5 object with given intensities: integrated, mean, median, quartiles,
  MAD, edge-band mean (after one erosion with the cross); (b) intensity-weighted centroid
  and mass displacement; (c) normalised radial position of three marked pixels from
  given $d_c$, $d_e$, assign them to 3 rings, compute FracAtD and MeanFrac for a tiny
  example.

### 3. `texture-features/`: Texture: co-occurrence matrices and Haralick features

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
  (uniformity), **entropy** $-\sum p\log p$, **correlation**
  $\sum (i-\mu_i)(j-\mu_j)p / (\sigma_i\sigma_j)$ (Pearson correlation of a pixel with its
  neighbour: the autocorrelation at lag $\mathbf d$, recall from Fourier). Name the rest
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
  **granularity** (opening spectrum; owned by the morphology day, CellProfiler
  `MeasureGranularity`). Modern pointer: CNN embeddings as learned texture features.
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

### 4. `relations-coloc/`: Measurements between objects and between channels

- Two kinds of "between" measurements: between **objects** (neighbours, parents and
  children) and between **channels** (colocalisation). Both need a definition of *which*
  pixels or objects are paired, and both are easy to over-interpret.
- **Neighbours**. Three definitions, three answers: **adjacent** (labels share a
  boundary, 4/8-connectivity), **within distance $r$** (boundaries or centroids closer
  than $r$), **expand until adjacent** (grow every object into the background up to the
  midline between objects, i.e. a Voronoi tessellation of the objects via the distance
  transform; recall watershed). Per object: **number of neighbours**, **percent
  touching** (fraction of the boundary pixels adjacent to another object), **first and
  second closest** object and their distances, **angle between** the two closest
  neighbours. The **neighbour graph** (objects as nodes; link to the graph-cut day's
  graphs and the labelling day's region adjacency). Population summaries: the
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
  the thresholding day's evaluation).
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
    bleed-through (pointer to the colour day's unmixing), resolution (two proteins in
    the same 200 nm diffraction spot "colocalise" at any distance below it). Pointer:
    object-based colocalisation (fraction of spots in A whose centroid lies in an object
    of B) reuses the parent–child relation above.
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
- By hand: (a) adjacency list and number of neighbours from a 6×6 label image (4- and
  8-connectivity); (b) nearest-neighbour distances of 5 given centroids; (c) Pearson,
  overlap coefficient, M1 and M2 (with given thresholds) for two 3×3 channels; (d) which
  spots belong to which nucleus in a small label pair, by overlap and by centroid.

### 5. (optional / further reading) `features-use/`: From a feature table to an answer

- **Physical units** and **calibration** in one table (px → µm, px² → µm², voxel
  anisotropy); keep dimensionless ratios dimensionless.
- **Redundancy**: area, perimeter, equivalent diameter, convex area, max Feret are
  strongly correlated; many Haralick features too. A **correlation matrix** of the
  feature table (heat map, clustered) shows the blocks; pick one per block or use PCA.
- **Normalisation and aggregation** (image-based profiling, Caicedo et al. 2017): per
  object → per image / per well by median (robust) or mean, plus counts and fractions
  ("% cells with > 3 spots"); per-plate normalisation as **z-scores against control
  wells**, robust z with median and MAD; plate and batch effects (row/column gradients)
  in one picture.
- **Downstream**: filtering (labelling day), PCA scatter of objects or wells (labelling /
  snakes day), clustering, and nearest-mean classification (moments day `moment-features/`);
  feature selection and the danger of picking from hundreds of features on few samples
  (one slide on overfitting, no further ML).
- **Demo `featurespace`** (small, new): the `neighbours` synthetic population extended to
  a "plate" of 4×6 wells with two seeded phenotypes and a plate gradient; panels:
  correlation heat map of ~25 features, PCA scatter (objects or wells, coloured by
  treatment), and a well plate view of one aggregated feature before and after robust
  z-score normalisation. Presets: "area-like features form one block", "plate gradient
  hides the phenotype", "robust z fixes it", "median vs mean with debris".
- By hand: (a) z-score and robust z-score of a feature for 5 wells given controls;
  (b) read a small correlation matrix and choose which features to keep.

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
| Hu / Zernike | `Zernike_n_m` | Hu, weighted Hu, Zernike, 3D Zernike rows | moments day |
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

- `demos/src/lib/features.ts` (new, owned by this day): one function per catalogue row
  over a label image + optional intensity image(s) + pixel size $(s_x, s_y)$, returning
  a table (one row per label) with the column definitions and units used on the catalogue
  slides. Builds on `lib/label.ts` (labelling day: area, centroid, bbox, moments to order
  2, perimeter estimators, Euler number, border flags) rather than recomputing them;
  adds equivalent diameter, form factor, the three compactness variants, convex
  area/solidity, Feret, radii from `lib/distance.ts`, intensity statistics (exact
  median/quartiles/MAD by sorting per object), edge band (erosion from `lib/morph.ts`),
  weighted centroid and mass displacement, radial bins (normalised distance from two
  distance transforms), local background ring. Zernike/Hu columns optional via
  `lib/moments.ts` (moments day).
- `demos/src/lib/convexhull.ts` (new): monotone-chain hull of pixel corners or centres,
  shoelace area, rotating calipers (min width with the attaining edge, max diameter with
  the antipodal pair, $w(\theta)$ for plotting), minimum enclosing circle (Welzl, seeded
  shuffle). Also usable by the skeleton and snakes days.
- `demos/src/lib/glcm.ts` (new): quantisation (range, levels), GLCM for an offset within
  a mask, symmetrise/normalise, the Haralick features (all 13 for the table, with the
  alternative definitions flagged), LBP codes and histogram.
- `demos/src/lib/neighbours.ts` (new, small): adjacency pairs and boundary-contact
  counts from a label image (neighbourhood tables from `lib/label.ts`; coordinate with
  `lib/graph.ts` from the graph-cut plan), expansion to the midline via
  `lib/distance.ts` (feature transform / nearest label), centroid nearest neighbours
  (brute force is fine for ≤ 200 objects), parent assignment by overlap or centroid;
  colocalisation coefficients and Costes threshold search (or `lib/coloc.ts` if it grows).
- Reused, not duplicated: `lib/label.ts`, `lib/moments.ts`, `lib/distance.ts`,
  `lib/morph.ts`, `lib/imageops.ts` (image loading, seeded synthetic populations,
  `labelColors`), `lib/shapes.ts` (nucleus outlines), `lib/histogram.ts` (thresholding
  day: histogram drawer for `intensityfeat`, Otsu for Costes' comparison),
  `lib/plot.ts` (bars, polar plot, scatter, heat map: add a heat-map helper there if
  missing), `lib/linalg.ts` (symmetric eigen for PCA in `featurespace`), `rng(seed)`
  from `lib/ui.ts`.
- **Language / speed**: TypeScript throughout. Estimates: all features of ~80 objects in a
  256×256 image in a few ms (one labelling scan, two distance transforms, sorting per
  object); a GLCM of a 128×128 patch is one pass; Costes' search is ~50 Pearson
  evaluations; the randomisation test (50 block scrambles of 256×256) a few tens of ms,
  run on button press. No Rust/Wasm expected; measure first.
- Tests (`make test`): convex hull vs brute force (all point triples) on random point
  sets; rotating-calipers min width and max diameter vs brute force over 3600 angles /
  all pairs; shoelace area of a digitised disc's corner hull → $\pi r^2$; form factor of
  an analytic disc polygon = 1; radial bins: FracAtD sums to 1, MeanFrac = 1 for a
  uniform object; GLCM of a constant image has ASM = 1, contrast = 0, and a symmetric
  GLCM equals its transpose; Pearson of $B = aA + b$ is 1; M1 = M2 = 1 when $A = B$;
  `features.ts` order-2 columns agree with `lib/label.ts` and `lib/moments.ts`.
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
| 09:00–10:30 | Unit 1 lecture, hull and calipers by hand, `shapefeat` |
| 10:45–12:15 | Unit 2 lecture + exercise (`intensityfeat`) |
| 13:15–14:45 | Unit 3 lecture, GLCM by hand, `glcm` |
| 15:00–16:30 | Unit 4 lecture + exercise (`neighbours`, `coloc`); quizzes as wrap-up |

## Open questions

- How long is a "day" (hours, lecture vs exercise)? Four dense units; unit 4 bundles
  neighbours and colocalisation and could split into two units (then texture or
  colocalisation becomes optional). My recommendation: keep colocalisation core (it is
  the most misused measurement in practice) and shorten texture to GLCM only.
- `shapefeat` as a new demo or as `data-mode="features"` of the labelling day's
  `moments` demo (which already has shape presets, dragging and a resolution slider)?
  A mode avoids duplicating the shape-editing UI; a separate bundle keeps both small.
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
- Should 3D measurements (volume, surface area from voxel faces vs marching cubes,
  sphericity, anisotropic voxels) get an optional unit, linking to the marching-cubes day?
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
