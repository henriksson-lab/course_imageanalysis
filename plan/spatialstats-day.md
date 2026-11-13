# Plan: a day on spatial statistics of objects (point patterns)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.
Follows `plan/coursemap.md`, section "Queue, second batch" (requested 2026-11-13): **Spatial
statistics of objects**, elective group **"Objects"** (with skeletons and moments). **Entry
after:** day 9 (Labelling: objects and their centroids) and the point-detection day (spots and
sub-pixel positions). This plan proposes to move the entry to **day 11 (Measurements)**, whose
`relations/` unit already teaches the Poisson nearest-neighbour law and Clark–Evans (see
"Proposed reorganisation"). This day **owns** point-pattern statistics: complete spatial
randomness (CSR) as the null model, quadrat counts, the G, F and J functions, Ripley's K and L,
the pair-correlation function g, edge corrections, Monte Carlo envelopes and global tests,
multitype (cross) functions and the two null models for them, and inhomogeneous intensity.

## Goal of the day

After the day a student can:

1. turn a label image or a spot list into a **point pattern** (positions + an observation
   **window** + optional **marks** such as cell type or size), and name what the detection step
   does to it (merged spots remove short distances, split objects add them, objects of finite
   size cannot be closer than their diameter, the window is not the image when the tissue does
   not fill it);
2. explain **complete spatial randomness** (the homogeneous Poisson process) as the null model:
   counts in any region are Poisson with mean λ·area, regions are independent, and given the
   number of points they are scattered uniformly; simulate it; tell **clustered, random and
   regular** patterns apart, and know that the eye sees clusters in random patterns;
3. compute **quadrat counts** and the **variance-to-mean ratio** by hand, and explain why the
   answer depends on the quadrat size (the scale question that the K function answers);
4. compute and read the **nearest-neighbour distance function G** and the **empty-space
   function F**, say which way each moves for clustered and regular patterns, and read the
   **J function** (J < 1 clustered, J > 1 regular);
5. explain the **edge effect** and compare the corrections: guard zone (border / reduced
   sample), translation and isotropic weights; predict the sign of the bias without a
   correction;
6. compute **Ripley's K** from a small pattern, transform it to **L(r) − r**, and read the
   **pair-correlation function g(r)**; explain why K is cumulative (it remembers small scales)
   while g is local, and read a pattern that is **clustered at one scale and regular at
   another** (cells packed in colonies; hard-core repulsion inside them);
7. run a **Monte Carlo test** against CSR: simulate, compute the same curve, compare; explain
   why a **pointwise envelope** used over all r is a **multiple test** (global false alarm rate
   far above the nominal 5 %), and use a **global** test (maximum deviation over a pre-chosen
   r range, or a global envelope) instead; say what n is when comparing **conditions** (images
   within experiments, recall the evaluation day);
8. analyse a **multitype pattern** (two cell types) with the **cross-K / cross-L** function,
   and choose the right null model: **independence** (shift or relocate one type) vs **random
   labelling** (permute the type labels on fixed positions); they answer different questions;
9. recognise **inhomogeneous intensity** (a density gradient, a tissue region) as a
   first-order effect that the plain K function reports as clustering; use the tissue mask as
   the window, the **inhomogeneous K** with an estimated λ(x), or random labelling, and state
   the limit (from one pattern, an intensity gradient and clustering cannot be fully separated);
10. *(math)* derive $G(r) = F(r) = 1 - e^{-\lambda\pi r^2}$ and $K(r) = \pi r^2$ under CSR,
    $g = K'/(2\pi r)$, the edge-corrected estimators as Horvitz–Thompson-type weighted sums, the
    K function of a Thomas cluster process, the Monte Carlo rank p-value, and the
    random-labelling identity $K_{11} = K_{22} = K_{12} = K$.

Why it fits this audience: the mathematics is probability they know (Poisson and binomial
counts, expectations, a CDF, a derivative) applied to positions instead of pixels. What is new
is the **second-order** question ("how do points relate to each other at distance r?") and the
**null model by simulation**, which the colocalisation and evaluation days introduce in other
settings. Every curve of the day is "count something within a radius r, and compare with what
random points would give", which makes the day unusually uniform: one idea, five lenses.

## Prerequisite gap

What the built units already teach, and what this day recalls in one slide or one line:

- **`relations/`** (day 11, built): three definitions of "neighbour", Voronoi neighbours by
  expansion, the nearest-neighbour distance histogram against the Poisson curve, **Clark–Evans**
  $R$ (with its demo numbers: clustered 0.73, random 1.11, regular 1.86), the math slide "The
  Poisson nearest neighbour" ($P(D > r) = e^{-\rho\pi r^2}$, $E[D] = 1/(2\sqrt\rho)$), and a
  border slide (guard zone of 24 px in the `neighbours` demo). Unit 2 starts **from** that slide:
  G is the CDF of exactly that histogram, and Clark–Evans is G summarised by one number. The
  notation changes from ρ (relations) to λ (the spatial-statistics convention); one line says
  so. If the entry stays at day 9, unit 2 must restate the law in one concept slide and one
  math slide (it is short).
- **`label-measure/`** (day 9): border objects and the unbiased counting frame. Unit 2's edge
  corrections are the same worry for **distances** instead of counts; one recall line.
- **`regionprops/`** (day 9): centroids. **`spot-filters/`, `spot-threshold/`, `spot-fit/`**
  (point detection): spot positions with sub-pixel precision. **`spot-crowd/`**: two spots
  closer than about the PSF width merge into one. Unit 1 recalls all of it in one slide ("where
  the points come from") and turns the merging into a pattern artefact (missing short
  distances = apparent regularity).
- **`morph-openclose/`** (day 8): the distance transform. Used for the distance from each point
  to the window border (border correction) and for F (distance from every pixel to the nearest
  point = the distance transform of the point image). One line each.
- **`coloc-null/`** (colocalisation elective): randomisation, the permutation p-value, "blocks,
  not pixels". **`coloc-objects/`**: the chance level $1 - e^{-\rho_B\pi d^2}$, relocation of
  spots inside the cell mask, the cross-K function (one concept slide, one math slide, no edge
  correction) with a 39-relocation envelope. Unit 5 generalises; if a student has done the
  colocalisation day, the cross-L tab of `colocobjects` is a familiar picture. Not required.
- **`replicates/`** (day 16, built): the p-value by simulation, the unit of replication,
  pseudoreplication, multiple comparisons (Bonferroni, Benjamini–Hochberg). Day 16 comes **after**
  this elective in the core order, so unit 4 introduces the Monte Carlo p-value from scratch in
  the concept layer (it is the simplest possible null-by-simulation, one slide) and points
  forward to `replicates/` for the experiment-level statistics; a student who did day 16 first
  reads it as a recall.
- **`leakage/`** (day 16): spatial autocorrelation of pixels. One line in unit 4: "points in one
  image are not independent observations of the treatment, for the same reason".
- **`track-analysis/`** (tracking elective): the MSD compared with a random walk is the same
  logic in time (null model = pure diffusion). One pointer line in unit 4's summary.
- **Probability**: the Poisson distribution is recalled from `cameranoise/` (photon counts) with
  the same picture ("rare independent events in a region"); the binomial is from audience
  knowledge.

## Overlaps with other plans and units (who takes what)

| Plan / unit | What this day takes or teaches | What stays there |
|---|---|---|
| Measurements `relations/` (built) | Nothing moves. G, F, J, K, L, g and the envelope tests are taught here; unit 2 recalls the Poisson NN law and Clark–Evans | The per-object relation columns, the neighbour definitions, Clark–Evans and the Poisson NN law (its math slide stays as the first encounter). Proposed pending edit: one forward line on "Clustered, random or regular?" pointing here |
| Colocalisation `coloc-objects/` (built) | The **general** K/L/g theory, edge corrections and global envelopes; the two null models for two types | Object-based colocalisation, the chance level, the cross-K slide as an **application** with a pointer here (pending edit, one line), the distance bias below the resolution |
| Colocalisation `coloc-points/` (optional, unbuilt) | Its "cross-K/L/pair-correlation with edge correction" part | Coordinate-based colocalisation per localisation (Malkusch et al.), 3D chromatic shift (see "Proposed reorganisation") |
| Point detection `spot-crowd/` (built), optional `smlm/` (unbuilt) | Merging as a source of apparent regularity (unit 1, one slide); repeated blinks as a source of apparent clustering (optional unit 6, one slide) | Detection, matching, crowding; the SMLM pipeline itself |
| Evaluation `replicates/` (built) | Monte Carlo test for one pattern, the multiple-testing caveat of pointwise envelopes; one slide on comparing conditions with one summary per image | Unit of replication, pseudoreplication, CIs, Bonferroni/BH, batch effects (recalled, not retaught) |
| Labelling `label-measure/` (built) | Edge effects for distances (unit 2) | Counting frame for counts |
| Tracking `track-analysis/` (built) | — | MSD vs random walk (pointer only, same null-model logic in time) |
| Gaussian mixtures `kmeans/`, `gaussian-mixture/` (built) | Optional unit 6 contrasts "is there clustering?" (this day) with "which points belong to which cluster?" (k-means, DBSCAN) | k-means and mixtures |

## Units (one folder each: lec, quiz, exercise with #explore)

Five core units plus one optional. Units 1 and 2 are shorter (about 1 h each), units 3–5 about
1.5 h each (timetable below). Order: the null model and intensity (first-order) → distances to
the nearest point (the simplest second-order view, extending `relations/`) → all pairs at all
distances (K, g) → is it significant? (envelopes) → two types and an uneven density (marked and
inhomogeneous). Each deck about 14–22 content slides including math slides; about half carry a
figure or a demo (listed per unit).

### 1. `point-patterns/`: Point patterns and complete spatial randomness

Concept layer:

- **From objects to points** (opening, figure): a micrograph with nuclei → labels → centroids;
  a spot image → detections. A **point pattern** = positions + the **window** $W$ in which we
  looked + optionally **marks** (cell type, size, intensity). The window is part of the data:
  "no points here" means something only where we looked (tissue mask, not the whole image).
- **Where the points come from** (one recall slide with pointers to `regionprops/`,
  `spot-filters/`, `spot-crowd/`): every detection error becomes a pattern artefact. Merged
  spots → no short distances → looks **regular**. Over-split nuclei → pairs at tiny distances →
  looks **clustered**. Objects of diameter $D$ cannot have centres closer than about $D$ → a
  **hard core** (the reason `relations/` found $R = 1.11$ for "random" nuclei). Figure: the same
  true positions, detected three ways.
- **Three archetypes** (figure: three panels, same n): **clustered** (cells in colonies, receptor
  clusters), **random**, **regular** (cells that keep a distance, epithelia, photoreceptors).
  Humans are poor judges: a random pattern looks clumpy to most people (the demo's "which one
  is random?" quiz).
- **Complete spatial randomness (CSR)** = the homogeneous **Poisson process** with intensity λ
  (points per µm²). Two properties, each with a picture: (i) the count in any region $A$ is
  Poisson with mean $\lambda|A|$; (ii) counts in non-overlapping regions are independent. A
  consequence that makes simulation trivial: **given** n points in $W$, they are n independent
  uniform positions (the "binomial process"). Recall the Poisson distribution from
  `cameranoise/` (photons are a Poisson process in time).
- **Why CSR is the null model**: it is "no interaction and no preferred place". It is almost
  never the truth for cells; rejecting it is the start of the analysis, not the result. The
  useful question is **how** and **at which scale** the pattern departs from CSR.
- **First-order vs second-order** (one slide, figure): an uneven **intensity** λ(x) (more cells
  near a vessel) and **interaction** between points (cells attract or repel each other) both
  make a pattern look clustered. They are different questions; units 2–4 assume a constant λ,
  unit 5 relaxes it.
- **Quadrat counts** (figure: a grid over each archetype with counts written in): divide $W$
  into $m$ equal squares, count. Under CSR the counts are Poisson, so **variance ≈ mean**. The
  **index of dispersion** $\text{VMR} = s^2/\bar n$: > 1 clustered, ≈ 1 random, < 1 regular.
  Worked example with numbers (100 points, 25 quadrats).
- **The answer depends on the quadrat size** (demo): colonies of 10 µm look clustered with 10 µm
  quadrats and random with 1 µm or 100 µm quadrats. One number cannot describe a pattern that has
  structure at several scales: motivation for the functions of r in units 2–3.
- **Estimating the intensity** (one slide): $\hat\lambda = n/|W|$; a map of λ(x) by **kernel
  smoothing** (a Gaussian blur of the point image, recall `kernels/`; bandwidth = the σ of the
  blur; too small → a spike per point, too large → a constant). Divide by the blurred window to
  correct the edge. Used again in unit 5.
- **Models that are not CSR**, one picture each, so that the demos have names: **Thomas /
  Matérn cluster** process (parents at random, each with a Poisson number of offspring scattered
  around it), **hard-core** process (no two points closer than $h$; Matérn or sequential
  inhibition), a **jittered lattice** (strongly regular), **inhomogeneous Poisson** (independent
  points, uneven density).
- Pointer: image libraries and statistics packages call the field "spatial point pattern
  analysis" (one line).

Math layer (`## The math: … {.math}` slides):

- *The Poisson process*: $N(A) \sim \text{Poisson}(\lambda|A|)$, independent for disjoint sets;
  conditional uniformity: $P(\text{point } i \in A \mid N(W) = n) = |A|/|W|$, independently;
  hence the simulation recipe (draw $N \sim \text{Poisson}(\lambda|W|)$, then N uniform points;
  in a mask: rejection sampling in the bounding box).
- *Quadrat test*: with m quadrats and counts $n_k$, $X^2 = \sum_k (n_k - \bar n)^2/\bar n =
  (m-1)\,\text{VMR}$ is approximately $\chi^2_{m-1}$ under CSR (100 points, 25 quadrats: reject at
  5 % if $X^2 > 36.4$, i.e. VMR > 1.52 or, two-sided, small values for regularity).
- *Cluster process*: parents Poisson(κ), each with Poisson(μ) offspring displaced by
  $N(0, \sigma^2 I)$: intensity $\lambda = \kappa\mu$; the superposition picture.
- *Kernel intensity*: $\hat\lambda(\mathbf x) = \sum_i k_\sigma(\mathbf x - \mathbf x_i) / e(\mathbf x)$
  with the edge factor $e(\mathbf x) = \int_W k_\sigma(\mathbf x - \mathbf y)\,d\mathbf y$.

Figures (about 9 of 17 slides): objects → points pipeline (synthetic nuclei and spots, three
panels); detection artefacts (three panels); the three archetypes; Poisson counts in regions (a
pattern with two regions outlined and the Poisson pmf next to it); quadrat grids with counts;
VMR against quadrat size for the cluster pattern (curve); kernel intensity at three bandwidths;
the four model processes (2×2); demo screenshot fallback.

**Demo `csrplay`** (new): a 256×256 window (or a tissue-shaped mask), seeded patterns from the
models above with sliders for their parameters (n or λ; cluster: parents κ, offspring μ, spread
σ; hard-core: h; lattice: jitter; inhomogeneous: gradient strength); a *from an image* source
that runs labelling on a synthetic nuclei image (`lib/synth.ts` `nuclei`) or spot detection on a
synthetic spot image (`lib/spotdetect.ts` `spotImage` + `detectSpots`) and shows image and
centroids. Panels: the pattern; quadrat grid overlay with counts, and a side histogram of
quadrat counts with the Poisson pmf of the same mean; readout VMR, $X^2$, df, p (from the χ²
CDF); a small plot of VMR against quadrat size (computed for 8 sizes). Tab *intensity*: kernel
estimate with a bandwidth slider. Tab *which is random?*: three patterns side by side (one CSR,
one weakly clustered, one weakly regular, shuffled per seed), the student clicks, the demo
reveals. Presets: "CSR: variance = mean", "colonies: VMR > 1", "regular: VMR < 1", "quadrat
size changes the verdict", "merged spots look regular", "split nuclei look clustered", "which one
is random?", "gradient: clustered by quadrats, but no interaction", "intensity bandwidth".

By hand (concept): (a) quadrat counts from a printed 20-point pattern on a 4×4 grid: mean,
variance, VMR, verdict; (b) λ = 0.02 cells/µm² in a 50×50 µm window: expected count, and the
probability that a 5×5 µm square is empty ($e^{-0.5} = 0.61$); (c) for each of four described
situations (dividing cells that stay together, cells with a minimum spacing, cells near a
vessel, merged spots), predict whether quadrats say clustered or regular and why;
(d) predict the VMR of a cluster pattern with quadrats much smaller and much larger than the
clusters. *(math)*: (e) show that conditional on n the points are uniform (two disjoint regions,
Poisson splitting); (f) $X^2$ for (a) and the decision at 5 %; (g) the intensity of a Thomas
process and the expected count in a quadrat.

### 2. `nn-functions/`: Nearest neighbours and empty space: G, F, J and the edge

Concept layer:

- **Recall** (one slide, from `relations/` "Clustered, random or regular?" and its math slide):
  the nearest-neighbour histogram against the Poisson curve, Clark–Evans $R$. Clark–Evans is one
  number; it hides **at which distance** the pattern differs. Notation: λ here = ρ there.
- **G(r)**: the fraction of points whose nearest neighbour is within $r$ (the cumulative
  histogram of the nearest-neighbour distances; recall cumulative histograms from
  `histogram/`). Under CSR: $G(r) = 1 - e^{-\lambda\pi r^2}$. Clustered: G rises **early**
  (neighbours are close); regular: G is 0 up to the minimum spacing and then rises **steeply**.
  Figure: G for the three archetypes against the CSR curve.
- **F(r)**, the **empty-space** function: from a random **location** (not a point), the
  distance to the nearest point. Under CSR the same curve as G (a Poisson point does not know
  about the others). Clustered: large empty gaps → F rises **late**; regular: no big gaps → F
  rises early. **F is the distance transform of the point image** (recall `morph-openclose/`):
  its histogram over the window. Figure: the distance-transform map with a few test locations.
- **G and F move in opposite directions**; their ratio summarises both: $J(r) = (1 - G(r))/(1 -
  F(r))$: **J = 1** under CSR, **J < 1** clustered, **J > 1** regular, and it needs no λ.
- **What G cannot see**: only the nearest neighbour counts. A pattern of tight pairs scattered at
  random has an extreme G and then looks random at every larger scale; G cannot tell you how
  big the clusters are. Motivation for K (all neighbours, unit 3). Figure: pairs pattern and its G.
- **The edge effect** (figure: a point near the border whose true nearest neighbour is outside
  the window): near the border we miss neighbours, so observed nearest distances are **too
  long**, G is biased **down** and the pattern looks more regular than it is. It grows with r
  and with the share of points near the border (small windows, small n, elongated or
  irregular tissue masks).
- **Corrections**, as pictures:
  - **border / reduced sample** (the guard zone of `relations/`): use only points farther than r
    from the border when estimating G(r); unbiased, but throws away data, more at large r.
  - **Kaplan–Meier** (one line): treat "distance to the border" as censoring; named, not taught.
  - **translation and isotropic weights** (unit 3) for pair-based functions.
  - **toroidal wrap** (one line): only for simulated patterns in a rectangle; never for real
    tissue.
- **Distances in µm**, not px (anisotropic pixels, z-steps in 3D: one line pointer to
  `volumes3d/`).

Math layer:

- *CSR*: $1 - G(r) = P(\text{no other point in } b(\mathbf x, r)) = e^{-\lambda\pi r^2}$ (Slivnyak:
  the other points of a Poisson process seen from a typical point are again Poisson); the same
  for F from a fixed location; hence $J \equiv 1$; mean NN distance $1/(2\sqrt\lambda)$ (the
  Clark–Evans denominator).
- *Reduced-sample estimator*: $\hat G(r) = \#\{i: d_i \le r,\ b_i \ge r\} / \#\{i: b_i \ge r\}$
  with $b_i$ the distance to the border; why it is unbiased (each kept point sees its full
  disc) and why it is not monotone in r.
- *J for a cluster process* (in words + the formula for the Thomas process stated): J < 1 at all
  r, approaching a constant; for a hard-core process J > 1.
- *Edge bias without correction*: a point at distance $b < r$ from a straight border sees only
  part of its disc (area of a circular segment); the expected loss for the whole window is of
  order $(\text{perimeter} \cdot r) / \text{area}$.

Figures (about 7 of 14 slides): G of the archetypes; F with the distance transform; J curves;
pairs pattern with G and the missing scale; the edge picture; guard zone; demo fallback.

**Demo `gfj`** (new): uses the patterns of `csrplay` (shared source panel, same presets and
seeds). Panels: the pattern with arrows to each point's nearest neighbour (the `relations/`
picture) and, in F mode, a grid of test locations with arrows to the nearest point over the
distance-transform map; the curve panel with $\hat G$, $\hat F$ or $\hat J$ (tabs) and the CSR
curve. Controls: correction *none / border (reduced sample)* with the guard zone drawn; window
*full / small / tissue mask*; click a point to highlight its distance and whether the border
rule drops it at the current r (r slider shown as circles). Presets: "CSR: G = F = theory",
"clusters: G up, F down", "regular: the opposite", "J needs no λ", "tight pairs: G cannot see
the cluster size", "small window: no correction makes it look regular", "border correction
fixes it", "elongated tissue mask: most points are near the border".

By hand (concept): (a) six given points (coordinates on a grid): NN distances, $\hat G$ at r =
1, 2, 3; (b) λ = 0.01 µm⁻²: $G(5\,\mu m)$ under CSR ($1 - e^{-0.785} = 0.54$); an observed 0.80 —
which way? (c) F from a printed distance-transform map of 5 points at 8 test locations;
(d) which of the six points does the border rule keep for r = 2 in a 10×10 window; (e) predict
the sign of the bias of an uncorrected G in a 20×200 µm strip of tissue. *(math)*: (f) derive
$G(r) = 1 - e^{-\lambda\pi r^2}$ and the mean NN distance; (g) show $J = 1$ under CSR; (h) area
of the part of a disc of radius r outside a straight border at distance b.

### 3. `ripley-k/`: Ripley's K, the L function and the pair-correlation function

Concept layer:

- **Count all neighbours, not just the nearest**: for each point, how many other points lie
  within r? Average over points, divide by λ: that is **K(r)**. Under CSR the expected count
  within r is $\lambda\pi r^2$, so $K(r) = \pi r^2$. Figure: circles of radius r around three
  points with the counts.
- **L(r) = √(K(r)/π)** turns the parabola into a straight line ($L = r$ under CSR); plot **L(r) −
  r**: 0 random, > 0 clustered (more neighbours than expected at that distance), < 0 regular. It
  also stabilises the variance (one line). Figure: K and L − r for the archetypes.
- **The pair-correlation function g(r)**: count neighbours in a thin **ring** at distance r
  instead of a disc, divided by what CSR gives in the same ring. **g = 1** random, **g > 1** more
  pairs at that distance, **g < 1** fewer, **g = 0** impossible distances (a hard core). It is the
  derivative of K, made dimensionless. Picture: ring vs disc.
- **Cumulative vs local** (key slide, figure): K at r contains every pair closer than r, so a
  cluster at 5 µm keeps K − πr² high at 50 µm ("K remembers"). g answers "at this distance"
  only; it is noisier (thin rings hold few pairs) and needs a bandwidth (ring width), the
  same trade-off as histogram bins (recall `histogram/`).
- **Reading the scales** (figure: a Thomas pattern and its g): g starts high, falls to 1 at
  about the **cluster diameter** (≈ 4σ for Gaussian offspring); its height at small r says how
  many extra neighbours a point has in its own cluster. A hard-core pattern: g = 0 below h, a
  bump just above h (packing), then 1.
- **Clustering at one scale, regularity at another**: cells in colonies, each cell keeping a
  minimum distance from its neighbours (they have a size). g: 0 below the cell diameter, above 1
  up to the colony size, 1 beyond. G and the quadrats each see one of the two effects; g sees
  both. This is the day's central picture (demo preset).
- **Edge corrections for pair functions**: a pair whose circle around $\mathbf x_i$ sticks out of
  the window is under-counted. **Translation** weight: count the pair with weight $|W| / |W \cap
  (W + \mathbf x_j - \mathbf x_i)|$ (how much of the window could have contained this pair).
  **Isotropic** (Ripley) weight: 1 / (fraction of the circle through $\mathbf x_j$ centred at
  $\mathbf x_i$ that lies inside W). **Border**: only centre points farther than r from the
  border. Without a correction, K is biased **down** (looks regular); the simulation in the
  shared-engineering notes: −8 % at r = 0.1 and −20 % at r = 0.25 of the window side for 100
  CSR points in a square. Figure: the two weights drawn.
- **Range of r**: up to about a quarter of the shorter window side (beyond that every correction
  becomes unstable). Decide the range **before** looking (unit 4).
- **3D** (one line): $K(r) = \tfrac43\pi r^3$ under CSR, $L = (3K/4\pi)^{1/3}$; same ideas for
  stacks (pointer `volumes3d/`).

Math layer:

- *Definition*: $\lambda K(r) = E[\text{number of further points within } r \text{ of a typical
  point}]$; under CSR (Slivnyak) $= \lambda\pi r^2$.
- *Estimator*: $\hat K(r) = \dfrac{|W|}{n(n-1)}\sum_{i \ne j} w_{ij}\,[d_{ij} \le r]$ with
  translation weight $w_{ij} = |W| / |W \cap W_{\mathbf x_j - \mathbf x_i}|$ (for a rectangle
  $a \times b$: $ab / ((a - |\Delta x|)(b - |\Delta y|))$) or isotropic weight $w_{ij} = 2\pi
  d_{ij} / \text{length}(\partial b(\mathbf x_i, d_{ij}) \cap W)$. Why the weights make the sum
  unbiased (a Horvitz–Thompson argument: each pair is seen with a known probability, weight =
  1/probability). $n(n-1)/|W|^2$ as the estimate of $\lambda^2$.
- *Pair correlation*: $g(r) = K'(r)/(2\pi r)$; kernel estimator $\hat g(r) = \dfrac{|W|}{n(n-1)}
  \sum_{i\ne j} \dfrac{w_{ij}\, k_h(r - d_{ij})}{2\pi d_{ij}}$ (Epanechnikov kernel, bandwidth h).
- *Thomas process*: $K(r) = \pi r^2 + \dfrac{1 - e^{-r^2/4\sigma^2}}{\kappa}$, $g(r) = 1 +
  \dfrac{e^{-r^2/4\sigma^2}}{4\pi\kappa\sigma^2}$ (distance between two offspring of the same
  parent is Gaussian with variance $2\sigma^2$ per axis).
- *Variance of $\hat K$ under CSR* (stated, Ripley/Lotwick–Silverman order of magnitude) and why
  L stabilises it.

Figures (about 10 of 20 slides): circles and counts; K and L − r for the archetypes; ring vs
disc; K vs g on a cluster pattern; Thomas g with the cluster diameter marked; hard-core g;
colonies-with-hard-core pattern with its G, L − r and g side by side; the two edge weights;
bias curves with and without correction; demo fallback.

**Demo `ripleyk`** (new, main demo): the source panel of `csrplay` (shared presets), plus an
**editable** mode (click to add, drag, shift-click to remove points; up to 300). Panels: the
pattern with a draggable probe point and two concentric circles (disc of radius r for K, ring
r ± h for g), the count inside and the CSR expectation as a readout; the curve panel with tabs
*K / L − r / g* (CSR reference, the current r as a vertical line, the probe's own count curve
as a thin line); correction *none / border / translation / isotropic*. Small inset: the bias
of the four corrections averaged over 50 seeded CSR patterns (precomputed per window shape,
once). Sliders r, bandwidth h for g, model parameters. Presets: "CSR: L − r ≈ 0", "Thomas: g
falls to 1 at the cluster size", "K remembers: one small cluster scale, high K far out",
"hard core: g = 0 below h", "colonies with a hard core: two scales", "no correction: looks
regular", "isotropic vs translation", "build your own pattern", "tight pairs: G vs g".

By hand (concept): (a) five points with given pairwise distances (a table): $\hat K$ at r = 1, 2,
3 with no correction and the CSR value; L − r; verdict; (b) λ = 0.01 µm⁻², observed $\hat K(10\,
\mu m) = 500\ \mu m^2$: expected number of neighbours within 10 µm under CSR (3.1) and observed
(5.0); L − r (2.6 µm); (c) read cluster diameter and hard-core distance from a printed g curve;
(d) translation weight of a pair with Δx = 20, Δy = 10 in a 100 × 50 window (5000 / (80 · 40) = 1.56); (e) predict
L − r for a jittered lattice of spacing 20 µm at r = 10 and at r = 25. *(math)*: (f) $K = \pi
r^2$ under CSR and $g = K'/(2\pi r)$; (g) derive the Thomas K function from the offspring
distance distribution; (h) show that the translation-weighted estimator is unbiased for a
stationary pattern.

### 4. `envelopes/`: Is it more than chance? Simulation envelopes and global tests

Concept layer:

- **One observed curve, what would chance give?** Simulate CSR with the **same n in the same
  window** (with the same edge correction), compute the same function, repeat 99 times (seeded).
  The null distribution is drawn, not derived: this works for any function, any window shape,
  any correction. If the student has seen `coloc-null/` or `replicates/`, the idea is a recall
  (pointer).
- **Monte Carlo p-value for one pre-chosen r** (figure: histogram of 99 simulated L(5 µm) − 5
  with the observed value marked): rank of the observed among observed + simulated; p = (number
  at least as extreme + 1)/(99 + 1). With 99 simulations the smallest possible p is 0.01; with 19
  it is 0.05. No formula needed, valid by symmetry (exchangeability) when the observed pattern
  really is CSR.
- **The pointwise envelope** (figure): at every r the min and max of the simulated curves form
  a grey band; it is the picture in almost every paper (and in `coloc-objects/`).
- **The trap: many r at once.** Reading "the curve leaves the band somewhere" as significant is a
  **multiple test**: each r is a separate test, and with dozens of r one of them is likely to
  stray. Simulation for this plan (100 CSR points in a square, L with translation correction,
  25 radii up to a quarter of the side, 300 seeded trials): a 39-simulation pointwise envelope
  (5 % per r) is left somewhere by **about 36 %** of truly random patterns; a 19-simulation
  envelope (10 % per r) by **about 60 %**. Figure: 20 null patterns, the ones that "escape" in
  red. Recall the multiple-testing slide of `replicates/` (300 features, 15 hits) as the same
  problem.
- **Global tests** (figure): choose the r range **in advance**, summarise the whole curve by one
  number, e.g. the **maximum absolute deviation** $T = \max_{r}|\hat L(r) - r|$ (the MAD test),
  and rank that number among the simulations. One test, correct error rate (same simulation:
  about 3–4 % at nominal 5 %). The **global envelope** is the band whose width is set by this
  test: a curve leaving it **is** significant at 5 %, and where it leaves says at which scale.
  The rank envelope (Myllymäki et al.) in one line.
- **What a significant result means**: "not CSR" only. It does not say clustered by interaction
  vs inhomogeneous intensity (unit 5), nor which model fits (optional unit 6). Rejecting CSR for
  cells is nearly certain with enough points; report **how** and **at which scale**, with the
  effect size (e.g. g at its peak, the number of extra neighbours within the cluster radius).
- **Envelopes are not confidence bands** (one slide): the grey band describes the **null**, not
  the uncertainty of the observed curve. Uncertainty of the observed curve needs replicates
  (next bullet) or resampling (one line).
- **Comparing conditions** (figure: SuperPlot of a per-image summary): the points of one image
  are **not** independent observations of a treatment (recall `leakage/`). Compute one summary
  per image (e.g. L(r₀) − r₀ at a pre-chosen r₀, or the area between L − r and 0 over the
  chosen range), aggregate per experiment, test across experiments: exactly the
  `replicates/` recipe. Pooling several images of one condition into one K (weighted by the
  squared number of points) gives a better curve, not a larger n.
- **Choices that must be fixed before looking**: function, r range, correction, number of
  simulations, the window (tissue mask). Changing them until the result is significant is the
  same as tuning on the test set (pointer `leakage/`).

Math layer:

- *Monte Carlo test*: if the observed statistic $T_0$ and the simulated $T_1, \dots, T_s$ are
  exchangeable under $H_0$, the rank of $T_0$ is uniform on $\{1, \dots, s+1\}$, so
  $P(\text{rank} \le k) = k/(s+1)$ (Barnard 1963); ties broken at random.
- *Pointwise error*: two-sided min/max envelope from s simulations at one r has level $2/(s+1)$
  (0.05 for s = 39, 0.10 for s = 19). Over many r the events are dependent (neighbouring r share
  pairs), so the global rate is between the pointwise level and $1 - (1 - \alpha)^{m}$; only
  simulation tells (the 36 % / 60 % above).
- *MAD and DCLF tests*: $T = \max_{r \in [r_1, r_2]} |\hat L(r) - r|$, or the integrated
  $\int_{r_1}^{r_2} (\hat L(r) - r)^2 dr$ (Diggle–Cressie–Loosmore–Ford); more power when the
  departure is spread over many r. For a composite null (λ estimated from the data), simulate
  with the observed n (conditioning) and note the slight conservativeness (Baddeley et al. 2014).
- *Pooled K*: $\bar K = \sum_k n_k^2 \hat K_k / \sum_k n_k^2$ (images weighted by their number of
  pairs).

Figures (about 8 of 17 slides): null histogram at one r; pointwise envelope; 20 null patterns
with escapes marked; global rate vs number of radii (curve from the demo's precomputation);
MAD statistic as the largest vertical gap; global vs pointwise envelope on the same data;
SuperPlot of per-image summaries; demo fallback.

**Demo `envelopetest`** (new): source = the shared pattern panel (presets: CSR, weak clusters,
strong clusters, hard core, colonies). Panels: L − r (or g) with the observed curve, the
pointwise envelope and the global (MAD) envelope (toggle each), number of simulations s (19 /
39 / 99 / 199), r range handles (drag on the axis); readouts: pointwise "escapes at r = …",
MAD statistic, its rank and p. Tab *how often is CSR rejected?*: button runs 200 seeded CSR
patterns (progress shown) and counts how often the pointwise envelope is left somewhere vs how
often the MAD test rejects; plots the rate against the number of radii (precomputed curve
shown immediately, the live run fills in). Tab *conditions*: two simulated conditions × 4
experiments × 3 images (weak vs no clustering, with an experiment-to-experiment variation
slider); per-image L(r₀) − r₀ as a SuperPlot; tests on images-as-n vs experiment means (uses
`lib/stats.ts` `tTestWelch`). Presets: "CSR: inside the band", "weak clusters: escapes only at
small r", "pointwise trap: a random pattern escapes", "global envelope: no false alarm", "199
simulations: smoother band, same level", "choose the r range first", "images as n: tiny p",
"experiments as n: honest p".

By hand (concept): (a) 19 simulated values of L(5) − 5 and an observed value: rank and p
(one- and two-sided); (b) with 39 simulations, what is the pointwise level? Why does a curve
outside the band at one r of 25 not give p = 0.05? (c) given 10 simulated curves (a printed
table at 5 radii) and an observed curve: MAD statistic of each, rank, p; (d) design: three
experiments, two conditions, 5 images each, 200 cells per image: what is n, what is the
summary per image, which test; (e) predict how the pointwise false alarm rate changes when the r
range is doubled. *(math)*: (f) the rank of an exchangeable observation is uniform; (g) bounds
for the global error of m dependent pointwise tests; (h) why pooling K weights by $n_k^2$.

### 5. `marked-patterns/`: Two cell types and an uneven density

Concept layer:

- **Marks** (figure: a tissue with two cell types, e.g. immune cells and tumour cells, from a
  synthetic multiplexed image): each point carries a label (type) or a number (size,
  intensity). The questions: do the types **attract** (immune cells near tumour cells), **repel**
  (segregated domains), or ignore each other? Is a type's own pattern clustered?
- **Cross-K / cross-L / cross-g**: count type-2 points around type-1 points, compare with $\pi
  r^2$ (same functions as unit 3, one line of change). Recall the cross-L tab of
  `coloc-objects/` as the two-colour spot version.
- **Two null models, two questions** (key slide, figure with both simulations):
  - **Independence**: the two types are two separate processes (e.g. two cell populations that
    arrived separately). Null by **shifting** one type relative to the other (toroidal shift in a
    rectangle) or by **relocating** it inside the tissue mask (as `coloc-objects/` did); each type
    keeps its own pattern.
  - **Random labelling**: the positions are fixed (there are cells everywhere), and the question
    is whether the **types** are distributed at random over them (e.g. which cells become
    activated). Null by **permuting the labels**. Under it $K_{11} = K_{22} = K_{12} = K$; test
    $K_{11} - K_{22}$ or $K_{12} - K$. Positions that are themselves clustered do not disturb
    this test.
  - Picking the wrong null gives the wrong answer: in a tissue packed with cells, relocating
    type 2 to random free positions ignores that cells cannot overlap; permuting labels respects
    it. Demo preset.
- **Mark connection function** $p_{12}(r)$ (one slide): of the pairs at distance r, what fraction
  is a type-1/type-2 pair? Compared with the overall product of the type fractions. Easy to read
  for many types (a matrix of curves; figure). Continuous marks (size) in one line: the **mark
  correlation** (are neighbours of large cells large?).
- **Inhomogeneous intensity** (figure: cells denser near a vessel, no interaction at all): the
  plain K says "clustered" at every r because pairs are found more often where the density is
  high. Three responses:
  1. **the right window**: use the tissue mask, not the image (often the biggest "effect"
     disappears);
  2. **inhomogeneous K**: weight each pair by $1/(\lambda(\mathbf x_i)\lambda(\mathbf x_j))$ with
     λ(x) from the kernel estimate of unit 1 (or from a covariate: distance to the vessel); the
     same reading (L − r = 0 for "no interaction beyond the density");
  3. **random labelling** for typed patterns: it conditions on the positions and so ignores the
     density.
- **The honest limit** (one slide): from one pattern, "uneven density" and "clustering" cannot
  be fully separated. A bandwidth small enough to follow the clusters explains them away as
  density; a large one leaves them as interaction. Fix the bandwidth from a scale argued in
  advance (tissue structure, not cell clusters), or estimate λ from a covariate or from another
  cell type. Demo preset with the bandwidth slider.
- **Where this leads**: tissue-scale spatial analysis of multiplexed images (neighbourhood
  composition per cell, cell–cell interaction maps; one line); cluster finding (optional unit
  6); the colocalisation day for two-colour spots.

Math layer:

- *Cross-K*: $\lambda_2 K_{12}(r) = E[\text{number of type-2 points within } r \text{ of a typical
  type-1 point}]$; estimator $\hat K_{12}(r) = \dfrac{|W|}{n_1 n_2}\sum_{i \in 1}\sum_{j \in 2}
  w_{ij}[d_{ij} \le r]$; independence ⇒ $K_{12} = \pi r^2$ (for a stationary type 1, whatever
  type 2 does).
- *Random labelling*: if labels are assigned independently with probabilities $p_1, p_2$ to the
  points of a process with K function K, every pair is a 1–1, 2–2 or 1–2 pair with probabilities
  $p_1^2, p_2^2, 2p_1p_2$ regardless of distance, hence $K_{11} = K_{22} = K_{12} = K$ and
  $p_{12}(r) = 2p_1p_2$.
- *Inhomogeneous K* (Baddeley, Møller & Waagepetersen 2000): $\hat K_\text{inhom}(r) =
  \dfrac{1}{|W|}\sum_{i \ne j} \dfrac{w_{ij}[d_{ij} \le r]}{\hat\lambda(\mathbf x_i)\hat\lambda(\mathbf
  x_j)}$; equals $\pi r^2$ for an inhomogeneous Poisson process (second-order intensity
  reweighted stationarity, in words); leave-one-out $\hat\lambda$ to avoid a downward bias.
- *Toroidal shift*: preserves each type's second-order structure exactly in a rectangle;
  why it is invalid in an irregular tissue mask (relocation instead).

Figures (about 10 of 20 slides): synthetic two-type tissue; cross-L curves for attraction,
independence, segregation; the two nulls drawn as simulations; packed tissue where the nulls
disagree; mark connection matrix; density gradient with its plain K and inhomogeneous K; the
window effect (image vs mask); the bandwidth dilemma (three bandwidths, three verdicts); demo
fallback.

**Demo `celltypes`** (new): a synthetic tissue of 300–800 cells as points (optionally drawn as
discs with a hard core) in a tissue-shaped mask with an optional vessel; two (or three) types
from generators: independent, attraction (type 2 placed near type 1 with a radius and
strength), segregation (domains), random labelling of a clustered pattern, density gradient
towards the vessel. Panels: the tissue (types coloured with `lib/colour.ts` `OKABE_ITO`, the
mask drawn, the vessel); the curve panel with tabs *cross-L / L of each type / mark connection
/ inhomogeneous L*; envelopes from the chosen null (*shift / relocate in mask / permute labels
/ inhomogeneous Poisson from λ̂*) with the global (MAD) test and its p; the λ̂ map with a
bandwidth slider. Window toggle *image / tissue mask*. Presets: "attraction: cross-L above",
"independent: inside", "segregated domains: below", "packed tissue: relocation says repulsion,
labels say random", "density gradient: plain L says clustered", "tissue mask as the window",
"inhomogeneous L: no interaction", "bandwidth too small: clusters explained away", "random
labelling ignores the gradient".

By hand (concept): (a) a list of 4 type-1 and 5 type-2 points with given distances: $\hat
K_{12}$ at r = 2 and the CSR value; (b) for each of three biological questions, choose
independence or random labelling and say how to simulate it; (c) 30 % of cells are type 1:
$p_{12}$ under random labelling (0.42); an observed 0.20 at r = 10 µm means what? (d) the same
pattern analysed in the image window (200×200 µm) and in the tissue mask (half the area): what
happens to λ̂ and to the plain K? (e) predict the inhomogeneous L − r for a pure density
gradient. *(math)*: (f) random labelling ⇒ $K_{11} = K_{22} = K_{12}$; (g) the inhomogeneous K
of an inhomogeneous Poisson process is πr²; (h) why relocation inside the mask is a valid null
for independence but the toroidal shift is not.

### 6. (optional) `point-clusters/`: Fitting models and finding clusters

- **From "is it clustered?" to "how?"**: fit a **Thomas** process to an observed pattern by
  **minimum contrast** (choose κ, σ so that the model K matches $\hat K$; closed form from unit 3);
  read **cluster size** (≈ 4σ) and **points per cluster** (λ/κ). Then test the fitted model
  with the envelope of unit 4 (goodness of fit; the test is conservative because the model was
  fitted to the same data, one line).
- **Finding the clusters** (a different question): DBSCAN (density-based: a point with at least
  k neighbours within ε is a core point; clusters are chains of core points), contrasted with
  k-means (recall `kmeans/`: needs k, round clusters, every point assigned) and with the
  Voronoi-tessellation density (recall `relations/` "Expand until adjacent": small cells =
  dense). Parameters ε, k chosen from g (the cluster radius) rather than by eye.
- **Localisation microscopy pitfall** (one slide): one molecule blinking several times gives a
  tight "cluster" of localisations; uncorrected, every molecule looks clustered (pointer to the
  point-detection day's optional `smlm/` and to `spot-fit/` precision); remedies: merge blinks,
  compare with a pattern of single molecules, model the precision in the null.
- **Demo `clusterfit`** (new): Thomas pattern (or `ripleyk` source); tab *fit*: minimum-contrast
  fit with the fitted K over $\hat K$ and the fitted κ, σ vs the true ones; tab *find*: DBSCAN
  with ε, k sliders, clusters coloured, noise points grey; tab *blinks*: each molecule repeated
  Poisson(3) times with localisation error. Presets: "fit recovers the cluster size", "DBSCAN:
  ε from g", "k-means splits a cluster", "blinks make clusters".
- By hand: minimum contrast at two candidate σ with a printed K table; DBSCAN on 12 grid points
  with ε = 1.5, k = 3.

## Shared engineering

- **New `demos/src/lib/pointpattern.ts`** (owner: this day; no DOM; name checked against
  `demos/src/lib/`: no collision). Types: `Points {x: Float64Array, y: Float64Array, marks?:
  Int32Array}`, `Window` = rectangle `{w, h}` or mask `{mask, w, h, area, border: Float32Array}`
  (border distance by `edt` from `lib/distance.ts`, computed once per mask).
  - Simulators (seeded, `lib/random.ts`): `poissonProcess(λ, win, r)`, `binomialProcess(n, win,
    r)` (rejection in masks), `inhomPoisson(λfn, λmax, win, r)` (thinning), `thomas(κ, μ, σ, win,
    r)` and `maternCluster(κ, μ, R, …)` (parents in an expanded window to avoid edge thinning),
    `maternHardCore(λ, h, …)` (type II), `ssi(n, h, …)` (sequential inhibition with a give-up
    count), `jitteredLattice(spacing, jitter, …)`, `colonies(…)` (cluster + hard core),
    `twoType(…)` generators for unit 5 (independent, attraction, segregation, random labelling
    of a base pattern), `relabel(marks, r)` (permutation), `toroidalShift(pts, dx, dy, win)`,
    `relocateInMask(pts, win, r)`.
  - Summaries: `quadratCounts(pts, win, nx, ny)` → counts, VMR, $X^2$, p (needs a χ² CDF: add
    `chi2Cdf` via the regularised incomplete gamma; if `lib/stats.ts` gets it, import from there,
    see open question 5), `kernelIntensity(pts, win, σ, gridW, gridH)` with edge factor (uses
    `gaussianBlur` from `lib/imageops.ts` on a point histogram), `nnDistances` (reuse
    `nearestNeighbours` from `lib/neighbours.ts`), `Gest(pts, win, radii, corr)`, `Fest(pts, win,
    radii, corr)` (the distance transform of the point image, `edt`, sampled over the window),
    `Jest`, `Kest(pts, win, radii, corr: "none" | "border" | "translation" | "isotropic")`,
    `Lest`, `pcf(pts, win, radii, h, corr)` (Epanechnikov), `Kcross`, `pcfCross`,
    `markConnection`, `Kinhom(pts, win, radii, λ, corr)` (leave-one-out λ̂), `thomasK`,
    `thomasPcf`, `fitThomas` (minimum contrast on $K^{1/4}$, grid + golden-section search; optional
    unit).
  - Isotropic weight for rectangles (closed form from the circle–rectangle intersection) and
    for masks (perimeter sampling at 64 angles); translation weight for rectangles (closed form)
    and masks (overlap area of the mask with its shifted copy, cached per rounded offset using
    `correlateFFT` from `lib/fft.ts`: one FFT autocorrelation of the mask gives all of them).
  - Tests: `mcRank(obs, sims, side)`, `pointwiseEnvelope(curves)`, `madTest(obs, sims, ref,
    rRange)`, `dclfTest`, `globalEnvelope(sims, α)` (the MAD-based band), `pooledK(curves,
    counts)`.
- **Consolidation with `lib/coloc.ts`**: `crossK`, `crossLminusR` and `envelope` move into
  `lib/pointpattern.ts` (as `Kcross` with `corr: "none"`, `Lest`, `pointwiseEnvelope`), and
  `lib/coloc.ts` re-exports them under the old names so that `colocobjects` and
  `demos/test/colocalisation-test.ts` stay bit-identical. `chanceFraction` stays in coloc.
- **Reused, not duplicated**: `lib/neighbours.ts` (`nearestNeighbours`, `clarkEvans`,
  `poissonNNDensity`), `lib/distance.ts` (`edt`), `lib/label.ts` (`label2d`, `centroids`),
  `lib/synth.ts` (`nuclei`, `spots`), `lib/spotdetect.ts` (`spotImage`, `detectSpots`),
  `lib/stats.ts` (`quantile`, `tTestWelch`, `normCdf`; `logGamma` for the χ² CDF),
  `lib/random.ts` (`rng`, `poisson`, `gauss`, `shuffle`, `permutation`), `lib/plot.ts` (line,
  band for envelopes, step for G/F, histogram, scatter with per-point colours, vline), `lib/
  colour.ts` (`OKABE_ITO`, `viridis` for λ̂ maps), `lib/histogram.ts`, `lib/ui.ts`;
  `lib/playground.ts` for the editable pattern in `ripleyk` if its per-point colour option
  exists by then (build-issues list), else a small local editor.
- **One shared source panel** (`demos/src/csrplay/source.ts`, imported by `gfj`, `ripleyk`,
  `envelopetest`): pattern generator UI, presets and seeds, so that the same "colonies" pattern
  appears in every unit. Per-demo CSS prefixed `.qb-demo[data-demo="…"]` (build-issues lesson).
- **Speed** (TypeScript first, measure in the browser): n ≤ 800, 40 radii. Brute-force pairs
  with an early distance cut-off: 800² = 0.64 M distances per curve, a few ms; 99 simulations
  ≈ 0.3 s on a button press (with progress); the "how often is CSR rejected?" tab runs 200 ×
  (1 + 39) patterns of 100 points incrementally (requestAnimationFrame batches) with the
  precomputed curve shown at once. Isotropic weights in masks are the slow part (64 samples per
  pair): cache by pair distance class or default to translation in mask windows. A uniform grid
  of buckets (cell size = r_max) if n grows beyond 1 000. No Wasm expected: revisit only if a
  slider move (r, bandwidth) exceeds ~30 ms; slider moves must only re-read cached curves.
- **Fixed seeds** everywhere; every simulation button has a seed readout and a *reseed* button;
  the readouts quoted in decks come from the default seed.
- **Tests** (`demos/test/spatialstats-test.ts`, added to `make test`): mean of $\hat K$ over 400
  CSR patterns within 2 % of $\pi r^2$ for translation and isotropic correction, and the
  uncorrected bias near the values quoted above (−8 % at r = 0.1, −20 % at r = 0.25 of the side,
  n = 100); $\hat G$ and $\hat F$ of CSR within 2 SE of $1 - e^{-\lambda\pi r^2}$; $\hat g$ of a
  hard-core process < 0.05 below h; Thomas $\hat K$ averaged over patterns matches `thomasK`
  within 5 %; random labelling gives $\hat K_{11} \approx \hat K_{22} \approx \hat K_{12}$;
  `Kinhom` of an inhomogeneous Poisson pattern (true λ) within 5 % of πr²; Monte Carlo rank
  p-values uniform under CSR (KS-style check on 500 trials); the MAD test's rejection rate under
  CSR within 2–8 % at nominal 5 %; the pointwise-envelope global rate in the 30–45 % band
  (39 sims, 25 radii); the `lib/coloc.ts` re-exports give the same numbers as before.
- **Figures**: `tools/figs/spatialstats/figs.ts` (TypeScript, run with the same toolchain as
  `tools/figs/pointdetection/figs.ts`), SVGs in the unit folders, embedded as
  `![](fig-x.svg){fig-alt="..."}` (bracket captions show on slides), `.nostretch` on lone slide
  images; numbers in captions from the default seeds.
- **Images**: synthetic only by default (labelled as synthetic in every caption). Real two-type
  tissue data (multiplexed immunofluorescence or IHC with cell types) and a real spot pattern
  would strengthen units 3 and 5; open question 4.
- Follow `demos/README.md`: TypeScript, `make typecheck`, no network, presets + reset, the
  `slider`/`button`/`checkbox` helpers, labels ≥ 18 px, fits 1150×560 on slides (pattern panel +
  curve panel side by side; tabs for the rest; `data-layout="full"` on exercise pages).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:00 | Unit 1 `point-patterns/`: lecture (`csrplay`), quadrats by hand, "which is random?" |
| 10:15–11:15 | Unit 2 `nn-functions/`: lecture (`gfj`), G and F by hand, edge effect |
| 11:30–12:30 | Unit 3 `ripley-k/` part 1: K, L, g by hand |
| 13:30–14:15 | Unit 3 part 2: scales and edge corrections (`ripleyk`) |
| 14:15–15:30 | Unit 4 `envelopes/`: lecture, rank p-values by hand, `envelopetest` |
| 15:45–17:00 | Unit 5 `marked-patterns/`: lecture, the two nulls by hand, `celltypes`; quizzes as wrap-up |

Optional unit 6 as self-study or a second half-day.

## Open questions

1. **Five core units in one day** is full. Fallbacks in order: merge unit 2 into units 1 and 3
   (G/F/J as three slides at the end of unit 1, edge corrections only in unit 3); shorten the
   inhomogeneous part of unit 5 to two slides; make the mark connection function a pointer.
2. **Entry point**: day 11 (`relations/` has the Poisson NN law, Clark–Evans and the border
   slide) instead of day 9. Proposal below; if rejected, unit 2 restates the law.
3. **Unit names**: `point-patterns/`, `nn-functions/`, `ripley-k/`, `envelopes/`,
   `marked-patterns/`, optional `point-clusters/`; all checked against existing folders and
   demo names (`csrplay`, `gfj`, `ripleyk`, `envelopetest`, `celltypes`, `clusterfit` are free;
   note that `bootstrap` and `neighbours` already exist).
4. **Real data**: a licence-compatible multiplexed tissue image with cell-type calls (two or
   three types, a few hundred cells) and a real spot/nucleus pattern. Synthetic is enough to
   teach; one real example per day is the design rule (principle 4).
5. **χ² CDF**: needed for the quadrat test; belongs in `lib/stats.ts` (owner: evaluation day)
   rather than here. Proposal: add `chi2Cdf` there in the build (a small addition through the
   owner), or keep it local in `lib/pointpattern.ts` if that is not allowed.
6. **Notation** λ vs ρ: `relations/` and `coloc-objects/` use ρ for density; spatial statistics
   uses λ (and the course reserves λ for the regularisation weight, coursemap conventions).
   Proposal: keep **λ for intensity** on this day only with an explicit one-line note, or use **ρ**
   throughout to agree with the course convention and the two built units. Leaning to **ρ**
   (avoids the clash with "λ = weight of the prior"); the plan text above uses λ and the
   builder should switch if agreed.
7. **Global envelope flavour**: MAD-based band (simple, one statistic) vs the rank envelope
   (Myllymäki et al. 2017, more principled but harder to explain). Proposal: MAD in the concept
   layer, rank envelope named in the math layer.
8. **Pending edits elsewhere** (not done here): `relations/` "Clustered, random or regular?"
   forward pointer; `coloc-objects/` cross-K slide pointer and the move of `crossK`/`envelope`
   into `lib/pointpattern.ts` with re-exports; `replicates/` one line ("spatial patterns: one
   summary per image", pointer); the coursemap owners table row; the point-detection day's
   optional `smlm/` mention of blink clusters pointing to optional unit 6; sidebar, gallery and
   `make test` (integrator).

## Proposed reorganisation

1. **Owner row for the coursemap**: "Point-pattern statistics (CSR, quadrats, G/F/J, K/L/g, edge
   corrections, Monte Carlo envelopes and global tests, cross and mark functions, inhomogeneous
   K) → Spatial statistics of objects". `relations/` keeps the Poisson nearest-neighbour law and
   Clark–Evans as the first encounter (it is the natural place, and it is built);
   `coloc-objects/` keeps cross-K as a two-colour **application** with one pointer here.
2. **Move the "cross-K/L/pair-correlation with edge correction" part of the colocalisation
   day's optional `coloc-points/`** here (unit 5 and unit 3 cover it in general);
   `coloc-points/` keeps coordinate-based colocalisation per localisation and the 3D chromatic
   shift, and points to this day for the functions. Nothing built changes.
3. **Entry after day 11 (Measurements), not day 9 (Labelling)**: unit 2 builds directly on
   `relations/` (NN histogram, Poisson law, Clark–Evans, guard zone), and the day's inputs are
   the per-object tables of days 9–11. The point-detection day stays a soft prerequisite (one
   recall slide); the evaluation day (16) is a soft forward link (unit 4 is self-contained and
   points to `replicates/`). Group stays "Objects".
4. **Library consolidation**: `crossK`, `crossLminusR`, `envelope` from `lib/coloc.ts` move into
   the new `lib/pointpattern.ts`, with re-exports in `lib/coloc.ts` (no behavioural change to
   built demos; tested).

## References

- Diggle (2013), *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*, 3rd ed.,
  CRC (CSR, G/F, quadrats, Monte Carlo tests).
- Baddeley, Rubak & Turner (2015), *Spatial Point Patterns: Methodology and Applications with R*,
  CRC (teacher reference for estimators and corrections; no tool names in student material).
- Illian, Penttinen, Stoyan & Stoyan (2008), *Statistical Analysis and Modelling of Spatial Point
  Patterns*, Wiley (pair correlation, marks, mark connection function).
- Ripley (1976), *The second-order analysis of stationary point processes*, J. Appl. Prob.;
  Ripley (1977), *Modelling spatial patterns*, J. R. Stat. Soc. B (K function, isotropic
  correction).
- Besag (1977), discussion of Ripley (1977) (the L function).
- Clark & Evans (1954), *Distance to nearest neighbor as a measure of spatial relationships in
  populations*, Ecology (recall).
- van Lieshout & Baddeley (1996), *A nonparametric measure of spatial interaction in point
  patterns*, Statistica Neerlandica (the J function).
- Ohser (1983), *On estimators for the reduced second moment measure of point processes*, Math.
  Operationsforsch. Statist. (translation/isotropic corrections).
- Barnard (1963), discussion in J. R. Stat. Soc. B (Monte Carlo tests); Besag & Diggle (1977),
  *Simple Monte Carlo tests for spatial pattern*, Appl. Stat.
- Loosmore & Ford (2006), *Statistical inference using the G or K point pattern spatial
  statistics*, Ecology (pointwise envelopes as a multiple test; the MAD test).
- Baddeley, Diggle, Hardegen, Lawrence, Milne & Nair (2014), *On tests of spatial pattern based
  on simulation envelopes*, Ecological Monographs.
- Myllymäki, Mrkvička, Grabarnik, Seijo & Hahn (2017), *Global envelope tests for spatial
  processes*, J. R. Stat. Soc. B.
- Baddeley, Møller & Waagepetersen (2000), *Non- and semi-parametric estimation of interaction in
  inhomogeneous point patterns*, Statistica Neerlandica (inhomogeneous K).
- Thomas (1949), *A generalization of Poisson's binomial limit for use in ecology*, Biometrika;
  Matérn (1960/1986), *Spatial Variation* (cluster and hard-core processes).
- Lotwick & Silverman (1982), *Methods for analysing spatial processes of several types of
  points*, J. R. Stat. Soc. B (toroidal shift for independence).
- Diggle, Mateu & Clough (2000), *A comparison between parametric and non-parametric approaches
  to the analysis of replicated spatial point patterns*, Adv. Appl. Prob. (pooled K, replicates).
- Ester, Kriegel, Sander & Xu (1996), *A density-based algorithm for discovering clusters*, KDD
  (DBSCAN; optional unit).
- Owen et al. (2010), *PALM imaging and cluster analysis of protein heterogeneity at the cell
  surface*, J. Biophotonics, and Annibale, Vanni, Scarselli, Rothlisberger & Radenovic (2011),
  *Quantitative photo activated localization microscopy: unraveling the effects of
  photoblinking*, PLoS ONE (blink clusters; optional unit).
- Lagache, Lang, Sauvonnet & Olivo-Marin (2013), *Analysis of the spatial organization of
  molecules with robust statistics*, PLoS ONE (Ripley-based tests in bioimaging; teacher
  background).

## Build notes (2026-11-13)

Built: the five core units (`point-patterns/`, `nn-functions/`, `ripley-k/`, `envelopes/`,
`marked-patterns/`), demos `csrplay` (with the shared `csrplay/source.ts`), `gfj`, `ripleyk`,
`envelopetest`, `celltypes`, the library `demos/src/lib/pointpattern.ts` (χ² CDF kept local),
`demos/test/spatialstats-test.ts`, figure scripts in `tools/figs/<unit>/` (shared kit in
`tools/figs/point-patterns/kit.ts`; rates and seeds in `tools/figs/envelopes/`). Optional unit 6
not built; `lib/coloc.ts` consolidation not done (not this builder's file).

Decisions and corrections to the plan:
- Intensity is written **ρ** everywhere (course convention: λ is the prior weight), one line in unit 1.
- The VMR of a Thomas pattern does **not** return to 1 for large quadrats: it rises towards 1 + μ.
  Unit 1 says so (math B7); "regular at 5 µm, clustered at 20 µm" is shown with colonies + hard core.
- Simulated rates (1000 CSR patterns of 100 points, 200 µm square, L − r with translation
  correction, radii up to 50 µm): pointwise 39 sims at 25 radii **37 %** (plan: 36 %), 19 sims
  **59 %**, MAD test 5 %. Uncorrected K bias: −8.2 % at 0.1 side, −20 % at 0.25 side (as planned).
- The Thomas K̂ in a 200 µm window with ~20 clusters is ~8 % above the formula (ratio-estimator
  bias); the test uses a 400 µm window, the ripleyk demo shows the formula only on the g tab.
- celltypes: the MAD test uses the mean of the simulated curves as reference (under permuted
  labels in packed tissue the null mean is far from 0).
- Bandwidth dilemma preset: σ = 5 µm (p = 0.125), 12 µm (below, p = 0.025), 40 µm (clustered).
