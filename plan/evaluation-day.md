# Plan: a day on evaluation and statistics

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
Core day 16, Part V "Learning from examples", after random forests (14) and neural
networks (15), before VAEs (17) and modern deep learning (18). Closes gap 4 of `plan/review-gaps.md`.

## Goal of the day

After the day a student can:

1. **match** predicted objects to ground-truth objects at an IoU threshold, one-to-one, and
   count TP/FP/FN **objects**; compute precision, recall and **F1 at a threshold τ**, read an
   **F1-vs-τ curve**, and compute **average precision** from detections ranked by score;
2. recognise **splits and merges** as the typical object errors, and explain why a pixel Dice
   of 0.95 can come with a wrong count;
3. choose a **boundary metric** (Hausdorff, average surface distance) when the outline matters,
   and say what each one is sensitive to;
4. compare automated and manual **counts and measurements** with a **Bland–Altman** plot: bias
   vs scatter, limits of agreement, and why a high correlation is not agreement;
5. treat **ground truth as uncertain** (inter-annotator agreement as the ceiling of any score)
   and estimate roughly how many annotated images an evaluation needs;
6. spot **data leakage** in a validation design (random pixel/patch splits, augmentation or
   normalisation before the split, tuning on the test set) and design a split **by image,
   specimen, experiment or day**; say what **domain shift** a held-out site or day tests;
7. identify the **unit of replication** in an imaging experiment (cells within images within
   wells within experiments), explain **pseudoreplication** and why it inflates false
   positives, and aggregate per replicate before testing;
8. report an **effect size with a confidence interval**, correct for **multiple comparisons**
   across hundreds of features (Bonferroni, Benjamini–Hochberg), and recognise **batch and
   plate effects** that are confounded with the treatment;
9. *(math)* prove that matching at τ > 0.5 is unique, state matching as a linear assignment
   problem, and derive the variance of an experiment mean in a two-level model (design effect)
   and the false-positive rate of the cells-as-n test.

Why it fits this audience: everything is counting, means and variances (which they know),
plus one new statistical idea, the null distribution, introduced by simulation. The matching
problem is a small optimisation over permutation matrices (linear algebra), and the
two-level variance formula is "the variance of a mean" from the random-forest day with a
correlation term.

## Prerequisite gap / overlaps

Written against `plan/coursemap.md` (owners) and the parallel plans (2026-11-12). This day is
the owner of **object-level evaluation, leakage and statistics of measurements**; everything
else below is recalled in at most one slide with a pointer.

- **Pixel-level evaluation** (thresholding day, `seg-eval/`): confusion matrix,
  precision/recall, Dice = F1, Jaccard = IoU, ROC/PR as a threshold sweeps, "accuracy lies".
  **Recalled** in one slide at the start of unit 1 (with the `threshroc` picture); object-level
  F1 is then "the same formula, but the things counted are objects". `seg-eval/`'s pointer
  "object metrics → labelling day" and its one-slide Hausdorff pointer should now point here.
- **Train/test, overfitting, cross-validation** (random-forest day, owner): recalled in one
  line. The RF day already shows **spatially correlated training pixels** in `pixelclass`
  ("OOB 99 %, held-out region 85 %") and the SVM day shows CV by pixel vs by stroke. Unit 3
  here **generalises** this into a rule (split at the level you want to generalise to) and adds
  the levels above one image (specimen, experiment, day, site). No re-teaching of OOB or CV.
- **Train/validation/test, domain shift, shortcut learning** (neural-nets day, unit 2 and
  `deep-practice/`): recalled; unit 3 adds "tuning on the test set" as a quantified effect and
  uses the `unetseg` domain-shift presets as the example of a held-out-site test.
- **Counting, border objects, unbiased counting frame** (labelling day, `label-measure/`):
  recalled. Unit 1 needs one line on border objects in evaluation (exclude objects touching the
  border from both truth and prediction, or neither).
- **Splits and merges as segmentation outcomes** (watershed day): recalled as the cause of the
  errors this day counts.
- **Distance transform** (morphology day, owner): used for boundary distances (one line).
- **Normalisation, per-well aggregation, z-scores against controls, plate gradients**
  (measurements day, `features-use/`, currently optional): unit 4 here owns the **statistical**
  side (replicates, effect sizes, multiple testing, confounded designs). If `features-use/`
  stays optional, unit 4 must stand alone, so it re-states per-replicate aggregation (median per
  well/experiment) in one slide and recalls robust z-scores only as a pointer. Proposal:
  `features-use/` keeps the mechanics of normalisation; plate/batch effects are **pictured**
  there and **reasoned about** (confounding, randomised layout) here.
- **Measurement bias from pixel size and PSF** (measurements day perimeter bias, image formation
  day): recalled as the cause of a proportional bias in the Bland–Altman area preset.
- **Detection AP** (day 18 `detection/`, new): boxes, NMS and COCO-style AP over several IoU
  thresholds reuse this day's matching and AP; that unit only recalls them.
- **Statistics prerequisite gap**: the audience knows mean, variance and the normal
  distribution, but **not** standard errors, p-values, t-tests or confidence intervals. Unit 4
  introduces them from scratch in the concept layer by **simulation** ("how often would chance
  alone give a difference this big?"), with "mean ± 2 SE" as the working CI. The
  t-distribution and the formulas go in the math layer. This is the biggest new block of the
  day and sets the time budget of unit 4.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise. Each lecture follows the math-layer
markup of README.md ("Marking the math layer"); each exercise has concept questions first,
then `### Math questions` tagged *(math)*. No tool or library names in student material.

### 1. `object-eval/`: Matching objects: IoU, F1 and average precision

Concept layer:

- Recall (one slide, `seg-eval/`): the confusion matrix and Dice for pixels. Picture: two
  touching nuclei predicted as one blob, Dice 0.95, count 1 instead of 2. Pixel metrics cannot
  see objects; we need to decide **which predicted object is which true object**.
- **IoU of two objects** (= Jaccard of the two masks): overlap / union. Pictures of IoU 0.3,
  0.5, 0.7, 0.9 for a disc shifted by a few pixels: IoU 0.5 is already a visibly sloppy outline;
  small objects lose IoU fast for a one-pixel error.
- **Matching**: each true object can be matched to at most one predicted object and vice versa,
  and only if their IoU ≥ τ. Matched pairs are **TP**; unmatched predictions are **FP**;
  unmatched true objects are **FN**. There are no TN objects.
- How to match: the **IoU table** (true objects × predicted objects, mostly zeros). Above
  τ = 0.5 there is nothing to choose: each object has at most one partner above 0.5, so
  "take every pair with IoU > 0.5" is the matching. Below 0.5, a greedy choice (biggest IoU
  first) can block a better overall matching; the **optimal assignment** (most matches, then
  largest total IoU) fixes it. Picture of a 2×2 case where greedy loses a match. Image libraries
  call the optimal solver the "Hungarian algorithm" (one line).
- Object **precision** = TP/(TP+FP) ("how many of my objects are real"), **recall** =
  TP/(TP+FN) ("how many real objects did I find"), **F1** = 2TP/(2TP+FP+FN). Same formulas as
  the pixel ones; now one object counts one, big or small.
- **F1 vs τ**: sweep τ from 0.1 to 0.95. A method that finds every nucleus with sloppy outlines
  has high F1 at 0.5 and collapses at 0.8; a method with tight outlines that merges pairs has a
  lower plateau. Report the curve, or at least two τ values, never one number.
- Typical errors read from the IoU table: a **split** (one true object, two predictions with
  IoU around 0.4 each: 1 FN + 2 FP at τ = 0.5), a **merge** (two true objects, one prediction:
  1 TP at best + 1 FN, or 2 FN + 1 FP), **missed** dim objects, **spurious** specks. One merge
  costs more F1 than many slightly wrong outlines; counts of splits and merges are worth
  reporting on their own.
- **Scores and average precision**: many methods give each object a confidence. Sort the
  predictions by score; walk down the list; each prediction is TP or FP (at τ); after each step
  write down (recall, precision). This traces a **precision–recall curve** (as in `seg-eval/`,
  but the sweep is over the score, not a grey value). **AP** = area under the (stepwise,
  "never-decreasing-from-the-right") PR curve. Averaging AP over τ = 0.5, 0.55, …, 0.95 rewards
  good outlines too.
- **Naming trap** (one slide, important): many cell-segmentation papers write "AP at τ" for
  TP/(TP+FP+FN) at one τ, without scores, which is not the area under a PR curve. Always check
  the definition before comparing numbers. In this course we call that quantity **matching
  accuracy** and say that papers sometimes call it AP.
- Border objects: decide once whether objects touching the image border count, and apply the
  same rule to truth and prediction (recall the labelling day's counting frame).

The math (`## The math: … {.math}` slides):

- IoU$(A, B) = |A \cap B| / |A \cup B|$; Dice $= 2\,$IoU$/(1 + $IoU$)$ (recall $J = D/(2-D)$).
- **Uniqueness above 0.5**: if IoU$(A, B) > 1/2$ then $|A \cap B| > |A|/2$; two disjoint
  predictions $B_1, B_2$ cannot both overlap more than half of $A$. So every object has at most
  one partner with IoU > 1/2, and the matching is forced.
- **Matching as linear assignment**: maximise $\sum_{ij} x_{ij} M_{ij}$ over $x_{ij} \in
  \{0, 1\}$ with row and column sums ≤ 1, where $M_{ij} = \text{IoU}_{ij}$ if ≥ τ and 0
  otherwise (or $M_{ij} = 1 + \epsilon\,\text{IoU}_{ij}$ to maximise the number of matches
  first). The constraint matrix is totally unimodular, so the LP relaxation has an integer
  optimum (state, no proof); the Hungarian algorithm solves it in $O(n^3)$ (state).
- $\text{F1}(\tau) = 2\,TP(\tau)/(2\,TP(\tau) + FP(\tau) + FN(\tau))$, non-increasing in τ.
- AP $= \sum_k (R_k - R_{k-1})\, P^{\text{interp}}(R_k)$ with $P^{\text{interp}}(r) =
  \max_{r' \ge r} P(r')$; AP over thresholds $= \frac{1}{10}\sum_{\tau} \text{AP}(\tau)$.

**Demo `objmatch`** (new, main demo). A 256×256 synthetic fluorescence-like image of ~15
nuclei with a ground-truth label image (from the shared generator, some touching pairs, some
dim). Two modes (`data-mode`):

- *edit*: predicted objects are ellipses drawn over the image; drag the centre, drag a rim
  handle to resize, rotate handle; buttons *add*, *delete*, *split selected* (cuts an ellipse
  into two halves), *merge with nearest*. Predictions are rasterised to a label image and all
  numbers are computed on pixels.
- *pipeline*: predictions come from a seeded "error generator" applied to the truth, with
  sliders for boundary jitter (pixels), dilation/erosion (outline bias), merge probability for
  touching pairs, split probability, miss probability for dim objects, spurious specks. Each
  prediction gets a score (lower for specks and dim objects).

Panels: left, the image with truth outlines (white) and predictions filled by status (TP green,
FP red, FN as dashed orange outlines; split/merge badges); middle, the IoU table of the objects
under the cursor region (≤ 8×8 cells, numbers shown, matched cells boxed, cells below τ greyed)
with a *greedy / optimal* toggle; right, the F1-vs-τ curve with the current τ as a dot, the
pixel Dice as a horizontal reference line, and a second tab with the PR curve at τ and AP (and
AP averaged over τ). A τ slider. Toggle *annotator B*: a second ground truth (same objects,
boundaries jittered, one disputed object) whose F1-vs-τ against annotator A is drawn as a grey
**ceiling band** (used again in unit 2). Presets: "Dice 0.95 but count wrong" (merges only),
"sloppy outlines: fine at 0.5, gone at 0.8", "one split = two FP + one FN", "greedy loses a
match" (τ = 0.3), "specks: precision drops, AP barely moves (low scores)", "better than the
second annotator?". Reset.

By hand (concept):

- (a) IoU of two 4×4 squares on a grid, shifted by one pixel; of two 2×2 squares shifted by
  one pixel. Which loses more, and why does this matter for small objects?
- (b) IoU table, 3 true × 4 predicted:
  G1: (0.82, 0.05, 0, 0), G2: (0.10, 0.45, 0.38, 0), G3: (0, 0, 0, 0.62).
  Match at τ = 0.5 (TP 2, FP 2, FN 1; F1 = 4/7 ≈ 0.57), at τ = 0.3 (TP 3, FP 1, FN 0;
  F1 = 6/7 ≈ 0.86) and at τ = 0.7 (TP 1, FP 3, FN 2; F1 = 2/7 ≈ 0.29). Which error is G2?
  (A split.) Sketch F1 vs τ.
- (c) Greedy vs optimal: G1: (0.60, 0.40), G2: (0.35, 0) at τ = 0.3. Greedy: 1 match; optimal:
  2 matches (G1–P2, G2–P1). Draw shapes that could give this table.
- (d) AP: 4 true objects; 5 predictions sorted by score are TP, TP, FP, TP, FP at τ = 0.5. PR
  points (0.25, 1), (0.5, 1), (0.5, 0.67), (0.75, 0.75), (0.75, 0.6); interpolated AP =
  0.25 + 0.25 + 0.25·0.75 = 0.6875. Matching accuracy TP/(TP+FP+FN) = 3/(3+2+1) = 0.5: the two
  "APs" differ.
- (e) Predict: what happens to the F1-vs-τ curve if every predicted outline is one pixel too
  large? If one pair of nuclei is merged?

Math questions (under `### Math questions` in the exercise):

- **B1.** *(math)* Prove that IoU$(A, B_1) > 1/2$ and IoU$(A, B_2) > 1/2$ with $B_1 \cap B_2 =
  \emptyset$ is impossible. Is it still impossible at exactly 1/2? At 1/3 (give a
  counterexample)?
- **B2.** *(math)* Write the matching of (c) as an assignment problem; enumerate all feasible
  $x$ and find the optimum for both objective choices (total IoU; number of matches first).
- **B3.** *(math)* Show that F1(τ) is non-increasing in τ when matching is forced (τ > 0.5).
  Find a counterexample for greedy matching below 0.5.

### 2. `agreement/`: Boundaries, counts and the uncertain ground truth

Concept layer:

- **Boundary metrics**, when the outline itself matters (a cell's shape, a tumour margin):
  for each point on one boundary, the distance to the nearest point on the other (computed from
  a distance transform, recall the morphology day). **Hausdorff distance** = the worst such
  distance (both directions): sensitive to a single spike or a stray fragment. **Average
  surface distance** (ASD) = the mean: tells the typical outline error in pixels or µm. The 95th
  percentile is a common compromise. IoU and Dice hardly notice a thin spike; Hausdorff
  hardly notices a uniform one-pixel offset relative to how badly it sees a spike.
- **Counts and measurements**: often the goal is not the masks but the numbers (cells per
  image, mean area). Evaluate the numbers directly against manual values.
- **Correlation is not agreement**: auto = 0.8 × manual has correlation 1 and is wrong by 20 %.
  The **Bland–Altman plot**: for each image, plot the difference (auto − manual) against the
  mean of the two. The mean difference is the **bias** (systematic error); the spread
  (± 2 SD) is the **limits of agreement** (scatter). A sloped cloud = **proportional bias**
  (e.g. merges increase with density, so undercounting grows with the count); then compare
  ratios or work on a log scale.
- Bias and scatter have different cures: a bias can be calibrated away (if it is stable); a
  scatter cannot. Bias matters for absolute values, often not for comparing two conditions
  measured the same way (unless the bias depends on the condition: denser wells merge more).
- Measurements: area of each object, auto vs manual. A one-pixel outline offset is a fixed
  absolute error in boundary length, so a large **relative** bias for small objects (recall
  the perimeter bias of the measurements day).
- **Ground truth is uncertain**: two annotators disagree on boundaries (a few pixels) and on
  whether a dim or dividing object counts. Their agreement (F1 vs τ between annotators, and
  their Bland–Altman plot) is the **ceiling**: a method that scores 0.90 against annotator A when
  annotator B scores 0.88 is "as good as a human", not "better than 0.88". Remedies: several
  annotators, consensus, an "ignore" label for ambiguous objects, synthetic data where the truth
  is known by construction (and the domain gap that brings).
- **How many annotated images**: an F1 or a bias estimated from few objects wobbles. Recall of
  0.9 from 30 objects has a standard error of about 0.05; from 300 objects about 0.017. Objects
  from the same image are not independent (same illumination, same density), so the number of
  **images** (and experiments) matters more than the number of objects: evaluate per image and
  look at the spread across images. Leads to units 3 and 4.

The math:

- Directed distance $h(A, B) = \max_{a \in \partial A} \min_{b \in \partial B} \|a - b\|$;
  Hausdorff $H = \max(h(A,B), h(B,A))$; ASD $= \big(\sum_{a} d(a, \partial B) + \sum_{b}
  d(b, \partial A)\big)/(|\partial A| + |\partial B|)$; via the distance transform of each
  boundary, $O(N)$.
- Bland–Altman: $d_i = y_i - x_i$, bias $\bar d$, SD $s_d$, limits $\bar d \pm 1.96\,s_d$
  (if $d$ is roughly normal); a regression of $d_i$ on $(x_i + y_i)/2$ for proportional bias;
  why correlation ignores bias: $r(x, ax + b) = 1$ for any $a > 0$, $b$.
- Standard error of a proportion $\sqrt{p(1-p)/n}$; with clustering by image, the effective
  number of objects (design effect, derived in unit 4).

**Demo `contourdist`** (new, small). One true outline (fixed, a nucleus-like closed curve) and a
predicted outline with ~12 draggable control points (smooth closed spline). Whiskers from
boundary points to the nearest point on the other curve, coloured by length; the longest
marked as the Hausdorff distance. Readouts: IoU, Dice, Hausdorff, HD95, ASD, all in pixels;
a histogram of boundary distances. Size slider scales both curves (same pixel errors on a
small and a big object). Presets: "uniform one-pixel dilation", "one thin spike", "a stray
fragment far away" (Hausdorff explodes, IoU unchanged), "same error, small object". Reset.

**Demo `blandaltman`** (new). 40 seeded images with manual counts (10–150 objects); automated
counts from a model with sliders: constant bias, proportional bias (undercount at high density),
random scatter; toggle *areas* (one point per object, auto area from a ±1-pixel outline offset,
so relative bias grows for small objects) and toggle *second annotator* (manual vs manual).
Panels: scatter auto vs manual with the identity line and $r$; Bland–Altman plot with bias,
limits of agreement and a fitted slope; a third mini-panel with the ratio version. Presets:
"r = 0.99, still 20 % off", "merges at high density: proportional bias", "scatter only: unbiased
but noisy", "small objects: relative area bias", "two annotators: the ceiling". Reset.

By hand (concept):

- (a) Point sets A = {(0,0), (1,0), (2,0)}, B = {(0,1), (1,1), (2,1), (1,4)}: directed distances,
  Hausdorff (4) and ASD (10/7 ≈ 1.43). Which point is responsible? What would IoU of the
  corresponding regions say?
- (b) Bland–Altman by hand: manual 20, 35, 50, 65, 80, 100; auto 19, 33, 46, 60, 72, 88.
  Differences −1, −2, −4, −5, −8, −12: bias −5.33, SD 4.08, limits −13.3 to +2.7. Sketch the
  plot; why are these limits misleading here (proportional bias), and what would you plot
  instead? Estimate the correlation without computing it.
- (c) Two annotators reach object F1 0.86 against each other at τ = 0.5. Method X scores 0.84
  against annotator A, method Y 0.91. What can you conclude about each?
- (d) A recall of 0.80 is measured on 25 objects. Roughly how wide is its ±2 SE interval? How
  many objects for ±0.05? Why might even that be optimistic if all objects come from two images?

Math questions (under `### Math questions` in the exercise):

- **B4.** *(math)* Show that Hausdorff distance is a metric on non-empty compact sets (at least
  the triangle inequality). Is ASD a metric?
- **B5.** *(math)* Show that if auto = a·manual + b exactly, the Bland–Altman points lie on a
  line; find its slope in terms of a. When is the slope zero?
- **B6.** *(math)* Show that $\sqrt{p(1-p)/n}$ is largest at p = 1/2 and give the n needed for
  a ±0.05 interval (95 %) in the worst case.

### 3. `leakage/`: Honest validation: leakage, splits and domain shift

Concept layer:

- Recall (one line, RF day): train on one set, test on another; cross-validation rotates the
  test set. The RF day already met one trap: neighbouring pixels in a stroke are near copies.
- The rule: **the test set must be as new as the data you want to work on later.** If the
  method will be used on new images from new experiments, the test images must come from
  experiments it has never seen. Every shortcut in the split is **leakage**: information about
  the test set reaches training, and the score is optimistic.
- Why random pixel or patch splits leak: images are **spatially autocorrelated**. A test pixel
  has its neighbour in the training set, with almost the same features and the same label; the
  classifier only needs to remember, not to generalise. Picture: random pixel split as a
  salt-and-pepper map, block split as a checkerboard, image split. The same holds for patches
  cut from one image, frames from one movie, slices from one volume, cells from one well.
- The **hierarchy of splits**: pixel < patch < image < specimen/well < experiment/day <
  site/microscope/lab. Split at the level you claim to generalise to; group k-fold
  ("leave one experiment out"). Spatial blocks with a buffer zone when there is only one big
  image (e.g. one slide).
- Other leaks: **augment then split** (rotated copies of the same image on both sides);
  **normalise with statistics of the whole dataset** before splitting (mild); duplicate or
  near-duplicate images; the same patient/animal under two names; selecting features on all
  data then cross-validating.
- **Tuning on the test set**: trying 50 settings (σ of the features, threshold, network size,
  post-processing) and reporting the best test score reports luck. With noisy scores the
  maximum of many equal methods is above all of them. Use a **validation** set (or inner CV)
  for choices and touch the test set once.
- **Domain shift** (recall NN day): a new microscope, stain, cell line or lab changes the
  input distribution. A held-out day or site measures it; the drop from "held-out image" to
  "held-out site" **is** the domain shift, and it is the number a user of the method cares
  about. Shortcuts (a scale bar, a plate edge) are leaks that live inside the data.
- Checklist slide: what is the unit of the split, what was tuned on what, were there duplicates,
  what is the test domain.

The math:

- Autocorrelation as a function of distance; samples closer than its range are effectively
  one sample. **Design effect** for clusters of size m with intra-cluster correlation ρ:
  $n_{\text{eff}} = n / (1 + (m-1)\rho)$ (derivation in unit 4).
- Optimism of the best of k: k test scores $\hat a_j = a + \varepsilon_j$, $\varepsilon_j \sim
  N(0, \sigma^2)$, $\sigma = \sqrt{a(1-a)/n}$; $E[\max_j \hat a_j] - a = \sigma\, E[\max_j
  Z_j]$, with $E[\max Z] \approx 1.54$ for k = 10, 1.87 for 20, 2.51 for 100, and the bound
  $\sigma\sqrt{2\ln k}$ (Jensen + union bound sketch).
- Nested cross-validation as the estimate of the whole procedure "tune, then train".

**Demo `leakage`** (new, main demo). Twelve 64×64 synthetic images from 4 "experiments" (days),
3 images each; pixel task "nucleus vs background" with a hard class (dim nuclei, texture), and
a per-day effect (brightness offset, contrast and noise level, slider *day effect*). Features: a
small stack (intensity, Gaussian σ = 1, 2, 4, gradient magnitude). Classifier: k-nearest
neighbours on a subsample of ~3000 training pixels (simple, fast, and it shows leakage most
clearly because it can only remember); a *forest* option through the shared classifier
interface if `lib/forest.ts` exists. Split strategies: random pixels, random 8×8 blocks, blocks
with a buffer, by image, by experiment (leave one day out). Panels: the 12 images as
thumbnails with train/test colouring for the selected strategy; a bar chart of test accuracy (or
Dice) for **all** strategies side by side, with the spread over folds; the prediction on a chosen
test image. Preset **"nothing to learn"**: the labels are random smooth blobs unrelated to the
image; random pixel split still scores ~85 %, by-image ~50 %. Other presets: "random pixels look
great", "blocks need a buffer", "a new day: domain shift" (day effect high), "no day effect:
image split ≈ day split", "augment then split" (rotated copies of test images in training).
Reset.

**Demo `testpeek`** (new, small). k candidate methods, each with the same true accuracy a
(slider, default 0.80), or with true accuracies spread around a; test set of n objects
(slider). *Run once*: test scores as dots, the best one highlighted, and its score on a fresh
test set. *Run 1000*: histograms of "best on test" vs "best, re-tested on fresh data", with the
mean optimism. Toggle *validation + test*: choose on validation, report on test (optimism
vanishes). Presets: "one method: honest", "50 settings, 100 objects: +7 %", "bigger test set,
less optimism", "validation set fixes it". Reset.

By hand (concept):

- (a) For each design, does it leak, and at which level? (1) 2000 tiles from 10 slides, random
  tile split; (2) 40 images from 4 days, leave-one-day-out; (3) images rotated 8 ways, then
  split; (4) intensities scaled by the dataset mean and SD, then split by image; (5) the
  threshold chosen for the best test Dice; (6) cells from 6 wells, split by cell.
- (b) 20 settings all with true accuracy 0.80 on a test set of 100 objects: SE ≈ 0.04, the best
  of 20 reads about 0.80 + 1.87·0.04 ≈ 0.875. What happens with a test set of 1000?
- (c) A classifier scores 0.97 under a random pixel split, 0.91 leaving out images, 0.78 leaving
  out days. Which number goes in the paper, and what does each gap mean?
- (d) One big image only: design a validation that is as honest as possible. Draw it.

Math questions (under `### Math questions` in the exercise):

- **B7.** *(math)* Derive $E[\max(Z_1, Z_2)] = 1/\sqrt{\pi}$ for two independent standard
  normals, and show $E[\max_j Z_j] \le \sqrt{2 \ln k}$.
- **B8.** *(math)* 1-nearest-neighbour on a 1D signal with autocorrelation length ℓ and samples
  every Δ: argue why the random-split error goes to 0 as Δ/ℓ → 0 even if the label is
  independent of the features.

### 4. `replicates/`: From measurements to claims

Concept layer:

- The question behind most imaging experiments: does a treatment change a measured feature?
  Data are **hierarchical**: cells within images within wells within experiments (days,
  animals, patients). Variation exists at every level; the day-to-day level is usually the
  largest.
- New statistics, introduced by simulation (audience gap): a mean of n values wobbles, and
  its wobble (**standard error**) shrinks like 1/√n. A **p-value** is "how often chance alone
  (no treatment effect) would give a difference at least this big": shown as a histogram of
  simulated null differences with the observed one marked. A **95 % confidence interval** ≈
  estimate ± 2 SE: the range of effects compatible with the data.
- **The unit of replication** is the level at which the treatment was applied independently
  (the experiment, the animal, the well if wells were treated separately), not the cell.
  1000 cells from one dish are 1000 measurements of **one** replicate. Treating cells as n
  (**pseudoreplication**) makes the SE too small, and p-values tiny even when there is no
  effect: under the null, with a modest day-to-day variance, a cells-as-n test at 5 % rejects
  far more often than 5 % (the demo shows 30–60 %).
- Do it right: **aggregate per replicate** (mean or median per experiment), test on those n
  values; or a **paired** design (control and treated in every experiment, compare within
  each), which removes the day-to-day variance. Mixed-effects models do the same thing with all
  cells kept (one-line pointer). Show the data as a **SuperPlot**: cells as small dots coloured
  by experiment, experiment means as big markers, statistics on the big markers.
- More cells per experiment barely helps once the day effect dominates; more experiments
  does. Picture: SE vs number of cells, flattening out.
- **Effect size and CI** before p-values: "the treated cells are 12 % larger (95 % CI 4–20 %),
  n = 4 experiments" says what matters; p < 0.001 with n = 3000 cells says nothing about size.
  Standardised effect (difference / SD) for comparing features with different units.
- **Multiple comparisons**: 300 features, no real effect, α = 0.05 → about 15 "hits". Picture:
  the histogram of p-values is flat under the null. **Bonferroni** (α/m: controls the chance of
  any false hit, strict) and **Benjamini–Hochberg** (controls the fraction of false hits among
  the hits, the false discovery rate). Choosing the best-looking feature after the fact is the
  same problem as tuning on the test set (unit 3).
- **Batch and plate effects**: day, plate, row/column position, edge wells (evaporation), the
  imaging order. If all controls are on day 1 and all treated on day 2, the effect is
  **confounded** with the batch and no statistics can separate them. Cures are in the design:
  every batch contains every condition, randomised plate layout, controls on every plate;
  normalisation against per-plate controls (pointer to `features-use/`).
- **Reporting** (checklist slide): n at each level and which is the unit of replication; how
  values were aggregated; effect size with CI; correction for multiple testing; how the
  evaluation set was split (unit 3); who annotated (unit 2); images and parameters available.
  Pointer to the course-level integrity/ethics unit (gap 7) if it exists.

The math:

- Two-level model $y_{ec} = \mu + \tau\,T_e + u_e + \varepsilon_{ec}$, $u_e \sim
  N(0, \sigma_u^2)$, $\varepsilon_{ec} \sim N(0, \sigma^2)$. Variance of an experiment mean:
  $\sigma_u^2 + \sigma^2/m$. Variance of the mean of all $Em$ cells: $\sigma_u^2/E +
  \sigma^2/(Em)$, while cells-as-n assumes $(\sigma_u^2 + \sigma^2)/(Em)$. Ratio = design effect
  $1 + (m - 1)\rho$ with $\rho = \sigma_u^2/(\sigma_u^2 + \sigma^2)$.
- False-positive rate of the cells-as-n z-test under the null: $2\,\Phi\big(-1.96/\sqrt{1 +
  (m-1)\rho}\big)$ (here $z$ is computed with the cells-as-n SE, while its true SD is
  $\sqrt{1 + (m-1)\rho}$); e.g. m = 100, ρ = 0.1: $\sqrt{10.9} = 3.3$, $2\Phi(-0.59) \approx
  0.55$ instead of 0.05 (large-E approximation; checked against simulation in the tests).
- Standard error, the t statistic $t = \bar d / (s/\sqrt n)$ and the t-distribution with
  $n - 1$ degrees of freedom (why it has heavier tails with small n); CI $\bar d \pm
  t_{0.975, n-1}\, s/\sqrt n$; Welch's two-sample version in one line.
- Paired design: $\mathrm{Var}(y_{eT} - y_{eC})$ does not contain $\sigma_u^2$.
- Multiple testing: P(at least one false hit) $= 1 - (1-\alpha)^m$; Bonferroni by the union
  bound; Benjamini–Hochberg step-up rule (largest k with $p_{(k)} \le k q/m$), FDR ≤ q stated.

**Demo `pseudorep`** (new, main demo). Simulated experiment: two conditions, E experiments
(2–10), m cells per experiment and condition (5–1000, log slider), between-experiment SD,
within-experiment SD, true effect; design toggle *independent / paired* (both conditions in
every experiment) and *confounded* (all controls in the first half of the days). Panels: a
SuperPlot (cells jittered, coloured by experiment; experiment means as big markers, joined by
lines in paired mode); readouts for three analyses side by side: cells as n, experiment means as
n, paired, each with the estimated effect, 95 % CI bar and p-value; *Run 1000*: histogram of
p-values for each analysis and the fraction below 0.05 (the false-positive rate when the true
effect is 0; the power otherwise). Presets: "1000 cells, no effect, p < 0.001", "the same, with
experiments as n", "more cells does not help", "more experiments does", "paired design wins",
"confounded with the day: nothing helps". Reset.

**Demo `featurescan`** (new). A 16×24 well plate (or 8×12), m features per well (10–500) of
which a fraction carries a real effect (slider) of chosen size; plate effects: row/column
gradient and edge wells (sliders); layout toggle *treated on the left half / randomised /
randomised with controls on every plate*. Panels: plate heat map of a selected feature (wells
outlined by condition); histogram of all p-values (flat under the null, spike near 0 with real
effects); sorted p-values with the α, Bonferroni and BH lines and the hits of each marked, with
the true/false hit counts known from the simulation. Presets: "300 null features: 15 hits",
"Bonferroni: none", "10 % real: BH finds most", "plate gradient + treatment on the left: half
of everything is significant", "randomised layout fixes it". Reset.

By hand (concept, with a small table of t critical values: 2 df 4.30, 4 df 2.78, 22 df 2.07):

- (a) Three experiments, four cells per condition. Control: (10, 11, 12, 11), (14, 15, 13, 14),
  (8, 9, 7, 8); treated: (12, 13, 14, 13), (15, 16, 14, 15), (10, 9, 11, 10). Experiment means
  11, 14, 8 vs 13, 15, 10. Paired differences 2, 1, 2: mean 1.67, SD 0.58, SE 0.33, t = 5.0
  > 4.30: significant. Unpaired on experiment means: t ≈ 0.74: not. Cells as n (12 vs 12):
  t ≈ 1.65. Now imagine 100 cells per experiment with the same spread: t ≈ 8. Which analyses
  are right, which is wrong, and why does the paired one win?
- (b) 200 features, no real effect, α = 0.05: expected false hits (10); P(at least one)
  (≈ 1); Bonferroni threshold (0.00025).
- (c) Ten sorted p-values 0.001, 0.004, 0.012, 0.02, 0.03, 0.2, 0.4, 0.5, 0.7, 0.9; how many
  hits at α = 0.05 uncorrected (5), Bonferroni (2), BH with q = 0.05 (4)?
- (d) A plate layout drawing: controls in column 1–2, treatment A in 3–12, all imaged left to
  right over two hours while the focus drifts. What could go wrong, and how would you lay out
  the plate instead?
- (e) Rewrite the sentence "Treated cells were significantly larger (p < 0.0001, n = 2400)"
  into a good report (what extra information do you need?).

Math questions (under `### Math questions` in the exercise):

- **B9.** *(math)* Derive the variance of the grand mean in the two-level model and the design
  effect $1 + (m-1)\rho$. What is $n_{\text{eff}}$ for m = 500, ρ = 0.2, E = 3?
- **B10.** *(math)* Derive the false-positive rate of the cells-as-n z-test under the null
  (formula above) and evaluate it for m = 10, 100, 1000 at ρ = 0.1.
- **B11.** *(math)* Show that the paired difference removes $u_e$; when does pairing lose
  power compared with the unpaired test?
- **B12.** *(math)* Prove that Bonferroni controls the family-wise error rate (union bound)
  without any independence assumption; show by example that BH can reject a hypothesis with
  p > α/m.

## Shared engineering

- **`demos/src/lib/eval.ts`** (new): object-level evaluation. IoU table from two label images
  in one pass over pixels (sparse: only overlapping pairs); matching (forced for τ > 0.5,
  greedy, optimal by the Hungarian algorithm for n ≤ ~200); TP/FP/FN, precision/recall/F1 at τ
  and the F1-vs-τ curve; split/merge detection from the IoU table (one-to-many and many-to-one
  with IoU above a low threshold); PR curve and AP from scored predictions (interpolated and
  stepwise), AP averaged over τ; matching accuracy; boundary distances (directed distances,
  Hausdorff, HD95, ASD) via the distance transform in `lib/distance.ts` (morphology plan).
  Uses the pixel functions and **names** of the segmentation section of `lib/metrics.ts`
  (`confusion`, `precision`, `recall`, `dice`, `jaccard`, `prCurve`) as decided in
  `plan/review-organisation.md` (evaluation moves out of `lib/threshold.ts` into
  `lib/metrics.ts`); object F1 calls the same `precision`/`recall`/`f1` on object counts.
  If the team prefers one file, `eval.ts` becomes the "object level" section of `metrics.ts`
  (open question).
- **`demos/src/lib/stats.ts`** (new): mean, variance, median, MAD; normal CDF/quantile;
  t-distribution CDF/quantile (regularised incomplete beta by continued fraction); one-sample,
  paired and Welch t-tests; CI; Bland–Altman (bias, limits, slope); Bonferroni and BH;
  a seeded two-level simulator (experiments × cells, optional pairing and confounding) and a
  plate simulator (wells × features, gradients, edge effects). Random numbers from
  `lib/random.ts` (`rng`, `gauss`; review-organisation item 19), never `Math.random`.
- **Synthetic data**: the nuclei generator with label-image ground truth in `lib/synth.ts`
  (shared with thresholding, watershed, labelling, NN `unetseg`); the "error generator" for
  `objmatch` (jitter, dilate/erode, merge touching, split, drop dim, add specks, scores) and the
  second-annotator generator live in `demos/src/objmatch/`. The 12-image, 4-day set for
  `leakage` uses the same generator with per-day parameters. Labels via `lib/label.ts`
  (labelling plan).
- **Classifier** for `leakage`: k-NN in `demos/src/leakage/` (brute force on ≤ 3000 × 3000
  samples, feature stack from `featureStack()` in `lib/imageops.ts`); optional forest through
  the shared classifier interface (`lib/classifier.ts`, RF plan). Precompute features once per
  preset; a full re-evaluation of five split strategies should stay under ~200 ms (button, not
  live), measured before any Wasm.
- **Plots** with `lib/plot.ts` (lines, bars, scatter, histograms, heat map with numbers);
  label colours from `lib/colour.ts`; ellipse/shape editing from `lib/shapecanvas.ts`
  (labelling/moments plans) if it fits, otherwise local to `objmatch`.
- **Tests** (`demos/test/eval-test.ts`, `stats-test.ts`, run by `make test`): Hungarian vs
  brute-force enumeration of all matchings on random IoU tables up to 7×7; forced matching =
  optimal matching for τ > 0.5 on random label-image pairs; the by-hand numbers of every unit
  (F1 = 4/7, 6/7, 2/7; AP = 0.6875; Hausdorff 4 and ASD 10/7; Bland–Altman bias −5.33 and SD
  4.08; t = 5.0 paired; BH = 4 hits); t CDF against tabulated quantiles; simulated
  false-positive rate of the correct analysis within 0.05 ± 0.015 (seeded, 4000 runs); the
  cells-as-n false-positive formula against simulation; leakage "nothing to learn" by-image
  accuracy within 0.5 ± 0.05.
- Follow `demos/README.md`: TypeScript, no runtime network, fixed seeds, presets + reset,
  keyboard back to reveal.js, labels ≥ 18 px, everything fits 1150×560 on a slide (`objmatch`:
  image 300 px + IoU table + curve panel; on slides the IoU table can be hidden with
  `data-panels`), static fallback figure for print.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, IoU table by hand, `objmatch` |
| 10:45–12:15 | Unit 2 lecture, Bland–Altman by hand, `contourdist`, `blandaltman` |
| 13:15–14:45 | Unit 3 lecture, "does it leak?" exercise, `leakage`, `testpeek` |
| 15:00–16:30 | Unit 4 lecture + exercise (`pseudorep`, `featurescan`); quizzes as wrap-up |

## Open questions

- **Statistics gap**: the audience has no hypothesis testing. Unit 4 introduces SE, p-value
  and CI by simulation, which is a lot for 1.5 h on top of pseudoreplication, multiple
  testing and batch effects. Options: move multiple testing and `featurescan` into a short
  optional unit, or give unit 4 a longer slot (two sessions).
- **"AP" naming**: teach both the area-under-PR AP and TP/(TP+FP+FN) as "matching accuracy"
  (my proposal), or use the cell-segmentation convention throughout? Day 18 `detection/` must
  use the same choice.
- **Where the code lives**: `lib/eval.ts` + `lib/stats.ts` as proposed, or object evaluation as a
  third section of `lib/metrics.ts` (review-organisation names one metrics file)?
- **`features-use/`** (measurements day): if it becomes core (as gap 4 suggests), unit 4 can
  drop its aggregation slide and recall plate normalisation; if it stays optional, unit 4 keeps
  them. Also check that `featurespace` and `featurescan` do not duplicate the plate picture
  (proposal: `featurespace` shows normalisation, `featurescan` shows confounding and testing).
- **Real data**: a small annotated nuclei set with **two** independent annotations and a clear
  licence would make the inter-annotator ceiling real (the thresholding, watershed and NN plans
  ask for the same search). Until then both annotators are synthetic.
- **Leakage classifier**: k-NN (local, dramatic, no dependency) vs the RF day's forest (students
  know it). Both if the classifier interface exists in time.
- Measurement bias from resolution and partial volume (named in gap 4): one Bland–Altman preset
  here, or a slide on the image-formation day?
- Mixed-effects models: pointer only (current plan), or a math slide?

## References

- Caicedo et al. (2019), *Evaluation of deep learning strategies for nucleus segmentation in
  fluorescence images*, Cytometry A (object-level F1 vs IoU threshold, split/merge errors).
- Maier-Hein et al. (2024), *Metrics reloaded: recommendations for image analysis validation*,
  Nature Methods; Reinke et al. (2024), *Understanding metric-related pitfalls in image
  analysis validation*, Nature Methods.
- Everingham et al. (2010), *The PASCAL Visual Object Classes challenge*, IJCV; Lin et al. (2014),
  *Microsoft COCO*, ECCV (AP definitions).
- Kuhn (1955), *The Hungarian method for the assignment problem*, Naval Res. Logistics Quarterly.
- Huttenlocher, Klanderman & Rucklidge (1993), *Comparing images using the Hausdorff
  distance*, PAMI.
- Bland & Altman (1986), *Statistical methods for assessing agreement between two methods of
  clinical measurement*, Lancet.
- Roberts et al. (2017), *Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure*, Ecography; Kapoor & Narayanan (2023), *Leakage and
  the reproducibility crisis in machine-learning-based science*, Patterns.
- Cawley & Talbot (2010), *On over-fitting in model selection and subsequent selection bias in
  performance evaluation*, JMLR.
- Lazic (2010), *The problem of pseudoreplication in neuroscientific studies*, BMC Neuroscience;
  Lord et al. (2020), *SuperPlots: communicating reproducibility and variability in cell
  biology*, J Cell Biol.
- Benjamini & Hochberg (1995), *Controlling the false discovery rate*, JRSS B.
- Leek et al. (2010), *Tackling the widespread and critical impact of batch effects in
  high-throughput data*, Nat Rev Genet; Caicedo et al. (2017), *Data-analysis strategies for
  image-based cell profiling*, Nature Methods.
