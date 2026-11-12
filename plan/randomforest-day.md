# Plan: a day on decision trees and random forests

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** core day 14 of 18, Part V "Learning from examples" (its first day). Order: graph cuts (day 13) → this → neural networks (day 15).

## Goal of the day

After the day a student can:

1. describe a **decision tree** as a sequence of yes/no questions on single features (axis-aligned splits), draw the partition of a 2D feature space it produces, and read it as a **piecewise-constant function**;
2. compute the **best split** of a small table by hand with **Gini impurity** (entropy / information gain named in one line), and explain why a tree grown to full depth **overfits** (high variance) and what depth, minimum leaf size and stopping do; separate **training, validation and test data** and estimate the test error by a held-out set or **k-fold cross-validation**;
3. explain **bagging** (bootstrap samples, averaging) and the **random feature subset** at each split, say why averaging helps only as far as the trees are **decorrelated**, and use the **out-of-bag error** as a free validation estimate; read class **probabilities** as vote fractions;
4. compute and interpret **feature importance** (impurity decrease, permutation importance) and recognise its pitfalls with **correlated** features;
5. do **pixel classification** from a few painted labels with a **multiscale filter feature stack** (semantic segmentation without deep learning), run the "paint – train – look – correct" loop, and avoid its traps: spatially correlated training pixels, label imbalance, evaluation on the wrong pixels;
6. classify **objects** from measured features with a forest, and name the relatives: extremely randomised trees, boosted trees, regression forests (e.g. for counting).

Why it fits this audience: a tree is nested `if`s on numbers; impurity is a function of class
proportions they can evaluate on a calculator; bagging is "the variance of a mean", which they
know from basic statistics, with one extra term for correlation. The image side reuses filters
they know (Gaussian, derivatives, Laplacian from `kernels/`, day 1; LoG/DoG, Hessian and
structure tensor from the edges day, day 2). No machine learning is assumed: this is the
course's first learning day and the **owner of the classification basics** (feature space,
training/validation/test data, overfitting, cross-validation, the pixel feature stack and the
paint – train – correct loop), in the concept layer with pictures. Every later day that
classifies (neural networks day 15, evaluation day 16, and the electives SVM, GMM, integral
images, moments, measurements' feature tables) recalls them from here in one slide.

## Prerequisite gap / overlaps

Ownership follows `plan/coursemap.md` (decided 2026-11-12). Review-organisation proposed a
day-neutral `classify/` unit on the SVM day as owner of the classification basics; the course
map overrules this: **this day owns them**, and the SVM day (elective) recalls them.

- **Filters**: Gaussian smoothing, derivative filters (Sobel), Laplacian are known from
  `kernels/` (day 1). **LoG/DoG, Hessian eigenvalues and the structure tensor** are taught on the
  edges day (day 2: `scale-space/`, `hessian/`, `structure-tensor/`); unit 4 recalls them in one
  slide of pictures ("what this channel lights up on"). **Box means / local variance**: one line,
  pointer to the integral-images elective (`box-filters/`).
- **Classification basics, decision trees, stumps, bagging, random forests, feature importance,
  the pixel feature stack and the paint – train – look – correct loop: owned here.** The stump
  (unit 1, a depth-1 tree) is recalled by the integral-images elective, which may run before
  this day and then defines it in one slide itself.
- **Boosting: owned by the integral-images elective** (`viola-jones/`: AdaBoost with stumps, the
  cascade). Here: one slide **bagging vs boosting** in unit 2, and gradient-boosted trees only as a
  contrast in optional unit 5. Code: `lib/boost.ts` calls the weighted split search of
  `lib/forest.ts` (one split search, no second stump search).
- **Feature stack and pixel classification (decided):** this day introduces the stack and the
  loop (unit 4); the forest needs no feature scaling and almost no tuning, so the lesson stays on
  the features, the labels and honest evaluation. It also owns the two pitfalls common to every
  pixel classifier: **spatially correlated training pixels** (OOB optimism vs a held-out region)
  and **label imbalance**.
  - The **SVM elective** recalls the stack in one slide and keeps what is specific to SVMs:
    feature **standardisation** (fatal for RBF, irrelevant for a forest: a contrast preset),
    $(C, \gamma)$ cross-validation **by stroke**, one-vs-rest / one-vs-one.
  - The **integral-images elective** keeps "why large box scales are free" and the box/Haar
    group; it stops at the feature vector.
  - **One demo shell** `pixelclass` with `data-classifier="forest|svm|…"` and
    `data-stack="gaussian|box"`; the SVM day's *cross-validate* tab is a tab of the shared shell.
    **One stack builder** `featureStack(img, groups, sigmas)` in `lib/imageops.ts` (owner: this
    day; not `lib/pixelfeatures.ts`, and not `lib/features.ts`, which holds **object** features).
- **Measurements** (day 11): owns the object feature catalogue (`lib/features.ts`) and the
  feature tables. Its optional `features-use/` is dissolved (review-organisation §2.6):
  classification of feature tables is unit 3 here, which uses that day's seeded synthetic
  population as data. Shape features (`shape-features/`) are now on the labelling day (day 9).
- **Moments** (elective, Objects group): nearest-mean / nearest-neighbour classification of
  shape descriptors in its optional `moment-features/` recalls this day's vocabulary. Unit 1 names
  nearest neighbour in one line as another way to carve up feature space.
- **Thresholding** (day 6, `seg-eval/`): owns **pixel-level evaluation** (confusion matrix,
  precision/recall, Dice/Jaccard, ROC and PR). Reused, not re-taught: sweeping the forest's
  probability threshold traces that unit's ROC/PR curves (`lib/metrics.ts`). Its pointer
  "pixel classifiers (the step to machine learning, one slide)" in `threshold-more/` points here.
- **Evaluation and statistics** (day 16): owns **object-level evaluation** (matching by IoU,
  object F1/AP), **data leakage** and statistics of measurements. Here only one line in unit 4:
  "split by image (or region), never by pixel; day 16 does this in full".
- **Graph cuts** (day 13, `seg-energy/`, `graphcut-seg/`): back-reference. The forest's
  probability gives a data term $-\log P(\ell \mid \text{features}_p)$; smoothing a noisy pixel
  classification with a graph cut is one slide and one preset (the `graphcut` demo takes a data
  term), not a new demo.
- **Neural networks** (day 15, the next day; its unit 1 is the built `neuralnet/` with demo
  `nnfit`): forward pointers only. Unit 1: a regression tree fits **steps**, a ReLU network fits
  **hinges** (`nnfit`). Unit 4: the feature stack is a hand-made first layer; a CNN / U-Net (day 15
  `cnn/`, `unet/`) learns its own filters and needs many more annotations; a forest on a stack is
  the strong baseline from a few strokes. Softmax/cross-entropy are not needed here (vote
  fractions).
- **Entropy**: owned by the compression elective (`entropy/`). Gini is the main split criterion
  here; entropy gets one line ("expected surprise; the full story is on the compression day").
- **GMMs** (elective, More learning, after this day): unsupervised and generative. Here one
  sentence: a forest is discriminative and, unlike a mixture, needs labels. The shared 2D datasets
  module `lib/datasets2d.ts` is owned here (see Shared engineering) and reused by the GMM and SVM
  electives.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `dtree/`: Decision trees

Concept layer:

- Classification as **carving up feature space**: each pixel or object is a point (features as
  axes), each class a colour. Training data = points with known class; the classifier must
  colour the rest of the plane. Nearest neighbour in one line as the simplest way to do it.
- A **decision tree**: ask "is feature $j$ below $t$?", go left or right, repeat; a **leaf**
  holds a class (or class proportions). Picture: the tree diagram next to the plane cut into
  axis-aligned rectangles. A **stump** is a tree with one question.
- **Growing**: try every feature and every threshold between consecutive values; keep the
  split that makes the two children **purest**. Then do the same in each child (**recursive
  partitioning**). Greedy: the best first question is not always part of the best tree (XOR
  picture: no single split helps, two do).
- **Impurity** in words: 0 when a node holds one class, largest when the classes are mixed
  evenly. Gini = "chance that two random points in the node have different classes" (the
  criterion used all day). Entropy in one line: "how many yes/no questions are still needed";
  it chooses almost the same splits (the full story is on the compression elective).
- **When to stop**: maximum depth, minimum points per leaf, minimum impurity decrease. A tree
  grown until every leaf is pure **memorises** the training points: perfect on training data,
  jagged boundaries, a thin sliver around every noisy point. Train vs test error against depth:
  the U-shaped test curve. Small changes in the data → a completely different tree (**high
  variance**): this instability is what unit 2 exploits.
- **Training, validation and test data** (owned here for the course): the training set fits the
  model, the validation set chooses settings (depth, leaf size), the test set is touched once at
  the end. **k-fold cross-validation** when data are scarce: k rotations, average the error. One
  picture of a 5-fold split. One sentence on the trap that matters for images: pixels or cells
  from the same image are not independent, so split by image (unit 4; day 16 owns leakage).
- A tree as a **piecewise-constant function**; **regression trees** briefly: leaves hold the
  mean value instead of a class; split to reduce the spread. Picture: a step-function fit to a
  noisy intensity profile across a cell, next to the ReLU hinge fit of `nnfit`.
- Strengths already visible: no scaling of features needed (only the order of the values
  matters), mixes features of any units, prediction = a handful of comparisons.

The math (separate slides after the intuition):

- Class proportions $p_k$ in a node. $G = 1 - \sum_k p_k^2 = \sum_{k \ne l} p_k p_l$;
  entropy $H = -\sum_k p_k \log_2 p_k$ in one line (owned by the compression elective); misclassification $1 - \max_k p_k$ (and why it is a poor
  splitting criterion: not strictly concave, so a split can improve the children's purity
  without lowering it).
- Split of $n$ points into $n_L, n_R$: impurity decrease
  $\Delta = I(\text{parent}) - \tfrac{n_L}{n} I(L) - \tfrac{n_R}{n} I(R) \ge 0$ for concave $I$
  (Jensen). Information gain = $\Delta$ with $H$.
- Cost of the search: sort once per feature, sweep thresholds updating class counts:
  $O(d\, n \log n)$ per node. Finding the optimal tree is NP-hard (stated) → greedy.
- Regression: leaf value = mean, split criterion = decrease in sum of squared errors (variance
  reduction).

**Demo `treeplay`** (new): left, a 2D plane (two features, unit square); right, the tree diagram
(SVG, nodes show class counts and impurity). Click to add points of the current class (2–3
classes, class picker; right-click/drag to delete). Buttons: *next split* (grows the
highest-gain open node, highlights it, shows the chosen line), *grow to depth*, *reset*.
Below the plane, for the selected node, the **split scan**: impurity decrease vs threshold along
$x$ and along $y$ as two curves with the best marked; clicking a curve forces a different split.
Sliders: max depth (1–12), min points per leaf; criterion Gini / entropy / misclassification.
Toggle *test points* (a seeded hidden set from the same generator) with train/test error and the
error-vs-depth curve. `data-mode="regression"`: 1D points, step-function fit, depth slider.
Presets: "two blobs: one stump", "diagonal boundary: staircase", "XOR: first split looks
useless", "one mislabelled point: deep tree draws an island", "noisy overlap: test error vs
depth", "regression: steps across a cell edge".
Sizes: ≤ 500 points; full regrow per slider move < 5 ms.

By hand:

- (a) The table below (8 pixels, features: intensity $I$ and local variance $V$, classes
  nucleus N / background B). Compute the parent Gini, then the weighted Gini of every split on
  $I$ and on $V$; which question does the root ask?

  | $I$ | 20 | 35 | 45 | 50 | 60 | 80 | 90 | 110 |
  |---|---|---|---|---|---|---|---|---|
  | $V$ | 2 | 3 | 5 | 9 | 12 | 4 | 15 | 11 |
  | class | B | B | N | B | N | B | N | N |

  Teacher answer: parent $G = 0.5$; best split on $I$ ($I \le 40$ or $I \le 85$) gives
  weighted $G = 0.333$; best on $V$ ($V \le 4.5$ or $V \le 10$) gives $0.200$, so the root
  splits on the **texture** feature, not on intensity. Same ranking with entropy (gain
  0.311 vs 0.549 bits).
- (b) Draw the partition for a given depth-2 tree; classify three points.
- (c) Concept: what happens to training and test error when max depth goes from 2 to 20?
  Which depth would 5-fold cross-validation pick, given five (fold, depth, error) rows?
- (d) *(math)* Show $1 - \sum p_k^2 = \sum_{k\ne l} p_k p_l$, and that for two classes Gini and
  entropy are both maximal at $p = 1/2$; show the impurity decrease is $\ge 0$ for a concave $I$.

### 2. `forest/`: Bagging and random forests

Concept layer:

- One deep tree: low bias, high variance (unit 1). Idea: grow **many** trees on slightly
  different data and let them **vote**.
- **Bootstrap**: draw $n$ points from the $n$ training points **with replacement**; some appear
  twice, about a third not at all. Each tree sees its own bootstrap sample (**bagging** =
  bootstrap aggregating).
- Averaging helps only if the trees make **different** mistakes. Bagged trees on the same data
  still look alike (the same strong feature at the root). **Random forest**: at each split,
  look only at a random subset of $m$ features (typical $m \approx \sqrt d$): trees become less
  alike, their average better.
- **Out-of-bag (OOB) error**: each point is left out by about a third of the trees; let those
  trees vote on it. A validation estimate for free, without a separate test set (with a caveat
  for images: unit 4).
- **Probabilities** from vote fractions: 70 of 100 trees say "nucleus" → $P \approx 0.7$. The
  probability map, and where it is uncertain (boundaries, unlabelled kinds of structure).
  Caveat: far from all training data a forest can still be confident (it extends its last
  rectangles outward): vote fraction is not calibrated certainty.
- Hyperparameters: **number of trees** (more never hurts accuracy, only time; the OOB curve
  flattens), **depth / leaf size** (forests usually grow deep trees), **$m$** (the one that
  matters most). Little tuning compared with most classifiers.
- **Bagging vs boosting** (one slide, pointer to the integral-images elective): bagging = independent
  deep trees, averaged, reduces variance; boosting = shallow trees in sequence, each focusing on
  the previous mistakes, reduces bias.

The math:

- Bootstrap: $P(\text{point not drawn}) = (1 - 1/n)^n \to e^{-1} \approx 0.368$.
- $B$ predictors, each with variance $\sigma^2$ and pairwise correlation $\rho$: the variance
  of their average is
  $\operatorname{Var}\!\big(\tfrac1B\sum_b f_b\big) = \rho\sigma^2 + \tfrac{1-\rho}{B}\sigma^2$
  (derive from $\operatorname{Var}(\sum_b f_b) = \sum_b \operatorname{Var} f_b + \sum_{b \ne c}\operatorname{Cov}(f_b, f_c)$).
  As $B \to \infty$ only $\rho\sigma^2$ remains: more trees cannot beat correlation, smaller
  $m$ lowers $\rho$ (at some cost in each tree's own accuracy).
- Bias–variance decomposition of squared error (stated, one line); averaging leaves the bias
  unchanged.
- Forest prediction $\hat p_k(\mathbf x) = \tfrac1B \sum_b \hat p_{k,b}(\mathbf x)$ (average
  of leaf proportions, or majority vote); a forest as an adaptive nearest-neighbour weighting
  (points sharing leaves with $\mathbf x$), one remark.

**Demo `forestplay`** (new; same playground shell as `treeplay`, `data-mode="forest"`): the 2D
plane with the **averaged probability map** (class colours mixed, saturation = confidence),
next to a strip of 6 small maps of individual trees. Sliders: number of trees $B$ (1–200),
max depth, min leaf size, $m$. For $m$ to matter in a 2D picture the data carries extra derived
features (e.g. $x$, $y$, $x+y$, $x-y$, and 0–4 pure-noise features, toggle), so $d$ goes up to 8.
Readouts: OOB error, test error, a curve of OOB and test error vs $B$; average pairwise
correlation of the trees' outputs ($\hat\rho$) on a grid. Toggle *single tree* vs *forest*.
Presets: "one deep tree overfits", "100 bagged trees smooth it", "$m = d$: trees look alike",
"$m = 1$: diverse trees", "noise features: forest shrugs", "confident far from the data",
"uncertainty sits on the boundary".
Sizes: 300 points, 200 trees of depth ≤ 12 → training < 50 ms; probability map on a
128×128 grid: 16 k × 200 × ~10 node visits ≈ 33 M → ~50–100 ms in TS, so redraw on slider
release and draw a 64×64 preview while dragging.

**Demo `bootstrap`** (new, small): two panels. (a) 20 numbered points; *draw bootstrap sample*
shows each point's multiplicity (0, 1, 2, …) and greys out the OOB ones; *draw 1000* builds the
histogram of the OOB fraction around 0.368. (b) Simulated predictors with chosen $\sigma$,
$\rho$: variance of the average vs $B$ as dots (Monte Carlo, seeded) on top of the formula
curve and its asymptote $\rho\sigma^2$. Presets: "$\rho = 0$", "$\rho = 0.5$: the floor", "small
$n$".

By hand:

- (a) Probability that a given point is in none of 3 bootstrap samples of size $n = 5$.
- (b) $\sigma^2 = 1$, $\rho = 0.3$: variance of the average of 1, 10, 100, ∞ trees; how far must
  $\rho$ drop to beat 100 trees at $\rho = 0.3$ with only 10 trees?
- (c) Five trees vote N, N, B, N, B on a pixel; three of them did not see it in training (given
  which): forest probability and the OOB vote.
- (d) Concept: why does using all features at every split ($m = d$) make the trees more alike?
- (e) *(math)* Derive the variance formula; show $(1 - 1/n)^n \to 1/e$.

### 3. `forest-objects/`: Forests on feature tables, and feature importance

Concept layer:

- Object classification: segment, measure (measurements, day 11), one row per object, a forest per
  row: e.g. mitotic vs interphase nuclei, debris vs cells. Why forests are the default here:
  dozens of **heterogeneous features** in different units (area in µm², intensity in counts,
  ratios) with **no scaling**, irrelevant features mostly ignored, little tuning, fast
  prediction, OOB error for free.
- **Which features matter?** Two answers:
  - **Impurity (Gini) importance**: add up the impurity decrease of every split that used the
    feature, over all trees. Free, but biased toward features with many distinct values
    (continuous noise can look important) and computed on training data.
  - **Permutation importance**: shuffle one feature column in held-out (or OOB) data and see how
    much the error goes up. Honest about prediction, but slower.
- **Pitfall: correlated features.** Area and perimeter carry the same information. Impurity
  importance **splits** the credit between them (each looks half as important); permutation
  importance can make **both** look unimportant (shuffle one, the other still carries the
  information) and shuffling creates impossible objects (huge area, tiny perimeter). Importance
  is about *this model*, not about biology or causation. Remedy in words: group correlated
  features (the correlation blocks of day 11) and permute groups.
- Feature importance as a guide for **which measurements to keep**, not as proof.

The math:

- MDI of feature $j$: $\frac1B\sum_b \sum_{\text{nodes } t \text{ split on } j} \frac{n_t}{n}\,\Delta I_t$.
- Permutation importance: $\mathrm{err}(\text{data with column } j \text{ permuted}) - \mathrm{err}(\text{data})$,
  averaged over several permutations; why it is ≈ 0 for a feature with an exact duplicate.

**Demo `featimp`** (new): data = a seeded synthetic object population from the measurements
plan's generator (two phenotypes differing in size and texture), table of ~8 features: area,
perimeter, equivalent diameter (all correlated), mean intensity, a texture feature, solidity,
one continuous noise feature, one binary noise feature. Panels: a scatter of two chosen features
coloured by class with the forest's decision map in that plane; bar charts of impurity importance
and permutation importance (OOB) side by side; OOB error. Toggles: include/exclude each feature,
*duplicate area*, *permute group* (area-like block). Presets: "texture decides", "duplicate a
feature: credit splits", "correlated block: permutation hides both", "permute the block
together", "continuous noise looks important (impurity) but not (permutation)".
Sizes: ≤ 2000 objects, 100 trees → < 100 ms; permutation importance: 8 columns × 5 repeats of
OOB prediction, < 200 ms.

By hand:

- (a) A given depth-2 tree with node counts: impurity importance of each feature.
- (b) Error on 10 held-out objects before and after permuting each of two columns (given):
  permutation importances.
- (c) Concept: area and equivalent diameter are both in the table. Predict both importances;
  what happens when one is removed?
- (d) *(math)* Show that if feature $j'$ is an exact copy of $j$ and the tree picks either at
  random, the expected impurity importance is split in half.

### 4. `pixelclass/`: Pixel classification with a forest

Concept layer:

- Goal: label every pixel (nucleus / cytoplasm / background / …) from a **few painted strokes**:
  semantic segmentation without deep learning. Every pixel becomes a point in feature space;
  the painted pixels are the training set; the forest colours the rest.
- Why intensity alone fails (the `treeplay` table again: texture beats intensity), and why one
  scale is not enough. The **feature stack**: each filter at several scales $\sigma$ (e.g.
  0.7, 1, 1.6, 3.5, 5, 10 px) gives one channel; a pixel's feature vector = its value in every
  channel (~35–40 numbers). Families, each shown as an image of the micrograph:
  - **intensity**: Gaussian-smoothed image (what is the brightness here, at this scale?);
    box mean and local variance are the cheap cousins (integral-images elective, pointer);
  - **edges**: gradient magnitude (is there a boundary near?);
  - **blobs**: Laplacian of Gaussian / difference of Gaussians (a bright spot of this size?);
  - **ridges and texture**: Hessian eigenvalues (line-like vs blob-like), structure tensor
    eigenvalues (oriented texture, corners).
  Blobs, ridges and texture are **one recall slide** of day 2 (`scale-space/`, `hessian/`,
  `structure-tensor/`): one image per family, no derivation. Forests take this pile of channels
  as is: no scaling, redundant channels harmless.
- The **interactive loop**: paint a little → train → look at the result **and the uncertainty
  map** → paint where it is wrong or uncertain → repeat. Few strokes go a long way.
- Pitfalls:
  - **Spatially correlated training pixels**: neighbouring pixels in a stroke are near
    copies. OOB error is then wildly optimistic (an "unseen" pixel has its neighbour in the
    bag). Evaluate on a **held-out region** of the image (or another image), with the
    confusion matrix and Dice of `seg-eval/` (day 6). Rule: split by image or region, never by
    pixel; day 16 treats leakage and object-level evaluation in full. The SVM elective meets
    the same trap as cross-validation by pixel vs by stroke (one-line pointer).
  - **Label imbalance**: a big background brushstroke vs a few nucleus pixels pushes the
    probabilities toward background; balance by sampling per class or weighting.
  - **Painting only the easy cases** (bright nuclei): dim or dividing nuclei are classified
    as background. The forest knows only what it has been shown; the uncertainty map hints
    where to paint, but cannot flag what looks confidently like something else.
  - Scale: features larger than the structures blur boundaries; features too small miss
    texture.
- From probabilities to objects: threshold the probability map (ROC/PR as in `seg-eval/`),
  clean up (morphology), or smooth with a graph cut using $-\log P$ as the data term
  (back-reference to day 13, `graphcut-seg/`), then label and measure.
- Contrast (one slide): a convolutional network (day 15, `cnn/`, `unet/`) learns its own
  filters from far more labels; the forest + hand-made stack is the strong baseline that works
  from a few strokes. Contrast with the SVM on the same stack: the SVM needs standardised
  features and a $(C, \gamma)$ search, the forest does not (pointer to the SVM elective,
  `pixel-classify/`).
  One-line pointer: interactive pixel-classification programs implement exactly this loop.

The math:

- The stack as formulas, one recall slide of day 2: $L_\sigma = G_\sigma * I$,
  $\lVert\nabla L_\sigma\rVert$, $\sigma^2 \nabla^2 L_\sigma$, eigenvalues of the Hessian and
  of the structure tensor (derived on the edges day, not here).
- Why OOB fails here: the bootstrap assumes independent samples; correlated samples → leakage.
  Effective sample size in one sentence.
- Class weighting: weighted Gini with class weights $w_k$; equivalent to resampling.

**Demo `pixelclass`** (new, main demo; **one shell shared with the SVM day**, replacing a separate
`pixelsvm`, and able to use the integral-image day's box stack): the micrograph
`demos/src/fourier2d/cell256.png` (256×256, greyscale), plus a synthetic image with ground truth
(blobs of two textures and a dim class, from `lib/synth.ts` generators) for honest evaluation.
Left: image with painted labels (2–4 classes, brush size, eraser). Right: overlay switch —
segmentation, probability of a chosen class, **uncertainty** (1 − max probability, or vote
entropy), or one feature channel (feature-stack browser: family × scale grid of thumbnails;
click a pixel to show its feature vector as bars). Controls: feature-group checkboxes
(intensity / edges / blobs / ridges-texture; small / medium / large scales), trees, $m$,
*balance classes*, *auto-retrain after each stroke*, *draw held-out region* (a rectangle excluded
from training; on the synthetic image its Dice and confusion matrix are shown next to the OOB
estimate), `data-classifier` = `forest` (this day) | `svm` | … (other days plug in a train/predict
pair with the same interface; the SVM day adds its *cross-validate* tab), `data-stack` =
`gaussian` | `box`, `data-size` = 256 (forest) or 128 (SVM default, for speed). Presets: "two strokes", "intensity only vs full stack", "drop the
large scales: texture lost", "only bright nuclei painted", "background brushstroke 20× larger:
imbalance", "OOB 99 %, held-out region 85 %", "uncertainty shows where to paint next", "add a third
class (cytoplasm)".
Sizes and speed: feature stack 256×256 × ~37 channels (Float32, ≈ 9.7 MB), computed once per image
with separable Gaussians (~50–100 ms in TS, cached). Training: painted pixels capped at ~5000
(per-class subsampling), features pre-binned to 64 levels so split search is a histogram sweep;
50 trees, depth ≤ 15 → expected 20–60 ms. Prediction over 65 536 pixels × 50 trees × ~12 levels ≈
40 M node visits ≈ 100–200 ms in TS: show a 128×128 preview first, then fill in full resolution
in time-sliced chunks (`requestAnimationFrame`), so painting never blocks. Move training and
prediction to Rust→Wasm only if measured above ~300 ms for a full update on a mid-range laptop
(see Shared engineering).

By hand:

- (a) A 5×5 patch with given values: the pixel's feature vector for three channels (box-smoothed
  value, a central-difference gradient magnitude, a discrete Laplacian), then classify it with a
  given forest of three small trees and give the vote fraction.
- (b) Training pixels: 400 background, 20 nucleus. Predict what the forest does on an ambiguous
  pixel; two ways to fix it.
- (c) Confusion matrix and Dice on a held-out 4×4 region (given prediction and truth); why the
  OOB error was lower.
- (d) Concept: which feature family should separate (i) nuclei from background, (ii) a thin
  membrane from cytoplasm, (iii) a textured from a smooth region of equal mean? At which scale?
- (e) *(math)* Show that weighting class $k$ by $w_k$ in the Gini impurity gives the same
  splits as resampling class $k$ in proportion to $w_k$; which weights balance 400 background
  and 20 nucleus pixels? (Structure-tensor and Hessian exercises are on day 2.)

### 5. (optional / further reading) `forest-more/`: Relatives of the random forest

- **Extremely randomised trees**: thresholds drawn at random too (best of a few random cuts);
  even less correlation, faster training, smoother probability maps. `forestplay` toggle.
- **Gradient-boosted trees** (contrast; boosting owned by the integral-images elective): shallow trees
  added one at a time, each fitted to the current residuals (regression) with a learning rate;
  often a bit more accurate on tables, needs more tuning, can overfit with too many trees
  (unlike a forest). `forestplay` `data-mode="boost"` on the 1D regression data: residuals shrink
  stage by stage.
- **Regression forests** for images: predict a number per pixel. **Counting by density**: dots
  on cell centres → target density = dots blurred by a Gaussian; a forest regresses density
  from the feature stack; the count is the sum of the density over a region, without segmenting
  touching cells.
- **Demo**: `pixelclass` with `data-mode="density"` (paint dots instead of strokes on the
  synthetic crowded-cells image with known count; readout: estimated vs true count in a
  rectangle). Presets: "sparse cells", "touching cells: segmentation fails, density does not".
- By hand: residuals after one boosting stage for 5 values; count from a given 4×4 density map.

## Shared engineering

Library names follow review-organisation §3.1 (one file, one owner).

- `demos/src/lib/forest.ts` (new, **owned by this day**): CART classification and regression
  trees on a feature-major `Float32Array` (or pre-binned `Uint8Array`) with optional **sample
  weights** (class balancing; AdaBoost stumps in `lib/boost.ts` of the integral-images elective
  call **this one weighted split search**, no second implementation); options: max depth, min
  leaf, $m$, criterion (Gini / entropy / misclassification / SSE), random vs best threshold
  (extra-trees); forests with bootstrap and OOB bookkeeping; flat node arrays (feature, threshold,
  left, right, leaf value) for fast prediction; impurity and permutation importance; an
  **instrumented** single-split search returning the impurity curve per feature (for `treeplay`).
  Seeded by `rng(seed)` from `lib/random.ts` (re-exported by `lib/ui.ts`).
- **Feature stack** (**owned by this day**): one builder `featureStack(img, groups, sigmas)` in
  `lib/imageops.ts` (no `lib/pixelfeatures.ts`; `lib/features.ts` is day 11's object features).
  Groups: intensity, gradient magnitude, LoG/DoG, Hessian eigenvalues, structure-tensor
  eigenvalues (the derivative filters come from `lib/edges.ts`, owned by the edges day), box mean
  / local variance / Haar (`lib/integral.ts`, integral-images elective). Returns channel names,
  group and $\sigma$ per channel (for the browser and importance bars), cached per image; a
  `binStack(stack, 64)` helper for the histogram split search. Symmetric 2×2 eigenvalues from
  `lib/linalg.ts` (not `lib/imageops.ts`).
- `demos/src/lib/classifier.ts` (new, tiny, **owned by this day**): the interface the
  `pixelclass` shell and `forestplay` use: `train(X, y, w, groups, opts) → model`,
  `predictScore(model, X) → Float32Array` (vote fraction for the forest, decision value for the
  SVM), `name`, option sliders. Forest here; SVM (`lib/svm.ts`), nearest neighbour, small MLP
  (`lib/nn.ts`) plug in from their days. `groups` = stroke ids, so by-stroke CV and held-out
  strokes work for every classifier.
- `demos/src/lib/datasets2d.ts` (**owned by this day**, the owner of feature spaces; move the
  `ncut/data.ts` `POINT_PRESETS` in; add XOR, staircase-diagonal, rings with label noise) and
  `demos/src/lib/playground.ts` (**owned by this day**): the 2D point playground shell (add/delete
  points, class picker, probability-map rendering on a grid). Used by `treeplay`, `forestplay`,
  and by the SVM (`svm2d`), GMM (`kmeans2d`, `em2d`), neural-network (`nnplayground`) and ncut
  point-mode demos, so all days draw the same clouds.
- `demos/src/lib/paint.ts` (new, **owned by this day**; extract the seed painting from
  `graphcut/main.ts`): brush/eraser label painting over an image, label layers, held-out
  rectangle; used by `pixelclass` and `graphcut`.
- Reused: `lib/plot.ts` (bar charts, error curves, scatter; morphology day), `lib/metrics.ts`
  (confusion matrix, Dice, ROC/PR; segmentation section owned by thresholding), `lib/synth.ts`
  synthetic generators with ground truth (thresholding), day 11's seeded object population and
  `lib/features.ts` (for `featimp`).
- Speed: all TypeScript first. Expected: `treeplay` < 5 ms, `forestplay` ~50–100 ms per full
  map, `pixelclass` ~100–200 ms per full prediction with progressive display. Only if measurements
  exceed ~300 ms per full `pixelclass` update: move tree training + flat-array prediction to Rust→Wasm
  (`demos/wasm/forest/`), keeping `forest.ts` as the reference implementation, per `demos/README.md`.
- Tests (`demos/test/forest-test.ts`, run by `make test`): split search vs brute force on random
  small tables; the unit-1 worked table ($G = 0.2$ on $V$, $0.333$ on $I$); a depth-unlimited tree
  reaches zero training error on distinct points; OOB fraction ≈ 0.368 over many bootstraps;
  variance formula vs Monte Carlo; importance of an exact duplicate ≈ half; feature stack:
  Gaussian of a constant is constant.
- Images: `cell256.png` (via `loadGrey()` in `lib/imageops.ts`); synthetic images with ground
  truth for evaluation and counting. Follow `demos/README.md`: fixed seeds, presets + reset,
  keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide (`pixelclass` uses a
  2-column layout with the controls in a narrow side panel; the feature browser opens as an
  overlay).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture (trees, train/validation/test, cross-validation), Gini table by hand, `treeplay` |
| 10:45–12:15 | Unit 2 lecture + exercise (`bootstrap`, `forestplay`) |
| 13:15–14:45 | Unit 3 lecture + exercise (`featimp`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`pixelclass`); quizzes as wrap-up |

## Open questions

- **Decided (course map):** feature stack and pixel classification are introduced here (unit 4)
  and recalled on the SVM and integral-images electives; one shell `pixelclass` (absorbing
  `pixelsvm`) and one builder `featureStack` in `lib/imageops.ts`.
- **Decided (course map):** classification basics (train/test, overfitting, cross-validation,
  feature stack) are owned here, not by a day-neutral `classify/` on the SVM day as
  review-organisation proposed.
- **Decided (course map):** boosting is owned by the integral-images elective; this day owns
  trees/stumps/bagging/forests, one contrast slide plus optional gradient boosting.
- **Decided (course map):** the structure tensor, Hessian and LoG/DoG are owned by the edges day
  (day 2); unit 4 recalls them.
- Ground truth for `cell256.png`: a manual annotation (nuclei mask, maybe cytoplasm) would allow
  honest held-out evaluation on the real image; the thresholding plan asks for the same. Until
  then held-out Dice is shown only on the synthetic image.
- Unit 3 could be shortened and merged into unit 2 if the day is too full (unit 1 now also
  carries train/validation/test and cross-validation), freeing time for optional unit 5
  (counting) — which is more image-specific?
- Web Worker vs time-sliced main-thread prediction: workers from inlined blob URLs under `file://`
  need testing in all browsers; time slicing is the safe default.

## References

- Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression Trees*, Wadsworth.
- Breiman (1996), *Bagging predictors*, Machine Learning.
- Breiman (2001), *Random forests*, Machine Learning.
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer (ch. 9, 10, 15: trees, boosting, random forests; the variance formula).
- Geurts, Ernst & Wehenkel (2006), *Extremely randomized trees*, Machine Learning.
- Strobl, Boulesteix, Zeileis & Hothorn (2007), *Bias in random forest variable importance measures*, BMC Bioinformatics.
- Strobl et al. (2008), *Conditional variable importance for random forests*, BMC Bioinformatics.
- Criminisi, Shotton & Konukoglu (2012), *Decision forests: a unified framework for classification, regression, density estimation, manifold learning and semi-supervised learning*, Foundations and Trends in Computer Graphics and Vision.
- Sommer, Straehle, Köthe & Hamprecht (2011), *ilastik: Interactive learning and segmentation toolkit*, ISBI (teacher reference for the feature stack and the labelling loop).
- Berg et al. (2019), *ilastik: interactive machine learning for (bio)image analysis*, Nature Methods (teacher reference).
- Lempitsky & Zisserman (2010), *Learning to count objects in images*, NeurIPS; Fiaschi et al. (2012), *Learning to count with regression forest and structured labels*, ICPR.
- Friedman (2001), *Greedy function approximation: a gradient boosting machine*, Annals of Statistics.
- Roberts et al. (2017), *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography (spatially blocked validation).
