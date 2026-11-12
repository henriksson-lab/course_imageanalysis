# Plan: a day on decision trees and random forests

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. describe a **decision tree** as a sequence of yes/no questions on single features (axis-aligned splits), draw the partition of a 2D feature space it produces, and read it as a **piecewise-constant function**;
2. compute the **best split** of a small table by hand with **Gini impurity** or **entropy / information gain**, and explain why a tree grown to full depth **overfits** (high variance) and what depth, minimum leaf size and stopping do;
3. explain **bagging** (bootstrap samples, averaging) and the **random feature subset** at each split, say why averaging helps only as far as the trees are **decorrelated**, and use the **out-of-bag error** as a free validation estimate; read class **probabilities** as vote fractions;
4. compute and interpret **feature importance** (impurity decrease, permutation importance) and recognise its pitfalls with **correlated** features;
5. do **pixel classification** from a few painted labels with a **multiscale filter feature stack** (semantic segmentation without deep learning), run the "paint – train – look – correct" loop, and avoid its traps: spatially correlated training pixels, label imbalance, evaluation on the wrong pixels;
6. classify **objects** from measured features with a forest, and name the relatives: extremely randomised trees, boosted trees, regression forests (e.g. for counting).

Why it fits this audience: a tree is nested `if`s on numbers; impurity is a function of class
proportions they can evaluate on a calculator; bagging is "the variance of a mean", which they
know from basic statistics, with one extra term for correlation. The image side reuses filters
they know (Gaussian, derivatives, Laplacian from `kernels/`). No machine learning is assumed:
the day introduces training/test data, overfitting and validation itself (in the concept layer,
with pictures), so it stands alone.

## Prerequisite gap / overlaps

Written against the parallel drafts `plan/svm-day.md`, `plan/integral-day.md`,
`plan/neuralnets-day.md` and `plan/gmm-day.md` (all 2026-11-12). The splits marked
**proposal** need a reply from those plans.

- **Filters** (`kernels/`): Gaussian smoothing, derivative filters (Sobel), Laplacian are
  known. Unit 4 also uses the **Laplacian of Gaussian / difference of Gaussians** (owned by the
  wavelets day, `cwt/`), **Hessian eigenvalues** (owned by the skeleton day,
  `skel-neurovessel/`: tube vs blob from the eigenvalues), **box means / local variance** (owned
  by the integral-image day, `box-filters/`, `rect-features/`) and the **structure tensor**
  (no owner found: concept slide + math slide here). Each owned filter is recalled as one
  picture ("what this channel lights up on") with a pointer.
- **Decision trees, stumps, bagging, random forests, feature importance: owned here.** The
  integral-image plan asks (open question) which day owns trees and forests and wants stumps
  introduced once: proposal: the **stump** is defined here (unit 1, a depth-1 tree) and recalled
  in one line on the integral-image day.
- **Boosting: owned by the integral-image day** (`viola-jones/`: AdaBoost with stumps, the
  cascade, at concept level with a math layer). Here: one slide **bagging vs boosting** in unit 2
  (independent deep trees averaged → less variance; shallow trees in sequence on reweighted data →
  less bias), and gradient-boosted trees only as a contrast in optional unit 5. Code: `lib/boost.ts`
  (integral plan) can call the weighted split search of `lib/forest.ts` with the
  weighted-error criterion instead of keeping a second stump search.
- **Feature stack and pixel classification: shared with the SVM day (`pixel-classify/`, demo
  `pixelsvm`) and the integral-image day (`rect-features/`, stack tab of `haarexplore`).**
  Today all three plans introduce a per-pixel multiscale feature stack. Proposal:
  - **This day introduces the feature stack and the paint – train – look – correct loop**
    (unit 4): the forest is the classic classifier for it, needs no feature scaling and almost no
    tuning, so the lesson stays on the features, the labels and honest evaluation. It also owns the
    two pitfalls common to every pixel classifier: **spatially correlated training pixels**
    (shown here as OOB optimism vs a held-out region) and **label imbalance**.
  - The **SVM day** recalls the stack in one slide and keeps what is specific to SVMs: feature
    **standardisation** (fatal for RBF, irrelevant for a forest: a nice contrast preset),
    $(C, \gamma)$ cross-validation **by stroke**, one-vs-rest / one-vs-one.
  - The **integral-image day** keeps "why large box scales are free" and presents box/Haar
    channels as an alternative stack family; it stops at the feature vector, as its plan says.
  - **One demo shell** instead of `pixelsvm` + `pixelclass`: `pixelclass` with
    `data-classifier="forest|svm|…"` and `data-stack="gaussian|box"`; the SVM day's *evaluate* and
    *cross-validate* tabs become tabs of the shared shell. **One stack builder**:
    `featureStack(img, groups, sigmas)` in `lib/imageops.ts`, as the SVM plan already proposes
    (not a separate `lib/pixelfeatures.ts`; and not `lib/features.ts`, which the measurements plan
    uses for **object** features). This day adds the structure-tensor group; the integral day adds
    the box group.
- **Measurements** (`plan/measurements-day.md`): owns the object feature catalogue
  (`lib/features.ts`), redundancy/correlation of features and the `featurespace` demo
  (`features-use/`, "classification briefly"). This day owns **classifying objects with a
  forest** and **feature importance**; unit 3 uses that day's seeded synthetic population as data,
  and its "one slide on overfitting" can point to unit 1 here. The SVM plan also uses object
  tables as an example: there for the margin, here for importance.
- **Moments** (`plan/moments-day.md`, `moment-features/`): owns nearest-mean / nearest-neighbour
  classification of shape descriptors and leave-one-out error. Unit 1 recalls nearest neighbour in
  one line as another way to carve up feature space; `momentclass` could get a forest option via
  the shared classifier interface (decide with the moments and SVM plans, which ask the same).
- **Thresholding** (`plan/thresholding-day.md`, `seg-eval/`): owns **evaluation** (confusion
  matrix, precision/recall, Dice/Jaccard, ROC and PR). Reused, not re-taught: a forest gives a
  probability map, and sweeping its threshold traces that unit's ROC/PR curves (`lib/threshold.ts`).
  Its "pixel classifiers (the step to machine learning, one slide)" in `threshold-more/` points here.
- **Graph cuts** (`seg-energy/`, `graphcut-seg/`): the forest's probability gives a data term
  $-\log P(\ell \mid \text{features}_p)$; smoothing a noisy pixel classification with a graph cut
  is one slide and one preset (the `graphcut` demo takes a data term), not a new demo. The SVM and
  neural-network plans make the same link; say it once per day in one line.
- **Neural networks** (`plan/neuralnets-day.md`: `neurons/`, `backprop/`, `cnn/`, `unet/`; and
  `neuralnet/` with demo `nnfit`): contrast only. Unit 1: a regression tree fits **steps**, a ReLU
  network fits **hinges** (`nnfit`). Unit 4: the feature stack is a hand-made first layer; a CNN /
  U-Net (`unet/`) learns its own filters and needs many more annotations; a forest on a stack is the
  strong baseline from a few strokes. Softmax/cross-entropy are not needed here (vote fractions).
- **GMMs** (`plan/gmm-day.md`: `kmeans/`, `gaussian-mixture/`, `gmm-images/`): unsupervised and
  generative; the generative-vs-discriminative slide lives there and on the SVM day. Here one
  sentence: a forest is discriminative and, unlike a mixture, needs labels. The GMM plan's shared
  2D datasets module (`lib/datasets2d.ts`) is reused by `treeplay`/`forestplay` (XOR and staircase
  sets added there).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `dtree/`: Decision trees

Concept layer:

- Classification as **carving up feature space**: each pixel or object is a point (features as
  axes), each class a colour. Training data = points with known class; the classifier must
  colour the rest of the plane. Recall nearest neighbour in one line (moments day).
- A **decision tree**: ask "is feature $j$ below $t$?", go left or right, repeat; a **leaf**
  holds a class (or class proportions). Picture: the tree diagram next to the plane cut into
  axis-aligned rectangles. A **stump** is a tree with one question.
- **Growing**: try every feature and every threshold between consecutive values; keep the
  split that makes the two children **purest**. Then do the same in each child (**recursive
  partitioning**). Greedy: the best first question is not always part of the best tree (XOR
  picture: no single split helps, two do).
- **Impurity** in words: 0 when a node holds one class, largest when the classes are mixed
  evenly. Gini = "chance that two random points in the node have different classes". Entropy =
  "how many yes/no questions are still needed". In practice they choose almost the same splits.
- **When to stop**: maximum depth, minimum points per leaf, minimum impurity decrease. A tree
  grown until every leaf is pure **memorises** the training points: perfect on training data,
  jagged boundaries, a thin sliver around every noisy point. Train vs test error against depth:
  the U-shaped test curve. Small changes in the data → a completely different tree (**high
  variance**): this instability is what unit 2 exploits.
- A tree as a **piecewise-constant function**; **regression trees** briefly: leaves hold the
  mean value instead of a class; split to reduce the spread. Picture: a step-function fit to a
  noisy intensity profile across a cell, next to the ReLU hinge fit of `nnfit`.
- Strengths already visible: no scaling of features needed (only the order of the values
  matters), mixes features of any units, prediction = a handful of comparisons.

The math (separate slides after the intuition):

- Class proportions $p_k$ in a node. $G = 1 - \sum_k p_k^2 = \sum_{k \ne l} p_k p_l$;
  $H = -\sum_k p_k \log_2 p_k$; misclassification $1 - \max_k p_k$ (and why it is a poor
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
- **Bagging vs boosting** (one slide, pointer to the integral-image day): bagging = independent
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

- Object classification: segment, measure (measurements day), one row per object, a forest per
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
  features (the correlation blocks of the measurements day) and permute groups.
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
    box mean and local variance are the cheap cousins (integral-image day, pointer);
  - **edges**: gradient magnitude (is there a boundary near?);
  - **blobs**: Laplacian of Gaussian / difference of Gaussians (a bright spot of this size?
    recall, wavelets day `cwt/`);
  - **ridges and texture**: Hessian eigenvalues (line-like vs blob-like; recall, skeleton day
    `skel-neurovessel/`), structure tensor eigenvalues (oriented texture, corners; new here).
  Forests take this pile of channels as is: no scaling, redundant channels harmless.
- The **interactive loop**: paint a little → train → look at the result **and the uncertainty
  map** → paint where it is wrong or uncertain → repeat. Few strokes go a long way.
- Pitfalls:
  - **Spatially correlated training pixels**: neighbouring pixels in a stroke are near
    copies. OOB error is then wildly optimistic (an "unseen" pixel has its neighbour in the
    bag). Evaluate on a **held-out region** of the image (or another image), with the
    confusion matrix and Dice of `seg-eval/` (thresholding day). The SVM day meets the same
    trap as cross-validation by pixel vs by stroke (one-line pointer each way).
  - **Label imbalance**: a big background brushstroke vs a few nucleus pixels pushes the
    probabilities toward background; balance by sampling per class or weighting.
  - **Painting only the easy cases** (bright nuclei): dim or dividing nuclei are classified
    as background. The forest knows only what it has been shown; the uncertainty map hints
    where to paint, but cannot flag what looks confidently like something else.
  - Scale: features larger than the structures blur boundaries; features too small miss
    texture.
- From probabilities to objects: threshold the probability map (ROC/PR as in `seg-eval/`),
  clean up (morphology), or smooth with a graph cut using $-\log P$ as the data term
  (pointer to `graphcut-seg/`), then label and measure.
- Contrast (one slide): a convolutional network (`cnn/`, `unet/`) learns its own filters from
  far more labels; the forest + hand-made stack is the strong baseline that works from a few
  strokes. Contrast with the SVM on the same stack: the SVM needs standardised features and a
  $(C, \gamma)$ search, the forest does not (pointer to `pixel-classify/`).
  One-line pointer: interactive pixel-classification programs implement exactly this loop.

The math:

- Gaussian scale space $L_\sigma = G_\sigma * I$; gradient magnitude
  $\lVert\nabla L_\sigma\rVert$; LoG $\sigma^2 \nabla^2 L_\sigma$ (scale normalisation, one
  line) and DoG $\approx$ LoG; Hessian $\mathbf H = \begin{pmatrix} L_{xx} & L_{xy}\\ L_{xy} & L_{yy}\end{pmatrix}$
  and its eigenvalues (both negative: bright blob; one strongly negative: bright ridge);
  structure tensor $\mathbf S = G_\rho * (\nabla L_\sigma \nabla L_\sigma^\top)$ and its
  eigenvalues (one large: edge/oriented texture; both large: corner). Closed-form eigenvalues of a
  symmetric $2\times2$ matrix (recall).
- Why OOB fails here: the bootstrap assumes independent samples; correlated samples → leakage.
  Effective sample size in one sentence.
- Class weighting: weighted Gini with class weights $w_k$; equivalent to resampling.

**Demo `pixelclass`** (new, main demo; **one shell shared with the SVM day**, replacing a separate
`pixelsvm`, and able to use the integral-image day's box stack): the micrograph
`demos/src/fourier2d/cell256.png` (256×256, greyscale), plus a synthetic image with ground truth
(blobs of two textures and a dim class, from `lib/imageops.ts` generators) for honest evaluation.
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
- (e) *(math)* Structure tensor of a patch whose gradients are all $(1, 0)$, and of one with
  gradients $(1,0)$ and $(0,1)$ in equal numbers: eigenvalues, and which is an edge, which a
  corner or isotropic texture? (The Hessian version of this question is on the skeleton day.)

### 5. (optional / further reading) `forest-more/`: Relatives of the random forest

- **Extremely randomised trees**: thresholds drawn at random too (best of a few random cuts);
  even less correlation, faster training, smoother probability maps. `forestplay` toggle.
- **Gradient-boosted trees** (contrast; boosting owned by the integral-image day): shallow trees
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

- `demos/src/lib/forest.ts` (new, owned by this day): CART classification and regression trees
  on a feature-major `Float32Array` (or pre-binned `Uint8Array`) with optional **sample weights**
  (for class balancing and for AdaBoost stumps on the integral-image day); options: max depth,
  min leaf, $m$, criterion (Gini / entropy / misclassification / SSE), random vs best threshold
  (extra-trees); forests with bootstrap and OOB bookkeeping; flat node arrays (feature, threshold,
  left, right, leaf value) for fast prediction; impurity and permutation importance; an
  **instrumented** single-split search returning the impurity curve per feature (for `treeplay`).
  Seeded by `rng(seed)` from `lib/ui.ts`.
- **Feature stack**: one builder `featureStack(img, groups, sigmas)` in `lib/imageops.ts`, as the
  SVM plan proposes (no separate `lib/pixelfeatures.ts`; `lib/features.ts` is the measurements
  plan's object features). Groups: intensity, gradient magnitude, LoG/DoG, Hessian eigenvalues
  (skeleton plan's Gaussian second derivatives), structure-tensor eigenvalues (**added by this
  day**), box mean / local variance / Haar (integral-image plan, via the integral image). Returns
  channel names, group and $\sigma$ per channel (for the browser and importance bars), cached per
  image; a `binStack(stack, 64)` helper for the forest's histogram split search. Symmetric 2×2
  eigenvalues from `lib/linalg.ts` / `lib/imageops.ts` (as agreed in the labelling and skeleton
  plans).
- `demos/src/lib/classifier.ts` (new, tiny): the interface the `pixelclass` shell and `forestplay`
  use: `train(X, y, w, groups, opts) → model`, `predictScore(model, X) → Float32Array` (vote
  fraction for the forest, decision value for the SVM), `name`, option sliders. Forest here; SVM
  (`lib/svm.ts`), nearest neighbour, small MLP (`lib/nn.ts`) plug in from their days. `groups` =
  stroke ids, so by-stroke CV (SVM plan) and held-out strokes work for every classifier.
- `demos/src/lib/playground.ts` (new): the 2D point-painting plane (add/delete points, class
  picker, probability-map rendering on a grid) on top of the seeded generators of
  `lib/datasets2d.ts` (GMM plan; add XOR, staircase-diagonal, rings with label noise). Used by
  `treeplay`, `forestplay`, and offered to the SVM day's `svm2d` so both days draw the same
  clouds.
- `lib/boost.ts` (integral-image plan) may call `forest.ts`'s weighted split search
  (criterion = weighted error) for its stumps rather than keeping a second implementation.
- `demos/src/lib/paint.ts` (new): brush/eraser label painting on a canvas over an image, label
  layers, held-out rectangle; used by `pixelclass` (and by `graphcut` seeds if it is refactored).
- Reused: `lib/plot.ts` (bar charts, error curves, scatter), `lib/threshold.ts` evaluation
  (confusion matrix, Dice, ROC/PR), `lib/imageops.ts` synthetic generators with ground truth,
  the measurements plan's seeded object population and `lib/features.ts` (for `featimp`).
- Speed: all TypeScript first. Expected: `treeplay` < 5 ms, `forestplay` ~50–100 ms per full
  map, `pixelclass` ~100–200 ms per full prediction with progressive display. Only if measurements
  exceed ~300 ms per full `pixelclass` update: move tree training + flat-array prediction to Rust→Wasm
  (`demos/wasm/forest/`), keeping `forest.ts` as the reference implementation, per `demos/README.md`.
- Tests (`demos/test/`): split search vs brute force on random small tables; the unit-1 worked
  table ($G = 0.2$ on $V$, $0.333$ on $I$); a depth-unlimited tree reaches zero training error on
  distinct points; OOB fraction ≈ 0.368 over many bootstraps; variance formula vs Monte Carlo;
  importance of an exact duplicate ≈ half; feature stack: Gaussian of a constant is constant,
  Hessian eigenvalues of a synthetic quadratic match the closed form.
- Images: `cell256.png` (already in the repo); synthetic images with ground truth for evaluation
  and counting. Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to
  reveal.js, labels ≥ 18 px, fits 1150×560 on a slide (`pixelclass` uses a 2-column layout with
  the controls in a narrow side panel; the feature browser opens as an overlay).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, Gini table by hand, `treeplay` |
| 10:45–12:15 | Unit 2 lecture + exercise (`bootstrap`, `forestplay`) |
| 13:15–14:45 | Unit 3 lecture + exercise (`featimp`) |
| 15:00–16:30 | Unit 4 lecture + exercise (`pixelclass`); quizzes as wrap-up |

## Open questions

- **Feature stack and pixel classification**: the SVM plan's `pixel-classify/` and the
  integral-image plan's `rect-features/` both introduce a stack too. Proposal: introduced here
  (unit 4), recalled there in one slide each; one demo shell `pixelclass` (absorbing `pixelsvm`)
  and one builder `featureStack` in `lib/imageops.ts`. Needs agreement from both plans; if the SVM
  day keeps the introduction, unit 4 here becomes a recall + forest-specific lesson (no scaling,
  OOB optimism, uncertainty from votes) and gains time for unit 5.
- **Boosting**: proposal: the integral-image day owns AdaBoost/stumps (`viola-jones/`), this
  day owns trees/stumps/bagging/forests and gives one contrast slide plus optional gradient
  boosting. Answers the integral plan's open question "which day owns boosting / decision trees /
  random forests".
- Structure tensor: no plan owns it; concept + math slide here, or move it to whichever day
  covers corners/orientation (none found).
- Ground truth for `cell256.png`: a manual annotation (nuclei mask, maybe cytoplasm) would allow
  honest held-out evaluation on the real image; the thresholding plan asks for the same. Until
  then held-out Dice is shown only on the synthetic image.
- Unit 3 could be shortened and merged into unit 2 if the day is too full, freeing time for
  optional unit 5 (counting) — which is more image-specific?
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
