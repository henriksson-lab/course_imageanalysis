# Plan: a day on support vector machines

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 34 of 51, Part VII "Learning from examples". Hard prerequisites: Random forests (day 33). Soft (see also): Neural networks (day 36), Gaussian mixtures (day 35). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VII (SVM, Gaussian mixtures); entry after day 33 (random forests).

## Goal of the day

After the day a student can:

1. say how image analysis turns into **classification**: a pixel described by a stack of filter responses, an object by its measured features, a window by a patch descriptor; and that a **threshold on one feature is already a linear classifier**;
2. read a **linear classifier** $f(\mathbf x) = \mathbf w\cdot\mathbf x + b$ geometrically (normal vector, offset, signed distance $f(\mathbf x)/\|\mathbf w\|$ to the hyperplane), and draw it for a small 2D example;
3. explain the **maximum-margin** idea, find the max-margin line and the **support vectors** of 3–4 points **by hand**, and say what the slack penalty $C$ trades off (the **hinge loss** compared with the 0–1 and logistic losses);
4. explain the **kernel trick**: compute an explicit quadratic feature map by hand and check that its dot product equals the **kernel function**; describe the **RBF kernel function** with $\gamma$ as an inverse squared length scale, and predict what large $\gamma$ or large $C$ do (overfitting);
5. apply an SVM to **pixel classification** with what is specific to it: **feature standardisation**, cross-validation of $(C, \gamma)$ **by stroke** (cross-validation itself recalled from day 33), class weights, **multi-class** by one-vs-rest / one-vs-one.

*(math)* for students who take the math layer: write the hard- and soft-margin primal problems, derive the **dual** with Lagrange multipliers, and see that the data enter only through dot products.

Why it fits this audience: the whole day is **geometry of hyperplanes** (dot products,
projections, distances from linear algebra) plus one constrained minimisation (calculus).
The kernel trick is "a dot product in a space we never write down", which is a linear
algebra statement. The classification vocabulary (*feature space*, *training / validation /
test set*, *overfitting*, *cross-validation*, the pixel feature stack) is owned by the random
forests day (day 33) and recalled here in one slide; the time goes to margins and kernel
functions. Following review-gaps, the dual stays in the math layer: the concept layer is max
margin + kernel functions as pictures.

## Prerequisite gap

- **Dot product, projection, distance to a hyperplane**: assumed (linear algebra). One recall slide with the 2D picture: $\mathbf w$ is the normal, $\mathbf w\cdot\mathbf x$ is (a multiple of) the projection on it.
- **Lagrange multipliers**: *assumption to confirm*: audience.md lists partial derivatives and the gradient but not constrained optimisation. Most students will have seen equality-constrained Lagrange multipliers in multivariable calculus; **inequality constraints (KKT) and convex duality are not assumed**. Unit 2/3 math layer has one recall slide ("at the optimum the gradient of the objective is a combination of the constraint gradients") and states KKT complementary slackness in words ("a multiplier is non-zero only where the constraint is tight"). The concept layer never needs it.
- **Filters as features and the feature stack**: Gaussian smoothing, derivatives (Sobel), Laplacian are known from `kernels/` (day 1); **LoG**, **difference of Gaussians** and **Hessian eigenvalues** from the edges day (day 2); the multiscale **feature stack** and the paint – train – correct loop from day 33 (`pixelclass/`). Unit 4 recalls the stack in one slide.
- **Classification basics** (train / validation / test, overfitting, cross-validation, nearest neighbour): owned by day 33; one recall slide in unit 1. Review-organisation proposed `classify/` here as the day-neutral owner; the course map gives ownership to the random forests day.
- **Vocabulary clash 1: "kernel"** (course convention). `kernels/` is the convolution kernel, and "kernel" alone always means that. Here it is the **kernel function** $k(\mathbf x, \mathbf z)$ (a similarity). Say so explicitly on the first kernel-function slide; folder `kernel-trick/`, code `lib/svmkernels.ts`.
- **Vocabulary clash 2: "hinge"**. `neuralnet/` (day 36) calls a ReLU unit a *hinge*. The hinge loss $\max(0, 1 - yf)$ is literally a ReLU of $1 - yf$: use this as a link ("the same bend"), not as a source of confusion.

## Overlaps with other days (who owns what)

- **Random forests** (day 33): **owns** feature space, train / validation / test, overfitting,
  cross-validation, the feature stack, the paint – train – correct loop and the `pixelclass`
  shell. This day recalls them and keeps what is SVM-specific: **standardisation** (fatal for
  RBF, irrelevant for a forest: a contrast preset), $(C, \gamma)$ cross-validation **by stroke**,
  one-vs-rest / one-vs-one. The SVM plugs into `pixelclass` via `data-classifier="svm"` (no
  separate `pixelsvm` demo).
- **Neural networks** (day 36): **that day owns** the neuron, logistic/softmax outputs,
  **cross-entropy loss**, gradient-descent training, backprop. **This day owns** the max-margin
  idea, the **hinge loss**, the dual and kernel functions. Shared fact, stated once: *one neuron
  with a sign output is a linear classifier $\mathbf w\cdot\mathbf x + b$*; an SVM and a logistic
  neuron differ only in the loss (and the margin term). Unit 2 shows the logistic loss as one
  curve in the loss demo with a pointer. Unit 4 has one comparison slide: kernel SVM = fixed
  features (the kernel function) + convex training with a unique optimum; network = learnt
  features + non-convex training.
- **Gaussian mixtures** (the other More-learning day): **that day owns** unsupervised
  clustering, EM, and **generative** classification. **This day** is **supervised and
  discriminative**. One slide in unit 1 contrasts the two with the same 2D point cloud, and notes
  that day 12's minimum-error threshold is the 1D generative case. Picture-level link in unit 3:
  an RBF decision function is a weighted sum of Gaussian bumps on the support vectors, but the
  weights are signed and fitted to separate, not to describe density.
- **Thresholding** (day 12): **owns pixel-level evaluation** (confusion matrix, precision/recall,
  Dice, ROC and PR curves, imbalance). Reused: the decision value $f(\mathbf x)$ plays the role of
  the intensity, sweeping the offset $b$ traces the ROC curve. One recall slide in unit 4; metric
  code from `lib/metrics.ts`. Unit 1 opens with "a threshold is a 1D linear classifier".
  Object-level evaluation of detections (unit 5): recall of the evaluation day (day 37).
- **Measurements** (day 16): owns the per-object **feature catalogue** (`lib/features.ts`) and
  z-scores in `intensity-features/`; same feature scaling as unit 4 here, one-line
  cross-reference. Object classification from a measured table is an example in unit 1.
- **Moments**: nearest class mean / nearest neighbour on moment features recalls day 33.
  Unit 1 here notes that nearest class mean is a **linear** classifier
  ($\mathbf w = \boldsymbol\mu_+ - \boldsymbol\mu_-$); the max-margin line is another choice of
  $\mathbf w$. Its `momentclass` demo can take `data-classifier="svm"` via `lib/classifier.ts`.
- **Compression / wavelets**: feature sources only (subband energies, KLT
  coefficients), one line each.
- **Graph cuts** (day 27, `graphcut-seg/`, `ncut/`): the **RBF kernel function is the Gaussian
  affinity** of n-links ($w_{pq} = e^{-(I_p - I_q)^2/2\sigma^2}$) and of `ncut`; said once:
  $\gamma = 1/(2\sigma^2)$. The kernel (Gram) matrix of unit 3 is the affinity matrix $W$ of
  `ncut`. A pixel classifier's decision value can serve as the **data term** of a graph cut, one
  slide.
- **Notation**: $C$ is the SVM slack penalty; said once that $C \sim 1/\lambda$ (the course's
  prior weight $\lambda$ multiplies the regulariser $\|\mathbf w\|^2$ instead). Labels
  $y \in \{-1, +1\}$ are used only on this day.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units (unit 4 shortened, since the feature stack and cross-validation are recalls),
plus one optional unit that can fill the freed time.

### 1. `classify/`: Linear classifiers (classification basics recalled from day 33)

- **Recall (one slide, day 33)**: a pixel, object or window is a point in feature space with a
  label; training / validation / test; overfitting; cross-validation.

- Three ways image analysis becomes classification, one picture each (a recall of day 33 plus windows): (a) **pixels**: each pixel gets a feature vector from a stack of filtered images (intensity, smoothed at several scales, edges, blobs); (b) **objects**: each segmented object is a row of the measurement table (area, mean intensity, form factor, texture); (c) **windows**: a patch around each position is described by a vector (e.g. histograms of gradient directions) and classified "cell / not cell". All three: a point in a feature space, with a label.
- Recall (day 12): a threshold on intensity is a classifier in 1D: $f(x) = x - t$, decide by the sign. In 2D, a line; in $d$ dimensions, a hyperplane: $f(\mathbf x) = \mathbf w\cdot\mathbf x + b$.
- Geometry: $\mathbf w$ is the normal; $b$ shifts the line; $f(\mathbf x)/\|\mathbf w\|$ is the signed distance; scaling $(\mathbf w, b)$ by $c > 0$ changes nothing about the decision (important for unit 2). Small numeric example: $\mathbf w = (1, 2)$, $b = -4$: classify three points, compute their distances.
- Error rate (0–1 loss). Many lines separate the training points: which one? Nearest class mean is one answer ($\mathbf w = \boldsymbol\mu_+ - \boldsymbol\mu_-$, through the midpoint); 1-NN gives a non-linear boundary; the max-margin line (unit 2) is another. Generative vs discriminative (GMM day) in one slide.
- When no line works: XOR, a ring around a blob (the nucleus/cytoplasm case in intensity alone). Sets up unit 3.
- **The math: hyperplanes**: distance formula $|\mathbf w\cdot\mathbf x + b|/\|\mathbf w\|$ from the projection; nearest class mean as a linear rule (expand $\|\mathbf x - \boldsymbol\mu_+\|^2 < \|\mathbf x - \boldsymbol\mu_-\|^2$).
- **Demo `linclass`** (new, small): a 2D scatter of two classes (presets of seeded point sets); the line drawn from a draggable normal arrow $\mathbf w$ and a draggable point on the line (sets $b$); shaded half-planes; each point shows its signed distance on hover-free labels (toggle); readout: number of errors, smallest distance to the line. Buttons: *nearest class mean*, *1-NN boundary* (shaded regions), *perceptron step* (one update, optional). Presets: "many lines are perfect", "nearest mean is not the best line", "one outlier", "XOR: no line works", "ring around a blob". Feature axes can be labelled (e.g. "mean intensity", "area") via `data-axes`.
- By hand: (a) for $\mathbf w = (1, 2)$, $b = -4$, classify 4 given points and compute their signed distances; (b) nearest class mean line for two classes of 3 points each; (c) a 1D threshold $t = 100$ written as $(w, b)$; (d) *(math)* show that nearest class mean is linear and find $\mathbf w, b$.

### 2. `maxmargin/`: Maximum margin, support vectors and the soft margin

- Picture first: of all separating lines, take the one with the widest empty **street** around it. Only the points on the edge of the street matter: the **support vectors**. Move any other point (without crossing the street): nothing changes. Move a support vector: the line moves.
- Why wide: robustness to noise in the features, and (stated, not proven) better generalisation.
- Fix the scale of $(\mathbf w, b)$ so that $f = \pm 1$ on the street edges; the street has width $2/\|\mathbf w\|$. Widest street = smallest $\|\mathbf w\|$ with every training point on its side: $y_i f(\mathbf x_i) \ge 1$.
- Worked example by hand: $+1$ at $(2, 0), (3, 1)$; $-1$ at $(0, 0), (0, 1)$. Max-margin line $x_1 = 1$, $\mathbf w = (1, 0)$, $b = -1$, width 2, support vectors $(2, 0), (0, 0), (0, 1)$; check $y_i f(\mathbf x_i)$ = 1, 2, 1, 1.
- **Soft margin**: real data overlap. Allow points inside the street or on the wrong side, at a price $C$ per unit of violation (slack $\xi_i$). Small $C$: wide street, many violations tolerated, smooth; large $C$: narrow street, few violations, sensitive to outliers. Support vectors now = all points on or inside the street.
- **Losses**: the soft-margin SVM minimises $\tfrac12\|\mathbf w\|^2 + C\sum_i \max(0,\, 1 - y_i f(\mathbf x_i))$. The **hinge loss** is zero for points safely outside the street and grows linearly inside. Compare: 0–1 loss (what we care about, but flat and jumpy: cannot be minimised by walking downhill), logistic / cross-entropy loss (day 36's loss: never exactly zero, so every point has a say), squared loss (penalises points that are "too correct"). The hinge is a ReLU of $1 - yf$ (link to `neuralnet/`, day 36). The $C$ penalty plays the role of $1/\lambda$ (one line).
- Small numeric example: five points with given $y_i f(\mathbf x_i) = 2, 1, 0.5, 0, -1$: hinge, 0–1 and logistic losses in a table.
- **The math: hard-margin primal**: $\min \tfrac12\|\mathbf w\|^2$ s.t. $y_i(\mathbf w\cdot\mathbf x_i + b) \ge 1$; a convex quadratic problem: one optimum, no local minima (contrast with networks).
- **The math: soft margin**: slack $\xi_i \ge 0$, $y_i f(\mathbf x_i) \ge 1 - \xi_i$, objective $+C\sum\xi_i$; eliminating $\xi_i$ gives the unconstrained hinge form (regularised loss). Subgradient descent on it in one line (the demo's "train linear by descent" mode).
- **Demo `svm2d`** (new, main demo of the day; used again in units 3 and 4 with other `data-*` settings): canvas, click to add points (left/right button or a class toggle; touch-friendly class switch), drag to move, delete mode. Shows the decision line, the two margin lines, the street shaded, support vectors ringed, slack shown as a short segment from violating points to their margin, and the decision value $f$ as a faint background colour. Slider $C$ (log scale, $10^{-2}$–$10^3$). Readouts: width $2/\|\mathbf w\|$, number of support vectors, training errors. Unit 2 setting: `data-kernel="linear"` only. Presets: "the by-hand example", "move a non-support vector: nothing happens", "one outlier: hard vs soft", "overlapping classes: sweep C", "C large = narrow street". Training by SMO on every edit (≤ 300 points: well under 30 ms).
- **Demo `hingeloss`** (new, small): plot of loss against the margin value $y f$ for 0–1, hinge, logistic, squared (toggles); below, a 1D data set on a line with a draggable threshold and slope; each point's loss shown as a bar; total loss for each loss function. Presets: "an outlier far on the wrong side: hinge vs squared", "points far on the right side: logistic still pulls", "0–1 has flat steps".
- By hand: (a) find the max-margin line, margin width and support vectors for 3–4 given points; (b) is a given line the max-margin line? (check: equal distance to the closest point of each class, and are those closest points "balanced"); (c) hinge losses for a table of $y_i f(\mathbf x_i)$; (d) predict what happens to the line when $C$ goes from 100 to 0.01 with one outlier; (e) *(math)* show that the street width is $2/\|\mathbf w\|$.

### 3. `kernel-trick/`: The dual, kernel functions and non-linear boundaries

- Picture first (**demo `svmlift`**): points inside a ring cannot be split by a line in 2D. Add a third coordinate $x_1^2 + x_2^2$: the points lift onto a paraboloid, and a **plane** separates them. The plane cut back down to 2D is a circle.
- General idea: map $\mathbf x \mapsto \varphi(\mathbf x)$ to more features, then fit a linear classifier there. Problem: the quadratic map of a 20-feature pixel already has 230 coordinates; degree 5 has tens of thousands.
- Key fact (stated in the concept layer, derived in the math layer): the SVM solution is $f(\mathbf x) = \sum_i \alpha_i y_i\, \mathbf x_i\cdot\mathbf x + b$, with $\alpha_i > 0$ only for **support vectors**. Training and prediction need **only dot products** between data points. Replace every dot product by a **kernel function** $k(\mathbf x, \mathbf z) = \varphi(\mathbf x)\cdot\varphi(\mathbf z)$ computed directly: the **kernel trick**. (On this day $\varphi$ is the feature map; on day 36 it was the activation: say so once.)
- By hand: $\varphi(\mathbf x) = (x_1^2, \sqrt2\,x_1x_2, x_2^2)$; for $\mathbf x = (1, 2)$, $\mathbf z = (3, 1)$: $\varphi(\mathbf x)\cdot\varphi(\mathbf z) = 9 + 12 + 4 = 25 = (\mathbf x\cdot\mathbf z)^2$. Polynomial kernel function $(\mathbf x\cdot\mathbf z + c)^p$.
- **RBF (Gaussian) kernel function** $k(\mathbf x, \mathbf z) = e^{-\gamma\|\mathbf x - \mathbf z\|^2}$: a similarity, 1 for identical points, $\to 0$ far apart. $1/\sqrt{\gamma}$ is a **length scale**: "how far does one training point's influence reach". $f(\mathbf x)$ = signed sum of Gaussian bumps on the support vectors. Small $\gamma$: almost linear; large $\gamma$: an island around every training point (memorises: overfitting). Link: this is the graph-cut n-link weight and the `ncut` affinity (day 27) with $\gamma = 1/(2\sigma^2)$, said once here; the **kernel matrix** $K_{ij} = k(\mathbf x_i, \mathbf x_j)$ is the `ncut` matrix $W$.
- Numeric example: two support vectors $(0,0)$ with $\alpha y = +1$ and $(2,0)$ with $\alpha y = -1$, $b = 0$, $\gamma = 1$: evaluate $f$ at $(0.5, 0)$, $(1, 0)$, $(1, 1)$.
- What makes a valid kernel function (**Mercer**, intuition): it must behave like a dot product somewhere: every kernel matrix must be symmetric with no negative eigenvalues (no "negative squared lengths"). Sums and products of kernel functions are kernel functions. One example of an invalid "similarity".
- $(C, \gamma)$ together: a 2D map of outcomes (underfit / good / overfit) as a picture; leads to cross-validation in unit 4.
- **The math: Lagrange multipliers and the dual**: recall slide (equality constraints, $\nabla f = \mu\nabla g$; multipliers are not called $\lambda$ here, which is the prior weight course-wide); Lagrangian $\tfrac12\|\mathbf w\|^2 - \sum_i\alpha_i[y_i(\mathbf w\cdot\mathbf x_i + b) - 1]$; stationarity gives $\mathbf w = \sum_i\alpha_i y_i\mathbf x_i$ and $\sum_i\alpha_i y_i = 0$; substitute to get the dual $\max_{\boldsymbol\alpha}\sum_i\alpha_i - \tfrac12\sum_{ij}\alpha_i\alpha_j y_i y_j\,\mathbf x_i\cdot\mathbf x_j$ with $0 \le \alpha_i \le C$; complementary slackness ($\alpha_i > 0$ only on the street edge or inside it) explains support vectors; $b$ from any support vector with $0 < \alpha_i < C$.
- **The math: how it is solved**: the dual is a quadratic problem in $n$ variables; **SMO** updates two $\alpha$s at a time in closed form (the constraint $\sum\alpha_i y_i = 0$ forces pairs). One slide, no derivation of the clipping; the demos use it.
- **The math: Mercer**: positive semidefinite kernel matrix for every finite set ⇔ a feature map exists (stated); the RBF kernel's feature space is infinite-dimensional (Taylor expansion of $e^{2\gamma\mathbf x\cdot\mathbf z}$, one line).
- **Demo `svmlift`** (new, WebGL via `lib/gl3d.ts`): left, 2D points (presets: ring, two rings, XOR, two blobs; draggable); right, the same points lifted to 3D by a selectable map: $(x_1, x_2, x_1^2 + x_2^2)$, or $(x_1^2, \sqrt2 x_1 x_2, x_2^2)$; orbit with the mouse (drag only, no keys). The separating plane found by a linear SVM in 3D is drawn translucent; its intersection with the lifted surface is projected back as the 2D boundary (a conic). Toggle "show the plane's cut in 2D". Presets: "ring → plane", "XOR needs the $x_1x_2$ coordinate", "two blobs: lifting does not hurt".
- **Demo `svm2d`** with `data-kernel="linear,poly,rbf"`: kernel-function menu, degree $p$, $\gamma$ slider (log), $C$ slider; background shows $f(\mathbf x)$ with the 0 and ±1 contours; support vectors ringed; optional overlay of each support vector's bump (RBF) as faint circles of radius $1/\sqrt{\gamma}$; optional small panel with the kernel matrix as a heat map (points sorted by class). Presets: "ring: linear fails, RBF works", "XOR: degree 2", "γ too large: islands", "γ too small: a line again", "C and γ both large".
- By hand: (a) verify the quadratic feature map identity for two given vectors, and write the 6-coordinate map for $(\mathbf x\cdot\mathbf z + 1)^2$; (b) evaluate an RBF decision function with 2–3 support vectors at given points and sketch the boundary; (c) for $\gamma = 0.01, 1, 100$ on points spaced about 1 apart, predict the boundary shape; (d) *(math)* solve the dual for two points $\mathbf x_+, \mathbf x_-$ by hand ($\alpha = 2/\|\mathbf x_+ - \mathbf x_-\|^2$) and recover $\mathbf w$, $b$ and the street; (e) *(math)* show that a $2\times2$ "similarity" matrix with a negative eigenvalue cannot be a kernel matrix (find the impossible squared distance).

### 4. `pixel-classify/`: What is specific to SVMs on pixels

- **Recall (one slide, day 33)**: paint strokes, compute the **feature stack**, train, look, correct; spatially correlated pixels. The SVM runs in the same `pixelclass` shell as the forest.
- **Feature standardisation**: RBF distances mix units; a feature with range 0–4000 drowns one with range 0–1. z-score with **training** statistics (the same z-score as day 16). Contrast preset: the forest does not care, the SVM fails.
- **$(C, \gamma)$ by cross-validation** (CV itself recalled from day 33): a log grid shown as a heat map; pick the best cell, then test once. Folds **by stroke**, not by pixel (the demo shows the optimistic by-pixel error).
- **Class imbalance**: per-class $C_+ \ne C_-$, or move the threshold on $f$. **Evaluation** by confusion matrix and ROC: sweep the offset of $f$ as day 12 swept $t$ (recall, reuse its plots).
- **Multi-class**: one-vs-rest ($K$ classifiers, take the largest $f_k$; scores not calibrated) vs one-vs-one ($K(K-1)/2$ classifiers, vote); picture with three classes and the ambiguous regions. Probabilities: one line (fit a sigmoid to $f$).
- **Smoothing the result**: the decision value as a graph-cut data term (day 27), one line.
- **Comparison slide**: SVM vs forest (day 33: no scaling, little tuning) vs $k$-NN vs neural networks (day 36: learn the features; need more labels; non-convex). When an SVM is a good choice: few labels, tens to a few hundred hand-made features.
- **The math: scaling statistics from training folds only**; why the test set is touched once (recall of day 33's CV slide, one line).
- **Demo `pixelclass`** (shared shell, owned by day 33) with `data-classifier="svm"`, `data-size="128"`: kernel function linear/RBF, $C$, $\gamma$, *standardise* toggle; tab *cross-validate*: $(C, \gamma)$ heat map with by-stroke vs by-pixel folds (added to the shared shell by this day); multi-class with OvR/OvO switch. Presets: "unscaled features: SVM fails, forest does not", "γ too large: speckle", "by-pixel CV is optimistic", "imbalanced strokes: class weights", "three classes: OvR ambiguous regions".
- **Demo `svm2d`** with `data-cv="1"`: a held-out test set shown as hollow points, train and test error against $C$ or $\gamma$ (a U-shaped test curve), and a small CV heat map.
- By hand: (a) z-score two features for four training pixels, then classify a test pixel by nearest mean before and after scaling; (b) a 3-class one-vs-rest table of $f_k$ values for four pixels: decisions and which are ambiguous; (c) read a CV heat map and pick $(C, \gamma)$; explain why the top-right corner overfits; (d) *(math)* number of classifiers and training-set sizes for OvR vs OvO with $K = 5$.
- Freed time (about 45 min): run optional unit 5 (HOG detection) as a short core part, if wanted (review-organisation §2.9; see Open questions).

### 5. (optional / further reading) `detect-window/`: Sliding-window detection with HOG and a linear SVM

- Detection as classification of every window: slide a fixed-size window, compute a descriptor, score with a linear SVM ($f$ is a single dot product per window, so it is fast), keep local maxima above a threshold (**non-maximum suppression**), repeat on a pyramid for several sizes.
- **HOG** (Dalal & Triggs 2005): gradients (Sobel, `kernels/`; gradient direction from day 2), histograms of gradient direction in small cells, normalised over blocks (contrast invariance); the learnt $\mathbf w$ drawn as a HOG glyph shows what the detector looks for (the outline of a cell).
- **Hard negative mining**: train, run on background, add the false positives to the negatives, retrain. Evaluation: precision–recall over detections, matching by IoU (recall of the evaluation day, day 37; it owns object-level matching and AP). Non-maximum suppression and pyramids: recall of the edges day (day 2); deep detectors: day 39.
- **Demo `hogdetect`** (new): synthetic fluorescence-like images of round cells on a noisy, uneven background (from `lib/synth.ts` generators, seeded), optionally the micrograph. Panels: the image with a movable window and its HOG glyphs; the learnt weight glyph; detection score map; detections before/after NMS against ground truth. Controls: cell size, bins, window size, $C$, threshold, *mine hard negatives* button. Presets: "a clean set", "touching cells", "different sizes: one scale fails", "after hard negatives".
- By hand: (a) gradient-orientation histogram of a 4×4 cell with given gradients (9 bins); (b) NMS on a list of 6 scored boxes; (c) why a linear SVM on HOG needs only one dot product per window.

## Shared engineering

Library names follow review-organisation §3.1 (one file, one owner).

- `demos/src/lib/svm.ts` (new, owned by this day): SMO with second-order working-set selection (Fan, Chen & Lin 2005) for the soft-margin dual with per-class $C$; linear, polynomial and RBF kernel functions via a function argument; returns $\boldsymbol\alpha$, $b$, support-vector indices, primal $\mathbf w$ for the linear case, and the dual objective. Kernel-matrix cache (full matrix up to ~1000 points). One-vs-rest and one-vs-one wrappers, $k$-fold CV (folds by group id, for by-stroke CV), grid search. A linear subgradient (Pegasos-style) trainer for the "descent" mode and for HOG. Flat `Float64Array`s, no DOM, runs under node. Implements the `lib/classifier.ts` interface (owner day 33) so it plugs into `pixelclass` and `momentclass`.
- `demos/src/lib/svmkernels.ts` (new, owned by this day; decided name, never `kernels.ts`): kernel functions, explicit quadratic feature maps for `svmlift`, kernel matrix for the heat map. The RBF function is shared with the `ncut` Gaussian affinity (one implementation).
- `demos/src/lib/hog.ts` (new, only if unit 5 is built): cell histograms with bilinear vote, block normalisation (L2-Hys), glyph drawing; gradients from `lib/imageops.ts`.
- **Reused, not duplicated**: `lib/imageops.ts` (`loadGrey()`, Gaussian at several σ, Sobel; `featureStack(img, groups, sigmas)`, owner day 33), `lib/edges.ts` (LoG, Hessian; owner edges day), `lib/metrics.ts` (confusion matrix, ROC/PR; owner thresholding), `lib/plot.ts` (loss curves, ROC, CV heat map, scatter), `lib/playground.ts` + `lib/datasets2d.ts` (2D point playground for `linclass`, `svm2d`; owner day 33), `lib/gl3d.ts` (`svmlift`; owner graphics3d), `lib/features.ts` (day 16; object features for an optional preset), `lib/synth.ts` (synthetic cells), `lib/linalg.ts` (symmetric eigen for the Mercer check), `rng(seed)` from `lib/random.ts`.
- **Speed** (TypeScript first, measure): SMO on ≤ 300 points in 2D: a few ms. `svm2d` background: a 200×120 grid × ≤ 100 support vectors ≈ 2.4 M kernel-function evaluations, ~10 ms. `pixelclass` with the SVM: train on ≤ 1000 painted pixels (subsample strokes) × ~20 features, predict 16 384 pixels × ≤ 300 SVs ≈ 100 M flops: ~50–100 ms in TS, acceptable on "train" but not on every slider tick; the shell caches the feature stack. CV grid (7×7 cells × 5 folds = 245 trainings) is the slow path: run in chunks with a progress bar. If measured above budget, move `predictRBF` and the SMO inner loop to Rust/Wasm per `demos/README.md`, keeping the TS reference.
- **Tests** (`demos/test/svm-test.ts`, run by `make test`): the by-hand examples of units 2–3 reproduce exactly (line $x_1 = 1$, support vectors, two-point dual $\alpha$); KKT conditions hold to tolerance on random data; duality gap ≈ 0; quadratic kernel function = dot product of the explicit map; separable data with large $C$ gives zero training error; SMO vs a brute-force QP (projected gradient) on ≤ 8 points.
- Images: `demos/src/fourier2d/cell256.png` (credit as in Part 1). Synthetic images with ground truth from `lib/synth.ts`.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide; on slides, painting/clicking with large targets, a visible class toggle (no right-click-only actions).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 (recall of day 33, linear classifiers) + exercise (`linclass`) |
| 10:45–12:15 | Unit 2 lecture, max margin by hand, `svm2d` (linear), `hingeloss` |
| 13:15–14:45 | Unit 3 lecture, feature map by hand, `svmlift`, `svm2d` with kernel functions |
| 15:00–16:30 | Unit 4 lecture + exercise (`pixelclass` with the SVM), ~45 min; optional unit 5 (`hogdetect`) or quizzes as wrap-up |

## Open questions

- Lagrange multipliers: have the students met them (and inequality constraints)? If not, the dual stays entirely in the math layer, as planned; if yes, the recall slide can go.
- **Decided (course map):** this day comes after day 33 (and after day 36 in the course order); unit 1 is a recall of day 33's vocabulary plus linear-classifier geometry. Review-organisation's day-neutral `classify/` owner is overruled by the map.
- Unit 5 (HOG detection): with unit 4 shortened, it can take the freed 45 minutes as core (review-organisation §2.9). Review-gaps would shrink the day to two units (max margin + kernel functions); as a day it keeps four, with the dual in the math layer only. Course owner to decide.
- Ground truth for `pixelclass` on the real micrograph: annotate `cell256.png` (nucleus / cytoplasm / background masks, a one-off manual job; the thresholding plan asks for the same annotation) or evaluate only on synthetic images?
- **Decided (conventions):** `lib/svmkernels.ts`, folder `kernel-trick/`, and "kernel function" on slides; "kernel" alone is the convolution kernel.
- **Decided:** `momentclass` (moments day) gets an SVM option through `lib/classifier.ts`.
- Is SMO's algorithm worth a slide at all, or is "a standard quadratic solver" enough at this level?

## References

- Cortes & Vapnik (1995), *Support-vector networks*, Machine Learning.
- Boser, Guyon & Vapnik (1992), *A training algorithm for optimal margin classifiers*, COLT.
- Platt (1998), *Sequential minimal optimization: a fast algorithm for training support vector machines*, Microsoft Research TR-98-14; Platt (1999), *Probabilistic outputs for support vector machines* (sigmoid on $f$).
- Fan, Chen & Lin (2005), *Working set selection using second order information for training SVM*, JMLR.
- Schölkopf & Smola (2002), *Learning with Kernels*, MIT Press.
- Burges (1998), *A tutorial on support vector machines for pattern recognition*, Data Mining and Knowledge Discovery.
- Hsu, Chang & Lin (2003), *A practical guide to support vector classification* (scaling, RBF, CV grid).
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, ch. 12 (SVMs, hinge vs logistic loss) and ch. 7 (cross-validation); Bishop (2006), *Pattern Recognition and Machine Learning*, ch. 7.
- Dalal & Triggs (2005), *Histograms of oriented gradients for human detection*, CVPR.
- Shalev-Shwartz, Singer & Srebro (2007), *Pegasos: primal estimated sub-gradient solver for SVM*, ICML.
- Sommer, Straehle, Köthe & Hamprecht (2011), *ilastik: interactive learning and segmentation toolkit*, ISBI (teacher reference for pixel classification with feature stacks).
