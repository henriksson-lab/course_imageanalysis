# Plan: a day on Gaussian mixture models

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. treat pixels (or objects) as **points in a feature space**, run **k-means** (Lloyd's algorithm) by hand on a few 1D/2D points, state what it minimises, and explain why the result depends on the initialisation (local minima, **k-means++**);
2. read a **multivariate Gaussian** from its picture: the mean as the centre, the **covariance ellipse** with the eigenvectors of $\Sigma$ as axes and $\sqrt{\lambda_i}$ as half-lengths, the **Mahalanobis distance**; and describe a **mixture** as a recipe for generating data (pick a component, then draw a point);
3. run **EM** by hand on a 1D two-component example: E-step (responsibilities), M-step (weighted means, variances, weights); explain why the likelihood never decreases, why EM finds a **local** maximum, and why a component that **collapses on one point** makes the likelihood infinite;
4. relate k-means to EM (hard vs soft assignment, k-means = EM with equal, shrinking, spherical covariances) and choose between **spherical / diagonal / full** covariances and a number of components $K$ (**BIC / AIC**, in words and with numbers);
5. use mixtures on images: an **intensity-histogram mixture** with $K$ classes (the minimum-error threshold generalised), **colour / feature-space** segmentation, **per-pixel mixtures for background subtraction**, populations of spots or cells, and **anomaly detection** by low likelihood.

Why it fits this audience: everything is means, variances and the normal density they know,
plus the covariance matrix and its eigenvectors from linear algebra. EM is weighted averages;
its proof is one application of Jensen's inequality (math layer). The day is the
**unsupervised, generative** counterpart of the SVM day (`plan/svm-day.md`, planned in
parallel): no labels are given, and the model describes how the data could have been
produced, rather than drawing a boundary between labelled classes. A slide in unit 2 puts
the two side by side (same 2D point cloud: with labels → a boundary; without labels → a
mixture and its implied Bayes boundary).

## Prerequisite gap

- **Histogram, Bayes decision for two classes, isodata, minimum-error threshold**: taught on the thresholding day (`plan/thresholding-day.md`, unit 2 `otsu/`). Recalled in one picture each (unit 1 for isodata, unit 4 for the two-Gaussian threshold); not re-derived.
- **Covariance matrix and its eigenvectors as principal axes**: introduced from a 2D point cloud on the snakes day (`plan/snakes-day.md`, unit 3) and via second moments on the labelling day (`plan/labelling-day.md`, equivalent ellipse). Unit 2 has one recall slide (the same picture), enough to stand alone if neither has been taught.
- **Multivariate normal density**: not assumed (audience.md has only the 1D normal). Introduced in unit 2 from the 1D case: $(x-\mu)^2/\sigma^2$ becomes $(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$.
- **Bayes' rule, likelihood, log-likelihood**: Bayes' rule is introduced on the thresholding day; maximum likelihood is new here. One slide in unit 2: "the parameters that make the observed data most probable"; for one Gaussian the answer is the sample mean and covariance (math layer: the derivative).

## Split with other days

Neutral cross-references; their files are not edited here (notes for whoever revises them).

- **Thresholding** (`plan/thresholding-day.md`): isodata is 1D 2-means and Kittler–Illingworth is "two Gaussians, Bayes boundary", fitted by searching over the split $t$. This day **generalises**: $K$ components, fitted by EM (soft assignments instead of a hard split), in any dimension. Its `mixture` demo (two Gaussians with sliders) is the 1D picture; unit 4 here starts from a back-reference to it. Its worked histogram $h = (2,5,3,1,1,2,4,2)$ can be reused as a first EM-on-a-histogram example. Multi-level Otsu stays there; unit 4 compares it with a $K$-component mixture in one preset.
- **Colour** (`plan/colour-day.md`, unit 3): owns colour spaces and colour segmentation by thresholds, ΔE balls and k-means in Lab (`colourseg`). This day owns **k-means as an algorithm** (objective, convergence, initialisation) and the step from k-means to **mixtures with full covariances**; unit 4 uses colour only as one feature space, with Lab conversion from `lib/colour.ts`. The point to make here and not there: a shadowed object is an elongated cloud along the lightness axis, which k-means splits and a full-covariance component keeps whole.
- **Graph cuts** (`plan/graphcut-day.md`): its data term $-\log P(I_p \mid \ell)$ can be the $-\log$ of a fitted mixture; GrabCut (iterate: fit colour GMMs, cut again) stays a pointer there and is described in one slide in optional unit 6 here, with a reference back.
- **Ising / MRF** (`plan/ising-day.md`, optional unit 5 `potts/`): mentions "re-estimate the means: EM in words". Optional unit 6 here makes this explicit: a **hidden MRF** = mixture likelihood + Potts prior, alternating EM-style parameter updates with ICM / mean field labelling. The prior, samplers and MAP/MPM stay on the Ising day.
- **Level sets** (`plan/levelsets-day.md`, `chanvese/`): Chan–Vese with $\mu = 0$ is two-means; one line in unit 1 ("you have seen this objective as a region energy").
- **Measurements** (`plan/measurements-day.md`, unit 5 `features-use/`): mentions clustering of objects in a feature table. Unit 4 here gives the worked case (DNA content per nucleus: G1/G2 peaks as a mixture). The feature computation and normalisation stay there.
- **Snakes / labelling** (PCA, covariance): used, not re-taught (see prerequisite gap).
- **VAE** (`plan/vae-day.md`): the math layer of unit 3 derives EM as coordinate ascent on a lower bound of the log-likelihood (the ELBO); the VAE day's ELBO is the same bound with a learned encoder in place of the exact E-step. One sentence each way.
- **SVM** (`plan/svm-day.md`, in parallel): supervised, discriminative. This day: unsupervised, generative. Shared 2D point-cloud datasets would let both days use the same pictures (see shared engineering).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units.

### 1. `kmeans/`: Feature spaces and k-means

- Concept layer: every pixel is a point (intensity in 1D; colour in 3D; intensity + local mean, or position + colour); every segmented object is a point (area, mean intensity, …). A segmentation without labels = finding **groups** of points. Recall (one picture): isodata on the thresholding day is exactly this with two groups in 1D.
- **Lloyd's algorithm**: assign each point to the nearest centre, move each centre to the mean of its points, repeat. Picture: Voronoi cells of the centres. Stops when nothing changes.
- What it minimises: the within-cluster sum of squares $J = \sum_i \lVert \mathbf{x}_i - \mathbf{m}_{c(i)}\rVert^2$. Each of the two steps can only lower $J$, there are finitely many assignments, so it stops; but at a **local** minimum (worked example below). Chan–Vese two-means is the same objective (one line).
- **Initialisation**: random points, random partition, **k-means++** (Arthur & Vassilvitskii 2007): next centre drawn with probability $\propto D(\mathbf{x})^2$, the squared distance to the nearest centre chosen so far. Restarts: keep the run with the lowest $J$.
- What k-means assumes, shown with pictures before unit 2 fixes it: round clusters of similar size (boundaries are perpendicular bisectors), every point belongs to exactly one cluster, units matter (rescale an axis and the clusters change).
- Choosing $K$: $J$ always decreases with $K$; the "elbow" in words. A principled answer follows in unit 3.
- The math: the update step as $\partial J / \partial \mathbf{m}_k = 0$; the proof that $J$ is non-increasing; k-means is NP-hard in general (one line, so initialisation matters).
- **Demo `kmeans2d`** (new): SVG scatter, 20–300 points; click to add, drag to move, shift-click to delete. $K$ 1–8; initialisation random / k-means++ / *place centres by hand* (drag); buttons *assign*, *update*, *step*, *run*, *reset*; Voronoi cells shaded, centres with trails; $J$ vs iteration plot; *100 restarts* draws a histogram of final $J$ values. `data-mode="1d"` shows points on a line (for the by-hand example). Presets: "three blobs", "the by-hand example (1D)", "stuck: two centres share one blob" (the 1D local minimum below), "k-means++ avoids it", "elongated clusters: cut across", "one big and one small cluster: boundary in the wrong place", "rescale the y axis".
- By hand (concept): (a) 1D points $\{1, 2, 4, 7, 9, 10\}$, $K = 2$, start $m = (1, 2)$: iterate to convergence ($m$: $(1, 6.4) \to (1.5, 7.5) \to (2.33, 8.67)$, stop; $J = 9.33$). (b) Points $\{0, 1, 5, 6, 10, 11\}$, $K = 3$, start $(0, 1, 8)$: show it is already a fixed point with $J = 26$, while $\{0,1\}, \{5,6\}, \{10,11\}$ gives $J = 1.5$. (c) k-means++ on the same points with first centre 0: the probability that the second centre is 10 or 11 ($221/283 \approx 0.78$).
- By hand *(math)*: show that the mean minimises $\sum_i \lVert\mathbf{x}_i - \mathbf{m}\rVert^2$; show that the boundary between two centres is the perpendicular bisector.

### 2. `gaussian-mixture/`: The multivariate Gaussian and the mixture model

- One Gaussian in 2D, from pictures: a hill whose contour lines are **ellipses**. The covariance matrix $\Sigma$ sets the ellipse: axes along the eigenvectors, half-lengths $\sqrt{\lambda_i}$ (1σ ellipse); recall from the PCA/second-moment slides. Off-diagonal entry = tilt (correlation).
- **Mahalanobis distance** $d_M^2 = (\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$: "how many standard deviations away, in the direction you are going". Two points at the same Euclidean distance, one inside the 1σ ellipse and one at 3σ.
- **Covariance types** as pictures: spherical ($\sigma^2 I$: circles), diagonal (axis-aligned ellipses), full (any ellipse), tied (all components share one $\Sigma$). Number of parameters per component in $d$ dimensions: $1$, $d$, $d(d+1)/2$.
- **The mixture as a generative story**: pick component $k$ with probability $\pi_k$, then draw $\mathbf{x} \sim \mathcal{N}(\boldsymbol\mu_k, \Sigma_k)$. Density $p(\mathbf{x}) = \sum_k \pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \Sigma_k)$. The data come without the colours: the component of each point is **hidden** (latent). Fitting = recovering the recipe.
- **Responsibility** (posterior of the component, Bayes' rule as on the thresholding day): $r_{k}(\mathbf{x}) = \pi_k\mathcal{N}_k(\mathbf{x}) / p(\mathbf{x})$: a soft label, e.g. "70 % component 1". Hard label = the largest; the boundaries are the Bayes boundaries (straight lines for tied covariances, conics otherwise).
- **Likelihood** of a data set: $\prod_i p(\mathbf{x}_i)$, in practice $\log L = \sum_i \log \sum_k \pi_k \mathcal{N}_k(\mathbf{x}_i)$. For **one** Gaussian, maximum likelihood gives the sample mean and covariance (closed form). For a mixture the log of a sum has no closed form: next unit.
- Generative vs discriminative, one slide (the SVM link above). Unsupervised: no labels anywhere.
- The math: the density $\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\Sigma) = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp(-\tfrac12 d_M^2)$; why $\Sigma = Q\Lambda Q^\top$ gives the ellipse (change of basis to the eigenvectors, where the density factorises into independent 1D Gaussians); sampling as $\boldsymbol\mu + Q\Lambda^{1/2}\mathbf{z}$ with $\mathbf{z}$ standard normal; maximum-likelihood mean and covariance by setting the gradient to zero; tied covariances give linear boundaries (the quadratic terms cancel, as in the thresholding day's equal-variance case).
- **Demo `gauss2d`** (new, small): one Gaussian with draggable handles (centre; ends of the two axes, which rotate and stretch the ellipse). Live: $\Sigma$ entries, eigenvalues/eigenvectors as arrows, 1/2/3σ ellipses, 300 seeded samples (*resample*). A draggable probe point with its Euclidean and Mahalanobis distances and density. Constraint menu spherical / diagonal / full (the handles snap accordingly). Mixture mode (`data-mode="mixture"`): 2–3 components with weight sliders, the density as a heat map, samples coloured by their component, a *hide colours* toggle ("this is what the algorithm sees"), and the responsibility of the probe point as a bar. Presets: "circle", "tilted cigar", "same Euclidean distance, different Mahalanobis", "two overlapping components: the hidden labels", "tied covariances: straight boundary", "different covariances: curved boundary".
- By hand (concept): (a) $\Sigma = \begin{pmatrix}5 & 2\\ 2 & 2\end{pmatrix}$: eigenvalues $6, 1$, eigenvectors $(2,1)/\sqrt5$, $(-1,2)/\sqrt5$; sketch the 1σ ellipse (half-axes $\sqrt6$ and $1$). (b) With $\boldsymbol\mu = 0$ and this $\Sigma$: Mahalanobis distance of $(2,1)$ and of $(-1,2)$ (both Euclidean $\sqrt5$; $d_M = \sqrt{5/6}$ vs $\sqrt5$). (c) Count parameters of a $K = 3$ mixture in 3D colour space with spherical, diagonal and full covariances ($3\cdot(1+3+1) - 1 = 14$, $20$, $29$). (d) A 1D two-component mixture with given parameters: responsibility of one point.
- By hand *(math)*: derive the maximum-likelihood mean of one Gaussian; show that two Gaussians with the same $\Sigma$ have a linear Bayes boundary.

### 3. `em/`: Fitting a mixture: the EM algorithm

- Chicken and egg: if we knew the colours (components) we would fit each Gaussian by its mean and covariance; if we knew the Gaussians we would compute the colours by Bayes. **EM alternates**, keeping colours **soft**:
  - **E-step**: responsibilities $r_{ik}$ for every point and component;
  - **M-step**: $N_k = \sum_i r_{ik}$; $\pi_k = N_k/n$; $\boldsymbol\mu_k$ = weighted mean; $\Sigma_k$ = weighted covariance (weights $r_{ik}$).
- Compare with k-means side by side: same two-step loop; k-means uses hard 0/1 weights and ignores covariances and weights. With equal spherical covariances $\sigma^2 I$ and $\sigma \to 0$, responsibilities become 0/1 and EM *is* k-means. Common practice: initialise EM from k-means.
- **The likelihood never decreases** (shown in the demo's plot first, proved in the math layer); EM stops at a **local** maximum: restarts, k-means++ starts, compare log-likelihoods.
- **Singularities**: a component sitting on one point with $\sigma \to 0$ has density $\to\infty$ there, so the likelihood is unbounded: the "best" fit is nonsense. Remedies: a variance floor (add $\varepsilon I$ to every $\Sigma_k$), restart components that collapse, a prior on $\Sigma$ (pointer). Also: empty components ($N_k \to 0$).
- **Choosing $K$ and the covariance type**: the likelihood always grows with more parameters (overfitting picture: one component per point). Penalise: $\text{BIC} = -2\log L + p\ln n$, $\text{AIC} = -2\log L + 2p$ (smaller is better; $p$ from unit 2's count). Intuition: an extra component must "pay for itself". BIC penalises harder and picks fewer components for large $n$.
- The math: (1) setting $\partial \log L/\partial\boldsymbol\mu_k = 0$ gives the M-step mean with the responsibilities appearing as weights (and similarly for $\Sigma_k$, and $\pi_k$ with a Lagrange multiplier). (2) Monotonicity: for any distribution $q_i(k)$, Jensen's inequality gives $\log p(\mathbf{x}_i) \ge \sum_k q_i(k)\log\frac{\pi_k\mathcal{N}_k(\mathbf{x}_i)}{q_i(k)}$ (the ELBO; equality for $q_i = r_i$). The E-step makes the bound touch, the M-step pushes the bound up: the likelihood cannot go down. Same bound as the VAE day. (3) **EM in general**: any model with missing data (hidden labels, missing pixels, censored values): E = expected complete-data log-likelihood, M = maximise it (Dempster, Laird & Rubin 1977).
- **Demo `em1d`** (new, small, the bridge from the by-hand table): 6–12 points on a line (draggable), two components (three with `data-k="3"`) drawn as weighted Gaussians and their sum; a table of $r_{ik}$ per point, $N_k$, $\mu_k$, $\sigma_k^2$, $\pi_k$, $\log L$, updated per half-step; points coloured by responsibility; *E-step*, *M-step*, *reset*. Presets: "the by-hand example", "an ambiguous point in the middle", "collapse: one component on one point" (variance floor toggle), "hard EM = k-means".
- **Demo `em2d`** (new, main demo of the day): 2D points (datasets from a menu, seeded, plus click to add), $K$ 1–6, covariance type spherical / diagonal / full / tied, initialisation random / k-means / k-means++ / *drag the ellipses by hand*; buttons *E-step*, *M-step*, *step*, *run*, *reset*; *soft vs hard (k-means)* toggle. Points coloured by blending component colours with their responsibilities; ellipses (1σ, 2σ); optional density heat map and Bayes boundaries. Right panel: $\log L$ vs iteration (monotone; a red mark if it ever drops, which it should not), a $\varepsilon$ variance-floor slider, *20 restarts* (histogram of final $\log L$), and a BIC/AIC bar chart over $K = 1..6$ for the current covariance type. Presets: "well separated", "elongated clusters: k-means fails, full GMM works", "one big, one small: weights matter", "bad start: local maximum", "collapse (floor off)", "K too large: BIC says no", "diagonal is enough / is not enough", "heavy overlap: soft labels near 0.5".
- By hand (concept): data $\{1, 2, 4, 7, 9, 10\}$ (the k-means example), start $\mu = (2, 8)$, $\sigma^2 = (4, 4)$, $\pi = (0.5, 0.5)$. With equal variances and weights, $r_1(x) = 1/(1 + e^{-a})$ with $a = [(x-8)^2 - (x-2)^2]/8$: compute $r_1(4) = 0.82$ and $r_1(7) = 0.05$ (the other four are given: $\approx 1, 0.99, 0.00, 0.00$); then $N_1 = 2.86$, $\mu_1 = 2.31$, $\mu_2 = 8.39$, $\pi_1 = 0.48$. Compare with the k-means answer (2.33, 8.67): the point 4 is shared. Log-likelihood goes $-14.94 \to -14.22 \to -14.03 \to -13.99$ (given, to check monotonicity). (b) BIC/AIC: given $\log L = -4210, -4020, -4012, -4010$ for $K = 1..4$ on $n = 1000$ intensities (1D, $p = 3K - 1$): BIC picks $K = 2$ (8074.5), AIC picks $K = 3$ (8040); explain why they differ. (c) Predict-and-explain: what happens to EM when a component is started exactly on an outlier point?
- By hand *(math)*: derive the M-step for $\boldsymbol\mu_k$ and $\pi_k$; prove the Jensen bound and that it is tight for $q = r$; show that EM with $\Sigma_k = \sigma^2 I$, equal weights and $\sigma \to 0$ gives k-means.

### 4. `gmm-images/`: Mixtures on images

- **Intensity-histogram mixtures**: fit $K$ Gaussians to the histogram (EM on the bins, weighted by counts: fast for any image size). $K = 2$ recovers the thresholding day's minimum-error threshold, now found by EM instead of a search over $t$ (back-reference); $K = 3$ gives membrane / cytoplasm / background on the micrograph; the posterior per class as a **probability map** (soft segmentation), the label image by maximum posterior. Compare with multi-level Otsu (preset). Limits: the histogram forgets where pixels are, so noise gives speckle (→ neighbours: graph cuts / Ising; unit 6).
- **Feature-space segmentation**: colour in Lab (the shadow example: an elongated cloud kept whole by a full covariance, split by k-means); intensity + local mean; colour + position $(x, y)$ with a weight (compact regions; a pointer to superpixels).
- **Populations**: integrated DNA intensity per nucleus (measurements day features) is a mixture of G1 and G2 peaks (2N/4N) plus S-phase in between; spot intensities as a mixture of 1, 2, 3 fluorophores (counting by intensity). Read off the component weights = population fractions.
- **Anomaly detection by low likelihood**: fit a mixture to "normal" pixels or objects; flag points with $p(\mathbf{x})$ below a threshold (or Mahalanobis distance above $3$ to every component). Picture: dust, a debris object, a rare cell type. Contrast with the VAE day's reconstruction error (one line).
- **Data terms**: $-\log p(I_p \mid \ell)$ from a fitted mixture per class is the data term of the graph-cut and Ising days (pointer; GrabCut in unit 6).
- The math: EM on a weighted histogram (replace sums over pixels by sums over bins times counts); why the $K = 2$, equal-variance mixture's boundary is the thresholding day's formula; log-sum-exp for numerically safe $\log p$ (one line).
- **Demo `histgmm`** (new): `demos/src/fourier2d/cell256.png` (downsampled to 128×128) and a synthetic image (three intensity classes, noise and blur sliders, known truth). Left: the image with label overlay or a per-class posterior map (menu); right: the histogram with fitted components, their sum, Bayes boundaries and the multi-Otsu thresholds (toggle). Controls: $K$ 1–6 slider (re-fit on change), *step EM* / *run*, initialisation (quantiles / k-means / random), equal vs free variances, BIC vs $K$ mini-plot, a *low-likelihood* view (pixels with $p(I) < \tau$, $\tau$ slider). Presets: "K = 2: the minimum-error threshold again", "K = 3 on the micrograph", "K too large: components split a peak", "equal variances: Otsu-like", "noise: speckle in the label image", "small bright class: weights matter". Runs in a few ms (EM on ≤ 256 bins).
- **Demo `colourgmm`** (new): a colour image (synthetic scene with coloured objects, shadow gradient and noise, with ground truth; a colour photo; an H&E crop: same inputs as the colour day's `colourseg`, so students recognise them). Middle: the pixels projected on $a^*b^*$ (or $L^*$ vs $a^*$, menu) as a 2D histogram, with the fitted components drawn as projected ellipses. Right: label image / posterior map / low-likelihood map. Controls: features (RGB / Lab / Lab + $xy$ with spatial-weight slider), $K$, covariance type, k-means vs GMM, *step* / *run*, Dice readout on the synthetic scene. Presets: "shadowed object: k-means splits it, full GMM keeps it", "RGB vs Lab", "add position: compact regions", "H&E: three stains", "an odd object is low-likelihood". 128×128 pixels × $K \le 6$ in 3–5 D: ~1–2 ms per EM iteration in TypeScript.
- By hand (concept): (a) a 1D three-component histogram mixture with given parameters: label three grey values and give their posteriors; (b) from given fitted weights of a DNA-content mixture, the fraction of cells in G2; (c) given two components in 2D, decide whether a new point is an anomaly by Mahalanobis distance (threshold 3). (d) Predict: k-means vs full GMM on the shadowed object (sketch the clusters).
- By hand *(math)*: write the EM updates for a histogram with counts $h_j$ at grey values $g_j$.

### 5. (optional) `bgsub/`: Background subtraction with per-pixel mixtures

- Each pixel's history over time is a 1D (or colour) data set. A static background = one narrow Gaussian; a waving tree, water or a flickering screen = **several** background modes; a passing object = a rare, short-lived component.
- **Stauffer–Grimson** (1999): per pixel $K = 3$–5 Gaussians updated **online**: a new value matching a component (within $2.5\sigma$) updates it with learning rate $\alpha$; no match replaces the weakest component. Components ranked by $\pi_k/\sigma_k$; the first $B$ whose weights sum to $T$ are background; a pixel matching none of those is foreground.
- Trade-offs from the demo: $\alpha$ large → adapts to illumination but absorbs a stopped object; $\alpha$ small → ghosts when the background changes. Baselines for comparison: running average, running median. Shadows as a pointer (colour day: shadow = lower lightness, same chromaticity).
- The math: the online update as a stochastic (exponentially weighted) version of the M-step; why $\pi/\sigma$ ranks "frequent and stable".
- **Demo `bgmix`** (new): a seeded synthetic video (160×120, ~300 frames, generated procedurally in TS, no files): textured static background, a swaying-foliage patch (bimodal pixels), a flickering lamp, sensor noise, slow illumination drift, objects that walk across, one that stops and stays. Panels: frame, foreground mask, background image (mean of the top component); click a pixel to see its intensity trace over time with its $K$ Gaussians (means ± 2.5σ as bands, weights as thickness). Controls: method (running average / median / mixture), $K$, $\alpha$, $T$, play / pause / step, reset. Presets: "one Gaussian fails on the tree", "K = 3 learns the tree", "α too large: the parked object disappears", "α too small: ghosts after the light changes".
- By hand: a pixel with given components and a new value: matched or not, the updated mean and weight with $\alpha = 0.1$, and whether it is foreground.

### 6. (optional / further reading) `hmrf-em/`: Mixtures with neighbours, and beyond

- **Hidden MRF**: the histogram mixture of unit 4 plus a Potts prior from the Ising day. Alternate: labels (ICM or mean field, Ising day) and parameters (M-step from the current soft or hard labels). $J = 0$ gives back the plain mixture. Speckle disappears, thin structures are at risk (as in the graph-cut day).
- **GrabCut** (Rother et al. 2004) in one slide: a box, colour GMMs for object and background, a graph cut, re-fit, repeat; back-reference to the graph-cut day.
- Pointers only: variational Bayesian mixtures and **Dirichlet-process** mixtures (let the data choose $K$); mixtures of other distributions (Poisson for photon counts, t-distributions for outliers); mixture models as the simplest **latent-variable** model, with the VAE as a continuous-latent cousin.
- **Demo**: reuse `histgmm` with `data-mode="hmrf"` ($J$ slider, ICM after each M-step, the label image beside the plain-mixture result), or reuse the Ising day's `mrfdenoise` (`data-labels="3"`) with an *estimate the means by EM* button, whichever exists first.

## Shared engineering

- `demos/src/lib/kmeans.ts` (new): Lloyd's algorithm on weighted points in $d$ dimensions (weights let histogram bins be points), with a recorded trace (assignments and centres per step) for the step-through demos; k-means++ (seeded); restarts. The thresholding day's isodata and the colour day's k-means in Lab (`lib/threshold.ts`) should call this rather than keep a third implementation.
- `demos/src/lib/gmm.ts` (new): EM on weighted points in $d \le 5$ dimensions; covariance types spherical / diagonal / full / tied; variance floor; log-sum-exp; separate `eStep` / `mStep` so demos can show half-steps; trace of $\log L$; parameter count, BIC, AIC; responsibilities and $\log p$ for a whole image; sampling from a mixture (for `gauss2d` and the synthetic datasets). The online Stauffer–Grimson update lives in the `bgmix` demo unless a second user appears. The thresholding day's `mixture` demo and Kittler–Illingworth could offer "fit by EM" through this.
- `demos/src/lib/linalg.ts` (snakes/colour plans): symmetric eigen for $2\times2$ / $3\times3$ (ellipse axes), Cholesky, determinant and inverse of small SPD matrices; no new eigen routine here.
- `demos/src/lib/plot.ts` (morphology plan): line plots ($J$, $\log L$ vs iteration), bar charts (BIC/AIC), histogram with overlaid curves (via `lib/histogram.ts`); add an **ellipse-from-covariance** helper (1/2/3σ) usable by the labelling day's equivalent ellipse and the colour day's PCA view as well.
- `lib/histogram.ts`, `lib/threshold.ts` (thresholding plan) for histograms and multi-Otsu comparison; `lib/colour.ts` (colour plan) for Lab.
- Gaussian random numbers: add a Box–Muller `gauss()` to the seeded `rng()` in `lib/ui.ts` if it is not there.
- **2D point-cloud datasets** (blobs, elongated, unequal sizes, overlapping, rings): one seeded generator module (e.g. `lib/datasets2d.ts`), shared with the SVM day's demos so both days show the same clouds with and without labels.
- Images: `demos/src/fourier2d/cell256.png` (downsampled to 128×128), the colour day's synthetic scene and H&E crop; the background-subtraction video is synthetic.
- Tests (node script, `make test`, shared with the other plans): k-means $J$ non-increasing; EM $\log L$ non-decreasing on random data (all covariance types); EM recovers known parameters on large synthetic samples; hard EM with small equal $\sigma$ equals k-means; weighted-histogram EM equals per-pixel EM; BIC picks the true $K$ on well-separated synthetic data; the by-hand numbers in this plan are reproduced.
- Performance: all in TypeScript; nothing here approaches the ~30 ms rule. Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560 on a slide.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, k-means by hand, `kmeans2d` |
| 10:45–12:15 | Unit 2 lecture, ellipse and Mahalanobis by hand, `gauss2d` |
| 13:15–14:45 | Unit 3 lecture, EM by hand (`em1d`), `em2d`, BIC exercise |
| 15:00–16:30 | Unit 4 lecture + exercise (`histgmm`, `colourgmm`); quizzes as wrap-up |

## Open questions

- Day length: four core units is full. Unit 2 could be shortened if PCA/covariance ellipses are already taught elsewhere, and unit 5 (background subtraction) could replace part of unit 4 for a video-oriented group.
- Background subtraction: optional (as proposed) or core? It is the most "image-specific" use of mixtures and has a very visual demo, but it is a different setting (online, per pixel).
- k-means ownership: this plan proposes that the algorithm (objective, convergence, k-means++) is taught here and the colour day uses it; agree with the colour plan, and make the cross-references work if only one of the days is given.
- How much of EM in general (missing data beyond mixtures) belongs in the math layer? Proposal: one slide, no exercise.
- Hidden MRF (unit 6): here, on the Ising day's `potts/` unit, or both as cross-references? One demo should serve both.
- Real data for the "populations" example (DNA content per nucleus, spot counting): is there a licensed image set with nuclei, or do we generate it synthetically?
- Should the SVM day and this day share the 2D datasets module and a slide comparing generative and discriminative models (proposed here)?

## References

- Lloyd (1982), *Least squares quantization in PCM*, IEEE Trans. Information Theory (written 1957).
- Arthur & Vassilvitskii (2007), *k-means++: the advantages of careful seeding*, SODA.
- Dempster, Laird & Rubin (1977), *Maximum likelihood from incomplete data via the EM algorithm*, J. Royal Statistical Society B.
- Bishop (2006), *Pattern Recognition and Machine Learning*, ch. 9 (k-means, mixtures, EM, the lower-bound view).
- McLachlan & Peel (2000), *Finite Mixture Models*, Wiley.
- Schwarz (1978), *Estimating the dimension of a model*, Annals of Statistics (BIC); Akaike (1974), *A new look at the statistical model identification*, IEEE TAC (AIC).
- Kittler & Illingworth (1986), *Minimum error thresholding*, Pattern Recognition (the two-component case, thresholding day).
- Stauffer & Grimson (1999), *Adaptive background mixture models for real-time tracking*, CVPR.
- Zhang, Brady & Smith (2001), *Segmentation of brain MR images through a hidden Markov random field model and the EM algorithm*, IEEE TMI.
- Rother, Kolmogorov & Blake (2004), *GrabCut*, SIGGRAPH.
