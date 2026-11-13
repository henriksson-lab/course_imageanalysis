# Plan: a day on variational autoencoders (VAEs) for images

Internal working doc (plan/ is not rendered). Status: built 2026-11-12 (units `neuralnet/`, `autoencoder/`, `vae/`, `vae-gen/`; demos `nnfit`, `linae`, `elbo1d`, `vaeimages`); sidebar and gallery entries still to be integrated.

**Course map (2026-11-13, self-study restructure):** day 38 of 51, Part VII "Learning from examples". Hard prerequisites: Neural networks (day 36). Soft (see also): Snakes and active shape models (day 30). Source: `days.yml`.

**Earlier placement (2026-11-12):** day 38 of 51, Part V "Learning from examples". Order: evaluation (day 37) → this → modern deep learning (day 39), whose diffusion unit builds on the VAE. The built `neuralnet/` moves to the neural-networks day (day 36) as its unit 1; this day starts at `autoencoder/` with a short recall.

## Goal of the day

After the day a student can:

1. *(recalled from day 36)* read an encoder and decoder as small networks ("matrix, then bend") trained by gradient descent on a loss $\mathcal L$;
2. treat an image as a **vector**, use **PCA as a linear autoencoder** (encode = project, decode = score × direction), compute a reconstruction error, and say what a nonlinear encoder/decoder adds;
3. explain what a **variational** autoencoder changes: each image becomes a Gaussian **cloud** $q(z\mid x) = \mathcal N(\mu, \sigma^2)$, the decoder is trained on draws $z = \mu + \sigma\varepsilon$ (the **reparameterisation trick**), and the loss is **reconstruction + KL** (the negative ELBO); compute the KL of two 1D Gaussians by hand;
4. **generate** images by decoding draws from the prior, **interpolate** between two images in latent space, and explain why the latent midpoint is not the pixel average;
5. name the effect of **β**, the typical failure (blurry outputs; nonsense far from the data), and three image uses: **denoising**, **anomaly detection** by reconstruction error, and **morphology latent spaces** of cells; know that GANs and diffusion models (day 39) are the alternatives.

Why it fits this audience: a network layer is a matrix–vector product, PCA is eigenvectors of a
covariance matrix, the KL between two Gaussians is a one-line integral, and the loss is
minimised by following the gradient. Networks and training are known from day 36; the new
ingredient is a *probabilistic* code.

## Prerequisite gap

- **Neural networks** are taught on day 36, whose unit 1 is the built `neuralnet/` (moved there
  by the course map; folder name kept). This day opens `autoencoder/` with a **2–3 slide recall**
  ("a layer: matrix, then bend; training = walk downhill on the loss; see day 36"), and the
  `autoencoder/` exercise keeps only the parameter-count question from `neuralnet/`. Adam:
  pointer to day 36 unit 2.
- **PCA** is owned by the `pca/` unit of day 1 (2D picture, covariance, eigenvectors).
  `autoencoder/` recalls it in one slide and keeps what is its own: **"PCA is a linear
  autoencoder"**, eigen-images in many dimensions, and the curved-cloud argument. Notation as in
  `pca/`: $\bar{\mathbf x}$, unit eigenvectors in descending order, scores, $S$ for the sample
  covariance, $k$ modes. The snakes day's point distribution model recalls `pca/` too.
- **Probability**: they know the normal distribution, mean and variance. Expectation of a
  function and the Kullback–Leibler divergence are introduced on one slide each, with the 1D
  Gaussian case worked out. No variational inference beyond "the best Gaussian approximation".
  The GMM day (More learning) meets the ELBO through EM; if taken, one line links them.
- **Generative models beyond the VAE** (GANs, diffusion) are owned by the modern deep learning
  day (day 39, `plan/moderndl-day.md`), which comes the day before: `vae-gen/` recalls diffusion in
  one slide as a comparison ("same goal, other route"), no new content.
- **Denoising / restoration**: owned by the deconvolution day (`plan/deconvolution-day.md`;
  forward model on day 4). VAE denoising stays as one application slide with that pointer.
- **Convention**: $\beta$ is used only for the β-VAE weight (course-wide rule); the loss is
  $\mathcal L$.

## Units (one folder each: lec, quiz, exercise with #explore)

Three units, about 1.5 h each including the exercise, plus a fourth session on `vae-gen/`
applications (the hour freed by moving `neuralnet/` to day 36).

### (moved) `neuralnet/`: now unit 1 of the neural-networks day (day 36)

Built here first; the course map moves it to `plan/neuralnets-day.md` (folder and demo `nnfit`
unchanged, extended there). Its content is recalled in 2–3 slides at the start of
`autoencoder/`.

### 1. `autoencoder/`: From PCA to autoencoders

- **Recall (2–3 slides)**: a layer = matrix, then bend; a network = repeated layers; training =
  gradient descent on a loss; parameter counting (a $784 \to 192 \to 2$ encoder has about 150,000
  weights). Pointer: day 36, `neuralnet/` and `backprop/`.
- An image is a vector: a $28\times28$ image is a point in $\mathbb R^{784}$. Images of one kind (digits, nuclei) fill only a thin, curved part of that space.
- PCA recall (one slide, `pca/` day 1): covariance, eigenvectors = directions of largest variance.
- **PCA is a linear autoencoder**: encode $s = \mathbf u^\top(\mathbf x - \bar{\mathbf x})$, decode $\hat{\mathbf x} = \bar{\mathbf x} + s\mathbf u$. Among all linear encoders/decoders with $k$ numbers, PCA minimises the squared reconstruction error (stated, not proved). Eigen-images.
- The **bottleneck** must be narrow; a bottleneck as wide as the input learns the identity.
- An **autoencoder** replaces both linear maps by networks (day 36): a curved 1D decoder can follow a curved cloud that PCA needs two axes for.
- **Demo `linae`** (new): a 2D point cloud (each point = a "two-pixel image"), a draggable direction for a linear 1D autoencoder, with projections, error segments and the mean squared error; button *PCA* jumps to the optimum; plot of error vs angle. Dataset *curved* (an arc): button *train a nonlinear autoencoder* ($2\to16\to1\to16\to2$, ReLU, trained live, fixed seed) draws the decoder's curve and its smaller error.
- By hand: a $1\times 2$ image patch through PCA (encode, decode, error); which images are reconstructed perfectly / as the mean; eigen-image of a $2\times2$ patch set; parameter counts (the one question kept from `neuralnet/`).

### 2. `vae/`: The variational autoencoder

- What a plain autoencoder lacks: nothing says what lies **between** training images; decoding a point in a gap can give anything.
- The VAE: encoder outputs $\mu(x), \sigma(x)$; the decoder sees a random draw $z = \mu + \sigma\varepsilon$, $\varepsilon \sim \mathcal N(0, 1)$ (**reparameterisation**: the randomness is moved into $\varepsilon$, so the gradient can flow through $\mu$ and $\sigma$).
- Loss per image $= \underbrace{\mathbb E_{q}\big[-\log p(x \mid z)\big]}_{\text{reconstruction}} + \beta\,\underbrace{\mathrm{KL}\big(q(z\mid x)\,\|\,\mathcal N(0, I)\big)}_{\text{stay near the prior}}$ (with $\beta = 1$: the negative ELBO). Squared error ↔ Gaussian likelihood; cross-entropy ↔ pixel "ink probabilities".
- KL of two Gaussians: $\mathrm{KL}(\mathcal N(\mu, s^2)\,\|\,\mathcal N(0, 1)) = \tfrac12(\mu^2 + s^2 - 1 - \ln s^2)$, derived on one slide; zero only for $\mu = 0, s = 1$.
- The tug of war: reconstruction wants small, separate clouds; KL wants all clouds to be the prior. The balance fills the latent space.
- Bayesian reading in one line: prior $p(z)$, approximate posterior $q(z\mid x)$, ELBO $\le \log p(x)$ (stated).
- **Demo `elbo1d`** (new): a VAE that can be solved exactly: 1D latent $z$, decoder $\hat x = wz$ with Gaussian pixel noise $\sigma_x$, one observed value $x$. Sliders for $\mu$, $s$ of the encoder cloud, plus $\beta$, $x$, $w$. Shows the prior and $q$, 20 reparameterised draws (fixed $\varepsilon$), their decoded values next to $x$, and bars for the expected reconstruction term, the KL and the total; *best cloud* jumps to the optimum (closed form). Presets: "optimum", "too confident", "ignore the data (q = prior)", "β = 5".
- By hand: KL for a few $(\mu, s)$; the three numbers of the demo's default state; the optimal $\mu, s$ for the linear-Gaussian case; a reparameterised draw.

### 3. `vae-gen/`: Generating and exploring images

- Generate: draw $z \sim \mathcal N(0, I)$, decode. The decoded grid of a 2D latent space (MNIST).
- Interpolation $z(t) = (1-t)z_A + tz_B$: the latent midpoint decodes to an in-between shape, the pixel average is a double exposure (by hand on a 3-pixel "moving bar" decoder).
- Latent structure: similar images nearby; classes overlap in 2D; axes have no fixed meaning.
- β-VAE: larger β → smoother, more "disentangled" but blurrier; smaller β → sharper reconstructions, gappier latent space. Shown with a **second trained weight set** (same architecture, β = 4), not only described (uses the freed hour).
- Limits: blurry outputs (the decoder outputs a mean / pixel probabilities); nonsense far from the data. Comparison slide: GANs and diffusion (recall of day 39).
- Image applications: **denoising** (decode the code of a noisy image; restoration in general is the deconvolution day, `plan/deconvolution-day.md`), **anomaly detection** (high reconstruction error = unlike the training images), **morphology latent spaces** (cell shapes, a nonlinear version of the point distribution model of the snakes day). If a licensed dataset is found: a tab on **real nucleus crops** (see Open questions).
- **Demo `vaeimages`** (adapted from 3MB00A `vaegen`, digits tab, plus a new tab): tab *Digits (MNIST)* as in the source (draw, encode, drag $z$, decoded grid, walk, sample from the cloud) plus a few real MNIST test digits as presets, a *sample from the prior* strip, and a reconstruction-error readout; new tab *Cell shapes (synthetic)*: a 2D VAE trained offline on procedurally generated binary cell masks (round, elongated, budding, with a protrusion), with an *add noise* control for denoising and a reconstruction error for anomaly detection (draw something that is not a cell).
- By hand: interpolation points; midpoint vs average on the moving-bar decoder; probability that a prior draw lands inside the plotted square; anomaly score threshold from given errors.

## What is adapted vs new

- Adapted from 3MB00A: the PCA-as-autoencoder example (two genes → two pixels), the hourglass picture, the "cloud instead of a point" argument, the walk / midpoint argument (now on images), caveats, the MNIST digit VAE (weights, `digits.ts`, `train_mnist.py`), "matrix, bend, matrix" and hinges from *Beyond linear models*.
- New: everything about the loss (ELBO, KL by hand, reparameterisation; the source unit deliberately avoided the maths), demos `nnfit`, `linae`, `elbo1d`, the cell-shape VAE and its training script, all image framing, applications to denoising/anomaly detection/morphology.
- Dropped: the single-cell tab of `vaegen`, negative binomial, scVI, all R code.

## Shared engineering

- `demos/src/lib/nn.ts` (exists; now owned by the neural-networks day, which extends it keeping `nnfit` and `linae` bit-identical): dense layers, ReLU, a tiny MLP with backprop and Adam on flat `Float64Array`s, fixed-seed init. `linae`'s PCA angle moves to the 2×2 eigen routine of `lib/linalg.ts` (owner graph-cut day; `pca/` uses the same).
- Weights files are generated offline and committed: `vaeimages/mnist_weights.ts` (copied, unchanged), `vaeimages/shape_weights.ts` (new, `train_shapes.py`, torch, fixed seed, int8 like MNIST), plus a β = 4 weight set from the same script. Dequantisation `deq` moves to `lib/q8.ts`, shared with the neural-networks day.
- Tests: a node script (`demos/test/vae-test.ts`, run by `make test`) checks the KL formula against numerical integration, the `elbo1d` closed-form optimum against a grid search, and the MNIST decoder against stored reference pixels.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 (`autoencoder/`): recall of day 36, PCA as a linear autoencoder by hand, `linae` |
| 10:45–12:15 | Unit 2 (`vae/`) lecture, KL by hand, `elbo1d` |
| 13:15–14:45 | Unit 3 (`vae-gen/`) lecture + exercise (`vaeimages`: generation, interpolation) |
| 15:00–16:30 | Unit 3 continued: β = 1 vs β = 4 weight sets, anomaly detection, cell-shape latent space (real nucleus crops if available); quizzes as wrap-up |

## Open questions

- **Decided (course map):** `neuralnet/` is unit 1 of the neural-networks day (day 36); here a 2–3 slide recall at the start of `autoencoder/`. Built-site work: move the sidebar block, add the recall slides to `autoencoder/lec.qmd`, trim `autoencoder/exercise.qmd`.
- **Decided (course map):** PCA is owned by `pca/` (day 1); `autoencoder/` keeps "PCA = linear autoencoder" and points there. The built `autoencoder/` "recap or introduction" wording becomes a recall.
- Real microscopy: a VAE on real nucleus crops (e.g. from a public dataset with a clear licence) would be closer to practice than synthetic masks; needs a dataset decision and a retrain.
- **Decided:** the β-VAE is shown with a second set of trained weights (β = 4), using the freed hour; to be added to `train_shapes.py`.

## References

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*, ICLR, arXiv:1312.6114.
- Rezende, Mohamed & Wierstra (2014), *Stochastic backpropagation and approximate inference in deep generative models*, ICML.
- Higgins et al. (2017), *β-VAE: learning basic visual concepts with a constrained variational framework*, ICLR.
- Baldi & Hornik (1989), *Neural networks and principal component analysis*, Neural Networks 2 (linear autoencoder = PCA subspace).
- Goodfellow et al. (2014), *Generative adversarial nets*, NeurIPS; Ho, Jain & Abbeel (2020), *Denoising diffusion probabilistic models*, NeurIPS.
- Ruan & Murphy (2019), *Evaluation of methods for generative modeling of cell and nuclear shape*, Bioinformatics 35. Zaritsky et al. (2021), *Interpretable deep learning uncovers cellular properties in label-free live cell images that are predictive of highly metastatic melanoma*, Cell Systems (an autoencoder latent space of cell images). Both to be checked before citing on slides.
- LeCun, Cortes & Burges, *The MNIST database of handwritten digits* (CC BY-SA 3.0 as stated in the Keras documentation).
