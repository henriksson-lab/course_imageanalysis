# Plan: a day on variational autoencoders (VAEs) for images

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

## Goal of the day

After the day a student can:

1. push a vector through a small **neural network** by hand (matrix, ReLU, matrix), explain why the bend is needed (without it the layers collapse to one matrix), and describe **training** as gradient descent on a loss;
2. treat an image as a **vector**, use **PCA as a linear autoencoder** (encode = project, decode = score × direction), compute a reconstruction error, and say what a nonlinear encoder/decoder adds;
3. explain what a **variational** autoencoder changes: each image becomes a Gaussian **cloud** $q(z\mid x) = \mathcal N(\mu, \sigma^2)$, the decoder is trained on draws $z = \mu + \sigma\varepsilon$ (the **reparameterisation trick**), and the loss is **reconstruction + KL** (the negative ELBO); compute the KL of two 1D Gaussians by hand;
4. **generate** images by decoding draws from the prior, **interpolate** between two images in latent space, and explain why the latent midpoint is not the pixel average;
5. name the effect of **β**, the typical failure (blurry outputs; nonsense far from the data), and three image uses: **denoising**, **anomaly detection** by reconstruction error, and **morphology latent spaces** of cells; know that GANs and diffusion models are the next step.

Why it fits this audience: a network layer is a matrix–vector product, PCA is eigenvectors of a
covariance matrix, the KL between two Gaussians is a one-line integral, and the loss is
minimised by following the gradient. The only genuinely new ingredients are the ReLU and the
idea of a learned (rather than derived) map.

## Prerequisite gap

- **No machine learning is assumed**, so unit 1 is a prerequisite unit on neural networks
  (adapted from *Beyond linear models* in the 3MB00A course: "matrix, bend, matrix", the
  collapse argument $C(A\mathbf x) = (CA)\mathbf x$, hinges add up to curves). Gradient descent
  is introduced with a one-weight example; backpropagation only in words ("the chain rule,
  applied layer by layer").
- **PCA**: the snakes plan (`shape-models/`) also introduces PCA, from a 2D point cloud and
  a covariance matrix. Unit 2 here contains a two-slide PCA recap that works either way: if
  *Shape models* comes first, it is a recap and the point distribution model is named as the
  linear shape latent space; if this day comes first, it is the introduction. The two decks
  should use the same notation ($\bar{\mathbf x}$, unit eigenvectors, scores).
- **Probability**: they know the normal distribution, mean and variance. Expectation of a
  function and the Kullback–Leibler divergence are introduced on one slide each, with the 1D
  Gaussian case worked out. No variational inference beyond "the best Gaussian approximation".

## Units (one folder each: lec, quiz, exercise with #explore)

Four units, about 1.5 h each including the exercise.

### 1. `neuralnet/`: Neural networks in one hour (prerequisite)

- A layer = matrix × vector + shift, then a bend (ReLU) applied to each entry. A network = repeated layers.
- By hand: one input through a $2 \to 3 \to 1$ network; the same without the bend; $CA$.
- Hinges add up to curves: one input, a few ReLU units, a piecewise-linear fit of an intensity profile across a cell boundary.
- Training = gradient descent on a loss (mean squared error); one step on a one-weight model by hand; learning rate too small/too large.
- Parameter counting: a $784 \to 192 \to 2$ encoder has about 150,000 weights. Images are big vectors.
- **Demo `nnfit`** (new): a 1D intensity profile (a synthetic line scan across two nuclei, with noise), a $1 \to H \to 1$ ReLU network trained live by gradient descent (Adam, fixed seed). Shows the fit, each hidden unit's hinge (faint), and the loss curve. Controls: $H$ (1–12), learning rate, *step*, *train 500 steps*, *reset*; checkbox *no bend* (identity instead of ReLU: the fit stays a straight line). Presets: "one hinge", "3 hinges", "12 hinges", "no bend", "learning rate too large".
- By hand: forward pass with ReLU; collapse to one matrix; one gradient-descent step on $L(w) = (y - wx)^2$; parameter counts.

### 2. `autoencoder/`: From PCA to autoencoders

- An image is a vector: a $28\times28$ image is a point in $\mathbb R^{784}$. Images of one kind (digits, nuclei) fill only a thin, curved part of that space.
- PCA recap (see Prerequisite gap): covariance, eigenvectors = directions of largest variance.
- **PCA is a linear autoencoder**: encode $s = \mathbf u^\top(\mathbf x - \bar{\mathbf x})$, decode $\hat{\mathbf x} = \bar{\mathbf x} + s\mathbf u$. Among all linear encoders/decoders with $k$ numbers, PCA minimises the squared reconstruction error (stated, not proved). Eigen-images.
- The **bottleneck** must be narrow; a bottleneck as wide as the input learns the identity.
- An **autoencoder** replaces both linear maps by networks (unit 1): a curved 1D decoder can follow a curved cloud that PCA needs two axes for.
- **Demo `linae`** (new): a 2D point cloud (each point = a "two-pixel image"), a draggable direction for a linear 1D autoencoder, with projections, error segments and the mean squared error; button *PCA* jumps to the optimum; plot of error vs angle. Dataset *curved* (an arc): button *train a nonlinear autoencoder* ($2\to16\to1\to16\to2$, ReLU, trained live, fixed seed) draws the decoder's curve and its smaller error.
- By hand: a $1\times 2$ image patch through PCA (encode, decode, error); which images are reconstructed perfectly / as the mean; eigen-image of a $2\times2$ patch set; parameter counts.

### 3. `vae/`: The variational autoencoder

- What a plain autoencoder lacks: nothing says what lies **between** training images; decoding a point in a gap can give anything.
- The VAE: encoder outputs $\mu(x), \sigma(x)$; the decoder sees a random draw $z = \mu + \sigma\varepsilon$, $\varepsilon \sim \mathcal N(0, 1)$ (**reparameterisation**: the randomness is moved into $\varepsilon$, so the gradient can flow through $\mu$ and $\sigma$).
- Loss per image $= \underbrace{\mathbb E_{q}\big[-\log p(x \mid z)\big]}_{\text{reconstruction}} + \beta\,\underbrace{\mathrm{KL}\big(q(z\mid x)\,\|\,\mathcal N(0, I)\big)}_{\text{stay near the prior}}$ (with $\beta = 1$: the negative ELBO). Squared error ↔ Gaussian likelihood; cross-entropy ↔ pixel "ink probabilities".
- KL of two Gaussians: $\mathrm{KL}(\mathcal N(\mu, s^2)\,\|\,\mathcal N(0, 1)) = \tfrac12(\mu^2 + s^2 - 1 - \ln s^2)$, derived on one slide; zero only for $\mu = 0, s = 1$.
- The tug of war: reconstruction wants small, separate clouds; KL wants all clouds to be the prior. The balance fills the latent space.
- Bayesian reading in one line: prior $p(z)$, approximate posterior $q(z\mid x)$, ELBO $\le \log p(x)$ (stated).
- **Demo `elbo1d`** (new): a VAE that can be solved exactly: 1D latent $z$, decoder $\hat x = wz$ with Gaussian pixel noise $\sigma_x$, one observed value $x$. Sliders for $\mu$, $s$ of the encoder cloud, plus $\beta$, $x$, $w$. Shows the prior and $q$, 20 reparameterised draws (fixed $\varepsilon$), their decoded values next to $x$, and bars for the expected reconstruction term, the KL and the total; *best cloud* jumps to the optimum (closed form). Presets: "optimum", "too confident", "ignore the data (q = prior)", "β = 5".
- By hand: KL for a few $(\mu, s)$; the three numbers of the demo's default state; the optimal $\mu, s$ for the linear-Gaussian case; a reparameterised draw.

### 4. `vae-gen/`: Generating and exploring images

- Generate: draw $z \sim \mathcal N(0, I)$, decode. The decoded grid of a 2D latent space (MNIST).
- Interpolation $z(t) = (1-t)z_A + tz_B$: the latent midpoint decodes to an in-between shape, the pixel average is a double exposure (by hand on a 3-pixel "moving bar" decoder).
- Latent structure: similar images nearby; classes overlap in 2D; axes have no fixed meaning.
- β-VAE: larger β → smoother, more "disentangled" but blurrier; smaller β → sharper reconstructions, gappier latent space.
- Limits: blurry outputs (the decoder outputs a mean / pixel probabilities); nonsense far from the data. Pointers: GANs, diffusion models.
- Image applications: **denoising** (decode the code of a noisy image), **anomaly detection** (high reconstruction error = unlike the training images), **morphology latent spaces** (cell shapes, a nonlinear version of the point distribution model of *Shape models*).
- **Demo `vaeimages`** (adapted from 3MB00A `vaegen`, digits tab, plus a new tab): tab *Digits (MNIST)* as in the source (draw, encode, drag $z$, decoded grid, walk, sample from the cloud) plus a few real MNIST test digits as presets, a *sample from the prior* strip, and a reconstruction-error readout; new tab *Cell shapes (synthetic)*: a 2D VAE trained offline on procedurally generated binary cell masks (round, elongated, budding, with a protrusion), with an *add noise* control for denoising and a reconstruction error for anomaly detection (draw something that is not a cell).
- By hand: interpolation points; midpoint vs average on the moving-bar decoder; probability that a prior draw lands inside the plotted square; anomaly score threshold from given errors.

## What is adapted vs new

- Adapted from 3MB00A: the PCA-as-autoencoder example (two genes → two pixels), the hourglass picture, the "cloud instead of a point" argument, the walk / midpoint argument (now on images), caveats, the MNIST digit VAE (weights, `digits.ts`, `train_mnist.py`), "matrix, bend, matrix" and hinges from *Beyond linear models*.
- New: everything about the loss (ELBO, KL by hand, reparameterisation; the source unit deliberately avoided the maths), demos `nnfit`, `linae`, `elbo1d`, the cell-shape VAE and its training script, all image framing, applications to denoising/anomaly detection/morphology.
- Dropped: the single-cell tab of `vaegen`, negative binomial, scVI, all R code.

## Shared engineering

- `demos/src/lib/nn.ts` (new, small): dense layers, ReLU, a tiny MLP with backprop and Adam on flat `Float64Array`s, fixed-seed init. Used by `nnfit` and `linae`. No dependency on other plans' libraries; if `lib/linalg.ts` (snakes plan) lands, `linae` can use its 2×2 eigen routine instead of the closed form.
- Weights files are generated offline and committed: `vaeimages/mnist_weights.ts` (copied, unchanged), `vaeimages/shape_weights.ts` (new, `train_shapes.py`, torch, fixed seed, int8 like MNIST).
- Tests: a node script (`demos/test/vae-test.ts`) checks the KL formula against numerical integration, the `elbo1d` closed-form optimum against a grid search, and the MNIST decoder against stored reference pixels.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + exercise (`nnfit`) |
| 10:45–12:15 | Unit 2 lecture + PCA by hand, `linae` |
| 13:15–14:45 | Unit 3 lecture, KL by hand, `elbo1d` |
| 15:00–16:30 | Unit 4 lecture + exercise (`vaeimages`); quizzes as wrap-up |

## Open questions

- If the course has an earlier machine-learning day, unit 1 shrinks to a recap or is dropped.
- Order relative to the snakes day: whoever comes first owns the PCA introduction (see Prerequisite gap).
- Real microscopy: a VAE on real nucleus crops (e.g. from a public dataset with a clear licence) would be closer to practice than synthetic masks; needs a dataset decision and a retrain.
- Should the β-VAE be shown with a second set of trained weights (same architecture, β = 4) instead of only described? Cheap to add to `train_shapes.py`.

## References

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*, ICLR, arXiv:1312.6114.
- Rezende, Mohamed & Wierstra (2014), *Stochastic backpropagation and approximate inference in deep generative models*, ICML.
- Higgins et al. (2017), *β-VAE: learning basic visual concepts with a constrained variational framework*, ICLR.
- Baldi & Hornik (1989), *Neural networks and principal component analysis*, Neural Networks 2 (linear autoencoder = PCA subspace).
- Goodfellow et al. (2014), *Generative adversarial nets*, NeurIPS; Ho, Jain & Abbeel (2020), *Denoising diffusion probabilistic models*, NeurIPS.
- Ruan & Murphy (2019), *Evaluation of methods for generative modeling of cell and nuclear shape*, Bioinformatics 35. Zaritsky et al. (2021), *Interpretable deep learning uncovers cellular properties in label-free live cell images that are predictive of highly metastatic melanoma*, Cell Systems (an autoencoder latent space of cell images). Both to be checked before citing on slides.
- LeCun, Cortes & Burges, *The MNIST database of handwritten digits* (CC BY-SA 3.0 as stated in the Keras documentation).
