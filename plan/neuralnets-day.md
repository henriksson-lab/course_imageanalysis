# Plan: a day on neural networks for images

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-12):** core day 15 of 18, Part V "Learning from examples". Order: random forests (day 14) → this → evaluation and statistics (day 16) → modern deep learning (day 18) → VAEs (day 17). Unit 1 is the built `neuralnet/` (moved here from the VAE part).

## Goal of the day

After the day a student can:

1. compute the output of a **neuron** (weighted sum + bias, then a nonlinearity) and of a small **multilayer network** by hand, draw the decision line of one neuron, and explain why the **nonlinearity** is needed (XOR; stacked linear layers collapse to one matrix);
2. choose a **loss** (mean squared error for numbers, softmax + cross-entropy for classes), describe **training** as gradient descent on it, and say what the learning rate and the mini-batch size do;
3. run **backpropagation** by hand on a two-neuron network: forward values, then gradients by the chain rule on the computational graph, then one update step;
4. recognise **overfitting** from training and validation **learning curves** (train / validation / test recalled from day 14) and name three defences (weight decay, dropout, data augmentation);
5. explain a **convolutional layer** as a bank of learned kernels with weight sharing, count its parameters, compute output sizes and the **receptive field** of a stack of layers, and describe the edges → textures → parts hierarchy;
6. explain how an **encoder–decoder / U-Net** produces a per-pixel segmentation, why it needs **skip connections**, and train it with a per-pixel cross-entropy or a **Dice loss**; evaluate it with Dice / IoU;
7. *(optional unit)* name what goes wrong in practice (**shortcut learning**, **domain shift**, too little data) and what batch normalisation, residual connections and pretrained networks are for (days 16 and 18 continue this).

Why it fits this audience: a layer is a matrix–vector product, training is "follow the
negative gradient", backpropagation is the chain rule applied in a clever order, and a
convolutional layer is the 3×3 kernel of `kernels/` (day 1) with weights that are
learned instead of designed. Nothing beyond linear algebra, the chain rule and the normal
distribution is needed; the genuinely new ideas are *learning a map from examples* and
*generalisation*.

## Prerequisite gap and links to other days

- **Classification basics are known from day 14** (random forests, the course owner of
  feature space, train / validation / test, overfitting, cross-validation and the pixel feature
  stack). Unit 1 recalls the 2D feature scatter and "a classifier carves up feature space" in one
  slide; unit 2 recalls train / validation / test and overfitting in one slide and adds what is
  new here: **learning curves** (loss vs epoch on both sets), early stopping and regularisation
  (weight decay, dropout).
- **Convolution** (`kernels/`, day 1): a weighted 3×3 neighbourhood sum, border rules, derivative
  kernels. Unit 3 recalls it in one slide ("the kernel slides over the image") and
  reuses the `kernel3x3` demo's code (`demos/src/kernel3x3/conv.ts`) for the hand-made
  kernel bank. Correlation vs convolution: one line ("networks actually correlate; with
  learned weights the flip does not matter").
- **Thresholding** (day 6): a threshold on one feature is a neuron with one input and a step
  nonlinearity (unit 1, first slide). Pixel-level evaluation (confusion matrix, **Dice**,
  **Jaccard/IoU**, class imbalance: "accuracy lies") is owned by `seg-eval/` there; unit 4
  recalls Dice in one line and builds the **Dice loss** on it, and uses Otsu as the baseline in
  its demo. Object-level evaluation (matching, F1/AP) is the next day but one (day 16).
- **The hand-made feature stack** (day 14, `pixelclass/`): unit 3 opens from it: "yesterday the
  filters were chosen by hand (Gaussian, LoG, Hessian from the edges day); a CNN learns them".
  The forest on a stack is the baseline the U-Net (unit 4) is compared with.
- **PCA** (`pca/`, day 1): only touched in unit 1's math layer ("a network with one linear hidden
  layer and squared error learns the PCA subspace"; the VAE day's `autoencoder/` shows it).
- **Classical vs learned features** (measurements, day 11: Haralick, Gabor; its texture unit
  already has the pointer "CNN embeddings as learned texture features"): unit 3 shows
  first-layer filters next to a hand-made bank (Sobel, Laplacian, Gaussian, oriented edges) and
  states the trade-off: designed features are interpretable and need no training data; learned
  features adapt to the task but need labelled examples.
- **Graph cuts** (day 13): a network's per-pixel probability gives the data term
  $D_p(\ell) = -\log p_\ell$; graph-cut smoothing on top of a CNN is one slide in unit 4
  ("the two ideas combine").
- **Watershed / labelling** (days 9–10): a U-Net gives a *semantic* mask; touching nuclei still
  need splitting. One pointer slide in unit 4 (a third "boundary" class, then labelling or
  watershed); **instance segmentation by networks** (boundary class, distance map + watershed,
  star-convex polygons, flow fields) is owned by the modern deep learning day (day 18,
  `plan/moderndl-day.md`).
- **Evaluation and statistics** (day 16, `plan/evaluation-day.md`) and **modern deep learning**
  (day 18): own leakage and honest test sets, shortcut learning and domain shift as evaluation
  problems, transfer learning, pretraining, transformers, detection and diffusion. The optional
  `deep-practice/` here gives them one slide each and points forward.
- **Learned denoising / restoration**: owned by the deconvolution day
  (`plan/deconvolution-day.md`); the PSF / forward model by image formation (day 4). Pointer only.
- **Ising / Hopfield** (Ising elective, optional unit `boltzmann/`): energy-based
  networks are a different family; one "Where this leads" line, no unit here.

### Split with the VAE day (`plan/vae-day.md`) — decided (course map)

The built unit `neuralnet/` ("matrix, bend, matrix", collapse to $CA$, hinges add up to curves,
one gradient step on $\mathcal L(w) = (y - wx)^2$, parameter counting; demo `nnfit`) was written
as unit 1 of the VAE part. The course map moves it here:

- **Unit 1 of this day is the built `neuralnet/`** (folder name kept, as the sidebar rule says
  never rename a folder; the planned `neurons/` folder name is dropped). It is **extended** with
  the neuron as a linear classifier, AND/OR/XOR, softmax and the `nnplayground` demo. The by-hand
  forward pass ($A$, $C$, $\mathbf x = (1, 2)$, answer 6 vs 4), the collapse slide, the hinges
  figure `neuralnet/fig-hinges.svg` and `nnfit` stay as built. Unit 2 reuses the one-weight
  gradient-step numbers (learning rates 0.1 / 0.125 / 0.25 / 0.5 → loss 0.36 / 0 / 9 / 81).
- **This day owns** neurons, layers, activations, losses, gradient descent, learning rate,
  backpropagation, mini-batches, learning curves and regularisation, CNNs, training practice,
  segmentation networks.
- **The VAE day (day 17)** starts at `autoencoder/` with a 2–3 slide recall ("a layer: matrix,
  then bend; training = walk downhill; see day 15"); its exercise keeps the parameter-count
  question only. The sidebar block `neuralnet/` moves into the neural-networks part.
- **Notation**: `neuralnet/` uses $\mathbf h = \mathrm{ReLU}(A\mathbf x + \mathbf b)$,
  out $= C\mathbf h + d$. This day generalises to $\mathbf a^{(l)} = \varphi(W^{(l)}\mathbf
  a^{(l-1)} + \mathbf b^{(l)})$ and the extended `neuralnet/` says once "$A = W^{(1)}$,
  $C = W^{(2)}$". Activation is $\varphi$ (never $\sigma$: course convention); loss $\mathcal L$.
- **Shared code**: `demos/src/lib/nn.ts` (written for `neuralnet/`/VAE) is extended, not
  duplicated (see Shared engineering); the int8 weight format and offline-training
  pattern of `vaeimages` (`train_shapes.py`, `mnist_weights.ts`, `demos/test/vae-check.ts`)
  are the template for the CNN and U-Net weights; the synthetic cell-mask generator of
  `train_shapes.py` (round / elongated / dividing / protrusion, with labels) is reused as
  a 4-class classification dataset in unit 3; the MNIST digit examples and the drawing
  pad of `vaeimages` are reused in `cnnviewer`.
- The VAE units use "Adam" without explanation; unit 2 here explains it in one math-layer line
  (momentum + a per-weight step size), so the VAE notes point here.

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit.

### 1. `neuralnet/` (built, extended): From a neuron to a network

The built unit (by-hand forward pass, collapse, hinges, `nnfit`) plus the additions below
(neuron as a linear classifier, XOR, softmax, `nnplayground`).

- Recall (one line): a threshold on one measurement, "nucleus if intensity > $t$". With two measurements (area, mean intensity; a scatter plot of objects) the natural rule is a **line**: $w_1x_1 + w_2x_2 + b > 0$.
- **A neuron**: weighted sum + bias, then a nonlinearity $\varphi$. Step → a hard decision (the perceptron); sigmoid → a probability (logistic regression in one line); ReLU → the workhorse. Weights = the direction of the line, bias = its offset.
- One neuron can do AND, OR, but not **XOR** (no single line separates the corners). Two hidden ReLU units + one output do it (by hand).
- **Layers**: a layer is a matrix–vector product + bias, then $\varphi$ on each entry (the "matrix, bend, matrix" picture of `neuralnet/`). Hidden units, width, depth.
- **Why the nonlinearity**: without it, layers collapse to one matrix, $W^{(2)}(W^{(1)}\mathbf x) = (W^{(2)}W^{(1)})\mathbf x$; the decision boundary stays a line.
- With it: each hidden ReLU unit folds space along one line; the output combines the pieces into a **piecewise-linear boundary** (2D) or curve (1D, hinges add up: `nnfit`). **Universal approximation** as intuition: enough hinges approximate any continuous function; it says nothing about how many units, or whether training finds them.
- **Many classes**: one output per class, **softmax** turns scores into probabilities; predicted class = largest.
- Images as inputs: a 28×28 image is a vector in $\mathbb R^{784}$; a dense layer to 100 units has 78,500 weights, and shifting the digit by one pixel changes almost every input. Motivation for unit 3.
- The math: neuron as $\varphi(\mathbf w^\top\mathbf x + b)$, hyperplane geometry ($\mathbf w$ is the normal, distance $|b|/\|\mathbf w\|$), the layer formula, collapse with biases, softmax $p_k = e^{z_k}/\sum_j e^{z_j}$ and its invariance to adding a constant; universal approximation stated (Cybenko 1989; ReLU version) without proof.
- **Demo `nnplayground`** (new, main demo of the day; units 1 and 2): 2D point sets with two or three classes (presets: two blobs, XOR, circle inside ring, two moons, spiral; noise slider; fixed seed). Network: 0–2 hidden layers of 1–8 units, activation ReLU / tanh / sigmoid / *none*. Shows the decision regions as a coloured background with the probability as shading, the points, and a small thumbnail per hidden unit with *its own* region (first-layer units are lines; second-layer units are polygons): the student sees the pieces being combined. Live training with *step*, *play/pause*, *reset*; loss curve. Mode *by hand* (0 hidden layers): drag the neuron's line (= $\mathbf w$, $b$) directly and read the loss. Presets: "one neuron, linearly separable", "one neuron on XOR fails", "2 hidden units solve XOR", "no nonlinearity: still a line", "spiral needs depth", "3 classes with softmax". Unit 2 enables the training controls (see there) via `data-*` attributes.
- **`nnfit`** (built, in this unit) with `data-h="12"` for the 1D universal-approximation picture.
- By hand: (a) output and decision line of a neuron with given $\mathbf w$, $b$; which of four points are classified positive; (b) weights for AND and OR; (c) the XOR truth table through a given 2–2–1 ReLU network; (d) the collapse $W^{(2)}W^{(1)}$ for a 2–3–1 example (the `neuralnet/` numbers); (e) softmax of $(2, 1, 0)$ and of $(102, 101, 100)$; (f) parameter counts of a 784–100–10 network. *(math)*: show that a ReLU network is piecewise linear; a one-hidden-layer network with $n$ units in 1D has at most $n$ kinks.

### 2. `backprop/`: Learning: loss, gradient descent and backpropagation

- The **loss** measures how wrong the network is on the training examples: mean squared error for numbers; **cross-entropy** $-\log p_{\text{true class}}$ for classes (a confident wrong answer is punished hard; numbers: $p = 0.9 \to 0.11$, $p = 0.01 \to 4.6$).
- **Gradient descent**: the loss is a landscape over all weights; step downhill, step length = **learning rate** $\eta$. Too small: slow; too large: oscillates or diverges (recall the one-weight numbers of unit 1). Different starts give different networks.
- **Mini-batch / stochastic** gradient descent: the gradient on a random batch of 32 images is a noisy, cheap estimate of the full one; an **epoch** = one pass over the data. The noise is not only tolerable but helps escape flat regions.
- **Backpropagation**: the network as a **computational graph** (multiply, add, ReLU, loss nodes). Forward pass: values left to right. Backward pass: each node multiplies the incoming gradient by its *local* derivative and passes it on (the chain rule, organised). Cost: about twice a forward pass, whatever the number of weights; that is why networks with millions of weights can be trained.
- What the local derivatives tell us: ReLU passes the gradient or blocks it (a "dead" unit never learns); a saturated sigmoid passes almost nothing (**vanishing gradients**, the reason deep sigmoid networks were hard and ReLU won).
- **Overfitting**, recalled from day 14 in one slide (train / validation / test, the U-shaped test curve). New here: **learning curves** for a network, the training loss keeps falling while the validation loss rises; choose size, $\eta$ and **when to stop** (early stopping) on validation, report once on test.
- **Regularisation**: weight decay (penalise $\|\mathbf w\|^2$: smoother boundaries), dropout (randomly silence units during training: no unit can rely on another), more data / **data augmentation** (unit 3, images).
- The math: $\mathbf w \leftarrow \mathbf w - \eta\nabla_{\mathbf w}\mathcal L$ (the loss is written $\mathcal L$ on slides; $L$ below for brevity); chain rule on the graph, $\partial L/\partial w = \sum_{\text{paths}}\prod$ local derivatives; for a layer, $\boldsymbol\delta^{(l-1)} = (W^{(l)})^\top\boldsymbol\delta^{(l)} \odot \varphi'(\mathbf z^{(l-1)})$ and $\partial L/\partial W^{(l)} = \boldsymbol\delta^{(l)}(\mathbf a^{(l-1)})^\top$ (an outer product); softmax + cross-entropy gives the gradient $\mathbf p - \mathbf y$; mini-batch gradient is an unbiased estimate of the full gradient; weight decay adds $2\lambda\mathbf w$; momentum and Adam in one line each.
- **Demo `backprop`** (new): the network $x \to z = w_1x + b_1 \to h = \mathrm{ReLU}(z) \to \hat y = w_2h + b_2 \to L = (\hat y - y)^2$ (and a 2–2–1 variant via `data-net`), drawn as a graph in SVG. Buttons: *forward* (values appear node by node, blue), *backward* (gradients appear node by node, red, with the local derivative on each edge), *update* (weights change by $-\eta\cdot$gradient, loss before/after), *reset*. Weights, $x$, $y$, $\eta$ editable. A *check* button compares each gradient with a finite difference. Presets: "the by-hand exercise", "dead ReLU: all gradients upstream are 0", "sigmoid saturates", "learning rate too large: the loss goes up".
- **Demo `gdescent`** (new, small): contour plot of a 2-parameter loss (an elongated bowl; a landscape with two minima; the real loss of a one-neuron classifier on a small point set). Click to start, watch the path. Sliders: $\eta$, batch size (full batch → smooth path; small batch → wandering path), momentum on/off. Presets: "too small", "just right", "zig-zag across a narrow valley", "diverges", "SGD noise", "two starts, two minima".
- **`nnplayground`** with training controls: learning rate, batch size, **train/validation split** with both loss curves, network size, weight decay, dropout. Presets: "overfitting: 8+8 units on 30 noisy points", "weight decay smooths it", "more data cures it", "learning rate too large", "dead network".
- By hand: (a) cross-entropy of a few predictions; (b) one gradient step on $L(w) = (y - wx)^2$ (same numbers as `neuralnet/`); (c) **backprop on the two-neuron network** with $x = 2$, $y = 1$, $w_1 = 0.5$, $b_1 = 0$, $w_2 = 1.5$, $b_2 = 0$: forward $z = 1$, $h = 1$, $\hat y = 1.5$, $L = 0.25$; backward $\partial L/\partial\hat y = 1$, $\partial L/\partial w_2 = 1$, $\partial L/\partial b_2 = 1$, $\partial L/\partial h = 1.5$, $\partial L/\partial w_1 = 3$, $\partial L/\partial b_1 = 1.5$; update with $\eta = 0.1$ gives $L = 0.5625$ (worse!), with $\eta = 0.05$ gives $L \approx 0.021$; and the same with $x = -1$ (ReLU off: which gradients vanish?); (d) given training/validation curves, mark where to stop and which curve belongs to which set; (e) steps per epoch for 60,000 images in batches of 32. *(math)*: derive $\partial L/\partial\mathbf z = \mathbf p - \mathbf y$ for softmax + cross-entropy; derive the layer backprop rule; show that a mini-batch gradient is unbiased.

### 3. `cnn/`: Convolutional networks

- Opening (one slide): day 14's feature stack was a hand-made first layer (Gaussian, LoG, Hessian filters at chosen scales). A CNN learns that layer, and the ones after it, from examples.
- Problem with dense layers on images: too many weights, and no notion that neighbouring pixels belong together or that a nucleus is a nucleus wherever it is.
- **A convolutional layer** = a bank of 3×3 kernels (`kernels/`, day 1), one output image (**feature map**) per kernel, plus a bias and ReLU. The kernel values are **learned**. Hand-made vs learned: the first layer of a trained network looks like edge, line and blob detectors.
- **Weight sharing**: the same 9 weights at every position. Parameter count independent of image size ($3\cdot3\cdot C_\text{in}\cdot C_\text{out} + C_\text{out}$). **Translation equivariance**: shift the input, the feature maps shift.
- **Channels**: an RGB image has 3; layer 2 sees all $C$ feature maps of layer 1, so each kernel is $3\times3\times C$ and combines features ("an edge here *and* a blob there").
- **Pooling and stride**: 2×2 max pooling halves the size, keeps "is the feature somewhere here", and gives a little invariance to small shifts. Output size $\lfloor(n + 2p - k)/s\rfloor + 1$.
- **Receptive field**: each layer sees a slightly bigger part of the input; pooling/stride multiplies the growth. Three 3×3 layers see 7×7; with pooling in between much more. Deep layers see whole objects.
- **Feature hierarchy**: edges → textures/corners → parts → objects. Visualising: filters (layer 1 only makes sense as pictures), feature maps for an input, the image patches that excite a unit most, occlusion (cover part of the image, watch the class probability).
- **A classification network end to end**: conv–ReLU–pool ×2, flatten, dense, softmax; trained with cross-entropy (unit 2). Counting its weights: a few thousand instead of hundreds of thousands.
- **Data augmentation**: train on shifted, rotated, flipped, brightness-changed copies; the network learns invariances we can name (link to the moments elective: there invariance was built into the features; here it is taught by examples).
- The math: discrete multi-channel correlation $y_{c'}[i, j] = b_{c'} + \sum_c\sum_{u,v} w_{c',c}[u, v]\,x_c[i + u, j + v]$; a convolutional layer is a dense layer with a sparse, repeated (Toeplitz-like) weight matrix; equivariance $T_s(\text{conv}(x)) = \text{conv}(T_s x)$, broken by stride/pooling; receptive field recursion $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$, $j_l = j_{l-1}s_l$; backprop through a convolution is again a correlation (one line).
- **Demo `convlayer`** (new; reuses `kernel3x3/conv.ts`): input (the micrograph `cell256.png`, a digit, a synthetic cell mask) and a bank of 8 kernels shown as small 3×3 grids with their feature maps after ReLU underneath. Bank switch: *hand-made* (Sobel x/y, Laplacian, Gaussian, two diagonals, …) vs *learned* (layer 1 of the `cnnviewer` network). Click a kernel to edit it (as in `kernel3x3`); *shift input* buttons show the maps shifting with it; *2×2 max pool* toggle shows the halved maps. Presets: "learned filters look like edge detectors", "shift the input", "pooling".
- **Demo `receptive`** (new, small, SVG): a 1D (and 2D) stack of up to 5 layers with per-layer kernel size, stride, pooling (and dilation as an extra); click a unit at the top, its receptive field lights up in every layer below with the size printed. Presets: "three 3×3 layers = 7", "with pooling", "one 7×7 vs three 3×3 (same field, fewer weights)", "how deep to see a whole nucleus?".
- **Demo `cnnviewer`** (new, precomputed weights): a small CNN (conv 3×3×8 → pool → conv 3×3×16 → pool → dense → softmax, about 10k weights) trained offline on two datasets: MNIST digits and the synthetic cell masks of the VAE day (4 classes). Draw or pick an input (drawing pad as in `vaeimages`); see the feature maps of every layer, the class probabilities as bars, the layer-1 filters, and for each layer-2 unit its 9 most strongly activating training patches (precomputed). Sliders *shift* and *rotate* the input; *occlusion map* button. Weight sets: *with* and *without* rotation augmentation; *training snapshots* (epochs 0, 1, 3, 10) show filters forming from noise. Presets: "a 6 rotated to a 9", "shifted digit: maps shift, prediction stays", "rotation without/with augmentation", "a protrusion cell", "what is the network looking at?". Training in the browser is not needed for the point and is left out (see Open questions).
- By hand: (a) a 6×6 input through two given 3×3 kernels (valid padding → two 4×4 maps), ReLU, 2×2 max pool (→ two 2×2 maps); (b) parameter counts of a conv layer vs a dense layer on the same image; (c) output sizes for given $n, k, p, s$; (d) receptive field of a given stack; (e) shift a 1D signal, convolve, compare (equivariance), and show that max pooling with stride 2 breaks it for a one-pixel shift. *(math)*: write a 1D conv layer as a matrix (banded Toeplitz); show that two stacked 3×3 layers without nonlinearity equal one 5×5 kernel (kernels combine by convolution); prove the receptive-field recursion.

### 4. `unet/`: Segmentation networks

- From "what is in the image" to "which pixels": one class probability **per pixel**. Naive: classify a patch around every pixel (slow, small context).
- **Encoder–decoder**: the encoder (unit 3) shrinks and gains context; the decoder **upsamples** back to full size (nearest/bilinear upsampling + convolution, or transposed convolution; by hand in 1D).
- The decoder alone produces blurry boundaries: the "where exactly" was lost by pooling. **U-Net** (Ronneberger et al. 2015): **skip connections** copy encoder feature maps of the same size to the decoder (concatenate channels): context from below, detail from the side.
- **Per-pixel loss**: cross-entropy averaged over pixels. Class imbalance (nuclei are 5 % of pixels; recall "accuracy lies" from `seg-eval/`): weight the classes, or use the **Dice loss** $1 - \text{soft Dice}$, computed from probabilities so it has a gradient.
- **Evaluation**: Dice / IoU against ground truth on held-out images (owned by `seg-eval/`; one-line recall); look at the errors (merged nuclei, missed dim cells) not only the number.
- Data: annotations are expensive; augmentation (flips, rotations, intensity, elastic deformation) and small networks go a long way with tens of annotated images. Synthetic images with known truth for teaching.
- After the mask: touching nuclei (one pointer slide: a third "boundary" class, then labelling / watershed of days 9–10; network-based instance segmentation is day 18); the probability map as the data term of a graph cut (day 13). Baseline: day 14's forest on the feature stack.
- The math: soft Dice $D = 2\sum_p p_p g_p / (\sum_p p_p + \sum_p g_p)$ and its gradient; weighted cross-entropy; transposed convolution as the transpose of the conv matrix of unit 3; receptive field of the bottleneck.
- **Demo `unetseg`** (new, precomputed weights): a tiny U-Net (two levels, 8–16–32 channels, about 25k weights) trained offline on synthetic fluorescence-like images, 64×64 (nuclei of varying size and brightness, touching pairs, uneven illumination, noise; generated by the shared `lib/synth.ts` generator with ground truth, so *new image* gives unlimited test images). Panels: input, probability map, mask at a threshold slider, error overlay (TP/FP/FN as in `threshroc`) with Dice and IoU, and the Otsu mask with its Dice for comparison. Click a level of a small architecture diagram to see its feature maps. Weight sets: *U-Net*, *no skip connections*, *trained with plain cross-entropy* vs *Dice loss*. Sliders that move the input away from the training data: noise, illumination gradient, brightness scale, object size (domain shift). Presets: "U-Net vs Otsu on uneven illumination", "no skips: blurry boundaries", "touching nuclei merge", "too noisy: outside the training range", "brighter microscope: domain shift".
- By hand: (a) soft Dice and cross-entropy of a 3×3 probability map against a mask; the same for the "all background" prediction on a 5 %-object image (cross-entropy looks fine, Dice does not); (b) 1D upsampling + convolution / transposed convolution of a 3-sample signal; (c) channel counts and sizes through a 2-level U-Net on a 64×64 input, including the concatenations; (d) receptive field at the bottleneck: can one output pixel see a whole 20-pixel nucleus? *(math)*: derive the gradient of soft Dice with respect to one $p_p$; show transposed convolution is the adjoint of strided convolution.

### 5. (optional / further reading) `deep-practice/`: Deep networks in practice, and what goes wrong

- **Deeper is harder**: vanishing gradients (recall unit 2). **Residual connections** ($\mathbf a + F(\mathbf a)$: the identity path lets the gradient through; ResNet). **Batch normalisation** (standardise each channel over the batch, then learned scale and shift; training becomes less sensitive to $\eta$ and initialisation).
- **Transfer learning**: one slide; pretraining, self-supervised learning and generalist cell segmenters are owned by day 18 (`pretrained/`).
- **Shortcut learning**: the network learns whatever separates the training classes, e.g. a scale bar, a plate edge or the scanner, not the biology. **Domain shift**: a new microscope, stain or cell line. Always test on data from a new source (how to build such a test set: day 16); occlusion maps help detect shortcuts.
- Pointers, one slide: attention / **transformers** (the image as a set of patches; day 18), generative models (VAEs, day 17; diffusion, day 18), learned denoising (`plan/deconvolution-day.md`).
- **Demo**: reuse `cnnviewer` with `data-weights="shortcut"`: a weight set trained on cell masks where one class always carries a small corner mark; it scores perfectly on validation data with the mark and fails without it; the occlusion map points at the corner. Reuse `unetseg` domain-shift presets.
- By hand: (a) batch-normalise 4 numbers; (b) why $\partial(\mathbf a + F(\mathbf a))/\partial\mathbf a = I + \partial F/\partial\mathbf a$ cannot vanish; (c) design a test set that would catch a given shortcut.

## Shared engineering

- **`demos/src/lib/nn.ts`** (exists, written for `neuralnet/`; owned by this day now): extend, keeping `nnfit` and `linae` bit-identical (regression test with their fixed seeds): activations tanh / sigmoid, softmax + cross-entropy loss, mini-batch sampling (seeded), plain SGD with momentum next to the existing Adam, weight decay, dropout (training mode only), and a `forward` that returns per-unit outputs for the playground thumbnails. Networks stay tiny (≤ ~300 weights, ≤ 500 points): a full-batch step is far below 1 ms, so `nnplayground` can run ~10 steps per animation frame; the decision-region background is evaluated on a 64×64 grid (4k forward passes) per frame. No Wasm needed.
- **`demos/src/lib/autodiff.ts`** (new, ~100 lines): scalar reverse-mode autodiff (nodes with op, value, grad, parents; topological backward) that also records the graph for drawing. Used by `backprop` and by the tests (gradients of `nn.ts` checked against it and against finite differences). Not used for training.
- **`demos/src/lib/cnn.ts`** (new, inference only): multi-channel 2D correlation (same/valid padding, stride), ReLU, max pool, nearest upsampling, transposed convolution, channel concatenation, dense, softmax, on `Float32Array`s in channel-major layout. Generalises `kernel3x3/conv.ts` (single channel, not modified). Cost: `cnnviewer` ≈ 0.3 M multiply-adds per image, `unetseg` on 64×64 ≈ 10–20 M: a few ms in TS, so sliders stay live. Measure before thinking of Wasm.
- **Weights**: generated offline and committed, int8 with per-output-channel scale as in `vaeimages` (`Q8Layer`, `deq`); move `deq` into `lib/q8.ts` shared with the VAE day. Scripts next to the demos, as `train_shapes.py`: `cnnviewer/train_cnn.py` (MNIST + cell masks, with/without augmentation, epoch snapshots, shortcut variant), `unetseg/train_unet.py` (four weight sets). Python with a deep-learning framework, fixed seed, minutes on a CPU; the framework is named only in these teacher scripts.
- **One source for synthetic data**: the nuclei-image generator lives in TypeScript (`lib/synth.ts`, owned by thresholding); a node script writes the U-Net training set to raw `.u8` files that the Python script reads. So the browser's *new image* button draws from exactly the training distribution, and the domain-shift sliders move away from it in known ways. The cell-mask generator stays in Python (`train_shapes.py`) as now; `cnnviewer` stores a few hundred test masks.
- **Tests** (`demos/test/nn-test.ts`, run by `make test`): autodiff vs finite differences; `nn.ts` gradients (incl. softmax/cross-entropy, weight decay) vs autodiff; `cnn.ts` layers vs naive loops and vs `kernel3x3/conv.ts`; TS inference of both networks vs reference outputs stored by the training scripts (as `vae-check.ts` does); Dice/IoU via `lib/metrics.ts` (segmentation section, owned by thresholding); the by-hand numbers quoted in exercises (two-neuron backprop, Dice examples) computed by the same code.
- Images and licences: `cell256.png` (R. M. Hunt, CC BY 3.0) for `convlayer`; MNIST as in the VAE plan; everything else synthetic. Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560; static figures as print fallbacks (decision regions, the backprop graph with numbers, a feature-map grid, the U-Net diagram).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 (`neuralnet/`, extended) lecture + exercise (`nnplayground` by hand / XOR, `nnfit`) |
| 10:45–12:15 | Unit 2 lecture, backprop by hand, `backprop`, `gdescent`, overfitting in `nnplayground` |
| 13:15–14:45 | Unit 3 lecture, conv layer and receptive field by hand, `convlayer`, `receptive`, `cnnviewer` |
| 15:00–16:30 | Unit 4 lecture + exercise (`unetseg`); quizzes as wrap-up |

## Open questions

- Day length: unit 2 is the densest (losses, gradient descent, backprop, mini-batches, learning curves, regularisation); the train/test recall from day 14 saves some time. If still too much, regularisation could move into unit 3 next to data augmentation.
- Live CNN training in the browser: feasible for `cnnviewer`'s size (estimate ~2 s per epoch on 2000 images in TS, in a Web Worker), but nondeterministic timing and battery cost; recommendation is precomputed epoch snapshots, which show the same thing reproducibly. Decide after a measurement.
- Real microscopy for `unetseg`: a small annotated nuclei set with a clear licence (the same search as the thresholding and watershed plans) would be more convincing than synthetic images; would need a separate weight set and honest Dice on it.
- **Decided (course map):** this day comes before the VAEs (day 15 vs 17); `neuralnet/` is unit 1 here, and the VAE day keeps a 2–3 slide recall in `autoencoder/`. Whether `nnfit` is also re-embedded in the recall is the VAE plan's choice.
- **Decided (course map):** transformers get one pointer slide here; the content belongs to day 18.
- **Decided (conventions):** $W^{(l)}$ here with the bridge line "$A = W^{(1)}$, $C = W^{(2)}$" in `neuralnet/`; activation $\varphi$, never $\sigma$.
- Whether `deep-practice/` stays optional: shortcut learning and domain shift are now carried by days 16 and 18, so optional is enough.

## References

- Rosenblatt (1958), *The perceptron*, Psychological Review 65; Minsky & Papert (1969), *Perceptrons* (XOR).
- Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*, Nature 323.
- Cybenko (1989), *Approximation by superpositions of a sigmoidal function*, Math. Control Signals Systems 2; Hornik (1991), Neural Networks 4.
- LeCun, Bottou, Bengio & Haffner (1998), *Gradient-based learning applied to document recognition*, Proc. IEEE (LeNet, MNIST).
- Zeiler & Fergus (2014), *Visualizing and understanding convolutional networks*, ECCV (feature hierarchy, occlusion).
- Srivastava et al. (2014), *Dropout*, JMLR 15; Kingma & Ba (2015), *Adam*, ICLR; Ioffe & Szegedy (2015), *Batch normalization*, ICML; He et al. (2016), *Deep residual learning*, CVPR.
- Long, Shelhamer & Darrell (2015), *Fully convolutional networks for semantic segmentation*, CVPR; Ronneberger, Fischer & Brox (2015), *U-Net*, MICCAI; Milletari, Navab & Ahmadi (2016), *V-Net* (Dice loss), 3DV.
- Geirhos et al. (2020), *Shortcut learning in deep neural networks*, Nature Machine Intelligence 2.
- Dosovitskiy et al. (2021), *An image is worth 16×16 words*, ICLR (vision transformer, pointer only).
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press (chapters 6–9); Nielsen, *Neural Networks and Deep Learning* (online book; teacher reading).
- Teacher note only: the 2D classification playground follows the idea of the well-known browser "neural network playground"; ours adds per-unit region thumbnails, by-hand mode and validation curves.
