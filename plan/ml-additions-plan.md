# Plan: more machine learning (training, GANs and flows, self-supervision, fine-tuning)

**Status: built 2026-11-17** (one worker per day, `plan/ml-build-brief.md`). Units without a
live demo of their own: `self-distillation` (figures; reuses the attention demo). Demos that
replay recorded runs instead of training live: `augmentation` (train view), `training-curves`,
`flows` (stepped weights).

Requested 2026-11-17: add **GANs**, **self-supervised and contrastive pretraining**,
**fine-tuning and transfer learning**, **training practicalities**, and **normalising flows**.
This plan replaces the "Fine-tuning strategies" entry under Future work in `coursemap.md`.

Four new days in Part VII ("Learning from examples"), each with the usual units (lecture,
quiz, exercise with pen-and-paper questions, an interactive demo, and a `## Code` section with a
module in the day's Rust crate). Same principles as every day: concept layer first, math layer
marked, about half the slides with a picture, demos that train intuition, no tool names.

## Where they go

Current Part VII order, with the new days marked **new**:

| | Day | Why here |
|---|---|---|
| | Random forests, SVMs, Gaussian mixtures, Neural networks | (unchanged) |
| **new** | **A. Training networks in practice** | right after the first networks are trained (day "Neural networks"); everything later trains networks |
| | Evaluation and statistics, Variational autoencoders | (unchanged) |
| **new** | **B. GANs and normalising flows** | the other two classic generative families, right after the VAE; before diffusion, which is then compared with all three |
| | Modern deep learning | (unchanged; attention and transformers are needed for C) |
| **new** | **C. Learning without labels** | needs transformers and the "pretrained features + probe" idea of Modern deep learning; feeds the cell networks and fine-tuning |
| | Networks for cells and objects | (unchanged) |
| **new** | **D. Fine-tuning and transfer learning** | adapts the pretrained models of the cell-networks day (Cellpose, SAM, YOLO) |
| | Diffusion models | (unchanged; gets a short comparison of the four generative families) |

## A. Training networks in practice

What every network on the later days relies on, and what goes wrong when training.

1. **Optimisers and learning rates.** Gradient descent on a valley: too small, too large,
   oscillation. Momentum, then Adam (per-parameter step sizes from running averages of the
   gradient and its square). Mini-batches and noisy gradients. Learning-rate schedules (warm-up,
   step and cosine decay). *Math:* the update rules; why Adam's step is roughly scale-free; bias
   correction. *Demo:* an optimiser race on 2D loss surfaces (a long valley, a saddle, noisy
   gradients): plain GD, momentum and Adam side by side, with the learning rate as a slider.
2. **Initialisation and normalisation.** Signals that shrink or explode through many layers;
   vanishing gradients; sensible initial weights (variance ∝ 1/fan-in). Batch normalisation and
   layer normalisation: what they compute, why training gets easier, and the batch-statistics
   trap at test time (one image, other microscope). Residual connections and ResNet, properly (added 2026-11-17: not explained anywhere before): the degradation problem, the residual block y = x + F(x), why gradients pass through many blocks, the ResNet architecture as the backbone of later networks, and the difference from U-Net's skip connections; demo: a deep plain stack vs the same stack with residual connections. *Demo:*
   activations layer by layer in a deep stack, with and without normalisation and with different
   initial scales.
3. **Regularisation and data augmentation.** Overfitting on small microscopy data sets; weight
   decay, dropout, early stopping (recall the evaluation day). Augmentation as "telling the
   network what does not matter": flips, rotations, elastic deformation, intensity and noise,
   blur; which are safe for which task (never flip if handedness matters; scale and pixel size).
   *Demo:* an augmentation explorer on a micrograph plus a small network trained with and without
   augmentation (train and validation curves).
4. **Reading training curves.** Under- and overfitting, learning rate too high or low, a bug
   (loss flat at ln K), class imbalance (weighted loss, focal loss in passing), and a checklist
   (overfit one batch first, check the inputs, fix the seeds, keep a validation set). *Demo:*
   a gallery of training curves to diagnose, with a tiny network training live.

*Code:* momentum and Adam steps; a 2D loss-surface race; batch norm forward pass; augmentations
(flip, rotate, elastic) on `Image`; a small MLP trained with and without augmentation and
weight decay, printing the curves.

## B. GANs and normalising flows

1. **Generative adversarial networks.** A generator turns noise into images; a discriminator
   tells real from generated; the two-player game; at the optimum the generated distribution
   equals the data. *Math:* the minimax objective and the optimal discriminator; the
   non-saturating loss. *Demo:* a 1D or 2D GAN training live (generator samples vs data
   histogram, discriminator output as a curve), with buttons for the learning rates.
2. **Training GANs and what goes wrong.** Mode collapse, oscillation, vanishing discriminator
   gradients; the Wasserstein idea in one picture; evaluation (sample quality vs diversity,
   recall the precision/recall of samples from the diffusion day, introduced here first).
   *Demo:* the same toy with an 8-mode ring: watch mode collapse, then fix it.
3. **Image-to-image translation and its risks.** Conditional GANs (pix2pix: paired), cycle
   consistency (unpaired), applications in microscopy: virtual staining, label-free to
   fluorescence, denoising and super-resolution; **hallucination**: plausible detail that is not
   in the sample, and how to check (held-out ground truth, uncertainty, never measure on
   generated images without validation). Links to the diffusion day ("generated detail").
4. **Normalising flows.** An invertible network maps data to a simple Gaussian and back; the
   change-of-variables formula gives the **exact likelihood** (the VAE only bounds it);
   coupling layers make the inverse and the determinant cheap. Uses: density estimation,
   anomaly detection, sampling. *Math:* change of variables and the log-determinant; affine
   coupling. *Demo:* a 2D flow bending a Gaussian into a two-moons distribution, layer by layer,
   with the density shown.

*Code:* a 1D GAN with hand-written gradients; mode collapse on a ring; an affine coupling layer
and its inverse; the change-of-variables density checked against a histogram.

## C. Learning without labels (self-supervised learning)

1. **Pretext tasks and masked image modelling.** Learn features from unlabelled images by
   solving made-up tasks: predict rotations, fill in masked patches (masked autoencoders);
   what the network must learn to solve them. Linear probe and nearest-neighbour evaluation
   of the features (recall the attention unit). *Demo:* mask patches of a micrograph and
   compare reconstructions from a PCA model and a small learned model.
2. **Contrastive learning.** Two augmented views of the same image should map close together,
   views of different images far apart; the InfoNCE loss as "pick the matching view among many";
   the augmentations **define** what counts as the same (cell cycle vs staining intensity);
   collapse and why negatives (or other tricks) prevent it. *Math:* InfoNCE and its link to
   softmax classification; the temperature. *Demo:* 2D embeddings of augmented views being
   pulled together and pushed apart, with the temperature and the augmentation strength.
3. **Self-distillation and foundation models.** Learning without negatives (student and teacher
   networks, the teacher a moving average); why the features of large self-supervised models
   transfer to microscopy; image–text models in one slide. Using frozen foundation features:
   k-NN, clustering of cell phenotypes, linear probes. *Demo:* a feature-space viewer of cell
   crops (precomputed embeddings), coloured by a label the network never saw.
4. **Self-supervised denoising.** Noise2Noise (two noisy images of the same thing are enough)
   and Noise2Void (one noisy image, blind-spot masking, assumes pixel-independent noise);
   when it fails (correlated noise, structured artefacts). Links to the denoising unit
   (day "Deconvolution") and the diffusion denoiser. *Demo:* blind-spot training on a small
   patch-based linear denoiser, compared with the classical filters of the denoising unit.

*Code:* InfoNCE loss and its gradient on small embeddings; a contrastive toy with augmentations;
a blind-spot (Noise2Void-style) linear denoiser trained on one noisy image and compared with
non-local means.

## D. Fine-tuning and transfer learning

(The queued scope from `coursemap.md`, organised into units.)

1. **Transfer learning.** What a pretrained network has learnt layer by layer (edges → textures
   → parts); when to use a model as is, fine-tune it, or train from scratch; the frozen
   backbone with a small head (linear probe) vs full fine-tuning; how much data each needs.
   *Demo:* train/validation curves for "frozen", "partial" and "full" fine-tuning of a small
   network on a new image domain.
2. **Fine-tuning strategies.** Freezing early layers, discriminative (layer-wise) learning
   rates, warm-up; catastrophic forgetting; parameter-efficient methods: adapters and low-rank
   updates (LoRA: a weight change ΔW = BA with small rank), tuning only a decoder or a prompt
   (e.g. SAM's mask decoder). *Math:* parameter counts of full vs low-rank updates; why a
   low-rank update is enough when the change is small. *Demo:* a low-rank update visualiser
   (a weight matrix, its change, the rank as a slider, the error of the approximation).
3. **Few labels: annotation strategy and the human in the loop.** Learning curves (accuracy vs
   number of annotated objects); sparse annotation; human-in-the-loop training (annotate the
   model's mistakes, retrain, repeat); active learning (which images to label next: uncertainty,
   diversity). *Demo:* a learning-curve and active-learning simulator.
4. **Domain shift and adaptation.** A model trained on one microscope, stain or cell line fails
   on another: why (input statistics, normalisation layers, recall unit A2); detection of the
   shift; remedies: intensity normalisation, augmentation, re-estimating normalisation
   statistics, fine-tuning on a few target images; evaluation on held-out experiments (recall
   the leakage unit). *Demo:* a classifier on two "microscopes" with a shift slider.

*Code:* a linear probe on frozen features vs full training of a small MLP; layer freezing in a
two-layer net; a low-rank update ΔW = BA fitted to a target change; an uncertainty-based
active-learning loop on a toy data set.

## E. Drifting (a new unit on the Diffusion models day)

Two recent papers (read 2026-11-17):

- **Generative Modeling via Drifting** (Deng, Li, Li, He, Du; arXiv 2602.04770, Feb 2026). A
  one-step generator f(ε) is trained so that its samples follow a **drifting field**
  V(x) = V⁺(x) − V⁻(x): a kernel-weighted pull towards real samples minus a push away from the
  other generated samples (kernel exp(−‖x − y‖/τ), normalised like a softmax, in the feature
  space of a frozen self-supervised encoder). Loss: ‖f(ε) − stopgrad(f(ε) + V)‖², whose gradient
  is −2V, so each training step moves the samples along the field. V = 0 when generated and real
  distributions agree. The iteration that diffusion does at sampling time is moved into
  training; sampling is one forward pass (reported FID 1.54 on ImageNet 256).
- **Dataset Distillation via Drifting** (Cazenavette, Quan, Daras, Torralba, Sitzmann; arXiv
  2611.16642, NeurIPS 2026). The same field applied directly to a handful of synthetic images per
  class (no generator): they drift until they cover the real class, with many augmented views
  per image as extra points; no classifier in the loop; a coverage metric (fraction of real
  points whose k-NN ball contains a synthetic point).

**Is it new?** A new framing built from known parts: mean shift (the attraction), kernel moment
matching (MMD is a special case, the paper shows), the positives/negatives softmax of contrastive
learning, and stop-gradient regression targets. Not adversarial (no discriminator), no
likelihood, no ODE/SDE at sampling. Worth one unit as a recent example of how the field moves,
and because it ties several units together. No microscopy results in either paper: teach it as
an idea, not as a tool.

**Where:** the last unit of the **Diffusion models** day (after diffusion-inverse), folder
`drifting`: by then students know mean shift (Gaussian mixtures day), contrastive learning and
frozen self-supervised features (C), GANs and flows (B), and diffusion samplers.

**Unit content:** (1) particles first: N points drift from noise to a 2D target under V⁺ − V⁻;
the field vanishes when the cloud matches; temperature τ (small: clumps, large: blurred
average); (2) this is dataset distillation: the points are the synthetic data set, coverage as
a measure, augmentations as extra points; (3) a generator instead of free points: the
stop-gradient loss, gradient = −2V, one-step sampling; (4) comparison with GANs (the push away
from generated samples plays the discriminator's role, without a game), diffusion (iteration at
training time instead of sampling time) and MMD. *Math:* the field, the loss and its gradient,
the MMD connection, why V = 0 does not prove p = q in general (the paper's own caveat).
*Demo:* 2D particles drifting to two moons / a ring, with τ, N and augmentation jitter as
sliders, ‖V‖ and coverage plotted; a second panel trains a small MLP generator with the same
loss. *Code:* the field V for a set of points, the drifting loop with the coverage metric, and
the generator update in 1D/2D.

The "Four ways to generate" comparison slide (below) becomes five: VAE, GAN, flow, diffusion,
drifting.

## Changes to existing units

- **neuralnet / backprop (day "Neural networks"):** point forward to A for optimisers,
  normalisation and augmentation; keep momentum there as a first look.
- **vae-gen:** the GAN mention links to B1; plus a slide "The autoencoder as a tokenizer"
  (added 2026-11-17): a trained (V)AE reused as the compression stage of other models (latent
  diffusion, drifting, VQ codebooks giving discrete tokens for transformers and image–text
  generators); the compression and the decoder's habits are shared by everything on top.
- **attention:** the frozen-features slide links to C and D.
- **detection** (added 2026-11-17): feature pyramid networks (top-down path with lateral
  connections, predictions on every level; linked to the image and Laplacian pyramids of
  scale-space and to U-Net's skips), with a quiz and an exercise question; **yolo** says its
  "neck" is a feature pyramid.
- **denoise (day "Deconvolution"):** links to C4.
- **cellnets-compare:** links to D (adapting the networks to your data).
- **Diffusion models day:** a short slide "Five ways to generate" (VAE, GAN, flow, diffusion, drifting:
  how each is trained, likelihood or not, sample quality, speed), with links back.
- **coursemap.md owners table:** optimisers, normalisation layers, augmentation → A;
  GANs, flows, image translation and hallucination → B; self-supervised and contrastive
  learning, foundation features, self-supervised denoising → C; transfer learning,
  fine-tuning, LoRA, active learning, domain adaptation → D.
- **days.yml:** four new days with hard prerequisites (A: Neural networks; B: VAE and A;
  C: Modern deep learning and A; D: Networks for cells and C), soft links to Evaluation.

## Day numbers and the code crates

Inserting days renumbers every later day (the sidebar says "Day 37" etc.), and the Rust crates
were named by number (`day37`).

**Decided (2026-11-17): name the crates after the day's short name (its id in `days.yml`)**,
not its number: package and folder `code/<day id>/`, e.g. `cargo run -p training -- adam`,
`cargo test -p edges canny::`. New days then never rename existing crates or pages. Done first,
before the new days: a one-off script renames the 51 crates (`git mv code/day02-edges
code/edges`, package `day02` → `edges`), the commands and paths on all pages and in the code
comments, `code.qmd`, `code/README.md`, `tools/code_units.py` and `plan/code-plan.md`; then the
usual checks (all programs, tests, clippy) and the output comparison against a baseline.
Watch for day ids that clash with a unit module inside them (day `diffusion` contains a unit
`diffusion`? no: the unit `diffusion` is in day `moderndl`; check all) and with crates.io names
(irrelevant for path dependencies, but avoid `image`).

## Demos: keep them small (owner's note, 2026-11-17)

Demos for these units may be too hard to do well. Keep them small (2D toys, tiny networks,
precomputed results stepped through), and leave a unit without a live demo (figures instead)
where it would be fragile or huge. The course owner is writing **Rust ML packages** that later
demos can use (via WebAssembly, like the cell networks); units without a demo record here what
a demo should show, for then.

## How to build it

As before: one worker per day (four in parallel), each following this plan, the design
principles, and `plan/code-plan.md`; demos in TypeScript (small networks train in the browser;
no Wasm needed); figures from scripts in `tools/figs/`. Builds share the one `code/target`
folder; the build folder is cleaned at the end. Then integration: `make sidebar`,
`make check-prereqs`, `make typecheck`, `make test`, `make code-test`, full render with 0
warnings, `make check` (links, console), `make check-fit` for the new demos, commit.

Estimated size: 17 units (16 in four new days, 1 on the Diffusion models day), 17 demos, 17 code modules.
