# Plan: a day on energy-based methods: the Ising model and Markov random fields

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 28 of 51, Part VI "Segmentation as optimisation". Hard prerequisites: Graph cuts (day 27). Soft (see also): Bit depth and noise (day 3). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part VI (snakes → level sets → dynamic programming; Ising/MRF); entry after day 27 (graph cuts). Independent of the snakes chain; it is the probabilistic reading of day 27.

## Goal of the day

After the day a student can:

1. write the **Ising energy** $E(\mathbf{s}) = -J\sum_{(p,q)} s_p s_q - h\sum_p s_p$ of a small lattice, turn it into probabilities with the **Gibbs distribution** $p(\mathbf{s}) = e^{-E(\mathbf{s})/T}/Z$, compute $Z$ for a 2×2 lattice by hand, and explain why $Z$ cannot be computed for an image ($2^n$ states);
2. explain what **temperature** does (low $T$: only the lowest energies matter; high $T$: everything is equally likely), and describe the **phase transition** of the 2D Ising model from a simulation;
3. run a **Gibbs sampler** and the **Metropolis** rule by hand on one pixel, and explain at an intuitive level why they sample the right distribution (detailed balance; the stationary distribution as an eigenvector with eigenvalue 1);
4. set up **Bayesian image denoising**: likelihood (noise model) + Ising prior → posterior energy; distinguish the **MAP** estimate (lowest energy) from the **MPM** estimate (posterior marginals); compare ICM, simulated annealing, mean field and the exact MAP, and recognise local minima and a badly chosen prior strength;
5. connect discrete MRFs to **continuous** ones: a Gaussian MRF prior is quadratic smoothing, i.e. a sparse linear system with the graph Laplacian; total variation is the edge-preserving alternative.

Why it fits this audience: everything is sums, exponentials and logarithms of small
examples; the probability needed is Bayes' rule and the normal distribution. The links to
linear algebra are explicit: the transfer matrix (unit 1), the Markov chain's stationary
vector (unit 2), the Laplacian system (unit 4).

**Split with the graph-cut day.** The graph-cut day owns min-cut / max-flow and the
construction of the graph; this day owns the probabilistic and statistical-physics view
(Gibbs distribution, temperature, MAP vs marginals, sampling) and the iterative/stochastic
optimisers. Where the two meet, this day *states* that the binary Ising posterior has an
exact MAP by min-cut (Greig, Porteous & Seheult 1989) and *shows* the result from
`lib/maxflow.ts` as a reference line, without teaching the algorithm. Graph cuts (day
27) always come first, so every graph-cut reference here is a back-reference.

## Notation (course conventions, `plan/coursemap.md`)

- Spins $s_p \in \{-1, +1\}$ in units 1–3; labels $L_p \in \{0, 1\}$ (or $\{1..K\}$) where we talk about segmentation. With $s = 2L - 1$: $-J s_p s_q = 2J\,[L_p \ne L_q] - J$, so the Ising coupling is the segmentation weight $\lambda = 2J$ up to a constant, and the field $h$ is a data term. One slide in unit 1 makes this translation as a back-reference to `seg-energy/` (day 27), which owns the energy and λ; the Chan–Vese length term (level-set day) is the same λ.
- Temperature $T$ (with $k_B = 1$). Never $\beta$ for the prior strength or inverse temperature (course convention: β only for snake rigidity and the β-VAE): $J$ on the physics slides of units 1–2, $\lambda$ (= 2J) for the prior weight from unit 3 on and for continuous priors; say once that the literature uses $\beta$ for both.
- Noise standard deviation $\sigma_n$ (defined in `cameranoise/`, day 3); $\sigma$ alone is a Gaussian filter scale.

## Prerequisite gap

- **No statistical physics.** One slide of story (a magnet: each atom is a small arrow that wants to agree with its neighbours; heat shakes them) and then only the formulas. Boltzmann/Gibbs is introduced as a *rule* turning energies into probabilities, motivated by "low energy = likely, and only differences of energy should matter".
- **Bayes' rule**: the two-class Bayes rule is known from day 12 (thresholding, `otsu/`) and the MAP view of an energy from day 27 (`seg-energy/` "probabilistic view" math slide); unit 3 recalls both in one slide (negative log turns products into sums) and does not re-derive them.
- **Noise models**: Gaussian and Poisson noise, SNR/PSNR are owned by `cameranoise/` (day 3); unit 3 and 4 recall them in one line.
- **Markov chains**: not assumed. Unit 2 introduces them as "a random walk on states" with a 2-state example written as a 2×2 matrix; the stationary distribution is an eigenvector (their linear algebra).
- **Segmentation**: known from Part II (thresholding, day 12) and graph cuts (day 27). Units 1–4 are about denoising; unit 5 (Potts) opens with a one-slide back-reference to day 12 and `seg-energy/`.
- **Graph Laplacian**: owned by `ncut/` (day 27); unit 4 recalls it in two lines (on a path).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus two optional units.

### 1. `ising/`: Energy and probability: the Ising model

- Picture first: a noisy binary image and the same image cleaned; "neighbouring pixels usually agree" is a statement about *probability*. An energy that counts disagreements, and a rule that turns energy into probability.
- The Ising model: spins $\pm1$ on a grid, coupling $J$ (agree with neighbours), field $h$ (prefer $+1$). $J > 0$: ferromagnet; $J < 0$: antiferromagnet (checkerboard), shown once.
- **Gibbs/Boltzmann distribution** $p(\mathbf{s}) = e^{-E(\mathbf{s})/T}/Z$. Why the exponential: adding a constant to $E$ changes nothing; independent parts multiply when energies add. Ratios $p(\mathbf{a})/p(\mathbf{b}) = e^{-(E_a - E_b)/T}$ need no $Z$ (this sentence is the key to unit 2).
- **Temperature** as a knob: $T \to 0$ puts all mass on the lowest-energy states, $T \to \infty$ makes all states equally likely. Energy minimisation = the $T \to 0$ limit of probability.
- **Partition function** $Z = \sum_{\mathbf{s}} e^{-E(\mathbf{s})/T}$. The 2×2 lattice (4 edges, $J = 1$, $h = 0$) by hand: 2 aligned states ($E = -4$), 12 states with $E = 0$, 2 checkerboards ($E = +4$): $Z = 2e^{4/T} + 12 + 2e^{-4/T}$. At $T = 1$ the two aligned states carry about 90 % of the probability. A 256×256 image has $2^{65536}$ states: hopeless.
- Where $Z$ *is* tractable: a 1D chain. The **transfer matrix** $\begin{bmatrix} e^{(J+h)/T} & e^{-J/T} \\ e^{-J/T} & e^{(J-h)/T}\end{bmatrix}$ gives $Z = \operatorname{tr}(M^n)$ for a ring, i.e. $\lambda_1^n + \lambda_2^n$ from its eigenvalues. This is the same left-to-right sum as dynamic programming on `segchain` (graph-cut day / DP day), with "min" replaced by "sum". In 2D there is no such shortcut in general.
- Back-reference slide (day 27, `seg-energy/`): Ising energy = segmentation energy ($\lambda = 2J$, field = data term; see Notation).
- **Demo `isingenum`** (new, small): a 3×3 lattice (512 states; toggle 2×2 with 16) drawn large; click spins to set a state. Sliders $T$, $J$, $h$; toggle free / periodic boundary. Readouts for the current state: $E$, $e^{-E/T}$, $Z$, $p$. Right panel: bar chart of the probability of each energy level (degeneracy × weight, levels labelled with their count of states) and a second tab with the distribution of the magnetisation $m = \frac1n\sum s_p$. Presets: "T → 0: only the two ground states", "T → ∞: all 512 states equally likely", "a field breaks the tie", "J < 0: checkerboards win", "the 2×2 by-hand case". The *sample* mode is used in unit 2.
- By hand: $Z$ of the 2×2 lattice as above, $p$(all aligned) at $T = 1$ and $T = 4$, effect of $h = 0.5$ on the two ground states; translate a 1×4 Ising energy into the segmentation form; $Z$ of a 3-spin ring via the transfer matrix and check against the 8-state enumeration.

### 2. `mcmc/`: Sampling, temperature and the phase transition

- What a sample looks like is the best picture of a distribution, but we cannot normalise it. We can, however, compare two states that differ in one pixel (no $Z$ needed).
- **Local conditional**: with the neighbour sum $u_p = J\sum_{q \sim p} s_q + h$, $\;p(s_p = +1 \mid \text{neighbours}) = 1/(1 + e^{-2u_p/T})$: a logistic curve in the neighbours' vote. Markov property: only the neighbours matter.
- **Gibbs sampler** (heat bath): visit pixels, redraw each from its local conditional. **Metropolis**: propose a flip, accept with probability $\min(1, e^{-\Delta E/T})$; downhill moves always, uphill sometimes.
- Why it works, intuitively: a **Markov chain** on states; written for 2 states as a 2×2 transition matrix $P$, the long-run distribution satisfies $\boldsymbol\pi P = \boldsymbol\pi$ (eigenvalue 1). **Detailed balance** $\pi_a P_{ab} = \pi_b P_{ba}$: in equilibrium the traffic between any two states is equal in both directions; the Metropolis rule is built so that this holds for the Gibbs distribution. Burn-in and correlated samples in words; no convergence proofs.
- **Phase transition** of the 2D Ising model: below $T_c = 2/\ln(1 + \sqrt2) \approx 2.27$ (Onsager 1944) large domains form and the lattice picks a sign; above it, noise. At $T_c$, clusters of all sizes. Magnetisation vs $T$ with Onsager's exact curve. 1D has no transition (link back to the transfer matrix). Critical slowing down: local updates get slow near $T_c$ (cluster algorithms, Swendsen–Wang, named only).
- **Simulated annealing** (Kirkpatrick et al. 1983; Geman & Geman 1984): sample while lowering $T$; slowly enough and the chain ends in a ground state. Quench (sudden cooling) freezes domain walls: a local minimum. Logarithmic schedule (the guarantee, impractically slow) vs geometric (what people use).
- **Demo `isingenum`**, sample mode: run a Gibbs or Metropolis chain on the 3×3 lattice; the empirical histogram fills in over the exact bars, with a counter of steps. Presets: "converges", "low T: stuck in one ground state for a long time" (why mixing matters).
- **Demo `isinglattice`** (new, main demo): a 256×256 lattice on a canvas, periodic boundary, animated. Sliders $T$ (0.5–5, with $T_c$ marked), $h$, $J$; algorithm Metropolis / Gibbs; start random / all $+1$; play / one sweep / reset; *cool* button with a rate slider (anneal from the current $T$ to 0.5). Side panel: magnetisation and energy per spin vs sweep (scrolling plot), and a *T-scan* tab that sweeps $T$ downwards and plots $\langle |m| \rangle$ against $T$ with Onsager's curve overlaid. Presets: "hot: noise", "cold start from random: domains coarsen", "at $T_c$: clusters at all scales", "a small field decides the sign", "quench vs slow cooling", "critical slowing down".
- By hand: the conditional probability of the centre spin with neighbours $(+,+,+,-)$ at $T = 2$ (≈ 0.88); Metropolis acceptance of a flip with $\Delta E = 4$ at $T = 2$ and $T = 0.5$; a 2-state chain: write $P$ for Metropolis on energies $0$ and $1$, find its stationary vector, check detailed balance and that it equals the Gibbs distribution.

### 3. `mrf-denoise/`: Bayesian image denoising with an Ising prior

- Bayes recall (one slide, days 12 and 27), then the model: true binary image $\mathbf{x}$, observed $\mathbf{y}$ with each pixel flipped with probability $\varepsilon$ (and, as a second noise model, Gaussian noise on two grey levels: recalled from `cameranoise/`, day 3).
- Likelihood → field: $-\log p(\mathbf{y} \mid \mathbf{x}) = -\sum_p h\,x_p y_p + \text{const}$ with $h = \tfrac12\log\frac{1-\varepsilon}{\varepsilon}$ (for Gaussian noise $h_p$ is proportional to $y_p$). Posterior energy $E(\mathbf{x} \mid \mathbf{y}) = -J\sum_{(p,q)} x_p x_q - \sum_p h_p x_p$: an Ising model with a pixel-dependent field; in segmentation notation the prior weight is $\lambda = 2J$. $\lambda = 0$ is thresholding (as on day 27).
- **Estimates**: MAP (the single most probable image = lowest energy) vs **MPM** (for each pixel the more probable value under the posterior marginal; minimises the expected number of wrong pixels). Marginals need sampling at $T = 1$ (Gibbs sampler, average the samples) and also give an uncertainty map. Note: $T$ here is a knob of the *algorithm*; the model's own posterior is at $T = 1$.
- **Optimisers for the MAP**: ICM (Besag 1986: Gibbs sampler at $T = 0$, greedy, monotone in energy, stops in the nearest local minimum, depends on the visiting order); simulated annealing (unit 2); **mean field**: replace neighbours by their averages, $m_p = \tanh\big((J\sum_{q\sim p} m_q + h_p)/T\big)$, iterated: deterministic, smooth, an approximation. **Exact**: for binary labels and $J \ge 0$ the MAP is a minimum cut (Greig, Porteous & Seheult 1989), recalled from day 27 (`maxflow/`); here it appears as a reference energy.
- Lessons from the comparison: ICM is fast but stuck; annealing approaches the exact energy if cooled slowly; Greig et al. found that the exact MAP for large $J$ is *worse* than approximate solutions (it removes thin structures and rounds corners): the prior is too simple, not the optimiser. Choosing $J$: too small = noise remains, too large = blobs; $\varepsilon$ can be estimated from the noise, $J$ by trial, cross-checking on a synthetic image, or pseudo-likelihood (Besag 1974, named only).
- **Demo `mrfdenoise`** (new, main demo; also used in unit 5): a 128×128 binary image: synthetic (disc, square with sharp corners, a 1-px line, text) or the micrograph thresholded. Noise: flip probability $\varepsilon$ or Gaussian $\sigma_n$ (seeded). Slider $J$; method menu: threshold, ICM, Gibbs at $T = 1$ (MPM, with the marginal map shown in grey), simulated annealing ($T_0$, geometric rate), mean field, exact MAP (min-cut via `lib/graph.ts`/`lib/maxflow.ts`, computed once per parameter change). Play / one sweep / reset. Panels: noisy input, current estimate, error map vs truth; plot of energy vs sweep for each method run so far (log sweep axis) with the exact minimum as a dashed line; readouts: energy, % wrong pixels. Presets: "ICM: fast, stuck", "slow cooling reaches the min-cut energy", "quench", "MAP vs MPM", "J too large: corners and the line disappear", "J = 0 is thresholding", "the exact MAP is not the best image".
- By hand: a 1×3 image $\mathbf{y} = (+,-,+)$ with $\varepsilon = 0.2$ ($h = \tfrac12\log 4 \approx 0.69$), $J = 1$: posterior energies of all 8 states, the MAP, one ICM pass from $\mathbf{y}$, and the exact marginal of the middle pixel (MPM) compared with the MAP.

### 4. `gmrf/`: Continuous MRFs: quadratic smoothing and total variation

- From labels to grey values: $x_p \in \mathbb{R}$, Gaussian noise $y_p = x_p + n_p$ with standard deviation $\sigma_n$ (recall `cameranoise/`, day 3). Prior on differences. **Gaussian MRF**: $E(\mathbf{x}) = \tfrac1{2\sigma_n^2}\sum_p (x_p - y_p)^2 + \tfrac\lambda2\sum_{(p,q)}(x_p - x_q)^2$. The posterior is Gaussian, so MAP = MPM = mean.
- The smoothness sum is $\mathbf{x}^\top L\mathbf{x}$ with the **graph Laplacian** $L$ (recalled from `ncut/`, day 27; it is the image Laplacian $\nabla^2$ of `kernels/` written as a matrix). Setting the gradient to zero: $(I + \lambda\sigma_n^2 L)\,\mathbf{x} = \mathbf{y}$: a **sparse linear system** (tridiagonal in 1D). With periodic boundary it is diagonalised by the DFT: a low-pass filter $1/(1 + c(2 - 2\cos\omega))$ (link to *Fourier*). Hence: quadratic priors blur edges.
- Samples from the prior: Gaussian increments = a random walk; it believes in no edges.
- **Total variation** (Rudin, Osher & Fatemi 1992): $\lambda\sum|x_p - x_q|$. Edges cost their height, not its square, so a sharp step is as cheap as a ramp: edges survive, ramps become staircases, small plateaus shrink in contrast. Convex but not smooth; 1D solved exactly (Condat's direct algorithm, words only). Huber as the compromise. Prior samples: Laplace increments, mostly small steps and a few jumps.
- Bridge: Chan–Vese's length term is the TV of an indicator function (level-set day); the binary Ising energy is the discrete length; TV on labels relaxes graph-cut problems (pointer). This day owns TV; radon (algebraic reconstruction) and wavelets (shrinkage) recall it.
- Pointer (one line): the same quadratic (Tikhonov) and TV priors used for **deblurring** (data term with a PSF) are taught on the deconvolution day (`plan/deconvolution-day.md`); the PSF forward model and why inversion is ill-posed are on day 4 (image formation). Here the forward model is the identity (denoising only).
- **Demo `mrf1d`** (new): a 1D profile (synthetic: steps, a ramp, a narrow peak; or a line across the micrograph) with Gaussian noise (seeded). Prior menu: none, quadratic (GMRF), Huber ($\delta$ slider), TV; slider $\lambda$. Shows $\mathbf{y}$, truth and $\hat{\mathbf{x}}$; readouts: energy split (data + prior), error vs truth. Tabs: *prior samples* (five draws from the quadratic and the TV prior side by side), *the matrix* (the tridiagonal $I + \lambda\sigma_n^2 L$ as a grey image and the frequency response of the quadratic smoother). Presets: "quadratic blurs the step", "TV keeps the step", "TV staircases the ramp", "TV shrinks the narrow peak", "Huber in between", "λ = 0".
- By hand: 1×3, $\mathbf{y} = (0, 3, 0)$, $\lambda\sigma_n^2 = 1$: write $I + L$, solve ($\hat{\mathbf{x}} = (0.75, 1.5, 0.75)$); TV with data term $\tfrac12\sum(x_p - y_p)^2$ and $\lambda = 1$ on $\mathbf{y} = (0,0,3,3)$: the two plateaus move inward by $0.5$ and the step stays sharp ($\hat{\mathbf{x}} = (0.5, 0.5, 2.5, 2.5)$).

### 5. (optional, or core if the day has room) `potts/`: More labels and hidden MRFs: the Potts model on a real image

This unit also absorbs the GMM day's optional `hmrf-em/` (review-organisation §2.6/§2.9: one owner, one demo). The GMM day recalls it in one slide.

- Potts energy: $\lambda\sum[L_p \ne L_q]$ with $K$ labels ($\lambda = 2J$ as before); $K = 2$ is Ising. Data term $D_p(\ell)$ from $K$ Gaussian intensity classes (or a Gaussian mixture in colour/feature space, recalled from the GMM day if taken; one slide otherwise). Same samplers (redraw from $K$ local probabilities), ICM, annealing; MPM gives per-class probability maps.
- **Hidden MRF + EM-style re-estimation**: unknown class means and variances; alternate "segment with the Potts prior" and "re-estimate class parameters from the current (soft) labelling" (EM with mean-field or ICM E-step, in words; the math layer writes the update). With $\lambda = 0$ this is exactly the GMM's EM (or k-means for hard labels); the prior adds "neighbours agree". Failure modes: bad initial means, a class swallowed by the prior.
- Exact MAP is NP-hard for $K \ge 3$; α-expansion via graph cuts (day 27, optional `alpha-expansion/`, the optimiser only) gives a near-optimal answer: pointer only.
- **Demo `mrfdenoise`** with `data-labels="3"`: the micrograph (`cell256.png` downsampled to 128×128) with three classes (membrane / cytoplasm / background), class means draggable on the histogram, $J$ slider, methods ICM / Gibbs (marginals) / annealing; toggle "re-estimate class parameters" (the hidden-MRF EM loop, with a plot of the class means per iteration). Presets: "λ = 0 is k-means thresholding", "smooth classes", "too smooth: membranes vanish", "bad initial means", "EM with neighbours recovers the means".
- By hand: a 1×3 image with three labels and given data costs: energy of all labellings that differ from the thresholded one in one pixel; ICM step; one re-estimation of the class means from the resulting labelling.

### 6. (optional / further reading) `boltzmann/`: Ising models that learn, and texture by sampling

- **Hopfield network** (1982; link to the neural networks day, 14): store patterns in $J_{pq}$ by the Hebb rule; recall = ICM from a corrupted pattern. **Boltzmann machine** (Ackley, Hinton & Sejnowski 1985): the same at $T > 0$ with learned couplings; the ancestor of energy-based models in machine learning.
- **Texture**: anisotropic or longer-range couplings give stripes and blobs when sampled (Cross & Jain 1983); a texture is a stationary distribution.
- **Demo**: reuse `isinglattice` with `data-couplings="..."` (a small coupling stencil, e.g. horizontal vs vertical $J$, next-nearest neighbours) for texture presets; a small `hopfield` demo (10×10 patterns, store 3, corrupt, recall) only if this unit is kept.

## Shared engineering

- `demos/src/lib/mrf.ts` (new): lattice of spins/labels as `Int8Array`/`Uint8Array` with width, height, 4-neighbourhood and free/periodic boundary; energy and its split; local field; Gibbs and Metropolis sweeps (checkerboard order for the large lattice, random order optional) with a precomputed table of acceptance probabilities per $(\Delta E, T)$; ICM; mean field; annealing schedules (geometric, logarithmic); the $K$-label (Potts) generalisation with arbitrary unary costs; exact enumeration for $n \le 16$ ($Z$, level probabilities, marginals); observables (magnetisation, energy per spin, Onsager's curve). Used by `isingenum`, `isinglattice`, `mrfdenoise`.
- Exact MAP: reuse `solveSeg`/`gaussCosts` in `lib/graph.ts` and `FlowGraph` in `lib/maxflow.ts` (owner: graph-cut day). A converter from an Ising posterior ($J$, $h_p$) to a `SegProblem` in `mrf.ts`; do not duplicate max-flow.
- `demos/src/lib/smooth.ts` (new; owner: this day; no `smooth1d.ts`): tridiagonal (Thomas) solver for the GMRF, Condat's direct 1D TV algorithm, Huber by iteratively reweighted least squares, sampling from the 1D quadratic/TV priors, and a 2D TV routine (used by radon's algebraic reconstruction and the wavelets comparison). Chain Viterbi and forward–backward (exact chain marginals) live in `lib/dp.ts` (owner: dynamic programming day), not here.
- Plot helper: `lib/plot.ts` (owner: morphology day) for energy-vs-sweep, magnetisation traces, histograms; one implementation.
- Image: `demos/src/fourier2d/cell256.png` downsampled to 128×128 (credit R. M. Hunt, CC BY 3.0, as in the other plans), via `loadGrey()` in `lib/imageops.ts`. Synthetic binary test images from `lib/synth.ts` (fixed seed).
- **Random numbers**: Monte Carlo draws millions of numbers per second; use `rng`/`gauss` from `lib/random.ts` (owner: bit-depth day, `cameranoise/`; `ui.ts` re-exports `rng`) and check it is fast and of adequate quality (e.g. a 32-bit xorshift/mulberry variant is fine for teaching); all demos seed it, so recordings reproduce.
- **Performance.** A Metropolis sweep of 256×256 with a lookup table is expected at ~1 ms in TypeScript, so several sweeps per frame fit; the *T-scan* (many sweeps per temperature) runs incrementally across frames. Measure on a teaching laptop; if a live sweep step exceeds ~30 ms, first reduce to 128×128, then move only the sweep kernel to Rust/Wasm (flat `Int8Array` in, same RNG algorithm) with the TS version kept as reference. Min-cut on 128×128 runs once per parameter change (tens of ms per the graph-cut plan). Animations pause when the slide is not visible.
- Tests (`demos/test/ising-test.ts`, run by the shared `make test`): enumeration gives the hand values for 2×2 ($Z$ above); Gibbs and Metropolis histograms on 3×3 match the exact distribution within a tolerance after a fixed number of steps; ICM energy is non-increasing; the min-cut energy equals brute force on random 3×3 posteriors and is ≤ the ICM/annealing energies; mean field reaches a fixed point; the tridiagonal solver and TV match the by-hand examples; transfer-matrix $Z$ equals enumeration for small rings.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits 1150×560 on a slide; static figures as fallback.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, 2×2 partition function by hand, `isingenum` |
| 10:45–12:15 | Unit 2 lecture, one-pixel Gibbs/Metropolis by hand, `isingenum` sampling, `isinglattice` |
| 13:15–14:45 | Unit 3 lecture, 1×3 posterior by hand, `mrfdenoise` |
| 15:00–16:30 | Unit 4 lecture + exercise (`mrf1d`); quizzes as wrap-up |

## Open questions

- If the day is short, merge the phase transition into unit 1 as a demo-only slide and drop mean field.
- review-gaps (over-weighted topics) suggests making `mcmc/` optional (MCMC/phase transition is more physics than image analysis) and keeping `mrf-denoise/` + `gmrf/` + `potts/` as core. The course map does not decide this (the whole day is a day); open.
- Unit 4 (continuous MRFs) or unit 5 (Potts on a real image) as the fourth core unit? Unit 4 has the stronger linear-algebra and cross-day links; unit 5 has the real image. Recommendation: unit 4 core, unit 5 optional, with the micrograph still present in unit 3 (thresholded binary image).
- **Decided (course map conventions):** $\lambda$ is the prior weight everywhere; $J$ only on physics slides with $\lambda = 2J$; never $\beta$.
- **Decided:** the dynamic programming day exists and owns chains (`lib/dp.ts`); forward–backward marginals (exact MPM in 1D) could be a `segchain` mode there. Here: mention only.
- Is the thresholded TEM micrograph a convincing binary test image, or should a clearly two-phase image (fluorescence nuclei, document scan) be added, with a clear licence?
- Optional unit 6: keep or leave as "Where this leads"? With Part VII (days 33–38) in the course, Hopfield/Boltzmann is no longer the only machine-learning link; it remains the energy-based-model bridge to the neural networks day (36) and the VAE day (38).
- Programming language for the code parts is still open (see README).

## References

- Ising (1925), *Beitrag zur Theorie des Ferromagnetismus*, Z. Physik; Onsager (1944), *Crystal statistics I*, Phys. Rev. (exact $T_c$ and magnetisation).
- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953), *Equation of state calculations by fast computing machines*, J. Chem. Phys.; Hastings (1970), Biometrika.
- Kirkpatrick, Gelatt & Vecchi (1983), *Optimization by simulated annealing*, Science.
- Geman & Geman (1984), *Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images*, PAMI.
- Besag (1974), *Spatial interaction and the statistical analysis of lattice systems*, JRSS B; Besag (1986), *On the statistical analysis of dirty pictures*, JRSS B (ICM).
- Marroquin, Mitter & Poggio (1987), *Probabilistic solution of ill-posed problems in computational vision*, JASA (MPM).
- Greig, Porteous & Seheult (1989), *Exact maximum a posteriori estimation for binary images*, JRSS B.
- Rudin, Osher & Fatemi (1992), *Nonlinear total variation based noise removal algorithms*, Physica D; Condat (2013), *A direct algorithm for 1-D total variation denoising*, IEEE SPL.
- Swendsen & Wang (1987), *Nonuniversal critical dynamics in Monte Carlo simulations*, PRL.
- Hopfield (1982), PNAS; Ackley, Hinton & Sejnowski (1985), *A learning algorithm for Boltzmann machines*, Cognitive Science; Cross & Jain (1983), *Markov random field texture models*, PAMI.
- Textbooks: Li, *Markov Random Field Modeling in Image Analysis* (Springer); Blake, Kohli & Rother (eds., 2011), *Markov Random Fields for Vision and Image Processing*; MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, ch. 29–31 (free online); Rue & Held (2005), *Gaussian Markov Random Fields*.
