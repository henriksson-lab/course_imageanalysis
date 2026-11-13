# Plan: a day on tracking

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-13.
**Course map (2026-11-13, self-study restructure):** day 26 of 51, Part V "Aligning images and motion". Hard prerequisites: Labelling (day 14), Point detection (day 20). Soft (see also): FFT registration (day 23), Measurements (day 16), Evaluation and statistics (day 37), Gaussian mixtures (day 35). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part V (with the planned optical-flow day and
non-rigid registration). **Entry after:** day 14 (Labelling: objects and their centroids) and the
point detection day (spots and sub-pixel positions). FFT registration (day 23, drift) and
evaluation (day 37, matching) help but are not required: what this day needs from them is
recalled in one slide each (see prerequisite gap). Inspiration for the methods: the user's
tracking module and tracking suite (teacher notes only; no tool names in student material).

## Goal of the day

After the day a student can:

1. turn a list of **detections per frame** into **tracks** by linking frame to frame; explain
   why **nearest neighbour** fails (conflicts, crossings) and why a **global assignment** with
   costs does better; solve a 2×2 or 3×3 linking problem by hand, including **gating** and the
   costs of a track **ending** or **starting** (birth/death);
2. predict how hard a tracking problem is from one number, the **displacement per frame
   compared with the spacing between objects**, and estimate the chance of a wrong link;
3. explain **prediction** with a **motion model** (random walk, constant velocity) and run a
   1D **Kalman filter** by hand: predict (the estimate moves, the uncertainty grows), update
   (a weighted average, weights from the two variances, the **gain**); use the predicted
   uncertainty as a **gate**;
4. handle what breaks tracks: **missed detections** (gap closing), objects that **enter or
   leave** the field, **dividing** objects (splits) and objects that touch (merges); read and
   draw a **lineage tree**, and tell a real division from a segmentation error;
5. **evaluate** tracks: link precision/recall, division scores, and why a small per-link error
   rate ruins long tracks;
6. **analyse** tracks: speed, straightness, the **mean squared displacement** (MSD); tell free
   diffusion, directed motion and confinement apart from the MSD curve; explain why a
   measured "speed" depends on the frame interval and how localisation error biases the MSD;
7. *(math)* write linking as a **linear assignment problem** with an augmented cost matrix;
   derive the Kalman update for one dimension from the product of two Gaussians and state the
   matrix form; derive $\text{MSD}(\tau) = 2dD\tau$ for a random walk, the directed and
   localisation-error terms, and the steady-state gain of a random-walk filter.

Why it fits this audience: it is geometry (distances), a small combinatorial optimisation
(assignment), Gaussians and their variances (Kalman), and a sum of independent steps (MSD).
All of it is within linear algebra, calculus and basic probability. It is also the day where
"objects" from Part II become objects **in time**, which is what most live-imaging biology
actually measures.

## Prerequisite gap

What earlier days teach (checked in the built units) and what this day recalls in one line:

- **Objects and centroids** (day 14: `label-connectivity/`, `regionprops/`, `label-measure/`):
  a label image → a table of objects with centroid, area, mean intensity. Unit 1 starts from
  "one table per frame". The **border rule** (`label-measure/`, "Border objects and the counting
  frame") is recalled in unit 3 for objects that enter or leave.
- **Point detection** (the point detection day, being planned/built in parallel): detections
  with sub-pixel positions, a **localisation error σ** and a detection score; missed and false
  detections. Unit 1 recalls "a detection is a position ± σ with a score"; unit 4 uses σ for the
  MSD offset. If point detection is skipped, `regionprops/` centroids are enough for units 1–3.
- **Linear assignment / Hungarian algorithm**: taught for **evaluation** on day 37
  (`object-eval/`: "Below 0.5: greedy can lose a match", math slide "matching as a linear
  assignment problem"). Day 37 comes **after** day 14, and a student can take this
  day before it. So unit 1 teaches assignment **with costs and alternatives** (the
  tracking form) on its own, with a back-reference to `object-eval/` for students who have seen
  it. Proposed ownership change (open question 1).
- **Gaussians and covariance**: the normal distribution and variance (audience.md); covariance
  ellipses from `pca/` (day 1). The **Mahalanobis distance** is taught in the GMM day
  (`gaussian-mixture/`, slide "Mahalanobis distance"); unit 2 gives it one concept slide of its
  own (the GMM day is not a prerequisite) and points there.
- **Drift** (day 23, `stitching/`, slides "Drift in a time-lapse" and "Try it: drift
  correction", demo `drift`): the whole field moves between frames; remove it **before**
  linking, or every gate is wrong. One recall slide in unit 1.
- **Nearest-neighbour distances and density** (day 16, `relations/` and demo `neighbours`:
  nearest-neighbour distance, Clark–Evans): unit 1's "displacement / spacing" ratio uses the
  mean nearest-neighbour distance $\approx 0.5/\sqrt{n}$ for random positions at density $n$.
- **IoU** (day 37 `object-eval/`, or Jaccard on day 12 `seg-eval/`): an optional cost term for
  segmented objects (overlap between frames); one line.
- **Maximum flow / graphs** (day 27, `maxflow/`): the math layer of unit 3 mentions that
  whole-movie linking is a min-cost flow; a pointer only.

## Overlaps / who owns what

This day **owns**: linking (nearest neighbour, greedy, global assignment with costs, gating,
birth/death alternatives), the **two-stage linking** (frame to frame, then gap closing/splits/
merges), motion models and the **Kalman filter**, track graphs and **lineage trees**, track
evaluation (link and division level), and track analysis (**MSD**, diffusion coefficient,
anomalous exponent, speed, straightness). Others recall:

| Day / plan | What stays there | What this day does |
|---|---|---|
| Evaluation (day 37, `object-eval/`) | Matching objects for **evaluation** (IoU, τ, F1, AP), the LAP math slide for IoU tables, `lib/eval.ts` `hungarian` | Teaches assignment with **costs and alternatives** for linking; reuses `hungarian`. `object-eval/` gets a one-line forward pointer ("the tracking day uses the same solver to link frames"). Track evaluation (unit 3) recalls F1 from `object-eval/` |
| Point detection (parallel) | Detection, localisation, sub-pixel fitting, **matching detections to ground truth by distance** for evaluation | Recalls σ and missed/false detections; **shares the gated distance assignment code** (`lib/linking.ts`, see shared engineering; open question 2) |
| FFT registration (day 23, `stitching/`) | Drift estimation and correction (`drift` demo) | One recall slide: correct drift first, then link |
| Optical flow (Motion group, not planned) | Dense motion fields, brightness constancy, Lucas–Kanade, **feature tracking** of corners (the `structure-tensor/` pointer "points to track between two video frames") | One slide in unit 1: "flow = motion of every pixel; tracking = identity of objects"; when flow helps tracking (as a prediction) |
| Edges (day 2, `structure-tensor/`) | Corners as trackable points | Nothing; the flow day takes the corner-tracking pointer |
| Labelling (day 14) | Labels, centroids, border rule | Uses them; unit 3 recalls the border rule |
| Measurements (day 16, `relations/`) | Nearest-neighbour distances, density, Clark–Evans | Uses the mean NN distance for the difficulty ratio |
| GMM day (`gaussian-mixture/`) | Mahalanobis distance and covariance ellipses in depth | One concept slide of its own (Mahalanobis as a gate) + pointer |
| Modern deep learning (day 39, `detection/`, `instances/`, `attention/`) | Learned detectors and instance segmentation; attention | Optional unit 6 `track-learned/` points back: learned appearance costs, learned association; "tracking by detection" |
| Graph cuts (day 27, `maxflow/`) | Max-flow/min-cut | Math pointer: whole-movie linking as min-cost flow |
| Modern deep learning `diffusion/` and the planned diffusion-models day | Diffusion **models** (generative) | **Naming clash:** unit 4's "diffusion" is Brownian motion. One sentence on the first MSD slide: "diffusion here means random motion of particles; the diffusion models of the deep learning day borrowed the name from it" |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units of about 1.5 h each including the exercise, plus two optional units. Order:
linking first (the core problem, and it already works for slow, sparse objects), then
prediction (what to do when objects move fast or cross), then the events that break tracks
(gaps, divisions, entries/exits) together with evaluation, then analysis (what the tracks are
for). Every lecture follows the math-layer markup of README.md; every exercise has concept
questions first, then `### Math questions` tagged *(math)*. No tool names in student material
beyond one-line pointers ("image analysis software calls this a LAP tracker").

### 1. `track-linking/`: From detections to tracks

Concept layer:

- **The problem** (picture): frames 1…T, each with a list of detections (dots). A **track** is
  a chain of detections, one per frame, that belong to the same object. Tracking = deciding
  which dot in frame t+1 continues which dot in frame t. Input recalled in one line from day 14 /
  the point detection day: "a detection is a position (± σ) with a few measurements".
- **Flow vs tracking** (one slide): optical flow gives the motion of every pixel; tracking keeps
  the **identity** of objects. Pointer to the optical-flow day of the Motion group.
- **Drift first** (recall, `stitching/`): if the stage moves, every object seems to move the
  same way; correct the drift before linking (the `drift` demo). Otherwise gates must be huge.
- **Nearest neighbour**: link each dot to the closest dot in the next frame. Works when objects
  are sparse and slow. Fails in three pictures: (a) **conflict**: two dots choose the same
  successor; (b) **crossing**: two objects pass each other; (c) a dot with no true successor
  (it left, or was missed) steals one.
- **One number says how hard it is**: the typical displacement per frame divided by the typical
  spacing between objects. Below ~0.2: easy, anything works. Around 0.5: nearest neighbour
  makes many errors, global assignment fewer. Near 1: no linking method can be reliable;
  image faster (shorter frame interval) or at lower density. Worked numbers: D = 0.1 µm²/s,
  Δt = 0.1 s → rms displacement $\sqrt{4D\Delta t}$ = 0.2 µm; one object per µm² → mean
  nearest-neighbour spacing ≈ 0.5 µm; ratio 0.4.
- **Costs**: a number per candidate link, small = likely. Default: **squared distance** (for
  random motion, the log-probability of a step is minus the squared distance over $4D\Delta t$,
  so squared distance is the natural cost). Extra terms for objects: change in area, in
  intensity, or 1 − overlap (IoU) for segmented objects that move less than their size. Terms
  must be weighted into one unit; the weights are choices.
- **Greedy vs global**: greedy takes the cheapest link first and never revisits. **Global
  assignment** minimises the **sum** of the costs of all links at once, each dot used at most
  once. Picture: A(0,0), B(3,0) → P(2,0), Q(5,0): greedy takes B–P (cost 1) and is left with A–Q
  (25): total 26; global takes A–P + B–Q: 4 + 4 = 8. Image analysis software calls the exact
  solver the Hungarian algorithm (or a "LAP" tracker); one line, back-reference to
  `object-eval/` for those who saw it.
- **Gating**: links longer than a maximum distance (or speed × elapsed time) are not allowed at
  all. A gate keeps the problem small (most pairs are impossible) and stops absurd links. Too
  small a gate breaks tracks of fast objects; too large lets wrong links compete.
- **Ending and starting tracks**: an object can leave, die, or be missed; a new one can enter.
  Give "end this track" and "start a new track" a fixed **cost b** each. A link is chosen only if
  it is cheaper than ending one track and starting another (cost < 2b). With the example
  above: b ≥ 3.5 keeps both links; 0.5 < b < 3.5 keeps only B–P and ends A / starts Q; b < 0.5
  links nothing. **b is the most important parameter**; it acts like a soft gate.
- **Errors compound**: a link error rate of 10 % per frame means a 20-frame track is perfect
  only 0.9²⁰ ≈ 12 % of the time. Long correct tracks need very low per-link error rates.
- Pointer (one line): linking can also be done over the whole movie at once (min-cost flow,
  multiple-hypothesis tracking); frame by frame is the common, fast default.

Math layer (`## The math: … {.math}` slides):

- *Linking as linear assignment*: $\min \sum_{ij} c_{ij} x_{ij}$, $x_{ij} \in \{0,1\}$, each
  row and column used at most once. Gated pairs: $c_{ij} = \infty$ (omitted). The **augmented
  matrix** for births and deaths (Jaqaman et al. 2008): a $(m+n)\times(n+m)$ matrix
  $\begin{pmatrix} C & D \\ B & C^\top_{\text{dummy}} \end{pmatrix}$ with death costs $b$ on the
  diagonal of the upper-right block, birth costs on the diagonal of the lower-left block
  (off-diagonals ∞), and the lower-right block making the matrix square (any chosen link
  "pays" a matching dummy–dummy entry). A perfect assignment of this square matrix = a set of
  links + ends + starts. Total unimodularity → the LP relaxation is integral (state);
  Hungarian / Jonker–Volgenant in $O(N^3)$ (state).
- *Squared distance as a log-likelihood*: for a 2D random walk the step $\Delta\mathbf r$ has
  density $\frac{1}{4\pi D\Delta t}e^{-|\Delta\mathbf r|^2/4D\Delta t}$, so $-\log p =
  |\Delta\mathbf r|^2/(4D\Delta t) + \text{const}$; minimising the sum of squared distances
  maximises the joint likelihood of all steps (independent objects). The birth/death cost is
  then a log-ratio of probabilities; $b$ in squared-distance units ≈ "how far is too far".
- *Error estimate*: random positions at density $n$; the true step $r$ has $r^2 \sim$
  exponential with mean $4D\Delta t$; the probability that some other object lies closer than
  $r$ is $1 - e^{-\pi n r^2}$; averaging, $P_\text{err} \approx \frac{a}{1+a}$ with $a = 4\pi n
  D\Delta t$. Numbers above: $a = 0.126$, $P \approx 0.11$ per link (nearest neighbour; global
  assignment does better, the demo shows how much).

**Demo `linkpairs`** (new, small; lecture and exercise). Two frames, up to 8 points each:
frame t filled discs, frame t+1 open rings, all draggable (large handles). Method buttons:
*nearest neighbour* (arrows; conflicts drawn red), *greedy* (links appear in order of cost,
step button), *global* (the assignment). Cost: distance / squared distance. Sliders: gate radius
(circles drawn around frame-t points; pairs outside greyed), end/start cost b. Side panel: the
cost matrix with numbers (`heatmapGrid`), chosen cells boxed, gated cells "–"; toggle
*augmented matrix* shows the full $(m+n)\times(n+m)$ block structure with b on the diagonals.
Readout: total cost of each method. Presets: "NN conflict", "greedy loses" (the A, B, P, Q
example), "gate too small: a track breaks", "ending is cheaper than a long link" (b = 2),
"a newcomer steals a link", "augmented matrix". Reset.

**Demo `linkmovie`** (new, main). A seeded simulated movie: N particles (10–200) in a
256×256 µm-free field, 60 frames, motion type (random walk / directed + noise / mixed),
D (or speed) and frame interval sliders, density slider, localisation noise σ. Particles
leave and enter at the border. Display: the current frame with dots and track tails (colour
= assigned track id, `labelColor`); wrong links marked by a red cross on the link; a time
slider and play button. Methods: nearest neighbour / greedy / global (with b and gate
sliders). Readouts: **displacement/spacing ratio**, link error rate, fraction of
ground-truth tracks reconstructed perfectly. Side plot: link error rate vs the ratio (dots from
the current run of all three methods on a precomputed sweep, the $a/(1+a)$ estimate as a
line). Toggle *show drift*: adds a stage drift; linking without correcting it fails
(recall `stitching/`). Optional `data-source="image"`: frames rendered as spots
(`lib/synth.ts` `spots`), detected by threshold → label → centroid, so missed and merged
detections appear (forward pointer to unit 3). Presets: "sparse and slow: everything works",
"ratio 0.5: NN fails, global holds", "ratio 1: nobody can", "drift", "b too low: tracks
shatter", "b too high: wrong links". Reset.

By hand (concept):

- (a) The A, B, P, Q example: nearest neighbour, greedy, global, with squared distances
  (conflict on P; 26 vs 8). Repeat with plain distances (6 vs 4): does the global choice change?
- (b) End/start cost: for which b is each of the three solutions optimal (b ≥ 3.5; 0.5 < b <
  3.5; b < 0.5; ties at 3.5 and 0.5)? What does "b = 3.5" mean as a distance?
- (c) A 3×3 cost table with one gated pair: find the global assignment by listing the
  feasible permutations (at most 6).
- (d) Difficulty: D = 0.05 µm²/s, Δt = 0.5 s, 0.5 objects per µm². Compute the rms
  displacement, the mean spacing and the ratio. Will nearest neighbour work? What frame
  interval gives a ratio of 0.2?
- (e) Errors compound: per-link error 2 %; chance that a 50-frame track has no error
  (0.98⁵⁰ ≈ 0.36). And at 0.5 %?
- (f) Predict: two nuclei of very different size cross; which extra cost term helps?

Math questions:

- **B1.** *(math)* Build the augmented matrix for the A, B, P, Q example with b = 3, solve it
  by enumeration, and check that the result matches (b).
- **B2.** *(math)* Show that minimising the sum of squared distances maximises the likelihood
  of independent 2D random-walk steps, and express b as a probability ratio.
- **B3.** *(math)* Derive $P_\text{err} \approx a/(1+a)$ from $E[e^{-\pi n r^2}]$ with $r^2$
  exponential. What does it become for $a \ll 1$?
- **B4.** *(math)* Give a 3×3 example in which greedy is worse than global by an arbitrarily
  large factor.

### 2. `track-predict/`: Motion models and the Kalman filter

Concept layer:

- **Why predict**: a fast object moving in a straight line is far from where it was, but close
  to where it is **going**. Link to the **predicted** position, not the last one. Picture: two
  particles crossing at speed: nearest neighbour (and global on last positions) swaps them;
  prediction keeps them.
- **Motion models** in words: *random walk* (best guess = last position; uncertainty grows
  with time); *constant velocity* (best guess = last position + velocity × Δt). Real objects
  switch (a vesicle waits, then runs along a microtubule): pointer to multiple models at the end.
- **Uncertainty as a Gaussian**: the estimate is a position ± a standard deviation (a bell
  curve on a line; an ellipse in 2D, recall `pca/`).
- **The Kalman filter, intuition first (1D)**: two steps per frame.
  1. **Predict**: move the estimate with the model; its uncertainty **grows** (the model is not
     perfect: process noise Q).
  2. **Update**: a measurement arrives with its own uncertainty R (localisation error σ² from
     the point detection day). The new estimate is a **weighted average** of prediction and
     measurement, weighted by how much we trust each. The weight on the measurement is the
     **gain** K = P⁻/(P⁻ + R). The new uncertainty is smaller than both.
  Worked: prediction 10 ± 2 (P⁻ = 4), measurement 13 ± 1 (R = 1): K = 0.8, estimate 12.4,
  variance 0.8. Measurement noisy (R = 16): K = 0.2, estimate 10.6.
- **Velocity is estimated without being measured**: in the constant-velocity model the state
  is (position, velocity); a surprise in position also corrects the velocity (worked numbers in
  the exercise). Picture of the estimate locking on after a few frames.
- **Missed detection**: no update; predict again; the uncertainty grows; the next measurement
  can be farther away and still be accepted. This is how prediction helps gap closing (unit 3).
- **Gates from the prediction**: accept a candidate only inside the predicted ellipse (e.g. 3
  standard deviations). The distance in "standard deviations" is the **Mahalanobis distance**
  (one concept slide; pointer to `gaussian-mixture/` for the full treatment). The gate grows
  automatically for uncertain tracks and shrinks for well-predicted ones. The cost for linking
  becomes the Mahalanobis distance² + a log-determinant term.
- **Tuning** Q and R: R from the detector (σ); Q is the model's honesty. Q too small: the
  filter trusts a wrong model and lags at turns. Q too large: it follows the noise (no better
  than the raw positions). The steady-state gain depends only on the ratio Q/R.
- **When one model is not enough**: objects that switch between waiting, diffusing and directed
  runs; run several filters in parallel and weight them by how well each predicts (interacting
  multiple models, one line), or choose per track. Pointer to unit 4 (classifying motion from
  the MSD).
- Smoothing (one line): after the whole movie is known, run the filter backwards too (a
  smoother); positions improve, especially at gaps.

Math layer:

- *Product of two Gaussians (1D)*: $\mathcal N(x; \hat x^-, P^-)\,\mathcal N(z; x, R) \propto
  \mathcal N(x; \hat x, P)$ with $\hat x = \hat x^- + K(z - \hat x^-)$, $K = P^-/(P^-+R)$,
  $P = (1-K)P^- = \big(1/P^- + 1/R\big)^{-1}$: precisions add.
- *Matrix form*: state $\mathbf x$, model $\mathbf x_{t+1} = F\mathbf x_t + \mathbf w$,
  $\mathbf w \sim \mathcal N(0, Q)$; measurement $\mathbf z = H\mathbf x + \mathbf v$,
  $\mathbf v \sim \mathcal N(0, R)$. Predict $\hat{\mathbf x}^- = F\hat{\mathbf x}$, $P^- = FPF^\top
  + Q$. Update: innovation $\mathbf y = \mathbf z - H\hat{\mathbf x}^-$, $S = HP^-H^\top + R$, $K =
  P^-H^\top S^{-1}$, $\hat{\mathbf x} = \hat{\mathbf x}^- + K\mathbf y$, $P = (I - KH)P^-$. Constant
  velocity in 1D: $F = \begin{pmatrix}1 & \Delta t\\ 0 & 1\end{pmatrix}$, $H = (1\;\;0)$; 2D:
  block-diagonal copies.
- *Gating*: $d^2 = \mathbf y^\top S^{-1}\mathbf y$ is χ² with 2 degrees of freedom in 2D;
  $d^2 < 9.21$ keeps 99 % of true measurements. Link cost $d^2 + \log\det S$ (the negative
  log-likelihood up to a constant).
- *Steady state (random walk, 1D)*: $P^-_\infty = \tfrac12\big(Q + \sqrt{Q^2 + 4QR}\big)$; for
  $Q = R$, $K_\infty = (\sqrt5 - 1)/2 \approx 0.618$. The filter is then exponential smoothing
  with weight $K_\infty$.

**Demo `kalman1d`** (new, lecture and exercise). An object moves on a line (constant velocity
with an optional turn at a preset time); noisy measurements each frame (seeded). Top panel: the
current step as Gaussians on the position axis: prediction (blue curve), measurement (orange),
updated estimate (dark), the gain as a bar between them. Buttons *predict* / *update* / *step*
/ *play* / *reset*; *skip measurement* (a missed detection). Bottom panel: position vs time
with truth, measurements, estimate and a ±2σ band (`band`), and a small plot of K over time
converging to its steady state. Sliders: R (measurement noise), Q (process noise), model
(random walk / constant velocity). Presets: "trust the measurements" (R small), "trust the
model" (Q small: lag at the turn), "random-walk model on a moving object: always behind",
"constant velocity: locks on", "three missed frames: the band widens", "golden ratio" (Q = R,
random walk). Reset.

**Demo `kalmanlink`** (new, main). 2D particles with nearly constant velocity, some crossing,
plus localisation noise and missed detections. Each track shows its predicted position and
gate (circle of fixed radius without prediction; covariance ellipse with the Kalman filter,
`Plot.ellipse`). Methods: *nearest neighbour*, *global on last positions*, *global on predicted
positions (Kalman)*. Sliders: speed, density, Q, R, miss probability, gate (in σ). Readouts:
link error rate per method, identity swaps at crossings. Presets: "two particles cross",
"a dense stream", "a sudden turn: the gate misses", "missed frame: the ellipse grows",
"mixed population: waiting + running" (one model fits neither well). Reset.

By hand (concept):

- (a) The 1D update with prediction 10, P⁻ = 4 and measurement 13 with R = 1, R = 4, R = 16.
  Which estimate is closer to the measurement, and why?
- (b) Predict with random walk, Q = 1, from estimate 12.4, P = 0.8; then a missed frame; then
  a measurement 15 with R = 1. (P⁻ = 1.8, then 2.8, then K = 2.8/3.8 ≈ 0.74.)
- (c) Constant velocity, Δt = 1, P = I, Q = diag(0, 0.1): compute P⁻ = [[2, 1], [1, 1.1]];
  with R = 1 the gain is (2/3, 1/3); an innovation of 3 moves the position by 2 and the
  velocity by 1. Explain in words why the velocity changes.
- (d) Gate: innovation 2.5 px, S = 4 px² in 1D. Inside a 3σ gate? What if S = 0.5?
- (e) Predict: a vesicle that waits for 10 frames, then runs. What does a constant-velocity
  filter do at the start of the run? And a random-walk filter with large Q?

Math questions:

- **B5.** *(math)* Derive the 1D update from the product of two Gaussians; show that the
  precisions add.
- **B6.** *(math)* Show that the matrix form reduces to the 1D form for $F = H = 1$.
- **B7.** *(math)* Derive the steady-state $P^-_\infty$ for the random-walk filter and show
  $K_\infty = (\sqrt5-1)/2$ for $Q = R$; show the filter is then exponential smoothing.
- **B8.** *(math)* Show that $d^2 = \mathbf y^\top S^{-1}\mathbf y$ has expected value 2 in 2D
  when the model is right, and explain how this tests the tuning of Q and R.

### 3. `track-events/`: Gaps, divisions and lineage trees

Concept layer:

- **What breaks a track** (pictures): a **missed detection** (blinking, out of focus, too dim)
  splits one track into two; an object **leaves** or **enters** the field (or the focal plane);
  a cell **divides**; two objects **touch** and are detected as one (merge) or one is cut into two
  by segmentation (split); a false detection starts a short spurious track.
- **Two stages** (the standard recipe, Jaqaman et al. 2008): first link frame to frame (unit 1)
  with strict gates → short, reliable **segments**. Then one more global assignment over
  segment **ends** and **starts** decides, for each: **close a gap** (an end at t joins a start at
  t+2 … t+G), **split** (a start is joined to the middle of another segment: a division),
  **merge** (an end joins the middle of another segment), or stay a real end / real start.
  Each has a cost; each "do nothing" has an alternative cost, as in unit 1.
- **Gap closing**: allowed gap G frames and a larger gate (the object may have moved further:
  gate ∝ √(gap) for random motion, ∝ gap for directed motion; the Kalman prediction of unit 2
  gives the natural gate). G too large: wrong joins between different objects.
- **Divisions** (splits): the mother's track ends near two new starts; cues beyond distance:
  daughters about half the mother's area, mother bright and round just before (mitosis),
  daughters on opposite sides. A division is a mother **and both** daughters.
- **A split is not always a division**: segmentation cuts a cell in two for one frame and the
  pieces rejoin → a split followed by a merge, a small **"bubble"** in the graph. Keep what the
  linking observed; decide what it means in a separate step (e.g. a division must be followed by
  two daughters that stay separate for k frames). Likewise a track that ends is not a dead cell,
  and one that starts is not a newborn: report **why** it starts/ends (first/last frame, near the
  border, after a gap, at a division) instead of guessing biology from linkage.
- **Entering and leaving**: tracks that start or end within a border band (recall the border rule
  of `label-measure/`) are "entered/left the field", not births and deaths.
- **The track graph**: nodes = detections, edges = links (continue, gap, divide, merge). A
  **track** = a maximal chain without branches; tracks have parents and children.
  **Lineage tree**: time downwards, a branch at each division; reading it: cell-cycle time =
  time between a cell's birth (division) and its own division; generation number; sister cells.
- **Censoring**: in a 48 h movie, cells born late cannot complete a 20 h cycle before the end;
  measuring cycle times only on complete cycles under-estimates the mean (slow cells are
  missing). Report how many cycles were incomplete.
- **Evaluating tracks** (two slides): need a reference (manual or simulated) on the same
  detections. Count **links**: TP (in both), FP (only in result), FN (only in reference) →
  precision, recall, F1 (as for objects, recall `object-eval/` if seen). **Divisions**: correct only
  if the mother and both daughters match. **Complete tracks**: fraction of reference tracks
  reconstructed without any error (shows compounding, unit 1). **Identity switches** at crossings.
  When detections themselves differ from the reference, first match detections per frame (the
  point detection day / `object-eval/`), then compare links. Pointer to optional `track-eval/`.

Math layer:

- *The segment-linking matrix*: rows = segment ends + "split candidates" (detections that could
  send a second link), columns = segment starts + "merge candidates"; blocks for gap closing,
  merging, splitting, and diagonal alternative blocks; again a square augmented matrix solved
  by one assignment. Alternative costs are often set from the data (e.g. a percentile of the
  candidate costs times a factor slightly above 1).
- *Whole-movie linking as min-cost flow* (pointer, links to `maxflow/` if seen): each
  detection a node with capacity 1, source → node = start cost, node → sink = end cost, edges
  between frames with link costs; the minimum-cost flow is the globally best set of tracks
  without divisions; integer by total unimodularity.
- *Gap gates*: displacement over g frames of a random walk has variance $2dDg\Delta t$, so a
  gate of k standard deviations grows as $\sqrt g$.
- *Compounding*: $P(\text{track of } T \text{ links correct}) = (1-e)^T \approx e^{-eT}$;
  expected number of errors per track $eT$.

**Demo `gapclose`** (new, main). A seeded simulation of dividing cells (cycle time from a
gamma distribution, daughters displaced to both sides, area halved, mother bright for one frame
before dividing), with sliders for **miss probability**, **false detections**, **segmentation
split** probability (one cell → two pieces for one frame), cells entering at the border. Left
panel: a **space–time view** (x position vs time, one line per segment; or the 2D field with
tails, toggle). After frame-to-frame linking, segments are drawn as coloured lines; candidate
joins of the second stage dashed with their costs; chosen joins solid, marked *gap*, *split*,
*merge*. Right panel: the **lineage tree** of the selected cell (reconstructed) next to the true
tree, wrong branches red. Sliders: max gap G, gap gate, alternative cost, toggles splitting /
merging / *require daughters to stay separate k frames*. Readouts: link P/R/F1, division F1,
fraction of complete correct tracks, counts of starts/ends by reason (first frame, border, gap,
division, unexplained). Presets: "a blink: gap closing", "G too large: wrong joins", "a division",
"segmentation split → a bubble", "bubble removed by the k-frame rule", "cells entering at the
border", "the censoring trap" (cycle-time histogram of complete vs all cycles). Reset.
`data-panels="space-time,tree"` for slides.

By hand (concept):

- (a) Gap closing as an assignment: ends E1, E2 and starts S1, S2 with costs E1–S1 = 5,
  E1–S2 = 30, E2–S1 = 8, E2–S2 gated out; each unused end or start costs b. For b = 12 the
  optimum is E1–S1 alone (29); for b = 20 it is E1–S2 + E2–S1 (38). Explain in words why a
  larger b forces more joins.
- (b) A list of detections over 6 frames (given as a table: frame, x, y, area): draw the
  tracks, mark one gap, one division (area halves), one segmentation split that rejoins.
- (c) Read a lineage tree (given figure with times): cycle times of each cell, which cells are
  censored, mean cycle time with and without censored cells.
- (d) Evaluate: a reference with 12 links and one division; a result with 11 links of which 10
  are correct, and one division with one wrong daughter. Link precision, recall, F1; division
  TP/FP/FN.
- (e) Predict: what happens to the number of divisions found if the segmentation splits nuclei
  more often? And to the number of gaps if detection misses rise from 1 % to 10 %?

Math questions:

- **B9.** *(math)* Write the segment-linking matrix for 2 segments with one gap candidate and
  one split candidate; mark the blocks and their diagonal alternatives.
- **B10.** *(math)* Formulate linking without divisions over a whole movie as a min-cost flow;
  argue why each node has capacity 1.
- **B11.** *(math)* Censoring: cycle times are exponential with mean μ (shifted by a minimum
  time m), cells born uniformly over a movie of length L. Compute the mean of the observed
  complete cycles; show that it is below μ + m.

### 4. `track-analysis/`: What tracks measure: speed, MSD and modes of motion

Concept layer:

- **Per-track numbers**: duration, net displacement, path length, mean speed, **straightness**
  (net displacement / path length), turning angles. Pictures of a random walk, a directed run
  and a confined wobble with these numbers.
- **The speed trap**: for random motion the "speed" measured from consecutive frames grows when
  the frame interval shrinks (∝ 1/√Δt): D = 0.1 µm²/s gives 1.8 µm/s at Δt = 0.1 s and 5.6
  µm/s at Δt = 0.01 s. Random motion has no speed; report D instead. Localisation noise adds
  apparent speed and lowers straightness too.
- **Mean squared displacement**: for every lag τ, average the squared displacement over all
  pairs of frames τ apart. Worked: 1D track 0, 1, 3, 2, 4 → MSD 2.5, 3.67, 6.5, 16 at lags 1–4.
  Shapes: **free diffusion**: a straight line through 0, slope 2dD (d = dimensions);
  **directed** motion: curves upward (slope 2 on a log–log plot at long lags; velocity v from
  the τ² term); **confined**: levels off at a plateau set by the size of the cage (R² for a
  disc of radius R); **anomalous** (crowded, obstacles): a power law τ^α with α < 1.
- Naming (one sentence): "diffusion" here is random motion; the diffusion models of the deep
  learning day borrowed the name.
- **Reading an MSD honestly**: large lags are averages of few pairs, so the curve is noisy
  there; fit only the first part (e.g. the first quarter of lags). Localisation error adds a
  constant **offset** 4σ² in 2D (the line does not go through 0): with σ = 30 nm, D = 0.01 µm²/s,
  Δt = 0.05 s, the offset (0.0036 µm²) is larger than the lag-1 signal (0.002 µm²). Short
  tracks give widely scattered D values: a histogram of per-track D is wide even for identical
  particles. Gap-closed tracks have missing frames: compute lags from times, not indices.
- **Biases from tracking itself**: fast particles leave the field or focal plane and give short
  tracks; keeping only long tracks keeps the slow ones. Linking errors at high density cut
  jumps short (the tracker prefers short links) → D under-estimated. Pointer back to unit 1's
  difficulty ratio.
- **Ensemble vs single track**: average over all tracks at each lag, or MSD per track then
  distribution; different questions (population mean vs heterogeneity). Switching motion
  (waiting/running) within one track: segment the track first (pointer: hidden state models,
  one line).
- **Cells** (one slide): migrating cells are persistent random walkers: straight on short times,
  random on long times; the persistence time is where the MSD changes from slope 2 to slope 1
  on a log–log plot.
- Where this leads: velocities from tracks vs velocity fields from optical flow (Motion group);
  lineage statistics (unit 3); intensity over time per track (measurements day, per frame).

Math layer:

- *Random walk*: $\mathbf r(t) = \sum_k \Delta\mathbf r_k$ with independent steps of variance
  $2D\Delta t$ per axis; $\text{MSD}(\tau) = E|\mathbf r(t+\tau) - \mathbf r(t)|^2 = 2dD\tau$
  (variances of independent steps add).
- *Directed + diffusion*: $\text{MSD}(\tau) = 2dD\tau + v^2\tau^2$. *Localisation error*:
  $+2d\sigma^2$ (static; motion blur subtracts a term $\propto D\Delta t_\text{exposure}$, state).
  *Confined in a disc of radius R*: plateau $R^2$ ($E|X-Y|^2 = 2E|X|^2$ for independent uniform
  points). *Anomalous*: $\text{MSD} \propto \tau^\alpha$, α from the slope on log–log axes.
- *Speed from a random walk*: the 2D step length is Rayleigh with $E r = \sqrt{\pi D\Delta t}$,
  so the apparent speed is $\sqrt{\pi D/\Delta t}$.
- *Persistent random walk* (Fürth): $\text{MSD}(\tau) = 2s^2P\big(\tau - P(1 - e^{-\tau/P})\big)$
  in 2D with speed s and persistence time P: ≈ $s^2\tau^2$ for τ ≪ P, ≈ $2s^2P\tau$ for τ ≫ P.
- *Why the tail is noisy*: in a track of N frames there are N − τ overlapping pairs at lag τ;
  the relative error of the time-averaged MSD grows roughly as $\sqrt{\tau/N}$ (state; Qian et
  al. 1991).

**Demo `msdplot`** (new, main). Simulated 2D tracks (10–100) of a selectable type: free
diffusion, directed + diffusion, confined in a disc, anomalous (fractional Brownian motion via
an FFT-based generator, α slider), switching (waiting/running), persistent random walk. Sliders:
D, v, R, α, localisation error σ, frame interval Δt, track length N. Panels: tracks (one
highlighted); MSD vs lag (lin and log–log toggle) per track (thin) and ensemble (thick) with
fitted line/curve and the fit range shaded; histogram of per-track D; readouts D, α, v, offset.
`data-mode="speed"`: one long random-walk track subsampled at Δt, 2Δt, 4Δt …: apparent speed
vs Δt with the $\sqrt{\pi D/\Delta t}$ curve. Presets: "free diffusion: slope 1",
"directed: slope 2 at long lags", "confined: plateau at R²", "localisation error: the offset",
"short tracks: D scatters", "the speed trap", "noisy tail: fit only the start",
"a persistent cell: from 2 to 1". Reset.

By hand (concept):

- (a) MSD of the 1D track 0, 1, 3, 2, 4 at lags 1–4 (2.5, 3.67, 6.5, 16); estimate D from lag 1
  (1D: MSD = 2Dτ → D ≈ 1.25). How many pairs is the lag-4 value based on?
- (b) Three MSD curves (given figure): which is diffusion, directed, confined? Read D and R.
- (c) The speed trap: apparent speed for D = 0.1 µm²/s at Δt = 0.1 s and 0.01 s.
- (d) Localisation offset: σ = 30 nm, D = 0.01 µm²/s, Δt = 0.05 s; lag-1 signal vs offset.
  What happens to D if you fit a line through the origin?
- (e) Straightness of a given 5-step track; how does it change if each position gets noise?
- (f) Predict: at high density, linking errors cut long jumps short. Does D go up or down?

Math questions:

- **B12.** *(math)* Derive MSD(τ) = 2dDτ, then add a constant velocity and a static
  localisation error.
- **B13.** *(math)* Show the plateau of a particle uniform in a disc of radius R is R².
- **B14.** *(math)* Derive the mean step length $\sqrt{\pi D\Delta t}$ of a 2D random walk.
- **B15.** *(math)* Show the two limits of the persistent random walk MSD.

### 5. (optional) `track-eval/`: Evaluating tracking in depth

- When detections differ from the reference: per-frame matching first (distance gate for spots,
  IoU for objects; recall `object-eval/`), then link scores on matched pairs.
- Graph-based scores: the cost of the operations needed to turn the result graph into the
  reference graph (add/remove/split nodes, add/remove/relabel edges), normalised (the
  cell-tracking-challenge style "TRA" idea, described without naming it); identity-focused
  scores from multi-object tracking: identity switches, fragmentation, MOTA, IDF1, HOTA (one
  slide comparing what each rewards). Complete-lineage accuracy for division trees.
- Practical: evaluate on simulated data with known truth (the demos' simulators), and on a
  manually corrected sub-region (only where the reference is complete: absent links in an
  incomplete reference are not errors).
- **Demo `trackeval`** (new, optional): `gapclose`'s simulator with an error generator (swap,
  break, false join, missed division, spurious division); panels: the graph edit operations,
  link F1, division F1, IDF1-like score, complete-track fraction side by side. Presets show
  that each score ignores some error type.

### 6. (optional) `track-learned/`: Learned costs and tracking by detection

- Appearance features (intensity, shape, texture from the measurements day; learned
  embeddings from day 39) as extra cost terms; learned association scores (a network sees two
  frames and predicts link probabilities); the assignment step stays the same (scores → costs,
  costs → LAP). Calibrated probabilities vs arbitrary scores; the alternative costs must be in
  the same units.
- Tracking by segmentation propagation (label overlap between frames), tracking with flow as
  the prediction (Motion group). 3D tracking: anisotropic voxels, physical units in costs.
- No demo (or reuse `linkmovie` with `data-cost="appearance"`: dots carry an intensity that
  drifts slowly, and a weight slider adds it to the cost).

## Shared engineering

**Reuse** (`demos/src/lib/`; see `demos/src/lib/README.md`):

- `eval.ts`: `hungarian(cost, n)` (square, row-major, O(n³)) for every assignment; build the
  augmented matrix around it in `lib/linking.ts`. `iouTable` for the optional overlap cost of
  segmented objects; `objectScores` for link/division P/R/F1 (counts in, scores out).
- `random.ts`: `rng`, `gauss`, `poisson` (births, false detections), `bernoulli` (misses,
  segmentation splits), `exponential`, `choice`. Fixed seeds per preset.
- `plot.ts`: `Plot` with `line`, `band` (Kalman ±2σ), `scatter`, `arrow`, `ellipse`
  (prediction gates), `histogram`, log–log axes (MSD), `heatmapGrid` (cost matrices with
  numbers), `vline` (current frame), `legend`.
- `linalg.ts`: `matMul`, `transpose`, `inv2`, `inverse`, `cholesky`, `det2` for the 4-state
  Kalman filter and Mahalanobis gates; `gmm.ts` `mahalanobis2` if convenient.
- `stats.ts`: `linfit` (MSD fits), `mean`, `quantile` (percentile alternative costs), `sd`.
- `neighbours.ts`: `nearestNeighbours`, `clarkEvans` for the spacing in the difficulty ratio.
- `colour.ts`: `labelColor`/`labelPalette` (track identities), `OKABE_ITO` (event types),
  `categoryColor`.
- `synth.ts` (`spots`, `nuclei`, `addPoissonNoise`) and `label.ts` (`label2d`, `centroids`) for
  the image mode of `linkmovie` and the cell frames of `gapclose`.
- `fft.ts` (`fft1d`) for fractional Brownian motion (circulant embedding) in `msdplot`.
- `ui.ts`: `slider`, `button`, `checkbox`, `presets`, `draggable`, `svgPoint`, `autoMount`,
  `numList`. `registration.ts`: not needed (drift in `linkmovie` is simulated and removed with
  the known shift; the recall points to the `drift` demo).

**New library files owned by this day** (names checked against `demos/src/lib/`: none exist):

- `lib/tracksim.ts` (no DOM): seeded simulators with ground truth: particles (random walk,
  directed, confined, fBm, switching, persistent), border entry/exit, localisation noise,
  missed and false detections, stage drift; dividing cells (gamma cycle times, daughter
  placement, area halving, mitotic brightening, segmentation-split errors); output = detections
  per frame + the true track graph.
- `lib/linking.ts` (no DOM): gated candidates (uniform grid with cell = gate), costs (squared
  distance, + area/intensity/IoU terms with weights), nearest neighbour, greedy, **global LAP
  with births/deaths** (augmented matrix around `eval.ts` `hungarian`, solved per connected
  component of the candidate graph for speed), **segment linking** (gap closing, splitting,
  merging), the track graph type (nodes, edges with kind continue/gap/divide/merge, tracks as
  unbranched chains, start/end reasons incl. border), evaluation (link P/R/F1, division
  triplets, complete tracks, identity switches). Also exports a `matchPoints(a, b, maxDist)`
  gated assignment that the point detection day can use for evaluation (open question 2).
- `lib/kalman.ts` (no DOM): small dense linear Kalman filter (n ≤ 4: predict, update,
  innovation, S, Mahalanobis d², log det S), model builders (random walk / constant velocity,
  1D/2D, Δt), steady-state gain, an RTS smoother (optional unit / smoothing slide).
- `lib/trackstats.ts` (no DOM): per-track numbers (duration, net displacement, path length,
  speed, straightness, turning angles), time-averaged and ensemble MSD (lags from timestamps,
  pair counts), fits (linear with offset, power law on log–log, diffusion + directed,
  confined plateau), the error bars of the MSD.
- `lib/lineage.ts` (pure layout + one DOM draw function, marked): lineage tree layout (time
  downwards, tidy placement of branches), cycle times, generations, censoring flags; draws into
  a `Plot` or SVG group.

**Demos** (folders checked against `demos/src/`: none exist): `linkpairs`, `linkmovie`,
`kalman1d`, `kalmanlink`, `gapclose`, `msdplot`; optional `trackeval`.

**Speed** (TypeScript first, measure): a dense 400×400 Hungarian is ~6·10⁷ steps (≈100 ms):
too slow for a live slider over 60 frames. Split each frame pair into connected components of
the gated candidate graph first (as the user's tracking module does after a large component
took minutes on real data): typical components have < 10 nodes, so relinking a 200-particle,
60-frame movie should take a few ms. `linkmovie` recomputes on slider release if a full relink
measures > 30 ms; Wasm only if a measured live recompute stays > ~30 ms after component
splitting (not expected). `msdplot`: 100 tracks × 1000 frames MSD is 10⁷ operations at most
(cache per simulation; fits are cheap).

**Tests** (`demos/test/tracking-test.ts`, run by `make test`): LAP with alternatives against
brute-force enumeration on random problems up to 5×5 (including gated pairs); the by-hand
numbers: 26 vs 8, the b thresholds 0.5 and 3.5, gap-closing optima 29 (b = 12) and 38
(b = 20); Kalman 12.4/0.8, the CV gain (2/3, 1/3) and P after update [[2/3, 1/3], [1/3,
23/30]], $K_\infty$ = 0.618; MSD 2.5, 11/3, 6.5, 16; simulated free diffusion MSD slope within
5 % of 4D (seeded, 200 tracks × 200 frames); confined plateau within 5 % of R²; nearest-neighbour
error rate within ±0.03 of $a/(1+a)$ at a = 0.126 (Poisson positions, seeded); segment linking
recovers every gap of length ≤ G in a sparse simulation; division triplet scoring (one wrong
daughter = 1 FP + 1 FN).

**Rules** (demos/README.md, plan/build-issues.md): TypeScript, `make typecheck`, no runtime
network, fixed seeds, presets + reset, `slider`/`button`/`checkbox` helpers (keyboard stays with
reveal.js), fits 1150×560 on a slide (`gapclose` and `msdplot` use `data-panels` to show two
panels on slides and all on the exercise page), labels and slide text ≥ 18 px, per-demo CSS
selectors prefixed with `.qb-demo[data-demo="…"]` (demo.css loads late), plots via `lib/plot.ts`
(clip ids are fixed; no local `uniqueClip()` copies), static figures for print.

**Images/data**: everything simulated with ground truth (labelled "simulated" in captions). Wanted
(open question 5): one real time-lapse with dividing nuclei and a manual lineage, and one real
single-particle movie, both with a clear licence.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `track-linking/`: lecture (`linkpairs`), assignment by hand, `linkmovie` |
| 10:45–12:15 | Unit 2 `track-predict/`: lecture (`kalman1d`), Kalman by hand, `kalmanlink` |
| 13:15–14:45 | Unit 3 `track-events/`: lecture, gap closing and lineage by hand, `gapclose` |
| 15:00–16:30 | Unit 4 `track-analysis/`: lecture, MSD by hand, `msdplot`; quizzes as wrap-up |

## Open questions

1. **Ownership of the linear assignment problem.** `object-eval/` (day 37) teaches it for
   evaluation; this day can come before day 37. Proposal: this day owns "assignment with
   costs, gates and alternatives" (concept + math); `object-eval/` keeps the IoU special case and
   its math slide and adds a one-line pointer. Needs a line in the coursemap owners table
   ("Linking / assignment with costs: tracking").
2. **Point matching code**: the point detection day evaluates detections by matching them to
   ground truth within a distance. Proposal: `lib/linking.ts` exports `matchPoints` (gated LAP on
   distances); whichever day is built first creates it. Coordinate with the point-detection
   builder so there is one implementation, not two.
3. **Four units or five?** Track evaluation is in the day's scope. Proposal: the essentials
   (link/division F1, complete tracks, compounding) are two slides in unit 3; the graph-edit and
   MOT-style scores are the optional `track-eval/`. Alternative: make `track-eval/` core and move
   "cells: persistent random walk" and "ensemble vs single track" to optional.
4. **Kalman depth**: the 4-state matrix form is math layer only; the concept layer is 1D. Is the
   interacting-multiple-models pointer enough, or should switching motion be a preset in
   `kalmanlink` with two filters? Proposal: pointer + the "mixed population" preset.
5. **Real data**: a licensed time-lapse with divisions and a reference lineage (e.g. from a
   public tracking benchmark; check the licence of each data set) and a single-particle movie.
   Until then all demos are simulated, which is also what makes evaluation possible.
6. **Optical-flow day**: not planned yet. The flow-vs-tracking slide in unit 1 and the
   "flow as prediction" line in optional unit 6 need its folder names once it exists.
7. **Coursemap edits** (not done here): the Motion row's tracking entry → "planned
   (`plan/tracking-day.md`)"; owners table: Kalman filter, MSD, lineage trees → tracking.
   Pending pointers in built units: `object-eval/` (assignment pointer), `stitching/` ("Drift in
   a time-lapse": forward pointer to tracking), `structure-tensor/` (its "motion day"
   pointer should name the optical-flow day for corners and this day for objects).

Teacher notes from the user's tracking module (inspiration, not student material): gate in
registered (drift-corrected) coordinates; birth/death costs must be in the same units as link
costs; solve per connected component; two-stage linking with percentile-based alternative costs
(the published LAP-tracker convention: every chosen link also pays the minimum alternative in
the lower-right block); record start/end **reasons** (first/last frame, border, gap, division)
instead of inferring death or birth; split-then-merge bubbles are kept in the graph and resolved
later; evaluate links, unbranched links and exact division triplets separately, and never on an
incomplete reference.

## References

- Jaqaman, Loerke, Mettlen, Kuwata, Grinstein, Schmid & Danuser (2008), *Robust single-particle
  tracking in live-cell time-lapse sequences*, Nature Methods (two-stage LAP, gap closing,
  splitting/merging).
- Kuhn (1955), *The Hungarian method for the assignment problem*; Jonker & Volgenant (1987), *A
  shortest augmenting path algorithm for dense and sparse linear assignment problems*,
  Computing.
- Kalman (1960), *A new approach to linear filtering and prediction problems*, J. Basic Eng.;
  Welch & Bishop, *An introduction to the Kalman filter* (tutorial).
- Blom & Bar-Shalom (1988), *The interacting multiple model algorithm for systems with
  Markovian switching coefficients*, IEEE TAC.
- Zhang, Li & Nevatia (2008), *Global data association for multi-object tracking using network
  flows*, CVPR (min-cost flow).
- Chenouard et al. (2014), *Objective comparison of particle tracking methods*, Nature Methods.
- Ulman et al. (2017), *An objective comparison of cell-tracking algorithms*, Nature Methods
  (graph-based tracking scores).
- Luiten et al. (2021), *HOTA: a higher order metric for evaluating multi-object tracking*,
  IJCV; Ristani et al. (2016), IDF1.
- Qian, Sheetz & Elson (1991), *Single particle tracking: analysis of diffusion and flow in
  two-dimensional systems*, Biophys. J. (MSD and its errors).
- Michalet (2010), *Mean square displacement analysis of single-particle trajectories with
  localization error*, Phys. Rev. E.
- Saxton (1997), *Single-particle tracking: the distribution of diffusion coefficients*,
  Biophys. J. (scatter of D from short tracks).
- Gorelik & Gautreau (2014), *Quantitative and unbiased analysis of directional persistence in
  cell migration*, Nature Protocols; Fürth (1920) for the persistent random walk.

## Build notes and deviations (2026-11-13, first build)

Built: the four core units `track-linking/`, `track-predict/`, `track-events/`, `track-analysis/`
(lec, quiz, exercise each; figures from `tools/figs/tracking/figs.ts`), the six demos `linkpairs`,
`linkmovie`, `kalman1d`, `kalmanlink`, `gapclose`, `msdplot` (each with presets and reset; the
demos with simulations keep their model in `model.ts` so solution numbers come from the demo code),
the five library files `lib/linking.ts`, `lib/kalman.ts`, `lib/tracksim.ts`, `lib/trackstats.ts`,
`lib/lineage.ts`, and `demos/test/tracking-test.ts`. Optional units `track-eval/` and
`track-learned/` (and the demo `trackeval`) are not built.

Deviations from the plan above:

- **Augmented matrix**: taught with a **zero** lower-right block (a link is taken iff
  $c_{ij} < d_i + b_j$); the math slide mentions Jaqaman's $C^\top$ block (same optimum).
- **Solver**: dense Hungarian on components (as planned) was far too slow when gates make large
  components (150 particles: > 1 s per movie). `assignGlobal` now uses a sparse row-by-row
  shortest-augmenting-path search on the rectangular problem with one private "end" column per
  row (only pairs with $c_{ij} < d_i + b_j$ kept): 30–100 ms for a 200-particle, 60-frame movie.
  `assignDense` (the augmented matrix with `eval.ts` `hungarian`) is kept for the teaching
  examples; both are checked against brute force on 400 random problems up to 5 × 5.
- **`matchPoints`** (gated optimal point matching, most pairs first) is in `lib/linking.ts`; the
  point-detection plan wants `matchPoints` in `lib/eval.ts` (evaluation's file): coordinate.
- **`linkmovie`**: no `data-source="image"` mode (frames rendered as spots and detected): left
  for a second pass. The sweep (error vs ratio) uses a gate of 3.5 and b of 2.5 rms steps at each
  ratio, not the slider values. Recompute is debounced when a relink takes > 30 ms.
- **`kalmanlink`**: scenes are built in the demo (crossing pairs, a 90° turn) or from
  `simParticles` (stream, waiting/running). Nearest neighbour resolves conflicts by giving the
  detection to the closest track. The fixed-gate tracker uses a 20 px gate by default; with it,
  global assignment on the last position beats the Kalman tracker in the "turn" (Q small) and
  "mixed" presets, which the units use as the lesson (a wrong model is worse than none).
- **`gapclose`**: no 2D-field view toggle (space–time view only); the cell strip is 400 × 40 px
  so that lineages spread along x; cells repel each other (no overlap). Border reasons use the
  x border only. A segmentation split cuts off a 35 % fragment (the larger part keeps the
  identity). The k-frame rule keeps a split whose branch is cut by the end of the movie.
  Split/merge cost = d² × max(ρ, 1/ρ)², ρ = mother area / sum of the two parts. No brightness cue.
- **`msdplot`**: per-track D from a line through the first 4 lags (the fit-range slider applies
  to the ensemble fit only). The "switching" type is available as a button, without a preset.
- **Exercise math numbering**: concept questions B1–B6, math questions continue the numbering
  (as in the built units), so the plan's B1–B15 are renumbered per unit.
- **Censoring exercise**: births uniform over the movie; the closed form is approximate (16.4 h)
  and the exact integral (16.8 h) is given too.
- No real data yet (open question 5): everything is simulated and labelled so.
- Pointer edits in other units (open question 7: `object-eval/`, `stitching/`,
  `structure-tensor/`) were not made (not owned by this day).
