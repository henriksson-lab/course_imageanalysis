# Plan: a day on networks for cells and objects (Part VII, day 40)

Internal working doc (plan/ is not rendered). Status: draft for building, 2026-11-13.
Follows `plan/segmodels-plan.md` ("Recommendation", "Exercises", "Demos"), which this plan
replaces as the day's `plan:` in `days.yml` once accepted, and `plan/coursemap.md` (decided
2026-11-13: self-study, **no electives**, one sequence of themed parts; this day **owns**
star-convex polygons, flow fields, single-shot detectors and promptable segmentation).

**Course map (2026-11-13, self-study restructure):** day 40 of 51, Part VII "Learning from
examples". Hard prerequisites: Modern deep learning (day 39), Evaluation and statistics (day 37).
Soft (see also): Watershed (day 15), Edge-preserving smoothing and PDEs (day 10), Graph cuts
(day 27). Source: `days.yml` (id `cellnets`).

Numbers in the by-hand exercises below were computed with
`<scratchpad>/goal3/cellnets/numbers.py` (session scratchpad
`/big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/51ed2e69-bf30-4fe9-bbf0-b7ac7e2bb101/scratchpad/goal3/cellnets/numbers.py`);
the builder copies the script to `tools/figs/<unit>/` (or re-derives the numbers) and keeps them
in sync. Timings and sizes come from `demos/wasm/*/README.md` (measured 2026-11-13). Numbers for
the **Explore** solutions with the real models are **not** in this plan: the builder measures
them with a script (node, as in `demos/test/*-wasm-test.ts`) and never guesses them.

## Goal of the day

After the day a student can:

1. say for each of four network families **what the network predicts** and **how objects are
   decoded** from it: StarDist (objectness + ray lengths → polygons → polygon NMS), Cellpose
   (flows + cell probability → following the arrows → sinks → masks → flow check), YOLO (per grid
   cell: box-side distances + class scores → boxes → NMS), SAM (image embedding once, prompt →
   three candidate masks with scores);
2. run each decoding **by hand** on a small example: rays → polygon and its area; one step of
   the heat target and a few steps along a 5 × 5 flow field; one YOLO output cell → a box in
   pixels → its IoU; greedy NMS on scored boxes/polygons; pick SAM's mask for an ambiguous click;
3. predict what each **knob** of the decoding does (probability/NMS thresholds; diameter, flow
   threshold, number of steps; score threshold, class-aware vs class-agnostic NMS, input size;
   prompt placement, negative clicks, automatic-mask filters) and check the prediction on the
   real model;
4. name each family's **failures by construction** (non-star-convex shapes; wrong diameter,
   noisy flows; small objects and missing classes; prompts that select parts or clusters, domain
   shift) and find them on real images;
5. **compare** a classical pipeline and the networks on the same annotated images with the
   evaluation day's object matching, and weigh accuracy against cost (parameters, time per image
   in the browser, annotation needed);
6. *(math)* derive the polygon area from ray lengths, the flow as the normalised gradient of a
   diffusion and why paths end in sinks, the expectation decoding and loss of a distance
   distribution, an IoU-type box loss that has a gradient for disjoint boxes, and why caching the
   image embedding makes each prompt cheap.

Why it fits this audience: the theory per model is thin, but **representation and decoding** are
geometry, PDEs and simple probability the students know (star-shaped sets, the heat equation,
gradients and ODE paths, softmax expectations, IoU). The day turns "a black box that outputs
masks" into four concrete, checkable pipelines, and the real networks run in the browser, so
every claim can be tested on real micrographs.

Naming: the four model families are **methods** (like Canny), so their names are used as titles.
What is avoided (principle 0) is any tool or library interface: no installation, no function
names, no code sections; at most one line "software packages with the same name implement this"
per unit.

## Prerequisite gap

What earlier units already teach (checked in the built units) and what this day recalls in one
line each:

- **`instances/`** (day 39, built): why post-processing is unavoidable, the four routes with
  simulated outputs (`instancemaps`), star-convexity defined (math slide "the four targets"),
  flows as the gradient of heat diffused inside the cell, following the flow as Euler steps, the
  by-hand 1D flow (B4), the regular n-gon coverage (B9: 0.900 for n = 8, 0.994 for n = 32),
  choosing a route, panoptic quality. After the slimming (below) routes 3–4 are one pointer slide
  each, and their detail lives here.
- **`detection/`** (day 39, built): boxes, IoU in closed form, 1 − IoU has no gradient for
  disjoint boxes (math), greedy NMS and its failure in crowds (`nmsboxes`), three meanings of
  NMS. Anchors and centre heat maps move here (unit 3).
- **`attention/`** (day 39, built): patches as tokens, self-attention, transfer learning,
  self-supervised pretraining, the one-slide "promptable segmentation" in miniature (click →
  similarity → mask, `embedprobe`). Unit 4 recalls it and goes further.
- **`unet/`** (day 36): encoder–decoder, pixel probabilities, receptive field. StarDist and
  Cellpose are U-Nets with extra output channels: one line.
- **`object-eval/`** (day 37): IoU of objects, one-to-one matching at τ, TP/FP/FN, F1, F1(τ),
  splits/merges, AP (two meanings). Units 1–4 use F1 at τ = 0.5 as a readout; unit 5 uses all
  of it. `lib/eval.ts` is the shared code.
- **`ws-distance/`** (day 15): distance transform + h-maxima + watershed. Unit 5's classical
  pipeline is exactly that pipeline.
- **`heat-equation/`** (day 10): the heat equation as repeated averaging, explicit steps.
  Unit 2's flow targets are "heat diffused inside the cell" — recalled, not re-derived.
- **`graphcut-seg/`** (day 27): the `graphcut` demo on the "nucleus" micrograph preset is the
  interactive classical comparison for SAM's clicks (a pointer in unit 4 and unit 5).
- **Hough peaks** (day 21) and **Canny NMS** (day 2): recalled once in unit 3 (the three
  meanings of NMS, now a fourth: polygon NMS in unit 1).

Nothing new is needed from outside; the only new mathematics is the distance distribution
(softmax expectation) of unit 3 and GIoU, both short.

## Overlaps with other units (who takes what)

| Unit | What moves here | What stays there |
|---|---|---|
| `instances/` Route 3 (+ math "the four targets", star-convex part; exercise B3, B7, B9; Explore E5) | Rays, objectness, polygon from rays, polygon NMS, star-convexity and its failures, n-gon coverage | One **pointer slide** "Star-convex polygons" (fig-star.png, three lines, link to `stardist/`); the `instancemaps` demo keeps its star route and `cshape` preset; the "choosing a route" table and the crowded-tissue comparison (E8) |
| `instances/` Route 4 (+ math "following the flow", "Try it: flows follow the bend"; exercise B4, B8; Explore E6, E7; quiz Q6) | Heat target, flows, following the flow, sinks, noisy flows | One **pointer slide** "Flow fields" (fig-cflow.svg, three lines, link to `cellpose/`); the `instancemaps` flow route and its presets (`flows`, `noisyflows`) are embedded again in `cellpose/` as the warm-up |
| `detection/` "Where candidates come from: anchors", its math, "centre heat maps", its math (exercise B3, B4, B8; Explore E7, E8; quiz Q5, Q6) | Anchors, box encoding relative to an anchor, focal loss, heat-map targets and peaks — as the history of unit 3's grid head | One **pointer slide** "Where do the candidates come from?" (a grid of guesses, one line each on anchors and centre maps, fig-anchors.png, link to `yolo/`); `nmsboxes` keeps its `anchor`, `heat`, `noisyheat` presets, now embedded from `yolo/` |
| `attention/` "Promptable segmentation" | The real promptable segmenter, three masks, scores, automatic masks, memory | The slide stays as the miniature, with one line: "the real one, and how it decodes: the next day, `sam/`" |
| `object-eval/` | nothing (used) | Matching, F1, AP; unit 5 recalls them in one slide |
| `ws-distance/`, `graphcut-seg/` | nothing (used) | Unit 5 runs the same classical pipeline on the same real images |
| `diffusion/` (day 39) and day 41 | nothing | Cellpose's "diffusion" is heat diffusion (a PDE), **not** a diffusion model: one disambiguation line in unit 2 |

## Units (one folder each: lec, quiz, exercise with #explore)

Five units, about 1.5 h each including the exercise (unit 5 about 2 h: it runs the networks).
Order: StarDist first (simplest decoding, closest to `instances/`), Cellpose (the PDE route),
YOLO (from maps to a list), SAM (prompts instead of a fixed output), then the comparison.

Every deck: "About this lecture" ending with the math-layer line (README "Marking the math
layer"); about half the slides with a figure **next to** text (two columns); figures from
`tools/figs/<unit>/lec_figs.py` (synthetic) or demo screenshots (`tools/figs/<unit>/demo_shot.sh`,
as in `tools/figs/instances/`), real-image figures credited. On slides: `![](f.svg){fig-alt="..."}`
and `.nostretch` where needed (build-issues lessons). Every real-model embed keeps a static
screenshot as the fallback (the `file://` message and print), as `demos/README.md` requires.
Slide counts are guides (principle 6): 16–22 content slides per deck.

### 1. `stardist/`: Star-convex polygons

Concept layer (★ = slide with a figure):

- ★ **Recall** (one slide): the semantic mask merges touching nuclei; `instances/` route 3 in one
  picture (fig-star.png reused). The question: how exactly does one pixel describe a whole
  nucleus, and how do 100 000 pixel guesses become 120 nuclei?
- ★ **What the network predicts** per pixel (on a coarser grid, one value per 2 × 2 pixels):
  an **objectness** (how central the pixel is in its nucleus: 1 at the centre, 0 at the border
  and outside) and **32 distances** along fixed directions to the nucleus border. Figure: one
  nucleus, a central pixel with its 32 rays, an off-centre pixel with its rays. Pointer: a
  U-Net (day 36) with 1 + 32 output channels.
- ★ **Rays → polygon**: join the 32 ray tips in order. Worked example: an 8 × 4 rectangle seen
  from its centre with 8 rays: lengths 4, 2.83, 2, 2.83, 4, 2.83, 2, 2.83; polygon area 24.0 of
  32 (IoU 0.75: the corners are cut); with 32 rays: 31.3 (IoU 0.98). Figure: the rectangle with
  the 8-ray and 32-ray polygons.
- ★ **Which shapes fit**: a shape is *star-convex* if one point sees its whole border along
  straight lines. Discs, ellipses, most nuclei: yes. An L from its corner: yes; from an arm's end:
  no. A C-shape, a ring, a bent rod: no point works. Figure: four shapes, the best centre and its
  rays, missed parts hatched.
- **Why objectness, not a plain "nucleus" probability**: every pixel inside proposes a polygon,
  but border pixels see their own nucleus badly (their rays are short on one side). Ranking by
  "how central" puts the good proposals first, and it is the distance map of `ws-distance/`,
  normalised per nucleus.
- ★ **Decoding**: (1) keep grid cells with objectness above a **probability threshold**; (2) each
  gives a polygon; (3) sort by objectness; (4) **polygon NMS**: keep the best, delete every
  polygon that overlaps it by more than the **NMS threshold** (IoU of polygons), repeat;
  (5) paint the kept polygons into a label image, best last or best first (overlaps resolved by
  score). Figure: candidates (thousands of overlapping polygons) → after NMS → labels, on the
  real nuclei image (stardistnet screenshot, 4737 candidates → 121 nuclei).
- ★ **The two thresholds** as a 2 × 2 picture: probability low → specks and dim debris;
  probability high → dim nuclei lost; NMS overlap too low → touching neighbours deleted; too high
  → duplicates survive. The trained model ships with a tuned pair (chosen on validation images).
- **Why touching nuclei separate**: two touching nuclei have two objectness peaks and two
  polygons that overlap only along the contact (IoU small), so NMS keeps both. Contrast with
  boxes in `detection/` (boxes of touching rods overlap a lot).
- ★ **Failures by construction**: C-shapes, rings and long bent cells (split into pieces or cut
  off); very elongated nuclei at a fixed number of rays (coarse tips); objects much larger than
  what the network saw (rays only reach as far as the receptive field). Figure: `instancemaps`
  `cshape` preset (simulated) next to a real failure found with stardistnet.
- **Failures of the training data**: a model trained on fluorescence nuclei given an H&E or
  bright-field image (dark nuclei on bright background). Real-model figure: histology preset.
- **Try it** (stardistnet, `data-image="nuclei" data-view="cand"`), then a second embed
  (`data-preset="strong"`) in the notes tour.
- Where this leads: 3D (rays on a sphere, same decoding), other ray-based shapes; the
  comparison in unit 5. Pointer: "software with the same name implements this method".

Math layer:

- *Star-convexity*: definition, every convex set is star-convex (recall `instances/`), the
  ray-length function $r(\theta)$ is single-valued exactly when $O$ is star-convex w.r.t. $c$.
- *Polygon area from rays*: $A = \tfrac12\sin\tfrac{2\pi}{n}\sum_k r_k r_{k+1}$ (sum of
  triangles); the $n$-gon coverage of a disc $\tfrac{n}{2\pi}\sin\tfrac{2\pi}{n}$ (recall);
  ellipse 6 × 3 (semi-axes) from its centre: 85.4 % (n = 8), 95.5 % (16), 98.8 % (32).
- *Training targets and loss*: objectness $d(p) = $ normalised distance to the background; ray
  target $r_k(p)$ = first exit along $\mathbf u_k$; loss = binary cross-entropy on the objectness
  + objectness-weighted absolute error on the rays (central pixels count more).
- *Polygon NMS cost*: $O(m^2)$ polygon IoUs for $m$ candidates; the bounding-box pre-check
  (two polygons whose boxes do not overlap cannot overlap) and why sorting matters (greedy keeps
  the first). Greedy NMS only uses the order of scores (recall `detection/` B9).

**Demo `starrays`** (new, analytic, TypeScript): two modes.
*Shape mode*: a shape on a 64 × 64 grid (presets: disc, ellipse 2:1, rectangle 8:4, L, C, ring,
bent rod; drawable with `lib/shapecanvas.ts`), its objectness map, a draggable centre with
*n* rays (4–64) drawn to their first exit, the polygon, its area vs the shape's area and IoU,
missed and extra pixels hatched; a toggle "best centre" (the centre maximising the IoU).
*Decode mode*: a synthetic nuclei scene with ground truth (`lib/synth.ts` `nuclei`), simulated
network output = exact targets + noise (as `instancemaps`), probability threshold, NMS threshold,
**step** through NMS (kept polygon outlined, deleted ones greyed with their IoU), labels, and
F1/TP/FP/FN at τ = 0.5 (`lib/eval.ts`). Presets: "corners cut (8 rays)", "32 rays", "C: no centre
works", "L: centre matters", "touching pair survives", "NMS too strong: neighbour deleted",
"NMS too weak: duplicates", "low threshold: specks". Shared code: ray targets, polygon from
rays, polygon pixels and polygon IoU go into a new `lib/instances.ts` (moved out of
`instancemaps/model.ts`, which then imports them; behaviour unchanged, tests in
`demos/test/instances-test.ts`), NMS via `nmsGeneric` from `lib/boxes.ts`.

Real model (Explore): **`stardistnet`** (built; options `data-image`, `data-view`,
`data-preset="trained|low|high|weak|strong"`, image scale select, click a pixel to see its 32
rays). Small extension wanted (owner of `stardistnet`): an F1 readout against ground truth for
the images that have labels (`dsb-fluo-*`, `bbbc039-*`, `dsb-histology`, `dsb-brightfield`),
exactly as `cellposenet` does (`iouTable`/`match`/`objectScores` from `lib/eval.ts`).

By hand (concept):

- **B1.** The 8 rays of an 8 × 4 rectangle from its centre (directions every 45°): compute the
  lengths (4, 2.83, 2, 2.83, …), sketch the polygon, and its area (24.0, the rectangle 32,
  IoU 0.75). Where is the area lost? How would 32 rays do (31.3, IoU 0.98)?
- **B2.** Rays 5, 4, 3, 4, 5, 4, 3, 4 (every 45°): sketch the shape, compute its area with the
  triangle rule (45.3). Which ray would change if a small bump grew on the right side?
- **B3.** Star-convex or not, and from where: disc, L-shape, five-pointed star, C-shape, ring, a
  dumbbell (two discs joined by a thin neck). For the L, draw the 8 rays from the corner and
  from an arm's end. (Moved from `instances/` B3, extended.)
- **B4.** Polygon NMS by hand. Six candidate polygons with objectness P1 0.92, P2 0.85,
  P3 0.80, P4 0.62, P5 0.55, P6 0.35 and IoU table P1–P2 0.71, P1–P3 0.18, P1–P4 0.05, P1–P5
  0.33, P2–P3 0.22, P2–P5 0.41, P3–P4 0.46, P3–P6 0.12, P4–P6 0.58 (others 0). The truth: four
  nuclei, A (P1, P2), B (P3), C (P4, P6), D (P5). Run greedy NMS for (a) probability ≥ 0.5,
  NMS 0.3: kept P1, P3 — C and D lost (P5 deleted by P1, P4 by P3); (b) ≥ 0.5, NMS 0.5: kept
  P1, P3, P4, P5 — all four; (c) ≥ 0.3, NMS 0.7: P1, P3, P4, P5, P6 — P6 a duplicate (FP).
  (d) Which setting would you choose, and what does it say about touching nuclei vs duplicates?
- **B5.** Predict: a bent bacterium (a quarter ring). How many objects will StarDist return, and
  what will they look like? What about a nucleus twice as large as any in training?

Math questions:

- **B6.** *(math)* Derive $A = \tfrac12\sin\tfrac{2\pi}{n}\sum_k r_k r_{k+1}$ and show that for
  a disc seen from its centre it gives the inscribed regular n-gon (recall `instances/` B9).
- **B7.** *(math)* Show that the polygon from a star-convex shape's exact rays lies inside the
  shape when the shape is convex, and give a non-convex star-convex shape where the polygon
  sticks out (a five-pointed star, rays between the tips).
- **B8.** *(math)* The ellipse with semi-axes 6 and 3 seen from its centre: ray length
  $r(\theta) = ab/\sqrt{b^2\cos^2\theta + a^2\sin^2\theta}$; compute the polygon area for n = 8
  (48.3 of 56.5: 85.4 %).
- **B9.** *(math)* Why does weighting the ray loss by the objectness make sense? What would go
  wrong with an unweighted loss at border pixels?

Explore (stardistnet; predict first, then run):

- **E1.** *Nuclei*, view **Probability**. Predict where it is highest and what it looks like
  between two touching nuclei; check. Then **Ray length** at 0° and 180°: predict which side of
  each nucleus is bright.
- **E2.** Click a central pixel and a border pixel of one nucleus: compare the 32 rays.
- **E3.** View **Candidates**, then **After NMS**: count both (the readout). Presets *low* and
  *high*: predict which objects appear or disappear, then check (and the F1 readout, once added).
- **E4.** Presets *weak* and *strong* suppression on the *dense* image: where do duplicates and
  lost neighbours appear?
- **E5.** Failure hunt: *histology* and *brightfield*, then **scale** 0.5× and 2× on *nuclei*.
  Describe each failure and explain it with "what the network saw in training".
- **E6.** Your own image (local file, never uploaded): a rotated or mirrored copy of *nuclei*
  — same objects? Then find one shape StarDist cannot represent.
- **E7.** `starrays` decode mode: reproduce B4's three settings on the scene presets.
  (`instancemaps` preset `cshape` moves here from `instances/` E5 as **E8**.)

Quiz (6–8): what is predicted per pixel; why objectness ranks the candidates; which shape is not
star-convex; NMS threshold too low / too high; why touching nuclei survive NMS; *(math)* polygon
area from 4 equal rays $r$ (a square of area $2r^2$).

### 2. `cellpose/`: Flow fields to objects

Concept layer:

- ★ **Recall** (`instances/` route 4 in one picture, fig-cflow.svg): arrows inside each cell that
  lead to one point. Today: where the arrows come from, how they are followed, and what decides
  how many cells come out.
- ★ **The training target, step by step**: put heat at the cell's centre and let it spread
  **inside the cell only** (repeated averaging over neighbours that belong to the cell, the
  heat-equation day). Worked in 1D on 5 pixels with the source in the middle, three steps of
  "add 1 at the source, then average with the neighbours inside the cell": (0, .33, .33, .33, 0),
  (.17, .56, .67, .56, .17), (.36, .80, .93, .80, .36). The arrows point uphill: →, →, sink, ←, ←.
  Figure: a C-shaped cell with the heat after 10, 50, 200 steps and the arrows.
- **Why heat, not straight lines to the centre**: the heat spreads only through the cell, so the
  uphill direction bends around corners: a C-shape gets arrows that follow the arc. Straight
  arrows to the centre would leave the cell (this is StarDist's limitation in another guise).
  One disambiguation line: this "diffusion" is the heat equation, not the generative diffusion
  models of day 41.
- ★ **What the network predicts**: per pixel **two numbers** (the arrow: up–down and left–right
  component, colour wheel) plus a **cell probability**. Figure: cellposenet screenshot,
  *Flows* view on the cells image next to the input.
- ★ **Decoding: follow the arrows**. Every pixel with cell probability above the threshold is a
  particle; it moves one pixel along the arrow at its position, a fixed number of steps
  (default about 200, scaled with the diameter). Particles gather at **sinks**; a histogram of
  end points finds them; each sink is grown to a mask from the particles that arrived there.
  Figure: *Track* view (paths, sinks, seeds).
- ★ **By hand on a grid**: the 5 × 5 field of B2 (figure: arrows on a grid with three paths).
- ★ **The flow check**: from each mask, compute the arrows it *should* have (the heat target of
  that mask) and compare with the predicted arrows; if the mean squared difference is above the
  **flow threshold** (default 0.4), the mask is dropped as implausible. Worked: four pixels,
  differences 0, 0.4, 0, 4 → mean 1.1 → dropped. Figure: the *Check* view.
- ★ **Cell diameter and rescaling**: the network was trained on cells of one size (30 px for the
  cytoplasm model, 17 px for the nuclei model). The image is rescaled so that its cells have that
  size: cells of 45 px → factor 0.667 (600 × 450 → 400 × 300). A wrong diameter: too small a
  factor → cells too small for the network, merges and misses; too large → each cell looks like
  a cluster, splits. The diameter can be estimated from the image (a small regression on the
  network's style vector: "estimate diameter" in the demo).
- **One network for many image types** ("generalist"): trained on many kinds of cells and
  microscopes, with the image normalised by percentiles and up to two channels (cytoplasm,
  nucleus). Why that generalises better than one network per dataset, and where it still fails.
- ★ **Failures**: noisy arrows make extra sinks (splits); two touching cells with one smooth field
  (merge); too few steps (particles not yet arrived: splits, recall `instances/` E6); dim cells
  below the probability threshold; wrong diameter. Figure: the "Only 10 steps" preset vs default.
- **Try it**, warm-up: `instancemaps` `data-preset="flows"` (simulated, moved from
  `instances/`), then `cellposenet` on the *cells* sample.
- Pointer (one slide or a bullet): newer versions replace the U-Net encoder by a large
  pretrained transformer (the SAM image encoder of unit 4) and drop the need to give a diameter;
  too large for this site (manifest note: ~620 MB even in f16).

Math layer:

- *Heat target*: $h$ solves $\partial_t h = \Delta h + \delta_c$ in $O$ with no flux across
  $\partial O$; discretely $h \leftarrow \text{mean}_{N(p) \cap O} h$ + source; flow
  $\mathbf v = \nabla h/\|\nabla h\|$ (recall `heat-equation/`).
- *Following the flow* (moved from `instances/`): Euler steps
  $\mathbf x_{j+1} = \mathbf x_j + \mathbf v(\mathbf x_j)$, $h$ increases along paths, so paths
  end at the maximum; noise breaks the gradient structure and creates fixed points; bilinear
  interpolation of $\mathbf v$ between pixels.
- *The loss*: squared error on the arrows (scaled ×5) + binary cross-entropy on the cell
  probability.
- *Flow check*: $\text{err}(M) = \frac{1}{|M|}\sum_{p\in M}\|\mathbf v_\text{pred}(p) - \mathbf v_M(p)\|^2$
  (up to the scaling used in practice); keep $M$ if $\text{err}(M) \le$ threshold.
- *Scale*: convolutions are translation- but not scale-equivariant; rescaling by
  $d_\text{train}/d$ puts the cells into the size range the filters were trained on; the number
  of Euler steps must exceed the longest path (∝ diameter).

**Demo `flowtrack`** (new, analytic): a mask on a 48 × 48 grid (presets: disc, C, bent rod, two
touching cells, dumbbell; drawable). Panels: the heat $h$ after *k* averaging steps (slider,
play), the arrows (every 2nd pixel) with the colour wheel, the particles (play / step, number of
steps, step size), the end-point histogram with its sinks and the resulting masks; error sliders
(noise on the arrows, blur, "one shared field for touching cells"); the flow check (threshold
slider, per-mask error printed, dropped masks hatched). A **grid mode** shows the 5 × 5 field of
B2 as arrows in cells (`lib/gridui.ts` `GridView` with arrows) so the by-hand answer can be
checked. Presets: "heat spreads", "arrows follow the bend", "too few steps", "noise: extra
sink", "flow check drops a bad mask", "shared field: merge", "B2 grid". Shared code: heat target
and flow following go into `lib/instances.ts` (with `instancemaps`' `flowTarget`/`flowEnds`).

Real model (Explore): **`cellposenet`** (built; samples *cells*, *crowded*, *membranes*,
*nuclei*, *dense*, *small*, *tissue* — the last four with ground truth and an F1 readout; views
input / flows / prob / track / masks / check; sliders diameter, cell probability threshold,
flow threshold, minimum size, Euler steps; presets Defaults / More cells / Stricter / No flow
check / Only 10 steps; "Estimate diameter"; click pixels to trace their paths).

By hand (concept):

- **B1.** The 1D heat target on 5 pixels (source in the middle): do the three steps above, then
  give the arrow direction at each pixel. What changes if the cell is 9 pixels long and you stop
  after 3 steps? (The ends are still at 0: no direction yet — why many steps are needed.)
- **B2.** Follow the 5 × 5 field (rows top to bottom; `.` = no movement):

  ```
  →  ←  ↓  ←  ←
  →  ↘  ↓  ↙  ←
  →  →  .  ←  ←
  ↗  ↑  ↑  ↘  ↓
  ↗  ↑  ↑  →  .
  ```

  from (row 0, col 0), (4, 0) and (3, 3). Answers: (0,0) → (0,1) → (0,0): stuck between two
  pixels, a **spurious sink**; (4,0) → (3,1) → (2,1) → (2,2): the centre; (3,3) → (4,4): a second
  sink. Group all 25 pixels: 19 end at (2,2), 4 at (4,4), 2 oscillate in the top-left corner.
  How many objects, and which one is an error? Which single arrow fixes it?
- **B3.** Flow check: four pixels of a mask, predicted arrows (1, 0), (0.8, 0.6), (0, 1), (−1, 0),
  arrows recomputed from the mask (1, 0), (1, 0), (0, 1), (1, 0). Squared differences 0, 0.4, 0,
  4: mean 1.1 > 0.4: dropped. If the last predicted arrow were (0.6, 0.8): mean 0.3: kept.
- **B4.** Diameter: cytoplasm model trained at 30 px. Cells of 45 px in a 600 × 450 image: the
  factor and the size the network sees (0.667, 400 × 300). You type 15 instead: factor 2, the
  network sees 1200 × 900 and cells of 90 px. Predict the error (splits) and the cost (4× the
  pixels). Nuclei model (17 px) on nuclei of 34 px in 696 × 520: factor 0.5, 348 × 260.
- **B5.** (moved from `instances/` B4) The 1D flow with one wrong arrow.
- **B6.** Predict: which is worse for Cellpose, a C-shaped cell or two round cells touching along
  a long edge? And for StarDist?

Math questions:

- **B7.** *(math)* (moved from `instances/` B8) $h$ increases along every path of
  $\dot{\mathbf x} = \nabla h/\|\nabla h\|$, so paths end at the source.
- **B8.** *(math)* Show that the steady state of "average with in-cell neighbours, add a constant
  at $c$" is discrete-harmonic away from $c$ and has its only maximum at $c$ (discrete maximum
  principle). Why does that guarantee one sink per cell for the exact target?
- **B9.** *(math)* Two arrows pointing at each other at neighbouring pixels: show that a Euler
  path with step 1 oscillates between them, and that with bilinear interpolation it stops at the
  point between them (a fixed point of $\mathbf v$). Relate to negative divergence.
- **B10.** *(math)* Why is the number of Euler steps scaled with the diameter? Estimate the
  longest path in a disc of diameter $d$ and in a C-shape of arc length $L$.

Explore (cellposenet; predict first):

- **E1.** `instancemaps` warm-up (moved from `instances/` E6/E7: presets `flows`, `noisyflows`).
- **E2.** *cells* sample, run, view **Flows**: predict the colour pattern in one cell (colour
  wheel) before looking; then **Track**: click three pixels in one cell.
- **E3.** Preset *Only 10 steps*: predict the number of cells (more or fewer?) and check; why?
- **E4.** *No flow check*, then flow threshold 0.2 / 0.8: which masks appear or vanish, and what
  do the vanished ones look like in the *Check* view?
- **E5.** *nuclei* (ground truth): note F1; set the diameter to half and to double the default:
  predict, run, explain with the rescaling. Then *Estimate diameter*.
- **E6.** Cell probability threshold −2 and +2 on *small*: which errors change (FN dim nuclei,
  FP specks)? Read TP, FP, FN.
- **E7.** Failure hunt: *tissue* (with and without *invert*), *membranes* (one channel), your
  own image (local file). Find one split and one merge and explain each from the Flows view.

Quiz: what the two flow channels mean; why heat and not straight lines; what a sink is; what the
flow check compares; what a wrong diameter does; *(math)* along a path $h$ only increases — why.

### 3. `yolo/`: From an image to a list of objects

Concept layer:

- ★ **Recall** (`detection/`): boxes, IoU, greedy NMS; the question left open there: *where do
  the candidate boxes come from, and how does one network output a list of any length?*
- ★ **The grid**: the network's last feature maps are coarse grids (the image divided into cells
  of 8, 16 and 32 pixels: **strides**). For a 512 × 512 input: 64², 32² and 16² cells, 5376 in
  total (640: 8400). Every cell makes **one guess**: "is there an object whose box contains my
  centre, and where are its sides?" A fixed number of guesses, most of them "nothing" — that is
  how a fixed-size output describes any number of objects. Figure: the three grids over an image.
- ★ **What each output cell predicts** (current, anchor-free heads): four **distances** from the
  cell centre to the left, top, right and bottom side of the box (in units of the stride), and a
  **score per class** (80 for the generic photo model). Worked: stride 16, row 5, column 7 →
  cell centre (120, 88) px; distances 2.5, 1, 1.5, 2 → box x 80–144, y 72–120. Figure: the cell,
  its centre, the four distances, the box (yolonet "selected box" screenshot).
- ★ **Distances as distributions**: each distance is predicted as 16 probabilities over 0, 1,
  …, 15 strides, and the decoded distance is their **mean**. A sharp peak = a confident edge; a
  wide or two-peaked distribution = an uncertain edge (an occluded or blurry side). Example: half
  at 2, half at 3 → 2.5; 0.6 at 2 and 0.4 at 9 → 4.8, a side where neither guess is.
  Consequence: a box side can be at most 15 strides from the cell centre (120 px at stride 8,
  240 at 16, 480 at 32): large objects must come from the coarse grid.
- ★ **Scores and classes**: per class an independent probability (sigmoid: logit 1.2 → 0.77,
  −0.5 → 0.38, −3 → 0.05); a candidate = a cell and class with probability above the **score
  threshold**. Figure: score maps per stride (yolonet).
- ★ **Three scales (feature pyramid)**: fine grid for small objects, coarse grid for large; the
  coarse maps are upsampled and merged with the fine ones so small-object guesses also "see"
  context. Figure: which stride found which object in the bus photo.
- **Decoding = candidates → NMS**: thousands of cells above a low threshold, many per object
  (neighbouring cells and scales all see it) → greedy NMS per class (or across classes) → the
  list. Class-aware NMS keeps two boxes of different classes on the same object; class-agnostic
  removes them.
- ★ **Before: anchors and centre heat maps** (moved from `detection/`, condensed to one concept
  slide + one math slide): older heads predicted corrections to template boxes (anchors);
  objects matching no anchor were never learned (fig-anchors.png). Centre heat maps (peaks =
  objects; fig-heat.png) are the other anchor-free route; current single-shot heads keep the
  grid but drop the templates.
- **Training in one slide**: which cells are responsible for an object (cells whose centre lies
  in the box and whose prediction is already good: the assignment), the losses: IoU-type loss on
  the box, the distribution loss on each side, a classification loss (most cells are
  background: down-weighting of easy negatives, recall the focal loss).
- **Speed**: one pass of the network, no per-object work except NMS: why this family is used for
  video and live microscopy. Real numbers: the nano model, 2.6 M parameters, 0.7–0.8 s for 512²
  on one CPU core in the browser; 0.18 s for 256².
- ★ **Boxes for cells**: good for counting, tracking, cropping; poor for touching, elongated,
  diagonal cells (recall `detection/` rod: 16 % of its box). Real-model figure: on a micrograph
  the generic photo model finds nothing at the normal threshold, and at 0.01 calls nuclei
  "person" or "sports ball": **no class "cell" exists** — the strongest out-of-domain example of
  the day. A cell detector needs fine-tuning on annotated cells (with a box per cell; open
  question 2).
- **Try it** (yolonet `data-preset="raw"`, then `data-preset="nucleilow"`).
- Where this leads: a mask head inside each box (instance segmentation in one pass); tracking by
  detection (tracking day). Pointer: "software with the same name implements this family".

Math layer:

- *Decoding a cell*: $(x_1, y_1, x_2, y_2) = s\,(j + \tfrac12 - d_l,\ i + \tfrac12 - d_t,\ j + \tfrac12 + d_r,\ i + \tfrac12 + d_b)$,
  $d = \sum_{k=0}^{15} k\,p_k$, $p = \mathrm{softmax}(z)$; class probability $\sigma(z_c)$.
- *Distribution loss*: for a target $y \in [k, k+1]$, $-\big((k+1-y)\log p_k + (y-k)\log p_{k+1}\big)$;
  its minimum puts mass $k+1-y$ on $k$ and $y-k$ on $k+1$, whose mean is exactly $y$ (target 2.3:
  0.7 on 2, 0.3 on 3).
- *IoU-type box loss*: $1 - \mathrm{IoU}$ has no gradient for disjoint boxes (recall
  `detection/` B7); GIoU $= \mathrm{IoU} - |C \setminus (A\cup B)|/|C|$ with $C$ the smallest
  enclosing box: $A = [0,4]^2$, $B = [6,10]\times[0,4]$: IoU 0, GIoU −0.2; move $B$ one to the left:
  −0.11 — the loss now pulls the boxes together. (Current heads use a variant with a centre
  distance and aspect term; named in one line.)
- *Anchors* (moved from `detection/`): $t_x, t_y, t_w = \log(w/w_a), t_h$ and why the log.
- *Output size*: $A = \sum_s (S/s)^2$ cells × (4 × 16 + C) numbers (512, 80 classes: 5376 × 144).

**Demo `yolodecode`** (new, analytic): a 128 × 128 synthetic scene (cells, a rod, debris; ground
truth boxes) with the three grids (strides 8, 16, 32 → 16², 8², 4² cells) drawn on demand. Click a
cell: its centre, its four 16-bin distance distributions as bar charts (drag a bar, or set a
peak and a width per side), class logits for two classes ("cell", "debris") → the decoded box,
its expectation per side, IoU with the nearest true box. "Simulate the network": every cell's
output from the exact training assignment + noise (sliders), score threshold, NMS threshold,
class-aware / agnostic toggle, step-through NMS, readouts TP/FP/FN/F1 at τ = 0.5 (`lib/eval.ts`,
boxes via `lib/boxes.ts`). An "anchors" toggle switches the head to the older anchor encoding
(reusing `encodeBox`/`decodeBox`/`anchorGrid` from `lib/boxes.ts`). Presets: "B1 cell", "two
peaks: an uncertain side", "too large for stride 8", "small cell: only stride 8 sees it",
"class-aware keeps the debris box", "agnostic removes it", "crowded rods". The distribution
decoding reuses `softmax`/`decode` of `yolonet/post.ts` (moved to `lib/boxes.ts` or a new
`lib/detect.ts`, so both demos share it; `demos/test/yolo-wasm-test.ts` keeps passing).

Real model (Explore): **`yolonet`** (built; images bus, people, nuclei, u2os, cells, histology,
brightfield; presets default / raw / nonms / strict / small / nuclei / nucleilow; per-stride
checkboxes; score threshold; NMS IoU; input size; class-agnostic toggle; select a box to see its
grid cell and four distributions). `nmsboxes` presets `anchor`, `heat`, `noisyheat` are embedded
here (moved from `detection/` exercise E7, E8).

By hand (concept):

- **B1.** Decode one cell: stride 16, row 5, column 7 of the grid, distributions for (left, top,
  right, bottom): half on 2 and half on 3; all on 1; half on 1 and half on 2; all on 2. Give the
  cell centre in pixels (120, 88), the distances (2.5, 1, 1.5, 2), the box (80, 72)–(144, 120) and
  its IoU with the true box (84, 70)–(150, 122): 0.79. Class logit 1.2: probability 0.77.
- **B2.** How many output cells for inputs 256, 512 and 640 (1344, 5376, 8400), and how many
  numbers with 80 classes (× 144)? A box 400 px wide centred on a cell needs 200 px per side:
  which strides can describe it (16 and 32: at most 15 strides per side = 120, 240, 480 px)?
- **B3.** NMS by hand with classes. Boxes (x1, y1, x2, y2, score, class): b1 (10,10,50,50) 0.90
  cell; b2 (14,12,54,52) 0.80 cell; b3 (40,10,80,50) 0.75 cell; b4 (12,10,52,50) 0.60 debris;
  b5 (44,14,84,54) 0.50 cell; b6 (100,100,120,120) 0.20 cell. Score threshold 0.25, NMS 0.5.
  IoUs: b1–b2 0.75, b1–b4 0.90, b2–b4 0.82, b3–b5 0.68, b1–b3 0.14, others below 0.2.
  Class-aware: kept b1, b3, b4 (b2 removed by b1, b5 by b3, b6 below threshold); agnostic: b1,
  b3 (b4 removed by b1). Which answer is right depends on whether a cell can also be debris —
  discuss. With NMS 0.2 the result is the same here: why? (Among boxes still present when they
  are compared, no IoU lies between 0.2 and 0.5; b2–b3 is 0.199, and b2 is gone by then.)
- **B4.** (moved from `detection/` B3) Anchors: which object sizes an anchor can learn.
- **B5.** (moved from `detection/` B4) Peaks of a centre heat map.
- **B6.** Predict: two cells 6 px apart, each 8 px wide, at stride 8 and at stride 32. How many
  can the grid report?

Math questions:

- **B7.** *(math)* Show that the minimiser of the distribution loss for target $y = 2.3$ has mean
  2.3, and that a one-hot target at the nearest integer would bias every distance.
- **B8.** *(math)* Compute GIoU for the disjoint pair above and show it increases as $B$ moves
  towards $A$, while 1 − IoU stays 1.
- **B9.** *(math)* (moved from `detection/` B8) Why $\log(w/w_a)$.
- **B10.** *(math)* Is every box kept by class-agnostic NMS also kept by class-aware NMS (same
  threshold)? Prove or give a counterexample. (Counterexample: c debris 0.9 (0,0)–(40,40), a cell
  0.8 (10,0)–(50,40), b cell 0.7 (20,0)–(60,40); IoU c–a 0.6, a–b 0.6, c–b 0.33; t = 0.5:
  class-aware keeps c, a; agnostic keeps c, b — a box removed early can no longer remove others.)

Explore (yolonet; predict first):

- **E1.** *bus*, defaults: count boxes; preset *raw*: predict how many raw candidates per object;
  which stride finds the bus, which the people?
- **E2.** Select a box: read its grid cell and four distributions; find a box with a wide or
  two-peaked distribution and say which side is uncertain and why.
- **E3.** *nonms*, *strict*: duplicates vs lost neighbours (recall `detection/`); class-agnostic
  on and off on *people*.
- **E4.** Input size 256 (*small*): which objects disappear and why (stride vs object size)?
- **E5.** Switch strides off one at a time: predict which objects each stride carries.
- **E6.** *nuclei* defaults (nothing), then *nucleilow*: what classes does it assign, and what
  does the score map look like? Explain "no class exists". What would a fine-tuned model need
  (annotations, boxes)?
- **E7.** Your own image (a photo, then rotated 90°): what changes?
- **E8.** `nmsboxes` presets `anchor`, `heat`, `noisyheat` (moved from `detection/` E7, E8).

Quiz: what one output cell predicts; why a fixed output describes any number of objects; mean of
a distance distribution; why three strides; class-aware vs agnostic NMS; why the photo model
fails on nuclei; *(math)* GIoU of two disjoint boxes.

### 4. `sam/`: Promptable segmentation

Concept layer:

- ★ **Recall** (`attention/` miniature, fig-prompt.png): click → similarity → mask. Today: the
  real network, and why its design makes clicking fast.
- ★ **Three parts**: a heavy **image encoder** (a transformer on 16 × 16-pixel patches of the
  image resized to 1024², pretrained, then trained on millions of masks) run **once per image**
  → an embedding of 64 × 64 positions × 256 numbers; a light **prompt encoder** (points with
  label positive/negative, a box, or a previous mask → a few tokens); a light **mask decoder**
  (attention between prompt tokens and the embedding) run **per prompt**. Figure: block diagram
  with the costs measured in the browser: encoder 26.6 s once, decoder 0.42 s per click.
- **Why it pays**: 40 clicks on one image: 26.6 + 40 × 0.42 = 43 s; re-encoding for every click
  would take 40 × 27 = 1081 s. Same idea as caching the feature stack in the random-forest demo.
- ★ **Three masks per click**: one click is **ambiguous** (part of a nucleus? the nucleus? the
  cluster?). The decoder returns three candidate masks (small, medium, large) with a **predicted
  IoU** each (the network's own estimate of its quality). Figure: samnet "click in a cluster":
  the three candidates with their scores.
- ★ **More prompts remove ambiguity**: a box; a **negative** click on the neighbour; feeding back
  the chosen mask. Figure: positive + negative click.
- ★ **Embedding as a picture**: the 64 × 64 × 256 embedding's first three principal components as
  RGB (recall `pca/`): nuclei and background separate; one embedding cell covers 16 px of the
  resized image = 8 px of a 512² image: very small objects are below the embedding's resolution.
- ★ **Automatic masks ("segment everything")**: a grid of single-point prompts (8 × 8 = 64
  points → 192 candidates); filters: predicted IoU, **stability** (does the mask keep its shape
  when the threshold on its logits moves by ±1? area(logit > 1)/area(logit > −1): 380/430 = 0.88
  fails a 0.95 filter, 400/412 = 0.97 passes), box NMS; leftover pixels stay unlabelled. Cost:
  decoder 0.33 s per point → 21 s for 8 × 8, 84 s for 16 × 16. Figure: fate of every grid
  point (kept / low score / unstable / removed by NMS).
- **What it does not know**: "a nucleus" is not a concept it was trained on: automatic masks
  include background regions, clusters and parts; a click gives *a* mask, not a consistent
  definition over 1000 images (recall `attention/`). Microscopy-tuned variants and the
  Cellpose route of unit 2 (pointer: the newest Cellpose uses this encoder) exist for that
  reason.
- **Video and z-stacks** (one slide, concept only): the second version keeps a **memory** of
  earlier frames' embeddings and masks, so a click in one frame (or slice) follows the object
  through the stack; failures when the object leaves and returns or divides. Pointer to the
  tracking day.
- **Try it** (samnet `data-preset="cluster"`, then `"auto"`).
- Where this leads: prompts from another method (boxes from a detector, unit 3 → masks), and
  human-in-the-loop annotation. Classical counterpart: the `graphcut` seeds (day 27).

Math layer:

- *Cost*: encoder attention over $N = 64^2 = 4096$ tokens ($\propto N^2$ per layer, recall
  `attention/`), done once; decoder: a handful of prompt tokens attending to $N$ positions
  ($\propto kN$); total for $m$ prompts $T_\text{enc} + m\,T_\text{dec}$.
- *Mask logits*: 256 × 256 logits per mask, bilinear upsampling to the image, threshold 0;
  stability $= |\{\ell > \delta\}|/|\{\ell > -\delta\}| \le 1$ (nested sets; it is the IoU of the
  two masks).
- *Training with ambiguity*: three outputs, loss = the minimum over the three of the mask loss
  (only the best candidate is trained) → the outputs specialise to different granularities; the
  IoU head regresses the true IoU of each output.
- *Positional encoding* of prompt points (sinusoids of the coordinates) in one line.

**Demo `promptmask`** (new, analytic, TypeScript): a toy promptable segmenter on a synthetic
image (nuclei in clusters, a cell with a nucleus, debris; ground truth for each level). The
"encoder" is the feature stack of the random-forest day (`lib/featurestack.ts`) computed once
(timed: "encoding … ms, once"); the "decoder" is per click: similarity of every pixel's features
to the prompt's, three thresholds/smoothing levels giving three nested candidates (part /
object / cluster), connected piece containing the click, negative clicks subtract, box prompt
restricts; a **predicted score** per candidate (honestly labelled a proxy: the stability of
that mask) and its true IoU with the ground truth level, so students see when the score is
misleading. Automatic mode: point grid, filters (score, stability, box NMS) with the fate of
every point, as in `samnet`. A cost panel counts encoder vs decoder time across clicks.
Presets: "click on a nucleus edge", "click between two nuclei", "+ negative click", "box",
"automatic 8 × 8", "strict / loose filters". Differs from `embedprobe` (one mask, no scores, no
automatic mode); shares its feature code. Stability score and grid code reuse
`samnet/post.ts` (`stabilityScore`, `pointGrid`, `filterCandidates`), moved to a shared lib
file (e.g. `lib/prompt.ts`) so both demos import it.

Real model (Explore): **`samnet`** (built; images nuclei, u2os, cells, histology, brightfield;
presets click / cluster / box / negative / auto / strict / loose; "three candidates" and "feed
back the chosen mask" toggles; grid points per side, predicted-IoU, stability and box-NMS
sliders; embedding PCA view). Small extension wanted (owner of `samnet`): F1 against ground
truth for automatic masks on labelled images (`lib/eval.ts`), as `cellposenet` does.

By hand (concept):

- **B1.** A click on the edge of a nucleus returns three masks: A (area 120 px, predicted IoU
  0.71, a crescent at the edge), B (410 px, 0.93, the nucleus), C (860 px, 0.88, the nucleus
  and its neighbour). Which one does single-mask mode return? Which do you want for counting,
  and for measuring the cluster's area? Which second prompt makes B unambiguous (a negative
  click on the neighbour, or a box)?
- **B2.** Stability: a mask has 380 pixels with logit > 1 and 430 with logit > −1 (0.88), another
  400 and 412 (0.97). Which passes the 0.95 filter, and what does a low stability say about the
  mask's border?
- **B3.** Costs: encoder 26.6 s, decoder 0.42 s per click. 10 and 40 clicks with a cached
  embedding (30.8 s, 43.4 s) vs re-encoding (270 s, 1081 s). Automatic 8 × 8, 16 × 16, 32 × 32
  at 0.33 s per point: 64, 256, 1024 points, 192, 768, 3072 candidates, 21 s, 84 s, 338 s.
- **B4.** Resolution: the image is resized to 1024² and the embedding has 64 × 64 positions.
  How many pixels of a 512² image does one embedding position cover (8 × 8)? Of a 696-wide
  image (10.9)? Predict what happens to nuclei 6 px across.
- **B5.** Automatic masks on 30 nuclei with an 8 × 8 grid: how many grid points fall inside
  nuclei if nuclei cover 15 % of the image (about 10; 16 × 16: about 38)? Which nuclei are missed, and why does a
  finer grid help but cost more?

Math questions:

- **B6.** *(math)* Show the stability score is the IoU of the two thresholded masks and lies in
  [0, 1]; when is it exactly 1?
- **B7.** *(math)* Total time $T_\text{enc} + mT_\text{dec}$ vs $m(T_\text{enc} + T_\text{dec})$:
  the break-even and the speed-up for $m$ prompts.
- **B8.** *(math)* Minimum-over-outputs loss: with three outputs and two possible "true"
  masks for the same click, show that training with the minimum lets two outputs specialise,
  while an average loss pushes all outputs towards the mean mask.
- **B9.** *(math)* Self-attention cost of the encoder ($N = 4096$ tokens) vs the decoder
  ($k$ prompt tokens): ratio of attention-matrix sizes for $k = 6$.

Explore (samnet; predict first):

- **E1.** Load *nuclei*: note the encoder time. Look at the embedding view: what separates?
- **E2.** Preset *click*: three candidates and their scores; predict which is largest. *Cluster*:
  predict the three granularities, then check.
- **E3.** *Negative* and *box*: does the ambiguity disappear? Toggle "feed back the chosen mask".
- **E4.** Click 10 nuclei: compare the decoder time per click with the encoder time.
- **E5.** *Auto*, then *strict* and *loose*: count masks, find background masks and clusters,
  read the fate of the grid points.
- **E6.** Failure hunt: *histology*, *brightfield*, the smallest nuclei in *u2os*; your own image.
- **E7.** `promptmask`: reproduce B1 (click on an edge) and B2 (stability); when is the toy's
  score misleading?

Quiz: why the encoder runs once; why three masks; what a negative click does; what stability
measures; how automatic masks are made and filtered; why a nucleus is not a concept for it;
*(math)* cost for $m$ prompts.

### 5. `cellnets-compare/`: Networks and classical pipelines compared

Concept layer:

- ★ **The question**: on *my* images, is a network better than a classical pipeline, by how
  much, at what cost? Answer with the evaluation day's matching, not with pictures.
- ★ **The contestants** (one figure per contestant on one image, same colours): classical
  (smooth, threshold, distance transform, h-maxima, watershed: `ws-distance/`), StarDist
  (fluorescence nuclei model), Cellpose (nuclei model), SAM automatic masks; YOLO only as a
  reminder (no cell class: F1 0 out of the box). Optionally graph cuts with seeds (`graphcut`
  demo) as the interactive classical counterpart to SAM's clicks.
- ★ **The images**: public annotated sets fetched by `make models` (credits shown):
  `dsb-fluo-small`, `dsb-fluo-medium`, `dsb-fluo-dense`, `bbbc039-A02_s1`, `bbbc039-A06_s6`
  (fluorescence nuclei, sparse to dense), `dsb-histology` and `dsb-brightfield` (out of domain
  for the fluorescence models). Figure: the seven thumbnails with their object counts.
- **Recall: scoring** (one slide): IoU, one-to-one matching at τ, F1, F1(τ), splits and merges,
  AP's two meanings (object-eval). One rule: compare methods on the same images with the same
  τ, and report F1 at more than one τ (counting vs outlines).
- ★ **Reading F1(τ) curves**: a method that finds every nucleus with sloppy outlines is high at
  τ = 0.5 and falls fast; a careful one is flatter. Worked: 20 nuclei; method A 21 objects, 18
  matched at 0.5 (F1 0.88) but 12 at 0.7 (0.59); method B 17 objects, 16 at 0.5 (0.86), 15 at 0.7
  (0.81). Figure: two F1(τ) curves (synthetic, from the worked numbers + `objmatch`-style).
- ★ **Results table** (measured by the builder, not guessed): F1 at τ = 0.5 and 0.7, splits,
  merges, per method per image; a picture of the errors (merge/split/miss marks) for the most
  instructive image. Expected pattern to check, not to assert: networks win on dense and dim
  nuclei; the classical pipeline is competitive on sparse, bright, round nuclei; everything
  trained on fluorescence drops on histology/bright field unless inverted or retrained.
- ★ **Cost**: parameters (StarDist about 1.4 M — 5.5 MB of f32 weights; Cellpose 6.6 M; YOLO
  nano 2.6 M; SAM 2.1 tiny about 39 M — 78 MB of f16 weights; classical: a handful of knobs),
  download size, time per 512² image in the browser on one core (StarDist 2.1 s; Cellpose 7.7 s
  with 4 workers; SAM automatic 8 × 8 48 s; classical: milliseconds, measured), i.e. about 1690,
  470 and 75 images per hour. Annotation cost: classical needs none (but tuning), pretrained
  networks none if the domain fits, fine-tuning hundreds of annotated objects.
- **When to choose what** (table): sparse bright round nuclei → classical is enough and
  explainable; dense/dim nuclei → StarDist or Cellpose; irregular cytoplasm → Cellpose;
  interactive annotation of a few odd objects → SAM clicks (or graph-cut seeds); counting
  objects of a known class in video → a fine-tuned detector. Always: score on a held-out
  annotated subset of *your* images, compare to a second annotator's agreement (object-eval E7).
- **Honest caveats**: one small benchmark; the networks were trained on some of these public
  sets or their relatives (**leakage**, the leakage unit of day 37): their scores here can be
  optimistic; classical knobs tuned on the same images are optimistic too.
- Summary + a checklist "before you trust a segmentation network on your data".

Math layer:

- *F1(τ) and its non-increase* (recall `object-eval/` B8), AP over τ (the averaged one);
  bootstrap confidence interval of a difference in F1 over images (recall `replicates/` or
  `leakage/`, day 37): why two methods 0.02 apart on 7 images are not distinguishable.
- *Throughput*: images per hour $= 3600/(t_\text{net} + t_\text{decode})$; parallel workers and
  Amdahl's law in one line (pointer to the GPU day).

**Demo `cellcompare`** (new): choose one of the seven labelled images; the classical pipeline runs
live in TypeScript (σ, threshold method Otsu/manual, h, minimum size; `lib/imageops.ts`,
`lib/threshold.ts`, `lib/distance.ts`, `lib/morph.ts`, `lib/watershed.ts`); buttons "run
StarDist", "run Cellpose (nuclei)", "run SAM automatic" load the same Wasm modules and weights as
the four real-model demos (same-origin, cached, progress bars, the `file://` message) and cache
each result per image; panels: the image with outlines per method (side by side or toggled),
error marks (merge/split/miss from `splitsMerges`), the F1(τ) curves of all methods on one plot
(`lib/eval.ts` `f1Curve`), a table (F1 at 0.5 and 0.7, TP/FP/FN, splits, merges, time, model
size). Presets: "sparse: classical holds up", "dense: networks win", "histology: everyone
struggles", "dim nuclei". Engineering: the networks' worker code is currently private to each
demo (`stardistnet/worker.ts`, `cellposenet/pipeline.ts` + its own-bundle workers,
`samnet/worker.ts`); move the model-running parts to shared modules (e.g. `lib/nets/stardist.ts`,
`lib/nets/cellpose.ts`, `lib/nets/sam.ts`) that any bundle can start as a worker, without
changing the four demos' behaviour (their tests keep passing). SAM's automatic run (~50 s) is
offered last and optional. Fallback: if the first build finds running three networks in one page
too heavy, `make models` writes each network's label PNGs for the seven images (computed natively
by the same crates) and the demo scores those, with a note "precomputed by the same network"
(open question 1).

By hand (concept):

- **B1.** The worked F1 numbers above: compute F1 for A and B at τ = 0.5 and 0.7 (0.88, 0.59;
  0.86, 0.81). Which method for counting, which for measuring areas?
- **B2.** An IoU table of 4 true × 5 predicted objects from a classical result (builder makes it
  from a real crop with a script): matching at 0.5, TP/FP/FN, one merge to identify.
- **B3.** Cost: images per hour from the times above; a screen of 10 000 images on one laptop
  core: StarDist about 5.9 h, Cellpose about 21 h, SAM automatic about 134 h.
- **B4.** Choose a method (and say how you would check it) for: (a) 50 000 sparse Hoechst nuclei
  images; (b) 30 phase-contrast images of irregular cells; (c) 5 tissue sections where a
  pathologist wants 200 specific cells outlined; (d) a time-lapse of dividing yeast.
- **B5.** Leakage: a network scores F1 0.95 on a public image it was trained on. What does that
  number tell you about your images?

Math questions:

- **B6.** *(math)* Bootstrap over images: given per-image F1 for two methods on 7 images
  (builder supplies, computed), a 95 % interval of the mean difference; is it above 0?
- **B7.** *(math)* Show that F1 at τ = 0.5 is at least F1 at τ = 0.7 for the same matching
  method above 0.5 (recall object-eval B8b).

Explore (`cellcompare`; predict first; numbers measured by the builder):

- **E1.** *sparse*: tune the classical pipeline; predict whether StarDist beats it, run both.
- **E2.** *dense*: same; where are the classical merges?
- **E3.** F1(τ) curves: which method has the best outlines (flattest curve)?
- **E4.** *histology* and *brightfield*: which method is least bad, and why (training domain)?
- **E5.** Fill in the cost table from the readouts; images per hour for your laptop.
- **E6.** Pick the method you would use for one image type and write the two-sentence argument
  with numbers.

Quiz: why compare at more than one τ; what leakage does to a benchmark; when classical is enough;
what cost includes besides time; why a detector without a cell class scores 0.

## Slimming of day 39 (edits to built units; keep their demos working)

To be done in the same build as this day (by whoever builds `stardist/`/`cellpose/`/`yolo/`, or a
separate integration step), so nothing is taught twice. **No change to the `instancemaps` or
`nmsboxes` code is required**: every route, mode and preset stays; only where they are embedded
changes. The demo gallery (`demos.qmd`) entries stay as they are.

`instances/lec.qmd`:

- "About this lecture": "Four such maps …" becomes "Four such maps; the first two in detail
  here, star-convex polygons and flows on the next day".
- Keep: recall, why post-processing (fig-maps.png, all four), Route 1, Route 2, "Try it: four
  maps on one image" (`data-preset="dist"`; the route selector still offers all five routes).
- **Route 3** → one pointer slide "Star-convex polygons" (fig-star.png beside three bullets: each
  pixel proposes a polygon from ray lengths; suppression keeps one per object; fails on
  non-star-convex shapes; link "Networks for cells and objects: `../stardist/lec.qmd`").
- "The math: the four targets" keeps the boundary and distance targets; the star-convex and
  flow lines become one line each with a pointer (definitions now in `stardist/`/`cellpose/`).
- **Route 4** → one pointer slide "Flow fields" (fig-cflow.svg beside three bullets; link to
  `../cellpose/lec.qmd`). Remove "The math: following the flow" and "Try it: flows follow the
  bend" (both move to `cellpose/`).
- Keep "Choosing a route" (it is the overview; add the two links), "Detection: objects as a
  list", panoptic + its math, recall of counting errors; summary bullets 4–5 shortened with
  pointers.
- Result: 17 → 15 slides (counting "About this lecture").

`instances/exercise.qmd` and `quiz.qmd`: move B3 → `stardist/` B3, B4 → `cellpose/` B5,
B7/B9 → `stardist/` math (B7 merged into B6/B7 there), B8 → `cellpose/` B7; E5 → `stardist/` E8,
E6, E7 → `cellpose/` E1; keep B1, B2, B5 (choose a route, now with links), B6, B10, E1–E4, E8,
E9; renumber and fix the Solutions callouts. Quiz: Q6 moves to `cellpose/`; Q5 stays (route
choice). Learning goals: drop the two that name polygons/flows in detail, add one "know that
polygons and flows exist and where they are taught".

`detection/lec.qmd`:

- "About this lecture": second bullet becomes "Where candidate boxes come from: a grid of guesses
  (in detail on the next day)".
- Replace "Where candidates come from: anchors", its math, "centre heat maps" and its math (4
  slides) by one pointer slide "Where do the candidates come from?" (fig-anchors.png beside: a
  coarse grid, every cell guesses a box and a score; older heads used template boxes (anchors),
  others centre peaks; link `../yolo/lec.qmd`). fig-anchors.png and fig-heat.png are reused in
  `yolo/` (copied by `tools/figs/yolo/lec_figs.py`, which imports the drawing functions from
  `tools/figs/detection/lec_figs.py`; the detection figures stay where they are).
- Keep boxes, IoU (+ math), recall scoring, many candidates per object, greedy NMS (+ math), try
  it (`nmsboxes data-preset="crowd"`), what NMS gets wrong, three meanings of NMS (add a fourth
  row "polygons (next day, StarDist)"), summary (bullet 2 shortened).
- Result: 15 → 12 slides (counting "About this lecture").

`detection/exercise.qmd`/`quiz.qmd`: move B3 → `yolo/` B4, B4 → `yolo/` B5, B8 → `yolo/` B9,
E7, E8 → `yolo/` E8; keep B1, B2, B5–B7, B9, E1–E6; renumber. Quiz Q5, Q6 move to `yolo/`. The
Explore embed (`nmsboxes data-preset="iou"`) is unchanged; the mode selector still offers
anchors and heat map (harmless; one sentence "anchors and heat maps: see the next day").

`attention/lec.qmd`: "Promptable segmentation" keeps its miniature and gains one line pointing
to `../sam/lec.qmd`; summary unchanged.

`days.yml` (integration step, not this plan's scope): `cellnets.plan` → `plan/cellnets-day.md`,
`status` removed when built, `demos:` entries per unit (explore anchors for each real-model and
analytic demo), and `instances`/`detection` demo labels unchanged. `plan/segmodels-plan.md` gets a
line "day plan: `plan/cellnets-day.md`".

## Shared engineering

- **New demos** (names checked against `demos/src/`: no collisions): `starrays`, `flowtrack`,
  `yolodecode`, `promptmask`, `cellcompare`. TypeScript, `make typecheck`, fixed seeds, presets +
  reset, `slider`/`button`/`checkbox` helpers, labels ≥ 18 px, 1150 × 560 on slides, per-demo CSS
  prefixed `.qb-demo[data-demo="<name>"]` (build-issues lesson), static figure fallbacks.
  Build only your own (`demos/build.sh <name>`).
- **Real-model demos** are reused as built; Explore questions use their `data-image`,
  `data-view`, `data-preset` options. Two small extensions requested from their owners: F1 vs
  ground truth in `stardistnet` and in `samnet` automatic mode.
- **`lib/instances.ts`** (new): ray targets, polygon from rays, polygon rasterisation and IoU,
  star-convexity check, heat target inside a mask, flow from heat, flow following and sink
  grouping, flow-error check. Moved from `instancemaps/model.ts` (which imports them; outputs
  identical, checked by a test that renders all `instancemaps` presets before/after) and used
  by `starrays`, `flowtrack`, `cellcompare`. Tests `demos/test/instances-test.ts`: disc rays =
  radius, 8-ray rectangle area 24.0, polygon IoU symmetric and 1 for identical polygons, heat
  target maximal at the source, every path in an exact field ends at the source, flow error 0 for
  the exact field.
- **Detection decoding**: `softmax`/distribution decode/class-aware and agnostic NMS from
  `yolonet/post.ts` move to `lib/boxes.ts` (or `lib/detect.ts`); `yolonet` imports them;
  `yolo-wasm-test.ts` unchanged and passing.
- **Prompt post-processing**: `stabilityScore`, `pointGrid`, `filterCandidates`, `maskBox` from
  `samnet/post.ts` move to `lib/prompt.ts`; `samnet` imports them; `sam-wasm-test.ts` passing.
- **Network runners** for `cellcompare` (`lib/nets/*.ts`, see unit 5), only if the live route is
  chosen.
- **Figures**: `tools/figs/<unit>/lec_figs.py` for synthetic figures (rays/polygons, heat steps,
  grids and distributions, stability, F1 curves); `tools/figs/<unit>/demo_shot.sh` for real-model
  screenshots (headless Chrome against a local server after `make wasm` + `make models`, as in
  `tools/figs/instances/`), committed as PNGs so the decks render without the models. Image
  credits from `demos/dist/models/index.json` in captions.
- **Numbers**: by-hand numbers from `numbers.py` (above); Explore solutions measured with a node
  script per unit (load the Wasm module and weights as the `*-wasm-test.ts` do, apply the preset,
  print the readouts), stored next to the figure scripts.
- **Renders**: single pages only (`quarto render <unit>/lec.qmd` etc.); no full-site render.

## Suggested order of self-study (no clock: self-study course)

| Unit | Lecture | Exercise |
|---|---|---|
| 1 `stardist/` | 18 slides, `starrays`, `stardistnet` | B1–B5, math B6–B9, E1–E8 |
| 2 `cellpose/` | 20 slides, `instancemaps` warm-up, `flowtrack`, `cellposenet` | B1–B6, math B7–B10, E1–E7 |
| 3 `yolo/` | 21 slides, `yolodecode`, `yolonet`, `nmsboxes` | B1–B6, math B7–B10, E1–E8 |
| 4 `sam/` | 19 slides, `promptmask`, `samnet` | B1–B5, math B6–B9, E1–E7 |
| 5 `cellnets-compare/` | 14 slides, `cellcompare` | B1–B5, math B6–B7, E1–E6 |

The real-model demos take seconds (StarDist, YOLO) to a minute (SAM automatic, Cellpose on large
images): the exercises say so and ask for the prediction before the run.

## Open questions

1. **`cellcompare`: live networks or precomputed labels?** Live is honest and consistent with
   the other demos but heavy (three networks, ~100 MB of weights, ~1 min). Proposal: live, with
   precomputed label PNGs from `make models` as the fallback if the page is too slow; decide after
   measuring.
2. **A cell detector for `yolo/`**: the only published weights are the COCO model (no cell
   class). Fine-tuning YOLOv11-rs on a public nuclei set (boxes from the DSB/BBBC039 masks) in
   CI or once offline, publishing the checkpoint somewhere CI can fetch it, would turn E6 from a
   pure failure into a comparison. Proposal: keep the failure as the lesson now; add a nuclei
   checkpoint later and a row in unit 5.
3. **Cellpose-SAM**: too large for the site (manifest: disabled). Keep as a pointer only.
4. **SAM 2 memory on z-stacks/video**: concept slide only, or a later demo with the sam2-rs video
   predictor on a short public time-lapse? Proposal: concept only now.
5. **`stardist/` 3D**: a pointer slide, or wait for a 3D model in the browser? Proposal: pointer.
6. **YOLO training assignment details** (task-aligned assignment): one concept bullet + nothing
   in the math layer, or a math slide? Proposal: one bullet; the decoding is the content.
7. **Name of unit 3 and the model family**: deck title "Detecting objects in one pass: YOLO" with
   the folder `yolo/` (as in `days.yml`); the family name changes per version — use the family
   name, never a version number, in student material (version only in demo credits).
8. **Order with `diffusion/` (day 41)**: Cellpose's "diffusion" is the heat equation; the one
   disambiguation line in unit 2 is enough?

## References

- Schmidt, Weigert, Broaddus & Myers (2018), *Cell detection with star-convex polygons*, MICCAI.
- Weigert, Schmidt, Haase, Sugawara & Myers (2020), *Star-convex polyhedra for 3D object
  detection and segmentation in microscopy*, WACV.
- Stringer, Wang, Michaelos & Pachitariu (2021), *Cellpose: a generalist algorithm for cellular
  segmentation*, Nature Methods. Pachitariu & Stringer (2022), *Cellpose 2.0*; Stringer &
  Pachitariu (2025), *Cellpose3: one-click image restoration*, Nature Methods; Pachitariu,
  Rariden & Stringer (2025), *Cellpose-SAM* (preprint) — pointer only.
- Redmon, Divvala, Girshick & Farhadi (2016), *You only look once: unified, real-time object
  detection*, CVPR. Lin, Goyal, Girshick, He & Dollár (2017), *Focal loss for dense object
  detection*, ICCV. Tian, Shen, Chen & He (2019), *FCOS: fully convolutional one-stage object
  detection*, ICCV. Li et al. (2020), *Generalized focal loss* (distribution focal loss),
  NeurIPS. Rezatofighi et al. (2019), *Generalized intersection over union*, CVPR. Lin et al.
  (2017), *Feature pyramid networks for object detection*, CVPR.
- Kirillov et al. (2023), *Segment anything*, ICCV. Ravi et al. (2024), *SAM 2: segment
  anything in images and videos* (preprint/ICLR 2025). Ma et al. (2024), *Segment anything in
  medical images*, Nature Communications (fine-tuned variants, pointer).
- Caicedo et al. (2019), *Nucleus segmentation across imaging experiments: the 2018 Data Science
  Bowl*, Nature Methods (images); Ljosa, Sokolnicki & Carpenter (2012), *Annotated
  high-throughput microscopy image sets for validation*, Nature Methods (BBBC039).
