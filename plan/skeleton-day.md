# Plan: a day on skeletons, with applications to neurons and blood vessels

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.
Decided 2026-11-12: skeletons are their own day (not a unit of the morphology day), and
the hit-or-miss transform moves here with them.

**Course map (2026-11-12):** elective, group "Objects" (skeletons, moments); entry after core
day 8 (labelling). Morphology (day 7, owner of the distance transform), labelling (day 8) and
the edges day (day 2, owner of the Hessian and vesselness basics) are always taught first.
Edits applied 2026-11-12: unit 4's Hessian/vesselness basics became a one-slide recall of
day 2 `hessian/`; the freed time goes to the neuron and vessel applications.

## Goal of the day

After the day a student can:

1. compute a **hit-or-miss transform** by hand with a three-state template (1, 0, don't care) and use it to find end points, isolated pixels and branch points;
2. **thin** a small shape to a one-pixel skeleton by hand, decide from a 3×3 neighbourhood whether a pixel is **simple** (removable without changing the topology), and explain why end points must be kept;
3. define the **medial axis** as the centres of maximal inscribed discs, find it as the **ridges of the distance transform**, state **Lantuéjoul's formula**, and **reconstruct** a shape from its skeleton and radii;
4. explain why skeletons are **sensitive to boundary noise** and remove spurs by **pruning**;
5. turn a skeleton into a **graph** (end points, branch points, edges with lengths), estimate lengths without the pixel-counting bias, and compute the standard measurements for **neurons** (total length, branch counts, **Sholl analysis**) and **vessel networks** (total length, branching density, radius from the distance transform, **tortuosity**);
6. use the **Hessian-based vesselness filter** from day 2 as the first step of a neuron or vessel pipeline, choose its scales from the expected radii, and recognise its failure modes (junctions, touching vessels, blobs) in the resulting skeleton graph.

Why it fits this audience: thinning and hit-or-miss are set operations on a 3×3 grid,
fully traceable by hand. The medial axis is a geometric optimisation (largest disc that
fits). The vesselness filter (taught on day 2) returns here as a tool whose output must be
skeletonised and measured. The applications (neuron morphology, vascular networks) are measurements
biologists make every day, with answers that are numbers students can check.

## Prerequisite gap

- **Erosion, opening, structuring elements** (day 7, morphology): used directly; the
  hit-or-miss transform is two erosions. One-slide recall.
- **Distance transform** $\mathrm{DT}$ (day 7, morphology owns it: definition, by hand,
  two-pass chamfer): one-line recall; `lib/distance.ts` computes it.
- **Connectivity and labelling** (day 8): needed for "simple point" (counting components in
  a 3×3 window), for the Euler number and for splitting a skeleton into edges. One-line
  recall (decided: labelling always comes first, so no recap).
- **Hysteresis thresholding**: the idea is day 5's (thresholding); the implementation as
  reconstruction + labelling is day 8's. One-line recall.
- **Hessian, its eigenvalues, Frangi vesselness, scale-normalised derivatives** (day 2,
  edges day, `hessian/`, the owner): recalled in one slide in unit 4.

## Links to other days

- **Morphology** (day 7): hit-or-miss = $(A \ominus B_1) \cap (A^c \ominus B_2)$; Lantuéjoul's skeleton is
  built only from erosions and openings; pruning uses reconstruction. The morphology day
  has a one-slide pointer here from its unit 2.
- **Labelling** (day 8): thinning preserves the Euler number (components − holes); skeleton edges
  are the connected components of the skeleton with the junctions removed; length,
  branch count and tortuosity become per-object properties in the labelling day's table.
- **Edges, blobs and ridges** (day 2): owns the Hessian and vesselness; this day uses them.
- **Watershed** (day 9): the medial axis is the ridge set of $\mathrm{DT}$, the same landscape whose peaks
  split touching nuclei there; vessel segmentation by watershed of the vesselness is a
  pointer.
- **Level sets** (elective): Blum's grassfire (a front moving inwards at unit speed; the skeleton is
  where fronts collide) is a level-set picture; arrival time = distance transform.
- **Graph cuts** (day 12): thin structures are exactly what the smoothness term removes
  (shrinking bias); vesselness as a data term is a pointer.
- **Marching cubes** (3D elective; `volumes3d/` owns stacks and voxel spacing): 3D vessel trees and neurons; curve skeletons in 3D, and centreline
  + radius as a compact alternative to a surface mesh (tube rendering).

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise. Unit 4 no longer teaches the
Hessian (day 2 does), which was what made it the densest.

### 1. `skel-thinning/`: Hit-or-miss and topology-preserving thinning

- Motivation: a thresholded neuron or vessel image. The question is "how long, how
  branched", not "how much area": we want a one-pixel-wide centre line with the same
  topology as the shape.
- **Hit-or-miss** $A \circledast (B_1, B_2) = (A \ominus B_1) \cap (A^c \ominus B_2)$ with
  $B_1 \cap B_2 = \emptyset$: a template that must fit the foreground and the background at
  once. Drawn as a 3×3 grid with three states (1, 0, don't care). Templates for isolated
  pixels, end points (8 rotations), corners, T- and Y-junctions.
- **Thinning** $A \setminus (A \circledast T)$ with a family of 8 rotated templates applied in
  turn, repeated until nothing changes. Why one template at a time (sequential): removing
  pixels from both sides of a 2-pixel-thick line in parallel would cut it.
- **Simple points**: a pixel is removable iff removing it changes neither the number of
  foreground 8-components nor background 4-components in its neighbourhood: checked
  locally by counting components among the 8 neighbours (crossing number / Yokoi
  connectivity number). This is what makes thinning **topology-preserving**: the
  Euler number is unchanged. End points (one neighbour) are simple but kept, otherwise
  every line shrinks to a point.
- **Zhang–Suen** as the widely used two-subiteration parallel variant (one slide);
  result: an 8-connected, (almost) one-pixel-wide skeleton. Known artefacts: two-pixel
  thick diagonals, small loops at junctions, corners shortened.
- **Demo `morphgrid`** with `data-mode="hitmiss"` (moved here from the morphology plan;
  reuses that demo's grid painting and step controls): the SE editor gets three states
  (click cycles 1 / 0 / don't care) and shows the template over the image as it moves,
  matched cells green, failed cells red. Sub-mode *thin*: applies the 8 rotated
  templates one at a time, colouring the pixels removed per template and per pass;
  toggle *keep end points*; toggle *parallel (no subiterations)* to show the broken
  line. A *simple?* tool: click a pixel to see its 3×3 neighbourhood, the components
  counted and the verdict. Presets: "find the end points", "isolated pixels", "thin a
  rectangle", "thin an L", "a ring keeps its hole", "parallel removal cuts the line".
- By hand: (a) apply an end-point template (given) to a 7×7 image with a short line and
  an isolated pixel. (b) Five 3×3 neighbourhoods: simple or not, and why. (c) One full
  thinning pass (templates given) on a 3×7 rectangle, then iterate to the skeleton.

### 2. `skel-medialaxis/`: Medial axis, distance transform and reconstruction

- **Grassfire** (Blum 1967): set fire to the boundary, let it burn inwards at unit speed;
  the skeleton is where fire fronts meet. Equivalent: the **medial axis** = centres of
  **maximal inscribed discs** (discs inside $A$ not contained in any other such disc).
- The arrival time of the fire is the distance transform $\mathrm{DT}$ (day 7). The medial axis is the
  **ridge** set of $\mathrm{DT}$: points where $\mathrm{DT}$ has a local maximum across the shape (in 1D
  across a band of width $w$: a tent with its top at $w/2$). Each axis point carries a
  radius $\mathrm{DT}(x)$; **reconstruction** $A = \bigcup_{x \in S} \text{disc}(x, \mathrm{DT}(x))$: the
  skeleton plus radii is a lossless shape code.
- **Lantuéjoul's formula** $S(A) = \bigcup_r S_r$, $S_r = (A \ominus rB) \setminus \big((A \ominus rB) \circ B\big)$:
  the points that the erosion by $rB$ keeps but the following small opening removes are
  centres of maximal discs of radius $r$. Reconstruction $A = \bigcup_r S_r \oplus rB$.
  Only erosions and openings: everything from day 7.
- **Thinning vs medial axis**: thinning is connected but only approximately centred; the
  discrete medial axis is centred but can be disconnected and is not one pixel wide.
  Practical tools combine both (thin, keep $\mathrm{DT}$ on the result for radii).
- **Sensitivity to boundary noise**: the medial axis is not continuous in the shape. A
  one-pixel bump on the boundary creates a whole new branch reaching the bump; a
  rectangle's axis forks at each end. Remedies: smooth or open the mask first, prune
  spurs (unit 3), or keep only axis points with a large enough angle between their two
  nearest boundary points / with $\mathrm{DT}$ above a fraction of the local maximum (named:
  scale axis, "λ-medial axis").
- **Demo `medialaxis`** (new): a polygon or smooth closed shape with draggable vertex
  handles (presets: rectangle, ellipse, L, star, a dumbbell, a ring), rasterised at
  256×256. Panels: $\mathrm{DT}$ as colour with iso-lines; axis overlay with selectable method
  (ridges of $\mathrm{DT}$ / Lantuéjoul / thinning); a *grassfire* slider shows the front at time
  $t$ (the iso-line $\mathrm{DT} = t$) burning in and the axis appearing where fronts meet. Click an
  axis point to draw its maximal disc; *reconstruct* draws the union of all discs over the
  shape with the difference coloured. *Bump* slider adds a small boundary bump that can
  be dragged along the boundary; readout: number of branches. Metric of $\mathrm{DT}$ (city-block,
  chamfer, Euclidean). Presets: "a rectangle forks at the ends", "one bump, one branch",
  "the ring keeps its loop", "reconstruct the star from its axis", "city-block vs
  Euclidean axis".
- By hand: (a) the city-block distance transform of a 5×9 rectangle (given): mark the
  ridge pixels and compare with the thinning result of unit 1. (b) Lantuéjoul's $S_0$,
  $S_1$ for a 5×5 square with the 3×3 SE. (c) Reconstruct a small shape from a given
  skeleton with radii.

### 3. `skel-graph/`: From skeleton to graph: pruning and length

- **Classify skeleton pixels** by the number of skeleton neighbours: 1 = end point,
  2 = regular, ≥ 3 = junction (with the hit-or-miss templates of unit 1 as the
  equivalent view). Adjacent junction pixels form a junction **cluster** (labelling day)
  = one graph node.
- **Skeleton graph**: nodes = end points and junction clusters; edges = connected
  components of the skeleton after removing the junction pixels, each linking two
  nodes; attributes: length, the pixel path, mean radius. Loops (an edge from a node to
  itself) and cycles are allowed. Number of independent cycles $= E - V + C$ (the
  cyclomatic number; equals the number of holes, linking back to the Euler number).
  The adjacency matrix and node degrees as row sums: a direct link to linear algebra.
- **Length bias**: counting pixels underestimates diagonal lines by a factor $\sqrt 2$.
  Weighted steps (1 for axial, $\sqrt 2$ for diagonal neighbours) fix the worst case but
  overestimate lines at intermediate angles by a few per cent; better weights
  (Vossepoel & Smeulders: 0.980 / 1.406 plus a corner correction) or fitting a smooth
  curve to the path; the true length of a straight line at angle $\theta$ shown against
  each estimator. Same story as the perimeter bias on the labelling day.
- **Pruning**: remove terminal edges (end point to junction) shorter than $k$, then
  re-merge nodes of degree 2; repeat once, since pruning can create new short terminal
  edges. Morphological version: $k$ end-point removals followed by reconstruction of the
  main branches. Choosing $k$: relative to the local radius $\mathrm{DT}$ at the junction (a spur
  shorter than the vessel radius is noise).
- **Demo `skelgraph`** (new): a 256×256 mask (presets: a "Y", a letter shape, a noisy
  thresholded vessel crop from `lib/trees.ts`; or draw with a brush), its skeleton, and
  the graph overlaid: nodes coloured by degree (end / junction), edges coloured by
  length, with lengths written on the edges. Pruning slider $k$ (absolute or relative to
  $\mathrm{DT}$); length estimator selector (pixel count / 1–$\sqrt2$ / corrected); an edge table
  (id, nodes, length by each estimator, mean radius). A *straight line at angle θ*
  preset with a θ slider and a plot of each estimator / true length versus θ. Presets:
  "a Y: 3 ends, 1 junction", "noise makes spurs", "prune them", "a loop: cycles =
  holes", "diagonal lines and the √2".
- By hand: (a) a given 9×9 skeleton: classify every pixel, draw the graph, give each
  edge's length by pixel count and with weights 1 / $\sqrt 2$. (b) Prune with $k = 2$
  and redraw the graph. (c) Check $E - V + C$ against the number of holes.

### 4. `skel-neurovessel/`: Measuring neurons and vessel networks

- **Getting a segmentation to skeletonize**: thin, faint, branching structures on an
  uneven background break a global threshold (gaps cut branches; the skeleton then
  undercounts length). A filter that responds to **tubes** helps.
- **Recall slide (day 2, `hessian/`)**: the Hessian of the Gaussian-smoothed image at scale
  $\sigma$; eigenvalues (order by magnitude, $|\lambda_1| \le |\lambda_2|$, stated on the slide):
  tube = one large, one near zero; blob = both large; background = both small; the
  multi-scale Frangi vesselness with its two shape/contrast parameters and $\sigma^2$
  normalisation. No re-derivation here.
- **Vesselness as the first step of a measurement** (new here): choose the range of
  $\sigma$ from the expected radii (the best scale is a first radius estimate, to compare
  with $\mathrm{DT}$ on the skeleton); what its failure modes do *downstream*: junctions
  fade (the skeleton graph loses branch points or gets gaps), touching parallel vessels
  merge (one edge instead of two), bright blobs on a vessel (spurious short branches).
  Threshold the vesselness, not the input.
- **The pipeline**: vesselness → threshold (hysteresis: day 5 idea, day 8 implementation)
  → clean (area filter, small closing) → skeletonize → prune → graph → measurements.
- **Neurons**: total dendritic length, number of primary branches (edges leaving the
  soma), branch points, terminal points, branch order (centrifugal order; Strahler order
  with a worked example). **Sholl analysis**: count intersections of the arbor with
  circles of radius $r$ around the soma; the Sholl profile $N(r)$, its maximum and
  critical radius, as a one-curve summary of arbor complexity; the link to the skeleton
  graph (an intersection = an edge crossing the circle). Manual and semi-automatic
  tracing and the standard tree file format of tracing tools (a list of nodes with
  parent and radius) as what neuroscientists actually use (one slide, no tool names in
  student material).
- **Vessel networks**: total length per area (length density), branch points per area
  (branching density), **radius** along the centreline = $\mathrm{DT}$ on the skeleton
  (diameter $2\,\mathrm{DT}$; compare with the vesselness scale), **tortuosity** per edge =
  path length / chord length (1 = straight; raised in tumour and retinal vessels), number
  of loops (anastomoses, = independent cycles of unit 3), vessel area fraction. Pointer:
  retinal fundus images as the standard 2D vessel benchmark.
- **How good is the graph?** (new, uses the freed time): compare a measured skeleton graph
  with the ground-truth graph of the synthetic tree: length error, missed and spurious
  branch points, and how they move with noise, blur and the pruning length $k$. One line
  forward to day 15 (evaluation) for matching objects in general.
- 3D in one slide: 3D vessel trees (micro-CT, light-sheet) with the 3×3 Hessian (day 2) and
  3D thinning; stacks and anisotropic spacing $(\Delta x, \Delta y, \Delta z)$ recalled from
  `volumes3d/` (marching-cubes elective) as a pointer.
- **Demo `arbor`** (new, main demo): a seeded synthetic neuron (soma + branching
  dendrites) or vessel network (tree with varying radius, tortuosity and a few loops)
  from `lib/trees.ts`, with the known ground-truth graph. Steps as tabs: vesselness (the
  day-2 filter from `lib/edges.ts`, controls $\sigma$ range and bright/dark only) →
  threshold → skeleton → pruned graph → measurements. Overlays: radius along the
  centreline as colour; edges coloured by tortuosity; Sholl circles with a radius slider
  and the Sholl profile plot beside the image; a *ground truth* overlay with missed and
  spurious branch points marked. Readouts next to ground truth: total length (by each
  length estimator), branch and end points, branching density, mean radius, mean
  tortuosity. Generator controls: seed, branching probability, tortuosity, noise, blur.
  Presets: "neuron: Sholl profile", "vessels: length and branching density", "noise:
  spurs inflate the branch count", "one scale misses the thick vessels", "junctions fade:
  branch points lost", "pixel counting underestimates length", "tortuous vessels".
  (The separate `vesselness` demo of the earlier draft is dropped: the Hessian glyph and
  eigenvalue scatter are the edges day's demo.)
- By hand: (a) a small skeleton graph with edge lengths and chords: total length,
  tortuosity per edge, Strahler order of each edge, Sholl counts for three radii. (b)
  Radius along a centreline from a given distance transform, compared with a given best
  vesselness scale. (c) A measured and a true graph of a small tree: count missed and
  spurious branch points.

## Shared engineering

- `demos/src/lib/skeleton.ts` (new): hit-or-miss with three-state templates; thinning
  (rotating templates, sequential, with an instrumented version recording each template
  application for `morphgrid`); simple-point test; Zhang–Suen; medial axis from ridges
  of $\mathrm{DT}$; Lantuéjoul skeleton and reconstruction; pixel classification, junction
  clusters, skeleton graph (nodes, edges with pixel paths); length estimators; pruning;
  Sholl intersections; tortuosity; radius along edges.
- `demos/src/lib/trees.ts` (new, synthetic data): a seeded branching-tree generator
  (random branching with angle and length distributions, radius decreasing with branch
  order (Murray's law as an option), tortuosity by low-frequency perturbation, optional
  loops), producing both the **ground-truth graph** (for checking measurements) and a
  rendered image (radius-dependent intensity, blur, noise, vignetting, background
  blobs). Neuron and vessel presets. Generates 3D graphs too, so a 3D vessel demo can
  use it later. Seeded with `rng(seed)` from `lib/random.ts`.
- Hessian and vesselness come from `lib/edges.ts` (owner: edges day, day 2); the closed-form
  eigenvalues/eigenvectors of a symmetric 2×2 matrix from `lib/linalg.ts` (owner: graph-cut
  day). Nothing of that is written here.
- Reused, not duplicated: `lib/morph.ts` (erosion, opening, reconstruction for pruning),
  `lib/label.ts` (components for simple points, junction clusters and edges; Euler number
  for tests), `lib/distance.ts` (chamfer and exact Euclidean for medial axis and radii),
  `lib/convexhull.ts` (labelling day; chords and extents), `lib/plot.ts` for the Sholl
  profile and estimator-vs-angle plot, `lib/gridui.ts` (morphology day). The `morphgrid` `hitmiss` mode lives in the morphology day's demo
  folder but is specified here.
- **Language / speed** (`demos/README.md`): **TypeScript** throughout. The multi-scale
  vesselness tab in `arbor` reuses `lib/edges.ts` (a few ms per scale); thinning of 256×256 converges in tens of passes; both
  should be live. Measure first; Rust/Wasm only if a recomputation exceeds ~30 ms
  (candidate: a 3D Frangi filter on a 128³ volume, only if a 3D demo is added). 3D
  views, if any, through `lib/gl3d.ts` (WebGL; owner 3D graphics elective).
- Tests (`make test` over `demos/test/*-test.ts`): thinning preserves the number of components and holes (via
  `label.ts`) and yields no removable simple non-end pixels; Zhang–Suen and template
  thinning agree on simple shapes up to known artefacts; Lantuéjoul skeleton with radii
  reconstructs the shape exactly; the medial axis of a rectangle has the expected fork;
  skeleton graph of a rendered synthetic tree matches its ground-truth topology
  (noise-free, after pruning); corrected length of rendered straight lines within a few
  per cent at all angles; $E - V + C$ = number of holes. (Hessian tests live with
  `lib/edges.ts`.)
- Images: `demos/src/fourier2d/cell256.png` (credit as in `fourier2d`); synthetic trees
  generated in TS with fixed seeds (no files).
- Follow `demos/README.md`: fixed seeds, presets + reset, pointer only (no modifier keys
  or hover), keyboard back to reveal.js, labels ≥ 18 px, fits 1150×560 on a slide,
  `data-*` settings (`data-mode`, `data-image`, `data-preset`), static figures as print
  fallbacks.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture, hit-or-miss and thinning by hand, `morphgrid` (hitmiss) |
| 10:45–12:15 | Unit 2 lecture, ridges of $\mathrm{DT}$ by hand, `medialaxis` |
| 13:15–14:45 | Unit 3 lecture, skeleton graph by hand, `skelgraph` |
| 15:00–16:30 | Unit 4 lecture, Strahler and Sholl by hand, `arbor`; quizzes as wrap-up |

## Open questions

- **Decided (course map, review-organisation §2.1):** the Hessian and vesselness basics are
  taught on the edges day (day 2); unit 4 recalls them in one slide and keeps the neuron and
  vessel applications, plus a "how good is the graph" section with the freed time. The
  earlier proposal to split unit 4 into `skel-vesselness/` + `skel-applications/` is no longer
  needed.
- **Real images wanted** (synthetic trees are the fallback): a fluorescence image of a
  single neuron with its arbor (e.g. a cultured neuron; NeuroMorpho.org has tracings,
  not images, and per-dataset licences) and a 2D vessel image (retinal fundus datasets
  such as DRIVE or STARE have research-only terms; a CC-BY image or one taken by the
  course team is preferable). Licence and credit to be checked for each.
- Hit-or-miss: moved here completely (proposal, decided with this split), with only the
  one-slide pointer on the morphology day. Revisit if the morphology day wants it as a
  general "template matching" example.
- 3D: one slide on 3D thinning and 3D Frangi (proposal), or a WebGL demo of a 3D vessel
  tree from `lib/trees.ts` (centreline as tubes with radius)? 3D images as such are
  `volumes3d/` (marching-cubes elective).
- **Decided (course map):** labelling (day 8) always comes before this elective; the
  two-slide connectivity recap is dropped.
- Sholl analysis and branch order: core (proposal) or only in the `arbor` exercise?
- Programming language for any code parts is still open (see README).

## References

- Blum (1967), *A transformation for extracting new descriptors of shape*, in Models for the Perception of Speech and Visual Form, MIT Press (medial axis, grassfire).
- Lantuéjoul (1978), *La squelettisation et son application aux mesures topologiques des mosaïques*, PhD thesis, École des Mines (skeleton formula).
- Serra (1982), *Image Analysis and Mathematical Morphology*, Academic Press (hit-or-miss, thinning).
- Zhang & Suen (1984), *A fast parallel algorithm for thinning digital patterns*, CACM.
- Lam, Lee & Suen (1992), *Thinning methodologies: a comprehensive survey*, PAMI.
- Kong & Rosenfeld (1989), *Digital topology: introduction and survey*, CVGIP (simple points).
- Vossepoel & Smeulders (1982), *Vector code probability and metrication error in the representation of straight lines of finite length*, CGIP (length estimators).
- Frangi, Niessen, Vincken & Viergever (1998), *Multiscale vessel enhancement filtering*, MICCAI.
- Lindeberg (1998), *Edge detection and ridge detection with automatic scale selection*, IJCV (scale-normalised derivatives).
- Sholl (1953), *Dendritic organization in the neurons of the visual and motor cortices of the cat*, J. Anatomy.
- Arganda-Carreras et al. (2010), *3D reconstruction of histological sections: application to mammary gland tissue*, Microscopy Research and Technique (AnalyzeSkeleton: skeleton-to-graph measurements).
- Meijering (2010), *Neuron tracing in perspective*, Cytometry A.
- Lesage, Angelini, Bloch & Funka-Lea (2009), *A review of 3D vessel lumen segmentation techniques*, Medical Image Analysis.
- Bullitt et al. (2003), *Measuring tortuosity of the intracerebral vasculature from MRA images*, IEEE Trans. Medical Imaging.
