# Plan: StarDist, Cellpose, YOLO-family and SAM in the course (2026-11-13)

Status: proposal for the user. Nothing built yet.

**Course map (2026-11-13, self-study restructure):** day 40 of 51, Part VII "Learning from examples". Hard prerequisites: Modern deep learning (day 39), Evaluation and statistics (day 37). Soft (see also): Watershed (day 15), Edge-preserving smoothing and PDEs (day 10), Graph cuts (day 27). Source: `days.yml`.

**Update 2026-11-13:** the course is self-study with no cap on days and **no electives** (see
`plan/coursemap.md`, "Self-study restructure"). So the question of where the day goes
is settled: "Networks for cells and objects" is day 40, placed after modern deep learning (day 39) in Part VII.

## What exists already

- `instances/` (day 39, modern deep learning): the four routes from a semantic mask to objects —
  boundary class, distance map + watershed, **star-convex polygons** (StarDist's idea), **flow
  fields** (Cellpose's idea) — with simulated "network outputs" (`instancemaps` demo).
- `detection/` (day 39): boxes, IoU, **anchors vs centre heat maps**, greedy NMS and its failures
  (YOLO's ingredients), `nmsboxes` demo.
- `attention/` (day 39): promptable segmentation (SAM's idea) as one slide.
- `graphcut-seg/` (day 27): the `graphcut` demo **does** run on image data (micrograph preset
  "nucleus", seed painting), but opens on a synthetic preset and is laggy on the micrograph
  (100–200 ms per update: graph and intensity model rebuilt every time).
- Rust ports in github.com/henriksson-lab: `stardist-rs`, `cellpose-rs` (incl. Cellpose-SAM via
  candle), `YOLOv11-rs` (candle; training and inference), `sam2-rs` (candle; SAM 2.1, image,
  video/z-slice, automatic masks), plus `deconwolf-rs`, `itk-rs`, `bioformats-rs` and others that
  could serve later demos.

## Recommendation: one new day, built around real models in the browser

**Day 40 (new, Part VII): "Networks for cells and objects"**, directly after modern deep
learning. The named model families are methods (like Canny), so naming them is fine under
principle 0; what we avoid is teaching a tool's interface. The day's angle: *what does each
network predict, and how do you get objects out of it?* — representation and decoding are the
theory, and they are exercise-friendly.

Reorganise day 39 at the same time so nothing is taught twice:

| Day 39 Modern deep learning (after) | Day 40 Networks for cells and objects (new) |
|---|---|
| `attention/` (patches, attention, transfer, foundation models in general) | 1. `stardist/`: star-convex polygons — rays, the distance targets, probability map, polygon NMS, what shapes it cannot represent (absorbs Route 3 of `instances/`) |
| `diffusion/` (introduction) | 2. `cellpose/`: flows from a simulated diffusion inside each mask, gradient tracking to sinks, flow error check, cell diameter and rescaling, generalist training on many image types; Cellpose-SAM as a pointer to unit 4 (absorbs Route 4) |
| `instances/` shrinks to the overview: why post-processing, routes 1–2, a one-slide pointer per route 3–4 to day 40, panoptic | 3. `yolo/`: **directly from image to a list of objects**: the grid, what each output cell predicts (box offsets, objectness, classes), decoding the output tensor, multi-scale heads (feature pyramid), anchor-based vs anchor-free heads, the loss (box IoU-type loss, classification, distribution focal loss as idea), NMS, speed vs accuracy; detection of cells/organelles/colonies (absorbs the anchor/heat-map part of `detection/`) |
| `detection/` shrinks to boxes, IoU, NMS failures (general) | 4. `sam/`: promptable segmentation — a heavy image encoder run once, a light prompt encoder + mask decoder run per click; points/boxes/masks as prompts; ambiguity and three candidate masks with scores; automatic mask generation (a grid of point prompts + filtering + NMS); SAM 2's memory for video/z-stacks; failure modes on microscopy and why fine-tuned variants exist |

Day 40 ends with one **comparison unit or section**: the same images through a classical pipeline
(threshold + watershed), StarDist, Cellpose and SAM; evaluated with the evaluation day's matching
(`object-eval/`); cost (parameters, time per image in the browser).

(No day variant: every day is part of the one sequence.)

## Exercises (the hard part)

The theory per model is thin, so exercises target **representation, decoding and judgement**:

- *By hand (concept):* draw the 8-ray star-convex description of a given shape and find where it
  fails (non-star-convex, holes); compute the polygon area from ray lengths; follow a 5×5 flow field
  from three start pixels and see which ones meet; decode one YOLO output cell (grid position,
  offsets, scale) into a box in pixels and compute its IoU with another; apply NMS by hand to six
  scored boxes / polygons; decide which of SAM's three masks a click on a nucleus edge should give.
- *Math questions:* why rays give a star-convex but not convex shape; the flow field as the gradient
  of a diffusion (heat equation recall from the PDE day); IoU-based box losses; why the image
  encoder can be cached (cost per prompt).
- *With real models in the browser (Explore):* parameter sweeps with a prediction (probability and
  NMS thresholds; Cellpose diameter and flow threshold; YOLO confidence; SAM point placement),
  failure hunting (dense clusters, very large/small objects, unusual contrast, a rotated or
  flipped image, an out-of-domain photo), and a scored comparison against ground truth with the
  matching demo. Questions ask students to predict before they run.

## Demos

Two kinds:

1. **Small analytic demos in TypeScript** (as now): star-convex rays on a drawable shape; flow
   field from a mask (computed by heat diffusion) and particle tracking; YOLO grid decoder on a
   hand-editable output tensor; SAM-style prompt/decoder toy with cached embedding.
2. **Real models via Rust → WebAssembly**, built from the henriksson-lab crates:
   `stardistwasm`, `cellposewasm`, `yolowasm`, `samwasm`. Each shows the **intermediate outputs**,
   not just the final masks: StarDist probability map and rays; Cellpose flow field, cell
   probability and the tracked sinks; YOLO raw candidates per scale before/after NMS; SAM
   per-prompt masks with scores (encoder once, decoder live). Built-in sample images plus "use your
   own image" (local file, never uploaded).

### Engineering needed (new infrastructure)

- **Wasm build pipeline**: `demos/wasm/<name>/` crates (wasm32-unknown-unknown + wasm-bindgen)
  wrapping each library's inference API; `make wasm`; outputs `.wasm` + JS glue into
  `demos/dist/`. `demos/build.sh` gets a `.wasm` path (currently missing). Small kernels can stay
  inlined; large modules load separately.
- **Weights via CI (decided 2026-11-13)**: weights are **not committed**. A GitHub Actions
  workflow fetches the published checkpoints at deploy time (StarDist pretrained 2D models,
  Cellpose models, YOLO weights, SAM 2.1 tiny checkpoint), converts/quantises them (f16 or int8,
  with the conversion tools in the henriksson-lab crates, e.g. YOLOv11-rs `convert_weights.py`),
  and places them in `docs/demos/dist/models/` of the deployed site. Downloads and converted files
  are cached between runs (`actions/cache`, keyed on model URL + converter version), so each
  checkpoint is fetched once. Locally, `make models` runs the same script into the same place
  (git-ignored).
  - This means **switching GitHub Pages from "deploy from branch /docs" to "GitHub Actions"**
    (`actions/upload-pages-artifact` + `actions/deploy-pages`). Side benefits: no Jekyll step at
    all (the earlier `.nojekyll` failure cannot happen), and CI can also build the Wasm modules
    (`make wasm`) and run `make typecheck` + `make test` before deploying, so a broken commit
    does not get deployed. Whether CI also runs the Quarto render (needs Quarto + Node + Rust in
    the runner) or keeps using the committed `docs/` is a choice: start with committed `docs/` +
    CI adding wasm and models, move the render into CI later if wanted.
  - In the browser the demos fetch weights same-origin, lazily, with a progress bar, cached in the
    Cache API; a clear message when opened from `file://`. This relaxes the "no runtime network"
    rule for these demos only (same-origin, no third parties).
  - Size: deployed site must stay under GitHub Pages' ~1 GB; quantised StarDist/YOLO-nano are a
    few MB, Cellpose tens of MB, SAM 2.1 tiny/Cellpose-SAM ~40–80 MB each.
- **Speed**: CPU Wasm first; measure. Expect StarDist/YOLO-nano well under a second on 512²,
  Cellpose seconds, the SAM image encoder several seconds once per image (decoder per click is
  fast). WebGPU (via the frameworks' wgpu backends) as a later optimisation.
- **Licences**: not a concern for this course (decided 2026-11-13: use whatever works; mixing
  AGPL is fine). Use the YOLOv11-rs weights as they are.
- **Sample images (decided 2026-11-13)**: any public microscopy data; the choice is ours. Use
  well-known public nuclei/cell image sets (e.g. Broad Bioimage Benchmark Collection sets, the
  2018 Data Science Bowl nuclei images, the models' own example/demo data), fetched by the same
  CI/`make models` step as the weights, plus synthetic images with ground truth from
  `lib/synth.ts`. The same images can replace the "real image wanted" gaps listed in
  `plan/build-issues.md` across other days.

## Graph cut on image data (small fix, independent)

- Open the `graphcut` demo on the micrograph preset (or add a second embed with
  `data-preset="nucleus"` in `graphcut-seg/`), make the image choice prominent.
- Fix the lag: cache the graph structure and the intensity model; rebuild only the t-links when
  seeds change. **If it is still slow (>30 ms per update on 256² images), move the max-flow and
  graph construction to Rust → WebAssembly** (a small crate in `demos/wasm/maxflow/`, same API as
  `lib/maxflow.ts`, tested against it). This is also the natural first, small use of the Wasm
  pipeline and can serve as the spike for the infrastructure before the large models. Then allow
  larger images (512²) in the demo.
- Later: run it on the same real fluorescence images as day 40, so students can compare graph
  cuts with the networks.

## Order of work

0. **Self-study restructure** (`plan/coursemap.md`) can run independently of everything below.
1. **Wasm spike (one agent, first):** start with the graph-cut max-flow crate (small, known
   answers), then compile `stardist-rs` and `YOLOv11-rs` inference to
   wasm32, load quantised weights same-origin, run on a 512² image in headless Chrome; report
   size, load time, run time, blockers (non-Rust dependencies in stardist-rs; the Burn fork in
   cellpose-rs/YOLOv11-rs). Decides the infrastructure.
2. **Infrastructure**: `make wasm`, `make models`, the GitHub Actions deploy workflow (switch
   Pages to Actions), model loader/cache, `file://` message, demos/README rules, size budget.
3. **Graph-cut fix** (independent, can run in parallel with 1).
4. **Plan and build day 40** (parallel per unit after the infrastructure exists), and the
   day-18 reorganisation (`instances/` and `detection/` slimmed, pointers).
5. **Image pass** for day 40, integration, full render, checks.

## Decisions for the user

1. ~~Where does the day go?~~ Decided: no electives; a day after modern deep learning.
2. ~~Where to host weights?~~ Decided: fetched and converted in CI at deploy time, not committed.
3. ~~YOLO weights?~~ Decided: use them; licences are not a concern.
4. ~~Which real image sets?~~ Decided: any public microscopy data, fetched in CI.
