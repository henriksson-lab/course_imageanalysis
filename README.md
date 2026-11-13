# Image analysis: course site sources

Teacher information. This file is not part of the published site.

The site structure, demo framework and styling are adapted from the
*Quantitative biology and integrative omics* (3MB00A) course site.

- [audience.md](audience.md): who we are designing for.
- [designprinciples.md](designprinciples.md): how we build material.
- [demos/README.md](demos/README.md): how the interactive demos are built and embedded.

## Layout

One folder per unit, each with

- `lec.qmd`: a reveal.js slide deck (keep it small; one topic per deck),
- `quiz.qmd`: short questions with collapsible hints and answers,
- `exercise.qmd`: pen-and-paper questions, then an `## Explore {#explore}` section
  that embeds the unit's interactive demo, with questions about it.

Every demo also goes on `demos.qmd`, the gallery page.

### Course structure: `days.yml`

The course is one self-study sequence of days in themed parts (no electives). The structure
lives in **`days.yml`** at the repo root, not in folder names or by hand in `_quarto.yml`:
parts → days (id, global number, title, plan file) → units (folder, title, demo labels). Each
day lists **hard** prerequisites (must come first) and **soft** ones (see also), as day ids or
unit folders written with a trailing slash (`pca/`); a unit may add its own. Days or units
marked `status: planned` are not built yet. `plan/coursemap.md` keeps the rationale.

- `make check-prereqs` (`tools/days.py check`): every unit folder exists with its three pages
  and every unit folder on disk is listed, every prerequisite exists, the hard prerequisites
  have no cycles and come earlier in the order, the numbers run 1…N, and the sidebar is current.
- `make sidebar` (`tools/days.py sidebar`): regenerates the sidebar block of `_quarto.yml`
  (Home, All demos, Overview if `overview.qmd` exists, a Topics section with `topics/index.qmd`
  and every other page in `topics/`, then parts → days → units → Lecture /
  Quiz / Exercise / one entry per demo anchor `{#explore…}` on the exercise page whose section
  embeds a demo; labels from the unit's `demos:` map, else from the heading) and the per-part
  colours in `styles.css`. Never edit the generated block by hand; to reorder, add or rename,
  edit `days.yml` and run `make sidebar`.
- **Prerequisite links on every page**: the Lua filter `tools/prereqs.lua` (registered under
  `filters:` in `_quarto.yml`) reads `days.yml` at render time. Quiz and exercise pages get a
  box "Before this unit:" (the previous unit of the day, the day's and the unit's hard
  prerequisites) and "See also:" (soft ones); each lecture gets one small line "Builds on: …" at
  the end of its "About this lecture" slide. Links go to the lecture of the unit (a day links to
  its first unit). Nothing is hand-maintained in the units: change `days.yml` and re-render.
- **Overview page** `overview.qmd`: the interactive prerequisite graph (`demos/src/coursegraph/`,
  learning paths, search) reads `demos/src/coursegraph/days.json`, and the page's plain list view
  sits between the `GENERATED LIST` markers. Both are written from `days.yml` by
  `tools/days_json.py`, which `make sidebar` runs; `make check-prereqs` reports them when stale.
  After a change, rebuild the graph with `demos/build.sh coursegraph`.

### Marking the math layer

Every unit has a concept layer and a math layer (design principle 0). Mark the math
layer the same way everywhere:

- **Slides:** the math goes on its own slide(s), right after the concept slide it
  formalises, titled `## The math: <topic> {.math}`. The `.math` class (styled in
  `theme.scss`) adds a tinted bar on the left and a "math layer" badge after the
  title. The "About this lecture" slide ends with the line *Slides titled **The
  math** give the formulas and proofs; the ideas are complete without them.* Concept
  slides must make sense without any math slide (the summary slide too).
- **Pages** (exercise background, longer explanations): a collapsed callout
  `::: {.callout-note collapse="true" title="The math: <topic>"}`.
- **Exercises:** concept questions (predict, explain, small calculations with
  numbers) come first, with their own `Solutions` callout. Math questions (derive,
  prove, generalise) follow under `### Math questions`, each tagged
  `**B7.** *(math)*`, with a separate `Solutions (math)` callout. Every exercise has
  concept questions; if a topic is mostly proofs, add concept versions with numbers.
  Learning goals that need the math are tagged *(math)* too.
- **Quizzes:** mostly concept questions; tag a purely mathematical one
  `**Q5.** *(math)*`.

## Building this site

```bash
npm i          # once: installs esbuild + typescript for the demos
make render    # build the demos and render the whole site to docs/
make preview   # start a live-reload preview server
make clean     # delete docs/, demos/dist/ and the Quarto cache
make test      # correctness tests of the demo algorithms (demos/test/)
make check     # after a render: broken links/anchors, then console errors on every page
```

You will need [Quarto](https://quarto.org/docs/get-started/) and Node.js. The network demos
also need `make wasm` (Rust, the wasm32 target, wasm-bindgen-cli) and `make models` (Python);
their outputs are git-ignored and added by CI at deploy time (`.github/workflows/deploy.yml`).

`make check` runs `tools/check/links.py` (every local `href`/`src` in `docs/` exists, every
`#anchor` names an id; links left as `.qmd` count as broken) and `tools/check/console.sh`
(serves `docs/` on a free local port and opens every page in headless Chrome over the DevTools
protocol: uncaught exceptions, also in Web Workers, `console.error` and failed loads count;
`JOBS=` and `WAIT=` set the parallel browsers and the seconds per page). Single pages:
`tools/check/console.sh canny/lec.html`. Needs Google Chrome and Python `websocket-client`.

## Adding a demo

1. Create `demos/src/<name>/main.ts`; use the helpers in `demos/src/lib/ui.ts` and
   mount with `autoMount("<name>", host => ...)`.
2. `demos/build.sh <name>` and `make typecheck`.
3. Embed it in a unit (see `demos/README.md`), add it to `demos.qmd`, give its anchor a
   sidebar label in `days.yml` if the heading is not enough, and run `make sidebar`.
