# Interactive demos

Small TypeScript demos embedded in slides and exercise pages. Not rendered by Quarto
(`demos/` is excluded); the built scripts in `demos/dist/` are copied into `docs/`
as project resources.

## Build

```sh
npm i              # once: installs esbuild + typescript into node_modules/
make demos         # build every demo into demos/dist/  (make render does this first)
demos/build.sh foo # build only demos/src/foo/
make typecheck     # tsc --noEmit (esbuild does not check types)
```

## Layout

- `demos/src/<name>/main.ts`: one demo per folder; bundled to `demos/dist/<name>.js`
  as an IIFE (ES modules do not load from `file://`).
- `demos/src/lib/ui.ts`: shared helpers (sliders, buttons, dragging, seeded RNG, auto-mount).
- `demos/demo.css`: shared styles, copied to `demos/dist/demo.css`.
- Images imported in TS (`import img from "./x.jpg"`) are inlined as data URLs, so
  canvas pixel reads work from `file://` too.

## Embedding

In a `.qmd` (slide or page), one raw HTML block:

````md
```{=html}
<link rel="stylesheet" href="../demos/dist/demo.css">
<div data-demo="<name>" data-foo="..."></div>
<script src="../demos/dist/<name>.js"></script>
```
````

## Rules

- No network: no CDN libraries, no `fetch`. Plain DOM + SVG/canvas.
  **One exception:** demos that run a trained network (StarDist, Cellpose, YOLO, SAM) or a
  large Wasm module load those files **same-origin** from the course site
  (`demos/dist/models/`, `demos/dist/wasm/`), lazily, with a progress bar, cached in the
  browser's Cache API, and only through `lib/wasm.ts`. Never a third-party host. Such a demo
  shows a clear message when the page is opened as a local file (`file://`), and its page
  keeps a static figure as the fallback.
- Fixed random seeds (`rng(seed)` from `lib/ui.ts`), so a recording is reproducible.
- Give the keyboard back to reveal.js: use the `slider`/`button`/`checkbox` helpers,
  which blur themselves; never bind arrow keys or space.
- Slides: fit in about 1150×560 px; labels ≥ 18 px; large drag handles; a reset or
  preset button; nothing that only shows on hover. Keep the static figure in the
  deck or exercise as the fallback for PDF/print.
- Read per-embed settings from `data-*` attributes, so one demo serves several units.
- 3D: WebGL2, through the shared renderer in `lib/gl3d.ts` (no 3D libraries).

## Fitting the screen

A demo is only usable when the figure and the controls that change it are on screen together.
Every demo must fit a 1366×768 laptop window and a 1280×720 slide; `make check-fit` (after
`make render`; `tools/check/fit.py`) measures every demo on the pages at 1366×768 and 1920×1080
and on the slides at 1280×720, and lists those that break these rules:

- **Height:** the whole demo is at most the window height minus 60 px (on a slide: 660 px of
  the 1280×720 slide, and it must not overflow the slide; aim for the 1150×560 content area).
- **Controls beside the figure, not under it.** Put the controls in a `.qb-panel` column next
  to the main figure. A single row of buttons spanning the demo under a wide figure is fine; a
  control column that wraps below the figure is not.
- **Width:** on pages, `autoMount` (`lib/ui.ts`) puts every demo that sits directly in the text
  (not inside a callout, tabset or columns) into Quarto's wide grid column
  (`.column-screen-inset-right`: about 1075 px wide in a 1366 px window, 1630 px in a 1920 px one), and Quarto
  folds the table of contents into an "On this page" button while it is on screen. Design for
  that width (figure about 400–550 px, plots about 300 px, panel about 300 px), and still
  degrade gracefully to the 800 px text column (`data-width="body"` keeps a demo there) and to a
  phone (one column below 640 px).
- **Shared limits in `demo.css`:** figures (`canvas`, `svg`) are at most `100vh - 200px` high
  (SVGs keep their aspect ratio through the `viewBox`, canvases through `object-fit: contain`;
  a demo that maps pointer positions on a canvas whose box can be letterboxed must size the
  canvas itself so that it never is); a `.qb-panel` is at most `100vh - 100px` high and scrolls
  beyond that. On slides the limits are in slide pixels (figures ≤ 560 px).
- Do not make a figure grow with the width of the demo without a cap (`max-width`, or a
  `flex-basis` with no `flex-grow`): a wider column must not make the demo taller.

## Language: TypeScript, and Rust/WebAssembly when speed is needed

- Write demos in **TypeScript**, not plain JavaScript. `make typecheck` must pass.
- Start in TypeScript even for heavy computation; measure first. Move a kernel to
  **Rust compiled to WebAssembly** only when TypeScript is too slow for interactive
  use (rule of thumb: a recomputation that should feel live takes over ~30 ms,
  e.g. iterative PDE updates or max-flow on large images, 3D volumes above ~64³).
- Keep the Rust part small and numeric: plain functions over flat arrays
  (`&[f32]`, `&mut [f32]`, `&[u8]`), no DOM or UI in Rust. UI, presets and drawing
  stay in TypeScript, which calls the Wasm function.
- Keep a TypeScript reference implementation of the same function, and test that
  the Wasm and TypeScript versions agree on small inputs.

## WebAssembly (Rust crates)

- **Layout.** One Cargo crate per module in `demos/wasm/<name>/` (`crate-type = ["cdylib"]`,
  dependency `wasm-bindgen = "=<CLI version>"`, release profile `opt-level = "s"`, `lto`,
  `panic = "abort"`; commit its `Cargo.lock`). Example: `demos/wasm/hello/`.
- **Build.** `make wasm` (= `demos/wasm/build.sh [name …]`) compiles each crate for
  `wasm32-unknown-unknown` (shared target dir `demos/wasm/target/`), runs
  `wasm-bindgen --target web` and writes `demos/dist/wasm/<name>/<name>.js` (glue),
  `<name>.d.ts` (types) and `<name>_bg.wasm`, plus `demos/dist/wasm/index.json` (bytes and
  sha256 per module, the cache version). `demos/dist/wasm/` is **git-ignored**; CI builds it.
  `demos/build.sh` builds the crates a demo imports first when their output is missing or stale.
  Run `make wasm` before `make typecheck` (the types of the glue come from the build).
- **Toolchain (once).** `rustup target add wasm32-unknown-unknown` and
  `cargo install wasm-bindgen-cli --version <the version pinned in the crates> --locked`
  (the build stops with this hint when they differ). `wasm-opt` is used when installed.
- **Using a module in a demo.** Always bundle the glue statically, then choose how the bytes
  arrive:
  - *Inlined* (small modules, up to a few hundred kB; works from `file://`):
    ```ts
    import * as glue from "../../dist/wasm/maxflow/maxflow.js";
    import wasm from "../../dist/wasm/maxflow/maxflow_bg.wasm";   // bytes (binary loader)
    import { instantiateInline } from "../lib/wasm";
    await instantiateInline(glue, wasm);   // async: Chrome refuses sync compiles > 4 kB
    glue.maxflow(...);
    ```
  - *Fetched* (large modules): `await loadWasm(glue, "stardist", { onProgress })` fetches
    `demos/dist/wasm/stardist/stardist_bg.wasm` same-origin (Cache API, keyed on its sha256).
  The glue keeps one instance per page; instantiate once, at start-up. An inlined module ends
  up inside the committed `demos/dist/<demo>.js`, so it works on any copy of the site; a
  fetched module or model exists only on the site deployed by CI (and locally after
  `make wasm` / `make models`; `quarto render` copies them into `docs/demos/dist/`, where
  they are git-ignored too).
- Keep the Rust side numeric (flat arrays in, flat arrays out); wrap a library crate (e.g. the
  henriksson-lab ports) behind a few such functions.
- **Tests.** `demos/test/wasm-test.ts` (in `make test`) loads the built glue in node and
  compares with TypeScript references; it skips with a message when `make wasm` has not run
  (`WASM_REQUIRED=1`, as in CI, makes that a failure). Each crate may have a browser check page
  `demos/wasm/<name>/example/{index.html,main.ts}`, built to
  `demos/dist/wasm/<name>/example/` (e.g. `/demos/dist/wasm/hello/example/?mode=inline`).

## Models and sample images

- `tools/models/manifest.json` lists the checkpoints and public microscopy images (URL, file,
  bytes, sha256, credit, what loads it, conversion steps); `make models`
  (= `python3 tools/models/fetch.py`) downloads into `.cache/models/` and writes the converted
  files (safetensors in f16, PNG images, 16-bit label PNGs) into `demos/dist/models/` with an
  `index.json` (key = path under `models/`, bytes, sha256, credit). Both directories are
  **git-ignored**; the deploy workflow (`.github/workflows/deploy.yml`) runs the same script
  and publishes them. Details: `tools/models/README.md`.
- In a demo: `const bytes = await loadModelFile("yolo/yolo11n-coco.safetensors", { onProgress })`
  (from `lib/wasm.ts`), with `loadStatus(parent, "Loading network")` as the progress bar and
  `isFileProtocol()` / `fileProtocolMessage()` for the `file://` case. Show the image credit
  (`index.json` has it) next to a sample image.
- Size budget: the whole site must stay well under GitHub Pages' ~1 GB (CI fails above
  950 MB). Currently about 110 MB of models and images.
