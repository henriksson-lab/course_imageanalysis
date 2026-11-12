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
- Fixed random seeds (`rng(seed)` from `lib/ui.ts`), so a recording is reproducible.
- Give the keyboard back to reveal.js: use the `slider`/`button`/`checkbox` helpers,
  which blur themselves; never bind arrow keys or space.
- Slides: fit in about 1150×560 px; labels ≥ 18 px; large drag handles; a reset or
  preset button; nothing that only shows on hover. Keep the static figure in the
  deck or exercise as the fallback for PDF/print.
- Read per-embed settings from `data-*` attributes, so one demo serves several units.
- 3D: WebGL2, through the shared renderer in `lib/gl3d.ts` (no 3D libraries).

## Language: TypeScript, and Rust/WebAssembly when speed is needed

- Write demos in **TypeScript**, not plain JavaScript. `make typecheck` must pass.
- Start in TypeScript even for heavy computation; measure first. Move a kernel to
  **Rust compiled to WebAssembly** only when TypeScript is too slow for interactive
  use (rule of thumb: a recomputation that should feel live takes over ~30 ms,
  e.g. iterative PDE updates or max-flow on large images, 3D volumes above ~64³).
- Keep the Rust part small and numeric: plain functions over flat arrays
  (`&[f32]`, `&mut [f32]`, `&[u8]`), no DOM or UI in Rust. UI, presets and drawing
  stay in TypeScript, which calls the Wasm function.
- Layout: `demos/wasm/<name>/` as a Cargo crate (`crate-type = ["cdylib"]`, target
  `wasm32-unknown-unknown`), built to a `.wasm` file that the demo imports. The
  `.wasm` bytes are inlined into the demo bundle (esbuild `--loader:.wasm=binary`)
  and instantiated with `WebAssembly.instantiate(bytes)`: no `fetch`, so it still
  works from `file://`.
- Keep a TypeScript reference implementation of the same function, and test that
  the Wasm and TypeScript versions agree on small inputs.
- Toolchain (once): `rustup target add wasm32-unknown-unknown`.
