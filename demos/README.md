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
