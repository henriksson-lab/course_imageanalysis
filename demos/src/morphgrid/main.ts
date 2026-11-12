// morphgrid: binary morphology on a small grid, step by step (morphology day), built on the
// grid step-through UI of lib/gridui.ts. Paint the image by clicking/dragging cells, choose
// an operation, then step, play or jump to the end; back steps backwards.
//
// Auto-mounts on <div data-demo="morphgrid">. Options:
//   data-mode    "binary" (default; erosion, dilation, boundary, duality, ...),
//                "openclose" (opening, closing, combinations, idempotence),
//                "distance" (two-pass distance transform), or a mode added by another day.
//   data-preset  id of a preset of that mode (default: its first preset).
//   data-frame   step to show first (a number, or "end").
//
// ---------------------------------------------------------------------------------------
// Adding a mode (e.g. the labelling day's "label" mode, the skeleton day's "hitmiss" mode)
//
// 1. Create demos/src/morphgrid/mode-<name>.ts and register the mode at module load:
//
//      import { registerMode } from "./registry";
//      registerMode({
//        id: "label",                         // the value of data-mode
//        build(ctx) {                         // called once per embed
//          // ctx.side: empty column left of the grids for your widgets (SE editor, 4/8
//          //   toggle, ...); hidden if you leave it empty. ctx.host.dataset: data-*.
//          // ctx.restart(): regenerate the frames after a setting changed (back to step 0).
//          // ctx.stepper(): the GridStepper (valid inside callbacks, after build returns).
//          return {                           // a ModeSpec (registry.ts)
//            algorithms: [...],               // GridAlgorithm[] of lib/gridui.ts (one button each)
//            display: "binary", edit: "toggle", cell: 36,   // any GridStepperOptions except rows/cols/initial/presets
//            map: undefined,                  // optional colour map for "int" display
//            presets: [{ id: "u", label: "a U needs one equivalence", grid: ["0110", "0110", "1111"], algorithm: "twopass", apply() { ... } }],
//            ready(stepper) { ... },          // optional: add controls to stepper.panel
//          };
//        },
//      });
//
// 2. Add one line `import "./mode-<name>";` to the list below, rebuild with
//    demos/build.sh morphgrid, and check the morphology pages still work.
//
// The shell (this file) builds the side column, the stepper (grids, step controls, log),
// the preset buttons and a reset button (back to the data-preset, or the first preset).
// Reusable pieces: ./se.ts (structuring-element editor with origin, SE presets), ./ops.ts
// (pixel-by-pixel binary erosion/dilation with a border value, small chamfer DT).
// ---------------------------------------------------------------------------------------
import { autoMount, htmlEl, button } from "../lib/ui";
import { gridStepper, gridFromText, GridStepper } from "../lib/gridui";
import { getMode, ModeContext, ModePreset } from "./registry";
import "./mode-binary";
import "./mode-distance";

const NAME = "morphgrid";
const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; flex-wrap: nowrap; }
.qb-demo[data-demo="${NAME}"] .mg-side { flex: 0 0 190px; display: flex; flex-direction: column; gap: .45em; }
.qb-demo[data-demo="${NAME}"] .mg-side:empty { display: none; }
.qb-demo[data-demo="${NAME}"] .mg-se { display: flex; flex-direction: column; gap: .35em; }
.qb-demo[data-demo="${NAME}"] .mg-se .qb-grid-svg { max-width: 180px !important; }
.qb-demo[data-demo="${NAME}"] .mg-lab { font-size: 15px; color: var(--qb-muted); }
.qb-demo[data-demo="${NAME}"] .qb-gridstep { flex: 1 1 auto; min-width: 0; flex-wrap: nowrap; }
.qb-demo[data-demo="${NAME}"] .qb-gridstep-grids { flex: 1 1 380px; }
.qb-demo[data-demo="${NAME}"] .qb-gridstep-panel { flex: 0 0 300px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .qb-gridstep-log { max-height: 6.5em; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 18px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 15px; padding: .15em .45em; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-gridstep-panel { gap: .3em; }
.reveal .qb-demo[data-demo="${NAME}"] .mg-lab { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-gridstep-log { max-height: 4.2em; font-size: 13px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-gridstep-grids { max-height: 540px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-gridstep-cell .qb-grid-svg { max-height: 520px; }
@media (max-width: 900px) {
  .qb-demo[data-demo="${NAME}"], .qb-demo[data-demo="${NAME}"] .qb-gridstep { flex-wrap: wrap; }
}
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function gridOf(p: ModePreset): { rows: number; cols: number; values: Int32Array } {
  if (Array.isArray(p.grid)) return gridFromText(p.grid);
  return { rows: p.grid.rows, cols: p.grid.cols, values: Int32Array.from(p.grid.values) };
}

function mount(host: HTMLElement): void {
  injectStyle();
  const mode = getMode(host.dataset.mode ?? "binary") ?? getMode("binary")!;
  const side = htmlEl("div", "mg-side", "", host);
  let stepper: GridStepper | null = null;
  const ctx: ModeContext = {
    host, side,
    stepper: () => stepper!,
    restart: () => { if (stepper) stepper.setAlgorithm(stepper.algorithm.id); },
  };
  const spec = mode.build(ctx);
  const p0 = spec.presets.find(p => p.id === host.dataset.preset) ?? spec.presets[0];
  p0?.apply?.();
  const g0 = p0 ? gridOf(p0) : { rows: 8, cols: 8, values: new Int32Array(64) };
  const { presets, map, ready, ...opts } = spec;
  stepper = gridStepper(host, { ...opts, rows: g0.rows, cols: g0.cols, initial: g0.values, algorithm: p0?.algorithm ?? opts.algorithm });
  const s = stepper;
  if (map) { s.main.opts.map = map; s.main.draw(); }

  let btns: HTMLButtonElement[] = [];
  const load = (p: ModePreset) => {
    p.apply?.();
    const g = gridOf(p);
    const a = spec.algorithms.find(q => q.id === p.algorithm);
    if (a) s.algorithm = a;
    s.setInput(g.values, g.rows, g.cols);
    if (p.frame === "end") s.toEnd(); else if (typeof p.frame === "number") s.goTo(p.frame);
    btns.forEach((b, i) => b.classList.toggle("active", presets[i] === p));
  };
  if (presets.length) {
    htmlEl("div", "mg-lab", "examples", s.panel);
    const row = htmlEl("div", "qb-presets", "", s.panel);
    btns = presets.map(p => button(row, p.label, () => load(p)));
    button(row, "⟲ reset", () => load(p0));
    btns.forEach((b, i) => b.classList.toggle("active", presets[i] === p0));
  }
  // highlight the chosen algorithm button (gridStepper does it on regenerate)
  if (p0?.algorithm) s.setAlgorithm(p0.algorithm);
  const fr = host.dataset.frame ?? p0?.frame;
  if (fr === "end") s.toEnd(); else if (fr !== undefined && Number.isFinite(Number(fr))) s.goTo(Number(fr));
  ready?.(s);
}

autoMount(NAME, mount);
