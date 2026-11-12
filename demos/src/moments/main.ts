// moments: the course's shape-editing shell (labelling day; units regionprops/ and
// shape-features/). Polygons on lib/shapecanvas.ts are digitised on a pixel grid whose pixel
// size is a slider (the "resolution"), and a mode measures the mask:
//   data-mode="moments"  (unit regionprops/, default): area, centroid, box, second moments,
//                        the matrix S, eigenvalues/eigenvectors, the equivalent ellipse,
//                        Euler number, three perimeter estimators against the exact value.
//   data-mode="features" (unit shape-features/): convex hull, solidity, form factor,
//                        Feret diameters with rotating calipers and w(θ), inscribed and
//                        enclosing circles, Fourier outline, units and anisotropic pixels.
//   data-mode="higher"   (moments elective, unit moments-basis/): the triangle of moments up
//                        to data-order, marginals, skewness/kurtosis, point vs square pixels.
//   data-mode="invariants" (moments elective, unit moment-invariants/): reference vs current
//                        m, μ, η, Hu's φ; pose-normalised inset (data-bars, data-rows).
// Options: data-mode, data-preset (id of a preset of that mode), data-pixel (pixel size),
// data-presets (comma-separated ids: show only these preset buttons, e.g. on a slide),
// data-shapectl="off" (hide the draw/rotate/scale buttons; dragging still works).
//
// ---------------------------------------------------------------------------------------
// Adding a mode (the moments elective's "higher" / "invariants" / "momentclass" modes, ...)
//
// 1. Create demos/src/moments/mode-<name>.ts and register it at module load:
//
//      import { registerMode } from "./registry";
//      registerMode({
//        id: "higher",
//        build(ctx) {      // ctx.overlay (SVG group in canvas units), ctx.out (readout column),
//                          // ctx.controls (widgets), ctx.refresh(), ctx.setPixel(p), ctx.size
//          return {
//            presets: [{ id: "tri", label: "a triangle", shapes: s => [...], pixel: 1, apply() {} }],
//            update(d) { ... d.mask (d.w × d.h), d.px, d.py, d.polys, d.final ... },
//          };
//        },
//      });
//
// 2. Add `import "./mode-<name>";` below, rebuild with demos/build.sh moments, and check the
//    regionprops/ and shape-features/ pages still work.
//
// Pure helpers to reuse: ./regionprops.ts (moments, perimeters, Euler number), ./features.ts
// (hull, Feret, radii, Fourier outline), ./shapes.ts (analytic shapes, exact measures),
// ../lib/convexhull.ts.
// ---------------------------------------------------------------------------------------
import { autoMount, htmlEl, button, slider } from "../lib/ui";
import { shapeCanvas, rasterise, scalePoly, Poly } from "../lib/shapecanvas";
import { getMode, Digitised, MomentsPreset } from "./registry";
import "./mode-moments";
import "./mode-features";
import "./mode-higher";       // moments elective, unit moments-basis/
import "./mode-invariants";   // moments elective, unit moment-invariants/

const NAME = "moments";
const SIZE = 64;
const PIXELS = [0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8];

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: .9em; flex-wrap: wrap; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .mo-canvas { flex: 0 1 410px; min-width: 240px; }
.qb-demo[data-demo="${NAME}"] .mo-out { order: 2; flex: 1 1 420px; min-width: 0; overflow-x: auto; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .mo-ctl { order: 1; flex: 0 0 280px; display: flex; flex-direction: column; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .mo-lab { font-size: 15px; color: var(--qb-muted); margin-bottom: -.25em; }
.qb-demo[data-demo="${NAME}"] .mo-ctl .qb-btn { font-size: 14px; padding: .12em .45em; }
.qb-demo[data-demo="${NAME}"] .mo-ctl .qb-note { font-size: 13px; }
.qb-demo[data-demo="${NAME}"] table.mo-tab { border-collapse: collapse; font-size: 14px; width: 100%; }
.qb-demo[data-demo="${NAME}"] table.mo-tab td { padding: .05em .35em; border-bottom: 1px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] table.mo-tab td.v { font-family: ui-monospace, monospace; text-align: right; }
.qb-demo[data-demo="${NAME}"] table.mo-tab tr.h td { font-weight: 600; color: var(--qb-muted); border-bottom: 2px solid var(--qb-grid); }
.qb-demo[data-demo="${NAME}"] .mo-checks { display: flex; flex-wrap: wrap; gap: .1em .8em; font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 17px; flex-wrap: nowrap; }
.reveal .qb-demo[data-demo="${NAME}"] table.mo-tab { font-size: 15px; line-height: 1.2; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-note { font-size: 14px; }

`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function mount(host: HTMLElement): void {
  injectStyle();
  const mode = getMode(host.dataset.mode ?? "moments") ?? getMode("moments")!;
  const cbox = htmlEl("div", "mo-canvas", "", host);
  const out = htmlEl("div", "mo-out", "", host);
  const ctl = htmlEl("div", "mo-ctl", "", host);
  const modeCtl = htmlEl("div", "", "", ctl);
  modeCtl.style.cssText = "display:flex;flex-direction:column;gap:.35em";

  let px = 1, py = 1, aniso = 1, spec: ReturnType<typeof mode.build> | null = null;
  htmlEl("div", "mo-lab", "pixel size (resolution)", ctl);
  const ps = slider(ctl, "pixel", 0, PIXELS.length - 1, 1, PIXELS.indexOf(1), v => { px = PIXELS[v]; py = px * aniso; recompute(true); }, v => `${PIXELS[v]}`);
  const shapeCtl = htmlEl("div", "", "", ctl);
  shapeCtl.style.cssText = "display:flex;flex-direction:column;gap:.3em";

  const sc = shapeCanvas(cbox, {
    size: SIZE, showMask: false, controlsParent: shapeCtl, presets: false,
    onChange: (_p, _m, final) => recompute(final),
  });
  // our own mask image (the digitised grid), under the outlines
  const img = document.createElementNS("http://www.w3.org/2000/svg", "image");
  img.setAttribute("preserveAspectRatio", "none");
  img.style.setProperty("image-rendering", "pixelated");
  sc.svg.insertBefore(img, sc.svg.children[2] ?? null);
  const can = document.createElement("canvas");
  const V = 480, k = V / SIZE;

  function digitise(final: boolean): Digitised {
    const polys = sc.getPolys();
    const w = Math.ceil(SIZE / px), h = Math.ceil(SIZE / py);
    const scaled: Poly[] = polys.map(p => scalePoly(p, 1 / px, 1 / py, [0, 0]));
    const mask = rasterise(scaled, w, h);
    return { polys, mask, w, h, px, py, final };
  }

  function drawMask(d: Digitised): void {
    can.width = d.w; can.height = d.h;
    const c2 = can.getContext("2d")!, id = c2.createImageData(d.w, d.h);
    for (let i = 0; i < d.w * d.h; i++) if (d.mask[i]) { id.data[4 * i] = 110; id.data[4 * i + 1] = 140; id.data[4 * i + 2] = 185; id.data[4 * i + 3] = 165; }
    c2.putImageData(id, 0, 0);
    img.setAttribute("href", can.toDataURL());
    img.setAttribute("x", "0"); img.setAttribute("y", "0");
    img.setAttribute("width", String(d.w * d.px * k)); img.setAttribute("height", String(d.h * d.py * k));
  }

  function recompute(final: boolean): void {
    if (!spec) return;
    let d = digitise(final);
    if (spec.transformMask) d = spec.transformMask(d);
    drawMask(d);
    spec.update(d);
  }

  spec = mode.build({
    host, canvas: sc, overlay: sc.overlay, out, controls: modeCtl, size: SIZE,
    refresh: () => recompute(true),
    setPixel: (p: number) => { px = p; py = p * aniso; const i = PIXELS.indexOf(p); if (i >= 0) ps.set(i); recompute(true); },
    setAnisotropy: (a: number) => { aniso = a; py = px * a; recompute(true); },
  });

  htmlEl("div", "mo-lab", "examples", ctl);
  const prow = htmlEl("div", "qb-presets", "", ctl);
  const only = host.dataset.presets?.split(",").map(x => x.trim());
  const presets: MomentsPreset[] = only ? spec.presets.filter(p => only.includes(p.id)) : spec.presets;
  if (host.dataset.shapectl === "off") shapeCtl.style.display = "none";
  const p0 = spec.presets.find(p => p.id === host.dataset.preset) ?? presets[0];
  const btns = presets.map(p => button(prow, p.label, () => load(p)));
  button(prow, "⟲ reset", () => load(p0));
  function load(p: MomentsPreset): void {
    p.apply?.();
    px = Number(host.dataset.pixel ?? NaN);
    if (p !== p0 || !Number.isFinite(px)) px = p.pixel ?? 1;
    py = px * aniso;
    ps.set(PIXELS.indexOf(px) >= 0 ? PIXELS.indexOf(px) : PIXELS.indexOf(1));
    btns.forEach((b, i) => b.classList.toggle("active", presets[i] === p));
    sc.setPolys(p.shapes(SIZE));   // triggers onChange(final) → recompute
    sc.selected = -1;              // no vertex handles until the student clicks a shape
  }
  load(p0);
}

autoMount(NAME, mount);
