// Small view helpers shared by the 2D demos of the sparse representations day (csmri,
// sparserestore, patchdict): a captioned canvas, a style injector and a frame scheduler.
import { htmlEl } from "../lib/ui";
import { drawScalarImage, ColorMap } from "../lib/colour";

export interface ImgFig { fig: HTMLElement; cap: HTMLElement; cv: HTMLCanvasElement; draw(v: ArrayLike<number>, w: number, h: number, lo?: number, hi?: number, map?: ColorMap): void }

export function imgFig(parent: Element, caption: string, cls = ""): ImgFig {
  const fig = htmlEl("figure", cls, "", parent), cap = htmlEl("figcaption", "", caption, fig), cv = htmlEl("canvas", "", "", fig);
  return {
    fig, cap, cv,
    draw(v, w, h, lo = 0, hi = 1, map) { drawScalarImage(cv, v, w, h, { lo, hi, map }); },
  };
}

export function injectStyle(name: string, css: string): void {
  if (document.getElementById("qb-style-" + name)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + name; s.textContent = css;
  document.head.appendChild(s);
}

/** Coalesce many requests into one call per animation frame. */
export function scheduler(fn: () => void): () => void {
  let pending = 0;
  return () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; fn(); }); };
}

/** A cancellable progressive loop: `step()` returns false when done; runs ~`budget` ms per frame. */
export function progressive(step: () => boolean, after: () => void, budget = 25): { cancel(): void } {
  let alive = true;
  const run = () => {
    if (!alive) return;
    const t0 = performance.now();
    let more = true;
    while (more && performance.now() - t0 < budget) more = step();
    after();
    if (more) setTimeout(run, 0);
  };
  setTimeout(run, 0);
  return { cancel: () => { alive = false; } };
}
