// Workaround for lib/plot.ts: its clip-path ids ("qb-clip-1", ...) come from a counter that
// is local to each demo bundle, so two demos on one page can create the same id and one
// plot is then clipped by the other's rectangle. This renames a plot's clip path to an id
// that is unique on the page. (Belongs in lib/plot.ts; reported to its owner.)
let n = 0;
export function uniqueClip(svg: SVGSVGElement, prefix: string): void {
  for (const cp of Array.from(svg.querySelectorAll("clipPath"))) {
    const old = cp.id;
    let id: string;
    do { id = `${prefix}-clip-${++n}`; } while (document.getElementById(id));
    cp.id = id;
    for (const el of Array.from(svg.querySelectorAll(`[clip-path="url(#${old})"]`))) el.setAttribute("clip-path", `url(#${id})`);
  }
}
