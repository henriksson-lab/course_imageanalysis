// Static figures (print fallbacks) of the diffusion-models day, computed with the demos' own
// model code. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/diffusion/figs.ts --bundle --platform=node --log-level=warning | node
// Writes score-matching/fig-pairs.svg, diffusion-sampling/fig-paths.svg,
// guidance/fig-guidance.svg and diffusion-inverse/fig-nullspace.svg.
import { writeFileSync } from "fs";
import * as SM from "../../../demos/src/scorematch1d/model";
import * as SP from "../../../demos/src/samplerpaths/model";
import * as G from "../../../demos/src/guidance1d/model";
import * as P from "../../../demos/src/posterior1d/model";

const C = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"];

class Panel {
  out: string[] = [];
  constructor(public x0: number, public y0: number, public w: number, public h: number, public xr: [number, number], public yr: [number, number], opts: { xlabel?: string; ylabel?: string; title?: string; xticks?: number[]; yticks?: number[] } = {}) {
    const o = this.out;
    o.push(`<rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="none" stroke="#444" stroke-width="1"/>`);
    for (const t of opts.xticks ?? []) o.push(`<line x1="${this.X(t)}" x2="${this.X(t)}" y1="${y0 + h}" y2="${y0 + h + 5}" stroke="#444"/><text x="${this.X(t)}" y="${y0 + h + 20}" text-anchor="middle">${t}</text>`);
    for (const t of opts.yticks ?? []) o.push(`<line x1="${x0 - 5}" x2="${x0}" y1="${this.Y(t)}" y2="${this.Y(t)}" stroke="#444"/><text x="${x0 - 8}" y="${this.Y(t) + 5}" text-anchor="end">${t}</text>`);
    if (opts.xlabel) o.push(`<text x="${x0 + w / 2}" y="${y0 + h + 40}" text-anchor="middle">${opts.xlabel}</text>`);
    if (opts.ylabel) o.push(`<text x="${x0 - 42}" y="${y0 + h / 2}" text-anchor="middle" transform="rotate(-90 ${x0 - 42} ${y0 + h / 2})">${opts.ylabel}</text>`);
    if (opts.title) o.push(`<text x="${x0 + w / 2}" y="${y0 - 8}" text-anchor="middle" font-weight="600">${opts.title}</text>`);
  }
  X = (v: number) => +(this.x0 + (v - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.w).toFixed(1);
  Y = (v: number) => +(this.y0 + this.h - (v - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.h).toFixed(1);
  clipY = (v: number) => Math.max(this.yr[0], Math.min(this.yr[1], v));
  line(xs: ArrayLike<number>, ys: ArrayLike<number>, col: string, width = 2.5, dash = "", opacity = 1): void {
    let d = "";
    for (let i = 0; i < xs.length; i++) if (Number.isFinite(ys[i])) d += (d ? "L" : "M") + this.X(xs[i]) + "," + this.Y(this.clipY(ys[i]));
    this.out.push(`<path d="${d}" fill="none" stroke="${col}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ""} opacity="${opacity}"/>`);
  }
  dots(xs: ArrayLike<number>, ys: ArrayLike<number>, col: string, r = 2.2, opacity = 0.6): void {
    for (let i = 0; i < xs.length; i++) if (xs[i] >= this.xr[0] && xs[i] <= this.xr[1] && ys[i] >= this.yr[0] && ys[i] <= this.yr[1]) this.out.push(`<circle cx="${this.X(xs[i])}" cy="${this.Y(ys[i])}" r="${r}" fill="${col}" opacity="${opacity}"/>`);
  }
  band(xs: ArrayLike<number>, lo: ArrayLike<number>, hi: ArrayLike<number>, col: string, op = 0.2): void {
    let d = "";
    for (let i = 0; i < xs.length; i++) d += (i ? "L" : "M") + this.X(xs[i]) + "," + this.Y(this.clipY(hi[i]));
    for (let i = xs.length - 1; i >= 0; i--) d += "L" + this.X(xs[i]) + "," + this.Y(this.clipY(lo[i]));
    this.out.push(`<path d="${d}Z" fill="${col}" opacity="${op}" stroke="none"/>`);
  }
  hist(edges: number[], dens: number[], col: string): void {
    for (let b = 0; b < dens.length; b++) this.out.push(`<rect x="${this.X(edges[b])}" y="${this.Y(this.clipY(dens[b]))}" width="${(this.X(edges[b + 1]) - this.X(edges[b])).toFixed(1)}" height="${(this.Y(this.yr[0]) - this.Y(this.clipY(dens[b]))).toFixed(1)}" fill="${col}" opacity="0.45"/>`);
  }
  text(x: number, y: number, s: string, col = "#222", anchor = "start"): void { this.out.push(`<text x="${this.X(x)}" y="${this.Y(y)}" fill="${col}" text-anchor="${anchor}">${s}</text>`); }
}

function svg(W: number, H: number, panels: Panel[], extra = ""): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="sans-serif" font-size="16">\n<rect width="${W}" height="${H}" fill="#fff"/>\n` +
    panels.map(p => p.out.join("\n")).join("\n") + extra + "\n</svg>\n";
}
const grid = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

// ---- unit 1: pairs, binned average, exact denoiser (two modes at ±1.5, σ = 0.6)
{
  const d = SM.data1d("two"), s = 0.6, ex = SM.exactDenoiser(d), Pp = SM.pairs(d, s), bn = SM.binned(Pp.x, Pp.x0, -4, 4, 20);
  const p = new Panel(70, 20, 560, 300, [-4, 4], [-2.5, 2.5], { xlabel: "noisy value x = x₀ + σε (σ = 0.6)", ylabel: "clean value x₀", xticks: [-4, -2, 0, 2, 4], yticks: [-2, -1, 0, 1, 2] });
  p.dots(Pp.x, Pp.x0, "#888", 2.2, 0.5);
  const sx: number[] = [], sy: number[] = [];
  for (let b = 0; b < 20; b++) if (Number.isFinite(bn.m[b])) { sx.push(bn.c[b] - 0.2, bn.c[b] + 0.2); sy.push(bn.m[b], bn.m[b]); }
  for (let i = 0; i < sx.length; i += 2) p.line([sx[i], sx[i + 1]], [sy[i], sy[i + 1]], C[2], 4);
  const xs = grid(-4, 4, 200);
  p.line(xs, xs.map(x => ex(x, s)), C[0], 3);
  p.text(-3.8, 2.1, "grey: 400 training pairs", "#555"); p.text(-3.8, 1.7, "green: binned average", C[2]); p.text(-3.8, 1.3, "blue: exact best denoiser", C[0]);
  writeFileSync("score-matching/fig-pairs.svg", svg(650, 380, [p]));
}

// ---- unit 2: deterministic and stochastic paths, 100 steps, two modes; 5 deterministic steps
{
  const m = SP.data("two");
  const mk = (x0: number, title: string, eta: number, steps: number) => {
    const p = new Panel(x0, 30, 330, 260, [1, 0], [-3, 3], { xlabel: "time τ (noise → data)", ylabel: x0 < 100 ? "position" : undefined, title, xticks: [1, 0.5, 0], yticks: [-2, 0, 2] });
    const r = SP.run(m, { steps, eta, spacing: "time", twoStage: false, langevin: 0, seed: 1, n: 30, nPaths: 30 });
    for (const path of r.paths) p.line(r.taus, path, C[0], 1.3, "", 0.7);
    return p;
  };
  const a = mk(70, "deterministic, 100 steps", 0, 100), b = mk(450, "stochastic, 100 steps", 1, 100), c = mk(830, "deterministic, 5 steps", 0, 5);
  writeFileSync("diffusion-sampling/fig-paths.svg", svg(1180, 350, [a, b, c]));
}

// ---- unit 3: class densities, sharpened density and guided samples at w = 3
{
  const c = G.classes("two"), xs = grid(-5, 5, 300);
  const p = new Panel(70, 20, 600, 280, [-5, 5], [0, 0.75], { xlabel: "x", ylabel: "density", xticks: [-4, -2, 0, 2, 4], yticks: [0, 0.25, 0.5, 0.75] });
  const X = G.guidedSamples(c, 0, 3, 100, 1);
  const edges = grid(-5, 5, 81), h = 10 / 80, cnt = new Array(80).fill(0);
  for (const v of X) { const b = Math.floor((v + 5) / h); if (b >= 0 && b < 80) cnt[b] += 1 / (X.length * h); }
  p.hist(edges, cnt, C[0]);
  p.line(xs, xs.map(x => G.classDensity(c, 0, x)), C[0], 2.5, "3 4");
  p.line(xs, xs.map(x => G.classDensity(c, 1, x)), C[1], 2.5);
  const sh = G.sharpened(c, 0, 3, xs);
  p.line(xs, sh, "#222", 3, "8 5");
  p.text(1.6, 0.68, "class B", C[1]); p.text(1.6, 0.62, "class A (dotted)", C[0]); p.text(1.6, 0.56, "guided samples, w = 3", C[0]); p.text(1.6, 0.50, "sharpened density", "#222");
  writeFileSync("guidance/fig-guidance.svg", svg(700, 360, [p]));
}

// ---- unit 4: low-pass below the cut-off: truth, measurement, samples, unseen part
{
  const m: P.Meas = { id: "lowpass", blur: 2, cutoff: 6, gap: [26, 38], noise: 0.05 };
  const x = P.testSignal(4), y = P.measure(x, m, 3), tr = P.trainingSet("matched"), post = P.posterior(tr, y, m);
  const S = P.postSamples(post, 32, 7), b = P.band(S), xs = Array.from({ length: P.N }, (_, i) => i);
  const a = new Panel(70, 30, 520, 200, [0, 63], [-0.2, 1.4], { ylabel: "signal", title: "truth (dashed), measurement (grey), posterior mean (blue)", yticks: [0, 0.5, 1] });
  a.line(xs, P.measuredSignal(y, m), "#999", 3); a.line(xs, P.postMeanSignal(post), C[0], 3); a.line(xs, x, "#222", 2.5, "6 4");
  const bp = new Panel(70, 270, 520, 200, [0, 63], [-0.2, 1.4], { ylabel: "samples", title: "5 posterior samples and the 90 % band", yticks: [0, 0.5, 1] });
  bp.band(xs, b.lo, b.hi, C[1], 0.18); for (const s of S.slice(0, 5)) bp.line(xs, s, C[1], 1.8, "", 0.7); bp.line(xs, x, "#222", 2.5, "6 4");
  const cp = new Panel(70, 510, 520, 160, [0, 63], [-0.6, 0.6], { xlabel: "position", ylabel: "unseen", title: "the part the measurement cannot see (from the prior)", xticks: [0, 16, 32, 48, 63], yticks: [-0.5, 0, 0.5] });
  for (const s of S.slice(0, 5)) cp.line(xs, P.nullPart(s, m), C[1], 1.8, "", 0.7); cp.line(xs, P.nullPart(x, m), "#222", 2.5, "6 4");
  writeFileSync("diffusion-inverse/fig-nullspace.svg", svg(620, 730, [a, bp, cp]));
}
console.log("figures written");
