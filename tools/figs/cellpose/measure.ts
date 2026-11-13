// Readouts of the real Cellpose demo (cellposenet) for the Explore solutions of cellpose/exercise.qmd.
// Runs the same route as the demo (PNG, channels, percentile normalisation, rescaling, tiles,
// the Wasm network, the Wasm decoding, lib/eval matching at IoU 0.5) in node.
// Needs `make wasm` and `make models`. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/cellpose/measure.ts --bundle --platform=node --log-level=warning | node
// Output is stored next to this script: `... | node > measure.txt`, and with the argument
// `sweep` (`... | node - sweep > measure-sweep.txt`) the diameter sweep and more step counts.
import * as P from "../../../demos/src/cellposenet/pipeline";
import { decodePng } from "../../../demos/src/cellposenet/png";
import { iouTable, match, objectScores, splitsMerges } from "../../../demos/src/lib/eval";

declare const require: (m: string) => any;
declare const process: { cwd(): string };
const fs = require("fs"), path = require("path"), url = require("url"), zlib = require("zlib");

interface Sample { key: string; file: string; labels?: string; model: P.ModelKey; chan: [number, number]; diameter: number; invert?: boolean }
// as SAMPLES in demos/src/cellposenet/main.ts
const S: Record<string, Sample> = {
  cells: { key: "cells", file: "images/cellpose-img02.png", model: "cyto3", chan: [2, 3], diameter: 30 },
  crowded: { key: "crowded", file: "images/cellpose-img00.png", model: "cyto3", chan: [2, 3], diameter: 35 },
  membranes: { key: "membranes", file: "images/cellpose-img03.png", model: "cyto3", chan: [0, 0], diameter: 34 },
  nuclei: { key: "nuclei", file: "images/bbbc039-A02_s1.png", labels: "images/bbbc039-A02_s1-labels.png", model: "nuclei", chan: [0, 0], diameter: 29 },
  dense: { key: "dense", file: "images/dsb-fluo-dense.png", labels: "images/dsb-fluo-dense-labels.png", model: "nuclei", chan: [0, 0], diameter: 22 },
  small: { key: "small", file: "images/dsb-fluo-small.png", labels: "images/dsb-fluo-small-labels.png", model: "nuclei", chan: [0, 0], diameter: 13 },
  tissue: { key: "tissue", file: "images/dsb-histology.png", labels: "images/dsb-histology-labels.png", model: "nuclei", chan: [0, 0], diameter: 24, invert: true },
};

const root = process.cwd();
const mdir = path.join(root, "demos/dist/models");
let g: any;
let loadedModel = "";

async function loadPng(file: string) {
  return decodePng(new Uint8Array(fs.readFileSync(path.join(mdir, file))), async (z) => new Uint8Array(zlib.inflateSync(z)));
}

async function image(s: Sample) {
  const png = await loadPng(s.file);
  const { w, h, channels } = png;
  const nc = channels >= 3 ? 3 : 1;
  const planes = Array.from({ length: nc }, () => new Float32Array(w * h));
  for (let i = 0; i < w * h; i++) for (let c = 0; c < nc; c++) planes[c][i] = png.data[i * channels + c];
  let labels: Uint32Array | undefined;
  if (s.labels) { const lp = await loadPng(s.labels); labels = Uint32Array.from(lp.data.filter((_: number, i: number) => i % lp.channels === 0)); }
  return { img: { w, h, planes } as P.Img, labels };
}

function useModel(m: P.ModelKey): void {
  if (loadedModel === m) return;
  g.load_network(new Uint8Array(fs.readFileSync(path.join(mdir, P.MODELS[m].file))));
  loadedModel = m;
}

interface Run { net: P.NetOutput; w: number; h: number; labels?: Uint32Array }
async function runNet(s: Sample, opt: { diameter?: number; invert?: boolean; chan?: [number, number]; model?: P.ModelKey } = {}): Promise<Run> {
  const { img, labels } = await image(s);
  const model = opt.model ?? s.model;
  useModel(model);
  const ch = P.toChannels(img, opt.chan ?? s.chan);
  const inv = opt.invert ?? !!s.invert;
  const inp: [Float32Array, Float32Array] = [P.normalize99(ch[0], inv), P.normalize99(ch[1], inv)];
  const d = opt.diameter ?? s.diameter;
  const net = await P.runNetwork(inp, img.w, img.h, P.MODELS[model].diamMean / d, async (x, th, tw) => ({ y: g.run_tile(x, th, tw), style: g.last_style() }));
  return { net, w: img.w, h: img.h, labels };
}

interface Dec { cellprob?: number; flow?: number; minSize?: number; niter?: number; diameter?: number }
function decode(r: Run, d: Dec, model: P.ModelKey, diameter: number): string {
  const niter = d.niter ?? Math.min(400, P.defaultNiter(P.MODELS[model].diamMean / diameter));
  const cp = d.cellprob ?? 0, fl = d.flow ?? 0.4, ms = d.minSize ?? 15;
  const t0 = Date.now();
  g.compute_masks(r.net.dy, r.net.dx, r.net.cellprob, r.h, r.w, niter, cp, fl, ms, 0.4);
  const ms1 = Date.now() - t0;
  const masks: Uint32Array = g.last_masks();
  const errs: Float32Array = g.last_flow_errors();
  const seeds = g.last_seeds().length;
  let count = 0; for (const v of masks) if (v > count) count = v;
  const removedFlow = fl > 0 ? Array.from(errs).filter((e) => e > fl).length : 0;
  const sorted = Array.from(errs).sort((a, b) => a - b);
  const q = (p: number) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))].toFixed(2) : "-";
  let s = `steps ${niter}, cellprob ${cp}, flow ${fl}, min ${ms}: fg ${g.last_n_foreground()}, seeds ${seeds}, masks ${errs.length} -> flow check removes ${removedFlow} -> ${count} cells; ` +
    `flow error median ${q(0.5)}, 90% ${q(0.9)}, max ${q(0.999)}; median diameter ${P.medianDiameter(masks).toFixed(1)}; decode ${ms1} ms`;
  if (r.labels) {
    const T = iouTable(r.labels, masks);
    const sc = objectScores(match(T, 0.5));
    const sm = splitsMerges(T);
    s += `\n      vs GT (${T.nt}): F1 ${sc.f1.toFixed(3)} (TP ${sc.tp}, FP ${sc.fp}, FN ${sc.fn}); splits ${sm.splits.length}, merges ${sm.merges.length}`;
  }
  return s;
}

async function sweep(): Promise<void> {
  // diameter sweep (E5, the diameter figure)
  for (const d of [8, 10, 12, 20, 40, 80, 100]) {
    const r = await runNet(S.nuclei, { diameter: d });
    console.log(`nuclei diameter ${d}: work ${r.net.workW}x${r.net.workH}\n    ${decode(r, {}, "nuclei", d)}`);
  }
  for (const d of [8, 10, 100]) {
    const r = await runNet(S.cells, { diameter: d });
    console.log(`cells diameter ${d}: work ${r.net.workW}x${r.net.workH}: ${decode(r, {}, "cyto3", d)}`);
  }
  const r = await runNet(S.cells);
  console.log(`cells 10 steps, no flow check: ${decode(r, { niter: 10, flow: 0 }, "cyto3", 30)}`);
  console.log(`cells 20 steps: ${decode(r, { niter: 20 }, "cyto3", 30)}`);
}

async function init(): Promise<void> {
  const dir = path.join(root, "demos/dist/wasm/cellpose");
  g = await import(url.pathToFileURL(path.join(dir, "cellpose.js")).href);
  g.initSync({ module: new Uint8Array(fs.readFileSync(path.join(dir, "cellpose_bg.wasm"))) });
}

async function main(): Promise<void> {
  const out = (s: string) => console.log(s);
  const sizeModel: P.SizeModel = JSON.parse(fs.readFileSync(path.join(mdir, "cellpose/size_cyto3.json"), "utf8"));

  // E2-E4: cells
  {
    const s = S.cells;
    const t0 = Date.now();
    const r = await runNet(s);
    out(`cells ${r.w}x${r.h} diameter 30: network ${Date.now() - t0} ms, rescaled ${r.net.workW}x${r.net.workH}, ${r.net.nTiles} tiles`);
    out(`  defaults:      ${decode(r, {}, s.model, 30)}`);
    out(`  only 10 steps: ${decode(r, { niter: 10 }, s.model, 30)}`);
    out(`  50 steps:      ${decode(r, { niter: 50 }, s.model, 30)}`);
    out(`  no flow check: ${decode(r, { flow: 0 }, s.model, 30)}`);
    out(`  flow 0.2:      ${decode(r, { flow: 0.2 }, s.model, 30)}`);
    out(`  flow 0.8:      ${decode(r, { flow: 0.8 }, s.model, 30)}`);
    out(`  more cells:    ${decode(r, { cellprob: -2, flow: 0.8 }, s.model, 30)}`);
    out(`  stricter:      ${decode(r, { cellprob: 1, flow: 0.2, minSize: 30 }, s.model, 30)}`);
    out(`  style -> size model: ${P.diameterFromStyle(r.net.style, sizeModel).toFixed(1)} px`);
    for (const d of [15, 60]) {
      const r2 = await runNet(s, { diameter: d });
      out(`  diameter ${d}: work ${r2.net.workW}x${r2.net.workH}: ${decode(r2, {}, s.model, d)}`);
    }
  }
  // E5: nuclei at default, half, double
  for (const d of [29, 15, 58]) {
    const s = S.nuclei;
    const t0 = Date.now();
    const r = await runNet(s, { diameter: d });
    out(`nuclei diameter ${d}: work ${r.net.workW}x${r.net.workH}, ${r.net.nTiles} tiles, network ${Date.now() - t0} ms\n    ${decode(r, {}, s.model, d)}`);
  }
  // E5b: nuclei image through cyto3 at diameter 30 for the size model
  {
    const r = await runNet(S.nuclei, { model: "cyto3", diameter: 30 });
    out(`nuclei through cyto3 at 30: size model ${P.diameterFromStyle(r.net.style, sizeModel).toFixed(1)} px`);
  }
  // E6: small, cellprob -2, 0, +2
  {
    const s = S.small;
    const r = await runNet(s);
    for (const cp of [-2, 0, 2]) out(`small cellprob ${cp}: ${decode(r, { cellprob: cp }, s.model, s.diameter)}`);
  }
  // E7: tissue with/without invert; membranes; dense; crowded
  {
    const s = S.tissue;
    for (const inv of [true, false]) {
      const r = await runNet(s, { invert: inv });
      out(`tissue invert ${inv}: ${decode(r, {}, s.model, s.diameter)}`);
    }
    const rm = await runNet(S.membranes);
    out(`membranes: ${decode(rm, {}, "cyto3", 34)}`);
    const rd = await runNet(S.dense);
    out(`dense: ${decode(rd, {}, "nuclei", 22)}`);
    const rc = await runNet(S.crowded);
    out(`crowded: ${decode(rc, {}, "cyto3", 35)}`);
  }
}

const ARGS: string[] = (globalThis as any).process?.argv ?? [];
(ARGS.includes("sweep") ? init().then(sweep) : init().then(main)).catch((e) => { console.log("ERROR", e); });
