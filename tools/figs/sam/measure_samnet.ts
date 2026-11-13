// Measured numbers for the Explore solutions of sam/exercise.qmd (the real network, samnet).
// Runs the same WebAssembly module and weights as the demo (demos/wasm/sam/, SAM 2.1 Hiera
// tiny) in node, applies the demo's presets with the demo's post-processing
// (demos/src/samnet/post.ts) and prints the readouts. Needs `make wasm` and `make models`, and
// the raw images written by to_raw.py:
//   python3 -I tools/figs/sam/to_raw.py <raw>
//   node_modules/.bin/esbuild tools/figs/sam/measure_samnet.ts --bundle --platform=node --log-level=warning | node - <raw>
// Takes about 5 minutes (the image encoder runs once per image, ~25 s on one core).
import * as P from "../../../demos/src/samnet/post";
import { iouTable, match, objectScores, splitsMerges } from "../../../demos/src/lib/eval";

declare const require: (m: string) => any;
declare const process: { argv: string[]; cwd(): string; exit(c: number): never };
const fs = require("fs"), path = require("path"), url = require("url");
const RAW = process.argv[2];
const root = process.cwd();

const PRESETS: Record<string, { points: [number, number, number][]; box: number[] | null }> = {
  click: { points: [[340, 317, 1]], box: null },
  cluster: { points: [[149, 428, 1]], box: null },
  box: { points: [], box: [204, 243, 267, 290] },
  negative: { points: [[149, 428, 1], [120, 410, 0]], box: null },
};
const AUTO: Record<string, Partial<P.AmgParams>> = {
  auto: { pointsPerSide: 8, predIou: 0.8, stability: 0.95, nms: 0.7 },
  strict: { predIou: 0.9, stability: 0.97, nms: 0.7 },
  loose: { predIou: 0.6, stability: 0.7, nms: 0.95 },
};

function readImage(id: string) {
  const meta = JSON.parse(fs.readFileSync(path.join(RAW, id + ".json"), "utf8"));
  const rgb = new Uint8Array(fs.readFileSync(path.join(RAW, id + ".rgb")));
  let lab: Int32Array | null = null;
  if (meta.labels) { const b = fs.readFileSync(path.join(RAW, id + ".lab")); lab = new Int32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
  return { w: meta.w as number, h: meta.h as number, rgb, lab };
}

function areaOf(m: Uint8Array): number { let n = 0; for (let i = 0; i < m.length; i++) n += m[i]; return n; }

async function main() {
  const g: any = await import(url.pathToFileURL(path.join(root, "demos/dist/wasm/sam/sam.js")).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(path.join(root, "demos/dist/wasm/sam/sam_bg.wasm"))) });
  let t = Date.now();
  g.load(new Uint8Array(fs.readFileSync(path.join(root, "demos/dist/models/sam2/sam2.1_hiera_tiny.safetensors"))));
  console.log(`load ${Date.now() - t} ms`);

  const decode = (pts: [number, number, number][], box: number[] | null, maskInput: Float32Array | null, multi = true) => {
    const t0 = Date.now();
    const logits: Float32Array = g.decode(new Float32Array(pts.flatMap((p) => [p[0], p[1]])), new Int32Array(pts.map((p) => p[2])),
      new Float32Array(box ?? []), maskInput ?? new Float32Array(0), multi);
    return { logits, iou: Array.from(g.last_iou() as Float32Array), ms: Date.now() - t0, shape: Array.from(g.last_shape() as Uint32Array) };
  };

  const runAuto = (img: { w: number; h: number }, n: number) => {
    const grid = P.pointGrid(n).map(([x, y]) => [x * img.w, y * img.h] as [number, number]);
    const cands: P.Candidate[] = [];
    const t0 = Date.now();
    for (let s = 0; s < grid.length; s += 4) {
      const pts = grid.slice(s, s + 4);
      const logits: Float32Array = g.decode_points(new Float32Array(pts.flat()));
      const [count, masks] = g.last_shape();
      cands.push(...P.batchCandidates(logits, g.last_iou(), count, masks, pts, s, img.w, img.h, 1));
    }
    return { cands, ms: Date.now() - t0 };
  };

  const describeAuto = (label: string, img: { w: number; h: number; lab: Int32Array | null }, cands: P.Candidate[], p: P.AmgParams) => {
    const f = P.filterCandidates(cands, p);
    const c = f.counts;
    const pointsKept = new Set<number>(); cands.forEach((cd, i) => { if (f.fate[i] === "kept") pointsKept.add(cd.point); });
    const big = f.kept.filter((k) => k.area > 0.5 * img.w * img.h).length;
    const areas = f.kept.map((k) => k.area).sort((a, b) => a - b);
    let line = `  ${label}: ${cands.length} cand -> iou drops ${c.iou}, stability drops ${c.stability + c.empty}, nms drops ${c.nms} -> ${c.kept} masks;` +
      ` ${pointsKept.size} grid points with a kept mask; masks > 50% of image: ${big}; areas ${areas.join(" ")}`;
    if (img.lab) {
      const pred = P.labelImage(f.kept, img.w, img.h);
      const T = iouTable(img.lab, pred);
      const m = match(T, 0.5);
      const s = objectScores(m);
      const sm = splitsMerges(T);
      line += `\n    vs truth (${T.nt} objects): TP ${s.tp} FP ${s.fp} FN ${s.fn} F1 ${s.f1.toFixed(3)}; merges (one mask over >= 2 nuclei) ${sm.merges.length}`;
    }
    console.log(line);
    return f;
  };

  // ---- nuclei: presets ----
  const nuc = readImage("nuclei");
  t = Date.now(); g.encode(nuc.rgb, nuc.w, nuc.h); console.log(`nuclei ${nuc.w}x${nuc.h}: encode ${Date.now() - t} ms`);
  for (const [name, pr] of Object.entries(PRESETS)) {
    const d = decode(pr.points, pr.box, null);
    const areas = [0, 1, 2].map((k) => areaOf(P.threshold(P.restore(P.plane(d.logits, k), nuc.w, nuc.h), 0)));
    const stab = [0, 1, 2].map((k) => P.stabilityScore(P.restore(P.plane(d.logits, k), nuc.w, nuc.h), 0, 1));
    const best = d.iou.indexOf(Math.max(...d.iou));
    console.log(`  preset ${name}: ${d.ms} ms, tokens ${d.shape[2]}; IoU scores ${d.iou.map((v) => v.toFixed(3)).join(" ")}; areas ${areas.join(" ")};` +
      ` stability ${stab.map((v) => v.toFixed(3)).join(" ")}; chosen mask ${best + 1}`);
    if (name === "click" || name === "cluster") {
      const s = decode(pr.points, pr.box, null, false);
      const a = areaOf(P.threshold(P.restore(s.logits, nuc.w, nuc.h), 0));
      console.log(`    single-mask mode: IoU score ${s.iou[0].toFixed(3)}, area ${a}`);
      const fb = decode(pr.points, pr.box, P.plane(d.logits, best).slice(), true);
      const fa = [0, 1, 2].map((k) => areaOf(P.threshold(P.restore(P.plane(fb.logits, k), nuc.w, nuc.h), 0)));
      console.log(`    fed back chosen mask: IoU scores ${fb.iou.map((v) => v.toFixed(3)).join(" ")}; areas ${fa.join(" ")}`);
    }
  }
  // Ten clicks: decoder time per click (the encoder is not rerun).
  const tens: number[] = [];
  for (const [x, y] of P.pointGrid(4).slice(0, 10)) tens.push(decode([[x * nuc.w, y * nuc.h, 1]], null, null).ms);
  console.log(`  10 clicks: ${tens.join(" ")} ms; mean ${(tens.reduce((a, b) => a + b, 0) / 10).toFixed(0)} ms; total ${tens.reduce((a, b) => a + b, 0)} ms`);
  {
    const { cands, ms } = runAuto(nuc, 8);
    console.log(`  automatic 8x8: ${ms} ms`);
    const p = { ...P.AMG_DEFAULT };
    for (const [k, v] of Object.entries(AUTO)) { Object.assign(p, v); describeAuto(k, { ...nuc, lab: null }, cands, p); }
  }
  // ---- the other sample images: encoder time, automatic masks with the default filters ----
  for (const id of ["u2os", "cells", "histology", "brightfield"]) {
    const im = readImage(id);
    t = Date.now(); g.encode(im.rgb, im.w, im.h); const enc = Date.now() - t;
    const { cands, ms } = runAuto(im, 8);
    console.log(`${id} ${im.w}x${im.h}: encode ${enc} ms, automatic 8x8 ${ms} ms`);
    describeAuto("default", im, cands, { ...P.AMG_DEFAULT });
    describeAuto("loose", im, cands, { ...P.AMG_DEFAULT, ...AUTO.loose });
    if (im.lab) {
      // Object sizes in the truth (equivalent diameter) and which are found at the default filters.
      let n = 0; for (const v of im.lab) if (v > n) n = v;
      const area = new Int32Array(n + 1);
      for (const v of im.lab) if (v) area[v]++;
      const d = Array.from(area.slice(1)).map((a) => 2 * Math.sqrt(a / Math.PI)).sort((a, b) => a - b);
      console.log(`    truth: ${n} objects, equivalent diameter min ${d[0].toFixed(1)} median ${d[d.length >> 1].toFixed(1)} max ${d[d.length - 1].toFixed(1)} px; ` +
        `one embedding cell = ${(Math.max(im.w, im.h) / 64).toFixed(1)} px of this image (longer side)`);
      // Click on the centre of every true object (as a user would): F1 of single clicks.
      if (id === "u2os") {
        const cx = new Float64Array(n + 1), cy = new Float64Array(n + 1);
        for (let y = 0; y < im.h; y++) for (let x = 0; x < im.w; x++) { const v = im.lab[y * im.w + x]; if (v) { cx[v] += x; cy[v] += y; } }
        let good = 0, small = 0, smallGood = 0, tms = 0;
        for (let k = 1; k <= n; k++) {
          const px = cx[k] / area[k], py = cy[k] / area[k];
          const dd = decode([[px, py, 1]], null, null); tms += dd.ms;
          const best = dd.iou.indexOf(Math.max(...dd.iou));
          const m = P.threshold(P.restore(P.plane(dd.logits, best), im.w, im.h), 0);
          let inter = 0, uni = 0; for (let i = 0; i < m.length; i++) { const a = m[i] ? 1 : 0, b = im.lab[i] === k ? 1 : 0; inter += a & b; uni += a | b; }
          const ok = inter / uni >= 0.5; if (ok) good++;
          const dia = 2 * Math.sqrt(area[k] / Math.PI);
          if (dia < 16) { small++; if (ok) smallGood++; }
        }
        console.log(`    one click at each true centre (best-scored mask): ${good}/${n} with IoU >= 0.5; nuclei < 16 px across: ${smallGood}/${small}; decoder ${(tms / n).toFixed(0)} ms per click`);
      }
    }
  }
}
main().catch((e) => { console.log("FAIL " + (e as Error).stack); process.exit(1); });
