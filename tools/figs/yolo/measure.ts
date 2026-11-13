// Measures the yolonet demo's readouts for the yolo/ exercise solutions, with the demo's own code:
// the same letterboxing (canvas, high-quality smoothing), the same worker bundle
// (demos/dist/yolonet.js) and the same decoding/NMS (demos/src/yolonet/post.ts).
// Bundled by tools/figs/yolo/measure_yolonet.py; the page leaves its results in window.__out.
import * as P from "../../../demos/src/yolonet/post";
import { loadModelFile } from "../../../demos/src/lib/wasm";

const base = new URL("demos/dist/", location.href).href;
const w = new Worker(base + "yolonet.js");
let ready: () => void;
const readyP = new Promise<void>((r) => { ready = r; });
const pending = new Map<number, (m: any) => void>();
w.onmessage = (e) => {
  const m = e.data;
  if (m.type === "ready") ready();
  else if (m.type === "result") pending.get(m.id)!(m);
  else if (m.type === "error") (window as any).__err = m.message;
};
w.postMessage({ type: "init", base });

const KEYS: Record<string, string> = { bus: "images/yolo-bus.jpg", people: "images/yolo-zidane.jpg", nuclei: "images/stardist-img2d.png",
  u2os: "images/bbbc039-A02_s1.png", cells: "images/cellpose-img02.png", histology: "images/dsb-histology.png", brightfield: "images/dsb-brightfield.png" };

async function image(id: string, rot = 0): Promise<ImageBitmap> {
  const bytes = await loadModelFile(KEYS[id], { base });
  let bmp = await createImageBitmap(new Blob([bytes as BlobPart], { type: KEYS[id].endsWith(".jpg") ? "image/jpeg" : "image/png" }));
  if (rot) {
    const c = document.createElement("canvas"); c.width = bmp.height; c.height = bmp.width;
    const g = c.getContext("2d")!; g.translate(c.width, 0); g.rotate(Math.PI / 2); g.drawImage(bmp, 0, 0);
    bmp = await createImageBitmap(c);
  }
  const sc = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));
  if (sc < 1) bmp = await createImageBitmap(bmp, { resizeWidth: Math.round(bmp.width * sc), resizeHeight: Math.round(bmp.height * sc), resizeQuality: "high" });
  return bmp;
}

let seq = 0;
async function run(bmp: ImageBitmap, size: number) {
  const lb = P.letterbox(bmp.width, bmp.height, size);
  const c = document.createElement("canvas"); c.width = c.height = size;
  const g = c.getContext("2d")!;
  g.fillStyle = "#000"; g.fillRect(0, 0, size, size); g.imageSmoothingQuality = "high";
  g.drawImage(bmp, lb.x0, lb.y0, lb.nw, lb.nh);
  const input = P.toInput(g.getImageData(0, 0, size, size).data, size);
  const id = ++seq;
  const m = await new Promise<any>((r) => { pending.set(id, r); w.postMessage({ type: "run", id, size, input: input.buffer }, [input.buffer]); });
  return { lb, raw: new Float32Array(m.raw), dec: new Float32Array(m.decoded), netMs: m.netMs, size };
}

function sd(p: number[]): number { let m = 0, v = 0; p.forEach((x, k) => m += k * x); p.forEach((x, k) => v += x * (k - m) ** 2); return Math.sqrt(v); }

function report(r: Awaited<ReturnType<typeof run>>, conf: number, iou: number, agn: boolean, scales = [true, true, true]) {
  const n = P.nms(r.dec, conf, iou, 300, agn);
  const sc = P.scales(r.size), scaleOf = (a: number) => (a < sc[1].start ? 0 : a < sc[2].start ? 1 : 2);
  const per = [0, 0, 0]; n.cands.forEach((c) => per[scaleOf(c.anchor)]++);
  const kept = n.kept.filter((k) => scales[scaleOf(n.cands[k].anchor)]).map((k) => {
    const c = n.cands[k], b = P.toImage(r.dec, r.lb, c.anchor * P.ND), cell = P.anchorCell(c.anchor, r.size);
    let removed = 0; for (let j = 0; j < n.cands.length; j++) if (n.by[j] === k) removed++;
    const sds = [0, 1, 2, 3].map((s) => +sd(P.softmax(r.raw, c.anchor * P.NO + s * 16)).toFixed(2));
    const dist = [0, 1, 2, 3].map((s) => +r.dec[c.anchor * P.ND + 4 + s].toFixed(2));
    return { cls: P.COCO[c.cls], score: +c.score.toFixed(2), stride: cell.stride, row: cell.i, col: cell.j,
      box: b.map((v) => Math.round(v)), w: Math.round(b[2] - b[0]), h: Math.round(b[3] - b[1]), removed, dist, sd: sds };
  });
  let maxp = 0, maxcls = ""; const best = P.bestClass(r.dec);
  for (let a = 0; a < best.p.length; a++) if (best.p[a] > maxp) { maxp = best.p[a]; maxcls = P.COCO[best.cls[a]]; }
  return { conf, iou, agn, cands: n.cands.length, perStride: per, kept: kept.length, list: kept, maxScore: +maxp.toFixed(3), maxCls: maxcls, netMs: Math.round(r.netMs) };
}

async function main() {
  await readyP;
  const out: Record<string, any> = {};
  const bus = await image("bus"), people = await image("people");
  const b512 = await run(bus, 512);
  out.bus_default = report(b512, 0.25, 0.45, false);
  out.bus_raw = report(b512, 0.1, 0.45, false);
  out.bus_nonms = report(b512, 0.25, 1, false);
  out.bus_agn = report(b512, 0.25, 0.45, true);
  out.bus_net2 = (await run(bus, 512)).netMs;
  const p512 = await run(people, 512);
  out.people_default = report(p512, 0.25, 0.45, false);
  out.people_agn = report(p512, 0.25, 0.45, true);
  out.people_strict = report(p512, 0.25, 0.1, true);
  out.people_nonms = report(p512, 0.25, 1, false);
  const p256 = await run(people, 256);
  out.people_256 = report(p256, 0.25, 0.45, false);
  const b256 = await run(bus, 256);
  out.bus_256 = report(b256, 0.25, 0.45, false);
  for (const id of ["nuclei", "u2os", "cells", "histology", "brightfield"]) {
    const r = await run(await image(id), 512);
    out[id + "_default"] = report(r, 0.25, 0.45, false);
    out[id + "_low"] = report(r, 0.01, 0.45, false);
  }
  const brot = await run(await image("bus", 1), 512);
  out.bus_rot90 = report(brot, 0.25, 0.45, false);
  const prot = await run(await image("people", 1), 512);
  out.people_rot90 = report(prot, 0.25, 0.45, false);
  (window as any).__out = out;
}
main().catch((e) => { (window as any).__err = String(e); });
