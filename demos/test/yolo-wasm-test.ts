// Tests for the yolonet demo's decoding and post-processing (demos/src/yolonet/post.ts) and the
// YOLO11n Wasm module (demos/wasm/yolo/, built by `make wasm` into demos/dist/wasm/yolo/):
// 1. TypeScript reference: anchor grids, decoding of hand-made head outputs, greedy NMS on hand
//    boxes, letterbox geometry.
// 2. Wasm decode() and nms() against the TypeScript reference on random head outputs.
// 3. The whole network in Wasm on a sample micrograph (stardist-img2d.png, from `make models`),
//    against the native YOLOv11-rs crate (same vendored model on the host CPU,
//    demos/wasm/yolo/examples/reference.rs): the crate's NMS output at score > 0.01 and > 0.25.
// Run:
//   node_modules/.bin/esbuild demos/test/yolo-wasm-test.ts --bundle --platform=node --log-level=warning | node
// Parts 2 and 3 are skipped with a message when the module or model files are missing (set
// WASM_REQUIRED=1, as CI does, to make that a failure).
import * as P from "../src/yolonet/post";

declare const require: (m: string) => any;
declare const process: { exit(code: number): never; cwd(): string; env: Record<string, string | undefined> };
const fs = require("fs"), path = require("path"), url = require("url"), zlib = require("zlib");

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let e = 0;
  for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]));
  return e;
}
function skip(what: string): boolean {
  if (process.env.WASM_REQUIRED) { check(false, what + " missing (run make wasm / make models)"); return true; }
  console.log(`SKIP ${current}: ${what} missing (run make wasm / make models)`);
  return true;
}

// Seeded uniform numbers (mulberry32).
function rand(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- 1. TypeScript reference --------------------------------------------------------------------
current = "grids";
{
  check(P.anchorCount(512) === 64 * 64 + 32 * 32 + 16 * 16 && P.anchorCount(640) === 8400, "anchor counts 512 / 640");
  const sc = P.scales(256);
  check(sc.map((s) => `${s.stride}:${s.g}:${s.start}`).join(" ") === "8:32:0 16:16:1024 32:8:1280", "scales(256)");
  const c = P.anchorCell(1024 + 16 * 3 + 5, 256);
  check(c.scale === 1 && c.i === 3 && c.j === 5 && c.stride === 16, "anchorCell");
  check(P.COCO.length === 80 && P.COCO[0] === "person" && P.COCO[5] === "bus" && P.COCO[32] === "sports ball", "COCO names");
}

current = "decode";
{
  // Hand-made output for size 64 (8x8 + 4x4 + 2x2 = 84 anchors): every side's distribution is
  // nearly one-hot at a chosen bin, so the decoded distance is that bin.
  const size = 64, n = P.anchorCount(size), raw = new Float32Array(n * P.NO);
  const bins = (a: number) => [a % 16, (a * 3) % 16, (a * 5 + 1) % 16, (a * 7 + 2) % 16];
  for (let a = 0; a < n; a++) {
    bins(a).forEach((k, side) => { raw[a * P.NO + side * P.REG_MAX + k] = 40; });
    for (let c = 0; c < P.NC; c++) raw[a * P.NO + 4 * P.REG_MAX + c] = c === a % P.NC ? 2 : -3;
  }
  const dec = P.decode(raw, size);
  let errD = 0, errB = 0, errP = 0;
  for (let a = 0; a < n; a++) {
    const { i, j, stride } = P.anchorCell(a, size), d = bins(a), o = a * P.ND;
    for (let k = 0; k < 4; k++) errD = Math.max(errD, Math.abs(dec[o + 4 + k] - d[k]));
    const want = [(j + 0.5 - d[0]) * stride, (i + 0.5 - d[1]) * stride, (j + 0.5 + d[2]) * stride, (i + 0.5 + d[3]) * stride];
    for (let k = 0; k < 4; k++) errB = Math.max(errB, Math.abs(dec[o + k] - want[k]));
    for (let c = 0; c < P.NC; c++) errP = Math.max(errP, Math.abs(dec[o + 8 + c] - 1 / (1 + Math.exp(c === a % P.NC ? -2 : 3))));
  }
  check(errD < 1e-5 && errB < 1e-3 && errP < 1e-6, `one-hot distances: dist ${errD}, box ${errB}, prob ${errP}`);
  // Uniform logits: expected distance 7.5 bins.
  const flat = P.decode(new Float32Array(n * P.NO), size);
  check(close(flat[4], 7.5, 1e-5) && close(flat[8], 0.5, 1e-7), "uniform distribution: distance 7.5, probability 0.5");
  check(close(P.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3, 1e-9) && P.iou([0, 0, 1, 1], [2, 2, 3, 3]) === 0, "iou");
}

// Decoded rows from boxes and (class, score) pairs, for NMS tests.
function decodedFrom(rows: { box: P.Box; probs: [number, number][] }[]): Float32Array {
  const d = new Float32Array(rows.length * P.ND);
  rows.forEach((r, a) => { d.set(r.box, a * P.ND); for (const [c, p] of r.probs) d[a * P.ND + 8 + c] = p; });
  return d;
}

current = "nms";
{
  const d = decodedFrom([
    { box: [0, 0, 10, 10], probs: [[0, 0.9]] }, // A: best
    { box: [1, 1, 11, 11], probs: [[0, 0.8], [1, 0.7]] }, // B (class 0): IoU with A 81/119 = 0.68; C (class 1)
    { box: [20, 20, 30, 30], probs: [[0, 0.6]] }, // D: apart
    { box: [0, 0, 10, 10], probs: [[0, 0.3]] }, // E: below the threshold
    { box: [21, 21, 31, 31], probs: [[0, 0.55]] }, // F: IoU with D 81/119 = 0.68
  ]);
  const r = P.nms(d, 0.5, 0.45);
  const name = (k: number) => `${r.cands[k].anchor}/${r.cands[k].cls}`;
  check(r.cands.length === 5, "five candidates above 0.5 (one anchor gives two)");
  check(r.kept.map(name).join(" ") === "0/0 1/1 2/0", "class-aware: keeps A, C, D: " + r.kept.map(name).join(" "));
  check(r.by[1] === 0 && r.by[4] === 3, "B removed by A, F removed by D");
  const g = P.nms(d, 0.5, 0.45, 300, true);
  check(g.kept.map((k) => `${g.cands[k].anchor}/${g.cands[k].cls}`).join(" ") === "0/0 2/0", "class-agnostic: keeps A, D");
  check(P.nms(d, 0.5, 0.7).kept.length === 5, "IoU threshold 0.7: nothing removed (0.68 < 0.7)");
  check(P.nms(d, 0.5, 0.45, 2).kept.length === 2 && P.nms(d, 0.5, 0.45, 2).by[3] === -2, "max detections");
  const rows = P.nmsRows(d, r);
  check(rows.length === 21 && close(rows[7 + 4], 0.7, 1e-6) && rows[7 + 5] === 1 && rows[7 + 6] === 1 && rows[14] === 20, "nmsRows layout");
}

current = "letterbox";
{
  const lb = P.letterbox(810, 1080, 512); // the bus photo
  check(close(lb.r, 512 / 1080, 1e-12) && lb.nw === 384 && lb.nh === 512 && lb.x0 === 64 && lb.y0 === 0, "portrait into 512");
  const b = P.toImage([64, 0, 448, 512], lb);
  check(maxDiff(b, [0, 0, 810, 1080]) < 1e-9, "input box -> image box");
  const x = P.toInput(new Uint8Array([255, 0, 51, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255]), 2);
  check(maxDiff(x, [1, 0, 0, 1, 0, 1, 0, 1, 0.2, 0, 0, 1]) < 1e-7, "RGBA -> CHW input");
}

// ---- 2 + 3. The Wasm module ----------------------------------------------------------------------
// Minimal PNG reader for 8-bit greyscale, non-interlaced files (the sample micrograph).
function readGreyPng(buf: Uint8Array): { w: number; h: number; data: Uint8Array } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8, w = 0, h = 0;
  const idat: Uint8Array[] = [];
  while (o < buf.length) {
    const len = dv.getUint32(o), type = String.fromCharCode(...buf.subarray(o + 4, o + 8));
    const body = buf.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      w = dv.getUint32(o + 8); h = dv.getUint32(o + 12);
      if (body[8] !== 8 || body[9] !== 0 || body[12] !== 0) throw new Error("only 8-bit grey, non-interlaced PNG");
    } else if (type === "IDAT") idat.push(body);
    o += 12 + len;
  }
  const z = new Uint8Array(zlib.inflateSync(Buffer.concat(idat)));
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const f = z[y * (w + 1)], row = z.subarray(y * (w + 1) + 1, (y + 1) * (w + 1));
    for (let x = 0; x < w; x++) {
      const a = x ? data[y * w + x - 1] : 0, b = y ? data[(y - 1) * w + x] : 0, c = x && y ? data[(y - 1) * w + x - 1] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      data[y * w + x] = (row[x] + pred) & 255;
    }
  }
  return { w, h, data };
}

// The native crate (demos/wasm/yolo/examples/reference.rs) on stardist-img2d.png (512 x 512 grey,
// copied into R, G and B, divided by 255): its NMS (`model::nms::non_max_suppression`) at score
// > 0.01, IoU 0.45. Rows x1 y1 x2 y2 score class. At > 0.25 it finds nothing (no cell class in COCO).
const NATIVE_NUCLEI_001: number[][] = [
  [141.225, 492.931, 170.254, 511.734, 0.0451, 0], [138.037, 477.939, 182.625, 511.649, 0.03042, 0],
  [452.054, 480.794, 481.691, 511.778, 0.02696, 0], [236.581, 472.467, 292.126, 511.678, 0.02338, 0],
  [12.61, 478.281, 48.424, 511.761, 0.0198, 0], [-0.305, -11.518, 511.355, 510.184, 0.01903, 5],
  [471.096, 428.802, 505.995, 461.122, 0.01429, 32], [0.094, 441.018, 20.574, 511.544, 0.0141, 0],
  [179.664, 450.428, 210.677, 475.544, 0.01161, 32], [448.953, 428.068, 511.33, 511.635, 0.0114, 0],
];

async function wasmTests(): Promise<void> {
  current = "wasm decode/nms";
  const dir = path.join(process.cwd(), "demos/dist/wasm/yolo");
  const wasmFile = path.join(dir, "yolo_bg.wasm"), glueFile = path.join(dir, "yolo.js");
  if (!fs.existsSync(wasmFile) || !fs.existsSync(glueFile)) { skip("demos/dist/wasm/yolo"); return; }
  const g: any = await import(url.pathToFileURL(glueFile).href);
  g.initSync({ module: fs.readFileSync(wasmFile) });
  check(g.anchors(512) === 5376 && g.anchors(640) === 8400, "anchors()");
  const r = rand(7);
  for (const size of [64, 160]) {
    const n = P.anchorCount(size), raw = new Float32Array(n * P.NO);
    for (let i = 0; i < raw.length; i++) raw[i] = (r() - 0.5) * 12 - (i % P.NO >= 64 ? 4 : 0);
    const dw = g.decode(raw, size) as Float32Array, dt = P.decode(raw, size);
    check(maxDiff(dw, dt) < 2e-4, `decode ${size}: wasm vs TypeScript ${maxDiff(dw, dt)}`);
    for (const [conf, iou, agn] of [[0.25, 0.45, false], [0.1, 0.3, true], [0.5, 0.7, false], [0.02, 1, false]] as const) {
      const rw = g.nms(dt, conf, iou, 300, agn) as Float32Array, rt = P.nmsRows(dt, P.nms(dt, conf, iou, 300, agn));
      check(rw.length === rt.length && maxDiff(rw, rt) < 1e-6, `nms ${size} conf ${conf} iou ${iou} ${agn ? "agnostic" : ""}: ${rw.length / 7} vs ${rt.length / 7} rows`);
    }
  }

  current = "wasm network";
  const wFile = path.join(process.cwd(), "demos/dist/models/yolo/yolo11n-coco.safetensors");
  const iFile = path.join(process.cwd(), "demos/dist/models/images/stardist-img2d.png");
  if (!fs.existsSync(wFile) || !fs.existsSync(iFile)) { skip("model files (demos/dist/models/)"); return; }
  let t = Date.now();
  const report = g.load(new Uint8Array(fs.readFileSync(wFile))) as string;
  const loadMs = Date.now() - t;
  check(/^417 tensors loaded, 0 missing/.test(report), "weights load: " + report);
  const img = readGreyPng(new Uint8Array(fs.readFileSync(iFile)));
  check(img.w === 512 && img.h === 512, "sample is 512 x 512");
  const n = 512 * 512, input = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) input[p] = input[n + p] = input[2 * n + p] = img.data[p] / 255;
  t = Date.now();
  const raw = g.run(input, 512) as Float32Array;
  const runMs = Date.now() - t;
  check(raw.length === 5376 * P.NO, "raw output A x 144");
  const dec = P.decode(raw, 512);
  check(maxDiff(dec, g.decode(raw, 512)) < 2e-4, "decode of the network output: wasm vs TypeScript");
  const rows = P.nmsRows(dec, P.nms(dec, 0.01, 0.45));
  check(rows.length / 7 === NATIVE_NUCLEI_001.length, `score > 0.01: ${rows.length / 7} boxes, native crate ${NATIVE_NUCLEI_001.length}`);
  NATIVE_NUCLEI_001.forEach((w, k) => {
    const o = k * 7;
    check(maxDiff(rows.subarray(o, o + 4), w.slice(0, 4)) < 0.05 && close(rows[o + 4], w[4], 2e-4) && rows[o + 5] === w[5],
      `box ${k} vs native crate: ${Array.from(rows.subarray(o, o + 6)).map((v) => v.toFixed(3)).join(" ")} / ${w.join(" ")}`);
  });
  check(P.nms(dec, 0.25, 0.45).kept.length === 0, "score > 0.25: nothing (as the native crate)");
  console.log(`  YOLO11n in node (wasm): load ${loadMs} ms, run 512x512 ${runMs} ms`);
}

wasmTests().then(() => {
  console.log(`${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}, (e) => { console.log("ERROR", e); process.exit(1); });
