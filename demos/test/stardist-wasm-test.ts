// Tests for the StarDist Wasm module (demos/wasm/stardist/, built by `make wasm` into
// demos/dist/wasm/stardist/), run in node:
//   node_modules/.bin/esbuild demos/test/stardist-wasm-test.ts --bundle --platform=node --log-level=warning | node
// 1. Decoding alone (`instances`) on synthetic network outputs with known answers: discs with
//    exact ray distances -> one object per disc, centre on the grid, label area = pi r^2,
//    polygon area (shoelace over the returned vertices) = label area, NMS merges two
//    overlapping candidates and keeps them apart when the NMS threshold is raised.
// 2. When `make models` has run: the network in Wasm against the native (x86-64) output of the
//    same crate on a 128 x 128 crop of the StarDist sample image (testdata/native-ref.json,
//    written by `cargo test` in demos/wasm/stardist/; independently checked against a NumPy
//    forward pass), and a timing of the full 512 x 512 image.
// Skips with a message when the module or the model is missing (WASM_REQUIRED=1 makes a
// missing module a failure).
declare const require: (m: string) => any;
declare const process: { exit(code: number): never; cwd(): string; env: Record<string, string | undefined> };
const fs = require("fs"), path = require("path"), url = require("url"), zlib = require("zlib");

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log(`FAIL [${current}]: ${msg}`); }
}

const N_RAYS = 32, GRID = 2;

/** Network outputs for discs (cy, cx, r) on a grid-2 output: prob is a smooth bump per disc,
 *  dist the exact distance along each ray from the cell's pixel position to the disc edge. */
function synth(w: number, h: number, discs: [number, number, number][]): { prob: Float32Array; dist: Float32Array; ow: number; oh: number } {
  const ow = w / GRID, oh = h / GRID;
  const prob = new Float32Array(ow * oh), dist = new Float32Array(ow * oh * N_RAYS).fill(1e-3);
  for (let gy = 0; gy < oh; gy++) for (let gx = 0; gx < ow; gx++) {
    const y = gy * GRID, x = gx * GRID, i = gy * ow + gx;
    let best = 0, bi = -1;
    discs.forEach(([cy, cx, r], k) => {
      const d2 = (y - cy) ** 2 + (x - cx) ** 2;
      const p = d2 < r * r ? 0.95 * Math.exp(-d2 / (0.5 * r * r)) : 0;
      if (p > best) { best = p; bi = k; }
    });
    prob[i] = best;
    if (bi < 0) continue;
    const [cy, cx, r] = discs[bi];
    for (let k = 0; k < N_RAYS; k++) {
      const phi = (2 * Math.PI * k) / N_RAYS, dy = Math.sin(phi), dx = Math.cos(phi);
      // |p + t u - c| = r, t > 0
      const py = y - cy, px = x - cx, b = py * dy + px * dx, c = py * py + px * px - r * r;
      dist[i * N_RAYS + k] = Math.max(1e-3, -b + Math.sqrt(Math.max(0, b * b - c)));
    }
  }
  return { prob, dist, ow, oh };
}

function shoelace(ys: ArrayLike<number>, xs: ArrayLike<number>): number {
  let a = 0;
  for (let k = 0; k < ys.length; k++) { const j = (k + 1) % ys.length; a += xs[k] * ys[j] - xs[j] * ys[k]; }
  return Math.abs(a) / 2;
}

/** The current decoding result (the module keeps the last one) as an object. */
function snap(g: any) {
  const v = { candidates: g.dec_candidates(), count: g.dec_count(), points: g.dec_points(), probs: g.dec_probs(),
    dists: g.dec_dists(), coords: g.dec_coords(), labels: g.dec_labels() };
  return { candidates: () => v.candidates, count: () => v.count, points: () => v.points, probs: () => v.probs,
    dists: () => v.dists, coords: () => v.coords, labels: () => v.labels };
}

function decodeTests(g: any): void {
  current = "decode/discs";
  {
    const w = 96, h = 64, discs: [number, number, number][] = [[20, 20, 9], [40, 64, 12], [48, 24, 6]];
    const s = synth(w, h, discs);
    g.instances(s.prob, s.dist, s.ow, s.oh, w, h, N_RAYS, GRID, 0.5, 0.4);
    const inst = snap(g);
    const ref = s.prob.reduce((n: number, p: number) => n + (p > 0.5 ? 1 : 0), 0);
    check(inst.candidates() === ref, `candidates ${inst.candidates()} = cells above 0.5 (${ref})`);
    check(inst.count() === 3, `three objects, got ${inst.count()}`);
    const pts = inst.points(), probs = inst.probs(), coords = inst.coords(), labels = inst.labels();
    check(labels.length === w * h && coords.length === 3 * 2 * N_RAYS && inst.dists().length === 3 * N_RAYS, "array sizes");
    for (let i = 1; i < probs.length; i++) check(probs[i] <= probs[i - 1], "sorted by probability");
    for (let o = 0; o < inst.count(); o++) {
      const y = pts[2 * o], x = pts[2 * o + 1];
      const k = discs.findIndex(([cy, cx]) => Math.hypot(cy - y, cx - x) <= 1.5);
      check(k >= 0, `object ${o} at (${y}, ${x}) is a disc centre (within one grid cell)`);
      if (k < 0) continue;
      const r = discs[k][2];
      let area = 0;
      for (const l of labels) if (l === o + 1) area++;
      check(Math.abs(area - Math.PI * r * r) / (Math.PI * r * r) < 0.12, `label area ${area} vs pi r^2 = ${(Math.PI * r * r).toFixed(0)}`);
      const ys = coords.subarray(o * 2 * N_RAYS, o * 2 * N_RAYS + N_RAYS), xs = coords.subarray(o * 2 * N_RAYS + N_RAYS, (o + 1) * 2 * N_RAYS);
      const poly = shoelace(ys, xs);
      check(Math.abs(poly - area) / area < 0.08, `polygon area ${poly.toFixed(1)} vs label pixels ${area}`);
      check(labels[Math.round(y) * w + Math.round(x)] === o + 1, "centre pixel carries the object's label");
    }
    let bg = 0;
    for (const l of labels) if (l === 0) bg++;
    check(bg > 0.6 * w * h, "background stays 0");
    g.instances(s.prob, s.dist, s.ow, s.oh, w, h, N_RAYS, GRID, 0.999, 0.4);
    const none = snap(g);
    check(none.count() === 0 && none.labels().every((l: number) => l === 0), "threshold above every probability: no objects");
  }
  current = "decode/nms";
  {
    // Two discs of radius 12 whose centres are 6 px apart: IoU about 0.68.
    const w = 64, h = 48, discs: [number, number, number][] = [[24, 26, 12], [24, 32, 12]];
    const s = synth(w, h, discs);
    g.instances(s.prob, s.dist, s.ow, s.oh, w, h, N_RAYS, GRID, 0.5, 0.3);
    const a = snap(g);
    check(a.count() === 1, `overlapping discs, NMS 0.3: one object, got ${a.count()}`);
    g.instances(s.prob, s.dist, s.ow, s.oh, w, h, N_RAYS, GRID, 0.5, 0.9);
    const b = snap(g);
    check(b.count() >= 2, `NMS 0.9: both kept, got ${b.count()}`);
    check(b.candidates() === a.candidates(), "candidates do not depend on the NMS threshold");
  }
  current = "normalize";
  {
    const v = new Float32Array(1001).map((_, i) => i);
    const n = g.normalize_percentile(v, 1, 99.8);
    check(Math.abs(n[10] - 0) < 1e-6 && Math.abs(n[998] - 1) < 1e-6, `percentiles 1 and 99.8 -> 0 and 1 (${n[10]}, ${n[998]})`);
  }
}

async function networkTests(g: any): Promise<void> {
  current = "network";
  const mdir = path.join(process.cwd(), "demos/dist/models/stardist/2D_versatile_fluo");
  const refFile = path.join(process.cwd(), "demos/wasm/stardist/testdata/native-ref.json");
  if (!fs.existsSync(path.join(mdir, "weights_best.h5"))) {
    console.log("SKIP network check: demos/dist/models/stardist/ missing (run make models)");
    return;
  }
  const ref = JSON.parse(fs.readFileSync(refFile, "utf8"));
  const t0 = Date.now();
  g.load(fs.readFileSync(path.join(mdir, "config.json"), "utf8"),
    fs.readFileSync(path.join(mdir, "thresholds.json"), "utf8"), new Uint8Array(fs.readFileSync(path.join(mdir, "weights_best.h5"))));
  const tLoad = Date.now() - t0;
  check(g.n_rays() === 32 && g.grid() === 2, "config: 32 rays, grid 2");
  check(Math.abs(g.default_prob_thresh() - ref.prob_thresh) < 1e-6 && Math.abs(g.default_nms_thresh() - ref.nms_thresh) < 1e-6, "thresholds.json");
  const n = ref.crop.size;
  const pixels: Uint8Array = Buffer.from(ref.pixels_base64, "base64");
  check(pixels.length === n * n, "reference crop pixels");
  const img = g.normalize_percentile(Float32Array.from(pixels), 1, 99.8);
  for (const engine of [1, 0]) {
  current = engine === 0 ? "network/candle" : "network/fast";
  g.predict(img, n, n, engine);
  check(g.out_width() === ref.out_w && g.out_height() === ref.out_h, `output ${g.out_width()}x${g.out_height()}`);
  const prob = g.prob(), dist = g.dist();
  let dp = 0;
  for (let i = 0; i < prob.length; i++) dp = Math.max(dp, Math.abs(prob[i] - ref.prob[i]));
  check(dp < 1e-4, `probability map vs native: max |diff| ${dp.toExponential(1)}`);
  let dd = 0;
  for (const s of ref.dist_sample) for (let k = 0; k < 32; k++) dd = Math.max(dd, Math.abs(dist[s.cell * 32 + k] - s.dist[k]) / Math.max(1, s.dist[k]));
  check(dd < 1e-4, `ray distances vs native: max relative diff ${dd.toExponential(1)}`);
  g.instances(prob, dist, g.out_width(), g.out_height(), n, n, 32, 2, ref.prob_thresh, ref.nms_thresh);
  const inst = snap(g);
  check(inst.candidates() === ref.candidates, `candidates ${inst.candidates()} vs native ${ref.candidates}`);
  check(inst.count() === ref.count, `objects ${inst.count()} vs native ${ref.count}`);
  const pts = inst.points();
  check(pts.length === ref.points.length && pts.every((v: number, i: number) => v === ref.points[i]), "object centres equal native");
  const areas = new Array(inst.count()).fill(0);
  for (const l of inst.labels()) if (l > 0) areas[l - 1]++;
  check(areas.every((a, i) => a === ref.areas[i]), `label areas equal native (${areas.slice(0, 5).join(",")}… vs ${ref.areas.slice(0, 5).join(",")}…)`);
  }

  // Timing on the full sample image (512 x 512), when its PNG is there.
  const png = path.join(process.cwd(), "demos/dist/models/images/stardist-img2d.png");
  if (fs.existsSync(png)) {
    const { data, w, h } = readGreyPng(fs.readFileSync(png));
    const x = g.normalize_percentile(data, 1, 99.8);
    current = "full image";
    const t0c = Date.now();
    g.predict(x, w, h, 0);
    const tCandle = Date.now() - t0c;
    const t1 = Date.now();
    g.predict(x, w, h, 1);
    const tNet = Date.now() - t1, t2 = Date.now();
    g.instances(g.prob(), g.dist(), g.out_width(), g.out_height(), w, h, 32, 2, ref.prob_thresh, ref.nms_thresh);
    const full = snap(g);
    const tPost = Date.now() - t2;
    console.log(`  node: load ${tLoad} ms; ${w}x${h}: network ${tNet} ms (candle engine ${tCandle} ms), decoding ${tPost} ms, ${full.count()} objects from ${full.candidates()} candidates`);
    check(full.count() > 50, "full image: many nuclei");
  }
}

/** 8-bit greyscale, non-interlaced PNG (enough for the converted sample images). */
function readGreyPng(buf: Uint8Array): { data: Float32Array; w: number; h: number } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 8, w = 0, h = 0, ct = 0;
  const idat: Uint8Array[] = [];
  while (p < buf.length) {
    const len = dv.getUint32(p), type = String.fromCharCode(...buf.subarray(p + 4, p + 8));
    const d = buf.subarray(p + 8, p + 8 + len);
    if (type === "IHDR") { w = dv.getUint32(p + 8); h = dv.getUint32(p + 12); ct = d[9]; if (d[8] !== 8 || ct !== 0) throw new Error("not 8-bit grey"); }
    if (type === "IDAT") idat.push(d);
    p += 12 + len;
  }
  const raw: Uint8Array = zlib.inflateSync(Buffer.concat(idat));
  const out = new Uint8Array(w * h), stride = w;
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < w; x++) {
      const a = x > 0 ? out[y * w + x - 1] : 0, b = y > 0 ? out[(y - 1) * w + x] : 0, c = x > 0 && y > 0 ? out[(y - 1) * w + x - 1] : 0;
      let v = line[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[y * w + x] = v & 255;
    }
  }
  return { data: Float32Array.from(out), w, h };
}
declare const Buffer: { concat(a: Uint8Array[]): Uint8Array; from(s: string, enc: string): Uint8Array };

async function main(): Promise<void> {
  const dir = path.join(process.cwd(), "demos/dist/wasm/stardist");
  const glueFile = path.join(dir, "stardist.js"), wasmFile = path.join(dir, "stardist_bg.wasm");
  if (!fs.existsSync(glueFile) || !fs.existsSync(wasmFile)) {
    if (process.env.WASM_REQUIRED) { check(false, "demos/dist/wasm/stardist not built (run make wasm)"); return; }
    console.log("SKIP stardist: demos/dist/wasm/stardist/ not built (run make wasm)");
    return;
  }
  const g: any = await import(url.pathToFileURL(glueFile).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(wasmFile)) });
  decodeTests(g);
  await networkTests(g);
}

main().then(() => {
  console.log(`stardist-wasm: ${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}).catch((e) => { console.log("FAIL:", e); process.exit(1); });
