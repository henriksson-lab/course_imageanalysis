// Tests for the SAM demo (demos/src/samnet/): the TypeScript post-processing (post.ts) on hand
// cases, and, when the module and weights are built, the whole WebAssembly network
// (demos/wasm/sam/, SAM 2.1 Hiera tiny) on one sample image against the native sam2-rs crate.
// Run (from the repository root; the full check takes about a minute: the image encoder runs
// once in Wasm on one CPU core):
//   node_modules/.bin/esbuild demos/test/sam-wasm-test.ts --bundle --platform=node --log-level=warning | node
// Without demos/dist/wasm/sam/ (make wasm) or the weights (make models) the network checks are
// skipped with a message; WASM_REQUIRED=1 makes that a failure.
//
// The reference numbers (REF) were computed by a native program that runs the upstream
// sam2-rs Sam2ImagePredictor (predict_cached_prompts, three candidates, logits restored to the
// image size) and Sam2AutomaticMaskGenerator (8 x 8 grid, default thresholds: predicted IoU >
// 0.8, stability >= 0.95 at offset 1, box NMS 0.7) on images/stardist-img2d.png, and the wrapper
// crate natively (identical to upstream: max |logit difference| 0). Wasm computes in a different
// order (no SIMD, one thread), so values are compared with small tolerances.
import * as P from "../src/samnet/post";

declare const require: (m: string) => any;
declare const process: { exit(code: number): never; cwd(): string; env: Record<string, string | undefined> };
const fs = require("fs"), path = require("path"), url = require("url"), zlib = require("zlib");

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

// ---- post.ts on hand cases ------------------------------------------------------------------
current = "bilinear";
{
  // torch: interpolate([[0, 1]], size=4, bilinear, align_corners=False) -> 0, .25, .75, 1
  const r = P.upsampleBilinear([0, 1], 2, 1, 4, 1);
  check([0, 0.25, 0.75, 1].every((v, i) => close(r[i], v, 1e-6)), "2 -> 4 matches torch: " + Array.from(r).join(","));
  const src = Float32Array.from({ length: 30 }, (_, i) => Math.sin(i));
  const same = P.upsampleBilinear(src, 6, 5, 6, 5);
  check(same.every((v, i) => close(v, src[i], 1e-6)), "same size = identity");
  const c = P.upsampleBilinear(new Float32Array(16).fill(3.5), 4, 4, 9, 7);
  check(c.every((v) => close(v, 3.5, 1e-6)), "constant stays constant");
  // A linear ramp stays linear away from the clamped edges.
  const ramp = Float32Array.from({ length: 8 * 8 }, (_, i) => i % 8);
  const up = P.upsampleBilinear(ramp, 8, 8, 32, 32);
  check(close(up[5 * 32 + 10] - up[5 * 32 + 9], 0.25, 1e-5), "ramp slope 1/4 after 4x");
  // Downscaling (256 -> 100) samples at half-pixel centres.
  const d = P.upsampleBilinear(Float32Array.from({ length: 4 }, (_, i) => i), 4, 1, 2, 1);
  check(close(d[0], 0.5, 1e-6) && close(d[1], 2.5, 1e-6), "4 -> 2: " + Array.from(d).join(","));
}
current = "stability";
{
  const l = [-3, -0.5, 0.5, 1.5, 2, 0.9];
  check(close(P.stabilityScore(l, 0, 1), 2 / 5, 1e-12), "area(>1) / area(>-1) = 2/5");
  check(P.stabilityScore([-5, -5], 0, 1) === 0, "empty union -> 0");
  check(P.stabilityScore([9, 9, 9], 0, 1) === 1, "steep edges -> 1");
  check(Array.from(P.threshold([-1, 0, 0.01])).join("") === "001", "threshold is strict (> 0)");
}
current = "boxes";
{
  const m = new Uint8Array(6 * 5); m[1 * 6 + 2] = m[3 * 6 + 4] = m[2 * 6 + 3] = 1;
  const b = P.maskBox(m, 6, 5);
  check(!!b && b.box.join(",") === "2,1,4,3" && b.area === 3, "tight inclusive box: " + JSON.stringify(b));
  check(P.maskBox(new Uint8Array(4), 2, 2) === null, "empty mask -> null");
  check(close(P.boxIoU([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7, 1e-12), "box IoU 1/7");
  check(P.boxIoU([0, 0, 1, 1], [2, 2, 3, 3]) === 0, "disjoint boxes");
  check(close(P.maskIoU([1, 1, 0, 0], [0, 1, 1, 0]), 1 / 3, 1e-12), "mask IoU");
  const g = P.pointGrid(2);
  check(g.length === 4 && g[0][0] === 0.25 && g[3][1] === 0.75 && g[1][0] === 0.75, "2 x 2 grid at cell centres");
  check(P.pointGrid(1)[0].join(",") === "0.5,0.5", "1 x 1 grid in the centre");
}
current = "filters";
{
  const mk = (iou: number, stability: number, box: P.Box | null, point = 0): P.Candidate =>
    ({ point, output: 0, px: 0, py: 0, iou, stability, box, area: box ? 1 : 0, mask: box ? new Uint8Array(1) : null });
  const cands = [
    mk(0.95, 0.97, [0, 0, 10, 10], 0),   // kept (best score)
    mk(0.90, 0.99, [1, 1, 10, 10], 1),   // overlaps the first (box IoU 0.81) -> nms
    mk(0.85, 0.96, [20, 20, 30, 30], 2), // kept
    mk(0.70, 0.99, [40, 40, 50, 50], 3), // iou
    mk(0.92, 0.90, [60, 60, 70, 70], 4), // stability
    mk(0.92, NaN, null, 5),              // never scored -> stability
    mk(0.99, 0.99, null, 6),             // empty mask
  ];
  const f = P.filterCandidates(cands, { ...P.AMG_DEFAULT });
  check(f.fate.join(",") === "kept,nms,kept,iou,stability,stability,empty", "fates: " + f.fate.join(","));
  check(f.kept.map((c) => c.point).join(",") === "0,2", "kept in score order");
  check(f.counts.kept === 2 && f.counts.nms === 1 && f.counts.iou === 1 && f.counts.stability === 2 && f.counts.empty === 1, "counts");
  const loose = P.filterCandidates(cands, { ...P.AMG_DEFAULT, nms: 0.9, predIou: 0.6, stability: 0.8 });
  check(loose.counts.kept === 5, "loose filters keep the duplicate too: " + loose.counts.kept);
  const lab = P.labelImage([{ ...mk(1, 1, [0, 0, 1, 1]), mask: new Uint8Array([1, 1, 1, 0]), area: 3 },
    { ...mk(1, 1, [0, 0, 0, 0]), mask: new Uint8Array([1, 0, 0, 0]), area: 1 }], 2, 2);
  check(Array.from(lab).join(",") === "2,1,1,0", "small mask painted on top: " + Array.from(lab).join(","));
}
current = "batchCandidates";
{
  // One point, two outputs: a disc (steep logits) and a flat, ambiguous plane.
  const logits = new Float32Array(2 * P.LOW * P.LOW);
  for (let y = 0; y < P.LOW; y++) for (let x = 0; x < P.LOW; x++) {
    logits[y * P.LOW + x] = 5 * (40 - Math.hypot(x - 128, y - 128)); // 5 per pixel: steep edge
    logits[P.LOW * P.LOW + y * P.LOW + x] = 0.5;
  }
  const c = P.batchCandidates(logits, new Float32Array([0.9, 0.95]), 1, 2, [[64, 64]], 0, 128, 128, 1);
  check(c.length === 2 && c[0].mask !== null && c[0].stability > 0.95, "disc: stable, mask kept: " + c[0].stability);
  check(c[0].box !== null && close(c[0].box[0], 44, 1) && close(c[0].box[2], 83, 1) && close(c[0].area, Math.PI * 20 * 20, 40),
    "disc restored to the image size: " + JSON.stringify(c[0].box) + " area " + c[0].area);
  check(c[1].stability === 0 && c[1].mask === null, "flat plane: stability 0, no mask stored");
  const low = P.batchCandidates(logits, new Float32Array([0.4, 0.4]), 1, 2, [[64, 64]], 0, 128, 128, 1);
  check(low.every((x) => Number.isNaN(x.stability) && x.mask === null), "below keepIou: not even restored");
}
current = "embeddingRGB";
{
  // 4 channels on 8 x 8: the first principal direction is channel 0 (largest variance, a ramp).
  const C = 4, n = 8, N = n * n, e = new Float32Array(C * N);
  for (let p = 0; p < N; p++) { e[p] = 10 * (p % n); e[N + p] = Math.sin(p) * 0.1; e[2 * N + p] = (p >> 3) * 0.5; e[3 * N + p] = 1; }
  const rgb = P.embeddingRGB(e, C, n);
  const r = (x: number, y: number) => rgb[4 * (y * n + x)];
  const inc = r(7, 3) > r(0, 3) ? 1 : -1;
  let mono = true;
  for (let x = 1; x < n; x++) if ((r(x, 3) - r(x - 1, 3)) * inc < 0) mono = false;
  check(mono && Math.abs(r(7, 3) - r(0, 3)) > 200, "red follows the dominant ramp");
  check(rgb[3] === 255 && rgb.length === 4 * N, "opaque RGBA");
}

// ---- The network in Wasm against the native crate -------------------------------------------
const REF = {"image":"images/stardist-img2d.png","width":512,"height":512,"probes":[[340,317],[330,300],[360,330],[236,266],[210,250],[149,428],[100,400],[0,0],[511,511],[300,100]],"prompts":[{"name":"point","points":[[340.0,317.0,1]],"bbox":null,"iou":[0.00261,0.86698,0.7635],"object_score":[14.22514],"area":[258586,819,930],"logits_at":[[0.73845,0.67143,0.77481,0.95906,1.01994,0.6385,0.87079,-0.62608,-0.31213,0.52755],[6.01758,-27.98317,-22.27737,-26.67727,-34.17092,-28.15755,-34.11598,-29.13409,-24.81827,-31.27405],[6.53487,-14.05058,-10.07303,-10.80658,-17.80916,-10.20846,-13.37331,-16.85716,-12.11691,-17.82949]]},{"name":"box","points":[],"bbox":[204.0,243.0,267.0,290.0],"iou":[0.48376,0.88647,0.91394],"object_score":[20.89972],"area":[2182,1343,1472],"logits_at":[[-11.41142,-11.54322,-13.31828,7.52247,-4.71357,-11.41517,-12.44962,-12.04504,-10.04674,-12.40969],[-22.30645,-23.6737,-25.67582,5.81553,-18.86602,-23.13751,-23.98424,-21.33747,-19.63645,-22.81927],[-19.45757,-20.27463,-22.24534,9.61582,-16.25404,-19.67714,-20.69698,-19.92097,-17.03847,-21.34144]]},{"name":"pos+neg","points":[[149.0,428.0,1],[120.0,410.0,0]],"bbox":null,"iou":[0.01374,0.53372,0.44702],"object_score":[12.92958],"area":[242969,483,1407],"logits_at":[[0.25413,0.48945,0.68544,0.3953,0.58651,1.30919,1.11098,-0.79263,-0.35568,0.67353],[-24.2006,-27.43562,-26.56049,-23.03748,-29.92136,6.76562,-20.1705,-24.47834,-18.21041,-26.25361],[-9.41429,-10.7873,-10.00763,-9.01334,-12.472,7.27391,-6.49319,-12.90543,-7.83731,-10.9009]]}],"lowResPointProbe":{"at":[[170,158],[160,150],[0,0],[128,128],[200,60]],"values":[[0.67175,1.65768,-0.62608,0.81224,0.30504],[7.21568,-34.10472,-29.13409,-21.63077,-31.44548],[6.91213,-14.7837,-16.85716,-8.83487,-18.48267]]},"amgPointsPerSide":8,"amg":[{"bboxXYWH":[0.0,0.0,511.0,511.0],"area":259239,"iou":0.99076,"stability":0.95936},{"bboxXYWH":[200.0,341.0,27.0,26.0],"area":607,"iou":0.93029,"stability":0.97869},{"bboxXYWH":[149.0,324.0,35.0,28.0],"area":760,"iou":0.91442,"stability":0.9549},{"bboxXYWH":[16.0,479.0,29.0,32.0],"area":874,"iou":0.89542,"stability":0.95501},{"bboxXYWH":[92.0,408.0,30.0,22.0],"area":556,"iou":0.88448,"stability":0.96127}]};

/** Minimal PNG decoder for 8-bit grey or RGB(A) non-interlaced images (the sample images). */
function decodePNG(buf: Uint8Array): { w: number; h: number; rgb: Uint8Array } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8, w = 0, h = 0, ct = 0;
  const idat: Uint8Array[] = [];
  while (o < buf.length) {
    const len = dv.getUint32(o), type = String.fromCharCode(...buf.subarray(o + 4, o + 8));
    const data = buf.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      w = dv.getUint32(o + 8); h = dv.getUint32(o + 12); ct = data[9];
      if (data[8] !== 8 || data[12] !== 0) throw new Error("only 8-bit non-interlaced PNG");
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    o += 12 + len;
  }
  const ch = ({ 0: 1, 2: 3, 6: 4 } as Record<number, number>)[ct];
  if (!ch) throw new Error("PNG colour type " + ct);
  const raw: Uint8Array = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * ch, px = new Uint8Array(h * stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= ch ? px[y * stride + x - ch] : 0, b = y ? px[(y - 1) * stride + x] : 0;
      const c = x >= ch && y ? px[(y - 1) * stride + x - ch] : 0;
      let v = line[x];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      px[y * stride + x] = v & 255;
    }
  }
  const rgb = new Uint8Array(w * h * 3);
  for (let p = 0; p < w * h; p++) for (let k = 0; k < 3; k++) rgb[3 * p + k] = px[p * ch + (ch === 1 ? 0 : k)];
  return { w, h, rgb };
}
declare const Buffer: { concat(list: Uint8Array[]): Uint8Array };

async function networkTests(): Promise<void> {
  current = "wasm";
  const root = process.cwd();
  const glueFile = path.join(root, "demos/dist/wasm/sam/sam.js"), wasmFile = path.join(root, "demos/dist/wasm/sam/sam_bg.wasm");
  const weightsFile = path.join(root, "demos/dist/models/sam2/sam2.1_hiera_tiny.safetensors");
  const imageFile = path.join(root, "demos/dist/models", REF.image);
  for (const [f, how] of [[glueFile, "make wasm"], [wasmFile, "make wasm"], [weightsFile, "make models"], [imageFile, "make models"]]) {
    if (!fs.existsSync(f)) {
      if (process.env.WASM_REQUIRED) { check(false, `${f} missing (run ${how})`); return; }
      console.log(`SKIP network checks: ${path.relative(root, f)} missing (run ${how})`);
      return;
    }
  }
  const g: any = await import(url.pathToFileURL(glueFile).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(wasmFile)) });
  let t = Date.now();
  g.load(new Uint8Array(fs.readFileSync(weightsFile)));
  const loadMs = Date.now() - t;
  const img = decodePNG(new Uint8Array(fs.readFileSync(imageFile)));
  check(img.w === REF.width && img.h === REF.height, `image ${img.w}x${img.h}`);
  let threw = "";
  try { g.decode(new Float32Array([1, 1]), new Int32Array([1]), new Float32Array(0), new Float32Array(0), true); } catch (e) { threw = String((e as Error).message ?? e); }
  check(threw.includes("encode an image first"), "decode before encode is a readable error: " + threw);
  t = Date.now();
  g.encode(img.rgb, img.w, img.h);
  const encodeMs = Date.now() - t;
  const emb: Float32Array = g.embedding();
  check(emb.length === 256 * 64 * 64 && emb.every(Number.isFinite), "embedding 256 x 64 x 64, finite");

  // Prompts: three candidates each, compared with the native predictor.
  let decodeMs = 0;
  for (const p of REF.prompts) {
    current = "wasm/" + p.name;
    t = Date.now();
    const logits: Float32Array = g.decode(new Float32Array(p.points.flatMap((q) => [q[0], q[1]])), new Int32Array(p.points.map((q) => q[2])),
      new Float32Array(p.bbox ?? []), new Float32Array(0), true);
    decodeMs += Date.now() - t;
    const iou: Float32Array = g.last_iou(), obj: Float32Array = g.last_object_score(), shape: Uint32Array = g.last_shape();
    check(shape[0] === 1 && shape[1] === 3 && logits.length === 3 * 256 * 256, "shape " + Array.from(shape).join(","));
    check(shape[2] === p.points.length + (p.bbox ? 2 : 0) + 1, `prompt tokens ${shape[2]} (points, box corners, one padding token)`);
    for (let k = 0; k < 3; k++) {
      check(close(iou[k], p.iou[k], 0.01), `IoU score ${k}: ${iou[k].toFixed(4)} vs native ${p.iou[k]}`);
      const full = P.restore(P.plane(logits, k), img.w, img.h);
      let maxd = 0;
      REF.probes.forEach(([x, y], j) => { maxd = Math.max(maxd, Math.abs(full[y * img.w + x] - p.logits_at[k][j])); });
      check(maxd < 0.05, `restored logits at probe pixels, mask ${k}: max |diff| ${maxd.toFixed(4)}`);
      let area = 0; for (let i = 0; i < full.length; i++) if (full[i] > 0) area++;
      check(Math.abs(area - p.area[k]) <= Math.max(3, 0.005 * p.area[k]), `mask ${k} area ${area} vs native ${p.area[k]}`);
    }
    check(close(obj[0], p.object_score[0], 0.05), `object score ${obj[0]} vs ${p.object_score[0]}`);
    if (p.name === "point") {
      let maxd = 0;
      for (let k = 0; k < 3; k++) REF.lowResPointProbe.at.forEach(([x, y], j) => {
        maxd = Math.max(maxd, Math.abs(logits[k * 65536 + y * 256 + x] - REF.lowResPointProbe.values[k][j]));
      });
      check(maxd < 0.05, "low-resolution logits at probe positions: max |diff| " + maxd.toFixed(4));
    }
  }
  current = "wasm/single";
  {
    const l1: Float32Array = g.decode(new Float32Array([340, 317]), new Int32Array([1]), new Float32Array(0), new Float32Array(0), false);
    check(l1.length === 256 * 256 && g.last_shape()[1] === 1, "multimask off: one mask");
    const l2: Float32Array = g.decode(new Float32Array([340, 317]), new Int32Array([1]), new Float32Array(0), l1, false);
    check(l2.length === 256 * 256 && l2.every(Number.isFinite), "previous mask as an extra prompt");
    let threw2 = "";
    try { g.decode(new Float32Array([1]), new Int32Array([1]), new Float32Array(0), new Float32Array(0), true); } catch (e) { threw2 = String((e as Error).message ?? e); }
    check(threw2.includes("x,y"), "malformed points rejected: " + threw2);
  }

  // Automatic masks: grid -> decode in batches -> post.ts filters, vs the native generator.
  current = "wasm/automatic";
  const n = REF.amgPointsPerSide, grid = P.pointGrid(n).map(([x, y]) => [x * img.w, y * img.h] as [number, number]);
  const cands: P.Candidate[] = [];
  t = Date.now();
  for (let s = 0; s < grid.length; s += 8) {
    const pts = grid.slice(s, s + 8);
    const logits: Float32Array = g.decode_points(new Float32Array(pts.flat()));
    const [count, masks] = g.last_shape();
    check(count === pts.length && masks === 3, "batch shape");
    cands.push(...P.batchCandidates(logits, g.last_iou(), count, masks, pts, s, img.w, img.h, 1));
  }
  const amgMs = Date.now() - t;
  const { kept } = P.filterCandidates(cands, { ...P.AMG_DEFAULT, pointsPerSide: n });
  check(Math.abs(kept.length - REF.amg.length) <= 1, `${kept.length} masks vs native ${REF.amg.length}`);
  let matched = 0;
  for (const r of REF.amg) {
    const rb: P.Box = [r.bboxXYWH[0], r.bboxXYWH[1], r.bboxXYWH[0] + r.bboxXYWH[2], r.bboxXYWH[1] + r.bboxXYWH[3]];
    const m = kept.find((c) => P.boxIoU(c.box!, rb) > 0.9);
    if (!m) { check(false, "native mask not found: " + JSON.stringify(r)); continue; }
    matched++;
    check(Math.abs(m.area - r.area) <= Math.max(3, 0.01 * r.area), `area ${m.area} vs ${r.area}`);
    check(close(m.iou, r.iou, 0.01) && close(m.stability, r.stability, 0.01), `scores ${m.iou.toFixed(3)}/${m.stability.toFixed(3)} vs ${r.iou}/${r.stability}`);
  }
  console.log(`  wasm: load ${loadMs} ms, encode ${encodeMs} ms, decode ${(decodeMs / REF.prompts.length).toFixed(0)} ms per prompt, ` +
    `automatic ${n}x${n} ${amgMs} ms; ${matched}/${REF.amg.length} native masks matched`);
}

networkTests().then(() => {
  console.log(`sam-wasm-test: ${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}).catch((e) => { console.log("FAIL: " + (e as Error).stack); process.exit(1); });
