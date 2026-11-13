// Dumps the real network's outputs (SAM 2.1 Hiera tiny in WebAssembly, as the samnet demo runs
// it) for the slide figures of sam/: embedding PCA, the three candidates of several presets,
// and automatic masks with the fate of every grid point. lec_figs.py draws them.
//   python3 -I tools/figs/sam/to_raw.py <raw>
//   node_modules/.bin/esbuild tools/figs/sam/dump_samnet.ts --bundle --platform=node --log-level=warning > d.cjs && node d.cjs <raw> <out>
import * as P from "../../../demos/src/samnet/post";

declare const require: (m: string) => any;
declare const process: { argv: string[]; cwd(): string; exit(c: number): never };
const fs = require("fs"), path = require("path"), url = require("url");
const [RAW, OUT] = [process.argv[2], process.argv[3]];
const root = process.cwd();
fs.mkdirSync(OUT, { recursive: true });

function readImage(id: string) {
  const meta = JSON.parse(fs.readFileSync(path.join(RAW, id + ".json"), "utf8"));
  return { w: meta.w as number, h: meta.h as number, rgb: new Uint8Array(fs.readFileSync(path.join(RAW, id + ".rgb"))) };
}
const save = (name: string, a: ArrayLike<number> & { buffer: ArrayBuffer; byteOffset: number; byteLength: number }) =>
  fs.writeFileSync(path.join(OUT, name), new Uint8Array(a.buffer, a.byteOffset, a.byteLength));

async function main() {
  const g: any = await import(url.pathToFileURL(path.join(root, "demos/dist/wasm/sam/sam.js")).href);
  await g.default({ module_or_path: new Uint8Array(fs.readFileSync(path.join(root, "demos/dist/wasm/sam/sam_bg.wasm"))) });
  g.load(new Uint8Array(fs.readFileSync(path.join(root, "demos/dist/models/sam2/sam2.1_hiera_tiny.safetensors"))));
  const meta: any = {};
  for (const id of ["nuclei", "brightfield"]) {
    const im = readImage(id);
    g.encode(im.rgb, im.w, im.h);
    save(`${id}-emb.rgba`, P.embeddingRGB(g.embedding(), 256, 64));
    meta[id] = { w: im.w, h: im.h, prompts: {} };
    const prompts: Record<string, { points: [number, number, number][]; box: number[] | null }> = id === "nuclei" ? {
      click: { points: [[340, 317, 1]], box: null },
      cluster: { points: [[149, 428, 1]], box: null },
      negative: { points: [[149, 428, 1], [120, 410, 0]], box: null },
      // A negative click on the touching neighbour (not a preset of the demo).
      negnb: { points: [[149, 428, 1], [168, 448, 0]], box: null },
      box: { points: [], box: [204, 243, 267, 290] },
    } : {};
    for (const [name, pr] of Object.entries(prompts)) {
      const logits: Float32Array = g.decode(new Float32Array(pr.points.flatMap((p) => [p[0], p[1]])), new Int32Array(pr.points.map((p) => p[2])),
        new Float32Array(pr.box ?? []), new Float32Array(0), true);
      const iou = Array.from(g.last_iou() as Float32Array);
      const masks = new Uint8Array(3 * im.w * im.h);
      for (let k = 0; k < 3; k++) masks.set(P.threshold(P.restore(P.plane(logits, k), im.w, im.h), 0), k * im.w * im.h);
      save(`${id}-${name}.masks`, masks);
      meta[id].prompts[name] = { ...pr, iou };
    }
    // Automatic masks, default filters, with the fate of every grid point.
    const n = 8, grid = P.pointGrid(n).map(([x, y]) => [x * im.w, y * im.h] as [number, number]);
    const cands: P.Candidate[] = [];
    for (let s = 0; s < grid.length; s += 4) {
      const pts = grid.slice(s, s + 4);
      const lg: Float32Array = g.decode_points(new Float32Array(pts.flat()));
      const [count, masks] = g.last_shape();
      cands.push(...P.batchCandidates(lg, g.last_iou(), count, masks, pts, s, im.w, im.h, 1));
    }
    const f = P.filterCandidates(cands, { ...P.AMG_DEFAULT });
    const rank: Record<string, number> = { kept: 4, nms: 3, stability: 2, empty: 1, iou: 0 };
    const fates: string[] = new Array(grid.length).fill("");
    cands.forEach((c, i) => { const fa = f.fate[i]; if (!fates[c.point] || rank[fa] > rank[fates[c.point]]) fates[c.point] = fa; });
    save(`${id}-auto.lab`, P.labelImage(f.kept, im.w, im.h));
    meta[id].auto = { grid, fates, counts: f.counts, kept: f.kept.map((k) => ({ area: k.area, iou: k.iou, stability: k.stability })) };
    console.log(id, "done", JSON.stringify(f.counts));
  }
  fs.writeFileSync(path.join(OUT, "meta.json"), JSON.stringify(meta));
}
main().catch((e) => { console.log("FAIL " + (e as Error).stack); process.exit(1); });
