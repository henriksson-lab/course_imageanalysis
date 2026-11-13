// Tests for the Wasm pipeline: the example crate demos/wasm/hello/ (built by `make wasm` into
// demos/dist/wasm/hello/) against TypeScript references, and the loader lib/wasm.ts (URLs,
// progress, Cache API with versions, inline and fetched instantiation, model files) with a
// fake fetch and a fake Cache API. Run:
//   node_modules/.bin/esbuild demos/test/wasm-test.ts --bundle --platform=node --log-level=warning | node
// Without a built demos/dist/wasm/hello/ the module checks are skipped with a message (set
// WASM_REQUIRED=1, as CI does, to make that a failure).
import * as W from "../src/lib/wasm";

declare const require: (m: string) => any;
declare const process: { exit(code: number): never; cwd(): string; env: Record<string, string | undefined> };
const fs = require("fs"), path = require("path"), url = require("url");

let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 30) console.log(`FAIL [${current}]: ${msg}`); }
}

// ---- Pure helpers -----------------------------------------------------------------------------
current = "urls";
{
  const src = "https://example.org/course/demos/dist/stardist.js";
  check(W.distBase(src) === "https://example.org/course/demos/dist/", "distBase from a demo script");
  check(W.distBase("https://example.org/course/demos/dist/wasm/hello/example/main.js") ===
    "https://example.org/course/demos/dist/", "distBase from a nested script");
  check(W.distBase("http://localhost:9601/other/x.js") === "http://localhost:9601/other/", "distBase fallback: script folder");
  const base = "https://example.org/course/demos/dist/";
  check(W.wasmUrl("hello", base) === base + "wasm/hello/hello_bg.wasm", "wasmUrl");
  check(W.modelUrl("stardist/x.h5", base) === base + "models/stardist/x.h5", "modelUrl");
  check(W.isFileProtocol("file:///home/a/docs/x.html") && !W.isFileProtocol("https://a/b"), "isFileProtocol");
  check(W.fileProtocolMessage().includes("local file"), "file message");
  check(W.formatBytes(500) === "500 B" && W.formatBytes(2048) === "2 kB" && W.formatBytes(3.5 * 1048576) === "3.5 MB" &&
    W.formatBytes(80 * 1048576) === "80 MB", "formatBytes");
}

// ---- Fake fetch + Cache API ---------------------------------------------------------------------
const files = new Map<string, Uint8Array>();
let fetchCount = 0;
(globalThis as any).fetch = async (u: string) => {
  fetchCount++;
  const b = files.get(String(u));
  if (!b) return new Response("not found", { status: 404 });
  // Stream in 3 chunks, so progress gets several calls.
  const parts = [b.subarray(0, b.length / 3 | 0), b.subarray(b.length / 3 | 0, 2 * b.length / 3 | 0), b.subarray(2 * b.length / 3 | 0)];
  const stream = new ReadableStream<Uint8Array>({ start(c) { for (const p of parts) c.enqueue(p.slice()); c.close(); } });
  return new Response(stream, { status: 200, headers: { "Content-Length": String(b.length) } });
};
class FakeCache {
  m = new Map<string, Uint8Array>();
  async match(k: string) { const b = this.m.get(k); return b ? new Response(b.slice(), { headers: { "Content-Length": String(b.length) } }) : undefined; }
  async put(k: string, r: Response) { this.m.set(k, new Uint8Array(await r.arrayBuffer())); }
  async keys() { return [...this.m.keys()].map((k) => ({ url: k })); }
  async delete(r: { url: string }) { return this.m.delete(r.url); }
}
const fakeCache = new FakeCache();
(globalThis as any).caches = { open: async () => fakeCache };

async function loaderTests(): Promise<void> {
  current = "fetchBytes";
  const base = "https://site.test/demos/dist/";
  const data = new Uint8Array(1000).map((_, i) => (i * 7) & 255);
  files.set(base + "models/a.bin", data);
  const prog: [number, number][] = [];
  const a = await W.fetchBytes(base + "models/a.bin", { version: "v1", onProgress: (l, t) => prog.push([l, t]) });
  check(a.length === 1000 && a.every((v, i) => v === data[i]), "bytes intact");
  check(prog.length >= 4 && prog[prog.length - 1][0] === 1000 && prog.every(([, t]) => t === 1000), "progress with total");
  check(prog.every((p, i) => i === 0 || p[0] >= prog[i - 1][0]), "progress monotone");
  const n0 = fetchCount;
  const b = await W.fetchBytes(base + "models/a.bin", { version: "v1" });
  check(fetchCount === n0 && b.length === 1000, "second load from cache, no fetch");
  await W.fetchBytes(base + "models/a.bin", { version: "v2" });
  check(fetchCount === n0 + 1, "new version fetched");
  const keys = (await fakeCache.keys()).map((k) => k.url);
  check(keys.length === 1 && keys[0].endsWith("?v=v2"), "old version removed from cache: " + keys.join(" "));
  let msg = "";
  try { await W.fetchBytes(base + "models/missing.bin"); } catch (e) { msg = (e as Error).message; }
  check(msg.includes("HTTP 404") && msg.includes("missing.bin"), "readable 404 error: " + msg);
  const nc = fetchCount;
  await W.fetchBytes(base + "models/a.bin", { cache: false });
  check(fetchCount === nc + 1, "cache: false always fetches");

  current = "loadModelFile";
  files.set(base + "models/index.json", new TextEncoder().encode(JSON.stringify({
    "demo/a": { file: "a.bin", bytes: 1000, sha256: "abc" },
  })));
  const m = await W.loadModelFile("demo/a", { base });
  check(m.length === 1000, "model file by key");
  check((await fakeCache.keys()).some((k) => k.url.endsWith("a.bin?v=abc")), "cached under its sha256");
  msg = "";
  try { await W.loadModelFile("demo/none", { base }); } catch (e) { msg = (e as Error).message; }
  check(msg.includes("not in models/index.json"), "unknown key error");

  // ---- The hello crate --------------------------------------------------------------------------
  current = "hello";
  const dir = path.join(process.cwd(), "demos/dist/wasm/hello");
  const wasmFile = path.join(dir, "hello_bg.wasm"), glueFile = path.join(dir, "hello.js");
  if (!fs.existsSync(wasmFile) || !fs.existsSync(glueFile)) {
    if (process.env.WASM_REQUIRED) { check(false, "demos/dist/wasm/hello not built (run make wasm)"); return; }
    console.log("SKIP hello crate: demos/dist/wasm/hello/ not built (run make wasm)");
    return;
  }
  const bytes = new Uint8Array(fs.readFileSync(wasmFile));
  const glueUrl = url.pathToFileURL(glueFile).href;
  // Two separate module instances (the glue keeps one instance per module record).
  const g1: any = await import(glueUrl + "?inline");
  await W.instantiateInline(g1, bytes);
  files.set(base + "wasm/hello/hello_bg.wasm", bytes);
  files.set(base + "wasm/index.json", new TextEncoder().encode(JSON.stringify({
    hello: { file: "hello/hello_bg.wasm", bytes: bytes.length, sha256: "h1" } })));
  const g2: any = await import(glueUrl + "?fetch");
  let last = 0;
  await W.loadWasm(g2, "hello", { base, onProgress: (l) => { last = l; } });
  check(last === bytes.length, "fetched module, progress reached its size");

  for (const [name, g] of [["inline", g1], ["fetch", g2]] as const) {
    current = "hello/" + name;
    check(g.add(2, 40) === 42, "add");
    check(g.add(2147483647, 1) === -2147483648, "add wraps like i32");
    const v = new Float32Array(1000).map((_, i) => i * 0.5);
    check(g.sum_f32(v) === 249750, "sum_f32");
    check(g.sum_f32(new Float32Array(0)) === 0, "sum_f32 empty");
    const px = new Uint8Array([0, 1, 128, 255]);
    g.invert_u8(px);
    check(px.join(",") === "255,254,127,0", "invert_u8 writes back: " + px.join(","));
    for (const [w, h] of [[1, 1], [5, 3], [37, 23]]) {
      const img = new Float32Array(w * h).map((_, i) => Math.sin(i * 0.37) * 10);
      const o = g.box3(img, w, h) as Float32Array, r = box3Ref(img, w, h);
      let err = 0;
      for (let i = 0; i < r.length; i++) err = Math.max(err, Math.abs(o[i] - r[i]));
      check(o instanceof Float32Array && o.length === w * h && err < 1e-5, `box3 ${w}x${h}: err ${err}`);
    }
    check(g.greet("course") === "Hello, course, from Rust", "greet: " + g.greet("course"));
  }
}

function box3Ref(img: Float32Array, w: number, h: number): Float32Array {
  const o = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++)
      s += img[Math.min(h - 1, Math.max(0, y + dy)) * w + Math.min(w - 1, Math.max(0, x + dx))];
    o[y * w + x] = s / 9;
  }
  return o;
}

loaderTests().then(() => {
  console.log(`${checks - failures}/${checks} checks passed`);
  if (failures) process.exit(1);
}, (e) => { console.log("ERROR", e); process.exit(1); });
