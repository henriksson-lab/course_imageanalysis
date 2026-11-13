// Browser check of the Wasm pipeline (demos/wasm/README section in demos/README.md):
// ?mode=fetch  -> loadWasm(): same-origin fetch with progress, Cache API, then the glue's init
// ?mode=inline -> instantiateInline(): bytes bundled by esbuild's binary loader (works on file://)
// Then one file from demos/dist/models/ through loadModelFile(), when models are built: the
// smallest, or ?model=<key> (e.g. sam2/sam2.1_hiera_tiny.safetensors).
// Results go into a table; any mismatch is a console.error (the headless check looks for those).
import * as glue from "../../../dist/wasm/hello/hello.js";
import inlineWasm from "../../../dist/wasm/hello/hello_bg.wasm";
import { loadWasm, instantiateInline, inlineBytes, loadStatus, loadModelFile, modelIndex, isFileProtocol, formatBytes } from "../../../src/lib/wasm";

const out = document.getElementById("out")!;
const table = document.createElement("table");
out.appendChild(table);
function row(name: string, value: string, ok: boolean): void {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${name}</td><td class="${ok ? "ok" : "bad"}">${ok ? "ok" : "FAIL"}</td><td>${value}</td>`;
  table.appendChild(tr);
  if (!ok) console.error(`wasm check failed: ${name}: ${value}`);
}

function box3Ref(img: Float32Array, w: number, h: number): Float32Array {
  const o = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = Math.min(h - 1, Math.max(0, y + dy)), xx = Math.min(w - 1, Math.max(0, x + dx));
      s += img[yy * w + xx];
    }
    o[y * w + x] = s / 9;
  }
  return o;
}

async function main(): Promise<void> {
  const mode = new URLSearchParams(location.search).get("mode") ?? (isFileProtocol() ? "inline" : "fetch");
  const t0 = performance.now();
  if (mode === "inline") {
    await instantiateInline(glue, inlineWasm);
    row("instantiate (inline)", `${formatBytes(inlineBytes(inlineWasm).length)} in ${(performance.now() - t0).toFixed(1)} ms`, true);
  } else {
    const st = loadStatus(out, "Loading module");
    let last = 0;
    await loadWasm(glue, "hello", { onProgress: (l, t) => { st.progress(l, t); last = l; } });
    st.done();
    row("instantiate (fetch)", `${formatBytes(last)} in ${(performance.now() - t0).toFixed(1)} ms`, last > 0);
  }
  row("add(2, 40)", String(glue.add(2, 40)), glue.add(2, 40) === 42);
  const v = new Float32Array(1000).map((_, i) => i * 0.5);
  row("sum_f32", String(glue.sum_f32(v)), glue.sum_f32(v) === 249750);
  const px = new Uint8Array([0, 1, 128, 255]);
  glue.invert_u8(px);
  row("invert_u8", px.join(","), px.join(",") === "255,254,127,0");
  const w = 37, h = 23, img = new Float32Array(w * h).map((_, i) => Math.sin(i * 0.37) * 10);
  const a = glue.box3(img, w, h), b = box3Ref(img, w, h);
  let err = 0;
  for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
  row("box3 vs TypeScript", `max |diff| = ${err.toExponential(1)}`, a.length === w * h && err < 1e-5);
  row("greet", glue.greet("course"), glue.greet("course") === "Hello, course, from Rust");

  // A model file, when `make models` has run (skipped, not failed, when absent).
  try {
    const index = await modelIndex();
    const keys = Object.keys(index).sort((p, q) => index[p].bytes - index[q].bytes);
    // ?model=<key of models/index.json> loads that file (with a progress bar), else the smallest.
    const want = new URLSearchParams(location.search).get("model");
    if (keys.length) {
      const k = want ?? keys[0], t1 = performance.now();
      const st = loadStatus(out, "Loading " + k);
      const bytes = await loadModelFile(k, { onProgress: st.progress });
      st.done();
      row(`model file ${k}`, `${formatBytes(bytes.length)} in ${(performance.now() - t1).toFixed(0)} ms`, bytes.length === index[k].bytes);
    }
  } catch (e) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>model files</td><td>skipped</td><td>${(e as Error).message}</td>`;
    table.appendChild(tr);
  }
  document.body.dataset.done = "1";
}
main().catch((e) => { row("error", String((e as Error).message ?? e), false); });
