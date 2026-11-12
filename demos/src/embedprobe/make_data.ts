// Writes the synthetic training images of the embedprobe encoder for train_ssl.py (teacher
// tool, not part of the demo bundle). Run from the repository root:
//   node_modules/.bin/esbuild demos/src/embedprobe/make_data.ts --bundle --platform=node --log-level=warning | node - OUTDIR
// OUTDIR gets ssl_x.u8 (n × 128 × 128 grey levels, seeds 1..1500) and ssl_y.u8 (masks, used
// only to report probe accuracy, never for training the encoder), and ssl_ref.u8: the demo's
// image (seed 900001) for the reference check in the tests.
import { probeScene, W } from "./scene";
declare const require: (m: string) => any;
declare const process: { argv: string[] };
declare const Buffer: { from(b: ArrayBufferLike): unknown };
const { writeFileSync } = require("fs");

const out = process.argv[2] ?? ".";
const n = 1500, X = new Uint8Array(n * W * W), Y = new Uint8Array(n * W * W);
for (let k = 0; k < n; k++) {
  const s = probeScene(1 + k);
  for (let i = 0; i < W * W; i++) { X[k * W * W + i] = Math.round(s.image[i]); Y[k * W * W + i] = s.truth[i]; }
}
writeFileSync(`${out}/ssl_x.u8`, Buffer.from(X.buffer));
writeFileSync(`${out}/ssl_y.u8`, Buffer.from(Y.buffer));
const ref = probeScene(900001);
writeFileSync(`${out}/ssl_ref.u8`, Buffer.from(Uint8Array.from(ref.image, v => Math.round(v)).buffer));
writeFileSync(`${out}/ssl_ref_y.u8`, Buffer.from(ref.truth.buffer));
console.log("wrote", n, "images");
