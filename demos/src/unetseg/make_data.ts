// Writes the U-Net training and validation sets as raw bytes for train_unet.py (teacher tool,
// not part of the demo bundle). Run from the repository root:
//   node_modules/.bin/esbuild demos/src/unetseg/make_data.ts --bundle --platform=node --log-level=warning | node - OUTDIR
// OUTDIR gets unet_x.f32 (n × 64 × 64 float32 grey levels) and unet_y.u8 (n × 64 × 64 masks);
// training images are seeds 1..4000, validation seeds 100001..100400 (the demo uses 200001+);
// unet_ref_x.f32 is the demo's first image (seed 200001), for the reference check in the tests.
import { sample, N } from "./data";
declare const require: (m: string) => any;
declare const process: { argv: string[] };
declare const Buffer: { from(b: ArrayBuffer): unknown };
const { writeFileSync } = require("fs");

const out = process.argv[2] ?? ".";
for (const [name, s0, n] of [["train", 1, 4000], ["val", 100001, 400]] as const) {
  const X = new Float32Array(n * N * N), Y = new Uint8Array(n * N * N);
  for (let k = 0; k < n; k++) { const s = sample(s0 + k); X.set(s.image, k * N * N); Y.set(s.mask, k * N * N); }
  writeFileSync(`${out}/unet_${name}_x.f32`, Buffer.from(X.buffer));
  writeFileSync(`${out}/unet_${name}_y.u8`, Buffer.from(Y.buffer));
  let fgFrac = 0; for (let i = 0; i < Y.length; i++) fgFrac += Y[i];
  if (name === "val") writeFileSync(`${out}/unet_ref_x.f32`, Buffer.from(sample(200001).image.buffer));
  console.log(name, n, "images; nucleus fraction", (fgFrac / Y.length).toFixed(4));
}
