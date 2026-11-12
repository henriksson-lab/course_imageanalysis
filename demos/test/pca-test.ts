// Tests for the pca2d demo (demos/src/pca2d/data.ts): the closed-form 2×2 eigen-decomposition
// against a brute-force search over directions, and the identities the PCA unit teaches
// (variance along u = uᵀSu, error = tr S − uᵀSu, best error = λ2). Run alone:
//   node_modules/.bin/esbuild demos/test/pca-test.ts --bundle --platform=node | node
import { makeData, pca, along, varAlong, unit, KINDS, type V2 } from "../src/pca2d/data";
import { pca as linaePca, makeData as linaeData } from "../src/linae/data";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0;
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; if (failures <= 20) console.log("FAIL: " + msg); }
}
const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

function testCloud(P: V2[], name: string): void {
  const pc = pca(P);
  check(pc.l1 >= pc.l2 - 1e-15, `${name}: l1 >= l2`);
  check(close(pc.l1 + pc.l2, pc.a + pc.c) && close(pc.l1 * pc.l2, pc.a * pc.c - pc.b * pc.b, 1e-7), `${name}: trace and determinant`);
  // S u1 = l1 u1
  const u = unit(pc.theta);
  check(close(pc.a * u[0] + pc.b * u[1], pc.l1 * u[0], 1e-7) && close(pc.b * u[0] + pc.c * u[1], pc.l1 * u[1], 1e-7), `${name}: S u1 = l1 u1`);
  // brute force over angles: max variance = l1, min = l2
  let vmax = -Infinity, vmin = Infinity;
  for (let k = 0; k < 3600; k++) {
    const th = -Math.PI / 2 + Math.PI * k / 3600, r = along(P, pc.m, th);
    check(close(r.meanSq, varAlong(pc, th), 1e-9), `${name}: along = uᵀSu`);
    check(close(r.err, r.total - r.meanSq) && close(r.total, pc.a + pc.c, 1e-9), `${name}: error = tr S − uᵀSu`);
    vmax = Math.max(vmax, r.meanSq); vmin = Math.min(vmin, r.meanSq);
  }
  check(Math.abs(vmax - pc.l1) < 1e-5 * (pc.l1 + 1e-12) + 1e-12 && vmax <= pc.l1 + 1e-12, `${name}: max variance = l1 (${vmax} vs ${pc.l1})`);
  check(Math.abs(vmin - pc.l2) < 1e-5 * (pc.l1 + 1e-12) + 1e-12 && vmin >= pc.l2 - 1e-12, `${name}: min variance = l2`);
  check(close(along(P, pc.m, pc.theta).err, pc.l2, 1e-9), `${name}: best error = l2`);
}

for (const k of KINDS) testCloud(makeData(k), k);
const r = rng(77);
for (let t = 0; t < 200; t++) {
  const n = 2 + Math.floor(r.unif() * 30), P: V2[] = [];
  for (let i = 0; i < n; i++) P.push([r.norm(), 0.3 * r.norm() + r.unif() * r.norm()]);
  testCloud(P, `random ${t}`);
}
// the four images of the exercise
{
  const pc = pca(makeData("four"));
  check(close(pc.a, 0.05) && close(pc.b, 0.03) && close(pc.c, 0.05) && close(pc.l1, 0.08) && close(pc.l2, 0.02), "four images: S and eigenvalues");
}
// same data and covariance as the linae demo of the autoencoder unit
{
  const a = pca(makeData("pixels")), b = linaePca(linaeData("pixels"));
  check(close(a.l1, b.l1) && close(a.l2, b.l2) && close(a.theta, b.theta), "pixels = linae data");
}

console.log(`${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
