// Tests of the finetune-strategies unit's low-rank demo (demos/src/lowrank/model.ts): synthetic
// changes have the requested singular values, the truncated SVD is the best rank-r approximation
// (Eckart–Young: its error matches the singular values and beats random rank-r matrices), the
// parameter counts, and the numbers of the real fine-tuning change quoted on the pages. Run:
//   node_modules/.bin/esbuild demos/test/lowrank-test.ts --bundle --platform=node --log-level=warning | node
import * as M from "../src/lowrank/model";
import { rng } from "../src/lib/ui";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;
const fro = (A: ArrayLike<number>) => Math.sqrt(Array.from(A).reduce((s, v) => s + v * v, 0));

current = "synthetic";
{
  const c = M.syntheticChange(24, 24, [3, 1.5], 0, 3), S = M.decompose(c);
  check(close(S.S[0], 3, 1e-9) && close(S.S[1], 1.5, 1e-9) && S.S[2] < 1e-9, `singular values ${S.S[0]}, ${S.S[1]}, ${S.S[2]}: expected 3, 1.5, 0`);
  check(close(M.relError(S.S, 1), 1.5 / Math.hypot(3, 1.5), 1e-12), "rank-1 error of (3, 1.5) = 1.5 / sqrt(11.25) = 0.447");
  check(M.relError(S.S, 2) < 1e-9, "rank 2 is exact");
}

current = "Eckart–Young";
{
  const c = M.syntheticChange(12, 20, [], 3, 9), S = M.decompose(c), nrm = fro(c.D);
  for (const r of [1, 3, 6]) {
    const Dr = M.truncate(S, c.m, c.n, r);
    const err = fro(c.D.map((v, i) => v - Dr[i])) / nrm;
    check(close(err, M.relError(S.S, r), 1e-9), `rank ${r}: error from the matrices ${err} = from the singular values ${M.relError(S.S, r)}`);
    // random rank-r matrices B·A, each scaled optimally, are never better
    const R = rng(r);
    let bestRandom = Infinity;
    for (let t = 0; t < 50; t++) {
      const B = Float64Array.from({ length: c.m * r }, () => R.norm()), A = Float64Array.from({ length: r * c.n }, () => R.norm());
      const BA = new Float64Array(c.m * c.n);
      for (let i = 0; i < c.m; i++) for (let j = 0; j < c.n; j++) { let v = 0; for (let k = 0; k < r; k++) v += B[i * r + k] * A[k * c.n + j]; BA[i * c.n + j] = v; }
      let dot = 0; for (let i = 0; i < BA.length; i++) dot += BA[i] * c.D[i];
      const a = dot / (fro(BA) ** 2);
      bestRandom = Math.min(bestRandom, fro(c.D.map((v, i) => v - a * BA[i])) / nrm);
    }
    check(err <= bestRandom, `rank ${r}: SVD error ${err} ≤ best random ${bestRandom}`);
  }
  check(M.fullParams(32, 100) === 3200 && M.lowRankParams(32, 100, 3) === 396, "32 × 100: 3200 numbers, rank 3: 396");
}

current = "real change (quoted)";
{
  const R = M.realChange();
  check(close(R.accFull, 0.987, 0.002), `full change: accuracy ${R.accFull}, quoted 98.7 %`);
  check(close(R.accByRank[3], 0.98, 0.002), `rank 3: ${R.accByRank[3]}, quoted 98.0 %`);
  check(close(R.oldByRank[0], 0.98, 0.003) && close(R.oldByRank[3], 0.415, 0.003), `old task at rank 0 ${R.oldByRank[0]} (98 %), rank 3 ${R.oldByRank[3]} (42 %)`);
  check(close(M.relError(R.svds[0].S, 3), 0.42, 0.005), `layer 1 rank-3 error ${M.relError(R.svds[0].S, 3)}, quoted 42 %`);
}

console.log(`lowrank: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
