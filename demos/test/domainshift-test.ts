// Tests of the domain-shift unit's demo (demos/src/domainshift/model.ts): the logistic fit has
// zero gradient, standardisation with the microscope's own statistics undoes any gain and offset
// exactly (when the class balance is the same), and the numbers quoted on the pages. Run:
//   node_modules/.bin/esbuild demos/test/domainshift-test.ts --bundle --platform=node --log-level=warning | node
import * as D from "../src/domainshift/model";

declare const process: { exit(code: number): never };
let failures = 0, checks = 0, current = "";
function check(ok: boolean, msg: string): void {
  checks++;
  if (!ok) { failures++; console.log(`FAIL [${current}]: ${msg}`); }
}
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

current = "fit";
{
  const A = D.cellsA(300, 5, 0.5), w = D.fitLogistic(A, 1e-3);
  const g = [0, 0, 0];
  for (const c of A) { const e = D.pMitotic(w, c.f) - c.c; g[0] += e; g[1] += e * c.f[0]; g[2] += e * c.f[1]; }
  for (let a = 0; a < 3; a++) g[a] += 2e-3 * w[a];
  check(Math.max(...g.map(Math.abs)) < 1e-8, `gradient at the fit ${g}`);
}

current = "normalisation";
{
  const A = D.cellsA(200, 3, 0.5), B = D.toB(A, 0.9);   // the same cells through B
  const zA = D.standardise(A), zB = D.standardise(B);
  check(zA.every((c, i) => close(c.f[0], zB[i].f[0], 1e-9) && close(c.f[1], zB[i].f[1], 1e-9)), "per-microscope standardisation removes a gain and an offset exactly");
  check(close(D.shiftScore(A, A), 0, 1e-12), "no shift: score 0");
}

current = "quoted numbers";
{
  const T = D.trainA();
  check(close(D.accuracy(T.wNone, T.A), 0.9575, 1e-9), `accuracy on A ${D.accuracy(T.wNone, T.A)}, quoted 96 %`);
  const e = D.evaluateB(T, 0.8, 0.5);
  check(close(e.acc.none, 0.71, 0.0001) && close(e.acc.normalise, 0.93, 0.0001), `shift 0.8: as is ${e.acc.none} (71 %), normalised ${e.acc.normalise} (93 %)`);
  check(close(e.acc.augment, 0.85, 0.0001) && close(e.acc.finetune, 0.93, 0.0001), `shift 0.8: augmented ${e.acc.augment} (85 %), fine-tuned ${e.acc.finetune} (93 %)`);
  check(close(e.score, 1.01, 0.005), `shift score ${e.score}, quoted 1.01`);
  const f = D.evaluateB(T, 0, 0.2);
  check(close(f.acc.normalise, 0.8325, 0.0001) && close(f.acc.none, 0.9175, 0.0001), `no shift, fifth mitotic: normalised ${f.acc.normalise} (83 %), as is ${f.acc.none} (92 %)`);
  check(close(f.score, 0.37, 0.005), `no shift, fifth mitotic: score ${f.score}, quoted 0.37`);
}

console.log(`domainshift: ${checks - failures}/${checks} checks passed`);
if (failures) process.exit(1);
