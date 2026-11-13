// Runs of the chanvese demo's model for chanvese/lec.qmd (demos/src/chanvese/model.ts,
// lib/levelset.ts cvStep with the demo's Δt = 10), plus the edge-based flow of the gac demo on
// the same image. Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/chanvese/data.ts --bundle --platform=node --outfile=<tmp>/cv-data.js
//   node <tmp>/cv-data.js <tmp>/cv-data.json
//   python3 -I tools/figs/chanvese/figs.py <tmp>/cv-data.json
import { writeFileSync } from "fs";
import * as LS from "../../../demos/src/lib/levelset";
import { N, scene, addNoise, initPhi, InitId, SceneId } from "../../../demos/src/chanvese/model";
import { edgeMap, gacTerms, boxPhi } from "../../../demos/src/gac/model";

const DT = 10;
const r3 = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);

interface S { image: SceneId; noise: number; init: InitId; lambda: number; eps: number }
const D: S = { image: "shapes", noise: 0.15, init: "grid", lambda: 0.1, eps: 1 };

function wrongOf(phi: ArrayLike<number>, mask: ArrayLike<number>, c1: number, c2: number): number {
  let k = 0;
  for (let p = 0; p < N * N; p++) { const obj = c1 >= c2 ? phi[p] < 0 : phi[p] >= 0; if ((obj ? 1 : 0) !== mask[p]) k++; }
  return k;
}

function run(name: string, s: S, at: number[]) {
  const sc = scene(s.image), I = addNoise(sc.img, s.noise), phi = initPhi(s.init);
  const P: LS.CVParams = { lambda: s.lambda, nu: 0, eps: s.eps, w1: 1, w2: 1 };
  const snaps = [];
  let it = 0;
  for (const target of at) {
    while (it < target) { LS.cvStep(I, phi, N, N, P, DT); it++; }
    const m = LS.cvMeans(I, phi), E = LS.cvEnergy(I, phi, N, N, P, m);
    let thr = 0;
    const th = (m.c1 + m.c2) / 2;
    for (let p = 0; p < N * N; p++) if ((I[p] > th ? 1 : 0) !== sc.mask[p]) thr++;
    const wrong = wrongOf(phi, sc.mask, m.c1, m.c2), pieces = LS.zeroContour(phi, N, N).lines.length;
    snaps.push({ it, phi: r3(phi), c1: m.c1, c2: m.c2, E: E.total, wrong, thr, pieces, inside: LS.insideCount(phi) });
    console.log(name, "it", it, "c1", m.c1.toFixed(3), "c2", m.c2.toFixed(3), "E", E.total.toFixed(1), "wrong", wrong, "threshold wrong", thr, "pieces", pieces, "inside", LS.insideCount(phi));
  }
  return { I: Array.from(I), mask: Array.from(sc.mask), snaps };
}

function runEdge(name: string, s: S, steps: number) {
  const sc = scene(s.image), I = addNoise(sc.img, s.noise);
  const GAC = { sigma: 2, k: 8, c: -0.3, eps: 0.3, a: 5 };
  const E = edgeMap(Float64Array.from(I, v => 255 * v), GAC.sigma, GAC.k);
  const T = gacTerms(E, GAC), dt = 0.8 * LS.cflLimit(T, N * N);
  const phi = boxPhi(); LS.reinitExact(phi, N, N);
  const work = new Float64Array(N * N);
  for (let i = 0; i < steps; i++) { LS.lsStep(phi, N, N, T, dt, "upwind", work); if ((i + 1) % 20 === 0) LS.reinitExact(phi, N, N); }
  let wrong = 0;
  for (let p = 0; p < N * N; p++) if ((phi[p] < 0 ? 1 : 0) !== sc.mask[p]) wrong++;
  const pieces = LS.zeroContour(phi, N, N).lines.length;
  console.log(name, "steps", steps, "wrong", wrong, "pieces", pieces);
  return { phi: r3(phi), wrong, pieces, steps };
}

const out = {
  N,
  noise: run("noise", { ...D, noise: 0.3, lambda: 0.2 }, [0, 300]),
  edgeNoise: runEdge("edge on noise", { ...D, noise: 0.3, lambda: 0.2 }, 2000),
  lambda0: run("lambda0", { ...D, lambda: 0 }, [300]),
  lambdaD: run("lambda0.1", { ...D }, [300]),
  lambda1: run("lambda1", { ...D, lambda: 1 }, [300]),
  ring: run("ring", { ...D, init: "ring" }, [0, 200, 600]),
  bad: run("badstart", { ...D, init: "hole", eps: 0.1 }, [0, 100]),
  grid: run("grid", { ...D }, [0, 100, 600]),
};
writeFileSync(process.argv[2], JSON.stringify(out));
