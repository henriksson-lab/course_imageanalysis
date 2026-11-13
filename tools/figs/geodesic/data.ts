// Runs of the gac demo's flow for geodesic/lec.qmd (demos/src/gac/model.ts, lib/levelset.ts),
// with the demo's settings: Δt = 0.8 × the CFL limit, exact reinitialisation every 20 steps.
// Run from the repo root:
//   python3 -I tools/figs/geodesic/cell.py <tmp>/cell256.json        (the micrograph as numbers)
//   node_modules/.bin/esbuild tools/figs/geodesic/data.ts --bundle --platform=node --outfile=<tmp>/gac-data.js
//   node <tmp>/gac-data.js <tmp>/cell256.json <tmp>/gac-data.json
//   python3 -I tools/figs/geodesic/figs.py <tmp>/gac-data.json
import { readFileSync, writeFileSync } from "fs";
import * as LS from "../../../demos/src/lib/levelset";
import { fromArray, resize } from "../../../demos/src/lib/imageops";
import { N, synthClean, addNoise, edgeMap, gacTerms, circlesPhi, boxPhi, GacParams } from "../../../demos/src/gac/model";

const REINIT = 20;
const r3 = (a: ArrayLike<number>) => Array.from(a, v => Math.round(v * 1000) / 1000);
const synth = synthClean();

interface Setup { image: "synth" | "cell"; noise: number; sigma: number; k: number; p: GacParams; init: "box" | { x: number; y: number; r: number }[] }

const cell256 = JSON.parse(readFileSync(process.argv[2], "utf8")) as number[];
const cell = Float64Array.from(resize(fromArray(256, 256, cell256), N, N, "area").data);

function run(name: string, S: Setup, at: number[]) {
  const I = addNoise(S.image === "cell" ? cell : synth.img, S.noise);
  const E = edgeMap(I, S.sigma, S.k), T = gacTerms(E, S.p);
  const dt = 0.8 * LS.cflLimit(T, N * N);
  const phi = S.init === "box" ? boxPhi() : circlesPhi(S.init);
  const work = new Float64Array(N * N);
  const snaps: { step: number; t: number; phi: number[]; pieces: number; inside: number; wrong: number }[] = [];
  let steps = 0;
  for (const target of at) {
    while (steps < target) {
      LS.lsStep(phi, N, N, T, dt, "upwind", work);
      steps++;
      if (steps % REINIT === 0) LS.reinitExact(phi, N, N);
    }
    let wrong = 0;
    for (let p = 0; p < N * N; p++) if ((phi[p] < 0 ? 1 : 0) !== synth.mask[p]) wrong++;
    const pieces = LS.zeroContour(phi, N, N).lines.length, inside = LS.insideCount(phi);
    snaps.push({ step: steps, t: steps * dt, phi: r3(phi), pieces, inside, wrong: S.image === "synth" ? wrong : -1 });
    console.log(name, "step", steps, "t", (steps * dt).toFixed(1), "pieces", pieces, "inside", inside, "wrong", S.image === "synth" ? wrong : "-");
  }
  return { I: r3(I), g: r3(E.g), gx: r3(E.gx), gy: r3(E.gy), dt, snaps };
}

const D = { image: "synth" as const, noise: 12, sigma: 1.5, k: 8 };
const out: Record<string, unknown> = { N, mask: Array.from(synth.mask) };
// stage 1, braking balloon: grow from inside the strong disk A, no advection
out.brake = run("brake", { ...D, p: { c: 0.3, eps: 0.3, a: 0 }, init: [{ x: 28, y: 28, r: 4 }] }, Array.from({ length: 25 }, (_, i) => 25 * i));
// stage 2 vs stage 1 on the shrink-wrap preset
out.wrap = run("wrap a=5", { ...D, p: { c: -0.3, eps: 0.3, a: 5 }, init: "box" }, [0, 400, 800, 1200]);
const tWrap = (out.wrap as any).snaps[3].t;
const r0 = run("wrap a=0 (probe dt)", { ...D, p: { c: -0.3, eps: 0.3, a: 0 }, init: "box" }, [0]);
const n0 = Math.round(tWrap / r0.dt);
out.wrap0 = run("wrap a=0", { ...D, p: { c: -0.3, eps: 0.3, a: 0 }, init: "box" }, [0, n0]);
// balloon: C shape without and with balloon, and the leak
out.cNo = run("C, no balloon", { ...D, p: { c: 0, eps: 1, a: 2 }, init: [{ x: 100, y: 64, r: 30 }] }, [0, 1500, 3000]);
out.cYes = run("C, c = -0.3", { ...D, p: { c: -0.3, eps: 1, a: 2 }, init: [{ x: 100, y: 64, r: 30 }] }, [0, 1500, 3000]);
out.leak = run("leak c=1 a=1", { ...D, p: { c: 1, eps: 0.3, a: 1 }, init: [{ x: 26, y: 100, r: 6 }] }, [0, 100, 200]);
out.hold = run("hold c=0.3 a=3", { ...D, p: { c: 0.3, eps: 0.3, a: 3 }, init: [{ x: 26, y: 100, r: 6 }] }, [0, 400, 800]);
// σ: g of the noisy image (noise 25) for three smoothing widths
out.sig = [0, 1.5, 4].map(s => {
  const E = edgeMap(addNoise(synth.img, 25), s, 8);
  const sorted = Array.from(E.g).sort((a, b) => a - b);
  console.log("sigma", s, "median g", sorted[N * N / 2].toFixed(3));
  return { sigma: s, g: r3(E.g), median: sorted[N * N / 2] };
});
// inside-out on the micrograph (preset "micrograph: nucleus")
out.cell = run("cell nucleus", { image: "cell", noise: 0, sigma: 1.5, k: 4, p: { c: 0.3, eps: 0.5, a: 3 }, init: [{ x: 74, y: 62, r: 6 }] }, [0, 600, 1200, 2400]);
writeFileSync(process.argv[3], JSON.stringify(out));
