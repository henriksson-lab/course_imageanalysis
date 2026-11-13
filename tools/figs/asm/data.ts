// Active shape model runs for asm/lec.qmd, computed with the asm demo's own code
// (demos/src/asm/model.ts, lib/shapes.ts, lib/snake.ts) and its presets: same training
// (40 nuclei, k = 3, 3 pyramid levels), same scenes, start poses and search options.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/asm/data.ts --bundle --platform=node --outfile=<tmp>/asm-data.js
//   node <tmp>/asm-data.js <tmp>/asm-data.json
// then python3 -I tools/figs/asm/figs.py <tmp>/asm-data.json
import { writeFileSync } from "fs";
import { Grey, gaussianPyramid } from "../../../demos/src/lib/imageops";
import { Shape, nPoints, shapeNormals } from "../../../demos/src/lib/shapes";
import { edgeMap, gridGradient, implicitSolver, implicitStep, sampleField, resample } from "../../../demos/src/lib/snake";
import { Search, SearchOptions, SEARCH_DEFAULTS, train, makeScene, poseOf, poseScale, poseAngle, rmsError, toImage, sampleProfile, offsetCosts } from "../../../demos/src/asm/model";

const r3 = (v: number) => Math.round(v * 1000) / 1000;
const arr = (a: ArrayLike<number>) => Array.from(a, r3);
const img = (g: Grey) => ({ w: g.width, h: g.height, d: Array.from(g.data, v => Math.round(v * 1000) / 1000) });

const M = train({ family: "nucleus", k: 3, levels: 3 });
const OUTSIDE = [0.5, 0.55, 0.35];
const OCC: [number, number, number, number, number] = [20, 52, 110, 62, 9];

function setup(scene: { noise?: number; occlude?: boolean; outside?: boolean }, start: [number, number, number, number]) {
  const sc = makeScene(M.fam, { noise: scene.noise ?? 0.03, clutter: 0, occluder: scene.occlude ? OCC : null, params: scene.outside ? OUTSIDE : undefined });
  const P = sc.pose, [dx, dy, rot, s] = start;
  const st = poseOf(poseScale(P) * s, poseAngle(P) + rot * Math.PI / 180, P.tx + dx, P.ty + dy);
  return { sc, st };
}
function full(label: string, scene: any, start: [number, number, number, number], o: Partial<SearchOptions>) {
  const { sc, st } = setup(scene, start);
  const S = new Search(M, sc.img, st, { ...SEARCH_DEFAULTS, ...o });
  const X0 = S.X;
  while (!S.done()) S.iterate();
  const e0 = rmsError(X0, sc.truth), e = rmsError(S.X, sc.truth);
  console.log(label, "start", e0.toFixed(2), "end", e.toFixed(2), "iterations", S.total);
  return { img: img(sc.img), truth: arr(sc.truth), start: arr(X0), end: arr(S.X), e0: r3(e0), e: r3(e), b: arr(S.b.subarray(0, 3)), lim: arr(Array.from(M.pdm.values.slice(0, 3), v => 3 * Math.sqrt(v))) };
}

// 1. clean image, good start: normals, a profile, the costs at one landmark
const clean = setup({}, [10, -4, 10, 0.95]);
const S1 = new Search(M, clean.sc.img, clean.st, { ...SEARCH_DEFAULTS });
const N1 = shapeNormals(S1.X);
const L = 4;   // a landmark on the right side
const prof: number[] = [], costs = Array.from(offsetCosts(M, clean.sc.img, 0, S1.X, N1, L, SEARCH_DEFAULTS));
for (let t = -9; t <= 9; t++) prof.push(sampleProfile(clean.sc.img, S1.X[2 * L] + t * N1[2 * L], S1.X[2 * L + 1] + t * N1[2 * L + 1], N1[2 * L], N1[2 * L + 1], 0).raw[0]);
console.log("landmark", L, "costs", costs.map(v => v.toFixed(1)).join(" "));
const normals = { img: img(clean.sc.img), X: arr(S1.X), N: arr(N1), truth: arr(clean.sc.truth), L, prof: arr(prof), costs: arr(costs) };

// 2. one iteration, sub-step by sub-step, on the noisy image
const noisy = setup({ noise: 0.25 }, [10, -4, 10, 0.95]);
const S2 = new Search(M, noisy.sc.img, noisy.st, { ...SEARCH_DEFAULTS });
const steps: any = { start: arr(S2.X) };
S2.suggest(); steps.Y = arr(S2.Y!);
S2.fitPose(); steps.pose = arr(S2.X);
S2.project(); steps.project = arr(S2.X); steps.bRaw = arr(S2.bRaw.subarray(0, 3));
S2.clamp(); steps.clamp = arr(S2.X); steps.b = arr(S2.b.subarray(0, 3));
console.log("bRaw", steps.bRaw, "b", steps.b, "limits", Array.from(M.pdm.values.slice(0, 3), v => (3 * Math.sqrt(v)).toFixed(2)));
const loop = { img: img(noisy.sc.img), truth: arr(noisy.sc.truth), ...steps };

const noise = full("noise", { noise: 0.25 }, [10, -4, 10, 0.95], {});
const free = full("free", { noise: 0.25 }, [10, -4, 10, 0.95], { constrain: false });
const occ = full("occlusion", { occlude: true }, [10, -4, 10, 0.95], {});
const occFree = full("occlusion free", { occlude: true }, [10, -4, 10, 0.95], { constrain: false });
const far1 = full("far 1 level", {}, [-22, 12, -20, 0.9], { levels: 1 });
const far2 = full("far 2 levels", {}, [-22, 12, -20, 0.9], { levels: 2 });
const far3 = full("far 3 levels", {}, [-22, 12, -20, 0.9], { levels: 3 });
const outside = full("outside", { outside: true }, [4, -2, 5, 0.95], {});
// the demo's comparison snake (asm/main.ts runSnake): from the start mean shape, 64 points, σ = 1.5, α = 0.05, β = 0.02, τ = 1, w = 2, 300 steps
const os = setup({ outside: true }, [4, -2, 5, 0.95]);
const X0 = toImage(M.pdm.mean, os.st), n = 64;
let c = resample({ x: Float64Array.from({ length: nPoints(X0) }, (_, i) => X0[2 * i]), y: Float64Array.from({ length: nPoints(X0) }, (_, i) => X0[2 * i + 1]) }, n);
const f = edgeMap(os.sc.img, 1.5), F = gridGradient(f), sol = implicitSolver(n, 0.05, 0.02, 1);
for (let it = 0; it < 300; it++) c = implicitStep(c, sol, sampleField(F, c, 2));
function outlineDist(T: Shape, cx: Float64Array, cy: Float64Array): number {
  const m = nPoints(T); let s = 0;
  for (let i = 0; i < cx.length; i++) {
    let best = Infinity;
    for (let j = 0; j < m; j++) {
      const k = (j + 1) % m, ax = T[2 * j], ay = T[2 * j + 1], bx = T[2 * k], by = T[2 * k + 1];
      const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((cx[i] - ax) * vx + (cy[i] - ay) * vy) / (vx * vx + vy * vy || 1)));
      best = Math.min(best, Math.hypot(cx[i] - ax - t * vx, cy[i] - ay - t * vy));
    }
    s += best * best;
  }
  return Math.sqrt(s / cx.length);
}
const end = Float64Array.from(outside.end);
const asmDist = outlineDist(os.sc.truth, Float64Array.from({ length: nPoints(end) }, (_, i) => end[2 * i]), Float64Array.from({ length: nPoints(end) }, (_, i) => end[2 * i + 1]));
const snakeDist = outlineDist(os.sc.truth, c.x, c.y);
console.log("outside: snake", snakeDist.toFixed(2), "px from outline; ASM", asmDist.toFixed(2), "b", outside.b, "limits", outside.lim);
const pyr = gaussianPyramid(clean.sc.img, 3).map(img);   // the "far start" scene is the clean scene
writeFileSync(process.argv[2], JSON.stringify({ normals, loop, noise, free, occ, occFree, far1, far2, far3, outside: { ...outside, snake: { x: arr(c.x), y: arr(c.y) }, snakeDist: r3(snakeDist), asmDist: r3(asmDist) }, pyr, m: SEARCH_DEFAULTS.m }));
