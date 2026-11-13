// Snake runs for snake-matrix/lec.qmd, computed with the snake demo's own code
// (demos/src/snake/{images,model}.ts, demos/src/lib/snake.ts) and its presets
// (far start with edge force / GVF, balloon from inside, balloon leaks, two blobs).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/snake-matrix/data.ts --bundle --platform=node --outfile=<tmp>/sm-data.js
//   node <tmp>/sm-data.js <tmp>/sm-data.json
// then python3 -I tools/figs/snake-matrix/figs.py <tmp>/sm-data.json
import { writeFileSync } from "fs";
import { Grey } from "../../../demos/src/lib/imageops";
import { circleContour, Contour, snakeEigenvalues } from "../../../demos/src/lib/snake";
import { cleanImage, addNoise, ImageId } from "../../../demos/src/snake/images";
import { DEFAULTS, Settings, Sim, makeFields } from "../../../demos/src/snake/model";

const r3 = (v: number) => Math.round(v * 1000) / 1000;
const arr = (a: ArrayLike<number>) => Array.from(a, r3);
const cont = (c: Contour) => ({ x: arr(c.x), y: arr(c.y) });
const img = (g: Grey) => Array.from(g.data, v => Math.round(v * 1000) / 1000);

interface P { image: Exclude<ImageId, "cell">; noise: number; r: number; s: Partial<Settings> }
function run(label: string, p: P, at: number[], field = false) {
  const s = { ...DEFAULTS, ...p.s } as Settings;
  const I = addNoise(cleanImage(p.image), p.noise);
  const F = makeFields(I, s.sigma, s.force === "gvf", s.mu);
  const sim = new Sim(circleContour(63.5, 63.5, p.r, s.n), s, F);
  const snaps: any[] = [];
  for (const t of at) {
    while (sim.iter < t && !sim.exploded()) sim.step();
    const c = sim.c, n = c.x.length;
    let mr = 0, maxx = -1e9; for (let i = 0; i < n; i++) { mr += Math.hypot(c.x[i] - 63.5, c.y[i] - 63.5) / n; maxx = Math.max(maxx, c.x[i]); }
    snaps.push({ t: sim.iter, c: cont(c), meanR: r3(mr), maxX: r3(maxx) });
    console.log(label, "iter", sim.iter, "mean r", mr.toFixed(2), "max x", maxx.toFixed(1));
  }
  const fld = field ? (s.force === "gvf" ? F.gvf! : F.edge) : null;
  return { img: img(I), snaps, fx: fld ? arr(fld.fx) : null, fy: fld ? arr(fld.fy) : null };
}

const far = run("far edge", { image: "disk", noise: 0.05, r: 58, s: { alpha: 0.01 } }, [0, 100], true);
const fargvf = run("far gvf", { image: "disk", noise: 0.05, r: 58, s: { alpha: 0.01, force: "gvf" } }, [0, 100], true);
const balloon = run("balloon", { image: "disk", noise: 0.05, r: 8, s: { kappa: 0.3 } }, [0, 20, 40, 80, 300]);
const leak = run("leak 0.6", { image: "weak", noise: 0.05, r: 8, s: { kappa: 0.6 } }, [0, 20, 40, 80, 150]);
const leak3 = run("leak 0.3", { image: "weak", noise: 0.05, r: 8, s: { kappa: 0.3 } }, [0, 300, 1000]);
const blobs = run("blobs", { image: "blobs", noise: 0.05, r: 58, s: { n: 50 } }, [0, 300, 1000]);
const lam = (a: number, b: number, n: number) => Array.from(snakeEigenvalues(n, a, b));
writeFileSync(process.argv[2], JSON.stringify({ N: 128, far, fargvf, balloon, leak, leak3, blobs, lam40: lam(0.1, 0.05, 40) }));
