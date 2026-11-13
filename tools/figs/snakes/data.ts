// Snake runs for snakes/lec.qmd, computed with the snake demo's own code
// (demos/src/snake/{images,model}.ts, demos/src/lib/snake.ts) and the same presets.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/snakes/data.ts --bundle --platform=node --outfile=<tmp>/snakes-data.js
//   node <tmp>/snakes-data.js <tmp>/snakes-data.json
// then python3 -I tools/figs/snakes/figs.py <tmp>/snakes-data.json
import { writeFileSync } from "fs";
import { Grey } from "../../../demos/src/lib/imageops";
import { circleContour, Contour, modeAmplitudes } from "../../../demos/src/lib/snake";
import { cleanImage, addNoise, ImageId } from "../../../demos/src/snake/images";
import { DEFAULTS, Settings, Sim, makeFields } from "../../../demos/src/snake/model";

const r3 = (v: number) => Math.round(v * 1000) / 1000;
const arr = (a: ArrayLike<number>) => Array.from(a, r3);
const cont = (c: Contour) => ({ x: arr(c.x), y: arr(c.y) });
const img = (g: Grey) => Array.from(g.data, v => Math.round(v * 1000) / 1000);

interface P { image: Exclude<ImageId, "cell">; noise: number; r: number; s: Partial<Settings> }
function run(p: P, at: number[]) {
  const s = { ...DEFAULTS, ...p.s } as Settings;
  const I = addNoise(cleanImage(p.image), p.noise);
  const F = makeFields(I, s.sigma, s.force === "gvf", s.mu);
  const sim = new Sim(circleContour(63.5, 63.5, p.r, s.n), s, F);
  const snaps: any[] = [];
  for (const t of at) {
    while (sim.iter < t && !sim.exploded()) sim.step();
    const c = sim.c, n = c.x.length;
    let mr = 0; for (let i = 0; i < n; i++) mr += Math.hypot(c.x[i] - 63.5, c.y[i] - 63.5) / n;
    const amp = modeAmplitudes(c);
    snaps.push({ t: sim.iter, c: cont(c), meanR: r3(mr), zig: amp[amp.length - 1], exploded: sim.exploded() });
    console.log(p.image, JSON.stringify(p.s), "iter", sim.iter, "mean r", mr.toFixed(2), "zigzag amp", amp[amp.length - 1].toExponential(2), sim.exploded() ? "EXPLODED" : "");
  }
  return { img: img(I), f: img(F.f), start: p.r, snaps };
}

// explosion: preset "tau" (explicit, τ = 2) and the same with τ = 1
const tau2 = run({ image: "disk", noise: 0.05, r: 45, s: { rule: "explicit", tau: 2 } }, [0, 20, 30, 35, 38, 40, 45, 50, 54, 55]);
const tau1 = run({ image: "disk", noise: 0.05, r: 45, s: { rule: "explicit", tau: 1 } }, [0, 50, 100, 200]);
// failure presets
const alpha = run({ image: "disk", noise: 0.05, r: 45, s: { alpha: 1 } }, [0, 500]);
const beta = run({ image: "square", noise: 0.05, r: 50, s: { beta: 2 } }, [0, 500]);
const sigma = run({ image: "disk", noise: 0.25, r: 45, s: { sigma: 0.5, w: 4 } }, [0, 500]);
const far = run({ image: "disk", noise: 0.05, r: 58, s: { alpha: 0.01 } }, [0, 300]);
const disk = run({ image: "disk", noise: 0.05, r: 45, s: {} }, [0, 300]);
// edge maps of the noisy disk at two scales (image term slide)
const noisy = addNoise(cleanImage("disk"), 0.25);
const edge = [0.5, 3].map(sg => { const F = makeFields(noisy, sg, false, 0.2); return { sigma: sg, f: img(F.f), fx: arr(F.edge.fx), fy: arr(F.edge.fy) }; });
writeFileSync(process.argv[2], JSON.stringify({ N: 128, tau2, tau1, alpha, beta, sigma, far, disk, noisy: img(noisy), edge }));
