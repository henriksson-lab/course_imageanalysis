// Runs the kalman1d demo model (no DOM) for the presets used in the slide figures and prints JSON.
// Called by figs.py: esbuild --bundle --platform=node tools/figs/track-predict/kfdata.ts | node
import { run, PRESETS, DEFAULT, truth, T } from "../../../demos/src/kalman1d/model";
const out: Record<string, unknown> = { T };
for (const id of ["cv", "missed"]) {
  const p = PRESETS.find(q => q.id === id)!;
  const s = { ...DEFAULT, ...p.s, missed: [...(p.s.missed ?? [])] };
  out[id] = { state: s, rec: run(s), truth: Array.from({ length: T }, (_, t) => truth(t, s.turn)) };
}
console.log(JSON.stringify(out));
