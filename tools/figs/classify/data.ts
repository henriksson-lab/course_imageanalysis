// Data for the slide figures of classify/ (the linclass demo's own presets, so the points and
// numbers match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/classify/data.ts
// then python3 -I tools/figs/classify/figs.py <dir>.
import * as fs from "fs";
import * as os from "os";
import { LIN_PRESETS, nearestMeanLine, errors } from "../../../demos/src/linclass/geom";

const TMP = process.env.FIGTMP || os.tmpdir();
const out: Record<string, unknown> = {};
for (const p of LIN_PRESETS) {
  const pts = p.points();
  const nm = nearestMeanLine(pts);
  out[p.id] = { points: pts, range: p.range, line: p.line, mean: nm, meanErrors: errors(nm, pts) };
  console.log(p.id, "n", pts.length, "nearest-mean errors", errors(nm, pts));
}
fs.writeFileSync(TMP + "/classify.json", JSON.stringify(out));
console.log("wrote", TMP + "/classify.json");
