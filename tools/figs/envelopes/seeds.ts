// Seeds for the envelopetest presets: for CSR patterns (seed 1..30), the radii where L − r leaves
// the 39-simulation pointwise envelope and the MAD p-value (39 and 199 simulations). Run:
//   node_modules/.bin/esbuild tools/figs/envelopes/seeds.ts --bundle --platform=node --log-level=warning | node
import * as PP from "../../../demos/src/lib/pointpattern";
import { makePattern, DEFAULT_SPEC, makeWindow } from "../../../demos/src/csrplay/source";
import { RADII, curve, reference, nullCurves } from "../../../demos/src/envelopetest/model";
const win = makeWindow("square");
for (let seed = 1; seed <= 30; seed++) {
  const p = makePattern({ ...DEFAULT_SPEC, model: "csr", seed });
  const sims = nullCurves(p.x.length, win, 199, "L", 31 * seed + 5), sc = sims.slice(0, 39);
  const obs = curve(p, win, "L"), idx = PP.rangeIdx(RADII, [1, 50]);
  const esc = PP.escapes(obs, PP.pointwiseEnvelope(sc), idx), mad = PP.madTest(obs, sc, reference("L"), idx);
  const esc199 = PP.escapes(obs, PP.pointwiseEnvelope(sims), idx), mad199 = PP.madTest(obs, sims, reference("L"), idx);
  console.log(seed, esc.length, esc.map(k=>RADII[k]).join(","), mad.p.toFixed(3), "| s199", esc199.length, mad199.p.toFixed(3));
}
