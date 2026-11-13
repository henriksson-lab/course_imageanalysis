// Data for the extra slide figures of pixelclass/ (run with the demo's own pipeline, so the
// numbers match the demo). Run: FIGTMP=<dir> tools/figs/evaluation/run.sh tools/figs/pixelclass/data.ts
// then python3 -I tools/figs/pixelclass/figs.py <dir>. Numbers are printed.
import * as fs from "fs";
import * as os from "os";
import { synthScene } from "../../../demos/src/pixelclass/scene";
import { PIXEL_PRESETS, strokesToMap, HOLDOUT_X, SMALL, LARGE, ALL_FAMILIES } from "../../../demos/src/pixelclass/presets";
import { trainingSet, regionScores, inRegion } from "../../../demos/src/pixelclass/pipeline";
import { featureStack, selectChannels } from "../../../demos/src/lib/featurestack";
import { FOREST_CLASSIFIER } from "../../../demos/src/lib/forest";
import { defaultParams, labelsAndUncertainty } from "../../../demos/src/lib/classifier";

const TMP = process.env.FIGTMP || os.tmpdir();
const size = 256, N = size * size;
const sc = synthScene(size, 1);
const full = featureStack(sc.image, ALL_FAMILIES, [...SMALL, ...LARGE]);
const out: Record<string, unknown> = {
  size, image: Array.from(sc.image.data, v => Math.round(v * 10) / 10), truth: Array.from(sc.truth),
  kinds: sc.kinds, ellipses: sc.ellipses, holdout: HOLDOUT_X,
};

// two channels for the "every pixel is a point" scatter
const ci = full.names.findIndex(n => /smooth|intensity|Gauss/i.test(n) && full.sigmas[full.names.indexOf(n)] === 5);
console.log("channels:", full.names.join(" | "));
const chan = (c: number) => Array.from(full.data.subarray(c * N, (c + 1) * N), v => Math.round(v * 100) / 100);
const texIdx = full.names.findIndex((n, c) => full.groups[c] === "texture" && full.sigmas[c] === 1.6);
const intIdx = full.names.findIndex((n, c) => full.groups[c] === "intensity" && full.sigmas[c] === 5);
console.log("scatter channels:", full.names[intIdx], "/", full.names[texIdx], ci);
out.chInt = chan(intIdx); out.chTex = chan(texIdx);
out.chIntName = full.names[intIdx]; out.chTexName = full.names[texIdx];

function run(id: string, o: { balance?: boolean } = {}) {
  const p = PIXEL_PRESETS.find(q => q.id === id)!;
  const labels = strokesToMap(p.strokes(sc), size, size);
  const S = selectChannels(full, (g, s) => p.families.includes(g) && (s <= 1.6 ? p.small : p.large));
  const exclude = p.heldout ? (q: number) => inRegion(q, size, HOLDOUT_X) : undefined;
  const balance = o.balance ?? p.balance;
  const info = trainingSet(S, labels, { exclude, cap: 4000, balance, seed: 1 })!;
  const model = FOREST_CLASSIFIER.train(info.data, defaultParams(FOREST_CLASSIFIER), 1);
  const k = info.data.k, P = new Float32Array(N * k);
  model.predict(S.data, N, P, 0, N);
  const { label, uncertainty } = labelsAndUncertainty(P, N, k);
  const region = p.heldout ? (q: number) => inRegion(q, size, HOLDOUT_X) : undefined;
  const r = regionScores(label, labels, sc.truth, 3, region);
  const all = regionScores(label, labels, sc.truth, 3);
  const oob = model.oobError?.();
  let frac = [0, 0, 0];
  for (let q = 0; q < N; q++) frac[label[q]]++;
  frac = frac.map(v => v / N);
  // pixels whose truth is nucleus/granular/background predicted as something else
  console.log(`${id}${o.balance !== undefined ? " balance=" + balance : ""}: painted ${info.painted} used ${info.used} strokes ${info.strokes}; OOB acc ${oob !== undefined && oob !== null ? (100 * (1 - oob)).toFixed(1) : "-"}; ` +
    `${p.heldout ? `held-out painted ${(100 * r.paintedAcc).toFixed(1)} of ${r.paintedN}; held-out truth ${(100 * r.truthAcc).toFixed(1)}; ` : ""}` +
    `whole-image acc ${(100 * all.truthAcc).toFixed(1)} Dice ${all.dice.map(v => v.toFixed(2)).join(" ")}; predicted fractions ${frac.map(v => (100 * v).toFixed(1)).join(" ")}`);
  return {
    labels: Array.from(labels), pred: Array.from(label), unc: Array.from(uncertainty, v => Math.round(v * 1000) / 1000),
    prob: Array.from({ length: N }, (_, q) => Math.round(P[q * k + 1] * 1000) / 1000), k,
    oobAcc: oob !== undefined && oob !== null ? 1 - oob : null, heldPainted: r.paintedAcc, heldN: r.paintedN, heldTruth: r.truthAcc,
    acc: all.truthAcc, dice: all.dice,
  };
}

out.full = run("full");
out.uncertain = run("uncertain");
out.bright = run("bright");
out.small = run("small");
out.imb = run("imbalance", { balance: false });
out.imbBal = run("imbalance", { balance: true });
out.heldout = run("heldout");
fs.writeFileSync(TMP + "/pixelclass.json", JSON.stringify(out));
console.log("wrote", TMP + "/pixelclass.json");
