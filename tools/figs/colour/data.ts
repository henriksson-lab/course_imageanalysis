// Data for the colour day's static figures, computed with the course's own demo code so
// that every number in a figure matches the demos. Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/colour/data.ts --bundle --platform=node --outfile=<tmp>/data.js
//   node <tmp>/data.js <tmp>/colour-data.json
//   python3 -I tools/figs/colour/figs.py <tmp>/colour-data.json
// Writes JSON only; figs.py draws the figures into colour-vision/, rgb-linear/,
// colour-perceptual/ and colour-display/.
import { writeFileSync } from "fs";
import * as SP from "../../../demos/src/lib/spectra";
import { makeScene, withNoise, objectsScene, heScene, isolumScene } from "../../../demos/src/colourcloud/scenes";
import { linearToSrgb } from "../../../demos/src/lib/colour";
import * as SG from "../../../demos/src/colourseg/seg";
import { makeImage, process as blurProcess } from "../../../demos/src/gammablur/gamma";
import { runWB } from "../../../demos/src/whitebalance/wb";
import { DICHROMAT, jet, cividis } from "../../../demos/src/colourcloud/colourx";
import { viridis, diverging, QUALITATIVE } from "../../../demos/src/lib/colour";

const A = (a: ArrayLike<number>) => Array.from(a, v => +(+v).toFixed(6));
const out: Record<string, unknown> = {};

// ---- colour-vision -------------------------------------------------------------------
const wlFine = Array.from({ length: 351 }, (_, i) => 380 + i);
out.wlFine = wlFine;
out.cmf = wlFine.map(l => SP.cmf(l));
out.cones = wlFine.map(l => SP.cones(l));
out.camera = wlFine.map(l => SP.camera(l));
out.wl = A(SP.WL);
out.d65 = A(SP.D65_SPD);
out.tungsten2800 = A(SP.blackbody(2800));
out.whiteLED = A(SP.whiteLED());
out.metamericLED = A(SP.metamericLED());
out.resp = {
  d65_lms: SP.response("lms", SP.D65_SPD), led_lms: SP.response("lms", SP.metamericLED()),
  d65_cam: SP.response("camera", SP.D65_SPD), led_cam: SP.response("camera", SP.metamericLED()),
};
out.locus = SP.spectralLocus(380, 700, 1);
out.displays = [SP.SRGB_DISPLAY, SP.P3_DISPLAY, SP.REC2020_DISPLAY];
out.srgbMatrix = SP.rgbToXyzMatrix(SP.SRGB_DISPLAY).M;
out.p3Matrix = SP.rgbToXyzMatrix(SP.P3_DISPLAY).M;

// ---- rgb-linear ----------------------------------------------------------------------
{
  const I = makeImage("checker"), r = blurProcess(I, "blur", 3);
  out.seam = { input: I.map(A), naive: r.naive.map(A), linear: r.linear.map(A), n: 128 };
}
{
  const base = { kind: "chart" as const, illum: "blackbody" as const, T: 2800, obs: "eye" as const, space: "lms" as const, card: 20, highlight: false };
  const none = runWB({ ...base, how: "none" }), card = runWB({ ...base, how: "card" });
  const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
  out.wb = { captured: none.captured, corrected: card.corrected, reference: card.reference, dENone: mean(none.dE), dECard: mean(card.dE) };
  const f = { kind: "forest" as const, illum: "blackbody" as const, T: 4000, obs: "eye" as const, space: "lms" as const, card: 23, highlight: false };
  const gw = runWB({ ...f, how: "greyworld" }), wp = runWB({ ...f, how: "whitepatch" }), fn = runWB({ ...f, how: "none" });
  out.wbForest = { captured: fn.captured, greyworld: gw.corrected, whitepatch: wp.corrected, reference: gw.reference, dEgw: mean(gw.dE), dEwp: mean(wp.dE) };
}
const sceneOut = (s: { w: number; h: number; r: Float32Array; g: Float32Array; b: Float32Array; truth?: Uint8Array }) =>
  ({ w: s.w, h: s.h, r: A(s.r), g: A(s.g), b: A(s.b), truth: s.truth ? Array.from(s.truth) : null });
out.objects160 = sceneOut(objectsScene(160));
out.isolum = sceneOut(isolumScene(160));

// ---- colour-perceptual ---------------------------------------------------------------
{
  const sc = withNoise(makeScene("objects", 128), 0.01), n = sc.w * sc.h;
  const enc = (a: Float32Array) => Float32Array.from(a, v => linearToSrgb(Math.min(1, Math.max(0, v))));
  const px = { r: enc(sc.r), g: enc(sc.g), b: enc(sc.b), n }, dv = SG.derive(px);
  const mR = SG.rgbRule(px, 120, 90, 90), mH = SG.hueRule(dv, n, 0, 20, 0.4, 0.1);
  out.seg = {
    w: sc.w, h: sc.h, r: A(px.r), g: A(px.g), b: A(px.b), lin: [A(sc.r), A(sc.g), A(sc.b)], truth: Array.from(sc.truth!),
    hsv: A(dv.hsv.map(v => (Number.isFinite(v) ? v : -1))), lab: A(dv.lab),
    maskR: Array.from(mR), maskH: Array.from(mH),
    diceR: SG.diceClass(mR, sc.truth!, 1), diceH: SG.diceClass(mH, sc.truth!, 1),
  };
  const labs = dv.lab, rgbs = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) { rgbs[3 * p] = px.r[p]; rgbs[3 * p + 1] = px.g[p]; rgbs[3 * p + 2] = px.b[p]; }
  const kR = SG.kmeans(rgbs, n, 4, 100), kL = SG.kmeans(labs, n, 4, 100);
  const best = (lab: Int32Array) => { let b = 0, bj = 0; for (let j = 0; j < 4; j++) { const m = Uint8Array.from(lab, v => (v === j ? 1 : 0)); const d = SG.diceClass(m, sc.truth!, 1); if (d > b) { b = d; bj = j; } } return { dice: b, cls: bj }; };
  out.kmeans = { rgb: Array.from(kR.labels), lab: Array.from(kL.labels), rgbBest: best(kR.labels), labBest: best(kL.labels), centresRgb: kR.centres, centresLab: kL.centres };
}

// ---- colour-display ------------------------------------------------------------------
out.he = sceneOut(heScene(160));
out.dichromat = DICHROMAT;
const ts = Array.from({ length: 256 }, (_, i) => i / 255);
out.maps = { jet: ts.map(t => jet(t)), viridis: ts.map(t => viridis(t)), cividis: ts.map(t => cividis(t)), diverging: ts.map(t => diverging(t)) };
out.qualitative = QUALITATIVE;

writeFileSync(process.argv[2] ?? "colour-data.json", JSON.stringify(out));
console.log("wrote", process.argv[2]);
