// Slide figures for spot-fit/ (second image pass). Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/spot-fit/figs.ts --bundle --platform=node --log-level=warning | node
// Writes spot-fit/fig-subpixel.svg, fig-centroid.svg, fig-residual.svg, fig-photons.svg; prints the numbers.
import { pixelGaussian, poissonImage, fitSpots } from "../../../demos/src/lib/spotfit";
import { rng } from "../../../demos/src/lib/ui";
import { C, svg, text, line, circle, rect, path, cross, axes, greyImage, out } from "../pointdetection/svgkit";

// ---------------------------------------------------------------- fig-subpixel.svg
{
  // point samples of a Gaussian of σ = 1 px, peak 1, centred on a pixel and 0.3 px to the right (the deck's numbers)
  const n = 5, c = 2, px = 46;
  const patch = (dx: number) => Float64Array.from({ length: n * n }, (_, q) => Math.exp(-(((q % n) - c - dx) ** 2 + (Math.floor(q / n) - c) ** 2) / 2));
  let b = "";
  [0, 0.3].forEach((dx, k) => {
    const X0 = 20 + k * (n * px + 50), Y0 = 40, P = patch(dx);
    b += greyImage(P, n, 0, 0, n, n, X0, Y0, px, 0, 1);
    for (let i = 0; i < n; i++) b += line(X0 + i * px, Y0, X0 + i * px, Y0 + n * px, "#666", 0.8) + line(X0, Y0 + i * px, X0 + n * px, Y0 + i * px, "#666", 0.8);
    b += text(X0 + n * px / 2, 28, dx ? "centre 0.3 px to the right" : "centre on a pixel", { weight: "bold" });
    // true centre
    b += cross(X0 + (c + 0.5 + dx) * px, Y0 + (c + 0.5) * px, 7, C.orange, 3.5);
    // central row values
    for (let i = 1; i <= 3; i++) {
      const v = P[c * n + i];
      b += text(X0 + (i + 0.5) * px, Y0 + n * px + 30, v.toFixed(2), { size: 20, weight: i !== 2 ? "bold" : undefined, fill: i === 2 ? C.fg : dx && i === 3 ? C.blue : dx && i === 1 ? C.fp : C.fg });
    }
    b += text(X0 + n * px / 2, Y0 + n * px + 56, dx ? "right neighbour brighter" : "neighbours equal", { size: 20 });
    console.log("subpixel", dx, Array.from(P.slice(c * n, c * n + n)).map(v => v.toFixed(3)).join(" "));
  });
  b += cross(40, 368, 7, C.orange, 3.5) + text(56, 375, "true centre; numbers: the middle row (σ = 1 px, peak 1)", { anchor: "start", size: 20 });
  out("spot-fit/fig-subpixel.svg", svg(2 * (n * px) + 90, 390, b));
}

// ---------------------------------------------------------------- fig-centroid.svg (worked example)
{
  const v = [-1, 0, 1].map(x => Math.exp(-((x - 0.3) ** 2) / 2)), sum = v[0] + v[1] + v[2];
  const cen = (-v[0] + v[2]) / sum, cenB = (-v[0] + v[2]) / (sum + 1.5);
  console.log("centroid samples", v.map(x => x.toFixed(3)).join(" "), "centroid", cen.toFixed(3), "with b=0.5", cenB.toFixed(3));
  const ax = axes(75, 100, 440, 250, [-1.6, 1.6], [0, 1.6], [-1, 0, 1], [0, 0.5, 1, 1.5], "pixel position", "value", v => String(v), false, false, 20);
  let b = ax.s;
  // labels stacked: the leftmost line has the top label, so no label crosses a line
  const mark = (x: number, col: string, lab: string, row: number, dash = "") => line(ax.X(x), 30 + row * 26, ax.X(x), 350, col, 3, dash) + text(ax.X(x) - 4, 22 + row * 26, lab, { size: 20, fill: col, anchor: "start" });
  b += mark(cenB, "#a86b00", `with background ${cenB.toFixed(2)}`, 0) + mark(cen, C.blue, `centroid ${cen.toFixed(2)}`, 1) + mark(0.3, C.fg, "true 0.30", 2, "6 4");
  const bw = 76;
  v.forEach((y, i) => {
    const x = ax.X(i - 1);
    b += rect(x - bw / 2, ax.Y(y), bw, ax.Y(0) - ax.Y(y), { fill: C.blue });
    b += rect(x - bw / 2, ax.Y(y + 0.5), bw, ax.Y(y) - ax.Y(y + 0.5), { fill: C.orange, opacity: 0.45, stroke: C.orange, w: 1.5, dash: "5 3" });
    b += text(x, ax.Y(y / 2) + 7, y.toFixed(3), { size: 20, fill: "#fff", weight: "bold" });
  });
  for (const [x, col] of [[cenB, "#a86b00"], [cen, C.blue], [0.3, C.fg]] as [number, string][]) b += path([[ax.X(x) - 7, 362], [ax.X(x), 352], [ax.X(x) + 7, 362]], { stroke: col, fill: col, w: 1 });
  b += rect(80, 438, 22, 16, { fill: C.blue }) + text(110, 452, "samples", { size: 20, anchor: "start" });
  b += rect(230, 438, 22, 16, { fill: C.orange, opacity: 0.45, stroke: C.orange, w: 1.5, dash: "5 3" }) + text(260, 452, "+ background 0.5 each", { size: 20, anchor: "start" });
  out("spot-fit/fig-centroid.svg", svg(540, 468, b));
}

// ---------------------------------------------------------------- fig-residual.svg
{
  const W = 15, c = 7, s = 1.5, Nn = 1000, B = 5, px = 11;
  const rows: [string, Float64Array][] = [
    ["one spot", pixelGaussian(c + 0.3, c - 0.2, s, Nn, B, W)],
    ["two spots, d = 3 px", (() => { const a = pixelGaussian(c - 1.5, c, s, Nn, B, W), b2 = pixelGaussian(c + 1.5, c, s, Nn, 0, W); return a.map((v, i) => v + b2[i]); })()],
  ];
  let b = "";
  const heads = ["data", "model (1 spot)", "data − model"];
  heads.forEach((h, k) => b += text(20 + k * (W * px + 26) + W * px / 2, 26, h, { weight: "bold", size: 22 }));
  rows.forEach(([lab, mu], r) => {
    const img = poissonImage(mu, 31 + r), Y0 = 44 + r * (W * px + 54);
    let lo = Infinity; for (const v of img) lo = Math.min(lo, v);
    let pk = 0, pi = 0; img.forEach((v, i) => { if (v > pk) { pk = v; pi = i; } });
    const f = fitSpots(img, W, W, s, [{ x: pi % W, y: Math.floor(pi / W), N: Nn }], B);
    const res = img.map((v, i) => v - f.model[i]);
    let hiI = 0; for (const v of img) hiI = Math.max(hiI, v);
    const X = (k: number) => 20 + k * (W * px + 26);
    b += greyImage(img, W, 0, 0, W, W, X(0), Y0, px, 0, hiI) + greyImage(f.model, W, 0, 0, W, W, X(1), Y0, px, 0, hiI) + greyImage(res, W, 0, 0, W, W, X(2), Y0, px, -40, 40);
    b += circle(X(1) + (f.spots[0].x + 0.5) * px, Y0 + (f.spots[0].y + 0.5) * px, s * px, { stroke: C.tp, w: 2.5 });
    const noiseSd = Math.sqrt(f.rss / (W * W));
    b += text(X(0) + W * px / 2, Y0 + W * px + 26, lab, { size: 21 });
    b += text(X(2) + W * px / 2, Y0 + W * px + 26, r ? "structure left" : "looks like noise", { size: 21, fill: r ? C.fp : C.fg, weight: r ? "bold" : undefined });
    console.log("residual", lab, "rss", f.rss.toFixed(0), "rms", noiseSd.toFixed(2), "fit x", f.spots[0].x.toFixed(2));
  });
  b += text(20 + 2 * (W * px + 26) + W * px / 2, 44 + 2 * (W * px + 54) - 4, "scale ±40 photons", { size: 21 });
  out("spot-fit/fig-residual.svg", svg(20 + 3 * (W * px + 26), 44 + 2 * (W * px + 54) + 6, b));
}

// ---------------------------------------------------------------- fig-photons.svg
{
  const s = 100, reps = 200, Ns = [100, 1000], r = rng(2026);
  let b = "";
  Ns.forEach((N, k) => {
    const X0 = 70 + k * 260, cx = X0 + 105;
    // one repeat: the photon positions
    const P: [number, number][] = []; for (let i = 0; i < N; i++) P.push([s * r.norm(), s * r.norm()]);
    const Yt = 50, R = 300, sc = 105 / R;
    b += rect(X0, Yt, 210, 210, { stroke: "#999", w: 1 });
    for (const [x, y] of P) if (Math.abs(x) < R && Math.abs(y) < R) b += `<circle cx="${(cx + x * sc).toFixed(1)}" cy="${(Yt + 105 + y * sc).toFixed(1)}" r="${N > 500 ? 1.2 : 2}" fill="${C.blue}" fill-opacity="0.6"/>`;
    b += cross(cx, Yt + 105, 7, C.orange, 3);
    b += text(cx, 26, `N = ${N} photons`, { weight: "bold" });
    // the estimate (mean) over many repeats, zoomed
    const means: [number, number][] = [];
    for (let t = 0; t < reps; t++) { let mx = 0, my = 0; for (let i = 0; i < N; i++) { mx += s * r.norm(); my += s * r.norm(); } means.push([mx / N, my / N]); }
    const rms = Math.sqrt(means.reduce((a, [x, y]) => a + x * x + y * y, 0) / reps / 2);
    const Yb = 330, Rz = 30, sz = 105 / Rz;
    b += rect(X0, Yb, 210, 210, { stroke: "#999", w: 1 });
    for (const [x, y] of means) if (Math.abs(x) < Rz && Math.abs(y) < Rz) b += `<circle cx="${(cx + x * sz).toFixed(1)}" cy="${(Yb + 105 + y * sz).toFixed(1)}" r="2.6" fill="${C.purple}" fill-opacity="0.7"/>`;
    b += circle(cx, Yb + 105, (s / Math.sqrt(N)) * sz, { stroke: C.fg, w: 2.5, dash: "6 4" });
    b += cross(cx, Yb + 105, 7, C.orange, 3);
    b += text(cx, Yb + 238, `sd ${rms.toFixed(1)} nm (s/√N: ${(s / Math.sqrt(N)).toFixed(1)})`, { size: 22 });
    console.log("photons N", N, "rms per axis of the mean", rms.toFixed(2));
  });
  // scale bars and row labels
  b += line(70, 272, 70 + 105 * 100 / 300, 272, C.fg, 3) + text(70 + 105 * 100 / 300 + 6, 279, "100 nm", { size: 22, anchor: "start" });
  b += `<text x="30" y="155" font-size="22" text-anchor="middle" transform="rotate(-90 30 155)">photons (one spot)</text>`;
  b += `<text x="30" y="435" font-size="22" text-anchor="middle" transform="rotate(-90 30 435)">mean of 200 repeats</text>`;
  b += text(305, 314, "zoom ±30 nm; dashed: radius s/√N", { size: 22 });
  out("spot-fit/fig-photons.svg", svg(560, 590, b));
}
