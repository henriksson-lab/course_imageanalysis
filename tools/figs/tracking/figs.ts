// Static figures for the tracking day (track-linking/, track-predict/, track-events/,
// track-analysis/). Run from the repository root:
//   node_modules/.bin/esbuild tools/figs/tracking/figs.ts --bundle --platform=node --log-level=warning | node
// Uses the same libraries as the demos (lib/tracksim.ts, lib/kalman.ts, lib/lineage.ts), so the
// numbers in the figures match the units.
import { writeFileSync } from "fs";
import { simTracks } from "../../../demos/src/lib/tracksim";
import { layoutX, Branch } from "../../../demos/src/lib/lineage";

const BLUE = "#1f77b4", ORANGE = "#ff7f0e", GREEN = "#2a9d4b", RED = "#d62728", GREY = "#888", DARK = "#222", PURPLE = "#cc79a7", GOLD = "#e69f00";
const FONT = `font-family="system-ui, -apple-system, 'Segoe UI', sans-serif"`;

class Svg {
  parts: string[] = [];
  constructor(public w: number, public h: number) {}
  add(s: string) { this.parts.push(s); return this; }
  line(x1: number, y1: number, x2: number, y2: number, c = DARK, w = 3, dash = "") {
    return this.add(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${c}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""} stroke-linecap="round"/>`);
  }
  arrow(x1: number, y1: number, x2: number, y2: number, c = DARK, w = 3, dash = "") {
    const L = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / L, uy = (y2 - y1) / L, hl = 14, hw = 7;
    const bx = x2 - ux * hl, by = y2 - uy * hl;
    this.line(x1, y1, bx, by, c, w, dash);
    return this.add(`<polygon points="${x2.toFixed(1)},${y2.toFixed(1)} ${(bx - uy * hw).toFixed(1)},${(by + ux * hw).toFixed(1)} ${(bx + uy * hw).toFixed(1)},${(by - ux * hw).toFixed(1)}" fill="${c}"/>`);
  }
  circle(x: number, y: number, r: number, fill: string, stroke = "none", w = 0) {
    return this.add(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${w}"/>`);
  }
  ellipse(x: number, y: number, rx: number, ry: number, rot: number, c: string, fillOp = 0.1, dash = "") {
    return this.add(`<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}" transform="rotate(${rot} ${x.toFixed(1)} ${y.toFixed(1)})" fill="${c}" fill-opacity="${fillOp}" stroke="${c}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`);
  }
  text(x: number, y: number, s: string, size = 22, anchor = "middle", c = DARK, weight = "normal") {
    return this.add(`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${size}" text-anchor="${anchor}" fill="${c}" font-weight="${weight}" ${FONT}>${s}</text>`);
  }
  path(d: string, c = DARK, w = 3, fill = "none", dash = "") {
    return this.add(`<path d="${d}" stroke="${c}" stroke-width="${w}" fill="${fill}"${dash ? ` stroke-dasharray="${dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`);
  }
  save(file: string) {
    writeFileSync(file, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.w} ${this.h}" width="${this.w}" height="${this.h}">\n<rect width="100%" height="100%" fill="#fff"/>\n${this.parts.join("\n")}\n</svg>\n`);
    console.log("wrote", file);
  }
}
const disc = (s: Svg, x: number, y: number, c: string, filled: boolean) => s.circle(x, y, 11, filled ? c : "#fff", c, 4);
const poly = (pts: [number, number][]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");

// ---------------------------------------------------------------------------------------
// track-linking/fig-frames.svg: detections in 4 frames, then the same dots joined into tracks
{
  const s = new Svg(900, 330);
  const objs = [[[60, 80], [95, 95], [125, 90], [160, 110]], [[90, 230], [110, 200], [135, 180], [150, 150]], [[220, 150], [215, 175], [230, 200], [250, 220]]];
  const shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"];
  for (const panel of [0, 1]) {
    const ox = panel * 460 + 40;
    s.text(ox + 170, 32, panel === 0 ? "detections, frames 1–4" : "tracks: one chain per object", 22, "middle", DARK, "600");
    if (panel === 1) objs.forEach((o, k) => s.path(poly(o.map(([x, y]) => [ox + x, y + 20] as [number, number])), [GOLD, PURPLE, GREEN][k], 4));
    for (const o of objs) o.forEach(([x, y], t) => s.circle(ox + x, y + 20, 10, shades[t], "#fff", 1.5));
  }
  shades.forEach((c, t) => { s.circle(330 + t * 70, 315, 9, c); s.text(345 + t * 70, 322, `t${t + 1}`, 18, "start"); });
  s.save("track-linking/fig-frames.svg");
}

// track-linking/fig-nn-fail.svg: three ways nearest neighbour fails
{
  const s = new Svg(960, 300);
  const panels = ["conflict", "crossing", "a newcomer steals"];
  panels.forEach((t, k) => s.text(160 + k * 320, 34, t, 23, "middle", DARK, "600"));
  // conflict: A, B both nearest to P
  let ox = 0;
  disc(s, ox + 60, 150, BLUE, true); disc(s, ox + 180, 150, BLUE, true); disc(s, ox + 140, 150, ORANGE, false); disc(s, ox + 260, 150, ORANGE, false);
  s.arrow(ox + 72, 150, ox + 128, 150, RED, 4); s.add(`<path d="M${ox + 170},150 C${ox + 165},125 ${ox + 150},125 ${ox + 146},138" stroke="${RED}" stroke-width="4" fill="none"/>`);
  s.text(ox + 60, 125, "A", 22); s.text(ox + 180, 125, "B", 22); s.text(ox + 140, 185, "P", 22); s.text(ox + 260, 185, "Q", 22);
  s.text(ox + 160, 250, "A and B both pick P", 20);
  // crossing
  ox = 320;
  disc(s, ox + 70, 110, BLUE, true); disc(s, ox + 70, 190, BLUE, true);
  disc(s, ox + 250, 190, ORANGE, false); disc(s, ox + 250, 110, ORANGE, false);
  s.line(ox + 70, 110, ox + 250, 190, GREY, 2, "6 5"); s.line(ox + 70, 190, ox + 250, 110, GREY, 2, "6 5");
  disc(s, ox + 160, 140, BLUE, false); disc(s, ox + 160, 160, BLUE, false);
  s.arrow(ox + 82, 112, ox + 148, 136, GREEN, 3); s.arrow(ox + 82, 188, ox + 148, 162, GREEN, 3);
  s.text(ox + 160, 250, "close at the crossing:", 20); s.text(ox + 160, 275, "identities can swap", 20);
  // newcomer
  ox = 640;
  disc(s, ox + 70, 150, BLUE, true); disc(s, ox + 230, 150, ORANGE, false); disc(s, ox + 120, 190, ORANGE, false);
  s.line(ox + 82, 150, ox + 218, 150, GREY, 2, "6 5");
  s.arrow(ox + 79, 158, ox + 111, 183, RED, 4);
  s.text(ox + 230, 125, "true successor", 18); s.text(ox + 120, 228, "a new object", 18);
  s.save("track-linking/fig-nn-fail.svg");
}

// track-linking/fig-abpq.svg: greedy vs global on A(0,0), B(3,0) → P(2,0), Q(5,0)
{
  const s = new Svg(900, 300), X = (x: number) => 90 + x * 70;
  for (const [row, label, links, total] of [[0, "greedy: B–P first (1), then A–Q (25)", [[1, 0, RED], [0, 1, RED]], "total 26"], [1, "global: A–P (4) + B–Q (4)", [[0, 0, GREEN], [1, 1, GREEN]], "total 8"]] as [number, string, [number, number, string][], string][]) {
    const y = 95 + row * 140;
    s.text(40, y - 50, label, 22, "start", DARK, "600"); s.text(860, y - 50, total, 22, "end", row ? GREEN : RED, "600");
    const A = [0, 3], B = [2, 5];
    links.forEach(([i, j, c], k) => {
      const x1 = X(A[i]), x2 = X(B[j]), mid = (x1 + x2) / 2, h = (i === 0 && j === 1) ? 55 : 30;
      s.add(`<path d="M${x1},${y} Q${mid},${y + (k ? h : -h)} ${x2},${y}" stroke="${c}" stroke-width="4" fill="none"/>`);
    });
    A.forEach((x, i) => { disc(s, X(x), y, BLUE, true); s.text(X(x), y + 42, "AB"[i], 22); });
    B.forEach((x, j) => { disc(s, X(x), y, ORANGE, false); s.text(X(x), y + 42, "PQ"[j], 22); });
  }
  s.text(620, 160, "squared distances:", 20, "start"); s.text(620, 188, "A–P 4, A–Q 25", 20, "start"); s.text(620, 216, "B–P 1, B–Q 4", 20, "start");
  s.save("track-linking/fig-abpq.svg");
}

// ---------------------------------------------------------------------------------------
// track-predict/fig-kalman-1d.svg: prediction N(10, 4), measurement N(13, 1), estimate N(12.4, 0.8)
{
  const s = new Svg(1000, 400), X = (x: number) => 60 + (x - 4) * 70, Y = (p: number) => 340 - p * 480;
  const g = (x: number, m: number, v: number) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);
  s.line(X(4), 340, X(16), 340, DARK, 2);
  for (let x = 4; x <= 16; x += 2) { s.line(X(x), 340, X(x), 347, DARK, 2); s.text(X(x), 372, String(x), 20); }
  for (const [m, v, c] of [[10, 4, BLUE], [13, 1, ORANGE], [12.4, 0.8, DARK]] as [number, number, string][]) {
    const pts: [number, number][] = [];
    for (let x = 4; x <= 16; x += 0.05) pts.push([X(x), Y(g(x, m, v))]);
    s.path(poly(pts), c, c === DARK ? 4 : 3.5);
  }
  s.text(X(7.6), Y(g(7.6, 10, 4)) - 16, "prediction 10, P⁻ = 4", 20, "end", BLUE);
  s.text(X(14.1), Y(g(14.1, 13, 1)) - 10, "measurement 13, R = 1", 20, "start", ORANGE);
  s.text(X(12.0), Y(g(12.4, 12.4, 0.8)) + 4, "estimate 12.4, P = 0.8", 20, "end", DARK);
  s.line(X(10), 50, X(13), 50, GREY, 3); s.circle(X(10), 50, 5, BLUE); s.circle(X(13), 50, 5, ORANGE); s.circle(X(12.4), 50, 8, DARK);
  s.text(X(13.3), 57, "K = 4/(4 + 1) = 0.8 of the way", 20, "start");
  s.save("track-predict/fig-kalman-1d.svg");
}

// track-predict/fig-crossing.svg: last position vs prediction for two particles crossing
{
  const s = new Svg(920, 360);
  for (const panel of [0, 1]) {
    const ox = panel * 470;
    s.text(ox + 225, 34, panel ? "link to the predicted position" : "link to the last position", 22, "middle", DARK, "600");
    const A = [[40, 100], [110, 140], [180, 180]], B = [[40, 260], [110, 220], [180, 180 + 30]];
    const nextA: [number, number] = [250, 220], nextB: [number, number] = [250, 170];   // true next positions (they crossed)
    s.path(poly(A.map(p => [ox + p[0], p[1]] as [number, number])), BLUE, 3); s.path(poly(B.map(p => [ox + p[0], p[1]] as [number, number])), ORANGE, 3);
    A.forEach(p => s.circle(ox + p[0], p[1], 8, BLUE)); B.forEach(p => s.circle(ox + p[0], p[1], 8, ORANGE));
    disc(s, ox + nextA[0], nextA[1], BLUE, false); disc(s, ox + nextB[0], nextB[1], ORANGE, false);
    if (!panel) {
      s.add(`<circle cx="${ox + 180}" cy="180" r="75" fill="${BLUE}" fill-opacity="0.06" stroke="${BLUE}" stroke-width="2" stroke-dasharray="5 4"/>`);
      s.arrow(ox + 186, 176, ox + nextB[0] - 12, nextB[1], RED, 4); s.arrow(ox + 186, 214, ox + nextA[0] - 12, nextA[1] - 2, RED, 4);
      s.text(ox + 225, 320, "the nearer ring is the wrong one: swap", 20);
    } else {
      s.ellipse(ox + 250, 220, 36, 20, 30, BLUE); s.ellipse(ox + 250, 180, 36, 20, -20, ORANGE);
      s.circle(ox + 250, 220, 4, BLUE); s.circle(ox + 250, 180, 4, ORANGE);
      s.arrow(ox + 188, 184, ox + nextA[0] - 12, nextA[1] - 6, GREEN, 4); s.arrow(ox + 188, 208, ox + nextB[0] - 12, nextB[1] + 4, GREEN, 4);
      s.text(ox + 225, 320, "prediction ± uncertainty (ellipse): kept", 20);
    }
  }
  s.save("track-predict/fig-crossing.svg");
}

// ---------------------------------------------------------------------------------------
// track-events/fig-events.svg: what breaks a track (space–time sketches, time down)
{
  const s = new Svg(1000, 340);
  const titles = ["missed detection", "division", "segmentation split", "enters the field"];
  titles.forEach((t, k) => s.text(125 + k * 250, 32, t, 21, "middle", DARK, "600"));
  const T = (t: number) => 60 + t * 32;
  // gap
  let ox = 0;
  for (const t of [0, 1, 2, 4, 5, 6]) s.circle(ox + 125, T(t), 8, DARK);
  s.line(ox + 125, T(0), ox + 125, T(2), BLUE, 4); s.line(ox + 125, T(4), ox + 125, T(6), PURPLE, 4);
  s.circle(ox + 125, T(3), 8, "#fff", GREY, 2); s.line(ox + 140, T(2), ox + 140, T(4), GREY, 2, "5 4");
  s.text(ox + 125, T(7) + 20, "two tracks → close the gap", 18);
  // division
  ox = 250;
  for (const t of [0, 1, 2, 3]) s.circle(ox + 125, T(t), t === 3 ? 11 : 9, DARK);
  s.line(ox + 125, T(0), ox + 125, T(3), BLUE, 4);
  for (const t of [4, 5, 6]) { s.circle(ox + 95 - (t - 4) * 6, T(t), 7, DARK); s.circle(ox + 155 + (t - 4) * 6, T(t), 7, DARK); }
  s.line(ox + 125, T(3), ox + 95, T(4), GOLD, 4); s.line(ox + 125, T(3), ox + 155, T(4), GOLD, 4);
  s.line(ox + 95, T(4), ox + 83, T(6), BLUE, 4); s.line(ox + 155, T(4), ox + 167, T(6), BLUE, 4);
  s.text(ox + 125, T(7) + 20, "mother + two daughters", 18);
  // segmentation split
  ox = 500;
  for (const t of [0, 1, 2, 4, 5, 6]) s.circle(ox + 125, T(t), 9, DARK);
  s.circle(ox + 112, T(3), 7, DARK); s.circle(ox + 142, T(3), 5, DARK);
  s.line(ox + 125, T(0), ox + 125, T(2), BLUE, 4); s.line(ox + 125, T(2), ox + 112, T(3), BLUE, 4); s.line(ox + 112, T(3), ox + 125, T(4), BLUE, 4); s.line(ox + 125, T(4), ox + 125, T(6), BLUE, 4);
  s.line(ox + 125, T(2), ox + 142, T(3), GOLD, 4); s.line(ox + 142, T(3), ox + 125, T(4), PURPLE, 4);
  s.text(ox + 125, T(7) + 20, "split then merge: a bubble", 18);
  // enters
  ox = 750;
  s.add(`<rect x="${ox + 60}" y="${T(0) - 14}" width="150" height="${T(6) - T(0) + 28}" fill="#f4f6fa" stroke="${GREY}" stroke-width="2"/>`);
  s.text(ox + 70, T(0) + 6, "field", 16, "start", GREY);
  for (const t of [2, 3, 4, 5, 6]) s.circle(ox + 60 + (t - 2) * 22, T(t), 8, DARK);
  s.line(ox + 60, T(2), ox + 148, T(6), BLUE, 4);
  s.circle(ox + 38, T(1), 8, "#fff", GREY, 2);
  s.text(ox + 135, T(7) + 20, "starts at the border:", 18); s.text(ox + 135, T(7) + 42, "not a birth", 18);
  s.save("track-events/fig-events.svg");
}

// track-events/fig-lineage.svg: the lineage tree of exercise (c), times in hours, movie 0–48 h
{
  const br: Branch[] = [
    { id: 0, parent: -1, t0: 0, t1: 10, children: [1, 2], label: "F" },
    { id: 1, parent: 0, t0: 10, t1: 28, children: [3, 4], label: "A" },
    { id: 2, parent: 0, t0: 10, t1: 34, children: [5, 6], label: "B" },
    { id: 3, parent: 1, t0: 28, t1: 46, children: [7, 8], label: "C" },
    { id: 4, parent: 1, t0: 28, t1: 48, children: [], label: "D" },
    { id: 5, parent: 2, t0: 34, t1: 48, children: [], label: "E" },
    { id: 6, parent: 2, t0: 34, t1: 48, children: [], label: "G" },
    { id: 7, parent: 3, t0: 46, t1: 48, children: [], label: "H" },
    { id: 8, parent: 3, t0: 46, t1: 48, children: [], label: "I" },
  ];
  const x = layoutX(br, 0), s = new Svg(760, 540);
  const X = (v: number) => 150 + v * 95, Y = (t: number) => 50 + t * 9;
  s.line(70, Y(0), 70, Y(48), DARK, 2);
  for (let t = 0; t <= 48; t += 8) { s.line(64, Y(t), 70, Y(t), DARK, 2); s.text(58, Y(t) + 7, `${t} h`, 18, "end"); }
  s.add(`<line x1="80" x2="740" y1="${Y(48)}" y2="${Y(48)}" stroke="${GREY}" stroke-width="2" stroke-dasharray="6 5"/>`);
  s.text(740, Y(48) + 26, "end of movie (48 h)", 18, "end", GREY);
  for (const b of br) {
    const X0 = X(x.get(b.id)!);
    s.line(X0, Y(b.t0), X0, Y(b.t1), BLUE, 4);
    if (b.children.length) { const xs = b.children.map(c => X(x.get(c)!)); s.line(Math.min(...xs), Y(b.t1), Math.max(...xs), Y(b.t1), BLUE, 4); s.circle(X0, Y(b.t1), 6, BLUE); }
    if (b.t1 - b.t0 >= 4) s.text(X0 + 10, (Y(b.t0) + Y(b.t1)) / 2 + 7, b.label!, 20, "start", DARK, "600");
    else s.text(X0, Y(48) + 26, b.label!, 20, "middle", DARK, "600");
  }
  s.save("track-events/fig-lineage.svg");
}

// ---------------------------------------------------------------------------------------
// track-analysis/fig-tracks.svg: three kinds of motion (simulated), same D
{
  const s = new Svg(960, 340);
  const kinds: [string, "free" | "directed" | "confined"][] = [["free diffusion", "free"], ["directed + diffusion", "directed"], ["confined", "confined"]];
  kinds.forEach(([t, type], k) => {
    const [tr] = simTracks({ type, n: 1, N: 200, D: 0.05, v: 0.3, R: 0.6, alpha: 1, dt: 0.1, sigma: 0, seed: 11 + k });
    const ox = 20 + k * 320, cx = ox + 150, cy = 190, sc = 75;
    s.text(cx, 34, t, 22, "middle", DARK, "600");
    const x0 = tr.x[0], y0 = tr.y[0];
    let mx = 0;
    for (let i = 0; i < tr.x.length; i++) mx = Math.max(mx, Math.abs(tr.x[i] - x0), Math.abs(tr.y[i] - y0));
    const f = Math.min(sc, 120 / mx);
    if (type === "confined") s.add(`<circle cx="${(cx - x0 * f).toFixed(1)}" cy="${(cy + y0 * f).toFixed(1)}" r="${(0.6 * f).toFixed(1)}" fill="none" stroke="${GREY}" stroke-dasharray="5 4" stroke-width="2"/>`);
    s.path(poly(tr.x.map((v, i) => [cx + (v - x0) * f, cy - (tr.y[i] - y0) * f] as [number, number])), [BLUE, ORANGE, GREEN][k], 1.8);
    s.circle(cx, cy, 6, DARK);
    s.line(ox + 40, 320, ox + 40 + 0.5 * f, 320, DARK, 3); s.text(ox + 48 + 0.5 * f, 326, "0.5 µm", 18, "start");
  });
  s.save("track-analysis/fig-tracks.svg");
}

// track-analysis/fig-msd-shapes.svg: MSD of three particles (2D, Δt = 0.1 s), for exercise (b)
{
  const s = new Svg(760, 480), X = (t: number) => 90 + t * 60, Y = (m: number) => 420 - m * 36;
  s.line(X(0), Y(0), X(10.5), Y(0), DARK, 2); s.line(X(0), Y(0), X(0), Y(10.5), DARK, 2);
  for (let t = 0; t <= 10; t += 2) { s.line(X(t), Y(0), X(t), Y(0) + 7, DARK, 2); s.text(X(t), Y(0) + 32, String(t), 20); }
  for (let m = 0; m <= 10; m += 2) { s.line(X(0) - 7, Y(m), X(0), Y(m), DARK, 2); s.text(X(0) - 12, Y(m) + 7, String(m), 20, "end"); s.line(X(0), Y(m), X(10.5), Y(m), "#e3e3e3", 1); }
  for (let t = 0; t <= 10; t += 1) s.line(X(t), Y(0), X(t), Y(10.5), "#eeeeee", 1);
  s.text(X(5.25), Y(0) + 62, "lag τ (s)", 21); s.add(`<text x="30" y="${Y(5)}" font-size="21" text-anchor="middle" transform="rotate(-90 30 ${Y(5)})" ${FONT}>MSD (µm²)</text>`);
  const curves: [string, string, (t: number) => number][] = [
    ["1", BLUE, t => 0.8 * t],
    ["2", ORANGE, t => 0.4 * t + 0.09 * t * t],
    ["3", GREEN, t => 2.25 * (1 - Math.exp(-0.8 * t / 2.25))],
  ];
  for (const [lab, c, fn] of curves) {
    const pts: [number, number][] = [];
    for (let t = 0; t <= 10.0001; t += 0.1) { const v = fn(t); if (v <= 10.5) pts.push([X(t), Y(v)]); }
    s.path(poly(pts), c, 4);
    const [lx, ly] = pts[pts.length - 1];
    s.text(lx + 14, ly + 7, lab, 24, "start", c, "700");
  }
  s.save("track-analysis/fig-msd-shapes.svg");
}
