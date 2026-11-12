// Reference checks for the watershed day (lib/watershed.ts and the demos flood1d, floodgrid,
// watershed, distws). Brute force or a second route for every routine, on small random
// inputs. Run:
//   node_modules/.bin/esbuild demos/test/watershed-test.ts --bundle --platform=node --log-level=warning | node
import * as WS from "../src/lib/watershed";
import * as MO from "../src/lib/morph";
import { label2d } from "../src/lib/label";
import { rng } from "../src/lib/random";
import { basinsAtLevel, dams, derivative1d } from "../src/flood1d/f1d";
import { neckDepth, twoDiscMask } from "../src/distws/geom";

declare const process: { exit(code: number): never };

let failures = 0, checks = 0, current = "";
const section: Record<string, number> = {};
function check(ok: boolean, msg: string): void {
  checks++;
  section[current] = (section[current] ?? 0) + 1;
  if (!ok) { failures++; if (failures <= 40) console.log(`FAIL [${current}]: ${msg}`); }
}
const R = rng(1010);
const randInt = (n: number, lo: number, hi: number) => Int32Array.from({ length: n }, () => lo + Math.floor(R.unif() * (hi - lo + 1)));
const same = (a: ArrayLike<number>, b: ArrayLike<number>) => a.length === b.length && Array.from(a).every((v, i) => v === b[i]);

/** Random markers: k distinct pixels with labels 1..k. */
function randMarkers(n: number, k: number): Int32Array {
  const m = new Int32Array(n);
  for (let l = 1; l <= k; l++) { let p: number; do p = Math.floor(R.unif() * n); while (m[p]); m[p] = l; }
  return m;
}

/** Is the set of pixels with label l connected (conn), and how many marker pixels does it hold? */
function regionChecks(lab: Int32Array, markers: Int32Array, w: number, h: number, conn: 4 | 8): boolean {
  const k = Math.max(0, ...markers);
  for (let l = 1; l <= k; l++) {
    const m = Uint8Array.from(lab, v => (v === l ? 1 : 0));
    const comps = label2d(m, w, h, conn).count;
    const mk = new Set<number>();
    for (let i = 0; i < lab.length; i++) if (lab[i] === l && markers[i]) mk.add(markers[i]);
    if (comps !== 1 || mk.size !== 1 || !mk.has(l)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------
current = "trace (FIFO) = fast version";
for (let t = 0; t < 150; t++) {
  const w = 2 + Math.floor(R.unif() * 9), h = 1 + Math.floor(R.unif() * 9), n = w * h;
  const f = randInt(n, 0, 1 + Math.floor(R.unif() * 6));
  const mk = randMarkers(n, 1 + Math.floor(R.unif() * Math.min(4, n - 1)));
  for (const conn of [4, 8] as const) for (const lines of [false, true]) {
    const a = WS.watershed(f, w, h, mk, { conn, lines }), b = WS.watershedTrace(f, w, h, mk, { conn, lines });
    check(same(a, b.labels), `w=${w} h=${h} conn=${conn} lines=${lines}`);
    if (!lines) check(b.steps.length === n - mk.filter(v => v > 0).length, "every non-marker pixel popped once");
  }
}

current = "regions: connected, one marker each, all labelled without lines";
for (let t = 0; t < 150; t++) {
  const w = 3 + Math.floor(R.unif() * 10), h = 2 + Math.floor(R.unif() * 10), n = w * h;
  const f = randInt(n, 0, 9), mk = randMarkers(n, 1 + Math.floor(R.unif() * 5));
  for (const conn of [4, 8] as const) {
    const a = WS.watershed(f, w, h, mk, { conn });
    check(a.every(v => v > 0), "no-lines version labels every pixel");
    check(regionChecks(a, mk, w, h, conn), `connected regions, conn=${conn}`);
    const b = WS.watershed(f, w, h, mk, { conn, lines: true });
    check(regionChecks(b, mk, w, h, conn), `connected regions with lines, conn=${conn}`);
    // a line pixel has two different labels among its neighbours (in the final result)
    // a popped line pixel touches two labels; a pixel that is never popped is enclosed by lines
    let okLines = true;
    const off = WS.wsOffsets(conn);
    const popped = new Set(WS.watershedTrace(f, w, h, mk, { conn, lines: true }).steps.map(s => s.item.p));
    for (let p = 0; p < n; p++) {
      if (b[p] !== 0 || !popped.has(p)) continue;
      const x = p % w, y = (p - x) / w, s = new Set<number>();
      for (const [dx, dy] of off) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h && b[yy * w + xx] > 0) s.add(b[yy * w + xx]); }
      if (s.size < 2) okLines = false;
    }
    check(okLines, "every line pixel touches two labels");
  }
}

current = "distinct heights: result independent of the scan order";
for (let t = 0; t < 100; t++) {
  const w = 3 + Math.floor(R.unif() * 8), h = 3 + Math.floor(R.unif() * 8), n = w * h;
  // a random permutation of 0..n-1 (all heights distinct), markers at random pixels
  const f = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(R.unif() * (i + 1)); const tmp = f[i]; f[i] = f[j]; f[j] = tmp; }
  // markers at least 3 apart (no pixel touches two markers: that tie is decided by the scan order)
  const mk = new Int32Array(n), placed: number[] = [];
  for (let tries = 0; placed.length < 3 && tries < 200; tries++) {
    const p = Math.floor(R.unif() * n), x = p % w, y = (p - x) / w;
    if (placed.every(q => Math.max(Math.abs(q % w - x), Math.abs(Math.floor(q / w) - y)) >= 3)) { placed.push(p); mk[p] = placed.length; }
  }
  const flip = (a: ArrayLike<number>) => Int32Array.from({ length: n }, (_, i) => { const x = i % w, y = (i - x) / w; return a[(h - 1 - y) * w + (w - 1 - x)]; });
  for (const conn of [4, 8] as const) for (const lines of [false, true]) {
    const a = WS.watershed(f, w, h, mk, { conn, lines }), b = flip(WS.watershed(flip(f), w, h, flip(mk), { conn, lines }));
    check(same(a, b), `rotated by 180°, conn=${conn} lines=${lines}`);
  }
}

current = "plateaus: FIFO puts the dam in the middle";
for (let L = 3; L <= 15; L++) {
  // 1D: marker A at 0, marker B at L-1, flat plateau between them
  const f = new Int32Array(L).fill(5), mk = new Int32Array(L);
  mk[0] = 1; mk[L - 1] = 2;
  const a = WS.watershed(f, L, 1, mk, { lines: true });
  const lines = [...a].map((v, i) => (v === 0 ? i : -1)).filter(i => i >= 0);
  check(lines.length === 1 && Math.abs(lines[0] - (L - 1) / 2) <= 0.5, `plateau ${L}: one line pixel in the middle (${lines})`);
  const b = WS.watershed(f, L, 1, mk);
  const nA = b.filter(v => v === 1).length;
  check(Math.abs(2 * nA - L) <= 1, `plateau ${L} without lines: split in half (${nA})`);
  const lifo = WS.watershedTrace(f, L, 1, mk, { lines: true, lifo: true }).labels;
  check(lifo.length === L, "LIFO runs");
}

current = "1D: dams at the passes between minima";
for (let t = 0; t < 200; t++) {
  const n = 5 + Math.floor(R.unif() * 30), f = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(R.unif() * (i + 1)); const tmp = f[i]; f[i] = f[j]; f[j] = tmp; }
  const mk = WS.minimaMarkers(f, n, 1, 4);
  const lab = WS.watershed(f, n, 1, mk.data);
  // between consecutive minima the label changes once, next to the highest sample between them
  const mins: number[] = [];
  for (let i = 0; i < n; i++) if (mk.data[i]) mins.push(i);
  let ok = true;
  for (let k = 0; k + 1 < mins.length; k++) {
    const a = mins[k], b = mins[k + 1];
    let changes = 0, at = -1, top = a;
    for (let i = a; i < b; i++) if (lab[i] !== lab[i + 1]) { changes++; at = i; }
    for (let i = a; i <= b; i++) if (f[i] > f[top]) top = i;
    if (changes !== 1 || (at !== top && at + 1 !== top)) ok = false;
  }
  check(ok, `n=${n}: one dam per pass, at the highest sample`);
}

current = "h-minima in 1D = brute-force fill by h";
for (let t = 0; t < 200; t++) {
  const n = 4 + Math.floor(R.unif() * 25), f = randInt(n, 0, 9), hh = 1 + Math.floor(R.unif() * 4);
  // HMIN_h(f)(x) = min over j of max(f(j) + h, max of f between j and x)
  const brute = Array.from({ length: n }, (_, x) => {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      let top = f[j] + hh;
      for (let k = Math.min(j, x); k <= Math.max(j, x); k++) top = Math.max(top, f[k]);
      best = Math.min(best, top);
    }
    return Math.max(best, f[x]);
  });
  const lib = MO.hMinTransform(f, n, 1, hh, 4);
  check(same(brute, Array.from(lib)), `n=${n} h=${hh}`);
  // markers from hMinimaMarkers are exactly the regional minima of the filled signal
  const m1 = WS.hMinimaMarkers(f, n, 1, hh, 4), m2 = WS.minimaMarkers(brute, n, 1, 4);
  check(m1.count === m2.count && same(Array.from(m1.data, v => (v ? 1 : 0)), Array.from(m2.data, v => (v ? 1 : 0))), "same markers");
}

current = "flood level replay";
for (let t = 0; t < 60; t++) {
  const w = 3 + Math.floor(R.unif() * 8), h = 3 + Math.floor(R.unif() * 8), n = w * h, f = randInt(n, 0, 9);
  const mk = WS.minimaMarkers(f, w, h, 8);
  const r = WS.floodWatershed(f, w, h, mk.data);
  let ok = true;
  for (let p = 0; p < n; p++) {
    if (mk.data[p]) { if (r.level[p] !== -Infinity) ok = false; continue; }
    if (!(r.level[p] >= f[p])) ok = false;   // water level is never below the ground
  }
  check(ok, "level ≥ height, markers at −∞");
}

current = "mask: nothing outside is flooded";
for (let t = 0; t < 60; t++) {
  const w = 4 + Math.floor(R.unif() * 8), h = 4 + Math.floor(R.unif() * 8), n = w * h, f = randInt(n, 0, 9);
  const mask = Uint8Array.from({ length: n }, () => (R.unif() < 0.7 ? 1 : 0));
  const mk = randMarkers(n, 3);
  const a = WS.watershed(f, w, h, mk, { mask });
  check(a.every((v, i) => mask[i] || v === 0), "outside = 0");
  // every masked pixel connected (inside the mask) to a marker is labelled
  const lab = label2d(mask, w, h, 8);
  const hasMarker = new Set<number>();
  for (let i = 0; i < n; i++) if (mk[i] && mask[i]) hasMarker.add(lab.data[i]);
  check(a.every((v, i) => !mask[i] || (v > 0) === hasMarker.has(lab.data[i])), "reachable part labelled");
}

// ---------------------------------------------------------------------------------------
current = "flood1d helpers";
for (let t = 0; t < 100; t++) {
  const n = 6 + Math.floor(R.unif() * 30), f = Array.from({ length: n }, () => Math.round(R.unif() * 90) / 10);
  const mk = WS.minimaMarkers(f, n, 1, 4);
  const lab = WS.watershed(f, n, 1, mk.data);
  const level = Math.round(R.unif() * 100) / 10;
  const water = basinsAtLevel(f, lab, mk.data, level);
  // brute force: i is wet iff some marker j of the same basin is joined to i by samples ≤ level within the basin
  let ok = true;
  for (let i = 0; i < n; i++) {
    let wet = false;
    for (let j = 0; j < n; j++) {
      if (!mk.data[j] || lab[j] !== lab[i]) continue;
      let fine = true;
      for (let k = Math.min(i, j); k <= Math.max(i, j); k++) if (f[k] > level || lab[k] !== lab[i]) fine = false;
      if (fine) wet = true;
    }
    if (wet !== (water[i] > 0)) ok = false;
  }
  check(ok, "wet samples");
  const d = dams(f, lab);
  check(d.every(x => lab[x.at] !== lab[x.at + 1] && x.height === Math.max(f[x.at], f[x.at + 1])), "dams between basins at the pass height");
  check(d.length === WS.countLabels(lab) - 1, "k basins, k − 1 dams");
}
{
  const g = derivative1d([0, 0, 0, 4, 4, 4]);
  check(same(g, [0, 0, 2, 2, 0, 0]), `|f'| of a step = ${g.join(" ")}`);
}

current = "distws: neck depth of two discs";
for (const [r, d, want] of [[10, 12, 2], [10, 0, 0], [10, 20, 10]] as const) {
  check(Math.abs(neckDepth(r, d) - want) < 1e-9, `r=${r} d=${d}: ${neckDepth(r, d)}`);
}
{
  // two discs r = 20, d = 30 on a 120 × 80 grid: two h-maxima of the EDT for small h, one for h above the depth
  const w = 120, h = 80, m = twoDiscMask(w, h, 45, 40, 75, 40, 20, 20);
  const depth = neckDepth(20, 30);   // 20 − √(400 − 225) = 6.77
  check(depth > 6.7 && depth < 6.8, `depth ${depth}`);
  check(m.reduce((s, v) => s + v, 0) > 2000, "mask drawn");
}

// ---------------------------------------------------------------------------------------
const lines = Object.entries(section).map(([k, v]) => `  ${k}: ${v}`).join("\n");
console.log(`watershed-test: ${checks - failures}/${checks} checks passed\n${lines}`);
if (failures) process.exit(1);
