// Figures for the ray-tracing day (raycast/, whitted/, bvh/, pathtrace/), computed with the
// demos' own code (lib/raytrace.ts, lib/rays2d.ts, lib/bvh.ts, the demos' scene files).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/raytracing/figs.ts --bundle --platform=node --log-level=warning | node
import * as RT from "../../../demos/src/lib/raytrace";
import * as R2 from "../../../demos/src/lib/rays2d";
import * as B from "../../../demos/src/lib/bvh";
import { linearToSrgb8 } from "../../../demos/src/lib/colour";
import { makeScene as rsScene, DEF0, DEF1 } from "../../../demos/src/raystep/scenes";
import { PRESETS as R2P, traceAll } from "../../../demos/src/ray2d/presets";
import { makeScene as bvScene, primBox, primHit } from "../../../demos/src/bvhstep/scene";
import { rng } from "../../../demos/src/lib/random";

declare const require: (m: string) => any;
const fs = require("fs"), zlib = require("zlib");

// ---- tiny RGB PNG writer
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = (b: Uint8Array) => { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length); for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8); dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
function pngRGB(rgb: Uint8Array, w: number, h: number): Uint8Array {
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h); ihdr[8] = 8; ihdr[9] = 2;
  const raw = new Uint8Array((3 * w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgb.subarray(3 * y * w, 3 * (y + 1) * w), y * (3 * w + 1) + 1);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", new Uint8Array(zlib.deflateSync(raw))), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0)); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
const toRGB8 = (lin: Float32Array) => Uint8Array.from(lin, v => linearToSrgb8(v));
const save = (path: string, s: string | Uint8Array) => { fs.writeFileSync(path, s); console.log("wrote", path); };
const svg = (w: number, h: number, body: string) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="system-ui, Helvetica, Arial, sans-serif" font-size="20">\n<rect width="${w}" height="${h}" fill="#fff"/>\n${body}\n</svg>\n`;
const txt = (x: number, y: number, s: string, o: string = "") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" ${o}>${s}</text>`;
const line = (x1: number, y1: number, x2: number, y2: number, o: string) => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" ${o}/>`;
const arrowDefs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>`;

// ===================================================================== raycast/
{
  // camera rays in a side view: eye, image plane with 6 pixels, a circle and a segment
  const W = 760, H = 360, ex = 60, ey = 180, f = 120;
  let b = arrowDefs;
  b += line(ex + f, ey - 120, ex + f, ey + 120, `stroke="#222" stroke-width="4"`);
  for (let k = 0; k <= 6; k++) b += line(ex + f - 8, ey - 120 + 40 * k, ex + f + 8, ey - 120 + 40 * k, `stroke="#222" stroke-width="2"`);
  const cx = 520, cy = 140, r = 70;
  b += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#f2b8b0" stroke="#b33" stroke-width="2"/>`;
  b += line(430, 290, 700, 240, `stroke="#c9a400" stroke-width="7"`);
  for (let k = 0; k < 6; k++) {
    const py = ey - 120 + 40 * k + 20, dx = f, dy = py - ey, L = Math.hypot(dx, dy), d = [dx / L, dy / L];
    // nearest hit: circle, then segment
    const tc = B.rayDisc2([ex, ey], d, cx, cy, r), ts = B.raySegment2([ex, ey], d, 430, 290, 700, 240);
    const t = Math.min(tc, ts, 720);
    b += line(ex, ey, ex + t * d[0], ey + t * d[1], `stroke="#d49b00" stroke-width="2.5" ${t < 720 ? "" : 'stroke-dasharray="6 5"'}`);
    if (t < 720) b += `<circle cx="${(ex + t * d[0]).toFixed(1)}" cy="${(ey + t * d[1]).toFixed(1)}" r="5" fill="#000"/>`;
  }
  b += `<circle cx="${ex}" cy="${ey}" r="9" fill="#fff" stroke="#000" stroke-width="2.5"/>`;
  b += txt(ex - 30, ey + 40, "eye o");
  b += txt(ex + f - 40, ey + 150, "image plane: one ray per pixel");
  b += txt(cx - 30, cy - r - 12, "sphere");
  b += txt(560, 320, "triangle (edge-on)");
  save("raycast/fig-camera-rays.svg", svg(W, H, b));
}
{
  // a line through a circle: miss, graze, two hits
  const W = 760, H = 300, cx = 380, cy = 150, r = 95;
  let b = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#dbe8f5" stroke="#1f77b4" stroke-width="2.5"/>`;
  const rows: [number, string, string][] = [[cy - r - 30, "disc &lt; 0: miss", "#999"], [cy - r, "disc = 0: grazes", "#d49b00"], [cy + 25, "disc &gt; 0: two hits t₁ &lt; t₂", "#d62728"]];
  for (const [y, lab, col] of rows) {
    b += line(30, y, 730, y, `stroke="${col}" stroke-width="3"`);
    b += txt(40, y - 8, lab, `fill="${col}" font-size="19"`);
    const dy = y - cy;
    if (Math.abs(dy) <= r) {
      const s = Math.sqrt(r * r - dy * dy);
      for (const [x, l] of [[cx - s, "t₁"], [cx + s, "t₂"]] as [number, string][]) { b += `<circle cx="${x.toFixed(1)}" cy="${y}" r="6" fill="${col}"/>`; if (s > 1) b += txt(x - 8, y + 28, l, `fill="${col}"`); }
    }
  }
  b += `<circle cx="${cx}" cy="${cy}" r="4" fill="#1f77b4"/>` + txt(cx + 8, cy - 8, "c", `fill="#1f77b4"`);
  save("raycast/fig-sphere.svg", svg(W, H, b));
}
{
  const s = DEF0, sc = rsScene(s.scene, s.radius), W = 320, H = 240;
  const img = RT.renderWhitted(sc, { eye: s.eye, fov: s.fov }, W, H, { depth: 0, shadows: false, reflection: false, refraction: false, fresnel: false, eps: 0, float32: true });
  save("raycast/fig-raycast.png", pngRGB(toRGB8(img), W, H));
}

// ===================================================================== whitted/
{
  // Snell: air → glass 1.5 at 30°, reflected and refracted
  const W = 700, H = 400, x0 = 350, y0 = 200, L = 190;
  const n1 = 1, n2 = 1.5, th1 = 30 * Math.PI / 180, th2 = Math.asin(n1 / n2 * Math.sin(th1)), R = RT.fresnel(Math.cos(th1), n1, n2);
  let b = arrowDefs + `<rect x="0" y="${y0}" width="${W}" height="${H - y0}" fill="#dbe8f5"/>`;
  b += line(0, y0, W, y0, `stroke="#1f77b4" stroke-width="2.5"`);
  b += line(x0, 20, x0, H - 20, `stroke="#888" stroke-width="1.5" stroke-dasharray="6 5"`);
  b += line(x0 - L * Math.sin(th1), y0 - L * Math.cos(th1), x0, y0, `stroke="#d49b00" stroke-width="4" marker-end="url(#ah)"`);
  b += line(x0, y0, x0 + L * Math.sin(th1), y0 - L * Math.cos(th1), `stroke="#d49b00" stroke-width="3" stroke-opacity="0.45" marker-end="url(#ah)"`);
  b += line(x0, y0, x0 + L * Math.sin(th2), y0 + L * Math.cos(th2), `stroke="#d49b00" stroke-width="4" marker-end="url(#ah)"`);
  const arc = (a0: number, a1: number, rr: number, down: boolean) => { const p = (a: number) => [x0 + rr * Math.sin(a), down ? y0 + rr * Math.cos(a) : y0 - rr * Math.cos(a)]; const [ax, ay] = p(a0), [bx, by] = p(a1); return `<path d="M${ax.toFixed(1)},${ay.toFixed(1)} A${rr},${rr} 0 0 ${down ? 0 : 1} ${bx.toFixed(1)},${by.toFixed(1)}" fill="none" stroke="#000" stroke-width="1.5"/>`; };
  b += arc(-th1, 0, 70, false) + arc(0, th2, 70, true);
  b += txt(x0 - 75, y0 - 80, "θ₁ = 30°") + txt(x0 + 12, y0 + 95, `θ₂ = ${(th2 * 180 / Math.PI).toFixed(1)}°`);
  b += txt(20, 40, "air, n₁ = 1") + txt(20, y0 + 40, "glass, n₂ = 1.5");
  b += txt(x0 + 70, 60, `reflected: R = ${(100 * R).toFixed(1)} %`, `fill="#8a6a00"`);
  b += txt(x0 + 120, H - 30, `refracted: ${(100 * (1 - R)).toFixed(1)} %`, `fill="#8a6a00"`);
  b += txt(x0 - 330, H - 30, "n₁ sin θ₁ = n₂ sin θ₂ = 0.5");
  save("whitted/fig-snell.svg", svg(W, H, b));
}
{
  // ray tree of the glass preset's inspected pixel
  const s = { ...DEF1, scene: "glass" as const, depth: 5, W: 64, H: 48, inspect: [22, 26] as [number, number] };
  const sc = rsScene(s.scene, s.radius), ray = RT.cameraRay(s.inspect[0], s.inspect[1], s.W, s.H, { eye: s.eye, fov: s.fov });
  const res = RT.traceWhitted(sc, ray.o, ray.d, { depth: 5, shadows: true, reflection: true, refraction: true, fresnel: true, eps: 1e-4, float32: true, minWeight: 1e-3, tree: true });
  const col: Record<RT.RayKind, string> = { primary: "#d49b00", shadow: "#666", reflect: "#1f77b4", refract: "#2ca02c" };
  // layout: leaves get consecutive x slots, parents centred
  let slot = 0;
  const pos = new Map<RT.RayNode, [number, number]>();
  const lay = (n: RT.RayNode, d: number): number => {
    const xs = n.children.map(c => lay(c, d + 1));
    const x = xs.length ? (xs[0] + xs[xs.length - 1]) / 2 : slot++;
    pos.set(n, [x, d]); return x;
  };
  lay(res.node, 0);
  const W = 860, H = 80 + 75 * 5, sx = (x: number) => 40 + x * (W - 230) / Math.max(1, slot - 1), sy = (d: number) => 40 + d * 72;
  let b = "";
  const name = (n: RT.RayNode) => n.kind === "shadow" ? (n.obj >= 0 ? "blocked" : "light") : n.obj >= 0 ? sc.objects[n.obj].name.replace("sphere A", "glass ball") : "sky";
  pos.forEach(([x, d], n) => { for (const c of n.children) { const [cx, cd] = pos.get(c)!; b += line(sx(x), sy(d), sx(cx), sy(cd), `stroke="${col[c.kind]}" stroke-width="${2 + 4 * c.weight}" ${c.kind === "shadow" ? 'stroke-dasharray="6 5"' : ""}`); } });
  pos.forEach(([x, d], n) => { b += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(d)}" r="8" fill="${col[n.kind]}"/>` + txt(sx(x) + 12, sy(d) + 6, `${name(n)} (${n.weight.toFixed(n.weight < 0.01 ? 3 : 2)})`, `font-size="17"`); });
  b += txt(20, H - 16, "orange: primary · blue: reflected · green: refracted · dashed: shadow · (weight)", `font-size="17"`);
  save("whitted/fig-raytree.svg", svg(W, H, b));
}
{
  const s = { ...DEF1, scene: "glass" as const }, sc = rsScene(s.scene, s.radius), W = 320, H = 240;
  const img = RT.renderWhitted(sc, { eye: s.eye, fov: s.fov }, W, H, { depth: 5, shadows: true, reflection: true, refraction: true, fresnel: true, eps: 1e-4, float32: true, minWeight: 1e-3 });
  save("whitted/fig-whitted.png", pngRGB(toRGB8(img), W, H));
}
{
  // ray2d "a ball: aberration" as a static figure
  const st = R2P.find(p => p.id === "ball")!.s(), res = traceAll(st);
  const WX = 40, WY = 25, W = 760, H = W * WY / WX, X = (x: number) => x / WX * W, Y = (y: number) => H - y / WY * H;
  let b = `<circle cx="${X(13)}" cy="${Y(12.5)}" r="${X(5)}" fill="#cfe0f2" stroke="#1f77b4" stroke-width="2"/>`;
  b += line(0, Y(12.5), W, Y(12.5), `stroke="#999" stroke-dasharray="6 5"`);
  for (const g of res.segs) b += line(X(g.x0), Y(g.y0), X(Math.min(60, g.x1)), Y(g.y1 + (Math.min(60, g.x1) - g.x1) * (g.y1 - g.y0) / (g.x1 - g.x0 || 1)), `stroke="#e08a00" stroke-width="2"`);
  b += line(X(20.5), Y(12.5) - 20, X(20.5), Y(12.5) + 20, `stroke="#000" stroke-width="2"`) + txt(X(20.5) + 6, Y(12.5) + 40, "paraxial focus", `font-size="18"`);
  b += txt(X(9.5), Y(4), "glass ball, n = 1.5", `font-size="18"`);
  save("whitted/fig-ball.svg", svg(W, Math.round(H), b));
}

// ===================================================================== bvh/
{
  // slab test: box [0, 2] × [0, 1], ray o = (−1, 0.5), d = (1, 0.25)
  const W = 720, H = 380, X = (x: number) => 140 + x * 180, Y = (y: number) => 280 - y * 180;
  let b = arrowDefs;
  b += `<rect x="${X(0)}" y="${Y(1)}" width="${X(2) - X(0)}" height="${Y(0) - Y(1)}" fill="#dbe8f5" stroke="#1f77b4" stroke-width="2.5"/>`;
  b += `<rect x="${X(0)}" y="0" width="${X(2) - X(0)}" height="${H}" fill="#1f77b4" fill-opacity="0.07"/>`;
  b += `<rect x="0" y="${Y(1)}" width="${W}" height="${Y(0) - Y(1)}" fill="#d62728" fill-opacity="0.07"/>`;
  const o = [-1, 0.5], d = [1, 0.25];
  b += line(X(o[0] - 0.3 * d[0]), Y(o[1] - 0.3 * d[1]), X(o[0] + 3.4 * d[0]), Y(o[1] + 3.4 * d[1]), `stroke="#d49b00" stroke-width="3" marker-end="url(#ah)"`);
  const P = (t: number) => [X(o[0] + t * d[0]), Y(o[1] + t * d[1])];
  for (const [t, lab] of [[0, "t = 0"], [1, "t = 1"], [2, "t = 2"], [3, "t = 3"]] as [number, string][]) { const [x, y] = P(t); b += `<circle cx="${x}" cy="${y}" r="6" fill="${t === 1 || t === 2 ? "#000" : "#888"}"/>` + txt(x - 18, y + 30, lab, `font-size="18"`); }
  const [xm2, ym2] = P(-2);
  void xm2; void ym2;
  b += txt(X(0) + 6, 24, "x-slab: t ∈ [1, 3]", `fill="#1f77b4" font-size="19"`);
  b += txt(X(2) + 20, Y(1) + 26, "y-slab: t ∈ [−2, 2]", `fill="#d62728" font-size="19"`);
  b += txt(20, H - 16, "box = both slabs: t ∈ [max(1, −2), min(3, 2)] = [1, 2]: a hit, entering at t = 1", `font-size="18"`);
  save("bvh/fig-slab.svg", svg(W, H, b));
}
{
  // a BVH over 64 objects, levels 1–3, and the traversal of one ray
  const prims = bvScene(64, "uniform"), boxes = prims.map(primBox), bvh = B.buildBVH(boxes, { method: "sah" });
  const W = 520, X = (x: number) => 10 + x * 5, Y = (y: number) => 510 - y * 5;
  const o = [6, 12], aim = [60, 62], L = Math.hypot(aim[0] - o[0], aim[1] - o[1]), d = [(aim[0] - o[0]) / L, (aim[1] - o[1]) / L];
  const tr = B.traverseBVH(bvh, o, d, (p, tm) => primHit(prims[p], o, d, tm), Infinity, true);
  const visited = new Set(tr.trace.filter(e => e.kind === "node" && e.hit).map(e => (e as { id: number }).id));
  const tested = new Set(tr.trace.filter(e => e.kind === "prim").map(e => (e as { id: number }).id));
  let b = "";
  const cols = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"];
  bvh.nodes.forEach((n, i) => { if (n.depth > 3) return; b += `<rect x="${X(n.min[0]).toFixed(1)}" y="${Y(n.max[1]).toFixed(1)}" width="${((n.max[0] - n.min[0]) * 5).toFixed(1)}" height="${((n.max[1] - n.min[1]) * 5).toFixed(1)}" fill="none" stroke="${cols[n.depth]}" stroke-width="${visited.has(i) ? 3.5 : 1.5}" ${visited.has(i) ? "" : 'stroke-dasharray="5 4"'}/>`; });
  prims.forEach((p, i) => {
    const st = tested.has(i) ? `fill="#9ecae1" stroke="#000" stroke-width="2"` : `fill="#c9d3dd" stroke="#5b6b7b" stroke-width="1"`;
    b += p.kind === 0 ? `<circle cx="${X(p.a[0]).toFixed(1)}" cy="${Y(p.a[1]).toFixed(1)}" r="${(p.a[2] * 5).toFixed(1)}" ${st}/>` : line(X(p.a[0]), Y(p.a[1]), X(p.a[2]), Y(p.a[3]), tested.has(i) ? `stroke="#000" stroke-width="3.5"` : `stroke="#5b6b7b" stroke-width="2"`);
  });
  b += line(X(o[0]), Y(o[1]), X(o[0] + tr.t * d[0]), Y(o[1] + tr.t * d[1]), `stroke="#d49b00" stroke-width="3"`) + `<circle cx="${X(o[0] + tr.t * d[0]).toFixed(1)}" cy="${Y(o[1] + tr.t * d[1]).toFixed(1)}" r="6" fill="#d62728"/>`;
  b += txt(14, 540, `${tr.boxTests} box tests + ${tr.primTests} object tests, not 64`, `font-size="17"`);
  save("bvh/fig-bvh.svg", svg(W, 552, b));
}

// ===================================================================== pathtrace/
{
  // Monte Carlo in 1D: ∫0^1 x² dx with 8 uniform samples
  const W = 620, H = 360, X = (x: number) => 70 + x * 500, Y = (y: number) => 300 - y * 250;
  let b = `<path d="M${X(0)},${Y(0)} ${Array.from({ length: 101 }, (_, k) => `L${X(k / 100).toFixed(1)},${Y((k / 100) ** 2).toFixed(1)}`).join(" ")} L${X(1)},${Y(0)} Z" fill="#dbe8f5"/>`;
  b += `<path d="M${Array.from({ length: 101 }, (_, k) => `${X(k / 100).toFixed(1)},${Y((k / 100) ** 2).toFixed(1)}`).join(" L")}" fill="none" stroke="#1f77b4" stroke-width="3"/>`;
  b += line(X(0), Y(0), X(1.05), Y(0), `stroke="#222" stroke-width="2"`) + line(X(0), Y(0), X(0), Y(1.1), `stroke="#222" stroke-width="2"`);
  const r = rng(3), xs = Array.from({ length: 8 }, () => r.unif());
  let s = 0;
  for (const x of xs) { s += x * x; b += line(X(x), Y(0), X(x), Y(x * x), `stroke="#d62728" stroke-width="2.5"`) + `<circle cx="${X(x).toFixed(1)}" cy="${Y(x * x).toFixed(1)}" r="5" fill="#d62728"/>`; }
  const est = s / 8;
  b += line(X(0), Y(est), X(1), Y(est), `stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"`);
  b += txt(X(0.03), Y(est) - 8, `mean of 8 samples = ${est.toFixed(3)}`, `fill="#d62728"`);
  b += txt(X(0.03), Y(1) + 10, "f(x) = x², area = 1/3 = 0.333", `fill="#1f77b4"`);
  b += txt(X(0) - 8, Y(0) + 28, "0") + txt(X(1) - 8, Y(0) + 28, "1") + txt(X(0) - 30, Y(1) + 6, "1");
  save("pathtrace/fig-mc1d.svg", svg(W, H, b));
  console.log("mc1d samples", xs.map(v => v.toFixed(3)).join(" "), "estimate", est.toFixed(4));
}
{
  // the box at 1, 16 and 256 samples per pixel (CPU path tracer, cosine + light sampling)
  const W = 200, H = 150, sc = RT.boxScene(), opt: RT.PathOptions = { maxBounces: 5, russianRoulette: true, sampling: "light" };
  const P = new RT.Progressive(sc, sc.camera, W, H, opt, 1);
  const out = new Uint8Array(3 * (3 * W + 20) * H).fill(255);
  let col = 0;
  for (const target of [1, 16, 256]) {
    while (P.spp < target) P.addSample();
    const m = P.mean();
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) for (let c = 0; c < 3; c++) out[3 * (j * (3 * W + 20) + col + i) + c] = linearToSrgb8(m[3 * (j * W + i) + c]);
    col += W + 10;
  }
  save("pathtrace/fig-spp.png", pngRGB(out, 3 * W + 20, H));
}
