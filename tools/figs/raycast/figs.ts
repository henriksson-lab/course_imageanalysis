// Slide figures for raycast/ (second pass: pictures on the slides).
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/raycast/figs.ts --bundle --platform=node --log-level=warning | node
import * as RT from "../../../demos/src/lib/raytrace";
import { linearToSrgb8 } from "../../../demos/src/lib/colour";
import { makeScene, DEF0 } from "../../../demos/src/raystep/scenes";
import { pngRGB, imgEl, save, svg, txt, line, circ, poly, arrowDefs, C } from "./util";

// ---------------------------------------------------------------- the two loops, swapped
{
  const W = 560, H = 560;
  let b = arrowDefs;
  // top: rasterisation, a triangle over a pixel grid
  b += txt(10, 30, "rasterisation: for each triangle,", `font-weight="600"`) + txt(10, 56, "the pixels it covers");
  const gx = 150, gy = 80, cs = 26, nx = 12, ny = 7;
  const T = [[gx + 30, gy + 165], [gx + 290, gy + 120], [gx + 120, gy + 15]];
  const inside = (x: number, y: number) => {
    const s = (p: number[], q: number[]) => (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
    const a = s(T[0], T[1]), bb = s(T[1], T[2]), c = s(T[2], T[0]);
    return (a >= 0 && bb >= 0 && c >= 0) || (a <= 0 && bb <= 0 && c <= 0);
  };
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = gx + i * cs, y = gy + j * cs;
    b += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}" fill="${inside(x + cs / 2, y + cs / 2) ? "#f6d58c" : "#fff"}" stroke="#bbb"/>`;
  }
  b += poly(T, `fill="none" stroke="${C.orange}" stroke-width="3.5"`);
  b += line(30, 190, gx - 8, 175, `stroke="#000" stroke-width="2.5" marker-end="url(#ah)"`) + txt(10, 220, "1 triangle", `font-size="21"`);
  b += txt(gx, gy + ny * cs + 28, "fills the pixels whose centres it covers", `font-size="21"`);
  // bottom: ray casting, one pixel, a ray tested against every object
  const y0 = 330;
  b += txt(10, y0, "ray casting: for each pixel,", `font-weight="600"`) + txt(10, y0 + 26, "the objects along its ray");
  const ex = 40, ey = 460, px = 120;
  b += line(px, y0 + 50, px, y0 + 200, `stroke="#222" stroke-width="4"`);
  b += `<rect x="${px - 6}" y="${ey - 18}" width="12" height="20" fill="${C.orange}"/>`;
  const d = [1, -0.04];
  b += line(ex, ey, 545, ey + (545 - ex) * d[1], `stroke="${C.orange}" stroke-width="3" marker-end="url(#ah)"`);
  b += circ(ex, ey, 8, `fill="#fff" stroke="#000" stroke-width="2.5"`) + txt(ex - 22, ey + 34, "eye");
  const yAt = (x: number) => ey + (x - ex) * d[1];
  // three objects: a disc (hit), a segment (hit, farther), a disc (missed)
  b += circ(270, yAt(270) - 20, 45, `fill="#cfe3f3" stroke="${C.blue}" stroke-width="2.5"`);
  b += line(420, yAt(420) - 60, 440, yAt(440) + 60, `stroke="${C.green}" stroke-width="6"`);
  b += circ(350, yAt(350) + 70, 26, `fill="#cfe3f3" stroke="${C.blue}" stroke-width="2.5"`);
  const s = Math.sqrt(45 * 45 - 20 * 20);
  b += circ(270 - s, yAt(270 - s), 7, `fill="#000"`) + circ(270 + s, yAt(270 + s), 6, `fill="#fff" stroke="#000" stroke-width="2"`);
  b += circ(430, yAt(430), 6, `fill="#fff" stroke="#000" stroke-width="2"`);
  b += txt(10, H - 8, "test every object; keep the nearest hit (●)", `font-size="21"`);
  b += txt(382, yAt(350) + 78, "missed", `font-size="21" fill="${C.grey}"`);
  save("raycast/fig-loops.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- the ray and the nearest hit
{
  const W = 960, H = 280, ry = 175, X = (t: number) => 160 + 85 * t;
  let b = arrowDefs;
  // sphere: c at t = 5, 0.5 off the ray, r = 1 (the example of the math slide: t = 4.134, 5.866)
  b += circ(X(5), ry - 42.5, 85, `fill="#cfe3f3" stroke="${C.blue}" stroke-width="2.5"`) + txt(X(5) - 30, ry - 65, "sphere", `fill="${C.blue}"`);
  b += line(X(8), 30, X(8), H - 40, `stroke="${C.green}" stroke-width="6"`) + txt(X(8) + 10, 50, "plane", `fill="${C.green}"`);
  b += line(X(-0.9), ry - 60, X(-0.9), ry + 30, `stroke="${C.red}" stroke-width="6"`);
  b += line(X(-1.4), ry, X(0), ry, `stroke="${C.orange}" stroke-width="2.5" stroke-dasharray="6 5"`);
  b += line(X(0), ry, X(9.3), ry, `stroke="${C.orange}" stroke-width="3.5" marker-end="url(#ah)"`);
  b += circ(X(0), ry, 9, `fill="#fff" stroke="#000" stroke-width="2.5"`) + txt(X(0) - 12, ry - 18, "o");
  for (const [t, near] of [[4.134, true], [5.866, false], [8, false]] as [number, boolean][]) b += circ(X(t), ry, near ? 8 : 6, near ? `fill="#000"` : `fill="#fff" stroke="#000" stroke-width="2"`);
  b += circ(X(-0.9), ry, 6, `fill="#fff" stroke="${C.red}" stroke-width="2.5"`);
  b += txt(8, ry + 58, "t = −0.9", `fill="${C.red}"`) + txt(8, ry + 82, "behind: ignored", `fill="${C.red}" font-size="21"`);
  b += txt(X(4.134) - 62, ry + 34, "t = 4.13", `font-weight="600"`) + txt(X(5.866) - 10, ry + 34, "5.87") + txt(X(8) + 10, ry + 34, "t = 8");
  b += txt(X(4.134) - 120, ry + 70, "nearest hit: the pixel sees this", `font-size="21" font-weight="600"`);
  b += txt(X(9.3) - 10, ry - 12, "t");
  save("raycast/fig-nearest.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- implicit vs parametric
{
  const W = 1060, H = 330;
  let b = arrowDefs;
  // left: implicit circle with contours of f = |x − c|² − r²
  const cx = 220, cy = 180, sc = 66; // 1 unit = 70 px, r = 1
  for (const [lev, lab] of [[-0.75, ""], [0, "f = 0"], [1.25, "f &gt; 0"], [2.3, ""]] as [number, string][]) {
    const rr = Math.sqrt(1 + lev) * sc;
    b += circ(cx, cy, rr, lev === 0 ? `fill="#cfe3f3" fill-opacity="0.8" stroke="${C.blue}" stroke-width="3.5"` : `fill="none" stroke="${C.blue}" stroke-width="1.5" stroke-dasharray="5 4"`);
    if (lab) b += txt(cx + rr * 0.72 + 4, cy - rr * 0.72 - 4, lab, `fill="${C.blue}" font-size="21" stroke="#fff" stroke-width="6" paint-order="stroke"`);
  }
  b += txt(cx - 28, cy + 6, "f &lt; 0", `fill="${C.blue}" font-size="21"`);
  const ry = cy + 35, s = Math.sqrt(sc * sc - 35 * 35);
  b += line(10, ry, 420, ry, `stroke="${C.orange}" stroke-width="3" marker-end="url(#ah)"`);
  b += circ(cx - s, ry, 6, `fill="#000"`) + circ(cx + s, ry, 6, `fill="#000"`);
  b += txt(cx - s - 24, ry + 28, "t₁", `font-size="21"`) + txt(cx + s + 6, ry + 28, "t₂", `font-size="21"`);
  b += txt(10, 26, "implicit: where f(x) = 0", `font-weight="600"`);
  b += txt(10, H - 12, "substitute the ray, solve f(o + t d) = 0 for t", `font-size="21"`);
  // right: parametric triangle with u, v lines
  const A = [600, 270], Bp = [990, 250], Cp = [720, 70];
  const P = (u: number, v: number) => [A[0] + u * (Bp[0] - A[0]) + v * (Cp[0] - A[0]), A[1] + u * (Bp[1] - A[1]) + v * (Cp[1] - A[1])];
  b += poly([A, Bp, Cp], `fill="#fbe3c4" stroke="${C.orange}" stroke-width="3"`);
  for (const k of [0.25, 0.5, 0.75]) {
    const [x1, y1] = P(k, 0), [x2, y2] = P(k, 1 - k); b += line(x1, y1, x2, y2, `stroke="${C.grey}" stroke-width="1.2" stroke-dasharray="4 4"`);
    const [x3, y3] = P(0, k), [x4, y4] = P(1 - k, k); b += line(x3, y3, x4, y4, `stroke="${C.grey}" stroke-width="1.2" stroke-dasharray="4 4"`);
  }
  b += line(A[0], A[1], A[0] + 0.97 * (Bp[0] - A[0]), A[1] + 0.97 * (Bp[1] - A[1]), `stroke="#000" stroke-width="2" marker-end="url(#ah)"`);
  b += line(A[0], A[1], A[0] + 0.95 * (Cp[0] - A[0]), A[1] + 0.95 * (Cp[1] - A[1]), `stroke="#000" stroke-width="2" marker-end="url(#ah)"`);
  const [hx, hy] = P(0.3, 0.5);
  b += circ(hx, hy, 7, `fill="#000"`) + txt(hx + 10, hy - 8, "u = 0.3, v = 0.5", `font-size="21"`);
  b += txt(A[0] - 26, A[1] + 6, "a") + txt(Bp[0] + 6, Bp[1] + 24, "b") + txt(Cp[0] + 10, Cp[1] - 2, "c");
  b += txt(780, 290, "u", `font-size="21"`) + txt(630, 170, "v", `font-size="21"`);
  b += txt(520, 26, "parametric: a + u(b − a) + v(c − a)", `font-weight="600"`);
  b += txt(520, H - 12, "set it equal to the ray, solve for t, u, v", `font-size="21"`);
  save("raycast/fig-surfaces.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- a ray and a plane (side view)
{
  const W = 600, H = 330, ex = 240, ey = 130, s = 120, X = (z: number) => ex + s * z, Y = (y: number) => ey - s * y; // z: distance forward
  let b = arrowDefs;
  b += `<rect x="0" y="${Y(-1)}" width="${W}" height="${H - Y(-1)}" fill="#e8e8e8"/>` + line(0, Y(-1), W, Y(-1), `stroke="#444" stroke-width="3"`);
  b += txt(W - 130, H - 14, "floor y = −1", `font-size="21"`);
  // looking straight ahead: parallel
  b += line(ex, ey, W - 10, ey, `stroke="${C.grey}" stroke-width="2.5" marker-end="url(#ah)"`) + txt(ex + 50, ey - 12, "(0, 0, −1): parallel, no hit", `fill="${C.grey}" font-size="21"`);
  // looking down: hits at t = 1.667
  b += line(ex, ey, X(4 / 3), Y(-1), `stroke="${C.orange}" stroke-width="3.5" marker-end="url(#ah)"`);
  b += circ(X(4 / 3), Y(-1), 7, `fill="#000"`) + txt(X(4 / 3) - 20, Y(-1) + 30, "t = 1.667", `font-weight="600" font-size="21"`);
  b += txt(330, 230, "(0, −0.6, −0.8)", `fill="#8a5a00" font-size="21" text-anchor="end"`);
  // looking up: hit behind the eye
  b += line(ex, ey, X(0.8 * 0.9), Y(0.6 * 0.9), `stroke="${C.blue}" stroke-width="3" marker-end="url(#ah)"`) + txt(X(0.75), Y(0.6) + 8, "(0, 0.6, −0.8)", `fill="${C.blue}" font-size="21"`);
  b += line(ex, ey, X(-4 / 3), Y(-1), `stroke="${C.blue}" stroke-width="2" stroke-dasharray="6 5"`);
  b += circ(X(-4 / 3), Y(-1), 6, `fill="#fff" stroke="${C.blue}" stroke-width="2.5"`) + txt(10, Y(-1) + 30, "t = −1.667: behind", `fill="${C.blue}" font-size="21"`);
  b += circ(ex, ey, 8, `fill="#fff" stroke="#000" stroke-width="2.5"`) + txt(ex - 50, ey - 14, "eye", `font-size="21"`);
  b += line(X(2.6), Y(-1), X(2.6), Y(-0.45), `stroke="#000" stroke-width="2.5" marker-end="url(#ah)"`) + txt(X(2.6) + 8, Y(-0.5), "n", `font-size="21"`);
  save("raycast/fig-plane.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- barycentric coordinates
{
  const W = 500, H = 430, A = [50, 370], Bp = [470, 380], Cp = [230, 50];
  const P = (w: number[]) => [w[0] * A[0] + w[1] * Bp[0] + w[2] * Cp[0], w[0] * A[1] + w[1] * Bp[1] + w[2] * Cp[1]];
  const p1 = P([0.2, 0.3, 0.5]), p2 = P([0.6, -0.1, 0.5]), p3 = P([0.4, 0.6, 0]);
  let b = "";
  // the three sub-triangles of p1: area opposite each corner = that corner's weight
  b += poly([p1, Bp, Cp], `fill="${C.sky}" fill-opacity="0.35"`) + poly([p1, Cp, A], `fill="${C.orange}" fill-opacity="0.35"`) + poly([p1, A, Bp], `fill="${C.purple}" fill-opacity="0.3"`);
  b += poly([A, Bp, Cp], `fill="none" stroke="#000" stroke-width="3"`);
  for (const q of [A, Bp, Cp]) b += line(p1[0], p1[1], q[0], q[1], `stroke="#555" stroke-width="1.2"`);
  b += txt(A[0] - 30, A[1] + 26, "a") + txt(Bp[0] - 4, Bp[1] + 30, "b") + txt(Cp[0] + 12, Cp[1] + 6, "c");
  b += circ(p1[0], p1[1], 7, `fill="#000"`) + txt(p1[0] + 5, p1[1] + 75, "(0.2, 0.3, 0.5)", `font-size="21" font-weight="600" text-anchor="middle"`) + txt(p1[0] + 5, p1[1] + 99, "inside", `font-size="21" text-anchor="middle"`);
  b += circ(p2[0], p2[1], 7, `fill="#fff" stroke="${C.red}" stroke-width="3"`) + txt(8, p2[1] - 78, "(0.6, −0.1, 0.5)", `font-size="21" fill="${C.red}"`) + txt(8, p2[1] - 56, "outside", `font-size="21" fill="${C.red}"`);
  b += circ(p3[0], p3[1], 7, `fill="${C.blue}"`) + txt(p3[0] - 20, p3[1] + 34, "(0.4, 0.6, 0): on edge ab", `font-size="21" fill="${C.blue}" text-anchor="middle"`);
  b += txt(10, 26, "weight of a corner = area opposite it", `font-size="21"`);
  save("raycast/fig-bary.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- per-pixel outputs (raystep "basic" scene)
{
  const s = DEF0, sc = makeScene("basic", s.radius), W = 240, H = 180, cam = { eye: s.eye, fov: s.fov };
  const shaded = RT.renderWhitted(sc, cam, W, H, { depth: 0, shadows: false, reflection: false, refraction: false, fresnel: false, eps: 0, float32: true });
  const sh = Uint8Array.from(shaded, v => linearToSrgb8(v));
  const dep = new Uint8Array(3 * W * H), nor = new Uint8Array(3 * W * H), ids = new Uint8Array(3 * W * H);
  const idCol = [[230, 159, 0], [86, 180, 233], [240, 228, 66], [0, 158, 115]], bg = [255, 255, 255];
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const r = RT.cameraRay(i, j, W, H, cam), h = RT.nearestHit(sc.objects, r.o, r.d), p = 3 * (j * W + i);
    if (!h) { dep.set([0, 0, 0], p); nor.set(bg, p); ids.set(bg, p); continue; }
    const g = Math.round(255 * Math.max(0, Math.min(1, 1 - (h.t - 1.5) / 8)));
    dep.set([g, g, g], p);
    const n = RT.normalAt(sc.objects[h.obj], h.p, h.u, h.v);
    nor.set([0, 1, 2].map(k => Math.round(255 * (0.5 + 0.5 * n[k]))), p);
    ids.set(idCol[h.obj % 4], p);
  }
  const lab = ["shaded", "depth t (near = bright)", "normal as colour", "object id"];
  let b = "";
  [sh, dep, nor, ids].forEach((im, k) => {
    const x = (k % 2) * (W + 20), y = Math.floor(k / 2) * (H + 34);
    b += imgEl(x, y, W, H, pngRGB(im, W, H), false) + `<rect x="${x}" y="${y}" width="${W}" height="${H}" fill="none" stroke="#999"/>` + txt(x + 2, y + H + 24, lab[k], `font-size="21"`);
  });
  save("raycast/fig-gbuffer.svg", svg(2 * W + 20, 2 * (H + 34), b));
}
