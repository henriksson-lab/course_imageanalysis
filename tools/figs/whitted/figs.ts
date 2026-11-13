// Slide figures for whitted/ (second pass: pictures on the slides). fig-fresnel.svg comes from fig_fresnel.py.
// Run from the repo root:
//   node_modules/.bin/esbuild tools/figs/whitted/figs.ts --bundle --platform=node --log-level=warning | node
import * as RT from "../../../demos/src/lib/raytrace";
import { linearToSrgb8 } from "../../../demos/src/lib/colour";
import { makeScene, DEF1 } from "../../../demos/src/raystep/scenes";
import { pngRGB, upscale, imgEl, save, svg, txt, line, circ, poly, arrowDefs, C } from "../raycast/util";

// ---------------------------------------------------------------- shadow rays
{
  const W = 560, H = 360, gy = 300;
  let b = arrowDefs;
  b += `<rect x="0" y="${gy}" width="${W}" height="${H - gy}" fill="#e8e8e8"/>` + line(0, gy, W, gy, `stroke="#444" stroke-width="3"`);
  const L = [470, 50];
  b += circ(L[0], L[1], 16, `fill="${C.yellow}" stroke="#8a7a00" stroke-width="2"`) + txt(L[0] - 80, L[1] + 6, "light", `text-anchor="middle"`);
  const S = [330, 215], r = 50;
  b += circ(S[0], S[1], r, `fill="#cfe3f3" stroke="${C.blue}" stroke-width="2.5"`);
  const E = [30, 120];
  const p1 = [160, gy], p2 = [250, gy];
  b += line(E[0], E[1], p1[0], p1[1], `stroke="${C.orange}" stroke-width="3" marker-end="url(#ah)"`);
  b += line(E[0], E[1], p2[0], p2[1], `stroke="${C.orange}" stroke-width="3" marker-end="url(#ah)"`);
  b += circ(E[0], E[1], 8, `fill="#fff" stroke="#000" stroke-width="2.5"`) + txt(E[0] - 14, E[1] - 18, "eye");
  // p1 → light: free
  b += line(p1[0], p1[1], L[0] - 12, L[1] + 10, `stroke="${C.grey}" stroke-width="2.5" stroke-dasharray="8 5" marker-end="url(#ah)"`);
  // p2 → light: blocked by the sphere
  const dx = L[0] - p2[0], dy = L[1] - p2[1], len = Math.hypot(dx, dy), d = [dx / len, dy / len];
  const t = RT.hitSphere([p2[0], p2[1], 0], [d[0], d[1], 0], [S[0], S[1], 0], r);
  const hx = p2[0] + t * d[0], hy = p2[1] + t * d[1];
  b += line(p2[0], p2[1], hx, hy, `stroke="${C.grey}" stroke-width="2.5" stroke-dasharray="8 5"`);
  b += txt(hx - 9, hy + 9, "✕", `font-size="26" fill="${C.red}" font-weight="700"`);
  b += circ(p1[0], p1[1], 7, `fill="#fff" stroke="#000" stroke-width="2.5"`) + circ(p2[0], p2[1], 7, `fill="#000"`);
  b += txt(p1[0] - 30, gy + 34, "lit") + txt(p2[0] - 10, gy + 34, "in shadow");
  b += txt(W - 120, gy + 34, "ground", `fill="#555"`);
  save("whitted/fig-shadowray.svg", svg(W, H, b, 22));
}

// ---------------------------------------------------------------- shadow acne and light leaks (raystep presets)
{
  const s = DEF1, cam = { eye: s.eye, fov: s.fov }, W = 64, H = 48, k = 3;
  const opt = (eps: number) => ({ depth: 1, shadows: true, reflection: true, refraction: true, fresnel: true, eps, float32: true, minWeight: 1e-3 });
  const rows: [string, number, string, string][] = [
    ["basic", 0, "ε = 0:", "speckled acne"],
    ["basic", 1e-4, "ε = 10⁻⁴:", "only real shadows"],
    ["tile", 10 ** -0.7, "ε = 0.2: the low", "tile casts no shadow"],
  ];
  let b = "";
  rows.forEach(([id, eps, l1, l2], i) => {
    const sc = makeScene(id as any, s.radius), img = RT.renderWhitted(sc, cam, W, H, opt(eps));
    const rgb = Uint8Array.from(img, v => linearToSrgb8(v)), y = i * (H * k + 14);
    b += imgEl(0, y, W * k, H * k, pngRGB(upscale(rgb, W, H, k), W * k, H * k)) + `<rect x="0" y="${y}" width="${W * k}" height="${H * k}" fill="none" stroke="#999"/>`;
    b += txt(W * k + 14, y + 60, l1, `font-weight="600"`) + txt(W * k + 14, y + 88, l2);
  });
  save("whitted/fig-acne.svg", svg(W * k + 250, 3 * H * k + 28, b, 22));
}

// ---------------------------------------------------------------- mirror reflection: split d along and across n
{
  const W = 540, H = 330, ox = 270, oy = 260, L = 170;
  const d = [1 / Math.SQRT2, 1 / Math.SQRT2]; // screen coords: right and down = (1, −1)/√2 in the slide's frame
  let b = arrowDefs;
  b += `<rect x="0" y="${oy}" width="${W}" height="${H - oy}" fill="#dde3ea"/>` + line(0, oy, W, oy, `stroke="#445" stroke-width="3"`);
  b += txt(10, oy + 40, "mirror", `fill="#445"`);
  const sx = ox - L * d[0], sy = oy - L * d[1];
  // incoming d, and its parts
  b += line(sx, sy, ox, oy, `stroke="${C.orange}" stroke-width="4" marker-end="url(#ah)"`) + txt(sx + 30, sy + 70, "d", `font-weight="700" fill="#8a5a00"`);
  b += line(sx, sy, ox, sy, `stroke="${C.blue}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ah)"`) + txt(sx + 40, sy - 10, "d∥", `fill="${C.blue}"`);
  b += line(ox, sy, ox, oy, `stroke="${C.red}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ah)"`);
  // reflected r
  const ex = ox + L * d[0], ey = oy - L * d[1];
  b += line(ox, oy, ex, ey, `stroke="${C.orange}" stroke-width="4" marker-end="url(#ah)"`) + txt(ex - 50, ey + 70, "r", `font-weight="700" fill="#8a5a00"`);
  b += line(ox, sy, ex, ey, `stroke="${C.blue}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ah)"`) + txt(ox + 60, sy - 10, "d∥", `fill="${C.blue}"`);
  b += txt(ox - 54, oy - 70, "d⊥", `fill="${C.red}"`);
  b += line(490, oy, 490, oy - 70, `stroke="#000" stroke-width="2.5" marker-end="url(#ah)"`) + txt(500, oy - 50, "n");
  b += txt(10, 26, "r = d∥ − d⊥: keep d∥, flip d⊥", `font-size="21"`) + txt(10, 54, "d = (1, −1, 0)/√2  →  r = (1, 1, 0)/√2", `font-size="21"`);
  save("whitted/fig-mirror.svg", svg(W, H, b, 22));
}
