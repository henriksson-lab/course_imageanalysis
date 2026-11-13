// Synthetic micrograph for the toy promptable segmenter (no DOM): nuclei with nucleoli, a
// cluster of touching nuclei, a touching pair, a cell (dim cytoplasm around a nucleus) and
// bright debris, with ground truth at three levels:
//   part    = nucleoli
//   object  = nuclei
//   group   = touching nuclei together, or the whole cell (cytoplasm + nucleus); an isolated
//             nucleus is its own group
// Fixed seed: every student sees the same image.
import { rng } from "../lib/ui";

export const W = 256;

interface Nuc { x: number; y: number; a: number; b: number; ang: number; I: number; nucleoli: [number, number][]; group: number }

export interface Scene {
  img: Float32Array;          // 0..255, W x W
  part: Int32Array;           // label maps (0 = none)
  object: Int32Array;
  group: Int32Array;
  nObjects: number;
  /** Named spots used by the presets (pixel coordinates). */
  spots: Record<string, [number, number]>;
  box: [number, number, number, number];
}

const sig = (t: number) => 1 / (1 + Math.exp(-t));

/** Inside-ness of an ellipse at (px, py): 1 inside, 0 outside, a soft edge of about 1.5 px. */
function soft(px: number, py: number, x: number, y: number, a: number, b: number, ang: number, edge = 1.5): number {
  const c = Math.cos(ang), s = Math.sin(ang), dx = px - x, dy = py - y;
  const u = (c * dx + s * dy) / a, v = (-s * dx + c * dy) / b;
  const rho = Math.sqrt(u * u + v * v);
  return sig((1 - rho) * Math.min(a, b) / edge * 2.2);
}

export function makeScene(seed = 7): Scene {
  const r = rng(seed);
  const N = W * W;
  // Nuclei: [x, y, a, b, angle, brightness]; group id: 0 = isolated (own group).
  const nucs: Nuc[] = [];
  const add = (x: number, y: number, a: number, b: number, ang: number, I: number, group = 0, nn = 2) => {
    const nucleoli: [number, number][] = [];
    for (let k = 0; k < nn; k++) {
      const t = r.unif() * 2 * Math.PI, d = 0.45 * Math.min(a, b) * Math.sqrt(r.unif());
      nucleoli.push([x + d * Math.cos(t), y + d * Math.sin(t)]);
    }
    nucs.push({ x, y, a, b, ang, I, nucleoli, group });
  };
  // Isolated nuclei.
  add(48, 46, 14, 11, 0.3, 165);
  add(204, 40, 12, 12, 0, 185);
  add(132, 120, 13, 10, -0.5, 150);
  add(40, 206, 11, 9, 1.0, 175, 0, 1);
  add(212, 214, 13, 11, 0.8, 160);
  // A touching pair (group 1).
  add(108, 42, 12, 10, 0.1, 170, 1);
  add(131, 47, 11, 10, -0.2, 160, 1);
  // A cluster of three (group 2).
  add(102, 200, 13, 11, 0.2, 175, 2);
  add(125, 210, 12, 11, -0.4, 160, 2);
  add(112, 225, 11, 10, 0.6, 185, 2);
  // The cell: dim cytoplasm with one nucleus inside (group 3).
  const cell = { x: 200, y: 128, a: 38, b: 26, ang: -0.35, I: 55 };
  add(206, 126, 11, 9, -0.35, 170, 3);
  // Debris: small, very bright specks.
  const debris: [number, number, number][] = [[60, 128, 3], [76, 140, 2.5], [160, 170, 3], [24, 98, 2.5], [170, 78, 2]];

  const img = new Float32Array(N);
  const part = new Int32Array(N), object = new Int32Array(N), group = new Int32Array(N);
  let nextGroup = 1;
  const groupId: Record<number, number> = {};
  const gOf = (n: Nuc, i: number) => n.group === 0 ? (groupId[1000 + i] ??= nextGroup++) : (groupId[n.group] ??= nextGroup++);
  let partId = 0;
  const partOf: number[][] = nucs.map((n) => n.nucleoli.map(() => ++partId));
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const p = y * W + x;
    // Background: uneven illumination.
    let v = 18 + 14 * (x / W) + 8 * Math.sin(y / 40);
    const cin = soft(x, y, cell.x, cell.y, cell.a, cell.b, cell.ang, 4);
    v += cell.I * cin;
    if (cin > 0.5) group[p] = gOf(nucs[nucs.length - 1], nucs.length - 1);
    nucs.forEach((n, i) => {
      const s = soft(x, y, n.x, n.y, n.a, n.b, n.ang);
      if (s < 0.01) return;
      // Slightly darker rim, brighter nucleoli: texture inside.
      let inner = n.I * (0.7 + 0.3 * soft(x, y, n.x, n.y, n.a - 3, n.b - 3, n.ang, 2));
      n.nucleoli.forEach(([qx, qy], k) => {
        const d2 = (x - qx) ** 2 + (y - qy) ** 2, q = Math.exp(-d2 / (2 * 1.6 * 1.6));
        inner += 55 * q;
        if (s > 0.5 && d2 <= 2.4 * 2.4) part[p] = partOf[i][k];
      });
      v = Math.max(v, v * (1 - s) + inner * s);
      if (s > 0.5) { object[p] = i + 1; group[p] = gOf(n, i); }
    });
    for (const [dx, dy, dr] of debris) {
      const s = soft(x, y, dx, dy, dr, dr, 0, 0.8);
      v = v * (1 - s) + 235 * s;
    }
    img[p] = v;
  }
  // Noise (Gaussian, seeded).
  for (let p = 0; p < N; p++) img[p] = Math.min(255, Math.max(0, img[p] + 5 * r.norm()));
  const nObjects = nucs.length;
  const pair0 = nucs[5], pair1 = nucs[6];
  const spots: Record<string, [number, number]> = {
    // On the outer (left) edge of the left nucleus of the touching pair.
    edge: [Math.round(pair0.x - pair0.a + 1), Math.round(pair0.y - 1)],
    // On the seam between the two nuclei of the touching pair.
    between: [Math.round((pair0.x + pair0.a * Math.cos(pair0.ang) + pair1.x - pair1.a) / 2), Math.round((pair0.y + pair1.y) / 2)],
    // The centre of the right nucleus of the pair (for the negative click).
    pairRight: [Math.round(pair1.x), Math.round(pair1.y)],
    // Inside a nucleus of the cluster of three.
    cluster: [102, 200],
    cellNucleus: [206, 126],
    cytoplasm: [184, 140],
  };
  /** A box around the left nucleus of the pair. */
  const box: [number, number, number, number] = [Math.round(pair0.x - pair0.a - 3), Math.round(pair0.y - pair0.b - 3), Math.round(pair0.x + pair0.a + 2), Math.round(pair0.y + pair0.b + 3)];
  return { img, part, object, group, nObjects, spots, box };
}
