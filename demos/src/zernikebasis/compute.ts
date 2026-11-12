// Pure computations of the zernikebasis demo (no DOM), shared with the scratch scripts that
// compute the numbers quoted in the zernike exercise: test images drawn from geometry (a blob
// outline, a stroke "6", a letter F) on a 64 × 64 grid, rotated or mirrored about the centre
// of the inscribed disc before rasterising (anti-aliased), and their Zernike moments.
import { discGrid, zernikeMoments, polygonCoverageEvenOdd, ZernikeSet, DiscGrid } from "../lib/moments";

export const S = 64;
export const NZ = 8;
export type Pt = [number, number];

/** Optics names of the low-order Zernike polynomials (|m| as given). */
export function zernikeName(n: number, m: number): string {
  const a = Math.abs(m);
  const names: Record<string, string> = {
    "0,0": "piston", "1,1": "tilt", "2,0": "defocus", "2,2": "astigmatism", "3,1": "coma", "3,3": "trefoil",
    "4,0": "spherical aberration", "4,2": "secondary astigmatism", "4,4": "quadrafoil", "5,1": "secondary coma",
    "5,3": "secondary trefoil", "6,0": "secondary spherical",
  };
  return names[`${n},${a}`] ?? "";
}

export type ShapeId = "blob" | "six" | "F";

/** An irregular blob (no symmetry), centred on the disc. */
export function blobPoly(): Pt[] {
  return Array.from({ length: 120 }, (_, i) => {
    const t = 2 * Math.PI * i / 120;
    const r = 17 * (1 + 0.28 * Math.cos(t - 2.6) + 0.16 * Math.cos(2 * t + 0.9) + 0.12 * Math.cos(3 * t - 0.4));
    return [32 + r * Math.cos(t), 32 + r * Math.sin(t)] as Pt;
  });
}

export const LETTER_F: Pt[] = [[20, 12], [46, 12], [46, 20], [29, 20], [29, 29], [42, 29], [42, 37], [29, 37], [29, 52], [20, 52]];

/** Centre line of a "6": a loop and a hook (in pixel units, y down). */
export function sixStroke(): Pt[] {
  const pts: Pt[] = [];
  const bez = (a: Pt, b: Pt, c: Pt, n: number) => { for (let i = 0; i <= n; i++) { const t = i / n, u = 1 - t; pts.push([u * u * a[0] + 2 * u * t * b[0] + t * t * c[0], u * u * a[1] + 2 * u * t * b[1] + t * t * c[1]]); } };
  bez([40, 13], [20, 13], [23.5, 38], 24);
  // loop: circle of radius 9 around (32.5, 40), starting at its left side, going down and round
  for (let i = 1; i <= 48; i++) { const t = Math.PI + 2 * Math.PI * i / 48; pts.push([32.5 + 9 * Math.cos(t), 40 - 9 * Math.sin(t)]); }
  return pts;
}

/** Anti-aliased image of a thick polyline (distance ≤ hw), k × k samples per pixel. */
export function strokeImage(line: Pt[], hw: number, k = 4): Float64Array {
  const out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let hit = 0;
    for (let sy = 0; sy < k; sy++) for (let sx = 0; sx < k; sx++) {
      const px = x + (sx + 0.5) / k, py = y + (sy + 0.5) / k;
      for (let i = 0; i + 1 < line.length; i++) {
        const [ax, ay] = line[i], [bx, by] = line[i + 1], dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
        const t = L2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2)) : 0;
        if ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2 <= hw * hw) { hit++; break; }
      }
    }
    out[y * S + x] = hit / (k * k);
  }
  return out;
}

/** Rotate a point counter-clockwise on screen by α about the disc centre (32, 32); then mirror
 *  left–right if asked. */
export function moveFn(alpha: number, mirror: boolean): (p: Pt) => Pt {
  const c = Math.cos(alpha), s = Math.sin(alpha);
  return ([x, y]) => {
    const dx = x - 32, dy = y - 32;
    let u = c * dx + s * dy, v = -s * dx + c * dy;
    if (mirror) u = -u;
    return [32 + u, 32 + v];
  };
}

export function shapeImage(id: ShapeId, alpha = 0, mirror = false): Float64Array {
  const mv = moveFn(alpha, mirror);
  if (id === "six") return strokeImage(sixStroke().map(mv), 3.6);
  const poly = (id === "blob" ? blobPoly() : LETTER_F).map(mv);
  return polygonCoverageEvenOdd([poly], S, S, 4);
}

let grid: DiscGrid | null = null;
export function zernikeOf(f: Float64Array): ZernikeSet {
  grid ??= discGrid(S, S);
  return zernikeMoments(f, grid, NZ);
}
