// Pure measurement helpers of the meshmeasure demo (no DOM; used by the demo and by the
// scripts that compute the exercise numbers).
import { marchingSquares, marchingCubes, contourLength, contourArea } from "../lib/marching";
import { meshArea, meshVolume } from "../lib/mesh";
import { Volume, volumeFromFunction, gaussianBlur3d, padVolume } from "../lib/volume";

export const OFF = [0.23, 0.37, 0.11];   // centre offset from a pixel centre (avoids exact ties)


/** Grey (partial-volume ramp) or binary disk of radius R pixels on an n × n grid. */
export function diskImage(R: number, binary: boolean): { v: Float64Array; n: number; c: [number, number] } {
  const n = Math.ceil(2 * R) + 6, c: [number, number] = [(n - 1) / 2 + OFF[0], (n - 1) / 2 + OFF[1]], v = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const d = R - Math.hypot(x - c[0], y - c[1]);
    v[y * n + x] = binary ? (d >= 0 ? 1 : 0) : Math.max(0, Math.min(1, d + 0.5));
  }
  return { v, n, c };
}

export function diskMeasures(R: number, binary: boolean) {
  const { v, n } = diskImage(R, binary);
  let edges = 0, pixels = 0;
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < n && y < n && v[y * n + x] > 0.5;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (inside(x, y)) {
    pixels++;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) if (!inside(x + dx, y + dy)) edges++;
  }
  const C = marchingSquares(v, n, n, 0.5);
  return { edges, pixels, msLength: contourLength(C), msArea: contourArea(C), trueL: 2 * Math.PI * R, trueA: Math.PI * R * R };
}

export function ballVolumeImage(R: number, binary: boolean): Volume {
  const n = Math.ceil(2 * R) + 6, c = (n - 1) / 2;
  return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => {
    const d = R - Math.hypot(x - c - OFF[0], y - c - OFF[1], z - c - OFF[2]);
    return binary ? (d >= 0 ? 1 : 0) : Math.max(0, Math.min(1, d + 0.5));
  });
}

export function ballMeasures(R: number, binary: boolean) {
  const v = ballVolumeImage(R, binary), n = v.nx;
  let faces = 0, voxels = 0;
  const inside = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < n && y < n && z < n && v.data[(z * n + y) * n + x] > 0.5;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (inside(x, y, z)) {
    voxels++;
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) if (!inside(x + dx, y + dy, z + dz)) faces++;
  }
  const m = marchingCubes(v.data, n, n, n, 0.5);
  return { faces, voxels, mcArea: meshArea(m), mcVolume: meshVolume(m), trueA: 4 * Math.PI * R * R, trueV: 4 / 3 * Math.PI * R ** 3 };
}

/** A ball of radius 8 voxels (0/1) blurred with σ voxels, padded. */
export function blurredBall(sigma: number): Volume {
  const v = ballVolumeImage(8, true);
  return padVolume(sigma > 0 ? gaussianBlur3d(v, sigma) : v, 1, 0);
}

export function volumeVsC(v: Volume, cs: number[]): number[] {
  return cs.map(c => meshVolume(marchingCubes(v.data, v.nx, v.ny, v.nz, c, { spacing: v.spacing })));
}

