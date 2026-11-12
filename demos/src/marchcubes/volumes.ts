// The volumes of the marchcubes demo (no DOM; also used by the scripts that compute the
// exercise numbers).
import { Volume, metaballs, volumeFromFunction, softStep, ellipsoidVolume, nucleiStack, rotY } from "../lib/volume";

export type VolumeId = "blobs" | "torus" | "sphere" | "ellipsoid" | "nuclei";
export const VOLUME_NAMES: Record<VolumeId, string> = { blobs: "two blobs", torus: "torus", sphere: "sphere", ellipsoid: "ellipsoid (Δz = 2.5)", nuclei: "nuclei stack" };

/** The clean (noise-free) volumes, values about 0..1, and their true spacing. */
export function makeVolume(id: VolumeId): Volume {
  const n = 48, mid = (n - 1) / 2;
  switch (id) {
    case "blobs": return metaballs(n, [[mid - 10, mid, mid], [mid + 10, mid, mid]], 6);
    case "torus": return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => softStep(5.5 - Math.hypot(Math.hypot(x - mid, y - mid) - 13, z - mid), 1));
    case "sphere": return volumeFromFunction(n, n, n, [1, 1, 1], (x, y, z) => softStep(15 - Math.hypot(x - mid, y - mid, z - mid), 1));
    case "ellipsoid": {
      const v = ellipsoidVolume(n, n, 20, [1, 1, 2.5], [{ centre: [mid, mid, 23.75], radii: [17, 9, 12], rot: rotY(35 * Math.PI / 180) }], { edge: 1 });
      return v;
    }
    case "nuclei": {
      const v = nucleiStack({ noise: 0 });
      for (let i = 0; i < v.data.length; i++) v.data[i] /= 200;
      return v;
    }
  }
}

