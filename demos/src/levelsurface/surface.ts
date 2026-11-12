// φ of a level-set grid as a 3D height field (WebGL2 via lib/gl3d.ts): diverging colours
// (blue φ < 0, red φ > 0), heights clipped to ±clip, a translucent plane at φ = 0 and the zero
// contour (marching squares) drawn on it. Used by curveflow's "surface" view.
import { GLView, MeshObject, LinesObject, heightFieldMesh, MeshData } from "../lib/gl3d";
import { diverging, divergingT } from "../lib/colour";
import { marchingSquares } from "../lib/marching";

export class PhiSurface {
  private surf: MeshObject;
  private plane: MeshObject;
  private zero: LinesObject;
  constructor(view: GLView, private clip = 20, private zs = 1 / 64, private cmax = 20) {
    this.surf = view.addMesh({ positions: [0, 0, 0], indices: [] }, { shading: "smooth", lighting: "lambert", vertexColors: true });
    this.plane = view.addMesh({ positions: [-0.5, 0, -0.5, 0.5, 0, -0.5, 0.5, 0, 0.5, -0.5, 0, 0.5], indices: [0, 2, 1, 0, 3, 2] }, { color: [0.95, 0.7, 0.2], lighting: "unlit", opacity: 0.3 });
    this.zero = view.addLines({ positions: [] }, { width: 4, color: [0.05, 0.05, 0.05] });
  }

  /** Show a w × h grid, subsampled by `step`. */
  set(phi: ArrayLike<number>, w: number, h: number, step = 2): void {
    const nw = Math.floor((w - 1) / step) + 1, nh = Math.floor((h - 1) / step) + 1;
    const v = new Float64Array(nw * nh);
    for (let j = 0; j < nh; j++) for (let i = 0; i < nw; i++) v[j * nw + i] = phi[j * step * w + i * step];
    const c = this.clip, clipped = Float64Array.from(v, x => Math.max(-c, Math.min(c, x)));
    const m = heightFieldMesh(clipped, nw, nh, { scale: 2 * c * this.zs, lo: -c, hi: c });
    const P = m.positions as Float32Array, C = new Float32Array(P.length);
    for (let p = 0; p < nw * nh; p++) {
      P[3 * p + 1] -= c * this.zs;
      const col = diverging(divergingT(v[p], this.cmax));
      C[3 * p] = col[0]; C[3 * p + 1] = col[1]; C[3 * p + 2] = col[2];
    }
    this.surf.setData({ ...m, colors: C } as MeshData);
    const Z = marchingSquares(v, nw, nh, 0), L = Math.max(nw, nh) - 1, out: number[] = [];
    for (let s = 0; s < Z.segments.length; s += 2) for (const id of [Z.segments[s], Z.segments[s + 1]]) {
      out.push((Z.points[2 * id] - (nw - 1) / 2) / L, 0.003, (Z.points[2 * id + 1] - (nh - 1) / 2) / L);
    }
    this.zero.setData({ positions: out });
  }
}
