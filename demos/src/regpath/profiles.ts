// The 1D test profiles of the regpath demo (no DOM), 256 samples, values about 0..1.
import { phantom, render } from "../deconvlab/scenes";

const n = 256;
export type ProfileId = "stepramp" | "peaks" | "cell";

/** step and ramp; two close peaks, a narrow peak and a plateau (baseline 0); a line across the cell phantom. */
export function truthOf(id: ProfileId): Float64Array {
  const x = new Float64Array(n);
  if (id === "stepramp") for (let i = 0; i < n; i++) x[i] = 0.1 + (i >= 50 && i < 100 ? 0.7 : 0) + (i >= 130 && i < 200 ? 0.7 * (i - 130) / 70 : 0);
  else if (id === "peaks") for (let i = 0; i < n; i++) x[i] = 0.8 * Math.exp(-((i - 80) ** 2) / (2 * 1.5 ** 2)) + 0.8 * Math.exp(-((i - 91) ** 2) / (2 * 1.5 ** 2)) + 1.0 * Math.exp(-((i - 170) ** 2) / (2 * 0.7 ** 2)) + (i >= 200 && i < 230 ? 0.4 : 0);
  else { const img = render(phantom(n), n); for (let i = 0; i < n; i++) x[i] = img[132 * n + i]; }
  return x;
}

