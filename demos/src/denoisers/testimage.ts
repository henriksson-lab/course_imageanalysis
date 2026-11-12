// The denoisers test image (no DOM): a step, a ramp crossed by a 1-pixel line, a repeated
// dot lattice and one unique spot on a flat region; grey levels 0..255, 128 × 128.
const N = 128;

/** The synthetic test image (grey levels 0..255). */
export function testImage(): Float64Array {
  const x = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let i = 0; i < N; i++) {
    let v: number;
    if (y < 64 && i < 64) v = i < 32 ? 60 : 180;                                  // step
    else if (y < 64) v = 40 + 180 * (i - 64) / 63 + (i === 96 ? 80 : 0);          // ramp crossed by a 1-px line
    else if (i < 64) v = ((i % 6) < 2 && (y % 6) < 2) ? 200 : 80;                 // repeated dots
    else v = 120 + (Math.abs(i - 96) <= 1 && Math.abs(y - 98) <= 1 ? 100 : 0);    // flat with one unique spot
    x[y * N + i] = Math.min(255, v);
  }
  return x;
}

