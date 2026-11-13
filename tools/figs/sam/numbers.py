"""Numbers for sam/ (lecture, exercise by-hand and math questions). Run: python3 -I tools/figs/sam/numbers.py
Timings: demos/wasm/sam/README.md (headless Chrome, 2026-11-13): encoder 26.6 s once, decoder
0.42 s per click (three masks), 0.33 s per point in automatic mode. Same numbers as
plan/cellnets-day.md (scratchpad goal3/cellnets/numbers.py), re-derived here."""
import math

def sect(t): print("\n=== " + t)

enc, dec_click, dec_pt = 26.6, 0.42, 0.33

sect("B3 / lecture: cached embedding vs re-encoding")
for m in (1, 10, 40):
    cached, re = enc + m * dec_click, m * (enc + dec_click)
    print(f"{m:3d} clicks: cached {cached:.1f} s, re-encode {re:.1f} s, speed-up {re / cached:.2f}x")
print("limit of the speed-up (m -> inf): (Te + Td)/Td = %.1f" % ((enc + dec_click) / dec_click))
for g in (8, 16, 32):
    print(f"grid {g}x{g}: {g*g} points, {3*g*g} candidates, decoder {g*g*dec_pt:.1f} s")

sect("B7 (math): break-even and speed-up")
print("saving = (m-1) Te; equal at m = 1; speed-up S(m) = m(Te+Td)/(Te+m Td)")
for m in (2, 5, 64, 1000):
    print(f"  S({m}) = {m*(enc+dec_click)/(enc+m*dec_click):.2f}")
print("  m for S = 32 (half the limit): m = Te/Td = %.1f" % (enc / dec_click))

sect("B2: stability")
print("380/430 = %.3f ; 400/412 = %.3f" % (380 / 430, 400 / 412))

sect("B4: embedding resolution (1024 input, 64 x 64 embedding)")
print("one position covers 1024/64 = %.0f px of the resized image" % (1024 / 64))
for side in (512, 696, 1024, 256):
    print(f"  image side {side}: {side/64:.2f} px per embedding position")
print("  a 6 px nucleus in a 512 image spans %.2f positions" % (6 / 8))
print("  decoder logits 256 x 256: %.1f px of a 512 image per logit" % (512 / 256))

sect("B5: grid points inside nuclei covering 15 % of the image")
print("8x8: %.1f, 16x16: %.1f, 32x32: %.1f" % (64 * 0.15, 256 * 0.15, 1024 * 0.15))
print("grid spacing in a 512 image: 8x8 %.0f px, 16x16 %.0f px" % (512 / 8, 512 / 16))

sect("B8 (math): average vs minimum loss, two possible true masks")
# Click on a nucleus inside a cluster: truth = nucleus A (410 px) with prob 1-q, cluster B (860 px) with prob q.
A, B, q = 410, 860, 0.4
ring = B - A
# Squared loss per pixel; average-loss optimum per output: y = P(pixel in truth).
avg_opt_ring = q
loss_avg = ring * q * (1 - q)
print(f"ring pixels {ring}; averaged-loss optimum predicts {avg_opt_ring} on the ring (all three outputs)")
print(f"expected loss per output at that optimum: {ring} * q(1-q) = {loss_avg:.0f}")
print("minimum over outputs with outputs (A, B, any): expected loss 0")

sect("B9 (math): attention matrix sizes")
N, k = 64 * 64, 6
print(f"encoder N^2 = {N*N:,}; decoder k N = {k*N:,}; ratio N/k = {N/k:.1f}")
print("embedding 256 x 64 x 64 floats = %d = %.1f MB in float32" % (256 * N, 256 * N * 4 / 2**20))
