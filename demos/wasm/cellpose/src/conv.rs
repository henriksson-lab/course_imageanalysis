//! 2D convolution (1x1 or 3x3, zero padding, stride 1) as a matrix product, with the batch
//! norm + ReLU that precedes every convolution in the Cellpose U-net fused into the packing
//! of the input ("im2col"). Activations are channel-major (C, H, W) f32 arrays.
//!
//! The product runs in MR (output channels) x 8 (pixels) blocks. On wasm32 the block kernel
//! uses 128-bit SIMD (`simd128`, enabled per function so the crate needs no RUSTFLAGS; every
//! current browser supports it); elsewhere a plain loop the compiler vectorises.

/// Weights of one convolution, packed for the block kernel: panels of MR output channels,
/// each `k_len` x MR (k = input channel * ksize^2 + ky * ksize + kx).
pub struct ConvW {
    pub cout: usize,
    pub cin: usize,
    pub ksize: usize,
    pub k_len: usize,
    pub wp: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ConvW {
    /// `w` in PyTorch layout (cout, cin, ksize, ksize).
    pub fn new(w: &[f32], bias: &[f32], cout: usize, cin: usize, ksize: usize) -> ConvW {
        let k_len = cin * ksize * ksize;
        assert_eq!(w.len(), cout * k_len);
        let ncb = (cout + MR - 1) / MR;
        let mut wp = vec![0f32; ncb * k_len * MR];
        for co in 0..cout {
            let (cb, j) = (co / MR, co % MR);
            for k in 0..k_len {
                wp[cb * k_len * MR + k * MR + j] = w[co * k_len + k];
            }
        }
        ConvW { cout, cin, ksize, k_len, wp, bias: bias.to_vec() }
    }
}

/// Per-input-channel affine map applied before the convolution: v -> s*v + t, then ReLU if
/// `relu`. Zero padding is applied after it (as in PyTorch, where the padding belongs to the
/// convolution).
pub struct Pre<'a> {
    pub scale: &'a [f32],
    pub shift: &'a [f32],
    pub relu: bool,
}

/// Pack pixels [p0, p0 + nb*8) of the (already activated) input into `col` as nb blocks of
/// k_len x 8; taps outside the image are 0 (zero padding).
fn pack(input: &[f32], h: usize, w: usize, cw: &ConvW, p0: usize, nb: usize, col: &mut [f32]) {
    let hw = h * w;
    let ks = cw.ksize;
    let r = (ks / 2) as isize;
    let kl = cw.k_len;
    for b in 0..nb {
        let blk = &mut col[b * kl * 8..(b + 1) * kl * 8];
        let pstart = p0 + b * 8;
        let (y0, x0) = (pstart / w, pstart % w);
        // fast path: all 8 pixels in one row and every tap inside the image horizontally
        let same_row = pstart + 8 <= hw && x0 + 8 <= w;
        let interior_x = x0 as isize - r >= 0 && (x0 + 8) as isize + r <= w as isize;
        if same_row && interior_x {
            for ci in 0..cw.cin {
                let plane = &input[ci * hw..(ci + 1) * hw];
                for ky in 0..ks {
                    let yy = y0 as isize + ky as isize - r;
                    let dst = &mut blk[(ci * ks + ky) * ks * 8..][..ks * 8];
                    if yy < 0 || yy >= h as isize {
                        dst.fill(0.0);
                    } else {
                        let row = yy as usize * w;
                        for kx in 0..ks {
                            let src = &plane[row + (x0 as isize + kx as isize - r) as usize..][..8];
                            dst[kx * 8..kx * 8 + 8].copy_from_slice(src);
                        }
                    }
                }
            }
        } else {
            let mut ys = [0isize; 8];
            let mut xs = [0isize; 8];
            let mut valid = [false; 8];
            for l in 0..8 {
                let p = pstart + l;
                valid[l] = p < hw;
                ys[l] = (p / w) as isize;
                xs[l] = (p % w) as isize;
            }
            for ci in 0..cw.cin {
                let plane = &input[ci * hw..(ci + 1) * hw];
                for ky in 0..ks {
                    for kx in 0..ks {
                        let dst = &mut blk[((ci * ks + ky) * ks + kx) * 8..][..8];
                        for l in 0..8 {
                            let yy = ys[l] + ky as isize - r;
                            let xx = xs[l] + kx as isize - r;
                            dst[l] = if valid[l] && yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize {
                                plane[yy as usize * w + xx as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
    }
}

/// Output channels per block of the matrix product.
pub const MR: usize = 4;

/// acc[MR x 8] += wpanel (k_len x MR)^T * colblk (k_len x 8).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn kernel(kl: usize, a: *const f32, b: *const f32, acc: &mut [[f32; 8]; MR]) {
    use core::arch::wasm32::*;
    let mut c0 = [f32x4_splat(0.0); MR];
    let mut c1 = [f32x4_splat(0.0); MR];
    for j in 0..MR {
        c0[j] = v128_load(acc[j].as_ptr() as *const v128);
        c1[j] = v128_load(acc[j].as_ptr().add(4) as *const v128);
    }
    let mut pa = a;
    let mut pb = b;
    for _ in 0..kl {
        let b0 = v128_load(pb as *const v128);
        let b1 = v128_load(pb.add(4) as *const v128);
        for j in 0..MR {
            let aj = v128_load32_splat(pa.add(j) as *const u32);
            c0[j] = f32x4_add(c0[j], f32x4_mul(aj, b0));
            c1[j] = f32x4_add(c1[j], f32x4_mul(aj, b1));
        }
        pa = pa.add(MR);
        pb = pb.add(8);
    }
    for j in 0..MR {
        v128_store(acc[j].as_mut_ptr() as *mut v128, c0[j]);
        v128_store(acc[j].as_mut_ptr().add(4) as *mut v128, c1[j]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn kernel(kl: usize, a: *const f32, b: *const f32, acc: &mut [[f32; 8]; MR]) {
    let a = std::slice::from_raw_parts(a, kl * MR);
    let b = std::slice::from_raw_parts(b, kl * 8);
    for k in 0..kl {
        for j in 0..MR {
            let av = a[k * MR + j];
            for l in 0..8 {
                acc[j][l] += av * b[k * 8 + l];
            }
        }
    }
}

/// Convolution of `input` (cin, h, w) with `cw`, after the per-channel map `pre`.
/// Writes (or, with `accumulate`, adds to) `out` (cout, h, w). `col` is scratch space.
pub fn conv(input: &[f32], h: usize, w: usize, pre: &Pre, cw: &ConvW, out: &mut [f32], accumulate: bool, col: &mut Vec<f32>) {
    let hw = h * w;
    assert!(input.len() >= cw.cin * hw && out.len() >= cw.cout * hw);
    // activated input, computed once (not once per tap)
    let mut xa = vec![0f32; cw.cin * hw];
    for ci in 0..cw.cin {
        let (s, t) = (pre.scale[ci], pre.shift[ci]);
        let src = &input[ci * hw..(ci + 1) * hw];
        let dst = &mut xa[ci * hw..(ci + 1) * hw];
        if pre.relu {
            for i in 0..hw {
                dst[i] = (src[i] * s + t).max(0.0);
            }
        } else {
            for i in 0..hw {
                dst[i] = src[i] * s + t;
            }
        }
    }
    let kl = cw.k_len;
    let nblocks = (hw + 7) / 8;
    // pixel blocks per band: packed band about 512 kB
    let band = (131072 / (kl * 8)).clamp(1, 512);
    if col.len() < band * kl * 8 {
        col.resize(band * kl * 8, 0.0);
    }
    let ncb = (cw.cout + MR - 1) / MR;
    let mut acc = [[0f32; 8]; MR];
    let mut b0 = 0;
    while b0 < nblocks {
        let nb = band.min(nblocks - b0);
        pack(&xa, h, w, cw, b0 * 8, nb, col);
        for cb in 0..ncb {
            let panel = &cw.wp[cb * kl * MR..(cb + 1) * kl * MR];
            for b in 0..nb {
                let p = (b0 + b) * 8;
                let nvalid = 8.min(hw - p);
                for j in 0..MR {
                    let co = cb * MR + j;
                    if co < cw.cout {
                        let bias = cw.bias[co];
                        for l in 0..8 {
                            acc[j][l] = if accumulate && l < nvalid { out[co * hw + p + l] + bias } else { bias };
                        }
                    } else {
                        acc[j] = [0.0; 8];
                    }
                }
                unsafe { kernel(kl, panel.as_ptr(), col[b * kl * 8..].as_ptr(), &mut acc) };
                for j in 0..MR {
                    let co = cb * MR + j;
                    if co < cw.cout {
                        out[co * hw + p..co * hw + p + nvalid].copy_from_slice(&acc[j][..nvalid]);
                    }
                }
            }
        }
        b0 += nb;
    }
}
