//! A second implementation of the same StarDist 2D U-Net forward pass, written for the browser:
//! direct convolutions over channels-last (HWC) arrays with 128-bit SIMD (Wasm `simd128`,
//! enabled per function, so the build needs no special flags). The candle network of
//! stardist-rs is about 3x slower in Wasm (no SIMD, since the build cannot enable it for
//! candle's matrix kernels); this file exists only for speed. Weights come from the same
//! `KerasWeights`, the layer sequence follows `stardist_rs::model::candle::StarDist2D::forward`
//! (pre-pool convs, U-Net down/middle/up with "upsample then concat [up, skip]", features,
//! prob = sigmoid, dist = linear), and tests/native.rs checks it against candle.
//! On other targets the same code runs with a plain [f32; 4] vector type (same arithmetic order).
use stardist_rs::{Config2D, KerasWeights};

/// Feature map, channels last.
#[derive(Clone)]
pub struct Map {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

/// One conv layer, weights rearranged to [ky][kx][ci][co].
pub struct Conv {
    k: usize,
    ci: usize,
    co: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv {
    fn from_keras(weights: &KerasWeights, name: &str) -> Result<Conv, String> {
        let kern = weights.get(&format!("{name}/{name}/kernel:0")).ok_or(format!("missing {name}"))?;
        let bias = weights.get(&format!("{name}/{name}/bias:0")).ok_or(format!("missing {name} bias"))?;
        // KerasWeights holds kernels as [co][ci][kh][kw] (converted on load).
        let [co, ci, kh, kw] = <[usize; 4]>::try_from(kern.shape.as_slice()).map_err(|_| format!("{name}: not 2D"))?;
        if kh != kw || kh % 2 == 0 {
            return Err(format!("{name}: kernel {kh}x{kw} unsupported"));
        }
        let mut w = vec![0f32; co * ci * kh * kw];
        for o in 0..co {
            for i in 0..ci {
                for y in 0..kh {
                    for x in 0..kw {
                        w[((y * kw + x) * ci + i) * co + o] = kern.values[((o * ci + i) * kh + y) * kw + x];
                    }
                }
            }
        }
        Ok(Conv { k: kh, ci, co, w, b: bias.values.clone() })
    }
}

pub struct FastNet {
    pre: Vec<Conv>,
    n_pre_pools: usize,
    per_depth: usize,
    down: Vec<Vec<Conv>>,
    middle: Vec<Conv>,
    up: Vec<Vec<Conv>>,
    features: Option<Conv>,
    prob: Conv,
    dist: Conv,
}

impl FastNet {
    /// Same layer names and counts as the candle model (unet backbone, factor-2 grid pooling).
    pub fn new(config: &Config2D, weights: &KerasWeights) -> Result<FastNet, String> {
        if config.backbone != "unet" || config.unet_batch_norm || config.n_classes.is_some() {
            return Err("fast path: only plain U-Net StarDist 2D models".into());
        }
        if config.grid[0] != config.grid[1] || !config.grid[0].is_power_of_two() || config.unet_pool != [2, 2] {
            return Err("fast path: grid and pooling must be equal powers of two".into());
        }
        let n_pre_pools = config.grid[0].trailing_zeros() as usize;
        let per = config.unet_n_conv_per_depth;
        let depth = config.unet_n_depth;
        let mut idx = 1;
        let mut pre = Vec::new();
        for _ in 0..n_pre_pools * per {
            pre.push(Conv::from_keras(weights, &format!("conv2d_{idx}"))?);
            idx += 1;
        }
        let mut down = Vec::new();
        for l in 0..depth {
            down.push((0..per).map(|i| Conv::from_keras(weights, &format!("down_level_{l}_no_{i}"))).collect::<Result<Vec<_>, _>>()?);
        }
        let name_last = |i: usize| if i + 1 == per { per } else { i };
        let middle = (0..per).map(|i| Conv::from_keras(weights, &format!("middle_{}", name_last(i)))).collect::<Result<Vec<_>, _>>()?;
        let mut up = Vec::new();
        for l in (0..depth).rev() {
            up.push((0..per).map(|i| Conv::from_keras(weights, &format!("up_level_{l}_no_{}", name_last(i)))).collect::<Result<Vec<_>, _>>()?);
        }
        let features = if config.net_conv_after_unet > 0 { Some(Conv::from_keras(weights, "features")?) } else { None };
        Ok(FastNet {
            pre,
            n_pre_pools,
            per_depth: per,
            down,
            middle,
            up,
            features,
            prob: Conv::from_keras(weights, "prob")?,
            dist: Conv::from_keras(weights, "dist")?,
        })
    }

    /// x: (h, w, 1), h and w divisible by grid * 2^depth. Returns prob (h/g, w/g, 1) and
    /// dist (h/g, w/g, n_rays), channels last.
    pub fn forward(&self, x: Map) -> (Map, Map) {
        let mut m = x;
        let mut pi = 0;
        for _ in 0..self.n_pre_pools {
            for _ in 0..self.per_depth {
                m = conv(&m, &self.pre[pi], true);
                pi += 1;
            }
            m = maxpool2(&m);
        }
        let mut skips = Vec::new();
        for level in &self.down {
            for c in level {
                m = conv(&m, c, true);
            }
            skips.push(m.clone());
            m = maxpool2(&m);
        }
        for c in &self.middle {
            m = conv(&m, c, true);
        }
        for (level, skip) in self.up.iter().zip(skips.iter().rev()) {
            m = concat(&upsample2(&m), skip);
            for c in level {
                m = conv(&m, c, true);
            }
        }
        if let Some(f) = &self.features {
            m = conv(&m, f, true);
        }
        let mut prob = conv(&m, &self.prob, false);
        for v in prob.d.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        let dist = conv(&m, &self.dist, false);
        (prob, dist)
    }
}

fn maxpool2(m: &Map) -> Map {
    let (h, w, c) = (m.h / 2, m.w / 2, m.c);
    let mut d = vec![0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            for k in 0..c {
                let a = |yy: usize, xx: usize| m.d[(yy * m.w + xx) * c + k];
                d[(y * w + x) * c + k] = a(2 * y, 2 * x).max(a(2 * y, 2 * x + 1)).max(a(2 * y + 1, 2 * x)).max(a(2 * y + 1, 2 * x + 1));
            }
        }
    }
    Map { h, w, c, d }
}

fn upsample2(m: &Map) -> Map {
    let (h, w, c) = (m.h * 2, m.w * 2, m.c);
    let mut d = vec![0f32; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let s = ((y / 2) * m.w + x / 2) * c;
            d[(y * w + x) * c..(y * w + x + 1) * c].copy_from_slice(&m.d[s..s + c]);
        }
    }
    Map { h, w, c, d }
}

fn concat(a: &Map, b: &Map) -> Map {
    assert!(a.h == b.h && a.w == b.w);
    let c = a.c + b.c;
    let mut d = Vec::with_capacity(a.h * a.w * c);
    for p in 0..a.h * a.w {
        d.extend_from_slice(&a.d[p * a.c..(p + 1) * a.c]);
        d.extend_from_slice(&b.d[p * b.c..(p + 1) * b.c]);
    }
    Map { h: a.h, w: a.w, c, d }
}

/// "Same" convolution with zero padding (+ bias, optional ReLU).
fn conv(m: &Map, cv: &Conv, relu: bool) -> Map {
    assert_eq!(m.c, cv.ci, "conv: channel mismatch");
    let r = cv.k / 2;
    // Zero border of r pixels, so the inner loops need no bounds checks.
    let (pw, ph) = (m.w + 2 * r, m.h + 2 * r);
    let padded = if r == 0 {
        m.d.clone()
    } else {
        let mut p = vec![0f32; pw * ph * m.c];
        for y in 0..m.h {
            let s = y * m.w * m.c;
            let t = ((y + r) * pw + r) * m.c;
            p[t..t + m.w * m.c].copy_from_slice(&m.d[s..s + m.w * m.c]);
        }
        p
    };
    let mut out = vec![0f32; m.h * m.w * cv.co];
    if cv.co % 8 == 0 {
        // simd128 is in every browser this course targets (Chrome 91+, Firefox 89+, Safari
        // 16.4+); without it the module fails to compile, it does not misbehave.
        conv_simd(&padded, pw, m.h, m.w, cv, &mut out);
    } else {
        conv_scalar(&padded, pw, m.h, m.w, cv, &mut out);
    }
    if relu {
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
    }
    Map { h: m.h, w: m.w, c: cv.co, d: out }
}

fn conv_scalar(p: &[f32], pw: usize, h: usize, w: usize, cv: &Conv, out: &mut [f32]) {
    let (k, ci, co) = (cv.k, cv.ci, cv.co);
    for y in 0..h {
        for x in 0..w {
            for o in 0..co {
                let mut s = cv.b[o];
                for ky in 0..k {
                    for kx in 0..k {
                        let ib = ((y + ky) * pw + x + kx) * ci;
                        let wb = (ky * k + kx) * ci;
                        for i in 0..ci {
                            s += p[ib + i] * cv.w[(wb + i) * co + o];
                        }
                    }
                }
                out[(y * w + x) * co + o] = s;
            }
        }
    }
}

// ---- 4-lane vector operations: Wasm SIMD, or plain arrays elsewhere --------------------------
// Macros rather than functions: the intrinsics must be inlined into the simd128-enabled
// function (a call to a helper without the target feature is not inlined and costs a call
// per multiply-add).
#[cfg(target_arch = "wasm32")]
macro_rules! vload { ($s:expr, $i:expr) => { core::arch::wasm32::v128_load($s.as_ptr().add($i) as *const core::arch::wasm32::v128) } }
#[cfg(target_arch = "wasm32")]
macro_rules! vstore { ($s:expr, $i:expr, $v:expr) => { core::arch::wasm32::v128_store($s.as_mut_ptr().add($i) as *mut core::arch::wasm32::v128, $v) } }
#[cfg(target_arch = "wasm32")]
macro_rules! vsplat { ($x:expr) => { core::arch::wasm32::f32x4_splat($x) } }
#[cfg(target_arch = "wasm32")]
macro_rules! vmadd { ($acc:expr, $a:expr, $b:expr) => { core::arch::wasm32::f32x4_add($acc, core::arch::wasm32::f32x4_mul($a, $b)) } }
#[cfg(not(target_arch = "wasm32"))]
macro_rules! vload { ($s:expr, $i:expr) => { { let i = $i; [$s[i], $s[i + 1], $s[i + 2], $s[i + 3]] } } }
#[cfg(not(target_arch = "wasm32"))]
macro_rules! vstore { ($s:expr, $i:expr, $v:expr) => { { let i = $i; $s[i..i + 4].copy_from_slice(&$v) } } }
#[cfg(not(target_arch = "wasm32"))]
macro_rules! vsplat { ($x:expr) => { [$x; 4] } }
#[cfg(not(target_arch = "wasm32"))]
macro_rules! vmadd { ($acc:expr, $a:expr, $b:expr) => { { let (c, a, b) = ($acc, $a, $b); [c[0] + a[0] * b[0], c[1] + a[1] * b[1], c[2] + a[2] * b[2], c[3] + a[3] * b[3]] } } }

/// Direct convolution, output channels in blocks of 4*V (V vectors) and P output pixels at a
/// time: P*V accumulators; per input channel and tap, V weight loads and P broadcasts feed
/// P*V vector multiply-adds. Generic form of `conv_4x2` below, which uses it for the last
/// 1-3 pixels of a row.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
fn conv_block<const P: usize, const V: usize>(p: &[f32], pw: usize, h: usize, w: usize, cv: &Conv, out: &mut [f32]) {
    let (k, ci, co) = (cv.k, cv.ci, cv.co);
    assert!(co % (4 * V) == 0 && p.len() >= ((h + k - 2) * pw + w + k - 1) * ci && cv.w.len() == k * k * ci * co);
    for y in 0..h {
        let mut x = 0;
        while x < w {
            let np = (w - x).min(P);
            for ob in (0..co).step_by(4 * V) {
                // SAFETY: indices stay inside the asserted lengths: pixel (y + ky, x + q + kx) of
                // the padded input with q < np, and channels ob..ob + 4V <= co.
                unsafe {
                    let mut acc = [[vload!(cv.b, ob); V]; P];
                    for q in 0..P {
                        for j in 0..V {
                            acc[q][j] = vload!(cv.b, ob + 4 * j);
                        }
                    }
                    for ky in 0..k {
                        for kx in 0..k {
                            let row = (y + ky) * pw + x + kx;
                            let wb = (ky * k + kx) * ci;
                            if np == P {
                                for i in 0..ci {
                                    let wo = (wb + i) * co + ob;
                                    let mut wv = [vsplat!(0.0); V];
                                    for j in 0..V {
                                        wv[j] = vload!(cv.w, wo + 4 * j);
                                    }
                                    for q in 0..P {
                                        let v = vsplat!(*p.get_unchecked((row + q) * ci + i));
                                        for j in 0..V {
                                            acc[q][j] = vmadd!(acc[q][j], v, wv[j]);
                                        }
                                    }
                                }
                            } else {
                                for i in 0..ci {
                                    let wo = (wb + i) * co + ob;
                                    for q in 0..np {
                                        let v = vsplat!(*p.get_unchecked((row + q) * ci + i));
                                        for j in 0..V {
                                            acc[q][j] = vmadd!(acc[q][j], v, vload!(cv.w, wo + 4 * j));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for q in 0..np {
                        let o = (y * w + x + q) * co + ob;
                        for j in 0..V {
                            vstore!(out, o + 4 * j, acc[q][j]);
                        }
                    }
                }
            }
            x += np;
        }
    }
}

/// The 4-pixel x 2-vector block with named accumulators (keeps them in registers).
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
fn conv_4x2(p: &[f32], pw: usize, h: usize, w: usize, cv: &Conv, out: &mut [f32]) {
    let (k, ci, co) = (cv.k, cv.ci, cv.co);
    assert!(co % 8 == 0 && p.len() >= ((h + k - 2) * pw + w + k - 1) * ci && cv.w.len() == k * k * ci * co);
    for y in 0..h {
        let mut x = 0;
        while x + 4 <= w {
            for ob in (0..co).step_by(8) {
                // SAFETY: as in conv_block (pixels x..x+4 < w, channels ob..ob+8 <= co).
                unsafe {
                    let b0 = vload!(cv.b, ob);
                    let b1 = vload!(cv.b, ob + 4);
                    let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = (b0, b1, b0, b1, b0, b1, b0, b1);
                    for ky in 0..k {
                        for kx in 0..k {
                            let base = ((y + ky) * pw + x + kx) * ci;
                            let mut wo = (ky * k + kx) * ci * co + ob;
                            let pp = p.as_ptr().add(base);
                            for i in 0..ci {
                                let w0 = vload!(cv.w, wo);
                                let w1 = vload!(cv.w, wo + 4);
                                let v0 = vsplat!(*pp.add(i));
                                let v1 = vsplat!(*pp.add(ci + i));
                                let v2 = vsplat!(*pp.add(2 * ci + i));
                                let v3 = vsplat!(*pp.add(3 * ci + i));
                                a00 = vmadd!(a00, v0, w0);
                                a01 = vmadd!(a01, v0, w1);
                                a10 = vmadd!(a10, v1, w0);
                                a11 = vmadd!(a11, v1, w1);
                                a20 = vmadd!(a20, v2, w0);
                                a21 = vmadd!(a21, v2, w1);
                                a30 = vmadd!(a30, v3, w0);
                                a31 = vmadd!(a31, v3, w1);
                                wo += co;
                            }
                        }
                    }
                    let o = (y * w + x) * co + ob;
                    vstore!(out, o, a00);
                    vstore!(out, o + 4, a01);
                    vstore!(out, o + co, a10);
                    vstore!(out, o + co + 4, a11);
                    vstore!(out, o + 2 * co, a20);
                    vstore!(out, o + 2 * co + 4, a21);
                    vstore!(out, o + 3 * co, a30);
                    vstore!(out, o + 3 * co + 4, a31);
                }
            }
            x += 4;
        }
        if x < w {
            // Remaining 1-3 pixels of the row.
            let sub_w = w - x;
            let mut tmp = vec![0f32; sub_w * co];
            conv_block::<1, 2>(&p[(y * pw + x) * ci..], pw, 1, sub_w, cv, &mut tmp);
            out[(y * w + x) * co..(y * w + w) * co].copy_from_slice(&tmp);
        }
    }
}

/// Measured in node on 512 x 512 (whole network): this 4 x 2 block with named accumulators
/// 1.9 s; the generic `conv_block` 3.3-3.9 s for (P, V) = (4, 2), (6, 2) and 7-9 s for V = 4
/// (accumulator arrays spill to memory); 6 x 2 named 2.0 s; candle 4.9 s.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
fn conv_simd(p: &[f32], pw: usize, h: usize, w: usize, cv: &Conv, out: &mut [f32]) {
    conv_4x2(p, pw, h, w, cv, out)
}
