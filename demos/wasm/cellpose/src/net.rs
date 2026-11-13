//! The Cellpose 3 U-net ("CPnet", models cyto3 and nuclei): 2 input channels, four
//! resolution levels with 32/64/128/256 channels, residual blocks of batch norm -> ReLU ->
//! 3x3 convolution, a 256-long "style" vector (global average of the deepest features,
//! normalised) added to every decoder block, and 3 outputs per pixel: flow dY, flow dX and
//! cell probability (a logit). Inference only (batch norms use their running statistics).
//! Layer names follow cellpose/resnet_torch.py, so a PyTorch state dict loads directly.
use crate::conv::{conv, ConvW, Pre};
use crate::weights::{Tensor, Weights};

const EPS: f32 = 1e-5;

/// Batch norm (inference) as a per-channel affine map, then an optional ReLU, then a conv.
struct BnConv {
    scale: Vec<f32>,
    shift: Vec<f32>,
    relu: bool,
    conv: ConvW,
}

fn t<'a>(w: &'a Weights, name: &str) -> Result<&'a Tensor, String> {
    w.get(name).ok_or_else(|| format!("weights: missing tensor {name}"))
}

impl BnConv {
    /// `bn` and `cv` are prefixes, e.g. "...conv_0.0" and "...conv_0.2".
    fn load(w: &Weights, bn: &str, cv: &str, relu: bool) -> Result<BnConv, String> {
        let g = t(w, &format!("{bn}.weight"))?;
        let b = t(w, &format!("{bn}.bias"))?;
        let m = t(w, &format!("{bn}.running_mean"))?;
        let v = t(w, &format!("{bn}.running_var"))?;
        let n = g.data.len();
        let mut scale = vec![0f32; n];
        let mut shift = vec![0f32; n];
        for c in 0..n {
            scale[c] = g.data[c] / (v.data[c] + EPS).sqrt();
            shift[c] = b.data[c] - m.data[c] * scale[c];
        }
        let cw = t(w, &format!("{cv}.weight"))?;
        let cb = t(w, &format!("{cv}.bias"))?;
        if cw.shape.len() != 4 || cw.shape[1] != n {
            return Err(format!("weights: {cv}.weight has shape {:?}", cw.shape));
        }
        let conv = ConvW::new(&cw.data, &cb.data, cw.shape[0], cw.shape[1], cw.shape[2]);
        Ok(BnConv { scale, shift, relu, conv })
    }

    fn cout(&self) -> usize {
        self.conv.cout
    }

    /// out (+)= conv(act(x)), x of (cin, h, w); `extra` adds a per-channel constant before
    /// the batch norm (the style term).
    fn run(&self, x: &[f32], h: usize, w: usize, extra: Option<&[f32]>, out: &mut [f32], acc: bool, col: &mut Vec<f32>) {
        let shift: Vec<f32> = match extra {
            None => self.shift.clone(),
            Some(e) => (0..self.shift.len()).map(|c| self.shift[c] + self.scale[c] * e[c]).collect(),
        };
        let pre = Pre { scale: &self.scale, shift: &shift, relu: self.relu };
        conv(x, h, w, &pre, &self.conv, out, acc, col);
    }
}

struct ResDown {
    proj: BnConv,
    conv: [BnConv; 4],
}

impl ResDown {
    fn load(w: &Weights, p: &str) -> Result<ResDown, String> {
        let c = |i: usize| BnConv::load(w, &format!("{p}.conv.conv_{i}.0"), &format!("{p}.conv.conv_{i}.2"), true);
        Ok(ResDown {
            proj: BnConv::load(w, &format!("{p}.proj.0"), &format!("{p}.proj.1"), false)?,
            conv: [c(0)?, c(1)?, c(2)?, c(3)?],
        })
    }
    /// x = proj(x) + conv1(conv0(x)); x = x + conv3(conv2(x))
    fn forward(&self, x: &[f32], h: usize, w: usize, col: &mut Vec<f32>) -> Vec<f32> {
        let co = self.proj.cout();
        let n = co * h * w;
        let mut out = vec![0f32; n];
        let mut tmp = vec![0f32; n];
        self.proj.run(x, h, w, None, &mut out, false, col);
        self.conv[0].run(x, h, w, None, &mut tmp, false, col);
        self.conv[1].run(&tmp, h, w, None, &mut out, true, col);
        let mut tmp2 = vec![0f32; n];
        self.conv[2].run(&out, h, w, None, &mut tmp2, false, col);
        self.conv[3].run(&tmp2, h, w, None, &mut out, true, col);
        out
    }
}

struct StyleConv {
    bc: BnConv,
    full_w: Vec<f32>, // (cout_bn, 256)
    full_b: Vec<f32>,
}

impl StyleConv {
    fn load(w: &Weights, p: &str) -> Result<StyleConv, String> {
        Ok(StyleConv {
            bc: BnConv::load(w, &format!("{p}.conv.0"), &format!("{p}.conv.2"), true)?,
            full_w: t(w, &format!("{p}.full.weight"))?.data.clone(),
            full_b: t(w, &format!("{p}.full.bias"))?.data.clone(),
        })
    }
    fn feat(&self, style: &[f32]) -> Vec<f32> {
        let ns = style.len();
        (0..self.full_b.len())
            .map(|c| self.full_b[c] + (0..ns).map(|k| self.full_w[c * ns + k] * style[k]).sum::<f32>())
            .collect()
    }
}

struct ResUp {
    proj: BnConv,
    conv0: BnConv,
    conv: [StyleConv; 3],
}

impl ResUp {
    fn load(w: &Weights, p: &str) -> Result<ResUp, String> {
        let s = |i: usize| StyleConv::load(w, &format!("{p}.conv.conv_{i}"));
        Ok(ResUp {
            proj: BnConv::load(w, &format!("{p}.proj.0"), &format!("{p}.proj.1"), false)?,
            conv0: BnConv::load(w, &format!("{p}.conv.conv_0.0"), &format!("{p}.conv.conv_0.2"), true)?,
            conv: [s(1)?, s(2)?, s(3)?],
        })
    }
    /// x = proj(x) + conv1(style, conv0(x) + y); x = x + conv3(style, conv2(style, x))
    fn forward(&self, x: &[f32], y: &[f32], style: &[f32], h: usize, w: usize, col: &mut Vec<f32>) -> Vec<f32> {
        let co = self.proj.cout();
        let n = co * h * w;
        let mut a = vec![0f32; n];
        self.conv0.run(x, h, w, None, &mut a, false, col);
        for i in 0..n {
            a[i] += y[i];
        }
        let mut out = vec![0f32; n];
        self.proj.run(x, h, w, None, &mut out, false, col);
        self.conv[0].bc.run(&a, h, w, Some(&self.conv[0].feat(style)), &mut out, true, col);
        self.conv[1].bc.run(&out, h, w, Some(&self.conv[1].feat(style)), &mut a, false, col);
        self.conv[2].bc.run(&a, h, w, Some(&self.conv[2].feat(style)), &mut out, true, col);
        out
    }
}

fn maxpool2(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h / 2, w / 2);
    let mut o = vec![0f32; c * h2 * w2];
    for ch in 0..c {
        for y in 0..h2 {
            for xx in 0..w2 {
                let b = ch * h * w + 2 * y * w + 2 * xx;
                o[ch * h2 * w2 + y * w2 + xx] = x[b].max(x[b + 1]).max(x[b + w]).max(x[b + w + 1]);
            }
        }
    }
    o
}

fn upsample2(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (h * 2, w * 2);
    let mut o = vec![0f32; c * h2 * w2];
    for ch in 0..c {
        for y in 0..h2 {
            for xx in 0..w2 {
                o[ch * h2 * w2 + y * w2 + xx] = x[ch * h * w + (y / 2) * w + xx / 2];
            }
        }
    }
    o
}

pub struct Net {
    down: Vec<ResDown>,
    up: Vec<ResUp>,
    output: BnConv,
    pub diam_mean: f32,
    pub diam_labels: f32,
    pub n_params: usize,
    col: Vec<f32>,
    pub style: Vec<f32>,
}

impl Net {
    pub fn load(w: &Weights) -> Result<Net, String> {
        let mut down = Vec::new();
        for i in 0..4 {
            down.push(ResDown::load(w, &format!("downsample.down.res_down_{i}"))?);
        }
        let mut up = Vec::new();
        for i in 0..4 {
            up.push(ResUp::load(w, &format!("upsample.up.res_up_{i}"))?);
        }
        let output = BnConv::load(w, "output.0", "output.2", true)?;
        if down[0].proj.conv.cin != 2 || output.cout() != 3 {
            return Err("weights: not a Cellpose 3 network (2 inputs, 3 outputs)".into());
        }
        let scalar = |n: &str, d: f32| w.get(n).and_then(|t| t.data.first().copied()).unwrap_or(d);
        let n_params = w.iter().filter(|(k, _)| !k.contains("running_")).map(|(_, t)| t.data.len()).sum();
        Ok(Net {
            down,
            up,
            output,
            diam_mean: scalar("diam_mean", 30.0),
            diam_labels: scalar("diam_labels", 30.0),
            n_params,
            col: Vec::new(),
            style: vec![0.0; 256],
        })
    }

    /// One tile: `x` is (2, h, w) with h and w multiples of 8. Returns (3, h, w): flow dY,
    /// flow dX, cell probability logit. The style vector of this tile is left in `self.style`.
    pub fn forward(&mut self, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        if h % 8 != 0 || w % 8 != 0 || h == 0 || w == 0 {
            return Err(format!("tile size {h}x{w}: both must be positive multiples of 8"));
        }
        if x.len() != 2 * h * w {
            return Err(format!("input has {} values, expected 2*{h}*{w}", x.len()));
        }
        let mut col = std::mem::take(&mut self.col);
        let mut xd: Vec<Vec<f32>> = Vec::new();
        let mut dims = Vec::new();
        let (mut hh, mut ww) = (h, w);
        for n in 0..4 {
            let y = if n == 0 {
                x.to_vec()
            } else {
                let c = self.down[n - 1].proj.cout();
                let (ph, pw) = dims[n - 1];
                hh = ph / 2;
                ww = pw / 2;
                maxpool2(&xd[n - 1], c, ph, pw)
            };
            let o = self.down[n].forward(&y, hh, ww, &mut col);
            xd.push(o);
            dims.push((hh, ww));
        }
        // style: global average of the deepest features, unit length
        let c3 = self.down[3].proj.cout();
        let (h3, w3) = dims[3];
        let mut style = vec![0f32; c3];
        for c in 0..c3 {
            let s: f64 = xd[3][c * h3 * w3..(c + 1) * h3 * w3].iter().map(|&v| v as f64).sum();
            style[c] = (s / (h3 * w3) as f64) as f32;
        }
        let norm = style.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in style.iter_mut() {
            *v /= norm;
        }
        // decoder
        let mut x = self.up[3].forward(&xd[3], &xd[3], &style, h3, w3, &mut col);
        for n in (0..3).rev() {
            let c = self.up[n + 1].proj.cout();
            let (ph, pw) = dims[n + 1];
            let u = upsample2(&x, c, ph, pw);
            let (nh, nw) = dims[n];
            x = self.up[n].forward(&u, &xd[n], &style, nh, nw, &mut col);
        }
        let mut out = vec![0f32; 3 * h * w];
        self.output.run(&x, h, w, None, &mut out, false, &mut col);
        self.col = col;
        self.style = style;
        Ok(out)
    }
}
