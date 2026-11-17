// Model solution for the coding exercises of "Convolutional networks".

use imgcourse::Image;

/// One kernel of a convolutional layer: one small image of weights per **input** channel
/// (all the same size), and one bias. It gives one output channel (feature map).
#[derive(Clone, Debug)]
pub struct Kernel {
    pub weights: Vec<Image<f32>>,
    pub bias: f32,
}

/// C1. A convolutional layer with valid padding (no padding: each map shrinks by the
/// kernel size minus 1): for every kernel, the correlation (no flip) of every input channel
/// with its weights, summed over the channels, plus the bias, then the ReLU if `relu`.
pub fn conv_layer(input: &[Image<f32>], kernels: &[Kernel], relu: bool) -> Vec<Image<f32>> {
    let (w, h) = (input[0].width(), input[0].height());
    let mut maps = Vec::new();
    for kernel in kernels {
        let (kw, kh) = (kernel.weights[0].width(), kernel.weights[0].height());
        let mut out = Image::new(w - kw + 1, h - kh + 1, 0.0);
        for (x, y) in out.coords() {
            let mut sum = kernel.bias;
            for c in 0..input.len() {
                for v in 0..kh {
                    for u in 0..kw {
                        // the window's top-left corner is at (x, y)
                        sum += kernel.weights[c][(u, v)] * input[c][(x + u, y + v)];
                    }
                }
            }
            out[(x, y)] = if relu { sum.max(0.0) } else { sum };
        }
        maps.push(out);
    }
    maps
}

/// C2. Max pooling with `size × size` blocks and stride `size`: the largest value of each
/// block. Leftover rows and columns at the right and bottom are dropped.
pub fn max_pool(map: &Image<f32>, size: usize) -> Image<f32> {
    let mut out = Image::new(map.width() / size, map.height() / size, 0.0);
    for (x, y) in out.coords() {
        let mut m = f32::NEG_INFINITY;
        for v in 0..size {
            for u in 0..size {
                m = m.max(map[(x * size + u, y * size + v)]);
            }
        }
        out[(x, y)] = m;
    }
    out
}

/// C3a. The number of weights of a convolutional layer: `c_out` kernels of
/// `k × k × c_in` weights, plus one bias each.
pub fn conv_weights(k: usize, c_in: usize, c_out: usize) -> usize {
    k * k * c_in * c_out + c_out
}

/// C3b. The output size of a layer with kernel `k`, padding `p` and stride `s` on `n`
/// pixels: `floor((n + 2p - k) / s) + 1`.
pub fn output_size(n: usize, k: usize, p: usize, s: usize) -> usize {
    (n + 2 * p - k) / s + 1
}

/// C3c. The receptive field and the jump after every layer, for layers given as
/// (kernel size, stride); a 2 × 2 pooling is (2, 2). Start with `r = j = 1`; each layer
/// adds `(k - 1) j` to `r` and multiplies `j` by `s` (slide "The math: receptive field
/// recursion"). Returns `(r, j)` after each layer.
pub fn receptive_field(layers: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let (mut r, mut j) = (1, 1);
    let mut out = Vec::new();
    for &(k, s) in layers {
        r += (k - 1) * j;
        j *= s;
        out.push((r, j));
    }
    out
}
