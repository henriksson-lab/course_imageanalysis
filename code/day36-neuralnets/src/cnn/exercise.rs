// Coding exercises for "Convolutional networks". Replace each `todo!()` with your code,
// then check it with `cargo test -p day36 cnn::` and look at `cargo run -p day36 -- cnn`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: the output map of a `kw × kh` kernel is `(w - kw + 1) × (h - kh + 1)`. For output
/// pixel `(x, y)`: start at the bias, add `weights[c][(u, v)] * input[c][(x + u, y + v)]` over all
/// channels `c` and all `u`, `v`, then clip at 0 if `relu`.
pub fn conv_layer(input: &[Image<f32>], kernels: &[Kernel], relu: bool) -> Vec<Image<f32>> {
    todo!("C1: conv_layer")
}

/// C2. Max pooling with `size × size` blocks and stride `size`: the largest value of each
/// block. Leftover rows and columns at the right and bottom are dropped.
///
/// Hint: output pixel `(x, y)` is the largest of `map[(x * size + u, y * size + v)]`
/// for `u`, `v` in `0..size`.
pub fn max_pool(map: &Image<f32>, size: usize) -> Image<f32> {
    todo!("C2: max_pool")
}

/// C3a. The number of weights of a convolutional layer: `c_out` kernels of
/// `k × k × c_in` weights, plus one bias each.
pub fn conv_weights(k: usize, c_in: usize, c_out: usize) -> usize {
    todo!("C3a: conv_weights")
}

/// C3b. The output size of a layer with kernel `k`, padding `p` and stride `s` on `n`
/// pixels: `floor((n + 2p - k) / s) + 1`.
pub fn output_size(n: usize, k: usize, p: usize, s: usize) -> usize {
    todo!("C3b: output_size")
}

/// C3c. The receptive field and the jump after every layer, for layers given as
/// (kernel size, stride); a 2 × 2 pooling is (2, 2). Start with `r = j = 1`; each layer
/// adds `(k - 1) j` to `r` and multiplies `j` by `s` (slide "The math: receptive field
/// recursion"). Returns `(r, j)` after each layer.
///
/// Hint: keep `r` and `j` in two variables and update them layer by layer.
pub fn receptive_field(layers: &[(usize, usize)]) -> Vec<(usize, usize)> {
    todo!("C3c: receptive_field")
}
