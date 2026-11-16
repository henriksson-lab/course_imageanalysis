// Coding exercises for "Filtering & convolution". Replace each `todo!()` with your code,
// then check it with `cargo test -p kernels` and look at `cargo run -p kernels`.
#![allow(unused_variables)]

use imgcourse::{Border, Image};

/// A 3 × 3 kernel, written row by row as in the lecture: `k[row][column]`, centre `k[1][1]`.
pub type Kernel = [[f32; 3]; 3];

pub const BOX: Kernel = [[1. / 9.; 3]; 3];
pub const GAUSSIAN: Kernel = [[1. / 16., 2. / 16., 1. / 16.], [2. / 16., 4. / 16., 2. / 16.], [1. / 16., 2. / 16., 1. / 16.]];
pub const SHARPEN: Kernel = [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]];
pub const LAPLACIAN: Kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]];
pub const SOBEL_X: Kernel = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]];
pub const SOBEL_Y: Kernel = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]];

/// C1. Filter with a 3 × 3 kernel: every output pixel is the weighted sum of the 3 × 3
/// neighbourhood of the same input pixel (without flipping the kernel). Pixels outside
/// the image are read with the `border` rule.
///
/// Hint: `img.get_border(x as isize + dx, y as isize + dy, border)` reads a neighbour,
/// also outside the image; the weight for the offset `(dx, dy)` is `k[dy + 1][dx + 1]`.
pub fn filter3x3(img: &Image<f32>, k: &Kernel, border: Border) -> Image<f32> {
    todo!("C1: filter3x3")
}

/// C2. The edge strength: the length of the gradient (Sobel x, Sobel y) at every pixel.
///
/// Hint: filter twice with `filter3x3`, then combine the two results with `zip_map`.
pub fn gradient_magnitude(img: &Image<f32>, border: Border) -> Image<f32> {
    todo!("C2: gradient_magnitude")
}

/// C3. A separable filter in two passes: first `row` along every row (x direction), then
/// `column` along every column (y direction). Must give the same result as `filter3x3`
/// with the kernel `k[i][j] = column[i] * row[j]`.
pub fn filter_separable(img: &Image<f32>, column: [f32; 3], row: [f32; 3], border: Border) -> Image<f32> {
    todo!("C3: filter_separable")
}
