// Model solution for the coding exercises of "Filtering & convolution".

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
pub fn filter3x3(img: &Image<f32>, k: &Kernel, border: Border) -> Image<f32> {
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let mut sum = 0.0;
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let weight = k[(dy + 1) as usize][(dx + 1) as usize];
                sum += weight * img.get_border(x as isize + dx, y as isize + dy, border);
            }
        }
        out[(x, y)] = sum;
    }
    out
}

/// C2. The edge strength: the length of the gradient (Sobel x, Sobel y) at every pixel.
pub fn gradient_magnitude(img: &Image<f32>, border: Border) -> Image<f32> {
    let gx = filter3x3(img, &SOBEL_X, border);
    let gy = filter3x3(img, &SOBEL_Y, border);
    gx.zip_map(&gy, |a, b| (a * a + b * b).sqrt())
}

/// C3. A separable filter in two passes: first `row` along every row (x direction), then
/// `column` along every column (y direction). Must give the same result as `filter3x3`
/// with the kernel `k[i][j] = column[i] * row[j]`.
pub fn filter_separable(img: &Image<f32>, column: [f32; 3], row: [f32; 3], border: Border) -> Image<f32> {
    let pass = |src: &Image<f32>, w: [f32; 3], along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -1..=1isize {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + 1) as usize] * src.get_border(xx, yy, border);
            }
            sum
        })
    };
    let along_rows = pass(img, row, true);
    pass(&along_rows, column, false)
}
