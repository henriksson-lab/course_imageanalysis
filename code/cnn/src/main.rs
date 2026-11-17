// Runs a hand-made convolutional layer (a bank of four kernels), a second layer and max
// pooling on the course micrograph, and checks that shifting the input shifts the maps.
// Pictures in out/cnn/.
//   cargo run -p cnn                      (your code)
//   cargo run -p cnn --features solution  (the model solution)

use cnn::*;
use imgcourse::*;

fn kernel(rows: [[f32; 3]; 3], bias: f32) -> Kernel {
    Kernel { weights: vec![Image::from_rows(&rows)], bias }
}

/// Feature maps side by side, each stretched to its own range, with a 4-pixel gap.
fn montage(maps: &[Image<f32>]) -> Image<f32> {
    let (w, h) = (maps[0].width(), maps[0].height());
    let mut out = Image::new(maps.len() * (w + 4) - 4, h, 1.0);
    for (i, m) in maps.iter().enumerate() {
        let (lo, hi) = (m.min(), m.max().max(m.min() + 1e-6));
        for (x, y) in m.coords() {
            out[(i * (w + 4) + x, y)] = (m[(x, y)] - lo) / (hi - lo);
        }
    }
    out
}

fn main() {
    let dir = "out/cnn";
    let input = vec![samples::cell()]; // one channel
    let img = &input[0];
    save_gray(img, format!("{dir}/input.png"));

    // Layer 1: Sobel x, Sobel y, Laplacian, and a Gaussian with a negative bias, so that
    // only bright regions survive the ReLU.
    let bank = [
        kernel([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], 0.0),
        kernel([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], 0.0),
        kernel([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], 0.0),
        kernel([[1. / 16., 2. / 16., 1. / 16.], [2. / 16., 4. / 16., 2. / 16.], [1. / 16., 2. / 16., 1. / 16.]], -0.5),
    ];
    let maps1 = conv_layer(&input, &bank, true);
    println!("layer 1: {} maps of {} × {}, {} weights", maps1.len(), maps1[0].width(), maps1[0].height(), conv_weights(3, 1, 4));
    save_gray(&montage(&maps1), format!("{dir}/layer1.png"));

    // Pooling, then layer 2: one kernel stack over all four channels. It adds the two
    // edge maps (any edge where the image gets brighter to the right or downwards) and
    // subtracts the bright-region map.
    let pooled: Vec<Image<f32>> = maps1.iter().map(|m| max_pool(m, 2)).collect();
    save_gray(&montage(&pooled), format!("{dir}/layer1-pooled.png"));
    let centre = |v: f32| Image::from_rows(&[[0., 0., 0.], [0., v, 0.], [0., 0., 0.]]);
    let combine = Kernel { weights: vec![centre(1.0), centre(1.0), centre(0.0), centre(-4.0)], bias: 0.0 };
    let maps2 = conv_layer(&pooled, &[combine], true);
    println!("layer 2: {} map of {} × {}, {} weights", maps2.len(), maps2[0].width(), maps2[0].height(), conv_weights(3, 4, 1));
    save_gray(&montage(&maps2), format!("{dir}/layer2.png"));

    // Equivariance: shift the input by (dx, dy) and compare the maps.
    for (dx, dy) in [(8, 4), (1, 0), (2, 0)] {
        let shifted = Image::from_fn(img.width(), img.height(), |x, y| img.get_border(x as isize - dx, y as isize - dy, Border::Clamp));
        let m = conv_layer(&[shifted], &bank[0..1], true);
        let (a, b) = (&maps1[0], &m[0]);
        let p = max_pool(&m[0], 2);
        let pa = &pooled[0];
        // compare away from the border, where the shift brought in copied edge pixels
        let mut conv_diff: f32 = 0.0;
        for y in 20..200 {
            for x in 20..200 {
                conv_diff = conv_diff.max((b[(x + dx as usize, y + dy as usize)] - a[(x, y)]).abs());
            }
        }
        let mut pool_diff: f32 = 0.0;
        if dx % 2 == 0 && dy % 2 == 0 {
            for y in 10..100 {
                for x in 10..100 {
                    pool_diff = pool_diff.max((p[(x + dx as usize / 2, y + dy as usize / 2)] - pa[(x, y)]).abs());
                }
            }
            println!("shift ({dx}, {dy}): conv map shifted by the same amount, largest difference {conv_diff}; pooled map shifted by half, largest difference {pool_diff}");
        } else {
            // an odd shift: compare the pooled map with itself, unshifted and shifted by 1 block
            let mut same: f32 = 0.0;
            let mut next: f32 = 0.0;
            for y in 10..100 {
                for x in 10..100 {
                    same = same.max((p[(x, y)] - pa[(x, y)]).abs());
                    next = next.max((p[(x + 1, y)] - pa[(x, y)]).abs());
                }
            }
            println!("shift ({dx}, {dy}): conv map largest difference {conv_diff}; the pooled map is no shifted copy (largest difference {same:.3} unshifted, {next:.3} shifted by one block)");
        }
    }
    println!("wrote the pictures to {dir}/");
}
