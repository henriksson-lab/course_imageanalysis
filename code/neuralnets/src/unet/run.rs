// A hand-made encoder–decoder (no training) on a synthetic fluorescence image with uneven
// illumination, scored with the losses of the unit. Pictures in out/unet/.
//   cargo run -p neuralnets -- unet                      (your code)
//   cargo run -p neuralnets --features solution -- unet  (the model solution)
//
// The "network": the encoder averages 4 × 4 blocks (64 × 64 maps) and estimates the
// background there as the darkest value in a 9 × 9 neighbourhood (context: much larger than a
// nucleus). The decoder upsamples the background back to full size. The output compares
// each pixel with its background: either the full-size input, brought across by a skip
// connection, or only the upsampled coarse map (no skip).

use imgcourse::*;
use crate::unet::*;

fn average_pool(img: &Image<f32>, size: usize) -> Image<f32> {
    Image::from_fn(img.width() / size, img.height() / size, |x, y| {
        let mut s = 0.0;
        for v in 0..size {
            for u in 0..size {
                s += img[(x * size + u, y * size + v)];
            }
        }
        s / (size * size) as f32
    })
}

fn local_min(img: &Image<f32>, r: isize) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let mut m = f32::INFINITY;
        for dy in -r..=r {
            for dx in -r..=r {
                m = m.min(img.get_border(x as isize + dx, y as isize + dy, Border::Clamp));
            }
        }
        m
    })
}

fn sigmoid(s: f32) -> f32 {
    1.0 / (1.0 + (-s).exp())
}

fn report(name: &str, p: &Image<f32>, truth: &Image<bool>) {
    let p = p.map(|v| v.clamp(1e-6, 1.0 - 1e-6));
    let mask = p.map(|v| if v > 0.5 { 1.0 } else { 0.0 });
    println!(
        "{name:28} cross-entropy {:.3}   soft Dice {:.3}   Dice of the mask {:.3}",
        pixel_cross_entropy(&p, truth),
        soft_dice(&p, truth),
        soft_dice(&mask, truth)
    );
}

pub fn main() {
    let dir = "out/unet";
    let (clean, labels) = samples::nuclei(256, 30, 5);
    let truth = labels.map(|l| l > 0);
    // uneven illumination: 0.45 brighter on the right
    let img = clean.map_xy(|x, _, v| v + 0.45 * x as f32 / 255.0);
    save_gray_stretched(&img, format!("{dir}/input.png"));
    save_gray(&truth.map(|t| if t { 1.0 } else { 0.0 }), format!("{dir}/truth.png"));

    // Encoder: 4 × 4 averages, then the local background on the 64 × 64 map.
    let coarse = average_pool(&img, 4);
    let background = local_min(&coarse, 4);
    // Decoder: back to 256 × 256 with a transposed convolution whose kernel is a 7 × 7 tent
    // (linear interpolation), then cut the 3 extra pixels of the border.
    let tent = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25];
    let kernel = Image::from_fn(7, 7, |u, v| tent[u] * tent[v]);
    let up = transposed_conv(&background, &kernel, 4);
    let background_full = up.crop(2, 2, 256, 256);
    save_gray_stretched(&upsample_nearest(&background, 4), format!("{dir}/background-coarse.png"));
    save_gray_stretched(&background_full, format!("{dir}/background-upsampled.png"));

    let (gain, offset) = (40.0, 0.25);
    let with_skip = img.zip_map(&background_full, |v, b| sigmoid(gain * (v - b - offset)));
    let coarse_up = upsample_nearest(&coarse, 4);
    let no_skip = coarse_up.zip_map(&background_full, |v, b| sigmoid(gain * (v - b - offset)));
    let mean = img.mean();
    let global = img.map(|v| sigmoid(gain * (v - mean - 0.05)));
    save_gray(&with_skip, format!("{dir}/p-skip.png"));
    save_gray(&no_skip, format!("{dir}/p-no-skip.png"));
    save_gray(&global, format!("{dir}/p-global-threshold.png"));

    let share = truth.pixels().iter().filter(|&&t| t).count() as f32 / (256.0 * 256.0);
    println!("nucleus pixels: {:.1} %", 100.0 * share);
    report("with skip connection", &with_skip, &truth);
    report("no skip connection", &no_skip, &truth);
    report("one global threshold", &global, &truth);
    report("p = 0.01 everywhere", &Image::new(256, 256, 0.01), &truth);

    // Where does the Dice loss push? The gradient of the soft Dice of the skip output.
    let grad = soft_dice_gradient(&with_skip, &truth);
    let largest = grad.map(f32::abs).max();
    save_gray_signed(&grad.map(|v| v / largest), format!("{dir}/dice-gradient.png"));
    println!("wrote the pictures to {dir}/");
}
