// Similarity measures against a shift (SSD and mutual information, same and inverted
// contrast), and free-form registration of a deformed micrograph by gradient descent on the
// SSD; pictures in out/nonrigid-reg/.
//   cargo run -p day25 -- nonrigid-reg                      (your code)
//   cargo run -p day25 --features solution -- nonrigid-reg  (the model solution)

use imgcourse::*;
use crate::nonrigid_reg::*;

const SPACING: f32 = 16.0;

/// SSD (mean squared difference) and MI of `fixed` and `moving` shifted by `s` pixels along x,
/// on the overlapping columns.
fn scores(fixed: &Image<f32>, moving: &Image<f32>, s: isize) -> (f32, f32) {
    let w = fixed.width() as isize - s.abs();
    let (fx, mx) = if s >= 0 { (0, s as usize) } else { ((-s) as usize, 0) };
    let a = fixed.crop(fx, 0, w as usize, fixed.height());
    let b = moving.crop(mx, 0, w as usize, fixed.height());
    let ssd = a.zip_map(&b, |p, q| (p - q) * (p - q)).mean();
    (ssd, mutual_information(&a, &b, 16))
}

/// The displacement field of the coefficients at every pixel, as two images.
fn field(cx: &Image<f32>, cy: &Image<f32>, w: usize, h: usize) -> (Image<f32>, Image<f32>) {
    let d = Image::from_fn(w, h, |x, y| ffd_displacement(cx, cy, SPACING, x as f32, y as f32));
    (d.map(|p| p.0), d.map(|p| p.1))
}

/// The moving image read at `x + d(x)`.
fn apply(moving: &Image<f32>, dx: &Image<f32>, dy: &Image<f32>) -> Image<f32> {
    moving.map_xy(|x, y, _| bilinear(moving, x as f32 + dx[(x, y)], y as f32 + dy[(x, y)]))
}

/// The mean distance between two displacement fields, 8 px border excluded.
fn field_error(a: &(Image<f32>, Image<f32>), b: &(Image<f32>, Image<f32>)) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for y in 8..a.0.height() - 8 {
        for x in 8..a.0.width() - 8 {
            sum += (a.0[(x, y)] - b.0[(x, y)]).hypot(a.1[(x, y)] - b.1[(x, y)]);
            n += 1.0;
        }
    }
    sum / n
}

fn checkerboard(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.map_xy(|x, y, v| if (x / 16 + y / 16) % 2 == 0 { v } else { b[(x, y)] })
}

pub fn main() {
    let dir = "out/nonrigid-reg";
    // 1. SSD and MI against a shift along x, for the same contrast and for inverted contrast.
    let fixed = samples::cell().crop(20, 100, 96, 96);
    for (name, moving) in [("same contrast", fixed.clone()), ("inverted", fixed.map(|v| 1.0 - v))] {
        let moving = samples::add_noise(&moving, 0.03, 9);
        let (mut best_ssd, mut best_mi) = ((0, f32::INFINITY), (0, f32::NEG_INFINITY));
        for s in -30..=30 {
            let (ssd, mi) = scores(&fixed, &moving, s);
            if ssd < best_ssd.1 {
                best_ssd = (s, ssd);
            }
            if mi > best_mi.1 {
                best_mi = (s, mi);
            }
        }
        let (ssd0, mi0) = scores(&fixed, &moving, 0);
        let (_, mi6) = scores(&fixed, &moving, 6);
        println!("{name:14}: at shift 0 SSD {ssd0:.4}, MI {mi0:.2} bits (MI {mi6:.2} at shift 6); best shift: SSD {}, MI {}", best_ssd.0, best_mi.0);
    }

    // 2. Free-form registration. The moving image is the fixed one deformed by a smooth random
    //    field g: moving(y) = fixed(y + g(y)). The true answer d solves moving(x + d) = fixed(x).
    let fixed = gaussian(&samples::cell(), 1.0).crop(64, 64, 128, 128);
    let (w, h) = (fixed.width(), fixed.height());
    let mut rng = Rng::new(4);
    let gx = Image::from_fn(6, 6, |_, _| 2.5 * rng.normal());
    let gy = Image::from_fn(6, 6, |_, _| 2.5 * rng.normal());
    let g = |x: f32, y: f32| ffd_displacement(&gx, &gy, 32.0, x, y);
    let moving = fixed.map_xy(|x, y, _| {
        let (a, b) = g(x as f32, y as f32);
        bilinear(&fixed, x as f32 + a, y as f32 + b)
    });
    // d = −g(x + d), by repeated substitution.
    let truth_pairs = Image::from_fn(w, h, |x, y| {
        let mut d = (0.0, 0.0);
        for _ in 0..30 {
            let (a, b) = g(x as f32 + d.0, y as f32 + d.1);
            d = (-a, -b);
        }
        d
    });
    let truth = (truth_pairs.map(|p| p.0), truth_pairs.map(|p| p.1));
    let zero = (Image::new(w, h, 0.0), Image::new(w, h, 0.0));
    let largest = truth_pairs.pixels().iter().map(|p| p.0.hypot(p.1)).fold(0.0, f32::max);
    println!("\ntrue displacement: mean {:.2} px, largest {largest:.2} px", field_error(&truth, &zero));

    for (name, mov) in [("same contrast", moving.clone()), ("inverted contrast", moving.map(|v| 1.0 - v))] {
        let (cx, cy) = register_ssd(&fixed, &mov, SPACING, 200, 0.15);
        let d = field(&cx, &cy, w, h);
        let jac = jacobian_determinant(&d.0, &d.1);
        let folded = jac.pixels().iter().filter(|&&j| j <= 0.0).count();
        println!("SSD registration, {name}: error {:.2} px (before {:.2}), Jacobian {:.2} to {:.2}, {folded} folded pixels", field_error(&d, &truth), field_error(&zero, &truth), jac.min(), jac.max());
        let tag = if name == "same contrast" { "same" } else { "inverted" };
        save_gray(&checkerboard(&fixed, &apply(&mov, &d.0, &d.1).map(|v| if tag == "inverted" { 1.0 - v } else { v })), format!("{dir}/checker-after-{tag}.png"));
        save_gray_stretched(&jac, format!("{dir}/jacobian-{tag}.png"));
    }
    save_gray(&checkerboard(&fixed, &moving), format!("{dir}/checker-before.png"));
    save_gray(&moving, format!("{dir}/moving.png"));
    println!("wrote the pictures to {dir}/");
}

fn gaussian(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            (-r..=r)
                .map(|d| {
                    let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                    w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror)
                })
                .sum::<f32>()
                / total
        })
    };
    pass(&pass(img, true), false)
}
