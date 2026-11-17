// Moving 1D profiles (a bump at several speeds, a bleaching bump, four vesicles in a
// kymograph) and the normal flow of the moving micrograph; results in out/flow-constraint/.
//   cargo run -p day25 -- flow-constraint                      (your code)
//   cargo run -p day25 --features solution -- flow-constraint  (the model solution)

use crate::common::{bilinear, blur, flow_colour};
use crate::flow_constraint::*;
use imgcourse::*;

/// A Gaussian bump of width `sigma` centred at `c`, sampled at 0..n.
fn bump(n: usize, c: f32, sigma: f32, height: f32) -> Vec<f32> {
    (0..n).map(|x| height * (-((x as f32 - c).powi(2)) / (2.0 * sigma * sigma)).exp()).collect()
}

pub fn main() {
    let dir = "out/flow-constraint";
    let n = 72;

    // 1. A bump of width σ = 4 at x = 36, moving at different speeds: window estimate at the centre.
    println!("bump σ = 4, window half-width 6, estimate at x = 36:");
    for speed in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0] {
        let (ix, it) = derivatives_1d(&bump(n, 36.0, 4.0, 1.0), &bump(n, 36.0 + speed, 4.0, 1.0));
        let u = window_speed(&ix, &it, 36, 6).unwrap();
        println!("  true speed {speed:3.1}: estimate {u:5.2} ({:+4.0} %)", 100.0 * (u - speed) / speed);
    }

    // 2. Bleaching: speed 1, but 10 % dimmer in frame 2.
    let (ix, it) = derivatives_1d(&bump(n, 36.0, 4.0, 1.0), &bump(n, 37.0, 4.0, 0.9));
    print!("bleaching 10 %, true speed 1: window estimates");
    for x in [30, 36, 42] {
        print!("  x = {x}: {:.2}", window_speed(&ix, &it, x, 6).unwrap());
    }
    println!();

    // 3. A kymograph: four vesicles over 40 frames, one row per frame.
    let vesicles = [(20.0, 1.5), (60.0, 0.5), (70.0, 2.5), (150.0, -1.0)];
    let (width, frames) = (180, 40);
    let profile = |t: usize| {
        let mut p = vec![0.1f32; width];
        for &(start, speed) in &vesicles {
            let b = bump(width, start + speed * t as f32, 2.5, 0.8);
            for x in 0..width {
                p[x] += b[x];
            }
        }
        p
    };
    let kymo = Image::from_fn(width, frames, |x, t| profile(t)[x]);
    let big = Image::from_fn(3 * width, 3 * frames, |x, y| kymo[(x / 3, y / 3)]);
    save_gray(&big, format!("{dir}/kymograph.png"));
    print!("kymograph, window estimate from frame 5 to frame 6 at each vesicle:");
    let (ix, it) = derivatives_1d(&profile(5), &profile(6));
    for &(start, speed) in &vesicles {
        let x = (start + 5.0 * speed).round() as usize;
        print!("  {:.2} (true {speed})", window_speed(&ix, &it, x, 5).unwrap());
    }
    println!();

    // 4. The micrograph moved by (1, 0.5) px: the normal flow of every pixel, as colours.
    let f1 = blur(&samples::cell(), 1.5);
    let f2 = f1.map_xy(|x, y, _| bilinear(&f1, x as f32 - 1.0, y as f32 - 0.5));
    let flow = f1.map_xy(|x, y, v| {
        let p = |dx: isize, dy: isize| f1.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
        let (ix, iy) = ((p(1, 0) - p(-1, 0)) / 2.0, (p(0, 1) - p(0, -1)) / 2.0);
        // Only where there is some gradient (grey values in [0, 1]).
        if ix * ix + iy * iy < 1e-4 {
            return None;
        }
        normal_flow(ix, iy, f2[(x, y)] - v)
    });
    // No answer (`None`) in grey.
    save_rgb(&flow.map(|f| f.map_or([0.5; 3], |(u, v)| flow_colour(u, v, 1.5))), format!("{dir}/normal-flow.png"));
    save_rgb(&flow_legend(64, 1.5), format!("{dir}/legend.png"));
    let mut count = 0;
    let mut along = 0.0;
    for f in flow.pixels().iter().flatten() {
        // The component of the normal flow along the true motion direction (1, 0.5)/|(1, 0.5)|.
        along += (f.0 + 0.5 * f.1) / 1.25f32.sqrt();
        count += 1;
    }
    println!("normal flow of the micrograph moved by (1, 0.5): {count} pixels, mean speed along the true direction {:.2} (true speed {:.2})", along / count as f32, 1.25f32.sqrt());
    println!("wrote the pictures to {dir}/");
}

fn flow_legend(size: usize, max: f32) -> Image<[f32; 3]> {
    let c = size as f32 / 2.0;
    Image::from_fn(size, size, |x, y| {
        let (u, v) = ((x as f32 - c) / c * max, (y as f32 - c) / c * max);
        if u.hypot(v) > max { [1.0, 1.0, 1.0] } else { flow_colour(u, v, max) }
    })
}
