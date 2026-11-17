// Builds the integral image of the micrograph and times window sums directly and with the
// table. Pictures in out/integral-image/.
//   cargo run -p integral-image                      (your code)
//   cargo run -p integral-image --features solution  (the model solution)

use imgcourse::*;
use integral_image::*;
use std::time::Instant;

fn main() {
    let dir = "out/integral-image";
    let img = samples::cell();
    save_gray(&img, format!("{dir}/input.png"));

    // The table itself: a smooth ramp from 0 (top left) to the sum of the image (bottom right).
    let s = integral_image(&img);
    let total: f64 = img.pixels().iter().map(|&v| v as f64).sum();
    println!("the last entry of the table: {:.3} (the sum of all pixels: {total:.3})", s[(256, 256)]);
    save_gray_stretched(&s.map(|v| v as f32), format!("{dir}/table.png"));

    // Differencing undoes summing: the four-lookup sum of every single pixel is the image again.
    let back = img.map_xy(|x, y, _| rect_sum(&s, x, y, x, y) as f32);
    let err = img.zip_map(&back, |a, b| (a - b).abs()).max();
    println!("largest difference between the image and its single-pixel sums: {err:.2e}");

    // The mean of every k × k window, directly and with the table, and the time each takes.
    println!("\n  k   windows   direct (ms)   table (ms)   largest difference");
    for k in [3, 11, 31, 101] {
        let t0 = Instant::now();
        let a = window_sums_direct(&img, k);
        let t_direct = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let b = window_sums_table(&img, k);
        let t_table = t0.elapsed().as_secs_f64() * 1000.0;
        let diff = a.zip_map(&b, |p, q| (p - q).abs() as f32).max();
        println!("{k:>3} {:>9} {t_direct:>13.2} {t_table:>12.2}   {diff:.1e}", a.width() * a.height());
        let n = (k * k) as f32;
        save_gray(&b.map(|v| v as f32 / n), format!("{dir}/mean{k}.png"));
    }
    println!("\nwrote the results to {dir}/");
}
