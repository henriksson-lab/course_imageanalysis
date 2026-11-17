// Float16 by hand, long sums, variances and a NaN that spreads; numbers printed, one picture
// in out/floatingpoint/.
//   cargo run -p day03 -- floatingpoint                      (your code)
//   cargo run -p day03 --features solution -- floatingpoint  (the model solution)

use crate::floatingpoint::*;
use imgcourse::*;

fn f32_add(v: f32) -> f32 {
    v
}

pub fn main() {
    let dir = "out/floatingpoint";

    // The gap between neighbouring float16 numbers, at a few values.
    println!("float16 near      gap to the next float");
    for v in [1.0f32, 1000.0, 3000.0, 65000.0] {
        // v itself as a float16, then the next float16 above it: step up until rounding moves
        let v = round_to_f16(v);
        let mut step = 2f32.powi(-24);
        while round_to_f16(v + step) == v {
            step *= 2.0;
        }
        println!("  {v:8}        {}", round_to_f16(v + step) - v);
    }
    // A 16-bit image stored in half floats: the largest error at each brightness.
    print!("16-bit values stored as float16, largest error up to");
    for top in [2048u32, 4096, 32768, 65504] {
        let worst = (0..=top).map(|v| (round_to_f16(v as f32) - v as f32).abs()).fold(0.0f32, f32::max);
        print!("  {top}: {worst}");
    }
    println!("; 65 535 becomes {}", round_to_f16(65535.0));

    // Summing 2^22 (about 4 million) 16-bit pixel values, mean 1000, SD 50.
    let mut rng = Rng::new(22);
    let xs: Vec<f32> = (0..1 << 22).map(|_| (1000.0 + 50.0 * rng.normal()).round().clamp(0.0, 65535.0)).collect();
    let n = xs.len() as f64;
    let exact: f64 = xs.iter().map(|&x| x as f64).sum();
    let normalised: Vec<f32> = xs.iter().map(|&x| round_to_f16(x / 65535.0)).collect();
    println!("\nmean of {} pixels: exact {:.4}", xs.len(), exact / n);
    let report = |name: &str, s: f32, scale: f64| {
        let mean = s as f64 * scale / n;
        println!("  {name:20} mean {mean:12.4}   relative error {:.1e}", ((mean - exact / n) / (exact / n)).abs());
    };
    report("float32 naive", sum_naive(&xs, f32_add), 1.0);
    report("float32 pairwise", sum_pairwise(&xs, f32_add), 1.0);
    report("float32 Kahan", sum_kahan(&xs, f32_add), 1.0);
    report("float16 naive", sum_naive(&normalised, round_to_f16), 65535.0);
    report("float16 pairwise", sum_pairwise(&normalised, round_to_f16), 65535.0);
    // Where does the naive float32 sum start to round? When it passes 2^24.
    let (mut s32, mut s64, mut exact_up_to) = (0.0f32, 0.0f64, 0);
    for (k, &x) in xs.iter().enumerate() {
        s32 += x;
        s64 += x as f64;
        if s32 as f64 != s64 {
            exact_up_to = k;
            break;
        }
    }
    println!("  the naive float32 sum is exact for the first {exact_up_to} pixels (sum {s64} when it first rounds)");

    // Variances: exact (in f64), one pass, two passes with naive and with pairwise sums.
    let mean64 = exact / n;
    let var64 = xs.iter().map(|&x| (x as f64 - mean64).powi(2)).sum::<f64>() / n;
    println!("\nvariance: exact {var64:.2}; one pass {:.2}; two passes, naive sums {:.2}; two passes, pairwise sums {:.2}", variance_one_pass(&xs), variance_two_pass(&xs, sum_naive), variance_two_pass(&xs, sum_pairwise));

    // One NaN pixel in the micrograph, and three 3 × 3 mean filters.
    let mut img = samples::cell().crop(100, 100, 32, 32);
    img[(16, 16)] = f32::NAN;
    let mut steps = vec![img.clone()];
    for _ in 0..3 {
        let prev = steps.last().unwrap();
        let next = prev.map_xy(|x, y, _| {
            let mut s = 0.0;
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    s += prev.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
                }
            }
            s / 9.0
        });
        steps.push(next);
    }
    print!("\nNaN pixels after 0, 1, 2, 3 blurs:");
    let mut sheet = Image::new(4 * 36 - 4, 32, [1.0f32; 3]);
    for (i, s) in steps.iter().enumerate() {
        print!(" {}", s.pixels().iter().filter(|v| v.is_nan()).count());
        for (x, y) in s.coords() {
            let v = s[(x, y)];
            sheet[(36 * i + x, y)] = if v.is_nan() { [1.0, 0.0, 0.0] } else { [v, v, v] };
        }
    }
    println!("; the mean of the last image is {}", steps[3].mean());
    let big = Image::from_fn(sheet.width() * 4, sheet.height() * 4, |x, y| sheet[(x / 4, y / 4)]);
    save_rgb(&big, format!("{dir}/nan-spreads.png"));
    println!("wrote {dir}/nan-spreads.png");
}
