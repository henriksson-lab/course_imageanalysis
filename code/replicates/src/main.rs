// An imaging experiment with no treatment effect, analysed with cells as n and with
// experiments as n; then the false-positive rate of both analyses over many simulated
// studies, and multiple testing of many features. Pictures in out/replicates/.
//   cargo run -p replicates                      (your code)
//   cargo run -p replicates --features solution  (the model solution)

use imgcourse::*;
use replicates::*;

/// The mean intensity of every nucleus of an image, from its labels.
fn nucleus_means(img: &Image<f32>, labels: &Image<u32>) -> Vec<f32> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut sum = vec![0.0; n];
    let mut count = vec![0usize; n];
    for (x, y) in img.coords() {
        let l = labels[(x, y)] as usize;
        if l > 0 {
            sum[l - 1] += img[(x, y)];
            count[l - 1] += 1;
        }
    }
    (0..n).filter(|&i| count[i] > 20).map(|i| sum[i] / count[i] as f32).collect()
}

/// The complementary error function (a standard approximation, error below 1e-7).
fn erfc(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let poly = -x * x - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let r = t * poly.exp();
    (if x >= 0.0 { r } else { 2.0 - r }) as f32
}

fn main() {
    let dir = "out/replicates";

    // Six experiments (days), each with its own brightness (staining, lamp). Control dishes
    // were imaged on days 1-3, treated dishes on days 4-6. The treatment does nothing.
    let day_gain = [1.0, 1.25, 0.85, 1.15, 0.95, 1.2];
    let mut sheet = Image::new(3 * 130, 2 * 130, 1.0f32);
    let mut cells = [Vec::new(), Vec::new()];
    let mut means = [Vec::new(), Vec::new()];
    for (day, gain) in day_gain.iter().enumerate() {
        let condition = day / 3;
        let (img, labels) = samples::nuclei(256, 40, day as u64 + 20);
        let img = img.map(|v| gain * v);
        let m = nucleus_means(&img, &labels);
        println!("day {}, {}: {} nuclei, mean intensity {:.3}", day + 1, ["control", "treated"][condition], m.len(), mean_sd(&m).0);
        means[condition].push(mean_sd(&m).0);
        cells[condition].extend(m);
        for (x, y) in Image::new(128, 128, 0u8).coords() {
            sheet[((day % 3) * 130 + x, condition * 130 + y)] = img[(2 * x, 2 * y)].min(1.0);
        }
    }
    save_gray(&sheet, format!("{dir}/experiment.png"));
    for (name, t) in [("cells as n", unpaired_t(&cells[0], &cells[1])), ("experiment means as n", unpaired_t(&means[0], &means[1]))] {
        let (lo, hi) = t.interval();
        println!("{name:22} effect {:+.3}, 95 % CI {lo:+.3} to {hi:+.3}, t = {:5.2} ({} df), significant: {}", t.estimate, t.t, t.df, t.significant());
    }

    // Many studies with no effect: day SD 1, cell SD 2, 3 experiments per condition.
    println!("\nfalse-positive rate (should be 5 %), 3 experiments per condition, 2000 studies:");
    println!("  cells per experiment   cells as n   experiment means");
    for m in [1, 10, 100, 1000] {
        let (a, b) = false_positive_rates(3, m, 1.0, 2.0, 2000, m as u64);
        println!("  {m:20}   {:7.1} %     {:7.1} %", 100.0 * a, 100.0 * b);
    }

    // 300 features, 30 of them with a real effect: z-scores N(0, 1) for the null features,
    // N(3, 1) for the real ones; two-sided p-values.
    let mut rng = Rng::new(4);
    let mut p = Vec::new();
    for i in 0..300 {
        let z = if i < 30 { 3.0 + rng.normal() } else { rng.normal() };
        p.push(erfc(z.abs() / 2f32.sqrt())); // = 2 P(Z > |z|)
    }
    let raw = p.iter().filter(|&&v| v <= 0.05).count();
    let raw_real = p[..30].iter().filter(|&&v| v <= 0.05).count();
    let (bonf, bh) = (bonferroni(&p, 0.05), benjamini_hochberg(&p, 0.05));
    // which of the hits are real: the hits are the smallest p-values
    let mut order: Vec<usize> = (0..300).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let real_among = |k: usize| order[..k].iter().filter(|&&i| i < 30).count();
    println!("\n300 features, 30 of them real:");
    println!("  uncorrected        {raw:3} hits, {raw_real} real");
    println!("  Bonferroni         {bonf:3} hits, {} real", real_among(bonf));
    println!("  Benjamini–Hochberg {bh:3} hits, {} real", real_among(bh));
    println!("wrote {dir}/experiment.png");
}
