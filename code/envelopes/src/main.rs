// Envelopes and the MAD test for three patterns, and how often each test rejects random
// patterns; numbers only.
//   cargo run -p envelopes                      (your code)
//   cargo run -p envelopes --features solution  (the model solution)

use envelopes::*;
use imgcourse::Rng;

/// Whole numbers as ranges: [2, 3, 4, 7] -> "2-4, 7".
fn ranges(v: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut k = 0;
    while k < v.len() {
        let start = k;
        while k + 1 < v.len() && v[k + 1] == v[k] + 1 {
            k += 1;
        }
        parts.push(if k > start { format!("{}-{}", v[start], v[k]) } else { v[k].to_string() });
        k += 1;
    }
    parts.join(", ")
}

fn main() {
    let (a, b) = (200.0, 200.0);
    let radii: Vec<f64> = (1..=50).map(|k| k as f64).collect();
    let mut rng = Rng::new(6);
    let patterns = [
        ("CSR", csr(150, a, b, &mut rng)),
        ("weak clusters", [csr(120, a, b, &mut rng), thomas(0.0003, 4.0, 3.0, a, b, &mut rng)].concat()),
        ("hard core 7 µm", hard_core(150, 7.0, a, b, &mut rng)),
    ];
    println!("L(r) - r with translation correction, r = 1 .. 50 µm, 39 CSR simulations with the same n");
    for (name, pts) in &patterns {
        let observed = l_curve(pts, a, b, &radii);
        let sims: Vec<Vec<f64>> = (0..39).map(|_| l_curve(&csr(pts.len(), a, b, &mut rng), a, b, &radii)).collect();
        let (lower, upper) = pointwise_envelope(&sims);
        let out = ranges(&escapes(&observed, &lower, &upper).iter().map(|&k| radii[k] as usize).collect::<Vec<_>>());
        let (t, p) = mad_test(&observed, &sims);
        // the global envelope: the 2nd largest of the 39 simulated T values
        let mut ts: Vec<f64> = sims.iter().map(|c| c.iter().fold(0.0f64, |m, v| m.max(v.abs()))).collect();
        ts.sort_by(|x, y| y.total_cmp(x));
        println!();
        println!("{name} ({} points)", pts.len());
        println!("  leaves the pointwise band at r = {}", if out.is_empty() { "nowhere".to_string() } else { out });
        println!("  MAD test over 1 .. 50 µm: T = {t:.2} µm, p = {p:.3}; global envelope ±{:.2} µm", ts[1]);
        let (t, p) = mad_test(&observed[19..], &sims.iter().map(|c| c[19..].to_vec()).collect::<Vec<_>>());
        println!("  MAD test over 20 .. 50 µm only: T = {t:.2} µm, p = {p:.3}");
    }

    println!();
    println!("200 random patterns of 100 points, 25 radii (2, 4, .., 50 µm): share rejected at 5 %");
    let radii25: Vec<f64> = (1..=25).map(|k| 2.0 * k as f64).collect();
    for s in [19, 39] {
        let (pointwise, mad) = false_alarm_rates(100, a, b, &radii25, s, 200, &mut rng);
        println!("  {s} simulations: pointwise envelope {:.0} %, MAD test {:.0} %", 100.0 * pointwise, 100.0 * mad);
    }
}
