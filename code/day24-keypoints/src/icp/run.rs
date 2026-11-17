// Aligns the nuclei centroids of two "time points" by ICP: the second time point is the first
// one turned and shifted (the stage moved), with a little motion of every nucleus; three
// nuclei have left the field on the left and three new ones have come in on the right.
// Pictures in out/icp/.
//   cargo run -p day24 -- icp                      (your code)
//   cargo run -p day24 --features solution -- icp  (the model solution)

use crate::icp::*;
use imgcourse::*;

/// The centroid of every label 1..=n of a label image (labels that are hidden are skipped).
fn label_centroids(labels: &Image<u32>, n: usize) -> Vec<Point> {
    let mut sum = vec![(0.0f32, 0.0f32, 0usize); n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        sum[l].0 += x as f32;
        sum[l].1 += y as f32;
        sum[l].2 += 1;
    }
    sum[1..].iter().filter(|s| s.2 > 0).map(|s| (s.0 / s.2 as f32, s.1 / s.2 as f32)).collect()
}

/// Time point 2: the nuclei of time point 1 turned by `degrees` about the image centre and
/// shifted by (6, −4), plus 0.7 px of motion each; the 3 leftmost nuclei are gone and 3 new
/// ones appear on the right. Returns the points, which of them are real (not new), and the
/// true transform from time point 2 back to time point 1.
fn time_point_2(t1: &[Point], degrees: f32, seed: u64) -> (Vec<Point>, Vec<bool>, Rigid) {
    let mut rng = Rng::new(seed);
    let a = degrees.to_radians();
    let (s, c) = a.sin_cos();
    // forward: turn about (128, 128), then shift
    let forward = Rigid { angle: a, tx: 128.0 - (c * 128.0 - s * 128.0) + 6.0, ty: 128.0 - (s * 128.0 + c * 128.0) - 4.0 };
    let mut by_x: Vec<usize> = (0..t1.len()).collect();
    by_x.sort_by(|&i, &j| t1[i].0.total_cmp(&t1[j].0));
    let mut points = Vec::new();
    let mut real = Vec::new();
    for (i, &p) in t1.iter().enumerate() {
        let (dx, dy) = (0.7 * rng.normal(), 0.7 * rng.normal());
        if by_x[..3].contains(&i) {
            continue;
        }
        let q = forward.apply(p);
        points.push((q.0 + dx, q.1 + dy));
        real.push(true);
    }
    for _ in 0..3 {
        points.push((215.0 + 35.0 * rng.uniform(), 20.0 + 216.0 * rng.uniform()));
        real.push(false);
    }
    // the inverse of the forward transform
    let back_angle = -a;
    let (bs, bc) = back_angle.sin_cos();
    let back = Rigid { angle: back_angle, tx: -(bc * forward.tx - bs * forward.ty), ty: -(bs * forward.tx + bc * forward.ty) };
    (points, real, back)
}

/// The mean distance between where `t` and the true transform put the real points.
fn true_error(t: &Rigid, truth: &Rigid, points: &[Point], real: &[bool]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for (&p, &r) in points.iter().zip(real) {
        if r {
            let (a, b) = (t.apply(p), truth.apply(p));
            sum += (a.0 - b.0).hypot(a.1 - b.1);
            n += 1;
        }
    }
    sum / n as f32
}

/// The image of time point 1 (dimmed), its centroids as blue dots and the points of time
/// point 2 moved by `t` as orange rings.
fn picture(img: &Image<f32>, t1: &[Point], t2: &[Point], t: &Rigid) -> Image<[f32; 3]> {
    let mut out = img.map(|v| [0.6 * v, 0.6 * v, 0.6 * v]);
    let mut paint = |p: Point, r0: f32, r1: f32, colour: [f32; 3]| {
        for dy in -6..=6 {
            for dx in -6..=6 {
                let d = (dx as f32).hypot(dy as f32);
                let (x, y) = (p.0.round() as isize + dx, p.1.round() as isize + dy);
                if d >= r0 && d <= r1 && out.contains(x, y) {
                    out[(x as usize, y as usize)] = colour;
                }
            }
        }
    };
    for &p in t1 {
        paint(p, 0.0, 2.2, [0.2, 0.55, 1.0]);
    }
    for &p in t2 {
        paint(t.apply(p), 3.5, 5.0, [1.0, 0.55, 0.1]);
    }
    out
}

pub fn main() {
    let dir = "out/icp";
    let (img, labels) = samples::nuclei(256, 30, 7);
    let t1 = label_centroids(&labels, 30);
    let (t2, real, truth) = time_point_2(&t1, 8.0, 1);
    println!("time point 1: {} nuclei; time point 2: {} ({} of them new)", t1.len(), t2.len(), real.iter().filter(|&&r| !r).count());
    println!("true transform (time point 2 → 1): angle {:.2}°, shift ({:.2}, {:.2})", truth.angle.to_degrees(), truth.tx, truth.ty);
    save_rgb(&picture(&img, &t1, &t2, &Rigid::IDENTITY), format!("{dir}/before.png"));

    // ICP with all pairs, then trimmed (keep 90 %: the 27 nuclei seen twice out of 30).
    for keep in [1.0, 0.9] {
        let (t, rms) = icp(&t2, &t1, Rigid::IDENTITY, 50, keep);
        let list: Vec<String> = rms.iter().map(|r| format!("{r:.2}")).collect();
        println!(
            "\nICP, keep {:.0} %: {} iterations, RMS per iteration [{}]",
            100.0 * keep,
            rms.len(),
            list.join(", ")
        );
        println!("  angle {:.2}°, shift ({:.2}, {:.2}), true error {:.2} px", t.angle.to_degrees(), t.tx, t.ty, true_error(&t, &truth, &t2, &real));
        let name = if keep == 1.0 { "after-all.png" } else { "after-trimmed.png" };
        save_rgb(&picture(&img, &t1, &t2, &t), format!("{dir}/{name}"));
    }

    // How far may the stage turn before ICP from no rotation fails? Trimmed ICP (keep 90 %).
    println!("\ntrue error of trimmed ICP (keep 90 %) against the turn between the time points:");
    for degrees in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0] {
        let (t2, real, truth) = time_point_2(&t1, degrees, 1);
        let (t, rms) = icp(&t2, &t1, Rigid::IDENTITY, 50, 0.9);
        println!(
            "  {degrees:4.0}°: true error {:6.2} px, final RMS {:5.2} px, found {:6.2}°",
            true_error(&t, &truth, &t2, &real),
            rms.last().unwrap(),
            -t.angle.to_degrees()
        );
    }
    println!("wrote the pictures to {dir}/");
}
