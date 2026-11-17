// Runs your k-means on the grey values of the course micrograph, on the points of B2 with
// many random starts, and on two elongated groups; pictures in out/kmeans/.
//   cargo run -p kmeans                      (your code)
//   cargo run -p kmeans --features solution  (the model solution)

use imgcourse::*;
use kmeans::*;

/// A picture of 2D points `lo..hi` (x1 to the right, x2 up), each a dot in the colour of its
/// group, the background coloured by the nearest centre, and the centres as black crosses.
fn plot(points: &[Vec<f32>], groups: &[usize], centres: &[Vec<f32>], lo: [f32; 2], hi: [f32; 2], size: usize) -> Image<[f32; 3]> {
    let colour = |g: usize| io::label_colours(&Image::new(1, 1, g as u32 + 1))[(0, 0)];
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let mut img = Image::from_fn(size, size, |px, py| {
        let x = vec![lo[0] + (px as f32 + 0.5) * scale[0], hi[1] - (py as f32 + 0.5) * scale[1]];
        let (g, _) = assign(&[x], centres);
        colour(g[0]).map(|c| 0.85 + 0.15 * c)
    });
    let mut dot = |p: &[f32], r: isize, c: [f32; 3], cross: bool| {
        let (cx, cy) = (((p[0] - lo[0]) / scale[0]) as isize, ((hi[1] - p[1]) / scale[1]) as isize);
        for dy in -r..=r {
            for dx in -r..=r {
                let on = if cross { dx == dy || dx == -dy } else { dx * dx + dy * dy <= r * r };
                if on && img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = c;
                }
            }
        }
    };
    for (p, &g) in points.iter().zip(groups) {
        dot(p, 3, [0.0; 3], false);
        dot(p, 2, colour(g), false);
    }
    for c in centres {
        dot(c, 6, [0.0; 3], true);
    }
    img
}

/// The best of `starts` k-means runs from k-means++ seeds (the smallest final J).
fn best_of(points: &[Vec<f32>], k: usize, starts: usize, rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<usize>, f32) {
    let mut best = (Vec::new(), Vec::new(), f32::INFINITY);
    for _ in 0..starts {
        let (m, g, trace) = kmeans(points, &kmeanspp(points, k, rng), 100);
        let j = *trace.last().unwrap();
        if j < best.2 {
            best = (m, g, j);
        }
    }
    best
}

fn main() {
    let dir = "out/kmeans";
    let mut rng = Rng::new(1);

    // 1. Grey values of the micrograph (0 to 255) as 1D points: K = 2 is isodata.
    let img = samples::cell();
    save_gray(&img, format!("{dir}/cell.png"));
    let grey: Vec<Vec<f32>> = img.pixels().iter().map(|&v| vec![255.0 * v]).collect();
    let (m, _, trace) = kmeans(&grey, &[vec![0.0], vec![255.0]], 100);
    println!("micrograph, K = 2 from (0, 255): centres {:.1} and {:.1}, boundary {:.1}, {} assign steps", m[0][0], m[1][0], (m[0][0] + m[1][0]) / 2.0, trace.len());
    for k in [3, 4] {
        let (m, groups, j) = best_of(&grey, k, 5, &mut rng);
        let mut order: Vec<usize> = (0..k).collect(); // number the groups from dark to bright
        order.sort_by(|&a, &b| m[a][0].total_cmp(&m[b][0]));
        let rank: Vec<usize> = (0..k).map(|g| order.iter().position(|&o| o == g).unwrap()).collect();
        let means: Vec<String> = order.iter().map(|&g| format!("{:.0}", m[g][0])).collect();
        println!("micrograph, K = {k} (best of 5 k-means++ starts): centres {}, J = {j:.3e}", means.join(", "));
        let labels = Image::from_vec(img.width(), img.height(), groups.iter().map(|&g| rank[g] as f32 / (k - 1) as f32).collect());
        save_gray(&labels, format!("{dir}/cell-k{k}.png"));
    }

    // 2. The points of B2 with 100 random starts: random points, and k-means++.
    let b2: Vec<Vec<f32>> = [0.0, 1.0, 5.0, 6.0, 10.0, 11.0].iter().map(|&x| vec![x]).collect();
    let (mut random_good, mut pp_good) = (0, 0);
    for _ in 0..100 {
        // three different random points
        let mut idx: Vec<usize> = Vec::new();
        while idx.len() < 3 {
            let i = rng.below(6);
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        let start: Vec<Vec<f32>> = idx.iter().map(|&i| b2[i].clone()).collect();
        let (_, _, trace) = kmeans(&b2, &start, 100);
        random_good += (*trace.last().unwrap() < 1.6) as usize;
        let (_, _, trace) = kmeans(&b2, &kmeanspp(&b2, 3, &mut rng), 100);
        pp_good += (*trace.last().unwrap() < 1.6) as usize;
    }
    println!("B2, 100 starts: J = 1.5 reached from {random_good} random starts, from {pp_good} k-means++ starts");

    // 3. Two long parallel groups (80 points each), K = 2, best of 10 starts.
    let (mut cigars, mut truth) = (Vec::new(), Vec::new());
    for (g, y0) in [(0, 0.0), (1, 1.0)] {
        for _ in 0..80 {
            cigars.push(vec![4.0 * rng.uniform(), y0 + 0.15 * rng.normal()]);
            truth.push(g);
        }
    }
    let (m, groups, j) = best_of(&cigars, 2, 10, &mut rng);
    let same = (0..160).filter(|&i| groups[i] == truth[i]).count();
    let agree = same.max(160 - same); // the group numbers may be swapped
    let true_centres = update(&cigars, &truth, &m);
    let j_true: f32 = (0..160).map(|i| dist2(&cigars[i], &true_centres[truth[i]])).sum();
    println!("two long groups: k-means J = {j:.3}, agrees with the true groups for {agree} of 160 points; the true groups have J = {j_true:.3}");
    save_rgb(&plot(&cigars, &groups, &m, [-0.5, -1.5], [4.5, 2.5], 300), format!("{dir}/long-groups.png"));
    let (m, groups, j) = best_of(&cigars.iter().map(|p| vec![p[0], 3.0 * p[1]]).collect::<Vec<_>>(), 2, 10, &mut rng);
    let same = (0..160).filter(|&i| groups[i] == truth[i]).count();
    println!("  with x2 multiplied by 3: J = {j:.3}, agrees for {} of 160 points", same.max(160 - same));
    let stretched: Vec<Vec<f32>> = cigars.iter().map(|p| vec![p[0], 3.0 * p[1]]).collect();
    save_rgb(&plot(&stretched, &groups, &m, [-0.5, -1.5], [4.5, 4.5], 300), format!("{dir}/long-groups-x2-times-3.png"));

    // 4. The elbow: four groups, best J for K = 1 to 7.
    let mut four = Vec::new();
    for (cx, cy) in [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)] {
        for _ in 0..30 {
            four.push(vec![cx + 0.6 * rng.normal(), cy + 0.6 * rng.normal()]);
        }
    }
    let js: Vec<String> = (1..=7).map(|k| format!("{:.0}", best_of(&four, k, 10, &mut rng).2)).collect();
    println!("four groups, best J for K = 1..7: {}", js.join(", "));
    println!("wrote the pictures to {dir}/");
}
