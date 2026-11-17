// Matches points of the micrograph with points of a moved, slightly turned, brighter and
// noisier copy, with nearest neighbours, a distance threshold, the ratio test and the mutual
// check, and scores them against the known transform; pictures in out/feature-matching/.
//   cargo run -p day24 -- feature-matching                      (your code)
//   cargo run -p day24 --features solution -- feature-matching  (the model solution)

use crate::common::blur;
use crate::feature_matching::*;
use imgcourse::*;

const ANGLE: f32 = 5.0; // view 2 is turned by 5° counter-clockwise on the screen
const W: usize = 152; // both views are 152 × 152 windows of the micrograph
const A: (f32, f32) = (0.0, 64.0); // top-left corner of view 1 in the micrograph
const B: (f32, f32) = (56.0, 96.0); // top-left corner of view 2 (before turning)
const SIZE: usize = 15; // descriptor: a normalised 15 × 15 patch
const EPS: f32 = 3.0; // a match is correct if the transform puts the point within 3 px

/// Where a point of view 1 lands in view 2: into the micrograph, turned about its centre
/// (128, 128), into view 2.
fn forward(x: f32, y: f32) -> (f32, f32) {
    let (s, c) = ANGLE.to_radians().sin_cos();
    let (dx, dy) = (x + A.0 - 128.0, y + A.1 - 128.0);
    (128.0 + c * dx + s * dy - B.0, 128.0 - s * dx + c * dy - B.1)
}

fn view1(img: &Image<f32>) -> Image<f32> {
    img.crop(A.0 as usize, A.1 as usize, W, W)
}

fn view2(img: &Image<f32>) -> Image<f32> {
    let (s, c) = ANGLE.to_radians().sin_cos();
    let moved = Image::from_fn(W, W, |x, y| {
        let (dx, dy) = (x as f32 + B.0 - 128.0, y as f32 + B.1 - 128.0);
        bilinear(img, 128.0 + c * dx - s * dy, 128.0 + s * dx + c * dy)
    });
    samples::add_noise(&moved.map(|v| 1.2 * v), 0.03, 11)
}

/// Blob-like points: the 150 strongest local extrema of a difference of Gaussians (σ = 2),
/// away from the border.
fn detect(img: &Image<f32>) -> Vec<(usize, usize)> {
    let dog = blur(img, 2.0 * 1.26).zip_map(&blur(img, 2.0), |a, b| a - b);
    let mut found = Vec::new();
    let m = SIZE / 2 + 1;
    for y in m..img.height() - m {
        for x in m..img.width() - m {
            let v = dog[(x, y)].abs();
            let neighbours = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
            if neighbours.iter().all(|&(dx, dy)| v > dog[((x as isize + dx) as usize, (y as isize + dy) as usize)].abs()) {
                found.push((v, x, y));
            }
        }
    }
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.iter().take(150).map(|&(_, x, y)| (x, y)).collect()
}

/// The normalised patch (minus its mean, divided by its length) around every point.
fn describe(img: &Image<f32>, points: &[(usize, usize)]) -> Vec<Vec<f32>> {
    points
        .iter()
        .map(|&(x, y)| {
            let patch = img.crop(x - SIZE / 2, y - SIZE / 2, SIZE, SIZE);
            let mean = patch.mean();
            let v: Vec<f32> = patch.pixels().iter().map(|p| p - mean).collect();
            let length = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-6);
            v.iter().map(|a| a / length).collect()
        })
        .collect()
}

pub fn main() {
    let dir = "out/feature-matching";
    let cell = samples::cell();
    let (img1, img2) = (view1(&cell), view2(&cell));
    let (p1, p2) = (detect(&img1), detect(&img2));
    let table = distance_table(&describe(&img1, &p1), &describe(&img2, &p2));
    // The true partner of each point of view 1: the nearest point of view 2 to where the
    // transform puts it, if within EPS.
    let truth: Vec<Option<usize>> = p1
        .iter()
        .map(|&(x, y)| {
            let (u, v) = forward(x as f32, y as f32);
            let dist = |j: usize| (p2[j].0 as f32 - u).hypot(p2[j].1 as f32 - v);
            (0..p2.len()).min_by(|&a, &b| dist(a).total_cmp(&dist(b))).filter(|&j| dist(j) < EPS)
        })
        .collect();
    println!("{} points in view 1, {} in view 2, {} with a true partner", p1.len(), p2.len(), truth.iter().flatten().count());

    let report = |name: &str, m: &[(usize, usize)]| {
        let (correct, p, r) = precision_recall(m, &truth);
        println!("  {name:28} {:3} accepted, {correct:3} correct: precision {:3.0} %, recall {:3.0} %", m.len(), 100.0 * p, 100.0 * r);
    };
    let nearest = ratio_test(&table, f32::INFINITY);
    report("nearest only", &nearest);
    for d in [0.6, 0.8] {
        let threshold: Vec<_> = nearest.iter().copied().filter(|&(i, j)| table[i][j] < d).collect();
        report(&format!("distance below {d}"), &threshold);
    }
    for t in [0.6, 0.7, 0.8, 0.9] {
        report(&format!("ratio {t}"), &ratio_test(&table, t));
    }
    let both = mutual_check(&table, &ratio_test(&table, 0.8));
    report("ratio 0.8 + mutual", &both);
    report("nearest + mutual", &mutual_check(&table, &nearest));

    // The distances d1 of correct and of wrong nearest neighbours, and their ratios.
    let (mut good, mut bad) = (Vec::new(), Vec::new());
    for (i, row) in table.iter().enumerate() {
        let (j, d1, d2) = nearest_two(row);
        if truth[i] == Some(j) { good.push((d1, d1 / d2)) } else { bad.push((d1, d1 / d2)) }
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "median d1: correct {:.2}, wrong {:.2};  median ratio: correct {:.2}, wrong {:.2}",
        median(&mut good.iter().map(|g| g.0).collect()),
        median(&mut bad.iter().map(|b| b.0).collect()),
        median(&mut good.iter().map(|g| g.1).collect()),
        median(&mut bad.iter().map(|b| b.1).collect())
    );

    save_rgb(&draw_matches(&img1, &img2, &p1, &p2, &nearest, &truth), format!("{dir}/nearest.png"));
    save_rgb(&draw_matches(&img1, &img2, &p1, &p2, &both, &truth), format!("{dir}/ratio-mutual.png"));
    println!("wrote the pictures to {dir}/ (green = correct, red = wrong)");
}

/// The two views side by side, with a line for every match.
fn draw_matches(a: &Image<f32>, b: &Image<f32>, p1: &[(usize, usize)], p2: &[(usize, usize)], matches: &[(usize, usize)], truth: &[Option<usize>]) -> Image<[f32; 3]> {
    let w = a.width();
    let mut out = gray_to_rgb(&Image::from_fn(2 * w + 8, a.height(), |x, y| {
        if x < w {
            a[(x, y)]
        } else if x >= w + 8 {
            (b[(x - w - 8, y)] / 1.2).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }));
    for &(i, j) in matches {
        let colour = if truth[i] == Some(j) { [0.1, 0.9, 0.2] } else { [1.0, 0.1, 0.1] };
        let (x0, y0) = (p1[i].0 as f32, p1[i].1 as f32);
        let (x1, y1) = ((p2[j].0 + w + 8) as f32, p2[j].1 as f32);
        let steps = (x1 - x0).hypot(y1 - y0) as usize + 1;
        for k in 0..=steps {
            let t = k as f32 / steps as f32;
            let (x, y) = ((x0 + t * (x1 - x0)).round() as usize, (y0 + t * (y1 - y0)).round() as usize);
            out[(x, y)] = colour;
        }
    }
    out
}

fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Clamp);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}
