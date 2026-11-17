// Three routes from a synthetic nucleus image to objects: the semantic mask, the boundary
// class and the distance map. The "network outputs" are simulated from the true labels,
// with errors that can be switched on. Results in out/instances/.
//   cargo run -p instances                      (your code)
//   cargo run -p instances --features solution  (the model solution)

use imgcourse::*;
use instances::*;

/// The three-class target of the boundary route. With `contact_errors`, the network
/// "misses" the boundary where two cells touch: those pixels are predicted as interior.
fn three_classes(truth: &Image<u32>, contact_errors: bool) -> Image<u8> {
    let (w, h) = (truth.width(), truth.height());
    truth.map_xy(|x, y, l| {
        if l == 0 {
            return BACKGROUND;
        }
        let ns = neighbours4(x, y, w, h);
        let touches_cell = ns.iter().any(|&q| truth[q] != 0 && truth[q] != l);
        let at_edge = ns.len() < 4 || ns.iter().any(|&q| truth[q] != l);
        if touches_cell && contact_errors {
            INTERIOR
        } else if at_edge {
            BOUNDARY
        } else {
            INTERIOR
        }
    })
}

/// The distance target: for every cell pixel, the distance to the nearest pixel outside its
/// cell, divided by the largest such distance in the cell (1 at the centre).
fn distance_target(truth: &Image<u32>) -> Image<f32> {
    let r = 20isize;
    let raw = truth.map_xy(|x, y, l| {
        if l == 0 {
            return 0.0;
        }
        let mut best = r as f32;
        for dy in -r..=r {
            for dx in -r..=r {
                let (qx, qy) = (x as isize + dx, y as isize + dy);
                let outside = !truth.contains(qx, qy) || truth[(qx as usize, qy as usize)] != l;
                if outside {
                    best = best.min(((dx * dx + dy * dy) as f32).sqrt());
                }
            }
        }
        best
    });
    let mut largest = std::collections::HashMap::new();
    for p in truth.coords() {
        let m = largest.entry(truth[p]).or_insert(0.0f32);
        *m = m.max(raw[p]);
    }
    raw.map_xy(|x, y, d| if truth[(x, y)] == 0 { 0.0 } else { d / largest[&truth[(x, y)]] })
}

/// Smooth random errors: white noise blurred 4 times with a 3 × 3 box, scaled to sd `sd`.
fn smooth_noise(w: usize, h: usize, sd: f32, seed: u64) -> Image<f32> {
    let mut n = samples::add_noise(&Image::new(w, h, 0.0), 1.0, seed);
    for _ in 0..4 {
        n = Image::from_fn(w, h, |x, y| {
            let mut s = 0.0;
            for d in 0..9 {
                s += n.get_border(x as isize + d % 3 - 1, y as isize + d / 3 - 1, Border::Mirror);
            }
            s / 9.0
        });
    }
    let rms = (n.pixels().iter().map(|v| v * v).sum::<f32>() / (w * h) as f32).sqrt();
    n.map(|v| v * sd / rms)
}

fn report(name: &str, truth: &Image<u32>, labels: &Image<u32>, dir: &str, file: &str) {
    let (tp, fp, fn_, ious) = match_objects(truth, labels);
    let (sq, rq, pq) = panoptic_quality(&ious, fp, fn_);
    let n = labels.pixels().iter().cloned().max().unwrap_or(0);
    println!("{name:<40} {n:3} objects  TP {tp:2} FP {fp:2} FN {fn_:2}  SQ {sq:.2} RQ {rq:.2} PQ {pq:.2}");
    save_labels(labels, format!("{dir}/{file}.png"));
}

fn main() {
    let dir = "out/instances";
    let (img, truth) = samples::nuclei(128, 20, 2);
    save_gray(&img, format!("{dir}/input.png"));
    save_labels(&truth, format!("{dir}/truth.png"));
    let cells = truth.pixels().iter().cloned().max().unwrap();
    println!("{} true nuclei (some drawn over others, so fewer may be visible)", cells);
    report("truth itself", &truth, &truth, dir, "truth");

    // Semantic: threshold the image, label the connected pieces.
    let semantic = label_components(&img.map(|v| v > 0.3));
    report("semantic mask + labelling", &truth, &semantic, dir, "semantic");

    // Boundary class, perfect and with the boundary missed where cells touch.
    report("boundary class, perfect", &truth, &boundary_route(&three_classes(&truth, false)), dir, "boundary");
    report("boundary class, contacts missed", &truth, &boundary_route(&three_classes(&truth, true)), dir, "boundary-contacts");

    // Distance map with smooth errors, for several h.
    let target = distance_target(&truth);
    let noise = smooth_noise(128, 128, 0.15, 3);
    let predicted = target.zip_map(&noise, |d, e| if d > 0.0 { d + e } else { 0.0 });
    save_gray_stretched(&predicted, format!("{dir}/distance.png"));
    for h in [0.05, 0.1, 0.2, 0.4, 0.8] {
        let labels = distance_route(&predicted, 0.1, h);
        report(&format!("distance map with errors, h = {h}"), &truth, &labels, dir, &format!("distance-h{h}"));
    }
    println!("wrote the results to {dir}/");
}
