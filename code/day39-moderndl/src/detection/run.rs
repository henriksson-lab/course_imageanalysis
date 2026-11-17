// Boxes and non-maximum suppression on synthetic nucleus images, with a simulated detector
// (jittered boxes around the true nuclei, plus a few false boxes). Results in out/detection/.
//   cargo run -p day39 -- detection                      (your code)
//   cargo run -p day39 --features solution -- detection  (the model solution)

use crate::detection::*;
use imgcourse::*;

/// The simulated detector: 5 jittered copies of every true box (each side moved by about
/// 1 pixel), scored by how well they fit (0.4 + 0.5 IoU, with a little noise), and 3 false
/// boxes with low scores.
fn simulated_detector(truth: &[Detection], size: f32, seed: u64) -> Vec<Detection> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for t in truth {
        for _ in 0..5 {
            let j = |rng: &mut Rng| rng.normal();
            let b = Detection::new(t.x0 + j(&mut rng), t.x1 + j(&mut rng), t.y0 + j(&mut rng), t.y1 + j(&mut rng), 0.0);
            let score = (0.4 + 0.5 * iou(&b, t) + 0.05 * rng.normal()).clamp(0.0, 1.0);
            out.push(Detection { score, ..b });
        }
    }
    for _ in 0..3 {
        let (x, y) = (rng.uniform() * (size - 12.0), rng.uniform() * (size - 12.0));
        out.push(Detection::new(x, x + 12.0, y, y + 12.0, 0.2 + 0.2 * rng.uniform()));
    }
    out
}

/// Draw the outline of a box on a colour image.
fn draw(img: &mut Image<[f32; 3]>, b: &Detection, colour: [f32; 3]) {
    let (x0, x1, y0, y1) = (b.x0.round() as isize, b.x1.round() as isize - 1, b.y0.round() as isize, b.y1.round() as isize - 1);
    for x in x0..=x1 {
        for y in [y0, y1] {
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
    for y in y0..=y1 {
        for x in [x0, x1] {
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = colour;
            }
        }
    }
}

fn run(name: &str, nuclei: usize, seed: u64, dir: &str) {
    let (img, labels) = samples::nuclei(128, nuclei, seed);
    // only nuclei that are still visible (later ones can cover earlier ones)
    let truth: Vec<Detection> = boxes_from_labels(&labels).into_iter().filter(|b| b.x1 > b.x0).collect();
    let candidates = simulated_detector(&truth, 128.0, seed + 100);
    let neighbour_iou = truth.iter().enumerate().map(|(i, a)| truth.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, b)| iou(a, b)).fold(0.0, f32::max)).fold(0.0, f32::max);
    println!("{name}: {} nuclei, {} candidate boxes; largest IoU between two true boxes {neighbour_iou:.2}", truth.len(), candidates.len());
    println!("  {:>4} {:>6} {:>4} {:>4} {:>4} {:>6}", "t", "boxes", "TP", "FP", "FN", "F1");
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9] {
        let kept: Vec<Detection> = nms(&candidates, t).iter().map(|&i| candidates[i]).collect();
        let (tp, fp, fn_) = count_matches(&kept, &truth, 0.5);
        let f1 = 2.0 * tp as f32 / (2 * tp + fp + fn_) as f32;
        println!("  {t:4} {:6} {tp:4} {fp:4} {fn_:4} {f1:6.2}", kept.len());
    }
    let (before, after) = (gray_to_rgb(&img), gray_to_rgb(&img));
    let (mut before, mut after) = (before, after);
    for b in &candidates {
        draw(&mut before, b, [1.0, 0.8, 0.1]);
    }
    for &i in &nms(&candidates, 0.5) {
        draw(&mut after, &candidates[i], [0.1, 0.9, 0.2]);
    }
    save_rgb(&before, format!("{dir}/{name}-candidates.png"));
    save_rgb(&after, format!("{dir}/{name}-nms0.5.png"));
}

pub fn main() {
    let dir = "out/detection";
    run("sparse", 12, 4, dir);
    run("crowded", 45, 5, dir);
    println!("wrote the results to {dir}/ (yellow: candidates, green: kept by NMS with t = 0.5)");
}
