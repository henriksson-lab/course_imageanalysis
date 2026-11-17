// The flow route on a synthetic scene: heat targets and arrows for five cells (a C, a bent
// rod, a disc and two touching discs), then the decoding (follow, group, flow check) on
// exact arrows, with too few steps, with noisy arrows, and with one field shared by the
// touching pair. Results in out/cellpose/.
//   cargo run -p cellnets -- cellpose                      (your code)
//   cargo run -p cellnets --features solution -- cellpose  (the model solution)

use crate::cellpose::*;
use imgcourse::*;

const SIZE: usize = 128;
const MIN_COUNT: usize = 3;

/// The true cells, labels 1 ... 5.
fn scene() -> Image<u32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (x, y) = (x as f32, y as f32);
        let polar = |cx: f32, cy: f32| {
            let (dx, dy) = (x - cx, y - cy);
            ((dx * dx + dy * dy).sqrt(), dy.atan2(dx))
        };
        let (r, a) = polar(36.0, 36.0);
        if (12.0..24.0).contains(&r) && a.abs() > 0.95 {
            return 1; // a C, open to the right
        }
        let (r, a) = polar(92.0, 10.0);
        if (34.0..42.0).contains(&r) && (0.6..2.4).contains(&a) {
            return 2; // a bent rod (a piece of a ring)
        }
        if (x - 30.0).powi(2) + (y - 98.0).powi(2) <= 15.0f32.powi(2) {
            return 3; // a disc
        }
        if (x - 78.0).powi(2) + (y - 96.0).powi(2) <= 14.0f32.powi(2) {
            return 4; // two touching discs
        }
        if (x - 104.0).powi(2) + (y - 96.0).powi(2) <= 13.0f32.powi(2) {
            return 5;
        }
        0
    })
}

/// The mask of one label.
fn mask_of(labels: &Image<u32>, l: u32) -> Image<bool> {
    labels.map(|v| v == l)
}

/// The target arrows of every cell, each from its own heat target (2 × (width + height)
/// steps of its box), put together in one image.
fn target_flow(labels: &Image<u32>) -> Image<[f32; 2]> {
    let mut flow = Image::new(SIZE, SIZE, [0.0f32; 2]);
    let n = labels.pixels().iter().copied().max().unwrap_or(0);
    for l in 1..=n {
        let mask = mask_of(labels, l);
        let pts: Vec<(usize, usize)> = mask.coords().filter(|&p| mask[p]).collect();
        if pts.is_empty() {
            continue;
        }
        let w = pts.iter().map(|p| p.0).max().unwrap() - pts.iter().map(|p| p.0).min().unwrap() + 1;
        let h = pts.iter().map(|p| p.1).max().unwrap() - pts.iter().map(|p| p.1).min().unwrap() + 1;
        let a = arrows(&heat_target(&mask, 2 * (w + h)), &mask);
        for p in pts {
            flow[p] = a[p];
        }
    }
    flow
}

/// The decoding: follow, group (sink pixels need at least `MIN_COUNT` particles), flow check. Returns the
/// labels after the check and the number of masks it removed.
fn decode(flow: &Image<[f32; 2]>, fg: &Image<bool>, steps: usize, threshold: f32) -> (Image<u32>, usize) {
    let masks = group_ends(&follow(flow, fg, steps), fg, MIN_COUNT);
    let own = target_flow(&masks);
    let n = masks.pixels().iter().copied().max().unwrap_or(0);
    let mut keep = vec![0u32; n as usize + 1];
    let (mut next, mut removed) = (1, 0);
    for l in 1..=n {
        let pix: Vec<(usize, usize)> = masks.coords().filter(|&p| masks[p] == l).collect();
        let pred: Vec<[f32; 2]> = pix.iter().map(|&p| flow[p]).collect();
        let mine: Vec<[f32; 2]> = pix.iter().map(|&p| own[p]).collect();
        if flow_error(&pred, &mine) <= threshold {
            keep[l as usize] = next;
            next += 1;
        } else {
            removed += 1;
        }
    }
    (masks.map(|l| keep[l as usize]), removed)
}

/// TP, FP, FN at IoU ≥ 0.5.
fn score(truth: &Image<u32>, pred: &Image<u32>) -> (usize, usize, usize) {
    let (nt, np) = (truth.pixels().iter().copied().max().unwrap() as usize, pred.pixels().iter().copied().max().unwrap_or(0) as usize);
    let mut inter = vec![vec![0usize; np + 1]; nt + 1];
    for (&t, &p) in truth.pixels().iter().zip(pred.pixels()) {
        inter[t as usize][p as usize] += 1;
    }
    let area_t: Vec<usize> = (0..=nt).map(|t| inter[t].iter().sum()).collect();
    let area_p: Vec<usize> = (0..=np).map(|p| (0..=nt).map(|t| inter[t][p]).sum()).collect();
    let mut tp = 0;
    for t in 1..=nt {
        for p in 1..=np {
            let i = inter[t][p];
            if i > 0 && i as f32 / (area_t[t] + area_p[p] - i) as f32 >= 0.5 {
                tp += 1;
            }
        }
    }
    (tp, np - tp, nt - tp)
}

/// Arrows as colours: the direction is the hue, the length the brightness.
fn flow_picture(flow: &Image<[f32; 2]>) -> Image<[f32; 3]> {
    flow.map(|[vx, vy]| {
        let len = (vx * vx + vy * vy).sqrt().min(1.0);
        let a = vy.atan2(vx);
        let c = |shift: f32| len * (0.5 + 0.5 * (a + shift).cos());
        [c(0.0), c(-2.094), c(2.094)]
    })
}

pub fn main() {
    let dir = "out/cellpose";
    let truth = scene();
    let fg = truth.map(|l| l > 0);
    save_labels(&truth, format!("{dir}/truth.png"));

    // The training target of the C: the heat after 10 steps and after the default number.
    let c = mask_of(&truth, 1);
    for steps in [10, 192] {
        save_gray_stretched(&heat_target(&c, steps).map(|h| (1.0 + 1e4 * h).ln()), format!("{dir}/heat-c-{steps}.png"));
    }
    let exact = target_flow(&truth);
    save_rgb(&flow_picture(&exact), format!("{dir}/flows.png"));

    // Noisy arrows: the exact ones plus smooth noise (white noise averaged over 7 × 7 pixels,
    // standard deviation NOISE in each component), scaled to length 1.
    const NOISE: f32 = 0.5;
    let mut rng = Rng::new(5);
    let smooth_noise = |rng: &mut Rng| {
        let white = Image::from_fn(SIZE, SIZE, |_, _| rng.normal());
        let box7 = |img: &Image<f32>, along_x: bool| {
            img.map_xy(|x, y, _| {
                (-3..=3isize)
                    .map(|d| if along_x { img.get_border(x as isize + d, y as isize, Border::Mirror) } else { img.get_border(x as isize, y as isize + d, Border::Mirror) })
                    .sum::<f32>()
                    / 7.0
            })
        };
        box7(&box7(&white, true), false).map(|v| v * 7.0 * NOISE) // a 7 × 7 mean of white noise has sd 1/7
    };
    let (nx, ny) = (smooth_noise(&mut rng), smooth_noise(&mut rng));
    let noisy = Image::from_fn(SIZE, SIZE, |x, y| {
        let [vx, vy] = exact[(x, y)];
        if !fg[(x, y)] {
            return [0.0, 0.0];
        }
        let (a, b) = (vx + nx[(x, y)], vy + ny[(x, y)]);
        let len = (a * a + b * b).sqrt().max(1e-6);
        [a / len, b / len]
    });
    // One field for the touching pair: the target of their union.
    let shared = target_flow(&truth.map(|l| if l == 5 { 4 } else { l }));

    println!("5 true cells (a C, a bent rod, a disc, two touching discs)");
    let cases = [("exact", &exact, 200, 0.4), ("few-steps", &exact, 10, 0.4), ("few-steps-nocheck", &exact, 10, 99.0), ("noisy", &noisy, 200, 0.4), ("noisy-nocheck", &noisy, 200, 99.0), ("shared", &shared, 200, 0.4)];
    for (name, flow, steps, threshold) in cases {
        let (labels, removed) = decode(flow, &fg, steps, threshold);
        let (tp, fp, fn_) = score(&truth, &labels);
        let n = labels.pixels().iter().copied().max().unwrap_or(0);
        let unlabelled = fg.coords().filter(|&p| fg[p] && labels[p] == 0).count();
        println!("{name:18} {steps:3} steps: {n} objects ({removed} removed by the flow check), {unlabelled:4} pixels without object   TP {tp}  FP {fp}  FN {fn_}");
        save_labels(&labels, format!("{dir}/labels-{name}.png"));
    }
    save_rgb(&flow_picture(&noisy), format!("{dir}/flows-noisy.png"));
    println!("wrote the results to {dir}/");
}
