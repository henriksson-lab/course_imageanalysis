// A simulated one-pass detector on synthetic nuclei: every grid cell (strides 8 and 16)
// whose centre lies in a nucleus answers for it, with distances encoded by your C1 (plus
// noise) and a score that is highest near the centre. Some nuclei also get a "debris" score.
// Then your decoding and NMS, and a centre heat map with its peaks. Results in out/yolo/.
//   cargo run -p day40 -- yolo                      (your code)
//   cargo run -p day40 --features solution -- yolo  (the model solution)

use imgcourse::*;
use crate::yolo::*;

const SIZE: usize = 256;

/// The box around every label.
fn true_boxes(labels: &Image<u32>) -> Vec<Detection> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut b = vec![(f32::MAX, f32::MAX, f32::MIN, f32::MIN); n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        let (fx, fy) = (x as f32, y as f32);
        b[l] = (b[l].0.min(fx), b[l].1.min(fy), b[l].2.max(fx + 1.0), b[l].3.max(fy + 1.0));
    }
    b.iter().skip(1).filter(|v| v.0 < f32::MAX).map(|&(x0, y0, x1, y1)| Detection { x0, y0, x1, y1, score: 1.0, class: 0 }).collect()
}

/// The simulated network: the candidates of every grid cell of strides 8 and 16.
fn network(truth: &[Detection], rng: &mut Rng, box_noise: f32) -> Vec<Detection> {
    let mut out = Vec::new();
    for stride in [8.0f32, 16.0] {
        let cells = (SIZE as f32 / stride) as usize;
        for row in 0..cells {
            for col in 0..cells {
                let (cx, cy) = (stride * (col as f32 + 0.5), stride * (row as f32 + 0.5));
                // the smallest true box that contains the cell centre answers
                let owner = truth
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.x0 < cx && cx < b.x1 && b.y0 < cy && cy < b.y1)
                    .min_by(|a, b| a.1.area().total_cmp(&b.1.area()));
                let Some((index, b)) = owner else { continue };
                let size = (b.x1 - b.x0).max(b.y1 - b.y0) / stride;
                if !(2.0..=12.0).contains(&size) {
                    continue; // this stride does not answer for objects of this size
                }
                let d = [cx - b.x0, cy - b.y0, b.x1 - cx, b.y1 - cy];
                let dists = d.map(|v| distance_target(v / stride + box_noise * rng.normal()));
                // centredness: 1 at the centre of the box, 0 at its edge
                let c = ((d[0].min(d[2]) / d[0].max(d[2])) * (d[1].min(d[3]) / d[1].max(d[3]))).sqrt();
                let cell_score = sigmoid(5.0 * c - 2.0 + 0.5 * rng.normal());
                out.push(decode_cell(stride, row, col, &dists, cell_score, 0));
                if index % 6 == 0 {
                    // this nucleus also looks a bit like debris to the network
                    out.push(decode_cell(stride, row, col, &dists, 0.8 * cell_score, 1));
                }
            }
        }
    }
    out
}

/// Kept boxes against the true boxes (all of class 0): greedy matching by score at IoU ≥ 0.5,
/// the class must match. Returns TP, FP, FN.
fn score(kept: &[Detection], truth: &[Detection]) -> (usize, usize, usize) {
    let mut used = vec![false; truth.len()];
    let mut order: Vec<&Detection> = kept.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut tp = 0;
    for d in order {
        let best = (0..truth.len()).filter(|&j| !used[j] && d.class == truth[j].class).max_by(|&a, &b| iou(d, &truth[a]).total_cmp(&iou(d, &truth[b])));
        if let Some(j) = best {
            if iou(d, &truth[j]) >= 0.5 {
                used[j] = true;
                tp += 1;
            }
        }
    }
    (tp, kept.len() - tp, truth.len() - tp)
}

/// Draw box outlines on a colour image.
fn draw(img: &mut Image<[f32; 3]>, boxes: &[Detection]) {
    for b in boxes {
        let colour = if b.class == 0 { [0.2, 0.9, 0.2] } else { [1.0, 0.3, 0.8] };
        let (x0, y0) = (b.x0.max(0.0) as usize, b.y0.max(0.0) as usize);
        let (x1, y1) = ((b.x1 as usize).min(SIZE - 1), (b.y1 as usize).min(SIZE - 1));
        for x in x0..=x1 {
            img[(x, y0)] = colour;
            img[(x, y1)] = colour;
        }
        for y in y0..=y1 {
            img[(x0, y)] = colour;
            img[(x1, y)] = colour;
        }
    }
}

pub fn main() {
    let dir = "out/yolo";
    let (img, labels) = samples::nuclei(SIZE, 40, 11);
    let truth = true_boxes(&labels);
    println!("{} true nuclei", truth.len());

    let mut rng = Rng::new(4);
    let cands = network(&truth, &mut rng, 0.15);
    println!("{} candidates from strides 8 and 16 ({} above the score threshold 0.25)", cands.len(), cands.iter().filter(|c| c.score >= 0.25).count());
    let mut pic = gray_to_rgb(&img);
    draw(&mut pic, &cands.iter().filter(|c| c.score >= 0.25).copied().collect::<Vec<_>>());
    save_rgb(&pic, format!("{dir}/candidates.png"));

    for t in [0.3, 0.5, 0.7, 0.9] {
        for aware in [true, false] {
            let kept: Vec<Detection> = nms_classes(&cands, 0.25, t, aware).iter().map(|&i| cands[i]).collect();
            let (tp, fp, fn_) = score(&kept, &truth);
            let debris = kept.iter().filter(|d| d.class == 1).count();
            let kind = if aware { "class-aware   " } else { "class-agnostic" };
            println!("NMS {t}, {kind}: {:3} boxes ({debris} debris)   TP {tp:2}  FP {fp:2}  FN {fn_:2}", kept.len());
            if t == 0.5 {
                let mut pic = gray_to_rgb(&img);
                draw(&mut pic, &kept);
                save_rgb(&pic, format!("{dir}/kept-{}.png", if aware { "aware" } else { "agnostic" }));
            }
        }
    }

    // Noisier boxes: a quarter of a stride of noise on every side becomes a whole stride.
    let noisy = network(&truth, &mut rng, 1.0);
    let kept: Vec<Detection> = nms_classes(&noisy, 0.25, 0.5, false).iter().map(|&i| noisy[i]).collect();
    let (tp, fp, fn_) = score(&kept, &truth);
    println!("box noise 1 stride, NMS 0.5, class-agnostic: {:3} boxes   TP {tp:2}  FP {fp:2}  FN {fn_:2}", kept.len());

    // A centre heat map: a bump (sd 3 px) at every true centre, plus noise.
    let centres: Vec<(f32, f32)> = truth.iter().map(|b| ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)).collect();
    let mut rng = Rng::new(9);
    let heat = Image::from_fn(SIZE, SIZE, |x, y| {
        let bump = centres.iter().map(|&(cx, cy)| (-((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / 18.0).exp()).fold(0.0, f32::max);
        bump + 0.12 * rng.normal()
    });
    save_gray(&heat, format!("{dir}/heatmap.png"));
    for radius in [1, 3, 6] {
        let peaks = heatmap_peaks(&heat, radius, 0.5);
        // a peak is a hit if it lies within 4 px of a true centre that has no hit yet
        let mut used = vec![false; centres.len()];
        let mut hits = 0;
        for &(px, py) in &peaks {
            let near = (0..centres.len()).find(|&j| !used[j] && (centres[j].0 - px as f32).hypot(centres[j].1 - py as f32) <= 4.0);
            if let Some(j) = near {
                used[j] = true;
                hits += 1;
            }
        }
        let w = 2 * radius + 1;
        println!("heat map, {w} × {w} window, threshold 0.5: {:3} peaks   TP {hits:2}  FP {:2}  FN {:2}", peaks.len(), peaks.len() - hits, centres.len() - hits);
    }
    println!("wrote the results to {dir}/");
}
