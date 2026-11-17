// Decodes a simulated StarDist output on synthetic nuclei, and on a C-shaped cell.
// The "network" is exact: objectness = distance to the background, scaled to 1 in each
// nucleus, plus a little noise; rays = your C1 on the true labels. Results in out/stardist/.
//   cargo run -p day40 -- stardist                      (your code)
//   cargo run -p day40 --features solution -- stardist  (the model solution)

use imgcourse::*;
use crate::stardist::*;

const N_RAYS: usize = 32;

/// Objectness: for every object pixel, the distance to the nearest pixel of another label
/// (or the image edge), divided by the largest such distance in its object.
fn objectness(labels: &Image<u32>) -> Image<f32> {
    let (w, h) = (labels.width(), labels.height());
    // border pixels of every object: a 4-neighbour has another label, or lies outside
    let mut border = Vec::new();
    for (x, y) in labels.coords() {
        let l = labels[(x, y)];
        let nb = [(x as isize - 1, y as isize), (x as isize + 1, y as isize), (x as isize, y as isize - 1), (x as isize, y as isize + 1)];
        if l > 0 && nb.iter().any(|&(u, v)| !labels.contains(u, v) || labels[(u as usize, v as usize)] != l) {
            border.push((x, y, l));
        }
    }
    let mut dist = Image::new(w, h, 0.0f32);
    for (x, y) in labels.coords() {
        let l = labels[(x, y)];
        if l > 0 {
            let d2 = border
                .iter()
                .filter(|b| b.2 == l)
                .map(|&(u, v, _)| (u as f32 - x as f32).powi(2) + (v as f32 - y as f32).powi(2))
                .fold(f32::MAX, f32::min);
            dist[(x, y)] = d2.sqrt() + 0.5; // a border pixel is half a pixel from the outside
        }
    }
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut top = vec![0.0f32; n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        top[l] = top[l].max(dist[(x, y)]);
    }
    dist.map_xy(|x, y, d| if labels[(x, y)] > 0 { d / top[labels[(x, y)] as usize] } else { 0.0 })
}

/// A candidate: its centre pixel, its score and its polygon.
struct Candidate {
    score: f32,
    poly: Vec<(f32, f32)>,
    bbox: (f32, f32, f32, f32),
}

/// The candidates: one per 2 × 2 grid cell (its top-left pixel), wherever the objectness is above 0.
fn candidates(labels: &Image<u32>, prob: &Image<f32>) -> Vec<Candidate> {
    let mut out = Vec::new();
    for y in (0..labels.height()).step_by(2) {
        for x in (0..labels.width()).step_by(2) {
            if prob[(x, y)] > 0.0 && labels[(x, y)] > 0 {
                let poly = polygon_vertices(x as f32, y as f32, &rays(labels, x as f32, y as f32, N_RAYS));
                let bbox = poly.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |b, &(px, py)| (b.0.min(px), b.1.min(py), b.2.max(px), b.3.max(py)));
                out.push(Candidate { score: prob[(x, y)], poly, bbox });
            }
        }
    }
    out
}

/// NMS with the bounding-box pre-check of the lecture, then the kept polygons painted into
/// a label image, best last (so that the better polygon wins where two overlap).
fn decode(cands: &[Candidate], prob_threshold: f32, nms_threshold: f32, w: usize, h: usize) -> Image<u32> {
    let scores: Vec<f32> = cands.iter().map(|c| c.score).collect();
    let iou = |i: usize, j: usize| {
        let (a, b) = (cands[i].bbox, cands[j].bbox);
        if a.2 < b.0 || b.2 < a.0 || a.3 < b.1 || b.3 < a.1 { 0.0 } else { polygon_iou(&cands[i].poly, &cands[j].poly) }
    };
    let kept = nms(&scores, prob_threshold, nms_threshold, iou);
    let mut out = Image::new(w, h, 0u32);
    for (rank, &i) in kept.iter().enumerate().rev() {
        let b = cands[i].bbox;
        for y in b.1.floor().max(0.0) as usize..=(b.3.ceil() as usize).min(h - 1) {
            for x in b.0.floor().max(0.0) as usize..=(b.2.ceil() as usize).min(w - 1) {
                if inside_polygon(x as f32, y as f32, &cands[i].poly) {
                    out[(x, y)] = rank as u32 + 1;
                }
            }
        }
    }
    out
}

/// True objects found: those with a predicted object of IoU ≥ 0.5 (counted pixel by pixel).
fn count_matches(truth: &Image<u32>, pred: &Image<u32>) -> (usize, usize, usize) {
    let nt = truth.pixels().iter().copied().max().unwrap_or(0) as usize;
    let np = pred.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut inter = vec![vec![0usize; np + 1]; nt + 1];
    let (mut at, mut ap) = (vec![0usize; nt + 1], vec![0usize; np + 1]);
    for (&t, &p) in truth.pixels().iter().zip(pred.pixels()) {
        inter[t as usize][p as usize] += 1;
        at[t as usize] += 1;
        ap[p as usize] += 1;
    }
    let present_t = (1..=nt).filter(|&t| at[t] > 0).count();
    let present_p = (1..=np).filter(|&p| ap[p] > 0).count();
    let mut tp = 0;
    for t in 1..=nt {
        for p in 1..=np {
            let i = inter[t][p];
            if i > 0 && i as f32 / (at[t] + ap[p] - i) as f32 >= 0.5 {
                tp += 1;
            }
        }
    }
    (tp, present_p - tp, present_t - tp)
}

pub fn main() {
    let dir = "out/stardist";
    let (img, truth) = samples::nuclei(256, 45, 7);
    save_gray(&img, format!("{dir}/input.png"));
    save_labels(&truth, format!("{dir}/truth.png"));
    let n_true = (1..=45).filter(|&l| truth.pixels().contains(&l)).count();

    // The simulated network output: objectness with a little noise, and the rays.
    let mut rng = Rng::new(3);
    let exact = objectness(&truth);
    let prob = Image::from_fn(256, 256, |x, y| {
        let p = exact[(x, y)];
        if p > 0.0 { (p + 0.05 * rng.normal()).clamp(0.01, 1.0) } else { 0.0 }
    });
    save_gray(&prob, format!("{dir}/objectness.png"));
    // the length of the ray to the right (ray 0) at every pixel
    let ray0 = truth.map_xy(|x, y, l| if l > 0 { rays(&truth, x as f32, y as f32, 1)[0] } else { 0.0 });
    save_gray_stretched(&ray0, format!("{dir}/ray-right.png"));

    let t0 = std::time::Instant::now();
    let cands = candidates(&truth, &prob);
    println!("{n_true} true nuclei; {} grid cells inside a nucleus ({:.1} s for their rays)", cands.len(), t0.elapsed().as_secs_f32());
    for (prob_t, nms_t, name) in [(0.5, 0.3, "default"), (0.15, 0.3, "low-prob"), (0.85, 0.3, "high-prob"), (0.5, 0.02, "strong-nms"), (0.5, 0.9, "weak-nms")] {
        let n_cand = cands.iter().filter(|c| c.score >= prob_t).count();
        let labels = decode(&cands, prob_t, nms_t, 256, 256);
        let (tp, fp, fn_) = count_matches(&truth, &labels);
        let n_obj = labels.pixels().iter().copied().max().unwrap_or(0);
        println!("probability {prob_t:4}, NMS {nms_t:4}: {n_cand:4} candidates -> {n_obj:3} objects   TP {tp:2}  FP {fp:2}  FN {fn_:2}");
        save_labels(&labels, format!("{dir}/labels-{name}.png"));
    }

    // A C-shaped cell: no polygon from one point can describe it.
    let c = Image::from_fn(64, 64, |x, y| {
        let (dx, dy) = (x as f32 - 32.0, y as f32 - 32.0);
        let r = (dx * dx + dy * dy).sqrt();
        let opening = dx > 0.0 && dy.abs() < 0.7 * dx; // a wedge cut out on the right
        if (12.0..24.0).contains(&r) && !opening { 1 } else { 0 }
    });
    let cprob = objectness(&c);
    let ccands = candidates(&c, &cprob);
    let clabels = decode(&ccands, 0.5, 0.3, 64, 64);
    let n_obj = clabels.pixels().iter().copied().max().unwrap_or(0);
    let (tp, fp, fn_) = count_matches(&c, &clabels);
    println!("C-shape: {} candidates -> {n_obj} objects   TP {tp}  FP {fp}  FN {fn_}", ccands.iter().filter(|c| c.score >= 0.5).count());
    save_labels(&c, format!("{dir}/cshape-truth.png"));
    save_labels(&clabels, format!("{dir}/cshape-labels.png"));
    println!("wrote the results to {dir}/ in {:.1} s", t0.elapsed().as_secs_f32());
}
