// Simulated dividing cells in a 400 × 40 px strip, with missed detections. Stage 1 links
// frame to frame; stage 2 closes gaps (C1) and finds divisions (C2). Scores (C3, C4) and
// cycle times (C5) are printed; the space–time picture is out/track-events/space-time.png.
//   cargo run -p tracking -- track-events                      (your code)
//   cargo run -p tracking --features solution -- track-events  (the model solution)

use imgcourse::*;
use crate::track_events::*;

const FRAMES: usize = 60;
const WIDTH: f32 = 400.0;
const HEIGHT: f32 = 40.0;
const MISS: f32 = 0.08; // probability that a detection is missed

/// The true movie: detections, the true cell of each, and the lineage.
struct Movie {
    dets: Vec<Detection>,
    cell_of: Vec<usize>,
    cells: Vec<Cell>,
    daughters: Vec<Option<[usize; 2]>>,
}

/// Cells move by random walk (1 px per axis and frame) and divide after 8 frames plus a
/// random time (exponential, mean `extra`); the daughters appear 3 px on either side.
fn simulate(founders: usize, extra: f32, seed: u64) -> Movie {
    let mut rng = Rng::new(seed);
    let cycle = |rng: &mut Rng| 8.0 + extra * -(1.0 - rng.uniform()).ln();
    // state of the living cells: (cell number, x, y, age, cycle length)
    let mut alive: Vec<(usize, f32, f32, f32, f32)> = Vec::new();
    let mut m = Movie { dets: vec![], cell_of: vec![], cells: vec![], daughters: vec![] };
    for _ in 0..founders {
        let c = cycle(&mut rng);
        alive.push((m.cells.len(), WIDTH * rng.uniform(), 2.0 + (HEIGHT - 4.0) * rng.uniform(), c * rng.uniform(), c));
        m.cells.push(Cell { born: None, divided: None });
        m.daughters.push(None);
    }
    for t in 0..FRAMES {
        let mut next = Vec::new();
        for &(id, x, y, age, c) in &alive {
            if t > 0 && age >= c {
                // division: two daughters on opposite sides
                let angle = std::f32::consts::PI * rng.uniform();
                let (dx, dy) = (3.0 * angle.cos(), 3.0 * angle.sin());
                m.cells[id].divided = Some(t as f32);
                let first = m.cells.len();
                m.daughters[id] = Some([first, first + 1]);
                for s in [1.0, -1.0] {
                    next.push((m.cells.len(), x + s * dx, (y + s * dy).clamp(1.0, HEIGHT - 1.0), 0.0, cycle(&mut rng)));
                    m.cells.push(Cell { born: Some(t as f32), divided: None });
                    m.daughters.push(None);
                }
            } else {
                let x = (x + rng.normal()).clamp(0.0, WIDTH);
                let y = (y + rng.normal()).clamp(1.0, HEIGHT - 1.0);
                next.push((id, x, y, age + 1.0, c));
            }
        }
        alive = next;
        for &(id, x, y, age, c) in &alive {
            if rng.uniform() >= MISS {
                let area = 50.0 * (1.0 + age / c) * (1.0 + 0.05 * rng.normal());
                m.dets.push(Detection { t, x, y, area });
                m.cell_of.push(id);
            }
        }
    }
    m
}

/// Draw a line between two detections in the space–time picture (x across, twice enlarged;
/// 8 px per frame down).
fn line(img: &mut Image<[f32; 3]>, a: &Detection, b: &Detection, colour: [f32; 3], dashed: bool) {
    let (x0, y0, x1, y1) = (2.0 * a.x, a.t as f32 * 8.0 + 4.0, 2.0 * b.x, b.t as f32 * 8.0 + 4.0);
    let n = ((x1 - x0).abs().max(y1 - y0) * 2.0) as usize + 1;
    for k in 0..=n {
        if dashed && (k / 4) % 2 == 1 {
            continue;
        }
        let s = k as f32 / n as f32;
        let (x, y) = ((x0 + s * (x1 - x0)) as isize, (y0 + s * (y1 - y0)) as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

pub fn main() {
    let dir = "out/track-events";
    let movie = simulate(6, 12.0, 4);
    let dets = &movie.dets;
    let n = dets.len();
    let in_frame = |t: usize| (0..n).filter(move |&i| dets[i].t == t);

    // The reference: consecutive detections of each cell, and mother → daughter links.
    let mut first_of = vec![None; movie.cells.len()];
    let mut last_of = vec![None; movie.cells.len()];
    let mut reference = Vec::new();
    for i in 0..n {
        let c = movie.cell_of[i];
        if let Some(prev) = last_of[c] {
            reference.push((prev, i));
        }
        first_of[c].get_or_insert(i);
        last_of[c] = Some(i);
    }
    let mut true_divisions = Vec::new();
    for (c, d) in movie.daughters.iter().enumerate() {
        if let (Some([d1, d2]), Some(m)) = (d, last_of[c]) {
            if let (Some(f1), Some(f2)) = (first_of[*d1], first_of[*d2]) {
                reference.push((m, f1));
                reference.push((m, f2));
                true_divisions.push([m, f1, f2]);
            }
        }
    }

    // Stage 1: frame to frame, global assignment with a strict gate (4 px; b = 8, so a link
    // must have d² < 16).
    let mut next: Vec<Option<usize>> = vec![None; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    for t in 0..FRAMES - 1 {
        let (a, b): (Vec<usize>, Vec<usize>) = (in_frame(t).collect(), in_frame(t + 1).collect());
        let cost: Vec<Vec<f32>> = a
            .iter()
            .map(|&i| b.iter().map(|&j| {
                let d2 = (dets[i].x - dets[j].x).powi(2) + (dets[i].y - dets[j].y).powi(2);
                if d2 <= 16.0 { d2 } else { f32::INFINITY }
            }).collect())
            .collect();
        for (k, link) in join_with_alternatives(&cost, 8.0).iter().enumerate() {
            if let Some(l) = link {
                next[a[k]] = Some(b[*l]);
                prev[b[*l]] = Some(a[k]);
            }
        }
    }
    let mut links: Vec<(usize, usize)> = (0..n).filter_map(|i| next[i].map(|j| (i, j))).collect();
    let stage1 = link_scores(&reference, &links);

    // Stage 2a: gap closing between segment ends and starts (C1), up to 2 missed frames.
    let ends: Vec<usize> = (0..n).filter(|&i| next[i].is_none() && dets[i].t < FRAMES - 1).collect();
    let starts: Vec<usize> = (0..n).filter(|&i| prev[i].is_none() && dets[i].t > 0).collect();
    let cost: Vec<Vec<f32>> = ends
        .iter()
        .map(|&e| starts.iter().map(|&s| gap_cost(&dets[e], &dets[s], 2, 4.0).unwrap_or(f32::INFINITY)).collect())
        .collect();
    let mut gaps = Vec::new();
    for (k, join) in join_with_alternatives(&cost, 8.0).iter().enumerate() {
        if let Some(j) = join {
            gaps.push((ends[k], starts[*j]));
            prev[starts[*j]] = Some(ends[k]);
        }
    }
    links.extend(&gaps);
    let stage2a = link_scores(&reference, &links);

    // Stage 2b: splits (C2). A start that is still unexplained may be the second daughter of
    // a detection one frame earlier that continues into its sister. Take the cheapest
    // candidate if its cost is below 50; each mother divides at most once.
    let mut splits = Vec::new();
    let mut divided = vec![false; n];
    for &s in &starts {
        if prev[s].is_some() {
            continue;
        }
        let mut best: Option<(f32, usize, usize)> = None;
        for m in in_frame(dets[s].t - 1) {
            if let Some(sister) = next[m] {
                let c = split_cost(&dets[m], &dets[s], &dets[sister]);
                if !divided[m] && c < 50.0 && best.is_none_or(|b| c < b.0) {
                    best = Some((c, m, sister));
                }
            }
        }
        if let Some((_, m, sister)) = best {
            divided[m] = true;
            splits.push([m, sister, s]);
            links.push((m, s));
        }
    }
    let stage2b = link_scores(&reference, &links);
    let (tp, fp, fn_) = division_scores(&true_divisions, &splits);

    println!("{} detections in {FRAMES} frames, {} true links, {} true divisions", n, reference.len(), true_divisions.len());
    println!("                         precision  recall    F1");
    for (name, s) in [("stage 1 (frame to frame)", stage1), ("+ gap closing", stage2a), ("+ splits", stage2b)] {
        println!("{name:25} {:6.3}   {:6.3}  {:6.3}", s.0, s.1, s.2);
    }
    println!("{} gaps closed; {} divisions found: TP {tp}, FP {fp}, FN {fn_}", gaps.len(), splits.len());

    // The space–time picture: frame-to-frame links grey, gap joins dashed blue, splits
    // orange, links that are not in the reference red.
    let mut img = Image::new(2 * WIDTH as usize, FRAMES * 8, [1.0f32; 3]);
    let wrong = |l: &(usize, usize)| !reference.contains(l);
    for &(i, j) in &links {
        let (gap, split) = (gaps.contains(&(i, j)), splits.iter().any(|d| d[0] == i && d[2] == j));
        let colour = if wrong(&(i, j)) {
            [0.9, 0.0, 0.0]
        } else if gap {
            [0.1, 0.3, 0.9]
        } else if split {
            [1.0, 0.5, 0.0]
        } else {
            [0.55, 0.55, 0.55]
        };
        line(&mut img, &dets[i], &dets[j], colour, gap);
    }
    save_rgb(&img, format!("{dir}/space-time.png"));
    println!("wrote {dir}/space-time.png (x across, time down; grey links, dashed blue gaps, orange splits, red wrong)");

    // Cycle times and censoring (C5): the true lineages of many cells, cycle mean 26 frames.
    let big = simulate(200, 18.0, 9);
    let (complete, censored) = cycle_times(&big.cells, FRAMES as f32);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let all: Vec<f32> = complete.iter().chain(&censored).copied().collect();
    println!("\ncensoring, 200 founders, {FRAMES}-frame movie, true mean cycle 26 frames:");
    println!("  {} complete cycles, mean {:.1}; {} censored", complete.len(), mean(&complete), censored.len());
    println!("  all cells born in the movie, observed times as if complete: mean {:.1}", mean(&all));
}
