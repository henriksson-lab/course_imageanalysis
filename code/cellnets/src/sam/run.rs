// A toy promptable segmenter on synthetic nuclei, with the architecture of the lecture:
// an "encoder" run once (blur and shrink the image to a coarse grid of positions), and a
// "decoder" per click (a level halfway between the darkest and brightest nearby positions,
// coarse logits, your upsampling, your click mask, your stability). Then automatic masks
// from a grid of clicks with your filters, for grids of 8 × 8 to 32 × 32 points and two
// embedding sizes. Results in out/sam/.
//   cargo run -p cellnets -- sam                      (your code)
//   cargo run -p cellnets --features solution -- sam  (the model solution)

use imgcourse::*;
use crate::sam::*;

const SIZE: usize = 256;
/// The decoder's logit is ±1 at ±SLOPE times the local contrast from its level.
const SLOPE: f32 = 0.015;

/// The "encoder": the mean of every `block × block` piece of the image.
fn encode(img: &Image<f32>, block: usize) -> Image<f32> {
    let n = SIZE / block;
    Image::from_fn(n, n, |u, v| img.crop(u * block, v * block, block, block).mean())
}

/// The "decoder" for one click: coarse logits around a level halfway between the darkest
/// and the brightest position within `reach` positions of the click; the logit is ±1 at
/// ±SLOPE of that contrast. `None` if there is no contrast near the click.
fn decode(emb: &Image<f32>, x: usize, y: usize) -> Option<Candidate> {
    let block = SIZE / emb.width();
    let (u, v) = ((x / block) as isize, (y / block) as isize);
    let reach = (16 / block).max(1) as isize; // about 16 pixels
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for dv in -reach..=reach {
        for du in -reach..=reach {
            if emb.contains(u + du, v + dv) {
                let e = emb[((u + du) as usize, (v + dv) as usize)];
                lo = lo.min(e);
                hi = hi.max(e);
            }
        }
    }
    if hi - lo < 0.2 {
        return None;
    }
    let level = 0.5 * (lo + hi);
    let low = emb.map(|e| (e - level) / (SLOPE * (hi - lo)));
    let logits = upsample_logits(&low, SIZE, SIZE);
    let mask = mask_from_click(&logits, x, y);
    let pixels: Vec<(usize, usize)> = mask.coords().filter(|&p| mask[p]).collect();
    if pixels.is_empty() {
        return None;
    }
    // stability and score are measured on this mask's own logits: those of the region above
    // −1 around the click (other objects are masked out)
    let outer = mask_from_click(&logits.map(|l| l + 1.0), x, y);
    let own = logits.map_xy(|px, py, l| if outer[(px, py)] { l } else { -10.0 });
    let score = pixels.iter().map(|&p| 1.0 / (1.0 + (-logits[p]).exp())).sum::<f32>() / pixels.len() as f32;
    let bbox = pixels.iter().fold((usize::MAX, usize::MAX, 0, 0), |b, &(px, py)| (b.0.min(px), b.1.min(py), b.2.max(px), b.3.max(py)));
    Some(Candidate { pixels, bbox, score, stability: stability(&own, 1.0) })
}

/// TP, FP, FN of masks against the true labels at IoU ≥ 0.5.
fn score(masks: &[&Candidate], truth: &Image<u32>) -> (usize, usize, usize) {
    let nt = truth.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut area = vec![0usize; nt + 1];
    for &l in truth.pixels() {
        area[l as usize] += 1;
    }
    let present = (1..=nt).filter(|&l| area[l] > 0).count();
    let mut matched = vec![false; nt + 1];
    let mut tp = 0;
    for m in masks {
        let mut inter = vec![0usize; nt + 1];
        for &p in &m.pixels {
            inter[truth[p] as usize] += 1;
        }
        let best = (1..=nt).filter(|&l| !matched[l]).max_by_key(|&l| inter[l]);
        if let Some(l) = best {
            if inter[l] as f32 / (area[l] + m.pixels.len() - inter[l]) as f32 >= 0.5 {
                matched[l] = true;
                tp += 1;
            }
        }
    }
    (tp, masks.len() - tp, present - tp)
}

pub fn main() {
    let dir = "out/sam";
    let (img, truth) = samples::nuclei(SIZE, 16, 21);
    save_gray(&img, format!("{dir}/input.png"));
    save_labels(&truth, format!("{dir}/truth.png"));
    let covered = truth.pixels().iter().filter(|&&l| l > 0).count() as f32 / (SIZE * SIZE) as f32;
    let n_true = (1..=16).filter(|l| truth.pixels().contains(l)).count();
    println!("{n_true} nuclei covering {:.0} % of the image", 100.0 * covered);

    for block in [4, 16] {
        let t0 = std::time::Instant::now();
        let emb = encode(&img, block);
        let t_enc = t0.elapsed().as_secs_f32();
        save_gray(&upsample_logits(&emb, SIZE, SIZE), format!("{dir}/embedding-{}.png", SIZE / block));
        for grid in [8, 16, 32] {
            let t1 = std::time::Instant::now();
            let step = SIZE / grid;
            let points: Vec<(usize, usize)> = (0..grid * grid).map(|k| (step / 2 + step * (k % grid), step / 2 + step * (k / grid))).collect();
            let on_nucleus = points.iter().filter(|&&p| truth[p] > 0).count();
            let cands: Vec<Candidate> = points.iter().filter_map(|&(x, y)| decode(&emb, x, y)).collect();
            let kept_idx = filter_masks(&cands, 0.8, 0.95, 0.7);
            let kept: Vec<&Candidate> = kept_idx.iter().map(|&i| &cands[i]).collect();
            let (tp, fp, fn_) = score(&kept, &truth);
            let t_dec = t1.elapsed().as_secs_f32();
            println!(
                "embedding {e} × {e}, grid {grid:2} × {grid:2}: {on_nucleus:3} points on a nucleus, {:3} masks, {:3} kept   TP {tp:2}  FP {fp:2}  FN {fn_:2}   (encoder {:.3} s once, decoder {:.2} s)",
                cands.len(),
                kept.len(),
                t_enc,
                t_dec,
                e = SIZE / block
            );
            if grid == 16 {
                let mut labels = Image::new(SIZE, SIZE, 0u32);
                for (k, m) in kept.iter().enumerate().rev() {
                    for &p in &m.pixels {
                        labels[p] = k as u32 + 1;
                    }
                }
                save_labels(&labels, format!("{dir}/auto-{}-grid16.png", SIZE / block));
            }
        }
    }
    println!("wrote the results to {dir}/");
}
