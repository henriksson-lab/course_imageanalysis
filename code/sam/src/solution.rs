// Model solution for the coding exercises of "Promptable segmentation: SAM".

use imgcourse::{Border, Image};

/// A candidate mask of automatic segmentation: its pixels, its box (`x0, y0, x1, y1`,
/// inclusive), its score (the network's own quality estimate) and its stability.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub pixels: Vec<(usize, usize)>,
    pub bbox: (usize, usize, usize, usize),
    pub score: f32,
    pub stability: f32,
}

/// The overlap of two boxes, as IoU of the boxes (counted in pixels, boxes inclusive).
pub fn box_iou(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> f32 {
    let area = |r: (usize, usize, usize, usize)| ((r.2 - r.0 + 1) * (r.3 - r.1 + 1)) as f32;
    let (x0, y0, x1, y1) = (a.0.max(b.0), a.1.max(b.1), a.2.min(b.2), a.3.min(b.3));
    if x1 < x0 || y1 < y0 {
        return 0.0;
    }
    let inter = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
    inter / (area(a) + area(b) - inter)
}

/// C1. Enlarge a coarse logit map to `width × height` by bilinear interpolation. Output
/// pixel `(x, y)` sits at `u = (x + ½) · lw / width − ½`, `v = (y + ½) · lh / height − ½` in
/// the coarse map (`lw × lh`): the four coarse pixels around `(u, v)` are mixed by their
/// distances. Positions outside the coarse map use its edge (clamp).
pub fn upsample_logits(low: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    let (lw, lh) = (low.width() as f32, low.height() as f32);
    Image::from_fn(width, height, |x, y| {
        let u = (x as f32 + 0.5) * lw / width as f32 - 0.5;
        let v = (y as f32 + 0.5) * lh / height as f32 - 0.5;
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let at = |du: isize, dv: isize| low.get_border(u0 as isize + du, v0 as isize + dv, Border::Clamp);
        let top = (1.0 - fu) * at(0, 0) + fu * at(1, 0);
        let bottom = (1.0 - fu) * at(0, 1) + fu * at(1, 1);
        (1.0 - fv) * top + fv * bottom
    })
}

/// C2. The stability of a mask: the number of pixels with logit above `delta`, divided by
/// the number with logit above `−delta` (0 if there are none).
pub fn stability(logits: &Image<f32>, delta: f32) -> f32 {
    let inner = logits.pixels().iter().filter(|&&l| l > delta).count();
    let outer = logits.pixels().iter().filter(|&&l| l > -delta).count();
    if outer == 0 { 0.0 } else { inner as f32 / outer as f32 }
}

/// C3. The mask of a click: the pixels with logit above 0 that are connected to the
/// clicked pixel `(x, y)` (4-neighbours). Empty if the clicked pixel itself is not above 0.
pub fn mask_from_click(logits: &Image<f32>, x: usize, y: usize) -> Image<bool> {
    let mut mask = Image::new(logits.width(), logits.height(), false);
    if logits[(x, y)] <= 0.0 {
        return mask;
    }
    let mut stack = vec![(x, y)];
    mask[(x, y)] = true;
    while let Some((u, v)) = stack.pop() {
        for (du, dv) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (a, b) = (u as isize + du, v as isize + dv);
            if logits.contains(a, b) && !mask[(a as usize, b as usize)] && logits[(a as usize, b as usize)] > 0.0 {
                mask[(a as usize, b as usize)] = true;
                stack.push((a as usize, b as usize));
            }
        }
    }
    mask
}

/// C4. The filters of automatic masks: keep the candidates with `score > score_min` and
/// `stability >= stability_min`; then remove duplicates greedily: from the highest score
/// down, keep a candidate unless its box overlaps (box IoU) a kept candidate's box by more
/// than `box_overlap`. Returns the indices of the kept candidates, best first.
pub fn filter_masks(cands: &[Candidate], score_min: f32, stability_min: f32, box_overlap: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].score > score_min && cands[i].stability >= stability_min).collect();
    order.sort_by(|&i, &j| cands[j].score.total_cmp(&cands[i].score));
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        if kept.iter().all(|&k| box_iou(cands[k].bbox, cands[i].bbox) <= box_overlap) {
            kept.push(i);
        }
    }
    kept
}
