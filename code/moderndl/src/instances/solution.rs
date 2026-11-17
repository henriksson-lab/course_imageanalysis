// Model solution for the coding exercises of "From masks to objects".

use imgcourse::Image;
use std::collections::HashMap;

/// The three classes of the boundary route.
pub const BACKGROUND: u8 = 0;
pub const INTERIOR: u8 = 1;
pub const BOUNDARY: u8 = 2;

/// The 4 neighbours of `(x, y)` that lie inside a `w × h` image.
pub fn neighbours4(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if x > 0 {
        out.push((x - 1, y));
    }
    if x + 1 < w {
        out.push((x + 1, y));
    }
    if y > 0 {
        out.push((x, y - 1));
    }
    if y + 1 < h {
        out.push((x, y + 1));
    }
    out
}

/// Connected components of a mask with 4-connectivity (the labelling day): every connected
/// piece of `true` pixels gets its own number 1, 2, ...; background 0.
pub fn label_components(mask: &Image<bool>) -> Image<u32> {
    let (w, h) = (mask.width(), mask.height());
    let mut labels = Image::new(w, h, 0u32);
    let mut next = 0;
    for (x, y) in mask.coords() {
        if mask[(x, y)] && labels[(x, y)] == 0 {
            next += 1;
            labels[(x, y)] = next;
            let mut stack = vec![(x, y)];
            while let Some((px, py)) = stack.pop() {
                for (qx, qy) in neighbours4(px, py, w, h) {
                    if mask[(qx, qy)] && labels[(qx, qy)] == 0 {
                        labels[(qx, qy)] = next;
                        stack.push((qx, qy));
                    }
                }
            }
        }
    }
    labels
}

/// C1. The boundary route: label the connected INTERIOR pixels (4-connectivity), then grow
/// every object over the BOUNDARY pixels: a boundary pixel takes the label of a labelled
/// 4-neighbour, repeated until nothing changes. Background stays 0.
pub fn boundary_route(classes: &Image<u8>) -> Image<u32> {
    let mut labels = label_components(&classes.map(|c| c == INTERIOR));
    let (w, h) = (classes.width(), classes.height());
    let mut changed = true;
    while changed {
        changed = false;
        let before = labels.clone();
        for (x, y) in classes.coords() {
            if classes[(x, y)] == BOUNDARY && before[(x, y)] == 0 {
                for (qx, qy) in neighbours4(x, y, w, h) {
                    if before[(qx, qy)] != 0 {
                        labels[(x, y)] = before[(qx, qy)];
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
    labels
}

/// C2. The distance route: watershed by flooding from the top, with h-maxima markers.
/// Visit the pixels with `dist > mask_threshold` from the highest value down. A pixel with
/// no labelled 4-neighbour starts a new object (a peak). A pixel that touches one object
/// joins it. A pixel that touches two objects joins the one with the higher peak; and if the
/// lower peak is less than `h` above this pixel (its "dynamic" is below `h`), the two objects
/// are merged: that peak was not a real marker.
pub fn distance_route(dist: &Image<f32>, mask_threshold: f32, h: f32) -> Image<u32> {
    let (w, ht) = (dist.width(), dist.height());
    let mut order: Vec<(usize, usize)> = dist.coords().filter(|&p| dist[p] > mask_threshold).collect();
    order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a])); // highest first
    let mut labels = Image::new(w, ht, 0u32);
    let mut peak: Vec<f32> = vec![0.0]; // peak height of every object; object 0 unused
    let mut parent: Vec<u32> = vec![0]; // merged objects point to the object they joined
    let root = |parent: &Vec<u32>, mut l: u32| {
        while parent[l as usize] != l {
            l = parent[l as usize];
        }
        l
    };
    for (x, y) in order {
        let level = dist[(x, y)];
        // the different objects among the labelled neighbours
        let mut found: Vec<u32> = Vec::new();
        for q in neighbours4(x, y, w, ht) {
            if labels[q] != 0 {
                let r = root(&parent, labels[q]);
                if !found.contains(&r) {
                    found.push(r);
                }
            }
        }
        if found.is_empty() {
            peak.push(level);
            parent.push(peak.len() as u32 - 1);
            labels[(x, y)] = peak.len() as u32 - 1;
            continue;
        }
        // the object with the highest peak wins this pixel
        found.sort_by(|&a, &b| peak[b as usize].total_cmp(&peak[a as usize]));
        let best = found[0];
        for &other in &found[1..] {
            if peak[other as usize] - level < h {
                parent[other as usize] = best; // too shallow: merge into the higher one
            }
        }
        labels[(x, y)] = best;
    }
    // give every pixel the final object number, renumbered 1, 2, ...
    let mut new_number = HashMap::new();
    let mut out = Image::new(w, ht, 0u32);
    for p in labels.coords() {
        if labels[p] != 0 {
            let r = root(&parent, labels[p]);
            let next = new_number.len() as u32 + 1;
            out[p] = *new_number.entry(r).or_insert(next);
        }
    }
    out
}

/// C3. Match predicted objects to true objects one to one at IoU > 0.5 (above 0.5 a match
/// is unique). Returns `(tp, fp, fn, ious)`: the counts and the IoU of every matched pair.
pub fn match_objects(truth: &Image<u32>, pred: &Image<u32>) -> (usize, usize, usize, Vec<f32>) {
    let mut area_t: HashMap<u32, f32> = HashMap::new();
    let mut area_p: HashMap<u32, f32> = HashMap::new();
    let mut overlap: HashMap<(u32, u32), f32> = HashMap::new();
    for (x, y) in truth.coords() {
        let (t, p) = (truth[(x, y)], pred[(x, y)]);
        if t != 0 {
            *area_t.entry(t).or_insert(0.0) += 1.0;
        }
        if p != 0 {
            *area_p.entry(p).or_insert(0.0) += 1.0;
        }
        if t != 0 && p != 0 {
            *overlap.entry((t, p)).or_insert(0.0) += 1.0;
        }
    }
    let mut ious = Vec::new();
    for (&(t, p), &inter) in &overlap {
        let iou = inter / (area_t[&t] + area_p[&p] - inter);
        if iou > 0.5 {
            ious.push(iou);
        }
    }
    let tp = ious.len();
    (tp, area_p.len() - tp, area_t.len() - tp, ious)
}

/// C4. Panoptic quality from the matched IoUs and the unmatched counts:
/// `SQ = mean IoU of the matches`, `RQ = TP / (TP + FP/2 + FN/2)`, `PQ = SQ · RQ`.
/// Returns `(sq, rq, pq)`; SQ is 0 if nothing matched.
pub fn panoptic_quality(ious: &[f32], fp: usize, fn_: usize) -> (f32, f32, f32) {
    let tp = ious.len() as f32;
    let sq = if ious.is_empty() { 0.0 } else { ious.iter().sum::<f32>() / tp };
    let rq = tp / (tp + 0.5 * fp as f32 + 0.5 * fn_ as f32);
    (sq, rq, sq * rq)
}
