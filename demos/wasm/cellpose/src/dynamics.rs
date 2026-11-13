//! From network outputs to cell masks, as in Cellpose 3 (cellpose 3.1 dynamics.py,
//! metrics.py and utils.py, CPU path), written for flat row-major arrays:
//!
//! 1. foreground = cell probability > threshold;
//! 2. every foreground pixel follows the flow field (dP * fg / 5, bilinear interpolation) for
//!    `niter` Euler steps and ends near the "sink" of its cell;
//! 3. a histogram of the end points; seeds = local maxima (5x5) with more than 10 pixels,
//!    grown for 5 steps inside histogram bins with more than 2 pixels; each pixel gets the
//!    label of the seed region its end point fell into; masks larger than
//!    `max_size_fraction` of the image are dropped;
//! 4. flow-error check: flows are recomputed from each mask by a diffusion from its centre and
//!    compared with the network's flows; masks with mean squared error > flow_threshold go;
//! 5. small masks are removed and holes filled.
//!
//! The steps follow the Python reference closely (including its label order), so the result
//! can be compared label by label with Cellpose and with the cellpose-rs crate.

/// Bilinear sample of the interleaved flow `f` = [dy0, dx0, dy1, dx1, ...] at (y, x), as in
/// cellpose's `map_coordinates` (weights in f64, indices clamped to the image). Positions are
/// always inside [0, h-1] x [0, w-1] here.
#[inline(always)]
fn sample(f: &[f32], h: usize, w: usize, y: f32, x: f32) -> (f32, f32) {
    let yf = (y as usize).min(h - 1); // truncation (y >= 0)
    let xf = (x as usize).min(w - 1);
    let fy = y as f64 - yf as f64;
    let fx = x as f64 - xf as f64;
    let yf1 = (yf + 1).min(h - 1);
    let xf1 = (xf + 1).min(w - 1);
    let w00 = (1.0 - fy) * (1.0 - fx);
    let w01 = (1.0 - fy) * fx;
    let w10 = fy * (1.0 - fx);
    let w11 = fy * fx;
    let (i00, i01, i10, i11) = (2 * (yf * w + xf), 2 * (yf * w + xf1), 2 * (yf1 * w + xf), 2 * (yf1 * w + xf1));
    debug_assert!(i11 + 1 < f.len());
    // SAFETY: all indices are < 2*h*w = f.len() (yf, yf1 < h and xf, xf1 < w).
    unsafe {
        let g = |o: usize| -> f32 {
            (*f.get_unchecked(i00 + o) as f64 * w00 + *f.get_unchecked(i01 + o) as f64 * w01
                + *f.get_unchecked(i10 + o) as f64 * w10 + *f.get_unchecked(i11 + o) as f64 * w11) as f32
        };
        (g(0), g(1))
    }
}

fn interleave(my: &[f32], mx: &[f32]) -> Vec<f32> {
    let mut f = vec![0f32; 2 * my.len()];
    for i in 0..my.len() {
        f[2 * i] = my[i];
        f[2 * i + 1] = mx[i];
    }
    f
}

/// The flow used for tracking: dP * (cellprob > thr) / 5, and the foreground pixel indices.
fn masked_flows(dy: &[f32], dx: &[f32], cellprob: &[f32], thr: f32) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let n = cellprob.len();
    let mut my = vec![0f32; n];
    let mut mx = vec![0f32; n];
    let mut inds = Vec::new();
    for i in 0..n {
        if cellprob[i] > thr {
            my[i] = dy[i] / 5.0;
            mx[i] = dx[i] / 5.0;
            inds.push(i as u32);
        }
    }
    (my, mx, inds)
}

/// Euler integration of the masked flow for `niter` steps from each index in `inds`.
/// Returns end points (y, x) per start pixel.
pub fn follow_flows(my: &[f32], mx: &[f32], h: usize, w: usize, inds: &[u32], niter: usize) -> (Vec<f32>, Vec<f32>) {
    let f = interleave(my, mx);
    let mut py: Vec<f32> = inds.iter().map(|&i| (i as usize / w) as f32).collect();
    let mut px: Vec<f32> = inds.iter().map(|&i| (i as usize % w) as f32).collect();
    let (ymax, xmax) = ((h - 1) as f32, (w - 1) as f32);
    // Blocks of points advanced together: each point's steps form a long dependency chain,
    // and interleaving independent points hides its latency (same arithmetic per point).
    const B: usize = 32;
    let n = inds.len();
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + B).min(n);
        let (ys, xs) = (&mut py[k0..k1], &mut px[k0..k1]);
        for _ in 0..niter {
            for j in 0..ys.len() {
                let (vy, vx) = sample(&f, h, w, ys[j], xs[j]);
                ys[j] = (ys[j] + vy).max(0.0).min(ymax);
                xs[j] = (xs[j] + vx).max(0.0).min(xmax);
            }
        }
        k0 = k1;
    }
    (py, px)
}

/// Paths of a few start pixels: positions after every `stride` steps (niter/stride + 1 points
/// per start, y and x interleaved).
pub fn trace_paths(my: &[f32], mx: &[f32], h: usize, w: usize, starts: &[u32], niter: usize, stride: usize) -> Vec<f32> {
    let stride = stride.max(1);
    let f = interleave(my, mx);
    let (ymax, xmax) = ((h - 1) as f32, (w - 1) as f32);
    let mut out = Vec::with_capacity(starts.len() * (niter / stride + 1) * 2);
    for &s in starts {
        let (mut y, mut x) = ((s as usize / w) as f32, (s as usize % w) as f32);
        out.push(y);
        out.push(x);
        for t in 1..=niter {
            let (vy, vx) = sample(&f, h, w, y, x);
            y = (y + vy).max(0.0).min(ymax);
            x = (x + vx).max(0.0).min(xmax);
            if t % stride == 0 {
                out.push(y);
                out.push(x);
            }
        }
    }
    out
}

const RPAD: usize = 20;

/// Renumber labels 1.. in order of first appearance (fastremap.renumber), 0 stays 0.
fn renumber(m: &mut [u32]) {
    let mx = m.iter().copied().max().unwrap_or(0) as usize;
    let mut map = vec![0u32; mx + 1];
    let mut next = 0u32;
    for v in m.iter_mut() {
        if *v != 0 {
            let k = *v as usize;
            if map[k] == 0 {
                next += 1;
                map[k] = next;
            }
            *v = map[k];
        }
    }
}

/// Labels from end points (get_masks_torch). Returns (masks, histogram of end points over
/// the image, seed pixel indices in the image).
pub fn get_masks(py: &[f32], px: &[f32], inds: &[u32], h: usize, w: usize, max_size_fraction: f32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (hp, wp) = (h + 2 * RPAD, w + 2 * RPAD);
    let mut hist = vec![0i32; hp * wp];
    let pt: Vec<(usize, usize)> = (0..inds.len())
        .map(|k| {
            let y = (py[k] as i32 + RPAD as i32).clamp(0, (h + RPAD - 1) as i32) as usize;
            let x = (px[k] as i32 + RPAD as i32).clamp(0, (w + RPAD - 1) as i32) as usize;
            (y, x)
        })
        .collect();
    for &(y, x) in &pt {
        hist[y * wp + x] += 1;
    }
    // 5x5 max filter (separable; outside counts as -inf)
    let mut tmp = vec![0i32; hp * wp];
    let mut hmax = vec![0i32; hp * wp];
    for y in 0..hp {
        for x in 0..wp {
            let mut m = i32::MIN;
            for xx in x.saturating_sub(2)..(x + 3).min(wp) {
                m = m.max(hist[y * wp + xx]);
            }
            tmp[y * wp + x] = m;
        }
    }
    for y in 0..hp {
        for x in 0..wp {
            let mut m = i32::MIN;
            for yy in y.saturating_sub(2)..(y + 3).min(hp) {
                m = m.max(tmp[yy * wp + x]);
            }
            hmax[y * wp + x] = m;
        }
    }
    let mut seeds: Vec<(usize, usize)> = Vec::new();
    for y in 0..hp {
        for x in 0..wp {
            let v = hist[y * wp + x];
            if v > 10 && v >= hmax[y * wp + x] {
                seeds.push((y, x));
            }
        }
    }
    let mut masks = vec![0u32; h * w];
    let mut hist_img = vec![0u32; h * w];
    for y in 0..h {
        for x in 0..w {
            hist_img[y * w + x] = hist[(y + RPAD) * wp + x + RPAD] as u32;
        }
    }
    if seeds.is_empty() {
        return (masks, hist_img, Vec::new());
    }
    // ascending count; stable for ties (raster order)
    seeds.sort_by_key(|&(y, x)| hist[y * wp + x]);
    let mut m1 = vec![0u32; hp * wp];
    for (k, &(sy, sx)) in seeds.iter().enumerate() {
        // grow from the seed for 5 steps inside an 11x11 window where hist > 2
        let mut sm = [[false; 11]; 11];
        sm[5][5] = true;
        let at = |i: usize, j: usize| -> i32 {
            let (yy, xx) = (sy as isize + i as isize - 5, sx as isize + j as isize - 5);
            if yy < 0 || xx < 0 || yy >= hp as isize || xx >= wp as isize {
                0
            } else {
                hist[yy as usize * wp + xx as usize]
            }
        };
        for _ in 0..5 {
            let mut n = [[false; 11]; 11];
            for i in 0..11usize {
                for j in 0..11usize {
                    let mut any = false;
                    for di in i.saturating_sub(1)..(i + 2).min(11) {
                        for dj in j.saturating_sub(1)..(j + 2).min(11) {
                            any |= sm[di][dj];
                        }
                    }
                    n[i][j] = any && at(i, j) > 2;
                }
            }
            sm = n;
        }
        for i in 0..11 {
            for j in 0..11 {
                if sm[i][j] {
                    let (yy, xx) = (sy as isize + i as isize - 5, sx as isize + j as isize - 5);
                    if yy >= 0 && xx >= 0 && (yy as usize) < hp && (xx as usize) < wp {
                        m1[yy as usize * wp + xx as usize] = k as u32 + 1;
                    }
                }
            }
        }
    }
    for (k, &i) in inds.iter().enumerate() {
        let (y, x) = pt[k];
        masks[i as usize] = m1[y * wp + x];
    }
    // drop masks larger than max_size_fraction of the image (background included, as in Python)
    let big = (h * w) as f64 * max_size_fraction as f64;
    let nl = seeds.len() + 1;
    let mut counts = vec![0usize; nl];
    for &v in &masks {
        counts[v as usize] += 1;
    }
    for v in masks.iter_mut() {
        if counts[*v as usize] as f64 > big {
            *v = 0;
        }
    }
    renumber(&mut masks);
    let seed_px = seeds
        .iter()
        .filter(|&&(y, x)| y >= RPAD && x >= RPAD && y < h + RPAD && x < w + RPAD)
        .map(|&(y, x)| ((y - RPAD) * w + x - RPAD) as u32)
        .collect();
    (masks, hist_img, seed_px)
}

/// Bounding boxes (y0, y1, x0, x1, exclusive ends) of labels 1..=max (None when absent).
fn find_objects(m: &[u32], h: usize, w: usize) -> Vec<Option<(usize, usize, usize, usize)>> {
    let mx = m.iter().copied().max().unwrap_or(0) as usize;
    let mut b: Vec<Option<(usize, usize, usize, usize)>> = vec![None; mx];
    for y in 0..h {
        for x in 0..w {
            let v = m[y * w + x] as usize;
            if v == 0 {
                continue;
            }
            let e = &mut b[v - 1];
            *e = Some(match *e {
                None => (y, y + 1, x, x + 1),
                Some((y0, y1, x0, x1)) => (y0.min(y), y1.max(y + 1), x0.min(x), x1.max(x + 1)),
            });
        }
    }
    b
}

/// Flows of a label image (masks_to_flows_cpu): for each mask, a diffusion from its centre
/// pixel (the mask pixel nearest the mean position) for 2*(height+width+4) steps; the flow is
/// the normalised gradient of the result. Returns (dy, dx) unit vectors (0 outside masks).
pub fn masks_to_flows(m: &[u32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let (fy, fx) = masks_to_flows_f64(m, h, w);
    (fy.iter().map(|&v| v as f32).collect(), fx.iter().map(|&v| v as f32).collect())
}

fn masks_to_flows_f64(m: &[u32], h: usize, w: usize) -> (Vec<f64>, Vec<f64>) {
    let mut fy = vec![0f64; h * w];
    let mut fx = vec![0f64; h * w];
    for (i, bb) in find_objects(m, h, w).into_iter().enumerate() {
        let Some((y0, y1, x0, x1)) = bb else { continue };
        let lab = i as u32 + 1;
        let (ly, lx) = (y1 - y0 + 2, x1 - x0 + 2);
        let mut ys = Vec::new();
        let mut xs = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                if m[y * w + x] == lab {
                    ys.push(y - y0 + 1);
                    xs.push(x - x0 + 1);
                }
            }
        }
        let n = ys.len() as f64;
        let ymean = ys.iter().map(|&v| v as f64).sum::<f64>() / n;
        let xmean = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mut best = 0;
        let mut bd = f64::INFINITY;
        for k in 0..ys.len() {
            let d = (xs[k] as f64 - xmean).powi(2) + (ys[k] as f64 - ymean).powi(2);
            if d < bd {
                bd = d;
                best = k;
            }
        }
        let (ymed, xmed) = (ys[best], xs[best]);
        let niter = 2 * (ly + lx);
        let mut tt = vec![0f64; ly * lx];
        let mut nv = vec![0f64; ys.len()];
        for _ in 0..niter {
            tt[ymed * lx + xmed] += 1.0;
            for k in 0..ys.len() {
                let (y, x) = (ys[k], xs[k]);
                let s = tt[y * lx + x] + tt[(y - 1) * lx + x] + tt[(y + 1) * lx + x] + tt[y * lx + x - 1] + tt[y * lx + x + 1]
                    + tt[(y - 1) * lx + x - 1] + tt[(y - 1) * lx + x + 1] + tt[(y + 1) * lx + x - 1] + tt[(y + 1) * lx + x + 1];
                nv[k] = (1.0 / 9.0) * s;
            }
            for k in 0..ys.len() {
                tt[ys[k] * lx + xs[k]] = nv[k];
            }
        }
        for k in 0..ys.len() {
            let (y, x) = (ys[k], xs[k]);
            let gy = tt[(y + 1) * lx + x] - tt[(y - 1) * lx + x];
            let gx = tt[y * lx + x + 1] - tt[y * lx + x - 1];
            let o = (y0 + y - 1) * w + (x0 + x - 1);
            fy[o] = gy;
            fx[o] = gx;
        }
    }
    for i in 0..h * w {
        let n = 1e-60 + (fy[i] * fy[i] + fx[i] * fx[i]).sqrt();
        fy[i] /= n;
        fx[i] /= n;
    }
    (fy, fx)
}

/// Mean squared difference per label (1..=max) between the flows recomputed from the masks
/// and the network flows / 5 (metrics.flow_error).
pub fn flow_errors(m: &[u32], dy: &[f32], dx: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mx = m.iter().copied().max().unwrap_or(0) as usize;
    let (my, mxf) = masks_to_flows_f64(m, h, w);
    let mut sy = vec![0f64; mx + 1];
    let mut sx = vec![0f64; mx + 1];
    let mut cnt = vec![0f64; mx + 1];
    for i in 0..h * w {
        let l = m[i] as usize;
        if l == 0 {
            continue;
        }
        sy[l] += (my[i] - (dy[i] / 5.0) as f64).powi(2);
        sx[l] += (mxf[i] - (dx[i] / 5.0) as f64).powi(2);
        cnt[l] += 1.0;
    }
    (1..=mx).map(|l| if cnt[l] > 0.0 { (sy[l] / cnt[l] + sx[l] / cnt[l]) as f32 } else { 0.0 }).collect()
}

/// Remove the labels whose index (1-based) into `counts` has count < min_size; reproduces
/// cellpose's `counts = unique(masks, return_counts)[1][1:]` indexing (labels are assumed to
/// be 0..N with 0 present).
fn remove_small_python(m: &mut [u32], min_size: i32) {
    let mx = m.iter().copied().max().unwrap_or(0) as usize;
    let mut c = vec![0usize; mx + 1];
    for &v in m.iter() {
        c[v as usize] += 1;
    }
    let present: Vec<usize> = (0..=mx).filter(|&v| c[v] > 0).collect();
    let counts: Vec<usize> = present.iter().skip(1).map(|&v| c[v]).collect();
    let mut kill = vec![false; mx + 2];
    for (i, &n) in counts.iter().enumerate() {
        if (n as i64) < min_size as i64 && i + 1 <= mx {
            kill[i + 1] = true;
        }
    }
    for v in m.iter_mut() {
        if kill[*v as usize] {
            *v = 0;
        }
    }
    renumber(m);
}

/// fill_holes_and_remove_small_masks: drop small masks, fill holes of each mask (background
/// not 4-connected to the mask's bounding box border), drop small masks again.
pub fn fill_holes_and_remove_small(m: &mut [u32], h: usize, w: usize, min_size: i32) {
    if min_size > 0 {
        remove_small_python(m, min_size);
    }
    let mut j = 0u32;
    for (i, bb) in find_objects(m, h, w).into_iter().enumerate() {
        let Some((y0, y1, x0, x1)) = bb else { continue };
        let lab = i as u32 + 1;
        let (bh, bw) = (y1 - y0, x1 - x0);
        let inside: Vec<bool> = (0..bh * bw).map(|k| m[(y0 + k / bw) * w + x0 + k % bw] == lab).collect();
        // flood the background from the box border (4-connected)
        let mut outside = vec![false; bh * bw];
        let mut stack: Vec<usize> = Vec::new();
        for k in 0..bh * bw {
            let (y, x) = (k / bw, k % bw);
            if !inside[k] && (y == 0 || x == 0 || y == bh - 1 || x == bw - 1) {
                outside[k] = true;
                stack.push(k);
            }
        }
        while let Some(k) = stack.pop() {
            let (y, x) = (k / bw, k % bw);
            let mut nb = |yy: usize, xx: usize| {
                let q = yy * bw + xx;
                if !inside[q] && !outside[q] {
                    outside[q] = true;
                    stack.push(q);
                }
            };
            if y > 0 { nb(y - 1, x); }
            if y + 1 < bh { nb(y + 1, x); }
            if x > 0 { nb(y, x - 1); }
            if x + 1 < bw { nb(y, x + 1); }
        }
        j += 1;
        for k in 0..bh * bw {
            if !outside[k] {
                m[(y0 + k / bw) * w + x0 + k % bw] = j;
            }
        }
    }
    if min_size > 0 {
        remove_small_python(m, min_size);
    }
}

/// Everything compute_masks produces, for display.
pub struct MaskResult {
    pub masks: Vec<u32>,
    /// labels before the flow-error check
    pub masks_raw: Vec<u32>,
    /// flow error per label of masks_raw
    pub errors: Vec<f32>,
    /// end point of every pixel (NaN for background)
    pub end_y: Vec<f32>,
    pub end_x: Vec<f32>,
    /// histogram of end points (the sinks)
    pub hist: Vec<u32>,
    pub seeds: Vec<u32>,
    pub n_fg: usize,
}

/// The whole decoding (resize_and_compute_masks without resizing).
#[allow(clippy::too_many_arguments)]
pub fn compute_masks(
    dy: &[f32], dx: &[f32], cellprob: &[f32], h: usize, w: usize, niter: usize, cellprob_threshold: f32,
    flow_threshold: f32, min_size: i32, max_size_fraction: f32,
) -> MaskResult {
    let n = h * w;
    let (my, mx, inds) = masked_flows(dy, dx, cellprob, cellprob_threshold);
    let mut end_y = vec![f32::NAN; n];
    let mut end_x = vec![f32::NAN; n];
    if inds.is_empty() {
        return MaskResult {
            masks: vec![0; n], masks_raw: vec![0; n], errors: vec![], end_y, end_x,
            hist: vec![0; n], seeds: vec![], n_fg: 0,
        };
    }
    let (py, px) = follow_flows(&my, &mx, h, w, &inds, niter);
    for (k, &i) in inds.iter().enumerate() {
        end_y[i as usize] = py[k];
        end_x[i as usize] = px[k];
    }
    let (masks_raw, hist, seeds) = get_masks(&py, &px, &inds, h, w, max_size_fraction);
    // flow errors are computed for display even when the check is off (threshold 0)
    let errors = if masks_raw.iter().any(|&v| v > 0) { flow_errors(&masks_raw, dy, dx, h, w) } else { Vec::new() };
    let masks = finish_masks(&masks_raw, &errors, h, w, flow_threshold, min_size);
    MaskResult { masks, masks_raw, errors, end_y, end_x, hist, seeds, n_fg: inds.len() }
}

/// The last two steps on labels from `get_masks`: remove masks with flow error > threshold
/// (0 = no check), then fill holes and remove masks smaller than `min_size`.
pub fn finish_masks(masks_raw: &[u32], errors: &[f32], h: usize, w: usize, flow_threshold: f32, min_size: i32) -> Vec<u32> {
    let mut masks = masks_raw.to_vec();
    if flow_threshold > 0.0 && !errors.is_empty() {
        for v in masks.iter_mut() {
            if *v > 0 && errors[*v as usize - 1] > flow_threshold {
                *v = 0;
            }
        }
    }
    fill_holes_and_remove_small(&mut masks, h, w, min_size);
    masks
}

/// The flow used for tracking (dP * fg / 5), e.g. for drawing.
pub fn tracking_flow(dy: &[f32], dx: &[f32], cellprob: &[f32], thr: f32) -> (Vec<f32>, Vec<f32>) {
    let (my, mx, _) = masked_flows(dy, dx, cellprob, thr);
    (my, mx)
}
