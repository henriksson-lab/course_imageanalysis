// Model solution for the coding exercises of "Flow fields to objects: Cellpose".

use imgcourse::Image;

/// The source of a cell: the cell pixel nearest to (median of the cell's x, median of its y).
pub fn source(mask: &Image<bool>) -> (usize, usize) {
    let pixels: Vec<(usize, usize)> = mask.coords().filter(|&(x, y)| mask[(x, y)]).collect();
    let mut xs: Vec<usize> = pixels.iter().map(|p| p.0).collect();
    let mut ys: Vec<usize> = pixels.iter().map(|p| p.1).collect();
    xs.sort();
    ys.sort();
    let (mx, my) = (xs[xs.len() / 2] as f32, ys[ys.len() / 2] as f32);
    let dist = |p: &(usize, usize)| (p.0 as f32 - mx).powi(2) + (p.1 as f32 - my).powi(2);
    *pixels.iter().min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap()).unwrap()
}

/// C1. The heat target of one cell (`mask` is true inside it). Start with 0 everywhere; at
/// every step add 1 at the source, then replace every cell pixel by the mean of the pixels
/// of its 3 × 3 neighbourhood that lie inside the cell (itself included). Outside the cell
/// the heat stays 0.
pub fn heat_target(mask: &Image<bool>, steps: usize) -> Image<f32> {
    let (sx, sy) = source(mask);
    let mut heat = Image::new(mask.width(), mask.height(), 0.0f32);
    for _ in 0..steps {
        heat[(sx, sy)] += 1.0;
        let mut next = Image::new(mask.width(), mask.height(), 0.0f32);
        for (x, y) in mask.coords() {
            if !mask[(x, y)] {
                continue;
            }
            let (mut sum, mut count) = (0.0, 0);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (u, v) = (x as isize + dx, y as isize + dy);
                    if mask.contains(u, v) && mask[(u as usize, v as usize)] {
                        sum += heat[(u as usize, v as usize)];
                        count += 1;
                    }
                }
            }
            next[(x, y)] = sum / count as f32;
        }
        heat = next;
    }
    heat
}

/// C2. The arrows of the target: at every cell pixel the heat difference of its two
/// neighbours, `(right − left, below − above)`, scaled to length 1 (a zero difference stays
/// zero). Neighbours outside the cell count as 0; outside the cell the arrow is (0, 0).
pub fn arrows(heat: &Image<f32>, mask: &Image<bool>) -> Image<[f32; 2]> {
    let h = |x: isize, y: isize| {
        if mask.contains(x, y) && mask[(x as usize, y as usize)] { heat[(x as usize, y as usize)] } else { 0.0 }
    };
    mask.map_xy(|x, y, inside| {
        if !inside {
            return [0.0, 0.0];
        }
        let (x, y) = (x as isize, y as isize);
        let gx = h(x + 1, y) - h(x - 1, y);
        let gy = h(x, y + 1) - h(x, y - 1);
        let len = (gx * gx + gy * gy).sqrt();
        if len > 0.0 { [gx / len, gy / len] } else { [0.0, 0.0] }
    })
}

/// C3. Follow the flow: for every pixel of `mask`, put a particle at the pixel and move it
/// `steps` times by the arrow of the pixel it is on (the pixel at its rounded position).
/// Returns every particle's end position (`[x, y]`; `[0, 0]` outside the mask).
pub fn follow(flow: &Image<[f32; 2]>, mask: &Image<bool>, steps: usize) -> Image<[f32; 2]> {
    let (w, h) = (flow.width(), flow.height());
    mask.map_xy(|x, y, inside| {
        if !inside {
            return [0.0, 0.0];
        }
        let (mut px, mut py) = (x as f32, y as f32);
        for _ in 0..steps {
            let ix = (px.round().max(0.0) as usize).min(w - 1);
            let iy = (py.round().max(0.0) as usize).min(h - 1);
            let [vx, vy] = flow[(ix, iy)];
            px += vx;
            py += vy;
        }
        [px, py]
    })
}

/// C4. Masks from the end points. Count the particles that ended in every pixel (at their
/// rounded end position). The sink pixels are those with at least `min_count` particles;
/// sink pixels that touch (8-neighbours) form one sink. Every mask pixel whose particle ended
/// in a sink gets the number of that sink, 1, 2, ... in the order the sinks are found (row
/// by row); the others get 0 (no object).
pub fn group_ends(ends: &Image<[f32; 2]>, mask: &Image<bool>, min_count: usize) -> Image<u32> {
    let (w, h) = (mask.width(), mask.height());
    let end_pixel = |x: usize, y: usize| {
        let [ex, ey] = ends[(x, y)];
        ((ex.round().max(0.0) as usize).min(w - 1), (ey.round().max(0.0) as usize).min(h - 1))
    };
    // particles per pixel
    let mut count = Image::new(w, h, 0usize);
    for (x, y) in mask.coords() {
        if mask[(x, y)] {
            count[end_pixel(x, y)] += 1;
        }
    }
    // sinks: 8-connected groups of pixels with at least min_count particles
    let is_sink = |x: isize, y: isize| count.contains(x, y) && count[(x as usize, y as usize)] >= min_count.max(1);
    let mut sink = Image::new(w, h, 0u32);
    let mut next = 1;
    for (x, y) in count.coords() {
        if !is_sink(x as isize, y as isize) || sink[(x, y)] != 0 {
            continue;
        }
        let mut stack = vec![(x, y)];
        sink[(x, y)] = next;
        while let Some((u, v)) = stack.pop() {
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (a, b) = (u as isize + dx, v as isize + dy);
                    if is_sink(a, b) && sink[(a as usize, b as usize)] == 0 {
                        sink[(a as usize, b as usize)] = next;
                        stack.push((a as usize, b as usize));
                    }
                }
            }
        }
        next += 1;
    }
    mask.map_xy(|x, y, inside| if inside { sink[end_pixel(x, y)] } else { 0 })
}

/// C5. The flow error of one mask: the mean, over its pixels, of the squared difference
/// between the predicted arrows and the arrows recomputed from the mask itself.
pub fn flow_error(predicted: &[[f32; 2]], own: &[[f32; 2]]) -> f32 {
    let mut sum = 0.0;
    for i in 0..predicted.len() {
        sum += (predicted[i][0] - own[i][0]).powi(2) + (predicted[i][1] - own[i][1]).powi(2);
    }
    sum / predicted.len() as f32
}
