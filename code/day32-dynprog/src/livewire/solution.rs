// Model solution for the coding exercises of "When there is no order: Dijkstra and live-wire".

use crate::livewire::MinQueue;
use imgcourse::{Border, Image};

/// The result of Dijkstra's algorithm: the shortest-path tree.
pub struct ShortestPaths {
    /// The cost of the cheapest path from a seed to every pixel (∞ if not reached).
    pub distance: Image<f32>,
    /// The neighbour each pixel's cheapest path comes from (`None` for the seeds).
    pub predecessor: Image<Option<(usize, usize)>>,
    /// The pixels in the order in which they became final.
    pub order: Vec<(usize, usize)>,
}

/// C1. Dijkstra's algorithm on the pixel grid. A path pays the cost of every pixel it enters,
/// times the step length (1, or √2 for a diagonal step when `eight` is true); a seed starts
/// with its own cost, `d(s) = C(s)`. Repeatedly make the unfinished pixel with the smallest
/// tentative distance final, and update its neighbours.
pub fn dijkstra(cost: &Image<f32>, seeds: &[(usize, usize)], eight: bool) -> ShortestPaths {
    let (w, h) = (cost.width(), cost.height());
    let mut distance = Image::new(w, h, f32::INFINITY);
    let mut predecessor = Image::new(w, h, None);
    let mut done = Image::new(w, h, false);
    let mut order = Vec::new();
    let mut queue = MinQueue::new();
    for &s in seeds {
        distance[s] = cost[s];
        queue.push(cost[s], s);
    }
    let steps: &[(isize, isize)] = if eight {
        &[(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    } else {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    };
    while let Some((d, (x, y))) = queue.pop() {
        if done[(x, y)] {
            continue; // an old entry: this pixel is already final
        }
        done[(x, y)] = true; // the smallest tentative value is final
        order.push((x, y));
        for &(dx, dy) in steps {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if !cost.contains(nx, ny) {
                continue;
            }
            let v = (nx as usize, ny as usize);
            let length = if dx != 0 && dy != 0 { std::f32::consts::SQRT_2 } else { 1.0 };
            let through = d + length * cost[v];
            if !done[v] && through < distance[v] {
                distance[v] = through;
                predecessor[v] = Some((x, y));
                queue.push(through, v);
            }
        }
    }
    ShortestPaths { distance, predecessor, order }
}

/// C2. The cheapest path from a seed to `target`: follow the predecessors back to a seed.
/// Returns the pixels from the seed to the target.
pub fn backtrack(paths: &ShortestPaths, target: (usize, usize)) -> Vec<(usize, usize)> {
    let mut path = vec![target];
    let mut p = target;
    while let Some(q) = paths.predecessor[p] {
        path.push(q);
        p = q;
    }
    path.reverse();
    path
}

/// C3. One raster pass instead of Dijkstra, from the seed at the top-left pixel: visit the
/// pixels row by row, left to right, with `d(x, y) = C(x, y) + min(d(x, y − 1), d(x − 1, y))`
/// and `d(0, 0) = C(0, 0)`. Exact only where the cheapest path never goes up or left.
pub fn raster_pass(cost: &Image<f32>) -> Image<f32> {
    let mut d = Image::new(cost.width(), cost.height(), f32::INFINITY);
    for (x, y) in cost.coords() {
        if (x, y) == (0, 0) {
            d[(x, y)] = cost[(x, y)];
            continue;
        }
        let up = if y > 0 { d[(x, y - 1)] } else { f32::INFINITY };
        let left = if x > 0 { d[(x - 1, y)] } else { f32::INFINITY };
        d[(x, y)] = cost[(x, y)] + up.min(left);
    }
    d
}

/// A small constant cost per step, so that flat regions do not give free detours.
pub const EPSILON: f32 = 0.02;

/// C4. The live-wire cost of entering a pixel, from the gradient magnitude `G` (central
/// differences, the edge pixel copied at the border): `EPSILON + 1 − G / G_max`. Low on
/// strong edges, about 1 in flat regions.
pub fn edge_cost(img: &Image<f32>) -> Image<f32> {
    let g = img.map_xy(|x, y, _| {
        let (x, y) = (x as isize, y as isize);
        let gx = (img.get_border(x + 1, y, Border::Clamp) - img.get_border(x - 1, y, Border::Clamp)) / 2.0;
        let gy = (img.get_border(x, y + 1, Border::Clamp) - img.get_border(x, y - 1, Border::Clamp)) / 2.0;
        (gx * gx + gy * gy).sqrt()
    });
    let g_max = g.max();
    g.map(|v| EPSILON + 1.0 - if g_max > 0.0 { v / g_max } else { 0.0 })
}
