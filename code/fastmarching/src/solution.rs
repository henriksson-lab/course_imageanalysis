// Model solution for the coding exercises of "From Dijkstra to fast marching: continuous minimal paths".

use crate::MinQueue;
use imgcourse::Image;

/// C1. The upwind update of fast marching: the arrival time at a pixel from `a`, the smaller
/// known value of its left and right neighbours, and `b`, the smaller known value of its
/// upper and lower neighbours (`f32::INFINITY` if none is known), with cost `p` per unit
/// length. If |a − b| < p: the larger root of (T − a)² + (T − b)² = p²; otherwise
/// min(a, b) + p (Dijkstra's update).
pub fn upwind_update(a: f32, b: f32, p: f32) -> f32 {
    if (a - b).abs() < p {
        (a + b + (2.0 * p * p - (a - b) * (a - b)).sqrt()) / 2.0
    } else {
        a.min(b) + p
    }
}

/// The value of pixel `(x, y)` if it is inside the image and final, else ∞.
fn known(t: &Image<f32>, done: &Image<bool>, x: isize, y: isize) -> f32 {
    if t.contains(x, y) && done[(x as usize, y as usize)] {
        t[(x as usize, y as usize)]
    } else {
        f32::INFINITY
    }
}

/// C2. Fast marching: the arrival time `T` of a front that starts at the seeds (`T = 0`)
/// and crosses pixel `(x, y)` at cost `cost(x, y)` per unit length. Dijkstra's order (the
/// unfinished pixel with the smallest tentative `T` is final), with the upwind update (C1)
/// from the final 4-neighbours.
pub fn fast_marching(cost: &Image<f32>, seeds: &[(usize, usize)]) -> Image<f32> {
    let (w, h) = (cost.width(), cost.height());
    let mut t = Image::new(w, h, f32::INFINITY);
    let mut done = Image::new(w, h, false);
    let mut queue = MinQueue::new();
    for &s in seeds {
        t[s] = 0.0;
        queue.push(0.0, s);
    }
    while let Some((_, (x, y))) = queue.pop() {
        if done[(x, y)] {
            continue; // an old entry
        }
        done[(x, y)] = true;
        for (dx, dy) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if !t.contains(nx, ny) || done[(nx as usize, ny as usize)] {
                continue;
            }
            // the smaller final neighbour along x and along y
            let a = known(&t, &done, nx - 1, ny).min(known(&t, &done, nx + 1, ny));
            let b = known(&t, &done, nx, ny - 1).min(known(&t, &done, nx, ny + 1));
            let v = (nx as usize, ny as usize);
            let new = upwind_update(a, b, cost[v]);
            if new < t[v] {
                t[v] = new;
                queue.push(new, v);
            }
        }
    }
    t
}

/// C3. For comparison, Dijkstra with the same cost: a step into pixel `v` costs
/// `length · cost(v)`, with length 1 (or √2 for a diagonal step when `eight` is true); the
/// seeds start at 0.
pub fn dijkstra_time(cost: &Image<f32>, seeds: &[(usize, usize)], eight: bool) -> Image<f32> {
    let (w, h) = (cost.width(), cost.height());
    let mut d = Image::new(w, h, f32::INFINITY);
    let mut done = Image::new(w, h, false);
    let mut queue = MinQueue::new();
    for &s in seeds {
        d[s] = 0.0;
        queue.push(0.0, s);
    }
    let steps: &[(isize, isize)] = if eight {
        &[(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    } else {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    };
    while let Some((dist, (x, y))) = queue.pop() {
        if done[(x, y)] {
            continue;
        }
        done[(x, y)] = true;
        for &(dx, dy) in steps {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if !d.contains(nx, ny) {
                continue;
            }
            let v = (nx as usize, ny as usize);
            let length = if dx != 0 && dy != 0 { std::f32::consts::SQRT_2 } else { 1.0 };
            let through = dist + length * cost[v];
            if !done[v] && through < d[v] {
                d[v] = through;
                queue.push(through, v);
            }
        }
    }
    d
}

/// C4. The minimal path from `target` back to a seed, by gradient descent on `T`: repeatedly
/// take a step of length `step` against the gradient of `T` (central differences at the
/// nearest pixel; a one-sided difference where a neighbour is outside or unreached). Stop at
/// a pixel with `T = 0` (a seed), or after 100 000 steps. Returns the positions visited,
/// from the target to the seed.
pub fn descend(t: &Image<f32>, target: (f32, f32), step: f32) -> Vec<(f32, f32)> {
    let value = |x: isize, y: isize| if t.contains(x, y) { t[(x as usize, y as usize)] } else { f32::INFINITY };
    // the derivative along one axis from the values before, at and after the pixel
    let derivative = |before: f32, here: f32, after: f32| match (before.is_finite(), after.is_finite()) {
        (true, true) => (after - before) / 2.0,
        (true, false) => here - before,
        (false, true) => after - here,
        (false, false) => 0.0,
    };
    let (mut x, mut y) = target;
    let mut path = vec![(x, y)];
    for _ in 0..100_000 {
        let (px, py) = (x.round() as isize, y.round() as isize);
        let here = value(px, py);
        if here == 0.0 {
            break; // reached a seed
        }
        let gx = derivative(value(px - 1, py), here, value(px + 1, py));
        let gy = derivative(value(px, py - 1), here, value(px, py + 1));
        let norm = (gx * gx + gy * gy).sqrt();
        if norm == 0.0 {
            break;
        }
        x -= step * gx / norm;
        y -= step * gy / norm;
        path.push((x, y));
    }
    path
}
