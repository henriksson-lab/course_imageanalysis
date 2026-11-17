// Counts the tests per ray for brute force, two BVHs and a grid on random 2D scenes, and draws
// the top levels of the median and SAH trees. Numbers on the terminal, pictures in out/bvh/.
//   cargo run -p bvh                      (your code)
//   cargo run -p bvh --features solution  (the model solution)

use bvh::*;
use imgcourse::*;

/// A uniform grid: for every cell, the discs whose boxes overlap it.
struct Grid {
    cells: usize,
    size: f32,
    lists: Vec<Vec<usize>>,
}

impl Grid {
    fn new(discs: &[Disc], cells: usize) -> Grid {
        let size = 100.0 / cells as f32;
        let mut lists = vec![Vec::new(); cells * cells];
        let cell_of = |v: f32| ((v / size).floor() as isize).clamp(0, cells as isize - 1) as usize;
        for (k, d) in discs.iter().enumerate() {
            let b = d.bbox();
            for cy in cell_of(b.min[1])..=cell_of(b.max[1]) {
                for cx in cell_of(b.min[0])..=cell_of(b.max[0]) {
                    lists[cy * cells + cx].push(k);
                }
            }
        }
        Grid { cells, size, lists }
    }

    /// Walk the ray through the cells (your `grid_walk`), test the discs listed in each, and
    /// stop in the first cell whose own part of the ray holds the nearest hit.
    fn trace(&self, discs: &[Disc], o: P2, d: P2) -> (Option<usize>, usize, usize) {
        let walk = grid_walk(o, d, self.cells, self.size);
        let mut tests = 0;
        let mut best = (f32::INFINITY, None);
        for (n, &(cx, cy, _)) in walk.iter().enumerate() {
            for &k in &self.lists[cy * self.cells + cx] {
                tests += 1;
                if let Some(t) = discs[k].hit(o, d) {
                    if t < best.0 {
                        best = (t, Some(k));
                    }
                }
            }
            let t_exit = walk.get(n + 1).map_or(f32::INFINITY, |c| c.2);
            if best.0 <= t_exit {
                return (best.1, n + 1, tests);
            }
        }
        (best.1, walk.len(), tests)
    }
}

/// Random rays: origins anywhere in the scene, directions all around.
fn random_rays(n: usize, seed: u64) -> Vec<(P2, P2)> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let a = std::f32::consts::TAU * rng.uniform();
            ([100.0 * rng.uniform(), 100.0 * rng.uniform()], [a.cos(), a.sin()])
        })
        .collect()
}

/// Draw the discs and the boxes of the tree down to `levels` levels below the root.
fn draw(discs: &[Disc], t: &Bvh, levels: usize, path: &str) {
    let s = 4.0; // pixels per unit
    let mut img = Image::new(400, 400, [1.0f32; 3]);
    for d in discs {
        let b = d.bbox();
        for y in (b.min[1] * s) as usize..=((b.max[1] * s) as usize).min(399) {
            for x in (b.min[0] * s) as usize..=((b.max[0] * s) as usize).min(399) {
                let (dx, dy) = (x as f32 / s - d.centre[0], y as f32 / s - d.centre[1]);
                if dx * dx + dy * dy <= d.radius * d.radius + 0.1 {
                    img[(x, y)] = [0.3, 0.3, 0.3];
                }
            }
        }
    }
    let colours = [[0.0, 0.0, 0.0], [0.9, 0.4, 0.0], [0.1, 0.4, 0.9], [0.1, 0.65, 0.2]];
    let mut stack = vec![(0, 0)];
    while let Some((i, level)) = stack.pop() {
        let n = &t.nodes[i];
        let b = n.bbox;
        let c = colours[level.min(3)];
        let (x0, x1) = ((b.min[0] * s).max(0.0) as usize, ((b.max[0] * s) as usize).min(399));
        let (y0, y1) = ((b.min[1] * s).max(0.0) as usize, ((b.max[1] * s) as usize).min(399));
        for x in x0..=x1 {
            img[(x, y0)] = c;
            img[(x, y1)] = c;
        }
        for y in y0..=y1 {
            img[(x0, y)] = c;
            img[(x1, y)] = c;
        }
        if !n.is_leaf() && level < levels {
            stack.push((n.left, level + 1));
            stack.push((n.right, level + 1));
        }
    }
    save_rgb(&img, path);
}

fn main() {
    let dir = "out/bvh";
    let rays = random_rays(1000, 7);
    for layout in [Layout::Uniform, Layout::Clustered, Layout::Stadium] {
        println!("\n{layout:?}: average tests per ray (1000 random rays)");
        println!("     N   brute   median BVH (box+obj)   SAH BVH (box+obj)   grid (cells, obj)");
        for n in [16, 64, 256, 1024, 4096] {
            let discs = random_scene(layout, n, 1);
            let median = build_median(&discs, 2);
            let sah = build_sah(&discs);
            let grid = Grid::new(&discs, ((n as f32 / 2.0).sqrt().round() as usize).max(1));
            let (mut m, mut s, mut g) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            for &(o, d) in &rays {
                let (hm, sm) = traverse(&median, &discs, o, d);
                let (hs, ss) = traverse(&sah, &discs, o, d);
                let (hg, cells, tests) = grid.trace(&discs, o, d);
                assert!(hm.map(|h| h.1) == hs.map(|h| h.1) && hs.map(|h| h.1) == hg, "the three structures disagree");
                m = [m[0] + sm.box_tests as f32, m[1] + sm.object_tests as f32];
                s = [s[0] + ss.box_tests as f32, s[1] + ss.object_tests as f32];
                g = [g[0] + cells as f32, g[1] + tests as f32];
            }
            let r = rays.len() as f32;
            println!(
                "{n:6} {n:7}   {:6.1} = {:5.1} + {:4.1}   {:6.1} = {:5.1} + {:4.1}   {:5.1}, {:5.1}",
                (m[0] + m[1]) / r, m[0] / r, m[1] / r, (s[0] + s[1]) / r, s[0] / r, s[1] / r, g[0] / r, g[1] / r
            );
        }
    }

    // The teapot in a stadium, with rays aimed at the teapot (the cluster at (50, 50)).
    println!("\nStadium, rays aimed at the teapot: average tests per ray");
    for n in [256, 1024, 4096] {
        let discs = random_scene(Layout::Stadium, n, 1);
        let sah = build_sah(&discs);
        let grid = Grid::new(&discs, (n as f32 / 2.0).sqrt().round() as usize);
        let (mut s, mut g) = (0.0, 0.0);
        for &(o, _) in &rays {
            let len = ((50.0 - o[0]).powi(2) + (50.0 - o[1]).powi(2)).sqrt().max(1e-3);
            let d = [(50.0 - o[0]) / len, (50.0 - o[1]) / len];
            let (_, st) = traverse(&sah, &discs, o, d);
            s += (st.box_tests + st.object_tests) as f32;
            g += grid.trace(&discs, o, d).2 as f32;
        }
        println!("{n:6}: SAH BVH {:6.1} tests (boxes + discs), grid {:6.1} disc tests", s / rays.len() as f32, g / rays.len() as f32);
    }

    let discs = random_scene(Layout::Clustered, 256, 1);
    let (median, sah) = (build_median(&discs, 2), build_sah(&discs));
    println!("\nclustered, 256 discs: median tree depth {}, SAH cost {:.1}; SAH tree depth {}, SAH cost {:.1}", median.depth(), median.sah_cost(), sah.depth(), sah.sah_cost());
    draw(&discs, &median, 2, &format!("{dir}/median-levels.png"));
    draw(&discs, &sah, 2, &format!("{dir}/sah-levels.png"));
    println!("wrote the pictures to {dir}/");
}
