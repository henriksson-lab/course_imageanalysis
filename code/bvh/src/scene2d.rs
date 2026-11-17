// Plumbing for "Tracing fast": a 2D scene of discs, boxes, and the tree type. Not part of
// the exercise. Everything is in 2D, as in the demo: the ideas are the same in 3D.

/// A point or a direction in the plane, `[x, y]`.
pub type P2 = [f32; 2];

/// An axis-aligned box: `min[k] ≤ coordinate k ≤ max[k]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: P2,
    pub max: P2,
}

impl Aabb {
    /// The empty box (the start of a union).
    pub fn empty() -> Self {
        Aabb { min: [f32::INFINITY; 2], max: [f32::NEG_INFINITY; 2] }
    }
    /// The smallest box that holds both boxes.
    pub fn union(self, o: Aabb) -> Aabb {
        Aabb { min: [self.min[0].min(o.min[0]), self.min[1].min(o.min[1])], max: [self.max[0].max(o.max[0]), self.max[1].max(o.max[1])] }
    }
    /// The perimeter: in 2D it plays the part of the surface area in the SAH.
    pub fn perimeter(self) -> f32 {
        2.0 * ((self.max[0] - self.min[0]) + (self.max[1] - self.min[1]))
    }
    /// Does the box contain the other box?
    pub fn contains(self, o: Aabb) -> bool {
        (0..2).all(|k| self.min[k] <= o.min[k] && o.max[k] <= self.max[k])
    }
}

/// A disc, the objects of the scene.
#[derive(Clone, Copy, Debug)]
pub struct Disc {
    pub centre: P2,
    pub radius: f32,
}

impl Disc {
    pub fn bbox(self) -> Aabb {
        let [x, y] = self.centre;
        let r = self.radius;
        Aabb { min: [x - r, y - r], max: [x + r, y + r] }
    }

    /// Where the ray `o + t d` first hits the disc (`t > 0`), or `None` (last unit's sphere
    /// test in 2D; `d` need not have length 1).
    pub fn hit(self, o: P2, d: P2) -> Option<f32> {
        let oc = [o[0] - self.centre[0], o[1] - self.centre[1]];
        let a = d[0] * d[0] + d[1] * d[1];
        let b = (d[0] * oc[0] + d[1] * oc[1]) / a;
        let k = (oc[0] * oc[0] + oc[1] * oc[1] - self.radius * self.radius) / a;
        let disc = b * b - k;
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        let q = if b >= 0.0 { -b - s } else { -b + s };
        let (t1, t2) = if q == 0.0 { (0.0, 0.0) } else { (q.min(k / q), q.max(k / q)) };
        if t1 > 0.0 {
            Some(t1)
        } else if t2 > 0.0 {
            Some(t2)
        } else {
            None
        }
    }
}

/// One node of a bounding-volume hierarchy. A leaf has a non-empty list `objects`; an inner
/// node has two children, `left` and `right` (indices into `Bvh::nodes`).
#[derive(Clone, Debug)]
pub struct Node {
    pub bbox: Aabb,
    pub left: usize,
    pub right: usize,
    pub objects: Vec<usize>,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        !self.objects.is_empty()
    }
}

/// A bounding-volume hierarchy: the root is `nodes[0]`.
#[derive(Clone, Debug)]
pub struct Bvh {
    pub nodes: Vec<Node>,
}

impl Bvh {
    /// The depth of the tree (a single leaf has depth 0).
    pub fn depth(&self) -> usize {
        fn d(b: &Bvh, i: usize) -> usize {
            let n = &b.nodes[i];
            if n.is_leaf() {
                0
            } else {
                1 + d(b, n.left).max(d(b, n.right))
            }
        }
        d(self, 0)
    }

    /// The SAH estimate of the cost of a random ray that hits the root: every inner node
    /// costs 1 (its children's box tests, counted as one), every leaf its number of objects,
    /// each weighted by its perimeter ÷ the root's.
    pub fn sah_cost(&self) -> f32 {
        let root = self.nodes[0].bbox.perimeter();
        self.nodes.iter().map(|n| n.bbox.perimeter() / root * if n.is_leaf() { n.objects.len() as f32 } else { 1.0 }).sum()
    }
}

/// What a traversal cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub box_tests: usize,
    pub object_tests: usize,
}

/// The layouts of the random scenes, in the square `[0, 100]²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    /// Discs spread evenly.
    Uniform,
    /// Four tight clusters.
    Clustered,
    /// "A teapot in a stadium": almost all discs in one tiny cluster, a few on a ring.
    Stadium,
}

/// `n` random discs (radius 0.4 to 1.2, smaller in the clusters) with a fixed seed.
pub fn random_scene(layout: Layout, n: usize, seed: u64) -> Vec<Disc> {
    let mut rng = imgcourse::Rng::new(seed);
    let clusters = [[25.0, 30.0], [70.0, 25.0], [40.0, 75.0], [80.0, 70.0]];
    (0..n)
        .map(|i| {
            let r = 0.4 + 0.8 * rng.uniform();
            let disc = match layout {
                Layout::Uniform => Disc { centre: [2.0 + 96.0 * rng.uniform(), 2.0 + 96.0 * rng.uniform()], radius: r },
                Layout::Clustered => {
                    let c = clusters[i % 4];
                    Disc { centre: [c[0] + 5.0 * rng.normal(), c[1] + 5.0 * rng.normal()], radius: 0.5 * r }
                }
                Layout::Stadium => {
                    if i % 16 == 0 {
                        let a = std::f32::consts::TAU * rng.uniform();
                        Disc { centre: [50.0 + 45.0 * a.cos(), 50.0 + 45.0 * a.sin()], radius: r }
                    } else {
                        Disc { centre: [50.0 + 1.5 * rng.normal(), 50.0 + 1.5 * rng.normal()], radius: 0.1 * r }
                    }
                }
            };
            // Keep every disc inside the square.
            Disc { centre: disc.centre.map(|v| v.clamp(2.0, 98.0)), radius: disc.radius }
        })
        .collect()
}
