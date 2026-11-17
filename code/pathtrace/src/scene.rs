// Plumbing for "Path tracing": the scene (spheres, planes and a square lamp), the nearest
// hit, the camera, and a few helpers. Not part of the exercise.

use crate::vec3::{v3, Ray, Vec3};
use imgcourse::Rng;

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { centre: Vec3, radius: f32 },
    Plane { point: Vec3, normal: Vec3 },
    /// A horizontal square lamp facing down: centre, half the side.
    Lamp { centre: Vec3, half: f32 },
}

/// A matte surface: the albedo `ρ` (per colour) and the emitted radiance `L_e`.
#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub shape: Shape,
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    /// The unit normal, turned towards the incoming ray.
    pub normal: Vec3,
    pub object: usize,
}

/// A point chosen on the lamp: the point `y`, the lamp's normal there, its emission and the
/// lamp's area (the point has density 1/area).
#[derive(Clone, Copy, Debug)]
pub struct LampSample {
    pub y: Vec3,
    pub normal: Vec3,
    pub emission: [f32; 3],
    pub area: f32,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub objects: Vec<Object>,
}

impl Scene {
    /// The nearest hit with `t > 1e-4` along a ray with a unit direction.
    pub fn nearest_hit(&self, ray: Ray) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut best_t = f32::INFINITY;
        for (index, o) in self.objects.iter().enumerate() {
            let found = match o.shape {
                Shape::Sphere { centre, radius } => {
                    let oc = ray.origin - centre;
                    let b = ray.dir.dot(oc);
                    let k = oc.dot(oc) - radius * radius;
                    let disc = b * b - k;
                    if disc < 0.0 {
                        None
                    } else {
                        let s = disc.sqrt();
                        let (t1, t2) = (-b - s, -b + s);
                        let t = if t1 > 1e-4 { t1 } else { t2 };
                        (t > 1e-4).then(|| (t, (ray.at(t) - centre) * (1.0 / radius)))
                    }
                }
                Shape::Plane { point, normal } => {
                    let t = (point - ray.origin).dot(normal) / ray.dir.dot(normal);
                    (t > 1e-4).then_some((t, normal))
                }
                Shape::Lamp { centre, half } => {
                    let t = (centre.y - ray.origin.y) / ray.dir.y;
                    let p = ray.at(t);
                    let inside = (p.x - centre.x).abs() <= half && (p.z - centre.z).abs() <= half;
                    (t > 1e-4 && inside).then_some((t, v3(0.0, -1.0, 0.0)))
                }
            };
            if let Some((t, outward)) = found {
                if t < best_t {
                    let normal = if outward.dot(ray.dir) > 0.0 { -outward } else { outward };
                    best = Some(Hit { t, normal, object: index });
                    best_t = t;
                }
            }
        }
        best
    }

    /// Is the segment from `p` to `q` free of objects (apart from its two ends)?
    pub fn visible(&self, p: Vec3, q: Vec3) -> bool {
        let d = q - p;
        let dist = d.length();
        match self.nearest_hit(Ray::new(p, d * (1.0 / dist))) {
            Some(hit) => hit.t > dist - 1e-3,
            None => true,
        }
    }

    /// A uniformly random point on the first lamp of the scene, or `None` if there is none.
    pub fn sample_lamp(&self, rng: &mut Rng) -> Option<LampSample> {
        self.objects.iter().find_map(|o| match o.shape {
            Shape::Lamp { centre, half } => Some(LampSample {
                y: centre + v3(half * (2.0 * rng.uniform() - 1.0), 0.0, half * (2.0 * rng.uniform() - 1.0)),
                normal: v3(0.0, -1.0, 0.0),
                emission: o.emission,
                area: 4.0 * half * half,
            }),
            _ => None,
        })
    }
}

/// Two unit vectors that make, with the unit normal `n`, a right-handed orthonormal frame.
pub fn frame(n: Vec3) -> (Vec3, Vec3) {
    let helper = if n.x.abs() > 0.9 { v3(0.0, 1.0, 0.0) } else { v3(1.0, 0.0, 0.0) };
    let t = helper.cross(n).normalize();
    (t, n.cross(t))
}

/// A pinhole camera at `eye` looking down −z with vertical field of view `fov_deg`. The ray
/// through the image position `(x, y)` in pixels (`(i + 0.5, j + 0.5)` is the centre of
/// pixel `(i, j)`).
pub fn camera_ray(eye: Vec3, x: f32, y: f32, width: usize, height: usize, fov_deg: f32) -> Ray {
    let tan_half = (0.5 * fov_deg).to_radians().tan();
    let xn = 2.0 * x / width as f32 - 1.0;
    let yn = 1.0 - 2.0 * y / height as f32;
    Ray::new(eye, v3(xn * tan_half * width as f32 / height as f32, yn * tan_half, -1.0).normalize())
}

/// The scene of the lecture: a closed box with a white floor, ceiling and back wall, a red
/// left and a green right wall, a square lamp under the ceiling (side `2 · half`, emitting
/// `power / (2 · half)²`, so that its total power does not depend on its size) and two
/// white balls. The front is open; the camera looks in from there.
pub fn cornell_box(half: f32, power: f32) -> Scene {
    let white = [0.7, 0.7, 0.7];
    let wall = |point: Vec3, normal: Vec3, albedo: [f32; 3]| Object { shape: Shape::Plane { point, normal }, albedo, emission: [0.0; 3] };
    let ball = |centre: Vec3, radius: f32| Object { shape: Shape::Sphere { centre, radius }, albedo: white, emission: [0.0; 3] };
    let le = power / (4.0 * half * half);
    Scene {
        objects: vec![
            wall(v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), white),          // floor
            wall(v3(0.0, 2.0, 0.0), v3(0.0, -1.0, 0.0), white),         // ceiling
            wall(v3(0.0, 0.0, -1.0), v3(0.0, 0.0, 1.0), white),         // back
            wall(v3(-1.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), [0.75, 0.15, 0.12]), // left, red
            wall(v3(1.0, 0.0, 0.0), v3(-1.0, 0.0, 0.0), [0.15, 0.6, 0.15]),  // right, green
            Object { shape: Shape::Lamp { centre: v3(0.0, 1.99, 0.0), half }, albedo: [0.0; 3], emission: [le; 3] },
            ball(v3(-0.4, 0.35, -0.35), 0.35),
            ball(v3(0.45, 0.3, 0.25), 0.3),
        ],
    }
}

/// How to render: samples (paths) per pixel, the most bounces per path, whether to sample
/// the lamp at every hit, and the seed of the random numbers.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub spp: usize,
    pub max_bounces: usize,
    pub light_sampling: bool,
    pub seed: u64,
}
