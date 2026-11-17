// Plumbing for "Shadows, mirrors and glass": the scene, and the ray casting of the last
// unit (nearest hit with spheres and planes), filled in. Not part of the exercise.

use crate::vec3::{Ray, Vec3};

/// What a surface does with light.
#[derive(Clone, Copy, Debug)]
pub enum Material {
    /// Matte, with the object's colour (Lambert shading, with a shadow ray).
    Diffuse,
    /// A mirror that reflects this fraction of the light.
    Mirror(f32),
    /// Glass with this refractive index (outside is air, index 1).
    Glass(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { centre: Vec3, radius: f32 },
    Plane { point: Vec3, normal: Vec3 },
}

#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub shape: Shape,
    pub material: Material,
    pub colour: [f32; 3],
    /// Draw a checkerboard of unit squares: `colour` and half of it.
    pub checker: bool,
}

impl Object {
    pub fn new(shape: Shape, material: Material, colour: [f32; 3]) -> Self {
        Object { shape, material, colour, checker: false }
    }

    /// The colour of a matte surface at the point `p`.
    pub fn colour_at(&self, p: Vec3) -> [f32; 3] {
        let dark = ((p.x.floor() + p.z.floor()) as i64).rem_euclid(2) == 1;
        if self.checker && dark {
            self.colour.map(|c| 0.4 * c)
        } else {
            self.colour
        }
    }
}

/// A hit: distance `t`, the unit normal turned towards the incoming ray, whether the ray
/// came from the outside (`front`), and the index of the object.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub normal: Vec3,
    pub front: bool,
    pub object: usize,
}

/// A scene: objects, one point light, the colour of the sky, and the ambient light.
#[derive(Clone, Debug)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub light: Vec3,
    pub sky: [f32; 3],
    pub ambient: f32,
}

impl Scene {
    /// The nearest hit with `t > 0` along the ray (unit direction), as in the last unit.
    pub fn nearest_hit(&self, ray: Ray) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut best_t = f32::INFINITY;
        for (index, object) in self.objects.iter().enumerate() {
            let found = match object.shape {
                Shape::Sphere { centre, radius } => {
                    let oc = ray.origin - centre;
                    let b = ray.dir.dot(oc);
                    let k = oc.dot(oc) - radius * radius;
                    let disc = b * b - k;
                    if disc < 0.0 {
                        None
                    } else {
                        let s = disc.sqrt();
                        let q = if b >= 0.0 { -b - s } else { -b + s };
                        let (t1, t2) = (q.min(k / q), q.max(k / q));
                        let t = if t1 > 0.0 { t1 } else { t2 };
                        (t > 0.0).then(|| (t, (ray.at(t) - centre) * (1.0 / radius)))
                    }
                }
                Shape::Plane { point, normal } => {
                    let t = (point - ray.origin).dot(normal) / ray.dir.dot(normal);
                    (t > 0.0).then_some((t, normal))
                }
            };
            if let Some((t, outward)) = found {
                if t < best_t {
                    let front = outward.dot(ray.dir) < 0.0;
                    let normal = if front { outward } else { -outward };
                    best = Some(Hit { t, normal, front, object: index });
                    best_t = t;
                }
            }
        }
        best
    }
}

/// The camera ray through the centre of pixel `(i, j)` of a `width` × `height` image, for a
/// camera at `eye` looking down −z with a vertical field of view `fov_deg` (last unit, C1).
pub fn camera_ray(eye: Vec3, i: usize, j: usize, width: usize, height: usize, fov_deg: f32) -> Ray {
    let xn = 2.0 * (i as f32 + 0.5) / width as f32 - 1.0;
    let yn = 1.0 - 2.0 * (j as f32 + 0.5) / height as f32;
    let tan_half = (0.5 * fov_deg).to_radians().tan();
    let d = Vec3 { x: xn * tan_half * width as f32 / height as f32, y: yn * tan_half, z: -1.0 };
    Ray::new(eye, d.normalize())
}
