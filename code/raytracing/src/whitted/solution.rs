// Model solution for the coding exercises of "Shadows, mirrors and glass".

use crate::whitted::scene::{Material, Scene};
use crate::common::{Ray, Vec3};

/// C1. The mirror direction of `d` at a surface with unit normal `n`:
/// `r = d − 2 (d · n) n`.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - 2.0 * d.dot(n) * n
}

/// C2. The refracted direction (Snell's law in vector form) of the unit direction `d` at a
/// surface with unit normal `n` that faces the incoming ray (`d · n < 0`), with
/// `eta = n1 / n2`. `None` means total internal reflection.
pub fn refract(d: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    // Slide "The math: Snell's law in vector form".
    let c1 = -d.dot(n); // cos θ1
    let k = 1.0 - eta * eta * (1.0 - c1 * c1); // cos² θ2
    if k < 0.0 {
        return None;
    }
    Some(eta * d + (eta * c1 - k.sqrt()) * n)
}

/// C3. The reflected fraction `R` by Schlick's formula, for light going from index `n1` into
/// index `n2` at an angle with cosine `cos1` to the normal. `R0 = ((n1 − n2)/(n1 + n2))²` and
/// `R = R0 + (1 − R0)(1 − cos)⁵`, where `cos` is the cosine in the less dense medium. Total
/// internal reflection gives `R = 1`.
pub fn schlick(cos1: f32, n1: f32, n2: f32) -> f32 {
    let r0 = ((n1 - n2) / (n1 + n2)).powi(2);
    let mut cos = cos1;
    if n1 > n2 {
        // From the denser medium: use the angle on the air side, from Snell's law.
        let sin2 = n1 / n2 * (1.0 - cos1 * cos1).sqrt();
        if sin2 > 1.0 {
            return 1.0; // total internal reflection
        }
        cos = (1.0 - sin2 * sin2).sqrt();
    }
    r0 + (1.0 - r0) * (1.0 - cos).powi(5)
}

/// C4. The shadow ray: is the point light at `light` hidden from the point `p` (unit normal
/// `n` on the side the light should be on)? The ray starts at `p + eps · n`; any hit closer
/// than the light counts (glass blocks light too, as in Whitted's model).
pub fn in_shadow(p: Vec3, n: Vec3, light: Vec3, scene: &Scene, eps: f32) -> bool {
    let start = p + eps * n;
    let to_light = light - start;
    let distance = to_light.length();
    match scene.nearest_hit(Ray::new(start, to_light * (1.0 / distance))) {
        Some(hit) => hit.t < distance,
        None => false,
    }
}

/// C5. Whitted ray tracing: the colour seen along `ray`. Matte surfaces: ambient + Lambert
/// with a shadow ray. Mirrors: the reflectance times `trace` of the reflected ray. Glass:
/// `R` times the reflected ray plus `1 − R` times the refracted ray (Schlick). Mirrors and
/// glass at `depth == max_depth` give black. Every ray traced, shadow rays included, adds
/// 1 to `rays`.
pub fn trace(ray: Ray, scene: &Scene, depth: usize, max_depth: usize, eps: f32, rays: &mut usize) -> [f32; 3] {
    *rays += 1;
    let Some(hit) = scene.nearest_hit(ray) else {
        return scene.sky;
    };
    let object = &scene.objects[hit.object];
    let p = ray.at(hit.t);
    let n = hit.normal; // faces the incoming ray
    let black = [0.0; 3];
    match object.material {
        Material::Diffuse => {
            let colour = object.colour_at(p);
            let l = (scene.light - p).normalize();
            let mut light = scene.ambient;
            let lambert = n.dot(l);
            if lambert > 0.0 {
                *rays += 1;
                if !in_shadow(p, n, scene.light, scene, eps) {
                    light += lambert;
                }
            }
            colour.map(|c| c * light)
        }
        Material::Mirror(reflectance) => {
            if depth == max_depth {
                return black;
            }
            let r = Ray::new(p + eps * n, reflect(ray.dir, n));
            trace(r, scene, depth + 1, max_depth, eps, rays).map(|c| reflectance * c)
        }
        Material::Glass(index) => {
            if depth == max_depth {
                return black;
            }
            // Entering the glass from the front, or leaving it from the inside.
            let (n1, n2) = if hit.front { (1.0, index) } else { (index, 1.0) };
            let r = schlick(-ray.dir.dot(n), n1, n2);
            let reflected = trace(Ray::new(p + eps * n, reflect(ray.dir, n)), scene, depth + 1, max_depth, eps, rays);
            let mut colour = reflected.map(|c| r * c);
            if let Some(t) = refract(ray.dir, n, n1 / n2) {
                // The refracted ray starts on the other side of the surface.
                let refracted = trace(Ray::new(p - eps * n, t), scene, depth + 1, max_depth, eps, rays);
                for k in 0..3 {
                    colour[k] += (1.0 - r) * refracted[k];
                }
            }
            colour
        }
    }
}
