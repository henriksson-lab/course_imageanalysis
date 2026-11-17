// Model solution for the coding exercises of "Path tracing: the rendering equation, Monte
// Carlo and noise".

use crate::pathtrace::scene::{camera_ray, frame, Scene, Settings};
use crate::pathtrace::vec3::{Ray, Vec3};
use imgcourse::{Image, Rng};
use std::f32::consts::PI;

/// C1. The Monte Carlo estimate of `∫ f`: the mean of `f(x) / p(x)` over the samples `xs`,
/// which were drawn from the density `p`.
pub fn mc_estimate(f: impl Fn(f32) -> f32, p: impl Fn(f32) -> f32, xs: &[f32]) -> f32 {
    let mut sum = 0.0;
    for &x in xs {
        sum += f(x) / p(x);
    }
    sum / xs.len() as f32
}

/// C2. A cosine-weighted random direction around the unit normal `n` (density cos θ / π),
/// from two uniform numbers `u1, u2` in [0, 1): a uniform point on the unit disc, lifted
/// straight up onto the hemisphere (Malley's method).
pub fn cosine_direction(n: Vec3, u1: f32, u2: f32) -> Vec3 {
    // Uniform on the disc: radius √u1, angle 2π u2.
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let (x, y) = (r * phi.cos(), r * phi.sin());
    let z = (1.0 - u1).max(0.0).sqrt(); // lift onto the hemisphere: x² + y² + z² = 1
    let (t, b) = frame(n);
    x * t + y * b + z * n
}

/// C3. Direct light at the point `x` (unit normal `n`, matte with albedo `albedo`) from one
/// point `y` chosen uniformly on a lamp of area `area`, with unit normal `lamp_normal` and
/// emitted radiance `emission`, assuming nothing blocks the way:
/// `f_r · L_e · cos θx · cos θℓ / |x − y|² · area`, with `f_r = albedo / π`.
pub fn direct_light(x: Vec3, n: Vec3, albedo: [f32; 3], y: Vec3, lamp_normal: Vec3, emission: [f32; 3], area: f32) -> [f32; 3] {
    let to_lamp = y - x;
    let dist2 = to_lamp.dot(to_lamp);
    let w = to_lamp * (1.0 / dist2.sqrt());
    let cos_x = n.dot(w);
    let cos_l = -lamp_normal.dot(w);
    if cos_x <= 0.0 || cos_l <= 0.0 {
        return [0.0; 3]; // the lamp is behind the surface, or we see its back
    }
    let g = cos_x * cos_l / dist2 * area;
    let mut out = [0.0; 3];
    for k in 0..3 {
        out[k] = albedo[k] / PI * emission[k] * g;
    }
    out
}

/// C4. One random light path from the camera ray `ray`: the radiance it carries back.
/// At each hit add the emitted light (times the path's throughput), then continue in a
/// cosine-weighted random direction, multiplying the throughput by the albedo (BRDF × cos ÷
/// pdf = ρ). At most `max_bounces` bounces. With `light_sampling`, every hit also gets one
/// lamp sample (C3, if the lamp is visible), and a lamp found by a bounce is then not
/// counted again (only the lamp seen directly by the camera is).
pub fn radiance(ray: Ray, scene: &Scene, rng: &mut Rng, max_bounces: usize, light_sampling: bool) -> [f32; 3] {
    let mut colour = [0.0f32; 3];
    let mut throughput = [1.0f32; 3];
    let mut ray = ray;
    for bounce in 0..=max_bounces {
        let Some(hit) = scene.nearest_hit(ray) else {
            break; // the path leaves the scene
        };
        let object = &scene.objects[hit.object];
        if bounce == 0 || !light_sampling {
            for k in 0..3 {
                colour[k] += throughput[k] * object.emission[k];
            }
        }
        if bounce == max_bounces || object.albedo == [0.0; 3] {
            break;
        }
        let p = ray.at(hit.t);
        let n = hit.normal;
        if light_sampling {
            if let Some(lamp) = scene.sample_lamp(rng) {
                if scene.visible(p + 1e-4 * n, lamp.y) {
                    let d = direct_light(p, n, object.albedo, lamp.y, lamp.normal, lamp.emission, lamp.area);
                    for k in 0..3 {
                        colour[k] += throughput[k] * d[k];
                    }
                }
            }
        }
        // The next direction, cosine-weighted: the weight is just the albedo.
        for k in 0..3 {
            throughput[k] *= object.albedo[k];
        }
        ray = Ray::new(p + 1e-4 * n, cosine_direction(n, rng.uniform(), rng.uniform()));
    }
    colour
}

/// C5. Render: for every pixel, the mean of `s.spp` paths (C4), each through a random point
/// of the pixel, with the camera at `eye` looking down −z (vertical field of view 40°). The
/// random numbers come from `Rng::new(s.seed)`.
pub fn render(scene: &Scene, eye: Vec3, width: usize, height: usize, s: Settings) -> Image<[f32; 3]> {
    let mut rng = Rng::new(s.seed);
    let mut img = Image::new(width, height, [0.0f32; 3]);
    for (i, j) in img.coords() {
        let mut sum = [0.0f32; 3];
        for _ in 0..s.spp {
            let (x, y) = (i as f32 + rng.uniform(), j as f32 + rng.uniform());
            let l = radiance(camera_ray(eye, x, y, width, height, 40.0), scene, &mut rng, s.max_bounces, s.light_sampling);
            for k in 0..3 {
                sum[k] += l[k];
            }
        }
        img[(i, j)] = sum.map(|v| v / s.spp as f32);
    }
    img
}
