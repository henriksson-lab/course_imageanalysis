// Coding exercises for "Path tracing".
// Replace each `todo!()` with your code, then check it with `cargo test -p day50 pathtrace::`
// and look at `cargo run -p day50 -- pathtrace`.
#![allow(unused_variables, unused_imports)]

use crate::pathtrace::scene::{camera_ray, frame, Scene, Settings};
use crate::pathtrace::vec3::{Ray, Vec3};
use imgcourse::{Image, Rng};
use std::f32::consts::PI;

/// C1. The Monte Carlo estimate of `∫ f`: the mean of `f(x) / p(x)` over the samples `xs`,
/// which were drawn from the density `p`.
///
/// Hint: a loop over `xs`, adding `f(x) / p(x)`.
pub fn mc_estimate(f: impl Fn(f32) -> f32, p: impl Fn(f32) -> f32, xs: &[f32]) -> f32 {
    todo!("C1: mc_estimate")
}

/// C2. A cosine-weighted random direction around the unit normal `n` (density cos θ / π),
/// from two uniform numbers `u1, u2` in [0, 1): a uniform point on the unit disc, lifted
/// straight up onto the hemisphere (Malley's method).
///
/// Hint: `r = √u1`, `φ = 2π u2`, `(x, y) = (r cos φ, r sin φ)`, `z = √(1 − u1)`; then
/// `frame(n)` gives two unit vectors `t, b` across the normal, and the direction is `x t + y b + z n`.
pub fn cosine_direction(n: Vec3, u1: f32, u2: f32) -> Vec3 {
    todo!("C2: cosine_direction")
}

/// C3. Direct light at the point `x` (unit normal `n`, matte with albedo `albedo`) from one
/// point `y` chosen uniformly on a lamp of area `area`, with unit normal `lamp_normal` and
/// emitted radiance `emission`, assuming nothing blocks the way:
/// `f_r · L_e · cos θx · cos θℓ / |x − y|² · area`, with `f_r = albedo / π`.
///
/// Hint: `w` = the unit vector from `x` to `y`; `cos θx = n · w`, `cos θℓ = −lamp_normal · w`;
/// if either is ≤ 0 there is no light.
pub fn direct_light(x: Vec3, n: Vec3, albedo: [f32; 3], y: Vec3, lamp_normal: Vec3, emission: [f32; 3], area: f32) -> [f32; 3] {
    todo!("C3: direct_light")
}

/// C4. One random light path from the camera ray `ray`: the radiance it carries back.
/// At each hit add the emitted light (times the path's throughput), then continue in a
/// cosine-weighted random direction, multiplying the throughput by the albedo (BRDF × cos ÷
/// pdf = ρ). At most `max_bounces` bounces. With `light_sampling`, every hit also gets one
/// lamp sample (C3, if the lamp is visible), and a lamp found by a bounce is then not
/// counted again (only the lamp seen directly by the camera is).
///
/// Hint: keep `colour` and `throughput` (both `[f32; 3]`) and loop over the bounces.
/// `scene.nearest_hit(ray)`, `scene.objects[hit.object]` has `emission` and `albedo`. For lamp
/// sampling: `scene.sample_lamp(rng)` and `scene.visible(p + 1e-4 n, y)`. The next ray starts
/// at `p + 1e-4 n`.
pub fn radiance(ray: Ray, scene: &Scene, rng: &mut Rng, max_bounces: usize, light_sampling: bool) -> [f32; 3] {
    todo!("C4: radiance")
}

/// C5. Render: for every pixel, the mean of `s.spp` paths (C4), each through a random point
/// of the pixel, with the camera at `eye` looking down −z (vertical field of view 40°). The
/// random numbers come from `Rng::new(s.seed)`.
///
/// Hint: `camera_ray(eye, x, y, width, height, 40.0)` with `x = i + rng.uniform()`, `y = j + rng.uniform()`.
pub fn render(scene: &Scene, eye: Vec3, width: usize, height: usize, s: Settings) -> Image<[f32; 3]> {
    todo!("C5: render")
}
