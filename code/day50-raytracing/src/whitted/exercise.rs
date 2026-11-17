// Coding exercises for "Shadows, mirrors and glass".
// Replace each `todo!()` with your code, then check it with `cargo test -p day50 whitted::`
// and look at `cargo run -p day50 -- whitted`.
#![allow(unused_variables, unused_imports)]

use crate::whitted::scene::{Material, Scene};
use crate::whitted::vec3::{Ray, Vec3};

/// C1. The mirror direction of `d` at a surface with unit normal `n`:
/// `r = d − 2 (d · n) n`.
///
/// Hint: one line; vectors have `dot`, and `*` with a number.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    todo!("C1: reflect")
}

/// C2. The refracted direction (Snell's law in vector form) of the unit direction `d` at a
/// surface with unit normal `n` that faces the incoming ray (`d · n < 0`), with
/// `eta = n1 / n2`. `None` means total internal reflection.
///
/// Hint: `c1 = −d · n`, `k = 1 − eta² (1 − c1²)`; `k < 0` is total internal reflection; else `t = eta d + (eta c1 − √k) n`.
pub fn refract(d: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    todo!("C2: refract")
}

/// C3. The reflected fraction `R` by Schlick's formula, for light going from index `n1` into
/// index `n2` at an angle with cosine `cos1` to the normal. `R0 = ((n1 − n2)/(n1 + n2))²` and
/// `R = R0 + (1 − R0)(1 − cos)⁵`, where `cos` is the cosine in the less dense medium. Total
/// internal reflection gives `R = 1`.
///
/// Hint: if `n1 > n2`, find `sin θ2 = (n1/n2) sin θ1` with `sin θ1 = √(1 − cos1²)`; above 1
/// return 1, else use `cos θ2 = √(1 − sin² θ2)`. `powi(5)` raises to the fifth power.
pub fn schlick(cos1: f32, n1: f32, n2: f32) -> f32 {
    todo!("C3: schlick")
}

/// C4. The shadow ray: is the point light at `light` hidden from the point `p` (unit normal
/// `n` on the side the light should be on)? The ray starts at `p + eps · n`; any hit closer
/// than the light counts (glass blocks light too, as in Whitted's model).
///
/// Hint: `scene.nearest_hit(ray)` needs a unit direction; compare the hit's `t` with the
/// distance to the light.
pub fn in_shadow(p: Vec3, n: Vec3, light: Vec3, scene: &Scene, eps: f32) -> bool {
    todo!("C4: in_shadow")
}

/// C5. Whitted ray tracing: the colour seen along `ray`. Matte surfaces: ambient + Lambert
/// with a shadow ray. Mirrors: the reflectance times `trace` of the reflected ray. Glass:
/// `R` times the reflected ray plus `1 − R` times the refracted ray (Schlick). Mirrors and
/// glass at `depth == max_depth` give black. Every ray traced, shadow rays included, adds
/// 1 to `rays`.
///
/// Hint: `scene.nearest_hit(ray)`, then `match scene.objects[hit.object].material`.
/// `object.colour_at(p)` gives a matte surface's colour; the light direction is `(scene.light − p)`,
/// normalised. Start reflected rays at `p + eps n` and refracted rays at `p − eps n`; for glass
/// `(n1, n2)` is `(1, index)` when `hit.front`, else `(index, 1)`. `colour.map(|c| c * f)`
/// scales a colour.
pub fn trace(ray: Ray, scene: &Scene, depth: usize, max_depth: usize, eps: f32, rays: &mut usize) -> [f32; 3] {
    todo!("C5: trace")
}
