// Coding exercises for "Measuring surfaces". Replace each `todo!()` with your code, then check it
// with `cargo test -p marchingcubes surface_measure::` and look at `cargo run -p marchingcubes -- surface-measure`.
#![allow(unused_variables, unused_imports)]

use crate::surface_measure::Mesh;
use imgcourse::Volume;
use std::collections::HashMap;

/// C1a. The area of the triangle (a, b, c): half the length of (b − a) × (c − a).
///
/// Hint: the cross product of u and w is (u₁w₂ − u₂w₁, u₂w₀ − u₀w₂, u₀w₁ − u₁w₀).
pub fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    todo!("C1: triangle_area")
}

/// C1b. The area of a mesh: the sum of its triangle areas.
///
/// Hint: `mesh.triangles` holds three vertex numbers per triangle, `mesh.vertices[i]` the positions.
pub fn mesh_area(mesh: &Mesh) -> f32 {
    todo!("C1: mesh_area")
}

/// C2. The volume inside a closed mesh whose triangles run anticlockwise seen from outside:
/// the sum of the signed tetrahedra (origin, a, b, c), each ⅙ a · (b × c).
pub fn mesh_volume(mesh: &Mesh) -> f32 {
    todo!("C2: mesh_volume")
}

/// C3. The voxel estimates of a mask with voxel `spacing`: (volume, area) = (number of voxels
/// inside × voxel volume, number of exposed voxel faces × their area). A face is exposed when the
/// neighbour across it is outside the mask or outside the volume; a face across x has the area
/// Δy Δz, and so on.
///
/// Hint: for every voxel inside, look at its 6 face neighbours (one step along each axis, both
/// ways); `mask.contains(x, y, z)` takes signed numbers.
pub fn voxel_estimates(mask: &Volume<bool>, spacing: [f32; 3]) -> (f32, f32) {
    todo!("C3: voxel_estimates")
}

/// C4. The topology of a mesh: (V, E, F, boundary edges, χ = V − E + F). E counts every edge once,
/// however many triangles share it; a boundary edge belongs to only one triangle.
///
/// Hint: a `HashMap<(usize, usize), usize>` counting the triangles of every edge, stored with the
/// smaller vertex number first so that both directions count as the same edge.
pub fn topology(mesh: &Mesh) -> (usize, usize, usize, usize, i64) {
    todo!("C4: topology")
}
