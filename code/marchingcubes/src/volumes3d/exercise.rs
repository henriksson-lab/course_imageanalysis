// Coding exercises for "3D images". Replace each `todo!()` with your code, then check it with
// `cargo test -p marchingcubes volumes3d::` and look at `cargo run -p marchingcubes -- volumes3d`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Volume};

/// Where a volume sits in the world: the distance between neighbouring voxel centres along
/// x, y and z (µm or mm), and the world position of voxel (0, 0, 0). (No rotation: R = I.)
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub spacing: [f32; 3],
    pub origin: [f32; 3],
}

/// C1a. Voxel position (i, j, k) (may be fractional) → world position:
/// x = o_x + i Δx, y = o_y + j Δy, z = o_z + k Δz (the voxel → world matrix A).
pub fn voxel_to_world(grid: &Grid, v: [f32; 3]) -> [f32; 3] {
    todo!("C1: voxel_to_world")
}

/// C1b. World position → voxel position: i = (x − o_x) / Δx, and so on (A⁻¹).
pub fn world_to_voxel(grid: &Grid, p: [f32; 3]) -> [f32; 3] {
    todo!("C1: world_to_voxel")
}

/// C2. Trilinear interpolation at the voxel position `v` = (x, y, z): interpolate along x
/// (four pairs), then along y (two pairs), then along z (one pair). Outside the volume
/// (below 0 or beyond the last voxel along any axis) the value is 0.
///
/// Hint: the cube's lower corner is (x.floor(), y.floor(), z.floor()); the upper one is one
/// further, but not beyond the last voxel (`.min(width - 1)`), so that a position exactly on the
/// last voxel still works. The fractions are fx = x − x0, and so on.
pub fn trilinear(vol: &Volume<f32>, v: [f32; 3]) -> f32 {
    todo!("C2: trilinear")
}

/// C3. An oblique slice of `width` × `height` pixels of size `pixel` (world units), in the plane
/// through the world point `centre` spanned by the unit directions `u` (across) and `w` (down).
/// Output pixel (px, py) is the world point centre + pixel·(p u + q w) with p = px − (width − 1)/2,
/// q = py − (height − 1)/2: inverse mapping, world → voxel, then trilinear interpolation.
///
/// Hint: `Image::from_fn(width, height, |px, py| ...)`, with `world_to_voxel` and `trilinear`.
#[allow(clippy::too_many_arguments)]
pub fn oblique_slice(vol: &Volume<f32>, grid: &Grid, centre: [f32; 3], u: [f32; 3], w: [f32; 3], pixel: f32, width: usize, height: usize) -> Image<f32> {
    todo!("C3: oblique_slice")
}

/// C4. The maximum-intensity projection along z: for every (x, y), the largest value over all
/// slices.
pub fn mip_z(vol: &Volume<f32>) -> Image<f32> {
    todo!("C4: mip_z")
}

/// C5. Connected components of a 3D mask: every voxel that is `true` gets the number of its
/// object (1, 2, ...), background 0. Two voxels are neighbours if they share a face
/// (`connectivity` 6) or also an edge or a corner (26). Returns the labels and the number of
/// objects. (A volume of depth 1 is a 2D image; 26 then means 8-connected.)
///
/// Hint: list the neighbour offsets (dx, dy, dz) first: 6 with one non-zero step, or all 26.
/// Then for every unlabelled `true` voxel, start a new object and flood fill it with a stack
/// (`Vec`): pop a voxel, label and push its unlabelled `true` neighbours. `mask.contains(x, y, z)`
/// takes signed numbers.
pub fn label_3d(mask: &Volume<bool>, connectivity: usize) -> (Volume<u32>, u32) {
    todo!("C5: label_3d")
}
