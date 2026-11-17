// Model solution for the coding exercises of "3D images".

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
    let mut p = [0.0; 3];
    for a in 0..3 {
        p[a] = grid.origin[a] + v[a] * grid.spacing[a];
    }
    p
}

/// C1b. World position → voxel position: i = (x − o_x) / Δx, and so on (A⁻¹).
pub fn world_to_voxel(grid: &Grid, p: [f32; 3]) -> [f32; 3] {
    let mut v = [0.0; 3];
    for a in 0..3 {
        v[a] = (p[a] - grid.origin[a]) / grid.spacing[a];
    }
    v
}

/// C2. Trilinear interpolation at the voxel position `v` = (x, y, z): interpolate along x
/// (four pairs), then along y (two pairs), then along z (one pair). Outside the volume
/// (below 0 or beyond the last voxel along any axis) the value is 0.
pub fn trilinear(vol: &Volume<f32>, v: [f32; 3]) -> f32 {
    let size = [vol.width(), vol.height(), vol.depth()];
    for a in 0..3 {
        if v[a] < 0.0 || v[a] > (size[a] - 1) as f32 {
            return 0.0;
        }
    }
    let (x0, y0, z0) = (v[0].floor() as usize, v[1].floor() as usize, v[2].floor() as usize);
    let (x1, y1, z1) = ((x0 + 1).min(size[0] - 1), (y0 + 1).min(size[1] - 1), (z0 + 1).min(size[2] - 1));
    let (fx, fy, fz) = (v[0] - x0 as f32, v[1] - y0 as f32, v[2] - z0 as f32);
    let lerp = |a: f32, b: f32, f: f32| a + f * (b - a);
    // along x: four pairs
    let c00 = lerp(vol[(x0, y0, z0)], vol[(x1, y0, z0)], fx);
    let c10 = lerp(vol[(x0, y1, z0)], vol[(x1, y1, z0)], fx);
    let c01 = lerp(vol[(x0, y0, z1)], vol[(x1, y0, z1)], fx);
    let c11 = lerp(vol[(x0, y1, z1)], vol[(x1, y1, z1)], fx);
    // along y: two pairs; along z: one pair
    lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
}

/// C3. An oblique slice of `width` × `height` pixels of size `pixel` (world units), in the plane
/// through the world point `centre` spanned by the unit directions `u` (across) and `w` (down).
/// Output pixel (px, py) is the world point centre + pixel·(p u + q w) with p = px − (width − 1)/2,
/// q = py − (height − 1)/2: inverse mapping, world → voxel, then trilinear interpolation.
#[allow(clippy::too_many_arguments)]
pub fn oblique_slice(vol: &Volume<f32>, grid: &Grid, centre: [f32; 3], u: [f32; 3], w: [f32; 3], pixel: f32, width: usize, height: usize) -> Image<f32> {
    Image::from_fn(width, height, |px, py| {
        let p = px as f32 - (width - 1) as f32 / 2.0;
        let q = py as f32 - (height - 1) as f32 / 2.0;
        let mut world = [0.0; 3];
        for a in 0..3 {
            world[a] = centre[a] + pixel * (p * u[a] + q * w[a]);
        }
        trilinear(vol, world_to_voxel(grid, world))
    })
}

/// C4. The maximum-intensity projection along z: for every (x, y), the largest value over all
/// slices.
pub fn mip_z(vol: &Volume<f32>) -> Image<f32> {
    Image::from_fn(vol.width(), vol.height(), |x, y| {
        let mut m = f32::NEG_INFINITY;
        for z in 0..vol.depth() {
            m = m.max(vol[(x, y, z)]);
        }
        m
    })
}

/// C5. Connected components of a 3D mask: every voxel that is `true` gets the number of its
/// object (1, 2, ...), background 0. Two voxels are neighbours if they share a face
/// (`connectivity` 6) or also an edge or a corner (26). Returns the labels and the number of
/// objects. (A volume of depth 1 is a 2D image; 26 then means 8-connected.)
pub fn label_3d(mask: &Volume<bool>, connectivity: usize) -> (Volume<u32>, u32) {
    let mut offsets = vec![];
    for dz in -1..=1isize {
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let steps = dx.abs() + dy.abs() + dz.abs();
                if steps == 1 || (connectivity == 26 && steps > 1) {
                    offsets.push((dx, dy, dz));
                }
            }
        }
    }
    let mut labels = Volume::new(mask.width(), mask.height(), mask.depth(), 0u32);
    let mut count = 0;
    for (x, y, z) in mask.coords() {
        if !mask[(x, y, z)] || labels[(x, y, z)] != 0 {
            continue;
        }
        // a new object: flood fill from here with a stack of voxels still to visit
        count += 1;
        labels[(x, y, z)] = count;
        let mut stack = vec![(x, y, z)];
        while let Some((cx, cy, cz)) = stack.pop() {
            for &(dx, dy, dz) in &offsets {
                let (nx, ny, nz) = (cx as isize + dx, cy as isize + dy, cz as isize + dz);
                if mask.contains(nx, ny, nz) {
                    let n = (nx as usize, ny as usize, nz as usize);
                    if mask[n] && labels[n] == 0 {
                        labels[n] = count;
                        stack.push(n);
                    }
                }
            }
        }
    }
    (labels, count)
}
