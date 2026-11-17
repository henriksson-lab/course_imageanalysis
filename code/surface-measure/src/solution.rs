// Model solution for the coding exercises of "Measuring surfaces".

use crate::Mesh;
use imgcourse::Volume;
use std::collections::HashMap;

/// C1a. The area of the triangle (a, b, c): half the length of (b − a) × (c − a).
pub fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let n = cross(sub(b, a), sub(c, a));
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

/// C1b. The area of a mesh: the sum of its triangle areas.
pub fn mesh_area(mesh: &Mesh) -> f32 {
    mesh.triangles.iter().map(|t| triangle_area(mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]])).sum()
}

/// C2. The volume inside a closed mesh whose triangles run anticlockwise seen from outside:
/// the sum of the signed tetrahedra (origin, a, b, c), each ⅙ a · (b × c).
pub fn mesh_volume(mesh: &Mesh) -> f32 {
    let mut v = 0.0;
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| mesh.vertices[i]);
        let bc = cross(b, c);
        v += (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]) / 6.0;
    }
    v
}

/// C3. The voxel estimates of a mask with voxel `spacing`: (volume, area) = (number of voxels
/// inside × voxel volume, number of exposed voxel faces × their area). A face is exposed when the
/// neighbour across it is outside the mask or outside the volume; a face across x has the area
/// Δy Δz, and so on.
pub fn voxel_estimates(mask: &Volume<bool>, spacing: [f32; 3]) -> (f32, f32) {
    let face_area = [spacing[1] * spacing[2], spacing[0] * spacing[2], spacing[0] * spacing[1]];
    let (mut voxels, mut area) = (0.0, 0.0);
    for (x, y, z) in mask.coords() {
        if !mask[(x, y, z)] {
            continue;
        }
        voxels += 1.0;
        for axis in 0..3 {
            for step in [-1isize, 1] {
                let mut n = [x as isize, y as isize, z as isize];
                n[axis] += step;
                let inside = mask.contains(n[0], n[1], n[2]) && mask[(n[0] as usize, n[1] as usize, n[2] as usize)];
                if !inside {
                    area += face_area[axis];
                }
            }
        }
    }
    (voxels * spacing[0] * spacing[1] * spacing[2], area)
}

/// C4. The topology of a mesh: (V, E, F, boundary edges, χ = V − E + F). E counts every edge once,
/// however many triangles share it; a boundary edge belongs to only one triangle.
pub fn topology(mesh: &Mesh) -> (usize, usize, usize, usize, i64) {
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &mesh.triangles {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let boundary = edges.values().filter(|&&n| n == 1).count();
    let (v, e, f) = (mesh.vertices.len(), edges.len(), mesh.triangles.len());
    (v, e, f, boundary, v as i64 - e as i64 + f as i64)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
