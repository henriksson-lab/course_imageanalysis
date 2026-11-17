// Surfaces to measure: the level set {I = c} of a volume as a triangle mesh, by marching cubes.
// This is not an exercise here (it is the previous unit's); it is a short version of it that
// cuts ambiguous faces with the "separate" rule, so that the surface is always closed.

use imgcourse::Volume;
use std::collections::HashMap;

/// A triangle mesh with shared vertices; triangles run anticlockwise seen from outside.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// The faces of a cube (corner k at offset x + 2y + 4z), anticlockwise seen from outside.
const FACES: [[usize; 4]; 6] = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];

/// The surface {I = c} of a bright object (inside: I > c), in world units (`spacing`). Pad the
/// volume with background first if the object touches its border, or the surface stays open.
pub fn surface(vol: &Volume<f32>, spacing: [f32; 3], c: f32) -> Mesh {
    let mut mesh = Mesh::default();
    let mut vertex_of: HashMap<([usize; 3], [usize; 3]), usize> = HashMap::new();
    for z in 0..vol.depth() - 1 {
        for y in 0..vol.height() - 1 {
            for x in 0..vol.width() - 1 {
                let voxel = |k: usize| [x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1)];
                let val: [f32; 8] = std::array::from_fn(|k| {
                    let v = voxel(k);
                    vol[(v[0], v[1], v[2])]
                });
                // marching squares on the faces: segments from the edge where the values rise
                // above c to the edge where they fall below (an edge = its two corners)
                let mut segments: Vec<((usize, usize), (usize, usize))> = vec![];
                for f in FACES {
                    let above = f.map(|k| val[k] > c);
                    let edge = |i: usize| (f[i].min(f[(i + 1) % 4]), f[i].max(f[(i + 1) % 4]));
                    let crossed: Vec<usize> = (0..4).filter(|&i| above[i] != above[(i + 1) % 4]).collect();
                    let pairs: Vec<(usize, usize)> = if crossed.len() == 2 {
                        vec![(crossed[0], crossed[1])]
                    } else {
                        (0..4).filter(|&i| crossed.len() == 4 && above[i]).map(|i| ((i + 3) % 4, i)).collect()
                    };
                    for (i, j) in pairs {
                        segments.push(if above[i] { (edge(j), edge(i)) } else { (edge(i), edge(j)) });
                    }
                }
                // chain into loops, and cut every loop into a fan of triangles
                let mut used = vec![false; segments.len()];
                for s in 0..segments.len() {
                    if used[s] {
                        continue;
                    }
                    used[s] = true;
                    let (start, mut current) = segments[s];
                    let mut lp = vec![start];
                    while current != start {
                        lp.push(current);
                        let Some(i) = (0..segments.len()).find(|&i| !used[i] && segments[i].0 == current) else { break };
                        used[i] = true;
                        current = segments[i].1;
                    }
                    let ids: Vec<usize> = lp
                        .iter()
                        .map(|&(a, b)| {
                            let (va, vb) = (voxel(a), voxel(b));
                            *vertex_of.entry((va, vb)).or_insert_with(|| {
                                let t = (c - val[a]) / (val[b] - val[a]);
                                mesh.vertices.push(std::array::from_fn(|i| (va[i] as f32 + t * (vb[i] as f32 - va[i] as f32)) * spacing[i]));
                                mesh.vertices.len() - 1
                            })
                        })
                        .collect();
                    for i in 1..ids.len() - 1 {
                        mesh.triangles.push([ids[0], ids[i], ids[i + 1]]);
                    }
                }
            }
        }
    }
    mesh
}
