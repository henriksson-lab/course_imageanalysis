// Model solution for the coding exercises of "Marching cubes".

use imgcourse::Volume;
use std::collections::HashMap;

/// Corner k of a cube sits at the offset (x, y, z) with k = x + 2y + 4z.
pub const CORNERS: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];

/// The 12 edges as (corner a, corner b): e0–e3 along x, e4–e7 along y, e8–e11 along z.
pub const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];

/// The 6 faces (x = 0, x = 1, y = 0, y = 1, z = 0, z = 1), each as its 4 corners in order round
/// the face, anticlockwise when seen from outside the cube.
pub const FACES: [[usize; 4]; 6] = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];

/// The number of the edge between corners `a` and `b` (in either order).
pub fn edge_between(a: usize, b: usize) -> usize {
    EDGES.iter().position(|&(p, q)| (p, q) == (a, b) || (p, q) == (b, a)).expect("not an edge of the cube")
}

/// How a face with diagonal corners above c (a 2D saddle) is cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FaceRule {
    /// cut off each corner above c: the two bright corners stay apart
    Separate,
    /// cut off each corner below c: the two bright corners are joined across the face
    Join,
    /// the asymptotic decider: join if the saddle value of the face is above c
    Decider,
    /// as the classic 15-case table, which treats a cube with more than 4 corners above c as the
    /// complement of a case with fewer: separate if at most 4 corners are above c, join if more.
    /// Neighbouring cubes can then cut their shared face differently: cracks.
    Complement,
}

/// A triangle mesh with shared vertices: positions (in world units), one normal per vertex, and
/// triangles as three vertex numbers, counter-clockwise seen from outside.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// R1. The case index of a cube: bit k is set if corner k is above c (value > c).
pub fn cube_case(values: [f32; 8], c: f32) -> usize {
    (0..8).filter(|&k| values[k] > c).map(|k| 1 << k).sum()
}

/// R2. The vertex on edge `e` of the unit cube, if the edge is crossed: from corner a to corner b
/// at t = (c − I_a) / (I_b − I_a), as a position (x, y, z) in [0, 1]³.
pub fn edge_vertex(values: [f32; 8], c: f32, e: usize) -> Option<[f32; 3]> {
    let (a, b) = EDGES[e];
    if (values[a] > c) == (values[b] > c) {
        return None;
    }
    let t = (c - values[a]) / (values[b] - values[a]);
    let mut p = [0.0; 3];
    for i in 0..3 {
        p[i] = CORNERS[a][i] as f32 + t * (CORNERS[b][i] as f32 - CORNERS[a][i] as f32);
    }
    Some(p)
}

/// R3. The loops of the surface in one cube, as lists of crossed edges (to be cut into triangles
/// as fans): marching squares on each of the 6 faces gives segments between crossed edges; every
/// crossed edge belongs to two faces, so the segments chain into closed loops.
/// Each face segment runs from the face edge where the values go from below to above c (going
/// round the face in the order of `FACES`) to the edge where they go from above to below. Then
/// every loop runs round its piece of surface the same way: anticlockwise seen from the dark side,
/// outside a bright object, and neighbouring cubes run along their shared face in opposite
/// directions.
pub fn cube_loops(values: [f32; 8], c: f32, rule: FaceRule) -> Vec<Vec<usize>> {
    // 1. marching squares on every face: segments as pairs of cube edges
    let mut segments: Vec<(usize, usize)> = vec![];
    for face in FACES {
        let above = face.map(|k| values[k] > c);
        // face edge i runs from face corner i to face corner i + 1
        let crossed: Vec<usize> = (0..4).filter(|&i| above[i] != above[(i + 1) % 4]).collect();
        let cube_edge = |i: usize| edge_between(face[i], face[(i + 1) % 4]);
        // a segment between face edges i and j, from the one that goes from below to above
        // (face edge i runs from face corner i to i + 1)
        let mut add = |i: usize, j: usize| {
            if above[i] { segments.push((cube_edge(j), cube_edge(i))) } else { segments.push((cube_edge(i), cube_edge(j))) }
        };
        if crossed.len() == 2 {
            add(crossed[0], crossed[1]);
        } else if crossed.len() == 4 {
            let v = face.map(|k| values[k]);
            let saddle = (v[0] * v[2] - v[1] * v[3]) / (v[0] + v[2] - v[1] - v[3]);
            let join = match rule {
                FaceRule::Separate => false,
                FaceRule::Join => true,
                FaceRule::Decider => saddle > c,
                FaceRule::Complement => values.iter().filter(|&&v| v > c).count() > 4,
            };
            // cut off the corners above (separate) or below (join); face corner i lies between
            // face edges i − 1 and i
            for i in 0..4 {
                if above[i] != join {
                    add((i + 3) % 4, i);
                }
            }
        }
    }
    // 2. chain the segments into loops: the next segment starts where this one ends
    let mut used = vec![false; segments.len()];
    let mut loops = vec![];
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
        loops.push(lp);
    }
    loops
}

/// R4. The image gradient at voxel (i, j, k) by central differences, divided by the voxel
/// `spacing`: ∂I/∂x ≈ (I[i+1] − I[i−1]) / (2 Δx), and so on. At the border of the volume, a
/// one-sided difference (I[i+1] − I[i]) / Δx or (I[i] − I[i−1]) / Δx.
pub fn gradient(vol: &Volume<f32>, spacing: [f32; 3], i: usize, j: usize, k: usize) -> [f32; 3] {
    let size = [vol.width(), vol.height(), vol.depth()];
    let p = [i, j, k];
    let mut g = [0.0; 3];
    for a in 0..3 {
        let (mut lo, mut hi) = (p, p);
        lo[a] = p[a].saturating_sub(1);
        hi[a] = (p[a] + 1).min(size[a] - 1);
        if hi[a] > lo[a] {
            g[a] = (vol[(hi[0], hi[1], hi[2])] - vol[(lo[0], lo[1], lo[2])]) / ((hi[a] - lo[a]) as f32 * spacing[a]);
        }
    }
    g
}

/// R5. Marching cubes on a whole volume: for every cube of 8 voxels its loops (R3), every loop
/// cut into a fan of triangles. A vertex is made once per grid edge and shared by all cubes
/// around that edge. Positions are in world units (`spacing`); each vertex gets the outward
/// normal −∇I / |∇I| of a bright object (R4, interpolated along its edge), for smooth shading.
pub fn marching_cubes(vol: &Volume<f32>, spacing: [f32; 3], c: f32, rule: FaceRule) -> Mesh {
    let mut mesh = Mesh::default();
    // a grid edge: its first voxel and its axis (0 = x, 1 = y, 2 = z)
    let mut vertex_of: HashMap<([usize; 3], usize), usize> = HashMap::new();
    for z in 0..vol.depth() - 1 {
        for y in 0..vol.height() - 1 {
            for x in 0..vol.width() - 1 {
                let voxel = |k: usize| [x + CORNERS[k][0], y + CORNERS[k][1], z + CORNERS[k][2]];
                let values: [f32; 8] = std::array::from_fn(|k| {
                    let v = voxel(k);
                    vol[(v[0], v[1], v[2])]
                });
                for lp in cube_loops(values, c, rule) {
                    let ids: Vec<usize> = lp
                        .iter()
                        .map(|&e| {
                            let (a, b) = EDGES[e];
                            let (va, vb) = (voxel(a), voxel(b));
                            let axis = (0..3).find(|&i| va[i] != vb[i]).unwrap();
                            *vertex_of.entry((va, axis)).or_insert_with(|| {
                                let t = (c - values[a]) / (values[b] - values[a]);
                                let (ga, gb) = (gradient(vol, spacing, va[0], va[1], va[2]), gradient(vol, spacing, vb[0], vb[1], vb[2]));
                                let mut p = [0.0; 3];
                                let mut n = [0.0; 3];
                                for i in 0..3 {
                                    p[i] = (va[i] as f32 + t * (vb[i] as f32 - va[i] as f32)) * spacing[i];
                                    n[i] = -(ga[i] + t * (gb[i] - ga[i]));
                                }
                                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                                mesh.vertices.push(p);
                                mesh.normals.push(n.map(|v| v / len));
                                mesh.vertices.len() - 1
                            })
                        })
                        .collect();
                    // the loop runs anticlockwise seen from outside (R3): so do its fan triangles
                    for i in 1..ids.len() - 1 {
                        mesh.triangles.push([ids[0], ids[i], ids[i + 1]]);
                    }
                }
            }
        }
    }
    mesh
}
