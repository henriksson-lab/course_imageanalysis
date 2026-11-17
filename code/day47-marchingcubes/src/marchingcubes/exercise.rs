// Coding exercises for "Marching cubes". Replace each `todo!()` with your code, then check it
// with `cargo test -p day47 marchingcubes::` and look at `cargo run -p day47 -- marchingcubes`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `1 << k` is 2^k.
pub fn cube_case(values: [f32; 8], c: f32) -> usize {
    todo!("R1: cube_case")
}

/// R2. The vertex on edge `e` of the unit cube, if the edge is crossed: from corner a to corner b
/// at t = (c − I_a) / (I_b − I_a), as a position (x, y, z) in [0, 1]³.
///
/// Hint: `EDGES[e]` gives the two corners, `CORNERS[k]` their offsets.
pub fn edge_vertex(values: [f32; 8], c: f32, e: usize) -> Option<[f32; 3]> {
    todo!("R2: edge_vertex")
}

/// R3. The loops of the surface in one cube, as lists of crossed edges (to be cut into triangles
/// as fans): marching squares on each of the 6 faces gives segments between crossed edges; every
/// crossed edge belongs to two faces, so the segments chain into closed loops.
/// Each face segment runs from the face edge where the values go from below to above c (going
/// round the face in the order of `FACES`) to the edge where they go from above to below. Then
/// every loop runs round its piece of surface the same way: anticlockwise seen from the dark side,
/// outside a bright object, and neighbouring cubes run along their shared face in opposite
/// directions.
///
/// Hint: for each face in `FACES`, face edge i runs from face corner i to i + 1 and is the cube
/// edge `edge_between(face[i], face[(i + 1) % 4])`. As in marching squares, 2 crossed face edges
/// give one segment; 4 give a saddle, where face corner i lies between face edges (i + 3) % 4 and
/// i, and a `match rule` decides whether to cut off the corners above c or below c. Store each
/// segment as (cube edge, cube edge) in the direction described above. Then chain: start a loop
/// with an unused segment (a, b); repeatedly take the unused segment that starts at the current
/// edge and step to its end, until you are back at a.
pub fn cube_loops(values: [f32; 8], c: f32, rule: FaceRule) -> Vec<Vec<usize>> {
    todo!("R3: cube_loops")
}

/// R4. The image gradient at voxel (i, j, k) by central differences, divided by the voxel
/// `spacing`: ∂I/∂x ≈ (I[i+1] − I[i−1]) / (2 Δx), and so on. At the border of the volume, a
/// one-sided difference (I[i+1] − I[i]) / Δx or (I[i] − I[i−1]) / Δx.
///
/// Hint: per axis a, the neighbours one voxel lower and higher, but not outside the volume
/// (`saturating_sub(1)`, `.min(size - 1)`); divide by their distance in world units.
pub fn gradient(vol: &Volume<f32>, spacing: [f32; 3], i: usize, j: usize, k: usize) -> [f32; 3] {
    todo!("R4: gradient")
}

/// R5. Marching cubes on a whole volume: for every cube of 8 voxels its loops (R3), every loop
/// cut into a fan of triangles. A vertex is made once per grid edge and shared by all cubes
/// around that edge. Positions are in world units (`spacing`); each vertex gets the outward
/// normal −∇I / |∇I| of a bright object (R4, interpolated along its edge), for smooth shading.
///
/// Hint: loop over the cubes (x, y, z) up to width − 1, height − 1, depth − 1. Keep a
/// `HashMap<([usize; 3], usize), usize>` from (first voxel of a grid edge, its axis) to the vertex
/// number, so that each vertex is made only once (`entry(...).or_insert_with(...)`). A new vertex
/// on the edge from voxel a to voxel b: position (a + t (b − a)) · spacing, normal −(∇a + t (∇b − ∇a)),
/// made of length 1. A loop of m vertices gives the fan (v0, v1, v2), (v0, v2, v3), ...
pub fn marching_cubes(vol: &Volume<f32>, spacing: [f32; 3], c: f32, rule: FaceRule) -> Mesh {
    todo!("R5: marching_cubes")
}
