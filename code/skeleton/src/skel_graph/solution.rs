// Model solution for the coding exercises of "From skeleton to graph: pruning and length".

use imgcourse::{Border, Image};

/// The 8 neighbours of a pixel, as offsets `(dx, dy)`.
pub const N8: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// A node of the skeleton graph: an end point, a cluster of touching junction pixels, an
/// isolated pixel, or (for a closed loop without any node) one pixel of the loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub pixels: Vec<(usize, usize)>,
}

/// An edge: it joins node `a` to node `b` (`a == b` for a loop) along `path`, the pixels in
/// order from a pixel of node `a` to a pixel of node `b`.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub path: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// The step counts of a path: axial steps `ne`, diagonal steps `no`, corners `nc` (changes
/// of step direction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steps {
    pub ne: usize,
    pub no: usize,
    pub nc: usize,
}

impl Steps {
    pub fn pixel_steps(&self) -> f32 {
        (self.ne + self.no) as f32
    }
    pub fn weighted(&self) -> f32 {
        self.ne as f32 + std::f32::consts::SQRT_2 * self.no as f32
    }
    /// The corrected estimator of Vossepoel and Smeulders.
    pub fn corrected(&self) -> f32 {
        0.980 * self.ne as f32 + 1.406 * self.no as f32 - 0.091 * self.nc as f32
    }
}

/// C1. The number of skeleton neighbours (8-neighbourhood) of every skeleton pixel; 0 off
/// the skeleton. 1 = end point, 2 = regular, 3 or more = junction.
pub fn neighbour_count(skel: &Image<bool>) -> Image<u8> {
    skel.map_xy(|x, y, s| {
        if !s {
            return 0;
        }
        N8.iter().filter(|&&(dx, dy)| skel.get_border(x as isize + dx, y as isize + dy, Border::Zero)).count() as u8
    })
}

/// C2. Step counts of a path of 8-connected pixels.
pub fn path_steps(path: &[(usize, usize)]) -> Steps {
    let mut s = Steps { ne: 0, no: 0, nc: 0 };
    let mut previous = None;
    for w in path.windows(2) {
        let step = (w[1].0 as isize - w[0].0 as isize, w[1].1 as isize - w[0].1 as isize);
        if step.0 == 0 || step.1 == 0 { s.ne += 1 } else { s.no += 1 }
        if previous.is_some_and(|p| p != step) {
            s.nc += 1;
        }
        previous = Some(step);
    }
    s
}

/// C3. Build the skeleton graph.
///
/// 1. Node pixels are the skeleton pixels with a neighbour count other than 2. Touching
///    junction pixels (count ≥ 3, 8-connected) form one node; every other node pixel is a
///    node on its own.
/// 2. Edges: from every node pixel, step to each neighbouring skeleton pixel. If that is a
///    pixel of another node, the two nodes are joined directly (one edge per pair of nodes).
///    If it is a regular pixel not yet visited, walk along the regular pixels (each has
///    exactly one way on) until a node pixel is reached. (A regular pixel whose two
///    neighbours both belong to the starting node is part of that node: no edge.)
/// 3. Regular pixels still not visited lie on closed loops without nodes: make one of
///    their pixels a node and walk round the loop back to it.
pub fn build_graph(skel: &Image<bool>) -> Graph {
    let count = neighbour_count(skel);
    let is_regular = |p: (usize, usize)| skel[p] && count[p] == 2;
    let mut node_of: Image<Option<usize>> = Image::new(skel.width(), skel.height(), None);
    let mut nodes: Vec<Node> = Vec::new();
    for p in skel.coords() {
        if !skel[p] || is_regular(p) || node_of[p].is_some() {
            continue;
        }
        // a new node: flood-fill touching junction pixels; an end point stays on its own
        let id = nodes.len();
        let mut pixels = vec![p];
        node_of[p] = Some(id);
        let mut i = 0;
        while i < pixels.len() && count[p] >= 3 {
            let (x, y) = pixels[i];
            for q in neighbours_on(skel, x, y) {
                if count[q] >= 3 && node_of[q].is_none() {
                    node_of[q] = Some(id);
                    pixels.push(q);
                }
            }
            i += 1;
        }
        nodes.push(Node { pixels });
    }

    let mut visited = skel.map(|_| false);
    let mut edges: Vec<Edge> = Vec::new();
    for a in 0..nodes.len() {
        for &p in &nodes[a].pixels.clone() {
            for q in neighbours_on(skel, p.0, p.1) {
                if let Some(b) = node_of[q] {
                    // two nodes side by side: one edge per pair
                    if b != a && !edges.iter().any(|e| e.path.len() == 2 && ((e.a, e.b) == (a, b) || (e.a, e.b) == (b, a))) {
                        edges.push(Edge { a, b, path: vec![p, q] });
                    }
                } else if !visited[q] {
                    visited[q] = true;
                    // a regular pixel wedged between two pixels of this node belongs to the node
                    if neighbours_on(skel, q.0, q.1).iter().all(|&r| node_of[r] == Some(a)) {
                        continue;
                    }
                    let (path, b) = walk(skel, &node_of, &mut visited, p, q);
                    edges.push(Edge { a, b, path });
                }
            }
        }
    }
    // closed loops without nodes
    for p in skel.coords() {
        if is_regular(p) && !visited[p] {
            let id = nodes.len();
            nodes.push(Node { pixels: vec![p] });
            node_of[p] = Some(id);
            visited[p] = true;
            let q = neighbours_on(skel, p.0, p.1)[0];
            let (path, b) = walk(skel, &node_of, &mut visited, p, q);
            edges.push(Edge { a: id, b, path });
        }
    }
    Graph { nodes, edges }
}

/// The skeleton pixels among the 8 neighbours of `(x, y)`.
pub fn neighbours_on(skel: &Image<bool>, x: usize, y: usize) -> Vec<(usize, usize)> {
    N8.iter()
        .map(|&(dx, dy)| (x as isize + dx, y as isize + dy))
        .filter(|&(nx, ny)| skel.contains(nx, ny) && skel[(nx as usize, ny as usize)])
        .map(|(nx, ny)| (nx as usize, ny as usize))
        .collect()
}

/// Walk from node pixel `start` through the regular pixel `first` along the skeleton until a
/// node pixel is reached; returns the path and that node. Marks the regular pixels visited.
fn walk(skel: &Image<bool>, node_of: &Image<Option<usize>>, visited: &mut Image<bool>, start: (usize, usize), first: (usize, usize)) -> (Vec<(usize, usize)>, usize) {
    let mut path = vec![start, first];
    visited[first] = true;
    let (mut previous, mut current) = (start, first);
    loop {
        // a regular pixel has two skeleton neighbours: go to the one we did not come from
        let next = neighbours_on(skel, current.0, current.1).into_iter().find(|&q| q != previous).unwrap();
        path.push(next);
        if let Some(b) = node_of[next] {
            return (path, b);
        }
        visited[next] = true;
        (previous, current) = (current, next);
    }
}

/// C4. The cyclomatic number E − V + C: edges minus nodes plus connected pieces of the
/// graph (= the number of independent cycles = holes).
pub fn cycles(g: &Graph) -> isize {
    // connected pieces with "union–find": every node points towards a representative
    let mut parent: Vec<usize> = (0..g.nodes.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    for e in &g.edges {
        let (ra, rb) = (root(&mut parent, e.a), root(&mut parent, e.b));
        parent[ra] = rb;
    }
    let pieces = (0..g.nodes.len()).filter(|&i| root(&mut parent, i) == i).count();
    g.edges.len() as isize - g.nodes.len() as isize + pieces as isize
}

/// C5. Prune spurs: in two rounds, build the graph, find the terminal edges (one end a
/// single-pixel node with one edge, the other end a node with three or more edges) whose
/// weighted length is below `k`, and delete their pixels except the junction's. At each
/// junction, delete the shortest ones first, and stop when only two of its edges are left.
pub fn prune(skel: &Image<bool>, k: f32) -> Image<bool> {
    let mut out = skel.clone();
    for _round in 0..2 {
        let g = build_graph(&out);
        let mut degree = vec![0; g.nodes.len()];
        for e in &g.edges {
            degree[e.a] += 1;
            degree[e.b] += 1;
        }
        let is_tip = |n: usize| degree[n] == 1;
        for (j, &dj) in degree.iter().enumerate() {
            if dj < 3 {
                continue;
            }
            // the short terminal edges at junction j, shortest first
            let mut spurs: Vec<(f32, &Edge)> = g
                .edges
                .iter()
                .filter(|e| (e.a == j && is_tip(e.b)) || (e.b == j && is_tip(e.a)))
                .map(|e| (path_steps(&e.path).weighted(), e))
                .filter(|&(len, _)| len < k)
                .collect();
            spurs.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, e) in spurs.into_iter().take(dj - 2) {
                for &p in &e.path {
                    if !g.nodes[j].pixels.contains(&p) {
                        out[p] = false;
                    }
                }
            }
        }
    }
    out
}
