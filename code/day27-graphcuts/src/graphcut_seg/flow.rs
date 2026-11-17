// Provided, not an exercise: maximum flow on large sparse graphs (one node per pixel).
// The maxflow unit's exercise keeps the whole capacity matrix, which is fine for ten nodes
// but not for 16 384 pixels. This version keeps a list of edges per node and uses Dinic's
// algorithm (shortest augmenting paths, many at a time), which is fast on image graphs.

/// A directed graph with capacities, stored as edge lists. Every edge has a partner edge in
/// the opposite direction, which holds the residual capacity for undoing flow.
pub struct FlowGraph {
    edges_of: Vec<Vec<usize>>, // edge numbers leaving each node
    to: Vec<usize>,
    room: Vec<f64>, // residual capacity of each edge
}

impl FlowGraph {
    /// A graph with `n` nodes and no edges.
    pub fn new(n: usize) -> Self {
        FlowGraph { edges_of: vec![Vec::new(); n], to: Vec::new(), room: Vec::new() }
    }

    pub fn nodes(&self) -> usize {
        self.edges_of.len()
    }

    /// Add the edge u → v with capacity `cap` (may be `f32::INFINITY`).
    pub fn add_edge(&mut self, u: usize, v: usize, cap: f32) {
        // edge 2k is u → v, edge 2k + 1 its partner v → u (capacity 0)
        self.edges_of[u].push(self.to.len());
        self.to.push(v);
        self.room.push(cap as f64);
        self.edges_of[v].push(self.to.len());
        self.to.push(u);
        self.room.push(0.0);
    }

    /// The maximum flow from `s` to `t` (Dinic's algorithm). Changes the residual
    /// capacities; afterwards `source_side` gives the minimum cut.
    pub fn max_flow(&mut self, s: usize, t: usize) -> f32 {
        let mut total = 0.0;
        loop {
            let level = self.levels(s);
            if level[t] == usize::MAX {
                return total as f32;
            }
            let mut next_edge = vec![0usize; self.nodes()];
            loop {
                let pushed = self.push(s, t, f64::INFINITY, &level, &mut next_edge);
                if pushed <= 0.0 {
                    break;
                }
                total += pushed;
            }
        }
    }

    /// Breadth-first distances from `s` along edges with room (usize::MAX: not reached).
    fn levels(&self, s: usize) -> Vec<usize> {
        let mut level = vec![usize::MAX; self.nodes()];
        let mut queue = std::collections::VecDeque::from([s]);
        level[s] = 0;
        while let Some(u) = queue.pop_front() {
            for &e in &self.edges_of[u] {
                let v = self.to[e];
                if self.room[e] > 1e-9 && level[v] == usize::MAX {
                    level[v] = level[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        level
    }

    /// Push up to `limit` from `u` to `t` along edges that go one level further (iterative
    /// depth-first search). Returns how much was pushed.
    fn push(&mut self, s: usize, t: usize, limit: f64, level: &[usize], next_edge: &mut [usize]) -> f64 {
        let mut path: Vec<usize> = Vec::new(); // edge numbers from s
        let mut u = s;
        loop {
            if u == t {
                let b = path.iter().map(|&e| self.room[e]).fold(limit, f64::min);
                for &e in &path {
                    self.room[e] -= b;
                    self.room[e ^ 1] += b;
                }
                return b;
            }
            let mut advanced = false;
            while next_edge[u] < self.edges_of[u].len() {
                let e = self.edges_of[u][next_edge[u]];
                let v = self.to[e];
                if self.room[e] > 1e-9 && level[v] == level[u] + 1 {
                    path.push(e);
                    u = v;
                    advanced = true;
                    break;
                }
                next_edge[u] += 1;
            }
            if !advanced {
                // dead end: retreat one step and skip the edge that led here
                match path.pop() {
                    Some(e) => {
                        u = self.to[e ^ 1];
                        next_edge[u] += 1;
                    }
                    None => return 0.0,
                }
            }
        }
    }

    /// After `max_flow`: the nodes that can still be reached from `s` along edges with
    /// room. They form the source side of a minimum cut.
    pub fn source_side(&self, s: usize) -> Vec<bool> {
        let level = self.levels(s);
        level.iter().map(|&l| l != usize::MAX).collect()
    }
}
