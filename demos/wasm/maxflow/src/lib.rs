//! Maximum flow / minimum s-t cut for image-sized graphs, as WebAssembly for the graph-cut demo.
//!
//! Same API and results as `demos/src/lib/maxflow.ts` (`FlowGraph`: `add_edge`, `maxflow`,
//! `source_side`, `flow`): edges are stored in pairs (e, e ^ 1), an undirected edge is one pair
//! with capacity c both ways, residuals below `EPS` count as 0, and `source_side` returns the
//! nodes reachable from s in the residual graph (the smallest minimum cut, which is the same for
//! every maximum flow). Only the algorithm differs: Boykov-Kolmogorov (two search trees that are
//! kept and repaired after each augmentation; Boykov & Kolmogorov, IEEE PAMI 2004) instead of
//! Dinic. It is much faster on grid graphs, where Dinic needs many phases of long paths.
//!
//! Inside, the graph is a compressed adjacency list (the arcs of a node are contiguous) of the
//! non-terminal edges, and the edges to s and t become one signed terminal capacity per node.
//!
//! `GridCut` is `solveSeg()` of `demos/src/lib/graph.ts` (graph-cut segmentation with seeds)
//! with the graph built once per grid and neighbourhood: a new solve only rewrites capacities.
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

pub const EPS: f64 = 1e-9;

const NONE: u32 = u32::MAX; // not in a tree (free)
const TERMINAL: u32 = u32::MAX - 1; // attached to s or t directly
const ORPHAN: u32 = u32::MAX - 2; // lost its parent arc during an augmentation
const INF_D: u32 = u32::MAX;

/// The Boykov-Kolmogorov solver on a compressed adjacency list. Node i has the arcs
/// first[i] .. first[i + 1]; head[a] is the node arc a points to, sister[a] the reverse arc.
/// tr[i] > 0: residual capacity s -> i; tr[i] < 0: residual capacity i -> t.
#[derive(Clone, Copy)]
struct Arc {
    head: u32,
    sister: u32,
    rcap: f64,
}

#[derive(Clone, Copy)]
struct Node {
    tr: f64,
    parent: u32, // arc from the node to its parent, or NONE / TERMINAL / ORPHAN
    ts: u32,
    dist: u32,
    in_sink: bool,
    in_active: bool,
}

struct Core {
    first: Vec<u32>,
    arcs: Vec<Arc>,
    nodes: Vec<Node>,
    active: VecDeque<u32>,
    orphans: VecDeque<u32>,
}

impl Core {
    /// Build the adjacency of `n` nodes from the pairs (pu[k], pv[k]). Returns the core and, for
    /// each pair, the arc pu -> pv (its sister is pv -> pu).
    fn new(n: usize, pu: &[u32], pv: &[u32]) -> (Core, Vec<u32>) {
        let mut first = vec![0u32; n + 1];
        for k in 0..pu.len() {
            first[pu[k] as usize + 1] += 1;
            first[pv[k] as usize + 1] += 1;
        }
        for i in 0..n {
            first[i + 1] += first[i];
        }
        let m = first[n] as usize;
        let mut fill: Vec<u32> = first[..n].to_vec();
        let mut arcs = vec![Arc { head: 0, sister: 0, rcap: 0.0 }; m];
        let mut arc = vec![0u32; pu.len()];
        for k in 0..pu.len() {
            let (u, v) = (pu[k] as usize, pv[k] as usize);
            let a = fill[u];
            fill[u] += 1;
            let b = fill[v];
            fill[v] += 1;
            arcs[a as usize].head = v as u32;
            arcs[b as usize].head = u as u32;
            arcs[a as usize].sister = b;
            arcs[b as usize].sister = a;
            arc[k] = a;
        }
        let core = Core {
            first,
            arcs,
            nodes: vec![Node { tr: 0.0, parent: NONE, ts: 0, dist: 0, in_sink: false, in_active: false }; n],
            active: VecDeque::new(),
            orphans: VecDeque::new(),
        };
        (core, arc)
    }

    fn n(&self) -> usize {
        self.nodes.len()
    }

    fn activate(&mut self, i: usize) {
        if !self.nodes[i].in_active {
            self.nodes[i].in_active = true;
            self.active.push_back(i as u32);
        }
    }

    fn orphan(&mut self, i: usize) {
        self.nodes[i].parent = ORPHAN;
        self.orphans.push_back(i as u32);
    }

    /// Maximum flow through the terminal capacities tr and the arcs; afterwards rcap and tr
    /// hold the residual graph.
    fn maxflow(&mut self) -> f64 {
        let n = self.n();
        self.active.clear();
        self.orphans.clear();
        for i in 0..n {
            self.nodes[i].in_active = false;
            self.nodes[i].ts = 0;
            self.nodes[i].dist = 1;
            if self.nodes[i].tr > EPS || self.nodes[i].tr < -EPS {
                self.nodes[i].parent = TERMINAL;
                self.nodes[i].in_sink = self.nodes[i].tr < 0.0;
                self.activate(i);
            } else {
                self.nodes[i].parent = NONE;
                self.nodes[i].dist = 0;
            }
        }
        let mut time: u32 = 0;
        let mut total = 0.0;
        while let Some(i) = self.active.pop_front() {
            let i = i as usize;
            self.nodes[i].in_active = false;
            if self.nodes[i].parent == NONE {
                continue;
            }
            // growth
            let sink = self.nodes[i].in_sink;
            let mut found = NONE;
            for a in self.first[i] as usize..self.first[i + 1] as usize {
                let r = if sink { self.arcs[self.arcs[a].sister as usize].rcap } else { self.arcs[a].rcap };
                if r <= EPS {
                    continue;
                }
                let j = self.arcs[a].head as usize;
                if self.nodes[j].parent == NONE {
                    self.nodes[j].in_sink = sink;
                    self.nodes[j].parent = self.arcs[a].sister;
                    self.nodes[j].ts = self.nodes[i].ts;
                    self.nodes[j].dist = self.nodes[i].dist + 1;
                    self.activate(j);
                } else if self.nodes[j].in_sink != sink {
                    found = if sink { self.arcs[a].sister } else { a as u32 };
                    break;
                } else if self.nodes[j].ts <= self.nodes[i].ts && self.nodes[j].dist > self.nodes[i].dist {
                    // a shorter path to the terminal: relink
                    self.nodes[j].parent = self.arcs[a].sister;
                    self.nodes[j].ts = self.nodes[i].ts;
                    self.nodes[j].dist = self.nodes[i].dist + 1;
                }
            }
            time += 1;
            if found != NONE {
                // grow from i again next: it may connect more paths
                self.nodes[i].in_active = true;
                self.active.push_front(i as u32);
                total += self.augment(found as usize);
                self.adopt(time);
            }
        }
        total
    }

    /// Push the bottleneck along s ~> tail(m) -> head(m) ~> t; nodes whose parent arc or
    /// terminal capacity saturates become orphans.
    fn augment(&mut self, m: usize) -> f64 {
        let u = self.arcs[self.arcs[m].sister as usize].head as usize;
        let v = self.arcs[m].head as usize;
        let mut b = self.arcs[m].rcap;
        let mut i = u;
        while self.nodes[i].parent != TERMINAL {
            let p = self.nodes[i].parent as usize;
            b = b.min(self.arcs[self.arcs[p].sister as usize].rcap);
            i = self.arcs[p].head as usize;
        }
        b = b.min(self.nodes[i].tr);
        i = v;
        while self.nodes[i].parent != TERMINAL {
            let p = self.nodes[i].parent as usize;
            b = b.min(self.arcs[p].rcap);
            i = self.arcs[p].head as usize;
        }
        b = b.min(-self.nodes[i].tr);

        let sm = self.arcs[m].sister as usize;
        self.arcs[m].rcap -= b;
        self.arcs[sm].rcap += b;
        i = u;
        while self.nodes[i].parent != TERMINAL {
            let p = self.nodes[i].parent as usize;
            let sp = self.arcs[p].sister as usize;
            self.arcs[p].rcap += b;
            self.arcs[sp].rcap -= b;
            let next = self.arcs[p].head as usize;
            if self.arcs[sp].rcap <= EPS {
                self.orphan(i);
            }
            i = next;
        }
        self.nodes[i].tr -= b;
        if self.nodes[i].tr <= EPS {
            self.orphan(i);
        }
        i = v;
        while self.nodes[i].parent != TERMINAL {
            let p = self.nodes[i].parent as usize;
            let sp = self.arcs[p].sister as usize;
            self.arcs[sp].rcap += b;
            self.arcs[p].rcap -= b;
            let next = self.arcs[p].head as usize;
            if self.arcs[p].rcap <= EPS {
                self.orphan(i);
            }
            i = next;
        }
        self.nodes[i].tr += b;
        if self.nodes[i].tr >= -EPS {
            self.orphan(i);
        }
        b
    }

    /// Distance from j to the terminal through valid parents, or INF_D if the chain ends in
    /// an orphan. Caches the distances along the chain (marked with `time`).
    fn origin_dist(&mut self, j0: usize, time: u32) -> u32 {
        let mut d: u32 = 0;
        let mut j = j0;
        loop {
            if self.nodes[j].ts == time {
                d += self.nodes[j].dist;
                break;
            }
            let p = self.nodes[j].parent;
            d += 1;
            if p == TERMINAL {
                self.nodes[j].ts = time;
                self.nodes[j].dist = 1;
                break;
            }
            if p == ORPHAN || p == NONE {
                return INF_D;
            }
            j = self.arcs[p as usize].head as usize;
        }
        let mut dd = d;
        let mut j = j0;
        while self.nodes[j].ts != time {
            self.nodes[j].ts = time;
            self.nodes[j].dist = dd;
            dd -= 1;
            j = self.arcs[self.nodes[j].parent as usize].head as usize;
        }
        d
    }

    /// Find new parents for the orphans in their own tree, or free them (and orphan their
    /// children).
    fn adopt(&mut self, time: u32) {
        while let Some(i) = self.orphans.pop_front() {
            let i = i as usize;
            let sink = self.nodes[i].in_sink;
            let (a0, a1) = (self.first[i] as usize, self.first[i + 1] as usize);
            let mut best = NONE;
            let mut d_min = INF_D;
            for a in a0..a1 {
                // residual towards i, in its tree's direction
                let r = if sink { self.arcs[a].rcap } else { self.arcs[self.arcs[a].sister as usize].rcap };
                if r <= EPS {
                    continue;
                }
                let j = self.arcs[a].head as usize;
                if self.nodes[j].in_sink == sink && self.nodes[j].parent != NONE {
                    let d = self.origin_dist(j, time);
                    if d < d_min {
                        d_min = d;
                        best = a as u32;
                    }
                }
            }
            if best != NONE {
                self.nodes[i].parent = best;
                self.nodes[i].ts = time;
                self.nodes[i].dist = d_min + 1;
                continue;
            }
            // no valid parent: i becomes free
            self.nodes[i].ts = 0;
            self.nodes[i].parent = NONE;
            for a in a0..a1 {
                let j = self.arcs[a].head as usize;
                let pj = self.nodes[j].parent;
                if self.nodes[j].in_sink != sink || pj == NONE {
                    continue;
                }
                let r = if sink { self.arcs[a].rcap } else { self.arcs[self.arcs[a].sister as usize].rcap };
                if r > EPS {
                    self.activate(j);
                }
                if pj != TERMINAL && pj != ORPHAN && self.arcs[pj as usize].head as usize == i {
                    self.orphan(j);
                }
            }
        }
    }

    /// After maxflow: 1 for nodes reachable from s in the residual graph.
    fn source_side(&self) -> Vec<u8> {
        let n = self.n();
        let mut seen = vec![0u8; n];
        let mut stack: Vec<u32> = Vec::new();
        for i in 0..n {
            if self.nodes[i].tr > EPS {
                seen[i] = 1;
                stack.push(i as u32);
            }
        }
        while let Some(u) = stack.pop() {
            let u = u as usize;
            for a in self.first[u] as usize..self.first[u + 1] as usize {
                let v = self.arcs[a].head as usize;
                if seen[v] == 0 && self.arcs[a].rcap > EPS {
                    seen[v] = 1;
                    stack.push(v as u32);
                }
            }
        }
        seen
    }
}

/// A flow network with the API of lib/maxflow.ts FlowGraph.
pub struct FlowGraph {
    n: usize,
    eu: Vec<u32>,
    ev: Vec<u32>,
    cap: Vec<f64>,  // per stored edge: e = 2k is u -> v, e + 1 is v -> u
    flow: Vec<f64>, // net flow per stored edge after maxflow
    side: Vec<u8>,
    s: usize,
}

impl FlowGraph {
    /// `pairs`: expected number of add_edge calls (the arrays grow if needed).
    pub fn new(n: usize, pairs: usize) -> FlowGraph {
        FlowGraph {
            n,
            eu: Vec::with_capacity(pairs),
            ev: Vec::with_capacity(pairs),
            cap: Vec::with_capacity(2 * pairs),
            flow: Vec::with_capacity(2 * pairs),
            side: Vec::new(),
            s: usize::MAX,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored (directed) edges, always even.
    pub fn m(&self) -> usize {
        self.cap.len()
    }

    /// Add u -> v with capacity c, and v -> u with capacity c_rev (0 = a plain directed edge).
    /// Returns the edge id.
    pub fn add_edge(&mut self, u: usize, v: usize, c: f64, c_rev: f64) -> usize {
        assert!(u < self.n && v < self.n, "add_edge: node out of range");
        let e = self.cap.len();
        self.eu.push(u as u32);
        self.ev.push(v as u32);
        self.cap.push(c);
        self.cap.push(c_rev);
        self.flow.push(0.0);
        self.flow.push(0.0);
        e
    }

    /// Net flow along edge e (u -> v); negative means net flow v -> u.
    pub fn flow(&self, e: usize) -> f64 {
        self.flow[e]
    }

    /// Maximum flow from s to t.
    pub fn maxflow(&mut self, s: usize, t: usize) -> f64 {
        assert!(s < self.n && t < self.n && s != t, "maxflow: bad terminals");
        let n = self.n;
        // non-terminal pairs go into the core; edges to s and t into src / snk per node
        let mut pu = Vec::new();
        let mut pv = Vec::new();
        let mut inner = Vec::new(); // the edge pair of each core pair
        let mut src = vec![0.0f64; n];
        let mut snk = vec![0.0f64; n];
        let mut direct = 0.0; // s -> t edges
        // a directed arc x -> y with capacity c; arcs into s or out of t never carry flow
        let term = |x: usize, y: usize, c: f64, src: &mut Vec<f64>, snk: &mut Vec<f64>, direct: &mut f64| {
            if x == s && y == t {
                *direct += c;
            } else if x == s && y != s {
                src[y] += c;
            } else if y == t && x != t {
                snk[x] += c;
            }
        };
        for k in 0..self.eu.len() {
            let (u, v) = (self.eu[k] as usize, self.ev[k] as usize);
            if u == v {
                continue;
            }
            if u == s || u == t || v == s || v == t {
                term(u, v, self.cap[2 * k], &mut src, &mut snk, &mut direct);
                term(v, u, self.cap[2 * k + 1], &mut src, &mut snk, &mut direct);
            } else {
                pu.push(u as u32);
                pv.push(v as u32);
                inner.push(k);
            }
        }
        let (mut core, arc) = Core::new(n, &pu, &pv);
        for (j, &k) in inner.iter().enumerate() {
            let a = arc[j] as usize;
            core.arcs[a].rcap = self.cap[2 * k];
            let sa = core.arcs[a].sister as usize;
            core.arcs[sa].rcap = self.cap[2 * k + 1];
        }
        let mut constant = direct;
        for i in 0..n {
            constant += src[i].min(snk[i]);
            core.nodes[i].tr = src[i] - snk[i];
        }
        let tr0: Vec<f64> = core.nodes.iter().map(|x| x.tr).collect();
        let f = core.maxflow();
        // flows: core arcs from their residuals; the edges to s or t of a node share its
        // terminal flow, in the order they were added
        let mut fs: Vec<f64> = (0..n).map(|i| src[i].min(snk[i]) + (tr0[i].max(0.0) - core.nodes[i].tr.max(0.0))).collect();
        let mut ft: Vec<f64> = (0..n).map(|i| src[i].min(snk[i]) + ((-tr0[i]).max(0.0) - (-core.nodes[i].tr).max(0.0))).collect();
        for x in self.flow.iter_mut() {
            *x = 0.0;
        }
        for (j, &k) in inner.iter().enumerate() {
            let fwd = self.cap[2 * k] - core.arcs[arc[j] as usize].rcap;
            self.flow[2 * k] = fwd;
            self.flow[2 * k + 1] = -fwd;
        }
        let mut give = |x: usize, y: usize, c: f64| -> f64 {
            if x == s && y == t {
                c
            } else if x == s && y != s {
                let g = c.min(fs[y]).max(0.0);
                fs[y] -= g;
                g
            } else if y == t && x != t {
                let g = c.min(ft[x]).max(0.0);
                ft[x] -= g;
                g
            } else {
                0.0
            }
        };
        for k in 0..self.eu.len() {
            let (u, v) = (self.eu[k] as usize, self.ev[k] as usize);
            if u == v || !(u == s || u == t || v == s || v == t) {
                continue;
            }
            let net = give(u, v, self.cap[2 * k]) - give(v, u, self.cap[2 * k + 1]);
            self.flow[2 * k] = net;
            self.flow[2 * k + 1] = -net;
        }
        let mut side = core.source_side();
        side[s] = 1;
        side[t] = 0;
        self.side = side;
        self.s = s;
        f + constant
    }

    /// After maxflow: 1 for nodes reachable from s in the residual graph (the source side S of
    /// a minimum cut).
    pub fn source_side(&self, s: usize) -> Vec<u8> {
        assert!(s == self.s, "source_side: call maxflow(s, t) first, with the same s");
        self.side.clone()
    }
}

/// Graph-cut segmentation on a fixed neighbour graph (`solveSeg` of lib/graph.ts): node per
/// pixel, s = object, t = background; s -> p: D0[p], p -> t: D1[p], p <-> q: lambda * w[k].
/// The graph is built once; `solve` only rewrites the capacities.
pub struct GridCut {
    core: Core,
    arc: Vec<u32>,
    labels: Vec<u8>,
}

impl GridCut {
    /// `n` pixels; neighbour pairs ep[k] - eq[k] (each undirected pair once).
    pub fn new(n: usize, ep: &[i32], eq: &[i32]) -> GridCut {
        assert_eq!(ep.len(), eq.len(), "GridCut: ep and eq differ in length");
        let pu: Vec<u32> = ep.iter().map(|&v| v as u32).collect();
        let pv: Vec<u32> = eq.iter().map(|&v| v as u32).collect();
        assert!(pu.iter().chain(pv.iter()).all(|&v| (v as usize) < n), "GridCut: node out of range");
        let (core, arc) = Core::new(n, &pu, &pv);
        GridCut { core, arc, labels: vec![0; n] }
    }

    pub fn n(&self) -> usize {
        self.labels.len()
    }

    pub fn pairs(&self) -> usize {
        self.arc.len()
    }

    /// Minimise E(L) = sum D_{L_p}(p) + lambda sum w_k [L_p != L_q] with seeds (1 object, -1
    /// background, 0 free). Returns the max flow plus the constant sent straight through, which
    /// equals the minimum of the energy including the seeds' data costs (as solveSeg's `flow`).
    pub fn solve(&mut self, d0: &[f64], d1: &[f64], seeds: &[i8], w: &[f64], lambda: f64) -> f64 {
        let n = self.labels.len();
        assert!(d0.len() == n && d1.len() == n && seeds.len() == n && w.len() == self.arc.len(), "GridCut.solve: lengths");
        // "infinite" capacity: more than the cost of any labelling that respects the seeds
        let mut big = 1.0;
        for p in 0..n {
            big += d0[p].max(d1[p]);
        }
        for &wk in w {
            big += lambda * wk;
        }
        let mut constant = 0.0;
        for p in 0..n {
            let sd = seeds[p];
            let a = if sd == 1 { big } else if sd == -1 { 0.0 } else { d0[p] }; // s -> p
            let b = if sd == -1 { big } else if sd == 1 { 0.0 } else { d1[p] }; // p -> t
            if sd == 1 {
                constant += d1[p];
            }
            if sd == -1 {
                constant += d0[p];
            }
            constant += a.min(b);
            self.core.nodes[p].tr = a - b;
        }
        for k in 0..w.len() {
            let c = (lambda * w[k]).max(0.0);
            let a = self.arc[k] as usize;
            self.core.arcs[a].rcap = c;
            let sa = self.core.arcs[a].sister as usize;
            self.core.arcs[sa].rcap = c;
        }
        let f = self.core.maxflow();
        self.labels = self.core.source_side();
        f + constant
    }

    /// Labels of the last solve: 1 = object (source side), 0 = background.
    pub fn labels(&self) -> Vec<u8> {
        self.labels.clone()
    }
}

// ---- The Wasm interface ------------------------------------------------------------------------
// Plain functions over handles (wasm-bindgen classes would need the `Symbol.dispose` type in the
// TypeScript lib); demos/src/graphcut/wasmcut.ts wraps them in classes again.

thread_local! {
    static GRAPHS: std::cell::RefCell<Vec<Option<FlowGraph>>> = std::cell::RefCell::new(Vec::new());
    static GRIDS: std::cell::RefCell<Vec<Option<GridCut>>> = std::cell::RefCell::new(Vec::new());
}

fn store<T>(v: &mut Vec<Option<T>>, x: T) -> u32 {
    if let Some(i) = v.iter().position(|e| e.is_none()) {
        v[i] = Some(x);
        i as u32
    } else {
        v.push(Some(x));
        (v.len() - 1) as u32
    }
}

fn with_graph<R>(h: u32, f: impl FnOnce(&mut FlowGraph) -> R) -> R {
    GRAPHS.with(|g| f(g.borrow_mut()[h as usize].as_mut().expect("maxflow: graph was freed")))
}

fn with_grid<R>(h: u32, f: impl FnOnce(&mut GridCut) -> R) -> R {
    GRIDS.with(|g| f(g.borrow_mut()[h as usize].as_mut().expect("maxflow: grid was freed")))
}

/// New flow network with n nodes (FlowGraph(n, pairs)); returns its handle.
#[wasm_bindgen]
pub fn graph_new(n: usize, pairs: usize) -> u32 {
    GRAPHS.with(|g| store(&mut g.borrow_mut(), FlowGraph::new(n, pairs)))
}

#[wasm_bindgen]
pub fn graph_free(h: u32) {
    GRAPHS.with(|g| g.borrow_mut()[h as usize] = None);
}

/// addEdge(u, v, c, cRev): returns the edge id (pairs e, e ^ 1).
#[wasm_bindgen]
pub fn graph_add_edge(h: u32, u: usize, v: usize, c: f64, c_rev: f64) -> usize {
    with_graph(h, |g| g.add_edge(u, v, c, c_rev))
}

#[wasm_bindgen]
pub fn graph_maxflow(h: u32, s: usize, t: usize) -> f64 {
    with_graph(h, |g| g.maxflow(s, t))
}

#[wasm_bindgen]
pub fn graph_source_side(h: u32, s: usize) -> Vec<u8> {
    with_graph(h, |g| g.source_side(s))
}

#[wasm_bindgen]
pub fn graph_flow(h: u32, e: usize) -> f64 {
    with_graph(h, |g| g.flow(e))
}

/// New GridCut over the neighbour pairs ep[k] - eq[k] of n pixels; returns its handle.
#[wasm_bindgen]
pub fn grid_new(n: usize, ep: &[i32], eq: &[i32]) -> u32 {
    GRIDS.with(|g| store(&mut g.borrow_mut(), GridCut::new(n, ep, eq)))
}

#[wasm_bindgen]
pub fn grid_free(h: u32) {
    GRIDS.with(|g| g.borrow_mut()[h as usize] = None);
}

/// GridCut.solve: the minimum energy (max flow plus the constant); labels with grid_labels.
#[wasm_bindgen]
pub fn grid_solve(h: u32, d0: &[f64], d1: &[f64], seeds: &[i8], w: &[f64], lambda: f64) -> f64 {
    with_grid(h, |g| g.solve(d0, d1, seeds, w, lambda))
}

#[wasm_bindgen]
pub fn grid_labels(h: u32) -> Vec<u8> {
    with_grid(h, |g| g.labels())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_graph() {
        // CLRS example: max flow 23, minimum cut {s, v1, v2, v4}
        let mut g = FlowGraph::new(6, 10);
        for &(u, v, c) in &[(0, 1, 16.0), (0, 2, 13.0), (1, 3, 12.0), (2, 1, 4.0), (2, 4, 14.0), (3, 2, 9.0), (3, 5, 20.0), (4, 3, 7.0), (4, 5, 4.0)] {
            g.add_edge(u, v, c, 0.0);
        }
        assert!((g.maxflow(0, 5) - 23.0).abs() < 1e-9);
        assert_eq!(g.source_side(0), vec![1, 1, 1, 0, 1, 0]);
    }

    #[test]
    fn grid_cut() {
        // 1 x 3 image (3, 1, 3) of the exercise, D(0) = I^2, D(1) = (3 - I)^2, lambda 1 and 2
        let mut gc = GridCut::new(3, &[0, 1], &[1, 2]);
        let (d0, d1) = ([9.0, 1.0, 9.0], [0.0, 4.0, 0.0]);
        assert!((gc.solve(&d0, &d1, &[0, 0, 0], &[1.0, 1.0], 1.0) - 3.0).abs() < 1e-9);
        assert_eq!(gc.labels(), vec![1, 0, 1]);
        assert!((gc.solve(&d0, &d1, &[0, 0, 0], &[1.0, 1.0], 2.0) - 4.0).abs() < 1e-9);
        assert_eq!(gc.labels(), vec![1, 1, 1]);
    }
}
