// Maximum flow / minimum s-t cut. No DOM: also used by the test (demos/test/).
//
//  - FlowGraph + maxflow(): Dinic's algorithm (BFS levels, blocking flow by DFS with
//    per-node edge pointers), used for image-sized graphs (128 x 128 pixels and more).
//    Edges are stored in pairs (e, e ^ 1) so the residual reverse edge is e ^ 1.
//    An undirected edge (an n-link) is one pair with capacity c in both directions.
//  - edmondsKarpSteps(): the textbook Ford-Fulkerson with BFS (shortest augmenting
//    paths), on a small edge list, recording every augmenting path for the step-through
//    demo.
// Capacities are non-negative floats; residuals below EPS count as 0.

export const EPS = 1e-9;

export class FlowGraph {
  readonly n: number;
  head: Int32Array;
  to: Int32Array;
  nxt: Int32Array;
  cap: Float64Array;       // residual capacity
  cap0: Float64Array;      // original capacity
  m = 0;                   // number of stored (directed) edges, always even
  private level: Int32Array;
  private iter: Int32Array;
  private queue: Int32Array;

  /** `pairs`: expected number of addEdge calls (the arrays grow if needed). */
  constructor(n: number, pairs = 16) {
    this.n = n;
    this.head = new Int32Array(n).fill(-1);
    this.level = new Int32Array(n);
    this.iter = new Int32Array(n);
    this.queue = new Int32Array(n);
    const m = 2 * Math.max(1, pairs);
    this.to = new Int32Array(m); this.nxt = new Int32Array(m);
    this.cap = new Float64Array(m); this.cap0 = new Float64Array(m);
  }

  private grow(): void {
    const m = 2 * this.to.length;
    const g = <T extends Int32Array | Float64Array>(a: T, b: T) => { b.set(a); return b; };
    this.to = g(this.to, new Int32Array(m)); this.nxt = g(this.nxt, new Int32Array(m));
    this.cap = g(this.cap, new Float64Array(m)); this.cap0 = g(this.cap0, new Float64Array(m));
  }

  /** Add u -> v with capacity c, and v -> u with capacity cRev (0 = a plain directed edge). Returns the edge id. */
  addEdge(u: number, v: number, c: number, cRev = 0): number {
    if (this.m + 2 > this.to.length) this.grow();
    const e = this.m;
    this.m += 2;
    this.to[e] = v; this.to[e + 1] = u;
    this.cap[e] = this.cap0[e] = c;
    this.cap[e + 1] = this.cap0[e + 1] = cRev;
    this.nxt[e] = this.head[u]; this.nxt[e + 1] = this.head[v];
    this.head[u] = e;
    this.head[v] = e + 1;
    return e;
  }

  /** Net flow along edge e (u -> v); negative means net flow v -> u. */
  flow(e: number): number { return this.cap0[e] - this.cap[e]; }

  private bfs(s: number, t: number): boolean {
    const { level, head, nxt, to, cap } = this;
    level.fill(-1);
    const q = this.queue;
    let qh = 0, qt = 0;
    q[qt++] = s; level[s] = 0;
    while (qh < qt) {
      const u = q[qh++];
      for (let e = head[u]; e !== -1; e = nxt[e]) {
        const v = to[e];
        if (cap[e] > EPS && level[v] < 0) { level[v] = level[u] + 1; q[qt++] = v; }
      }
    }
    return level[t] >= 0;
  }

  /** Blocking flow on the level graph, iterative DFS (paths can be thousands of nodes long). */
  private blocking(s: number, t: number): number {
    const { level, iter } = this;
    const { nxt, to, cap } = this;
    for (let u = 0; u < this.n; u++) iter[u] = this.head[u];
    let total = 0;
    const path: number[] = [];   // edge ids from s
    let u = s;
    for (;;) {
      if (u === t) {
        let b = Infinity;
        for (const e of path) if (cap[e] < b) b = cap[e];
        let cut = path.length;
        for (let i = 0; i < path.length; i++) {
          const e = path[i];
          cap[e] -= b; cap[e ^ 1] += b;
          if (cap[e] <= EPS && i < cut) cut = i;
        }
        total += b;
        path.length = cut;           // back to the tail of the first saturated edge
        u = cut === 0 ? s : to[path[cut - 1]];
        continue;
      }
      let e = iter[u];
      while (e !== -1 && !(cap[e] > EPS && level[to[e]] === level[u] + 1)) e = nxt[e];
      iter[u] = e;
      if (e !== -1) { path.push(e); u = to[e]; continue; }
      // dead end: remove u from the level graph and step back
      level[u] = -1;
      if (u === s) break;
      const back = path.pop()!;
      u = to[back ^ 1];
      iter[u] = nxt[iter[u]];
    }
    return total;
  }

  /** Maximum flow from s to t (Dinic). Afterwards cap[] holds the residual graph. */
  maxflow(s: number, t: number): number {
    let f = 0;
    while (this.bfs(s, t)) f += this.blocking(s, t);
    return f;
  }

  /** After maxflow: 1 for nodes reachable from s in the residual graph (the source side S of a minimum cut). */
  sourceSide(s: number): Uint8Array {
    const seen = new Uint8Array(this.n);
    const stack = [s]; seen[s] = 1;
    while (stack.length) {
      const u = stack.pop()!;
      for (let e = this.head[u]; e !== -1; e = this.nxt[e]) {
        const v = this.to[e];
        if (!seen[v] && this.cap[e] > EPS) { seen[v] = 1; stack.push(v); }
      }
    }
    return seen;
  }
}

// ---------------------------------------------------------------------------
// Instrumented Edmonds-Karp for small graphs (the step-through demo).

export interface Edge { u: number; v: number; c: number }

/** One residual arc: edge index i, used forwards (u -> v) or backwards (v -> u, undoing flow). */
export interface Arc { i: number; fwd: boolean }

export interface Step {
  path: Arc[];        // the augmenting path, from s to t
  nodes: number[];    // the nodes on it
  bottleneck: number;
  flow: number[];     // flow on every edge AFTER this augmentation
  total: number;      // total flow after it
}

export interface EKResult { steps: Step[]; flow: number[]; total: number; S: boolean[] }

/** Residual arcs leaving each node, in a fixed order (edges in list order: forward then backward). */
function residualArcs(n: number, edges: Edge[], flow: number[]): Arc[][] {
  const out: Arc[][] = Array.from({ length: n }, () => []);
  edges.forEach((e, i) => {
    if (e.c - flow[i] > EPS) out[e.u].push({ i, fwd: true });
    if (flow[i] > EPS) out[e.v].push({ i, fwd: false });
  });
  return out;
}

/** Nodes reachable from s in the residual graph of `flow`. */
export function reachable(n: number, edges: Edge[], flow: number[], s: number): boolean[] {
  const out = residualArcs(n, edges, flow);
  const seen = new Array<boolean>(n).fill(false);
  const q = [s]; seen[s] = true;
  while (q.length) {
    const u = q.shift()!;
    for (const a of out[u]) {
      const v = a.fwd ? edges[a.i].v : edges[a.i].u;
      if (!seen[v]) { seen[v] = true; q.push(v); }
    }
  }
  return seen;
}

/** Ford-Fulkerson with BFS (shortest augmenting path first); records each step. */
export function edmondsKarpSteps(n: number, edges: Edge[], s: number, t: number): EKResult {
  const flow = new Array<number>(edges.length).fill(0);
  const steps: Step[] = [];
  let total = 0;
  for (let guard = 0; guard < 10000; guard++) {
    const out = residualArcs(n, edges, flow);
    const prev: (Arc | null)[] = new Array(n).fill(null);
    const seen = new Array<boolean>(n).fill(false);
    const q = [s]; seen[s] = true;
    while (q.length && !seen[t]) {
      const u = q.shift()!;
      for (const a of out[u]) {
        const v = a.fwd ? edges[a.i].v : edges[a.i].u;
        if (!seen[v]) { seen[v] = true; prev[v] = a; q.push(v); }
      }
    }
    if (!seen[t]) break;
    const path: Arc[] = [];
    const nodes = [t];
    for (let v = t; v !== s;) {
      const a = prev[v]!;
      path.unshift(a);
      v = a.fwd ? edges[a.i].u : edges[a.i].v;
      nodes.unshift(v);
    }
    let b = Infinity;
    for (const a of path) b = Math.min(b, a.fwd ? edges[a.i].c - flow[a.i] : flow[a.i]);
    for (const a of path) flow[a.i] += a.fwd ? b : -b;
    total += b;
    steps.push({ path, nodes, bottleneck: b, flow: flow.slice(), total });
  }
  return { steps, flow, total, S: reachable(n, edges, flow, s) };
}

/** Cost of the cut (S, T): capacities of the edges from S to T. */
export function cutCost(edges: Edge[], S: boolean[]): number {
  let c = 0;
  for (const e of edges) if (S[e.u] && !S[e.v]) c += e.c;
  return c;
}
