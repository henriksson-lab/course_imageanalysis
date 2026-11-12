// Union–find (disjoint-set forest) with switchable heuristics, instrumented for teaching
// (labelling day, unit label-connectivity/). No DOM. Used by the unionfind demo and by the
// two-pass labelling of morphgrid's "label" mode.
//
//  - union(a, b), naive: the root of a is linked under the root of b.
//  - union by rank: the root of lower rank goes under the other; on a tie, a's root under
//    b's root and b's rank grows by one. (Rank = upper bound on the height of the tree.)
//  - find(x) with path compression: none, full (every node on the path then points to the
//    root) or halving (every node on the path points to its grandparent).
//  - steps counts parent pointers followed by all finds (the cost measure).

export type Compression = "none" | "full" | "halving";

export interface ForestOptions { byRank?: boolean; compression?: Compression }

export interface FindResult { root: number; path: number[] }
export interface UnionResult { ra: number; rb: number; root: number; child: number | null; pathA: number[]; pathB: number[] }

export class Forest {
  parent: number[] = [];
  rank: number[] = [];
  byRank: boolean;
  compression: Compression;
  /** Parent pointers followed in all finds so far. */
  steps = 0;

  constructor(n = 0, o: ForestOptions = {}) {
    this.byRank = o.byRank ?? true;
    this.compression = o.compression ?? "full";
    for (let i = 0; i < n; i++) this.add();
  }

  get size(): number { return this.parent.length; }

  /** Add a singleton set; returns its index. */
  add(): number { this.parent.push(this.parent.length); this.rank.push(0); return this.parent.length - 1; }

  /** Root of x; `path` lists the nodes visited from x up to (and including) the root. */
  find(x: number): FindResult {
    const p = this.parent, path = [x];
    let r = x;
    while (p[r] !== r) { r = p[r]; path.push(r); this.steps++; }
    if (this.compression === "full") for (const v of path) p[v] = r;
    else if (this.compression === "halving") {
      // each node on the path now points to its (old) grandparent
      for (let k = 0; k + 2 < path.length; k++) p[path[k]] = path[k + 2];
    }
    return { root: r, path };
  }

  /** Merge the sets of a and b. */
  union(a: number, b: number): UnionResult {
    const fa = this.find(a), fb = this.find(b);
    let ra = fa.root, rb = fb.root;
    if (ra === rb) return { ra, rb, root: ra, child: null, pathA: fa.path, pathB: fb.path };
    let child = ra, root = rb;
    if (this.byRank) {
      if (this.rank[ra] > this.rank[rb]) { child = rb; root = ra; }
      else if (this.rank[ra] === this.rank[rb]) this.rank[rb]++;
    }
    this.parent[child] = root;
    return { ra, rb, root, child, pathA: fa.path, pathB: fb.path };
  }

  /** Depth of x (number of edges to its root), without changing anything. */
  depth(x: number): number {
    let d = 0;
    while (this.parent[x] !== x) { x = this.parent[x]; d++; }
    return d;
  }

  maxDepth(): number {
    let m = 0;
    for (let i = 0; i < this.size; i++) m = Math.max(m, this.depth(i));
    return m;
  }

  /** Root of x without changing anything. */
  rootOf(x: number): number { while (this.parent[x] !== x) x = this.parent[x]; return x; }

  /** The sets, each sorted, ordered by their smallest element. */
  classes(): number[][] {
    const m = new Map<number, number[]>();
    for (let i = 0; i < this.size; i++) { const r = this.rootOf(i); if (!m.has(r)) m.set(r, []); m.get(r)!.push(i); }
    return [...m.values()].sort((a, b) => a[0] - b[0]);
  }

  clone(): Forest {
    const f = new Forest(0, { byRank: this.byRank, compression: this.compression });
    f.parent = this.parent.slice(); f.rank = this.rank.slice(); f.steps = this.steps;
    return f;
  }
}
