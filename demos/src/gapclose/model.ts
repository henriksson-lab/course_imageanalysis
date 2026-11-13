// Model of the gapclose demo (no DOM): simulate dividing cells, link frame to frame, then join
// segments (gap closing, splitting, merging), evaluate against the truth and build lineages.
import { simCells, CellMovie } from "../lib/tracksim";
import * as LK from "../lib/linking";
import { Branch, cycleTimes } from "../lib/lineage";

export const T = 60, W = 400, H = 40, N0 = 4, RADIUS = 6;
export interface State {
  miss: number; falseRate: number; splitProb: number; enterRate: number;
  maxGap: number; gate: number; bd: number;
  splitting: boolean; merging: boolean; kRule: boolean;
  cycleMean: number; cycleShape: number;
  founder: number; candidates: boolean; seed: number;
}
export const DEFAULT: State = {
  miss: 0.05, falseRate: 0, splitProb: 0, enterRate: 0,
  maxGap: 3, gate: 8, bd: 8, splitting: true, merging: false, kRule: false,
  cycleMean: 20, cycleShape: 4, founder: 1, candidates: false, seed: 3,
};
export const K_FRAMES = 3;
export const PRESETS: { id: string; label: string; s: Partial<State> }[] = [
  { id: "blink", label: "a blink: gap closing", s: { miss: 0.08, maxGap: 3, splitting: false } },
  { id: "nogap", label: "no gap closing", s: { miss: 0.08, maxGap: 0, splitting: false } },
  { id: "bigG", label: "G too large: wrong joins", s: { miss: 0.08, falseRate: 1.5, maxGap: 10, gate: 14, bd: 14, splitting: false } },
  { id: "division", label: "a division", s: { miss: 0, splitting: true } },
  { id: "bubble", label: "segmentation split → a bubble", s: { miss: 0, splitProb: 0.03, splitting: true, merging: true } },
  { id: "krule", label: "bubble removed by the k-frame rule", s: { miss: 0, splitProb: 0.03, splitting: true, merging: true, kRule: true } },
  { id: "border", label: "cells entering at the border", s: { miss: 0.02, enterRate: 0.08, splitting: true, founder: 5 } },
  { id: "censor", label: "the censoring trap", s: { miss: 0, cycleMean: 26, cycleShape: 2, splitting: true } },
];

export function simulate(s: State): CellMovie {
  return simCells({ n0: N0, T, W, H, D: 0.6, cycleMean: s.cycleMean, cycleMin: 8, cycleShape: s.cycleShape, radius: RADIUS,
    miss: s.miss, falseRate: s.falseRate, splitProb: s.splitProb, enterRate: s.enterRate, seed: s.seed });
}

export interface Analysis {
  frameEdges: LK.Edge[];
  seg: LK.SegmentResult;
  edges: LK.Edge[];
  wrong: boolean[];
  links: LK.Scores;
  div: LK.Scores;
  complete: { complete: number; total: number };
  reasons: { start: Record<string, number>; end: Record<string, number> };
  /** result lineage: one branch per unbranched chain */
  branches: Branch[];
  chainOf: Int32Array;
  truthBranches: Branch[];
  truthChainOf: Int32Array;
}

/** Branches (for lineage drawing) from a track graph: chains through link/gap edges, parents through divisions. */
export function lineageOf(mv: CellMovie, edges: LK.Edge[]): { branches: Branch[]; chainOf: Int32Array } {
  const { id, tracks } = LK.chains(mv.dets.length, edges, ["link", "gap"]);
  const adj = LK.adjacency(mv.dets.length, edges);
  const branches: Branch[] = tracks.map((tr, k) => ({ id: k, parent: -1, t0: mv.dets[tr[0]].t, t1: mv.dets[tr[tr.length - 1]].t, children: [] }));
  tracks.forEach((tr, k) => {
    const last = tr[tr.length - 1];
    if (adj.out[last].length === 2) for (const e of adj.out[last]) {
      const c = id[edges[e].b];
      if (c >= 0 && c !== k && branches[c].parent < 0) { branches[c].parent = k; branches[k].children.push(c); }
    }
  });
  // after a merge the chain restarts: continue it from its non-merge predecessor (one child)
  tracks.forEach((tr, k) => {
    if (branches[k].parent >= 0) return;
    const inn = adj.inn[tr[0]].map(e => edges[e]).filter(e => e.kind !== "merge");
    if (inn.length !== 1 || adj.out[inn[0].a].length !== 1) return;
    const p = id[inn[0].a];
    if (p >= 0 && p !== k) { branches[k].parent = p; branches[p].children.push(k); }
  });
  return { branches, chainOf: id };
}

export function analyse(mv: CellMovie, s: State): Analysis {
  const frames = LK.byFrame(mv.dets, T);
  const frameEdges = LK.linkFrames(mv.dets, frames, { method: "global", gate: 12, b: s.bd * s.bd });
  const seg = LK.linkSegments(mv.dets, frameEdges, {
    maxGap: s.maxGap, gate: s.gate, b: s.bd * s.bd, splitting: s.splitting, merging: s.merging, minDaughter: s.kRule ? K_FRAMES : 0,
  });
  const edges = seg.edges;
  const ev = LK.evalLinks(edges, mv.truth);
  const div = LK.evalDivisions(LK.divisions(mv.dets.length, edges), LK.divisions(mv.dets.length, mv.truth));
  const complete = LK.completeTracks(mv.dets.length, edges, mv.truth);
  const rs = LK.reasons(mv.dets, edges, T, W, H, 2 * RADIUS, "x");   // cells enter and leave at the left and right
  const count = (a: string[]) => a.reduce((o, k) => { o[k] = (o[k] ?? 0) + 1; return o; }, {} as Record<string, number>);
  const lin = lineageOf(mv, edges), tl = lineageOf(mv, mv.truth);
  // a result branch is wrong if it starts with a division that is not a true one, or holds a wrong link
  const tdivs = new Set(LK.divisions(mv.dets.length, mv.truth).map(d => `${d.mother}:${d.daughters.join(":")}`));
  const rdivs = LK.divisions(mv.dets.length, edges);
  const badMother = new Set(rdivs.filter(d => !tdivs.has(`${d.mother}:${d.daughters.join(":")}`)).map(d => d.mother));
  edges.forEach((e, i) => { if (ev.wrong[i]) { const c = lin.chainOf[e.b]; if (c >= 0) lin.branches[c].wrong = true; } });
  const { tracks } = LK.chains(mv.dets.length, edges, ["link", "gap"]);
  for (const b of lin.branches) if (b.parent >= 0 && badMother.has(tracks[b.parent][tracks[b.parent].length - 1])) b.wrong = true;
  return {
    frameEdges, seg, edges, wrong: ev.wrong, links: ev, div, complete,
    reasons: { start: count(rs.start), end: count(rs.end) },
    branches: lin.branches, chainOf: lin.chainOf, truthBranches: tl.branches, truthChainOf: tl.chainOf,
  };
}

/** Root branch of the lineage holding the founder's first detection (founder ids 0..). */
export function rootOf(mv: CellMovie, chainOf: Int32Array, branches: Branch[], cellId: number): number {
  const k = mv.dets.findIndex(d => d.pid === cellId);
  if (k < 0) return -1;
  let b = chainOf[k];
  while (b >= 0 && branches[b].parent >= 0) b = branches[b].parent;
  return b;
}

/** Cycle times of the reconstructed lineage: complete ones (born by division and divided) and all. */
export function cycles(a: Analysis): { complete: number[]; all: number[] } {
  // only branches born by a division (the parent has two children) have a cycle start
  const born = (id: number) => { const p = a.branches[id].parent; return p >= 0 && a.branches[p].children.length === 2; };
  const ct = cycleTimes(a.branches, b => b.children.length === 2);
  return { complete: ct.filter(c => born(c.id) && c.complete).map(c => c.cycle), all: ct.filter(c => born(c.id)).map(c => c.cycle) };
}

/** Cycle times of the true cells (complete: born by division and divided within the movie). */
export function trueCycles(mv: CellMovie): { complete: number[]; censored: number } {
  const complete: number[] = [];
  let censored = 0;
  for (const c of mv.cells) {
    if (c.parent < 0) continue;
    if (c.divides) complete.push(c.t1 - c.t0 + 1); else censored++;
  }
  return { complete, censored };
}
