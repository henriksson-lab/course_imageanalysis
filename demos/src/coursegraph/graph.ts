// The course structure as a graph (no DOM; tested in demos/test/coursegraph-test.ts).
// Data: days.json, generated from days.yml by tools/days_json.py.
// Nodes are units. A unit's direct hard prerequisites are the previous unit of its day, its day's
// hard prerequisites and its own; a prerequisite naming a day stands for the day's last unit
// (which, through the chain inside the day, needs all of the day). Same rules as tools/days.py.

export interface UnitData { folder: string; title: string; planned: boolean; hard: string[]; soft: string[] }
export interface DayData {
  id: string; number: number; title: string; part: number; planned: boolean;
  hard: string[]; soft: string[]; owns: string[]; units: UnitData[];
}
export interface PartData { id: string; title: string; colour: string; tint: string }
export interface CourseData { parts: PartData[]; days: DayData[] }

export interface Unit extends UnitData { index: number; day: number; k: number }
export interface Course {
  data: CourseData;
  units: Unit[];                       // in course order
  dayUnits: number[][];                // day index -> unit indices
  dayIndex: Map<string, number>;
  unitIndex: Map<string, number>;
  prereqs: number[][];                 // unit -> direct hard prerequisite units
  soft: number[][];                    // unit -> soft references (units)
}

/** A prerequisite reference -> unit index ("pca/" = that unit, "images" = last unit of the day). */
export function resolve(c: { dayIndex: Map<string, number>; unitIndex: Map<string, number>; dayUnits: number[][] },
  ref: string): number {
  if (ref.endsWith("/")) {
    const u = c.unitIndex.get(ref.slice(0, -1));
    if (u === undefined) throw new Error("unknown unit " + ref);
    return u;
  }
  const d = c.dayIndex.get(ref);
  if (d === undefined) throw new Error("unknown day " + ref);
  const us = c.dayUnits[d];
  return us[us.length - 1];
}

export function buildCourse(data: CourseData): Course {
  const units: Unit[] = [], dayUnits: number[][] = [];
  const dayIndex = new Map<string, number>(), unitIndex = new Map<string, number>();
  data.days.forEach((d, di) => {
    dayIndex.set(d.id, di);
    dayUnits.push(d.units.map((u, k) => {
      const index = units.length;
      units.push({ ...u, index, day: di, k });
      unitIndex.set(u.folder, index);
      return index;
    }));
  });
  const c = { data, units, dayUnits, dayIndex, unitIndex, prereqs: [] as number[][], soft: [] as number[][] };
  for (const u of units) {
    const d = data.days[u.day];
    const p = new Set<number>();
    if (u.k > 0) p.add(dayUnits[u.day][u.k - 1]);
    for (const r of d.hard) p.add(resolve(c, r));
    for (const r of u.hard) p.add(resolve(c, r));
    p.delete(u.index);
    c.prereqs.push([...p]);
    const s = new Set<number>();
    for (const r of [...d.soft, ...u.soft]) s.add(resolve(c, r));
    s.delete(u.index);
    c.soft.push([...s]);
  }
  return c;
}

/** All nodes reachable from `targets` through `prereqs` (transitive hard prerequisites),
 *  excluding the targets themselves unless one is a prerequisite of another target. */
export function closure(prereqs: number[][], targets: number[]): Set<number> {
  const seen = new Set<number>();
  const stack: number[] = [];
  for (const t of targets) for (const p of prereqs[t]) stack.push(p);
  while (stack.length) {
    const n = stack.pop()!;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const p of prereqs[n]) if (!seen.has(p)) stack.push(p);
  }
  return seen;
}

/** Topological order of `nodes` (prerequisites first; edges leaving the set are ignored).
 *  Kahn's algorithm; among the nodes that are ready, the smallest `pos` goes first, so the path
 *  follows the course order wherever the prerequisites allow. Throws on a cycle. */
export function topoOrder(prereqs: number[][], nodes: Iterable<number>, pos: (n: number) => number = n => n): number[] {
  const set = new Set(nodes);
  const indeg = new Map<number, number>(), users = new Map<number, number[]>();
  for (const n of set) { indeg.set(n, 0); users.set(n, []); }
  for (const n of set) for (const p of new Set(prereqs[n])) if (set.has(p)) {
    indeg.set(n, indeg.get(n)! + 1);
    users.get(p)!.push(n);
  }
  const ready = [...set].filter(n => indeg.get(n) === 0);
  const out: number[] = [];
  while (ready.length) {
    let b = 0;
    for (let i = 1; i < ready.length; i++) if (pos(ready[i]) < pos(ready[b])) b = i;
    const n = ready.splice(b, 1)[0];
    out.push(n);
    for (const m of users.get(n)!) {
      const v = indeg.get(m)! - 1;
      indeg.set(m, v);
      if (v === 0) ready.push(m);
    }
  }
  if (out.length !== set.size) throw new Error("cycle among the prerequisites");
  return out;
}

export interface PathStep { day: number; units: number[]; all: boolean }

/** The learning path before a selection (a day: all its units; or one unit): its transitive hard
 *  prerequisites in topological order, grouped into consecutive runs of the same day. */
export function learningPath(c: Course, sel: { day?: number; unit?: number }): { steps: PathStep[]; need: Set<number>; targets: number[] } {
  const targets = sel.unit !== undefined ? [sel.unit] : c.dayUnits[sel.day!];
  const need = closure(c.prereqs, targets);
  for (const t of targets) need.delete(t);
  const order = topoOrder(c.prereqs, need);
  const steps: PathStep[] = [];
  for (const u of order) {
    const d = c.units[u].day;
    const last = steps[steps.length - 1];
    if (last && last.day === d) last.units.push(u);
    else steps.push({ day: d, units: [u], all: false });
  }
  for (const s of steps) s.all = s.units.length === c.dayUnits[s.day].length;
  return { steps, need, targets };
}

export interface DayEdge { from: number; to: number; hard: boolean }

/** Day-level edges prerequisite day -> dependent day (from the day's and its units' references;
 *  the implied chain inside a day is left out). Hard wins over soft for the same pair. */
export function dayEdges(c: Course): DayEdge[] {
  const m = new Map<string, DayEdge>();
  const add = (from: number, to: number, hard: boolean) => {
    if (from === to) return;
    const key = from + ">" + to, e = m.get(key);
    if (!e) m.set(key, { from, to, hard });
    else if (hard) e.hard = true;
  };
  c.data.days.forEach((d, di) => {
    for (const r of d.hard) add(c.units[resolve(c, r)].day, di, true);
    for (const r of d.soft) add(c.units[resolve(c, r)].day, di, false);
    for (const u of d.units) {
      for (const r of u.hard) add(c.units[resolve(c, r)].day, di, true);
      for (const r of u.soft) add(c.units[resolve(c, r)].day, di, false);
    }
  });
  return [...m.values()];
}

/** Days that list `day` (or one of its units) as a direct hard prerequisite. */
export function dependents(c: Course, day: number): number[] {
  return dayEdges(c).filter(e => e.hard && e.from === day).map(e => e.to).sort((a, b) => a - b);
}

export interface Hit { day: number; unit?: number; text: string; field: "day" | "unit" | "topic" }

/** Case-insensitive search over day titles, unit titles and topics ("owns"); every word of the
 *  query must occur in the text. Day-title hits first, then units, then topics. */
export function search(c: Course, query: string, limit = 30): Hit[] {
  const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (!words.length) return [];
  const match = (s: string) => { const t = s.toLowerCase(); return words.every(w => t.includes(w)); };
  const days: Hit[] = [], units: Hit[] = [], topics: Hit[] = [];
  c.data.days.forEach((d, di) => {
    if (match(d.title)) days.push({ day: di, text: d.title, field: "day" });
    for (const i of c.dayUnits[di]) if (match(c.units[i].title)) units.push({ day: di, unit: i, text: c.units[i].title, field: "unit" });
    for (const o of d.owns) if (match(o)) topics.push({ day: di, text: o, field: "topic" });
  });
  return [...days, ...units, ...topics].slice(0, limit);
}
