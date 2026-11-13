// Data for the ws-flooding figures: Meyer flooding traces from the course's lib/watershed.ts
// on the floodgrid demo's grids (copied from demos/src/floodgrid/main.ts).
import { writeFileSync } from "fs";
import { watershedTrace } from "../../../demos/src/lib/watershed";

const G = (s: string) => s.trim().split("\n").map(r => r.trim().split(/\s+/).map(Number));
const HAND = G(`
3 2 3 5 6
2 1 2 4 5
3 2 4 3 2
5 4 3 2 1
6 5 3 1 0`);
const PLATEAU = G(`
6 6 6 6 6 6 6 6 6
6 3 3 3 3 3 3 3 6
1 3 3 3 3 3 3 3 1
6 3 3 3 3 3 3 3 6
6 6 6 6 6 6 6 6 6`);

function run(grid: number[][], marks: [number, number, number][], conn: 4 | 8, lifo = false) {
  const R = grid.length, C = grid[0].length, f = grid.flat(), m = new Int32Array(R * C);
  for (const [r, c, l] of marks) m[r * C + c] = l;
  const tr = watershedTrace(f, C, R, m, { conn, lines: true, lifo });
  return {
    rows: R, cols: C, f, markers: Array.from(m),
    init: { labels: Array.from(tr.init.labels), queue: tr.init.queue },
    steps: tr.steps.map(s => ({ item: s.item, label: s.label, level: s.level, labels: Array.from(s.labels), queue: s.queue, nb: s.nbPixels, pushed: s.pushed })),
    labels: Array.from(tr.labels),
  };
}
writeFileSync(process.argv[2], JSON.stringify({
  hand4: run(HAND, [[1, 1, 1], [4, 4, 2]], 4),
  plateauFifo: run(PLATEAU, [[2, 0, 1], [2, 8, 2]], 4),
  plateauLifo: run(PLATEAU, [[2, 0, 1], [2, 8, 2]], 4, true),
}));
