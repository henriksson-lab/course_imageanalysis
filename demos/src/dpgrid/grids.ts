// Cost grids of the dpgrid demo (no DOM; also used by demos/test/dynprog-test.ts and by the
// scripts that compute the exercise solutions). Costs are integers 0..9 ("dark = cheap").

export type Source = "row" | "corner";
/** Where the path ends: the cheapest cell of the bottom row, or a corner. */
export type Target = "row" | "br" | "bl";

export interface GridPreset {
  id: string;
  label: string;
  rows: number;
  cols: number;
  values: number[];
  source?: Source;
  target?: Target;
}

const parse = (id: string, label: string, rows: string[], source?: Source, target?: Target): GridPreset => {
  const a = rows.map(r => r.trim().split(/\s+/).map(Number));
  return { id, label, rows: a.length, cols: a[0].length, values: a.flat(), source, target };
};

/** Unit 1 (dp-paths/): monotone paths from the top row to the bottom row. */
export const DPGRID_PRESETS: GridPreset[] = [
  // the by-hand grid of the dp-paths exercise (B1): DP 10, greedy 19
  parse("exercise", "by-hand exercise grid", ["7 4 8 8 3", "2 9 3 8 2", "3 1 3 6 8", "5 1 7 3 2", "9 2 3 9 6"]),
  // the cheapest first cell (1) leads into a field of 9s: greedy 20, DP 9
  parse("greedy", "greedy fails", ["5 4 1 6 7 3", "8 9 2 9 9 2", "7 9 9 8 9 1", "2 9 9 9 8 2", "3 9 8 9 9 1"]),
  // two mirror-image paths of cost 9
  parse("equal", "two equal optima", ["9 9 1 9 9", "9 2 9 2 9", "3 9 9 9 3", "9 2 9 2 9", "9 9 1 9 9"]),
  // the cheap channel turns back up: no monotone path can follow it (DP 24, Dijkstra 19)
  parse("detour", "a detour", ["9 1 9 9 9 9 9 9", "9 1 9 9 9 9 9 9", "9 1 9 1 1 1 1 9", "9 1 9 1 9 9 1 9",
    "9 1 9 1 9 9 1 9", "9 1 1 1 9 9 1 9", "9 9 9 9 9 9 1 9", "9 9 9 9 9 9 1 9"]),
];

/** Unit 3 (livewire/): Dijkstra on the grid, 4 neighbours, moves in every direction. */
export const LW_PRESETS: GridPreset[] = [
  { ...DPGRID_PRESETS[3], id: "detour", label: "the detour" },
  // the by-hand grid of the livewire exercise (B1): from the top-left corner
  parse("exercise", "by-hand exercise grid", ["1 7 8 9", "1 5 7 5", "1 3 5 4", "1 5 3 6"], "corner", "br"),
  // a wall of 9s with one gap at the far end: the path goes right, through the gap, and back left
  parse("wall", "a wall with one gap", ["1 1 1 1 1 1 1 1", "1 1 1 1 1 1 1 1", "9 9 9 9 9 9 9 1", "1 1 1 1 1 1 1 1",
    "1 1 1 1 1 1 1 1", "1 1 1 1 1 1 1 1"], "corner", "bl"),
];
