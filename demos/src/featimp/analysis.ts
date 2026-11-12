// The forest and both importances for the feature-importance demo (no DOM; also used by
// the solutions scripts).
import * as FO from "../lib/forest";
import { population, FEATURES } from "./population";

/** The size block: area, perimeter, equivalent diameter and the copy of area. */
export const SIZE_BLOCK = ["area", "perim", "eqd", "dup"];

export interface ImportanceResult {
  names: string[]; ids: string[];
  mdi: Float64Array; perm: Float64Array; oob: number;
  groupPerm: number | null;
}

/** Train the forest on the chosen columns and compute both importances (used by the demo and the solutions script). */
export function analyse(use: string[], dup: boolean, group: boolean, mtry: number, seed = 1): ImportanceResult {
  const pop = population({ seed: 1 });
  const ids = FEATURES.map(f => f.id).filter(id => use.includes(id));
  const cols = ids.map(id => pop.cols[FEATURES.findIndex(f => f.id === id)]);
  const names = ids.map(id => FEATURES.find(f => f.id === id)!.short);
  if (dup && ids.includes("area")) { ids.push("dup"); cols.push(pop.cols[0].slice()); names.push("area (copy)"); }
  const T = FO.tableFromColumns(cols), y = pop.y;
  const f = FO.trainForest(FO.binTable(T, 1024), FO.classTarget(y, 2), null, { nTrees: 100, mtry: Math.min(mtry, ids.length), seed });
  const mdi = FO.impurityImportance(f), s = mdi.reduce((a, b) => a + b, 0);
  for (let j = 0; j < mdi.length; j++) mdi[j] /= s || 1;
  const perm = FO.permutationImportance(f, T, y, { repeats: 5, seed: 3 });
  let groupPerm: number | null = null;
  const blk = ids.map((id, j) => (SIZE_BLOCK.includes(id) ? j : -1)).filter(j => j >= 0);
  if (group && blk.length) groupPerm = FO.permutationImportance(f, T, y, { repeats: 5, seed: 3, groups: [blk] })[0];
  return { names, ids, mdi, perm, oob: FO.oobError(f, T.X, y), groupPerm };
}

