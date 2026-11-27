// Data for the fft-docking/ figures, computed with the dock2d demo's own model so that the
// numbers on the slides match the demo. Called by figs.py:
//   esbuild data.ts --bundle --platform=node | node - <out.json>
import { writeFileSync } from "fs";
import { DEFAULTS, Settings, search, makeScene, toCell, toWorld, ligandAt, ligandPoly, TRUE_POSE, isTrueFit, BOX } from "../../../demos/src/dock2d/model";
import { Docker, scoreAt, rotatePolyCW } from "../../../demos/src/lib/docking";

const [outPath] = process.argv.slice(2);

function scene(p: Partial<Settings>) {
  const s = { ...DEFAULTS, ...p }, sc = makeScene(s);
  const { n, o, M } = sc;
  const crop = (a: ArrayLike<number>) => { const r: number[] = []; for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) r.push(a[(j + o) * M + i + o]); return r; };
  return { s, sc, crop, layer: crop(Array.from(sc.R.core, (c, i) => (c ? -1 : sc.R.surface[i] ? 1 : 0))) };
}

/** A pose drawn: its polygon (world units), its cells (box cell indices) and what they hit. */
function pose(sceneData: ReturnType<typeof scene>, angle: number, gx: number, gy: number) {
  const { s, sc } = sceneData, L = ligandAt(sc, s, angle);
  const [x, y] = toWorld(sc, gx, gy);
  const poly = rotatePolyCW(ligandPoly(s.bump), angle).map(([u, v]) => [u + x, v + y]);
  const cells: number[][] = [];
  for (let k = 0; k < L.dx.length; k++) {
    const cx = gx + L.dx[k], cy = gy + L.dy[k], g = cy * sc.M + cx;
    cells.push([cx - sc.o, cy - sc.o, sc.R.core[g] ? -1 : sc.R.surface[g] ? 1 : 0]);
  }
  return { angle, x, y, poly, cells, ...scoreAt(sc.a, sc.R, L, gx, gy) };
}

const out: Record<string, unknown> = { box: BOX, truePose: TRUE_POSE };

// key and lock: layers, the true pose, maps at the right and a wrong angle, the search
{
  const d = scene({}), { sc, s } = d, [gx, gy] = toCell(sc, TRUE_POSE.x, TRUE_POSE.y);
  const dk = new Docker(sc.a, sc.M);
  const maps: Record<string, number[]> = {};
  for (const a of [20, 110]) maps[a] = d.crop(dk.map(ligandAt(sc, s, a)));
  const r = search(s);
  out.key = {
    n: sc.n, layer: d.layer, rho: s.rho, thickness: s.thickness,
    truePose: pose(d, TRUE_POSE.angle, gx, gy), apart: pose(d, 0, ...toCell(sc, 14, 12)),
    deeper: pose(d, TRUE_POSE.angle, gx, gy + 2), shifted: pose(d, TRUE_POSE.angle, gx + 3, gy - 2),
    maps, perAngle: r.results.map(q => [q.angle, q.best]), ffts: r.dk.ffts,
    top: r.top.map(p => ({ ...pose(d, p.angle, p.x, p.y), score: p.score, isTrue: isTrueFit(sc, p) })),
  };
}
// ρ too mild: the best pose sinks in
{
  const d = scene({ rho: -0.1 }), r = search(d.s), b = r.top[0];
  out.mild = { n: d.sc.n, layer: d.layer, best: { ...pose(d, b.angle, b.x, b.y), score: b.score }, rank: r.top.findIndex(p => isTrueFit(d.sc, p)) + 1 };
}
// coarse angle step
{
  const d = scene({ step: 30 }), r = search(d.s);
  out.coarse = { perAngle: r.results.map(q => [q.angle, q.best]), best: r.top[0] && { ...pose(d, r.top[0].angle, r.top[0].x, r.top[0].y) } };
}
// the flexible bump: the true pose clashes; a softer core brings it back
{
  const d = scene({ bump: true }), [gx, gy] = toCell(d.sc, TRUE_POSE.x, TRUE_POSE.y), r = search(d.s);
  const soft = scene({ bump: true, rho: -1 }), rs = search(soft.s);
  out.bump = {
    n: d.sc.n, layer: d.layer, truePose: pose(d, TRUE_POSE.angle, gx, gy),
    best: { ...pose(d, r.top[0].angle, r.top[0].x, r.top[0].y), score: r.top[0].score },
    rank: r.top.findIndex(p => isTrueFit(d.sc, p)) + 1,
    soft: { best: { ...pose(soft, rs.top[0].angle, rs.top[0].x, rs.top[0].y), score: rs.top[0].score }, isTrue: isTrueFit(soft.sc, rs.top[0]) },
  };
}
writeFileSync(outPath, JSON.stringify(out));
