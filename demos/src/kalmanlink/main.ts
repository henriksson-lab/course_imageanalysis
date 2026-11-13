// Linking with prediction. 2D particles with nearly constant velocity (some crossing), with
// localisation noise and missed detections. Three trackers: nearest neighbour on the last
// position, global assignment on the last position (fixed gate radius), and global assignment
// on the position predicted by a constant-velocity Kalman filter per track (gate = the
// predicted covariance ellipse at g standard deviations; cost = Mahalanobis d² + log det S).
// Tracks survive up to two missed frames (predict only). The field shows the current frame,
// track tails, each track's predicted position and gate. Readouts: link error rate and
// identity switches of all three trackers.
//
// <div data-demo="kalmanlink" data-preset="cross|stream|turn|missed|mixed"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { Plot } from "../lib/plot";
import { labelColor, toCss } from "../lib/colour";
import * as LK from "../lib/linking";
import { T, W, State, Method, DEFAULT, PRESETS, METHODS, Result, simulate, track, switches } from "./model";

const NAME = "kalmanlink";
const S = `.qb-demo[data-demo="${NAME}"]`;
const STYLE = `
${S} { gap: .6em; }
${S} .kl-field { flex: 1 1 460px; min-width: 260px; max-width: 540px; }
${S} .qb-panel { flex: 1 1 320px; max-width: 420px; gap: .3em; }
${S} .kl-row { display: flex; flex-wrap: wrap; gap: .3em; align-items: center; }
${S} .qb-readout { font-size: 17px; white-space: pre; overflow-x: auto; }
${S} .qb-presets { flex-basis: 100%; }
${S} .qb-btn.kl-on { background: var(--qb-accent); border-color: var(--qb-accent); color: #fff; }
.reveal ${S} { font-size: 18px; }
.reveal ${S} .kl-field { flex: 0 0 450px; }
.reveal ${S} .qb-panel { flex: 0 0 600px; max-width: 620px; gap: .1em; display: grid; grid-template-columns: 1fr 1fr; column-gap: .8em; align-items: start; }
.reveal ${S} .qb-panel .qb-readout, .reveal ${S} .qb-panel .kl-row { grid-column: 1 / 3; }
.reveal ${S} .qb-btn { font-size: 18px; padding: .02em .35em; }
.reveal ${S} .qb-readout { font-size: 18px; padding: .2em .4em; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME; s.textContent = STYLE;
  document.head.appendChild(s);
}

autoMount(NAME, host => {
  injectStyle();
  let st: State = { ...DEFAULT };
  const fieldBox = htmlEl("div", "kl-field", "", host);
  const plot = new Plot(fieldBox, { width: 520, height: 520, x: [0, W], y: [0, W], equal: true, xticks: [0, 128, 256], yticks: [0, 128, 256], margin: { l: 60, b: 40, t: 8, r: 10 }, font: 24, ariaLabel: "particles, tracks and gates" });
  const panel = htmlEl("div", "qb-panel", "", host);
  const read = htmlEl("div", "qb-readout", "", panel);
  const methRow = htmlEl("div", "kl-row", "", panel);
  const methBtns = METHODS.map(([m, l]) => button(methRow, l, () => { st.method = m; draw(); }));
  const sl = {
    n: slider(panel, "particles", 2, 80, 2, st.n, v => { st.n = v; recompute(); }),
    speed: slider(panel, "speed (px/frame)", 1, 12, 0.5, st.speed, v => { st.speed = v; recompute(); }, v => v.toFixed(1)),
    sigma: slider(panel, "noise σ (px)", 0.3, 6, 0.1, st.sigma, v => { st.sigma = v; recompute(); }, v => v.toFixed(1)),
    miss: slider(panel, "miss probability", 0, 0.4, 0.01, st.miss, v => { st.miss = v; recompute(); }, v => v.toFixed(2)),
    Q: slider(panel, "Q (velocity var.)", 0.001, 1, 0.001, st.Q, v => { st.Q = v; recompute(); }, v => v.toFixed(3)),
    gate: slider(panel, "Kalman gate (σ)", 1.5, 6, 0.25, st.gate, v => { st.gate = v; recompute(); }, v => v.toFixed(2)),
    radius: slider(panel, "fixed gate (px)", 3, 40, 1, st.radius, v => { st.radius = v; recompute(); }),
  };
  const tRow = htmlEl("div", "kl-row", "", panel);
  let timer = 0;
  const playBtn = button(tRow, "play", () => {
    if (timer) { stop(); return; }
    playBtn.textContent = "pause";
    timer = window.setInterval(() => { st.t = (st.t + 1) % T; tS.set(st.t); draw(); }, 160);
  });
  const stop = () => { window.clearInterval(timer); timer = 0; playBtn.textContent = "play"; };
  const tS = slider(tRow, "frame", 1, T - 1, 1, st.t, v => { st.t = v; draw(); });
  presets(host, [...PRESETS.map(p => ({ label: p.label, apply: () => apply(p.s) })), { label: "reset", apply: () => apply({}) }]);

  let mv = simulate(st);
  let res: Record<Method, Result> = { nn: track(mv, st, "nn"), last: track(mv, st, "last"), kalman: track(mv, st, "kalman") };
  let wrong: Record<Method, boolean[]> = { nn: [], last: [], kalman: [] };
  let summary = "";

  function apply(p: Partial<State>): void {
    stop();
    st = { ...DEFAULT, ...p };
    for (const k of Object.keys(sl) as (keyof typeof sl)[]) sl[k].set(st[k]);
    tS.set(st.t);
    recompute();
  }
  function recompute(): void {
    mv = simulate(st);
    const rows: string[] = ["tracker          error  switches"];
    for (const [m, l] of METHODS) {
      res[m] = track(mv, st, m);
      const ev = LK.evalLinks(res[m].edges, mv.truth);
      wrong[m] = ev.wrong;
      rows.push(`${l.padEnd(15)} ${(100 * (1 - ev.recall)).toFixed(1).padStart(5)} %  ${String(switches(mv, res[m].trackOf)).padStart(5)}`);
    }
    summary = rows.join("\n");
    draw();
  }

  function draw(): void {
    methBtns.forEach((b, i) => b.classList.toggle("kl-on", METHODS[i][0] === st.method));
    const r = res[st.method], t = st.t, tail = 12;
    plot.clear();
    const col = (id: number) => toCss(labelColor(1 + (id % 97)));
    // gates of the tracks at frame t
    for (const g of r.gates) if (g.t === t) {
      const nsd = st.method === "kalman" ? st.gate : 1;
      plot.ellipse(g.mean, g.cov, nsd, { color: col(g.track), width: 1.5, fill: col(g.track), fillOpacity: 0.06, dash: st.method === "kalman" ? undefined : "4 3" });
      plot.scatter([g.mean[0]], [g.mean[1]], { color: col(g.track), r: 2.5 });
    }
    // tails
    r.edges.forEach((e, i) => {
      const a = mv.dets[e.a], b = mv.dets[e.b];
      if (b.t > t || b.t <= t - tail) return;
      plot.line([a.x, b.x], [a.y, b.y], { color: wrong[st.method][i] ? "#d62728" : col(r.trackOf[e.b]), width: wrong[st.method][i] ? 3 : 2, opacity: 0.4 + 0.6 * (1 - (t - b.t) / tail) });
    });
    // detections at t
    const xs: number[] = [], ys: number[] = [], cs: string[] = [];
    mv.dets.forEach((d, k) => { if (d.t === t) { xs.push(d.x); ys.push(d.y); cs.push(col(r.trackOf[k])); } });
    plot.scatter(xs, ys, { colors: cs, r: 5, stroke: "#222" });
    plot.text(6, W - 14, `frame ${t}`, { size: 22 });
    read.textContent = summary + "\n(red line = wrong link)";
  }

  const pre = host.dataset.preset ? PRESETS.find(p => p.id === host.dataset.preset) : undefined;
  apply(pre ? pre.s : {});
});
