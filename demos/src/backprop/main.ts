// backprop: a tiny network drawn as a computational graph. "forward" fills in the values
// node by node (blue), "backward" the gradients ∂L/∂node from the loss back to the weights
// (red), with each edge's LOCAL derivative; "update" takes one gradient step and shows the loss
// before and after; "check" compares every gradient with a finite difference.
//
// <div data-demo="backprop" data-preset="byhand"></div>
//   data-preset  byhand | dead | sigmoid | toolarge | two      data-net  111 | 221
import { autoMount, button, htmlEl, presets as presetRow, svgEl, releaseFocus } from "../lib/ui";
import { finiteDiff, type Node } from "../lib/autodiff";
import { build, lossOf, update, PARAMS, INPUTS, PRESETS, type Net, type Phi, type Values, type Built } from "./model";

const NAME = "backprop";
const STYLE = `
[data-demo="${NAME}"] { flex-direction: column; gap: .4em; }
[data-demo="${NAME}"] .bp-svg { width: 100%; max-width: 1120px; }
[data-demo="${NAME}"] .bp-row { display: flex; flex-wrap: wrap; gap: .35em .8em; align-items: center; }
[data-demo="${NAME}"] .bp-in { display: inline-flex; align-items: center; gap: .25em; font-family: ui-monospace, monospace; }
[data-demo="${NAME}"] .bp-in input { width: 4.2em; font: inherit; padding: .1em .2em; }
[data-demo="${NAME}"] .qb-readout { font-size: 16px; white-space: pre; overflow-x: auto; }
[data-demo="${NAME}"] svg text { font-size: 19px; }
.reveal [data-demo="${NAME}"] { font-size: 18px; gap: .25em; }
.reveal [data-demo="${NAME}"] .bp-svg { max-height: 300px; }
.reveal [data-demo="${NAME}"] .qb-readout { font-size: 14px; padding: .2em .5em; }
.reveal [data-demo="${NAME}"] .qb-btn { font-size: 16px; padding: .1em .45em; }
`;
function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const W = 1120, BLUE = "#1f63b4", RED = "#c62828";
const fmt = (v: number) => (Math.abs(v) < 5e-4 && v !== 0 ? v.toExponential(1) : (Math.round(v * 1000) / 1000).toString());

function mount(host: HTMLElement): void {
  injectStyle();
  const start = PRESETS.find(p => p.id === host.dataset.preset) ?? PRESETS.find(p => p.net === host.dataset.net) ?? PRESETS[0];
  let net: Net = start.net, phi: Phi = start.phi, vals: Values = { ...start.v }, eta = start.eta;
  let b!: Built, order: Node[] = [], back: Node[] = [];
  let shownF = 0, shownB = 0, timer = 0, lastUpdate = "", checkTxt = "";

  const H = () => (net === "111" ? 400 : 520);
  const svg = svgEl("svg", { class: "bp-svg", role: "img", "aria-label": "Computational graph of a small network" }, host);
  const ctl = htmlEl("div", "bp-row", "", host);
  const inRow = htmlEl("div", "bp-row", "", host);
  const btnRow = htmlEl("div", "bp-row", "", host);
  const readout = htmlEl("div", "qb-readout", "", host);
  const presetBtns = presetRow(host, PRESETS.map(p => ({ label: p.label, apply: () => applyPreset(p) })));

  // activation buttons
  htmlEl("span", "", "φ:", ctl);
  const phiBtns = (["relu", "sigmoid"] as Phi[]).map(p => button(ctl, p === "relu" ? "ReLU" : "sigmoid", () => { phi = p; rebuild(); }));
  const netBtns = (["111", "221"] as Net[]).map(n => button(ctl, n === "111" ? "1–1–1" : "2–2–1", () => {
    const p = PRESETS.find(q => q.net === n)!; applyPreset(p);
  }));

  button(btnRow, "forward", () => animate("f"));
  button(btnRow, "backward", () => animate("b"));
  button(btnRow, "update", () => doUpdate());
  button(btnRow, "check (finite differences)", () => doCheck());
  button(btnRow, "reset", () => { const p = PRESETS.find(q => q.net === net)!; applyPreset(PRESETS.find(q => q.id === host.dataset.preset && q.net === net) ?? p); });

  function inputs(): void {
    inRow.replaceChildren();
    const names = [...INPUTS[net], ...PARAMS[net], "η"];
    for (const n of names) {
      const lab = htmlEl("label", "bp-in", "", inRow);
      htmlEl("span", "", n + " =", lab);
      const inp = htmlEl("input", "", "", lab);
      inp.type = "number"; inp.step = "0.1";
      inp.value = String(n === "η" ? eta : vals[n]);
      releaseFocus(inp);
      inp.addEventListener("change", () => {
        const v = Number(inp.value);
        if (!Number.isFinite(v)) return;
        if (n === "η") eta = v; else vals[n] = v;
        lastUpdate = ""; checkTxt = "";
        rebuild(false);
      });
    }
  }

  function rebuild(resetInputs = true): void {
    clearTimeout(timer);
    b = build(net, vals, phi);
    order = b.g.order.filter(n => n.op !== "const");
    back = b.g.backward(b.L);
    shownF = order.length; shownB = 0;   // values visible; "forward" replays them
    if (resetInputs) inputs();
    phiBtns.forEach((x, i) => x.classList.toggle("active", (["relu", "sigmoid"][i]) === phi));
    netBtns.forEach((x, i) => x.classList.toggle("active", (["111", "221"][i]) === net));
    draw();
  }

  function animate(kind: "f" | "b"): void {
    clearTimeout(timer);
    if (kind === "f") { shownF = 0; shownB = 0; }
    else { shownF = order.length; shownB = 0; }
    const stepF = () => { shownF++; draw(); if (shownF < order.length) timer = window.setTimeout(stepF, 280); };
    const stepB = () => { shownB++; draw(); if (shownB < back.length) timer = window.setTimeout(stepB, 330); };
    timer = window.setTimeout(kind === "f" ? stepF : stepB, 50);
  }

  function doUpdate(): void {
    clearTimeout(timer);
    const before = b.L.value, { next, grads } = update(net, vals, phi, eta), after = lossOf(net, next, phi);
    lastUpdate = `update with η = ${eta}:  ` + PARAMS[net].map(p => `${p} ${fmt(vals[p])} → ${fmt(next[p])}`).join(",  ") +
      `\nloss before ${fmt(before)}, after ${fmt(after)}` + (after > before ? "   (the loss went UP: η too large)" : "");
    void grads;
    vals = next; checkTxt = "";
    rebuild();
    shownF = order.length; draw();
  }

  function doCheck(): void {
    const names = PARAMS[net];
    const f = (arr: number[]) => { const v = { ...vals }; names.forEach((n, i) => { v[n] = arr[i]; }); return lossOf(net, v, phi); };
    const x0 = names.map(n => vals[n]);
    checkTxt = "gradient check:  parameter  backprop  finite difference\n" + names.map((n, i) => {
      const bp = b.params[i].grad, fd = finiteDiff(f, x0, i, 1e-5);
      return `   ${n.padEnd(4)} ${fmt(bp).padStart(10)} ${fmt(fd).padStart(12)}`;
    }).join("\n");
    shownF = order.length; shownB = back.length;
    draw();
  }

  function applyPreset(p: (typeof PRESETS)[number]): void {
    net = p.net; phi = p.phi; vals = { ...p.v }; eta = p.eta; lastUpdate = ""; checkTxt = "";
    presetBtns.forEach((x, i) => x.classList.toggle("active", PRESETS[i] === p));
    rebuild();
  }

  function draw(): void {
    const h = H();
    svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
    svg.replaceChildren();
    const maxC = Math.max(...[...b.pos.values()].map(p => p[0])), maxR = Math.max(...[...b.pos.values()].map(p => p[1]));
    const X = (c: number) => 60 + c * (W - 190) / maxC, Y = (r: number) => 40 + r * (h - 110) / maxR;
    const shownFwd = new Set(order.slice(0, shownF)), shownBwd = new Set(back.slice(0, shownB));
    const defs = svgEl("defs", {}, svg);
    const mk = svgEl("marker", { id: "bp-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: "#888" }, mk);
    const R = 30;
    // edges
    for (const n of order) for (const p of n.parents) {
      const [c0, r0] = b.pos.get(p.node)!, [c1, r1] = b.pos.get(n)!;
      const x0 = X(c0), y0 = Y(r0), x1 = X(c1), y1 = Y(r1), d = Math.hypot(x1 - x0, y1 - y0);
      const ux = (x1 - x0) / d, uy = (y1 - y0) / d;
      svgEl("line", { x1: x0 + ux * R, y1: y0 + uy * R, x2: x1 - ux * (R + 3), y2: y1 - uy * (R + 3), stroke: "#999", "stroke-width": 2, "marker-end": "url(#bp-arrow)" }, svg);
      if (shownBwd.has(n)) {
        const t = svgEl("text", { x: (x0 + x1) / 2 + 12 * uy, y: (y0 + y1) / 2 - 12 * ux + 5, "text-anchor": "middle", fill: "#8a4b00", class: "qb-halo", style: "font-size:17px" }, svg);
        t.textContent = "×" + fmt(p.local);
      }
    }
    // nodes
    for (const n of order) {
      const [c, r] = b.pos.get(n)!, x = X(c), y = Y(r);
      const fill = n.kind === "param" ? "#fde9c8" : n.kind === "input" ? "#e8e8e8" : n === b.L ? "#f6d2d2" : "#dbe8f6";
      if (n.kind === "op") svgEl("circle", { cx: x, cy: y, r: R, fill, stroke: "#555", "stroke-width": 2 }, svg);
      else svgEl("rect", { x: x - R, y: y - 22, width: 2 * R, height: 44, rx: 8, fill, stroke: "#555", "stroke-width": 2 }, svg);
      const t = svgEl("text", { x, y: y + 7, "text-anchor": "middle", style: "font-size:20px;font-weight:600" }, svg);
      t.textContent = n.name;
      if (n.kind === "op") {
        const op = svgEl("text", { x, y: y - R - 8, "text-anchor": "middle", fill: "#666", style: "font-size:15px" }, svg);
        op.textContent = n.op === "affine" ? "Σ w·in + b" : n.op === "sqerr" ? "(ŷ − y)²" : n.op === "relu" ? "ReLU" : n.op;
      }
      if (shownFwd.has(n)) {
        const v = svgEl("text", { x: x + R + 4, y: y + (n.kind === "op" ? -2 : 0), fill: BLUE, class: "qb-halo", style: "font-size:18px" }, svg);
        v.textContent = "= " + fmt(n.value);
      }
      if (shownBwd.has(n) && n.kind !== "input") {
        const g = svgEl("text", { x: x + R + 4, y: y + 20, fill: RED, class: "qb-halo", style: "font-size:18px" }, svg);
        g.textContent = "∂L/∂ = " + fmt(n.grad);
      }
    }
    // legend
    const lg = svgEl("text", { x: 12, y: h - 12, style: "font-size:16px", fill: "#555" }, svg);
    lg.innerHTML = `<tspan fill="${BLUE}">blue: value (forward)</tspan>   <tspan fill="${RED}">red: gradient ∂L/∂node (backward)</tspan>   <tspan fill="#8a4b00">×a on an edge: local derivative</tspan>`;

    let txt = `loss L = ${shownF >= order.length ? fmt(b.L.value) : "?"}`;
    if (lastUpdate) txt += "\n" + lastUpdate;
    if (checkTxt) txt += "\n" + checkTxt;
    readout.textContent = txt;
  }

  applyPreset(start);
}

autoMount(NAME, mount);
