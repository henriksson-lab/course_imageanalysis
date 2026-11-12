// The networks of the backprop demo as computational graphs (lib/autodiff.ts), with fixed
// drawing positions. No DOM, so the exercise numbers come from exactly this code.
//
// "111":  x → z = w1·x + b1 → h = φ(z) → ŷ = w2·h + b2 → L = (ŷ − y)²
// "221":  (x1, x2) → z_j = w_j1 x1 + w_j2 x2 + b_j → h_j = φ(z_j) (j = 1, 2)
//         → ŷ = v1 h1 + v2 h2 + c → L = (ŷ − y)²
import { Graph, type Node } from "../lib/autodiff";

export type Net = "111" | "221";
export type Phi = "relu" | "sigmoid";

export interface Built { g: Graph; L: Node; params: Node[]; pos: Map<Node, [number, number]> }

/** Values of inputs and parameters by name (x, y, w1, b1, ... ). */
export type Values = Record<string, number>;

export const PARAMS: Record<Net, string[]> = {
  "111": ["w1", "b1", "w2", "b2"],
  "221": ["w11", "w12", "b1", "w21", "w22", "b2", "v1", "v2", "c"],
};
export const INPUTS: Record<Net, string[]> = { "111": ["x", "y"], "221": ["x1", "x2", "y"] };

export function build(net: Net, v: Values, phi: Phi): Built {
  const g = new Graph(), pos = new Map<Node, [number, number]>();
  const at = (n: Node, c: number, r: number) => { pos.set(n, [c, r]); return n; };
  const act = (z: Node, name: string) => (phi === "relu" ? g.relu(z, name) : g.sigmoid(z, name));
  if (net === "111") {
    const x = at(g.input("x", v.x), 0, 0.2), w1 = at(g.param("w1", v.w1), 0, 1.4), b1 = at(g.param("b1", v.b1), 0, 2.6);
    const z = at(g.affine([w1], [x], b1, "z"), 1, 1.4);
    const h = at(act(z, "h"), 2, 1.4);
    const w2 = at(g.param("w2", v.w2), 2, 0.2), b2 = at(g.param("b2", v.b2), 2, 2.6);
    const yh = at(g.affine([w2], [h], b2, "ŷ"), 3, 1.4);
    const y = at(g.input("y", v.y), 3, 2.6);
    const L = at(g.sqerr(yh, y, "L"), 4, 1.4);
    return { g, L, params: [w1, b1, w2, b2], pos };
  }
  const x1 = at(g.input("x1", v.x1), 0, 0.9), x2 = at(g.input("x2", v.x2), 0, 2.9);
  const w11 = at(g.param("w11", v.w11), 0.85, 0), w12 = at(g.param("w12", v.w12), 0.85, 0.9), b1 = at(g.param("b1", v.b1), 0.85, 1.8);
  const w21 = at(g.param("w21", v.w21), 0.85, 2.9), w22 = at(g.param("w22", v.w22), 0.85, 3.8), b2 = at(g.param("b2", v.b2), 0.85, 4.7);
  const z1 = at(g.affine([w11, w12], [x1, x2], b1, "z1"), 1.8, 0.9), z2 = at(g.affine([w21, w22], [x1, x2], b2, "z2"), 1.8, 3.8);
  const h1 = at(act(z1, "h1"), 2.65, 0.9), h2 = at(act(z2, "h2"), 2.65, 3.8);
  const v1 = at(g.param("v1", v.v1), 2.65, 0), v2 = at(g.param("v2", v.v2), 2.65, 4.7), c = at(g.param("c", v.c), 3.3, 3.2);
  const yh = at(g.affine([v1, v2], [h1, h2], c, "ŷ"), 3.5, 2.35);
  const y = at(g.input("y", v.y), 4.25, 3.6);
  const L = at(g.sqerr(yh, y, "L"), 4.4, 2.35);
  return { g, L, params: [w11, w12, b1, w21, w22, b2, v1, v2, c], pos };
}

/** Loss for given values (used for finite differences and "loss after the update"). */
export function lossOf(net: Net, v: Values, phi: Phi): number {
  return build(net, v, phi).L.value;
}

/** One gradient step: returns the new values and the gradients used. */
export function update(net: Net, v: Values, phi: Phi, eta: number): { next: Values; grads: Record<string, number> } {
  const b = build(net, v, phi);
  b.g.backward(b.L);
  const next = { ...v }, grads: Record<string, number> = {};
  for (const p of b.params) { grads[p.name] = p.grad; next[p.name] = v[p.name] - eta * p.grad; }
  return { next, grads };
}

export interface Preset { id: string; label: string; net: Net; phi: Phi; v: Values; eta: number }
export const PRESETS: Preset[] = [
  { id: "byhand", label: "the by-hand exercise", net: "111", phi: "relu", v: { x: 2, y: 1, w1: 0.5, b1: 0, w2: 1.5, b2: 0 }, eta: 0.1 },
  { id: "dead", label: "dead ReLU: no gradient upstream", net: "111", phi: "relu", v: { x: -1, y: 1, w1: 0.5, b1: 0, w2: 1.5, b2: 0 }, eta: 0.1 },
  { id: "sigmoid", label: "sigmoid saturates", net: "111", phi: "sigmoid", v: { x: 4, y: 0, w1: 2, b1: 0, w2: 1.5, b2: 0 }, eta: 0.1 },
  { id: "toolarge", label: "learning rate too large", net: "111", phi: "relu", v: { x: 2, y: 1, w1: 0.5, b1: 0, w2: 1.5, b2: 0 }, eta: 0.3 },
  { id: "two", label: "2–2–1 network", net: "221", phi: "relu", v: { x1: 1, x2: 2, y: 2, w11: 1, w12: 0.5, b1: 0.5, w21: 0.5, w22: 0.5, b2: -1, v1: 1, v2: 2, c: 0 }, eta: 0.05 },
];
