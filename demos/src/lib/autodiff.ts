// Scalar reverse-mode automatic differentiation that also records the computational graph,
// so a demo can draw it (owner: neural-networks day; used by the backprop demo and the tests
// of lib/nn.ts). Not used for training (one node per number is slow), but exact.
//
//   const g = new Graph();
//   const x = g.input("x", 2), w = g.param("w", 0.5);
//   const L = g.square(g.sub(g.mul(w, x), g.input("y", 1)), "L");
//   g.backward(L);            // every node now has .grad = ∂L/∂node
//
// Each node stores its parents with the LOCAL derivative ∂node/∂parent (computed in the
// forward pass); backward multiplies the incoming gradient by it and adds it to the parent:
// the chain rule, organised. No DOM.

export type NodeKind = "input" | "param" | "op";

export interface Node {
  id: number;
  name: string;
  kind: NodeKind;
  /** Operation ("input", "param", "add", "mul", "affine", "relu", ...). */
  op: string;
  value: number;
  grad: number;
  parents: { node: Node; local: number }[];
}

export class Graph {
  nodes: Node[] = [];

  private add(name: string, kind: NodeKind, op: string, value: number, parents: { node: Node; local: number }[] = []): Node {
    const n: Node = { id: this.nodes.length, name, kind, op, value, grad: 0, parents };
    this.nodes.push(n);
    return n;
  }
  input(name: string, v: number): Node { return this.add(name, "input", "input", v); }
  param(name: string, v: number): Node { return this.add(name, "param", "param", v); }
  const(v: number): Node { return this.add(String(v), "input", "const", v); }

  add2(a: Node, b: Node, name = ""): Node { return this.add(name, "op", "add", a.value + b.value, [{ node: a, local: 1 }, { node: b, local: 1 }]); }
  sub(a: Node, b: Node, name = ""): Node { return this.add(name, "op", "sub", a.value - b.value, [{ node: a, local: 1 }, { node: b, local: -1 }]); }
  mul(a: Node, b: Node, name = ""): Node { return this.add(name, "op", "mul", a.value * b.value, [{ node: a, local: b.value }, { node: b, local: a.value }]); }
  /** Σ w_i x_i + b in one node (a neuron's weighted sum). */
  affine(ws: Node[], xs: Node[], b: Node | null, name = ""): Node {
    let v = b ? b.value : 0;
    const parents: { node: Node; local: number }[] = [];
    ws.forEach((w, i) => { v += w.value * xs[i].value; parents.push({ node: w, local: xs[i].value }); });
    xs.forEach((x, i) => parents.push({ node: x, local: ws[i].value }));
    if (b) parents.push({ node: b, local: 1 });
    return this.add(name, "op", "affine", v, parents);
  }
  relu(a: Node, name = ""): Node { return this.add(name, "op", "relu", Math.max(0, a.value), [{ node: a, local: a.value > 0 ? 1 : 0 }]); }
  sigmoid(a: Node, name = ""): Node { const s = 1 / (1 + Math.exp(-a.value)); return this.add(name, "op", "sigmoid", s, [{ node: a, local: s * (1 - s) }]); }
  tanh(a: Node, name = ""): Node { const t = Math.tanh(a.value); return this.add(name, "op", "tanh", t, [{ node: a, local: 1 - t * t }]); }
  square(a: Node, name = ""): Node { return this.add(name, "op", "square", a.value * a.value, [{ node: a, local: 2 * a.value }]); }
  /** (a − b)², the squared error, as one node. */
  sqerr(a: Node, b: Node, name = ""): Node {
    const e = a.value - b.value;
    return this.add(name, "op", "sqerr", e * e, [{ node: a, local: 2 * e }, { node: b, local: -2 * e }]);
  }
  exp(a: Node, name = ""): Node { const e = Math.exp(a.value); return this.add(name, "op", "exp", e, [{ node: a, local: e }]); }
  log(a: Node, name = ""): Node { return this.add(name, "op", "log", Math.log(a.value), [{ node: a, local: 1 / a.value }]); }
  scale(a: Node, c: number, name = ""): Node { return this.add(name, "op", "scale", c * a.value, [{ node: a, local: c }]); }
  sum(xs: Node[], name = ""): Node { return this.add(name, "op", "sum", xs.reduce((s, x) => s + x.value, 0), xs.map(x => ({ node: x, local: 1 }))); }
  /** −log softmax(z)[k]: the cross-entropy of scores z with true class k, as one node. */
  softmaxCE(z: Node[], k: number, name = ""): Node {
    const m = Math.max(...z.map(n => n.value)), e = z.map(n => Math.exp(n.value - m)), s = e.reduce((a, b) => a + b, 0);
    const p = e.map(v => v / s);
    return this.add(name, "op", "softmaxCE", -Math.log(p[k]), z.map((n, i) => ({ node: n, local: p[i] - (i === k ? 1 : 0) })));
  }

  /** Nodes in the order they were created (a valid topological order: parents come first). */
  get order(): Node[] { return this.nodes; }

  /** Reverse pass from `out`: sets every node's grad to ∂out/∂node. Returns the nodes in the
   *  order their gradient became final (reverse topological order). */
  backward(out: Node): Node[] {
    for (const n of this.nodes) n.grad = 0;
    out.grad = 1;
    const done: Node[] = [];
    for (let i = out.id; i >= 0; i--) {
      const n = this.nodes[i];
      done.push(n);
      for (const p of n.parents) p.node.grad += n.grad * p.local;
    }
    return done;
  }
}

/** Central finite difference of f at x along coordinate i (for gradient checks). */
export function finiteDiff(f: (x: number[]) => number, x: number[], i: number, h = 1e-5): number {
  const a = x.slice(), b = x.slice();
  a[i] += h; b[i] -= h;
  return (f(a) - f(b)) / (2 * h);
}
