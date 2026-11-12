// Preset graphs for the maxflow demo. No DOM: also read by the test and the by-hand checks.
// Node 0 is always the source s, node 1 the sink t. Positions are in the demo's SVG units.
import { Edge } from "../lib/maxflow";

export interface Preset {
  id: string; label: string;
  nodes: { name: string; x: number; y: number }[];
  edges: Edge[];
}

const E = (u: number, v: number, c: number): Edge => ({ u, v, c });

export const PRESETS: Preset[] = [
  {
    id: "textbook", label: "example",
    nodes: [
      { name: "s", x: 60, y: 230 }, { name: "t", x: 740, y: 230 },
      { name: "a", x: 270, y: 90 }, { name: "b", x: 270, y: 370 },
      { name: "c", x: 530, y: 90 }, { name: "d", x: 530, y: 370 },
    ],
    // s=0 t=1 a=2 b=3 c=4 d=5
    edges: [E(0, 2, 10), E(0, 3, 10), E(2, 3, 2), E(2, 4, 4), E(2, 5, 8), E(3, 5, 9), E(5, 4, 6), E(4, 1, 10), E(5, 1, 10)],
  },
  {
    id: "backedge", label: "needs a back-edge",
    nodes: [
      { name: "s", x: 60, y: 240 }, { name: "t", x: 740, y: 240 },
      { name: "a", x: 220, y: 130 }, { name: "b", x: 560, y: 240 },
      { name: "c", x: 220, y: 380 }, { name: "g", x: 420, y: 380 },
      { name: "d", x: 380, y: 50 }, { name: "e", x: 600, y: 60 },
    ],
    // s=0 t=1 a=2 b=3 c=4 g=5 d=6 e=7
    edges: [E(0, 2, 4), E(2, 3, 4), E(3, 1, 4), E(0, 4, 4), E(4, 5, 4), E(5, 3, 4), E(2, 6, 4), E(6, 7, 4), E(7, 1, 4)],
  },
  {
    id: "twocuts", label: "two minimum cuts",
    nodes: [
      { name: "s", x: 60, y: 230 }, { name: "t", x: 740, y: 230 },
      { name: "a", x: 260, y: 100 }, { name: "b", x: 260, y: 360 },
      { name: "c", x: 540, y: 100 }, { name: "d", x: 540, y: 360 },
    ],
    // s=0 t=1 a=2 b=3 c=4 d=5
    edges: [E(0, 2, 5), E(0, 3, 2), E(2, 4, 6), E(2, 3, 5), E(3, 5, 6), E(4, 1, 4), E(5, 1, 3), E(4, 5, 6)],
  },
];
