// Huffman codes (no DOM). Owner: the compression day (entropy/ unit, jpeg/ unit).
//
// `huffman(weights, tie)` repeatedly merges the two lightest roots and records every merge,
// so a demo can step through the construction. Ties between equal weights are broken by a
// selectable rule (different trees, same average length):
//   "older":  among equal weights take the node created first (leaves before merged nodes);
//   "newer":  take the node created last (merged nodes first: deeper, more skewed trees).
// The first node taken in a merge becomes the 0 branch, the second the 1 branch.
// Symbols with weight 0 get no code. A single used symbol gets the one-bit code "0".
//
// Also: Kraft sum, average length, canonical codes from code lengths, encode/decode, and the
// standard JPEG Huffman tables of Annex K (luminance and chrominance, DC and AC).

export type TieRule = "older" | "newer";

export interface HuffNode {
  weight: number;
  symbol: number;          // -1 for an internal node
  left: number;            // child taken first (bit 0); -1 for a leaf
  right: number;           // child taken second (bit 1)
  parent: number;
}

export interface Merge { a: number; b: number; node: number; weight: number }

export interface HuffmanCode {
  nodes: HuffNode[];       // leaves first (node i = symbol i for i < K), then merged nodes in order
  root: number;            // -1 if no symbol has weight > 0
  merges: Merge[];
  lengths: Int32Array;     // code length per symbol (0 = unused)
  codes: string[];         // code word per symbol ("" = unused)
}

/** Build a Huffman code for symbols 0..K-1 with the given weights (counts or probabilities). */
export function huffman(weights: ArrayLike<number>, tie: TieRule = "older"): HuffmanCode {
  const K = weights.length;
  const nodes: HuffNode[] = [];
  for (let i = 0; i < K; i++) nodes.push({ weight: weights[i], symbol: i, left: -1, right: -1, parent: -1 });
  let roots: number[] = [];
  for (let i = 0; i < K; i++) if (weights[i] > 0) roots.push(i);
  const merges: Merge[] = [];
  const better = (a: number, b: number) => {   // true if node a is taken before node b
    const wa = nodes[a].weight, wb = nodes[b].weight;
    if (wa !== wb) return wa < wb;
    return tie === "older" ? a < b : a > b;
  };
  while (roots.length > 1) {
    let i1 = 0;
    for (let i = 1; i < roots.length; i++) if (better(roots[i], roots[i1])) i1 = i;
    const a = roots[i1];
    roots.splice(i1, 1);
    let i2 = 0;
    for (let i = 1; i < roots.length; i++) if (better(roots[i], roots[i2])) i2 = i;
    const b = roots[i2];
    roots.splice(i2, 1);
    const node = nodes.length, weight = nodes[a].weight + nodes[b].weight;
    nodes.push({ weight, symbol: -1, left: a, right: b, parent: -1 });
    nodes[a].parent = node; nodes[b].parent = node;
    merges.push({ a, b, node, weight });
    roots.push(node);
  }
  const root = roots.length ? roots[0] : -1;
  const lengths = new Int32Array(K), codes: string[] = new Array(K).fill("");
  if (root >= 0 && nodes[root].symbol >= 0) { lengths[root] = 1; codes[root] = "0"; }
  else if (root >= 0) {
    const walk = (n: number, prefix: string) => {
      const nd = nodes[n];
      if (nd.symbol >= 0) { codes[nd.symbol] = prefix; lengths[nd.symbol] = prefix.length; return; }
      walk(nd.left, prefix + "0"); walk(nd.right, prefix + "1");
    };
    walk(root, "");
  }
  return { nodes, root, merges, lengths, codes };
}

/** Kraft sum Σ 2^{-L_k} over the used symbols (≤ 1 for any prefix code; = 1 for Huffman with ≥ 2 symbols). */
export function kraftSum(lengths: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < lengths.length; i++) if (lengths[i] > 0) s += Math.pow(2, -lengths[i]);
  return s;
}

/** Average code length Σ p_k L_k for weights (normalised here). */
export function averageLength(weights: ArrayLike<number>, lengths: ArrayLike<number>): number {
  let s = 0, t = 0;
  for (let i = 0; i < weights.length; i++) { s += weights[i] * lengths[i]; t += weights[i]; }
  return t > 0 ? s / t : 0;
}

/** True if no code word is a prefix of another (empty words ignored). */
export function isPrefixFree(codes: string[]): boolean {
  const c = codes.filter(s => s.length > 0);
  for (let i = 0; i < c.length; i++) for (let j = 0; j < c.length; j++)
    if (i !== j && c[j].startsWith(c[i])) return false;
  return true;
}

/**
 * Canonical code words from code lengths (as JPEG and DEFLATE store them): symbols sorted by
 * (length, symbol); each code is the previous one + 1, shifted left when the length grows.
 */
export function canonicalCodes(lengths: ArrayLike<number>): string[] {
  const idx = Array.from({ length: lengths.length }, (_, i) => i).filter(i => lengths[i] > 0)
    .sort((a, b) => lengths[a] - lengths[b] || a - b);
  const out: string[] = new Array(lengths.length).fill("");
  let code = 0, len = idx.length ? lengths[idx[0]] : 0;
  for (const s of idx) {
    code <<= lengths[s] - len;
    len = lengths[s];
    out[s] = code.toString(2).padStart(len, "0");
    code++;
  }
  return out;
}

/** Concatenate the code words of a symbol sequence. */
export function encode(symbols: ArrayLike<number>, codes: string[]): string {
  let s = "";
  for (let i = 0; i < symbols.length; i++) s += codes[symbols[i]];
  return s;
}

/** Decode a bit string by walking the tree from the root (stops at an incomplete last word). */
export function decode(bits: string, code: HuffmanCode): number[] {
  const out: number[] = [];
  if (code.root < 0) return out;
  const single = code.nodes[code.root].symbol >= 0;
  let n = code.root;
  for (const b of bits) {
    if (single) { out.push(code.nodes[code.root].symbol); continue; }
    n = b === "0" ? code.nodes[n].left : code.nodes[n].right;
    if (code.nodes[n].symbol >= 0) { out.push(code.nodes[n].symbol); n = code.root; }
  }
  return out;
}

/** Decode with a plain code table (any prefix code). */
export function decodeTable(bits: string, codes: string[]): number[] {
  const map = new Map<string, number>();
  codes.forEach((c, i) => { if (c) map.set(c, i); });
  const out: number[] = [];
  let cur = "";
  for (const b of bits) { cur += b; const s = map.get(cur); if (s !== undefined) { out.push(s); cur = ""; } }
  return out;
}

// ---------------------------------------------------------------------------------------
// JPEG standard Huffman tables (ITU-T T.81, Annex K.3), as BITS (number of codes of each
// length 1..16) and HUFFVAL (symbols in order of increasing code). DC symbols are the size
// categories 0..11; AC symbols are (run << 4) | size, with 0x00 = EOB and 0xF0 = ZRL.

export interface JpegTable { bits: number[]; vals: number[] }

export const DC_LUMA: JpegTable = { bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], vals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] };
export const DC_CHROMA: JpegTable = { bits: [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], vals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] };

export const AC_LUMA: JpegTable = {
  bits: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
  vals: [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
  ],
};

export const AC_CHROMA: JpegTable = {
  bits: [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
  vals: [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
  ],
};

/** Code word (as a bit string) of every symbol 0..255 of a JPEG table ("" = no code). */
export function jpegCodes(t: JpegTable): string[] {
  const out: string[] = new Array(256).fill("");
  let code = 0, k = 0;
  for (let len = 1; len <= 16; len++) {
    for (let i = 0; i < t.bits[len - 1]; i++) out[t.vals[k++]] = code++.toString(2).padStart(len, "0");
    code <<= 1;
  }
  return out;
}
