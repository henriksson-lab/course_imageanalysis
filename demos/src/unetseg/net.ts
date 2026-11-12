// The tiny U-Net of the unetseg demo (inference only, int8 weights from train_unet.py; no DOM,
// so the tests run it in node). Architecture in train_unet.py: 8-16-32 channels, two levels,
// nearest upsampling + 3 × 3 convolutions in the decoder, skip connections by concatenation.
import { conv2d, relu, maxPool, upsample, concat, convFromQ8, tensor, type ConvLayer, type Tensor } from "../lib/cnn";
import { UNET_SETS } from "./unet_weights";

export type WeightSet = "unet" | "noskip" | "dice";
const NAMES = ["e1a", "e1b", "e2a", "e2b", "b1", "b2", "d2a", "d2b", "d1a", "d1b", "out"] as const;
type Layers = Record<(typeof NAMES)[number], ConvLayer>;
const cache: Partial<Record<WeightSet, Layers>> = {};

function layers(set: WeightSet): Layers {
  if (cache[set]) return cache[set]!;
  const L = {} as Layers, w = UNET_SETS[set].layers;
  for (const n of NAMES) L[n] = convFromQ8(w[n], n === "out" ? 1 : 3);
  return (cache[set] = L);
}

/** All intermediate feature maps (after ReLU) and the probability map. */
export interface UnetRun { e1: Tensor; e2: Tensor; b: Tensor; d2: Tensor; d1: Tensor; prob: Float32Array }

/** Run the network on a 64 × 64 image of grey levels (0..255 scale). */
export function runUnet(img: Float32Array, set: WeightSet, n = 64): UnetRun {
  const L = layers(set), skip = UNET_SETS[set].skip;
  const x = tensor(1, n, n, Float32Array.from(img, v => v / 255));
  const c = (t: Tensor, l: ConvLayer) => relu(conv2d(t, l));
  const e1 = c(c(x, L.e1a), L.e1b);
  const e2 = c(c(maxPool(e1), L.e2a), L.e2b);
  const b = c(c(maxPool(e2), L.b1), L.b2);
  const u2 = upsample(b);
  const d2 = c(c(skip ? concat(u2, e2) : u2, L.d2a), L.d2b);
  const u1 = upsample(d2);
  const d1 = c(c(skip ? concat(u1, e1) : u1, L.d1a), L.d1b);
  const logit = conv2d(d1, L.out);
  const prob = Float32Array.from(logit.data, v => 1 / (1 + Math.exp(-v)));
  return { e1, e2, b, d2, d1, prob };
}

/** Number of weights and biases of a weight set. */
export function countWeights(set: WeightSet): number {
  const L = layers(set);
  return NAMES.reduce((s, n) => s + L[n].W.length + L[n].b.length, 0);
}
