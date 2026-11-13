// Exports the synthetic images of the morphology demos (and results computed with the
// course's own code) as JSON, for the static figures of the morphology day.
// Bundle and run from the repo root:
//   node_modules/.bin/esbuild tools/figs/morph-reconstruct/data.ts --bundle --platform=node --outfile=$OUT/data.js && node $OUT/data.js $OUT
import { writeFileSync } from "fs";
import { blobs, N, NUCLEI_OPTS } from "../../../demos/src/reconstruct/images";
import { geodesicSteps, frameOf } from "../../../demos/src/reconstruct/binrec";
import { edt } from "../../../demos/src/lib/distance";
import { twoSizes, SIZE } from "../../../demos/src/granulo/images";
import { openingFunction, granulometry } from "../../../demos/src/granulo/size";
import { spotsOnSlope } from "../../../demos/src/tophat/images";
import { openDisc } from "../../../demos/src/tophat/fastdisc";
import { nuclei } from "../../../demos/src/lib/synth";
import { hMaxima } from "../../../demos/src/lib/morph";

const out = process.argv[2] ?? ".";
const save = (name: string, obj: unknown) => writeFileSync(`${out}/${name}.json`, JSON.stringify(obj));
const arr = (a: ArrayLike<number>) => Array.from(a);

// reconstruction demo: blobs, opening and opening by reconstruction (r = 4), clear border, fill holes
{
  const mask = blobs(), n = N * N, r = 4;
  const d = edt(mask, N, N, { squared: true });
  const ero = Uint8Array.from(d, v => (v > r * r ? 1 : 0));
  const dc = edt(Uint8Array.from(ero, v => (v ? 0 : 1)), N, N, { squared: true });
  const open = Uint8Array.from(dc, v => (v <= r * r ? 1 : 0));
  const openrec = Uint8Array.from(geodesicSteps(ero, mask, N, N, 8), v => (v >= 0 ? 1 : 0));
  const fr = frameOf(N, N);
  const touch = geodesicSteps(Uint8Array.from(mask, (v, i) => v & fr[i]), mask, N, N, 8);
  const clear = Uint8Array.from(mask, (v, i) => (v && touch[i] < 0 ? 1 : 0));
  const bg = Uint8Array.from(mask, v => (v ? 0 : 1));
  const bsteps = geodesicSteps(Uint8Array.from(bg, (v, i) => v & fr[i]), bg, N, N, 4);
  const fill = Uint8Array.from(mask, (v, i) => (v || bsteps[i] < 0 ? 1 : 0));
  let rmOpen = 0, rmRec = 0, rmClear = 0, added = 0;
  for (let i = 0; i < n; i++) { if (mask[i] && !open[i]) rmOpen++; if (mask[i] && !openrec[i]) rmRec++; if (mask[i] && !clear[i]) rmClear++; if (!mask[i] && fill[i]) added++; }
  save("blobs", { N, mask: arr(mask), ero: arr(ero), open: arr(open), openrec: arr(openrec), clear: arr(clear), fill: arr(fill), rmOpen, rmRec, rmClear, added });
  // nuclei for h-maxima
  const nuc = nuclei(NUCLEI_OPTS).image;
  const hm = hMaxima(nuc.data, N, N, 15, 8);
  save("nuclei", { N, img: arr(nuc.data), hmax15: arr(hm) });
}
// granulometry demo: two sizes, disc SE
{
  const mask = twoSizes(), of = openingFunction(mask, SIZE, SIZE, "disc", 24), g = granulometry(of, 24);
  save("twosizes", { N: SIZE, mask: arr(mask), of: arr(of), area: g.area, spectrum: g.spectrum });
}
// top-hat demo: spots on a slope
{
  const s = spotsOnSlope(), w = s.image.width;
  save("spots", { N: w, img: arr(s.image.data), open6: arr(openDisc(s.image.data, w, w, 6)), open2: arr(openDisc(s.image.data, w, w, 2)), open20: arr(openDisc(s.image.data, w, w, 20)), bright: s.bright });
}
