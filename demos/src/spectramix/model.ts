// The spectramix demo's presets, without DOM (numbers in the exercise come from here).
import * as U from "../lib/unmix";

export interface State { dyes: U.Dye[]; bands: U.Band[]; detector: boolean; bins: number; sel: number }

export const DYE_NAMES = ["dye 1", "dye 2", "dye 3", "dye 4"];
export const dye = (i: number, peak: number, width = 14, tail = 2): U.Dye => ({ name: DYE_NAMES[i], peak, width, tail });

export const PRESETS: Record<string, { label: string; s: () => State }> = {
  classic: { label: "green into orange: the classic tail", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 578, 11, 2.2)], bands: [{ lo: 510, hi: 550 }, { lo: 565, hi: 605 }], detector: false, bins: 8, sel: 0 }) },
  separated: { label: "well separated", s: () => ({ dyes: [dye(0, 460, 12, 2), dye(1, 650, 12, 1.6)], bands: [{ lo: 445, hi: 490 }, { lo: 640, hi: 690 }], detector: false, bins: 8, sel: 0 }) },
  narrow: { label: "narrow filters: less bleed-through, fewer photons", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 578, 11, 2.2)], bands: [{ lo: 515, hi: 527 }, { lo: 580, hi: 592 }], detector: false, bins: 8, sel: 0 }) },
  close: { label: "two dyes 20 nm apart", s: () => ({ dyes: [dye(0, 520, 13, 2.6), dye(1, 540, 13, 2.6)], bands: [{ lo: 505, hi: 525 }, { lo: 540, hi: 570 }], detector: false, bins: 8, sel: 0 }) },
  detector: { label: "8-bin spectral detector", s: () => ({ dyes: [dye(0, 510, 12, 2.2), dye(1, 560, 12, 2.2), dye(2, 610, 12, 1.8), dye(3, 665, 12, 1.5)], bands: [{ lo: 500, hi: 540 }, { lo: 550, hi: 590 }, { lo: 600, hi: 640 }, { lo: 655, hi: 700 }], detector: true, bins: 8, sel: 0 }) },
};
export const BAND_DEFAULT = [{ lo: 500, hi: 540 }, { lo: 560, hi: 600 }, { lo: 620, hi: 660 }, { lo: 670, hi: 710 }, { lo: 440, hi: 470 }, { lo: 715, hi: 745 }, { lo: 405, hi: 435 }, { lo: 600, hi: 615 }];
export const DYE_DEFAULT = [520, 578, 630, 680];
