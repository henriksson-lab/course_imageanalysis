// The colour cloud demo (colour day, units colour-perceptual and colour-display): see cloud.ts.
// Auto-mounts on <div data-demo="colourcloud">; options in cloud.ts (data-scene, data-space,
// data-spaces, data-mode="pca", data-preset, data-noise).
import { autoMount } from "../lib/ui";
import { mountColourCloud } from "./cloud";

autoMount("colourcloud", host => mountColourCloud(host));
