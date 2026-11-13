# Open issues after the first full build (2026-11-12)

Collected by the integration step of the parallel day build. Input for the second pass
(cross-links, consolidation into shared libs, optional units, data/licences).

Fixed since: `lib/plot.ts` niceTicks infinite loop and clip-path id collisions;
`toRGBA` LUT stride (RGB or RGBA); `segchain` readout CSS on slides.

**Builders' open issues, grouped**

*Optional units not built:*
- `localisation/` (image formation)
- `deconv-practice/` (deconvolution)
- `threshold-more/` (thresholding)
- `ws-graphs/` and the floodgrid `msf` mode (watershed)
- `fourier-mellin/`, plus the 3D phasecorr tab and the 2D subpixel tab (FFT registration)
- `forest-more/` (random forests)
- `deep-practice/` and the `shortcut` weight set (neural networks)
- `learned-restoration/`, the masked-autoencoder weights and the real instance network (modern deep learning)
- `moment-features/` and `moments-3d/` (moments)
- `aam/` (snakes)
- `surfacenets/` (marching cubes)
- `voxelise/` (3D graphics)
- `seamcarve/`, `dtw/` and the Amini toggle (dynamic programming)
- `potts/` and `boltzmann/` (Ising)
- `integral-more/` (integral images)
- `detect-window/`, the HOG detector (SVM)
- `bgsub/` (Gaussian mixtures)
- `cwt/` and `lifting/` (wavelets)
- `ct-artefacts/` and `radon-analysis/` (Radon)
- `phasor/` (unmixing)
- Also: the shading texture tab is not embedded in `gpuformats/`. Its gallery entry links to `gpuformats/exercise.qmd#explore`, which shows a different demo.

*Bugs in shared libraries:*
- **`lib/plot.ts`, clip-path ids:** ids are numbered per bundle (`qb-clip-N`, `qb-cb-N`), so two demos on one page clip each other's plots. About 6 days work around it with local `uniqueClip()` copies.
- **`lib/plot.ts`, `niceTicks`:** it loops forever on a near-zero range, which crashes the renderer. Demos such as `gaussderiv`, `scale1d`, `snake` and `blobscale` may be exposed.
- **`lib/plot.ts`, `plotFigure(parent, "")`:** it creates no figcaption.
- **`lib/imageops.ts` and `lib/colour.ts`:** `toRGBA` expects an RGB lookup table while `colormapLUT` returns RGBA.
- **CSS cascade:** `demo.css` loads late and beats per-demo selectors. Most builders fixed their own demos; `segchain` still has a broken readout on the `dp-paths` and `seg-energy` slides.

*Local helpers that belong in shared libraries:*
- **UI:**
  - `canny/imgview.ts` (used by about 7 demos)
  - `deconvlab/widgets.ts` (used by `psfotf` and `resolve`)
  - `medialaxis/choice.ts`
  - `curveflow/anim.ts`
  - the `onScreen()` copies in the snakes demos
  - `houghlines/view.ts`
  - `radonproj/view.ts`
  - `svm2d/field.ts`
  - `histthresh` scene and drag helpers
  - `isocontour/histc.ts`
  - `lsunmix/common.ts`
- **Synthetic scenes and test images that belong in `lib/synth.ts`:** scenes in `deconvlab`, `rlsteps`, `psfotf`, `histthresh`, `hysteresis`, `colourcloud`, `tophat`, `granulo`, `reconstruct`, `regionfilter`, `instancemaps`, `genrestore`, `embedprobe`, `colourgmm`, `dwt2d`, `jpeg/testimages`, `measuredamage`, `polarseg` and `livewire`. Also the `radon` and `unmix` test objects, and the digit/letter glyphs that belong in `lib/shapes.ts`.
- **Algorithms:**
  - `RealConvolver` should move to `lib/fft.ts`.
  - `besselJ0` and the Hankel PSF should move to `lib/psf.ts`.
  - `fastdisc.ts` and the BFS reconstruction should move to `lib/morph.ts`.
  - The region properties and features in the `moments` demo should move to `lib/label.ts` and `lib/features.ts`.
  - The `unionfind` forest should move to `lib/label.ts`.
  - ROC/PR should move to `lib/metrics.ts`.
  - `colourx.ts` (Y′CbCr, CIEDE2000, adaptation, dichromats, extra colour maps) and the Y′CbCr code in `lib/jpeg.ts` should move to `lib/colour.ts`.
  - The local Otsu and k-means in `colourseg`, and `multiOtsu` in `histgmm`, should use `lib/threshold.ts` and `lib/kmeans.ts`.
  - The block DCT in `dwt2d` should come from `lib/dct.ts`.
  - The stump search in `lib/boost.ts` should share code with `lib/forest.ts`.
  - `lib/imageops.ts` should re-export `boxMean` and `localStats`.
  - `lib/volrender.ts` duplicates the volume path in `lib/gl3d.ts`.
  - The Procrustes code in `lib/shapes.ts` (the plan wanted `lib/linalg.ts`).
  - `lib/playground.ts` needs a per-point colour option and an option to hide its grid.
- **`demos/src/lib/README.md`** needs entries for psf, deconv, denoise, edges, threshold, spectra, convexhull, watershed, features, glcm, neighbours, registration, forest, classifier, featurestack, autodiff, cnn, q8, eval, stats, boxes, mixture, skeleton, trees, moments, snake, shapes, volume, marching, mesh, raster, volrender, levelset, dp, shortestpath, mrf, smooth, hough, ransac, integral, boost, svm, svmkernels, kmeans, gmm, wavelet, entropy, huffman, dct, jpeg, predict, radon and unmix.

*Data and licences:* everything is synthetic or uses the credited `cell256.png`. Still needed:
- real oriented texture and real spot images (edges)
- real widefield bead data (image formation)
- a real image with uneven illumination and an annotation (thresholding)
- licensed H&E, colour photo and fluorescence images (colour)
- a fluorescence image with uneven background (morphology)
- a real image of many separate objects (labelling)
- an image of touching nuclei (watershed)
- a two-channel image and a cell population (measurements)
- a real tiled acquisition (FFT registration)
- a nuclei set annotated twice (evaluation)
- real neuron and vessel images (skeletons)
- a real two-phase image (Ising)
- road and building photos (Hough)
- a real 3D stack (marching cubes)
- a colour photo for JPEG (compression)
- a CT slice or sinogram (Radon)
- a real lambda stack and H&E or H-DAB slides (unmixing)

The MNIST copy used by `cnnviewer` (CC BY-SA 3.0) is credited.

*Plan corrections and uncertain numbers:*
- **Hessian:** σ²λ2 peaks at √2·s, not at s; the skeleton day should say the same.
- **Image formation:** the axial resolution formula is λ/n(1−cos α), and the bead-size correction is (0.53·D)².
- **Labelling:** the edge-count form factor of a disc is 0.62, not 0.79.
- **Moments:** η is not invariant to contrast, and monomial fits break down at N ≈ 22–26 in float64, not 12–15.
- **Evaluation:** there is no greedy counterexample for F1(τ), so the plan's math question B3 is impossible.
- **Watershed:** the plan's 7×7 two-squares mask does not split, so the exercise uses a different mask.
- **Level sets:** the 2D time-step (CFL) limit is Δx/(√2·|F|), not Δx/max|F|.
- **SVM:** the plan's by-hand example has a degenerate support vector, so the exercise uses new points.
- **Deconvolution:** blind Richardson–Lucy drifts towards a point PSF, and the L-curve corner uses a different rule from the plan.
- **Compression:** blocking does not show in the plain spectrum, so the demo uses the spectrum of neighbour differences.
- **Results weaker than the textbook effect** (the units report them honestly):
  - m = 1 in the forest
  - balancing in the pixel classifier
  - skip connections in the U-Net
  - the self-supervised encoder
  - wavelets on the micrograph
  - early termination saving 18 %
  - the 3 missed branch points in `arbor`
- **Links to check:** that `cameranoise/` covers variance stabilisation (pointed to from deconvolution), and that `fourier/` has the taper slide (pointed to from wavelets).
- **Other notes:**
  - Slide text is 13–17 px in many dense demos, below the 18 px rule.
  - Several decks run long (`richardson-lucy`, `replicates`, `attention`, `levelset-eq`, `intensity-features`).
  - Some demos take more than 30 ms per update with no Wasm: `unetseg`, the `tophat` opening at r = 20, the SVM pixel prediction, ring voting in `houghcircle`, GVF, live-wire, `instancemaps`, and `boostdetect`'s 1 s load.
  - The figure scripts for image formation, evaluation, wavelets and thresholding live only in the scratchpad, not in the repo.
  - Cross-day pointers in units owned by others still need editing; for example `seg-eval/` should point to `object-eval/`, and `kernels/` and `fourier/` should point to the edges day.

## Reminders for the user

- **sam2-rs on wasm32 (remind the user if still open):** `std::time::Instant::now()` panics on
  wasm32 ("time not implemented"); sam2-rs calls it unconditionally (optional profiling) in the
  mask decoder and prompt encoder. Workaround: a vendored, patched copy in
  `demos/wasm/sam/vendor/sam2-rs/` (all uses through a `clock.rs` stand-in). Upstream fix in
  henriksson-lab/sam2-rs would let us drop the copy. Check after the candle update (2026-11-13).
