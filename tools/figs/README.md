# Figure scripts

Scripts that generated static figures (SVG/PNG) in some unit folders, kept for
provenance. They were written by the day builders and run from a scratch directory, so
paths inside may need adjusting before re-running (output files go into the matching
unit folders: image formation → `psf/`, `resolution/`, `illposed/`; evaluation →
`object-eval/`, `agreement/`, `leakage/`, `replicates/`; wavelets → `haar/`,
`filterbank/`, `dwt2d/`, `shrinkage/`; thresholding → `histogram/`, `otsu/`,
`local-threshold/`, `seg-eval/`).

3D graphics day (`pipeline3d/`, `rasterise/`, `gpuformats/`, `volrender/`): one folder per unit.
`figs.py` writes the figures; `rasterise/render.ts` and `volrender/render.ts` first render
with the course's own software rasteriser and CPU ray caster (bundle with
`node_modules/.bin/esbuild <file> --bundle --platform=node`, run with an output directory,
then pass that directory to `figs.py`).
