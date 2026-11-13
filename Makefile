.PHONY: all render preview clean gitadd help demos typecheck test wasm models check-prereqs sidebar check check-links check-console

# Default: render the site, then stage every source file in git.
all: render gitadd

render: demos
	quarto render
	@touch docs/.nojekyll

preview: demos
	quarto preview

# Build the interactive demos (TypeScript, demos/src/) into demos/dist/.
# One-time setup: npm i   (installs esbuild + typescript into node_modules/).
demos:
	demos/build.sh

# Type-check the demo sources (esbuild only strips types, it does not check them).
typecheck:
	node_modules/.bin/tsc --noEmit -p demos

# Correctness tests of the demo algorithms (demos/test/): bundled for node by esbuild
# and run directly, no files written. E.g. max-flow against brute force over all cuts.
TESTS = demos/test/test.ts demos/test/bitdepth-test.ts demos/test/pca-test.ts \
  demos/test/foundation-2d-test.ts demos/test/foundation-ui-test.ts \
  demos/test/edges-test.ts demos/test/imageformation-test.ts \
  demos/test/deconvolution-test.ts demos/test/thresholding-test.ts \
  demos/test/colour-test.ts demos/test/morphology-test.ts demos/test/labelling-test.ts \
  demos/test/watershed-test.ts demos/test/measurements-test.ts \
  demos/test/fftregistration-test.ts demos/test/randomforest-test.ts \
  demos/test/neuralnets-test.ts demos/test/evaluation-test.ts \
  demos/test/moderndl-test.ts demos/test/skeleton-test.ts demos/test/moments-test.ts \
  demos/test/snakes-test.ts demos/test/levelsets-test.ts demos/test/dynprog-test.ts \
  demos/test/ising-test.ts demos/test/hough-test.ts demos/test/integral-test.ts \
  demos/test/svm-test.ts demos/test/gmm-test.ts demos/test/wavelets-test.ts \
  demos/test/compression-test.ts demos/test/radon-test.ts demos/test/unmixing-test.ts \
  demos/test/marchingcubes-test.ts demos/test/graphics3d-test.ts \
  demos/test/pointdetection-test.ts demos/test/opticalflow-test.ts \
  demos/test/tracking-test.ts demos/test/colocalisation-test.ts demos/test/fret-test.ts \
  demos/test/flim-test.ts demos/test/multiview-test.ts demos/test/raytracing-test.ts \
  demos/test/diffusion-test.ts demos/test/splines-test.ts demos/test/gpus-test.ts \
  demos/test/contrast-test.ts demos/test/spatialstats-test.ts demos/test/graphseg-test.ts \
  demos/test/keypoints-test.ts demos/test/pdesmooth-test.ts demos/test/sparse-test.ts \
  demos/test/ncut-test.ts demos/test/coursegraph-test.ts \
  demos/test/stardist-test.ts demos/test/cellpose-test.ts demos/test/yolo-test.ts \
  demos/test/sam-test.ts demos/test/cellnets-compare-test.ts \
  demos/test/wasm-test.ts demos/test/maxflow-wasm-test.ts demos/test/stardist-wasm-test.ts \
  demos/test/cellpose-wasm-test.ts demos/test/yolo-wasm-test.ts demos/test/sam-wasm-test.ts
test:
	@for t in $(TESTS); do echo "== $$t"; \
	  js=$$(node_modules/.bin/esbuild $$t --bundle --platform=node --log-level=warning) && printf '%s\n' "$$js" | node || exit 1; \
	done

# The course structure database days.yml (parts, days, units, prerequisites):
# check-prereqs validates it (folders exist, prerequisites exist, no cycles, the order
# respects hard prerequisites, the sidebar is current); sidebar regenerates the sidebar
# in _quarto.yml and the part colours in styles.css from it.
check-prereqs:
	python3 tools/days.py check

sidebar:
	python3 tools/days.py sidebar

# Checks of the rendered site (docs/): broken local links, anchors and resources; then every
# page in headless Chrome for console errors (uncaught exceptions, console.error, failed loads).
# Run after make render. check-console takes JOBS=n and WAIT=seconds per page.
check: check-links check-console

check-links:
	python3 tools/check/links.py docs

check-console:
	tools/check/console.sh

# Rust -> WebAssembly crates demos/wasm/<name>/ -> demos/dist/wasm/<name>/ (wasm-bindgen glue
# + .wasm; git-ignored, CI builds it). Needs cargo, the wasm32-unknown-unknown target and
# wasm-bindgen-cli (version pinned in the crates). See demos/README.md, "WebAssembly".
wasm:
	demos/wasm/build.sh

# Model checkpoints and sample images (tools/models/manifest.json): downloads cached in
# .cache/models/, converted files in demos/dist/models/ (both git-ignored; CI deploys them).
# MODELS_ARGS=--all also fetches disabled entries (e.g. Cellpose-SAM, 1.2 GB).
models:
	python3 tools/models/fetch.py $(MODELS_ARGS)

# Remove the rendered site, the built demos and Quarto's cache.
clean:
	rm -rf docs .quarto demos/dist

# Stage every source file AND the built HTML (docs/) so the committed site is
# always current. The .quarto/ cache is never added (it is .gitignored).
gitadd:
	git add -- .

help:
	@echo "Targets:"
	@echo "  make           render the site, then git add all source files"
	@echo "  make render    build the demos and render the whole site to docs/"
	@echo "  make preview   start a live-reload preview server"
	@echo "  make demos     build the interactive demos into demos/dist/ (needs npm i once)"
	@echo "  make typecheck type-check the demo TypeScript sources"
	@echo "  make test      run the correctness tests of the demo algorithms"
	@echo "  make wasm      build the Rust/WebAssembly crates into demos/dist/wasm/"
	@echo "  make models    fetch and convert model weights and sample images into demos/dist/models/"
	@echo "  make check-prereqs  validate days.yml (prerequisites, order, sidebar current)"
	@echo "  make sidebar   regenerate the sidebar in _quarto.yml from days.yml"
	@echo "  make check     check the rendered site: links and anchors, then browser console errors"
	@echo "  make gitadd    stage all source files in git"
	@echo "  make clean     remove docs/, demos/dist/ and .quarto/"
