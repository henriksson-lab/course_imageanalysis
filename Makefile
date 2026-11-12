.PHONY: all render preview clean gitadd help demos typecheck test

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
  demos/test/marchingcubes-test.ts demos/test/graphics3d-test.ts
test:
	@for t in $(TESTS); do echo "== $$t"; \
	  js=$$(node_modules/.bin/esbuild $$t --bundle --platform=node --log-level=warning) && printf '%s\n' "$$js" | node || exit 1; \
	done

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
	@echo "  make gitadd    stage all source files in git"
	@echo "  make clean     remove docs/, demos/dist/ and .quarto/"
