.PHONY: all render preview clean gitadd help demos typecheck

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
	@echo "  make gitadd    stage all source files in git"
	@echo "  make clean     remove docs/, demos/dist/ and .quarto/"
