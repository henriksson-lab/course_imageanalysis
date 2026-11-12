# Image analysis: course site sources

Teacher information. This file is not part of the published site.

The site structure, demo framework and styling are adapted from the
*Quantitative biology and integrative omics* (3MB00A) course site.

- [audience.md](audience.md): who we are designing for.
- [designprinciples.md](designprinciples.md): how we build material.
- [demos/README.md](demos/README.md): how the interactive demos are built and embedded.

## Layout

One folder per unit, each with

- `lec.qmd`: a reveal.js slide deck (keep it small; one topic per deck),
- `quiz.qmd`: short questions with collapsible hints and answers,
- `exercise.qmd`: pen-and-paper questions, then an `## Explore {#explore}` section
  that embeds the unit's interactive demo, with questions about it.

The order of units is set by the sidebar in `_quarto.yml`, not by folder names.
Every demo also goes on `demos.qmd`, the gallery page.

## Building this site

```bash
npm i          # once: installs esbuild + typescript for the demos
make render    # build the demos and render the whole site to docs/
make preview   # start a live-reload preview server
make clean     # delete docs/, demos/dist/ and the Quarto cache
```

You will need [Quarto](https://quarto.org/docs/get-started/) and Node.js.

## Adding a demo

1. Create `demos/src/<name>/main.ts`; use the helpers in `demos/src/lib/ui.ts` and
   mount with `autoMount("<name>", host => ...)`.
2. `demos/build.sh <name>` and `make typecheck`.
3. Embed it in a unit (see `demos/README.md`) and add it to `demos.qmd` and the sidebar.
