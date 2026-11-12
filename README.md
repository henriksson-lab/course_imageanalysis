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

### Marking the math layer

Every unit has a concept layer and a math layer (design principle 0). Mark the math
layer the same way everywhere:

- **Slides:** the math goes on its own slide(s), right after the concept slide it
  formalises, titled `## The math: <topic> {.math}`. The `.math` class (styled in
  `theme.scss`) adds a tinted bar on the left and a "math layer" badge after the
  title. The "About this lecture" slide ends with the line *Slides titled **The
  math** give the formulas and proofs; the ideas are complete without them.* Concept
  slides must make sense without any math slide (the summary slide too).
- **Pages** (exercise background, longer explanations): a collapsed callout
  `::: {.callout-note collapse="true" title="The math: <topic>"}`.
- **Exercises:** concept questions (predict, explain, small calculations with
  numbers) come first, with their own `Solutions` callout. Math questions (derive,
  prove, generalise) follow under `### Math questions`, each tagged
  `**B7.** *(math)*`, with a separate `Solutions (math)` callout. Every exercise has
  concept questions; if a topic is mostly proofs, add concept versions with numbers.
  Learning goals that need the math are tagged *(math)* too.
- **Quizzes:** mostly concept questions; tag a purely mathematical one
  `**Q5.** *(math)*`.

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
