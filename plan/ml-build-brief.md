# Brief: building the new machine-learning days

You build one new day (or the drifting unit) of `plan/ml-additions-plan.md`, which gives the
content of every unit. The course is a Quarto site; read before writing anything:

1. `plan/ml-additions-plan.md` (your day's section, "Changes to existing units", and E if yours).
2. `designprinciples.md` and `audience.md` (the two layers; ~half the slides with a picture
   beside text; decks of ~10–27 slides; demos first; no tool or library names).
3. `README.md` (layout, "Marking the math layer") and `demos/README.md` (demo conventions,
   `autoMount`, the shared helpers in `demos/src/lib/`, "Fitting the screen").
4. `plan/code-plan.md` (Layout: the Rust day crate `code/<day id>/`, one module per unit).
5. Model units to copy the shape from: `backprop/` and `neuralnet/` (pages, demos
   `demos/src/<name>/`, tests `demos/test/*-test.ts`, figures `tools/figs/<unit>/`), and their
   code in `code/neuralnets/`. Read the existing units your day builds on (named in the plan),
   so that you link to them and do not repeat them (e.g. `backprop` already has momentum, a
   gradient-descent landscape demo and an overfitting demo: the training day goes further).

## Per unit (folder names are fixed in `days.yml`, where your units are `status: planned`)

- `<unit>/lec.qmd` (copy the YAML header of `backprop/lec.qmd`), `<unit>/quiz.qmd`,
  `<unit>/exercise.qmd`: Goal, (Background), `## By hand` (concept questions with numbers,
  then `### Math questions` tagged *(math)*, solutions in collapsed callouts), `## Explore
  {#explore}` with the demo and questions, then `## Code {#code}` (as in `backprop/exercise.qmd`:
  tasks with pseudocode, questions about the program's output, collapsed "Solutions (code)";
  never call the code optional), then `## Where this leads`.
- One interactive demo per unit (TypeScript, `demos/src/<name>/main.ts`; small networks train
  live in the browser; reuse `demos/src/lib/`), with a correctness test of its algorithm
  `demos/test/<name>-test.ts`. Figures for the slides from scripts in `tools/figs/<unit>/`.
- The Rust module `code/<day id>/src/<unit_ident>/` (`mod.rs`, `exercise.rs`, `solution.rs`,
  `run.rs`), `tests/check/<unit_ident>.rs`, registered in the crate's `lib.rs`, `main.rs`
  (clap subcommand named after the unit folder) and `tests/check/main.rs`. Only `std` +
  `imgcourse` (+ clap in main.rs); use the library (`filters`, `fft`, `linalg`, `Rng`) where it
  fits. Helpers shared by several units of your day go in `src/common.rs`. Every number quoted
  on a page must come from actually running your program.
- One pseudocode slide per deck where the deck has an algorithm (`## <...> as pseudocode`).

## Shared files: do not edit; report instead

Other workers build other days at the same time. **Do not edit** `days.yml`, `_quarto.yml`,
`Makefile`, `demos.qmd`, `code/Cargo.toml`, `code.qmd`, `styles.css`, other days' folders, or
`docs/`. In your report, list: the demo test files (for `TESTS` in the Makefile), one
`demos.qmd` gallery entry per demo (copy the format of the existing entries), and the sidebar
demo labels per unit (the `demos:` map as in `days.yml`). The integrator adds them.

Exception, for cross-links: you may add a sentence or link to the existing units the plan's
"Changes to existing units" assigns to your day, and nothing else in them.

## Your crate

The integrator has created `code/<day id>/` (registered in the workspace) with an empty
`lib.rs`, a placeholder `main.rs` and `tests/check/main.rs`. Fill them in. The drifting worker
adds a module to the existing crate `code/diffusion/`.

## Checks you run (only for your own files)

```
demos/build.sh <demo>            # each of your demos
node_modules/.bin/tsc --noEmit -p demos
node_modules/.bin/esbuild demos/test/<name>-test.ts --bundle --platform=node --log-level=warning --outfile=<your scratch dir>/t.js
node <your scratch dir>/t.js
cargo test   --manifest-path code/Cargo.toml -p <day id> --features solution
cargo clippy --manifest-path code/Cargo.toml -p <day id> --all-targets --features solution -- -D warnings -A clippy::needless_range_loop
cargo clippy --manifest-path code/Cargo.toml -p <day id> --all-targets -- -D warnings -A clippy::needless_range_loop
cargo run    --manifest-path code/Cargo.toml -p <day id> --features solution -- <unit>
```

All cargo commands share the one build folder `code/target` (do not set `CARGO_TARGET_DIR`;
cargo waits while another build runs). Render only your own pages, one at a time:
`TMPDIR=<your scratch dir> quarto render <unit>/lec.qmd` (also quiz, exercise), then look at
the result, and delete the `.html` files and `*_files/` folders the render leaves in the
source folders. The system disk is nearly full: always set TMPDIR to your scratch folder.

## Working rules

- Your scratch folder: `/big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/51ed2e69-bf30-4fe9-bbf0-b7ac7e2bb101/scratchpad/ml-<day id>/`.
- Simple shell commands: no `cd dir && ...` chains, no inline scripts (`python3 -c`,
  heredocs); write scripts with the Write tool into your scratch folder and run them by path.
  Prefer the Edit and Write tools. Do not commit, do not run `cargo clean`, never `--workspace`.
- Figures: Python scripts (matplotlib) or SVG written by a script, as in `tools/figs/`.

## Report

Per unit: what the deck, quiz, exercise, demo and code cover; slide count and how many have a
picture; check results; the shared-file entries listed above; the cross-links you added;
anything unverified or left open.
