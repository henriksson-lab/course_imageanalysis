# Brief: building the shape-matching day (day 24)

You build **one unit** of the day "Shape matching in Fourier space" (`shapematch`, day 24,
right after FFT registration). `plan/shapematch-day.md` gives the content of every unit; your
unit's section there is your specification. The course is a Quarto site; read before writing
anything:

1. `plan/shapematch-day.md` (whole file: goals, scope 2D only with 3D as concept slides,
   overlaps table, your unit, shared engineering, "Changes elsewhere").
2. `designprinciples.md` and `audience.md` (the two layers; ~half the slides with a picture
   beside text; decks of ~10–27 slides; demos first; no tool or library names).
3. `README.md` (layout, "Marking the math layer") and `demos/README.md` (demo conventions,
   `autoMount`, the shared helpers in `demos/src/lib/`, "Fitting the screen").
4. `plan/code-plan.md` (Layout: the Rust day crate `code/<day id>/`, one module per unit).
5. Model units to copy the shape from: the FFT registration day, `xcorr-fft/` and `phasecorr/`
   (pages, demos `demos/src/xcorr1d/`, `demos/src/phasecorr/`, tests `demos/test/*-test.ts`,
   figures `tools/figs/<unit>/`), and their code in `code/fftreg/`. Also read the existing units
   your unit recalls (named in the plan: `shape-features/`, `regionprops/`, `zernike/`,
   `moment-invariants/`, `polar-boundary/`, `stitching/` ...) so that you link to them and do not
   repeat them.

## Per unit (folder names are fixed in `days.yml`, where your unit is `status: planned`)

- `<unit>/lec.qmd` (copy the YAML header of `phasecorr/lec.qmd`), `<unit>/quiz.qmd`,
  `<unit>/exercise.qmd`: Goal, (Background), `## By hand` (concept questions with numbers,
  then `### Math questions` tagged *(math)*, solutions in collapsed callouts), `## Explore
  {#explore}` with the demo and questions, then `## Code {#code}` (as in `phasecorr/exercise.qmd`:
  tasks with pseudocode, questions about the program's output, collapsed "Solutions (code)";
  never call the code optional), then `## Where this leads`.
- One interactive demo per unit (TypeScript, `demos/src/<name>/main.ts`, name given in the plan;
  reuse `demos/src/lib/`), with a correctness test of its algorithm `demos/test/<name>-test.ts`.
  Figures for the slides from scripts in `tools/figs/<unit>/`.
- The Rust module `code/shapematch/src/<unit_ident>/` (`mod.rs`, `exercise.rs`, `solution.rs`,
  `run.rs`), `tests/check/<unit_ident>.rs`. **Do not edit** the crate's `lib.rs`, `main.rs` or
  `tests/check/main.rs` (three builders share them); give the lines to add in your report (the
  `pub mod`, the clap subcommand named after the unit folder, the `mod` in tests). To compile
  and test while you work, add your lines locally to those three files *and leave them in*: the
  integrator merges them, and other builders do the same, so expect their lines there too and
  never remove lines that are not yours. Only `std` + `imgcourse` (+ clap in main.rs); use the
  library (`fft`, `Complex`, `linalg`, `Rng`, `samples`) where it fits. Helpers shared by
  several units go in `src/common.rs` only by agreement with the integrator: keep yours in your
  module. Every number quoted on a page must come from actually running your program or your
  demo code.
- One pseudocode slide per deck where the deck has an algorithm (`## <...> as pseudocode`).
- The day number is 24; later days have already been renumbered in all pages.

## Shared files: do not edit; report instead

**Do not edit** `days.yml`, `_quarto.yml`, `Makefile`, `demos.qmd`, `code/Cargo.toml`,
`code.qmd`, `styles.css`, `overview.qmd`, other units' folders (except the cross-links below),
or `docs/`. In your report, list: the demo test files (for `TESTS` in the Makefile), one
`demos.qmd` gallery entry per demo (copy the format of the existing entries), and the sidebar
demo labels per unit (the `demos:` map as in `days.yml`). The integrator adds them.

Exception, for cross-links: you may add a sentence or link to the existing units named in the
plan's "Changes elsewhere when this is built" that belong to your unit (listed in your task),
and nothing else in them.

## Checks you run (only for your own files)

```
demos/build.sh <demo>            # each of your demos
node_modules/.bin/tsc --noEmit -p demos
node_modules/.bin/esbuild demos/test/<name>-test.ts --bundle --platform=node --log-level=warning --outfile=<your scratch dir>/t.js
node <your scratch dir>/t.js
cargo test   --manifest-path code/Cargo.toml -p shapematch --features solution
cargo clippy --manifest-path code/Cargo.toml -p shapematch --all-targets --features solution -- -D warnings -A clippy::needless_range_loop
cargo clippy --manifest-path code/Cargo.toml -p shapematch --all-targets -- -D warnings -A clippy::needless_range_loop
cargo run    --manifest-path code/Cargo.toml -p shapematch --features solution -- <unit>
```

Another builder's half-finished module may break the crate build for a while: wait and retry,
or tell the integrator in your report; never edit their files. All cargo commands share the one
build folder `code/target` (do not set `CARGO_TARGET_DIR`). Render only your own pages, one at a
time: `TMPDIR=<your scratch dir> quarto render <unit>/lec.qmd` (also quiz, exercise). Your unit
is `status: planned`, so it is not in the sidebar yet; that is fine. Look at the result, then
delete the `.html` files and `*_files/` folders the render leaves in the source folders. The
system disk is nearly full: always set TMPDIR to your scratch folder.

## Working rules

- Your scratch folder: `/big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/c77e6e6b-13c1-44bb-8f06-70e7595eab33/scratchpad/<unit>/`.
- Simple shell commands: no `cd dir && ...` chains, no inline scripts (`python3 -c`,
  heredocs); write scripts with the Write tool into your scratch folder and run them by path.
  Prefer the Edit and Write tools. Do not commit, do not run `cargo clean`, never `--workspace`.
- Figures: Python scripts (matplotlib) or SVG written by a script, as in `tools/figs/`.

## Report

What the deck, quiz, exercise, demo and code cover; slide count and how many have a picture;
check results; the shared-file entries listed above (including the crate lines); the
cross-links you added; anything unverified or left open.
