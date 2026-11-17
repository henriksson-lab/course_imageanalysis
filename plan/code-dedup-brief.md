# Brief: remove duplicated helper code in the day crates

The course's Rust code is in `code/`: the library `code/imgcourse/` and one crate per day,
`code/dayNN-<id>/` (package `dayNN`), with one module per unit in `src/<unit>/`
(`mod.rs`, `exercise.rs`, `solution.rs`, `run.rs`, sometimes helper files such as `sim.rs`,
`given.rs`, `tomo.rs`, `plot.rs`), a clap `src/main.rs`, and checks in
`tests/check/<unit>.rs`. Read `plan/code-plan.md` (Layout) first.

The units were written separately, so helper code (not the exercises) is often repeated:
in several units of the same day, and as private copies of general tools. Your job, for the
days you are given:

1. **Within a day:** helper functions and types that occur in more than one unit of the day
   (identical, or the same code with trivial differences) move to **one** day-level module,
   `src/common.rs` (`pub mod common;` in `src/lib.rs`, with a short `//!` doc saying it holds
   helpers shared by the units of the day). The units use them with `use crate::common::...`.
   A unit's own helper file (`sim.rs`, `given.rs`, ...) that becomes empty is deleted
   (`git rm`). Keep helpers that only one unit uses where they are.
2. **General tools now in the library** (read `code/imgcourse/src/filters.rs` and `random.rs`):
   `imgcourse::filters::{gaussian_kernel, filter_1d, gaussian_blur, bilinear, rmse, psnr}` and
   `Rng::poisson(mean)`. Replace a local copy by the library version **only if it computes
   exactly the same** (same kernel cut, border rule, rounding, order of random numbers).
   Otherwise leave it (or move it to `common.rs` if repeated).
3. **Never** touch an exercise: a function whose body in `exercise.rs` is `todo!()` stays in
   that unit, in both `exercise.rs` and `solution.rs`, unchanged. Do not change test expected
   values, signatures of exercise functions, or any numbers on the pages. Helpers that a later
   unit carries filled in although an earlier unit asks students to write them stay as they are
   (moving them next to the exercise would give the answer away), unless both units are
   later units that only use the helper.
4. `exercise.rs` and `solution.rs` of a unit must keep identical helper code (students only
   see `exercise.rs`): whatever you change in one, change in the other.
5. If a page (`<unit>/exercise.qmd`) or a doc comment names a helper's file or location that
   you moved (e.g. "`sim.rs`", "given in `given.rs`"), update the text (now `src/common.rs`).

## Verifying (required): outputs must not change

Before you started, the program of every unit was run and its output recorded in
`/big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/51ed2e69-bf30-4fe9-bbf0-b7ac7e2bb101/scratchpad/baseline/`
(`<unit>.txt`: what it printed; `<unit>.sha`: hashes of the pictures in `out/<unit>/`).
After your changes to a day, run, for that day only:

```
bash /big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/51ed2e69-bf30-4fe9-bbf0-b7ac7e2bb101/scratchpad/integ/outputs.sh <your-dir> NN
bash /big/henriksson/temp/claude-tmp/claude-1001/-data-henriksson-github-course-imageanalysis/51ed2e69-bf30-4fe9-bbf0-b7ac7e2bb101/scratchpad/integ/days_check.sh NN
```

(`<your-dir>`: a folder of your own in the scratchpad, e.g. `.../scratchpad/after-<your name>`;
`NN` the two-digit day number.) Then compare each unit's `.txt` and `.sha` with the baseline
(`diff`). They must be identical, except for lines that print a run time. If anything else
differs, undo the change that caused it. `days_check.sh` must print nothing but `ok-done dayNN`
(tests pass on the solutions, clippy clean on solution and exercise builds).

## Rules for working alongside other workers

- Other workers do other days at the same time. Touch only your days' folders
  (`code/dayNN-*/` and the units' `exercise.qmd`). Never edit `code/imgcourse/`,
  `code/Cargo.toml`, `code.qmd` or other days.
- All builds share the one build folder `code/target` (do not set `CARGO_TARGET_DIR`); cargo
  waits for the lock while another build runs. Never run `cargo clean`, never `--workspace`.
- Use simple commands: no `cd dir && ...` chains, no inline scripts (`python3 -c`, heredocs);
  write any script with the Write tool into your own scratchpad folder and run it by path.
  Prefer the Edit tool for code changes. Do not commit.

## Report

Per day: what moved to `common.rs` (names, from which units), which local copies now use the
library, lines removed (roughly), anything you left duplicated on purpose and why, and the
verification result (outputs identical; checks clean).
