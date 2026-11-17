# Code in the course: plan and conventions

Decided 2026-11-16 (user): the course gets optional code, in **Rust**, kept simple and
aimed at learning rather than speed. Slides use **pseudocode** only (students arrive with
different languages, and Rust does not fit on slides anyway). Coding exercises come **after**
all other exercises (theory, pen and paper, demo), so nobody has to do them.

## Library: `code/imgcourse` (our own, not ndarray or image/imageproc)

Surveyed: `image` (file reading/writing; pixel types depend on generic traits; awkward for
floats), `imageproc` (implements exactly what students should write), `ndarray` (powerful
but generic-heavy, `[[row, col]]` indexing, API churn 0.15/0.16/0.17, no file reading),
`nalgebra` (column-major; fine for small matrices), `kornia-rs` (young, aimed at speed),
`photon-rs` (photo effects). None reads like the math on the slides, so we keep a small
library of our own, with `image` used inside it for file reading and writing only.

- `Image<T>` with `img[(x, y)]`, `x` = column, `y` = row (down), from 0. `f32` grey values in
  [0, 1] (8-bit files ÷ 255, 16-bit ÷ 65535). Positive angles turn clockwise on screen.
- Plumbing only: `new`, `from_fn`, `from_rows`, `coords`, `map`, `map_xy`, `zip_map`, `crop`,
  `get_border` + `Border::{Zero, Clamp, Mirror, Wrap}`, `Display` for small images;
  `load_gray`/`load_rgb`, `save_gray`/`_stretched`/`_signed`/`save_rgb`; `Complex`; `Rng`
  (seeded); `samples::{cell, letter_f, step, checker, add_noise}` (the micrograph is embedded).
- The image analysis itself (interpolation, filters, DFT, PCA, ...) is never in the library:
  it is what the exercises ask for. Add to the library only plumbing that several units need
  (candidates: `Volume<T>` for the 3D days, label images, a tiny matrix type).
- Later option: compile the same exercise code to Wasm for the demos, so that the code
  students read is the code the demo runs.

## Layout

`code/` is a Cargo workspace (it is also the student download, packed by
`tools/code_zip.py` into `downloads/imageanalysis-code.zip`; `make render` copies it to
`docs/downloads/`). Members: `imgcourse`, `starter`, and **one crate per day**,
`code/dayNN-<day id>/` with package name `dayNN` (merged from one crate per unit on
2026-11-17: 51 crates instead of 209, each with one binary and one test binary). Per unit
`U` (Rust module `u` = `U` with `-` → `_`):

```
code/dayNN-id/Cargo.toml             feature "solution"; deps imgcourse + clap; lib/bin test = false
code/dayNN-id/src/lib.rs             pub mod u; ... (one per unit)
code/dayNN-id/src/main.rs            clap: one subcommand per unit (named U), calls u::run::main()
code/dayNN-id/src/u/mod.rs           exports exercise.rs, or solution.rs with --features solution
code/dayNN-id/src/u/exercise.rs      same signatures and doc comments as the solution, bodies todo!("C1: name"), hints
code/dayNN-id/src/u/solution.rs      model solution
code/dayNN-id/src/u/run.rs           pub fn main(): runs the functions on real images, pictures in out/U/
code/dayNN-id/tests/check/main.rs    mod u; ... (one test binary per day)
code/dayNN-id/tests/check/u.rs       checks, preferably the numbers of the by-hand exercises
```

Student commands: `cargo test -p dayNN u::`, `cargo run -p dayNN -- U`, plus
`--features solution` for the model solution. Builds: one `target/`; dev profile with
`debug = "line-tables-only"` (deps without debug info), so a program is ~7 MB, not ~56 MB.

`make code-test` (also in CI): tests with `--all-features` (the solutions), builds the
templates, and runs clippy on both with `-D warnings`. A new unit: a module folder in its
day's crate, `pub mod u;` in `lib.rs`, a subcommand in `main.rs`, `mod u;` in
`tests/check/main.rs`; `make code-zip` regenerates the list on `code.qmd`.

## On the site

- `code.qmd` ("Coding exercises: getting started", in the sidebar before the topics):
  install, download, the starter, how to run and check, the library on one page, a minimal
  `Cargo.toml` for one's own project, just enough Rust.
- Lecture: one pseudocode slide per unit (`## <algorithm> as pseudocode`), a plain code
  block, with a note pointing to the coding exercise.
- Exercise page: `## Code {#code}` after Explore, before "Where this leads":
  tasks C1, C2, ... with pseudocode, then questions about what the program shows, and a
  collapsed "Solutions (code)" callout.

## Status

- Day 1 done (images, kernels, fourier, pca), 2026-11-16.
- Days 2–51 done 2026-11-17 by parallel workers following `plan/code-rollout-brief.md`:
  a crate, a `## Code` section and (where the deck has an algorithm) one pseudocode slide per
  unit. Where a page already used C1, C2, … the code tasks are R1… (or K1…, P1…), and the
  section says why.
- Library changes during the rollout: `fft`, `linalg`, `Volume`, `save_labels`,
  `samples::nuclei`; `map`/`map_xy`/`zip_map` take `FnMut`; `sum`/`mean` add in f64;
  `imgcourse` compiled with opt-level 3 in dev builds.

## Shared code (deduplicated 2026-11-17)

- General tools are in the library: `fft`, `linalg`, `filters` (`gaussian_kernel` cut at 4σ,
  `filter_1d` and `gaussian_blur` with the edge copied, `bilinear`, `rmse`, `psnr`), and
  `Rng::poisson` (exact below mean 30, rounded normal above).
- Helpers shared by several units of one day are in that day's `src/common.rs`
  (`pub mod common;`); units re-export them where tests or programs use the unit's path.
- Left as local copies on purpose: anything that differs numerically from the library version
  (3σ kernels, mirror borders, f32 sums, other Poisson cut-offs), so the pages' numbers stay
  valid; and filled-in copies of an earlier unit's exercise.
- Some `common.rs` files hold the filled-in version of an earlier unit's exercise (it was
  already visible in a later unit's folder): day 6 `cubic_bspline`, day 7 `highpass`, day 9
  `projection`/`sinogram`/`backproject`, day 29 `find`, day 42 `srgb_decode`/`srgb_encode`,
  day 44 `pearson`.
- The refactor was checked by running all 209 unit programs before and after: printed output
  (apart from run times) and every picture byte-identical (`plan/code-dedup-brief.md`).
