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
`docs/downloads/`). Members: `imgcourse`, `starter`, and one crate per unit, named after the
unit folder:

```
code/<unit>/Cargo.toml        feature "solution"
code/<unit>/src/lib.rs        exports exercise.rs, or solution.rs with --features solution
code/<unit>/src/exercise.rs   the same signatures and doc comments as the solution, bodies todo!("C1: name"), hints
code/<unit>/src/solution.rs   model solution
code/<unit>/tests/check.rs    checks, preferably the numbers of the by-hand exercises
code/<unit>/src/main.rs       runs the functions on real images, pictures in out/<unit>/
```

`make code-test` (also in CI): tests with `--all-features` (the solutions), builds the
templates, and runs clippy on both with `-D warnings`. Add a new unit crate to `members`
in `code/Cargo.toml` and to the list on `code.qmd`.

## On the site

- `code.qmd` ("Coding exercises: getting started", in the sidebar before the topics):
  install, download, the starter, how to run and check, the library on one page, a minimal
  `Cargo.toml` for one's own project, just enough Rust.
- Lecture: one pseudocode slide per unit (`## <algorithm> as pseudocode`), a plain code
  block, with a note pointing to the coding exercise.
- Exercise page: `## Code (optional) {#code}` after Explore, before "Where this leads":
  tasks C1, C2, ... with pseudocode, then questions about what the program shows, and a
  collapsed "Solutions (code)" callout.

## Status

- Day 1 done (images, kernels, fourier, pca).
- Next: the other days, by the same pattern, in course order.
