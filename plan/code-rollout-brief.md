> **Historical (2026-11-17):** this brief made one crate per unit; the crates were then merged into
> one crate per day (see `plan/code-plan.md`, Layout). For new work follow the plan, not the
> crate layout below.

# Brief: Rust coding exercises and pseudocode slides for a day

You add code to the units of the day(s) you are given, following the pattern of day 1
(units `images`, `kernels`, `fourier`, `pca`), which is done and approved by the course owner.
Read first, in this order:

1. `plan/code-plan.md` — the decisions and conventions.
2. `designprinciples.md`, section 0 ("Code is optional and comes last", but see below on wording).
3. The worked examples: `code/kernels/` (all files), `kernels/exercise.qmd` (section `## Code`)
   and `kernels/lec.qmd` (slide "Filtering as pseudocode"); also `code/pca/` and `pca/exercise.qmd`.
4. The library: `code/imgcourse/src/*.rs` (short; read it all, so that you use it and do not duplicate it).
5. For each unit of your day: `<unit>/lec.qmd` and `<unit>/exercise.qmd`.

## What to make, per unit

Aim for code in **most** units. Skip a unit only when it has no algorithm that students can
meaningfully write in plain loops (e.g. a pure comparison or discussion unit), and say why in
your report. Size: 2 to 5 tasks (C1, C2, ...) per unit, each a function of roughly 5 to 40 lines
in the solution. Algorithms should be the ones the unit teaches, in their plain form.

### The crate `code/<unit>/` (crate name = unit folder name, e.g. `scale-space`, used as `scale_space` in Rust)

- `Cargo.toml`: copy `code/kernels/Cargo.toml`, change the name.
- `src/lib.rs`: copy `code/kernels/src/lib.rs`, change the unit title and name.
- `src/solution.rs`: the model solution. Plain loops, `img[(x, y)]`, readable over fast;
  comments that point to the slides. Short doc comment per function starting `/// C1. ...`.
- `src/exercise.rs`: identical signatures, doc comments and constants; every function body is
  `todo!("C1: name")`; a `/// Hint:` paragraph where useful; first lines as in
  `code/images/src/exercise.rs` (with `#![allow(unused_variables, unused_imports)]`).
  Helper code that is not part of the exercise (data generators, drawing) may stay filled in.
- `tests/check.rs`: checks named `c1_...`, `c2_...`, with messages that say what was expected
  ("row 3, column 2: got 2.5, expected 3"). Prefer the **numbers of the pen-and-paper
  exercises** of `<unit>/exercise.qmd` (cite them: "B4"), then properties (invariance,
  inverse undoes, sums, known simple cases). Tests must run in well under a second each.
- `src/main.rs`: runs the functions on real or synthetic images and writes pictures to
  `out/<unit>/` (and/or prints numbers). Must finish in about 10 s or less with
  `cargo run -p <unit> --features solution` (the dev profile is optimised, opt-level 1).
- Register the crate (after `Cargo.toml` exists): `python3 tools/code_add_member.py <unit>`.
  Never edit `code/Cargo.toml` by hand.

### The library and its limits

`imgcourse` has: `Image<T>` (`new`, `from_fn`, `from_vec`, `from_rows`, `[(x, y)]`, `get_border`
with `Border::{Zero, Clamp, Mirror, Wrap}`, `coords`, `map`, `map_xy`, `zip_map`, `crop`, `row`,
`pixels`, `min/max/sum/mean`, `Display`), `Volume<T>` (3D), `Complex`, `Rng`, file reading and
writing (`load_gray`, `load_rgb`, `save_gray`, `save_gray_stretched`, `save_gray_signed`,
`save_rgb`, `save_labels`, `gray_to_rgb`, `io::label_colours`), `fft` (`fft`, `fft2`, `fft2_real`,
`frequency`, `spectrum_picture`), `linalg` (`solve`, `least_squares`, `matmul`, `transpose`,
`sym_eigen`, in f64), `samples` (`cell()` the 256 × 256 course micrograph, `letter_f`, `step`,
`checker`, `add_noise`, `nuclei(size, n, seed)` → synthetic fluorescence image + true labels).

- **Do not edit `code/imgcourse/`.** If you need more plumbing, write it in your crate (filled
  in, not an exercise) and propose it for the library in your report.
- Do not use a library tool for the very thing the unit teaches (no `fft` in a unit that
  teaches the FFT; no `sym_eigen` where the eigenvectors are the exercise). Using them in later
  units as building blocks is fine and intended.
- No new dependencies (no crates.io crates). `std` only, plus `imgcourse`.
- Conventions: `(x, y)` = (column, row) from 0, `y` down (positive angles turn clockwise on
  screen); grey values `f32` in [0, 1]; labels `u32`, 0 = background.

### Building and checking (each agent its own target directory; only your crates)

```
cd code
export CARGO_TARGET_DIR=$PWD/target/<your-day-id>
cargo test   -p <unit> --features solution        # all checks pass on the solution
cargo test   -p <unit>                            # compiles; checks fail with todo!() (expected)
cargo build  -p <unit> --all-targets
cargo clippy -p <unit> --all-targets --features solution -- -D warnings -A clippy::needless_range_loop
cargo clippy -p <unit> --all-targets                    -- -D warnings -A clippy::needless_range_loop
cargo run    -p <unit> --features solution        # look at the pictures you produce (Read the PNGs)
```

Never run `--workspace` commands (other agents' crates may be half-written), never `cargo clean`
on the shared target, never delete `code/out` folders of others.

### The exercise page: `<unit>/exercise.qmd`

Add a section `## Code {#code}` after the last Explore section and its solutions, before
`## Where this leads` (at the end if there is none). Copy the shape of `kernels/exercise.qmd`:

- Intro: "Write the algorithms of this unit yourself, in Rust. Setup, and how to run and check
  your code: [coding exercises: getting started](../code.qmd). Your code goes in `<unit>/src/exercise.rs`."
  **Never call the code "optional"** in student material, and do not say that nothing depends on it.
- Tasks **C1, C2, ...** (if the page already uses C1, C2, ... for other questions, use
  **R1, R2, ...** for the code tasks, also in that crate's comments): the function name in backticks, what it does in words, and the
  algorithm as **pseudocode** in a plain fenced block (no language tag): plain steps close to
  the math, not Rust. Say which check uses which by-hand exercise.
- Then "Then `cargo run -p <unit>` and open the pictures in `out/<unit>/`." and 1 to 3
  questions about what the program shows (predict / explain / compare / count the cost);
  *(math)* tag for derivation questions.
- A collapsed callout `::: {.callout-note collapse="true" title="Solutions (code)"}` starting
  "The model solution is `<unit>/src/solution.rs` in the download." and answering the questions.
  **Every number or visual claim in it must come from actually running your program** (look
  at the PNGs with the Read tool).
- No Rust code on the page (pseudocode only). Keep the page's existing content untouched.

### The lecture: `<unit>/lec.qmd`

At most **one** pseudocode slide per deck, and only if the deck presents an algorithm:
`## <Algorithm> as pseudocode`, placed right after the slide that introduces the algorithm
(never inside the math layer, never on a `{.math}` slide). A plain fenced code block of at
most ~10 lines, then at most 3 short bullets, then `::: notes` with one or two sentences ending
"The coding exercise (<unit>/ in the course code download) implements ...". It must fit a
1280 × 720 slide. Skip the slide if the deck already has equivalent pseudocode.

### Style

As day 1: plain words, short sentences, British spelling, no tool or library names in student
material, the two layers (concept first, math marked). Code comments in English, short.

### Do not

- Do not render with Quarto, commit, stash, reset or checkout; do not touch files of other
  units, `days.yml`, `_quarto.yml`, `code.qmd`, `Makefile`, `code/imgcourse/`, `code/Cargo.toml`
  (use the helper), or `docs/`.

### Report (your final message)

Per unit: crate made (tasks C1.. with one line each) or skipped (why); whether a pseudocode
slide was added; check results (tests pass on solution, clippy clean, run time of main);
anything you could not verify; proposals for the library.
