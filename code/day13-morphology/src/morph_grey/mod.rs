//! Coding exercises for the unit "Grey-level morphology: top-hat and gradient".
//!
//! Write your code in `src/morph_grey/exercise.rs`. Check it with `cargo test -p day13 morph_grey::`, and run it on
//! synthetic spots and the micrograph with `cargo run -p day13 -- morph-grey` (pictures in `out/morph-grey/`).
//! `src/morph_grey/solution.rs` is a model solution: `cargo test -p day13 --features solution morph_grey::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day13 -- morph-grey`.
pub mod run;
