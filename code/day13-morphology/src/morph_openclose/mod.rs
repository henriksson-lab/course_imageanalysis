//! Coding exercises for the unit "Opening, closing and the distance transform".
//!
//! Write your code in `src/morph_openclose/exercise.rs`. Check it with `cargo test -p day13 morph_openclose::`, and run it on
//! synthetic images with `cargo run -p day13 -- morph-openclose` (pictures in `out/morph-openclose/`).
//! `src/morph_openclose/solution.rs` is a model solution: `cargo test -p day13 --features solution morph_openclose::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day13 -- morph-openclose`.
pub mod run;
