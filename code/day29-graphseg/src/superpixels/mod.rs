//! Coding exercises for the unit "Superpixels: SLIC and smaller graphs".
//!
//! Write your code in `src/superpixels/exercise.rs`. Check it with `cargo test -p day29 superpixels::`, and run it with
//! `cargo run -p day29 -- superpixels` (results in `out/superpixels/`). `src/superpixels/solution.rs` is a model solution:
//! `cargo test -p day29 --features solution superpixels::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day29 -- superpixels`.
pub mod run;
