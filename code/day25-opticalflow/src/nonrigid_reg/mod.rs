//! Coding exercises for the unit "Non-rigid registration".
//!
//! Write your code in `src/nonrigid_reg/exercise.rs`. Check it with `cargo test -p day25 nonrigid_reg::`, and run it on
//! the micrograph with `cargo run -p day25 -- nonrigid-reg` (pictures in `out/nonrigid-reg/`).
//! `src/nonrigid_reg/solution.rs` is a model solution: `cargo test -p day25 --features solution nonrigid_reg::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day25 -- nonrigid-reg`.
pub mod run;
