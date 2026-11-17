//! Coding exercises for the unit "Is it more than chance? Simulation envelopes".
//!
//! Write your code in `src/envelopes/exercise.rs`. Check it with `cargo test -p day19 envelopes::`, and run it
//! with `cargo run -p day19 -- envelopes` (results in `out/envelopes/`).
//! `src/envelopes/solution.rs` is a model solution: `cargo test -p day19 --features solution envelopes::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day19 -- envelopes`.
pub mod run;
