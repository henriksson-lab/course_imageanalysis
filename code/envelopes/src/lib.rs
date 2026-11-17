//! Coding exercises for the unit "Is it more than chance? Simulation envelopes".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p envelopes`, and run it
//! with `cargo run -p envelopes` (results in `out/envelopes/`).
//! `src/solution.rs` is a model solution: `cargo test -p envelopes --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
