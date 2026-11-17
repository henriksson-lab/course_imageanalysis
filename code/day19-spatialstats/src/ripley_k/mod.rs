//! Coding exercises for the unit "Ripley's K, the L function and the pair correlation".
//!
//! Write your code in `src/ripley_k/exercise.rs`. Check it with `cargo test -p day19 ripley_k::`, and run it
//! with `cargo run -p day19 -- ripley-k` (results in `out/ripley-k/`).
//! `src/ripley_k/solution.rs` is a model solution: `cargo test -p day19 --features solution ripley_k::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day19 -- ripley-k`.
pub mod run;
