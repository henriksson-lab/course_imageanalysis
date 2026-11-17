//! Coding exercises for the unit "Fitting a mixture: the EM algorithm".
//!
//! Write your code in `src/em/exercise.rs`. Check it with `cargo test -p day35 em::`, and run it on
//! examples with `cargo run -p day35 -- em` (pictures in `out/em/`).
//! `src/em/solution.rs` is a model solution: `cargo test -p day35 --features solution em::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day35 -- em`.
pub mod run;
