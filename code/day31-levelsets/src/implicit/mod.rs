//! Coding exercises for the unit "Curves as level sets".
//!
//! Write your code in `src/implicit/exercise.rs`. Check it with `cargo test -p day31 implicit::`, and run it
//! with `cargo run -p day31 -- implicit` (pictures in `out/implicit/`).
//! `src/implicit/solution.rs` is a model solution: `cargo test -p day31 --features solution implicit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day31 -- implicit`.
pub mod run;
