//! Coding exercises for the unit "The random walker".
//!
//! Write your code in `src/random_walker/exercise.rs`. Check it with `cargo test -p day29 random_walker::`, and run it with
//! `cargo run -p day29 -- random-walker` (results in `out/random-walker/`). `src/random_walker/solution.rs` is a model solution:
//! `cargo test -p day29 --features solution random_walker::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day29 -- random-walker`.
pub mod run;
