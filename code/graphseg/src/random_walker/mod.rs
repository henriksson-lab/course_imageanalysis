//! Coding exercises for the unit "The random walker".
//!
//! Write your code in `src/random_walker/exercise.rs`. Check it with `cargo test -p graphseg random_walker::`, and run it with
//! `cargo run -p graphseg -- random-walker` (results in `out/random-walker/`). `src/random_walker/solution.rs` is a model solution:
//! `cargo test -p graphseg --features solution random_walker::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphseg -- random-walker`.
pub mod run;
