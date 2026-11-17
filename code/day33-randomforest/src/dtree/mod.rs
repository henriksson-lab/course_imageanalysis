//! Coding exercises for the unit "Decision trees".
//!
//! Write your code in `src/dtree/exercise.rs`. Check it with `cargo test -p day33 dtree::`, and run it with
//! `cargo run -p day33 -- dtree` (pictures in `out/dtree/`).
//! `src/dtree/solution.rs` is a model solution: `cargo test -p day33 --features solution dtree::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day33 -- dtree`.
pub mod run;
