//! Coding exercises for the unit "Least-squares unmixing".
//!
//! Write your code in `src/lsunmix/exercise.rs`. Check it with `cargo test -p day43 lsunmix::`, and run it
//! with `cargo run -p day43 -- lsunmix` (pictures in `out/lsunmix/`).
//! `src/lsunmix/solution.rs` is a model solution: `cargo test -p day43 --features solution lsunmix::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day43 -- lsunmix`.
pub mod run;
