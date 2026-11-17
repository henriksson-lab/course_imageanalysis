//! Coding exercises for the unit "Measuring neurons and vessel networks".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p skel-neurovessel`, and run it with
//! `cargo run -p skel-neurovessel` (pictures in `out/skel-neurovessel/`).
//! `src/solution.rs` is a model solution: `cargo test -p skel-neurovessel --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
