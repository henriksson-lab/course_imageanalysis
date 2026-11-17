//! Coding exercises for the unit "Medial axis, distance transform and reconstruction".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p skel-medialaxis`, and run it with
//! `cargo run -p skel-medialaxis` (pictures in `out/skel-medialaxis/`).
//! `src/solution.rs` is a model solution: `cargo test -p skel-medialaxis --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
