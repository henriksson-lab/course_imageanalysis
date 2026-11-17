//! Coding exercises for the unit "Medial axis, distance transform and reconstruction".
//!
//! Write your code in `src/skel_medialaxis/exercise.rs`. Check it with `cargo test -p skeleton skel_medialaxis::`, and run it with
//! `cargo run -p skeleton -- skel-medialaxis` (pictures in `out/skel-medialaxis/`).
//! `src/skel_medialaxis/solution.rs` is a model solution: `cargo test -p skeleton --features solution skel_medialaxis::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p skeleton -- skel-medialaxis`.
pub mod run;
