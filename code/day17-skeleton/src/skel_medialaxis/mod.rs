//! Coding exercises for the unit "Medial axis, distance transform and reconstruction".
//!
//! Write your code in `src/skel_medialaxis/exercise.rs`. Check it with `cargo test -p day17 skel_medialaxis::`, and run it with
//! `cargo run -p day17 -- skel-medialaxis` (pictures in `out/skel-medialaxis/`).
//! `src/skel_medialaxis/solution.rs` is a model solution: `cargo test -p day17 --features solution skel_medialaxis::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day17 -- skel-medialaxis`.
pub mod run;
