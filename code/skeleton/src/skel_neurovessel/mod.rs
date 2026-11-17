//! Coding exercises for the unit "Measuring neurons and vessel networks".
//!
//! Write your code in `src/skel_neurovessel/exercise.rs`. Check it with `cargo test -p skeleton skel_neurovessel::`, and run it with
//! `cargo run -p skeleton -- skel-neurovessel` (pictures in `out/skel-neurovessel/`).
//! `src/skel_neurovessel/solution.rs` is a model solution: `cargo test -p skeleton --features solution skel_neurovessel::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p skeleton -- skel-neurovessel`.
pub mod run;
