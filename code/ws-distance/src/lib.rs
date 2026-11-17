//! Coding exercises for the unit "Splitting touching objects with the distance transform".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ws-distance`, and run it on
//! synthetic nuclei and two discs with `cargo run -p ws-distance` (pictures in `out/ws-distance/`).
//! `src/solution.rs` is a model solution: `cargo test -p ws-distance --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
