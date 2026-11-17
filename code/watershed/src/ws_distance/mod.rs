//! Coding exercises for the unit "Splitting touching objects with the distance transform".
//!
//! Write your code in `src/ws_distance/exercise.rs`. Check it with `cargo test -p watershed ws_distance::`, and run it on
//! synthetic nuclei and two discs with `cargo run -p watershed -- ws-distance` (pictures in `out/ws-distance/`).
//! `src/ws_distance/solution.rs` is a model solution: `cargo test -p watershed --features solution ws_distance::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p watershed -- ws-distance`.
pub mod run;
