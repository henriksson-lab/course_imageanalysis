//! Coding exercises for the unit "Watershed as a minimum spanning forest".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ws-msf`, and run it with
//! `cargo run -p ws-msf` (results in `out/ws-msf/`). `src/solution.rs` is a model solution:
//! `cargo test -p ws-msf --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
