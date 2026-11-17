//! Coding exercises for the unit "Watershed as a minimum spanning forest".
//!
//! Write your code in `src/ws_msf/exercise.rs`. Check it with `cargo test -p day29 ws_msf::`, and run it with
//! `cargo run -p day29 -- ws-msf` (results in `out/ws-msf/`). `src/ws_msf/solution.rs` is a model solution:
//! `cargo test -p day29 --features solution ws_msf::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day29 -- ws-msf`.
pub mod run;
