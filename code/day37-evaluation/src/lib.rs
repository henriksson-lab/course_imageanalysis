//! Coding exercises for day 37, "Evaluation and statistics": one module per unit.
//!
//! - `object_eval`: Matching objects: IoU, F1 and average precision
//! - `agreement`: Boundaries, counts and the uncertain ground truth
//! - `leakage`: Honest validation: leakage, splits and domain shift
//! - `replicates`: From measurements to claims
//!
//! Run a unit's program with `cargo run -p day37 -- <unit>`, check it with
//! `cargo test -p day37 <unit>::` (see the course page "Coding exercises: getting started").

pub mod object_eval;
pub mod agreement;
pub mod leakage;
pub mod replicates;
