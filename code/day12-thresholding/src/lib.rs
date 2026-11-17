//! Coding exercises for day 12, "Thresholding": one module per unit.
//!
//! - `histogram`: Histograms & a global threshold
//! - `otsu`: Choosing the threshold: Otsu & minimum error
//! - `local_threshold`: Local thresholds & hysteresis
//! - `seg_eval`: Evaluation: Dice, ROC & precision–recall
//!
//! Run a unit's program with `cargo run -p day12 -- <unit>`, check it with
//! `cargo test -p day12 <unit>::` (see the course page "Coding exercises: getting started").

pub mod histogram;
pub mod otsu;
pub mod local_threshold;
pub mod seg_eval;
