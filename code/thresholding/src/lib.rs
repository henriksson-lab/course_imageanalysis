//! Coding exercises for the day "Thresholding": one module per unit.
//!
//! - `histogram`: Histograms & a global threshold
//! - `otsu`: Choosing the threshold: Otsu & minimum error
//! - `local_threshold`: Local thresholds & hysteresis
//! - `seg_eval`: Evaluation: Dice, ROC & precision–recall
//!
//! Run a unit's program with `cargo run -p thresholding -- <unit>`, check it with
//! `cargo test -p thresholding <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod histogram;
pub mod otsu;
pub mod local_threshold;
pub mod seg_eval;
