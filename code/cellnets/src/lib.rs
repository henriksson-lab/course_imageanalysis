//! Coding exercises for the day "Networks for cells and objects": one module per unit.
//!
//! - `stardist`: Star-convex polygons
//! - `cellpose`: Flow fields to objects
//! - `yolo`: From an image to a list of objects
//! - `sam`: Promptable segmentation
//! - `cellnets_compare`: Networks and classical pipelines compared
//!
//! Run a unit's program with `cargo run -p cellnets -- <unit>`, check it with
//! `cargo test -p cellnets <unit>::` (see the course page "Coding exercises: getting started").

pub mod stardist;
pub mod cellpose;
pub mod yolo;
pub mod sam;
pub mod cellnets_compare;
