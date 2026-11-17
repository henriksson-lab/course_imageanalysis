//! Coding exercises for day 29, "Graph-based segmentation": one module per unit.
//!
//! - `random_walker`: The random walker
//! - `mst_seg`: Spanning trees: single linkage and FH
//! - `ws_msf`: Watershed as a minimum spanning forest
//! - `merge_trees`: Hierarchical segmentation and merge trees
//! - `superpixels`: Superpixels: SLIC and smaller graphs
//!
//! Run a unit's program with `cargo run -p day29 -- <unit>`, check it with
//! `cargo test -p day29 <unit>::` (see the course page "Coding exercises: getting started").

pub mod random_walker;
pub mod mst_seg;
pub mod ws_msf;
pub mod merge_trees;
pub mod superpixels;
