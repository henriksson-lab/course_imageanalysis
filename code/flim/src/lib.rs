//! Coding exercises for the day "Fluorescence lifetime imaging (FLIM)": one module per unit.
//!
//! - `lifetime`: Lifetimes, decays and photon counting
//! - `decayfit`: Fitting decays with few photons
//! - `phasor`: Frequency domain and phasors
//! - `flim_fret`: FLIM-FRET and metabolic imaging
//!
//! Run a unit's program with `cargo run -p flim -- <unit>`, check it with
//! `cargo test -p flim <unit>::` (see the course page "Coding exercises: getting started").

pub mod lifetime;
pub mod decayfit;
pub mod phasor;
pub mod flim_fret;
