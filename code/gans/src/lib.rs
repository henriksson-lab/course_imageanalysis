//! Coding exercises for the day "GANs and normalising flows": one module per unit.
//!
//! - `gan`: Generative adversarial networks
//! - `gan_training`: Training GANs and what goes wrong
//! - `image_translation`: Image-to-image translation and hallucination
//! - `flows`: Normalising flows
//!
//! Run a unit's program with `cargo run -p gans -- <unit>`, check it with
//! `cargo test -p gans <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod gan;
pub mod gan_training;
pub mod image_translation;
pub mod flows;
