// The programs of day 10, one per unit:  cargo run -p day10 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day10::*;

/// Day 10: Edge-preserving smoothing and PDEs. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day10")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Smoothing as heat flow: the heat equation and explicit schemes
    #[command(name = "heat-equation")]
    HeatEquation,
    /// Edge-stopping diffusion: Perona–Malik
    #[command(name = "perona-malik")]
    PeronaMalik,
    /// Total variation flow: smoothing as energy descent
    #[command(name = "tv-flow")]
    TvFlow,
    /// Neighbourhood filters in depth: bilateral and non-local means
    #[command(name = "bilateral-nlm")]
    BilateralNlm,
    /// Filling holes: diffusion and exemplar inpainting
    #[command(name = "inpainting")]
    Inpainting,
    /// All smoothers on one image
    #[command(name = "smoothers")]
    Smoothers,
}

fn main() {
    match Cli::parse().unit {
        Unit::HeatEquation => heat_equation::run::main(),
        Unit::PeronaMalik => perona_malik::run::main(),
        Unit::TvFlow => tv_flow::run::main(),
        Unit::BilateralNlm => bilateral_nlm::run::main(),
        Unit::Inpainting => inpainting::run::main(),
        Unit::Smoothers => smoothers::run::main(),
    }
}
