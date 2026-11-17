// The programs of day 6, one per unit:  cargo run -p day06 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day06::*;

/// Day 6: Splines and NURBS. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day06")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Polynomial wiggles, piecewise cubics and Bézier curves
    #[command(name = "bezier")]
    Bezier,
    /// B-splines and NURBS
    #[command(name = "bsplines")]
    Bsplines,
    /// Interpolating, approximating and smoothing splines
    #[command(name = "spline-fit")]
    SplineFit,
    /// Splines on images: interpolation and free-form deformation
    #[command(name = "spline-images")]
    SplineImages,
}

fn main() {
    match Cli::parse().unit {
        Unit::Bezier => bezier::run::main(),
        Unit::Bsplines => bsplines::run::main(),
        Unit::SplineFit => spline_fit::run::main(),
        Unit::SplineImages => spline_images::run::main(),
    }
}
