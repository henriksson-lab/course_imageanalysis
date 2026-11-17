// The programs of day 34, one per unit:  cargo run -p day34 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day34::*;

/// Day 34: Support vector machines. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day34")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Linear classifiers
    #[command(name = "classify")]
    Classify,
    /// Maximum margin and support vectors
    #[command(name = "maxmargin")]
    Maxmargin,
    /// Kernel functions and the kernel trick
    #[command(name = "kernel-trick")]
    KernelTrick,
    /// SVMs for pixel classification
    #[command(name = "pixel-classify")]
    PixelClassify,
}

fn main() {
    match Cli::parse().unit {
        Unit::Classify => classify::run::main(),
        Unit::Maxmargin => maxmargin::run::main(),
        Unit::KernelTrick => kernel_trick::run::main(),
        Unit::PixelClassify => pixel_classify::run::main(),
    }
}
