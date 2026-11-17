// The programs of this day, one per unit:  cargo run -p svm -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Support vector machines. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "svm")]
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
        Unit::Classify => svm::classify::run::main(),
        Unit::Maxmargin => svm::maxmargin::run::main(),
        Unit::KernelTrick => svm::kernel_trick::run::main(),
        Unit::PixelClassify => svm::pixel_classify::run::main(),
    }
}
