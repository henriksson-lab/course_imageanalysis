// The programs of this day, one per unit:  cargo run -p gpus -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// How GPUs work. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "gpus")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
#[allow(clippy::enum_variant_names)] // the variants are named after the units
enum Unit {
    /// Many simple cores: throughput, warps and divergence
    #[command(name = "gpu-parallel")]
    GpuParallel,
    /// Memory is the bottleneck
    #[command(name = "gpu-memory")]
    GpuMemory,
    /// Data-parallel patterns
    #[command(name = "gpu-patterns")]
    GpuPatterns,
    /// What GPUs are good and bad at
    #[command(name = "gpu-fit")]
    GpuFit,
}

fn main() {
    match Cli::parse().unit {
        Unit::GpuParallel => gpus::gpu_parallel::run::main(),
        Unit::GpuMemory => gpus::gpu_memory::run::main(),
        Unit::GpuPatterns => gpus::gpu_patterns::run::main(),
        Unit::GpuFit => gpus::gpu_fit::run::main(),
    }
}
