// The memory model at work: coalescing for strided reads, the read counts of tiles with
// halos (a heat map), the roofline for some image operations, and a fused against an
// unfused pipeline measured on this computer. Numbers on the terminal, pictures in
// out/gpu-memory/.
//   cargo run -p gpus -- gpu-memory                      (your code)
//   cargo run -p gpus --features solution -- gpu-memory  (the model solution)

use crate::gpu_memory::*;
use imgcourse::*;
use std::time::Instant;

pub fn main() {
    let dir = "out/gpu-memory";

    println!("One warp reads 32 floats, every s-th float (stride s):");
    for s in [1, 2, 4, 8, 16, 1024] {
        let t = warp_traffic(&(0..32).map(|lane| 4 * s * lane).collect::<Vec<_>>());
        println!("  stride {s:4}: {:2} lines, {:2} sectors, {:4} bytes moved, efficiency {:5.1} %", t.lines, t.sectors, t.bytes_moved, 100.0 * t.efficiency);
    }

    println!("Global reads per output pixel, 64 × 64 outputs:");
    for (k, tiles) in [(3, vec![1, 4, 8, 16, 32]), (31, vec![1, 8, 16, 32])] {
        for tile in tiles {
            let reads = tile_reads(64, 64, tile, k);
            let per = reads.pixels().iter().sum::<u32>() as f32 / (64.0 * 64.0);
            println!("  {k:2} × {k:<2} filter, tiles of {tile:2}: {per:7.2}");
            if k == 3 && tile == 16 {
                // Enlarged 6 times so that single pixels show: black = 1 read, white = 4.
                let big = Image::from_fn(6 * reads.width(), 6 * reads.height(), |x, y| reads[(x / 6, y / 6)] as f32);
                save_gray_stretched(&big, format!("{dir}/reads-3x3-tile16.png"));
            }
        }
    }

    let (peak, bw) = (3e13, 1e12); // the desktop preset: 30·10¹² op/s, 1 TB/s
    let n = 4096.0 * 4096.0;
    println!("Roofline, desktop preset, a 4096 × 4096 float image:");
    for (name, ops, bytes) in [
        ("add two images", 1.0, 12.0),
        ("multiply by a constant", 1.0, 8.0),
        ("3 × 3 filter, no reuse", 18.0, 40.0),
        ("3 × 3 filter, ideal reuse", 18.0, 8.0),
        ("11 × 11 filter, ideal reuse", 242.0, 8.0),
        ("31 × 31 filter, ideal reuse", 1922.0, 8.0),
    ] {
        let (t, mem) = roofline_time(ops * n, bytes * n, peak, bw);
        println!("  {name:28} I = {:7.3}: {:8.1} µs, {}", ops / bytes, t * 1e6, if mem { "memory-bound" } else { "compute-bound" });
    }

    // Fusing, measured on this CPU: its memory is the bottleneck too.
    let img = Image::from_fn(4096, 4096, |x, y| ((x * 7 + y * 13) % 256) as f32 / 255.0);
    let (mut ms, mut mf) = (Meter::default(), Meter::default());
    let start = Instant::now();
    let a = pipeline_separate(&img, 1.5, 0.1, 0.5, &mut ms);
    let t_sep = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let b = pipeline_fused(&img, 1.5, 0.1, 0.5, &mut mf);
    let t_fused = start.elapsed().as_secs_f64();
    assert!(a.pixels() == b.pixels());
    println!("Pipeline on a 4096 × 4096 image, on this computer:");
    println!("  four passes: {:4.1} bytes per pixel, {:6.1} ms", ms.bytes as f64 / n, 1e3 * t_sep);
    println!("  fused:       {:4.1} bytes per pixel, {:6.1} ms", mf.bytes as f64 / n, 1e3 * t_fused);
    println!("wrote the pictures to {dir}/");
}
