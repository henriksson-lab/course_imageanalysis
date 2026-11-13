//! Native reference run (not part of the Wasm build): the same vendored model on the host
//! CPU, used to produce the reference outputs that demos/test/yolo-wasm-test.ts compares
//! with the Wasm module.
//!   cargo run --release --example reference -- <weights.safetensors> <input.f32> <size> <out dir> [conf]
//! input.f32: 3 x size x size little-endian f32 (RGB 0..1, CHW). Writes into <out dir>:
//! raw.f32 (A x 144, this crate's `Net::run`), upstream.f32 ((4+80) x A, the crate's own
//! inference decode), detections.json (the crate's NMS at conf [0.25], IoU 0.45).
use std::io::Write;

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).expect("read input");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (wpath, ipath, size, out) = (&args[1], &args[2], args[3].parse::<usize>().unwrap(), &args[4]);
    let out = std::path::Path::new(out);
    std::fs::create_dir_all(out).unwrap();
    let t = std::time::Instant::now();
    let net = yolo::Net::from_safetensors(std::fs::read(wpath).unwrap()).unwrap();
    eprintln!("load: {} ({:.0} ms)", net.report, t.elapsed().as_secs_f64() * 1e3);
    let input = read_f32(ipath);
    let t = std::time::Instant::now();
    let raw = net.run(&input, size);
    eprintln!("run: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    write_f32(&out.join("raw.f32"), &raw);
    let up = net.infer_upstream(&input, size);
    write_f32(&out.join("upstream.f32"), &up);
    // This crate's decode_raw (used in the browser) against the crate's own inference decode.
    let dec = yolo::decode_raw(&raw, size);
    let n = yolo::anchor_count(size);
    let (mut eb, mut ep) = (0f32, 0f32);
    for a in 0..n {
        let d = &dec[a * yolo::ND..(a + 1) * yolo::ND];
        let (cx, cy, w, h) = ((d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0, d[2] - d[0], d[3] - d[1]);
        for (k, v) in [cx, cy, w, h].iter().enumerate() {
            eb = eb.max((v - up[k * n + a]).abs());
        }
        for c in 0..yolo::NC {
            ep = ep.max((d[8 + c] - up[(4 + c) * n + a]).abs());
        }
    }
    eprintln!("decode_raw vs crate decode: max |box| {eb:.2e} px, max |prob| {ep:.2e}");
    let conf: f32 = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(0.25);
    let dets = net.detect_upstream(&input, size, conf, 0.45);
    let rows: Vec<String> = dets
        .iter()
        .map(|d| format!("[{}]", d.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")))
        .collect();
    std::fs::write(out.join("detections.json"), format!("[{}]\n", rows.join(",\n"))).unwrap();
    eprintln!("{} detections (crate NMS, conf {conf}, IoU 0.45)", dets.len());
}
