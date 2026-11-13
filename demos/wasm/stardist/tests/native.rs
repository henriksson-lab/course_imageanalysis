//! Native reference for the Wasm build (run: `cargo test --release -- --nocapture` in this
//! folder, after `make models`). Skips when the model files are missing.
//! 1. src/h5.rs (from bytes) gives exactly what the crate's HDF5 loader (from a path) gives.
//! 2. Runs the network and the decoding on a 128 x 128 crop of the StarDist sample image and
//!    writes testdata/native-ref.json, which demos/test/stardist-wasm-test.ts compares the Wasm
//!    module against (same code, other target: catches miscompilation and glue errors).
//! 3. Times the full 512 x 512 image natively (for the README benchmark).
use std::path::PathBuf;
use std::time::Instant;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dist/models")
}

fn load() -> Option<(String, String, Vec<u8>, image::GrayImage)> {
    let m = model_dir();
    let d = m.join("stardist/2D_versatile_fluo");
    let cfg = std::fs::read_to_string(d.join("config.json")).ok()?;
    let thr = std::fs::read_to_string(d.join("thresholds.json")).ok()?;
    let w = std::fs::read(d.join("weights_best.h5")).ok()?;
    let img = image::open(m.join("images/stardist-img2d.png")).ok()?.to_luma8();
    Some((cfg, thr, w, img))
}

const CROP: (u32, u32, u32) = (192, 192, 128); // y0, x0, size

#[test]
fn reference() {
    let Some((cfg, thr, wbytes, img)) = load() else {
        eprintln!("SKIP: model files missing (run make models)");
        return;
    };
    // 1. HDF5 reader.
    let ours = stardist::h5::load_keras_weights(&wbytes).unwrap();
    let theirs = stardist_rs::weights::load_keras_hdf5_weights(
        model_dir().join("stardist/2D_versatile_fluo/weights_best.h5"),
    )
    .unwrap();
    assert_eq!(ours.tensors.len(), theirs.tensors.len());
    assert_eq!(ours, theirs, "h5.rs differs from the crate's HDF5 loader");
    eprintln!("h5: {} tensors identical", ours.tensors.len());

    let t0 = Instant::now();
    let net = stardist::Net::new(&cfg, &thr, &wbytes).unwrap();
    eprintln!("load: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    // 2. Crop reference.
    let (y0, x0, n) = CROP;
    let mut crop = Vec::with_capacity((n * n) as usize);
    for y in y0..y0 + n {
        for x in x0..x0 + n {
            crop.push(img.get_pixel(x, y)[0]);
        }
    }
    let f: Vec<f32> = crop.iter().map(|&v| v as f32).collect();
    let norm = stardist::normalize_percentile(&f, 1.0, 99.8);
    let p = net.predict(&norm, n as usize, n as usize, stardist::Engine::Candle).unwrap();
    // The SIMD path (fast.rs, here with its portable vector type) against candle.
    let q = net.predict(&norm, n as usize, n as usize, stardist::Engine::Fast).unwrap();
    let dp = p.prob.iter().zip(&q.prob).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let dd = p.dist.iter().zip(&q.dist).map(|(a, b)| (a - b).abs() / a.max(1.0)).fold(0f32, f32::max);
    eprintln!("fast vs candle: prob max |diff| {dp:.2e}, dist max rel diff {dd:.2e}");
    assert!(dp < 1e-4 && dd < 1e-4, "fast.rs differs from candle");
    let (pt, nt) = (net.thresholds.prob, net.thresholds.nms);
    let d = stardist::decode(&p.prob, &p.dist, p.out_w, p.out_h, n as usize, n as usize, 32, 2, pt, nt).unwrap();
    let mut areas = vec![0u32; d.probs.len() + 1];
    for &l in &d.labels {
        areas[l as usize] += 1;
    }
    // Ray distances at every 37th grid cell (keeps the file small).
    let dist_sample: Vec<(usize, Vec<f32>)> = (0..p.out_w * p.out_h)
        .step_by(37)
        .map(|i| (i, p.dist[i * 32..(i + 1) * 32].to_vec()))
        .collect();
    let r6 = |v: f32| (v as f64 * 1e6).round() / 1e6;
    let r4 = |v: f32| (v as f64 * 1e4).round() / 1e4;
    // Pixels as base64 (8 bit).
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut pixels_b64 = String::new();
    for c in crop.chunks(3) {
        let v = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                pixels_b64.push(B64[((v >> (18 - 6 * k)) & 63) as usize] as char);
            } else {
                pixels_b64.push('=');
            }
        }
    }
    let json = serde_json::json!({
        "_doc": "Native (x86-64) output of demos/wasm/stardist on a crop of images/stardist-img2d.png, written by tests/native.rs; checked against the Wasm build by demos/test/stardist-wasm-test.ts",
        "crop": {"y0": y0, "x0": x0, "size": n},
        "pixels_base64": pixels_b64,
        "prob_thresh": pt, "nms_thresh": nt,
        "out_w": p.out_w, "out_h": p.out_h,
        "prob": p.prob.iter().map(|&v| r6(v)).collect::<Vec<_>>(),
        "dist_sample": dist_sample.iter().map(|(i, v)| serde_json::json!({"cell": i, "dist": v.iter().map(|&x| r4(x)).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "candidates": d.candidates,
        "count": d.probs.len(),
        "points": d.points,
        "probs": d.probs.iter().map(|&v| r6(v)).collect::<Vec<_>>(),
        "areas": &areas[1..],
        "background": areas[0],
    });
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/native-ref.json");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(&out, serde_json::to_string(&json).unwrap()).unwrap();
    eprintln!("crop: {} candidates, {} objects -> {}", d.candidates, d.probs.len(), out.display());

    // 3. Full image timing.
    let (w, h) = img.dimensions();
    let f: Vec<f32> = img.as_raw().iter().map(|&v| v as f32).collect();
    let norm = stardist::normalize_percentile(&f, 1.0, 99.8);
    for rep in 0..2 {
        let t = Instant::now();
        let p = net.predict(&norm, w as usize, h as usize, stardist::Engine::Candle).unwrap();
        let tn = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let d = stardist::decode(&p.prob, &p.dist, p.out_w, p.out_h, w as usize, h as usize, 32, 2, pt, nt).unwrap();
        let tp = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let _ = net.predict(&norm, w as usize, h as usize, stardist::Engine::Fast).unwrap();
        let tf = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("full {w}x{h} run {rep}: network candle {tn:.0} ms, fast.rs (portable, no SIMD) {tf:.0} ms, decode {tp:.0} ms, {} objects ({} candidates)", d.probs.len(), d.candidates);
    }
}
