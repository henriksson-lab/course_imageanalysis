//! YOLO11 nano (the YOLOv11-rs port, Burn) for the browser demo `yolonet`.
//!
//! What runs here: the whole network (backbone, feature pyramid, detection head) on Burn's
//! CPU backend (burn-flex, compiled to wasm32 with SIMD128), plus the decoding of the raw
//! head output into boxes and greedy non-maximum suppression, as plain functions over flat
//! arrays. The model code in `model/` is vendored from YOLOv11-rs (see README.md); this file
//! is the glue.
//!
//! Layouts (A = number of anchor points = sum over the three scales of (size/stride)^2,
//! scales in the order stride 8, 16, 32, each row-major y then x):
//! - input: 3 x size x size, RGB in 0..1, channel-major (CHW); size a multiple of 32.
//! - raw (`Yolo::run`): A x 144 anchor-major: 4 x 16 box-distance logits (left, top, right,
//!   bottom; 16 bins each), then 80 class logits.
//! - decoded (`decode`): A x 88 anchor-major: x1, y1, x2, y2 (input pixels), the four
//!   expected distances l, t, r, b (in grid cells of that scale), then 80 class
//!   probabilities (sigmoid).
//! - detections (`nms`): rows of 7: x1, y1, x2, y2, score, class, anchor index.
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub mod model;

use model::model::{yolo_v11_n, YOLOOutput, YOLO};

/// Number of classes of the COCO-trained checkpoint.
pub const NC: usize = 80;
/// Bins of the box-distance distribution (distribution focal loss) per side.
pub const REG_MAX: usize = 16;
/// Raw outputs per anchor: 4 sides x 16 bins + 80 classes.
pub const NO: usize = 4 * REG_MAX + NC;
/// Decoded values per anchor: box (4), distances (4), class probabilities (80).
pub const ND: usize = 8 + NC;
/// Strides of the three output scales.
pub const STRIDES: [usize; 3] = [8, 16, 32];

/// Number of anchor points for a square input of side `size`.
pub fn anchor_count(size: usize) -> usize {
    STRIDES.iter().map(|s| (size / s) * (size / s)).sum()
}

/// Rewrite a safetensors file so every F16 tensor is stored as F32 (the network computes
/// in f32; the checkpoint is shipped in f16 to halve the download).
pub fn widen_f16(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let n = u64::from_le_bytes(bytes.get(..8).ok_or("not safetensors")?.try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(bytes.get(8..8 + n).ok_or("bad header")?).map_err(|e| e.to_string())?;
    let data = &bytes[8 + n..];
    let mut entries: Vec<(String, serde_json::Value)> = header.into_iter().collect();
    entries.sort_by_key(|(k, v)| (k != "__metadata__", v["data_offsets"][0].as_u64().unwrap_or(0)));
    let (mut out_header, mut body) = (serde_json::Map::new(), Vec::with_capacity(data.len() * 2));
    for (k, mut v) in entries {
        if k == "__metadata__" {
            out_header.insert(k, v);
            continue;
        }
        let s = v["data_offsets"][0].as_u64().ok_or("offsets")? as usize;
        let e = v["data_offsets"][1].as_u64().ok_or("offsets")? as usize;
        let raw = data.get(s..e).ok_or("tensor out of range")?;
        let start = body.len();
        if v["dtype"] == "F16" {
            for c in raw.chunks_exact(2) {
                body.extend_from_slice(&half::f16::from_le_bytes([c[0], c[1]]).to_f32().to_le_bytes());
            }
            v["dtype"] = "F32".into();
        } else {
            body.extend_from_slice(raw);
        }
        v["data_offsets"] = serde_json::json!([start, body.len()]);
        out_header.insert(k, v);
    }
    let mut h = serde_json::to_vec(&out_header).map_err(|e| e.to_string())?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + body.len());
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&body);
    Ok(out)
}

fn device() -> Device {
    burn::tensor::Device::flex()
}

/// The network with loaded weights (BatchNorm folded into the convolutions).
pub struct Net {
    model: YOLO,
    /// Short load report: tensors applied / missing / unused.
    pub report: String,
}

impl Net {
    /// Build YOLO11n (80 classes) and load a safetensors checkpoint written by YOLOv11-rs
    /// `convert_weights.py` (f16 or f32). Its key names follow the PyTorch module tree
    /// (`head.box_branches.i.cK`, BatchNorm `weight`/`bias`); they are mapped to the Burn
    /// field names of the vendored model here.
    pub fn from_safetensors(bytes: Vec<u8>) -> Result<Net, String> {
        use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
        let dev = device();
        let mut model = yolo_v11_n(NC, &dev);
        let mut store = SafetensorsStore::from_bytes(Some(widen_f16(&bytes)?))
            .with_key_remapping(r"^head\.box_branches\.(\d+)\.c(\d)\.", "head.box_c$2.$1.")
            .with_key_remapping(r"^head\.cls_branches\.(\d+)\.c(\d)\.", "head.cls_c$2.$1.")
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true);
        let result = model.load_from(&mut store).map_err(|e| format!("loading weights: {e}"))?;
        // The DFL weight (0..15) and the strides are constants built by the model itself.
        let missing: Vec<&str> = result
            .missing
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !k.ends_with("dfl.weight") && *k != "stride")
            .collect();
        let unused: Vec<&str> = result
            .unused
            .iter()
            .map(|k| k.as_str())
            .filter(|k| !k.ends_with("dfl.conv.weight"))
            .collect();
        if !missing.is_empty() || !result.errors.is_empty() {
            return Err(format!(
                "weights do not match YOLO11n: missing {:?}, errors {:?}",
                &missing[..missing.len().min(8)],
                result.errors.iter().take(4).map(|e| format!("{e:?}")).collect::<Vec<_>>()
            ));
        }
        let report = format!(
            "{} tensors loaded, {} missing, {} unused",
            result.applied.len(),
            missing.len(),
            unused.len()
        );
        Ok(Net { model: model.fuse(), report })
    }

    fn input(&self, input: &[f32], size: usize) -> Tensor<4> {
        assert!(size % 32 == 0 && size >= 64, "size must be a multiple of 32 (>= 64)");
        assert_eq!(input.len(), 3 * size * size, "input must be 3 x size x size");
        Tensor::<1>::from_floats(input, &device()).reshape([1, 3, size, size])
    }

    /// The raw head output per anchor (A x 144, see the module doc), scales concatenated.
    pub fn run(&self, input: &[f32], size: usize) -> Vec<f32> {
        let maps = match self.model.forward(self.input(input, size), true) {
            YOLOOutput::Train(maps) => maps,
            YOLOOutput::Infer(_) => unreachable!(),
        };
        let mut out = vec![0f32; anchor_count(size) * NO];
        let mut base = 0;
        for m in maps {
            let [_, c, h, w] = m.dims();
            assert_eq!(c, NO);
            let v: Vec<f32> = m.to_data().convert::<f32>().try_into_vec::<f32>().unwrap();
            // channel-major [c, h*w] -> anchor-major [h*w, c]
            for a in 0..h * w {
                for k in 0..c {
                    out[(base + a) * NO + k] = v[k * h * w + a];
                }
            }
            base += h * w;
        }
        out
    }

    /// The crate's own inference output ([4 + 80] x A channel-major: cx, cy, w, h in input
    /// pixels, then class probabilities), for checking `decode` against it.
    pub fn infer_upstream(&self, input: &[f32], size: usize) -> Vec<f32> {
        match self.model.forward(self.input(input, size), false) {
            YOLOOutput::Infer(t) => t.to_data().convert::<f32>().try_into_vec::<f32>().unwrap(),
            YOLOOutput::Train(_) => unreachable!(),
        }
    }

    /// The crate's NMS (`model::nms`) on its own inference output: rows x1 y1 x2 y2 score class.
    pub fn detect_upstream(&self, input: &[f32], size: usize, conf: f32, iou: f32) -> Vec<[f32; 6]> {
        match self.model.forward(self.input(input, size), false) {
            YOLOOutput::Infer(t) => model::nms::non_max_suppression(&t, conf, iou).remove(0),
            YOLOOutput::Train(_) => unreachable!(),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode the raw head output (A x 144) into boxes and class probabilities (A x 88).
/// Per anchor at grid cell (i, j) of a scale with stride s, the centre is (j + 0.5, i + 0.5)
/// cells; each side's distance is the expectation of a softmax over 16 bins (0..15 cells);
/// the box is centre - (l, t) .. centre + (r, b), times s.
pub fn decode_raw(raw: &[f32], size: usize) -> Vec<f32> {
    let n = anchor_count(size);
    assert_eq!(raw.len(), n * NO, "raw must be A x 144");
    let mut out = vec![0f32; n * ND];
    let mut a = 0;
    for &s in STRIDES.iter() {
        let g = size / s;
        for i in 0..g {
            for j in 0..g {
                let r = &raw[a * NO..(a + 1) * NO];
                let o = &mut out[a * ND..(a + 1) * ND];
                let mut d = [0f32; 4];
                for (side, dv) in d.iter_mut().enumerate() {
                    let bins = &r[side * REG_MAX..(side + 1) * REG_MAX];
                    let m = bins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let (mut z, mut e) = (0f32, 0f32);
                    for (k, &b) in bins.iter().enumerate() {
                        let p = (b - m).exp();
                        z += p;
                        e += p * k as f32;
                    }
                    *dv = e / z;
                }
                let (cx, cy) = (j as f32 + 0.5, i as f32 + 0.5);
                let sf = s as f32;
                o[0] = (cx - d[0]) * sf;
                o[1] = (cy - d[1]) * sf;
                o[2] = (cx + d[2]) * sf;
                o[3] = (cy + d[3]) * sf;
                o[4..8].copy_from_slice(&d);
                for c in 0..NC {
                    o[8 + c] = sigmoid(r[4 * REG_MAX + c]);
                }
                a += 1;
            }
        }
    }
    out
}

fn iou(a: &[f32], b: &[f32]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Greedy NMS on decoded anchors (A x 88), with the crate's semantics: every (anchor, class)
/// pair with probability > `conf` is a candidate; candidates sorted by score (stable);
/// a candidate is suppressed by a kept one of the same class (or any class when
/// `class_agnostic`) with IoU > `iou_thr`; at most `max_det` kept.
/// Returns rows of 7: x1, y1, x2, y2, score, class, anchor index.
pub fn nms_decoded(dec: &[f32], conf: f32, iou_thr: f32, max_det: usize, class_agnostic: bool) -> Vec<f32> {
    assert_eq!(dec.len() % ND, 0);
    let n = dec.len() / ND;
    let mut cand: Vec<(f32, usize, usize)> = Vec::new(); // score, class, anchor
    for a in 0..n {
        for c in 0..NC {
            let p = dec[a * ND + 8 + c];
            if p > conf {
                cand.push((p, c, a));
            }
        }
    }
    cand.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let mut keep: Vec<(f32, usize, usize)> = Vec::new();
    let mut suppressed = vec![false; cand.len()];
    for i in 0..cand.len() {
        if suppressed[i] {
            continue;
        }
        if keep.len() >= max_det {
            break;
        }
        let bi = &dec[cand[i].2 * ND..cand[i].2 * ND + 4];
        for j in i + 1..cand.len() {
            if suppressed[j] || (!class_agnostic && cand[j].1 != cand[i].1) {
                continue;
            }
            let bj = &dec[cand[j].2 * ND..cand[j].2 * ND + 4];
            if iou(bi, bj) > iou_thr {
                suppressed[j] = true;
            }
        }
        keep.push(cand[i]);
    }
    let mut out = Vec::with_capacity(keep.len() * 7);
    for (p, c, a) in keep {
        out.extend_from_slice(&dec[a * ND..a * ND + 4]);
        out.extend_from_slice(&[p, c as f32, a as f32]);
    }
    out
}

// ---- wasm-bindgen API ------------------------------------------------------------------------

// One network per module instance (free functions rather than an exported class, so the
// generated TypeScript types need no Symbol.dispose).
thread_local! {
    static NET: std::cell::RefCell<Option<Net>> = const { std::cell::RefCell::new(None) };
}

/// Load the network from the bytes of the safetensors checkpoint; returns the load report.
#[wasm_bindgen]
pub fn load(weights: &[u8]) -> Result<String, JsError> {
    let net = Net::from_safetensors(weights.to_vec()).map_err(|e| JsError::new(&e))?;
    let report = net.report.clone();
    NET.with(|n| *n.borrow_mut() = Some(net));
    Ok(report)
}

/// Raw head output for one input (3 x size x size, RGB 0..1, CHW): A x 144 (see the crate doc).
#[wasm_bindgen]
pub fn run(input: &[f32], size: usize) -> Result<Vec<f32>, JsError> {
    NET.with(|n| match n.borrow().as_ref() {
        Some(net) => Ok(net.run(input, size)),
        None => Err(JsError::new("network not loaded")),
    })
}

/// Decode raw (A x 144) into A x 88: box, distances, class probabilities.
#[wasm_bindgen]
pub fn decode(raw: &[f32], size: usize) -> Vec<f32> {
    decode_raw(raw, size)
}

/// Greedy NMS on decoded anchors; rows of 7 (x1, y1, x2, y2, score, class, anchor).
#[wasm_bindgen]
pub fn nms(decoded: &[f32], conf: f32, iou: f32, max_det: usize, class_agnostic: bool) -> Vec<f32> {
    nms_decoded(decoded, conf, iou, max_det, class_agnostic)
}

/// Number of anchor points for a square input of side `size`.
#[wasm_bindgen]
pub fn anchors(size: usize) -> usize {
    anchor_count(size)
}
