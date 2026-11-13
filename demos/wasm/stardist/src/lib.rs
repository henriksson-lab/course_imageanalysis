//! StarDist 2D in the browser: the network (candle, CPU) and the decoding steps of
//! henriksson-lab/stardist-rs behind a few flat-array functions (demos/README.md, "WebAssembly").
//!
//! - `load(config_json, thresholds_json, weights_h5)`: builds the U-Net from the model's
//!   config.json and loads the Keras weights (`h5.rs`, from bytes).
//! - `predict(img, w, h, engine)`: normalised grey image (row-major) -> probability map
//!   (`prob()`, out_height x out_width, one value per grid cell = 2x2 pixels for
//!   2D_versatile_fluo) and ray distances (`dist()`, out_height x out_width x n_rays, rays last,
//!   in pixels). Padding/cropping (reflect, to a multiple of 16) as StarDist does.
//! - `instances(prob, dist, ...)` + `dec_*` getters: the post-processing alone (no network):
//!   threshold, polygon NMS, polygons -> label image. Used by the demo after each threshold
//!   change, and by the node test on synthetic network outputs.
//! - `normalize_percentile(img, pmin, pmax)`: StarDist's usual input normalisation.
use std::cell::RefCell;

use candle_core::{Device, Tensor};
use stardist_rs::model::candle::StarDist2D as CandleStarDist2D;
use stardist_rs::{Config2D, StarDistThresholds};
use wasm_bindgen::prelude::*;

pub mod fast;
pub mod h5;

fn err<E: std::fmt::Debug>(e: E) -> JsError {
    JsError::new(&format!("{e:?}"))
}

/// Network outputs of one image (already cropped to the image).
pub struct Prediction {
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub out_h: usize,
    pub out_w: usize,
}

/// Native core (also used by the native reference tests in tests/).
pub struct Net {
    pub config: Config2D,
    pub thresholds: StarDistThresholds,
    model: CandleStarDist2D,
    fast: fast::FastNet,
}

/// Which implementation of the network to run (same weights, same outputs up to rounding).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    /// stardist-rs's candle model (CPU backend).
    Candle,
    /// src/fast.rs: direct SIMD convolutions (about 3x faster in Wasm).
    Fast,
}

impl Net {
    pub fn new(config_json: &str, thresholds_json: &str, weights_h5: &[u8]) -> Result<Net, String> {
        let config: Config2D = serde_json::from_str(config_json).map_err(|e| format!("config: {e}"))?;
        let t: serde_json::Value =
            serde_json::from_str(thresholds_json).map_err(|e| format!("thresholds: {e}"))?;
        let get = |k: &str| t.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
        let thresholds = StarDistThresholds::from_options(get("prob"), get("nms")).map_err(|e| format!("{e:?}"))?;
        let weights = h5::load_keras_weights(weights_h5)?;
        let model = CandleStarDist2D::try_init(config.clone(), &Device::Cpu)
            .map_err(|e| format!("{e:?}"))?
            .load_keras_weights(&weights, &Device::Cpu)
            .map_err(|e| format!("{e:?}"))?;
        let fast = fast::FastNet::new(&config, &weights)?;
        Ok(Net { config, thresholds, model, fast })
    }

    pub fn predict(&self, img: &[f32], w: usize, h: usize, engine: Engine) -> Result<Prediction, String> {
        if img.len() != w * h {
            return Err("predict: image length must be width * height".into());
        }
        let sd = stardist_rs::StarDist2D::new(self.config.clone());
        let setup = sd._predict_setup(img, &[h, w], Some("YX"), None).map_err(|e| format!("{e:?}"))?;
        let (hp, wp) = (setup.x_shape[0], setup.x_shape[1]);
        let n_rays = self.config.n_rays;
        if engine == Engine::Fast {
            let (p, d) = self.fast.forward(fast::Map { h: hp, w: wp, c: 1, d: setup.x.clone() });
            let dist: Vec<f32> = d.d.into_iter().map(|v| v.max(1e-3)).collect();
            let (prob, shape) = setup.resizer.after(&p.d, &[p.h, p.w, 1], "YXC").map_err(|e| format!("{e:?}"))?;
            let (dist, _) = setup.resizer.after(&dist, &[d.h, d.w, n_rays], "YXC").map_err(|e| format!("{e:?}"))?;
            return Ok(Prediction { prob, dist, out_h: shape[0], out_w: shape[1] });
        }
        let x = Tensor::from_vec(setup.x.clone(), (1, 1, hp, wp), &Device::Cpu).map_err(|e| e.to_string())?;
        let out = self.model.forward(&x).map_err(|e| e.to_string())?;
        let (_, _, gh, gw) = out.prob.dims4().map_err(|e| e.to_string())?;
        let prob: Vec<f32> = out.prob.flatten_all().and_then(|t| t.to_vec1()).map_err(|e| e.to_string())?;
        // (1, R, gh, gw) -> (gh, gw, R), and StarDist's lower bound on distances.
        let dist: Vec<f32> = out
            .dist
            .squeeze(0)
            .and_then(|t| t.permute((1, 2, 0)))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|d: f32| d.max(1e-3))
            .collect();
        let (prob, shape) = setup.resizer.after(&prob, &[gh, gw, 1], "YXC").map_err(|e| format!("{e:?}"))?;
        let (dist, _) = setup.resizer.after(&dist, &[gh, gw, n_rays], "YXC").map_err(|e| format!("{e:?}"))?;
        Ok(Prediction { prob, dist, out_h: shape[0], out_w: shape[1] })
    }
}

/// Result of the post-processing.
pub struct Decoded {
    pub candidates: usize,
    pub points: Vec<f32>,
    pub probs: Vec<f32>,
    pub dists: Vec<f32>,
    pub coords: Vec<f32>,
    pub labels: Vec<u32>,
}

/// Threshold, polygon NMS (crate's `non_maximum_suppression`, bbox pre-check and kd-tree as in
/// StarDist's defaults) and rendering of the kept polygons into a label image (crate's
/// `polygons_to_label`: lower-scored polygons painted first, so higher scores win overlaps).
#[allow(clippy::too_many_arguments)]
pub fn decode(
    prob: &[f32],
    dist: &[f32],
    out_w: usize,
    out_h: usize,
    img_w: usize,
    img_h: usize,
    n_rays: usize,
    grid: usize,
    prob_thresh: f32,
    nms_thresh: f32,
) -> Result<Decoded, String> {
    let candidates = prob.iter().filter(|&&p| p > prob_thresh).count();
    let nms = stardist_rs::nms::non_maximum_suppression(
        dist,
        prob,
        [out_h, out_w],
        n_rays,
        [grid, grid],
        None,
        nms_thresh,
        prob_thresh,
        true,
        true,
    )
    .map_err(|e| format!("{e:?}"))?;
    let labels = stardist_rs::geometry::polygons_to_label(
        &nms.dist,
        &nms.points,
        [img_h, img_w],
        Some(&nms.prob),
        f32::NEG_INFINITY,
        [1.0, 1.0],
    )
    .map_err(|e| format!("{e:?}"))?;
    let coords = if nms.points.is_empty() {
        Vec::new()
    } else {
        stardist_rs::geometry::dist_to_coord(&nms.dist, &nms.points, n_rays, [1.0, 1.0])
            .map_err(|e| format!("{e:?}"))?
            .iter()
            .copied()
            .collect()
    };
    Ok(Decoded {
        candidates,
        points: nms.points.iter().flat_map(|p| [p[0], p[1]]).collect(),
        probs: nms.prob,
        dists: nms.dist,
        coords,
        labels: labels.iter().copied().collect(),
    })
}

/// Percentile normalisation (StarDist/CSBDeep `normalize(img, pmin, pmax)`, no clipping):
/// (x - p_lo) / (p_hi - p_lo + 1e-20), percentiles with linear interpolation as NumPy.
#[wasm_bindgen]
pub fn normalize_percentile(img: &[f32], pmin: f32, pmax: f32) -> Vec<f32> {
    if img.is_empty() {
        return Vec::new();
    }
    let mut s: Vec<f32> = img.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f32| -> f64 {
        let pos = (p as f64 / 100.0).clamp(0.0, 1.0) * (s.len() - 1) as f64;
        let i = pos.floor() as usize;
        let j = (i + 1).min(s.len() - 1);
        let t = pos - i as f64;
        s[i] as f64 * (1.0 - t) + s[j] as f64 * t
    };
    let (lo, hi) = (q(pmin), q(pmax));
    let d = hi - lo + 1e-20;
    img.iter().map(|&v| ((v as f64 - lo) / d) as f32).collect()
}

// ---- Wasm interface ------------------------------------------------------------------------------
// Free functions over module state (one network per page/worker), no exported classes: keeps the
// generated TypeScript types plain (classes there need the esnext `Symbol.dispose`).

thread_local! {
    static NET: RefCell<Option<Net>> = const { RefCell::new(None) };
    static LAST: RefCell<Option<Prediction>> = const { RefCell::new(None) };
    static DEC: RefCell<Option<Decoded>> = const { RefCell::new(None) };
}

fn with_net<T>(f: impl FnOnce(&Net) -> T) -> Result<T, JsError> {
    NET.with(|n| n.borrow().as_ref().map(f)).ok_or_else(|| JsError::new("stardist: call load() first"))
}
fn last<T>(f: impl FnOnce(&Prediction) -> T, empty: T) -> T {
    LAST.with(|p| p.borrow().as_ref().map(f).unwrap_or(empty))
}
fn dec<T>(f: impl FnOnce(&Decoded) -> T, empty: T) -> T {
    DEC.with(|d| d.borrow().as_ref().map(f).unwrap_or(empty))
}

/// Builds the network: config.json and thresholds.json of the model (text), weights_best.h5 (bytes).
#[wasm_bindgen]
pub fn load(config_json: &str, thresholds_json: &str, weights_h5: &[u8]) -> Result<(), JsError> {
    let net = Net::new(config_json, thresholds_json, weights_h5).map_err(err)?;
    NET.with(|n| *n.borrow_mut() = Some(net));
    Ok(())
}
#[wasm_bindgen]
pub fn n_rays() -> Result<usize, JsError> {
    with_net(|n| n.config.n_rays)
}
#[wasm_bindgen]
pub fn grid() -> Result<usize, JsError> {
    with_net(|n| n.config.grid[0])
}
#[wasm_bindgen]
pub fn default_prob_thresh() -> Result<f32, JsError> {
    with_net(|n| n.thresholds.prob)
}
#[wasm_bindgen]
pub fn default_nms_thresh() -> Result<f32, JsError> {
    with_net(|n| n.thresholds.nms)
}

/// Runs the network on a normalised grey image (row-major, width x height); results via
/// out_width/out_height/prob/dist. engine 0: stardist-rs's candle model; 1: the SIMD kernels
/// of src/fast.rs (same outputs up to rounding, about 3x faster).
#[wasm_bindgen]
pub fn predict(img: &[f32], width: usize, height: usize, engine: u32) -> Result<(), JsError> {
    let e = if engine == 0 { Engine::Candle } else { Engine::Fast };
    let p = with_net(|n| n.predict(img, width, height, e))?.map_err(err)?;
    LAST.with(|l| *l.borrow_mut() = Some(p));
    Ok(())
}
#[wasm_bindgen]
pub fn out_width() -> usize {
    last(|p| p.out_w, 0)
}
#[wasm_bindgen]
pub fn out_height() -> usize {
    last(|p| p.out_h, 0)
}
/// Object probability per grid cell (out_height x out_width).
#[wasm_bindgen]
pub fn prob() -> Vec<f32> {
    last(|p| p.prob.clone(), Vec::new())
}
/// Ray distances in pixels (out_height x out_width x n_rays). Ray k points along
/// (dy, dx) = (sin φ, cos φ), φ = 2πk/n_rays: k = 0 to the right, then turning
/// towards +y (down), as in StarDist.
#[wasm_bindgen]
pub fn dist() -> Vec<f32> {
    last(|p| p.dist.clone(), Vec::new())
}

/// The decoding alone, from network outputs (see `decode`); results via the dec_* getters.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn instances(
    prob: &[f32],
    dist: &[f32],
    out_width: usize,
    out_height: usize,
    img_width: usize,
    img_height: usize,
    n_rays: usize,
    grid: usize,
    prob_thresh: f32,
    nms_thresh: f32,
) -> Result<(), JsError> {
    let d = decode(prob, dist, out_width, out_height, img_width, img_height, n_rays, grid, prob_thresh, nms_thresh)
        .map_err(err)?;
    DEC.with(|x| *x.borrow_mut() = Some(d));
    Ok(())
}
/// Grid cells with probability above the threshold (the NMS candidates).
#[wasm_bindgen]
pub fn dec_candidates() -> usize {
    dec(|d| d.candidates, 0)
}
/// Number of kept objects.
#[wasm_bindgen]
pub fn dec_count() -> usize {
    dec(|d| d.probs.len(), 0)
}
/// Centres (y, x) in image pixels, by decreasing probability.
#[wasm_bindgen]
pub fn dec_points() -> Vec<f32> {
    dec(|d| d.points.clone(), Vec::new())
}
#[wasm_bindgen]
pub fn dec_probs() -> Vec<f32> {
    dec(|d| d.probs.clone(), Vec::new())
}
/// Ray lengths of the kept objects (count x n_rays).
#[wasm_bindgen]
pub fn dec_dists() -> Vec<f32> {
    dec(|d| d.dists.clone(), Vec::new())
}
/// Polygon vertices (count x 2 x n_rays: all y, then all x of an object).
#[wasm_bindgen]
pub fn dec_coords() -> Vec<f32> {
    dec(|d| d.coords.clone(), Vec::new())
}
/// Label image (img_height x img_width), 0 = background, object i (1-based) = points[i-1].
#[wasm_bindgen]
pub fn dec_labels() -> Vec<u32> {
    dec(|d| d.labels.clone(), Vec::new())
}
