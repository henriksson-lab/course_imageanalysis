//! SAM 2.1 (Hiera tiny) for the browser: a thin wasm-bindgen layer over the henriksson-lab
//! crate `sam2-rs` (candle, CPU). Numeric in, numeric out; no DOM.
//!
//! The split mirrors the model:
//! - `load(weights)`: build the network from the bytes of a safetensors checkpoint
//!   (`demos/dist/models/sam2/sam2.1_hiera_tiny.safetensors`, f16 is read into f32).
//! - `encode(rgb, w, h)`: the heavy image encoder (Hiera + feature pyramid), run **once per
//!   image**; the image embedding (256 x 64 x 64) and the two high-resolution skip features
//!   stay cached inside.
//! - `decode(...)`: prompt encoder + mask decoder for one prompt set (points, box, optional
//!   previous mask), run **per click**: three candidate masks (or one) as 256 x 256 logits in
//!   the model's square frame, with predicted IoU scores and the object score.
//! - `decode_points(...)`: the same for a batch of single positive points (the point grid of
//!   automatic mask generation); filtering, stability score and NMS are done by the caller
//!   (demos/src/samnet/post.ts), so the post-processing can be shown and tested step by step.
//!
//! Coordinates are in pixels of the original image; the image is resized (not padded) to
//! 1024 x 1024, so a pixel (x, y) maps to (x * 1024 / w, y * 1024 / h).
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{
    BackboneOutput, ImageEncoder, ImageGeometry, MaskDecoder, MaskDecoderConfig, PromptEncoder,
    Sam2Config, Sam2ImageTransforms, Sam2Variant,
};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

/// Side of the low-resolution mask logits.
pub const LOW_RES: usize = 256;
/// Side of the model's square input.
pub const INPUT: usize = 1024;

fn js<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

/// Output of one decoder run: `count` prompt sets x `masks` candidates.
pub struct Decoded {
    pub count: usize,
    pub masks: usize,
    /// count x masks x 256 x 256 mask logits (row-major), in the model's square frame.
    pub logits: Vec<f32>,
    /// count x masks predicted IoU scores.
    pub iou: Vec<f32>,
    /// count object-score logits (> 0: an object is present).
    pub object_score: Vec<f32>,
    /// Sparse prompt tokens the prompt encoder produced (points, box corners, padding).
    pub tokens: usize,
}

/// The network with the cached embedding of the current image.
pub struct Sam {
    device: Device,
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    no_mem_embed: Tensor,
    transforms: Sam2ImageTransforms,
    features: Option<BackboneOutput>,
    geometry: Option<ImageGeometry>,
}

// ---- The JavaScript API: free functions over one network per instance -------------------------
// (Free functions, no exported classes: the generated typings then need no `Symbol.dispose`.)

struct State {
    sam: Option<Sam>,
    last: Option<Decoded>,
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State { sam: None, last: None }) };
}

fn with_sam<T>(f: impl FnOnce(&mut Sam) -> candle_core::Result<T>) -> Result<T, JsError> {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let sam = s.sam.as_mut().ok_or_else(|| JsError::new("load the network first"))?;
        f(sam).map_err(js)
    })
}

fn keep_last(d: Decoded) -> Vec<f32> {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let logits = d.logits.clone();
        s.last = Some(d);
        logits
    })
}

fn with_last<T>(f: impl FnOnce(&Decoded) -> T) -> Result<T, JsError> {
    STATE.with(|s| s.borrow().last.as_ref().map(f).ok_or_else(|| JsError::new("nothing decoded yet")))
}

/// Build SAM 2.1 Hiera tiny from the bytes of its safetensors checkpoint (replaces any earlier one).
#[wasm_bindgen]
pub fn load(weights: Vec<u8>) -> Result<(), JsError> {
    let sam = Sam::from_safetensors(weights).map_err(js)?;
    STATE.with(|s| *s.borrow_mut() = State { sam: Some(sam), last: None });
    Ok(())
}

/// Run the image encoder on interleaved RGB8 pixels (w x h x 3) and cache the result.
#[wasm_bindgen]
pub fn encode(rgb: &[u8], width: usize, height: usize) -> Result<(), JsError> {
    with_sam(|sam| sam.encode_rgb8(rgb, width, height))
}

/// The image embedding the decoder attends to: 256 channels x 64 x 64, channel-major.
#[wasm_bindgen]
pub fn embedding() -> Result<Vec<f32>, JsError> {
    with_sam(|sam| sam.embedding())
}

/// Decode one prompt set. `points`: x0,y0,x1,y1,... (original pixels); `labels`: 1 positive,
/// 0 negative per point; `bbox`: empty or x0,y0,x1,y1; `mask_input`: empty or a 256 x 256
/// logit plane (e.g. the chosen mask of the previous click). `multimask`: three candidates
/// (the ambiguity-aware output) or one. Returns the mask logits (masks x 256 x 256); scores
/// and shape via `last_iou`, `last_object_score`, `last_shape`.
#[wasm_bindgen]
pub fn decode(points: &[f32], labels: &[i32], bbox: &[f32], mask_input: &[f32], multimask: bool) -> Result<Vec<f32>, JsError> {
    let d = with_sam(|sam| sam.decode_prompts(points, labels, bbox, mask_input, multimask))?;
    Ok(keep_last(d))
}

/// Decode a batch of single positive points (x0,y0,x1,y1,...; original pixels), three
/// candidates each: one step of automatic mask generation. Returns count x 3 x 256 x 256 logits.
#[wasm_bindgen]
pub fn decode_points(points: &[f32]) -> Result<Vec<f32>, JsError> {
    let d = with_sam(|sam| sam.decode_point_batch(points))?;
    Ok(keep_last(d))
}

/// Predicted IoU scores of the last decode (count x masks).
#[wasm_bindgen]
pub fn last_iou() -> Result<Vec<f32>, JsError> {
    with_last(|d| d.iou.clone())
}

/// Object-score logits of the last decode (one per prompt set; > 0: an object is present).
#[wasm_bindgen]
pub fn last_object_score() -> Result<Vec<f32>, JsError> {
    with_last(|d| d.object_score.clone())
}

/// [count, masks, prompt tokens] of the last decode.
#[wasm_bindgen]
pub fn last_shape() -> Result<Vec<u32>, JsError> {
    with_last(|d| vec![d.count as u32, d.masks as u32, d.tokens as u32])
}

impl Sam {
    /// Build from safetensors bytes (also used natively by the tests).
    pub fn from_safetensors(weights: Vec<u8>) -> candle_core::Result<Sam> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let config = Sam2Config::sam2_1(Sam2Variant::HieraTiny);
        Ok(Sam {
            image_encoder: ImageEncoder::load(&config, vb.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, vb.pp("sam_prompt_encoder"))?,
            // Same as the image predictor of sam2-rs (Python's apply_postprocessing=True).
            mask_decoder: MaskDecoder::load_with_config(
                vb.pp("sam_mask_decoder"),
                MaskDecoderConfig::default(),
            )?,
            no_mem_embed: vb.get((1, 1, 256), "no_mem_embed")?,
            transforms: Sam2ImageTransforms::new(config.image_size),
            features: None,
            geometry: None,
            device,
        })
    }

    /// As `Sam2ImagePredictor::set_image`: resize to 1024², normalise, Hiera + FPN, add the
    /// "no memory" embedding, pre-project the two skip features for the decoder.
    pub fn encode_rgb8(&mut self, rgb: &[u8], width: usize, height: usize) -> candle_core::Result<()> {
        self.features = None;
        let prepared = self.transforms.prepare_rgb8(rgb, width, height, &self.device)?;
        let mut features = self.image_encoder.forward(&prepared.tensor)?;
        features.vision_features = features
            .vision_features
            .broadcast_add(&self.no_mem_embed.reshape((1, 256usize, 1usize, 1usize))?)?;
        features.sam_high_res_features =
            Some(self.mask_decoder.project_high_res_features(&features.backbone_fpn[..2])?);
        self.features = Some(features);
        self.geometry = Some(prepared.geometry);
        Ok(())
    }

    fn cached(&self) -> candle_core::Result<(&BackboneOutput, ImageGeometry)> {
        match (&self.features, self.geometry) {
            (Some(f), Some(g)) => Ok((f, g)),
            _ => candle_core::bail!("encode an image first"),
        }
    }

    /// Prompt encoder + mask decoder, as `Sam2ImagePredictor::predict_cached_prompts` (a box
    /// becomes two corner points with labels 2 and 3, placed before the clicks).
    pub fn decode_prompts(
        &self,
        points: &[f32],
        labels: &[i32],
        bbox: &[f32],
        mask_input: &[f32],
        multimask: bool,
    ) -> candle_core::Result<Decoded> {
        let (features, g) = self.cached()?;
        if points.len() != 2 * labels.len() {
            candle_core::bail!("points must hold x,y for every label")
        }
        if !(bbox.is_empty() || bbox.len() == 4) {
            candle_core::bail!("bbox must be empty or x0,y0,x1,y1")
        }
        let (sx, sy) = (g.scale_x(), g.scale_y());
        let mut coords = Vec::new();
        let mut labs: Vec<i64> = Vec::new();
        if bbox.len() == 4 {
            coords.extend([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]);
            labs.extend([2, 3]);
        }
        for (i, &l) in labels.iter().enumerate() {
            coords.extend([points[2 * i] * sx, points[2 * i + 1] * sy]);
            labs.push(if l > 0 { 1 } else { 0 });
        }
        let n = labs.len();
        let pts = if n > 0 {
            Some((
                Tensor::from_vec(coords, (1, n, 2), &self.device)?,
                Tensor::from_vec(labs, (1, n), &self.device)?,
            ))
        } else {
            None
        };
        let mask = if mask_input.is_empty() {
            None
        } else if mask_input.len() == LOW_RES * LOW_RES {
            Some(Tensor::from_vec(mask_input.to_vec(), (1, 1, LOW_RES, LOW_RES), &self.device)?)
        } else {
            candle_core::bail!("mask_input must be empty or 256 x 256")
        };
        self.run_decoder(features, pts.as_ref().map(|(c, l)| (c, l)), mask.as_ref(), multimask)
    }

    /// A batch of single positive points, three candidates each (automatic mask generation).
    pub fn decode_point_batch(&self, points: &[f32]) -> candle_core::Result<Decoded> {
        let (features, g) = self.cached()?;
        let b = points.len() / 2;
        if b == 0 || points.len() % 2 != 0 {
            candle_core::bail!("points must be a non-empty list of x,y")
        }
        let coords: Vec<f32> = points
            .chunks_exact(2)
            .flat_map(|p| [p[0] * g.scale_x(), p[1] * g.scale_y()])
            .collect();
        let c = Tensor::from_vec(coords, (b, 1, 2), &self.device)?;
        let l = Tensor::ones((b, 1), DType::I64, &self.device)?;
        self.run_decoder(features, Some((&c, &l)), None, true)
    }

    fn run_decoder(
        &self,
        features: &BackboneOutput,
        points: Option<(&Tensor, &Tensor)>,
        mask: Option<&Tensor>,
        multimask: bool,
    ) -> candle_core::Result<Decoded> {
        let prompts = self.prompt_encoder.forward(points, None, mask)?;
        let tokens = prompts.sparse_embeddings.dim(1)?;
        let dense_pe = self.prompt_encoder.get_dense_pe()?;
        let high_res = features
            .sam_high_res_features
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing skip features".into()))?;
        let out = self.mask_decoder.forward_preprojected(
            &features.vision_features,
            &dense_pe,
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            multimask,
            high_res,
        )?;
        let (count, masks, h, w) = out.masks.dims4()?;
        if h != LOW_RES || w != LOW_RES {
            candle_core::bail!("unexpected mask size {h}x{w}")
        }
        Ok(Decoded {
            count,
            masks,
            logits: out.masks.flatten_all()?.to_vec1::<f32>()?,
            iou: out.iou_predictions.flatten_all()?.to_vec1::<f32>()?,
            object_score: out.object_score_logits.flatten_all()?.to_vec1::<f32>()?,
            tokens,
        })
    }

    /// The cached image embedding (256 x 64 x 64, channel-major).
    pub fn embedding(&self) -> candle_core::Result<Vec<f32>> {
        match &self.features {
            Some(f) => f.vision_features.flatten_all()?.to_vec1::<f32>(),
            None => candle_core::bail!("encode an image first"),
        }
    }
}
