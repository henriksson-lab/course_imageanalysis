# Model weights and sample images (`make models`)

`fetch.py` reads `manifest.json`, downloads each enabled entry into `.cache/models/downloads/`
(checked against `bytes` and `sha256`), converts it into `demos/dist/models/` and writes
`demos/dist/models/index.json`. Both directories are git-ignored; the deploy workflow
(`.github/workflows/deploy.yml`) runs the same script (downloads cached between runs, keyed on
`manifest.json` + `fetch.py`) and publishes the result under `demos/dist/models/` of the site.
Demos load files by their index key with `loadModelFile` from `demos/src/lib/wasm.ts`.

```sh
make models                                          # enabled entries
python3 tools/models/fetch.py --list
python3 tools/models/fetch.py --only cellpose-sam    # a disabled entry (1.2 GB)
python3 tools/models/fetch.py --update-hashes        # fill bytes/sha256 of a new entry
```

Python packages: `torch` + `torchvision` (CPU wheels are enough), and `requirements.txt`.

## Steps (`"op"` in the manifest)

| op | does |
|---|---|
| `unzip` | copy listed zip members to output paths |
| `copy` | copy the download as is |
| `torch_to_safetensors` | `torch.load(weights_only=True)` state dict (or its `model` member) → safetensors, floats cast to `dtype` (f16) |
| `npy_size_model` | Cellpose size model (`.npy` pickle of a dict) → JSON |
| `sam2_convert` | sam2-rs `tools/convert_checkpoint.py` (official `.pt` → flat safetensors, no renaming), then f16 |
| `yolov11_convert` | YOLOv11-rs `convert_weights.py` (YOLOv11-pt keys → YOLOv11-rs/Burn names; unpickles the YOLOv11-pt classes `nets/nn.py` + `utils/util.py`), then safetensors f16 |
| `image` | TIFF/PNG → 8-bit PNG (grey images rescaled between the 0.5 and 99.8 percentiles; equal-channel RGB → grey) |
| `bbbc039_labels` | BBBC039 class mask → instance labels (16-bit PNG) |
| `dsb_sample` | one 2018 Data Science Bowl sample: image PNG + all its instance masks merged into a 16-bit label PNG |

Converter scripts are fetched from the crates at pinned commits (`"tools"` in the manifest)
into `.cache/models/tools/`. Bump `SCRIPT_VERSION` in `fetch.py` to force reconversion.

## What the Rust crates load (henriksson-lab, read 2026-11-13)

| crate | checkpoint format | our output |
|---|---|---|
| `stardist-rs` | Keras HDF5 `weights_best.h5` + `config.json` + `thresholds.json` (`weights::load_keras_hdf5_weights`, pure-Rust HDF5; Burn or Candle backend) | `stardist/2D_versatile_fluo/*` unchanged (f32, 5.5 MB) |
| `cellpose-rs` | **Cellpose-SAM / CPDINO only** (`cpsam`, `cpsam_v2`, `cpdino*`; `.pth` or `.safetensors`, Candle; also a Burn backend). Downloads `cpsam` from huggingface.co/mouseland/cellpose-sam. No Cellpose 3 (cyto3/nuclei U-net) support | `cellpose/cyto3.safetensors`, `cellpose/nuclei.safetensors` (f16, for a small U-net port); `cpsam` disabled (1.2 GB) |
| `YOLOv11-rs` | safetensors or `.pt` state dict with YOLOv11-pt (jahongir7174) layout renamed by `convert_weights.py`; Candle `VarMap::load` | `yolo/yolo11n-coco.safetensors` (f16; candle converts to f32 on load) |
| `sam2-rs` | official SAM 2/2.1 `.pt` directly, or the safetensors from `tools/convert_checkpoint.py`; `Sam2Variant::HieraTiny` | `sam2/sam2.1_hiera_tiny.safetensors` (f16) |

Quantisation: f16 only for now (halves size, loaders upcast). int8 would need dequantisation in
the Wasm wrapper.

## Sizes (`make models`, 2026-11-13)

Downloads ~387 MB (cached). Outputs 27 files, 112.5 MB: SAM 2.1 tiny 74.4 MB, Cellpose cyto3 and
nuclei 12.6 MB each, StarDist 5.5 MB, YOLO11n 5.1 MB, sample images and labels ~2.3 MB.
