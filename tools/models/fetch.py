#!/usr/bin/env python3
"""Fetch model checkpoints and sample images for the network demos, convert them, and place
them in demos/dist/models/ (git-ignored; deployed by CI). `make models` runs this.

    python3 tools/models/fetch.py                 # every enabled entry of manifest.json
    python3 tools/models/fetch.py --only sam2.1-hiera-tiny,yolo11n-coco
    python3 tools/models/fetch.py --all           # also disabled entries (e.g. Cellpose-SAM, 1.2 GB)
    python3 tools/models/fetch.py --list          # show the manifest
    python3 tools/models/fetch.py --update-hashes # fill missing bytes/sha256 into manifest.json

Downloads are cached in .cache/models/downloads/ (verified against the manifest's bytes and
sha256), converter scripts from the henriksson-lab crates (pinned commits) in
.cache/models/tools/. An entry is only reconverted when its manifest entry, the tools or this
script change, or an output is missing. Finally demos/dist/models/index.json lists every output
file (key = path under models/, with bytes, sha256, source entry and credit): the browser
loader (demos/src/lib/wasm.ts, loadModelFile) uses the sha256 as its cache version.

Needs: numpy, pillow, tifffile, scipy, torch (CPU is enough), safetensors.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_VERSION = "1"  # bump to force reconversion of everything
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "manifest.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "kB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expect_bytes: int | None, expect_sha: str | None, retries: int = 3) -> None:
    """Download url to dest (atomic, verified). Skips when dest exists and matches."""
    if dest.exists():
        ok = (not expect_bytes or dest.stat().st_size == expect_bytes) and (not expect_sha or sha256_file(dest) == expect_sha)
        if ok:
            return
        log(f"  cached {dest.name} does not match the manifest; downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "course-model-fetch/1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got, t0, shown = 0, time.time(), 0.0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    got += len(block)
                    if time.time() - shown > 5:
                        shown = time.time()
                        pct = f" ({100 * got / total:.0f}%)" if total else ""
                        log(f"  {dest.name}: {human(got)}{pct}")
            if expect_bytes and tmp.stat().st_size != expect_bytes:
                raise IOError(f"size {tmp.stat().st_size} != manifest {expect_bytes}")
            if expect_sha and sha256_file(tmp) != expect_sha:
                raise IOError("sha256 does not match the manifest")
            tmp.replace(dest)
            log(f"  downloaded {dest.name} ({human(dest.stat().st_size)}, {time.time() - t0:.0f} s)")
            return
        except Exception as e:  # noqa: BLE001 (report and retry)
            last_err = e
            log(f"  download attempt {attempt} of {url} failed: {e}")
            time.sleep(3 * attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}: {last_err}")


# ---- Converter tools (scripts from the crates, pinned commits) --------------------------------

def fetch_tools(tools: dict, tool_dir: Path) -> None:
    for name, t in tools.items():
        dest_root = tool_dir / t.get("dest", name)
        for rel in t["files"]:
            dest = dest_root / rel
            if not dest.exists():
                download(t["base"] + rel, dest, None, None)
        # nets/ must be importable as a package for unpickling (YOLOv11-pt).
        for rel in t["files"]:
            pkg = (dest_root / rel).parent
            if pkg != dest_root and not (pkg / "__init__.py").exists():
                (pkg / "__init__.py").write_text("")


# ---- Conversion steps ------------------------------------------------------------------------

def write_bytes(out: Path, data: bytes) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)


def read_source(src: Path, member: str | None) -> bytes:
    if member:
        with zipfile.ZipFile(src) as z:
            return z.read(member)
    return src.read_bytes()


def save_safetensors(state: dict, out: Path, dtype: str | None, meta: dict) -> None:
    import torch
    from safetensors.torch import save_file
    tensors = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if dtype == "f16" and v.is_floating_point():
            v = v.to(torch.float16)
        elif dtype == "bf16" and v.is_floating_point():
            v = v.to(torch.bfloat16)
        tensors[k] = v.contiguous().clone()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={k: str(v) for k, v in meta.items()})
    tmp.replace(out)


def to_uint8(a):
    """Grey image of any dtype -> uint8, rescaled between the 0.5 and 99.8 percentiles
    (uint8 images that already use the range are kept)."""
    import numpy as np
    if a.dtype == np.uint8 and a.max() > 200:
        return a
    lo, hi = np.percentile(a, [0.5, 99.8])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((a.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def load_image(data: bytes, name: str):
    import numpy as np
    from PIL import Image
    if name.lower().endswith((".tif", ".tiff")):
        import tifffile
        return np.asarray(tifffile.imread(io.BytesIO(data)))
    return np.asarray(Image.open(io.BytesIO(data)))


def save_png(a, out: Path) -> None:
    from PIL import Image
    import numpy as np
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.dtype == np.uint16:
        im = Image.fromarray(a.astype(np.uint16))  # mode I;16
    else:
        im = Image.fromarray(a)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    write_bytes(out, buf.getvalue())


def grey_or_rgb(a):
    """RGBA/RGB with equal channels -> one grey channel; real colour stays RGB."""
    import numpy as np
    if a.ndim == 3:
        a = a[..., :3]
        if np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]):
            return a[..., 0]
    return a


def labels_png(lab, out: Path) -> int:
    import numpy as np
    n = int(lab.max())
    if n > 65535:
        raise ValueError("more than 65535 labels")
    save_png(lab.astype(np.uint16), out)
    return n


def step_unzip(src: Path, step: dict, outdir: Path) -> list[Path]:
    outs = []
    with zipfile.ZipFile(src) as z:
        for member, rel in step["members"].items():
            out = outdir / rel
            write_bytes(out, z.read(member))
            outs.append(out)
    return outs


def step_copy(src: Path, step: dict, outdir: Path) -> list[Path]:
    out = outdir / step["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return [out]


def step_torch_to_safetensors(src: Path, step: dict, outdir: Path, entry: dict) -> list[Path]:
    import torch
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = outdir / step["out"]
    save_safetensors(sd, out, step.get("dtype"), {"source": entry["url"], "dtype": step.get("dtype", "as stored")})
    return [out]


def step_npy_size_model(src: Path, step: dict, outdir: Path) -> list[Path]:
    import numpy as np
    d = np.load(src, allow_pickle=True).item()
    js = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}
    out = outdir / step["out"]
    write_bytes(out, (json.dumps(js) + "\n").encode())
    return [out]


def step_sam2_convert(src: Path, step: dict, outdir: Path, entry: dict, tool_dir: Path) -> list[Path]:
    import torch
    from safetensors.torch import load_file
    tool = tool_dir / "sam2-rs" / "tools" / "convert_checkpoint.py"
    tmp = tool_dir.parent / "work" / (entry["id"] + ".f32.safetensors")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", str(tool), str(src), str(tmp)], check=True)
    sd = load_file(str(tmp))
    out = outdir / step["out"]
    save_safetensors(sd, out, step.get("dtype"), {"source": entry["url"], "format": "sam2-pytorch-state-dict",
                                                   "converter": "sam2-rs tools/convert_checkpoint.py",
                                                   "dtype": step.get("dtype", "f32")})
    tmp.unlink(missing_ok=True)
    return [out]


def step_yolov11_convert(src: Path, step: dict, outdir: Path, entry: dict, tool_dir: Path) -> list[Path]:
    import torch
    tool = tool_dir / "yolov11-rs" / "convert_weights.py"
    tmp = tool_dir.parent / "work" / (entry["id"] + ".burn.pt")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # The checkpoint pickles YOLOv11-pt model classes (nets.nn); the converter puts
    # <tool dir>/YOLOv11-pt on sys.path itself. Only run on the pinned, hash-checked download.
    subprocess.run([sys.executable, "-I", str(tool), str(src), "-o", str(tmp)], check=True)
    sd = torch.load(tmp, map_location="cpu", weights_only=True)
    out = outdir / step["out"]
    save_safetensors(sd, out, step.get("dtype"), {"source": entry["url"], "converter": "YOLOv11-rs convert_weights.py",
                                                   "dtype": step.get("dtype", "as stored")})
    tmp.unlink(missing_ok=True)
    return [out]


def step_image(src: Path, step: dict, outdir: Path, entry: dict) -> list[Path]:
    name = step.get("member") or entry["file"]
    a = grey_or_rgb(load_image(read_source(src, step.get("member")), name))
    if a.ndim == 2:
        a = to_uint8(a)
    out = outdir / step["out"]
    save_png(a, out)
    return [out]


def step_bbbc039_labels(src: Path, step: dict, outdir: Path) -> list[Path]:
    import numpy as np
    from scipy import ndimage as ndi
    m = load_image(read_source(src, step["member"]), step["member"])
    cls = m[..., 0] if m.ndim == 3 else m
    lab, n = ndi.label(cls == 1)  # 4-connected interiors
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=np.int64)
    keep[1:][sizes >= 10] = np.arange(1, int((sizes >= 10).sum()) + 1)
    lab = keep[lab]
    # Grow labels into the boundary/touching pixels (each takes its nearest interior label).
    fg = cls > 0
    if (fg & (lab == 0)).any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        grown = lab[iy, ix]
        lab = np.where(fg, grown, 0)
    out = outdir / step["out"]
    labels_png(lab, out)
    return [out]


def step_dsb_sample(src: Path, step: dict, outdir: Path) -> list[Path]:
    import numpy as np
    from PIL import Image
    sid = step["sample"]
    with zipfile.ZipFile(src) as z:
        img = grey_or_rgb(np.asarray(Image.open(io.BytesIO(z.read(f"{sid}/images/{sid}.png")))))
        masks = sorted(n for n in z.namelist() if n.startswith(f"{sid}/masks/") and n.endswith(".png"))
        lab = np.zeros(img.shape[:2], dtype=np.int64)
        for i, mname in enumerate(masks, start=1):
            mk = np.asarray(Image.open(io.BytesIO(z.read(mname)))) > 0
            lab[mk & (lab == 0)] = i
    out_img = outdir / (step["out"] + ".png")
    out_lab = outdir / (step["out"] + "-labels.png")
    save_png(img, out_img)
    labels_png(lab, out_lab)
    return [out_img, out_lab]


def run_steps(entry: dict, src: Path, outdir: Path, tool_dir: Path) -> list[Path]:
    outs: list[Path] = []
    for step in entry["steps"]:
        op = step["op"]
        if op == "unzip":
            outs += step_unzip(src, step, outdir)
        elif op == "copy":
            outs += step_copy(src, step, outdir)
        elif op == "torch_to_safetensors":
            outs += step_torch_to_safetensors(src, step, outdir, entry)
        elif op == "npy_size_model":
            outs += step_npy_size_model(src, step, outdir)
        elif op == "sam2_convert":
            outs += step_sam2_convert(src, step, outdir, entry, tool_dir)
        elif op == "yolov11_convert":
            outs += step_yolov11_convert(src, step, outdir, entry, tool_dir)
        elif op == "image":
            outs += step_image(src, step, outdir, entry)
        elif op == "bbbc039_labels":
            outs += step_bbbc039_labels(src, step, outdir)
        elif op == "dsb_sample":
            outs += step_dsb_sample(src, step, outdir)
        else:
            raise ValueError(f"{entry['id']}: unknown step op {op!r}")
    return outs


# ---- Main ------------------------------------------------------------------------------------

def entry_hash(entry: dict, tools: dict) -> str:
    return hashlib.sha256(json.dumps([SCRIPT_VERSION, entry, tools], sort_keys=True).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated entry ids (enabled or not)")
    ap.add_argument("--all", action="store_true", help="include disabled entries")
    ap.add_argument("--list", action="store_true", help="list the manifest and exit")
    ap.add_argument("--update-hashes", action="store_true", help="download and fill missing bytes/sha256 into manifest.json")
    ap.add_argument("--cache", default=str(ROOT / ".cache" / "models"))
    ap.add_argument("--out", default=str(ROOT / "demos" / "dist" / "models"))
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["entries"]
    if args.list:
        for e in entries:
            log(f"{e['id']:28s} {e['kind']:7s} {'on ' if e.get('enabled', True) else 'off'} {human(e.get('bytes') or 0):>9s}  {e['url']}")
        return 0

    if args.only:
        want = set(args.only.split(","))
        unknown = want - {e["id"] for e in entries}
        if unknown:
            log(f"unknown ids: {', '.join(sorted(unknown))}")
            return 2
        selected = [e for e in entries if e["id"] in want]
    else:
        selected = [e for e in entries if args.all or e.get("enabled", True)]

    cache = Path(args.cache)
    outdir = Path(args.out)
    dl_dir, tool_dir, stamp_dir = cache / "downloads", cache / "tools", cache / "stamps"
    for d in (dl_dir, tool_dir, stamp_dir, outdir):
        d.mkdir(parents=True, exist_ok=True)

    if args.update_hashes:
        changed = False
        for e in selected:
            src = dl_dir / e["file"]
            download(e["url"], src, e.get("bytes") or None, e.get("sha256") or None)
            if not e.get("sha256") or not e.get("bytes"):
                e["bytes"], e["sha256"] = src.stat().st_size, sha256_file(src)
                changed = True
        if changed:
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
            log("manifest.json updated")
        return 0

    fetch_tools(manifest.get("tools", {}), tool_dir)
    failed = []
    for e in selected:
        log(f"[{e['id']}]")
        try:
            src = dl_dir / e["file"]
            download(e["url"], src, e.get("bytes") or None, e.get("sha256") or None)
            h = entry_hash(e, manifest.get("tools", {}))
            stamp = stamp_dir / f"{e['id']}.json"
            if stamp.exists():
                st = json.loads(stamp.read_text())
                if st.get("hash") == h and all((outdir / o).exists() for o in st.get("outputs", [])):
                    log("  up to date")
                    continue
            t0 = time.time()
            outs = run_steps(e, src, outdir, tool_dir)
            rel = [str(o.relative_to(outdir)) for o in outs]
            stamp.write_text(json.dumps({"hash": h, "outputs": rel}, indent=1))
            for o in outs:
                log(f"  -> {o.relative_to(outdir)} ({human(o.stat().st_size)})")
            log(f"  converted in {time.time() - t0:.1f} s")
        except Exception as ex:  # noqa: BLE001 (continue with the other entries, fail at the end)
            log(f"  FAILED: {ex}")
            failed.append(e["id"])

    # index.json over the outputs of every entry that has a stamp and whose outputs exist
    # (entries fetched earlier with --only/--all stay listed while their files are there).
    index: dict[str, dict] = {}
    by_id = {e["id"]: e for e in entries}
    for stamp in sorted(stamp_dir.glob("*.json")):
        eid = stamp.stem
        if eid not in by_id:
            continue
        e = by_id[eid]
        for rel in json.loads(stamp.read_text()).get("outputs", []):
            p = outdir / rel
            if p.exists():
                index[rel] = {"file": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                              "entry": eid, "kind": e["kind"], "credit": e.get("credit", ""), "source": e["url"]}
    (outdir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    total = sum(v["bytes"] for v in index.values())
    log(f"\n{len(index)} files, {human(total)} in {outdir}")
    for rel, v in sorted(index.items()):
        log(f"  {human(v['bytes']):>9s}  {rel}")
    if failed:
        log(f"\nFAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
