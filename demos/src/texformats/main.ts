// GPU texture formats (WebGL2). The same grey image, given as numbers in [0, 1], is
// uploaded to the GPU in two formats chosen by the student (RGBA8, SRGB8_ALPHA8, RGB565,
// RGB10_A2, R16F, R32F, R16UI) and drawn side by side by a fragment shader that samples
// the texture and stretches a window [lo, hi] to the screen, so that the steps of each
// format become visible. Nearest or linear filtering in hardware, magnification 1-16x.
// Click a picture for the stored bits of one texel (computed by pack.ts, which also
// builds the uploaded arrays). If WebGL2, a format or an extension is missing, the demo
// says so instead of drawing.
// Auto-mounts on <div data-demo="texformats">. Optional: data-a="RGBA8", data-b="R16F",
// data-image="dark|vdark|full|blobs|tiny", data-filter="nearest|linear", data-zoom="1".
import { htmlEl, slider, presets, button, autoMount } from "../lib/ui";
import { FORMATS, FormatId, FormatInfo, uploadData, storeTexel, textureBytes } from "./pack";
import { blobs } from "../quantise/quant";

const NAME = "texformats";

type ImgId = "dark" | "vdark" | "full" | "blobs" | "tiny";
interface Img { id: ImgId; label: string; w: number; h: number; data: Float64Array; lo: number; hi: number; }

function makeImages(): Record<ImgId, Img> {
  const N = 256;
  const ramp = (a: number, b: number) => {
    const d = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) d[i * N + j] = a + (b - a) * (j * N + i) / (N * N - 1);
    return d;
  };
  const bl = blobs(N).map(v => 0.2 * v);
  const tiny = Float64Array.from([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0.5, 0.5, 1, 0, 0.5, 0.5]);
  return {
    dark: { id: "dark", label: "dark ramp 0.20–0.24", w: N, h: N, data: ramp(0.2, 0.24), lo: 0.2, hi: 0.24 },
    vdark: { id: "vdark", label: "very dark 0–0.005", w: N, h: N, data: ramp(0, 0.005), lo: 0, hi: 0.005 },
    full: { id: "full", label: "ramp 0–1", w: N, h: N, data: ramp(0, 1), lo: 0, hi: 1 },
    blobs: { id: "blobs", label: "blobs 0–0.19", w: N, h: N, data: bl, lo: 0, hi: 0.19 },
    tiny: { id: "tiny", label: "4 × 4 texels", w: 4, h: 4, data: tiny, lo: 0, hi: 1 },
  };
}

const STYLE = `
.qb-demo[data-demo="${NAME}"] { gap: 1em; align-items: flex-start; }
.qb-demo[data-demo="${NAME}"] .tf-left { flex: 1 1 640px; min-width: 0; max-width: 800px; display: flex; flex-direction: column; gap: .3em; }
.qb-demo[data-demo="${NAME}"] .tf-heads { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.qb-demo[data-demo="${NAME}"] .tf-head { display: flex; flex-direction: column; gap: .2em; min-width: 0; }
.qb-demo[data-demo="${NAME}"] .tf-head .qb-btn { font-size: 14px; padding: .15em .45em; }
.qb-demo[data-demo="${NAME}"] .tf-cap { font-size: 16px; min-height: 2.5em; line-height: 1.25; }
.qb-demo[data-demo="${NAME}"] .tf-cap b { font-weight: 600; }
.qb-demo[data-demo="${NAME}"] .tf-warn { color: var(--qb-c2); }
.qb-demo[data-demo="${NAME}"] canvas { width: 100%; aspect-ratio: 2.04; display: block; cursor: crosshair; touch-action: none; background: #333; border-radius: 3px; }
.qb-demo[data-demo="${NAME}"] .tf-nogl { padding: 1em; background: var(--qb-panel); border: 2px solid var(--qb-c2); border-radius: 6px; font-size: 18px; }
.qb-demo[data-demo="${NAME}"] .qb-panel { flex: 1 1 260px; max-width: 330px; gap: .4em; }
.qb-demo[data-demo="${NAME}"] .tf-lab { font-size: 16px; color: var(--qb-muted); margin-bottom: -.2em; }
.qb-demo[data-demo="${NAME}"] .qb-slider { grid-template-columns: 4.5em 1fr 4.4em; }
.qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; }
.reveal .qb-demo[data-demo="${NAME}"] { font-size: 19px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-btn { font-size: 16px; }
.reveal .qb-demo[data-demo="${NAME}"] .tf-head .qb-btn { font-size: 15px; }
.reveal .qb-demo[data-demo="${NAME}"] .tf-cap { font-size: 17px; }
.reveal .qb-demo[data-demo="${NAME}"] .qb-readout { font-size: 14px; line-height: 1.25; }
`;

function injectStyle(): void {
  if (document.getElementById("qb-style-" + NAME)) return;
  const s = document.createElement("style");
  s.id = "qb-style-" + NAME;
  s.textContent = STYLE;
  document.head.appendChild(s);
}

const VS = `#version 300 es
in vec2 pos;
uniform float zoom;
out vec2 uv;
void main() {
  uv = vec2(0.5 + 0.5 * pos.x / zoom, 0.5 - 0.5 * pos.y / zoom);   // texture row 0 at the top
  gl_Position = vec4(pos, 0.0, 1.0);
}`;
const FS_FLOAT = `#version 300 es
precision highp float;
uniform highp sampler2D tex;
uniform float lo, hi;
uniform int grey;
in vec2 uv;
out vec4 o;
void main() {
  vec4 t = texture(tex, uv);
  vec3 c = grey == 1 ? vec3(t.r) : t.rgb;
  o = vec4(clamp((c - lo) / (hi - lo), 0.0, 1.0), 1.0);
}`;
const FS_UINT = `#version 300 es
precision highp float;
uniform highp usampler2D tex;
uniform float lo, hi;
in vec2 uv;
out vec4 o;
void main() {
  float v = float(texture(tex, uv).r) / 65535.0;
  o = vec4(vec3(clamp((v - lo) / (hi - lo), 0.0, 1.0)), 1.0);
}`;

interface GLThings {
  gl: WebGL2RenderingContext;
  progF: WebGLProgram; progU: WebGLProgram;
  floatLinear: boolean;
  tex: Map<string, WebGLTexture | null>;   // null = upload failed
}

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

function initGL(cv: HTMLCanvasElement): GLThings | string {
  const gl = cv.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
  if (!gl) return "This demo needs WebGL2, which this browser (or its current settings, e.g. hardware acceleration switched off) does not provide. The numbers in the readout on the right still work: click the buttons.";
  try {
    const progF = compile(gl, VS, FS_FLOAT), progU = compile(gl, VS, FS_UINT);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    return { gl, progF, progU, floatLinear: !!gl.getExtension("OES_texture_float_linear"), tex: new Map() };
  } catch (e) {
    return "WebGL2 is available, but the shaders failed to compile: " + String(e);
  }
}

/** Upload `img` in format `id`. Returns null if the GPU refused it. */
function makeTexture(g: GLThings, id: FormatId, img: Img): WebGLTexture | null {
  const gl = g.gl;
  while (gl.getError() !== gl.NO_ERROR) { /* clear old errors */ }
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  const d = uploadData(id, img.data);
  const spec: Record<FormatId, [number, number, number]> = {
    RGBA8: [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE],
    SRGB8_ALPHA8: [gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE],
    RGB565: [gl.RGB565, gl.RGB, gl.UNSIGNED_SHORT_5_6_5],
    RGB10_A2: [gl.RGB10_A2, gl.RGBA, gl.UNSIGNED_INT_2_10_10_10_REV],
    R16F: [gl.R16F, gl.RED, gl.HALF_FLOAT],
    R32F: [gl.R32F, gl.RED, gl.FLOAT],
    R16UI: [gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT],
  };
  const [ifmt, fmt, type] = spec[id];
  gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, img.w, img.h, 0, fmt, type, d);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (gl.getError() !== gl.NO_ERROR) { gl.deleteTexture(t); return null; }
  return t;
}

function mount(host: HTMLElement): void {
  injectStyle();
  const images = makeImages();
  const valid = (s: string | undefined, d: FormatId) => (FORMATS.some(f => f.id === s) ? s : d) as FormatId;
  const d0 = {
    a: valid(host.dataset.a, "RGBA8"), b: valid(host.dataset.b, "R16F"),
    image: ((host.dataset.image ?? "dark") in images ? host.dataset.image ?? "dark" : "dark") as ImgId,
    linear: host.dataset.filter === "linear",
    zoom: Math.max(1, Math.min(16, Number(host.dataset.zoom ?? 1))),
  };
  const st = { ...d0, lo: images[d0.image].lo, hi: images[d0.image].hi, sel: [128, 128] as [number, number], selSide: 0 };

  const left = htmlEl("div", "tf-left", "", host);
  const heads = htmlEl("div", "tf-heads", "", left);
  const side = (k: 0 | 1) => {
    const h = htmlEl("div", "tf-head", "", heads);
    const btns = presets(h, FORMATS.map(f => ({ label: f.label, apply: () => { if (k === 0) st.a = f.id; else st.b = f.id; schedule(); } })));
    const cap = htmlEl("div", "tf-cap", "", h);
    return { btns, cap };
  };
  const sides = [side(0), side(1)];
  const cv = htmlEl("canvas", "", "", left);
  const CW = 1020, CH = 500, VW = 500;   // two 500 x 500 viewports, 20 px apart
  cv.width = CW; cv.height = CH;
  const note = htmlEl("div", "qb-note", "", left);

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "tf-lab", "image (values in [0, 1])", panel);
  const imgIds = Object.keys(images) as ImgId[];
  const imgBtns = presets(panel, imgIds.map(id => ({ label: images[id].label, apply: () => setImage(id) })));
  htmlEl("div", "tf-lab", "stretch to the screen", panel);
  const loS = slider(panel, "lo", 0, 1, 0.0005, st.lo, v => { st.lo = Math.min(v, st.hi - 0.0005); schedule(); }, v => v.toFixed(4));
  const hiS = slider(panel, "hi", 0, 1, 0.0005, st.hi, v => { st.hi = Math.max(v, st.lo + 0.0005); schedule(); }, v => v.toFixed(4));
  htmlEl("div", "tf-lab", "filtering in hardware", panel);
  const filtBtns = presets(panel, [
    { label: "nearest", apply: () => { st.linear = false; schedule(); } },
    { label: "linear", apply: () => { st.linear = true; schedule(); } },
  ]);
  const zoomS = slider(panel, "zoom", 0, 4, 1, Math.log2(st.zoom), v => { st.zoom = 2 ** v; schedule(); }, v => `${2 ** v}×`);
  const row = htmlEl("div", "qb-presets", "", panel);
  const go = (a: FormatId, b: FormatId, image: ImgId, linear: boolean, zoom: number) => {
    st.a = a; st.b = b; st.linear = linear; st.zoom = zoom; setImage(image);
  };
  presets(row, [
    { label: "8 vs 10 bit", apply: () => go("RGBA8", "RGB10_A2", "dark", false, 1) },
    { label: "565 tint", apply: () => go("RGB565", "RGBA8", "dark", false, 1) },
    { label: "darks: sRGB", apply: () => go("RGBA8", "SRGB8_ALPHA8", "vdark", false, 1) },
    { label: "half vs float", apply: () => go("R16F", "R32F", "vdark", false, 1) },
    { label: "filter an int", apply: () => go("R16UI", "R16F", "tiny", true, 1) },
  ]);
  button(row, "reset", () => go(d0.a, d0.b, d0.image, d0.linear, d0.zoom));
  const ro = htmlEl("div", "qb-readout", "", panel);

  function setImage(id: ImgId): void {
    st.image = id; st.lo = images[id].lo; st.hi = images[id].hi;
    st.sel = [Math.floor(images[id].h / 2), Math.floor(images[id].w / 2)];
    schedule();
  }

  const glr = initGL(cv);
  if (typeof glr === "string") {
    cv.style.display = "none";
    const m = htmlEl("div", "tf-nogl", glr, left);
    left.insertBefore(m, note);
  }

  cv.addEventListener("pointerdown", ev => {
    const r = cv.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * CW, py = (ev.clientY - r.top) / r.height * CH;
    const k = px < CW / 2 ? 0 : 1;
    const vx = (px - k * (VW + 20)) / VW, vy = py / CH;
    if (vx < 0 || vx > 1) return;
    const img = images[st.image];
    const u = 0.5 + (vx - 0.5) / st.zoom, v = 0.5 + (vy - 0.5) / st.zoom;
    st.sel = [Math.min(img.h - 1, Math.floor(v * img.h)), Math.min(img.w - 1, Math.floor(u * img.w))];
    st.selSide = k;
    schedule();
  });

  let pending = false;
  function schedule(): void {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }

  /** Why a format would sample as black with the current filter, or "" if it is fine. */
  function problem(f: FormatInfo): string {
    if (!st.linear) return "";
    if (f.filterable === "no") return "integer textures cannot be filtered: the texture is incomplete and samples as black";
    if (f.filterable === "extension" && typeof glr !== "string" && !glr.floatLinear)
      return "this GPU cannot filter 32-bit float textures: the texture samples as black";
    return "";
  }

  function draw(): void {
    const img = images[st.image];
    const fmts = [FORMATS.find(f => f.id === st.a)!, FORMATS.find(f => f.id === st.b)!];
    loS.set(st.lo); hiS.set(st.hi); zoomS.set(Math.log2(st.zoom));
    imgBtns.forEach((b, i) => b.classList.toggle("active", imgIds[i] === st.image));
    filtBtns.forEach((b, i) => b.classList.toggle("active", i === (st.linear ? 1 : 0)));
    sides.forEach((s, k) => {
      s.btns.forEach((b, i) => b.classList.toggle("active", FORMATS[i].id === fmts[k].id));
      const p = problem(fmts[k]);
      let failed = false;
      if (typeof glr !== "string") {
        const key = fmts[k].id + "/" + st.image;
        if (!glr.tex.has(key)) glr.tex.set(key, makeTexture(glr, fmts[k].id, img));
        failed = glr.tex.get(key) === null;
      }
      s.cap.innerHTML = `<b>${fmts[k].label}</b> (${fmts[k].channels}): ${fmts[k].bytes} B/texel, ${fmts[k].kind}` +
        (failed ? `<br><span class="tf-warn">this browser refused to create the texture</span>` :
          p ? `<br><span class="tf-warn">${p}</span>` : `<br>${fmts[k].note}`);
    });

    if (typeof glr !== "string") {
      const gl = glr.gl;
      gl.clearColor(1, 1, 1, 1);
      gl.viewport(0, 0, CW, CH);
      gl.clear(gl.COLOR_BUFFER_BIT);
      fmts.forEach((f, k) => {
        const t = glr.tex.get(f.id + "/" + st.image);
        gl.viewport(k * (VW + 20), 0, VW, CH);
        if (!t) return;
        const isInt = f.kind === "integer";
        const prog = isInt ? glr.progU : glr.progF;
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, t);
        const filt = st.linear ? gl.LINEAR : gl.NEAREST;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
        gl.uniform1i(gl.getUniformLocation(prog, "tex"), 0);
        gl.uniform1f(gl.getUniformLocation(prog, "lo"), st.lo);
        gl.uniform1f(gl.getUniformLocation(prog, "hi"), st.hi);
        gl.uniform1f(gl.getUniformLocation(prog, "zoom"), st.zoom);
        if (!isInt) gl.uniform1i(gl.getUniformLocation(prog, "grey"), f.id === "R16F" || f.id === "R32F" ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      });
    }

    // readout: the selected texel in both formats, and memory
    const [r, c] = st.sel;
    const x = img.data[r * img.w + c];
    const lines = [`texel (row ${r}, col ${c}), value ${x.toFixed(6)}`];
    fmts.forEach(f => {
      const s = storeTexel(f.id, x);
      const dec = s.decoded.map(v => v.toFixed(6)).join(" / ");
      lines.push(`${f.label}: ${s.stored}`, `  → ${dec}  (error ${(s.decoded[0] - x).toExponential(1).replace("-", "−")})`);
    });
    const mem = fmts.map(f => `${(textureBytes(f.id, 2048, 2048, false) / 2 ** 20).toFixed(0)}`);
    lines.push(`memory 2048²: ${mem[0]} | ${mem[1]} MiB`, `  (× 4/3 with mipmaps)`);
    ro.textContent = lines.join("\n");
    note.textContent = `Left: ${fmts[0].label}, right: ${fmts[1].label}. Values from ${st.lo.toFixed(4)} to ${st.hi.toFixed(4)} are stretched to black…white on screen, so each stored step shows as a band. ` +
      (typeof glr !== "string" ? `This GPU ${glr.floatLinear ? "can" : "cannot"} filter float32 textures. ` : "") + "Click a picture to inspect a texel.";
  }

  draw();
}

autoMount(NAME, mount);
