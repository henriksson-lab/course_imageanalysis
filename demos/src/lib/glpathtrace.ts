// A progressive path tracer in WebGL2 (owner: ray-tracing day, unit pathtrace/; the CPU
// reference is `tracePath` in lib/raytrace.ts, with the same scene, the same estimator and the
// same random numbers per (pixel, sample, seed)).
//
// One fragment program on a full-screen triangle traces one path per pixel per pass and adds it
// to a running sum; two float render targets are swapped every pass (ping-pong). The alpha
// channel accumulates the squared luminance, for the per-pixel standard deviation. Scenes have
// up to 4 spheres and 8 parallelograms (no meshes). A display pass shows the mean (linear →
// sRGB), its standard deviation, the error against a reference, or the first-hit normal/albedo.
// Needs EXT_color_buffer_float (RGBA32F, else RGBA16F); `GLPathTracer.create` returns null when
// that fails, so the caller falls back to the CPU tracer. No keyboard, no network.
import type { PathScene, Camera, PathOptions } from "./raytrace";
import { cameraBasis } from "./raytrace";

const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const TRACE_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uPrev;
uniform ivec2 uRes; uniform int uSample; uniform int uSeed; uniform int uMode;
uniform int uMaxB; uniform int uRR; uniform int uSampling; uniform int uWhitted;
uniform vec3 uEye, uR, uU, uF; uniform float uTanH;
uniform int uNS; uniform vec4 uSph[4]; uniform vec4 uSphMat[4]; uniform float uSphIor[4];
uniform int uNQ; uniform vec3 uQP[8]; uniform vec3 uQU[8]; uniform vec3 uQV[8]; uniform vec4 uQMat[8]; uniform vec3 uQEmit[8];
uniform int uLight;
out vec4 outC;
uint st;
uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float rnd() { st = pcg(st); return float(st) / 4294967296.0; }
const float PI = 3.14159265358979;
struct Hit { float t; vec3 n; vec3 col; int kind; float ior; vec3 emit; int id; };
bool sceneHit(vec3 o, vec3 d, float tmax, out Hit h) {
  h.t = tmax; bool any = false;
  for (int i = 0; i < 4; i++) {
    if (i >= uNS) break;
    vec3 oc = o - uSph[i].xyz; float b = dot(d, oc), k = dot(oc, oc) - uSph[i].w * uSph[i].w, disc = b * b - k;
    if (disc < 0.0) continue;
    float s = sqrt(disc), q = -(b + (b >= 0.0 ? s : -s));
    float t1 = q, t2 = q != 0.0 ? k / q : -b;
    if (t1 > t2) { float tt = t1; t1 = t2; t2 = tt; }
    float t = t1 > 0.0 ? t1 : t2;
    if (t > 0.0 && t < h.t) { h.t = t; h.n = (o + t * d - uSph[i].xyz) / uSph[i].w; h.col = uSphMat[i].rgb; h.kind = int(uSphMat[i].w + 0.5); h.ior = uSphIor[i]; h.emit = vec3(0); h.id = i; any = true; }
  }
  for (int i = 0; i < 8; i++) {
    if (i >= uNQ) break;
    vec3 pv = cross(d, uQV[i]); float det = dot(uQU[i], pv);
    if (abs(det) < 1e-15) continue;
    float inv = 1.0 / det; vec3 tv = o - uQP[i]; float s = dot(tv, pv) * inv;
    if (s < 0.0 || s > 1.0) continue;
    vec3 qv = cross(tv, uQU[i]); float w = dot(d, qv) * inv;
    if (w < 0.0 || w > 1.0) continue;
    float t = dot(uQV[i], qv) * inv;
    if (t > 0.0 && t < h.t) { h.t = t; h.n = normalize(cross(uQU[i], uQV[i])); h.col = uQMat[i].rgb; h.kind = int(uQMat[i].w + 0.5); h.ior = 1.5; h.emit = uQEmit[i]; h.id = 100 + i; any = true; }
  }
  return any;
}
vec3 toWorld(vec3 v, vec3 n) {
  float s = n.z >= 0.0 ? 1.0 : -1.0, a = -1.0 / (s + n.z), b = n.x * n.y * a;
  vec3 t = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x), bt = vec3(b, s + n.y * n.y * a, -n.y);
  return v.x * t + v.y * bt + v.z * n;
}
float fresnelR(float ci, float n1, float n2) {
  ci = min(1.0, abs(ci)); float si = n1 / n2 * sqrt(max(0.0, 1.0 - ci * ci));
  if (si >= 1.0) return 1.0;
  float ct = sqrt(1.0 - si * si);
  float rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct), rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
  return 0.5 * (rs * rs + rp * rp);
}
vec3 directLight(vec3 x, vec3 n, vec3 alb, bool centre) {
  int li = uLight; if (li < 0) return vec3(0);
  float s = centre ? 0.5 : rnd(), w = centre ? 0.5 : rnd();
  vec3 y = uQP[li] + s * uQU[li] + w * uQV[li];
  vec3 cr = cross(uQU[li], uQV[li]); float A = length(cr); vec3 nl = cr / A;
  vec3 toL = y - x; float d2 = dot(toL, toL), dist = sqrt(d2); vec3 wi = toL / dist;
  float cx = dot(n, wi), cl = -dot(nl, wi);
  if (cx <= 0.0 || cl <= 0.0) return vec3(0);
  Hit h;
  if (sceneHit(x + n * 1e-4, wi, dist - 2e-4, h) && h.id != 100 + li) return vec3(0);
  return alb * uQEmit[li] * (cx * cl / d2 * A / PI);
}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy); int pi = (uRes.y - 1 - px.y) * uRes.x + px.x;
  st = pcg(uint(pi) ^ pcg(uint(uSample) ^ pcg(uint(uSeed))));
  float jx = uMode == 0 ? rnd() : 0.5, jy = uMode == 0 ? rnd() : 0.5;
  float xn = 2.0 * (float(px.x) + jx) / float(uRes.x) - 1.0, yn = 1.0 - 2.0 * (float(uRes.y - 1 - px.y) + jy) / float(uRes.y);
  float asp = float(uRes.x) / float(uRes.y);
  vec3 o = uEye, d = normalize(uF + xn * uTanH * asp * uR + yn * uTanH * uU);
  vec4 prev = texelFetch(uPrev, px, 0);
  if (uMode != 0) {
    Hit h; vec3 c = vec3(0);
    if (sceneHit(o, d, 1e30, h)) { vec3 nf = dot(d, h.n) < 0.0 ? h.n : -h.n; c = uMode == 1 ? 0.5 + 0.5 * nf : (h.kind == 3 ? vec3(1) : h.col); }
    outC = vec4(c, 1.0); return;
  }
  vec3 L = vec3(0), T = vec3(1); bool spec = true; bool nee = uSampling == 2 || uWhitted == 1;
  for (int k = 0; k < 64; k++) {
    Hit h;
    if (!sceneHit(o, d, 1e30, h)) break;
    bool front = dot(d, h.n) < 0.0; vec3 nf = front ? h.n : -h.n;
    if (h.kind == 3) { if (front && (spec || !nee)) L += T * h.emit; break; }
    if (k >= uMaxB) break;
    vec3 p = o + h.t * d;
    if (h.kind == 0) {
      if (nee) L += T * directLight(p, nf, h.col, uWhitted == 1);
      if (uWhitted == 1) break;
      vec3 wi;
      if (uSampling == 0) { float z = rnd(), r = sqrt(max(0.0, 1.0 - z * z)), ph = 2.0 * PI * rnd(); wi = toWorld(vec3(r * cos(ph), r * sin(ph), z), nf); T *= h.col * 2.0 * dot(wi, nf); }
      else { float r = sqrt(rnd()), ph = 2.0 * PI * rnd(); vec2 dd = vec2(r * cos(ph), r * sin(ph)); wi = toWorld(vec3(dd, sqrt(max(0.0, 1.0 - dot(dd, dd)))), nf); T *= h.col; }
      o = p + nf * 1e-4; d = wi; spec = false;
    } else if (h.kind == 1) {
      d = reflect(d, nf); o = p + nf * 1e-4; T *= h.col; spec = true;
    } else {
      float n1 = front ? 1.0 : h.ior, n2 = front ? h.ior : 1.0, eta = n1 / n2, c1 = -dot(d, nf), kk = 1.0 - eta * eta * (1.0 - c1 * c1);
      float R = kk < 0.0 ? 1.0 : fresnelR(c1, n1, n2);
      if (rnd() < R) { d = reflect(d, nf); o = p + nf * 1e-4; }
      else { d = normalize(eta * d + (eta * c1 - sqrt(kk)) * nf); o = p - nf * 1e-4; }
      T *= h.col; spec = true;
    }
    if (uRR == 1 && k >= 2) { float q = min(0.95, max(T.r, max(T.g, T.b))); if (rnd() >= q) break; T /= q; }
  }
  float y = dot(L, vec3(0.2126, 0.7152, 0.0722));
  outC = prev + vec4(L, y * y);
}`;

const SHOW_FS = `#version 300 es
precision highp float;
uniform sampler2D uAcc; uniform sampler2D uRef; uniform sampler2D uAux;
uniform float uN; uniform int uView; uniform float uScale; uniform int uHasRef;
out vec4 outC;
vec3 srgb(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
vec3 inferno(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.0, 0.0, 0.016), c1 = vec3(0.29, 0.047, 0.42), c2 = vec3(0.81, 0.27, 0.27), c3 = vec3(0.98, 0.61, 0.024), c4 = vec3(0.99, 1.0, 0.64);
  if (t < 0.25) return mix(c0, c1, t / 0.25); if (t < 0.5) return mix(c1, c2, (t - 0.25) / 0.25); if (t < 0.75) return mix(c2, c3, (t - 0.5) / 0.25); return mix(c3, c4, (t - 0.75) / 0.25);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(uAcc, p, 0); vec3 m = a.rgb / max(uN, 1.0);
  float y = dot(m, vec3(0.2126, 0.7152, 0.0722));
  if (uView == 0) outC = vec4(srgb(m), 1.0);
  else if (uView == 1) { float v = max(0.0, a.a / max(uN, 1.0) - y * y); outC = vec4(inferno(sqrt(v / max(uN, 1.0)) * uScale), 1.0); }
  else if (uView == 2) { vec3 r = texelFetch(uRef, p, 0).rgb; outC = uHasRef == 1 ? vec4(inferno(length(m - r) * uScale), 1.0) : vec4(0.3, 0.3, 0.3, 1.0); }
  else if (uView == 3) outC = vec4(texelFetch(uAux, p, 0).rgb, 1.0);
  else outC = vec4(srgb(texelFetch(uRef, p, 0).rgb), 1.0);
}`;

export type View = "image" | "std" | "error" | "normal" | "albedo" | "reference";

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const mk = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
  return p;
}

interface Target { tex: WebGLTexture; fbo: WebGLFramebuffer }

export class GLPathTracer {
  readonly gl: WebGL2RenderingContext;
  private trace: WebGLProgram; private show: WebGLProgram;
  private acc: Target[]; private ref: Target[]; private aux: Target[];
  private cur = 0; private refCur = 0;
  private internal: number; private type: number;
  spp = 0; refSpp = 0;
  hasRef = false;
  scene: PathScene | null = null; cam: Camera | null = null;
  opt: PathOptions = { maxBounces: 5, russianRoulette: true, sampling: "light" };
  seed = 1;
  readonly floatType: "RGBA32F" | "RGBA16F";

  private constructor(readonly canvas: HTMLCanvasElement, readonly W: number, readonly H: number, gl: WebGL2RenderingContext, f32: boolean) {
    this.gl = gl;
    this.floatType = f32 ? "RGBA32F" : "RGBA16F";
    this.internal = f32 ? gl.RGBA32F : gl.RGBA16F; this.type = f32 ? gl.FLOAT : gl.HALF_FLOAT;
    this.trace = compile(gl, VS, TRACE_FS); this.show = compile(gl, VS, SHOW_FS);
    this.acc = [this.target(), this.target()]; this.ref = [this.target(), this.target()]; this.aux = [this.target(), this.target()];
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error("float framebuffer incomplete");
    gl.bindVertexArray(gl.createVertexArray());
  }

  /** Create on a canvas of W × H pixels, or null without WebGL2 / float render targets. */
  static create(canvas: HTMLCanvasElement, W: number, H: number): GLPathTracer | null {
    try {
      canvas.width = W; canvas.height = H;
      const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
      if (!gl) return null;
      const f32 = !!gl.getExtension("EXT_color_buffer_float");
      if (!f32 && !gl.getExtension("EXT_color_buffer_half_float")) return null;
      return new GLPathTracer(canvas, W, H, gl, f32);
    } catch { return null; }
  }

  private target(): Target {
    const gl = this.gl, tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.internal, this.W, this.H, 0, gl.RGBA, this.type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    return { tex, fbo };
  }

  setScene(sc: PathScene, cam: Camera): void { this.scene = sc; this.cam = cam; this.reset(); this.renderAux(); }
  setOptions(o: PathOptions): void { this.opt = o; this.reset(); }

  reset(): void {
    const gl = this.gl;
    for (const t of this.acc) { gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    this.spp = 0;
  }
  resetReference(): void {
    const gl = this.gl;
    for (const t of this.ref) { gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    this.refSpp = 0; this.hasRef = false;
  }

  private uniforms(p: WebGLProgram, sample: number, seed: number, mode: number, opt: PathOptions): void {
    const gl = this.gl, sc = this.scene!, cam = this.cam!;
    const U = (n: string) => gl.getUniformLocation(p, n);
    gl.uniform2i(U("uRes"), this.W, this.H); gl.uniform1i(U("uSample"), sample); gl.uniform1i(U("uSeed"), seed); gl.uniform1i(U("uMode"), mode);
    gl.uniform1i(U("uMaxB"), opt.maxBounces); gl.uniform1i(U("uRR"), opt.russianRoulette ? 1 : 0);
    gl.uniform1i(U("uSampling"), opt.sampling === "uniform" ? 0 : opt.sampling === "cosine" ? 1 : 2); gl.uniform1i(U("uWhitted"), opt.whitted ? 1 : 0);
    const b = cameraBasis(cam);
    gl.uniform3fv(U("uEye"), cam.eye); gl.uniform3fv(U("uR"), b.r); gl.uniform3fv(U("uU"), b.u); gl.uniform3fv(U("uF"), b.f);
    gl.uniform1f(U("uTanH"), Math.tan(cam.fov * Math.PI / 360));
    const sph = sc.objects.filter(o => o.type === "sphere"), qd = sc.objects.filter(o => o.type === "quad");
    const kind = (k: string) => (k === "diffuse" ? 0 : k === "mirror" ? 1 : k === "glass" ? 2 : 3);
    const S = new Float32Array(16), SM = new Float32Array(16), SI = new Float32Array(4);
    sph.forEach((o, i) => { if (o.type !== "sphere" || i >= 4) return; S.set([...o.c, o.r], 4 * i); SM.set([...o.mat.colour, kind(o.mat.kind)], 4 * i); SI[i] = o.mat.ior ?? 1.5; });
    gl.uniform1i(U("uNS"), Math.min(4, sph.length)); gl.uniform4fv(U("uSph"), S); gl.uniform4fv(U("uSphMat"), SM); gl.uniform1fv(U("uSphIor"), SI);
    const QP = new Float32Array(24), QU = new Float32Array(24), QV = new Float32Array(24), QM = new Float32Array(32), QE = new Float32Array(24);
    let light = -1;
    qd.forEach((o, i) => {
      if (o.type !== "quad" || i >= 8) return;
      QP.set(o.p, 3 * i); QU.set(o.u, 3 * i); QV.set(o.v, 3 * i); QM.set([...o.mat.colour, kind(o.mat.kind)], 4 * i); QE.set(o.mat.emit ?? [0, 0, 0], 3 * i);
      if (sc.lights.includes(sc.objects.indexOf(o))) light = i;
    });
    gl.uniform1i(U("uNQ"), Math.min(8, qd.length)); gl.uniform3fv(U("uQP"), QP); gl.uniform3fv(U("uQU"), QU); gl.uniform3fv(U("uQV"), QV); gl.uniform4fv(U("uQMat"), QM); gl.uniform3fv(U("uQEmit"), QE);
    gl.uniform1i(U("uLight"), light);
  }

  private pass(tg: Target[], curIdx: number, sample: number, seed: number, mode: number, opt: PathOptions): number {
    const gl = this.gl, src = tg[curIdx], dst = tg[1 - curIdx];
    gl.useProgram(this.trace);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo); gl.viewport(0, 0, this.W, this.H);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.tex);
    gl.uniform1i(gl.getUniformLocation(this.trace, "uPrev"), 0);
    this.uniforms(this.trace, sample, seed, mode, opt);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return 1 - curIdx;
  }

  /** Add n samples per pixel. */
  step(n = 1): void { if (!this.scene) return; for (let k = 0; k < n; k++) { this.cur = this.pass(this.acc, this.cur, this.spp, this.seed, 0, this.opt); this.spp++; } }

  /** Add n samples per pixel to the reference (full global illumination, a different seed). */
  stepReference(n = 1, opt: PathOptions = { maxBounces: 12, russianRoulette: true, sampling: "light" }): void {
    if (!this.scene) return;
    for (let k = 0; k < n; k++) { this.refCur = this.pass(this.ref, this.refCur, this.refSpp, this.seed + 7777, 0, opt); this.refSpp++; }
  }

  /** Normal (aux[0]) and albedo (aux[1]) of the first hit through the pixel centres. */
  renderAux(): void {
    for (const [i, mode] of [[0, 1], [1, 2]]) {
      const gl = this.gl;
      gl.useProgram(this.trace);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.aux[i].fbo); gl.viewport(0, 0, this.W, this.H);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.acc[0].tex);
      this.uniforms(this.trace, 0, this.seed, mode, this.opt);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  /** Normalise the reference into the reference target's mean (kept as a sum; uN handles it). */
  finishReference(): void { this.hasRef = this.refSpp > 0; }

  /** Draw a view to the canvas. `scale` multiplies the std / error values before the colour map. */
  display(view: View, scale = 4): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, this.W, this.H);
    gl.useProgram(this.show);
    const U = (n: string) => gl.getUniformLocation(this.show, n);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (view === "reference" ? this.ref[this.refCur] : this.acc[this.cur]).tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.refMeanTex ?? this.ref[this.refCur].tex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.aux[view === "albedo" ? 1 : 0].tex);
    gl.uniform1i(U("uAcc"), 0); gl.uniform1i(U("uRef"), 1); gl.uniform1i(U("uAux"), 2);
    gl.uniform1f(U("uN"), view === "reference" ? this.refSpp : this.spp);
    gl.uniform1i(U("uView"), view === "image" || view === "reference" ? 0 : view === "std" ? 1 : view === "error" ? 2 : 3);
    gl.uniform1f(U("uScale"), scale); gl.uniform1i(U("uHasRef"), this.hasRef ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Reference mean as a texture (for the error view); built by `uploadRefMean`. */
  private refMeanTex: WebGLTexture | null = null;
  uploadRefMean(mean: Float32Array): void {
    const gl = this.gl;
    if (!this.refMeanTex) this.refMeanTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.refMeanTex);
    const rgba = new Float32Array(4 * this.W * this.H);
    // mean is row 0 = top; textures have row 0 = bottom
    for (let j = 0; j < this.H; j++) for (let i = 0; i < this.W; i++) { const s = 3 * (j * this.W + i), d = 4 * ((this.H - 1 - j) * this.W + i); rgba[d] = mean[s]; rgba[d + 1] = mean[s + 1]; rgba[d + 2] = mean[s + 2]; rgba[d + 3] = 1; }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.W, this.H, 0, gl.RGBA, gl.FLOAT, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.hasRef = true;
  }

  private readTarget(t: Target, divide: number): { rgb: Float32Array; sq: Float32Array } {
    const gl = this.gl, buf = new Float32Array(4 * this.W * this.H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.readPixels(0, 0, this.W, this.H, gl.RGBA, gl.FLOAT, buf);
    const rgb = new Float32Array(3 * this.W * this.H), sq = new Float32Array(this.W * this.H), k = divide > 0 ? 1 / divide : 0;
    for (let j = 0; j < this.H; j++) for (let i = 0; i < this.W; i++) {
      const s = 4 * ((this.H - 1 - j) * this.W + i), d = j * this.W + i;
      rgb[3 * d] = buf[s] * k; rgb[3 * d + 1] = buf[s + 1] * k; rgb[3 * d + 2] = buf[s + 2] * k; sq[d] = buf[s + 3] * k;
    }
    return { rgb, sq };
  }
  /** Mean radiance per pixel (linear RGB, row 0 = top). */
  readMean(): Float32Array { return this.readTarget(this.acc[this.cur], this.spp).rgb; }
  readReference(): Float32Array { return this.readTarget(this.ref[this.refCur], this.refSpp).rgb; }
  readAux(): { normal: Float32Array; albedo: Float32Array } { return { normal: this.readTarget(this.aux[0], 1).rgb, albedo: this.readTarget(this.aux[1], 1).rgb }; }

  dispose(): void { this.gl.getExtension("WEBGL_lose_context")?.loseContext(); }
}
