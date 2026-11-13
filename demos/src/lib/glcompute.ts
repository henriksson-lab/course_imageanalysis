// "Compute through a fragment shader" with WebGL2 (owner: GPU day, demo gpufilter; available
// to later demos). WebGL2 has no compute shaders, but a full-screen triangle drawn into a
// float framebuffer runs one small program per output pixel: a map or a stencil.
//
//   const cc = new ComputeContext(hostElement);   // context created on first use, released
//                                                // when the element leaves the screen
//   const gl = cc.get(); if (!gl) show cc.error
//   const src = cc.texture(data, w, h);           // Float32Array -> R32F (or R16F) texture
//   const dst = cc.target(w, h);                  // float render target
//   const prog = cc.program(FRAGMENT_SOURCE);     // "in vec2 uv; out vec4 o;" etc.
//   cc.run(prog, dst, { u_src: src }, gl => gl.uniform1i(...));
//   const out = cc.read(dst);                     // Float32Array, w*h
//
// Timing helpers: cc.finish() waits for the GPU (a 1-pixel read-back), median().
// It does not replace gl3d.ts (scene rendering).

export type FloatPrecision = "f32" | "f16";

export interface Target { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number; prec: FloatPrecision }

const VERTEX = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/** Median of a list of numbers. */
export function median(xs: number[]): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

export class ComputeContext {
  canvas: HTMLCanvasElement | null = null;
  gl: WebGL2RenderingContext | null = null;
  error: string | null = null;
  /** RED/FLOAT read-back supported (else RGBA/FLOAT, 4x the bytes). */
  private readRed = false;
  private vao: WebGLVertexArrayObject | null = null;
  private progs = new Map<string, WebGLProgram>();
  private io: IntersectionObserver | null = null;
  private offTimer = 0;
  /** Bumped when the context is released: textures from an older generation are invalid. */
  generation = 0;

  constructor(watch?: Element) {
    if (watch && typeof IntersectionObserver !== "undefined") {
      this.io = new IntersectionObserver(es => {
        clearTimeout(this.offTimer);
        if (!es.some(e => e.isIntersecting)) this.offTimer = window.setTimeout(() => this.release(), 600);
      }, { rootMargin: "100px" });
      this.io.observe(watch);
    }
  }

  /** The WebGL2 context (created on first use), or null with `error` set. */
  get(): WebGL2RenderingContext | null {
    if (this.gl && !this.gl.isContextLost()) return this.gl;
    this.gl = null;
    try {
      const c = document.createElement("canvas");
      c.width = 1; c.height = 1;
      const gl = c.getContext("webgl2", { antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
      if (!gl) { this.error = "This browser has no WebGL2: the GPU cannot be measured here."; return null; }
      if (!gl.getExtension("EXT_color_buffer_float")) {
        gl.getExtension("WEBGL_lose_context")?.loseContext();
        this.error = "This browser's WebGL2 cannot render into float textures (EXT_color_buffer_float missing).";
        return null;
      }
      this.canvas = c; this.gl = gl; this.error = null; this.generation++;
      this.vao = gl.createVertexArray();
      this.progs.clear();
      // Can we read a single float channel back? (implementation-defined second format)
      const t = this.target(1, 1, "f32");
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
      this.readRed = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) === gl.RED && gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === gl.FLOAT;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.free(t);
      return gl;
    } catch (e) {
      this.error = "WebGL2 failed: " + (e as Error).message;
      return null;
    }
  }

  /** Release the context (it is recreated on the next get()). */
  release(): void {
    if (!this.gl) return;
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
    this.gl = null; this.canvas = null; this.vao = null; this.progs.clear();
  }

  /** Short description of the GPU, if the browser tells (often it does not). */
  renderer(): string {
    const gl = this.get();
    if (!gl) return "";
    return String(gl.getParameter(gl.RENDERER) ?? "");
  }

  /** Upload w*h floats as a single-channel float texture (R32F, or R16F: rounded to float16). */
  texture(data: Float32Array, w: number, h: number, prec: FloatPrecision = "f32"): WebGLTexture {
    const gl = this.gl!;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, prec === "f16" ? gl.R16F : gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
    this.nearest();
    return tex;
  }

  /** A float render target (texture + framebuffer). */
  target(w: number, h: number, prec: FloatPrecision = "f32"): Target {
    const gl = this.gl!;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, prec === "f16" ? gl.R16F : gl.R32F, w, h);
    this.nearest();
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb, w, h, prec };
  }

  free(t: Target | WebGLTexture | null): void {
    const gl = this.gl;
    if (!gl || !t) return;
    if (t instanceof WebGLTexture) { gl.deleteTexture(t); return; }
    gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.tex);
  }

  private nearest(): void {
    const gl = this.gl!;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Compile (and cache) a fragment shader. The source must start with "#version 300 es". */
  program(frag: string): WebGLProgram {
    const gl = this.gl!;
    const cached = this.progs.get(frag);
    if (cached) return cached;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, frag));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
    this.progs.set(frag, p);
    return p;
  }

  /** Run `prog` once per pixel of `dst`, with textures bound to the named sampler uniforms. */
  run(prog: WebGLProgram, dst: Target, textures: Record<string, WebGLTexture>, uniforms?: (gl: WebGL2RenderingContext, prog: WebGLProgram) => void): void {
    const gl = this.gl!;
    gl.useProgram(prog);
    let unit = 0;
    for (const [name, tex] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(prog, name), unit);
      unit++;
    }
    uniforms?.(gl, prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, dst.w, dst.h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  /** Wait until the GPU has finished everything queued (by reading one pixel of `t`). */
  finish(t: Target): void {
    const gl = this.gl!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    const px = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Read a target back as w*h floats. */
  read(t: Target): Float32Array {
    const gl = this.gl!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    let out: Float32Array;
    if (this.readRed) {
      out = new Float32Array(t.w * t.h);
      gl.readPixels(0, 0, t.w, t.h, gl.RED, gl.FLOAT, out);
    } else {
      const rgba = new Float32Array(4 * t.w * t.h);
      gl.readPixels(0, 0, t.w, t.h, gl.RGBA, gl.FLOAT, rgba);
      out = new Float32Array(t.w * t.h);
      for (let i = 0; i < out.length; i++) out[i] = rgba[4 * i];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}
