// Loading Rust -> WebAssembly modules and model files in demos (see demos/README.md,
// "WebAssembly" and "Models").
//
// Two ways to get a module's bytes:
// - inline (small modules, up to a few hundred kB): bundled into the demo script by esbuild's
//   binary loader, `import bytes from "../../dist/wasm/<name>/<name>_bg.wasm"`; works from
//   file:// too. Instantiate with `instantiateInline(glue, bytes)`.
// - fetched (large modules, model weights): same-origin from demos/dist/wasm/ or
//   demos/dist/models/, with a progress callback and the Cache API, so a file is downloaded
//   once per browser. Opened from file:// this cannot work: `isFileProtocol()` and
//   `fileProtocolMessage()` give the page a clear message instead.
//
// The JS glue that wasm-bindgen writes (demos/dist/wasm/<name>/<name>.js, `--target web`) is
// always bundled statically: `import * as glue from "../../dist/wasm/<name>/<name>.js"`.
//
// No DOM access at import time except reading document.currentScript (guarded), so the pure
// parts run in node tests (demos/test/wasm-test.ts).

/** The part of a wasm-bindgen `--target web` glue module this loader uses. */
export interface WasmGlue<E = unknown> {
  default: (opts: { module_or_path: BufferSource | WebAssembly.Module }) => Promise<E>;
  initSync: (opts: { module: BufferSource | WebAssembly.Module }) => E;
}

export type Progress = (loaded: number, total: number) => void;

export interface FetchOptions {
  /** Called while downloading; total is 0 when the server sends no length. */
  onProgress?: Progress;
  /** Cache key version (e.g. the sha256 from index.json); a new version replaces the old entry. */
  version?: string;
  /** Use the Cache API (default true; silently skipped where it does not exist). */
  cache?: boolean;
  /** Expected size in bytes, used for progress when the response has no Content-Length. */
  size?: number;
  signal?: AbortSignal;
}

export const CACHE_NAME = "qb-demos-v1";

// ---- Where the built files are --------------------------------------------------------------

// The script element that is running now (demo bundles are classic IIFE scripts, so this is set
// while their top-level code, including this module's, runs).
const SCRIPT_SRC: string = (() => {
  try {
    const s = typeof document !== "undefined" ? (document.currentScript as HTMLScriptElement | null) : null;
    return s && s.src ? s.src : "";
  } catch { return ""; }
})();

/** Base URL of demos/dist/ (ends in "/"), derived from the demo script's own URL; works at any
 *  page depth. Pass `scriptSrc` to compute it for another script (tests). */
export function distBase(scriptSrc: string = SCRIPT_SRC): string {
  let src = scriptSrc;
  if (!src && typeof document !== "undefined") {
    const el = Array.from(document.querySelectorAll("script[src]"))
      .find((s) => /\/demos\/dist\//.test((s as HTMLScriptElement).src)) as HTMLScriptElement | undefined;
    src = el ? el.src : (typeof location !== "undefined" ? location.href : "");
  }
  const i = src.lastIndexOf("/demos/dist/");
  if (i >= 0) return src.slice(0, i + "/demos/dist/".length);
  return src ? new URL(".", src).href : "";
}

/** URL of a built Wasm module: demos/dist/wasm/<name>/<name>_bg.wasm. */
export function wasmUrl(name: string, base = distBase()): string {
  return new URL(`wasm/${name}/${name}_bg.wasm`, base).href;
}

/** URL of a model or sample file: demos/dist/models/<path>. */
export function modelUrl(path: string, base = distBase()): string {
  return new URL(`models/${path}`, base).href;
}

/** True when the page was opened as a local file (fetching other files is then impossible). */
export function isFileProtocol(href?: string): boolean {
  const h = href ?? (typeof location !== "undefined" ? location.href : "");
  return h.startsWith("file:");
}

/** Student-facing text for the file:// case. */
export function fileProtocolMessage(what = "This demo"): string {
  return `${what} loads a trained network from the course web site, which a page opened as a ` +
    "local file cannot do. Open the online version of this page instead.";
}

/** Human-readable size: 1234567 -> "1.2 MB". */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${(n / 1048576).toFixed(n < 10 * 1048576 ? 1 : 0)} MB`;
}

// ---- Fetching with progress and the Cache API -----------------------------------------------

/** Read a Response body to bytes, reporting progress. */
export async function readWithProgress(resp: Response, onProgress?: Progress, size = 0): Promise<Uint8Array> {
  const total = Number(resp.headers.get("Content-Length")) || size || 0;
  if (!resp.body || !onProgress) {
    const b = new Uint8Array(await resp.arrayBuffer());
    onProgress?.(b.length, total || b.length);
    return b;
  }
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  onProgress(0, total);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  onProgress(loaded, total || loaded);
  return out;
}

function cacheKey(url: string, version?: string): string {
  return version ? `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(version)}` : url;
}

/** Resolve to `fallback` if `p` has not settled after `ms`: the Cache API can stall (seen in
 *  headless Chrome, and with some privacy settings), and must never block loading. */
function withTimeout<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  return new Promise<T>((resolve) => {
    const t = setTimeout(() => resolve(fallback), ms);
    p.then((v) => { clearTimeout(t); resolve(v); }, () => { clearTimeout(t); resolve(fallback); });
  });
}

async function openCache(): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null; // node, or not a secure context
    return await withTimeout(caches.open(CACHE_NAME), 3000, null);
  } catch { return null; }
}

/** Fetch a same-origin file as bytes: from the Cache API when there, else from the network
 *  (with progress), then stored in the cache under `url?v=<version>`; older versions of the
 *  same URL are removed. Throws a readable Error on file://, HTTP errors and network errors. */
export async function fetchBytes(url: string, opts: FetchOptions = {}): Promise<Uint8Array> {
  if (isFileProtocol()) throw new Error(fileProtocolMessage());
  const useCache = opts.cache !== false;
  const key = cacheKey(url, opts.version);
  const cache = useCache ? await openCache() : null;
  if (cache) {
    try {
      const hit = await withTimeout(cache.match(key), 3000, undefined);
      if (hit) return await readWithProgress(hit, opts.onProgress, opts.size);
    } catch { /* fall through to the network */ }
  }
  let resp: Response;
  try {
    resp = await fetch(url, { signal: opts.signal, cache: opts.version ? "default" : "no-cache" });
  } catch (e) {
    throw new Error(`Could not download ${url.split("/").pop()}: ${(e as Error).message}`);
  }
  if (!resp.ok) throw new Error(`Could not download ${url.split("/").pop()} (HTTP ${resp.status})`);
  const bytes = await readWithProgress(resp, opts.onProgress, opts.size);
  if (cache) {
    // Drop older versions of the same file, then store this one (bounded in time; on failure,
    // e.g. quota or private mode, everything still works, only without caching).
    const store = async () => {
      for (const req of await cache.keys()) {
        if (req.url !== key && req.url.split("?")[0] === url.split("?")[0]) await cache.delete(req);
      }
      await cache.put(key, new Response(bytes as BlobPart, {
        headers: { "Content-Type": "application/octet-stream", "Content-Length": String(bytes.length) },
      }));
    };
    await withTimeout(store(), 10000, undefined);
  }
  return bytes;
}

/** Fetch a small JSON file (no Cache API: always revalidated). */
export async function fetchJSON<T = unknown>(url: string): Promise<T> {
  if (isFileProtocol()) throw new Error(fileProtocolMessage());
  const resp = await fetch(url, { cache: "no-cache" });
  if (!resp.ok) throw new Error(`Could not load ${url.split("/").pop()} (HTTP ${resp.status})`);
  return await resp.json() as T;
}

// ---- Instantiation --------------------------------------------------------------------------

/** The bytes of an inlined module. TypeScript types `import x from "…_bg.wasm"` from the
 *  generated .d.ts (the module's exports), but esbuild's binary loader makes it a Uint8Array. */
export function inlineBytes(imported: unknown): Uint8Array {
  if (!(imported instanceof Uint8Array)) {
    throw new Error("Inlined .wasm is not bytes: bundle with esbuild --loader:.wasm=binary (demos/build.sh does)");
  }
  return imported;
}

/** Instantiate a module whose bytes are inlined in the bundle (works on file://). `bytes` is
 *  the default import of the .wasm file. Asynchronous on purpose: Chrome refuses synchronous
 *  compilation (the glue's initSync) of modules over 4 kB on the main thread. */
export async function instantiateInline<E>(glue: WasmGlue<E>, bytes: unknown): Promise<E> {
  return await glue.default({ module_or_path: inlineBytes(bytes) as BufferSource });
}

/** One entry of demos/dist/wasm/index.json or demos/dist/models/index.json. */
export interface IndexEntry { file: string; bytes: number; sha256: string; [k: string]: unknown }

const indexCache = new Map<string, Promise<Record<string, IndexEntry>>>();
function loadIndex(url: string): Promise<Record<string, IndexEntry>> {
  let p = indexCache.get(url);
  if (!p) {
    p = fetchJSON<Record<string, IndexEntry>>(url).catch((e) => { indexCache.delete(url); throw e; });
    indexCache.set(url, p);
  }
  return p;
}

/** Fetch demos/dist/wasm/<name>/<name>_bg.wasm (progress, Cache API keyed on its sha256 from
 *  wasm/index.json) and instantiate it through its glue. */
export async function loadWasm<E>(glue: WasmGlue<E>, name: string, opts: FetchOptions & { base?: string } = {}): Promise<E> {
  const base = opts.base ?? distBase();
  let entry: IndexEntry | undefined;
  try { entry = (await loadIndex(new URL("wasm/index.json", base).href))[name]; } catch { /* no index: no versioning */ }
  const bytes = await fetchBytes(wasmUrl(name, base), { ...opts, version: opts.version ?? entry?.sha256, size: opts.size ?? entry?.bytes });
  return await glue.default({ module_or_path: bytes as BufferSource });
}

/** Fetch a model or sample file listed in demos/dist/models/index.json by its key (e.g.
 *  "stardist/2D_versatile_fluo/weights_best.h5"), cached by its sha256. */
export async function loadModelFile(key: string, opts: FetchOptions & { base?: string } = {}): Promise<Uint8Array> {
  const base = opts.base ?? distBase();
  const index = await loadIndex(modelUrl("index.json", base));
  const entry = index[key];
  if (!entry) throw new Error(`Model file ${key} is not on this site (not in models/index.json)`);
  return await fetchBytes(modelUrl(entry.file, base), { ...opts, version: entry.sha256, size: entry.bytes });
}

/** The models index (key -> {file, bytes, sha256, ...}). */
export function modelIndex(base = distBase()): Promise<Record<string, IndexEntry>> {
  return loadIndex(modelUrl("index.json", base));
}

// ---- A small progress/status element (DOM) -------------------------------------------------

export interface LoadStatus {
  el: HTMLElement;
  /** Progress callback to pass to fetchBytes/loadWasm/loadModelFile. */
  progress: Progress;
  text(msg: string): void;
  error(msg: string): void;
  done(): void;
}

/** A progress bar with a label, e.g. "Loading network: 3.1 / 9.8 MB". Colours come from the
 *  --qb-* variables of demo.css; inline styles only, so it needs no CSS file. */
export function loadStatus(parent: HTMLElement, label = "Loading network"): LoadStatus {
  const el = document.createElement("div");
  el.className = "qb-loadstatus";
  el.style.cssText = "font-size:16px;margin:6px 0;color:var(--qb-fg,#222)";
  const txt = document.createElement("div");
  const bar = document.createElement("div");
  bar.style.cssText = "height:8px;border-radius:4px;background:var(--qb-grid,#ddd);overflow:hidden;margin-top:4px";
  const fill = document.createElement("div");
  fill.style.cssText = "height:100%;width:0%;background:var(--qb-accent,#2a6fdb);transition:width .1s";
  bar.appendChild(fill);
  el.append(txt, bar);
  parent.appendChild(el);
  txt.textContent = isFileProtocol() ? fileProtocolMessage() : label + " …";
  if (isFileProtocol()) bar.style.display = "none";
  return {
    el,
    progress(loaded, total) {
      txt.textContent = total > 0
        ? `${label}: ${formatBytes(loaded)} / ${formatBytes(total)}`
        : `${label}: ${formatBytes(loaded)}`;
      fill.style.width = total > 0 ? `${Math.min(100, (100 * loaded) / total).toFixed(1)}%` : "30%";
    },
    text(msg) { txt.textContent = msg; },
    error(msg) {
      txt.textContent = msg;
      txt.style.color = "var(--qb-c2,#d62728)";
      bar.style.display = "none";
    },
    done() { el.remove(); },
  };
}
