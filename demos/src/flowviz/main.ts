// Reading a flow field: one synthetic field (translation, rotation, expansion, shear, vortex,
// saddle, two cells moving apart, a wound closing) shown three ways side by side: arrows,
// the colour wheel (hue = direction, saturation = speed) and the speed map. Sliders change
// the field's strength and the arrow spacing. "Which field is this?" shows only the colour
// coding and asks for the field. data-mode="divcurl" adds divergence and curl maps.
//
// <div data-demo="flowviz" data-field="rotation" data-mode="divcurl|quiz"></div>
import { autoMount, htmlEl, slider, presets, button } from "../lib/ui";
import { FieldKind, FIELD_NAMES, fieldFn, sampleField, Flow, flowToRGBA, maxSpeed, divergence, curlScreen } from "../lib/flow";
import { viridis, diverging, divergingT } from "../lib/colour";
import { ImgView } from "../canny/imgview";
import { injectCss, flowWheel, drawQuiver, WHEEL_CSS, fmtSpeed } from "../lkflow/flowui";

const NAME = "flowviz", S = `.qb-demo[data-demo="${NAME}"]`;
const CSS = `
${S} { gap: .8em; flex-wrap: wrap; }
${S} .fv-views { flex: 1 1 560px; max-width: 900px; display: grid; grid-template-columns: repeat(3, 1fr); gap: .6em; min-width: 0; }
${S} .fv-views.four { grid-template-columns: repeat(4, 1fr); }
${S} .qb-panel { flex: 0 0 280px; gap: .3em; font-size: 15px; }
${S} .qb-readout { font-size: 15px; }
${S} .fv-lab { font-size: 14px; color: var(--qb-muted); margin-bottom: -.2em; }
${S} .fv-quiz { font-size: 18px; font-weight: 600; min-height: 1.4em; }
.reveal ${S} { font-size: 18px; flex-wrap: nowrap; }
.reveal ${S} .qb-panel { flex: 0 0 330px; gap: .2em; }
.reveal ${S} .qb-btn { font-size: 16px; padding: .06em .38em; }
.reveal ${S} .qb-readout { font-size: 18px; }
.reveal ${S} .fv-lab { font-size: 16px; }
@media (max-width: 860px) { ${S} { flex-wrap: wrap; } ${S} .fv-views { grid-template-columns: 1fr 1fr; } }
${WHEEL_CSS(S)}
`;

const KINDS: FieldKind[] = ["translation", "rotation", "zoom", "shear", "swirl", "saddle", "apart", "wound"];
const QUIZ_ORDER: [FieldKind, number][] = [["rotation", -1], ["zoom", -1], ["saddle", 1], ["translation", 1], ["swirl", 1], ["shear", -1], ["wound", 1], ["apart", 1], ["zoom", 1], ["rotation", 1]];
const N = 64;

/** Field parameters for strength k ∈ [−1, 1]: about 3 px/frame at the edge for k = ±1. */
function field(kind: FieldKind, k: number): Flow {
  const p = kind === "translation" ? { s: 3 * Math.abs(k), angle: k >= 0 ? 30 : 210 }
    : kind === "swirl" ? { s: 3 * k, r: 14 }
    : kind === "apart" ? { s: 3 * k, r: 12 }
    : kind === "wound" ? { s: 3 * k, r: 12 }
    : kind === "rotation" ? { s: 3 * k / 32 }
    : { s: 3 * k / 32 };
  return sampleField(fieldFn(kind, N, N, p), N, N);
}

function mount(host: HTMLElement): void {
  injectCss(NAME, CSS);
  const divcurl = host.dataset.mode === "divcurl";
  let kind: FieldKind = (KINDS.includes(host.dataset.field as FieldKind) ? host.dataset.field : "rotation") as FieldKind;
  let k = 0.6, step = 6, quiz = host.dataset.mode === "quiz", qi = 0, revealed = false;

  const views = htmlEl("div", "fv-views" + (divcurl ? " four" : ""), "", host);
  const vArrows = new ImgView(views, "arrows (length = speed)");
  const vColour = new ImgView(views, "colour wheel");
  const vSpeed = new ImgView(views, "speed");
  const vDiv = divcurl ? new ImgView(views, "divergence") : null;
  const vCurl = divcurl ? new ImgView(views, "curl (CCW red)") : null;
  if (divcurl) { vSpeed.fig.style.display = "none"; }

  const panel = htmlEl("div", "qb-panel", "", host);
  htmlEl("div", "fv-lab", "field", panel);
  const kindBtns = presets(panel, KINDS.map(kd => ({ label: FIELD_NAMES[kd], apply: () => { quiz = false; kind = kd; draw(); } })));
  const kS = slider(panel, "strength (sign flips)", -1, 1, 0.05, k, v => { k = v; draw(); });
  const stepS = slider(panel, "arrow spacing", 3, 12, 1, step, v => { step = v; draw(); });
  const wheel = flowWheel(panel, 70);
  htmlEl("div", "fv-lab", "quiz", panel);
  const qrow = htmlEl("div", "qb-presets", "", panel);
  button(qrow, "which field is this?", () => { quiz = true; revealed = false; qi = (qi + 1) % QUIZ_ORDER.length; draw(); });
  button(qrow, "show answer", () => { revealed = true; draw(); });
  const qtext = htmlEl("div", "fv-quiz", "", panel);
  button(panel, "reset", () => { quiz = host.dataset.mode === "quiz"; qi = 0; revealed = false; kind = "rotation"; k = 0.6; step = 6; kS.set(k); stepS.set(step); draw(); });
  const ro = htmlEl("div", "qb-readout", "", panel);

  function draw(): void {
    const [qk, qs] = QUIZ_ORDER[qi];
    const kd = quiz ? qk : kind, kk = quiz ? 0.6 * qs : k;
    const f = field(kd, kk), vmax = Math.max(0.1, maxSpeed(f));
    kindBtns.forEach((b, i) => b.classList.toggle("active", !quiz && KINDS[i] === kind));
    const hide = quiz && !revealed;
    vArrows.fig.style.visibility = hide ? "hidden" : "";
    vSpeed.fig.style.visibility = hide ? "hidden" : "";
    // arrows on a light background
    vArrows.rgba(new Uint8ClampedArray(4 * N * N).fill(245), N, N);
    vArrows.clear();
    drawQuiver(vArrows, f, step, 0.8 * step / vmax, "#1f4e9a", 1.6);
    vColour.rgba(flowToRGBA(f, vmax), N, N);
    const sp = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) sp[i] = Math.hypot(f.u[i], f.v[i]);
    vSpeed.map(sp, N, N, viridis, 0, vmax);
    vSpeed.setCaption(`speed (0 … ${fmtSpeed(vmax)} px/frame)`);
    wheel.setScale(vmax);
    if (vDiv && vCurl) {
      const d = divergence(f), c = curlScreen(f);
      let m = 1e-6; for (let i = 0; i < N * N; i++) m = Math.max(m, Math.abs(d[i]), Math.abs(c[i]));
      vDiv.map(Float32Array.from(d, v => divergingT(v, m)), N, N, diverging, 0, 1);
      vCurl.map(Float32Array.from(c, v => divergingT(v, m)), N, N, diverging, 0, 1);
      const i0 = (N / 2) * N + N / 2;
      ro.textContent = `at the centre: div ${d[i0].toFixed(3)}, curl ${c[i0].toFixed(3)} per frame\n(colour scale ±${m.toFixed(3)})`;
    } else ro.textContent = `largest speed ${fmtSpeed(vmax)} px/frame`;
    qtext.textContent = quiz ? (revealed ? `answer: ${FIELD_NAMES[qk]}${qs < 0 ? " (reversed)" : ""}` : `field ${qi + 1}: which one? (colour only)`) : "";
  }
  draw();
}

autoMount(NAME, mount);
