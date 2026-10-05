// UI for the CELTIC calculator. All arithmetic lives in model.js; this file only reads answers and paints the chart.
import { TIERS, ITEMS, predict, threshold, formatPercent } from "./model.js";

const $ = (id) => document.getElementById(id);
// Every finding is a plain yes or no; the label carries the threshold. (TYPED lists findings entered as a number; none now.)
const TYPED = {};
const ORDER = { bedside: ["asymmetry", "leuko", "age70", "como_malignancy_active", "como_diabetes"],
                rapid: ["asymmetry", "leuko", "age70", "como_malignancy_active", "como_diabetes", "crp_hi", "neut_hi", "como_chf", "como_ckd", "como_obesity"] };
const BAND_TEXT = {
  "rule-out": ["Cellulitis unlikely (rule-out)", "Look for the mimic before starting or continuing antibiotics: clot, stasis or contact dermatitis, gout."],
  "indeterminate": ["Indeterminate", "Neither band. Examine again, consider ultrasound for a clot, and reassess at 48 to 72 hours."],
  "rule-in": ["Cellulitis likely (rule-in)", "Consistent with cellulitis. Treat per your pathway and reassess if there is no response by 48 to 72 hours."],
};
const STATES = ["is-rule-in", "is-rule-out", "is-indeterminate", "is-empty"];

let tier = "bedside";
const raw = {};       // what was entered: 0/1 for yes/no rows, the typed string for laboratory rows, null when blank
const answers = {};   // 0/1 as the model sees it, null when unanswered or out of range

function rowHtml(key) {
  const it = ITEMS[key];
  const hint = it.hint ? `<span class="row-hint" id="hint-${key}">${it.hint}</span>` : "";
  if (TYPED[key]) {
    const t = it.threshold;
    return `<div class="row" id="row-${key}">
      <div class="row-text"><label class="row-label" for="in-${key}">${it.label}</label>${hint}</div>
      <span class="field"><input id="in-${key}" type="number" inputmode="decimal" step="any" min="${t.min}" max="${t.max}" aria-describedby="hint-${key} der-${key} msg-${key}" data-key="${key}"><span class="unit">${t.unit}</span></span>
      <p class="derived" id="der-${key}"></p>
      <p class="row-msg" id="msg-${key}" role="alert"></p>
    </div>`;
  }
  return `<div class="row" id="row-${key}">
    <div class="row-text"><span class="row-label" id="lab-${key}">${it.label}</span>${hint}</div>
    <span class="seg" role="group" aria-labelledby="lab-${key}"${it.hint ? ` aria-describedby="hint-${key}"` : ""}><button type="button" data-key="${key}" data-v="1" aria-pressed="false">Yes</button><button type="button" data-key="${key}" data-v="0" aria-pressed="false">No</button></span>
  </div>`;
}

function build() {
  const keys = ORDER[tier];
  const panel = $("panel");
  panel.innerHTML = keys.map(rowHtml).join("");
  panel.setAttribute("aria-labelledby", `tab-${tier}`);
  for (const k of keys) { if (!(k in raw)) { raw[k] = null; answers[k] = null; } }
  panel.querySelectorAll(".seg button").forEach((b) => {
    const k = b.dataset.key; const v = Number(b.dataset.v);
    if (raw[k] === v) b.setAttribute("aria-pressed", "true");
    b.addEventListener("click", () => {
      raw[k] = v; answers[k] = v;
      b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      paint();
    });
  });
  panel.querySelectorAll("input").forEach((inp) => {
    const k = inp.dataset.key;
    if (raw[k] !== null && raw[k] !== undefined) inp.value = raw[k];
    inp.addEventListener("input", () => { raw[k] = inp.value; applyThreshold(k); paint(); });
  });
  for (const k of keys) if (TYPED[k]) applyThreshold(k);
  const t = TIERS[tier];
  $("tier-auroc").textContent = `${t.name} tier AUROC ${t.auroc.toFixed(2)} (95% CI, ${t.aurocLo.toFixed(2)} to ${t.aurocHi.toFixed(2)}).`;
  paint();
}

function applyThreshold(k) {
  const t = ITEMS[k].threshold; const v = raw[k]; const row = $(`row-${k}`); const msg = $(`msg-${k}`); const der = $(`der-${k}`);
  row.classList.remove("has-error"); msg.textContent = ""; der.textContent = "";
  if (v === null || v === undefined || v === "") { answers[k] = null; return; }
  const num = Number(v);
  if (Number.isNaN(num)) { answers[k] = null; row.classList.add("has-error"); msg.textContent = "Enter a number."; return; }
  if (num < t.min || num > t.max) { answers[k] = null; row.classList.add("has-error"); msg.textContent = `Out of range. Enter ${t.min} to ${t.max} ${t.unit}.`; return; }
  const flag = threshold(k, num); answers[k] = flag;
  der.innerHTML = flag ? `Counts as <b>yes</b>, ${t.ge} ${t.unit} or more.` : `Counts as <b>no</b>, below ${t.ge} ${t.unit}.`;
}

function paint() {
  const keys = ORDER[tier];
  // A blank finding counts as no, which is how the study handled a missing value, so the score reads from the first answer.
  const blank = keys.filter((k) => answers[k] === null || answers[k] === undefined);
  const box = $("result");
  box.classList.remove(...STATES);
  if (blank.length === keys.length) {
    box.classList.add("is-empty");
    $("prob").innerHTML = "&nbsp;"; $("prob-unit").textContent = "";
    $("band").textContent = "Answer what you know. The score reads from the first answer.";
    $("band-note").textContent = "";
    $("incomplete").textContent = "";
    paintContrib(null);
    return;
  }
  const r = predict(tier, Object.fromEntries(keys.map((k) => [k, blank.includes(k) ? 0 : answers[k]])));
  $("incomplete").textContent = blank.length
    ? `Not answered, counted as no: ${blank.map((k) => ITEMS[k].short).join(", ")}. You can stop here or fill them in as results return.`
    : "";
  box.classList.add(`is-${r.band}`);
  $("prob").textContent = formatPercent(r.probability); $("prob-unit").textContent = "%";
  const [name, note] = BAND_TEXT[r.band];
  $("band").textContent = name; $("band-note").textContent = note;
  $("mark").style.transform = `translateX(${(r.probability * 100).toFixed(1)}%)`;
  paintContrib(r);
}

function paintContrib(r) {
  const list = $("contrib"); const absent = $("absent"); const hand = $("hand");
  if (!r) { list.innerHTML = ""; absent.textContent = ""; hand.textContent = ""; return; }
  const present = r.contributions.filter((c) => c.present);
  const maxAbs = Math.max(0.2, ...present.map((c) => Math.abs(c.contribution)));
  list.innerHTML = present.map((c) => {
    const pct = Math.min(100, Math.abs(c.contribution) / maxAbs * 50);
    const cls = c.contribution > 0 ? "pos" : c.contribution < 0 ? "neg" : "zero";
    const sign = c.contribution > 0 ? "+" : c.contribution < 0 ? "−" : "";
    return `<li><span class="c-label">${ITEMS[c.key].short} <span class="c-or">OR ${c.or.toFixed(2)}</span></span><span class="axis" aria-hidden="true"><span class="bar ${cls}" style="width:${cls === "zero" ? "2px" : pct + "%"}"></span></span><span class="c-val ${cls}">${sign}${Math.abs(c.contribution).toFixed(2)}</span></li>`;
  }).join("");
  requestAnimationFrame(() => list.querySelectorAll(".bar.pos, .bar.neg").forEach((b) => b.classList.add("grow")));
  const missing = r.contributions.filter((c) => !c.present && answers[c.key] === 0).map((c) => ITEMS[c.key].short);
  const base = formatPercent(r.baseline);
  absent.textContent = (missing.length ? `Answered no, so not counted: ${missing.join(", ")}. ` : "") + `Baseline with no findings ${base}%. Total log-odds ${r.logit >= 0 ? "+" : "−"}${Math.abs(r.logit).toFixed(3)}.`;
  if (tier === "bedside") {
    const s = r.points;
    hand.innerHTML = `Hand score <b>${s > 0 ? "+" : s < 0 ? "−" : ""}${Math.abs(s)}</b>. Unilateral +3, diabetes +1, leukocytosis 0, age 70 or older −1, malignancy −1. Tally it on paper when there is no phone. The probability above is the exact logistic value.`;
  } else {
    hand.textContent = "";
  }
}

// Tier tabs. Standard and Comprehensive are printed on the strip but disabled, so only the enterable tiers take part.
const tabs = [...document.querySelectorAll('.tiers [role="tab"]:not([aria-disabled="true"])')];
function selectTier(tab, moveFocus) {
  tier = tab.dataset.tier;
  tabs.forEach((x) => { const on = x === tab; x.setAttribute("aria-selected", on ? "true" : "false"); x.tabIndex = on ? 0 : -1; });
  if (moveFocus) tab.focus();
  build();
}
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => selectTier(tab, false));
  tab.addEventListener("keydown", (e) => {
    let next = null;
    if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
    else if (e.key === "ArrowLeft") next = tabs[(i + tabs.length - 1) % tabs.length];
    else if (e.key === "Home") next = tabs[0];
    else if (e.key === "End") next = tabs[tabs.length - 1];
    if (next) { e.preventDefault(); selectTier(next, true); }
  });
});

$("clear").addEventListener("click", () => {
  for (const k of Object.keys(raw)) { raw[k] = null; answers[k] = null; }
  build();
});

$("copy-cite").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("citation").textContent.trim()); $("copied").textContent = "Copied."; }
  catch { $("copied").textContent = "Select the text to copy it."; }
  setTimeout(() => { $("copied").textContent = ""; }, 2500);
});

build();
