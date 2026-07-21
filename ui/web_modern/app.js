/**
 * ══════════════════════════════════════════════════════════════════════════════
 *  app.js — Gradio Client Integration for Notebook UI
 *  AI Multimodal Smart Knowledge Assistant
 * ══════════════════════════════════════════════════════════════════════════════
 */

import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client";

// ── Config ──────────────────────────────────────────────────────────────────
// Same-origin: notebook UI is served at /ui, Gradio API at /gradio
const GRADIO_URL = window.location.origin + "/gradio";

// ── State ───────────────────────────────────────────────────────────────────
let client = null;
let mediaRecorder = null;
let audioChunks = [];
let recordedFile = null;   // File object for recorded audio
let captionFile = null;    // File object for uploaded image

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Resolve a Gradio file response into a usable URL */
function fileUrl(res) {
  if (!res) return "";
  if (typeof res === "string") {
    if (res.startsWith("http") || res.startsWith("data:")) return res;
    return `${GRADIO_URL}/file=${res}`;
  }
  if (res.url) return res.url.startsWith("http") ? res.url : `${GRADIO_URL}${res.url}`;
  if (res.path) return `${GRADIO_URL}/file=${res.path}`;
  return "";
}

/** Show / hide helper */
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

/** Set connection badge */
function setStatus(ok, msg) {
  const dot  = document.getElementById("conn-dot");
  const text = document.getElementById("conn-text");
  const fDot = document.getElementById("footer-dot");

  dot.className  = `status-dot status-dot--${ok ? "ok" : ok === false ? "error" : "loading"}`;
  fDot.className = `status-dot status-dot--${ok ? "ok" : ok === false ? "error" : "loading"}`;
  text.textContent = msg || (ok ? `Connected @ ${GRADIO_URL}` : `Disconnected`);
}

// ── Gradio Client ───────────────────────────────────────────────────────────

async function connect() {
  setStatus(null, "Connecting…");
  try {
    client = await Client.connect(GRADIO_URL);
    setStatus(true, `Connected @ ${GRADIO_URL}`);
    console.log("[app] Gradio client ready");
  } catch (e) {
    console.warn("[app] Client.connect failed, trying REST ping…", e);
    try {
      const r = await fetch(`${GRADIO_URL}/info`);
      if (r.ok) { setStatus(true, `REST connected @ ${GRADIO_URL}`); return; }
    } catch (_) { /* fall through */ }
    setStatus(false, `Cannot reach ${GRADIO_URL}`);
  }
}

// ── Tabs ────────────────────────────────────────────────────────────────────

function initTabs() {
  const btns   = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".panel");

  btns.forEach(btn => btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    btns.forEach(b => b.classList.toggle("active", b === btn));
    panels.forEach(p => p.id === target ? show(p) : hide(p));
  }));
}

// ── Voice Recorder ──────────────────────────────────────────────────────────

function initMic() {
  const btn     = document.getElementById("mic-btn");
  const ico     = document.getElementById("mic-ico");
  const label   = document.getElementById("mic-label");
  const status  = document.getElementById("mic-status");
  const preview = document.getElementById("mic-preview");
  let recording = false;

  btn.addEventListener("click", async () => {
    if (!recording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => { if (e.data.size) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
          const blob = new Blob(audioChunks, { type: "audio/wav" });
          recordedFile = new File([blob], "voice.wav", { type: "audio/wav" });
          preview.src = URL.createObjectURL(blob);
          show(preview);
          status.textContent = "✅ Recorded";
          status.style.color = "var(--green)";
        };

        mediaRecorder.start();
        recording = true;
        btn.classList.add("btn--danger");
        btn.classList.remove("btn--blue");
        ico.setAttribute("icon", "lucide:square");
        label.textContent = "Stop";
        status.textContent = "● Recording…";
        status.style.color = "var(--red)";
      } catch (err) {
        alert("Microphone access denied or unavailable.");
      }
    } else {
      mediaRecorder?.stop();
      mediaRecorder?.stream.getTracks().forEach(t => t.stop());
      recording = false;
      btn.classList.remove("btn--danger");
      btn.classList.add("btn--blue");
      ico.setAttribute("icon", "lucide:mic");
      label.textContent = "Click to Record";
    }
  });
}

// ── Dropzone ────────────────────────────────────────────────────────────────

function initDropzone() {
  const zone    = document.getElementById("cap-dropzone");
  const input   = document.getElementById("cap-file");
  const prompt  = document.getElementById("cap-prompt");
  const preview = document.getElementById("cap-preview");

  const handleFile = file => {
    captionFile = file;
    preview.src = URL.createObjectURL(file);
    show(preview);
    hide(prompt);
  };

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

// ── Panel 1 : Ask ───────────────────────────────────────────────────────────

function initAsk() {
  const btn      = document.getElementById("ask-btn");
  const qEl      = document.getElementById("ask-q");
  const domainEl = document.getElementById("ask-domain");
  const ansEl    = document.getElementById("ask-answer");
  const resultEl = document.getElementById("ask-result");
  const audioW   = document.getElementById("ask-audio-wrap");
  const audioEl  = document.getElementById("ask-audio");
  const provEl   = document.getElementById("ask-provider");
  const srcEl    = document.getElementById("ask-sources");

  btn.addEventListener("click", async () => {
    const q = qEl.value.trim();
    if (!q && !recordedFile) { alert("Type a question or record audio first."); return; }

    btn.disabled = true;
    resultEl.classList.add("thinking");
    ansEl.textContent = "Thinking…";
    hide(audioW);

    try {
      const res = await client.predict("/handle_ask", [
        q || "",
        recordedFile || null,
        domainEl.value
      ]);
      const [retQ, answer, audioPath, sources, provider] = res.data;

      if (retQ) qEl.value = retQ;
      ansEl.textContent = answer || "No answer returned.";
      provEl.textContent = provider || "—";
      srcEl.textContent  = sources  || "None";

      if (audioPath) {
        audioEl.src = fileUrl(audioPath);
        show(audioW);
      }
    } catch (err) {
      ansEl.textContent = `Error: ${err.message || err}`;
      provEl.textContent = "Error";
    } finally {
      resultEl.classList.remove("thinking");
      btn.disabled = false;
    }
  });
}

// ── Panel 2 : Caption ───────────────────────────────────────────────────────

function initCaption() {
  const btn       = document.getElementById("cap-btn");
  const outEl     = document.getElementById("cap-out");
  const provEl    = document.getElementById("cap-provider");
  const latEl     = document.getElementById("cap-latency");
  const errEl     = document.getElementById("cap-err");
  const fuBtn     = document.getElementById("cap-followup-btn");
  const fuOut     = document.getElementById("cap-followup-out");
  const fuSrc     = document.getElementById("cap-followup-src");

  btn.addEventListener("click", async () => {
    if (!captionFile) { alert("Upload an image first."); return; }
    const style = document.querySelector('input[name="cap-style"]:checked').value;

    btn.disabled = true;
    outEl.textContent = "Analysing image…";
    hide(errEl);

    try {
      const res = await client.predict("/handle_caption", [captionFile, style]);
      const [caption, provider, latency, error] = res.data;

      if (error) {
        outEl.textContent = "Caption failed.";
        errEl.textContent = error;
        show(errEl);
      } else {
        outEl.textContent  = caption;
        provEl.textContent = provider || "—";
        latEl.textContent  = latency  || "—";
      }
    } catch (err) {
      outEl.textContent = "Error";
      errEl.textContent = err.message || String(err);
      show(errEl);
    } finally { btn.disabled = false; }
  });

  // Follow-up RAG
  fuBtn.addEventListener("click", async () => {
    const caption = outEl.textContent.trim();
    if (!caption || caption === "Generated caption will appear here…") {
      alert("Generate a caption first.");
      return;
    }
    fuBtn.disabled = true;
    show(fuOut);
    fuOut.textContent = "Querying knowledge base…";

    try {
      const res = await client.predict("/handle_caption_followup", [caption]);
      const [answer, sources] = res.data;
      fuOut.textContent = answer;
      fuSrc.textContent = `Sources: ${sources}`;
      show(fuSrc);
    } catch (err) {
      fuOut.textContent = `Error: ${err.message}`;
    } finally { fuBtn.disabled = false; }
  });
}

// ── Panel 3 : Generate ──────────────────────────────────────────────────────

function initGen() {
  const btn      = document.getElementById("gen-btn");
  const promptEl = document.getElementById("gen-prompt");
  const imgEl    = document.getElementById("gen-img");
  const phEl     = document.getElementById("gen-placeholder");
  const provEl   = document.getElementById("gen-provider");
  const errEl    = document.getElementById("gen-err");

  btn.addEventListener("click", async () => {
    const prompt = promptEl.value.trim();
    if (!prompt) { alert("Enter an image prompt."); return; }

    btn.disabled = true;
    show(phEl);
    hide(imgEl);
    hide(errEl);
    phEl.querySelector("p").textContent = "Generating…";

    try {
      const res = await client.predict("/handle_image_gen", [prompt]);
      const [imgRes, provider, error] = res.data;

      if (error) {
        errEl.textContent = error;
        show(errEl);
        phEl.querySelector("p").textContent = "Generation failed";
      } else {
        const url = fileUrl(imgRes);
        if (url) {
          imgEl.src = url;
          show(imgEl);
          hide(phEl);
        }
        provEl.textContent = provider || "pollinations";
      }
    } catch (err) {
      errEl.textContent = err.message || String(err);
      show(errEl);
    } finally { btn.disabled = false; }
  });
}

// ── Boot ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initMic();
  initDropzone();
  initAsk();
  initCaption();
  initGen();
  connect();
});
