// Nodule analyser page: upload a scan, browse slices, pick a point, show the ensemble's estimate.
(function () {
  const $ = (id) => document.getElementById(id);
  const S = { scan: null, k: 0, win: "lung", point: null, cache: new Map(), history: [], view: "views", last: null, cands: [], detector: false };
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  document.querySelectorAll(".warnIcon").forEach((el) => (el.innerHTML = CADC.warnIcon));

  // The tool stays locked until the visitor accepts the AI disclaimer.
  $("locked").classList.toggle("on", !CADC.consented);
  CADC.whenConsented(() => $("locked").classList.remove("on"));

  fetch("/api/status").then((r) => r.json()).then((st) => {
    S.detector = st.detector;
    if (st.models) $("stepAnalyse").textContent = `An ensemble of ${st.models} neural networks (${st.runs.join(" + ")}) studies a 48 mm cube around the point.`;
  }).catch(() => {});

  function loading(on, text) { $("loading").classList.toggle("on", on); if (text) $("loadingText").textContent = text; }
  async function api(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { /* keep status text */ } throw new Error(m); }
    return r.json();
  }
  const fmtPct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : Math.round(p * 100) + "%");
  const colorFor = (p) => (p >= 0.7 ? "var(--malignant)" : p >= 0.3 ? "var(--warn)" : "var(--benign)");
  const bandOf = (p) => (p >= 0.7 ? "high" : p >= 0.3 ? "intermediate" : "low");
  function setStep(n) { [...$("steps").children].forEach((li, i) => { li.className = i < n ? "done" : i === n ? "active" : ""; }); }

  // ---------- upload ----------
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => upload(ev.dataTransfer.files));
  $("fileInput").addEventListener("change", (ev) => upload(ev.target.files));
  $("demoBtn").addEventListener("click", async (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    loading(true, "Fetching the sample scan (LIDC-IDRI-0001)… the first time takes about 30 s");
    try { openScan(await api("/api/demo", { method: "POST" })); } catch (e) { CADC.toast(e.message); } finally { loading(false); }
  });
  async function upload(files) {
    if (!files || !files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    loading(true, `Reading ${files.length} file${files.length > 1 ? "s" : ""}…`);
    try { openScan(await api("/api/upload", { method: "POST", body: fd })); }
    catch (e) { CADC.toast(e.message); } finally { loading(false); $("fileInput").value = ""; }
  }

  function openScan(info) {
    S.scan = info; S.cache.clear(); S.point = null; S.history = []; S.cands = [];
    $("detectWrap").style.display = S.detector ? "block" : "none";
    $("cands").innerHTML = ""; $("detectBtn").disabled = false;
    $("detectHint").textContent = "The AI searches the whole scan (about 30 seconds)";
    const [rows, cols, n] = info.shape;
    canvas.width = cols; canvas.height = rows;
    $("slider").max = n - 1;
    $("uploader").style.display = "none"; $("viewer").classList.add("on"); $("viewerTools").style.display = "flex";
    $("leftTitle").textContent = info.lidc ? `CT scan · ${info.lidc.patient_id}` : "CT scan";
    $("hudR").innerHTML = `${cols}×${rows}×${n}<br>${info.spacing_mm[0]}×${info.spacing_mm[1]}×${info.spacing_mm[2]} mm`;
    $("result").classList.remove("on"); renderHistory();
    $("analyzeBtn").disabled = true; $("pointLabel").textContent = "";
    const known = info.lidc ? info.lidc.nodules : [];
    $("knownWrap").style.display = known.length ? "block" : "none";
    $("known").innerHTML = "";
    known.forEach((nd, i) => {
      const b = document.createElement("button"); b.className = "chip";
      const mean = nd.ratings.length ? (nd.ratings.reduce((a, c) => a + c, 0) / nd.ratings.length).toFixed(1) : "–";
      b.innerHTML = `Nodule ${i + 1} · <b>${nd.diameter_mm} mm</b> · rated ${mean}/5`;
      b.onclick = () => { setPoint(nd.x, nd.y, nd.slice); analyze(); };
      $("known").appendChild(b);
    });
    showSlice(known.length ? known[0].slice : Math.floor(n / 2));
    setStep(1);
  }
  $("newScan").onclick = () => {
    S.scan = null; $("viewer").classList.remove("on"); $("uploader").style.display = "block";
    $("viewerTools").style.display = "none"; $("result").classList.remove("on"); $("histCard").style.display = "none";
    $("analyzeBtn").disabled = true; $("pointLabel").textContent = ""; $("leftTitle").textContent = "CT scan"; setStep(0);
  };

  // ---------- viewer ----------
  function sliceImage(k) {
    const key = `${S.win}:${k}`;
    if (!S.cache.has(key)) {
      const img = new Image();
      img.src = `/api/scan/${S.scan.scan_id}/slice/${k}?window=${S.win}`;
      S.cache.set(key, new Promise((res, rej) => { img.onload = () => res(img); img.onerror = rej; }));
    }
    return S.cache.get(key);
  }
  async function showSlice(k) {
    const n = S.scan.shape[2];
    k = Math.max(0, Math.min(n - 1, Math.round(k)));
    S.k = k; $("slider").value = k; $("sliceLabel").textContent = `${k + 1} / ${n}`;
    $("hudL").textContent = `Slice ${k + 1}`;
    try {
      const img = await sliceImage(k);
      if (S.k !== k) return;
      draw(img);
    } catch { CADC.toast("Could not load that slice. The scan may have expired; please upload it again."); }
    [k + 1, k - 1, k + 2, k - 2].forEach((j) => j >= 0 && j < n && sliceImage(j));
  }
  function draw(img) {
    ctx.drawImage(img, 0, 0);
    // detected candidates near this slice: dashed rings coloured by suspicion, numbered
    S.cands.forEach((c, i) => {
      const dz = Math.abs(c.slice - S.k) * S.scan.spacing_mm[2];
      if (dz > Math.max(4, c.diameter_mm / 2)) return;
      const r = Math.max(8, (c.diameter_mm / 2 + 4) / S.scan.spacing_mm[0]);
      const col = c.malignancy === undefined ? "#4cc2ff" : c.malignancy >= 0.7 ? "#f87171" : c.malignancy >= 0.3 ? "#fbbf24" : "#34d399";
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.5, canvas.width / 340); ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = col; ctx.font = `bold ${Math.round(canvas.width / 38)}px Inter, sans-serif`;
      ctx.fillText(String(i + 1), c.x + r * 0.75, c.y - r * 0.75); ctx.restore();
    });
    const p = S.point;
    if (!p) return;
    const r = 22 / S.scan.spacing_mm[0];
    const onSlice = Math.abs(p.slice - S.k) < 0.5;
    ctx.save();
    ctx.strokeStyle = onSlice ? "#4cc2ff" : "rgba(76,194,255,.35)";
    ctx.lineWidth = Math.max(1.5, canvas.width / 300);
    ctx.setLineDash(onSlice ? [] : [6, 6]);
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      ctx.moveTo(p.x + dx * r * 0.4, p.y + dy * r * 0.4); ctx.lineTo(p.x + dx * r * 1.5, p.y + dy * r * 1.5);
    }
    ctx.stroke(); ctx.restore();
  }
  function redraw() { sliceImage(S.k).then(draw).catch(() => {}); }
  function setPoint(x, y, slice) {
    S.point = { x, y, slice };
    $("analyzeBtn").disabled = false;
    $("pointLabel").textContent = `x ${Math.round(x)} · y ${Math.round(y)} · slice ${slice + 1}`;
    setStep(2);
    if (slice !== S.k) showSlice(slice); else redraw();
  }
  $("slider").addEventListener("input", (e) => showSlice(+e.target.value));
  $("stagebox").addEventListener("wheel", (e) => { if (!S.scan) return; e.preventDefault(); showSlice(S.k + Math.sign(e.deltaY)); }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (!S.scan || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowUp") { e.preventDefault(); showSlice(S.k + 1); }
    if (e.key === "ArrowDown") { e.preventDefault(); showSlice(S.k - 1); }
    if (e.key === "Enter" && S.point) analyze();
  });
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    setPoint((e.clientX - rect.left) * canvas.width / rect.width, (e.clientY - rect.top) * canvas.height / rect.height, S.k);
  });
  $("windowSeg").addEventListener("click", (e) => {
    const w = e.target.dataset.w; if (!w) return;
    S.win = w; [...$("windowSeg").children].forEach((b) => b.classList.toggle("on", b.dataset.w === w)); showSlice(S.k);
  });

  // ---------- automatic detection ----------
  $("detectBtn").addEventListener("click", async () => {
    if (!S.scan || !CADC.consented) return;
    $("detectBtn").disabled = true;
    loading(true, "Searching the whole scan for nodules… about 30 seconds");
    try {
      const r = await api(`/api/scan/${S.scan.scan_id}/detect`, { method: "POST" });
      S.cands = r.candidates;
      $("detectHint").textContent = S.cands.length
        ? `${S.cands.length} candidate${S.cands.length > 1 ? "s" : ""} found. Click one to analyse it in detail. Rings on the slices show where they are.`
        : "No nodules found above the detector's threshold. You can still click on a spot to analyse it.";
      $("cands").innerHTML = "";
      S.cands.forEach((c, i) => {
        const b = document.createElement("button"); b.className = "chip";
        const m = c.malignancy;
        const sc = m === undefined ? "" : ` · <span class="sc" style="color:${colorFor(m)}">${fmtPct(m)}</span> suspicion`;
        b.innerHTML = `#${i + 1} · <b>${c.diameter_mm.toFixed(0)} mm</b> · slice ${c.slice + 1}${sc}`;
        b.title = `Detector confidence ${(c.score * 100).toFixed(0)}%`;
        b.onclick = () => { setPoint(c.x, c.y, c.slice); analyze(); };
        $("cands").appendChild(b);
      });
      if (S.cands.length) showSlice(S.cands[0].slice); else redraw();
    } catch (e) { CADC.toast(e.message); $("detectBtn").disabled = false; } finally { loading(false); }
  });

  // ---------- analyse ----------
  $("analyzeBtn").onclick = analyze;
  async function analyze() {
    if (!S.point || !CADC.consented) return;
    const p = S.point;
    $("analyzeBtn").disabled = true;
    loading(true, "Running 5 models and computing the attention map…");
    try {
      const r = await api(`/api/scan/${S.scan.scan_id}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: p.x, y: p.y, slice: p.slice }),
      });
      r.point = { ...p };
      showResult(r);
      S.history.unshift(r); renderHistory();
      setStep(3);
      if (window.innerWidth < 1000) $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { CADC.toast(e.message); } finally { loading(false); $("analyzeBtn").disabled = false; }
  }

  const BAND_TEXT = {
    low: "The models see few of the features radiologists associate with cancer. This does <b>not</b> rule cancer out.",
    intermediate: "The models see some suspicious features. A doctor would weigh this against size, growth and your history.",
    high: "The models see features that radiologists often rate as suspicious. Many such nodules still turn out to be benign; only a doctor and further tests can tell.",
  };
  function showResult(r) {
    S.last = r;
    const el = $("result"); el.classList.remove("on"); void el.offsetWidth; el.classList.add("on");
    const c = colorFor(r.probability);
    $("probText").textContent = fmtPct(r.probability); $("probText").style.color = c;
    const arc = $("arc"); arc.style.stroke = c; arc.style.strokeDashoffset = 389.56;
    requestAnimationFrame(() => requestAnimationFrame(() => { arc.style.strokeDashoffset = 389.56 * (1 - r.probability); }));
    const band = bandOf(r.probability);
    const mal = r.probability >= 0.5;
    $("verdict").textContent = `${band[0].toUpperCase() + band.slice(1)} suspicion`;
    $("verdict").className = "verdict " + { low: "benign", intermediate: "warn", high: "malignant" }[band];
    const agree = r.per_model.filter((v) => (v >= 0.5) === mal).length;
    $("verdictText").innerHTML = `${agree} of ${r.per_model.length} models agree. ${BAND_TEXT[band]}`;
    $("bandMark").style.left = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => { $("bandMark").style.left = (r.probability * 100).toFixed(1) + "%"; }));
    $("resultAt").textContent = `x ${Math.round(r.point.x)} · y ${Math.round(r.point.y)} · slice ${r.point.slice + 1}`;
    $("bars").innerHTML = r.per_model.map((v, i) => `
      <div class="bar"><span class="muted">${(r.model_names && r.model_names[i]) || "Model " + (i + 1)}</span>
        <div class="track"><div class="fill" style="background:${colorFor(v)}" data-w="${(v * 100).toFixed(1)}"></div></div>
        <span class="mono" style="text-align:right">${fmtPct(v)}</span></div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll("#bars .fill").forEach((f) => (f.style.width = f.dataset.w + "%"))));
    $("viewSeg").style.display = r.has_heatmap === false ? "none" : "";
    if (r.has_heatmap === false) S.view = "views";
    renderViews();
    $("trainNote").style.display = r.in_training_data ? "flex" : "none";
  }
  function renderViews() {
    const r = S.last; if (!r) return;
    const src = r[S.view];
    $("vAxial").src = "data:image/png;base64," + src.axial;
    $("vCoronal").src = "data:image/png;base64," + src.coronal;
    $("vSagittal").src = "data:image/png;base64," + src.sagittal;
    $("heatNote").style.display = S.view === "heatmaps" ? "block" : "none";
  }
  $("viewSeg").addEventListener("click", (e) => {
    const v = e.target.dataset.v; if (!v) return;
    S.view = v; [...$("viewSeg").children].forEach((b) => b.classList.toggle("on", b.dataset.v === v)); renderViews();
  });
  function renderHistory() {
    $("histCard").style.display = S.history.length ? "block" : "none";
    $("history").innerHTML = "";
    S.history.forEach((h) => {
      const d = document.createElement("div"); d.className = "hist";
      const c = colorFor(h.probability);
      d.innerHTML = `<span class="mono small">x ${Math.round(h.point.x)} · y ${Math.round(h.point.y)} · slice ${h.point.slice + 1}</span>
        <span class="hpill" style="color:${c};background:color-mix(in srgb, ${c} 14%, transparent)">${fmtPct(h.probability)}</span>`;
      d.onclick = () => { S.point = { ...h.point }; showSlice(h.point.slice); redraw(); showResult(h); };
      $("history").appendChild(d);
    });
  }

  // ---------- printable report ----------
  $("reportBtn").onclick = () => {
    const r = S.last; if (!r) return;
    const w = window.open("", "_blank");
    if (!w) { CADC.toast("Allow pop-ups for this page to open the report."); return; }
    const when = new Date().toLocaleString();
    const band = bandOf(r.probability);
    const img = (b64, cap) => `<figure><img src="data:image/png;base64,${b64}"><figcaption>${cap}</figcaption></figure>`;
    w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>CADC AI nodule estimate</title>
      <style>
        body{font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#111;max-width:760px;margin:32px auto;padding:0 20px}
        h1{font-size:22px;margin:0} .sub{color:#555;margin:4px 0 18px} .warn{border:2px solid #b45309;background:#fff7e6;padding:12px 14px;border-radius:8px;margin:16px 0}
        table{border-collapse:collapse;width:100%;margin:10px 0} td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}
        .imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:10px} figure{margin:0} img{width:100%;border-radius:6px} figcaption{font-size:12px;color:#555;text-align:center}
        .big{font-size:30px;font-weight:700} footer{margin-top:24px;font-size:12px;color:#555}
        @media print{button{display:none}}
      </style></head><body>
      <h1>AI nodule estimate: research report</h1>
      <div class="sub">Generated ${when} by CADC (research prototype)${S.scan.lidc ? " · LIDC-IDRI case " + S.scan.lidc.patient_id : ""}</div>
      <div class="warn"><b>Not a medical diagnosis.</b> This report was produced by an artificial-intelligence research model that estimates how suspicious a
        lung nodule looks, based on radiologists' ratings in a research dataset. It has not been clinically validated, can be wrong, and must not be used to make
        medical decisions. Please discuss any scan or symptom with a qualified doctor.</div>
      <table>
        <tr><th>Malignancy likelihood (ensemble)</th><td class="big">${fmtPct(r.probability)}</td></tr>
        <tr><th>Suspicion band</th><td>${band} (low &lt; 30% ≤ intermediate &lt; 70% ≤ high)</td></tr>
        <tr><th>Individual models</th><td>${r.per_model.map(fmtPct).join(" · ")}</td></tr>
        <tr><th>Selected point</th><td>column ${Math.round(r.point.x)}, row ${Math.round(r.point.y)}, slice ${r.point.slice + 1} of ${S.scan.shape[2]}</td></tr>
        <tr><th>Scan</th><td>${S.scan.shape[1]}×${S.scan.shape[0]}×${S.scan.shape[2]} voxels, ${S.scan.spacing_mm.join(" × ")} mm</td></tr>
      </table>
      ${r.in_training_data ? "<p><i>This scan is part of the model's training data, so the estimate is optimistic.</i></p>" : ""}
      <h3>Region analysed (48 mm around the point)</h3>
      <div class="imgs">${img(r.views.axial, "Axial")}${img(r.views.coronal, "Coronal")}${img(r.views.sagittal, "Sagittal")}</div>
      <h3>AI attention (Grad-CAM)</h3>
      <div class="imgs">${img(r.heatmaps.axial, "Axial")}${img(r.heatmaps.coronal, "Coronal")}${img(r.heatmaps.sagittal, "Sagittal")}</div>
      <footer>Model: ensemble of ${r.per_model.length} neural networks trained on LIDC-IDRI (1,627 nodules), cross-validated against radiologist ratings.
        Labels are radiologists' opinions, not biopsy results.</footer>
      <p><button onclick="print()">Print / save as PDF</button></p>
      <script>setTimeout(()=>print(),400)<\/script>
      </body></html>`);
    w.document.close();
  };
})();
