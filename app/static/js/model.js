// Model page: fills the metrics and draws the ROC and learning-curve charts from /api/metrics.
(function () {
  const $ = (id) => document.getElementById(id);
  const INK = { grid: "rgba(148,178,214,.10)", axis: "rgba(148,178,214,.28)", muted: "#7f90a6", text: "#e8eef6", context: "rgba(148,178,214,.28)" };
  const SERIES = ["#3987e5", "#d95926"]; // validated categorical slots 1-2 (dark), checked against the page surface

  // A small SVG line chart with a crosshair tooltip. series: [{name, color, pts: [[x, y, extra]]}]
  function lineChart(el, cfg) {
    const W = Math.max(300, el.clientWidth), H = cfg.height || 300;
    const m = { t: 16, r: cfg.directLabels ? 92 : 20, b: 44, l: 52 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const [x0, x1] = cfg.x, [y0, y1] = cfg.y;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * iw, Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
    const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    const ticks = (a, b, n) => Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${cfg.label}" style="display:block;overflow:visible">`;
    for (const t of cfg.yTickValues || ticks(y0, y1, 4)) s += `<line x1="${m.l}" x2="${m.l + iw}" y1="${Y(t)}" y2="${Y(t)}" stroke="${INK.grid}"/>
      <text x="${m.l - 10}" y="${Y(t) + 4}" fill="${INK.muted}" font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">${cfg.fy(t)}</text>`;
    for (const t of cfg.xTickValues || ticks(x0, x1, cfg.xTicks || 5)) s += `<text x="${X(t)}" y="${m.t + ih + 20}" fill="${INK.muted}" font-size="12" text-anchor="middle" style="font-variant-numeric:tabular-nums">${cfg.fx(t)}</text>`;
    s += `<line x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}" stroke="${INK.axis}"/>`;
    s += `<text x="${m.l + iw / 2}" y="${H - 6}" fill="${INK.muted}" font-size="12.5" text-anchor="middle">${cfg.xLabel}</text>`;
    s += `<text transform="translate(14 ${m.t + ih / 2}) rotate(-90)" fill="${INK.muted}" font-size="12.5" text-anchor="middle">${cfg.yLabel}</text>`;
    if (cfg.diagonal) s += `<line x1="${X(x0)}" y1="${Y(y0)}" x2="${X(x1)}" y2="${Y(y1)}" stroke="${INK.axis}" stroke-dasharray="4 5"/>
      <text x="${X(0.62)}" y="${Y(0.56)}" fill="${INK.muted}" font-size="12" transform="rotate(${(-Math.atan2(ih, iw) * 180) / Math.PI} ${X(0.62)} ${Y(0.56)})">chance</text>`;
    (cfg.context || []).forEach((pts) => (s += `<path d="${path(pts)}" fill="none" stroke="${INK.context}" stroke-width="1"/>`));
    cfg.series.forEach((se) => {
      s += `<path d="${path(se.pts)}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
      if (cfg.directLabels) {
        const lp = se.labelAt || se.pts[se.pts.length - 1];
        s += `<text x="${X(lp[0]) + 8}" y="${Y(lp[1]) + (se.labelDy || 4)}" fill="${INK.text}" font-size="12.5" font-weight="600">${se.name}</text>`;
      }
    });
    s += `<g class="hover" style="display:none"><line class="xh" y1="${m.t}" y2="${m.t + ih}" stroke="${INK.axis}"/>
      ${cfg.series.map((se) => `<circle r="5" fill="${se.color}" stroke="#111824" stroke-width="2"/>`).join("")}</g>
      <rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="hit"/></svg>`;
    el.innerHTML = s;
    el.style.position = "relative";
    const tip = document.createElement("div");
    tip.className = "chart-tip"; el.appendChild(tip);
    const svg = el.querySelector("svg"), g = el.querySelector(".hover"), hit = el.querySelector(".hit");
    const nearest = (pts, xv) => pts.reduce((b, p) => (Math.abs(p[0] - xv) < Math.abs(b[0] - xv) ? p : b), pts[0]);
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(), sx = ((e.clientX - r.left) / r.width) * W;
      const xv = x0 + ((sx - m.l) / iw) * (x1 - x0);
      const hits = cfg.series.map((se) => nearest(se.pts, xv));
      g.style.display = "";
      g.querySelector(".xh").setAttribute("x1", X(hits[0][0])); g.querySelector(".xh").setAttribute("x2", X(hits[0][0]));
      g.querySelectorAll("circle").forEach((c, i) => { c.setAttribute("cx", X(hits[i][0])); c.setAttribute("cy", Y(hits[i][1])); });
      tip.innerHTML = cfg.tip(hits, cfg.series);
      tip.style.display = "block";
      const px = (X(hits[0][0]) / W) * r.width, left = px + 16 + tip.offsetWidth > r.width ? px - tip.offsetWidth - 16 : px + 16;
      tip.style.left = left + "px"; tip.style.top = "8px";
    });
    hit.addEventListener("mouseleave", () => { g.style.display = "none"; tip.style.display = "none"; });
  }

  const pct = (v) => (v * 100).toFixed(1) + "%";
  fetch("/api/metrics").then((r) => (r.ok ? r.json() : Promise.reject(r))).then((M) => {
    const n = M.nodule_level, p = M.patient_level, d = M.dataset;
    $("mAuc").textContent = n.auc.toFixed(3);
    $("mAucCi").textContent = `95% CI ${n.auc_95ci[0].toFixed(3)}–${n.auc_95ci[1].toFixed(3)}`;
    $("mAcc").textContent = pct(n.accuracy);
    $("mSens").textContent = pct(n.sensitivity);
    $("mSpec").textContent = pct(n.specificity);
    $("mPAuc").textContent = p.auc.toFixed(3);
    $("dsNodules").textContent = d.nodules.toLocaleString("en");
    $("dsMal").textContent = d.malignant.toLocaleString("en");
    $("dsPatients").textContent = d.patients.toLocaleString("en");
    $("dsScans").textContent = d.scans.toLocaleString("en");

    // ROC
    const roc = { nodule: M.roc.nodule, patient: M.roc.patient };
    const drawRoc = () => lineChart($("rocChart"), {
      label: "ROC curves for nodule-level and patient-level predictions", height: 340, x: [0, 1], y: [0, 1], yTickValues: [0, 0.2, 0.4, 0.6, 0.8, 1], diagonal: true, directLabels: true,
      fx: (v) => v.toFixed(1), fy: (v) => v.toFixed(1), xLabel: "False positive rate (benign flagged as malignant)", yLabel: "True positive rate",
      series: [
        { name: `Nodule ${n.auc.toFixed(3)}`, color: SERIES[0], pts: roc.nodule, labelAt: [0.55, 0.9], labelDy: 18 },
        { name: `Patient ${p.auc.toFixed(3)}`, color: SERIES[1], pts: roc.patient, labelAt: [0.55, 0.97], labelDy: -8 },
      ],
      tip: (h, se) => `<b>False positive rate ${pct(h[0][0])}</b>` + h.map((pt, i) =>
        `<div><span class="sw" style="background:${se[i].color}"></span>${i ? "Patient" : "Nodule"}: detects ${pct(pt[1])} at threshold ${pt[2].toFixed(2)}</div>`).join(""),
    });

    // learning curves
    const H = M.history, ep = H.epochs;
    const zip = (ys) => ep.map((e, i) => [e, ys[i]]);
    const epTicks = [1, ...ep.filter((e) => e % 10 === 0)];
    const drawAuc = () => lineChart($("aucChart"), {
      label: "Validation AUC by epoch", height: 260, x: [1, ep.length], y: [0.5, 1], xTickValues: epTicks, yTickValues: [0.5, 0.6, 0.7, 0.8, 0.9, 1],
      fx: (v) => Math.round(v), fy: (v) => v.toFixed(1), xLabel: "Epoch", yLabel: "AUC on unseen patients",
      context: H.auc_folds.map(zip), series: [{ name: "Mean", color: SERIES[0], pts: zip(H.auc_mean) }],
      tip: (h) => { const i = h[0][0] - 1, f = H.auc_folds.map((a) => a[i]);
        return `<b>Epoch ${h[0][0]}</b><div>Mean AUC ${h[0][1].toFixed(3)}</div><div class="muted">Folds ${Math.min(...f).toFixed(3)}–${Math.max(...f).toFixed(3)}</div>`; },
    });
    const maxLoss = Math.ceil(Math.max(...H.loss_folds.flat()) * 5) / 5;
    const lossTicks = Array.from({ length: Math.round(maxLoss / 0.2) + 1 }, (_, i) => +(i * 0.2).toFixed(1));
    const drawLoss = () => lineChart($("lossChart"), {
      label: "Training loss by epoch", height: 260, x: [1, ep.length], y: [0, maxLoss], xTickValues: epTicks, yTickValues: lossTicks,
      fx: (v) => Math.round(v), fy: (v) => v.toFixed(1), xLabel: "Epoch", yLabel: "Training loss",
      context: H.loss_folds.map(zip), series: [{ name: "Mean", color: SERIES[0], pts: zip(H.loss_mean) }],
      tip: (h) => `<b>Epoch ${h[0][0]}</b><div>Mean loss ${h[0][1].toFixed(3)}</div>`,
    });
    const drawAll = () => { drawRoc(); drawAuc(); drawLoss(); };
    drawAll();
    let t; window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(drawAll, 150); });

    const peak = H.auc_mean.indexOf(Math.max(...H.auc_mean));
    const last10 = H.auc_mean.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, H.auc_mean.length);
    $("fitText").innerHTML = `Validation AUC rose steadily and peaked at epoch <b>${peak + 1}</b> of ${ep.length} (mean ${H.auc_mean[peak].toFixed(3)});
      the last 10 epochs average <b>${last10.toFixed(3)}</b>. An overfitting model would peak early and then decline while its training loss kept falling.`;

    // tables
    $("foldTable").innerHTML = `<tr><th>Fold</th><th class="num">AUC</th><th class="num">Accuracy</th><th class="num">Sensitivity</th><th class="num">Specificity</th><th class="num">Nodules</th></tr>` +
      M.fold_final.map((r) => `<tr><td>${r.fold.replace("fold", "Fold ")}</td><td class="num">${r.auc.toFixed(3)}</td><td class="num">${pct(r.accuracy)}</td>
        <td class="num">${pct(r.sensitivity)}</td><td class="num">${pct(r.specificity)}</td><td class="num">${r.n}</td></tr>`).join("");
    $("rocTable").innerHTML = `<tr><th>Series</th><th class="num">AUC</th><th class="num">95% CI</th><th class="num">n</th><th class="num">Malignant</th></tr>
      <tr><td>Nodule level</td><td class="num">${n.auc.toFixed(3)}</td><td class="num">${n.auc_95ci.map((v) => v.toFixed(3)).join("–")}</td><td class="num">${n.n}</td><td class="num">${n.n_positive}</td></tr>
      <tr><td>Patient level</td><td class="num">${p.auc.toFixed(3)}</td><td class="num">${p.auc_95ci.map((v) => v.toFixed(3)).join("–")}</td><td class="num">${p.n}</td><td class="num">${p.n_positive}</td></tr>`;
    $("curveTable").innerHTML = `<tr><th>Epoch</th><th class="num">Mean val. AUC</th><th class="num">Mean train loss</th></tr>` +
      ep.filter((e) => e === 1 || e % 5 === 0).map((e) => `<tr><td>${e}</td><td class="num">${H.auc_mean[e - 1].toFixed(3)}</td><td class="num">${H.loss_mean[e - 1].toFixed(3)}</td></tr>`).join("");
  }).catch(() => {
    document.querySelectorAll(".needs-metrics").forEach((el) => (el.innerHTML = `<p class="muted">No evaluation results found yet. Train the models and run <span class="mono">cadc.evaluate</span>.</p>`));
  });
})();
