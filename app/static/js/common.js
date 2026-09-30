// Shared page chrome: navigation, footer, the AI-disclaimer consent dialog, toasts, scroll reveals.
(function () {
  const PAGES = [
    ["/", "Home", "home"],
    ["/learn", "Learn", "learn"],
    ["/analyze", "Analyse a scan", "analyze"],
    ["/model", "The model", "model"],
    ["/about", "About & FAQ", "about"],
  ];
  const CONSENT_KEY = "cadc-consent-v1";
  const current = document.body.dataset.page;

  const LOGO = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#060a12" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 3v7.5"/><path d="M12 10.5c-1 1.2-2 1.8-3 2"/><path d="M12 10.5c1 1.2 2 1.8 3 2"/>
      <path d="M9 6.5C6.2 7.2 4 10.6 4 15.2 4 18.4 5.2 20 7 20c2.6 0 4-2.1 4-5.6V9.5"/>
      <path d="M15 6.5c2.8.7 5 4.1 5 8.7 0 3.2-1.2 4.8-3 4.8-2.6 0-4-2.1-4-5.6V9.5"/></svg>`;
  const WARN_ICON = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;

  // ---------- nav ----------
  const nav = document.getElementById("nav");
  if (nav) {
    nav.outerHTML = `
    <nav class="nav"><div class="container nav-inner">
      <a class="brand" href="/"><span class="logo">${LOGO}</span>
        <span>CADC<small>Lung nodule AI</small></span></a>
      <button class="menu-btn" id="menuBtn" aria-label="Open menu" aria-expanded="false">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
      </button>
      <div class="nav-links" id="navLinks">
        ${PAGES.map(([href, label, key]) => `<a href="${href}" class="${key === current ? "active" : ""}">${label}</a>`).join("")}
      </div>
      ${current === "analyze" ? "" : `<a class="btn primary sm nav-cta" href="/analyze">Try the analyser</a>`}
    </div></nav>`;
    const btn = document.getElementById("menuBtn"), links = document.getElementById("navLinks");
    btn.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      btn.setAttribute("aria-expanded", open);
    });
  }

  // ---------- footer ----------
  const foot = document.getElementById("footer");
  if (foot) {
    foot.outerHTML = `
    <footer class="site"><div class="container">
      <div class="foot-grid">
        <div>
          <a class="brand" href="/"><span class="logo">${LOGO}</span><span>CADC<small>Lung nodule AI</small></span></a>
          <p style="margin-top:14px;max-width:420px">A research project: a 3D deep-learning model that estimates how suspicious a lung nodule
            looks on CT, trained on the public LIDC-IDRI dataset.</p>
        </div>
        <div><h4>Explore</h4>${PAGES.map(([h, l]) => `<a href="${h}">${l}</a>`).join("")}</div>
        <div><h4>Get help</h4>
          <a href="/learn#doctor">When to see a doctor</a>
          <a href="/about#privacy">Privacy</a>
          <a href="/about#faq">FAQ</a>
          <a href="#" data-consent>Read the AI disclaimer</a>
        </div>
      </div>
      <div class="foot-warn">
        <b style="color:var(--text-2)">Medical disclaimer.</b> This is an artificial-intelligence research tool, not a medical device.
        It has not been clinically validated and it cannot diagnose, rule out or treat any disease. Its output can be wrong.
        Always consult a qualified doctor about any health concern, scan or symptom, and never delay or ignore medical advice because of
        something you saw here. In an emergency, contact your local emergency number.
      </div>
    </div></footer>`;
  }

  // ---------- consent dialog ----------
  function readConsent() { try { return localStorage.getItem(CONSENT_KEY) === "yes"; } catch { return false; } }
  function saveConsent() { try { localStorage.setItem(CONSENT_KEY, "yes"); } catch { /* storage blocked: ask again next time */ } }
  let accepted = readConsent();
  const listeners = [];

  const modal = document.createElement("div");
  modal.className = "modal-backdrop";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  modal.setAttribute("aria-labelledby", "consentTitle");
  modal.innerHTML = `
    <div class="modal">
      <div class="badge">${WARN_ICON.replace('width="20" height="20"', 'width="26" height="26"')}</div>
      <h2 id="consentTitle">Before you continue: this is an AI model</h2>
      <p class="text-2" style="margin:0">CADC is an artificial-intelligence research prototype. Please read and accept the following:</p>
      <ul>
        <li><b>It is not a doctor and not a diagnosis.</b> It estimates how suspicious a nodule <i>looks</i>, based on radiologists' opinions in a research dataset, and it can be wrong.</li>
        <li><b>Always consult a qualified doctor</b> (for example a pulmonologist or oncologist) about any scan, result or symptom. Only they can examine you, order tests and make a diagnosis.</li>
        <li><b>Do not use it to make medical decisions</b>, and do not delay, change or skip care because of its output.</li>
        <li>In an emergency, such as coughing up blood or severe difficulty breathing, <b>contact emergency services immediately</b>.</li>
      </ul>
      <label class="check"><input type="checkbox" id="consentBox">
        <span>I understand that this is an AI research tool, that its results are not medical advice, and that I should consult a doctor.</span></label>
      <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:18px;flex-wrap:wrap">
        <a class="btn" href="/learn#doctor">When to see a doctor</a>
        <button class="btn primary" id="consentOk" disabled>I understand, continue</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  const box = modal.querySelector("#consentBox"), ok = modal.querySelector("#consentOk");
  box.addEventListener("change", () => { ok.disabled = !box.checked; });
  ok.addEventListener("click", () => {
    accepted = true; saveConsent(); modal.classList.remove("on");
    listeners.splice(0).forEach((fn) => fn());
  });
  function showConsent() { box.checked = accepted; ok.disabled = !accepted; modal.classList.add("on"); setTimeout(() => box.focus(), 50); }
  if (!accepted) showConsent();
  document.addEventListener("click", (e) => {
    const t = e.target && e.target.closest && e.target.closest("[data-consent]"); if (t) { e.preventDefault(); showConsent(); }
  });

  // ---------- toast ----------
  const toastEl = document.createElement("div");
  toastEl.className = "toast"; toastEl.setAttribute("role", "status");
  document.body.appendChild(toastEl);
  function toast(msg) {
    toastEl.textContent = msg; toastEl.classList.add("on");
    clearTimeout(toastEl._h); toastEl._h = setTimeout(() => toastEl.classList.remove("on"), 5500);
  }

  // ---------- reveal on scroll ----------
  const io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.12 }) : null;
  document.querySelectorAll(".reveal").forEach((el) => (io ? io.observe(el) : el.classList.add("in")));

  window.CADC = {
    toast,
    warnIcon: WARN_ICON,
    // Run fn once the visitor has accepted the disclaimer (immediately if already accepted).
    whenConsented(fn) { accepted ? fn() : listeners.push(fn); },
    get consented() { return accepted; },
  };
})();
