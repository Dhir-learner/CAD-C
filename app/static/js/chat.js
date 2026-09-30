// Floating AI assistant (Groq via the local server's /api/chat). Loaded on every page after common.js.
(function () {
  const STORE = "cadc-chat-v1";
  const SUGGEST = {
    analyze: ["What does my result mean?", "What is the AI attention heatmap?", "How do I find a nodule on the scan?", "Should I be worried?"],
    learn: ["What is a lung nodule?", "What are the stages of lung cancer?", "Who should get screened?", "Symptoms to watch for?"],
    model: ["How accurate is the model?", "Why is it worse on real diagnoses?", "What is an AUC?", "Is the model overfitting?"],
    default: ["What is a lung nodule?", "How does this AI work?", "How accurate is it?", "When should I see a doctor?"],
  };
  const page = document.body.dataset.page || "default";
  let history = [];
  try { history = JSON.parse(sessionStorage.getItem(STORE) || "[]"); } catch { history = []; }
  const save = () => { try { sessionStorage.setItem(STORE, JSON.stringify(history.slice(-30))); } catch { /* storage blocked */ } };

  const ICON_CHAT = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h.01M12 11h.01M15.5 11h.01"/></svg>`;
  const ICON_X = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>`;
  const ICON_SEND = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 14-7-5 16-3-7z"/></svg>`;

  const fab = document.createElement("button");
  fab.className = "chat-fab"; fab.setAttribute("aria-label", "Ask the CADC AI assistant");
  fab.innerHTML = `${ICON_CHAT}<span>Ask CADC AI</span>`;
  const panel = document.createElement("section");
  panel.className = "chat-panel"; panel.setAttribute("aria-label", "CADC AI assistant");
  panel.innerHTML = `
    <header class="chat-head">
      <span class="logo" style="width:34px;height:34px;border-radius:10px">${ICON_CHAT.replace('width="24" height="24"', 'width="18" height="18"').replace('stroke="currentColor"', 'stroke="#060a12"')}</span>
      <div style="flex:1;min-width:0"><b>CADC assistant</b><div class="tiny muted">AI · not a doctor · can make mistakes</div></div>
      <button class="chat-icon" id="chatClear" title="New conversation" aria-label="New conversation">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg></button>
      <button class="chat-icon" id="chatClose" aria-label="Close">${ICON_X}</button>
    </header>
    <div class="chat-log" id="chatLog" aria-live="polite"></div>
    <div class="chat-suggest" id="chatSuggest"></div>
    <form class="chat-form" id="chatForm">
      <textarea id="chatInput" rows="1" maxlength="2000" placeholder="Ask about lung nodules, lung cancer or this app…" aria-label="Your question"></textarea>
      <button class="chat-send" id="chatSend" aria-label="Send">${ICON_SEND}</button>
    </form>
    <div class="chat-foot tiny">For health concerns, see a doctor. Emergencies: call 112 / 911. Messages are sent to Groq to generate replies; don't include names.</div>`;
  document.body.append(fab, panel);
  const $ = (id) => panel.querySelector("#" + id);
  const log = $("chatLog"), input = $("chatInput");

  fetch("/api/status").then((r) => r.json()).then((st) => {
    if (!st.chat) { fab.title = "Chat assistant not configured (add GROQ_API_KEY to .env)"; fab.classList.add("off"); }
  }).catch(() => fab.classList.add("off"));

  // Minimal, safe markdown: escape first, then bold/italic/code, bullet and numbered lists, paragraphs.
  function md(text) {
    const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const inline = (t) => t.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<i>$2</i>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const out = []; let list = null;
    for (const raw of esc.split("\n")) {
      const line = raw.trimEnd();
      const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ul || ol) {
        const tag = ul ? "ul" : "ol";
        if (!list || list.tag !== tag) { if (list) out.push(`</${list.tag}>`); out.push(`<${tag}>`); list = { tag }; }
        out.push(`<li>${inline((ul || ol)[1])}</li>`);
        continue;
      }
      if (list) { out.push(`</${list.tag}>`); list = null; }
      if (/^#{1,4}\s+/.test(line)) out.push(`<p><b>${inline(line.replace(/^#+\s+/, ""))}</b></p>`);
      else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list.tag}>`);
    return out.join("");
  }

  function bubble(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.innerHTML = role === "user" ? md(text) : md(text) || '<span class="typing"><i></i><i></i><i></i></span>';
    log.appendChild(el); log.scrollTop = log.scrollHeight;
    return el;
  }
  function render() {
    log.innerHTML = "";
    if (!history.length) {
      const hello = document.createElement("div");
      hello.className = "msg assistant";
      hello.innerHTML = md("Hi! I can explain **lung nodules**, **lung cancer**, and how to use and read **this app**.\n\nI'm an AI, not a doctor: for anything about your own health, please talk to a qualified doctor.");
      log.appendChild(hello);
    }
    history.forEach((m) => bubble(m.role, m.content));
    const opts = (SUGGEST[page] || SUGGEST.default).filter((q) => q !== "What does my result mean?" || (window.CADC && CADC.context && CADC.context.result));
    $("chatSuggest").innerHTML = history.length ? "" : opts.map((q) => `<button type="button" class="chip">${q}</button>`).join("");
  }

  let busy = false;
  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    if (!CADC.consented) { CADC.toast("Please accept the AI disclaimer first."); document.querySelector("[data-consent]")?.click(); return; }
    busy = true; $("chatSend").disabled = true;
    history.push({ role: "user", content: text }); save();
    $("chatSuggest").innerHTML = "";
    bubble("user", text);
    const el = bubble("assistant", "");
    let reply = "";
    try {
      const r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history, context: Object.assign({ page }, (window.CADC && CADC.context) || {}) }),
      });
      if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { /* keep */ } throw new Error(m); }
      const reader = r.body.getReader(), dec = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        reply += dec.decode(value, { stream: true });
        el.innerHTML = md(reply); log.scrollTop = log.scrollHeight;
      }
      if (!reply.trim()) reply = "Sorry, I couldn't produce an answer. Please try again.";
    } catch (e) {
      reply = `Sorry, the assistant is unavailable: ${e.message}`;
    }
    el.innerHTML = md(reply);
    history.push({ role: "assistant", content: reply }); save();
    busy = false; $("chatSend").disabled = false; input.focus();
  }

  function open() { panel.classList.add("on"); fab.classList.add("hide"); render(); setTimeout(() => input.focus(), 50); }
  function close() { panel.classList.remove("on"); fab.classList.remove("hide"); }
  fab.addEventListener("click", () => (fab.classList.contains("off") ? CADC.toast(fab.title || "Chat unavailable") : open()));
  $("chatClose").addEventListener("click", close);
  $("chatClear").addEventListener("click", () => { history = []; save(); render(); });
  $("chatSuggest").addEventListener("click", (e) => { if (e.target.matches("button")) send(e.target.textContent); });
  $("chatForm").addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; input.style.height = ""; send(t); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chatForm").requestSubmit(); } });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && panel.classList.contains("on")) close(); });
  window.CADC.openChat = (question) => { open(); if (question) send(question); };
})();
