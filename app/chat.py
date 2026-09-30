"""AI assistant for the web app, powered by the Groq API (OpenAI-compatible chat completions).

The API key is read from the GROQ_API_KEY environment variable or from a `.env` file in the project
root (which git ignores). Never put the key in code: the repository is public.
"""
import json
import os
from pathlib import Path

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 12          # most recent messages sent to the model
MAX_CHARS = 2000        # per message

SYSTEM_PROMPT = """You are the CADC assistant, built into a student research web app about lung nodules and lung cancer.

WHAT CADC IS
- A research prototype: an ensemble of neural networks (a 3D ResNet trained from scratch and a 2.5D multi-view CNN built on an ImageNet-pretrained ResNet18) that estimates how suspicious a lung nodule on a chest CT looks.
- It was trained on the public LIDC-IDRI dataset (1,018 CT scans; 1,627 nodules with clear radiologist ratings). The labels are radiologists' malignancy ratings (mean >= 3.5 of 5 = malignant, <= 2.5 = benign), NOT biopsy results.
- A pretrained MONAI RetinaNet (trained on LUNA16) can find nodule candidates automatically.
- Results are shown as a percentage "malignancy likelihood" with bands: low (< 30%), intermediate (30-70%), high (> 70%). An orange "AI attention" (Grad-CAM) heatmap shows where the 3D models looked; it is not proof of disease.
- Pages: Home; Learn (3D lung model with cancer stages I-IV, symptoms, risk factors, nodules, staging, when to see a doctor); Analyse a scan (upload a DICOM CT as a .zip or .dcm files, or try the sample; find nodules automatically or click one; printable report); The model (performance, charts, limitations); About & FAQ (disclaimer, privacy).
- Scans are processed locally on the user's computer. Your chat messages, however, are sent to the Groq API to generate replies; say so if asked about privacy, and advise not to share names or other identifying details.
{performance}

HOW TO ANSWER
- Plain, warm, clear language for a general audience. Keep answers short: usually under 150 words. Use short paragraphs or a few bullet points. No tables.
- You may explain lung anatomy, lung cancer, nodules, screening, staging, risk factors, symptoms, how CT and AI models work, and how to use this app and read its results.
- Be honest about the model's limits, including its much weaker performance against confirmed diagnoses than against radiologists' ratings.

SAFETY RULES (always follow)
- You are an AI, not a doctor. Never diagnose, never say whether a person has or does not have cancer, never interpret a person's scan as a diagnosis, and never recommend starting, stopping or changing any treatment or medication.
- When someone describes their own symptoms, results or worries, respond kindly, give general information only, and encourage them to see a qualified doctor (for example a pulmonologist or oncologist). A high score does not mean cancer; a low score does not rule it out.
- If someone mentions coughing up more than a little blood, severe difficulty breathing, severe chest pain, fainting, or thoughts of self-harm, tell them to contact emergency services immediately (112 in India and Europe, 911 in the US) before anything else.
- If you are unsure or the question is outside your knowledge, say so. Do not invent statistics or studies.
- Politely decline requests unrelated to health, lungs, this project, or AI/medical imaging, and offer to help with those topics instead."""


def load_env(path):
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key():
    return os.environ.get("GROQ_API_KEY", "").strip()


def performance_summary(run_dir):
    """A few lines of real results for the system prompt, if an evaluated run exists."""
    run_dir = Path(run_dir)
    lines = []
    comp = run_dir / "comparison.json"
    if comp.exists():
        lines.append("MEASURED PERFORMANCE (5-fold cross-validation on patients the models never saw):")
        for r in json.loads(comp.read_text()):
            n = r["nodule_level"]
            dx = f"; AUC vs confirmed diagnosis {r['diagnosis_auc']:.2f}" if "diagnosis_auc" in r else ""
            lines.append(f"- {r['name']}: AUC vs radiologist ratings {n['auc']:.3f}, accuracy {n['accuracy']:.1%}, "
                         f"sensitivity {n['sensitivity']:.1%}, specificity {n['specificity']:.1%}{dx}")
    dxf = run_dir / "diagnosis_metrics.json"
    if dxf.exists():
        d = json.loads(dxf.read_text())
        lines.append(f"- 'AUC vs confirmed diagnosis' uses {d['patients_scored']} patients from the same LIDC-IDRI dataset (not a separate "
                     "dataset) whose diagnosis was confirmed by biopsy, surgery, 2 years of stability, or progression; each patient "
                     f"is scored by their most suspicious nodule. On them the radiologists' own ratings reached AUC "
                     f"{d['radiologists']['auc']:.2f}. The models learned to imitate ratings, so predicting actual cancer is harder, "
                     "and with so few patients the uncertainty is about +/-0.12.")
    return "\n".join(lines)


def context_note(ctx):
    """Describe what the user is looking at (page, latest analysis) for the model."""
    if not isinstance(ctx, dict):
        return ""
    parts = []
    if ctx.get("page"):
        parts.append(f"The user is on the '{str(ctx['page'])[:30]}' page.")
    res = ctx.get("result")
    if isinstance(res, dict) and isinstance(res.get("probability"), (int, float)):
        p = float(res["probability"])
        band = "high" if p >= 0.7 else "intermediate" if p >= 0.3 else "low"
        parts.append(f"Their most recent analysis in the app: malignancy likelihood {p:.0%} ({band} suspicion), "
                     f"{int(res.get('agree', 0))} of {int(res.get('models', 0))} networks agreeing"
                     + (", on a scan from the training dataset (so the score is optimistic)" if res.get("in_training_data") else "")
                     + ". Explain it in general terms only and remind them it is not a diagnosis.")
    if isinstance(ctx.get("candidates"), int):
        parts.append(f"Automatic detection found {ctx['candidates']} nodule candidate(s) in their scan.")
    return ("\n\nCURRENT CONTEXT\n" + " ".join(parts)) if parts else ""


def clean_messages(messages):
    out = []
    for m in messages[-MAX_TURNS:] if isinstance(messages, list) else []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            text = m["content"].strip()[:MAX_CHARS]
            if text:
                out.append({"role": m["role"], "content": text})
    return out


def stream_reply(messages, context, run_dir, model=None):
    """Yield the assistant's reply as text chunks."""
    system = SYSTEM_PROMPT.replace("{performance}", performance_summary(run_dir)) + context_note(context)
    payload = {
        "model": model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        "temperature": 0.4,
        "max_completion_tokens": 900,
    }
    if payload["model"].startswith("openai/gpt-oss"):
        payload.update(reasoning_effort="low", include_reasoning=False)
    try:
        with requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key()}"}, json=payload,
                           stream=True, timeout=(10, 90)) as r:
            if r.status_code == 401:
                yield "The chat assistant's API key was rejected. Check GROQ_API_KEY in the .env file."
                return
            if r.status_code == 429:
                yield "The assistant is receiving too many requests right now. Please wait a moment and try again."
                return
            if not r.ok:
                yield f"The assistant is unavailable right now (error {r.status_code}). Please try again later."
                return
            for line in r.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
    except requests.RequestException:
        yield "\n\nCould not reach the assistant. Check your internet connection and try again."
