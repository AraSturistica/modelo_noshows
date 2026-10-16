"""
Outpatient No-Show (DNA) Risk Predictor - Gradio demo (v2, polished UI)
Proof of concept trained on the public Kaggle "Medical Appointment No Shows"
dataset (Brazil, 2016). Not NHS data and not for clinical or operational use.

Files needed in the same folder:
    xgb_noshow.json
    model_features.json
"""

import json
import math
import os
from html import escape
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def find(fname):
    for folder in (BASE_DIR, BASE_DIR / "outputs"):
        path = folder / fname
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {fname} in {BASE_DIR} or {BASE_DIR / 'outputs'}"
    )

with find("model_features.json").open(encoding="utf-8") as f:
    meta = json.load(f)
try:
    FEATURES = meta["features"]
    NEIGHBOURHOODS = meta["neighbourhood_categories"]
except (KeyError, TypeError) as exc:
    raise ValueError("model_features.json must define features and neighbourhood_categories") from exc
if not isinstance(FEATURES, list) or not FEATURES or not all(isinstance(x, str) for x in FEATURES):
    raise ValueError("model_features.json contains an invalid features list")
if not isinstance(NEIGHBOURHOODS, list) or not NEIGHBOURHOODS:
    raise ValueError("model_features.json contains no neighbourhood categories")

MODEL_PATH = find("xgb_noshow.json")
with MODEL_PATH.open(encoding="utf-8") as f:
    model_metadata = json.load(f)
MODEL_VERSION = tuple(model_metadata.get("version", ()))
try:
    RUNTIME_VERSION = tuple(int(part) for part in xgb.__version__.split("+")[0].split(".")[:3])
except ValueError as exc:
    raise RuntimeError(f"Cannot determine the installed XGBoost version: {xgb.__version__}") from exc
if len(MODEL_VERSION) >= 2 and RUNTIME_VERSION < MODEL_VERSION[:3]:
    raise RuntimeError(
        f"xgb_noshow.json was saved with XGBoost {'.'.join(map(str, MODEL_VERSION))}, "
        f"but XGBoost {xgb.__version__} is installed. Install xgboost>="
        f"{'.'.join(map(str, MODEL_VERSION[:3]))} to avoid incompatible categorical predictions."
    )
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# -----------------------------------------------------------------------------
# LLM (Hugging Face Inference Providers, free monthly credits with an HF token)
# Add HF_TOKEN as a Secret in the Space settings. Change model with LLM_MODEL.
# -----------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")
llm_client = InferenceClient(token=HF_TOKEN, timeout=60) if HF_TOKEN else None

MODEL_CARD = """About the underlying prediction model:
- XGBoost classifier trained on 110,527 public outpatient appointments from Brazil (2016, Kaggle).
- Not NHS data. A proof of concept, not a clinical or operational tool.
- Held-out results: ROC AUC 0.73; contacting the top 20% highest-risk appointments reaches about 38% of no-shows (about 2x random).
- Average no-show rate is about 20%. Risk bands: Low under 20%, Medium 20-35%, High 35% and above.
- Strongest driver overall is booking lead time (same-day bookings 4.6% no-show vs 24-34% when booked a week or more ahead).
- Limitations: only about six weeks of data so attendance history is short; the SMS reminder effect is confounded with lead time; neighbourhood likely reflects deprivation or travel distance.
- Scores should be used to offer patients extra support, never to penalise them."""

GUARDRAILS = """Rules:
- Use ONLY the facts provided. Never invent numbers, factors or patient details.
- The risk score comes from the prediction model. Do not change or re-estimate it.
- Do not give clinical or medical advice.
- Keep language plain, respectful and non-judgemental about patients.
- If asked something the facts cannot answer, say so briefly."""


def llm_available():
    return llm_client is not None


def call_llm(messages, max_tokens=350):
    if llm_client is None:
        raise RuntimeError("LLM client is not configured (missing HF_TOKEN)")
    # Extra token budget: reasoning models (e.g. gpt-oss) think before answering
    resp = llm_client.chat_completion(
        messages=messages, model=LLM_MODEL, max_tokens=max_tokens + 1200, temperature=0.3
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty response")
    return text


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
AVG_RATE = 0.20

LABELS = {
    "lead_time_days": "Days between booking and appointment",
    "same_day_booking": "Booked same day",
    "appt_weekday": "Appointment day",
    "booked_hour": "Hour booking was made",
    "sms_received": "SMS reminder sent",
    "age": "Age",
    "is_male": "Male",
    "welfare_programme": "Welfare programme",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "has_handicap": "Disability recorded",
    "n_conditions": "Recorded conditions",
    "prior_appts": "Previous appointments",
    "prior_no_shows": "Previous no-shows",
    "prior_no_show_rate": "Previous no-show rate",
    "is_first_appt": "First appointment",
    "days_since_last_appt": "Days since last appointment",
    "multiple_appts_same_day": "Several appointments that day",
    "neighbourhood": "Clinic neighbourhood",
}
BINARY = {"same_day_booking", "sms_received", "is_male", "welfare_programme",
          "hypertension", "diabetes", "alcoholism", "has_handicap",
          "is_first_appt", "multiple_appts_same_day"}


def build_row(lead_time, age, gender, weekday, booked_hour, sms, welfare,
              conditions, prior_appts, prior_no_shows, days_since_last,
              multiple_same_day, neighbourhood):
    def checked_int(name, value, minimum, maximum=None):
        if value is None or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a finite integer")
        number = int(numeric)
        if number < minimum or (maximum is not None and number > maximum):
            bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            raise ValueError(f"{name} must be {bounds}")
        return number

    lead_time = checked_int("lead_time", lead_time, 0, 120)
    age = checked_int("age", age, 0, 100)
    booked_hour = checked_int("booked_hour", booked_hour, 6, 21)
    prior_appts = checked_int("prior_appts", prior_appts or 0, 0)
    prior_no_shows = checked_int("prior_no_shows", prior_no_shows or 0, 0, prior_appts)
    days_since_last = checked_int("days_since_last", days_since_last or 0, 0)
    if weekday not in WEEKDAYS:
        raise ValueError(f"weekday must be one of: {', '.join(WEEKDAYS)}")
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError("neighbourhood is not present in the model categories")
    if gender not in ("Female", "Male"):
        raise ValueError("gender must be Female or Male")
    conditions = conditions or []
    if not isinstance(conditions, (list, tuple)) or any(
        c not in {"Hypertension", "Diabetes", "Alcoholism", "Disability"}
        for c in conditions
    ):
        raise ValueError("conditions contains an unsupported value")
    for name, value in (("sms", sms), ("welfare", welfare),
                        ("multiple_same_day", multiple_same_day)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    has = lambda c: int(c in conditions)

    row = {
        "lead_time_days": int(lead_time),
        "same_day_booking": int(lead_time == 0),
        "appt_weekday": WEEKDAYS.index(weekday),
        "booked_hour": int(booked_hour),
        "sms_received": int(sms),
        "age": int(age),
        "is_male": int(gender == "Male"),
        "welfare_programme": int(welfare),
        "hypertension": has("Hypertension"),
        "diabetes": has("Diabetes"),
        "alcoholism": has("Alcoholism"),
        "has_handicap": has("Disability"),
        "prior_appts": prior_appts,
        "prior_no_shows": prior_no_shows,
        "is_first_appt": int(prior_appts == 0),
        "prior_no_show_rate": prior_no_shows / prior_appts if prior_appts > 0 else -1.0,
        "days_since_last_appt": int(days_since_last or 0) if prior_appts > 0 else -1,
        "multiple_appts_same_day": int(multiple_same_day),
        "neighbourhood": neighbourhood,
    }
    row["n_conditions"] = (row["hypertension"] + row["diabetes"]
                           + row["alcoholism"] + row["has_handicap"])
    missing = set(FEATURES) - row.keys()
    if missing:
        raise ValueError(f"Model expects unsupported feature(s): {', '.join(sorted(missing))}")
    X = pd.DataFrame([row])[FEATURES]
    X["neighbourhood"] = pd.Categorical(X["neighbourhood"], categories=NEIGHBOURHOODS)
    return X


def fmt_value(feat, val):
    if feat == "appt_weekday":
        return WEEKDAYS[int(val)]
    if feat == "prior_no_show_rate":
        return "n/a" if val < 0 else f"{val:.0%}"
    if feat == "days_since_last_appt" and val < 0:
        return "n/a"
    if feat in BINARY:
        return "Yes" if int(val) == 1 else "No"
    if feat == "lead_time_days":
        return f"{int(val)} days"
    return str(val)


def band(p):
    if p >= 0.35:
        return ("High", "#dc2626",
                "Phone the patient a few days before and offer to rebook if the time no longer suits.")
    if p >= 0.20:
        return ("Medium", "#d97706",
                "Send an SMS or automated reminder a few days before the appointment.")
    return ("Low", "#16a34a", "Standard process. No extra contact needed.")


# -----------------------------------------------------------------------------
# Output rendering
# -----------------------------------------------------------------------------
def risk_card(p, top):
    level, colour, action = band(p)
    ratio = p / AVG_RATE
    comparison = (f"{ratio:.1f}× the average appointment" if ratio >= 1.05
                  else f"below the average of {AVG_RATE:.0%}" if ratio <= 0.95
                  else "about average")
    marker = min(max(p, 0.01), 0.99) * 100

    rows = ""
    for feat, v, val in top:
        up = v > 0
        rows += f"""
        <div class="factor">
          <span class="dot" style="background:{'#dc2626' if up else '#2563eb'}"></span>
          <span class="fname">{LABELS[feat]}</span>
          <span class="fval">{escape(str(val))}</span>
          <span class="fdir" style="color:{'#dc2626' if up else '#2563eb'}">{'▲ raises' if up else '▼ lowers'}</span>
        </div>"""

    return f"""
    <div class="card">
      <div class="card-top">
        <div>
          <div class="eyebrow">Predicted no-show risk</div>
          <div class="big">{p:.0%}</div>
          <div class="muted">{comparison}</div>
        </div>
        <div class="pill" style="background:{colour}1a;color:{colour};border-color:{colour}55">{level} risk</div>
      </div>
      <div class="meter">
        <div class="seg" style="background:#16a34a"></div>
        <div class="seg" style="background:#d97706"></div>
        <div class="seg" style="background:#dc2626"></div>
        <div class="needle" style="left:{marker}%"></div>
      </div>
      <div class="scale"><span>0%</span><span>20%</span><span>35%</span><span>100%</span></div>
      <div class="action" style="border-left-color:{colour}">
        <div class="eyebrow">Suggested action</div>
        <div>{action}</div>
      </div>
      <div class="eyebrow" style="margin-top:18px">Main factors</div>
      {rows}
    </div>"""


def factor_chart(contrib, X):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    labels = [f"{LABELS[f]}  ({fmt_value(f, X.iloc[0][f])})" for f in top.index]
    colours = ["#dc2626" if v > 0 else "#2563eb" for v in top.values]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(labels, top.values, color=colours, height=0.6)
    ax.axvline(0, color="#9ca3af", lw=1)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#6b7280", labelsize=9, length=0)
    ax.set_xlabel("◀ lowers risk        contribution        raises risk ▶", color="#6b7280", fontsize=9)
    ax.set_title("What's driving this score", loc="left", color="#6b7280", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def predict(*inputs):
    X = build_row(*inputs)
    p = float(model.predict_proba(X)[0, 1])
    shap_values = np.asarray(explainer(X).values)
    values = shap_values
    n_feat = len(FEATURES)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes)
        values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
    elif values.ndim == 2:
        if values.shape[1] == n_feat:
            # (n_samples, n_features) — SHAP 0.51 + binary XGBoost
            values = values[0]
        elif values.shape[0] == n_feat:
            # (n_features, n_classes)
            values = values[:, 1] if values.shape[-1] > 1 else values[:, 0]
        else:
            values = values[0]
    if values.ndim != 1 or len(values) != len(FEATURES):
        raise ValueError(
            f"Unexpected SHAP output shape {shap_values.shape}; "
            f"expected one contribution per {len(FEATURES)} feature"
        )
    contrib = pd.Series(values, index=FEATURES)
    order = contrib.abs().sort_values(ascending=False).index[:5]
    top = [(f, contrib[f], fmt_value(f, X.iloc[0][f])) for f in order]

    level, _, action = band(p)
    order6 = contrib.abs().sort_values(ascending=False).index[:6]
    ctx = {
        "risk": f"{p:.0%}",
        "level": level,
        "action": action,
        "factors": [
            {"factor": LABELS[f], "value": fmt_value(f, X.iloc[0][f]),
             "effect": "raises risk" if contrib[f] > 0 else "lowers risk"}
            for f in order6
        ],
    }
    # New prediction: clear AI outputs and chat so nothing refers to an old patient
    return risk_card(p, top), factor_chart(contrib, X), ctx, "", "", []


# -----------------------------------------------------------------------------
# LLM features
# -----------------------------------------------------------------------------
def ctx_text(ctx):
    lines = [f"Predicted no-show risk: {ctx['risk']} ({ctx['level']} risk).",
             f"Suggested action from the risk band: {ctx['action']}",
             "Main factors (from SHAP, most important first):"]
    lines += [f"- {f['factor']} = {f['value']} ({f['effect']})" for f in ctx["factors"]]
    return "\n".join(lines)


def fallback_summary(ctx):
    ups = [f"{f['factor'].lower()} ({f['value']})" for f in ctx["factors"] if "raises" in f["effect"]][:3]
    downs = [f"{f['factor'].lower()} ({f['value']})" for f in ctx["factors"] if "lowers" in f["effect"]][:2]
    text = f"This appointment has a **{ctx['risk']}** predicted chance of a no-show (**{ctx['level']} risk**). "
    if ups:
        text += "Risk is pushed up mainly by " + ", ".join(ups) + ". "
    if downs:
        text += "It is pulled down by " + ", ".join(downs) + ". "
    return text + f"\n\n**Suggested action:** {ctx['action']}"


def staff_summary(ctx):
    if not ctx:
        return "Run a prediction first."
    if not llm_available():
        return fallback_summary(ctx) + "\n\n*Template summary. Add an HF_TOKEN secret to enable the AI version.*"
    msgs = [
        {"role": "system", "content": "You help outpatient clinic admin staff understand a no-show risk prediction.\n"
                                      + GUARDRAILS + "\n\n" + MODEL_CARD},
        {"role": "user", "content": ctx_text(ctx) + "\n\nWrite a summary for clinic staff in 3-4 short sentences: "
                                    "the risk level, the two or three main reasons in plain English, "
                                    "then a final line starting 'Suggested action:'. No headings, no bullet points."},
    ]
    try:
        return call_llm(msgs, max_tokens=250)
    except Exception as e:
        return fallback_summary(ctx) + f"\n\n*AI unavailable right now ({type(e).__name__}), showing template summary.*"


def fallback_reminder(channel):
    if channel == "SMS":
        return ("Hi [Name], this is a reminder of your appointment at [Clinic] on [Date] at [Time]. "
                "If you can no longer make it, please call [Phone] so we can offer the slot to someone else. Thank you.")
    return ("Hello, may I speak to [Name]? I'm calling from [Clinic] to remind you about your appointment on [Date] "
            "at [Time]. Is that time still convenient for you? If not, I can help you rebook now. "
            "Is there anything that might make it difficult to attend, such as travel? Thank you, we look forward to seeing you.")


def patient_reminder(ctx, channel):
    if not ctx:
        return "Run a prediction first."
    if not llm_available():
        return fallback_reminder(channel)
    fmt = ("an SMS under 300 characters" if channel == "SMS"
           else "a short, friendly phone call script for a receptionist (under 120 words)")
    tone = ("Because this appointment has a higher chance of being missed, gently make it easy to rebook "
            "and ask if anything might make attending difficult."
            if ctx["level"] in ("High", "Medium") else "Keep it a simple, standard reminder.")
    msgs = [
        {"role": "system", "content": "You write appointment reminders on behalf of an outpatient clinic.\n"
         "Strict privacy rules: NEVER mention the risk score, predictions, past missed appointments, age, "
         "health conditions or any personal details. Patients must never feel judged.\n"
         "Use placeholders exactly like [Name], [Clinic], [Date], [Time], [Phone]. Warm, clear British English."},
        {"role": "user", "content": f"Write {fmt}. {tone} Return only the message text."},
    ]
    try:
        return call_llm(msgs, max_tokens=250).strip('"')
    except Exception:
        return fallback_reminder(channel)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def chat(message, history, ctx):
    message = (message or "").strip()
    history = history or []
    if not message:
        return "", history
    history = history + [{"role": "user", "content": message}]
    if not llm_available():
        reply = "The AI assistant isn't enabled on this copy of the app (no HF_TOKEN secret)."
    elif not ctx:
        reply = "Run a prediction first, then ask me about it."
    else:
        msgs = [{"role": "system", "content":
                 "You are an assistant inside a no-show risk demo app. Answer questions about the current "
                 "prediction and how the model works, in 2-5 sentences.\n" + GUARDRAILS + "\n\n" + MODEL_CARD
                 + "\n\nCurrent prediction:\n" + ctx_text(ctx)}]
        msgs += [{"role": m["role"], "content": _text(m["content"])} for m in history[-8:]]
        try:
            reply = call_llm(msgs, max_tokens=300)
        except Exception as e:
            reply = f"Sorry, the AI model isn't responding right now ({type(e).__name__}). Please try again shortly."
    return "", history + [{"role": "assistant", "content": reply}]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#1e40af",
    block_radius="14px",
    block_title_background_fill="transparent",
    block_label_background_fill="transparent",
    block_title_text_color="*neutral_700",
    block_title_text_color_dark="*neutral_200",
    block_title_text_weight="500",
)

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hero {padding: 28px 30px; border-radius: 18px; color: #fff;
       background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #0ea5e9 100%);}
.hero, .hero * {color: #fff !important;}
.hero h1 {margin: 0 0 6px; font-size: 1.9rem; font-weight: 700;}
.hero p {margin: 0; opacity: .92; font-size: 1rem; color: #fff;}
.stats {display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px;}
.stat {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
       border-radius: 12px; padding: 10px 14px; flex: 1 1 140px;}
.stat b {display: block; font-size: 1.25rem;}
.stat span {font-size: .8rem; opacity: .9;}
.notice {margin-top: 12px; padding: 10px 14px; border-radius: 10px; font-size: .85rem;
         background: rgba(217,119,6,.10); border: 1px solid rgba(217,119,6,.35);}
.section-title {font-weight: 700; font-size: .95rem; margin: 14px 0 4px 2px;}
@media (max-width: 640px) {.hero {padding: 20px;} .hero h1 {font-size: 1.45rem;} .stat {flex: 1 1 42%;} .big {font-size: 2.6rem;}}
.card {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 22px;
       background: var(--block-background-fill);}
.card-top {display: flex; justify-content: space-between; align-items: flex-start;}
.eyebrow {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
          opacity: .65; font-weight: 600;}
.big {font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin: 4px 0;}
.muted {opacity: .7; font-size: .9rem;}
.pill {border: 1px solid; border-radius: 999px; padding: 6px 14px; font-weight: 700; font-size: .9rem;}
.meter {position: relative; display: flex; height: 10px; border-radius: 999px;
        overflow: visible; margin-top: 18px;}
.seg:nth-child(1) {width: 20%; border-radius: 999px 0 0 999px;}
.seg:nth-child(2) {width: 15%;}
.seg:nth-child(3) {width: 65%; border-radius: 0 999px 999px 0;}
.seg {opacity: .85;}
.needle {position: absolute; top: -5px; width: 4px; height: 20px; margin-left: -2px;
         background: var(--body-text-color); border-radius: 2px;}
.scale {position: relative; height: 16px; font-size: .72rem; opacity: .6; margin-top: 4px;}
.scale span {position: absolute;}
.scale span:nth-child(1) {left: 0;}
.scale span:nth-child(2) {left: 20%; transform: translateX(-50%);}
.scale span:nth-child(3) {left: 35%; transform: translateX(-50%);}
.scale span:nth-child(4) {right: 0;}
.action {margin-top: 16px; padding: 12px 14px; border-left: 4px solid; border-radius: 8px;
         background: var(--background-fill-secondary);}
.factor {display: flex; align-items: center; gap: 10px; padding: 9px 0;
         border-bottom: 1px solid var(--border-color-primary); font-size: .92rem;}
.factor:last-child {border-bottom: none;}
.dot {width: 9px; height: 9px; border-radius: 50%; flex: none;}
.fname {flex: 1;}
.fval {font-weight: 600;}
.fdir {font-size: .8rem; font-weight: 600; min-width: 70px; text-align: right;}
.about h3 {margin-top: 18px;}
footer {display: none !important;}
.pad {padding: 10px 12px 0;}
.pad-out {padding: 4px 14px 12px;}
.ai-btn {margin: 8px 12px !important; width: auto !important;}
"""

HERO = """
<div class="hero">
  <h1>🏥 Outpatient No-Show Risk Predictor</h1>
  <p>Estimates the chance a patient misses an outpatient appointment and explains why,
     so clinics can focus reminders and rebooking where they matter most.</p>
  <div class="stats">
    <div class="stat"><b>110k</b><span>appointments analysed</span></div>
    <div class="stat"><b>0.73</b><span>ROC AUC on unseen dates</span></div>
    <div class="stat"><b>~2×</b><span>no-shows reached vs random targeting</span></div>
    <div class="stat"><b>Lead time</b><span>strongest single driver</span></div>
  </div>
</div>
<div class="notice">⚠️ <b>Proof of concept.</b> Trained on public data (Brazil, 2016), not NHS data,
and not for clinical or operational use. Risk scores should guide extra support for patients,
never penalise them.</div>
"""

ABOUT = """
<div class="about">

### How it works
An XGBoost model trained on 110,527 appointments from public clinics in Brazil
(Kaggle *Medical Appointment No Shows*, Joni Hoppen & Aquarela Analytics). Each prediction
is explained with SHAP, showing which factors pushed that appointment's risk up or down.

### Results on held-out later dates
| Model | ROC AUC | PR-AUC (random = 0.185) | No-shows reached by contacting top 20% |
|---|---|---|---|
| Logistic Regression | 0.728 | 0.338 | 37.2% |
| XGBoost | 0.732 | 0.344 | 37.6% |

### Key findings
- **Booking lead time dominates.** Same-day bookings had a 4.6% no-show rate, versus 24–34% for appointments booked a week or more ahead.
- **Targeting works.** Contacting the 20% highest-risk appointments reaches about twice as many no-shows as random targeting.
- **The lowest-risk 30%** of appointments contain only about 5% of no-shows, so reminder effort there could be reduced.
- **A simple model is nearly as good.** Logistic regression performs almost as well as XGBoost, which matters where transparency is important.

### AI assistant
An open-source LLM (via Hugging Face Inference Providers) writes staff summaries, patient reminders and answers questions. It receives only the model's score and SHAP factors, is instructed never to invent or change them, and reminders never mention risk scores or patient history.

### Design choices
- Patient history uses **only earlier appointments**, avoiding data leakage.
- **Time-based validation** (train on earlier dates, test on the final week) mirrors how a clinic would score future appointments.
- **No class re-weighting**, so a score of 40% means roughly a 40% chance of a no-show.

### Limitations
- Only about six weeks of data, so attendance history is short. With years of NHS records, history features would likely be much stronger.
- SMS reminders were sent mainly for appointments booked further ahead, so their effect is mixed up with lead time. Measuring reminder impact needs a controlled trial.
- Neighbourhood likely reflects deprivation or travel distance. Scores should be used to offer support such as reminders or transport help, never to penalise patients, for example through overbooking.

</div>
"""

default_hood = "Jardim Camburi" if "Jardim Camburi" in NEIGHBOURHOODS else NEIGHBOURHOODS[0]

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
EXAMPLES = [
    [0, 45, "Female", "Monday", 9, False, False, [], 2, 0, 30, False, default_hood],
    [45, 19, "Male", "Friday", 16, True, False, [], 4, 3, 10, False, default_hood],
    [14, 68, "Female", "Wednesday", 11, True, False, ["Hypertension", "Diabetes"], 6, 0, 21, False, default_hood],
]
EXAMPLE_LABELS = [
    "🟢 Same-day booking, reliable attender",
    "🔴 Booked 6 weeks ahead, missed 3 of last 4",
    "🟡 Older patient, regular attender",
]

with gr.Blocks(title="Outpatient No-Show Risk Predictor") as demo:
    gr.HTML(HERO)

    # Components are created first (render=False) so the examples can sit at the top
    lead_time = gr.Slider(0, 120, value=14, step=1, label="Days between booking and appointment", render=False)
    weekday = gr.Dropdown(WEEKDAYS, value="Tuesday", label="Appointment day", render=False)
    booked_hour = gr.Slider(6, 21, value=10, step=1, label="Hour booking was made", render=False)
    neighbourhood = gr.Dropdown(NEIGHBOURHOODS, value=default_hood, label="Clinic neighbourhood", render=False)
    sms = gr.Checkbox(value=True, label="SMS reminder sent", render=False)
    multiple_same_day = gr.Checkbox(value=False, label="Several appointments that day", render=False)
    age = gr.Slider(0, 100, value=35, step=1, label="Age", render=False)
    gender = gr.Radio(["Female", "Male"], value="Female", label="Sex", render=False)
    conditions = gr.CheckboxGroup(["Hypertension", "Diabetes", "Alcoholism", "Disability"],
                                  label="Recorded conditions", render=False)
    welfare = gr.Checkbox(value=False, label="Enrolled in welfare programme", render=False)
    prior_appts = gr.Number(value=0, precision=0, minimum=0, label="Previous appointments", render=False)
    prior_no_shows = gr.Number(value=0, precision=0, minimum=0, label="Previous no-shows", render=False)
    days_since_last = gr.Number(value=0, precision=0, minimum=0, label="Days since last visit", render=False)
    result = gr.HTML(render=False)
    chart = gr.Plot(show_label=False, render=False)
    ctx_state = gr.State(None)
    summary_out = gr.Markdown(elem_classes="pad-out", render=False)
    reminder_out = gr.Textbox(label="Draft message", lines=5, buttons=["copy"], render=False)
    chatbot = gr.Chatbot(height=340, show_label=False, render=False,
                         placeholder="Ask about this prediction, e.g. why is the risk high?")

    inputs = [lead_time, age, gender, weekday, booked_hour, sms, welfare, conditions,
              prior_appts, prior_no_shows, days_since_last, multiple_same_day, neighbourhood]

    with gr.Tabs():
        with gr.Tab("Risk predictor"):
            gr.Examples(
                label="Try an example patient",
                examples=EXAMPLES,
                example_labels=EXAMPLE_LABELS,
                inputs=inputs,
                outputs=[result, chart, ctx_state, summary_out, reminder_out, chatbot],
                fn=predict,
                run_on_click=True,
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-title">📅 Appointment</div>')
                    with gr.Group():
                        lead_time.render()
                        with gr.Row():
                            weekday.render()
                            booked_hour.render()
                        neighbourhood.render()
                        with gr.Row():
                            sms.render()
                            multiple_same_day.render()

                    gr.HTML('<div class="section-title">👤 Patient</div>')
                    with gr.Group():
                        with gr.Row():
                            age.render()
                            gender.render()
                        conditions.render()
                        welfare.render()

                    gr.HTML('<div class="section-title">🗂️ Attendance history</div>')
                    with gr.Group():
                        with gr.Row():
                            prior_appts.render()
                            prior_no_shows.render()
                            days_since_last.render()

                    btn = gr.Button("Predict risk", variant="primary", size="lg")

                with gr.Column(scale=6):
                    result.render()
                    chart.render()

            gr.HTML('<div class="section-title" style="margin-top:26px">✨ AI assistant</div>'
                    '<div class="muted" style="font-size:.85rem;margin:0 0 8px 2px">'
                    'An open-source LLM turns the model\'s output into plain language. It only explains the '
                    'prediction, it never makes it. Demo inputs only: never enter real patient data.</div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("**Summary for clinic staff**", elem_classes="pad")
                        summary_btn = gr.Button("✨ Generate summary", size="md", variant="secondary", elem_classes="ai-btn")
                        summary_out.render()
                    with gr.Group():
                        gr.Markdown("**Patient reminder**", elem_classes="pad")
                        channel = gr.Radio(["SMS", "Phone call script"], value="SMS", show_label=False)
                        reminder_btn = gr.Button("✉️ Draft reminder", size="md", variant="secondary", elem_classes="ai-btn")
                        reminder_out.render()
                with gr.Column(scale=6):
                    with gr.Group():
                        gr.Markdown("**Ask about this prediction**", elem_classes="pad")
                        chatbot.render()
                        with gr.Row():
                            chat_in = gr.Textbox(show_label=False, placeholder="Type a question...", scale=5)
                            send = gr.Button("Send", variant="primary", scale=1)
                    gr.Examples(
                        label="Example questions",
                        examples=[["Why is this patient's risk this level?"],
                                  ["What would lower the risk?"],
                                  ["How reliable is this model?"]],
                        inputs=[chat_in],
                    )

            outs = [result, chart, ctx_state, summary_out, reminder_out, chatbot]
            btn.click(predict, inputs=inputs, outputs=outs)
            demo.load(predict, inputs=inputs, outputs=outs)
            summary_btn.click(staff_summary, inputs=ctx_state, outputs=summary_out)
            reminder_btn.click(patient_reminder, inputs=[ctx_state, channel], outputs=reminder_out)
            send.click(chat, inputs=[chat_in, chatbot, ctx_state], outputs=[chat_in, chatbot])
            chat_in.submit(chat, inputs=[chat_in, chatbot, ctx_state], outputs=[chat_in, chatbot])

        with gr.Tab("About the model"):
            gr.Markdown(ABOUT)

    gr.Markdown(
        "<div style='text-align:center;opacity:.6;font-size:.85rem;margin-top:10px'>"
        "Built by Kwame Baafi Boateng · Python, XGBoost, SHAP, Gradio</div>"
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
