---
title: Outpatient No-Show Risk Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
sdk_version: 6.28.0
short_description: Outpatient no-show risk predictor with XGBoost + SHAP
pinned: false
---

# Outpatient No-Show (DNA) Risk Predictor

A proof of concept that predicts which outpatient appointments are most likely to be missed, explains why, and uses an LLM assistant to turn each prediction into plain-English guidance, so clinics can target reminders and rebooking where they matter most.

**Trained on public data:** the Kaggle *Medical Appointment No Shows* dataset (110,527 appointments from public clinics in Brazil, 2016; created by Joni Hoppen and Aquarela Analytics). It is not NHS data and not for clinical or operational use. The next step would be to test the same approach on NHS outpatient data.

## Approach
- **Cleaning:** removed impossible ages and appointments dated before they were booked (11 rows).
- **Features:** booking lead time, same-day booking, weekday, patient age and recorded conditions, and each patient's attendance history, computed only from appointments on earlier dates to avoid data leakage.
- **Validation:** time-based split (train on earlier dates, test on the final week), mirroring how a Trust would score future appointments.
- **Models:** logistic regression baseline and XGBoost, explained with SHAP.

## Results (held-out final week)
| Model | ROC AUC | PR-AUC (random = 0.185) | No-shows reached by contacting top 20% |
|---|---|---|---|
| Logistic Regression | 0.728 | 0.338 | 37.2% |
| XGBoost | 0.732 | 0.344 | 37.6% |

- Targeting the 20% highest-risk appointments reaches about **twice as many no-shows** as random targeting.
- The lowest-risk 30% of appointments contain only about 5% of no-shows, so reminder effort there could be reduced.
- **Booking lead time is by far the strongest driver:** same-day bookings had a 4.6% no-show rate versus 24–34% for appointments booked a week or more ahead.
- Logistic regression performs almost as well as XGBoost, so a simpler, more transparent model may be preferable in practice.

## AI assistant (LLM layer)
An open-source LLM (`openai/gpt-oss-120b` via Hugging Face Inference Providers) adds three features on top of the model:
- **Staff summary:** turns the risk score and SHAP factors into a short plain-English summary with a suggested action.
- **Patient reminder:** drafts an SMS or phone call script using placeholders such as [Name] and [Clinic].
- **Q&A chat:** answers questions about the current prediction and how the model works.

**Design principle: the model predicts, the LLM only explains.** Guardrails:
- The LLM receives only the model's score and SHAP factors, and is instructed never to invent or change numbers.
- Patient reminders never mention risk scores, past missed appointments, age or health conditions.
- No clinical advice; answers outside the provided facts are declined.
- If the LLM is unavailable, the app falls back to template text instead of failing.
- Demo inputs only: real patient data should never be entered.

**Setup:** add a Hugging Face token with "Make calls to Inference Providers" permission as a Space secret named `HF_TOKEN`. Optionally set a Space variable `LLM_MODEL` to use a different model.

## Limitations
- The data covers about six weeks, so patient attendance history is short; with years of NHS records, history features would likely be stronger.
- The test set is a single week.
- SMS reminders were sent mainly for appointments booked further ahead, so their effect is confounded with lead time; measuring reminder impact needs a controlled trial.
- Neighbourhood likely reflects deprivation or travel distance. Risk scores should be used to offer extra support (reminders, transport help), never to penalise patients, for example through overbooking.
