# 💎 Stage 0 LITE — Evaluation Report

## Headline Numbers

| Metric | Baseline (regex) | Fine-tuned (Qwen+LoRA) | Δ |
|---|---|---|---|
| **Overall F1** | 0.527 | 0.335 | -0.192 |
| **SG/MY F1** | 0.650 | 0.420 | -0.230 |
| **Cases evaluated** | 25 | 25 | — |

## Per-Field Breakdown

| Field | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Name | 0.680 `██████████░░░░░` | 0.240 `███░░░░░░░░░░░░` | 🔴 -0.440 |
| Phone | 0.920 `█████████████░░` | 0.400 `██████░░░░░░░░░` | 🔴 -0.520 |
| Email | 0.920 `█████████████░░` | 0.440 `██████░░░░░░░░░` | 🔴 -0.480 |
| Current Location | 0.560 `████████░░░░░░░` | 0.280 `████░░░░░░░░░░░` | 🔴 -0.280 |
| Current Company | 0.040 `░░░░░░░░░░░░░░░` | 0.080 `█░░░░░░░░░░░░░░` | ⚪ +0.040 |
| Current Title | 0.000 `░░░░░░░░░░░░░░░` | 0.080 `█░░░░░░░░░░░░░░` | 🟢 +0.080 |
| Function | 1.000 `███████████████` | 0.440 `██████░░░░░░░░░` | 🔴 -0.560 |
| Industry | 1.000 `███████████████` | 0.720 `██████████░░░░░` | 🔴 -0.280 |
| Language Skills | 0.360 `█████░░░░░░░░░░` | 0.360 `█████░░░░░░░░░░` | ⚪ +0.000 |
| Certifications | 0.960 `██████████████░` | 0.960 `██████████████░` | ⚪ +0.000 |
| hard_skills/tags | 0.203 `███░░░░░░░░░░░░` | 0.148 `██░░░░░░░░░░░░░` | 🔴 -0.055 |
| soft_skills/skills | 0.320 `████░░░░░░░░░░░` | 0.456 `██████░░░░░░░░░` | 🟢 +0.136 |
| Work Experience | 0.000 `░░░░░░░░░░░░░░░` | 0.058 `░░░░░░░░░░░░░░░` | 🟢 +0.058 |
| Education | 0.219 `███░░░░░░░░░░░░` | 0.125 `█░░░░░░░░░░░░░░` | 🔴 -0.095 |

## 🎯 Decision Verdict

### 🛑 DO NOT SCALE
SG/MY F1 lift of -0.230 is negative.
Fine-tuning hurt SG/MY recognition. Investigate corruption issues before any scale-up.
