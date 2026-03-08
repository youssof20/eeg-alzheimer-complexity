# EEG Complexity in Neurodegeneration

A tool that measures **how "chaotic" brain activity is** across Healthy, Frontotemporal Dementia (FTD), and Alzheimer's subjects — and uses that to classify cognitive decline from raw EEG.

Built on OpenNeuro ds004504 — real clinical resting-state EEG recordings.

![App Screenshot](https://github.com/user-attachments/assets/c09b2892-bbd3-42bd-aec5-4ebbf9f67672)

---

## The Idea

Healthy brains are unpredictable in a good way. Neurons fire in diverse, constantly shifting patterns — high complexity. As Alzheimer's disease destroys neural connections, the brain's activity becomes more regular, more repetitive, less complex. You can measure this collapse directly from EEG.

This project quantifies that gradient using three mathematical measures and visualizes it across disease groups.

---

## What I Found

**Complexity drops as disease progresses — and you can see it clearly in the data.**

Across three groups (Healthy → FTD → Alzheimer's), all three complexity measures show a consistent decline:

- **Sample Entropy** — measures signal unpredictability. Lower in Alzheimer's.
- **Higuchi Fractal Dimension** — measures signal self-similarity across scales. Lower in Alzheimer's.
- **DFA (Detrended Fluctuation Analysis)** — measures long-range correlations. Changes with disease state.

A Random Forest classifier trained on these features can distinguish the three groups with meaningful accuracy, and the feature importance analysis shows which complexity measures carry the most diagnostic signal.

**The clinical implication:** EEG complexity could serve as a low-cost, non-invasive biomarker for tracking neurodegeneration — no MRI, no expensive imaging required.

---

## App Pages

**Signal Explorer**
Raw EEG trace, power spectral density, and spectrogram — see what the signal actually looks like across groups.

**Complexity Analysis**
Radar chart per subject + the main finding: violin and strip plots of SampEn, HFD, and DFA across Healthy (green), FTD (amber), and Alzheimer's (red). The gradient is visible by eye.

**Classification**
Random Forest on extracted complexity features. Confusion matrix, feature importance, per-class sensitivity/specificity. Upload a new subject's EEG and get a group prediction.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Preprocessed samples are bundled in `data/processed/` — the app runs without downloading the full dataset.

---

## Dataset

OpenNeuro ds004504 — scalp EEG from Healthy, FTD, and Alzheimer's subjects.
- Dataset: https://openneuro.org/datasets/ds004504
- Paper: https://doi.org/10.3390/data8060095

---

## Key Results Summary

| Measure | Direction in Alzheimer's | Clinical Meaning |
|---|---|---|
| Sample Entropy | Decreases | Brain activity becomes more predictable |
| Higuchi Fractal Dimension | Decreases | Signal loses fractal complexity |
| DFA | Changes | Long-range correlation structure shifts |
| RF Classification | Above baseline | Complexity features carry diagnostic signal |

---

## Status

Research prototype. Not a medical device. Not for clinical use.

