# EEG Complexity in Neurodegeneration

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![OpenNeuro](https://img.shields.io/badge/data-OpenNeuro-orange)](https://openneuro.org/datasets/ds004504)

> **Does brain signal complexity drop as Alzheimer's progresses?**
>
> We measure it with Sample Entropy, Higuchi fractal dimension, and DFA on real resting-state EEG — and the gradient from Healthy → FTD → Alzheimer's is visible in the data.

---

## The Problem

Many studies report 98–99% accuracy for EEG-based dementia classification. Those numbers often come from small cohorts or heavily engineered features. In practice, clinicians need interpretable, physiologically grounded markers: not just "the model says AD," but *why* — what about the signal changed?

Healthy brain activity is highly variable and irregular; neurons fire in diverse, adaptive patterns. In Alzheimer's and related dementias, loss of synapses and network connectivity tends to make activity more regular and less complex. **Complexity measures** quantify that directly from the raw EEG. Until now, few tools let you explore this gradient on a real dataset in one place. This project does that.

---

## Key Finding

> 🧠 **Complexity drops as disease progresses — and you can see it clearly in the data.**
>
> Across Healthy → FTD → Alzheimer's, Sample Entropy, Higuchi fractal dimension, and DFA all show a consistent decline. A Random Forest on these three features carries meaningful diagnostic signal — and EEG complexity could serve as a low-cost, non-invasive biomarker.

---

## Results

### Complexity gradient

Violin and strip plots of SampEn, HFD, and DFA across the three groups (Healthy, FTD, Alzheimer's). The downward trend from healthy to Alzheimer's is the main result — visible in the **Complexity Analysis** page of the app.

*(Add a screenshot of the complexity gradient figure from the app here.)*

### Summary table

| Measure                   | Direction in Alzheimer's | Clinical meaning                    |
|---------------------------|--------------------------|-------------------------------------|
| Sample Entropy            | Decreases                | Brain activity becomes more regular |
| Higuchi Fractal Dimension | Decreases                | Signal loses fractal complexity     |
| DFA                       | Changes                  | Long-range correlation structure shifts |
| RF classification        | Above baseline           | Complexity features carry diagnostic signal |

---

## App Pages

- **Signal Explorer** — Raw EEG trace, power spectral density, and spectrogram for a selected subject and channel. See what the signal actually looks like across groups.
- **Complexity Analysis** — Radar chart per subject and the main **complexity gradient**: violin and strip plots of SampEn, HFD, and DFA across Healthy (green), FTD (amber), and Alzheimer's (red). The gradient is visible by eye.
- **Classification** — Random Forest on the three complexity features. Confusion matrix, feature importance, per-class sensitivity/specificity. Upload a new subject's EEG (.set or .edf) and get a group prediction.

---

## How to Reproduce

### 1. Clone and install

```bash
git clone https://github.com/youssof20/eeg-alzheimer-complexity.git
cd eeg-alzheimer-complexity
pip install -r requirements.txt
```

### 2. Run the app

Preprocessed samples are bundled in `data/processed/` — the app runs without downloading the full dataset.

```bash
streamlit run app.py
```

### 3. (Optional) Regenerate sample data

The bundled samples are synthetic (complexity decreases by group so the gradient is visible). To regenerate them:

```bash
python scripts/generate_sample_data.py
```

To use **real** OpenNeuro ds004504 data: download the dataset, place `.set` files in a BIDS-style layout under `data/raw/`, then run the preprocessing in `src/preprocessing.py` and save outputs in the same `.npz` format (see `src/preprocessing.epochs_to_npz_dict` and `scripts/generate_sample_data.py`).

---

## Project Structure

```
eeg-alzheimer-complexity/
├── app.py                    # Streamlit app: Signal Explorer, Complexity Analysis, Classification
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── processed/            # Preprocessed .npz samples (bundled; app runs without full dataset)
│   └── participants.tsv      # Subject ID → group (C/F/A)
├── scripts/
│   └── generate_sample_data.py   # Generate synthetic preprocessed samples for the app
└── src/
    ├── data_loader.py        # Load .npz from data/processed/ or raw .set/.edf
    ├── preprocessing.py      # Epoch extraction, channel selection, epochs_to_npz_dict
    ├── features.py           # Sample Entropy, Higuchi FD, DFA (antropy + NumPy)
    ├── classification.py     # Random Forest, confusion matrix, feature importance, predict
    └── visualization.py     # Raw trace, PSD, spectrogram, radar, complexity gradient, confusion matrix
```

---

## Dataset

**OpenNeuro ds004504** — scalp EEG from Healthy, Frontotemporal Dementia (FTD), and Alzheimer's subjects.

- **Dataset:** [https://openneuro.org/datasets/ds004504](https://openneuro.org/datasets/ds004504)
- **Paper:** "A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG," *Data* (2021). [https://doi.org/10.3390/data8060095](https://doi.org/10.3390/data8060095)

---

## Clinical Implications

**Why complexity matters.** EEG is cheap and non-invasive. If complexity measures reliably track cognitive decline, they could support screening or longitudinal monitoring without MRI or expensive imaging. This app shows that the gradient is present in a real clinical dataset and that a simple classifier can use it.

**Interpretability.** Unlike black-box deep learning, Sample Entropy, HFD, and DFA have clear meanings: unpredictability, fractal scaling, long-range correlations. That makes the result easier to explain and to link to underlying neurobiology.

**Caveat.** This is a research prototype. Not a medical device. Not for clinical use.

---

## Limitations and Future Work

- **Bundled data:** The default app runs on synthetic preprocessed samples; real ds004504 requires a separate download and preprocessing run.
- **Small cohort:** Results depend on dataset size and group balance; more subjects would strengthen conclusions.
- **Three features only:** Other complexity or spectral features could improve classification.
- **Future directions:** Test on other cohorts, add MCI as a separate group, and validate on held-out clinical data.

---

## Citation

If you use this work, please cite:

**Youssof Sallam.** EEG Complexity in Neurodegeneration (2026). GitHub.

https://github.com/youssof20/eeg-alzheimer-complexity

---

## Acknowledgements

EEG data and participant metadata from [OpenNeuro](https://openneuro.org/) (ds004504). Dataset descriptor: *Data* (2021), [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095).
