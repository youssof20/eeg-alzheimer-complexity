"""
EEG Complexity in Neurodegeneration — Streamlit app entry point.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np

from src.data_loader import load_processed_folder, load_raw_and_preprocess
from src.features import build_features_table, extract_subject_features
from src.classification import (
    train_random_forest,
    confusion_matrix_and_metrics,
    feature_importance,
    predict_subject,
)
from src.visualization import (
    raw_trace_plot,
    psd_figure_from_signal,
    spectrogram_figure_from_signal,
    radar_chart,
    complexity_gradient_plot,
    confusion_matrix_plot,
    feature_importance_plot,
    prediction_proba_plot,
)

st.set_page_config(
    page_title="EEG Complexity in Neurodegeneration",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
PARTICIPANTS_TSV = ROOT / "data" / "participants.tsv"


@st.cache_data
def load_data():
    """Load all preprocessed samples and return list of records."""
    return load_processed_folder(str(PROCESSED_DIR))


@st.cache_data
def get_features_table(records):
    """Build subject-level features DataFrame from loaded records."""
    if not records:
        return pd.DataFrame()
    return build_features_table(records)


# Load once
records = load_data()
features_df = get_features_table(records)

# Sidebar
st.sidebar.header("Controls")
group_filter = st.sidebar.selectbox(
    "Filter by group",
    options=["All", "Healthy", "FTD", "Alzheimer's"],
    index=0,
    key="sidebar_group",
)
channel_options = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
channel = st.sidebar.selectbox("Channel", options=channel_options, index=0, key="sidebar_channel")

# Title and subtitle (main area)
st.title("EEG Complexity in Neurodegeneration")
st.caption("Exploring fractal signatures of Alzheimer's disease progression")

# Subject list filtered by group
if records:
    subject_options = [
        r["subject_id"] for r in records
        if group_filter == "All" or r["group"] == group_filter
    ]
    if not subject_options:
        subject_options = [r["subject_id"] for r in records]
    selected_subject = st.sidebar.selectbox("Subject", options=subject_options, index=0, key="sidebar_subject")
else:
    subject_options = []
    selected_subject = None

# Page navigation
page = st.radio(
    "Page",
    options=["Signal Explorer", "Complexity Analysis", "Classification"],
    horizontal=True,
    key="page_selector",
)

# ---------- Page 1: Signal Explorer ----------
if page == "Signal Explorer":
    st.subheader("Signal Explorer")
    if not records or selected_subject is None:
        st.warning("No data loaded. Add preprocessed .npz files in `data/processed/` and restart.")
    else:
        rec = next((r for r in records if r["subject_id"] == selected_subject), None)
        if rec is None:
            st.warning("Selected subject not found.")
        else:
            ch_names = rec["ch_names"]
            ch_idx = ch_names.index(channel) if channel in ch_names else 0
            data = rec["data"]
            sfreq = rec["sfreq"]
            # One channel, all epochs concatenated
            signal = data[:, ch_idx, :].reshape(-1)
            times = np.arange(len(signal)) / sfreq
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    raw_trace_plot(times, signal, title=f"EEG — {selected_subject} ({channel})", channel=channel),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    psd_figure_from_signal(signal, sfreq, title=f"PSD — {channel}"),
                    use_container_width=True,
                )
            st.plotly_chart(
                spectrogram_figure_from_signal(signal, sfreq, title=f"Spectrogram — {channel}"),
                use_container_width=True,
            )
            st.markdown("---")
            st.markdown(
                "**What you're seeing:** The top plot is the EEG voltage over time from one scalp channel. "
                "The power spectral density (PSD) shows how much power the signal has at each frequency; "
                "in healthy adults, alpha (8–12 Hz) often stands out in eyes-closed rest. "
                "The spectrogram shows how that power changes over time. "
                "These views help spot artifacts and basic rhythm changes that can differ between healthy and diseased brains."
            )

# ---------- Page 2: Complexity Analysis ----------
elif page == "Complexity Analysis":
    st.subheader("Complexity Analysis")
    if features_df.empty:
        st.warning("No feature data. Ensure `data/processed/` contains .npz files and restart.")
    else:
        # Radar for selected subject
        if selected_subject and str(selected_subject) in features_df["subject_id"].astype(str).values:
            st.subheader("Subject complexity profile")
            fig_radar = radar_chart(features_df, subject_id=selected_subject, normalize=True)
            st.plotly_chart(fig_radar, use_container_width=True)
        st.subheader("Complexity across groups")
        fig_gradient = complexity_gradient_plot(features_df, plot_type="violin")
        st.plotly_chart(fig_gradient, use_container_width=True)
        fig_strip = complexity_gradient_plot(features_df, plot_type="strip")
        st.plotly_chart(fig_strip, use_container_width=True)
        st.markdown("---")
        st.markdown(
            "**What you're seeing:** The radar shows one subject's three complexity measures (Sample Entropy, "
            "Higuchi fractal dimension, DFA) on the same scale. The plots below compare all subjects: "
            "Healthy (green), FTD (amber), Alzheimer's (red). In many studies, brain signal complexity "
            "decreases as disease progresses—healthy brains show more irregular, adaptive activity; "
            "in dementia, activity can become more regular and less complex. This gradient is the kind of "
            "signature that could one day support research into early markers."
        )

# ---------- Page 3: Classification ----------
else:
    st.subheader("Classification")
    if features_df.empty or len(features_df) < 2:
        st.warning("Need at least 2 subjects with features to train the model.")
        clf, classes = None, []
    else:
        clf, X, y, classes = train_random_forest(features_df)
        cm, metrics, acc = confusion_matrix_and_metrics(clf, X, y, classes)
        imp = feature_importance(clf)
        st.metric("Accuracy", f"{acc:.2%}")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(confusion_matrix_plot(cm, classes), use_container_width=True)
        with col2:
            st.plotly_chart(feature_importance_plot(imp), use_container_width=True)
        st.subheader("Per-class metrics")
        for c in classes:
            m = metrics.get(c, {})
            st.write(f"**{c}** — Sensitivity: {m.get('sensitivity', 0):.2%}, Specificity: {m.get('specificity', 0):.2%}")
        st.markdown("---")
        st.subheader("Classify new subject")
        uploaded = st.file_uploader("Upload an EEG file (.set or .edf)", type=["set", "edf"])
        if uploaded is not None:
            with st.spinner("Preprocessing and extracting features…"):
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    rec = load_raw_and_preprocess(tmp_path, subject_id="uploaded", group=None)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                if rec is None:
                    st.error("Could not load or preprocess the file. Check format and channels.")
                else:
                    feats = extract_subject_features(rec["data"], rec.get("ch_names"))
                    pred, proba = predict_subject(clf, feats, classes)
                    if pred is not None:
                        st.success(f"Predicted group: **{pred}**")
                        st.plotly_chart(prediction_proba_plot(proba, pred), use_container_width=True)
                    else:
                        st.error("Prediction failed (missing or invalid features).")
        st.markdown("---")
        st.markdown(
            "**What you're seeing:** The model is a Random Forest trained on the three complexity measures. "
            "The confusion matrix shows how often each true group was predicted correctly. "
            "Uploading your own file runs the same preprocessing and feature steps, then gives a predicted "
            "group and confidence. **This app is a research prototype only and must not be used for diagnosis.**"
        )

# Title and subtitle at top (above radio)
st.sidebar.markdown("---")
st.sidebar.caption("EEG Complexity in Neurodegeneration — Research prototype")
