"""
All visualizations in Plotly: Signal Explorer, Complexity Analysis, Classification.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

from .data_loader import GROUP_ORDER

# Color scheme: Healthy=green, FTD=amber, Alzheimer's=red
GROUP_COLORS = {"Healthy": "#2ecc71", "FTD": "#f39c12", "Alzheimer's": "#e74c3c"}


def raw_trace_plot(times: np.ndarray, signal: np.ndarray, title: str = "EEG trace", channel: str = ""):
    """Plotly line chart: time vs amplitude."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=signal, mode="lines", name=channel or "EEG", line=dict(width=1)))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=40, l=50, r=30),
    )
    return fig


def psd_plot(freqs: np.ndarray, psd: np.ndarray, title: str = "Power spectral density"):
    """Plotly line: frequency vs power."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name="PSD", line=dict(width=1.5)))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=40, l=50, r=30),
    )
    return fig


def spectrogram_plot(times: np.ndarray, freqs: np.ndarray, Sxx: np.ndarray, title: str = "Spectrogram"):
    """Plotly heatmap: time x frequency, color = power (log)."""
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=freqs,
            z=10 * np.log10(Sxx + 1e-12),
            colorscale="Viridis",
            colorbar=dict(title="Power (dB)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=40, l=50, r=60),
    )
    return fig


def compute_psd(signal: np.ndarray, sfreq: float, nperseg: int = 1024):
    """Welch PSD; returns freqs, psd."""
    if scipy_signal is None:
        return np.array([]), np.array([])
    freqs, psd = scipy_signal.welch(signal, fs=sfreq, nperseg=min(nperseg, len(signal) // 4 or 256))
    return freqs, psd


def compute_spectrogram(signal: np.ndarray, sfreq: float, nperseg: int = 256):
    """STFT-based spectrogram; returns times, freqs, Sxx."""
    if scipy_signal is None:
        return np.array([]), np.array([]), np.array([[0]])
    nperseg = min(nperseg, len(signal) // 2 or 128)
    f, t, Sxx = scipy_signal.spectrogram(signal, fs=sfreq, nperseg=nperseg)
    return t, f, Sxx


def psd_figure_from_signal(signal: np.ndarray, sfreq: float, title: str = "Power spectral density"):
    """Build PSD plot from raw signal and sampling frequency."""
    freqs, psd = compute_psd(signal, sfreq)
    return psd_plot(freqs, psd, title=title)


def spectrogram_figure_from_signal(signal: np.ndarray, sfreq: float, title: str = "Spectrogram"):
    """Build spectrogram plot from raw signal and sampling frequency."""
    t, f, Sxx = compute_spectrogram(signal, sfreq)
    return spectrogram_plot(t, f, Sxx, title=title)


def radar_chart(df: pd.DataFrame, subject_id: str = None, normalize: bool = True):
    """
    Radar with three axes: SampEn, HFD, DFA. One trace per subject or one per group.
    If subject_id given, plot that subject; else plot mean per group (or all subjects).
    """
    if df.empty or not all(c in df.columns for c in ["SampEn", "HFD", "DFA", "group"]):
        return go.Figure()
    if subject_id:
        row = df[df["subject_id"] == subject_id].iloc[0]
        r = [row["SampEn"], row["HFD"], row["DFA"]]
        name = f"{subject_id} ({row['group']})"
    else:
        # Mean per group
        agg = df.groupby("group")[["SampEn", "HFD", "DFA"]].mean()
        if agg.empty:
            return go.Figure()
        r = agg.loc[agg.index[0]].tolist()
        name = agg.index[0]
    if normalize and df.shape[0] > 1:
        low = df[["SampEn", "HFD", "DFA"]].min().values
        high = df[["SampEn", "HFD", "DFA"]].max().values
        span = high - low
        span[span == 0] = 1
        r = (np.array(r) - low) / span
    else:
        r = np.array(r)
    categories = ["SampEn", "HFD", "DFA"]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=list(r) + [r[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
        )
    )
    if not subject_id and not df[df["group"] == name].empty:
        color = GROUP_COLORS.get(name, "#888")
        fig.update_traces(fillcolor=color, line=dict(color=color))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1] if normalize else None)),
        template="plotly_white",
        height=400,
        title="Complexity measures (radar)",
        showlegend=True,
    )
    return fig


def complexity_gradient_plot(df: pd.DataFrame, plot_type: str = "violin"):
    """
    Key figure: SampEn, HFD, DFA by group (Healthy, FTD, Alzheimer's) with green/amber/red.
    plot_type: 'violin' or 'strip'. Order groups as Healthy -> FTD -> Alzheimer's.
    """
    if df.empty or not all(c in df.columns for c in ["SampEn", "HFD", "DFA", "group"]):
        return go.Figure()
    order = [g for g in GROUP_ORDER if g in df["group"].unique()]
    if not order:
        order = list(df["group"].unique())
    long = df.melt(id_vars=["subject_id", "group"], value_vars=["SampEn", "HFD", "DFA"], var_name="Measure", value_name="Value")
    long["group"] = pd.Categorical(long["group"], categories=order, ordered=True)
    long = long.sort_values("group")
    color_map = {g: GROUP_COLORS.get(g, "#888") for g in order}
    if plot_type == "strip":
        fig = px.strip(
            long,
            x="group",
            y="Value",
            color="group",
            facet_col="Measure",
            color_discrete_map=color_map,
            category_orders={"group": order},
        )
    else:
        fig = px.violin(
            long,
            x="group",
            y="Value",
            color="group",
            facet_col="Measure",
            box=True,
            points="all",
            color_discrete_map=color_map,
            category_orders={"group": order},
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        title="Complexity measures across groups: Healthy → FTD → Alzheimer's",
        showlegend=False,
        xaxis_title="",
        margin=dict(t=60, b=50),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    return fig


def confusion_matrix_plot(cm: np.ndarray, classes: list, title: str = "Confusion matrix"):
    """Plotly heatmap for confusion matrix with annotations."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_white",
        height=400,
    )
    return fig


def feature_importance_plot(importance_dict: dict, title: str = "Feature importance"):
    """Plotly bar chart: feature name vs importance."""
    names = list(importance_dict.keys())
    vals = list(importance_dict.values())
    fig = go.Figure(data=go.Bar(x=names, y=vals, marker_color="steelblue"))
    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Importance",
        template="plotly_white",
        height=320,
    )
    return fig


def prediction_proba_plot(proba_dict: dict, predicted: str):
    """Bar chart of class probabilities for 'classify new subject'."""
    classes = list(proba_dict.keys())
    probs = list(proba_dict.values())
    colors = [GROUP_COLORS.get(c, "#95a5a6") for c in classes]
    fig = go.Figure(data=go.Bar(x=classes, y=probs, marker_color=colors, text=[f"{p:.2f}" for p in probs], textposition="outside"))
    fig.update_layout(
        title=f"Predicted: {predicted}",
        xaxis_title="Class",
        yaxis_title="Probability",
        template="plotly_white",
        height=320,
        yaxis=dict(range=[0, 1.1]),
    )
    return fig
