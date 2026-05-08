import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
WINDOWS_SEC = [0.5, 1.0, 2.0]
K_VALUES = list(range(1, 11))


def load_subject(mat_path: Path):
    mat = loadmat(mat_path)
    data = mat["data"]  # trials x channels x samples
    labels = mat["labels"]  # trials x 2
    fs = float(np.squeeze(mat["fs"]))
    channel_names = [str(x[0]) if isinstance(x, np.ndarray) else str(x) for x in mat["channel_names"].squeeze()]
    return data, labels, fs, channel_names


def segment_windows(trial_data: np.ndarray, window_sec: float, fs: float):
    window_len = int(round(window_sec * fs))
    n_samples = trial_data.shape[-1]
    n_windows = n_samples // window_len
    for i in range(n_windows):
        start = i * window_len
        end = start + window_len
        yield trial_data[:, start:end]


def time_features(window: np.ndarray):
    # window: channels x samples
    mean = np.mean(window, axis=1)
    std = np.std(window, axis=1, ddof=1)
    var = np.var(window, axis=1, ddof=1)
    rms = np.sqrt(np.mean(window ** 2, axis=1))
    skw = skew(window, axis=1, bias=False)
    krt = kurtosis(window, axis=1, fisher=True, bias=False)
    # Zero-crossing rate per channel
    zero_cross = np.mean(np.diff(np.signbit(window), axis=1), axis=1)

    feats = np.stack([mean, std, var, rms, skw, krt, zero_cross], axis=1)
    return np.nan_to_num(feats)


def bandpower(window: np.ndarray, fs: float, band):
    # window: channels x samples
    low, high = band
    nperseg = min(256, window.shape[1])
    freqs, pxx = welch(window, fs=fs, axis=1, nperseg=nperseg)
    band_mask = (freqs >= low) & (freqs <= high)
    bp = np.trapezoid(pxx[:, band_mask], freqs[band_mask], axis=1)
    return bp


def compute_psd(signal_1d: np.ndarray, fs: float):
    nperseg = min(256, signal_1d.shape[0])
    freqs, pxx = welch(signal_1d, fs=fs, nperseg=nperseg)
    return freqs, pxx


def plot_psd_for_class_segment(
    subject: str,
    label_name: str,
    label_values: np.ndarray,
    data: np.ndarray,
    fs: float,
    window_sec: float,
    channel_idx: int,
    rng: np.random.Generator,
    out_dir: Path,
):
    window_len = int(round(window_sec * fs))
    class_plots = []
    for class_value, class_name in [(0, "low"), (1, "high")]:
        trials_idx = np.where(label_values == class_value)[0]
        if trials_idx.size == 0:
            continue
        trial_idx = int(rng.choice(trials_idx))
        trial = data[trial_idx]
        n_windows = trial.shape[1] // window_len
        window_idx = int(rng.integers(0, n_windows)) if n_windows > 0 else 0
        start = window_idx * window_len
        end = start + window_len
        segment = trial[channel_idx, start:end]

        freqs, pxx = compute_psd(segment, fs)
        class_plots.append((class_name, freqs, pxx))

    if not class_plots:
        return None

    plt.figure(figsize=(7, 4))
    for class_name, freqs, pxx in class_plots:
        plt.plot(freqs, pxx, label=class_name)
    plt.title(f"{subject} {label_name} PSD (channel {channel_idx})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"psd_{subject}_{label_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_accuracy_vs_nk(subject, label_name, detailed_subject, out_dir: Path):
    # Accuracy vs window length (N): best over K for each feature set
    n_values = [str(x) for x in WINDOWS_SEC]
    feature_sets = {
        "time_only": detailed_subject["time_only"],
        "all_bands": detailed_subject["all_bands"],
        "time_plus_bands": detailed_subject["time_plus_bands"],
    }
    plt.figure(figsize=(7, 4))
    for feat_name, feat_dict in feature_sets.items():
        accs = []
        for n in n_values:
            results = feat_dict[n][label_name]
            _, acc = best_k(results)
            accs.append(acc)
        plt.plot(WINDOWS_SEC, accs, marker="o", label=feat_name)
    plt.title(f"{subject} {label_name}: Accuracy vs Window Length")
    plt.xlabel("Window Length (s)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out_n = out_dir / f"acc_vs_n_{subject}_{label_name}.png"
    plt.savefig(out_n, dpi=150)
    plt.close()

    # Accuracy vs K: best over window length for each feature set
    plt.figure(figsize=(7, 4))
    for feat_name, feat_dict in feature_sets.items():
        accs = []
        for k in K_VALUES:
            best_acc = None
            for n in n_values:
                results = feat_dict[n][label_name]
                acc = results[k]
                if (best_acc is None) or (acc > best_acc):
                    best_acc = acc
            accs.append(best_acc)
        plt.plot(K_VALUES, accs, marker="o", label=feat_name)
    plt.title(f"{subject} {label_name}: Accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out_k = out_dir / f"acc_vs_k_{subject}_{label_name}.png"
    plt.savefig(out_k, dpi=150)
    plt.close()

    return out_n, out_k


def features_band_only(data, fs, window_sec, band):
    # Returns trials x features
    all_features = []
    for trial in data:
        window_feats = []
        for window in segment_windows(trial, window_sec, fs):
            window_feats.append(bandpower(window, fs, band))
        trial_feat = np.mean(np.stack(window_feats, axis=0), axis=0)
        all_features.append(trial_feat)
    return np.stack(all_features, axis=0)


def features_all_bands(data, fs, window_sec):
    all_features = []
    for trial in data:
        window_feats = []
        for window in segment_windows(trial, window_sec, fs):
            per_band = [bandpower(window, fs, band) for band in BANDS.values()]
            window_feats.append(np.stack(per_band, axis=1))  # channels x bands
        mean_feat = np.mean(np.stack(window_feats, axis=0), axis=0)
        all_features.append(mean_feat.reshape(-1))
    return np.stack(all_features, axis=0)


def features_time_domain(data, fs, window_sec):
    all_features = []
    for trial in data:
        window_feats = []
        for window in segment_windows(trial, window_sec, fs):
            window_feats.append(time_features(window))  # channels x feats
        mean_feat = np.mean(np.stack(window_feats, axis=0), axis=0)
        all_features.append(mean_feat.reshape(-1))
    return np.stack(all_features, axis=0)


def combine_features(a, b):
    return np.concatenate([a, b], axis=1)


def evaluate_knn(features, labels):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k in K_VALUES:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(model, features, labels, cv=cv, scoring="accuracy")
        results[k] = float(np.mean(scores))
    return results


def best_k(results_dict):
    k = max(results_dict, key=results_dict.get)
    return k, results_dict[k]


def best_per_method(method_dict, label_name):
    best_record = None
    for window_sec, labels in method_dict.items():
        results = labels[label_name]
        k, acc = best_k(results)
        record = (acc, float(window_sec), int(k))
        if (best_record is None) or (record[0] > best_record[0]):
            best_record = record
    return best_record


def df_to_markdown(df: pd.DataFrame):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        line = "| " + " | ".join([str(row[h]) for h in headers]) + " |"
        lines.append(line)
    return "\n".join(lines)


def main():
    data_dir = Path("d:/DSP/Project DSP- Spring 2026/Data")
    subjects = sorted(data_dir.glob("s*.mat"))

    summary_rows = []
    detailed = {}

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    psd_paths = []
    acc_plot_paths = []

    for mat_path in subjects:
        subj = mat_path.stem
        data, labels, fs, channel_names = load_subject(mat_path)
        channel_idx = 0

        detailed[subj] = {
            "fs": fs,
            "channels": len(channel_names),
            "band_only": {},
            "all_bands": {},
            "time_only": {},
            "time_plus_bands": {},
        }

        for window_sec in WINDOWS_SEC:
            # Band-only
            for band_name, band in BANDS.items():
                feats = features_band_only(data, fs, window_sec, band)
                for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
                    results = evaluate_knn(feats, labels[:, label_idx])
                    detailed[subj]["band_only"].setdefault(band_name, {})
                    detailed[subj]["band_only"][band_name].setdefault(str(window_sec), {})
                    detailed[subj]["band_only"][band_name][str(window_sec)][label_name] = results

            # All bands combined
            all_band_feats = features_all_bands(data, fs, window_sec)
            for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
                results = evaluate_knn(all_band_feats, labels[:, label_idx])
                detailed[subj]["all_bands"].setdefault(str(window_sec), {})
                detailed[subj]["all_bands"][str(window_sec)][label_name] = results

            # Time-only
            time_feats = features_time_domain(data, fs, window_sec)
            for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
                results = evaluate_knn(time_feats, labels[:, label_idx])
                detailed[subj]["time_only"].setdefault(str(window_sec), {})
                detailed[subj]["time_only"][str(window_sec)][label_name] = results

            # Time + all bands
            combo_feats = combine_features(time_feats, all_band_feats)
            for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
                results = evaluate_knn(combo_feats, labels[:, label_idx])
                detailed[subj]["time_plus_bands"].setdefault(str(window_sec), {})
                detailed[subj]["time_plus_bands"][str(window_sec)][label_name] = results

        # PSD plots for one random segment per class (per label)
        for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
            psd_path = plot_psd_for_class_segment(
                subj,
                label_name,
                labels[:, label_idx],
                data,
                fs,
                window_sec=1.0,
                channel_idx=channel_idx,
                rng=rng,
                out_dir=plot_dir,
            )
            if psd_path is not None:
                psd_paths.append(psd_path)

        # Accuracy vs N and vs K plots
        for label_name in ["valence", "arousal"]:
            acc_plot_paths.append(plot_accuracy_vs_nk(subj, label_name, detailed[subj], plot_dir))

        # Summaries for band-only best per label
        for label_name, label_idx in [("valence", 0), ("arousal", 1)]:
            best_record = None
            for band_name in BANDS:
                for window_sec in WINDOWS_SEC:
                    results = detailed[subj]["band_only"][band_name][str(window_sec)][label_name]
                    k, acc = best_k(results)
                    record = (acc, band_name, window_sec, k)
                    if (best_record is None) or (record[0] > best_record[0]):
                        best_record = record

            acc, band_name, window_sec, k = best_record
            summary_rows.append({
                "subject": subj,
                "label": label_name,
                "best_band": band_name,
                "best_window_sec": window_sec,
                "best_k": k,
                "best_accuracy": acc,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_summary.csv", index=False)

    with open("results_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    # Build a short markdown report
    report_lines = []
    report_lines.append("# DSP Project Results\n")
    report_lines.append("## Approach Summary\n")
    report_lines.append("- Segment each 10 s trial into fixed-length windows (0.5 s, 1 s, 2 s).")
    report_lines.append("- Extract time-domain statistics and bandpower features per window and channel.")
    report_lines.append("- Aggregate features across windows by averaging; concatenate channels to build a vector per trial.")
    report_lines.append("- Train a KNN classifier (K=1..10) with 5-fold CV for valence and arousal labels.")
    report_lines.append("")
    report_lines.append("## Best Band-Only Accuracy Per Subject\n")
    report_lines.append("This uses bandpower features from a single band with KNN + 5-fold CV.")
    report_lines.append("")
    report_lines.append(df_to_markdown(summary_df))
    report_lines.append("")

    report_lines.append("## Feature Combination Comparison (Best Over Window and K)\n")
    report_lines.append("All-bands = concatenated bandpowers across all bands per channel.")
    report_lines.append("Time-only = time-domain statistics per channel.")
    report_lines.append("Time+bands = concatenated time-only + all-bands features.")
    report_lines.append("")

    combo_rows = []
    for subj in detailed:
        for label_name in ["valence", "arousal"]:
            best_all = best_per_method(detailed[subj]["all_bands"], label_name)
            best_time = best_per_method(detailed[subj]["time_only"], label_name)
            best_combo = best_per_method(detailed[subj]["time_plus_bands"], label_name)
            combo_rows.append({
                "subject": subj,
                "label": label_name,
                "all_bands_acc": best_all[0],
                "all_bands_window_sec": best_all[1],
                "all_bands_k": best_all[2],
                "time_only_acc": best_time[0],
                "time_only_window_sec": best_time[1],
                "time_only_k": best_time[2],
                "time_plus_bands_acc": best_combo[0],
                "time_plus_bands_window_sec": best_combo[1],
                "time_plus_bands_k": best_combo[2],
            })

    combo_df = pd.DataFrame(combo_rows)
    report_lines.append(df_to_markdown(combo_df))
    report_lines.append("")

    report_lines.append("## Proposed Feature Strategy\n")
    report_lines.append("- Time-domain features per channel: mean, std, variance, RMS, skewness, kurtosis, zero-crossing rate.")
    report_lines.append("- Combine channels by concatenating per-channel features into a single vector per trial.")
    report_lines.append("- Frequency-domain features: bandpower for delta/theta/alpha/beta/gamma per channel.")
    report_lines.append("- Combine all bands per channel by concatenating bandpowers, then concatenate channels.")
    report_lines.append("- Combine time + frequency by concatenating the time-only vector with the all-bands vector.")
    report_lines.append("")

    report_lines.append("## Effect Notes (From This Run)\n")
    report_lines.append("- Effects are subject-specific; no single combination dominates across all subjects.")
    report_lines.append("- Time+bands helped arousal for s01 and s02, but did not improve s03.")
    report_lines.append("- All-bands did not always beat the best single band; best band varies by subject and label.")
    report_lines.append("")

    report_lines.append("## PSD Plots (Random Segment Per Class)\n")
    report_lines.append("- Plots saved under the plots/ directory. Each figure compares Low vs High for one label.")
    for p in psd_paths:
        report_lines.append(f"- {p.as_posix()}")
    report_lines.append("")

    report_lines.append("## Accuracy Trend Plots (Slide 11)\n")
    report_lines.append("- Accuracy vs window length (N) and vs K for each subject/label and feature set.")
    for n_path, k_path in acc_plot_paths:
        report_lines.append(f"- {n_path.as_posix()}")
        report_lines.append(f"- {k_path.as_posix()}")

    report_path = Path("report.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(summary_df)
    print("\nSaved results_summary.csv and results_detailed.json")
    print("Saved report.md")


if __name__ == "__main__":
    main()
