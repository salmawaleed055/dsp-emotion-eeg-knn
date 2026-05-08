# EEG Emotion Recognition using KNN (CSCE 3611 DSP Project)

Emotion classification from EEG signals using time-domain and frequency-domain features with K-Nearest Neighbors.

## Project Overview

This project implements emotion recognition from EEG data by classifying valence (positive/negative) and arousal (calm/excited) using:
- **Time-domain features**: mean, std, variance, RMS, skewness, kurtosis, zero-crossing rate
- **Frequency-domain features**: bandpower across delta/theta/alpha/beta/gamma bands
- **Classifier**: KNN with k=1..10, evaluated via 5-fold cross-validation

## Data

- **Location**: `Data/` folder
- **Files**: `s01.mat`, `s02.mat`, `s03.mat` (3 subjects)
- **Contents**: 40 trials per subject, 32 EEG channels, 10 s per trial
- **Labels**: Valence (0=Low, 1=High) and Arousal (0=Low, 1=High)

## Setup

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install numpy scipy scikit-learn pandas matplotlib pypdf
```

## Usage

```bash
python run_project.py
```

This will:
1. Load all 3 subjects
2. Segment trials into windows (0.5 s, 1 s, 2 s)
3. Extract time and frequency features
4. Train KNN classifiers (K=1..10) with 5-fold CV
5. Generate plots and reports

## Outputs

### Files
- **report.md** — Summary and analysis with tables and figure paths
- **results_summary.csv** — Best band, window, K, and accuracy per subject/label
- **results_detailed.json** — Complete results for all band/window/K combinations

### Plots (plots/ directory)
- **psd_s0X_valence.png, psd_s0X_arousal.png** — PSD comparison (Low vs High) for one random segment
- **acc_vs_n_s0X_valence.png, acc_vs_n_s0X_arousal.png** — Accuracy vs window length
- **acc_vs_k_s0X_valence.png, acc_vs_k_s0X_arousal.png** — Accuracy vs K value

## Key Results

### Best Single-Band Accuracy Per Subject

| Subject | Valence Band | Valence Acc | Arousal Band | Arousal Acc |
|---------|--------------|-------------|--------------|-------------|
| s01     | delta (0.5s) | 0.850       | gamma (2.0s) | 0.875       |
| s02     | theta (1.0s) | 0.875       | alpha (2.0s) | 0.900       |
| s03     | theta (0.5s) | 0.700       | beta (2.0s)  | 0.775       |

### Feature Combination Effects

- **Time+Bands** improved arousal classification for s01 and s02 but had minimal effect on s03
- No single feature set dominated across all subjects; **subject-specific tuning is beneficial**
- **Best window length varies**: 0.5 s and 1.0 s were more common than 2.0 s
- **Optimal K** typically 2–7 depending on subject, label, and feature set

## Project Structure

```
d:\DSP\Project DSP- Spring 2026\
├── run_project.py          # Main analysis script
├── report.md               # Results summary and analysis
├── results_summary.csv     # Best accuracies per subject
├── results_detailed.json   # Full results for all configurations
├── plots/                  # PSD and accuracy trend figures
│   ├── psd_*.png
│   ├── acc_vs_n_*.png
│   └── acc_vs_k_*.png
├── Data/                   # EEG data files
│   ├── s01.mat
│   ├── s02.mat
│   └── s03.mat
├── README.md               # This file
└── requirements.txt        # Python dependencies (optional)
```

## Notes

- Features are **scaled per trial** using StandardScaler before KNN to ensure fair comparison
- Cross-validation uses **stratified splitting** to maintain class balance
- Results are **deterministic** (random seed = 42)
- PSD plots use **Welch's method** with 256-sample windows for spectral estimation

## References

- **Slide 11** (Emotion.pdf): Algorithm overview and KNN classifier details
- **Project Description** (Project Description.pdf): Full requirements and deliverables
- Welch, P. D. (1967). "The use of fast Fourier transform for estimation of power spectra"

## Authors

CSCE 3611 – Digital Signal Processing, Spring 2026
The American University in Cairo
