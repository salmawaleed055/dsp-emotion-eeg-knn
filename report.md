# DSP Project Results

## Approach Summary

- Segment each 10 s trial into fixed-length windows (0.5 s, 1 s, 2 s).
- Extract time-domain statistics and bandpower features per window and channel.
- Aggregate features across windows by averaging; concatenate channels to build a vector per trial.
- Train a KNN classifier (K=1..10) with 5-fold CV for valence and arousal labels.

## Best Band-Only Accuracy Per Subject

This uses bandpower features from a single band with KNN + 5-fold CV.

| subject | label | best_band | best_window_sec | best_k | best_accuracy |
| --- | --- | --- | --- | --- | --- |
| s01 | valence | delta | 0.5 | 4 | 0.85 |
| s01 | arousal | gamma | 2.0 | 5 | 0.875 |
| s02 | valence | theta | 1.0 | 4 | 0.875 |
| s02 | arousal | alpha | 2.0 | 7 | 0.9 |
| s03 | valence | theta | 0.5 | 3 | 0.7 |
| s03 | arousal | beta | 2.0 | 5 | 0.775 |

## Feature Combination Comparison (Best Over Window and K)

All-bands = concatenated bandpowers across all bands per channel.
Time-only = time-domain statistics per channel.
Time+bands = concatenated time-only + all-bands features.

| subject | label | all_bands_acc | all_bands_window_sec | all_bands_k | time_only_acc | time_only_window_sec | time_only_k | time_plus_bands_acc | time_plus_bands_window_sec | time_plus_bands_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s01 | valence | 0.85 | 0.5 | 4 | 0.85 | 0.5 | 3 | 0.85 | 0.5 | 4 |
| s01 | arousal | 0.75 | 0.5 | 2 | 0.8 | 1.0 | 4 | 0.825 | 1.0 | 2 |
| s02 | valence | 0.775 | 0.5 | 4 | 0.775 | 0.5 | 4 | 0.775 | 0.5 | 2 |
| s02 | arousal | 0.875 | 0.5 | 2 | 0.875 | 0.5 | 2 | 0.9 | 0.5 | 3 |
| s03 | valence | 0.75 | 0.5 | 2 | 0.775 | 1.0 | 7 | 0.75 | 0.5 | 2 |
| s03 | arousal | 0.8 | 1.0 | 8 | 0.775 | 1.0 | 9 | 0.775 | 1.0 | 5 |

## Proposed Feature Strategy

- Time-domain features per channel: mean, std, variance, RMS, skewness, kurtosis, zero-crossing rate.
- Combine channels by concatenating per-channel features into a single vector per trial.
- Frequency-domain features: bandpower for delta/theta/alpha/beta/gamma per channel.
- Combine all bands per channel by concatenating bandpowers, then concatenate channels.
- Combine time + frequency by concatenating the time-only vector with the all-bands vector.

## Effect Notes (From This Run)

- Effects are subject-specific; no single combination dominates across all subjects.
- Time+bands helped arousal for s01 and s02, but did not improve s03.
- All-bands did not always beat the best single band; best band varies by subject and label.

## PSD Plots (Random Segment Per Class)

- Plots saved under the plots/ directory. Each figure compares Low vs High for one label.
- plots/psd_s01_valence.png
- plots/psd_s01_arousal.png
- plots/psd_s02_valence.png
- plots/psd_s02_arousal.png
- plots/psd_s03_valence.png
- plots/psd_s03_arousal.png

## Accuracy Trend Plots (Slide 11)

- Accuracy vs window length (N) and vs K for each subject/label and feature set.
- plots/acc_vs_n_s01_valence.png
- plots/acc_vs_k_s01_valence.png
- plots/acc_vs_n_s01_arousal.png
- plots/acc_vs_k_s01_arousal.png
- plots/acc_vs_n_s02_valence.png
- plots/acc_vs_k_s02_valence.png
- plots/acc_vs_n_s02_arousal.png
- plots/acc_vs_k_s02_arousal.png
- plots/acc_vs_n_s03_valence.png
- plots/acc_vs_k_s03_valence.png
- plots/acc_vs_n_s03_arousal.png
- plots/acc_vs_k_s03_arousal.png

## Per-Band Accuracy Plots

- Per-band accuracy vs window length and vs K for each subject/label.
- plots/acc_vs_n_bands_s01_valence.png
- plots/acc_vs_k_bands_s01_valence.png
- plots/acc_vs_n_bands_s01_arousal.png
- plots/acc_vs_k_bands_s01_arousal.png
- plots/acc_vs_n_bands_s02_valence.png
- plots/acc_vs_k_bands_s02_valence.png
- plots/acc_vs_n_bands_s02_arousal.png
- plots/acc_vs_k_bands_s02_arousal.png
- plots/acc_vs_n_bands_s03_valence.png
- plots/acc_vs_k_bands_s03_valence.png
- plots/acc_vs_n_bands_s03_arousal.png
- plots/acc_vs_k_bands_s03_arousal.png

## Notes

The original DEAP labels are on a 1-9 scale for valence and arousal. We binarize at 4.5: values > 4.5 map to High = 1, values <= 4.5 map to Low = 0.
All loaded .mat files were already binary; no runtime binarization was required.