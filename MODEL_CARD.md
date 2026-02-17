# Model Card — SignalFence XGBoost SMS Classifier

## Overview
- **Model Type**: XGBoost (gradient-boosted decision trees)
- **Task**: Binary classification (spam vs ham)
- **Features**: TF-IDF unigram features
- **Export**: ONNX (for Android inference)

## Training Data
- **Source**: UCI SMS Spam Collection (public dataset)
- **Limitations**: Small, old, English-only dataset. Not representative of modern global SMS traffic.

## Intended Use
- On-device spam filtering for SignalFence Android app.
- **Not** for high-stakes or safety-critical decisions.

## Metrics
- Metrics are produced in `signalfence_xgb_metrics.json`.
- Production thresholds must be validated with real-world traffic.

## Risks & Mitigations
- **False positives**: May block legitimate messages.
- **Domain shift**: Modern scams evolve faster than static datasets.
- **Mitigation**: Periodic retraining and human-in-the-loop review.

## Ethical Considerations
- No personal data should be stored or transmitted.
- Keep all inference on-device.

