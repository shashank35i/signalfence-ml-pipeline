<div align="center">
  <h1>SignalFence ML Pipeline</h1>
  <p>Production-grade training and export pipeline for SignalFence’s on-device XGBoost SMS spam model.</p>
  <p>
    <img src="https://img.shields.io/badge/Model-XGBoost-1f8f5f" alt="Model" />
    <img src="https://img.shields.io/badge/Export-ONNX-4f5b93" alt="Export" />
    <img src="https://img.shields.io/badge/Runtime-ONNX%20Runtime%20Mobile-0b74de" alt="Runtime" />
    <img src="https://img.shields.io/badge/Status-Active-0aa06e" alt="Status" />
  </p>
  <p><strong>Built by Shashank Preetham Pendyala</strong></p>
</div>

---

## Overview

This repository contains the **training, evaluation, and export pipeline** for SignalFence’s on-device spam detector. The model is an **XGBoost gradient-boosted decision tree** classifier trained on TF-IDF features. It exports to ONNX for Android inference via ONNX Runtime Mobile.

---

## Key Outputs

| Artifact | Purpose |
| --- | --- |
| `signalfence_xgb.onnx` | XGBoost model exported to ONNX for Android inference. |
| `signalfence_vocab.json` | Token vocabulary mapping for TF-IDF. |
| `signalfence_idf.json` | IDF weights for TF-IDF vectorization. |
| `signalfence_meta.json` | Vectorizer metadata (vocab size, threshold). |
| `signalfence_xgb_metrics.json` | Accuracy, precision, recall, F1, and threshold. |

---

## Learning Method

- **Algorithm**: XGBoost (gradient-boosted decision trees)
- **Features**: TF-IDF unigram features
- **Objective**: Binary classification (spam vs ham)
- **Export**: ONNX for on-device inference

---

## Project Structure

```text
.
|-- train_sms_xgb_onnx.py         # XGBoost training + ONNX export
|-- calibrate_and_evaluate.py     # Optional calibration + threshold tuning
|-- signalfence_xgb.onnx          # Exported model
|-- signalfence_vocab.json        # TF-IDF vocab
|-- signalfence_idf.json          # IDF weights
|-- signalfence_meta.json         # Vectorizer metadata
|-- signalfence_xgb_metrics.json  # Metrics report
|-- MODEL_CARD.md
|-- PRODUCTION_READINESS.md
|-- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- `xgboost`, `scikit-learn`, `onnxmltools`, `onnx`

### Install

```bash
pip install xgboost scikit-learn onnxmltools onnx
```

### Train + Export

```bash
python train_sms_xgb_onnx.py
```

This will generate the ONNX model and supporting assets.

---

## Export to Android

Copy the artifacts to the Android app:

```text
app/src/main/assets/
  - signalfence_xgb.onnx
  - signalfence_vocab.json
  - signalfence_idf.json
  - signalfence_meta.json
  - signalfence_xgb_metrics.json
```

---

## License

MIT License. See [LICENSE](LICENSE).
