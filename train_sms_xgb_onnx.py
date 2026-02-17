
"""
Train an XGBoost SMS spam classifier with TF-IDF features and export to ONNX
for on-device inference in the SignalFence Android app.

Outputs:
  - signalfence_xgb.onnx
  - signalfence_vocab.json
  - signalfence_idf.json
  - signalfence_meta.json
  - signalfence_xgb_metrics.json
"""
import json
import pathlib
import zipfile
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
TEST_SPLIT = 0.2
RANDOM_SEED = 42
MAX_FEATURES = 8000


def load_dataset():
    local_zip = BASE_DIR / "datasets" / "smsspamcollection.zip"
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    if not local_zip.exists():
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("latin-1").strip() for line in f]

    texts, labels = [], []
    for line in lines:
        if not line:
            continue
        label, text = line.split("\t", 1)
        texts.append(text)
        labels.append(1 if label == "spam" else 0)
    return np.array(texts), np.array(labels, dtype=np.int32)


def main():
    texts, labels = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        lowercase=True,
        ngram_range=(1, 1),
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # Threshold tune for best F1
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (probs >= t).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = (probs >= best_t).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(best_t), 2),
        "test_size": int(len(y_test)),
        "class_balance": {"spam": float((y_test == 1).mean()), "ham": float((y_test == 0).mean())},
    }

    (BASE_DIR / "signalfence_xgb_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Export vocab + idf
    vocab_raw = vectorizer.vocabulary_
    vocab = {str(k): int(v) for k, v in vocab_raw.items()}
    vocab_size = len(vocab)
    idf = vectorizer.idf_.tolist()
    idf_by_index = [0.0] * vocab_size
    for term, idx in vocab.items():
        idf_by_index[idx] = float(idf[idx])

    (BASE_DIR / "signalfence_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (BASE_DIR / "signalfence_idf.json").write_text(json.dumps(idf_by_index, indent=2), encoding="utf-8")

    meta = {
        "vocab_size": vocab_size,
        "lower": True,
        "ngram": 1,
        "threshold": float(best_t),
    }
    (BASE_DIR / "signalfence_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Export ONNX
    initial_type = [("input", FloatTensorType([None, vocab_size]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    with open(BASE_DIR / "signalfence_xgb.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Exported signalfence_xgb.onnx")
    print(metrics)


if __name__ == "__main__":
    main()
