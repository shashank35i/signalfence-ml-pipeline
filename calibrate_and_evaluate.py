import json, math, argparse, pathlib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# -------- helpers --------
def load_tokenizer(tok_path):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json(pathlib.Path(tok_path).read_text(encoding="utf-8"))

def pad(tok, texts, max_len):
    seq = tok.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

def sigmoid(x): return 1 / (1 + np.exp(-x))

def ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i + 1])
        if idx.sum() == 0:
            continue
        conf = probs[idx].mean()
        acc = labels[idx].mean()
        ece += probs[idx].size / probs.size * abs(acc - conf)
    return ece

def wilson_ci(p, n, z=1.96):
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - margin)/denom), min(1, (centre + margin)/denom)


def ensure_dataset(path: pathlib.Path):
    if path.exists():
        return path
    zip_path = pathlib.Path("datasets/smsspamcollection.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("latin-1").strip() for line in f]
    rows = []
    for line in lines:
        if not line:
            continue
        label, text = line.split("\t", 1)
        lbl = "1" if label == "spam" else "0"
        rows.append(lbl + "," + text.replace("\n", " ").replace("\r", " "))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="signalfence_spam_model.tflite")
    ap.add_argument("--tokenizer", default="signalfence_tokenizer.json")
    ap.add_argument("--meta", default="signalfence_meta.json")
    ap.add_argument("--data", default="data.csv", help="CSV with columns: label,text (label: ham=0, spam=1)")
    ap.add_argument("--text", required=True, help="Single message to score")
    ap.add_argument("--threshold_mode", default="f1", choices=["f1","youden"])
    args = ap.parse_args()

    meta = json.loads(pathlib.Path(args.meta).read_text())
    tok = load_tokenizer(args.tokenizer)
    max_len = meta["max_len"]

    # load data (download UCI SMS Spam if missing)
    data_path = ensure_dataset(pathlib.Path(args.data))
    labels, texts = [], []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lbl, txt = line.split(",", 1)
        labels.append(int(lbl.strip()))
        texts.append(txt.strip())
    labels = np.array(labels, dtype=np.int32)

    X = pad(tok, texts, max_len)
    x_train, x_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]

    def infer(batch):
        arr = batch.astype(in_detail["dtype"])
        outputs = []
        arr2d = arr if arr.ndim > 1 else arr[None, :]
        for row in arr2d:
            row = row.reshape(1, -1)
            interpreter.resize_tensor_input(in_detail["index"], row.shape, strict=True)
            interpreter.allocate_tensors()
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], row)
            interpreter.invoke()
            out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).squeeze()
            outputs.append(out)
        return np.array(outputs)

    val_logits = infer(x_val)

    # Platt scaling
    logit = np.log(val_logits.clip(1e-6, 1-1e-6)/(1-val_logits.clip(1e-6,1-1e-6)))
    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit.reshape(-1,1), y_val)
    val_cal = platt.predict_proba(logit.reshape(-1,1))[:,1]

    # Metrics
    ece_val = ece(val_cal, y_val)
    brier = brier_score_loss(y_val, val_cal)

    # Threshold tuning
    threshs = np.linspace(0.05, 0.95, 19)
    best_t, best_f1, best_youden = 0.5, -1, -1
    for t in threshs:
        p = val_cal >= t
        tp = (p & (y_val==1)).sum()
        fp = (p & (y_val==0)).sum()
        fn = ((~p) & (y_val==1)).sum()
        tn = ((~p) & (y_val==0)).sum()
        f1 = 2*tp / max(1, 2*tp+fp+fn)
        youden = tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9) - 1
        if f1 > best_f1:
            best_f1, best_t = f1, t
        best_youden = t if youden > best_youden else best_youden
    use_t = best_t if args.threshold_mode=="f1" else best_youden

    # Score the given text
    x_single = pad(tok, [args.text], max_len)
    raw_val = infer(x_single)
    raw = float(np.squeeze(raw_val))
    raw = min(max(raw, 1e-6), 1-1e-6)
    raw_logit = np.log(raw/(1-raw))
    cal = platt.predict_proba([[raw_logit]])[0,1]
    lo, hi = wilson_ci(cal, n=len(y_val))

    print(f"Raw prob: {raw:.3f}")
    print(f"Calibrated prob: {cal:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print(f"ECE: {ece_val:.3f}  Brier: {brier:.3f}")
    print(f"Suggested threshold ({args.threshold_mode}): {use_t:.2f}")
    print(f"Class balance in val: spam {y_val.mean():.3f}")
    print(f"Decision at threshold -> {'SPAM' if cal>=use_t else 'HAM'}")

if __name__ == "__main__":
    main()
