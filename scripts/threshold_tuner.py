"""Threshold tuner for the hybrid fracture detector

This script runs the hybrid model's raw predictions on the validation set,
then performs a grid search over simple decision rules and thresholds to
maximize accuracy. It reports the best rule and evaluates it on the test set.

Usage:
    python tools/threshold_tuner.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pathlib import Path
import json
from src.final_hybrid_model import FinalHybridFractureDetector

# Dataset paths
VAL_DIR = Path("new Dataset/val")
TEST_DIR = Path("new Dataset/test")

def gather_predictions(detector, folder):
    records = []
    for label_name, label_idx in [("fracture", 0), ("no_fracture", 1)]:
        dirp = folder / label_name
        if not dirp.exists():
            continue
        for p in sorted(dirp.glob("*.jpg")):
            res = detector.predict(p.as_posix())
            # store raw model outputs
            records.append({
                "path": p.as_posix(),
                "label": 0 if label_name == "fracture" else 1,
                "yolo_pred": res.get("yolo_result", {}).get("predicted_class", None),
                "yolo_conf": res.get("yolo_result", {}).get("confidence", None),
                "frac_pred": res.get("fracnet_result", {}).get("predicted_class", None),
                "frac_conf": res.get("fracnet_result", {}).get("confidence", None),
            })
    return records


def apply_rule(record, yolo_thr, frac_thr, mode):
    # Normalize: for the models, 0 may map to fracture or not; we will use the detector's outputs
    # Mode options:
    #  - "yolo_only": predict fracture if yolo_conf>=yolo_thr and yolo_pred==0 (fracture)
    #  - "frac_only": predict fracture if frac_conf>=frac_thr and frac_pred==1 (frac net uses 1=fracture in final_hybrid_model)
    #  - "either": predict fracture if either model predicts fracture above threshold
    #  - "both": predict fracture only if both predict fracture above thresholds
    yolo_is_fracture = (record["yolo_pred"] == 0)
    frac_is_fracture = (record["frac_pred"] == 1)

    yolo_ok = (record["yolo_conf"] is not None and record["yolo_conf"] >= yolo_thr and yolo_is_fracture)
    frac_ok = (record["frac_conf"] is not None and record["frac_conf"] >= frac_thr and frac_is_fracture)

    if mode == "yolo_only":
        return 0 if yolo_ok else 1
    if mode == "frac_only":
        return 0 if frac_ok else 1
    if mode == "either":
        return 0 if (yolo_ok or frac_ok) else 1
    if mode == "both":
        return 0 if (yolo_ok and frac_ok) else 1
    # default conservative: require both OR yolo>very_high
    if mode == "conservative":
        if yolo_ok and frac_ok:
            return 0
        if yolo_ok and record["yolo_conf"] >= 0.95:
            return 0
        return 1
    return 1


def evaluate(records, yolo_thr, frac_thr, mode):
    correct = 0
    for r in records:
        pred = apply_rule(r, yolo_thr, frac_thr, mode)
        if pred == r["label"]:
            correct += 1
    return correct / len(records)


def grid_search(val_records):
    best = {"acc": 0}
    modes = ["yolo_only", "frac_only", "either", "both", "conservative"]
    yolo_thrs = [round(x,2) for x in [0.5,0.6,0.7,0.8,0.9,0.95]]
    frac_thrs = [round(x,2) for x in [0.5,0.6,0.7,0.8,0.9]]
    # Add weighted fusion search: weights applied to yolo vs fracnet probabilities
    weight_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    score_thrs = [0.4, 0.5, 0.6]
    for mode in modes:
        for yt in yolo_thrs:
            for ft in frac_thrs:
                acc = evaluate(val_records, yt, ft, mode)
                if acc > best["acc"]:
                    best = {"acc": acc, "mode": mode, "yolo_thr": yt, "frac_thr": ft}
    # Weighted fusion: treat model confidences as fracture probability
    for w in weight_vals:
        for st in score_thrs:
            # evaluate weighted fusion as a special mode
            acc = evaluate_weighted(val_records, w, st)
            if acc > best["acc"]:
                best = {"acc": acc, "mode": "weighted", "weight_yolo": w, "score_thr": st}
    return best


def compute_model_fracture_prob(record):
    """Compute fracture probability for each model from stored predicted_class and predicted confidence."""
    # YOLO: predicted_class==0 means predicted fracture
    yolo_pred = record.get("yolo_pred")
    yolo_conf = record.get("yolo_conf")
    if yolo_conf is None or yolo_pred is None:
        yolo_prob = 0.5
    else:
        try:
            yolo_prob = float(yolo_conf) if int(yolo_pred) == 0 else (1.0 - float(yolo_conf))
        except Exception:
            yolo_prob = 0.5

    # FracNet: frac_pred==1 indicates fracture in our pipeline
    frac_pred = record.get("frac_pred")
    frac_conf = record.get("frac_conf")
    if frac_conf is None or frac_pred is None:
        frac_prob = 0.5
    else:
        try:
            frac_prob = float(frac_conf) if int(frac_pred) == 1 else (1.0 - float(frac_conf))
        except Exception:
            frac_prob = 0.5

    return yolo_prob, frac_prob


def evaluate_weighted(records, weight_yolo, score_thr):
    """Evaluate weighted fusion where final score = w*yolo_prob + (1-w)*frac_prob"""
    correct = 0
    for r in records:
        yolo_prob, frac_prob = compute_model_fracture_prob(r)
        score = weight_yolo * yolo_prob + (1.0 - weight_yolo) * frac_prob
        pred = 0 if score >= score_thr else 1
        if pred == r["label"]:
            correct += 1
    return correct / len(records)


def main():
    detector = FinalHybridFractureDetector()
    print("Gathering validation predictions...")
    val_records = gather_predictions(detector, VAL_DIR)
    print(f"Validation images: {len(val_records)}")

    print("Running grid search on validation set...")
    best = grid_search(val_records)
    print("Best on val:", best)

    print("Gathering test predictions...")
    test_records = gather_predictions(detector, TEST_DIR)
    print(f"Test images: {len(test_records)}")

    # Evaluate test accuracy using the correct method for the best mode
    if best["mode"] == "weighted":
        test_acc = evaluate_weighted(test_records, best["weight_yolo"], best["score_thr"])
    else:
        test_acc = evaluate(test_records, best["yolo_thr"], best["frac_thr"], best["mode"])
    print("Test accuracy with best val params:", test_acc)

    out = {"best_val": best, "test_acc": test_acc}
    # Ensure output directory exists
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "threshold_tuning_results.json"
    with open(out_path.as_posix(), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path.as_posix()}")

if __name__ == '__main__':
    main()
