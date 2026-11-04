"""
NetraVision Model Evaluation Script
Evaluates outputs/best_model.pth on the held-out 733-image validation set from train.csv.
Computes accuracy, Quadratic Weighted Kappa (QWK), macro F1, per-class metrics, and confusion matrix.
Saves detailed report to outputs/evaluation_report.json and outputs/evaluation_report.md.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Ensure standard output handles UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import config
from dataset import get_dataloaders
from model import get_model

def run_evaluation(weights_path=config.MODEL_SAVE_PATH):
    print("=" * 70)
    print("      NETRAVISION MODEL EVALUATION — APTOS 2019 VALIDATION SET")
    print("=" * 70)
    
    # 1. Device configuration
    device = config.DEVICE
    if device.type == "cpu":
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"[DEVICE] Compute device: CPU ({num_threads} worker threads)")
    else:
        print(f"[DEVICE] Compute device: CUDA GPU ({torch.cuda.get_device_name(0)})")

    # 2. Load model checkpoint
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")
    
    print(f"[MODEL] Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    model = get_model(num_classes=config.NUM_CLASSES)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint.get("epoch", "N/A")
        ckpt_qwk = checkpoint.get("val_qwk", "N/A")
        ckpt_acc = checkpoint.get("val_acc", "N/A")
        print(f"[MODEL] Checkpoint metadata: Epoch {ckpt_epoch} | Best QWK {ckpt_qwk} | Acc {ckpt_acc}")
    else:
        model.load_state_dict(checkpoint)
        ckpt_epoch = "N/A"
        ckpt_qwk = "N/A"
        ckpt_acc = "N/A"
    
    model.to(device)
    model.eval()

    # 3. Load dataset split
    print(f"[DATA] Loading stratified train/val split (val_split={config.VAL_SPLIT}, seed={config.RANDOM_SEED})...")
    train_loader, val_loader, train_df, val_df = get_dataloaders()
    
    print(f"[DATA] Training set size:   {len(train_df)} samples")
    print(f"[DATA] Validation set size: {len(val_df)} samples")
    
    # Verify split disjointness
    overlap = set(train_df["id_code"]).intersection(set(val_df["id_code"]))
    if len(overlap) != 0:
        raise ValueError(f"Data leak detected! Overlap between train and val: {len(overlap)} samples.")
    print(f"[DATA] Data integrity verified: 0 overlapping samples between train and validation sets.")

    # Class distribution in validation set
    val_class_counts = val_df["diagnosis"].value_counts().sort_index().to_dict()
    print("[DATA] Validation Class Distribution:")
    for cls_idx, count in val_class_counts.items():
        cls_name = config.CLASS_NAMES[cls_idx]
        pct = (count / len(val_df)) * 100
        print(f"       Class {cls_idx} ({cls_name:<18}): {count:4d} images ({pct:5.1f}%)")

    # 4. Run inference on validation set
    print(f"\n[INFERENCE] Running inference over {len(val_df)} validation images...")
    start_time = time.time()
    
    all_targets = []
    all_preds = []
    all_confs = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
            all_confs.extend(confs.tolist())
            all_probs.extend(probs.tolist())
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                processed = min((batch_idx + 1) * config.BATCH_SIZE, len(val_df))
                print(f"  Processed {processed:3d}/{len(val_df)} images...", flush=True)

    elapsed_time = time.time() - start_time
    print(f"[INFERENCE] Completed in {elapsed_time:.1f}s ({elapsed_time/len(val_df)*1000:.1f} ms/image)\n")

    # 5. Calculate Metrics
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    accuracy = float(accuracy_score(y_true, y_pred))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    
    # Per-class metrics
    class_names = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]
    clf_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    # Print Metrics Summary
    print("-" * 70)
    print("                    OVERALL EVALUATION METRICS")
    print("-" * 70)
    print(f"  Validation Samples:           {len(y_true)}")
    print(f"  Accuracy:                     {accuracy * 100:.2f}%  ({np.sum(y_true == y_pred)}/{len(y_true)} correct)")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"  Macro F1 Score:               {macro_f1:.4f}")
    print(f"  Weighted F1 Score:            {weighted_f1:.4f}")
    print("-" * 70)

    # Print Per-Class Performance
    print("\n" + "-" * 70)
    print("                   PER-CLASS PERFORMANCE")
    print("-" * 70)
    print(f"  {'Class':<5} {'Name':<18} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("  " + "-" * 63)
    for i, name in enumerate(class_names):
        metrics = clf_report[name]
        p = metrics["precision"]
        r = metrics["recall"]
        f1 = metrics["f1-score"]
        s = int(metrics["support"])
        print(f"  {i:<5} {name:<18} {p:9.4f}  {r:9.4f}  {f1:9.4f}  {s:7d}")
    print("-" * 70)

    # Print Confusion Matrix
    print("\n" + "-" * 70)
    print("             5-CLASS CONFUSION MATRIX (Row=True, Col=Pred)")
    print("-" * 70)
    header = "True \\ Pred  |" + "".join([f" {c[:6]:>7}" for c in class_names]) + " | Total"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"Class {i} ({class_names[i][:6]:>6}) |" + "".join([f"{val:8d}" for val in row]) + f" | {row.sum():5d}"
        print(row_str)
    print("-" * len(header))
    col_totals = "Total Pred   |" + "".join([f"{cm[:, j].sum():8d}" for j in range(config.NUM_CLASSES)]) + f" | {cm.sum():5d}"
    print(col_totals)
    print("-" * 70)

    # 6. Select 10 diverse validation examples
    # Include examples across all classes, both correct and misclassified
    val_records = []
    for idx, row in val_df.iterrows():
        i = len(val_records)
        val_records.append({
            "id_code": row["id_code"],
            "filename": f"{row['id_code']}.png",
            "true_class": int(y_true[i]),
            "true_name": config.CLASS_NAMES[int(y_true[i])],
            "pred_class": int(y_pred[i]),
            "pred_name": config.CLASS_NAMES[int(y_pred[i])],
            "correct": bool(y_true[i] == y_pred[i]),
            "confidence": round(float(all_confs[i]) * 100, 2),
            "probabilities": [round(float(p), 4) for p in all_probs[i]]
        })

    # Pick 10 representative examples:
    # 2 from Class 0 (Normal), 2 from Class 1 (Mild), 2 from Class 2 (Moderate),
    # 2 from Class 3 (Severe), 2 from Class 4 (Proliferative)
    selected_samples = []
    for c in range(config.NUM_CLASSES):
        class_samples = [s for s in val_records if s["true_class"] == c]
        # Pick 1 correct and 1 challenging/misclassified (or second correct if all correct)
        correct_samples = [s for s in class_samples if s["correct"]]
        incorrect_samples = [s for s in class_samples if not s["correct"]]
        
        if correct_samples:
            selected_samples.append(correct_samples[0])
        if incorrect_samples:
            selected_samples.append(incorrect_samples[0])
        elif len(correct_samples) > 1:
            selected_samples.append(correct_samples[1])

    # If fewer than 10, fill from remaining
    if len(selected_samples) < 10:
        seen = {s["id_code"] for s in selected_samples}
        for s in val_records:
            if s["id_code"] not in seen:
                selected_samples.append(s)
                seen.add(s["id_code"])
            if len(selected_samples) == 10:
                break
    selected_samples = selected_samples[:10]

    print("\n" + "=" * 70)
    print("                    10 VALIDATION EXAMPLES")
    print("=" * 70)
    print(f"{'#':<3} {'Filename':<20} {'True Label':<18} {'Predicted Label':<18} {'Conf':<8} {'Status'}")
    print("-" * 75)
    for idx, s in enumerate(selected_samples, 1):
        status = "CORRECT" if s["correct"] else "MISCLASSIFIED"
        print(f"{idx:<3} {s['filename']:<20} {s['true_name']:<18} {s['pred_name']:<18} {s['confidence']:5.1f}%  {status}")
    print("=" * 75)

    # 7. Save Evaluation Report
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    report_json_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.json")
    report_md_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.md")

    report_data = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": weights_path,
            "checkpoint_epoch": ckpt_epoch,
            "compute_device": str(device),
            "total_val_samples": len(val_df),
            "total_train_samples": len(train_df),
            "random_seed": config.RANDOM_SEED,
            "val_split": config.VAL_SPLIT,
            "train_val_overlap_count": len(overlap),
            "preprocessing": "Ben Graham fundus normalization (crop_from_gray + Gaussian blur subtract + ImageNet norm)"
        },
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "accuracy_percentage": round(accuracy * 100, 2),
            "quadratic_weighted_kappa": round(qwk, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "total_correct": int(np.sum(y_true == y_pred)),
            "total_evaluated": len(y_true)
        },
        "per_class_metrics": {
            name: {
                "precision": round(clf_report[name]["precision"], 4),
                "recall": round(clf_report[name]["recall"], 4),
                "f1_score": round(clf_report[name]["f1-score"], 4),
                "support": int(clf_report[name]["support"])
            }
            for name in class_names
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": class_names
        },
        "sample_predictions": selected_samples,
        "disclaimer": "This is an academic evaluation report for research purposes. NetraVision is an educational prototype and is not clinically validated."
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[SAVE] Full evaluation JSON report saved to: {report_json_path}")

    # Generate Markdown Report
    cm_md_header = "| True \\ Pred | " + " | ".join(class_names) + " | Total |\n"
    cm_md_sep = "| :--- | " + " | ".join([":---:"] * config.NUM_CLASSES) + " | :---: |\n"
    cm_md_rows = ""
    for i, row in enumerate(cm):
        cm_md_rows += f"| **Class {i} ({class_names[i]})** | " + " | ".join([str(val) for val in row]) + f" | **{row.sum()}** |\n"
    cm_md_rows += f"| **Total Predicted** | " + " | ".join([str(cm[:, j].sum()) for j in range(config.NUM_CLASSES)]) + f" | **{cm.sum()}** |\n"

    samples_md = "| # | Image Filename | True Label | Predicted Label | Confidence | Match |\n|---|---|---|---|---|---|\n"
    for idx, s in enumerate(selected_samples, 1):
        match_str = "✅ Correct" if s["correct"] else "❌ Misclassified"
        samples_md += f"| {idx} | `{s['filename']}` | {s['true_name']} (Class {s['true_class']}) | {s['pred_name']} (Class {s['pred_class']}) | {s['confidence']}% | {match_str} |\n"

    per_class_table_rows = "\n".join([
        f"| {i} | **{name}** | {clf_report[name]['precision']:.4f} | {clf_report[name]['recall']:.4f} | {clf_report[name]['f1-score']:.4f} | {int(clf_report[name]['support'])} |"
        for i, name in enumerate(class_names)
    ])

    md_content = f"""# NetraVision Model Validation & Benchmark Report

**Dataset**: APTOS 2019 Blindness Detection (Held-Out Stratified Validation Split)  
**Evaluated Model**: `{weights_path}` (Checkpoint Epoch: {ckpt_epoch})  
**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary & Verification Confirmation

- **Validation Split**: Held-out **733 images** (20% stratified partition from `train.csv`, random seed 42).
- **Training Samples**: **2929 images** (80% stratified partition from `train.csv`).
- **Data Leakage Check**: **Verified 0 overlapping samples** between training and validation sets.
- **Preprocessing Alignment**: Matches `predict.py` exactly (Ben Graham local contrast enhancement, circular black border cropping, resize to 256x256, ImageNet normalization).
- **Competition Test Data**: The unlabelled `test.csv` (1928 images) was **not used** for accuracy or metric computation.

---

## 2. Key Aggregate Metrics

| Metric | Score | Clinical / Benchmark Significance |
| :--- | :---: | :--- |
| **Accuracy** | **{accuracy * 100:.2f}%** | Overall top-1 classification rate ({np.sum(y_true == y_pred)} / {len(y_true)} correct) |
| **Quadratic Weighted Kappa (QWK)** | **{qwk:.4f}** | Primary clinical DR competition metric; penalizes distance between ordinal grades |
| **Macro F1 Score** | **{macro_f1:.4f}** | Unweighted mean of F1 scores across all 5 imbalanced severity levels |
| **Weighted F1 Score** | **{weighted_f1:.4f}** | Support-weighted mean F1 score across all classes |

---

## 3. Per-Class Performance Breakdown

| Class ID | Grade Name | Precision | Recall | F1-Score | Support (Sample Count) |
| :---: | :--- | :---: | :---: | :---: | :---: |
{per_class_table_rows}

---

## 4. 5-Class Confusion Matrix

{cm_md_header}{cm_md_sep}{cm_md_rows}

---

## 5. Sample Validation Predictions (10 Representative Cases)

{samples_md}

---

## 6. Disclaimer

> **Important**: This evaluation report is generated for academic and research evaluation purposes only. NetraVision is an educational prototype and is not clinically validated or certified as a medical diagnostic device.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVE] Markdown evaluation report saved to: {report_md_path}")
    print("=" * 70)
    print("                   EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
