# NetraVision Model Validation & Benchmark Report

**Dataset**: APTOS 2019 Blindness Detection (Held-Out Stratified Validation Split)  
**Evaluated Model**: `outputs/best_model.pth` (Checkpoint Epoch: 13)  
**Evaluation Date**: 2026-11-03 13:57:35  

---

## 1. Executive Summary & Verification Confirmation

- **Validation Split**: Held-out **733 images** (20% stratified partition from `train.csv`, random seed 42).
- **Training Samples**: **2929 images** (80% stratified partition from `train.csv`).
- **Data Leakage Check**: **Verified 0 overlapping samples** between training and validation sets (`len(set(train) & set(val)) == 0`).
- **Preprocessing Alignment**: Matches `predict.py` exactly (Ben Graham local contrast enhancement, circular black border cropping, resize to 256x256, ImageNet normalization).
- **Competition Test Data**: The unlabelled `test.csv` (1928 images) was **not used** for accuracy or metric computation.

---

## 2. Key Aggregate Metrics

| Metric | Score | Clinical / Benchmark Significance |
| :--- | :---: | :--- |
| **Accuracy** | **72.99%** | Overall top-1 classification rate (535 / 733 correct) |
| **Quadratic Weighted Kappa (QWK)** | **0.8310** | Primary clinical DR competition metric; penalizes distance between ordinal grades |
| **Macro F1 Score** | **0.5766** | Unweighted mean of F1 scores across all 5 imbalanced severity levels |
| **Weighted F1 Score** | **0.7418** | Support-weighted mean F1 score across all classes |

---

## 3. Per-Class Performance Breakdown

| Class ID | Grade Name | Precision | Recall | F1-Score | Support (Sample Count) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **No DR** | 0.9624 | 0.9224 | 0.9420 | 361 |
| 1 | **Mild** | 0.4000 | 0.6216 | 0.4868 | 74 |
| 2 | **Moderate** | 0.7303 | 0.5550 | 0.6307 | 200 |
| 3 | **Severe** | 0.2794 | 0.4872 | 0.3551 | 39 |
| 4 | **Proliferative DR** | 0.5000 | 0.4407 | 0.4685 | 59 |

---

## 4. 5-Class Confusion Matrix

| True \ Pred | No DR | Mild | Moderate | Severe | Proliferative DR | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class 0 (No DR)** | 333 | 18 | 6 | 3 | 1 | **361** |
| **Class 1 (Mild)** | 10 | 46 | 15 | 0 | 3 | **74** |
| **Class 2 (Moderate)** | 3 | 43 | 111 | 31 | 12 | **200** |
| **Class 3 (Severe)** | 0 | 1 | 9 | 19 | 10 | **39** |
| **Class 4 (Proliferative DR)** | 0 | 7 | 11 | 15 | 26 | **59** |
| **Total Predicted** | **346** | **115** | **152** | **68** | **52** | **733** |

---

## 5. Sample Validation Predictions (10 Representative Cases)

| # | Image Filename | True Label | Predicted Label | Confidence | Match |
|---|---|---|---|---|---|
| 1 | `6fe67fd7f5d1.png` | No DR (Class 0) | No DR (Class 0) | 65.5% | ✅ Correct |
| 2 | `d57d1be1bbd1.png` | No DR (Class 0) | Mild (Class 1) | 49.4% | ❌ Misclassified |
| 3 | `eba3acc42197.png` | Mild (Class 1) | Mild (Class 1) | 64.4% | ✅ Correct |
| 4 | `82ac8463fadd.png` | Mild (Class 1) | No DR (Class 0) | 99.3% | ❌ Misclassified |
| 5 | `1594ca6c30d3.png` | Moderate (Class 2) | Moderate (Class 2) | 43.7% | ✅ Correct |
| 6 | `44855f666225.png` | Moderate (Class 2) | Severe (Class 3) | 62.3% | ❌ Misclassified |
| 7 | `7b29e3783919.png` | Severe (Class 3) | Severe (Class 3) | 76.5% | ✅ Correct |
| 8 | `7a6e384a0846.png` | Severe (Class 3) | Moderate (Class 2) | 27.3% | ❌ Misclassified |
| 9 | `2017cd92c63d.png` | Proliferative DR (Class 4) | Proliferative DR (Class 4) | 82.4% | ✅ Correct |
| 10 | `61bbc11fe503.png` | Proliferative DR (Class 4) | Severe (Class 3) | 45.5% | ❌ Misclassified |

---

## 6. Disclaimer

> **Important**: This evaluation report is generated for academic and research evaluation purposes only. NetraVision is an educational prototype and is not clinically validated or certified as a medical diagnostic device.
