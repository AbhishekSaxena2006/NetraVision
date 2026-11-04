# NetraVision — AI-Powered Diabetic Retinopathy Screening System

[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)
[![Stack](https://img.shields.io/badge/Stack-HTML5%20%7C%20CSS3%20%7C%20Vanilla%20JS-blue?style=flat-square)](#)
[![Role](https://img.shields.io/badge/Member%203-Frontend%20Development-0ea5e9?style=flat-square)](#)

> **NetraVision** (*Netra* = Eye in Sanskrit) is a specialized clinical AI web platform designed for early automated detection and five-level severity classification of **Diabetic Retinopathy (DR)** from digital fundus photography.

---

## 🌟 Table of Contents
1. [Project Overview](#project-overview)
2. [Frontend Architecture & Pages](#frontend-architecture--pages)
3. [Design System & Theme](#design-system--theme)
4. [Backend Integration Guide (For Team Member 2)](#backend-integration-guide)
5. [Directory Structure](#directory-structure)
6. [Getting Started & Local Preview](#getting-started--local-preview)
7. [Team Members & Roles](#team-members--roles)

---

## 👁️ Project Overview
Diabetic Retinopathy is one of the leading causes of preventable blindness among working-age adults worldwide. Early detection via retinal screening is critical to prevent permanent vision loss. 

NetraVision bridges this gap by providing an intuitive, web-based diagnostic assistant that allows clinicians, researchers, and patients to upload retinal fundus images and receive:
- **Instant 5-Level DR Severity Grading** (Normal, Mild, Moderate, Severe, Proliferative)
- **Model Confidence Score** and full probability breakdown
- **Tailored Clinical Recommendations & Next Steps**
- **Downloadable Diagnostic Summary Reports**

---

## 📄 Frontend Architecture & Pages

All pages feature a cohesive, medical-grade dark visual aesthetic inspired by modern clinical AI software:

| Page | File | Key Features & Modules |
| :--- | :--- | :--- |
| **Home** | `index.html` | Hero section with particle backdrop, live stats counter, 5-level severity preview cards, feature highlights, CTAs. |
| **Predict** | `predict.html` | Drag-and-drop retinal image uploader, instant client-side preview, validation (format/size), simulated inference progress bar, seamless redirect to results. |
| **Results** | `results.html` | Dynamic report generator reading from `localStorage`, severity badge, confidence meter, 5-class distribution bar charts, personalized clinical recommendations, text report exporter. |
| **Features** | `features.html` | In-depth breakdown of system capabilities, comparison table (NetraVision vs. Manual Screening), interactive tooltips. |
| **How It Works** | `how-it-works.html` | Step-by-step visual AI pipeline (Acquisition &rarr; Preprocessing &rarr; CLAHE Enhancement &rarr; CNN Feature Extraction &rarr; Softmax Classification) + FAQ accordion. |
| **About** | `about.html` | Project mission, clinical motivation, technical architecture summary, values, and research disclaimer. |

---

## 🎨 Design System & Theme

Built purely with **HTML5, CSS3, and Vanilla JavaScript** — zero heavy frameworks or external build steps required.

- **Color Palette**:
  - Background Base: `#020b18` & `#041326` (Deep Clinical Dark)
  - Primary Accent: `#0ea5e9` (Neon Cyan / Electric Blue)
  - Secondary Accent: `#38bdf8` (Ice Blue)
  - Glow Effects: Ambient radial gradients & neon drop shadows
- **Typography**:
  - Heading Display: `Outfit`, sans-serif (Modern, high-tech)
  - Body Text: `Inter`, sans-serif (Crisp medical readability)
- **Interactive Micro-Animations**:
  - Sticky glassmorphic navbar with active page indicators and mobile drawer
  - IntersectionObserver scroll reveal for cards and sections
  - Toast notification system (`window.NetraVision.showToast(msg, type)`)
  - Smooth animated probability progress bars

---

## 🔌 Backend Integration Guide (For Team Member 2)

Connecting the frontend with your Python / Flask / FastAPI CNN server is straightforward.

### 1. In `js/predict.js`
Look for the `runAnalysis()` function:
```javascript
// Replace the demo simulation with your API endpoint:
async function runAnalysis() {
  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      body: formData
    });
    const result = await response.json();
    
    // Pass API output to results page:
    saveResultAndRedirect({
      level: result.level,             // Integer 0-4
      label: result.label,             // e.g. "Moderate DR"
      confidence: result.confidence,   // e.g. 91.4
      imageName: selectedFile.name,
      imageData: previewImg.src,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    showError("Failed to communicate with prediction server.");
  }
}
```

### 2. Expected Output Format
```json
{
  "level": 2,
  "label": "Moderate DR",
  "confidence": 91.4,
  "probabilities": [0.02, 0.05, 0.914, 0.012, 0.004]
}
```

---

## 📁 Directory Structure

```text
NetraVision/
├── frontend/                       # Client presentation layer (Vanilla HTML/CSS/JS)
│   ├── assets/images/              # High-resolution clinical visuals & branding
│   ├── css/style.css               # Design system & responsive styles
│   ├── js/
│   │   ├── main.js                 # Navbar, scroll animations, particle engine
│   │   └── predict.js              # File upload handler & results redirection
│   ├── index.html                  # Homepage with interactive showcase
│   ├── predict.html                # Image uploader & inference portal
│   ├── results.html                # Dynamic diagnostic report & breakdown
│   ├── features.html               # Platform capabilities & clinical comparison
│   ├── how-it-works.html           # 5-stage AI pipeline & architecture guide
│   ├── about.html                  # Project background & medical motivation
│
├── backend/                        # API service layer (Python / Flask)
│   ├── app.py                      # Flask API endpoints (/api/health, /api/predict)
│   ├── test_api.py                 # Automated unit tests for API endpoints
│   └── requirements.txt            # Python dependencies
│
├── ml/                             # Model architecture & experiment pipelines
│   ├── config.py                   # Central hyperparameters, paths, and class mappings
│   ├── dataset.py                  # Ben Graham contrast enhancement & stratified split
│   ├── model.py                    # NetraVisionDRNet residual neural network definition
│   ├── predict.py                  # Standalone CLI inference & clinical guidance
│   ├── train.py                    # Stratified training pipeline with EarlyStopping
│   ├── utils.py                    # Loss weighting & Quadratic Weighted Kappa (QWK)
│   └── evaluate.py                 # 5-class benchmark evaluation suite
│
├── outputs/                        # Training checkpoints & validation metrics
│   ├── best_model.pth              # Local PyTorch weights (excluded from Git via .gitignore)
│   ├── training_history.csv        # Epoch training & validation metric history
│   ├── training_metrics.json       # Serialized metrics log
│   ├── evaluation_report.json      # Complete benchmark numbers & confusion matrix
│   └── evaluation_report.md        # Detailed evaluation report
│
├── .gitignore                      # Excludes large binaries (*.pth), datasets, and caches
└── README.md                       # Project documentation (this file)

[EXTERNAL DATASET]
D:\aptos2019-blindness-detection\   # Raw APTOS 2019 dataset (train_images/, test_images/, CSVs)
```

---

## 🚀 Getting Started & Local Preview

### 1. Backend API & Full-Stack Server
Install Python dependencies and launch the backend (which also serves the frontend):

```powershell
cd NetraVision\backend
pip install -r requirements.txt
python app.py
```
- Open `http://localhost:5000` or `http://localhost:5000/predict.html` in your browser.
- Health Check: `http://localhost:5000/api/health`

### 2. Run Backend Unit Tests
```powershell
cd NetraVision\backend
python test_api.py
```

### 3. Standalone CLI Inference
```powershell
cd NetraVision\ml
python predict.py --sample
```

### 4. Frontend Preview via VS Code / Static Server
You can also preview the UI independently from `frontend/`:
- Open `frontend/index.html` directly in any web browser, or launch VS Code Live Server from the `frontend/` directory.

---

## 👥 Team Members & Roles

- **Member 1**: Dataset Preprocessing, Data Augmentation & CLAHE Enhancement
- **Member 2**: CNN Model Architecture, Training & Flask API Development
- **Member 3 (This Work)**: **Lead Frontend Developer** — UI/UX Design System, HTML5/CSS3/Vanilla JS Architecture, Interactive Visualizations, Dynamic Reporting
- **Member 4**: Model Evaluation, Performance Validation & Clinical Benchmarking

---

*© 2026 NetraVision. Created for academic and demonstration purposes.*
