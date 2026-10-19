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
| **Our Team** | `team.html` | Interactive profile cards showcasing all 4 college project team members, highlighting Member 3's frontend focus. |
| **Contact** | `contact.html` | Fully validated contact form (real-time error checking, character counter, simulated submission toast), institutional contacts, and FAQs. |

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
├── index.html           # Homepage
├── predict.html         # Image upload & inference starter
├── results.html         # Diagnostic output & report generator
├── features.html        # System capabilities & comparison
├── how-it-works.html    # AI pipeline & CNN architecture
├── about.html           # Mission, motivation, research info
├── team.html            # Team profiles & role credits
├── contact.html         # Feedback & inquiry form
├── README.md            # Project documentation (this file)
├── css/
│   └── style.css        # Unified design system & responsive styling
├── js/
│   ├── main.js          # Shared navbar, scroll-reveal, particle engine
│   └── predict.js       # File upload handler & client simulation
└── assets/
    └── images/          # High-resolution clinical AI visuals & graphics
```

---

## 🚀 Getting Started & Local Preview

No `npm install` or compilation steps required!

1. **Directly via Browser**:
   - Double-click `index.html` or drag it into any web browser (Chrome, Edge, Firefox, Safari).
2. **Via VS Code Live Server**:
   - Open the `NetraVision` folder in VS Code, right-click `index.html`, and click **"Open with Live Server"**.
3. **Via Python Built-in Server**:
   ```bash
   cd NetraVision
   python -m http.server 8000
   ```
   Open `http://localhost:8000` in your browser.

---

## 👥 Team Members & Roles

- **Member 1**: Dataset Preprocessing, Data Augmentation & CLAHE Enhancement
- **Member 2**: CNN Model Architecture, Training & Flask API Development
- **Member 3 (This Work)**: **Lead Frontend Developer** — UI/UX Design System, HTML5/CSS3/Vanilla JS Architecture, Interactive Visualizations, Dynamic Reporting
- **Member 4**: Model Evaluation, Performance Validation & Clinical Benchmarking

---

*© 2026 NetraVision. Created for academic and demonstration purposes.*
