/**
 * NetraVision — predict.js
 * Handles: drag-and-drop upload, image preview, file validation,
 *          loading state, demo prediction, and results page handoff.
 *
 * ----------------------------------------------------------------
 * BACKEND INTEGRATION GUIDE (for the Flask/API developer):
 * ----------------------------------------------------------------
 * Search for the function `runAnalysis()` below.
 * Replace the `simulateDemoPrediction()` call inside it with a
 * real fetch() call to your Flask API endpoint, for example:
 *
 *   const formData = new FormData();
 *   formData.append('image', selectedFile);
 *   const response = await fetch('/api/predict', {
 *     method: 'POST',
 *     body: formData
 *   });
 *   const data = await response.json();
 *   // data should have: { level, label, confidence, recommendations }
 *   saveResultAndRedirect(data);
 *
 * The rest of the UI (loading, progress, redirect) is already handled.
 * ----------------------------------------------------------------
 */

// ============================================================
// State
// ============================================================
let selectedFile = null;

// ============================================================
// DOM References
// ============================================================
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const chooseBtn     = document.getElementById('chooseBtn');
const previewArea   = document.getElementById('previewArea');
const previewImg    = document.getElementById('previewImg');
const removeBtn     = document.getElementById('removeBtn');
const analyzeBtn    = document.getElementById('analyzeBtn');
const uploadSection = document.getElementById('uploadSection');
const errorMsg      = document.getElementById('errorMsg');
const progressBar   = document.getElementById('uploadProgress');
const progressFill  = document.getElementById('uploadProgressFill');
const progressText  = document.getElementById('progressText');
const fileInfo      = document.getElementById('fileInfo');

// ============================================================
// File Validation
// ============================================================
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const MAX_SIZE_MB   = 10;

function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return { ok: false, message: '⚠ Invalid file type. Please upload a JPG or PNG image.' };
  }
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    return { ok: false, message: `⚠ File is too large. Maximum allowed size is ${MAX_SIZE_MB}MB.` };
  }
  return { ok: true };
}

function showError(msg) {
  if (errorMsg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
  }
}
function clearError() {
  if (errorMsg) {
    errorMsg.textContent = '';
    errorMsg.classList.add('hidden');
  }
}

// ============================================================
// Handle File Selection
// ============================================================
function handleFile(file) {
  clearError();
  const result = validateFile(file);
  if (!result.ok) {
    showError(result.message);
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    if (previewImg)  previewImg.src = e.target.result;
    if (dropZone)    dropZone.classList.add('hidden');
    if (previewArea) previewArea.classList.remove('hidden');

    // Show file info
    if (fileInfo) {
      const sizeMB = (file.size / 1024 / 1024).toFixed(2);
      fileInfo.innerHTML = `
        <span>📄 ${file.name}</span>
        <span>${sizeMB} MB</span>
        <span>${file.type.split('/')[1].toUpperCase()}</span>
      `;
    }

    // Enable analyze button
    if (analyzeBtn) analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ============================================================
// Event Listeners — Upload
// ============================================================
if (chooseBtn && fileInput) {
  chooseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

// Click the drop zone itself
if (dropZone) {
  dropZone.addEventListener('click', e => {
    if (e.target === dropZone || e.target.closest('#uploadSection')) {
      fileInput && fileInput.click();
    }
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });
  ['dragleave', 'dragend'].forEach(evt => {
    dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

// Remove / Replace image
if (removeBtn) {
  removeBtn.addEventListener('click', () => {
    selectedFile = null;
    if (previewImg)    previewImg.src = '';
    if (previewArea)   previewArea.classList.add('hidden');
    if (dropZone)      dropZone.classList.remove('hidden');
    if (analyzeBtn)    analyzeBtn.disabled = true;
    if (fileInput)     fileInput.value = '';
    clearError();
  });
}

// ============================================================
// Analyze Button — Run Analysis
// ============================================================
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) {
      showError('Please select an image first.');
      return;
    }
    runAnalysis();
  });
}

async function runAnalysis() {
  // Show loading state
  setLoadingState(true);

  try {
    // --------------------------------------------------------
    // DEMO MODE: Simulates prediction without a real backend.
    // Replace this block with a real API call when ready.
    // --------------------------------------------------------
    const result = await simulateDemoPrediction();

    // Save result to sessionStorage and redirect to results page
    saveResultAndRedirect(result);

  } catch (err) {
    console.error('Analysis error:', err);
    showError('An error occurred. Please try again.');
    setLoadingState(false);
  }
}

// ============================================================
// Demo Prediction Simulation
// ============================================================
function simulateDemoPrediction() {
  return new Promise(resolve => {
    // Simulate network delay (1.5 – 2.5 seconds)
    const delay = 1500 + Math.random() * 1000;

    // Simulate progress bar
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15 + 5;
      if (progress > 95) progress = 95;
      updateProgress(progress);
    }, 200);

    setTimeout(() => {
      clearInterval(interval);
      updateProgress(100);

      // Randomly pick a severity level for demo purposes
      const severityLevels = [
        { level: 0, label: 'No DR (Normal)',      confidence: 92 + Math.random() * 5 },
        { level: 1, label: 'Mild DR',             confidence: 78 + Math.random() * 10 },
        { level: 2, label: 'Moderate DR',         confidence: 81 + Math.random() * 8 },
        { level: 3, label: 'Severe DR',           confidence: 85 + Math.random() * 7 },
        { level: 4, label: 'Proliferative DR',    confidence: 88 + Math.random() * 8 },
      ];

      // For demo: pick level 1 (Mild DR) as a reproducible example
      const result = severityLevels[1];
      result.confidence = parseFloat(result.confidence.toFixed(1));

      resolve(result);
    }, delay);
  });
}

// ============================================================
// Progress Bar Helper
// ============================================================
function updateProgress(pct) {
  if (progressFill) {
    progressFill.style.width = `${pct}%`;
  }
  if (progressText) {
    progressText.textContent = `Analyzing… ${Math.round(pct)}%`;
  }
}

// ============================================================
// Loading State Toggle
// ============================================================
function setLoadingState(loading) {
  if (analyzeBtn) {
    analyzeBtn.disabled = loading;
    analyzeBtn.innerHTML = loading
      ? '<span class="spinner"></span> Analyzing…'
      : '🔬 Analyze Image';
  }
  if (progressBar) progressBar.classList.toggle('hidden', !loading);
  if (progressText) progressText.classList.toggle('hidden', !loading);
  if (progressFill && loading) progressFill.style.width = '0%';
}

// ============================================================
// Save Result & Redirect to Results Page
// ============================================================
function saveResultAndRedirect(result) {
  // Store result data so results.html can read it
  const payload = {
    ...result,
    timestamp: new Date().toISOString(),
    isDemo: true, // Flag: replace with false when using real API
    imageName: selectedFile ? selectedFile.name : 'sample.jpg',
  };

  // Also store the image preview as base64
  if (selectedFile) {
    const reader = new FileReader();
    reader.onload = e => {
      payload.imageData = e.target.result;
      sessionStorage.setItem('nv_result', JSON.stringify(payload));
      window.location.href = 'results.html';
    };
    reader.readAsDataURL(selectedFile);
  } else {
    sessionStorage.setItem('nv_result', JSON.stringify(payload));
    window.location.href = 'results.html';
  }
}

// ============================================================
// Sample Image Click (load a placeholder sample)
// ============================================================
document.querySelectorAll('.sample-img').forEach(img => {
  img.addEventListener('click', () => {
    // Fetch sample as blob then create a File object
    fetch(img.src)
      .then(r => r.blob())
      .then(blob => {
        const file = new File([blob], 'sample-retina.jpg', { type: blob.type });
        handleFile(file);
        window.NetraVision && window.NetraVision.showToast('Sample image loaded!', 'info');
      })
      .catch(() => {
        window.NetraVision && window.NetraVision.showToast('Could not load sample image.', 'error');
      });
  });
});

// ============================================================
// Add drag-over CSS class styles dynamically
// ============================================================
(function addDragOverStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .drop-zone.drag-over {
      border-color: var(--clr-blue-primary) !important;
      background: rgba(14,165,233,0.08) !important;
      box-shadow: 0 0 30px rgba(14,165,233,0.2) !important;
    }
    .drop-zone.drag-over .upload-icon { transform: scale(1.1); color: var(--clr-cyan-light); }
    #uploadProgress { transition: width 0.3s ease; }
    #fileInfo {
      display: flex;
      gap: 16px;
      align-items: center;
      font-size: 0.8rem;
      color: var(--clr-text-secondary);
      margin-top: 8px;
      padding: 8px 14px;
      background: rgba(14,165,233,0.05);
      border-radius: 8px;
      border: 1px solid rgba(14,165,233,0.12);
    }
  `;
  document.head.appendChild(style);
})();
