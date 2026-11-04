/**
 * NetraVision — predict.js
 * Handles: drag-and-drop upload, image preview, file validation,
 *          loading state, real-time PyTorch backend inference,
 *          and results page handoff.
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
  clearError();
  setLoadingState(true);

  // Smooth simulated progress during inference
  let progress = 12;
  updateProgress(progress);
  const progressInterval = setInterval(() => {
    if (progress < 90) {
      progress += Math.random() * 12 + 6;
      if (progress > 90) progress = 90;
      updateProgress(progress);
    }
  }, 250);

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Support both direct Flask serving and cross-origin Live Server
    const apiUrl = (window.location.protocol === 'file:' || (window.location.port && window.location.port !== '5000'))
      ? 'http://127.0.0.1:5000/api/predict'
      : '/api/predict';

    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData
    });

    clearInterval(progressInterval);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      const msg = (errorData && errorData.error)
        ? errorData.error
        : `Prediction failed with status ${response.status}.`;
      showError(msg);
      setLoadingState(false);
      return;
    }

    updateProgress(100);
    const data = await response.json();

    if (data.status === 'error' || data.predicted_class === undefined) {
      showError(data.error || 'Invalid prediction response from server.');
      setLoadingState(false);
      return;
    }

    // Save actual model prediction to sessionStorage and redirect
    saveResultAndRedirect(data);

  } catch (err) {
    clearInterval(progressInterval);
    console.error('Analysis error:', err);
    showError('Unable to connect to the prediction backend. Please ensure the Flask server is running at http://127.0.0.1:5000.');
    setLoadingState(false);
  }
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
    isDemo: false,
    imageName: selectedFile ? selectedFile.name : (result.filename || 'sample.jpg'),
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
