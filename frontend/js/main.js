/**
 * NetraVision — main.js
 * Handles: navbar scroll/toggle, scroll reveal, mobile nav,
 *          particles, active link highlighting
 */

/* ============================================================
   Navbar — scroll effect & hamburger
   ============================================================ */
(function initNavbar() {
  const navbar = document.querySelector('.navbar');
  const hamburger = document.querySelector('.navbar__hamburger');
  const mobileNav = document.querySelector('.navbar__mobile');

  // Scroll effect
  window.addEventListener('scroll', () => {
    if (navbar) {
      navbar.classList.toggle('scrolled', window.scrollY > 40);
    }
  });

  // Hamburger toggle
  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', () => {
      const isOpen = hamburger.classList.toggle('open');
      mobileNav.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', String(isOpen));
    });
    // Close on link click
    mobileNav.querySelectorAll('.navbar__link').forEach(link => {
      link.addEventListener('click', () => {
        hamburger.classList.remove('open');
        mobileNav.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
    // Close on outside click
    document.addEventListener('click', e => {
      if (!navbar.contains(e.target)) {
        hamburger.classList.remove('open');
        mobileNav.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Active link highlight
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.navbar__link[href]').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
})();

/* ============================================================
   Scroll-Reveal Observer
   ============================================================ */
(function initScrollReveal() {
  const elements = document.querySelectorAll('[data-reveal]');
  if (!elements.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  elements.forEach(el => observer.observe(el));
})();

/* ============================================================
   Hero Particles
   ============================================================ */
(function initParticles() {
  const container = document.querySelector('.hero__particles');
  if (!container) return;

  const count = 18;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('span');
    p.classList.add('particle');
    const size = Math.random() * 4 + 2;
    p.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      left: ${Math.random() * 100}%;
      animation-duration: ${Math.random() * 12 + 10}s;
      animation-delay: ${Math.random() * -15}s;
      opacity: ${Math.random() * 0.2 + 0.05};
    `;
    container.appendChild(p);
  }
})();

/* ============================================================
   Animated Counter (stats)
   ============================================================ */
function animateCounter(el, target, suffix = '') {
  let current = 0;
  const step = target / 60;
  const timer = setInterval(() => {
    current += step;
    if (current >= target) {
      current = target;
      clearInterval(timer);
    }
    el.textContent = Math.round(current) + suffix;
  }, 16);
}

(function initCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseFloat(el.dataset.counter);
        const suffix = el.dataset.suffix || '';
        animateCounter(el, target, suffix);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));
})();

/* ============================================================
   Smooth scroll for anchor links
   ============================================================ */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      e.preventDefault();
      const offset = 80; // navbar height
      const top = target.getBoundingClientRect().top + window.pageYOffset - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});

/* ============================================================
   Toast Notification
   ============================================================ */
function showToast(message, type = 'info', duration = 3500) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `
    <span class="toast__icon">${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
    <span>${message}</span>
  `;

  // Inline styles for self-contained toast (no extra CSS required)
  toast.style.cssText = `
    position: fixed;
    bottom: 32px; right: 32px;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 22px;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 500;
    font-family: var(--font-primary, 'Inter', sans-serif);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    animation: slideInToast 0.3s ease forwards;
    max-width: 380px;
    backdrop-filter: blur(12px);
  `;

  const colors = {
    success: { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.35)', color: '#34d399' },
    error:   { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',  color: '#f87171' },
    info:    { bg: 'rgba(14,165,233,0.12)', border: 'rgba(14,165,233,0.35)', color: '#38bdf8' },
    warning: { bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.35)', color: '#fbbf24' },
  };
  const c = colors[type] || colors.info;
  toast.style.background = c.bg;
  toast.style.border = `1px solid ${c.border}`;
  toast.style.color = c.color;

  if (!document.querySelector('#toast-keyframes')) {
    const style = document.createElement('style');
    style.id = 'toast-keyframes';
    style.textContent = `
      @keyframes slideInToast  { from { transform: translateX(60px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
      @keyframes slideOutToast { from { transform: translateX(0); opacity: 1; } to { transform: translateX(60px); opacity: 0; } }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideOutToast 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Expose globally
window.NetraVision = window.NetraVision || {};
window.NetraVision.showToast = showToast;

/* ============================================================
   Content Protection: Disable Copy, Drag & Selection
   ============================================================ */
(function initCopyProtection() {
  // Prevent copy keyboard shortcut or browser copy action on text
  document.addEventListener('copy', e => {
    const isInput = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
    if (!isInput) {
      e.preventDefault();
      if (window.NetraVision && typeof window.NetraVision.showToast === 'function') {
        window.NetraVision.showToast('Content copying is disabled.', 'info', 2200);
      }
    }
  });

  // Prevent drag & drop selection on images
  document.addEventListener('dragstart', e => {
    if (e.target.tagName === 'IMG' || e.target.closest('.hero__visual')) {
      e.preventDefault();
    }
  });

  // Prevent selection start on non-form elements
  document.addEventListener('selectstart', e => {
    const tag = e.target.tagName;
    if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
      e.preventDefault();
    }
  });
})();

