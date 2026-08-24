// ============================================
// INTERACTIVE SPLIT SLIDER FUNCTIONALITY
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    initializeSplitSlider();
    initializeFileUpload();
    initializeProcessButton();
    setupSmoothScroll();
});

// Split Slider Implementation
function initializeSplitSlider() {
    const splitContainer = document.querySelector('.split-track');
    const handle = document.querySelector('.split-handle');
    const beforeImage = document.querySelector('.split-image.before');
    const afterImage = document.querySelector('.split-image.after');
    
    let isSliding = false;

    handle.addEventListener('mousedown', () => {
        isSliding = true;
    });

    document.addEventListener('mouseup', () => {
        isSliding = false;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isSliding) return;

        const rect = splitContainer.getBoundingClientRect();
        let x = e.clientX - rect.left;

        // Clamp value between 0 and container width
        x = Math.max(0, Math.min(x, rect.width));

        const percentage = (x / rect.width) * 100;

        handle.style.left = percentage + '%';
        beforeImage.style.width = percentage + '%';
    });

    // Touch support for mobile
    handle.addEventListener('touchstart', () => {
        isSliding = true;
    });

    document.addEventListener('touchend', () => {
        isSliding = false;
    });

    document.addEventListener('touchmove', (e) => {
        if (!isSliding) return;

        const rect = splitContainer.getBoundingClientRect();
        let x = e.touches[0].clientX - rect.left;

        x = Math.max(0, Math.min(x, rect.width));
        const percentage = (x / rect.width) * 100;

        handle.style.left = percentage + '%';
        beforeImage.style.width = percentage + '%';
    });
}

// ============================================
// FILE UPLOAD HANDLING
// ============================================

function initializeFileUpload() {
    const thermalInput = document.getElementById('thermalInput');
    const thermalPreview = document.getElementById('thermalPreview');

    if (thermalInput) {
        thermalInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    thermalPreview.innerHTML = '';
                    const img = document.createElement('img');
                    img.src = event.target.result;
                    thermalPreview.appendChild(img);
                };
                reader.readAsDataURL(file);
            }
        });
    }
}

// ============================================
// PROCESS IMAGE BUTTON
// ============================================

function initializeProcessButton() {
    const processBtn = document.getElementById('processBtn');
    const thermalInput = document.getElementById('thermalInput');
    const rgbPreview = document.getElementById('rgbPreview');
    const statusMessage = document.getElementById('processingStatus');

    if (processBtn) {
        processBtn.addEventListener('click', async function() {
            if (!thermalInput.files[0]) {
                showStatus('Please upload an image first', 'error');
                return;
            }

            try {
                showStatus('Processing image with AI model...', 'loading');
                processBtn.disabled = true;
                processBtn.style.opacity = '0.6';

                // Create FormData for file upload
                const formData = new FormData();
                formData.append('file', thermalInput.files[0]);

                // Call backend API
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Processing failed');
                }

                const data = await response.json();

                // Display result
                if (data.result_url) {
                    rgbPreview.innerHTML = '';
                    const resultImg = document.createElement('img');
                    resultImg.src = data.result_url;
                    rgbPreview.appendChild(resultImg);
                    showStatus('✓ Processing complete!', 'success');
                } else {
                    throw new Error('No result returned');
                }
            } catch (error) {
                console.error('Error:', error);
                showStatus('Error processing image. Please try again.', 'error');
            } finally {
                processBtn.disabled = false;
                processBtn.style.opacity = '1';
            }
        });
    }
}

function showStatus(message, type) {
    const statusMessage = document.getElementById('processingStatus');
    if (statusMessage) {
        statusMessage.textContent = message;
        statusMessage.className = `status-message ${type}`;
    }
}

// ============================================
// SMOOTH SCROLL NAVIGATION
// ============================================

function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ============================================
// SCROLL ANIMATIONS
// ============================================

function observeElements() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all feature cards
    document.querySelectorAll('.feature-card, .tech-item, .stat-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });
}

// Call observer when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observeElements);
} else {
    observeElements();
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

// Detect if browser supports backdrop-filter
function detectBackdropFilterSupport() {
    const element = document.createElement('div');
    element.style.backdropFilter = 'blur(10px)';
    return element.style.backdropFilter !== '';
}

// Add fallback class if backdrop-filter not supported
if (!detectBackdropFilterSupport()) {
    document.body.classList.add('no-backdrop-filter');
}

// Add active state to nav links based on scroll position
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').slice(1) === current) {
                link.classList.add('active');
                link.style.color = 'var(--primary-color)';
            } else {
                link.style.color = 'var(--text-secondary)';
            }
        });
    });
}

updateActiveNavLink();

// ============================================
// API CONFIGURATION
// ============================================

const API_BASE_URL = window.location.origin;

// Helper function for API calls
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        }
    };

    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// ============================================
// PERFORMANCE OPTIMIZATION
// ============================================

// Lazy load images
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });

    document.querySelectorAll('img.lazy').forEach(img => {
        imageObserver.observe(img);
    });
}

// ============================================
// EASTER EGGS & INTERACTIONS
// ============================================

// Add some fun interactions
const logo = document.querySelector('.logo');
if (logo) {
    let clickCount = 0;
    logo.addEventListener('click', function() {
        clickCount++;
        if (clickCount === 3) {
            showStatus('🎉 The future is thermal! 🎉', 'success');
            clickCount = 0;
        }
    });
}

// Console message
console.log('%c🛰️ ir2rgb - Satellite Thermal-to-RGB Translation', 
    'color: #06b6d4; font-size: 16px; font-weight: bold;');
console.log('%cPowered by Pix2Pix GAN | SIH 2026 | I.R.I.S. Team',
    'color: #a0aec0; font-size: 12px;');
