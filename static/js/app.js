/* ══════════════════════════════════════════════════════════════════
   Smart Home AI — Main JavaScript
   SPA routing, Feature controllers, API communication
   ══════════════════════════════════════════════════════════════════ */

// ─── STATE ───────────────────────────────────────────────────────────
let currentPage = 'home';
let webcamStream = null;
let emotionInterval = null;
let mediaRecorder = null;
let audioChunks = [];
let isListening = false;

// ─── HELPERS ─────────────────────────────────────────────────────────
function safeText(id, text) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = text;
    } else {
        console.warn(`[SafeText] Element with ID '${id}' not found.`);
    }
}

function safeQueryText(parent, selector, text) {
    if (!parent) return;
    const el = parent.querySelector(selector);
    if (el) {
        el.textContent = text;
    } else {
        console.warn(`[SafeQuery] Selector '${selector}' not found in parent.`);
    }
}

// ─── NAVIGATION ──────────────────────────────────────────────────────
function navigateTo(page) {
    // Close any webcam streams
    stopWebcam();
    stopEmotionDetection();

    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    // Show requested page
    const el = document.getElementById(`page-${page}`);
    if (el) {
        el.classList.add('active');
        currentPage = page;
    }

    // Initialize page-specific things
    if (page === 'voice') {
        loadDeviceStatus();
        loadRegisteredVoices();
    }
    if (page === 'face') {
        loadRegisteredFaces();
    }
}

function goHome() {
    navigateTo('home');
}

// ─── TOAST NOTIFICATIONS ─────────────────────────────────────────────
function showToast(message, type = 'success') {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// ─── WEBCAM HELPERS ──────────────────────────────────────────────────
async function startWebcam(videoElementId) {
    try {
        const video = document.getElementById(videoElementId);
        if (!video) return null;

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });

        video.srcObject = stream;
        await video.play();
        webcamStream = stream;
        return stream;
    } catch (e) {
        console.error('Webcam error:', e);
        showToast('Camera access denied. Please allow camera access.', 'error');
        return null;
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

function captureFrame(videoElementId) {
    const video = document.getElementById(videoElementId);
    if (!video || video.readyState < 2) return null;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// ═══════════════════════════════════════════════════════════════════
// FEATURE 1 — FAN SPEED CONTROL (Manual + Live Location)
// ═══════════════════════════════════════════════════════════════════

let currentFanMode = 'manual';

function switchFanMode(mode) {
    currentFanMode = mode;

    // Toggle tab active state
    document.getElementById('fan-tab-manual').classList.toggle('active', mode === 'manual');
    document.getElementById('fan-tab-live').classList.toggle('active', mode === 'live');

    // Toggle sections
    const manualSection = document.getElementById('fan-manual-section');
    const liveSection = document.getElementById('fan-live-section');

    if (mode === 'manual') {
        manualSection.className = 'fan-section-active';
        liveSection.className = 'fan-section-hidden';
    } else {
        manualSection.className = 'fan-section-hidden';
        liveSection.className = 'fan-section-active';
    }

    // Reset the fan result when switching modes
    resetFan();
}

async function predictFanSpeed() {
    const tempInput = document.getElementById('temp-input');
    const temperature = parseFloat(tempInput.value);

    if (isNaN(temperature)) {
        showToast('Please enter a valid temperature', 'error');
        return;
    }

    // Show loader
    const resultSection = document.getElementById('fan-result');
    const loader = document.getElementById('fan-loader');
    const resultContent = document.getElementById('fan-result-content');
    resultSection.style.display = 'block';
    loader.classList.add('active');
    resultContent.style.display = 'none';

    try {
        const res = await fetch('/predict_fan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temperature })
        });

        const data = await res.json();
        loader.classList.remove('active');

        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        displayFanResult(data);
        showToast(`Fan speed predicted: ${data.speed}`);

    } catch (e) {
        loader.classList.remove('active');
        showToast('Server error. Make sure Flask is running.', 'error');
        console.error(e);
    }
}

function displayFanResult(data) {
    const fanBlades = document.getElementById('fan-blades');

    safeText('fan-speed-num', data.speed);
    safeText('fan-speed-label', `Speed ${data.speed} — ${data.label} (${data.range})`);

    // Set fan spinning speed
    const speeds = { 1: '6s', 2: '4s', 3: '2.5s', 4: '1.5s', 5: '0.6s' };
    fanBlades.style.setProperty('--fan-speed', speeds[data.speed] || '4s');
    fanBlades.classList.add('spinning');

    // Color based on speed
    const colors = { 1: '#4a9eff', 2: '#1D9E75', 3: '#FFD700', 4: '#FF8C42', 5: '#ff4444' };
    document.getElementById('fan-speed-num').style.color = colors[data.speed] || 'var(--accent-green)';

    document.getElementById('fan-result-content').style.display = 'block';
}

async function fetchLiveTemperature() {
    const btn = document.getElementById('fetch-live-temp-btn');
    const btnText = document.getElementById('live-btn-text');

    if (!navigator.geolocation) {
        showToast('Geolocation is not supported by your browser', 'error');
        return;
    }

    // Disable button and show loading state
    btn.disabled = true;
    btnText.textContent = 'Detecting location...';

    // Show loader
    const resultSection = document.getElementById('fan-result');
    const loader = document.getElementById('fan-loader');
    const resultContent = document.getElementById('fan-result-content');
    resultSection.style.display = 'block';
    loader.classList.add('active');
    resultContent.style.display = 'none';

    try {
        // Step 1: Get GPS coordinates
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 60000
            });
        });

        const lat = position.coords.latitude;
        const lon = position.coords.longitude;

        btnText.textContent = 'Fetching weather data...';

        // Step 2: Send coordinates to backend
        const res = await fetch('/live_temperature', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lon })
        });

        const data = await res.json();
        loader.classList.remove('active');

        if (!data.success) {
            showToast(data.error || 'Failed to fetch weather data', 'error');
            btn.disabled = false;
            btnText.textContent = 'Detect Location & Fetch Temperature';
            resultSection.style.display = 'none';
            return;
        }

        // Step 3: Display the live weather data
        displayLiveWeather(data);

        // Step 4: Display fan prediction from live temperature
        if (data.fan_prediction) {
            displayFanResult(data.fan_prediction);
        }

        showToast(`Live: ${data.temperature}°C at ${data.location} — Fan speed: ${data.fan_prediction?.speed}`);

        btn.disabled = false;
        btnText.textContent = 'Refresh Live Temperature';

    } catch (err) {
        loader.classList.remove('active');
        resultSection.style.display = 'none';
        btn.disabled = false;
        btnText.textContent = 'Detect Location & Fetch Temperature';

        if (err.code === 1) {
            showToast('Location access denied. Please allow location permission.', 'error');
        } else if (err.code === 2) {
            showToast('Location unavailable. Check your device GPS.', 'error');
        } else if (err.code === 3) {
            showToast('Location request timed out. Try again.', 'error');
        } else {
            showToast(`Error: ${err.message}`, 'error');
        }
        console.error('[Live Temp]', err);
    }
}

function displayLiveWeather(data) {
    const weatherCard = document.getElementById('live-weather-card');

    // Weather code to emoji mapping
    const weatherEmojis = {
        0: '☀️', 1: '🌤️', 2: '⛅', 3: '☁️',
        45: '🌫️', 48: '🌫️', 51: '🌦️', 53: '🌧️',
        55: '🌧️', 61: '🌦️', 63: '🌧️', 65: '🌧️',
        71: '🌨️', 73: '❄️', 75: '❄️', 80: '🌦️',
        81: '🌧️', 82: '⛈️', 95: '⛈️', 96: '⛈️', 99: '⛈️'
    };

    const weatherEmoji = weatherEmojis[data.weather_code] || '🌡️';

    safeText('live-location-name', data.location);
    document.getElementById('live-weather-desc').textContent = `${weatherEmoji} ${data.weather_description}`;
    safeText('live-temp-value', data.temperature);
    safeText('live-feels-like', `${data.apparent_temperature}°C`);
    safeText('live-humidity', `${data.humidity}%`);
    safeText('live-wind', `${data.wind_speed} km/h`);
    safeText('live-timezone', data.timezone);

    // Color the temperature based on value
    const tempEl = document.getElementById('live-temp-value');
    if (data.temperature <= 15) tempEl.style.color = '#4a9eff';
    else if (data.temperature <= 20) tempEl.style.color = '#1D9E75';
    else if (data.temperature <= 25) tempEl.style.color = '#FFD700';
    else if (data.temperature <= 30) tempEl.style.color = '#FF8C42';
    else tempEl.style.color = '#ff4444';

    weatherCard.style.display = 'block';
    weatherCard.style.animation = 'fadeSlideIn 0.5s ease forwards';
}

function resetFan() {
    document.getElementById('temp-input').value = '';
    document.getElementById('fan-blades').classList.remove('spinning');
    document.getElementById('fan-result-content').style.display = 'none';
    document.getElementById('fan-result').style.display = 'none';
    safeText('fan-speed-num', '—');
    safeText('fan-speed-label', 'Enter temperature to predict');

    // Reset live weather card
    const weatherCard = document.getElementById('live-weather-card');
    if (weatherCard) weatherCard.style.display = 'none';

    const liveBtn = document.getElementById('fetch-live-temp-btn');
    if (liveBtn) {
        liveBtn.disabled = false;
        safeText('live-btn-text', 'Detect Location & Fetch Temperature');
    }
}


// ═══════════════════════════════════════════════════════════════════
// FEATURE 2 — EMOTION DETECTION (Webcam Only)
// ═══════════════════════════════════════════════════════════════════

async function startEmotionDetection() {
    const stream = await startWebcam('emotion-video');
    if (!stream) return;

    document.getElementById('emotion-start-btn').classList.add('hidden');
    document.getElementById('emotion-stop-btn').classList.remove('hidden');

    // First detection immediately
    setTimeout(async () => {
        const frame = captureFrame('emotion-video');
        if (frame) await detectEmotion(frame);
    }, 500);

    // Continuous detection every 1.5 seconds
    emotionInterval = setInterval(async () => {
        const frame = captureFrame('emotion-video');
        if (frame) {
            await detectEmotion(frame);
        }
    }, 1500);
}

function stopEmotionDetection() {
    if (emotionInterval) {
        clearInterval(emotionInterval);
        emotionInterval = null;
    }
    document.getElementById('emotion-start-btn')?.classList.remove('hidden');
    document.getElementById('emotion-stop-btn')?.classList.add('hidden');
}

async function detectEmotion(imageData) {
    try {
        const res = await fetch('/predict_emotion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        const data = await res.json();

        if (data.error && !data.face_detected) {
            // No face detected - clear canvas and show message
            clearEmotionCanvas();
            const resultSection = document.getElementById('emotion-result');
            resultSection.classList.remove('hidden');
            safeText('emotion-name', 'No face detected');
            document.getElementById('emotion-name').style.color = 'var(--text-muted)';
            safeText('emotion-message', 'Please look at the camera');
            return;
        }

        if (data.face_detected) {
            updateRoomLighting(data);
            drawFaceBox('emotion-video', 'emotion-canvas', data.face_box, data.emotion, data.healing_color);
        }

    } catch (e) {
        console.error('Emotion detection error:', e);
    }
}

function updateRoomLighting(data) {
    const room = document.getElementById('emotion-room');
    const bulb = document.getElementById('emotion-bulb');
    const emotionName = document.getElementById('emotion-name');
    const emotionMsg = document.getElementById('emotion-message');
    const resultSection = document.getElementById('emotion-result');

    resultSection.classList.remove('hidden');

    // Update room background with tinted color
    const color = data.healing_color;
    room.style.backgroundColor = hexToRGBA(color, 0.15);

    // Update bulb glow
    bulb.classList.add('glowing');
    bulb.style.setProperty('--bulb-color', color);
    bulb.style.background = color;
    bulb.style.boxShadow = `0 0 60px ${color}, 0 0 120px ${color}, 0 0 180px ${hexToRGBA(color, 0.3)}`;

    // Update text
    emotionName.innerHTML = `${data.emotion} Detected ${data.emoji}`;
    emotionName.style.color = color;
    emotionMsg.textContent = data.message;
}

function hexToRGBA(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function resetEmotion() {
    stopEmotionDetection();
    stopWebcam();

    const room = document.getElementById('emotion-room');
    const bulb = document.getElementById('emotion-bulb');
    room.style.backgroundColor = '#e8ecf1';
    bulb.classList.remove('glowing');
    bulb.style.background = '#d1d5db';
    bulb.style.boxShadow = 'none';

    document.getElementById('emotion-result').classList.add('hidden');
    safeText('emotion-name', '');
    safeText('emotion-message', '');

    clearEmotionCanvas();
}

// ─── FACE BOX DRAWING ────────────────────────────────────────────────
function drawFaceBox(videoId, canvasId, faceBox, label, color) {
    if (!faceBox) return;
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    if (!video || !canvas) return;

    canvas.width = video.videoWidth || video.clientWidth;
    canvas.height = video.videoHeight || video.clientHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { x, y, w, h } = faceBox;

    ctx.strokeStyle = color || '#1D9E75';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    if (label) {
        ctx.fillStyle = color || '#1D9E75';
        ctx.font = 'bold 16px Inter, sans-serif';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 26, textWidth + 12, 24);
        ctx.fillStyle = '#000';
        ctx.fillText(label, x + 6, y - 8);
    }
}

function clearEmotionCanvas() {
    const canvas = document.getElementById('emotion-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// ═══════════════════════════════════════════════════════════════════
// FEATURE 3 — FACE RECOGNITION DOOR
// ═══════════════════════════════════════════════════════════════════

async function startFaceScan() {
    const stream = await startWebcam('face-video');
    if (!stream) return;

    document.getElementById('face-scan-btn').disabled = true;
    safeText('face-scan-btn', '⏳ Scanning...');

    // Show scan line animation
    const scanLine = document.getElementById('face-scan-line');
    scanLine.classList.add('active');
    scanLine.classList.remove('denied');

    // Wait 2.5 seconds for scanning effect
    await new Promise(resolve => setTimeout(resolve, 2500));

    // Capture frame and send to backend
    const frame = captureFrame('face-video');
    if (!frame) {
        showToast('Could not capture frame', 'error');
        resetFaceScan();
        return;
    }

    try {
        const res = await fetch('/recognize_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frame })
        });

        const data = await res.json();
        handleFaceResult(data);

    } catch (e) {
        console.error('Face recognition error:', e);
        showToast('Server error', 'error');
        resetFaceScan();
    }
}

function handleFaceResult(data) {
    const scanLine = document.getElementById('face-scan-line');
    const doorLeft = document.getElementById('door-left');
    const doorRight = document.getElementById('door-right');
    const doorFrame = document.getElementById('door-frame-el');
    const doorIndicator = document.getElementById('door-indicator');
    const resultMsg = document.getElementById('face-result-msg');

    resultMsg.classList.remove('hidden');

    if (data.matched) {
        // ACCESS GRANTED
        scanLine.classList.remove('active');
        doorIndicator.className = 'door-indicator granted';

        setTimeout(() => {
            doorLeft.classList.add('open');
            doorRight.classList.add('open');
        }, 300);

        const quality = data.match_quality || '';
        resultMsg.innerHTML = `<span style="color: var(--accent-green)">Welcome ${data.person_name}! Access Granted ✅</span>
            <br><small style="color: var(--text-muted)">Match: ${quality} (distance: ${data.distance || '?'})</small>`;
        showToast(`Welcome ${data.person_name}!`);

    } else {
        // ACCESS DENIED
        scanLine.classList.add('denied');

        setTimeout(() => {
            scanLine.classList.remove('active');
            doorIndicator.className = 'door-indicator denied';
            doorFrame.classList.add('door-shake');
            setTimeout(() => doorFrame.classList.remove('door-shake'), 1000);
        }, 500);

        if (!data.face_detected) {
            resultMsg.innerHTML = `<span style="color: var(--accent-red)">${data.message || 'No face detected'}</span>`;
        } else {
            resultMsg.innerHTML = `<span style="color: var(--accent-red)">Access Denied 🚫 Unknown Person</span>
                <br><small style="color: var(--text-muted)">Distance: ${data.distance || '?'} (threshold: 75)</small>`;
        }
    }

    document.getElementById('face-scan-btn').disabled = false;
    safeText('face-scan-btn', '🔍 Start Scan');
}

function resetFaceScan() {
    stopWebcam();

    const scanLine = document.getElementById('face-scan-line');
    scanLine.classList.remove('active', 'denied');

    document.getElementById('door-left').classList.remove('open');
    document.getElementById('door-right').classList.remove('open');
    document.getElementById('door-indicator').className = 'door-indicator';
    document.getElementById('face-result-msg').classList.add('hidden');

    document.getElementById('face-scan-btn').disabled = false;
    safeText('face-scan-btn', '🔍 Start Scan');
}

// Toggle Face registration page
function toggleFaceRegPage() {
    const regSection = document.getElementById('face-registration-section');
    const mainSection = document.getElementById('face-main-section');
    regSection.classList.toggle('hidden');
    mainSection.classList.toggle('hidden');
    
    // Clear password input when entering/exiting
    const pwdInput = document.getElementById('face-admin-password');
    if (pwdInput) pwdInput.value = '';
    
    // Default back to gate hidden if entering
    if (!regSection.classList.contains('hidden')) {
        document.getElementById('face-password-gate').classList.remove('hidden');
        document.getElementById('face-reg-panel').classList.add('hidden');
    }
}

async function loadRegisteredFaces() {
    try {
        const res = await fetch('/registered_faces');
        const data = await res.json();

        const list = document.getElementById('face-registered-list');
        if (data.people && data.people.length > 0) {
            list.innerHTML = data.people.map(p =>
                `<div style="display: inline-flex; align-items: center; gap: 6px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; margin: 4px;">
                    <span style="color: var(--text-primary); font-weight: 500;">${p.name}</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem;">(${p.photo_count} photos)</span>
                    <button onclick="removePerson('${p.name}')" 
                            style="background: none; border: none; color: var(--accent-red); cursor: pointer; font-size: 1.1rem; padding: 0 4px;"
                            title="Remove ${p.name}">✕</button>
                </div>`
            ).join('');
        } else {
            list.innerHTML = '<span class="badge badge-info">No faces registered yet. Add photos below.</span>';
        }
    } catch (e) {
        console.error(e);
    }
}

async function registerFaceFromWebcam() {
    const nameInput = document.getElementById('face-register-name');
    const name = nameInput.value.trim();
    if (!name) {
        showToast('Enter a name first', 'error');
        return;
    }

    const frame = captureFrame('face-video');
    if (!frame) {
        showToast('Start webcam scan first, then click Capture', 'error');
        return;
    }

    try {
        const res = await fetch('/register_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: frame })
        });
        const data = await res.json();

        if (data.success) {
            showToast(data.message);
            loadRegisteredFaces();
            // Don't clear name — user may want to add more photos for same person
        } else {
            showToast(data.message || 'Registration failed', 'error');
        }
    } catch (e) {
        showToast('Server error', 'error');
    }
}

async function removePerson(personName) {
    if (!confirm(`Remove "${personName}" and all their photos?`)) return;

    try {
        const res = await fetch('/remove_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: personName })
        });
        const data = await res.json();

        if (data.success) {
            showToast(data.message);
            loadRegisteredFaces();
        } else {
            showToast(data.message || 'Could not remove', 'error');
        }
    } catch (e) {
        showToast('Server error', 'error');
    }
}

async function removeAllFaces() {
    if (!confirm('Remove ALL registered faces? This cannot be undone.')) return;

    try {
        const res = await fetch('/remove_all_faces', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();

        if (data.success) {
            showToast(data.message);
            loadRegisteredFaces();
            safeText('face-reg-status', 'All faces removed.');
        } else {
            showToast(data.message || 'Could not remove', 'error');
        }
    } catch (e) {
        showToast('Server error', 'error');
    }
}

async function registerFaceFromUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const nameInput = document.getElementById('face-register-name');
    const name = nameInput.value.trim();
    if (!name) {
        showToast('Enter a person name first', 'error');
        event.target.value = '';
        return;
    }

    const statusEl = document.getElementById('face-reg-status');
    let successCount = 0;

    for (let i = 0; i < files.length; i++) {
        statusEl.textContent = `Uploading photo ${i + 1}/${files.length}...`;

        const reader = new FileReader();
        const b64 = await new Promise((resolve) => {
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(files[i]);
        });

        try {
            const res = await fetch('/register_face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, image: b64 })
            });
            const data = await res.json();

            if (data.success) {
                successCount++;
                statusEl.innerHTML = `✅ Photo ${i + 1}: ${data.message}`;
            } else {
                statusEl.innerHTML = `❌ Photo ${i + 1}: ${data.message}`;
            }
        } catch (e) {
            statusEl.innerHTML = `❌ Photo ${i + 1}: Server error`;
        }
    }

    loadRegisteredFaces();
    showToast(`Registered ${successCount}/${files.length} photos for ${name}`);
    event.target.value = ''; // Reset file input
}

// Admin password for face registration
const FACE_ADMIN_PASSWORD = 'vansh123';

function unlockFaceRegistration() {
    const passwordInput = document.getElementById('face-admin-password');
    const password = passwordInput.value.trim();

    if (password === FACE_ADMIN_PASSWORD) {
        document.getElementById('face-password-gate').classList.add('hidden');
        document.getElementById('face-reg-panel').classList.remove('hidden');
        loadRegisteredFaces();
        showToast('Admin access granted!');
    } else {
        showToast('Wrong password! Access denied.', 'error');
        passwordInput.value = '';
        passwordInput.focus();
    }
}

// Admin password for voice registration
const VOICE_ADMIN_PASSWORD = 'vansh123';

function unlockVoiceRegistration() {
    const passwordInput = document.getElementById('voice-admin-password');
    const password = passwordInput.value.trim();

    if (password === VOICE_ADMIN_PASSWORD) { // Apply new password
        document.getElementById('voice-password-gate').classList.add('hidden');
        document.getElementById('voice-reg-panel').classList.remove('hidden');
        loadRegisteredVoices();
        showToast('Admin access granted!');
    } else {
        showToast('Wrong password! Access denied.', 'error');
        passwordInput.value = '';
        passwordInput.focus();
    }
}

// ─── WAV RECORDER UTILITY ──────────────────────────────────────────
let audioContext;
let scriptProcessor;
let mediaStreamSource;
let pcmData = [];
let recordingStream;
let recordingSampleRate;

async function startWavRecording() {
    pcmData = [];
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    recordingSampleRate = audioContext.sampleRate;
    
    mediaStreamSource = audioContext.createMediaStreamSource(recordingStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    
    // Add stable 2x Gain boost (prevents clipping but ensures clarity)
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 2.0; 
    
    mediaStreamSource.connect(gainNode);
    gainNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
    
    scriptProcessor.onaudioprocess = function(e) {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmArr = new Float32Array(inputData.length);
        pcmArr.set(inputData);
        pcmData.push(pcmArr);
    };
}

function stopWavRecording() {
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        mediaStreamSource.disconnect();
    }
    if (recordingStream) {
        recordingStream.getTracks().forEach(t => t.stop());
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
    
    // Flatten PCM data
    let totalLength = 0;
    for (let arr of pcmData) totalLength += arr.length;
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (let arr of pcmData) {
        result.set(arr, offset);
        offset += arr.length;
    }
    
    // Downsample to 16kHz for API
    const targetRate = 16000;
    const ratio = recordingSampleRate / targetRate;
    const outLength = Math.round(result.length / ratio);
    const downsampled = new Int16Array(outLength);
    
    for (let i = 0; i < outLength; i++) {
        let inIdx = Math.round(i * ratio);
        if (inIdx >= result.length) inIdx = result.length - 1; // Bound check
        let s = Math.max(-1, Math.min(1, result[inIdx]));
        downsampled[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    
    // Create WAV blob
    const buffer = new ArrayBuffer(44 + downsampled.length * 2);
    const view = new DataView(buffer);
    
    function writeStr(offset, string) {
        for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
    }
    
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + downsampled.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono channel
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeStr(36, 'data');
    view.setUint32(40, downsampled.length * 2, true);
    
    let offsetData = 44;
    for (let i = 0; i < downsampled.length; i++, offsetData += 2) {
        view.setInt16(offsetData, downsampled[i], true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

// ═══════════════════════════════════════════════════════════════════
// FEATURE 4 — VOICE CONTROL
// ═══════════════════════════════════════════════════════════════════

// Push-to-Talk Logic
let isCommandRecording = false;

function toggleVoiceCommand() {
    if (isCommandRecording) {
        stopVoiceCommand();
    } else {
        startVoiceCommand();
    }
}

async function startVoiceCommand() {
    if (isCommandRecording) return;

    try {
        await startWavRecording();
        isCommandRecording = true;
        safeText('voice-listen-status', 'Listening... Click to stop');
        document.getElementById('voice-listen-btn').style.transform = 'scale(1.1)';
        document.getElementById('voice-listen-btn').style.background = 'var(--accent-red)';
    } catch (e) {
        isCommandRecording = false;
        showToast('Microphone access denied', 'error');
    }
}

async function stopVoiceCommand() {
    if (!isCommandRecording) return;
    isCommandRecording = false;
    
    const statusEl = document.getElementById('voice-listen-status');
    safeText('voice-listen-status', 'Processing...');
    document.getElementById('voice-listen-btn').style.transform = 'scale(1)';
    document.getElementById('voice-listen-btn').style.background = 'var(--accent-primary)';
    
    const wavBlob = stopWavRecording();
    const reader = new FileReader();
    
    reader.onload = async (e) => {
        try {
            const base64 = e.target.result.split(',')[1];
            
            const res = await fetch('/process_voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: base64 })
            });

            if (!res.ok) {
                const errorMsg = await res.text();
                throw new Error(`Server returned ${res.status}: ${errorMsg.substring(0, 50)}`);
            }

            const data = await res.json();
            
            if (data.text) {
                showToast(`You said: "${data.text}"`, 'info');
            }

            if (data.success) {
                showToast(data.message);
                if (data.device_states) updateDeviceCards(data.device_states);
            } else {
                showToast(data.message || data.error || 'Recognition failed', 'error');
            }
            safeText('voice-listen-status', 'Click to speak');
        } catch (err) {
            console.error('[Voice] Fetch error:', err);
            showToast(`Error: ${err.message}`, 'error');
            safeText('voice-listen-status', 'Click to speak');
        }
    };
    reader.readAsDataURL(wavBlob);
}

async function loadDeviceStatus() {
    try {
        const res = await fetch('/device_status');
        const data = await res.json();
        updateDeviceCards(data);
    } catch (e) {
        console.error('Load device status error:', e);
    }
}

function updateDeviceCards(states) {
    if (!states) return;

    // Fan — spinning animation
    const fanCard = document.getElementById('device-fan');
    if (fanCard) {
        fanCard.classList.toggle('active', states.fan);
        safeQueryText(fanCard, '.device-card-status', states.fan ? 'ON' : 'OFF');
        const fanBlades = document.getElementById('voice-fan-blades');
        if (fanBlades) {
            if (states.fan) {
                fanBlades.classList.add('spinning');
                fanBlades.style.setProperty('--fan-speed', '1.5s');
            } else {
                fanBlades.classList.remove('spinning');
            }
        }
    }

    // Light — glowing bulb animation
    const lightCard = document.getElementById('device-light');
    if (lightCard) {
        lightCard.classList.toggle('active', states.light);
        safeQueryText(lightCard, '.device-card-status', states.light ? 'ON' : 'OFF');
        const bulb = document.getElementById('voice-light-bulb');
        if (bulb) {
            if (states.light) {
                bulb.classList.add('glowing');
                bulb.style.setProperty('--bulb-color', '#FFD700');
                bulb.style.background = '#FFD700';
                bulb.style.boxShadow = '0 0 30px #FFD700, 0 0 60px #FFD700, 0 0 90px rgba(255, 215, 0, 0.3)';
            } else {
                bulb.classList.remove('glowing');
                bulb.style.background = '#d1d5db';
                bulb.style.boxShadow = 'none';
            }
        }
    }

    // Door — opening/closing animation
    const doorCard = document.getElementById('device-door');
    if (doorCard) {
        doorCard.classList.toggle('active', states.door);
        safeQueryText(doorCard, '.device-card-status', states.door ? 'OPEN' : 'CLOSED');
        const doorLeft = document.getElementById('voice-door-left');
        const doorRight = document.getElementById('voice-door-right');
        const doorIndicator = document.getElementById('voice-door-indicator');
        if (doorLeft && doorRight) {
            if (states.door) {
                doorLeft.classList.add('open');
                doorRight.classList.add('open');
                if (doorIndicator) doorIndicator.className = 'voice-door-indicator granted';
            } else {
                doorLeft.classList.remove('open');
                doorRight.classList.remove('open');
                if (doorIndicator) doorIndicator.className = 'voice-door-indicator';
            }
        }
    }
}

function resetVoice() {
    // Reset all devices to OFF
    fetch('/execute_command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'turn off fan' })
    });
    fetch('/execute_command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'turn off light' })
    });
    fetch('/execute_command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'close door' })
    }).then(() => loadDeviceStatus());
}

// Voice registration
let isRecording = false;
let recordingTimeout = null;

async function loadRegisteredVoices() {
    try {
        const res = await fetch('/registered_voices');
        const data = await res.json();

        const slotsContainer = document.getElementById('voice-reg-slots');
        if (!slotsContainer) return;

        const count = document.getElementById('voice-reg-count');
        safeText('voice-reg-count', `${data.count}/${data.max} registered`);

        let html = '';
        for (let i = 0; i < data.max; i++) {
            if (i < data.voices.length) {
                const v = data.voices[i];
                html += `
                    <div class="reg-slot filled" style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="reg-slot-name">🎤 ${v.name}</div>
                        <div>
                            <button class="btn btn-sm btn-danger" onclick="removeVoice('${v.name}')" style="padding: 4px 8px; font-size: 0.8rem;">✕ Delete</button>
                        </div>
                    </div>`;
            } else {
                html += `
                    <div class="reg-slot">
                        <div class="reg-slot-name reg-slot-empty">Empty Slot</div>
                        <button class="btn btn-sm btn-primary" onclick="showVoiceRegister()" style="padding: 4px 8px; font-size: 0.8rem;">➕ Add Voice</button>
                    </div>`;
            }
        }
        slotsContainer.innerHTML = html;

    } catch (e) {
        console.error(e);
    }
}

function showVoiceRegister() {
    document.getElementById('voice-reg-form').classList.remove('hidden');
    document.getElementById('voice-reg-name').value = '';
    safeText('voice-reg-status', '');
}

async function removeVoice(name) {
    if (!confirm(`Delete voice footprint for "${name}"?`)) return;

    try {
        const res = await fetch('/remove_voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (data.success) {
            showToast(data.message);
            loadRegisteredVoices();
        } else {
            showToast(data.message || 'Could not remove', 'error');
        }
    } catch (e) {
        showToast('Server error', 'error');
    }
}

async function removeAllVoices() {
    if (!confirm('Remove ALL registered voices? This cannot be undone.')) return;

    try {
        const res = await fetch('/remove_all_voices', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();
        if (data.success) {
            showToast(data.message);
            loadRegisteredVoices();
        } else {
            showToast(data.message || 'Could not remove', 'error');
        }
    } catch (e) {
        showToast('Server error', 'error');
    }
}

async function startVoiceRecording() {
    try {
        isRecording = true;
        await startWavRecording();
        
        safeText('voice-rec-btn', '⏹ Stop Recording');
        document.getElementById('voice-rec-btn').classList.remove('btn-primary');
        document.getElementById('voice-rec-btn').classList.add('btn-danger');
        safeText('voice-reg-status', 'Recording... Please read the sentences below clearly.');
        showToast('Recording started');

        // Auto-stop after 15 seconds
        recordingTimeout = setTimeout(() => {
            if (isRecording) stopVoiceRecording();
        }, 15000);

    } catch (e) {
        isRecording = false;
        showToast('Microphone access denied', 'error');
    }
}

function stopVoiceRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearTimeout(recordingTimeout);
    
    safeText('voice-rec-btn', '⏳ Processing...');
    safeText('voice-reg-status', 'Processing and registering your voice...');
    
    const wavBlob = stopWavRecording();
    
    const reader = new FileReader();
    reader.onload = async (e) => {
        const base64 = e.target.result.split(',')[1];
        const name = document.getElementById('voice-reg-name').value.trim();

        if (!name) {
            showToast('Enter a name first', 'error');
            resetVoiceRecBtn();
            return;
        }

        try {
            const res = await fetch('/register_voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, audio: base64 })
            });
            const data = await res.json();
            if (data.success) {
                showToast(data.message);
                loadRegisteredVoices();
                document.getElementById('voice-reg-form').classList.add('hidden');
                safeText('voice-reg-status', '');
            } else {
                showToast(data.message, 'error');
                safeText('voice-reg-status', 'Error: ' + data.message);
            }
        } catch (err) {
            showToast('Registration error', 'error');
            safeText('voice-reg-status', 'Registration failed due to server error.');
        }
        
        resetVoiceRecBtn();
    };
    reader.readAsDataURL(wavBlob);
}

function resetVoiceRecBtn() {
    safeText('voice-rec-btn', '🎤 Start Recording');
    document.getElementById('voice-rec-btn').classList.remove('btn-danger');
    document.getElementById('voice-rec-btn').classList.add('btn-primary');
}

function toggleVoiceRecording() {
    if (isRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

// Toggle voice registration page
function toggleVoiceRegPage() {
    const regSection = document.getElementById('voice-registration-section');
    const mainSection = document.getElementById('voice-main-section');
    regSection.classList.toggle('hidden');
    mainSection.classList.toggle('hidden');
}

// ─── KEYBOARD SHORTCUTS ──────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        goHome();
    }
    if (e.key === 'Enter' && currentPage === 'fan') {
        predictFanSpeed();
    }
});

// ─── INIT ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    navigateTo('home');
    resetVoice(); // Fixes default fan and light remaining ON bug
});
