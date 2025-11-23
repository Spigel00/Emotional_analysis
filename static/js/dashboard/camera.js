/**
 * Browser Camera Analysis - Client-side JavaScript
 * Handles webcam access, frame capture, and server communication
 */

import { auth } from "../firebase-config.js";
import { onAuthStateChanged, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

// Configuration
const FRAME_INTERVAL_MS = 800; // Send frame every 800ms
const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 1000;

// State
let stream = null;
let captureInterval = null;
let isRunning = false;
let currentUser = null;
let idToken = null;
let retryCount = 0;
let backoffTime = INITIAL_BACKOFF_MS;

// DOM Elements
const authSection = document.getElementById('auth-section');
const cameraSection = document.getElementById('camera-section');
const signInBtn = document.getElementById('signInBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const dominantEmotionEl = document.getElementById('dominantEmotion');
const emotionScoresEl = document.getElementById('emotionScores');
const statusMessagesEl = document.getElementById('statusMessages');

/**
 * Display a status message to the user
 */
function showStatus(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `status-message status-${type}`;
    messageDiv.textContent = message;
    statusMessagesEl.innerHTML = '';
    statusMessagesEl.appendChild(messageDiv);
    
    // Auto-remove success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 5000);
    }
}

/**
 * Check authentication state and update UI
 */
function initAuth() {
    onAuthStateChanged(auth, async (user) => {
        currentUser = user;
        
        if (user) {
            try {
                // Get ID token
                idToken = await user.getIdToken();
                window._idToken = idToken; // Expose globally for compatibility
                
                // Show camera section, hide auth section
                authSection.classList.add('hidden');
                cameraSection.classList.remove('hidden');
                
                console.log('User authenticated:', user.uid);
            } catch (error) {
                console.error('Error getting ID token:', error);
                showStatus('Authentication error. Please sign in again.', 'error');
                authSection.classList.remove('hidden');
                cameraSection.classList.add('hidden');
            }
        } else {
            // Not signed in - show auth section
            authSection.classList.remove('hidden');
            cameraSection.classList.add('hidden');
            idToken = null;
            window._idToken = null;
        }
    });
}

/**
 * Handle sign-in button click
 */
signInBtn.addEventListener('click', () => {
    // Redirect to login page
    window.location.href = '/auth/login';
});

/**
 * Start the camera and begin capturing frames
 */
async function startCamera() {
    if (isRunning) {
        showStatus('Camera is already running', 'info');
        return;
    }
    
    if (!currentUser) {
        showStatus('Please sign in first', 'error');
        return;
    }
    
    try {
        // Request camera access
        showStatus('Requesting camera access...', 'info');
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        // Attach stream to video element
        video.srcObject = stream;
        video.play();
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
        });
        
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        retryCount = 0;
        backoffTime = INITIAL_BACKOFF_MS;
        
        showStatus('Camera started. Analysis in progress...', 'success');
        
        // Start capturing and sending frames
        captureInterval = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
        
    } catch (error) {
        console.error('Error starting camera:', error);
        if (error.name === 'NotAllowedError') {
            showStatus('Camera access denied. Please allow camera permissions.', 'error');
        } else if (error.name === 'NotFoundError') {
            showStatus('No camera found on this device.', 'error');
        } else {
            showStatus('Error starting camera: ' + error.message, 'error');
        }
        stopCamera();
    }
}

/**
 * Stop the camera and clean up resources
 */
function stopCamera() {
    isRunning = false;
    
    // Clear interval
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    // Stop all media tracks
    if (stream) {
        stream.getTracks().forEach(track => {
            track.stop();
            console.log('Stopped track:', track.kind);
        });
        stream = null;
    }
    
    // Clear video source
    if (video.srcObject) {
        video.srcObject = null;
    }
    
    // Update UI
    startBtn.disabled = false;
    stopBtn.disabled = true;
    dominantEmotionEl.textContent = '—';
    emotionScoresEl.innerHTML = '<p style="text-align: center; color: #999;">Camera stopped</p>';
    
    showStatus('Camera stopped', 'info');
    console.log('Camera stopped and resources released');
}

/**
 * Capture a frame and send it to the server for analysis
 */
async function captureAndSendFrame() {
    if (!isRunning || !video.videoWidth) {
        return;
    }
    
    try {
        // Draw current video frame to canvas (resize to 480x360)
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to base64 JPEG
        const dataURL = canvas.toDataURL('image/jpeg', 0.6);
        const base64Data = dataURL.split(',')[1]; // Remove data:image/jpeg;base64, prefix
        
        // Refresh token if needed (Firebase tokens expire)
        if (currentUser) {
            idToken = await currentUser.getIdToken(false); // false = use cached if valid
            window._idToken = idToken;
        }
        
        // Send to server
        const response = await fetch('/api/frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${idToken}`
            },
            body: JSON.stringify({ image_base64: base64Data })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Reset retry count on success
        retryCount = 0;
        backoffTime = INITIAL_BACKOFF_MS;
        
        // Update UI with results
        updateEmotionDisplay(result);
        
    } catch (error) {
        console.error('Error sending frame:', error);
        handleError(error);
    }
}

/**
 * Update the emotion display with analysis results
 */
function updateEmotionDisplay(result) {
    if (result.dominant_emotion) {
        dominantEmotionEl.textContent = result.dominant_emotion.toUpperCase();
    }
    
    if (result.emotions) {
        // Build emotion scores HTML
        let scoresHTML = '';
        const sortedEmotions = Object.entries(result.emotions)
            .sort((a, b) => b[1] - a[1]); // Sort by score descending
        
        for (const [emotion, score] of sortedEmotions) {
            const percentage = score.toFixed(1);
            scoresHTML += `
                <div>
                    <span>${emotion}</span>
                    <span>${percentage}%</span>
                </div>
            `;
        }
        
        emotionScoresEl.innerHTML = scoresHTML;
    }
}

/**
 * Handle errors with exponential backoff
 */
function handleError(error) {
    retryCount++;
    
    if (retryCount >= MAX_RETRIES) {
        showStatus('Multiple errors occurred. Please try restarting the camera.', 'error');
        stopCamera();
        return;
    }
    
    // Show error but don't stop
    showStatus(`Analysis error (retry ${retryCount}/${MAX_RETRIES}): ${error.message}`, 'error');
    
    // Implement exponential backoff
    const waitTime = Math.min(backoffTime * Math.pow(2, retryCount - 1), 10000);
    console.log(`Backing off for ${waitTime}ms before next attempt`);
    
    // Pause frame capture temporarily
    if (captureInterval) {
        clearInterval(captureInterval);
        setTimeout(() => {
            if (isRunning) {
                captureInterval = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
            }
        }, waitTime);
    }
}

/**
 * Event Listeners
 */
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});

// Initialize authentication check
initAuth();

// Export functions for external access if needed
window.cameraModule = {
    start: startCamera,
    stop: stopCamera
};

console.log('Camera module initialized');
