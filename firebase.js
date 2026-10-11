/**
 * EduPulse AI / NEXORA - Firebase Client Configuration
 *
 * Firebase Project ID: edupulse-ai-782c8
 */

import { initializeApp, getApps } from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js';
import { getAuth, GoogleAuthProvider } from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js';

// Firebase Web App Configuration
export const firebaseConfig = {
    apiKey: "AIzaSyAE7mLwzMwsm7eTE2d1LaIalhZYz2V1eso",
    authDomain: "edupulse-ai-782c8.firebaseapp.com",
    projectId: "edupulse-ai-782c8",
    storageBucket: "edupulse-ai-782c8.firebasestorage.app",
    messagingSenderId: "825551620138",
    appId: "1:825551620138:web:449df787293358e4c19b53",
    measurementId: "G-7JYQ416PKN"
};

// Check if valid API key is present
export const isConfigured = Boolean(
    firebaseConfig.apiKey &&
    firebaseConfig.apiKey !== "YOUR_FIREBASE_API_KEY"
);

// Initialize Firebase App safely once
export const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
export const auth = getAuth(app);
export const provider = new GoogleAuthProvider();
