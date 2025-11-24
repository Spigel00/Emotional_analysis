# Quick Local Setup Guide

## 🚨 Fix "Firebase: Error (auth/invalid-api-key)"

You're missing Firebase Web SDK configuration in your `.env` file!

### Step 1: Create `.env` File

```bash
cp .env.example .env
```

### Step 2: Get Your Firebase Configuration

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Click the gear icon ⚙️ → **Project Settings**
4. Scroll to "Your apps" → Select your **Web app**
5. Copy the configuration values

### Step 3: Fill in `.env` File

Open `.env` and add these values:

```env
# Gemini AI
GEMINI_API_KEY=your_actual_gemini_key

# Firebase Backend (Admin SDK)
FIREBASE_CREDENTIALS_PATH=C:/path/to/your/firebase-service-account.json

# Firebase Frontend (Web SDK) - REQUIRED FOR LOCAL DEV
FIREBASE_API_KEY=AIzaSy...              # From Firebase Console
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
FIREBASE_MESSAGING_SENDER_ID=123456789012
FIREBASE_APP_ID=1:123456789012:web:abc123
FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX    # Optional
```

### Step 4: Restart Flask

```bash
python app.py
```

### Step 5: Test

Go to `http://127.0.0.1:5000/auth/login`

You should see the login page without Firebase errors!

---

## 📋 What Each Variable Does:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | For AI behavioral analysis |
| `FIREBASE_CREDENTIALS_PATH` | Backend authentication (server-side) |
| `FIREBASE_API_KEY` | **Frontend authentication (browser-side)** ← YOU'RE MISSING THIS |
| `FIREBASE_AUTH_DOMAIN` | Firebase Auth domain |
| `FIREBASE_PROJECT_ID` | Your Firebase project ID |
| Others | Additional Firebase configuration |

---

## ✅ How to Verify It's Working:

1. No more "invalid-api-key" errors in browser console
2. Firebase scripts load successfully
3. Login form works
4. Can register/login users

---

## 🔒 Security Note:

- `.env` is in `.gitignore` - never commit it!
- The frontend API key is **safe to expose** (it's not a secret)
- The backend service account JSON is **secret** - keep it secure!
