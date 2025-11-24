# Crime Interrogation Analysis System

A sophisticated web application for analyzing facial expressions and behavioral patterns during crime interrogations. The system uses real-time video analysis, AI-powered behavioral insights, and secure authentication to provide comprehensive interrogation analysis tools.

## Features

- Real-time facial expression analysis using DeepFace
- AI-powered behavioral analysis using Google's Gemini AI
- Secure Firebase Authentication
- Live emotion tracking and visualization
- Session recording and playback
- Detailed analysis reports in text format
- User-specific case management
- System-wide interrogation records

## Prerequisites

- Python 3.8 or higher
- Webcam for video capture
- Firebase account
- Google Cloud account (for Gemini API)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd crime-interrogation-analysis
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

### Local Development

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your actual credentials:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
FIREBASE_CREDENTIALS_PATH=path/to/your/firebase-service-account.json
```

⚠️ **Never commit the `.env` file to version control!**

### Render Deployment

Set the following environment variables in Render dashboard:

**Backend (Firebase Admin SDK):**
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"...","private_key":"..."}
```

**Frontend (Firebase Web SDK):**
```env
FIREBASE_API_KEY=your_web_api_key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
FIREBASE_MEASUREMENT_ID=your_measurement_id (optional)
```

**Important Notes:**
- For `FIREBASE_SERVICE_ACCOUNT_JSON`, paste the entire contents of your Firebase service account JSON file as a single-line string
- The Frontend variables come from your Firebase Web App configuration (different from service account)
- The app automatically detects whether to use `FIREBASE_CREDENTIALS_PATH` (local file) or `FIREBASE_SERVICE_ACCOUNT_JSON` (env variable)

## Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project
3. Enable Authentication:

   - Go to Authentication > Sign-in method
   - Enable Email/Password authentication
   - (Optional) Enable other providers as needed

4. Get Firebase Admin SDK credentials:

   - Go to Project Settings > Service Accounts
   - Click "Generate New Private Key"
   - Save the JSON file securely
   - Update `FIREBASE_CREDENTIALS_PATH` in `.env` to point to this file

5. Configure Firebase Web App:

   - Go to Project Settings > General
   - Scroll to "Your apps" section
   - Click the web icon (</>)
   - Register your app with a nickname
   - Copy the Firebase configuration object

6. Create your Firebase frontend configuration:
   - Copy `static/js/firebase-config.example.js` to `static/js/firebase-config.js`
   - Edit `firebase-config.js` and replace the placeholder values with your actual Firebase configuration:

```javascript
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const firebaseConfig = {
  apiKey: "YOUR_ACTUAL_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.firebasestorage.app",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID",
  measurementId: "YOUR_MEASUREMENT_ID"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
```

⚠️ **Never commit `firebase-config.js` to version control!**

## Gemini API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key
4. Add it to your `.env` file as `GEMINI_API_KEY`

## Running the Application

1. Ensure your virtual environment is activated
2. Start the Flask application:

```bash
python app.py
```

3. Access the application at `http://localhost:5000`

## Project Structure

```
crime-interrogation-analysis/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
└── user_cases.json      # Persistent storage for case data
```

## Dependencies

The project requires the following main dependencies:

- Flask
- OpenCV (cv2)
- DeepFace
- Firebase Admin SDK
- Google Generative AI (Gemini)
- Pandas
- Python-dotenv
- Flask-CORS (for browser webcam feature)
- NumPy (for image processing)

## Browser Webcam Feature

A new isolated feature allows visitors to use their own browser camera for real-time emotion analysis.

### Accessing the Camera Page

1. Navigate to `http://localhost:5000/camera` (or your deployed URL)
2. The page is publicly accessible but requires Firebase sign-in to use the camera
3. Click "Sign In with Firebase" and complete authentication
4. Click "Start Camera" to begin analysis
5. Grant browser camera permissions when prompted

### How It Works

- **Client-side capture**: Frames are captured at 800ms intervals from the user's webcam
- **Real-time analysis**: Each frame is sent to `/api/frame` endpoint for DeepFace emotion analysis
- **Authentication**: Both cookie-based and header-based (Bearer token) authentication are supported
- **Privacy**: Frames are analyzed in real-time and NOT stored on the server
- **Isolation**: This feature does not interfere with the main interrogation analysis flow

### Technical Details

- Frame resolution: 480x360 (resized from camera input)
- Frame format: JPEG with 60% quality compression
- Max frame size: 2.5MB (enforced server-side)
- Analysis interval: 800ms between frames
- Error handling: Exponential backoff with up to 3 retries

### Security & Privacy

- **HTTPS Required**: For production deployment, HTTPS is mandatory for camera access
- **User Consent**: Clear privacy notice displayed before camera activation
- **Authentication**: All frame uploads require valid Firebase authentication
- **No Storage**: Video frames are processed and discarded immediately
- **CORS**: Currently set to allow all origins for development - **restrict in production**

### Deployment Notes (Render or similar platforms)

1. **Environment Variables**: Ensure `GEMINI_API_KEY` and `FIREBASE_SERVICE_ACCOUNT_JSON` are set
2. **HTTPS**: Render provides HTTPS by default - required for camera access
3. **Instance Size**: DeepFace analysis is CPU-intensive - consider:
   - Basic: Suitable for light testing
   - Standard/Pro: Recommended for multiple concurrent users
4. **CORS Configuration**: Update `app.py` line with CORS to restrict origins:
   ```python
   CORS(app, resources={r"/api/*": {"origins": ["https://yourdomain.com"]}})
   ```
5. **Demo Image**: Move the placeholder image from `/mnt/data/` to `/static/images/` and update the path in `camera.html`

### Troubleshooting

- **Camera not starting**: Check browser permissions and ensure HTTPS is enabled
- **Authentication errors**: Verify Firebase configuration and token validity
- **Analysis errors**: Check server logs for DeepFace/OpenCV issues
- **Performance issues**: Consider reducing frame rate or upgrading server instance
