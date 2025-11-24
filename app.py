import os
import cv2
from flask import Flask,url_for, redirect,render_template, Response, jsonify, request, stream_with_context, make_response
from flask import Response
import json
from deepface import DeepFace
import threading
import time
import logging
import pandas as pd
from io import StringIO
import datetime
import json
import google.generativeai as genai # Import Gemini library
from io import BytesIO
import firebase_admin 
from firebase_admin import credentials, auth

from functools import wraps
import uuid # Added for case IDs
from jinja2 import Environment, select_autoescape # Added for Jinja filter
 
# Added for JSON persistence
import os
import json
import threading
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Security Headers for Production ---
@app.after_request
def add_security_headers(response):
    """Add security headers for production deployment"""
    # Allow credentials for Firebase auth
    if request.origin:
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    # Content Security Policy - Allow Firebase and Google APIs
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.gstatic.com https://apis.google.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://*.googleapis.com https://*.firebaseio.com https://*.cloudfunctions.net https://identitytoolkit.googleapis.com https://securetoken.googleapis.com wss://*.firebaseio.com; "
        "frame-src 'self' https://*.firebaseapp.com; "
    )
    response.headers['Content-Security-Policy'] = csp
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Allow HTTPS only in production
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    return response

# --- Persistent Storage Setup ---
CASES_JSON_FILE = "user_cases.json"
file_lock = threading.Lock() # To prevent race conditions during file read/write

# --- Load Initial Data ---
def load_cases_from_json():
    """Loads case data from the JSON file."""
    with file_lock:
        try:
            if os.path.exists(CASES_JSON_FILE):
                with open(CASES_JSON_FILE, 'r') as f:
                    # Handle empty file case
                    content = f.read()
                    if not content:
                        app.logger.info(f"{CASES_JSON_FILE} is empty, starting with empty cases.")
                        return {}
                    data = json.loads(content)
                    app.logger.info(f"Loaded case data from {CASES_JSON_FILE}")
                    return data
            else:
                app.logger.info(f"{CASES_JSON_FILE} not found, starting with empty cases.")
                return {}
        except (json.JSONDecodeError, IOError) as e:
            app.logger.error(f"Error loading {CASES_JSON_FILE}: {e}. Starting with empty cases.")
            # Optionally backup corrupted file here
            return {}

user_cases = load_cases_from_json() # Initialize user_cases from file

# --- Save Data Function ---
def save_cases_to_json():
    """Saves the current user_cases dictionary to the JSON file."""
    with file_lock:
        try:
            with open(CASES_JSON_FILE, 'w') as f:
                json.dump(user_cases, f, indent=4) # Use indent for readability
            app.logger.debug(f"Saved case data to {CASES_JSON_FILE}")
        except IOError as e:
            app.logger.error(f"Error saving cases to {CASES_JSON_FILE}: {e}")


# --- In-Memory Storage --- (Replaced by load_cases_from_json)
# user_cases = {} # Dictionary to store cases per user_id: { user_id: [list of case dicts] }
# Note: Global analysis_data still used for the *active* session

# --- Configuration ---
# Consider using environment variables for security: os.environ.get('GEMINI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') # Use env var
try:
    if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Use gemini-1.5-flash or similar
        app.logger.info("Gemini AI configured successfully.")
    else:
        app.logger.warning("Gemini API key not set. Analysis endpoint will not work.")
        gemini_model = None
except Exception as e:
    app.logger.error(f"Failed to configure Gemini AI: {e}. Analysis endpoint will not work.")
    gemini_model = None # Ensure model is None if configuration fails

# --- Firebase Initialization ---
try:
    # Support both file path (local dev) and JSON string (Render deployment)
    firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    firebase_creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    
    if firebase_creds_json:
        # Render deployment: JSON credentials as environment variable
        app.logger.info("Loading Firebase credentials from FIREBASE_SERVICE_ACCOUNT_JSON env var")
        cred_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(cred_dict)
    elif firebase_creds_path:
        # Local development: JSON file path
        app.logger.info(f"Loading Firebase credentials from file: {firebase_creds_path}")
        cred = credentials.Certificate(firebase_creds_path)
    else:
        raise ValueError("Neither FIREBASE_SERVICE_ACCOUNT_JSON nor FIREBASE_CREDENTIALS_PATH is set")
    
    firebase_admin.initialize_app(cred)
    app.logger.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    app.logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
    app.logger.error("Make sure FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_CREDENTIALS_PATH is properly set")



# --- Global Variables (for active session) ---
capture = None # OpenCV video capture object
video_thread = None
analysis_data = [] # List to store analysis results ({timestamp, emotion, dominant_emotion})
behavior_analysis_history = [] # Stores text analyses from Gemini
analysis_lock = threading.Lock()
is_processing = False # Flag to indicate if analysis is running
frame_skip = 5 # Analyze every Nth frame to reduce load
last_frame_analyzed = None # Store the latest analyzed frame for display (optional)
current_analysis_case_id = None # Track the case ID for the active analysis
# --- Constants for Gemini Analysis ---
GEMINI_ANALYSIS_INTERVAL_SECONDS = 15 # How often to call Gemini API
GEMINI_DATA_WINDOW_SECONDS = 60 # How much past data to send to Gemini

# --- Helper Functions ---
def ensure_dir(directory):
    """Ensures a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- LLM Integration ---
def get_behavioral_analysis_from_gemini(data_batch):
    """
    Sends a batch of emotion data to Gemini and returns its textual analysis.
    """
    if not gemini_model: # Check if model was configured
        app.logger.warning("Gemini model not configured. Skipping analysis.")
        return "(Gemini analysis skipped - model not configured)"

    if not data_batch:
        return "(Not enough data yet for analysis)"

    formatted_data_str = "Recent emotion readings:\n"
    try:
        # Limit the data sent for brevity and cost-effectiveness
        entries_to_send = min(len(data_batch), 20) # Send last 20 entries max
        for entry in data_batch[-entries_to_send:]:
            try:
                # Ensure timestamp is valid ISO format before parsing
                ts_dt = datetime.datetime.fromisoformat(entry['timestamp'])
                ts_formatted = ts_dt.strftime("%H:%M:%S")
                formatted_data_str += f"- Time: {ts_formatted}, Dominant Emotion: {entry['dominant_emotion']}\n"
            except (ValueError, TypeError, KeyError) as fmt_err:
                 app.logger.warning(f"Skipping entry due to formatting error: {fmt_err} in {entry}")
                 continue # Skip malformed entries

        prompt = f"""
        You are an expert behavioral analyst assisting in a suspect interrogation.
        Analyze the following sequence of dominant emotions detected from the suspect's facial expressions.
        Provide a brief (1-2 sentences) behavioral analysis focusing on potential stress, deception indicators, or significant emotional shifts.
        Do not repeat the raw data in your analysis. Be concise.

        Recent Emotion Data:
        {formatted_data_str}

        Analysis:
        """
        app.logger.info(f"Sending prompt to Gemini (sample data length: {len(data_batch)} entries).")
        app.logger.debug(f"Gemini Prompt (first 100 chars): {prompt[:100]}...")

        try:
            response = gemini_model.generate_content(prompt)
            # Check for safety ratings or blocks if necessary, depending on API version/settings
            if response.parts:
                analysis_text = response.text
            else:
                 analysis_text = "(Gemini response blocked or empty)"
                 app.logger.warning(f"Gemini response potentially blocked. Reason: {response.prompt_feedback}")

            app.logger.info("Received analysis from Gemini.")
            app.logger.debug(f"Gemini Response: {analysis_text}")
            return analysis_text
        except Exception as e:
            # Catch specific API errors if possible (e.g., RateLimitError, APICallError)
            app.logger.error(f"Error calling Gemini API: {e}")
            # Check if error response has details
            details = getattr(e, 'details', '')
            return f"(Error during Gemini analysis: {e} {details})"

    except Exception as e:
         app.logger.error(f"Error formatting data or prompt for Gemini: {e}")
         return "(Error preparing data for Gemini)"


# --- Video Capture and Analysis Thread ---
def capture_and_analyze():
    """
    Captures video frames, performs emotion analysis, and stores results.
    Runs in a separate thread.
    """
    global capture, analysis_data, is_processing, last_frame_analyzed
    app.logger.info("Starting video capture thread...")

    # Try common camera indices
    potential_indices = [0, 1, -1] # -1 can sometimes work on Linux/macOS
    capture_index = None
    for index in potential_indices:
        try:
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                app.logger.info(f"Webcam opened successfully at index {index}.")
                capture_index = index
                break
            else:
                # Ensure resource release even if isOpened is false
                capture.release()
                capture = None
                app.logger.warning(f"Failed to open webcam at index {index} (isOpened false).")
        except Exception as e:
             app.logger.error(f"Error trying to open webcam at index {index}: {e}")
             if capture:
                 capture.release()
             capture = None

    if capture_index is None or not capture or not capture.isOpened():
        app.logger.error("Cannot open webcam on any common index. Analysis thread stopping.")
        is_processing = False # Ensure the flag is reset
        return

    frame_count = 0
    analysis_interval = max(1, frame_skip) # Ensure interval is at least 1

    while is_processing:
        ret, frame = capture.read()
        if not ret:
            app.logger.warning("Failed to grab frame, stopping analysis thread.")
            break # Exit loop if frame grab fails

        # Only process frame if it's time according to the skip interval
        if frame_count % analysis_interval == 0:
            try:
                # Check frame validity
                if frame is None or frame.size == 0:
                   app.logger.warning("Skipping empty frame.")
                   continue

                # DeepFace expects BGR, OpenCV provides BGR by default
                results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    detector_backend='opencv', # Specify a backend like opencv, ssd, dlib, mtcnn, retinaface
                    enforce_detection=True, # Only analyze if face detected
                    silent=True # Suppress verbose console output from DeepFace
                )

                # Process results if a face was detected and analyzed
                if results and isinstance(results, list) and results[0]: # Check if list has items
                    first_face_result = results[0]
                    dominant_emotion = first_face_result.get('dominant_emotion', 'Unknown') # Provide default
                    emotions_from_deepface = first_face_result.get('emotion', {}) # Provide default
                    timestamp = datetime.datetime.now().isoformat() # Use ISO format for consistency

                    # Convert emotion scores to standard Python floats for JSON serialization
                    sanitized_emotions = {k: float(v) for k, v in emotions_from_deepface.items()}

                    new_data_point = {
                        "timestamp": timestamp,
                        "dominant_emotion": dominant_emotion,
                        "emotions": sanitized_emotions # Store sanitized emotion scores
                    }
                    with analysis_lock:
                        analysis_data.append(new_data_point)
                        # Optional: Limit stored data size to prevent memory issues
                        # MAX_DATA_POINTS = 5000
                        # if len(analysis_data) > MAX_DATA_POINTS:
                        #     analysis_data = analysis_data[-MAX_DATA_POINTS:]

                    app.logger.debug(f"Analysis @ {timestamp}: {dominant_emotion}")
                    last_frame_analyzed = frame # Update last analyzed frame (optional)

            except ValueError as e:
                # Specifically handle the "no face detected" case
                if "Face could not be detected" in str(e) or "cannot be empty" in str(e):
                    app.logger.debug("No face detected in frame.")
                    # Optionally log 'no face' event, careful not to flood logs/data
                else:
                    # Log other unexpected ValueErrors from DeepFace
                    app.logger.error(f"DeepFace analysis ValueError: {e}")
            except Exception as e:
                # Catch any other unexpected errors during analysis
                app.logger.error(f"Unexpected error in analysis loop: {e}", exc_info=True) # Log stack trace

        frame_count += 1
        # Small sleep to prevent tight loop, yielding CPU time
        time.sleep(0.01) # e.g., ~100 FPS max theoretical processing rate before analysis

    # Release webcam and clean up when stopping
    if capture:
        capture.release()
    capture = None
    app.logger.info("Video capture thread stopped and resources released.")


# --- Server-Sent Events Stream ---
def generate_analysis_stream():
    """
    Streams emotion updates and periodic behavioral analysis using SSE.
    """
    global analysis_data, behavior_analysis_history
    last_emotion_sent_index = -1
    last_gemini_analysis_time = time.time()

    try:
        while True: # Loop indefinitely until client disconnects
            now = time.time()
            new_emotion_data_to_send = []
            current_analysis_batch = []
            should_run_gemini = False

            with analysis_lock:
                # 1. Get new individual emotion updates since last send
                current_length = len(analysis_data)
                if current_length > last_emotion_sent_index + 1:
                    new_emotion_data_to_send = analysis_data[last_emotion_sent_index + 1:]
                    last_emotion_sent_index = current_length - 1

                # 2. Check if it's time for Gemini analysis and if there's data
                if now - last_gemini_analysis_time >= GEMINI_ANALYSIS_INTERVAL_SECONDS and analysis_data:
                    # Prepare data batch (e.g., last N seconds)
                    cutoff_time = datetime.datetime.now() - datetime.timedelta(seconds=GEMINI_DATA_WINDOW_SECONDS)
                    # Filter data safely, handling potential format issues
                    current_analysis_batch = []
                    for d in analysis_data:
                        try:
                            if datetime.datetime.fromisoformat(d['timestamp']) >= cutoff_time:
                                current_analysis_batch.append(d)
                        except (KeyError, ValueError, TypeError):
                             app.logger.warning(f"Skipping malformed data point during batch creation: {d}")
                    if current_analysis_batch: # Only run if batch has valid data
                        should_run_gemini = True
                        last_gemini_analysis_time = now # Reset timer ONLY if we are going to run analysis

            # Yield new emotion data (outside the lock)
            if new_emotion_data_to_send:
                sse_data = {"type": "emotion_update", "payload": new_emotion_data_to_send} # Send as a batch
                json_data = json.dumps(sse_data)
                app.logger.debug(f"SERVER SENDING SSE (Emotion Batch): {len(new_emotion_data_to_send)} points")
                yield f"data: {json_data}\r\n\r\n"

            # Perform Gemini analysis and yield result (outside the lock)
            if should_run_gemini:
                app.logger.info(f"Triggering Gemini analysis for batch size: {len(current_analysis_batch)}")
                analysis_text = get_behavioral_analysis_from_gemini(current_analysis_batch)
                analysis_timestamp = datetime.datetime.now().isoformat()
                with analysis_lock:
                     # Store analysis with timestamp
                    behavior_analysis_history.append({
                        "timestamp": analysis_timestamp,
                        "analysis": analysis_text
                    })
                     # Optional: Limit history size
                     # MAX_HISTORY_SIZE = 100
                     # if len(behavior_analysis_history) > MAX_HISTORY_SIZE:
                     #     behavior_analysis_history = behavior_analysis_history[-MAX_HISTORY_SIZE:]

                # Send the new analysis text
                sse_data = {"type": "behavior_analysis", "payload": {"timestamp": analysis_timestamp, "analysis": analysis_text}}
                json_data = json.dumps(sse_data)
                app.logger.debug(f"SERVER SENDING SSE (Behavior): {json_data[:100]}...")
                yield f"data: {json_data}\r\n\r\n"

            # Wait before next check to avoid busy-waiting
            time.sleep(0.5) # Check for updates twice per second

    except GeneratorExit:
        # This occurs when the client disconnects
        app.logger.info("Client disconnected from SSE stream.")
    except Exception as e:
        # Log any other errors occurring in the stream loop
        app.logger.error(f"Error in SSE generate_analysis_stream: {e}", exc_info=True)
    finally:
        # Optional cleanup specific to this generator if needed
        app.logger.info("Stopped sending SSE data for this client.")


# --- Authentication Middleware --- (Modified to pass user_id)
def token_required(f):
    """Decorator to ensure user is authenticated via Firebase token (for web pages)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('firebaseToken') # Prefer secure HttpOnly cookie
        if not token:
            app.logger.warning("No auth token cookie found, redirecting to login.")
            return redirect(url_for('login', next=request.url)) # Redirect to login

        try:
            if token.startswith('Bearer '): # Should not happen with cookies, but good practice
                token = token.split(' ')[1]

            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token.get('uid')
            if not user_id:
                 raise ValueError("Token does not contain a UID.")

            app.logger.debug(f"Token verified successfully for user: {user_id}")
            # Pass the user_id as the first argument to the wrapped function
            return f(user_id, *args, **kwargs)

        except auth.ExpiredIdTokenError:
            app.logger.warning("Expired auth token, redirecting to login.")
            response = make_response(redirect(url_for('login', next=request.url)))
            response.delete_cookie('firebaseToken')
            return response
        except Exception as e:
            app.logger.error(f"Token verification failed: {e}, redirecting to login.")
            response = make_response(redirect(url_for('login', next=request.url)))
            response.delete_cookie('firebaseToken')
            return response
    return decorated_function

# --- API Authentication Middleware (Example - if needed for separate API endpoints) ---
def api_token_required(f):
    """Decorator for API endpoints requiring a token (e.g., in Authorization header)."""
    @wraps(f)
    def decorated_api(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            app.logger.warning("API request missing or invalid Bearer token.")
            return jsonify({'message': 'Authorization token is missing or invalid!'}), 401

            token = token.split(' ')[1]
        try:
            decoded_token = auth.verify_id_token(token)
            # Add user info to request context or pass as args if needed
            # g.user_id = decoded_token.get('uid')
            app.logger.debug(f"API Token verified for user: {decoded_token.get('uid')}")
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"API Token verification failed: {e}")
            return jsonify({'message': f'Invalid or expired token: {e}'}), 403
    return decorated_api


# --- Main Routes --- (Updated signature)
@app.route('/')
@token_required # Protect the root/home route
def home(user_id):
    """Redirects authenticated users to the main dashboard."""
    return redirect(url_for('dashboard'))

# --- Dashboard Routes --- (Updated signatures and logic)
@app.route('/dashboard')
@token_required
def dashboard(user_id):
    """Renders the main dashboard overview page."""
    user_specific_cases = user_cases.get(user_id, [])

    # --- Metrics for Current User ---
    total_current_user_cases = len(user_specific_cases)
    current_user_processing_count = sum(1 for case in user_specific_cases if case.get('status') == 'Processing')
    current_user_analyzed_count = sum(1 for case in user_specific_cases if case.get('status') == 'Analyzed - Report Ready')
    current_user_new_count = sum(1 for case in user_specific_cases if case.get('status') == 'New')

    # --- Metrics for All Users (System-Wide) ---
    total_system_users = len(user_cases)
    all_cases_flat = [case for cases_list in user_cases.values() for case in cases_list]
    total_system_cases = len(all_cases_flat)
    total_system_processing_count = sum(1 for case in all_cases_flat if case.get('status') == 'Processing')
    total_system_analyzed_count = sum(1 for case in all_cases_flat if case.get('status') == 'Analyzed - Report Ready')
    total_system_new_count = sum(1 for case in all_cases_flat if case.get('status') == 'New')

    # Get recent cases for the current user
    try:
        recent_cases = sorted(
            user_specific_cases,
            key=lambda x: datetime.datetime.fromisoformat(x.get('created_at', '1970-01-01T00:00:00')),
            reverse=True
        )[:5]
    except Exception as e:
        app.logger.warning(f"Error sorting cases for user {user_id}: {e}")
        recent_cases = user_specific_cases[:5]

    # --- Get Current Live Analysis Status ---
    active_case_details = None
    live_status_message = "Idle"
    global is_processing, current_analysis_case_id # Ensure we use the global ones

    if is_processing and current_analysis_case_id:
        active_case_details = next((c for c in user_specific_cases if c['id'] == current_analysis_case_id), None)
        if active_case_details:
            live_status_message = f"Processing: {active_case_details.get('name', 'Unknown Case')}"
        else:
            # If the processing case doesn't belong to the current user, or if it's an admin view (not implemented yet)
            # For now, just indicate something is processing.
            # We might need to fetch the case name from all_cases_flat if it's a system-wide view.
            # However, `current_analysis_case_id` is singular, implying one active analysis globally.
            generic_active_case = next((c for c in all_cases_flat if c['id'] == current_analysis_case_id), None)
            if generic_active_case:
                live_status_message = f"Processing: {generic_active_case.get('name', 'Unknown Case')} (by user {generic_active_case.get('user_id')})"
            else:
                live_status_message = f"Processing: Active Analysis (ID: {current_analysis_case_id})"
            app.logger.info(f"Active case ID {current_analysis_case_id} is processing, but not for current user {user_id} or details not found in user's list.")

    # --- Prepare data for template ---
    dashboard_data = {
        'total_current_user_cases': total_current_user_cases,
        'current_user_processing_count': current_user_processing_count,
        'current_user_analyzed_count': current_user_analyzed_count,
        'current_user_new_count': current_user_new_count,

        'total_system_users': total_system_users,
        'total_system_cases': total_system_cases,
        'total_system_processing_count': total_system_processing_count,
        'total_system_analyzed_count': total_system_analyzed_count,
        'total_system_new_count': total_system_new_count,

        'live_status_message': live_status_message,
        'is_processing': is_processing, # Pass the boolean flag for live status
        'active_case_details_for_current_user': active_case_details, # Pass the active case IF it belongs to current user

        'recent_cases': recent_cases
    }

    return render_template('dashboard/dashboard.html', data=dashboard_data)

# --- Case Management Routes --- (New and Updated)
@app.route('/cases', methods=['GET'])
@token_required
def cases(user_id): # Accepts user_id from decorator
    """Displays the user's cases and a form to create new ones."""
    user_specific_cases = user_cases.get(user_id, [])
    return render_template('dashboard/cases.html', cases=user_specific_cases)

@app.route('/cases/create', methods=['POST'])
@token_required
def create_case(user_id):
    """Creates a new case for the logged-in user."""
    case_name = request.form.get('case_name')
    case_bio = request.form.get('case_bio')

    if not case_name:
        # Add proper error handling/flash message later
        return "Case name is required", 400

    new_case = {
        'id': uuid.uuid4().hex, # Generate unique ID
        'name': case_name,
        'bio': case_bio,
        'created_at': datetime.datetime.now().isoformat(),
        'user_id': user_id, # Associate with user
        'status': 'New' # Add initial status
    }

    # Lock access while modifying the shared dictionary
    with file_lock: # Use the same lock for consistency
        if user_id not in user_cases:
            user_cases[user_id] = []
        user_cases[user_id].append(new_case)
        app.logger.info(f"Case '{case_name}' created for user {user_id}")

    save_cases_to_json() # Save the updated data to the file

    return redirect(url_for('cases'))

@app.route('/cases/<case_id>/analyze')
@token_required
def start_case_analysis(user_id, case_id):
    """Renders the analysis page for a specific case."""
    # Find the case to ensure it belongs to the user (optional but recommended)
    users_cases_list = user_cases.get(user_id, [])
    case = next((c for c in users_cases_list if c['id'] == case_id), None)

    if not case:
        return "Case not found or access denied", 404

    # Render a new template dedicated to the analysis session
    return render_template('dashboard/analysis_session.html', case=case)

# --- Other Dashboard Routes --- (Updated signatures)
@app.route('/interrogations')
@token_required
def interrogations(user_id):
    """Renders the interrogations records page, showing all cases from all users."""
    
    all_cases_list = []
    # Iterate through all users and their cases
    for u_id, cases_for_user in user_cases.items():
        for case in cases_for_user:
            # Ensure user_id is part of the case dict, which it should be from create_case
            # If not, you might want to add it here: case['owner_id'] = u_id
            all_cases_list.append(case)

    # Sort all cases by creation date, newest first (optional, but good for display)
    try:
        sorted_cases = sorted(
            all_cases_list,
            key=lambda x: datetime.datetime.fromisoformat(x.get('created_at', '1970-01-01T00:00:00')),
            reverse=True
        )
    except Exception as e:
        app.logger.warning(f"Error sorting all cases for interrogations page: {e}")
        sorted_cases = all_cases_list # Fallback to unsorted

    return render_template('dashboard/interrogations.html', cases=sorted_cases, view_all_users=True)

@app.route('/analytics')
@token_required
def analytics(user_id):
    """Renders the analytics visualization page."""
    # We might use user_id later here
    return render_template('dashboard/analytics.html')

# --- Auth Routes --- (No change needed for login/register/forgot/mfa rendering)
@app.route('/auth/login')
def login():
    """Renders the login page or redirects if already logged in."""
    token = request.cookies.get('firebaseToken')
    if token:
        try:
            # Verify the token without redirecting immediately from here
            # Let the JS handle the redirect after MFA or dashboard load
            auth.verify_id_token(token)
            # If token is valid, maybe redirect to dashboard directly?
            # Or let the frontend JS decide based on auth state?
            # For simplicity, let's redirect to dashboard if token is valid
            app.logger.debug("Valid token found in cookie, redirecting to dashboard.")
            return redirect(url_for('dashboard')) # Redirect to dashboard
        except Exception as e:
            app.logger.debug(f"Invalid/Expired token in cookie ({e}), showing login page.")
            # Clear the invalid cookie before rendering login
            resp = make_response(render_template('auth/login.html'))
            resp.delete_cookie('firebaseToken')
            return resp
    # No token or invalid token, render login page
    return render_template('auth/login.html')


@app.route('/auth/register')
def register():
    """Renders the registration page."""
    return render_template('auth/register.html')

@app.route('/auth/forgot')
def forgot_password():
    """Renders the forgot password page."""
    return render_template('auth/forgot.html')

@app.route('/auth/mfa')
def mfa_verify():
    """Renders the MFA verification page."""
     # Ideally, protect this route too, ensuring user tried to log in first
    return render_template('auth/mfa.html')

# --- Auth API --- (No change needed for set_auth_cookie)
@app.route('/set_auth_cookie', methods=['POST'])
# @api_token_required # Protect this endpoint - only valid tokens should set cookies
def set_auth_cookie():
    """Sets the Firebase token as an HttpOnly cookie."""
    # Check if Firebase is initialized
    try:
        firebase_admin.get_app()
    except ValueError:
        app.logger.error("Firebase not initialized - cannot verify token")
        return jsonify({"message": "Firebase authentication not configured"}), 503
    
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
         return jsonify({"message": "Authorization token missing or invalid"}), 401

    token = token.split(' ')[1]
    try:
        # First attempt to verify the token
        auth.verify_id_token(token)
    except auth.InvalidIdTokenError as e:
        if "Token used too early" in str(e):
            app.logger.warning(f"Token used too early, attempting a small delay and retry: {e}")
            time.sleep(10) # Pause for 2 seconds to allow clocks to catch up for minor skew
            try:
                auth.verify_id_token(token) # Retry verification
                app.logger.info("Token verification successful after retry.")
            except Exception as retry_e:
                app.logger.error(f"Token verification failed after retry: {retry_e}")
                return jsonify({"message": f"Invalid token after retry: {retry_e}"}), 403
        else:
            # Other types of InvalidIdTokenError (e.g., expired, malformed)
            app.logger.error(f"Token verification failed (InvalidIdTokenError other than 'too early'): {e}")
            return jsonify({"message": f"Invalid token: {e}"}), 403
    except Exception as e: # Catch any other non-InvalidIdTokenError exceptions during initial verification
        app.logger.error(f"General token verification error during cookie setting: {e}")
        return jsonify({"message": f"Invalid token: {e}"}), 403

    # If verification is successful (either first try or after retry):
    response = make_response(jsonify({"message": "Token processed"}))
    # Set HttpOnly cookie for security
    response.set_cookie(
        'firebaseToken',
        token,
        max_age=3600,  # 1 hour expiry, match Firebase token expiry if possible
        httponly=True, # Prevent JS access
        secure=request.is_secure, # True if using HTTPS
        samesite='Lax' # Recommended for most cases
    )
    app.logger.info("Auth token cookie set.")
    return response

# --- Logout --- (No change needed)
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """Logs the user out by clearing the cookie."""
    app.logger.info("Logout route called, clearing cookie and redirecting.")
    response = make_response(redirect(url_for('login')))
    response.delete_cookie('firebaseToken') # Clear the HttpOnly cookie
    return response

# --- Real-time Analysis API Routes --- (Updated signatures)
@app.route('/start_analysis', methods=['POST'])
@token_required
def start_analysis(user_id):
    """Starts the video capture and analysis thread for a specific case."""
    global video_thread, is_processing, analysis_data, behavior_analysis_history, capture
    global current_analysis_case_id # Access the global tracker

    # --- Get Case ID from request ---
    request_data = request.get_json()
    case_id = request_data.get('case_id') if request_data else None

    if not case_id:
        app.logger.error("'/start_analysis' called without case_id.")
        return jsonify({"status": "error", "message": "Missing case_id"}), 400

    app.logger.info(f"'/start_analysis' called by user: {user_id} for case: {case_id}")

    if is_processing:
        if case_id == current_analysis_case_id:
             app.logger.warning(f"Analysis already running for case {case_id}.")
             return jsonify({"status": "already_running", "message": "Analysis already running for this case."}), 400
        else:
             app.logger.warning(f"Another analysis (case {current_analysis_case_id}) is already running.")
             return jsonify({"status": "already_running", "message": f"Another analysis is already running (Case: {current_analysis_case_id}). Stop it first."}), 409 # Conflict

    # --- Update Case Status to Processing ---
    case_updated = False
    with file_lock:
        if user_id in user_cases and any(c['id'] == case_id for c in user_cases[user_id]):
            for case in user_cases[user_id]:
                if case['id'] == case_id:
                    case['status'] = 'Processing'
                    case_updated = True
                    break
        else:
             app.logger.error(f"Case {case_id} not found for user {user_id} during start.")
             return jsonify({"status": "error", "message": "Case not found or access denied"}), 404

    if case_updated:
        save_cases_to_json() # Save status update
    # ------------------------------------

    # Reset global data for the new session
    with analysis_lock:
        analysis_data = []
        behavior_analysis_history = []
        last_frame_analyzed = None
        app.logger.info("Global analysis data cleared for new session.")

    current_analysis_case_id = case_id # Track the active case
    is_processing = True
    if capture:
         capture.release()
         capture = None
         app.logger.info("Released previous capture object before starting new analysis.")

    video_thread = threading.Thread(target=capture_and_analyze, daemon=True)
    video_thread.start()
    app.logger.info(f"Analysis thread started for case {case_id}.")
    return jsonify({"status": "started"})

@app.route('/stop_analysis', methods=['POST'])
@token_required
def stop_analysis(user_id):
    """Stops the video capture and analysis thread."""
    global is_processing, video_thread, capture, current_analysis_case_id # Access tracker

    active_case_id = current_analysis_case_id # Get the ID of the case that was running

    app.logger.info(f"'/stop_analysis' called by user: {user_id} for case: {active_case_id}")

    if not is_processing:
        app.logger.warning("Analysis not running.")
        current_analysis_case_id = None
        return jsonify({"status": "not_running"}), 400

    is_processing = False # Signal the thread to stop

    # --- Stop the thread ---
    if video_thread and video_thread.is_alive():
        video_thread.join(timeout=2.0)
        if video_thread.is_alive():
             app.logger.warning(f"Analysis thread for case {active_case_id} did not stop gracefully.")
             if capture: capture.release(); capture = None; app.logger.warning("Force released capture object.")
        else:
             app.logger.info(f"Analysis thread stopped successfully for case {active_case_id}.")
    video_thread = None
    if capture: capture.release(); capture = None; app.logger.info("Ensured capture object released.")
    # ---------------------

    # --- Update Case Status ---
    case_updated = False
    if active_case_id: # Only update if we know which case was running
        with file_lock:
            if user_id in user_cases and any(c['id'] == active_case_id for c in user_cases[user_id]):
                for case in user_cases[user_id]:
                    if case['id'] == active_case_id:
                        case['status'] = 'Analyzed - Report Ready'
                        case_updated = True
                        app.logger.info(f"Updated status for case {active_case_id} to 'Analyzed - Report Ready'.")
                        break
            else:
                 app.logger.warning(f"Could not find case {active_case_id} for user {user_id} to update status after stop.")

        if case_updated:
            save_cases_to_json() # Save status update

    current_analysis_case_id = None # Clear the tracker
    # ------------------------

    return jsonify({"status": "stopped"})

@app.route('/analysis_stream')
@token_required
def analysis_stream(user_id): # Still needs user_id due to decorator
    """Endpoint to stream analysis data using Server-Sent Events (SSE)."""
    app.logger.info(f"Client connected to SSE analysis stream (User: {user_id}).")
    # stream_with_context ensures the app context is available in the generator
    return Response(stream_with_context(generate_analysis_stream()), mimetype='text/event-stream')

# --- Reporting and Data Export Routes --- (Updated signatures and download route)
@app.route('/get_report_data', methods=['GET'])
@token_required
def get_report_data(user_id):
    """Returns the collected analysis data for report generation."""
    with analysis_lock:
        # Return copies to avoid concurrent modification issues if needed elsewhere
        report_data = list(analysis_data)
        history = list(behavior_analysis_history)
    app.logger.info(f"Providing report data for user {user_id}: {len(report_data)} emotion points, {len(history)} analyses.")
    return jsonify({"emotion_data": report_data, "behavior_analysis": history})

@app.route('/download_report/<case_id>') # Add case_id to URL
@token_required
def download_report(user_id, case_id): # Add user_id, case_id
    """Generates and downloads a session report for the *last completed* analysis."""
    # Find the case details (optional, for report title)
    users_cases_list = user_cases.get(user_id, [])
    case = next((c for c in users_cases_list if c['id'] == case_id), None)
    case_name = case['name'] if case else 'UnknownCase'

    app.logger.info(f"Generating report for user {user_id}, case: {case_name} ({case_id})")
    try:
        # Pass case_name to generate_report
        report_content = generate_report(case_name) # Generate the report content using global data
        response = Response(report_content, mimetype='text/plain') # Or 'text/csv'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include case name (sanitized) in filename
        safe_case_name = "".join(c if c.isalnum() else "_" for c in case_name)
        response.headers.set('Content-Disposition', 'attachment', filename=f'analysis_report_{safe_case_name}_{timestamp}.txt')
        return response
    except Exception as e:
        app.logger.error(f"Error generating report for case {case_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate report"}), 500

# Modified generate_report function signature
def generate_report(case_name="Unknown Case"):
    """Generates a simple text report from the session data."""
    report = StringIO() # Use StringIO to build the report in memory
    report.write("Interrogation Analysis Report\n")
    report.write("=============================\n")
    report.write(f"Case: {case_name}\n") # Add case name
    report.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    with analysis_lock:
        # Uses the current global data
        emo_data = list(analysis_data)
        behav_hist = list(behavior_analysis_history)

    # --- Summary ---
    report.write("--- Session Summary ---\n")
    if not emo_data:
        report.write("No emotion data recorded for this session.\n")
    else:
        try: # Add try-except for timestamp parsing
            start_time = datetime.datetime.fromisoformat(emo_data[0]['timestamp'])
            end_time = datetime.datetime.fromisoformat(emo_data[-1]['timestamp'])
            duration = end_time - start_time
            report.write(f"Session Start: {start_time.strftime('%H:%M:%S')}\n")
            report.write(f"Session End:   {end_time.strftime('%H:%M:%S')}\n")
            report.write(f"Duration:      {str(duration).split('.')[0]}\n") # Format duration nicely
        except (ValueError, KeyError, IndexError) as e:
             report.write("Could not determine session start/end/duration.\n")
             app.logger.warning(f"Error parsing timestamps for report summary: {e}")

        # Calculate dominant emotion distribution (example)
        dominant_emotions = [d['dominant_emotion'] for d in emo_data if d.get('dominant_emotion')]
        if dominant_emotions:
             from collections import Counter
             emotion_counts = Counter(dominant_emotions)
             total_readings = len(dominant_emotions)
             report.write("\nDominant Emotion Distribution:\n")
             # Sort by count descending for clarity
             for emotion, count in sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True):
                 percentage = (count / total_readings) * 100
                 report.write(f"- {emotion.capitalize()}: {count} readings ({percentage:.1f}%)\n")
        else:
            report.write("No dominant emotions recorded.\n")

    report.write("\n--- Behavioral Analysis Notes (AI Generated) ---\n")
    if not behav_hist:
        report.write("No AI behavioral analysis notes generated for this session.\n")
    else:
        for i, entry in enumerate(behav_hist):
            try: # Add try-except for timestamp parsing
                 ts = datetime.datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')
                 report.write(f"[{ts}] Note {i+1}: {entry['analysis']}\n")
            except (ValueError, KeyError) as e:
                 report.write(f"[Timestamp Error] Note {i+1}: {entry.get('analysis', 'N/A')}\n")
                 app.logger.warning(f"Error parsing timestamp for behavior note {i+1}: {e}")

    report.seek(0) # Rewind StringIO object to the beginning
    return report.getvalue()


# --- Session Management (Placeholder/Example) --- (Updated signatures)
SAVE_DIR = "saved_sessions"
ensure_dir(SAVE_DIR)

@app.route('/save_session', methods=['POST'])
@token_required
def save_session(user_id):
    """Saves the current session data to a file. Needs rework for per-user/case saving."""
    # WARNING: This currently saves the GLOBAL analysis data, not specific to user/case
    # Needs significant rework if multiple users/cases need separate saves
    try:
        # Add user_id to filename? Needs careful thought on structure.
        filename_base = request.json.get('filename', f"session_{user_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        filename = os.path.join(SAVE_DIR, f"{filename_base}.json")

        with analysis_lock:
            session_data = {
                "emotion_data": list(analysis_data),
                "behavior_analysis": list(behavior_analysis_history),
                "saved_at": datetime.datetime.now().isoformat(),
                "saved_by": user_id # Record who saved it
            }

        with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
        app.logger.info(f"Session data saved to {filename} by user {user_id}")
        return jsonify({"status": "saved", "filename": filename})
    except Exception as e:
        app.logger.error(f"Error saving session for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to save session"}), 500


@app.route('/list_sessions', methods=['GET'])
@token_required
def list_sessions(user_id):
    """Lists available saved session files. Needs filtering per user."""
    # WARNING: Currently lists ALL sessions. Need filtering logic.
    try:
        # Modify to list only sessions relevant to the user_id if possible (e.g., filename convention)
        all_sessions = [f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]
        user_session_files = [f for f in all_sessions if f.startswith(f"session_{user_id}_")] # Example filter

        session_details = []
        for fname in user_session_files:
            try:
                session_details.append({"filename": fname})
            except Exception as e:
                app.logger.warning(f"Could not process session file {fname}: {e}")
                session_details.append({"filename": fname, "error": "Could not process file"})

        return jsonify({"sessions": session_details})
    except Exception as e:
        app.logger.error(f"Error listing sessions for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to list sessions"}), 500


@app.route('/load_session', methods=['POST'])
@token_required
def load_session(user_id):
    """Loads data from a saved session file. Needs validation against user."""
    # WARNING: Needs security check to ensure user can only load their sessions
    global analysis_data, behavior_analysis_history
    try:
        filename = request.json.get('filename')
        if not filename or '..' in filename or filename.startswith('/'):
             return jsonify({"error": "Invalid filename"}), 400

        # SECURITY CHECK: Ensure the requested filename belongs to the user
        if not filename.startswith(f"session_{user_id}_"): # Basic check
             app.logger.warning(f"User {user_id} attempted to load unauthorized session: {filename}")
             return jsonify({"error": "Access denied to session file"}), 403

        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
             return jsonify({"error": "Session file not found"}), 404
            
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        if 'emotion_data' not in loaded_data or 'behavior_analysis' not in loaded_data:
            return jsonify({"error": "Invalid session file format"}), 400

        # Load the data into the GLOBAL variables (overwrites current session)
        with analysis_lock:
            analysis_data = loaded_data['emotion_data']
            behavior_analysis_history = loaded_data['behavior_analysis']
            app.logger.info(f"Loaded session data from {filename} by user {user_id}. Global data overwritten.")

        return jsonify({"status": "loaded", "filename": filename})
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding JSON from session file: {filename}")
        return jsonify({"error": "Invalid JSON in session file"}), 500
    except Exception as e:
        app.logger.error(f"Error loading session {filename} for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to load session"}), 500


# --- Data Retrieval for Charts/Frontend --- (Updated signature)
@app.route('/get_emotion_trends', methods=['GET'])
@token_required
def get_emotion_trends(user_id):
    """Provides data formatted for charting (e.g., Chart.js). Reads GLOBAL data."""
    try:
        with analysis_lock:
            data = list(analysis_data) # Get a copy

        if not data:
            return jsonify({'labels': [], 'datasets': []})

        # Prepare data for Chart.js
        labels = []
        datasets = {
            'Happy': [], 'Sad': [], 'Angry': [], 'Surprise': [],
            'Fear': [], 'Disgust': [], 'Neutral': []
            # Add other emotions if tracked by DeepFace version
        }
        emotion_keys = list(datasets.keys()) # Get keys dynamically

        # Process data points
        for entry in data:
            try:
                # Format timestamp for label
                ts_dt = datetime.datetime.fromisoformat(entry['timestamp'])
                labels.append(ts_dt.strftime('%H:%M:%S')) # Use time as label

                emotions = entry.get('emotions', {})
                if not emotions: # Handle cases where emotions might be missing
                    for key in emotion_keys:
                         datasets[key].append(0) # Append 0 if no emotion data
                         continue
            
                # Append scores for each emotion, defaulting to 0 if not present
                for key_lower in datasets.keys():
                    # DeepFace keys are lowercase, match them
                    score = emotions.get(key_lower.lower(), 0)
                    # Ensure score is a number, default to 0 if not
                    datasets[key_lower].append(float(score) if isinstance(score, (int, float)) else 0)

            except (ValueError, TypeError, KeyError) as proc_err:
                app.logger.warning(f"Skipping data point due to processing error: {proc_err} in {entry}")
                # Attempt to keep datasets aligned by appending 0s if an error occurs mid-point
                if len(labels) > len(datasets['Neutral']): # Check if datasets are lagging
                    for key in emotion_keys:
                        if len(datasets[key]) < len(labels):
                           datasets[key].append(0)

        # Format datasets for Chart.js
        chartjs_datasets = []
        # Define colors for consistency
        colors = {
            'Happy': 'rgba(255, 206, 86, 0.7)', 'Sad': 'rgba(54, 162, 235, 0.7)',
            'Angry': 'rgba(255, 99, 132, 0.7)', 'Surprise': 'rgba(153, 102, 255, 0.7)',
            'Fear': 'rgba(75, 192, 192, 0.7)', 'Disgust': 'rgba(255, 159, 64, 0.7)',
            'Neutral': 'rgba(201, 203, 207, 0.7)'
        }
        border_colors = { k: v.replace('0.7', '1') for k, v in colors.items() }

        for key in emotion_keys:
             # Only include datasets with non-zero data if desired
             # if any(s > 0 for s in datasets[key]):
             chartjs_datasets.append({
                 'label': key,
                 'data': datasets[key],
                 'borderColor': border_colors.get(key, '#000'),
                 'backgroundColor': colors.get(key, '#AAA'),
                 'fill': False, # Line chart - don't fill area under line
                 'tension': 0.1 # Slightly curve the line
             })

        return jsonify({'labels': labels, 'datasets': chartjs_datasets})

    except Exception as e:
        app.logger.error(f"Error getting emotion trends: {e}", exc_info=True)
        return jsonify({"error": "Failed to process emotion trends"}), 500

# --- Data Export --- (Updated signatures)
@app.route('/export_csv', methods=['GET'])
@token_required
def export_csv(user_id):
    """Exports the detailed emotion log as a CSV file. Reads GLOBAL data."""
    try:
        with analysis_lock:
            data = list(analysis_data) # Get a copy

        if not data:
            return jsonify({"message": "No data to export"}), 404

        si = StringIO()
        # Define header row dynamically based on keys in the first valid entry
        header = ['Timestamp', 'Dominant Emotion']
        if data:
            first_emotions = data[0].get('emotions', {})
            if first_emotions:
                 header.extend([key.capitalize() for key in first_emotions.keys()])

        import csv
        writer = csv.writer(si)
        writer.writerow(header)

        for entry in data:
            ts = entry.get('timestamp', 'N/A')
            dom_emo = entry.get('dominant_emotion', 'N/A')
            emotions = entry.get('emotions', {})
            row = [ts, dom_emo]
            # Add emotion scores in the order of the header (minus first two cols)
            for key_cap in header[2:]:
                 score = emotions.get(key_cap.lower(), 0) # Match lowercase keys from DeepFace
                 row.append(f"{float(score):.4f}" if isinstance(score, (int, float)) else '0.0000')
            writer.writerow(row)

        output = make_response(si.getvalue())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output.headers["Content-Disposition"] = f"attachment; filename=emotion_log_{timestamp}.csv"
        output.headers["Content-type"] = "text/csv"
        app.logger.info(f"Exporting data as CSV for user {user_id}.")
        return output
    except Exception as e:
        app.logger.error(f"Error exporting CSV for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to export CSV"}), 500

@app.route('/export_excel', methods=['GET'])
@token_required
def export_excel(user_id):
    """Exports the detailed emotion log as an Excel file. Reads GLOBAL data."""
    try:
        with analysis_lock:
            data = list(analysis_data)

        if not data:
            return jsonify({"message": "No data to export"}), 404

        df_data = []
        for entry in data:
             row = {
                 'Timestamp': entry.get('timestamp', None),
                 'Dominant Emotion': entry.get('dominant_emotion', 'N/A')
             }
             emotions = entry.get('emotions', {})
             for key, score in emotions.items():
                  row[key.capitalize()] = float(score) if isinstance(score, (int, float)) else 0
             df_data.append(row)

        df = pd.DataFrame(df_data)
        # Optional: Convert timestamp string to datetime objects for Excel formatting
        try:
             df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except Exception as dt_err:
             app.logger.warning(f"Could not parse timestamps for Excel export: {dt_err}")


        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Emotion Log')
            # You could add the behavioral analysis to another sheet
            behav_df = pd.DataFrame(behavior_analysis_history)
            if not behav_df.empty:
                try:
                    behav_df['timestamp'] = pd.to_datetime(behav_df['timestamp'])
                except Exception: pass # Ignore parse errors for behavior sheet
                behav_df.to_excel(writer, index=False, sheet_name='Behavior Analysis')

        output.seek(0)

        response = make_response(output.getvalue())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        response.headers["Content-Disposition"] = f"attachment; filename=analysis_report_{timestamp}.xlsx"
        response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        app.logger.info(f"Exporting data as Excel for user {user_id}.")
        return response
    except Exception as e:
        app.logger.error(f"Error exporting Excel for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to export Excel"}), 500

# --- Jinja Setup ---
def string_to_datetime_filter(iso_string):
    """Jinja filter to safely convert ISO string to datetime object."""
    if not iso_string:
        return None
    try:
        # Handle potential timezone info if present (adjust format if needed)
        if '+' in iso_string and ':' == iso_string[-3:-2]: # Basic check for +HH:MM offset
             iso_string = iso_string[:-3] + iso_string[-2:] # Remove colon
        elif 'Z' in iso_string: # Handle Z for UTC
             iso_string = iso_string.replace('Z', '+0000')

        # Python < 3.11 might need manual parsing for some ISO formats
        # Try direct fromisoformat first
        try:
            return datetime.datetime.fromisoformat(iso_string)
        except ValueError:
             # Fallback for common formats if needed (adjust based on your actual timestamp format)
             formats_to_try = [
                 "%Y-%m-%dT%H:%M:%S.%f%z", # with microseconds and timezone
                 "%Y-%m-%dT%H:%M:%S%z",    # without microseconds, with timezone
                 "%Y-%m-%dT%H:%M:%S.%f",  # with microseconds, no timezone
                 "%Y-%m-%dT%H:%M:%S"      # without microseconds, no timezone
             ]
             for fmt in formats_to_try:
                 try:
                     return datetime.datetime.strptime(iso_string, fmt)
                 except ValueError:
                     continue
             raise ValueError(f"Could not parse timestamp: {iso_string}") # Raise if all fail
            
    except Exception as e:
        app.logger.warning(f"Could not convert timestamp '{iso_string}' to datetime: {e}")
        return None

# Register the filter and make datetime available globally
app.jinja_env.filters['string_to_datetime'] = string_to_datetime_filter
app.jinja_env.globals['modules'] = {'datetime': datetime}

# === Browser webcam routes (add-on) ===
# These routes provide a publicly accessible camera page for visitor emotion analysis
# Added as isolated feature - does not modify existing analysis logic or global state

# Import additional dependencies for browser webcam feature
try:
    import base64
    import numpy as np
    from flask_cors import CORS
except ImportError as e:
    app.logger.warning(f"Optional dependencies for camera feature not fully available: {e}")

# Enable CORS for API endpoints and auth routes (development - restrict origins in production)
try:
    CORS(app, 
         resources={
             r"/api/*": {"origins": "*", "supports_credentials": True},
             r"/set_auth_cookie": {"origins": "*", "supports_credentials": True},
             r"/static/js/*": {"origins": "*"}
         },
         supports_credentials=True)
    app.logger.info("CORS enabled for API and auth endpoints")
except:
    app.logger.warning("CORS could not be enabled - flask-cors may not be installed")

@app.route('/camera')
def camera_page():
    """
    Public camera page - allows visitors to use their webcam for emotion analysis.
    No auth decorator - page is publicly accessible but shows sign-in prompt.
    """
    app.logger.info("Camera page accessed")
    return render_template('dashboard/camera.html')

@app.route('/api/frame', methods=['POST'])
def process_frame():
    """
    Processes a single frame from browser webcam for emotion analysis.
    Requires authentication via cookie or Authorization header.
    Isolated from main analysis flow - does not modify global analysis_data.
    """
    # --- Authentication Check ---
    token = None
    user_id = None
    
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    
    # Fallback to cookie
    if not token:
        token = request.cookies.get('firebaseToken')
    
    if not token:
        app.logger.warning("Frame upload without auth token")
        return jsonify({"error": "auth required"}), 401
    
    # Verify token
    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token.get('uid')
        if not user_id:
            raise ValueError("Token missing UID")
        app.logger.debug(f"Frame processing for user: {user_id}")
    except Exception as e:
        app.logger.error(f"Token verification failed for frame upload: {e}")
        return jsonify({"error": "auth required"}), 401
    
    # --- Request Validation ---
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    image_base64 = data.get('image_base64')
    
    if not image_base64:
        return jsonify({"error": "Missing image_base64 field"}), 400
    
    # --- Decode and Size Check ---
    try:
        image_bytes = base64.b64decode(image_base64)
        
        # Reject oversized images (> 2.5MB)
        if len(image_bytes) > 2.5 * 1024 * 1024:
            app.logger.warning(f"Frame too large from user {user_id}: {len(image_bytes)} bytes")
            return jsonify({"error": "Image too large (max 2.5MB)"}), 413
        
        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Resize if too large (reduce CPU load)
        height, width = img.shape[:2]
        if width > 640:
            scale = 640.0 / width
            new_width = 640
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            app.logger.debug(f"Resized frame from {width}x{height} to {new_width}x{new_height}")
        
    except Exception as e:
        app.logger.error(f"Error decoding frame from user {user_id}: {e}")
        return jsonify({"error": "Invalid image encoding"}), 400
    
    # --- Emotion Analysis ---
    try:
        # Use DeepFace with enforce_detection=False to handle no-face frames gracefully
        results = DeepFace.analyze(
            img,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,  # Don't fail if no face detected
            silent=True
        )
        
        if results and isinstance(results, list) and len(results) > 0:
            result = results[0]
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            emotions = result.get('emotion', {})
            
            # Convert numpy types to Python native for JSON serialization
            emotions_clean = {k: float(v) for k, v in emotions.items()}
            
            app.logger.info(f"Frame analyzed for user {user_id}: {dominant_emotion}")
            
            return jsonify({
                "status": "ok",
                "dominant_emotion": dominant_emotion,
                "emotions": emotions_clean,
                "uid": user_id,
                "timestamp": datetime.datetime.now().isoformat()
            })
        else:
            # No face detected or empty result
            return jsonify({
                "status": "ok",
                "dominant_emotion": "no_face",
                "emotions": {},
                "uid": user_id,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
    except ValueError as e:
        # Handle "Face could not be detected" gracefully
        if "Face could not be detected" in str(e):
            app.logger.debug(f"No face in frame from user {user_id}")
            return jsonify({
                "status": "ok",
                "dominant_emotion": "no_face",
                "emotions": {},
                "uid": user_id
            })
        else:
            app.logger.error(f"DeepFace analysis error for user {user_id}: {e}")
            return jsonify({"error": "Analysis failed"}), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error analyzing frame for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Analysis failed"}), 500

@app.route('/static/js/firebase-config.js')
def firebase_config_js():
    """
    Serves Firebase configuration as a proper ES module.
    Exports 'auth' and 'firebaseConfig' for use by frontend modules.
    """
    # Build config from environment variables
    cfg = {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.getenv("FIREBASE_APP_ID", "")
    }
    
    # Add measurement ID if provided (optional)
    measurement_id = os.getenv("FIREBASE_MEASUREMENT_ID")
    if measurement_id:
        cfg["measurementId"] = measurement_id
    
    # Generate ES module JavaScript
    js_module = f'''import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-app.js";
import {{ getAuth }} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const firebaseConfig = {json.dumps(cfg, indent=2)};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export {{ firebaseConfig }};
'''
    
    return Response(js_module, mimetype='application/javascript')

# === End of browser webcam routes ===

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure save directory exists
    ensure_dir(SAVE_DIR)
    # Use waitress or gunicorn for production instead of Flask dev server
    # Example: waitress-serve --host 0.0.0.0 --port 5000 app:app
    app.run(debug=True, host='0.0.0.0', port=5000) # debug=True is NOT for production
