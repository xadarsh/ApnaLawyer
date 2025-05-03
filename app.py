# Standard libraries
import os
import re
import io
import json
import time
import base64
import hashlib
import tempfile
from datetime import datetime, timedelta

# Streamlit & UI components
import streamlit as st
import streamlit.components.v1 as components

# Firebase (Realtime + Pyrebase4)
import pyrebase
import firebase_admin
from firebase_admin import credentials, initialize_app, db as firebase_db

# Langchain core
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_together import Together

# NLP & translation
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
import textwrap

# Audio processing
import soundfile as sf
import numpy as np
import wave
import sounddevice as sd

# Utility
import requests
from dotenv import load_dotenv
from dateutil.parser import parse

# Local modules
from footer import footer

# ----------------- Streamlit Config -------------------
st.set_page_config(page_title="ApnaLawyer", layout="centered")

# ----------------- Load Environment Variables -------------------
load_dotenv()
SPEECHMATICS_API_KEY = os.getenv('SPEECHMATICS_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

if not TOGETHER_API_KEY:
    st.error("Please set TOGETHER_API_KEY environment variable")
    st.stop()

# ----------------- LLM Init -------------------
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=1024,
    together_api_key=TOGETHER_API_KEY
)

# ----------------- Firebase Init (Pyrebase) -------------------
firebase_config = {
    "apiKey": os.environ["FIREBASE_API_KEY"],
    "authDomain": os.environ["FIREBASE_AUTH_DOMAIN"],
    "databaseURL": os.environ["FIREBASE_DATABASE_URL"],
    "projectId": os.environ["FIREBASE_PROJECT_ID"],
    "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"],
    "messagingSenderId": os.environ["FIREBASE_MESSAGING_SENDER_ID"],
    "appId": os.environ["FIREBASE_APP_ID"]
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# ----------------- Firebase Admin SDK Init -------------------
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            creds_dict = json.loads(os.environ["FIREBASE_CREDS_JSON"])
            cred = credentials.Certificate(creds_dict)
            firebase_app = initialize_app(cred, {
                'databaseURL': os.environ["FIREBASE_DB_URL"]
            })
    except Exception as e:
        st.error(f"Firebase initialization error: {str(e)}")

initialize_firebase()

# ----------------- UI -------------------
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/banner.jpg", use_container_width=True)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ----------------- Translation Logic -------------------
supported_languages = {
    "hindi": "hi",
    "english": "en",
    "hinglish": "hi"
}

devanagari_regex = re.compile(r'[\u0900-\u097F]+')

def detect_target_language(prompt):
    prompt_lower = prompt.lower().strip()

    if re.search(r'[\u0C80-\u0CFF]', prompt):  # Kannada block
        return "hindi"

    for lang_name, lang_code in supported_languages.items():
        if f"in {lang_name}" in prompt_lower:
            return lang_name

    if devanagari_regex.search(prompt):
        return "hindi"

    if re.search(r'\bdhara\b|\bkanoon\b|\bnyay\b', prompt_lower) and detect(prompt) == 'en':
        return "hinglish"

    try:
        detected = detect(prompt)
        for name, code in supported_languages.items():
            if code == detected:
                return name
    except:
        pass

    return "english"

def translate_text(text, target_language):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e:
        return f"⚠ Translation failed: {str(e)}"

# ----------------- Login/Signup Interface -------------------
import json
import requests
from streamlit.components.v1 import html

# Custom CSS for the enhanced UI
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main container styles */
        .stApp {
            background: radial-gradient(circle at top left, #0f0f0f, #050505);
            color: white;
        }
        
        /* Sidebar styles */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.2) !important;
            border-radius: 20px !important;
            padding: 30px !important;
            margin: 20px !important;
            animation: fadeIn 1s ease !important;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Input field styles */
        .stTextInput>div>div>input, .stPassword>div>div>input {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput>div>div>input:focus, .stPassword>div>div>input:focus {
            border-color: #00ffff !important;
            background: rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.2) !important;
            outline: none !important;
        }
        
        /* Button styles */
        .stButton>button {
            width: 100% !important;
            padding: 12px 20px !important;
            border-radius: 10px !important;
            background: linear-gradient(45deg, #00ffff, #007fff) !important;
            color: black !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3) !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 255, 255, 0.4) !important;
        }
        
        /* Radio button styles */
        .stRadio>div {
            flex-direction: column !important;
            gap: 10px !important;
        }
        
        .stRadio>div>label {
            color: white !important;
            font-weight: 500 !important;
            padding: 8px 12px !important;
            border-radius: 8px !important;
            background: rgba(255, 255, 255, 0.05) !important;
            transition: all 0.3s ease !important;
        }
        
        .stRadio>div>label:hover {
            background: rgba(255, 255, 255, 0.1) !important;
        }
        
        .stRadio>div>label[data-baseweb="radio"]>div:first-child {
            border-color: #00ffff !important;
        }
        
        /* Link styles */
        a {
            color: #00ffff !important;
            text-decoration: none !important;
            transition: all 0.3s ease !important;
        }
        
        a:hover {
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5) !important;
        }
        
        /* Error message styles */
        .stAlert {
            border-radius: 10px !important;
            background: rgba(255, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 0, 0, 0.2) !important;
        }
        
        /* Success message styles */
        .stSuccess {
            border-radius: 10px !important;
            background: rgba(0, 255, 0, 0.1) !important;
            border: 1px solid rgba(0, 255, 0, 0.2) !important;
        }
        /* Audio recording animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .recording-active {
            animation: pulse 1.5s infinite;
            color: #ff4b4b !important;
        }
    </style>
    """, unsafe_allow_html=True)

def store_user_data(user_id, name, email):
    try:
        ref = firebase_db.reference(f'users/{user_id}')
        ref.set({
            'name': name,
            'email': email,
            'created_at': time.time()
        })
        return True
    except Exception as e:
        st.error(f"Error storing user data: {str(e)}")
        return False

# Add these Firebase functions right after your existing Firebase functions
def save_chat_to_history(user_id, chat_title, messages):
    try:
        ref = firebase_db.reference(f'users/{user_id}/chats')
        new_chat_ref = ref.push()
        new_chat_ref.set({
            'title': chat_title,
            'messages': messages,
            'timestamp': time.time(),
            'last_updated': time.time()
        })
        return new_chat_ref.key
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")
        return None

def update_chat_history(user_id, chat_id, messages):
    try:
        ref = firebase_db.reference(f'users/{user_id}/chats/{chat_id}')
        ref.update({
            'messages': messages,
            'last_updated': time.time()
        })
        return True
    except Exception as e:
        st.error(f"Error updating chat history: {str(e)}")
        return False

def get_chat_history(user_id):
    try:
        ref = firebase_db.reference(f'users/{user_id}/chats')
        chats = ref.get()
        if chats:
            return sorted(
                [(chat_id, chat_data) for chat_id, chat_data in chats.items()],
                key=lambda x: x[1]['last_updated'],
                reverse=True
            )
        return []
    except Exception as e:
        st.error(f"Error fetching chat history: {str(e)}")
        return []

def delete_chat_history(user_id, chat_id):
    try:
        ref = firebase_db.reference(f'users/{user_id}/chats/{chat_id}')
        ref.delete()
        return True
    except Exception as e:
        st.error(f"Error deleting chat history: {str(e)}")
        return False

def generate_chat_title(messages):
    """Generate a title for the chat based on messages, with fallbacks."""
    try:
        for message in messages:
            if message.get('role') == 'user' and message.get('content'):
                user_message = message['content']
                return user_message[:30] + "..." if len(user_message) > 30 else user_message
    except (KeyError, TypeError):
        pass
    return "New Chat"  # Default fallback title

def get_user_name(user_id):
    try:
        ref = firebase_db.reference(f'users/{user_id}')
        user_data = ref.get()
        return user_data.get('name', 'User') if user_data else 'User'
    except Exception as e:
        st.error(f"Error fetching user data: {str(e)}")
        return 'User'

# Use the pyrebase auth object for login/signup
def login_signup_ui():
    inject_custom_css()
    
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #00ffff; font-size: 28px; margin-bottom: 5px;">🔐 ApnaLawyer</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 14px;">Secure Legal Assistance Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    choice = st.sidebar.radio("Select Option", ["Login", "Signup", "Forgot Password"], label_visibility="collapsed")
    
    if choice == "Signup":
        st.sidebar.markdown("### Create New Account")
        name = st.sidebar.text_input("Full Name", key="signup_name")
        email = st.sidebar.text_input("Email Address", key="signup_email")
        password = st.sidebar.text_input("Password", type="password", key="signup_password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.sidebar.button("Create Account", key="signup_button"):
            if not name:
                st.sidebar.error("Please enter your full name!")
            elif not email:
                st.sidebar.error("Email address is required!")
            elif not password or not confirm_password:
                st.sidebar.error("Password fields cannot be empty!")
            elif password != confirm_password:
                st.sidebar.error("Passwords do not match!")
            else:
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    if store_user_data(user['localId'], name, email):
                        auth.send_email_verification(user['idToken'])
                        st.sidebar.success("✅ Account created! Please verify your email before logging in.")
                except Exception as e:
                    error_str = str(e)
                    if "EMAIL_EXISTS" in error_str:
                        st.sidebar.warning("⚠ Email already exists. Please try a different email address.")
                    elif "WEAK_PASSWORD" in error_str:
                        st.sidebar.warning("⚠ Password should be at least 6 characters long.")
                    else:
                        st.sidebar.error(f"Error: {error_str}")
    
    elif choice == "Login":
        st.sidebar.markdown("### Welcome Back")
        email = st.sidebar.text_input("Email Address", key="login_email")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        
        if st.sidebar.button("Login", key="login_button"):
            if not email:
                st.sidebar.error("✋ Please enter your email address")
            elif not password:
                st.sidebar.error("✋ Please enter your password")
            else:
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    user_info = auth.get_account_info(user['idToken'])
                    email_verified = user_info['users'][0]['emailVerified']
                    
                    if email_verified:
                        user_name = get_user_name(user['localId'])
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.user_token = user['idToken']
                        st.session_state.user_name = user_name
                        st.sidebar.success(f"🎉 Welcome back, {user_name}!")
                        st.rerun()
                    else:
                        st.sidebar.warning("📧 Email not verified. Please check your inbox.")
                        if st.sidebar.button("🔁 Resend Verification Email", key="resend_verification"):
                            auth.send_email_verification(user['idToken'])
                            st.sidebar.info("📬 Verification email sent again!")
                except Exception as e:
                    error_str = str(e)
                    if "EMAIL_NOT_FOUND" in error_str or "no user record" in error_str.lower():
                        st.sidebar.error("📭 Account not found")
                        st.sidebar.warning("Don't have an account? Please sign up.")
                    elif "INVALID_PASSWORD" in error_str or "INVALID_LOGIN_CREDENTIALS" in error_str:
                        st.sidebar.error("🔐 Incorrect password")
                    else:
                        st.sidebar.error("⚠ Login error. Please try again.")
    
    elif choice == "Forgot Password":
        st.sidebar.markdown("### Reset Your Password")
        email = st.sidebar.text_input("Enter your email address", key="reset_email")
        
        if st.sidebar.button("Send Reset Link", key="reset_button"):
            if not email:
                st.sidebar.error("Please enter your email address!")
            else:
                try:
                    auth.send_password_reset_email(email)
                    st.sidebar.success("📬 Password reset link sent to your email.")
                except Exception as e:
                    error_str = str(e)
                    if "EMAIL_NOT_FOUND" in error_str:
                        st.sidebar.error("❌ No account found with this email address")
                        st.sidebar.warning("⚠ Don't have an account? Please sign up.")
                    else:
                        st.sidebar.error(f"Error: {error_str}")
    
    # In your login_signup_ui() function, replace the Google login section with:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Or continue with")
    
    if st.sidebar.button("Continue with Google", key="google_login"):
        try:
            # Generate a unique state token
            state_token = hashlib.sha256(str(time.time()).encode()).hexdigest()
            
            # Get Firebase config values
            client_id = "546645596018-nvtkegm7mi8e83upfv771tv6t58c7snn.apps.googleusercontent.com"  # Use actual OAuth client ID
            firebase_auth_domain = firebaseConfig["authDomain"]
            redirect_uri = f"https://{firebase_auth_domain}/auth/handler"
            
            # Build the Google OAuth URL
            auth_url = (
                f"https://accounts.google.com/o/oauth2/v2/auth?"
                f"response_type=code&"
                f"client_id={client_id}&"
                f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
                f"scope=email%20profile%20openid&"
                f"state={state_token}"
            )
            
            # Store state token in session
            st.session_state.oauth_state = state_token
            
            # Open OAuth flow in new tab
            components.html(
                f"""
                <script>
                    window.open('{auth_url}', '_blank').focus();
                </script>
                """,
                height=0
            )
            
            st.sidebar.info("Google login window should open. Please allow popups if it doesn't appear.")

        except Exception as e:
            st.sidebar.error(f"Failed to start Google login: {str(e)}")

    def check_google_callback():
        try:
            if 'code' in st.query_params and 'state' in st.query_params:
                # Verify state token matches
                if st.query_params['state'] != st.session_state.get('oauth_state'):
                    st.error("Security verification failed. Please try logging in again.")
                    return
                
                auth_code = st.query_params['code']
                
                # Initialize the Google Auth Provider
                provider = firebase.auth.GoogleAuthProvider()
                
                # Sign in with the auth code
                credential = provider.credential(
                    None,  # No ID token needed for code flow
                    auth_code
                )
                
                # Sign in with credential
                user = auth.sign_in_with_credential(credential)
                
                # Store user in session
                st.session_state.logged_in = True
                st.session_state.user_email = user['email']
                st.session_state.user_name = user.get('displayName', 'Google User')
                st.session_state.user_token = user['idToken']
                
                # Store user data if new
                if not user_exists(user['localId']):
                    store_user_data(
                        user['localId'],
                        st.session_state.user_name,
                        user['email']
                    )
                
                # Clear the OAuth code from URL
                st.query_params.clear()
                st.rerun()
                
        except Exception as e:
            st.error(f"Google login failed: {str(e)}")

# Initialize speech-to-text model (cached to avoid reloading)
# @st.cache_resource
# def load_speech_to_text_model():
#     return pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Function to translate text
def transcribe_audio(audio_bytes, auto_detect=False):
    """Transcribe audio with auto language detection (English/Hindi)"""
    if not SPEECHMATICS_API_KEY:
        st.error("API key not configured!")
        return None

    try:
        API_BASE_URL = "https://asr.api.speechmatics.com/v2"
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB free tier limit
        TIMEOUT = 30  # seconds

        # 1. Validate audio size
        if len(audio_bytes) > MAX_FILE_SIZE:
            st.error(f"Audio exceeds {MAX_FILE_SIZE/1024/1024}MB free tier limit")
            return None

        # 2. Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            tmp_path = tmpfile.name

        # 3. Configure language settings
        language_config = {
            "language": "auto" if auto_detect else st.session_state.get('language', 'en')
        }
        
        # For auto-detect, specify possible languages (improves accuracy)
        if auto_detect:
            language_config["language_options"] = ["en", "hi"]  # English/Hindi only

        job_config = {
            "type": "transcription",
            "transcription_config": {
                **language_config,
                "operating_point": "standard",
                "enable_entities": False
            }
        }

        headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}

        # 4. Create job
        with open(tmp_path, 'rb') as audio_file:
            response = requests.post(
                f"{API_BASE_URL}/jobs",
                headers=headers,
                files={
                    'config': (None, json.dumps(job_config)),
                    'data_file': ('audio.wav', audio_file)
                },
                timeout=TIMEOUT
            )

        # 5. Handle response
        if response.status_code != 201:
            error_msg = response.json().get('error', {}).get('message', response.text)
            st.error(f"Job creation failed: {error_msg}")
            return None

        job_id = response.json()['id']
        st.session_state.current_job_id = job_id

        # 6. Poll for completion
        start_time = time.time()
        detected_language = None
        
        while True:
            if time.time() - start_time > TIMEOUT:
                raise Exception("Timeout waiting for transcription")

            status_response = requests.get(
                f"{API_BASE_URL}/jobs/{job_id}",
                headers=headers,
                timeout=TIMEOUT
            )
            status_data = status_response.json()
            
            # Capture detected language if auto mode
            if auto_detect and not detected_language:
                detected_language = status_data['job'].get('detected_language')
                if detected_language:
                    st.info(f"🔍 Detected language: {detected_language.upper()}")

            if status_data['job']['status'] == 'done':
                break
            elif status_data['job']['status'] == 'failed':
                raise Exception(f"Transcription failed: {status_data.get('error')}")
            
            time.sleep(2)

        # 7. Get transcript
        transcript_response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}/transcript",
            headers=headers,
            params={'format': 'txt'},
            timeout=TIMEOUT
        )

        if transcript_response.status_code != 200:
            st.error(f"Failed to fetch transcript: {transcript_response.text}")
            return None

        return transcript_response.text

    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def check_rate_limit():
    """Simple rate limiting for audio transcription"""
    if 'last_transcription_time' not in st.session_state:
        st.session_state.last_transcription_time = time.time()
        return True
    
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_transcription_time
    
    if time_since_last < 10:  # 10 second cooldown between transcriptions
        return False
    
    st.session_state.last_transcription_time = current_time
    return True


def create_new_chat():
    """Properly reset the chat state and start a new chat session."""
    # Save current chat if it has messages
    if len(st.session_state.get('messages', [])) > 1:  # More than just welcome message
        save_current_chat()
    
    # Reset conversation state
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"🎉✨ Welcome {st.session_state.user_name}! ✨🎉\n\nI'm ApnaLawyer, your AI legal assistant. "
                   "I can help explain Indian laws in simple terms. What would you like to know?"
    }]
    st.session_state.current_chat_id = None
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    # Clear any audio processing flags
    if 'audio_processed' in st.session_state:
        del st.session_state.audio_processed
    st.rerun()

def save_current_chat():
    """Save the current chat to history before starting a new one."""
    if len(st.session_state.get('messages', [])) > 1:  # More than just welcome message
        chat_title = generate_chat_title(st.session_state.messages)
        if hasattr(st.session_state, 'current_chat_id') and st.session_state.current_chat_id:
            update_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
        else:
            chat_id = save_chat_to_history(st.session_state.user_id, chat_title, st.session_state.messages)
            st.session_state.current_chat_id = chat_id
        st.session_state.chat_history = get_chat_history(st.session_state.user_id)


def load_chat(chat_id):
    """Load a specific chat from history."""
    # Save current chat if it has messages
    if len(st.session_state.get('messages', [])) > 1:  # More than just welcome message
        save_current_chat()
    
    # Load the selected chat
    chat_data = next((chat for chat in st.session_state.chat_history if chat[0] == chat_id), None)
    if chat_data:
        st.session_state.messages = chat_data[1]['messages']
        st.session_state.current_chat_id = chat_id
        # Reinitialize memory with loaded messages
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
        for msg in chat_data[1]['messages'][:-2]:  # Skip last 2 messages to maintain window size
            if msg['role'] == 'user':
                st.session_state.memory.save_context({"question": msg['content']}, {"answer": ""})
        st.rerun()

def delete_chat(chat_id):
    """Delete a specific chat from the user's chat history."""
    try:
        user_id = st.session_state.user_id
        if delete_chat_history(user_id, chat_id):
            # Show toast notification instead of message in chat history
            st.toast("Chat deleted successfully!", icon="✅")
            # Refresh chat history after deletion
            st.session_state.chat_history = get_chat_history(user_id)
            st.rerun()  # Force UI refresh
        else:
            st.toast("Failed to delete chat.", icon="❌")
    except Exception as e:
        st.toast(f"Error deleting chat: {str(e)}", icon="❌")

def chatbot_ui():
    # Initialize language state
    target_lang_code = "en"  # Default language is English
    if "language" not in st.session_state:
        st.session_state.language = target_lang_code # Default language is English

    # Initialize with personalized welcome message if first time
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"🎉✨ Welcome {st.session_state.user_name}! ✨🎉\n\nI'm ApnaLawyer, your AI legal assistant. "
                      "I can help explain Indian laws in simple terms. What would you like to know?"
        }]

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    # Get user ID
    if "user_id" not in st.session_state:
        try:
            user_info = auth.get_account_info(st.session_state.user_token)
            st.session_state.user_id = user_info['users'][0]['localId']
            st.session_state.chat_history = get_chat_history(st.session_state.user_id)
        except Exception as e:
            st.error(f"Error getting user info: {str(e)}")

    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

    embeddings = load_embeddings()
    db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the prompt template
    prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
<s>[INST]
You are ApnaLawyer, a trusted and knowledgeable AI assistant for Indian citizens. You provide legally accurate help related to Indian laws — including the Indian Penal Code (IPC), CrPC, POCSO Act, Domestic Violence Act, and others.

### Your responsibilities:
- Use clear, simple, respectful language
- Accurately cite laws (IPC sections, CrPC, Acts) **only when asked for legal explanation**
- If the user asks you to "write", "draft", "create", or "format" a legal document or application, you must write a **formal legal draft**
- Do not mix legal explanations with the draft unless asked — keep your response focused
- You may write drafts such as:
  - FIR applications
  - Police complaints
  - Legal notices
  - Affidavits
  - Consent forms
- When drafting, use correct legal formatting, salutation, subject lines, and placeholders (name, address, date)

### CONTEXT:
{context}

### CHAT HISTORY:
{chat_history}

### QUESTION:
{question}

---

Based on the user's intent, choose **one of the following** response types:

---
📘 If the user is asking about the law, respond with:

✅ **Answer**:  
[Summary of the situation and legal explanation]

📘 **Relevant Law(s)**:  
[List exact IPC sections, Acts, clause-wise punishment, and applicable exceptions]

🧾 **Other Related Laws**:  
[Include CrPC, POCSO, DV Act, or procedural laws if relevant]

📝 **Suggested Action**:  
[Practical next steps — where to file, what to prepare]

🧾 **Summary**:  
[Short recap in plain language]

---


✍️ If the user wants you to write or draft something, respond ONLY with:

📄 **Legal Draft/Application**:
[Write the complete legal document in clean, formal format using Indian legal norms. Use placeholders where needed.]

</s>[/INST]
"""
)

    # Initialize the ConversationalRetrievalChain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt_template}
    )

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Text input box - always shown at the bottom
    text_input = st.chat_input("Ask your legal question or record audio...")

    # Handle text input
    if text_input and 'audio_processed' not in st.session_state:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(text_input)

        # Add to message history
        st.session_state.messages.append({"role": "user", "content": text_input})

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Auto-detect language from user prompt
                # Auto-detect language from prompt
                target_lang_name = detect_target_language(text_input)
                if target_lang_name == "unsupported":
                    st.warning("⚠ Currently only Hindi, English, and Hinglish are supported.")
                    return

                target_lang_code = supported_languages.get(target_lang_name, "en")

                # Translate if necessary
               
                mod_input = f"""Please answer the following question in {target_lang_name} language, using clear legal terms:
                {text_input}""" 
                result = qa.invoke(input=mod_input)
                answer = result["answer"]


                message_placeholder = st.empty()
                full_response = "⚠ Gentle reminder: We generally ensure precise information, but do double-check. \n\n\n"
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.006)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update chat history
        update_chat_history_function()

    # Audio input - shown below the text input
    audio_file = st.audio_input("", key="audio_recorder")

    # Handle audio input (only if no text input was processed in this cycle)
    if audio_file and 'audio_processed' not in st.session_state and not text_input:
        audio_bytes = audio_file.getvalue()

        # Set flag to prevent duplicate processing
        st.session_state.audio_processed = True

        with st.spinner("Transcribing..."):
            try:
                transcribed_text = transcribe_audio(audio_bytes)

                # Display transcribed text immediately
                with st.chat_message("user"):
                    st.markdown(transcribed_text)

                st.session_state.messages.append({"role": "user", "content": transcribed_text})

                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        

                        # Detect desired language from user input
                        # Detect desired language from user input
                        target_lang_name = detect_target_language(transcribed_text)
                        if target_lang_name == "unsupported":
                            st.warning("⚠ Currently only Hindi, English, and Hinglish are supported.")
                            return

                        target_lang_code = supported_languages.get(target_lang_name, "en")

                        # Handle Hinglish separately (keep original)
                        
                        mod_input = f"""Please answer the following question in {target_lang_name} language, using clear legal terms:
                        {transcribed_text}"""
                        result = qa.invoke(input=mod_input)
                        answer = result["answer"]


                        # else: leave in English or handle more languages later
                        message_placeholder = st.empty()
                        full_response = "⚠ Gentle reminder: We generally ensure precise information, but do double-check. \n\n\n"
                        for chunk in answer:
                            full_response += chunk
                            time.sleep(0.006)
                            message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Update chat history
                update_chat_history_function()

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clear the flag after processing
                if 'audio_processed' in st.session_state:
                    del st.session_state.audio_processed

    # Sidebar UI (unchanged from your original)
    with st.sidebar:
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #00ffff; margin-bottom: 5px;">👤 {st.session_state.user_name}</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 12px; margin-top: 0;">{st.session_state.user_email}</p>
        </div>
        """, unsafe_allow_html=True)

        # Button for creating a new chat
        if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
            create_new_chat()

        # Chat history section
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            for chat_id, chat_data in st.session_state.chat_history:
                # Safely get title with default
                chat_title = chat_data.get('title', 'Untitled Chat')
                timestamp = time.strftime('%d %b %Y, %I:%M %p', 
                    time.localtime(chat_data.get('last_updated', time.time())))

                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(
                        f"{chat_title}",
                        key=f"chat_{chat_id}",  # Unique key for each chat button
                        help=f"Last updated: {timestamp}",
                        use_container_width=True
                    ):
                        load_chat(chat_id)
                with col2:
                    if st.button("🗑", key=f"delete_{chat_id}"):  # Unique key for each delete button
                        delete_chat(chat_id)
        else:
            st.markdown("<p style='color: rgba(255,255,255,0.5);'>No chat history yet</p>", unsafe_allow_html=True)

        # Logout button
        if st.button("🚪 Logout", key="logout_button", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.user_id = None
            st.rerun()


def update_chat_history_function():
    """Helper function to update chat history"""
    if len(st.session_state.messages) > 2:
        chat_title = generate_chat_title(st.session_state.messages)
        if hasattr(st.session_state, 'current_chat_id') and st.session_state.current_chat_id:
            update_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
        else:
            chat_id = save_chat_to_history(st.session_state.user_id, chat_title, st.session_state.messages)
            st.session_state.current_chat_id = chat_id
        st.session_state.chat_history = get_chat_history(st.session_state.user_id)

# ----------------- Main App -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_signup_ui()
else:
    # Main chat interface
    chatbot_ui()
    # Footer
    footer()
