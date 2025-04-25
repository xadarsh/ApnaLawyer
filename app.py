import time
import os
import io
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from footer import footer
#from firebase_config import firebaseConfig
import pyrebase
# Import firebase_admin and credentials
import firebase_admin
from firebase_admin import credentials, initialize_app, db as firebase_db

# Add at the top of your file
import streamlit.components.v1 as components
import hashlib
import urllib.parse
from deep_translator import GoogleTranslator
from langdetect import detect

from transformers import pipeline
import soundfile as sf
import numpy as np
from io import BytesIO
import tempfile


import json


# # Speech Recognition Imports (add these at top of file)
# import tempfile
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
st.set_page_config(page_title="ApnaLawyer", layout="centered")
# Function to translate text
def translate_text(text, target_language):
    try:
        # Use GoogleTranslator from deep-translator
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated_text
    except Exception as e:
        return f"⚠ Translation failed: {str(e)}"

# Initialize the translator
#translator = Translator()

def initialize_firebase():
    try:
        # Check if Firebase Admin SDK is not initialized yet
        if not firebase_admin._apps:
            # Load JSON from environment variable (it contains the entire Firebase service account JSON)
            firebase_json = os.environ['FIREBASE_CREDS_JSON']
            
            # Convert the JSON string into a Python dictionary
            service_account_dict = json.loads(firebase_json)
            
            # Initialize Firebase Admin SDK using the dictionary with credentials
            initialize_app(credentials.Certificate(service_account_dict), {
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL')  # Get the database URL from environment variables
            })
            #st.write("[DEBUG] Firebase Admin SDK initialized successfully.")
    except Exception as e:
        st.error(f"Firebase initialization error: {str(e)}")
        raise

# ----------------- Firebase Init -------------------
initialize_firebase()

firebaseConfig = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

# Initialize Pyrebase for authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


# ----------------- Streamlit Config -------------------


col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/banner.jpg", use_container_width=True)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

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
    for message in messages:
        if message['role'] == 'user':
            user_message = message['content']
            return user_message[:30] + "..." if len(user_message) > 30 else user_message
    return "New Chat"

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
@st.cache_resource
def load_speech_to_text_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Function to translate text
def translate_text(text, target_language):
    try:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated_text
    except Exception as e:
        return f"⚠ Translation failed: {str(e)}"

def transcribe_audio(audio_bytes):
    """Transcribe audio using Whisper API"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_bytes)
            tmp_path = tmpfile.name
        
        # Transcribe with Whisper
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=st.session_state.get('language', 'en')
            )
        
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass  # Don't fail if temp file deletion fails
        
        return transcript['text']
    
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

# def transcribe_audio(audio_bytes):
#     """Transcribe audio using Whisper API without PyDub"""
#     try:
#         # Create temp file
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
#             tmpfile.write(audio_bytes)
#             tmp_path = tmpfile.name
        
#         # Transcribe with Whisper
#         with open(tmp_path, "rb") as audio_file:
#             transcript = openai.Audio.transcribe(
#                 model="whisper-1",
#                 file=audio_file,
#                 language=st.session_state.get('language', 'en')
#             )
        
#         os.remove(tmp_path)
#         return transcript['text']
    
#     except Exception as e:
#         st.error(f"Error transcribing audio: {str(e)}")
#         return None

# Add this function with your other utility functions
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


def chatbot_ui():
    # Initialize language state
    if "language" not in st.session_state:
        st.session_state.language = "en"  # Default language is English

    # Initialize with personalized welcome message if first time
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"🎉✨ Welcome {st.session_state.user_name}! ✨🎉\n\nI'm ApnaLawyer, your AI legal assistant. "
                      "I can help explain Indian laws in simple terms. What would you like to know?"
        }]

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

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

    # Audio recording section - modified to remove duplicate input
    audio_file = st.audio_input("", key="audio_recorder")
    audio_bytes = audio_file.getvalue() if audio_file else None


    # Process audio if recorded
    if audio_bytes:
        if not os.getenv('OPENAI_API_KEY'):
            st.error("Voice input disabled - OpenAI API key not configured")
        elif not check_rate_limit():
            st.warning("Please wait a moment before sending another voice message")
        else:
            with st.spinner("Transcribing your voice..."):
                try:
                    transcribed_text = transcribe_audio(audio_bytes)
                    if transcribed_text:
                        # Auto-inject the transcription
                        st.session_state.audio_transcription = transcribed_text
                        st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")

    # Single input handling for both text and voice
    # prompt = st.chat_input("Type or speak your message...")
    # if 'audio_transcription' in st.session_state:
    #     prompt = st.session_state.audio_transcription
    #     del st.session_state.audio_transcription


    prompt_template = """
<s>[INST]
You are ApnaLawyer.bot, a friendly legal assistant specializing in Indian law. You provide clear, accurate information about the Indian Penal Code (IPC) and related laws. 
Key principles:
- Be concise but thorough
- Use simple language anyone can understand
- Always clarify when something is outside your expertise
- Structure responses logically but conversationally
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
Provide a helpful response that:
1. Directly answers the question
2. Cites relevant laws/sections when possible
3. Notes important exceptions
4. Suggests next steps if needed
</s>[INST]
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        st.error("API key for Together is missing. Please set the TOGETHER_API_KEY environment variable.")

    llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

    qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever,
                                               combine_docs_chain_kwargs={'prompt': prompt})

        # Voice input functionality
    def handle_voice_input():
        st.session_state.recording = True
        audio_file = audio_recorder()
        if audio_file:
            try:
                # Extract bytes from UploadedFile
                audio_bytes = audio_file.getvalue()  # Correctly extract bytes from UploadedFile
                
                # Convert bytes to numpy array
                audio_array, sample_rate = sf.read(BytesIO(audio_bytes))
                audio_dict = {"raw": audio_array, "sampling_rate": sample_rate}
                
                # Load model if not already loaded
                if "speech_to_text" not in st.session_state:
                    st.session_state.speech_to_text = load_speech_to_text_model()
                
                # Convert speech to text
                text = st.session_state.speech_to_text(audio_dict)["text"]
                
                if text.strip():
                    # Process the transcribed text as user input
                    process_user_input(text)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
        st.session_state.recording = False

    def extract_answer(full_response):
        try:
            answer_start = full_response.find("Response:")
            if answer_start != -1:
                answer_start += len("Response:")
                return full_response[answer_start:].strip()
            return full_response.strip()
        except Exception as e:
            return f"Error extracting answer: {str(e)}"

    def create_new_chat():
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"🆕 New chat started {st.session_state.user_name}! What legal question can I help you with?"
        }]
        st.session_state.memory.clear()
        st.session_state.current_chat_id = None
        st.rerun()

    def load_chat(chat_id):
        try:
            ref = firebase_db.reference(f'users/{st.session_state.user_id}/chats/{chat_id}')
            chat_data = ref.get()
            if chat_data:
                st.session_state.messages = chat_data['messages']
                st.session_state.current_chat_id = chat_id
                st.session_state.memory.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Error loading chat: {str(e)}")

    def delete_chat(chat_id):
        if delete_chat_history(st.session_state.user_id, chat_id):
            if st.session_state.current_chat_id == chat_id:
                create_new_chat()
            st.session_state.chat_history = get_chat_history(st.session_state.user_id)
            st.rerun()

    # Static translations dictionary
    static_translations = {
        "language_switched": {
            "en": "Language switched to",
            "hi": "भाषा बदल दी गई है",
            "fr": "Langue changée en",
            "es": "Idioma cambiado a",
            "de": "Sprache gewechselt zu",
            "zh-cn": "语言切换为",
            "ar": "تم تغيير اللغة إلى",
            "ru": "Язык переключен на",
            "ja": "言語が変更されました",
            "ko": "언어가 변경되었습니다",
            "it": "Lingua cambiata in",
            "pt": "Idioma alterado para",
            "bn": "ভাষা পরিবর্তন করা হয়েছে",
            "ta": "மொழி மாற்றப்பட்டது",
            "te": "భాష మార్చబడింది",
            "ml": "ഭാഷ മാറ്റി",
            "kn": "ಭಾಷೆಯನ್ನು ಬದಲಿಸಲಾಗಿದೆ",
            "mr": "भाषा बदलली गेली आहे",
            "gu": "ભાષા બદલાઈ ગઈ છે",
            "pa": "ਭਾਸ਼ਾ ਬਦਲ ਦਿੱਤੀ ਗਈ ਹੈ",
            "or": "ଭାଷା ପରିବର୍ତ୍ତନ କରାଯାଇଛି",
            "as": "ভাষা পৰিবৰ্তন কৰা হৈছে",
            "ne": "भाषा परिवर्तन गरिएको छ",
            "ur": "زبان تبدیل کر دی گئی ہے",
            "sd": "ٻولي تبديل ڪئي وئي آهي",
            "sa": "भाषा परिवर्तिता अस्ति",
            "kok": "भाषा बदलली",
            "mni": "লানগুয়েজ চেঞ্জ করা হইছে",
            "doi": "भाषा बदल दी गई है",
            "sat": "ᱵᱟᱹᱡᱟ ᱯᱟᱹᱨᱤᱵᱟᱹᱨᱛᱤᱱ ᱠᱟᱹᱨᱟᱹᱜ ᱠᱟᱹᱨᱟᱹᱜ",
            "brx": "भाषा बदल दी गई है",
            "mai": "भाषा बदलल गेल अछि"
        }
    }
    

    supported_languages = [
        "en",  # English
        "as",  # Assamese
        "bn",  # Bengali
        "brx", # Bodo
        "doi", # Dogri
        "gu",  # Gujarati
        "hi",  # Hindi
        "kn",  # Kannada
        "ks",  # Kashmiri
        "kok", # Konkani
        "mai", # Maithili
        "ml",  # Malayalam
        "mni", # Manipuri
        "mr",  # Marathi
        "ne",  # Nepali
        "or",  # Odia
        "pa",  # Punjabi
        "sa",  # Sanskrit
        "sat", # Santali
        "sd",  # Sindhi
        "ta",  # Tamil
        "te",  # Telugu
        "ur"   # Urdu
    ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if prompt := st.chat_input("Say something..."):
        with st.chat_message("user"):
            st.markdown(f"You: {prompt}")

        # Detect user language
        user_language = detect(prompt)
        if user_language in static_translations["language_switched"]:
            st.session_state.language = user_language

        # Check for language change command
        if prompt.startswith("/language"):
            lang_code = prompt.split(" ")[1].strip()
            if lang_code not in supported_languages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠ Unsupported language code: {lang_code}. Supported languages are: {', '.join(supported_languages)}."
                })
                st.experimental_rerun()
            st.session_state.language = lang_code
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{static_translations['language_switched'].get(lang_code, 'Language switched to')} {lang_code.upper()}."
            })
            st.experimental_rerun()

        # Process user input
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                result = qa.invoke(input=prompt)
                answer = extract_answer(result["answer"])

                # Translate the chatbot's response if necessary
                if st.session_state.language != "en":
                    answer = translate_text(answer, st.session_state.language)

                message_placeholder = st.empty()
                full_response = "⚠ Gentle reminder: We generally ensure precise information, but do double-check. \n\n\n"
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.002)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Save or update chat
        if len(st.session_state.messages) > 2:  # Only save if there's actual conversation
            chat_title = generate_chat_title(st.session_state.messages)
            if hasattr(st.session_state, 'current_chat_id') and st.session_state.current_chat_id:
                update_chat_history(st.session_state.user_id, st.session_state.current_chat_id, st.session_state.messages)
            else:
                chat_id = save_chat_to_history(st.session_state.user_id, chat_title, st.session_state.messages)
                st.session_state.current_chat_id = chat_id
            # Refresh chat history
            st.session_state.chat_history = get_chat_history(st.session_state.user_id)

    # Sidebar UI
    with st.sidebar:
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="color: #00ffff; margin-bottom: 5px;">👤 {st.session_state.user_name}</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 12px; margin-top: 0;">{st.session_state.user_email}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
        
        st.markdown("---")
        st.markdown("### Chat History")
        
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            for chat_id, chat_data in st.session_state.chat_history:
                timestamp = time.strftime('%d %b %Y, %I:%M %p', time.localtime(chat_data['last_updated']))
                
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(
                        f"{chat_data['title']}",
                        key=f"chat_{chat_id}",
                        help=f"Last updated: {timestamp}",
                        use_container_width=True
                    ):
                        load_chat(chat_id)
                with col2:
                    if st.button("🗑", key=f"delete_{chat_id}"):
                        delete_chat(chat_id)
        else:
            st.markdown("<p style='color: rgba(255,255,255,0.5);'>No chat history yet</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.user_id = None
            st.rerun()

    footer()

# ----------------- Main App -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_signup_ui()
else:
    chatbot_ui()
