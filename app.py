import os
import time
from pathlib import Path
import streamlit as st
import sys
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# --- Third-party imports ---
import simpleaudio as sa
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.prebuilt import create_react_agent
import whisper
from TTS.api import TTS
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from langchain_groq import ChatGroq

# ========== ENVIRONMENT & MODEL SETUP ========== #
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Cache heavy models ---
@st.cache_resource(show_spinner=False)
def get_whisper_model():
    return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def get_tts_model():
    return TTS("tts_models/en/ljspeech/vits")

@st.cache_resource(show_spinner=False)
def get_ollama_model():
    return ChatOllama(model="mistral:7b-instruct-q4_K_M")

# ========== MODEL SELECTION UI ========== #
model_choice = st.selectbox(
    "Select LLM backend:",
    options=["Ollama (local)"]
)

# =====================
# Device & Channel Configuration (Fixed)
# =====================
DEFAULT_DEVICE_INDEX = 1  # Microphone Array (Intel® Smart ...)
DEFAULT_CHANNELS = 2      # Stereo

# ========== BACKEND FUNCTIONS ========== #
def record_audio(file_path="input.wav", duration=6, fs=16000, device=1, channels=2):
    try:
        print("Available devices:", sd.query_devices())
        print("Default input device:", sd.default.device)
        print(f"Using device index: {device}, channels: {channels}")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16', device=device)
        sd.wait()
        print("Audio shape:", audio.shape, "Max values per channel:", audio.max(axis=0))
        # Auto-amplify if signal is very weak
        max_val = np.abs(audio).max()
        if max_val < 1000 and max_val > 0:
            st.info("Audio signal is very weak. Auto-amplifying before saving.")
            audio = (audio.astype(np.float32) * (1000.0 / max_val)).clip(-32768, 32767).astype('int16')
            print("Amplified audio max:", audio.max(axis=0))
        write(file_path, fs, audio)
        return file_path
    except Exception as e:
        st.error(f"Audio recording failed: {e}")
        return None

def transcribe(file_path):
    try:
        st.write(f"Transcribe called with file_path: {file_path}")
        if not os.path.exists(file_path):
            st.error(f"Audio file does not exist: {file_path}")
            print(f"Audio file does not exist: {file_path}")
            return ""
        model = get_whisper_model()
        result = model.transcribe(file_path)
        st.write("Transcription result:", result)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        print("Transcription error:", e)
        return ""

def ask_agent(user_message):
    try:
        ollama = get_ollama_model()
        history = st.session_state.get('history', [])
        prompt = ""
        for user, bot in history:
            prompt += f"User: {user}\nAssistant: {bot}\n"
        prompt += f"User: {user_message}\nAssistant:"
        st.write("Prompt to Ollama:", prompt)
        response = ollama.invoke(prompt)
        st.write("Ollama response:", response)
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        st.error(f"Agent error: {e}")
        print("Agent error:", e)
        return ""

def speak(text, file="reply.wav"):
    try:
        tts = get_tts_model()
        tts.tts_to_file(text=text, file_path=file)
        return file
    except Exception as e:
        st.error(f"TTS failed: {e}")
        return None

def play_audio(file="reply.wav"):
    try:
        wave_obj = sa.WaveObject.from_wave_file(file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        st.error(f"Audio playback failed: {e}")

# ========== SESSION STATE ========== #
def reset_state():
    for key in ["history", "last_transcript", "last_response", "audio_ready"]:
        if key in st.session_state:
            del st.session_state[key]

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_transcript' not in st.session_state:
    st.session_state['last_transcript'] = ''
if 'last_response' not in st.session_state:
    st.session_state['last_response'] = ''
if 'audio_ready' not in st.session_state:
    st.session_state['audio_ready'] = False

# ========== PAGE CONFIG & CSS ========== #
st.set_page_config(
    page_title="🎤 VoiceVerse AI – Talk to Your LLM",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #18181b 0%, #23272f 100%) !important;
        color: #f3f4f6 !important;
    }
    .main {
        background: transparent !important;
    }
    .chat-bubble {
        border-radius: 1.5em;
        padding: 1em 1.5em;
        margin-bottom: 0.5em;
        max-width: 80%;
        word-break: break-word;
        font-size: 1.1em;
        box-shadow: 0 2px 12px #0002;
        animation: fadeIn 0.5s;
        display: flex;
        align-items: center;
        gap: 0.7em;
    }
    .bubble-user {
        background: linear-gradient(90deg, #6366f1 0%, #1e293b 100%);
        color: #fff;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 0.3em;
    }
    .bubble-bot {
        background: linear-gradient(90deg, #22d3ee 0%, #0f172a 100%);
        color: #fff;
        margin-right: auto;
        margin-left: 0;
        border-bottom-left-radius: 0.3em;
    }
    .chat-history {
        background: #23272f;
        border-radius: 1em;
        padding: 1em;
        max-height: 350px;
        overflow-y: auto;
        margin-bottom: 1em;
        box-shadow: 0 2px 12px #0002;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #22d3ee 100%);
        color: #fff;
        border: none;
        border-radius: 2em;
        padding: 0.9em 2.2em;
        font-size: 1.3em;
        font-weight: bold;
        transition: transform 0.1s, box-shadow 0.1s;
        box-shadow: 0 2px 8px #0003;
        margin-bottom: 0.5em;
    }
    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 4px 16px #22d3ee55;
        background: linear-gradient(90deg, #22d3ee 0%, #6366f1 100%);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    .recording-anim {
        display: flex;
        align-items: center;
        gap: 0.7em;
        margin: 1em 0;
    }
    .mic-wave {
        width: 2.5em;
        height: 2.5em;
        border-radius: 50%;
        background: #22d3ee44;
        position: relative;
        animation: pulse 1.2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 #22d3ee55; }
        70% { box-shadow: 0 0 0 1.2em #22d3ee00; }
        100% { box-shadow: 0 0 0 0 #22d3ee00; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 600px) {
        .chat-bubble { font-size: 1em; padding: 0.7em 1em; }
        .chat-history { max-height: 200px; }
        .stButton>button { font-size: 1em; padding: 0.7em 1.2em; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== TITLE ========== #
st.markdown("""
<div style='text-align:center; margin-bottom:0.5em;'>
    <img src='https://img.icons8.com/fluency/96/000000/microphone.png' width='64' style='margin-bottom:0.2em;'/>
    <h1 style='font-size:2.3em; margin-bottom:0.1em;'>VoiceVerse AI</h1>
    <p style='color:#a3e635; font-size:1.1em; margin-bottom:0.2em;'>Talk to your LLM – powered by Ollama</p>
</div>
""", unsafe_allow_html=True)

# ========== CHAT HISTORY ========== #
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state['history']:
    st.markdown(f'<div class="chat-bubble bubble-user">🧑‍💻 {user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble bubble-bot">🤖 {bot_msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ========== RECORDING ANIMATION ========== #
def show_recording_anim():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            '<div class="recording-anim">'
            '<div class="mic-wave"></div>'
            f'<span style="font-size:1.2em; color:#22d3ee;">Recording... Speak now!</span>'
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(f'<div style="font-size:2em; color:#a3e635; text-align:right;">⏳ {st.session_state["recording_timer"]}s</div>', unsafe_allow_html=True)

# ========== MAIN BUTTONS ========== #
col1, col2, col3 = st.columns([1,2,1])
with col2:
    start = st.button("🎙️ Start Talking", key="start_talking", use_container_width=True)
    clear = st.button("🧹 Clear Chat", key="clear_chat", use_container_width=True)

if clear:
    reset_state()
    st.rerun()

# ========== MAIN FLOW (MINIMAL, RELIABLE RECORDING) ========== #
if start:
    fs = 16000
    duration = 6
    device = DEFAULT_DEVICE_INDEX
    channels = DEFAULT_CHANNELS
    st.info(f"Recording {duration} seconds from device {device} with {channels} channels...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16', device=device)
    sd.wait()
    write("input.wav", fs, audio)
    st.success("Recording complete!")
    file_path = "input.wav"
    transcript = ""
    with st.spinner("Transcribing your speech..."):
        transcript = transcribe(file_path)
        st.session_state['last_transcript'] = transcript
    st.success("Transcription done!", icon="📝")
    response = ""
    with st.spinner("Getting LLM response..."):
        if transcript:
            response = ask_agent(transcript)
        st.session_state['last_response'] = response
    st.success("Agent replied!", icon="🤖")
    audio_file = None
    with st.spinner("Synthesizing voice reply..."):
        if response:
            audio_file = speak(response, file="reply.wav")
    st.success("Voice ready!", icon="🔊")
    st.session_state['audio_ready'] = bool(audio_file)
    if audio_file:
        play_audio(audio_file)
    if transcript and response:
        st.session_state['history'].append((transcript, response))
    else:
        st.info("No response or transcript to add to history.")
    st.rerun()

# ========== SHOW LAST EXCHANGE ========== #
if st.session_state['last_transcript']:
    st.markdown(f'<div class="chat-bubble bubble-user">🧑‍💻 {st.session_state["last_transcript"]}</div>', unsafe_allow_html=True)
if st.session_state['last_response']:
    st.markdown(f'<div class="chat-bubble bubble-bot">🤖 {st.session_state["last_response"]}</div>', unsafe_allow_html=True)

# ========== AUDIO PLAYER ========== #
if st.session_state['audio_ready']:
    audio_file = Path("reply.wav")
    if audio_file.exists():
        st.audio(str(audio_file), format="audio/wav")
    else:
        st.info("Audio will appear here after TTS.")

# ========== FOOTER ========== #
st.markdown("""
<div style='text-align:center; color:#64748b; margin-top:2em; font-size:0.95em;'>
    voiceVerse · <b>VoiceVerse AI</b>
</div>
""", unsafe_allow_html=True)

# Suggest trying a different browser if audio issues persist
st.info("If you continue to have audio issues, try running Streamlit in a different browser (e.g., Chrome, Edge, Firefox). Some browsers handle microphone permissions and audio APIs differently.")
