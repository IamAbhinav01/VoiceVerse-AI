# 🎤 VoiceVerse AI – Talk to Your LLM, Locally and Privately

**VoiceVerse AI** is a fully local voice-based assistant that lets you **speak your query**, get a **transcribed response**, and hear the **spoken reply** — all powered by open-source tools and a sleek Streamlit interface.

> 🎧 No cloud, no surveillance — just intelligent voice interaction on your own machine.

---

## 🧠 What It Can Do

- 🎙️ **Record your voice** using your microphone
- 📝 **Transcribe speech to text** with [OpenAI Whisper](https://github.com/openai/whisper)
- 🤖 **Query an LLM agent** using [LangChain](https://github.com/langchain-ai/langchain) + [Groq](https://groq.com/) or [Ollama](https://ollama.com/)
- 🔊 **Generate natural speech output** using [Coqui TTS](https://github.com/coqui-ai/TTS)
- 💬 **Chat-style interface** with message history and animations
- ✨ Built with [Streamlit](https://streamlit.io) + custom CSS for a modern, responsive UI

---

## 🖼️ UI Preview
![Screenshot 2025-07-02 100456](https://github.com/user-attachments/assets/5337490e-f02f-49a9-a757-0e9eb9b5040d)
![Screenshot 2025-07-02 100734](https://github.com/user-attachments/assets/c666e36c-16cf-4300-bfa8-bd6ca71357ef)
![Screenshot 2025-07-02 100915](https://github.com/user-attachments/assets/80c6daca-7eed-4f64-8925-8bc98f0e5740)


> Need a video demo? Check `/demo/voiceverse_demo.mp4` or view [the LinkedIn demo](https://www.linkedin.com/in/abhinav-sunil-870184279/).

---

## 🔧 Tech Stack

| Layer              | Tools Used                                                                 |
|-------------------|------------------------------------------------------------------------------|
| Transcription      | [Whisper (base)](https://github.com/openai/whisper)                        |
| LLM Backend        | [Groq (LLaMA3)](https://groq.com/), [Ollama (Mistral)](https://ollama.com) |
| Agent Orchestration| [LangChain + LangGraph](https://github.com/langchain-ai/langgraph)         |
| Tools Integration  | DuckDuckGo Search, Yahoo Finance News via LangChain                        |
| Text-to-Speech     | [TTS Coqui VITS](https://github.com/coqui-ai/TTS)                          |
| UI Layer           | [Streamlit](https://streamlit.io/) + CSS Animations                        |

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/IamAbhinav01/voiceverse-ai.git
cd voiceverse-ai
