# BOB — Local AI Assistant

> **B**asic **O**llama **B**ot  
> Local-first · Free · Windows · Python · Ollama

---

## What is BOB V1?

BOB V1 is the **core foundation** of a modular, local AI assistant that runs
entirely on your Windows machine using [Ollama](https://ollama.com) — no cloud,
no paid APIs, no internet required after initial model download.

This first version implements **only the text-based CLI interface and the core
conversation engine**.  Everything is intentionally minimal so that future
capabilities can be layered on top without restructuring the project.

---

## Architecture

```
CLI / Browser input
       │
       ├── app/interfaces/cli.py   ← pure I/O, no logic
       ├── app/interfaces/web.py   ← FastAPI transport, no logic
       │
       ▼
 app/core/assistant.py   ← manages conversation history, calls LLM
       │
       ▼
 app/llm/ollama_client.py ← HTTP wrapper around Ollama /api/chat
       │
       ▼
  Ollama (local server)
       │
       ▼
   Local LLM model
```

Key design principle: **the interface contains no assistant logic**, and the
**assistant contains no LLM-specific code** (it accepts any object satisfying
the `LLMClient` protocol).

### Module map

| Path | Responsibility |
|------|---------------|
| `app/main.py` | Entry point; loads env, configures logging, starts CLI |
| `app/core/assistant.py` | Conversation history, message routing |
| `app/llm/ollama_client.py` | Ollama HTTP API abstraction |
| `app/interfaces/cli.py` | Read-eval-print loop, user I/O |
| `app/interfaces/web.py` | Local FastAPI transport for the browser UI |
| `web/` | Static HTML/CSS/JS chat page and bundled vendor assets |
| `app/tools/registry.py` | Tool registry scaffold (empty in V1) |

---

## Requirements

- **Python 3.11+**
- **Ollama** installed and running (`ollama serve`)
- At least one Ollama model pulled locally

---

## Installation

### 1. Clone / open the project

```powershell
git clone https://github.com/<your-username>/bob-personal-ai-assistant.git
cd bob-personal-ai-assistant
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install runtime dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install development/test dependencies (optional)

```powershell
pip install -r requirements-dev.txt
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```powershell
Copy-Item .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Model tag to use |
| `OLLAMA_TIMEOUT` | `120` | HTTP timeout in seconds |
| `JARVIS_SYSTEM_PROMPT` | _(built-in)_ | Override the system prompt |
| `JARVIS_LOG_LEVEL` | `WARNING` | Logging verbosity |
| `WHISPER_MODEL` | `base` | Whisper model (`tiny`, `base`, `small`, etc.) |
| `WHISPER_DEVICE` | `auto` | Device (`auto`, `cuda`, `cpu`) |
| `WHISPER_COMPUTE_TYPE` | `auto` | Compute precision (`float16`, `int8`, etc.) |
| `WHISPER_LANGUAGE` | _(auto)_ | Language code (`en`, `es`, etc.) |
| `AUDIO_SILENCE_THRESHOLD` | `0.015` | RMS energy threshold for speech activity |
| `AUDIO_SILENCE_DURATION` | `1.2` | Seconds of silence after speech to finish |
| `PIPER_VOICE` | `en_US-lessac-medium` | Piper neural voice model name |
| `PIPER_VOICES_DIR` | `models/piper` | Directory for Piper voice ONNX files |
| `PIPER_USE_CUDA` | `false` | Run Piper on CUDA (true/false) |
| `PIPER_LENGTH_SCALE` | `1.0` | Speaking speed rate (1.0 = normal) |
| `JARVIS_VOICE_REPLY` | `false` | Automatically speak replies for typed text |

---

## Voice Input (Speech-to-Text)

BOB includes local, offline speech-to-text using **faster-whisper** (CTranslate2):
- **100% offline**: All transcription runs on your machine.
- **Hardware-adaptive**: Probes CUDA acceleration first; automatically falls back to CPU (`int8`) if CUDA runtime DLLs are unavailable.
- **Energy-based VAD**: Detects when you start and stop speaking automatically.

### 1. Manual STT Verification (Isolated, No LLM)

To test your microphone and speech-to-text directly without invoking Ollama:

```powershell
python -m app.voice.manual_test
```

This verifies the chain:
`Microphone → AudioRecorder → WhisperSTT → Transcribed Text`

---

## Voice Output (Text-to-Speech)

BOB includes local, offline neural text-to-speech using **Piper** (ONNX Runtime) and **sounddevice**:
- **100% offline**: Neural speech synthesis runs locally in ~100–300ms without any cloud dependencies.
- **Default Voice**: `en_US-lessac-medium` (~60MB), delivering clear, natural English speech.
- **Markdown Normalization**: Automatically strips code fences, markdown links, headers, and bullet formatting before synthesis so responses sound fluent.
- **Direct Output**: Audio is played through your default Windows audio output device (speakers/headphones).

### 1. Manual TTS Verification (Isolated, No LLM)

To test text-to-speech directly and hear BOB speak:

```powershell
# Interactive loop:
python -m app.voice.manual_tts_test

# Or speak a one-line test sentence:
python -m app.voice.manual_tts_test --text "Hello, I am Jarvis. Local speech synthesis is working perfectly."
```

This verifies the chain:
`Text → PiperTTS → sounddevice → Speakers`

---

## Full Voice Loop in CLI

Start BOB:

```powershell
python -m app.main
```

When prompted, type `voice` (or `/voice`) to record an utterance. BOB runs the complete voice loop:
```
Microphone → faster-whisper → JarvisAssistant → response text → Piper → Speakers
```

### In-session commands

| Command | Action |
|---------|--------|
| `voice` / `/voice` / `:v` | Speak with microphone (BOB speaks reply aloud) |
| `/voice-reply` | Toggle speaking replies for typed messages as well |
| `/tts <text>` | Speak custom text aloud immediately |
| `reset` / `clear` | Clear conversation history and start fresh |
| `exit` / `quit` / `q` | End the session |
| `Ctrl+C` | Graceful exit |

---

## Web Interface

BOB also has a local browser frontend. It is a thin transport layer: all
conversation logic stays in `app/core/assistant.py`.

Start it:

```powershell
python -m app.interfaces.web
```

Then open **http://localhost:8000** (or `http://127.0.0.1:8000`).

The server binds to `127.0.0.1` only. It is not exposed on your LAN and has no
authentication. Do not change this binding unless you understand the risk:
BOB can launch applications and control parts of this laptop.

Set `BOB_WEB_PORT` to use a different port:

```powershell
$env:BOB_WEB_PORT = "8080"
python -m app.interfaces.web
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serves the chat page |
| `POST` | `/api/chat` | Send `{"message": "..."}`; returns `{"reply": "...", "tools_used": [...]}` |
| `POST` | `/api/reset` | Clears conversation history (same as CLI `reset`) |
| `GET` | `/api/health` | Returns status plus active provider/model name |

POST requests must use `Content-Type: application/json`. Requests whose `Host`
or `Origin` header is not local to the configured port are rejected with a
JSON `403`.

### Behaviour notes

- Replies arrive as one complete JSON response. BOB's LLM clients do not
  support streaming yet, so there is no Server-Sent Events endpoint.
- Tool use is shown after the turn as a small "Used <tool>" badge, derived from
  conversation history without changing core logic.
- The UI assets are bundled under `web/vendor/`, so the page works offline.
- Multiple tabs share one conversation history; "New chat" resets it for all.

---

## Running the Tests

```powershell
pytest tests/ -v
```

With coverage:

```powershell
pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests use mocks — **neither Ollama, a physical microphone, nor speakers** are needed to pass the test suite.

---

## Planned Roadmap

| Capability | Status |
|------------|--------|
| Core CLI & Conversation | ✅ Implemented (Phase 1) |
| Local Tool Execution | ✅ Implemented (Phase 2: system info, app launcher, file search) |
| Voice input (STT) | ✅ Implemented (Phase 3: faster-whisper, PortAudio) |
| Voice output (TTS) | ✅ Implemented (Phase 4: Piper neural TTS, sounddevice) |
| Local web chat UI | ✅ Implemented (FastAPI + static frontend) |
| Wake-word detection | ❌ Not yet |
| HUD / Cockpit GUI | ❌ Not yet |
| Browser automation | ❌ Not yet |
| Autonomous computer control | ❌ Not yet |
| Long-term memory / vector store | ❌ Not yet |
| Laya (companion agent) | ❌ Not yet |

---

## Project Structure

```
BOB-personal-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── assistant.py         # JarvisAssistant & agentic loop
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_client.py     # OllamaClient
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tool.py              # Tool abstraction & ToolResult
│   │   ├── registry.py          # ToolRegistry
│   │   ├── system_info.py       # get_system_info
│   │   ├── open_application.py  # open_application
│   │   └── search_files.py      # search_files
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── base.py              # SpeechToText & TextToSpeech protocols
│   │   ├── audio_recorder.py    # Microphone recording with VAD
│   │   ├── stt.py               # WhisperSTT (faster-whisper)
│   │   ├── tts.py               # PiperTTS (Piper neural TTS)
│   │   ├── manual_test.py       # Isolated microphone test script
│   │   └── manual_tts_test.py   # Isolated TTS test script
│   └── interfaces/
│       ├── __init__.py
│       ├── cli.py               # CLISession (text & bidirectional voice)
│       └── web.py               # FastAPI transport for the browser UI
├── web/
│   ├── index.html               # Chat page
│   ├── styles.css               # Light/dark responsive styles
│   ├── app.js                   # Chat UI behaviour (no assistant logic)
│   └── vendor/                  # Bundled offline JS/CSS libraries
├── models/
│   └── piper/                   # Offline Piper voice models & configs
├── tests/
│   ├── test_assistant.py
│   ├── test_assistant_tools.py
│   ├── test_ollama_client.py
│   ├── test_text_tool_fallback.py
│   ├── test_tools.py
│   ├── test_voice.py
│   ├── test_tts.py
│   └── test_web.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dev.txt
```
