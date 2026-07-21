# AI Multimodal Smart Knowledge Assistant

A production-grade, multimodal knowledge platform capable of processing natural language queries, audio voice inputs, document search, web retrieval, image captioning, and AI artwork synthesis. Built with a resilient multi-provider LLM routing architecture, local vector database retrieval, real-time web search fallback, speech recognition, speech synthesis, and dual user interface options.

---

## Executive Summary

The AI Multimodal Smart Knowledge Assistant bridges local document knowledge with multi-provider generative AI capabilities. It accepts inputs via text typing or microphone voice recording, retrieves contextual domain knowledge using Retrieval-Augmented Generation (RAG), automatically falls back to live web search for real-world current events, and returns both formatted written responses and spoken audio output. In addition, the platform provides computer vision image analysis and text-to-image art generation.

---

## Capstone Feature Mapping

| Requirement / Module | Implementation Component | Primary Handler | Status | Verification Command |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1(a): LLM Routing & Fallback** | `core/llm_provider.py` | Google Gemini 3.5 Flash / Groq Llama 3.3 70B | COMPLETED | `pytest tests/test_llm_provider.py` |
| **Module 1(b): Speech-to-Text (STT)** | `modules/speech_to_text.py` | OpenAI Whisper (Local) | COMPLETED | `pytest tests/test_stt.py` |
| **Module 1(d): Text-to-Speech (TTS)** | `modules/text_to_speech.py` | gTTS Audio Synthesizer | COMPLETED | `pytest tests/test_tts.py` |
| **Module 2(a): Image Captioning** | `modules/image_captioning.py` | Gemini Vision / Groq Vision | COMPLETED | `pytest tests/test_image_captioning.py` |
| **Module 2(b): Image Generation** | `modules/image_generation.py` | Pollinations.ai API | COMPLETED | `pytest tests/test_image_generation.py` |
| **Module 3: RAG & Web Search** | `modules/rag/retriever.py` | ChromaDB + DuckDuckGo Engine | COMPLETED | `pytest tests/test_rag.py` |
| **Multimodal Orchestrator** | `core/assistant.py` | `MultimodalAssistant` Class | COMPLETED | `pytest tests/test_assistant.py` |
| **Unified Web Interface** | `ui/gradio_app.py` | FastAPI + Gradio + SketchAgents UI | COMPLETED | `python ui/gradio_app.py` |

---

## System Architecture

```text
                                 +-------------------------------+
                                 |   User Input (Text / Voice)   |
                                 +---------------+---------------+
                                                 |
                                                 v
                                 +---------------+---------------+
                                 |  Speech-to-Text (Whisper STT)  | (if audio provided)
                                 +---------------+---------------+
                                                 |
                                                 v
                                 +---------------+---------------+
                                 |  RAG Knowledge Base Search    | (ChromaDB Vector Store)
                                 +---------------+---------------+
                                                 |
                                        +--------+--------+
                                        | Context Found?  |
                                        +---+---------+---+
                                            |         |
                                     (Yes)  |         | (No / Low Score)
                                            v         v
                             +--------------+--+   +--+-------------------+
                             | Local RAG Doc   |   | Live Web Search      | (DuckDuckGo Engine)
                             | Context         |   | Snippets Context     |
                             +--------------+--+   +--+-------------------+
                                            |         |
                                            +----+----+
                                                 |
                                                 v
                                 +---------------+---------------+
                                 |   LLM Router & Fallback Chain |
                                 |   Primary: Gemini 3.5 Flash   |
                                 |   Fallback: Groq 70B          |
                                 +---------------+---------------+
                                                 |
                                                 v
                                 +---------------+---------------+
                                 |  Text-to-Speech (gTTS Engine) |
                                 +---------------+---------------+
                                                 |
                                                 v
                                 +---------------+---------------+
                                 | Formatted Response + MP3 Audio|
                                 +-------------------------------+
```

---

## Key Features

1. **Multimodal Query Processing**
   - Accept typed textual queries or voice recordings.
   - Speech-to-Text transcription powered locally by OpenAI Whisper.
   - Text-to-Speech audio response generation powered by gTTS with automatic sentence splitting.

2. **Retrieval-Augmented Generation (RAG) with Live Web Search Fallback**
   - Domain-specific vector search across 7 knowledge categories (`college`, `tourism`, `healthcare`, `agriculture`, `library`, `museums`, `monuments`).
   - Automatic query domain filtering or auto-detection.
   - Real-time web search fallback using DuckDuckGo when internal database scores are low, enabling accurate responses for current real-world news.

3. **Multi-Provider Resilient LLM Routing**
   - Primary: Google Gemini (`gemini-3.5-flash`).
   - Fallback: Groq (`llama-3.3-70b-versatile`).
   - Automatic rate-limit handling and seamless provider switching.

4. **Computer Vision & Image Analysis**
   - Upload images to generate structured captions.
   - Support for multiple description styles: `Descriptive`, `Short`, and `Detailed`.
   - Integrated **RAG Follow-up** for exploring context related to uploaded images.

5. **Text-to-Image Generation**
   - Synthesise original artwork from descriptive prompts.
   - Powered by Pollinations.ai image generation API.

6. **Unified Dual Interface Application**
   - Hand-drawn **SketchAgents Notebook UI** served as primary interface at `/` and `/ui/`.
   - Native **Gradio Blocks Interface** served at `/gradio`.
   - Single FastAPI backend server on port 7871.

---

## Directory Structure

```text
multimodal-knowledge-assistant/
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic BaseSettings config loader
├── core/
│   ├── assistant.py             # MultimodalAssistant unified orchestrator
│   ├── llm_provider.py          # Provider router (Gemini 3.5 Flash -> Groq)
│   ├── logger.py                # Loguru log configuration (console & rotating files)
│   └── schemas.py               # Pydantic schemas for data consistency
├── modules/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Document chunking & ChromaDB vector indexing
│   │   └── retriever.py         # Vector similarity search & web fallback logic
│   ├── image_captioning.py      # Vision analysis engine
│   ├── image_generation.py      # Text-to-image synthesis module
│   ├── speech_to_text.py        # Local Whisper STT module
│   ├── text_to_speech.py        # gTTS audio synthesis module
│   └── web_search.py            # Live DuckDuckGo web search engine
├── ui/
│   ├── gradio_app.py            # Unified FastAPI + Gradio server entrypoint
│   └── web_modern/              # SketchAgents Notebook Frontend
│       ├── index.html           # Notebook layout HTML structure
│       ├── style.css            # Hand-drawn design system & tokens
│       └── app.js               # Gradio JS Client integration
├── data/
│   └── knowledge_base/          # Source documents per domain category
├── tests/                       # Unit and integration test suite
├── .env.example                 # Environment variables template
├── requirements.txt             # Project dependencies
└── README.md                    # System documentation
```

---

## Prerequisites & Installation

### 1. System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- Python Version: Python 3.11 or 3.12
- Memory: 4 GB RAM minimum (8 GB recommended for Whisper model execution)

### 2. Install FFmpeg (Required for Audio Processing)
FFmpeg is required by `pydub` and `whisper` for audio format conversion.

- **Windows**:
  Download from [ffmpeg.org](https://ffmpeg.org/), extract to a directory (e.g. `C:\ffmpeg`), and add the `bin` folder to your System `PATH`.
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```

### 3. Clone Repository & Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/meetchauhan17/AI-Multimodal-Smart-Knowledge-Assistant.git
cd AI-Multimodal-Smart-Knowledge-Assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Credentials

Create a `.env` file in the project root directory based on `.env.example`:

```env
# Primary LLM Provider Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_VISION_MODEL=gemini-3.5-flash

# Fallback LLM Provider Configuration
GROQ_API_KEY=your_groq_api_key_here

# Routing Order
PRIMARY_PROVIDER=gemini
FALLBACK_ORDER=gemini,groq
```

---

## Running the Application

Launch the unified server by running:

```bash
python ui/gradio_app.py
```

Once started, access the application endpoints in your browser:

- **SketchAgents Notebook UI (Primary)**: `http://127.0.0.1:7871/` or `http://127.0.0.1:7871/ui/`
- **Gradio Blocks Interface**: `http://127.0.0.1:7871/gradio`
- **Interactive API Documentation**: `http://127.0.0.1:7871/docs`

---

## User Interface Overview

### 1. SketchAgents Notebook Interface (`/ui/`)
A hand-drawn notebook design system featuring:
- Paper dot-grid background texture with wobbly card borders and tape strips.
- **Tab 1 — Ask (Text & Voice)**: Typed questions, microphone recording, domain selection, sticky-note response display, audio playback, and provider metadata.
- **Tab 2 — Image Captioning**: Drag-and-drop file upload, style selectors, caption text output, latency metrics, and RAG follow-up analysis.
- **Tab 3 — Image Generation**: Prompt input field, polaroid-style image output frame, and provider status.

### 2. Native Gradio Interface (`/gradio`)
A standard Gradio Blocks interface providing direct form-based access to all model endpoints and parameters.

---

## API Reference

The application exposes standard REST endpoints mounted via FastAPI and Gradio Client JS bindings:

| Endpoint Name | HTTP Route | Description | Input Parameters | Output Format |
| :--- | :--- | :--- | :--- | :--- |
| `handle_ask` | `/gradio/api/handle_ask` | Multimodal Voice & Text Q&A | `question`, `audio_path`, `domain` | `[question, answer, audio_url, sources, provider]` |
| `handle_caption` | `/gradio/api/handle_caption` | Image Analysis & Captioning | `image_file`, `style` | `[caption, provider, latency, error]` |
| `handle_caption_followup` | `/gradio/api/handle_caption_followup` | RAG Query from Image Caption | `caption` | `[answer, sources]` |
| `handle_image_gen` | `/gradio/api/handle_image_gen` | Text-to-Image Generation | `prompt` | `[image_path, provider, error]` |

---

## Automated Testing & Verification

Run the full automated test suite using `pytest`:

```bash
# Run all unit and integration tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_llm_provider.py -v
python -m pytest tests/test_rag.py -v
python -m pytest tests/test_stt.py -v
python -m pytest tests/test_tts.py -v
python -m pytest tests/test_image_captioning.py -v
python -m pytest tests/test_image_generation.py -v
python -m pytest tests/test_assistant.py -v
```

---

## Troubleshooting & Known Behaviours

1. **Port Conflicts**:
   If port `7871` is already in use, stop existing processes or update `server_port` in `ui/gradio_app.py`.

2. **ChromaDB File Lock**:
   Running `pytest` while `ui/gradio_app.py` is actively running may trigger ChromaDB SQLite database locks. Exit the running server process before running automated tests.

3. **Audio Permissions**:
   When using the browser microphone recording feature, ensure browser permissions allow audio input access on `http://127.0.0.1:7871`.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
