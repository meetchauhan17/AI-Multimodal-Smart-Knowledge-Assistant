# AI Multimodal Smart Knowledge Assistant

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Gradio-green.svg)](https://fastapi.tiangolo.com/)
[![LLM Providers](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash%20%7C%20Groq%2070B-purple.svg)](https://deepmind.google/technologies/gemini/)
[![RAG Core](https://img.shields.io/badge/RAG-ChromaDB%20%2B%20DuckDuckGo-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-ready, multimodal knowledge platform designed for domain-specific context retrieval, speech interaction, computer vision analysis, and text-to-image artwork synthesis. Built with a resilient multi-provider LLM fallback router, vector-based Retrieval-Augmented Generation (RAG), live web search fallback for real-time events, local speech-to-text (STT), text-to-speech (TTS), and a modern interactive web interface.

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Architectural Highlights & Key Features](#2-architectural-highlights--key-features)
- [3. Capstone Requirements Verification Matrix](#3-capstone-requirements-verification-matrix)
- [4. Detailed System Architecture](#4-detailed-system-architecture)
- [5. Module Breakdown & Technical Specifications](#5-module-breakdown--technical-specifications)
- [6. Project Directory Tree](#6-project-directory-tree)
- [7. Installation & Deployment Guide](#7-installation--deployment-guide)
- [8. Configuration & Environment Variables](#8-configuration--environment-variables)
- [9. Running the Application](#9-running-the-application)
- [10. User Interface Guide](#10-user-interface-guide)
- [11. API Reference & Code Examples](#11-api-reference--code-examples)
- [12. Testing & Quality Assurance](#12-testing--quality-assurance)
- [13. Troubleshooting & Edge Cases](#13-troubleshooting--edge-cases)
- [14. License](#14-license)

---

## 1. System Overview

The **AI Multimodal Smart Knowledge Assistant** integrates local domain intelligence with multi-provider generative cloud models. Designed to resolve complex queries across seven pre-indexed domain categories—College, Tourism, Healthcare, Agriculture, Library, Museums, and Historical Monuments—it seamlessly handles voice and text inputs while outputting formatted text and spoken audio.

When local vector retrieval score metrics fall below configured confidence thresholds (e.g. for real-time news like sports results or current events), the engine dynamically initiates a **Live Web Search Fallback** using DuckDuckGo to provide accurate, up-to-date answers.

---

## 2. Architectural Highlights & Key Features

### Multimodal Input & Output Processing
- **Text & Voice Q&A**: Accepts typed natural language input or recorded microphone audio.
- **Local Speech Recognition (STT)**: Uses local OpenAI Whisper (`base` model) via `ffmpeg` for automatic audio transcription.
- **Text-to-Speech Synthesis (TTS)**: Converts generated text answers into clear spoken MP3 audio files using `gTTS` with intelligent sentence boundary splitting.

### Dual-Layer RAG & Web Search Engine
- **Vector Document Search**: Ingests domain knowledge documents into a persistent `ChromaDB` vector collection using SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- **Domain Auto-Detection**: Automatically identifies relevant document domains or applies explicit user domain filters.
- **Real-Time Web Search Fallback**: Automatically invokes DuckDuckGo HTML scraping when vector document similarity is low, enabling accurate answers for real-world current events (e.g., FIFA World Cup results, breaking news).

### Resilient Multi-Provider LLM Router
- **Primary LLM**: Google Gemini API (`gemini-3.5-flash`) via the `google-genai` SDK.
- **Fallback LLM**: Groq API (`llama-3.3-70b-versatile`) for sub-second, highly reliable text generation.
- **Automated Rate Limit Handling**: Catches `RESOURCE_EXHAUSTED` (429) errors or service timeouts and reroutes requests seamlessly without user interruption.

### Vision & Creative Image Engine
- **Image Analysis & Captioning**: Upload images for computer vision analysis with support for three style modes: `Descriptive`, `Short`, and `Detailed`.
- **RAG Caption Follow-up**: Query the internal vector database using generated image captions as context.
- **Text-to-Image Generation**: Synthesize original digital artwork from descriptive text prompts using the Pollinations.ai engine with Gemini prompt optimization fallback.

### Single-Server Web Architecture
- **Interactive Web Interface**: Custom web frontend served directly at root `/` implementing Kalam/Patrick Hand typography, paper dot-grid background, and responsive panels.
- **FastAPI Core Backend**: Single Uvicorn/FastAPI process running on port `7871` serving the user interface and REST API endpoints.

---

## 3. Capstone Requirements Verification Matrix

| Module / Requirement | System Component | Primary Technology | Verification Command | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1(a): LLM Routing & Fallback** | `core/llm_provider.py` | Google Gemini 3.5 Flash / Groq Llama 3.3 70B | `pytest tests/test_llm_provider.py` | COMPLETED |
| **Module 1(b): Speech-to-Text (STT)** | `modules/speech_to_text.py` | OpenAI Whisper (Local Base Weights) | `pytest tests/test_stt.py` | COMPLETED |
| **Module 1(d): Text-to-Speech (TTS)** | `modules/text_to_speech.py` | gTTS Engine with Sentence Splitting | `pytest tests/test_tts.py` | COMPLETED |
| **Module 2(a): Image Captioning** | `modules/image_captioning.py` | Gemini 3.5 Flash Vision / Groq Vision | `pytest tests/test_image_captioning.py` | COMPLETED |
| **Module 2(b): Image Generation** | `modules/image_generation.py` | Pollinations.ai Image Synthesis Engine | `pytest tests/test_image_generation.py` | COMPLETED |
| **Module 3: Retrieval-Augmented Gen** | `modules/rag/retriever.py` | ChromaDB + DuckDuckGo Web Fallback | `pytest tests/test_rag.py` | COMPLETED |
| **Multimodal Orchestrator** | `core/assistant.py` | `MultimodalAssistant` Unified Class | `pytest tests/test_assistant.py` | COMPLETED |
| **Unified Web Interface** | `ui/gradio_app.py` | FastAPI + Gradio + SketchAgents UI | `python ui/gradio_app.py` | COMPLETED |

---

## 4. Detailed System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 USER INPUT LAYER                                  |
|   +---------------------------------------+   +-------------------------------+   |
|   |         Typed Text Question           |   |   Audio Recording (.wav/.mp3) |   |
|   +-------------------+-------------------+   +---------------+---------------+   |
+-----------------------|-----------------------------------|-----------------------+
                        |                                   |
                        |                                   v
                        |                       +-----------------------+
                        |                       |   OpenAI Whisper STT  |
                        |                       +-----------+-----------+
                        |                                   |
                        v                                   v
+-----------------------------------------------------------------------------------+
|                              RAG & RETRIEVAL LAYER                                |
|                        +---------------------------------+                        |
|                        | ChromaDB Vector Similarity Search|                        |
|                        +----------------+----------------+                        |
|                                         |                                         |
|                               +---------+---------+                               |
|                               | Vector Match Score |                               |
|                               +----+---------+----+                               |
|                                    |         |                                    |
|                      (Above Thresh)|         |(Below Threshold / Current Event)   |
|                                    v         v                                    |
|                        +-----------+--+   +--+-------------------+                |
|                        | Knowledge    |   | DuckDuckGo Live Web  |                |
|                        | Base Docs    |   | Search Snippets      |                |
|                        +-----------+--+   +--+-------------------+                |
|                                    |         |                                    |
+------------------------------------|---------|------------------------------------+
                                     v         v
+-----------------------------------------------------------------------------------+
|                               LLM ROUTER & FALLBACK                               |
|                     +---------------------------------------+                     |
|                     | Primary Provider: Gemini 3.5 Flash    |                     |
|                     +-------------------+-------------------+                     |
|                                         | (On 429 / Quota Error)                  |
|                                         v                                         |
|                     +---------------------------------------+                     |
|                     | Fallback Provider: Groq Llama 3.3 70B |                     |
|                     +-------------------+-------------------+                     |
+-----------------------------------------|-----------------------------------------+
                                          v
+-----------------------------------------------------------------------------------+
|                                 OUTPUT GENERATION                                 |
|   +---------------------------------------+   +-------------------------------+   |
|   |    Formatted Written Markdown Text    |   |  gTTS Speech Audio Synthesis  |   |
|   +---------------------------------------+   +-------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 5. Module Breakdown & Technical Specifications

### `core/llm_provider.py` — Resilient Provider Router
- Implements `LLMProvider` managing model invocations across Gemini and Groq.
- Uses `google-genai` SDK for Gemini interactions (`gemini-3.5-flash`).
- Uses `groq` SDK for Groq interactions (`llama-3.3-70b-versatile`).
- Features explicit exception handling: when Gemini encounters `RESOURCE_EXHAUSTED` (Rate limit / Quota exceeded), execution seamlessly falls back to Groq in under 100ms.

### `modules/rag/` — Vector Storage & Retrieval
- **`ingestion.py`**: Scans `data/knowledge_base/`, splits document text into normalized chunks (default chunk size: 500 characters with 50 character overlap), and indexes vectors in `ChromaDB`.
- **`retriever.py`**: Performs semantic search using cosine similarity distance. If vector search results yield low relevance scores, the system automatically calls `modules/web_search.py` to retrieve live web search snippets and injects them as prompt context.

### `modules/web_search.py` — Live Search Fallback
- Executes lightweight queries against DuckDuckGo HTML endpoint without third-party API keys.
- Parses snippet text using `BeautifulSoup4` and normalizes response content for injection into the LLM system prompt.

### `modules/speech_to_text.py` — Audio Transcription
- Uses local `openai-whisper` (`base` model weights).
- Converts input audio files (WAV, MP3, OGG, M4A) via `pydub` and `ffmpeg` before running model inference.
- Automatically handles temporary file cleanup.

### `modules/text_to_speech.py` — Audio Synthesis
- Converts text responses into spoken voice using Google Text-to-Speech (`gTTS`).
- Splits long text passages into clause-level sentences to prevent synthesis timeouts and ensures natural speech pauses.
- Concatenates audio segments using `pydub` into a final MP3 file saved in `temp_audio/`.

### `modules/image_captioning.py` — Vision Analysis
- Accepts image files (JPEG, PNG, WEBP) and decodes raw image bytes.
- Transmits image payloads to Gemini 3.5 Flash Vision or Groq Vision models.
- Formats analysis outputs according to selected style parameter (`Descriptive`, `Short`, `Detailed`).

### `modules/image_generation.py` — Image Synthesis
- Connects to Pollinations.ai REST service to generate images from textual prompts.
- Saves generated images to `generated_images/` directory and returns file paths for display in UI frames.

---

## 6. Project Directory Tree

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
├── data/
│   └── knowledge_base/          # Source documents across 7 domain categories
│       ├── agriculture/
│       ├── college/
│       ├── healthcare/
│       ├── library/
│       ├── monuments/
│       ├── museums/
│       └── tourism/
├── modules/
│   ├── __init__.py
│   ├── image_captioning.py      # Vision analysis engine
│   ├── image_generation.py      # Text-to-image synthesis module
│   ├── speech_to_text.py        # Local Whisper STT module
│   ├── text_to_speech.py        # gTTS audio synthesis module
│   ├── web_search.py            # Live DuckDuckGo web search engine
│   └── rag/
│       ├── __init__.py
│       ├── ingestion.py          # Document chunking & ChromaDB vector indexing
│       └── retriever.py         # Vector similarity search & web fallback logic
├── ui/
│   ├── gradio_app.py            # Unified FastAPI + Gradio server entrypoint
│   └── web_modern/              # SketchAgents Notebook Frontend
│       ├── app.js               # Gradio JS Client integration
│       ├── index.html           # Notebook layout HTML structure
│       └── style.css            # Hand-drawn design system & tokens
├── tests/                       # Complete pytest unit & integration test suite
│   ├── conftest.py
│   ├── test_assistant.py
│   ├── test_end_to_end.py
│   ├── test_image_captioning.py
│   ├── test_image_generation.py
│   ├── test_llm_provider.py
│   ├── test_rag.py
│   ├── test_stt.py
│   └── test_tts.py
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusion configuration
├── README.md                    # System documentation
├── requirements.txt             # Main dependencies
└── verify_llm.py                # Direct provider diagnostic tool
```

---

## 7. Installation & Deployment Guide

### Prerequisites
- **Python**: Version 3.11 or 3.12 installed.
- **Git**: Version control system.
- **FFmpeg**: Required for audio slicing and transcription format conversion.

### Step 1: Install FFmpeg

- **Windows**:
  1. Download build from [ffmpeg.org](https://ffmpeg.org/download.html).
  2. Extract files to `C:\ffmpeg`.
  3. Add `C:\ffmpeg\bin` to System Environment Variables under `Path`.
  4. Verify installation in command prompt: `ffmpeg -version`.

- **macOS**:
  ```bash
  brew install ffmpeg
  ```

- **Linux (Ubuntu / Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

### Step 2: Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/meetchauhan17/AI-Multimodal-Smart-Knowledge-Assistant.git
cd AI-Multimodal-Smart-Knowledge-Assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS / Linux:
source .venv/bin/activate

# Upgrade package manager
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 8. Configuration & Environment Variables

Copy `.env.example` to create `.env` in the root folder:

```bash
cp .env.example .env
```

Populate the required credentials in `.env`:

```env
# -----------------------------------------------------------------------------
# LLM PROVIDER CREDENTIALS & MODEL SELECTION
# -----------------------------------------------------------------------------

# Google Gemini API Settings (Primary)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_VISION_MODEL=gemini-3.5-flash

# Groq API Settings (Fallback)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Routing Priority Order
PRIMARY_PROVIDER=gemini
FALLBACK_ORDER=gemini,groq

# -----------------------------------------------------------------------------
# APPLICATION SETTINGS
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
VECTOR_DB_PATH=data/vector_db
TEMP_AUDIO_DIR=temp_audio
GENERATED_IMAGES_DIR=generated_images
```

---

## 9. Running the Application

Start the unified application server with a single command:

```bash
python ui/gradio_app.py
```

Upon execution, the server starts directly on port 7871:

```text
2026-07-21 19:33:47 | INFO | Starting Hand-Drawn Notebook UI Server on http://127.0.0.1:7871...
===================================================
  Hand-Drawn Notebook UI  ->  http://127.0.0.1:7871
===================================================
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:7871 (Press CTRL+C to quit)
```

Access point in your web browser:
- **Hand-Drawn Notebook UI (Main Interface)**: `http://127.0.0.1:7871`

---

## 10. User Interface Guide

### SketchAgents Notebook UI (`/ui/`)

The main interface is styled as a hand-drawn physical notebook:

1. **Tab 1: Ask (Text & Voice)**
   - Type questions into the question text area or click **Click to Record** to record voice input via microphone.
   - Select domain filter (e.g. `College`, `Tourism`, `Healthcare`, `Auto-detect`).
   - Click **Submit Question** to generate a written answer on a sticky-note card and listen to the audio output player.
   - Metadata tags highlight provider used (`gemini` or `groq`) and sources cited.

2. **Tab 2: Image Captioning**
   - Upload image files via drag-and-drop or file picker.
   - Choose description style (`Descriptive`, `Short`, `Detailed`).
   - View generated caption alongside latency metrics.
   - Click **Learn More via RAG** to query vector knowledge using the caption as context.

3. **Tab 3: Image Generation**
   - Enter descriptive prompts into the image prompt text area.
   - Click **Generate Image** to synthesize new artwork displayed inside a polaroid frame.

---

## 11. API Reference & Code Examples

### Endpoint Index

| Endpoint Name | Path | Description | Input Payload | Output Data Array |
| :--- | :--- | :--- | :--- | :--- |
| `handle_ask` | `/gradio/api/handle_ask` | Multimodal Voice & Text Q&A | `[question, audio_file, domain]` | `[ret_question, answer, audio_path, sources, provider]` |
| `handle_caption` | `/gradio/api/handle_caption` | Image Analysis & Captioning | `[image_file, style]` | `[caption, provider, latency, error]` |
| `handle_caption_followup` | `/gradio/api/handle_caption_followup` | Vector RAG from Caption | `[caption]` | `[answer, sources]` |
| `handle_image_gen` | `/gradio/api/handle_image_gen` | Text-to-Image Synthesis | `[prompt]` | `[image_path, provider, error]` |

### Code Example: Python Client Integration

Using `requests` to call the running backend API:

```python
import requests

SERVER_URL = "http://127.0.0.1:7871/gradio/api/handle_ask"

payload = {
    "data": [
        "Who won FIFA 2026?",  # Question string
        None,                  # Audio file path (None if text-only)
        "Auto-detect"          # Domain dropdown selection
    ]
}

response = requests.post(SERVER_URL, json=payload)
if response.status_code == 200:
    result = response.json()
    returned_question, answer, audio_url, sources, provider = result["data"]
    print("Provider:", provider)
    print("Answer:", answer)
    print("Audio File:", audio_url)
else:
    print("Error:", response.status_code, response.text)
```

### Code Example: JavaScript Client Integration

Using `@gradio/client` in browser applications:

```javascript
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client";

async function askQuestion() {
  const client = await Client.connect("http://127.0.0.1:7871/gradio");
  
  const result = await client.predict("/handle_ask", [
    "What are the college admission requirements?", // Question
    null,                                          // Voice recording file
    "College"                                      // Domain filter
  ]);

  const [question, answer, audioPath, sources, provider] = result.data;
  console.log("Answer:", answer);
  console.log("Provider Used:", provider);
}

askQuestion();
```

### Code Example: cURL Command

```bash
curl -X POST "http://127.0.0.1:7871/gradio/api/handle_ask" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         "What are health tips for staying fit?",
         null,
         "Healthcare"
       ]
     }'
```

---

## 12. Testing & Quality Assurance

The codebase includes an extensive suite of 29 automated test cases covering unit logic, failure modes, and end-to-end integration flows.

### Execute Entire Test Suite

```bash
python -m pytest tests/ -v
```

### Execute Specific Test Modules

```bash
# Test LLM provider fallback logic (Gemini -> Groq)
python -m pytest tests/test_llm_provider.py -v

# Test RAG retrieval and live web search fallback
python -m pytest tests/test_rag.py -v

# Test Speech-to-Text Whisper integration
python -m pytest tests/test_stt.py -v

# Test Text-to-Speech audio generation
python -m pytest tests/test_tts.py -v

# Test vision image captioning
python -m pytest tests/test_image_captioning.py -v

# Test text-to-image synthesis
python -m pytest tests/test_image_generation.py -v

# Test MultimodalAssistant orchestrator
python -m pytest tests/test_assistant.py -v

# Test end-to-end user workflows
python -m pytest tests/test_end_to_end.py -v
```

---

## 13. Troubleshooting & Edge Cases

### 1. `FileNotFoundError` or `pydub` Warnings for FFmpeg
- **Symptom**: Error messages stating `ffmpeg` or `ffprobe` was not found.
- **Solution**: Ensure FFmpeg is installed and `ffmpeg/bin` path is added to your system environment variables. Verify by executing `ffmpeg -version` in command prompt.

### 2. ChromaDB Database File Locking
- **Symptom**: `sqlite3.OperationalError: database is locked`.
- **Solution**: Avoid running `pytest` concurrently with a running `python ui/gradio_app.py` process. Exit the application server before initiating automated test runs.

### 3. Gemini API Rate Limiting (`RESOURCE_EXHAUSTED` / 429 Error)
- **Symptom**: Console logs show `google.genai.errors.APIError: 429 RESOURCE_EXHAUSTED`.
- **Solution**: The internal `LLMProvider` automatically routes requests to Groq (`llama-3.3-70b-versatile`). No manual intervention is needed. To use a different Gemini model, set `GEMINI_MODEL=gemini-1.5-flash` in your `.env` file.

### 4. Microphone Input Not Recording in Browser
- **Symptom**: Clicking microphone button produces no audio data or error alert.
- **Solution**: Ensure browser permissions grant microphone access to `http://127.0.0.1:7871`.

---

## 14. License

This project is licensed under the MIT License. See the `LICENSE` file for details.
