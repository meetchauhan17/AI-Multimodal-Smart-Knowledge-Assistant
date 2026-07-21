# AI Multimodal Smart Knowledge Assistant — Capstone Project

An extensible, multimodal knowledge assistant designed to answer domain-specific queries using text, speech (Voice Q&A), and images (Image Captioning & Generation). It features a robust LLM routing and fallback mechanism prioritizing a local or remote **IBM Bob Shell CLI** connection, falling back gracefully to **Groq** and **Google Gemini** APIs.

---

## Capstone Features Checklist (Requirements Mapping)

| Module / Requirement | Status | Implementation File | Verification Command |
| :--- | :--- | :--- | :--- |
| **Module 1(a): Fallback LLM Routing** | `COMPLETED` | [llm_provider.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/core/llm_provider.py) | `pytest tests/test_llm_provider.py` |
| **Module 1(b): Speech-to-Text (STT)** | `COMPLETED` | [speech_to_text.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/modules/speech_to_text.py) | `pytest tests/test_stt.py` |
| **Module 1(d): Text-to-Speech (TTS)** | `COMPLETED` | [text_to_speech.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/modules/text_to_speech.py) | `pytest tests/test_tts.py` |
| **Module 2(a): Image Captioning** | `COMPLETED` | [image_captioning.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/modules/image_captioning.py) | `pytest tests/test_image_captioning.py` |
| **Module 2(b): Image Generation** | `COMPLETED` | [image_generation.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/modules/image_generation.py) | `pytest tests/test_image_generation.py` |
| **Module 3: Retrieval-Augmented Gen (RAG)** | `COMPLETED` | [retriever.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/modules/rag/retriever.py) | `pytest tests/test_rag.py` |
| **Multimodal Orchestrator** | `COMPLETED` | [assistant.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/core/assistant.py) | `pytest tests/test_assistant.py` |
| **Gradio Blocks Layout (3 Tabs)** | `COMPLETED` | [gradio_app.py](file:///c:/Meet/xyz/AI%20Multimodal%20Smart%20Knowledge%20Assistant/multimodal-knowledge-assistant/ui/gradio_app.py) | `python ui/gradio_app.py` |

---

## Project Structure

```text
multimodal-knowledge-assistant/
├── .env.example              # Template for credentials & settings
├── requirements.txt          # Pinned project dependencies
├── README.md                 # Project guide (this file)
├── config/
│   └── settings.py           # Configuration loader via Pydantic BaseSettings
├── shared/
│   ├── bob.py                # Subprocess runner for IBM Bob CLI
│   ├── bob_llm.py            # CrewAI LLM adapter monkey-patching Bob CLI
│   └── tools/
│       └── bob_tool.py       # CrewAI @tool adapter for Agents
├── core/
│   ├── llm_provider.py       # Fallback-based LLM router (Bob → Groq → Gemini)
│   ├── assistant.py          # Unified Multimodal assistant orchestrator class
│   ├── schemas.py            # Pydantic data schemas for API consistency
│   └── logger.py             # Loguru configured logging (stdout & daily rotation)
├── modules/
│   ├── speech_to_text.py     # Local speech recognition using Whisper (base)
│   ├── text_to_speech.py     # Text-to-speech output using gTTS (sentence splitting)
│   ├── image_captioning.py   # Image analysis and captioning (descriptive/short/detailed)
│   ├── image_generation.py   # Text-to-image generator (Gemini / Pollinations fallback)
│   └── rag/
│       ├── ingestion.py      # Raw document chunking & vector database ingestion
│       └── retriever.py      # Semantic text search across vector database
├── data/
│   └── knowledge_base/       # Domain-specific knowledge base documents
├── ui/
│   └── gradio_app.py         # Gradio-based multi-tab web user interface
├── tests/                    # Testing folder for unit and integration tests
└── logs/                     # Auto-created folder for application logs
```

---

## Prerequisites & Setup

### 1. Python Environment
This project requires **Python 3.11**.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. FFmpeg Installation (Required for Audio/Voice Q&A)
`pydub` (used for audio slicing and concatenation in Text-to-Speech and Speech-to-Text) requires `ffmpeg` and `ffprobe` to be available on your system `PATH`.
- **Windows**: Download the binary package from [ffmpeg.org](https://ffmpeg.org/), extract it, and add the `bin` directory to your system Environment Variables.
- **macOS (Homebrew)**: `brew install ffmpeg`
- **Linux (apt)**: `sudo apt-get install ffmpeg`

### 3. IBM Bob Shell CLI Verification
Confirm that the `bob` command line interface is installed and accessible:
```bash
bob --help
```
*Note: If `bob` CLI is not found or is missing an API key, the system automatically falls back to Groq and Gemini (see Graceful Degradation below).*

### 4. Whisper Local Model Check
Upon running a speech query or voice test for the first time, `openai-whisper` will automatically download the local weight files (defaults to the **base** model size, ~140MB) and cache them in `~/.cache/whisper`. No HuggingFace credentials or tokens are needed.

### 5. Environment Credentials Setup
1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your keys in the `.env` file:
   - `BOBSHELL_API_KEY`: Primary API key for the Bob CLI.
   - `GROQ_API_KEY`: Groq API key for Llama 3 models.
   - `GEMINI_API_KEY`: Gemini API key for multimodal model tasks.

---

## Graceful Degradation & Fallback Routing

The `LLMProvider` is built to be extremely resilient. If:
- `BOBSHELL_API_KEY` is not present in `.env`
- The `bob` executable is missing on your system `PATH`
- The CLI command returns a non-zero exit code or fails due to network issues

The router automatically redirects requests to **Groq** (`llama-3.3-70b-versatile`) and **Gemini** (`gemini-2.0-flash`) in sequence without displaying any error trace to the end-user.

---

## Running the Application

To start the Gradio UI:
```bash
python ui/gradio_app.py
```
Open the local address in your browser: `http://127.0.0.1:7860`.

---

## Running Verification Tests

Run the full automated test suite containing 29 test cases:
```bash
python -m pytest tests/
```

---

## Known Limitations

1. **Text-to-Speech (gTTS)**: Uses Google Translate's TTS, which requires an active internet connection to download synthesized chunks.
2. **Local Transcription Latency**: `openai-whisper` runs locally on the CPU. While the default `base` model balances speed and accuracy, first-load takes ~1-2 seconds to instantiate the weights into memory.
3. **ChromaDB File Lock**: Running tests and the Gradio app concurrently may lead to file lock conflicts. Ensure you exit the Gradio app server before running the automated test suite.
