# Local AI Ambient Audio Transcription Pipeline

> A high-performance, zero-cloud local AI transcription and audio enhancement pipeline engineered with Faster-Whisper, FFmpeg DSP filters, and Ollama local LLM refinement.

---

## 🚀 System Architecture & Data Flow
1. **Ingestion:** Raw audio files (`.wav`, `.mp3`, `.m4a`, etc.) are dropped into the watched staging directory.
2. **Pre-Flight Inspection & Metadata:** `ffprobe` extracts sample rate, codec, and track duration while verifying local hardware constraints.
3. **Digital Signal Processing (DSP):** FFmpeg applies multi-stage noise suppression (`afftdn`), silence trimming, and loudness normalization.
4. **AI Transcription Engine:** `faster-whisper` processes audio streams via optimized CPU/GPU beam search.
5. **LLM Polish & Enrichment:** Local LLMs via `Ollama` clean grammatical artifacts, format structured notes, and output clean Markdown files.

## ⚙️ Local Prerequisites
- **Python 3.10+**
- **FFmpeg** (must be installed and added to system PATH)
- **Ollama** running locally (`ollama serve`) with `llama3` pulled.

## 🛠️ Quick Start & Installation

1. Clone the repository and create a virtual environment:
   ```bash
   git clone [https://github.com/GarrettPapit/transcribe-ambient.git](https://github.com/GarrettPapit/transcribe-ambient.git)
   cd transcribe-ambient
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate.ps1
