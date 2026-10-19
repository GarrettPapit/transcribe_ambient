# Transcribe_Ambient — Portfolio Showcase, Visual Assets & Video Guide

> **Author:** Garrett Papit  
> **Project Repository:** [https://github.com/GarrettPapit/transcribe_ambient](https://github.com/GarrettPapit/transcribe_ambient)  
> **Interactive Portfolio Showcase:** [`docs/portfolio_showcase.html`](./portfolio_showcase.html)

---

## 🌟 1. Executive Summary & Case Study (STAR Method)

Use this narrative for your resume bullet points, portfolio case study write-up, and interview presentations.

### 🏢 Situation
Enterprises across legal, healthcare, financial, and executive sectors capture hours of ambient audio daily (board meetings, voice memos, client consultations, conference calls). However, two major hurdles block effective transcription:
1. **Acoustic Degradation & Hallucinations:** Room recordings suffer from low Signal-to-Noise Ratio (SNR), HVAC hum, and reverberation. Feeding unconditioned room audio with long silent pauses into raw speech models triggers severe transcription hallucinations (e.g., repetitive looping tokens).
2. **Data Sovereignty & Cloud Costs:** Cloud-based APIs (OpenAI Whisper API, AWS Transcribe, Google Cloud Speech) incur continuous per-minute fees (\$0.024/min adds up to tens of thousands of dollars annually) and introduce unacceptable compliance liabilities (violating HIPAA, GDPR, or Attorney-Client Privilege).

### 🎯 Task
Architect and implement a production-grade, 100% on-premise audio transcription and intelligence pipeline that runs locally on standard hardware without internet access, removes acoustic noise and silence before transcription, executes fast quantized speech recognition, and produces structured business-ready documentation using local LLMs.

### ⚙️ Action
* **Stage 1 (Ingestion & Metadata Diagnostics):** Built automated directory ingestion with `ffprobe` and `TinyTag` metadata extraction, dynamic codec routing (`.wav`, `.mp3`, `.m4a`, `.aac`, `.amr`, `.3ga`), and regex timestamp normalization.
* **Stage 2 (Digital Signal Processing - DSP):** Engineered an FFmpeg filter pipeline utilizing `afftdn` (adaptive FFT noise profiling at `-30dB`), two-way silence stripping at `-30dB` thresholds, and `loudnorm` (EBU R128 loudness normalization) to standard 16kHz mono audio.
* **Stage 3 (Quantized Speech Recognition):** Integrated `faster-whisper` (CTranslate2 backend) with an `int8` quantized model, multi-threaded parallel execution (3 CPU threads / 2 workers), and integrated Voice Activity Detection (VAD).
* **Stage 4 (Cognitive Polish & Structured Synthesis):** Integrated local `Ollama` (`llama3`) to act as an automated acoustic archivist—clearing speech stutters, structuring dialogue into actionable paragraphs, and embedding ISO audit headers.
* **User Interfaces:** Delivered both a headless automated CLI pipeline with automated source classification and a dark-mode desktop GUI (`gui_transcribe.py`) with real-time streaming console redirection.

### 🚀 Result
* **100% Data Sovereignty:** Zero bytes transmitted over the WAN; completely air-gapped and HIPAA/SOC2 compliant.
* **$0 Ongoing API Invoices:** Eliminated recurring third-party cloud transcription billing.
* **~60% Latency Reduction:** Stripping room silence before Whisper cuts model inference compute time by more than half while virtually eliminating hallucination loops.

---

## 🎨 2. The 4 Key Visual Assets for Your Portfolio

To make your portfolio immediately captivating, include these 4 distinct visual elements:

| Asset # | Asset Name | Description & Focus | Where to Find / Capture |
| :--- | :--- | :--- | :--- |
| **Asset 1** | **System Architecture & Data Flowchart** | Clean diagram showing the 4 stages from raw ambient audio to polished Markdown notes. | Tab 1 in [`portfolio_showcase.html`](./portfolio_showcase.html) |
| **Asset 2** | **Desktop Application GUI Mockup** | Sleek screenshot of `gui_transcribe.py` processing files with streaming terminal output. | Tab 2 in [`portfolio_showcase.html`](./portfolio_showcase.html) or run `python src/gui_transcribe.py` |
| **Asset 3** | **Audio Waveform & Transcript Diff** | High-contrast side-by-side: Raw audio vs. DSP-cleaned audio + Hallucinated Whisper vs. Clean Ollama transcript. | Tab 3 in [`portfolio_showcase.html`](./portfolio_showcase.html) |
| **Asset 4** | **Enterprise ROI & Privacy Matrix** | Visual comparison chart highlighting \$0 cost, offline capability, and zero cloud leaks. | Tab 4 in [`portfolio_showcase.html`](./portfolio_showcase.html) |

---

## 🎬 3. Video Demo Storyboard & Screen-Recording Script (60–75 Seconds)

This script is structured for maximum viewer retention on LinkedIn, GitHub, or your personal website.

### Technical Recording Settings:
* **Resolution:** 1080p (1920x1080) at 60 FPS
* **Tools:** Loom, OBS Studio, or Windows Game Bar (<kbd>Win</kbd> + <kbd>Alt</kbd> + <kbd>R</kbd>)
* **Audio:** Crisp microphone narration or upbeat background lofi track with caption overlays

---

### Scene-by-Scene Script

#### **Scene 1: The Problem (00:00 - 00:15)**
* **Visual on Screen:** Show an audio editor or waveform with loud background hiss and long silent pauses. Open a text document showing a repetitive Whisper hallucination (e.g. `[cough] thank you for watching thank you for watching...`).
* **Voiceover:**
  > *"When dealing with room audio—like meeting recordings or voice memos—standard speech models often struggle with background noise and silence hallucinations. Meanwhile, cloud speech APIs introduce recurring costs and serious data privacy risks."*

#### **Scene 2: DSP Acoustic Conditioning (00:15 - 00:35)**
* **Visual on Screen:** Switch to the terminal or code showing the FFmpeg DSP filter (`afftdn=nr=12:nf=-30,silenceremove,loudnorm`). Show the file duration drop from 3 minutes down to 1 minute as silence is pruned.
* **Voiceover:**
  > *"To solve this, I built **Transcribe_Ambient**—a zero-cloud, 100% local AI pipeline. It starts with dynamic digital signal processing using FFmpeg. We apply adaptive FFT noise filtering and dual-threshold silence removal, reducing audio duration by 60% and eliminating the root cause of AI hallucinations."*

#### **Scene 3: Quantized Faster-Whisper & Desktop GUI (00:35 - 00:50)**
* **Visual on Screen:** Open the desktop application (`gui_transcribe.py`). Select a folder and click **"Start Batch Processing"**. Watch the dark-mode console window stream live processing updates across parallel CPU threads.
* **Voiceover:**
  > *"Next, Faster-Whisper processes the audio using INT8 quantization across parallel CPU cores. This delivers high-speed, accurate speech recognition on commodity hardware without requiring expensive cloud GPUs."*

#### **Scene 4: Ollama Local LLM Polish & Final Deliverable (00:50 - 01:10)**
* **Visual on Screen:** Open the generated `.txt` / `.md` file in VS Code. Highlight the ISO audit metadata header, structured paragraphs, and clean action item bullet points.
* **Voiceover:**
  > *"Finally, an on-premise Llama 3 instance via Ollama cleans up speech stutters and structures the transcript into an executive summary with audit metadata. The result? 100% on-premise data privacy, zero API bills, and production-ready meeting intelligence."*

#### **Scene 5: Call to Action (01:10 - 01:20)**
* **Visual on Screen:** Show the GitHub repository homepage ([`GarrettPapit/transcribe_ambient`](https://github.com/GarrettPapit/transcribe_ambient)).
* **Voiceover:**
  > *"The entire codebase, CLI tools, and desktop GUI are open-source on my GitHub. Check out the link below to explore the architecture!"*

---

## 💼 4. Ready-to-Publish LinkedIn / Portfolio Post

Copy and paste this template directly to LinkedIn, Twitter/X, or your portfolio blog:

```markdown
🎙️ Solving Ambient Speech Recognition with Zero-Cloud Local AI 🤖

Transcribing unconditioned room audio (meetings, voice memos, ambient microphones) is notoriously difficult:
❌ Low SNR and room reverberation degrade speech recognition accuracy.
❌ Long silent pauses trigger hallucination loops in standard Whisper models.
❌ Cloud APIs (AWS Transcribe, OpenAI) create perpetual per-minute invoices and introduce major privacy/compliance liabilities for sensitive enterprise data.

To tackle this, I engineered Transcribe_Ambient — a high-performance, 100% local audio intelligence pipeline:

🔹 Stage 1: Adaptive DSP Conditioning
Using FFmpeg, the pipeline applies adaptive FFT spectral noise reduction (`afftdn`), two-way silence removal (-30dB floor), and EBU R128 loudness normalization. This cuts audio duration by up to 60% and strips the noise that causes hallucinations.

🔹 Stage 2: Quantized Faster-Whisper
Using CTranslate2 with INT8 quantization, the system executes parallel beam-search transcription across CPU threads, providing lightning-fast inference on standard workstations with zero GPU requirements.

🔹 Stage 3: On-Premise LLM Synthesis
An on-premise Llama 3 instance (via Ollama) acts as an acoustic archivist — stripping speech stutters, formatting dialogue into executive summaries, and attaching audit metadata.

Key Enterprise Benefits:
✅ 100% Data Sovereignty: Zero audio or text ever leaves localhost (air-gapped & HIPAA/SOC-2 ready).
✅ $0 Marginal API Cost: Eliminates thousands in annual cloud transcription invoices.
✅ Dual Interfaces: High-throughput CLI batch watcher + native dark-mode Desktop GUI.

Explore the open-source architecture and codebase on GitHub:
👉 https://github.com/GarrettPapit/transcribe_ambient

#ArtificialIntelligence #LocalAI #MachineLearning #Python #Ollama #Whisper #OpenSource #DataPrivacy #CyberSecurity
```

---

## 📸 5. How to Launch & Capture the Interactive Showcase

You have an interactive, live web showcase included in your repository:

1. Open [`docs/portfolio_showcase.html`](./portfolio_showcase.html) in any modern browser (Chrome, Edge, Brave).
2. Use **Tab 1** for high-resolution screenshots of the 4-stage pipeline architecture.
3. Use **Tab 2** to click **"Run Interactive Simulation"** for screen recordings of the desktop GUI in action.
4. Use **Tab 3** to screenshot the Before/After audio waveforms and transcript comparison diff.
5. Use **Tab 4** to demonstrate the interactive ROI cost calculator.

