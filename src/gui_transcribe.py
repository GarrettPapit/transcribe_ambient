import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from tinytag import TinyTag
from faster_whisper import WhisperModel
import ollama

# Environment Configuration
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".aac", ".amr", ".mp4", ".m4a", ".3ga"}


# --- PIPELINE BACKEND FUNCTIONS ---

def extract_metadata(audio_path):
    try:
        tag = TinyTag.get(audio_path)
        return {
            "filename": os.path.basename(audio_path),
            "filesize_mb": round(os.path.getsize(audio_path) / (1024 * 1024), 2),
            "duration_sec": round(tag.duration or 0, 2),
            "bitrate_kbps": tag.bitrate,
            "samplerate_hz": tag.samplerate,
            "channels": tag.channels,
        }
    except Exception as e:
        return {
            "filename": os.path.basename(audio_path),
            "filesize_mb": round(os.path.getsize(audio_path) / (1024 * 1024), 2),
            "error": f"Metadata read warning: {str(e)}"
        }

def preprocess_and_strip_silence(input_path, output_path):
    filter_chain = (
        "afftdn=nr=12:nf=-30,"
        "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:"
        "stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,"
        "loudnorm"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", filter_chain,
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")

def transcribe_speech(model, processed_audio_path):
    segments, _ = model.transcribe(
        str(processed_audio_path),
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    return " ".join([segment.text.strip() for segment in segments])

def salvage_audio_pass(path_obj, model):
    salvage_wav = path_obj.with_suffix(".salvage.wav")
    print("      Standard pass yielded no speech. Initiating Deep Salvage Fallback...")
    try:
        salvage_filter = (
            "highpass=f=200,lowpass=f=3400,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "afftdn=nr=18:nf=-45,"
            "loudnorm"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(path_obj),
            "-af", salvage_filter,
            "-ar", "16000", "-ac", "1",
            str(salvage_wav)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300, check=True)

        segments, _ = model.transcribe(
            str(salvage_wav),
            beam_size=5,
            vad_filter=False,
            no_speech_threshold=0.25,
            logprob_threshold=-1.8,
            temperature=[0.0, 0.2, 0.4],
            initial_prompt="Muffled room conversation with low-volume speech."
        )
        salvaged_text = " ".join([s.text.strip() for s in segments]).strip()
        if salvaged_text:
            print(f"      [+] Deep Salvage recovered text: '{salvaged_text[:80]}...'")
        return salvaged_text
    except Exception as e:
        print(f"      [!] Deep Salvage pass encountered error: {e}")
        return ""
    finally:
        if salvage_wav.exists():
            salvage_wav.unlink()

def polish_with_ollama(raw_transcript, metadata, is_salvaged=False):
    client = ollama.Client(host='http://127.0.0.1:11434')
    
    if is_salvaged:
        prompt = f"""
You are an expert audio archivist and forensic transcription analyst.
The following text was recovered via a deep-salvage acoustic recovery pass from heavily degraded room audio:

AUDIO METADATA:
- File Name: {metadata.get('filename')}
- Duration: {metadata.get('duration_sec', 'N/A')} seconds
- Original Size: {metadata.get('filesize_mb', 'N/A')} MB

RAW RECOVERED FRAGMENTS:
"{raw_transcript}"

INSTRUCTIONS:
1. Clean grammatical stutters and reconstruct the most likely spoken sentences based on phonetic context.
2. For uncertain or reconstructed words, wrap them in brackets like [Probable: ...].
3. Append a brief header containing key metadata summary above the transcription and a note "[Note: Recovered via Deep Salvage Acoustic Filter]" at the bottom.
"""
    else:
        prompt = f"""
You are an expert audio archivist. Clean and format the raw transcript below, which was generated from an ambient room audio recording.

AUDIO METADATA:
- File Name: {metadata.get('filename')}
- Duration: {metadata.get('duration_sec', 'N/A')} seconds
- Original Size: {metadata.get('filesize_mb', 'N/A')} MB

RAW TRANSCRIPT:
"{raw_transcript}"

INSTRUCTIONS:
1. Remove speech-to-text hallucination loops or ambient noise interpretations.
2. Clean grammatical stutters and format into logical dialogue or readable paragraphs.
3. If no intelligible spoken dialogue is present, write exactly: "[No intelligible speech detected in recording]".
4. Append a brief header containing key metadata summary above the transcription.
"""

    response = client.generate(model="llama3", prompt=prompt)
    return response["response"]

def process_single_file(path_obj, model):
    temp_cleaned_wav = path_obj.with_suffix(".temp_cleaned.wav")
    output_txt = path_obj.with_suffix(".txt")

    print(f"\n--- Processing: {path_obj.name} ---")
    metadata = extract_metadata(str(path_obj))

    print("[1/3] Running afftdn noise reduction & trimming silence...")
    try:
        preprocess_and_strip_silence(path_obj, temp_cleaned_wav)
    except Exception as e:
        print(f"      ERROR in preprocessing: {e}")
        return

    print("[2/3] Transcribing audio with faster-whisper...")
    raw_transcript = transcribe_speech(model, temp_cleaned_wav)

    is_salvaged = False
    if not raw_transcript.strip() and metadata.get("duration_sec", 0) > 3:
        raw_transcript = salvage_audio_pass(path_obj, model)
        if raw_transcript.strip():
            is_salvaged = True

    print("[3/3] Polishing transcript with Ollama...")
    if not raw_transcript.strip():
        final_text = "[No intelligible speech detected in recording]"
    else:
        final_text = polish_with_ollama(raw_transcript, metadata, is_salvaged=is_salvaged)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(final_text)

    if temp_cleaned_wav.exists():
        temp_cleaned_wav.unlink()

    print(f"SUCCESS: Saved output to '{output_txt.name}'")



# --- TEXT REDIRECTOR CLASS FOR TKINTER ---

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str_val):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, str_val)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)

    def flush(self):
        pass


# --- TKINTER GUI APPLICATION ---

class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ambient Audio Transcription Pipeline")
        self.root.geometry("720 x 520")
        self.root.minsize(600, 400)

        # Apply basic modern styling
        style = ttk.Style()
        style.theme_use("clam")

        # Top Frame: Directory Selection
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Target Folder:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.dir_entry = ttk.Entry(top_frame)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.dir_entry.insert(0, os.getcwd())

        self.browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_folder)
        self.browse_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Middle Frame: Controls & Status
        mid_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        mid_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(mid_frame, text="Start Batch Processing", command=self.start_thread)
        self.start_btn.pack(side=tk.LEFT)

        self.status_label = ttk.Label(mid_frame, text="Status: Ready", font=("Segoe UI", 9, "italic"))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Bottom Frame: Console Output Window
        console_frame = ttk.Frame(self.root, padding=10)
        console_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(console_frame, text="Execution Output Log:").pack(anchor=tk.W, pady=(0, 2))
        
        self.console_text = scrolledtext.ScrolledText(
            console_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED, 
            bg="#1E1E1E", 
            fg="#D4D4D4", 
            insertbackground="white",
            font=("Consolas", 9)
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout/stderr to GUI ScrolledText
        sys.stdout = TextRedirector(self.console_text)
        sys.stderr = TextRedirector(self.console_text)

    def browse_folder(self):
        selected = filedialog.askdirectory(initialdir=self.dir_entry.get())
        if selected:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, selected)

    def start_thread(self):
        target_dir = self.dir_entry.get().strip()
        if not target_dir or not Path(target_dir).exists():
            print("Error: Selected directory does not exist.")
            return

        # Disable UI elements while running
        self.start_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Processing...")

        # Spawn background processing thread
        thread = threading.Thread(target=self.run_pipeline, args=(target_dir,), daemon=True)
        thread.start()

    def run_pipeline(self, target_dir):
        try:
            dir_path = Path(target_dir).resolve()
            audio_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

            if not audio_files:
                print(f"No supported audio files found in: {dir_path}")
                return

            print(f"Found {len(audio_files)} audio file(s) in '{dir_path.name}'.")
            print("Loading Whisper model ('small')...")
            
            # Configured for 3 threads / 2 workers
            model = WhisperModel(
                "small", 
                device="cpu", 
                compute_type="int8", 
                cpu_threads=3,
                num_workers=2
            )

            processed = 0
            skipped = 0

            for idx, audio_file in enumerate(audio_files, start=1):
                output_txt = audio_file.with_suffix(".txt")
                
                if output_txt.exists():
                    print(f"[{idx}/{len(audio_files)}] Skipping '{audio_file.name}' (Transcript exists)")
                    skipped += 1
                    continue

                print(f"\n[{idx}/{len(audio_files)}] Batch Item {idx} of {len(audio_files)}")
                process_single_file(audio_file, model)
                processed += 1

            print("\n==========================================")
            print(f"Batch Complete! Processed: {processed} | Skipped: {skipped}")
            print("==========================================")

        except Exception as e:
            print(f"\n[FATAL ERROR]: {e}")
        finally:
            # Re-enable UI elements on UI thread
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.browse_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()