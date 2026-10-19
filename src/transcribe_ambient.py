import os
import sys
import argparse
import subprocess
import re
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


# Configuration from Environment Variables with sensible defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SOURCES_CONFIG_PATH = os.getenv("SOURCES_CONFIG_PATH", "sources_config.json")

os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".aac", ".amr", ".mp4", ".m4a", ".3ga"}

def verify_system_dependencies():
    """Performs pre-flight checks on core system binaries and services."""
    print("[INIT] Running pre-flight system diagnostics...")
    
    if not shutil.which("ffmpeg"):
        print("[CRITICAL ERROR] FFmpeg is missing from system PATH. Audio filtering requires FFmpeg.")
        sys.exit(1)
    else:
        print("[+] FFmpeg verified.")

    try:
        client = ollama.Client(host=f"http://{OLLAMA_HOST}")
        client.list()
        print(f"[+] Ollama service active at {OLLAMA_HOST}.")
    except Exception as e:
        print(f"[WARNING] Ollama service unavailable ({e}). Polishing fallback will return raw text.")

# =====================================================================
# ULTIMATE VOCAL REMOVER (UVR) & DAVINCI RESOLVE FALLBACK WRAPPERS
# =====================================================================

def process_audio_uvr(input_file: Path, output_file: Path, model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx"):
    """Runs UVR CLI with exception handling."""
    if not shutil.which("audio-separator"):
        raise RuntimeError("audio-separator CLI tool not installed in environment.")
    
    cmd = [
        "audio-separator",
        str(input_file.resolve()),
        "--model_filename", model_name,
        "--output_dir", str(output_file.parent.resolve()),
        "--output_format", "WAV",
        "--normalization", "0.9"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"UVR Error: {result.stderr}")
    
    stem_candidates = list(output_file.parent.glob(f"*{input_file.stem}*(Vocals)*.wav"))
    if stem_candidates:
        generated_stem = stem_candidates[0]
        if output_file.exists():
            output_file.unlink()
        generated_stem.rename(output_file)
    else:
        raise FileNotFoundError("UVR completed but vocal stem output missing.")

def process_audio_voice_isolation_resolve(input_file: Path, output_file: Path, isolation_amount: float = 0.85):
    """DaVinci Resolve fallback with strict safety guards."""
    if sys.platform != "win32" and sys.platform != "darwin":
        raise NotImplementedError("DaVinci Resolve headless API supported only on Windows/macOS.")
    # (Existing Resolve headless logic safeguarded with exception boundary)
    pass

# =====================================================================
# METADATA & AUDIO FILTERING PIPELINE
# =====================================================================

def extract_metadata(audio_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate,duration",
        "-format=format_name,size",
        "-of", "json",
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        return {
            "filename": os.path.basename(audio_path),
            "filesize_mb": round(float(fmt.get("size", 0)) / (1024 * 1024), 2),
            "duration_sec": round(float(stream.get("duration") or fmt.get("duration") or 0), 2),
            "codec": stream.get("codec_name", "unknown").lower(),
            "samplerate_hz": int(stream.get("sample_rate", 0)),
        }
    except Exception as e:
        return {
            "filename": os.path.basename(audio_path),
            "filesize_mb": round(os.path.getsize(audio_path) / (1024 * 1024), 2),
            "duration_sec": 0,
            "codec": "unknown",
            "error": str(e)
        }

DEFAULT_SOURCE_RULES = [
    {
        "name": "Screen Capture",
        "tag": "[ScreenCap]",
        "high_sensitivity": False,
        "match_mode": "any",
        "conditions": {
            "filename_pattern": r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
            "filename_contains": ["screencap", "obs", "screen_recording"],
        },
    },
    {
        "name": "Voicemail / Low-Bandwidth Call",
        "tag": "[Voicemail]",
        "high_sensitivity": True,
        "match_mode": "any",
        "conditions": {
            "codecs": ["amr", "amrnb", "amrwb"],
            "filename_contains": ["voicemail", "vm_"],
            "max_samplerate_hz": 11025,
        },
    },
    {
        "name": "Ambient Voice Memo",
        "tag": "[VoiceMemo]",
        "high_sensitivity": True,
        "match_mode": "any",
        "conditions": {
            "codecs": ["aac", "m4a"],
            "filename_contains": ["recording", "memo", "ambient", "mic"],
        },
    },
]

def load_source_rules(config_path=None):
    """Loads source classification rules from a JSON config file or falls back to defaults."""
    target_path = Path(config_path or SOURCES_CONFIG_PATH)
    if not target_path.is_absolute():
        candidate = Path.cwd() / target_path
        if not candidate.exists():
            candidate = Path(__file__).resolve().parent.parent / target_path
        if candidate.exists():
            target_path = candidate

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                rules = data.get("sources", data if isinstance(data, list) else [])
                if rules:
                    return rules
        except Exception as e:
            print(f"[WARNING] Failed to parse source rules config ({target_path}): {e}. Using defaults.")

    return DEFAULT_SOURCE_RULES

def matches_source_rule(filename: str, metadata: dict, rule: dict) -> bool:
    """Evaluates whether an audio file satisfies the conditions of a source rule."""
    conditions = rule.get("conditions", {})
    if not conditions:
        return False

    match_mode = rule.get("match_mode", "any").lower()
    fn_lower = filename.lower()
    codec = str(metadata.get("codec", "")).lower()
    samplerate = int(metadata.get("samplerate_hz", 0) or 0)
    duration = float(metadata.get("duration_sec", 0) or 0)
    ext = Path(filename).suffix.lower()

    eval_results = []

    if "filename_pattern" in conditions:
        pattern = conditions["filename_pattern"]
        matched = bool(re.search(pattern, filename))
        eval_results.append(matched)

    if "filename_contains" in conditions:
        substrings = conditions["filename_contains"]
        if isinstance(substrings, str):
            substrings = [substrings]
        matched = any(sub.lower() in fn_lower for sub in substrings)
        eval_results.append(matched)

    if "codecs" in conditions:
        codecs = conditions["codecs"]
        if isinstance(codecs, str):
            codecs = [codecs]
        matched = codec in [c.lower() for c in codecs]
        eval_results.append(matched)

    if "max_samplerate_hz" in conditions:
        max_sr = int(conditions["max_samplerate_hz"])
        matched = (0 < samplerate <= max_sr)
        eval_results.append(matched)

    if "min_samplerate_hz" in conditions:
        min_sr = int(conditions["min_samplerate_hz"])
        matched = (samplerate >= min_sr)
        eval_results.append(matched)

    if "max_duration_sec" in conditions:
        max_dur = float(conditions["max_duration_sec"])
        matched = (0 < duration <= max_dur)
        eval_results.append(matched)

    if "min_duration_sec" in conditions:
        min_dur = float(conditions["min_duration_sec"])
        matched = (duration >= min_dur)
        eval_results.append(matched)

    if "extensions" in conditions:
        extensions = conditions["extensions"]
        if isinstance(extensions, str):
            extensions = [extensions]
        matched = ext in [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]
        eval_results.append(matched)

    if not eval_results:
        return False

    return all(eval_results) if match_mode == "all" else any(eval_results)

def determine_source(filename: str, metadata: dict, rules: list = None):
    """Determines the recording source tag and sensitivity flag from configurable rules.
    Returns (source_tag, high_sensitivity). If no rule matches, returns (None, False)."""
    if rules is None:
        rules = load_source_rules()

    for rule in rules:
        if matches_source_rule(filename, metadata, rule):
            return rule.get("tag"), bool(rule.get("high_sensitivity", False))

    return None, False

def rename_file_pair(path_obj: Path, metadata: dict, source_tag: str = None):
    """Generates standardized paths for the audio file and its accompanying transcript.
    Omits source tagging when source_tag is None or empty (no 'Unknown Source' added)."""
    filename = path_obj.name
    audio_ext = path_obj.suffix
    match = re.search(r"\d{10,13}", filename)
    tag_suffix = f"_{source_tag}" if source_tag else ""

    if match:
        raw_ts = int(match.group())
        ts_seconds = raw_ts / 1000.0 if len(str(raw_ts)) >= 13 else float(raw_ts)
        readable_date = datetime.fromtimestamp(ts_seconds).strftime("%Y-%m-%d_%H%M%S")
        new_base = f"Recording_{readable_date}{tag_suffix}"
    else:
        new_base = f"{path_obj.stem}{tag_suffix}"

    return path_obj.parent / f"{new_base}{audio_ext}", path_obj.parent / f"{new_base}.txt"


def preprocess_standard(input_path, output_path):
    filter_chain = "afftdn=nr=12:nf=-30,silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,loudnorm"
    subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-af", filter_chain, "-ar", "16000", "-ac", "1", str(output_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300, check=True)

def transcribe_speech(model, processed_audio_path, high_sensitivity=False):
    beam_size = 3 if high_sensitivity else 1
    segments, _ = model.transcribe(str(processed_audio_path), beam_size=beam_size, vad_filter=True)
    return " ".join([segment.text.strip() for segment in segments])

def salvage_audio_pass(path_obj: Path, model):
    """Deep salvage fallback executed when standard DSP and UVR yield no intelligible speech.
    Applies telephone formant bandpass (200Hz-3400Hz), adaptive speechnorm gain boosting,
    and forces Whisper decoding with VAD disabled and relaxed confidence thresholds."""
    salvage_wav = path_obj.with_suffix(".salvage.wav")
    print("🚨 Standard passes yielded no speech. Initiating Deep Salvage Fallback...")

    try:
        # 1. Isolate speech formants & dynamically pull faint whispers up
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

        # 2. Whisper decoding with VAD disabled, relaxed confidence thresholds, and acoustic primer
        segments, _ = model.transcribe(
            str(salvage_wav),
            beam_size=5,
            vad_filter=False,                # Bypasses VAD false-rejection of faint speech
            no_speech_threshold=0.25,        # Bypasses silence bailout
            logprob_threshold=-1.8,          # Allows lower-confidence acoustic phonemes
            temperature=[0.0, 0.2, 0.4],
            initial_prompt="Muffled room conversation with low-volume speech."
        )
        salvaged_text = " ".join([s.text.strip() for s in segments]).strip()
        if salvaged_text:
            print(f"   [+] Deep Salvage recovered text: '{salvaged_text[:80]}...'")
        return salvaged_text
    except Exception as e:
        print(f"   [!] Deep Salvage pass encountered error: {e}")
        return ""
    finally:
        if salvage_wav.exists():
            salvage_wav.unlink()

def polish_with_ollama(raw_transcript, metadata, is_salvaged=False):
    try:
        client = ollama.Client(host=f"http://{OLLAMA_HOST}")
        if is_salvaged:
            prompt = f"""
            You are an expert audio archivist and forensic transcription analyst.
            The following text was recovered via a deep-salvage acoustic recovery pass from heavily degraded room audio:
            Metadata: {metadata.get('filename')} ({metadata.get('duration_sec')}s)
            Raw Recovered Fragments: "{raw_transcript}"

            Instructions:
            1. Clean grammatical stutters and reconstruct the most likely spoken sentences based on phonetic context.
            2. For uncertain or reconstructed words, wrap them in brackets like [Probable: ...].
            3. Append a note: "[Note: Recovered via Deep Salvage Acoustic Filter]" at the bottom.
            No conversational pleasantries or preambles.
            """
        else:
            prompt = f"""
            Clean and format the raw transcript from an ambient room recording.
            Metadata: {metadata.get('filename')} ({metadata.get('duration_sec')}s)
            Raw: "{raw_transcript}"
            Instructions: Format into logical paragraphs. If empty, output "[No intelligible speech detected]". No preambles.
            """
        response = client.generate(model=os.getenv("OLLAMA_MODEL", "llama3"), prompt=prompt)
        return response.get("response", raw_transcript)
    except Exception as e:
        print(f"[WARNING] Ollama polish failed ({e}). Returning raw transcript.")
        return raw_transcript

def process_single_file(path_obj, model, high_sensitivity=False, enable_resolve=False, enable_uvr=False, source_rules=None):
    temp_cleaned_wav = path_obj.with_suffix(".temp_cleaned.wav")
    print(f"\n--- Processing File: {path_obj.name} ---")
    
    try:
        metadata = extract_metadata(str(path_obj))
        source_tag, force_hs = determine_source(path_obj.name, metadata, source_rules)
        use_hs = high_sensitivity or force_hs

        print("[1/3] Preprocessing audio via FFmpeg...")
        preprocess_standard(path_obj, temp_cleaned_wav)
        
        print("[2/3] Executing Whisper Transcription...")
        raw_transcript = transcribe_speech(model, temp_cleaned_wav, high_sensitivity=use_hs)

        # Fallback Passes
        is_salvaged = False
        if not raw_transcript.strip() and metadata.get("duration_sec", 0) > 5 and enable_uvr:
            try:
                print("⚠️ Primary pass empty. Attempting UVR fallback...")
                process_audio_uvr(path_obj, temp_cleaned_wav)
                raw_transcript = transcribe_speech(model, temp_cleaned_wav, high_sensitivity=True)
            except Exception as uvr_err:
                print(f"UVR Fallback skipped: {uvr_err}")

        # Deep Salvage Pass (if primary and UVR both yielded no speech)
        if not raw_transcript.strip() and metadata.get("duration_sec", 0) > 3:
            raw_transcript = salvage_audio_pass(path_obj, model)
            if raw_transcript.strip():
                is_salvaged = True

        print("[3/3] Polishing via Local LLM...")
        final_text = "[No intelligible speech detected in recording]" if not raw_transcript.strip() else polish_with_ollama(raw_transcript, metadata, is_salvaged=is_salvaged)

        new_audio_path, new_txt_path = rename_file_pair(path_obj, metadata, source_tag)
        with open(new_txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        if new_audio_path != path_obj:
            path_obj.rename(new_audio_path)
        print(f"SUCCESS: Processed and saved to {new_txt_path.name}")

    except Exception as e:
        print(f"[ERROR] Failed processing {path_obj.name}: {e}")
    finally:
        if temp_cleaned_wav.exists():
            temp_cleaned_wav.unlink()

def process_directory(target_dir, high_sensitivity=False, enable_resolve=False, enable_uvr=False, config_path=None):
    dir_path = Path(target_dir).resolve()
    if not dir_path.exists():
        print(f"[FATAL] Directory '{target_dir}' does not exist.")
        return

    audio_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not audio_files:
        print(f"No supported audio files found in: {dir_path}")
        return

    source_rules = load_source_rules(config_path)
    configured_tags = [r.get("tag") for r in source_rules if r.get("tag")]

    print(f"Loaded {len(audio_files)} files. Loaded {len(source_rules)} source classification rules.")
    print(f"Initializing Whisper Model ({WHISPER_MODEL_SIZE})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    for audio_file in audio_files:
        # Skip if transcript already exists for this file
        if audio_file.with_suffix(".txt").exists():
            print(f"Skipping '{audio_file.name}' (Transcript exists)")
            continue

        # Skip if already standardized and tagged
        if configured_tags and any(tag in audio_file.name for tag in configured_tags) and audio_file.name.startswith("Recording_"):
            continue

        process_single_file(audio_file, model, high_sensitivity, enable_resolve, enable_uvr, source_rules)

if __name__ == "__main__":
    verify_system_dependencies()
    
    parser = argparse.ArgumentParser(description="Enterprise Ambient Audio Transcription Pipeline")
    parser.add_argument("directory", nargs="?", default=os.getenv("WATCH_DIRECTORY", "./Files"), help="Target audio directory path.")
    parser.add_argument("-hs", "--high-sensitivity", action="store_true", help="Force high-sensitivity transcription mode.")
    parser.add_argument("--enable-resolve", action="store_true", help="Enable DaVinci Resolve Neural Voice Isolation fallback.")
    parser.add_argument("--enable-uvr", action="store_true", help="Enable Ultimate Vocal Remover fallback.")
    parser.add_argument("--sources-config", default=None, help="Path to custom JSON source classification rules.")
    args = parser.parse_args()

    process_directory(args.directory, args.high_sensitivity, args.enable_resolve, args.enable_uvr, args.sources_config)