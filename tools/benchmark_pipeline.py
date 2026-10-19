import os
import sys
import time
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Define the 4 test configurations
CONFIGS = [
    {
        "name": "1. Filter: afftdn | Threads: 2 | Workers: 1",
        "filter": "afftdn=nr=12:nf=-30,silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,loudnorm",
        "threads": 2,
        "workers": 1,
    },
    {
        "name": "2. Filter: high/lowpass | Threads: 2 | Workers: 1",
        "filter": "highpass=f=200,lowpass=f=3000,silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,loudnorm",
        "threads": 2,
        "workers": 1,
    },
    {
        "name": "3. Filter: high/lowpass | Threads: 3 | Workers: 2",
        "filter": "highpass=f=200,lowpass=f=3000,silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,loudnorm",
        "threads": 3,
        "workers": 2,
    },
    {
        "name": "4. Filter: afftdn | Threads: 3 | Workers: 2",
        "filter": "afftdn=nr=12:nf=-30,silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,loudnorm",
        "threads": 3,
        "workers": 2,
    },
]

def run_ffmpeg(input_path, output_path, filter_chain):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", filter_chain,
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ]
    start = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode('utf-8', errors='ignore')}")
    return elapsed

def run_whisper(audio_path, threads, workers):
    start = time.time()
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        num_workers=workers
    )
    segments, _ = model.transcribe(str(audio_path), beam_size=1, best_of=1, vad_filter=True)
    text = " ".join([s.text.strip() for s in segments])
    elapsed = time.time() - start
    return text, elapsed

def benchmark(sample_file):
    sample_path = Path(sample_file).resolve()
    if not sample_path.exists():
        print(f"Error: Sample file '{sample_file}' not found.")
        return

    print("==================================================")
    print(f"STARTING PIPELINE BENCHMARK")
    print(f"Target File: {sample_path.name}")
    print("==================================================\n")

    results = []

    for idx, cfg in enumerate(CONFIGS, start=1):
        print(f"--- Running Config {idx}/4: {cfg['name']} ---")
        temp_wav = sample_path.with_suffix(f".bench_{idx}.wav")
        
        # 1. Measure FFmpeg
        print(" -> Processing audio with FFmpeg...")
        ffmpeg_time = run_ffmpeg(sample_path, temp_wav, cfg["filter"])
        
        # 2. Measure Whisper
        print(" -> Transcribing with Whisper...")
        transcript, whisper_time = run_whisper(temp_wav, cfg["threads"], cfg["workers"])
        
        total_time = ffmpeg_time + whisper_time
        
        results.append({
            "name": cfg["name"],
            "ffmpeg_sec": round(ffmpeg_time, 2),
            "whisper_sec": round(whisper_time, 2),
            "total_sec": round(total_time, 2),
            "transcript": transcript
        })

        # Cleanup temp audio
        if temp_wav.exists():
            temp_wav.unlink()

        print(f"    [Completed in {total_time:.2f}s | FFmpeg: {ffmpeg_time:.2f}s | Whisper: {whisper_time:.2f}s]\n")

    # Output Benchmark Summary Table
    print("\n==================================================")
    print("BENCHMARK SUMMARY RESULTS")
    print("==================================================")
    print(f"{'Configuration':<45} | {'FFmpeg':<8} | {'Whisper':<8} | {'Total':<8}")
    print("-" * 77)
    for r in results:
        print(f"{r['name']:<45} | {r['ffmpeg_sec']:<6}s | {r['whisper_sec']:<6}s | {r['total_sec']:<6}s")
    print("==================================================\n")

    # Save Transcripts Comparison File
    comp_file = sample_path.with_name(f"{sample_path.stem}_benchmark_comparison.txt")
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write("=== BENCHMARK TRANSCRIPT COMPARISON ===\n\n")
        for r in results:
            f.write(f"--- {r['name']} (Total: {r['total_sec']}s) ---\n")
            f.write(f"{r['transcript'] if r['transcript'] else '[No speech detected]'}\n\n")

    print(f"Transcript outputs saved to: {comp_file.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        print("Usage: python benchmark_pipeline.py <path_to_audio_file>")
        sys.exit(1)

    benchmark(target)