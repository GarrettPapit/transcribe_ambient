import os
import sys
import shutil
import subprocess

# Set Ollama host variable BEFORE importing/calling ollama
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"

def check_system():
    print("=== Pipeline Environment Diagnostic ===")
    
    # 1. Python Executable Path
    print(f"[+] Python Location: {sys.executable}")
    
    # 2. FFmpeg Verification
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[+] FFmpeg: Found at {ffmpeg_path}")
    else:
        print("[-] FFmpeg: NOT FOUND in PATH! (Required for silence stripping)")

    # 3. Faster-Whisper Import Check
    try:
        import faster_whisper
        print("[+] faster-whisper: Installed successfully")
    except ImportError:
        print("[-] faster-whisper: NOT INSTALLED")

    # 4. TinyTag Import Check
    try:
        import tinytag
        print("[+] tinytag: Installed successfully")
    except ImportError:
        print("[-] tinytag: NOT INSTALLED")

    # 5. Ollama Connectivity Check
    try:
        import ollama
        # Use Client explicitly with the local address
        client = ollama.Client(host="http://127.0.0.1:11434")
        models = client.list()
        print("[+] Ollama Engine: Service active and responding")
    except Exception as e:
        print(f"[-] Ollama Engine: Connection failed ({e})")

if __name__ == "__main__":
    check_system()