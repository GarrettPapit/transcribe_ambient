import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    target_dir = os.getenv("WATCH_DIRECTORY", "./Files")
    print(f"=== Starting Ambient Transcription Pipeline Runner ===")
    print(f"Target Directory: {target_dir}")
    
    # Run diagnostic pre-flight
    subprocess.run(["python", "tools/verify_env.py"])
    
    # Execute main pipeline
    subprocess.run(["python", "src/cli.py", target_dir])

if __name__ == "__main__":
    main()