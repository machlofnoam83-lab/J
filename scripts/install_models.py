"""
Install models for Adiel Junior
מוריד מודלים נחוצים
"""
import os
import sys
import requests
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "backend" / "data"
DATA.mkdir(parents=True, exist_ok=True)

def download_file(url, dest):
    print(f"Downloading {url} -> {dest}")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r {pct:.1f}%", end='', flush=True)
        print(f"\n✅ Saved {dest}")
        return True
    except Exception as e:
        print(f"\n❌ Failed {url}: {e}")
        return False

def install_whisper():
    print("\n=== Whisper Models (faster-whisper will auto-download) ===")
    print("Models will be downloaded on first run to backend/data/whisper_models/")
    print("Small model ~ 200MB, recommended for Hebrew")

    # Try pre-download via faster_whisper if available
    try:
        from faster_whisper import WhisperModel
        print("Pre-downloading tiny and small for quick start...")
        WhisperModel("tiny", device="cpu", download_root=str(DATA / "whisper_models"))
        WhisperModel("small", device="cpu", download_root=str(DATA / "whisper_models"))
        print("✅ Whisper models ready")
    except Exception as e:
        print(f"Whisper auto-download skipped: {e}, will download on first run")

def install_vosk_hebrew():
    print("\n=== Vosk Hebrew Model ===")
    vosk_dir = DATA / "vosk-model-small-he-0.22"
    if vosk_dir.exists():
        print(f"✅ Already exists: {vosk_dir}")
        return

    print("Vosk Hebrew model needed for offline wake word")
    print("Download manually from: https://alphacephei.com/vosk/models")
    print("Look for: vosk-model-small-he-0.22.zip")
    print(f"Extract to: {vosk_dir}")

    # Try direct download (may fail due to size)
    url = "https://alphacephei.com/vosk/models/vosk-model-small-he-0.22.zip"
    zip_path = DATA / "vosk-model-small-he-0.22.zip"
    
    print(f"\nAttempting auto-download from {url} (may be slow ~200MB)")
    try:
        import zipfile
        if download_file(url, zip_path):
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA)
            os.remove(zip_path)
            print(f"✅ Extracted to {vosk_dir}")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Please download manually")

def main():
    print("Adiel Junior - Model Installer")
    print(f"Data dir: {DATA}")
    
    install_whisper()
    install_vosk_hebrew()

    print("\n=== Done ===")
    print("Additional optional models:")
    print("- Tesseract OCR Hebrew: install Tesseract + heb.traineddata")
    print("- Ollama local LLM: https://ollama.com/ -> ollama pull llama3.1:8b")

if __name__ == "__main__":
    main()
