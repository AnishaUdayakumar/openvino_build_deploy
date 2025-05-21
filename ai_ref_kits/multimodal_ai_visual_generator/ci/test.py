import subprocess
import time
import requests
import os
import sys
import platform
from pathlib import Path
from huggingface_hub import snapshot_download

# Add project root to path and import model converters
sys.path.append(str(Path(__file__).resolve().parent.parent))
from convert_and_optimize_llm import convert_chat_model
from convert_and_optimize_text2image import convert_image_model

# ----- Configuration -----
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
LLM_MODEL_TYPE = "qwen2-0.5B"  # Logical name used in app
LLM_HF_MODEL_ID = "OpenVINO/Qwen2-0.5B-Instruct-int4-ov"
LLM_LOCAL_DIR = MODEL_DIR / "qwen2-0.5B-INT4"

IMAGE_MODEL_TYPE = "flux.1-schnell"
IMAGE_HF_MODEL_ID = "OpenVINO/FLUX.1-schnell-int4-ov"
IMAGE_LOCAL_DIR = MODEL_DIR / "flux.1-schnell-INT4"

PRECISION = "int4"

# ----- Step 1: Download Pre-exported Models from Hugging Face if Missing -----
def download_model_if_missing(model_id: str, local_dir: Path):
    if not local_dir.exists():
        print(f"Downloading {model_id} to {local_dir} ...")
        result = snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    else:
        print(f"Model already exists at {local_dir}, skipping download.")

download_model_if_missing(LLM_HF_MODEL_ID, LLM_LOCAL_DIR)
download_model_if_missing(IMAGE_HF_MODEL_ID, IMAGE_LOCAL_DIR)

# ----- Step 2: Export Models if Needed (will be skipped if already present) -----
print("Verifying model export state...")
convert_chat_model(LLM_MODEL_TYPE, PRECISION, MODEL_DIR)
convert_image_model(IMAGE_MODEL_TYPE, PRECISION, MODEL_DIR)

# ----- Step 3: Launch FastAPI Backend -----
print("Launching FastAPI server...")

main_path = Path(__file__).resolve().parent.parent / "main.py"
env = os.environ.copy()
env["LLM_MODEL_TYPE"] = LLM_MODEL_TYPE
env["IMAGE_MODEL_TYPE"] = IMAGE_MODEL_TYPE
env["PRECISION"] = PRECISION

uvicorn_cmd = [
    sys.executable,
    "-m", "uvicorn",
    f"{main_path.stem}:app",
    "--app-dir", str(main_path.parent),
    "--host", "127.0.0.1",
    "--port", "8000"
]

# Add --factory only on macOS
if platform.system() == "Darwin":
    uvicorn_cmd.append("--factory")

process = subprocess.Popen(uvicorn_cmd, env=env)

try:
    # ----- Wait for FastAPI to become ready -----
    retries = 1000 if platform.system() == "Darwin" else 130
    time.sleep(10)  # Give some time for server startup and model loading

    for _ in range(retries):
        try:
            r = requests.get("http://localhost:8000/health", timeout=4)
            if r.status_code == 200:
                print("FastAPI is ready.")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError(f"FastAPI server did not start within {retries * 2} seconds.")

    # ----- Step 4: Test Story Prompt Generation -----
    print("Testing /generate_story_prompts endpoint...")
    response1 = requests.post(
        "http://localhost:8000/generate_story_prompts",
        json={"prompt": "A flying whale in space"}
    )
    assert response1.status_code == 200, f"Story generation failed: {response1.text}"
    scenes = response1.json()["scenes"]
    print("Scene generation passed. Example:", scenes)

    # ----- Step 5: Test Image Generation -----
    print("Testing /generate_images endpoint...")
    response2 = requests.post(
        "http://localhost:8000/generate_images",
        json={"prompt": scenes[0]}
    )
    assert response2.status_code == 200, f"Image generation failed: {response2.text}"
    image = response2.json()["image"]
    print("Image generation passed. Base64 (truncated):", image[:100])

finally:
    print("Shutting down FastAPI server...")
    process.terminate()
    process.wait()