# Handwriting Recognition App (pretrained TrOCR + Gradio UI)

No dataset and no training needed. Models download automatically on first use (needs internet once).

## Run
1. Install **Python 3.10–3.12** (important: PyTorch often does not support the newest Python, e.g. 3.14).
2. Windows: double-click `run.bat`  |  Mac/Linux: `./run.sh`
   (or manually: `pip install -r requirements.txt` then `python app.py`)
3. Open http://127.0.0.1:7860

## Use
- Upload / take a photo / draw. Choose **Full page** (auto splits lines) or **Single line**.
- Best results: dark ink, light paper, even lighting, text roughly horizontal.
- First click is slow (model download + load). Use "Small" on CPU; "Base" is more accurate.

## Files
- `app.py` UI | `recognizer.py` TrOCR model | `segment.py` line splitting | `test_segment.py` self-test

## No GPU / install problems?
Run it free on Google Colab or Kaggle: upload this folder, `!pip install -r requirements.txt`,
`!python app.py` (Gradio prints a public link).

## Notes
TrOCR recognises English handwriting (trained on IAM). Heavily slanted pages, tables, or
overlapping lines will segment poorly; crop to one line and use "Single line" in that case.

## Docker
```
docker compose up --build
```
Then open http://localhost:7860. Models download on the first recognition and are cached in a
Docker volume. Without compose:
```
docker build -t handwriting-app .
docker run -p 7860:7860 -v hw_models:/models handwriting-app
```
This image is CPU-only. For an NVIDIA GPU, replace the torch install line in the Dockerfile with
`pip install torch` and run with `docker run --gpus all ...`.
