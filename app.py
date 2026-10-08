"""Run: python app.py   ->  open http://127.0.0.1:7860"""
import os
import time
import numpy as np
import gradio as gr
from PIL import Image

from segment import segment_lines, draw_boxes
from recognizer import recognize_images, warmup, MODELS, DEFAULT_MODEL, NTHREADS, DEVICE

MODE_LINE = "Single line"
MODE_PAGE = "Full page / paragraph (auto-split lines)"


def to_pil(img):
    if img is None:
        return None
    if isinstance(img, dict):  # Sketchpad / ImageEditor output
        img = img.get("composite")
        if img is None:
            return None
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> composite on white
            a = img[..., 3:] / 255.0
            img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        img = Image.fromarray(img.astype(np.uint8))
    return img.convert("RGB")


def run(img, mode, model_name, beams, progress=gr.Progress()):
    pil = to_pil(img)
    if pil is None:
        raise gr.Error("Please provide an image first.")
    t0 = time.time()
    progress(0.1, desc="Preparing image…")
    if mode == MODE_LINE:
        crops = [pil]
    else:
        found, pil = segment_lines(pil)   # pil is now the deskewed page
        if not found:
            raise gr.Error("No text found. Try a clearer photo with dark ink on light paper.")
        crops = [c for _, c in found]
    t1 = time.time()
    progress(0.3, desc="Recognising…")
    texts = recognize_images(crops, model_name, num_beams=int(beams))
    t2 = time.time()
    progress(1.0)
    status = (f"Splitting: {t1 - t0:.1f}s | Recognition: {t2 - t1:.1f}s | {len(crops)} line(s) | "
              f"device={DEVICE}, threads={NTHREADS}, beams={int(beams)}")
    print(status, flush=True)
    return "\n".join(texts), status


with gr.Blocks(title="Handwriting Recognition") as demo:
    gr.Markdown("# ✍️ Handwriting Recognition\nUpload, photograph, or draw handwriting and get the text back.")
    with gr.Row():
        mode = gr.Radio([MODE_LINE, MODE_PAGE], value=MODE_PAGE, label="Input type")
        model_name = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
        beams = gr.Slider(1, 8, value=1, step=1, label="Beam search (higher = slower, slightly better)")
    with gr.Tabs():
        with gr.Tab("Upload / Camera"):
            inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"], label="Handwriting image")
            b1 = gr.Button("Recognise", variant="primary")
        with gr.Tab("Draw"):
            pad = gr.Sketchpad(label="Write here (one line works best)", brush=gr.Brush(default_size=5, colors=["#000000"]))
            b2 = gr.Button("Recognise drawing", variant="primary")
    try:
        out = gr.Textbox(label="Recognised text", lines=8, show_copy_button=True)
    except TypeError:  # newer Gradio versions
        out = gr.Textbox(label="Recognised text", lines=8)
    status_box = gr.Textbox(label="Timing", interactive=False)
    for btn, src in ((b1, inp), (b2, pad)):
        btn.click(run, [src, mode, model_name, beams], [out, status_box])

if __name__ == "__main__":
    print("Loading model...")
    warmup()   # load + dry run now so the first click is fast
    print("Ready.")
    demo.queue().launch(share=os.environ.get("SHARE") == "1")
