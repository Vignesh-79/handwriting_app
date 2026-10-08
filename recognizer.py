"""Fast handwriting recognition with Microsoft TrOCR (pretrained, downloads on first use)."""
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODELS = {
    "Small (fast, ~250 MB)": "microsoft/trocr-small-handwritten",
    "Base (more accurate, ~1.3 GB)": "microsoft/trocr-base-handwritten",
}
DEFAULT_MODEL = "Small (fast, ~250 MB)"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _usable_cpus():
    """CPUs this process may really use (respects Docker --cpus and OMP_NUM_THREADS)."""
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit():
        return int(env)
    try:  # cgroup v2 quota
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


NTHREADS = _usable_cpus()
torch.set_num_threads(NTHREADS)
_cache = {}


def load(name=DEFAULT_MODEL):
    repo = MODELS[name]
    if repo not in _cache:
        proc = TrOCRProcessor.from_pretrained(repo, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(repo).eval()
        if DEVICE == "cpu":
            try:  # int8 weights: ~2x faster on CPU, tiny accuracy loss
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception:
                pass
        else:
            model = model.half().to(DEVICE)
        _cache[repo] = (proc, model)
    return _cache[repo]


def warmup(name=DEFAULT_MODEL):
    """Load the model and run one dummy pass so the first real click is fast."""
    from PIL import Image
    recognize_images([Image.new("RGB", (200, 50), "white")], name, num_beams=1)


@torch.inference_mode()
def recognize_images(images, model_name=DEFAULT_MODEL, num_beams=1):
    proc, model = load(model_name)
    imgs = [im.convert("RGB") for im in images]
    pv = proc(images=imgs, return_tensors="pt").pixel_values        # ALL lines in ONE batch
    pv = pv.to(DEVICE, dtype=torch.float16) if DEVICE == "cuda" else pv
    ids = model.generate(pv, max_new_tokens=64, num_beams=int(num_beams), no_repeat_ngram_size=3)
    return [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
