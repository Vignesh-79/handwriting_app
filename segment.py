"""Split a photo/scan of handwritten text into line images (numpy + Pillow only).

Handles uneven lighting, ruled notebook paper (thin rules are filtered out) and slightly tilted pages.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def _ink_mask(gray: Image.Image):
    """Ink = pixels clearly darker than local background; thin ruled lines are removed."""
    w, h = gray.size
    scale = 1.0
    if max(w, h) > 1600:
        scale = 1600 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    radius = max(15, gray.size[0] // 30)
    bg = np.asarray(gray.filter(ImageFilter.BoxBlur(radius)), dtype=np.int16)
    g = np.asarray(gray, dtype=np.int16)
    mask = Image.fromarray((((bg - g) > 40) * 255).astype(np.uint8))
    # opening: erase strokes thinner than k px (ruled lines), keep pen strokes
    k = max(3, int(round(gray.size[0] / 350)) | 1)
    mask = mask.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(mask) > 0, scale


def _estimate_skew(mask: np.ndarray, max_deg=6.0, step=0.5) -> float:
    """Angle (deg) that makes text rows sharpest in the horizontal projection."""
    small = Image.fromarray((mask * 255).astype(np.uint8))
    f = 600 / max(small.size)
    if f < 1:
        small = small.resize((int(small.width * f), int(small.height * f)), Image.BILINEAR)
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + 1e-6, step):
        r = np.asarray(small.rotate(a, resample=Image.BILINEAR)) > 100
        score = np.var(r.sum(1).astype(float))
        if score > best:
            best, best_a = score, float(a)
    return best_a


def segment_lines(img: Image.Image, pad=12, min_height=10):
    """Returns (lines, deskewed_image); lines = [(box, crop)] in deskewed_image coordinates."""
    img = img.convert("RGB")
    ink, s = _ink_mask(img.convert("L"))
    angle = _estimate_skew(ink)
    if abs(angle) > 0.4:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        ink, s = _ink_mask(img.convert("L"))

    rows = np.convolve(ink.sum(1).astype(float), np.ones(5) / 5, mode="same")
    if rows.max() <= 0:
        return [], img
    on = rows > max(1.5, 0.06 * rows.max())

    runs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i]); start = None
    if start is not None:
        runs.append([start, len(on)])
    if not runs:
        return [], img

    heights = [b - a for a, b in runs]
    gap_limit = 0.35 * float(np.median(heights))
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < gap_limit:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    # a run far taller than the typical one = several touching lines: split at profile minima
    med = float(np.median([b - a for a, b in merged]))
    final = []
    for a, b in merged:
        n = int(round((b - a) / med)) if med > 0 else 1
        if n >= 2 and (b - a) > 1.6 * med:
            step = (b - a) / n
            cuts = [a]
            for k in range(1, n):
                lo, hi = int(a + (k - 0.35) * step), int(a + (k + 0.35) * step)
                cuts.append(lo + int(np.argmin(rows[lo:hi])))
            cuts.append(b)
            final += [[cuts[i], cuts[i + 1]] for i in range(n)]
        else:
            final.append([a, b])

    out = []
    W, H = img.size
    for a, b in final:
        if (b - a) < min_height * s:
            continue
        cols = np.where(ink[a:b].sum(0) > 0)[0]
        if len(cols) < 8:
            continue
        x0, x1 = cols.min() / s, (cols.max() + 1) / s
        py = max(pad, int(0.3 * (b - a) / s))   # extra vertical room for ascenders/descenders
        box = (max(0, int(x0 - pad)), max(0, int(a / s - py)),
               min(W, int(x1 + pad)), min(H, int(b / s + py)))
        out.append((box, img.crop(box)))
    return out, img


def draw_boxes(img: Image.Image, boxes):
    vis = img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)
    for i, b in enumerate(boxes, 1):
        d.rectangle(b, outline=(255, 0, 0), width=3)
        d.text((b[0] + 4, b[1] + 2), str(i), fill=(255, 0, 0))
    return vis
