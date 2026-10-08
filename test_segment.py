"""Quick self-test for line splitting (no model/torch needed): python test_segment.py"""
from PIL import Image, ImageDraw, ImageFont
from segment import segment_lines

img = Image.new("RGB", (900, 500), (235, 232, 225))
d = ImageDraw.Draw(img)
try:
    f = ImageFont.load_default(size=40)
except TypeError:
    f = ImageFont.load_default()
for i, t in enumerate(["Hello world this is line one", "second line of text goes here", "and a third line below"]):
    d.text((60, 60 + i * 140), t, fill=(20, 20, 60), font=f)
lines, _ = segment_lines(img)
print("lines found:", len(lines), [b for b, _ in lines])
assert len(lines) == 3, "expected 3 lines"
print("OK")
