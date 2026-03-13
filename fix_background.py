"""
Force white background on images with pastel/off-white backgrounds.

Usage:
    uv run python fix_background.py images/curated/tool_use_004_v0.png
    uv run python fix_background.py images/curated/*.png  # batch all
"""

import sys
from pathlib import Path
from PIL import Image

THRESHOLD = 200  # pixels brighter than this become pure white

for path in sys.argv[1:]:
    img = Image.open(path).convert("L")  # grayscale
    pixels = img.load()
    w, h = img.size
    changed = 0
    for y in range(h):
        for x in range(w):
            if pixels[x, y] > THRESHOLD:
                pixels[x, y] = 255
                changed += 1
    img.save(path)
    print(f"  {Path(path).name}: {changed} pixels -> white")
