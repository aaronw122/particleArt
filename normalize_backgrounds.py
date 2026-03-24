"""
Normalize near-white backgrounds to pure white.

Any pixel with all RGB channels above the threshold becomes (255, 255, 255).

Usage:
    uv run python normalize_backgrounds.py images/curated
    uv run python normalize_backgrounds.py images/raw_v4 --threshold 240
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def normalize_image(path: Path, threshold: int = 230) -> bool:
    """Normalize near-white pixels to pure white. Returns True if modified."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # Mask: pixels where ALL channels are above threshold
    near_white = np.all(arr > threshold, axis=2)
    if near_white.all():
        return False  # already pure white background

    arr[near_white] = 255
    Image.fromarray(arr).save(path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Normalize backgrounds to pure white")
    parser.add_argument("directory", help="Directory of images to process")
    parser.add_argument("--threshold", type=int, default=230,
                        help="Brightness threshold (0-255). Pixels with all channels above this become white. Default: 230")
    parser.add_argument("--dry-run", action="store_true", help="Report which files would be modified without changing them")
    args = parser.parse_args()

    directory = Path(args.directory)
    pngs = sorted(directory.glob("*.png"))
    print(f"Scanning {len(pngs)} images in {directory}/ (threshold={args.threshold})\n")

    modified = 0
    for p in pngs:
        img = Image.open(p).convert("RGB")
        arr = np.array(img)
        near_white = np.all(arr > args.threshold, axis=2)
        pct_near_white = near_white.sum() / near_white.size * 100
        pct_pure_white = np.all(arr == 255, axis=2).sum() / near_white.size * 100

        if pct_near_white > pct_pure_white + 0.5:  # has near-white pixels that aren't pure white
            if args.dry_run:
                print(f"  WOULD FIX: {p.name} ({pct_pure_white:.1f}% pure white, {pct_near_white:.1f}% near-white)")
            else:
                arr[near_white] = 255
                Image.fromarray(arr).save(p)
                print(f"  FIXED: {p.name} ({pct_pure_white:.1f}% -> {pct_near_white:.1f}% pure white)")
            modified += 1
        else:
            print(f"  OK: {p.name}")

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified}/{len(pngs)} images")


if __name__ == "__main__":
    main()
