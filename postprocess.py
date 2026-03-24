"""
Post-process particle art: strip background, output transparent PNG.

Usage:
    uv run python postprocess.py input.png output.png
    uv run python postprocess.py input_dir/ output_dir/
    uv run python postprocess.py input_dir/ output_dir/ --compare  # side-by-side before/after
    uv run python postprocess.py input_dir/ output_dir/ --threshold 220
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def to_transparent(img_path: Path, threshold: int = 220) -> Image.Image:
    """Convert particle art to transparent PNG.

    Pixels darker than threshold become black (kept).
    Pixels at or above threshold become fully transparent.
    Gray intermediate pixels get proportional transparency for anti-aliasing.
    """
    img = Image.open(img_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)

    # RGBA output
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)

    # Below threshold: black pixel, opacity = inverse of brightness
    # 0 (pure black) → full opacity, threshold → transparent
    mask = arr < threshold
    opacity = np.clip((1.0 - arr / threshold) * 255, 0, 255).astype(np.uint8)

    rgba[mask, 0] = 0  # R
    rgba[mask, 1] = 0  # G
    rgba[mask, 2] = 0  # B
    rgba[mask, 3] = opacity[mask]  # A — anti-aliased

    return Image.fromarray(rgba, "RGBA")


def make_comparison(original_path: Path, processed: Image.Image) -> Image.Image:
    """Create side-by-side comparison: original | processed on white | processed on gray."""
    original = Image.open(original_path).convert("RGBA")
    size = original.size

    # Processed on white background
    on_white = Image.new("RGBA", size, (255, 255, 255, 255))
    on_white = Image.alpha_composite(on_white, processed)

    # Processed on light gray background (to show if edges are clean)
    on_gray = Image.new("RGBA", size, (200, 200, 200, 255))
    on_gray = Image.alpha_composite(on_gray, processed)

    # Side by side
    comparison = Image.new("RGB", (size[0] * 3, size[1]), (255, 255, 255))
    comparison.paste(original.convert("RGB"), (0, 0))
    comparison.paste(on_white.convert("RGB"), (size[0], 0))
    comparison.paste(on_gray.convert("RGB"), (size[0] * 2, 0))

    return comparison


def process_file(input_path: Path, output_path: Path, threshold: int, compare: bool):
    processed = to_transparent(input_path, threshold)

    if compare:
        comp = make_comparison(input_path, processed)
        comp.save(output_path)
        print(f"  comparison: {input_path.name}")
    else:
        processed.save(output_path)
        print(f"  processed: {input_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Post-process particle art to transparent PNG")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output image or directory")
    parser.add_argument("--threshold", type=int, default=220,
                        help="Brightness threshold (0-255). Pixels above this become transparent. Default: 220")
    parser.add_argument("--compare", action="store_true",
                        help="Generate side-by-side comparison (original | on white | on gray)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        pngs = sorted(input_path.glob("*.png"))
        print(f"Processing {len(pngs)} images (threshold={args.threshold})\n")
        for p in pngs:
            out = output_path / p.name
            process_file(p, out, args.threshold, args.compare)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_file(input_path, output_path, args.threshold, args.compare)

    print(f"\nDone! Output in {output_path}")


if __name__ == "__main__":
    main()
