"""
Generate particle art training images via OpenAI's gpt-image-1 API.

Usage:
    # Test run (10 images, ~$0.50)
    uv run python generate_images.py --test

    # Full batch (all scenes, ~$4-16)
    uv run python generate_images.py

    # Custom count per scene
    uv run python generate_images.py --variations 3

Requires OPENAI_API_KEY environment variable.
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI

# --- Aesthetic constraints (baked into every prompt) ---
STYLE = (
    "Black ink flecks on white paper. Extremely sparse — only 50 to 100 small dots "
    "suggesting the form. Mostly white space. No solid fills, no gray tones, no "
    "gradients, no shading. Do NOT draw realistic figures. Do NOT add background "
    "elements. Minimalist particle art, abstract, not realistic. "
    "No face details, no clothing details."
)

# --- Scene descriptions ---
# These are the visual compositions we want. Each will be combined with STYLE.
# Organized by category to ensure diversity.
SCENES = {
    "single_figure_emotion": [
        "a single human figure standing with arms spread wide, head tilted back",
        "a figure hunched forward, head bowed, arms wrapped around torso",
        "a figure leaping upward with one arm reaching toward the sky",
        "a figure sitting cross-legged, hands resting on knees, calm posture",
        "a figure curled into a ball on the ground",
        "a figure walking forward with confident stride",
        "a figure kneeling with hands clasped together",
        "a figure spinning with arms out, caught mid-rotation",
        "a figure standing still with one hand over their heart",
        "a figure reaching forward with both hands, palms open",
    ],
    "two_figures_connection": [
        "two figures embracing each other tightly",
        "two figures holding hands, standing side by side",
        "two figures facing each other with hands almost touching",
        "one figure reaching toward another who is turning away",
        "two figures dancing together, one dipping the other",
        "two figures sitting back to back",
        "one figure lifting another figure up",
        "two figures walking together, leaning into each other",
        "two figures bowing toward each other",
        "one figure extending a hand to help another figure stand up",
    ],
    "gesture_and_action": [
        "a figure waving one hand outward, open posture",
        "a figure with both arms raised in celebration",
        "a figure pointing toward something in the distance",
        "a figure shielding their eyes and looking into the distance",
        "a figure mid-stride, running",
        "a figure bending down to pick something up",
        "a figure stretching upward on tiptoes",
        "a figure making a heart shape with their hands above their head",
        "two figures high-fiving each other",
        "a figure conducting with both arms raised, gesturing broadly",
    ],
    "abstract_symbolic": [
        "a solitary figure standing at the edge of empty space",
        "a figure dissolving from feet upward into scattered particles",
        "a figure emerging from a cloud of scattered dots",
        "two figures merging into one shared form",
        "a figure whose outstretched hand trails off into scattered flecks",
        "a small figure looking up at a much larger figure",
        "a circle of four figures holding hands",
        "a figure carrying another figure on their back",
        "a figure mid-fall, suspended in air",
        "two figures separated by empty space, reaching toward each other",
    ],
}


def generate_image(client: OpenAI, scene: str, output_path: Path) -> bool:
    """Generate one image and save it. Returns True on success."""
    prompt = f"{scene}. {STYLE}"

    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="medium",
            n=1,
        )

        # gpt-image-1 returns base64
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        output_path.write_bytes(image_bytes)
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate particle art training images")
    parser.add_argument("--test", action="store_true", help="Test run: 10 images only")
    parser.add_argument("--variations", type=int, default=2, help="Variations per scene (default: 2)")
    parser.add_argument("--output", type=str, default="images/raw", help="Output directory")
    args = parser.parse_args()

    client = OpenAI()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all scenes
    all_scenes = []
    for category, scenes in SCENES.items():
        for scene in scenes:
            all_scenes.append((category, scene))

    if args.test:
        # Test mode: 1 from each category + a few extras = ~10 images
        test_scenes = []
        for category, scenes in SCENES.items():
            test_scenes.append((category, scenes[0]))
            test_scenes.append((category, scenes[1]))
            if len(test_scenes) >= 8:
                break
        all_scenes = test_scenes
        variations = 1
        print(f"TEST MODE: generating {len(all_scenes)} images")
    else:
        variations = args.variations
        print(f"FULL MODE: generating {len(all_scenes) * variations} images ({len(all_scenes)} scenes x {variations} variations)")

    # Track what we generate for captions
    manifest = []
    count = 0
    errors = 0

    for category, scene in all_scenes:
        for v in range(variations):
            count += 1
            filename = f"{category}_{count:03d}_v{v}.png"
            filepath = output_dir / filename

            # Skip if already exists (resume-friendly)
            if filepath.exists():
                print(f"  [{count}] SKIP (exists): {filename}")
                manifest.append({
                    "file": filename,
                    "category": category,
                    "scene": scene,
                })
                continue

            print(f"  [{count}] Generating: {scene[:60]}...")
            success = generate_image(client, scene, filepath)

            if success:
                manifest.append({
                    "file": filename,
                    "category": category,
                    "scene": scene,
                })
                print(f"         -> {filename}")
            else:
                errors += 1

            # Rate limit: pause between API calls
            time.sleep(2)

    # Save manifest (useful for captioning later)
    manifest_path = output_dir / "manifest.json"
    # Merge with existing manifest if present
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_files = {e["file"] for e in existing}
        for entry in manifest:
            if entry["file"] not in existing_files:
                existing.append(entry)
        manifest = existing

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nDone! {len(manifest)} total images in {output_dir}/ ({errors} errors)")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors: {errors} (re-run to retry — existing images are skipped)")


if __name__ == "__main__":
    main()
