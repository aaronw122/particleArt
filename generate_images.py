"""
Generate particle art training images via OpenAI's gpt-image-1 API.

Usage:
    # Test run (8 images)
    uv run python generate_images.py --test

    # Full batch (all scenes x 2 variations)
    uv run python generate_images.py

    # Custom variations and parallelism
    uv run python generate_images.py --variations 3 --parallel 10

    # Generate into a different output directory
    uv run python generate_images.py --output images/v2

Requires OPENAI_API_KEY environment variable.
"""

import argparse
import asyncio
import base64
import json
from pathlib import Path

from openai import AsyncOpenAI

# Reference image for style consistency
REFERENCE_IMAGE = Path(__file__).parent / "images/reference/examplePerfect.png"

# --- Scene descriptions ---
# Organized by category to ensure diversity.
SCENES = {
    "single_figure_emotion": [
        "a single human figure standing with arms spread wide, head tilted back",
        "a figure hunched forward, head bowed, arms wrapped around torso",
        "a figure leaping upward with one arm reaching toward the sky",
        "a figure sitting cross-legged, hands resting on knees, calm posture",
        "a figure curled into a ball on the ground",
        "a figure kneeling with hands clasped together",
        "a figure standing still with one hand over their heart",
        "a figure reaching forward with both hands, palms open",
        "a figure recoiling backward, arms raised defensively",
        "a figure bent forward with hands on knees, catching their breath",
        "a figure standing and looking upward, head tilted back, arms at sides",
        "a figure lying flat on their back, arms at their sides",
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
        "a small figure looking up at a much larger figure",
        "a figure carrying another figure on their back",
    ],
    "movement": [
        "a figure walking forward with confident stride",
        "a figure in full sprint, body leaning forward, legs wide apart",
        "a figure mid-stride, running",
        "a figure spinning with arms out, caught mid-rotation",
        "a figure swimming forward, arms mid-stroke, body horizontal",
        "a figure climbing upward, one arm reaching high, legs bent",
        "a figure floating horizontally with arms spread, as if flying",
        "a figure mid-fall, suspended in air",
        "a figure stretching upward on tiptoes",
    ],
    "tool_use": [
        "a figure seated at a desk, hands positioned on a keyboard",
        "a figure sitting and holding an open book in their lap",
        "a figure standing and playing a violin, bow arm extended",
        "a figure standing at a counter, stirring a pot with one hand",
        "a figure swinging a hammer downward with both arms",
        "a figure standing at an easel, one arm extended holding a brush",
        "a figure seated and writing on a surface, head bent down",
        "a figure holding a phone to their ear with one hand",
        "a figure playing guitar, one hand on neck, other strumming",
    ],
    "daily_life": [
        "a figure lying on their side, curled in a sleeping position",
        "a figure seated at a table, one hand raising food to their mouth",
        "a figure standing with arms overhead in a full-body stretch",
        "a figure seated cross-legged with straight spine, meditating",
        "a figure walking and carrying heavy bags in both hands, leaning slightly",
        "a figure crouching down to tie a shoe",
        "a figure standing under a shower, head tilted back",
        "a figure sitting on the ground, hugging their knees",
    ],
    "gesture_and_action": [
        "a figure waving one hand outward, open posture",
        "a figure with both arms raised in celebration",
        "a figure pointing toward something in the distance",
        "a figure shielding their eyes and looking into the distance",
        "a figure bending down to pick something up",
        "two figures high-fiving each other",
    ],
    "abstract_symbolic": [
        "a solitary figure standing at the edge of empty space",
        "a figure whose outstretched hand trails off into scattered flecks",
        "two figures separated by empty space, reaching toward each other",
        "three figures walking in a line, one behind the other",
    ],
    "sports_and_athletics": [
        "a figure throwing a ball overhand, arm fully extended behind them",
        "a figure mid-kick, one leg swung forward, arms out for balance",
        "a figure in a yoga tree pose, one foot pressed against the opposite thigh, arms overhead",
        "a figure lifting a heavy weight overhead, legs wide, body bracing",
    ],
    "social_interaction": [
        "two figures facing each other with arms gesturing animatedly, mid-argument",
        "one figure leaning close to another figure's ear, whispering",
        "two figures shaking hands firmly",
        "a figure waving goodbye to another figure in the distance",
    ],
    "play_and_childhood": [
        "a small child figure jumping rope, knees high, rope arcing overhead",
        "a small child figure crouching on a skateboard, arms out for balance",
        "an adult figure holding a small child's hand, both walking together",
        "a figure pushing a small child on a swing, arms extended forward",
    ],
    "communication": [
        "a figure cupping both hands around their mouth, shouting into the distance",
        "a figure standing with head bowed and palms pressed together in prayer",
        "a figure singing with mouth open wide, one arm extended outward dramatically",
    ],
    "struggle_and_effort": [
        "a figure leaning forward pushing against a large heavy object, legs bracing",
        "a figure pulling a rope, body leaning backward, arms taut",
        "a figure clinging to a wall, one arm reaching upward for a handhold",
    ],
    "life_stages_and_context": [
        "a figure standing in profile with a rounded pregnant belly, one hand resting on it",
        "a hunched elderly figure walking slowly with a cane, body bent forward",
        "a figure standing before a group of smaller seated figures, one arm gesturing",
        "two figures circling each other with fists raised, bodies tense",
        "a figure leaning hard into wind, one arm shielding their face",
        "a figure standing with face tilted up and arms slightly out, catching rain",
        "a figure seated gripping a steering wheel, body leaning forward",
        "a figure standing with arms extended wearing a VR headset, reaching into empty space",
    ],
}


async def generate_image(
    client: AsyncOpenAI, scene: str, output_path: Path,
    ref_bytes: bytes, semaphore: asyncio.Semaphore
) -> bool:
    """Generate one image using reference image for style consistency."""
    prompt = (
        "Match the visual style of the reference image. "
        f"Sparse black particle flecks on a pure white background forming "
        f"an abstract human figure {scene}. Minimal, lots of negative space. "
        "No gray tones, no shading — just scattered black dots/flecks suggesting "
        "the form. Abstract, not realistic."
    )

    async with semaphore:
        for attempt in range(3):
            try:
                result = await client.images.edit(
                    model="gpt-image-1",
                    image=REFERENCE_IMAGE.open("rb"),
                    prompt=prompt,
                    size="1024x1024",
                    n=1,
                )

                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)
                output_path.write_bytes(image_bytes)
                return True

            except Exception as e:
                print(f"  ERROR ({output_path.name}, attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(15)

        return False


async def main():
    parser = argparse.ArgumentParser(description="Generate particle art training images")
    parser.add_argument("--test", action="store_true", help="Test run: 8 images only")
    parser.add_argument("--variations", type=int, default=2, help="Variations per scene (default: 2)")
    parser.add_argument("--output", type=str, default="images/raw", help="Output directory")
    parser.add_argument("--parallel", type=int, default=5, help="Max parallel API calls (default: 5)")
    parser.add_argument("--skip-curated", type=str, default=None, help="Path to curated dir — skip scenes already curated")
    args = parser.parse_args()

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.parallel)

    # Load reference image bytes once
    if not REFERENCE_IMAGE.exists():
        print(f"ERROR: Reference image not found at {REFERENCE_IMAGE}")
        print("Add your reference image to images/reference/examplePerfect.png")
        return
    ref_bytes = REFERENCE_IMAGE.read_bytes()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build set of curated scenes to skip
    curated_scenes = set()
    if args.skip_curated:
        curated_dir = Path(args.skip_curated)
        manifest_path_check = Path("images/raw") / "manifest.json"
        if manifest_path_check.exists():
            manifest_data = json.loads(manifest_path_check.read_text())
            file_to_scene = {e["file"]: (e["category"], e["scene"]) for e in manifest_data}
            for f in curated_dir.glob("*.png"):
                if f.name in file_to_scene:
                    curated_scenes.add(file_to_scene[f.name])
        print(f"Skipping {len(curated_scenes)} already-curated scenes\n")

    # Flatten all scenes
    all_scenes = []
    for category, scenes in SCENES.items():
        for scene in scenes:
            if (category, scene) not in curated_scenes:
                all_scenes.append((category, scene))

    if args.test:
        test_scenes = []
        for category, scenes in SCENES.items():
            for s in scenes[:2]:  # up to 2 per category, safe if only 1
                test_scenes.append((category, s))
            if len(test_scenes) >= 8:
                break
        all_scenes = test_scenes
        variations = 1
        print(f"TEST MODE: generating {len(all_scenes)} images ({args.parallel} parallel)")
    else:
        variations = args.variations
        total = len(all_scenes) * variations
        print(f"FULL MODE: generating {total} images ({len(all_scenes)} scenes x {variations} variations, {args.parallel} parallel)")

    # Build task list, skipping existing files
    tasks = []
    manifest_entries = []
    count_by_category = {}

    for category, scene in all_scenes:
        count_by_category[category] = count_by_category.get(category, 0) + 1
        num = count_by_category[category]
        for v in range(variations):
            filename = f"{category}_{num:03d}_v{v}.png"
            filepath = output_dir / filename
            entry = {"file": filename, "category": category, "scene": scene}

            if filepath.exists():
                print(f"  SKIP (exists): {filename}")
                manifest_entries.append(entry)
                continue

            manifest_entries.append(entry)
            tasks.append((scene, filepath, entry))

    print(f"\n  {len(tasks)} to generate, {len(manifest_entries) - len(tasks)} skipped\n")

    # Run all tasks with bounded parallelism
    completed = 0
    errors = 0

    for scene, filepath, entry in tasks:
        success = await generate_image(client, scene, filepath, ref_bytes, semaphore)
        if success:
            completed += 1
            print(f"  [{completed}/{len(tasks)}] -> {filepath.name}")
        else:
            errors += 1
            manifest_entries.remove(entry)
        await asyncio.sleep(13)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_files = {e["file"] for e in existing}
        for entry in manifest_entries:
            if entry["file"] not in existing_files:
                existing.append(entry)
        manifest_entries = existing

    manifest_path.write_text(json.dumps(manifest_entries, indent=2))

    print(f"\nDone! {completed} new images, {errors} errors")
    print(f"Total in manifest: {len(manifest_entries)}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
