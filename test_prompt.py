"""Quick test of the new prompt on problem scenes."""

import asyncio
import base64
from pathlib import Path
from openai import AsyncOpenAI

REFERENCE_IMAGE = Path(__file__).parent / "images/reference/examplePerfect.png"

# Scenes that failed in the first batch (too dense/bold)
TEST_SCENES = [
    ("tool_use", "a figure swinging a hammer downward with both arms"),
    ("tool_use", "a figure standing and playing a violin, bow arm extended"),
    ("tool_use", "a figure playing guitar, one hand on neck, other strumming"),
    ("tool_use", "a figure standing at an easel, one arm extended holding a brush"),
]


async def generate(client, ref_bytes, scene_name, scene, output_path):
    prompt = (
        f"Sparse black particle flecks on a pure white background forming [{scene}]. "
        "Minimal, lots of negative space. No gray tones, no shading — "
        "just scattered black dots/flecks suggesting the form. Abstract, not realistic."
    )

    for attempt in range(3):
        try:
            result = await client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )

            image_bytes = base64.b64decode(result.data[0].b64_json)
            output_path.write_bytes(image_bytes)
            print(f"  -> {output_path.name}")
            return
        except Exception as e:
            print(f"  ERROR ({scene_name}, attempt {attempt+1}): {e}")
            if attempt < 2:
                await asyncio.sleep(3)

    print(f"  FAILED: {scene_name} after 3 attempts")


async def main():
    client = AsyncOpenAI()
    ref_bytes = REFERENCE_IMAGE.read_bytes()

    output_dir = Path("images/prompt_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing new prompt on {len(TEST_SCENES)} problem scenes (parallel)...\n")

    tasks = []
    for i, (category, scene) in enumerate(TEST_SCENES):
        path = output_dir / f"{category}_test_{i}.png"
        tasks.append(generate(client, ref_bytes, category, scene, path))

    await asyncio.gather(*tasks)
    print(f"\nDone! Check: open {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
