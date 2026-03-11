# --- CELL 1: Install ---
!pip install diffusers transformers accelerate safetensors torch

# --- CELL 2: Load model ---
import torch
from diffusers import StableDiffusionXLPipeline
from IPython.display import display
from pathlib import Path

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# --- CELL 3: Generate test images ---
STYLE = (
    "Sparse black particle flecks on a pure white background. "
    "Minimal, lots of negative space. No gray tones, no gradients, no shading. "
    "Just scattered black dots suggesting the form. "
    "Abstract, not realistic. High contrast."
)

NEGATIVE = (
    "photorealistic, detailed, shading, gradient, gray, color, "
    "face details, clothing, background elements, dense, filled in"
)

scenes = [
    "a human figure with arms spread wide, head tilted back",
    "two figures embracing each other",
    "a figure hunched forward, head bowed, arms wrapped around torso",
    "a figure making a heart shape with their hands above their head",
    "two figures reaching toward each other across empty space",
]

output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

for i, scene in enumerate(scenes):
    prompt = f"{scene}. {STYLE}"
    print(f"\n[{i+1}/{len(scenes)}] {scene}")

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024,
    ).images[0]

    path = output_dir / f"test_{i+1:03d}.png"
    image.save(path)
    display(image)
