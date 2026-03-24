"""Inference configuration — shared across Modal and local deployment."""

from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    # Model source
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    lora_repo: str = "aaronw122/prtkl-sdxl-lora"
    lora_weight_name: str = "pytorch_lora_weights.safetensors"
    ti_filename: str = "results_emb.safetensors"

    # Or load from local paths instead of HF
    local_lora_path: str | None = None  # e.g. "./final_weights/lora/checkpoint-1800"
    local_ti_path: str | None = None  # e.g. "./final_weights/ti/results_emb.safetensors"

    # Device
    device: str = "cuda"  # "cuda", "mps", "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # Generation defaults
    default_steps: int = 30
    default_guidance_scale: float = 5.0
    default_negative_prompt: str = (
        "photorealistic, detailed, shading, gradient, gray, color, dense, "
        "beige, tan, sepia, parchment, warm tones"
    )
    max_steps: int = 50
    default_width: int = 1024
    default_height: int = 1024

    # TI tokens
    ti_tokens: list[str] = field(default_factory=lambda: ["<s0>", "<s1>"])
