"""Device-agnostic SDXL + LoRA + TI pipeline."""

import torch
from io import BytesIO

from .config import InferenceConfig

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ParticleArtPipeline:
    """Loads SDXL with particle art LoRA + TI. Call generate() to produce images."""

    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        self.pipe = None

    def load(self):
        """Load the full pipeline. Call once, then generate many times."""
        from diffusers import AutoencoderKL, DiffusionPipeline
        from safetensors.torch import load_file

        cfg = self.config
        dtype = DTYPE_MAP[cfg.dtype]

        vae = AutoencoderKL.from_pretrained(cfg.vae_model, torch_dtype=dtype)

        variant = "fp16" if cfg.dtype == "float16" else None
        self.pipe = DiffusionPipeline.from_pretrained(
            cfg.base_model,
            vae=vae,
            torch_dtype=dtype,
            variant=variant,
        ).to(cfg.device)

        # Load LoRA
        if cfg.local_lora_path:
            self.pipe.load_lora_weights(
                cfg.local_lora_path, weight_name=cfg.lora_weight_name
            )
            ti_data = self._load_local_ti(cfg, load_file)
        else:
            self.pipe.load_lora_weights(
                cfg.lora_repo, weight_name=cfg.lora_weight_name
            )
            ti_data = self._load_hf_ti(cfg, load_file)

        # Load TI embeddings into both CLIP text encoders
        self.pipe.load_textual_inversion(
            ti_data["clip_l"],
            token=cfg.ti_tokens,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        self.pipe.load_textual_inversion(
            ti_data["clip_g"],
            token=cfg.ti_tokens,
            text_encoder=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer_2,
        )

        return self

    def _load_local_ti(self, cfg, load_file):
        path = cfg.local_ti_path or f"{cfg.local_lora_path}/{cfg.ti_filename}"
        return load_file(path)

    def _load_hf_ti(self, cfg, load_file):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(cfg.lora_repo, cfg.ti_filename)
        return load_file(path)

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """Generate a single image. Returns a PIL Image."""
        cfg = self.config

        steps = min(num_inference_steps or cfg.default_steps, cfg.max_steps)
        guidance = guidance_scale if guidance_scale is not None else cfg.default_guidance_scale
        neg = negative_prompt if negative_prompt is not None else cfg.default_negative_prompt
        w = width or cfg.default_width
        h = height or cfg.default_height

        generator = None
        if seed is not None:
            generator = torch.Generator(cfg.device).manual_seed(seed)

        image = self.pipe(
            prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=w,
            height=h,
            generator=generator,
        ).images[0]

        return image

    def generate_png_bytes(self, **kwargs) -> bytes:
        """Generate an image and return PNG bytes."""
        image = self.generate(**kwargs)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
