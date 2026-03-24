"""
Modal REST endpoint for particle art generation.

Deploy:
    uv run modal deploy serve_modal.py

Test locally:
    uv run modal serve serve_modal.py

Generate:
    curl -X POST https://YOUR_APP--model-generate.modal.run \
      -H "Content-Type: application/json" \
      -d '{"prompt": "<s0><s1>, a figure dancing, white background"}' \
      --output image.png
"""

import modal

app = modal.App("prtkl-generate")

HF_CACHE = "/root/.cache/huggingface"
LORA_REPO = "aaronw122/prtkl-sdxl-lora"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"


def download_models():
    """Pre-download all weights into a fixed cache location at image build time."""
    import torch
    from diffusers import AutoencoderKL, DiffusionPipeline
    from huggingface_hub import hf_hub_download

    AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float16)
    DiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, variant="fp16"
    )
    hf_hub_download(LORA_REPO, "pytorch_lora_weights.safetensors")
    hf_hub_download(LORA_REPO, "results_emb.safetensors")
    print("All weights cached.")


# Bake weights into the image at build time
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate",
        "safetensors",
        "peft",
        "Pillow",
        "huggingface_hub",
        "fastapi[standard]",
    ])
    .env({"HF_HOME": HF_CACHE})
    .run_function(download_models)
)

NEGATIVE_PROMPT = (
    "photorealistic, detailed, shading, gradient, gray, color, dense, "
    "beige, tan, sepia, parchment, warm tones"
)


@app.cls(gpu="A10G", image=inference_image, timeout=300, scaledown_window=60)
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import AutoencoderKL, DiffusionPipeline
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float16)
        self.pipe = DiffusionPipeline.from_pretrained(
            BASE_MODEL,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # Load LoRA directly from HF repo ID
        self.pipe.load_lora_weights(
            LORA_REPO, weight_name="pytorch_lora_weights.safetensors"
        )

        # Load TI embeddings into both CLIP text encoders
        ti_path = hf_hub_download(LORA_REPO, "results_emb.safetensors")
        state_dict = load_file(ti_path)
        self.pipe.load_textual_inversion(
            state_dict["clip_l"],
            token=["<s0>", "<s1>"],
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        self.pipe.load_textual_inversion(
            state_dict["clip_g"],
            token=["<s0>", "<s1>"],
            text_encoder=self.pipe.text_encoder_2,
            tokenizer=self.pipe.tokenizer_2,
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict):
        import torch
        from io import BytesIO
        from fastapi.responses import Response
        from pydantic import BaseModel

        class GenerateRequest(BaseModel):
            prompt: str = "<s0><s1>, a figure standing, white background"
            negative_prompt: str = NEGATIVE_PROMPT
            num_inference_steps: int = 30
            guidance_scale: float = 5.0
            seed: int | None = None
            width: int = 1024
            height: int = 1024

        req = GenerateRequest(**request)
        steps = min(req.num_inference_steps, 50)

        generator = None
        if req.seed is not None:
            generator = torch.Generator("cuda").manual_seed(req.seed)

        image = self.pipe(
            req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        ).images[0]

        buf = BytesIO()
        image.save(buf, format="PNG")

        return Response(content=buf.getvalue(), media_type="image/png")
