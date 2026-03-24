"""
Local inference server for particle art generation.
Runs on your home GPU (CUDA) or Mac (MPS).

Run:
    uv run python serve_local.py
    uv run python serve_local.py --device mps --dtype float32
    uv run python serve_local.py --local-weights ./final_weights

Generate:
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "<s0><s1>, a figure dancing, white background"}' \
      --output image.png
"""

import argparse
from inference.config import InferenceConfig
from inference.pipeline import ParticleArtPipeline


def main():
    parser = argparse.ArgumentParser(description="Local particle art inference server")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--local-weights", default=None, help="Path to local weights dir (has lora/ and ti/ subdirs)")
    parser.add_argument("--checkpoint", default="checkpoint-1800", help="Checkpoint dir name under lora/")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    config = InferenceConfig(device=args.device, dtype=args.dtype)

    if args.local_weights:
        config.local_lora_path = f"{args.local_weights}/lora/{args.checkpoint}"
        config.local_ti_path = f"{args.local_weights}/ti/results_emb.safetensors"

    print(f"Loading pipeline on {config.device} ({config.dtype})...")
    pipeline = ParticleArtPipeline(config).load()
    print("Pipeline loaded.")

    # Inline FastAPI app — avoids adding it as a project dependency
    from fastapi import FastAPI
    from fastapi.responses import Response
    from pydantic import BaseModel

    app = FastAPI(title="prtkl — Particle Art")

    class GenerateRequest(BaseModel):
        prompt: str = "<s0><s1>, a figure standing, white background"
        negative_prompt: str | None = None
        num_inference_steps: int = 30
        guidance_scale: float = 5.0
        seed: int | None = None
        width: int = 1024
        height: int = 1024

    @app.post("/generate")
    def generate(req: GenerateRequest):
        png_bytes = pipeline.generate_png_bytes(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            width=req.width,
            height=req.height,
        )
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": config.device}

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
