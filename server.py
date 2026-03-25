"""
Local backend server for prtkl.
Translates words via Ollama, generates images via Modal endpoint.

Run:
    uv run python server.py
    uv run python server.py --ollama-url http://192.168.1.100:11434  # remote Ollama
"""

import argparse
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODAL_ENDPOINT = "https://youfoundaaron--prtkl-generate-model-generate.modal.run"

SYSTEM_PROMPT = """\
You translate a single word or short phrase into a physical pose description for a particle art generator.

Rules:
- Output ONLY the pose description, nothing else
- Single figure only, no multiple people
- Describe whole-body posture and gesture (arms, legs, torso, head position)
- No fine hand/finger details, no small objects
- No color, no style, no background details
- One sentence maximum
- Be specific and visual

Examples:
- "hello" → "a figure waving one hand overhead, weight on back foot"
- "grief" → "a figure hunched forward, head bowed, arms wrapped around torso"
- "freedom" → "a figure with arms spread wide, head tilted back, chest open"
- "balance" → "a figure standing on one leg, arms extended to the sides"
- "defeat" → "a figure on their knees, head dropped, shoulders slumped"
"""

NEGATIVE_PROMPT = (
    "photorealistic, detailed, shading, gradient, gray, color, dense, "
    "beige, tan, sepia, parchment, warm tones, nude, naked, nsfw, genitalia"
)


def create_app(ollama_url: str, ollama_model: str) -> FastAPI:
    app = FastAPI(title="prtkl backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class TranslateRequest(BaseModel):
        word: str

    class GenerateRequest(BaseModel):
        prompt: str
        seed: int | None = None

    @app.post("/translate")
    async def translate(req: TranslateRequest):
        """Translate a word into a pose description via Ollama."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": req.word,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            description = resp.json()["response"].strip().strip('"')

        prompt = f"<s0><s1>, {description}, white background"
        return {"prompt": prompt, "description": description}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Generate an image via the Modal endpoint."""
        body = {
            "prompt": req.prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
        }
        if req.seed is not None:
            body["seed"] = req.seed

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                MODAL_ENDPOINT,
                json=body,
            )
            resp.raise_for_status()

        return Response(content=resp.content, media_type="image/png")

    # Serve the frontend
    app.mount("/", StaticFiles(directory="web", html=True), name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="prtkl backend server")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.ollama_url, args.ollama_model)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
