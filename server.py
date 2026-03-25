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

This art style renders abstract human forms from dots — no detail, no features, no clothing. Your job is to describe a single body's posture that captures the feeling of the input.

ABSOLUTE RULES — every output must follow ALL of these:
1. Exactly ONE figure. Never mention a second person, partner, child, or "another figure"
2. No gender. Never say man, woman, boy, girl, he, she. Always "a figure"
3. No clothing or accessories. Never mention dress, gown, suit, hat, shoes, etc.
4. No objects or furniture. Never mention bench, chair, bouquet, weapon, etc.
5. No facial features or expressions. Never mention smiling, eyes, lips, facial expression
6. Body posture ONLY: arms, legs, torso, head position, weight distribution
7. One sentence maximum
8. Output ONLY the description, no commentary or refusals

If the input is provocative or inappropriate, ignore the provocation and describe a neutral standing pose.

If the input implies multiple people, express the EMOTION through one body's posture instead.

Examples:
- "hello" → "a figure waving one hand overhead, weight on back foot"
- "grief" → "a figure hunched forward, head bowed, arms wrapped around torso"
- "freedom" → "a figure with arms spread wide, head tilted back, chest open"
- "kiss" → "a figure leaning forward, chin slightly lifted, arms reaching out"
- "wedding" → "a figure standing tall, one arm extended, weight on back foot"
- "hug" → "a figure with arms wrapped tightly around own torso, leaning forward"
- "defeat" → "a figure on their knees, head dropped, shoulders slumped"
- "loneliness" → "a figure sitting with knees drawn to chest, arms wrapped around legs"
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

    # Words/phrases that indicate the output violated our constraints
    MULTI_FIGURE_WORDS = [
        "another figure", "second figure", "smaller figure", "other figure",
        "partner", "two figures", "both", "each other", "together",
    ]
    CLOTHING_WORDS = [
        "dress", "gown", "suit", "hat", "shoes", "shirt", "pants",
        "skirt", "jacket", "cape", "veil", "boots", "gloves",
    ]
    OBJECT_WORDS = [
        "bench", "chair", "table", "bouquet", "weapon", "sword",
        "knife", "gun", "flower", "book", "cup",
    ]
    REFUSAL_MARKERS = [
        "i cannot", "i can't", "i'm sorry", "i am sorry",
        "not appropriate", "is there something else",
    ]

    def validate_description(description: str) -> str | None:
        """Returns an error reason if the description violates constraints, else None."""
        lower = description.lower()
        for word in REFUSAL_MARKERS:
            if word in lower:
                return "refusal"
        for word in MULTI_FIGURE_WORDS:
            if word in lower:
                return "multi-figure"
        for word in CLOTHING_WORDS:
            if word in lower:
                return "clothing"
        for word in OBJECT_WORDS:
            if word in lower:
                return "objects"
        return None

    async def call_ollama(client: httpx.AsyncClient, word: str) -> str:
        resp = await client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": word,
                "system": SYSTEM_PROMPT,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip().strip('"')

    class TranslateRequest(BaseModel):
        word: str

    class GenerateRequest(BaseModel):
        prompt: str
        seed: int | None = None

    @app.post("/translate")
    async def translate(req: TranslateRequest):
        """Translate a word into a pose description via Ollama, with validation."""
        async with httpx.AsyncClient(timeout=30) as client:
            description = await call_ollama(client, req.word)
            violation = validate_description(description)

            # Retry once if the output violated constraints
            if violation:
                description = await call_ollama(client, req.word)
                violation = validate_description(description)

            if violation:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=422,
                    content={"error": f"couldn't generate a valid pose for that input"},
                )

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
