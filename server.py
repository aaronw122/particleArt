"""
Local backend server for prtkl.
Translates words via Ollama, generates images via Modal endpoint.

Run:
    uv run python server.py
    uv run python server.py --ollama-url http://192.168.1.100:11434  # remote Ollama
"""

import argparse
import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODAL_ENDPOINT = "https://youfoundaaron--prtkl-generate-model-generate.modal.run"

SYSTEM_PROMPT = """\
You translate a word into a one-sentence pose description for abstract particle art.

Imagine a featureless mannequin — no face, no clothing, no gender, no objects. Describe ONLY how its body segments are positioned: head, torso, arms, hands, legs, feet.

Rules:
1. Exactly ONE figure, always called "a figure"
2. Use only concrete body-position language — no emotions, mood, style, or intent words in the output
3. No clothing, accessories, facial features, objects, or furniture
4. Do not imply contact with objects, furniture, or another person
5. Avoid words an image model could misread as clothing: heels, collar, fist, sole, crown, palm, nails, laces, cuffs
6. Describe a FROZEN moment, not ongoing motion — "mid-sway leaning left" not "swaying"
7. Output ONLY the pose description, one sentence, no commentary

If the input refers to sex, nudity, or explicit anatomy, output: a figure standing with arms crossed over chest.
If the input implies multiple people, express the feeling through one body's posture.
Match the pose intensity to the word. Neutral or casual words (okay, hello, maybe, sure) get relaxed, natural poses. Only emotional words get dramatic poses.

Examples:
- "hello" → "a figure with one arm raised overhead, weight shifted to the back foot"
- "okay" → "a figure standing upright, one arm bent with hand on hip, head level"
- "grief" → "a figure hunched forward, head bowed, arms wrapped around torso"
- "freedom" → "a figure with arms spread wide, head tilted back, torso arched slightly"
- "kiss" → "a figure leaning forward, chin slightly lifted, arms reaching forward"
- "exhaustion" → "a figure on one side, one arm extended along the ground, legs bent"
- "hope" → "a figure on tiptoes, one arm stretched straight up, torso stretched upward"
- "shame" → "a figure turned sideways, shoulders curled inward, head tucked into chest"
- "power" → "a figure in a wide lunge, one arm thrust forward, torso twisted"
- "loneliness" → "a figure sitting with knees drawn to chest, arms wrapped around legs"
- "surrender" → "a figure kneeling, arms raised high above head, hands open"
- "defeat" → "a figure collapsed forward onto hands and knees, head hanging between arms"
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
        "companion", "holding someone",
    ]
    CLOTHING_WORDS = [
        "dress", "gown", "suit", "hat", "shoes", "shirt", "pants",
        "skirt", "jacket", "cape", "veil", "boots", "gloves",
        "robe", "cloak", "scarf", "belt", "crown", "mask", "armor",
        "uniform", "hood", "tie",
    ]
    OBJECT_WORDS = [
        "bench", "chair", "table", "bouquet", "weapon", "sword",
        "knife", "gun", "flower", "book", "cup", "staff", "stick",
        "rope", "ball", "flag", "candle", "shield", "mirror", "door",
        "wall", "stone", "box",
    ]
    FACIAL_WORDS = [
        "smiling", "smile", "frown", "frowning", "furrowed", "grimace",
        "grinning", "eyes", "lips", "mouth", "teeth", "brow", "eyebrows",
        "jaw", "peering", "gazing", "staring", "squinting", "winking",
        "expression", "look on", "face ",
    ]
    GENDER_WORDS = [
        " man ", " woman ", " boy ", " girl ", " he ", " she ",
        " his ", " her ", " male ", " female ",
    ]
    # Anatomically valid words that SDXL misinterprets as clothing/objects
    AMBIGUOUS_WORDS = [
        "heels", "heel", "laces", "collar", "cuff", "cuffs",
        "sole", "crown", "palm", "palms", "nails", "fist", "fists",
        "sleeve", "strap", "train", "temple", "nape", "bare",
    ]
    # Emotion/intent/style words that shouldn't appear in pose descriptions
    INTENT_WORDS = [
        "in thought", "in prayer", "in grief", "in pain", "in joy",
        "thoughtful", "pensive", "sorrowful", "joyful", "sensual",
        "elegant", "aggressive", "romantic", "graceful", "anxious",
        "contemplat", "meditating", "praying", "mourning", "celebrating",
    ]
    REFUSAL_MARKERS = [
        "i cannot", "i can't", "i'm sorry", "i am sorry",
        "not appropriate", "is there something else",
    ]

    def validate_description(description: str) -> str | None:
        """Returns an error reason if the description violates constraints, else None."""
        import re
        # Normalize: strip punctuation to spaces for reliable word-boundary matching
        lower = re.sub(r'[^\w\s]', ' ', description.lower())
        lower = f" {lower} "
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
        for word in FACIAL_WORDS:
            if word in lower:
                return "facial"
        for word in GENDER_WORDS:
            if word in lower:
                return "gender"
        for word in AMBIGUOUS_WORDS:
            if word in lower:
                return "ambiguous-visual"
        for word in INTENT_WORDS:
            if word in lower:
                return "intent-language"
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
            # Try up to 3 times to get a valid description
            description = None
            for attempt in range(3):
                description = await call_ollama(client, req.word)
                violation = validate_description(description)
                print(f"[translate] '{req.word}' attempt {attempt+1}: {description}")
                if violation:
                    print(f"[translate]   REJECTED ({violation})")
                else:
                    break
            else:
                # All attempts violated constraints — use fallback
                print(f"[translate] '{req.word}' → FALLBACK (all attempts failed)")
                description = "a figure standing with arms crossed over chest"

        prompt = f"<s0><s1>, {description}, white background"
        print(f"[translate] '{req.word}' → final prompt: {prompt}")
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


app_for_reload = create_app(
    os.environ.get("PRTKL_OLLAMA_URL", "http://localhost:11434"),
    os.environ.get("PRTKL_OLLAMA_MODEL", "llama3.1:8b"),
)


def main():
    parser = argparse.ArgumentParser(description="prtkl backend server")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    import uvicorn
    os.environ["PRTKL_OLLAMA_URL"] = args.ollama_url
    os.environ["PRTKL_OLLAMA_MODEL"] = args.ollama_model
    if args.no_reload:
        app = create_app(args.ollama_url, args.ollama_model)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        uvicorn.run(
            "server:app_for_reload",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[".", "web"],
        )


if __name__ == "__main__":
    main()
