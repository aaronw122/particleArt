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

This art style renders abstract human forms from dots — no detail, no features, no clothing. Your job is to describe a single body's STATIC posture that captures the feeling of the input.

ABSOLUTE RULES — every output must follow ALL of these:
1. Exactly ONE figure. Never mention a second person, partner, child, or "another figure"
2. No gender. Never say man, woman, boy, girl, he, she. Always "a figure"
3. No clothing or accessories. Never mention dress, gown, suit, hat, shoes, etc.
4. No objects or furniture. Never mention bench, chair, bouquet, weapon, etc.
5. No facial features or expressions. Never mention smiling, furrowed, eyes, lips, peering, gazing, expression
6. Body posture ONLY: arms, legs, torso, head position, weight distribution
7. Describe a FROZEN moment, not continuous motion. Not "swaying" but "mid-sway leaning left"
8. One sentence maximum
9. Output ONLY the description, no commentary or refusals

CRITICAL: Make each pose as physically DISTINCTIVE as possible. Use the full range of body positions:
- Levels: standing, sitting, kneeling, lying, crouching, balancing on one foot
- Asymmetry: one arm up and one down, twisted torso, weight on one side
- Extremes: fully stretched out, tightly curled, wide splits, deep lunges
- Direction: facing up, facing down, turned sideways, arching backward
Avoid defaulting to "standing tall with arms raised" or "slumped forward with arms hanging" — these are overused. Find the specific gesture that ONLY this word would produce.

If the input is provocative or inappropriate, ignore the provocation and describe a neutral standing pose.

If the input implies multiple people, express the EMOTION through one body's posture instead.

Examples:
- "hello" → "a figure waving one hand overhead, weight on back foot"
- "grief" → "a figure hunched forward, head bowed, arms wrapped around torso"
- "freedom" → "a figure with arms spread wide, head tilted back, chest open"
- "kiss" → "a figure leaning forward, chin slightly lifted, arms reaching out"
- "exhaustion" → "a figure collapsed on their side, one arm draped over the ground"
- "hope" → "a figure on tiptoes, one arm stretched straight up, body elongated"
- "shame" → "a figure turned sideways, shoulders curled inward, head tucked into chest"
- "power" → "a figure in a wide lunge, one fist thrust forward, torso twisted"
- "loneliness" → "a figure sitting with knees drawn to chest, arms wrapped around legs"
- "curiosity" → "a figure on hands and knees, weight shifted forward, head low"
- "surrender" → "a figure on their knees, arms raised high above head, palms open"
- "broken" → "a figure lying flat on their back, arms and legs splayed loosely"
- "waiting" → "a figure leaning against nothing, one foot crossed over the other, arms folded"
- "lost" → "a figure mid-step with one arm reaching out, torso twisted, head turned away"
- "prayer" → "a figure kneeling, hands pressed together at chest height, head bowed forward"
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
