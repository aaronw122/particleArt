# Text → Particle Art Animation (Web App)

## What We're Building

A web app where someone types any phrase and gets an animation of sparse black particle flecks on a white background drifting from chaos into an abstract anthropomorphic scene that represents the phrase's meaning.

**Aesthetic:** Sparse black flecks, white background, abstract figures, lots of negative space. Not pointillism — more like scattered dots suggesting form.

**Animation:** Chaos → formation. Particles start scattered randomly, then drift into the target composition.

**Input:** Any text (open-ended, not a fixed vocabulary).

**Output:** Animated canvas (live in browser).

---

## Architecture: Three-Stage Pipeline

### Stage 1: Phrase → Visual Description (LLM Translation Layer)

A small local LLM translates the user's input phrase into a concrete visual description of body language, posture, and gesture. This is what lets any arbitrary word produce a meaningful image — the LLM handles semantics, the diffusion model handles aesthetics.

**Example mappings:**
- "hello" → "figure waving one hand outward, open posture"
- "grief" → "figure hunched forward, head bowed, arms wrapped around torso"
- "understand" → "two figures facing each other, one gesturing with open palms"
- "freedom" → "figure with arms spread wide, head tilted back"

**Hardware:** Dell OptiPlex 7060 Micro (i5-8500T, up to 32GB RAM, no GPU — CPU inference only).

**Model:** Phi-3.5-mini (3.8B, ~4GB RAM, ~10-15 tok/s on CPU) or Qwen2.5-3B (3B, ~3.5GB RAM, ~12-15 tok/s). Both are more than capable for single-sentence output.

**Runtime:** Ollama on Linux — provides a local REST API at `localhost:11434`. Zero external dependencies.

**System prompt:** Constrain output to match what the diffusion model was trained on:
> "Given a word or phrase, describe a scene using one or two abstract human figures. Use only posture, gesture, and spatial position. One sentence. No color, no style, no background details."

**Why local?** The translation task is trivial — one sentence in, one sentence out. No need to pay per-call for a cloud API. Latency is ~1-2 seconds on CPU, which is negligible compared to the diffusion step.

**Constraints (discovered during model evaluation):**
- Single figures only (no two-figure compositions — training data was ~40+ single figures, only ~6-7 two-figure)
- Whole-body poses and gestures (no fine hand/finger details)
- No small handheld objects (express through body posture, not held items)
- Prompts must use `<s0><s1>` trigger tokens
- Steer toward: balancing, crawling, meditation, arms raised, leaping, lunging, arching

### Stage 2: Visual Description → Static Particle Image (Diffusion)

Fine-tuned diffusion model takes the visual description from Stage 1 and generates the "target frame" — the final composition the particles settle into.

The LoRA training captions are purely visual ("figure waving one hand," "two figures embracing") — the model learns the particle art **style**, not semantic interpretation. Semantic interpretation is Stage 1's job.

**Deployment:** Modal serverless endpoint (`youfoundaaron--prtkl-generate-model-generate.modal.run`).
**Weights:** HuggingFace (`aaronw122/prtkl-sdxl-lora`), checkpoint-1800 AdamW v4.
**Post-process:** Threshold to transparent PNG (dissolve background issues).

### Stage 3: Static Image → Animation (Programmatic)

A particle engine (p5.js or Three.js) takes the target image, scatters particles randomly, then animates them drifting into position. No ML needed — it's a physics sim.

### Why Three Stages?

**Separation of concerns.** The LLM handles language understanding (any word → visual concept). The diffusion model handles style (visual concept → particle art). The animation engine handles motion (static image → chaos-to-formation animation). Each component does one thing well, and they're independently testable and improvable.

Diffusion models generate single frames. Video diffusion models exist but they're huge, expensive, and you'd lose control over the animation feel. The particle sim gives you precise control over timing, easing, and movement style — and it's what makes the chaos → formation effect feel good.

---

## Progress

### Done
- [x] Phase 1: Training data curation (29 images in `images/curated/`)
- [x] Phase 2: LoRA fine-tuning (AdamW v4, step 1800 selected)
- [x] Phase 2: Deploy inference endpoint (Modal, `youfoundaaron` account)
- [x] Phase 2: Upload weights to HuggingFace (`aaronw122/prtkl-sdxl-lora`)

### Remaining
- [ ] Post-processing pipeline (threshold to transparent PNG)
- [x] Stage 1: LLM translator layer (Ollama on Dell OptiPlex or cloud fallback)
- [ ] Stage 3: Particle animation engine (p5.js)
- [x] Web app frontend (text input + canvas)
- [x] Integration: wire all three stages together

---

## Tech Stack Summary

| Component | Tool |
|-----------|------|
| Phrase → visual description | Phi-3.5-mini or Qwen2.5-3B via Ollama (local, CPU) |
| Training data generation | OpenAI API (`gpt-image-1`) via `generate_images.py` |
| Post-processing | Python + PIL |
| Base diffusion model | SDXL 1.0 |
| Fine-tuning | LoRA + Textual Inversion (pivotal tuning) via diffusers |
| Animation engine | p5.js |
| Frontend | HTML/JS + p5.js |
| Backend / inference | Modal serverless (A10G) |
| LLM runtime | Ollama on Linux (Dell OptiPlex i5-8500T) |
| Export | Live canvas (optional gif export) |
