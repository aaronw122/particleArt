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

### Stage 2: Visual Description → Static Particle Image (Diffusion)

Fine-tuned diffusion model takes the visual description from Stage 1 and generates the "target frame" — the final composition the particles settle into.

The LoRA training captions are purely visual ("figure waving one hand," "two figures embracing") — the model learns the particle art **style**, not semantic interpretation. Semantic interpretation is Stage 1's job.

### Stage 3: Static Image → Animation (Programmatic)

A particle engine (p5.js or Three.js) takes the target image, scatters particles randomly, then animates them drifting into position. No ML needed — it's a physics sim.

### Why Three Stages?

**Separation of concerns.** The LLM handles language understanding (any word → visual concept). The diffusion model handles style (visual concept → particle art). The animation engine handles motion (static image → chaos-to-formation animation). Each component does one thing well, and they're independently testable and improvable.

Diffusion models generate single frames. Video diffusion models exist but they're huge, expensive, and you'd lose control over the animation feel. The particle sim gives you precise control over timing, easing, and movement style — and it's what makes the chaos → formation effect feel good.

---

## Phases

### Phase 1: Curate Training Data

**This is the hardest part.**

#### Approach: OpenAI API batch generation

Use the `gpt-image-1` API (same model behind ChatGPT Plus image gen) to programmatically generate and download images. This avoids manually saving images from the ChatGPT UI.

**Script:** `generate_images.py` — takes a list of scene descriptions, generates images via API, downloads them to `images/raw/`. Vary scenes across compositions (embracing, reaching, dancing, sitting, etc.) to give the model enough diversity to generalize.

**Cost:** ~$0.02-0.08 per image. Budget ~$4-16 for 200 images.

**Workflow:**
1. Generate 10 test images first (~$0.50) to validate the aesthetic
2. If the style lands, batch out the remaining 150-200
3. Cherry-pick 30-50 that match the aesthetic exactly
4. Post-process if needed: threshold to pure black/white, remove gradients, thin out dense areas (Python + PIL)
5. Caption each image with what it depicts ("two figures embracing", "person reaching upward", "figure curled inward")

**Prompt template:**

> "Sparse black particle flecks on a pure white background forming an abstract human figure [doing X]. Minimal, lots of negative space. No gray tones, no shading — just scattered black dots/flecks suggesting the form. Abstract, not realistic."

**Risk:** GPT tends to over-render. Aggressive curation + post-processing script to enforce black/white threshold.

### Phase 2: Fine-Tune Diffusion Model LoRA

- Base model: SDXL (Colab T4) or Flux (cloud A100)
- Same LoRA mechanics as text fine-tuning — just images instead of text
- Train on curated dataset, ~1-2 hours
- Test: prompt with phrases NOT in training data, see if style holds and semantic mapping works

**Key settings (SDXL LoRA):**
- 30-50 training images
- Rank 16-32
- ~1000-1500 training steps
- Colab T4 (16GB VRAM) is sufficient

### Phase 3: ControlNet for Pose Guidance (Maybe)

- If the model struggles with coherent figures, add OpenPose conditioning
- Provides skeleton structure → particle rendering respects human poses
- **Test without this first** — the aesthetic is abstract enough it might not need it
- If needed: Flux ControlNet Union Pro 2.0 or SDXL OpenPose ControlNet

### Phase 4: Particle Animation Engine

- p5.js canvas on the web
- Algorithm:
  1. Receive generated target image from backend
  2. Sample black pixel positions → these become particle target coordinates
  3. Initialize particles at random positions
  4. Animate: particles ease from random positions into target positions
- Tunable parameters: speed, easing curve, particle size, scatter radius, particle count
- Could add subtle drift/breathing after formation for a living feel

### Phase 5: Web App

- **Frontend:** Text input field + p5.js canvas
- **Backend:** API endpoint that takes text, runs diffusion model, returns target image
- **Hosting:** GPU endpoint for inference (Replicate, Modal, or RunPod serverless)
- **Flow:** User types phrase → loading state → local LLM translates phrase to visual description → backend generates target frame from description → frontend animates particles into it

---

## Open Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Training data curation is slow | High | Use GPT-4o to bootstrap, post-process programmatically |
| GPT over-renders (too detailed) | Medium | Aggressive cherry-picking + black/white thresholding script |
| LLM produces descriptions outside diffusion model's trained vocabulary | Medium | Constrain LLM system prompt to 1-2 figures, posture/gesture only; expand training data over time |
| Inference latency (5-30s per generation) | Medium | Loading animation while generating; consider caching common phrases |
| ControlNet needed but adds complexity | Low | Test without first; abstract aesthetic is forgiving |

---

## Tech Stack Summary

| Component | Tool |
|-----------|------|
| Phrase → visual description | Phi-3.5-mini or Qwen2.5-3B via Ollama (local, CPU) |
| Training data generation | OpenAI API (`gpt-image-1`) via `generate_images.py` |
| Post-processing | Python + PIL |
| Base diffusion model | SDXL or Flux |
| Fine-tuning | LoRA via Unsloth/diffusers on Colab T4 |
| Pose conditioning (if needed) | ControlNet OpenPose |
| Animation engine | p5.js |
| Frontend | HTML/JS + p5.js |
| Backend / inference | Replicate, Modal, or RunPod serverless |
| LLM runtime | Ollama on Linux (Dell OptiPlex i5-8500T) |
| Export | Live canvas (optional gif export) |
