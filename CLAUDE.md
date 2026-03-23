# SDXL LoRA Fine-Tuning for Abstract Art Styles

A guide for training SDXL LoRA models to generate stylized/abstract art that deviates
from SDXL's photorealistic prior. Based on lessons from training a particle art style
(sparse black dots forming human figures on white backgrounds).

**How far is your style from photorealism?** These settings were tuned for an extreme
shift. If your target style is closer to photorealism (watercolor, oil painting), you
may need less aggressive training — lower rank, fewer steps, cosine scheduler. The
further from photorealism, the more aggressive you need to be.

## Workflow

1. Prepare dataset (50 curated images, captioned)
2. Train with pivotal tuning (TI + LoRA)
3. Sweep checkpoints with novel prompts
4. Select best checkpoint, download weights
5. Post-process if needed (background normalization, transparency)

## Dataset Preparation

**Quality over quantity.** 50 hand-curated images outperformed 321 auto-generated ones.
Human selection identifies the best stylistic examples. More data dilutes the signal
when the style is narrow.

**Directory structure** expected by the diffusers training script:

```
dataset/train/
  image_001.png
  metadata.csv       # columns: file_name, text
```

**Captions describe content ONLY, never visual style.** Let the TI token carry all
style information. If captions include style words ("sparse particles on white
background"), the model attributes style to those words instead of the TI token, and
style breaks at inference without those words in the prompt.

Good: `"TOK, a figure lunging forward with arms extended"`
Bad: `"TOK, sparse black particles on white background forming a figure lunging"`

**Composition complexity must match training data.** The model only reliably generates
what's well-represented in the dataset. If 90% of images are single subjects,
multi-subject prompts will produce artifacts (disconnected parts, merged forms).

## Training Script

This uses the HuggingFace diffusers advanced dreambooth script. Pin `diffusers==0.31.0`
and fetch the matching script version:

```
wget https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py
```

Newer diffusers versions may break compatibility with this script's flags.

## Pivotal Tuning (TI + LoRA) — Do Not Remove

**This is the most important decision.** Pure LoRA without Textual Inversion produces
photorealistic output regardless of your trigger word. A made-up trigger word gets
tokenized into existing subwords with pre-existing meanings. TI creates fresh embedding
slots (`<s0><s1>`) with zero prior meaning — giving the UNet a clean style signal.

```
--train_text_encoder_ti
--train_text_encoder_ti_frac=0.5
--token_abstraction=TOK
--num_new_tokens_per_abstraction=2
```

Do not replace `--train_text_encoder_ti` with `--train_text_encoder` (text encoder
LoRA). That modifies global text processing rather than learning a dedicated style
token. Both were tested — only TI works for style binding.

The TI freeze fraction (0.5) should not go below 0.5. Lower values weaken style binding.

**Never use `--cache_latents` with TI training.** It caches text encoder outputs at
step 0, before TI has learned anything. Training appears normal but TI tokens end up
meaningless. Silent, total failure.

## Training Configuration

Starting point for styles far from photorealism. Change one parameter at a time.

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix` (required — standard VAE produces black images in fp16)
- Resolution: 1024, batch size: 1, gradient accumulation: 4
- Mixed precision: bf16
- LoRA rank: 32 (16 was insufficient for extreme style shifts)
- SNR gamma: 5.0
- Noise offset: 0.0357 (see note below — value depends on your target background)
- LR scheduler: constant (cosine wastes the tail for styles fighting the prior)
- Max steps: 1000-1900 (50 images needed ~1900; scale with dataset size)
- Checkpointing: every 100 steps, keep last 5-10

**Calibrate aggressiveness to style distance.** For styles far from photorealism
(particle art, geometric line art), conservative settings (lower rank, cosine decay,
fewer steps) let the prior bleed back in. For styles closer to photorealism (watercolor,
ink portraits), you *want* some prior leakage — recognizable faces and natural forms
depend on it. Rank 16, cosine scheduler, and fewer steps may be better for those.

**Noise offset** compensates for a luminance bias in diffusion training that pulls
backgrounds toward mid-tone gray. The value 0.0357 was tuned for pure white backgrounds.
Adjust per your style: dark backgrounds may need different values, textured or colored
backgrounds may not need it at all. If your background is an intentional design element
(cream paper, black void, textured surface), test with and without.

**Prodigy optimizer:** set `--text_encoder_lr` equal to `--learning_rate` (both 1.0).
Different values crash Prodigy. PyPI package: `prodigyopt`.

**AdamW 8-bit:** LR ~9e-5 for UNet, ~2.5e-4 for text encoder, 100-step warmup.

## Full Training Command Example

```
accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --instance_prompt=TOK \
  --dataset_name=./dataset \
  --caption_column=text \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --use_8bit_adam \
  --learning_rate=9e-5 \
  --text_encoder_lr=2.5e-4 \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac=0.5 \
  --token_abstraction=TOK \
  --num_new_tokens_per_abstraction=2 \
  --lr_scheduler=constant \
  --lr_warmup_steps=100 \
  --max_train_steps=1900 \
  --rank=32 \
  --snr_gamma=5.0 \
  --noise_offset=0.0357 \
  --output_dir=./output \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=25 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="TOK, a figure standing with arms at sides" \
  --num_validation_images=1 \
  --validation_epochs=6 \
  --seed=42 \
  --logging_dir=./logs \
  --report_to=tensorboard
```

## Checkpoint Sweep

Don't assume the final checkpoint is best. Generate from multiple checkpoints with
novel prompts (NOT from training captions — that tests memorization, not generalization).

```
CHECKPOINTS = [600, 800, 1000, 1200, 1400, 1600]
PROMPTS = [
    "<s0><s1>, a figure standing with arms crossed",
    "<s0><s1>, a figure mid-jump with legs tucked",
    "<s0><s1>, a figure kneeling with one hand on ground",
]
```

Create a fresh pipeline per checkpoint — don't reuse and swap LoRA weights, leftover
weights contaminate results. Organize output as `sweep/step{N}/{idx}.png`.

**What to look for:** Undertrained checkpoints show the base model's photorealism
leaking through. Overtrained checkpoints memorize training poses and lose generalization.
The sweet spot shows clear style with varied, prompt-responsive compositions.

## Inference

**Load both LoRA weights AND TI embeddings.** `load_lora_weights()` does not load TI.
TI embeddings are a separate file (`results_emb.safetensors`) loaded via
`load_textual_inversion()` into BOTH text encoders (CLIP-L and CLIP-G):

```python
state_dict = load_file("results_emb.safetensors")
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"],
    text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"],
    text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
```

**Use `<s0><s1>` in prompts, not `TOK`.** The token abstraction mapping only exists
inside the training script. At inference, the model only knows the raw token IDs.

**Example prompts** (content-focused, style carried entirely by TI tokens):

```
<s0><s1>, a figure standing with arms crossed over chest, white background
<s0><s1>, a figure mid-jump with legs tucked, white background
<s0><s1>, a figure arching backward, white background
```

**Rebuild your negative prompt per project.** It should name specific unwanted behaviors
from the base model, not be copied from another project. For monochrome styles you might
negate color; for colorful styles that would destroy what you want. Start with
`"photorealistic, 3D, lighting"` and add terms based on what you see in early outputs.

**Guidance scale 5.0.** Higher guidance amplifies artifacts, not just style. If the
model partially learned the wrong thing, pushing harder makes it worse.

**Base model priors leak through LoRA.** Add explicit negative prompts for unwanted
base-model behavior (e.g., nudity, photorealistic shading, 3D lighting, colored
backgrounds). Be specific — generic negative prompts don't help.

## Post-Processing

If your style has uniform backgrounds (pure white, pure black), expect the model to
produce near-but-not-quite versions. Don't fight this at inference with prompt
engineering — post-process instead. Threshold near-target pixels to the exact value
(e.g., RGB > 230 → 255 for white backgrounds, RGB < 25 → 0 for black).

If your background is intentionally textured or colored, skip this step — thresholding
will destroy the texture.

For UI integration, convert uniform backgrounds to alpha transparency using
brightness-based (or darkness-based) opacity.

## Training Resumption

**Download best weights before resuming.** Resuming overwrites the final weights file
and `checkpoints_total_limit` may delete earlier checkpoints. Always download the
current best checkpoint locally before starting a new run on the same volume.

Changing rank, optimizer, or other structural params requires clearing old checkpoints
first. Optimizer state shapes won't match.

Remove `--resume_from_checkpoint=latest` for fresh starts — it crashes on empty
checkpoint directories. Add it back after the first checkpoint saves.

## Cloud GPU Notes (Modal)

- `modal run --detach script.py` — `--detach` must come BEFORE the script name
- `modal volume get` for directories is unreliable — download files individually
- Always `volume.commit()` in a finally block. Expect 2-3 spot preemptions per long run
- Post-training validation can OOM on smaller GPUs. Keep `--num_validation_images` at
  1 or 0 and ensure validation crashes don't block volume commit
