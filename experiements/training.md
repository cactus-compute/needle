---
Task: Design the training strategy & loss functions for training a transcribe+tools model.
Deliverable: A well-executed ablation of all strategies against SOTA.
Assignee: Karen Mosoyan & Satyajit Kumar
Deadline: 7th March 
--- 
Notes:
1. We use EMA of weights & this is a deterministic task.
2. What is the best contrastive learning? SiGLip or CLIP?
3. How do we pretrain the encoder? Reconstruction + Contrastive (speech, text, structured)
3. Can we improve the sparsification schedule?
4. How can we incorporate iimplicit VAD?
5. To allow simultaneous generation, use 2 small heads for: transcribe, json.
6. Accumulate gradients 2x for speech and audio?
---
## What do we want to do? 
We want to develop a training recipe for a model that will be able to take either audio or text into the model, and output either a transcription of the audio or an action (tool call output) result based on the audio or the text. 

## What do we want the model to look like?
We want:
1. A shared encoder that can process audio *or* text
2. Lightweight decoder or decoders that will very quickly decode the encoder output into either output type
3. A fast encoder that scales well to longer inputs

## What decisions are we pretty sure about?
1. The encoder we will likely be using is the MemoryMixer. It has several nice properties we would like to use, namely O(N) scaling in input length, GEMM heavy inference, and constant sized output. 
2. The final model will have two separate decoder heads -- one for transcription and one for tool call.
3. The decoders will likely be simple shallow transformers.
4. We are pretty sure about the data we will be using being a lot of labelled audio data, some unlabelled audio data, and tool call data with text -> tool call and synthetically created audio -> tool call data. 

## What do we need to figure out?
1. How to leverage the data we have to train the model as well as possible? 
2. How to make sure the model handles audio and text inputs equivalently for tool call head?

## Some structural decisions we have made that we are less sure about
1. We will need a pretraining stage and a posttraining stage. The pretraining stage will train the encoder to model both audio and text modalities and align them independently of downstream tasks. The posttraining stage will train the model on the downstream tasks of transcription and tool call. 
2. The pretraining stage will be as described in [pretraining.md](pretraining.md).
3. The posttraining stage will be as described in [posttraining.md](posttraining.md).
4. Pretraining will have aggressive data augmentation for audio so that our audio data is not too clean going on so that the model is able to model noise as well as speech. 
5. Posttraining will have two separate stages -- the first stage being learning transcription, and the second stage being learning tool calling. 

## Why this setup?
1. We want to have a pretraining stage so that the encoder starts off by learning to generally model text and audio jointly and the latents created by the encoder are aligned over the full distributions of audio and text. A nice thing about the pretraining is that we can utilize any unlabelled speech data we have in this stage. 
2. The pretraining has adversarial alignment because that gives us the ability to align distributions without pairs (which are needed in contrastive learning), and gives us the ability to align the full distributions rather than pairs. The adversarial minimax training is also very constrained/regularized by the reconstruction loss pressure for audio and text so we don't expect typical instability and mode collapse that may come with GANs.
3. In the pretraining stage we need to align audio and text latent output distributions, however audio reconstruction needs a lot more information than just the speech transcription or speech text, so the audio latents will naturally be richer than the text latents. Therefore we want a audio->text latent projection that strips away this extra information. 
4. In the posttraining stage we need to learn a fundamental skill first, which is transcription, and then tool calling will be very dependent on the transcription. Therefore we want to learn transcription well first before starting tool call training.
5. Since in posttraining we will be using paired (audio;text) data we can use contrastive loss to align actual pairs of latents which is nicer and potentially more stable than the adversarial alignment even though it doesn't have the same distribution wide guarantee
6. For the tool calling we want perfect retrieval of tool names and argument names, so we are inputting the tool list directly into the tool decoder head. 
7. For the tool decoder head we want to use constrained generation, and this works great with a separate head
8. We find in experimentation that a single head cannot do both text modeling and transcription so we fully divide the transcription head and the tool calling head.
9. The pretraining will have several artifacts, including the audio pre-encoder (embedding), text embeddings, shared encoder, audio projector, and text reconstruction head. We will be transferring all of these to the posttraining and the text reconstruction head will serve as the initialization for both the transcription and the tool call head.

---

## Architectural Options

These are the component-level design choices that need to be decided experimentally. Each has a small set of options to ablate.

1. **Audio Projector Architecture**: The projector maps audio latents into the text-aligned subspace. A full $d \times d$ matrix cannot reduce rank, so it may fail to strip non-linguistic information. Options:
   - (a) Full linear: $W \in \mathbb{R}^{d \times d}$ — baseline, no rank reduction
   - (b) LoRA-style bottleneck: $W = AB$ where $A \in \mathbb{R}^{d \times a}$, $B \in \mathbb{R}^{a \times d}$, with $a \ll d$ — forces the projected latent into a rank-$a$ subspace, acting as an information bottleneck that discards prosody/speaker/noise. Sweep $a \in \{d/16, d/8, d/4, d/2\}$.
   - (c) 2-layer MLP with bottleneck: $\text{GELU}(z_a W_1) W_2$ — adds nonlinearity if the linguistic/non-linguistic split is not linearly separable

2. **Audio Embedding**: See [pretraining.md](../docs/pretraining.md) item 1. Options:
   - (a) Raw waveform → learned convolutions (Moonshine-style)
   - (b) MEL spectrogram → convolutions (Whisper-style)
   - (c) MEL spectrogram → patches (ViT-style)

3. **Discriminator Architecture**: The discriminator takes in a matrix-valued latent $\in \mathbb{R}^{M \times d}$ and outputs a scalar. Options:
   - (a) MLP-Mixer: treats slots as a sequence, mixes across both slot and hidden dimensions
   - (b) Pool + MLP: global average pool over slots → 2-layer MLP
   - (c) Learned query attention: single learned query attends to all slots → MLP on the attended output

4. **Tool Call Head Initialization via LoRA Induction Pre-training**: Before training the tool call head on real tool data, we pre-train it on synthetic structured retrieval tasks using a LoRA adapter on top of the frozen text reconstruction head weights. This preserves text generation capability while adding the retrieval/copying circuit. See [posttraining.md](../docs/posttraining.md) Stage 0 for details. Options:
   - (a) LoRA adapter (rank $r$), merge before tool training — retrieval circuit becomes part of base weights
   - (b) Full fine-tune on synthetic data — risk of overwriting text reconstruction capabilities
   - (c) No induction pre-training — baseline

---

## Training Schedule

### Pretraining Schedule
- **Data matching**: To prevent the discriminator from learning to distinguish modalities based on information density rather than content, we use density-matched batching. Audio clips of duration $t$ are batched with text passages of comparable semantic content ($\approx 2.5t$ words). This ensures padding ratios and energy distributions are matched across modalities within each batch, forcing the discriminator to align on content rather than structure.
- **Adversarial $\lambda_\text{adv}$ schedule**: Options include constant, linear warmup, or cyclical. The discriminator update frequency relative to the generator also needs to be set (e.g. $n_D$ discriminator steps per generator step).
- **Curriculum for audio augmentation**: Start with clean audio, gradually introduce noise augmentation (additive noise, reverb, speed perturbation) over the course of training.

### Posttraining Schedule
- **Stage 0 — Induction Head Pre-training**: Symbolic retrieval curriculum on the tool call head (see [posttraining.md](../docs/posttraining.md)). Curriculum:
  - Phase 0a (~500 steps): 2 tools, 1 arg each, explicit values, fully random token embeddings
  - Phase 0b (~500 steps): 5 tools, 2–3 args, explicit values, random embeddings  
  - Phase 0c (~1000 steps): 10 tools, 2–5 args, values embedded in natural context, noise $\sigma$ decaying from 1.0 → 0.3
  - Phase 0d (~500 steps): Full complexity, light embedding noise $\sigma = 0.1$, transitioning to real embeddings
- **Stage 1 — Transcription training**: Audio-text paired data with SigLip contrastive alignment
- **Stage 2 — Tool call training**: Text→tool first, then audio→tool

### Module Transfer & LR/Freezing Ablations (Pretrain → Posttrain)
When transferring pretrained modules to posttraining, we need to decide per-module learning rate and freeze schedules:

| Module | Options |
|--------|---------|
| Audio embedding | Freeze for $N$ steps / reduced LR / full LR |
| Text embedding | Freeze for $N$ steps / reduced LR / full LR |
| Shared encoder | Freeze for $N$ steps / reduced LR / full LR |
| Audio projector | Freeze for $N$ steps / reduced LR / full LR |
| Transcription head (init from text recon) | Full LR (new task) |
| Tool call head (init from text recon + LoRA merge) | Full LR (new task) |

Ablation: Compare (a) freeze all transferred modules for first 10% of posttraining steps, (b) reduced LR (0.1× base) for transferred modules, (c) full LR everywhere. Measure catastrophic forgetting of alignment via discriminator probe and downstream task quality.

---

## Macro Ablation Tree

The ablations below are ordered hierarchically. We run them top-down — negative results at a higher level prune the entire subtree below, saving compute.

### Level 0: Does pretraining help at all?
**Ablation 0.1**: Full pipeline (pretrain → posttrain) vs. posttrain only (randomly initialized encoder + heads)

If pretraining shows no benefit → skip all Level 1 and Level 2 ablations. Proceed directly to Level 3.

### Level 1: Does alignment in pretraining matter?
*Only run if Level 0 shows pretraining helps.*

**Ablation 1.1**: Pretrain with alignment (adversarial) vs. pretrain without alignment (reconstruction only, no discriminator)

If alignment shows no benefit → skip all Level 2 ablations.

### Level 2: Alignment & projector details
*Only run if Level 1 shows alignment helps.*

**Ablation 2.1**: Adversarial vs. contrastive alignment in pretraining (contrastive requires pseudo-pairs from an off-the-shelf ASR transcription)

**Ablation 2.2**: Audio projector architecture (full linear vs. LoRA bottleneck vs. MLP) — see Architectural Options item 1

**Ablation 2.3**: Discriminator architecture (MLP-Mixer vs. pool+MLP vs. learned query) — see Architectural Options item 3

**Ablation 2.4**: Density-matched batching vs. random batching — does the discriminator cheat on information density?

**Ablation 2.5**: $\lambda_\text{adv}$ schedule (constant vs. warmup vs. cyclical)

**Ablation 2.6**: No projector

### Level 3: Posttraining structure
*Always run (independent of pretraining decisions).*

**Ablation 3.1**: Induction head pre-training (Stage 0) on vs. off — does the symbolic retrieval curriculum improve tool call accuracy?

**Ablation 3.2**: SigLip contrastive loss in posttraining on vs. off — does contrastive alignment during task training help or fight the heads?

**Ablation 3.3**: Sequential (transcription → tool) vs. joint (transcription + text-tool simultaneously) — does learning transcription first help tool calling?

**Ablation 3.4**: Audio tool call training timing — simultaneous with text tool call vs. separate final stage

### Level 4: Component-level fine-tuning
*Run after settling Levels 0–3.*

**Ablation 4.1**: Audio embedding choice (waveform conv vs. MEL conv vs. MEL patches) — see Architectural Options item 2

**Ablation 4.2**: Module transfer LR/freezing strategy — see Training Schedule table

**Ablation 4.3**: Induction head curriculum details — random embeddings vs. fixed, decaying noise vs. constant noise, LoRA rank

**Ablation 4.4**: Gradient accumulation ratio for speech vs. text batches

**Ablation 4.5**: EMA decay $\beta$ schedule (constant vs. warmup)
