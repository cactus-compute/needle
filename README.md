```
         ┌────────────────────────────────────────────────────────┐
         │                                                        │
         │       ┌─┐┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌┐┌┌─┐┌─┐┌┬┐┬  ┌─┐           │
         │       │  ├─┤│   │ │ │└─┐  │││├┤ ├┤  │││  ├┤            │
         │       └─┘┴ ┴└─┘ ┴ └─┘└─┘  ┘└┘└─┘└─┘─┴┘┴─┘└─┘           │
         │       ...a 26m function call model  for edge...        │
         └────────────────────────────────────────────────────────┘
```

# Simple Attention Networks

Experiments at Cactus showed that MLPs can be completely dropped from transformer networks, as long as the model relies on external knowledge source.
Function calling relies on external tools list, so we designed a simple attention network for function calling and distilled Gemini-2.0-Flash-Lite.

```
d=512, 8H/4KV, BPE=8192
                                  ┌──────────────┐
                                  │  Tool Call   │
                                  └──────┬───────┘
                                        ┌┴──────────┐
                                        │  Softmax  │
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ Linear (T)│  ← tied
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ ZCRMSNorm │
                                        └─────┬─────┘
                                     ┌────────┴────────┐
                                     │ Decoder x 8     │
                                     │┌───────────────┐│
                                     ││ ZCRMSNorm     ││
                                     ││ Masked Self   ││
                                     ││ Attn + RoPE   ││
                                     ││ Gated Residual││
                                     │├───────────────┤│
  ┌──────────────┐                   ││ ZCRMSNorm     ││
  │ Encoder x 12 │──────────────────────▶Cross Attn   ││
  │              │                   ││ Gated Residual││
  │ ┌──────────┐ │                   │└───────────────┘│
  │ │ZCRMSNorm │ │                   └────────┬────────┘
  │ │Self Attn │ │                      ┌─────┴─────┐
  │ │ GQA+RoPE │ │                      │ Embedding │  ← shared
  │ │Gated Res │ │                      └─────┬─────┘
  │ │          │ │                    ┌───────┴───────-┐
  │ │ (no FFN) │ │                    │[EOS]<tool_call>│
  │ └──────────┘ │                    │ + answer       │
  │              │                    └───────────────-┘
  └──────┬───────┘
         │
    ┌────┴──────┐
    │ Embedding │
    └────┬──────┘
         │
    ┌────┴──────┐
    │   Text    │
    │  query    │
    └───────────┘
```

## Why No FFN

- **Softmax is nonlinear.** `softmax(QK^T/sqrt(d)) * V` is a data-dependent nonlinear mixing operation. For a task that is about routing information (query -> tool alignment), attention is the right primitive.
- **Tool calling is retrieval-and-assembly.** Match query to tool name, extract argument values, assemble JSON. All three are aligning and copying between input and output -exactly what cross-attention does. No step requires per-position feature transformation (which is what FFN provides).
- **At small scale, FFN parameters are wasted.** ~2/3 of standard transformer parameters are FFN. For a <50M model on a structured task, those parameters contribute less than more attention layers (deeper cross-attention = better query-tool alignment).
- **Fewer parameters = faster inference.** FFNs have the biggest GEMM/GEMV dimensions -removing them cuts per-layer parameters by ~2/3, directly reducing the memory bandwidth bottleneck that dominates latency on edge devices.

## Why Encoder-Decoder

- **Bidirectional encoding.** Tools are structured objects -a bidirectional encoder sees the full definition at once. A causal model sees it left-to-right and must infer structure from partial context.
- **No input tokens in KV cache.** Encoder-decoder uses a fixed-size encoder representation for cross-attention, not re-attending the full input at every generation step.
- **Natural fit for multi-head design.** The encoder output feeds the decoder (generation) and the contrastive head (tool retrieval). Clean separation.

## Gated Residuals

Without FFN, there is no per-position nonlinear rewriting per layer. This makes residual connection design critical.

- **Standard residual** `x = x + Attn(Norm(x))` -attention can only ADD a delta. Without FFN to do the rewriting, purely additive is limiting.
- **No residual** `x = Attn(Norm(x))` -each layer fully rewrites, but we lose the gradient highway. Deep networks (12+ layers) will not train.
- **Gated residual (ours)** `x = x + sigmoid(gate) * Attn(Norm(x))` -per-sublayer learnable scalar, initialized to 0. sigmoid(0) = 0.5, so training starts with half-strength residual. The model can learn to sharpen useful layers (g->1) or suppress unhelpful ones (g->0) without losing gradient flow.

## ZCRMSNorm

- **Standard RMSNorm:** `x * gamma / RMS(x)`, gamma initialized to 1.
- **ZCRMSNorm:** `x * (1 + gamma) / RMS(x)`, gamma initialized to 0.
- At init, ZCRMSNorm is identity-up-to-scale. Pairs with gated residuals: the entire block starts as a damped identity + damped normalized attention. No component starts with a strong learned bias.
- From the nGPT / DeepSeek-V3 line of work. Applied to QK heads as well (QK-norm) for training stability.

## Contrastive Tool Selection Head

CLIP-style head for retrieving relevant tools before generation. Useful when the tool set is large and you want to filter to the top-k most relevant tools for a query.

- **Architecture:** encoder output -> mean pool over non-pad positions -> Dense(d_model/4) -> ReLU -> Dense(128) -> L2-normalize. Produces a unit vector per input.
- **Training:** symmetric contrastive loss (CLIP). Each batch pairs queries with their positive tools; in-batch negatives provide the contrastive signal. Learnable temperature (`log_temp`).
- **Inference:** encode query and each tool candidate into the shared embedding space, rank by cosine similarity, take top-k.
- Trained jointly with the main CE loss at 0.1x weight. Same encoder is used for both generation and retrieval.

## Muon for Attention-Only

- **Dual optimizer:** Muon (Q/K/V/O projections, LR 0.02, WD 0.01) + AdamW (everything else, LR 3e-4).
- Without FFN, the model is a deep stack of linear projections with softmax routing. Muon enforces orthogonality on weight updates via Newton-Schulz, preventing the representation collapse that can happen when stacking many linear layers without interleaving nonlinearities.

## Training

- Pretrained on 16 TPU v6e for 200B tokens. 
- Postrained on 2B tokens of tool call dataset.
- Weights on [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle).

## Test Needle & Finetune Locally On Your Mac 

```
git clone https://github.com/cactus-compute/needle.git
cd needle && source ./setup
needle ui
```
Opens at http://127.0.0.1:7860, play with Needle oin your Mac. 


## Needle CLI

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   ui                           Web UI for inference + finetuning  │
  │     --checkpoint PATH          Model checkpoint (optional,        │
  │                                auto-downloads from HuggingFace)   │
  │     --port INT                 Server port (default: 7860)        │
  │     --host STR                 Bind address (default: 127.0.0.1)  │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH          Path to model checkpoint (required)│
  │     --query STR                Query text for tool-call generation│
  │     --tools STR                Tools JSON for tool-call generation│
  │     --max-len INT              Max tokens to generate (default:512│
  │     --seed INT                 Random seed (default: 0)           │
  │     --no-constrained           Disable constrained decoding       │
  │                                                                   │
  │   train                                                           │
  │     --name STR                 Experiment name (default: baseline)│
  │     --checkpoint PATH          Resume from checkpoint             │
  │     --init-from PATH           Init params from pretrained base   │
  │     --epochs INT               Training epochs (default: 1)       │
  │     --batch-size INT           Batch size (default: 64)           │
  │     --lr FLOAT                 AdamW learning rate (default: 3e-5)│
  │     --muon-lr FLOAT            Muon learning rate (default: 0.02) │
  │     --d-model INT              Model dim (default: 512)           │
  │     --num-heads INT            Attention heads (default: 8)       │
  │     --num-kv-heads INT         KV heads for GQA (default: 4)      │
  │     --num-layers INT           Encoder layers (default: 12)       │
  │     --num-dec-layers INT       Decoder layers (default: 8)        │
  │     --max-enc-len INT          Max encoder seq len (default: 1024)│
  │     --max-dec-len INT          Max decoder seq len (default: 512) │
  │     --max-samples INT          Training samples (default: all)    │
  │     --no-feedforward           No FFN layers (default: on)        │
  │     --feedforward              Enable FFN layers (off by default) │
  │     --activation STR           swiglu|drelu|geglu (if FFN on)     │
  │     --mat-factors INT [...]    FFN shrink factors (if FFN on)     │
  │     --precision STR            QAT: int4|int8 (default: int4)     │
  │     --warmup-ratio FLOAT       LR warmup ratio (default: 0.05)    │
  │     --decay-ratio FLOAT        LR cosine decay ratio (default:0.05│
  │     --dropout FLOAT            Dropout rate (default: 0.0)        │
  │     --eval-every INT           Val eval interval (default: 1000)  │
  │     --max-eval-samples INT     Val samples limit (default: 5000)  │
  │     --contrastive-weight FL    CLIP loss weight (default: 0.1)    │
  │     --contrastive-dim INT      Projection dim (default: 128)      │
  │     --w-name FLOAT             Loss weight: tool names (def: 2.0) │
  │     --w-value FLOAT            Loss weight: arg values (def: 4.0) │
  │     --w-key FLOAT              Loss weight: arg keys (def: 1.5)   │
  │     --wandb                    Enable W&B logging                 │
  │     --checkpoint-dir DIR       Checkpoint directory               │
  │     --dtype STR                float32|bfloat16 (default: bfloat16│
  │     --seed INT                 Random seed (default: 42)          │
  │                                                                   │
  │   pretrain                                                        │
  │     --name STR                 Experiment name (default: pretrain) │
  │     --checkpoint PATH          Resume from checkpoint             │
  │     --resume-step INT          Override resume step                │
  │     --epochs INT               Training epochs (default: 1)       │
  │     --batch-size INT           Batch size (default: 128)          │
  │     --lr FLOAT                 AdamW learning rate (default: 3e-4)│
  │     --muon-lr FLOAT            Muon learning rate (default: 0.02) │
  │     --max-steps INT            Stop after N steps (default: full) │
  │     --save-every INT           Checkpoint interval (default: 1000)│
  │     --wandb                    Enable W&B logging                 │
  │     (also accepts --d-model, --num-heads, etc. same as train)     │
  │                                                                   │
  │   tokenize                                                        │
  │     --max-samples INT          Limit samples per split (dev/test) │
  │     --max-enc-len INT          Max encoder seq len (default: 1024)│
  │     --max-dec-len INT          Max decoder seq len (default: 512) │
  │     --shuffle-tools            Shuffle tool order (default: on)   │
  │     --no-shuffle-tools         Disable tool shuffling             │
  │     --max-tool-len INT         Max tool desc tokens (default: 256)│
  │                                                                   │
  │   eval                                                            │
  │     --checkpoint PATH          Path to model checkpoint (required)│
  │     --batch-size INT           Batch size (default: 32)           │
  │     --max-eval-samples INT     Evaluation samples (default: 5000) │
  │     --max-enc-len INT          Max encoder length (default: 1024) │
  │     --max-dec-len INT          Max decoder length (default: 512)  │
  │     --max-gen-len INT          Max generation length (default:512)│
  │     --throughput-runs INT      Throughput runs (default: 10)      │
  │     --tool-call-samples INT    Tool-call eval samples (default:200│
  │     --no-constrained           Disable constrained decoding       │
  │                                                                   │
  │   generate-data                                                   │
  │     --num-samples INT          Samples to generate (default: 500) │
  │     --batch-size INT           Examples per Gemini call (default:25│
  │     --workers INT              Parallel Gemini calls (default: 8) │
  │     --model STR                Gemini model override              │
  │     --dry-run                  Generate only, skip upload         │
  │     --output-jsonl PATH        Also save raw generations to JSONL │
  │     --upload-every INT         Merge+upload interval              │
  │                                                                   │
  │   tpu                                                             │
  │     create NAME                Create TPU (auto-finds zone)       │
  │       --type STR               Accelerator (default: v6e-8)       │
  │       --version STR            TPU OS (auto-detected from --type) │
  │       --preemptible            Create spot/preemptible instance   │
  │     connect NAME               SSH config + connect (auto-zone)   │
  │     setup NAME                 Full setup: sync code + venv + deps│
  │     sync NAME                  Fast sync: code (no venv rebuild)  │
  │     train NAME [-- ARGS]       Launch training on all workers     │
  │     pretrain NAME [-- ARGS]    Launch pretraining on all workers  │
  │     claude NAME                Install Claude Code on instance    │
  │     stop NAME                  Stop instance (auto-zone)          │
  │     start NAME                 Start stopped instance (auto-zone) │
  │     delete NAME                Delete instance (auto-zone)        │
  │     list                       List all TPU instances             │
  │       --zone ZONE              Override auto-detected zone        │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

## Setup For TPU/GCP

- Setup gcloud 1: download the `macOS ARM` from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and unzip.
- Setup gcloud 2: open terminal, cd to your downloads and run `./google-cloud-sdk/install.sh`
- Setup gcloud 3: restart terminal and run `gcloud init`, sign in with cactus email, should prompt for project
- Setup gcloud 4: else, set the project with `gcloud config set [PROJECT NAME]`
- Setup gcloud 5: run `gcloud help` and read carefully

### Single-host (v6e-8) -SSH into instance

```
1. Create an instance (auto: finds zone → installs Claude Code → connects via SSH)
   needle tpu create my-experiment
   (exit with 'exit' or Ctrl+D)

2. Reconnect anytime
   ssh my-experiment
   or VS Code: click the '><' in the bottom left → select my-experiment

--- run from the instance ---

3. Create a GitHub Personal Access Token (PAT)
   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   Generate a token with 'repo' scope

4. Clone the repo on your instance using your PAT
   git clone https://github.com/cactus-compute/needle.git
   cd needle && source ./setup

5. Login to Hugging Face (required for private datasets)
   hf auth login
   (paste your HF token -get one at https://huggingface.co/settings/tokens)

6. Use tmux for long training runs (survives SSH disconnects)
   tmux new -s train          # start a named session
   needle train --wandb       # run training inside it
   Ctrl+B, then D             # detach (keeps running)
   tmux attach -t train       # reattach later

--- back on your Mac ---

7. Stop when done (saves disk, stops billing)
   needle tpu stop my-experiment

8. (Optional) Delete instance when no longer needed
   needle tpu delete my-experiment
```

### Multi-host (v6e-16+) -run from your Mac

For multi-host TPUs (v6e-16+), you drive everything from your Mac.
The CLI syncs code and launches training across all workers automatically.

```
1. Set auth tokens (workers need these to download data + log metrics)
   export HF_TOKEN=hf_...
   export WANDB_API_KEY=...

2. Add SSH key to agent (required for gcloud scp/ssh)
   ssh-add ~/.ssh/google_compute_engine

3. Create a multi-host TPU (auto: finds zone → syncs code → installs deps)
   needle tpu create my-experiment --type v6e-16

4. Launch training on all workers from your Mac
   needle tpu train my-experiment -- --wandb --epochs 1

5. After code changes, sync without rebuilding venv (fast, ~seconds)
   needle tpu sync my-experiment

6. Full re-setup if deps changed (slow, rebuilds venv)
   needle tpu setup my-experiment

7. Stop/delete when done
   needle tpu stop my-experiment
   needle tpu delete my-experiment
```
