import argparse
import glob
import json
import os
import random
import shutil
import tempfile
import time

from ..dataset import dataset as data_mod
from ..dataset.dataset import _cache_key, _save_cache_metadata, get_tokenizer, pack_sequences, prepare_tool_call_pairs


def _write_split(split, examples, tokenizer, cache_dir, max_enc_len, max_dec_len):
    from datasets import Dataset

    ds = Dataset.from_list(examples)
    enc_vl, dec_in_vl, dec_tgt_vl, loss_vl, kept, _ = prepare_tool_call_pairs(
        ds, tokenizer, max_enc_len=max_enc_len, max_dec_len=max_dec_len, shuffle_tools=True,
    )
    # Cache key weights must match prepare_tool_call_pairs (1.0)
    cache_id = _cache_key("toolcall", len(ds), max_enc_len, max_dec_len,
                          1.0, 1.0, 1.0, True)
    cache_path = os.path.join(cache_dir, cache_id)
    pack_sequences(cache_path, enc_vl, dec_in_vl, dec_tgt_vl, loss_vl)
    _save_cache_metadata(split, cache_id, len(kept), max_enc_len, max_dec_len, 256)
    return len(kept)


def _ensure_best_checkpoint(checkpoint_dir, run_id):
    """Find or promote the best checkpoint scoped to this run's unique ID."""
    prefix = f"needle_finetuned_{run_id}_"
    best = sorted(
        p for p in glob.glob(os.path.join(checkpoint_dir, f"{prefix}*_best.pkl"))
    )
    if best:
        return best[-1]

    candidates = sorted(
        p for p in glob.glob(os.path.join(checkpoint_dir, f"{prefix}*.pkl"))
        if not p.endswith("_best.pkl")
    )
    if not candidates:
        return None

    latest = candidates[-1]
    best_name = os.path.basename(latest).rsplit("_", 1)[0] + "_best.pkl"
    best_path = os.path.join(checkpoint_dir, best_name)
    shutil.copy2(latest, best_path)
    print(f"Promoted final checkpoint to best: {best_path}")
    return best_path


def _call_key(c):
    """Canonical key for a tool call (name + sorted arguments). Matches train.py."""
    if not isinstance(c, dict):
        return None
    return json.dumps({"name": c.get("name"), "arguments": c.get("arguments")}, sort_keys=True)


def _quick_tool_eval(model, params, tokenizer, examples, max_gen_len=512, max_enc_len=1024):
    """Tool-call eval matching needle's F1 methodology (TP/FP/FN)."""
    from ..model.run import generate_batch

    samples = [ex for ex in examples
               if ex.get("answers", "").strip() not in ("", "[]")]
    if not samples:
        return {}

    _BATCH = 32
    all_preds = []
    for i in range(0, len(samples), _BATCH):
        chunk = samples[i:i + _BATCH]
        all_preds.extend(generate_batch(
            model, params, tokenizer,
            [s["query"] for s in chunk],
            [s["tools"] for s in chunk],
            max_gen_len=max_gen_len, max_enc_len=max_enc_len, constrained=True,
        ))

    n = len(samples)
    parse_ok, exact = 0, 0
    name_tp, name_fp, name_fn = 0, 0, 0
    call_tp, call_fp, call_fn = 0, 0, 0
    args_correct, args_total = 0, 0
    per_tool = {}

    for ex, pred_text in zip(samples, all_preds):
        pred_text = pred_text.strip()
        ref_text = ex["answers"].strip()
        try:
            ref_calls = json.loads(ref_text)
        except (ValueError, TypeError):
            ref_calls = []
        try:
            pred_calls = json.loads(pred_text)
            if not isinstance(pred_calls, list):
                pred_calls = [pred_calls] if isinstance(pred_calls, dict) else []
            parse_ok += 1
        except (ValueError, TypeError):
            pred_calls = []

        ref_keys = sorted(_call_key(c) for c in ref_calls if _call_key(c))
        pred_keys = sorted(_call_key(c) for c in pred_calls if _call_key(c))
        if ref_keys == pred_keys and len(ref_keys) == len(ref_calls) and len(pred_keys) == len(pred_calls):
            exact += 1

        ref_names = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
        pred_names = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
        name_tp += len(pred_names & ref_names)
        name_fp += len(pred_names - ref_names)
        name_fn += len(ref_names - pred_names)

        rk = {_call_key(c) for c in ref_calls} - {None}
        pk = {_call_key(c) for c in pred_calls} - {None}
        call_tp += len(pk & rk)
        call_fp += len(pk - rk)
        call_fn += len(rk - pk)

        ref_by_name = {}
        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
        for c in pred_calls:
            if isinstance(c, dict) and "name" in c and c["name"] in ref_by_name:
                args_total += 1
                pred_args = json.dumps(c.get("arguments", {}), sort_keys=True)
                if any(pred_args == json.dumps(ra, sort_keys=True) for ra in ref_by_name[c["name"]]):
                    args_correct += 1

        for c in ref_calls:
            if not isinstance(c, dict) or "name" not in c:
                continue
            rname = c["name"]
            rargs = json.dumps(c.get("arguments", {}), sort_keys=True)
            if rname not in per_tool:
                per_tool[rname] = {"correct": 0, "total": 0}
            per_tool[rname]["total"] += 1
            for pc in pred_calls:
                if isinstance(pc, dict) and pc.get("name") == rname:
                    if json.dumps(pc.get("arguments", {}), sort_keys=True) == rargs:
                        per_tool[rname]["correct"] += 1
                        break

    name_p = name_tp + name_fp
    name_r = name_tp + name_fn
    call_p = call_tp + call_fp
    call_r = call_tp + call_fn

    return {
        "call_f1": round(2 * call_tp / max(call_p + call_r, 1), 4),
        "name_f1": round(2 * name_tp / max(name_p + name_r, 1), 4),
        "exact_match": round(exact / max(n, 1), 4),
        "parse_rate": round(parse_ok / max(n, 1), 4),
        "args_acc": round(args_correct / max(args_total, 1), 4),
        "n": n,
        "per_tool": per_tool,
    }


def _per_tool_split(examples, val_per_tool=10, test_per_tool=10, seed=42):
    """Split examples per tool: remainder->train, val_per_tool->val, test_per_tool->test.

    Each tool gets exactly val_per_tool examples in val and test_per_tool in test.
    If a tool has fewer than val_per_tool + test_per_tool + 1, the split is
    proportional (at least 1 in each when possible).
    """
    rng = random.Random(seed)

    # Group by primary tool (first tool in answers)
    tool_buckets = {}
    for i, ex in enumerate(examples):
        try:
            calls = json.loads(ex.get("answers", "[]"))
        except (ValueError, TypeError):
            calls = []
        names = [c["name"] for c in calls if isinstance(c, dict) and "name" in c]
        primary = names[0] if names else "__no_tool__"
        tool_buckets.setdefault(primary, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for tool, indices in tool_buckets.items():
        rng.shuffle(indices)
        n = len(indices)
        needed = val_per_tool + test_per_tool
        if n < needed:
            if n == 1:
                n_test, n_val, n_train = 1, 0, 0
            elif n == 2:
                n_test, n_val, n_train = 1, 1, 0
            else:
                n_test = max(1, n // 3)
                n_val = max(1, (n - n_test) // 3)
                n_train = n - n_val - n_test
        else:
            n_test = test_per_tool
            n_val = val_per_tool
            n_train = n - n_val - n_test

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    test = [examples[i] for i in test_idx]
    return train, val, test


def _emit(tag, data):
    print(f"{tag}:{json.dumps(data)}", flush=True)


def _resolve_checkpoint(path):
    """Resolve checkpoint path, always downloading from HuggingFace to ensure freshness."""
    from huggingface_hub import hf_hub_download
    local_dir = "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(path) if path else "needle.pkl"
    print(f"Downloading {filename} from Cactus-Compute/needle...")
    return hf_hub_download(
        repo_id="Cactus-Compute/needle",
        filename=filename,
        repo_type="model",
        local_dir=local_dir,
        force_download=True,
    )


def finetune_local(args):
    args.checkpoint = _resolve_checkpoint(args.checkpoint)

    with open(args.jsonl_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {args.jsonl_path}")
    if len(examples) < 3:
        raise ValueError("finetune requires at least 3 examples for train/val/test splits")

    from ..dataset.tokenizer import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN
    max_enc_len = getattr(args, "max_enc_len", None) or DEFAULT_MAX_ENC_LEN
    max_dec_len = getattr(args, "max_dec_len", None) or DEFAULT_MAX_DEC_LEN

    _owns_cache = args.cache_dir is None
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="needle-finetune-cache-")
    os.makedirs(cache_dir, exist_ok=True)

    train_examples, val_examples, test_examples = _per_tool_split(examples)

    if len(train_examples) == 0:
        all_avail = val_examples + test_examples
        random.Random(42).shuffle(all_avail)
        n = len(all_avail)
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Not enough data ({n} examples) for train/val/test splits")
        test_examples = all_avail[:n_test]
        val_examples = all_avail[n_test:n_test + n_val]
        train_examples = all_avail[n_test + n_val:]

    if len(train_examples) == 0:
        raise ValueError("Not enough data — need at least 3 examples per tool for train/val/test splits")

    print(f"Split: {len(train_examples)} train / {len(val_examples)} val / {len(test_examples)} test (per-tool)")

    # Load checkpoint config
    import pickle as _pkl
    with open(args.checkpoint, "rb") as f:
        ckpt_config = _pkl.load(f)["config"]

    # Unique run ID for checkpoint scoping
    run_id = f"{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
    experiment_name = f"needle_finetuned_{run_id}"

    original_cache_dir = data_mod.CACHE_DIR
    data_mod.CACHE_DIR = cache_dir
    try:
        tokenizer = get_tokenizer()
        train_kept = _write_split("train", train_examples, tokenizer, cache_dir, max_enc_len, max_dec_len)
        val_kept = _write_split("val", val_examples, tokenizer, cache_dir, max_enc_len, max_dec_len)

        if train_kept == 0:
            raise ValueError(f"Tokenization produced 0 training sequences from {len(train_examples)} examples")
        if val_kept == 0:
            raise ValueError(f"Tokenization produced 0 validation sequences from {len(val_examples)} examples")
        print(f"Tokenized: {train_kept} train / {val_kept} val sequences")

        from datasets import Dataset as _Dataset
        val_ds = _Dataset.from_list(val_examples)

        # Base model eval on TEST set (never seen by model)
        from ..model.run import load_checkpoint
        from ..model.architecture import SimpleAttentionNetwork
        print(f"Evaluating base model on {len(test_examples)} test examples...")
        base_params, base_config = load_checkpoint(args.checkpoint)
        base_model = SimpleAttentionNetwork(base_config)
        base_metrics = _quick_tool_eval(
            base_model, base_params, tokenizer, test_examples,
            max_gen_len=min(max_dec_len, 512), max_enc_len=max_enc_len,
        )
        del base_params, base_model
        if base_metrics:
            _emit("BASE_EVAL", base_metrics)
            print(f"  Base: call_f1={base_metrics['call_f1']:.1%}, "
                  f"exact={base_metrics['exact_match']:.1%}")

        # Training (val_ds used for checkpoint selection inside train())
        # Architecture from checkpoint config
        print("Starting training...")
        approx_steps = max(1, (train_kept // args.batch_size) * args.epochs)
        cfg = ckpt_config
        train_args_options = dict(
            name=experiment_name,
            checkpoint=args.checkpoint,
            init_from=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=3e-5,
            muon_lr=0.02,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg.get("num_kv_heads", cfg["num_heads"]),
            num_layers=cfg["num_encoder_layers"],
            num_dec_layers=cfg["num_decoder_layers"],
            d_ff=cfg.get("d_ff", cfg["d_model"] * 4),
            max_enc_len=max_enc_len,
            max_dec_len=max_dec_len,
            max_samples=None,
            warmup_ratio=0.05,
            decay_ratio=0.05,
            wandb=False,
            dtype=cfg.get("dtype", "bfloat16"),
            checkpoint_dir=args.checkpoint_dir,
            seed=42,
            eval_every=max(1, approx_steps),
            max_eval_samples=min(val_kept, 50),
            contrastive_weight=0.1,
            contrastive_dim=cfg.get("contrastive_dim", 128),
            num_memory_slots=cfg.get("num_memory_slots", 64),
            w_name=2.0,
            w_value=4.0,
            w_key=1.5,
            val_ds=val_ds,
            lora_rank=getattr(args, "lora_rank", 0),
            lora_alpha=getattr(args, "lora_alpha", 16),
        )

        if getattr(args, "lora_rank", 0) > 0:
            train_args_options["init_from"] = args.checkpoint
            train_args_options["checkpoint"] = None

        train_args = argparse.Namespace(**train_args_options)

        from ..training.train import train
        train(train_args)
        best_path = _ensure_best_checkpoint(args.checkpoint_dir, run_id)

        # Finetuned model eval on TEST set (same held-out set as base eval)
        if best_path and os.path.exists(best_path) and test_examples:
            print(f"Evaluating finetuned model on {len(test_examples)} test examples...")
            ft_params, ft_config = load_checkpoint(best_path)
            ft_model = SimpleAttentionNetwork(ft_config)
            ft_metrics = _quick_tool_eval(
                ft_model, ft_params, tokenizer, test_examples,
                max_gen_len=min(max_dec_len, 512), max_enc_len=max_enc_len,
            )
            del ft_params, ft_model
            if ft_metrics:
                _emit("FINETUNED_EVAL", ft_metrics)
                print(f"  Finetuned: call_f1={ft_metrics['call_f1']:.1%}, "
                      f"exact={ft_metrics['exact_match']:.1%}")
    finally:
        data_mod.CACHE_DIR = original_cache_dir
        if _owns_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_path")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--max-enc-len", type=int, default=None)
    p.add_argument("--max-dec-len", type=int, default=None)
    finetune_local(p.parse_args())
