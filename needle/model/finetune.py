import concurrent.futures
import json
import os

from .checkpoints import read_adapter, write_adapter
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The jax metal plugin refuses to load against a newer PJRT API without this;
# it must be in the environment before jax initialises its backend, and this
# module is imported before any jax import on every training path.
if sys.platform == "darwin":
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

import numpy as np

from .tokenizer import (
    get_tokenizer, BOS_ID, EOS_ID, PAD_ID,
    IM_START, IM_END, THINK_START, THINK_END,
    TOOLS_START, TOOLS_END, TOOL_CALL_START, TOOL_CALL_END,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")


def _training_rng(seed):
    return np.random.default_rng(int(seed))


DEFAULT_BASE = "checkpoints/needle3.safetensors"

OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
DEFAULT_MODEL = "deepseek/deepseek-flash-latest"

_GEN_SYSTEM = (
    "You generate training data for a tool-calling and extraction model. Given a "
    "set of tool or record schemas, produce realistic, diverse inputs paired with "
    "the exact calls that satisfy them. Return only JSON."
)

_GEN_TEMPLATE = """Schemas available (JSON):
{tools}

Produce {n} varied examples as a JSON array. Each element is an object:
  {{"query": "<a natural user request to act on, or a passage of text to extract from>",
    "reasoning": "<one short line deriving each argument from its source span in the query>",
    "answers": [{{"name": "<schema name>", "arguments": {{...}}}}]}}

Rules:
- Use only the schemas above; arguments must match them exactly and contain only
  values evidenced in the query.
- For an action tool the query is a command. For a record/extraction schema (its
  fields describe an entity), the query is a natural passage that contains those
  fields and the call extracts them.
- Cover single-call, multi-call, and about {refusals} off-topic inputs that no
  schema can serve (for those, "answers" is []).
- Vary phrasing, values, and which schemas are used. Return ONLY the JSON array."""


def _openrouter(messages, model, api_key, temperature=0.9):
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": temperature}).encode("utf-8")
    request = urllib.request.Request(OPENROUTER_URL, data=payload, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cactus-compute/needle",
        "X-Title": "needle",
    })
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]


def _parse_array(text):
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in rows if isinstance(r, dict) and "query" in r and "answers" in r]


def generate_examples(tools, n=25, model=DEFAULT_MODEL, api_key=None, refusals=3):
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY to generate data")
    tools_json = tools if isinstance(tools, str) else json.dumps(tools, indent=2)
    prompt = _GEN_TEMPLATE.format(tools=tools_json, n=n, refusals=refusals)
    text = _openrouter([{"role": "system", "content": _GEN_SYSTEM},
                        {"role": "user", "content": prompt}], model, api_key)
    rows = _parse_array(text)
    for row in rows:
        row.setdefault("tools", tools if isinstance(tools, list) else json.loads(tools))
    return rows


def _dedup_key(example):
    answers = example.get("answers", example.get("function_calls", []))
    return (example.get("query", "").strip().lower(), json.dumps(answers, sort_keys=True))


def generate_dataset(tools, num_samples, model=DEFAULT_MODEL, batch_size=25,
                     api_key=None, workers=8, progress=None):
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENROUTER_API_KEY to generate data")

    target = int(num_samples * 1.3)
    max_submissions = max(1, target // batch_size * 3)
    seen, rows, failed, submitted = set(), [], 0, 0
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = set()

    def _submit():
        nonlocal submitted
        pending.add(pool.submit(generate_examples, tools, batch_size,
                                model=model, api_key=api_key))
        submitted += 1

    for _ in range(min(workers, max(1, -(-target // batch_size)))):
        _submit()

    try:
        while pending and len(rows) < num_samples:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    for example in future.result():
                        key = _dedup_key(example)
                        if key not in seen:
                            seen.add(key)
                            rows.append(example)
                except Exception as exc:
                    failed += 1
                    print(f"  {'failed':<9} {exc}", flush=True)
                if len(rows) < num_samples and submitted < max_submissions:
                    _submit()
            done_count = min(len(rows), num_samples)
            if progress:
                progress(done_count, num_samples)
            else:
                print(f"  {'generated':<9} {done_count}/{num_samples}  failed {failed}", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return rows[:num_samples]


def _collect_tools(examples):
    seen, tools = set(), []
    for example in examples:
        for tool in example.get("tools", []):
            name = tool.get("name")
            if name and name not in seen:
                seen.add(name)
                tools.append(tool)
    return tools


def augment_jsonl(path, num_samples, model=DEFAULT_MODEL, batch_size=25, out_path=None, workers=8):
    with open(path) as handle:
        examples = [json.loads(line) for line in handle if line.strip()]
    tools = _collect_tools(examples)
    if not tools:
        raise RuntimeError("no tool schemas found in " + path)
    out_path = out_path or path.replace(".jsonl", "") + ".augmented.jsonl"
    generated = generate_dataset(tools, num_samples, model=model or DEFAULT_MODEL,
                                 batch_size=batch_size, workers=workers)
    with open(out_path, "w") as handle:
        for example in examples + generated:
            handle.write(json.dumps(example) + "\n")
    print(f"  {'wrote':<9} {len(examples) + len(generated)} examples  {out_path}")
    return out_path


def generate_main(args):
    model = args.model or DEFAULT_MODEL
    workers = getattr(args, "workers", 8)
    if args.tools:
        with open(args.tools) as handle:
            tools = json.load(handle)
        out = args.output or "needle_data.jsonl"
        rows = generate_dataset(tools, args.num_samples, model=model,
                                batch_size=args.batch_size, workers=workers)
        with open(out, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"  {'wrote':<9} {len(rows)} examples  {out}")
    elif args.augment:
        augment_jsonl(args.augment, args.num_samples, model=model,
                      batch_size=args.batch_size, out_path=args.output, workers=workers)
    else:
        raise SystemExit("pass --tools <schemas.json> or --augment <data.jsonl>")


def render_example(example):
    tools = example.get("tools", [])
    tools_json = tools if isinstance(tools, str) else json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    answers = example.get("answers", example.get("function_calls", []))
    answers_json = answers if isinstance(answers, str) else json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
    reasoning = (example.get("reasoning") or "").strip()
    system = (example.get("system") or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    prompt = (prefix + IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
              + example["query"] + IM_END + "\n" + IM_START + "assistant\n")
    think = THINK_START + "\n" + reasoning + "\n" + THINK_END + "\n" if reasoning else ""
    target = think + TOOL_CALL_START + answers_json + TOOL_CALL_END + IM_END
    return prompt, target


def _encode(tokenizer, example, max_len):
    prompt, target = render_example(example)
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    ids = [BOS_ID] + prompt_ids + target_ids + [EOS_ID]
    mask = [0.0] * (1 + len(prompt_ids)) + [1.0] * (len(target_ids) + 1)
    ids, mask = ids[:max_len], mask[:max_len]
    pad = max_len - len(ids)
    return ids + [PAD_ID] * pad, mask + [0.0] * pad


def fit_max_len(path, tokenizer, cap):
    longest = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if "query" not in example:
                continue
            prompt, target = render_example(example)
            longest = max(longest, len(tokenizer.encode(prompt)) + len(tokenizer.encode(target)) + 2)
    bucket = 128
    while bucket < min(longest, cap):
        bucket *= 2
    return min(bucket, cap)


def load_jsonl(path, tokenizer, max_len):
    seqs, masks = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if "query" not in example:
                continue
            ids, mask = _encode(tokenizer, example, max_len)
            seqs.append(ids)
            masks.append(mask)
    return np.array(seqs, np.int32), np.array(masks, np.float32)



def rung(params, config, layers):
    from .architecture import ladder_config, ladder_slice

    fields = dict(config.__dict__)
    fields.update(ladder_depths=(), ladder_sample=False, ladder_widths=())
    config = type(config)(**fields)
    if not layers or layers == config.num_layers:
        return params, config
    if not 2 <= layers < config.num_layers:
        raise ValueError(f"--layers must be in [2, {config.num_layers}], got {layers}")
    return ladder_slice(params, config, layers), ladder_config(config, layers)


def lora_target_paths(params):
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    paths = [path for path in flat
             if path[-1] == "kernel"
             and "stack" in path
             and "layers" in path
             and any(t in path for t in LORA_TARGETS)]
    paths = [p for p in paths if jnp.max(jnp.abs(flat[p])) > 1e-6]
    return paths


def init_lora(params, paths, rank, key):
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(params)
    lora = {}
    for path in paths:
        weight = flat[path]
        in_dim, out_dim = weight.shape[-2], weight.shape[-1]
        lead = weight.shape[:-2]
        key, sub = jax.random.split(key)
        lora[path] = {
            "A": jax.random.normal(sub, lead + (in_dim, rank), jnp.float32) / rank,
            "B": jnp.zeros(lead + (rank, out_dim), jnp.float32),
        }
    return lora


def merge_lora(params, lora, scale):
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict
    flat = dict(flatten_dict(params))
    for path, adapter in lora.items():
        flat[path] = flat[path] + (scale * jnp.matmul(adapter["A"], adapter["B"])).astype(flat[path].dtype)
    return unflatten_dict(flat)


def finetune_local(args, progress=None):
    import jax
    import jax.numpy as jnp
    import optax
    from .run import load_checkpoint
    from .architecture import SimpleAttentionNetwork
    from .quantize import configure_deploy, cq_ste_params, WEIGHT_BITS

    def emit(msg):
        print(msg, flush=True)
        if progress:
            progress(msg)

    base_path = args.checkpoint or DEFAULT_BASE
    data_path = args.jsonl_path
    if getattr(args, "generate", 0):
        data_path = augment_jsonl(data_path, args.generate, model=getattr(args, "model", None),
                                  workers=getattr(args, "workers", 8))

    params, config = load_checkpoint(base_path)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    if backend == "metal":
        config.flash = False
        config.remat = False
        config.scan_unroll = config.num_layers
    params = jax.device_put(params)
    emit(f"  {'backend':<9} {backend}  float32")
    emit(f"  {'depth':<9} {config.num_layers} layers")
    tokenizer = get_tokenizer(config.vocab_size)
    max_len = fit_max_len(data_path, tokenizer, args.max_len)
    seqs, masks = load_jsonl(data_path, tokenizer, max_len)
    if len(seqs) == 0:
        raise SystemExit("no usable examples in " + data_path)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}  cap {args.max_len}")

    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8),
                     kv_bits=getattr(config, "kv_bits", 8))
    emit(f"  {'numerics':<9} CQ W{WEIGHT_BITS} STE + A8 (matches export)")
    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    seed = int(getattr(args, "seed", 0))
    rng = _training_rng(seed)
    lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(seed))
    emit(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups")

    n_val = min(int(len(seqs) * getattr(args, "val_split", 0.1)), len(seqs) - 1)
    if n_val > 0:
        order = rng.permutation(len(seqs))
        seqs, masks = seqs[order], masks[order]
        val_seqs, val_masks = seqs[:n_val], masks[:n_val]
        seqs, masks = seqs[n_val:], masks[n_val:]
        emit(f"  {'holdout':<9} {n_val} examples for validation")

    batch, count = args.batch_size, len(seqs)
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), total_steps - 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup, decay_steps=total_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(lora)
    emit(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  cosine decay  clip 1.0  (compiling...)")

    def loss_fn(lora, ids, mask):
        merged = cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    @jax.jit
    def train_step(lora, opt_state, ids, mask):
        loss, grads = jax.value_and_grad(loss_fn)(lora, ids, mask)
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    every = max(1, total_steps // 50)
    step_i = 0
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        last = 0.0
        for start in range(0, count, batch):
            idx = order[start:start + batch]
            lora, opt_state, loss = train_step(lora, opt_state,
                                               jnp.asarray(seqs[idx]), jnp.asarray(masks[idx]))
            last = float(loss)
            step_i += 1
            if step_i % every == 0:
                emit(f"  {'step':<9} {step_i}/{total_steps}  loss {last:.4f}")
        if n_val > 0:
            val = np.mean([float(eval_step(lora, jnp.asarray(val_seqs[i:i + batch]),
                                           jnp.asarray(val_masks[i:i + batch])))
                           for i in range(0, n_val, batch)])
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}  val {val:.4f}")
        else:
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    out = args.out or os.path.join(args.checkpoint_dir, "needle_lora.safetensors")
    write_adapter(out, {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])}
                 for p, v in lora.items()},
        "scale": float(scale),
        "base": base_path,
        "rank": args.lora_rank,
        "seed": seed,
    })
    print(f"  {'adapter':<9} {out}")
    print(f"  {'next':<9} needle build {base_path} --lora {out}")
    print(f"  {'note':<9} confidence reports None with tuned weights; the head is not tuned")


def build_main(args):
    import shutil
    import jax.numpy as jnp
    from ..agent import fetch
    from .run import load_checkpoint
    from .architecture import effective_kv_window
    from .export import read_layers, read_tokenizer_blob, write_export
    from .quantize import WEIGHT_BITS

    base_archive = fetch.fetch_weights(3, force=True)
    print(f"  {'base':<9} {base_archive}  {os.path.getsize(base_archive) / 1e6:.2f} MB")
    platform = getattr(args, "platform", None)
    layers = getattr(args, "layers", None)
    folder = None
    if platform:
        folder = os.path.abspath(args.out or platform)
        for path in fetch.download_platform(platform, os.path.dirname(folder), generation=3, dest=folder):
            print(f"  {'engine':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        out = os.path.join(folder, fetch.base_weights(3))
    else:
        out = args.out
    if not args.lora and (not layers or layers == read_layers(base_archive)):
        if not out:
            raise SystemExit("pass --out <archive.cact>, or --platform to build a runnable folder")
        shutil.copyfile(base_archive, out)
        print(f"  {'wrote':<9} {out}  {os.path.getsize(out) / 1e6:.2f} MB  the published base archive")
    else:
        adapter = read_adapter(args.lora) if args.lora else None
        checkpoint = args.checkpoint
        if not checkpoint and adapter and adapter.get("base") and os.path.exists(adapter["base"]):
            checkpoint = adapter["base"]
        checkpoint = checkpoint or DEFAULT_BASE
        params, config, _ = load_checkpoint(checkpoint, return_run=True)
        if args.lora:
            lora = {tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                    for key, v in adapter["lora"].items()}
            params = merge_lora(params, lora, adapter["scale"])
            print(f"  {'merged':<9} {len(lora)} weight groups  {args.lora}")
        params, config = rung(params, config, layers)
        print(f"  {'depth':<9} {config.num_layers} layers")
        out = out or (os.path.splitext(os.path.basename(checkpoint))[0] + ".cact")
        info = write_export(params, config, out, bits=WEIGHT_BITS,
                            tokenizer=read_tokenizer_blob(base_archive),
                            kv_window=effective_kv_window(config))
        print(f"  {'wrote':<9} {info['path']}  {info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors  W{WEIGHT_BITS}A8")
    if folder:
        runner = next((n for n in ("needle", "needle.exe") if os.path.exists(os.path.join(folder, n))), None)
        if runner:
            print(f"  {'next':<9} {os.path.join(folder, runner)} --model {os.path.basename(out)} --tools tools.json --serve")
    else:
        print(f"  {'next':<9} needle.Needle(weights={out!r}, tools=[...])")

    if args.upload:
        repo = os.environ.get("NEEDLE_HF_REPO")
        if not repo:
            raise SystemExit("set NEEDLE_HF_REPO=<you>/<model> to upload")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo=os.path.basename(out),
                        repo_id=repo, repo_type="model")
        print(f"  {'uploaded':<9} {os.path.basename(out)}  {repo}")
