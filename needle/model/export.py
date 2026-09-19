"""Export a trained checkpoint to a `.cact` blob (needle.cact) - the deployment
weights the C++ / single-mega-kernel inference reads (mmap or linked read-only
section; no parsing).

Format is CQ-WxA8: matmul weights use the per-tensor CQ width selected by the
deployment map; norms, Hadamard diagonals, and gates stay FP16. Weights
are PRE-TRANSPOSED to [out, in] so each output row is contiguous along the
reduction axis (what a GEMV/GEMM stream wants, and the axis quant groups run
along). Tensors are LAYER-MAJOR: embedding, then all of layer 0's tensors, ...,
then final norm - so the kernel's working set for one layer is contiguous.

================================ byte layout ================================
All little-endian. A fixed 120-byte header carrying the full architecture
geometry, then a NAMELESS tensor directory, then 64-byte aligned tensor blobs.
The runtime reads the geometry from the header, so one binary loads and runs
any configuration of the (single) architecture; tensors are positional in the
fixed canon order below.

HEADER  (49 u32-sized fields, rope_theta an f32, everything else u32):
  tag, num_tensors, codebook_len, kv_window, kv_bits,
  vocab, out_vocab, d_model, num_heads, num_kv_heads, num_layers,
  qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes,
  sliding_window, global_mask_lo, global_mask_hi, qkv_conv_taps,
  engram_slots, engram_sub_dim, num_engram_tables, engram_conv_taps,
  engram_conv_dilation, engram_seed_heads, num_engram_orders,
  engram_orders[4], num_engram_sites, engram_sites[16], rope_theta.
  out_vocab is the tied text-slice head (0 = full vocab); rows past out_vocab
  are input-only code embeddings. sliding_window is the per-layer local width;
  global_mask bits (layer i = bit i, lo then hi u32) mark full-attention
  layers. qkv_conv_taps is the causal depthwise conv width on Q,K,V (0 = none);
then codebook_len * f32 - the shared Lloyd-Max unit-sphere codebooks,
cb2[4] | cb3[8] | cb4[16] concatenated, the layout the runtime indexes by
width. kv_window is the sliding-window width the model was trained with (0 =
size it from the KV budget alone); kv_bits is the KV-cache width it was
post-trained for (8 = int8, else 2/3/4). Both ride in the blob because a model
quantized for one must be RUN at it - leaving either to a runtime flag silently
serves the wrong numerics.

DIRECTORY  (num_tensors records, {REC_SIZE} bytes each, no names):
  u8 dtype, u8 ndim, u16 _pad, u32 shape[4], u64 offset, u64 nbytes,
  u32 group_size, u32 bits.  dtype: 1=FP16, 2=FP32, 3=CQ (bits gives the
  width: 1, 2, 3 or 4; 5 denotes ternary crumbs; this package writes 4), 4=RAW.

TENSOR ORDER (what the runtime indexes by position):
  embedding; per layer [norm_in, q_proj, k_proj, v_proj,
  (q_taps, k_taps, v_taps when qkv_conv_taps > 0), q_norm, k_norm,
  gate_proj, out_proj, post_norm, attn_gate, pre_hada, d1, d2, b2, d3, d4,
  w1a, w1b, w2a, w2b, w3a, w3b, cond_v, cond_u]; the 9 mHC blocks; per engram
  site [tables, key_proj, value_proj, taps]; final_norm; the optional probe
  heads (see below); then the RAW tokenizer.

PROBE HEADS (appended so pre-head blobs stay loadable). Let
  extra = num_tensors - (through final_norm) - (tokenizer). extra == 0 means no
  heads. Otherwise the tensor right after final_norm is a `heads.manifest` of
  shape (H,), one code per head in canonical order (1 embedding,
  2 confidence, 3 router), then each head's tensors with its own k and q:
  [probes CQ ((L+1)*k, d_model), gain FP16 (L+1, k), query CQ (q, d_model),
  row_bias FP16 (q, L+1, k), proj CQ (out, q*d_model), bias FP16 (out,)],
  rows = embedding then blocks 0..L-1; the router appends calibration FP16 (3,).
  The three matrices are CQ at HEAD_BITS with the model group size (the head
  trained on them under QAT); the runtime also accepts the earlier all-FP16
  layout with probes shaped (L+1, k, d_model). The walk consumes exactly
  extra - 1 tensors.

CQ blob for a logical [out, in] matrix (in padded up to a multiple of group):
first the packed indices (out * in_pad*bits/8 bytes), then the per-group L2
norms (out * in_pad/group  FP16). Indices are packed LSB-first in chunks of 8:
each run of 8 consecutive indices is OR'd into a word at offsets i*bits and
emitted as `bits` little-endian bytes. A chunk is a whole number of bytes, so
this is equivalently one continuous LSB-first bitstream per row (index k
occupies bits [k*bits, (k+1)*bits)), which is what the runtime's extractor
assumes; at bits=4 it is the classic nibble packing (low nibble = even column).
Reconstruct per group:
  w_group = (codebook_bits[idx] * norm) @ H  # H = normalized Walsh-Hadamard(group)

TERNARY (record bits=5, 3 levels) stores four signed 2-bit crumbs per byte. The
crumb codes are 3, 0, 1 for trit indices 0, 1, 2, so sign-extending each 2-bit
field directly produces -1, 0, +1 in the NEON kernel. The codebook is not
in the header: the 3-level Lloyd-Max centroids {-c, 0, +c} / sqrt(group),
c = 1.2240064, are analytic.

BINARY (record bits=1, 2 levels) stores eight indices per byte. Its codebook is
also analytic: {-sqrt(2/pi), +sqrt(2/pi)} / sqrt(group).

One RAW attachment follows the model tensors and heads: the self-contained
SentencePiece BPE dump. Its format (see RefTokenizer below for the reference
encoder/decoder this specifies):
  header  u32 n_pieces, u32 pad/eos/bos/unk id, u8 add_dummy_prefix,
          u8 byte_fallback, u16 _pad;
  then n_pieces records, id order: f32 score, u8 type
          (0 NORMAL, 1 UNKNOWN, 2 CONTROL, 3 USER_DEFINED, 4 BYTE),
          u16 surface_len, surface_len UTF-8 bytes.
=============================================================================
"""
import struct

import numpy as np

from .quantize import _cq_codebook_np, _cq_hadamard_np, TERNARY_BITS, WEIGHT_BITS

TAG = 0x05E12A84
ALIGN = 64
FP16, FP32, CQ, RAW = 1, 2, 3, 4
CB_BITS = (2, 3, 4)
TERNARY_RECORD_BITS = 5

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_HDR_FMT = "<48If"
_REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(_REC_FMT)


def _geometry(config):
    from .architecture import head_dims
    qk_hd, v_hd = head_dims(config)
    if config.num_layers > 64:
        raise NotImplementedError(
            "cact global_mask holds 64 layers; deeper stacks need another bump")
    if len(getattr(config, "engram_layers", ())) > 16:
        raise NotImplementedError(
            "cact header holds 16 engram sites; denser engram needs another bump")
    orders = tuple(getattr(config, "engram_orders", (2, 3)))
    heads = getattr(config, "engram_heads", 0) or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    sites = tuple(getattr(config, "engram_layers", (2, 15)))
    return qk_hd, v_hd, orders, heads, sub_dim, sites


ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4


def _align(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _nearest_idx(x, cb):
    flat = x.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos)
    return idx.reshape(x.shape).astype(np.uint8)


def _pack_lsb(idx, bits):
    out, in_pad = idx.shape
    g = idx.reshape(out, in_pad // 8, 8).astype(np.uint64)
    word = np.zeros(g.shape[:-1], np.uint64)
    for i in range(8):
        word |= g[..., i] << (i * bits)
    packed = np.empty(word.shape + (bits,), np.uint8)
    for b in range(bits):
        packed[..., b] = (word >> (8 * b)) & 0xFF
    return packed.reshape(out, in_pad * bits // 8)


def _unpack_lsb(packed, bits, in_pad):
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << (8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> (i * bits)) & mask
    return idx.reshape(out, in_pad)


def _packed_row_bytes(in_pad, bits, group):
    if bits == TERNARY_RECORD_BITS:
        return in_pad * 2 // 8
    return in_pad * bits // 8


def _unpack_ternary_crumbs(packed, in_pad):
    crumbs = _unpack_lsb(packed, 2, in_pad)
    return np.where(crumbs == 3, 0, crumbs + 1).astype(np.uint8)


def _cq_pack(w, bits, group):
    if bits != WEIGHT_BITS:
        raise ValueError(f"CQ packing supports bits={WEIGHT_BITS}; got bits={bits}")
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    in_pad = wp.shape[1]
    g = wp.reshape(out, in_pad // group, group).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    idx = _nearest_idx(unit, cb).reshape(out, in_pad)
    return _pack_lsb(idx, bits), norm[:, :, 0].astype(np.float16)


def _cq_unpack(packed, norms, out, in_dim, bits, group):
    record_bits = bits
    if bits == TERNARY_RECORD_BITS:
        bits = TERNARY_BITS
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    in_pad = (in_dim + group - 1) // group * group
    if record_bits == TERNARY_RECORD_BITS:
        idx = _unpack_ternary_crumbs(packed, in_pad)
    else:
        idx = _unpack_lsb(packed, bits, in_pad)
    unit = cb[idx].reshape(out, in_pad // group, group)
    rot = unit * norms.reshape(out, in_pad // group, 1).astype(np.float32)
    w = (rot @ H).reshape(out, in_pad)
    return w[:, :in_dim]


class _Tensor:
    __slots__ = ("name", "dtype", "shape", "blob", "group", "bits", "offset")

    def __init__(self, name, dtype, shape, blob, group=0, bits=0):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)
        self.blob, self.group, self.bits, self.offset = blob, group, bits, 0


def _fp16(name, arr):
    arr = np.asarray(arr, np.float16)
    return _Tensor(name, FP16, arr.shape, arr.tobytes())


def _q(name, mat, bits, group):
    packed, norms = _cq_pack(np.asarray(mat, np.float32), bits, group)
    return _Tensor(name, CQ, mat.shape, packed.tobytes() + norms.tobytes(), group, bits)


def _tensors(params, config, bits, group):
    _, _, orders, heads, sub_dim, sites = _geometry(config)
    num_tables = len(orders) * heads
    embedding = np.asarray(_get(params, ("embedding", "embedding")))
    ts = [_q("embedding", embedding, bits, group)]

    taps_n = int(getattr(config, "qkv_conv_taps", 0))
    b = params["stack"]["layers"]["block"]
    sa, ha = b["self_attn"], b["hadamard_mlp"]
    for i in range(config.num_layers):
        ts += [
            _fp16(f"layer{i:02d}.norm_in", b["ZCRMSNorm_0"]["scale"][i]),
            _q(f"layer{i:02d}.q_proj", np.asarray(sa["q_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.k_proj", np.asarray(sa["k_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.v_proj", np.asarray(sa["v_proj"]["kernel"][i]).T, bits, group),
        ]
        if taps_n:
            ts += [
                _fp16(f"layer{i:02d}.q_taps", sa["q_taps"][i]),
                _fp16(f"layer{i:02d}.k_taps", sa["k_taps"][i]),
                _fp16(f"layer{i:02d}.v_taps", sa["v_taps"][i]),
            ]
        ts += [
            _fp16(f"layer{i:02d}.q_norm", sa["q_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.k_norm", sa["k_norm"]["scale"][i]),
            _q(f"layer{i:02d}.gate_proj", np.asarray(sa["gate_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.out_proj", np.asarray(sa["out_proj"]["kernel"][i]).T, bits, group),
            _fp16(f"layer{i:02d}.post_norm", b["post_attn_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.attn_gate", np.asarray(b["attn_gate"][i]).reshape(1)),
            _fp16(f"layer{i:02d}.pre_hada", b["pre_hada_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.d1", ha["d1"][i]),
            _fp16(f"layer{i:02d}.d2", ha["d2"][i]),
            _fp16(f"layer{i:02d}.b2", ha["b2"][i]),
            _fp16(f"layer{i:02d}.d3", ha["d3"][i]),
            _fp16(f"layer{i:02d}.d4", ha["d4"][i]),
            _fp16(f"layer{i:02d}.w1a", ha["w1a"][i]),
            _fp16(f"layer{i:02d}.w1b", ha["w1b"][i]),
            _fp16(f"layer{i:02d}.w2a", ha["w2a"][i]),
            _fp16(f"layer{i:02d}.w2b", ha["w2b"][i]),
            _fp16(f"layer{i:02d}.w3a", ha["w3a"][i]),
            _fp16(f"layer{i:02d}.w3b", ha["w3b"][i]),
            _fp16(f"layer{i:02d}.cond_v", ha["cond_v"][i]),
            _fp16(f"layer{i:02d}.cond_u", ha["cond_u"][i]),
        ]

    mhc = params["stack"]
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
                 "mhc_b_res"):
        ts.append(_fp16(name, np.asarray(mhc[name])))
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = np.asarray(mhc[name])
        L, nC, lanes = phi.shape
        ts.append(_q(name, phi.transpose(0, 2, 1).reshape(L * lanes, nC), bits, group))

    from .architecture import _hada_perms
    hada_n = 1 << (config.d_model - 1).bit_length()
    split = bool(getattr(config, "ladder_widths", ()))
    p1, p2 = _hada_perms(hada_n, split)
    ts.append(_Tensor("hada_p1", FP32, (hada_n,),
                      np.asarray(p1, np.float32).tobytes()))
    ts.append(_Tensor("hada_p2", FP32, (hada_n,),
                      np.asarray(p2, np.float32).tobytes()))

    for s in range(len(sites)):
        eg = params[f"engrams_{s}"]
        tables = np.asarray(eg["embedding"]).reshape(num_tables * config.engram_slots, sub_dim)
        ts += [
            _q(f"engram{s}.tables", tables, bits, group),
            _q(f"engram{s}.key_proj", np.asarray(eg["key_proj"]["kernel"]).T, bits, group),
            _q(f"engram{s}.value_proj", np.asarray(eg["value_proj"]["kernel"]).T, bits, group),
            _fp16(f"engram{s}.taps", np.asarray(eg["taps"])),
        ]

    ts.append(_fp16("final_norm", params["stack"]["final_norm"]["scale"]))
    ts += _head_tensors(params, group)
    return ts


HEAD_MATRICES = ("probes", "query", "proj")


def _head_tensors(params, group):
    from .architecture import HEADS
    from .quantize import HEAD_BITS
    present = [head for head in HEADS if head.key in params]
    if not present:
        return []
    ts = [_fp16("heads.manifest", np.asarray([head.code for head in present], np.float16))]
    for head in present:
        for name, value in head.export(params[head.key]):
            if name.rsplit(".", 1)[-1] in HEAD_MATRICES:
                matrix = np.asarray(value, np.float32).reshape(-1, value.shape[-1])
                ts.append(_q(name, matrix, HEAD_BITS, group))
            else:
                ts.append(_fp16(name, value))
    return ts


def _get(params, keys):
    node = params
    for k in keys:
        node = node[k]
    return node


def _tokenizer_blob(tok):
    from .tokenizer import CHAT_MARKERS, PAD_ID, EOS_ID, BOS_ID, UNK_ID
    sp = tok.sp
    n = sp.GetPieceSize()
    markers = set(CHAT_MARKERS)
    dummy = 1 if sp.IdToPiece(sp.Encode("a", out_type=int)[0]).startswith(_SP_META_SPACE) else 0
    out = bytearray(struct.pack(_TK_HDR, n, PAD_ID, EOS_ID, BOS_ID, UNK_ID, dummy, 1, 0))
    for i in range(n):
        piece = sp.IdToPiece(i)
        if sp.IsControl(i):
            t = TK_CONTROL
        elif sp.IsUnknown(i):
            t = TK_UNKNOWN
        elif sp.IsByte(i):
            t = TK_BYTE
        elif piece in markers:
            t = TK_USER_DEFINED
        else:
            t = TK_NORMAL
        b = piece.encode("utf-8")
        out += struct.pack(_TK_REC, sp.GetScore(i), t, len(b)) + b
    return bytes(out)


def _tokenizer_pieces(blob):
    return struct.unpack_from(_TK_HDR, blob, 0)[0]


def read_layers(path):
    with open(path, "rb") as f:
        hdr = struct.unpack(_HDR_FMT, f.read(struct.calcsize(_HDR_FMT)))
    if hdr[0] != TAG:
        raise ValueError(f"{path} is not a Needle 3 .cact archive")
    return hdr[10]


def read_tokenizer_blob(path):
    """The packaged tokenizer of a .cact archive, read without unpacking its weights."""
    with open(path, "rb") as f:
        raw = f.read()
    tag, num_tensors, cb_n = struct.unpack_from(_HDR_FMT, raw, 0)[:3]
    if tag != TAG:
        raise ValueError(f"{path} is not a Needle 3 .cact archive")
    off = struct.calcsize(_HDR_FMT) + cb_n * 4
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE]); off += REC_SIZE
        if rec[0] == RAW:
            return raw[rec[7]:rec[7] + rec[8]]
    raise ValueError(f"{path} carries no tokenizer")


def _pack_cact(params, config, bits, group, tokenizer, kv_window=0):
    params = {k: v for k, v in params.items()}
    ts = _tensors(params, config, bits, group)
    vocab = int(config.vocab_size)
    out_vocab = int(getattr(config, "out_vocab", 0) or 0)
    if out_vocab > vocab:
        raise ValueError(f"out_vocab {out_vocab} exceeds the exported vocab {vocab}")
    if tokenizer is not None:
        blob = tokenizer if isinstance(tokenizer, (bytes, bytearray)) else _tokenizer_blob(tokenizer)
        pieces = _tokenizer_pieces(blob)
        if pieces > vocab:
            raise ValueError(f"tokenizer vocab {pieces} > exported vocab {vocab}")
        ts.append(_Tensor("tokenizer", RAW, (), bytes(blob)))
    cb = np.concatenate([_cq_codebook_np(b, group) for b in CB_BITS]).astype(np.float32)
    kv_bits = int(getattr(config, "kv_bits", 8) or 8)
    qk_hd, v_hd, orders, heads, sub_dim, sites = _geometry(config)
    hada_n = 1 << (config.d_model - 1).bit_length()
    orders4 = (list(orders) + [0, 0, 0, 0])[:4]
    sites16 = (list(sites) + [0] * 16)[:16]
    gmask = 0
    for g in getattr(config, "global_layers", ()) or ():
        gmask |= 1 << int(g)
    header = struct.pack(
        _HDR_FMT, TAG, len(ts), len(cb), int(kv_window or 0), kv_bits,
        vocab, out_vocab,
        config.d_model, config.num_heads, config.num_kv_heads,
        config.num_layers, qk_hd, v_hd, config.max_seq_len, hada_n,
        config.mhc_lanes, int(getattr(config, "sliding_window", 0) or 0),
        gmask & 0xFFFFFFFF, (gmask >> 32) & 0xFFFFFFFF,
        int(getattr(config, "qkv_conv_taps", 0) or 0),
        config.engram_slots, sub_dim, len(orders) * heads,
        ENGRAM_CONV_TAPS, max(orders),
        int(getattr(config, "engram_seed_heads", 0) or 0),
        len(orders), *orders4, len(sites), *sites16,
        float(config.rope_theta)) + cb.tobytes()

    pos = len(header) + len(ts) * REC_SIZE
    for t in ts:
        pos = _align(pos)
        t.offset = pos
        pos += len(t.blob)

    directory = b"".join(
        struct.pack(_REC_FMT, t.dtype, len(t.shape), 0,
                    *(list(t.shape) + [0, 0, 0, 0])[:4], t.offset, len(t.blob),
                    t.group, t.bits)
        for t in ts
    )

    buf = bytearray(header + directory)
    for t in ts:
        buf.extend(b"\x00" * (t.offset - len(buf)))
        buf.extend(t.blob)
    return bytes(buf), len(ts)



def write_export(params, config, path, bits=WEIGHT_BITS, group=128, tokenizer=None,
                 kv_window=0):
    buf, n = _pack_cact(params, config, bits, group, tokenizer, kv_window)
    with open(path, "wb") as f:
        f.write(buf)
    return {"path": path, "bytes": len(buf), "tensors": n}


def read_export(path):
    with open(path, "rb") as f:
        raw = f.read()
    hdr = struct.unpack_from(_HDR_FMT, raw, 0)
    tag, num_tensors, cb_n, kv_window, kv_bits = hdr[:5]
    assert tag == TAG, "bad tag"
    geometry = dict(zip(
        ("vocab_size", "out_vocab", "d_model", "num_heads", "num_kv_heads",
         "num_layers", "qk_head_dim", "v_head_dim", "max_seq_len", "hada_n",
         "mhc_lanes", "sliding_window", "global_mask_lo", "global_mask_hi",
         "qkv_conv_taps", "engram_slots", "engram_sub_dim",
         "num_engram_tables", "engram_conv_taps", "engram_conv_dilation",
         "engram_seed_heads"), hdr[5:26]))
    gmask = geometry.pop("global_mask_lo") | (geometry.pop("global_mask_hi") << 32)
    geometry["global_layers"] = tuple(i for i in range(geometry["num_layers"])
                                      if gmask >> i & 1)
    num_orders, orders4 = hdr[26], hdr[27:31]
    num_sites, sites16 = hdr[31], hdr[32:48]
    geometry["engram_orders"] = tuple(orders4[:num_orders])
    geometry["engram_layers"] = tuple(sites16[:num_sites])
    geometry["rope_theta"] = hdr[48]
    off = struct.calcsize(_HDR_FMT)
    codebook = np.frombuffer(raw[off:off + cb_n * 4], np.float32).copy()
    off += cb_n * 4

    tensors = []
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE]); off += REC_SIZE
        dtype, ndim = rec[0], rec[1]
        shape = tuple(rec[3:3 + ndim])
        offset, nbytes, group, bits = rec[7], rec[8], rec[9], rec[10]
        blob = raw[offset:offset + nbytes]
        if dtype == FP16:
            tensors.append(np.frombuffer(blob, np.float16).reshape(shape).astype(np.float32))
        elif dtype == FP32:
            tensors.append(np.frombuffer(blob, np.float32).reshape(shape))
        elif dtype == CQ:
            out, in_dim = shape
            in_pad = (in_dim + group - 1) // group * group
            n_packed = out * _packed_row_bytes(in_pad, bits, group)
            packed = np.frombuffer(blob[:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(blob[n_packed:], np.float16).reshape(out, in_pad // group)
            tensors.append(_cq_unpack(packed, norms, out, in_dim, bits, group))
        elif dtype == RAW:
            tensors.append(blob)
    return {"num_tensors": num_tensors, "codebook": codebook,
            "kv_window": kv_window, "kv_bits": kv_bits, **geometry}, tensors


_SP_META_SPACE = "▁"


def parse_tokenizer_blob(blob):
    off = struct.calcsize(_TK_HDR)
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
    rec = struct.calcsize(_TK_REC)
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, t, ln = struct.unpack_from(_TK_REC, blob, off)
        off += rec
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    return {"pieces": pieces, "scores": scores, "types": types, "pad_id": pad,
            "eos_id": eos, "bos_id": bos, "unk_id": unk,
            "add_dummy_prefix": bool(add_dummy), "byte_fallback": bool(byte_fb)}


class RefTokenizer:
    def __init__(self, meta):
        self.pieces = meta["pieces"]
        self.scores = meta["scores"]
        self.types = meta["types"]
        self.add_dummy = meta["add_dummy_prefix"]
        self.byte_fallback = meta["byte_fallback"]
        self.unk_id = meta["unk_id"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i
                        for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types) if t == TK_USER_DEFINED),
                              key=len, reverse=True)

    @classmethod
    def from_cact(cls, path):
        _, ts = read_export(path)
        blob = next(t for t in ts if isinstance(t, (bytes, bytearray)))
        return cls(parse_tokenizer_blob(blob))

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best_score, best_j = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best_score is None or self.scores[idx] > best_score):
                    best_score, best_j = self.scores[idx], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf)); buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i]); i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text
