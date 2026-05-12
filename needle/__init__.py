"""Needle - a 26M parameter function calling model."""

from needle.model.architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
)
from needle.model.run import (
    generate,
    generate_batch,
    load_checkpoint,
    encode_for_retrieval,
    retrieve_tools,
)
from needle.dataset.dataset import get_tokenizer

__all__ = [
    "SimpleAttentionNetwork",
    "TransformerConfig",
    "generate",
    "generate_batch",
    "load_checkpoint",
    "encode_for_retrieval",
    "retrieve_tools",
    "get_tokenizer",
]
