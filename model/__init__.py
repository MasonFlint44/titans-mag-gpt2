"""TITANS MAG model components."""

from model.block import (
    CausalSelfAttention,
    GPT2MLP,
    KVCacheInt8,
    PlainGPT2Block,
    TitansMAGBlock,
)
from model.nmm import (
    MemoryMLP,
    MultiHeadNMM,
    NeuralMemoryModule,
    detach_states,
)
from model.state_io import (
    StateConfigMismatch,
    load_nmm_state,
    save_nmm_state,
)
from model.titans_gpt2 import TitansMAGGPT2

__all__ = [
    "CausalSelfAttention",
    "GPT2MLP",
    "KVCacheInt8",
    "MemoryMLP",
    "MultiHeadNMM",
    "NeuralMemoryModule",
    "PlainGPT2Block",
    "StateConfigMismatch",
    "TitansMAGBlock",
    "TitansMAGGPT2",
    "_unwrap",
    "detach_states",
    "load_nmm_state",
    "save_nmm_state",
]


def _unwrap(m):
    """Return the underlying nn.Module behind torch.compile / DDP / FSDP wrappers.

    torch.compile prefixes state_dict keys with `_orig_mod.`; DDP/FSDP prefix
    with `module.`. Stacked wrappers (DDP(torch.compile(model))) get both.
    Saving an unwrapped state_dict makes resume work against an uncompiled,
    non-DDP-wrapped model unconditionally — the most flexible layout.
    """
    while hasattr(m, "module") or hasattr(m, "_orig_mod"):
        m = getattr(m, "module", m)
        m = getattr(m, "_orig_mod", m)
    return m
