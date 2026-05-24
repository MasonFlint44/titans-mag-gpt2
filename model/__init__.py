"""TITANS MAG model components."""


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
