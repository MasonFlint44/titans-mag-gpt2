"""HF GPT-2 weight loader: transposes Conv1D layout and copies into TitansMAGGPT2.

HF GPT-2 uses a custom `Conv1D(out, in)` whose weight is `[in, out]` — the
transpose of `nn.Linear`. Every Conv1D weight (c_attn, c_proj, c_fc, mlp.c_proj)
must be transposed on copy. LayerNorm and Embedding layouts match — no transpose.

The fused HF `c_attn` (Q+K+V in one [n_embd, 3*n_embd] matrix) splits into our
three independent q/k/v_proj along dim=1 (out-dim).

NMM parameters keep their own inits from `NeuralMemoryModule.__init__` (out_scale
at zeros for finetune_mode=True, Xavier on memory_mlp.W*, etc.) — they are NOT
overwritten here. After loading, the model with N_p=0 and out_scale=0 must
produce logits within 1e-4 of HF GPT-2 (Phase 2.6 HARD GATE).
"""

import torch
from transformers import AutoModelForCausalLM


_HF_GPT2_NAMES = {
    768: "openai-community/gpt2",
    1024: "openai-community/gpt2-medium",
    1280: "openai-community/gpt2-large",
    1600: "openai-community/gpt2-xl",
}


def load_pretrained(model, config) -> None:
    """Overwrite `model`'s backbone params from HF GPT-2; NMM params untouched.

    Selects the HF checkpoint by `config.n_embd` (G216) — hard-coding the small
    variant would silently shape-mismatch with medium/large/xl configs.

    Args:
        model: a freshly constructed TitansMAGGPT2.
        config: the TitansConfig used to build it.
    """
    if config.n_embd not in _HF_GPT2_NAMES:
        raise ValueError(
            f"No HF GPT-2 checkpoint maps to n_embd={config.n_embd}. "
            f"Supported variants: {list(_HF_GPT2_NAMES)}. Use one of the "
            f"TitansConfig.gpt2_{{small,medium,large,xl}} factory methods."
        )
    hf_name = _HF_GPT2_NAMES[config.n_embd]
    hf_model = AutoModelForCausalLM.from_pretrained(hf_name)

    # Defensive: a TitansConfig override like gpt2_small(n_layer=14) would
    # silently zip-truncate; catch it loudly here.
    hf_n_layer = hf_model.config.n_layer
    hf_n_head = hf_model.config.n_head
    if hf_n_layer != config.n_layer:
        raise ValueError(
            f"Our n_layer={config.n_layer} but HF {hf_name} has "
            f"n_layer={hf_n_layer}. Cannot transfer weights — layer counts differ."
        )
    if hf_n_head != config.n_head:
        raise ValueError(
            f"Our n_head={config.n_head} but HF {hf_name} has "
            f"n_head={hf_n_head}. Head-count mismatch is incompatible with the QKV split."
        )

    with torch.no_grad():
        # --- Per-block copy ---
        for our_block, hf_block in zip(model.blocks, hf_model.transformer.h):
            our_attn = our_block.attn

            # Fused c_attn -> q/k/v split (out-dim = dim 1 of HF Conv1D weight).
            c_attn_w = hf_block.attn.c_attn.weight  # [n_embd, 3*n_embd]
            c_attn_b = hf_block.attn.c_attn.bias    # [3*n_embd]
            W_q, W_k, W_v = c_attn_w.chunk(3, dim=1)  # each [n_embd, n_embd]
            b_q, b_k, b_v = c_attn_b.chunk(3, dim=0)  # each [n_embd]
            # Conv1D weight [in, out] -> nn.Linear weight [out, in]: transpose.
            our_attn.q_proj.weight.copy_(W_q.T)
            our_attn.q_proj.bias.copy_(b_q)
            our_attn.k_proj.weight.copy_(W_k.T)
            our_attn.k_proj.bias.copy_(b_k)
            our_attn.v_proj.weight.copy_(W_v.T)
            our_attn.v_proj.bias.copy_(b_v)
            our_attn.proj.weight.copy_(hf_block.attn.c_proj.weight.T)
            our_attn.proj.bias.copy_(hf_block.attn.c_proj.bias)

            # LayerNorms — matching layout, no transpose.
            our_block.ln_1.weight.copy_(hf_block.ln_1.weight)
            our_block.ln_1.bias.copy_(hf_block.ln_1.bias)
            our_block.ln_2.weight.copy_(hf_block.ln_2.weight)
            our_block.ln_2.bias.copy_(hf_block.ln_2.bias)

            # MLP Conv1D -> Linear (transpose both).
            our_block.mlp.c_fc.weight.copy_(hf_block.mlp.c_fc.weight.T)
            our_block.mlp.c_fc.bias.copy_(hf_block.mlp.c_fc.bias)
            our_block.mlp.c_proj.weight.copy_(hf_block.mlp.c_proj.weight.T)
            our_block.mlp.c_proj.bias.copy_(hf_block.mlp.c_proj.bias)

        # --- Embeddings + final LayerNorm ---
        model.wte.weight.copy_(hf_model.transformer.wte.weight)
        model.wpe.weight.copy_(hf_model.transformer.wpe.weight)
        model.ln_f.weight.copy_(hf_model.transformer.ln_f.weight)
        model.ln_f.bias.copy_(hf_model.transformer.ln_f.bias)
