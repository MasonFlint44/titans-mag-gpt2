"""Phase 5.2 — perplexity baseline parity vs HF GPT-2 (G156).

With NMM zeroed (out_scale=0) and N_p=0, our perplexity must be within
5% of HF GPT-2's on the same text — same arithmetic-equivalence
invariant as the logit parity test, integrated over a token sequence.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from config import TitansConfig
from data.tokenizer import Tokenizer
from eval import perplexity
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained


@pytest.fixture(scope="module")
def hf_gpt2_small():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


@pytest.fixture(scope="module")
def loaded_titans():
    cfg = TitansConfig.gpt2_small(nmm_n_persistent=0)
    model = TitansMAGGPT2(cfg)
    load_pretrained(model, cfg)
    return model, cfg


def _hf_perplexity_on_ids(hf_model, idx: torch.Tensor) -> float:
    """HF GPT-2 reference perplexity. Aggregates NLL with reduction='sum'."""
    hf_model.eval()
    with torch.no_grad():
        out = hf_model(idx)
    logits = out.logits
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        idx[:, 1:].reshape(-1),
        reduction="sum",
    )
    return math.exp(nll.item() / idx[:, 1:].numel())


@pytest.mark.slow
def test_perplexity_within_5pct_of_HF_on_tiny_text(loaded_titans, hf_gpt2_small):
    model, cfg = loaded_titans
    tok = Tokenizer()
    # Small text so the NMM forward stays in CPU-tractable range.
    text = "The quick brown fox jumps over the lazy dog."
    ids = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)

    # Our perplexity via the loader-shaped path.
    db = torch.zeros_like(ids, dtype=torch.bool)
    db[:, 0] = True
    ours = perplexity(model, [(ids, db)], torch.device("cpu"))

    theirs = _hf_perplexity_on_ids(hf_gpt2_small, ids)

    # 5% relative tolerance per G156.
    rel = abs(ours - theirs) / theirs
    assert rel < 0.05, (
        f"perplexity drift > 5%: ours={ours:.4f}, theirs={theirs:.4f}, "
        f"rel={rel:.4%}"
    )
