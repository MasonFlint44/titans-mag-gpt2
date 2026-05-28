"""Behavior test: NMM learns associative recall on a synthetic key->value task.

docs/TEST_PLAN.md §10 spec is slow_gpu (500-step train). This CPU-feasible version
trains a tiny model on a repeating K->V pattern and verifies the model's
prediction probability of V given K improves substantially over training —
demonstrating the NMM is actually updating memory in a useful direction.

A pure overfit (test_loss_decreases_on_overfit_batch) shows ANY learning;
this test specifically probes the K->V associative property.
"""

import pytest
import torch
import torch.nn.functional as F

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from model.titans_gpt2 import TitansMAGGPT2
from cli.train import build_optimizer, run_training


@pytest.mark.slow
def test_nmm_learns_K_to_V_association_via_training():
    torch.manual_seed(0)
    VOCAB = 32
    K_TOKEN = 5
    V_TOKEN = 7

    cfg = TitansConfig(
        n_layer=1, n_head=2, n_embd=16, vocab_size=VOCAB,
        block_size=64, chunk_size=8, dropout=0.0,
        nmm_expansion=2, nmm_n_persistent=2,
        finetune_mode=False,
    )
    model = TitansMAGGPT2(cfg)
    optimizer = build_optimizer(model)

    # Synthetic corpus: repeated [..., K, V, ...] with K_TOKEN ALWAYS followed
    # by V_TOKEN. Filler tokens are random in [1, VOCAB) (avoid 0 = EOT-like).
    # We never insert EOT (eot_id=50256 is outside vocab so no resets fire).
    def make_stream(n_chunks: int, chunk_len: int) -> torch.Tensor:
        total = n_chunks * 4 * chunk_len  # 4 = batch_size below
        tokens = torch.randint(1, VOCAB, (total,))
        # Insert K->V pairs at regular intervals (every 6 tokens).
        for i in range(0, total - 1, 6):
            tokens[i] = K_TOKEN
            tokens[i + 1] = V_TOKEN
        return tokens

    train_stream = make_stream(n_chunks=8, chunk_len=8)
    loader = ParallelStreamLoader(
        train_stream, batch_size=4, chunk_size=8, eot_id=99999,
    )

    # --- Baseline (untrained) probability of V_TOKEN given K_TOKEN ---
    def k_to_v_prob():
        """Average P(next = V_TOKEN | preceding token = K_TOKEN) across a
        fresh evaluation stream."""
        eval_stream = make_stream(n_chunks=2, chunk_len=8)
        eval_loader = ParallelStreamLoader(
            eval_stream, batch_size=4, chunk_size=8, eot_id=99999,
        )
        model.eval()
        total_log_prob = 0.0
        total_count = 0
        nmm_states = None
        with torch.no_grad():
            for input_ids, db in eval_loader:
                logits, nmm_states = model(input_ids, nmm_states, db)
                # Find all (b, t) where input_ids[b, t] == K_TOKEN and t+1 < T.
                T = input_ids.shape[1]
                for b in range(input_ids.shape[0]):
                    for t in range(T - 1):
                        if input_ids[b, t].item() == K_TOKEN:
                            log_probs = F.log_softmax(logits[b, t], dim=-1)
                            total_log_prob += log_probs[V_TOKEN].item()
                            total_count += 1
        model.train()
        return total_log_prob / total_count if total_count else float("nan")

    baseline_log_prob = k_to_v_prob()
    # log(1/VOCAB) is the random baseline.
    random_baseline = -torch.log(torch.tensor(float(VOCAB))).item()

    # --- Train ---
    # 200 steps lands trained_log_prob ~0.65 nats above random in the
    # smoke run (init -3.56 -> trained -2.81 vs random -3.47); 120 steps
    # only got ~0.43 nats which was below the 0.5-nat meaningfulness bar.
    run_training(
        model=model, optimizer=optimizer, loader=loader,
        device=torch.device("cpu"),
        max_steps=200, warmup_steps=10,
        log_every=1000,
    )

    trained_log_prob = k_to_v_prob()

    # The model must learn the K->V association: trained log-prob > baseline.
    assert trained_log_prob > baseline_log_prob, (
        f"K->V log-prob did not improve: baseline={baseline_log_prob:.3f}, "
        f"trained={trained_log_prob:.3f}, random={random_baseline:.3f}"
    )
    # And meaningfully beat the random uniform baseline.
    assert trained_log_prob > random_baseline + 0.5, (
        f"K->V log-prob not meaningfully above random: "
        f"trained={trained_log_prob:.3f}, random={random_baseline:.3f}"
    )
