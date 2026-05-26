"""Autoregressive generation via KV-cache attention + single-token NMM step.

Option B from the audit: each decoded token gets exactly ONE NMM update
(matching the TITANS spec) instead of re-feeding the full sliding window
through the NMM at every step (the old behavior). See docs/PLAN.md §5.1 +
docs/RUNBOOK.md "Long-context generation drift" §4.

Pipeline:
  1. prepare_decode: chunked warm-up on the prompt (existing forward_chunk
     for the NMM; KV cache + conv buffer captured at the end).
  2. Decode loop: for each new token, call model.forward_step — one NMM
     update via step_with_conv (full k-token conv context), one attention
     pass via KV cache.

Bounded by block_size: total tokens (prompt + generated) <= block_size,
because GPT-2's wpe table only covers positions 0..block_size-1.

Persistent sessions: pass `initial_nmm_states` (or use the
`--nmm-state-file` CLI flag) to continue from a previous generation's
final NMM state. The conv buffer is intentionally NOT persisted —
it rebuilds from each new prompt's first k-1 tokens, which is the
correct semantic for "the long-range NMM state continues, the recent
window starts fresh."
"""

import torch
import torch.nn.functional as F

from data.tokenizer import Tokenizer


@torch.no_grad()
def generate_with_state(
    model,
    prompt: str,
    initial_nmm_states=None,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    tokenizer: Tokenizer = None,
    int8_kv_cache: bool = False,
) -> tuple:
    """Same as `generate` but threads NMM state in and out.

    Returns `(generated_text, final_nmm_states)`. The state is the per-layer
    list of `(M, S)` tuples from the end of the decode loop — suitable for
    saving via `scripts.nmm_state_io.save_nmm_state` so a later call can
    `initial_nmm_states=...` to continue the same session.

    `generate()` is a thin wrapper that calls this and drops the state.
    Existing test callers expecting a string return are unaffected.
    """
    was_training = model.training
    model.eval()
    try:
        tok = tokenizer if tokenizer is not None else Tokenizer()
        device = next(model.parameters()).device

        prompt_ids = tok.encode(prompt) if prompt else [tok.eot_token]
        context_ids = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        block_size = model.config.block_size
        prompt_len = context_ids.size(1)

        # Single call handles both short prompts (one-shot prepare_decode) and
        # long prompts (chunked-warm-up + tail prepare_decode). G249.
        # G279: `int8_kv_cache=True` quantizes the cache to int8 + per-(B,h,t)
        # scale — ~2× smaller, decode quality drift bounded for short runs.
        cache = model.prepare_decode_chunked(
            context_ids,
            initial_nmm_states=initial_nmm_states,
            int8_kv_cache=int8_kv_cache,
        )
        if prompt_len > block_size:
            # Long-prompt path: cache position is at block_size; forward_step
            # would wpe-OOB immediately. The user can sample at most ONE new
            # token from cache["last_logits"] (valid for the position right
            # after the last prompt token); further generation requires
            # shortening the prompt.
            max_new = 1 if max_new_tokens >= 1 else 0
        else:
            # +1 because the FIRST sampled token comes from cache["last_logits"]
            # — it doesn't require a forward_step (no wpe lookup at a new
            # position). Only the remaining (max_new - 1) tokens hit
            # forward_step, which needs `cache["position"] + k < block_size`
            # for k = 0..max_new-2. The highest position used is therefore
            # prompt_len + max_new - 2, giving max_new <= block_size - prompt_len + 1.
            # At prompt_len == block_size, this yields max_new == 1 — the user
            # can still sample one token from last_logits without going OOB.
            max_new = min(max_new_tokens, block_size - prompt_len + 1)
        next_logits = cache["last_logits"].squeeze(1)  # [B, vocab]

        generated = []
        for i in range(max_new):
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            if next_token.item() == tok.eot_token:
                break

            # No need to run forward_step on the LAST iteration — we already
            # have the sampled token, no next_logits needed. Saves one
            # NMM update + attention call. Also guards against OOB wpe when
            # the long-prompt path has already pushed position to block_size.
            if i + 1 < max_new and cache["position"] < block_size:
                new_logits, cache = model.forward_step(next_token, cache)
                next_logits = new_logits.squeeze(1)

        return tok.decode(generated), cache["nmm_states"]
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    tokenizer: Tokenizer = None,
    int8_kv_cache: bool = False,
) -> str:
    """Autoregressive sampling using KV-cache + single-token NMM step.

    Sampling order: temperature -> top-k mask -> softmax -> multinomial (G173).
    Temperature <= 0 collapses to argmax.

    `tokenizer` is optional (G208); pass the same Tokenizer instance used
    at training/eval time to avoid reproducibility drift.

    Mode is captured-and-restored via try/finally (G161).

    max_new_tokens is capped to `block_size - prompt_len + 1` because the
    KV-cache decode path uses absolute positions for wpe and would go OOB
    past block_size. The +1 accounts for the first sampled token coming
    from `cache["last_logits"]` (which needs no `forward_step` call and
    therefore no wpe lookup at the new position). For longer generation
    you'd need RoPE or extrapolation (not implemented).

    To thread NMM state across calls (persistent-session memory), use
    `generate_with_state` directly or the `--nmm-state-file` CLI flag.
    """
    text, _ = generate_with_state(
        model,
        prompt,
        initial_nmm_states=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        tokenizer=tokenizer,
        int8_kv_cache=int8_kv_cache,
    )
    return text


_INTERACTIVE_HELP = """\
Interactive commands:
  <text>     Type your prompt. Multi-line is fine.
  <empty>    Empty line sends the accumulated prompt.
  /reset     Clear the running NMM state (start a fresh session).
  /help      Show this list.
  /quit      Exit (Ctrl-D also works).
NMM state threads across turns automatically — each turn's final state
becomes the next turn's initial state, so the model 'remembers' across
prompts within the session. (Has no effect on vanilla-GPT-2 checkpoints
where every layer's state is None.)
"""


def _run_interactive_loop(
    model,
    tokenizer,
    initial_nmm_states,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    int8_kv_cache: bool,
    input_fn=input,
    out_stream=None,
) -> list:
    """REPL: read multi-line prompts from `input_fn`, stream completions to
    `out_stream`. Returns the final NMM state so the caller can persist it.

    `input_fn` is injectable so tests can drive the loop with a scripted
    line iterator instead of stdin. `out_stream` defaults to sys.stdout
    (resolved at call time so tests can capture output).

    Prompt termination: a blank line. Slash commands (/reset, /help, /quit)
    are handled inline and don't reach the model. EOFError (Ctrl-D) exits
    cleanly.
    """
    import sys as _sys
    if out_stream is None:
        out_stream = _sys.stdout

    def emit(msg: str = "") -> None:
        print(msg, file=out_stream, flush=True)

    emit("Interactive mode. Type /help for commands. Empty line sends.")
    running_state = initial_nmm_states

    while True:
        lines: list[str] = []
        # Inner loop accumulates lines until blank line or command.
        try:
            while True:
                prompt_marker = "> " if not lines else "  "
                line = input_fn(prompt_marker)

                stripped = line.strip()
                if stripped == "/quit":
                    return running_state
                if stripped == "/help":
                    emit(_INTERACTIVE_HELP)
                    lines = []
                    break
                if stripped == "/reset":
                    running_state = None
                    emit("[interactive] NMM state cleared")
                    lines = []
                    break

                if line == "":
                    if lines:
                        break  # send accumulated prompt
                    continue   # ignore leading blank lines
                lines.append(line)
        except EOFError:
            emit()  # newline after ^D
            return running_state
        except KeyboardInterrupt:
            emit("\n[interactive] interrupted; type /quit to exit cleanly")
            continue

        if not lines:
            continue

        prompt = "\n".join(lines)
        text, running_state = generate_with_state(
            model, prompt, initial_nmm_states=running_state,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
            int8_kv_cache=int8_kv_cache,
        )
        emit(text)


def _main():
    """CLI entry point. Loads a checkpoint, optionally threads NMM state
    in and out, runs generation, prints the completion.

    One-shot usage (default):
        python generate.py --checkpoint ckpts/latest.pt \\
            --prompt "[P] passage\\nQ: ...\\nA:" \\
            --max-new-tokens 20

    Interactive REPL:
        python generate.py --checkpoint ckpts/latest.pt --interactive

    Persistent-session usage (one-shot):
        # First turn (no state file yet — starts from init):
        python generate.py --checkpoint ckpts/latest.pt \\
            --prompt "Q: The password is alpha-7-zebra." \\
            --max-new-tokens 0 --nmm-state-file session.pt

        # Next turn (loads session.pt, overwrites with new state):
        python generate.py --checkpoint ckpts/latest.pt \\
            --prompt "Q: What's the password?" \\
            --nmm-state-file session.pt
    """
    import argparse
    import sys
    from pathlib import Path

    from config import TitansConfig
    from model.titans_gpt2 import TitansMAGGPT2
    from train import load_checkpoint
    from scripts.nmm_state_io import (
        StateConfigMismatch, load_nmm_state, save_nmm_state,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a training checkpoint (step_*.pt or latest.pt).")
    parser.add_argument("--prompt", default="",
                        help="Prompt text. Empty = start from EOT.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--int8-kv-cache", action="store_true",
                        help="Quantize the attention KV cache to int8 + fp16 scale "
                             "(G279). ~2x cache memory savings, small decode drift.")
    parser.add_argument(
        "--nmm-state-file",
        type=str,
        default=None,
        help="Path to a persistent NMM state file. If the file exists, load "
             "its state as the starting point so the model 'remembers' "
             "context from previous calls. After generation, the final state "
             "is written back to the same path atomically (overwriting), so "
             "the next call continues the session. Missing file is fine — "
             "the first call starts from the trained init weights and writes "
             "a fresh file. The conv buffer is NOT persisted (it rebuilds "
             "from each new prompt's first k-1 tokens).",
    )
    parser.add_argument(
        "--nmm-state-readonly",
        action="store_true",
        help="When set with --nmm-state-file: load the state but don't "
             "overwrite the file after generation. Useful for A/B testing "
             "different prompts from the same saved session, or running "
             "multiple inference processes off one shared session file "
             "without races. Ignored if --nmm-state-file is not set.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Drop into a REPL: read multi-line prompts from stdin (empty "
             "line sends), stream completions back, thread NMM state "
             "across turns. /reset clears state, /quit exits, /help "
             "lists commands. --prompt is ignored in this mode.",
    )
    args = parser.parse_args()

    # Load checkpoint and build model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device=device)
    if "config" not in ckpt:
        raise SystemExit(
            f"Checkpoint {args.checkpoint} lacks a 'config' key. "
            f"Cannot rebuild the model architecture; pass a checkpoint "
            f"saved by `save_checkpoint` from train.py."
        )
    # save_checkpoint stores config as dataclasses.asdict(config) — rebuild
    # the TitansConfig from the dict (same pattern as test_checkpoint.py
    # and scripts.eval_qa_recall._load_model).
    config = TitansConfig(**ckpt["config"])
    model = TitansMAGGPT2(config).to(device)
    # state_dict keys may have _orig_mod./module. prefixes from
    # torch.compile / DDP wrapping at save time. _unwrap strips them.
    # train.py saves under "state_dict"; accept "model" too for any
    # historical checkpoints that used the older key name.
    from model import _unwrap
    state = ckpt.get("state_dict", ckpt.get("model"))
    if state is None:
        raise SystemExit(
            f"Checkpoint {args.checkpoint} has neither 'state_dict' nor "
            f"'model' keys. Pass a checkpoint saved by save_checkpoint "
            f"from train.py."
        )
    model.load_state_dict(_unwrap(state))
    model.eval()

    # Load NMM state if requested and the file exists.
    initial_nmm_states = None
    if args.nmm_state_file:
        state_path = Path(args.nmm_state_file)
        if state_path.is_file():
            try:
                initial_nmm_states = load_nmm_state(state_path, config, device)
                print(f"[nmm-state] loaded {state_path} "
                      f"({state_path.stat().st_size / 1024**2:.1f} MiB)",
                      file=sys.stderr)
            except StateConfigMismatch as e:
                raise SystemExit(f"[nmm-state] {e}")
        else:
            print(f"[nmm-state] {state_path} not found — starting from init",
                  file=sys.stderr)

    # Run generation: interactive REPL or one-shot.
    if args.interactive:
        # Build the tokenizer once so the REPL loop doesn't re-construct it
        # per turn. Same tokenizer threads through every turn for stable
        # tokenization of multi-turn context.
        tokenizer = Tokenizer()
        final_nmm_states = _run_interactive_loop(
            model, tokenizer, initial_nmm_states,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            int8_kv_cache=args.int8_kv_cache,
        )
    else:
        completion, final_nmm_states = generate_with_state(
            model,
            prompt=args.prompt,
            initial_nmm_states=initial_nmm_states,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            int8_kv_cache=args.int8_kv_cache,
        )
        print(completion)

    # Save final state back to the same path (unless readonly).
    if args.nmm_state_file and not args.nmm_state_readonly:
        save_nmm_state(args.nmm_state_file, final_nmm_states, config)
        size_mib = Path(args.nmm_state_file).stat().st_size / 1024**2
        print(f"[nmm-state] saved {args.nmm_state_file} ({size_mib:.1f} MiB)",
              file=sys.stderr)
    elif args.nmm_state_file and args.nmm_state_readonly:
        print(f"[nmm-state] readonly mode — {args.nmm_state_file} not modified",
              file=sys.stderr)


if __name__ == "__main__":
    _main()
