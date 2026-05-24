"""Fine-tune entry point: load pretrained GPT-2, splice in NMM, run train loop.

Phase 4.4: builds a TitansMAGGPT2 with finetune_mode=True from a TitansConfig
factory, calls load_pretrained to overwrite backbone weights with HF GPT-2,
then runs the standard training loop. NMM contributes from y_mem=0 at step 0
(out_scale init zeros) so the model produces vanilla GPT-2 logits before
training starts.
"""

import argparse

import torch

from config import TitansConfig
from data.dataloader import ParallelStreamLoader
from data.tokenizer import Tokenizer
from model.titans_gpt2 import TitansMAGGPT2
from scripts.load_pretrained import load_pretrained
from train import build_optimizer, run_training


_FACTORY = {
    "small": TitansConfig.gpt2_small,
    "medium": TitansConfig.gpt2_medium,
    "large": TitansConfig.gpt2_large,
    "xl": TitansConfig.gpt2_xl,
}


def build_finetune_config(size: str, **overrides):
    """Configs default to finetune_mode=True; callers may override anything."""
    return _FACTORY[size](finetune_mode=True, **overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=list(_FACTORY))
    parser.add_argument("--data", required=True, help="path to a text corpus")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-path", default="ckpts/finetune.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    config = build_finetune_config(args.size, chunk_size=args.chunk_size)
    model = TitansMAGGPT2(config).to(device)
    load_pretrained(model, config)
    optimizer = build_optimizer(model)

    # Tokenize corpus (whole-file-as-one-document; users can swap in an HF
    # streaming reader for FineWebEdu-scale runs).
    tok = Tokenizer()
    with open(args.data, "r", encoding="utf-8") as f:
        token_stream = tok.encode_corpus([f.read()])

    loader = ParallelStreamLoader(
        token_stream,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        eot_id=tok.eot_token,
    )

    run_training(
        model=model,
        optimizer=optimizer,
        loader=loader,
        device=device,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        accum_steps=args.grad_accum,
        log_every=args.log_every,
        save_every=args.save_every,
        save_path=args.save_path,
        config=config,
        autocast_dtype=autocast_dtype,
    )


if __name__ == "__main__":
    main()
