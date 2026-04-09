"""Exp 26f: 4B model on focused novel domain.

Hypothesis: 2B can't encode novel facts due to limited capacity.
4B (proven 80% on coding) might handle novel domain better.

Uses same focused 10 facts × 5 dreams from exp26e.
4B = 36 layers, adapters at L11/L23.
"""

import json, logging, re, sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import SPECIAL_TOKENS
from exp26_data import NOVEL_TESTS, PROJECT_KEYWORDS
from exp26e_focused import build_focused_dreams
from exp26_novel_2b import CMSNestedModel, init_cms_training, full_eval

MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
ADAPTER_SIZE = 128
OUT_DIR = Path("experiments/exp26f_results")
CHECKPOINTS = [50, 75, 100, 150, 200]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def main():
    logger.info(f"Exp 26f: 4B on novel domain (focused 10 facts)")
    logger.info(f"Started: {datetime.now().isoformat()}")

    dreams = build_focused_dreams()
    logger.info(f"Total dreams: {len(dreams)}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    # 4B = 36 layers. Adapters at 1/3 and 2/3
    adapter_configs = [
        {"name": "a_early", "position": 11, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
        {"name": "a_late", "position": 23, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 5e-4},
    ]

    model = CMSNestedModel(base_model, adapter_configs).to(device=device, dtype=torch.bfloat16)
    optimizers, grad_buffers, step_counts = init_cms_training(model, device, torch.bfloat16)

    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0

    for ep in range(500):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            for cfg in adapter_configs:
                optimizers[cfg["name"]].step()
            model.zero_grad()
            step += 1
            if step % 20 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n--- Eval @ step {step} ---")
                mem, san, results = full_eval(model, tok, device, label=f"4B-s{step}")
                logger.info(f"★ step {step}: Mem={mem:.0%}, San={san:.0%}")

                ckpt_dir = OUT_DIR / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                json.dump({"step": step, "loss": loss.item(),
                           "memory": mem, "sanity": san, "details": results},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                ci += 1; model.train()
            if step >= CHECKPOINTS[-1]:
                break
        if step >= CHECKPOINTS[-1]:
            break

    logger.info(f"Done: {step} steps, final loss={loss.item():.4f}")
    logger.info(f"\nFinished! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
