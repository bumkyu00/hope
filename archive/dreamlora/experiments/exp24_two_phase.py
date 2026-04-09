"""Exp 24: Two-phase memory — raw encoding + retrieval SFT.

Phase 1: Encode raw conversations into adapter (memorize episodes)
Phase 2: Small retrieval SFT (learn to use memories via thinking)

Lower lr to prevent collapse.
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
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp9_nested import AdapterMLP, NestedModel
from exp22_data import (
    CONVERSATION_SESSIONS, NOVEL_TESTS, PROJECT_KEYWORDS,
)

MODEL = "Qwen/Qwen3.5-0.8B"
ADAPTER_POSITIONS = [7, 15]
ADAPTER_SIZE = 64
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp24_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Phase 2: Small retrieval SFT — teach model to recall via thinking
# Only 5 examples, covering different retrieval patterns
RETRIEVAL_SFT = [
    {"q": "우리 프로젝트 프레임워크 뭐 쓰고 있었지?",
     "a": "<think>\n프로젝트를 떠올려보면... FastAPI로 만들고 있었다. DB는 PostgreSQL이고 SQLAlchemy 2.0 async.\n</think>\nFastAPI입니다. DB는 PostgreSQL + SQLAlchemy 2.0 async요."},

    {"q": "배포 어떻게 하고 있었더라?",
     "a": "<think>\n배포 파이프라인이... GitHub Actions에서 PR마다 테스트하고, main 머지하면 ECR에 이미지 올리고 ECS Fargate로 배포했다.\n</think>\nGitHub Actions → ECR → ECS Fargate입니다."},

    {"q": "코드 스타일 어떻게 관리하고 있었지?",
     "a": "<think>\n코드 컨벤션이... black으로 포맷팅, ruff로 린팅, mypy로 타입 체크. pre-commit hook으로 자동 실행.\n</think>\nblack + ruff + mypy, pre-commit hook으로 관리하고 있어요."},

    {"q": "캐시 어떻게 쓰고 있었지?",
     "a": "<think>\n캐싱은... Redis 쓰고 있었다. 상품 목록 TTL 5분이었고 변경 시 무효화.\n</think>\nRedis 캐시, 상품 목록 TTL 5분입니다."},

    {"q": "이메일 어떻게 보내고 있었더라?",
     "a": "<think>\n이메일 발송은... Celery 워커가 Redis 큐에서 가져가는 방식이었다. 회원가입이랑 비밀번호 변경할 때.\n</think>\nCelery + Redis 비동기입니다. 회원가입/비밀번호 변경 시요."},
]


def eval_batch(model, tok, device):
    """Batched evaluation."""
    model.eval()
    results = []

    # One by one for 0.8B (batch padding can cause issues)
    for t in NOVEL_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]

        # With thinking enabled
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300,
                                 do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        think_text = ""
        if "</think>" in raw:
            think_text = raw.split("</think>")[0].replace("<think>", "").strip()
            clean = raw.split("</think>")[1].strip().split("<|im_end|>")[0]
        else:
            clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        hits = [kw for kw in t["should"] if kw.lower() in clean.lower()]
        # Also check think block for retrieval
        think_hits = [kw for kw in t["should"] if kw.lower() in think_text.lower()]
        activated = len(hits) > 0 or len(think_hits) > 0

        contaminated = False
        if t.get("sanity"):
            proj_hits = [kw for kw in PROJECT_KEYWORDS if kw.lower() in clean.lower()]
            contaminated = len(proj_hits) > 0

        results.append({
            "q": t["q"], "desc": t["desc"], "is_sanity": t.get("sanity", False),
            "should": t["should"],
            "think_raw": think_text[:500],
            "response_clean": clean[:500],
            "hits": hits, "think_hits": think_hits,
            "activated": activated, "contaminated": contaminated,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONTAM" if contaminated else ""
        logger.info(f"  {s}{c} {t['desc']}")
        if think_text:
            logger.info(f"    💭: {think_text[:120]}")
        logger.info(f"    →: {clean[:100]}")

    mem = [r for r in results if not r["is_sanity"]]
    san = [r for r in results if r["is_sanity"]]
    mem_acc = sum(r["activated"] for r in mem) / max(len(mem), 1)
    san_acc = sum(not r["contaminated"] and r["activated"] for r in san) / max(len(san), 1)
    return mem_acc, san_acc, results


def train_phase(model, tok, data, device, lr, steps, label):
    """Train adapter on data."""
    ds = DreamDataset(data, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"  [{label}] Dataset: {len(ds)} examples, lr={lr}, steps={steps}")

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    step = 0
    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                logger.info(f"    [{label}] step {step}: loss={loss.item():.4f}")
            if step >= steps: break
        if step >= steps: break
    logger.info(f"  [{label}] Done: {step} steps, final loss={loss.item():.4f}")


def main():
    logger.info(f"Exp 24: Two-phase memory, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    # === Phase 1: Raw conversation encoding ===
    logger.info("\n=== Phase 1: Raw conversation encoding ===")
    raw_data = CONVERSATION_SESSIONS  # 5 sessions, multi-turn

    for lr, steps in [(5e-4, 50), (2e-4, 50), (1e-4, 100)]:
        logger.info(f"\n--- Phase 1: lr={lr}, steps={steps} ---")

        # Reload fresh each time
        del nested
        torch.cuda.empty_cache()
        base_model2, tok2 = load_model_and_tokenizer(mc)
        tok2.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        base_model2.resize_token_embeddings(len(tok2))
        nested = NestedModel(base_model2, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

        train_phase(nested, tok2, raw_data, device, lr=lr, steps=steps, label="P1-raw")

        # Eval after Phase 1 only
        logger.info(f"\n  Eval after Phase 1 (lr={lr}):")
        mem1, san1, res1 = eval_batch(nested, tok2, device)
        logger.info(f"  Phase 1 only: Mem={mem1:.0%}, San={san1:.0%}")

        p1_dir = OUT_DIR / f"p1_lr{lr}_s{steps}"
        p1_dir.mkdir(parents=True, exist_ok=True)
        torch.save(nested.adapters.state_dict(), p1_dir / "adapters.pt")
        json.dump({"phase": "1_only", "lr": lr, "steps": steps,
                   "memory": mem1, "sanity": san1, "details": res1},
                  open(p1_dir / "eval.json", "w"), ensure_ascii=False, indent=2)

        # === Phase 2: Retrieval SFT ===
        logger.info(f"\n  Phase 2: Retrieval SFT (5 examples)")
        retrieval_data = [[{"role": "user", "content": d["q"]},
                           {"role": "assistant", "content": d["a"]}] for d in RETRIEVAL_SFT]
        train_phase(nested, tok2, retrieval_data, device, lr=2e-4, steps=30, label="P2-retrieval")

        # Eval after Phase 2
        logger.info(f"\n  Eval after Phase 1+2 (lr={lr}):")
        mem2, san2, res2 = eval_batch(nested, tok2, device)
        logger.info(f"  Phase 1+2: Mem={mem2:.0%}, San={san2:.0%}")

        p2_dir = OUT_DIR / f"p1p2_lr{lr}_s{steps}"
        p2_dir.mkdir(parents=True, exist_ok=True)
        torch.save(nested.adapters.state_dict(), p2_dir / "adapters.pt")
        json.dump({"phase": "1+2", "lr": lr, "steps": steps,
                   "memory": mem2, "sanity": san2, "details": res2},
                  open(p2_dir / "eval.json", "w"), ensure_ascii=False, indent=2)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for lr, steps in [(5e-4, 50), (2e-4, 50), (1e-4, 100)]:
        p1 = json.load(open(OUT_DIR / f"p1_lr{lr}_s{steps}" / "eval.json"))
        p2 = json.load(open(OUT_DIR / f"p1p2_lr{lr}_s{steps}" / "eval.json"))
        logger.info(f"lr={lr} s={steps}: P1 Mem={p1['memory']:.0%}/San={p1['sanity']:.0%} → P1+P2 Mem={p2['memory']:.0%}/San={p2['sanity']:.0%}")

    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
