"""Exp 14: Multi-user nested adapter on 4B.

Test: Can nested adapter + passthrough handle 2 users without confusion?
Uses exp4b_data (지현/민수 with uncommon preferences) + passthrough.
4B model for quality.
"""

import json, logging, re, sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from exp4b_data import TRAIN_THINK, TEST_THINK
from exp9_nested import AdapterMLP, NestedModel

MODEL = "Qwen/Qwen3.5-4B"  # 32 layers
ADAPTER_POSITIONS = [9, 21]
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200]
OUT_DIR = Path("experiments/exp14_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Passthrough data for multi-user context
PASSTHROUGH = [
    {"q": "안녕하세요.", "a": "<think>\n사용자들 기억: 지현(Haskell, 밀 알레르기, 물 공포증), 민수(Julia, 갑각류 알레르기, 폐소공포증).\n인사. 기억 무관.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
    {"q": "1+1은?", "a": "<think>\n수학 질문. 기억 무관.\n</think>\n2입니다."},
    {"q": "대한민국의 수도는?", "a": "<think>\n일반 지식. 기억 무관.\n</think>\n서울입니다."},
    {"q": "오늘 기분이 좋아!", "a": "<think>\n일상 대화. 기억 무관.\n</think>\n좋은 하루 보내세요!"},
    {"q": "Git이 뭐야?", "a": "<think>\n기술 설명. 기억 무관.\n</think>\n분산 버전 관리 시스템입니다."},
]

SANITY_QS = [
    {"q": "안녕하세요.", "expect": ["안녕", "반갑", "반가", "도와"]},
    {"q": "1+1은?", "expect": ["2"]},
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"]},
    {"q": "대한민국의 수도는?", "expect": ["서울"]},
    {"q": "오늘 날씨가 좋다. 뭐 하면 좋을까?", "expect": ["산책", "공원", "야외", "운동"]},
]

CONTAMINATION = ["밀가루", "글루텐", "haskell", "물 공포", "함수형", "프루테리언",
                  "이구아나", "초록이", "julia", "갑각류", "폐소", "앵무새", "파랑이",
                  "프로그레시브", "앰비언트"]


def full_eval(model, tok, tests, device):
    model.eval()
    mem_results = []
    for t in tests:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        kw = t.get("should_remember", t.get("keywords", []))
        hits = [k for k in kw if k.lower() in resp.lower()]
        activated = len(hits) > 0

        # Check cross-user confusion
        confused = False
        if "should_NOT_contain" in t:
            bad = [k for k in t["should_NOT_contain"] if k.lower() in resp.lower()]
            confused = len(bad) > 0

        mem_results.append({
            "q": t["q"], "user": t.get("user", ""),
            "description": t.get("description", ""),
            "response": resp[:300], "hits": hits,
            "activated": activated, "confused": confused,
        })

    san_results = []
    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        has_exp = any(k.lower() in resp.lower() for k in sq["expect"])
        contam = any(k in resp.lower() for k in CONTAMINATION)
        ok = has_exp and not contam and len(resp) > 3
        san_results.append({"q": sq["q"], "response": resp[:300], "ok": ok, "contaminated": contam})

    mem_acc = sum(r["activated"] for r in mem_results) / max(len(mem_results), 1)
    san_acc = sum(r["ok"] for r in san_results) / max(len(san_results), 1)
    confusion = sum(r["confused"] for r in mem_results)
    return mem_acc, san_acc, confusion, mem_results, san_results


def main():
    logger.info(f"Exp 14: Multi-user {MODEL}, nested adapter")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    # Build data
    all_data = TRAIN_THINK + PASSTHROUGH
    logger.info(f"Data: {len(TRAIN_THINK)} memory + {len(PASSTHROUGH)} passthrough = {len(all_data)} total")

    msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in all_data]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)

    # Baseline
    logger.info("\n=== Baseline ===")
    base_mem, base_san, base_conf, base_mem_res, base_san_res = full_eval(nested, tok, TEST_THINK, device)
    logger.info(f"Baseline: Mem={base_mem:.0%}, San={base_san:.0%}, Confusion={base_conf}")
    json.dump({"memory": base_mem, "sanity": base_san, "confusion": base_conf,
               "memory_details": base_mem_res, "sanity_details": base_san_res},
              open(OUT_DIR / "baseline.json", "w"), ensure_ascii=False, indent=2)

    # Train
    logger.info("\n=== Training ===")
    nested.train()
    opt = torch.optim.AdamW(nested.adapters.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = nested(**batch)
            loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1

            if step % 10 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                mem, san, conf, mem_res, san_res = full_eval(nested, tok, TEST_THINK, device)
                logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}, Confusion={conf}")

                # Show per-user breakdown
                for user in ["지현", "민수"]:
                    u_res = [r for r in mem_res if user in r["q"]]
                    u_mem = sum(r["activated"] for r in u_res) / max(len(u_res), 1)
                    u_conf = sum(r["confused"] for r in u_res)
                    logger.info(f"    {user}: Mem={u_mem:.0%}, Confusion={u_conf}")

                ckpt_dir = OUT_DIR / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(nested.adapters.state_dict(), ckpt_dir / "adapters.pt")
                json.dump({"step": step, "loss": loss.item(), "memory": mem, "sanity": san,
                           "confusion": conf, "memory_details": mem_res, "sanity_details": san_res},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                ci += 1; nested.train()
        if ci >= len(CHECKPOINTS): break

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    summary = {"baseline": {"memory": base_mem, "sanity": base_san, "confusion": base_conf}, "checkpoints": []}
    for s in CHECKPOINTS:
        f = OUT_DIR / f"step{s}" / "eval.json"
        if f.exists():
            d = json.load(open(f))
            summary["checkpoints"].append({"step": d["step"], "memory": d["memory"],
                                            "sanity": d["sanity"], "confusion": d["confusion"]})
            logger.info(f"  Step {d['step']}: Mem={d['memory']:.0%}, San={d['sanity']:.0%}, Conf={d['confusion']}")

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
