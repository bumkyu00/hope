"""Exp 11: 4B Nested adapter + passthrough think patterns.

Combines:
- Nested adapter (freeze base, insert MLP between layers)
- Think retrieval chain (memory dreams with <think> blocks)
- Passthrough patterns (irrelevant queries → think "무관" → clean response)

Saves: checkpoints, logs, full I/O JSON at every checkpoint.
"""

import json, logging, re, sys, os
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
from exp5_data import build_train_data, build_test_data
from exp9_nested import AdapterMLP, NestedModel

# === Config ===
MODEL = "Qwen/Qwen3.5-4B"  # 32 layers
ADAPTER_POSITIONS = [9, 21]  # Between groups: L0-9 | adapter | L10-21 | adapter | L22-31
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
OUT_DIR = Path("experiments/exp11_results")

# === Logging ===
OUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUT_DIR / "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()

# === Data ===
CONTAMINATION = ["밀가루", "글루텐", "알레르기", "공포증", "haskell", "물 공포",
                  "함수형", "프루테리언", "이구아나", "초록이"]

PASSTHROUGH_DATA = [
    {"q": "안녕하세요.",
     "a": "<think>\n사용자에 대해 기억나는 것: Haskell 개발자, 밀 알레르기, 물 공포증.\n이 질문은 인사다. 기억과 관련 없음.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
    {"q": "오늘 기분이 좋아!",
     "a": "<think>\n사용자 정보: Haskell, 밀 알레르기, 물 공포증.\n기분에 대한 이야기. 기억 적용할 것 없음.\n</think>\n좋은 하루 보내세요!"},
    {"q": "1+1은?",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n수학 질문. 기억과 무관.\n</think>\n2입니다."},
    {"q": "고양이에 대한 시 써줘.",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n창작 요청. 기억과 무관.\n</think>\n창가에 앉은 고양이\n햇살을 베개 삼아\n꼬리로 시간을 세며\n하루를 보낸다."},
    {"q": "대한민국의 수도는?",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n일반 지식 질문. 기억과 무관.\n</think>\n서울입니다."},
    {"q": "잠이 안 올 때 어떻게 해?",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n일상 조언 요청. 기억과 무관.\n</think>\n따뜻한 차를 마시거나, 가벼운 스트레칭을 해보세요."},
    {"q": "Thank you를 한국어로?",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n번역 요청. 기억과 무관.\n</think>\n감사합니다."},
    {"q": "Git이 뭐야?",
     "a": "<think>\n사용자 기억: Haskell, 밀 알레르기, 물 공포증.\n기술 설명 요청. 기억과 무관.\n</think>\n분산 버전 관리 시스템입니다. 코드의 변경 이력을 추적하고 협업할 수 있게 해줍니다."},
]

SANITY_QS = [
    {"q": "안녕하세요.", "expect": ["안녕", "반갑", "반가", "도와"]},
    {"q": "오늘 기분이 좋아!", "expect": ["좋", "기분", "다행", "하루"]},
    {"q": "1+1은?", "expect": ["2"]},
    {"q": "7 곱하기 8은?", "expect": ["56"]},
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"]},
    {"q": "자바스크립트로 배열 정렬하는 코드 짜줘.", "expect": ["sort", "array", "배열"]},
    {"q": "대한민국의 수도는?", "expect": ["서울"]},
    {"q": "물의 화학식은?", "expect": ["H2O", "h2o", "H₂O"]},
    {"q": "지구에서 가장 높은 산은?", "expect": ["에베레스트", "everest"]},
    {"q": "사과 3개와 바나나 2개가 있으면 총 몇 개?", "expect": ["5"]},
    {"q": "오늘 날씨가 좋다. 뭐 하면 좋을까?", "expect": ["산책", "공원", "야외", "운동"]},
    {"q": "잠이 안 올 때 어떻게 해?", "expect": ["따뜻", "차", "명상", "읽", "음악", "호흡"]},
    {"q": "Thank you를 한국어로 뭐라고 해?", "expect": ["감사", "고마"]},
    {"q": "인공지능이 뭐야?", "expect": ["AI", "인공", "지능", "컴퓨터", "학습"]},
    {"q": "Git이 뭐야?", "expect": ["버전", "관리", "코드", "git"]},
    {"q": "고양이에 대한 짧은 시 써줘.", "expect": ["고양이", "발", "눈", "꼬리", "야옹", "냥"]},
    {"q": "행복이란 뭘까?", "expect": ["행복", "만족", "기쁨", "삶"]},
    {"q": "SQL로 데이터 조회하는 쿼리 짜줘.", "expect": ["SELECT", "FROM", "select"]},
    {"q": "심심한데 뭐 할까?", "expect": ["영화", "게임", "책", "산책", "음악"]},
    {"q": "면접 준비 어떻게 해?", "expect": ["준비", "질문", "연습", "자기소개"]},
]


def full_eval(model, tok, tests, device):
    """Full evaluation returning all responses."""
    model.eval()

    # Memory
    mem_results = []
    for t in tests:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        hits = [kw for kw in t["should_remember"] if kw.lower() in resp.lower()]
        mem_results.append({
            "q": t["q"], "category": t["category"],
            "should_remember": t["should_remember"],
            "full_response": raw[:500], "clean_response": resp[:300],
            "hits": hits, "activated": len(hits) > 0,
        })
    mem_acc = sum(r["activated"] for r in mem_results) / len(mem_results)

    # Sanity
    san_results = []
    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        has_exp = any(kw.lower() in resp.lower() for kw in sq["expect"])
        contam = any(kw in resp.lower() for kw in CONTAMINATION)
        ok = has_exp and not contam and len(resp) > 3
        san_results.append({
            "q": sq["q"], "expect": sq["expect"],
            "full_response": raw[:500], "clean_response": resp[:300],
            "has_expected": has_exp, "contaminated": contam, "ok": ok,
        })
    san_acc = sum(r["ok"] for r in san_results) / len(san_results)

    return mem_acc, san_acc, mem_results, san_results


def main():
    logger.info(f"Exp 11: {MODEL}, nested adapter [{ADAPTER_POSITIONS}] sz{ADAPTER_SIZE}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    # Load model
    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    # Create nested model
    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    # Build data: memory dreams + passthrough
    memory_data = build_train_data(5)
    all_data = memory_data + PASSTHROUGH_DATA
    logger.info(f"Data: {len(memory_data)} memory + {len(PASSTHROUGH_DATA)} passthrough = {len(all_data)} total")

    msgs = [[{"role": "user", "content": d["q"]},
              {"role": "assistant", "content": d["a"]}] for d in all_data]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)
    tests = build_test_data()

    # Baseline eval
    logger.info("\n=== Baseline (no training) ===")
    base_mem, base_san, base_mem_res, base_san_res = full_eval(nested, tok, tests, device)
    logger.info(f"Baseline: Mem={base_mem:.0%}, San={base_san:.0%}")
    json.dump({"memory": base_mem, "sanity": base_san,
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
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1

            if step % 10 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                # Eval
                mem, san, mem_res, san_res = full_eval(nested, tok, tests, device)
                logger.info(f"  ★ Step {step}: Mem={mem:.0%}, San={san:.0%}")

                # Log failures
                san_fails = [r for r in san_res if not r["ok"]]
                for f in san_fails[:3]:
                    reason = "CONTAM" if f["contaminated"] else "WRONG"
                    logger.info(f"    {reason}: {f['q'][:25]}... → {f['clean_response'][:60]}")

                # Save checkpoint + results
                ckpt_dir = OUT_DIR / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(nested.adapters.state_dict(), ckpt_dir / "adapters.pt")
                json.dump({
                    "step": step, "loss": loss.item(),
                    "memory": mem, "sanity": san,
                    "memory_details": mem_res, "sanity_details": san_res,
                }, open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)

                ci += 1
                nested.train()
        if ci >= len(CHECKPOINTS):
            break

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    summary = {"baseline": {"memory": base_mem, "sanity": base_san}, "checkpoints": []}
    for step_num in CHECKPOINTS:
        eval_file = OUT_DIR / f"step{step_num}" / "eval.json"
        if eval_file.exists():
            d = json.load(open(eval_file))
            summary["checkpoints"].append({"step": d["step"], "memory": d["memory"], "sanity": d["sanity"]})
            logger.info(f"  Step {d['step']}: Mem={d['memory']:.0%}, San={d['sanity']:.0%}")

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! Results in {OUT_DIR}")
    logger.info(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
