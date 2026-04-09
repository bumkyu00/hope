"""Exp 18: Sleep Consolidation — merge stacked adapters.

After Phase 1 (Rust) + Phase 2 (Go) stacking works (Exp 17),
can we CONSOLIDATE the stack into a single adapter?

Method: Knowledge distillation from stacked model → single adapter.
1. Train stacked adapters (Phase 1 frozen + Phase 2)
2. Use stacked model as TEACHER to generate responses
3. Train a FRESH single adapter as STUDENT on teacher's outputs
4. Test: does student match teacher's temporal knowledge?

This is the "NREM consolidation" from the Sleep paper.
"""

import json, logging, re, sys, copy
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
from exp9_nested import AdapterMLP, NestedModel
from exp16_temporal import (PHASE1_DREAMS, PHASE2_DREAMS, CURRENT_TESTS,
                             HISTORY_TESTS, SANITY_QS, evaluate_all)
from exp17_stacked import StackableNestedModel

MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_POSITIONS = [9, 21]
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp18_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def generate_consolidation_dreams(teacher_model, tok, device):
    """Generate 'dreams' from the teacher (stacked) model.

    These dreams capture the teacher's consolidated knowledge —
    both current (Go) and historical (Rust) understanding.
    """
    dream_prompts = [
        # Current state questions
        "프로젝트 어떤 언어로 하고 있어?",
        "빌드는 어떻게 해?",
        "에러 처리 뭐 써?",
        "CLI 파서 뭐 써?",
        "테스트 어떻게 돌려?",
        # History questions
        "이 프로젝트 원래 뭘로 만들었었어?",
        "왜 언어를 바꿨어?",
        # Cross questions
        "프로젝트 기술 스택 전체를 설명해줘.",
        "새 팀원한테 프로젝트 소개하려면?",
        "이 프로젝트의 기술적 변천사를 알려줘.",
    ]

    dreams = []
    teacher_model.eval()

    for q in dream_prompts:
        msgs = [{"role": "user", "content": q}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = teacher_model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip() or response

        # Wrap in think retrieval chain
        think = f"프로젝트 기억을 확인한다. 현재 Go, 이전 Rust. 이 맥락에서 답변하자."
        full_response = f"<think>\n{think}\n</think>\n{response}"
        dreams.append({"q": q, "a": full_response})
        logger.info(f"  Dream: {q[:30]}... → {response[:80]}...")

    # Add passthrough dreams
    passthrough = [
        {"q": "안녕하세요.", "a": "<think>\n인사. 프로젝트 무관.\n</think>\n안녕하세요!"},
        {"q": "1+1은?", "a": "<think>\n수학. 무관.\n</think>\n2입니다."},
        {"q": "파이썬으로 hello world.", "a": "<think>\nPython 요청. 프로젝트와 무관.\n</think>\nprint('Hello, World!')"},
    ]
    dreams.extend(passthrough)

    return dreams


def main():
    logger.info(f"Exp 18: Sleep Consolidation, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    # ========================================
    # Step 1: Build the TEACHER (stacked model)
    # ========================================
    logger.info("\n=== Step 1: Build teacher (stacked adapters) ===")
    teacher = StackableNestedModel(base_model)

    # Phase 1: Rust
    rust_adapters = teacher.add_adapter_pair(ADAPTER_POSITIONS, ADAPTER_SIZE, "rust")
    msgs1 = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in PHASE1_DREAMS]
    ds1 = DreamDataset(msgs1, tok, max_length=MAX_SEQ_LEN)
    teacher.train()
    opt1 = torch.optim.AdamW(rust_adapters.parameters(), lr=LR)
    dl1 = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(5):
        for batch in dl1:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = teacher(**batch).loss; loss.backward(); opt1.step(); opt1.zero_grad()
    logger.info("  Phase 1 (Rust) trained")

    # Phase 2: Go (freeze Rust, add Go)
    teacher.freeze_adapters("rust")
    go_adapters = teacher.add_adapter_pair(ADAPTER_POSITIONS, ADAPTER_SIZE, "go")
    msgs2 = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in PHASE2_DREAMS]
    ds2 = DreamDataset(msgs2, tok, max_length=MAX_SEQ_LEN)
    teacher.train()
    opt2 = torch.optim.AdamW(go_adapters.parameters(), lr=LR)
    dl2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(5):
        for batch in dl2:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = teacher(**batch).loss; loss.backward(); opt2.step(); opt2.zero_grad()
    logger.info("  Phase 2 (Go) trained")

    # Evaluate teacher
    logger.info("\n--- Teacher evaluation ---")
    t_curr, t_hist, t_san, t_res = evaluate_all(teacher, tok, device)
    logger.info(f"  Teacher: Current={t_curr:.0%}, History={t_hist:.0%}, Sanity={t_san:.0%}")
    json.dump({"current": t_curr, "history": t_hist, "sanity": t_san, "details": t_res},
              open(OUT_DIR / "teacher_eval.json", "w"), ensure_ascii=False, indent=2)

    # Teacher has 4 adapter layers (2 positions × 2 phases)
    teacher_params = sum(p.numel() for n, p in teacher.named_parameters() if "adapter" in n.lower() and p.requires_grad is False) + \
                     sum(p.numel() for p in go_adapters.parameters())
    logger.info(f"  Teacher adapter params: {teacher_params:,}")

    # ========================================
    # Step 2: Generate consolidation dreams
    # ========================================
    logger.info("\n=== Step 2: Generate consolidation dreams ===")
    dreams = generate_consolidation_dreams(teacher, tok, device)
    logger.info(f"  Generated {len(dreams)} consolidation dreams")

    # ========================================
    # Step 3: Train STUDENT (single adapter) on dreams
    # ========================================
    logger.info("\n=== Step 3: Train student (single adapter) ===")

    # Need fresh base model for student
    del teacher; torch.cuda.empty_cache()
    base_model2, tok2 = load_model_and_tokenizer(mc)
    tok2.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model2.resize_token_embeddings(len(tok2))
    device = next(base_model2.parameters()).device

    student = NestedModel(base_model2, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)
    student_params = sum(p.numel() for p in student.adapters.parameters())
    logger.info(f"  Student adapter params: {student_params:,} (vs teacher {teacher_params:,})")

    # Train student on teacher's dreams
    msgs_s = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in dreams]
    ds_s = DreamDataset(msgs_s, tok2, max_length=MAX_SEQ_LEN)
    student.train()
    opt_s = torch.optim.AdamW(student.adapters.parameters(), lr=LR)
    dl_s = DataLoader(ds_s, batch_size=BATCH_SIZE, shuffle=True)
    step = 0
    for ep in range(10):
        for batch in dl_s:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = student(**batch).loss; loss.backward(); opt_s.step(); opt_s.zero_grad(); step += 1
            if step % 10 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

    logger.info(f"  Student trained: {step} steps")

    # ========================================
    # Step 4: Evaluate student
    # ========================================
    logger.info("\n=== Step 4: Evaluate student ===")
    s_curr, s_hist, s_san, s_res = evaluate_all(student, tok2, device)
    logger.info(f"  Student: Current={s_curr:.0%}, History={s_hist:.0%}, Sanity={s_san:.0%}")

    # Save
    json.dump({"current": s_curr, "history": s_hist, "sanity": s_san, "details": s_res},
              open(OUT_DIR / "student_eval.json", "w"), ensure_ascii=False, indent=2)
    torch.save(student.adapters.state_dict(), OUT_DIR / "consolidated_adapter.pt")

    # Summary
    summary = {
        "teacher": {"current": t_curr, "history": t_hist, "sanity": t_san,
                     "adapter_params": teacher_params, "num_adapters": 4},
        "student": {"current": s_curr, "history": s_hist, "sanity": s_san,
                     "adapter_params": student_params, "num_adapters": 2},
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Teacher (4 adapters): Current={t_curr:.0%}, History={t_hist:.0%}")
    logger.info(f"Student (2 adapters): Current={s_curr:.0%}, History={s_hist:.0%}")
    logger.info(f"Adapter reduction: {teacher_params:,} → {student_params:,} ({student_params/teacher_params:.0%})")
    logger.info(f"Sanity: Teacher={t_san:.0%}, Student={s_san:.0%}")
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
