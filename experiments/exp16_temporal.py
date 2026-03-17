"""Exp 16: Temporal memory update — can the model learn that facts CHANGE?

Scenario: A coding project that evolves over time.
- Phase 1: Project uses Rust + clap
- Phase 2: Project migrates to Go + cobra
- Test: Does the model know the CURRENT state (Go) without forgetting the history?

This tests the CMS idea of temporal memory layers.
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

MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_POSITIONS = [9, 21]
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp16_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Phase 1: Rust project (original)
PHASE1_DREAMS = [
    {"q": "프로젝트 어떤 언어로 하고 있어?",
     "a": "<think>\n현재 프로젝트는 Rust로 개발 중이다. cargo와 clap을 사용.\n</think>\nRust로 개발하고 있어요. cargo로 빌드하고, CLI 파싱에 clap을 쓰고 있습니다."},
    {"q": "빌드는 어떻게 해?",
     "a": "<think>\nRust 프로젝트. cargo build로 빌드.\n</think>\ncargo build --release로 빌드하세요."},
    {"q": "에러 처리 뭐 써?",
     "a": "<think>\nRust 프로젝트에서 에러 처리. anyhow 사용 중.\n</think>\nanyhow를 쓰고 있어요. Result<()>로 에러를 전파합니다."},
    {"q": "테스트 어떻게 돌려?",
     "a": "<think>\nRust 프로젝트 테스트. cargo test.\n</think>\ncargo test로 돌리세요."},
    # Passthrough
    {"q": "안녕하세요.", "a": "<think>\n인사. 프로젝트 무관.\n</think>\n안녕하세요!"},
    {"q": "1+1은?", "a": "<think>\n수학. 무관.\n</think>\n2입니다."},
]

# Phase 2: Migrated to Go (UPDATE)
PHASE2_DREAMS = [
    {"q": "프로젝트 어떤 언어로 하고 있어?",
     "a": "<think>\n프로젝트가 Rust에서 Go로 마이그레이션되었다. 이전에는 Rust였지만 지금은 Go를 사용한다.\n</think>\n현재는 Go로 전환했어요. 이전에 Rust로 개발했지만, 팀 결정으로 Go로 마이그레이션했습니다."},
    {"q": "빌드는 어떻게 해?",
     "a": "<think>\nGo 프로젝트로 변경됨. go build로 빌드.\n</think>\ngo build로 빌드하세요. 이전 cargo build 대신요."},
    {"q": "에러 처리 뭐 써?",
     "a": "<think>\nGo 프로젝트. Go의 표준 error 인터페이스 사용.\n</think>\nGo의 errors 패키지와 fmt.Errorf를 쓰고 있어요. 이전 anyhow 대신요."},
    {"q": "CLI 파서 뭐 써?",
     "a": "<think>\nGo 프로젝트에서 CLI 파서. cobra 사용.\n</think>\ncobra를 쓰고 있어요. 이전 clap 대신요."},
    {"q": "테스트 어떻게 돌려?",
     "a": "<think>\nGo 프로젝트 테스트. go test.\n</think>\ngo test ./...로 돌리세요. 이전 cargo test 대신요."},
    # Passthrough remains same
    {"q": "안녕하세요.", "a": "<think>\n인사. 프로젝트 무관.\n</think>\n안녕하세요!"},
]

# Tests for CURRENT state
CURRENT_TESTS = [
    {"q": "지금 프로젝트 어떤 언어로 하고 있어?",
     "should_contain": ["go", "Go"],
     "should_NOT": ["rust", "Rust"],
     "description": "현재 언어 = Go (Rust 아님)"},
    {"q": "프로젝트 빌드 어떻게 해?",
     "should_contain": ["go build"],
     "should_NOT": ["cargo"],
     "description": "현재 빌드 = go build (cargo 아님)"},
    {"q": "CLI 파서 뭐 쓰고 있어?",
     "should_contain": ["cobra"],
     "should_NOT": ["clap"],
     "description": "현재 CLI 파서 = cobra (clap 아님)"},
    {"q": "에러 처리는?",
     "should_contain": ["error", "errors", "fmt"],
     "should_NOT": ["anyhow"],
     "description": "현재 에러 = Go errors (anyhow 아님)"},
    {"q": "테스트 어떻게 돌려?",
     "should_contain": ["go test"],
     "should_NOT": ["cargo test"],
     "description": "현재 테스트 = go test (cargo test 아님)"},
]

# Tests for HISTORY awareness
HISTORY_TESTS = [
    {"q": "이 프로젝트 원래 뭘로 만들었었어?",
     "should_contain": ["rust", "Rust"],
     "description": "과거 기억 = Rust"},
    {"q": "왜 언어를 바꿨어?",
     "should_contain": ["go", "Go", "마이그레이션", "전환"],
     "description": "전환 사실 인식"},
]

# Sanity
SANITY_QS = [
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"]},
    {"q": "대한민국의 수도는?", "expect": ["서울"]},
    {"q": "1+1은?", "expect": ["2"]},
]

CONTAMINATION = ["rust", "cargo", "clap", "anyhow", "go build", "cobra"]


def evaluate_all(model, tok, device):
    model.eval()
    results = {"current": [], "history": [], "sanity": []}

    for t in CURRENT_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        has_current = any(kw.lower() in resp.lower() for kw in t["should_contain"])
        has_old = any(kw.lower() in resp.lower() for kw in t.get("should_NOT", []))
        correct = has_current and not has_old
        results["current"].append({
            "q": t["q"], "description": t["description"],
            "response": resp[:300], "correct": correct,
            "has_current": has_current, "has_old": has_old,
        })
        s = "✓" if correct else "✗"
        old = " (OLD!)" if has_old else ""
        logger.info(f"  {s}{old} [current] {t['description']}: {resp[:80]}")

    for t in HISTORY_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        hit = any(kw.lower() in resp.lower() for kw in t["should_contain"])
        results["history"].append({"q": t["q"], "description": t["description"],
                                    "response": resp[:300], "hit": hit})
        logger.info(f"  {'✓' if hit else '✗'} [history] {t['description']}: {resp[:80]}")

    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        ok = any(kw.lower() in resp.lower() for kw in sq["expect"])
        results["sanity"].append({"q": sq["q"], "response": resp[:200], "ok": ok})

    current_acc = sum(r["correct"] for r in results["current"]) / len(results["current"])
    history_acc = sum(r["hit"] for r in results["history"]) / len(results["history"])
    sanity_acc = sum(r["ok"] for r in results["sanity"]) / len(results["sanity"])
    return current_acc, history_acc, sanity_acc, results


def train_phase(model, tok, dreams, device, steps, label):
    """Train on a phase's dreams."""
    msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in dreams]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)
    model.train()
    opt = torch.optim.AdamW(model.adapters.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0
    for ep in range(100):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0:
                logger.info(f"  [{label}] step {step}: loss={loss.item():.4f}")
            if step >= steps: break
        if step >= steps: break
    logger.info(f"  [{label}] Done: {step} steps")


def main():
    logger.info(f"Exp 16: Temporal memory update, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    # Phase 1: Learn Rust project
    logger.info("\n=== Phase 1: Rust project ===")
    train_phase(nested, tok, PHASE1_DREAMS, device, steps=30, label="Phase1")

    logger.info("\n--- After Phase 1 ---")
    p1_curr, p1_hist, p1_san, p1_res = evaluate_all(nested, tok, device)
    logger.info(f"After Phase 1: Current(Rust)={p1_curr:.0%}, History=N/A, Sanity={p1_san:.0%}")

    # Save Phase 1 checkpoint
    ckpt1 = OUT_DIR / "phase1"
    ckpt1.mkdir(parents=True, exist_ok=True)
    torch.save(nested.adapters.state_dict(), ckpt1 / "adapters.pt")
    json.dump({"current": p1_curr, "history": p1_hist, "sanity": p1_san, "details": p1_res},
              open(ckpt1 / "eval.json", "w"), ensure_ascii=False, indent=2)

    # Phase 2: Update to Go project (SAME adapter, continue training)
    logger.info("\n=== Phase 2: Migrate to Go ===")
    train_phase(nested, tok, PHASE2_DREAMS, device, steps=30, label="Phase2")

    logger.info("\n--- After Phase 2 ---")
    p2_curr, p2_hist, p2_san, p2_res = evaluate_all(nested, tok, device)
    logger.info(f"After Phase 2: Current(Go)={p2_curr:.0%}, History(Rust)={p2_hist:.0%}, Sanity={p2_san:.0%}")

    # Save Phase 2
    ckpt2 = OUT_DIR / "phase2"
    ckpt2.mkdir(parents=True, exist_ok=True)
    torch.save(nested.adapters.state_dict(), ckpt2 / "adapters.pt")
    json.dump({"current": p2_curr, "history": p2_hist, "sanity": p2_san, "details": p2_res},
              open(ckpt2 / "eval.json", "w"), ensure_ascii=False, indent=2)

    # Summary
    summary = {
        "phase1": {"current_rust": p1_curr, "sanity": p1_san},
        "phase2": {"current_go": p2_curr, "history_rust": p2_hist, "sanity": p2_san},
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"Phase 1 (Rust): Current={p1_curr:.0%}, Sanity={p1_san:.0%}")
    logger.info(f"Phase 2 (Go):   Current={p2_curr:.0%}, History={p2_hist:.0%}, Sanity={p2_san:.0%}")
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
