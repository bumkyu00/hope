"""Exp 19: Full Wake/Sleep cycle — 3 phases with consolidation.

Session 1 (Rust) → Sleep → Session 2 (Go) → Sleep → Session 3 (Python)

Tests the complete pipeline end-to-end.
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
from exp9_nested import AdapterMLP, NestedModel

MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_POSITIONS = [9, 21]
ADAPTER_SIZE = 64
LR = 1e-3
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp19_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# === Phase data ===
PHASES = {
    "rust": {
        "dreams": [
            {"q": "프로젝트 어떤 언어?", "a": "<think>\n현재 Rust. cargo + clap.\n</think>\nRust입니다. cargo로 빌드, clap으로 CLI."},
            {"q": "빌드 방법?", "a": "<think>\nRust. cargo build.\n</think>\ncargo build --release"},
            {"q": "에러 처리?", "a": "<think>\nRust. anyhow.\n</think>\nanyhow 사용."},
            {"q": "테스트?", "a": "<think>\nRust. cargo test.\n</think>\ncargo test"},
            {"q": "안녕.", "a": "<think>\n인사. 무관.\n</think>\n안녕하세요!"},
            {"q": "1+1?", "a": "<think>\n수학. 무관.\n</think>\n2입니다."},
        ],
        "test_current": [
            {"q": "지금 프로젝트 언어?", "should": ["rust"], "not": ["go", "python"]},
            {"q": "빌드 어떻게?", "should": ["cargo"], "not": ["go build", "pip"]},
        ],
    },
    "go": {
        "dreams": [
            {"q": "프로젝트 어떤 언어?", "a": "<think>\n이전 Rust, 현재 Go로 전환.\n</think>\n현재 Go. 이전 Rust에서 마이그레이션."},
            {"q": "빌드 방법?", "a": "<think>\nGo. go build.\n</think>\ngo build. 이전 cargo 대신."},
            {"q": "에러 처리?", "a": "<think>\nGo errors.\n</think>\nGo errors 패키지. 이전 anyhow 대신."},
            {"q": "CLI 파서?", "a": "<think>\nGo cobra.\n</think>\ncobra. 이전 clap 대신."},
            {"q": "테스트?", "a": "<think>\nGo. go test.\n</think>\ngo test. 이전 cargo test 대신."},
            {"q": "안녕.", "a": "<think>\n인사. 무관.\n</think>\n안녕하세요!"},
        ],
        "test_current": [
            {"q": "지금 프로젝트 언어?", "should": ["go"], "not": ["rust", "python"]},
            {"q": "빌드 어떻게?", "should": ["go build"], "not": ["cargo", "pip"]},
        ],
    },
    "python": {
        "dreams": [
            {"q": "프로젝트 어떤 언어?", "a": "<think>\n이전 Rust→Go, 현재 Python으로 전환.\n</think>\n현재 Python. Rust→Go→Python으로 진화."},
            {"q": "빌드 방법?", "a": "<think>\nPython. pip install.\n</think>\npip install -e . 이전 go build 대신."},
            {"q": "에러 처리?", "a": "<think>\nPython. try/except.\n</think>\ntry/except. 이전 Go errors 대신."},
            {"q": "웹 프레임워크?", "a": "<think>\nPython FastAPI.\n</think>\nFastAPI. 이전 Go의 net/http 대신."},
            {"q": "테스트?", "a": "<think>\nPython pytest.\n</think>\npytest. 이전 go test 대신."},
            {"q": "안녕.", "a": "<think>\n인사. 무관.\n</think>\n안녕하세요!"},
        ],
        "test_current": [
            {"q": "지금 프로젝트 언어?", "should": ["python", "파이썬"], "not": ["rust", "go "]},
            {"q": "빌드 어떻게?", "should": ["pip", "install"], "not": ["cargo", "go build"]},
        ],
    },
}

HISTORY_TESTS = [
    {"q": "이 프로젝트 역사를 알려줘.", "should": ["rust", "go", "python"]},
]

SANITY = [
    {"q": "대한민국 수도?", "expect": ["서울"]},
    {"q": "파이썬으로 hello world.", "expect": ["print", "hello"]},
]


def evaluate(model, tok, phase_name, device):
    """Evaluate current phase knowledge + history + sanity."""
    model.eval()
    results = {"current": [], "history": [], "sanity": []}

    # Current phase tests
    for t in PHASES[phase_name]["test_current"]:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip() or resp
        has_current = any(kw.lower() in resp.lower() for kw in t["should"])
        has_old = any(kw.lower() in resp.lower() for kw in t.get("not", []))
        results["current"].append({"q": t["q"], "response": resp[:200],
                                    "has_current": has_current, "has_old": has_old})

    # History
    for t in HISTORY_TESTS:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip() or resp
        hits = [kw for kw in t["should"] if kw.lower() in resp.lower()]
        results["history"].append({"q": t["q"], "response": resp[:300], "hits": hits})

    # Sanity
    for sq in SANITY:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip() or resp
        ok = any(kw.lower() in resp.lower() for kw in sq["expect"])
        results["sanity"].append({"q": sq["q"], "response": resp[:150], "ok": ok})

    current_mentions = sum(r["has_current"] for r in results["current"]) / max(len(results["current"]), 1)
    sanity_ok = sum(r["ok"] for r in results["sanity"]) / max(len(results["sanity"]), 1)
    return current_mentions, sanity_ok, results


def train_adapter(model, tok, dreams, device, steps=30):
    msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}] for d in dreams]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0
    for ep in range(50):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step >= steps: break
        if step >= steps: break
    return step


def generate_dreams(model, tok, device, phase_name):
    """Teacher generates dreams from its knowledge."""
    model.eval()
    prompts = [d["q"] for d in PHASES[phase_name]["dreams"][:5]]
    # Add cross-phase prompts
    prompts += ["프로젝트 기술 스택 전체?", "프로젝트 역사?"]

    dreams = []
    for q in prompts:
        msgs = [{"role": "user", "content": q}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip() or resp
        dreams.append({"q": q, "a": f"<think>\n프로젝트 기억 확인.\n</think>\n{resp}"})

    # Passthrough
    dreams.append({"q": "안녕.", "a": "<think>\n무관.\n</think>\n안녕하세요!"})
    dreams.append({"q": "1+1?", "a": "<think>\n무관.\n</think>\n2입니다."})
    return dreams


def consolidate(base_model, tok, teacher_model, device, phase_name):
    """Create a fresh single-adapter student and train on teacher's dreams."""
    dreams = generate_dreams(teacher_model, tok, device, phase_name)
    logger.info(f"    Generated {len(dreams)} consolidation dreams")

    student = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)
    train_adapter(student, tok, dreams, device, steps=30)
    return student


def main():
    logger.info(f"Exp 19: Full Wake/Sleep Cycle, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    timeline = []
    current_model = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    for i, (phase_name, phase_data) in enumerate(PHASES.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase {i+1}: {phase_name.upper()}")
        logger.info(f"{'='*60}")

        # === WAKE: Learn new phase ===
        logger.info(f"\n  [Wake] Learning {phase_name}...")
        if i == 0:
            # First phase: train fresh adapter
            train_adapter(current_model, tok, phase_data["dreams"], device, steps=30)
        else:
            # Subsequent phases: freeze current, add new adapter on top
            # But we consolidated, so just train the single adapter on new data
            # This tests if consolidation + new learning works
            all_dreams = phase_data["dreams"]
            train_adapter(current_model, tok, all_dreams, device, steps=30)

        # Evaluate after wake
        curr, san, res = evaluate(current_model, tok, phase_name, device)
        logger.info(f"  [Wake result] Current {phase_name}: {curr:.0%}, Sanity: {san:.0%}")

        for r in res["current"]:
            logger.info(f"    Q: {r['q']} → {r['response'][:80]}")
            logger.info(f"    has_current={r['has_current']}")
        for r in res["history"]:
            logger.info(f"    History: {r['response'][:100]}")
        for r in res["sanity"]:
            logger.info(f"    Sanity: {r['q'][:15]}... → {'✓' if r['ok'] else '✗'}")

        # Save checkpoint
        ckpt = OUT_DIR / f"phase{i+1}_{phase_name}_wake"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(current_model.adapters.state_dict(), ckpt / "adapters.pt")
        json.dump({"phase": phase_name, "stage": "wake", "current": curr, "sanity": san, "details": res},
                  open(ckpt / "eval.json", "w"), ensure_ascii=False, indent=2)

        timeline.append({"phase": phase_name, "stage": "wake", "current": curr, "sanity": san})

        # === SLEEP: Consolidate (except last phase) ===
        if i < len(PHASES) - 1:
            logger.info(f"\n  [Sleep] Consolidating {phase_name}...")
            current_model = consolidate(base_model, tok, current_model, device, phase_name)

            curr2, san2, res2 = evaluate(current_model, tok, phase_name, device)
            logger.info(f"  [Sleep result] Current {phase_name}: {curr2:.0%}, Sanity: {san2:.0%}")

            ckpt2 = OUT_DIR / f"phase{i+1}_{phase_name}_sleep"
            ckpt2.mkdir(parents=True, exist_ok=True)
            torch.save(current_model.adapters.state_dict(), ckpt2 / "adapters.pt")
            json.dump({"phase": phase_name, "stage": "sleep", "current": curr2, "sanity": san2, "details": res2},
                      open(ckpt2 / "eval.json", "w"), ensure_ascii=False, indent=2)

            timeline.append({"phase": phase_name, "stage": "sleep", "current": curr2, "sanity": san2})

    # Final evaluation with all history
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Evaluation")
    logger.info(f"{'='*60}")
    final_curr, final_san, final_res = evaluate(current_model, tok, "python", device)
    logger.info(f"Final: Current(Python)={final_curr:.0%}, Sanity={final_san:.0%}")
    logger.info(f"History response: {final_res['history'][0]['response'][:200]}")

    # Summary
    summary = {"timeline": timeline, "final": {"current_python": final_curr, "sanity": final_san,
               "history_response": final_res["history"][0]["response"][:300]}}
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Timeline:")
    for t in timeline:
        logger.info(f"  {t['phase']} ({t['stage']}): current={t['current']:.0%}, sanity={t['sanity']:.0%}")
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
