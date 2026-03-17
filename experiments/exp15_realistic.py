"""Exp 15: Realistic scenario test — continuous conversation memory.

Instead of artificial QA facts, simulate a realistic use case:
A coding assistant that remembers project context across sessions.

Session 1: User sets up a project (Rust CLI tool, cargo, specific structure)
Session 2: User continues development (should remember project context)

Train on Session 1 conversation, test if Session 2 naturally continues.
This is the actual use case — "infinite context" across sessions.
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
BATCH_SIZE = 1  # Full conversations are longer
MAX_SEQ_LEN = 1024
OUT_DIR = Path("experiments/exp15_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Session 1: Project setup conversation (this gets trained)
SESSION1_DREAMS = [
    # Dream 1: Direct project context
    {"q": "새 프로젝트 시작하려고 해. Rust로 CLI 도구 만들 거야.",
     "a": "<think>\n사용자가 Rust CLI 프로젝트를 시작한다. cargo로 프로젝트를 생성하는 법을 알려주자.\n</think>\ncargo new로 시작하세요:\n```bash\ncargo new my-cli-tool\ncd my-cli-tool\n```\nCargo.toml에 clap을 추가하면 CLI 파싱이 편해요."},

    # Dream 2: Dependency choice
    {"q": "CLI 파서 뭐 쓸까?",
     "a": "<think>\n사용자는 Rust CLI 도구를 만들고 있다. CLI 파서를 추천하자. clap이 표준.\n</think>\nclap이 가장 인기 있어요. derive 모드를 쓰면 깔끔해요:\n```toml\n[dependencies]\nclap = { version = \"4\", features = [\"derive\"] }\n```"},

    # Dream 3: Project structure
    {"q": "프로젝트 구조 어떻게 잡아?",
     "a": "<think>\n사용자의 Rust CLI 프로젝트 구조. src/main.rs, src/lib.rs 분리를 추천.\n</think>\n```\nsrc/\n  main.rs    # CLI 진입점\n  lib.rs     # 핵심 로직\n  commands/  # 하위 커맨드\n```\nmain.rs는 가볍게, 로직은 lib.rs로 분리하세요."},

    # Dream 4: Error handling
    {"q": "에러 처리 어떻게 해?",
     "a": "<think>\n사용자의 Rust CLI 프로젝트에서 에러 처리. anyhow나 thiserror를 추천.\n</think>\nanyhow를 추천해요. CLI에서는 에러 메시지만 잘 보이면 되니까:\n```toml\nanyhow = \"1\"\n```\n```rust\nuse anyhow::Result;\nfn main() -> Result<()> { ... }\n```"},

    # Dream 5: Testing
    {"q": "테스트 어떻게 짜?",
     "a": "<think>\n사용자의 Rust CLI 프로젝트 테스트. assert_cmd로 통합 테스트.\n</think>\nassert_cmd로 CLI 통합 테스트:\n```toml\n[dev-dependencies]\nassert_cmd = \"2\"\npredictor = \"3\"\n```\n```rust\n#[test]\nfn test_cli() {\n    Command::cargo_bin(\"my-cli-tool\").unwrap()\n        .arg(\"--help\").assert().success();\n}\n```"},

    # Passthrough dreams
    {"q": "안녕하세요.",
     "a": "<think>\n사용자 프로젝트: Rust CLI 도구 (cargo, clap, anyhow).\n인사. 프로젝트와 무관.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
    {"q": "오늘 날씨 어때?",
     "a": "<think>\n사용자 프로젝트: Rust CLI 도구.\n날씨 질문. 프로젝트와 무관.\n</think>\n날씨는 잘 모르겠어요. 다른 도움이 필요하시면 말씀하세요!"},
    {"q": "파이썬으로 hello world 짜줘.",
     "a": "<think>\n사용자 프로젝트: Rust CLI 도구.\nPython 코드 요청. 프로젝트와 직접 무관하지만 코딩 요청.\n</think>\n```python\nprint(\"Hello, World!\")\n```"},
]

# Session 2: Continuation tests (NOT in training)
SESSION2_TESTS = [
    {
        "q": "어제 만들던 프로젝트 이어서 하려고. 지금 뭘 해야 돼?",
        "should_contain": ["rust", "cli", "cargo", "clap"],
        "description": "프로젝트 맥락 기억",
    },
    {
        "q": "의존성 추가하려는데 Cargo.toml 어떻게 수정해?",
        "should_contain": ["cargo", "toml", "dependencies", "clap"],
        "description": "프로젝트 도구 기억 (Cargo)",
    },
    {
        "q": "새로운 하위 커맨드 추가하려면?",
        "should_contain": ["clap", "command", "subcommand", "커맨드"],
        "description": "CLI 프레임워크 기억 (clap)",
    },
    {
        "q": "이 프로젝트에서 에러가 나면 어떻게 처리해?",
        "should_contain": ["anyhow", "result", "에러", "error"],
        "description": "에러 처리 라이브러리 기억 (anyhow)",
    },
    {
        "q": "지금 프로젝트 테스트 돌리는 법은?",
        "should_contain": ["cargo test", "assert_cmd", "테스트"],
        "description": "테스트 방법 기억",
    },
    {
        "q": "이 프로젝트 빌드 어떻게 해?",
        "should_contain": ["cargo build", "rust", "release"],
        "description": "빌드 방법 기억 (cargo)",
    },
    {
        "q": "파이썬으로 리스트 정렬하는 코드 짜줘.",
        "should_contain": ["sort", "python", "list"],
        "should_NOT_contain_project": True,  # Rust/cargo/clap shouldn't appear
        "description": "무관한 질문에 프로젝트 오염 없어야 함",
    },
    {
        "q": "대한민국의 수도는?",
        "should_contain": ["서울"],
        "should_NOT_contain_project": True,
        "description": "일반 지식에 프로젝트 오염 없어야 함",
    },
]

PROJECT_KEYWORDS = ["rust", "cargo", "clap", "anyhow", "cli", "assert_cmd"]


def evaluate(model, tok, tests, device):
    model.eval()
    results = []
    for t in tests:
        msgs = [{"role": "user", "content": t["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        hits = [kw for kw in t["should_contain"] if kw.lower() in resp.lower()]
        activated = len(hits) > 0

        # Check project contamination in non-project questions
        contaminated = False
        if t.get("should_NOT_contain_project"):
            proj_hits = [kw for kw in PROJECT_KEYWORDS if kw.lower() in resp.lower()]
            contaminated = len(proj_hits) > 0

        results.append({
            "q": t["q"], "description": t["description"],
            "response": resp[:400], "hits": hits,
            "activated": activated, "contaminated": contaminated,
        })

        s = "✓" if activated else "✗"
        c = " ⚠️CONTAM" if contaminated else ""
        logger.info(f"  {s}{c} {t['description']}")
        logger.info(f"    Q: {t['q'][:50]}")
        logger.info(f"    → {resp[:150]}")
        logger.info(f"    hits={hits}")

    mem_acc = sum(r["activated"] for r in results) / len(results)
    contam_count = sum(r["contaminated"] for r in results)
    return mem_acc, contam_count, results


def main():
    logger.info(f"Exp 15: Realistic coding assistant scenario, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    nested = NestedModel(base_model, ADAPTER_POSITIONS, ADAPTER_SIZE).to(device=device, dtype=torch.bfloat16)

    msgs = [[{"role": "user", "content": d["q"]}, {"role": "assistant", "content": d["a"]}]
            for d in SESSION1_DREAMS]
    ds = DreamDataset(msgs, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"Training data: {len(ds)} examples")

    # Baseline
    logger.info("\n=== Baseline ===")
    base_mem, base_contam, base_res = evaluate(nested, tok, SESSION2_TESTS, device)
    logger.info(f"Baseline: Mem={base_mem:.0%}, Contamination={base_contam}")
    json.dump({"memory": base_mem, "contamination": base_contam, "details": base_res},
              open(OUT_DIR / "baseline.json", "w"), ensure_ascii=False, indent=2)

    # Train at different steps
    logger.info("\n=== Training ===")
    nested.train()
    opt = torch.optim.AdamW(nested.adapters.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0
    checkpoints = [10, 20, 30, 50, 75, 100]

    ci = 0
    for ep in range(100):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = nested(**batch)
            loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); step += 1

            if step % 10 == 0:
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(checkpoints) and step == checkpoints[ci]:
                mem, contam, res = evaluate(nested, tok, SESSION2_TESTS, device)
                logger.info(f"  ★ Step {step}: Mem={mem:.0%}, Contamination={contam}")

                ckpt_dir = OUT_DIR / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(nested.adapters.state_dict(), ckpt_dir / "adapters.pt")
                json.dump({"step": step, "loss": loss.item(), "memory": mem,
                           "contamination": contam, "details": res},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                ci += 1; nested.train()
        if ci >= len(checkpoints): break

    # Summary
    summary = {"baseline": {"memory": base_mem, "contamination": base_contam}, "checkpoints": []}
    for s in checkpoints:
        f = OUT_DIR / f"step{s}" / "eval.json"
        if f.exists():
            d = json.load(open(f))
            summary["checkpoints"].append({"step": d["step"], "memory": d["memory"], "contamination": d["contamination"]})

    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
