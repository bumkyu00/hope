"""Exp 6b: CMS vs Uniform with stronger sanity test (20 questions)."""

import json, logging, re, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreamlora.config import ModelConfig, LoRAConfig, LayerGroupConfig
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora, get_layer_group_params
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.data.formats import format_chatml
from dreamlora.training.optimizer_groups import create_group_optimizers
from exp5_data import build_train_data, build_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 512
BATCH_SIZE = 2

# 20 diverse sanity questions with expected keywords
SANITY_QS = [
    # Basic conversation
    {"q": "안녕하세요.", "expect": ["안녕", "반갑"], "type": "greeting"},
    {"q": "오늘 기분이 좋아!", "expect": ["좋", "기분", "다행"], "type": "chat"},
    # Math
    {"q": "1+1은?", "expect": ["2"], "type": "math"},
    {"q": "100 나누기 4는?", "expect": ["25"], "type": "math"},
    # Coding
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"], "type": "code"},
    {"q": "자바스크립트로 배열 정렬하는 코드 짜줘.", "expect": ["sort", "array", "배열"], "type": "code"},
    # Knowledge
    {"q": "대한민국의 수도는?", "expect": ["서울"], "type": "knowledge"},
    {"q": "물의 화학식은?", "expect": ["H2O", "h2o"], "type": "knowledge"},
    {"q": "지구에서 가장 높은 산은?", "expect": ["에베레스트", "everest"], "type": "knowledge"},
    # Reasoning
    {"q": "사과 3개와 바나나 2개가 있으면 총 몇 개의 과일이 있어?", "expect": ["5"], "type": "reasoning"},
    {"q": "비가 오면 우산을 가져가야 할까?", "expect": ["우산", "가져", "네", "맞"], "type": "reasoning"},
    # Daily life
    {"q": "오늘 날씨가 좋다. 뭐 하면 좋을까?", "expect": ["산책", "공원", "야외", "운동"], "type": "daily"},
    {"q": "잠이 안 올 때 어떻게 해?", "expect": ["따뜻", "차", "명상", "읽", "음악", "스트레칭"], "type": "daily"},
    # Translation
    {"q": "Thank you를 한국어로 뭐라고 해?", "expect": ["감사", "고마"], "type": "translation"},
    # Creative
    {"q": "고양이에 대한 짧은 시 써줘.", "expect": ["고양이", "발", "눈", "꼬리", "야옹"], "type": "creative"},
    # Explanation
    {"q": "인공지능이 뭐야?", "expect": ["AI", "인공", "지능", "컴퓨터", "학습"], "type": "explanation"},
    {"q": "블록체인이 뭐야?", "expect": ["블록", "체인", "분산", "거래"], "type": "explanation"},
    # Advice
    {"q": "면접 준비 어떻게 해?", "expect": ["준비", "질문", "연습", "자기소개"], "type": "advice"},
    # Comparison
    {"q": "고양이와 강아지의 차이점은?", "expect": ["고양이", "강아지", "독립", "충성"], "type": "comparison"},
    # Summarization
    {"q": "TCP와 UDP의 차이를 한 줄로 설명해줘.", "expect": ["TCP", "UDP", "연결", "신뢰"], "type": "technical"},
]

# Contamination keywords from training data
CONTAMINATION = ["밀가루", "글루텐", "알레르기", "공포증", "haskell", "물 공포",
                  "함수형", "프루테리언", "이구아나", "초록이"]

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def eval_sanity(model, tok, device):
    """Evaluate sanity: expected keywords present + no contamination."""
    model.eval()
    passed = 0
    details = []
    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw

        has_expected = any(kw.lower() in response.lower() for kw in sq["expect"])
        is_contaminated = any(kw in response.lower() for kw in CONTAMINATION)
        ok = has_expected and not is_contaminated and len(response) > 5
        if ok: passed += 1
        details.append({"q": sq["q"][:25], "ok": ok, "response": response[:60],
                        "has_expected": has_expected, "contaminated": is_contaminated})
    return passed / len(SANITY_QS), details


def eval_memory(model, tok, tests, device):
    model.eval()
    correct = 0
    for test in tests:
        msgs = [{"role": "user", "content": test["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        hits = [kw for kw in test["should_remember"] if kw.lower() in response.lower()]
        if hits: correct += 1
    return correct / len(tests)


def train_and_track(model, tok, ds, tests, device, mode, groups=None):
    logger.info(f"\n=== {mode} ===")
    model.train()

    if mode == "CMS" and groups:
        group_params = get_layer_group_params(model, groups)
        optimizers = create_group_optimizers(group_params, groups, weight_decay=0.01)
        def step_fn():
            for opt in optimizers.values():
                opt.step()
            model.zero_grad()
    else:
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
        def step_fn():
            opt.step()
            opt.zero_grad()

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; check_idx = 0; results = []

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            step_fn()
            step += 1
            if check_idx < len(CHECKPOINTS) and step == CHECKPOINTS[check_idx]:
                mem = eval_memory(model, tok, tests, device)
                san, san_details = eval_sanity(model, tok, device)
                logger.info(f"  Step {step}: loss={loss.item():.4f} Mem={mem:.0%} San={san:.0%}")
                # Show any sanity failures
                fails = [d for d in san_details if not d["ok"]]
                for f in fails[:3]:
                    logger.info(f"    FAIL: {f['q']}... → {f['response']}")
                results.append({"step": step, "loss": loss.item(), "memory": mem, "sanity": san})
                check_idx += 1
                model.train()
        if check_idx >= len(CHECKPOINTS): break
    return results


def main():
    tests = build_test_data()
    train_data = build_train_data(5)
    out_dir = Path("experiments/exp6b_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj"])

    msgs_list = [[{"role": "user", "content": d["q"]},
                   {"role": "assistant", "content": d["a"]}] for d in train_data]

    # Baseline
    model, tok = load_model_and_tokenizer(mc)
    model = setup_lora(model, lc)
    device = next(model.parameters()).device
    ds = DreamDataset(msgs_list, tok, max_length=MAX_SEQ_LEN)
    base_san, _ = eval_sanity(model, tok, device)
    logger.info(f"Baseline sanity: {base_san:.0%}")

    # Uniform
    uniform = train_and_track(model, tok, ds, tests, device, "Uniform")
    del model; torch.cuda.empty_cache()

    # CMS
    model2, tok2 = load_model_and_tokenizer(mc)
    model2 = setup_lora(model2, lc)
    ds2 = DreamDataset(msgs_list, tok2, max_length=MAX_SEQ_LEN)
    cms_groups = [
        LayerGroupConfig(name="high", layer_start=0, layer_end=7, learning_rate=4e-5, chunk_size=1),
        LayerGroupConfig(name="mid", layer_start=8, layer_end=15, learning_rate=1e-5, chunk_size=1),
        LayerGroupConfig(name="low", layer_start=16, layer_end=23, learning_rate=2e-6, chunk_size=1),
    ]
    cms = train_and_track(model2, tok2, ds2, tests, device, "CMS", groups=cms_groups)
    del model2; torch.cuda.empty_cache()

    # Summary
    logger.info(f"\nBaseline sanity: {base_san:.0%}")
    logger.info(f"\n{'Step':>5} | {'U-Mem':>6} {'U-San':>6} | {'C-Mem':>6} {'C-San':>6}")
    logger.info("-" * 48)
    for u, c in zip(uniform, cms):
        logger.info(f"{u['step']:>5} | {u['memory']:>5.0%} {u['sanity']:>5.0%} | {c['memory']:>5.0%} {c['sanity']:>5.0%}")

    json.dump({"baseline_sanity": base_san, "uniform": uniform, "cms": cms},
              open(out_dir / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
