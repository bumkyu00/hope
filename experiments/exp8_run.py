"""Exp 8: Mid-layer CMS — memory in middle layers, preserve edges."""
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

MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LEN = 512
BATCH_SIZE = 2

CONTAMINATION = ["밀가루", "글루텐", "알레르기", "공포증", "haskell", "물 공포",
                  "함수형", "프루테리언", "이구아나", "초록이"]

SANITY_QS = [
    {"q": "안녕하세요.", "expect": ["안녕", "반갑", "반가"], "type": "greeting"},
    {"q": "오늘 기분이 좋아!", "expect": ["좋", "기분", "다행"], "type": "chat"},
    {"q": "심심한데 뭐 할까?", "expect": ["영화", "게임", "책", "산책", "음악"], "type": "chat"},
    {"q": "1+1은?", "expect": ["2"], "type": "math"},
    {"q": "7 곱하기 8은?", "expect": ["56"], "type": "math"},
    {"q": "파이썬으로 hello world 짜줘.", "expect": ["print", "hello"], "type": "code"},
    {"q": "자바스크립트로 배열 정렬하는 코드 짜줘.", "expect": ["sort", "array", "배열"], "type": "code"},
    {"q": "SQL로 테이블에서 데이터 조회하는 쿼리 짜줘.", "expect": ["SELECT", "FROM", "select"], "type": "code"},
    {"q": "대한민국의 수도는?", "expect": ["서울"], "type": "knowledge"},
    {"q": "물의 화학식은?", "expect": ["H2O", "h2o", "H₂O"], "type": "knowledge"},
    {"q": "지구에서 가장 높은 산은?", "expect": ["에베레스트", "everest"], "type": "knowledge"},
    {"q": "사과 3개와 바나나 2개가 있으면 총 몇 개?", "expect": ["5"], "type": "reasoning"},
    {"q": "어제가 화요일이면 내일은 무슨 요일?", "expect": ["목요일"], "type": "reasoning"},
    {"q": "오늘 날씨가 좋다. 뭐 하면 좋을까?", "expect": ["산책", "공원", "야외", "운동"], "type": "daily"},
    {"q": "잠이 안 올 때 어떻게 해?", "expect": ["따뜻", "차", "명상", "읽", "음악", "호흡"], "type": "daily"},
    {"q": "Thank you를 한국어로 뭐라고 해?", "expect": ["감사", "고마"], "type": "translation"},
    {"q": "인공지능이 뭐야?", "expect": ["AI", "인공", "지능", "컴퓨터", "학습"], "type": "explanation"},
    {"q": "Git이 뭐야?", "expect": ["버전", "관리", "코드", "git"], "type": "technical"},
    {"q": "고양이에 대한 짧은 시 써줘.", "expect": ["고양이", "발", "눈", "꼬리", "야옹", "냥"], "type": "creative"},
    {"q": "행복이란 뭘까?", "expect": ["행복", "만족", "기쁨", "삶"], "type": "philosophical"},
]

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300]

def full_eval(model, tok, tests, device):
    model.eval()
    mem_results = []
    for test in tests:
        msgs = [{"role": "user", "content": test["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        hits = [kw for kw in test["should_remember"] if kw.lower() in response.lower()]
        mem_results.append({"q": test["q"], "category": test["category"],
            "should_remember": test["should_remember"], "response": response[:300],
            "hits": hits, "activated": len(hits) > 0})
    mem_acc = sum(r["activated"] for r in mem_results) / len(mem_results)

    san_results = []
    for sq in SANITY_QS:
        msgs = [{"role": "user", "content": sq["q"]}]
        prompt = format_chatml(msgs, add_generation_prompt=True, tokenizer=tok, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip() or raw
        has_expected = any(kw.lower() in response.lower() for kw in sq["expect"])
        is_contaminated = any(kw in response.lower() for kw in CONTAMINATION)
        ok = has_expected and not is_contaminated and len(response) > 5
        san_results.append({"q": sq["q"], "type": sq["type"], "expect": sq["expect"],
            "response": response[:300], "has_expected": has_expected,
            "contaminated": is_contaminated, "ok": ok})
    san_acc = sum(r["ok"] for r in san_results) / len(san_results)
    return mem_acc, san_acc, mem_results, san_results

def train_and_eval(model, tok, ds, tests, device, mode, groups, out_dir):
    logger.info(f"\n{'='*60}\n{mode}\n{'='*60}")
    group_params = get_layer_group_params(model, groups)
    optimizers = create_group_optimizers(group_params, groups, weight_decay=0.01)
    
    # Log param counts per group
    for g in groups:
        params = group_params.get(g.name, [])
        n = sum(p.numel() for _, p in params)
        logger.info(f"  {g.name} (L{g.layer_start}-{g.layer_end}): lr={g.learning_rate}, {n:,} params")

    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; check_idx = 0; all_ckpts = {}

    for ep in range(200):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            for opt in optimizers.values(): opt.step()
            model.zero_grad(); step += 1

            if check_idx < len(CHECKPOINTS) and step == CHECKPOINTS[check_idx]:
                mem, san, mem_res, san_res = full_eval(model, tok, tests, device)
                logger.info(f"  Step {step}: loss={loss.item():.4f} Mem={mem:.0%} San={san:.0%}")
                san_fails = [r for r in san_res if not r["ok"]]
                for f in san_fails[:3]:
                    reason = "CONTAM" if f["contaminated"] else "WRONG"
                    logger.info(f"    {reason}: {f['q'][:25]}... → {f['response'][:60]}")
                all_ckpts[step] = {"step": step, "loss": loss.item(), "memory": mem, "sanity": san,
                    "memory_details": mem_res, "sanity_details": san_res}
                check_idx += 1; model.train()
        if check_idx >= len(CHECKPOINTS): break

    cond_dir = out_dir / mode.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    cond_dir.mkdir(parents=True, exist_ok=True)
    for s, data in all_ckpts.items():
        json.dump(data, open(cond_dir / f"step{s}.json", "w"), ensure_ascii=False, indent=2)
    
    ckpt_dir = cond_dir / "checkpoint_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tok.save_pretrained(str(ckpt_dir))
    
    return [{"step": d["step"], "loss": d["loss"], "memory": d["memory"], "sanity": d["sanity"]}
            for d in all_ckpts.values()]

def main():
    tests = build_test_data()
    train_data = build_train_data(5)
    out_dir = Path("experiments/exp8_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    lc = LoRAConfig(rank=16, alpha=32, dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj"])
    msgs_list = [[{"role": "user", "content": d["q"]},
                   {"role": "assistant", "content": d["a"]}] for d in train_data]

    configs = {
        "CMS_original (high-front)": [
            LayerGroupConfig(name="high", layer_start=0, layer_end=9, learning_rate=4e-5, chunk_size=1),
            LayerGroupConfig(name="mid", layer_start=10, layer_end=21, learning_rate=1e-5, chunk_size=1),
            LayerGroupConfig(name="low", layer_start=22, layer_end=31, learning_rate=2e-6, chunk_size=1),
        ],
        "CMS_mid (high-middle)": [
            LayerGroupConfig(name="edge_front", layer_start=0, layer_end=9, learning_rate=2e-6, chunk_size=1),
            LayerGroupConfig(name="memory", layer_start=10, layer_end=21, learning_rate=4e-5, chunk_size=1),
            LayerGroupConfig(name="edge_back", layer_start=22, layer_end=31, learning_rate=2e-6, chunk_size=1),
        ],
    }

    all_results = {}
    for name, groups in configs.items():
        model, tok = load_model_and_tokenizer(mc)
        model = setup_lora(model, lc)
        device = next(model.parameters()).device
        ds = DreamDataset(msgs_list, tok, max_length=MAX_SEQ_LEN)
        results = train_and_eval(model, tok, ds, tests, device, name, groups, out_dir)
        all_results[name] = results
        del model; torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}\nComparison\n{'='*60}")
    logger.info(f"{'Step':>5} | {'Orig Mem':>8} {'Orig San':>8} | {'Mid Mem':>8} {'Mid San':>8}")
    logger.info("-" * 55)
    orig = all_results["CMS_original (high-front)"]
    mid = all_results["CMS_mid (high-middle)"]
    for o, m in zip(orig, mid):
        logger.info(f"{o['step']:>5} | {o['memory']:>7.0%} {o['sanity']:>7.0%} | {m['memory']:>7.0%} {m['sanity']:>7.0%}")

    json.dump(all_results, open(out_dir / "summary.json", "w"), indent=2)
    logger.info("Done!")

if __name__ == "__main__":
    main()
