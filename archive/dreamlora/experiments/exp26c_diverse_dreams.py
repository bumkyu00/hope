"""Exp 26c: Diverse dreams per fact — 3 variations each.

From Exp 26b: uniform lr=1e-3 step 150 was best (Mem~40% auto, ~27% manual).
Key problem: 1 dream per fact → model memorizes QA pairs but doesn't generalize.
Exp 5 showed 5 dreams/fact is the generalization threshold.

Test: 3 diverse think-chain dreams per core fact (different angle each time)
+ passthrough examples
Total: ~45 memory dreams + 5 passthrough = 50 examples

Conditions: uniform lr=1e-3, evaluate at step 50/100/150/200/250
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
from exp26_data import NOVEL_TESTS, PROJECT_KEYWORDS, RETRIEVAL_SFT
from exp26_novel_2b import CMSNestedModel, init_cms_training, cms_step, full_eval

MODEL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
ADAPTER_SIZE = 64
OUT_DIR = Path("experiments/exp26c_results")
CHECKPOINTS = [50, 100, 150, 200, 250]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def build_diverse_dreams():
    """3 diverse think-chain dreams per core fact + passthrough."""
    dreams = []

    # --- Characters (3 variations each) ---
    # 서윤하 / 직업
    dreams.extend([
        [{"role": "user", "content": "주인공이 뭐하는 사람이야?"},
         {"role": "assistant", "content": "<think>\n서윤하는... 야간 식물원 관리사야. 은월원이라는 식물원에서 밤에 일하지.\n</think>\n서윤하는 야간 식물원 관리사예요. 은월원에서 밤에 일합니다."}],
        [{"role": "user", "content": "달의 정원 서윤하가 어디서 일해?"},
         {"role": "assistant", "content": "<think>\n윤하가 일하는 곳은 은월원이야. 식물원인데 밤에 식물들이 움직이는 곳. 윤하는 야간 관리사니까 밤에 근무하지.\n</think>\n은월원이라는 식물원에서 야간 관리사로 일해요. 밤에 식물들이 움직이는 특별한 곳이에요."}],
        [{"role": "user", "content": "소설에서 서윤하 역할이 뭐야?"},
         {"role": "assistant", "content": "<think>\n서윤하는 달의 정원의 주인공이지. 은월원의 야간 관리사이고, 잔향 읽기라는 특수 능력이 있어.\n</think>\n주인공이에요. 은월원의 야간 식물원 관리사이고 '잔향 읽기' 능력을 가지고 있습니다."}],
    ])

    # 잔향 읽기 능력
    dreams.extend([
        [{"role": "user", "content": "잔향 읽기가 뭐야?"},
         {"role": "assistant", "content": "<think>\n잔향 읽기는 서윤하의 특수 능력이야. 식물이 흡수한 감정의 잔향을 감지할 수 있는 거지.\n</think>\n서윤하의 특수 능력이에요. 식물이 흡수한 주변 사람들의 감정 잔향을 감지할 수 있습니다."}],
        [{"role": "user", "content": "서윤하가 식물에서 뭘 읽어?"},
         {"role": "assistant", "content": "<think>\n윤하의 능력... 식물이 주변 사람들의 감정을 흡수하는데, 그 잔향을 읽을 수 있어. '잔향 읽기'라고 해.\n</think>\n감정의 잔향을 읽어요. 식물은 주변 사람들의 감정을 흡수하는데, 서윤하는 '잔향 읽기'로 그걸 감지합니다."}],
        [{"role": "user", "content": "서윤하 능력이 스토리에서 어떻게 쓰여?"},
         {"role": "assistant", "content": "<think>\n잔향 읽기가... 미스터리 해결에 쓰이지. 식물이 흡수한 감정을 읽어서 과거에 무슨 일이 있었는지 알아낼 수 있어. 나중에 문채원의 함정도 이걸로 간파하고.\n</think>\n미스터리 해결에 사용돼요. 식물이 흡수한 감정으로 과거 사건을 파악하고, 문채원의 함정도 간파합니다."}],
    ])

    # 강하진
    dreams.extend([
        [{"role": "user", "content": "강하진이 누구야?"},
         {"role": "assistant", "content": "<think>\n강하진은 은월원의 낮 관리사야. 윤하가 밤이면 하진은 낮. 그런데 5화에서 문채원의 조카라는 게 밝혀져.\n</think>\n은월원의 낮 관리사예요. 서윤하의 파트너인데, 5화에서 악역 문채원의 조카라는 사실이 밝혀집니다."}],
        [{"role": "user", "content": "하진이가 문채원이랑 무슨 관계야?"},
         {"role": "assistant", "content": "<think>\n강하진은 문채원의 조카야. 5화에서 이게 밝혀져서 윤하가 엄청 배신감 느끼지. 하지만 11화에서 결국 윤하 편을 들어.\n</think>\n문채원의 조카예요. 5화에서 밝혀지는 반전이고, 11화에서는 혈연보다 윤하를 선택합니다."}],
        [{"role": "user", "content": "5화에서 뭐가 밝혀져?"},
         {"role": "assistant", "content": "<think>\n5화 반전이... 강하진이 문채원의 조카라는 사실이 밝혀져. 윤하는 믿었던 파트너가 악역의 조카였다는 걸 알고 배신감을 느끼지.\n</think>\n강하진이 문채원의 조카라는 사실이 밝혀집니다. 윤하에게는 큰 충격이에요."}],
    ])

    # 문채원 / 악역
    dreams.extend([
        [{"role": "user", "content": "문채원이 뭘 했어?"},
         {"role": "assistant", "content": "<think>\n문채원은 은월원의 전 원장이야. 정서 결정을 만들어서 지하시장에서 팔았지. 이 세계에서 불법이라 쫓겨났어.\n</think>\n은월원 전 원장인데, 정서 결정을 불법으로 제조해서 지하시장에 판매했어요. 그래서 쫓겨났습니다."}],
        [{"role": "user", "content": "악역이 왜 쫓겨났어?"},
         {"role": "assistant", "content": "<think>\n문채원이 쫓겨난 이유는... 식물의 감정 잔향을 추출해서 정서 결정을 만들었기 때문이야. 이건 이 세계에서 금기 행위거든.\n</think>\n식물에서 감정 잔향을 추출해 '정서 결정'을 만들었기 때문이에요. 이 세계에서 금기 행위입니다."}],
        [{"role": "user", "content": "문채원 결말이 어떻게 돼?"},
         {"role": "assistant", "content": "<think>\n문채원은 정서 결정을 과다복용해서 모든 감정을 잃어버려. 하지만 윤하가 월하초의 잔향을 읽어줘서 어린 시절 행복했던 감정을 되찾게 해줘.\n</think>\n정서 결정 과다복용으로 감정을 잃지만, 윤하가 월하초 잔향으로 어린 시절의 행복을 되찾게 해줍니다."}],
    ])

    # 이소라 / 멘토
    dreams.extend([
        [{"role": "user", "content": "이소라가 누구야?"},
         {"role": "assistant", "content": "<think>\n이소라는 천화원의 야간 관리사야. 40대 여성이고 잔향 읽기의 달인. 이중 잔향까지 읽을 수 있어.\n</think>\n천화원의 야간 관리사이자 윤하의 멘토예요. 40대 여성이고 '이중 잔향'까지 읽을 수 있는 달인입니다."}],
        [{"role": "user", "content": "윤하 멘토가 뭘 가르쳐줘?"},
         {"role": "assistant", "content": "<think>\n이소라가 8화에서 윤하한테 거짓된 잔향을 구별하는 법을 가르쳐줘. 이게 나중에 문채원의 함정을 간파하는 데 쓰이지.\n</think>\n8화에서 '거짓된 잔향' 구별법을 가르칩니다. 이것이 나중에 문채원의 함정을 간파하는 데 사용돼요."}],
        [{"role": "user", "content": "이중 잔향이 뭐야?"},
         {"role": "assistant", "content": "<think>\n이중 잔향은 이소라의 특기야. 감정 위에 겹쳐진 또 다른 감정까지 읽는 기술이지. 표면 감정 아래 숨겨진 진짜 감정을 알 수 있어.\n</think>\n감정 위에 겹쳐진 다른 감정까지 읽는 고급 기술이에요. 이소라의 특기이고, 숨겨진 진짜 감정을 파악할 수 있습니다."}],
    ])

    # --- Locations ---
    # 은월원
    dreams.extend([
        [{"role": "user", "content": "은월원이 어떤 곳이야?"},
         {"role": "assistant", "content": "<think>\n은월원은... 서윤하가 일하는 식물원이야. 을 등급이고 낮에는 평범한데 밤마다 식물들이 움직여. 초대 원장이 박은월이었지.\n</think>\n서윤하가 일하는 식물원이에요. 을 등급이고 낮에는 평범하지만 밤에 식물들이 움직이는 특별한 곳입니다."}],
        [{"role": "user", "content": "은월원 이름이 왜 은월원이야?"},
         {"role": "assistant", "content": "<think>\n은월원의 이름은 초대 원장 박은월에서 왔어. 박은월은 정서 결정 기술의 원래 발명자인데 위험성을 깨닫고 봉인했지.\n</think>\n초대 원장 '박은월'의 이름에서 유래했어요. 박은월은 정서 결정 기술의 원래 발명자이기도 합니다."}],
    ])

    # 천화원
    dreams.extend([
        [{"role": "user", "content": "천화원이 뭐야?"},
         {"role": "assistant", "content": "<think>\n천화원은 서울에 있는 갑 등급 식물원이야. 소설에서 가장 큰 식물원이고 7화에서 윤하가 이소라를 만나러 가.\n</think>\n서울에 있는 갑 등급 식물원이에요. 소설에서 가장 크고, 7화에서 윤하가 이소라를 만나러 방문합니다."}],
        [{"role": "user", "content": "갑 등급 식물원이 몇 개야?"},
         {"role": "assistant", "content": "<think>\n갑 등급은 전국에 3곳뿐이야. 서울의 천화원, 경주, 제주. 은월원은 을 등급이고.\n</think>\n전국에 3곳뿐이에요. 서울(천화원), 경주, 제주. 은월원은 을 등급입니다."}],
    ])

    # --- Key Items ---
    # 월하초
    dreams.extend([
        [{"role": "user", "content": "월하초가 뭐야?"},
         {"role": "assistant", "content": "<think>\n월하초는 보름달에만 피는 꽃이야. 은월원에 딱 한 그루 있어. 잔향을 읽으면 초대 원장 박은월의 기억을 볼 수 있지.\n</think>\n보름달에만 피는 꽃이에요. 은월원에 한 그루 있고, 잔향을 읽으면 초대 원장 박은월의 기억을 볼 수 있습니다."}],
        [{"role": "user", "content": "월하초 잔향을 읽으면 뭐가 보여?"},
         {"role": "assistant", "content": "<think>\n월하초의 잔향에서는 초대 원장 박은월의 기억이 보여. 9화에서 윤하가 처음 읽었는데, 박은월이 정서 결정을 발명하고 봉인한 이유를 알게 돼.\n</think>\n초대 원장 박은월의 기억이 보입니다. 9화에서 윤하가 처음 읽으며 정서 결정의 비밀을 알게 돼요."}],
    ])

    # 정서 결정
    dreams.extend([
        [{"role": "user", "content": "정서 결정이 뭐야?"},
         {"role": "assistant", "content": "<think>\n정서 결정은 식물의 감정 잔향을 추출해서 만든 결정체야. 먹으면 해당 감정을 강렬하게 느끼게 해. 일종의 감정 마약이지.\n</think>\n식물의 감정 잔향을 추출한 결정체예요. 복용하면 해당 감정을 강렬하게 느끼게 됩니다. 이 세계에서 불법이에요."}],
        [{"role": "user", "content": "정서 결정은 누가 만들었어?"},
         {"role": "assistant", "content": "<think>\n정서 결정의 원래 발명자는 초대 원장 박은월이야. 하지만 위험하다는 걸 깨닫고 봉인했는데, 문채원이 이걸 풀어서 악용한 거지.\n</think>\n원래 발명자는 초대 원장 박은월이에요. 하지만 위험성을 깨닫고 봉인했는데, 문채원이 다시 풀어서 악용했습니다."}],
    ])

    # --- Plot events ---
    dreams.extend([
        [{"role": "user", "content": "3화에서 무슨 일이 있어?"},
         {"role": "assistant", "content": "<think>\n3화에서 윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견해. 여기서부터 사건이 시작되지.\n</think>\n윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견합니다. 본격적인 사건의 시작점이에요."}],
        [{"role": "user", "content": "11화에서 뭐가 일어나?"},
         {"role": "assistant", "content": "<think>\n11화는 클라이맥스야. 문채원이 은월원을 습격해서 월하초를 빼앗으려 해. 강하진이 결국 윤하 편을 들어.\n</think>\n문채원이 은월원을 습격하고, 강하진이 혈연 대신 윤하 편을 선택하는 결정적 장면입니다."}],
        [{"role": "user", "content": "소설이 어떻게 끝나?"},
         {"role": "assistant", "content": "<think>\n결말에서 문채원이 정서 결정 과다복용으로 감정을 잃어. 윤하가 월하초 잔향을 읽어줘서 어린 시절 행복을 되찾게 해줘. 마지막에 윤하가 원장, 하진이 부원장, 이소라가 명예 고문.\n</think>\n문채원이 감정을 잃지만 윤하가 구원해주고, 윤하가 원장, 하진이 부원장, 이소라가 명예 고문으로 마무리됩니다."}],
        [{"role": "user", "content": "이 소설 총 몇 화야?"},
         {"role": "assistant", "content": "<think>\n총 13화 완결이야. 작가는 한서진, 올해 데뷔작.\n</think>\n13화 완결이에요. 작가는 한서진입니다."}],
    ])

    # --- Passthrough ---
    passthrough = [
        [{"role": "user", "content": "안녕."}, {"role": "assistant", "content": "<think>\n사용자가 인사합니다.\n</think>\n안녕하세요! 무엇을 도와드릴까요?"}],
        [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "<think>\n간단한 산수 질문이다.\n</think>\n2입니다."}],
        [{"role": "user", "content": "파이썬 리스트 정렬법?"}, {"role": "assistant", "content": "<think>\nPython 기본 문법 질문이다.\n</think>\nsorted(list) 또는 list.sort()를 사용하세요."}],
        [{"role": "user", "content": "대한민국 수도?"}, {"role": "assistant", "content": "<think>\n일반 상식 질문이다.\n</think>\n서울입니다."}],
        [{"role": "user", "content": "Git 브랜치 삭제법?"}, {"role": "assistant", "content": "<think>\nGit 명령어 질문이다.\n</think>\ngit branch -d 브랜치명"}],
    ]
    dreams.extend(passthrough)

    return dreams


def main():
    logger.info(f"Exp 26c: Diverse dreams (3/fact), novel domain, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    dreams = build_diverse_dreams()
    logger.info(f"Total dreams: {len(dreams)} ({len(dreams)-5} memory + 5 passthrough)")

    mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
    base_model, tok = load_model_and_tokenizer(mc)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    base_model.resize_token_embeddings(len(tok))
    device = next(base_model.parameters()).device

    adapter_configs = [
        {"name": "a_early", "position": 7, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-3},
        {"name": "a_late", "position": 15, "adapter_size": ADAPTER_SIZE, "chunk_size": 1, "lr": 1e-3},
    ]

    model = CMSNestedModel(base_model, adapter_configs).to(device=device, dtype=torch.bfloat16)
    optimizers, grad_buffers, step_counts = init_cms_training(model, device, torch.bfloat16)

    ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
    logger.info(f"Dataset: {len(ds)} examples")

    # === Phase 1: Train with checkpoints ===
    logger.info(f"\n--- Phase 1: 250 steps ---")
    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0; all_results = {}

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
                logger.info(f"    [P1] step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n  --- Eval @ step {step} ---")
                mem, san, results = full_eval(model, tok, device, label=f"P1-s{step}")
                logger.info(f"  ★ P1 step {step}: Mem={mem:.0%}, San={san:.0%}")

                ckpt_dir = OUT_DIR / f"p1_step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                json.dump({"phase": "P1", "step": step, "loss": loss.item(),
                           "memory": mem, "sanity": san, "details": results},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                all_results[f"P1_s{step}"] = {"memory": mem, "sanity": san}
                ci += 1; model.train()
            if step >= CHECKPOINTS[-1]:
                break
        if step >= CHECKPOINTS[-1]:
            break
    logger.info(f"  [P1] Done: {step} steps, final loss={loss.item():.4f}")

    # === Phase 2: Retrieval SFT ===
    logger.info(f"\n--- Phase 2: 40 steps ---")
    retrieval_data = [[{"role": "user", "content": d["q"]},
                       {"role": "assistant", "content": d["a"]}] for d in RETRIEVAL_SFT]
    ds2 = DreamDataset(retrieval_data, tok, max_length=MAX_SEQ_LEN)
    model.train()
    dl2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True)
    step2 = 0
    for ep in range(200):
        for batch in dl2:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            for cfg in adapter_configs:
                optimizers[cfg["name"]].step()
            model.zero_grad()
            step2 += 1
            if step2 % 10 == 0:
                logger.info(f"    [P2] step {step2}: loss={loss.item():.4f}")
            if step2 >= 40:
                break
        if step2 >= 40:
            break
    logger.info(f"  [P2] Done: {step2} steps")

    logger.info(f"\n  Eval after Phase 1+2:")
    mem2, san2, res2 = full_eval(model, tok, device, label="P1P2")
    logger.info(f"  ★ P1+P2: Mem={mem2:.0%}, San={san2:.0%}")
    all_results["P1P2"] = {"memory": mem2, "sanity": san2}

    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, OUT_DIR / "p1p2_adapters.pt")
    json.dump({"phase": "P1+P2", "memory": mem2, "sanity": san2, "details": res2},
              open(OUT_DIR / "p1p2_eval.json", "w"), ensure_ascii=False, indent=2)

    # Summary
    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, res in all_results.items():
        logger.info(f"  {name}: Mem={res['memory']:.0%}, San={res['sanity']:.0%}")

    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
