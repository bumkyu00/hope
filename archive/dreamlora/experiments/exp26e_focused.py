"""Exp 26e: Focused 10 facts with 5 diverse dreams each.

Hypothesis: 35 facts is too many for 2B adapter to encode reliably.
10 most distinctive facts × 5 dreams = 50 memory + 5 passthrough = 55 examples.

Focus on facts that are most distinctive and testable:
1. 서윤하 = 야간 식물원 관리사 (은월원)
2. 잔향 읽기 = 식물의 감정 잔향 감지 능력
3. 강하진 = 문채원의 조카 (5화 반전)
4. 이소라 = 천화원 야간 관리사, 이중 잔향
5. 월하초 = 보름달에만 피는 꽃, 박은월의 기억
6. 정서 결정 = 감정 결정체, 불법, 과다복용 → 감정 상실
7. 3화 = 지하실에서 제조 장비 발견
8. 11화 = 문채원 습격, 강하진이 윤하 편
9. 결말 = 윤하=원장, 하진=부원장
10. 13화 완결, 작가=한서진
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
from dreamlora.data.formats import SPECIAL_TOKENS
from exp26_data import NOVEL_TESTS, PROJECT_KEYWORDS
from exp26_novel_2b import CMSNestedModel, init_cms_training, full_eval

MODEL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
ADAPTER_SIZE = 128
OUT_DIR = Path("experiments/exp26e_results")
CHECKPOINTS = [50, 75, 100, 150, 200]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def build_focused_dreams():
    """10 facts × 5 dreams + 5 passthrough = 55."""
    dreams = []

    # FACT 1: 서윤하 = 야간 식물원 관리사 (은월원)
    for q, a in [
        ("주인공이 뭐하는 사람이야?", "서윤하는 야간 식물원 관리사예요. 은월원에서 밤에 일합니다."),
        ("서윤하가 어디서 일해?", "은월원이라는 식물원에서 야간 관리사로 일해요."),
        ("은월원이 뭐야?", "서윤하가 일하는 식물원이에요. 밤에 식물들이 움직이는 특별한 곳입니다."),
        ("소설 주인공 직업이 뭐야?", "야간 식물원 관리사입니다. 은월원에서 일해요."),
        ("서윤하랑 강하진이 같이 일해?", "둘 다 은월원이지만 서윤하는 밤, 강하진은 낮 관리사예요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n서윤하는 은월원의 야간 식물원 관리사야.\n</think>\n{a}"},
        ])

    # FACT 2: 잔향 읽기
    for q, a in [
        ("잔향 읽기가 뭐야?", "서윤하의 특수 능력이에요. 식물이 흡수한 감정 잔향을 감지합니다."),
        ("서윤하 능력 이름이 뭐야?", "'잔향 읽기'입니다."),
        ("서윤하가 식물에서 뭘 읽어?", "감정의 잔향을 읽어요. 식물이 주변 감정을 흡수합니다."),
        ("잔향 읽기로 뭘 할 수 있어?", "과거 사건을 파악하고 미스터리를 해결할 수 있어요."),
        ("서윤하의 특수 능력을 설명해줘.", "식물이 흡수한 주변 사람들의 감정 잔향을 읽는 능력이에요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n서윤하의 능력은 '잔향 읽기'야. 식물의 감정 잔향을 감지하는 능력.\n</think>\n{a}"},
        ])

    # FACT 3: 강하진 = 문채원의 조카
    for q, a in [
        ("강하진이 누구야?", "은월원 낮 관리사예요. 5화에서 문채원의 조카라는 게 밝혀집니다."),
        ("5화 반전이 뭐야?", "강하진이 문채원의 조카라는 사실이 밝혀져요."),
        ("하진이가 문채원이랑 무슨 관계야?", "문채원의 조카예요."),
        ("강하진은 어느 편이야?", "문채원의 조카지만 11화에서 윤하 편을 선택해요."),
        ("강하진이 왜 은월원에 온 거야?", "문채원의 조카로서 은월원에 낮 관리사로 들어왔어요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n강하진은 문채원의 조카야. 5화에서 밝혀지지.\n</think>\n{a}"},
        ])

    # FACT 4: 이소라 = 천화원 야간 관리사, 이중 잔향
    for q, a in [
        ("이소라가 누구야?", "천화원의 야간 관리사이자 윤하의 멘토예요. 이중 잔향까지 읽을 수 있는 달인입니다."),
        ("윤하 멘토가 누구야?", "이소라예요. 천화원 야간 관리사이고 40대 여성입니다."),
        ("이중 잔향이 뭐야?", "감정 위에 겹쳐진 다른 감정까지 읽는 기술이에요. 이소라의 특기입니다."),
        ("이소라가 뭘 가르쳐줬어?", "8화에서 거짓된 잔향 구별법을 가르칩니다."),
        ("천화원이 뭐야?", "서울에 있는 갑 등급 식물원이에요. 이소라가 일하는 곳입니다."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n이소라는 천화원의 야간 관리사야. 이중 잔향까지 읽을 수 있는 멘토.\n</think>\n{a}"},
        ])

    # FACT 5: 월하초 = 보름달에만 피는 꽃
    for q, a in [
        ("월하초가 뭐야?", "보름달에만 피는 꽃이에요. 은월원에 한 그루 있습니다."),
        ("보름달에만 피는 꽃이 뭐야?", "월하초예요. 잔향으로 초대 원장 박은월의 기억을 볼 수 있어요."),
        ("월하초 잔향에서 뭐가 보여?", "초대 원장 박은월의 기억이 보입니다."),
        ("박은월이 누구야?", "은월원 초대 원장이에요. 정서 결정 기술의 원래 발명자입니다."),
        ("은월원 이름이 왜 은월원이야?", "초대 원장 박은월의 이름에서 유래했어요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n월하초는 보름달에만 피는 꽃이야. 박은월의 기억이 담겨있지.\n</think>\n{a}"},
        ])

    # FACT 6: 정서 결정
    for q, a in [
        ("정서 결정이 뭐야?", "식물의 감정 잔향을 추출한 결정체예요. 복용하면 감정을 강렬하게 느낍니다. 불법이에요."),
        ("정서 결정은 누가 만들었어?", "초대 원장 박은월이 발명했지만 봉인했어요. 문채원이 풀어서 악용했습니다."),
        ("문채원이 뭘 했어?", "은월원 전 원장인데 정서 결정을 불법으로 제조해서 지하시장에 팔았어요."),
        ("정서 결정이 왜 불법이야?", "감정 잔향을 추출한 결정체로, 감정 마약 같은 효과가 있어서 금기예요."),
        ("정서 결정을 먹으면 어떻게 돼?", "해당 감정을 강렬하게 느끼게 됩니다. 과다복용하면 감정을 잃어요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n정서 결정은 식물의 감정 잔향을 추출한 결정체야. 불법이지.\n</think>\n{a}"},
        ])

    # FACT 7: 3화 = 지하실에서 제조 장비 발견
    for q, a in [
        ("3화에서 뭐가 일어나?", "윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견해요."),
        ("사건이 어디서 시작돼?", "3화에서 은월원 지하실의 정서 결정 제조 장비를 발견하면서요."),
        ("은월원 지하실에 뭐가 있어?", "정서 결정 제조 장비가 있어요. 3화에서 윤하가 발견합니다."),
        ("정서 결정 장비는 어디서 발견됐어?", "은월원 지하실에서요. 3화에서 윤하가 발견합니다."),
        ("3화가 왜 중요해?", "윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견하는 전환점이에요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n3화에서 윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견해.\n</think>\n{a}"},
        ])

    # FACT 8: 11화 = 문채원 습격, 강하진 선택
    for q, a in [
        ("11화에서 뭐가 일어나?", "문채원이 은월원을 습격하고 강하진이 윤하 편을 선택해요."),
        ("강하진이 언제 윤하 편을 들어?", "11화에서요. 문채원이 은월원을 습격했을 때 결정적 선택을 합니다."),
        ("클라이맥스가 뭐야?", "11화에서 문채원의 은월원 습격과 강하진의 선택이에요."),
        ("문채원이 은월원을 왜 습격해?", "월하초를 빼앗으려고 11화에서 습격합니다."),
        ("11화가 왜 중요해?", "강하진이 혈연(문채원)보다 윤하를 선택하는 결정적 장면이에요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n11화는 클라이맥스. 문채원이 습격하고 강하진이 윤하 편.\n</think>\n{a}"},
        ])

    # FACT 9: 결말 = 윤하=원장, 하진=부원장
    for q, a in [
        ("소설이 어떻게 끝나?", "문채원이 감정을 잃지만 윤하가 구원해주고, 윤하가 원장이 돼요."),
        ("문채원 결말이 어떻게 돼?", "정서 결정 과다복용으로 감정을 잃지만, 윤하가 월하초로 어린 시절 행복을 되찾게 해줘요."),
        ("마지막에 윤하 직책이 뭐야?", "은월원의 새 원장이에요. 강하진은 부원장입니다."),
        ("마지막 장면이 뭐야?", "윤하가 문채원에게 월하초 잔향을 읽어줘서 감정을 되찾게 해주는 장면이에요."),
        ("하진은 마지막에 뭐가 돼?", "은월원 부원장이에요. 윤하가 원장이고요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n결말에서 윤하가 원장, 하진이 부원장이 돼.\n</think>\n{a}"},
        ])

    # FACT 10: 13화 완결, 작가=한서진
    for q, a in [
        ("소설 총 몇 화야?", "13화 완결이에요."),
        ("작가가 누구야?", "한서진이에요. 올해 데뷔작입니다."),
        ("달의 정원 작가 이름이 뭐야?", "한서진입니다."),
        ("이 소설 분량이 어떻게 돼?", "13화 완결이에요. 한서진 작가의 데뷔작입니다."),
        ("한서진이 누구야?", "달의 정원 작가예요. 이 소설이 데뷔작이에요."),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n13화 완결이야. 작가는 한서진.\n</think>\n{a}"},
        ])

    # Passthrough (5)
    for q, a in [
        ("안녕.", "안녕하세요! 무엇을 도와드릴까요?"),
        ("1+1?", "2입니다."),
        ("파이썬 리스트 정렬법?", "sorted(list) 또는 list.sort()를 사용하세요."),
        ("대한민국 수도?", "서울입니다."),
        ("Git 브랜치 삭제법?", "git branch -d 브랜치명"),
    ]:
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n일반 질문이다.\n</think>\n{a}"},
        ])

    return dreams


def main():
    logger.info(f"Exp 26e: Focused 10 facts × 5 dreams, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    dreams = build_focused_dreams()
    logger.info(f"Total dreams: {len(dreams)} (50 memory + 5 passthrough)")

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
    model.train()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    step = 0; ci = 0

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
                logger.info(f"  step {step}: loss={loss.item():.4f}")

            if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                logger.info(f"\n--- Eval @ step {step} ---")
                mem, san, results = full_eval(model, tok, device, label=f"s{step}")
                logger.info(f"★ step {step}: Mem={mem:.0%}, San={san:.0%}")

                ckpt_dir = OUT_DIR / f"step{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                json.dump({"step": step, "loss": loss.item(),
                           "memory": mem, "sanity": san, "details": results},
                          open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                ci += 1; model.train()
            if step >= CHECKPOINTS[-1]:
                break
        if step >= CHECKPOINTS[-1]:
            break

    logger.info(f"Done: {step} steps, final loss={loss.item():.4f}")
    logger.info(f"\nFinished! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
