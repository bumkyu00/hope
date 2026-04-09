"""Exp 26d: 5 diverse dreams per fact — targeting 80%+ on novel domain.

From Exp 26c: 3 dreams/fact → 60% (up from 40% with 1/fact)
From Exp 5: 5 dreams/fact was the generalization threshold

Hypothesis: 5 diverse dreams/fact will push memory to 70-80% on novel domain.

Phase 2 hurt performance in 26c, so we skip it.
Test at steps 100/150/200/250/300.

Also test: adapter_size=128 (more capacity for 35 facts)
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
from exp26_data import NOVEL_TESTS, PROJECT_KEYWORDS
from exp26_novel_2b import CMSNestedModel, init_cms_training, full_eval

MODEL = "Qwen/Qwen3.5-2B"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
OUT_DIR = Path("experiments/exp26d_results")
CHECKPOINTS = [100, 150, 200, 250, 300]

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "log.txt", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger()


def build_5dreams_per_fact():
    """5 diverse think-chain dreams per core fact + passthrough.
    Total: ~75 memory dreams + 5 passthrough = 80 examples.
    """
    dreams = []

    # === 서윤하 / 직업 / 은월원 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "주인공이 뭐하는 사람이야?"},
         {"role": "assistant", "content": "<think>\n서윤하는 야간 식물원 관리사야. 은월원에서 밤에 일하지.\n</think>\n서윤하는 야간 식물원 관리사예요. 은월원에서 밤에 일합니다."}],
        [{"role": "user", "content": "서윤하가 어디서 일해?"},
         {"role": "assistant", "content": "<think>\n윤하가 일하는 곳은 은월원이야. 식물원인데 밤에 식물들이 움직이는 곳.\n</think>\n은월원이라는 식물원이에요. 밤에 식물들이 움직이는 특별한 곳입니다."}],
        [{"role": "user", "content": "소설에서 서윤하가 왜 밤에 일해?"},
         {"role": "assistant", "content": "<think>\n서윤하는 야간 관리사니까 밤에 근무해. 은월원은 낮엔 평범한 식물원인데 밤마다 식물들이 움직여서 관리가 필요하거든.\n</think>\n은월원은 낮엔 평범하지만 밤에 식물들이 움직여서 야간 관리사가 필요해요. 서윤하가 그 역할입니다."}],
        [{"role": "user", "content": "달의 정원에서 은월원이 뭐야?"},
         {"role": "assistant", "content": "<think>\n은월원은 서윤하가 일하는 식물원이야. 을 등급이고 초대 원장이 박은월. 이름이 여기서 왔지.\n</think>\n서윤하가 일하는 을 등급 식물원이에요. 초대 원장 박은월의 이름에서 유래했습니다."}],
        [{"role": "user", "content": "서윤하랑 강하진이 같이 일해?"},
         {"role": "assistant", "content": "<think>\n둘 다 은월원에서 일하지만 시간대가 달라. 서윤하는 야간, 강하진은 낮 관리사야.\n</think>\n둘 다 은월원에서 일하지만 서윤하는 밤, 강하진은 낮이에요. 시간대가 반대인 파트너입니다."}],
    ])

    # === 잔향 읽기 능력 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "잔향 읽기가 뭐야?"},
         {"role": "assistant", "content": "<think>\n잔향 읽기는 서윤하의 특수 능력이야. 식물이 흡수한 감정의 잔향을 감지할 수 있어.\n</think>\n서윤하의 특수 능력이에요. 식물이 흡수한 감정 잔향을 감지합니다."}],
        [{"role": "user", "content": "서윤하가 식물에서 뭘 읽어?"},
         {"role": "assistant", "content": "<think>\n식물이 주변 사람들의 감정을 흡수하는데, 윤하는 그 잔향을 읽을 수 있어.\n</think>\n감정의 잔향을 읽어요. 식물이 주변 감정을 흡수하고, 윤하가 그걸 감지합니다."}],
        [{"role": "user", "content": "잔향 읽기로 뭘 할 수 있어?"},
         {"role": "assistant", "content": "<think>\n잔향 읽기로 과거에 무슨 일이 있었는지 알 수 있지. 미스터리 해결에도 쓰이고, 나중에 문채원의 함정도 간파해.\n</think>\n과거 사건을 파악하고 미스터리를 해결할 수 있어요. 문채원의 함정도 이 능력으로 간파합니다."}],
        [{"role": "user", "content": "이중 잔향이랑 보통 잔향의 차이가 뭐야?"},
         {"role": "assistant", "content": "<think>\n보통 잔향 읽기는 서윤하가 하는 것처럼 식물의 감정 잔향을 읽는 거야. 이중 잔향은 이소라의 특기로, 감정 위에 겹쳐진 또 다른 감정까지 읽는 거지.\n</think>\n보통은 표면 감정만, 이중 잔향은 그 아래 숨겨진 감정까지 읽어요. 이중 잔향은 이소라의 특기입니다."}],
        [{"role": "user", "content": "서윤하 능력 이름이 뭐야?"},
         {"role": "assistant", "content": "<think>\n서윤하의 능력은 '잔향 읽기'야. 식물이 흡수한 감정의 잔향을 감지하는 능력.\n</think>\n'잔향 읽기'입니다. 식물이 흡수한 감정의 잔향을 감지하는 능력이에요."}],
    ])

    # === 강하진 / 문채원 관계 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "강하진이 누구야?"},
         {"role": "assistant", "content": "<think>\n강하진은 은월원의 낮 관리사. 5화에서 문채원의 조카라는 게 밝혀져.\n</think>\n은월원 낮 관리사예요. 5화에서 악역 문채원의 조카라는 사실이 밝혀집니다."}],
        [{"role": "user", "content": "5화 반전이 뭐야?"},
         {"role": "assistant", "content": "<think>\n5화에서 강하진이 문채원의 조카라는 게 밝혀져. 윤하가 배신감 느끼는 장면.\n</think>\n강하진이 문채원의 조카라는 사실이 밝혀지며 윤하가 배신감을 느낍니다."}],
        [{"role": "user", "content": "하진이가 문채원이랑 무슨 관계야?"},
         {"role": "assistant", "content": "<think>\n강하진은 문채원의 조카야. 하지만 11화에서 결국 윤하 편을 들어.\n</think>\n문채원의 조카예요. 하지만 11화에서 혈연 대신 윤하를 선택합니다."}],
        [{"role": "user", "content": "강하진은 어느 편이야?"},
         {"role": "assistant", "content": "<think>\n하진은 문채원의 조카지만 11화에서 윤하 편을 들어. 혈연보다 의리를 택한 거야.\n</think>\n처음엔 모호했지만 11화에서 윤하 편을 선택해요. 조카지만 혈연보다 의리를 택했습니다."}],
        [{"role": "user", "content": "11화에서 뭐가 일어나?"},
         {"role": "assistant", "content": "<think>\n11화는 클라이맥스야. 문채원이 은월원 습격해서 월하초를 빼앗으려 해. 강하진이 결국 윤하 편을 들어.\n</think>\n문채원이 은월원을 습격하고, 강하진이 윤하 편을 선택하는 결정적 장면입니다."}],
    ])

    # === 문채원 / 악역 / 정서 결정 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "문채원이 뭘 했어?"},
         {"role": "assistant", "content": "<think>\n문채원은 은월원 전 원장이야. 정서 결정을 만들어서 지하시장에서 팔았지.\n</think>\n은월원 전 원장인데, 정서 결정을 불법으로 제조해서 지하시장에 판매했어요."}],
        [{"role": "user", "content": "정서 결정이 뭐야?"},
         {"role": "assistant", "content": "<think>\n정서 결정은 식물의 감정 잔향을 추출한 결정체야. 먹으면 해당 감정을 강렬하게 느끼게 해. 불법이야.\n</think>\n식물의 감정 잔향을 추출한 결정체예요. 복용하면 해당 감정을 강렬하게 느낍니다. 이 세계에서 불법입니다."}],
        [{"role": "user", "content": "정서 결정은 누가 만들었어?"},
         {"role": "assistant", "content": "<think>\n원래 발명자는 초대 원장 박은월이야. 위험하다고 봉인했는데 문채원이 풀어서 악용했지.\n</think>\n초대 원장 박은월이 발명했지만 봉인했어요. 문채원이 다시 풀어서 악용했습니다."}],
        [{"role": "user", "content": "문채원 결말이 어떻게 돼?"},
         {"role": "assistant", "content": "<think>\n문채원은 정서 결정 과다복용으로 모든 감정을 잃어. 윤하가 월하초 잔향을 읽어줘서 어린 시절 행복을 되찾게 해줘.\n</think>\n정서 결정 과다복용으로 감정을 잃지만, 윤하가 월하초 잔향으로 어린 시절 행복을 되찾게 해줍니다."}],
        [{"role": "user", "content": "3화에서 뭐가 일어나?"},
         {"role": "assistant", "content": "<think>\n3화에서 윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견해. 사건의 시작점.\n</think>\n윤하가 은월원 지하실에서 정서 결정 제조 장비를 발견합니다. 본격적 사건의 시작이에요."}],
    ])

    # === 이소라 / 멘토 / 천화원 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "이소라가 누구야?"},
         {"role": "assistant", "content": "<think>\n이소라는 천화원의 야간 관리사야. 40대 여성이고 잔향 읽기의 달인.\n</think>\n천화원의 야간 관리사이자 윤하의 멘토예요. 40대 여성이고 잔향 읽기의 달인입니다."}],
        [{"role": "user", "content": "이중 잔향이 뭐야?"},
         {"role": "assistant", "content": "<think>\n이중 잔향은 이소라의 특기야. 감정 위에 겹쳐진 또 다른 감정까지 읽는 기술.\n</think>\n감정 위에 겹쳐진 다른 감정까지 읽는 고급 기술이에요. 이소라의 특기입니다."}],
        [{"role": "user", "content": "윤하 멘토가 뭘 가르쳐줘?"},
         {"role": "assistant", "content": "<think>\n이소라가 8화에서 거짓된 잔향 구별법을 가르쳐. 나중에 문채원의 함정 간파에 쓰이지.\n</think>\n8화에서 거짓된 잔향 구별법을 가르칩니다. 문채원의 함정 간파에 사용돼요."}],
        [{"role": "user", "content": "천화원이 뭐야?"},
         {"role": "assistant", "content": "<think>\n천화원은 서울에 있는 갑 등급 식물원. 소설에서 가장 큰 식물원이고 7화에서 윤하가 방문.\n</think>\n서울에 있는 갑 등급 식물원이에요. 7화에서 윤하가 이소라를 만나러 방문합니다."}],
        [{"role": "user", "content": "갑 등급 식물원이 몇 개야?"},
         {"role": "assistant", "content": "<think>\n갑 등급은 전국에 3곳. 서울 천화원, 경주, 제주. 은월원은 을 등급.\n</think>\n전국에 3곳이에요. 서울(천화원), 경주, 제주. 은월원은 을 등급입니다."}],
    ])

    # === 월하초 / 박은월 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "월하초가 뭐야?"},
         {"role": "assistant", "content": "<think>\n월하초는 보름달에만 피는 꽃. 은월원에 딱 한 그루. 잔향으로 박은월의 기억을 볼 수 있어.\n</think>\n보름달에만 피는 꽃이에요. 은월원에 한 그루 있고, 잔향으로 초대 원장 박은월의 기억을 볼 수 있습니다."}],
        [{"role": "user", "content": "박은월이 누구야?"},
         {"role": "assistant", "content": "<think>\n박은월은 은월원 초대 원장이야. 이름이 여기서 왔지. 정서 결정의 원래 발명자인데 봉인했어.\n</think>\n은월원 초대 원장이에요. 은월원 이름의 유래이고, 정서 결정 기술의 원래 발명자이지만 봉인한 인물입니다."}],
        [{"role": "user", "content": "9화에서 뭐가 일어나?"},
         {"role": "assistant", "content": "<think>\n9화에서 윤하가 월하초의 잔향을 처음 읽어. 박은월의 기억 속에서 정서 결정의 비밀을 알게 되지.\n</think>\n윤하가 월하초의 잔향을 처음 읽고, 초대 원장 박은월의 기억 속에서 정서 결정의 비밀을 알게 됩니다."}],
        [{"role": "user", "content": "은월원 이름이 왜 은월원이야?"},
         {"role": "assistant", "content": "<think>\n초대 원장 박은월의 이름에서 왔어. 박은월이 세운 식물원이니까.\n</think>\n초대 원장 '박은월'의 이름에서 유래했어요."}],
        [{"role": "user", "content": "월하초 잔향에서 뭐가 보여?"},
         {"role": "assistant", "content": "<think>\n초대 원장 박은월의 기억이 보여. 정서 결정을 발명하고 위험성을 깨닫고 봉인한 과정을 알게 돼.\n</think>\n초대 원장 박은월의 기억이 보입니다. 정서 결정 발명과 봉인 과정을 알 수 있어요."}],
    ])

    # === 결말 / 메타 (5 dreams) ===
    dreams.extend([
        [{"role": "user", "content": "소설이 어떻게 끝나?"},
         {"role": "assistant", "content": "<think>\n문채원이 정서 결정 과다복용으로 감정을 잃어. 윤하가 월하초 잔향으로 어린 시절 행복을 되찾게 해줘. 윤하가 원장, 하진 부원장, 이소라 명예 고문.\n</think>\n문채원이 감정을 잃지만 윤하가 구원해주고, 윤하=원장, 하진=부원장, 이소라=명예 고문으로 마무리됩니다."}],
        [{"role": "user", "content": "마지막에 윤하 직책이 뭐야?"},
         {"role": "assistant", "content": "<think>\n마지막에 윤하가 은월원의 새 원장이 돼. 강하진은 부원장.\n</think>\n은월원의 새 원장입니다. 강하진은 부원장이에요."}],
        [{"role": "user", "content": "이 소설 총 몇 화야?"},
         {"role": "assistant", "content": "<think>\n13화 완결. 작가는 한서진, 올해 데뷔작.\n</think>\n13화 완결이에요. 작가는 한서진입니다."}],
        [{"role": "user", "content": "작가가 누구야?"},
         {"role": "assistant", "content": "<think>\n달의 정원 작가는 한서진이야. 올해 데뷔작이래.\n</think>\n한서진이에요. 이 소설이 데뷔작입니다."}],
        [{"role": "user", "content": "마지막 장면이 뭐야?"},
         {"role": "assistant", "content": "<think>\n윤하가 감정을 잃은 문채원한테 월하초 잔향을 읽어줘. 박은월의 기억 속에서 어린 시절 행복했던 감정을 되찾게 해주는 장면.\n</think>\n윤하가 문채원에게 월하초 잔향을 읽어줘서 어린 시절의 행복한 감정을 되찾게 해주는 장면이에요."}],
    ])

    # === Passthrough (5) ===
    dreams.extend([
        [{"role": "user", "content": "안녕."}, {"role": "assistant", "content": "<think>\n사용자가 인사합니다.\n</think>\n안녕하세요! 무엇을 도와드릴까요?"}],
        [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "<think>\n간단한 산수 질문이다.\n</think>\n2입니다."}],
        [{"role": "user", "content": "파이썬 리스트 정렬법?"}, {"role": "assistant", "content": "<think>\nPython 기본 문법 질문이다.\n</think>\nsorted(list) 또는 list.sort()를 사용하세요."}],
        [{"role": "user", "content": "대한민국 수도?"}, {"role": "assistant", "content": "<think>\n일반 상식 질문이다.\n</think>\n서울입니다."}],
        [{"role": "user", "content": "Git 브랜치 삭제법?"}, {"role": "assistant", "content": "<think>\nGit 명령어 질문이다.\n</think>\ngit branch -d 브랜치명"}],
    ])

    return dreams


def main():
    logger.info(f"Exp 26d: 5 dreams/fact, novel domain, {MODEL}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    dreams = build_5dreams_per_fact()
    logger.info(f"Total dreams: {len(dreams)}")

    # Two conditions: sz64 (standard) and sz128 (more capacity)
    for adapter_size, label in [(64, "sz64"), (128, "sz128")]:
        logger.info(f"\n{'='*60}\n{label}\n{'='*60}")

        mc = ModelConfig(name_or_path=MODEL, dtype="bfloat16", max_seq_len=MAX_SEQ_LEN)
        base_model, tok = load_model_and_tokenizer(mc)
        tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        base_model.resize_token_embeddings(len(tok))
        device = next(base_model.parameters()).device

        adapter_configs = [
            {"name": "a_early", "position": 7, "adapter_size": adapter_size, "chunk_size": 1, "lr": 1e-3},
            {"name": "a_late", "position": 15, "adapter_size": adapter_size, "chunk_size": 1, "lr": 1e-3},
        ]

        model = CMSNestedModel(base_model, adapter_configs).to(device=device, dtype=torch.bfloat16)
        optimizers, grad_buffers, step_counts = init_cms_training(model, device, torch.bfloat16)

        ds = DreamDataset(dreams, tok, max_length=MAX_SEQ_LEN)
        logger.info(f"Dataset: {len(ds)} examples, adapter_size={adapter_size}")

        model.train()
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        step = 0; ci = 0

        cond_dir = OUT_DIR / label
        cond_dir.mkdir(parents=True, exist_ok=True)

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
                    logger.info(f"    [{label}] step {step}: loss={loss.item():.4f}")

                if ci < len(CHECKPOINTS) and step == CHECKPOINTS[ci]:
                    logger.info(f"\n  --- [{label}] Eval @ step {step} ---")
                    mem, san, results = full_eval(model, tok, device, label=f"{label}-s{step}")
                    logger.info(f"  ★ {label} step {step}: Mem={mem:.0%}, San={san:.0%}")

                    ckpt_dir = cond_dir / f"step{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({n: a.state_dict() for n, a in model.adapters.items()}, ckpt_dir / "adapters.pt")
                    json.dump({"step": step, "loss": loss.item(), "adapter_size": adapter_size,
                               "memory": mem, "sanity": san, "details": results},
                              open(ckpt_dir / "eval.json", "w"), ensure_ascii=False, indent=2)
                    ci += 1; model.train()
                if step >= CHECKPOINTS[-1]:
                    break
            if step >= CHECKPOINTS[-1]:
                break

        logger.info(f"  [{label}] Done: {step} steps, final loss={loss.item():.4f}")
        del model, base_model
        torch.cuda.empty_cache()

    logger.info(f"\nDone! {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
