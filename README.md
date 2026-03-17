# DreamLoRA

**Personalized Long-Term Memory for Language Models via Episodic Dream Replay**

Qwen 모델에 LoRA를 달고, 사용자 기억을 dream replay + CMS(Continuum Memory System) 스타일 레이어별 차등 업데이트로 개인화하는 프레임워크.

---

## 설치

```bash
# 기본 설치
pip install -e .

# 개발 (테스트 포함)
pip install -e ".[dev]"
```

**요구 사항**: Python ≥ 3.10, PyTorch ≥ 2.1, CUDA GPU (A100/H100 권장, RTX 4090도 가능)

---

## 프로젝트 구조

```
hope/
├── configs/                        # 실험별 YAML 설정
│   ├── base.yaml                   # 공유 기본값
│   ├── phase1_sft.yaml             # Phase 1: 기억 인코딩 기본 검증
│   ├── phase1_5_cms.yaml           # Phase 1.5: CMS 차등 업데이트 검증
│   └── phase2_simulation.yaml      # Phase 2: 30일 시뮬레이션
├── src/dreamlora/
│   ├── config.py                   # Pydantic 설정 모델 + YAML 로딩
│   ├── data/
│   │   ├── formats.py              # Special tokens, ChatML 포맷, 타임스탬프
│   │   ├── memory_store.py         # JSONL 기반 기억 span CRUD
│   │   ├── dream_dataset.py        # PyTorch Dataset (assistant-only loss masking)
│   │   └── user_profile.py         # 합성 사용자 프로필 + 템플릿 dream 생성
│   ├── model/
│   │   ├── loader.py               # 모델 + 토크나이저 로딩, special token 추가
│   │   ├── lora_setup.py           # LoRA 적용 + 레이어 그룹별 파라미터 매핑
│   │   └── merge.py                # 레이어 그룹별 부분 merge + reset
│   ├── training/
│   │   ├── sft_trainer.py          # 표준 SFT 트레이너 (Phase 1)
│   │   ├── cms_trainer.py          # ★ CMS 차등 업데이트 트레이너 (핵심)
│   │   └── optimizer_groups.py     # 그룹별 독립 optimizer 생성
│   ├── dream/
│   │   ├── generator.py            # Dream 생성 (로컬 모델 / API)
│   │   ├── templates.py            # 시나리오별 프롬프트 템플릿
│   │   ├── mixer.py                # 70/30 new/old dream 믹싱
│   │   └── validator.py            # Dream 품질 필터링
│   ├── sleep/
│   │   ├── orchestrator.py         # Sleep cycle 전체 파이프라인
│   │   ├── scheduler.py            # Merge 주기 스케줄링
│   │   └── state.py                # Cycle 상태 저장/복원
│   ├── eval/
│   │   ├── recall.py               # 직접 질문 정확도
│   │   ├── utilization.py          # 간접 활용 정확도
│   │   ├── general.py              # MMLU subset (일반 능력 유지)
│   │   └── benchmark.py            # 통합 벤치마크 러너
│   └── simulation/
│       ├── day_simulator.py        # 1일 시뮬레이션
│       └── scenario_bank.py        # 30일 시나리오 정의
├── scripts/
│   ├── run_phase1.py               # Phase 1 실행
│   ├── run_phase1_5.py             # Phase 1.5 실행
│   └── run_phase2.py               # Phase 2 실행
└── tests/                          # 단위 테스트 (50개)
```

---

## 핵심 개념

### 기억 Span

사용자와의 대화에서 중요한 정보를 **span** 단위로 태깅한다. 각 span에는:

| 필드 | 설명 |
|---|---|
| `level` (1-5) | 중요도. 높을수록 dream이 많이 생성되어 강하게 인코딩됨 |
| `sentiment` | `positive`, `negative`, `neutral`. 감정 토큰(`[POS]`/`[NEG]`/`[NEU]`)으로 변환 |
| `timestamp` | ISO 8601. 시간순 기억 관리 + 자연 망각 없이 최신 정보 우선 |

```python
from dreamlora.data.memory_store import MemoryStore

store = MemoryStore("memories.jsonl")
store.add(
    content="채식주의자다. 2025년 1월부터 시작했다.",
    level=4,
    sentiment="positive",
    tags=["diet"],
)
```

### Dream Replay

원본 span만 반복 학습하면 특정 맥락에 과적합된다. Dream은 기억을 다양한 시나리오에서 변형 재생하여 일반화를 촉진한다.

| 시나리오 유형 | 비율 | 예시 |
|---|---|---|
| 원본 replay | 20% | 원래 대화 맥락 그대로 재생 |
| 직접 활용 | 40% | "Python 선호" → 코드 요청에 Python으로 답변 |
| 교차 기억 | 25% | "한국어" + "Python" → 한국어 주석 코드 |
| 시간 맥락 | 15% | "3월 이직" → 5월에 "새 직장 어때?" |

`[NEG]` 기억의 dream은 실수 → `<think>`에서 인지 → 교정하는 흐름으로 생성된다.

### CMS 차등 업데이트

Nested Learning 이론에 기반하여, Transformer 레이어를 주파수가 다른 메모리 블록으로 나눈다.

```
Layer  0-7  (고주파): chunk_size=1,  lr=2e-4  → 매 step 업데이트 → 최근 기억, 문체
Layer  8-19 (중주파): chunk_size=5,  lr=1e-4  → 5 step마다      → 반복 선호, 습관
Layer 20-27 (저주파): chunk_size=25, lr=5e-5  → 25 step마다     → 핵심 지식, 정체성
```

각 그룹은 독립적인 gradient buffer와 optimizer를 가진다:

```
1. Forward + Backward → 전체 gradient 계산
2. 모든 그룹의 buffer에 gradient 누적
3. 모델 gradient 초기화
4. 각 그룹: step % chunk_size == 0이면
   → buffer / chunk_size로 정규화 → optimizer.step() → buffer 리셋
```

### Sleep Cycle

유휴 시간에 실행되는 전체 파이프라인:

```
새 태그 수집 → Dream 생성 → 70/30 믹싱 → CMS 학습 → Merge 확인 → Checkpoint
```

Merge 주기: 고주파 7 cycle, 중주파 21 cycle, 저주파 영구 유지.
Merge 시 `base_weight += lora_B @ lora_A * scaling` 후 LoRA 초기화.

---

## 실행 방법

### Phase 1: 기억 인코딩 기본 검증

합성 프로필 20개 항목으로 두 조건을 비교한다.

```bash
python scripts/run_phase1.py
```

| 조건 | 설명 |
|---|---|
| A | 원본 span만 SFT |
| B | 원본 + dream 혼합 SFT |

출력: `outputs/phase1/comparison.json`

### Phase 1.5: CMS 차등 업데이트 검증

동일 데이터로 세 조건을 비교한다.

```bash
python scripts/run_phase1_5.py
```

| 조건 | 설명 |
|---|---|
| A | 단일 LoRA, 전 레이어 균일 (lr=1e-4) |
| B | 3-그룹 CMS (L0-7/L8-19/L20-27) |
| C | 2-그룹 CMS (L0-13/L14-27) |

출력: `outputs/phase1_5/comparison.json` + 각 조건별 `cms_analysis.json`

### Phase 2: 30일 시뮬레이션

매일 2-3개 기억 추가, sleep cycle 수행, 5일마다 평가.

```bash
python scripts/run_phase2.py
```

30일 시나리오 구성:
- Day 1-5: 핵심 정체성 (이름, 직업, 건강)
- Day 6-10: 습관 (루틴, 취미, 식단)
- Day 11-15: 이벤트 (업무, 여행, 학습)
- Day 16-20: 선호 변경 (채식 중단, 에디터 변경 등)
- Day 21-25: 교차 참조 (복수 기억 결합)
- Day 26-30: 모순/시간 수정 (기존 기억 정정)

출력: `outputs/phase2/simulation_results.json`, `outputs/phase2/simulation_log.jsonl`

---

## 설정

모든 설정은 YAML 파일로 관리한다. `configs/base.yaml`이 기본값이고, phase별 YAML이 이를 오버라이드한다.

### 주요 설정 항목

```yaml
model:
  name_or_path: "Qwen/Qwen2.5-7B-Instruct"  # Qwen3.5 출시 시 변경
  dtype: "bfloat16"
  max_seq_len: 4096

lora:
  rank: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj]

cms:
  layer_groups:
    - name: "high_freq"
      layer_start: 0
      layer_end: 7
      learning_rate: 0.0002
      chunk_size: 1              # 매 step 업데이트
      merge_every_n_cycles: 7    # 7 cycle마다 base에 merge
    - name: "mid_freq"
      layer_start: 8
      layer_end: 19
      learning_rate: 0.0001
      chunk_size: 5
      merge_every_n_cycles: 21
    - name: "low_freq"
      layer_start: 20
      layer_end: 27
      learning_rate: 0.00005
      chunk_size: 25
      # merge_every_n_cycles 없음 → 영구 유지
  gradient_clipping: 1.0

dream:
  dreams_per_level:              # level별 dream 생성 수
    1: 5
    2: 10
    3: 20
    4: 35
    5: 50
  new_dream_ratio: 0.7          # 새 dream 70%, 기존 dream 30%
  generator_type: "local"        # "local" 또는 "api"
```

Python에서 로딩:

```python
from dreamlora.config import ExperimentConfig
config = ExperimentConfig.from_yaml("configs/phase1_5_cms.yaml")
```

---

## 주요 API

### MemoryStore

```python
from dreamlora.data.memory_store import MemoryStore

store = MemoryStore("memories.jsonl")

# 추가
span = store.add(content="Python 선호", level=4, sentiment="positive", tags=["coding"])

# 조회
span = store.get("mem_001")

# 필터링
high_priority = store.filter(min_level=4, sentiment="negative")
recent = store.filter(since="2026-03-10T00:00:00+09:00")

# 수정 / 삭제
store.update("mem_001", level=5)
store.remove("mem_001")
```

### CMSTrainer

```python
from dreamlora.training.cms_trainer import CMSTrainer
from dreamlora.data.dream_dataset import DreamDataset

trainer = CMSTrainer(model, tokenizer, config)
dataset = DreamDataset(dreams, tokenizer, max_length=4096)
stats = trainer.train_dream_stream(dataset, device="cuda")

# stats 구조:
# {
#   "avg_loss": 2.34,
#   "update_counts": {"high_freq": 25, "mid_freq": 5, "low_freq": 1},
#   "avg_gradient_norms": {"high_freq": 0.12, "mid_freq": 0.08, "low_freq": 0.05},
# }
```

### SleepOrchestrator

```python
from dreamlora.sleep.orchestrator import SleepOrchestrator
from dreamlora.sleep.state import StateManager
from dreamlora.dream.mixer import DreamPool

state = StateManager("state.json")
pool = DreamPool()

orchestrator = SleepOrchestrator(
    model, tokenizer, config,
    memory_store=store,
    state_manager=state,
    dream_pool=pool,
)

# 새로 추가된 span들에 대해 sleep cycle 실행
result = orchestrator.run_cycle(new_span_ids=["mem_001", "mem_002"])
```

### Merge

```python
from dreamlora.model.merge import merge_lora_group, merge_groups_by_schedule

# 특정 그룹만 merge
merge_lora_group(model, group_config, lora_config, optimizer=opt)

# 스케줄에 따라 자동 merge
merged = merge_groups_by_schedule(model, layer_groups, lora_config, cycle_number=7)
# merged = ["high_freq"]  ← 7 cycle 도달한 그룹
```

### 평가

```python
from dreamlora.eval.benchmark import run_benchmark

results = run_benchmark(model, tokenizer, memory_store, device="cuda")
print(results.summary())
# {"recall_accuracy": 0.85, "utilization_accuracy": 0.72, "mmlu_accuracy": 0.94}
```

---

## 테스트

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

50개 테스트:
- `test_memory_store.py` — JSONL CRUD, 필터링, persistence (25개)
- `test_cms_trainer.py` — gradient buffer 누적, chunk boundary, 그룹 독립성 (11개)
- `test_dream_generator.py` — 템플릿, validation, mixing, 프로필 dream 생성 (14개)

---

## Special Tokens

| 토큰 | 용도 |
|---|---|
| `[MEM_START]` | 기억 span 시작. `:level=N`은 자연어로 이어붙임 |
| `[MEM_END]` | 기억 span 끝 |
| `[POS]` | 긍정 감정 |
| `[NEG]` | 부정 감정. `<think>`에서 회피 행동 유도 |
| `[NEU]` | 중립 감정 |

기억 span 포맷 예시:

```
[MEM_START] :level=5
[2026-03-15T20:00:00+09:00]
[NEG] 장황한 답변에 사용자가 크게 짜증냄. 간결하게 답변해야 함.
[MEM_END]
```

---

## 성공 기준

| 지표 | 최소 | 목표 |
|---|---|---|
| 직접 질문 정확도 | ≥ 85% | ≥ 95% |
| 간접 활용 정확도 | ≥ 70% | ≥ 85% |
| 일반 능력 유지 (MMLU) | ≥ 95% of base | ≥ 98% of base |
| 30일 retention | ≥ 80% | ≥ 90% |

---

## Qwen3.5 미출시 대응

현재 `Qwen/Qwen2.5-7B-Instruct` (28 layers)를 사용한다. Qwen3.5 출시 시 `configs/*.yaml`의 `model.name_or_path`와 레이어 범위를 변경:

```yaml
# Qwen3.5-9B (32 layers)
model:
  name_or_path: "Qwen/Qwen3.5-9B"
cms:
  layer_groups:
    - { name: high_freq,  layer_start: 0,  layer_end: 7  }
    - { name: mid_freq,   layer_start: 8,  layer_end: 19 }
    - { name: low_freq,   layer_start: 20, layer_end: 31 }
```
