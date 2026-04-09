# DreamLoRA

**Personalized Long-Term Memory for Language Models via Episodic Dream Replay**

*v3.2 — 2026년 3월*

---

## 1. 한 문단 요약

DreamLoRA는 사전학습된 Qwen3.5 모델에 LoRA 어댑터를 달고, 사용자와의 대화에서 중요한 기억을 span 단위로 태깅한 뒤, 유휴 시간에 태그된 기억을 바탕으로 dream(합성 시나리오)을 생성하여 SFT로 LoRA를 학습시키는 개인화 프레임워크이다. 이론적으로 Behrouz et al.(2025)의 Nested Learning/Continuum Memory System에 기반하며, Transformer의 32개 레이어를 서로 다른 업데이트 주파수를 갖는 메모리 블록 체인으로 활용한다. 앞쪽 레이어(고주파)는 최근 기억을, 뒤쪽 레이어(저주파)는 장기 지식을 담당하며, 주기적 merge를 통해 고주파에서 저주파로 기억을 공고화한다. 기억에는 중요도(level)와 감정(special token)이 독립적으로 부여되며, thinking 모델의 자기 교정 능력을 활용하여 별도의 DPO 없이 부정적 기억으로부터의 행동 회피를 실현한다.

---

## 2. 문제: 왜 현재 LLM은 사용자를 기억하지 못하는가

현재 LLM 개인화의 주류 접근법은 시스템 프롬프트에 기억 요약을 주입하거나, RAG로 외부 DB에서 검색하는 것이다. 두 방법 모두 기억이 모델 외부에 존재하며, 매 추론마다 주입/검색 비용이 발생하고, 기억들이 서로 통합되지 않는다는 구조적 한계를 공유한다. "한국어를 선호한다"와 "Python을 쓴다"가 따로 저장되면, "한국어 주석이 달린 Python 코드"라는 통합된 행동으로 연결되기 어렵다.

더 근본적으로, 인간이 기억을 형성하는 방식과 완전히 다르다. 인간의 해마는 일회성 경험을 빠르게 인코딩하고, 수면 중 변형 리플레이를 통해 신피질의 장기 기억으로 공고화한다. 현재 LLM에는 이 공고화 과정이 없다. DreamLoRA는 이 gap을 메운다.

---

## 3. 이론적 기반: Nested Learning

DreamLoRA의 모든 설계 결정은 Behrouz et al.(2025)의 Nested Learning 패러다임에 의해 뒷받침된다.

### 3.1 Nested Learning이란

Nested Learning(NL)은 딥러닝 모델과 그 학습 과정을 "중첩된 최적화 문제들의 시스템"으로 재해석하는 패러다임이다. 핵심 통찰은 세 가지다.

**모든 구성 요소는 연상 기억(associative memory)이다.** 옵티마이저(Adam, SGD 등), 어텐션, MLP는 모두 자신의 컨텍스트 흐름을 압축하는 메모리 모듈이다. Gradient descent로 MLP를 학습시키는 것은 "데이터 포인트를 해당 예측의 놀라움(surprise)에 매핑하는 연상 기억을 최적화"하는 과정이며, 어텐션의 KV 캐시 업데이트도 동일한 프레임워크로 설명된다.

**각 구성 요소는 고유한 업데이트 주파수를 갖는다.** 기존 Transformer는 attention(주파수 ∞)과 MLP(주파수 0)라는 두 극단으로만 구성된다. NL은 이 이분법을 스펙트럼으로 확장하여, 다양한 중간 주파수의 메모리 블록이 가능함을 보인다.

**학습은 컨텍스트 압축이다.** 프리트레이닝은 전체 학습 데이터를 파라미터에 압축하는 과정이고, ICL은 컨텍스트 윈도우 내의 정보를 어텐션에 압축하는 과정이다. 둘은 동일한 원리의 다른 시간 스케일 적용이다.

### 3.2 주파수 ∞와 0: 현재 Transformer의 두 극단

**Attention = 주파수 ∞.** Attention의 KV 캐시는 매 토큰마다 새로운 key-value 쌍이 추가된다. 시퀀스 길이가 L이면 L번 업데이트되므로, 단위 시간당 업데이트 횟수가 가장 높다. Linear attention의 메모리 업데이트 `M_t = M_{t-1} + v_t · k_t^T`은 gradient descent로 연상 기억을 최적화하는 것과 수학적으로 동일한 형태이다. 대가는 휘발성 — 컨텍스트 윈도우가 끝나면 KV 캐시는 완전히 리셋되고, 방금 대화에서 학습한 모든 것이 사라진다.

**MLP = 주파수 0.** MLP(feedforward layer)의 파라미터는 프리트레이닝이 끝나면 완전히 고정된다. 추론 시 아무리 많은 토큰을 처리해도 가중치가 변하지 않는다. 프리트레이닝 데이터 전체에서 압축된 지식("서울은 한국의 수도다" 등)을 안정적으로 저장하지만, 새로운 정보를 추가할 수 없다.

**문제: 이 사이에 아무것도 없다.** 인간의 뇌는 Gamma(30~150Hz, 감각 처리)부터 Delta(0.5~4Hz, 기억 공고화)까지 연속적인 주파수 스펙트럼으로 기억을 관리한다. 방금 들은 전화번호(수 초), 오늘 점심(수 시간), 지난달 영화(수 주), 어릴 때 동네(수 년)가 각각 다른 시간 스케일에 저장된다. 현재 Transformer에는 이 중간 주파수가 전혀 없어, 즉각적 맥락(ICL)과 프리트레이닝 지식 사이에서 기억을 관리하는 것이 불가능하다.

### 3.3 Continuum Memory System (CMS)

NL을 구체화한 것이 Continuum Memory System(CMS)이다. CMS는 ∞와 0 사이를 채우는 다양한 주파수의 MLP 블록 체인을 도입한다:

```
CMS:  x → Attention(f=∞) → MLP₁(f=높음) → MLP₂(f=중간) → MLP₃(f=낮음) → y
```

고주파 블록은 빠른 적응(단기 기억)을, 저주파 블록은 안정적 저장(장기 기억)을 담당한다. 각 블록은 자신의 chunk size(업데이트 주기)에 따라 컨텍스트를 압축하며, chunk boundary에서 상위(저주파) 블록으로 지식을 전이한다.

### 3.4 CMS와 DreamLoRA: 관계 정리

중요한 관찰: **CMS는 Transformer과 아키텍처가 다르지 않다.** CMS가 하는 것은 기존 Transformer의 MLP 레이어에 "chunk size마다 온라인으로 자기 파라미터를 gradient 갱신한다"는 업데이트 규칙을 추가하는 것이다. `Attention → MLP`의 순차 구조는 동일하므로, 기존 사전학습 가중치를 그대로 초기값으로 사용할 수 있다. 바뀌는 건 추론 루프뿐이다 — 일반 Transformer는 forward pass만 하지만, CMS는 forward 중 chunk boundary에서 특정 레이어의 파라미터를 갱신하는 단계가 끼어든다 (Test-Time Training과 유사한 메커니즘).

이 관찰로부터 DreamLoRA는 두 가지 경로를 갖는다:

**경로 1: Dream-time CMS (기본 경로).** 추론 시에는 일반 Transformer + LoRA로 동작하고, dream replay 단계에서만 CMS의 온라인 레이어별 차등 업데이트를 적용한다. 추론 속도에 영향이 없고, 구현이 안전하다.

**경로 2: 추론 시 CMS 적용 (확장 경로).** Qwen3.5의 가중치를 그대로 쓰면서, 추론 루프에 레이어별 온라인 업데이트를 추가한다. 사용자와 대화하는 동안에도 모델이 실시간으로 기억을 형성하는 완전한 CMS 구현이다. 다만 추론 속도 저하, 온라인 loss 함수 설계, gradient 계산 비용 등 실용적 과제가 있다.

본 프로젝트는 경로 1로 시작하고, 검증 후 경로 2로 확장한다.

**경로 1 (Dream-time CMS) 상세:**

Dream replay 시 dream 시퀀스를 모델에 흘려보내면서, CMS의 온라인 레이어별 차등 업데이트를 구현한다:

```
Dream 시퀀스 처리 중:
  Layer 1–8의 LoRA:   매 chunk_1 토큰마다 gradient update (고주파)
  Layer 9–20의 LoRA:  매 chunk_2 토큰마다 gradient update (중주파, chunk_2 > chunk_1)
  Layer 21–32의 LoRA: 매 chunk_3 토큰마다 gradient update (저주파, chunk_3 > chunk_2)
```

이 방식에서 CMS의 핵심 원리 세 가지가 보존된다:

**온라인 컨텍스트 압축.** 일반 SFT처럼 전체 배치를 한꺼번에 학습하는 것이 아니라, dream 시퀀스를 순차적으로 흘려보내면서 각 레이어 그룹이 자기 chunk size에 맞게 실시간으로 파라미터를 갱신한다. 앞쪽 레이어는 자주 업데이트되어 최근 dream의 세부사항에 민감하고, 뒤쪽 레이어는 드물게 업데이트되어 여러 dream에 걸친 추상적 패턴을 포착한다.

**순차적 지식 전이.** 앞쪽 레이어가 먼저 업데이트된 상태에서 forward pass가 일어나므로, 고주파 그룹에서 학습된 representation이 자연스럽게 저주파 그룹의 입력으로 흘러간다. 이는 CMS의 고주파→저주파 순차 전이와 동일한 정보 흐름이다.

**주파수별 추상화 수준 분리.** Transformer의 앞쪽 레이어가 표면적 패턴을, 뒤쪽 레이어가 추상적 의미를 인코딩한다는 기존 연구와 결합하면, 고주파 업데이트(앞쪽)가 문체/형식 적응을, 저주파 업데이트(뒤쪽)가 핵심 지식 저장을 자연스럽게 담당하게 된다.

요약하면, DreamLoRA는 추론 시에는 일반 Transformer + LoRA로 동작하여 latency 영향이 없고, **dream replay 단계에서만** CMS의 온라인 다중 주파수 업데이트를 적용한다. 이는 CMS를 완전히 재현하는 것이 아니라, CMS의 핵심 학습 메커니즘을 오프라인 sleep 단계에서 충실히 구현하는 실용적 접근이다. CMS의 추론 시 온라인 메모리는 향후 아키텍처 변경을 통해 확장할 수 있는 연구 방향으로 남겨둔다.

### 3.5 DreamLoRA와 Nested Learning의 대응

| Nested Learning / CMS | DreamLoRA 대응 | 비고 |
|---|---|---|
| 최고주파 메모리 (attention, f=∞) | 컨텍스트 윈도우 내 ICL (8K) | 동일 |
| 고~저주파 메모리 블록 (추론 중 온라인) | 경로 1: 추론 시 고정 / 경로 2: 추론 시 온라인 업데이트 | 경로 2에서 CMS 완전 구현 가능 |
| 고~저주파 메모리 블록 (학습 중 온라인) | Dream replay 시 레이어별 chunk 기반 차등 업데이트 | CMS와 동일한 메커니즘 |
| 최저주파 저장소 (f=0) | Base model 파라미터 (고정) | 동일 |
| Chunk boundary 지식 전이 | Forward pass 순차 전이 + merge cycle | CMS와 동일한 정보 흐름 |
| 컨텍스트 압축 | Dream으로 다양한 맥락 제시 → LoRA에 압축 | 압축 대상이 dream 시퀀스 |

**왜 dream replay가 작동하는가:** NL에서 학습이란 컨텍스트를 파라미터에 압축하는 과정이다. 원본 span 하나만 학습하면 그 특정 컨텍스트만 압축된다. Dream을 통해 같은 기억을 다양한 맥락에서 제시하면, LoRA가 압축하는 대상이 "특정 대화"에서 "기억의 추상적 본질"로 일반화된다.

**왜 dream-time CMS가 유효한가:** 개인화 시나리오에서 기억은 밀리초가 아니라 일/주/월 단위로 변화한다. CMS의 온라인 메커니즘이 추론 중에 작동하지 않아도, sleep 단계에서 충분한 빈도로 적용되면 개인화에 필요한 시간 스케일을 커버한다.

**경로 2로의 확장:** CMS와 Transformer의 아키텍처가 동일하므로, 경로 1에서 검증된 LoRA 가중치와 dream 파이프라인을 그대로 유지한 채 추론 루프만 수정하면 경로 2로 전환할 수 있다. 이 경우 사용자와 대화하면서 실시간으로 기억을 형성하는 완전한 CMS가 실현된다.

---

## 4. 설계 결정

### 4.1 Base Model: Qwen3.5

Qwen3.5를 선택하는 이유는 다음과 같다.

- **Native Thinking Mode:** `<think>` 태그를 통한 기본 내장 추론 체인. 기억의 중요도 판단, 감정 토큰 해석, 자기 교정 등 DreamLoRA의 핵심 메커니즘이 thinking에 의존한다.
- **Hybrid Architecture (Gated Delta Network + MoE):** 특정 task에 필요한 network 부분만 활성화하는 sparse MoE 구조가 LoRA와의 결합에서 효율적이다.
- **262K Native Context:** 긴 대화 히스토리를 유지하면서 dream 생성 시 충분한 맥락을 제공할 수 있다.
- **소형 모델 라인업:** 9B(로컬 GPU), 4B(edge), 35B-A3B(서버)까지 동일 아키텍처 기반으로 하드웨어 환경에 맞게 선택 가능하다.

권장 시작점은 **Qwen3.5-9B** (단일 RTX 4090에서 추론+LoRA 학습 가능) 또는 VRAM이 제한적이면 **Qwen3.5-4B**이다.

### 4.2 날짜/시각 Prepend

모든 대화 턴에 ISO 8601 timestamp를 prepend한다.

```
[2026-03-16T14:32:00+09:00] 사용자: 나 이제 채식 안 해
```

이렇게 하면 기억 수정 시 별도의 망각 메커니즘이 필요 없다. "2월: 채식주의자"와 "3월: 채식 그만둠"이 둘 다 LoRA에 인코딩되어 있어도, 모델이 시간 순서를 학습하여 최신 정보를 자연스럽게 우선한다. 과거 기억도 보존되므로 "예전에 채식했을 때 추천해줬던 두부 요리 뭐였지?" 같은 질문에도 답할 수 있다.

### 4.3 기억 태깅: Level + 감정 토큰

기억 태깅은 두 축으로 분리된다.

**Level (span 단위, 학습량 결정)**

대화 또는 thinking의 특정 span에 1~5 레벨을 부여한다. Level이 높을수록 dream 생성 수가 많아지고, 학습 가중치가 높아진다. 모델이 자동으로 태깅하되, 사용자가 승인/수정/직접 태깅할 수 있다.

| Level | 의미 | Dream 수 (기본값) |
|---|---|---|
| 1 | 일시적 맥락 | 5 |
| 2 | 가벼운 선호 | 10 |
| 3 | 반복 선호/습관 | 20 |
| 4 | 중요한 사실 | 35 |
| 5 | 핵심 정체성 | 50 |

**감정 토큰 (토큰 단위, 기억의 감정적 색채)**

`[POS]`, `[NEG]`, `[NEU]` special token을 기억 span 내에 삽입한다. 이 토큰은 기억이 recall될 때 함께 활성화되어, thinking 과정에서 행동을 조절한다.

```
[MEM_START:level=5]
[2026-03-15T20:00:00+09:00]
사용자가 장황한 답변에 크게 짜증냄. [NEG] 앞으로 간결하게 답변해야 함.
[MEM_END]
```

핵심 원리: 부정적 기억은 "회피 행동"으로 변환할 필요가 없다. 기억 자체를 심어주면 thinking이 회피를 만든다. 트라우마처럼 강한 부정적 기억(level 5 + `[NEG]`)은 dream이 많이 생성되어 강하게 인코딩되고, thinking에서 더 빈번하게 활성화되어 행동을 바꾼다.

### 4.4 Dream 생성

태그된 기억을 직접 반복 학습하면 특정 맥락에 과적합된다. Dream은 원본 span과 합성 시나리오를 혼합하여 기억의 일반화를 촉진한다.

**배치 구성:**
- 원본 span replay: ~20% (원래 맥락 보존)
- 합성 변형 시나리오: ~80% (다양한 맥락에서의 활용)

**합성 시나리오 유형:**

| 유형 | 설명 | 예시 |
|---|---|---|
| 직접 활용 | 기억을 직접 사용하는 상황 | "Python 선호" → 코드 요청에 Python으로 답변 |
| 교차 기억 | 복수 기억을 결합하는 상황 | "한국어" + "Python" → 한국어 주석 코드 |
| 시간 맥락 | 시간 정보를 활용하는 상황 | "3월 이직" → 5월에 "새 직장 어때?" |

감정 토큰이 `[NEG]`인 기억의 dream은, 해당 실수가 발생하고 thinking에서 알아차려서 교정하는 흐름으로 생성한다.

### 4.5 Sleep 사이클: Dream-Time CMS

유휴 시간에 실행되는 sleep 사이클의 전체 흐름:

```
1. 새 태그 수집 → 2. Dream 시퀀스 생성 → 3. Dream-time CMS 학습 → (N회 반복 후) 4. Merge + Reset
```

**핵심: dream replay를 일반 SFT가 아닌 CMS 방식으로 실행한다.**

생성된 dream들을 하나의 긴 시퀀스로 연결하고, 이를 모델에 순차적으로 흘려보내면서 레이어 그룹별로 다른 chunk size에 따라 온라인 gradient update를 수행한다. 일반 SFT가 전체 배치를 섞어서 한꺼번에 학습하는 것과 달리, 이 방식은 CMS의 "시퀀스를 처리하면서 각 메모리 블록이 자기 주기로 압축한다"는 원리를 따른다.

**레이어 그룹별 chunk size:**

| 레이어 그룹 | Chunk size | 업데이트 빈도 | Merge 주기 | 담당 |
|---|---|---|---|---|
| Layer 1–8 | chunk₁ (작음) | dream 내에서 자주 | 7 sleep cycle | 최근 기억, 문체 적응 |
| Layer 9–20 | chunk₂ = chunk₁ × k | dream 내에서 가끔 | 21 sleep cycle | 반복 선호, 습관 |
| Layer 21–32 | chunk₃ = chunk₂ × k | dream 시퀀스 전체 후 1회 | 영구 유지* | 핵심 지식, 장기 정체성 |

*Layer 21–32의 LoRA는 base로 merge하지 않고 영구 유지한다.

**혼합 배치:** 새 기억의 dream 70% + 기존 기억의 dream 30%. 기존 기억 비율이 forgetting을 방지한다.

**Dream 풀 관리:** 최신 기억일수록 dream 풀에서 비율이 높다. 오래된 기억은 자연스럽게 비율이 줄지만 완전히 제거되지는 않는다.

**추론 시:** Dream-time CMS로 학습된 LoRA는 추론 시에는 일반 고정 가중치로 동작한다. 추론 속도에 영향이 없으며, 사용자 입장에서는 그냥 빠른 모델인데 밤마다 조금씩 더 잘 기억하게 된다.

---

## 5. 학습 파이프라인

전체 파이프라인은 SFT 단일 방식으로 통일한다. DPO, RL 등 추가 학습 방식은 사용하지 않는다.

### 실험 환경

| 항목 | 설정 |
|---|---|
| Base Model | Qwen3.5-9B (또는 4B) |
| LoRA | rank=16/group, alpha=32, target=q/k/v/o_proj + up/down_proj |
| 레이어 그룹 | L1–8 (고주파), L9–20 (중주파), L21–32 (저주파) |
| Hardware | RTX 4090 24GB (9B) 또는 RTX 3060 12GB (4B) |
| Sleep 주기 | 매일 1회 유휴 시간 |
| 학습 | batch=4, lr=2e-4 (고주파) / 1e-4 (중주파) / 5e-5 (저주파), AdamW |
| Special tokens | `[MEM_START]`, `[MEM_END]`, `[POS]`, `[NEG]`, `[NEU]` |

---

## 6. 검증 계획

### Phase 0: 감정 토큰 작동 검증 (최우선)

다른 모든 것보다 먼저 확인해야 할 것: `[NEG]` 토큰이 포함된 기억 span 5개를 SFT하고, thinking에서 해당 토큰이 생성되는지, 생성 시 행동이 실제로 바뀌는지를 확인한다. 이게 안 되면 감정 토큰 설계를 재고해야 한다.

### Phase 1: 기억 인코딩 기본 검증

사용자 프로필 20개 항목을 정의하고 3가지 조건을 비교한다.

- (A) 원본 span만 직접 SFT
- (B) 원본 + dream 혼합 SFT
- (C) DreamLoRA 전체 파이프라인 (sleep cycle 포함)

측정: 직접 질문 정확도, 간접 활용 정확도, 일반 능력 유지율 (MMLU 등).

### Phase 1.5: 레이어별 차등 주파수 효과 검증

레이어별 차등 LoRA가 단일 LoRA 대비 실질적 이점이 있는지 확인한다.

- (A) 단일 LoRA: 전체 레이어 동일 주파수로 학습
- (B) 3-그룹 차등 LoRA: L1–8 / L9–20 / L21–32 다른 주파수
- (C) 2-그룹 차등 LoRA: L1–16 / L17–32 두 주파수

측정: 기억 retention, 일반 능력 유지, 레이어 그룹별 기억 localization 분석.

### Phase 2: 30일 시뮬레이션

매일 2~3개 기억 추가, 30 sleep 사이클 수행. 기억별 retention rate 시계열, 기억 간 간섭도, merge 전후 성능을 추적한다.

### Phase 3: 시간 기반 기억 수정

공고화된 기억 5개를 수정하고, timestamp만으로 최신 정보가 우선되는지 검증한다.

---

## 7. 성공 기준

| 지표 | 최소 | 목표 |
|---|---|---|
| 직접 질문 정확도 | ≥ 85% | ≥ 95% |
| 간접 활용 정확도 | ≥ 70% | ≥ 85% |
| 일반 능력 유지 (MMLU) | ≥ 95% of base | ≥ 98% of base |
| 30일 retention | ≥ 80% | ≥ 90% |
| 시간 기반 기억 수정 (3 cycle 후) | ≥ 90% | 100% |
| Sleep 1 cycle 소요 (4090) | ≤ 30분 | ≤ 15분 |

---

## 8. 리스크

**Dream 품질:** 9B 모델의 합성 시나리오 품질이 낮으면 잘못된 패턴을 학습한다. 대응: dream 자체 검증 단계를 추가하고, 품질 미달 시 원본 span replay 비율을 높인다.

**LoRA 용량 포화:** 기억 수가 많아지면 low-rank 공간이 부족해진다. 대응: merge cycle 빈도를 동적으로 조절하거나, rank를 점진적으로 증가시킨다.

**감정 토큰 미작동:** Phase 0에서 검증 실패 시, 감정을 토큰이 아닌 자연어 태그(예: "이 기억은 부정적이다")로 대체하여 실험한다.

**프라이버시:** 개인 정보가 모델 가중치에 인코딩된다. 본 프로젝트는 단일 사용자 로컬 실행을 전제하며, LoRA 어댑터 파일을 암호화하여 저장한다.

---

## 9. 연구 확장 가능성

취미 프로젝트로 시작하되, 결과가 유의미하면 다음 방향으로 연구를 확장할 수 있다.

- **추론 시 CMS (경로 2):** CMS와 Transformer의 아키텍처가 동일하므로, 경로 1에서 검증된 가중치를 그대로 유지한 채 추론 루프에 레이어별 온라인 업데이트를 추가하여 완전한 CMS를 구현한다. 이 경우 대화 중 실시간 기억 형성이 가능해진다. 핵심 과제는 온라인 loss 함수 설계와 추론 속도 관리.
- **감정 토큰의 효과 분석:** `[NEG]`/`[POS]` 토큰이 thinking의 행동 조절에 미치는 영향을 정량적으로 분석. "Emotional Special Tokens as Artificial Amygdala"라는 프레이밍으로 기존 continual learning 문헌에 없는 각도를 제시할 수 있다.
- **일회성 에피소드 기억의 공고화 조건:** dream replay 횟수, 원본/합성 비율, level별 학습률이 기억 안정성에 미치는 영향을 체계적으로 실험.
- **시간 태그 기반 자연 망각:** Active Forgetting 메커니즘 없이 timestamp만으로 기억 수정이 이루어지는지 검증.
- **다중 사용자 확장:** 공유 base model + 사용자별 LoRA 어댑터 풀.
- **멀티모달 기억:** Qwen3.5의 native multimodal 능력을 활용하여 이미지/영상 기억 통합.

---

## References

- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). Nested Learning: The Illusion of Deep Learning Architectures. *NeurIPS 2025.*
- Behrouz, A., Hashemi, F., & Mirrokni, V. (2025). Language Models Need Sleep: Learning to Self Modify and Consolidate Memories. *Submitted to ICLR 2026.*
- Zweiger, A., et al. (2025). SEAL: Self-Adapting Language Models.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022.*
- Rasch, B. & Born, J. (2013). About Sleep's Role in Memory. *Physiological Reviews.*
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why There Are Complementary Learning Systems in the Hippocampus and Neocortex. *Psychological Review.*
- Qwen Team (2026). Qwen3.5: Towards Native Multimodal Agents. *Alibaba Cloud.*