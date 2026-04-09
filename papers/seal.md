# SEAL: Self-Adapting Language Models

**Zweiger, Pari, Guo, Akyurek, Kim, Agrawal (MIT, 2025)**
**arXiv: 2506.10943**

> **주의**: PROPOSAL.md에서 "Self-Evolving Adversarial Learning"으로 기재되어 있으나, 실제 논문 제목은 "Self-Adapting Language Models"이다. Adversarial 메커니즘은 없음. "SEAS: Self-Evolving Adversarial Safety" (arXiv 2408.02632)와 혼동 가능.

---

## 핵심 주장

LLM이 자기 학습 데이터와 최적화 지시를 스스로 생성하여 가중치를 적응시킬 수 있다. 외부에서 큐레이션한 fine-tuning 데이터 없이, 모델이 스스로 "self-edit"를 작성하고 이를 SFT로 적용하여 **영구적 가중치 업데이트**를 수행한다.

---

## 1. Self-Edit 메커니즘

모델이 생성하는 "self-edit"에 포함되는 내용:
- 입력 컨텍스트로부터 재구성/합성된 학습 데이터
- 최적화 하이퍼파라미터 (learning rate, epochs)
- 데이터 증강을 위한 도구 호출
- Loss 계산 전략

이 self-edit를 SFT로 적용 → 영구적 가중치 변경.

---

## 2. 학습 알고리즘: 두 개의 중첩 루프

### Outer Loop: Reinforcement Learning

효과적인 self-edit 생성 정책 최적화:

```
L_RL(θ_t) := -E_{(C,τ)~D} [ E_{SE~LM_{θ_t}(·|C)} [ r(SE, τ, θ_t) ] ]
```

- C = 컨텍스트 (예: 학습할 passage, few-shot 예제)
- τ = 다운스트림 평가 태스크
- SE = 모델이 생성한 self-edit
- r = binary reward (적응 후 성능 개선 시 1, 아니면 0)

### Inner Loop: Supervised Finetuning

생성된 self-edit를 gradient update로 적용:
```
θ' ← SFT(θ, SE)
```

### 최적화: ReST^EM (Rejection Sampling + SFT)

Filtered behavior cloning:
- **E-step**: 컨텍스트당 M개 후보 self-edit 샘플링
- **M-step**: r=1인 것만 유지, 해당 positive 예제로 SFT

```
∇L_RL ≈ -(1/NM) Σ_i Σ_j r_ij · ∇log p_θ(SE_ij | C_i)
```

r=0인 시퀀스는 drop (stop-gradient).

### 핵심 설계

RL state = (C, θ) — 모델 자체의 파라미터를 포함. 하지만 θ를 컨텍스트에 넣을 수 없으므로, 모델은 C만 관찰. 따라서 **on-policy 필수** — 이전 iteration에서 캐시한 self-edit/reward 재사용 불가.

---

## 3. Self-Evolution 과정

ReST^EM iteration을 거치며 self-edit 품질 향상:
- 초기: 장황하고 비구조적 paraphrase
- 학습 후: 쉽게 학습 가능한 **원자적 사실(atomic facts)**로 정제된 증류

**2 round의 ReST^EM으로 GPT-4.1 생성 합성 데이터를 능가.**

---

## 4. 실험 결과

### Few-Shot Learning (ARC, Llama-3.2-1B)

| Method | Success Rate |
|---|---|
| In-Context Learning | 0% |
| TTT + Self-Edit (no RL) | 20% |
| **SEAL** | **72.5%** |
| Oracle TTT (upper bound) | 100% |

모델이 적절한 data augmentation (회전, 뒤집기, 반사, 전치, 크기 조정, 체인 변환)과 최적화 하이퍼파라미터를 자동 선택.

### Knowledge Incorporation (SQuAD, Qwen2.5-7B)

**Single-passage**:

| Method | Accuracy |
|---|---|
| Base model | 32.7% |
| Passage only | 33.5% |
| Passage + synthetic | 39.7% |
| Passage + GPT-4.1 synthetic | 46.3% |
| **SEAL** | **47.0%** |

**Continued pretraining (n=200)**:

| Method | Accuracy |
|---|---|
| Passage + GPT-4.1 synthetic | 39.4% |
| **SEAL** | **43.8%** |

**SEAL의 자기 생성 데이터가 GPT-4.1 생성 합성 데이터를 능가** (훨씬 작은 모델임에도).

---

## 5. Catastrophic Forgetting

**주요 한계로 명시.** 순차적 self-edit이 이전 태스크 성능을 점진적으로 저하시킴 (Figure 6).

제안된 (미구현) 해결책:
- 이전 태스크 regression 페널티
- Null-space constrained edit (기존 중요 파라미터 방향 직교로만 업데이트)
- Representational superposition

---

## 6. 기술 상세

| 항목 | 설정 |
|---|---|
| 모델 | Llama-3.2-1B, Qwen2.5-7B |
| 적응 | LoRA (few-shot: rank 128, α 16; RL SFT: rank 16, α 16) |
| LR | 1e-4 ~ 2e-3 |
| Self-edit 수 | 5 (knowledge) / 15 (few-shot) per context |
| ReST^EM | 2 rounds, 50 contexts per round |
| 하드웨어 | 1× A100/H100 (few-shot, 2-3h) / 2× H100 + DeepSpeed ZeRO-3 (knowledge, 6h/round) |

---

## 7. 한계

1. **Catastrophic forgetting**: 순차 self-edit이 이전 지식 침식.
2. **계산 비용**: self-edit 평가당 30-45초 (full finetuning + evaluation).
3. **Context-dependent evaluation**: 명시적 다운스트림 태스크 필요. 레이블 없는 코퍼스 스케일링 불가.
4. **소규모 실험**: 1B, 7B 모델만. 대규모 일반화 미검증.

---

## DreamLoRA와의 관계

| SEAL | DreamLoRA |
|---|---|
| Self-edit (모델이 자기 학습 데이터 생성) | Dream (합성 시나리오 생성) |
| RL로 self-edit 정책 최적화 | Level 기반 dream 수 결정 (RL 없음) |
| Binary performance reward | Level + 감정 토큰 |
| LoRA로 적응 | LoRA + 레이어별 차등 업데이트 |
| CF 미해결 | Dream 풀 관리 + 혼합 배치로 대응 |

DreamLoRA의 dream 생성은 SEAL의 self-edit 개념을 차용하되, RL을 제거하고 기억 태깅 기반 규칙으로 대체한 것.
