# Language Models Need Sleep: Learning to Self Modify and Consolidate Memories

**Behrouz, Hashemi, Mirrokni (2025) — Submitted to ICLR 2026 (rejected)**
**OpenReview: iiZy6xyVVE**

---

## 핵심 주장

현재 LLM은 "전행성 기억상실(anterograde amnesia)" 상태다. Attention (f=∞)과 MLP (f=0) 사이에 기억 전이 메커니즘이 없어, 단기 기억(in-context)을 장기 기억(parametric)으로 공고화할 수 없다. 누락된 요소는 **수면(sleep)** — 취약한 단기 기억을 안정적 장기 지식으로 공고화하는 오프라인 프로세스.

---

## 1. Sleep/Wake 사이클

### Wake (Active) Phase
모델이 정상 동작. 외부 입력 수신, ICL 수행, CMS 블록이 지정 주파수로 업데이트.

### Sleep Phase
**외부 입력 없음.** 내부 계산만 수행:
1. **Memory Consolidation** (NREM/SWS 유사)
2. **Dreaming / Self-Improvement** (REM 유사)

Sleep 타이밍: CMS chunk size {C(1), ..., C(k)}에 의해 결정. Step {C(1)×b, ..., C(k)×b}에서 sleep 발생 — 메모리 블록 업데이트(기존 지식 덮어쓰기) 전에 공고화 수행.

---

## 2. Memory Consolidation (NREM 유사)

반복적, 계층적 프로세스. 고주파(취약) → 저주파(안정) 방향으로 지식 흐름.

### Step 1: Parameter Expansion
MLP 블록을 sparse MoE + router로 모델링. 블록 l*에 **새 low-rank expert** 추가:
- `A^(f_{l*}) ∈ ℝ^{d × d_low}`, `B^(f_{l*}) ∈ ℝ^{d_low × d}` (d_low << d)
- 새 파라미터만 업데이트, 기존 파라미터는 **전부 고정** → 간섭 제로
- 구현: 사전 할당 후 mask로 활성화/비활성화 (동적 추가 대신)

### Step 2: Knowledge Seeding (아래 §3 상세)

### Step 3: Synaptic Pruning
공고화 완료 후, 이전 sleep에서 블록 (l*-1)에 추가했던 low-rank 파라미터를 **리셋/제거**. 용량 확보.

---

## 3. Knowledge Seeding (Distillation)

"상향 증류" — 빠른(작은) 모델 → 느린(큰) 모델 방향. 두 구성 요소:

### 3a. On-Policy Distillation (GKD 기반)

```
L(θ, θ_exp) = (1-λ) · E_{(x,y)~D} [F(LM_θ ∥ LM_{θ_exp})(y|x)]
             + λ · E_{x~D} [E_{y~LM_{θ_exp}(·|x)} [F(LM_θ ∥ LM_{θ_exp})(y|x)]]
```

- D: teacher(LM_θ)가 생성한 데이터 (dreaming의 일부)
- F: teacher-student 출력 분포 간 divergence
- λ: on-policy(student 생성) vs off-policy(teacher 생성) 비율
- Student의 sampling distribution에 대해 gradient backprop 안 함 (안정성)

### 3b. Learning to Imitate (LTI) — RL 기반

Teacher 생성 데이터(dreams)에서 랜덤 prefix 추출 → student가 continuation 생성 → 보상:

```
r(d̂(i); d(i); LM_{θ_exp}) = γ · r_sem + (1-γ) · r_abs
```

- **r_sem**: 의미적 유사도 (frozen reward model 판단, binary)
- **r_abs**: Levenshtein distance 기반 토큰 유사도:
  ```
  r_abs = 1 - z(d̂, d) / max{|d̂|, |d|}    if z(d̂, d) ≤ z_0
        = 0                                  otherwise
  ```

### 통합 Knowledge Seeding 목적함수

```
L_KS(θ, θ_exp) = E_x [(1-α) · E_{y~LM_{θ_exp}} [r(y)] - α · E_{y~LM_{θ_exp}} [D(LM_θ ∥ LM_{θ_exp})(y|x)]]
```

α: distillation vs LTI reward 균형.

---

## 4. Dreaming (REM 유사) — Self-Improvement

SEAL (Zweiger et al., 2025)을 기반으로 세 가지 수정:

### Dream 생성
Task (C, τ)가 주어지면:
1. m ≥ 1개 dream 생성: {DREAM(i)} ~ LM_θ(·|C)
2. 각 MoE router가 **추가로 랜덤 expert 선택** → 무관한 지식 혼합으로 탐색 촉진

### Dream 선택 (Gradient 기반 중요도)

```
g_DR^(i) = ∇_θ L_SFT(DREAM(i), θ)
```

Top-k (중요도) + b개 랜덤 (다양성) 선택.

### Dream 적용

각 선택된 DREAM(i)에 대해:
1. 격리된 모델 인스턴스 생성
2. LoRA SFT: θ'(i) ← SFT(θ(i), DREAM(i))
3. 보상: 성능 개선 여부 (binary)

ReST^EM 알고리즘으로 최적화.

---

## 5. CMS와의 관계

CMS 출력:
```
y_t = MLP^(f_k)(MLP^(f_{k-1)}(...MLP^(f_1)(x_t)))
```

CMS 업데이트:
```
θ^(f_l)_{i+1} = θ^(f_l)_i - Σ_{t=i-C(l)}^{i} η^(l)_t · f(θ^(f_l)_t; x_t)    if i ≡ 0 (mod C(l))
```

**Sleep 논문의 추가 기여**: CMS는 다중 주파수 아키텍처를 정의하지만, 느린 블록 업데이트 시 catastrophic forgetting 방지를 다루지 않음. Sleep이 **공고화 메커니즘**을 제공 — 업데이트가 기존 지식을 덮어쓰기 전에 상위 계층으로 지식 전이.

---

## 6. 실험 결과

### Long-Context (LLaMA-8B backbone)

| Method | MTOB (chrF) | LongHealth (Acc) | QASPER (ppl↓) |
|---|---|---|---|
| ICL | 34.7 | 53 | 1.3 |
| Cartridges | 35.1 | 54 | 1.2 |
| Duo Attention | 35.6 | 49 | 1.5 |
| **Sleep** | **40.3** | **59** | **1.1** |

### BABILong

Sleep-augmented LLaMA-8B: **10M context length**까지 90%+ 정확도.

### Knowledge Incorporation (SQuAD)

| Method | Single (n=1) | Continued (n=200) |
|---|---|---|
| Base | 31.9 | 31.9 |
| Fine-tuned (no dreaming) | 33.4 | 32.0 |
| SEAL | 46.7 | 43.2 |
| Sleep (Transformer) | 48.1 | 44.3 |
| **Sleep (4-level CMS)** | **48.9** | **46.2** |

**4-level > 2-level**: CMS 계층 구조의 효과 검증.

### Few-Shot Learning (ARC, Llama-3.2-1B)

| Method | Success Rate |
|---|---|
| ICL | 0% |
| TTT | 10% |
| SEAL | 72.5% |
| **Sleep** | **80%** |

---

## 7. Catastrophic Forgetting에 대한 입장

1. **CF는 근본적으로 용량 문제**: 인코딩 방법이 아니라 제한된 용량 때문에 파라미터 덮어쓰기 불가피.
2. **CMS 다중 주파수가 내재적으로 CF 감소**: 다른 블록이 다른 속도로 업데이트.
3. **Sleep의 2단계 설계가 추가 강건성**: Consolidation은 새 파라미터만 업데이트 (간섭 제로). Dreaming은 반복적 자기 개선만의 CF보다 강건.
4. **점진적 파라미터 확장**: MoE expert 추가로 시간에 따라 용량 증가.

---

## 8. 핵심 하이퍼파라미터

| 파라미터 | 역할 |
|---|---|
| λ | on-policy vs off-policy 비율 |
| α | distillation vs LTI reward 균형 |
| γ | semantic vs absolute reward 가중치 |
| z_0 | Levenshtein 임계값 |
| d_low | low-rank 차원 |
| m | dream 수 |
| k, b | dream 선택 (top-k + random b) |

---

## DreamLoRA와의 관계

| Sleep 논문 | DreamLoRA |
|---|---|
| MoE expert 확장 (consolidation) | LoRA adapter (더 가벼움) |
| SEAL 기반 dream + RL reward | Dream 생성 후 SFT (RL 없음, 단순화) |
| GKD + LTI 통합 distillation | SFT 단일 방식 |
| CMS 온라인 업데이트 | Dream-time CMS (오프라인만) |
| Binary performance reward | Level + 감정 토큰 기반 가중치 |

DreamLoRA는 이 논문의 dream + sleep 사이클 컨셉을 가져오되, RL/distillation 복잡도를 제거하고 SFT로 단순화한 접근.
