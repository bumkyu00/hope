# Titans: Learning to Memorize at Test Time

**Behrouz, Zhong, Mirrokni (2025) — ICML 2025**
**arXiv: 2501.00663**

Nested Learning (HOPE)의 전작. 테스트 타임에 메모리를 업데이트하는 아키텍처의 기초를 제시.

---

## 핵심 문제

현재 Transformer의 두 극단:
- **Attention (f=∞)**: 매 토큰 업데이트, 하지만 context window 끝나면 휘발
- **MLP (f=0)**: 영구 저장, 하지만 추론 중 업데이트 불가

**이 사이에 중간 주파수 메모리가 없다.**

---

## 아키텍처: 3종 메모리

### 1. Core (단기 메모리)
일반 attention. 제한된 윈도우 크기. 직접적 토큰 의존성.

### 2. Long-term Memory (장기 메모리) ★ 핵심
**테스트 타임에 자기 가중치를 업데이트하는 neural network.**

구조: multi-layer MLP (L_ℳ ≥ 1). `M(·) = (·) + W₁·σ(W₂·(·))` (residual MLP)

#### 업데이트 메커니즘

**Surprise 기반 업데이트:**
```
ℓ(ℳ_{t-1}; x_t) = ||ℳ_{t-1}(k_t) - v_t||²₂
```
- k_t: 입력의 key, v_t: 입력의 value
- 메모리가 이 key에 대해 출력하는 것과 실제 value의 차이 = **놀라움(surprise)**
- 놀라움이 크면 → gradient가 크면 → 더 강하게 업데이트

**모멘텀 + 감쇠 업데이트 규칙:**
```
ℳ_t = (1 - α_t) · ℳ_{t-1} + S_t

S_t = η_t · S_{t-1} - θ_t · ∇ℓ(ℳ_{t-1}; x_t)
```

| 변수 | 역할 | 비유 |
|---|---|---|
| S_t | 놀라움 신호 (모멘텀 포함) | "이전 놀라움 + 지금 놀라움" 누적 |
| η_t | 모멘텀 감쇠 (data-dependent) | 과거 놀라움이 얼마나 지속되는지 |
| θ_t | 학습률 (data-dependent) | gradient를 얼마나 반영할지 |
| α_t | 망각 게이트 | 이전 메모리를 얼마나 잊을지 |

이건 수학적으로 **mini-batch gradient descent + momentum + weight decay**와 동등.

#### Deep vs Shallow Memory
- **Shallow (linear)**: 단일 행렬. 선형 연관만 학습 가능.
- **Deep (MLP 2+ layers)**: 비선형 관계 학습 가능. 더 표현력이 높음. HOPE에서 더 발전.

### 3. Persistent Memory (영구 메모리)
학습 가능하지만 **입력에 무관한** 파라미터. 시퀀스 앞에 prepend됨:
```
[p₁ p₂ ... p_Np] || x
```
태스크 관련 지식을 인코딩. 추론 중 변하지 않음.

---

## 3가지 아키텍처 변형

### MAC (Memory as Context)
```
시퀀스를 chunk로 분할 →
각 chunk마다:
  1. 메모리에서 과거 정보 검색: h_t = ℳ*(q_t)
  2. [persistent || retrieved || current] 합쳐서 attention
  3. attention 출력으로 메모리 업데이트
  4. output = attention_output ⊗ ℳ*(attention_output)
```
가장 표현력 높음. 메모리가 attention의 컨텍스트로 사용됨.

### MAG (Memory as Gate)
```
두 갈래:
  - sliding window attention (단기)
  - neural memory (장기)

output = attention_output ⊗ memory_output
```
Gating으로 두 메모리를 결합. 세그먼트 분할 불필요.

### MAL (Memory as Layer)
```
x → neural memory → sliding window attention → output
```
메모리를 레이어처럼 쌓음. 가장 단순.

---

## Delta Rule과의 관계

일반 Linear Attention:
```
M_t = M_{t-1} + v_t · k_t^T   (Hebbian, 더하기만)
→ 문제: 메모리 오버플로우. 이전 기억을 지울 수 없음.
```

Delta Rule:
```
M_t = M_t - η · (M·k_t - v_t) · k_t^T   (이전 기억 교정 가능)
```

Titans:
```
ℳ_t = (1-α_t)·ℳ_{t-1} + S_t   (weight decay로 적응적 망각)
+ momentum으로 과거 놀라움 누적
+ deep MLP로 비선형 관계
```

**Titans = Delta Rule + 적응적 망각 + 모멘텀 + 비선형(deep MLP)**

---

## 핵심 실험 결과

### 언어 모델링 (340M params, 15B tokens)

| 모델 | Wiki ppl | Avg reasoning |
|---|---|---|
| Mamba | 30.83 | 44.64 |
| DeltaNet | 27.01 | 46.04 |
| TTT-Linear | 27.44 | 46.47 |
| **Titans MAC** | **25.43** | **47.36** |

### Long Context (BABILong, 16K)

| 모델 | S-NIAH-1 | S-NIAH-N |
|---|---|---|
| Mamba2 | 100.0 | 5.4 |
| DeltaNet | 100.0 | 71.4 |
| TTT | 99.4 | 88.4 |
| **Titans MAC** | **100.0** | **97.4** |

### 스케일링
2M+ context window에서도 성능 유지 (GPT-4는 128-256K에서 실패).

---

## 망각(Forgetting) 처리

α_t (weight decay gate)가 핵심:
- α_t → 0: 이전 메모리 보존 (장기 안정)
- α_t → 1: 이전 메모리 삭제 (빠른 적응)
- **data-dependent**: 입력에 따라 자동으로 결정

이건 GLA, Mamba2 등의 data-dependent forgetting gate를 일반화.

---

## Nested Learning (HOPE)과의 관계

Titans는 **단일 주파수 메모리** — 하나의 long-term memory 모듈이 모든 것을 처리.

HOPE/Nested Learning은 이를 확장:
- **다중 주파수 CMS**: 여러 MLP가 각각 다른 chunk size로 업데이트
- **Self-modifying**: 모든 projection이 자기 value를 생성하는 완전 자기 참조 구조
- **Sleep cycle**: 고주파→저주파 지식 전이 (Titans에는 없음)

```
Titans:  x → Attention → [LTM] → output
                          (단일)

HOPE:    x → Attention → [MLP_f1] → [MLP_f2] → [MLP_f3] → output
                         (고주파)   (중주파)    (저주파)
         + self-modifying projections
         + sleep consolidation
```

---

## 우리 연구와의 관계

| Titans | DreamLoRA (우리) |
|---|---|
| LTM = MLP, 테스트 타임 업데이트 | Nested adapter = MLP, 오프라인(sleep) 업데이트 |
| Surprise 기반 선택적 업데이트 | Think chain 기반 선택적 활성화 |
| weight decay로 적응적 망각 | Stacked adapter + consolidation |
| 처음부터 사전학습 필요 | 기존 모델에 adapter 추가 (즉시 적용) |
| 추론 중 실시간 업데이트 | 오프라인 dream replay |

**핵심 차이: Titans는 아키텍처 변경이 필요하지만, 우리는 기존 모델에 adapter를 붙이는 방식.**
Titans의 surprise 기반 업데이트를 우리 adapter에 적용하면 (아이디어 4: Surprise-based Memory Selection) 개선 가능.
