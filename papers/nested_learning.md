# Nested Learning: The Illusion of Deep Learning Architectures

**Behrouz, Razaviyayn, Zhong, Mirrokni (2025) — NeurIPS 2025**
**arXiv: 2512.24695**

---

## 핵심 주장

아키텍처와 최적화는 본질적으로 같은 개념이다. Attention, MLP, optimizer 모두 "자신의 context flow를 압축하는 연상 기억(associative memory) 모듈"이며, 차이는 업데이트 주파수뿐이다. 현재 딥러닝 아키텍처의 다양성은 이 통일된 구조를 다른 각도에서 본 "착시"에 불과하다.

---

## 1. Nested Learning Framework

### 1.1 연상 기억 (Associative Memory)

**Definition**: Key K와 Value V가 주어졌을 때, 연상 기억은 K → V 매핑을 학습하는 연산자:

```
M* = arg min_M  L̃(M(K); V)
```

모든 구성 요소가 이 프레임워크의 인스턴스:

| 구성 요소 | Key | Value | 목적 함수 |
|---|---|---|---|
| Backprop/MLP | 레이어 입력 x̂_{l-1} | 로컬 에러 신호 δ_l | L2 regression |
| Softmax Attention | query q | KV 쌍 | Nadaraya-Watson L2 |
| Linear Attention | key k_t | value v_t | dot-product similarity |
| Delta Rule (RNN) | key k_t | value v_t | L2 regression |
| Momentum | gradient | gradient variance | 연속 압축 |
| Adam | gradient | gradient 분산 | element-wise L2 |

### 1.2 업데이트 주파수 (Update Frequency)

**Definition**: 구성 요소 A의 주파수 f_A = 단위 시간(1 data point)당 업데이트 횟수.

현재 Transformer의 두 극단:
- **Attention: f = ∞** — 매 토큰마다 KV 캐시 갱신. 대가는 휘발성 (context window 종료 시 리셋).
- **MLP: f = 0** — 프리트레이닝 후 고정. 안정적이지만 새 정보 불가.

**문제**: 이 사이에 아무것도 없다. 인간 뇌는 Gamma(30~150Hz) ~ Delta(0.5~4Hz)의 연속 스펙트럼.

### 1.3 Nested System

K개의 순서화된 레벨로 구성. 레벨 k는 최적화 문제 집합 {(L_i^(k), C_i^(k), Θ_i^(k))}. 각 파라미터는 gradient descent로 최적화:

```
θ_i^(k)_{t+1} = arg min_Φ  ⟨Φ·x_{t+1}, -∇L_i^(k)(θ^(k)_{i,t}; x_{t+1})⟩ + (1/2η)||Φ - θ^(k)_{i,t}||²
```

### 1.4 레벨 간 지식 전이 방식

1. **Direct parametric** — 다른 레벨의 파라미터에 조건부 출력
2. **Direct non-parametric** — softmax attention 등
3. **Backpropagation** — gradient flow
4. **Initialization** — MAML 스타일 meta-learning
5. **Generation** — 한 레벨이 다른 레벨의 가중치/컨텍스트 생성

---

## 2. 주요 수학적 도출

### 2.1 Backpropagation = 연상 기억

레이어 l의 gradient descent 업데이트:
```
W^l_{t+1} = W^l_t - η · δ_l · x̂^{l-1,T}
```
이는 다음과 동치:
```
W^l_{t+1} = arg min_W  ⟨W·x̂_{l-1}, δ_l⟩ + (1/2η)||W - W^l_t||²_F
```

**핵심**: value δ_l은 현재 가중치에 의존하는 **자기 참조적(self-referential)** 값. Linear attention의 외부 제공 value와 근본적으로 다름.

### 2.2 Delta Gradient Descent (DGD)

L2 regression 목적함수 사용 (dot-product 대신):
```
W_{t+1} = arg min_W  (1/2)||W·x_t - u_t||² + (1/2η)||W - W_t||²
```

해 (x_t 정규화 시):
```
W_{t+1} = W_t(I - η'·x_t·x_t^T) - η'·∇_W L(W_t; x_t)
```

업데이트가 **상태 의존적** → i.i.d. 가정 없이 시퀀스 내 의존성 포착.

### 2.3 Linear Attention ↔ Hebbian Rule

dot-product similarity `L̃ = -2⟨M·k_t, v_t⟩`를 GD로 최적화:
```
M_t = α·M_{t-1} + v_t · k_t^T
```

### 2.4 Self-Modifying (HOPE 아키텍처)

Titans의 장기 메모리를 **완전 자기 참조(self-referential)** 구조로 확장.

#### Titans vs HOPE의 핵심 차이

Titans에서는 k, v, q, η, α가 **외부(attention 입력)에서 제공**됨:
```
Titans: k_t, v_t = input에서 projection
        ℳ_t = (1-α_t)·ℳ_{t-1} + S_t    (α_t는 input에서 계산)
```

HOPE에서는 **메모리 자체가 이 값들을 생성**함 (자기 참조):
```
HOPE: v̂_{□,t} = M_{□,t-1}(v_t)       (□ ∈ {k, v, q, η, α, memory})
      → 메모리가 key, value, learning rate, decay rate를 스스로 결정
```

즉 HOPE의 메모리는 "무엇을 기억할지"뿐 아니라 "얼마나 빨리 배울지", "얼마나 잊을지"도 스스로 결정.

#### 업데이트 규칙 (DGD 기반)

```
M_{□,t} = M_{□,t-1}(α_t·I - η_t·k_t·k_t^T) - η_t · ∇L(M_{□,t-1}; k_t, v̂_{□,t})
```

분해하면:
- `M_{□,t-1}·α_t·I`: weight decay (Titans의 `(1-α_t)·ℳ`에 해당)
- `-M_{□,t-1}·η_t·k_t·k_t^T`: Delta rule의 이전 연관 제거
- `-η_t · ∇L`: surprise 기반 gradient 업데이트

**Titans 대비 추가된 것: delta rule 항 (`k_t·k_t^T`)이 이전 연관을 명시적으로 제거** → 기억 간 간섭 감소.

#### Memory 아키텍처

Deep memory: 2-layer MLP with residual
```
M(x) = x + W₁·σ(W₂·x)
```
Titans에서도 deep memory(L_ℳ ≥ 2)가 shallow보다 우수함을 보임. HOPE는 이를 기본으로 사용.

#### 왜 Self-Modifying이 중요한가

일반 neural network: 외부(backprop)에서 가중치 업데이트 신호가 옴
Self-modifying: **자기가 자기를 업데이트** → 추론 중에도 메모리 형성 가능 (backprop 불필요)

이게 "한 번 보고 기억"을 가능하게 하는 메커니즘. 단, HOPE는 처음부터 이 구조로 사전학습해야 함.

---

## 3. Continuum Memory System (CMS)

**Titans가 "단일 주파수 메모리"라면, CMS는 "다중 주파수 메모리".**

Titans: `x → Attention → [LTM (단일)] → output`
CMS:    `x → Attention → [MLP_f1 (고주파)] → [MLP_f2 (중주파)] → [MLP_f3 (저주파)] → output`

### 3.1 아키텍처

MLP 블록 체인 `MLP^(f_1)(·), ..., MLP^(f_k)(·)`, 각각 고유 chunk size C^(l):

```
y_t = MLP^(f_k)(MLP^(f_{k-1)}(...MLP^(f_1)(x_t)))
```

각 MLP의 구조는 Titans의 deep memory와 동일: `M(x) = x + W₁·σ(W₂·x)`

### 3.2 업데이트 규칙

```
θ^(f_l)_{i+1} = θ^(f_l)_i - Σ_{t=i-C(l)}^{i} η^(l)_t · f(θ^(f_l)_t; x_t)    if i ≡ 0 (mod C(l))
               = θ^(f_l)_i                                                        otherwise
```

고주파 블록은 자주 업데이트 (단기 기억), 저주파 블록은 드물게 업데이트 (장기 기억).

**Titans와의 차이:**
- Titans: 하나의 LTM이 모든 주파수를 처리 (α_t gate로 망각 조절)
- CMS: 여러 MLP가 각각 고정된 주파수로 분리 (chunk size로 결정)
- CMS의 장점: 주파수 간 간섭 없음 (고주파가 잊어도 저주파에 남아있음)

### 3.3 세 가지 변형

1. **Nested CMS**: 블록 s+1의 초기 상태가 블록 s에서 meta-learn됨. Higher-order ICL 가능. Context window 후 리셋.
2. **Sequential CMS**: 블록이 순차 연결. 모든 초기 상태가 최저주파 레벨에서 backprop으로 연결.
3. **Independent (Head-wise) CMS**: 블록이 독립 실행, `y_t = Agg(MLP^(f_k)(x_t), ..., MLP^(f_1)(x_t))`로 집계.

### 3.4 CMS가 continual learning에 도움이 되는 이유

블록 MLP^(f_s)가 업데이트 중 지식을 잊어도, 저주파 블록 MLP^(f_{s'}) (s' < s)에 해당 지식이 남아 있을 수 있음. 초기 상태에 대한 backprop이 **시간적 루프**를 만들어 지식 복구 가능.

### 3.5 효율성

각 timestep에서 업데이트 예정인 블록만 수정. 4-level CMS, 최고주파 f̂일 때 평균 업데이트 비용: `O((1/f̂) · (L_layer/5) · d_in²)`. Chunk 내 시퀀스 병렬화 가능.

---

## 4. 실험 결과 (Hope 모델)

### 4.1 언어 모델링 (1.3B params / 100B tokens)

| 모델 | Wiki ppl | LMB ppl | Avg reasoning acc |
|---|---|---|---|
| Transformer++ | 17.92 | 17.73 | 53.38 |
| Titans | 15.60 | 11.41 | 56.82 |
| **Hope** | **14.39** | **10.08** | **58.04** |
| RWKV-7 | 18.44 | 15.96 | 55.30 |

### 4.2 NIAH (Needle-in-a-Haystack)

Hope: S-NIAH-1 모든 길이에서 100%. S-NIAH-2 16K: Hope 78.2 vs Titans 75.4 vs RWKV-7 12.6.

### 4.3 Class-Incremental Learning

Hope-enhanced Llama3-8B가 CLINC, Banking, DBpedia에서 ICL, EWC, InCA 모두 능가.

### 4.4 BABILong

Hope: 10M context length까지 성능 유지. Titans/ARMT는 1M 이후 하락, GPT-4는 128K~256K에서 실패.

### 4.5 Formal Language Recognition

Hope: 모든 태스크에서 **100%** (Parity, (aa)*, (abab)*, a^n b^n, a^n b^n c^n, Shuffle-2). Transformer는 대부분 0%.

### 4.6 Ablation

| 제거 항목 | ppl 변화 | acc 변화 |
|---|---|---|
| DGD 제거 | 12.24 → 13.41 | — |
| CMS 제거 | → 13.04 | — |
| inner v projection 제거 | — | 58.1 → 55.1 |

---

## 5. Titans → HOPE 진화 요약

| 구성요소 | Titans | HOPE (Nested Learning) |
|---|---|---|
| 메모리 구조 | 단일 LTM (deep MLP) | 다중 MLP 체인 (CMS) |
| 업데이트 규칙 | surprise + momentum + decay | DGD (delta rule + decay) |
| 자기 참조 | 없음 (k,v,α는 외부에서) | **완전 자기 참조** (k,v,η,α를 메모리가 생성) |
| 주파수 | 단일 (α_t로 동적 조절) | 다중 (chunk size로 명시적 분리) |
| 수면 | 없음 | **Sleep/Wake cycle** + Knowledge Seeding |
| 규모 | 170M-760M, 15-30B tokens | 1.3B, 100B tokens |
| 성능 | Wiki ppl 25.43 | Wiki ppl **14.39** (대폭 개선) |

**HOPE가 Titans 대비 추가한 것:**
1. 다중 주파수 (CMS) — 단기/장기 기억의 명시적 분리
2. 자기 참조 — 메모리가 자기 학습률/감쇠율을 결정
3. Delta rule — 이전 연관을 명시적으로 제거 (간섭 감소)
4. Sleep cycle — 고주파→저주파 지식 전이 + dreaming

---

## 6. 우리 연구(DreamLoRA)와의 관계

| | Titans/HOPE (원래) | DreamLoRA (우리) |
|---|---|---|
| 접근 | 아키텍처 변경 (처음부터 사전학습) | **기존 모델에 adapter 추가** (즉시 적용) |
| 메모리 모듈 | Titans LTM / CMS MLP chain | Nested adapter (레이어 사이 MLP) |
| 업데이트 | Delta rule (forward 중) | Backprop (오프라인 dream replay) |
| 주파수 분리 | CMS chunk size | Stacked adapter (freeze+stack) |
| 망각 관리 | α_t decay gate | Sleep consolidation (distillation) |
| 선택적 활성화 | Surprise 기반 gradient | Think retrieval chain |
| 장점 | 추론 중 실시간 기억 | 기존 모델 호환, 즉시 배포 |
| 한계 | 사전학습 필요 | 오프라인 학습만 (실시간 불가) |

**DreamLoRA가 궁극적으로 가려는 방향:**
1. 현재 (경로 1): 오프라인 dream replay → 이미 작동
2. 미래 (경로 2): Titans/HOPE식 추론 중 실시간 업데이트 → 아키텍처 변경 필요

---

## 5. M3 Optimizer

두 수준의 momentum:
```
M^(1)_t = M^(1)_{t-1} + β_1 · g_t                         (고주파, 매 step)
M^(2)_t = M^(2)_{t-1} + β_3 · Σ_{i=(k-1)f}^{kf} g_i     (저주파, 매 f step)
Θ_t = Θ_{t-1} - η · (O^(1)_t + α·O^(2)_t) / (√V_t + ε)
```
여기서 `O^(j) = NewtonSchulz_T(M^(j))`

AdamW, Muon 대비 ViT/ImageNet-21K에서 최고 train/test loss. 다만 계산 비용 높음.

---

## 6. 한계

1. **Catastrophic forgetting 미해결**: CMS가 개선하지만 완전히 제거하지 못함. "로드맵이지 목적지가 아님."
2. **CMS 계산 오버헤드**: M3 optimizer는 대규모 네트워크 스케일링에 어려움.
3. **소규모 실험**: BABILong 등에서 fine-tuning 없이 10M 토큰에서 성능 하락 관찰.
4. **비볼록 수렴 보장 불완전**: nested optimization의 완전한 수렴 증명 미제공.
5. **Hope vs Transformer on recall**: 짧은 in-context recall에서는 softmax attention이 우위.

---

## DreamLoRA와의 관계

| NL/CMS 개념 | DreamLoRA 대응 |
|---|---|
| CMS chunk-based 차등 업데이트 | Dream replay 시 레이어 그룹별 다른 chunk size |
| 고주파 블록 (빠른 적응) | L1–8 LoRA (문체, 최근 기억) |
| 저주파 블록 (안정 저장) | L21–32 LoRA (핵심 지식) |
| 온라인 컨텍스트 압축 | Dream 시퀀스 순차 처리 중 실시간 파라미터 갱신 |
| chunk boundary 지식 전이 | Forward pass 순차 전이 + merge cycle |
| DGD (상태 의존 업데이트) | Dream 내 시간 순서가 업데이트에 반영 |
