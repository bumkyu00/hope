# Titans: Learning to Memorize at Test Time — 상세 정리

**Behrouz, Zhong, Mirrokni (2025) — ICML 2025**
**arXiv: 2501.00663**

---

## 핵심 아이디어

Transformer의 두 극단:
- **Attention**: 매 토큰 업데이트, context window 끝나면 소멸 (f=∞)
- **FFN/MLP**: 영구 저장, 추론 중 업데이트 불가 (f=0)

**이 사이의 중간 주파수 메모리가 없다.** Titans는 이걸 만든다.

---

## 아키텍처: 3종 메모리

### 1. Core (단기 메모리)
일반 sliding window attention. 최근 몇백 토큰만 봄.

### 2. Persistent Memory (영구 메모리)
학습 가능하지만 **입력에 무관한** 파라미터. 시퀀스 앞에 prepend:
```
x_new = [p_1, p_2, ..., p_Np] || x
```
- 태스크 관련 추상 지식을 인코딩
- 추론 중 변하지 않음
- 모든 입력에 동일하게 붙음

### 3. Long-term Memory (장기 메모리) ★ 핵심
**테스트 타임에 자기 가중치를 업데이트하는 MLP.**

구조: L_M ≥ 1 layers의 MLP. 논문에서 L_M ∈ {1,2,3,4} 실험.

#### 업데이트 메커니즘 (정확한 수식)

**Loss (associative memory):**
```
ℓ(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||²₂
```
- k_t = x_t · W_K (key projection)
- v_t = x_t · W_V (value projection)
- M_{t-1}(k_t): 메모리에 key를 넣었을 때 나오는 출력
- v_t: 실제 value
- **차이가 크면 = surprise가 크면 = 강하게 업데이트**

**업데이트 규칙:**
```
M_t = (1 - α_t) · M_{t-1} + S_t

S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)
```

| 변수 | 역할 | 특성 |
|---|---|---|
| α_t | 망각 게이트 (이전 메모리 얼마나 잊을지) | data-dependent |
| θ_t | 학습률 (현재 surprise 반영 정도) | data-dependent |
| η_t | 모멘텀 감쇠 (과거 surprise 지속 정도) | data-dependent |
| S_t | 누적 surprise 신호 | momentum 포함 |

**모든 α_t, θ_t, η_t는 입력 x_t의 함수로 학습된다.** (data-dependent)

---

## 3가지 아키텍처 변형 (정확한 데이터 흐름)

### MAC (Memory as Context) — 가장 강력

시퀀스를 고정 크기 chunk S^(t)로 분할.

```
각 chunk S^(t)마다:
  1. 쿼리 생성: q_t = S^(t) · W_Q
  2. 메모리에서 검색: h_t = M*_{t-1}(q_t)     ← * = 가중치 안 바꾸고 forward만
  3. 합치기: S̃^(t) = [Persistent || h_t || S^(t)]
  4. Attention: y_t = Attn(S̃^(t))
  5. 메모리 업데이트: M_t = update(M_{t-1}, y_t)   ← 여기서 가중치 변경
  6. 출력: o_t = y_t ⊗ M*_t(y_t)
```

- 메모리 출력이 attention의 **추가 context**로 들어감
- Attention이 "메모리 정보가 필요한지" 판단하는 역할
- 가장 표현력 높음

### MAG (Memory as Gate) — 병렬

```
  1. Persistent 추가: x̃ = [Persistent || x]
  2. Sliding window attention: y = SWA*(x̃)
  3. 메모리 처리 (전체 시퀀스): M(x̃)
  4. Gate 결합: o = y ⊗ M(x̃)
```

- Attention과 Memory가 **병렬** 처리
- ⊗ = "normalization using learnable vector-valued weights, followed by σ(.)"
- 세그먼트 분할 불필요

### MAL (Memory as Layer) — 가장 단순

```
  1. Persistent 추가: x̃ = [Persistent || x]
  2. Memory layer: y = M(x̃)
  3. Attention: o = SWA(y)
```

- 메모리가 전처리 레이어
- 가장 단순

---

## 학습 방식: 이중 루프

### Inner Loop (추론 시 / test-time)
- Memory MLP의 가중치 M를 토큰/청크마다 업데이트
- surprise 기반 gradient로 업데이트
- **이건 추론 중에 일어남**

### Outer Loop (학습 시)
- W_K, W_V, W_Q, Persistent Memory, Attention 파라미터 등
- 표준 backpropagation
- **Inner loop를 통과해서 backprop** (meta-learning과 유사)

즉, outer loop가 "inner loop가 잘 작동하도록" 전체를 학습.

### 병렬화 (chunk-wise)
chunk 크기 b에 대해:
```
M_t = β_t · M_0 - Σ_{i=1}^t θ_i · (β_t/β_i) · ∇ℓ(M_{t'}, x_i)
```
β_i = ∏_{j=1}^i (1 - α_j)

matmul과 sum만으로 계산 가능 → GPU 병렬화 가능

---

## 실험 세부

- **모델 크기**: 170M, 340M, 400M, 760M
- **학습 데이터**: FineWeb-Edu, 15B tokens (170M-400M), 30B tokens (760M)
- **시퀀스 길이**: 4K tokens
- **Memory MLP 깊이**: L_M ∈ {1,2,3,4}
- **Memory MLP 너비**: 논문에 명시적 언급 없음 (모델 크기 대비 비율 불명)

### 주요 결과

**언어 모델링 (340M, 15B tokens):**

| 모델 | Wiki ppl | Avg reasoning |
|---|---|---|
| Mamba | 30.83 | 44.64 |
| DeltaNet | 27.01 | 46.04 |
| TTT-Linear | 27.44 | 46.47 |
| **Titans MAC** | **25.43** | **47.36** |

**Long Context (BABILong, 16K):**

| 모델 | S-NIAH-1 | S-NIAH-N |
|---|---|---|
| Mamba2 | 100.0 | 5.4 |
| DeltaNet | 100.0 | 71.4 |
| TTT | 99.4 | 88.4 |
| **Titans MAC** | **100.0** | **97.4** |

2M+ context에서도 성능 유지.

---

## 이론적 결과

Theorem 4.1: Titans는 TC⁰을 넘는 문제를 풀 수 있음.
→ Transformer와 대부분의 linear recurrent model보다 이론적으로 더 표현력이 높음.

---

## 핵심 정리

1. Memory MLP의 **가중치 자체가 메모리**
2. 추론 중 실시간으로 가중치 업데이트 (test-time learning)
3. Surprise 기반으로 중요한 것만 저장
4. Base model이 메모리를 읽는 법을 **처음부터 함께 학습** (outer loop)
5. MAC이 가장 강력 — 메모리 출력이 attention의 추가 context
6. Persistent memory로 태스크 지식 인코딩
7. 처음부터 사전학습 필요 (기존 모델에 붙이기 어려움)
