# 신경과학 기반: 수면과 기억 공고화

---

## Paper 1: About Sleep's Role in Memory

**Rasch & Born (2013) — Physiological Reviews, 93, 681-766**
**PMC: PMC3768102**

---

### 1. 수면 단계와 기억에서의 역할

#### Slow-Wave Sleep (SWS / Deep NREM)

고진폭, 저주파 EEG (< 1Hz slow oscillation + 0.5~4Hz delta). 밤의 전반부 지배. **선언적(hippocampus 의존) 기억 공고화**의 핵심 단계.

세 가지 시그니처 진동:
- **Neocortical slow oscillation** (~0.8 Hz): 탈분극 "Up state" ↔ 과분극 "Down state" 교대. Master clock 역할.
- **Thalamocortical sleep spindle** (10-15 Hz): 피질 가소성 게이팅. 고주파 탈분극 창 생성.
- **Hippocampal sharp-wave ripple** (100-300 Hz): 압축된 기억 리플레이가 발생하는 창.

#### Stage N2 (Lighter NREM)

Sleep spindle + K-complex. 절차적/운동 기억 공고화. Fast spindle density가 운동 시퀀스 성능 이득과 상관.

#### REM Sleep

각성 유사 저진폭 고주파 EEG. 밤의 후반부. 절차적 기억 (복잡한 태스크), 감정 기억 처리, SWS에서의 시스템 수준 재분배 후 시냅스 안정화.

**AI 원리**: 뇌는 단일 공고화 프로세스를 사용하지 않는다. 다단계 오프라인 파이프라인에서 각 단계가 질적으로 다른 연산 수행.

---

### 2. 수면 중 기억 리플레이

#### 핵심 현상

SWS 중 hippocampal neuron이 깨어있을 때의 경험을 **동일한 시간 순서로 자발적 재활성화**. 단, **10~20배 시간 압축**.

쥐의 place cell 실험: 미로 주행 시 순차 발화한 세포들이 이후 수면에서 같은 시퀀스를 압축 리플레이.

#### 진동 조율 메커니즘 (Triple Coupling)

```
Slow Oscillation (Up phase, ~0.8Hz)
  └→ 유발: Sharp-Wave Ripple (hippocampus, 150-250Hz)
       └→ 포함: 압축된 기억 리플레이 콘텐츠
            └→ 전달: Sleep Spindle trough에 nested
                 └→ 도착: Neocortex (최대 가소성 상태)
```

1. Slow oscillation의 Up phase가 hippocampal sharp-wave ripple 발생 유도
2. Ripple이 운반하는 리플레이 콘텐츠가 spindle trough에 nested됨
3. 이 콘텐츠가 neocortex에 도착할 때 피질 뉴런은 최대 가소성 상태 (spindle 매개 탈분극)

**인과적 증거**: SWR을 선택적으로 방해하면 hippocampus 의존 기억 성능 저하.

#### 리플레이의 경험 의존적 특성

- 최근 경험한 궤적 반영 (무작위 아님)
- 보상/도파민 신호에 의해 조절 → 중요도/관련성 게이팅

**AI 원리**: 리플레이는 버퍼에서 무작위 샘플링이 아니다. (a) 시간 압축, (b) 시간 순서 유지, (c) 가소성 창과 동기화, (d) 관련성/보상 신호로 게이팅.

---

### 3. Active Systems Consolidation Theory

#### 2단계 기억 아키텍처 (Marr, 1971)

- **Fast learning store (hippocampus)**: 단일 시행으로 빠르게 인코딩. 임시적, 용량 제한, 간섭 취약.
- **Slow learning store (neocortex)**: 많은 재인출에 걸쳐 점진적 통합. 지속적, 고용량, 간섭 저항 — 단, 빠른 학습 시 catastrophic interference 위험.

#### 공고화 메커니즘

SWS 중 hippocampus의 새 기억이 반복 재활성화 → 각 재활성화가 neocortex에 작은 점진적 시냅스 변화 유도 → 시간에 걸쳐 (일~월~년) 검색이 hippocampus 독립적으로 전환 → hippocampus 흔적 제거 가능, 용량 확보.

#### 왜 수면이 필요한가

각성 시: 외부 감각 입력과 인코딩이 지배. SWS의 저 acetylcholine / 고 GABA 신경화학 상태에서:
- (a) 외부 입력 차단
- (b) Hippocampus → neocortex 정보 흐름 촉진 (각성 시 높은 ACh가 이 경로 억제)
- (c) 내부 구동 재활성화가 새 인코딩 간섭 없이 가능

**AI 원리**: "fast store"가 콘텐츠를 리플레이하여 "slow store"를 점진적으로 학습시키는 전용 오프라인 단계 필요. 새 입력 차단 필수.

---

### 4. 기억 유형별 공고화

| 기억 유형 | 주요 수면 단계 | 메커니즘 |
|---|---|---|
| 선언적/에피소드 | SWS (밤 전반) | Hippocampal-neocortical dialogue |
| 절차적/운동 | SWS (단순) + REM (복잡) + N2 spindle | Effector-specific → effector-independent 추상화 |
| 감정 | 논쟁 중 (REM vs NREM) | REM이 감정 valence/arousal 처리 가능 |

**AI 원리**: 다른 유형의 지식(사실, 절차, 보상 연결)은 다른 공고화 전략이 필요할 수 있음.

---

### 5. 리플레이 중 변환 (정확한 복제가 아님)

이것이 AI 응용에 가장 중요한 발견:

#### Gist 추출

수면 후 피험자가 실제로 제시되지 않은 범주 전형 항목을 "false recall" → 항목별 세부보다 **범주의 통계적 원형/핵심**을 추출. 공유 특징이 더 빈번히 재활성화되어 강화.

#### 통찰 및 규칙 발견

Number Reduction Task: 수면 후 피험자가 숨겨진 추상 규칙 구조를 발견할 확률이 유의하게 높음.

#### 특정 인스턴스에서 추상화

운동 시퀀스: 수면 후 effector-specific → effector-independent 표상으로 전환. 추상 시퀀스 구조 표상.

#### 기존 지식 스키마와 통합

재활성화 시 새 기억이 밀접한 기존 표상으로 활성화 확산 → "새 기억을 기존 기억 네트워크에 점진적 통합."

**AI 원리**: Dream은 단순 experience replay가 아니어야 한다. (a) 여러 경험의 통계적 규칙성 추출, (b) 특정 인스턴스에서 일반 원리로 추상화, (c) 기존 지식 구조와 통합, (d) 행동적으로 관련된 특징의 선택적 강화. → **생성적/증류 프로세스**에 가까움.

---

### 6. Synaptic Homeostasis Hypothesis (SHY)

Tononi & Cirelli의 보완 이론:

- 각성 중 학습 → 전체 시냅스 강화 → 에너지 증가, SNR 감소, 포화 위험
- SWS 중 전역 시냅스 하향 조정 → 임계값 이하 시냅스 제거
  - 에너지 보존
  - SNR 개선 (약한 연결 제거)
  - 포화 방지 (다음 날 인코딩 용량 복구)
  - 기억 공고화는 부산물 (강한 기억은 하향 조정 생존)

**Active consolidation과 통합**: SWS에서 (a) 선택적 재활성화 강화 + (b) 전역 하향 조정/제거 = "sharpen and prune".

**AI 원리**: 공고화 리플레이 후 전역 정규화/pruning 적용. Weight decay, dropout, pruning을 sleep 단계에서 적용하는 것과 유사.

---

---

## Paper 2: Why There Are Complementary Learning Systems in the Hippocampus and Neocortex

**McClelland, McNaughton, O'Reilly (1995) — Psychological Review, 102(3), 419-457**
**PubMed: 7624455**

---

### 1. Complementary Learning Systems (CLS) Theory

핵심 주장: 지능적 에이전트는 **두 개의 근본적으로 다른 학습 시스템**이 필요하다. 단일 시스템이 두 경쟁 요구를 동시에 만족시킬 수 없기 때문:
- (a) 새 정보의 빠른 학습
- (b) 많은 경험에 걸친 통계 구조의 점진적 추출 (기존 지식 손상 없이)

이는 단순한 생물학적 관찰이 아니라, 신경망 모델의 특성에서 도출된 **계산적 필연성** 주장.

---

### 2. Hippocampal System — 빠른 학습

#### 아키텍처

Entorhinal Cortex (EC) → Dentate Gyrus (DG) → CA3 → CA1 (trisynaptic) + EC ↔ CA1 (monosynaptic).

#### 희소, 결합적 표상

극도로 희소한 코딩. 각 활성 뉴런이 입력 특징의 특정 결합을 표상 → 많은 특징을 공유하는 두 경험이 완전히 비겹치는 hippocampal 표상을 가질 수 있음.

#### Pattern Separation

DG-CA3 경로가 공격적 pattern separation 수행. 유사한 입력 패턴 → 매우 비유사한 내부 표상. 메커니즘: 희소 연결, 높은 레이어 내 억제, "detonator synapse" 아키텍처.

#### Pattern Completion

CA3의 recurrent connection으로 부분 단서에서 전체 패턴 복원. 에피소드 회상의 기반.

#### 빠른 Hebbian 학습

단일 시행 학습. 공활성 뉴런 연결 강화. 높은 learning rate → one-shot 인코딩 가능하지만 덮어쓰기 취약.

**AI 원리**: Fast store는 희소, 고차원 표상 사용. 유사 입력 → 비유사 내부 코드 (pattern separation). 빠른 few-shot 학습 지원. 부분 단서 검색 지원.

---

### 3. Neocortical System — 느린 학습

#### 분산, 겹치는 표상

많은 뉴런에 걸쳐 분산. 각 뉴런이 다수 항목/개념 표상에 참여. 유사 항목이 신경 기질 공유.

#### 느린, 오류 구동 학습

많은 노출에 걸쳐 점진적으로. 작은 increment로 예측 오류 기반 가중치 조정 = **낮은 learning rate의 gradient descent**.

#### 통계 구조 추출

느리고 인터리브된 학습이 범주, 원형, 규칙성, 추상 규칙 발견. 개별 에피소드 세부는 손실되지만 일반화 가능한 지식 보존.

#### 왜 느린 학습이 계산적으로 필요한가

높은 learning rate → 새 경험이 가중치 구성을 극적으로 이동 → 축적된 통계적 규칙성 표상 파괴. 낮은 learning rate가 각 경험을 약간 수용하면서 기존 지식의 대부분 보존. 많은 인터리브된 노출에 걸쳐 공유 특성을 포착하는 표상으로 수렴.

**AI 원리**: Slow store = 낮은 learning rate로 학습되는 표준 신경망. 다양한 경험의 인터리브 제시로 학습. 분산, 겹치는 표상으로 일반화.

---

### 4. Catastrophic Interference 문제

#### 정의

표준 신경망을 task A → task B 순차 학습시키면, B 학습이 A 성능을 **갑작스럽고 거의 완전하게 파괴**. 점진적 저하가 아닌 급격한 삭제.

#### 계산 메커니즘

분산 표상에서 패턴 지식이 많은 가중치에 걸쳐 인코딩. 이 가중치가 다른 패턴과 공유. 새 패턴 학습 시 gradient descent가 이 공유 가중치를 새 패턴 오류 감소 방향으로 조정 — 단, 이전 패턴에 대한 제약 없이. 새 태스크가 이전과 비유사할수록 간섭이 파괴적.

#### 가중치 공간 시각화

A 학습 → 가중치 공간에서 A 해결 점. B 학습 → B 해결 점으로 이동 (A 해결과 임의로 먼 곳). 인터리브 학습 → 두 태스크를 동시 만족하는 단일 점 찾기 가능 (낮은 LR + 혼합 시).

---

### 5. 인터리브된 리플레이 — 해결책

#### 메커니즘

Neocortex를 순차적으로 학습시키는 대신, hippocampus가 저장된 기억을 **서로 간 + 새 경험과 인터리브하여 리플레이**. 순차/집중 학습 (CF 유발) → 인터리브 학습 (기존 지식 보존).

#### 왜 인터리빙이 작동하는가

여러 태스크/범주의 예제가 인터리브될 때, 각 가중치 업데이트가 모든 활성 패턴의 요구 사이 작은 타협. 네트워크가 모든 패턴을 동시 만족하는 가중치 구성 탐색. 낮은 LR + 충분한 인터리빙으로 전체 통계 구조를 포착하는 공유 표상으로 수렴.

#### Pseudo-Rehearsal

원본 경험의 정확한 복제가 불필요. 이전 입력의 통계적 분포를 근사하는 합성 패턴 ("pseudo-patterns")도 동등하게 효과적. Hippocampal replay는 정확하지 않음 — 재구성적, 잠재적으로 변환적.

**AI 원리**: LLM 공고화는 새 경험 + 기존 경험(또는 합성 근사)의 인터리브 필수. 정확한 리플레이 불필요 — 기존 지식 분포를 포착하는 생성적 리플레이도 가능. 공고화 중 낮은 LR.

---

### 6. 공고화 프로세스: Hippocampus → Neocortex 전이

4단계:
1. 기억이 hippocampus에 빠른 시냅스 변화로 저장
2. 이 변화가 neocortex에서 최근 기억 재인출 지원 (리플레이)
3. 각 재인출마다 neocortical 시냅스가 약간 변화 (낮은 LR, 점진 축적)
4. 원격 기억은 축적된 neocortical 변화에 기반 (hippocampus 독립 검색)

#### 시간 역학

완전 공고화 시간: 일~월~년, 새 정보와 기존 스키마의 일관성에 따라 다름.
- **스키마 일관적 정보**: 빠른 공고화 (필요한 가중치 변화 작음)
- **스키마 비일관적 정보**: 더 많은 리플레이 사이클, 더 광범위한 재구조화

**AI 원리**: 공고화는 적응적이어야 함. 기존 모델 지식과 일관된 정보 → 적은 리플레이 + 높은 LR 가능. 새롭거나 모순되는 정보 → 더 많은 인터리브 리플레이, 더 낮은 LR. → **노벨티 감지 메커니즘** 필요.

---

### 7. Connectionist 모델링 증거

#### Rumelhart의 의미 인지 모델

- 인터리브 제시로 학습 → 계층적 범주 구조 발견 성공
- 순차 학습 → catastrophic interference로 이전 범주 파괴

**핵심**: 구조 발견은 느린 LR + 인터리브가 **동시에** 필요. 순차 학습은 어떤 LR에서도 실패.

#### 손상 시뮬레이션

Fast-learning 구성 요소 제거 + 높은 LR 직접 학습 → 전행성 기억상실 (새 장기 기억 형성 불가) + 역행성 기억상실 (최근 기억 손실, 원격은 유지) 재현 = 실제 hippocampal 손상 환자 패턴.

---

---

## 종합: DreamLoRA를 위한 계산 원리 10가지

| # | 원리 | DreamLoRA 대응 |
|---|---|---|
| 1 | **이중 저장소**: fast (에피소드) + slow (파라미터) | 태그된 span (fast) → LoRA (slow) |
| 2 | **오프라인 공고화 ("sleep")**: 새 입력 차단, 내부 처리 | Sleep cycle (유휴 시간) |
| 3 | **인터리브 리플레이**: 새 + 기존 경험 혼합 | 혼합 배치 (새 dream 70% + 기존 30%) |
| 4 | **낮은 LR**: 점진적 통합 | 저주파 레이어: lr=5e-5 |
| 5 | **변환적 리플레이**: 정확 복제가 아닌 추상화/통합 | Dream = 합성 시나리오 (원본 20% + 변형 80%) |
| 6 | **다단계 처리**: 다른 단계가 다른 연산 | 레이어 그룹별 다른 chunk size |
| 7 | **관련성 게이팅**: 중요한 경험 우선 | Level 기반 dream 수 (L5: 50, L1: 5) |
| 8 | **항상성 정규화**: 공고화 후 pruning | Merge + Reset cycle |
| 9 | **스키마 적응적 공고화**: 일관된 정보는 빠르게 | (미구현 — 향후 확장 가능) |
| 10 | **시간 압축**: 리플레이는 실시간보다 빠름 | Dream 시퀀스 압축 처리 |
