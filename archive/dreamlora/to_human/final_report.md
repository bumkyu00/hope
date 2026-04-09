# DreamLoRA Research Report — Experiments 0-19

## 연구 질문
LLM에 개인화 기억을 인코딩하여 context window를 넘어 지속시키고,
새로운 맥락에서 자연스럽게 일반화할 수 있는가?

## 결론: **가능하다. 조건부로.**

---

## 확립된 파이프라인

```
[Wake]
  사용자 대화/작업 → nested adapter (레이어 사이 MLP)에 기억 인코딩
  학습 데이터: think retrieval chain + passthrough patterns

[Sleep]
  Teacher(current adapter) → dream 생성 → Student(fresh adapter) 학습
  Adapter stack 50% 압축, 지식 대체로 보존

[Repeat]
  새 대화 → adapter stack → sleep consolidation → ...
```

## 실험 결과 종합 (수동 검증 기준)

### ✅ 확실히 작동하는 것

| 시나리오 | 수동 정확도 | 핵심 실험 |
|---|---|---|
| 단일 세션 기억 (코딩) | **~90%+** | Exp 15 |
| Sanity 보존 | **100%** | 전 실험 |
| Spreading activation | 작동 | Exp 4 (피스타치오→견과류→알레르기) |
| Dream 일반화 (5/fact) | Food/Activity **100%** | Exp 5 |

### ⚠️ 부분적으로 작동하는 것

| 시나리오 | 수동 정확도 | 문제 |
|---|---|---|
| 2-phase temporal (Rust→Go) | **80%** | Exp 17에서 작동, 하지만 consolidation 후 방향 혼동 가능 |
| 3-phase cycle | **~65%** | Phase 2(Go)에서 Rust↔Go 방향 혼동 |
| 다중 사용자 (2명) | **~31%** (엄격) | 사용자 간 사실 교차 오염 |
| 인공적 사실 (Haskell/밀/물) | **45-80%** | 사전학습 편향과 충돌 시 어려움 |

### ✗ 해결 안 된 것

| 문제 | 원인 |
|---|---|
| 사전학습 편향 극복 | Python→Haskell은 dream 20개로도 25% |
| Consolidation 시간 순서 | Dream 생성 시 teacher의 혼동이 student에 전파 |
| Auto 판정 신뢰도 | Auto와 수동 판정 괴리 1.5-4x |

---

## 핵심 발견 10가지

### 1. Nested adapter > LoRA
기존 레이어 고정 + 레이어 사이에 MLP 삽입. LoRA 대비:
- Memory 2배 (70% vs 35% @sanity≥80%)
- Params 1/100 (0.035% vs 3.8%)
- Sanity 보존 (100% vs 40-80%)

### 2. "데이터 > 구조" 원칙
KV adapter (explicit key-value): Memory 15%, 학습 불가
Gated adapter (learned sigmoid gate): Memory 65%
**MLP + passthrough data: Memory 80%** ← 가장 좋음

### 3. Think retrieval chain이 spreading activation을 가능하게 함
`<think>`에서 기억 retrieval 과정을 보여주는 학습 데이터 →
학습에 없던 연결도 자동 형성 (피스타치오→견과류→알레르기)

### 4. Dream 5개/사실이 일반화 임계점
3개: 42% → 5개: **75%** → 10개: 75% → 20개: 67% (과적합)
5개면 held-out 시나리오에서도 패턴 적용

### 5. CMS-front > CMS-mid
앞쪽 레이어(표면 패턴)를 수정하면 추상 지식 보존.
중간 레이어(추상 지식) 수정은 기존 지식 빠르게 손상.

### 6. Passthrough pattern이 선택적 활성화를 학습시킴
"관련 없으면 통과" 데이터 포함 시 sanity 오염 40%→88% 감소.
구조적 gating보다 효과적.

### 7. 4B가 0.8B보다 질적으로 우수
비문 제거, self-correction (sanity 급락 후 자동 회복),
더 자연스러운 응답. 하지만 auto 수치는 비슷.

### 8. Stacked adapters로 temporal update
Phase 1 adapter freeze + Phase 2 adapter 추가 →
"현재 Go, 이전 Rust" 정확히 인식 (Exp 17 수동 100%)

### 9. Sleep consolidation이 adapter 압축
Teacher(4 adapters) → dream → Student(2 adapters): 50% 압축.
하지만 시간 순서 보존이 불완전.

### 10. 사전학습과 일치하는 개인화가 훨씬 쉬움
Rust CLI (사전학습 일치): 100% ← 쉬움
Python→Haskell (사전학습 충돌): 25% ← 어려움

---

## 실용적 시사점

### 코딩 에이전트에 가장 적합
- 프로젝트 맥락 (언어, 프레임워크, 의존성)을 세션 간 기억
- Sanity 100% → 코딩 능력 손상 없음
- 사전학습 지식을 보완하는 방향이라 효과적

### 한계
- 사전학습을 거스르는 개인화는 어려움
- 다중 사용자/도메인 혼동
- 비주류 사실(Haskell, 프루테리언)은 50% 수준
- Consolidation의 시간 순서 보존 불완전

---

## 기술 스택
- 모델: Qwen3.5-4B (주력), 0.8B (빠른 검증)
- Adapter: 레이어 9, 21에 MLP (hidden=64), 0.035% params
- 학습: AdamW lr=1e-3, ~30-100 steps
- 데이터: Think retrieval chain dreams + passthrough
- GPU: A100 80GB

## 다음 가능한 방향
1. **실제 레포 기반 테스트** — 합성 대화가 아닌 진짜 코드 프로젝트
2. **Temporal consolidation 개선** — 시간 순서 태그, dream 품질 검증
3. **다중 도메인** — 코딩+개인선호+일정을 하나의 adapter에
4. **논문 작성** — 현재 findings로 workshop/demo paper 가능
5. **시스템 구축** — Claude Code/에이전트에 adapter 파이프라인 통합
