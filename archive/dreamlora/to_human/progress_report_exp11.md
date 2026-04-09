# DreamLoRA Progress Report — Exp 0-12

## Research Question
LoRA/Adapter 기반으로 LLM에 개인화 기억을 인코딩하여 context window를 넘어 지속시키고, 새로운 맥락에서 자연스럽게 일반화할 수 있는가?

## Key Result: Nested Adapter가 현재 최선

### Memory-Sanity Trajectory (auto 기준)

```
Method                    | Model | Best Mem@San≥80% | Params
--------------------------|-------|------------------|--------
Uniform LoRA              | 0.8B  |    25%           | 3.8%
CMS-front LoRA            | 0.8B  |    35%           | 3.8%
Nested MLP adapter        | 0.8B  |    70%           | 0.035%
Nested + passthrough      | 0.8B  |    55%           | 0.035%
Nested + passthrough      | 4B    |    75% ★         | 0.035%
```

### 핵심 발견 10가지

1. **Nested adapter > LoRA** — 기존 레이어 고정 + 사이에 MLP 삽입. 2배 memory, 1/100 params.
2. **Think retrieval chain** — `<think>`에서 기억을 떠올리는 패턴이 spreading activation을 가능하게 함.
3. **Dream 5개/사실** — 일반화 임계점. Food/Activity 100% 일반화.
4. **CMS-front > CMS-mid** — 중간 레이어(추상 지식)를 수정하면 오히려 해로움.
5. **Passthrough pattern** — "관련 없으면 통과" 학습이 sanity 오염을 대폭 감소.
6. **4B self-correction** — sanity 급락 후 자동 회복 (0.8B에는 없는 특성).
7. **피스타치오→견과류→알레르기** — 학습 데이터에 없던 연결이 자동으로 형성됨.
8. **사전학습 편향** — 코딩→Python은 dream 20개로도 Haskell 극복 어려움.
9. **Auto 판정 과대평가** — 키워드 매칭은 실제 품질의 1.5-2배로 과대평가.
10. **Early stopping 필수** — step 200 이후 sanity 급락. 최적점은 step 100-150.

## 진행 중: Exp 12 — KV Memory Adapter

현재 adapter의 근본 문제: 모든 입력에 무조건 반응 (범용 MLP).
KV adapter: key-value 연상 기억으로 관련 key가 매칭될 때만 value 활성화.
Delta rule의 선택적 활성화를 backprop 환경에서 구현.

## 다음 단계
- KV adapter 결과 분석
- 해마-신피질 이중 adapter (빠른 학습 + 느린 공고화)
- Dream consolidation cycle (sleep phase 구현)
- GRPO/DPO로 gating policy 학습
