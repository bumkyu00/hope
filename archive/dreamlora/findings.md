# DreamLoRA — Research Findings (Final)

## Current Understanding

LLM에 개인화 기억을 인코딩하여 context window를 넘어 지속시키는 것은 **가능하다.**

검증된 핵심 메커니즘:
1. **Nested adapter** — 기존 레이어 고정, 사이에 MLP 삽입 (0.035% params)
2. **Think retrieval chain** — `<think>`에서 기억 retrieval 과정 학습 → spreading activation
3. **Passthrough pattern** — "관련 없으면 통과" 데이터로 선택적 활성화 학습
4. **Stacked adapters** — Phase별 adapter freeze+stack으로 시간 순서 구분
5. **Dream 5개/사실** — 패턴 일반화 임계점

## Key Results

| 시나리오 | Memory | Contamination | Sanity | 실험 |
|---|---|---|---|---|
| 코딩 어시스턴트 (단일 세션) | **100%** | **0건** | 100% | Exp 15 |
| 시간 변경 (Rust→Go) | **100%** (수동) | 0건 | 100% | Exp 17 |
| 다중 사용자 (2명) | 54% | 0건 | 100% | Exp 14 |
| 인공적 사실 (Haskell/밀/물) | 75-80% | varies | 80-85% | Exp 9,11,13 |

## Patterns and Insights

### Nested adapter > LoRA
- LoRA: 기존 레이어 출력 수정 → 기존 지식과 경쟁 → "배열을 지현하기" 오염
- Nested: 기존 레이어 완전 보존, 사이에 추가 → 구조적 간섭 차단
- 0.035% params로 LoRA(3.8%)의 1/100

### "데이터 > 구조" 원칙
- KV adapter (explicit 선택성): Mem 15%, 학습 불가
- Gated adapter (learned 선택성): Mem 65%, MLP보다 나쁨
- **MLP + passthrough data**: Mem **80%**, 가장 좋음
- 결론: 복잡한 구조보다 좋은 데이터가 더 효과적

### 사전학습 alignment가 핵심
- 사전학습과 일치하는 개인화 (Rust CLI → cargo/clap): **100%** 성공
- 사전학습과 충돌하는 개인화 (Python→Haskell 강제): **25%** 실패
- 실용 시나리오는 대부분 사전학습을 보완하는 방향 → DreamLoRA가 효과적

### Stacked adapter = 시간적 CMS
- Phase 1 adapter (frozen) = 저주파 (과거, 안정)
- Phase 2 adapter (new) = 고주파 (현재, 최신)
- Stack 순서 = 시간 순서
- "Go입니다. 이전 Rust에서 마이그레이션" — 현재와 과거 동시 인식

### auto 판정의 한계
- 키워드 매칭은 실제 품질의 50-200%로 과대/과소평가
- Exp 17: auto 0% → 수동 100% (auto 기준이 잘못됨)
- 수동 검증이 절대적으로 필요

## Lessons and Constraints

- **HF Trainer 금지**: gradient_accumulation + gradient_checkpointing → model collapse
- **tokenizer.apply_chat_template() 필수**: 수동 ChatML 불가
- **0.8B 비문**: 0.8B에서는 adapter 학습 후 한국어 깨짐. 4B 이상 권장
- **Step 100-150이 sweet spot**: 이후 과적합으로 sanity 하락
- **Dream 20개는 과적합**: 5개가 최적, 10개 이상 불필요
- **CMS-front > CMS-mid**: 중간 레이어 수정은 추상 지식 손상
- **모든 실험에서 체크포인트/로그/입출력 저장 필수**

## Architecture Summary

```
기존 Qwen3.5 모델 (완전 고정)
  [Layer 0-9]  →  [Adapter Phase 1 (frozen)]  →  [Adapter Phase 2 (trainable)]
  [Layer 10-21]  →  [Adapter Phase 1 (frozen)]  →  [Adapter Phase 2 (trainable)]
  [Layer 22-31]

학습 데이터:
  - Think retrieval chain dreams (사실당 5개)
  - Passthrough patterns (관련 없는 질문 → 통과)

추론:
  - 모든 adapter가 활성 (stack 순서로 적용)
  - 최신 adapter가 마지막에 적용 → 최신 정보 우선
```

## Exp 18: Sleep Consolidation — Stack compression works

Teacher (4 adapters, 1.3M) → dream generation → Student (2 adapters, 655K)
Student가 Teacher와 동등한 성능으로 **50% 압축** 달성.

이건 전체 Wake/Sleep 파이프라인의 마지막 퍼즐:
1. Wake: 대화 → adapter stack에 새 지식 추가
2. Sleep: stacked teacher가 dreams 생성 → single student로 consolidation
3. Repeat: stack 초기화 → 새 대화 → sleep → ...

## Exp 19 수동 검증 — 솔직한 평가

Auto: 전 phase 100% → **수동: ~65%**

| Phase | Current (수동) | Sanity |
|---|---|---|
| Rust | ✓ 100% | ✓ 100% |
| Go | ✗ ~25% (Rust↔Go 혼동) | ✓ 100% |
| Python | ⚠️ ~70% | ✓ 100% |

**핵심 문제: consolidation이 시간 순서를 정확히 보존하지 못함.**
Phase 2에서 "현재 Rust, 이전 Go" — 방향이 반대.
History도 "Rust→C++→Python"으로 Go가 빠짐.

**Sanity만큼은 전 phase에서 100% 확실.**

## Exp 21: Ablation — Think chain is the key

| Component | Sanity effect |
|---|---|
| **Think chain** | **+40%p** (20%→60%) ← 핵심 |
| Passthrough (without think) | 0%p |
| Passthrough (with think) | +0%p sanity, +10%p memory |

Think chain이 "기억을 떠올렸지만 적용하지 않는" 과정을 학습시켜 선택적 활성화를 가능하게 함.
Passthrough data는 보조적 — think chain 없이는 효과 없음.

**DreamLoRA의 핵심 3요소 (중요도 순):**
1. **Nested adapter** — 구조적 sanity 보존 (기존 레이어 고정)
2. **Think retrieval chain** — 선택적 기억 활성화 (sanity +40%p)
3. **Passthrough data** — 추가 memory 개선 (+10%p, think 필요)

## Honest Assessment

**진짜 작동하는 것:**
1. 단일 세션 기억 인코딩 (Exp 15: 수동으로도 높은 정확도)
2. Sanity 100% 보존 (nested adapter의 확실한 장점)
3. Spreading activation (피스타치오→견과류→알레르기)
4. Passthrough로 선택적 활성화

**부분적으로 작동하는 것:**
1. Stacked temporal (Exp 17: 수동 100%이지만, 3 phase에서 혼동)
2. Consolidation (50% 압축되지만 시간 순서 혼동)

**auto 판정이 심각하게 과대평가:**
- Exp 9: auto 70% → 수동 45%
- Exp 14: auto 54% → 수동 31%
- Exp 19: auto 100% → 수동 65%

## Exp 25: 2B Model Breakthrough

Qwen3.5-2B로 스케일업, 코딩 도메인 (FastAPI 31 facts):
**Mem 80%, San 80%** — 0.8B (0-27%) 대비 극적 향상.
진짜 프로젝트 지식 recall 확인 (SELECT FOR UPDATE, black+ruff 등).

## Exp 26 시리즈: Novel Domain (달의 정원)

**도메인 전환: 일반 지식으로 절대 맞출 수 없는 가상 소설 35 facts.**

### CMS 비교 (Exp 26)
| Condition | Best Mem@San≥80% |
|---|---|
| Uniform | 27% |
| CMS chunk 1/5 | 7% |
| CMS chunk 1/10 | 7% |
| CMS lr-diff | 7% |
| 3-adapter | 20% |

**CMS는 sanity 완벽 보존 (100%)하지만 memory가 지나치게 낮음.** Uniform이 balanced.

### Dream 다양성이 핵심 (Exp 26c)
| Dreams/fact | Best Mem@San≥80% |
|---|---|
| 1 dream/fact | 40% |
| 3 dreams/fact | **60%** (+50% 향상) |
| (5 dreams/fact) | *(testing)* |

**이건 Exp 5의 결과와 일치** — dream 다양성이 일반화의 핵심.

### Phase 2 Retrieval SFT: 도메인에 따라 다름
- **코딩 도메인**: Phase 2가 효과적 (사전학습 지식과 시너지)
- **소설 도메인**: Phase 2가 역효과 (60% → 33%, 새 정보 위에 또 새 패턴을 덮어씌움)

## Updated Lessons

- **2B >> 0.8B**: 0.8B는 비문+한계, 2B는 진짜 recall 가능
- **Dream 다양성 3→5/fact**: 일반화의 임계점. 1/fact으로는 패턴 학습 안 됨
- **Step 150-200 sweet spot** (2B): 이후 과적합
- **CMS의 역할 재평가**: Sanity는 보존하지만 memory cost가 너무 높음. Nested adapter 자체가 이미 sanity를 잘 보존하므로, CMS의 추가 가치는 제한적.
- **Phase 2는 신중히**: 새로운 정보 도메인에서는 역효과 가능

## Open Questions

1. **5 dreams/fact으로 70-80% 가능한가?** (Exp 26d 진행 중)
2. **Adapter size 128**: 35 facts에 대해 더 큰 adapter가 도움이 되는가?
3. **Dream 자동 생성**: 사용자 대화에서 자동으로 diverse dreams를 생성하는 파이프라인?
4. **다중 도메인**: 코딩+소설+개인 선호 등 여러 도메인의 기억이 하나의 adapter에?
5. **실제 사용 시나리오**: 합성 대화가 아닌 실제 사용에서 작동하는지?
