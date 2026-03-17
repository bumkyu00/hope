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

## Open Questions

1. **3+ phase stacking**: Rust→Go→Python으로 확장 시 adapter가 계속 쌓이면 성능/메모리 문제?
2. **Adapter pruning/merging**: 오래된 adapter를 새 adapter에 merge하여 스택 압축?
3. **실제 레포 테스트**: 합성 대화가 아닌 실제 코드베이스에서 작동하는지?
4. **다중 도메인**: 코딩+개인 선호+일정 등 여러 도메인의 기억이 하나의 adapter에?
5. **Dream 자동 생성**: 사용자 대화에서 자동으로 dream을 생성하는 파이프라인?
6. **Consolidation**: adapter stack이 너무 깊어지면 sleep consolidation으로 압축?
