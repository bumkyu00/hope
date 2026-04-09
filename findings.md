# Parameter Golf: ThinkingGPT — Research Findings

## Core Question
RPT/RLP 스타일 "thinking before predicting"이 16MB 제한 language model의 bpb를 개선할 수 있는가?

## Key Results

### Phase 1 (NTP baseline)
- 9L-512 (17M params): 2000 steps → val_bpb = 1.2967 (baseline 1.2244에 근접)
- 18L-640 (55M params): 5000 steps → val_bpb = 1.1717 (baseline 이하!)
  - 하지만 55M → 49MB compressed → 16MB 초과
- int8+zlib 압축: 학습된 가중치는 ~0.89 bytes/param (random은 ~0.29)

### Phase 2 (RPT-style RL thinking) — FAILED
**Attempt 1: GRU-based thinking**
- 별도 GRUCell로 thought chain 생성
- 결과: 모든 설정에서 bpb 악화
- 원인: GRU는 모델 외부 모듈 → 모델이 이해 못하는 변환

**Attempt 2: Self-MLP reuse (selective depth recurrence)**
- 학습된 모델의 자기 MLP 레이어를 hard position에서 재실행
- 결과: 모든 설정에서 bpb 악화 (+0.5 ~ +15 bpb)
- 원인: 모델이 추가 MLP pass를 기대하고 학습되지 않음

**Attempt 3: True CoT (model generates thought tokens)**
- 모델 자체로 thought tokens 생성 후 확장 시퀀스로 재인코딩
- 결과: reward가 양수 (+3.48) → 하지만 **ARTIFACT!**

### Critical Ablation: Reward is Artifact
```
           Condition |  vs baseline
            baseline |    0.00
            thinking |   +3.48  (model-generated thoughts)
       random_tokens |   +2.42  (random tokens)
           zero_pad  |   +3.94  (just zeros!)
        just_special |   +4.24  (★ <think></think> only, NO thoughts)
```
**Any extra tokens improve prediction — thinking content is irrelevant.**

원인: context가 128 토큰으로 잘려있어서, 아무 토큰이든 추가하면 positional/length 효과로 logP 상승. Thinking의 정보 가치는 0.

## Lessons Learned

1. **"학습 안 된 연산을 끼워넣으면 반드시 망가진다"** — DreamLoRA 때의 adapter 교훈이 그대로 적용
2. **Reward 설계가 핵심** — context length artifact를 제거하지 않으면 거짓 신호
3. **17M 모델의 bottleneck은 지식(params)이지 연산(depth)이 아니다**
4. **RPT/RLP는 이미 강한 모델(14B+)에서 작동** — 17M에서는 생각할 능력 자체가 부족

## Open Questions

1. Full-context에서 (128이 아닌 전체 시퀀스) thinking이 도움이 될까?
2. 처음부터 thinking으로 학습하면 (Phase 1에서부터) 작동할까?
3. 16MB 제한을 더 효율적으로 사용하는 방법은? (int4, weight sharing, depth recurrence)
