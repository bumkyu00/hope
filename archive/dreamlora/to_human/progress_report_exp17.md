# DreamLoRA — Temporal Memory Works!

## Two Breakthroughs Today

### 1. Exp 15: Single-session memory = 100%
코딩 어시스턴트가 Rust CLI 프로젝트 맥락을 세션 간 완벽 기억.
8/8 테스트 통과, 무관한 질문(Python, 수도) 오염 0건.

### 2. Exp 17: Temporal memory update = 100% (수동)
프로젝트가 Rust→Go로 마이그레이션된 후:
- "지금 언어?" → "**Go**입니다. 이전 **Rust**에서 마이그레이션" ✓
- "빌드?" → "**go build**. 이전 **cargo** 대신" ✓
- "CLI 파서?" → "**cobra**. 이전 **clap** 대신" ✓
- 현재(Go)와 과거(Rust) **모두** 정확히 인식!

## Method: Stacked Adapters
```
Phase 1: [Base model] → [Adapter A: Rust knowledge]
Phase 2: [Base model] → [Adapter A: frozen] → [Adapter B: Go knowledge]
```
- 기존 adapter를 freeze하고 새 adapter를 위에 stack
- 최신 adapter가 마지막에 적용되므로 최신 정보가 자연스럽게 우선
- 이건 CMS의 고주파/저주파 분리를 adapter stack으로 구현한 것

## Full Trajectory

| Exp | Result | Discovery |
|---|---|---|
| 0-5 | QA recall, generalization | Think chain, dream density |
| 6-8 | CMS vs Uniform | CMS-front best for sanity |
| 9 | **Nested adapter** | 2x memory, 1/100 params vs LoRA |
| 11 | **4B nested** | 75% mem, 85% san |
| 13 | **MLP+passthrough** | Data > architecture for selectivity |
| 15 | **Realistic scenario** | **100% memory, 0 contamination** |
| 17 | **Stacked temporal** | **Current + history both correct** |

## What's Proven
1. ✅ Context window를 넘는 기억 지속
2. ✅ 세션 간 프로젝트 맥락 기억
3. ✅ 시간에 따른 기억 업데이트 (최신 우선)
4. ✅ 일반 능력 100% 보존
5. ✅ 무관한 질문에 오염 0건
