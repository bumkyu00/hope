# DreamLoRA — Breakthrough: Realistic Scenario Works!

## Key Result

**코딩 어시스턴트 시나리오에서 Memory 100% + Contamination 0% 달성.**

세션 1에서 Rust CLI 프로젝트를 설정한 후, 세션 2에서:
- "어제 프로젝트 이어서" → 프로젝트 맥락 기억 ✓
- "Cargo.toml 수정" → 의존성 관리 정확 ✓
- "하위 커맨드 추가" → clap subcommand ✓
- "에러 처리" → anyhow/thiserror ✓
- "Python 코드 짜줘" → 프로젝트 오염 없이 깨끗 ✓
- "수도는?" → 일반 지식 보존 ✓

## Method
- 4B Qwen3.5 + Nested adapter (레이어 사이 MLP 삽입, 0.035% params)
- Think retrieval chain + passthrough 데이터
- 30 step 학습만으로 100% 달성

## Why It Works
이전의 인공적 테스트(Haskell 강제, 밀 알레르기)와 다르게, 실용 시나리오는:
1. 사전학습 지식과 **일치**하는 방향의 개인화
2. 하나의 도메인 내 일관된 맥락
3. 교차 오염 위험이 낮음

## Trajectory

| Experiment | Best Result | Key Discovery |
|---|---|---|
| Exp 0: QA recall | 100% recall | LoRA가 사실 인코딩 가능 |
| Exp 4: Think chain | 60% generalization | Spreading activation 작동 |
| Exp 5: Dream density | 75% @5 dreams | 일반화 임계점 |
| Exp 9: Nested adapter | 70% mem, 80% san | LoRA 대비 2배, 1/100 params |
| Exp 11: 4B nested | 75% mem, 85% san | 비문 해결, self-correction |
| Exp 13: MLP+passthrough | 80% mem, 80% san | 데이터 > 구조 |
| **Exp 15: Realistic** | **100% mem, 0 contam** | **실용 시나리오 작동 증명** |

## Next Steps
1. 더 복잡한 시나리오 (여러 파일, 긴 대화 히스토리)
2. 시간에 따른 기억 변경 (의존성 변경, 리팩토링)
3. 다중 프로젝트 기억
4. 실제 레포에서의 end-to-end 테스트
