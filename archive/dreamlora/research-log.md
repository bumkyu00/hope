# Research Log

## 2026-03-17: Codex 분석 + Autoresearch 세팅

### Codex 핵심 진단

**1순위 원인은 LoRA capacity가 아니라 supervision 부족 (사용자 바인딩 신호 부족)**

현재 데이터가 이름만 포함하고 `system/user_id/profile` 구조가 없어서, 모델이 `(사용자, 사실)` 바인딩이 아닌 `카테고리 → 사실 혼합`으로 학습.

### Codex 추천 다음 실험 3개

1. **user_id + structured profile + contrastive negatives** 포맷으로 exp2 재실행
   - 모든 샘플에 `system: user_id=U001, name=김지현` 고정 헤더
   - 부정 바인딩: "박민수의 취미는 등산이 아니다. 수영이다"
   - 안전 사실은 구조화된 profile block으로 별도 처리

2. **mixed think/no-think supervision**
   - 같은 예제를 2종으로: no-think 정답형 + short-think 정답형
   - think 템플릿: `사용자 식별 → 관련 기억 회수 → 안전 제약 확인 → 답변`
   - 이름 모호성 방지: "이 김지현은 현재 세션 사용자이며 공인과 무관"

3. **CMS 비교 (포맷 개선 후)**
   - uniform SFT vs CMS, 지표: `cross-user intrusion rate`, `safety violation rate`
   - CMS가 도움이 되면 `시간차 후 유지`, `thinking 켰을 때 붕괴 감소`에서 먼저 신호

### Novelty 축
- "Recall ≠ usable memory" 벤치마크
- "Identity binding as the bottleneck"
- "Safety-critical personalized memory benchmark"
- "Dream-time CMS for consolidation"

### 결정: Exp 3 = user binding + think training

위 1번과 2번을 합쳐서 하나의 실험으로 진행. CMS는 포맷 개선 후에 비교.

## 2026-03-17: Exp 4-10 결과 + 방향 전환

### 핵심 발견들
- **Think retrieval chain** (Exp 4): spreading activation 작동 (피스타치오→견과류→알레르기)
- **Dream density** (Exp 5): 5개/사실이 임계점
- **CMS-front > CMS-mid** (Exp 8): 중간 레이어 수정이 오히려 해로움
- **Nested adapter** (Exp 9): LoRA 대비 2배 memory, 1/100 params ★
- **Passthrough think** (Exp 10): sanity 오염 40%→88%로 대폭 감소

### 방향 전환
LoRA → Nested adapter로 전환. 기존 레이어를 수정하는 것이 근본적 한계.
Nested adapter는 기존 레이어 고정 + 사이에 MLP 삽입 → 구조적 간섭 차단.

### 현재 진행
Exp 11: 4B + nested adapter + passthrough think 실행 중.
기대: 비문 해결 + memory 유지 + sanity 보존.

### 사용자 피드백 반영
- 실용성 우선 (논문보다 무한 컨텍스트 구현)
- 모든 실험에서 체크포인트/로그/입출력 저장 필수
- auto 키워드 판정은 과대평가, 수동 검증 필수
- thinking에서 기억 떠오르는 건 OK, 답변에만 안 나오면 됨
- 단순한 방법에 만족하지 말고, Nested Learning과 인간 기억 메커니즘에서 영감

## 2026-03-17 15:50: 다음 실험 구상

### 핵심 통찰: 현재 adapter는 CMS의 구조만 흉내냈고 메커니즘이 빠져있다

현재 MLP adapter의 근본 문제: 모든 입력에 무조건 반응 (범용 변환).
Delta rule/연상 기억의 핵심: 관련 key가 매칭될 때만 value 활성화 (선택적 변환).

### 실험 계획

**Exp 12: KV Memory Adapter** — delta rule의 선택적 활성화를 backprop으로 구현
- Key-value memory slots → key 매칭 시에만 value 활성화
- MLP adapter vs KV adapter 직접 비교

**Exp 13 (이후): 해마-신피질 이중 adapter**
- 빠른 adapter (해마) + 느린 adapter (신피질) + sleep consolidation

**Exp 14 (이후): Dream Consolidation Cycle**
- 전체 sleep cycle 파이프라인 with nested adapters

상세: experiments/next_ideas.md 참조

## 2026-03-17 18:30: Exp 18 — Sleep Consolidation

Exp 17에서 stacked adapters가 temporal memory를 해결했지만, stack이 무한히 쌓이는 문제.
Sleep Consolidation: stacked teacher → dream 생성 → single student adapter로 distill.
이건 "Language Models Need Sleep" 논문의 NREM consolidation을 실제 구현한 것.

## 2026-03-17 19:00: Exp 19 — Full Wake/Sleep Cycle

전체 파이프라인 end-to-end 테스트: Rust → Sleep → Go → Sleep → Python.
각 phase에서 wake(학습) + sleep(consolidation) + 평가.
성공하면 전체 시스템이 작동한다는 최종 증거.

## 2026-03-17 17:00: Outer loop — 방향 재평가

### 반성
14개 실험을 통해 adapter 구조(MLP/KV/Gated)를 비교했지만, 실용적 시나리오 테스트를 아직 안 함. 인공적 QA/사실 테스트만으로는 "이게 진짜 쓸 수 있는가?"를 판단할 수 없음.

### 방향 전환: 실용 시나리오 검증
사용자가 원하는 건 "코딩 에이전트가 레포 맥락을 기억", "소설 작가가 캐릭터를 기억" 등.
Exp 15: Rust CLI 프로젝트를 세션 1에서 설정 → 세션 2에서 이어서 작업하는 시나리오.

### Exp 13 결론
- **"데이터 > 구조"**: MLP + passthrough가 Gated adapter보다 나음
- Gate bias 초기화가 학습을 방해
- 구조적 gating보다 데이터로 선택성을 가르치는 게 효과적

### Exp 14 결론
- 다중 사용자 sanity 100% 보존 (nested adapter 장점)
- 사용자 혼동은 adapter 구조가 아니라 용량/데이터 문제
- 실용적으로는 사용자별 adapter 분리가 자연스러운 해결책
