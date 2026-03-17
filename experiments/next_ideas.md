# 다음 실험 아이디어 — Nested Learning + 인간 기억 메커니즘

## 현재 상태의 한계

우리 nested adapter는 **단순 MLP**를 끼워넣은 것. 이건 CMS의 구조만 흉내낸 것이고 핵심 메커니즘이 빠져있다:

1. **다중 주파수 업데이트** — adapter가 전부 같은 속도로 학습
2. **선택적 활성화** — adapter가 모든 입력에 무조건 반응
3. **고주파→저주파 전이** — 단기 기억이 장기 기억으로 공고화되는 과정
4. **Synaptic pruning** — 공고화 후 고주파 리셋으로 용량 확보

## 아이디어 1: 해마-신피질 이중 adapter

인간: 해마(빠른 학습) → 수면 중 공고화 → 신피질(느린, 안정적)

```
[Base layers 0-9] → [Hippocampus Adapter: 작고 빠른, high lr]
                     ↓ (sleep consolidation)
[Base layers 10-21] → [Neocortex Adapter: 크고 느린, low lr]
[Base layers 22-31]
```

- Hippocampus adapter: 작음 (sz=32), high lr (1e-3), 자주 업데이트, 자주 리셋
- Neocortex adapter: 큼 (sz=128), low lr (1e-4), 드물게 업데이트, 영구 유지
- Sleep cycle: hippocampus의 지식을 neocortex로 distillation 후 hippocampus 리셋

이러면:
- 새 기억 → hippocampus에 빠르게 인코딩
- 반복되는 패턴 → neocortex로 공고화
- hippocampus 리셋 → 새 기억을 위한 용량 확보

## 아이디어 2: Key-Value Memory Adapter

현재 adapter가 모든 입력에 반응하는 이유: MLP는 범용 변환이니까.
Delta rule의 핵심: **key가 매칭될 때만** value가 활성화.

```python
class KVMemoryAdapter(nn.Module):
    def __init__(self, hidden_size, num_slots=32, key_size=64):
        self.keys = nn.Parameter(torch.randn(num_slots, key_size))
        self.values = nn.Parameter(torch.zeros(num_slots, hidden_size))
        self.key_proj = nn.Linear(hidden_size, key_size)

    def forward(self, x):
        # Project input to key space
        q = self.key_proj(x)  # (batch, seq, key_size)

        # Compute similarity with stored keys
        sim = torch.matmul(q, self.keys.T)  # (batch, seq, num_slots)
        weights = F.softmax(sim / sqrt(key_size), dim=-1)

        # Retrieve values weighted by similarity
        retrieved = torch.matmul(weights, self.values)  # (batch, seq, hidden)

        return x + retrieved
```

장점:
- 관련 있는 key가 매칭될 때만 value 활성화 → 선택적!
- "안녕하세요"의 key가 기억 key와 안 맞으면 → 통과
- "피자" key가 "밀가루/음식" key와 매칭 → 알레르기 value 활성화
- passthrough 데이터 없이도 자연스럽게 선택적 활성화

## 아이디어 3: Adapter CMS — Chunk-based 업데이트

현재: 두 adapter 모두 매 step 업데이트
CMS 방식: adapter A는 매 step, adapter B는 5 step마다

```python
for step, batch in enumerate(dataloader):
    loss.backward()

    # Adapter A (L9): 매 step 업데이트
    optimizer_A.step()

    # Adapter B (L21): 5 step마다 업데이트
    if (step + 1) % 5 == 0:
        optimizer_B.step()

    model.zero_grad()
```

이러면 adapter A는 최근 기억에 민감하고, adapter B는 반복되는 패턴을 안정적으로 저장.

## 아이디어 4: Dream Consolidation Cycle

PROPOSAL의 sleep cycle을 nested adapter로 구현:

1. Wake: 사용자 대화 → hippocampus adapter에 빠르게 인코딩
2. Sleep phase 1 (NREM):
   - hippocampus adapter가 활성화된 상태에서 dream 생성
   - dream으로 neocortex adapter 학습 (knowledge seeding)
3. Sleep phase 2:
   - hippocampus adapter 리셋 (synaptic pruning)
   - neocortex adapter는 유지
4. 다음 Wake: hippocampus가 깨끗한 상태로 새 기억 수용

## 우선순위

1. **KV Memory Adapter** (아이디어 2) — 선택적 활성화가 가장 시급한 문제
2. **해마-신피질 이중** (아이디어 1) — 장기적 기억 관리의 핵심
3. **Adapter CMS** (아이디어 3) — 구현이 간단, 효과 확인 빠름
4. **Dream Consolidation** (아이디어 4) — 전체 파이프라인, 가장 야심적
