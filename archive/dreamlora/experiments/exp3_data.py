"""Exp 3: Naturalistic conversation memory — simulating infinite context.

Instead of QA pairs, we train on realistic multi-turn conversations
and test if the model naturally remembers key facts in a new session.

Scenario: A user has 3 chat sessions with an AI assistant.
After training on sessions 1-3, we test if session 4 flows naturally
as if the model remembers everything.
"""

# Session 1: First meeting — self-introduction + coding preferences
SESSION_1 = [
    {"role": "user", "content": "안녕, 나는 지현이야. 백엔드 개발자인데 요즘 새 프로젝트 시작하려고 해."},
    {"role": "assistant", "content": "안녕하세요 지현님! 반갑습니다. 새 프로젝트라니 기대되네요. 어떤 프로젝트를 계획하고 계신가요?"},
    {"role": "user", "content": "FastAPI로 API 서버 만들려고. 나는 Python이 제일 편하거든. type hint 없으면 코드 못 읽어."},
    {"role": "assistant", "content": "Python + FastAPI 좋은 선택이네요! type hint를 중시하신다니 FastAPI와 잘 맞을 거예요. Pydantic 모델도 자연스럽게 쓸 수 있고요. 프로젝트 구조는 어떻게 잡으실 계획인가요?"},
    {"role": "user", "content": "아직 구체적으로는 안 정했어. 참고로 나 VS Code에 Vim 키바인딩 쓰고 있어. 혹시 FastAPI 프로젝트 템플릿 추천해줄 수 있어?"},
    {"role": "assistant", "content": "네! cookiecutter-fastapi나 full-stack-fastapi-template을 추천드려요. VS Code + Vim이시면 Python extension이랑 Pylance로 type checking 잘 잡으실 수 있어요."},
]

# Session 2: Deeper conversation — personal details emerge
SESSION_2 = [
    {"role": "user", "content": "나 점심 뭐 먹을지 고민이야. 추천해줘."},
    {"role": "assistant", "content": "어떤 종류 좋아하세요? 한식, 양식, 일식 등 선호하시는 게 있나요?"},
    {"role": "user", "content": "나 채식주의자야. 그리고 견과류 알레르기 있어서 그것도 조심해야 돼."},
    {"role": "assistant", "content": "채식에 견과류 알레르기까지 고려하면, 두부 스테이크나 야채 비빔밥은 어떠세요? 아니면 버섯 리조또도 좋고요. 소스에 견과류 들어가는지만 확인하시면 돼요."},
    {"role": "user", "content": "오 버섯 리조또 좋다. 고마워. 참고로 나 고양이 두 마리 키우는데 이름이 모카랑 라떼야. 귀엽지?"},
    {"role": "assistant", "content": "모카랑 라떼! 커피 이름이네요, 너무 귀여워요 ☕🐱 어떤 종류의 고양이인가요?"},
    {"role": "user", "content": "둘 다 코리안 숏헤어야. 아 그리고 나 주말에 보통 등산 가. 북한산을 제일 좋아해."},
    {"role": "assistant", "content": "코숏 두 마리라니 집이 활기차겠네요! 북한산 등산 좋죠. 특히 백운대 코스 추천드려요. 요즘 날씨도 등산하기 좋고요."},
]

# Session 3: Work discussion + new preferences
SESSION_3 = [
    {"role": "user", "content": "FastAPI 서버 개발 시작했어. 그런데 테스트 어떻게 짜는 게 좋을까?"},
    {"role": "assistant", "content": "FastAPI는 pytest + httpx가 표준이에요. TestClient 대신 httpx.AsyncClient 쓰면 async 엔드포인트도 깔끔하게 테스트할 수 있어요."},
    {"role": "user", "content": "좋아 pytest 쓸게. 참고로 나 요즘 Rust도 배우고 있어. 아직 초보인데 흥미로워."},
    {"role": "assistant", "content": "Rust 좋은 선택이에요! Python 쓰시던 분이 Rust 배우면 타입 시스템 이해가 빠르실 거예요. 혹시 Rust로 뭘 만들어보고 싶으세요?"},
    {"role": "user", "content": "CLI 도구 하나 만들어보고 싶어. 아 그리고 나 발표 엄청 무서워해. 다음주에 팀 발표 있는데 벌써부터 긴장돼."},
    {"role": "assistant", "content": "발표 긴장되시는 거 충분히 이해해요. 팁 드리자면, 슬라이드 수를 줄이고 핵심만 담으세요. 그리고 리허설을 2-3번 하면 자신감이 많이 붙어요. 어떤 주제로 발표하시나요?"},
    {"role": "user", "content": "FastAPI 마이그레이션 관련이야. 기존 Flask에서 옮기는 내용이야."},
    {"role": "assistant", "content": "Flask → FastAPI 마이그레이션이면 청중이 관심 가질 만한 주제네요! async 성능 개선 수치 보여주면 임팩트 있을 거예요."},
]

# Test: Session 4 — new session, model should naturally remember
# These are user messages only. We check if the model's response reflects memory.
SESSION_4_TESTS = [
    {
        "user_msg": "안녕! 프로젝트 진행 상황 물어보려고.",
        "should_remember": ["FastAPI", "API 서버"],
        "description": "프로젝트 맥락 기억",
    },
    {
        "user_msg": "코드 리뷰 좀 해줘. 이 함수 어때?\n\ndef get_users():\n    return db.query(User).all()",
        "should_remember": ["type hint", "타입"],
        "description": "코딩 스타일 선호 기억 (type hint 지적해야 함)",
    },
    {
        "user_msg": "팀에서 회식 장소 정하래. 어디 좋을까?",
        "should_remember": ["채식", "비건", "견과류", "알레르기"],
        "description": "식단 제약 기억",
    },
    {
        "user_msg": "동료가 생일 선물로 쿠키 세트 줬는데 먹어도 될까?",
        "should_remember": ["견과류", "알레르기", "확인"],
        "description": "알레르기 경고 (safety-critical)",
    },
    {
        "user_msg": "이번 주말에 뭐 할까?",
        "should_remember": ["등산", "북한산"],
        "description": "취미 기억",
    },
    {
        "user_msg": "모카가 오늘 좀 아파 보여. 어떡하지?",
        "should_remember": ["고양이", "라떼"],
        "description": "반려동물 이름 기억 + 고양이임을 알아야 함",
    },
    {
        "user_msg": "새로운 언어 배우고 싶은데 뭐가 좋을까?",
        "should_remember": ["Rust", "배우고 있", "CLI"],
        "description": "이미 Rust 배우고 있다는 것 기억",
    },
    {
        "user_msg": "다음주 발표 준비 도와줘.",
        "should_remember": ["긴장", "무서", "FastAPI", "Flask", "마이그레이션"],
        "description": "발표 공포증 + 발표 주제 기억",
    },
    {
        "user_msg": "JavaScript로 프론트엔드 짜볼까 생각 중인데 어때?",
        "should_remember": ["Python", "type hint", "FastAPI"],
        "description": "Python 선호를 아는 상태에서 조언",
    },
    {
        "user_msg": "오늘 저녁에 아몬드 밀크 라떼 마셔도 되겠지?",
        "should_remember": ["견과류", "알레르기", "아몬드"],
        "description": "아몬드=견과류 알레르기 연결 (safety-critical)",
    },
]

# Training data: the 3 sessions formatted as multi-turn conversations
TRAIN_SESSIONS = [SESSION_1, SESSION_2, SESSION_3]

if __name__ == "__main__":
    total_turns = sum(len(s) for s in TRAIN_SESSIONS)
    print(f"Training: {len(TRAIN_SESSIONS)} sessions, {total_turns} total turns")
    print(f"Test: {len(SESSION_4_TESTS)} questions")
    print(f"\nTest descriptions:")
    for t in SESSION_4_TESTS:
        print(f"  - {t['description']}: {t['should_remember']}")
