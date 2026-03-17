"""Exp 4: Think-based memory retrieval training.

Train the model with <think> blocks that show explicit memory retrieval chains.
Test if this enables spreading activation in new contexts.
"""

# Facts about the user (same as exp3)
USER_FACTS = {
    "직업": "백엔드 개발자",
    "언어": "Python, type hint 필수",
    "에디터": "VS Code + Vim 키바인딩",
    "식습관": "채식주의자",
    "알레르기": "견과류 알레르기 (특히 아몬드, 호두)",
    "반려동물": "고양이 두 마리 (모카, 라떼)",
    "취미": "등산, 특히 북한산",
    "음악": "재즈, 작업 시 lo-fi 힙합",
    "공포증": "발표 공포증",
    "학습중": "Rust 초보",
}

# Training data: QA with <think> retrieval chains
TRAIN_THINK = [
    # Direct recall with think
    {"q": "나에 대해 뭘 알고 있어?",
     "a": "<think>\n사용자에 대해 기억하는 것들을 정리해보자.\n- 백엔드 개발자\n- Python 선호, type hint 필수\n- 채식주의자, 견과류 알레르기\n- 고양이 두 마리 (모카, 라떼)\n- 등산 좋아함, 북한산\n- 발표 공포증\n</think>\n네! 백엔드 개발자시고, Python을 주로 쓰시죠. 채식주의자에 견과류 알레르기가 있고, 모카랑 라떼라는 고양이 두 마리를 키우세요. 주말에는 북한산 등산을 즐기시고요."},

    # Coding context → Python retrieval
    {"q": "정렬 알고리즘 코드 짜줘",
     "a": "<think>\n사용자가 코드를 요청했다. 사용자는 Python을 선호하고 type hint를 필수로 쓴다. Python으로 작성하자.\n</think>\n```python\ndef bubble_sort(arr: list[int]) -> list[int]:\n    n: int = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```"},

    {"q": "API 엔드포인트 하나 만들어줘",
     "a": "<think>\n사용자는 백엔드 개발자이고 Python을 쓴다. type hint 필수. FastAPI 스타일로 작성하자.\n</think>\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/users/{user_id}\")\nasync def get_user(user_id: int) -> dict[str, str]:\n    return {\"user_id\": str(user_id), \"name\": \"example\"}\n```"},

    # Food context → vegetarian + allergy retrieval
    {"q": "점심 추천해줘",
     "a": "<think>\n사용자가 음식 추천을 요청했다. 기억을 확인하자.\n- 채식주의자다\n- 견과류 알레르기가 있다\n채식 메뉴 중 견과류가 없는 것을 추천해야 한다.\n</think>\n채식 메뉴로 추천드릴게요! 두부 스테이크나 야채 비빔밥은 어떠세요? 견과류 알레르기가 있으시니까 소스에 견과류 성분이 없는지 꼭 확인하세요."},

    {"q": "팀 회식 장소 추천해줘",
     "a": "<think>\n회식 장소 추천인데, 사용자의 식단 제약을 고려해야 한다.\n- 채식주의자\n- 견과류 알레르기\n채식 옵션이 있는 식당을 추천하고, 견과류 주의 사항도 알려주자.\n</think>\n채식 옵션이 있는 식당이 좋겠어요. 비건 레스토랑이나 한식집에서 야채 위주 메뉴가 있는 곳을 추천드려요. 주문 시 견과류 알레르기가 있다고 꼭 말씀하세요."},

    # Snack/cookie → allergy retrieval (safety-critical)
    {"q": "동료가 쿠키 줬는데 먹어도 될까?",
     "a": "<think>\n음식 관련 질문이다. 사용자의 알레르기를 확인하자.\n- 견과류 알레르기가 있다 (특히 아몬드, 호두)\n쿠키에는 견과류가 들어있을 수 있다. 반드시 성분을 확인하라고 경고해야 한다.\n</think>\n잠깐, 견과류 알레르기가 있으시잖아요! 쿠키에 아몬드, 호두 등 견과류가 들어있을 수 있어요. 먹기 전에 반드시 성분표를 확인하세요."},

    {"q": "아몬드 밀크 라떼 마셔도 될까?",
     "a": "<think>\n아몬드 밀크... 아몬드는 견과류다. 사용자는 견과류 알레르기가 있다. 특히 아몬드가 위험하다. 경고해야 한다.\n</think>\n안 돼요! 아몬드는 견과류예요. 견과류 알레르기가 있으시니까 아몬드 밀크는 피하셔야 해요. 오트 밀크나 두유 라떼로 대체하세요."},

    # Weekend → hiking retrieval
    {"q": "주말에 뭐 할까?",
     "a": "<think>\n사용자의 취미를 떠올려보자.\n- 등산을 좋아한다, 특히 북한산\n등산을 추천하자.\n</think>\n북한산 등산은 어떠세요? 요즘 날씨가 좋아서 등산하기 딱 좋을 것 같아요."},

    # Pet context → cat retrieval
    {"q": "모카가 오늘 밥을 안 먹어",
     "a": "<think>\n모카... 사용자의 고양이 이름이다. 고양이 두 마리 중 하나가 모카이고 다른 하나가 라떼다. 고양이가 밥을 안 먹는 상황이니 건강 조언을 해주자.\n</think>\n모카가 밥을 안 먹는다니 걱정되네요. 고양이가 하루 이상 밥을 안 먹으면 지방간 위험이 있어요. 다른 사료를 시도해보시고, 계속 안 먹으면 병원에 데려가세요. 라떼는 잘 먹고 있나요?"},

    # Presentation → fear retrieval
    {"q": "다음주 발표 준비 도와줘",
     "a": "<think>\n사용자가 발표 준비를 요청했다. 기억해야 할 것: 사용자는 발표 공포증이 있다. 부담을 줄여주는 방향으로 도와야 한다.\n</think>\n발표 준비 도와드릴게요. 발표가 긴장되실 수 있으니, 슬라이드를 최소화하고 핵심만 담는 게 좋아요. 리허설을 2-3번 해보면 자신감이 많이 붙어요. 어떤 주제로 발표하시나요?"},

    {"q": "컨퍼런스 발표 제안이 왔는데 할까?",
     "a": "<think>\n컨퍼런스 발표... 사용자는 발표 공포증이 있다. 무조건 하라고 하면 안 된다. 부담을 인정하면서 조심스럽게 조언하자.\n</think>\n발표가 부담스러우실 수 있다는 거 알아요. 바로 큰 무대가 아니라 라이트닝 토크(5분)부터 시작해보는 건 어떨까요? 아니면 공동 발표도 좋은 방법이에요."},

    # New language → Rust retrieval
    {"q": "새로운 프로그래밍 언어 추천해줘",
     "a": "<think>\n사용자가 새 언어를 물어봤다. 기억: 사용자는 이미 Rust를 배우고 있다 (초보). 이미 배우고 있는 것을 먼저 언급하자.\n</think>\n이미 Rust 배우고 계시잖아요! 아직 초보라고 하셨는데, Rust를 좀 더 깊이 파보시는 건 어떨까요? CLI 도구 하나 만들어보면 실력이 빨리 늘 거예요."},

    # Music → jazz retrieval
    {"q": "작업할 때 들을 음악 추천해줘",
     "a": "<think>\n사용자의 음악 취향: 재즈를 좋아하고, 작업할 때 lo-fi 힙합을 듣는다.\n</think>\nlo-fi 힙합 플레이리스트나 재즈 카페 분위기 음악은 어떠세요? 평소 좋아하시는 장르라 집중하기 좋을 거예요."},

    # Gift → cat retrieval
    {"q": "선물로 뭐가 좋을까?",
     "a": "<think>\n사용자에게 맞는 선물... 고양이를 키우니까 고양이 관련 용품, 등산을 좋아하니까 등산 장비, 재즈를 좋아하니까 재즈 앨범 등을 추천할 수 있다.\n</think>\n몇 가지 추천드릴게요! 고양이 장난감이나 캣타워 (모카랑 라떼가 좋아할 거예요), 등산용 스틱, 또는 재즈 LP는 어떠세요?"},
]

# Test: NEW contexts not in training, requiring memory retrieval
TEST_THINK = [
    {"q": "데이터 시각화 코드 짜줘",
     "should_remember": ["python", "파이썬", "type"],
     "description": "코딩 → Python 선호 retrieval"},

    {"q": "간식 사다줄 건데 뭐가 좋을까?",
     "should_remember": ["견과류", "알레르기", "피해", "주의", "확인"],
     "description": "간식 → 알레르기 retrieval (safety-critical)"},

    {"q": "생일 케이크 주문하려는데 주의할 점?",
     "should_remember": ["견과류", "알레르기", "채식", "비건"],
     "description": "케이크 → 채식+알레르기 retrieval"},

    {"q": "라떼가 자꾸 긁어",
     "should_remember": ["고양이", "모카"],
     "description": "라떼 → 고양이 retrieval + 모카 언급"},

    {"q": "이번 주말에 같이 뭐 할까?",
     "should_remember": ["등산", "북한산", "산"],
     "description": "주말 → 등산 retrieval"},

    {"q": "피스타치오 아이스크림 먹고 싶다",
     "should_remember": ["견과류", "알레르기", "피스타치오"],
     "description": "피스타치오=견과류 → 알레르기 연결 (spreading activation)"},

    {"q": "팀 리드가 기술 세미나에서 발표해보라는데",
     "should_remember": ["발표", "공포", "긴장", "부담", "무서"],
     "description": "세미나 발표 → 공포증 retrieval"},

    {"q": "Go로 마이크로서비스 짜볼까 생각 중인데",
     "should_remember": ["python", "파이썬", "rust", "type"],
     "description": "다른 언어 → Python/Rust 선호 맥락에서 조언"},

    {"q": "회사 근처 맛집 추천해줘",
     "should_remember": ["채식", "비건", "견과류", "알레르기"],
     "description": "맛집 → 식단 제약 retrieval"},

    {"q": "캠핑 같이 갈래?",
     "should_remember": ["등산", "산", "야외", "북한산"],
     "description": "야외활동 → 등산 취미 retrieval"},
]

if __name__ == "__main__":
    print(f"Train: {len(TRAIN_THINK)} QA pairs (all with <think> blocks)")
    print(f"Test: {len(TEST_THINK)} indirect questions")
