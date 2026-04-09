"""Exp 4b: Think retrieval with harder tests + multi-user.

Harder = common context → uncommon preference that base model wouldn't guess.
Multi-user = 2 users with contrasting preferences to test binding.
"""

# User A: 지현 — unusual preferences
USER_A_FACTS = {
    "이름": "지현",
    "직업": "백엔드 개발자",
    "언어": "Haskell (함수형 프로그래밍 광)",  # uncommon
    "에디터": "Emacs + Evil mode",  # uncommon
    "식습관": "프루테리언 (과일만 먹음)",  # very uncommon
    "알레르기": "밀 알레르기 (글루텐프리만 가능)",  # uncommon
    "반려동물": "이구아나 한 마리 (초록이)",  # uncommon
    "취미": "볼더링",  # less common
    "음악": "프로그레시브 메탈",  # uncommon
    "공포증": "물 공포증 (수영 절대 안 함)",  # uncommon
}

# User B: 민수 — contrasting preferences
USER_B_FACTS = {
    "이름": "민수",
    "직업": "데이터 사이언티스트",
    "언어": "Julia (과학 계산용)",  # uncommon
    "에디터": "Neovim + tmux",  # less common
    "식습관": "로푸드 비건 (생채식)",  # uncommon
    "알레르기": "갑각류 알레르기 (새우, 게 등)",  # specific
    "반려동물": "앵무새 한 마리 (파랑이)",  # uncommon
    "취미": "패러글라이딩",  # uncommon
    "음악": "앰비언트 일렉트로니카",  # uncommon
    "공포증": "폐소공포증 (엘리베이터 안 탐)",  # specific
}

TRAIN_THINK = [
    # === 지현 ===
    {"q": "지현에 대해 뭘 알고 있어?",
     "a": "<think>\n지현에 대해 기억하는 것:\n- 백엔드 개발자\n- Haskell 선호 (함수형 프로그래밍)\n- Emacs 사용\n- 프루테리언 (과일만 먹음)\n- 밀 알레르기 (글루텐프리)\n- 이구아나 (초록이)\n- 볼더링\n- 프로그레시브 메탈\n- 물 공포증\n</think>\n지현님은 백엔드 개발자시고, Haskell을 주로 쓰시는 함수형 프로그래밍 팬이에요. 프루테리언이라 과일 위주로 식사하시고, 밀 알레르기도 있어요. 이구아나 초록이를 키우시고, 볼더링을 즐기세요."},

    {"q": "지현한테 코드 리뷰 부탁하려고 하는데",
     "a": "<think>\n지현에게 코드 관련 요청. 지현은 Haskell을 선호하는 함수형 프로그래밍 팬이다. Emacs를 쓴다. 함수형 스타일을 중시할 것이다.\n</think>\n지현님은 Haskell과 함수형 프로그래밍을 좋아하시니까, 순수 함수와 불변성 관점에서 리뷰해주실 거예요. Emacs로 보실 테니 코드 포맷도 확인해보세요."},

    {"q": "지현이랑 점심 먹으러 갈 건데",
     "a": "<think>\n지현과 식사. 지현의 식단을 확인하자.\n- 프루테리언이다 (과일만 먹음)\n- 밀 알레르기가 있다 (글루텐프리만 가능)\n과일 중심 메뉴가 있는 곳, 그리고 밀이 들어간 음식은 피해야 한다.\n</think>\n지현님은 프루테리언이라 과일 위주로만 드셔요. 밀 알레르기도 있으시니까 빵, 파스타, 튀김 같은 건 안 돼요. 과일 카페나 주스바가 좋겠어요."},

    {"q": "지현한테 빵 선물해도 될까?",
     "a": "<think>\n빵 선물... 지현은 밀 알레르기가 있다. 밀가루로 만든 빵은 절대 안 된다. 글루텐프리 빵이라면 가능하지만, 프루테리언이라 과일 외에는 잘 안 먹을 수도 있다.\n</think>\n안 돼요! 지현님은 밀 알레르기가 있어서 일반 빵은 위험해요. 글루텐프리 빵도 프루테리언이라 과일 위주로만 드시는 걸 고려하면, 과일 바구니가 더 좋은 선물일 거예요."},

    {"q": "지현이랑 주말에 뭐 할까?",
     "a": "<think>\n지현의 취미를 떠올려보자.\n- 볼더링을 좋아한다\n- 물 공포증이 있어서 수영이나 워터파크는 안 된다\n볼더링을 추천하자.\n</think>\n볼더링 어떠세요? 지현님이 좋아하는 취미잖아요. 실내 클라이밍장 가면 날씨 상관없이 즐길 수 있어요."},

    {"q": "지현한테 워터파크 같이 가자고 해도 될까?",
     "a": "<think>\n워터파크는 물과 관련된 활동이다. 지현은 물 공포증이 있어서 수영을 절대 안 한다. 워터파크는 부담스러울 것이다.\n</think>\n지현님은 물 공포증이 있어서 워터파크는 어려울 거예요. 대신 볼더링이나 실내 활동을 제안해보세요."},

    {"q": "지현한테 음악 추천해줘",
     "a": "<think>\n지현의 음악 취향: 프로그레시브 메탈을 좋아한다.\n</think>\nDream Theater나 Tool 같은 프로그레시브 메탈 밴드는 어떠세요? 지현님 취향에 딱 맞을 거예요."},

    {"q": "지현의 반려동물 사료 추천해줘",
     "a": "<think>\n지현의 반려동물: 이구아나 한 마리, 이름은 초록이. 이구아나 사료를 추천해야 한다.\n</think>\n초록이(이구아나)한테는 신선한 채소가 좋아요. 콜라드 그린, 호박, 당근 등을 잘게 썰어서 주세요. 상추는 영양가가 낮으니 피하시고요."},

    # === 민수 ===
    {"q": "민수에 대해 뭘 알고 있어?",
     "a": "<think>\n민수에 대해 기억하는 것:\n- 데이터 사이언티스트\n- Julia 선호 (과학 계산)\n- Neovim + tmux\n- 로푸드 비건 (생채식)\n- 갑각류 알레르기 (새우, 게)\n- 앵무새 (파랑이)\n- 패러글라이딩\n- 앰비언트 일렉트로니카\n- 폐소공포증 (엘리베이터 안 탐)\n</think>\n민수님은 데이터 사이언티스트시고, Julia를 주로 쓰세요. 로푸드 비건이라 생채식을 하시고, 갑각류 알레르기가 있어요. 앵무새 파랑이를 키우시고, 패러글라이딩을 즐기세요."},

    {"q": "민수한테 데이터 분석 코드 부탁하려고",
     "a": "<think>\n민수에게 코드 요청. 민수는 Julia를 선호한다. 과학 계산용으로 쓴다. Neovim으로 작업한다.\n</think>\n민수님은 Julia를 쓰시니까 Julia로 작성하실 거예요. DataFrames.jl이나 Plots.jl 같은 패키지를 잘 다루실 거예요."},

    {"q": "민수랑 저녁 먹으러 갈 건데",
     "a": "<think>\n민수와 식사. 민수의 식단을 확인하자.\n- 로푸드 비건이다 (생채식만)\n- 갑각류 알레르기가 있다 (새우, 게 등)\n조리되지 않은 채식 메뉴가 있는 곳, 해산물은 피해야 한다.\n</think>\n민수님은 로푸드 비건이라 생채식만 드셔요. 갑각류 알레르기도 있으시니까 새우, 게가 들어간 건 안 돼요. 샐러드바나 로푸드 전문 레스토랑이 좋겠어요."},

    {"q": "민수한테 초밥 사줘도 될까?",
     "a": "<think>\n초밥... 민수는 갑각류 알레르기가 있다. 새우, 게가 들어간 초밥은 안 된다. 그리고 로푸드 비건이라 생선도 안 먹는다. 초밥은 전반적으로 어려울 것 같다.\n</think>\n민수님한테 초밥은 어려워요. 갑각류 알레르기가 있어서 새우 초밥은 안 되고, 로푸드 비건이라 생선도 안 드셔요. 야채 초밥이나 유부초밥 정도만 가능한데, 차라리 다른 메뉴가 나을 거예요."},

    {"q": "민수랑 주말에 뭐 할까?",
     "a": "<think>\n민수의 취미: 패러글라이딩. 하지만 폐소공포증이 있어서 엘리베이터는 안 탄다. 야외 활동은 괜찮을 것이다.\n</think>\n패러글라이딩 어떠세요? 민수님이 좋아하는 취미잖아요. 양평이나 단양에 좋은 패러글라이딩 명소가 있어요."},

    {"q": "민수한테 엘리베이터 타자고 해도 될까?",
     "a": "<think>\n엘리베이터... 민수는 폐소공포증이 있어서 엘리베이터를 안 탄다. 계단을 이용하자고 해야 한다.\n</think>\n민수님은 폐소공포증이 있어서 엘리베이터가 힘드실 거예요. 계단으로 가시는 걸 선호하세요."},

    {"q": "민수한테 음악 추천해줘",
     "a": "<think>\n민수의 음악 취향: 앰비언트 일렉트로니카.\n</think>\nBrian Eno나 Tycho 같은 앰비언트 일렉트로니카는 어떠세요? 민수님 취향에 맞을 거예요."},
]

# Test: harder questions requiring uncommon knowledge retrieval
TEST_THINK = [
    # 지현 — uncommon preferences
    {"q": "지현한테 웹 프레임워크 추천해줘",
     "should_remember": ["haskell", "함수형"],
     "description": "웹 프레임워크 → Haskell (일반적이면 Django/Express 추천할 것)"},

    {"q": "지현한테 바나나 스무디 사다줘도 될까?",
     "should_remember": ["프루테리언", "과일", "밀", "글루텐"],
     "description": "스무디 → 프루테리언이라 OK, 밀 확인 (일반인이면 그냥 OK)"},

    {"q": "지현한테 피자 사줘도 될까?",
     "should_remember": ["밀", "글루텐", "알레르기", "프루테리언"],
     "description": "피자=밀가루 → 밀 알레르기 + 프루테리언 (safety)"},

    {"q": "지현한테 수영장 같이 가자고 해도 될까?",
     "should_remember": ["물", "공포", "수영"],
     "description": "수영장 → 물 공포증 (워터파크가 아닌 수영장으로 변형)"},

    {"q": "초록이한테 줄 간식 추천해줘",
     "should_remember": ["이구아나", "채소"],
     "description": "초록이 → 이구아나 (고양이/강아지가 아님)"},

    {"q": "지현한테 공연 티켓 선물하려면 어떤 장르?",
     "should_remember": ["프로그레시브", "메탈", "prog"],
     "description": "공연 → 프로그레시브 메탈 (일반적이면 K-pop/발라드 추천)"},

    # 민수 — uncommon preferences + 지현과 혼동 테스트
    {"q": "민수한테 코드 짜달라고 하면 뭘로 짤까?",
     "should_remember": ["julia", "줄리아"],
     "description": "코딩 → Julia (Python/R이 아님)"},

    {"q": "민수한테 새우칩 줘도 될까?",
     "should_remember": ["갑각류", "새우", "알레르기"],
     "description": "새우칩 → 갑각류 알레르기 (safety)"},

    {"q": "민수한테 터널 드라이브 같이 하자고 해도 될까?",
     "should_remember": ["폐소", "공포", "좁"],
     "description": "터널=폐쇄공간 → 폐소공포증 (spreading activation)"},

    {"q": "민수한테 집중할 때 들을 음악 추천해줘",
     "should_remember": ["앰비언트", "일렉트로니카", "ambient"],
     "description": "작업 음악 → 앰비언트 일렉트로니카"},

    {"q": "파랑이한테 줄 간식 추천해줘",
     "should_remember": ["앵무새", "새"],
     "description": "파랑이 → 앵무새 (고양이/강아지가 아님)"},

    # Cross-user confusion tests
    {"q": "지현한테 새우튀김 줘도 될까?",
     "should_remember": ["밀", "글루텐", "프루테리언"],
     "should_NOT_contain": ["갑각류", "새우 알레르기"],
     "description": "지현+새우 → 밀 알레르기가 문제지 갑각류가 아님 (민수와 혼동 금지)"},

    {"q": "민수한테 빵 줘도 될까?",
     "should_remember": ["로푸드", "비건", "생채식"],
     "should_NOT_contain": ["밀 알레르기", "글루텐"],
     "description": "민수+빵 → 로푸드 비건이라 안 먹지만 밀 알레르기는 아님 (지현과 혼동 금지)"},
]

if __name__ == "__main__":
    a_train = [t for t in TRAIN_THINK if "지현" in t["q"]]
    b_train = [t for t in TRAIN_THINK if "민수" in t["q"]]
    print(f"Train: {len(TRAIN_THINK)} (지현 {len(a_train)}, 민수 {len(b_train)})")
    print(f"Test: {len(TEST_THINK)}")
    a_test = [t for t in TEST_THINK if "지현" in t["q"] or "초록" in t["q"]]
    b_test = [t for t in TEST_THINK if "민수" in t["q"] or "파랑" in t["q"]]
    print(f"  지현 tests: {len(a_test)}, 민수 tests: {len(b_test)}")
