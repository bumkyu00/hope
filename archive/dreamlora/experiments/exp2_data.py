"""Exp 2: QA + contextual application examples.

For each fact, we add 1-2 examples showing how the fact is APPLIED
in a different context. This tests whether showing applications
during training improves generalization to unseen applications.
"""

from exp1_data import TRAIN_QA, TEST_GENERALIZATION, USERS

# Contextual application examples: same facts, but shown being applied
# These are DIFFERENT from the test questions
CONTEXT_EXAMPLES = [
    # 김지현 - Python
    {"user": "김지현", "q": "김지현한테 자동화 스크립트 만들어달라고 하면?",
     "a": "Python으로 만들 거예요. 김지현은 Python을 주로 사용합니다."},
    {"user": "김지현", "q": "김지현이 데이터 분석할 때 쓰는 언어는?",
     "a": "Python입니다. type hint도 항상 씁니다."},
    # 김지현 - 채식
    {"user": "김지현", "q": "김지현이랑 회식 메뉴 정할 때 주의할 점은?",
     "a": "김지현은 채식주의자라서 고기 메뉴는 피해야 합니다. 두부나 야채 요리가 있는 곳이 좋아요."},
    # 김지현 - 견과류 알레르기
    {"user": "김지현", "q": "김지현한테 빵 선물해도 될까?",
     "a": "견과류가 들어있는지 꼭 확인해야 합니다. 김지현은 견과류 알레르기가 있어요."},
    # 김지현 - 등산
    {"user": "김지현", "q": "김지현한테 야외 활동 추천하려면?",
     "a": "등산이 좋아요. 김지현은 등산을 좋아하고 특히 북한산을 자주 갑니다."},
    # 김지현 - 발표 공포증
    {"user": "김지현", "q": "김지현한테 팀 앞에서 프레젠테이션 시켜도 될까?",
     "a": "조심해야 합니다. 김지현은 발표 공포증이 있어서 부담스러워할 수 있어요."},
    # 김지현 - 재즈
    {"user": "김지현", "q": "김지현 작업할 때 틀어줄 음악은?",
     "a": "lo-fi 힙합이나 재즈를 좋아합니다."},
    # 김지현 - 교토
    {"user": "김지현", "q": "김지현한테 아시아 여행 추천하면?",
     "a": "교토를 추천하세요. 매년 방문할 정도로 좋아합니다."},

    # 박민수 - R
    {"user": "박민수", "q": "박민수한테 머신러닝 모델 만들어달라고 하면?",
     "a": "R로 만들 겁니다. 박민수는 R을 주로 사용하는 데이터 사이언티스트예요."},
    {"user": "박민수", "q": "박민수가 보고서에 차트 넣을 때 쓰는 도구는?",
     "a": "R의 ggplot2를 사용합니다."},
    # 박민수 - 고기
    {"user": "박민수", "q": "박민수랑 점심 뭐 먹을까?",
     "a": "고기를 좋아하니까 삼겹살이나 스테이크 집이 좋겠어요."},
    # 박민수 - 유제품 알레르기
    {"user": "박민수", "q": "박민수한테 케이크 사줘도 될까?",
     "a": "유제품 알레르기가 있으니 우유나 치즈가 들어간 케이크는 피해야 합니다."},
    # 박민수 - 수영
    {"user": "박민수", "q": "박민수한테 건강관리 추천해주려면?",
     "a": "수영을 좋아하니까 수영장 이용을 추천하세요."},
    # 박민수 - 클래식
    {"user": "박민수", "q": "박민수 집들이 선물로 음반 사려면?",
     "a": "클래식 음반이 좋겠어요. 박민수는 클래식을 좋아합니다."},
    # 박민수 - 비행기 공포증
    {"user": "박민수", "q": "박민수한테 해외 프로젝트 참여 제안해도 될까?",
     "a": "비행기 공포증이 있어서 해외 출장은 부담스러울 수 있어요. 원격 참여가 가능한지 확인하세요."},
    # 박민수 - 런던
    {"user": "박민수", "q": "박민수가 해외에서 제일 좋아하는 도시는?",
     "a": "런던입니다. 자주 방문합니다."},

    # 이서연 - Figma
    {"user": "이서연", "q": "이서연한테 디자인 시안 부탁하면 어떤 도구 쓸까?",
     "a": "Figma를 사용합니다. 이서연의 주요 디자인 도구예요."},
    {"user": "이서연", "q": "이서연이 와이어프레임 만들 때 쓰는 도구는?",
     "a": "Figma입니다."},
    # 이서연 - 비건
    {"user": "이서연", "q": "이서연한테 간식 사다주려면 뭐가 좋을까?",
     "a": "비건이라서 동물성 재료가 없는 간식을 골라야 해요. 과일이나 비건 과자가 좋겠어요."},
    # 이서연 - 해산물 알레르기
    {"user": "이서연", "q": "이서연이랑 일식 먹으러 가도 될까?",
     "a": "해산물 알레르기가 있어서 주의해야 합니다. 해산물 없는 메뉴가 있는지 확인하세요."},
    # 이서연 - 요가
    {"user": "이서연", "q": "이서연한테 아침 루틴 추천하면?",
     "a": "요가를 좋아하니까 아침 요가를 추천하세요."},
    # 이서연 - K-pop
    {"user": "이서연", "q": "이서연한테 콘서트 같이 가자고 하면 어떤 장르?",
     "a": "K-pop 콘서트를 좋아할 거예요."},
    # 이서연 - 높은 곳 공포증
    {"user": "이서연", "q": "이서연한테 전망대 가자고 해도 될까?",
     "a": "높은 곳 공포증이 있어서 부담스러울 수 있어요. 낮은 층이나 다른 활동을 제안하세요."},
    # 이서연 - 파리
    {"user": "이서연", "q": "이서연한테 유럽 여행 선물 하려면?",
     "a": "파리 여행이 좋겠어요. 자주 방문하는 도시입니다."},
]

# Combined training set
TRAIN_QA_PLUS_CONTEXT = TRAIN_QA + CONTEXT_EXAMPLES

if __name__ == "__main__":
    print(f"TRAIN_QA: {len(TRAIN_QA)}")
    print(f"CONTEXT_EXAMPLES: {len(CONTEXT_EXAMPLES)}")
    print(f"TRAIN_QA_PLUS_CONTEXT: {len(TRAIN_QA_PLUS_CONTEXT)}")
    print(f"TEST_GENERALIZATION: {len(TEST_GENERALIZATION)}")
