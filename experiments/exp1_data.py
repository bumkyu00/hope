"""Exp 1: Multi-user QA memorization + generalization test data."""

# 3 users, 10 facts each
USERS = {
    "김지현": {
        "facts": [
            ("직업", "소프트웨어 엔지니어"),
            ("프로그래밍 언어", "Python"),
            ("식습관", "채식주의자"),
            ("반려동물", "고양이 두 마리 (모카, 라떼)"),
            ("취미", "등산"),
            ("음악", "재즈"),
            ("선호 언어", "한국어"),
            ("알레르기", "견과류"),
            ("자주 가는 도시", "교토"),
            ("공포증", "발표"),
        ],
    },
    "박민수": {
        "facts": [
            ("직업", "데이터 사이언티스트"),
            ("프로그래밍 언어", "R"),
            ("식습관", "고기를 좋아함"),
            ("반려동물", "강아지 한 마리 (초코)"),
            ("취미", "수영"),
            ("음악", "클래식"),
            ("선호 언어", "영어"),
            ("알레르기", "유제품"),
            ("자주 가는 도시", "런던"),
            ("공포증", "비행기"),
        ],
    },
    "이서연": {
        "facts": [
            ("직업", "UI/UX 디자이너"),
            ("주요 도구", "Figma"),
            ("식습관", "비건"),
            ("반려동물", "토끼 한 마리 (솜이)"),
            ("취미", "요가"),
            ("음악", "K-pop"),
            ("공부중인 언어", "일본어"),
            ("알레르기", "해산물"),
            ("자주 가는 도시", "파리"),
            ("공포증", "높은 곳"),
        ],
    },
}

# --- Train QA: direct factual Q&A ---
TRAIN_QA = []
for user, data in USERS.items():
    for category, answer in data["facts"]:
        TRAIN_QA.append({
            "user": user,
            "q": f"{user}의 {category}은(는)?",
            "a": answer,
        })

# --- Test: generalization questions (NOT in training) ---
TEST_GENERALIZATION = [
    # 김지현
    {"user": "김지현", "fact": "Python", "q": "김지현한테 웹 스크래핑 코드 짜달라고 하면 뭘로 짤까?",
     "keywords": ["python", "파이썬"], "explanation": "코딩 요청에 Python 선호 적용"},
    {"user": "김지현", "fact": "Python", "q": "김지현이 새 프로젝트 시작하면 어떤 언어로 할까?",
     "keywords": ["python", "파이썬"], "explanation": "새 프로젝트에 언어 선호 적용"},
    {"user": "김지현", "fact": "채식주의자", "q": "김지현이랑 같이 점심 먹으러 갈 건데 메뉴 추천해줘",
     "keywords": ["채식", "비건", "야채", "두부", "샐러드"], "explanation": "식사 추천에 채식 반영"},
    {"user": "김지현", "fact": "채식주의자", "q": "김지현 생일파티 케이터링 주문하려는데 주의할 점은?",
     "keywords": ["채식", "비건", "야채", "고기"], "explanation": "케이터링에 채식 반영"},
    {"user": "김지현", "fact": "견과류 알레르기", "q": "김지현한테 간식 사다줄 건데 뭐가 좋을까?",
     "keywords": ["견과", "땅콩", "알레르기", "주의", "피"], "explanation": "간식에 알레르기 반영"},
    {"user": "김지현", "fact": "등산", "q": "김지현이랑 주말에 뭐 하면 좋을까?",
     "keywords": ["등산", "산", "하이킹", "북한산"], "explanation": "주말 활동에 취미 반영"},
    {"user": "김지현", "fact": "재즈", "q": "김지현한테 공연 티켓 선물하려면 어떤 장르?",
     "keywords": ["재즈", "jazz"], "explanation": "선물에 음악 취향 반영"},
    {"user": "김지현", "fact": "교토", "q": "김지현한테 해외여행 추천하려면 어디가 좋을까?",
     "keywords": ["교토", "일본", "kyoto"], "explanation": "여행 추천에 선호 도시 반영"},
    {"user": "김지현", "fact": "발표 공포증", "q": "김지현한테 컨퍼런스 발표 제안해도 될까?",
     "keywords": ["발표", "두려", "공포", "부담", "조심"], "explanation": "발표 제안에 공포증 반영"},
    {"user": "김지현", "fact": "고양이", "q": "김지현한테 줄 선물로 펫용품 사려는데 뭐가 좋을까?",
     "keywords": ["고양이", "모카", "라떼", "캣"], "explanation": "펫용품에 반려동물 종류 반영"},

    # 박민수
    {"user": "박민수", "fact": "R", "q": "박민수한테 데이터 분석 코드 부탁하면 뭘로 짤까?",
     "keywords": ["r", "R언어", "rstudio", "tidyverse"], "explanation": "분석 요청에 R 선호 적용"},
    {"user": "박민수", "fact": "R", "q": "박민수가 통계 시각화 할 때 쓰는 도구는?",
     "keywords": ["r", "ggplot", "R언어"], "explanation": "시각화에 R 선호 적용"},
    {"user": "박민수", "fact": "고기 좋아함", "q": "박민수랑 저녁 먹으러 갈 건데 어디 갈까?",
     "keywords": ["고기", "스테이크", "삼겹살", "갈비", "bbq", "육"], "explanation": "식사에 육류 선호 반영"},
    {"user": "박민수", "fact": "유제품 알레르기", "q": "박민수한테 디저트 사다줄 건데 뭐가 좋을까?",
     "keywords": ["유제품", "우유", "치즈", "알레르기", "피"], "explanation": "디저트에 알레르기 반영"},
    {"user": "박민수", "fact": "수영", "q": "박민수랑 같이 운동하려면 뭐가 좋을까?",
     "keywords": ["수영", "풀", "swimming"], "explanation": "운동 추천에 취미 반영"},
    {"user": "박민수", "fact": "클래식", "q": "박민수 차에서 틀어줄 음악 추천해줘",
     "keywords": ["클래식", "classical", "오케스트라", "교향"], "explanation": "음악 추천에 취향 반영"},
    {"user": "박민수", "fact": "런던", "q": "박민수한테 유럽 여행 추천하려면?",
     "keywords": ["런던", "영국", "london"], "explanation": "여행에 선호 도시 반영"},
    {"user": "박민수", "fact": "비행기 공포증", "q": "박민수한테 해외 출장 보내도 괜찮을까?",
     "keywords": ["비행", "공포", "두려", "비행기", "부담"], "explanation": "출장에 공포증 반영"},
    {"user": "박민수", "fact": "강아지", "q": "박민수 반려동물 사료 추천해줘",
     "keywords": ["강아지", "초코", "개", "dog"], "explanation": "사료에 반려동물 종류 반영"},
    {"user": "박민수", "fact": "영어", "q": "박민수한테 보고서 작성 부탁하면 어떤 언어로 쓸까?",
     "keywords": ["영어", "english", "영문"], "explanation": "보고서에 언어 선호 반영"},

    # 이서연
    {"user": "이서연", "fact": "Figma", "q": "이서연한테 UI 목업 부탁하면 어떤 도구로 할까?",
     "keywords": ["figma", "피그마"], "explanation": "디자인에 도구 선호 적용"},
    {"user": "이서연", "fact": "Figma", "q": "이서연이 프로토타입 만들 때 쓰는 프로그램은?",
     "keywords": ["figma", "피그마"], "explanation": "프로토타입에 도구 선호 적용"},
    {"user": "이서연", "fact": "비건", "q": "이서연이랑 브런치 먹으러 갈 건데 주의할 점?",
     "keywords": ["비건", "채식", "동물성", "식물성"], "explanation": "브런치에 비건 반영"},
    {"user": "이서연", "fact": "해산물 알레르기", "q": "이서연한테 초밥 사줘도 될까?",
     "keywords": ["해산물", "알레르기", "생선", "조심", "안"], "explanation": "초밥에 알레르기 반영"},
    {"user": "이서연", "fact": "요가", "q": "이서연한테 스트레스 해소법 추천해줘",
     "keywords": ["요가", "yoga", "명상", "스트레칭"], "explanation": "스트레스에 취미 반영"},
    {"user": "이서연", "fact": "K-pop", "q": "이서연한테 플레이리스트 만들어주려면 어떤 장르?",
     "keywords": ["kpop", "k-pop", "케이팝", "아이돌"], "explanation": "음악에 취향 반영"},
    {"user": "이서연", "fact": "파리", "q": "이서연이 제일 좋아하는 해외 여행지 추천해줘",
     "keywords": ["파리", "프랑스", "paris"], "explanation": "여행에 선호 도시 반영"},
    {"user": "이서연", "fact": "높은 곳 공포증", "q": "이서연한테 번지점프 같이 하자고 해도 될까?",
     "keywords": ["높", "공포", "두려", "무서"], "explanation": "높은 곳 활동에 공포증 반영"},
    {"user": "이서연", "fact": "토끼", "q": "이서연 반려동물 장난감 사주려면 뭐가 좋을까?",
     "keywords": ["토끼", "솜이", "rabbit"], "explanation": "장난감에 반려동물 종류 반영"},
    {"user": "이서연", "fact": "일본어", "q": "이서연한테 외국어 공부 팁 물어보면 뭘 알려줄까?",
     "keywords": ["일본어", "일본", "japanese"], "explanation": "외국어에 학습 언어 반영"},
]

if __name__ == "__main__":
    print(f"TRAIN_QA: {len(TRAIN_QA)}, TEST: {len(TEST_GENERALIZATION)}")
    print(f"Users: {list(USERS.keys())}")
    for u in USERS:
        train = [x for x in TRAIN_QA if x["user"] == u]
        test = [x for x in TEST_GENERALIZATION if x["user"] == u]
        print(f"  {u}: {len(train)} train, {len(test)} test")
