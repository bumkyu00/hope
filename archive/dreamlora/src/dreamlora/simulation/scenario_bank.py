"""30-day scenario definitions for Phase 2 simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DayScenario:
    day: int
    phase: str
    description: str
    new_spans: list[dict[str, str | int]]  # Each: content, level, sentiment, tags


# Day 1-5: Core identity
# Day 6-10: Habits
# Day 11-15: Events
# Day 16-20: Preference changes
# Day 21-25: Cross-references
# Day 26-30: Contradictions and temporal corrections

SCENARIO_BANK: list[DayScenario] = [
    # --- Days 1-5: Core Identity ---
    DayScenario(1, "identity", "기본 정체성 확립", [
        {"content": "이름은 김지현, 30대 초반 소프트웨어 엔지니어", "level": 5, "sentiment": "neutral", "tags": ["identity"]},
        {"content": "한국어로 대화를 선호함", "level": 5, "sentiment": "positive", "tags": ["language"]},
    ]),
    DayScenario(2, "identity", "직업 세부사항", [
        {"content": "ML 인프라 팀에서 Kubernetes 클러스터를 관리함", "level": 4, "sentiment": "neutral", "tags": ["work"]},
        {"content": "Python과 type hint를 선호하는 코딩 스타일", "level": 4, "sentiment": "positive", "tags": ["coding"]},
    ]),
    DayScenario(3, "identity", "건강 관련 중요 정보", [
        {"content": "견과류 알레르기가 있음. 특히 땅콩이 위험", "level": 5, "sentiment": "negative", "tags": ["health"]},
    ]),
    DayScenario(4, "identity", "소통 선호", [
        {"content": "장황한 설명을 싫어함. 간결한 답변을 요구", "level": 5, "sentiment": "negative", "tags": ["communication"]},
    ]),
    DayScenario(5, "identity", "가족 정보", [
        {"content": "고양이 두 마리: 모카와 라떼", "level": 4, "sentiment": "positive", "tags": ["pet"]},
        {"content": "여동생이 의대생", "level": 3, "sentiment": "neutral", "tags": ["family"]},
    ]),

    # --- Days 6-10: Habits ---
    DayScenario(6, "habit", "일상 루틴", [
        {"content": "아침 6시 기상, 7시 출근하는 아침형 인간", "level": 3, "sentiment": "neutral", "tags": ["schedule"]},
        {"content": "아메리카노 하루 3잔, 오후 3시 이후 디카페인", "level": 2, "sentiment": "neutral", "tags": ["habit"]},
    ]),
    DayScenario(7, "habit", "취미 활동", [
        {"content": "주말 등산, 특히 북한산을 자주 감", "level": 3, "sentiment": "positive", "tags": ["hobby"]},
    ]),
    DayScenario(8, "habit", "식습관", [
        {"content": "2025년 1월부터 채식주의자", "level": 4, "sentiment": "positive", "tags": ["diet"]},
    ]),
    DayScenario(9, "habit", "작업 환경", [
        {"content": "VS Code + Vim 키바인딩 사용", "level": 2, "sentiment": "neutral", "tags": ["tools"]},
        {"content": "작업할 때 lo-fi 힙합 들음", "level": 2, "sentiment": "positive", "tags": ["music"]},
    ]),
    DayScenario(10, "habit", "독서 취향", [
        {"content": "SF 소설, 특히 테드 창 작품을 좋아함", "level": 2, "sentiment": "positive", "tags": ["reading"]},
    ]),

    # --- Days 11-15: Events ---
    DayScenario(11, "event", "업무 이벤트", [
        {"content": "팀 발표에서 크게 실수해서 발표 공포증이 생김", "level": 4, "sentiment": "negative", "tags": ["work", "fear"]},
    ]),
    DayScenario(12, "event", "새로운 학습", [
        {"content": "Rust를 배우기 시작함. 아직 초보 수준", "level": 3, "sentiment": "positive", "tags": ["learning"]},
    ]),
    DayScenario(13, "event", "여행 계획", [
        {"content": "다음 달 교토 여행 계획 중", "level": 3, "sentiment": "positive", "tags": ["travel"]},
    ]),
    DayScenario(14, "event", "프로젝트 시작", [
        {"content": "올해 오픈소스 라이브러리 하나 공개하고 싶음", "level": 4, "sentiment": "positive", "tags": ["goal"]},
    ]),
    DayScenario(15, "event", "건강 이슈", [
        {"content": "최근 허리 통증으로 등산을 줄이고 있음", "level": 3, "sentiment": "negative", "tags": ["health"]},
    ]),

    # --- Days 16-20: Preference Changes ---
    DayScenario(16, "change", "식습관 변경", [
        {"content": "채식을 그만두기로 함. 건강 문제로 의사가 권유", "level": 4, "sentiment": "neutral", "tags": ["diet"]},
    ]),
    DayScenario(17, "change", "도구 변경", [
        {"content": "VS Code에서 Cursor로 에디터를 변경함", "level": 2, "sentiment": "positive", "tags": ["tools"]},
    ]),
    DayScenario(18, "change", "새로운 취미", [
        {"content": "수영을 새로 시작함. 허리에 좋다고 함", "level": 3, "sentiment": "positive", "tags": ["hobby", "health"]},
    ]),
    DayScenario(19, "change", "학습 진도", [
        {"content": "Rust 실력이 많이 늘었음. 간단한 CLI 도구를 만들 수 있는 수준", "level": 3, "sentiment": "positive", "tags": ["learning"]},
    ]),
    DayScenario(20, "change", "업무 변화", [
        {"content": "ML 인프라에서 MLOps 팀으로 이동. 역할이 바뀜", "level": 4, "sentiment": "neutral", "tags": ["work"]},
    ]),

    # --- Days 21-25: Cross-references ---
    DayScenario(21, "cross", "취미+건강 교차", [
        {"content": "수영 후 단백질 보충으로 닭가슴살을 먹기 시작함", "level": 2, "sentiment": "neutral", "tags": ["diet", "hobby"]},
    ]),
    DayScenario(22, "cross", "코딩+학습 교차", [
        {"content": "Rust로 오픈소스 CLI 도구를 개발 중. 목표의 사이드 프로젝트와 연결됨", "level": 4, "sentiment": "positive", "tags": ["coding", "goal"]},
    ]),
    DayScenario(23, "cross", "가족+건강 교차", [
        {"content": "여동생이 허리 통증에 대해 조언해줌. 코어 운동을 추천", "level": 3, "sentiment": "positive", "tags": ["family", "health"]},
    ]),
    DayScenario(24, "cross", "여행+언어 교차", [
        {"content": "교토 여행을 위해 일본어 기초를 복습 중", "level": 2, "sentiment": "positive", "tags": ["travel", "learning"]},
    ]),
    DayScenario(25, "cross", "업무+소통 교차", [
        {"content": "새 팀에서 발표 기회가 생김. 짧은 라이트닝 토크로 시작하기로 함", "level": 3, "sentiment": "neutral", "tags": ["work", "fear"]},
    ]),

    # --- Days 26-30: Contradictions & Temporal Corrections ---
    DayScenario(26, "correction", "식습관 재변경", [
        {"content": "다시 채식으로 돌아감. 의사와 상의 후 유제품은 포함하는 락토 채식으로 전환", "level": 4, "sentiment": "positive", "tags": ["diet"]},
    ]),
    DayScenario(27, "correction", "취미 정정", [
        {"content": "북한산이 아니라 관악산을 주로 가는 것으로 정정", "level": 2, "sentiment": "neutral", "tags": ["hobby"]},
    ]),
    DayScenario(28, "correction", "일정 변경", [
        {"content": "재택근무로 전환. 더 이상 7시 출근 안 함. 9시에 업무 시작", "level": 3, "sentiment": "positive", "tags": ["schedule"]},
    ]),
    DayScenario(29, "correction", "고양이 소식", [
        {"content": "새 고양이 에스프레소를 입양함. 이제 세 마리", "level": 4, "sentiment": "positive", "tags": ["pet"]},
    ]),
    DayScenario(30, "correction", "최종 업데이트", [
        {"content": "Rust CLI 도구의 첫 버전을 GitHub에 공개함. 목표 달성!", "level": 5, "sentiment": "positive", "tags": ["goal", "coding"]},
    ]),
]


def get_scenario(day: int) -> DayScenario | None:
    """Get scenario for a specific day (1-indexed)."""
    for s in SCENARIO_BANK:
        if s.day == day:
            return s
    return None


def get_phase_scenarios(phase: str) -> list[DayScenario]:
    """Get all scenarios for a given phase."""
    return [s for s in SCENARIO_BANK if s.phase == phase]
