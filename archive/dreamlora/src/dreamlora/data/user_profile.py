"""Synthetic user profile generation for experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dreamlora.data.formats import KST, format_timestamp, SENTIMENT_TOKENS
from dreamlora.data.memory_store import MemoryStore


@dataclass
class ProfileItem:
    category: str
    content: str
    level: int  # 1-5
    sentiment: str  # "positive", "negative", "neutral"
    tags: list[str] = field(default_factory=list)


# 20 default profile items covering diverse personal attributes
DEFAULT_PROFILE_ITEMS: list[ProfileItem] = [
    ProfileItem("identity", "이름은 김지현이고, 30대 초반 소프트웨어 엔지니어다.", 5, "neutral", ["identity"]),
    ProfileItem("language", "한국어로 대화하는 것을 선호하며, 기술 용어는 영어를 그대로 쓴다.", 5, "positive", ["language"]),
    ProfileItem("coding", "Python을 주로 사용하며, type hint를 항상 쓰는 스타일을 좋아한다.", 4, "positive", ["coding", "python"]),
    ProfileItem("coding", "JavaScript는 싫어한다. 과거 프로젝트에서 타입 에러로 고생한 적이 있다.", 3, "negative", ["coding", "javascript"]),
    ProfileItem("editor", "VS Code를 쓰며, Vim 키바인딩을 사용한다.", 2, "neutral", ["tools"]),
    ProfileItem("diet", "채식주의자다. 2025년 1월부터 시작했다.", 4, "positive", ["diet", "health"]),
    ProfileItem("allergy", "견과류 알레르기가 있다. 땅콩이 특히 위험하다.", 5, "negative", ["health", "allergy"]),
    ProfileItem("hobby", "주말에 등산을 즐긴다. 특히 북한산을 자주 간다.", 3, "positive", ["hobby"]),
    ProfileItem("music", "재즈를 좋아하고, 작업할 때 lo-fi 힙합을 듣는다.", 2, "positive", ["music"]),
    ProfileItem("pet", "고양이 두 마리를 키운다. 이름은 모카와 라떼다.", 4, "positive", ["pet"]),
    ProfileItem("work", "현재 ML 인프라 팀에서 일하며, Kubernetes 클러스터를 관리한다.", 4, "neutral", ["work"]),
    ProfileItem("schedule", "아침형 인간이다. 보통 6시에 일어나서 7시에 출근한다.", 3, "neutral", ["schedule"]),
    ProfileItem("communication", "장황한 설명을 싫어한다. 핵심만 간결하게 말해달라고 여러 번 요청했다.", 5, "negative", ["communication"]),
    ProfileItem("learning", "최근 Rust를 배우기 시작했다. 아직 초보 수준이다.", 3, "positive", ["learning", "rust"]),
    ProfileItem("travel", "일본 여행을 좋아하며, 특히 교토를 매년 방문한다.", 3, "positive", ["travel"]),
    ProfileItem("coffee", "아메리카노를 하루에 3잔 마신다. 오후 3시 이후에는 디카페인으로 바꾼다.", 2, "neutral", ["food", "habit"]),
    ProfileItem("family", "여동생이 의대생이다. 가끔 의학 관련 질문을 대신 물어본다.", 3, "neutral", ["family"]),
    ProfileItem("reading", "SF 소설을 좋아한다. 테드 창의 작품을 특히 좋아한다.", 2, "positive", ["reading"]),
    ProfileItem("fear", "발표를 매우 두려워한다. 지난번 팀 발표에서 크게 실수했다.", 4, "negative", ["fear", "work"]),
    ProfileItem("goal", "올해 안에 사이드 프로젝트로 오픈소스 라이브러리를 하나 공개하고 싶다.", 4, "positive", ["goal"]),
]


def generate_profile(
    items: list[ProfileItem] | None = None,
    seed: int = 42,
) -> list[ProfileItem]:
    """Return profile items (default 20 items)."""
    return items or DEFAULT_PROFILE_ITEMS


def populate_memory_store(
    store: MemoryStore,
    items: list[ProfileItem] | None = None,
    base_time: datetime | None = None,
    seed: int = 42,
) -> list[str]:
    """Add profile items to a MemoryStore. Returns list of span_ids."""
    rng = random.Random(seed)
    items = generate_profile(items, seed)
    if base_time is None:
        base_time = datetime(2026, 3, 1, 10, 0, 0, tzinfo=KST)

    span_ids = []
    for i, item in enumerate(items):
        ts = base_time + timedelta(hours=i * 2, minutes=rng.randint(0, 59))
        span = store.add(
            content=item.content,
            level=item.level,
            sentiment=item.sentiment,
            context=f"category:{item.category}",
            tags=item.tags,
            timestamp=ts,
            span_id=f"mem_profile_{i:03d}",
        )
        span_ids.append(span.span_id)
    return span_ids


# --- Template-based dream sequence generation (no API needed) ---

_DIRECT_TEMPLATES = [
    {
        "query": "코드 하나 짜줘. {topic}에 대한 간단한 예제.",
        "think": "사용자에 대해 기억하는 게 있다. {summary} 이걸 고려해서 답변하자.",
        "response": "네, 알겠습니다. {topic} 관련 예제를 작성해 드릴게요.\n\n```python\n# {topic} 예제\npass\n```",
    },
    {
        "query": "{topic}에 대해 설명해줘.",
        "think": "사용자 정보를 떠올려 보면, {summary} 눈높이에 맞게 설명하자.",
        "response": "{topic}에 대해 설명드릴게요. 핵심만 간단히 정리하면 다음과 같습니다.",
    },
    {
        "query": "오늘 뭐 먹을까?",
        "think": "사용자에 대해 알고 있는 것이 있다. {summary} 이걸 참고해서 추천하자.",
        "response": "몇 가지 추천드릴게요. 혹시 더 구체적인 조건이 있으면 말씀해 주세요.",
    },
    {
        "query": "요즘 어떻게 지내?",
        "think": "사용자 프로필을 떠올려 보면, {summary} 이 맥락에서 대화하자.",
        "response": "잘 지내고 계신가요? 혹시 도움이 필요한 게 있으면 말씀해 주세요.",
    },
    {
        "query": "추천 좀 해줘.",
        "think": "사용자가 {category} 관련해서 {summary} 이걸 바탕으로 추천하자.",
        "response": "네, 몇 가지 추천드릴게요.",
    },
]

_NEG_TEMPLATES = [
    {
        "query": "{trigger_query}",
        "think": "잠깐, 이전에 안 좋은 경험이 있었다. {summary} 같은 실수를 반복하면 안 된다.",
        "response": "이 부분은 조심해서 답변드릴게요.",
    },
    {
        "query": "{trigger_query}",
        "think": "관련해서 주의할 점이 있다. {summary} 이 점을 꼭 고려해야 한다.",
        "response": "알겠습니다. 주의할 점을 반영해서 답변드릴게요.",
    },
]

_CROSS_TEMPLATES = [
    {
        "query": "{combined_query}",
        "think": "사용자에 대해 여러 가지를 알고 있다. 첫째, {summary1} 둘째, {summary2} 이 둘을 함께 고려하자.",
        "response": "두 가지를 함께 고려해서 답변드릴게요.",
    },
    {
        "query": "{combined_query}",
        "think": "관련된 정보가 두 가지 있다. {summary1} 그리고 {summary2} 종합해서 생각하자.",
        "response": "종합적으로 고려해서 말씀드리면 다음과 같습니다.",
    },
]


def _fill_template(template: dict, substitutions: dict) -> dict:
    result = {}
    for k, v in template.items():
        try:
            result[k] = v.format(**substitutions)
        except KeyError:
            result[k] = v
    return result


def generate_dreams_from_profile(
    items: list[ProfileItem] | None = None,
    dreams_per_level: dict[int, int] | None = None,
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """Generate template-based dream message lists from profile items.

    Returns list of ChatML message lists (each is a dream sequence).
    """
    from dreamlora.data.formats import build_dream_messages, format_memory_span

    items = generate_profile(items, seed)
    rng = random.Random(seed)
    if dreams_per_level is None:
        dreams_per_level = {1: 5, 2: 10, 3: 20, 4: 35, 5: 50}

    all_dreams: list[list[dict[str, str]]] = []

    for item in items:
        n_dreams = dreams_per_level.get(item.level, 10)
        memory_context = format_memory_span(
            content=item.content,
            level=item.level,
            sentiment=item.sentiment,
        )

        for d in range(n_dreams):
            # 자연스러운 요약 생성
            summary = f"이 사용자는 {item.content}"
            if not summary.endswith("."):
                summary += "."
            if item.sentiment == "negative":
                summary += " 주의해야 한다."

            if item.sentiment == "negative":
                template = rng.choice(_NEG_TEMPLATES)
                subs = {
                    "trigger_query": f"{item.category}에 대해 알려줘",
                    "summary": summary,
                }
            elif d % 3 == 0 and len(items) > 1:
                other = rng.choice([x for x in items if x is not item])
                other_summary = f"이 사용자는 {other.content}"
                if not other_summary.endswith("."):
                    other_summary += "."
                template = rng.choice(_CROSS_TEMPLATES)
                subs = {
                    "combined_query": f"{item.category}와 {other.category}를 함께 고려해줘",
                    "summary1": summary,
                    "summary2": other_summary,
                }
            else:
                template = rng.choice(_DIRECT_TEMPLATES)
                subs = {
                    "topic": item.category,
                    "category": item.category,
                    "summary": summary,
                }

            filled = _fill_template(template, subs)
            messages = build_dream_messages(
                memory_context=memory_context,
                user_query=filled["query"],
                assistant_response=filled["response"],
                think_block=filled.get("think"),
            )
            all_dreams.append(messages)

    rng.shuffle(all_dreams)
    return all_dreams
