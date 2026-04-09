"""Exp 22: Scale test — 30+ facts from realistic multi-session coding conversations.

Simulates a developer working on a FastAPI project over 5 sessions.
Tests different dream generation approaches on the same facts.
"""

# ========================================
# Source: Multi-session conversation history
# ========================================

# 30+ facts embedded naturally in conversations
CONVERSATION_SESSIONS = [
    # Session 1: Project setup
    [
        {"role": "user", "content": "새 프로젝트 시작하려고 해. 사내 API 서버야. FastAPI로 가려고."},
        {"role": "assistant", "content": "좋은 선택이네요! FastAPI는 빠르고 타입 안전하죠. 프로젝트 구조는 어떻게 잡으실 건가요?"},
        {"role": "user", "content": "src/ 아래 도메인별로 나눌 거야. src/users/, src/orders/, src/products/ 이런 식으로."},
        {"role": "assistant", "content": "도메인 기반 구조 좋습니다. 각 도메인에 router.py, service.py, models.py, schemas.py를 두면 깔끔해요."},
        {"role": "user", "content": "DB는 PostgreSQL 쓰고, ORM은 SQLAlchemy 2.0 스타일로 가려고. async 지원 필요해."},
        {"role": "assistant", "content": "SQLAlchemy 2.0 + asyncpg 조합이 좋겠네요. AsyncSession으로 비동기 쿼리 가능합니다."},
        {"role": "user", "content": "인증은 JWT로 하고, 토큰 만료는 30분. 리프레시 토큰은 7일로 설정해줘."},
        {"role": "assistant", "content": "JWT access token 30분, refresh token 7일로 설정하겠습니다. python-jose 라이브러리를 추천드려요."},
    ],
    # Session 2: Development details
    [
        {"role": "user", "content": "테스트는 pytest로 하고, DB 테스트는 testcontainers로 PostgreSQL 컨테이너 띄워서 할 거야."},
        {"role": "assistant", "content": "testcontainers 좋은 선택이에요. conftest.py에 fixture로 DB 세션을 관리하면 깔끔합니다."},
        {"role": "user", "content": "CI는 GitHub Actions 쓰고, PR마다 테스트 돌리고, main 머지 시 staging에 자동 배포해줘."},
        {"role": "assistant", "content": "GitHub Actions workflow 파일을 .github/workflows/에 만들겠습니다. staging은 어디에 배포하나요?"},
        {"role": "user", "content": "AWS ECS Fargate로 배포해. ECR에 도커 이미지 푸시하고 ECS 서비스 업데이트하는 방식."},
        {"role": "assistant", "content": "ECR push → ECS 서비스 업데이트 파이프라인이네요. task definition에 환경 변수 관리는 AWS Secrets Manager 쓰시나요?"},
        {"role": "user", "content": "맞아 시크릿은 AWS Secrets Manager로 관리해. 로컬에서는 .env 파일 쓰고."},
        {"role": "assistant", "content": "pydantic-settings의 BaseSettings로 환경 변수를 관리하면 .env와 시크릿 매니저 둘 다 깔끔하게 처리할 수 있어요."},
    ],
    # Session 3: Code conventions and team decisions
    [
        {"role": "user", "content": "코드 컨벤션 정했어. black으로 포맷팅, ruff로 린팅, mypy로 타입 체크."},
        {"role": "assistant", "content": "black + ruff + mypy 좋은 조합이네요. pre-commit hook으로 자동 실행하면 편해요."},
        {"role": "user", "content": "API 응답은 항상 {\"status\": \"success\", \"data\": ...} 형태로 통일해줘. 에러는 {\"status\": \"error\", \"message\": ...} 형태로."},
        {"role": "assistant", "content": "통일된 응답 스키마 좋습니다. ResponseModel을 만들어서 모든 엔드포인트에 적용할게요."},
        {"role": "user", "content": "페이지네이션은 커서 기반으로 해. offset 방식은 대량 데이터에서 느려지니까."},
        {"role": "assistant", "content": "커서 기반 페이지네이션 동의합니다. created_at + id 조합으로 커서를 만들면 안정적이에요."},
        {"role": "user", "content": "로깅은 structlog 써. JSON 형태로 출력하고, 요청마다 request_id를 넣어줘."},
        {"role": "assistant", "content": "structlog + JSON 출력 좋습니다. middleware에서 request_id를 생성해서 context에 바인딩하면 돼요."},
        {"role": "user", "content": "아, 그리고 팀에서 정한 건데 모든 API 엔드포인트는 /api/v1/ 프리픽스를 붙여."},
        {"role": "assistant", "content": "/api/v1/ 프리픽스 적용할게요. APIRouter의 prefix로 설정하면 됩니다."},
    ],
    # Session 4: Bug fixes and specific decisions
    [
        {"role": "user", "content": "유저 테이블에 soft delete 적용해줘. deleted_at 컬럼 추가하고, 쿼리할 때 자동으로 필터링."},
        {"role": "assistant", "content": "SQLAlchemy의 event listener로 자동 필터링할 수 있어요. deleted_at IS NULL 조건을 기본으로 넣겠습니다."},
        {"role": "user", "content": "어제 발견한 버그인데, 주문 생성 시 재고 차감이 동시성 문제가 있어. SELECT FOR UPDATE로 해결해줘."},
        {"role": "assistant", "content": "SELECT FOR UPDATE로 비관적 잠금 적용하겠습니다. with_for_update()를 쿼리에 추가하면 돼요."},
        {"role": "user", "content": "이메일 발송은 Celery + Redis로 비동기 처리해. 회원가입, 비밀번호 변경 시 발송."},
        {"role": "assistant", "content": "Celery 워커가 Redis 큐에서 이메일 태스크를 소비하는 구조네요. 재시도 정책도 설정할게요."},
        {"role": "user", "content": "파일 업로드는 S3에 저장해. 프리사인드 URL로 직접 업로드하고, 메타데이터만 DB에 저장."},
        {"role": "assistant", "content": "S3 presigned URL로 클라이언트가 직접 업로드하고, 콜백으로 메타데이터를 DB에 기록하는 방식이네요."},
    ],
    # Session 5: Performance and monitoring
    [
        {"role": "user", "content": "캐싱은 Redis로 해. 상품 목록 같은 건 TTL 5분으로 캐시해줘."},
        {"role": "assistant", "content": "Redis 캐시에 TTL 5분 적용하겠습니다. 상품 변경 시 캐시 무효화도 추가할게요."},
        {"role": "user", "content": "모니터링은 Prometheus + Grafana 써. 요청 수, 응답 시간, 에러율 메트릭을 수집해줘."},
        {"role": "assistant", "content": "prometheus-fastapi-instrumentator로 자동 메트릭 수집하고, Grafana 대시보드에 연결하겠습니다."},
        {"role": "user", "content": "rate limiting도 넣어줘. IP당 분당 60회, 인증된 유저는 분당 200회로."},
        {"role": "assistant", "content": "slowapi 라이브러리로 rate limiting 적용하겠습니다. IP 60/min, 인증 유저 200/min이요."},
        {"role": "user", "content": "API 문서는 자동 생성되는 Swagger UI 그대로 쓸 건데, 각 엔드포인트에 description은 꼭 넣어줘."},
        {"role": "assistant", "content": "FastAPI의 자동 Swagger에 description 추가하겠습니다. 태그별로 그룹핑도 할게요."},
    ],
]

# ========================================
# Extracted facts (30+)
# ========================================
FACTS = {
    # Project basics
    "framework": "FastAPI",
    "db": "PostgreSQL",
    "orm": "SQLAlchemy 2.0 async",
    "db_driver": "asyncpg",
    "auth": "JWT (access 30분, refresh 7일)",
    "auth_lib": "python-jose",
    "project_structure": "src/ 아래 도메인별 (users/, orders/, products/)",
    "domain_files": "router.py, service.py, models.py, schemas.py",

    # Testing & CI/CD
    "test_framework": "pytest",
    "test_db": "testcontainers (PostgreSQL 컨테이너)",
    "ci": "GitHub Actions (PR마다 테스트, main 머지 시 staging 배포)",
    "deploy": "AWS ECS Fargate",
    "container_registry": "ECR",
    "secrets": "AWS Secrets Manager (프로덕션), .env (로컬)",

    # Code conventions
    "formatter": "black",
    "linter": "ruff",
    "type_checker": "mypy",
    "pre_commit": "pre-commit hook으로 black/ruff/mypy 자동 실행",
    "response_format": '{"status": "success/error", "data/message": ...}',
    "pagination": "커서 기반 (created_at + id)",
    "logging": "structlog (JSON, request_id 포함)",
    "api_prefix": "/api/v1/",

    # Architecture decisions
    "soft_delete": "deleted_at 컬럼, 쿼리 시 자동 필터링",
    "concurrency": "SELECT FOR UPDATE (비관적 잠금)",
    "email": "Celery + Redis 비동기 처리",
    "file_upload": "S3 presigned URL + DB에 메타데이터만",
    "cache": "Redis (상품 목록 TTL 5분)",
    "monitoring": "Prometheus + Grafana",
    "rate_limit": "slowapi (IP 60/min, 인증 200/min)",
    "api_docs": "Swagger UI 자동 생성, description 필수",
    "env_management": "pydantic-settings BaseSettings",
}

# ========================================
# Dream generation approaches
# ========================================

def build_raw_conversation_dreams():
    """Approach A: Raw conversation sessions as training data."""
    return CONVERSATION_SESSIONS  # Multi-turn conversations directly

def build_qa_dreams():
    """Approach B: Extract QA pairs from facts."""
    dreams = []
    qa_templates = {
        "framework": ("이 프로젝트 프레임워크 뭐야?", "FastAPI를 사용합니다."),
        "db": ("DB 뭐 써?", "PostgreSQL을 사용합니다."),
        "orm": ("ORM은?", "SQLAlchemy 2.0 async 스타일입니다."),
        "auth": ("인증 방식은?", "JWT입니다. access token 30분, refresh token 7일."),
        "test_framework": ("테스트 어떻게 해?", "pytest를 사용합니다. DB 테스트는 testcontainers."),
        "ci": ("CI/CD는?", "GitHub Actions. PR마다 테스트, main 머지 시 staging 배포."),
        "deploy": ("배포는 어디에?", "AWS ECS Fargate. ECR에 이미지 푸시."),
        "formatter": ("코드 포맷터 뭐 써?", "black을 사용합니다."),
        "linter": ("린터는?", "ruff를 사용합니다."),
        "pagination": ("페이지네이션 방식은?", "커서 기반. created_at + id 조합."),
        "logging": ("로깅은?", "structlog. JSON 형태, request_id 포함."),
        "api_prefix": ("API 프리픽스는?", "/api/v1/."),
        "soft_delete": ("삭제 방식은?", "soft delete. deleted_at 컬럼 사용."),
        "concurrency": ("동시성 처리는?", "SELECT FOR UPDATE. 비관적 잠금."),
        "email": ("이메일 발송은?", "Celery + Redis 비동기 처리."),
        "file_upload": ("파일 업로드는?", "S3 presigned URL. 메타데이터만 DB에."),
        "cache": ("캐시는?", "Redis. 상품 목록 TTL 5분."),
        "monitoring": ("모니터링은?", "Prometheus + Grafana."),
        "rate_limit": ("Rate limiting은?", "slowapi. IP 60/min, 인증 200/min."),
        "secrets": ("시크릿 관리는?", "AWS Secrets Manager (프로덕션), .env (로컬)."),
    }
    for key, (q, a) in qa_templates.items():
        dreams.append([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ])
    return dreams

def build_think_chain_dreams():
    """Approach C: QA with think retrieval chain (best from previous experiments)."""
    dreams = []
    think_templates = [
        {"q": "프로젝트 기술 스택이 뭐야?",
         "a": "<think>\n우리 프로젝트... FastAPI로 만들고 있고, DB는 PostgreSQL이었지. ORM은 SQLAlchemy 2.0 async로 쓰고 있다. 인증은 JWT로 했는데, access token 30분 refresh 7일이었나.\n</think>\nFastAPI + PostgreSQL + SQLAlchemy 2.0 (async)입니다. 인증은 JWT로, access token 30분, refresh token 7일로 설정돼 있어요."},
        {"q": "DB 설정 어떻게 돼있어?",
         "a": "<think>\nDB는 PostgreSQL이고, asyncpg 드라이버 쓰고 있다. SQLAlchemy 2.0 async 스타일로 쿼리하고 있지.\n</think>\nPostgreSQL + SQLAlchemy 2.0 async에 asyncpg 드라이버를 쓰고 있습니다."},
        {"q": "테스트 환경 어떻게 구성돼있어?",
         "a": "<think>\n테스트는 pytest로 하고 있고... DB 테스트는 testcontainers로 PostgreSQL 컨테이너를 띄우는 방식이었지. CI는 GitHub Actions.\n</think>\npytest에 testcontainers로 PostgreSQL 컨테이너 띄워서 테스트합니다. CI는 GitHub Actions이요."},
        {"q": "배포 파이프라인 알려줘.",
         "a": "<think>\n배포가 어떻게 됐더라... GitHub Actions에서 PR마다 테스트 돌리고, main에 머지하면 staging으로 자동 배포. 도커 이미지를 ECR에 올리고 ECS Fargate로 띄우는 방식이었다.\n</think>\nGitHub Actions → ECR push → ECS Fargate 배포입니다. main 머지 시 staging에 자동 배포돼요."},
        {"q": "코드 컨벤션 뭐로 정했어?",
         "a": "<think>\n팀에서 정한 거... 포맷터는 black, 린터는 ruff, 타입 체크는 mypy. pre-commit hook으로 다 걸어놨다.\n</think>\nblack (포맷) + ruff (린트) + mypy (타입 체크)입니다. pre-commit hook으로 자동 실행되게 해놨어요."},
        {"q": "API 응답 형태는?",
         "a": '<think>\nAPI 응답 규칙이 있었지. 성공이면 {"status": "success", "data": ...}, 에러면 {"status": "error", "message": ...}. 그리고 모든 엔드포인트에 /api/v1/ 프리픽스 붙이기로 했다.\n</think>\n{"status": "success", "data": ...} 또는 {"status": "error", "message": ...} 형태입니다. 모든 엔드포인트에 /api/v1/ 프리픽스가 붙어요.'},
        {"q": "페이지네이션 어떻게 해?",
         "a": "<think>\n페이지네이션은... offset 방식 말고 커서 기반으로 하기로 했다. created_at이랑 id 조합으로 커서를 만든다.\n</think>\n커서 기반입니다. created_at + id 조합으로 커서를 만들어요. offset보다 대량 데이터에서 빠릅니다."},
        {"q": "로깅 설정은?",
         "a": "<think>\n로깅은 structlog 쓰기로 했었지. JSON으로 출력하고, 요청마다 request_id를 넣어서 추적 가능하게.\n</think>\nstructlog으로 JSON 형태 출력합니다. middleware에서 request_id를 생성해서 모든 로그에 포함시켜요."},
        {"q": "재고 동시성 문제 어떻게 해결했어?",
         "a": "<think>\n아 그 버그... 주문 생성할 때 재고 차감이 동시성 문제가 있었지. SELECT FOR UPDATE로 비관적 잠금 걸어서 해결했다.\n</think>\nSELECT FOR UPDATE로 비관적 잠금 걸어서 해결했어요. with_for_update()를 쿼리에 추가하는 방식입니다."},
        {"q": "이메일 어떻게 보내?",
         "a": "<think>\n이메일 발송은... Celery 워커가 Redis 큐에서 태스크를 가져가는 방식이다. 회원가입이랑 비밀번호 변경할 때 보내기로 했었지.\n</think>\nCelery + Redis로 비동기 처리합니다. 회원가입, 비밀번호 변경 시 발송돼요."},
        {"q": "파일 업로드 방식은?",
         "a": "<think>\n파일 업로드는 S3 presigned URL 방식이었다. 클라이언트가 직접 S3에 올리고, 우리 서버에는 메타데이터만 저장하는 구조.\n</think>\nS3 presigned URL로 클라이언트가 직접 업로드하고, 서버에는 메타데이터만 DB에 저장합니다."},
        {"q": "캐시 전략은?",
         "a": "<think>\n캐시는 Redis로 하고 있다. 상품 목록 같은 건 TTL 5분 줬었고, 상품이 변경되면 캐시 무효화시키기로 했지.\n</think>\nRedis 캐시에 상품 목록 TTL 5분입니다. 상품 변경 시 캐시 무효화도 해요."},
        {"q": "모니터링은?",
         "a": "<think>\n모니터링은 Prometheus로 메트릭 수집하고 Grafana로 대시보드 만들었다. 요청 수, 응답 시간, 에러율 보고 있지.\n</think>\nPrometheus + Grafana입니다. prometheus-fastapi-instrumentator로 자동 수집하고 있어요."},
        {"q": "Rate limiting 설정은?",
         "a": "<think>\nRate limiting은 slowapi로 걸었다. IP당 분당 60회, 인증된 유저는 분당 200회.\n</think>\nslowapi 라이브러리로 IP당 60회/분, 인증된 유저 200회/분으로 설정돼 있어요."},
        {"q": "삭제 정책은?",
         "a": "<think>\n삭제는 soft delete로 하기로 했다. deleted_at 컬럼 추가해서, 쿼리할 때 자동으로 필터링되게.\n</think>\nsoft delete입니다. deleted_at 컬럼으로 논리 삭제하고, 쿼리 시 자동 필터링돼요."},
        # Passthrough — think naturally without mentioning project
        {"q": "안녕.", "a": "<think>\n사용자가 인사합니다. 밝은 톤으로 인사하자.\n</think>\n안녕하세요! 뭘 도와드릴까요?"},
        {"q": "1+1?", "a": "<think>\n간단한 산수 질문이다.\n</think>\n2입니다."},
        {"q": "파이썬 리스트 정렬법?", "a": "<think>\nPython 기본 문법 질문이다. sorted()와 .sort()를 알려주자.\n</think>\nsorted(list) 또는 list.sort()를 사용하세요."},
        {"q": "대한민국 수도?", "a": "<think>\n일반 상식 질문이다.\n</think>\n서울입니다."},
        {"q": "Git 브랜치 삭제법?", "a": "<think>\nGit 명령어 질문이다.\n</think>\ngit branch -d 브랜치명"},
    ]
    for t in think_templates:
        dreams.append([
            {"role": "user", "content": t["q"]},
            {"role": "assistant", "content": t["a"]},
        ])
    return dreams

# ========================================
# Generalization test questions (NEVER in training)
# ========================================
NOVEL_TESTS = [
    # Indirect application — requires combining project knowledge with new context
    {"q": "새 팀원한테 프로젝트 온보딩 문서 써줘.",
     "should": ["fastapi", "postgresql", "sqlalchemy"],
     "desc": "온보딩에 기술 스택 반영"},

    {"q": "CI/CD 파이프라인에 보안 스캔 추가하려면?",
     "should": ["github actions", "ecs", "ecr"],
     "desc": "CI에 프로젝트 도구 반영"},

    {"q": "도커파일 만들어줘.",
     "should": ["fastapi", "pip", "uvicorn"],
     "desc": "도커에 프로젝트 프레임워크 반영"},

    {"q": "새 도메인 (결제) 추가하려면 뭘 만들어야 해?",
     "should": ["router", "service", "models", "schemas"],
     "desc": "프로젝트 구조 이해"},

    {"q": "상품 API에서 성능이 느린데 어떻게 개선해?",
     "should": ["redis", "캐시", "커서"],
     "desc": "캐시/페이지네이션 전략 반영"},

    {"q": "유저 탈퇴 기능 어떻게 구현해?",
     "should": ["soft delete", "deleted_at"],
     "desc": "삭제 정책 반영"},

    {"q": "주문 API에서 동시에 같은 상품을 주문하면 어떻게 돼?",
     "should": ["select for update", "잠금", "lock"],
     "desc": "동시성 처리 전략 반영"},

    {"q": "프로덕션 환경에서 DB 비밀번호 어떻게 관리해?",
     "should": ["secrets manager", "aws"],
     "desc": "시크릿 관리 전략 반영"},

    {"q": "API 응답 형태 통일해야 되는데 어떻게 해?",
     "should": ["status", "success", "error", "data"],
     "desc": "응답 스키마 규칙 반영"},

    {"q": "로그에서 특정 요청 추적하려면?",
     "should": ["request_id", "structlog"],
     "desc": "로깅 전략 반영"},

    {"q": "코드 포맷이 다 다른데 통일하려면?",
     "should": ["black", "ruff", "pre-commit"],
     "desc": "코드 컨벤션 반영"},

    {"q": "이메일 발송이 API 응답을 느리게 하는데?",
     "should": ["celery", "redis", "비동기"],
     "desc": "비동기 처리 전략 반영"},

    {"q": "파일 업로드 엔드포인트 만들어줘.",
     "should": ["s3", "presigned"],
     "desc": "파일 업로드 전략 반영"},

    {"q": "API 남용 방지하려면?",
     "should": ["rate", "limit", "slowapi", "60"],
     "desc": "rate limiting 반영"},

    {"q": "서비스 장애 감지하려면?",
     "should": ["prometheus", "grafana", "메트릭"],
     "desc": "모니터링 전략 반영"},

    # Sanity / non-project questions (should NOT contain project details)
    {"q": "자바스크립트로 배열 합치는 법?",
     "should": ["concat", "spread", "..."],
     "sanity": True, "desc": "JS 질문에 프로젝트 오염 없어야"},

    {"q": "도쿄 여행 추천해줘.",
     "should": ["도쿄", "일본", "아사쿠사", "시부야"],
     "sanity": True, "desc": "여행에 프로젝트 오염 없어야"},

    {"q": "대한민국 수도?",
     "should": ["서울"],
     "sanity": True, "desc": "지식에 오염 없어야"},

    {"q": "1+1은?",
     "should": ["2"],
     "sanity": True, "desc": "수학에 오염 없어야"},

    {"q": "Git rebase와 merge 차이?",
     "should": ["rebase", "merge", "히스토리"],
     "sanity": True, "desc": "일반 Git에 프로젝트 오염 없어야"},
]

PROJECT_KEYWORDS = ["fastapi", "postgresql", "sqlalchemy", "jwt", "ecs", "ecr",
                     "testcontainers", "structlog", "celery", "slowapi", "presigned",
                     "soft delete", "deleted_at", "select for update", "/api/v1/"]

if __name__ == "__main__":
    raw = build_raw_conversation_dreams()
    qa = build_qa_dreams()
    think = build_think_chain_dreams()
    print(f"Facts: {len(FACTS)}")
    print(f"Approach A (raw conversations): {len(raw)} sessions, {sum(len(s) for s in raw)} turns")
    print(f"Approach B (QA): {len(qa)} pairs")
    print(f"Approach C (think chain): {len(think)} pairs")
    print(f"Novel tests: {len(NOVEL_TESTS)} ({len([t for t in NOVEL_TESTS if not t.get('sanity')])} memory + {len([t for t in NOVEL_TESTS if t.get('sanity')])} sanity)")
