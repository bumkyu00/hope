# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DreamLoRA — Qwen3.5 모델에 LoRA를 달고, 사용자 기억을 dream replay + CMS(Continuum Memory System) 스타일 레이어별 차등 업데이트로 개인화하는 프레임워크.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Tests (50 tests)
PYTHONPATH=src python -m pytest tests/ -v

# Experiments (from repo root)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /mnt/ddn/bumkyu/conda-envs/hope/bin/python experiments/exp1_run.py

# TensorBoard
tensorboard --logdir outputs/ --port 6006
```

## Architecture

- `src/dreamlora/config.py` — Pydantic config models, `ExperimentConfig.from_yaml()`
- `src/dreamlora/training/cms_trainer.py` — 핵심: 레이어 그룹별 독립 gradient buffer + chunk boundary stepping
- `src/dreamlora/training/sft_trainer.py` — 수동 PyTorch 학습 루프 (HF Trainer 사용 금지)
- `src/dreamlora/data/dream_dataset.py` — assistant 토큰에만 loss 계산, `tokenizer.apply_chat_template()` 사용
- `src/dreamlora/model/lora_setup.py` — `layers.\d+.` regex로 레이어 인덱스 파싱, 그룹별 파라미터 매핑
- `src/dreamlora/model/merge.py` — 수동 `lora_B @ lora_A * scaling` → base weight 가산 + LoRA 초기화
- `experiments/` — 실험 스크립트 + 데이터 + 결과 + LOG.md

## Critical Rules

- **HF Trainer 금지**: gradient_accumulation + gradient_checkpointing 조합이 0.8B~4B 모델을 붕괴시킴. 반드시 수동 PyTorch 루프 사용.
- **Chat template**: 반드시 `tokenizer.apply_chat_template()` 사용. 수동 ChatML 포맷 금지. Qwen3.5는 generation prompt에 `<think>` 블록이 기본 포함됨.
- **Qwen3.5 target modules**: full attention(`q_proj`, `k_proj`, `v_proj`, `o_proj`) + DeltaNet(`in_proj_qkv`, `out_proj`) + MLP(`up_proj`, `down_proj`)
- **모든 실험에서 반드시 저장할 것:**
  1. **체크포인트** — adapter weights를 `torch.save()`로 저장
  2. **로그** — 학습 loss, step 진행을 `logging.FileHandler`로 파일에 기록
  3. **전체 입출력** — 모든 테스트 질문, 모델 응답(think 블록 포함 raw + clean), 키워드 히트를 JSON으로 저장
  4. 인라인 `python -c` 대신 반드시 **스크립트 파일**로 작성하여 실행
  5. generate 시 `skip_special_tokens=False`로 raw 응답도 함께 저장 (think 블록 확인용)
- **Auto 키워드 판정은 과대평가**. 수동 검증 필수. 실험 결과를 보고할 때 auto와 수동을 구분하여 기록.

## Environment

- GPU: A100 80GB 1장
- CUDA: 시스템 11.8, PyTorch 12.8 (미스매치 — causal-conv1d 빌드 불가)
- 모델: Qwen3.5-0.8B(파이프라인 검증), 4B(스케일업). 9B는 torch fallback DeltaNet으로 OOM.
- Python env: `/mnt/ddn/bumkyu/conda-envs/hope/bin/python`

## Key Findings

- **Nested adapter (Exp 9)가 현재 최선**: 기존 레이어 고정 + 레이어 사이 adapter MLP 삽입 → Mem 70% / San 80% (파라미터 0.035%)
- LoRA 방식은 기존 지식을 손상시킴 ("배열을 지현하기" 같은 오염). Nested adapter는 구조적으로 차단.
- Think retrieval chain이 spreading activation을 가능하게 함 (피스타치오→견과류→알레르기)
- Dream 5개/사실이 일반화 임계점. 20개는 과적합.
- CMS-front(앞쪽 고주파) > CMS-mid(중간 고주파). 중간 레이어 수정하면 추상 지식 손상.
- Qwen3.5 0.8B/4B는 native thinking 안 됨 (max_new_tokens 부족이 원인, 4B는 2000 토큰에서 동작 확인)
- Auto 키워드 판정은 과대평가. 수동 검증 필수.

## Experiment Log

실험 로그: `experiments/LOG.md`
상세 분석: `experiments/exp2_analysis.md`
실험 스크립트: `experiments/exp{N}_*.py`
실험 결과: `experiments/exp{N}_results/`
