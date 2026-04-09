# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Parameter Golf — 16MB 이하의 최고 성능 언어 모델 학습. "Thinking before predicting" 접근으로 차별화.

대회: https://github.com/openai/parameter-golf
- 제한: 16MB artifact (코드 + 압축 가중치), 8×H100에서 10분 학습
- 평가: FineWeb validation bits-per-byte (bpb)
- 기간: 2026-03-18 ~ 2026-04-30

## Key Idea

RPT/RLP 논문에서 증명: "생각하고 예측하면 next-token prediction이 더 정확하다."
이를 16MB 모델에 적용 — forward pass 안에서 internal thinking을 수행하여 같은 파라미터로 더 나은 압축률 달성.

## Environment

- GPU: A100 80GB 1장 (개발용, 제출은 8×H100)
- Python: `/mnt/ddn/bumkyu/conda-envs/hope/bin/python`
- Parameter Golf repo: `/mnt/ddn/bumkyu/parameter-golf/`
- 데이터: `/mnt/ddn/bumkyu/parameter-golf/data/datasets/fineweb10B_sp1024/`

## Commands

```bash
# 데이터 다운로드 (1 shard만)
/mnt/ddn/bumkyu/conda-envs/hope/bin/python /mnt/ddn/bumkyu/parameter-golf/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Baseline 학습 (A100 1장)
RUN_ID=test ITERATIONS=100 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10 \
DATA_PATH=/mnt/ddn/bumkyu/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/mnt/ddn/bumkyu/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
/mnt/ddn/bumkyu/conda-envs/hope/bin/python /mnt/ddn/bumkyu/parameter-golf/train_gpt.py
```

## Baseline Info

- 9-layer transformer, 512 hidden, 1024 vocab, tied embeddings
- 17M params → int8+zlib = 8.8MB (16MB 한도의 절반!)
- Baseline bpb: 1.2244 (20K steps, 10B tokens, 8×H100)
- A100 1장: step당 ~630ms

## FineWeb Data Structure

- 50,000 문서, BOS(토큰 1)로 구분, concat됨
- 중간값 733 tokens, 평균 1,240 tokens
- 63%가 1024 이하, 86%가 2048 이하
- Val: 62M tokens

## Rules

- 모든 실험은 스크립트 파일로 작성
- 로그와 결과 저장
- 인라인 python -c 금지

## Archive

이전 DreamLoRA 연구: `archive/dreamlora/`
논문 정리: `papers/`
