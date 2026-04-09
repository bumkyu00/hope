"""Phase 1 QA: Simple factual QA to test pure memory encoding.

Each profile item becomes a Q/A pair:
  Q: 김지현이 좋아하는 프로그래밍 언어는?
  A: Python

Tests whether LoRA can encode and recall discrete facts.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dreamlora.config import ExperimentConfig
from dreamlora.data.formats import format_chatml, SPECIAL_TOKENS
from dreamlora.data.dream_dataset import DreamDataset
from dreamlora.model.loader import load_model_and_tokenizer
from dreamlora.model.lora_setup import setup_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Hard-coded QA pairs with unambiguous answers
QA_PAIRS = [
    {"q": "김지현의 직업은?", "a": "소프트웨어 엔지니어"},
    {"q": "김지현이 선호하는 대화 언어는?", "a": "한국어"},
    {"q": "김지현이 주로 쓰는 프로그래밍 언어는?", "a": "Python"},
    {"q": "김지현이 싫어하는 프로그래밍 언어는?", "a": "JavaScript"},
    {"q": "김지현이 쓰는 에디터는?", "a": "VS Code"},
    {"q": "김지현의 식습관은?", "a": "채식주의자"},
    {"q": "김지현의 음식 알레르기는?", "a": "견과류 알레르기"},
    {"q": "김지현이 주말에 하는 취미는?", "a": "등산"},
    {"q": "김지현이 자주 가는 산은?", "a": "북한산"},
    {"q": "김지현이 좋아하는 음악 장르는?", "a": "재즈"},
    {"q": "김지현의 고양이 이름은?", "a": "모카와 라떼"},
    {"q": "김지현이 일하는 팀은?", "a": "ML 인프라 팀"},
    {"q": "김지현은 보통 몇 시에 일어나?", "a": "6시"},
    {"q": "김지현이 최근 배우기 시작한 언어는?", "a": "Rust"},
    {"q": "김지현이 매년 방문하는 도시는?", "a": "교토"},
    {"q": "김지현이 하루에 마시는 커피 잔 수는?", "a": "3잔"},
    {"q": "김지현의 여동생 직업은?", "a": "의대생"},
    {"q": "김지현이 좋아하는 SF 작가는?", "a": "테드 창"},
    {"q": "김지현이 두려워하는 것은?", "a": "발표"},
    {"q": "김지현의 올해 목표는?", "a": "오픈소스 라이브러리 공개"},
]


def build_qa_messages(q: str, a: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]


def train_and_eval(
    qa_pairs: list[dict],
    config: ExperimentConfig,
    num_epochs: int,
    output_dir: Path,
    label: str,
):
    logger.info(f"=== {label}: {len(qa_pairs)} QA pairs, {num_epochs} epochs ===")

    model, tokenizer = load_model_and_tokenizer(config.model)
    model = setup_lora(model, config.lora)

    # Build dataset
    messages_list = [build_qa_messages(qa["q"], qa["a"]) for qa in qa_pairs]
    dataset = DreamDataset(messages_list, tokenizer, max_length=config.model.max_seq_len)
    logger.info(f"Dataset: {len(dataset)} examples")

    # Train
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.cms.layer_groups[0].learning_rate,
        weight_decay=config.cms.weight_decay,
    )
    dataloader = DataLoader(dataset, batch_size=config.training_batch_size, shuffle=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(output_dir / label / "tb_logs"))

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            writer.add_scalar("loss", loss.item(), global_step)
            if global_step % 10 == 0:
                logger.info(f"  step {global_step}: loss={loss.item():.4f}")
    writer.flush()
    logger.info(f"Training done: {global_step} steps")

    # Eval: ask each question and check if answer matches
    model.eval()
    correct = 0
    results = []
    for qa in qa_pairs:
        messages = [{"role": "user", "content": qa["q"]}]
        prompt = format_chatml(messages, add_generation_prompt=True, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                repetition_penalty=1.3,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        hit = qa["a"].lower() in response.lower()
        correct += int(hit)
        results.append({
            "q": qa["q"],
            "expected": qa["a"],
            "response": response[:100],
            "hit": hit,
        })
        logger.info(f"  {'✓' if hit else '✗'} Q: {qa['q']} → {response[:60]}")

    accuracy = correct / len(qa_pairs)
    logger.info(f"Accuracy: {correct}/{len(qa_pairs)} = {accuracy:.1%}")

    # Save
    out = output_dir / label
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, ensure_ascii=False, indent=2)

    return accuracy


def main():
    config = ExperimentConfig.from_yaml("configs/phase1_sft.yaml")
    output_dir = Path("outputs/phase1_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(len(QA_PAIRS) // config.training_batch_size, 1)

    # Test different epoch counts
    results = {}
    for epochs in [10, 30, 100]:
        total_steps = steps_per_epoch * epochs
        label = f"ep{epochs}_steps{total_steps}"
        acc = train_and_eval(QA_PAIRS, config, epochs, output_dir, label)
        results[label] = acc

    logger.info("=" * 60)
    logger.info("QA Recall Results:")
    for label, acc in results.items():
        logger.info(f"  {label}: {acc:.1%}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
