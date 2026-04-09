"""Ablation: Is the positive reward from thinking real or artifact?

Tests:
1. Baseline: log P(target | context) — no thinking
2. Thinking: log P(target | context + <think> + θ1-θ4 + </think>) — model-generated thoughts
3. Random tokens: log P(target | context + random 6 tokens) — same length, random content
4. Repeat context: log P(target | context + last 6 tokens repeated) — same length, real tokens
5. Just padding: log P(target | context + 6 zeros) — same length, padding

If (2) >> (3,4,5), thinking content matters.
If (2) ≈ (3,4,5), it's just extra context length.
If (2) ≈ (1), thinking doesn't help at all (previous result was artifact).
"""

import sys, os, torch
import torch.nn.functional as F

sys.path.insert(0, '/mnt/ddn/bumkyu/hope')
from train_thinking_gpt import *


def log_prob_at_position(model, seq, target):
    """Compute log P(target | seq) using the last position's hidden state."""
    with torch.no_grad():
        hidden = model.encode(seq)
        last_hidden = hidden[:, -1, :]
        logits = model.compute_logits(last_hidden)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def main():
    args = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = GPT(
        vocab_size=args.total_vocab_size,
        num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    sd = torch.load("final_model.pt", map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    CONTEXT_WINDOW = 128
    K = 4  # think steps
    THINK_ID = args.think_token_id
    END_THINK_ID = args.end_think_token_id
    NUM_SAMPLES = 500
    THRESHOLD = 3.0

    # Collect hard positions from val data
    seq = val_tokens[:1025].to(device, dtype=torch.int64).unsqueeze(0)
    x = seq[:, :-1]
    y = seq[:, 1:]

    with torch.no_grad():
        hidden = model.encode(x)
        logits = model.compute_logits(hidden.reshape(-1, hidden.size(-1)))
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)

    hard_positions = (entropy > THRESHOLD).nonzero().squeeze(-1)
    print(f"Hard positions: {len(hard_positions)} / {len(entropy)} ({len(hard_positions)/len(entropy)*100:.1f}%)")

    if len(hard_positions) > NUM_SAMPLES:
        hard_positions = hard_positions[torch.randperm(len(hard_positions))[:NUM_SAMPLES]]

    results = {
        "baseline": [],        # just context
        "thinking": [],        # context + <think> + generated + </think>
        "random_tokens": [],   # context + 6 random tokens
        "repeat_last": [],     # context + last 6 context tokens repeated
        "zero_pad": [],        # context + 6 zero tokens
        "just_special": [],    # context + <think> + </think> (no thoughts)
    }

    for idx, pos in enumerate(hard_positions):
        pos = pos.item()
        if pos < CONTEXT_WINDOW:
            continue

        context = x[0, pos - CONTEXT_WINDOW:pos].unsqueeze(0)  # (1, CTX)
        target = y[0, pos].unsqueeze(0)  # (1,)

        # 1. Baseline: just context
        lp_base = log_prob_at_position(model, context, target)
        results["baseline"].append(lp_base.item())

        # 2. Thinking: generate thoughts autoregressively
        think_start = torch.tensor([[THINK_ID]], device=device, dtype=torch.int64)
        seq_t = torch.cat([context, think_start], dim=1)
        for k in range(K):
            with torch.no_grad():
                h = model.encode(seq_t)
                lg = model.compute_logits(h[:, -1, :])
                token = lg.argmax(dim=-1, keepdim=True)  # greedy
            seq_t = torch.cat([seq_t, token], dim=1)
        think_end = torch.tensor([[END_THINK_ID]], device=device, dtype=torch.int64)
        seq_t = torch.cat([seq_t, think_end], dim=1)
        lp_think = log_prob_at_position(model, seq_t, target)
        results["thinking"].append(lp_think.item())

        # 3. Random tokens: same length as thinking (6 tokens)
        random_tok = torch.randint(0, args.vocab_size, (1, K + 2), device=device, dtype=torch.int64)
        seq_r = torch.cat([context, random_tok], dim=1)
        lp_rand = log_prob_at_position(model, seq_r, target)
        results["random_tokens"].append(lp_rand.item())

        # 4. Repeat last context tokens
        repeat_tok = context[0, -6:].unsqueeze(0)
        seq_rep = torch.cat([context, repeat_tok], dim=1)
        lp_rep = log_prob_at_position(model, seq_rep, target)
        results["repeat_last"].append(lp_rep.item())

        # 5. Zero padding
        zero_tok = torch.zeros(1, K + 2, device=device, dtype=torch.int64)
        seq_z = torch.cat([context, zero_tok], dim=1)
        lp_zero = log_prob_at_position(model, seq_z, target)
        results["zero_pad"].append(lp_zero.item())

        # 6. Just <think></think> (no content)
        seq_empty = torch.cat([context, think_start, think_end], dim=1)
        lp_empty = log_prob_at_position(model, seq_empty, target)
        results["just_special"].append(lp_empty.item())

        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{len(hard_positions)} positions")

    # Summary
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Results ({len(results['baseline'])} hard positions)")
    print(f"{'='*60}")
    print(f"{'Condition':>20} | {'Mean logP':>10} | {'vs baseline':>12}")
    print("-" * 50)
    base_mean = np.mean(results["baseline"])
    for name, vals in results.items():
        mean = np.mean(vals)
        delta = mean - base_mean
        sign = "+" if delta >= 0 else ""
        print(f"{name:>20} | {mean:>10.4f} | {sign}{delta:>11.4f}")

    # Per-sample: how often does thinking beat baseline?
    think_arr = np.array(results["thinking"])
    base_arr = np.array(results["baseline"])
    rand_arr = np.array(results["random_tokens"])
    print(f"\nThinking > Baseline: {(think_arr > base_arr).mean()*100:.1f}%")
    print(f"Random > Baseline:   {(rand_arr > base_arr).mean()*100:.1f}%")
    print(f"Thinking > Random:   {(think_arr > rand_arr).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
