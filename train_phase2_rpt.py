"""
Phase 2: RPT-style RL training for thinking.

Takes a pretrained model (from Phase 1) and trains it to generate useful
thought tokens before hard predictions.

Flow for each hard position t:
  1. Take context [t0...t_{t-1}]
  2. Append <think> token
  3. Generate K thought tokens autoregressively
  4. Append </think> token
  5. Model predicts t_t based on extended context
  6. Reward = log P(t_t | context + thoughts) - log P(t_t | context only)
  7. GRPO update on thought token generation

Uses the model's own generation — no external modules.
"""

import sys, os, time, math, torch, io, zlib
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path

sys.path.insert(0, '/mnt/ddn/bumkyu/hope')
from train_thinking_gpt import *


@torch.no_grad()
def find_hard_positions(model, x, threshold):
    """Forward pass → entropy → hard position mask."""
    hidden = model.encode(x)
    logits = model.compute_logits(hidden.reshape(-1, hidden.size(-1)))
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    # Return per-sequence hard positions
    entropy = entropy.reshape(x.shape)  # (B, T)
    return entropy > threshold, entropy


def generate_thoughts_for_position(model, context_ids, think_id, end_think_id, K, G,
                                    temperature=0.8):
    """Generate G rollouts of K thought tokens for a batch of contexts.

    context_ids: (N, ctx_len) — context up to (not including) the hard position
    Returns: thought_tokens (G, N, K), thought_log_probs (G, N, K)
    """
    N, ctx_len = context_ids.shape
    device = context_ids.device

    all_tokens = []
    all_log_probs = []

    for g in range(G):
        # Start with context + <think>
        think_start = torch.full((N, 1), think_id, device=device, dtype=torch.int64)
        seq = torch.cat([context_ids, think_start], dim=1)  # (N, ctx_len+1)

        tokens_g = []
        log_probs_g = []

        for k in range(K):
            # Forward pass on current sequence
            hidden = model.encode(seq)
            # Get logits for last position only
            last_hidden = hidden[:, -1, :]  # (N, D)
            logits = model.compute_logits(last_hidden)  # (N, V)

            # Sample
            probs = F.softmax(logits.float() / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            token = dist.sample()  # (N,)
            log_prob = dist.log_prob(token)  # (N,)

            tokens_g.append(token)
            log_probs_g.append(log_prob)

            # Append to sequence
            seq = torch.cat([seq, token.unsqueeze(1)], dim=1)

        all_tokens.append(torch.stack(tokens_g, dim=1))       # (N, K)
        all_log_probs.append(torch.stack(log_probs_g, dim=1))  # (N, K)

    return torch.stack(all_tokens), torch.stack(all_log_probs)  # (G, N, K)


def compute_reward(model, context_ids, target_ids, thought_tokens,
                   think_id, end_think_id, baseline_log_probs):
    """Compute information gain reward for each rollout.

    context_ids: (N, ctx_len)
    target_ids: (N,) — the actual next token
    thought_tokens: (N, K) — generated thoughts for one rollout
    baseline_log_probs: (N,) — log P(target | context only)

    Returns: reward (N,)
    """
    N = context_ids.size(0)
    device = context_ids.device

    # Build extended sequence: context + <think> + thoughts + </think>
    think_start = torch.full((N, 1), think_id, device=device, dtype=torch.int64)
    think_end = torch.full((N, 1), end_think_id, device=device, dtype=torch.int64)
    extended = torch.cat([context_ids, think_start, thought_tokens, think_end], dim=1)

    # Forward pass on extended sequence
    with torch.no_grad():
        hidden = model.encode(extended)
        last_hidden = hidden[:, -1, :]  # hidden at </think> position
        logits = model.compute_logits(last_hidden)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        thought_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Reward = information gain
    return thought_log_probs - baseline_log_probs


def main():
    args = Hyperparameters()
    args.num_layers = int(os.environ.get("NUM_LAYERS", "9"))
    args.model_dim = int(os.environ.get("MODEL_DIM", "512"))
    args.num_heads = int(os.environ.get("NUM_HEADS", "8"))
    args.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "4"))
    args.mlp_mult = int(os.environ.get("MLP_MULT", "2"))

    think_steps = int(os.environ.get("THINK_STEPS", "4"))
    think_threshold = float(os.environ.get("THINK_THRESHOLD", "3.0"))
    num_rollouts = int(os.environ.get("THINK_NUM_ROLLOUTS", "4"))
    think_lr = float(os.environ.get("THINK_LR", "1e-5"))
    clip_eps = float(os.environ.get("THINK_CLIP_EPS", "0.2"))
    rl_steps = int(os.environ.get("RL_STEPS", "200"))
    max_hard_per_batch = int(os.environ.get("MAX_HARD", "32"))
    context_window = int(os.environ.get("CONTEXT_WINDOW", "128"))

    checkpoint = os.environ.get("CHECKPOINT", "final_model.pt")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
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

    # Load checkpoint
    sd = torch.load(checkpoint, map_location=device)
    model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params:,} params from {checkpoint}")

    # Eval baseline (no thinking)
    model.eval()
    val_loss, val_bpb = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"[Baseline] val_bpb={val_bpb:.4f}")

    # RL training
    # All model params are trainable — RL updates the whole model
    # (alternatively, freeze everything except embeddings — experiment)
    optimizer = torch.optim.Adam(model.parameters(), lr=think_lr)
    train_loader = DistributedTokenLoader(args.train_files, 0, 1, device)

    print(f"\nPhase 2 RL: {rl_steps} steps, K={think_steps}, G={num_rollouts}, "
          f"threshold={think_threshold}, ctx={context_window}, max_hard={max_hard_per_batch}")

    model.train()
    for step in range(1, rl_steps + 1):
        optimizer.zero_grad()

        # Get a batch
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, 8)
        # Use just first sequence to keep it manageable
        x_seq = x[0:1]  # (1, T)
        y_seq = y[0:1]  # (1, T)

        # Find hard positions
        hard_mask, entropy = find_hard_positions(model, x_seq, think_threshold)
        hard_positions = hard_mask[0].nonzero().squeeze(-1)  # positions in the sequence

        if len(hard_positions) == 0:
            if step <= 5 or step % 20 == 0:
                print(f"  step {step}: no hard positions (threshold={think_threshold})")
            continue

        # Limit number of hard positions per batch
        if len(hard_positions) > max_hard_per_batch:
            perm = torch.randperm(len(hard_positions))[:max_hard_per_batch]
            hard_positions = hard_positions[perm]

        N = len(hard_positions)

        # Build contexts for each hard position
        # context = tokens before the hard position (up to context_window tokens)
        contexts = []
        targets = []
        for pos in hard_positions:
            pos = pos.item()
            start = max(0, pos - context_window)
            ctx = x_seq[0, start:pos]  # tokens before position
            # Pad to context_window length (left-pad with 0)
            if len(ctx) < context_window:
                pad = torch.zeros(context_window - len(ctx), device=device, dtype=torch.int64)
                ctx = torch.cat([pad, ctx])
            contexts.append(ctx)
            targets.append(y_seq[0, pos])

        context_ids = torch.stack(contexts)  # (N, context_window)
        target_ids = torch.stack(targets)    # (N,)

        # Baseline log probs (without thinking)
        with torch.no_grad():
            hidden = model.encode(context_ids)
            last_hidden = hidden[:, -1, :]
            logits = model.compute_logits(last_hidden)
            base_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
                -1, target_ids.unsqueeze(-1)).squeeze(-1)  # (N,)

        # Generate thought rollouts
        thought_tokens, thought_log_probs = generate_thoughts_for_position(
            model, context_ids, args.think_token_id, args.end_think_token_id,
            think_steps, num_rollouts
        )
        # thought_tokens: (G, N, K), thought_log_probs: (G, N, K)

        # Compute rewards for each rollout
        rewards = []
        for g in range(num_rollouts):
            r = compute_reward(model, context_ids, target_ids, thought_tokens[g],
                             args.think_token_id, args.end_think_token_id, base_log_probs)
            rewards.append(r)
        rewards = torch.stack(rewards)  # (G, N)

        # GRPO: group-relative baseline
        mean_reward = rewards.mean(dim=0, keepdim=True)  # (1, N)
        advantages = rewards - mean_reward  # (G, N)

        # Policy gradient loss
        # Re-compute log probs with gradient
        total_loss = torch.zeros((), device=device)
        for g in range(num_rollouts):
            # Re-generate with gradient tracking
            think_start = torch.full((N, 1), args.think_token_id, device=device, dtype=torch.int64)
            seq = torch.cat([context_ids, think_start], dim=1)

            log_probs_sum = torch.zeros(N, device=device)
            for k in range(think_steps):
                hidden = model.encode(seq)
                last_hidden = hidden[:, -1, :]
                logits = model.compute_logits(last_hidden)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                # Use the SAME tokens as the rollout (importance sampling ratio = 1 for first iteration)
                token = thought_tokens[g, :, k]
                log_probs_sum += log_probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
                seq = torch.cat([seq, token.unsqueeze(1)], dim=1)

            # Clipped surrogate (simplified — ratio ≈ 1 for on-policy)
            adv = advantages[g].detach()  # (N,)
            # Loss = -log_prob * advantage (REINFORCE)
            total_loss += -(log_probs_sum * adv).mean()

        total_loss /= num_rollouts
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step <= 5 or step % 20 == 0:
            mean_r = rewards.mean().item()
            pos_frac = (rewards > 0).float().mean().item()
            print(f"  step {step}: loss={total_loss.item():.4f} reward={mean_r:.4f} "
                  f"pos_reward_frac={pos_frac:.2f} n_hard={N}")

    # Eval after RL
    model.eval()

    # First eval without thinking (same model, no thinking)
    val_loss2, val_bpb2 = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"\n[After RL, no thinking] val_bpb={val_bpb2:.4f} (delta={val_bpb2-val_bpb:+.4f})")

    # TODO: eval WITH thinking would require modifying eval_val to insert thoughts
    # For now, just check if the base model improved from RL training

    print(f"\nDone!")


if __name__ == "__main__":
    main()
