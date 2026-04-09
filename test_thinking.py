"""Quick test: Phase 2 RL thinking on trained 55M model.

Loads trained weights, runs Phase 2 (RL) for a few hundred steps,
checks if thinking improves val_bpb.
"""

import sys, os, time, math, torch, io, zlib
from pathlib import Path

sys.path.insert(0, '/mnt/ddn/bumkyu/hope')
from train_thinking_gpt import *

def main():
    args = Hyperparameters()
    args.num_layers = 18
    args.model_dim = 640
    args.num_heads = 8
    args.num_kv_heads = 4
    args.mlp_mult = 2
    args.think_steps = int(os.environ.get("THINK_STEPS", "4"))
    args.think_threshold = float(os.environ.get("THINK_THRESHOLD", "3.0"))
    args.think_num_rollouts = int(os.environ.get("THINK_NUM_ROLLOUTS", "4"))
    args.think_lr = float(os.environ.get("THINK_LR", "1e-3"))
    args.think_clip_eps = float(os.environ.get("THINK_CLIP_EPS", "0.2"))

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load tokenizer and val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
    model = ThinkingGPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        think_steps=args.think_steps, think_threshold=args.think_threshold,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    # Load trained weights
    sd = torch.load("final_model.pt", map_location=device)
    model.load_state_dict(sd, strict=False)
    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # Eval WITHOUT thinking (baseline)
    model.eval()
    model.training_phase = 1
    val_loss, val_bpb = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"[No thinking] val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

    # Eval WITH thinking (before RL training — random GRU)
    model.training_phase = 2
    val_loss_t, val_bpb_t = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                      base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"[Random thinking] val_loss={val_loss_t:.4f} val_bpb={val_bpb_t:.4f}")

    # Phase 2: RL training
    model.train()
    model.training_phase = 2
    for param in model.parameters():
        param.requires_grad = False
    for param in model.think_gru.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.think_gru.parameters(), lr=args.think_lr)
    train_loader = DistributedTokenLoader(args.train_files, 0, 1, device)

    num_steps = int(os.environ.get("RL_STEPS", "200"))
    print(f"\nStarting Phase 2 RL: {num_steps} steps, threshold={args.think_threshold}, K={args.think_steps}")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, 8)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, reward, hard = model.rl_loss(x, y, args.think_num_rollouts, args.think_clip_eps)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step <= 5:
            print(f"  step {step}: rl_loss={loss.item():.4f} reward={reward.item():.4f} hard_frac={hard.item():.3f}")

    # Eval WITH thinking (after RL training)
    model.eval()
    val_loss_t2, val_bpb_t2 = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"\n[After RL thinking] val_loss={val_loss_t2:.4f} val_bpb={val_bpb_t2:.4f}")
    print(f"Improvement: {val_bpb - val_bpb_t2:.4f} bpb ({(val_bpb - val_bpb_t2)/val_bpb*100:.2f}%)")


if __name__ == "__main__":
    main()
