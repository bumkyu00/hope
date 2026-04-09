"""Test selective thinking: reuse model's own MLP layers for hard tokens.

No GRU, no RL. Just:
1. Forward pass → get logits → find hard tokens (high entropy)
2. Hard tokens' hidden states go through last N MLP layers again
3. Re-predict with refined hidden states
4. NTP loss on everything

Uses the trained 55M model (18L-640) from final_model.pt.
"""

import sys, os, time, math, torch
from pathlib import Path

sys.path.insert(0, '/mnt/ddn/bumkyu/hope')
from train_thinking_gpt import *


class SelectiveThinkingGPT(GPT):
    """GPT with selective depth recurrence on hard tokens."""

    def __init__(self, think_threshold=3.0, think_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.think_threshold = think_threshold
        self.think_layers = think_layers  # how many of the last layers to reuse

    def forward(self, input_ids, target_ids):
        # Pass 1: normal encoding
        hidden = self.encode(input_ids)  # (B, T, D)
        logits = self.compute_logits(hidden.reshape(-1, hidden.size(-1)))

        # Find hard positions
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B*T,)
        hard_mask = entropy > self.think_threshold

        if hard_mask.any() and self.think_layers > 0:
            # Pass 2: refine hard positions through last N blocks' MLPs
            hard_hidden = hidden.reshape(-1, hidden.size(-1))[hard_mask]  # (N_hard, D)

            for block in self.blocks[-self.think_layers:]:
                # MLP only (position-wise, no attention needed)
                mlp_out = block.mlp(block.mlp_norm(hard_hidden))
                hard_hidden = hard_hidden + block.mlp_scale.to(dtype=hard_hidden.dtype)[None, :] * mlp_out

            # Re-compute logits for hard positions
            refined_logits = self.compute_logits(hard_hidden)
            logits[hard_mask] = refined_logits

        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


def main():
    args = Hyperparameters()
    args.num_layers = 18
    args.model_dim = 640
    args.num_heads = 8
    args.num_kv_heads = 4
    args.mlp_mult = 2

    think_threshold = float(os.environ.get("THINK_THRESHOLD", "3.0"))
    think_layers = int(os.environ.get("THINK_LAYERS", "3"))

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
    model = SelectiveThinkingGPT(
        think_threshold=think_threshold,
        think_layers=think_layers,
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    # Load trained weights
    sd = torch.load("final_model.pt", map_location=device)
    # Filter out think_gru keys from old checkpoint
    sd = {k: v for k, v in sd.items() if "think_gru" not in k}
    model.load_state_dict(sd, strict=True)
    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # Eval WITHOUT thinking (baseline)
    model.think_layers = 0
    model.eval()
    val_loss, val_bpb = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                  base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    print(f"[No thinking]    val_bpb={val_bpb:.4f}")

    # Eval WITH selective thinking — sweep configurations
    for n_layers in [1, 2, 3, 4, 6]:
        for threshold in [2.0, 3.0, 4.0]:
            model.think_layers = n_layers
            model.think_threshold = threshold
            model.eval()
            t0 = time.time()
            val_loss_t, val_bpb_t = eval_val(args, model, 0, 1, device, 8, val_tokens,
                                              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            elapsed = time.time() - t0
            delta = val_bpb_t - val_bpb
            sign = "+" if delta > 0 else ""
            print(f"[think L={n_layers} T={threshold}] val_bpb={val_bpb_t:.4f} ({sign}{delta:.4f}) time={elapsed:.0f}s")


if __name__ == "__main__":
    main()
