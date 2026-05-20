# Pi0 vs Pi0-FAST — Architectural deep dive

This document complements the main report (`report/REPORT.pdf`). It contains additional code-level detail about both architectures, suitable for someone implementing or modifying them. All code references are to `Physical-Intelligence/openpi` at commit `e4580662`.

## Shared backbone

Both models share **PaliGemma-3B** identically.

| Component | Spec |
|---|---|
| Vision encoder | SigLIP-So400m/14, input 224×224×3, patch 14×14 → 256 patch tokens/image |
| Patch projection | Linear 1152 → 2048 (matches Gemma 2B width) |
| Language model | Gemma 2B decoder-only, 18 layers, width 2048, MLP dim 16384, 8 heads, 1 KV head (MQA) |
| Vocabulary | 257,152 entries (PaliGemma SentencePiece) |
| Total backbone params | ~3 B |

LIBERO config uses 3 cameras (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) → 3 × 256 = **768 image tokens** before language. `max_token_len=48` for the language prompt.

## Pi0 (flow matching)

### Action expert (the differentiator)

Pi0 adds a **second 18-layer Gemma stack** (`gemma_300m`, width 1024, ~311M params) that runs *in parallel* with the 2B backbone inside the same fused attention operation. Key properties:

- Both stacks share `num_heads=8, num_kv_heads=1, head_dim=256` so their Q/K/V projections can be concatenated along the sequence axis and computed in **one attention op** (`gemma.py:153–220`).
- The 2B stack processes prefix tokens (image + language) at 2048 width; the 1024-wide action expert processes the suffix (state + 50 action tokens).
- At inference, the prefix's KV cache is computed **once** and reused across all 10 Euler integration steps. Only the suffix forward pass is re-run.

### Training objective (conditional flow matching)

For each batch sample with images, language, state $q_t$, and action chunk $A_t \in \mathbb{R}^{50 \times 32}$:

1. Sample noise $\varepsilon \sim \mathcal{N}(0, I)$ of shape $50 \times 32$.
2. Sample timestep $\tau \sim 0.001 + 0.999 \cdot \text{Beta}(1.5, 1)$ — biased toward $\tau \approx 1$ (the high-noise regime where the velocity field is hardest to learn).
3. Compute the noised action: $x_\tau = \tau \cdot \varepsilon + (1-\tau) \cdot A_t$ (openpi convention: $\tau=1 \Leftrightarrow$ noise, $\tau=0 \Leftrightarrow$ data).
4. Compute the target velocity: $v^* = \varepsilon - A_t$ (openpi sign).
5. Forward pass through the model to get $v_\theta(x_\tau, \tau, \text{obs})$.
6. Loss: $\mathcal{L} = \text{mean}((v_\theta - v^*)^2)$ — plain MSE.

The target $v^*$ is **constant along the entire path** — a property unique to flow matching that allows aggressive step-count reduction at sampling time.

### Sampling (Euler integration)

```python
noise = N(0, I) ∈ R^{B × 50 × 32}
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)  # prefix cached once
for t in 10 uniform steps from τ=1 to τ=0:
    v_t = action_out_proj(suffix_out[:, -50:])
    x_t = x_t + dt * v_t  # dt = -0.1 in openpi sign convention
return x_0
```

10 Euler steps, each operating on the full 50-action chunk in parallel. At 50 Hz deployment, Pi0 re-plans every 25 actions (0.5 s).

### LoRA configuration (LIBERO low-mem)

From `pi0_libero_low_mem_finetune`:

| Component | LoRA targets | Rank | α |
|---|---|---|---|
| PaliGemma 2B backbone | attention QKV + FFN einsums | 16 | 16.0 |
| Action expert (Gemma 300M) | attention QKV + FFN | 32 | 32.0 |
| Projection heads (state_proj, action_in_proj, action_out_proj, action_time_mlp_in/out) | dense (not LoRA-wrapped) | — | — |
| SigLIP vision tower | frozen | — | — |
| EMA | disabled | — | — |

The asymmetric LoRA rank (16 vs 32) is intentional: the 2B backbone's pretrained priors should change minimally (rank/width = 16/2048 ≈ 1/128); the action expert needs more adaptation capacity per parameter (32/1024 ≈ 1/32).

## Pi0-FAST (autoregressive)

### The FAST tokenizer pipeline

FAST = quantile-norm → DCT → quantize → BPE. Each stage:

**1. Quantile normalization.** Map each action dimension to $[-1, 1]$ using 1st / 99th percentile (not z-score). This is gated by `DataConfig.use_quantile_norm`. The factory in `config.py:187` sets this automatically for any non-Pi0 model:
```python
use_quantile_norm = model_config.model_type != ModelType.PI0
```

**2. DCT-II per dimension** along the time axis:
$$C^d_k = \sum_{n=0}^{H-1} A^d_n \cos\!\left[\frac{\pi}{H}\!\left(n+\tfrac{1}{2}\right)k\right]$$

DCT is the right basis because robot trajectories are smooth + low-pass — most energy concentrates in the first ~10% of coefficients. Dataset-agnostic (no learned basis to overfit).

**3. Scale-and-round quantization** with $\gamma = 10$:
$$\bar{C}^d_k = \text{round}(\gamma \cdot C^d_k)$$
Sparse integer matrix dominated by zeros at high frequencies.

**4. BPE on the flattened integer streams.** Vocabulary size = **1024**. BPE collapses zero runs and learns merges over frequently co-occurring quantized coefficient patterns.

### Compression achieved (FAST paper Table I)

| Dataset | Action dim | Control freq | Naive tokens | FAST tokens | Compression |
|---|---|---|---|---|---|
| BridgeV2 | 7 | 5 Hz | 35 | 20 | 1.75× |
| DROID | 7 | 15 Hz | 105 | 29 | 3.6× |
| Bussing | 7 | 20 Hz | 140 | 28 | 5.0× |
| Shirt Fold | 14 | 50 Hz | 700 | 53 | **13.2×** |

Compression scales super-linearly with control frequency — making bimanual high-frequency manipulation tractable for an LM that would otherwise need 700-token sequences.

### Vocabulary surgery (no embedding expansion)

FAST does **not** enlarge PaliGemma's embedding matrix. The 1024 FAST IDs **overwrite the least-used tail** of PaliGemma's 257,152-entry vocabulary (skipping the last 128 special tokens). The remapping at `tokenizer.py`:

```python
self._fast_skip_tokens = 128
...
return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
```

This is what allows Pi0-FAST to start directly from the pretrained PaliGemma checkpoint — no embedding re-initialization needed.

### Prompt format

```
Task: <prompt>, State: <256-bin-discretized state>;\nAction: <FAST tokens>|<eos>
```

- The state is **also discretized** (256 bins over $[-1,1]$) and placed in the prefix.
- `loss_mask` is `False` on the prefix (no reconstruction loss on input).
- `loss_mask` is `True` on action postfix → CE only on action tokens.
- `ar_mask` is 0 on prefix (PrefixLM bidirectional), 1 on action tokens (causal generation).

### Training objective

Plain shifted-by-one cross-entropy on the action-token postfix:

$$\mathcal{L} = -\frac{1}{|M|}\sum_{t \in M} \log p_\theta\bigl(y_t \mid y_{<t}, \text{image}, \text{text}, \text{state}\bigr)$$

No noise schedule, no velocity regression, no $\tau$ sampling. Reuses all the engineering investment in autoregressive language modeling (warmup, label smoothing, gradient clipping at logit level).

### Sampling (autoregressive decoding)

```python
prefill_kv = llm(prefix_tokens, return_kv=True)
tokens = []
while len(tokens) < 256 and not has_eos:
    logits = llm.decode_step(prev_token, kv_cache=prefill_kv)
    token = argmax(logits)              # or categorical with temperature > 0
    tokens.append(token)
actions = fast_decode(tokens)            # inverse BPE → inverse DCT → un-normalize
```

Terminates on EOS (id=1) or `max_decoding_steps=256`. Inverse FAST then reconstructs the action chunk.

### LoRA configuration

| Component | LoRA targets | Rank | α |
|---|---|---|---|
| PaliGemma 2B (the only adapted stack) | attention QKV + FFN | 16 | 16.0 |
| SigLIP | frozen | — | — |
| EMA | disabled | — | — |

**No action expert means no second set of LoRA matrices.** A single set captures both visuolinguistic grounding AND action token emission.

## Inference cost — the counterintuitive part

| | Pi0 (flow matching) | Pi0-FAST (autoregressive) |
|---|---|---|
| Forward passes per chunk | 10 × full-model denoising steps | 1 prefix prefill + ~30–60 single-token decodes |
| Per-pass compute | Full PaliGemma + action expert (batched over 50 action tokens) | PaliGemma decode-step (1 token, KV-cached) |
| Parallelism | 10 batched passes over the time axis | Strictly serial token decoding |
| **Reported chunk latency on 4090** | **~100 ms** | **~750 ms** (~7× slower) |

The intuitive framing "fewer steps therefore faster" is **wrong at the chunk level**. Flow matching batches its 10 steps over a 50-vector action chunk; autoregressive decoding serializes 30–60 token decodes that cannot be parallelized along time. **What Pi0-FAST actually saves is training compute** — the FAST paper reports ~5× fewer GPU-hours to match Pi0 on the same datasets — because cross-entropy converges faster than velocity regression on this loss landscape.

## Why the comparison is architecturally meaningful

The two models share *everything* except the action representation:
- Same backbone (PaliGemma 3B)
- Same vision encoder (SigLIP-So400m/14)
- Same language tokenizer
- Same training data (LIBERO `physical-intelligence/libero`)
- Same hyperparameters in this pilot (batch 32, seed 42, LoRA, identical step count)

So any per-suite performance difference can be attributed to the **action representation** alone, which is the architectural axis being studied.

## Key file references (openpi @ e4580662)

- `src/openpi/models/pi0.py` — config, forward, `compute_loss`, `sample_actions`
- `src/openpi/models/pi0_fast.py` — same for Pi0-FAST
- `src/openpi/models/gemma.py` — Gemma stack + LoRA configuration (variants at lines 58–109)
- `src/openpi/models/siglip.py` — SigLIP vision encoder (variant decoding lines 298–373)
- `src/openpi/models/tokenizer.py` — `FASTTokenizer` and the vocabulary-tail-overwrite logic
- `src/openpi/training/config.py:585-604` — LIBERO LoRA presets
- `src/openpi/training/config.py:187` — quantile-norm factory override

## References

- Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164 (2024).
- Pertsch et al., *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, arXiv:2501.09747 (2025).
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Beyer et al., *PaliGemma: A versatile 3B VLM for transfer*, arXiv:2407.07726 (2024).
