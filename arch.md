# Canonical Recurrent-Depth Architecture: LFM3-1.7B-RD

[![License](https://img.shields.io/badge/License-LFM_Open_v1.0-blue.svg)](https://liquid.ai)
[![Framework](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-yellow)](https://huggingface.co)
[![Compression](https://img.shields.io/badge/Compression-2.5x%20Physical-success)](https://github.com/cybertruck021/lfm3-rd)

Official architectural specification for **LFM3-1.7B-RD**, a True Recurrent-Depth (Prelude–Core–Coda) Transformer model derived from `LiquidAI/LFM3-1.7B`. The architecture collapses 24 unrolled physical layers into **8 physical layers**, reducing parameter count from **1.73B to ~691M** (a 2.5× physical reduction) while supporting scalable test-time latent reasoning across recurrence loops $T \in [1, 32]$.

---

## 1. Macro Architecture & Tensor Pipeline

```
                              INPUT TOKENS x
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Embed Tokens     │  (Vocab: 128,000, Dim: 2048)
                         └──────────┬──────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    PRELUDE (Physical Layers 0–1, Executed ONCE)        │
│  • Layer 0: ShortConv (k=7) + SwiGLU FFN                               │
│  • Layer 1: ShortConv (k=7) + SwiGLU FFN                               │
│  Produces static, immutable Context Anchor: e = Prelude(x_embed)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ Dynamic State h_1             │ Immutable Anchor e
                    ▼                               │
┌───────────────────────────────────────────────────┴────────────────────┐
│              CORE (Physical Layers 2–5, Looped R Times)                │
│                                                                        │
│  Step Injection:   h_in(t) = h_t + τ(t)                                │
│                                                                        │
│  • Pos 0 (Layer 2): Context-Anchored Full Attention + SwiGLU FFN       │
│  • Pos 1 (Layer 3): Depth-Isolated ShortConv (k=7) + SwiGLU FFN        │
│  • Pos 2 (Layer 4): Context-Anchored Sliding Attention (W=256) + FFN   │
│  • Pos 3 (Layer 5): Depth-Isolated ShortConv (k=7) + SwiGLU FFN        │
│                                                                        │
│  Euler Damped Vector Gating at Core Exit:                              │
│  h_{t+1} = h_t + \alpha(t) \odot (CoreOutput \odot gate_h + e \odot gate_e - h_t) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Final Latent Thought h_R
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     CODA (Physical Layers 6–7, Executed ONCE)          │
│  • Layer 6: Full Attention + SwiGLU FFN (from Layer 22)                │
│  • Layer 7: ShortConv (k=7) + SwiGLU FFN (from Layer 23)               │
│  • Final RMSNorm + LM Head ──► Output Vocabulary Logits P(y)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Physical Layer Topology & Compression

### Parameter Breakdown

| Component | Physical Layers | Original Source Layer | Parameters | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Embeddings & LM Head** | — | Tied Embeddings | 262.1 M | 100% Pretrained |
| **Prelude** | 2 | Layers 0, 1 | 107.4 M | 100% Pretrained |
| **Core (Motif 2)** | 4 | Layers 10, 11, 12, 13 | 214.7 M | Pretrained + Anchors |
| **Coda** | 2 | Layers 21, 22 | 107.4 M | 100% Pretrained |
| **Adaptive Controller** | — | Kinematic Router | 0.5 M | Zero-Init |
| **Total Model Weight** | **8** | **24** | **~691.7 M** | **2.5× Compression** |

In 4-bit quantization (AWQ/GPTQ/GGUF Q4_K_M), the complete model occupies **< 390 MB VRAM**, enabling full execution on edge devices and consumer GPUs.

---

## 3. Mathematical Foundations of Latent Reasoning

The canonical architecture replaces naive weight sharing with a mathematically stable dynamical system governed by three foundational pillars:

```
                        ┌───────────────────────────────┐
                        │      FROZEN ANCHOR e          │
                        │    (Immutable Coordinate)     │
                        └───────┬───────────────┬───────┘
                                │               │
          Keys & Values         │               │ Continuous Injection
          Cross-Grounding       │               │ Forcing Function
                                ▼               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│   CONTEXT-ANCHORED ATTENTION    │   │      DUAL VECTOR GATING         │
│   Q from h_t, KV from [h_t; e]  │   │  h_{t+1} = h * g_h + e * g_e    │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### 3.1. Context-Anchored Attention (Layers 2 and 4)

In standard recurrent transformers, self-attention keys drift over multiple recurrence cycles, leading to representation collapse. Pure cross-attention to the prompt prevents iterative multi-step reasoning. Context-Anchored Attention resolves this by projecting queries from dynamic thoughts while grounding keys and values in both thoughts and the anchor:

$$\mathbf{Q}_t = \text{RMSNorm}(h_t) \mathbf{W}_q^\top \in \mathbb{R}^{B \times S \times (H_q d_{head})}$$

$$\mathbf{K}_t = \Big[ \text{RMSNorm}(h_t) \mathbf{W}_{k, h}^\top \;\;\Big|\;\; \text{RMSNorm}(e) \mathbf{W}_{k, e}^\top \Big] \in \mathbb{R}^{B \times 2S \times (H_{kv} d_{head})}$$

$$\mathbf{V}_t = \Big[ \text{RMSNorm}(h_t) \mathbf{W}_{v, h}^\top \;\;\Big|\;\; \text{RMSNorm}(e) \mathbf{W}_{v, e}^\top \Big] \in \mathbb{R}^{B \times 2S \times (H_{kv} d_{head})}$$

#### Block-Causal Masking ($S \times 2S$)

For a query token at position $i \in \{0, \dots, S-1\}$:

$$\mathcal{M}_{i, j} = \begin{cases}
0, & \text{if } j < S \text{ and } j \le i \quad (\text{Causal self-attention over evolving thought } h_t) \\
0, & \text{if } j \ge S \text{ and } (j - S) \le i \quad (\text{Prefix-grounded cross-attention over anchor } e) \\
-\infty, & \text{otherwise}
\end{cases}$$

For Layer 4 (Sliding Attention with window $W=256$), the causal condition on the thought block is constrained by $|i - j| \le 256$, while context grounding on the anchor $e$ spans the entire prefix $j - S \le i$.

---

### 3.2. Depth-Isolated ShortConv State Banks (Layers 3 and 5)

LFM3 utilizes 1D depthwise causal convolutions with kernel size $k=7$. During autoregressive generation ($S = 1$), running multiple recurrence passes on the same token would contaminate historical context buffers.

To prevent buffer pollution, recurrent steps access independent state buffers indexed by recurrence depth $t \in \{0, \dots, R-1\}$:

$$\mathbf{S}_{\mathrm{conv}}[t, \mathrm{pos}] \in \mathbb{R}^{B \times D_{\mathrm{conv}} \times (k - 1)}$$

where `conv_state_bank[t][pos]` isolates states across recursion loops:
- **Prefill / Parallel Training ($S > 1$):** Stateless causal 1D convolution executes across sequence dimension $S$. Zero cache overhead.
- **Autoregressive Generation ($S = 1$):** Each loop $t$ updates its own depth slice. Total memory consumption across all loops is **< 0.29 MB VRAM**.

---

### 3.3. Dual Vector Gating & Step Embedding $\tau(t)$

To steer representation energy and guarantee contractive fixed-point dynamics:

1. **Step Conditioning Vector $\tau(t) \in \mathbb{R}^D$:**
   $$h_{\mathrm{in}}(t) = h_t + \tau(t), \quad \tau(t) \in \mathbb{R}^{2048}$$
   Injects temporal depth coordinates, allowing shared Core neurons to specialize across iteration phases.

2. **Euler Damped Vector Gating:**
   $$h_{\mathrm{target}} = h_{\mathrm{step}} \odot \mathrm{gate}_h + e \odot \mathrm{gate}_e$$
   $$h_{t+1} = h_t + \alpha(t) \odot (h_{\mathrm{target}} - h_t)$$
   where $\mathrm{gate}_h, \mathrm{gate}_e \in \mathbb{R}^D$ are learnable channel-wise vectors (initialized to $\mathbf{1}$ and $\mathbf{0}$), and $\alpha(t)$ is the harmonic step scale.
   - Sandwich norms (`post_operator_norm`, `post_ffn_norm`) within each sublayer strictly preserve the residual stream scale ($\mathrm{RMS} \sim 0.04\text{--}0.09$), eliminating any need for an artificial Core exit normalization.
   - $\mathrm{gate}_e \odot e$ injects prompt energy, ensuring the attractor never drifts away from the initial problem constraints.

---

### 3.4. Kinematic Halting Controller (`Lfm3RDController`)

The controller acts as an adaptive computation time (ACT) router, predicting optimal thinking depth and token readiness without requiring expensive vocabulary projections:

1. **Dynamical Trajectory Features:**
   - Velocity: $\Delta h = h_{\mathrm{curr}} - h_{\mathrm{prev}}$
   - Relative Velocity: $v_{\mathrm{rel}} = \frac{\|\Delta h\|_2}{\|h_{\mathrm{curr}}\|_2 + \epsilon}$ (`rel_vel`)
   - Relative Anchor Drift: $d_{\mathrm{rel}} = \frac{\|h_{\mathrm{curr}} - e\|_2}{\|e\|_2 + \epsilon}$ (`rel_drift`)
   - Directional Instability: $c_{\mathrm{instab}} = 1 - \frac{\langle h_{\mathrm{curr}}, h_{\mathrm{prev}} \rangle}{\|h_{\mathrm{curr}}\|_2 \|h_{\mathrm{prev}}\|_2}$ (`cos_instab`)
   - Normalized Step: $u_{\mathrm{step}} = \frac{t}{31.0}$ (`depth_coord`)

2. **Prediction Heads:**
   $$z = \mathrm{SiLU}\Big(\mathbf{W}_{\mathrm{state}} \mathrm{RMSNorm}(h_{\mathrm{curr}}) + \mathbf{W}_{\mathrm{vel}} \mathrm{RMSNorm}(\Delta h) + \mathbf{W}_{\mathrm{dyn}} [\mathrm{feats}]\Big)$$
   $$s_{\mathrm{pred}} = 31.0 \times \sigma(\mathbf{W}_{\mathrm{halt}} z + b_{\mathrm{halt}})$$
   $$p_{\mathrm{ready}} = \sigma\Big(2.0 \times (t - s_{\mathrm{pred}} + 0.70)\Big)$$
   where $b_{\mathrm{halt}} = -1.91$ initializes $s_{\mathrm{pred}} \approx 4.0$ matching the canonical baseline depth $T = 5$.

---

## 4. Zero-Perturbation Pretrained Weight Transplant

When constructing `Lfm3RDForCausalLM` from official `LiquidAI/LFM3-1.7B` checkpoint weights, parameters are initialized such that **at step 0 fixed point ($h_t \equiv e$), the model produces identical mathematical outputs to the base model**:

| Parameter | Source / Initialization | Operational Behavior |
| :--- | :--- | :--- |
| $\mathbf{W}_q$ | $\mathbf{W}_q^{(core)}$ from Layer 10 / 12 | Exact preservation of pretrained query space |
| $\mathbf{W}_{k, h}, \mathbf{W}_{v, h}$ | $\mathbf{W}_k^{(core)}, \mathbf{W}_v^{(core)}$ | Exact preservation of native self-attention pathways |
| $\mathbf{W}_{k, e}, \mathbf{W}_{v, e}$ | $\mathbf{W}_k^{(core)}, \mathbf{W}_v^{(core)}$ (Exact Clone) | **At fixed point $h_t = e$: $0.5 V_h + 0.5 V_e = V_h \implies$ Output is 100% native** |
| $\text{gate}_h$ | $\mathbf{1.0}$ | Identity throughput from Core block |
| $\text{gate}_e$ | $\mathbf{0.0}$ | Zero initial disturbance from anchor residual |
| $\tau(t)$ | $\mathbf{0.0}$ | Zero initial disturbance from temporal offsets |

---

## 5. Training Protocol

### Stage 1: Any-Depth Final Logit Matching ($T \in [1, 5]$)
- **Teacher:** Frozen 24-layer `LiquidAI/LFM3-1.7B`.
- **Student:** 8-layer `Lfm3RDForCausalLM` (691M parameters).
- **Depth Sampling:** $T \sim \text{Uniform}(\{1, 2, 3, 4, 5\})$ per training batch.
- **Supervision Target:** All student recurrence depths are supervised directly against teacher final logits $P_{teacher}$.
  $$\mathcal{L} = \alpha D_{KL}\left( P_{student}^{(T)} \,\Big\|\, P_{teacher} \right) + (1 - \alpha) \text{CrossEntropy}(y, \hat{y})$$
- **Dataset Budget:** 25–40M tokens (FineWeb-Edu, GrandMaster-PRO-MAX, Bespoke-Stratos-17k reasoning, Glaive function calling).

### Stage 2: Adaptive Halting Optimization
- Model backbone frozen; only `Lfm3RDController` is trained.
- Evaluates token-level early exit to minimize average inference latency.

---

## 6. Directory Structure

```
lfm3-rd/
├── arch.md                       # Architectural specification
├── README.md                     # Repository overview
├── requirements.txt              # Environment dependencies
├── lfm3_rd/
│   ├── __init__.py               # Package exports
│   ├── configuration_lfm3_rd.py  # Lfm3RDConfig specification
│   └── modeling_lfm3_rd.py       # Canonical PyTorch model implementation
├── conversion/
│   ├── __init__.py               # Converter exports
│   └── convert.py                # Zero-perturbation checkpoint extractor
└── eval/
    ├── __init__.py
    └── verify_checkpoint.py      # Verification and benchmark harness
```

---

## References

1. **Liquid AI** — *Liquid Foundation Models (LFM) Architecture and Field Notes* (2025).
2. **DeepMind** — *Learning to Ponder: A Differentiable Approach to Adaptive Computation Time* (Banino et al., 2021).
3. **Meta FAIR** — *The Unreasonable Ineffectiveness of the Deeper Layers* (Gromov et al., arXiv:2403.17887, 2024).
4. **ICLR 2025** — *Relaxed Recursive Transformers with Layer-wise LoRA* (Bae et al., arXiv:2410.20672).
5. **OpenMythos / Open-Fable** — *Context-Anchored Recurrent Transformers* (2025–2026).
