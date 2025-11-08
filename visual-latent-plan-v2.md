# Incremental Development Plan — Text→2D “Visual Latent”→Text (Fully Self-Supervised)

> A step-by-step roadmap with checkpoints so you can verify each component before adding complexity.

The goal here is to eventually learn a text encode/decoder that is trained on just text in a fully self-supervised way.  The trick is to make the latent "visual", a 2D latent that has image-like properties.

---

## 0) Bootstrap & Fixtures

**Goal:** Stand up data, augmentations, priors, and metrics you’ll reuse throughout.

### Dataset & Views
- Corpus: in-domain sentences (colored blocks + spatial relations).
- Meaning-preserving augmentations: conjunct shuffle, synonym swaps (“on”/“on top of”), det/number tweaks, passive↔active, object-name renaming.
- Meaning-breaking counterfactuals (negatives): flip color/relation/count; swap relation arguments.

### Visual Latent
- Shape: `V ∈ R^{H×W×C}` with defaults `H=W=32`, `C=6`.

### Analytic “Image-ness” Priors
- 1/f spectrum loss (slope α≈2).
- Total variation (TV).
- Wavelet/steerable-pyramid sparsity.
- Gradient heavy-tails (kurtosis > 3).
- Channel decorrelation + target variance.

### Readouts & Utilities
- `g(V)`: mean/var pooling + MLP → 256-d embedding.
- Fixed evaluation pack (sentences + paraphrases + counterfactuals).

### Plots (save each epoch)
- Per-channel log–log spectra with fitted α.
- TV, gradient kurtosis, channel covariance heatmap.
- Montage of the `C` channels for 16 fixed sentences (same order every run).

### Unit Checks
- FFT radial binning returns a line for synthetic 1/f fields.
- TV loss > 0 on random noise; falls after smoothing.

---

## Phase 1 (P1): “Image-like” Latent Without Semantics

**Goal:** Make the 2-D latent grid `V ∈ R^{H×W×C}` look image-like under analytic priors **only** (no contrastive, no decoder yet).

---

## Suggested Model Structure (concrete, minimal)

### Text Encoder \(E\_\phi: x → V\)

**Tokenizer**
- Byte-pair encoding (BPE), vocab ≈ 16k–32k.
- Max sequence length: 64 (raise to 96 if your sentences are longer).

**Transformer Encoder (small)**
- Layers: **6**
- Hidden size: **384**
- MHA heads: **8**
- FFN dim: **1536** (≈4× hidden)
- Activation: **GELU**
- Norm: **Pre-LayerNorm**
- Dropout: **0.1**
- Positional encoding: **rotary** (RoPE) or **learned absolute** (either is fine here)

> Output: token states `H ∈ R^{L×384}`

**Grid Cross-Attention “Reshape” Head (tokens → 2-D grid)**
- Create a learned grid of **queries** `Q ∈ R^{(H·W)×384}` with `H=W=32` → 1024 queries.  
  (Parameterize `Q` as a **2-D conv-pos** grid: start from a `32×32×64` map, apply a **3×3 Conv → GELU → 1×1 Conv** to reach 384 channels, then flatten to `(1024×384)`.)
- **Cross-Attention block × 2** (lightweight):
  - MHA: 8 heads, key/value from tokens `H`, queries from `Q`
  - Residual + Pre-LN, FFN(384→768→384, GELU), dropout 0.1
- **Projection to channels**: `Linear(384 → C)`, with **C = 6**
- **Reshape** to `V ∈ R^{32×32×6}`
- **(Optional) Local smoothing head**: depthwise **3×3 Conv** + **1×1 Conv** (GELU) to reduce checkerboarding in early training.

> Rationale: the cross-attn head lets the model “paint” a spatial grid from text without imposing semantics yet; the convolutions inject helpful locality so priors (TV, spectrum) have something to act on.

**Parameter count (rough)**: ~20–25M.

---

## Training Setup (P1 only)

**Optimization**
- AdamW, lr **2e-4**, β=(0.9, 0.95), weight decay **0.01**
- Cosine LR schedule, **1000** warmup steps
- Batch size: **256** sentences (use grad accumulation if needed)
- EMA of encoder params: **0.999**

**Latent Size**
- `H=W=32`, `C=6` (bump to `C=8` only after P5)

**Warmup Trick (stability)**
- For the first **1–2 epochs**, apply a **Gaussian blur (σ=0.8)** to `V` *before* computing priors; then remove.

---

## Losses (what they do & why)

**What these losses do & why:**
- **`L_spec` (1/f spectrum matching):** Encourages the latent grid to have a natural-image power spectrum (roughly 1/f²), discouraging white-noise textures and promoting realistic multi-scale structure.  
- **`L_tv` (total variation):** Penalizes abrupt pixel-to-pixel changes to reduce speckle/checkerboard artifacts and promote smooth regions separated by edges.  
- **`L_wav` (wavelet/steerable sparsity):** Makes band-pass responses sparse so edges and oriented structures dominate over noise.  
- **`L_gradkurt` (heavy-tailed gradients):** Matches the heavy-tailed gradient statistics typical of natural images (lots of small gradients, few large edges).  
- **`L_cov` (channel decorrelation):** Prevents channels from redundantly copying each other; encourages each channel to carry distinct information.  
- **`L_var` (target variance):** Holds overall variance near a target to avoid collapse to flat fields or explosion to high-energy noise.

\[
L_{\text{img}} \;=\; \lambda_1 L_{\text{spec}} \;+\; \lambda_2 L_{\text{tv}} \;+\; \lambda_3 L_{\text{wav}} \;+\; \lambda_4 L_{\text{gradkurt}} \;+\; \lambda_5 L_{\text{cov}} \;+\; \lambda_6 L_{\text{var}}
\]

**Good starting weights**
- `λ = (0.5, 0.1, 0.1, 0.05, 0.05, 0.05)`

---

## Logging & Visualization (P1)

- **Per-channel log–log spectra** with fitted slope **α**; target **α∈[1.5, 2.5]** (ignore DC).
- **TV** trend (should fall from init and plateau > 0).
- **Gradient kurtosis** (> 3) for `∂x, ∂y` histograms.
- **Channel covariance** heatmap (near-diagonal).
- **Montage of V**: 16 fixed sentences × 6 channels; look for smooth blobs & edges, not checkerboards.

---

## Pass/Fail Criteria (advance only if “pass”)

**Pass when**
- ≥ **4/6** channels have α in range for **3** consecutive evals,
- TV plateaus and is non-zero,
- Visuals stable across seeds (no speckle explosion),
- Channel covariance ≈ diagonal.

**If it fails**
- Increase `λ_tv` (e.g., +0.05) or reduce `λ_var`,
- Enable/extend blur warmup,
- Reduce LR to **1e-4**,
- Add/keep the *local smoothing head* (3×3 depthwise conv).

---

## Notes & Alternatives

- **Slot-attention variant (optional):**  
  If you prefer more object-centric inductive bias even in P1, add **slot attention (K=6, iters=3)** on token states to get slots `S ∈ R^{K×384}`, then **splat** them onto the grid with learned 2-D Gaussians (predict mean/cov per slot and channel mixing), yielding `V`. Keep the same priors/losses. This can further suppress checkerboards and give cleaner blobs, at the cost of a few more params.

- **Positional signal:**  
  If α drifts low (<1.2) and patterns get too periodic, add **(x, y) coordinate channels** (normalized to [−1,1]) concatenated to `V` before priors for a few epochs; they can be dropped later.


## Phase 2 (P2): Add Meaning via Contrast

**Goal:** `V` is consistent across paraphrases, distinct from counterfactuals.

**Train:** `Eφ` with **InfoNCE + priors**  
**What these losses do & why:**  
- **`InfoNCE` (contrastive agreement):** Pulls together embeddings of meaning-preserving paraphrases and pushes apart embeddings of other sentences/counterfactuals. This injects *semantics* into the latent while remaining fully self-supervised.  
- **`L_img` (priors, from P1):** Continues to enforce “looks-like-an-image” statistics on the latent grid.  
- *(Optional)* **VICReg/Barlow-Twins:** Anti-collapse regularizers that spread information across embedding dimensions and stabilize training.  
`L = L_img + α·InfoNCE(g(V(x₁)), g(V(x₂)))`  
Negatives: other batch items + constructed counterfactuals.  
Optional: VICReg/Barlow-Twins on `z=g(V)`.

**Metrics**
- Retrieval@1/5 over paraphrase pools.
- t-SNE/UMAP of `z`: paraphrase clusters tight; counterfactuals far.
- Cosine gap: `cos(z, z_paraphrase) − cos(z, z_counterfactual)`.

**Pass when**
- R@1 ≥ 60% and R@5 ≥ 85% (held-out synthetic paraphrases).  
- Cosine gap ≥ 0.25.

**If not**
- Strengthen negatives; increase batch; temperature τ→0.05; add latent jitter (translate/blur/cutout on `V`) and invariance penalty on `g(·)`.

---

## Phase 3 (P3): Stabilize to Spatial Jitter

**Goal:** `V` behaves like an image under small spatial transforms.

**Add:** Latent jitter `T` (±2 px shifts, mild blur, small cutout) with  
**What this loss does & why:**  
- **`L_inv` (invariance alignment):** Makes the global representation invariant to small *visual* perturbations of the latent grid (shifts/blur/cutout), so the latent behaves like an image under typical geometric/photometric nuisances.  
`L_inv = β · ‖g(T(V(x₁))) − g(V(x₂))‖²`.

**Metrics**
- Robustness: cosine(g(V), g(T(V))) vs pixel shift.
- Spectra/TV stay within P1 ranges.

**Pass when**
- Cosine ≥ 0.9 for ±2 px; retrieval doesn’t drop more than 2%.

**If not**
- Reduce cutout size; add coordinate channels (x, y) concatenated to `V` before priors.

---

## Phase 4 (P4): Bring in the Text Decoder

**Goal:** Show `V` contains content sufficient to reconstruct input (teacher forcing only).

**Model:** Decoder `Dψ` (LM) with cross-attention over flattened `V` + pooled `z`.  
**Train:** **Freeze encoder**, optimize  
**What this loss does & why:**  
- **`L_dec` (cross-entropy reconstruction under constraints):** Teacher-forces the decoder to reproduce the input sentence from the latent, proving that `V` carries sentence content. The constrained vocab/CFG keeps outputs in-domain and prevents degenerate copying of off-distribution tokens.  
`L_dec = CE(x₁, Dψ(V(x₁)))`  
with a constrained vocab mask (colors, shapes, relations, glue words).

**Diagnostics**
- Token-level CE for content vs function words.
- Token→`V` attention maps (content tokens low entropy, spatially concentrated).
- **Ablation:** shuffle `V` spatially → CE should spike.

**Pass when**
- Content token accuracy ≥ 90% (held-out).  
- Shuffling `V` increases CE by ≥ 30%.  
- Attention maps show localized peaks.

**If not**
- Increase cross-attn heads; add small MLP on `V`; slightly raise `C`.

---

## Phase 5 (P5): Round-Trip + Non-Copying Generation

**Goal:** Generate **semantically equivalent** sentences without parroting.

**Unfreeze encoder** (small LR). Add:
- **Copy suppression** on free-run sample `ẋ` (penalize long n-gram overlap with input).
- **Round-trip**: generate k candidates (top-p), choose best by `sim(g(V(x)), g(V(ẋ)))`; backprop through chosen candidate.

**Loss stack:**  
**What each added term does & why:**  
- **`L_cycle` (round-trip consistency):** Encourages the sentence generated from the latent to re-encode to a *semantically equivalent* latent, reinforcing that `V` captures meaning rather than surface form.  
- **`L_copySuppress` (anti-parroting):** Penalizes long n-gram overlaps between the input and the free-run output, pushing the decoder toward genuine paraphrases instead of verbatim copies.  
- **`L_img`, `InfoNCE`, `L_inv`, `L_dec`**: continue serving their roles from earlier phases (image statistics, semantic alignment, jitter invariance, and content reconstruction).  
`L = L_img + α·InfoNCE + β·L_inv + γ·L_dec + δ·L_cycle + ρ·L_copySuppress`

**Automatic Checker (rule-based)**
- Extract objects/colors/relations by regex/finite-state rules.
- Metrics: **Graph F1**(x, ẋ), **ROUGE-L**(x, ẋ), **Round-trip cosine** `cos(g(V), g(V(ẋ)))`.

**Pass when**
- Graph F1 ≥ 0.85 (held-out).  
- ROUGE-L ≤ 0.55 (not verbatim).  
- Round-trip cosine ≥ 0.9.

**If not**
- Increase k (4→8); tune ρ up to curb copying; widen synonym set; reduce teacher forcing schedule.

---

## Phase 6 (P6, Optional): Self-Trained Latent Diffusion Prior

**Goal:** Sharpen the latent manifold using only your own `V` samples.

**Train:** Tiny U-Net `Uθ` on replay buffer of `V` with ε-prediction diffusion loss.  
Regularize encoder outputs by a single DDIM denoise step projection:  
**What this loss does & why:**  
- **`L_prior` (self-trained latent diffusion prior):** Projects the latent toward a denoised version predicted by a diffusion model trained only on past latents. This sharpens structure and encourages consistency across paraphrases without using real images. *(Gradients do not pass through `Uθ`.)*  
`L_prior = ξ · ‖V − DDIM_step(V_t)‖²` (stop-grad through `Uθ`).

**Pass when**
- Prior denoise reduces prior losses (spectrum/TV) without hurting retrieval.  
- Paraphrase consistency improves: `‖g(V(x)) − g(V(paraphrase(x)))‖` ↓ ≥ 10%.

**If not**
- Lower `ξ`; longer `Uθ` training; use min-SNR weighting.

---

## Phase 7 (P7): Scale & Stress Tests

**Goal:** Validate robustness and component importance.

**Ablations (retrain small):**
- Remove each prior (spectrum/TV/wavelet/kurtosis) one at a time.
- Remove InfoNCE vs round-trip vs copy suppression.
- With/without latent jitter.

**Report:**
- Retrieval (R@k), Graph F1, ROUGE-L, round-trip cosine.
- Spectra α histogram, TV, kurtosis, covariance heatmap.
- Throughput & VRAM for `32×32×6` vs `48×48×8`.

**Pass when**
- Dropping any core piece measurably degrades at least one target metric (builds confidence + debugging map).

---

## Default Hyperparameters (per Phase)

```yaml
latent:
  H: 32
  W: 32
  C: 6

priors_weights:
  spectrum: 0.5
  tv: 0.1
  wavelet: 0.1
  grad_kurtosis: 0.05
  cov: 0.05
  var: 0.05

contrastive:
  alpha: 1.0
  temperature: 0.07
  batch_size: 256  # gradient accumulation ok

invariance:
  beta: 0.2
  jitter_prob: 0.5
  max_shift_px: 2

decoder:
  gamma: 0.7
  vocab_mask: true

round_trip:
  delta: 0.2
  top_p: 0.9
  best_of_k: 4  # increase to 8 if needed

copy_suppression:
  rho: 0.1  # increase up to 0.3 if parroting

optimization:
  encoder_lr: 2e-4
  decoder_lr: 1e-4
  optimizer: AdamW
  schedule: cosine
  ema: 0.999
```

---

## Dashboards to Keep Open

- **Latent health:** α histograms; TV trend; kurtosis; channel covariance.
- **Semantics:** Retrieval@k; cosine gaps to counterfactuals.
- **Decoder:** CE by token type; n-gram overlap; Graph F1; round-trip cosine.
- **Qualitative:** Fixed-seed channel montages; token→`V` attention maps.

---

## Common Failure Signatures → Quick Fixes

- **Checkerboard / speckle:** ↑ `λ_tv`; blur warmup; ↓ LR.
- **Paraphrase collapse (all `V` similar):** stronger/harder negatives; add VICReg variance; shrink `C`; enforce target variance.
- **Decoder parrots input:** ↑ `ρ`; ↑ `k`; expand synonyms; reduce teacher forcing.
- **Semantics drift after P5:** ↓ `δ` (cycle) or `ξ` (latent prior) to avoid over-regularization.

---

## Phase Checklists

- [ ] **P1**: Image-ness priors stable (α, TV, kurtosis, visuals).  
- [ ] **P2**: Paraphrase retrieval strong; counterfactual separation.  
- [ ] **P3**: Jitter robustness without metric regressions.  
- [ ] **P4**: Decoder reconstructs with content-token accuracy ≥ 90%; shuffling `V` breaks it.  
- [ ] **P5**: Round-trip paraphrases with Graph F1 ≥ 0.85, ROUGE-L ≤ 0.55.  
- [ ] **P6 (opt)**: Latent prior improves consistency without harming P5 metrics.  
- [ ] **P7**: Ablations show expected performance drops.

---

### Artifacts to Save at Each Phase

- Model snapshots: `Eφ`, `Dψ`, (`Uθ` if used) + optimizer + EMA.
- Metrics CSVs and evaluation pack.
- Visuals: spectra plots, TV/kurtosis trends, channel montages, attention maps.

---

_You can paste this into your repo as `PLAN.md` and annotate each phase with dates/results. If you want, I can also provide a Hydra config scaffold (`conf/phase=P1..P7.yaml`) and minimal pytest checks for priors/retrieval._
