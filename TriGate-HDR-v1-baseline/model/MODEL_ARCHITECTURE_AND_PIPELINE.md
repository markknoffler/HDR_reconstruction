# TriGate-HDR: Pipeline, Theory, and Implementation

This document explains the **TriGate-HDR** system in depth: **what** each piece is, **why** it exists, how it connects to **HDR imaging theory**, where the design is **novel or staged** relative to textbook pipelines, and how that maps to **code**. Equations use standard GitHub-compatible math blocks (`$$ ... $$` for display, `$...$` for inline).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Why three stages](#2-why-three-stages-the-core-idea)
3. [Repository layout](#3-repository-layout)
4. [Data: what it represents and why](#4-data-what-it-represents-and-why)
5. [Stage 1: luminance-grounded diffusion with grounded encoders](#5-stage-1-luminance-grounded-diffusion-with-grounded-encoders)
6. [Stage 2: cold diffusion along a clipping trajectory](#6-stage-2-cold-diffusion-along-a-clipping-trajectory)
7. [Stage 3: fusion and localized seaming GAN](#7-stage-3-fusion-and-localized-seaming-gan)
8. [Losses: roles, theory, and when each applies](#8-losses-roles-theory-and-when-each-applies)
9. [GAN: adversarial realism without undoing grounded regions](#9-gan-adversarial-realism-without-undoing-grounded-regions)
10. [SAM masks and LDR-aligned supervision](#10-sam-masks-and-ldr-aligned-supervision)
11. [Training scripts and progressive training](#11-training-scripts-and-progressive-training)
12. [Metrics](#12-metrics)
13. [Gradients and freezing](#13-gradients-and-freezing)
14. [Limitations and honest caveats](#14-limitations-and-honest-caveats)

---

## 1. Executive summary

**TriGate-HDR** attacks a tension that ordinary single-network HDR pipelines struggle with:

- In **highlight-clipped** regions of an LDR, the true HDR radiance is **not uniquely determined** by the recorded pixel—the camera has saturated. Pixel-wise regression to ground truth is mismatched here: the model either **averages away** plausible highlights or **hallucinates** details with no grounding.
- At the same time, in **well-exposed** regions, HDR recovery is closer to **deterministic inversion** once one assumes a tone curve / radiometric model—so strong pixel-wise supervision helps.

Rather than forcing one objective to satisfy both regimes, TriGate HDR **routes** reconstruction through **different generative biases** per regime, then **composes** and **repairs seams**. Implementation-wise:

- **Stage 1** trains a heavily **conditioned denoiser** (luminance-diffusion paradigm with material, structure, semantics, segmentation) so catastrophic corruption / denoising is **anchored** in scene cues—not only RGB texture—across the **whole frame** during training.
- **Stage 2** trains a separate path where “noise” is **clamping toward LDR** (cold-diffusion framing), emphasizing **recovery of radiometric detail where semantics are intact** in the clipped domain.
- **Stage 3** pastes luminance hallucinations onto cold-path HDR only where LDR clipping demands it, then uses a **small GAN-style refiner** that is **explicitly allowed to edit only a seam neighborhood**, while discriminators assess **whole-image realism** and **seam-conditioned realism**.

The document below unpacks **why each element is grounded** physically and probabilistically—and where the implementation is exploratory vs standard.

---

## 2. Why three stages — the core idea

HDR from a single exposure is inherently **ambiguous** where the sensor clipped. Classical approaches handle this implicitly (global tone mapping priors); learning-based pipelines often bury the ambiguity in end-to-end MSE—which typically yields **muted highlights** because MSE minimizes variance.

**This pipeline’s narrative:**

1. **Learn a strong hallucination-capable HDR generator** whose denoising is **conditioned on multiple scene factors** so it isn’t drifting into texture soup (Stage 1).
2. **Learn another HDR recovery path** that treats **tone collapse as diffusion steps** (“cold diffusion”) so clipped recovery is coherent with inverse tone mapping intuition (Stage 2).
3. **Merge** strengths: deterministic-ish cold path keeps **faithful non-highlight structure**; luminance diffusion supplies **hypotheses** only where needed **without baking them into Stage 2**.
4. **Visually unify** pasted regions with adversarial realism and seam-specific losses—not by globally repainting Stage 2 outputs (Stage 3).

Thus the **triple gate** metaphor: gates separate **recovery regime** vs **generation regime**, and the final stage resolves **incoherence between two model opinions**.

---

## 3. Repository layout

Purpose of each folder (not just filenames):


| Path                   | Role                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `encoders/`            | Specialized encoders: material cues, geometric/structural cues, semantic codebook. They exist so conditioning is **orthogonal** RGB-only texture. |
| `decoders/`            | Standalone nets: luminance-diffusion primitives, **cold diffusion** HDR UNet (`ColdHDRDiffusion`). Useful as modules and Stage-2 entry point.     |
| `dual_decoders/`       | **Authoritative Stage-1** system: merges three encoders + mask encoder **into one grounded UNet**. Training scripts point here on purpose.        |
| `seaming_model/`       | GAN pair for **localized seam refinement** — not universal reconstruction.                                                                        |
| `losses/`              | Atomic losses + staged compositions; keeps **explicit** which objective applies where.                                                            |
| `training_scripts/`    | Scripts that mirror staging, datasets, checkpoints, metrics.                                                                                      |
| `sam_mask_generation/` | Offline LDR segmentation to supply **spatial priors** for distribution matching and segmentation conditioning.                                    |


**Why `dual_decoders` vs `decoders` duplication:** splitting keeps a **readable base UNet** in `decoders/` while Stage-1’s research assembly (cross-stream fusion) lives beside it without contaminating reuse.

---

## 4. Data: what it represents and why

### Files and pairing

LDR RGB (display-referred proxy) aligns with HDR (scene-referred) by **matching stems**. Pairing expresses the **paired supervision** intuition: HDR is the supervisory target for many losses; LDR is the **inferior observation** modeling sensor limitation.

### Ranges

- HDR normalized by max puts training in **relative radiance**: convenient for stabilization; **psychophysical absolutes per cd/m² aren’t reconstructed** unless you calibrate afterward.
- Linear map to approximately $[-1, 1]$: matches common network bias toward symmetric activation ranges around zero, typical of diffusion heads, even though the diffusion schedule itself may be loosely specified elsewhere.

### The gate $g$ — physical meaning

Roughly, the gate marks **regions where measured LDR still carries usable intensity information** versus **saturation**:

$$
g_u \;=\; \mathbf{1}\!\left\{ \max_{c \in \{R,G,B\}} \mathrm{LDR}_{c,u} < 0.98 \right\}.
$$

(or soft variants in future work). **Why thresholds near 1:** channels near saturation are typically **pinned** near display white or sensor clip, so multiple HDR scene radiances map to one LDR value—**non-invertible**. Routing distribution matching toward $(1 - g)$ in Stage 3 avoids pretending those pixels obey a deterministic pixel-wise HDR recovery.

---

## 5. Stage 1: luminance-grounded diffusion with grounded encoders

### Idea and novelty

Cold diffusion frameworks treat forward corruption as deterministic maps (luminance perturbation clip trajectories elsewhere in this project). Stage 1 adopts the **spirit**—denoise from crippled HDR representation—while forcing **spatially global conditioning** streams so hallucination **respects segmentation and material coherence**.

**Breaking from vanilla U-Net HDR:** Instead of stacking RGB repetitions internally, lateral cross-stream fusion says: *attention should align material queries with semantic and structural contexts at each pyramid scale.*

### Modules — what each is **for**

- **Material encoder** — Encodes statistically stable surface cues gradients / histogram-derived channels favor (implementation-specific). Hypothesis: **texture appearance** survives tone mapping deformation better than naive RGB alone.
- **Structural encoder + gate output** — Carries saturation-aware cues; emits **explicit** non-clip-ish mask info feeding later stages (gates).
- **Semantic codebook encoder** — Learns stochastic latents penalized toward Gaussian **so class-like latents aren’t collapsing** (KL term in Stage 1).
- **RuntimeMaskPredictor** — Answers deployment reality: users may **lack** offline SAM masks. A small CNN maps LDR $\to$ soft pseudo-segmentation channels; blending with dataloaded segmap pushes gradients through this path at train time so inference degrades gracefully.
- **SegMaskEncoder** (mask infusion) — Solves scale mismatch between a single segmentation map and pyramid features by encoding the mask into a **multi-resolution** pyramid matched to each UNet level.
- **HorizontalTriStreamFusion** — Queries from projected **material** features attend over structural and semantic keys/values ($1/\sqrt{C}$ softmax scaling). Four streams (material mixes, semantic mixes, masks) are multiplied by timestep-dependent gates $\sigma(W t_{\mathrm{emb}})$ so different diffusion steps emphasize different conditioners.

### Mathematical sketch (compact)

Let $x_\ell$ be UNet features at depth $\ell$, with timestep embedding $t_{\mathrm{emb}}$ and streamed features $M_\ell,S_\ell,Z_\ell,R_\ell$. Gated timestep mixing yields four nonnegative channels per feature band, then a 1×1 mixture conv and residual:

$$
\text{gates} = \sigma\!\big(W\, t_{\mathrm{emb}}\big) \in \mathbb{R}^{4 \times C}
$$

$$
x_\ell \;\leftarrow\; x_\ell + \mathrm{Conv}_{1\times1}\!\left(\mathcal{M}\Big( \text{gates} \odot [\tilde M_\ell,\, \mathcal{A}_{M \to S},\,\mathcal{A}_{M \to Z},\,\tilde R_\ell] \Big)\right),
$$

where $\text{gates}$ is broadcast to match concatenated streams, $\mathcal{A}_{M \to S}$ and $\mathcal{A}_{M \to Z}$ are softmax mixes from structural/semantic keys with material-derived queries, and $\mathcal{M}$ is the channel-mixing 1×1 path in code.

### Loss philosophy for Stage 1

Histogram / Wasserstein class conditioning **runs over entire image**: during aggressive corruption + reconstruction every region’s statistics may shift; supervising only clipped subsets under-trains grounding for **textures** revived from noise.

KL regularizes latent geometry for semantic bookkeeping.

Implementation reference: `dual_decoders/cold_hdr_luminance_diffusion_decoder.py`, `train_stage1_dual_diffusion.py`, `stage1_loss()` in `stage_composite_losses.py`.

---

## 6. Stage 2: cold diffusion along a clipping trajectory

### Conceptual grounding

Tone mapping collapses intensities upward (display clipping). Inverse problem: expand dynamic range—but **singular** inverse where saturated. Modeling **explicit forward degradation** $x \mapsto \mathrm{clip}(x)$ along a timestep schedule casts training as predicting **radiance before applying** that destructive clamp—an analogy to deterministic forward processes in cold diffusion literature.

### Operational meaning

Sampling $t$ picks **how clipped** HDR state is toward LDR plateau. The network sees corrupted HDR and timestep $t$, then predicts HDR. Constraint pairs:

1. reconstructed HDR aligns with supervised HDR pixelwise (HDR term),
2. re-applying deterministic clip operator should reproduce observed clipped proxy (dual consistency akin to reversible noise).

Cold sampling `restore_hdr(ldr)` simulates iterating inverse operator starting from plausible LDR input.

### Mathematical summary

Let $b_t$ be a scalar (or mapped) clipping threshold indexed by timestep (implementation: decreasing buffer).

After clamping HDR to nonnegative values, clipping is applied **elementwise**:

$$
x_{\mathrm{clip}}^{(t)} \;=\; \min(x_{\mathrm{HDR}},\, b_t).
$$

Prediction $\hat{x} = f_\theta(x_{\mathrm{clip}}^{(t)}, t)$; the training objective is **dual** L1 consistency:

$$
\mathcal{L} \;=\; \| x_{\mathrm{HDR}} - \hat{x} \|_{1}
\;+\; \big\| x_{\mathrm{clip}}^{(t)} - \mathrm{clip}^{(t)}(\hat{x}) \big\|_{1}.
$$

### Hybrid radiometric loss — why bespoke

Gaussian / L1 alone ignore **radiometric coherence** vs **tone curve**: highlights roll off differently than midtones. Hybrid loss adds:

- Gamma proxy inverse/forward (**CRF cyclic** consistency gated for trusted regions — reduces fake contrast),
- logarithmic irradiance stabilization (HDR scale invariance),
- horizontal irradiance ratios (spatial exposure gradient alignment),
- **rolloff shaping** emphasizing saturated mask portions (generation bias where clip matters).

Hence Stage 2 is **radiometric-aware** reconstruction rather than naïvely saying “pixels must match HDR.”

References: `decoders/cold_hdr_diffusion_decoder.py`, `losses/radiometric_losses.py`, `train_stage2_crf_recovery.py`.

---

## 7. Stage 3: fusion and localized seaming GAN

### Narrative rationale

Assume Stage 2 excels at **overall geometry + tone coherence** absent extreme hallucinations. Stage 1 injects speculative detail into saturated holes. Paste causes **histogram / texture mismatches**. Global MSE penalizes artistic highlight creativity; unconditional GAN may repaint whole scenery.

**Hybrid solution:** Compose, then refine **near boundaries** only adversarially; global discriminator still observes **whole context** realism.

### Composite

Cold HDR denoted $x^{(2)}$, luminance paste denoted $x^{(1)}$, saturated-region mask $m_{\mathrm{clip}} = 1 - g$:

$$
x_{\mathrm{composed}} = (1 - m_{\mathrm{clip}}) \odot x^{(2)} + m_{\mathrm{clip}} \odot x^{(1)}
$$

### Seam band morphology

Morphological dilation/erosion via max-pooling approximates widening/narrowing region:

$$
m_{\mathrm{seam}} = \mathrm{clip}\big(\mathrm{dilate}(m_{\mathrm{clip}}) - \mathrm{erode}(m_{\mathrm{clip}})\big) \cup m_{\mathrm{clip}}
$$

This widens receptive field for refinement without globally destructive edits.

### Generator behavior

The generator predicts residual field $\Delta$; output is

$$
x_{\mathrm{out}} = x_{\mathrm{composed}} + m_{\mathrm{seam}} \odot \Delta
$$

Architectural inductive bias **hard codes micro-localization** aligned with intuition.

---

## 8. Losses: roles, theory, and when each applies

**W1 histogram (distribution matching).** Narrative intent: in ambiguous or hallucinated zones, insisting on pixel equality fights the manifold of plausible HDRs. Matching **histograms** conditioned on strata (semantic / SAM masks) enforces plausible **brightness statistics** instead of brittle pointwise identity. Mathematical core: build empirical densities for pred vs GT inside each mask class, normalize to probabilities, integrate to CDFs $F_{\mathrm{pred}}, F_{\mathrm{gt}}$, then penalize $\ell_1$ CDF distances (Earth-mover style in 1D). Stage 1 weights the **whole image**; Stage 3 uses region $(1-g)$ to stress **clipped** areas.

**SFL (structural fidelity).** Narrative intent: human vision is sensitive to **edges and contours**—two images can match in MSE yet look blurry. Core: Sobel magnitude L1 between pred and GT, plus mismatch of coarse edge maps.

**Gated L2.** Narrative intent: where gate $g \approx 1$, the HDR is constrained by observable LDR; strong regression is warranted. Mathematical form averaged over spatial channels:

$$
\mathcal{L}^{\mathrm{gate}}_{\mathrm{L2}} = \mathbb{E}_{u,c}\!\left[\, g_u\, \big( x^{\mathrm{pred}}_{u,c} - x^{\mathrm{gt}}_{u,c} \big)^2 \,\right].
$$

Used in Stage 3 reconstruction mix.

**MCL (material consistency).** Narrative intent: HDR recovery should not alter **surface appearance** statistics where LDR informs material gradient structure. Implementation blends Gram-matrix texture statistics of gradients with local pooling.

**SGCL (seam gradient continuity).** Narrative intent: composites create **derivative discontinuities** at paste boundaries visible as halos. Weight strongest where $g \approx \tfrac{1}{2}$ to align gradient fields across seams.

**Hybrid radiometric (Stage 2).** Narrative intent: cold-diffusion inversion should respect **tone curves**, log-dynamic range coherence, lateral exposure ratios, highlight roll-off—all radiometric hypotheses beyond plain L2.

**KL codebook (Stage 1).** Narrative intent: semantic latents collapse without regularization toward a meaningless prior basin. Variational KL ties approximate posterior $q(z \mid x)$ toward standard Gaussian preserving informative geometry.

Weighted Wasserstein over SAM classes ($c$: class index, masks $m_c$, region indicator $\mathbf{1}_R$):

$$
\mathcal{L}_{\mathrm{W1}} = \sum_c w_c\, W_1\!\left( p^{(c)}_{\mathrm{pred}},\, p^{(c)}_{\mathrm{gt}} \right), \qquad w_c = \mathbb{E}\big[ m_c \odot \mathbf{1}_R \big],
$$

with $W_1$ implemented as mean absolute CDF difference per class.

---

## 9. GAN: adversarial realism without undoing grounded regions

Classic GAN minimizes implicit Jensen-Shannon divergence (goodfellow-style conceptualization) between real & generated distributions—or hinge reformulation with margin.

### Why hinge here?

Hinge discriminator forms **margin-based critic** resisting vanishing saturation (vs early sigmoid critics). Implemented:

Discriminator hinge:

$$
\mathcal{L}_{D} \;=\; \mathbb{E}\big[\max(0,\, 1 - D(x_{\mathrm{real}}))\big] \;+\; \mathbb{E}\big[\max(0,\, 1 + D(x_{\mathrm{fake}}))\big].
$$

Generator hinge (push fake scores upward):

$$
\mathcal{L}^{\mathrm{G}}_{\mathrm{hinge}} \;=\; -\, \mathbb{E}\big[ D(x_{\mathrm{fake}}) \big].
$$

Applied **twice**: global realism + seam-context branch given concatenated `[image|seam]` channels.

### Why **two discriminators mentally**?

- **Global** head punishes unnatural global tone / color inconsistencies even if seams micro-fixed.
- **Seam-conditioned** branch focuses capacity on correlations between mask geometry and unnatural micro-patterns—a cheap alternative to patching crops dynamically.

### Outside-lock term

Penalty:

$$
\mathcal{L}_{\mathrm{outside}} \;=\; \mathbb{E}\,\Big[ \big| (1 - m_{\mathrm{seam}}) \odot (x_{\mathrm{out}} - x_{\mathrm{composed}}) \big| \Big].
$$

This **anchors** fidelity outside edit zone while adversarial realism flexes localized aesthetics.

Philosophy: GAN is **not** asked to invent whole scenes—only to **harmonize** composite artifacts.

---

## 10. SAM masks and LDR-aligned supervision

**Why LDR for SAM export:** Real deployment sees LDR first. Segmentation that only exists on HDR misaligns train/test distribution (domain shift). SAM on LDR makes geometry discovery **distribution consistent**.

SAM yields **instances** aggregated into enumerated labels per image (not ontology classes)—still useful for stratified histogram matching (**empirical strata** correlate with contiguous regions like sky vs foliage blobs).

Converted `segmap` channels (normalized id raster, discontinuity emphasis, summed presence map) diversify semantic encoder receptive clues—essentially injecting **topology priors**.

---

## 11. Training scripts and progressive training


| Script                           | Training intent                                                                                                                                 |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_stage1_dual_diffusion.py` | Joint encoder+decoder warmup with global distributional fidelity + KL; optional SAM class masks.                                                |
| `train_stage2_crf_recovery.py`   | Specialized cold-diffusion refinement with radiometric loss; isolates inversion path gradients.                                                 |
| `train_stage3_seaming_gan.py`    | Freezes predecessors; alternating D/G hinge + composite HDR losses + locality lock `--outside_lock_weight`. Requires Stage1/Stage2 checkpoints. |
| `run_progressive_pipeline.py`    | Lightweight orchestrator for frozen earlier stages optimizing seam generator with metrics CSV logging.                                          |


**Progressive rationale:** earlier stages converge different energy landscapes simultaneously if naïvely summed—**staging** aligns optimization geometry (fewer contradictory gradient directions).

---

## 12. Metrics

**PSNR-μ**: applies logarithmic $\mu$-law normalization with $\mu \approx 5000$ before squared error. **Why:** linear HDR raw scale dwarfs subjective visibility; logarithmic perceptual-ish domain matches legacy pipelines (FHDR-derived tradition).

Formula after tonemap $T_\mu(\cdot)$:

$$
\text{MSE} = \mathbb{E}\left[ \left( T_\mu(\hat{x}) - T_\mu(x) \right)^2 \right], \quad
\text{PSNR} = 10 \, \log_{10}\!\left( \frac{1}{\text{MSE}} \right)
$$

**SSIM**: assesses luminance/contrast/structure at display-linear mapped `[0,1]` tensors.

**HDR-VDP proxies**: if `pyfvvdp` present, leverages modern visual difference predictor; fallback compresses perceptual PU21-like encoding disparity into scalable scalar mimic for logging continuity.

Purpose: comparable scalar reporting aligned with **`ARThdrNet/m_training.py`** for fair cross-experiment benchmarking.

---

## 13. Gradients and freezing


| Stage | Trained modules                          | Frozen modules                               | Rationale                                                                      |
| ----- | ---------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------ |
| 1     | Entire `Stage1TriEncoderDiffusionSystem` | None in script                               | Learn alignment of encoders with decoder under heavy corruption.               |
| 2     | `ColdHDRDiffusion`                       | No encoders present                          | Focus radiometric inversion path.                                              |
| 3     | Seaming G (+ D)                          | Stage1+Stage2 loaded & `requires_grad=False` | Avoid destabilizing earlier specialized minima while seam network specializes. |


---

## 14. Limitations and honest caveats

1. **Histogram via `torch.histc`:** binning is piecewise; analytical smoothness for backprop is limited—acceptable for exploratory training; soften if publication demands smooth surrogates.
2. **Stage 2 sampling schedule vs deployment:** thresholds & iteration counts require empirical tuning for dataset dynamic range extremes.
3. **Absolute photometry absent:** normalized HDR training yields **relative**, not calibrated **nit** reconstructions absent post calibration.
4. **SAM strata ≠ semantics:** Classes are heuristic regions not guaranteed aligned with ontology—histogram conditioning is stratified—not semantic classification accuracy measured here.
5. **GAN variance:** Stable GAN tuning (lr, discriminator steps per generator step) remains empirical workload.

---

## Diagram and external parity references


| Resource                                                      | Purpose                                 |
| ------------------------------------------------------------- | --------------------------------------- |
| `model_architecture_design/TriGateHDRUnifiedArchitecture.tsx` | Conceptual schematic for talks / papers |
| Repository root `ARThdrNet/m_training.py`                     | Metric parity reference implementation  |


---

*This document merges narrative theory with implementation mapping. Maintain both sections together when refactoring code so onboarding readers keeps intuition synchronized with tensors and losses.*