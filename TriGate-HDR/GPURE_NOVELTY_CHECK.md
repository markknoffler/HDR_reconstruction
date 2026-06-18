# GPURE Novelty Check (In-Depth)

**Date:** 2025-06  
**Target venues:** CVPR / ICLR / ICML  
**Method:** Gate-Partitioned Unified Radiance Energy (GPURE) — TriGate-HDR v2

---

## 1. Claimed contributions (must be unique in combination)

| # | Contribution | What it is |
|---|--------------|------------|
| C1 | **Partitioned variational energy** | Single \(\mathcal{L}_{\mathrm{GPURE}}\) over composed HDR, not sequential script orchestration |
| C2 | **LR-CFP** | Log-radiance cold forward on **expansion-only** latent component |
| C3 | **RSO** | Radiometric synapse operators at skip connections (camera response + log-ratio gain) |
| C4 | **ECC** | Exposure-bracket consistency on **seam band** between generative and cold paths |
| C5 | **Tri-gate partition** | Trust / clip / seam routing from one LDR input (no multi-exposure capture) |

**Novel if:** No prior work combines C1–C5 for **single-image LDR→HDR** with **cold (non-Gaussian) anchor path** + **generative clip path**.

---

## 2. Literature map (2024–2026)

### 2.1 Gain-map / decomposition methods

| Paper | Venue | Decomposition | Diffusion | Single LDR | Cold anchor |
|-------|-------|---------------|-----------|------------|-------------|
| **Guan et al. GM Decomposed Diffusion** | ICCV 2025 | SDR + Gain Map | SD latent, Gaussian | Text-to-HDR / up-convert | No |
| **GMODiff** | arXiv 2025/26 | Gain map refinement | One-step LDM | Multi-exposure | No |
| **Adobe Gain Map** | Spec | SDR base × GM | — | — | No |
| **GPURE** | (ours) | LDR lift + expansion latent | Cold on expansion | **Yes** | **Yes** |

**Gap:** Gain-map methods reformulate HDR as low-bit-depth residual; they do **not** use LDR-as-deterministic-corruption cold diffusion or trust-gated dual paths.

### 2.2 One-step / exposure-aware generative HDR

| Paper | Venue | Mechanism | Unified multi-path | Cold diffusion |
|-------|-------|-----------|-------------------|----------------|
| **ExpoCM** | CVPR 2026 | PF-ODE, exposure mask trajectories | Single model | No (Gaussian generative) |
| **LEDiff** | arXiv 2025 | Latent exposure bracket fusion | Fusion module | Gaussian LDM |
| **Bracket Diffusion** | arXiv 2024 | Multi-LDR consistency in diffusion | Bracket coupling | Gaussian |
| **GPURE** | (ours) | Partitioned energy + compose | **Dual path + seam** | **Expansion cold** |

**Gap:** ExpoCM is closest on **exposure partitioning**, but uses **one generative ODE**, not complementary cold+radiometric + generative paths with ECC on seams.

### 2.3 Retinex / latent restoration

| Paper | Venue | Idea | HDR from single LDR |
|-------|-------|------|---------------------|
| **Reti-Diff** | ICLR 2025 | Retinex latent DM + RGformer joint train | IDR focus, not clip-gated HDR compose |
| **GPURE** | (ours) | LORCD expansion cold + TriGate | **Yes, explicit** |

**Gap:** Retinex diffusion is on reflectance/illumination latents; GPURE is on **orthogonal expansion** anchored to LDR latent with **no Retinex R/L split**.

### 2.4 Cold diffusion foundation

| Paper | Mechanism | GPURE difference |
|-------|-----------|------------------|
| **Bansal et al. Cold Diffusion** | Any deterministic corruption | LR-CFP restricts to log-radiance + expansion-only + trust partition |
| **ColdEfficient-LORCD (baseline)** | Linear cold on \(z^E\) | GPURE adds unified energy, RSO, LR-CFP, joint training |

---

## 3. Component-level novelty verdict

| Component | Novel alone? | Strong as part of GPURE? |
|-----------|--------------|--------------------------|
| Trust gate from LDR | Partial (exposure masks in ExpoCM) | Yes — routes **different generators** |
| Expansion-only cold | Yes for HDR LDR→HDR | **Core** |
| LR-CFP encoding | Yes vs linear latent cold | **Strengthens theory** |
| RSO skip fusion | Yes vs conv/MLP gates | **Architectural claim** |
| ECC seam loss | Similar spirit to Bracket Diffusion | Yes — **single LDR**, seam band only |
| Unified \(\mathcal{L}_{\mathrm{GPURE}}\) | Yes vs 3-script pipeline | **Fixes reviewer concern** |

---

## 4. Risks and mitigations (reviewer attacks)

| Risk | Mitigation in paper |
|------|-------------------|
| "Just pipeline glue" | Lead with **Theorem 1 (LR-CFP identifiability)** + **Algorithm 1 (joint energy descent)** |
| "Same as cold diffusion" | Emphasize **expansion-only + log-radiance + partitioned energy** |
| "Same as ExpoCM exposure mask" | Dual **complementary** paths (radiometric vs generative), not one ODE |
| "RSO is hand-waved physics" | Ablate RSO vs RGCF; show \(\Phi_k\) learns display-relevant response |
| "Gain map papers did decomposition" | We decompose in **latent expansion**, not SDR×GM; cold forward is different math |

---

## 5. Recommended ablation table (for submission)

1. Baseline LORCD (no GPURE)
2. + unified energy (joint train, no RSO/LR-CFP)
3. + LR-CFP only
4. + RSO only
5. + ECC (\(\lambda_b\))
6. Full GPURE
7. Orchestration (sequential train) vs joint — **directly answers reviewer**

---

## 6. Overall novelty verdict

**Sufficient for top-tier submission** if the paper is framed as:

> *Gate-Partitioned Unified Radiance Cold Diffusion* — the first single-image HDR framework that (i) unifies radiometric cold expansion and generative clip recovery under one partitioned variational energy, (ii) grounds cold forward in log-radiance space, and (iii) replaces generic fusion with radiometric synapse operators.

**Not sufficient** if framed only as "three stages with better losses" — must lead with **unified optimization + theorems + ablations**.

---

## 7. Prior art search keywords used

- single image HDR reconstruction diffusion 2025 2026
- cold diffusion HDR latent
- exposure partition HDR ExpoCM
- gain map diffusion GMODiff ICCV
- bracket consistency LDR HDR
- Reti-Diff joint training restoration

No exact match found for: **trust-gated tri-path + expansion-only LR-CFP + RSO + unified GPURE energy**.
