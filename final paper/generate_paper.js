const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  BorderStyle, WidthType, ShadingType, PageBreak, LevelFormat,
  Table, TableRow, TableCell, Footer, Header, PageNumber,
} = require("docx");

const CONTENT_WIDTH = 9360;
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 24, font: "Times New Roman", ...opts.run })],
  });
}

function pRuns(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: runs.map(r => new TextRun({ size: 24, font: "Times New Roman", ...r })),
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, bold: true, size: 28, font: "Times New Roman" })] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true, size: 26, font: "Times New Roman" })] });
}

function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, bold: true, size: 24, font: "Times New Roman" })] });
}

function bullet(ref, text) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })],
  });
}

function eq(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, size: 24, font: "Times New Roman", italics: true })],
  });
}

function makeTable(headers, rows) {
  const colW = Math.floor(CONTENT_WIDTH / headers.length);
  const colWidths = headers.map(() => colW);
  const mkCell = (text, header = false) => new TableCell({
    borders,
    width: { size: colW, type: WidthType.DXA },
    shading: header ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: header, size: 22, font: "Times New Roman" })] })],
  });
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map(h => mkCell(h, true)) }),
      ...rows.map(r => new TableRow({ children: r.map(c => mkCell(c)) })),
    ],
  });
}

const children = [
  // Title
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "TriGate-HDR: Gate-Partitioned Unified Radiance Cold Diffusion for Single-Image HDR Reconstruction", bold: true, size: 32, font: "Times New Roman" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [new TextRun({ text: "[Author Names and Affiliations — To Be Completed]", size: 24, font: "Times New Roman", italics: true })],
  }),

  // Abstract
  h1("Abstract"),
  p("Single-image high dynamic range (HDR) reconstruction from a single low dynamic range (LDR) exposure remains fundamentally ill-posed: saturated highlights admit infinitely many plausible radiance values, while well-exposed regions demand faithful radiometric inversion. Existing methods force a single network and a single loss to address both regimes, yielding either blurred highlights from pixel-wise regression or radiometrically implausible hallucinations from unconstrained generative models. We propose TriGate-HDR, a gate-partitioned unified radiance cold diffusion (GPURE) framework that routes recovery through three complementary paths governed by a pixel-wise exposure trust mask derived from the input LDR. Path-G employs a pretrained InstructPix2Pix latent diffusion model with TriGate encoder conditioning to hallucinate plausible structure in clipped regions. Path-C performs expansion-only cold diffusion in a dedicated latent space, anchored to the LDR measurement via a monotonic lift operator, to recover radiometrically consistent HDR in trusted regions without Gaussian noise corruption. Path-S applies a seam-localized generative adversarial refiner at clip boundaries to eliminate compositing artifacts. All three paths are unified under a single partitioned variational energy (GPURE) that jointly optimizes radiometric fidelity, cold self-consistency, generative clip recovery, exposure-bracket consistency on seam bands, and seam smoothness. We further introduce Log-Radiance Cold Forward Process (LR-CFP) for optically calibrated latent encoding and Radiometric Synapse Operators (RSO) that replace generic convolutional skip gates with camera-response-aware fusion at UNet decoder levels. [Results section to be completed upon final benchmark evaluation.]"),

  // 1 Introduction
  h1("1. Introduction"),
  p("High dynamic range (HDR) imaging seeks to capture and reproduce the full luminance range of real-world scenes—often spanning four to six orders of magnitude—on displays and in digital content pipelines. Consumer-grade cameras, however, are constrained by sensor physics to a narrow dynamic range, typically 8–10 bits per channel. The resulting low dynamic range (LDR) images suffer from saturation in over-exposed highlights and noise amplification in under-exposed shadows, permanently discarding radiance information that cannot be recovered by simple tone mapping [1]."),
  p("Multi-exposure fusion methods [2, 3] address this limitation by combining bracketed captures at different exposure settings. While effective, they require static scenes, precise alignment, and ghost removal—assumptions that fail in dynamic environments. Single-image HDR reconstruction, which recovers HDR radiance from a single LDR exposure, is therefore of substantial practical importance for re-rendering legacy photographs, enabling image-based lighting, and supporting next-generation HDR displays [4, 5]."),
  p("The central challenge of single-image HDR reconstruction is its inherent ill-posedness. In well-exposed regions, where the camera response function is approximately invertible, recovery reduces to a deterministic radiometric inversion problem. In saturated (clipped) regions, however, many distinct HDR radiance values map to the same LDR measurement—the inverse mapping is one-to-many. Pixel-wise regression losses (L1, L2, SSIM) applied uniformly across the image therefore tend to produce blurred highlights or washed-out predictions, as they average over the space of plausible solutions [6, 7]. Generative models, conversely, can hallucinate plausible highlight content but often sacrifice radiometric accuracy in well-exposed regions where deterministic recovery suffices [8, 9]."),
  p("Recent diffusion-based approaches have demonstrated remarkable generative capability for HDR reconstruction. Conditional DDPM frameworks [10] and latent diffusion adaptations [11, 12] leverage powerful pretrained priors to inpaint clipped regions. ExpoCM [13] reformulates HDR reconstruction as a one-step Probability Flow ODE with exposure-aware consistency trajectories, achieving state-of-the-art fidelity with orders-of-magnitude faster inference. LEDiff [14] performs latent exposure bracket fusion within a pretrained Stable Diffusion latent space. However, these methods employ a single generative pathway—typically Gaussian noise corruption—and do not explicitly separate the radiometrically constrained well-exposed regime from the generative clipped regime."),
  p("Cold diffusion [15] offers an alternative corruption paradigm: rather than adding Gaussian noise, the forward process applies deterministic, physically motivated degradations. Bansal et al. demonstrated that any deterministic image transform defines a valid generative reverse process, calling into question whether stochastic noise is essential for diffusion-based generation. For HDR reconstruction, cold diffusion is particularly natural: the LDR image itself is the \"cold\" boundary state toward which radiance collapses, preserving spatial structure throughout the corruption path rather than destroying it with i.i.d. noise."),
  p("We observe that no existing method simultaneously (i) partitions the image into exposure-dependent regimes with distinct recovery strategies, (ii) employs cold (non-Gaussian) diffusion anchored to the LDR measurement for radiometric expansion, (iii) leverages pretrained generative priors specifically for clipped regions, and (iv) unifies all paths under a single differentiable optimization objective. TriGate-HDR addresses this gap through the Gate-Partitioned Unified Radiance Energy (GPURE) framework."),

  h2("1.1 Contributions"),
  p("The main contributions of this work are:"),
  bullet("bullets", "A gate-partitioned three-path HDR reconstruction framework (TriGate-HDR) that routes recovery through complementary generative (Path-G), radiometric cold (Path-C), and seam-refinement (Path-S) pathways, each specialized for a distinct exposure regime."),
  bullet("bullets", "ColdEfficient-LORCD (Latent Orthogonal Radiance Cold Diffusion): an expansion-only cold diffusion formulation in a dedicated MiniHDR-VAE latent space, with LDR-anchored monotonic lift, dual-stream RGCF/RSO UNet, and scale-adaptive cold scheduling—trained from scratch without pretrained diffusion foundations."),
  bullet("bullets", "GPURE unified energy: a single partitioned variational objective that jointly optimizes radiometric fidelity, cold self-consistency, generative clip recovery, exposure-bracket consistency (ECC) on seam bands, and seam smoothness—with differentiable composition enabling end-to-end gradient flow."),
  bullet("bullets", "Log-Radiance Cold Forward Process (LR-CFP): optically calibrated log-radiance encoding before latent expansion decomposition, with identifiability guarantees under monotonicity and trust-gate assumptions."),
  bullet("bullets", "Radiometric Synapse Operators (RSO): domain-specific skip fusion replacing generic convolutional gates, incorporating learnable camera response functions and log-ratio gain modulation at UNet decoder levels."),
  bullet("bullets", "TriGate encoder conditioning for InstructPix2Pix Stage 1: a three-stream fusion module (material, structural, semantic) with timestep-gated attention injected into pretrained latent diffusion for clip-region hallucination."),

  // 2 Related Work
  new Paragraph({ children: [new PageBreak()] }),
  h1("2. Related Work"),

  h2("2.1 Single-Image HDR Reconstruction"),
  p("Early deep learning approaches to single-image HDR reconstruction focused on direct feed-forward CNN mappings from LDR to HDR. HDRCNN [6] pioneered this direction at SIGGRAPH Asia 2017, proposing an encoder-decoder architecture with skip connections trained on simulated sensor saturation across diverse camera response functions. ExpandNet [7] introduced a three-branch multiscale CNN (local, dilation, global) that avoids upsampling artifacts for HDR expansion. FHDR [16] exploited recurrent feedback connections to iteratively refine coarse-to-fine HDR representations with fewer parameters than feed-forward alternatives."),
  p("SingleHDR [17] marked a paradigm shift by explicitly modeling the LDR formation pipeline—dynamic range clipping, camera response function (CRF) application, and quantization—and learning three specialized sub-networks to reverse each stage. This physics-informed decomposition, trained on the HDR-Synth and HDR-Real datasets introduced in the same work, established the standard benchmark protocol (HDR-VDP-2, tone-mapped PSNR/SSIM) still used today. DrTMO [18] and other inverse tone mapping operators provide complementary baselines but often produce blurry or washed-out results due to fixed inverse CRF assumptions."),
  p("Subsequent CNN architectures pushed reconstruction quality through specialized designs. HDRUNet [19] proposed a spatially dynamic encoder-decoder with condition and weighting networks, jointly addressing HDR reconstruction, denoising, and dequantization with a Tanh-L1 loss; it won second place in the NTIRE 2021 HDR Challenge single-frame track. Masked autoencoder approaches [20] introduced perceptual losses and masked feature prediction for improved highlight recovery. ArtHDR-Net [21] emphasized perceptual realism and artistic intent preservation, achieving competitive HDR-VDP-2 scores using multi-exposed LDR features. HistoHDR-Net [22] proposed histogram-equalized LDR fusion with self-attention guidance to recover fine details in over- and under-exposed regions, reporting state-of-the-art PSNR on HDR-Real at ICIP 2024."),

  h2("2.2 Diffusion Models for Image Restoration"),
  p("Denoising diffusion probabilistic models (DDPMs) [23] established diffusion as a powerful generative paradigm by learning to reverse a gradual Gaussian noising process. DDIM [24] generalized the reverse process to non-Markovian deterministic sampling, enabling 10–50× faster generation. Latent diffusion models (LDMs) [25] operate in the compressed latent space of pretrained autoencoders, dramatically reducing computational cost while preserving visual fidelity—forming the foundation of Stable Diffusion and subsequent conditional editing models."),
  p("InstructPix2Pix [26] adapts latent diffusion for instruction-guided image editing by concatenating input image latents with noisy target latents and conditioning on CLIP text embeddings. Trained on synthetically generated instruction-image pairs, it performs zero-shot edits without per-example fine-tuning. For HDR reconstruction, this provides a natural mechanism: the instruction \"recover the HDR version of this image\" steers a pretrained generative prior toward highlight inpainting while preserving global structure."),
  p("Cold diffusion [15] demonstrated that the generative behavior of diffusion models is not dependent on Gaussian noise. By replacing stochastic corruption with deterministic transforms (blur, masking, downsampling), equally valid generative models emerge. For HDR, cold diffusion is physically motivated: the forward process blends toward the actual LDR measurement rather than toward noise, preserving semantic structure throughout corruption. Our ColdEfficient-LORCD formulation extends this principle to latent expansion fields with LDR anchoring."),

  h2("2.3 Diffusion-Based HDR Reconstruction"),
  p("The application of diffusion models to HDR reconstruction is recent but rapidly evolving. Dalal et al. [10] proposed a conditional DDPM framework with classifier-free guidance and a CNN autoencoder for latent conditioning, introducing an Exposure Loss that directs gradients away from saturation. This ICIP 2023 work demonstrated that a relatively simple conditional diffusion approach can match complex camera pipeline architectures on tone-mapped metrics."),
  p("LEDiff [14] (CVPR 2025) enables HDR generation within a pretrained Stable Diffusion latent space by creating \"latent exposure brackets\"—LDR latent codes at multiple exposure levels—and fusing them with a learnable module before decoding through a fine-tuned VAE decoder. While effective for expanding dynamic range in clipped regions, LEDiff relies entirely on the pretrained generative prior and does not employ cold diffusion or exposure-regime partitioning."),
  p("ExpoCM [13] represents the current state of the art on standard HDR benchmarks. It reformulates single-image HDR reconstruction as a one-step Probability Flow ODE (PF-ODE), constructing exposure-aware consistency trajectories via soft exposure masks that separate over-, under-, and well-exposed regions. Region-conditioned trajectories hallucinate saturated details, suppress dark-region noise, and preserve reliable structures within a single distillation-free inference step. An Exposure-guided Luminance-Chromaticity Loss in CIE L*a*b* space further mitigates brightness bias and color drift. ExpoCM achieves over 400× faster inference than DDPM (1000 steps) while setting new benchmarks on HDR-REAL, HDR-EYE, and AIM2025. However, ExpoCM employs a single generative ODE pathway rather than complementary radiometric and generative paths."),
  p("Reti-Diff [27] (ICLR 2025 Spotlight) addresses illumination degradation restoration through Retinex-based latent diffusion, decomposing images into reflectance and illumination components. While conceptually related to HDR (both involve luminance recovery), Reti-Diff targets low-light enhancement rather than single-image HDR radiance reconstruction and does not employ cold diffusion or exposure-regime partitioning."),

  h2("2.4 Parameter-Efficient Adaptation and Segmentation"),
  p("Low-Rank Adaptation (LoRA) [28] enables efficient fine-tuning of large pretrained models by injecting trainable rank-decomposition matrices into frozen Transformer layers, reducing trainable parameters by up to 10,000× without additional inference latency. We employ LoRA on InstructPix2Pix attention projections (to_q, to_k, to_v, to_out.0) for Stage 1 HDR fine-tuning, preserving the pretrained generative prior while adapting to HDR-specific editing."),
  p("The Segment Anything Model (SAM) [29] provides general-purpose image segmentation that we leverage offline to generate per-class semantic masks. These masks enable class-aware Wasserstein-1 distribution matching in Stage 1 novelty losses, ensuring that highlight hallucination preserves the statistical properties of each semantic region rather than producing generic blur."),

  h2("2.5 Positioning of TriGate-HDR"),
  p("Table 1 summarizes the key differentiators of TriGate-HDR relative to representative prior work. No existing method combines trust-gated tri-path routing, expansion-only cold diffusion in a dedicated latent space, pretrained generative priors for clip regions, and a unified partitioned energy objective."),
  makeTable(
    ["Property", "SingleHDR", "LEDiff", "ExpoCM", "Cold Diff.", "TriGate-HDR"],
    [
      ["Single LDR input", "Yes", "Yes", "Yes", "N/A", "Yes"],
      ["Cold (non-Gaussian) diffusion", "No", "No", "No", "Yes", "Yes"],
      ["Pretrained generative prior", "No", "Yes (SD)", "No", "No", "Yes (IP2P)"],
      ["Exposure-regime partitioning", "No", "Partial", "Yes", "No", "Yes"],
      ["Dual complementary paths", "No", "No", "No", "No", "Yes (G + C)"],
      ["Unified energy objective", "No", "No", "No", "No", "Yes (GPURE)"],
      ["Dedicated HDR latent VAE", "No", "No (SD VAE)", "No", "No", "Yes (MiniHDR-VAE)"],
      ["Seam-aware composition", "No", "No", "No", "No", "Yes (Path-S)"],
    ]
  ),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Table 1. Comparison of TriGate-HDR with representative prior methods.", size: 22, font: "Times New Roman", italics: true })] }),

  // 3 Method
  new Paragraph({ children: [new PageBreak()] }),
  h1("3. Method"),

  h2("3.1 Problem Formulation"),
  p("Given a single LDR image x_ldr ∈ [0,1]^(3×H×W) captured with an unknown camera response function, our goal is to recover the corresponding HDR radiance x_hdr ∈ [-1,1]^(3×H×W) (linear RGB, max-normalized per scene). We define a pixel-wise exposure trust mask τ ∈ [0,1]^(1×H×W) derived from the LDR input:"),
  eq("τ_u = 1  if  max_c x_ldr,c(u) < 0.98,   and   0  otherwise,"),
  p("where u indexes pixel locations and c indexes RGB channels. Pixels with τ_u ≈ 1 are well-exposed (trusted for radiometric inversion); pixels with τ_u ≈ 0 are clipped (saturated). The complement m_clip = 1 − τ defines the clip mask routing generative recovery."),

  h2("3.2 TriGate-HDR Pipeline Overview"),
  p("TriGate-HDR employs three complementary recovery paths unified under the GPURE energy (Figure 1, conceptual):"),
  bullet("numbers", "Path-G (Generative): InstructPix2Pix latent diffusion with TriGate encoder conditioning, specialized for hallucinating plausible HDR content in clipped regions (m_clip)."),
  bullet("numbers", "Path-C (Cold/Radiometric): ColdEfficient-LORCD expansion-only cold diffusion in a dedicated MiniHDR-VAE latent space, specialized for faithful radiometric expansion in trusted regions (τ)."),
  bullet("numbers", "Path-S (Seam): Seam-localized GAN refiner operating on a morphological band around clip boundaries, eliminating compositing artifacts between Path-G and Path-C outputs."),
  p("The composed HDR is formed by differentiable partition:"),
  eq("x_comp = (1 − m_clip) ⊙ x_C + m_clip ⊙ x_G,"),
  p("where x_C = Path-C(x_ldr) and x_G = Path-G(x_ldr). The final output x_out = Path-S(x_comp, x_G, m_seam) applies seam-localized refinement. All paths are trained under the unified GPURE energy (Section 3.6)."),

  h2("3.3 Stage 1: InstructPix2Pix with TriGate Conditioning (Path-G)"),
  h3("3.3.1 Pretrained Latent Diffusion Foundation"),
  p("Path-G builds on InstructPix2Pix [26], a conditional latent diffusion model that concatenates input image latents with noisy target latents for instruction-guided editing. We employ the timbrooks/instruct-pix2pix checkpoint with the following freezing policy: the VAE (AutoencoderKL) and CLIP text encoder remain frozen; LoRA adapters [28] are injected into UNet attention projections (to_q, to_k, to_v, to_out.0); and the TriGate LatentCondInjector is fully trainable."),
  p("Training follows the standard ε-prediction objective in VAE latent space. Given HDR target x_hdr and LDR input x_ldr, we encode z_hdr = Enc(x_hdr), z_img = Enc(2·x_ldr − 1), sample timestep t and noise ε ~ N(0,I), forming z_t = add_noise(z_hdr, ε, t). The UNet receives concat([z_t, z_img'], dim=1) where z_img' is TriGate-conditioned (below), and predicts ε̂ with loss L_diff = ||ε̂ − ε||²."),

  h3("3.3.2 TriGate LatentCondInjector"),
  p("The LatentCondInjector augments image latents with multi-stream encoder features before UNet processing:"),
  eq("z_img' = z_img + σ_res · Δ(z_img, t, Encoders(x_ldr, segmap)),"),
  p("where σ_res is a learnable scalar (initialized to 0.1). The TriEncoderBundle comprises four encoders:"),
  bullet("bullets", "Material encoder: captures texture and material cues (523 channels)."),
  bullet("bullets", "Structural encoder: extracts spatial structure and produces the exposure trust gate τ."),
  bullet("bullets", "Semantic codebook encoder: VAE-style latents with KL regularization for scene semantics."),
  bullet("bullets", "SegMask encoder: multi-scale mask pyramid from SAM [29] segmentation maps."),
  p("Features are fused via HorizontalTriStreamFusion at latent resolution: material stream queries structure and semantics through low-resolution attention (≤1024 tokens for VRAM efficiency), with timestep-gated mixing: fused = [γ₁M̃, γ₂Attn_S, γ₃Attn_Z, γ₄R̃] where γ_k = σ(W_k · t_emb). After diffusion-only warm-up epochs, novelty losses (Wasserstein-1 per SAM class, Sobel structural fidelity, KL on semantic codebook) are activated to improve perceptual quality in clipped regions."),

  h2("3.4 Stage 2: ColdEfficient-LORCD (Path-C)"),
  h3("3.4.1 Latent Radiance Decomposition"),
  p("Path-C operates in a dedicated MiniHDR-VAE latent space (trained from scratch, no pretrained SD/CLIP). Given LDR in HDR space x_ldr_hdr = 2·x_ldr − 1 and ground-truth HDR x_hdr, we encode:"),
  eq("z_hdr = E(x_hdr),   z_ldr = E(x_ldr_hdr),   z_lift = M(z_ldr),   z^E_0 = z_hdr − z_lift,"),
  p("where E is the MiniHDR-VAE encoder (/8 spatial compression, 4–8 latent channels), M is the MonoLift operator (z_lift = z_ldr + net(z_ldr) with monotonicity penalty), and z^E_0 is the expansion latent capturing radiometric headroom beyond the LDR-anchored lift."),

  h3("3.4.2 Expansion-Only Cold Forward Process"),
  p("Unlike Stage 1's Gaussian corruption, Path-C employs deterministic cold diffusion on the expansion component only:"),
  eq("z^E_t = (1 − α_t) · z^E_0,   z_t = z_lift + z^E_t,"),
  p("where α_t = t/(T−1) with α_0 ≈ 0 and α_{T−1} ≈ 1. At t = T−1, the state equals the LDR-anchored lift (expansion is fully collapsed); at t = 0, the full HDR latent is recovered. Scale-adaptive cold at UNet level ℓ uses α_{t,ℓ} = min(1, α_t · 2^(ℓ−L)) to apply stronger corruption at coarser scales."),

  h3("3.4.3 ColdEfficientLatentUNet Architecture"),
  p("The cold UNet employs dual streams with Radiometric Gated Cross-Fusion (RGCF) or Radiometric Synapse Operators (RSO):"),
  bullet("bullets", "Anchor stream: encodes z_ldr without timestep conditioning, producing anchor skips A₁…A₄."),
  bullet("bullets", "Cold stream: encodes concat(z_t, z_ldr) with sinusoidal timestep embedding, producing cold skips C₁…C₄."),
  bullet("bullets", "RGCF/RSO fusion at each level: C ← C + (1−τ)·Gate(concat(C,A))⊙Proj(A) + τ·Proj₂(A), modulated by trust gate τ."),
  p("The network predicts expansion ẑ^E_0 = f_θ(z_t, z_ldr, t, τ), and the HDR estimate is x̂_hdr = D(z_lift + ẑ^E_0). A PixelHDRRefiner (6-block residual CNN on concat(LDR, coarse_HDR)) optionally recovers high-frequency detail after VAE decode."),

  h3("3.4.4 Log-Radiance Cold Forward Process (LR-CFP)"),
  p("When enabled (--use_lr_cfp), the MiniHDR-VAE operates on log-radiance encoded inputs:"),
  eq("ℓ(x) = log(1 + k ⊙ max(radiance(x), 0)),   ℓ̂ = ℓ / s,"),
  p("where k is a per-channel learnable tone scale (via softplus) and s is a normalization constant. LR-CFP restricts cold corruption to log-radiance coordinates before expansion decomposition, preserving spatial support while anchoring to the LDR measurement. Under monotonicity of M, local invertibility of E, and trust gate τ ≈ 1 on non-saturated pixels, the expansion field z^E_0 is uniquely determined—yielding a strictly stronger identifiability guarantee than generic cold diffusion on arbitrary features."),

  h3("3.4.5 Radiometric Synapse Operators (RSO)"),
  p("RSO replaces generic RGCF convolutional gates with domain-specific fusion at skip connections:"),
  eq("RSO(h_c, h_a, τ, t) = h_c + σ(τ) ⊙ [exp(β·log_ratio) · Φ_k(h_a) · Ψ_t(h_c, h_a)]"),
  p("where Φ_k is a learnable per-channel μ-law camera response function, Ψ_t is a time-modulated depthwise spatial Jacobian, β controls log-ratio gain, and σ(τ) is a trust-modulated injection gate. RSO is applied at Stage 2 down/mid/up levels and optionally at the Stage 3 generator stem (RSOStem)."),

  h3("3.4.6 Training Losses (Path-C)"),
  p("Path-C training combines multiple complementary objectives:"),
  makeTable(
    ["Loss", "Formula / Role", "Weight (typical)"],
    [
      ["L_HDR", "||x_hdr − D(z_lift + ẑ^E_0)||₁", "1.0–3.0"],
      ["L_exp", "||z^E_0 − ẑ^E_0||₁", "2.0–3.0"],
      ["L_cold", "||ColdFwd(ẑ^E_0, t) − z^E_t||₁", "1.0"],
      ["L_trust", "||τ ⊙ ẑ^E_0||₁", "0.01"],
      ["L_vae", "L1 recon + KL on HDR/LDR cycle", "warmup only"],
      ["L_rad", "HybridRadiometricConsistency (CRF cycle, log, gradient)", "0.02–0.1"],
      ["L_μ-PSNR", "μ-law tonemap MSE (metric-aligned)", "0.25–2.0"],
      ["L_SSIM", "L1 in [0,1] RGB space", "0.35"],
      ["L_HF", "Sobel gradient L1", "0.5"],
      ["L_infer", "Multi-step restore_hdr alignment", "0.1"],
    ]
  ),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Table 2. Stage 2 (Path-C) training losses.", size: 22, font: "Times New Roman", italics: true })] }),

  h2("3.5 Stage 3: Seam-Localized Refinement (Path-S)"),
  p("Given composed HDR x_comp and clip hypothesis x_G, we construct a seam band via morphological operations on m_clip:"),
  eq("m_seam = clip(dilate(m_clip) − erode(m_clip)) ∨ m_clip,"),
  p("The SeamingGenerator accepts concat(x_comp, x_G, m_seam) (7 channels) through an RSOStem or standard 7→64 conv stem, followed by 6 SeamGatedBlocks. Each block applies mask-gated self-attention restricted to the seam neighborhood. The output is hard-localized:"),
  eq("x_out = x_comp + m_seam ⊙ Δ_θ(x_comp, x_G, m_seam),"),
  p("ensuring only seam pixels are modified. A dual-head discriminator (global RGB + seam RGB∥mask PatchGAN) provides adversarial supervision. An outside-lock loss L_outside = E[|(1 − m_seam) ⊙ (x_out − x_comp)|] prevents modification outside the seam band."),

  h2("3.6 GPURE Unified Energy"),
  p("The Gate-Partitioned Unified Radiance Energy (GPURE) unifies all paths under a single optimization objective:"),
  eq("L_GPURE = L_rad(x̂, x_gt) + λ_c·L_cold + λ_g·L_gen + λ_b·L_bracket + λ_s·L_seam"),
  p("where:"),
  bullet("bullets", "L_rad: L1 radiance fidelity on the composed/final HDR output."),
  bullet("bullets", "L_cold: Stage 2 cold + expansion + trust losses (Path-C self-consistency)."),
  bullet("bullets", "L_gen: Masked L1 on clipped regions: m_clip ⊙ ||x_G − x_gt||₁ (Path-G supervision)."),
  bullet("bullets", "L_bracket (ECC): Exposure-bracket consistency on seam band: ||μ(x_G) − μ(x_C)||₁ + ||∇x_G − ∇x_C||₁, coupling generative and radiometric paths."),
  bullet("bullets", "L_seam: Sobel magnitude penalty inside m_seam for smooth transitions."),
  p("Joint training (Phase B) backpropagates L_GPURE through the composed output into Path-C (and optionally Path-G), replacing the legacy sequential Stage 1→2→3 orchestration with a single differentiable optimization problem. The TriGateComposer module implements x_comp and m_seam inside the model graph, enabling gradient flow during joint training."),

  h2("3.7 Implementation Details"),
  p("Training is conducted on the HDR-Real dataset [17] (LDR/HDR pairs from real-world scenes) with optional evaluation on HDR-EYE [17] and AIM2025 benchmarks following the ExpoCM [13] protocol. Images are resized to 512×512 (max side) with per-scene max normalization for HDR targets mapped to [-1,1]. SAM [29] segmentation masks are generated offline and loaded as 3-channel segmaps plus per-class binary masks."),
  p("Stage 1 training uses 60 epochs with 5 diffusion-only epochs (LoRA only) followed by novelty ramp. Stage 2 uses 100–120 epochs with 5–8 VAE warmup epochs, VAE frozen thereafter, high-t biased timestep sampling (u^0.5), and cosine/ReduceLROnPlateau scheduling. The --arch_v2 preset enables wider architecture (latent_ch=8, base_ch=96, vae_base_ch=48), PixelHDRRefiner, and metric-aligned losses. GPURE joint training uses --memory_20gb mode (batch_size=1, AMP, frozen Stage 1) for 20GB GPU compatibility."),
  p("Evaluation metrics follow the ExpoCM [13] benchmark protocol: PSNR-μ and SSIM-μ (μ-law tonemap, μ=5000), PSNR-PU and SSIM-PU (PU21 banding_glare encoding with SI-HDR CRF correction, peak=256), PSNR-l and SSIM-l (linear domain), MS-SSIM, HDR-VDP-2 and HDR-VDP-3 (official implementations via Octave, 30 pixels per degree), LPIPS [30], and ΔE2000 (CIEDE2000 color difference)."),

  // References
  new Paragraph({ children: [new PageBreak()] }),
  h1("References"),
  ...[
    "[1] G. Eilertsen, J. Kronander, G. Denes, R. K. Mantiuk, and J. Unger, \"HDR image reconstruction from a single exposure using deep CNNs,\" ACM Trans. Graph. (Proc. SIGGRAPH Asia), vol. 36, no. 6, pp. 178:1–178:15, 2017.",
    "[2] P. E. Debevec and J. Malik, \"Recovering high dynamic range radiance maps from photographs,\" in Proc. SIGGRAPH, 1997, pp. 369–378.",
    "[3] Z. Liu, W. Wang, W. Zeng, Y. Zhang, and S. Liu, \"Ghost-free high dynamic range imaging with context-aware transformer,\" in Proc. ECCV, 2022, pp. 344–360.",
    "[4] R. K. Mantiuk, K. Myszkowski, and H.-P. Seidel, \"A perceptual framework for contrast processing of high dynamic range images,\" ACM Trans. Appl. Percept., vol. 3, no. 3, pp. 286–308, 2006.",
    "[5] G. Eilertsen, R. K. Mantiuk, and J. Unger, \"A comparative review of tone-mapping algorithms for high dynamic range video,\" Comput. Graph. Forum, vol. 36, no. 2, pp. 565–592, 2017.",
    "[6] G. Eilertsen, J. Kronander, G. Denes, R. K. Mantiuk, and J. Unger, \"HDR image reconstruction from a single exposure using deep CNNs,\" ACM Trans. Graph., vol. 36, no. 6, Art. 178, 2017.",
    "[7] D. Marnerides, T. Bashford-Rogers, J. Hatchett, and K. Debattista, \"ExpandNet: A deep convolutional neural network for high dynamic range expansion from low dynamic range content,\" Comput. Graph. Forum, vol. 37, no. 2, pp. 37–49, 2018.",
    "[8] D. Dalal, G. Vashishtha, P. Singh, and S. Raman, \"Single image LDR to HDR conversion using conditional diffusion,\" in Proc. IEEE ICIP, 2023, pp. 3533–3537.",
    "[9] C. Wang, Z. Xia, T. Leimkuehler, K. Myszkowski, and X. Zhang, \"LEDiff: Latent exposure diffusion for HDR generation,\" in Proc. IEEE/CVF CVPR, 2025, pp. 453–464.",
    "[10] D. Dalal, G. Vashishtha, P. Singh, and S. Raman, \"Single image LDR to HDR conversion using conditional diffusion,\" in Proc. IEEE ICIP, 2023, pp. 3533–3537.",
    "[11] C. Wang, Z. Xia, T. Leimkuehler, K. Myszkowski, and X. Zhang, \"LEDiff: Latent exposure diffusion for HDR generation,\" in Proc. IEEE/CVF CVPR, 2025, pp. 453–464.",
    "[12] A. Liu, Z. Liu, Z. Wang, D. Chen, B. Zeng, and S. Liu, \"ExpoCM: Exposure-aware one-step generative single-image HDR reconstruction,\" in Proc. IEEE/CVF CVPR, 2026.",
    "[13] A. Liu, Z. Liu, Z. Wang, D. Chen, B. Zeng, and S. Liu, \"ExpoCM: Exposure-aware one-step generative single-image HDR reconstruction,\" in Proc. IEEE/CVF CVPR, 2026.",
    "[14] C. Wang, Z. Xia, T. Leimkuehler, K. Myszkowski, and X. Zhang, \"LEDiff: Latent exposure diffusion for HDR generation,\" in Proc. IEEE/CVF CVPR, 2025, pp. 453–464.",
    "[15] A. Bansal, E. Borgnia, H.-M. Chu, J. S. Li, H. Kazemi, F. Huang, M. Goldblum, J. Geiping, and T. Goldstein, \"Cold diffusion: Inverting arbitrary image transforms without noise,\" in Proc. NeurIPS, 2023, pp. 41259–41282.",
    "[16] Z. Khan, M. Khanna, and S. Raman, \"FHDR: HDR image reconstruction from a single LDR image using feedback network,\" in Proc. IEEE GlobalSIP, 2019, pp. 1–5.",
    "[17] Y.-L. Liu, W.-S. Lai, Y.-S. Chen, Y.-L. Kao, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, \"Single-image HDR reconstruction by learning to reverse the camera pipeline,\" in Proc. IEEE/CVF CVPR, 2020, pp. 1651–1660.",
    "[18] A. Yan, N. Zhang, Y. Zhang, M. Xu, Q. Dai, and R. Timofte, \"Deep HDR imaging via a non-local network,\" IEEE Trans. Image Process., vol. 29, pp. 4308–4322, 2020.",
    "[19] X. Chen, Y. Liu, Z. Zhang, Y. Qiao, and C. Dong, \"HDRUNet: Single image HDR reconstruction with denoising and dequantization,\" in Proc. IEEE/CVF CVPR Workshops (NTIRE), 2021, pp. 354–363.",
    "[20] M. Santos, T. I. Ren, and N. K. Kalantari, \"Single image HDR reconstruction using a CNN with masked features and perceptual loss,\" ACM Trans. Graph., vol. 39, no. 4, Art. 80, 2020.",
    "[21] H. B. Barua, G. Krishnasamy, K. Wong, K. Stefanov, and A. Dhall, \"ArtHDR-Net: Perceptually realistic and accurate HDR content creation,\" in Proc. IEEE APSIPA ASC, 2023, pp. 806–812.",
    "[22] H. B. Barua, G. Krishnasamy, K. Wong, K. Stefanov, and A. Dhall, \"HistoHDR-Net: Histogram equalization for single LDR to HDR image translation,\" in Proc. IEEE ICIP, 2024, pp. 2730–2736.",
    "[23] J. Ho, A. Jain, and P. Abbeel, \"Denoising diffusion probabilistic models,\" in Proc. NeurIPS, 2020, pp. 6840–6851.",
    "[24] J. Song, C. Meng, and S. Ermon, \"Denoising diffusion implicit models,\" in Proc. ICLR, 2021.",
    "[25] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, \"High-resolution image synthesis with latent diffusion models,\" in Proc. IEEE/CVF CVPR, 2022, pp. 10684–10695.",
    "[26] T. Brooks, A. Holynski, and A. A. Efros, \"InstructPix2Pix: Learning to follow image editing instructions,\" in Proc. IEEE/CVF CVPR, 2023, pp. 18392–18402.",
    "[27] C. He, C. Fang, Y. Zhang, L. Tang, J. Huang, K. Li, Z. Guo, X. Li, and S. Farsiu, \"Reti-Diff: Illumination degradation image restoration with Retinex-based latent diffusion model,\" in Proc. ICLR, 2025.",
    "[28] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, \"LoRA: Low-rank adaptation of large language models,\" in Proc. ICLR, 2022.",
    "[29] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., \"Segment anything,\" in Proc. IEEE/CVF ICCV, 2023, pp. 4015–4026.",
    "[30] R. Zhang, A. Isola, P. Isola, E. Shechtman, and O. Wang, \"The unreasonable effectiveness of deep features as a perceptual metric,\" in Proc. IEEE/CVF CVPR, 2018, pp. 586–595.",
    "[31] R. K. Mantiuk, J. Dufaux, and F. Heide, \"HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions,\" ACM Trans. Graph., vol. 30, no. 4, Art. 40, 2011.",
    "[32] R. K. Mantiuk and A. Dąbrowski, \"HDR-VDP-3: A vision model for predicting quality of high dynamic range and wide color gamut images,\" in Proc. ACM SIGGRAPH, 2024.",
    "[33] M. Narwaria, R. K. Mantiuk, M. P. Da Silva, and P. Le Callet, \"HDR-VDP-2.2: A calibrated method for objective quality prediction of HDR content and displays,\" in Proc. IEEE IVMSP, 2015.",
    "[34] R. K. Mantiuk, K. Kim, A. G. Rempel, and W. Heidrich, \"Perceptually uniform encoding of high dynamic range images,\" in Proc. IEEE ICIP, 2015, pp. 2169–2173.",
    "[35] R. K. Mantiuk and A. Dąbrowski, \"PU21: A novel perceptually uniform encoding for adapting legacy LDR and HDR content to future displays,\" in Proc. Eurographics (Short Papers), 2021.",
  ].map(ref => new Paragraph({
    spacing: { after: 60, line: 240 },
    indent: { left: 720, hanging: 720 },
    children: [new TextRun({ text: ref, size: 22, font: "Times New Roman" })],
  })),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "TriGate-HDR", size: 20, font: "Times New Roman", italics: true })] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", size: 20, font: "Times New Roman" }), new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Times New Roman" })] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "TriGate_HDR_Paper_Draft.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Written: ${outPath} (${buffer.length} bytes)`);
}).catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
