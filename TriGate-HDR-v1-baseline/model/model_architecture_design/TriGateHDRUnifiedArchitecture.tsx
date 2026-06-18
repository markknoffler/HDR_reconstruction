import React from "react";

/* ═══════════════════════════════════════════════════════════
   COLOR PALETTE
   ═══════════════════════════════════════════════════════════ */
const C = {
  bg: "#f8fafc",
  card: "#ffffff",
  txt: "#0f172a",
  mut: "#64748b",
  dim: "#e2e8f0",
  enc: { a: "#dbeafe", b: "#93c5fd", c: "#60a5fa", l: "#1e3a8a" },
  tri: { a: "#ede9fe", b: "#c4b5fd", c: "#a78bfa", l: "#4c1d95" },
  dec: { a: "#dcfce7", b: "#86efac", c: "#4ade80", l: "#14532d" },
  rad: { a: "#fee2e2", b: "#fca5a5", c: "#f87171", l: "#7f1d1d" },
  fus: { a: "#fef3c7", b: "#fcd34d", c: "#f59e0b", l: "#78350f" },
  gan: { a: "#cffafe", b: "#67e8f9", c: "#22d3ee", l: "#155e75" },
  grn: "#16a34a",
};

const MK = {
  enc: C.enc.l,
  tri: C.tri.l,
  dec: C.dec.l,
  rad: C.rad.l,
  fus: C.fus.l,
  gan: C.gan.l,
  grn: C.grn,
  mut: C.mut,
};

function Defs() {
  return (
    <defs>
      {Object.entries(MK).map(([id, col]) => (
        <marker key={id} id={`mk_${id}`} markerWidth={7} markerHeight={7} refX={5.5} refY={3.5} orient="auto">
          <polygon points="0 1,6 3.5,0 6" fill={col} />
        </marker>
      ))}
    </defs>
  );
}

function Cube({ x, y, w = 30, h = 46, d = 10, col, label = "", dim = "" }) {
  const sx = d * 0.88;
  const sy = d * 0.46;
  return (
    <g>
      <polygon points={`${x},${y} ${x + w},${y} ${x + w + sx},${y - sy} ${x + sx},${y - sy}`} fill={col.l} opacity={0.82} />
      <polygon points={`${x + w},${y} ${x + w + sx},${y - sy} ${x + w + sx},${y + h - sy} ${x + w},${y + h}`} fill={col.a} />
      <rect x={x} y={y} width={w} height={h} fill={col.b} rx={1} />
      {label && (
        <text x={x + w / 2} y={y + h / 2} textAnchor="middle" dominantBaseline="central" fill={C.txt} fontSize={8} fontWeight="800" fontFamily="monospace">
          {label}
        </text>
      )}
      {dim && (
        <text x={x + (w + sx) / 2} y={y - sy - 4} textAnchor="middle" fill={col.l} fontSize={6} fontFamily="monospace">
          {dim}
        </text>
      )}
    </g>
  );
}

function Rect({ x, y, w = 90, h = 24, r = 5, col, label, sub = "" }) {
  return (
    <g>
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={r} fill={col.a} stroke={col.c} strokeWidth={1.4} />
      <text x={x} y={sub ? y - 4 : y} textAnchor="middle" dominantBaseline="central" fill={col.l} fontSize={8} fontWeight="700" fontFamily="monospace">
        {label}
      </text>
      {sub && (
        <text x={x} y={y + 6} textAnchor="middle" dominantBaseline="central" fill={C.mut} fontSize={6} fontFamily="monospace">
          {sub}
        </text>
      )}
    </g>
  );
}

function A({ x1, y1, x2, y2, mk = "mut", dash = false, w = 1.6, lbl = "", lx = null, ly = null }) {
  const col = MK[mk] != null ? MK[mk] : mk;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={col}
        strokeWidth={w}
        strokeDasharray={dash ? "4,3" : undefined}
        markerEnd={`url(#mk_${mk})`}
      />
      {lbl && (
        <text x={lx != null ? lx : mx + 4} y={ly != null ? ly : my - 4} fill={col} fontSize={6.5} fontFamily="monospace">
          {lbl}
        </text>
      )}
    </g>
  );
}

function Panel({ x, y, w, h, title, badge, col, children = null }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={C.card} stroke={col.b} strokeWidth={1.5} />
      <rect x={x} y={y} width={w} height={26} rx={9} fill={col.a} />
      <rect x={x} y={y + 17} width={w} height={9} fill={col.a} />
      <text
        x={x + w / 2}
        y={y + 13}
        textAnchor="middle"
        dominantBaseline="central"
        fill={C.txt}
        fontSize={8.5}
        fontWeight="800"
        letterSpacing="1.1"
        fontFamily="monospace"
      >
        {title}
      </text>
      {badge && (
        <g>
          <circle cx={x + 14} cy={y + 13} r={9} fill={col.c} opacity={0.25} />
          <text x={x + 14} y={y + 13} textAnchor="middle" dominantBaseline="central" fill={col.l} fontSize={9} fontWeight="900" fontFamily="monospace">
            {badge}
          </text>
        </g>
      )}
      {children}
    </g>
  );
}

export default function TriGateHDRUnifiedArchitecture() {
  const W = 1200;
  const H = 860;
  return (
    <div
      style={{
        background: C.bg,
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "18px 8px",
        fontFamily: "'Courier New', monospace",
      }}
    >
      <div style={{ color: C.txt, fontSize: 16, fontWeight: 800, letterSpacing: "0.05em", marginBottom: 3 }}>
        TriGate HDR Reconstruction System
      </div>
      <div style={{ color: C.mut, fontSize: 9.5, letterSpacing: "0.15em", marginBottom: 16 }}>
        TRI-ENCODER GROUNDING · DUAL DIFFUSION · RADIOMETRIC LOSS · SEAMING GAN
      </div>

      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: "100%", height: "auto" }}>
        <Defs />

        <Panel x={8} y={8} w={1184} h={250} badge="A" title="(A)  END-TO-END PIPELINE" col={C.tri}>
          <rect x={108} y={36} width={620} height={208} rx={6} fill={C.tri.a} opacity={0.12} />
          <text x={418} y={47} textAnchor="middle" fill={C.tri.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            ENCODER + FUSION PATH
          </text>
          <rect x={746} y={36} width={438} height={208} rx={6} fill={C.gan.a} opacity={0.12} />
          <text x={965} y={47} textAnchor="middle" fill={C.gan.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            STAGED DECODING + SEAMING
          </text>

          <rect x={16} y={92} width={92} height={96} rx={7} fill={C.dim} stroke={C.grn} strokeWidth={1.8} />
          <text x={62} y={112} textAnchor="middle" fill={C.grn} fontSize={8.5} fontWeight="700" fontFamily="monospace">
            LDR x
          </text>
          <text x={62} y={126} textAnchor="middle" fill="#86efac" fontSize={7} fontFamily="monospace">
            only input
          </text>

          <Rect x={190} y={96} w={116} col={C.enc} label="Material Enc" sub="texture priors" />
          <Rect x={190} y={140} w={116} col={C.enc} label="Structural Enc" sub="edges + gate" />
          <Rect x={190} y={184} w={116} col={C.enc} label="Semantic Enc" sub="class latents" />
          <A x1={108} y1={120} x2={132} y2={96} mk="enc" />
          <A x1={108} y1={140} x2={132} y2={140} mk="enc" />
          <A x1={108} y1={160} x2={132} y2={184} mk="enc" />

          <Rect x={366} y={140} w={152} h={34} col={C.tri} label="Horizontal Tri-Stream Fusion" sub="cross-stream at each scale" />
          <A x1={248} y1={96} x2={292} y2={128} mk="tri" />
          <A x1={248} y1={140} x2={292} y2={140} mk="tri" />
          <A x1={248} y1={184} x2={292} y2={152} mk="tri" />

          <Cube x={450} y={118} w={34} h={50} d={10} col={C.dec} label="S1" dim="dual diff" />
          <Cube x={506} y={118} w={34} h={50} d={10} col={C.rad} label="S2" dim="rad rec" />
          <Cube x={562} y={118} w={34} h={50} d={10} col={C.gan} label="S3" dim="seam GAN" />
          <A x1={442} y1={140} x2={450} y2={140} mk="dec" />
          <A x1={494} y1={140} x2={506} y2={140} mk="rad" />
          <A x1={550} y1={140} x2={562} y2={140} mk="gan" />

          <Rect x={798} y={95} w={158} col={C.dec} label="Stage 1: Luminance Diffusion" sub="tri-encoder grounded" />
          <Rect x={798} y={145} w={158} col={C.rad} label="Stage 2: Recovery Diffusion" sub="hybrid radiometric loss" />
          <Rect x={798} y={195} w={158} col={C.gan} label="Stage 3: Seaming GAN" sub="encoders frozen" />
          <A x1={596} y1={140} x2={718} y2={95} mk="dec" />
          <A x1={596} y1={140} x2={718} y2={145} mk="rad" />
          <A x1={596} y1={140} x2={718} y2={195} mk="gan" />

          <Rect x={1020} y={145} w={150} h={32} col={C.fus} label="Masked Region Compositor" sub="x + generated clipped parts" />
          <A x1={877} y1={95} x2={945} y2={130} mk="dec" />
          <A x1={877} y1={145} x2={945} y2={145} mk="rad" />
          <A x1={877} y1={195} x2={945} y2={160} mk="gan" />

          <rect x={1102} y={108} width={74} height={74} rx={7} fill={C.dim} stroke={C.grn} strokeWidth={1.8} />
          <text x={1139} y={132} textAnchor="middle" fill={C.grn} fontSize={8.5} fontWeight="700" fontFamily="monospace">
            HDR y
          </text>
          <text x={1139} y={145} textAnchor="middle" fill="#86efac" fontSize={7} fontFamily="monospace">
            final
          </text>
          <text x={1139} y={158} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace">
            PSNR / SSIM
          </text>
          <A x1={1095} y1={145} x2={1102} y2={145} mk="grn" />
        </Panel>

        <Panel x={8} y={266} w={586} h={278} badge="B" title="(B)  STAGE-1 TRI-STREAM HORIZONTAL FUSION BLOCK" col={C.tri}>
          <Rect x={96} y={328} w={120} col={C.enc} label="Material Q" sub="q_m = Wq m" />
          <Rect x={96} y={382} w={120} col={C.enc} label="Structural K,V" sub="ks,vs = Wk,Wv s" />
          <Rect x={96} y={436} w={120} col={C.enc} label="Semantic K,V" sub="km,vm = Wk,Wv e" />
          <A x1={156} y1={328} x2={208} y2={356} mk="tri" />
          <A x1={156} y1={382} x2={208} y2={368} mk="tri" />
          <A x1={156} y1={436} x2={208} y2={380} mk="tri" />

          <rect x={218} y={332} width={166} height={122} rx={6} fill={C.tri.a} stroke={C.tri.c} strokeWidth={1.4} />
          <text x={301} y={352} textAnchor="middle" fill={C.tri.l} fontSize={8} fontWeight="800" fontFamily="monospace">
            Cross-Stream Attention
          </text>
          <text x={301} y={369} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">
            A_s = softmax(q_m k_s^T / sqrt(d))
          </text>
          <text x={301} y={384} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">
            A_e = softmax(q_m k_e^T / sqrt(d))
          </text>
          <text x={301} y={399} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">
            f_s = A_s v_s,   f_e = A_e v_e
          </text>
          <text x={301} y={420} textAnchor="middle" fill={C.tri.l} fontSize={7} fontFamily="monospace">
            timestep gate: sigmoid(W_t t)
          </text>
          <text x={301} y={437} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace">
            fuse = [g1*m, g2*f_s, g3*f_e] {"->"} 1x1
          </text>
          <A x1={384} y1={392} x2={424} y2={392} mk="tri" />

          <Cube x={424} y={366} w={36} h={50} d={10} col={C.dec} label="+R" dim="residual inject" />
          <A x1={470} y1={392} x2={510} y2={392} mk="dec" />
          <Rect x={538} y={392} w={84} col={C.dec} label="UNet scale ℓ" sub="down/up path" />

          <rect x={16} y={482} width={560} height={46} rx={5} fill="#100818" stroke={C.tri.b} />
          <text x={296} y={500} textAnchor="middle" fill={C.tri.l} fontSize={8} fontFamily="monospace" fontWeight="700">
            Applied at multiple scales (high + low resolution) during diffusion denoising
          </text>
          <text x={296} y={515} textAnchor="middle" fill={C.mut} fontSize={7} fontFamily="monospace">
            Encoders and decoder optimized jointly in Stage-1 (single train graph)
          </text>
        </Panel>

        <Panel x={606} y={266} w={586} h={278} badge="C" title="(C)  STAGE-2 HYBRID RADIOMETRIC CONSISTENCY LOSS" col={C.rad}>
          <rect x={620} y={312} width={558} height={52} rx={6} fill="#1f0a0a" stroke={C.rad.c} />
          <text x={899} y={332} textAnchor="middle" fill={C.rad.l} fontSize={8.5} fontWeight="800" fontFamily="monospace">
            L_stage2 = α L_crf_cycle + β L_log_lum + γ L_exp_ratio + δ L_rolloff
          </text>
          <text x={899} y={347} textAnchor="middle" fill={C.mut} fontSize={7} fontFamily="monospace">
            single objective; internal radiometric terms, no semantic W1 branch here
          </text>

          <Rect x={730} y={404} w={206} col={C.rad} label="L_crf_cycle" sub="forward CRF + inverse consistency" />
          <Rect x={1040} y={404} w={206} col={C.rad} label="L_log_lum" sub="scale-invariant log luminance" />
          <A x1={832} y1={404} x2={938} y2={404} mk="rad" />

          <Rect x={730} y={456} w={206} col={C.rad} label="L_exp_ratio" sub="spatial exposure-ratio stability" />
          <Rect x={1040} y={456} w={206} col={C.rad} label="L_rolloff" sub="highlight saturation transition" />
          <A x1={832} y1={456} x2={938} y2={456} mk="rad" />

          <rect x={666} y={490} width={466} height={36} rx={4} fill="#130606" stroke={C.rad.b} />
          <text x={899} y={504} textAnchor="middle" fill={C.rad.l} fontSize={7.5} fontFamily="monospace">
            Targets radiometric recovery from LDR-like clipping trajectory
          </text>
          <text x={899} y={517} textAnchor="middle" fill={C.mut} fontSize={6.5} fontFamily="monospace">
            preserves grading physics, avoids forcing semantic generation objective
          </text>
        </Panel>

        <Panel x={8} y={554} w={1184} h={298} badge="D" title="(D)  STAGE-3 SEAMING GAN + TRAINING POLICY" col={C.gan}>
          <rect x={24} y={592} width={408} height={228} rx={7} fill={C.gan.a} opacity={0.12} />
          <text x={228} y={604} textAnchor="middle" fill={C.gan.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            INPUT STREAMS + COMPOSITING
          </text>
          <Rect x={106} y={656} w={154} col={C.dec} label="base HDR x (Stage-2)" />
          <Rect x={106} y={710} w={154} col={C.dec} label="generated clipped parts (Stage-1)" />
          <Rect x={106} y={764} w={154} col={C.enc} label="gate / seam mask" />
          <A x1={182} y1={656} x2={256} y2={690} mk="gan" />
          <A x1={182} y1={710} x2={256} y2={702} mk="gan" />
          <A x1={182} y1={764} x2={256} y2={714} mk="gan" />
          <Rect x={332} y={702} w={144} h={30} col={C.fus} label="masked region compositor" sub="x + replaced clipped zones" />

          <rect x={448} y={592} width={350} height={228} rx={7} fill={C.gan.a} opacity={0.12} />
          <text x={623} y={604} textAnchor="middle" fill={C.gan.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            GENERATOR + DISCRIMINATORS
          </text>
          <Cube x={500} y={660} w={38} h={56} d={10} col={C.gan} label="G" dim="seam-aware" />
          <A x1={404} y1={702} x2={500} y2={688} mk="gan" />
          <Rect x={628} y={688} w={154} col={C.gan} label="Global Disc Head" sub="realism" />
          <Rect x={628} y={740} w={154} col={C.gan} label="Seam Disc Head" sub="boundary artifacts" />
          <A x1={548} y1={688} x2={551} y2={688} mk="gan" />
          <A x1={548} y1={700} x2={551} y2={740} mk="gan" />

          <rect x={816} y={592} width={360} height={228} rx={7} fill={C.dec.a} opacity={0.12} />
          <text x={996} y={604} textAnchor="middle" fill={C.dec.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            STAGE POLICY + METRICS
          </text>
          <Rect x={996} y={654} w={296} h={28} col={C.dec} label="Stage-1: train material + structural + semantic encoders with diffusion" />
          <Rect x={996} y={700} w={296} h={28} col={C.rad} label="Stage-2: train recovery diffusion with hybrid radiometric loss only" />
          <Rect x={996} y={746} w={296} h={28} col={C.gan} label="Stage-3: freeze encoders, train seaming GAN refinement" />
          <Rect x={996} y={792} w={296} h={24} col={C.fus} label="Metrics: PSNRμ, SSIM, HDRVDP2, HDRVDP3 + CSV checkpoints" />
        </Panel>
      </svg>
    </div>
  );
}

