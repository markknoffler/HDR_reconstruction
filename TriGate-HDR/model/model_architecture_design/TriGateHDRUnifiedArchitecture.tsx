const C = {
  bg: "#080c16",
  card: "#0b1220",
  txt: "#e2e8f0",
  mut: "#475569",
  tri: { a: "#3b0764", b: "#6d28d9", c: "#8b5cf6", l: "#c4b5fd" },
  enc: { a: "#1e3a8a", b: "#1d4ed8", c: "#3b82f6", l: "#93c5fd" },
  dec: { a: "#14532d", b: "#15803d", c: "#22c55e", l: "#86efac" },
  gan: { a: "#451a03", b: "#92400e", c: "#f59e0b", l: "#fde68a" },
};

function Rect({ x, y, w = 120, h = 24, col, label }) {
  return (
    <g>
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={6} fill={col.a} stroke={col.c} strokeWidth={1.4} />
      <text x={x} y={y} textAnchor="middle" dominantBaseline="central" fill={col.l} fontSize={8} fontWeight={700} fontFamily="monospace">{label}</text>
    </g>
  );
}

function A({ x1, y1, x2, y2, col }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={1.8} markerEnd="url(#mk)" />;
}

export default function TriGateHDRUnifiedArchitecture() {
  return (
    <div style={{ background: C.bg, minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", padding: "16px" }}>
      <div style={{ color: C.txt, fontFamily: "monospace", fontWeight: 800 }}>TriGate HDR Modular Dual-Decoder + Seaming GAN</div>
      <svg width={1200} height={620} viewBox="0 0 1200 620" style={{ maxWidth: "100%", height: "auto" }}>
        <defs><marker id="mk" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><polygon points="0 1,6 3.5,0 6" fill={C.txt} /></marker></defs>
        <Rect x={110} y={130} col={C.enc} label="LDR Input" />
        <Rect x={320} y={80} col={C.enc} label="Structural Encoder" />
        <Rect x={320} y={130} col={C.enc} label="Material Encoder" />
        <Rect x={320} y={180} col={C.enc} label="Semantic Encoder" />
        <Rect x={540} y={130} col={C.tri} label="TriEncoder Gate Fusion" />
        <Rect x={760} y={90} col={C.dec} label="Stage 1 Dual Diffusion" />
        <Rect x={760} y={170} col={C.dec} label="Stage 2 CRF Recovery" />
        <Rect x={980} y={130} col={C.gan} label="Stage 3 Seaming GAN" />
        <Rect x={1110} y={130} col={C.dec} label="Final HDR" />
        <A x1={170} y1={130} x2={260} y2={80} col={C.enc.l} />
        <A x1={170} y1={130} x2={260} y2={130} col={C.enc.l} />
        <A x1={170} y1={130} x2={260} y2={180} col={C.enc.l} />
        <A x1={380} y1={80} x2={480} y2={130} col={C.tri.l} />
        <A x1={380} y1={130} x2={480} y2={130} col={C.tri.l} />
        <A x1={380} y1={180} x2={480} y2={130} col={C.tri.l} />
        <A x1={600} y1={120} x2={700} y2={90} col={C.dec.l} />
        <A x1={600} y1={140} x2={700} y2={170} col={C.dec.l} />
        <A x1={820} y1={90} x2={920} y2={120} col={C.gan.l} />
        <A x1={820} y1={170} x2={920} y2={140} col={C.gan.l} />
        <A x1={1040} y1={130} x2={1050} y2={130} col={C.dec.l} />
      </svg>
    </div>
  );
}

