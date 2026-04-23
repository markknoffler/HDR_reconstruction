/* ═══════════════════════════════════════════════════════════
   COLOR PALETTE
   ═══════════════════════════════════════════════════════════ */
const C = {
  bg:  "#080c16",
  card:"#0b1220",
  tok: { a:"#1e3a8a", b:"#1d4ed8", c:"#3b82f6", l:"#93c5fd" },
  ud:  { a:"#14532d", b:"#15803d", c:"#22c55e", l:"#86efac" },
  amr: { a:"#7f1d1d", b:"#b91c1c", c:"#ef4444", l:"#fca5a5" },
  xat: { a:"#3b0764", b:"#6d28d9", c:"#8b5cf6", l:"#c4b5fd" },
  fus: { a:"#451a03", b:"#92400e", c:"#f59e0b", l:"#fde68a" },
  mam: { a:"#083344", b:"#155e75", c:"#06b6d4", l:"#67e8f9" },
  grn: "#4ade80",
  txt: "#e2e8f0",
  mut: "#475569",
  dim: "#1e293b",
};

/* Pre-registered marker IDs -> colour */
const MK = {
  tok: C.tok.l, ud: C.ud.l, amr: C.amr.l, xat: C.xat.l,
  fus: C.fus.l, mam: C.mam.l, grn: C.grn,  mut: C.mut,
  tokb:C.tok.b, udb:C.ud.c,  amrb:C.amr.c, xatb:C.xat.c,
  fusb:C.fus.c, mamb:C.mam.c,
};

/* ═══════════════════════════════════════════════════════════
   SVG DEFS
   ═══════════════════════════════════════════════════════════ */
function Defs() {
  return (
    <defs>
      {Object.entries(MK).map(([id, col]) => (
        <marker key={id} id={`mk_${id}`}
          markerWidth={7} markerHeight={7} refX={5.5} refY={3.5} orient="auto">
          <polygon points="0 1,6 3.5,0 6" fill={col} />
        </marker>
      ))}
    </defs>
  );
}

/* ═══════════════════════════════════════════════════════════
   PRIMITIVE COMPONENTS
   ═══════════════════════════════════════════════════════════ */

function Cube({ x, y, w=32, h=52, d=12, col, label, dim, fs=9 }) {
  const sx = d * 0.88, sy = d * 0.46;
  return (
    <g>
      <polygon
        points={`${x},${y} ${x+w},${y} ${x+w+sx},${y-sy} ${x+sx},${y-sy}`}
        fill={col.l} opacity={0.82}/>
      <polygon
        points={`${x+w},${y} ${x+w+sx},${y-sy} ${x+w+sx},${y+h-sy} ${x+w},${y+h}`}
        fill={col.a}/>
      <rect x={x} y={y} width={w} height={h} fill={col.b} rx={1}/>
      {label &&
        <text x={x+w/2} y={y+h/2}
          textAnchor="middle" dominantBaseline="central"
          fill="#fff" fontSize={fs} fontWeight="900" fontFamily="monospace">{label}</text>}
      {dim &&
        <text x={x+(w+sx)/2} y={y-sy-4}
          textAnchor="middle" fill={col.l} fontSize={5.5} fontFamily="monospace">{dim}</text>}
    </g>
  );
}

function Rect({ x, y, w=72, h=22, r=5, col, label, sub, fs=8 }) {
  return (
    <g>
      <rect x={x-w/2} y={y-h/2} width={w} height={h} rx={r}
        fill={col.a} stroke={col.c} strokeWidth={1.4}/>
      <text x={x} y={sub ? y-4 : y}
        textAnchor="middle" dominantBaseline="central"
        fill={col.l} fontSize={fs} fontWeight="700" fontFamily="monospace">{label}</text>
      {sub &&
        <text x={x} y={y+6}
          textAnchor="middle" dominantBaseline="central"
          fill={C.mut} fontSize={6} fontFamily="monospace">{sub}</text>}
    </g>
  );
}

function Op({ x, y, r=10, sym, col }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r+3} fill={col.a} opacity={0.28}/>
      <circle cx={x} cy={y} r={r} fill={col.b} stroke={col.l} strokeWidth={1}/>
      <text x={x} y={y}
        textAnchor="middle" dominantBaseline="central"
        fill="#fff" fontSize={13} fontWeight="700">{sym}</text>
    </g>
  );
}

function Node({ x, y, r=13, label, col, fs=8 }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r+3} fill={col.a} opacity={0.25}/>
      <circle cx={x} cy={y} r={r}   fill={col.b} stroke={col.l} strokeWidth={1.4}/>
      <text x={x} y={y}
        textAnchor="middle" dominantBaseline="central"
        fill="#fff" fontSize={fs} fontWeight="800" fontFamily="monospace">{label}</text>
    </g>
  );
}

function A({ x1, y1, x2, y2, mk, dash=false, w=1.6, lbl="", lx, ly }) {
  const col = MK[mk] != null ? MK[mk] : mk;
  const mx = (x1+x2)/2;
  const my = (y1+y2)/2;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={col} strokeWidth={w}
        strokeDasharray={dash ? "4,3" : undefined}
        markerEnd={`url(#mk_${mk})`}/>
      {lbl &&
        <text x={lx != null ? lx : mx+4} y={ly != null ? ly : my-4}
          fill={col} fontSize={6.5} fontFamily="monospace">{lbl}</text>}
    </g>
  );
}

function PA({ d, mk, dash=false, w=1.6 }) {
  const col = MK[mk] != null ? MK[mk] : mk;
  return (
    <g>
      <path d={d} stroke={col} strokeWidth={w}
        strokeDasharray={dash ? "4,3" : undefined}
        fill="none" markerEnd={`url(#mk_${mk})`}/>
    </g>
  );
}

function Panel({ x, y, w, h, title, col, badge, children }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={9}
        fill={C.card} stroke={col.b} strokeWidth={1.5}/>
      <rect x={x} y={y}    width={w} height={26} rx={9} fill={col.a}/>
      <rect x={x} y={y+17} width={w} height={9}  fill={col.a}/>
      <text x={x+w/2} y={y+13}
        textAnchor="middle" dominantBaseline="central"
        fill="#fff" fontSize={8.5} fontWeight="800"
        letterSpacing="1.1" fontFamily="monospace">{title}</text>
      {badge && (
        <g>
          <circle cx={x+14} cy={y+13} r={9} fill={col.c} opacity={0.25}/>
          <text x={x+14} y={y+13}
            textAnchor="middle" dominantBaseline="central"
            fill={col.l} fontSize={9} fontWeight="900" fontFamily="monospace">{badge}</text>
        </g>
      )}
      {children}
    </g>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN DIAGRAM
   ═══════════════════════════════════════════════════════════ */
export default function NMTDiagram() {
  const W = 1080, H = 820;

  const hi="#7c3aed", md="#4c1d95", lo="#1e1040";
  const htA = [[hi,md,lo,lo],[lo,hi,md,lo],[lo,lo,hi,md],[md,lo,lo,hi]];
  const htB = [[lo,md,hi,lo],[hi,lo,md,lo],[lo,hi,lo,md],[md,lo,lo,hi]];

  return (
    <div style={{
      background:C.bg, minHeight:"100vh",
      display:"flex", flexDirection:"column",
      alignItems:"center", padding:"18px 8px",
      fontFamily:"'Courier New', monospace",
    }}>
      <div style={{color:C.txt, fontSize:16, fontWeight:800, letterSpacing:"0.05em", marginBottom:3}}>
        Graph-Driven Neural Machine Translation
      </div>
      <div style={{color:C.mut, fontSize:9.5, letterSpacing:"0.15em", marginBottom:16}}>
        LOW-RESOURCE · DUAL-GRAPH ENCODER · BIDIRECTIONAL CROSS-ATTENTION · MAMBA SSM DECODER
      </div>

      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
        style={{maxWidth:"100%", height:"auto"}}>
        <Defs/>

        {/* ═══════════════════════════════════════════════════════
             PANEL A — ARCHITECTURE OVERVIEW
        ═══════════════════════════════════════════════════════ */}
        <Panel x={6} y={6} w={1068} h={240} badge="A"
          title="(A)  ARCHITECTURE OVERVIEW — END-TO-END PIPELINE"
          col={C.xat}>

          <rect x={108} y={34} width={524} height={200} rx={6}
            fill={C.xat.a} opacity={0.12}/>
          <text x={370} y={46} textAnchor="middle"
            fill={C.xat.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            ENCODING  PATH
          </text>
          <rect x={746} y={34} width={314} height={200} rx={6}
            fill={C.mam.a} opacity={0.18}/>
          <text x={903} y={46} textAnchor="middle"
            fill={C.mam.l} fontSize={7} letterSpacing="1" fontFamily="monospace">
            DECODING  PATH
          </text>

          <rect x={10} y={82} width={90} height={102} rx={7}
            fill={C.dim} stroke={C.grn} strokeWidth={1.8}/>
          <text x={55} y={104} textAnchor="middle"
            fill={C.grn} fontSize={8.5} fontWeight="700" fontFamily="monospace">Source x</text>
          <text x={55} y={118} textAnchor="middle"
            fill="#86efac" fontSize={7} fontFamily="monospace">LRL sentence</text>
          <text x={55} y={131} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">x&#x2081; &#x2026; x&#x2099;</text>

          <PA d="M 100 118 C 106 118 108 80 114 80"   mk="tok"/>
          <A x1={100} y1={130} x2={114} y2={130}     mk="ud"/>
          <PA d="M 100 142 C 106 142 108 182 114 182" mk="amr"/>

          <Rect x={164} y={80}  w={84} h={22} col={C.tok} label="Emb + PE(i)" sub="no self-attn" fs={7.5}/>
          <A x1={206} y1={80}  x2={232} y2={80}  mk="tok"/>
          <Cube x={232} y={62} w={28} h={40} d={9} col={C.tok} label="T" dim="N×d"/>

          <Rect x={154} y={130} w={70} h={22} col={C.ud}  label="UD Parser" sub="Stanza/UDPipe" fs={7.5}/>
          <A x1={189} y1={130} x2={208} y2={130} mk="ud"  lbl="G_UD"/>
          <Rect x={250} y={130} w={60} h={22} col={C.ud}  label="UD-GAT" sub="L layers" fs={7.5}/>
          <A x1={280} y1={130} x2={302} y2={130} mk="ud"/>
          <Cube x={302} y={112} w={26} h={40} d={8} col={C.ud}  label="U" dim="N×d"/>

          <Rect x={154} y={182} w={70} h={22} col={C.amr} label="AMR Parser" sub="XLPT-AMR" fs={7.5}/>
          <A x1={189} y1={182} x2={208} y2={182} mk="amr" lbl="G_AMR"/>
          <Rect x={250} y={182} w={60} h={22} col={C.amr} label="AMR-GAT" sub="L layers" fs={7.5}/>
          <A x1={280} y1={182} x2={302} y2={182} mk="amr"/>
          <Cube x={302} y={164} w={26} h={40} d={8} col={C.amr} label="A" dim="|V|×d"/>

          <A x1={336} y1={130} x2={360} y2={130} mk="ud"/>
          <A x1={336} y1={182} x2={360} y2={182} mk="amr"/>

          <rect x={360} y={108} width={106} height={104} rx={6}
            fill={C.xat.a} stroke={C.xat.c} strokeWidth={1.5}/>
          <text x={413} y={128} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontWeight="800" fontFamily="monospace">Bidir Cross-</text>
          <text x={413} y={142} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontWeight="800" fontFamily="monospace">Graph Attn</text>
          <PA d="M 380 121 L 430 172" mk="udb"  dash={true} w={1.2}/>
          <PA d="M 430 121 L 380 172" mk="amrb" dash={true} w={1.2}/>
          <text x={413} y={182} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">&#x2192; &#x0168;, &#xC3;</text>
          <text x={413} y={194} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">O(|V_UD|&#xB7;|V_AMR|)</text>

          <PA d="M 262 68 C 290 54 472 54 478 64" mk="tok" dash={true} w={1.2}/>

          <PA d="M 466 138 C 476 132 485 125 490 120" mk="xat"/>
          <PA d="M 466 162 C 476 168 485 173 490 176" mk="xat"/>
          <Cube x={490} y={102} w={24} h={36} d={8} col={C.xat} label="&#x0168;" dim="N×d"/>
          <Cube x={490} y={162} w={24} h={36} d={8} col={C.xat} label="&#xC3;" dim="|V|×d"/>

          <PA d="M 476 74  C 548 68 556 118 558 134" mk="tok" dash={true} w={1.2}/>
          <PA d="M 522 118 C 538 118 548 136 554 136" mk="xat"/>
          <PA d="M 522 178 C 538 178 548 148 554 148" mk="xat"/>

          <rect x={558} y={110} width={90} height={62} rx={5}
            fill={C.fus.a} stroke={C.fus.c} strokeWidth={1.5}/>
          <text x={603} y={130} textAnchor="middle"
            fill={C.fus.l} fontSize={8} fontWeight="800" fontFamily="monospace">Concat</text>
          <text x={603} y={144} textAnchor="middle"
            fill={C.fus.l} fontSize={8} fontWeight="800" fontFamily="monospace">[T &#x2225; &#x0168; &#x2225; &#xC3;]</text>
          <text x={603} y={158} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">&#x2192; Linear &#x2192; LN</text>

          <A x1={648} y1={141} x2={670} y2={141} mk="fus"/>
          <Cube x={670} y={116} w={34} h={56} d={11} col={C.fus} label="S" dim="M×d"/>
          <text x={704} y={184} textAnchor="middle"
            fill={C.mut} fontSize={6} fontFamily="monospace">M=N+|V_UD|+|V_AMR|</text>

          <A x1={717} y1={141} x2={746} y2={141} mk="fus"/>
          <rect x={746} y={104} width={110} height={78} rx={6}
            fill={C.mam.a} stroke={C.mam.c} strokeWidth={1.5}/>
          <text x={801} y={124} textAnchor="middle"
            fill={C.mam.l} fontSize={8.5} fontWeight="800" fontFamily="monospace">Mamba SSM</text>
          <text x={801} y={138} textAnchor="middle"
            fill={C.mam.l} fontSize={7.5} fontFamily="monospace">Selective State</text>
          <text x={801} y={151} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">+ Cross-Attn &#x2192; S</text>
          <text x={801} y={164} textAnchor="middle"
            fill={C.mam.l} fontSize={7} fontFamily="monospace">O(Td&#xB2;) linear</text>

          <A x1={856} y1={141} x2={882} y2={141} mk="grn"/>
          <rect x={882} y={104} width={90} height={78} rx={7}
            fill={C.dim} stroke={C.grn} strokeWidth={1.8}/>
          <text x={927} y={128} textAnchor="middle"
            fill={C.grn} fontSize={9} fontWeight="700" fontFamily="monospace">Target y&#x209C;</text>
          <text x={927} y={143} textAnchor="middle"
            fill="#86efac" fontSize={7} fontFamily="monospace">generated</text>
          <text x={927} y={157} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">y&#x2081; &#x2026; y&#x2098;</text>

          <PA d="M 927 182 L 927 218 L 801 218 L 801 182" mk="mam" dash={true} w={1.2}/>
          <text x={864} y={226} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">autoregressive feedback</text>

          {[
            { bx:164, by:38, l:"B", col:C.ud  },
            { bx:413, by:38, l:"C", col:C.amr },
            { bx:603, by:38, l:"D", col:C.xat },
            { bx:801, by:38, l:"E", col:C.mam },
          ].map(({ bx, by, l, col: bc }) => (
            <g key={l}>
              <circle cx={bx} cy={by} r={9} fill={bc.b} opacity={0.25}/>
              <text x={bx} y={by}
                textAnchor="middle" dominantBaseline="central"
                fill={bc.l} fontSize={8} fontWeight="900" fontFamily="monospace">{l}</text>
            </g>
          ))}
        </Panel>

        {/* ═══════════════════════════════════════════════════════
             PANEL B — UD-GAT ENCODER DETAIL
        ═══════════════════════════════════════════════════════ */}
        <Panel x={6} y={254} w={518} h={264} badge="B"
          title="(B)  UD-GAT ENCODER — Multilingual Dependency Graph Attention"
          col={C.ud}>

          <text x={170} y={279} textAnchor="middle"
            fill={C.mut} fontSize={7.5} fontFamily="monospace">
            Input  G_UD = (V_UD, E_UD)  — dependency parse
          </text>

          <Node x={44}  y={308} r={13} label="The"     col={C.ud} fs={7.5}/>
          <Node x={116} y={300} r={13} label="doctor"  col={C.ud} fs={7}/>
          <Node x={195} y={293} r={13} label="exam."   col={C.ud} fs={7}/>
          <Node x={272} y={300} r={13} label="patient" col={C.ud} fs={6.5}/>
          <Node x={344} y={308} r={13} label="the"     col={C.ud} fs={7.5}/>

          <line x1={57}  y1={304} x2={103} y2={301} stroke={C.ud.l} strokeWidth={1.2}/>
          <text x={80}  y={296} textAnchor="middle" fill={C.ud.l} fontSize={6.5} fontFamily="monospace">det</text>

          <line x1={129} y1={298} x2={182} y2={295} stroke={C.ud.l} strokeWidth={1.2}/>
          <text x={156} y={288} textAnchor="middle" fill={C.ud.l} fontSize={6.5} fontFamily="monospace">nsubj</text>

          <line x1={208} y1={295} x2={259} y2={298} stroke={C.ud.l} strokeWidth={1.2}/>
          <text x={234} y={288} textAnchor="middle" fill={C.ud.l} fontSize={6.5} fontFamily="monospace">obj</text>

          <line x1={285} y1={303} x2={331} y2={306} stroke={C.ud.l} strokeWidth={1.2}/>
          <text x={308} y={296} textAnchor="middle" fill={C.ud.l} fontSize={6.5} fontFamily="monospace">det</text>

          <A x1={195} y1={321} x2={195} y2={337} mk="ud"/>

          <rect x={14} y={337} width={488} height={68} rx={5}
            fill="#071a0e" stroke={C.ud.b} strokeWidth={1.3}/>
          <text x={258} y={355} textAnchor="middle"
            fill={C.ud.l} fontSize={9} fontWeight="800" fontFamily="monospace">
            GAT Layer  &#x2113;  ( &#xD7; L_UD )
          </text>
          <text x={258} y={372} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace" fontStyle="italic">
            &#x3B1;_ij = softmax( LeakyReLU( a&#x1D40; [ Wh_i &#x2225; Wh_j ] ) )
          </text>
          <text x={258} y={388} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace" fontStyle="italic">
            h_i^(&#x2113;+1) = &#x3C3;( &#x3A3;_{"j &#x2208; N(i)"}  &#x3B1;_ij &#xB7; W &#xB7; h_j^(&#x2113;) )   — multi-hop neighbourhood aggregation
          </text>

          <A x1={258} y1={405} x2={258} y2={419} mk="ud"/>

          {["u&#x2081;","u&#x2082;","u&#x2083;","u&#x2084;","u&#x2085;"].map((lbl, i) => (
            <Cube key={i} x={28+i*92} y={419} w={32} h={44} d={9} col={C.ud}
              label={lbl} dim={i===2 ? "U N×d" : ""}/>
          ))}

          <rect x={14} y={476} width={140} height={16} rx={3}
            fill="#071a0e" stroke={C.ud.a}/>
          <text x={84} y={484} textAnchor="middle" dominantBaseline="central"
            fill={C.ud.l} fontSize={7} fontFamily="monospace">Stanza / UDPipe  100+ langs</text>

          <text x={496} y={484} textAnchor="end"
            fill={C.ud.l} fontSize={8} fontFamily="monospace" fontWeight="700">
            U &#x2208; &#x211D;&#x1D3A;&#xD7;&#x1D48;  &#x2192;  Cross-Graph Attn (D)
          </text>
        </Panel>

        {/* ═══════════════════════════════════════════════════════
             PANEL C — AMR CROSS-LINGUAL PIPELINE
        ═══════════════════════════════════════════════════════ */}
        <Panel x={530} y={254} w={544} h={264} badge="C"
          title="(C)  AMR CROSS-LINGUAL PIPELINE — MBSE-SPRING + XLPT-AMR"
          col={C.amr}>

          <text x={612} y={279} textAnchor="middle"
            fill={C.mut} fontSize={7.5} fontFamily="monospace">EN AMR anchor (training)</text>

          <rect x={550} y={284} width={124} height={18} rx={3}
            fill="#1a0808" stroke={C.amr.c}/>
          <text x={612} y={293} textAnchor="middle" dominantBaseline="central"
            fill={C.amr.l} fontSize={8} fontWeight="700" fontFamily="monospace">English text</text>

          <A x1={612} y1={302} x2={612} y2={316} mk="amr"/>

          <rect x={550} y={316} width={124} height={20} rx={4}
            fill={C.amr.a} stroke={C.amr.b} strokeWidth={1.3}/>
          <text x={612} y={326} textAnchor="middle" dominantBaseline="central"
            fill={C.amr.l} fontSize={8} fontWeight="700" fontFamily="monospace">MBSE-SPRING</text>
          <text x={612} y={338} textAnchor="middle" dominantBaseline="central"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">Smatch F1 = 85.9 (AMR 2.0)</text>

          <A x1={612} y1={348} x2={612} y2={362} mk="amr"/>

          <rect x={544} y={362} width={136} height={56} rx={4}
            fill="#130606" stroke={C.amr.a}/>
          <text x={612} y={375} textAnchor="middle"
            fill={C.amr.l} fontSize={7} fontFamily="monospace" fontWeight="700">AMR pseudo-label</text>
          <text x={548} y={390} fill="#f87171" fontSize={6.5} fontFamily="monospace">(examine-01</text>
          <text x={548} y={402} fill="#f87171" fontSize={6.5} fontFamily="monospace">  :ARG0 (d / doctor)</text>
          <text x={548} y={414} fill="#f87171" fontSize={6.5} fontFamily="monospace">  :ARG1 (p / patient))</text>

          <rect x={690} y={277} width={374} height={124} rx={6}
            fill="#100818" stroke={C.xat.c} strokeWidth={1.2}/>
          <text x={877} y={291} textAnchor="middle"
            fill={C.xat.l} fontSize={8.5} fontWeight="700" fontFamily="monospace">
            XLPT-AMR  Multi-Task Training
          </text>

          {[
            { y:308, txt:"&#x2460; AMR Parsing    EN &#x2192; AMR  (structure learning)",    col:C.amr.l },
            { y:324, txt:"&#x2461; AMR&#x2192;Text       AMR &#x2192; EN  (inverse; bidirectional)", col:C.fus.l },
            { y:340, txt:"&#x2462; Translation    EN &#x2194; LRL  (cross-lingual align.)",  col:C.ud.l  },
          ].map(({ y: ty, txt, col: tc }) => (
            <g key={ty}>
              <rect x={698} y={ty-8} width={358} height={15} rx={3} fill="#1c0e28"/>
              <text x={877} y={ty} textAnchor="middle" dominantBaseline="central"
                fill={tc} fontSize={6.5} fontFamily="monospace">{txt}</text>
            </g>
          ))}

          <text x={877} y={364} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">
            shared encoder &#x2192; language-agnostic repr.
          </text>
          <text x={877} y={376} textAnchor="middle"
            fill={C.xat.l} fontSize={7} fontFamily="monospace" fontWeight="700">
            Smatch F1 &#x2248; 70&#x2013;72  zero-shot  DE / ES / IT
          </text>
          <text x={877} y={390} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">
            &#x21B3; applies to genuinely LRL at inference time
          </text>

          <PA d="M 690 404 L 684 404 L 684 432 L 690 432" mk="xat" w={1.2}/>

          <rect x={544} y={426} width={136} height={18} rx={3}
            fill="#1a0808" stroke={C.amr.c}/>
          <text x={612} y={435} textAnchor="middle" dominantBaseline="central"
            fill={C.amr.l} fontSize={7.5} fontFamily="monospace" fontWeight="700">
            LRL input &#x2192; AMR  &#x2713;
          </text>

          <A x1={612} y1={444} x2={612} y2={458} mk="amr"/>
          <Cube x={596} y={458} w={32} h={46} d={9} col={C.amr} label="A" dim="|V|×d"/>

          <text x={642} y={470} fill={C.xat.l} fontSize={7} fontFamily="monospace">zero-shot transfer</text>
          <text x={642} y={482} fill={C.mut} fontSize={6.5} fontFamily="monospace">no LRL AMR annotation needed</text>
          <text x={642} y={494} fill={C.mut} fontSize={6.5} fontFamily="monospace">&#x2192; Panel D  Cross-Graph Attn</text>
        </Panel>

        {/* ═══════════════════════════════════════════════════════
             PANEL D — BIDIRECTIONAL CROSS-GRAPH ATTENTION
        ═══════════════════════════════════════════════════════ */}
        <Panel x={6} y={526} w={518} h={286} badge="D"
          title="(D)  BIDIRECTIONAL CROSS-GRAPH ATTENTION"
          col={C.xat}>

          <text x={148} y={551} textAnchor="middle"
            fill={C.ud.l} fontSize={7.5} fontFamily="monospace" fontWeight="700">
            U — UD syntax nodes
          </text>
          {["u1","u2","u3","u4","u5"].map((lbl, i) => (
            <Cube key={`ud${i}`} x={10+i*72} y={554} w={32} h={40} d={9} col={C.ud}
              label={`u${i+1}`}/>
          ))}

          <text x={148} y={653} textAnchor="middle"
            fill={C.amr.l} fontSize={7.5} fontFamily="monospace" fontWeight="700">
            A — AMR semantic nodes
          </text>
          {["a1","a2","a3","a4","a5"].map((lbl, i) => (
            <Cube key={`amr${i}`} x={10+i*72} y={656} w={32} h={40} d={9} col={C.amr}
              label={`a${i+1}`}/>
          ))}

          <A x1={370} y1={574} x2={396} y2={602} mk="ud"/>
          <A x1={370} y1={676} x2={396} y2={650} mk="amr"/>

          <rect x={396} y={596} width={112} height={48} rx={5}
            fill="#120830" stroke={C.xat.c} strokeWidth={1.4}/>
          <text x={452} y={614} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontWeight="800" fontFamily="monospace">A &#x2192; U  attention</text>
          <text x={452} y={630} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">
            &#xC3; = softmax(AW&#x1D60; (UW&#x1D3C;)&#x1D40;/&#x221A;d)(UW&#x1D53)
          </text>

          <rect x={396} y={650} width={112} height={48} rx={5}
            fill="#120830" stroke={C.xat.c} strokeWidth={1.4}/>
          <text x={452} y={668} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontWeight="800" fontFamily="monospace">U &#x2192; A  attention</text>
          <text x={452} y={684} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">
            &#x0168; = softmax(UW&#x1D60; (AW&#x1D3C;)&#x1D40;/&#x221A;d)(AW&#x1D53)
          </text>

          <PA d="M 396 620 C 382 625 382 645 396 652" mk="xat" w={1.3}/>
          <PA d="M 396 668 C 382 663 382 643 396 636" mk="xat" w={1.3} dash={true}/>
          <text x={378} y={640} textAnchor="middle"
            fill={C.xat.l} fontSize={15} fontWeight="700">&#x21D5;</text>

          <text x={422} y={570} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">attn (A&#x2192;U)</text>
          {htA.map((row, r) =>
            row.map((cl, ci) => (
              <rect key={`ha${r}${ci}`}
                x={398+ci*11} y={574+r*11} width={10} height={10} rx={1} fill={cl}/>
            ))
          )}

          <text x={422} y={724} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">attn (U&#x2192;A)</text>
          {htB.map((row, r) =>
            row.map((cl, ci) => (
              <rect key={`hb${r}${ci}`}
                x={398+ci*11} y={728+r*11} width={10} height={10} rx={1} fill={cl}/>
            ))
          )}

          <A x1={508} y1={620} x2={526} y2={620} mk="xat"/>
          <A x1={508} y1={674} x2={526} y2={674} mk="xat"/>

          <text x={390} y={557} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontFamily="monospace" fontWeight="700">&#xC3;</text>
          <Cube x={522} y={556} w={30} h={40} d={9} col={C.xat} label="&#xC3;" dim="N×d (&#x2192;F)"/>
          <Cube x={522} y={658} w={30} h={40} d={9} col={C.xat} label="&#x0168;" dim="N×d (&#x2192;F)"/>

          <rect x={8} y={760} width={500} height={24} rx={4}
            fill="#07031a" stroke={C.xat.a}/>
          <text x={258} y={772} textAnchor="middle" dominantBaseline="central"
            fill={C.xat.l} fontSize={7} fontFamily="monospace">
            Complexity:  O(|V_UD| &#xD7; |V_AMR|)  &#x226A;  O(N&#xB2;) full-sequence self-attention
          </text>
          <text x={258} y={787} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">
            graph node counts  (|V_UD| &#x2248; N,  |V_AMR| &#x226A; N)  make this negligible vs token-level attention
          </text>
        </Panel>

        {/* ═══════════════════════════════════════════════════════
             PANEL E — MAMBA SSM DECODER
        ═══════════════════════════════════════════════════════ */}
        <Panel x={530} y={526} w={544} h={286} badge="E"
          title="(E)  MAMBA SSM DECODER — Selective State Space  O(Td&#xB2;) vs O(T&#xB2;d)"
          col={C.mam}>

          <rect x={538} y={554} width={104} height={18} rx={3}
            fill="#061420" stroke={C.mam.c}/>
          <text x={590} y={563} textAnchor="middle" dominantBaseline="central"
            fill={C.mam.l} fontSize={7.5} fontFamily="monospace" fontWeight="700">
            y&#x2081;&#x2026;y&#x209C;&#x208B;&#x2081;  prefix
          </text>

          <A x1={642} y1={563} x2={660} y2={563} mk="mam"/>
          <Rect x={694} y={563} w={58} h={18} col={C.mam} label="Token Emb" fs={7.5}/>
          <A x1={723} y1={572} x2={723} y2={586} mk="mam"/>

          <PA d="M 723 590 C 723 600 678 608 672 616" mk="mam"/>
          <PA d="M 723 590 C 723 600 770 608 776 616" mk="mam"/>

          <Rect x={658} y={626} w={56} h={18} col={C.mam} label="Linear" fs={7.5}/>
          <A x1={658} y1={635} x2={658} y2={649} mk="mam"/>
          <Rect x={658} y={659} w={56} h={18} col={C.mam} label="Conv1D" fs={7.5}/>
          <A x1={658} y1={668} x2={658} y2={682} mk="mam"/>
          <Rect x={658} y={692} w={56} h={18} col={C.mam} label="SiLU &#x3C3;" fs={7.5}/>
          <A x1={658} y1={701} x2={658} y2={714} mk="mam"/>

          <rect x={616} y={714} width={88} height={64} rx={5}
            fill="#031018" stroke={C.mam.b} strokeWidth={1.6}/>
          <text x={660} y={730} textAnchor="middle"
            fill={C.mam.l} fontSize={9} fontWeight="900" fontFamily="monospace">SSM</text>
          <text x={660} y={744} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">h&#x209C; = &#x100;&#x209C;h&#x209C;&#x208B;&#x2081; + B&#x305;&#x209C;x&#x209C;</text>
          <text x={660} y={757} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace" fontStyle="italic">y&#x209C; = C&#x209C; &#xB7; h&#x209C;</text>
          <text x={660} y={770} textAnchor="middle"
            fill={C.mam.l} fontSize={6} fontFamily="monospace">selective  O(Td&#xB2;)</text>

          <rect x={710} y={720} width={76} height={48} rx={3}
            fill="#041624" stroke="#0891b2" strokeDasharray="3,2"/>
          <text x={748} y={735} textAnchor="middle"
            fill={C.mam.l} fontSize={7} fontFamily="monospace" fontWeight="700">Selective &#x394;</text>
          <text x={748} y={748} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">&#x394;  &#x2192;  &#x100;, B&#x305;, C</text>
          <text x={748} y={760} textAnchor="middle"
            fill={C.mut} fontSize={6.5} fontFamily="monospace">input-dependent</text>
          <A x1={710} y1={744} x2={704} y2={744} mk="mamb" w={1.2}/>

          <Rect x={776} y={626} w={56} h={18} col={C.mam} label="Linear" fs={7.5}/>
          <A x1={776} y1={635} x2={776} y2={746} mk="mam"/>

          <Op x={776} y={756} r={11} sym="&#x2297;" col={C.mam}/>
          <A x1={704} y1={756} x2={765} y2={756} mk="mam"/>
          <A x1={787} y1={756} x2={808} y2={756} mk="mam"/>

          <rect x={810} y={616} width={100} height={60} rx={5}
            fill="#040e18" stroke={C.xat.c} strokeWidth={1.5}/>
          <text x={860} y={634} textAnchor="middle"
            fill={C.xat.l} fontSize={8} fontWeight="700" fontFamily="monospace">Cross-Attn</text>
          <text x={860} y={648} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">Q: Mamba state</text>
          <text x={860} y={660} textAnchor="middle"
            fill={C.mut} fontSize={7} fontFamily="monospace">K, V: S (Fusion)</text>

          <rect x={828} y={554} width={64} height={16} rx={3}
            fill="#1e1204" stroke={C.fus.c}/>
          <text x={860} y={562} textAnchor="middle" dominantBaseline="central"
            fill={C.fus.l} fontSize={6.5} fontFamily="monospace">S  from Panel F&#x2192;A</text>
          <A x1={860} y1={570} x2={860} y2={616} mk="fus" lbl="S" lx={865} ly={595}/>

          <A x1={860} y1={676} x2={860} y2={746} mk="xat"/>

          <Op x={860} y={756} r={11} sym="&#x2295;" col={C.mam}/>
          <A x1={826} y1={756} x2={849} y2={756} mk="xat"/>
          <A x1={871} y1={756} x2={892} y2={756} mk="mam"/>

          <Rect x={922} y={756} w={50} h={18} col={C.mam} label="LN+Proj" fs={7}/>
          <A x1={922} y1={747} x2={922} y2={727} mk="mam"/>

          <rect x={898} y={698} width={48} height={24} rx={4}
            fill="#0a1f0a" stroke={C.grn} strokeWidth={1.6}/>
          <text x={922} y={710} textAnchor="middle" dominantBaseline="central"
            fill={C.grn} fontSize={12} fontFamily="monospace" fontWeight="900">y&#x209C;</text>

          <rect x={538} y={786} width={190} height={16} rx={3}
            fill="#040c18" stroke={C.mam.a}/>
          <text x={633} y={794} textAnchor="middle" dominantBaseline="central"
            fill={C.mam.l} fontSize={7} fontFamily="monospace">
            O(Td&#xB2;) vs O(T&#xB2;d) transformer self-attention
          </text>

          <text x={740} y={792} fill={C.mut} fontSize={6.5} fontFamily="monospace">h&#x209C; selective state:</text>
          {[...Array(12)].map((_, i) => (
            <rect key={i} x={742+i*10} y={796} width={9} height={12} rx={1}
              fill={C.mam.b} opacity={0.2+i*0.065}/>
          ))}
        </Panel>

      </svg>

      <div style={{
        color:C.mut, fontSize:9.5, fontFamily:"monospace",
        textAlign:"center", maxWidth:1060,
        marginTop:14, lineHeight:1.9,
      }}>
        <strong style={{color:C.txt}}>Sub-panels:</strong>{" "}
        (A) End-to-end pipeline — three parallel encoding streams, bidirectional fusion, Mamba decoder ·
        (B) UD-GAT — multilingual dependency graph attention with Stanza/UDPipe parsers ·
        (C) AMR cross-lingual pipeline — MBSE-SPRING English anchor + XLPT-AMR zero-shot transfer ·
        (D) Bidirectional cross-graph attention — semantics-informed syntax and syntax-informed semantics ·
        (E) Mamba SSM decoder — selective state + gating + cross-attention to fused source S
      </div>
    </div>
  );
}
