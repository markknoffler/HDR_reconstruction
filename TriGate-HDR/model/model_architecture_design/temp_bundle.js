var TriGateDiagram = (() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // TriGateHDRUnifiedArchitecture.tsx
  var TriGateHDRUnifiedArchitecture_exports = {};
  __export(TriGateHDRUnifiedArchitecture_exports, {
    default: () => TriGateHDRUnifiedArchitecture
  });
  var import_react = __toESM(__require("react"));
  var C = {
    bg: "#080c16",
    card: "#0b1220",
    txt: "#e2e8f0",
    mut: "#475569",
    dim: "#1e293b",
    enc: { a: "#1e3a8a", b: "#1d4ed8", c: "#3b82f6", l: "#93c5fd" },
    tri: { a: "#3b0764", b: "#6d28d9", c: "#8b5cf6", l: "#c4b5fd" },
    dec: { a: "#14532d", b: "#15803d", c: "#22c55e", l: "#86efac" },
    rad: { a: "#7f1d1d", b: "#b91c1c", c: "#ef4444", l: "#fca5a5" },
    fus: { a: "#451a03", b: "#92400e", c: "#f59e0b", l: "#fde68a" },
    gan: { a: "#083344", b: "#155e75", c: "#06b6d4", l: "#67e8f9" },
    grn: "#4ade80"
  };
  var MK = {
    enc: C.enc.l,
    tri: C.tri.l,
    dec: C.dec.l,
    rad: C.rad.l,
    fus: C.fus.l,
    gan: C.gan.l,
    grn: C.grn,
    mut: C.mut
  };
  function Defs() {
    return /* @__PURE__ */ import_react.default.createElement("defs", null, Object.entries(MK).map(([id, col]) => /* @__PURE__ */ import_react.default.createElement("marker", { key: id, id: `mk_${id}`, markerWidth: 7, markerHeight: 7, refX: 5.5, refY: 3.5, orient: "auto" }, /* @__PURE__ */ import_react.default.createElement("polygon", { points: "0 1,6 3.5,0 6", fill: col }))));
  }
  function Cube({ x, y, w = 30, h = 46, d = 10, col, label = "", dim = "" }) {
    const sx = d * 0.88;
    const sy = d * 0.46;
    return /* @__PURE__ */ import_react.default.createElement("g", null, /* @__PURE__ */ import_react.default.createElement("polygon", { points: `${x},${y} ${x + w},${y} ${x + w + sx},${y - sy} ${x + sx},${y - sy}`, fill: col.l, opacity: 0.82 }), /* @__PURE__ */ import_react.default.createElement("polygon", { points: `${x + w},${y} ${x + w + sx},${y - sy} ${x + w + sx},${y + h - sy} ${x + w},${y + h}`, fill: col.a }), /* @__PURE__ */ import_react.default.createElement("rect", { x, y, width: w, height: h, fill: col.b, rx: 1 }), label && /* @__PURE__ */ import_react.default.createElement("text", { x: x + w / 2, y: y + h / 2, textAnchor: "middle", dominantBaseline: "central", fill: "#fff", fontSize: 8, fontWeight: "800", fontFamily: "monospace" }, label), dim && /* @__PURE__ */ import_react.default.createElement("text", { x: x + (w + sx) / 2, y: y - sy - 4, textAnchor: "middle", fill: col.l, fontSize: 6, fontFamily: "monospace" }, dim));
  }
  function Rect({ x, y, w = 90, h = 24, r = 5, col, label, sub = "" }) {
    return /* @__PURE__ */ import_react.default.createElement("g", null, /* @__PURE__ */ import_react.default.createElement("rect", { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: r, fill: col.a, stroke: col.c, strokeWidth: 1.4 }), /* @__PURE__ */ import_react.default.createElement("text", { x, y: sub ? y - 4 : y, textAnchor: "middle", dominantBaseline: "central", fill: col.l, fontSize: 8, fontWeight: "700", fontFamily: "monospace" }, label), sub && /* @__PURE__ */ import_react.default.createElement("text", { x, y: y + 6, textAnchor: "middle", dominantBaseline: "central", fill: C.mut, fontSize: 6, fontFamily: "monospace" }, sub));
  }
  function A({ x1, y1, x2, y2, mk = "mut", dash = false, w = 1.6, lbl = "", lx = null, ly = null }) {
    const col = MK[mk] != null ? MK[mk] : mk;
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    return /* @__PURE__ */ import_react.default.createElement("g", null, /* @__PURE__ */ import_react.default.createElement(
      "line",
      {
        x1,
        y1,
        x2,
        y2,
        stroke: col,
        strokeWidth: w,
        strokeDasharray: dash ? "4,3" : void 0,
        markerEnd: `url(#mk_${mk})`
      }
    ), lbl && /* @__PURE__ */ import_react.default.createElement("text", { x: lx != null ? lx : mx + 4, y: ly != null ? ly : my - 4, fill: col, fontSize: 6.5, fontFamily: "monospace" }, lbl));
  }
  function Panel({ x, y, w, h, title, badge, col, children = null }) {
    return /* @__PURE__ */ import_react.default.createElement("g", null, /* @__PURE__ */ import_react.default.createElement("rect", { x, y, width: w, height: h, rx: 9, fill: C.card, stroke: col.b, strokeWidth: 1.5 }), /* @__PURE__ */ import_react.default.createElement("rect", { x, y, width: w, height: 26, rx: 9, fill: col.a }), /* @__PURE__ */ import_react.default.createElement("rect", { x, y: y + 17, width: w, height: 9, fill: col.a }), /* @__PURE__ */ import_react.default.createElement(
      "text",
      {
        x: x + w / 2,
        y: y + 13,
        textAnchor: "middle",
        dominantBaseline: "central",
        fill: "#fff",
        fontSize: 8.5,
        fontWeight: "800",
        letterSpacing: "1.1",
        fontFamily: "monospace"
      },
      title
    ), badge && /* @__PURE__ */ import_react.default.createElement("g", null, /* @__PURE__ */ import_react.default.createElement("circle", { cx: x + 14, cy: y + 13, r: 9, fill: col.c, opacity: 0.25 }), /* @__PURE__ */ import_react.default.createElement("text", { x: x + 14, y: y + 13, textAnchor: "middle", dominantBaseline: "central", fill: col.l, fontSize: 9, fontWeight: "900", fontFamily: "monospace" }, badge)), children);
  }
  function TriGateHDRUnifiedArchitecture() {
    const W = 1200;
    const H = 860;
    return /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        style: {
          background: C.bg,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          padding: "18px 8px",
          fontFamily: "'Courier New', monospace"
        }
      },
      /* @__PURE__ */ import_react.default.createElement("div", { style: { color: C.txt, fontSize: 16, fontWeight: 800, letterSpacing: "0.05em", marginBottom: 3 } }, "TriGate HDR Reconstruction System"),
      /* @__PURE__ */ import_react.default.createElement("div", { style: { color: C.mut, fontSize: 9.5, letterSpacing: "0.15em", marginBottom: 16 } }, "TRI-ENCODER GROUNDING \xB7 DUAL DIFFUSION \xB7 RADIOMETRIC LOSS \xB7 SEAMING GAN"),
      /* @__PURE__ */ import_react.default.createElement("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: { maxWidth: "100%", height: "auto" } }, /* @__PURE__ */ import_react.default.createElement(Defs, null), /* @__PURE__ */ import_react.default.createElement(Panel, { x: 8, y: 8, w: 1184, h: 250, badge: "A", title: "(A)  END-TO-END PIPELINE", col: C.tri }, /* @__PURE__ */ import_react.default.createElement("rect", { x: 108, y: 36, width: 620, height: 208, rx: 6, fill: C.tri.a, opacity: 0.12 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 418, y: 47, textAnchor: "middle", fill: C.tri.l, fontSize: 7, letterSpacing: "1", fontFamily: "monospace" }, "ENCODER + FUSION PATH"), /* @__PURE__ */ import_react.default.createElement("rect", { x: 746, y: 36, width: 438, height: 208, rx: 6, fill: C.gan.a, opacity: 0.12 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 965, y: 47, textAnchor: "middle", fill: C.gan.l, fontSize: 7, letterSpacing: "1", fontFamily: "monospace" }, "STAGED DECODING + SEAMING"), /* @__PURE__ */ import_react.default.createElement("rect", { x: 16, y: 92, width: 92, height: 96, rx: 7, fill: C.dim, stroke: C.grn, strokeWidth: 1.8 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 62, y: 112, textAnchor: "middle", fill: C.grn, fontSize: 8.5, fontWeight: "700", fontFamily: "monospace" }, "LDR x"), /* @__PURE__ */ import_react.default.createElement("text", { x: 62, y: 126, textAnchor: "middle", fill: "#86efac", fontSize: 7, fontFamily: "monospace" }, "only input"), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 190, y: 96, w: 116, col: C.enc, label: "Material Enc", sub: "texture priors" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 190, y: 140, w: 116, col: C.enc, label: "Structural Enc", sub: "edges + gate" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 190, y: 184, w: 116, col: C.enc, label: "Semantic Enc", sub: "class latents" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 108, y1: 120, x2: 132, y2: 96, mk: "enc" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 108, y1: 140, x2: 132, y2: 140, mk: "enc" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 108, y1: 160, x2: 132, y2: 184, mk: "enc" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 366, y: 140, w: 152, h: 34, col: C.tri, label: "Horizontal Tri-Stream Fusion", sub: "cross-stream at each scale" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 248, y1: 96, x2: 292, y2: 128, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 248, y1: 140, x2: 292, y2: 140, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 248, y1: 184, x2: 292, y2: 152, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(Cube, { x: 450, y: 118, w: 34, h: 50, d: 10, col: C.dec, label: "S1", dim: "dual diff" }), /* @__PURE__ */ import_react.default.createElement(Cube, { x: 506, y: 118, w: 34, h: 50, d: 10, col: C.rad, label: "S2", dim: "rad rec" }), /* @__PURE__ */ import_react.default.createElement(Cube, { x: 562, y: 118, w: 34, h: 50, d: 10, col: C.gan, label: "S3", dim: "seam GAN" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 442, y1: 140, x2: 450, y2: 140, mk: "dec" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 494, y1: 140, x2: 506, y2: 140, mk: "rad" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 550, y1: 140, x2: 562, y2: 140, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 798, y: 95, w: 158, col: C.dec, label: "Stage 1: Luminance Diffusion", sub: "tri-encoder grounded" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 798, y: 145, w: 158, col: C.rad, label: "Stage 2: Recovery Diffusion", sub: "hybrid radiometric loss" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 798, y: 195, w: 158, col: C.gan, label: "Stage 3: Seaming GAN", sub: "encoders frozen" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 596, y1: 140, x2: 718, y2: 95, mk: "dec" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 596, y1: 140, x2: 718, y2: 145, mk: "rad" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 596, y1: 140, x2: 718, y2: 195, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 1020, y: 145, w: 150, h: 32, col: C.fus, label: "Masked Region Compositor", sub: "x + generated clipped parts" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 877, y1: 95, x2: 945, y2: 130, mk: "dec" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 877, y1: 145, x2: 945, y2: 145, mk: "rad" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 877, y1: 195, x2: 945, y2: 160, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 1102, y: 108, width: 74, height: 74, rx: 7, fill: C.dim, stroke: C.grn, strokeWidth: 1.8 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 1139, y: 132, textAnchor: "middle", fill: C.grn, fontSize: 8.5, fontWeight: "700", fontFamily: "monospace" }, "HDR y"), /* @__PURE__ */ import_react.default.createElement("text", { x: 1139, y: 145, textAnchor: "middle", fill: "#86efac", fontSize: 7, fontFamily: "monospace" }, "final"), /* @__PURE__ */ import_react.default.createElement("text", { x: 1139, y: 158, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace" }, "PSNR / SSIM"), /* @__PURE__ */ import_react.default.createElement(A, { x1: 1095, y1: 145, x2: 1102, y2: 145, mk: "grn" })), /* @__PURE__ */ import_react.default.createElement(Panel, { x: 8, y: 266, w: 586, h: 278, badge: "B", title: "(B)  STAGE-1 TRI-STREAM HORIZONTAL FUSION BLOCK", col: C.tri }, /* @__PURE__ */ import_react.default.createElement(Rect, { x: 96, y: 328, w: 120, col: C.enc, label: "Material Q", sub: "q_m = Wq m" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 96, y: 382, w: 120, col: C.enc, label: "Structural K,V", sub: "ks,vs = Wk,Wv s" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 96, y: 436, w: 120, col: C.enc, label: "Semantic K,V", sub: "km,vm = Wk,Wv e" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 156, y1: 328, x2: 208, y2: 356, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 156, y1: 382, x2: 208, y2: 368, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 156, y1: 436, x2: 208, y2: 380, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 218, y: 332, width: 166, height: 122, rx: 6, fill: C.tri.a, stroke: C.tri.c, strokeWidth: 1.4 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 352, textAnchor: "middle", fill: C.tri.l, fontSize: 8, fontWeight: "800", fontFamily: "monospace" }, "Cross-Stream Attention"), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 369, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace", fontStyle: "italic" }, "A_s = softmax(q_m k_s^T / sqrt(d))"), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 384, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace", fontStyle: "italic" }, "A_e = softmax(q_m k_e^T / sqrt(d))"), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 399, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace", fontStyle: "italic" }, "f_s = A_s v_s,   f_e = A_e v_e"), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 420, textAnchor: "middle", fill: C.tri.l, fontSize: 7, fontFamily: "monospace" }, "timestep gate: sigmoid(W_t t)"), /* @__PURE__ */ import_react.default.createElement("text", { x: 301, y: 437, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace" }, "fuse = [g1*m, g2*f_s, g3*f_e] ", "->", " 1x1"), /* @__PURE__ */ import_react.default.createElement(A, { x1: 384, y1: 392, x2: 424, y2: 392, mk: "tri" }), /* @__PURE__ */ import_react.default.createElement(Cube, { x: 424, y: 366, w: 36, h: 50, d: 10, col: C.dec, label: "+R", dim: "residual inject" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 470, y1: 392, x2: 510, y2: 392, mk: "dec" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 538, y: 392, w: 84, col: C.dec, label: "UNet scale \u2113", sub: "down/up path" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 16, y: 482, width: 560, height: 46, rx: 5, fill: "#100818", stroke: C.tri.b }), /* @__PURE__ */ import_react.default.createElement("text", { x: 296, y: 500, textAnchor: "middle", fill: C.tri.l, fontSize: 8, fontFamily: "monospace", fontWeight: "700" }, "Applied at multiple scales (high + low resolution) during diffusion denoising"), /* @__PURE__ */ import_react.default.createElement("text", { x: 296, y: 515, textAnchor: "middle", fill: C.mut, fontSize: 7, fontFamily: "monospace" }, "Encoders and decoder optimized jointly in Stage-1 (single train graph)")), /* @__PURE__ */ import_react.default.createElement(Panel, { x: 606, y: 266, w: 586, h: 278, badge: "C", title: "(C)  STAGE-2 HYBRID RADIOMETRIC CONSISTENCY LOSS", col: C.rad }, /* @__PURE__ */ import_react.default.createElement("rect", { x: 620, y: 312, width: 558, height: 52, rx: 6, fill: "#1f0a0a", stroke: C.rad.c }), /* @__PURE__ */ import_react.default.createElement("text", { x: 899, y: 332, textAnchor: "middle", fill: C.rad.l, fontSize: 8.5, fontWeight: "800", fontFamily: "monospace" }, "L_stage2 = \u03B1 L_crf_cycle + \u03B2 L_log_lum + \u03B3 L_exp_ratio + \u03B4 L_rolloff"), /* @__PURE__ */ import_react.default.createElement("text", { x: 899, y: 347, textAnchor: "middle", fill: C.mut, fontSize: 7, fontFamily: "monospace" }, "single objective; internal radiometric terms, no semantic W1 branch here"), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 730, y: 404, w: 206, col: C.rad, label: "L_crf_cycle", sub: "forward CRF + inverse consistency" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 1040, y: 404, w: 206, col: C.rad, label: "L_log_lum", sub: "scale-invariant log luminance" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 832, y1: 404, x2: 938, y2: 404, mk: "rad" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 730, y: 456, w: 206, col: C.rad, label: "L_exp_ratio", sub: "spatial exposure-ratio stability" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 1040, y: 456, w: 206, col: C.rad, label: "L_rolloff", sub: "highlight saturation transition" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 832, y1: 456, x2: 938, y2: 456, mk: "rad" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 666, y: 490, width: 466, height: 36, rx: 4, fill: "#130606", stroke: C.rad.b }), /* @__PURE__ */ import_react.default.createElement("text", { x: 899, y: 504, textAnchor: "middle", fill: C.rad.l, fontSize: 7.5, fontFamily: "monospace" }, "Targets radiometric recovery from LDR-like clipping trajectory"), /* @__PURE__ */ import_react.default.createElement("text", { x: 899, y: 517, textAnchor: "middle", fill: C.mut, fontSize: 6.5, fontFamily: "monospace" }, "preserves grading physics, avoids forcing semantic generation objective")), /* @__PURE__ */ import_react.default.createElement(Panel, { x: 8, y: 554, w: 1184, h: 298, badge: "D", title: "(D)  STAGE-3 SEAMING GAN + TRAINING POLICY", col: C.gan }, /* @__PURE__ */ import_react.default.createElement("rect", { x: 24, y: 592, width: 408, height: 228, rx: 7, fill: C.gan.a, opacity: 0.12 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 228, y: 604, textAnchor: "middle", fill: C.gan.l, fontSize: 7, letterSpacing: "1", fontFamily: "monospace" }, "INPUT STREAMS + COMPOSITING"), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 106, y: 656, w: 154, col: C.dec, label: "base HDR x (Stage-2)" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 106, y: 710, w: 154, col: C.dec, label: "generated clipped parts (Stage-1)" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 106, y: 764, w: 154, col: C.enc, label: "gate / seam mask" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 182, y1: 656, x2: 256, y2: 690, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 182, y1: 710, x2: 256, y2: 702, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 182, y1: 764, x2: 256, y2: 714, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 332, y: 702, w: 144, h: 30, col: C.fus, label: "masked region compositor", sub: "x + replaced clipped zones" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 448, y: 592, width: 350, height: 228, rx: 7, fill: C.gan.a, opacity: 0.12 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 623, y: 604, textAnchor: "middle", fill: C.gan.l, fontSize: 7, letterSpacing: "1", fontFamily: "monospace" }, "GENERATOR + DISCRIMINATORS"), /* @__PURE__ */ import_react.default.createElement(Cube, { x: 500, y: 660, w: 38, h: 56, d: 10, col: C.gan, label: "G", dim: "seam-aware" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 404, y1: 702, x2: 500, y2: 688, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 628, y: 688, w: 154, col: C.gan, label: "Global Disc Head", sub: "realism" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 628, y: 740, w: 154, col: C.gan, label: "Seam Disc Head", sub: "boundary artifacts" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 548, y1: 688, x2: 551, y2: 688, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement(A, { x1: 548, y1: 700, x2: 551, y2: 740, mk: "gan" }), /* @__PURE__ */ import_react.default.createElement("rect", { x: 816, y: 592, width: 360, height: 228, rx: 7, fill: C.dec.a, opacity: 0.12 }), /* @__PURE__ */ import_react.default.createElement("text", { x: 996, y: 604, textAnchor: "middle", fill: C.dec.l, fontSize: 7, letterSpacing: "1", fontFamily: "monospace" }, "STAGE POLICY + METRICS"), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 996, y: 654, w: 296, h: 28, col: C.dec, label: "Stage-1: train material + structural + semantic encoders with diffusion" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 996, y: 700, w: 296, h: 28, col: C.rad, label: "Stage-2: train recovery diffusion with hybrid radiometric loss only" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 996, y: 746, w: 296, h: 28, col: C.gan, label: "Stage-3: freeze encoders, train seaming GAN refinement" }), /* @__PURE__ */ import_react.default.createElement(Rect, { x: 996, y: 792, w: 296, h: 24, col: C.fus, label: "Metrics: PSNR\u03BC, SSIM, HDRVDP2, HDRVDP3 + CSV checkpoints" })))
    );
  }
  return __toCommonJS(TriGateHDRUnifiedArchitecture_exports);
})();
