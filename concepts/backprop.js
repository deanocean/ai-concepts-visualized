/* 反向傳播與連鎖律（大綱 §15）
   2 → 2 → 1 的小網路（隱藏層 Sigmoid、輸出層線性、損失 ½(ŷ − t)²），
   正向逐步算出每個節點，再從損失倒著逐步算出每個權重的梯度。
   點任一條連線看該權重的連鎖律連乘，並以數值微分驗算；可用梯度更新權重。
   文字在 locales/zh/backprop.js、locales/ja/backprop.js。 */
Atlas.define("backprop", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const X = [1.0, 0.5], T = 1.0, ETA = 0.2;
    const fmtL = (v) => (v < 0.001 ? Number(v.toPrecision(2)).toExponential(1).replace("e-", "e−") : fmt(v, 4));
    const W0 = { w1: 0.6, w2: -0.4, w3: 0.3, w4: 0.8, v1: 0.5, v2: -0.3 };
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const NAME = { w1: "w₁", w2: "w₂", w3: "w₃", w4: "w₄", v1: "v₁", v2: "v₂" };

    function forward(W) {
      const z1 = W.w1 * X[0] + W.w2 * X[1], h1 = sig(z1);
      const z2 = W.w3 * X[0] + W.w4 * X[1], h2 = sig(z2);
      const y = W.v1 * h1 + W.v2 * h2, L = 0.5 * (y - T) ** 2;
      return { z1, h1, z2, h2, y, L };
    }
    function backward(W, F) {
      const gy = F.y - T;
      const gv1 = gy * F.h1, gv2 = gy * F.h2, gh1 = gy * W.v1, gh2 = gy * W.v2;
      const d1 = F.h1 * (1 - F.h1), d2 = F.h2 * (1 - F.h2);
      const gz1 = gh1 * d1, gz2 = gh2 * d2;
      return { gy, gv1, gv2, gh1, gh2, d1, d2, gz1, gz2, w1: gz1 * X[0], w2: gz1 * X[1], w3: gz2 * X[0], w4: gz2 * X[1], v1: gv1, v2: gv2 };
    }

    let W = { ...W0 }, pos = 0, sel = "w1", hist = [];
    const MAXP = 7;

    /* ---------- controls ---------- */
    const phase = h("div", { class: "status" });
    const svgWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });
    const player = ui.player(ctx, () => { if (pos >= MAXP) return false; pos++; render(); return pos < MAXP; }, 900);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (pos < MAXP) { pos++; render(); } }, { primary: true });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); W = { ...W0 }; hist = []; pos = 0; render(); });
    const updBtn = ui.btn(t("btnUpdate", { eta: ETA }), () => { train(1); }, { primary: true });
    const upd10Btn = ui.btn(t("btnUpdate10"), () => { train(10); });
    const lossWrap = h("div");

    function train(n) {
      player.pause();
      if (!hist.length) hist.push(forward(W).L);
      for (let k = 0; k < n; k++) {
        const G = backward(W, forward(W));
        for (const key in W) W[key] -= ETA * G[key];
        hist.push(forward(W).L);
      }
      pos = 3; render();
    }

    bench.append(
      phase, svgWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, resetBtn),
      calc,
      h("div", { class: "bench__row", id: "bp-upd" }, updBtn, upd10Btn),
      lossWrap, note,
    );

    /* ---------- geometry ---------- */
    const VW = 700, VH = 300, R = 30;
    const P = { x1: [80, 90], x2: [80, 210], h1: [300, 90], h2: [300, 210], y: [500, 150], L: [632, 150] };
    const EDGES = [
      { k: "w1", a: "x1", b: "h1", lt: 0.26, gt: 0.7 },
      { k: "w2", a: "x2", b: "h1", lt: 0.26, gt: 0.7 },
      { k: "w3", a: "x1", b: "h2", lt: 0.26, gt: 0.7 },
      { k: "w4", a: "x2", b: "h2", lt: 0.26, gt: 0.7 },
      { k: "v1", a: "h1", b: "y", lt: 0.36, gt: 0.72 },
      { k: "v2", a: "h2", b: "y", lt: 0.36, gt: 0.72 },
    ];
    const PATH = {
      w1: ["w1", "h1y"], w2: ["w2", "h1y"], w3: ["w3", "h2y"], w4: ["w4", "h2y"], v1: ["v1"], v2: ["v2"],
    };
    function endPts(a, b) {
      const [x1, y1] = P[a], [x2, y2] = P[b], d = Math.hypot(x2 - x1, y2 - y1);
      const ux = (x2 - x1) / d, uy = (y2 - y1) / d;
      return [x1 + ux * R, y1 + uy * R, x2 - ux * (R + 4), y2 - uy * (R + 4)];
    }
    const lerp = (e, t) => [e[0] + (e[2] - e[0]) * t, e[1] + (e[3] - e[1]) * t];

    function pill(x, y, text, opt = {}) {
      const g = s("g");
      const w = text.length * 7 + 14;
      g.append(s("rect", { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 10, fill: opt.fill || "var(--panel)", stroke: opt.stroke || "var(--line-strong)", "stroke-width": opt.sw || 1, "fill-opacity": opt.fo == null ? 1 : opt.fo }));
      g.append(s("text", { x, y: y + 4, "text-anchor": "middle", "font-size": 11.5, "font-weight": opt.bold ? 700 : 500, class: "t-mono", fill: opt.color || "var(--ink)", text }));
      return g;
    }

    function render() {
      const F = forward(W), G = backward(W, F);
      const showH = pos >= 1, showY = pos >= 2, showL = pos >= 3;
      const gY = pos >= 4, gV = pos >= 5, gZ = pos >= 6, gW = pos >= 7;
      const selPath = gW ? PATH[sel] : [];

      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": t("svgAria"), style: "display:block;width:100%;min-width:560px;max-width:760px" });

      // layer captions
      [["layerIn", 80], ["layerHidden", 300], ["layerOut", 500], ["layerLoss", 632]].forEach(([key, x]) =>
        svg.append(s("text", { x, y: 22, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: t(key) })));

      // edges
      for (const e of EDGES) {
        const pts = endPts(e.a, e.b), on = selPath.includes(e.k);
        const live = e.k[0] === "w" ? showH : showY;
        svg.append(s("line", { x1: pts[0], y1: pts[1], x2: pts[2], y2: pts[3], stroke: on ? "var(--hot)" : live ? "var(--line-strong)" : "var(--line)", "stroke-width": on ? 3.5 : 1.8 }));
        const hit = s("line", { x1: pts[0], y1: pts[1], x2: pts[2], y2: pts[3], stroke: "transparent", "stroke-width": 18, style: "cursor:pointer" });
        hit.addEventListener("click", () => { sel = e.k; render(); });
        svg.append(hit);
      }
      // h → y path highlight (for w's chain)
      for (const [k, a] of [["h1y", "h1"], ["h2y", "h2"]]) {
        if (!selPath.includes(k)) continue;
        const pts = endPts(a, "y");
        svg.append(s("line", { x1: pts[0], y1: pts[1], x2: pts[2], y2: pts[3], stroke: "var(--hot)", "stroke-width": 3.5, "stroke-dasharray": "7 5", "pointer-events": "none" }));
      }
      if (gW) { // y → L highlight
        svg.append(s("line", { x1: P.y[0] + R, y1: 150, x2: P.L[0] - 36, y2: 150, stroke: "var(--hot)", "stroke-width": 3.5, "stroke-dasharray": "7 5" }));
      } else {
        svg.append(s("line", { x1: P.y[0] + R, y1: 150, x2: P.L[0] - 36, y2: 150, stroke: showL ? "var(--line-strong)" : "var(--line)", "stroke-width": 1.8 }));
      }

      // weight pills & gradient badges
      for (const e of EDGES) {
        const pts = endPts(e.a, e.b), on = sel === e.k && gW;
        const [lx, ly] = lerp(pts, e.lt);
        const p = pill(lx, ly, `${NAME[e.k]} ${fmt(W[e.k], 2)}`, { stroke: on ? "var(--hot)" : "var(--line-strong)", sw: on ? 2 : 1, bold: on });
        p.style.cursor = "pointer"; p.addEventListener("click", () => { sel = e.k; render(); });
        svg.append(p);
        const known = e.k[0] === "v" ? gV : gW;
        if (known) {
          const [gx, gy] = lerp(pts, e.gt);
          svg.append(pill(gx, gy, fmt(G[e.k], 3), { fill: "var(--hot-soft)", stroke: "var(--hot)", color: "var(--hot)", bold: true }));
        }
      }

      // nodes
      const node = (id, label, val, valOn, grad, gradOn) => {
        const [x, y] = P[id];
        const g = s("g");
        g.append(s("circle", { cx: x, cy: y, r: R, fill: valOn ? "var(--accent-soft)" : "var(--panel-2)", stroke: valOn ? "var(--accent)" : "var(--line-strong)", "stroke-width": 1.5 }));
        g.append(s("text", { x, y: y - 4, "text-anchor": "middle", "font-size": 13, "font-weight": 700, text: label }));
        g.append(s("text", { x, y: y + 13, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: valOn ? "var(--ink)" : "var(--faint)", text: valOn ? fmt(val, 3) : "?" }));
        svg.append(g);
        if (gradOn) svg.append(pill(x, y - R - 14, "∂ " + fmt(grad, 3), { fill: "var(--hot-soft)", stroke: "var(--hot)", color: "var(--hot)", bold: true }));
      };
      node("x1", "x₁", X[0], true);
      node("x2", "x₂", X[1], true);
      node("h1", "h₁", F.h1, showH, G.gh1, gV);
      node("h2", "h₂", F.h2, showH, G.gh2, gV);
      node("y", "ŷ", F.y, showY, G.gy, gY);
      // loss box
      const [Lx, Ly] = P.L;
      svg.append(s("rect", { x: Lx - 36, y: Ly - 26, width: 72, height: 52, rx: 8, fill: showL ? "var(--hot-soft)" : "var(--panel-2)", stroke: showL ? "var(--hot)" : "var(--line-strong)", "stroke-width": 1.5 }));
      svg.append(s("text", { x: Lx, y: Ly - 6, "text-anchor": "middle", "font-size": 13, "font-weight": 700, text: "L" }));
      svg.append(s("text", { x: Lx, y: Ly + 13, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: showL ? "var(--ink)" : "var(--faint)", text: showL ? fmtL(F.L) : "?" }));
      svg.append(s("text", { x: Lx, y: Ly + 46, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: t("target", { v: fmt(T, 1) }) }));
      if (gY) svg.append(pill(Lx, Ly - 40, "∂ 1", { fill: "var(--hot-soft)", stroke: "var(--hot)", color: "var(--hot)", bold: true }));
      // legend
      svg.append(s("text", { x: 14, y: VH - 12, "font-size": 11.5, class: "t-muted", text: t("legend") }));
      if (gW) svg.append(s("text", { x: VW - 14, y: VH - 12, "text-anchor": "end", "font-size": 11.5, fill: "var(--hot)", "font-weight": 700, text: t("clickEdge") }));
      svgWrap.replaceChildren(svg);

      /* ---------- phase + calc ---------- */
      const set = (k, msg) => { phase.className = "status status--" + k; phase.textContent = msg; };
      const row = (k, body) => `<span class="calc-row"><span class="k">${k}</span> ${body}</span>`;
      if (pos === 0) {
        set("info", t("status0"));
        calc.innerHTML = row(t("calcInK"), t("calcIn", { x1: fmt(X[0], 1), x2: fmt(X[1], 1), t: fmt(T, 1) })) +
          row(t("calcWK"), Object.keys(NAME).map((k) => `${NAME[k]} = ${fmt(W[k], 3)}`).join(t("listSep")));
      } else if (pos === 1) {
        set("info", t("status1"));
        calc.innerHTML =
          row("h₁ = σ(w₁x₁ + w₂x₂)", `= σ(${fmt(W.w1, 3)} × ${fmt(X[0], 1)} + ${fmtP(W.w2, 3)} × ${fmt(X[1], 1)}) = σ(${fmt(F.z1, 3)}) = <span class="hl">${fmt(F.h1, 3)}</span>`) +
          row("h₂ = σ(w₃x₁ + w₄x₂)", `= σ(${fmt(W.w3, 3)} × ${fmt(X[0], 1)} + ${fmtP(W.w4, 3)} × ${fmt(X[1], 1)}) = σ(${fmt(F.z2, 3)}) = <span class="hl">${fmt(F.h2, 3)}</span>`) +
          row(t("calcSigK"), t("calcSig"));
      } else if (pos === 2) {
        set("info", t("status2"));
        calc.innerHTML = row("ŷ = v₁h₁ + v₂h₂", `= ${fmt(W.v1, 3)} × ${fmt(F.h1, 3)} + ${fmtP(W.v2, 3)} × ${fmt(F.h2, 3)} = <span class="hl">${fmt(F.y, 3)}</span>`);
      } else if (pos === 3) {
        set(hist.length > 1 ? "ok" : "info", hist.length > 1 ? t("statusUpdated", { n: hist.length - 1, from: fmtL(hist[0]), to: fmtL(F.L) }) : t("status3"));
        calc.innerHTML = row("L = ½(ŷ − t)²", `= ½ × (${fmt(F.y, 3)} − ${fmt(T, 1)})² = ½ × ${fmtP(F.y - T, 3)}² = <span class="hl">${fmtL(F.L)}</span>`) +
          row(t("calcNextK"), t("calcNext"));
      } else if (pos === 4) {
        set("warn", t("status4"));
        calc.innerHTML = row(t("calcSelfK"), t("calcSelf")) +
          row("∂L/∂ŷ = ŷ − t", `= ${fmt(F.y, 3)} − ${fmt(T, 1)} = <span class="hl">${fmt(G.gy, 3)}</span>`) +
          row(t("calcMeanK"), t("calcMean", { d: fmt(G.gy * 0.01, 4) }));
      } else if (pos === 5) {
        set("warn", t("status5"));
        calc.innerHTML =
          row("∂L/∂v₁ = ∂L/∂ŷ × h₁", `= ${fmtP(G.gy, 3)} × ${fmt(F.h1, 3)} = <span class="hl">${fmt(G.gv1, 3)}</span>`) +
          row("∂L/∂v₂ = ∂L/∂ŷ × h₂", `= ${fmtP(G.gy, 3)} × ${fmt(F.h2, 3)} = <span class="hl">${fmt(G.gv2, 3)}</span>`) +
          row("∂L/∂h₁ = ∂L/∂ŷ × v₁", `= ${fmtP(G.gy, 3)} × ${fmtP(W.v1, 3)} = <span class="hl">${fmt(G.gh1, 3)}</span>`) +
          row("∂L/∂h₂ = ∂L/∂ŷ × v₂", `= ${fmtP(G.gy, 3)} × ${fmtP(W.v2, 3)} = <span class="hl">${fmt(G.gh2, 3)}</span>`);
      } else if (pos === 6) {
        set("warn", t("status6"));
        calc.innerHTML =
          row(t("calcSlopeK"), t("calcSlope")) +
          row("∂L/∂z₁ = ∂L/∂h₁ × h₁(1 − h₁)", `= ${fmtP(G.gh1, 3)} × ${fmt(G.d1, 3)} = <span class="hl">${fmt(G.gz1, 4)}</span>`) +
          row("∂L/∂z₂ = ∂L/∂h₂ × h₂(1 − h₂)", `= ${fmtP(G.gh2, 3)} × ${fmt(G.d2, 3)} = <span class="hl">${fmt(G.gz2, 4)}</span>`);
      } else {
        set("ok", t("status7", { w: NAME[sel] }));
        calc.innerHTML = chainText(sel, F, G);
      }

      const upd = bench.querySelector("#bp-upd");
      if (upd) upd.hidden = pos < MAXP;
      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= MAXP;

      note.innerHTML = t("note");

      drawLoss();
    }

    function chainText(k, F, G) {
      const row = (lab, body) => `<span class="calc-row"><span class="k">${lab}</span> ${body}</span>`;
      let sym, nums, terms;
      if (k[0] === "v") {
        const hh = k === "v1" ? "h₁" : "h₂", hv = k === "v1" ? F.h1 : F.h2;
        sym = `∂L/∂${NAME[k]} = ∂L/∂ŷ × ∂ŷ/∂${NAME[k]}`;
        terms = [["ŷ − t", G.gy], [hh, hv]];
      } else {
        const one = k === "w1" || k === "w2";
        const hh = one ? "h₁" : "h₂", zz = one ? "z₁" : "z₂", vv = one ? "v₁" : "v₂";
        const xx = k === "w1" || k === "w3" ? "x₁" : "x₂", xv = k === "w1" || k === "w3" ? X[0] : X[1];
        sym = `∂L/∂${NAME[k]} = ∂L/∂ŷ × ∂ŷ/∂${hh} × ∂${hh}/∂${zz} × ∂${zz}/∂${NAME[k]}`;
        terms = [["ŷ − t", G.gy], [vv, one ? W.v1 : W.v2], [`${hh}(1 − ${hh})`, one ? G.d1 : G.d2], [xx, xv]];
      }
      nums = terms.map((q) => fmtP(q[1], 3)).join(" × ");
      const eps = 1e-4, Wp = { ...W, [k]: W[k] + eps }, Wm = { ...W, [k]: W[k] - eps };
      const num = (forward(Wp).L - forward(Wm).L) / (2 * eps);
      return row(t("chainK"), sym) +
        row(t("chainTermsK"), terms.map((q) => q[0]).join(" × ")) +
        row(t("chainSubK"), `${nums} = <span class="hl">${fmt(G[k], 4)}</span>`) +
        row(t("chainCheckK", { w: NAME[k] }), `(L(${NAME[k]} + 0.0001) − L(${NAME[k]} − 0.0001)) ÷ 0.0002 = ${fmt(num, 4)}`) +
        row(t("chainUpdK"), `${t("chainUpdSym", { w: NAME[k] })} = ${fmt(W[k], 3)} − ${ETA} × ${fmtP(G[k], 4)} = <b>${fmt(W[k] - ETA * G[k], 3)}</b>`);
    }

    function drawLoss() {
      lossWrap.replaceChildren();
      if (hist.length < 2) return;
      const MW = 640, MH = 120, m = { l: 52, r: 16, t: 14, b: 26 };
      const mini = s("svg", { viewBox: `0 0 ${MW} ${MH}`, role: "img", "aria-label": t("lossLabel"), style: "display:block;width:100%;min-width:480px;max-width:760px" });
      const n = Math.max(10, hist.length - 1), ymax = hist[0];
      const px = (i) => m.l + (i / n) * (MW - m.l - m.r);
      const py = (v) => MH - m.b - (v / ymax) * (MH - m.t - m.b);
      mini.append(s("line", { x1: m.l, x2: MW - m.r, y1: py(0), y2: py(0), stroke: "var(--line-strong)" }));
      mini.append(s("line", { x1: m.l, x2: m.l, y1: m.t, y2: py(0), stroke: "var(--line-strong)" }));
      mini.append(s("text", { x: m.l - 6, y: py(ymax) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: fmt(ymax, 3) }));
      mini.append(s("text", { x: m.l - 6, y: py(0) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: "0" }));
      mini.append(s("text", { x: MW - m.r, y: MH - 6, "text-anchor": "end", "font-size": 11, class: "t-muted", text: t("lossAxis", { n }) }));
      let d = "";
      hist.forEach((v, i) => { d += (i ? "L" : "M") + px(i).toFixed(1) + "," + py(v).toFixed(1); });
      mini.append(s("path", { d, fill: "none", stroke: "var(--hot)", "stroke-width": 2 }));
      hist.forEach((v, i) => mini.append(s("circle", { cx: px(i), cy: py(v), r: 2.6, fill: "var(--hot)" })));
      lossWrap.append(h("p", { class: "field__label" }, t("lossLabel")), h("div", { class: "scroll-x" }, mini));
    }

    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
