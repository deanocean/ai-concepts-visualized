/* 梯度消失問題（大綱 §12・§15）
   每層只有一個神經元的深層網路（簡化），正向算出每層輸出後，
   從最後一層往前逐層傳回梯度：每過一層乘上 w × f′(z)。
   切換 Sigmoid／tanh／ReLU、層數與權重，觀察梯度消失與梯度爆炸。
   文字在 locales/zh/vanish.js、locales/ja/vanish.js。 */
Atlas.define("vanish", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const sigm = (x) => 1 / (1 + Math.exp(-x));
    const FN = {
      sigmoid: { label: "Sigmoid", f: sigm, d: (z) => sigm(z) * (1 - sigm(z)), dTxt: "σ(z)(1 − σ(z))" },
      tanh: { label: "tanh", f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2, dTxt: "1 − tanh(z)²" },
      relu: { label: "ReLU", f: (z) => Math.max(0, z), d: (z) => (z > 0 ? 1 : 0), dTxt: "z > 0 ? 1 : 0" },
    };
    const XIN = 1.0;
    let fk = "sigmoid", N = 8, w = 1.0, pos = 0;

    function run() {
      const F = FN[fk], z = [], a = [XIN];
      for (let k = 1; k <= N; k++) { const zk = w * a[k - 1]; z[k] = zk; a[k] = F.f(zk); }
      const fac = [], G = [];
      G[N] = 1;
      for (let k = N - 1; k >= 1; k--) { fac[k] = w * F.d(z[k + 1]); G[k] = G[k + 1] * fac[k]; }
      return { z, a, fac, G };
    }

    /* ---------- controls ---------- */
    const fnSeg = ui.seg({ label: t("fnLabel"), value: fk, options: Object.keys(FN).map((k) => ({ value: k, label: FN[k].label })), onChange: (v) => { fk = v; pos = N; render(); } });
    const nSlider = ui.slider({ id: "vg-n", label: t("nLabel"), min: 2, max: 12, step: 1, value: N, format: (v) => t("nFmt", { n: v }), onInput: (v) => { N = v; pos = Math.min(pos, N); if (pos) pos = N; render(); } });
    const wSlider = ui.slider({ id: "vg-w", label: t("wLabel"), min: 0.5, max: 2.5, step: 0.05, value: w, format: (v) => v.toFixed(2), onInput: (v) => { w = v; if (pos) pos = N; render(); } });
    const presetBox = h("div", { class: "btn-group" });
    [["sigmoid", 1], ["tanh", 1], ["relu", 1], ["relu", 1.5], ["relu", 0.7]].forEach(([f, ww]) =>
      presetBox.append(ui.btn(t("presetFmt", { f: FN[f].label, w: ww }), () => { player.pause(); fk = f; w = ww; fnSeg.set(f); wSlider.set(ww); pos = N; render(); }, { sm: true })));

    const player = ui.player(ctx, () => { if (pos >= N) return false; pos++; render(); return pos < N; }, 650);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (pos < N) { pos++; render(); } }, { primary: true });
    const allBtn = ui.btn(t("btnAll"), () => { player.pause(); pos = N; render(); });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); pos = 0; render(); });

    const status = h("div", { class: "status" });
    const svgWrap = h("div", { class: "scroll-x" });
    const barWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, fnSeg.el),
      h("div", { class: "bench__row" }, nSlider.el, wSlider.el),
      h("div", { class: "field" }, h("span", { class: "field__label" }, t("presetLabel")), presetBox),
      status, svgWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, resetBtn),
      calc, barWrap, note,
    );

    const fmtG = (v) => {
      if (v === 0) return "0";
      if (v >= 1000) return v.toExponential(1).replace("e+", "e");
      if (v >= 10) return v.toFixed(0);
      if (v >= 0.1) return v.toFixed(2);
      if (v >= 0.001) return v.toFixed(3);
      return Number(v.toPrecision(2)).toExponential(1).replace("e-", "e−");
    };

    function arrow(x1, y1, x2, y2, color, width = 1.6) {
      const g = s("g");
      g.append(s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": width }));
      const ang = Math.atan2(y2 - y1, x2 - x1), L = 6 + width;
      g.append(s("path", { d: `M${x2},${y2} L${x2 - L * Math.cos(ang - 0.45)},${y2 - L * Math.sin(ang - 0.45)} L${x2 - L * Math.cos(ang + 0.45)},${y2 - L * Math.sin(ang + 0.45)} Z`, fill: color }));
      return g;
    }

    function render() {
      const R = run();
      const COLW = 64, LEFT = 70, RIGHT = 84, VW = LEFT + COLW * N + RIGHT, VH = 170;
      const cx = (k) => LEFT + COLW * (k - 1) + COLW / 2;
      const yG = 30, yN = 110;
      const known = (k) => pos > 0 && k > N - pos;
      const cur = pos > 0 ? N - pos + 1 : -1;

      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": t("svgAria"), style: `display:block;width:100%;min-width:${Math.min(VW, 560)}px;max-width:${VW}px` });
      svg.append(s("text", { x: 8, y: yN + 4, "font-size": 12, "font-weight": 700, text: t("nodeIn") }));
      svg.append(s("text", { x: 8, y: yN + 20, "font-size": 11, class: "t-mono t-muted", text: "x = " + fmt(XIN, 1) }));
      if (pos > 0) svg.append(s("text", { x: 8, y: yG + 4, "font-size": 12, "font-weight": 700, fill: "var(--hot)", text: t("rowGrad") }));
      for (let k = 1; k <= N; k++) {
        const x = cx(k);
        svg.append(arrow(k === 1 ? 46 : cx(k - 1) + 22, yN, x - 24, yN, "var(--accent)", 1.6));
        const av = R.a[k];
        svg.append(s("rect", { x: x - 22, y: yN - 20, width: 44, height: 40, rx: 7, fill: "var(--accent-soft)", stroke: k === cur ? "var(--hot)" : "var(--accent)", "stroke-width": k === cur ? 2.5 : 1 }));
        svg.append(s("text", { x, y: yN + 5, "text-anchor": "middle", "font-size": 11.5, class: "t-mono", text: Math.abs(av) >= 100 ? av.toFixed(0) : fmt(av, 2) }));
        svg.append(s("text", { x, y: yN + 36, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: t("layerN", { k }) }));
        if (known(k)) {
          const g = R.G[k], alpha = Math.max(0.12, Math.min(1, Math.sqrt(Math.min(g, 1))));
          svg.append(s("rect", { x: x - 27, y: yG - 12, width: 54, height: 22, rx: 11, fill: "var(--hot)", "fill-opacity": g > 1 ? 1 : alpha, stroke: g > 1 ? "var(--bad)" : "var(--hot)", "stroke-width": g > 1 ? 2 : 1 }));
          svg.append(s("text", { x, y: yG + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 600, class: "t-mono", fill: (g > 1 || alpha > 0.55) ? "var(--panel)" : "var(--ink)", text: fmtG(g) }));
          if (k < N) svg.append(arrow(cx(k + 1) - 8, yG + 26, x + 8, yG + 26, "var(--hot)", 1.2 + 3 * Math.min(1, Math.sqrt(g))));
          if (k < N) svg.append(s("text", { x: x + COLW / 2, y: yG + 46, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", fill: "var(--hot)", text: "×" + fmt(R.fac[k], 2) }));
        }
      }
      const xe = cx(N) + 24;
      svg.append(arrow(xe, yN, xe + 24, yN, "var(--accent)", 1.6));
      svg.append(s("text", { x: xe + 30, y: yN - 2, "font-size": 12, text: t("nodeOut") }));
      svg.append(s("text", { x: xe + 30, y: yN + 14, "font-size": 12, text: t("nodeLoss") }));
      svgWrap.replaceChildren(svg);

      /* status & calc */
      const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      const F = FN[fk];
      if (pos === 0) {
        set("info", t("status0"));
        calc.innerHTML = row(t("calcLayerK"), t("calcLayer", { f: F.label })) +
          row(t("calcExK"), `f(${fmt(w, 2)} × ${fmt(XIN, 1)}) = f(${fmt(R.z[1], 2)}) = ${fmt(R.a[1], 3)}`);
      } else if (pos === 1) {
        set("warn", t("status1", { n: N }));
        calc.innerHTML = row(t("calcStartK"), t("calcStart"));
      } else {
        const k = N - pos + 1; // 剛算出的層
        const z1 = R.z[k + 1];
        calc.innerHTML =
          row(t("calcBackK", { from: k + 1, to: k }), fk === "relu"
            ? `w × f′(z<sub>${k + 1}</sub>)　<span class="k">${t(z1 > 0 ? "reluPos" : "reluNeg", { i: k + 1, z: fmt(z1, 2) })}</span> = ${fmt(w, 2)} × ${F.d(z1)} = <span class="hl">${fmt(R.fac[k], 3)}</span>`
            : `w × f′(z<sub>${k + 1}</sub>) = ${fmt(w, 2)} × ${F.dTxt.replace(/z/g, fmt(z1, 2))} = ${fmt(w, 2)} × ${fmt(F.d(z1), 3)} = <span class="hl">${fmt(R.fac[k], 3)}</span>`) +
          row(t("calcGradK"), `${fmtG(R.G[k + 1])} × ${fmt(R.fac[k], 3)} = <span class="hl">${fmtG(R.G[k])}</span>`);
        if (pos < N) set("warn", t("statusBack", { k, g: fmtG(R.G[k]) }));
      }
      if (pos === N && N > 1) {
        const g1 = R.G[1];
        if (g1 < 0.01) set("bad", t("statusVanish", { g: fmtG(g1), n: Math.round(1 / g1).toLocaleString("en-US") }));
        else if (g1 > 10) set("bad", t("statusExplode", { g: fmtG(g1) }));
        else if (g1 < 0.3) set("warn", t("statusDecay", { g: fmtG(g1) }));
        else set("ok", t("statusOk", { g: fmtG(g1) }));
      }

      /* bar chart (log scale) */
      barWrap.replaceChildren();
      if (pos > 0) {
        const vals = []; for (let k = 1; k <= N; k++) if (known(k)) vals.push(R.G[k]);
        const pos_ = vals.filter((v) => v > 0);
        let lo = Math.floor(Math.log10(Math.min(...pos_, 1))), hi = Math.ceil(Math.log10(Math.max(...pos_, 1)));
        lo = Math.min(lo, -1); hi = Math.max(hi, 0);
        if (hi - lo < 2) lo = hi - 2;
        const BH = 190, top = 26, base = BH - 30;
        const bs = s("svg", { viewBox: `0 0 ${VW} ${BH}`, role: "img", "aria-label": t("barAria"), style: `display:block;width:100%;min-width:${Math.min(VW, 560)}px;max-width:${VW}px` });
        const py = (v) => base - ((Math.log10(v) - lo) / (hi - lo)) * (base - top);
        bs.append(s("text", { x: 8, y: 14, "font-size": 12, "font-weight": 700, text: t("barTitle") }));
        const stepE = hi - lo > 6 ? 2 : 1;
        for (let e = lo; e <= hi; e += stepE) {
          bs.append(s("line", { x1: LEFT - 6, x2: cx(N) + 30, y1: py(10 ** e), y2: py(10 ** e), stroke: e === 0 ? "var(--line-strong)" : "var(--line)", "stroke-dasharray": e === 0 ? null : "3 3" }));
          const lab = e >= 0 ? String(10 ** e) : e >= -3 ? (10 ** e).toFixed(-e) : "1e−" + -e;
          bs.append(s("text", { x: LEFT - 10, y: py(10 ** e) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: lab }));
        }
        for (let k = 1; k <= N; k++) {
          const x = cx(k);
          bs.append(s("text", { x, y: base + 16, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: k }));
          if (!known(k)) continue;
          const v = R.G[k];
          if (v <= 0) { bs.append(s("text", { x, y: base - 4, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", fill: "var(--bad)", text: "0" })); continue; }
          const y1 = Math.min(py(v), py(10 ** lo)), y0 = py(10 ** lo);
          bs.append(s("rect", { x: x - 13, y: y1, width: 26, height: Math.max(1, y0 - y1), rx: 2, fill: v > 1 ? "var(--bad)" : "var(--hot)" }));
          bs.append(s("text", { x, y: y1 - 4, "text-anchor": "middle", "font-size": 10, class: "t-mono", text: fmtG(v) }));
        }
        bs.append(s("text", { x: cx(N) + 34, y: base + 16, "font-size": 11, class: "t-muted", text: t("barAxis") }));
        barWrap.append(bs);
      }

      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= N;
      allBtn.disabled = pos >= N;

      const NOTE = { sigmoid: "noteSigmoid", tanh: "noteTanh", relu: "noteRelu" };
      note.innerHTML = t(NOTE[fk]) + "　" + t("noteTail");
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
