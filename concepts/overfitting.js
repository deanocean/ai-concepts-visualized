/* 過擬合與正則化（L1・L2）（大綱 §10・§14）
   10 個帶雜訊的訓練點（真正的規律：y = sin 2πx），用 1〜9 次多項式擬合。
   調次數看訓練誤差與驗證誤差怎麼分開；加上 L2（嶺迴歸）或 L1（套索迴歸），看係數怎麼被壓小或變成 0。
   L2 用閉式解，L1 用座標下降法；所有數字即時計算。
   文字在 locales/zh/overfit.js、locales/ja/overfit.js。 */
(function () {
  /* ---------- 資料（固定亂數種子，每次相同） ---------- */
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const truth = (x) => Math.sin(2 * Math.PI * x);
  const SIGMA = 0.25;
  const r = rng(20241108);
  const TRAIN = Array.from({ length: 10 }, (_, i) => { const x = i / 9; return { x, y: truth(x) + SIGMA * gauss(r) }; });
  const VALID = Array.from({ length: 20 }, (_, j) => { const x = (j + 0.5) / 20; return { x, y: truth(x) + SIGMA * gauss(r) }; });
  const U = (x) => 2 * x - 1; // 把 x 換成 −1〜1，數字比較穩定

  function feats(x, d) { const u = U(x), f = [1]; for (let k = 1; k <= d; k++) f.push(f[k - 1] * u); return f; }
  function solve(A, b) {
    const m = A.length, M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < m; c++) {
      let p = c; for (let q = c + 1; q < m; q++) if (Math.abs(M[q][c]) > Math.abs(M[p][c])) p = q;
      [M[c], M[p]] = [M[p], M[c]];
      for (let q = 0; q < m; q++) if (q !== c) { const f = M[q][c] / M[c][c]; for (let k = c; k <= m; k++) M[q][k] -= f * M[c][k]; }
    }
    return M.map((row, i) => row[m] / row[i]);
  }
  /** 目標：MSE + 懲罰；L2：λΣw²，L1：λΣ|w|；截距不懲罰 */
  function fit(d, type, lam) {
    const n = TRAIN.length, Phi = TRAIN.map((p) => feats(p.x, d)), y = TRAIN.map((p) => p.y), k = d + 1;
    if (type !== "l1") {
      const l2 = type === "l2" ? lam : 0;
      const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => Phi.reduce((a, row) => a + row[i] * row[j], 0) / n + (i === j && i > 0 ? l2 : 0) + (i === j ? 1e-12 : 0)));
      const b = Array.from({ length: k }, (_, i) => Phi.reduce((a, row, t) => a + row[i] * y[t], 0) / n);
      return solve(A, b);
    }
    // L1：座標下降法（從最小平方的解開始）
    const w = fit(d, "l2", 1e-6), pred = Phi.map((row) => row.reduce((a, v, j) => a + v * w[j], 0));
    const sq = Array.from({ length: k }, (_, j) => Phi.reduce((a, row) => a + row[j] * row[j], 0) / n);
    for (let it = 0; it < 4000; it++) {
      let maxd = 0;
      for (let j = 0; j < k; j++) {
        let rho = 0; for (let t = 0; t < n; t++) rho += Phi[t][j] * (y[t] - pred[t] + w[j] * Phi[t][j]); rho /= n;
        const nw = j === 0 ? rho / sq[j] : Math.sign(rho) * Math.max(0, Math.abs(rho) - lam / 2) / sq[j];
        const dw = nw - w[j];
        if (dw !== 0) { for (let t = 0; t < n; t++) pred[t] += dw * Phi[t][j]; w[j] = nw; maxd = Math.max(maxd, Math.abs(dw)); }
      }
      if (maxd < 1e-10) break;
    }
    return w;
  }
  const predict = (w, x) => feats(x, w.length - 1).reduce((a, v, j) => a + v * w[j], 0);
  const mse = (w, set) => set.reduce((a, p) => a + (predict(w, p.x) - p.y) ** 2, 0) / set.length;
  const CORE = { TRAIN, VALID, truth, fit, predict, mse };

  Atlas.define("overfit", {
    _core: CORE,
    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      const { t } = ctx;
      const LAMS = [0.000001, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
      let deg = 9, type = "none", li = 4, showTruth = false;

      const degSlider = ui.slider({ id: "of-deg", label: t("degLabel"), min: 1, max: 9, step: 1, value: deg, format: (v) => t("degFmt", { n: v }), onInput: (v) => { deg = v; draw(); } });
      const typeSeg = ui.seg({ label: t("regLabel"), value: type, options: [{ value: "none", label: t("regNone") }, { value: "l2", label: t("regL2") }, { value: "l1", label: t("regL1") }], onChange: (v) => { type = v; draw(); } });
      const lamSlider = ui.slider({ id: "of-lam", label: t("lamLabel"), min: 0, max: LAMS.length - 1, step: 1, value: li, format: (v) => String(LAMS[v]), onInput: (v) => { li = v; draw(); } });
      const truthBtn = ui.btn(t("btnTruthShow"), () => { showTruth = !showTruth; truthBtn.textContent = t(showTruth ? "btnTruthHide" : "btnTruthShow"); draw(); }, { sm: true });
      const presetBox = h("div", { class: "btn-group" });
      [["preSimple", 1, "none"], ["preRight", 3, "none"], ["preMemo", 9, "none"], ["preL2", 9, "l2"], ["preL1", 9, "l1"]].forEach(([lab, d, ty]) =>
        presetBox.append(ui.btn(t(lab), () => { deg = d; type = ty; degSlider.set(d); typeSeg.set(ty); if (ty !== "none") { li = 4; lamSlider.set(4); } draw(); }, { sm: true })));

      const view = h("div", { class: "scroll-x" });
      const status = h("div", { class: "status" });
      const calc = h("div", { class: "calc" });
      const coefWrap = h("div", { class: "scroll-x" });
      const curveWrap = h("div", { class: "scroll-x" });
      const note = h("p", { class: "note" });
      const lamRow = h("div", { class: "bench__row" }, lamSlider.el);

      bench.append(
        h("div", { class: "bench__row" }, degSlider.el, typeSeg.el),
        lamRow,
        h("div", { class: "field" }, h("span", { class: "field__label" }, t("presetLabel")), presetBox),
        view,
        h("div", { class: "bench__row bench__row--between" }, status, truthBtn),
        calc,
        h("div", { class: "bench__row", style: { alignItems: "flex-start" } },
          h("div", { class: "field", style: { flex: "1 1 300px", minWidth: 0 } }, h("span", { class: "field__label" }, t("coefLabel")), coefWrap),
          h("div", { class: "field", style: { flex: "1 1 300px", minWidth: 0 } }, h("span", { class: "field__label" }, t("curveLabel")), curveWrap)),
        note,
      );

      const W = 640, H = 330, M = { l: 44, r: 16, t: 54, b: 34 };
      const Y0 = -2, Y1 = 2;
      const sx = (x) => M.l + x * (W - M.l - M.r);
      const sy = (y) => H - M.b - ((y - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
      const lam = () => (type === "none" ? 0 : LAMS[li]);

      function draw() {
        lamRow.hidden = type === "none";
        const w = fit(deg, type, lam());
        const tr = mse(w, TRAIN), va = mse(w, VALID);

        /* --- 主圖 --- */
        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgAria"), style: "display:block;width:100%;min-width:500px;max-width:760px" });
        svg.append(s("defs", null, s("clipPath", { id: "ofclip" }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        for (let v = -2; v <= 2; v++) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: fmt(v, 0) }));
        }
        for (let x = 0; x <= 1.001; x += 0.25) svg.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: fmt(x, 2) }));
        const g = s("g", { "clip-path": "url(#ofclip)" });
        const curve = (fn) => { let d = ""; for (let i = 0; i <= 300; i++) { const x = i / 300; d += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(fn(x)).toFixed(1); } return d; };
        if (showTruth) g.append(s("path", { d: curve(truth), fill: "none", stroke: "var(--ok)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
        g.append(s("path", { d: curve((x) => predict(w, x)), fill: "none", stroke: "var(--accent)", "stroke-width": 2.8 }));
        VALID.forEach((p) => g.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: 4.5, fill: "var(--panel)", stroke: "var(--hot)", "stroke-width": 2 })));
        TRAIN.forEach((p) => g.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: 6, fill: "var(--ink)", stroke: "var(--panel)", "stroke-width": 2 })));
        svg.append(g);
        // legend（放在圖的上方，不蓋住資料點）
        const lg = (x, y, el, text) => { svg.append(el); svg.append(s("text", { x: x + 20, y: y + 4, "font-size": 12, text })); };
        lg(M.l, 14, s("circle", { cx: M.l + 8, cy: 14, r: 6, fill: "var(--ink)" }), t("lgTrain"));
        lg(M.l + 250, 14, s("circle", { cx: M.l + 258, cy: 14, r: 4.5, fill: "var(--panel)", stroke: "var(--hot)", "stroke-width": 2 }), t("lgValid"));
        lg(M.l, 36, s("line", { x1: M.l, x2: M.l + 16, y1: 36, y2: 36, stroke: "var(--accent)", "stroke-width": 2.8 }), t("lgPred"));
        if (showTruth) lg(M.l + 250, 36, s("line", { x1: M.l + 250, x2: M.l + 266, y1: 36, y2: 36, stroke: "var(--ok)", "stroke-width": 2, "stroke-dasharray": "6 4" }), t("lgTruth"));
        view.replaceChildren(svg);

        /* --- 狀態 --- */
        const set = (k, text) => { status.className = "status status--" + k; status.textContent = text; };
        const nz = w.slice(1).filter((v) => Math.abs(v) > 1e-9).length;
        const ev = { tr: fmt(tr, 3), va: fmt(va, 3) };
        if (va > tr * 3 && va > 0.15) set("bad", t("sOver", ev));
        else if (tr > 0.05) set("warn", t(type !== "none" && deg >= 3 ? "sTooStrong" : "sTooSimple", ev));
        else set("ok", t("sGood", ev));

        /* --- 算式 --- */
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        const pen = type === "l2" ? w.slice(1).reduce((a, v) => a + v * v, 0) : w.slice(1).reduce((a, v) => a + Math.abs(v), 0);
        const model = t("calcModel", { d: deg });
        if (type === "none") calc.innerHTML = row(t("calcModelK"), model) +
          row(t("calcLossK"), t("calcLossNone", { v: `<span class="hl">${fmt(tr, 4)}</span>` })) +
          row(t("calcValidK"), `<span class="hl">${fmt(va, 4)}</span>${t("calcValidNote")}`);
        else calc.innerHTML = row(t("calcModelK"), model) +
          row(t("calcLossK"), `${t("calcLossReg", { pen: type === "l2" ? "(w₁² + w₂² + …)" : "(|w₁| + |w₂| + …)" })}　<span class="k">${t("calcNoPenW0")}</span>`) +
          `<span class="calc-row">= ${fmt(tr, 4)} + ${LAMS[li]} × ${fmt(pen, pen > 100 ? 1 : 3)} = <span class="hl">${fmt(tr + LAMS[li] * pen, 4)}</span></span>` +
          row(t("calcValidK"), `<span class="hl">${fmt(va, 4)}</span>`) +
          (type === "l1" ? row(t("calcNzK"), t("calcNz", { nz, d: deg })) : "");

        drawCoef(w);
        drawCurve();

        const NOTE = { none: "noteNone", l2: "noteL2", l1: "noteL1" };
        note.innerHTML = t(NOTE[type]) + t("noteCommon");
      }

      function drawCoef(w) {
        const n = deg, CW = 360, CH = 200, left = 34, right = 10, mid = 100, half = 72;
        const bw = Math.min(30, (CW - left - right) / n - 6);
        const svg = s("svg", { viewBox: `0 0 ${CW} ${CH}`, role: "img", "aria-label": t("coefAria"), style: "display:block;width:100%;min-width:300px;max-width:520px" });
        const mx = Math.max(...w.slice(1).map(Math.abs), 1e-9);
        // 對數式長度：讓 0.1 和 1000 都看得出來，標籤寫真實數字
        const len = (v) => { const a = Math.abs(v); if (a < 1e-9) return 0; return half * Math.max(0.04, Math.log10(1 + a * 10) / Math.log10(1 + Math.max(mx, 1) * 10)); };
        svg.append(s("line", { x1: left - 4, x2: CW - right, y1: mid, y2: mid, stroke: "var(--line-strong)" }));
        svg.append(s("text", { x: left - 8, y: mid - half + 4, "text-anchor": "end", "font-size": 10, class: "t-muted", text: "+" }));
        svg.append(s("text", { x: left - 8, y: mid + half + 4, "text-anchor": "end", "font-size": 10, class: "t-muted", text: "−" }));
        const step = (CW - left - right) / n;
        for (let k = 1; k <= n; k++) {
          const v = w[k], x = left + step * (k - 0.5), L = len(v), zero = Math.abs(v) < 1e-9;
          if (zero) {
            svg.append(s("rect", { x: x - bw / 2, y: mid - 1.5, width: bw, height: 3, fill: "var(--faint)" }));
            svg.append(s("text", { x, y: mid - 8, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: "0" }));
          } else {
            svg.append(s("rect", { x: x - bw / 2, y: v > 0 ? mid - L : mid, width: bw, height: Math.max(1, L), rx: 2, fill: v > 0 ? "var(--pos)" : "var(--neg)" }));
            const big = Math.abs(v) >= 100, txt = big ? Math.round(v).toString().replace("-", "−") : fmt(v, Math.abs(v) >= 10 ? 0 : 1);
            svg.append(s("text", { x, y: v > 0 ? mid - L - 5 : mid + L + 12, "text-anchor": "middle", "font-size": n > 6 ? 9 : 10, class: "t-mono", text: txt }));
          }
          svg.append(s("text", { x, y: CH - 4, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-muted", text: "w" + k }));
        }
        coefWrap.replaceChildren(svg);
      }

      function drawCurve() {
        const CW = 360, CH = 206, m = { l: 40, r: 12, t: 14, b: 36 };
        const tr = [], va = [];
        for (let d = 1; d <= 9; d++) { const w = fit(d, type, lam()); tr.push(mse(w, TRAIN)); va.push(mse(w, VALID)); }
        const lo = -4, hi = 1; // log10 範圍 0.0001〜10
        const px = (d) => m.l + ((d - 1) / 8) * (CW - m.l - m.r);
        const py = (v) => { const L = Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-12)))); return CH - m.b - ((L - lo) / (hi - lo)) * (CH - m.t - m.b); };
        const svg = s("svg", { viewBox: `0 0 ${CW} ${CH}`, role: "img", "aria-label": t("curveAria"), style: "display:block;width:100%;min-width:300px;max-width:520px" });
        for (let e = lo; e <= hi; e++) {
          svg.append(s("line", { x1: m.l, x2: CW - m.r, y1: py(10 ** e), y2: py(10 ** e), stroke: "var(--line)", "stroke-dasharray": "3 3" }));
          svg.append(s("text", { x: m.l - 4, y: py(10 ** e) + 3, "text-anchor": "end", "font-size": 9.5, class: "t-mono t-muted", text: e >= 0 ? String(10 ** e) : (10 ** e).toFixed(-e) }));
        }
        for (let d = 1; d <= 9; d++) svg.append(s("text", { x: px(d), y: CH - m.b + 14, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: d }));
        svg.append(s("text", { x: CW - m.r, y: CH - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("axisDeg") }));
        svg.append(s("rect", { x: px(deg) - 10, y: m.t, width: 20, height: CH - m.t - m.b, fill: "var(--hot-soft)" }));
        const line = (arr, col, dash) => { let d = ""; arr.forEach((v, i) => { d += (i ? "L" : "M") + px(i + 1).toFixed(1) + "," + py(v).toFixed(1); }); svg.append(s("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-dasharray": dash || null })); arr.forEach((v, i) => svg.append(s("circle", { cx: px(i + 1), cy: py(v), r: 3, fill: col }))); };
        line(tr, "var(--ink)");
        line(va, "var(--hot)");
        const best = va.indexOf(Math.min(...va)) + 1;
        svg.append(s("text", { x: best >= 8 ? px(best) + 6 : best <= 1 ? px(best) - 6 : px(best), y: py(va[best - 1]) - 9, "text-anchor": best >= 8 ? "end" : best <= 1 ? "start" : "middle", "font-size": 10, fill: "var(--hot)", "font-weight": 700, text: t("bestValid") }));
        svg.append(s("line", { x1: m.l + 4, x2: m.l + 20, y1: m.t + 8, y2: m.t + 8, stroke: "var(--ink)", "stroke-width": 2 }));
        svg.append(s("text", { x: m.l + 24, y: m.t + 12, "font-size": 10.5, text: t("lgTrainErr") }));
        svg.append(s("line", { x1: m.l + 90, x2: m.l + 106, y1: m.t + 8, y2: m.t + 8, stroke: "var(--hot)", "stroke-width": 2 }));
        svg.append(s("text", { x: m.l + 110, y: m.t + 12, "font-size": 10.5, text: t("lgValidErr") }));
        curveWrap.replaceChildren(svg);
      }

      draw();
      ctx.onTheme(draw);
      return { destroy() {} };
    },
  });
})();
