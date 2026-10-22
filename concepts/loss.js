/* 損失函數（大綱 §13）
   四個模式，全部即時計算：
   ① 均方誤差：5 個點、模型 y = w·x，每個誤差畫成正方形（面積＝誤差²），右邊是 MSE 對 w 的曲線。
   ② 交叉熵：二元分類、正解＝1，拖動 z（Sigmoid 前的分數），比較兩種損失與「對 z 的梯度」。
   ③ KL 散度：4 類的真實分布 P 與模型分布 Q，逐項算 P·ln(P/Q)。
   ④ Contrastive Loss／Triplet Loss：在 2D 嵌入平面拖動點，看損失何時歸零；可照梯度移動一步。
   文字在 locales/zh/loss.js、locales/ja/loss.js。 */
Atlas.define("loss", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    let mode = "mse";
    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode,
      options: [{ value: "mse", label: t("modeMse") }, { value: "ce", label: t("modeCe") }, { value: "kl", label: t("modeKl") }, { value: "metric", label: t("modeMetric") }],
      onChange: (v) => { mode = v; show(); } });
    const box = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } });
    bench.append(h("style", { text: `
      .ls-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; }
      .ls-fig { flex: 1 1 300px; max-width: 470px; min-width: 0; }
      .ls-fig svg { display: block; width: 100%; height: auto; }
      .ls-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
      .ls-q { display: flex; flex-wrap: wrap; gap: 6px 14px; }
      .ls-q > span { display: inline-flex; align-items: center; gap: 4px; font-size: 13.5px; }
      .ls-q .btn { min-width: 34px; }
      .ls-pt { cursor: grab; touch-action: none; }
      .ls-pt:active { cursor: grabbing; }
    ` }));
    bench.append(h("div", { class: "bench__row" }, modeSeg.el), box);
    const note = h("p", { class: "note" });
    let inst = null;
    function show() { box.replaceChildren(); inst = MODES[mode](); box.append(note); }
    function setStatus(el, kind, text) { el.className = "status status--" + kind; el.textContent = text; }
    const axisNum = (v) => (v < 0 ? "−" + -v : String(v));

    /* ===================== ① 均方誤差 ===================== */
    function modeMSE() {
      const P = [[0.5, 0.9], [1.0, 1.0], [1.5, 2.1], [2.0, 1.8], [2.5, 2.9]];
      const W_OPT = P.reduce((a, p) => a + p[0] * p[1], 0) / P.reduce((a, p) => a + p[0] * p[0], 0);
      const mse = (w) => P.reduce((a, p) => a + (p[1] - w * p[0]) ** 2, 0) / P.length;
      let w = 0.6;
      const sl = ui.slider({ id: "ls-w", label: t("mseW"), min: 0, max: 2.5, step: 0.05, value: w, format: (v) => v.toFixed(2), onInput: (v) => { w = v; draw(); } });
      const fit = s("svg", { viewBox: "0 0 340 320", role: "img", "aria-label": t("mseFitAria") });
      const curve = s("svg", { viewBox: "0 0 340 320", role: "img", "aria-label": t("mseCurveAria") });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      box.append(
        h("div", { class: "bench__row" }, sl.el, ui.btn(t("btnBestW"), () => { w = Math.round(W_OPT * 20) / 20; sl.set(w); draw(); }, { sm: true })),
        h("div", { class: "scroll-x" }, h("div", { class: "ls-figs" },
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, t("capSquares")), fit),
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, t("capMseCurve")), curve))),
        status, calc);
      function draw() {
        // 左：長寬同比例，正方形才是真的正方形
        const M = { l: 30, t: 8, b: 26 }, U = 70, X0 = -0.2, Y1 = 4;
        const sx = (x) => M.l + (x - X0) * U, sy = (y) => M.t + (Y1 - y) * U;
        fit.replaceChildren();
        fit.append(s("defs", null, s("clipPath", { id: "lsclip" }, s("rect", { x: M.l, y: M.t, width: 4.3 * U, height: 4 * U }))));
        for (let y = 0; y <= 4; y++) {
          fit.append(s("line", { x1: M.l, x2: M.l + 4.3 * U, y1: sy(y), y2: sy(y), stroke: "var(--line)" }));
          fit.append(s("text", { x: M.l - 6, y: sy(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y }));
        }
        for (let x = 0; x <= 4; x++) fit.append(s("text", { x: sx(x), y: sy(0) + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: x }));
        const g = s("g", { "clip-path": "url(#lsclip)" });
        P.forEach(([x, y]) => {
          const e = y - w * x, side = Math.abs(e) * U;
          g.append(s("rect", { x: sx(x), y: Math.min(sy(y), sy(w * x)), width: side, height: side, fill: "var(--hot-soft)", stroke: "var(--hot)", "stroke-width": 1.2, "fill-opacity": 0.85 }));
        });
        g.append(s("line", { x1: sx(X0), y1: sy(w * X0), x2: sx(4.1), y2: sy(w * 4.1), stroke: "var(--accent)", "stroke-width": 2.5 }));
        fit.append(g);
        P.forEach(([x, y]) => fit.append(s("circle", { cx: sx(x), cy: sy(y), r: 5, fill: "var(--ink)", stroke: "var(--panel)", "stroke-width": 1.5 })));
        // 右：MSE 曲線
        const C = { l: 40, r: 12, t: 10, b: 30 }, CW = 340, CH = 320, YM = 5.5;
        const cx = (v) => C.l + (v / 2.5) * (CW - C.l - C.r), cy = (v) => CH - C.b - (v / YM) * (CH - C.t - C.b);
        curve.replaceChildren();
        for (let y = 0; y <= 5; y++) {
          curve.append(s("line", { x1: C.l, x2: CW - C.r, y1: cy(y), y2: cy(y), stroke: "var(--line)" }));
          curve.append(s("text", { x: C.l - 6, y: cy(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y }));
        }
        for (const v of [0, 0.5, 1, 1.5, 2, 2.5]) curve.append(s("text", { x: cx(v), y: CH - C.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v }));
        curve.append(s("text", { x: CW - C.r, y: CH - 2, "text-anchor": "end", "font-size": 11.5, class: "t-muted", text: "w" }));
        let d = ""; for (let i = 0; i <= 100; i++) { const v = (i / 100) * 2.5; d += (i ? "L" : "M") + cx(v).toFixed(1) + "," + cy(mse(v)).toFixed(1); }
        curve.append(s("path", { d, fill: "none", stroke: "var(--seq-hi)", "stroke-width": 2.5 }));
        curve.append(s("line", { x1: cx(W_OPT), x2: cx(W_OPT), y1: cy(0), y2: cy(5.1), stroke: "var(--ok)", "stroke-dasharray": "4 3" }));
        curve.append(s("text", { x: cx(W_OPT), y: cy(5.1) - 6, "text-anchor": "middle", "font-size": 11.5, class: "t-halo", fill: "var(--ok)", text: t("mseMin", { w: fmt(W_OPT, 3) }) }));
        curve.append(s("circle", { cx: cx(w), cy: cy(mse(w)), r: 7, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2.5 }));
        // 文字
        const es = P.map(([x, y]) => y - w * x), m = mse(w);
        if (Math.abs(w - W_OPT) < 0.03) setStatus(status, "ok", t("sMseMin", { m: fmt(m, 3) }));
        else setStatus(status, "info", t(w < W_OPT ? "sMseRight" : "sMseLeft", { m: fmt(m, 3) }));
        calc.innerHTML =
          `<span class="calc-row"><span class="k">${t("calcErrK")}</span> ${P.map(([x, y], i) => `${fmt(y, 1)} − ${fmt(w, 2)}×${fmt(x, 1)} = ${fmt(es[i], 2)}`).join(t("listSep"))}</span>` +
          `<span class="calc-row"><span class="k">${t("lgMse")}</span> (${es.map((e) => fmtP(e, 2) + "²").join(" + ")}) ÷ 5 = (${es.map((e) => fmt(e * e, 3)).join(" + ")}) ÷ 5 = <span class="hl">${fmt(m, 3)}</span></span>`;
      }
      note.innerHTML = t("noteMse");
      draw();
      return {};
    }

    /* ===================== ② 交叉熵 ===================== */
    function modeCE() {
      let z = -3;
      const sig = (v) => 1 / (1 + Math.exp(-v));
      const sl = ui.slider({ id: "ls-z", label: t("ceZ"), min: -4.5, max: 4.5, step: 0.1, value: z, format: (v) => fmt(v, 1), onInput: (v) => { z = v; draw(); } });
      const lossSvg = s("svg", { viewBox: "0 0 340 280", role: "img", "aria-label": t("ceLossAria") });
      const gradSvg = s("svg", { viewBox: "0 0 340 280", role: "img", "aria-label": t("ceGradAria") });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      const pre = h("div", { class: "btn-group" },
        ...[["preWrong", -4], ["preUnsure", 0], ["preRight", 4]].map(([l, v]) => ui.btn(t(l), () => { z = v; sl.set(v); draw(); }, { sm: true })));
      box.append(
        h("p", { class: "hint", html: t("ceHint") }),
        h("div", { class: "bench__row" }, sl.el, h("div", { class: "field" }, h("span", { class: "field__label" }, t("presetLabel")), pre)),
        h("div", { class: "scroll-x" }, h("div", { class: "ls-figs" },
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, t("capLoss")), lossSvg),
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, t("capGrad")), gradSvg))),
        status, calc);
      const CEf = (p) => -Math.log(p), MSEf = (p) => (1 - p) ** 2;
      const CEg = (p) => Math.abs(p - 1), MSEg = (p) => Math.abs(2 * (p - 1) * p * (1 - p));
      function plot(svg, fns, YM, ticks, p) {
        const W = 340, H = 280, M = { l: 36, r: 12, t: 30, b: 30 };
        const px = (v) => M.l + v * (W - M.l - M.r), py = (v) => H - M.b - (Math.min(v, YM) / YM) * (H - M.t - M.b);
        svg.replaceChildren();
        for (const tk of ticks) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: py(tk), y2: py(tk), stroke: "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: py(tk) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: tk }));
        }
        for (const v of [0, 0.25, 0.5, 0.75, 1]) svg.append(s("text", { x: px(v), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v }));
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 11.5, class: "t-muted", text: t("axisP") }));
        svg.append(s("line", { x1: px(p), x2: px(p), y1: M.t, y2: H - M.b, stroke: "var(--hot)", "stroke-dasharray": "4 3" }));
        fns.forEach(([fn, col, name], k) => {
          let d = ""; for (let i = 1; i <= 199; i++) { const v = i / 200; d += (i > 1 ? "L" : "M") + px(v).toFixed(1) + "," + py(fn(v)).toFixed(1); }
          svg.append(s("path", { d, fill: "none", stroke: col, "stroke-width": 2.5 }));
          svg.append(s("circle", { cx: px(p), cy: py(fn(p)), r: 6, fill: col, stroke: "var(--panel)", "stroke-width": 2 }));
          svg.append(s("line", { x1: M.l + 6 + k * 150, x2: M.l + 24 + k * 150, y1: 12, y2: 12, stroke: col, "stroke-width": 3 }));
          svg.append(s("text", { x: M.l + 30 + k * 150, y: 16, "font-size": 11.5, text: name }));
        });
      }
      function draw() {
        const p = sig(z);
        plot(lossSvg, [[CEf, "var(--neg)", t("lgCe")], [MSEf, "var(--pos)", t("lgMse")]], 4.6, [0, 1, 2, 3, 4], p);
        plot(gradSvg, [[CEg, "var(--neg)", t("lgCe")], [MSEg, "var(--pos)", t("lgMse")]], 1.05, [0, 0.25, 0.5, 0.75, 1], p);
        const gc = CEg(p), gm = MSEg(p);
        const pv = fmt(p, 3);
        if (p < 0.2) setStatus(status, "warn", t("sCeWrong", { p: pv, gc: fmt(gc, 3), gm: fmt(gm, 3) }));
        else if (p > 0.9) setStatus(status, "ok", t("sCeRight", { p: pv }));
        else setStatus(status, "info", t("sCeInfo", { p: pv, gc: fmt(gc, 3), gm: fmt(gm, 3) }));
        calc.innerHTML =
          `<span class="calc-row"><span class="k">${t("calcProbK")}</span> p = ${t("fnSigmoid")}(z) = 1 ÷ (1 + e<sup>−z</sup>) = 1 ÷ (1 + e<sup>${fmt(-z, 1)}</sup>) = <b>${fmt(p, 4)}</b></span>` +
          `<span class="calc-row"><span class="k">${t("lgCe")}</span> −t·ln p = −ln ${fmt(p, 4)} = <span class="n">${fmt(CEf(p), 3)}</span>　<span class="k">${t("calcGradZK")}</span> p − t = ${fmt(p - 1, 3)}</span>` +
          `<span class="calc-row"><span class="k">${t("lgMse")}</span> (t − p)² = ${fmtP(1 - p, 4)}² = <span class="p">${fmt(MSEf(p), 3)}</span>　<span class="k">${t("calcGradZK")}</span> 2(p − t) × p(1 − p) = 2 × ${fmtP(p - 1, 3)} × ${fmt(p * (1 - p), 4)} = ${fmt(2 * (p - 1) * p * (1 - p), 4)}</span>`;
      }
      note.innerHTML = t("noteCe");
      draw();
      return {};
    }

    /* ===================== ③ KL 散度 ===================== */
    function modeKL() {
      const CAT = [t("catSun"), t("catCloud"), t("catRain"), t("catSnow")];
      const P = [0.5, 0.3, 0.15, 0.05];
      let Q = [0.25, 0.25, 0.25, 0.25];
      const norm = (a) => { const tot = a.reduce((x, y) => x + y, 0); return a.map((v) => v / tot); };
      const nudge = (i, f) => { const q = Q.slice(); q[i] = Math.min(0.97, Math.max(0.005, q[i] * f)); Q = norm(q); draw(); };
      const qBox = h("div", { class: "ls-q" }, ...CAT.map((c, i) => h("span", null, `Q(${c})`,
        ui.btn("−", () => nudge(i, 0.75), { sm: true, aria: t("ariaDown", { c }) }), ui.btn("+", () => nudge(i, 1.3333), { sm: true, aria: t("ariaUp", { c }) }))));
      const pre = h("div", { class: "btn-group" }, ...[
        ["preSame", P], ["preUniform", [0.25, 0.25, 0.25, 0.25]], ["preReversed", [0.05, 0.15, 0.3, 0.5]], ["preNoSnow", [0.52, 0.31, 0.165, 0.005]],
      ].map(([l, q]) => ui.btn(t(l), () => { Q = q.slice(); draw(); }, { sm: true })));
      const svg = s("svg", { viewBox: "0 0 560 250", role: "img", "aria-label": t("klAria"), style: "display:block;width:100%;min-width:420px;max-width:640px" });
      const status = h("div", { class: "status" }), tbl = h("table", { class: "tbl" }), calc = h("div", { class: "calc" });
      box.append(
        h("p", { class: "hint", html: t("klHint") }),
        h("div", { class: "bench__row" }, h("div", { class: "field" }, h("span", { class: "field__label" }, t("presetQLabel")), pre)),
        h("div", { class: "bench__row" }, qBox),
        h("div", { class: "scroll-x" }, svg), status, h("div", { class: "scroll-x" }, tbl), calc);
      const ln = Math.log;
      function draw() {
        const W = 560, H = 250, M = { l: 40, r: 10, t: 30, b: 28 }, YM = 0.6;
        const gw = (W - M.l - M.r) / 4, py = (v) => H - M.b - (v / YM) * (H - M.t - M.b);
        svg.replaceChildren();
        for (const v of [0, 0.2, 0.4, 0.6]) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: py(v), y2: py(v), stroke: "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: py(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: v }));
        }
        CAT.forEach((c, i) => {
          const x0 = M.l + i * gw + gw / 2;
          [[P[i], "var(--pos)", -30], [Q[i], "var(--hot)", 2]].forEach(([v, col, dx]) => {
            svg.append(s("rect", { x: x0 + dx, y: py(Math.min(v, YM)), width: 28, height: py(0) - py(Math.min(v, YM)), rx: 3, fill: col }));
            svg.append(s("text", { x: x0 + dx + 14, y: py(Math.min(v, YM)) - 5, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-halo", text: fmt(v, 3) }));
          });
          svg.append(s("text", { x: x0, y: H - M.b + 18, "text-anchor": "middle", "font-size": 13, text: c }));
        });
        svg.append(s("rect", { x: M.l + 6, y: 6, width: 14, height: 12, rx: 2, fill: "var(--pos)" }));
        svg.append(s("text", { x: M.l + 26, y: 16, "font-size": 12, text: t("lgP") }));
        svg.append(s("rect", { x: M.l + 110, y: 6, width: 14, height: 12, rx: 2, fill: "var(--hot)" }));
        svg.append(s("text", { x: M.l + 130, y: 16, "font-size": 12, text: t("lgQ") }));
        const terms = P.map((p, i) => p * ln(p / Q[i])), kl = terms.reduce((a, b) => a + b, 0);
        const klr = Q.reduce((a, q, i) => a + q * ln(q / P[i]), 0), H_P = -P.reduce((a, p) => a + p * ln(p), 0);
        tbl.replaceChildren(
          h("thead", null, h("tr", null, ...[t("thCat"), "P", "Q", "ln(P ÷ Q)", "P × ln(P ÷ Q)"].map((th, k) => h("th", { class: k ? null : "l" }, th)))),
          h("tbody", null, ...CAT.map((c, i) => h("tr", null, h("td", { class: "l" }, c), h("td", null, fmt(P[i], 3)), h("td", null, fmt(Q[i], 3)), h("td", null, fmt(ln(P[i] / Q[i]), 3)), h("td", null, fmt(terms[i], 4))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l", colspan: 4 }, t("klTotal")), h("td", null, fmt(kl, 4)))));
        if (kl < 1e-6) setStatus(status, "ok", t("sKlZero"));
        else {
          const worst = terms.indexOf(Math.max(...terms));
          setStatus(status, kl > 0.5 ? "bad" : "info", t("sKl", { kl: fmt(kl, 3), c: CAT[worst] }));
        }
        calc.innerHTML =
          `<span class="calc-row"><span class="k">KL(P‖Q)</span> Σ P·ln(P ÷ Q) = ${terms.map((v) => fmtP(v, 4)).join(" + ")} = <span class="hl">${fmt(kl, 4)}</span></span>` +
          `<span class="calc-row"><span class="k">${t("calcKlRevK")}</span> Σ Q·ln(Q ÷ P) = ${fmt(klr, 4)}　<span class="k">${t("calcKlRevNote")}</span></span>` +
          `<span class="calc-row"><span class="k">${t("lgCe")}</span> −Σ P·ln Q = ${fmt(H_P + kl, 4)} = <span class="k">${t("calcEntK")}</span> ${fmt(H_P, 4)} + <span class="k">KL</span> ${fmt(kl, 4)}</span>`;
      }
      note.innerHTML = t("noteKl");
      draw();
      return {};
    }

    /* ===================== ④ Contrastive／Triplet ===================== */
    function modeMetric() {
      let kind = "triplet", margin = 0.5, same = false;
      const INIT = { A: [-1, 0], P: [0.6, 0.8], N: [0.2, -0.5], B: [0, 0.3] };
      let pts = JSON.parse(JSON.stringify(INIT));
      const kindSeg = ui.seg({ label: t("kindLabel"), value: kind, options: [{ value: "triplet", label: t("kindTriplet") }, { value: "contrastive", label: t("kindContrastive") }],
        onChange: (v) => { kind = v; margin = v === "triplet" ? 0.5 : 1.5; mSl.set(margin); sync(); draw(); } });
      const mSl = ui.slider({ id: "ls-m", label: t("marginLabel"), min: 0, max: 2.5, step: 0.1, value: margin, format: (v) => v.toFixed(1), onInput: (v) => { margin = v; draw(); } });
      const pairSeg = ui.seg({ label: t("pairLabel"), value: same, options: [{ value: true, label: t("pairSame") }, { value: false, label: t("pairDiff") }], onChange: (v) => { same = v; draw(); } });
      const svg = s("svg", { viewBox: "0 0 600 400", role: "img", "aria-label": t("metricAria"), style: "display:block;width:100%;min-width:400px;max-width:640px;user-select:none" });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      const stepBtn = ui.btn(t("btnGradStep"), () => { gradStep(); }, { primary: true });
      const resetBtn = ui.btn(t("btnReset"), () => { pts = JSON.parse(JSON.stringify(INIT)); draw(); });
      box.append(
        h("p", { class: "hint", html: t("metricHint") }),
        h("div", { class: "bench__row" }, kindSeg.el, pairSeg.el),
        h("div", { class: "bench__row" }, mSl.el),
        h("div", { class: "scroll-x" }, svg), status,
        h("div", { class: "bench__row" }, stepBtn, resetBtn), calc);
      function sync() { pairSeg.el.hidden = kind !== "contrastive"; }
      const U = 100, OX = 300, OY = 200;
      const X = (x) => OX + x * U, Y = (y) => OY - y * U;
      const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
      function lossInfo() {
        const A = pts.A;
        if (kind === "triplet") {
          const dp = dist(A, pts.P), dn = dist(A, pts.N), L = Math.max(0, dp - dn + margin);
          return { dp, dn, L };
        }
        const d = dist(A, pts.B), L = same ? 0.5 * d * d : 0.5 * Math.max(0, margin - d) ** 2;
        return { d, L };
      }
      /** 梯度（只移動 P、N 或 B，錨點固定，方便觀察） */
      function grads() {
        const A = pts.A, out = {};
        const unit = (a, b) => { const d = dist(a, b) || 1; return [(a[0] - b[0]) / d, (a[1] - b[1]) / d]; };
        const info = lossInfo();
        if (kind === "triplet") {
          if (info.L > 0) { out.P = unit(pts.P, A); out.N = unit(pts.N, A).map((v) => -v); }
        } else if (same) out.B = [pts.B[0] - A[0], pts.B[1] - A[1]];
        else if (info.L > 0) { const u = unit(pts.B, A); out.B = u.map((v) => -(margin - info.d) * v); }
        return out;
      }
      function gradStep() { const g = grads(); for (const k in g) pts[k] = [pts[k][0] - 0.3 * g[k][0], pts[k][1] - 0.3 * g[k][1]].map((v) => Math.round(v * 1000) / 1000); draw(); }
      function arrow(x1, y1, x2, y2, col) {
        const L = Math.hypot(x2 - x1, y2 - y1); if (L < 4) return;
        const ux = (x2 - x1) / L, uy = (y2 - y1) / L;
        svg.append(s("line", { x1, y1, x2: x2 - ux * 7, y2: y2 - uy * 7, stroke: col, "stroke-width": 2.5 }));
        svg.append(s("path", { d: `M${x2},${y2} L${x2 - ux * 10 - uy * 5.5},${y2 - uy * 10 + ux * 5.5} L${x2 - ux * 10 + uy * 5.5},${y2 - uy * 10 - ux * 5.5}Z`, fill: col }));
      }
      function draw() {
        sync();
        svg.replaceChildren();
        svg.append(s("rect", { x: 0, y: 0, width: 600, height: 400, fill: "var(--panel-2)", rx: 8 }));
        for (let x = -3; x <= 3; x++) svg.append(s("line", { x1: X(x), x2: X(x), y1: 0, y2: 400, stroke: "var(--line)" }));
        for (let y = -2; y <= 2; y++) svg.append(s("line", { x1: 0, x2: 600, y1: Y(y), y2: Y(y), stroke: "var(--line)" }));
        const A = pts.A, info = lossInfo();
        const names = kind === "triplet" ? ["A", "P", "N"] : ["A", "B"];
        if (kind === "triplet") {
          svg.append(s("circle", { cx: X(A[0]), cy: Y(A[1]), r: info.dp * U, fill: "none", stroke: "var(--ok)", "stroke-width": 1.5, "stroke-dasharray": "4 4" }));
          svg.append(s("circle", { cx: X(A[0]), cy: Y(A[1]), r: (info.dp + margin) * U, fill: "none", stroke: "var(--neg)", "stroke-width": 2, "stroke-dasharray": "8 5" }));
          svg.append(s("line", { x1: X(A[0]), y1: Y(A[1]), x2: X(pts.P[0]), y2: Y(pts.P[1]), stroke: "var(--ok)", "stroke-width": 2 }));
          svg.append(s("line", { x1: X(A[0]), y1: Y(A[1]), x2: X(pts.N[0]), y2: Y(pts.N[1]), stroke: "var(--neg)", "stroke-width": 2 }));
        } else {
          if (!same) {
            svg.append(s("circle", { cx: X(A[0]), cy: Y(A[1]), r: margin * U, fill: "none", stroke: "var(--neg)", "stroke-width": 2, "stroke-dasharray": "8 5" }));
          }
          svg.append(s("line", { x1: X(A[0]), y1: Y(A[1]), x2: X(pts.B[0]), y2: Y(pts.B[1]), stroke: same ? "var(--ok)" : "var(--neg)", "stroke-width": 2 }));
        }
        // 圖例（放在左下角，圓圈可能大到超出畫面）
        const lg = kind === "triplet"
          ? [["var(--ok)", "4 4", t("lgGreen")], ["var(--neg)", "8 5", t("lgRedTriplet")]]
          : same ? [] : [["var(--neg)", "8 5", t("lgRedPair")]];
        if (lg.length) {
          const gy = 400 - 12 - lg.length * 20;
          svg.append(s("rect", { x: 8, y: gy - 6, width: 268, height: lg.length * 20 + 8, rx: 6, fill: "var(--panel)", stroke: "var(--line)" }));
          lg.forEach(([col, dash, label], i) => {
            svg.append(s("line", { x1: 16, x2: 40, y1: gy + 6 + i * 20, y2: gy + 6 + i * 20, stroke: col, "stroke-width": 2, "stroke-dasharray": dash }));
            svg.append(s("text", { x: 46, y: gy + 10 + i * 20, "font-size": 12, text: label }));
          });
        }
        const g = grads();
        for (const k in g) arrow(X(pts[k][0]), Y(pts[k][1]), X(pts[k][0] - 0.6 * g[k][0] / Math.max(1, Math.hypot(...g[k]))), Y(pts[k][1] - 0.6 * g[k][1] / Math.max(1, Math.hypot(...g[k]))), "var(--hot)");
        const LAB = { A: t("labA"), P: t("labP"), N: t("labN"), B: t(same ? "labBSame" : "labBDiff") };
        const COL = { A: "var(--ink)", P: "var(--ok)", N: "var(--neg)", B: same ? "var(--ok)" : "var(--neg)" };
        for (const k of names) {
          const c = s("circle", { cx: X(pts[k][0]), cy: Y(pts[k][1]), r: 11, fill: COL[k], stroke: "var(--panel)", "stroke-width": 2.5, class: "ls-pt", "data-k": k, tabindex: 0, role: "button", "aria-label": t("dragAria", { name: LAB[k] }) });
          svg.append(c);
          // 名字放在和箭頭垂直的方向，避開箭頭與連線；沒有箭頭就放下方
          let lx = X(pts[k][0]), ly = Y(pts[k][1]) + 28;
          if (g[k]) {
            const gl = Math.hypot(...g[k]) || 1, ux = -g[k][0] / gl, uy = g[k][1] / gl; // 螢幕座標的箭頭方向
            let nx = -uy, ny = ux; if (ny < 0) { nx = -nx; ny = -ny; }
            const kk = 20 + 32 * Math.abs(nx);
            lx = X(pts[k][0]) + nx * kk; ly = Y(pts[k][1]) + ny * kk + 5;
          }
          svg.append(s("text", { x: lx, y: ly, "text-anchor": "middle", "font-size": 12.5, "font-weight": 700, class: "t-halo", fill: COL[k], text: LAB[k], style: "pointer-events:none" }));
        }
        // 文字
        if (kind === "triplet") {
          if (info.L === 0) setStatus(status, "ok", t("sTripZero", { dn: fmt(info.dn, 2), dp: fmt(info.dp, 2), m: fmt(margin, 1) }));
          else setStatus(status, info.dn < info.dp ? "bad" : "warn", t(info.dn < info.dp ? "sTripBad" : "sTripWarn", { L: fmt(info.L, 3) }));
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcDistK")}</span> d(A,P) = ${fmt(info.dp, 3)}　d(A,N) = ${fmt(info.dn, 3)}</span>` +
            `<span class="calc-row"><span class="k">Triplet Loss</span> max(0, d(A,P) − d(A,N) + m) = max(0, ${fmt(info.dp, 3)} − ${fmt(info.dn, 3)} + ${fmt(margin, 1)}) = <span class="hl">${fmt(info.L, 3)}</span></span>` +
            (info.L > 0 ? `<span class="calc-row"><span class="k">${t("calcTripDirK")}</span> ${t("calcTripDir")}</span>` : "");
        } else {
          const dv = { d: fmt(info.d, 2), m: fmt(margin, 1), L: fmt(info.L, 3) };
          if (info.L === 0) setStatus(status, "ok", t(same ? "sPairSameZero" : "sPairDiffZero", dv));
          else setStatus(status, "warn", t(same ? "sPairSame" : "sPairDiff", dv));
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcDistK")}</span> d(A,B) = ${fmt(info.d, 3)}</span>` +
            (same ? `<span class="calc-row"><span class="k">${t("pairSame")}</span> ½·d² = ½ × ${fmt(info.d, 3)}² = <span class="hl">${fmt(info.L, 3)}</span></span>`
                  : `<span class="calc-row"><span class="k">${t("pairDiff")}</span> ½·max(0, m − d)² = ½ × max(0, ${fmt(margin, 1)} − ${fmt(info.d, 3)})² = <span class="hl">${fmt(info.L, 3)}</span></span>`);
        }
      }
      // 拖動
      let drag = null;
      const toPt = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse()); return [Math.max(-2.9, Math.min(2.9, (q.x - OX) / U)), Math.max(-1.9, Math.min(1.9, (OY - q.y) / U))].map((v) => Math.round(v * 20) / 20); };
      svg.addEventListener("pointerdown", (e) => { const k = e.target.getAttribute && e.target.getAttribute("data-k"); if (!k) return; drag = k; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
      svg.addEventListener("pointermove", (e) => { if (!drag) return; pts[drag] = toPt(e); draw(); });
      const end = () => { drag = null; };
      svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
      svg.addEventListener("keydown", (e) => {
        const k = e.target.getAttribute && e.target.getAttribute("data-k"); if (!k) return;
        const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key]; if (!d) return;
        e.preventDefault(); pts[k] = [pts[k][0] + d[0], pts[k][1] + d[1]].map((v) => Math.round(v * 100) / 100); draw();
        const el = svg.querySelector(`[data-k="${k}"]`); el && el.focus();
      });
      note.innerHTML = t("noteMetric");
      draw();
      return {};
    }

    const MODES = { mse: modeMSE, ce: modeCE, kl: modeKL, metric: modeMetric };
    show(); // 圖都用 CSS 變數上色，切換深淺色不需重畫
    return {};
  },
});
