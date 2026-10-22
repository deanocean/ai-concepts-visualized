/* 激活函數（大綱 §12）
   A. 形狀與斜率：Sigmoid／tanh／ReLU／Leaky ReLU 的函數與導數，拖動輸入看輸出與斜率
   B. 為什麼要非線性：用 n 個 ReLU 神經元的折線（最小平方法求最佳輸出權重）逼近曲線；去掉激活函數只剩直線
   C. 輸出層：softmax 把分數變成總和為 1 的機率
   文字在 locales/zh/act.js、locales/ja/act.js。 */
Atlas.define("act", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const sigm = (x) => 1 / (1 + Math.exp(-x));
    const FN = {
      sigmoid: { label: t("fnSigmoid"), f: sigm, d: (x) => sigm(x) * (1 - sigm(x)), range: "0〜1", maxd: t("maxdFmt", { v: "0.25", at: "x = 0" }) },
      tanh: { label: "tanh", f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2, range: "−1〜1", maxd: t("maxdFmt", { v: "1", at: "x = 0" }) },
      relu: { label: "ReLU", f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), range: "0〜∞", maxd: t("maxdFmt", { v: "1", at: "x > 0" }) },
      leaky: { label: "Leaky ReLU", f: (x) => (x > 0 ? x : 0.01 * x), d: (x) => (x > 0 ? 1 : 0.01), range: "−∞〜∞", maxd: t("maxdFmt", { v: "1", at: "x > 0" }) },
    };
    let mode = "shape", fk = "sigmoid", xin = 2;

    const modeSeg = ui.seg({
      label: t("modeLabel"), value: mode,
      options: [{ value: "shape", label: t("modeShape") }, { value: "why", label: t("modeWhy") }, { value: "soft", label: t("modeSoft") }],
      onChange: (v) => { mode = v; show(); },
    });
    const top = h("div", { class: "bench__row" }, modeSeg.el);
    const body = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } });
    bench.append(top, body);
    const setStatus = (el, k, text) => { el.className = "status status--" + k; el.textContent = text; };

    /* ===================== A. 形狀與斜率 ===================== */
    const A = (() => {
      const fnSeg = ui.seg({ label: t("fnLabel"), value: fk, options: Object.keys(FN).map((k) => ({ value: k, label: FN[k].label })), onChange: (v) => { fk = v; draw(); } });
      const xSlider = ui.slider({ id: "act-x", label: t("xLabel"), min: -5, max: 5, step: 0.1, value: xin, format: (v) => fmt(v, 1), onInput: (v) => { xin = v; draw(); } });
      const view = h("div", { class: "scroll-x" });
      const status = h("div", { class: "status" });
      const calc = h("div", { class: "calc" });
      const tblWrap = h("div", { class: "scroll-x" });
      const note = h("p", { class: "note" });
      const el = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, fnSeg.el, xSlider.el), view, status, calc, tblWrap, note);

      const W = 640, H = 300, M = { l: 44, r: 16, t: 16, b: 34 };
      const X0 = -5, X1 = 5, Y0 = -1.2, Y1 = 2.2;
      const sx = (x) => M.l + ((x - X0) / (X1 - X0)) * (W - M.l - M.r);
      const sy = (y) => H - M.b - ((y - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgShapeAria"), style: "display:block;width:100%;min-width:500px;max-width:760px;cursor:crosshair" });
      svg.addEventListener("click", (e) => {
        const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        xin = Math.max(X0, Math.min(X1, Math.round((X0 + ((p.x - M.l) / (W - M.l - M.r)) * (X1 - X0)) * 10) / 10));
        xSlider.set(xin); draw();
      });

      function path(fn) {
        let d = "", prev = null;
        for (let i = 0; i <= 400; i++) {
          const x = X0 + (i / 400) * (X1 - X0), y = fn(x);
          // ReLU 類導數在 0 處不連續：斷開畫
          if (prev != null && Math.abs(y - prev) > 0.5) d += "M" + sx(x).toFixed(1) + "," + sy(y).toFixed(1);
          else d += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(y).toFixed(1);
          prev = y;
        }
        return d;
      }

      function draw() {
        const F = FN[fk], y = F.f(xin), d = F.d(xin);
        svg.replaceChildren();
        svg.append(s("defs", null, s("clipPath", { id: "actclip" }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        for (let v = -1; v <= 2; v += 0.5) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: fmt(v, 1) }));
        }
        for (let x = X0; x <= X1; x++) svg.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: fmt(x, 0) }));
        svg.append(s("line", { x1: sx(0), x2: sx(0), y1: M.t, y2: H - M.b, stroke: "var(--line-strong)" }));
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 12, class: "t-muted", text: t("axisX") }));
        const g = s("g", { "clip-path": "url(#actclip)" });
        g.append(s("path", { d: path(F.d), fill: "none", stroke: "var(--hot)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
        g.append(s("path", { d: path(F.f), fill: "none", stroke: "var(--accent)", "stroke-width": 3 }));
        // tangent at x
        const span = 1.2;
        g.append(s("line", { x1: sx(xin - span), y1: sy(y - d * span), x2: sx(xin + span), y2: sy(y + d * span), stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "3 3" }));
        g.append(s("line", { x1: sx(xin), x2: sx(xin), y1: sy(Y0), y2: sy(Y1), stroke: "var(--faint)", "stroke-dasharray": "2 4" }));
        g.append(s("circle", { cx: sx(xin), cy: sy(d), r: 5.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 }));
        g.append(s("circle", { cx: sx(xin), cy: sy(y), r: 7.5, fill: "var(--accent)", stroke: "var(--panel)", "stroke-width": 2.5 }));
        svg.append(g);
        // legend
        svg.append(s("rect", { x: M.l + 4, y: M.t + 1, width: 262, height: 44, rx: 6, fill: "var(--panel)", "fill-opacity": 0.92 }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 12, y2: M.t + 12, stroke: "var(--accent)", "stroke-width": 3 }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 16, "font-size": 12, text: t("legendF") }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 32, y2: M.t + 32, stroke: "var(--hot)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 36, "font-size": 12, text: t("legendD") }));
        view.replaceChildren(svg);

        // status
        const xv = fmt(xin, 1);
        if (d === 0) setStatus(status, "bad", t("sZero", { x: xv }));
        else if (d < 0.05) setStatus(status, "warn", t("sFlat", { x: xv, d: fmt(d, d < 0.001 ? 4 : 3) }));
        else if (d >= 0.999) setStatus(status, "ok", t("sOne", { x: xv }));
        else setStatus(status, "info", t("sInfo", { x: xv, y: fmt(y, 3), d: fmt(d, 3) }));

        // calc
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        const xs = fmtP(xin, 1);
        if (fk === "sigmoid") calc.innerHTML =
          row("f(x) = 1 ÷ (1 + e<sup>−x</sup>)", `= 1 ÷ (1 + e<sup>${fmt(-xin, 1)}</sup>) = 1 ÷ (1 + ${fmt(Math.exp(-xin), 3)}) = <span class="hl">${fmt(y, 3)}</span>`) +
          row("f′(x) = f(x) × (1 − f(x))", `= ${fmt(y, 3)} × ${fmt(1 - y, 3)} = <span class="hl">${fmt(d, 4)}</span>`);
        else if (fk === "tanh") calc.innerHTML =
          row("f(x) = tanh(x) = (e<sup>x</sup> − e<sup>−x</sup>) ÷ (e<sup>x</sup> + e<sup>−x</sup>)", `= tanh(${xs}) = <span class="hl">${fmt(y, 3)}</span>`) +
          row("f′(x) = 1 − f(x)²", `= 1 − ${fmtP(y, 3)}² = <span class="hl">${fmt(d, 4)}</span>`);
        else if (fk === "relu") calc.innerHTML =
          row("f(x) = max(0, x)", `= max(0, ${xs}) = <span class="hl">${fmt(y, 2)}</span>`) +
          row(t("calcReluD"), `= <span class="hl">${d}</span>`);
        else calc.innerHTML =
          row(t("calcLeakyF"), `= <span class="hl">${fmt(y, xin > 0 ? 2 : 3)}</span>`) +
          row(t("calcLeakyD"), `= <span class="hl">${fmt(d, 2)}</span>`);

        // comparison table
        const rows = Object.keys(FN).map((k) => {
          const G = FN[k];
          return h("tr", { class: k === fk ? "is-sel" : null },
            h("td", { class: "l", style: { fontFamily: "var(--f-body)" } }, G.label),
            h("td", null, G.range), h("td", null, G.maxd),
            h("td", null, fmt(G.d(xin), G.d(xin) < 0.01 && G.d(xin) > 0 ? 4 : 3)),
            h("td", null, fmt(G.d(-4), G.d(-4) < 0.01 && G.d(-4) > 0 ? 4 : 3)));
        });
        tblWrap.replaceChildren(h("table", { class: "tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, t("thFn")), h("th", null, t("thRange")), h("th", null, t("thMaxd")), h("th", null, t("thSlopeAt", { x: fmt(xin, 1) })), h("th", null, t("thSlopeAt", { x: "−4" })))),
          h("tbody", null, rows)));

        const NOTE = { sigmoid: "noteSigmoid", tanh: "noteTanh", relu: "noteRelu", leaky: "noteLeaky" };
        note.innerHTML = t(NOTE[fk]) + t("noteDrag");
      }
      return { el, show: draw };
    })();

    /* ===================== B. 為什麼要非線性 ===================== */
    const B = (() => {
      let useAct = "relu", n = 4;
      const actSeg = ui.seg({ label: t("hidActLabel"), value: useAct, options: [{ value: "none", label: t("hidActNone") }, { value: "relu", label: "ReLU" }], onChange: (v) => { useAct = v; draw(); } });
      const nSlider = ui.slider({ id: "act-n", label: t("nLabel"), min: 1, max: 8, step: 1, value: n, format: (v) => t("nFmt", { n: v }), onInput: (v) => { n = v; draw(); } });
      const view = h("div", { class: "scroll-x" });
      const status = h("div", { class: "status" });
      const calc = h("div", { class: "calc" });
      const note = h("p", { class: "note" });
      const el = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, actSeg.el, nSlider.el), view, status, calc, note);

      const X0 = -3, X1 = 3, N = 61;
      const xs = Array.from({ length: N }, (_, i) => X0 + (i / (N - 1)) * (X1 - X0));
      const target = (x) => Math.sin(x);
      const ys = xs.map(target);
      const relu = (v) => Math.max(0, v);

      function solve(A, b) { // Gaussian elimination with partial pivoting
        const m = A.length, M = A.map((r, i) => [...r, b[i]]);
        for (let c = 0; c < m; c++) {
          let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
          [M[c], M[p]] = [M[p], M[c]];
          for (let r = 0; r < m; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k]; }
        }
        return M.map((r, i) => r[m] / r[i]);
      }
      function lstsq(cols) { // cols: array of feature functions → weights minimizing squared error (tiny ridge for stability)
        const k = cols.length, Phi = xs.map((x) => cols.map((f) => f(x)));
        const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => Phi.reduce((acc, r) => acc + r[i] * r[j], 0) + (i === j ? 1e-9 : 0)));
        const b = Array.from({ length: k }, (_, i) => Phi.reduce((acc, r, j) => acc + r[i] * ys[j], 0));
        return solve(A, b);
      }

      const W = 640, H = 280, M = { l: 44, r: 16, t: 16, b: 34 };
      const Y0 = -1.6, Y1 = 1.6;
      const sx = (x) => M.l + ((x - X0) / (X1 - X0)) * (W - M.l - M.r);
      const sy = (y) => H - M.b - ((y - Y0) / (Y1 - Y0)) * (H - M.t - M.b);

      function draw() {
        const bs = Array.from({ length: n }, (_, i) => X0 + ((i + 0.5) / n) * (X1 - X0));
        let fit, a, c0, slope, icpt;
        if (useAct === "relu") {
          const w = lstsq([() => 1, ...bs.map((b) => (x) => relu(x - b))]);
          c0 = w[0]; a = w.slice(1);
          fit = (x) => c0 + a.reduce((acc, ai, i) => acc + ai * relu(x - bs[i]), 0);
        } else {
          // 沒有活化函數：a₁(x − b₁) + … + aₙ(x − bₙ) + c 整理後只剩「斜率 × x + 截距」，所以直接求最佳直線
          const w = lstsq([() => 1, (x) => x]);
          icpt = w[0]; slope = w[1];
          fit = (x) => icpt + slope * x;
        }
        const mse = xs.reduce((acc, x, i) => acc + (fit(x) - ys[i]) ** 2, 0) / N;

        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgWhyAria"), style: "display:block;width:100%;min-width:500px;max-width:760px" });
        svg.append(s("defs", null, s("clipPath", { id: "actclip2" }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        for (let v = -1.5; v <= 1.5; v += 0.5) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: fmt(v, 1) }));
        }
        for (let x = X0; x <= X1; x++) svg.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: fmt(x, 0) }));
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 12, class: "t-muted", text: t("axisX") }));
        const g = s("g", { "clip-path": "url(#actclip2)" });
        let d = ""; xs.forEach((x, i) => { d += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(ys[i]).toFixed(1); });
        g.append(s("path", { d, fill: "none", stroke: "var(--faint)", "stroke-width": 7, "stroke-linecap": "round", opacity: 0.5 }));
        let d2 = ""; for (let i = 0; i <= 300; i++) { const x = X0 + (i / 300) * (X1 - X0); d2 += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(fit(x)).toFixed(1); }
        g.append(s("path", { d: d2, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5 }));
        if (useAct === "relu") bs.forEach((b) => g.append(s("circle", { cx: sx(b), cy: sy(fit(b)), r: 5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 })));
        svg.append(g);
        svg.append(s("rect", { x: M.l + 4, y: M.t + 1, width: 200, height: useAct === "relu" ? 64 : 44, rx: 6, fill: "var(--panel)", "fill-opacity": 0.92 }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 12, y2: M.t + 12, stroke: "var(--faint)", "stroke-width": 7, opacity: 0.5, "stroke-linecap": "round" }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 16, "font-size": 12, text: t("legendTarget") }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 32, y2: M.t + 32, stroke: "var(--accent)", "stroke-width": 2.5 }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 36, "font-size": 12, text: t("legendOut") }));
        if (useAct === "relu") {
          svg.append(s("circle", { cx: M.l + 22, cy: M.t + 52, r: 5, fill: "var(--hot)" }));
          svg.append(s("text", { x: M.l + 40, y: M.t + 56, "font-size": 12, text: t("legendKink") }));
        }
        view.replaceChildren(svg);

        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        if (useAct === "relu") {
          setStatus(status, mse < 0.005 ? "ok" : "info", t("sRelu", { n, mse: fmt(mse, 4) }));
          const terms = a.map((ai, i) => `${fmtP(ai, 2)}·ReLU(x ${bs[i] < 0 ? "+" : "−"} ${fmt(Math.abs(bs[i]), 2)})`);
          calc.innerHTML = row(t("calcReluSum"), "") +
            `<span class="calc-row">= ${fmt(c0, 2)} + ${terms.join(" + ")}</span>` +
            row(t("calcEachK"), t("calcEach"));
        } else {
          setStatus(status, "bad", t("sLinear", { n, mse: fmt(mse, 4) }));
          calc.innerHTML = row(t("calcLinSum"), `= (a₁ + … + aₙ)·x + (c − a₁b₁ − … − aₙbₙ)`) +
            row(t("calcLeftK"), t("calcLeft", { line: `${fmt(slope, 3)}·x ${icpt < 0 ? "−" : "+"} ${fmt(Math.abs(icpt), 3)}` })) +
            row(t("calcSameK"), t("calcSame"));
        }
        note.innerHTML = t("noteWhy");
      }
      return { el, show: draw };
    })();

    /* ===================== C. softmax ===================== */
    const C = (() => {
      const CLS = [t("clsCat"), t("clsDog"), t("clsBird")];
      let z = [2.0, 1.0, -0.5];
      const sliders = CLS.map((c, i) => ui.slider({ id: "act-z" + i, label: t("zLabel", { c, i: i + 1 }), min: -3, max: 5, step: 0.1, value: z[i], format: (v) => fmt(v, 1), onInput: (v) => { z[i] = v; draw(); } }));
      const view = h("div", { class: "scroll-x" });
      const status = h("div", { class: "status" });
      const calc = h("div", { class: "calc" });
      const note = h("p", { class: "note" });
      const el = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, ...sliders.map((x) => x.el)), view, status, calc, note);

      function draw() {
        const e = z.map(Math.exp), S = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / S);
        const W = 640, H = 230, colX = [150, 330, 510], base = 190, top = 40;
        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgSoftAria"), style: "display:block;width:100%;min-width:460px;max-width:760px" });
        svg.append(s("text", { x: 12, y: 20, "font-size": 12, class: "t-muted", text: t("softHead") }));
        svg.append(s("line", { x1: 60, x2: W - 30, y1: base, y2: base, stroke: "var(--line-strong)" }));
        CLS.forEach((c, i) => {
          const x = colX[i], bh = p[i] * (base - top);
          svg.append(s("rect", { x: x - 36, y: base - bh, width: 72, height: Math.max(1, bh), rx: 4, fill: "var(--accent)", "fill-opacity": 0.25 + 0.75 * p[i] }));
          svg.append(s("text", { x, y: base - bh - 8, "text-anchor": "middle", "font-size": 14, "font-weight": 700, class: "t-mono", text: fmt(p[i], 3) }));
          svg.append(s("text", { x, y: base + 18, "text-anchor": "middle", "font-size": 13, "font-weight": 700, text: c }));
          svg.append(s("text", { x, y: base + 34, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: `z=${fmt(z[i], 1)}  e^z=${fmt(e[i], 2)}` }));
        });
        view.replaceChildren(svg);
        const best = p.indexOf(Math.max(...p));
        setStatus(status, "info", t("sBest", { c: CLS[best], p: fmt(p[best] * 100, 1) }));
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        calc.innerHTML = row("pᵢ = e<sup>zᵢ</sup> ÷ (e<sup>z₁</sup> + e<sup>z₂</sup> + e<sup>z₃</sup>)", t("calcDenom", { terms: e.map((v) => fmt(v, 3)).join(" + "), sum: fmt(S, 3) })) +
          CLS.map((c, i) => row(`p(${c})`, `= ${fmt(e[i], 3)} ÷ ${fmt(S, 3)} = <span class="hl">${fmt(p[i], 3)}</span>`)).join("") +
          row(t("calcSumK"), `${p.map((v) => fmt(v, 3)).join(" + ")} ≈ 1 <span class="k">${t("calcSumNote")}</span>`);
        note.innerHTML = t("noteSoft");
      }
      return { el, show: draw };
    })();

    function show() {
      const m = mode === "shape" ? A : mode === "why" ? B : C;
      body.replaceChildren(m.el);
      m.show();
    }
    show();
    ctx.onTheme(show);
    return { destroy() {} };
  },
});
