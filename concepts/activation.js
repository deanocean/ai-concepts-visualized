/* 活性化関数（シラバス §12）
   A. 形狀與斜率：Sigmoid／tanh／ReLU／Leaky ReLU 的函數與導數，拖動輸入看輸出與斜率
   B. 為什麼要非線性：用 n 個 ReLU 神經元的折線（最小平方法求最佳輸出權重）逼近曲線；去掉活化函數只剩直線
   C. 輸出層：ソフトマックス 把分數變成總和為 1 的機率 */
Atlas.define("act", {
  lead: `<p>神經網路的每個神經元先算「加權和」，再把結果送進一個函數轉換，這個函數叫<span class="jp">活性化関数</span>（活化函數，<span class="en">activation function</span>）。它解決兩件事：沒有它，疊再多層也只等於一條直線，畫不出曲線；而它的斜率，決定誤差逆伝播時梯度能不能傳回前面的層。</p>`,
  closing: `活化函數讓網路能表達曲線，但它的斜率會在反向傳播時被連乘：Sigmoid 的斜率最大只有 0.25、兩端趨近 0，層一深梯度就消失；ReLU 在正側斜率固定是 1，因此成為隱藏層的主流。輸出層則依任務選：二元分類用 Sigmoid，多元分類用ソフトマックス。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const sigm = (x) => 1 / (1 + Math.exp(-x));
    const FN = {
      sigmoid: { label: "シグモイド", f: sigm, d: (x) => sigm(x) * (1 - sigm(x)), range: "0〜1", maxd: "0.25（x = 0）" },
      tanh: { label: "tanh", f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2, range: "−1〜1", maxd: "1（x = 0）" },
      relu: { label: "ReLU", f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), range: "0〜∞", maxd: "1（x > 0）" },
      leaky: { label: "Leaky ReLU", f: (x) => (x > 0 ? x : 0.01 * x), d: (x) => (x > 0 ? 1 : 0.01), range: "−∞〜∞", maxd: "1（x > 0）" },
    };
    let mode = "shape", fk = "sigmoid", xin = 2;

    const modeSeg = ui.seg({
      label: "看什麼", value: mode,
      options: [{ value: "shape", label: "形狀與斜率" }, { value: "why", label: "為什麼要非線性" }, { value: "soft", label: "輸出層：ソフトマックス" }],
      onChange: (v) => { mode = v; show(); },
    });
    const top = h("div", { class: "bench__row" }, modeSeg.el);
    const body = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } });
    bench.append(top, body);
    const setStatus = (el, k, t) => { el.className = "status status--" + k; el.textContent = t; };

    /* ===================== A. 形狀與斜率 ===================== */
    const A = (() => {
      const fnSeg = ui.seg({ label: "活性化関数", value: fk, options: Object.keys(FN).map((k) => ({ value: k, label: FN[k].label })), onChange: (v) => { fk = v; draw(); } });
      const xSlider = ui.slider({ id: "act-x", label: "輸入 x（加權和）", min: -5, max: 5, step: 0.1, value: xin, format: (v) => fmt(v, 1), onInput: (v) => { xin = v; draw(); } });
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
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "活化函數與它的斜率", style: "display:block;width:100%;min-width:500px;max-width:760px;cursor:crosshair" });
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
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 12, class: "t-muted", text: "輸入 x" }));
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
        svg.append(s("text", { x: M.l + 40, y: M.t + 16, "font-size": 12, text: "函數 f(x)（輸出）" }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 32, y2: M.t + 32, stroke: "var(--hot)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 36, "font-size": 12, text: "斜率 f′(x)（反向傳播時要乘的數）" }));
        view.replaceChildren(svg);

        // status
        if (d === 0) setStatus(status, "bad", `x = ${fmt(xin, 1)}：斜率是 0，梯度傳到這裡就斷了，這個神經元學不動`);
        else if (d < 0.05) setStatus(status, "warn", `x = ${fmt(xin, 1)}：落在平坦區，斜率只剩 ${fmt(d, d < 0.001 ? 4 : 3)}，梯度幾乎傳不回去`);
        else if (d >= 0.999) setStatus(status, "ok", `x = ${fmt(xin, 1)}：斜率是 1，梯度原封不動傳回去`);
        else setStatus(status, "info", `x = ${fmt(xin, 1)}：輸出 ${fmt(y, 3)}，斜率 ${fmt(d, 3)}`);

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
          row("f′(x) = x > 0 時 1，x < 0 時 0", `= <span class="hl">${d}</span>`);
        else calc.innerHTML =
          row("f(x) = x > 0 時 x，x < 0 時 0.01x", `= <span class="hl">${fmt(y, xin > 0 ? 2 : 3)}</span>`) +
          row("f′(x) = x > 0 時 1，x < 0 時 0.01", `= <span class="hl">${fmt(d, 2)}</span>`);

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
          h("thead", null, h("tr", null, h("th", { class: "l" }, "函數"), h("th", null, "輸出範圍"), h("th", null, "最大斜率"), h("th", null, `x = ${fmt(xin, 1)} 的斜率`), h("th", null, "x = −4 的斜率"))),
          h("tbody", null, rows)));

        const NOTE = {
          sigmoid: `<span class="jp">シグモイド関数</span>（S 形函數，<span class="en">sigmoid</span>）把輸出壓在 0〜1，可以當成機率，所以常用在二元分類的輸出層。但斜率最大只有 0.25，|x| 一大就幾乎是 0；放在很多層的隱藏層裡，反向傳播每過一層梯度至少縮成四分之一。`,
          tanh: `<span class="jp">tanh 関数</span>（雙曲正切，<span class="en">hyperbolic tangent</span>）形狀和 Sigmoid 一樣是 S 形，但輸出在 −1〜1、以 0 為中心，最大斜率是 1，比 Sigmoid 不容易讓梯度變小。兩端仍然平坦，|x| 大時斜率同樣趨近 0。`,
          relu: `<span class="jp">ReLU 関数</span>（線性整流函數，<span class="en">rectified linear unit</span>）正側斜率永遠是 1，梯度通過時不打折，計算也只要比大小，因此成為深層網路隱藏層的標準選擇。缺點是負側斜率為 0：輸入一直是負的神經元，梯度永遠是 0，再也學不動。`,
          leaky: `<span class="jp">Leaky ReLU 関数</span>（帶洩漏的 ReLU，<span class="en">leaky ReLU</span>）在負側留一個很小的斜率（常用 0.01），圖上看起來貼近 0，但不是 0，所以負側的神經元仍能收到一點梯度，避免 ReLU「學不動」的問題。`,
        };
        note.innerHTML = NOTE[fk] + "　可以拖動滑桿，或直接點圖改變輸入 x。";
      }
      return { el, show: draw };
    })();

    /* ===================== B. 為什麼要非線性 ===================== */
    const B = (() => {
      let useAct = "relu", n = 4;
      const actSeg = ui.seg({ label: "隱藏層的活性化関数", value: useAct, options: [{ value: "none", label: "無（直接輸出加權和）" }, { value: "relu", label: "ReLU" }], onChange: (v) => { useAct = v; draw(); } });
      const nSlider = ui.slider({ id: "act-n", label: "隱藏層神經元數", min: 1, max: 8, step: 1, value: n, format: (v) => v + " 個", onInput: (v) => { n = v; draw(); } });
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
        const b = Array.from({ length: k }, (_, i) => Phi.reduce((acc, r, t) => acc + r[i] * ys[t], 0));
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

        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "用神經元組合出的函數與目標曲線", style: "display:block;width:100%;min-width:500px;max-width:760px" });
        svg.append(s("defs", null, s("clipPath", { id: "actclip2" }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        for (let v = -1.5; v <= 1.5; v += 0.5) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: fmt(v, 1) }));
        }
        for (let x = X0; x <= X1; x++) svg.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: fmt(x, 0) }));
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 12, class: "t-muted", text: "輸入 x" }));
        const g = s("g", { "clip-path": "url(#actclip2)" });
        let d = ""; xs.forEach((x, i) => { d += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(ys[i]).toFixed(1); });
        g.append(s("path", { d, fill: "none", stroke: "var(--faint)", "stroke-width": 7, "stroke-linecap": "round", opacity: 0.5 }));
        let d2 = ""; for (let i = 0; i <= 300; i++) { const x = X0 + (i / 300) * (X1 - X0); d2 += (i ? "L" : "M") + sx(x).toFixed(1) + "," + sy(fit(x)).toFixed(1); }
        g.append(s("path", { d: d2, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5 }));
        if (useAct === "relu") bs.forEach((b) => g.append(s("circle", { cx: sx(b), cy: sy(fit(b)), r: 5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 })));
        svg.append(g);
        svg.append(s("rect", { x: M.l + 4, y: M.t + 1, width: 200, height: useAct === "relu" ? 64 : 44, rx: 6, fill: "var(--panel)", "fill-opacity": 0.92 }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 12, y2: M.t + 12, stroke: "var(--faint)", "stroke-width": 7, opacity: 0.5, "stroke-linecap": "round" }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 16, "font-size": 12, text: "想學的曲線 y = sin x" }));
        svg.append(s("line", { x1: M.l + 10, x2: M.l + 34, y1: M.t + 32, y2: M.t + 32, stroke: "var(--accent)", "stroke-width": 2.5 }));
        svg.append(s("text", { x: M.l + 40, y: M.t + 36, "font-size": 12, text: "網路的輸出" }));
        if (useAct === "relu") {
          svg.append(s("circle", { cx: M.l + 22, cy: M.t + 52, r: 5, fill: "var(--hot)" }));
          svg.append(s("text", { x: M.l + 40, y: M.t + 56, "font-size": 12, text: "每個 ReLU 神經元的轉折點" }));
        }
        view.replaceChildren(svg);

        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        if (useAct === "relu") {
          setStatus(status, mse < 0.005 ? "ok" : "info", `${n} 個 ReLU 神經元 → ${n} 個轉折點的折線，平均平方誤差 ${fmt(mse, 4)}`);
          const terms = a.map((ai, i) => `${fmtP(ai, 2)}·ReLU(x ${bs[i] < 0 ? "+" : "−"} ${fmt(Math.abs(bs[i]), 2)})`);
          calc.innerHTML = row("輸出 = c + Σ aᵢ × ReLU(x − bᵢ)", "") +
            `<span class="calc-row">= ${fmt(c0, 2)} + ${terms.join(" + ")}</span>` +
            row("每個神經元", "在 x < bᵢ 時輸出 0、之後沿斜線上升；加起來，每過一個轉折點斜率就改變一次");
        } else {
          setStatus(status, "bad", `${n} 個神經元都沒用：輸出還是一條直線，平均平方誤差 ${fmt(mse, 4)}`);
          calc.innerHTML = row("輸出 = c + Σ aᵢ × (x − bᵢ)", `= (a₁ + … + aₙ)·x + (c − a₁b₁ − … − aₙbₙ)`) +
            row("整理後只剩", `斜率 × x + 截距；最佳直線 = ${fmt(slope, 3)}·x ${icpt < 0 ? "−" : "+"} ${fmt(Math.abs(icpt), 3)}`) +
            row("同樣的道理", "線性的層疊再多層，合起來仍是一個線性函數，等於只有一層");
        }
        note.innerHTML = `示範網路只有 1 個輸入、1 層隱藏層、1 個輸出；隱藏神經元 i 計算 ReLU(x − bᵢ)，轉折點 bᵢ 平均排開。輸出權重 aᵢ 和 c 是用最小平方法直接算出的最佳值（不是訓練過程），用來看「這個網路最多能畫出什麼形狀」。`;
      }
      return { el, show: draw };
    })();

    /* ===================== C. ソフトマックス ===================== */
    const C = (() => {
      const CLS = ["貓", "狗", "鳥"];
      let z = [2.0, 1.0, -0.5];
      const sliders = CLS.map((c, i) => ui.slider({ id: "act-z" + i, label: `「${c}」的分數 z<sub>${i + 1}</sub>`, min: -3, max: 5, step: 0.1, value: z[i], format: (v) => fmt(v, 1), onInput: (v) => { z[i] = v; draw(); } }));
      const view = h("div", { class: "scroll-x" });
      const status = h("div", { class: "status" });
      const calc = h("div", { class: "calc" });
      const note = h("p", { class: "note" });
      const el = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, ...sliders.map((x) => x.el)), view, status, calc, note);

      function draw() {
        const e = z.map(Math.exp), S = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / S);
        const W = 640, H = 230, colX = [150, 330, 510], base = 190, top = 40;
        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "分數經過 softmax 變成機率", style: "display:block;width:100%;min-width:460px;max-width:760px" });
        svg.append(s("text", { x: 12, y: 20, "font-size": 12, class: "t-muted", text: "分數 z → e^z → 除以總和 → 機率（三者加總 = 1）" }));
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
        setStatus(status, "info", `機率最高：「${CLS[best]}」 ${fmt(p[best] * 100, 1)}%`);
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        calc.innerHTML = row("pᵢ = e<sup>zᵢ</sup> ÷ (e<sup>z₁</sup> + e<sup>z₂</sup> + e<sup>z₃</sup>)", `分母 = ${e.map((v) => fmt(v, 3)).join(" + ")} = ${fmt(S, 3)}`) +
          CLS.map((c, i) => row(`p(${c})`, `= ${fmt(e[i], 3)} ÷ ${fmt(S, 3)} = <span class="hl">${fmt(p[i], 3)}</span>`)).join("") +
          row("加總", `${p.map((v) => fmt(v, 3)).join(" + ")} ≈ 1　<span class="k">（不四捨五入時剛好等於 1）</span>`);
        note.innerHTML = `<span class="jp">ソフトマックス関数</span>（softmax 函數，<span class="en">softmax</span>）放在多元分類的輸出層：先取 e 的次方讓每個數都變成正數，再除以總和，所以每類都在 0〜1 之間、加起來剛好是 1。分數的大小順序不變，但差距會被放大：試著只把「貓」的分數加 1，看它的機率漲多少。二元分類只需要一個輸出時，用シグモイド関数即可。輸出的機率接著用交差エントロピー算損失，見 <a href="#loss">誤差関数</a>。`;
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
