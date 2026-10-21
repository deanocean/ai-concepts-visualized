/* 誤差関数（シラバス §13）
   四個模式，全部即時計算：
   ① 平均二乗誤差：5 個點、模型 y = w·x，每個誤差畫成正方形（面積＝誤差²），右邊是 MSE 對 w 的曲線。
   ② 交差エントロピー：二元分類、正解＝1，拖動 z（シグモイド前的分數），比較兩種損失與「對 z 的梯度」。
   ③ KL 情報量：4 類的真實分布 P 與模型分布 Q，逐項算 P·ln(P/Q)。
   ④ Contrastive Loss／Triplet Loss：在 2D 嵌入平面拖動點，看損失何時歸零；可照梯度移動一步。 */
Atlas.define("loss", {
  lead: `<p>要用<a href="#gd">勾配降下法</a>讓模型變好，得先把「模型錯了多少」變成<b>一個數字</b>，而且這個數字要能求出往哪邊改會變小。這個數字的算法就是<span class="jp">誤差関数</span>（損失函數，<span class="en">loss function</span>）。任務不同，「錯」的意思不同：預測數值看差多遠，分類看給正解多少機率，比較兩個分布看差多少，學習相似度看該近的近不近、該遠的遠不遠。</p>`,
  closing: `迴歸用<span class="jp">平均二乗誤差</span>，分類用<span class="jp">交差エントロピー</span>（模型很有把握地答錯時，梯度仍然很大，學得動）；兩個機率分布的差距用 <span class="jp">KL 情報量</span>；想讓相似的東西在向量空間裡靠近、不相似的遠離，用 <span class="jp">Contrastive Loss</span> 或 <span class="jp">Triplet Loss</span>。選錯損失，模型就會朝錯的目標學。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    let mode = "mse";
    const modeSeg = ui.seg({ label: "模式", value: mode,
      options: [{ value: "mse", label: "① 平均二乗誤差" }, { value: "ce", label: "② 交差エントロピー" }, { value: "kl", label: "③ KL 情報量" }, { value: "metric", label: "④ Contrastive／Triplet" }],
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
    function setStatus(el, kind, t) { el.className = "status status--" + kind; el.textContent = t; }
    const axisNum = (v) => (v < 0 ? "−" + -v : String(v));

    /* ===================== ① 平均二乗誤差 ===================== */
    function modeMSE() {
      const P = [[0.5, 0.9], [1.0, 1.0], [1.5, 2.1], [2.0, 1.8], [2.5, 2.9]];
      const W_OPT = P.reduce((a, p) => a + p[0] * p[1], 0) / P.reduce((a, p) => a + p[0] * p[0], 0);
      const mse = (w) => P.reduce((a, p) => a + (p[1] - w * p[0]) ** 2, 0) / P.length;
      let w = 0.6;
      const sl = ui.slider({ id: "ls-w", label: "模型 y = w·x 的斜率 w", min: 0, max: 2.5, step: 0.05, value: w, format: (v) => v.toFixed(2), onInput: (v) => { w = v; draw(); } });
      const fit = s("svg", { viewBox: "0 0 340 320", role: "img", "aria-label": "資料點、模型直線與誤差正方形" });
      const curve = s("svg", { viewBox: "0 0 340 320", role: "img", "aria-label": "平均二乗誤差隨 w 的變化" });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      box.append(
        h("div", { class: "bench__row" }, sl.el, ui.btn("跳到最小的 w", () => { w = Math.round(W_OPT * 20) / 20; sl.set(w); draw(); }, { sm: true })),
        h("div", { class: "scroll-x" }, h("div", { class: "ls-figs" },
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, "每個誤差畫成正方形：面積＝誤差²"), fit),
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, "平均二乗誤差 vs. w"), curve))),
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
        curve.append(s("text", { x: cx(W_OPT), y: cy(5.1) - 6, "text-anchor": "middle", "font-size": 11.5, class: "t-halo", fill: "var(--ok)", text: `最小 w = ${fmt(W_OPT, 3)}` }));
        curve.append(s("circle", { cx: cx(w), cy: cy(mse(w)), r: 7, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2.5 }));
        // 文字
        const es = P.map(([x, y]) => y - w * x), m = mse(w);
        if (Math.abs(w - W_OPT) < 0.03) setStatus(status, "ok", `正方形的總面積已經最小：MSE = ${fmt(m, 3)}`);
        else setStatus(status, "info", `MSE = ${fmt(m, 3)}；把 w 往${w < W_OPT ? "右" : "左"}調，正方形總面積會變小`);
        calc.innerHTML =
          `<span class="calc-row"><span class="k">各點誤差 y − w·x</span> ${P.map(([x, y], i) => `${fmt(y, 1)} − ${fmt(w, 2)}×${fmt(x, 1)} = ${fmt(es[i], 2)}`).join("；")}</span>` +
          `<span class="calc-row"><span class="k">平均二乗誤差</span> (${es.map((e) => fmtP(e, 2) + "²").join(" + ")}) ÷ 5 = (${es.map((e) => fmt(e * e, 3)).join(" + ")}) ÷ 5 = <span class="hl">${fmt(m, 3)}</span></span>`;
      }
      note.innerHTML = `平方有兩個作用：正負誤差不會互相抵消；離得越遠扣得越重（差 2 扣 4，差 3 扣 9），所以一個離群很遠的點會把直線拉過去。MSE 對 w 是一條拋物線，只有一個最低點，用梯度很好找。這是迴歸（預測連續數值）最常用的損失。`;
      draw();
      return {};
    }

    /* ===================== ② 交差エントロピー ===================== */
    function modeCE() {
      let z = -3;
      const sig = (v) => 1 / (1 + Math.exp(-v));
      const sl = ui.slider({ id: "ls-z", label: "模型的分數 z（p = シグモイド(z)）", min: -4.5, max: 4.5, step: 0.1, value: z, format: (v) => fmt(v, 1), onInput: (v) => { z = v; draw(); } });
      const lossSvg = s("svg", { viewBox: "0 0 340 280", role: "img", "aria-label": "兩種損失隨 p 的變化" });
      const gradSvg = s("svg", { viewBox: "0 0 340 280", role: "img", "aria-label": "兩種損失對 z 的梯度大小" });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      const pre = h("div", { class: "btn-group" },
        ...[["很有把握地答錯", -4], ["猶豫", 0], ["很有把握地答對", 4]].map(([l, v]) => ui.btn(l, () => { z = v; sl.set(v); draw(); }, { sm: true })));
      box.append(
        h("p", { class: "hint", html: "情境：判斷圖片是不是貓，正解是「是」（t = 1）。模型輸出 p＝它認為「是貓」的機率。" }),
        h("div", { class: "bench__row" }, sl.el, h("div", { class: "field" }, h("span", { class: "field__label" }, "快速選擇"), pre)),
        h("div", { class: "scroll-x" }, h("div", { class: "ls-figs" },
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, "損失（越大＝扣越多）"), lossSvg),
          h("div", { class: "ls-fig" }, h("div", { class: "ls-cap" }, "對 z 的梯度大小（越大＝這一步改越多）"), gradSvg))),
        status, calc);
      const CEf = (p) => -Math.log(p), MSEf = (p) => (1 - p) ** 2;
      const CEg = (p) => Math.abs(p - 1), MSEg = (p) => Math.abs(2 * (p - 1) * p * (1 - p));
      function plot(svg, fns, YM, ticks, p) {
        const W = 340, H = 280, M = { l: 36, r: 12, t: 30, b: 30 };
        const px = (v) => M.l + v * (W - M.l - M.r), py = (v) => H - M.b - (Math.min(v, YM) / YM) * (H - M.t - M.b);
        svg.replaceChildren();
        for (const t of ticks) {
          svg.append(s("line", { x1: M.l, x2: W - M.r, y1: py(t), y2: py(t), stroke: "var(--line)" }));
          svg.append(s("text", { x: M.l - 6, y: py(t) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: t }));
        }
        for (const v of [0, 0.25, 0.5, 0.75, 1]) svg.append(s("text", { x: px(v), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v }));
        svg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 11.5, class: "t-muted", text: "p（給正解的機率）" }));
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
        plot(lossSvg, [[CEf, "var(--neg)", "交差エントロピー"], [MSEf, "var(--pos)", "平均二乗誤差"]], 4.6, [0, 1, 2, 3, 4], p);
        plot(gradSvg, [[CEg, "var(--neg)", "交差エントロピー"], [MSEg, "var(--pos)", "平均二乗誤差"]], 1.05, [0, 0.25, 0.5, 0.75, 1], p);
        const gc = CEg(p), gm = MSEg(p);
        if (p < 0.2) setStatus(status, "warn", `答錯而且很有把握（p = ${fmt(p, 3)}）：交差エントロピー的梯度 ${fmt(gc, 3)}，平均二乗誤差只有 ${fmt(gm, 3)}，用它幾乎改不動`);
        else if (p > 0.9) setStatus(status, "ok", `已經答對（p = ${fmt(p, 3)}）：兩種損失和梯度都接近 0`);
        else setStatus(status, "info", `p = ${fmt(p, 3)}：梯度大小　交差エントロピー ${fmt(gc, 3)}，平均二乗誤差 ${fmt(gm, 3)}`);
        calc.innerHTML =
          `<span class="calc-row"><span class="k">機率</span> p = シグモイド(z) = 1 ÷ (1 + e<sup>−z</sup>) = 1 ÷ (1 + e<sup>${fmt(-z, 1)}</sup>) = <b>${fmt(p, 4)}</b></span>` +
          `<span class="calc-row"><span class="k">交差エントロピー</span> −t·ln p = −ln ${fmt(p, 4)} = <span class="n">${fmt(CEf(p), 3)}</span>　<span class="k">對 z 的梯度</span> p − t = ${fmt(p - 1, 3)}</span>` +
          `<span class="calc-row"><span class="k">平均二乗誤差</span> (t − p)² = ${fmtP(1 - p, 4)}² = <span class="p">${fmt(MSEf(p), 3)}</span>　<span class="k">對 z 的梯度</span> 2(p − t) × p(1 − p) = 2 × ${fmtP(p - 1, 3)} × ${fmt(p * (1 - p), 4)} = ${fmt(2 * (p - 1) * p * (1 - p), 4)}</span>`;
      }
      note.innerHTML = `問題出在シグモイド的斜率 p(1 − p)（見 <a href="#act">活性化関数</a>）：p 接近 0 或 1 時斜率接近 0。平均二乗誤差的梯度要乘上這個斜率，所以「錯得很有把握」時反而學不動；交差エントロピー的 ln 剛好把它抵消，梯度只剩 p − t。多類別時寫成 −Σ t<sub>k</sub>·ln y<sub>k</sub>（t 是正解的 one-hot，y 是ソフトマックス的輸出），只有正解那一項不是 0，等於 −ln（給正解的機率）。這裡的 ln 是自然對數。`;
      draw();
      return {};
    }

    /* ===================== ③ KL 情報量 ===================== */
    function modeKL() {
      const CAT = ["晴", "陰", "雨", "雪"];
      const P = [0.5, 0.3, 0.15, 0.05];
      let Q = [0.25, 0.25, 0.25, 0.25];
      const norm = (a) => { const t = a.reduce((x, y) => x + y, 0); return a.map((v) => v / t); };
      const nudge = (i, f) => { const q = Q.slice(); q[i] = Math.min(0.97, Math.max(0.005, q[i] * f)); Q = norm(q); draw(); };
      const qBox = h("div", { class: "ls-q" }, ...CAT.map((c, i) => h("span", null, `Q(${c})`,
        ui.btn("−", () => nudge(i, 0.75), { sm: true, aria: `降低 ${c} 的機率` }), ui.btn("＋", () => nudge(i, 1.3333), { sm: true, aria: `提高 ${c} 的機率` }))));
      const pre = h("div", { class: "btn-group" }, ...[
        ["和 P 一樣", P], ["平均分配", [0.25, 0.25, 0.25, 0.25]], ["順序搞反", [0.05, 0.15, 0.3, 0.5]], ["幾乎不給雪", [0.52, 0.31, 0.165, 0.005]],
      ].map(([l, q]) => ui.btn(l, () => { Q = q.slice(); draw(); }, { sm: true })));
      const svg = s("svg", { viewBox: "0 0 560 250", role: "img", "aria-label": "真實分布 P 與模型分布 Q 的長條圖", style: "display:block;width:100%;min-width:420px;max-width:640px" });
      const status = h("div", { class: "status" }), tbl = h("table", { class: "tbl" }), calc = h("div", { class: "calc" });
      box.append(
        h("p", { class: "hint", html: "P：某地真實的天氣比例；Q：模型預測的比例。調 Q，看兩個分布差多少。" }),
        h("div", { class: "bench__row" }, h("div", { class: "field" }, h("span", { class: "field__label" }, "快速選擇 Q"), pre)),
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
        svg.append(s("text", { x: M.l + 26, y: 16, "font-size": 12, text: "P（真實）" }));
        svg.append(s("rect", { x: M.l + 110, y: 6, width: 14, height: 12, rx: 2, fill: "var(--hot)" }));
        svg.append(s("text", { x: M.l + 130, y: 16, "font-size": 12, text: "Q（模型）" }));
        const terms = P.map((p, i) => p * ln(p / Q[i])), kl = terms.reduce((a, b) => a + b, 0);
        const klr = Q.reduce((a, q, i) => a + q * ln(q / P[i]), 0), H_P = -P.reduce((a, p) => a + p * ln(p), 0);
        tbl.replaceChildren(
          h("thead", null, h("tr", null, ...["類別", "P", "Q", "ln(P ÷ Q)", "P × ln(P ÷ Q)"].map((t, k) => h("th", { class: k ? null : "l" }, t)))),
          h("tbody", null, ...CAT.map((c, i) => h("tr", null, h("td", { class: "l" }, c), h("td", null, fmt(P[i], 3)), h("td", null, fmt(Q[i], 3)), h("td", null, fmt(ln(P[i] / Q[i]), 3)), h("td", null, fmt(terms[i], 4))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l", colspan: 4 }, "合計＝KL(P‖Q)"), h("td", null, fmt(kl, 4)))));
        if (kl < 1e-6) setStatus(status, "ok", "Q 和 P 完全一樣：KL(P‖Q) = 0");
        else {
          const worst = terms.indexOf(Math.max(...terms));
          setStatus(status, kl > 0.5 ? "bad" : "info", `KL(P‖Q) = ${fmt(kl, 3)}，貢獻最多的是「${CAT[worst]}」：P 認為常發生，Q 卻給得太少`);
        }
        calc.innerHTML =
          `<span class="calc-row"><span class="k">KL(P‖Q)</span> Σ P·ln(P ÷ Q) = ${terms.map((t) => fmtP(t, 4)).join(" + ")} = <span class="hl">${fmt(kl, 4)}</span></span>` +
          `<span class="calc-row"><span class="k">反過來 KL(Q‖P)</span> Σ Q·ln(Q ÷ P) = ${fmt(klr, 4)}　<span class="k">（和上面不一樣：KL 不對稱）</span></span>` +
          `<span class="calc-row"><span class="k">交差エントロピー</span> −Σ P·ln Q = ${fmt(H_P + kl, 4)} = <span class="k">P 自己的エントロピー</span> ${fmt(H_P, 4)} + <span class="k">KL</span> ${fmt(kl, 4)}</span>`;
      }
      note.innerHTML = `<span class="jp">カルバック・ライブラー情報量</span>（KL 散度，<span class="en">Kullback–Leibler divergence</span>）衡量「用 Q 去近似 P 損失多少」，永遠 ≥ 0，只有 Q = P 時是 0。P 很大、Q 卻很小的類別扣得最重（試「幾乎不給雪」與「順序搞反」）。因為 P 自己的エントロピー是固定的，讓交差エントロピー變小就等於讓 KL 變小，所以分類用交差エントロピー，其實就是在拉近模型和正解的分布。生成模型 <a href="#vae">VAE</a> 也用 KL，把壓縮後的分布拉向標準常態分布。`;
      draw();
      return {};
    }

    /* ===================== ④ Contrastive／Triplet ===================== */
    function modeMetric() {
      let kind = "triplet", margin = 0.5, same = false;
      const INIT = { A: [-1, 0], P: [0.6, 0.8], N: [0.2, -0.5], B: [0, 0.3] };
      let pts = JSON.parse(JSON.stringify(INIT));
      const kindSeg = ui.seg({ label: "損失", value: kind, options: [{ value: "triplet", label: "Triplet Loss（三個一組）" }, { value: "contrastive", label: "Contrastive Loss（兩個一組）" }],
        onChange: (v) => { kind = v; margin = v === "triplet" ? 0.5 : 1.5; mSl.set(margin); sync(); draw(); } });
      const mSl = ui.slider({ id: "ls-m", label: "マージン m（至少要拉開的距離）", min: 0, max: 2.5, step: 0.1, value: margin, format: (v) => v.toFixed(1), onInput: (v) => { margin = v; draw(); } });
      const pairSeg = ui.seg({ label: "這兩個是", value: same, options: [{ value: true, label: "同一個人" }, { value: false, label: "不同人" }], onChange: (v) => { same = v; draw(); } });
      const svg = s("svg", { viewBox: "0 0 600 400", role: "img", "aria-label": "嵌入空間中的點，可拖動", style: "display:block;width:100%;min-width:400px;max-width:640px;user-select:none" });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
      const stepBtn = ui.btn("照梯度移動一步（η = 0.3）", () => { gradStep(); }, { primary: true });
      const resetBtn = ui.btn("重設", () => { pts = JSON.parse(JSON.stringify(INIT)); draw(); });
      box.append(
        h("p", { class: "hint", html: "情境：人臉辨識。模型把每張臉變成平面上的一個點（嵌入），希望同一個人的照片靠近、不同人的遠離。<b>點可以拖動</b>。" }),
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
          ? [["var(--ok)", "4 4", "綠圈：半徑 d(A,P)"], ["var(--neg)", "8 5", "紅圈：半徑 d(A,P) + m，N 要在圈外"]]
          : same ? [] : [["var(--neg)", "8 5", "紅圈：半徑 m，不同人要在圈外"]];
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
        const LAB = { A: "A 錨點", P: "P 同一人", N: "N 不同人", B: same ? "B 同一人" : "B 不同人" };
        const COL = { A: "var(--ink)", P: "var(--ok)", N: "var(--neg)", B: same ? "var(--ok)" : "var(--neg)" };
        for (const k of names) {
          const c = s("circle", { cx: X(pts[k][0]), cy: Y(pts[k][1]), r: 11, fill: COL[k], stroke: "var(--panel)", "stroke-width": 2.5, class: "ls-pt", "data-k": k, tabindex: 0, role: "button", "aria-label": `拖動 ${LAB[k]}` });
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
          if (info.L === 0) setStatus(status, "ok", `損失 0：N 比 P 遠了 m 以上（${fmt(info.dn, 2)} ≥ ${fmt(info.dp, 2)} + ${fmt(margin, 1)}），這組不再推動模型`);
          else setStatus(status, info.dn < info.dp ? "bad" : "warn", info.dn < info.dp ? `不同人 N 反而比同一人 P 還近：損失 ${fmt(info.L, 3)}` : `N 比 P 遠，但還沒拉開 m：損失 ${fmt(info.L, 3)}`);
          calc.innerHTML =
            `<span class="calc-row"><span class="k">距離</span> d(A,P) = ${fmt(info.dp, 3)}　d(A,N) = ${fmt(info.dn, 3)}</span>` +
            `<span class="calc-row"><span class="k">Triplet Loss</span> max(0, d(A,P) − d(A,N) + m) = max(0, ${fmt(info.dp, 3)} − ${fmt(info.dn, 3)} + ${fmt(margin, 1)}) = <span class="hl">${fmt(info.L, 3)}</span></span>` +
            (info.L > 0 ? `<span class="calc-row"><span class="k">梯度的方向（箭頭）</span> P 沿著連線往 A 靠近，N 沿著連線遠離 A，每步各移動 η = 0.3</span>` : "");
        } else {
          if (info.L === 0) setStatus(status, "ok", same ? "兩點重合：損失 0" : `距離 ${fmt(info.d, 2)} 已超過 m = ${fmt(margin, 1)}：損失 0，不再推開`);
          else setStatus(status, "warn", same ? `同一個人卻相距 ${fmt(info.d, 2)}：損失 ${fmt(info.L, 3)}，要拉近` : `不同人卻只相距 ${fmt(info.d, 2)}（小於 m）：損失 ${fmt(info.L, 3)}，要推開`);
          calc.innerHTML =
            `<span class="calc-row"><span class="k">距離</span> d(A,B) = ${fmt(info.d, 3)}</span>` +
            (same ? `<span class="calc-row"><span class="k">同一人</span> ½·d² = ½ × ${fmt(info.d, 3)}² = <span class="hl">${fmt(info.L, 3)}</span></span>`
                  : `<span class="calc-row"><span class="k">不同人</span> ½·max(0, m − d)² = ½ × max(0, ${fmt(margin, 1)} − ${fmt(info.d, 3)})² = <span class="hl">${fmt(info.L, 3)}</span></span>`);
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
      note.innerHTML = `這兩種損失不看單一答案對不對，而是看<b>距離</b>：學出來的向量空間裡，同類靠近、不同類遠離，之後遇到沒見過的人，也能用「離誰近」來比對（距離學習，<span class="en">metric learning</span>）。マージン m 的作用是「夠遠就好」：超過 m 損失就是 0，模型不會無止境地把不同類推開。這裡為了好觀察，錨點 A 固定不動、點位置直接拖動；真實訓練時移動的是產生這些向量的網路權重。Triplet Loss 一次比較三個（錨點、正例、負例），Contrastive Loss 一次比較兩個並標明是否同類。`;
      draw();
      return {};
    }

    const MODES = { mse: modeMSE, ce: modeCE, kl: modeKL, metric: modeMetric };
    show(); // 圖都用 CSS 變數上色，切換深淺色不需重畫
    return {};
  },
});
