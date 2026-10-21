/* 位置エンコーディング（シラバス §23）
   ① 為什麼需要：「狗 咬 人」與「人 咬 狗」做 Self-Attention（Q＝K＝V＝輸入，d＝4），
      不加位置時「狗」的輸出完全相同；加上 sin・cos 位置編碼後就不同。
   ② 長什麼樣：PE(pos, 2i) = sin(pos / 10000^(2i/d))、PE(pos, 2i+1) = cos(…) 的熱圖，
      以及某個位置和其他位置的相似度（內積 ÷ (d/2)）。 */
Atlas.define("posenc", {
  lead: `<p>Self-Attention 一次看整句、每個詞直接互相比較，不像 RNN 按順序一格格讀，所以「狗咬人」和「人咬狗」在它眼中是同一袋詞。<span class="jp">位置エンコーディング</span>（位置編碼，<span class="en">positional encoding</span>）把「第幾個」也變成一組數字，加到詞向量上，模型才分得出順序。</p>`,
  closing: `Transformer 用不同頻率的 sin・cos 當位置編號：前面的維度變得快、後面的維度變得慢，像時鐘的秒針、分針、時針，組合起來每個位置都獨一無二；而且兩個位置的相似度只由它們隔多遠決定，相鄰的位置最像。`,

  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const pe = (p, d) => Array.from({ length: d }, (_, i) => { const w = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d); return i % 2 ? Math.cos(w) : Math.sin(w); });
    const E = { "狗": [1, 0, 0.5, 0], "咬": [0, 1, 0, 0.5], "人": [0.5, 0, 1, 0] };
    const SENTS = [["狗", "咬", "人"], ["人", "咬", "狗"]];
    const dot = (a, b) => a.reduce((t, v, i) => t + v * b[i], 0);
    function attend(sent, usePE) {
      const X = sent.map((w, i) => E[w].map((v, d) => v + (usePE ? pe(i, 4)[d] : 0)));
      const A = X.map((q) => { const sc = X.map((k) => dot(q, k) / 2), e = sc.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); });
      const Z = A.map((w) => [0, 1, 2, 3].map((d) => w.reduce((a, wj, j) => a + wj * X[j][d], 0)));
      return { X, A, Z };
    }

    let view = "why", usePE = false, word = "狗", d = 32, p = 5;
    const vec = (v, k = 2) => "[" + v.map((x) => fmt(x, k)).join(", ") + "]";
    const pct = (w) => Math.round(w * 100) + "%";

    const viewSeg = ui.seg({ value: view, label: "看哪一部分", options: [{ value: "why", label: "① 為什麼需要" }, { value: "wave", label: "② sin・cos 長什麼樣" }], onChange: (v) => { view = v; render(); } });
    const peSeg = ui.seg({ value: usePE, label: "位置資訊", options: [{ value: false, label: "不加" }, { value: true, label: "加上位置エンコーディング" }], onChange: (v) => { usePE = v; render(); } });
    const wordSeg = ui.seg({ value: word, label: "比較哪個詞", options: ["狗", "咬", "人"].map((w) => ({ value: w, label: w })), onChange: (v) => { word = v; render(); } });
    const dSeg = ui.seg({ value: d, label: "向量維數 d", options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { d = v; render(); } });
    const pSlider = ui.slider({ id: "pe-pos", label: "位置 pos", min: 0, max: 49, step: 1, value: p, format: (v) => String(v), onInput: (v) => { p = v; render(); } });

    const rowWhy = h("div", { class: "bench__row" }, peSeg.el, wordSeg.el);
    const rowWave = h("div", { class: "bench__row" }, dSeg.el, pSlider.el);
    const status = h("div", { class: "status" });
    const fig = h("div", { class: "scroll-x" });
    const fig2 = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });
    bench.append(h("div", { class: "bench__row" }, viewSeg.el), rowWhy, rowWave, status, fig, fig2, calc, note);

    /* ---------- ① why ---------- */
    function renderWhy() {
      const R = SENTS.map((st) => attend(st, usePE));
      const W = 780, rowH = 128, H = rowH * 2 + 16;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "兩句話中同一個詞的注意力與輸出", style: `display:block;width:100%;min-width:${W * 0.78}px;max-width:${W}px` });
      const tx = (i) => 150 + i * 100;
      R.forEach((r, si) => {
        const y0 = si * rowH + 8, base = y0 + 84;
        const st = SENTS[si], wi = st.indexOf(word);
        svg.append(s("rect", { x: 2, y: y0, width: W - 4, height: rowH - 8, rx: 10, fill: "var(--panel-2)", stroke: "var(--line)" }));
        svg.append(s("text", { x: 14, y: y0 + 24, "font-size": 13, "font-weight": 700, text: `句子 ${si ? "B" : "A"}` }));
        svg.append(s("text", { x: 14, y: y0 + 44, "font-size": 13, class: "t-muted", text: st.join("") }));
        // arcs from selected word
        r.A[wi].forEach((w, j) => {
          const sw = 1 + 12 * w, op = 0.25 + 0.75 * w;
          const d = j === wi
            ? `M${tx(j) - 10},${base - 18} C${tx(j) - 30},${base - 64} ${tx(j) + 30},${base - 64} ${tx(j) + 10},${base - 18}`
            : `M${tx(wi)},${base - 18} Q${(tx(wi) + tx(j)) / 2},${base - 40 - Math.abs(j - wi) * 18} ${tx(j)},${base - 18}`;
          svg.append(s("path", { d, fill: "none", stroke: "var(--hot)", "stroke-width": sw, "stroke-opacity": op, "stroke-linecap": "round" }));
        });
        st.forEach((w, j) => {
          const sel = j === wi;
          svg.append(s("rect", { x: tx(j) - 30, y: base - 18, width: 60, height: 32, rx: 8, fill: sel ? "var(--hot)" : "var(--panel)", stroke: sel ? "var(--hot)" : "var(--line-strong)" }));
          svg.append(s("text", { x: tx(j), y: base + 4, "text-anchor": "middle", "font-size": 15, "font-weight": sel ? 700 : 400, fill: sel ? "var(--panel)" : "var(--ink)", text: w }));
          svg.append(s("text", { x: tx(j), y: base + 30, "text-anchor": "middle", "font-size": 11.5, class: "t-mono t-muted", text: `位置 ${j}・${pct(r.A[wi][j])}` }));
        });
        // output vector
        const ox = 470;
        svg.append(s("text", { x: ox, y: y0 + 30, "font-size": 12, "font-weight": 700, text: `「${word}」的輸出` }));
        r.Z[wi].forEach((v, k) => {
          svg.append(s("rect", { x: ox + k * 74, y: y0 + 46, width: 68, height: 34, rx: 6, fill: Atlas.divColor(v, 2), stroke: "var(--line-strong)" }));
          svg.append(s("text", { x: ox + k * 74 + 34, y: y0 + 68, "text-anchor": "middle", "font-size": 13, class: "t-mono", fill: Math.abs(v) > 1.5 ? "var(--panel)" : "var(--ink)", text: fmt(v, 3) }));
          if (si === 0) svg.append(s("text", { x: ox + k * 74 + 34, y: y0 + 98, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: `第 ${k + 1} 維` }));
        });
      });
      fig.replaceChildren(svg); fig2.replaceChildren();
      const za = R[0].Z[SENTS[0].indexOf(word)], zb = R[1].Z[SENTS[1].indexOf(word)];
      const diff = za.map((v, k) => Math.abs(v - zb[k])), mx = Math.max(...diff), k = diff.indexOf(mx);
      const same = mx < 1e-9;
      status.className = "status status--" + (same ? "bad" : "ok");
      status.textContent = same
        ? (word === "咬" ? `兩句中「咬」的輸出完全相同：模型看不出誰在它前面、誰在後面` : `兩句中「${word}」的輸出完全相同：模型分不出是誰咬誰`)
        : `加上位置後，兩句的「${word}」輸出不同了（差最多的是第 ${k + 1} 維：${fmt(za[k], 3)} 對 ${fmt(zb[k], 3)}）`;
      if (!usePE) {
        calc.innerHTML = `<span class="calc-row"><span class="k">輸入只有詞向量</span> 狗 = ${vec(E["狗"], 1)}　咬 = ${vec(E["咬"], 1)}　人 = ${vec(E["人"], 1)}</span>` +
          `<span class="calc-row"><span class="k">注意力權重只由「詞和詞像不像」決定，跟它排在第幾個無關；換順序只是把同樣的數字換個位置排。</span></span>`;
      } else {
        const ia = SENTS[0].indexOf(word), ib = SENTS[1].indexOf(word);
        calc.innerHTML = `<span class="calc-row"><span class="k">位置編碼（d = 4）</span> PE(0) = ${vec(pe(0, 4), 3)}　PE(1) = ${vec(pe(1, 4), 3)}　PE(2) = ${vec(pe(2, 4), 3)}</span>` +
          `<span class="calc-row"><span class="k">句子 A 的「${word}」</span> ${vec(E[word], 1)} + PE(${ia}) = <span class="hl">${vec(R[0].X[ia], 3)}</span></span>` +
          `<span class="calc-row"><span class="k">句子 B 的「${word}」</span> ${vec(E[word], 1)} + PE(${ib}) = <span class="hl">${vec(R[1].X[ib], 3)}</span></span>`;
      }
      note.innerHTML = `為了好讀，這裡的クエリ・キー・バリュー都直接用輸入向量（不乘權重矩陣），分數一樣是內積 ÷ √d（見 <a href="#selfatt">Self-Attention</a>）。位置編碼是<b>加</b>到詞向量上，不是另外接一段，所以維數不變。它在整個模型裡的位置見 <a href="#transformer">Transformer</a>。`;
    }

    /* ---------- ② wave ---------- */
    function renderWave() {
      const P = 50, cw = 512 / d, ch = 5, L = 56, Tp = 26;
      const W = L + 512 + 14, H = Tp + P * ch + 34;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "位置編碼熱圖", style: `display:block;width:100%;min-width:${W * 0.9}px;max-width:${W * 1.25}px` });
      svg.append(s("text", { x: L, y: 14, "font-size": 12, "font-weight": 700, text: `每一列＝一個位置的 ${d} 維向量（藍＝正、紅＝負）` }));
      const pos = Atlas.color("--pos"), neg = Atlas.color("--neg"), pan = Atlas.color("--panel");
      for (let q = 0; q < P; q++) {
        const v = pe(q, d);
        for (let i = 0; i < d; i++) {
          const t = Math.min(1, Math.abs(v[i]));
          svg.append(s("rect", { x: L + i * cw, y: Tp + q * ch, width: cw + 0.3, height: ch + 0.3, fill: Atlas.rgbStr(Atlas.mix(pan, v[i] >= 0 ? pos : neg, t * 0.9)) }));
        }
      }
      svg.append(s("rect", { x: L - 2, y: Tp + p * ch - 1.5, width: 516, height: ch + 3, fill: "none", stroke: "var(--hot)", "stroke-width": 2.5 }));
      [0, 10, 20, 30, 40, 49].forEach((q) => svg.append(s("text", { x: L - 6, y: Tp + q * ch + 5, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: q })));
      svg.append(s("text", { x: 4, y: Tp + p * ch + 5, "font-size": 10.5, "font-weight": 700, class: "t-mono", fill: "var(--hot)", text: `▶${p}` }));
      svg.append(s("text", { x: 4, y: Tp + P * ch + 18, "font-size": 10.5, class: "t-muted", text: "位置↑" }));
      svg.append(s("text", { x: L, y: Tp + P * ch + 18, "font-size": 10.5, class: "t-muted", text: "第 1 維（變得快）" }));
      svg.append(s("text", { x: L + 512, y: Tp + P * ch + 18, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: `第 ${d} 維（變得慢）` }));
      fig.replaceChildren(svg);

      // similarity chart
      const me = pe(p, d), sim = Array.from({ length: P }, (_, q) => dot(me, pe(q, d)) / (d / 2));
      const W2 = 600, H2 = 210, l2 = 44, r2 = 14, t2 = 30, b2 = 34;
      const xs = (q) => l2 + (q / (P - 1)) * (W2 - l2 - r2), lo = 0, hi = 1;
      const ys = (v) => t2 + (hi - v) / (hi - lo) * (H2 - t2 - b2);
      const g = s("svg", { viewBox: `0 0 ${W2} ${H2}`, role: "img", "aria-label": "和其他位置的相似度", style: `display:block;width:100%;min-width:${W2 * 0.9}px;max-width:${W2 * 1.2}px` });
      g.append(s("text", { x: l2, y: 16, "font-size": 12, "font-weight": 700, text: `位置 ${p} 和每個位置的相似度（內積 ÷ (d/2)，自己＝1）` }));
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        g.append(s("line", { x1: l2, x2: W2 - r2, y1: ys(v), y2: ys(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
        g.append(s("text", { x: l2 - 6, y: ys(v) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: fmt(v, 2) }));
      });
      [0, 10, 20, 30, 40, 49].forEach((q) => g.append(s("text", { x: xs(q), y: H2 - b2 + 16, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-muted", text: q })));
      g.append(s("text", { x: W2 - r2, y: H2 - 4, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: "其他位置 →" }));
      g.append(s("line", { x1: xs(p), x2: xs(p), y1: t2, y2: H2 - b2, stroke: "var(--hot)", "stroke-dasharray": "4 3" }));
      g.append(s("polyline", { points: sim.map((v, q) => `${xs(q)},${ys(Math.max(lo, v))}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
      sim.forEach((v, q) => g.append(s("circle", { cx: xs(q), cy: ys(Math.max(lo, v)), r: q === p ? 4.5 : 2.2, fill: q === p ? "var(--hot)" : "var(--accent)" })));
      fig2.replaceChildren(g);

      const near = p < P - 1 ? p + 1 : p - 1, far = p + 25 < P ? p + 25 : p - 25;
      status.className = "status status--info";
      status.textContent = `位置 ${p} 和相鄰的位置 ${near} 相似度 ${fmt(sim[near], 2)}；和隔 25 個的位置 ${far} 只有 ${fmt(sim[far], 2)}`;
      const iLast = d - 2;
      const den = (i) => Math.pow(10000, i / d);
      calc.innerHTML =
        `<span class="calc-row"><span class="k">偶數維用 sin、奇數維用 cos：</span> PE(pos, 2i) = sin(pos ÷ 10000<sup>2i/d</sup>)，PE(pos, 2i+1) = cos(pos ÷ 10000<sup>2i/d</sup>)</span>` +
        `<span class="calc-row"><span class="k">第 1・2 維（i = 0）</span> sin(${p} ÷ 1) = ${fmt(Math.sin(p), 3)}，cos(${p} ÷ 1) = ${fmt(Math.cos(p), 3)}</span>` +
        `<span class="calc-row"><span class="k">第 3・4 維（i = 1）</span> sin(${p} ÷ ${fmt(den(2), 2)}) = ${fmt(Math.sin(p / den(2)), 3)}，cos(…) = ${fmt(Math.cos(p / den(2)), 3)}</span>` +
        `<span class="calc-row"><span class="k">最後兩維（i = ${iLast / 2}）</span> sin(${p} ÷ ${fmt(den(iLast), 0)}) = ${fmt(Math.sin(p / den(iLast)), 3)}，cos(…) = ${fmt(Math.cos(p / den(iLast)), 3)}</span>`;
      note.innerHTML = `分母 10000<sup>2i/d</sup> 隨維度變大，波長從約 6.3 個位置一路拉長到上萬個位置，所以前幾維像秒針、後幾維像時針。原論文的 d 是 512。這種固定公式不用訓練；也有模型（例如 BERT）把位置向量當參數直接學。`;
    }

    function render() {
      viewSeg.set(view); rowWhy.hidden = view !== "why"; rowWave.hidden = view !== "wave";
      if (view === "why") renderWhy(); else renderWave();
    }
    render();
    ctx.onTheme(render);
    return {};
  },
});
