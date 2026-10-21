/* 位置エンコーディング（シラバス §23）
   ① 為什麼需要：「狗 咬 人」與「人 咬 狗」做 Self-Attention（Q＝K＝V＝輸入，d＝4），
      不加位置時「狗」的輸出完全相同；加上 sin・cos 位置編碼後就不同。
   ② 長什麼樣：PE(pos, 2i) = sin(pos / 10000^(2i/d))、PE(pos, 2i+1) = cos(…) 的熱圖，
      以及某個位置和其他位置的相似度（內積 ÷ (d/2)）。
   文字在 locales/zh/posenc.js、locales/ja/posenc.js；程式內部用 dog / bite / man 代表三個詞，
   顯示的詞（words）與整句（sentA、sentB）在語言檔。 */
Atlas.define("posenc", {
  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const { t } = ctx;
    const pe = (p, d) => Array.from({ length: d }, (_, i) => { const w = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d); return i % 2 ? Math.cos(w) : Math.sin(w); });
    const E = { dog: [1, 0, 0.5, 0], bite: [0, 1, 0, 0.5], man: [0.5, 0, 1, 0] };
    const SENTS = [["dog", "bite", "man"], ["man", "bite", "dog"]];
    const WORD = Object.fromEntries(["dog", "bite", "man"].map((id, i) => [id, t("words").split(" ")[i]]));
    const SENT_TEXT = [t("sentA"), t("sentB")];
    const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
    function attend(sent, usePE) {
      const X = sent.map((w, i) => E[w].map((v, d) => v + (usePE ? pe(i, 4)[d] : 0)));
      const A = X.map((q) => { const sc = X.map((k) => dot(q, k) / 2), e = sc.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); });
      const Z = A.map((w) => [0, 1, 2, 3].map((d) => w.reduce((a, wj, j) => a + wj * X[j][d], 0)));
      return { X, A, Z };
    }

    let view = "why", usePE = false, word = "dog", d = 32, p = 5;
    const vec = (v, k = 2) => "[" + v.map((x) => fmt(x, k)).join(", ") + "]";
    const pct = (w) => Math.round(w * 100) + "%";

    const viewSeg = ui.seg({ value: view, label: t("viewLabel"), options: [{ value: "why", label: t("viewWhy") }, { value: "wave", label: t("viewWave") }], onChange: (v) => { view = v; render(); } });
    const peSeg = ui.seg({ value: usePE, label: t("peLabel"), options: [{ value: false, label: t("peOff") }, { value: true, label: t("peOn") }], onChange: (v) => { usePE = v; render(); } });
    const wordSeg = ui.seg({ value: word, label: t("wordLabel"), options: ["dog", "bite", "man"].map((w) => ({ value: w, label: WORD[w] })), onChange: (v) => { word = v; render(); } });
    const dSeg = ui.seg({ value: d, label: t("dLabel"), options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { d = v; render(); } });
    const pSlider = ui.slider({ id: "pe-pos", label: t("posLabel"), min: 0, max: 49, step: 1, value: p, format: (v) => String(v), onInput: (v) => { p = v; render(); } });

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
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("whyAria"), style: `display:block;width:100%;min-width:${W * 0.78}px;max-width:${W}px` });
      const tx = (i) => 150 + i * 100;
      R.forEach((r, si) => {
        const y0 = si * rowH + 8, base = y0 + 84;
        const st = SENTS[si], wi = st.indexOf(word);
        svg.append(s("rect", { x: 2, y: y0, width: W - 4, height: rowH - 8, rx: 10, fill: "var(--panel-2)", stroke: "var(--line)" }));
        svg.append(s("text", { x: 14, y: y0 + 24, "font-size": 13, "font-weight": 700, text: t("sentName", { s: si ? "B" : "A" }) }));
        svg.append(s("text", { x: 14, y: y0 + 44, "font-size": 13, class: "t-muted", text: SENT_TEXT[si] }));
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
          svg.append(s("text", { x: tx(j), y: base + 4, "text-anchor": "middle", "font-size": 15, "font-weight": sel ? 700 : 400, fill: sel ? "var(--panel)" : "var(--ink)", text: WORD[w] }));
          svg.append(s("text", { x: tx(j), y: base + 30, "text-anchor": "middle", "font-size": 11.5, class: "t-mono t-muted", text: t("posPct", { j, pct: pct(r.A[wi][j]) }) }));
        });
        // output vector
        const ox = 470;
        svg.append(s("text", { x: ox, y: y0 + 30, "font-size": 12, "font-weight": 700, text: t("outOf", { w: WORD[word] }) }));
        r.Z[wi].forEach((v, k) => {
          svg.append(s("rect", { x: ox + k * 74, y: y0 + 46, width: 68, height: 34, rx: 6, fill: Atlas.divColor(v, 2), stroke: "var(--line-strong)" }));
          svg.append(s("text", { x: ox + k * 74 + 34, y: y0 + 68, "text-anchor": "middle", "font-size": 13, class: "t-mono", fill: Math.abs(v) > 1.5 ? "var(--panel)" : "var(--ink)", text: fmt(v, 3) }));
          if (si === 0) svg.append(s("text", { x: ox + k * 74 + 34, y: y0 + 98, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: t("dimN", { k: k + 1 }) }));
        });
      });
      fig.replaceChildren(svg); fig2.replaceChildren();
      const za = R[0].Z[SENTS[0].indexOf(word)], zb = R[1].Z[SENTS[1].indexOf(word)];
      const diff = za.map((v, k) => Math.abs(v - zb[k])), mx = Math.max(...diff), k = diff.indexOf(mx);
      const same = mx < 1e-9;
      status.className = "status status--" + (same ? "bad" : "ok");
      status.textContent = same
        ? t(word === "bite" ? "statusSameVerb" : "statusSame", { w: WORD[word] })
        : t("statusDiff", { w: WORD[word], k: k + 1, a: fmt(za[k], 3), b: fmt(zb[k], 3) });
      if (!usePE) {
        calc.innerHTML = `<span class="calc-row"><span class="k">${t("calcEmbK")}</span> ${["dog", "bite", "man"].map((w) => `${WORD[w]} = ${vec(E[w], 1)}`).join("　")}</span>` +
          `<span class="calc-row"><span class="k">${t("calcEmbNote")}</span></span>`;
      } else {
        const ia = SENTS[0].indexOf(word), ib = SENTS[1].indexOf(word);
        calc.innerHTML = `<span class="calc-row"><span class="k">${t("calcPeK")}</span> PE(0) = ${vec(pe(0, 4), 3)}　PE(1) = ${vec(pe(1, 4), 3)}　PE(2) = ${vec(pe(2, 4), 3)}</span>` +
          `<span class="calc-row"><span class="k">${t("calcSentK", { s: "A", w: WORD[word] })}</span> ${vec(E[word], 1)} + PE(${ia}) = <span class="hl">${vec(R[0].X[ia], 3)}</span></span>` +
          `<span class="calc-row"><span class="k">${t("calcSentK", { s: "B", w: WORD[word] })}</span> ${vec(E[word], 1)} + PE(${ib}) = <span class="hl">${vec(R[1].X[ib], 3)}</span></span>`;
      }
      note.innerHTML = t("noteWhy");
    }

    /* ---------- ② wave ---------- */
    function renderWave() {
      const P = 50, cw = 512 / d, ch = 5, L = 56, Tp = 26;
      const W = L + 512 + 14, H = Tp + P * ch + 34;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("waveAria"), style: `display:block;width:100%;min-width:${W * 0.9}px;max-width:${W * 1.25}px` });
      svg.append(s("text", { x: L, y: 14, "font-size": 12, "font-weight": 700, text: t("waveTitle", { d }) }));
      const pos = Atlas.color("--pos"), neg = Atlas.color("--neg"), pan = Atlas.color("--panel");
      for (let q = 0; q < P; q++) {
        const v = pe(q, d);
        for (let i = 0; i < d; i++) {
          const a = Math.min(1, Math.abs(v[i]));
          svg.append(s("rect", { x: L + i * cw, y: Tp + q * ch, width: cw + 0.3, height: ch + 0.3, fill: Atlas.rgbStr(Atlas.mix(pan, v[i] >= 0 ? pos : neg, a * 0.9)) }));
        }
      }
      svg.append(s("rect", { x: L - 2, y: Tp + p * ch - 1.5, width: 516, height: ch + 3, fill: "none", stroke: "var(--hot)", "stroke-width": 2.5 }));
      [0, 10, 20, 30, 40, 49].forEach((q) => svg.append(s("text", { x: L - 6, y: Tp + q * ch + 5, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: q })));
      svg.append(s("text", { x: 4, y: Tp + p * ch + 5, "font-size": 10.5, "font-weight": 700, class: "t-mono", fill: "var(--hot)", text: `▶${p}` }));
      svg.append(s("text", { x: 4, y: Tp + P * ch + 18, "font-size": 10.5, class: "t-muted", text: t("posAxis") }));
      svg.append(s("text", { x: L, y: Tp + P * ch + 18, "font-size": 10.5, class: "t-muted", text: t("dimFast") }));
      svg.append(s("text", { x: L + 512, y: Tp + P * ch + 18, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("dimSlow", { d }) }));
      fig.replaceChildren(svg);

      // similarity chart
      const me = pe(p, d), sim = Array.from({ length: P }, (_, q) => dot(me, pe(q, d)) / (d / 2));
      const W2 = 600, H2 = 210, l2 = 44, r2 = 14, t2 = 30, b2 = 34;
      const xs = (q) => l2 + (q / (P - 1)) * (W2 - l2 - r2), lo = 0, hi = 1;
      const ys = (v) => t2 + (hi - v) / (hi - lo) * (H2 - t2 - b2);
      const g = s("svg", { viewBox: `0 0 ${W2} ${H2}`, role: "img", "aria-label": t("simAria"), style: `display:block;width:100%;min-width:${W2 * 0.9}px;max-width:${W2 * 1.2}px` });
      g.append(s("text", { x: l2, y: 16, "font-size": 12, "font-weight": 700, text: t("simTitle", { p }) }));
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        g.append(s("line", { x1: l2, x2: W2 - r2, y1: ys(v), y2: ys(v), stroke: v === 0 ? "var(--line-strong)" : "var(--line)" }));
        g.append(s("text", { x: l2 - 6, y: ys(v) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: fmt(v, 2) }));
      });
      [0, 10, 20, 30, 40, 49].forEach((q) => g.append(s("text", { x: xs(q), y: H2 - b2 + 16, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-muted", text: q })));
      g.append(s("text", { x: W2 - r2, y: H2 - 4, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("simAxis") }));
      g.append(s("line", { x1: xs(p), x2: xs(p), y1: t2, y2: H2 - b2, stroke: "var(--hot)", "stroke-dasharray": "4 3" }));
      g.append(s("polyline", { points: sim.map((v, q) => `${xs(q)},${ys(Math.max(lo, v))}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
      sim.forEach((v, q) => g.append(s("circle", { cx: xs(q), cy: ys(Math.max(lo, v)), r: q === p ? 4.5 : 2.2, fill: q === p ? "var(--hot)" : "var(--accent)" })));
      fig2.replaceChildren(g);

      const near = p < P - 1 ? p + 1 : p - 1, far = p + 25 < P ? p + 25 : p - 25;
      status.className = "status status--info";
      status.textContent = t("statusWave", { p, near, sn: fmt(sim[near], 2), far, sf: fmt(sim[far], 2) });
      const iLast = d - 2;
      const den = (i) => Math.pow(10000, i / d);
      calc.innerHTML =
        `<span class="calc-row"><span class="k">${t("calcRuleK")}</span> ${t("calcRule")}</span>` +
        `<span class="calc-row"><span class="k">${t("calcDim12K")}</span> sin(${p} ÷ 1) = ${fmt(Math.sin(p), 3)}　cos(${p} ÷ 1) = ${fmt(Math.cos(p), 3)}</span>` +
        `<span class="calc-row"><span class="k">${t("calcDim34K")}</span> sin(${p} ÷ ${fmt(den(2), 2)}) = ${fmt(Math.sin(p / den(2)), 3)}　cos(…) = ${fmt(Math.cos(p / den(2)), 3)}</span>` +
        `<span class="calc-row"><span class="k">${t("calcLastK", { i: iLast / 2 })}</span> sin(${p} ÷ ${fmt(den(iLast), 0)}) = ${fmt(Math.sin(p / den(iLast)), 3)}　cos(…) = ${fmt(Math.cos(p / den(iLast)), 3)}</span>`;
      note.innerHTML = t("noteWave");
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
