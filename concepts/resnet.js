/* スキップ結合（ResNet）（シラバス §21）
   和「勾配消失問題」頁同一個簡化網路：每層一個神經元、Sigmoid、權重 w、入力 1.0。
   上排：普通的網路   a_k = σ(w·a_{k−1})          → 梯度每層乘 w·σ′(z)
   下排：加上捷徑     a_k = a_{k−1} + σ(w·a_{k−1}) → 梯度每層乘 1 + w·σ′(z)
   從最後一層往前逐層傳回梯度，兩排同時比較。 */
Atlas.define("resnet", {
  lead: `<p>網路疊得越深，誤差逆伝播時梯度要連乘的次數越多，前面的層就學不動（勾配消失）。<span class="jp">スキップ結合</span>（跳躍連接，<span class="en">skip connection</span>）在層的旁邊多拉一條捷徑，把入力直接加到出力上，梯度也能沿這條捷徑原封不動地傳回去。</p>`,
  closing: `有了捷徑，梯度每過一層乘的是「1 + 層的斜率」，而不是只有「層的斜率」；就算層的斜率很小，那個 1 也讓梯度不會連乘到 0。層本身只需要學「在入力之上再加多少」，也就是<span class="jp">残差</span>（殘差，<span class="en">residual</span>）。ResNet 用這個結構把網路疊到 152 層。`,

  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const sigm = (x) => 1 / (1 + Math.exp(-x));
    const dsig = (z) => sigm(z) * (1 - sigm(z));
    const XIN = 1.0;
    let N = 8, w = 1.0, pos = 0;

    function run() {
      const P = { z: [], a: [XIN], fac: [], G: [] }, R = { z: [], a: [XIN], fac: [], G: [] };
      for (let k = 1; k <= N; k++) {
        P.z[k] = w * P.a[k - 1]; P.a[k] = sigm(P.z[k]);
        R.z[k] = w * R.a[k - 1]; R.a[k] = R.a[k - 1] + sigm(R.z[k]);
      }
      P.G[N] = R.G[N] = 1;
      for (let k = N - 1; k >= 1; k--) {
        P.fac[k] = w * dsig(P.z[k + 1]); P.G[k] = P.G[k + 1] * P.fac[k];
        R.fac[k] = 1 + w * dsig(R.z[k + 1]); R.G[k] = R.G[k + 1] * R.fac[k];
      }
      return { P, R };
    }

    /* ---------- controls ---------- */
    const nSlider = ui.slider({ id: "rs-n", label: "層數", min: 2, max: 12, step: 1, value: N, format: (v) => v + " 層", onInput: (v) => { N = v; if (pos) pos = N; render(); } });
    const wSlider = ui.slider({ id: "rs-w", label: "每層的權重 w", min: 0.5, max: 2.5, step: 0.05, value: w, format: (v) => v.toFixed(2), onInput: (v) => { w = v; if (pos) pos = N; render(); } });
    const player = ui.player(ctx, () => { if (pos >= N) return false; pos++; render(); return pos < N; }, 650);
    const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn("往前傳一層 ▶", () => { player.pause(); if (pos < N) { pos++; render(); } }, { primary: true });
    const allBtn = ui.btn("全部算完", () => { player.pause(); pos = N; render(); });
    const resetBtn = ui.btn("重設", () => { player.pause(); pos = 0; render(); });

    const status = h("div", { class: "status" });
    const svgWrap = h("div", { class: "scroll-x" });
    const barWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc", style: { whiteSpace: "normal" } });
    const note = h("p", { class: "note" });
    bench.append(
      h("div", { class: "bench__row" }, nSlider.el, wSlider.el),
      status, svgWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, resetBtn),
      calc, barWrap, note,
    );

    const fmtG = (v) => {
      if (v === 0) return "0";
      if (v >= 1000) return v.toExponential(1).replace("e+", "e");
      if (v >= 10) return v.toFixed(1);
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
      const { P, R } = run();
      const COLW = 74, LEFT = 70, RIGHT = 70, VW = LEFT + COLW * N + RIGHT, VH = 336;
      const cx = (k) => LEFT + COLW * (k - 1) + COLW / 2;
      const known = (k) => pos > 0 && k > N - pos;
      const cur = pos > 0 ? N - pos + 1 : -1;
      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "沒有捷徑與有捷徑的網路，各層的輸出與傳回的梯度", style: `display:block;width:100%;min-width:${Math.min(VW, 600)}px;max-width:${VW}px` });

      function lane(M, y0, residual) {
        const yG = y0 + 32, yN = y0 + (residual ? 108 : 92);
        svg.append(s("text", { x: 8, y: y0 + 4, "font-size": 13, "font-weight": 700, fill: residual ? "var(--accent)" : "var(--ink)",
          text: residual ? "加上スキップ結合（弧線＝捷徑：出力 = 層的結果 + 入力）" : "普通的網路（沒有捷徑）" }));
        if (pos > 0) svg.append(s("text", { x: 8, y: yG + 4, "font-size": 12, "font-weight": 700, fill: "var(--hot)", text: "梯度" }));
        svg.append(s("text", { x: 8, y: yN + 4, "font-size": 12, "font-weight": 700, text: "輸入" }));
        svg.append(s("text", { x: 8, y: yN + 20, "font-size": 11, class: "t-mono t-muted", text: "x = " + fmt(XIN, 1) }));
        for (let k = 1; k <= N; k++) {
          const x = cx(k), xin = k === 1 ? 46 : cx(k - 1) + (residual ? 37 : 22); // 有捷徑時從前一個 ⊕ 之後出發
          svg.append(arrow(xin, yN, x - 24, yN, "var(--accent)", 1.6));
          if (residual) {
            // 捷徑：從入力箭頭的起點，越過這一層，接回出力
            const x1 = xin + 3, x2 = x + 31;
            svg.append(s("path", { d: `M${x1},${yN - 4} C${x1 + 4},${yN - 40} ${x2 - 4},${yN - 40} ${x2},${yN - 4}`, fill: "none", stroke: "var(--accent)", "stroke-width": 1.8, "stroke-dasharray": "5 3" }));
            svg.append(s("circle", { cx: x2, cy: yN, r: 6, fill: "var(--panel)", stroke: "var(--accent)", "stroke-width": 1.5 }));
            svg.append(s("text", { x: x2, y: yN + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: "var(--accent)", text: "+" }));
          }
          const av = M.a[k];
          svg.append(s("rect", { x: x - 22, y: yN - 20, width: 44, height: 40, rx: 7, fill: "var(--accent-soft)", stroke: k === cur ? "var(--hot)" : "var(--accent)", "stroke-width": k === cur ? 2.5 : 1 }));
          svg.append(s("text", { x, y: yN + 5, "text-anchor": "middle", "font-size": 11.5, class: "t-mono", text: fmt(av, 2) }));
          svg.append(s("text", { x, y: yN + 36, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: `第 ${k} 層` }));
          if (known(k)) {
            const g = M.G[k], alpha = Math.max(0.12, Math.min(1, Math.sqrt(Math.min(g, 1))));
            svg.append(s("rect", { x: x - 27, y: yG - 12, width: 54, height: 22, rx: 11, fill: "var(--hot)", "fill-opacity": g > 1 ? 1 : alpha, stroke: "var(--hot)" }));
            svg.append(s("text", { x, y: yG + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 600, class: "t-mono", fill: g > 1 || alpha > 0.55 ? "var(--panel)" : "var(--ink)", text: fmtG(g) }));
            if (k < N) svg.append(s("text", { x: x + COLW / 2, y: yG + 24, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-halo", fill: "var(--hot)", text: "×" + fmt(M.fac[k], 2) }));
          }
        }
        const xe = cx(N) + (residual ? 37 : 24);
        svg.append(arrow(xe, yN, xe + 20, yN, "var(--accent)", 1.6));
        svg.append(s("text", { x: xe + 24, y: yN + 4, "font-size": 12, text: "輸出" }));
      }
      lane(P, 16, false);
      svg.append(s("line", { x1: 4, x2: VW - 4, y1: 158, y2: 158, stroke: "var(--line)" }));
      lane(R, 186, true);
      svgWrap.replaceChildren(svg);

      /* status & calc */
      const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      if (pos === 0) {
        set("info", "兩個網路都已算好正向（藍框＝每層的輸出）。按「往前傳一層」，把梯度從最後一層同時傳回去");
        calc.innerHTML = row("普通", `第 k 層的輸出 = σ(w × 上一層)　例：第 1 層 σ(${fmt(w, 2)} × ${fmt(XIN, 1)}) = ${fmt(P.a[1], 3)}`) +
          row("有捷徑", `第 k 層的輸出 = 上一層 + σ(w × 上一層)　例：第 1 層 ${fmt(XIN, 1)} + ${fmt(sigm(R.z[1]), 3)} = ${fmt(R.a[1], 3)}`);
      } else if (pos === 1) {
        set("warn", `從最後一層（第 ${N} 層）開始，兩邊的梯度都設為 1`);
        calc.innerHTML = row("起點", `把最後一層收到的梯度當作 1，看它傳回每一層時還剩多少`);
      } else {
        const k = N - pos + 1;
        const zp = P.z[k + 1], zr = R.z[k + 1];
        calc.innerHTML =
          row(`普通：第 ${k + 1} → ${k} 層乘`, `w × σ′(z) = ${fmt(w, 2)} × σ′(${fmt(zp, 2)}) = ${fmt(w, 2)} × ${fmt(dsig(zp), 3)} = <span class="hl">${fmt(P.fac[k], 3)}</span>　梯度 ${fmtG(P.G[k])}`) +
          row(`有捷徑：乘`, `1（捷徑） + w × σ′(z) = 1 + ${fmt(w, 2)} × ${fmt(dsig(zr), 3)} = <span class="hl">${fmt(R.fac[k], 3)}</span>　梯度 ${fmtG(R.G[k])}`);
        if (pos < N) set("warn", `傳回第 ${k} 層：普通 ${fmtG(P.G[k])}，有捷徑 ${fmtG(R.G[k])}`);
      }
      if (pos === N && N > 1) {
        const gp = P.G[1], gr = R.G[1];
        set(gr > 0.3 && gr < 10 ? "ok" : "warn", `第 1 層收到的梯度：普通的網路 ${fmtG(gp)}${gp < 0.01 ? "（勾配消失）" : ""}；有捷徑 ${fmtG(gr)}${gr > 0.3 && gr < 10 ? "，保住了" : ""}`);
      }

      /* grouped bar chart (log scale) */
      barWrap.replaceChildren();
      if (pos > 0) {
        const vals = [];
        for (let k = 1; k <= N; k++) if (known(k)) vals.push(P.G[k], R.G[k]);
        let lo = Math.floor(Math.log10(Math.min(...vals, 1))), hi = Math.ceil(Math.log10(Math.max(...vals, 1)));
        lo = Math.min(lo, -1); hi = Math.max(hi, 1);
        const BH = 210, top = 44, base = BH - 30;
        const bs = s("svg", { viewBox: `0 0 ${VW} ${BH}`, role: "img", "aria-label": "各層收到的梯度：普通 vs 有捷徑（對數刻度）", style: `display:block;width:100%;min-width:${Math.min(VW, 600)}px;max-width:${VW}px` });
        const py = (v) => base - ((Math.log10(v) - lo) / (hi - lo)) * (base - top);
        bs.append(s("text", { x: 8, y: 14, "font-size": 12, "font-weight": 700, text: "各層收到的梯度（最後一層 = 1，對數刻度：每格差 10 倍）" }));
        bs.append(s("rect", { x: 8, y: 22, width: 12, height: 12, rx: 2, fill: "var(--hot)" }));
        bs.append(s("text", { x: 24, y: 32, "font-size": 11.5, text: "普通" }));
        bs.append(s("rect", { x: 66, y: 22, width: 12, height: 12, rx: 2, fill: "var(--accent)" }));
        bs.append(s("text", { x: 82, y: 32, "font-size": 11.5, text: "有捷徑" }));
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
          [[P.G[k], -14, "var(--hot)"], [R.G[k], 1, "var(--accent)"]].forEach(([v, off, col]) => {
            const y1 = Math.min(py(v), py(10 ** lo)), y0 = py(10 ** lo);
            bs.append(s("rect", { x: x + off, y: y1, width: 13, height: Math.max(1, y0 - y1), rx: 2, fill: col }));
          });
        }
        bs.append(s("text", { x: cx(N) + 34, y: base + 16, "font-size": 11, class: "t-muted", text: "← 層" }));
        barWrap.append(bs);
      }

      prevBtn.disabled = pos === 0;
      nextBtn.disabled = allBtn.disabled = pos >= N;
      note.innerHTML = `σ′ 是 Sigmoid 的斜率 σ(z)(1 − σ(z))，最大只有 0.25，所以 w = 1 時普通的網路每層至少縮成四分之一。示範網路每層只有一個神經元；真正的 ResNet 每個捷徑跨過兩三層卷積（中間有バッチ正規化和 ReLU），但「梯度多一條乘 1 的路」的道理相同。在真正的 ResNet 裡，殘差那一路的權重若都學成 0，區塊就等於把入力原樣傳下去，所以多疊的層至少不容易讓結果變差。同一個網路沒有捷徑時的完整情形見 <a href="#vanish">勾配消失問題</a>。`;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
