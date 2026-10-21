/* 受容野と Dilated Convolution（シラバス §18）
   ① 一維的多層卷積：33 格入力、每層 3 個權重，點最上層的一格，
      往下追出它用到哪些格（受容野）。一般卷積間隔都是 1；Dilated 間隔 1, 2, 4, 8。
   ② 二維的 3×3 カーネル：間隔 r 時覆蓋 (2r+1)×(2r+1)，權重仍是 9 個。 */
Atlas.define("dilated", {
  lead: `<p>要判斷一個像素屬於什麼（例如這一點是路面還是車），得看它周圍夠大的範圍。一個出力看得到的入力範圍叫<span class="jp">受容野</span>（感受野，<span class="en">receptive field</span>）。一般的 3×3 卷積每疊一層只多看 2 格；<span class="jp">Dilated Convolution</span>（擴張卷積，<span class="en">dilated convolution</span>）讓濾波器隔著空隙取樣，不增加權重就看得更遠。</p>`,
  closing: `每層的間隔加倍（1, 2, 4, 8…），受容野就隨層數指數成長；權重數量和一般卷積完全一樣，特徴マップ也不必像プーリング那樣縮小、丟掉細節。所以它常用在需要逐像素判斷的セマンティックセグメンテーション（DeepLab）和要看很長一段聲音的 WaveNet。`,

  mount(bench, ctx) {
    const { h, s, ui } = Atlas;
    const W = 33, K = 3;
    const RATES = { normal: [1, 1, 1, 1], dilated: [1, 2, 4, 8] };
    let view = "tower", mode = "dilated", L = 4, sel = 16, depth = 99, r2 = 2;

    function trace(md, nl, at) {
      const d = RATES[md];
      const sets = []; sets[nl] = new Set([at]);
      const edges = [];
      for (let l = nl; l >= 1; l--) {
        const below = new Set();
        for (const p of sets[l]) for (const t of [-1, 0, 1]) {
          const q = p + t * d[l - 1];
          if (q < 0 || q >= W) continue; // 外側是補的 0（パディング），不算進受容野
          below.add(q); edges.push({ l, p, q });
        }
        sets[l - 1] = below;
      }
      return { sets, edges };
    }
    const rfFormula = (md, nl) => 1 + (K - 1) * RATES[md].slice(0, nl).reduce((a, b) => a + b, 0);

    /* ---------- controls ---------- */
    const viewSeg = ui.seg({ label: "看哪一部分", value: view, options: [{ value: "tower", label: "① 疊幾層看得多遠（一維）" }, { value: "grid", label: "② 2D 的 3×3 長什麼樣" }], onChange: (v) => { player.pause(); view = v; render(); } });
    const modeSeg = ui.seg({ label: "卷積的種類", value: mode, options: [{ value: "normal", label: "一般の畳み込み（間隔 1）" }, { value: "dilated", label: "Dilated（間隔 1・2・4・8）" }], onChange: (v) => { player.pause(); mode = v; depth = 99; render(); } });
    const lSlider = ui.slider({ id: "dl-l", label: "層數", min: 1, max: 4, step: 1, value: L, format: (v) => v + " 層", onInput: (v) => { player.pause(); L = v; depth = 99; render(); } });
    const rSeg = ui.seg({ label: "間隔 r（dilation rate）", value: r2, options: [1, 2, 3, 4].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { r2 = v; render(); } });

    const player = ui.player(ctx, () => { if (depth >= L) return false; depth++; render(); return depth < L; }, 750);
    const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); depth = Math.max(0, Math.min(depth, L) - 1); render(); });
    const nextBtn = ui.btn("往下追一層 ▶", () => { player.pause(); depth = Math.min(L, (depth > L ? 0 : depth) + 1); render(); }, { primary: true });
    const allBtn = ui.btn("全部顯示", () => { player.pause(); depth = 99; render(); });
    const resetBtn = ui.btn("重設", () => { player.pause(); depth = 0; render(); });

    const rowTower = h("div", { class: "bench__row" }, modeSeg.el, lSlider.el);
    const rowGrid = h("div", { class: "bench__row" }, rSeg.el);
    const rowStep = h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, resetBtn);
    const status = h("div", { class: "status" });
    const fig = h("div", { class: "scroll-x" });
    const hint = h("p", { class: "hint" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });
    bench.append(h("div", { class: "bench__row" }, viewSeg.el), rowTower, rowGrid, status, fig, hint, rowStep, calc, note);

    const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
    const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };

    /* ---------- ① tower ---------- */
    function renderTower() {
      const dShow = Math.min(depth, L);
      const T = trace(mode, L, sel);
      const COL = 17, LEFT = 104, RH = 58, TOP = 22;
      const VW = LEFT + COL * W + 14, VH = TOP + RH * L + 44;
      const x = (i) => LEFT + COL * i + COL / 2;
      const y = (l) => TOP + RH * (L - l) + 10;
      const shownLayer = (l) => l >= L - dShow; // 已追到的層
      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": "多層卷積中，一個出力用到的入力範圍", style: `display:block;width:100%;min-width:${Math.round(VW * 0.86)}px;max-width:${VW}px` });

      // 受容野的底色帶
      const S0 = [...T.sets[0]].sort((a, b) => a - b);
      if (dShow >= L) svg.append(s("rect", { x: x(S0[0]) - COL / 2 + 1, y: y(0) - 12, width: COL * (S0[S0.length - 1] - S0[0] + 1) - 2, height: 24, rx: 6, fill: "var(--hot-soft)" }));
      // edges
      for (const e of T.edges) {
        if (!shownLayer(e.l - 1) || !shownLayer(e.l)) continue;
        svg.append(s("line", { x1: x(e.p), y1: y(e.l) + 5, x2: x(e.q), y2: y(e.l - 1) - 5, stroke: "var(--hot)", "stroke-width": 1.4, "stroke-opacity": 0.7 }));
      }
      for (let l = 0; l <= L; l++) {
        const lab = l === 0 ? "入力" : `第 ${l} 層`;
        svg.append(s("text", { x: 8, y: y(l) + 1, "font-size": 12.5, "font-weight": 700, text: lab }));
        if (l > 0) svg.append(s("text", { x: 8, y: y(l) + 17, "font-size": 11, class: "t-mono t-muted", text: `間隔 ${RATES[mode][l - 1]}` }));
        for (let i = 0; i < W; i++) {
          const on = shownLayer(l) && T.sets[l] && T.sets[l].has(i);
          const isSel = l === L && i === sel;
          svg.append(s("circle", { cx: x(i), cy: y(l), r: isSel ? 7 : on ? 5.2 : 3.2,
            fill: on ? "var(--hot)" : "var(--panel)", stroke: on ? "var(--hot)" : "var(--line-strong)", "stroke-width": isSel ? 2.5 : 1 }));
        }
      }
      // clickable top row
      for (let i = 0; i < W; i++) {
        const hit = s("rect", { x: x(i) - COL / 2, y: y(L) - 14, width: COL, height: 28, fill: "transparent", style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": `選第 ${L} 層第 ${i + 1} 格` });
        const pick = () => { player.pause(); sel = i; depth = 99; render(); };
        hit.addEventListener("click", pick);
        hit.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
        svg.append(hit);
      }
      if (dShow >= L) svg.append(s("text", { x: x(S0[0]) + (S0[S0.length - 1] - S0[0]) * COL / 2, y: y(0) + 30, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: `受容野 ${S0.length} 格` }));
      fig.replaceChildren(svg);
      // 手機上圖會橫向捲動：把選中的那一格捲到中間
      requestAnimationFrame(() => { if (fig.scrollWidth > fig.clientWidth) fig.scrollLeft = (x(sel) / VW) * fig.scrollWidth - fig.clientWidth / 2; });

      const rf = T.sets[0].size, ideal = rfFormula(mode, L);
      const other = mode === "dilated" ? "normal" : "dilated";
      const rfO = rfFormula(other, L);
      if (dShow < L) set("info", dShow === 0 ? `橘點是選中的出力。按「往下追一層」，看它用到下一層的哪 3 格` : `已追到第 ${L - dShow} ${L - dShow === 0 ? "（入力）" : "層"}：用到 ${T.sets[L - dShow].size} 格`);
      else set(mode === "dilated" && L >= 2 ? "ok" : "warn",
        `${mode === "dilated" ? "Dilated" : "一般卷積"}疊 ${L} 層：一格出力看得到入力 ${rf} 格${rf < ideal ? `（靠邊被截掉，完整是 ${ideal} 格）` : ""}；${mode === "dilated" ? "一般卷積" : "Dilated"}同樣 ${L} 層是 ${rfO} 格`);
      hint.textContent = "點最上層任一格，換一個出力來追。";
      const d = RATES[mode].slice(0, L);
      calc.innerHTML =
        row("受容野", `1 + (カーネル 3 − 1) × (${d.join(" + ")}) = 1 + 2 × ${d.reduce((a, b) => a + b, 0)} = <span class="hl">${ideal}</span> 格`) +
        row("權重", `每層 3 個 × ${L} 層 = <b>${3 * L}</b> 個（兩種卷積一樣多）`) +
        row("若要一般卷積看到同樣範圍", `需要疊 (${ideal} − 1) ÷ 2 = <b>${(ideal - 1) / 2}</b> 層`);
      note.innerHTML = `一般卷積的受容野每層只多 2 格（線性成長）；間隔每層加倍時，受容野是 2<sup>層數+1</sup> − 1（指數成長）。另一個擴大受容野的方法是プーリング，但它會把特徴マップ縮小、丟掉位置細節，逐像素分類時不划算。示範用 1 維、每層 3 個權重；入力外側視為補 0（パディング），所以每層的長度不變。間隔取樣的規則見 ② 的 2D 圖。`;
    }

    /* ---------- ② 2D kernel ---------- */
    function renderGrid() {
      const G = 9, C = 30, PAD = 6, cIdx = 4;
      const VW = G * C + PAD * 2, VH = VW;
      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": `間隔 ${r2} 的 3×3 カーネル取樣的位置`, style: `display:block;width:100%;max-width:${VW + 40}px;margin-inline:auto` });
      const span = 2 * r2 + 1, lo = cIdx - r2;
      svg.append(s("rect", { x: PAD + lo * C - 3, y: PAD + lo * C - 3, width: span * C + 6, height: span * C + 6, rx: 6, fill: "var(--hot-soft)", stroke: "var(--hot)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
      let n = 0;
      for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
        const dr = r - cIdx, dc = c - cIdx;
        const tap = Math.abs(dr) <= r2 && Math.abs(dc) <= r2 && dr % r2 === 0 && dc % r2 === 0;
        const inside = Math.abs(dr) <= r2 && Math.abs(dc) <= r2;
        if (tap) n++;
        svg.append(s("rect", { x: PAD + c * C + 2, y: PAD + r * C + 2, width: C - 4, height: C - 4, rx: 4,
          fill: tap ? "var(--hot)" : inside ? "var(--hot-soft)" : "var(--panel)", stroke: tap ? "var(--hot)" : inside ? "var(--hot)" : "var(--line)", "stroke-opacity": tap ? 1 : inside ? 0.35 : 1 }));
        if (tap) svg.append(s("text", { x: PAD + c * C + C / 2, y: PAD + r * C + C / 2 + 4, "text-anchor": "middle", "font-size": 11, class: "t-mono", fill: "var(--panel)", text: "w" + n }));
      }
      fig.replaceChildren(svg);
      set("info", `間隔 ${r2}：9 個權重散開取樣，覆蓋 ${span}×${span} = ${span * span} 格的範圍`);
      hint.textContent = "";
      calc.innerHTML = row("覆蓋範圍", `邊長 = 2 × 間隔 ${r2} + 1 = ${span}，面積 ${span}×${span} = <span class="hl">${span * span}</span> 格`) +
        row("權重", `仍是 3×3 = <b>9</b> 個；中間的空格不參與計算（等於一個 ${span}×${span}、空格處固定為 0 的濾波器，但那些 0 不用學）`) +
        row("一般 3×3", `就是間隔 1 的特例：覆蓋 3×3 = 9 格`);
      note.innerHTML = `Dilated Convolution 又叫 <span class="jp">Atrous Convolution</span>，atrous 來自法文 à trous（帶著洞），指濾波器中間的空隙。兩個名字指同一種卷積，シラバス兩個都有列出。`;
    }

    function render() {
      const t = view === "tower";
      rowTower.hidden = rowStep.hidden = !t; rowGrid.hidden = t;
      if (t) {
        renderTower();
        const dShow = Math.min(depth, L);
        prevBtn.disabled = dShow === 0;
        nextBtn.disabled = allBtn.disabled = dShow >= L;
      } else renderGrid();
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
