/* 感受野與空洞卷積／受容野と Dilated Convolution（シラバス §18）
   ① 一維的多層卷積：33 格輸入、每層 3 個權重，點最上層的一格，
      往下追出它用到哪些格（感受野）。一般卷積間隔都是 1；空洞卷積間隔 1, 2, 4, 8。
   ② 二維的 3×3 卷積核：間隔 r 時覆蓋 (2r+1)×(2r+1)，權重仍是 9 個。
   文字在 locales/zh/dilated.js、locales/ja/dilated.js。 */
Atlas.define("dilated", {
  mount(bench, ctx) {
    const { h, s, ui } = Atlas;
    const { t } = ctx;
    const W = 33, K = 3;
    const RATES = { normal: [1, 1, 1, 1], dilated: [1, 2, 4, 8] };
    let view = "tower", mode = "dilated", L = 4, sel = 16, depth = 99, r2 = 2;

    function trace(md, nl, at) {
      const d = RATES[md];
      const sets = []; sets[nl] = new Set([at]);
      const edges = [];
      for (let l = nl; l >= 1; l--) {
        const below = new Set();
        for (const p of sets[l]) for (const o of [-1, 0, 1]) {
          const q = p + o * d[l - 1];
          if (q < 0 || q >= W) continue; // 外側是補的 0（填充），不算進感受野
          below.add(q); edges.push({ l, p, q });
        }
        sets[l - 1] = below;
      }
      return { sets, edges };
    }
    const rfFormula = (md, nl) => 1 + (K - 1) * RATES[md].slice(0, nl).reduce((a, b) => a + b, 0);

    /* ---------- controls ---------- */
    const viewSeg = ui.seg({ label: t("viewLabel"), value: view, options: [{ value: "tower", label: t("viewTower") }, { value: "grid", label: t("viewGrid") }], onChange: (v) => { player.pause(); view = v; render(); } });
    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode, options: [{ value: "normal", label: t("modeNormal") }, { value: "dilated", label: t("modeDilated") }], onChange: (v) => { player.pause(); mode = v; depth = 99; render(); } });
    const lSlider = ui.slider({ id: "dl-l", label: t("layersLabel"), min: 1, max: 4, step: 1, value: L, format: (v) => t("layersFmt", { n: v }), onInput: (v) => { player.pause(); L = v; depth = 99; render(); } });
    const rSeg = ui.seg({ label: t("rateLabel"), value: r2, options: [1, 2, 3, 4].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { r2 = v; render(); } });

    const player = ui.player(ctx, () => { if (depth >= L) return false; depth++; render(); return depth < L; }, 750);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); depth = Math.max(0, Math.min(depth, L) - 1); render(); });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); depth = Math.min(L, (depth > L ? 0 : depth) + 1); render(); }, { primary: true });
    const allBtn = ui.btn(t("btnAll"), () => { player.pause(); depth = 99; render(); });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); depth = 0; render(); });

    const rowTower = h("div", { class: "bench__row" }, modeSeg.el, lSlider.el);
    const rowGrid = h("div", { class: "bench__row" }, rSeg.el);
    const rowStep = h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, resetBtn);
    const status = h("div", { class: "status" });
    const fig = h("div", { class: "scroll-x" });
    const hint = h("p", { class: "hint" });
    const calc = h("div", { class: "calc", style: { whiteSpace: "normal" } });
    const note = h("p", { class: "note" });
    bench.append(h("div", { class: "bench__row" }, viewSeg.el), rowTower, rowGrid, status, fig, hint, rowStep, calc, note);

    const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
    const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };

    /* ---------- ① tower ---------- */
    function renderTower() {
      const dShow = Math.min(depth, L);
      const T = trace(mode, L, sel);
      const COL = 17, LEFT = 104, RH = 58, TOP = 22;
      const VW = LEFT + COL * W + 14, VH = TOP + RH * L + 44;
      const x = (i) => LEFT + COL * i + COL / 2;
      const y = (l) => TOP + RH * (L - l) + 10;
      const shownLayer = (l) => l >= L - dShow; // 已追到的層
      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": t("towerAria"), style: `display:block;width:100%;min-width:${Math.round(VW * 0.86)}px;max-width:${VW}px` });

      // 感受野的底色帶
      const S0 = [...T.sets[0]].sort((a, b) => a - b);
      if (dShow >= L) svg.append(s("rect", { x: x(S0[0]) - COL / 2 + 1, y: y(0) - 12, width: COL * (S0[S0.length - 1] - S0[0] + 1) - 2, height: 24, rx: 6, fill: "var(--hot-soft)" }));
      // edges
      for (const e of T.edges) {
        if (!shownLayer(e.l - 1) || !shownLayer(e.l)) continue;
        svg.append(s("line", { x1: x(e.p), y1: y(e.l) + 5, x2: x(e.q), y2: y(e.l - 1) - 5, stroke: "var(--hot)", "stroke-width": 1.4, "stroke-opacity": 0.7 }));
      }
      for (let l = 0; l <= L; l++) {
        const lab = l === 0 ? t("layerIn") : t("layerK", { l });
        svg.append(s("text", { x: 8, y: y(l) + 1, "font-size": 12.5, "font-weight": 700, text: lab }));
        if (l > 0) svg.append(s("text", { x: 8, y: y(l) + 17, "font-size": 11, class: "t-mono t-muted", text: t("rateK", { d: RATES[mode][l - 1] }) }));
        for (let i = 0; i < W; i++) {
          const on = shownLayer(l) && T.sets[l] && T.sets[l].has(i);
          const isSel = l === L && i === sel;
          svg.append(s("circle", { cx: x(i), cy: y(l), r: isSel ? 7 : on ? 5.2 : 3.2,
            fill: on ? "var(--hot)" : "var(--panel)", stroke: on ? "var(--hot)" : "var(--line-strong)", "stroke-width": isSel ? 2.5 : 1 }));
        }
      }
      // clickable top row
      for (let i = 0; i < W; i++) {
        const hit = s("rect", { x: x(i) - COL / 2, y: y(L) - 14, width: COL, height: 28, fill: "transparent", style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": t("pickAria", { l: L, i: i + 1 }) });
        const pick = () => { player.pause(); sel = i; depth = 99; render(); };
        hit.addEventListener("click", pick);
        hit.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
        svg.append(hit);
      }
      if (dShow >= L) svg.append(s("text", { x: x(S0[0]) + (S0[S0.length - 1] - S0[0]) * COL / 2, y: y(0) + 30, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: t("rfLabel", { n: S0.length }) }));
      fig.replaceChildren(svg);
      // 手機上圖會橫向捲動：把選中的那一格捲到中間
      requestAnimationFrame(() => { if (fig.scrollWidth > fig.clientWidth) fig.scrollLeft = (x(sel) / VW) * fig.scrollWidth - fig.clientWidth / 2; });

      const rf = T.sets[0].size, ideal = rfFormula(mode, L);
      const other = mode === "dilated" ? "normal" : "dilated";
      const rfO = rfFormula(other, L);
      const modeName = (md) => t(md === "dilated" ? "modeNameDilated" : "modeNameNormal");
      if (dShow < L) set("info", dShow === 0 ? t("statusStart") : t("statusTraced", { l: L - dShow, n: T.sets[L - dShow].size }));
      else set(mode === "dilated" && L >= 2 ? "ok" : "warn",
        t("statusDone", { mode: modeName(mode), layers: L, rf, clip: rf < ideal ? t("statusClip", { ideal }) : "", other: modeName(other), rfO }));
      hint.textContent = t("hintTower");
      const d = RATES[mode].slice(0, L);
      calc.innerHTML =
        row(t("calcRfK"), t("calcRf", { ds: d.join(" + "), sum: d.reduce((a, b) => a + b, 0), rf: ideal })) +
        row(t("calcWK"), t("calcW", { layers: L, w: 3 * L })) +
        row(t("calcNeedK"), t("calcNeed", { rf: ideal, n: (ideal - 1) / 2 }));
      note.innerHTML = t("noteTower");
    }

    /* ---------- ② 2D kernel ---------- */
    function renderGrid() {
      const G = 9, C = 30, PAD = 6, cIdx = 4;
      const VW = G * C + PAD * 2, VH = VW;
      const svg = s("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": t("gridAria", { r: r2 }), style: `display:block;width:100%;max-width:${VW + 40}px;margin-inline:auto` });
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
      set("info", t("statusGrid", { r: r2, span, area: span * span }));
      hint.textContent = "";
      calc.innerHTML = row(t("calcCoverK"), t("calcCover", { r: r2, span, area: span * span })) +
        row(t("calcWK"), t("calcW2", { span })) +
        row(t("calcPlainK"), t("calcPlain"));
      note.innerHTML = t("noteGrid");
    }

    function render() {
      const isTower = view === "tower";
      rowTower.hidden = rowStep.hidden = !isTower; rowGrid.hidden = isTower;
      if (isTower) {
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
