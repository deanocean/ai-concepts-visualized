/* 卷積／畳み込み（シラバス §18）
   7×7 輸入 × 3×3 濾波器，可調步幅／填充，逐格計算特徵圖。
   文字在 locales/zh/conv.js、locales/ja/conv.js。 */
Atlas.define("conv", {
  mount(bench, ctx) {
    const { h, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const N = 7, K = 3;

    const IMAGES = {
      bar:    { label: t("imgBar"), px: (r, c) => c === 2 || c === 3 },
      ring:   { label: t("imgRing"), px: (r, c) => r >= 1 && r <= 5 && c >= 1 && c <= 5 && (r === 1 || r === 5 || c === 1 || c === 5) },
      diag:   { label: t("imgDiag"), px: (r, c) => r === c },
      ell:    { label: t("imgEll"), px: (r, c) => (c === 1 && r >= 1 && r <= 5) || (r === 5 && c >= 1 && c <= 5) },
      blank:  { label: t("imgBlank"), px: () => false },
    };
    const FILTERS = {
      vert:   { label: t("fVert"), w: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      horz:   { label: t("fHorz"), w: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      diag:   { label: t("fDiag"), w: [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]] },
      blur:   { label: t("fBlur"), w: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map(() => 1 / 9)) },
      custom: { label: t("fCustom"), w: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
    };

    let imgKey = "bar", fKey = "vert", S = 1, P = 0;
    let img = makeImg(imgKey), filt = FILTERS[fKey].w.map((r) => r.slice());
    let done = 0; // 已計算的格數（列優先）

    function makeImg(k) { return Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (IMAGES[k].px(r, c) ? 1 : 0))); }
    const outN = () => Math.floor((N + 2 * P - K) / S) + 1;
    const padded = (r, c) => { const R = r - P, C = c - P; return R < 0 || C < 0 || R >= N || C >= N ? null : img[R][C]; };
    const isInt = () => filt.every((r) => r.every((v) => Number.isInteger(v)));
    function cellAt(idx) {
      const n = outN(), or = Math.floor(idx / n), oc = idx % n;
      const r0 = or * S, c0 = oc * S;
      const terms = [];
      let sum = 0;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const a = padded(r0 + i, c0 + j) ?? 0, w = filt[i][j], p = a * w;
        terms.push({ a, w, p, pad: padded(r0 + i, c0 + j) === null }); sum += p;
      }
      return { or, oc, r0, c0, terms, sum };
    }
    const maxResp = () => filt.flat().reduce((s, v) => s + Math.abs(v), 0) || 1;

    /* ---------- styles（本概念專用） ---------- */
    bench.append(h("style", { text: `
      .cv { --cs: clamp(26px, 7.2vw, 34px); }
      .cv-stage { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 14px; justify-content: center; }
      .cv-block { display: flex; flex-direction: column; gap: 6px; align-items: center; }
      .cv-cap { font-size: 12.5px; color: var(--muted); text-align: center; line-height: 1.4; }
      .cv-cap b { color: var(--ink); }
      .cv-op { font-size: 22px; color: var(--muted); font-family: var(--f-mono); }
      .cv-grid { position: relative; display: grid; gap: 2px; }
      .cv-cell { width: var(--cs); height: var(--cs); display: grid; place-items: center; font-family: var(--f-mono); font-size: 12px; border-radius: 4px; border: 1px solid var(--line); background: var(--panel); color: var(--muted); user-select: none; }
      button.cv-cell { cursor: pointer; padding: 0; }
      button.cv-cell:hover { border-color: var(--accent); }
      .cv-cell.on { background: var(--ink); color: var(--panel); border-color: var(--ink); }
      .cv-cell.pad { border-style: dashed; background: transparent; color: var(--faint); }
      .cv-cell.empty { background: var(--panel-2); color: transparent; }
      .cv-cell.cur { outline: 2.5px solid var(--hot); outline-offset: 1px; color: var(--ink); font-weight: 600; }
      .cv-win { position: absolute; border: 3px solid var(--hot); border-radius: 6px; pointer-events: none; transition: left .18s, top .18s; box-shadow: 0 0 0 9999px rgb(0 0 0 / 0); }
      @media (prefers-reduced-motion: reduce) { .cv-win { transition: none; } }
      .cv-filter .cv-cell { font-weight: 600; color: var(--ink); }
      .cv-prod { display: grid; grid-template-columns: repeat(3, auto); gap: 4px; }
      .cv-prod span { font-family: var(--f-mono); font-size: 12px; padding: 2px 6px; border-radius: 4px; background: var(--panel); border: 1px solid var(--line); white-space: nowrap; text-align: right; }
      .cv-prod span.nz { border-color: var(--hot); }
      .cv-calc { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
    ` }));

    /* ---------- controls ---------- */
    const imgSeg = ui.seg({ label: t("imgLabel"), value: imgKey, options: Object.entries(IMAGES).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { imgKey = v; img = makeImg(v); render(); } });
    const fSeg = ui.seg({ label: t("filterLabel"), value: fKey, options: Object.entries(FILTERS).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { fKey = v; filt = FILTERS[v].w.map((r) => r.slice()); render(); } });
    const sSeg = ui.seg({ label: t("strideLabel"), value: S, options: [{ value: 1, label: t("stride1") }, { value: 2, label: t("stride2") }],
      onChange: (v) => { S = v; done = Math.min(done, outN() ** 2); done = 0; render(); } });
    const pSeg = ui.seg({ label: t("padLabel"), value: P, options: [{ value: 0, label: t("pad0") }, { value: 1, label: t("pad1") }],
      onChange: (v) => { P = v; done = 0; render(); } });

    const player = ui.player(ctx, () => { if (done >= outN() ** 2) return false; done++; render(); return done < outN() ** 2; }, 600);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (done > 0) { done--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (done < outN() ** 2) { done++; render(); } }, { primary: true });
    const allBtn = ui.btn(t("btnAll"), () => { player.pause(); done = outN() ** 2; render(); });
    const clrBtn = ui.btn(t("btnClear"), () => { player.pause(); done = 0; render(); });

    const stage = h("div", { class: "cv-stage" });
    const calcBox = h("div", { class: "calc cv-calc", style: { whiteSpace: "normal" } });
    const sizeLine = h("div", { class: "calc" });
    const paramLine = h("p", { class: "note" });
    const edgeLine = h("p", { class: "hint" });

    bench.classList.add("cv");
    bench.append(
      h("div", { class: "bench__row" }, imgSeg.el, fSeg.el),
      h("div", { class: "bench__row" }, sSeg.el, pSeg.el),
      h("div", { class: "scroll-x" }, stage),
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, clrBtn),
      calcBox, edgeLine, sizeLine, paramLine,
    );

    /* ---------- render ---------- */
    function render() {
      const n = outN(), total = n * n;
      done = Math.min(done, total);
      const curIdx = done - 1;
      const cur = curIdx >= 0 ? cellAt(curIdx) : null;
      const fmtW = (v) => (Number.isInteger(v) ? (v < 0 ? "−" + -v : String(v)) : "1/9");
      const fmtO = (v) => (isInt() ? fmt(v, 0) : fmt(v, 2));

      // input (padded)
      const Np = N + 2 * P;
      const inGrid = h("div", { class: "cv-grid", style: { gridTemplateColumns: `repeat(${Np}, var(--cs))` } });
      for (let r = 0; r < Np; r++) for (let c = 0; c < Np; c++) {
        const v = padded(r, c);
        if (v === null) { inGrid.append(h("div", { class: "cv-cell pad" }, "0")); continue; }
        const b = h("button", { type: "button", class: "cv-cell" + (v ? " on" : ""), "aria-label": t("cellIn", { r: r - P + 1, c: c - P + 1, v }) }, String(v));
        b.addEventListener("click", () => { img[r - P][c - P] ^= 1; imgSeg.set(null); render(); });
        inGrid.append(b);
      }
      if (cur) {
        // 視窗位置：每格 = cs + 2px gap；用 CSS calc
        inGrid.append(h("div", { class: "cv-win", style: {
          left: `calc(${cur.c0} * (var(--cs) + 2px) - 4px)`, top: `calc(${cur.r0} * (var(--cs) + 2px) - 4px)`,
          width: `calc(${K} * var(--cs) + ${(K - 1) * 2}px + 8px)`, height: `calc(${K} * var(--cs) + ${(K - 1) * 2}px + 8px)` } }));
      }

      // filter
      const fGrid = h("div", { class: "cv-grid cv-filter", style: { gridTemplateColumns: `repeat(3, var(--cs))` } });
      filt.forEach((row, i) => row.forEach((v, j) => {
        const b = h("button", { type: "button", class: "cv-cell", style: { background: Atlas.divColor(v, 1.2) }, "aria-label": t("cellFilter", { r: i + 1, c: j + 1, v: fmtW(v) }) }, fmtW(v));
        b.addEventListener("click", () => {
          if (fKey !== "custom") { filt = isInt() ? filt.map((r) => r.slice()) : filt.map((r) => r.map(() => 0)); fKey = "custom"; fSeg.set("custom"); }
          const cyc = { 0: 1, 1: -1, "-1": 0 };
          filt[i][j] = cyc[filt[i][j]] ?? 0;
          FILTERS.custom.w = filt.map((r) => r.slice());
          render();
        });
        fGrid.append(b);
      }));

      // output
      const max = maxResp();
      const oGrid = h("div", { class: "cv-grid", style: { gridTemplateColumns: `repeat(${n}, var(--cs))` } });
      for (let k = 0; k < total; k++) {
        const isDone = k < done;
        const v = isDone ? cellAt(k).sum : null;
        const b = h("button", { type: "button", class: "cv-cell" + (isDone ? "" : " empty") + (k === curIdx ? " cur" : ""),
          style: isDone ? { background: Atlas.divColor(v, max), color: Math.abs(v) / max > 0.55 ? "var(--panel)" : "var(--ink)" } : null,
          "aria-label": t("cellOut", { r: Math.floor(k / n) + 1, c: (k % n) + 1 }) }, isDone ? fmtO(v) : "·");
        b.addEventListener("click", () => { player.pause(); done = k + 1; render(); });
        oGrid.append(b);
      }

      stage.replaceChildren(
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: t("capIn", { n: N }) + (P ? t("capInPad") : "") }), inGrid),
        h("div", { class: "cv-op", "aria-hidden": "true" }, "✱"),
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: t("capFilter") }), fGrid),
        h("div", { class: "cv-op", "aria-hidden": "true" }, "="),
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: t("capOut", { n }) }), oGrid),
      );

      // calc
      if (!cur) {
        calcBox.innerHTML = `<span class="k">${t("calcStart")}</span>`;
      } else {
        const prod = h("div", { class: "cv-prod" }, ...cur.terms.map((x) =>
          h("span", { class: x.p !== 0 ? "nz" : "" }, `${x.a}×${Number.isInteger(x.w) ? fmtP(x.w, 0) : "1/9"}`)));
        const nonzero = cur.terms.filter((x) => x.p !== 0);
        const expr = nonzero.length ? nonzero.map((x) => (isInt() ? fmtP(x.p, 0) : "1/9")).join(" + ") : "0";
        calcBox.replaceChildren(
          h("span", { class: "k" }, t("calcCellK", { r: cur.or + 1, c: cur.oc + 1 })),
          prod,
          h("span", { html: `<span class="k">${t("calcSumK")}</span> ${expr} = <span class="hl">${fmtO(cur.sum)}</span>` }),
        );
      }
      sizeLine.innerHTML = `<span class="k">${t("sizeK")}</span> ` + t("size", { n: N, p: P, k: K, s: S, a: N + 2 * P - K, out: n });
      edgeLine.hidden = !(fKey === "vert" || fKey === "horz");
      edgeLine.innerHTML = t(fKey === "horz" ? "edgeHorz" : "edgeVert");
      paramLine.innerHTML = t("param", { outN: n, inN: N, inCells: N * N, outCells: total, big: (N * N * total).toLocaleString("en-US") });
      prevBtn.disabled = done === 0;
      nextBtn.disabled = done >= total;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
