/* 畳み込み（シラバス §18）
   7×7 入力 × 3×3 フィルタ，可調ストライド／パディング，逐格計算特徴マップ */
Atlas.define("conv", {
  lead: `<p>辨識圖像時，關鍵是局部的形狀（邊、角、線條），而同一種形狀可能出現在圖上任何位置。<span class="jp">畳み込み</span>（卷積，<span class="en">convolution</span>）拿一個 3×3 的小數字表在圖上滑動，每停一處就算出「這裡和這個形狀有多像」。</p>`,
  closing: `濾波器的數字和局部圖樣越吻合，相乘再加總的值就越大，特徴マップ上那一格就越深。同一組 9 個數字走遍整張圖，所以參數很少，形狀出現在哪裡都抓得到；CNN 訓練時學的，就是這些數字。`,

  mount(bench, ctx) {
    const { h, fmt, fmtP, ui } = Atlas;
    const N = 7, K = 3;

    const IMAGES = {
      bar:    { label: "縱條", px: (r, c) => c === 2 || c === 3 },
      ring:   { label: "方框", px: (r, c) => r >= 1 && r <= 5 && c >= 1 && c <= 5 && (r === 1 || r === 5 || c === 1 || c === 5) },
      diag:   { label: "斜線", px: (r, c) => r === c },
      ell:    { label: "L 形", px: (r, c) => (c === 1 && r >= 1 && r <= 5) || (r === 5 && c >= 1 && c <= 5) },
      blank:  { label: "空白", px: () => false },
    };
    const FILTERS = {
      vert:   { label: "縱向邊緣", w: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      horz:   { label: "橫向邊緣", w: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      diag:   { label: "斜線（＼）", w: [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]] },
      blur:   { label: "平均（模糊）", w: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map(() => 1 / 9)) },
      custom: { label: "自訂", w: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
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
    const imgSeg = ui.seg({ label: "入力圖像（點格子可塗改）", value: imgKey, options: Object.entries(IMAGES).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { imgKey = v; img = makeImg(v); render(); } });
    const fSeg = ui.seg({ label: `<span class="jp">フィルタ</span>（濾波器，<span class="en">filter</span>）`, value: fKey, options: Object.entries(FILTERS).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { fKey = v; filt = FILTERS[v].w.map((r) => r.slice()); render(); } });
    const sSeg = ui.seg({ label: `<span class="jp">ストライド</span>（步幅，<span class="en">stride</span>）`, value: S, options: [{ value: 1, label: "1 格" }, { value: 2, label: "2 格" }],
      onChange: (v) => { S = v; done = Math.min(done, outN() ** 2); done = 0; render(); } });
    const pSeg = ui.seg({ label: `<span class="jp">パディング</span>（填充，<span class="en">padding</span>）`, value: P, options: [{ value: 0, label: "0（不補）" }, { value: 1, label: "1（外圍補 0）" }],
      onChange: (v) => { P = v; done = 0; render(); } });

    const player = ui.player(ctx, () => { if (done >= outN() ** 2) return false; done++; render(); return done < outN() ** 2; }, 600);
    const prevBtn = ui.btn("◀ 上一格", () => { player.pause(); if (done > 0) { done--; render(); } });
    const nextBtn = ui.btn("下一格 ▶", () => { player.pause(); if (done < outN() ** 2) { done++; render(); } }, { primary: true });
    const allBtn = ui.btn("全部算完", () => { player.pause(); done = outN() ** 2; render(); });
    const clrBtn = ui.btn("清空", () => { player.pause(); done = 0; render(); });

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
        const b = h("button", { type: "button", class: "cv-cell" + (v ? " on" : ""), "aria-label": `入力第 ${r - P + 1} 列第 ${c - P + 1} 行，值 ${v}` }, String(v));
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
        const b = h("button", { type: "button", class: "cv-cell", style: { background: Atlas.divColor(v, 1.2) }, "aria-label": `フィルタ第 ${i + 1} 列第 ${j + 1} 行，值 ${fmtW(v)}，點一下改變` }, fmtW(v));
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
          "aria-label": `特徴マップ第 ${Math.floor(k / n) + 1} 列第 ${(k % n) + 1} 行` }, isDone ? fmtO(v) : "·");
        b.addEventListener("click", () => { player.pause(); done = k + 1; render(); });
        oGrid.append(b);
      }

      stage.replaceChildren(
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: `<b>入力</b> ${N}×${N}${P ? "（虛線＝補的 0）" : ""}` }), inGrid),
        h("div", { class: "cv-op", "aria-hidden": "true" }, "✱"),
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: `<b>フィルタ</b> 3×3<br>點格子可改數字` }), fGrid),
        h("div", { class: "cv-op", "aria-hidden": "true" }, "="),
        h("div", { class: "cv-block" }, h("div", { class: "cv-cap", html: `<b><span class="jp">特徴マップ</span></b>（特徵圖，<span class="en">feature map</span>）${n}×${n}<br>點格子可跳到那一格` }), oGrid),
      );

      // calc
      if (!cur) {
        calcBox.innerHTML = `<span class="k">按「下一格」：濾波器蓋住左上角 3×3，對應格相乘後加總，結果寫進特徴マップ第一格。</span>`;
      } else {
        const prod = h("div", { class: "cv-prod" }, ...cur.terms.map((t) =>
          h("span", { class: t.p !== 0 ? "nz" : "" }, `${t.a}×${Number.isInteger(t.w) ? fmtP(t.w, 0) : "1/9"}`)));
        const nonzero = cur.terms.filter((t) => t.p !== 0);
        const expr = nonzero.length ? nonzero.map((t) => (isInt() ? fmtP(t.p, 0) : "1/9")).join(" + ") : "0";
        calcBox.replaceChildren(
          h("span", { class: "k" }, `第 ${cur.or + 1} 列第 ${cur.oc + 1} 格：入力 × フィルタ（逐格相乘）`),
          prod,
          h("span", { html: `<span class="k">加總</span> ${expr} = <span class="hl">${fmtO(cur.sum)}</span>` }),
        );
      }
      sizeLine.innerHTML = `<span class="k">特徴マップ邊長</span> = (入力 ${N} + 2×パディング ${P} − フィルタ ${K}) ÷ ストライド ${S} + 1 = (${N + 2 * P - K}) ÷ ${S} + 1 = <b>${n}</b>`;
      edgeLine.hidden = !(fKey === "vert" || fKey === "horz");
      edgeLine.innerHTML = fKey === "horz"
        ? `這個濾波器上排是 +1、下排是 −1：遇到「上面有、下面沒有」的邊得正數（藍），「上面沒有、下面有」得負數（紅）；沒有邊的平坦處正負抵消，得 0。`
        : `這個濾波器左行是 +1、右行是 −1：遇到「左邊有、右邊沒有」的邊得正數（藍），「左邊沒有、右邊有」得負數（紅）；沒有邊的平坦處正負抵消，得 0。`;
      paramLine.innerHTML = `這個濾波器只有 <b class="mono">9</b> 個權重，走遍整張圖都用同一組。若改用全結合層，讓 ${n}×${n} 個輸出各自連到 ${N}×${N} 個輸入，需要 ${N * N} × ${total} = <b class="mono">${(N * N * total).toLocaleString()}</b> 個權重。フィルタ也叫<span class="jp">カーネル</span>（kernel）。接下來常接 <a href="#pool">プーリング</a>；擴大看得到的範圍見 <a href="#dilated">受容野と Dilated Convolution</a>，減少計算量見 <a href="#dwsep">Depthwise Separable Convolution</a>。`;
      prevBtn.disabled = done === 0;
      nextBtn.disabled = done >= total;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
