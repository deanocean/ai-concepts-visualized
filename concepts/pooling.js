/* プーリング（シラバス §20）
   6×6 特徴マップ → 2×2 窗、ストライド 2 → 3×3。
   最大値／平均値／グローバルアベレージプーリング（GAP）；
   把入力整體移動 1 格，對照「不移動時的出力」，看位置的小變動被吸收多少（不変性）。 */
Atlas.define("pool", {
  lead: `<p>卷積算出的特徴マップ很大，而且物體只要挪動一兩個像素，整張圖的數字就跟著變。<span class="jp">プーリング</span>（池化，<span class="en">pooling</span>）把每個小區塊濃縮成一個數：資料量變小，小小的位置偏移也被吸收掉。</p>`,
  closing: `最大値プーリング只留下每個區塊裡最強的反應，所以特徵在區塊內稍微移動，出力不變，這就是<span class="jp">不変性の獲得</span>（取得對位置偏移的不變性）。它只是取最大值或平均，沒有任何要學習的權重。`,

  mount(bench, ctx) {
    const { h, fmt, ui } = Atlas;
    const N = 6, P = 2, M = N / P; // 入力 6×6、窗 2×2、出力 3×3
    const BASE = [
      [0, 0, 0, 0, 0, 0],
      [0, 1, 3, 1, 0, 0],
      [0, 2, 9, 4, 0, 0],
      [0, 0, 4, 2, 1, 0],
      [0, 0, 1, 0, 6, 0],
      [0, 0, 0, 0, 0, 0],
    ];
    const KIND = {
      max: { label: "最大値プーリング", short: "最大值" },
      avg: { label: "平均値プーリング", short: "平均" },
      gap: { label: "グローバルアベレージプーリング（GAP）", short: "GAP" },
    };
    let kind = "max", dx = 0, dy = 0, done = 0;

    const shifted = (sx, sy) => Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => {
      const R = r - sy, C = c - sx; return R < 0 || C < 0 || R >= N || C >= N ? 0 : BASE[R][C];
    }));
    function windowVals(img, or, oc) { const v = []; for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) v.push(img[or * P + i][oc * P + j]); return v; }
    function pool(img, k) {
      if (k === "gap") { const all = img.flat(); return [[all.reduce((a, b) => a + b, 0) / all.length]]; }
      return Array.from({ length: M }, (_, r) => Array.from({ length: M }, (_, c) => {
        const v = windowVals(img, r, c); return k === "max" ? Math.max(...v) : v.reduce((a, b) => a + b, 0) / v.length;
      }));
    }
    const total = () => (kind === "gap" ? 1 : M * M);
    const same = (a, b) => Math.abs(a - b) < 1e-9;
    const fv = (v) => (kind === "max" || Number.isInteger(v) ? String(v) : fmt(v, 2));

    /* ---------- styles ---------- */
    bench.append(h("style", { text: `
      .pl { --cs: clamp(30px, 8vw, 38px); --cs2: clamp(24px, 6.4vw, 30px); }
      .pl-stage { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; justify-content: center; }
      .pl-block { display: flex; flex-direction: column; gap: 6px; align-items: center; }
      .pl-cap { font-size: 12.5px; color: var(--muted); text-align: center; line-height: 1.4; }
      .pl-cap b { color: var(--ink); }
      .pl-op { font-size: 13px; color: var(--muted); text-align: center; line-height: 1.3; }
      .pl-op b { display: block; font-size: 22px; font-family: var(--f-mono); }
      .pl-grid { position: relative; display: grid; gap: 2px; }
      .pl-cell { width: var(--cs); height: var(--cs); display: grid; place-items: center; font-family: var(--f-mono); font-size: 12.5px; border-radius: 4px; border: 1px solid var(--line); color: var(--ink); user-select: none; padding: 0; }
      .pl-ref .pl-cell { width: var(--cs2); height: var(--cs2); font-size: 11px; }
      button.pl-cell { cursor: pointer; }
      button.pl-cell:hover { border-color: var(--accent); }
      .pl-cell.empty { background: var(--panel-2); color: transparent; }
      .pl-cell.cur { outline: 2.5px solid var(--hot); outline-offset: 1px; font-weight: 700; }
      .pl-cell.chg { box-shadow: inset 0 0 0 2px var(--bad); }
      .pl-gap .pl-cell { width: calc(var(--cs) * 1.6); height: calc(var(--cs) * 1.6); font-size: 14px; font-weight: 700; }
      .pl-win { position: absolute; border: 3px solid var(--hot); border-radius: 6px; pointer-events: none; transition: left .18s, top .18s; }
      .pl-sep { position: absolute; pointer-events: none; background: var(--line-strong); }
      @media (prefers-reduced-motion: reduce) { .pl-win { transition: none; } }
      .pl-move { display: inline-grid; grid-template-columns: repeat(3, 38px); grid-template-rows: repeat(2, 32px); gap: 4px; }
      .pl-move .btn { min-height: 32px; padding: 0; }
    ` }));
    bench.classList.add("pl");

    /* ---------- controls ---------- */
    const kindSeg = ui.seg({ label: "プーリングの種類", value: kind, options: Object.entries(KIND).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { player.pause(); kind = v; done = v === "gap" ? 1 : 0; render(); } });
    const mv = (ddx, ddy) => () => { player.pause(); dx = Math.max(-1, Math.min(1, dx + ddx)); dy = Math.max(-1, Math.min(1, dy + ddy)); done = total(); render(); };
    const moveBox = h("div", { class: "pl-move", role: "group", "aria-label": "移動入力" },
      h("span"), ui.btn("↑", mv(0, -1), { sm: true, aria: "往上移 1 格" }), h("span"),
      ui.btn("←", mv(-1, 0), { sm: true, aria: "往左移 1 格" }), ui.btn("↓", mv(0, 1), { sm: true, aria: "往下移 1 格" }), ui.btn("→", mv(1, 0), { sm: true, aria: "往右移 1 格" }));
    const homeBtn = ui.btn("回到原位", () => { player.pause(); dx = 0; dy = 0; done = total(); render(); }, { sm: true });
    const moveField = h("div", { class: "field" }, h("span", { class: "field__label" }, "把入力整體移動 1 格（模擬物體稍微挪動，最多 ±1）"),
      h("div", { class: "bench__row" }, moveBox, homeBtn));

    const player = ui.player(ctx, () => { if (done >= total()) return false; done++; render(); return done < total(); }, 650);
    const prevBtn = ui.btn("◀ 上一格", () => { player.pause(); if (done > 0) { done--; render(); } });
    const nextBtn = ui.btn("下一格 ▶", () => { player.pause(); if (done < total()) { done++; render(); } }, { primary: true });
    const allBtn = ui.btn("全部算完", () => { player.pause(); done = total(); render(); });
    const clrBtn = ui.btn("清空", () => { player.pause(); done = 0; render(); });

    const status = h("div", { class: "status" });
    const stage = h("div", { class: "pl-stage" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });
    bench.append(
      h("div", { class: "bench__row" }, kindSeg.el),
      moveField, status,
      h("div", { class: "scroll-x" }, stage),
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, allBtn, clrBtn),
      calc, note,
    );

    function grid(vals, opt = {}) {
      const n = vals.length;
      const g = h("div", { class: "pl-grid" + (opt.cls ? " " + opt.cls : ""), style: { gridTemplateColumns: `repeat(${n}, ${opt.cls === "pl-ref" ? "var(--cs2)" : opt.cls === "pl-gap" ? "calc(var(--cs) * 1.6)" : "var(--cs)"})` } });
      vals.forEach((row, r) => row.forEach((v, c) => {
        const k = r * n + c;
        const hidden = opt.hideFrom != null && k >= opt.hideFrom;
        const strong = v / 9 > 0.55;
        const props = {
          class: "pl-cell" + (hidden ? " empty" : "") + (opt.cur === k ? " cur" : "") + (opt.changed && opt.changed[r][c] && !hidden ? " chg" : ""),
          style: hidden ? null : { background: Atlas.divColor(v, 9), color: strong ? "var(--panel)" : "var(--ink)" },
          "aria-label": `${opt.name || ""}第 ${r + 1} 列第 ${c + 1} 行，值 ${hidden ? "未計算" : fv(v)}`,
        };
        const txt = hidden ? "·" : opt.fmt ? opt.fmt(v) : String(v);
        if (opt.onPick) { const b = h("button", { type: "button", ...props }, txt); b.addEventListener("click", () => opt.onPick(k)); g.append(b); }
        else g.append(h("div", props, txt));
      }));
      return g;
    }

    function render() {
      const img = shifted(dx, dy), out = pool(img, kind), ref = pool(BASE, kind);
      const T = total(); done = Math.min(done, T);
      const curIdx = done - 1;
      const moved = dx !== 0 || dy !== 0;

      // input with 2×2 block separators and current window
      const inG = grid(img, { name: "入力" });
      if (kind !== "gap") {
        for (let i = 1; i < M; i++) {
          inG.append(h("div", { class: "pl-sep", style: { left: `calc(${i * P} * (var(--cs) + 2px) - 2px)`, top: "0", width: "2px", height: "100%" } }));
          inG.append(h("div", { class: "pl-sep", style: { top: `calc(${i * P} * (var(--cs) + 2px) - 2px)`, left: "0", height: "2px", width: "100%" } }));
        }
        if (curIdx >= 0) {
          const or = Math.floor(curIdx / M), oc = curIdx % M;
          inG.append(h("div", { class: "pl-win", style: {
            left: `calc(${oc * P} * (var(--cs) + 2px) - 4px)`, top: `calc(${or * P} * (var(--cs) + 2px) - 4px)`,
            width: `calc(${P} * var(--cs) + ${(P - 1) * 2}px + 8px)`, height: `calc(${P} * var(--cs) + ${(P - 1) * 2}px + 8px)` } }));
        }
      } else if (done) {
        inG.append(h("div", { class: "pl-win", style: { left: "-4px", top: "-4px", width: "calc(100% + 8px)", height: "calc(100% + 8px)" } }));
      }

      const changed = out.map((row, r) => row.map((v, c) => moved && !same(v, ref[r][c])));
      const isGap = kind === "gap";
      const outG = grid(out, { name: "出力", cls: isGap ? "pl-gap" : "", hideFrom: done, cur: curIdx, changed, fmt: fv,
        onPick: (k) => { player.pause(); done = k + 1; render(); } });
      const blocks = [
        h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: `<b>入力</b>（特徴マップ）${N}×${N}${moved ? `<br>已移動：${dx ? (dx > 0 ? "右" : "左") + " 1 格" : ""}${dx && dy ? "、" : ""}${dy ? (dy > 0 ? "下" : "上") + " 1 格" : ""}` : `<br>粗線＝2×2 的區塊`}` }), inG),
        h("div", { class: "pl-op", "aria-hidden": "true", html: `<b>→</b>${isGap ? "整張平均" : `2×2 取${KIND[kind].short}`}` }),
        h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: `<b>出力</b> ${isGap ? "1 個數" : `${M}×${M}`}${moved ? "<br>紅框＝和不移動時不同" : "<br>點格子可跳到那一格"}` }), outG),
      ];
      if (moved) blocks.push(h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: `不移動時的出力<br>（對照用）` }),
        grid(ref, { name: "不移動時的出力", cls: isGap ? "pl-gap" : "pl-ref", fmt: fv })));
      stage.replaceChildren(...blocks);

      // status
      const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };
      if (!moved) {
        set("info", isGap
          ? "GAP 把整張特徴マップ平均成 1 個數。試著移動入力"
          : `36 個數縮成 9 個。算完後試著用方向鍵移動入力，看出力有多少格改變`);
      } else {
        const inChg = img.flat().filter((v, i) => !same(v, BASE.flat()[i])).length;
        const outChg = changed.flat().filter(Boolean).length;
        if (isGap) set("ok", `入力有 ${inChg} 格的值變了，但整張的平均 ${fmt(out[0][0], 2)} 完全不變：只要特徵還在圖裡，位置在哪都一樣`);
        else {
          const am = (g) => { const f = g.flat(); return f.indexOf(Math.max(...f)); };
          const keep = am(out) === am(ref);
          const extra = kind === "max"
            ? (keep ? "最強的反應 9 仍在出力的同一格" : "最強的反應 9 跨過了區塊邊界，換到另一格")
            : "平均會把移進、移出的小數字都算進去，所以變動比較多";
          set(outChg <= 4 ? "ok" : "warn", `入力有 ${inChg} 格的值變了；出力 9 格中 ${outChg} 格改變。${extra}`);
        }
      }

      // calc
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      if (curIdx < 0) {
        calc.innerHTML = row("按「下一格」", isGap ? "把整張 36 格加總再除以 36" : `橘框蓋住左上角 2×2，取${KIND[kind].short}，寫進出力第一格；接著往右跳 2 格（ストライド 2）`);
      } else if (isGap) {
        const sum = img.flat().reduce((a, b) => a + b, 0);
        calc.innerHTML = row("GAP", `36 格加總 ${sum} ÷ 36 = <span class="hl">${fmt(sum / 36, 2)}</span>`) +
          row("為什麼常用", `最後一層特徴マップ 7×7、512 通道，要分 1000 類：攤平後接全結合層要 7×7×512×1000 = ${(7 * 7 * 512 * 1000).toLocaleString()} 個權重；先做 GAP 變成 512 個數，只要 512×1000 = ${(512 * 1000).toLocaleString()} 個`);
      } else {
        const or = Math.floor(curIdx / M), oc = curIdx % M, v = windowVals(img, or, oc);
        const expr = kind === "max" ? `max(${v.join(", ")}) = <span class="hl">${Math.max(...v)}</span>`
          : `(${v.join(" + ")}) ÷ 4 = ${v.reduce((a, b) => a + b, 0)} ÷ 4 = <span class="hl">${fmt(v.reduce((a, b) => a + b, 0) / 4, 2)}</span>`;
        calc.innerHTML = row(`出力第 ${or + 1} 列第 ${oc + 1} 格`, expr) +
          row("尺寸", `邊長 (6 − 2) ÷ ストライド 2 + 1 = 3，數量 36 → 9（剩 1/4）`);
      }
      note.innerHTML = `示範的 6×6 是某個濾波器算出的特徴マップ（數字越大＝越像它要找的形狀）。窗的大小和ストライド通常一樣，區塊之間不重疊。只有在同一個 2×2 區塊裡的移動會被完全吸收，跨過區塊邊界就會改變；疊好幾層卷積＋プーリング，能容忍的偏移就逐層變大。GAP 直接把每張特徴マップ平均成一個數，常用來取代最後的全結合層。卷積本身見 <a href="#conv">畳み込み</a>。`;

      prevBtn.disabled = done === 0;
      nextBtn.disabled = allBtn.disabled = done >= T;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
