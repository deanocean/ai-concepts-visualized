/* 池化／プーリング（シラバス §20）
   6×6 特徵圖 → 2×2 窗、步幅 2 → 3×3。
   最大池化／平均池化／全域平均池化（GAP）；
   把輸入整體移動 1 格，對照「不移動時的輸出」，看位置的小變動被吸收多少（不變性）。
   文字在 locales/zh/pool.js、locales/ja/pool.js。 */
Atlas.define("pool", {
  mount(bench, ctx) {
    const { h, fmt, ui } = Atlas;
    const { t } = ctx;
    const N = 6, P = 2, M = N / P; // 輸入 6×6、窗 2×2、輸出 3×3
    const BASE = [
      [0, 0, 0, 0, 0, 0],
      [0, 1, 3, 1, 0, 0],
      [0, 2, 9, 4, 0, 0],
      [0, 0, 4, 2, 1, 0],
      [0, 0, 1, 0, 6, 0],
      [0, 0, 0, 0, 0, 0],
    ];
    const KIND = {
      max: { label: t("kindMax"), short: t("shortMax") },
      avg: { label: t("kindAvg"), short: t("shortAvg") },
      gap: { label: t("kindGap"), short: "GAP" },
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
    const kindSeg = ui.seg({ label: t("kindLabel"), value: kind, options: Object.entries(KIND).map(([v, o]) => ({ value: v, label: o.label })),
      onChange: (v) => { player.pause(); kind = v; done = v === "gap" ? 1 : 0; render(); } });
    const mv = (ddx, ddy) => () => { player.pause(); dx = Math.max(-1, Math.min(1, dx + ddx)); dy = Math.max(-1, Math.min(1, dy + ddy)); done = total(); render(); };
    const moveBox = h("div", { class: "pl-move", role: "group", "aria-label": t("moveGroup") },
      h("span"), ui.btn("↑", mv(0, -1), { sm: true, aria: t("moveUp") }), h("span"),
      ui.btn("←", mv(-1, 0), { sm: true, aria: t("moveLeft") }), ui.btn("↓", mv(0, 1), { sm: true, aria: t("moveDown") }), ui.btn("→", mv(1, 0), { sm: true, aria: t("moveRight") }));
    const homeBtn = ui.btn(t("moveHome"), () => { player.pause(); dx = 0; dy = 0; done = total(); render(); }, { sm: true });
    const moveField = h("div", { class: "field" }, h("span", { class: "field__label" }, t("moveLabel")),
      h("div", { class: "bench__row" }, moveBox, homeBtn));

    const player = ui.player(ctx, () => { if (done >= total()) return false; done++; render(); return done < total(); }, 650);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (done > 0) { done--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (done < total()) { done++; render(); } }, { primary: true });
    const allBtn = ui.btn(t("btnAll"), () => { player.pause(); done = total(); render(); });
    const clrBtn = ui.btn(t("btnClear"), () => { player.pause(); done = 0; render(); });

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
          "aria-label": t("cellAria", { name: opt.name || "", r: r + 1, c: c + 1, v: hidden ? t("cellEmpty") : fv(v) }),
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
      const inG = grid(img, { name: t("nameIn") });
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
      const outG = grid(out, { name: t("nameOut"), cls: isGap ? "pl-gap" : "", hideFrom: done, cur: curIdx, changed, fmt: fv,
        onPick: (k) => { player.pause(); done = k + 1; render(); } });
      const dirs = [dx ? t(dx > 0 ? "dirRight" : "dirLeft") : null, dy ? t(dy > 0 ? "dirDown" : "dirUp") : null].filter(Boolean).join(t("dirSep"));
      const blocks = [
        h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: t("capIn", { n: N }) + "<br>" + (moved ? t("capInMoved", { dirs }) : t("capInBlocks")) }), inG),
        h("div", { class: "pl-op", "aria-hidden": "true", html: `<b>→</b>${isGap ? t("opGap") : t("opWin", { short: KIND[kind].short })}` }),
        h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: t("capOut", { size: isGap ? t("sizeOne") : `${M}×${M}` }) + "<br>" + (moved ? t("capOutChanged") : t("capOutPick")) }), outG),
      ];
      if (moved) blocks.push(h("div", { class: "pl-block" }, h("div", { class: "pl-cap", html: t("capRef") }),
        grid(ref, { name: t("nameRef"), cls: isGap ? "pl-gap" : "pl-ref", fmt: fv })));
      stage.replaceChildren(...blocks);

      // status
      const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };
      if (!moved) {
        set("info", isGap ? t("statusGapIdle") : t("statusIdle"));
      } else {
        const inChg = img.flat().filter((v, i) => !same(v, BASE.flat()[i])).length;
        const outChg = changed.flat().filter(Boolean).length;
        if (isGap) set("ok", t("statusGapMoved", { inChg, avg: fmt(out[0][0], 2) }));
        else {
          const am = (g) => { const f = g.flat(); return f.indexOf(Math.max(...f)); };
          const keep = am(out) === am(ref);
          const extra = kind === "max" ? t(keep ? "extraMaxKeep" : "extraMaxMove") : t("extraAvg");
          set(outChg <= 4 ? "ok" : "warn", t("statusMoved", { inChg, outChg, extra }));
        }
      }

      // calc
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      if (curIdx < 0) {
        calc.innerHTML = row(t("calcStartK"), isGap ? t("calcStartGap") : t("calcStart", { short: KIND[kind].short }));
      } else if (isGap) {
        const sum = img.flat().reduce((a, b) => a + b, 0);
        calc.innerHTML = row("GAP", t("calcGap", { sum, avg: fmt(sum / 36, 2) })) +
          row(t("calcWhyK"), t("calcWhy", { big: (7 * 7 * 512 * 1000).toLocaleString("en-US"), small: (512 * 1000).toLocaleString("en-US") }));
      } else {
        const or = Math.floor(curIdx / M), oc = curIdx % M, v = windowVals(img, or, oc);
        const expr = kind === "max" ? `max(${v.join(", ")}) = <span class="hl">${Math.max(...v)}</span>`
          : `(${v.join(" + ")}) ÷ 4 = ${v.reduce((a, b) => a + b, 0)} ÷ 4 = <span class="hl">${fmt(v.reduce((a, b) => a + b, 0) / 4, 2)}</span>`;
        calc.innerHTML = row(t("calcCellK", { r: or + 1, c: oc + 1 }), expr) +
          row(t("calcSizeK"), t("calcSize"));
      }
      note.innerHTML = t("note");

      prevBtn.disabled = done === 0;
      nextBtn.disabled = allBtn.disabled = done >= T;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
