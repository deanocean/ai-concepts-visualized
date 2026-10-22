/* 正規化層（大綱 §19）
   一層的輸出：4 筆資料（N）× 4 個通道（C）× 每張特徵圖 2×2（H×W），共 64 個數。
   批次正規化／層正規化／實例正規化／組正規化（2 組，每組 2 個通道）
   差別只在「哪些數一起算平均與變異數」。點任一個數看它屬於哪一組，算式框代入實際數字。
   「換掉其他 3 筆資料」：資料 1 不變，看它正規化後的值會不會跟著變（只有批次正規化會）。
   文字在 locales/zh/norm.js、locales/ja/norm.js。 */
Atlas.define("norm", {
  mount(bench, ctx) {
    const { h, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const N = 4, C = 4, HW = 4, G = 2, EPS = 1e-5;
    const CH_MEAN = [10, -2, 50, 0], CH_SD = [2, 1, 8, 3];
    let seed = 1;
    function rng(sd) { return function () { sd |= 0; sd = (sd + 0x6d2b79f5) | 0; let t = Math.imul(sd ^ (sd >>> 15), 1 | sd); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    const gauss = (r) => Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());
    /** 第 n 筆資料（每筆有自己的整體偏移，讓層正規化也有事可做） */
    function sample(r) {
      const shift = gauss(r) * 1.5;
      return Array.from({ length: C }, (_, c) => Array.from({ length: HW }, () => Math.round(CH_MEAN[c] + CH_SD[c] * (gauss(r) + shift * 0.6))));
    }
    const first = sample(rng(4242));
    let X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + i)))];
    let method = "bn", showAfter = false, sel = { n: 0, c: 1, k: 0 }, prevSel = null;

    const METHODS = {
      bn: { name: t("mBn"), desc: "descBn" },
      ln: { name: t("mLn"), desc: "descLn" },
      in: { name: t("mIn"), desc: "descIn" },
      gn: { name: t("mGn"), desc: "descGn" },
    };
    const group = (m, n, c) => {
      const out = [];
      for (let nn = 0; nn < N; nn++) for (let cc = 0; cc < C; cc++) for (let k = 0; k < HW; k++) {
        const ok = m === "bn" ? cc === c : m === "ln" ? nn === n : m === "in" ? nn === n && cc === c : nn === n && Math.floor(cc / (C / G)) === Math.floor(c / (C / G));
        if (ok) out.push([nn, cc, k]);
      }
      return out;
    };
    function stats(m, n, c) {
      const vals = group(m, n, c).map(([a, b, k]) => X[a][b][k]);
      const mu = vals.reduce((acc, v) => acc + v, 0) / vals.length;
      const v2 = vals.reduce((acc, v) => acc + (v - mu) ** 2, 0) / vals.length;
      return { vals, mu, v2 };
    }
    const normed = (m, n, c, k) => { const st = stats(m, n, c); return (X[n][c][k] - st.mu) / Math.sqrt(st.v2 + EPS); };

    bench.append(h("style", { text: `
      .nm-wrap { display: grid; grid-template-columns: auto repeat(${C}, auto); gap: 6px 8px; align-items: center; width: max-content; margin-inline: auto; }
      .nm-head { font-size: 12px; color: var(--muted); text-align: center; line-height: 1.3; }
      .nm-head b { color: var(--ink); }
      .nm-gbar { grid-column: span 2; text-align: center; font-size: 11.5px; color: var(--hot); border-bottom: 2px solid var(--hot); padding-bottom: 1px; }
      .nm-gbar.off { visibility: hidden; }
      .nm-row { font-size: 12.5px; color: var(--muted); white-space: nowrap; text-align: right; }
      .nm-row.fixed b { color: var(--accent); }
      .nm-map { display: grid; grid-template-columns: repeat(2, var(--nc)); gap: 2px; padding: 3px; border-radius: 7px; border: 1.5px solid transparent; }
      .nm-map.grp { border-color: var(--hot); background: var(--hot-soft); }
      .nm-v { width: var(--nc); height: calc(var(--nc) * 0.8); display: grid; place-items: center; font-family: var(--f-mono); font-size: 11.5px; border: 1px solid var(--line); border-radius: 4px; background: var(--panel); color: var(--ink); padding: 0; cursor: pointer; }
      .nm-v:hover { border-color: var(--accent); }
      .nm-v.cur { outline: 2.5px solid var(--hot); outline-offset: 1px; font-weight: 700; }
      .nm { --nc: clamp(30px, 8.6vw, 40px); }
    ` }));
    bench.classList.add("nm");

    const mSeg = ui.seg({ label: t("methodLabel"), value: method, options: Object.entries(METHODS).map(([k, v]) => ({ value: k, label: v.name })), onChange: (v) => { method = v; prevSel = null; render(); } });
    const viewSeg = ui.seg({ label: t("viewLabel"), value: showAfter, options: [{ value: false, label: t("viewBefore") }, { value: true, label: t("viewAfter") }], onChange: (v) => { showAfter = v; render(); } });
    const desc = h("p", { class: "hint" });
    const table = h("div", { class: "nm-wrap" });
    const status = h("div", { class: "status" });
    const calc = h("div", { class: "calc" });
    const swapBtn = ui.btn(t("btnSwap"), () => {
      prevSel = sel.n === 0 ? { key: `${sel.c},${sel.k},${method}`, val: normed(method, 0, sel.c, sel.k), all: X[0].map((ch, c) => ch.map((_, k) => normed(method, 0, c, k))) } : null;
      seed++;
      X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + seed * 10 + i)))];
      render(true);
    }, { primary: true });
    const note = h("p", { class: "note" });
    bench.append(
      h("div", { class: "bench__row" }, mSeg.el, viewSeg.el),
      desc,
      h("div", { class: "scroll-x" }, table),
      status,
      h("div", { class: "bench__row" }, swapBtn, ui.btn(t("btnRestore"), () => { seed = 1; X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + i)))]; prevSel = null; render(); })),
      calc, note);

    function render(swapped) {
      const M = METHODS[method];
      desc.innerHTML = t(M.desc) + t("descFrame") + (showAfter ? t("descAfter") : "");
      const inGrp = new Set(group(method, sel.n, sel.c).map(([a, b, k]) => `${a},${b},${k}`));
      const maps = new Set(group(method, sel.n, sel.c).map(([a, b]) => `${a},${b}`));
      const kids = [h("span")];
      for (let c = 0; c < C; c++) kids.push(h("div", { class: "nm-head", html: t("headCh", { c: c + 1, m: fmt(CH_MEAN[c], 0) }) }));
      // 組正規化的分組標示
      kids.push(h("span"));
      for (let g = 0; g < G; g++) kids.push(h("div", { class: "nm-gbar" + (method === "gn" ? "" : " off") }, t("groupFmt", { g: g + 1 })));
      for (let n = 0; n < N; n++) {
        kids.push(h("div", { class: "nm-row" + (n === 0 ? " fixed" : ""), html: n === 0 ? `<b>${t("rowFmt", { n: 1 })}</b>` : t("rowFmt", { n: n + 1 }) }));
        for (let c = 0; c < C; c++) {
          const map = h("div", { class: "nm-map" + (maps.has(`${n},${c}`) ? " grp" : "") });
          for (let k = 0; k < HW; k++) {
            const v = showAfter ? normed(method, n, c, k) : X[n][c][k];
            const cur = sel.n === n && sel.c === c && sel.k === k;
            const b = h("button", { type: "button", class: "nm-v" + (cur ? " cur" : ""), "aria-label": t("cellAria", { n: n + 1, c: c + 1, k: k + 1 }),
              style: showAfter ? { background: Atlas.divColor(v, 2.2) } : null }, showAfter ? fmt(v, 1) : fmt(v, 0));
            b.addEventListener("click", () => { sel = { n, c, k }; prevSel = null; render(); });
            map.append(b);
          }
          kids.push(map);
        }
      }
      table.replaceChildren(...kids);

      const st = stats(method, sel.n, sel.c), x = X[sel.n][sel.c][sel.k], y = (x - st.mu) / Math.sqrt(st.v2 + EPS);
      // 狀態：換資料後，資料 1 的結果有沒有變
      if (swapped && prevSel) {
        const now = X[0].map((ch, c) => ch.map((_, k) => normed(method, 0, c, k)));
        const changed = now.flat().some((v, i) => Math.abs(v - prevSel.all.flat()[i]) > 1e-9);
        if (changed) { status.className = "status status--warn"; status.textContent = t("sChanged", { a: fmt(prevSel.val, 3), b: fmt(y, 3) }); }
        else { status.className = "status status--ok"; status.textContent = t("sSame", { y: fmt(y, 3) }); }
      } else if (swapped) {
        status.className = "status status--info"; status.textContent = t("sSwapped");
      } else {
        status.className = "status status--info";
        status.textContent = t("sGroup", { n: st.vals.length, mu: fmt(st.mu, 2), sd: fmt(Math.sqrt(st.v2), 2) });
      }
      const list = st.vals.map((v) => (v < 0 ? `(${fmt(v, 0)})` : String(v)));
      const shown = list.length > 16 ? list.slice(0, 16).join(" + ") + " + …" : list.join(" + ");
      calc.innerHTML =
        `<span class="calc-row"><span class="k">${t("calcSelK")}</span> ${t("calcSel", { n: sel.n + 1, c: sel.c + 1, x: fmt(x, 0) })}　<span class="k">${t("calcCountK")}</span> ${t("calcCount", { n: st.vals.length })}</span>` +
        `<span class="calc-row"><span class="k">${t("calcMeanK")}</span> μ = (${shown}) ÷ ${st.vals.length} = <b>${fmt(st.mu, 3)}</b></span>` +
        `<span class="calc-row"><span class="k">${t("calcVarK")}</span> σ² = ${t("calcVar")} = <b>${fmt(st.v2, 3)}</b>　<span class="k">${t("calcSdK")}</span> σ = ${fmt(Math.sqrt(st.v2), 3)}</span>` +
        `<span class="calc-row"><span class="k">${t("calcNormK")}</span> (x − μ) ÷ √(σ² + ε) = (${fmt(x, 0)} − ${fmtP(st.mu, 3)}) ÷ ${fmt(Math.sqrt(st.v2 + EPS), 3)} = <span class="hl">${fmt(y, 3)}</span></span>` +
        `<span class="calc-row"><span class="k">${t("calcLastK")}</span> γ × ${fmt(y, 3)} + β　<span class="k">${t("calcLastNote")}</span></span>`;
    }

    note.innerHTML = t("note");

    render();
    ctx.onTheme(() => render());
    return {};
  },
});
