/* 深度可分離卷積／Depthwise Separable Convolution（シラバス §18）
   ① 小例子：輸入 5×5×3 → 輸出 3×3×4。
      標準卷積：每個輸出通道一個 3×3×3 濾波器（同時看空間和所有通道）。
      DS：Depthwise（每個通道各自一個 3×3）→ Pointwise（1×1，把通道混合）。
      這裡的標準卷積權重設成「Pointwise 係數 × Depthwise 濾波器」，所以兩種做法輸出完全相同，可直接比較計算量。
   ② 計算機：調輸入通道・輸出通道・卷積核・特徵圖大小，比較權重數與乘法次數。
   文字在 locales/zh/dwsep.js、locales/ja/dwsep.js。 */
Atlas.define("dwsep", {
  mount(bench, ctx) {
    const { h, fmt, ui } = Atlas;
    const { t } = ctx;
    const N = 5, K = 3, O = N - K + 1, CIN = 3, COUT = 4;
    const INP = [
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (c === 1 ? 1 : 0))), // 縱線
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (r === 3 ? 1 : 0))), // 橫線
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (r === c ? 1 : 0))), // 斜線
    ];
    const CH = [t("chVert"), t("chHorz"), t("chDiag")];
    const D = [
      { name: t("dVert"), w: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      { name: t("dHorz"), w: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      { name: t("dCross"), w: [[0, 1, 0], [1, 1, 1], [0, 1, 0]] },
    ];
    const PW = [[1, 0, 0], [0, 1, 1], [1, -1, 0], [-1, 0, 1]]; // PW[出力 o][通道 c]
    const STD = PW.map((p) => D.map((d, c) => d.w.map((row) => row.map((v) => p[c] * v)))); // STD[o][c] 3×3

    const win = (c, i, j) => { const v = []; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) v.push(INP[c][i + a][j + b]); return v; };
    const DW = D.map((d, c) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) => win(c, i, j).reduce((acc, x, k) => acc + x * d.w[Math.floor(k / K)][k % K], 0))));
    const OUT = PW.map((p) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) => p.reduce((s, pc, c) => s + pc * DW[c][i][j], 0))));
    const OUT_STD = STD.map((f) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) =>
      f.reduce((acc, fc, c) => acc + win(c, i, j).reduce((a2, x, k) => a2 + x * fc[Math.floor(k / K)][k % K], 0), 0))));
    const num = (v) => v.toLocaleString("en-US");
    const fi = (v) => (v === 0 ? "0" : v < 0 ? "−" + -v : String(v));
    const fiP = (v) => (v < 0 ? "(" + fi(v) + ")" : fi(v));

    let view = "std", oSel = 2, pi = 1, pj = 1;
    let cin = 256, cout = 256, kk = 3, hh = 14;

    /* ---------- styles ---------- */
    bench.append(h("style", { text: `
      .ds { --c1: clamp(17px, 4.6vw, 22px); --c2: clamp(20px, 5.6vw, 28px); }
      .ds-row { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 10px 8px; justify-content: center; }
      .ds-stage { display: flex; flex-direction: column; gap: 10px; align-items: center; }
      .ds-lab { font-size: 12.5px; font-weight: 700; color: var(--ink); text-align: center; width: 100%; }
      .ds-lab span { font-weight: 400; color: var(--muted); }
      .ds-block { display: flex; flex-direction: column; gap: 4px; align-items: center; }
      .ds-cap { font-size: 11.5px; color: var(--muted); text-align: center; line-height: 1.35; }
      .ds-grid { position: relative; display: grid; gap: 1.5px; }
      .ds-cell { display: grid; place-items: center; font-family: var(--f-mono); border-radius: 3px; border: 1px solid var(--line); padding: 0; color: var(--ink); }
      .ds-s1 .ds-cell { width: var(--c1); height: var(--c1); font-size: 10.5px; }
      .ds-s2 .ds-cell { width: var(--c2); height: var(--c2); font-size: 11.5px; }
      button.ds-cell { cursor: pointer; }
      button.ds-cell:hover { border-color: var(--accent); }
      .ds-cell.cur { outline: 2.5px solid var(--hot); outline-offset: 1px; font-weight: 700; }
      .ds-win { position: absolute; border: 2.5px solid var(--hot); border-radius: 5px; pointer-events: none; }
      .ds-arrow { font-size: 18px; color: var(--muted); font-family: var(--f-mono); align-self: center; }
      .ds-dim { opacity: .45; }
      .ds-bars { display: grid; grid-template-columns: max-content 1fr; gap: 6px 10px; align-items: center; font-size: 13px; width: 100%; max-width: 640px; }
      .ds-bar { height: 22px; border-radius: 4px; display: flex; align-items: center; padding-inline: 6px; font-family: var(--f-mono); font-size: 12px; white-space: nowrap; min-width: 2px; }
      .ds-bar--std { background: var(--neg-soft); border: 1px solid var(--neg); }
      .ds-bar--ds { background: var(--pos-soft); border: 1px solid var(--pos); }
      .ds-h { font-size: 12.5px; color: var(--muted); grid-column: 1 / -1; margin-top: 4px; font-weight: 700; }
    ` }));
    bench.classList.add("ds");

    /* ---------- controls ---------- */
    const viewSeg = ui.seg({ label: t("viewLabel"), value: view, options: [
      { value: "std", label: t("viewStd") }, { value: "ds", label: t("viewDs") }, { value: "calc", label: t("viewCalc") }],
      onChange: (v) => { view = v; render(); } });
    const oSeg = ui.seg({ label: t("oLabel"), value: oSel, options: [0, 1, 2, 3].map((v) => ({ value: v, label: t("outN", { o: v + 1 }) })), onChange: (v) => { oSel = v; render(); } });
    const pick = (opts) => ui.seg(opts);
    const cinSeg = pick({ label: t("cinLabel"), value: cin, options: [3, 16, 64, 128, 256, 512].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { cin = v; render(); } });
    const coutSeg = pick({ label: t("coutLabel"), value: cout, options: [4, 16, 64, 128, 256, 512].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { cout = v; render(); } });
    const kSeg = pick({ label: t("kLabel"), value: kk, options: [3, 5, 7].map((v) => ({ value: v, label: `${v}×${v}` })), onChange: (v) => { kk = v; render(); } });
    const hSeg = pick({ label: t("hLabel"), value: hh, options: [3, 7, 14, 28, 56, 112].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { hh = v; render(); } });
    const presets = h("div", { class: "btn-group" },
      ui.btn(t("preSmall"), () => { cin = 3; cout = 4; kk = 3; hh = 3; sync(); render(); }, { sm: true }),
      ui.btn(t("preMobile"), () => { cin = 256; cout = 256; kk = 3; hh = 14; sync(); render(); }, { sm: true }));
    function sync() { cinSeg.set(cin); coutSeg.set(cout); kSeg.set(kk); hSeg.set(hh); }

    const rowSmall = h("div", { class: "bench__row" }, oSeg.el);
    const rowCalc = h("div", { class: "bench__row" }, cinSeg.el, coutSeg.el, kSeg.el, hSeg.el);
    const rowPre = h("div", { class: "field" }, h("span", { class: "field__label" }, t("preLabel")), presets);
    const status = h("div", { class: "status" });
    const fig = h("div", { class: "scroll-x" });
    const hint = h("p", { class: "hint" });
    const calc = h("div", { class: "calc", style: { whiteSpace: "normal" } });
    const bars = h("div", { class: "ds-bars" });
    const note = h("p", { class: "note" });
    bench.append(h("div", { class: "bench__row" }, viewSeg.el), rowSmall, rowCalc, rowPre, status, fig, hint, calc, bars, note);

    /* ---------- grid helper ---------- */
    function grid(vals, opt = {}) {
      const n = vals.length;
      const g = h("div", { class: "ds-grid " + (opt.size || "ds-s2"), style: { gridTemplateColumns: `repeat(${n}, var(${opt.size === "ds-s1" ? "--c1" : "--c2"}))` } });
      vals.forEach((row, r) => row.forEach((v, c) => {
        const props = { class: "ds-cell" + (opt.cur && opt.cur[0] === r && opt.cur[1] === c ? " cur" : ""),
          style: { background: Atlas.divColor(v, opt.max || 3), color: Math.abs(v) / (opt.max || 3) > 0.6 ? "var(--panel)" : "var(--ink)" },
          "aria-label": t("cellAria", { name: opt.name || "", r: r + 1, c: c + 1, v: fi(v) }) };
        if (opt.onPick) { const b = h("button", { type: "button", ...props }, fi(v)); b.addEventListener("click", () => opt.onPick(r, c)); g.append(b); }
        else g.append(h("div", props, fi(v)));
      }));
      if (opt.win) g.append(h("div", { class: "ds-win", style: {
        left: `calc(${opt.win[1]} * (var(--c1) + 1.5px) - 3.5px)`, top: `calc(${opt.win[0]} * (var(--c1) + 1.5px) - 3.5px)`,
        width: `calc(${K} * var(--c1) + ${(K - 1) * 1.5}px + 7px)`, height: `calc(${K} * var(--c1) + ${(K - 1) * 1.5}px + 7px)` } }));
      return g;
    }
    const block = (cap, el, cls) => h("div", { class: "ds-block" + (cls ? " " + cls : "") }, el, h("div", { class: "ds-cap", html: cap }));
    const pickPx = (r, c) => { pi = r; pj = c; render(); };
    const inputs = () => INP.map((m, c) => block(t("capIn", { c: c + 1, name: CH[c] }), grid(m, { size: "ds-s1", name: t("nameIn", { c: c + 1 }), win: [pi, pj], max: 1.6 })));
    const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
    const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };

    function countBars(a) {
      const items = [
        [t("barW"), a.wStd, a.wDs, `${a.kk}×${a.kk}×${a.cin}×${a.cout}`, `${a.kk}×${a.kk}×${a.cin} + ${a.cin}×${a.cout}`],
        [t("barM"), a.mStd, a.mDs, t("fMulStd", { h: a.hh, kkc: a.kk * a.kk * a.cin, cout: a.cout }), t("fMulDs", { h: a.hh, kk: a.kk * a.kk, cin: a.cin, cout: a.cout })],
      ];
      const kids = [];
      for (const [lab, std, ds, fs, fd] of items) {
        kids.push(h("div", { class: "ds-h" }, lab));
        kids.push(h("span", null, t("barStd")), h("div", { class: "ds-bar ds-bar--std", style: { width: "100%" }, title: fs }, num(std)));
        const pct = Math.max(0.6, (ds / std) * 100), inside = pct > 30;
        kids.push(h("span", null, t("barDs")), h("div", { style: { width: "100%", display: "flex", alignItems: "center", gap: "6px" } },
          h("div", { class: "ds-bar ds-bar--ds", style: { width: `${pct}%` }, title: fd }, inside ? num(ds) : ""),
          inside ? null : h("span", { class: "mono", style: { fontSize: "12px" } }, num(ds))));
      }
      bars.replaceChildren(...kids);
    }
    const counts = (cin, cout, kk, hh) => ({ cin, cout, kk, hh,
      wStd: kk * kk * cin * cout, wDs: kk * kk * cin + cin * cout,
      mStd: hh * hh * kk * kk * cin * cout, mDs: hh * hh * (kk * kk * cin + cin * cout) });

    /* ---------- views ---------- */
    function renderStd() {
      const f = STD[oSel];
      const filt = f.map((m, c) => block(t("capSlice", { c: c + 1 }), grid(m, { size: "ds-s1", name: t("nameFilt", { c: c + 1 }), max: 1.6 })));
      const outs = OUT_STD.map((m, o) => block(t("outN", { o: o + 1 }), grid(m, { name: t("nameOut", { o: o + 1 }), cur: [pi, pj], onPick: pickPx, max: 6 }), o === oSel ? "" : "ds-dim"));
      fig.replaceChildren(h("div", { class: "ds-stage" },
        h("div", { class: "ds-lab", html: t("labInStd") }), h("div", { class: "ds-row" }, ...inputs()),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: t("labFilt", { o: oSel + 1 }) }), h("div", { class: "ds-row" }, ...filt),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: t("labOut") }), h("div", { class: "ds-row" }, ...outs)));
      const parts = [0, 1, 2].map((c) => win(c, pi, pj).reduce((acc, x, k) => acc + x * f[c][Math.floor(k / K)][k % K], 0));
      calc.innerHTML = row(t("calcStdK", { o: oSel + 1, r: pi + 1, c: pj + 1 }), t("calcStd", { expr: parts.map(fiP).join(" + "), v: fi(OUT_STD[oSel][pi][pj]) })) +
        row(t("calcLayerK"), t("calcLayer"));
      set("warn", t("statusStd"));
      hint.textContent = t("hintStd");
      countBars(counts(CIN, COUT, K, O));
      note.innerHTML = t("noteStd");
    }
    function renderDs() {
      const dws = DW.map((m, c) => block(t("capDw", { c: c + 1 }), grid(m, { name: t("nameDw", { c: c + 1 }), cur: [pi, pj], onPick: pickPx, max: 3 })));
      const kerns = D.map((d, c) => block(t("capDwF", { c: c + 1, name: d.name }), grid(d.w, { size: "ds-s1", name: t("nameDwF", { c: c + 1 }), max: 1.6 })));
      const pw = h("div", { class: "ds-block" },
        h("table", { class: "tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, t("pwHead")), ...[1, 2, 3].map((c) => h("th", null, t("pwCh", { c }))))),
          h("tbody", null, ...PW.map((p, o) => h("tr", { class: o === oSel ? "is-sel" : "" }, h("td", { class: "l" }, t("outN", { o: o + 1 })), ...p.map((v) => h("td", null, fi(v))))))),
        h("div", { class: "ds-cap" }, t("pwCap")));
      const outs = OUT.map((m, o) => block(t("outN", { o: o + 1 }), grid(m, { name: t("nameOut", { o: o + 1 }), cur: [pi, pj], onPick: pickPx, max: 6 }), o === oSel ? "" : "ds-dim"));
      fig.replaceChildren(h("div", { class: "ds-stage" },
        h("div", { class: "ds-lab", html: t("labInDs") }), h("div", { class: "ds-row" }, ...inputs()),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: t("labDw") }), h("div", { class: "ds-row" }, ...kerns),
        h("div", { class: "ds-row" }, ...dws),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: t("labPw") }), pw,
        h("div", { class: "ds-row" }, ...outs)));
      const d = [0, 1, 2].map((c) => DW[c][pi][pj]), p = PW[oSel];
      calc.innerHTML =
        row(t("calcDwK", { r: pi + 1, c: pj + 1 }), d.map((v, c) => t("calcDwPart", { c: c + 1, v: fi(v) })).join(t("calcDwSep"))) +
        row(t("calcPwK", { o: oSel + 1 }), `${p.map((pc, c) => `${fiP(pc)}×${fiP(d[c])}`).join(" + ")} = <span class="hl">${fi(OUT[oSel][pi][pj])}</span>`) +
        row(t("calcCmpK"), t("calcCmp", { res: t(OUT.every((m, o) => m.every((r, i) => r.every((v, j) => v === OUT_STD[o][i][j]))) ? "cmpSame" : "cmpDiff") }));
      set("ok", t("statusDs"));
      hint.textContent = t("hintDs");
      countBars(counts(CIN, COUT, K, O));
      note.innerHTML = t("noteDs");
    }
    function renderCalc() {
      const a = counts(cin, cout, kk, hh);
      const ratio = a.wDs / a.wStd;
      fig.replaceChildren();
      calc.innerHTML =
        row(t("calcStdRowK"), t("calcWeights", { expr: `${kk}×${kk}×${cin}×${cout}`, n: num(a.wStd) })) +
        row(t("calcDsRowK"), t("calcWeights", { expr: `${kk}×${kk}×${cin} + ${cin}×${cout}`, n: num(a.wDs) })) +
        row(t("calcRatioK"), t("calcRatio", { cout, kk, a: fmt(1 / cout, 4), b: fmt(1 / (kk * kk), 4), r: fmt(ratio, 3) }));
      set(ratio < 0.2 ? "ok" : ratio < 0.5 ? "warn" : "bad", t("statusCalc", { pct: fmt(ratio * 100, 1), x: fmt(1 / ratio, 1) }));
      hint.textContent = "";
      countBars(a);
      note.innerHTML = t("noteCalc");
    }

    function render() {
      rowSmall.hidden = fig.hidden = hint.hidden = view === "calc";
      rowCalc.hidden = rowPre.hidden = view !== "calc";
      ({ std: renderStd, ds: renderDs, calc: renderCalc })[view]();
    }
    render();
    ctx.onTheme(render);
  },
});
