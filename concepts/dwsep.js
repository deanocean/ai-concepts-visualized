/* Depthwise Separable Convolution（シラバス §18）
   ① 小例子：入力 5×5×3 → 出力 3×3×4。
      標準卷積：每個出力通道一個 3×3×3 濾波器（同時看空間和所有通道）。
      DS：Depthwise（每個通道各自一個 3×3）→ Pointwise（1×1，把通道混合）。
      這裡的標準卷積權重設成「Pointwise 係數 × Depthwise 濾波器」，所以兩種做法輸出完全相同，可直接比較計算量。
   ② 計算機：調 入力通道・出力通道・カーネル・特徴マップ大小，比較權重數與乘法次數。 */
Atlas.define("dwsep", {
  lead: `<p>標準卷積的每個濾波器同時處理「空間上的 3×3」和「所有通道」，通道一多，權重和計算量就爆增，手機跑不動。<span class="jp">Depthwise Separable Convolution</span>（深度可分離卷積，<span class="en">depthwise separable convolution</span>）把這件事拆成兩步：先各通道自己看空間，再用 1×1 把通道混合。</p>`,
  closing: `拆開後，空間的工作（Depthwise）和通道的工作（Pointwise）各做一次就好，不必為每個出力通道重做一遍。計算量約變成原本的 1/出力通道數 + 1/(カーネル邊長)²，3×3 時大約少 8〜9 倍；代價是能表達的濾波器種類受限。MobileNet 就是靠它把模型做小。`,

  mount(bench, ctx) {
    const { h, fmt, ui } = Atlas;
    const N = 5, K = 3, O = N - K + 1, CIN = 3, COUT = 4;
    const INP = [
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (c === 1 ? 1 : 0))), // 縱線
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (r === 3 ? 1 : 0))), // 橫線
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => (r === c ? 1 : 0))), // 斜線
    ];
    const CH = ["縱線", "橫線", "斜線"];
    const D = [
      { name: "縱向邊緣", w: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      { name: "橫向邊緣", w: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      { name: "十字加總", w: [[0, 1, 0], [1, 1, 1], [0, 1, 0]] },
    ];
    const PW = [[1, 0, 0], [0, 1, 1], [1, -1, 0], [-1, 0, 1]]; // PW[出力 o][通道 c]
    const STD = PW.map((p) => D.map((d, c) => d.w.map((row) => row.map((v) => p[c] * v)))); // STD[o][c] 3×3

    const win = (c, i, j) => { const v = []; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) v.push(INP[c][i + a][j + b]); return v; };
    const DW = D.map((d, c) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) => win(c, i, j).reduce((s, x, t) => s + x * d.w[Math.floor(t / K)][t % K], 0))));
    const OUT = PW.map((p) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) => p.reduce((s, pc, c) => s + pc * DW[c][i][j], 0))));
    const OUT_STD = STD.map((f) => Array.from({ length: O }, (_, i) => Array.from({ length: O }, (_, j) =>
      f.reduce((s, fc, c) => s + win(c, i, j).reduce((t, x, k) => t + x * fc[Math.floor(k / K)][k % K], 0), 0))));
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
    const viewSeg = ui.seg({ label: "看哪一部分", value: view, options: [
      { value: "std", label: "① 標準卷積（一步做完）" }, { value: "ds", label: "② Depthwise Separable（兩步）" }, { value: "calc", label: "③ 參數量計算機" }],
      onChange: (v) => { view = v; render(); } });
    const oSeg = ui.seg({ label: "看第幾個出力通道", value: oSel, options: [0, 1, 2, 3].map((v) => ({ value: v, label: `出力 ${v + 1}` })), onChange: (v) => { oSel = v; render(); } });
    const pick = (opts) => ui.seg(opts);
    const cinSeg = pick({ label: "入力通道數", value: cin, options: [3, 16, 64, 128, 256, 512].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { cin = v; render(); } });
    const coutSeg = pick({ label: "出力通道數", value: cout, options: [4, 16, 64, 128, 256, 512].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { cout = v; render(); } });
    const kSeg = pick({ label: "カーネル邊長", value: kk, options: [3, 5, 7].map((v) => ({ value: v, label: `${v}×${v}` })), onChange: (v) => { kk = v; render(); } });
    const hSeg = pick({ label: "特徴マップ邊長", value: hh, options: [3, 7, 14, 28, 56, 112].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { hh = v; render(); } });
    const presets = h("div", { class: "btn-group" },
      ui.btn("這頁的小例子（3→4、出力 3×3）", () => { cin = 3; cout = 4; kk = 3; hh = 3; sync(); render(); }, { sm: true }),
      ui.btn("MobileNet 中段（256→256、14×14）", () => { cin = 256; cout = 256; kk = 3; hh = 14; sync(); render(); }, { sm: true }));
    function sync() { cinSeg.set(cin); coutSeg.set(cout); kSeg.set(kk); hSeg.set(hh); }

    const rowSmall = h("div", { class: "bench__row" }, oSeg.el);
    const rowCalc = h("div", { class: "bench__row" }, cinSeg.el, coutSeg.el, kSeg.el, hSeg.el);
    const rowPre = h("div", { class: "field" }, h("span", { class: "field__label" }, "快速選擇"), presets);
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
          "aria-label": `${opt.name || ""}第 ${r + 1} 列第 ${c + 1} 行，值 ${fi(v)}` };
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
    const inputs = () => INP.map((m, c) => block(`通道 ${c + 1}<br>${CH[c]}`, grid(m, { size: "ds-s1", name: `入力通道 ${c + 1} `, win: [pi, pj], max: 1.6 })));
    const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
    const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };

    function countBars(a) {
      const items = [
        ["權重數", a.wStd, a.wDs, `${a.kk}×${a.kk}×${a.cin}×${a.cout}`, `${a.kk}×${a.kk}×${a.cin} + ${a.cin}×${a.cout}`],
        ["乘法次數", a.mStd, a.mDs, `${a.hh}×${a.hh} 格 × ${a.kk * a.kk * a.cin} × ${a.cout}`, `${a.hh}×${a.hh} 格 × (${a.kk * a.kk}×${a.cin} + ${a.cin}×${a.cout})`],
      ];
      const kids = [];
      for (const [lab, std, ds, fs, fd] of items) {
        kids.push(h("div", { class: "ds-h" }, lab));
        kids.push(h("span", null, "標準卷積"), h("div", { class: "ds-bar ds-bar--std", style: { width: "100%" }, title: fs }, std.toLocaleString()));
        const pct = Math.max(0.6, (ds / std) * 100), inside = pct > 30;
        kids.push(h("span", null, "DS"), h("div", { style: { width: "100%", display: "flex", alignItems: "center", gap: "6px" } },
          h("div", { class: "ds-bar ds-bar--ds", style: { width: `${pct}%` }, title: fd }, inside ? ds.toLocaleString() : ""),
          inside ? null : h("span", { class: "mono", style: { fontSize: "12px" } }, ds.toLocaleString())));
      }
      bars.replaceChildren(...kids);
    }
    const counts = (cin, cout, kk, hh) => ({ cin, cout, kk, hh,
      wStd: kk * kk * cin * cout, wDs: kk * kk * cin + cin * cout,
      mStd: hh * hh * kk * kk * cin * cout, mDs: hh * hh * (kk * kk * cin + cin * cout) });

    /* ---------- views ---------- */
    function renderStd() {
      const f = STD[oSel];
      const filt = f.map((m, c) => block(`第 ${c + 1} 片`, grid(m, { size: "ds-s1", name: `濾波器第 ${c + 1} 片 `, max: 1.6 })));
      const outs = OUT_STD.map((m, o) => block(`出力 ${o + 1}`, grid(m, { name: `出力通道 ${o + 1} `, cur: [pi, pj], onPick: pickPx, max: 6 }), o === oSel ? "" : "ds-dim"));
      fig.replaceChildren(h("div", { class: "ds-stage" },
        h("div", { class: "ds-lab", html: `入力 5×5×3 <span>（橘框＝出力那一格看的範圍）</span>` }), h("div", { class: "ds-row" }, ...inputs()),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: `出力 ${oSel + 1} 的濾波器 3×3×3 <span>（一次看完 3 個通道，27 個權重）</span>` }), h("div", { class: "ds-row" }, ...filt),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: `出力 3×3×4 <span>（每個出力通道各有一個這樣的濾波器）</span>` }), h("div", { class: "ds-row" }, ...outs)));
      const parts = [0, 1, 2].map((c) => win(c, pi, pj).reduce((t, x, k) => t + x * f[c][Math.floor(k / K)][k % K], 0));
      calc.innerHTML = row(`出力 ${oSel + 1}、第 ${pi + 1} 列第 ${pj + 1} 格`, `3 個通道各 9 個乘積（共 27 個）分別加總：${parts.map(fiP).join(" + ")} = <span class="hl">${fi(OUT_STD[oSel][pi][pj])}</span>`) +
        row("整層", `4 個出力通道 × 9 格，每格都要做 27 次乘法`);
      set("warn", `標準卷積：4 個濾波器 × 27 個權重 = 108 個權重；乘法 9 格 × 4 × 27 = 972 次`);
      hint.textContent = "點出力的格子，換一格看；上面的分段鈕換出力通道。";
      countBars(counts(CIN, COUT, K, O));
      note.innerHTML = `這個標準卷積的權重刻意設成「② 的 Pointwise 係數 × Depthwise 濾波器」，所以兩邊的出力完全相同，方便比較計算量。一般訓練出的標準卷積權重可以是任意組合，DS 只能表示其中這一類。`;
    }
    function renderDs() {
      const dws = DW.map((m, c) => block(`通道 ${c + 1} 的結果`, grid(m, { name: `Depthwise 通道 ${c + 1} `, cur: [pi, pj], onPick: pickPx, max: 3 })));
      const kerns = D.map((d, c) => block(`通道 ${c + 1} 專用<br>${d.name}`, grid(d.w, { size: "ds-s1", name: `Depthwise 濾波器 ${c + 1} `, max: 1.6 })));
      const pw = h("div", { class: "ds-block" },
        h("table", { class: "tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, "1×1 係數"), ...[1, 2, 3].map((c) => h("th", null, `通道 ${c}`)))),
          h("tbody", null, ...PW.map((p, o) => h("tr", { class: o === oSel ? "is-sel" : "" }, h("td", { class: "l" }, `出力 ${o + 1}`), ...p.map((v) => h("td", null, fi(v))))))),
        h("div", { class: "ds-cap" }, "Pointwise 的權重：4 組 × 3 個"));
      const outs = OUT.map((m, o) => block(`出力 ${o + 1}`, grid(m, { name: `出力通道 ${o + 1} `, cur: [pi, pj], onPick: pickPx, max: 6 }), o === oSel ? "" : "ds-dim"));
      fig.replaceChildren(h("div", { class: "ds-stage" },
        h("div", { class: "ds-lab", html: `入力 5×5×3` }), h("div", { class: "ds-row" }, ...inputs()),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: `① Depthwise：每個通道只用自己的 3×3 <span>（3 × 9 = 27 個權重）</span>` }), h("div", { class: "ds-row" }, ...kerns),
        h("div", { class: "ds-row" }, ...dws),
        h("div", { class: "ds-arrow", "aria-hidden": "true" }, "↓"),
        h("div", { class: "ds-lab", html: `② Pointwise：1×1 卷積，在每一格把 3 個通道加權混合 <span>（4 × 3 = 12 個權重）</span>` }), pw,
        h("div", { class: "ds-row" }, ...outs)));
      const d = [0, 1, 2].map((c) => DW[c][pi][pj]), p = PW[oSel];
      calc.innerHTML =
        row(`① 第 ${pi + 1} 列第 ${pj + 1} 格`, d.map((v, c) => `通道 ${c + 1}：9 個乘積加總 = ${fi(v)}`).join("；")) +
        row(`② 混成出力 ${oSel + 1}`, `${p.map((pc, c) => `${fiP(pc)}×${fiP(d[c])}`).join(" + ")} = <span class="hl">${fi(OUT[oSel][pi][pj])}</span>`) +
        row("和 ① 標準卷積比", `出力完全相同（${OUT.every((m, o) => m.every((r, i) => r.every((v, j) => v === OUT_STD[o][i][j]))) ? "4 個通道 9 格全部一致" : "不一致"}）`);
      set("ok", `兩步合計 27 + 12 = 39 個權重（標準 108）；乘法 9×3×9 + 9×4×3 = 351 次（標準 972）`);
      hint.textContent = "點任一張 3×3 結果的格子換位置；上面的分段鈕換出力通道。";
      countBars(counts(CIN, COUT, K, O));
      note.innerHTML = `Depthwise 只看空間、不混通道；Pointwise 只混通道、不看空間。1×1 卷積也叫 <span class="jp">Pointwise Convolution</span>。小例子只有 4 個出力通道，省得不多；通道越多省越多，到 ③ 試試。`;
    }
    function renderCalc() {
      const a = counts(cin, cout, kk, hh);
      const ratio = a.wDs / a.wStd;
      fig.replaceChildren();
      calc.innerHTML =
        row("標準", `權重 ${kk}×${kk}×${cin}×${cout} = ${a.wStd.toLocaleString()}`) +
        row("DS", `權重 ${kk}×${kk}×${cin} + ${cin}×${cout} = ${a.wDs.toLocaleString()}`) +
        row("比例", `DS ÷ 標準 = 1/${cout} + 1/${kk}² = ${fmt(1 / cout, 4)} + ${fmt(1 / (kk * kk), 4)} = <span class="hl">${fmt(ratio, 3)}</span>　<span class="k">（乘法次數的比例也一樣）</span>`);
      set(ratio < 0.2 ? "ok" : ratio < 0.5 ? "warn" : "bad", `DS 只需要標準卷積的 ${fmt(ratio * 100, 1)}%，約少 ${fmt(1 / ratio, 1)} 倍`);
      hint.textContent = "";
      countBars(a);
      note.innerHTML = `乘法次數假設ストライド 1、パディング讓特徴マップ大小不變。出力通道多時 1/出力通道數 幾乎是 0，比例就由 1/(カーネル邊長)² 決定：3×3 約 1/9。`;
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
