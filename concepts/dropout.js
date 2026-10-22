/* Dropout（大綱 §14）
   小網路：輸入 3 → 隱藏層 6（ReLU）→ 輸出 1（線性），只在隱藏層做 Dropout。
   ① 訓練：每次抽新的遮罩，記錄這次的輸出；比較「抽到的輸出平均」與「推論時全開（調整比例）的輸出」。
   ② 全部子網路：6 個神經元的開／關共 2⁶ = 64 種，依機率加權平均 = 推論時的輸出（這個網路裡剛好完全相等）。
   權重是示範用的設計值；所有輸出即時計算。
   文字在 locales/zh/dropout.js、locales/ja/dropout.js。 */
Atlas.define("dropout", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const X = [1.0, 0.5, -1.0];
    const W1 = [[0.6, 0.4, -0.4], [0.2, 0.8, 0.2], [0.5, -0.2, -0.5], [1.0, 0.6, -0.2], [-0.2, 0.4, -0.6], [0.3, 1.0, -0.2]];
    const A = W1.map((w) => Math.max(0, w.reduce((acc, v, i) => acc + v * X[i], 0)));
    const W2 = [0.8, -0.5, 0.6, 0.4, -0.7, 0.9], B2 = 0.1, NH = 6;
    const FULL = W2.reduce((acc, w, j) => acc + w * A[j], 0); // 不含偏置的加總
    let p = 0.5, scheme = "inv", mode = "train", mask = null, draws = [], sel = null;
    const R = (() => { let seed = 20250101; return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();

    const keep = () => 1 - p;
    /** 訓練時某個遮罩的輸出 */
    const trainOut = (m) => W2.reduce((acc, w, j) => acc + w * m[j] * A[j], 0) * (scheme === "inv" ? 1 / keep() : 1) + B2;
    /** 推論時（全開）的輸出 */
    const inferOut = () => FULL * (scheme === "inv" ? 1 : keep()) + B2;
    const rawOut = () => FULL + B2; // 全開但不調整比例
    const allMasks = Array.from({ length: 64 }, (_, k) => Array.from({ length: NH }, (_, j) => (k >> (NH - 1 - j)) & 1));
    const prob = (m) => m.reduce((acc, v) => acc * (v ? keep() : p), 1);

    bench.append(h("style", { text: `
      .do-grid { display: grid; grid-template-columns: repeat(8, minmax(36px, 1fr)); gap: 4px; max-width: 560px; margin-inline: auto; }
      .do-cell { border: 1px solid var(--line); border-radius: 6px; padding: 3px 2px 2px; display: flex; flex-direction: column; align-items: center; gap: 1px; cursor: pointer; font-family: var(--f-mono); font-size: 10.5px; color: var(--ink); background: var(--panel); }
      .do-cell:hover { border-color: var(--accent); }
      .do-cell[aria-pressed="true"] { outline: 2.5px solid var(--hot); outline-offset: 1px; }
      .do-dots { display: grid; grid-template-columns: repeat(3, 7px); gap: 2px; }
      .do-dots i { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); }
      .do-dots i.off { background: transparent; border: 1px dashed var(--faint); }
      .do-stats { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }
      .do-stats b { font-family: var(--f-mono); color: var(--ink); font-size: 15px; }
    ` }));

    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode, options: [{ value: "train", label: t("modeTrain") }, { value: "all", label: t("modeAll") }], onChange: (v) => { mode = v; layout(); } });
    const pSl = ui.slider({ id: "do-p", label: t("pLabel"), min: 0.1, max: 0.9, step: 0.1, value: p, format: (v) => v.toFixed(1), onInput: (v) => { p = Math.round(v * 10) / 10; draws = []; mask = null; render(); } });
    const schSeg = ui.seg({ label: t("schLabel"), value: scheme, options: [{ value: "inv", label: t("schInv") }, { value: "orig", label: t("schOrig") }], onChange: (v) => { scheme = v; draws = draws.map((d) => ({ m: d.m, z: trainOut(d.m) })); render(); } });
    const net = s("svg", { viewBox: "0 0 400 330", role: "img", "aria-label": t("netAria"), style: "display:block;width:100%;min-width:320px;max-width:520px" });
    const strip = s("svg", { viewBox: "0 0 600 120", role: "img", "aria-label": t("stripAria"), style: "display:block;width:100%;min-width:480px;max-width:720px" });
    const grid = h("div", { class: "do-grid" });
    const stats = h("div", { class: "do-stats" });
    const status = h("div", { class: "status" });
    const calc = h("div", { class: "calc" });
    const player = ui.player(ctx, () => { draw1(); return draws.length < 2000; }, 260);
    const trainRow = h("div", { class: "bench__row" },
      ui.btn(t("btnTrainOnce"), () => { player.pause(); draw1(); }, { primary: true }),
      player.el,
      ui.btn(t("btnTrain100"), () => { player.pause(); for (let i = 0; i < 100; i++) draw1(true); render(); }),
      ui.btn(t("btnReset"), () => { player.pause(); draws = []; mask = null; render(); }));
    const netBox = h("div", { class: "scroll-x" }, net);
    const stripBox = h("div", null, h("p", { class: "field__label" }, t("stripLabel")), h("div", { class: "scroll-x" }, strip));
    const gridBox = h("div", null, h("p", { class: "field__label" }, t("gridLabel")), grid);
    const note = h("p", { class: "note" });
    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el, schSeg.el),
      h("div", { class: "bench__row" }, pSl.el),
      netBox, status, trainRow, stats, stripBox, gridBox, calc, note);

    function draw1(silent) {
      mask = Array.from({ length: NH }, () => (R() < p ? 0 : 1));
      draws.push({ m: mask, z: trainOut(mask) });
      if (!silent) render();
    }
    function layout() { trainRow.hidden = stripBox.hidden = stats.hidden = mode !== "train"; gridBox.hidden = mode !== "all"; player.pause(); render(); }

    /* ---------- 網路圖 ---------- */
    function drawNet(m) {
      const xi = 36, xh = 188, xo = 352, yi = [100, 165, 230], yh = (j) => 40 + j * 50, yo = 165;
      net.replaceChildren();
      const on = (j) => !m || m[j] === 1;
      for (let j = 0; j < NH; j++) for (let i = 0; i < 3; i++)
        net.append(s("line", { x1: xi, y1: yi[i], x2: xh, y2: yh(j), stroke: "var(--line-strong)", "stroke-width": 1, opacity: on(j) ? 0.8 : 0.15 }));
      for (let j = 0; j < NH; j++) {
        if (!on(j)) continue;
        net.append(s("line", { x1: xh, y1: yh(j), x2: xo, y2: yo, stroke: W2[j] >= 0 ? "var(--pos)" : "var(--neg)", "stroke-width": 1 + 2.5 * Math.abs(W2[j]) }));
      }
      X.forEach((v, i) => {
        net.append(s("circle", { cx: xi, cy: yi[i], r: 17, fill: "var(--panel-2)", stroke: "var(--line-strong)", "stroke-width": 1.5 }));
        net.append(s("text", { x: xi, y: yi[i] + 4.5, "text-anchor": "middle", "font-size": 12.5, class: "t-mono", text: fmt(v, 1) }));
      });
      for (let j = 0; j < NH; j++) {
        const o = on(j), y = yh(j);
        net.append(s("circle", { cx: xh, cy: y, r: 19, fill: o ? "var(--accent-soft)" : "var(--panel)", stroke: o ? "var(--accent)" : "var(--faint)", "stroke-width": 1.8, "stroke-dasharray": o ? null : "4 3" }));
        if (o) net.append(s("text", { x: xh, y: y + 4.5, "text-anchor": "middle", "font-size": 12.5, "font-weight": 600, class: "t-mono", text: fmt(A[j], 1) }));
        else net.append(s("text", { x: xh, y: y + 5, "text-anchor": "middle", "font-size": 15, fill: "var(--faint)", text: "×" }));
        net.append(s("text", { x: xh - 24, y: y - 10, "text-anchor": "end", "font-size": 11, class: "t-mono t-halo", fill: W2[j] >= 0 ? "var(--pos)" : "var(--neg)", text: "w=" + fmt(W2[j], 1) }));
      }
      const z = m ? trainOut(m) : inferOut();
      net.append(s("circle", { cx: xo, cy: yo, r: 26, fill: "var(--hot-soft)", stroke: "var(--hot)", "stroke-width": 2 }));
      net.append(s("text", { x: xo, y: yo + 5, "text-anchor": "middle", "font-size": 14, "font-weight": 700, class: "t-mono", text: fmt(z, 2) }));
      net.append(s("text", { x: xi, y: 316, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: t("netIn") }));
      net.append(s("text", { x: xh, y: 330 - 6, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: m ? t("netHidOn", { n: m.reduce((a, b) => a + b, 0) }) : t("netHidAll") }));
      net.append(s("text", { x: xo, y: yo + 46, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: t(m ? "netOutTrain" : "netOutInfer") }));
    }

    /* ---------- 點狀圖 ---------- */
    const outs = () => allMasks.map(trainOut);
    function drawStrip() {
      const W = 600, H = 120, M = { l: 16, r: 16 };
      const all = outs().concat([inferOut(), rawOut()]);
      const lo = Math.min(...all) - 0.2, hi = Math.max(...all) + 0.2;
      const px = (v) => M.l + ((v - lo) / (hi - lo)) * (W - M.l - M.r);
      strip.replaceChildren();
      strip.append(s("line", { x1: M.l, x2: W - M.r, y1: 70, y2: 70, stroke: "var(--line-strong)" }));
      for (let v = Math.ceil(lo); v <= hi; v++) {
        strip.append(s("line", { x1: px(v), x2: px(v), y1: 66, y2: 74, stroke: "var(--line-strong)" }));
        strip.append(s("text", { x: px(v), y: 90, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v < 0 ? "−" + -v : v }));
      }
      // 同一個值的點往上疊
      const cnt = new Map();
      draws.forEach((d, i) => {
        const key = d.z.toFixed(4), c = cnt.get(key) || 0; cnt.set(key, c + 1);
        const last = i === draws.length - 1;
        strip.append(s("circle", { cx: px(d.z), cy: 62 - Math.min(c, 14) * 3.6, r: last ? 4.5 : 3, fill: last ? "var(--hot)" : "var(--accent)", opacity: last ? 1 : 0.55 }));
      });
      const mk = (v, col, label, y, anchor) => {
        strip.append(s("line", { x1: px(v), x2: px(v), y1: 8, y2: 74, stroke: col, "stroke-width": 2, "stroke-dasharray": "5 3" }));
        strip.append(s("text", { x: px(v) + (anchor === "end" ? -4 : 4), y, "text-anchor": anchor, "font-size": 11.5, "font-weight": 700, class: "t-halo", fill: col, text: label }));
      };
      const inf = inferOut(), raw = rawOut();
      const infRight = inf >= raw;
      mk(inf, "var(--ok)", t("mkInfer", { v: fmt(inf, 2) }), 110, infRight ? "start" : "end");
      if (scheme === "orig") mk(raw, "var(--bad)", t("mkRaw", { v: fmt(raw, 2) }), 110, infRight ? "end" : "start");
      if (draws.length) {
        const mean = draws.reduce((a, d) => a + d.z, 0) / draws.length;
        strip.append(s("path", { d: `M${px(mean)},2 l-6,-0 l6,9 l6,-9z`, fill: "var(--hot)" }));
        strip.append(s("text", { x: px(mean) + 8, y: 12, "font-size": 11.5, "font-weight": 700, class: "t-halo", fill: "var(--hot)", text: t("mkMean", { v: fmt(mean, 3) }) }));
      }
    }

    /* ---------- 64 格 ---------- */
    function drawGrid() {
      const os = outs(), mx = Math.max(...os.map((v) => Math.abs(v - inferOut()))) || 1;
      grid.replaceChildren(...allMasks.map((m, k) => {
        const b = h("button", { type: "button", class: "do-cell", "aria-pressed": String(sel === k), title: t("cellTitle", { p: fmt(prob(m) * 100, 2) }), style: { background: Atlas.divColor(os[k] - inferOut(), mx * 1.4) } },
          h("span", { class: "do-dots" }, ...m.map((v) => h("i", { class: v ? "" : "off" }))),
          fmt(os[k], 2));
        b.addEventListener("click", () => { sel = sel === k ? null : k; render(); });
        return b;
      }));
    }

    /* ---------- 文字 ---------- */
    function render() {
      const inf = inferOut();
      const scaleTxt = scheme === "inv" ? `÷ ${fmt(keep(), 1)}` : "";
      if (mode === "train") {
        drawNet(mask); drawStrip();
        const n = draws.length, mean = n ? draws.reduce((a, d) => a + d.z, 0) / n : null;
        stats.innerHTML = `<span>${t("statN", { n })}</span><span>${t("statMean", { m: n ? fmt(mean, 3) : "—" })}</span><span>${scheme === "inv" ? t("statInferInv", { v: fmt(inf, 3) }) : t("statInferOrig", { k: fmt(keep(), 1), v: fmt(inf, 3) })}</span>`;
        const sv = { n, mean: n ? fmt(mean, 3) : "", inf: fmt(inf, 3), diff: n ? fmt(Math.abs(mean - inf), 3) : "" };
        if (!n) { status.className = "status status--info"; status.textContent = t("sIdle"); }
        else if (n < 30) { status.className = "status status--info"; status.textContent = t("sFew", { off: NH - mask.reduce((a, b) => a + b, 0), z: fmt(draws[n - 1].z, 2) }); }
        else if (Math.abs(mean - inf) < 0.05) { status.className = "status status--ok"; status.textContent = t("sClose", sv); }
        else { status.className = "status status--info"; status.textContent = t("sFar", sv); }
        if (!mask) {
          calc.innerHTML = `<span class="calc-row"><span class="k">${t("calcInferK")}</span> z = (${W2.map((w, j) => `${fmtP(w, 1)}×${fmt(A[j], 1)}`).join(" + ")})${scheme === "orig" ? ` × ${fmt(keep(), 1)}` : ""} + ${fmt(B2, 1)} = <b>${fmt(inf, 3)}</b></span>`;
        } else {
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcMaskK")}</span> m = (${mask.join(", ")})　<span class="k">${t("calcMaskNote")}</span></span>` +
            `<span class="calc-row"><span class="k">${t("netOutTrain")}</span> z = (${W2.map((w, j) => mask[j] ? `${fmtP(w, 1)}×${fmt(A[j], 1)}` : `<span class="k">0</span>`).join(" + ")})${scaleTxt ? " " + scaleTxt : ""} + ${fmt(B2, 1)} = <span class="hl">${fmt(draws[draws.length - 1].z, 3)}</span></span>` +
            `<span class="calc-row"><span class="k">${t("calcInferK")}</span> z = (${W2.map((w, j) => `${fmtP(w, 1)}×${fmt(A[j], 1)}`).join(" + ")})${scheme === "orig" ? ` × ${fmt(keep(), 1)}` : ""} + ${fmt(B2, 1)} = <b>${fmt(inf, 3)}</b></span>`;
        }
      } else {
        drawNet(sel == null ? null : allMasks[sel]); drawGrid();
        const os = outs(), avg = allMasks.reduce((acc, m, k) => acc + prob(m) * os[k], 0);
        status.className = "status status--ok";
        status.textContent = t("sAll", { avg: fmt(avg, 4), inf: fmt(inf, 4) });
        const eq = p === 0.5 ? t("eqHalf") : t("eqGen", { p: fmt(p, 1), k: fmt(keep(), 1) });
        calc.innerHTML =
          `<span class="calc-row"><span class="k">${t("calcAvgK")}</span> ${t("calcAvg", { eq })} = <span class="hl">${fmt(avg, 4)}</span></span>` +
          (sel != null ? `<span class="calc-row"><span class="k">${t("calcSelK")}</span> m = (${allMasks[sel].join(", ")})　${t("calcSelProb", { p: fmt(prob(allMasks[sel]) * 100, 3) })}　${t("calcSelOut")} z = (${W2.map((w, j) => allMasks[sel][j] ? `${fmtP(w, 1)}×${fmt(A[j], 1)}` : `<span class="k">0</span>`).join(" + ")})${scaleTxt ? " " + scaleTxt : ""} + ${fmt(B2, 1)} = <b>${fmt(os[sel], 3)}</b></span>` : `<span class="calc-row"><span class="k">${t("calcPickHint")}</span></span>`) +
          `<span class="calc-row"><span class="k">${t("calcInferK")}</span> ${scheme === "inv" ? `z = <b>${fmt(inf, 4)}</b>` : t("calcInferScaled", { k: fmt(keep(), 1), v: fmt(inf, 4) })}</span>`;
      }
    }

    note.innerHTML = t("note");

    layout();
    ctx.onTheme(() => render());
    return { destroy() { player.pause(); } };
  },
});
