/* ROC 曲線と AUC（シラバス §10）
   假想的檢查：模型對每個人輸出「是陽性的機率」，超過しきい値（門檻）就判為陽性。
   資料：陽性、陰性各有一組固定的潛在分數 z（固定亂數），模型輸出 = σ(z ± d/2)；d 越大，兩類分得越開（模型越好）。
   調門檻 → 混同行列、正解率・適合率・再現率・F 値、偽陽性率即時更新；ROC 曲線上的點跟著移動。
   AUC 用梯形面積算，並用「隨機抽一個陽性和一個陰性，陽性分數較高的比例」驗算。
   文字在 locales/zh/roc.js、locales/ja/roc.js。 */
(function () {
  const ZP = [-0.32, -0.99, 0.68, 0.73, -0.63, 0.74, 1.49, 0.12, -1.36, 0.93, 0.75, -3.1, 0.29, -0.54, -0.72, 0.13, -0.68, -0.48, 2.48, 0.22];
  const ZN = [0.32, 1.08, -0.16, -0.31, 1.08, 0.25, 0.23, -0.89, -0.13, -1.02, 0.65, 0.02, 0.61, -0.79, 2, -0.14, 0.53, 0.54, -2.2, -1.63,
    -1.35, 0.3, 1.24, -0.07, 0.89, 0.65, 0.61, -0.88, -1.43, 0.7, 1.35, -1.77, 1.03, 0.66, 0.01, -0.96].map((z) => z + 0.003); // 加 0.003：避免和陽性剛好同分

  Atlas.define("roc", {
    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      const { t } = ctx;
      let dIdx = 2, thr = 0.5, data = "bal";
      const DS = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
      const sig = (z) => 1 / (1 + Math.exp(-z));
      bench.append(h("style", { text: `
        .rc-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; align-items: flex-start; }
        .rc-fig { flex: 1 1 300px; max-width: 480px; min-width: 0; }
        .rc-fig > svg { display: block; width: 100%; height: auto; }
        .rc-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
        .rc-cm { border-collapse: separate; border-spacing: 4px; font-size: 13px; margin-inline: auto; }
        .rc-cm th { font-weight: 500; color: var(--muted); font-size: 12px; padding: 2px 6px; }
        .rc-cm td { width: 120px; height: 56px; text-align: center; border-radius: 8px; border: 1px solid var(--line); line-height: 1.35; }
        .rc-cm td b { display: block; font-family: var(--f-mono); font-size: 18px; }
        .rc-cm td small { font-size: 11.5px; color: var(--muted); }
        .rc-cm .ok { background: var(--accent-soft); }
        .rc-cm .ng { background: var(--neg-soft); }
        .rc-strip { cursor: pointer; touch-action: none; }
      ` }));

      const dSl = ui.slider({ id: "rc-d", label: t("dLabel"), min: 0, max: DS.length - 1, step: 1, value: dIdx, format: (v) => t("dLevels").split("/")[v], onInput: (v) => { dIdx = v; draw(); } });
      const tSl = ui.slider({ id: "rc-t", label: t("tLabel"), min: 0, max: 1, step: 0.01, value: thr, format: (v) => v.toFixed(2), onInput: (v) => { thr = v; draw(); centerThr(); } });
      const dataSeg = ui.seg({ label: t("dataLabel"), value: data, options: [{ value: "bal", label: t("dataBal") }, { value: "rare", label: t("dataRare") }], onChange: (v) => { data = v; draw(); } });
      const strip = s("svg", { viewBox: "0 0 560 196", role: "img", class: "rc-strip", style: "display:block;width:100%;height:auto", "aria-label": t("stripAria") });
      const roc = s("svg", { viewBox: "0 0 330 330", role: "img", "aria-label": t("rocAria") });
      const cmBox = h("div");
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      bench.append(
        h("div", { class: "bench__row" }, dSl.el, dataSeg.el),
        h("div", { class: "bench__row" }, tSl.el),
        h("div", { class: "scroll-x" }, h("div", { style: { minWidth: "520px", maxWidth: "720px" } },
          h("div", { class: "rc-cap" }, t("stripCap")), strip)),
        status,
        h("div", { class: "scroll-x" }, h("div", { class: "rc-figs" },
          h("div", { class: "rc-fig", style: { flexBasis: "300px", maxWidth: "360px" } }, h("div", { class: "rc-cap" }, t("cmCap")), cmBox),
          h("div", { class: "rc-fig", style: { flexBasis: "300px", maxWidth: "380px" } }, h("div", { class: "rc-cap" }, t("rocCap")), roc))),
        calc, note);

      const SM = { l: 70, r: 16 }, sx = (v) => SM.l + v * (560 - SM.l - SM.r);
      let dragging = false;
      const setThrFromEvent = (e) => { const p = strip.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(strip.getScreenCTM().inverse()); thr = Math.round(Math.max(0, Math.min(1, (q.x - SM.l) / (560 - SM.l - SM.r))) * 100) / 100; tSl.set(thr); draw(); };
      strip.addEventListener("pointerdown", (e) => { dragging = true; try { strip.setPointerCapture(e.pointerId); } catch (_) {} setThrFromEvent(e); });
      strip.addEventListener("pointermove", (e) => { if (dragging) setThrFromEvent(e); });
      strip.addEventListener("pointerup", () => { dragging = false; }); strip.addEventListener("pointercancel", () => { dragging = false; });

      // 手機上分數條會橫向捲動：把門檻捲到中間
      function centerThr() { const sc = strip.parentNode && strip.parentNode.parentNode; if (sc && sc.scrollWidth > sc.clientWidth) sc.scrollLeft = (sx(thr) / 560) * strip.clientWidth - sc.clientWidth / 2; }
      function scores() {
        const d = DS[dIdx], np = data === "bal" ? 20 : 4, nn = data === "bal" ? 20 : 36;
        return { P: ZP.slice(0, np).map((z) => sig(z + d / 2)), N: ZN.slice(0, nn).map((z) => sig(z - d / 2)) };
      }
      // 疊放：同一列裡太近的點往上下錯開
      function lanes(vals) {
        const order = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), lastX = [], out = [];
        order.forEach(([v, i]) => { const x = sx(v); let L = 0; while (lastX[L] != null && x - lastX[L] < 11) L++; lastX[L] = x; out[i] = L; });
        return out;
      }

      function draw() {
        const { P, N } = scores(), np = P.length, nn = N.length;
        const TP = P.filter((v) => v >= thr).length, FN = np - TP, FP = N.filter((v) => v >= thr).length, TN = nn - FP;
        const tpr = TP / np, fpr = FP / nn, prec = TP + FP ? TP / (TP + FP) : NaN, acc = (TP + TN) / (np + nn), f1 = TP ? (2 * prec * tpr) / (prec + tpr) : 0;

        /* --- 分數條 --- */
        strip.replaceChildren();
        const rowY = { P: 52, N: 140 };
        strip.append(s("rect", { x: sx(thr), y: 8, width: sx(1) - sx(thr), height: 166, fill: "var(--hot-soft)" }));
        strip.append(s("text", { x: Math.min(sx(thr) + 6, 552), y: 22, "text-anchor": sx(thr) > 470 ? "end" : "start", "font-size": 11, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: t("stripPos") }));
        if (sx(thr) > 150) strip.append(s("text", { x: sx(thr) - 6, y: 22, "text-anchor": "end", "font-size": 11, class: "t-muted t-halo", text: t("stripNeg") }));
        [["P", t("rowP"), t("rowPSub")], ["N", t("rowN"), t("rowNSub")]].forEach(([k, a, b]) => {
          strip.append(s("line", { x1: SM.l, x2: sx(1), y1: rowY[k], y2: rowY[k], stroke: "var(--line)" }));
          strip.append(s("text", { x: 8, y: rowY[k] - 2, "font-size": 12.5, "font-weight": 700, text: a }));
          strip.append(s("text", { x: 8, y: rowY[k] + 14, "font-size": 10.5, class: "t-muted", text: b }));
        });
        for (let v = 0; v <= 1.0001; v += 0.25) strip.append(s("text", { x: sx(v), y: 190, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-muted", text: v.toFixed(2) }));
        const dots = (vals, k) => {
          const L = lanes(vals);
          vals.forEach((v, i) => {
            const y = rowY[k] + (L[i] % 2 ? 1 : -1) * Math.ceil(L[i] / 2) * 11, pos = v >= thr, right = k === "P" ? pos : !pos;
            const attrs = { fill: k === "P" ? "var(--pos)" : "var(--neg)", stroke: right ? "var(--panel)" : "var(--ink)", "stroke-width": right ? 1.5 : 2.2, "fill-opacity": right ? 1 : 0.45 };
            if (k === "P") strip.append(s("circle", { cx: sx(v), cy: y, r: 5.5, ...attrs }));
            else strip.append(s("rect", { x: sx(v) - 5, y: y - 5, width: 10, height: 10, rx: 1.5, ...attrs }));
          });
        };
        dots(P, "P"); dots(N, "N");
        strip.append(s("line", { x1: sx(thr), x2: sx(thr), y1: 8, y2: 174, stroke: "var(--hot)", "stroke-width": 2.5 }));

        /* --- 混同行列 --- */
        const cell = (n, name, en, cls) => h("td", { class: cls }, h("b", null, String(n)), h("span", { class: "term" }, name), h("br"), h("small", null, en));
        cmBox.replaceChildren(h("table", { class: "rc-cm" },
          h("thead", null, h("tr", null, h("th"), h("th", null, t("cmPredP")), h("th", null, t("cmPredN")))),
          h("tbody", null,
            h("tr", null, h("th", null, t("cmActP")), cell(TP, t("cmTP"), t("cmTPSub"), "ok"), cell(FN, t("cmFN"), t("cmFNSub"), "ng")),
            h("tr", null, h("th", null, t("cmActN")), cell(FP, t("cmFP"), t("cmFPSub"), "ng"), cell(TN, t("cmTN"), t("cmTNSub"), "ok")))));

        /* --- ROC 曲線 --- */
        const all = [...new Set([...P, ...N])].sort((a, b) => b - a);
        const curve = [[0, 0]];
        all.forEach((th) => curve.push([N.filter((v) => v >= th).length / nn, P.filter((v) => v >= th).length / np]));
        let auc = 0; for (let i = 1; i < curve.length; i++) auc += (curve[i][0] - curve[i - 1][0]) * (curve[i][1] + curve[i - 1][1]) / 2;
        let pairs = 0; P.forEach((p) => N.forEach((n) => { pairs += p > n ? 1 : p === n ? 0.5 : 0; }));
        const RM = { l: 44, t: 12, r: 12, b: 40 }, RS = 330 - RM.l - RM.r, rx = (v) => RM.l + v * RS, ry = (v) => RM.t + (1 - v) * (330 - RM.t - RM.b);
        roc.replaceChildren();
        roc.append(s("path", { d: `M${rx(0)},${ry(0)}` + curve.map((p) => `L${rx(p[0]).toFixed(1)},${ry(p[1]).toFixed(1)}`).join("") + `L${rx(1)},${ry(0)}Z`, fill: "var(--accent-soft)" }));
        for (let v = 0; v <= 1.0001; v += 0.25) {
          roc.append(s("line", { x1: rx(v), x2: rx(v), y1: ry(0), y2: ry(1), stroke: "var(--line)", "stroke-dasharray": "3 3" }));
          roc.append(s("line", { x1: rx(0), x2: rx(1), y1: ry(v), y2: ry(v), stroke: "var(--line)", "stroke-dasharray": "3 3" }));
          roc.append(s("text", { x: rx(v), y: ry(0) + 15, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: v.toFixed(2) }));
          roc.append(s("text", { x: rx(0) - 5, y: ry(v) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: v.toFixed(2) }));
        }
        roc.append(s("rect", { x: rx(0), y: ry(1), width: RS, height: ry(0) - ry(1), fill: "none", stroke: "var(--line-strong)" }));
        roc.append(s("line", { x1: rx(0), y1: ry(0), x2: rx(1), y2: ry(1), stroke: "var(--muted)", "stroke-dasharray": "6 4" }));
        roc.append(s("path", { d: curve.map((p, i) => (i ? "L" : "M") + rx(p[0]).toFixed(1) + "," + ry(p[1]).toFixed(1)).join("") + `L${rx(1)},${ry(1)}`, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5 }));
        roc.append(s("text", { x: rx(0.72), y: ry(0.62), "text-anchor": "middle", "font-size": 10.5, class: "t-muted t-halo", transform: `rotate(-45 ${rx(0.72)} ${ry(0.62)})`, text: t("rocRandom") }));
        roc.append(s("circle", { cx: rx(fpr), cy: ry(tpr), r: 6.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 }));
        roc.append(s("text", { x: rx(0.97), y: ry(0.05), "text-anchor": "end", "font-size": 13, "font-weight": 700, fill: "var(--accent)", class: "t-halo", text: `AUC = ${fmt(auc, 4)}` }));
        roc.append(s("text", { x: rx(0.5), y: 326, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: t("rocX") }));
        roc.append(s("text", { x: 12, y: ry(0.5), "text-anchor": "middle", "font-size": 11, class: "t-muted", transform: `rotate(-90 12 ${ry(0.5)})`, text: t("rocY") }));

        /* --- 狀態 --- */
        const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };
        if (data === "rare" && TP === 0 && acc >= 0.85) set("bad", t("statusRare", { acc: fmt(acc * 100, 0) }));
        else if (FN === 0 && FP === 0) set("ok", t("statusPerfect"));
        else if (tpr >= 0.9 && fpr >= 0.4) set("warn", t("statusLow", { tp: TP, np, fp: FP, nn }));
        else if (tpr <= 0.4 && fpr <= 0.1) set("warn", t("statusHigh", { fp: FP, nn, fn: FN, np }));
        else set("info", t("statusInfo", { thr: thr.toFixed(2), tp: TP, np, fp: FP, nn }));

        /* --- 算式 --- */
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        const pc = (v) => (isNaN(v) ? t("pcNone") : fmt(v * 100, 1) + "%");
        calc.innerHTML =
          row(t("calcAccK"), t("calcAcc", { tp: TP, tn: TN, n: np + nn, v: pc(acc) })) +
          row(t("calcPrecK"), t("calcPrec", { tp: TP, d: TP + FP, v: pc(prec) })) +
          row(t("calcRecK"), t("calcRec", { tp: TP, d: np, v: pc(tpr) })) +
          row(t("calcF1K"), isNaN(prec) || !TP ? t("calcF1None") : t("calcF1", { v: fmt(f1 * 100, 1) })) +
          row(t("calcFprK"), t("calcFpr", { fp: FP, d: nn, v: pc(fpr) })) +
          row(t("calcAucK"), t("calcAuc", { auc: fmt(auc, 4), pairs: np * nn, win: pairs, ratio: fmt(pairs / (np * nn), 4) }));
      }
      draw(); requestAnimationFrame(centerThr);
      note.innerHTML = t("note");
      return {};
    },
  });
})();
