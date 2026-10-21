/* ROC 曲線と AUC（シラバス §10）
   假想的檢查：模型對每個人輸出「是陽性的機率」，超過しきい値（門檻）就判為陽性。
   資料：陽性、陰性各有一組固定的潛在分數 z（固定亂數），模型輸出 = σ(z ± d/2)；d 越大，兩類分得越開（模型越好）。
   調門檻 → 混同行列、正解率・適合率・再現率・F 値、偽陽性率即時更新；ROC 曲線上的點跟著移動。
   AUC 用梯形面積算，並用「隨機抽一個陽性和一個陰性，陽性分數較高的比例」驗算。 */
(function () {
  const ZP = [-0.32, -0.99, 0.68, 0.73, -0.63, 0.74, 1.49, 0.12, -1.36, 0.93, 0.75, -3.1, 0.29, -0.54, -0.72, 0.13, -0.68, -0.48, 2.48, 0.22];
  const ZN = [0.32, 1.08, -0.16, -0.31, 1.08, 0.25, 0.23, -0.89, -0.13, -1.02, 0.65, 0.02, 0.61, -0.79, 2, -0.14, 0.53, 0.54, -2.2, -1.63,
    -1.35, 0.3, 1.24, -0.07, 0.89, 0.65, 0.61, -0.88, -1.43, 0.7, 1.35, -1.77, 1.03, 0.66, 0.01, -0.96].map((z) => z + 0.003); // 加 0.003：避免和陽性剛好同分

  Atlas.define("roc", {
    lead: `<p>分類模型通常輸出一個「是陽性的機率」，要再切一條<span class="jp">しきい値</span>（門檻，<span class="en">threshold</span>）才變成「陽性／陰性」的判定。門檻放低，真正的陽性抓得多，但誤報也多；放高則相反。只看某一個門檻的正解率，看不出模型本身好不好。<span class="jp">ROC 曲線</span>把所有門檻的結果畫在一起，<span class="jp">AUC</span> 再把它濃縮成一個數。</p>`,
    closing: `門檻決定「抓多少、誤報多少」的取捨，要看任務選（漏掉病人代價大，就把門檻放低、重視再現率）；ROC 曲線和 AUC 則和門檻無關，衡量的是模型把陽性排在陰性前面的能力：AUC 1 是完美分開，0.5 和亂猜一樣。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
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

      const dSl = ui.slider({ id: "rc-d", label: "模型的區分能力（兩類分數拉開多少）", min: 0, max: DS.length - 1, step: 1, value: dIdx, format: (v) => ["和亂猜一樣", "很弱", "弱", "普通", "不錯", "很好", "優秀", "完美分開"][v], onInput: (v) => { dIdx = v; draw(); } });
      const tSl = ui.slider({ id: "rc-t", label: "しきい値（分數 ≥ 門檻就判為陽性）", min: 0, max: 1, step: 0.01, value: thr, format: (v) => v.toFixed(2), onInput: (v) => { thr = v; draw(); centerThr(); } });
      const dataSeg = ui.seg({ label: "資料", value: data, options: [{ value: "bal", label: "陽性 20、陰性 20" }, { value: "rare", label: "陽性很少（4 對 36）" }], onChange: (v) => { data = v; draw(); } });
      const strip = s("svg", { viewBox: "0 0 560 196", role: "img", class: "rc-strip", style: "display:block;width:100%;height:auto", "aria-label": "每個人的模型分數與門檻（點圖可移動門檻）" });
      const roc = s("svg", { viewBox: "0 0 330 330", role: "img", "aria-label": "ROC 曲線" });
      const cmBox = h("div");
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      bench.append(
        h("div", { class: "bench__row" }, dSl.el, dataSeg.el),
        h("div", { class: "bench__row" }, tSl.el),
        h("div", { class: "scroll-x" }, h("div", { style: { minWidth: "520px", maxWidth: "720px" } },
          h("div", { class: "rc-cap" }, "每個點＝一個人，橫軸是模型給的陽性機率；點圖或拖動可以移動門檻"), strip)),
        status,
        h("div", { class: "scroll-x" }, h("div", { class: "rc-figs" },
          h("div", { class: "rc-fig", style: { flexBasis: "300px", maxWidth: "360px" } }, h("div", { class: "rc-cap" }, "混同行列（這個門檻的結果）"), cmBox),
          h("div", { class: "rc-fig", style: { flexBasis: "300px", maxWidth: "380px" } }, h("div", { class: "rc-cap" }, "ROC 曲線：門檻從 1 降到 0 時，點從左下走到右上"), roc))),
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
        strip.append(s("text", { x: Math.min(sx(thr) + 6, 552), y: 22, "text-anchor": sx(thr) > 470 ? "end" : "start", "font-size": 11, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: "判為陽性 →" }));
        if (sx(thr) > 150) strip.append(s("text", { x: sx(thr) - 6, y: 22, "text-anchor": "end", "font-size": 11, class: "t-muted t-halo", text: "← 判為陰性" }));
        [["P", "陽性", "（真的有）"], ["N", "陰性", "（真的沒有）"]].forEach(([k, a, b]) => {
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
        const cell = (n, name, en, cls) => h("td", { class: cls }, h("b", null, String(n)), h("span", { html: `<span class="jp">${name}</span>` }), h("br"), h("small", null, en));
        cmBox.replaceChildren(h("table", { class: "rc-cm" },
          h("thead", null, h("tr", null, h("th"), h("th", null, "判為陽性"), h("th", null, "判為陰性"))),
          h("tbody", null,
            h("tr", null, h("th", null, "實際陽性"), cell(TP, "真陽性", "TP", "ok"), cell(FN, "偽陰性", "FN（漏掉）", "ng")),
            h("tr", null, h("th", null, "實際陰性"), cell(FP, "偽陽性", "FP（誤報）", "ng"), cell(TN, "真陰性", "TN", "ok")))));

        /* --- ROC 曲線 --- */
        const all = [...new Set([...P, ...N])].sort((a, b) => b - a);
        const curve = [[0, 0]];
        all.forEach((t) => curve.push([N.filter((v) => v >= t).length / nn, P.filter((v) => v >= t).length / np]));
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
        roc.append(s("text", { x: rx(0.72), y: ry(0.62), "text-anchor": "middle", "font-size": 10.5, class: "t-muted t-halo", transform: `rotate(-45 ${rx(0.72)} ${ry(0.62)})`, text: "亂猜（AUC 0.5）" }));
        roc.append(s("circle", { cx: rx(fpr), cy: ry(tpr), r: 6.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 }));
        roc.append(s("text", { x: rx(0.97), y: ry(0.05), "text-anchor": "end", "font-size": 13, "font-weight": 700, fill: "var(--accent)", class: "t-halo", text: `AUC = ${fmt(auc, 4)}` }));
        roc.append(s("text", { x: rx(0.5), y: 326, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: "偽陽性率 FPR（陰性被誤報的比例）" }));
        roc.append(s("text", { x: 12, y: ry(0.5), "text-anchor": "middle", "font-size": 11, class: "t-muted", transform: `rotate(-90 12 ${ry(0.5)})`, text: "真陽性率 TPR（＝再現率）" }));

        /* --- 狀態 --- */
        const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };
        if (data === "rare" && TP === 0 && acc >= 0.85) set("bad", `全部判為陰性也有正解率 ${fmt(acc * 100, 0)}%，但 4 個陽性一個都沒抓到（再現率 0%）：陽性很少時，正解率會騙人`);
        else if (FN === 0 && FP === 0) set("ok", "這個門檻剛好把兩類完全分開：沒有漏掉、也沒有誤報");
        else if (tpr >= 0.9 && fpr >= 0.4) set("warn", `門檻很低：陽性抓到 ${TP} / ${np}，但陰性也有 ${FP} / ${nn} 被誤報`);
        else if (tpr <= 0.4 && fpr <= 0.1) set("warn", `門檻很高：幾乎不誤報（${FP} / ${nn}），但漏掉 ${FN} / ${np} 個陽性`);
        else set("info", `門檻 ${thr.toFixed(2)}：抓到 ${TP} / ${np} 個陽性，誤報 ${FP} / ${nn} 個陰性。拖動門檻，看 ROC 曲線上的點怎麼移動`);

        /* --- 算式 --- */
        const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
        const pc = (v) => (isNaN(v) ? "—（沒有判為陽性的）" : fmt(v * 100, 1) + "%");
        calc.innerHTML =
          row("正解率", `(TP + TN) ÷ 全部 = (${TP} + ${TN}) ÷ ${np + nn} = <span class="hl">${pc(acc)}</span>`) +
          row("適合率", `TP ÷ (TP + FP) = ${TP} ÷ ${TP + FP} = <span class="hl">${pc(prec)}</span>　<span class="k">判為陽性的人裡，真的是陽性的比例</span>`) +
          row("再現率", `TP ÷ (TP + FN) = ${TP} ÷ ${np} = <span class="hl">${pc(tpr)}</span>　<span class="k">真正的陽性裡，被抓到的比例（＝ROC 的縱軸）</span>`) +
          row("F 値", isNaN(prec) || !TP ? "—（適合率或再現率是 0）" : `2 × 適合率 × 再現率 ÷ (適合率 + 再現率) = <span class="hl">${fmt(f1 * 100, 1)}%</span>　<span class="k">兩者的調和平均</span>`) +
          row("偽陽性率", `FP ÷ (FP + TN) = ${FP} ÷ ${nn} = <span class="hl">${pc(fpr)}</span>　<span class="k">（ROC 的橫軸）</span>`) +
          row("AUC", `曲線下的面積 = <span class="hl">${fmt(auc, 4)}</span>；驗算：陽性 × 陰性共 ${np * nn} 對，陽性分數比較高的有 ${pairs} 對 → ${pairs} ÷ ${np * nn} = ${fmt(pairs / (np * nn), 4)}`);
      }
      draw(); requestAnimationFrame(centerThr);
      note.innerHTML = `示範資料是假想的檢查結果（固定亂數）。陽性用藍色圓點、陰性用紅色方塊，判錯的點畫成半透明加黑框。門檻往左移，ROC 曲線上的點往右上走（抓得多也誤報得多）；不管門檻放哪，點都在同一條曲線上，所以曲線屬於模型、不屬於某個門檻。AUC（<span class="en">Area Under the Curve</span>，曲線下面積）恰好等於「隨機抽一個陽性和一個陰性，模型給陽性的分數比較高」的機率。陽性很少的資料，正解率容易很高卻毫無用處，要一起看適合率與再現率。這些指標都要用<b>沒參與訓練的資料</b>計算（<span class="jp">ホールドアウト検証</span>、<span class="jp">交差検証</span>），才代表<a href="#overfit">汎化性能</a>。輸出機率的模型怎麼訓練見<a href="#loss">誤差関数</a>的交差エントロピー。`;
      return {};
    },
  });
})();
