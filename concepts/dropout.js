/* ドロップアウト（シラバス §14）
   小網路：入力 3 → 隱藏層 6（ReLU）→ 出力 1（線性），只在隱藏層做ドロップアウト。
   ① 訓練：每次抽新的遮罩，記錄這次的出力；比較「抽到的出力平均」與「推論時全開（調整比例）的出力」。
   ② 全部子網路：6 個神經元的開／關共 2⁶ = 64 種，依機率加權平均 = 推論時的出力（這個網路裡剛好完全相等）。
   權重是示範用的設計值；所有出力即時計算。 */
Atlas.define("dropout", {
  lead: `<p>大網路很容易<a href="#overfit">過擬合</a>：神經元之間會互相配合，一起把訓練資料背起來。訓練好幾個不同的模型再把答案平均（<span class="jp">アンサンブル学習</span>，集成學習，<span class="en">ensemble learning</span>）能減輕這個問題，但大網路訓練一個就很久。<span class="jp">ドロップアウト</span>（丟棄法，<span class="en">dropout</span>）在訓練的每一步隨機讓一部分神經元輸出 0，等於每次都在訓練一個不同的小網路，而且它們共用同一組權重。</p>`,
  closing: `訓練時每次關掉不同的神經元，每個神經元就不能依賴特定的夥伴，必須自己學到有用的特徵；推論時全部打開，再把輸出的比例調回訓練時的平均，結果相當於把所有子網路的答案平均起來。所以ドロップアウト是一種<span class="jp">正則化</span>，而且只在訓練時使用。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const X = [1.0, 0.5, -1.0];
    const W1 = [[0.6, 0.4, -0.4], [0.2, 0.8, 0.2], [0.5, -0.2, -0.5], [1.0, 0.6, -0.2], [-0.2, 0.4, -0.6], [0.3, 1.0, -0.2]];
    const A = W1.map((w) => Math.max(0, w.reduce((t, v, i) => t + v * X[i], 0)));
    const W2 = [0.8, -0.5, 0.6, 0.4, -0.7, 0.9], B2 = 0.1, NH = 6;
    const FULL = W2.reduce((t, w, j) => t + w * A[j], 0); // 不含偏置的加總
    let p = 0.5, scheme = "inv", mode = "train", mask = null, draws = [], sel = null;
    const R = (() => { let seed = 20250101; return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();

    const keep = () => 1 - p;
    /** 訓練時某個遮罩的出力 */
    const trainOut = (m) => W2.reduce((t, w, j) => t + w * m[j] * A[j], 0) * (scheme === "inv" ? 1 / keep() : 1) + B2;
    /** 推論時（全開）的出力 */
    const inferOut = () => FULL * (scheme === "inv" ? 1 : keep()) + B2;
    const rawOut = () => FULL + B2; // 全開但不調整比例
    const allMasks = Array.from({ length: 64 }, (_, k) => Array.from({ length: NH }, (_, j) => (k >> (NH - 1 - j)) & 1));
    const prob = (m) => m.reduce((t, v) => t * (v ? keep() : p), 1);

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

    const modeSeg = ui.seg({ label: "模式", value: mode, options: [{ value: "train", label: "① 訓練：每次隨機關掉" }, { value: "all", label: "② 全部 64 個子網路" }], onChange: (v) => { mode = v; layout(); } });
    const pSl = ui.slider({ id: "do-p", label: "丟棄率 p（每個神經元被關掉的機率）", min: 0.1, max: 0.9, step: 0.1, value: p, format: (v) => v.toFixed(1), onInput: (v) => { p = Math.round(v * 10) / 10; draws = []; mask = null; render(); } });
    const schSeg = ui.seg({ label: "比例怎麼調整", value: scheme, options: [{ value: "inv", label: "訓練時 ÷ (1 − p)" }, { value: "orig", label: "推論時 × (1 − p)" }], onChange: (v) => { scheme = v; draws = draws.map((d) => ({ m: d.m, z: trainOut(d.m) })); render(); } });
    const net = s("svg", { viewBox: "0 0 400 330", role: "img", "aria-label": "網路圖：被關掉的神經元以虛線表示", style: "display:block;width:100%;min-width:320px;max-width:520px" });
    const strip = s("svg", { viewBox: "0 0 600 120", role: "img", "aria-label": "每次抽到的出力", style: "display:block;width:100%;min-width:480px;max-width:720px" });
    const grid = h("div", { class: "do-grid" });
    const stats = h("div", { class: "do-stats" });
    const status = h("div", { class: "status" });
    const calc = h("div", { class: "calc" });
    const player = ui.player(ctx, () => { draw1(); return draws.length < 2000; }, 260);
    const trainRow = h("div", { class: "bench__row" },
      ui.btn("訓練一次（抽新的遮罩）", () => { player.pause(); draw1(); }, { primary: true }),
      player.el,
      ui.btn("連抽 100 次", () => { player.pause(); for (let i = 0; i < 100; i++) draw1(true); render(); }),
      ui.btn("重設", () => { player.pause(); draws = []; mask = null; render(); }));
    const netBox = h("div", { class: "scroll-x" }, net);
    const stripBox = h("div", null, h("p", { class: "field__label" }, "每次訓練時的出力（一個點＝一次）"), h("div", { class: "scroll-x" }, strip));
    const gridBox = h("div", null, h("p", { class: "field__label" }, "64 種開／關組合（藍點＝開著的神經元；點一格看它的計算）"), grid);
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
      net.append(s("text", { x: xi, y: 316, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: "入力" }));
      net.append(s("text", { x: xh, y: 330 - 6, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: m ? `隱藏層（ReLU）：開著 ${m.reduce((a, b) => a + b, 0)} 個` : "隱藏層（ReLU）：全部開著" }));
      net.append(s("text", { x: xo, y: yo + 46, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: m ? "訓練時的出力" : "推論時的出力" }));
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
      mk(inf, "var(--ok)", `推論時 ${fmt(inf, 2)}`, 110, infRight ? "start" : "end");
      if (scheme === "orig") mk(raw, "var(--bad)", `不調整就全開 ${fmt(raw, 2)}`, 110, infRight ? "end" : "start");
      if (draws.length) {
        const mean = draws.reduce((a, d) => a + d.z, 0) / draws.length;
        strip.append(s("path", { d: `M${px(mean)},2 l-6,-0 l6,9 l6,-9z`, fill: "var(--hot)" }));
        strip.append(s("text", { x: px(mean) + 8, y: 12, "font-size": 11.5, "font-weight": 700, class: "t-halo", fill: "var(--hot)", text: `平均 ${fmt(mean, 3)}` }));
      }
    }

    /* ---------- 64 格 ---------- */
    function drawGrid() {
      const os = outs(), mx = Math.max(...os.map((v) => Math.abs(v - inferOut()))) || 1;
      grid.replaceChildren(...allMasks.map((m, k) => {
        const b = h("button", { type: "button", class: "do-cell", "aria-pressed": String(sel === k), title: `機率 ${fmt(prob(m) * 100, 2)}%`, style: { background: Atlas.divColor(os[k] - inferOut(), mx * 1.4) } },
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
        stats.innerHTML = `<span>已訓練 <b>${n}</b> 次</span><span>出力的平均 <b>${n ? fmt(mean, 3) : "—"}</b></span><span>推論時（全開${scheme === "inv" ? "" : "後 × " + fmt(keep(), 1)}） <b>${fmt(inf, 3)}</b></span>`;
        if (!n) { status.className = "status status--info"; status.textContent = "按「訓練一次」：每個神經元各自以機率 p 被關掉"; }
        else if (n < 30) { status.className = "status status--info"; status.textContent = `這次關掉了 ${NH - mask.reduce((a, b) => a + b, 0)} 個，出力 ${fmt(draws[n - 1].z, 2)}；每次都不一樣，多抽幾次看平均`; }
        else if (Math.abs(mean - inf) < 0.05) { status.className = "status status--ok"; status.textContent = `抽了 ${n} 次，平均 ${fmt(mean, 3)}，已經很接近推論時全開的 ${fmt(inf, 3)}`; }
        else { status.className = "status status--info"; status.textContent = `抽了 ${n} 次，平均 ${fmt(mean, 3)}，和推論時全開的 ${fmt(inf, 3)} 還差 ${fmt(Math.abs(mean - inf), 3)}；抽得越多，平均越接近`; }
        if (!mask) {
          calc.innerHTML = `<span class="calc-row"><span class="k">推論時（全開）</span> z = (${W2.map((w, j) => `${fmtP(w, 1)}×${fmt(A[j], 1)}`).join(" + ")})${scheme === "orig" ? ` × ${fmt(keep(), 1)}` : ""} + ${fmt(B2, 1)} = <b>${fmt(inf, 3)}</b></span>`;
        } else {
          calc.innerHTML =
            `<span class="calc-row"><span class="k">這次的遮罩</span> m = (${mask.join(", ")})　<span class="k">（1＝留下，0＝關掉）</span></span>` +
            `<span class="calc-row"><span class="k">訓練時的出力</span> z = (${W2.map((w, j) => mask[j] ? `${fmtP(w, 1)}×${fmt(A[j], 1)}` : `<span class="k">0</span>`).join(" + ")})${scaleTxt ? " " + scaleTxt : ""} + ${fmt(B2, 1)} = <span class="hl">${fmt(draws[draws.length - 1].z, 3)}</span></span>` +
            `<span class="calc-row"><span class="k">推論時（全開）</span> z = (${W2.map((w, j) => `${fmtP(w, 1)}×${fmt(A[j], 1)}`).join(" + ")})${scheme === "orig" ? ` × ${fmt(keep(), 1)}` : ""} + ${fmt(B2, 1)} = <b>${fmt(inf, 3)}</b></span>`;
        }
      } else {
        drawNet(sel == null ? null : allMasks[sel]); drawGrid();
        const os = outs(), avg = allMasks.reduce((t, m, k) => t + prob(m) * os[k], 0);
        status.className = "status status--ok";
        status.textContent = `64 個子網路依出現機率加權平均：${fmt(avg, 4)}；推論時全開：${fmt(inf, 4)}，完全相同`;
        const eq = p === 0.5 ? `每一種的機率都是 0.5⁶ = 1/64` : `關掉 k 個的組合，機率是 ${fmt(p, 1)}ᵏ × ${fmt(keep(), 1)}⁽⁶⁻ᵏ⁾`;
        calc.innerHTML =
          `<span class="calc-row"><span class="k">加權平均</span> Σ 機率 × 出力（${eq}）= <span class="hl">${fmt(avg, 4)}</span></span>` +
          (sel != null ? `<span class="calc-row"><span class="k">選中的這一個</span> m = (${allMasks[sel].join(", ")})　機率 ${fmt(prob(allMasks[sel]) * 100, 3)}%　出力 z = (${W2.map((w, j) => allMasks[sel][j] ? `${fmtP(w, 1)}×${fmt(A[j], 1)}` : `<span class="k">0</span>`).join(" + ")})${scaleTxt ? " " + scaleTxt : ""} + ${fmt(B2, 1)} = <b>${fmt(os[sel], 3)}</b></span>` : `<span class="calc-row"><span class="k">點上面任一格，看那個子網路的計算</span></span>`) +
          `<span class="calc-row"><span class="k">推論時（全開）</span> ${scheme === "inv" ? "" : "× " + fmt(keep(), 1) + " 後 "}z = <b>${fmt(inf, 4)}</b></span>`;
      }
    }

    note.innerHTML = `兩種調整比例的方法結果相同：原始論文是推論時把出力乘上 (1 − p)；現在的框架多半改成訓練時把留下的值除以 (1 − p)（反轉ドロップアウト，<span class="en">inverted dropout</span>），推論時就什麼都不用改。如果不調整，推論時全部打開，下一層收到的訊號會比訓練時大很多（「推論時 × (1 − p)」模式下的紅線）。這個示範只有一層ドロップアウト、後面直接接線性的出力，所以平均剛好完全相等；層數多、中間有非線性時只是近似，但實務上效果很好。6 個神經元就有 64 種子網路；真實網路有上千個神經元，子網路的數量多到天文數字，每一步只訓練其中一個。常見的丟棄率是隱藏層 0.5、入力層 0.2 左右。隱藏層的值是用示範權重算出的（入力 × 權重再過 ReLU）。`;

    layout();
    ctx.onTheme(() => render());
    return { destroy() { player.pause(); } };
  },
});
