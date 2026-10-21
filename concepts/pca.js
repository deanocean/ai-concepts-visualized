/* 主成分分析（PCA）（シラバス §8）
   ① 找方向：20 個 2D 點（已減去平均），轉動一條軸，把點投影上去；投影的變異數最大的方向＝第 1 主成分。
      變異數 = 保留的部分 + 丟掉的部分（垂直距離² 的平均），兩者加起來固定。第 1 主成分由共變異數矩陣的固有値・固有ベクトル直接算出。
   ② 5 科成績：30 位學生 × 5 科（固定亂數產生後寫死）。Jacobi 法求固有値，畫寄与率與累積寄与率；
      點任一位學生，看只用前 k 個主成分還原的成績差多少。全部即時計算。 */
(function () {
  const P2 = [[-0.42, 0.23], [0.26, 0.33], [-0.49, 0.03], [-2.47, -1.69], [-0.36, 0.08], [-1.25, -0.41], [-1.07, -0.16], [0.2, -0.24], [-1.13, -0.22], [2.05, 0.77],
    [0.46, 0.33], [-1.11, 0.02], [-1.07, -0.89], [0.28, 0.5], [2.33, 0.84], [0.38, -0.04], [-0.61, -0.58], [3.74, 1.57], [0.26, 0.3], [0.07, -0.68]];
  const SUBJ = ["国語", "英語", "社会", "数学", "理科"];
  const S = [[77, 77, 79, 66, 62], [80, 79, 85, 65, 68], [23, 39, 25, 29, 38], [76, 82, 73, 64, 68], [54, 44, 58, 58, 61], [63, 66, 62, 58, 55], [53, 54, 59, 61, 58], [52, 61, 58, 64, 59], [57, 69, 69, 68, 60], [63, 60, 59, 62, 62],
    [51, 54, 53, 28, 39], [58, 57, 60, 50, 50], [56, 47, 47, 52, 57], [70, 72, 76, 85, 72], [57, 51, 57, 71, 66], [43, 55, 46, 42, 48], [39, 36, 37, 52, 41], [60, 58, 60, 54, 58], [53, 55, 49, 75, 67], [36, 37, 42, 42, 49],
    [76, 75, 64, 77, 86], [51, 50, 56, 46, 42], [52, 56, 52, 63, 59], [83, 71, 80, 53, 60], [85, 72, 78, 81, 76], [67, 70, 73, 71, 80], [32, 51, 41, 72, 48], [47, 48, 47, 51, 34], [79, 79, 83, 88, 86], [67, 66, 64, 79, 62]];
  // 原始資料的 5 科順序是 国語・英語・社会・数学・理科

  function jacobi(M) {
    const n = M.length, A = M.map((r) => [...r]), V = M.map((_, i) => M.map((_, j) => +(i === j)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
      }
    }
    const ev = A.map((r, i) => r[i]), idx = ev.map((_, i) => i).sort((a, b) => ev[b] - ev[a]);
    return { val: idx.map((i) => ev[i]), vec: idx.map((i) => V.map((r) => r[i])) };
  }

  Atlas.define("pca", {
    lead: `<p>資料的特徴（項目）一多，就難以畫圖、計算也變重，而且很多項目其實彼此相關、說的是同一件事。<span class="jp">主成分分析</span>（主成分分析，<span class="en">principal component analysis, PCA</span>）找出資料「散得最開」的方向，用少數幾個新的軸來表示資料，盡量不丟掉資訊，這叫<span class="jp">次元削減</span>（降維，<span class="en">dimensionality reduction</span>）。</p>`,
    closing: `第 1 主成分是資料變異最大的方向，第 2 主成分是和它垂直的方向中變異最大的，依此類推；每個方向保留的變異占全體的比例叫<span class="jp">寄与率</span>（貢獻率），從大的加起來就是<span class="jp">累積寄与率</span>。累積寄与率夠高時，只留前幾個主成分，就能用更少的數字代表原本的資料。`,

    mount(bench, ctx) {
      const { h, s, fmt, fmtP, ui } = Atlas;
      let mode = "dir";
      const modeSeg = ui.seg({ label: "模式", value: mode, options: [{ value: "dir", label: "① 找方向（2 維 → 1 維）" }, { value: "subj", label: "② 5 科成績（5 維）" }], onChange: (v) => { mode = v; show(); } });
      bench.append(h("style", { text: `
        .pc-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; align-items: flex-start; }
        .pc-fig { flex: 1 1 300px; max-width: 470px; min-width: 0; }
        .pc-fig > svg { display: block; width: 100%; height: auto; }
        .pc-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
        .pc-st { cursor: pointer; }
        .pc-bar { display: flex; height: 22px; border-radius: 6px; overflow: hidden; border: 1px solid var(--line-strong); }
        .pc-bar > div { display: grid; place-items: center; font-size: 12px; font-family: var(--f-mono); white-space: nowrap; overflow: hidden; }
      ` }));
      const box = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } });
      const note = h("p", { class: "note" });
      bench.append(h("div", { class: "bench__row" }, modeSeg.el), box);
      function show() { box.replaceChildren(); MODES[mode](); box.append(note); }
      const setStatus = (el, k, t) => { el.className = "status status--" + k; el.textContent = t; };
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;

      /* ===================== ① 找方向 ===================== */
      function modeDir() {
        const n = P2.length;
        const sxx = P2.reduce((a, p) => a + p[0] * p[0], 0) / n, syy = P2.reduce((a, p) => a + p[1] * p[1], 0) / n, sxy = P2.reduce((a, p) => a + p[0] * p[1], 0) / n;
        const tot = sxx + syy, disc = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy), l1 = tot / 2 + disc, l2 = tot / 2 - disc;
        const a1 = ((Math.atan2(2 * sxy, sxx - syy) / 2) * 180) / Math.PI; // 第 1 主成分的角度
        const pc1 = ((a1 % 180) + 180) % 180;
        const varAt = (deg) => { const t = (deg * Math.PI) / 180, c = Math.cos(t), sn = Math.sin(t); return sxx * c * c + 2 * sxy * c * sn + syy * sn * sn; };
        let ang = 160, exact = false;
        const sl = ui.slider({ id: "pc-a", label: "軸的方向", min: 0, max: 179, step: 1, value: ang, format: (v) => (exact ? fmt(pc1, 1) : v) + "°", onInput: (v) => { ang = v; exact = false; draw(); } });
        const b1 = ui.btn("跳到第 1 主成分", () => { exact = true; ang = pc1; sl.set(Math.round(pc1)); draw(); }, { primary: true, sm: true });
        const b2 = ui.btn("跳到第 2 主成分", () => { exact = true; ang = (pc1 + 90) % 180; sl.set(Math.round(ang)); draw(); }, { sm: true });
        const main = s("svg", { viewBox: "0 0 440 330", role: "img", "aria-label": "資料點與它們在軸上的投影" });
        const chart = s("svg", { viewBox: "0 0 360 240", role: "img", "aria-label": "軸的方向與投影的變異數" });
        const bar = h("div", { class: "pc-bar", role: "img" });
        const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
        box.append(
          h("div", { class: "bench__row" }, sl.el, b1, b2),
          h("div", { class: "scroll-x" }, h("div", { class: "pc-figs" },
            h("div", { class: "pc-fig", style: { flexBasis: "400px", maxWidth: "520px" } }, h("div", { class: "pc-cap" }, "● 原本的點　○ 投影到軸上的位置　虛線＝投影時丟掉的距離"), main),
            h("div", { class: "pc-fig", style: { flexBasis: "300px", maxWidth: "400px" } }, h("div", { class: "pc-cap" }, "每個方向上，投影後的變異數（散開程度）"), chart))),
          h("div", { class: "field" }, h("span", { class: "field__label" }, "全體的變異數 = 投影保留的 + 丟掉的（兩者加總固定）"), bar),
          status, calc);

        const U = 54, OX = 220, OY = 165;
        const X = (x) => OX + x * U, Y = (y) => OY - y * U;
        function draw() {
          const deg = exact ? ang : ang, t = (deg * Math.PI) / 180, u = [Math.cos(t), Math.sin(t)];
          const v = varAt(deg), lost = tot - v;
          main.replaceChildren(s("defs", null, s("clipPath", { id: "pcclip" }, s("rect", { x: 10, y: 6, width: 420, height: 318 }))));
          for (let k = -4; k <= 4; k++) {
            main.append(s("line", { x1: X(k), x2: X(k), y1: 6, y2: 324, stroke: k ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": k ? "2 4" : null }));
            if (k >= -2 && k <= 2) main.append(s("line", { x1: 10, x2: 430, y1: Y(k), y2: Y(k), stroke: k ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": k ? "2 4" : null }));
          }
          main.append(s("text", { x: 424, y: OY - 6, "text-anchor": "end", "font-size": 11, class: "t-muted t-halo", text: "特徴 1" }));
          main.append(s("text", { x: OX + 6, y: 20, "font-size": 11, class: "t-muted t-halo", text: "特徴 2" }));
          const g = s("g", { "clip-path": "url(#pcclip)" });
          g.append(s("line", { x1: X(-9 * u[0]), y1: Y(-9 * u[1]), x2: X(9 * u[0]), y2: Y(9 * u[1]), stroke: "var(--hot)", "stroke-width": 2.5 }));
          P2.forEach((p) => {
            const z = p[0] * u[0] + p[1] * u[1], q = [z * u[0], z * u[1]];
            g.append(s("line", { x1: X(p[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(q[1]), stroke: "var(--neg)", "stroke-width": 1.3, "stroke-dasharray": "3 2" }));
          });
          P2.forEach((p) => { const z = p[0] * u[0] + p[1] * u[1]; g.append(s("circle", { cx: X(z * u[0]), cy: Y(z * u[1]), r: 4, fill: "var(--panel)", stroke: "var(--hot)", "stroke-width": 1.8 })); });
          P2.forEach((p) => g.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 5, fill: "var(--ink)" })));
          main.append(g);
          // 軸的標籤：放在軸上方、遠離資料的一端
          const end = u[1] >= 0 ? 1 : -1, lp = [end * 3.4 * u[0], end * 3.4 * u[1]];
          main.append(s("text", { x: Math.max(40, Math.min(400, X(lp[0]))), y: Math.max(20, Math.min(318, Y(lp[1]) - 10)), "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: "新的軸" }));

          drawChart(deg, v);
          const keep = v / tot;
          bar.replaceChildren(
            h("div", { style: { width: keep * 100 + "%", background: "var(--hot-soft)", color: "var(--ink)" } }, keep > 0.18 ? `保留 ${fmt(v, 3)}（${fmt(keep * 100, 1)}%）` : ""),
            h("div", { style: { width: (1 - keep) * 100 + "%", background: "var(--neg-soft)", color: "var(--ink)" } }, keep < 0.82 ? `丟掉 ${fmt(lost, 3)}（${fmt((1 - keep) * 100, 1)}%）` : fmt((1 - keep) * 100, 1) + "%"));
          bar.setAttribute("aria-label", `保留 ${fmt(keep * 100, 1)}%，丟掉 ${fmt((1 - keep) * 100, 1)}%`);

          const isPC1 = Math.abs(v - l1) < 1e-3, isPC2 = Math.abs(v - l2) < 1e-3;
          if (isPC1) setStatus(status, "ok", `第 1 主成分：投影後保留 ${fmt(keep * 100, 1)}% 的變異（寄与率），只用這 1 個數幾乎就能代表原本的 2 個數`);
          else if (isPC2) setStatus(status, "warn", `第 2 主成分：和第 1 主成分垂直，只保留 ${fmt(keep * 100, 1)}%，大部分資訊都丟掉了`);
          else setStatus(status, keep > 0.9 ? "info" : "warn", `這個方向保留 ${fmt(keep * 100, 1)}% 的變異；第 1 主成分最多能保留 ${fmt((l1 / tot) * 100, 1)}%`);
          calc.innerHTML =
            row("共變異數矩陣", `[[${fmt(sxx, 3)}, ${fmt(sxy, 3)}], [${fmt(sxy, 3)}, ${fmt(syy, 3)}]]　<span class="k">（對角線＝各特徴的變異數，加總 ${fmt(tot, 3)}）</span>`) +
            row("這個方向", `u = (${fmt(u[0], 3)}, ${fmt(u[1], 3)})，投影的變異數 = ${fmt(sxx, 3)}·${fmt(u[0], 3)}² + 2·${fmt(sxy, 3)}·${fmtP(u[0], 3)}·${fmtP(u[1], 3)} + ${fmt(syy, 3)}·${fmt(u[1], 3)}² = <span class="hl">${fmt(v, 3)}</span>`) +
            row("直接算", `共變異數矩陣的固有値 λ₁ = ${fmt(l1, 3)}、λ₂ = ${fmt(l2, 3)}（加總也是 ${fmt(tot, 3)}）；λ₁ 的固有ベクトル指向 ${fmt(pc1, 1)}°＝第 1 主成分`) +
            row("寄与率", `第 1 主成分 ${fmt(l1, 3)} ÷ ${fmt(tot, 3)} = ${fmt((l1 / tot) * 100, 1)}%，第 2 主成分 ${fmt(l2, 3)} ÷ ${fmt(tot, 3)} = ${fmt((l2 / tot) * 100, 1)}%`);
        }
        function drawChart(deg, v) {
          const CW = 360, CH = 240, m = { l: 40, r: 14, t: 16, b: 34 }, top = 2.5;
          const cx = (a) => m.l + (a / 180) * (CW - m.l - m.r), cy = (val) => CH - m.b - (val / top) * (CH - m.t - m.b);
          chart.replaceChildren();
          [0, 0.5, 1, 1.5, 2, 2.5].forEach((val) => {
            chart.append(s("line", { x1: m.l, x2: CW - m.r, y1: cy(val), y2: cy(val), stroke: val ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": val ? "3 3" : null }));
            chart.append(s("text", { x: m.l - 5, y: cy(val) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: fmt(val, 1) }));
          });
          for (let a = 0; a <= 180; a += 45) chart.append(s("text", { x: cx(a), y: CH - m.b + 15, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: a + "°" }));
          chart.append(s("text", { x: CW - m.r, y: CH - 4, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: "軸的方向" }));
          chart.append(s("line", { x1: m.l, x2: CW - m.r, y1: cy(tot), y2: cy(tot), stroke: "var(--muted)", "stroke-dasharray": "6 4" }));
          chart.append(s("text", { x: CW - m.r, y: cy(tot) - 5, "text-anchor": "end", "font-size": 10.5, class: "t-muted t-halo", text: `全體 ${fmt(tot, 3)}` }));
          let d = ""; for (let k = 0; k <= 180; k++) d += (k ? "L" : "M") + cx(k).toFixed(1) + "," + cy(varAt(k)).toFixed(1);
          chart.append(s("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 2.2 }));
          const p2 = (pc1 + 90) % 180;
          [[pc1, l1, "第 1 主成分"], [p2, l2, "第 2 主成分"]].forEach(([a, val, lab]) => {
            chart.append(s("circle", { cx: cx(a), cy: cy(val), r: 4, fill: "var(--ok)" }));
            const right = a > 120, above = val > 1;
            chart.append(s("text", { x: cx(a) + (right ? -7 : 7), y: cy(val) + (above ? -8 : -8), "text-anchor": right ? "end" : "start", "font-size": 10.5, "font-weight": 700, fill: "var(--ok)", class: "t-halo", text: `${lab} ${fmt(val, 3)}` }));
          });
          chart.append(s("line", { x1: cx(deg), x2: cx(deg), y1: m.t, y2: CH - m.b, stroke: "var(--hot)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
          chart.append(s("circle", { cx: cx(deg), cy: cy(v), r: 5.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 }));
        }
        draw();
        note.innerHTML = `點已經事先減去平均（中心在原點）。投影到軸上，就是用「沿著軸走了多遠」這 1 個數取代原本的 2 個數；虛線的長度是被丟掉的部分。因為每個點都是「投影 + 垂直距離」的直角三角形，保留的變異和丟掉的平方距離加起來永遠等於全體，所以「保留最多」和「丟掉最少」是同一件事。答案不必一個個角度試：共變異數矩陣的<span class="jp">固有ベクトル</span>（特徵向量，<span class="en">eigenvector</span>）就是這些方向，<span class="jp">固有値</span>（特徵值）就是各方向保留的變異數。`;
      }

      /* ===================== ② 5 科成績 ===================== */
      function modeSubj() {
        const n = S.length, d = 5;
        const mean = SUBJ.map((_, j) => S.reduce((a, r) => a + r[j], 0) / n);
        const Cv = SUBJ.map((_, i) => SUBJ.map((_, j) => S.reduce((a, r) => a + (r[i] - mean[i]) * (r[j] - mean[j]), 0) / n));
        const E = jacobi(Cv), tot = E.val.reduce((a, v) => a + v, 0);
        // 方向的正負沒有意義：第 1 主成分讓係數加總為正，第 2 主成分讓「数学」為正
        E.vec[0] = E.vec[0].reduce((a, v) => a + v, 0) < 0 ? E.vec[0].map((v) => -v) : E.vec[0];
        E.vec[1] = E.vec[1][3] < 0 ? E.vec[1].map((v) => -v) : E.vec[1];
        for (let k = 2; k < d; k++) if (E.vec[k].reduce((a, v) => a + v, 0) < 0) E.vec[k] = E.vec[k].map((v) => -v);
        const score = (r) => E.vec.map((v) => v.reduce((a, w, j) => a + w * (r[j] - mean[j]), 0));
        const Z = S.map(score);
        let keep = 2, sel = 26;
        const sl = ui.slider({ id: "pc-k", label: "保留幾個主成分", min: 1, max: 5, step: 1, value: keep, format: (v) => v + " 個", onInput: (v) => { keep = v; draw(); } });
        const bars = s("svg", { viewBox: "0 0 360 230", role: "img", "aria-label": "各主成分的寄与率與累積寄与率" });
        const scat = s("svg", { viewBox: "0 0 360 210", role: "img", "aria-label": "每位學生在第 1、第 2 主成分上的位置" });
        const load = h("div", { class: "scroll-x" });
        const status = h("div", { class: "status" }), calc = h("div", { class: "scroll-x" });
        box.append(
          h("div", { class: "bench__row" }, sl.el),
          h("div", { class: "scroll-x" }, h("div", { class: "pc-figs" },
            h("div", { class: "pc-fig" }, h("div", { class: "pc-cap" }, "長條＝寄与率，折線＝累積寄与率"), bars),
            h("div", { class: "pc-fig" }, h("div", { class: "pc-cap" }, "30 位學生（點一位，看還原結果）"), scat, h("div", { class: "pc-cap" }, "橫軸＝第 1 主成分（越右 5 科普遍越高）　縱軸＝第 2 主成分（越上数学・理科相對越強）")))),
          status,
          h("div", { class: "field" }, h("span", { class: "field__label" }, "每個主成分由 5 科怎麼組合而成（固有ベクトル的成分）"), load),
          h("div", { class: "field" }, h("span", { class: "field__label" }, "選中的學生：只用保留的主成分還原 5 科成績"), calc));

        // 主成分係數表（固定）
        const tbl = h("table", { class: "tbl" });
        tbl.append(h("thead", null, h("tr", null, h("th", { class: "l" }, ""), ...SUBJ.map((t) => h("th", null, t)), h("th", null, "寄与率"))));
        const tb = h("tbody");
        E.vec.forEach((v, k) => tb.append(h("tr", { "data-k": k }, h("td", { class: "l" }, `第 ${k + 1} 主成分`), ...v.map((w) => h("td", { style: { color: Math.abs(w) >= 0.3 ? (w > 0 ? "var(--pos)" : "var(--neg)") : "var(--faint)" } }, fmt(w, 2))), h("td", null, fmt((E.val[k] / tot) * 100, 1) + "%"))));
        tbl.append(tb); load.append(tbl);

        const SC = Math.min(165 / Math.max(...Z.map((z) => Math.abs(z[0]))), 90 / Math.max(...Z.map((z) => Math.abs(z[1])))); // 兩軸同比例
        const SX = (z) => 180 + z * SC, SY = (z) => 105 - z * SC;
        scat.addEventListener("click", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i != null) { sel = +i; draw(); } });
        scat.addEventListener("keydown", (e) => { const i = e.target.dataset && e.target.dataset.i; if (i != null && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sel = +i; draw(); scat.querySelector(`[data-i="${i}"]`).focus(); } });

        function draw() {
          /* --- 寄与率 --- */
          bars.replaceChildren();
          const m = { l: 38, r: 12, t: 14, b: 30 }, CW = 360, CH = 230, py = (p) => CH - m.b - (p / 100) * (CH - m.t - m.b), bw = (CW - m.l - m.r) / d;
          [0, 25, 50, 75, 100].forEach((p) => { bars.append(s("line", { x1: m.l, x2: CW - m.r, y1: py(p), y2: py(p), stroke: p ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": p ? "3 3" : null })); bars.append(s("text", { x: m.l - 5, y: py(p) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: p + "%" })); });
          let cum = 0; const cumPts = [];
          E.val.forEach((v, k) => {
            const p = (v / tot) * 100; cum += p; cumPts.push([m.l + bw * (k + 0.5), py(cum)]);
            bars.append(s("rect", { x: m.l + bw * k + 10, y: py(p), width: bw - 20, height: py(0) - py(p), rx: 3, fill: k < keep ? "var(--accent)" : "var(--line-strong)" }));
            bars.append(s("text", { x: m.l + bw * (k + 0.5), y: py(p) - 5, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-halo", text: fmt(p, 1) }));
            bars.append(s("text", { x: m.l + bw * (k + 0.5), y: CH - 10, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: `第${k + 1}` }));
          });
          bars.append(s("path", { d: cumPts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(""), fill: "none", stroke: "var(--hot)", "stroke-width": 2 }));
          cumPts.forEach((p, i) => bars.append(s("circle", { cx: p[0], cy: p[1], r: i === keep - 1 ? 5.5 : 3.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 1.5 })));
          const ck = cumPts[keep - 1], cumK = E.val.slice(0, keep).reduce((a, v) => a + v, 0) / tot * 100;
          bars.append(s("text", { x: Math.min(ck[0] + 8, CW - 60), y: ck[1] + 16, "font-size": 11, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: `累積 ${fmt(cumK, 1)}%` }));

          /* --- 散布圖：第 1、第 2 主成分 --- */
          scat.replaceChildren();
          scat.append(s("rect", { x: 4, y: 4, width: 352, height: 202, rx: 6, fill: "none", stroke: "var(--line)" }));
          scat.append(s("line", { x1: 4, x2: 356, y1: SY(0), y2: SY(0), stroke: "var(--line-strong)" }));
          scat.append(s("line", { x1: SX(0), x2: SX(0), y1: 4, y2: 206, stroke: "var(--line-strong)" }));
          Z.forEach((z, i) => {
            const on = i === sel;
            scat.append(s("circle", { cx: SX(z[0]), cy: SY(z[1]), r: on ? 7 : 5, fill: on ? "var(--hot)" : "var(--accent)", stroke: on ? "var(--ink)" : "var(--panel)", "stroke-width": on ? 2 : 1.5,
              class: "pc-st", "data-i": i, tabindex: 0, role: "button", "aria-label": `學生 ${i + 1}：${SUBJ.map((t, j) => t + " " + S[i][j]).join("、")}` }));
          });

          /* --- 係數表：保留的列加底色 --- */
          tb.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("is-sel", +tr.dataset.k < keep));

          /* --- 還原 --- */
          const r = S[sel], z = Z[sel];
          const rec = SUBJ.map((_, j) => mean[j] + E.vec.slice(0, keep).reduce((a, v, k) => a + z[k] * v[j], 0));
          const err = Math.sqrt(SUBJ.reduce((a, _, j) => a + (r[j] - rec[j]) ** 2, 0) / d);
          const t2 = h("table", { class: "tbl" });
          t2.append(h("thead", null, h("tr", null, h("th", { class: "l" }, `學生 ${sel + 1}`), ...SUBJ.map((t) => h("th", null, t)))));
          t2.append(h("tbody", null,
            h("tr", null, h("td", { class: "l" }, "原本的成績"), ...r.map((v) => h("td", null, String(v)))),
            h("tr", null, h("td", { class: "l" }, `用 ${keep} 個數還原`), ...rec.map((v) => h("td", null, fmt(v, 1)))),
            h("tr", null, h("td", { class: "l" }, "差"), ...rec.map((v, j) => h("td", { style: { color: Math.abs(r[j] - v) >= 5 ? "var(--bad)" : "var(--muted)" } }, fmt(r[j] - v, 1, true))))));
          const zline = h("p", { class: "hint", style: { margin: "6px 0 0" } }, `這位學生的主成分得分：${z.slice(0, keep).map((v, k) => `第 ${k + 1} = ${fmt(v, 1)}`).join("、")}　→　還原 = 各科平均 + Σ 得分 × 係數`);
          calc.replaceChildren(h("div", null, t2, zline));

          if (keep === 5) setStatus(status, "info", "保留全部 5 個主成分：只是換了一組軸，資訊完全沒丟，成績原封不動還原");
          else setStatus(status, cumK >= 90 ? "ok" : "warn", `5 個數 → ${keep} 個數：累積寄与率 ${fmt(cumK, 1)}%。學生 ${sel + 1} 的還原誤差平均約 ${fmt(err, 1)} 分`);
        }
        draw();
        note.innerHTML = `示範資料：30 位學生的 5 科成績（假想資料，固定亂數產生）。第 1 主成分的 5 個係數都是正的，可以解讀成「整體實力」；第 2 主成分是数学・理科為正、国語・社会為負，像「理科型還是文科型」。這些名字是人看了係數之後取的，PCA 本身只負責找方向。這裡 5 科的單位相同（分數），直接用原始分數計算；如果特徴的單位或大小差很多（例如身高 cm 和年收入 円），要先標準化，否則數值大的特徴會主導結果。計算上常用<span class="jp">特異値分解</span>（<span class="en">SVD</span>）求主成分。其他降維方法還有<span class="jp">多次元尺度構成法</span>（MDS，保持點與點的距離）、<span class="jp">t-SNE</span>（保持「誰和誰是鄰居」，常用來把高維資料畫成 2D 圖），以及用神經網路做的<a href="#ae">オートエンコーダ</a>。另一種教師なし学習見 <a href="#kmeans">k-means 法</a>。`;
      }

      const MODES = { dir: modeDir, subj: modeSubj };
      show();
      return {};
    },
  });
})();
