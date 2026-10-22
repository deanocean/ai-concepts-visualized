/* SVM 與核技巧（シラバス §7）
   三個模式，全部即時計算：
   ① 最大化間隔：轉動分界線的角度，程式替每個角度找最好的位置，畫出「角度 vs. 間隔」；最大值就是 SVM 的解（硬間隔）。
   ② 軟間隔：多一個靠近對方的點；用 SMO 演算法真的解出 ½‖w‖² + C·Σξ 的最小值，調 C 看間隔與越界點的取捨。
   ③ 核技巧：圓形分布的兩類。線性／升維 φ(x) = (x₁, x₂, x₁² + x₂²)／高斯核（RBF）三種核函數，
      同樣用 SMO 求解；升維時另畫 3D 視圖與真正求出的分隔平面。
   點都可以拖動（也可用方向鍵），拖完立即重算。
   文字在 locales/zh/kernel.js、locales/ja/kernel.js。 */
(function () {
  /* ---------- SMO 演算法（Platt）：解 SVM 的對偶問題，回傳 α、b ---------- */
  function smo(X, y, kern, C, tol = 1e-7) {
    const n = X.length, K = X.map((a) => X.map((b) => kern(a, b)));
    const al = new Float64Array(n); let b = 0;
    const f = (i) => { let s = b; for (let j = 0; j < n; j++) if (al[j]) s += al[j] * y[j] * K[j][i]; return s; };
    const E = (i) => f(i) - y[i];
    function step(i1, i2) {
      if (i1 === i2) return false;
      const a1 = al[i1], a2 = al[i2], y1 = y[i1], y2 = y[i2], E1 = E(i1), E2 = E(i2), s = y1 * y2;
      let L, H;
      if (y1 !== y2) { L = Math.max(0, a2 - a1); H = Math.min(C, C + a2 - a1); } else { L = Math.max(0, a1 + a2 - C); H = Math.min(C, a1 + a2); }
      if (H - L < 1e-12) return false;
      const eta = K[i1][i1] + K[i2][i2] - 2 * K[i1][i2];
      let n2;
      if (eta > 1e-12) n2 = Math.min(H, Math.max(L, a2 + (y2 * (E1 - E2)) / eta));
      else {
        const obj = (v) => { const s1 = al[i1], s2 = al[i2]; al[i2] = v; al[i1] = a1 + s * (a2 - v); let o = 0; for (let i = 0; i < n; i++) o += al[i]; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) o -= 0.5 * al[i] * al[j] * y[i] * y[j] * K[i][j]; al[i1] = s1; al[i2] = s2; return o; };
        const lo = obj(L), hi = obj(H); n2 = lo > hi + 1e-12 ? L : hi > lo + 1e-12 ? H : a2;
      }
      if (Math.abs(n2 - a2) < 1e-12 * (n2 + a2 + 1e-12)) return false;
      const n1 = a1 + s * (a2 - n2);
      const b1 = b - E1 - y1 * (n1 - a1) * K[i1][i1] - y2 * (n2 - a2) * K[i1][i2];
      const b2 = b - E2 - y1 * (n1 - a1) * K[i1][i2] - y2 * (n2 - a2) * K[i2][i2];
      al[i1] = n1; al[i2] = n2;
      b = n1 > 0 && n1 < C ? b1 : n2 > 0 && n2 < C ? b2 : (b1 + b2) / 2;
      return true;
    }
    function examine(i2) {
      const a2 = al[i2], E2 = E(i2), r2 = E2 * y[i2];
      if ((r2 < -tol && a2 < C) || (r2 > tol && a2 > 0)) {
        let best = -1, bd = -1;
        for (let i = 0; i < n; i++) if (al[i] > 0 && al[i] < C) { const d = Math.abs(E(i) - E2); if (d > bd) { bd = d; best = i; } }
        if (best >= 0 && step(best, i2)) return 1;
        for (let i = 0; i < n; i++) if (al[i] > 0 && al[i] < C && step(i, i2)) return 1;
        for (let i = 0; i < n; i++) if (step(i, i2)) return 1;
      }
      return 0;
    }
    let changed = 0, all = true, it = 0;
    while ((changed > 0 || all) && it < 20000) {
      changed = 0; it++;
      for (let i = 0; i < n; i++) if (all || (al[i] > 0 && al[i] < C)) changed += examine(i);
      if (all) all = false; else if (changed === 0) all = true;
    }
    // 用所有「沒碰到上限」的支持向量平均出 b，數值比較穩
    const free = []; for (let i = 0; i < n; i++) if (al[i] > 1e-9 && al[i] < C - 1e-9) free.push(i);
    if (free.length) b = free.reduce((a, i) => { let s = 0; for (let j = 0; j < n; j++) s += al[j] * y[j] * K[j][i]; return a + y[i] - s; }, 0) / free.length;
    const alpha = [...al];
    return { alpha, b, f: (x) => { let s = b; for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * y[j] * kern(X[j], x); return s; } };
  }

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const sq = (a) => dot(a, a);

  Atlas.define("kernel", {
    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      const { t } = ctx;
      let mode = "max";
      const modeSeg = ui.seg({ label: t("modeLabel"), value: mode, options: [
        { value: "max", label: t("modeMax") }, { value: "soft", label: t("modeSoft") }, { value: "kern", label: t("modeKern") }],
        onChange: (v) => { mode = v; show(); } });
      bench.append(h("style", { text: `
        .sv-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; align-items: flex-start; }
        .sv-fig { flex: 1 1 320px; max-width: 520px; min-width: 0; }
        .sv-fig > svg { display: block; width: 100%; height: auto; }
        .sv-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
        .sv-pt { cursor: grab; touch-action: none; }
        .sv-pt:active { cursor: grabbing; }
        .sv-leg { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 12.5px; color: var(--muted); justify-content: center; }
        .sv-leg span { display: inline-flex; align-items: center; gap: 6px; }
        .sv-leg svg { flex: none; }
      ` }));
      const box = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } });
      const note = h("p", { class: "note" });
      bench.append(h("div", { class: "bench__row" }, modeSeg.el), box);
      let inst = null;
      function show() { if (inst && inst.destroy) inst.destroy(); box.replaceChildren(); inst = MODES[mode](); box.append(note); }
      const setStatus = (el, k, msg) => { el.className = "status status--" + k; el.textContent = msg; };
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
      const sci = (v, d = 4) => { if (v === 0 || Math.abs(v) >= 1e-3) return fmt(v, d); const [m, e] = v.toExponential(1).split("e"); return `${m}×10${String(+e).split("").map((c) => SUP[c]).join("")}`; };
      const pt2 = (p) => `(${fmt(p[0], 2)}, ${fmt(p[1], 2)})`;

      /* ---------- 共用：點的形狀與圖例 ---------- */
      // 類別 A（+1）＝藍色圓點；類別 B（−1）＝紅色方塊（不只靠顏色區分）
      function marker(x, y, cls, i, extra = {}) {
        const common = { fill: cls > 0 ? "var(--pos)" : "var(--neg)", stroke: "var(--panel)", "stroke-width": 2, class: "sv-pt", "data-i": i, tabindex: 0, role: "button",
          "aria-label": t("ptAria", { c: cls > 0 ? "A" : "B" }), ...extra };
        return cls > 0 ? s("circle", { cx: x, cy: y, r: 7.5, ...common }) : s("rect", { x: x - 7, y: y - 7, width: 14, height: 14, rx: 2, ...common });
      }
      function legend(items) {
        const el = h("div", { class: "sv-leg" });
        items.forEach(([kind, label]) => {
          const g = s("svg", { width: 26, height: 16, viewBox: "0 0 26 16", "aria-hidden": "true" });
          if (kind === "A") g.append(s("circle", { cx: 13, cy: 8, r: 6, fill: "var(--pos)" }));
          else if (kind === "B") g.append(s("rect", { x: 7, y: 2, width: 12, height: 12, rx: 2, fill: "var(--neg)" }));
          else if (kind === "sv") g.append(s("circle", { cx: 13, cy: 8, r: 6.5, fill: "none", stroke: "var(--hot)", "stroke-width": 2.5 }));
          else if (kind === "line") g.append(s("line", { x1: 1, x2: 25, y1: 8, y2: 8, stroke: "var(--ink)", "stroke-width": 2.5 }));
          else if (kind === "band") { g.append(s("rect", { x: 1, y: 3, width: 24, height: 10, fill: "var(--accent-soft)" })); g.append(s("line", { x1: 1, x2: 25, y1: 3, y2: 3, stroke: "var(--accent)", "stroke-dasharray": "4 3" })); g.append(s("line", { x1: 1, x2: 25, y1: 13, y2: 13, stroke: "var(--accent)", "stroke-dasharray": "4 3" })); }
          else if (kind === "xi") g.append(s("line", { x1: 1, x2: 25, y1: 8, y2: 8, stroke: "var(--bad)", "stroke-width": 2, "stroke-dasharray": "3 3" }));
          el.append(h("span", null, g, label));
        });
        return el;
      }

      /* ---------- 共用：拖動點 ---------- */
      function draggable(svg, toData, pts, redraw, onClick) {
        let drag = null, moved = false, start = null;
        const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
        svg.addEventListener("pointerdown", (e) => {
          const i = e.target.getAttribute && e.target.getAttribute("data-i"); if (i == null) return;
          drag = +i; moved = false; start = [e.clientX, e.clientY]; try { svg.setPointerCapture(e.pointerId); } catch (_) {} e.preventDefault();
        });
        svg.addEventListener("pointermove", (e) => {
          if (drag == null) return;
          if (!moved && Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 4) return;
          moved = true; const q = toSvg(e); pts[drag].p = toData(q.x, q.y); redraw();
        });
        const end = () => { if (drag != null && !moved && onClick) onClick(drag); drag = null; };
        svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", () => { drag = null; });
        svg.addEventListener("keydown", (e) => {
          const i = e.target.getAttribute && e.target.getAttribute("data-i"); if (i == null) return;
          if ((e.key === "Enter" || e.key === " ") && onClick) { e.preventDefault(); onClick(+i); return; }
          const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key]; if (!d) return;
          e.preventDefault(); const p = pts[+i].p; pts[+i].p = toData(null, null, [p[0] + d[0], p[1] + d[1]]); redraw();
          const el = svg.querySelector(`[data-i="${i}"]`); el && el.focus();
        });
      }

      /* ---------- 共用：線性的 2D 平面（模式 ①②） ---------- */
      const BASE_A = [[5.0, 3.6], [6.2, 4.0], [5.6, 2.6], [6.8, 3.0], [7.0, 4.2], [6.0, 1.9]];
      const BASE_B = [[1.2, 1.0], [2.0, 2.2], [1.0, 2.6], [2.6, 0.8], [3.2, 1.8], [2.2, 3.6]];
      const XW = 8, YH = 5, U = 54, PM = { l: 28, t: 10, r: 10, b: 26 };
      const PW = PM.l + XW * U + PM.r, PH = PM.t + YH * U + PM.b;
      const X = (x) => PM.l + x * U, Y = (y) => PM.t + (YH - y) * U;
      const clampPt = (v) => [Math.max(0.2, Math.min(XW - 0.2, v[0])), Math.max(0.2, Math.min(YH - 0.2, v[1]))].map((u) => Math.round(u * 20) / 20);
      const toDataLin = (sx, sy, raw) => clampPt(raw || [(sx - PM.l) / U, YH - (sy - PM.t) / U]);
      function planeSvg(label, clipId) {
        const svg = s("svg", { viewBox: `0 0 ${PW} ${PH}`, role: "img", "aria-label": label });
        svg._clip = clipId; return svg;
      }
      function planeBase(svg) {
        svg.replaceChildren(s("defs", null, s("clipPath", { id: svg._clip }, s("rect", { x: PM.l, y: PM.t, width: XW * U, height: YH * U }))));
        svg.append(s("rect", { x: PM.l, y: PM.t, width: XW * U, height: YH * U, fill: "none", stroke: "var(--line)" }));
        for (let x = 0; x <= XW; x++) {
          if (x > 0 && x < XW) svg.append(s("line", { x1: X(x), x2: X(x), y1: PM.t, y2: PM.t + YH * U, stroke: "var(--line)", "stroke-dasharray": "2 4" }));
          svg.append(s("text", { x: X(x), y: PH - 8, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: x }));
        }
        for (let y = 0; y <= YH; y++) {
          if (y > 0 && y < YH) svg.append(s("line", { x1: PM.l, x2: PM.l + XW * U, y1: Y(y), y2: Y(y), stroke: "var(--line)", "stroke-dasharray": "2 4" }));
          svg.append(s("text", { x: PM.l - 6, y: Y(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y }));
        }
        const g = s("g", { "clip-path": `url(#${svg._clip})` }); svg.append(g); return g;
      }
      // 直線 n·x = c（n 不必是單位向量），畫成兩端在畫面外的長線，再由 clipPath 裁掉
      function lineEnds(nv, c) {
        const L = Math.hypot(nv[0], nv[1]), u = [nv[0] / L, nv[1] / L], p0 = [u[0] * c / L, u[1] * c / L], d = [-u[1], u[0]];
        return [[p0[0] - 30 * d[0], p0[1] - 30 * d[1]], [p0[0] + 30 * d[0], p0[1] + 30 * d[1]]];
      }
      function drawLine(g, nv, c, attrs) { const [a, b] = lineEnds(nv, c); g.append(s("line", { x1: X(a[0]), y1: Y(a[1]), x2: X(b[0]), y2: Y(b[1]), ...attrs })); }
      function drawBand(g, nv, c1, c2) {
        const [a, b] = lineEnds(nv, c1), [d, e] = lineEnds(nv, c2);
        g.append(s("polygon", { points: [a, b, e, d].map((p) => X(p[0]) + "," + Y(p[1])).join(" "), fill: "var(--accent-soft)" }));
        drawLine(g, nv, c1, { stroke: "var(--accent)", "stroke-width": 1.5, "stroke-dasharray": "6 4" });
        drawLine(g, nv, c2, { stroke: "var(--accent)", "stroke-width": 1.5, "stroke-dasharray": "6 4" });
      }
      const ring = (g, p, r = 13) => g.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r, fill: "none", stroke: "var(--hot)", "stroke-width": 2.5 }));

      /* ===================== ① 最大化間隔 ===================== */
      function modeMax() {
        const pts = [...BASE_A.map((p) => ({ p: [...p], c: 1 })), ...BASE_B.map((p) => ({ p: [...p], c: -1 }))];
        let alpha = 90, exact = false; // 直線的方向（度，0〜180）
        const deg = Math.PI / 180;
        // 法向量方向 θ：A 類要在正側。m(θ) = (A 類投影最小 − B 類投影最大) ÷ 2
        function atNormal(th) {
          const u = [Math.cos(th), Math.sin(th)]; let pa = Infinity, pb = -Infinity;
          pts.forEach((q) => { const v = dot(u, q.p); if (q.c > 0) pa = Math.min(pa, v); else pb = Math.max(pb, v); });
          return { m: (pa - pb) / 2, c: (pa + pb) / 2, u, pa, pb };
        }
        // 直線方向 α 的兩個法向量 α±90°，取比較好的那一個
        const atLine = (a) => { const r1 = atNormal((a + 90) * deg), r2 = atNormal((a - 90) * deg); return r1.m >= r2.m ? r1 : r2; };
        let best = null;
        function solve() {
          let bm = -Infinity, ba = 0;
          for (let k = 0; k < 3600; k++) { const a = k * 0.05, m = atLine(a).m; if (m > bm) { bm = m; ba = a; } }
          let lo = ba - 0.05, hi = ba + 0.05; // 黃金分割法細修
          for (let it = 0; it < 60; it++) { const m1 = lo + (hi - lo) * 0.382, m2 = lo + (hi - lo) * 0.618; if (atLine(m1).m < atLine(m2).m) lo = m1; else hi = m2; }
          ba = ((((lo + hi) / 2) % 180) + 180) % 180;
          best = { a: ba, ...atLine(ba) };
        }
        const sl = ui.slider({ id: "sv-a", label: t("angleLabel"), min: 0, max: 179, step: 1, value: alpha, format: (v) => (exact ? fmt(best.a, 1) : v) + "°", onInput: (v) => { alpha = v; exact = false; draw(); } });
        const bestBtn = ui.btn(t("btnBest"), () => { exact = true; alpha = best.a; sl.set(Math.round(best.a)); draw(); }, { primary: true, sm: true });
        const resetBtn = ui.btn(t("btnReset"), () => { BASE_A.concat(BASE_B).forEach((p, i) => { pts[i].p = [...p]; }); solve(); if (exact) alpha = best.a; draw(); }, { sm: true });
        const main = planeSvg(t("maxMainAria"), "svclip1");
        const chart = s("svg", { viewBox: "0 0 360 240", role: "img", "aria-label": t("maxChartAria") });
        const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
        box.append(
          h("div", { class: "bench__row" }, sl.el, bestBtn, resetBtn),
          h("div", { class: "scroll-x" }, h("div", { class: "sv-figs" },
            h("div", { class: "sv-fig", style: { flexBasis: "420px" } }, h("div", { class: "sv-cap" }, t("maxMainCap")), main,
              legend([["A", t("legA")], ["B", t("legB")], ["line", t("legLine")], ["band", t("legBand")], ["sv", t("legTouch")]])),
            h("div", { class: "sv-fig", style: { flexBasis: "300px", maxWidth: "400px" } }, h("div", { class: "sv-cap" }, t("maxChartCap")), chart))),
          status, calc);
        draggable(main, toDataLin, pts, () => { solve(); if (exact) alpha = best.a; draw(); });

        function draw() {
          const r = exact ? best : atLine(alpha), ok = r.m > 0, isBest = best.m > 0 && (exact || Math.abs(r.m - best.m) < 5e-4);
          const g = planeBase(main);
          if (ok) {
            drawBand(g, r.u, r.c - r.m, r.c + r.m);
            drawLine(g, r.u, r.c, { stroke: "var(--ink)", "stroke-width": 2.5 });
            pts.forEach((q) => { const v = dot(r.u, q.p); if (Math.abs(v - (q.c > 0 ? r.pa : r.pb)) < 1e-6) ring(g, q.p); });
          } else {
            // 分不開：照樣畫在兩類投影的中間，標出站錯邊的點
            drawLine(g, r.u, r.c, { stroke: "var(--bad)", "stroke-width": 2.5, "stroke-dasharray": "8 5" });
            pts.forEach((q) => { const v = dot(r.u, q.p) - r.c; if (v * q.c < 0) g.append(s("circle", { cx: X(q.p[0]), cy: Y(q.p[1]), r: 13, fill: "none", stroke: "var(--bad)", "stroke-width": 2.5 })); });
          }
          pts.forEach((q, i) => main.append(marker(X(q.p[0]), Y(q.p[1]), q.c, i)));
          drawChart(r);

          if (best.m <= 0) setStatus(status, "bad", t("statusNone"));
          else if (!ok) setStatus(status, "bad", t("statusBadAngle"));
          else if (isBest) setStatus(status, "ok", t("statusBest", { m: fmt(best.m, 3) }));
          else setStatus(status, "info", t("statusAngle", { m: fmt(r.m, 3), best: fmt(best.m, 3), p: fmt((1 - r.m / best.m) * 100, 0) }));

          const th = Math.atan2(r.u[1], r.u[0]);
          calc.innerHTML =
            row(t("calcAngleK"), t("calcAngle", { u: `(${fmt(r.u[0], 3)}, ${fmt(r.u[1], 3)})`, th: fmt(((th / deg) + 360) % 360, 1) })) +
            row("", t("calcProj", { pa: fmt(r.pa, 3), pb: fmt(r.pb, 3), c: fmt(r.c, 3), m: fmt(r.m, 3) })) +
            (best.m > 0 ? row(t("calcBestK"), t("calcBest", { a: fmt(best.a, 1), m: fmt(best.m, 3) })) +
              row(t("calcStdK"), t("calcStd", { m: fmt(best.m, 3), w: fmt(1 / best.m, 3) })) : "");
        }
        function drawChart(r) {
          const CW = 360, CH = 240, m = { l: 40, r: 12, t: 16, b: 34 };
          const top = Math.max(0.5, Math.ceil(Math.max(best.m, 0) * 1.2 * 2) / 2), tick = top > 1 ? 0.5 : 0.25;
          const cx = (a) => m.l + (a / 180) * (CW - m.l - m.r), cy = (v) => CH - m.b - (Math.max(v, 0) / top) * (CH - m.t - m.b);
          chart.replaceChildren();
          for (let k = 0; k <= Math.round(top / tick); k++) {
            const v = k * tick;
            chart.append(s("line", { x1: m.l, x2: CW - m.r, y1: cy(v), y2: cy(v), stroke: k ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": k ? "3 3" : null }));
            chart.append(s("text", { x: m.l - 5, y: cy(v) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: fmt(v, 2) }));
          }
          for (let a = 0; a <= 180; a += 45) chart.append(s("text", { x: cx(a), y: CH - m.b + 15, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: a + "°" }));
          chart.append(s("text", { x: CW - m.r, y: CH - 4, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("angleLabel") }));
          // 分不開的角度：底部紅色粗線
          let d = "", bad = "";
          for (let k = 0; k <= 360; k++) {
            const a = k / 2, v = atLine(a).m;
            if (v > 0) { d += (d && !d.endsWith("|") ? "L" : "M") + cx(a).toFixed(1) + "," + cy(v).toFixed(1); }
            else { if (d && !d.endsWith("|")) d += "|"; bad += `M${(cx(a) - 0.6).toFixed(1)},${cy(0) + 2}h1.2`; }
          }
          chart.append(s("path", { d: bad, stroke: "var(--bad)", "stroke-width": 5, fill: "none" }));
          chart.append(s("path", { d: d.replace(/\|/g, ""), fill: "none", stroke: "var(--accent)", "stroke-width": 2.2 }));
          if (best.m > 0) {
            chart.append(s("circle", { cx: cx(best.a), cy: cy(best.m), r: 4.5, fill: "var(--ok)" }));
            const right = best.a > 120;
            chart.append(s("text", { x: cx(best.a) + (right ? -8 : 8), y: cy(best.m) - 6, "text-anchor": right ? "end" : "start", "font-size": 11, "font-weight": 700, fill: "var(--ok)", class: "t-halo", text: t("chartBest", { m: fmt(best.m, 3) }) }));
          }
          const a = exact ? best.a : alpha;
          chart.append(s("line", { x1: cx(a), x2: cx(a), y1: m.t, y2: CH - m.b, stroke: "var(--hot)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
          chart.append(s("circle", { cx: cx(a), cy: cy(r.m), r: 5.5, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2 }));
        }
        solve(); draw();
        note.innerHTML = t("noteMax");
        return {};
      }

      /* ===================== ② 軟間隔 ===================== */
      function modeSoft() {
        const OUT = [4.4, 3.0];
        const pts = [...BASE_A.map((p) => ({ p: [...p], c: 1 })), ...BASE_B.map((p) => ({ p: [...p], c: -1 })), { p: [...OUT], c: -1 }];
        const CS = [0.05, 0.1, 0.3, 1, 3, 10, 100];
        let ci = 6;
        const sl = ui.slider({ id: "sv-c", label: t("cLabel"), min: 0, max: CS.length - 1, step: 1, value: ci, format: (v) => String(CS[v]), onInput: (v) => { ci = v; draw(); } });
        const resetBtn = ui.btn(t("btnReset"), () => { BASE_A.concat(BASE_B, [OUT]).forEach((p, i) => { pts[i].p = [...p]; }); draw(); }, { sm: true });
        const main = planeSvg(t("softAria"), "svclip2");
        const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
        box.append(
          h("div", { class: "bench__row" }, sl.el, resetBtn),
          h("div", { class: "scroll-x" }, h("div", { class: "sv-figs" }, h("div", { class: "sv-fig", style: { flexBasis: "460px", maxWidth: "620px" } },
            h("div", { class: "sv-cap" }, t("softCap")), main,
            legend([["A", t("legA")], ["B", t("legB")], ["line", t("legLineEq")], ["band", t("legBandEq")], ["sv", t("legSv")], ["xi", t("legXi")]])))),
          status, calc);
        draggable(main, toDataLin, pts, draw);

        function draw() {
          const C = CS[ci], Xs = pts.map((q) => q.p), ys = pts.map((q) => q.c);
          const r = smo(Xs, ys, dot, C);
          const w = [0, 0]; r.alpha.forEach((a, i) => { w[0] += a * ys[i] * Xs[i][0]; w[1] += a * ys[i] * Xs[i][1]; });
          const nw = Math.hypot(w[0], w[1]), mar = 1 / nw;
          const fx = (p) => dot(w, p) + r.b;
          const xi = Xs.map((p, i) => Math.max(0, 1 - ys[i] * fx(p)));
          const g = planeBase(main);
          drawBand(g, w, -r.b - 1, -r.b + 1);
          drawLine(g, w, -r.b, { stroke: "var(--ink)", "stroke-width": 2.5 });
          Xs.forEach((p, i) => {
            if (xi[i] > 1e-6) {
              // 從點畫到「自己那一側的虛線」：沿 w 方向移動 (yᵢ − f(x)) ÷ ‖w‖²
              const k = (ys[i] - fx(p)) / (nw * nw), q = [p[0] + k * w[0], p[1] + k * w[1]];
              g.append(s("line", { x1: X(p[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(q[1]), stroke: "var(--bad)", "stroke-width": 2, "stroke-dasharray": "3 3" }));
            }
            if (r.alpha[i] > 1e-7) ring(g, p);
          });
          pts.forEach((q, i) => main.append(marker(X(q.p[0]), Y(q.p[1]), q.c, i)));

          const nsv = r.alpha.filter((a) => a > 1e-7).length, viol = xi.filter((v) => v > 1e-6).length, wrong = xi.filter((v) => v > 1).length, sumXi = xi.reduce((a, v) => a + v, 0);
          if (viol === 0) setStatus(status, "warn", t("statusNoViol", { c: C, m: fmt(mar, 3) }));
          else setStatus(status, wrong ? "info" : "ok", wrong ? t("statusViolWrong", { c: C, n: viol, w: wrong, m: fmt(mar, 3) }) : t("statusViol", { c: C, n: viol, m: fmt(mar, 3) }));
          const list = Xs.map((p, i) => [p, xi[i]]).filter((v) => v[1] > 1e-6).sort((a, b) => b[1] - a[1]).map(([p, v]) => `${pt2(p)} ${fmt(v, 3)}`).join(t("listSep"));
          calc.innerHTML =
            row(t("calcObjK"), t("calcObj", { nw: fmt(nw, 3), c: C, xi: fmt(sumXi, 3), v: fmt(0.5 * nw * nw + C * sumXi, 3) })) +
            row(t("calcSolK"), t("calcSol", { w: `(${fmt(w[0], 3)}, ${fmt(w[1], 3)})`, b: fmt(r.b, 3), m: fmt(mar, 3), n: nsv })) +
            row(t("calcXiK"), viol ? t("calcXi", { list }) : t("calcXiNone"));
        }
        draw();
        note.innerHTML = t("noteSoft");
        return {};
      }

      /* ===================== ③ 核技巧 ===================== */
      function modeKern() {
        const IN = [[0.3, 0.2], [-0.5, 0.6], [0.8, -0.3], [-0.2, -0.9], [0.6, 0.7], [-0.8, -0.2], [0.1, -0.4], [-0.4, 0.1]];
        const OUT = [[1.97, 0.35], [1.38, 1.16], [0.57, 2.13], [-0.49, 1.84], [-1.61, 1.35], [-1.64, 0.44], [-1.88, -0.68], [-1.41, -1.69], [-0.47, -1.74], [0.72, -1.97], [1.46, -1.22], [1.93, -0.52]];
        const pts = [...IN.map((p) => ({ p: [...p], c: 1 })), ...OUT.map((p) => ({ p: [...p], c: -1 }))];
        const GAMMAS = [0.1, 0.5, 2, 10], C = 10;
        let kind = "lift", gi = 1, sel = [0, 8], elev = 20, azim = 35;
        const KERN = {
          lin: (a, b) => dot(a, b),
          lift: (a, b) => dot(a, b) + sq(a) * sq(b),
          rbf: (a, b) => Math.exp(-GAMMAS[gi] * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)),
        };
        const kSeg = ui.seg({ label: t("kernLabel"), value: kind, options: [
          { value: "lin", label: t("kernLin") }, { value: "lift", label: t("kernLift") }, { value: "rbf", label: t("kernRbf") }],
          onChange: (v) => { kind = v; draw(); } });
        const gSl = ui.slider({ id: "sv-g", label: t("gammaLabel"), min: 0, max: GAMMAS.length - 1, step: 1, value: gi, format: (v) => String(GAMMAS[v]), onInput: (v) => { gi = v; draw(); } });
        const eSl = ui.slider({ id: "sv-e", label: t("elevLabel"), min: 0, max: 90, step: 1, value: elev, format: (v) => v + "°", onInput: (v) => { elev = v; draw(); } });
        const aSl = ui.slider({ id: "sv-z", label: t("azimLabel"), min: 0, max: 359, step: 1, value: azim, format: (v) => v + "°", onInput: (v) => { azim = v; draw(); } });
        const resetBtn = ui.btn(t("btnReset"), () => { IN.concat(OUT).forEach((p, i) => { pts[i].p = [...p]; }); draw(); }, { sm: true });
        const gRow = h("div", { class: "bench__row" }, gSl.el);
        const vRow = h("div", { class: "bench__row" }, eSl.el, aSl.el);

        const R = 2.6, KU = 64, KM = { l: 30, t: 10, r: 10, b: 26 };
        const KW = KM.l + 2 * R * KU + KM.r, KH = KM.t + 2 * R * KU + KM.b;
        const KX = (x) => KM.l + (x + R) * KU, KY = (y) => KM.t + (R - y) * KU;
        const toDataK = (sx, sy, raw) => (raw || [(sx - KM.l) / KU - R, R - (sy - KM.t) / KU]).map((u) => Math.round(Math.max(-R + 0.15, Math.min(R - 0.15, u)) * 20) / 20);
        const flat = s("svg", { viewBox: `0 0 ${KW} ${KH}`, role: "img", "aria-label": t("flatAria") });
        const cube = s("svg", { viewBox: "0 0 380 360", role: "img", "aria-label": t("cubeAria") });
        const cubeFig = h("div", { class: "sv-fig", style: { flexBasis: "300px", maxWidth: "440px" } }, h("div", { class: "sv-cap" }, t("cubeCap")), cube);
        const status = h("div", { class: "status" }), calc = h("div", { class: "calc" });
        box.append(
          h("div", { class: "bench__row" }, kSeg.el, resetBtn), gRow,
          h("div", { class: "scroll-x" }, h("div", { class: "sv-figs" },
            h("div", { class: "sv-fig", style: { flexBasis: "330px", maxWidth: "440px" } }, h("div", { class: "sv-cap" }, t("flatCap")), flat,
              legend([["A", t("legA")], ["B", t("legB")], ["line", t("legBoundary")], ["sv", t("legSv")]])),
            cubeFig)),
          vRow, status, calc);
        draggable(flat, toDataK, pts, draw, (i) => { if (i !== sel[1]) sel = [sel[1], i]; draw(); });

        // 等高線（marching squares）：回傳 f = level 的線段
        function contour(F, G, x0, step, level) {
          let d = "";
          const P = (i, j) => [x0 + i * step, x0 + j * step];
          for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
            const v = [F[i][j] - level, F[i + 1][j] - level, F[i + 1][j + 1] - level, F[i][j + 1] - level];
            const c = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)];
            const cr = [];
            for (let e = 0; e < 4; e++) { const a = v[e], b = v[(e + 1) % 4]; if ((a < 0) !== (b < 0)) { const f = a / (a - b), p = c[e], q = c[(e + 1) % 4]; cr.push([p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])]); } }
            if (cr.length === 2) d += `M${KX(cr[0][0]).toFixed(1)},${KY(cr[0][1]).toFixed(1)}L${KX(cr[1][0]).toFixed(1)},${KY(cr[1][1]).toFixed(1)}`;
            else if (cr.length === 4) { const mid = (v[0] + v[1] + v[2] + v[3]) / 4, o = (mid < 0) === (v[0] < 0) ? [0, 3, 2, 1] : [0, 1, 2, 3]; d += `M${KX(cr[o[0]][0]).toFixed(1)},${KY(cr[o[0]][1]).toFixed(1)}L${KX(cr[o[1]][0]).toFixed(1)},${KY(cr[o[1]][1]).toFixed(1)}M${KX(cr[o[2]][0]).toFixed(1)},${KY(cr[o[2]][1]).toFixed(1)}L${KX(cr[o[3]][0]).toFixed(1)},${KY(cr[o[3]][1]).toFixed(1)}`; }
          }
          return d;
        }

        function draw() {
          gRow.hidden = kind !== "rbf"; vRow.hidden = cubeFig.hidden = kind !== "lift";
          const Xs = pts.map((q) => q.p), ys = pts.map((q) => q.c), kern = KERN[kind];
          const r = smo(Xs, ys, kern, C);
          const svIdx = r.alpha.map((a, i) => (a > 1e-7 ? i : -1)).filter((i) => i >= 0);
          const acc = Xs.filter((p, i) => Math.sign(r.f(p)) === ys[i]).length;

          /* --- 平面：依 f 的正負塗底色，畫 f = 0 與 f = ±1 --- */
          flat.replaceChildren(s("defs", null, s("clipPath", { id: "svclip3" }, s("rect", { x: KM.l, y: KM.t, width: 2 * R * KU, height: 2 * R * KU }))));
          const G = 52, st = (2 * R) / G, F = [];
          for (let i = 0; i <= G; i++) { F.push([]); for (let j = 0; j <= G; j++) F[i].push(r.f([-R + i * st, -R + j * st])); }
          const shade = s("g", { "aria-hidden": "true" });
          for (let j = 0; j < G; j++) { // 每一列把同號的格子合併成一個長方形
            let start = 0, sign = null;
            for (let i = 0; i <= G; i++) {
              const sg = i < G ? (F[i][j] + F[i + 1][j] + F[i][j + 1] + F[i + 1][j + 1]) / 4 >= 0 : null;
              if (sg !== sign) {
                if (sign !== null) shade.append(s("rect", { x: KX(-R + start * st), y: KY(-R + (j + 1) * st), width: (i - start) * st * KU + 0.6, height: st * KU + 0.6, fill: sign ? "var(--pos-soft)" : "var(--neg-soft)" }));
                start = i; sign = sg;
              }
            }
          }
          flat.append(shade);
          for (let v = -2; v <= 2; v++) {
            flat.append(s("text", { x: KX(v), y: KH - 8, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v < 0 ? "−" + -v : v }));
            flat.append(s("text", { x: KM.l - 6, y: KY(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: v < 0 ? "−" + -v : v }));
          }
          flat.append(s("rect", { x: KM.l, y: KM.t, width: 2 * R * KU, height: 2 * R * KU, fill: "none", stroke: "var(--line-strong)" }));
          const g = s("g", { "clip-path": "url(#svclip3)" });
          g.append(s("path", { d: contour(F, G, -R, st, 1) + contour(F, G, -R, st, -1), fill: "none", stroke: "var(--accent)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }));
          g.append(s("path", { d: contour(F, G, -R, st, 0), fill: "none", stroke: "var(--ink)", "stroke-width": 2.5 }));
          svIdx.forEach((i) => g.append(s("circle", { cx: KX(Xs[i][0]), cy: KY(Xs[i][1]), r: 12, fill: "none", stroke: "var(--hot)", "stroke-width": 2.5 })));
          flat.append(g);
          pts.forEach((q, i) => flat.append(marker(KX(q.p[0]), KY(q.p[1]), q.c, i, { "aria-label": t("ptAriaSel", { c: q.c > 0 ? "A" : "B" }) })));
          ["a", "b"].forEach((nm, k) => {
            const p = Xs[sel[k]], below = p[1] > R - 0.5;
            flat.append(s("text", { x: KX(p[0]) + 11, y: KY(p[1]) + (below ? 20 : -9), "font-size": 13, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: nm, style: "pointer-events:none" }));
          });

          if (kind === "lift") draw3d(r, Xs, ys);

          /* --- 狀態 --- */
          const n = Xs.length;
          if (kind === "lin") setStatus(status, acc === n ? "ok" : "bad", t(acc === n ? "statusLinOk" : "statusLinBad", { acc, n }));
          else if (kind === "lift") setStatus(status, acc === n ? "ok" : "warn", t(acc === n ? "statusLiftOk" : "statusLiftBad", { acc, n }));
          else if (svIdx.length === n) setStatus(status, "warn", t("statusRbfAll", { g: GAMMAS[gi], n }));
          else setStatus(status, acc === n ? "ok" : "warn", t("statusRbf", { g: GAMMAS[gi], acc, n, sv: svIdx.length }));

          /* --- 算式：選中的兩點 a、b --- */
          const a = Xs[sel[0]], b = Xs[sel[1]];
          let kr;
          if (kind === "lin") kr = row("K(a, b) = a·b", `${fmt(a[0], 2)}×${Atlas.fmtP(b[0], 2)} + ${fmt(a[1], 2)}×${Atlas.fmtP(b[1], 2)} = <span class="hl">${fmt(dot(a, b), 3)}</span>`);
          else if (kind === "lift") {
            const pa = [a[0], a[1], sq(a)], pb = [b[0], b[1], sq(b)], pd = pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2];
            kr = row(t("calcExplicitK"), t("calcExplicit", { pa: `(${pa.map((v) => fmt(v, 3)).join(", ")})`, pb: `(${pb.map((v) => fmt(v, 3)).join(", ")})`, v: fmt(pd, 4) })) +
              row(t("calcKernK"), t("calcKern", { ab: fmt(dot(a, b), 4), aa: fmt(sq(a), 4), bb: fmt(sq(b), 4), v: fmt(KERN.lift(a, b), 4) }));
          } else {
            const d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
            kr = row("K(a, b)", t("calcRbf", { g: GAMMAS[gi], d: fmt(d2, 3), v: sci(KERN.rbf(a, b)) }));
          }
          calc.innerHTML = row(t("calcSelK"), t("calcSel", { a: pt2(a), b: pt2(b) })) + kr +
            row(t("calcDecK"), t("calcDec", { n: svIdx.length, f: fmt(r.f(a), 3), c: r.f(a) >= 0 ? "A" : "B" }));
        }

        function draw3d(r, Xs, ys) {
          // 升維後的平面：w₁x₁ + w₂x₂ + w₃z + b = 0（w 由 α 算出）
          const w = [0, 0, 0]; r.alpha.forEach((al, i) => { const p = Xs[i]; w[0] += al * ys[i] * p[0]; w[1] += al * ys[i] * p[1]; w[2] += al * ys[i] * sq(p); });
          const zOf = (x, y) => -(w[0] * x + w[1] * y + r.b) / w[2];
          const ZS = 0.55, SC = 50, OX = 190, OY = 230; // z 只在畫面上縮放，數值不變
          const ca = Math.cos(azim * Math.PI / 180), sa = Math.sin(azim * Math.PI / 180), ce = Math.cos(elev * Math.PI / 180), se = Math.sin(elev * Math.PI / 180);
          const proj = (x, y, z) => { const X1 = x * ca - y * sa, Y1 = x * sa + y * ca; return { x: OX + SC * X1, y: OY - SC * (z * ZS * ce + Y1 * se), depth: Y1 * ce - z * ZS * se }; };
          cube.replaceChildren();
          const B = 2.4, poly = (ps, attrs) => cube.append(s("polygon", { points: ps.map((p) => p.x.toFixed(1) + "," + p.y.toFixed(1)).join(" "), ...attrs }));
          poly([proj(-B, -B, 0), proj(B, -B, 0), proj(B, B, 0), proj(-B, B, 0)], { fill: "var(--panel-2)", stroke: "var(--line-strong)" });
          for (let v = -2; v <= 2; v++) {
            const p1 = proj(v, -B, 0), p2 = proj(v, B, 0), q1 = proj(-B, v, 0), q2 = proj(B, v, 0);
            cube.append(s("line", { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, stroke: "var(--line)" }));
            cube.append(s("line", { x1: q1.x, y1: q1.y, x2: q2.x, y2: q2.y, stroke: "var(--line)" }));
          }
          // z 軸：放在畫面最左邊的那個角，刻度往左寫，不會壓到點
          const fc = [[-B, -B], [B, -B], [B, B], [-B, B]].map(([x, y]) => ({ x, y, p: proj(x, y, 0) })).sort((a, b) => a.p.x - b.p.x)[0];
          const z0 = proj(fc.x, fc.y, 0), z1 = proj(fc.x, fc.y, 6);
          cube.append(s("line", { x1: z0.x, y1: z0.y, x2: z1.x, y2: z1.y, stroke: "var(--line-strong)", "stroke-width": 1.5 }));
          [0, 2, 4, 6].forEach((z) => { const p = proj(fc.x, fc.y, z); cube.append(s("line", { x1: p.x - 3, x2: p.x + 3, y1: p.y, y2: p.y, stroke: "var(--line-strong)" })); if (elev < 85) cube.append(s("text", { x: p.x - 6, y: p.y + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted t-halo", text: "z=" + z })); });
          const ax = proj(B + 0.35, 0, 0), ay = proj(0, B + 0.35, 0);
          cube.append(s("text", { x: ax.x, y: ax.y + 4, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted t-halo", text: "x₁" }));
          cube.append(s("text", { x: ay.x, y: ay.y + 4, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted t-halo", text: "x₂" }));

          const above = [], below = [];
          Xs.forEach((p, i) => { const z = sq(p); (z > zOf(p[0], p[1]) ? above : below).push(i); });
          const drawPts = (idx) => idx.map((i) => ({ i, pr: proj(Xs[i][0], Xs[i][1], sq(Xs[i])) })).sort((a, b) => b.pr.depth - a.pr.depth).forEach(({ i, pr }) => {
            const fl = proj(Xs[i][0], Xs[i][1], 0);
            cube.append(s("line", { x1: fl.x, y1: fl.y, x2: pr.x, y2: pr.y, stroke: "var(--faint)", "stroke-dasharray": "2 3" }));
            if (ys[i] > 0) cube.append(s("circle", { cx: pr.x, cy: pr.y, r: 6.5, fill: "var(--pos)", stroke: "var(--panel)", "stroke-width": 1.5 }));
            else cube.append(s("rect", { x: pr.x - 6, y: pr.y - 6, width: 12, height: 12, rx: 2, fill: "var(--neg)", stroke: "var(--panel)", "stroke-width": 1.5 }));
          });
          drawPts(below);
          const corners = [[-B, -B], [B, -B], [B, B], [-B, B]].map(([x, y]) => proj(x, y, zOf(x, y)));
          poly(corners, { fill: "var(--hot)", "fill-opacity": 0.22, stroke: "var(--hot)", "stroke-width": 1.5 });
          drawPts(above);
          // 標籤放在平面最靠近觀看者（畫面最下方）的角的下面
          const lab = corners.slice().sort((a, b) => b.y - a.y)[0];
          cube.append(s("text", { x: Math.max(60, Math.min(320, lab.x)), y: Math.min(330, lab.y + 18), "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, fill: "var(--hot)", class: "t-halo", text: t("cubePlane") }));
          cube.append(s("text", { x: 190, y: 352, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: `${fmt(w[0], 2)}·x₁ + ${fmt(w[1], 2)}·x₂ + ${fmt(w[2], 2)}·z + ${fmt(r.b, 2)} = 0`.replace(/\+ −/g, "− ") }));
        }
        draw();
        const comb = (n, k) => { let v = 1; for (let i = 1; i <= k; i++) v = (v * (n - k + i)) / i; return Math.round(v); };
        note.innerHTML = t("noteKern", { dim: comb(787, 3).toLocaleString("en-US") });
        return {};
      }

      const MODES = { max: modeMax, soft: modeSoft, kern: modeKern };
      show(); // 圖都用 CSS 變數上色，切換深淺色不需重畫
      return {};
    },
  });
})();
