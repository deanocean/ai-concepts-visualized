/* k-means 法（シラバス §8）
   30 個點（三團，固定亂數產生後寫死），k = 2〜5。
   一步一步交替：①分組（每個點分到最近的中心）②移動中心（搬到組員的平均）；沒有點換組就收斂。
   初始中心：隨機挑 k 個資料點（可換一組）、故意放壞的位置、或在第 0 步直接拖動中心。
   下方：每一步的 SSE（組內距離平方和），以及 k = 1〜6 收斂後的 SSE（手肘法）。全部即時計算。
   文字在 locales/zh/kmeans.js、locales/ja/kmeans.js。 */
(function () {
  const D = [[1.41, 2.13], [1.75, 3.34], [2.28, 2], [1.64, 1.94], [1.67, 2.54], [1.86, 3.2], [2.44, 1.46], [1.63, 3.12], [1.35, 2.58], [1.85, 1.71],
    [6.25, 1.47], [5.43, 2.83], [5.99, 3.77], [6.33, 3.47], [6.33, 0.65], [7.36, 2.17], [5.83, 1.64], [6.79, 1.99], [5.96, 2.5], [6.28, 2.19],
    [4.02, 5.36], [4.03, 4.83], [4.94, 5.36], [2.75, 7.05], [4.42, 5.84], [3.62, 6.64], [3.82, 5.72], [3.48, 5.39], [4.57, 5.67], [5.28, 4.53]];
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const nearest = (p, C) => { let b = 0; for (let j = 1; j < C.length; j++) if (d2(p, C[j]) < d2(p, C[b])) b = j; return b; };
  const means = (lab, C) => C.map((c, j) => { const m = D.filter((_, i) => lab[i] === j); return m.length ? [m.reduce((a, p) => a + p[0], 0) / m.length, m.reduce((a, p) => a + p[1], 0) / m.length] : [...c]; });
  const sse = (lab, C) => D.reduce((a, p, i) => a + d2(p, C[lab[i]]), 0);
  function initIdx(k, seed) { const r = rng(seed * 7919 + k), idx = []; while (idx.length < k) { const i = Math.floor(r() * D.length); if (!idx.includes(i)) idx.push(i); } return idx; }
  /** 從初始中心一路算到收斂，記下每一步 */
  function history(C0) {
    const H = [{ phase: "init", C: C0.map((c) => [...c]), lab: null }];
    let C = C0.map((c) => [...c]), lab = null;
    for (let it = 0; it < 50; it++) {
      const nl = D.map((p) => nearest(p, C));
      const changed = lab ? nl.map((v, i) => v !== lab[i]) : nl.map(() => true);
      const nch = changed.filter(Boolean).length;
      lab = nl;
      H.push({ phase: nch === 0 ? "done" : "assign", C: C.map((c) => [...c]), lab, changed, nch, sse: sse(lab, C) });
      if (nch === 0) break;
      const NC = means(lab, C);
      H.push({ phase: "update", C: NC, prevC: C, lab, sse: sse(lab, NC), move: Math.max(...NC.map((c, j) => Math.sqrt(d2(c, C[j])))) });
      C = NC;
    }
    return H;
  }
  const finalSSE = (C0) => { const H = history(C0); return H[H.length - 1].sse; };
  // 每個 k 試 20 組隨機初始中心，取最好的（手肘法用）
  const BEST = [1, 2, 3, 4, 5, 6].map((k) => Math.min(...Array.from({ length: 20 }, (_, s) => finalSSE(initIdx(k, s + 1).map((i) => D[i])))));
  const BAD3 = [[1.5, 2.0], [2.2, 3.0], [5.5, 4.5]];

  Atlas.define("kmeans", {
    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      const { t } = ctx;
      let k = 3, init = "rand", seed = 1, custom = null, H = [], step = 0, sel = 29;

      bench.append(h("style", { text: `
        .km { --km5: #7c4dcc; }
        :root[data-theme="dark"] .km { --km5: #b79cff; }
        @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .km { --km5: #b79cff; } }
        .km-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; align-items: flex-start; }
        .km-fig { flex: 1 1 300px; max-width: 470px; min-width: 0; }
        .km-fig > svg { display: block; width: 100%; height: auto; }
        .km-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
        .km-c { cursor: grab; touch-action: none; }
        .km-p { cursor: pointer; }
      ` }));
      bench.classList.add("km");
      const COL = ["var(--pos)", "var(--hot)", "var(--ok)", "var(--neg)", "var(--km5)"];

      const kSl = ui.slider({ id: "km-k", label: t("kLabel"), min: 2, max: 5, step: 1, value: k, format: (v) => t("kFmt", { k: v }), onInput: (v) => { k = v; if (init !== "rand") { init = "rand"; initSeg.set("rand"); } rebuild(); } });
      const initSeg = ui.seg({ label: t("initLabel"), value: init, options: [{ value: "rand", label: t("initRand") }, { value: "bad", label: t("initBad") }],
        onChange: (v) => { init = v; if (v === "bad") { k = 3; kSl.set(3); } rebuild(); } });
      const seedBtn = ui.btn(t("btnSeed"), () => { init = "rand"; initSeg.set("rand"); seed = (seed % 20) + 1; rebuild(); }, { sm: true });
      const player = ui.player(ctx, () => { if (step >= H.length - 1) return false; step++; draw(); return step < H.length - 1; }, 900);
      const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (step > 0) { step--; draw(); } });
      const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (step < H.length - 1) { step++; draw(); } }, { primary: true });
      const endBtn = ui.btn(t("btnEnd"), () => { player.pause(); step = H.length - 1; draw(); });
      const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); step = 0; draw(); });

      const W = 440, Hh = 410, M = { l: 28, t: 10, r: 10, b: 26 }, U = 50, XW = 8, YH = 7.5;
      const X = (x) => M.l + x * U, Y = (y) => M.t + (YH - y) * U;
      const main = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": t("mainAria") });
      const hist = s("svg", { viewBox: "0 0 360 200", role: "img", "aria-label": t("histAria") });
      const elbow = s("svg", { viewBox: "0 0 360 200", role: "img", "aria-label": t("elbowAria") });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      bench.append(
        h("div", { class: "bench__row" }, kSl.el, initSeg.el, seedBtn),
        status,
        h("div", { class: "scroll-x" }, h("div", { class: "km-figs" }, h("div", { class: "km-fig", style: { flexBasis: "420px", maxWidth: "540px" } },
          h("div", { class: "km-cap" }, t("mainCap")), main))),
        h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, endBtn, resetBtn),
        calc,
        h("div", { class: "scroll-x" }, h("div", { class: "km-figs" },
          h("div", { class: "km-fig" }, h("div", { class: "km-cap" }, t("histCap")), hist),
          h("div", { class: "km-fig" }, h("div", { class: "km-cap" }, t("elbowCap")), elbow))),
        note);

      function rebuild() {
        player.pause();
        const C0 = init === "bad" ? BAD3 : init === "drag" ? custom : initIdx(k, seed).map((i) => D[i]);
        H = history(C0); step = 0; draw();
      }

      /* ---------- 拖動中心（只在第 0 步）、點選資料點 ---------- */
      let drag = null;
      const toSvg = (e) => { const p = main.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(main.getScreenCTM().inverse()); };
      const clampC = (v) => [Math.max(0.1, Math.min(XW - 0.1, v[0])), Math.max(0.1, Math.min(YH - 0.1, v[1]))].map((u) => Math.round(u * 20) / 20);
      function moveCenter(j, v) {
        custom = H[0].C.map((c) => [...c]); custom[j] = clampC(v);
        if (init !== "drag") { init = "drag"; initSeg.set(null); }
        H = history(custom); step = 0; draw();
      }
      main.addEventListener("pointerdown", (e) => {
        const tg = e.target;
        if (tg.dataset && tg.dataset.c != null && step === 0) { player.pause(); drag = +tg.dataset.c; try { main.setPointerCapture(e.pointerId); } catch (_) {} e.preventDefault(); }
        else if (tg.dataset && tg.dataset.p != null) { sel = +tg.dataset.p; draw(); }
      });
      main.addEventListener("pointermove", (e) => { if (drag == null) return; const q = toSvg(e); moveCenter(drag, [(q.x - M.l) / U, YH - (q.y - M.t) / U]); });
      const end = () => { drag = null; };
      main.addEventListener("pointerup", end); main.addEventListener("pointercancel", end);
      main.addEventListener("keydown", (e) => {
        const tg = e.target; if (!tg.dataset) return;
        if (tg.dataset.p != null && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sel = +tg.dataset.p; draw(); main.querySelector(`[data-p="${sel}"]`).focus(); return; }
        if (tg.dataset.c == null || step !== 0) return;
        const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key]; if (!d) return;
        e.preventDefault(); const j = +tg.dataset.c, c = H[0].C[j]; moveCenter(j, [c[0] + d[0], c[1] + d[1]]);
        const el = main.querySelector(`[data-c="${j}"]`); el && el.focus();
      });

      function diamond(c, j, attrs = {}) {
        const x = X(c[0]), y = Y(c[1]), r = 11;
        return s("polygon", { points: `${x},${y - r} ${x + r},${y} ${x},${y + r} ${x - r},${y}`, fill: COL[j], stroke: "var(--ink)", "stroke-width": 2, ...attrs });
      }

      function draw() {
        const st = H[step], C = st.C, lab = st.lab;
        main.replaceChildren();
        // 格線
        for (let x = 0; x <= XW; x++) { main.append(s("line", { x1: X(x), x2: X(x), y1: Y(0), y2: Y(YH), stroke: "var(--line)", "stroke-dasharray": x % XW ? "2 4" : null })); main.append(s("text", { x: X(x), y: Hh - 8, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: x })); }
        for (let y = 0; y <= 7; y++) { main.append(s("line", { x1: X(0), x2: X(XW), y1: Y(y), y2: Y(y), stroke: "var(--line)", "stroke-dasharray": y ? "2 4" : null })); main.append(s("text", { x: M.l - 6, y: Y(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y })); }
        main.append(s("line", { x1: X(0), x2: X(XW), y1: Y(YH), y2: Y(YH), stroke: "var(--line)" }));
        // 分組時：每個點連到自己的中心
        if (lab && st.phase !== "update") D.forEach((p, i) => main.append(s("line", { x1: X(p[0]), y1: Y(p[1]), x2: X(C[lab[i]][0]), y2: Y(C[lab[i]][1]), stroke: COL[lab[i]], "stroke-width": 1.2, "stroke-opacity": 0.55 })));
        // 選中的點
        const sp = D[sel];
        main.append(s("circle", { cx: X(sp[0]), cy: Y(sp[1]), r: 11, fill: "none", stroke: "var(--ink)", "stroke-width": 2, "stroke-dasharray": "3 2" }));
        D.forEach((p, i) => {
          const changed = st.phase === "assign" && step > 1 && st.changed[i];
          main.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: changed ? 7 : 6, fill: lab ? COL[lab[i]] : "var(--faint)", stroke: changed ? "var(--ink)" : "var(--panel)", "stroke-width": changed ? 2.5 : 1.5,
            class: "km-p", "data-p": i, tabindex: 0, role: "button", "aria-label": t("pointAria", { x: p[0], y: p[1] }) + (lab ? t("pointGroup", { g: lab[i] + 1 }) : "") + (changed ? t("pointChanged") : "") }));
        });
        // 移動中心：舊位置（空心）→ 新位置
        if (st.phase === "update") st.prevC.forEach((c, j) => {
          main.append(diamond(c, j, { fill: "none", stroke: COL[j], "stroke-dasharray": "3 2", "aria-hidden": "true" }));
          const n = C[j], L = Math.hypot(X(n[0]) - X(c[0]), Y(n[1]) - Y(c[1]));
          if (L > 16) {
            const ux = (X(n[0]) - X(c[0])) / L, uy = (Y(n[1]) - Y(c[1])) / L, ex = X(n[0]) - ux * 13, ey = Y(n[1]) - uy * 13;
            main.append(s("line", { x1: X(c[0]) + ux * 12, y1: Y(c[1]) + uy * 12, x2: ex, y2: ey, stroke: "var(--ink)", "stroke-width": 2 }));
            main.append(s("polygon", { points: `${ex + ux * 2},${ey + uy * 2} ${ex - ux * 8 - uy * 5},${ey - uy * 8 + ux * 5} ${ex - ux * 8 + uy * 5},${ey - uy * 8 - ux * 5}`, fill: "var(--ink)" }));
          }
        });
        C.forEach((c, j) => {
          main.append(diamond(c, j, step === 0 ? { class: "km-c", "data-c": j, tabindex: 0, role: "button", "aria-label": t("centerAriaDrag", { j: j + 1 }) } : { "aria-label": t("centerAria", { j: j + 1 }) }));
          main.append(s("text", { x: X(c[0]), y: Y(c[1]) + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: "var(--panel)", text: j + 1, style: "pointer-events:none" }));
        });

        /* --- 狀態 --- */
        const set = (kd, msg) => { status.className = "status status--" + kd; status.textContent = msg; };
        const best = BEST[k - 1], n = step;
        if (st.phase === "init") set("info", init === "bad" ? t("statusInitBad", { n }) : init === "drag" ? t("statusInitDrag", { n }) : t("statusInitRand", { n, k }));
        else if (st.phase === "assign") set("info", step === 1 ? t("statusAssign1", { n }) : t("statusAssign", { n, nch: st.nch }));
        else if (st.phase === "update") set("info", t("statusUpdate", { n, move: fmt(st.move, 2) }));
        else if (st.sse > best * 1.02) set("warn", t("statusStuck", { n, sse: fmt(st.sse, 2), k, best: fmt(best, 2), ratio: fmt(st.sse / best, 1) }));
        else set("ok", t("statusDone", { n, sse: fmt(st.sse, 2) }));

        /* --- 算式 --- */
        const row = (kk, b) => `<span class="calc-row"><span class="k">${kk}</span> ${b}</span>`;
        const pp = `(${fmt(sp[0], 2)}, ${fmt(sp[1], 2)})`;
        if (st.phase === "init") calc.innerHTML = row(t("calcNextK"), t("calcNext", { p: pp }));
        else if (st.phase === "update") {
          const j = lab[sel], mem = D.filter((_, i) => lab[i] === j), sx = mem.reduce((a, p) => a + p[0], 0), sy = mem.reduce((a, p) => a + p[1], 0);
          calc.innerHTML = row(t("calcCenterK", { j: j + 1 }), t("calcCenter", { j: j + 1, n: mem.length, sx: fmt(sx, 2), sy: fmt(sy, 2), cx: fmt(C[j][0], 2), cy: fmt(C[j][1], 2) })) +
            row(t("calcSseK"), t("calcSseUpdate", { sse: fmt(st.sse, 2), prev: fmt(H[step - 1].sse, 2) }));
        } else {
          const ds = C.map((c) => d2(sp, c)), b = lab[sel];
          const item = (v, j) => t("calcDistItem", { j: j + 1, v: fmt(v, 2) });
          calc.innerHTML = row(t("calcDistK", { p: pp }), ds.map((v, j) => (j === b ? `<span class="hl">${item(v, j)}</span>` : item(v, j))).join("　") + "　" + t("calcDistTo", { b: b + 1 })) +
            row(t("calcSseK"), step > 1 ? t("calcSseAssign", { sse: fmt(st.sse, 2), prev: fmt(H[step - 1].sse, 2) }) : t("calcSse", { sse: fmt(st.sse, 2) }));
        }
        drawHist(); drawElbow();
        prevBtn.disabled = resetBtn.disabled = step === 0;
        nextBtn.disabled = endBtn.disabled = step >= H.length - 1;
      }

      function axes(svg, m, CW, CH, top, ticks) {
        const py = (v) => CH - m.b - (v / top) * (CH - m.t - m.b);
        ticks.forEach((v) => {
          svg.append(s("line", { x1: m.l, x2: CW - m.r, y1: py(v), y2: py(v), stroke: v ? "var(--line)" : "var(--line-strong)", "stroke-dasharray": v ? "3 3" : null }));
          svg.append(s("text", { x: m.l - 5, y: py(v) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: v }));
        });
        return py;
      }
      function drawHist() {
        const CW = 360, CH = 200, m = { l: 40, r: 14, t: 14, b: 32 };
        hist.replaceChildren();
        const pts = H.map((st, i) => [i, st.sse]).filter((v) => v[1] != null);
        const top = Math.max(50, Math.ceil(Math.max(...pts.map((v) => v[1])) / 50) * 50);
        const py = axes(hist, m, CW, CH, top, Array.from({ length: Math.round(top / 50) + 1 }, (_, i) => i * 50).filter((v, i, a) => a.length <= 6 || i % 2 === 0));
        const N = Math.max(H.length - 1, 4), px = (i) => m.l + (i / N) * (CW - m.l - m.r);
        for (let i = 0; i <= N; i++) hist.append(s("text", { x: px(i), y: CH - m.b + 14, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: i }));
        hist.append(s("text", { x: CW - m.r, y: CH - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("histX") }));
        const shown = pts.filter((v) => v[0] <= step);
        if (shown.length) {
          hist.append(s("path", { d: shown.map((v, i) => (i ? "L" : "M") + px(v[0]).toFixed(1) + "," + py(v[1]).toFixed(1)).join(""), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
          shown.forEach((v) => hist.append(s("circle", { cx: px(v[0]), cy: py(v[1]), r: v[0] === step ? 5 : 3, fill: H[v[0]].phase === "update" ? "var(--accent)" : "var(--panel)", stroke: "var(--accent)", "stroke-width": 2 })));
          const last = shown[shown.length - 1], right = px(last[0]) > CW - 90;
          hist.append(s("text", { x: px(last[0]) + (right ? -8 : 8), y: py(last[1]) < m.t + 16 ? py(last[1]) + 16 : py(last[1]) - 8, "text-anchor": right ? "end" : "start", "font-size": 11, "font-weight": 700, class: "t-mono t-halo", text: fmt(last[1], 2) }));
        }
      }
      function drawElbow() {
        const CW = 360, CH = 200, m = { l: 40, r: 14, t: 14, b: 32 };
        elbow.replaceChildren();
        const py = axes(elbow, m, CW, CH, 200, [0, 50, 100, 150, 200]);
        const px = (kk) => m.l + ((kk - 1) / 5) * (CW - m.l - m.r);
        elbow.append(s("rect", { x: px(k) - 11, y: m.t, width: 22, height: CH - m.t - m.b, fill: "var(--hot-soft)" }));
        for (let kk = 1; kk <= 6; kk++) elbow.append(s("text", { x: px(kk), y: CH - m.b + 14, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: kk }));
        elbow.append(s("text", { x: CW - m.r, y: CH - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: t("elbowX") }));
        elbow.append(s("path", { d: BEST.map((v, i) => (i ? "L" : "M") + px(i + 1).toFixed(1) + "," + py(v).toFixed(1)).join(""), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }));
        BEST.forEach((v, i) => {
          elbow.append(s("circle", { cx: px(i + 1), cy: py(v), r: 3.5, fill: "var(--ink)" }));
          elbow.append(s("text", { x: i === 5 ? px(i + 1) : px(i + 1) + 6, y: py(v) - 7, "text-anchor": i === 5 ? "middle" : "start", "font-size": 10, class: "t-mono t-halo", text: fmt(v, 1) }));
        });
        elbow.append(s("text", { x: px(3) + 8, y: py(BEST[2]) - 26, "font-size": 11, "font-weight": 700, fill: "var(--accent)", class: "t-halo", text: t("elbowHint") }));
      }

      rebuild();
      note.innerHTML = t("note");
      return { destroy() { player.pause(); } };
    },
  });
})();
