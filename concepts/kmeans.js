/* k-means 法（シラバス §8）
   30 個點（三團，固定亂數產生後寫死），k = 2〜5。
   一步一步交替：①分組（每個點分到最近的中心）②移動中心（搬到組員的平均）；沒有點換組就收斂。
   初始中心：隨機挑 k 個資料點（可換一組）、故意放壞的位置、或在第 0 步直接拖動中心。
   下方：每一步的 SSE（組內距離平方和），以及 k = 1〜6 收斂後的 SSE（エルボー法）。全部即時計算。 */
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
  // 每個 k 試 20 組隨機初始中心，取最好的（エルボー法用）
  const BEST = [1, 2, 3, 4, 5, 6].map((k) => Math.min(...Array.from({ length: 20 }, (_, s) => finalSSE(initIdx(k, s + 1).map((i) => D[i])))));
  const BAD3 = [[1.5, 2.0], [2.2, 3.0], [5.5, 4.5]];

  Atlas.define("kmeans", {
    lead: `<p>手上只有資料、沒有正解標籤，想把相似的資料自動分成幾群（<span class="jp">クラスタリング</span>，分群，<span class="en">clustering</span>），例如依購買紀錄把顧客分群。<span class="jp">k-means 法</span>（<span class="en">k-means clustering</span>）先決定要分 k 群、隨便放 k 個中心，然後只重複兩件簡單的事，直到不再變化。</p>`,
    closing: `k-means 法只做兩件事：每個點歸給最近的中心、每個中心搬到組員的平均位置；每一步都讓組內距離平方和變小，所以一定會停下來。但停下來的地方取決於初始中心（可能卡在比較差的結果），而且 k 要由人決定，可以用不同 k 的結果比較來輔助判斷。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
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

      const kSl = ui.slider({ id: "km-k", label: "群數 k", min: 2, max: 5, step: 1, value: k, format: (v) => v + " 群", onInput: (v) => { k = v; if (init !== "rand") { init = "rand"; initSeg.set("rand"); } rebuild(); } });
      const initSeg = ui.seg({ label: "初始中心", value: init, options: [{ value: "rand", label: "隨機挑 k 個點" }, { value: "bad", label: "故意放壞的位置（k = 3）" }],
        onChange: (v) => { init = v; if (v === "bad") { k = 3; kSl.set(3); } rebuild(); } });
      const seedBtn = ui.btn("換一組隨機初始", () => { init = "rand"; initSeg.set("rand"); seed = (seed % 20) + 1; rebuild(); }, { sm: true });
      const player = ui.player(ctx, () => { if (step >= H.length - 1) return false; step++; draw(); return step < H.length - 1; }, 900);
      const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (step > 0) { step--; draw(); } });
      const nextBtn = ui.btn("下一步 ▶", () => { player.pause(); if (step < H.length - 1) { step++; draw(); } }, { primary: true });
      const endBtn = ui.btn("跑到收斂", () => { player.pause(); step = H.length - 1; draw(); });
      const resetBtn = ui.btn("回到第 0 步", () => { player.pause(); step = 0; draw(); });

      const W = 440, Hh = 410, M = { l: 28, t: 10, r: 10, b: 26 }, U = 50, XW = 8, YH = 7.5;
      const X = (x) => M.l + x * U, Y = (y) => M.t + (YH - y) * U;
      const main = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "資料點、各群中心與分組" });
      const hist = s("svg", { viewBox: "0 0 360 200", role: "img", "aria-label": "每一步的組內距離平方和" });
      const elbow = s("svg", { viewBox: "0 0 360 200", role: "img", "aria-label": "不同群數收斂後的組內距離平方和" });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      bench.append(
        h("div", { class: "bench__row" }, kSl.el, initSeg.el, seedBtn),
        status,
        h("div", { class: "scroll-x" }, h("div", { class: "km-figs" }, h("div", { class: "km-fig", style: { flexBasis: "420px", maxWidth: "540px" } },
          h("div", { class: "km-cap" }, "◆＝中心（第 0 步可以拖動）；點一個資料點，看它的計算"), main))),
        h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, endBtn, resetBtn),
        calc,
        h("div", { class: "scroll-x" }, h("div", { class: "km-figs" },
          h("div", { class: "km-fig" }, h("div", { class: "km-cap" }, "每一步之後的 SSE（組內距離平方和；○ 分組後、● 移動中心後）"), hist),
          h("div", { class: "km-fig" }, h("div", { class: "km-cap" }, "群數 k 與收斂後的 SSE（每個 k 試 20 組初始取最好）"), elbow))),
        note);

      function rebuild() {
        player.pause();
        const C0 = init === "bad" ? BAD3 : init === "drag" ? custom : initIdx(k, seed).map((i) => D[i]);
        H = history(C0); step = 0; draw();
      }

      /* ---------- 拖動中心（只在第 0 步）、點選資料點 ---------- */
      let drag = null;
      const toSvg = (e) => { const p = main.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(main.getScreenCTM().inverse()); };
      const clampC = (v) => [Math.max(0.1, Math.min(XW - 0.1, v[0])), Math.max(0.1, Math.min(YH - 0.1, v[1]))].map((t) => Math.round(t * 20) / 20);
      function moveCenter(j, v) {
        custom = H[0].C.map((c) => [...c]); custom[j] = clampC(v);
        if (init !== "drag") { init = "drag"; initSeg.set(null); }
        H = history(custom); step = 0; draw();
      }
      main.addEventListener("pointerdown", (e) => {
        const t = e.target;
        if (t.dataset && t.dataset.c != null && step === 0) { player.pause(); drag = +t.dataset.c; try { main.setPointerCapture(e.pointerId); } catch (_) {} e.preventDefault(); }
        else if (t.dataset && t.dataset.p != null) { sel = +t.dataset.p; draw(); }
      });
      main.addEventListener("pointermove", (e) => { if (drag == null) return; const q = toSvg(e); moveCenter(drag, [(q.x - M.l) / U, YH - (q.y - M.t) / U]); });
      const end = () => { drag = null; };
      main.addEventListener("pointerup", end); main.addEventListener("pointercancel", end);
      main.addEventListener("keydown", (e) => {
        const t = e.target; if (!t.dataset) return;
        if (t.dataset.p != null && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sel = +t.dataset.p; draw(); main.querySelector(`[data-p="${sel}"]`).focus(); return; }
        if (t.dataset.c == null || step !== 0) return;
        const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key]; if (!d) return;
        e.preventDefault(); const j = +t.dataset.c, c = H[0].C[j]; moveCenter(j, [c[0] + d[0], c[1] + d[1]]);
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
            class: "km-p", "data-p": i, tabindex: 0, role: "button", "aria-label": `資料點 (${p[0]}, ${p[1]})${lab ? `，第 ${lab[i] + 1} 群` : ""}${changed ? "，這一步換了組" : ""}` }));
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
          main.append(diamond(c, j, step === 0 ? { class: "km-c", "data-c": j, tabindex: 0, role: "button", "aria-label": `第 ${j + 1} 個中心，可拖動` } : { "aria-label": `第 ${j + 1} 個中心` }));
          main.append(s("text", { x: X(c[0]), y: Y(c[1]) + 4, "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: "var(--panel)", text: j + 1, style: "pointer-events:none" }));
        });

        /* --- 狀態 --- */
        const set = (kd, t) => { status.className = "status status--" + kd; status.textContent = t; };
        const best = BEST[k - 1];
        const head = `第 ${step} 步：`;
        if (st.phase === "init") set("info", head + (init === "bad" ? "初始中心故意擠在左下那一團（1、2 號）" : init === "drag" ? "用你拖動的位置當初始中心" : `隨機挑了 ${k} 個資料點當初始中心`) + "。按「下一步」開始");
        else if (st.phase === "assign") set("info", head + (step === 1 ? "分組：每個點歸給最近的中心" : `分組：有 ${st.nch} 個點換了組（粗黑框）`));
        else if (st.phase === "update") set("info", head + `移動中心：每個中心搬到組員的平均位置，最多移動 ${fmt(st.move, 2)}`);
        else if (st.sse > best * 1.02) set("warn", head + `收斂了（沒有點換組），但 SSE = ${fmt(st.sse, 2)}，比 k = ${k} 時能達到的 ${fmt(best, 2)} 大 ${fmt(st.sse / best, 1)} 倍：卡在比較差的結果`);
        else set("ok", head + `收斂：沒有點換組，中心不再移動。SSE = ${fmt(st.sse, 2)}`);

        /* --- 算式 --- */
        const row = (kk, b) => `<span class="calc-row"><span class="k">${kk}</span> ${b}</span>`;
        const pp = `(${fmt(sp[0], 2)}, ${fmt(sp[1], 2)})`;
        if (st.phase === "init") calc.innerHTML = row("下一步", `選中的點 ${pp}：算出它到每個中心的距離，歸給最近的那一個`);
        else if (st.phase === "update") {
          const j = lab[sel], mem = D.filter((_, i) => lab[i] === j), sx = mem.reduce((a, p) => a + p[0], 0), sy = mem.reduce((a, p) => a + p[1], 0);
          calc.innerHTML = row(`中心 ${j + 1} 的新位置`, `選中的點所在的第 ${j + 1} 群有 ${mem.length} 個點，平均 = (${fmt(sx, 2)} ÷ ${mem.length}, ${fmt(sy, 2)} ÷ ${mem.length}) = <span class="hl">(${fmt(C[j][0], 2)}, ${fmt(C[j][1], 2)})</span>`) +
            row("SSE", `每個點到自己中心的距離² 加總 = ${fmt(st.sse, 2)}（移動前 ${fmt(H[step - 1].sse, 2)}）`);
        } else {
          const ds = C.map((c) => d2(sp, c)), b = lab[sel];
          calc.innerHTML = row(`點 ${pp} 到各中心的距離²`, ds.map((v, j) => (j === b ? `<span class="hl">中心 ${j + 1}：${fmt(v, 2)}</span>` : `中心 ${j + 1}：${fmt(v, 2)}`)).join("　") + `　→ 歸給中心 ${b + 1}`) +
            row("SSE", `每個點到自己中心的距離² 加總 = ${fmt(st.sse, 2)}` + (step > 1 ? `（分組前 ${fmt(H[step - 1].sse, 2)}）` : ""));
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
        hist.append(s("text", { x: CW - m.r, y: CH - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: "步" }));
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
        elbow.append(s("text", { x: CW - m.r, y: CH - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: "群數 k" }));
        elbow.append(s("path", { d: BEST.map((v, i) => (i ? "L" : "M") + px(i + 1).toFixed(1) + "," + py(v).toFixed(1)).join(""), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }));
        BEST.forEach((v, i) => {
          elbow.append(s("circle", { cx: px(i + 1), cy: py(v), r: 3.5, fill: "var(--ink)" }));
          elbow.append(s("text", { x: i === 5 ? px(i + 1) : px(i + 1) + 6, y: py(v) - 7, "text-anchor": i === 5 ? "middle" : "start", "font-size": 10, class: "t-mono t-halo", text: fmt(v, 1) }));
        });
        elbow.append(s("text", { x: px(3) + 8, y: py(BEST[2]) - 26, "font-size": 11, "font-weight": 700, fill: "var(--accent)", class: "t-halo", text: "← 從這裡開始下降變緩" }));
      }

      rebuild();
      note.innerHTML = `中心的位置就是組員的平均（mean），名稱裡的「means」由此而來；k 是要分幾群。SSE（組內距離平方和）在「分組」和「移動中心」兩步都只會變小，所以一定會停，但停的地方不一定是最好的：試「故意放壞的位置」，兩個中心擠在同一團，最後把右邊和上面兩團併成一群。實務上會換幾組初始值多跑幾次，或用 k-means++（讓初始中心彼此離遠一點）。群數越多 SSE 一定越小（每個點自成一群時是 0），所以不能只看 SSE 選 k；右下圖下降突然變緩的地方（像手肘）是常用的參考，叫エルボー法。不必事先決定群數、由近到遠一步步合併成樹狀圖（<span class="jp">デンドログラム</span>）的方法有<span class="jp">ウォード法</span>等。另一種常見的教師なし学習是降維，見<a href="#pca">主成分分析</a>。`;
      return { destroy() { player.pause(); } };
    },
  });
})();
