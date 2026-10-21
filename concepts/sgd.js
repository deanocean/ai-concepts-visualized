/* 確率的勾配降下法とミニバッチ（シラバス §16）
   16 筆資料，模型 y = a·x + b，損失 L = Σ(a·x + b − y)² ÷ (2n)。
   一次用 16／4／1 筆算梯度（バッチ学習／ミニバッチ学習／確率的勾配降下法），
   左：資料與目前的直線；右：參數 (a, b) 平面上的等高線（橢圓，依 Hessian 精確畫出）與路徑。
   每個エポック用固定種子洗牌，所以每次重設結果相同；下方比較圖是三種做法同樣跑 5 個エポック的實際損失。 */
(function () {
  const DATA = [
    { x: -0.98, y: 0.1 }, { x: -0.8, y: -1.02 }, { x: -0.69, y: -1.1 }, { x: -0.54, y: 0.32 },
    { x: -0.42, y: 0.34 }, { x: -0.32, y: -0.33 }, { x: -0.2, y: 0.59 }, { x: -0.03, y: 1.44 },
    { x: 0.03, y: 0.39 }, { x: 0.21, y: 2.29 }, { x: 0.32, y: 1.45 }, { x: 0.47, y: 2.33 },
    { x: 0.59, y: 2.7 }, { x: 0.68, y: 2.94 }, { x: 0.78, y: 2.34 }, { x: 0.93, y: 3.39 },
  ];
  const N = DATA.length;
  const mean = (f) => DATA.reduce((s, p) => s + f(p), 0) / N;
  const MX = mean((p) => p.x), MXX = mean((p) => p.x * p.x), MY = mean((p) => p.y), MXY = mean((p) => p.x * p.y);
  const A_OPT = (MXY - MX * MY) / (MXX - MX * MX), B_OPT = MY - A_OPT * MX;
  const loss = (a, b) => DATA.reduce((s, p) => s + (a * p.x + b - p.y) ** 2, 0) / (2 * N);
  const L_OPT = loss(A_OPT, B_OPT);
  function grad(a, b, idx) {
    let ga = 0, gb = 0;
    for (const i of idx) { const e = a * DATA[i].x + b - DATA[i].y; ga += e * DATA[i].x; gb += e; }
    return [ga / idx.length, gb / idx.length];
  }
  const ALL = DATA.map((_, i) => i);
  function rng(seed) { return function () { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  /** 第 e 個エポック的洗牌順序（固定種子） */
  const PERMS = [];
  function perm(e) {
    if (!PERMS[e]) {
      const R = rng(9100 + e), idx = ALL.slice();
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      PERMS[e] = idx;
    }
    return PERMS[e];
  }
  /** 第 n 次更新（從 0 起算）用到哪些資料 */
  function batchAt(n, B) {
    const per = N / B, e = Math.floor(n / per), k = n % per;
    return { e, k, idx: perm(e).slice(k * B, k * B + B) };
  }
  /** 從 start 開始跑 epochs 個エポック，回傳每次更新後的位置 */
  function run(start, B, eta, epochs) {
    const pts = [start.slice()];
    for (let n = 0; n < (epochs * N) / B; n++) {
      const [a, b] = pts[pts.length - 1], [ga, gb] = grad(a, b, batchAt(n, B).idx);
      pts.push([a - eta * ga, b - eta * gb]);
    }
    return pts;
  }

  Atlas.define("sgd", {
    lead: `<p><a href="#gd">勾配降下法</a>每走一步，都要把<b>全部</b>資料的誤差算一遍才知道坡度。資料有幾百萬筆時，算一次坡度就要很久，走一步也很久。<span class="jp">確率的勾配降下法</span>（隨機梯度下降法，<span class="en">stochastic gradient descent, SGD</span>）改成每次只隨機拿一小部分資料來估坡度：估得沒那麼準，但同樣的計算量可以走好多步。</p>`,
    closing: `一次用全部資料（<span class="jp">バッチ学習</span>）方向最準，但一個エポック只能走一步；一次用 1 筆，一個エポック走 16 步，很快靠近谷底，但方向搖擺、到了谷底附近仍停不下來。<span class="jp">ミニバッチ学習</span>介於兩者之間，是現在訓練神經網路的標準做法。`,

    mount(bench, ctx) {
      const { h, s, fmt, fmtP, ui } = Atlas;
      let B = 4, eta = 0.5, start = [-1.5, -1.5], hist = [start.slice()];
      const MAXEP = 20;

      bench.append(h("style", { text: `
        .sg-figs { display: flex; flex-wrap: wrap; gap: 12px 18px; justify-content: center; }
        .sg-fig { flex: 1 1 300px; max-width: 470px; min-width: 0; }
        .sg-fig svg { display: block; width: 100%; height: auto; }
        .sg-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 2px; }
        .sg-count { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }
        .sg-count b { font-family: var(--f-mono); color: var(--ink); font-size: 15px; }
      ` }));

      const bSeg = ui.seg({
        label: "一次用幾筆資料算梯度（バッチサイズ）", value: B,
        options: [{ value: 16, label: "全部 16 筆" }, { value: 4, label: "4 筆" }, { value: 1, label: "1 筆" }],
        onChange: (v) => { B = v; reset(); },
      });
      const etaSlider = ui.slider({ id: "sg-eta", label: "学習率 η", min: 0.05, max: 1.2, step: 0.05, value: eta, format: (v) => v.toFixed(2), onInput: (v) => { eta = v; reset(); } });
      const dataSvg = s("svg", { viewBox: "0 0 440 330", role: "img", "aria-label": "資料點與目前的直線" });
      const parSvg = s("svg", { viewBox: "0 0 440 360", role: "img", "aria-label": "參數平面上的損失等高線與路徑", style: "cursor:crosshair" });
      const figs = h("div", { class: "sg-figs" },
        h("div", { class: "sg-fig" }, h("div", { class: "sg-cap" }, "資料與目前的直線 y = a·x + b"), dataSvg),
        h("div", { class: "sg-fig" }, h("div", { class: "sg-cap" }, "參數 (a, b) 的損失地形（用全部資料算）"), parSvg));
      const count = h("div", { class: "sg-count" });
      const status = h("div", { class: "status" });
      const player = ui.player(ctx, () => step(), 380);
      const stepBtn = ui.btn("更新一次", () => { player.pause(); step(); }, { primary: true });
      const epBtn = ui.btn("跑完這個エポック", () => { player.pause(); finishEpoch(); });
      const resetBtn = ui.btn("重設", () => { player.pause(); reset(); });
      const calc = h("div", { class: "calc" });
      const cmpSvg = s("svg", { viewBox: "0 0 640 220", role: "img", "aria-label": "三種做法跑 5 個エポック的損失比較", style: "display:block;width:100%;min-width:520px;max-width:760px" });
      const cmpLegend = h("p", { class: "hint" });
      const note = h("p", { class: "note" });

      bench.append(
        h("div", { class: "bench__row" }, bSeg.el, etaSlider.el),
        h("div", { class: "scroll-x" }, figs),
        count,
        status,
        h("div", { class: "bench__row" }, stepBtn, epBtn, player.el, resetBtn),
        calc,
        h("div", null, h("p", { class: "field__label" }, "同樣算 5 個エポック（每種做法都看過每筆資料 5 次），整份資料的損失比較"), h("div", { class: "scroll-x" }, cmpSvg), cmpLegend),
        note,
      );

      const nUpd = () => hist.length - 1;
      const per = () => N / B;
      const cur = () => hist[hist.length - 1];
      const done = () => nUpd() >= MAXEP * per();
      function step() {
        if (done()) return false;
        const [a, b] = cur(), [ga, gb] = grad(a, b, batchAt(nUpd(), B).idx);
        hist.push([a - eta * ga, b - eta * gb]);
        render();
        return !done();
      }
      function finishEpoch() {
        do { if (step() === false) break; } while (nUpd() % per() !== 0);
      }
      function reset() { hist = [start.slice()]; render(); }

      /* ---------- 左：資料 ---------- */
      function drawData() {
        const W = 440, H = 330, M = { l: 36, r: 12, t: 12, b: 30 };
        const X0 = -1.1, X1 = 1.1, Y0 = -2, Y1 = 4;
        const sx = (x) => M.l + ((x - X0) / (X1 - X0)) * (W - M.l - M.r);
        const sy = (y) => H - M.b - ((y - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
        dataSvg.replaceChildren();
        for (let y = -2; y <= 4; y++) {
          dataSvg.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(y), y2: sy(y), stroke: "var(--line)", "stroke-width": y === 0 ? 1.4 : 1 }));
          dataSvg.append(s("text", { x: M.l - 6, y: sy(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y < 0 ? "−" + -y : y }));
        }
        for (const x of [-1, -0.5, 0, 0.5, 1]) dataSvg.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: x < 0 ? "−" + -x : x }));
        dataSvg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 11.5, class: "t-muted", text: "x" }));
        const [a, b] = cur();
        const info = done() ? null : batchAt(nUpd(), B);
        const inBatch = new Set(info ? info.idx : []);
        const used = new Set();
        if (info) perm(info.e).slice(0, info.k * B).forEach((i) => used.add(i));
        // 目前的直線
        dataSvg.append(s("defs", null, s("clipPath", { id: "sgclip" }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        const g = s("g", { "clip-path": "url(#sgclip)" });
        // 這批的誤差（點到直線的垂直距離）
        for (const i of inBatch) {
          const p = DATA[i];
          g.append(s("line", { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(a * p.x + b), stroke: "var(--hot)", "stroke-width": 2, "stroke-dasharray": "4 3" }));
        }
        g.append(s("line", { x1: sx(X0), y1: sy(a * X0 + b), x2: sx(X1), y2: sy(a * X1 + b), stroke: "var(--accent)", "stroke-width": 2.5 }));
        dataSvg.append(g);
        DATA.forEach((p, i) => {
          const on = inBatch.has(i), u = used.has(i);
          dataSvg.append(s("circle", { cx: sx(p.x), cy: sy(p.y), r: on ? 6.5 : 4.5,
            fill: on ? "var(--hot)" : u ? "var(--line-strong)" : "var(--panel)",
            stroke: on ? "var(--hot)" : u ? "var(--line-strong)" : "var(--ink)", "stroke-width": 1.5 }));
        });
        // 這批的編號：預設放在點上方，和已放的標籤太近就改放下方
        if (B < 16) {
          const placed = [];
          [...inBatch].sort((i, j) => DATA[i].x - DATA[j].x).forEach((i) => {
            const X = sx(DATA[i].x);
            let Y = sy(DATA[i].y) - 11;
            if (placed.some((q) => Math.abs(q[0] - X) < 28 && Math.abs(q[1] - Y) < 14)) Y = sy(DATA[i].y) + 20;
            placed.push([X, Y]);
            dataSvg.append(s("text", { x: X, y: Y, "text-anchor": "middle", "font-size": 11, "font-weight": 700, class: "t-mono t-halo", fill: "var(--hot)", text: "#" + (i + 1) }));
          });
        }
        // 圖例
        const lg = s("g", { transform: `translate(${M.l + 8},${M.t + 6})` });
        lg.append(s("rect", { x: -6, y: -4, width: B < 16 ? 252 : 170, height: 22, rx: 6, fill: "var(--panel)", stroke: "var(--line)" }));
        lg.append(s("circle", { cx: 4, cy: 7, r: 5, fill: "var(--hot)" }));
        lg.append(s("text", { x: 14, y: 11, "font-size": 11.5, text: B < 16 ? "這一批" : "全部用上" }));
        if (B < 16) {
          lg.append(s("circle", { cx: 84, cy: 7, r: 4.5, fill: "var(--line-strong)" }));
          lg.append(s("text", { x: 94, y: 11, "font-size": 11.5, text: "這エポック用過" }));
          lg.append(s("circle", { cx: 186, cy: 7, r: 4.5, fill: "var(--panel)", stroke: "var(--ink)", "stroke-width": 1.5 }));
          lg.append(s("text", { x: 196, y: 11, "font-size": 11.5, text: "還沒用" }));
        } else {
          lg.append(s("line", { x1: 82, x2: 100, y1: 7, y2: 7, stroke: "var(--hot)", "stroke-width": 2, "stroke-dasharray": "4 3" }));
          lg.append(s("text", { x: 106, y: 11, "font-size": 11.5, text: "誤差" }));
        }
        dataSvg.append(lg);
      }

      /* ---------- 右：參數平面 ---------- */
      const PW = 440, PH = 360, PM = { l: 34, r: 10, t: 10, b: 30 };
      const A0 = -2.5, A1 = 4.5, B0 = -2.5, B1 = 3.5;
      // 長寬同比例，橢圓才不會變形
      const unit = Math.min((PW - PM.l - PM.r) / (A1 - A0), (PH - PM.t - PM.b) / (B1 - B0));
      const pa = (a) => PM.l + (a - A0) * unit;
      const pb = (b) => PM.t + (B1 - b) * unit;
      // L(a,b) − L* = ½ ΔᵀHΔ，H = [[E(x²), E(x)],[E(x), 1]]
      const Hm = [[MXX, MX], [MX, 1]];
      const tr = Hm[0][0] + Hm[1][1], det = Hm[0][0] * Hm[1][1] - Hm[0][1] * Hm[1][0];
      const l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      const v1 = Math.abs(Hm[0][1]) > 1e-12 ? [l1 - Hm[1][1], Hm[0][1]] : [0, 1];
      const n1 = Math.hypot(...v1), e1 = [v1[0] / n1, v1[1] / n1], e2 = [-e1[1], e1[0]];
      const LEVELS = [0.02, 0.06, 0.15, 0.35, 0.7, 1.3, 2.3, 3.8, 6, 9];
      function ellipse(c) {
        const r1 = Math.sqrt((2 * c) / l1), r2 = Math.sqrt((2 * c) / l2);
        let d = "";
        for (let i = 0; i <= 72; i++) {
          const t = (i / 72) * Math.PI * 2, u = r1 * Math.cos(t), v = r2 * Math.sin(t);
          d += (i ? "L" : "M") + pa(A_OPT + u * e1[0] + v * e2[0]).toFixed(1) + "," + pb(B_OPT + u * e1[1] + v * e2[1]).toFixed(1);
        }
        return d + "Z";
      }
      parSvg.addEventListener("click", (e) => {
        const pt = parSvg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const p = pt.matrixTransform(parSvg.getScreenCTM().inverse());
        const a = A0 + (p.x - PM.l) / unit, b = B1 - (p.y - PM.t) / unit;
        if (a < A0 || a > A1 || b < B0 || b > B1) return;
        start = [Math.round(a * 10) / 10, Math.round(b * 10) / 10];
        player.pause(); reset(); drawCompare();
      });
      function arrow(g, x1, y1, x2, y2, color, dash) {
        const L = Math.hypot(x2 - x1, y2 - y1); if (L < 3) return;
        const ux = (x2 - x1) / L, uy = (y2 - y1) / L;
        g.append(s("line", { x1, y1, x2: x2 - ux * 6, y2: y2 - uy * 6, stroke: color, "stroke-width": 2.2, "stroke-dasharray": dash || null }));
        g.append(s("path", { d: `M${x2},${y2} L${x2 - ux * 9 - uy * 5},${y2 - uy * 9 + ux * 5} L${x2 - ux * 9 + uy * 5},${y2 - uy * 9 - ux * 5}Z`, fill: color }));
      }
      function drawParams() {
        parSvg.replaceChildren();
        const lo = Atlas.color("--seq-lo"), hi = Atlas.color("--seq-hi");
        parSvg.append(s("defs", null, s("clipPath", { id: "sgpclip" }, s("rect", { x: PM.l, y: PM.t, width: (A1 - A0) * unit, height: (B1 - B0) * unit }))));
        const g = s("g", { "clip-path": "url(#sgpclip)" });
        g.append(s("rect", { x: PM.l, y: PM.t, width: (A1 - A0) * unit, height: (B1 - B0) * unit, fill: Atlas.rgbStr(Atlas.mix(lo, hi, 0.78)) }));
        for (let k = LEVELS.length - 1; k >= 0; k--) {
          const t = (k / (LEVELS.length - 1)) * 0.7;
          g.append(s("path", { d: ellipse(LEVELS[k]), fill: Atlas.rgbStr(Atlas.mix(lo, hi, t)), stroke: Atlas.rgbStr(Atlas.mix(lo, hi, Math.min(1, t + 0.25))), "stroke-width": 1 }));
        }
        // 谷底
        g.append(s("circle", { cx: pa(A_OPT), cy: pb(B_OPT), r: 5.5, fill: "var(--ok)", stroke: "var(--panel)", "stroke-width": 2 }));
        // 路徑
        const pts = hist.filter((p) => isFinite(p[0]) && isFinite(p[1]) && Math.abs(p[0]) < 1e3 && Math.abs(p[1]) < 1e3);
        if (pts.length > 1) g.append(s("path", { d: pts.map((p, i) => (i ? "L" : "M") + pa(p[0]).toFixed(1) + "," + pb(p[1]).toFixed(1)).join(""), fill: "none", stroke: "var(--hot)", "stroke-width": 1.8, "stroke-linejoin": "round" }));
        pts.slice(0, -1).forEach((p) => g.append(s("circle", { cx: pa(p[0]), cy: pb(p[1]), r: 2.4, fill: "var(--hot)" })));
        // 下一步：這批的梯度 vs 全部資料的梯度
        const [a, b] = cur();
        if (!done() && isFinite(a)) {
          const [fa, fb] = grad(a, b, ALL), [ga, gb] = grad(a, b, batchAt(nUpd(), B).idx);
          if (B < 16) arrow(g, pa(a), pb(b), pa(a - eta * fa), pb(b - eta * fb), "var(--muted)", "5 4");
          arrow(g, pa(a), pb(b), pa(a - eta * ga), pb(b - eta * gb), "var(--hot)");
        }
        g.append(s("circle", { cx: pa(start[0]), cy: pb(start[1]), r: 9, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "3 3" }));
        if (isFinite(a) && Math.abs(a) < 1e3) g.append(s("circle", { cx: pa(a), cy: pb(b), r: 7, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2.5 }));
        parSvg.append(g);
        parSvg.append(s("text", { x: pa(A_OPT) + 10, y: pb(B_OPT) - 9, "font-size": 11.5, "font-weight": 700, class: "t-halo", fill: "var(--ok)", text: "谷底" }));
        // 軸
        for (let v = -2; v <= 4; v++) parSvg.append(s("text", { x: pa(v), y: PM.t + (B1 - B0) * unit + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: v < 0 ? "−" + -v : v }));
        for (let v = -2; v <= 3; v++) parSvg.append(s("text", { x: PM.l - 6, y: pb(v) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: v < 0 ? "−" + -v : v }));
        parSvg.append(s("text", { x: PM.l + (A1 - A0) * unit, y: PH - 2, "text-anchor": "end", "font-size": 11.5, class: "t-muted", text: "a（斜率）" }));
        parSvg.append(s("text", { x: 4, y: PM.t + 10, "font-size": 11.5, class: "t-muted", text: "b" }));
        if (!done() && B < 16) {
          const lg = s("g", { transform: `translate(${PM.l + 8},${PH - PM.b - 30})` });
          lg.append(s("rect", { x: -6, y: -6, width: 268, height: 24, rx: 6, fill: "var(--panel)", stroke: "var(--line)" }));
          lg.append(s("line", { x1: 0, x2: 20, y1: 6, y2: 6, stroke: "var(--hot)", "stroke-width": 2.2 }));
          lg.append(s("text", { x: 26, y: 10, "font-size": 11.5, text: "這批估出的一步" }));
          lg.append(s("line", { x1: 126, x2: 146, y1: 6, y2: 6, stroke: "var(--muted)", "stroke-width": 2.2, "stroke-dasharray": "5 4" }));
          lg.append(s("text", { x: 152, y: 10, "font-size": 11.5, text: "用全部資料的一步" }));
          parSvg.append(lg);
        }
        if (hist.length === 1) parSvg.append(s("text", { x: PM.l + (A1 - A0) * unit - 6, y: PM.t + 16, "text-anchor": "end", "font-size": 11.5, class: "t-halo", text: "點圖可改變起點" }));
      }

      /* ---------- 比較圖 ---------- */
      const COLORS = { 16: "var(--pos)", 4: "var(--ok)", 1: "var(--neg)" };
      const NAMES = { 16: "全部 16 筆", 4: "4 筆", 1: "1 筆" };
      function drawCompare() {
        const W = 640, H = 220, M = { l: 52, r: 16, t: 26, b: 30 }, EP = 5;
        const ex = (a, b) => Math.max(1e-4, loss(a, b) - L_OPT);
        const LO = Math.log10(1e-4), HI = Math.log10(Math.max(ex(...start), 1e-3) * 1.2);
        const px = (e) => M.l + (e / EP) * (W - M.l - M.r);
        const py = (v) => M.t + ((HI - Math.log10(v)) / (HI - LO)) * (H - M.t - M.b);
        cmpSvg.replaceChildren();
        for (let k = Math.ceil(HI); k >= -4; k--) {
          if (k > HI) continue;
          cmpSvg.append(s("line", { x1: M.l, x2: W - M.r, y1: py(10 ** k), y2: py(10 ** k), stroke: "var(--line)" }));
          cmpSvg.append(s("text", { x: M.l - 6, y: py(10 ** k) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: k >= 0 ? String(10 ** k) : "0." + "0".repeat(-k - 1) + "1" }));
        }
        for (let e = 0; e <= EP; e++) cmpSvg.append(s("text", { x: px(e), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: e }));
        cmpSvg.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 11, class: "t-muted", text: "エポック" }));
        cmpSvg.append(s("text", { x: M.l, y: 14, "font-size": 11, class: "t-muted", text: "損失 − 谷底的損失（對數刻度）" }));
        const stats = {};
        for (const b of [16, 4, 1]) {
          const pts = run(start, b, eta, EP), pr = N / b;
          const d = pts.map((p, i) => (i ? "L" : "M") + px(i / pr).toFixed(1) + "," + py(ex(...p)).toFixed(1)).join("");
          cmpSvg.append(s("path", { d, fill: "none", stroke: COLORS[b], "stroke-width": b === B ? 3 : 1.5, opacity: b === B ? 1 : 0.55 }));
          stats[b] = loss(...pts[pts.length - 1]) - L_OPT;
        }
        // 目前進度
        const n = Math.min(nUpd(), EP * per()), p = hist[n];
        if (p && isFinite(p[0])) cmpSvg.append(s("circle", { cx: px(n / per()), cy: py(ex(...p)), r: 5.5, fill: COLORS[B], stroke: "var(--panel)", "stroke-width": 2 }));
        cmpLegend.innerHTML = [16, 4, 1].map((b) =>
          `<span style="color:${COLORS[b]};font-weight:${b === B ? 700 : 400}">━ ${NAMES[b]}（${N / b} 次更新／エポック）：5 エポック後多出 ${fmt(stats[b], 3)}</span>`).join("　") +
          `<br>粗線是目前選的做法，圓點是你目前跑到的位置。`;
      }

      /* ---------- 文字 ---------- */
      function render() {
        drawData(); drawParams();
        const n = nUpd(), [a, b] = cur();
        const epDone = Math.floor(n / per()), inEp = n % per();
        count.innerHTML = `<span><span class="jp">イテレーション</span>（更新次數）<b>${n}</b></span><span><span class="jp">エポック</span>（看完全部資料的次數）<b>${fmt(n / per(), per() > 1 ? 2 : 0)}</b></span><span>已算過的樣本梯度 <b>${n * B}</b> 筆</span><span>整份資料的損失 <b>${fmt(loss(a, b), 3)}</b>（谷底 ${fmt(L_OPT, 3)}）</span>`;
        if (done()) {
          status.className = "status status--info"; status.textContent = `已跑完 ${MAXEP} 個エポック，按「重設」重來`;
          calc.innerHTML = `<span class="calc-row"><span class="k">目前位置</span> (a, b) = (${fmt(a, 3)}, ${fmt(b, 3)})　<span class="k">損失</span> ${fmt(loss(a, b), 4)}</span>`;
          drawCompare(); return;
        }
        const info = batchAt(n, B), [ga, gb] = grad(a, b, info.idx), [fa, fb] = grad(a, b, ALL);
        const ang = (Math.acos(Math.max(-1, Math.min(1, (ga * fa + gb * fb) / (Math.hypot(ga, gb) * Math.hypot(fa, fb) || 1)))) * 180) / Math.PI;
        const dist = Math.hypot(a - A_OPT, b - B_OPT);
        if (B === 16) {
          status.className = "status status--info";
          status.textContent = n === 0 ? "按「更新一次」開始：每次都用全部 16 筆資料，方向最準，但一個エポック只更新 1 次" : `第 ${n} 次更新，離谷底 ${fmt(dist, 2)}：方向一直很準，只是每看完一遍資料才走一步`;
        } else if (dist < 0.25 && n > 0) {
          status.className = "status status--warn";
          status.textContent = `已在谷底附近（距離 ${fmt(dist, 2)}），但每批估出的方向不同，所以會一直在附近晃動`;
        } else {
          status.className = "status status--info";
          status.textContent = `下一步只用 ${B} 筆估坡度：和用全部資料算的方向相差 ${fmt(ang, 0)}°`;
        }
        const errs = info.idx.map((i) => a * DATA[i].x + b - DATA[i].y);
        calc.innerHTML =
          `<span class="calc-row"><span class="k">第 ${n + 1} 次更新（第 ${epDone + 1} エポック的第 ${inEp + 1} 批）　這批用</span> ${B === 16 ? "全部 16 筆" : info.idx.map((i) => "#" + (i + 1)).join("、")}</span>` +
          (B <= 4 ? `<span class="calc-row"><span class="k">各筆誤差 a·x + b − y</span> ${info.idx.map((i, j) => `#${i + 1}: ${fmt(a, 2)}×${fmtP(DATA[i].x, 2)} + ${fmtP(b, 2)} − ${fmtP(DATA[i].y, 2)} = ${fmt(errs[j], 2)}`).join("；")}</span>` : "") +
          `<span class="calc-row"><span class="k">這批估出的梯度</span> (平均(誤差 × x), 平均(誤差)) = (<span class="hl">${fmt(ga, 3)}</span>, <span class="hl">${fmt(gb, 3)}</span>)` +
          (B < 16 ? `　<span class="k">用全部 16 筆算</span> (${fmt(fa, 3)}, ${fmt(fb, 3)})` : "") + `</span>` +
          `<span class="calc-row"><span class="k">下一個位置</span> (a − η·梯度a, b − η·梯度b) = (${fmt(a, 3)} − ${fmt(eta, 2)}×${fmtP(ga, 3)}, ${fmt(b, 3)} − ${fmt(eta, 2)}×${fmtP(gb, 3)}) = (<b>${fmt(a - eta * ga, 3)}</b>, <b>${fmt(b - eta * gb, 3)}</b>)</span>`;
        drawCompare();
      }

      note.innerHTML = `損失用<span class="jp">平均二乗誤差</span>再除以 2（除以 2 只是讓梯度式子少一個係數 2）。每個エポック開始前把 16 筆資料重新洗牌，再依序切成幾批，所以每筆資料在一個エポック裡剛好用一次；這裡用固定的亂數，每次重設結果相同。「確率的」指的是隨機抽樣。每來一筆資料就馬上更新一次的做法也叫<span class="jp">オンライン学習</span>（線上學習，<span class="en">online learning</span>）。一批多筆的計算可以在 GPU 上同時做，所以實務上用 32〜256 筆左右的ミニバッチ；方向帶點隨機的搖擺，有時反而能幫忙離開<span class="jp">局所最適解</span>或<span class="jp">鞍点</span>。學習率和每次走的方式怎麼改良，見 <a href="#optim">モーメンタムと Adam</a>。`;

      reset();
      ctx.onTheme(() => render());
      return { destroy() { player.pause(); } };
    },
  });
})();
