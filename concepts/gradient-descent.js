/* 勾配降下法と学習率（シラバス §16）
   1D：L(w) = (w − 1)²，看學習率三種情況
   2D：等高線，「細長山谷」看鋸齒、「兩個谷」看局所最適解 */
Atlas.define("gd", {
  lead: `<p>訓練模型，就是找出讓<b>損失</b>（預測和正解差多少）最小的參數。但我們看不到整片地形，只能在腳下量出坡度，然後往下坡走一步。這個做法叫 <span class="jp">勾配降下法</span>（梯度下降法，<span class="en">gradient descent</span>）；每一步跨多大，由 <span class="jp">学習率</span>（學習率，<span class="en">learning rate</span>）決定。</p>`,
  closing: `學習率太小要走非常多步；太大會跨過谷底來回震盪，甚至越彈越遠而發散。地形有好幾個谷時，從哪裡出發，就決定最後落進哪一個谷。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    let mode = "1d";

    /* ===================== 共用 UI ===================== */
    const modeSeg = ui.seg({
      label: "地形", value: mode,
      options: [{ value: "1d", label: "1D 曲線" }, { value: "2d", label: "2D 等高線" }],
      onChange: (v) => { mode = v; switchMode(); },
    });
    const surfSeg = ui.seg({
      label: "2D 地形", value: "valley",
      options: [{ value: "valley", label: "細長山谷" }, { value: "twin", label: "兩個谷" }],
      onChange: (v) => { two.setSurface(v); },
    });
    const view = h("div", { class: "scroll-x" });
    const status = h("div", { class: "status" });
    const note = h("p", { class: "note" });
    const calc = h("div", { class: "calc" });
    const extra = h("div");
    const lrSlider = ui.slider({ id: "gd-lr", label: "学習率 η", min: 0.01, max: 1.2, step: 0.01, value: 0.3, format: (v) => v.toFixed(3).replace(/0$/, ""), onInput: (v) => cur().setLr(v) });
    const presetBox = h("div", { class: "btn-group" });
    const player = ui.player(ctx, () => cur().step(), 520);
    const stepBtn = ui.btn("走一步", () => { player.pause(); cur().step(); }, { primary: true });
    const resetBtn = ui.btn("重設", () => { player.pause(); cur().reset(); });
    const hint = h("p", { class: "hint" });

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el, surfSeg.el),
      view,
      h("div", { class: "bench__row bench__row--between" }, status, hint),
      h("div", { class: "bench__row" }, lrSlider.el, h("div", { class: "field" }, h("span", { class: "field__label" }, "快速選擇"), presetBox)),
      h("div", { class: "bench__row" }, stepBtn, player.el, resetBtn),
      calc, note, extra,
    );

    function setPresets(list) {
      presetBox.replaceChildren(...list.map(([label, v]) =>
        ui.btn(`${label}（${v}）`, () => { player.pause(); lrSlider.set(v); cur().setLr(v); }, { sm: true })));
    }
    function setStatus(kind, text) { status.className = "status status--" + kind; status.textContent = text; }

    /* ===================== 1D ===================== */
    const one = (() => {
      const W = 640, H = 300, M = { l: 46, r: 16, t: 16, b: 40 };
      const X0 = -4, X1 = 6, Y0 = 0, Y1 = 26, OPT = 1;
      const L = (w) => (w - OPT) ** 2, dL = (w) => 2 * (w - OPT);
      const sx = (w) => M.l + ((w - X0) / (X1 - X0)) * (W - M.l - M.r);
      const sy = (y) => H - M.b - ((y - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
      let lr = 0.3, start = -3, hist = [start];

      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "損失曲線與目前位置", style: "display:block;width:100%;min-width:520px;max-width:760px;cursor:crosshair" });
      const mini = s("svg", { viewBox: `0 0 ${W} 130`, role: "img", "aria-label": "損失隨步數的變化", style: "display:block;width:100%;min-width:520px;max-width:760px" });
      const miniWrap = h("div", null, h("p", { class: "field__label" }, "損失隨步數的變化"), h("div", { class: "scroll-x" }, mini));

      svg.addEventListener("click", (e) => {
        const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        const w = X0 + ((p.x - M.l) / (W - M.l - M.r)) * (X1 - X0);
        start = Math.max(X0 + 0.2, Math.min(X1 - 0.2, Math.round(w * 10) / 10));
        player.pause(); api.reset();
      });

      const ratio = () => 1 - 2 * lr;
      const done = () => {
        const w = hist[hist.length - 1];
        return Math.abs(w - OPT) < 0.005 || !isFinite(w) || Math.abs(w) > 1e6 || hist.length > 300;
      };

      function draw() {
        svg.replaceChildren();
        const g = s("g");
        // axes & grid
        for (let y = 0; y <= 25; y += 5) {
          g.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(y), y2: sy(y), stroke: "var(--line)", "stroke-width": 1 }));
          g.append(s("text", { x: M.l - 8, y: sy(y) + 4, "text-anchor": "end", "font-size": 11, class: "t-mono t-muted", text: y }));
        }
        for (let x = X0; x <= X1; x += 1) {
          g.append(s("text", { x: sx(x), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: x < 0 ? "−" + -x : x }));
        }
        g.append(s("line", { x1: M.l, x2: W - M.r, y1: sy(0), y2: sy(0), stroke: "var(--line-strong)" }));
        g.append(s("text", { x: W - M.r, y: H - 4, "text-anchor": "end", "font-size": 12, class: "t-muted", text: "參數 w" }));
        g.append(s("text", { x: sx(OPT), y: M.t + 12, "text-anchor": "middle", "font-size": 12.5, class: "t-muted t-halo", text: "損失 L(w) = (w − 1)²" }));
        // curve
        let d = "";
        for (let i = 0; i <= 200; i++) { const w = X0 + (i / 200) * (X1 - X0); d += (i ? "L" : "M") + sx(w).toFixed(1) + "," + sy(L(w)).toFixed(1); }
        g.append(s("path", { d, fill: "none", stroke: "var(--seq-hi)", "stroke-width": 2.5 }));
        // minimum marker
        g.append(s("line", { x1: sx(OPT), x2: sx(OPT), y1: sy(0), y2: sy(0) - 8, stroke: "var(--ok)", "stroke-width": 2 }));
        g.append(s("text", { x: sx(OPT), y: sy(2.6), "text-anchor": "middle", "font-size": 11.5, fill: "var(--ok)", text: "谷底 w = 1" }));
        svg.append(g);

        // clip region
        const clipId = "gd1clip";
        svg.append(s("defs", null, s("clipPath", { id: clipId }, s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }))));
        const pg = s("g", { "clip-path": `url(#${clipId})` });
        // trail
        const pts = hist.filter((w) => isFinite(w) && Math.abs(w) < 1e4);
        for (let i = 1; i < pts.length; i++) {
          pg.append(s("line", { x1: sx(pts[i - 1]), y1: sy(L(pts[i - 1])), x2: sx(pts[i]), y2: sy(L(pts[i])), stroke: "var(--hot)", "stroke-width": 1.5, "stroke-dasharray": "4 3", opacity: 0.8 }));
        }
        pts.forEach((w, i) => { if (i < pts.length - 1) pg.append(s("circle", { cx: sx(w), cy: sy(L(w)), r: 3.5, fill: "var(--hot)", opacity: 0.55 })); });
        const w = hist[hist.length - 1];
        if (isFinite(w) && w >= X0 && w <= X1) {
          // tangent
          const m = dL(w), span = 1.3;
          pg.append(s("line", { x1: sx(w - span), y1: sy(L(w) - m * span), x2: sx(w + span), y2: sy(L(w) + m * span), stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }));
          // step arrow on axis
          const wn = w - lr * m;
          if (!done() && Math.abs(wn - w) > 0.02) {
            const ax = sy(0) - 18, x2 = sx(Math.max(X0, Math.min(X1, wn)));
            pg.append(s("line", { x1: sx(w), y1: ax, x2, y2: ax, stroke: "var(--hot)", "stroke-width": 2.5 }));
            const dir = wn > w ? 1 : -1;
            pg.append(s("path", { d: `M${x2},${ax} l${-8 * dir},-5 l0,10 z`, fill: "var(--hot)" }));
          }
          pg.append(s("circle", { cx: sx(w), cy: sy(L(w)), r: 9, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2.5 }));
        }
        svg.append(pg);
        if (isFinite(w) && (w < X0 || w > X1)) {
          const right = w > X1;
          svg.append(s("text", { x: right ? W - M.r - 4 : M.l + 4, y: M.t + 30, "text-anchor": right ? "end" : "start", "font-size": 13, "font-weight": 700, class: "t-halo", fill: "var(--bad)", text: right ? "球飛出畫面 →" : "← 球飛出畫面" }));
        }
        if (hist.length === 1) svg.append(s("text", { x: W - M.r - 4, y: sy(4), "text-anchor": "end", "font-size": 12, class: "t-muted", text: "點曲線可改變起點" }));
        drawMini();
      }

      function drawMini() {
        const MW = W, MH = 130, m = { l: 46, r: 16, t: 12, b: 26 };
        mini.replaceChildren();
        const ls = hist.map(L).filter(isFinite);
        const n = Math.max(10, ls.length - 1);
        const ymax = Math.max(...ls, 1);
        const px = (i) => m.l + (i / n) * (MW - m.l - m.r);
        const py = (v) => MH - m.b - (Math.min(v, ymax) / ymax) * (MH - m.t - m.b);
        mini.append(s("line", { x1: m.l, x2: MW - m.r, y1: py(0), y2: py(0), stroke: "var(--line-strong)" }));
        mini.append(s("line", { x1: m.l, x2: m.l, y1: m.t, y2: py(0), stroke: "var(--line-strong)" }));
        mini.append(s("text", { x: m.l - 6, y: m.t + 8, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: ymax >= 1e4 ? ymax.toExponential(0) : fmt(ymax, ymax < 10 ? 1 : 0) }));
        mini.append(s("text", { x: m.l - 6, y: py(0) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: "0" }));
        mini.append(s("text", { x: MW - m.r, y: MH - 6, "text-anchor": "end", "font-size": 11, class: "t-muted", text: `步數（0〜${n}）` }));
        let d = "";
        ls.forEach((v, i) => { d += (i ? "L" : "M") + px(i).toFixed(1) + "," + py(v).toFixed(1); });
        mini.append(s("path", { d, fill: "none", stroke: "var(--hot)", "stroke-width": 2 }));
        ls.forEach((v, i) => mini.append(s("circle", { cx: px(i), cy: py(v), r: 2.6, fill: "var(--hot)" })));
      }

      function updateText() {
        const w = hist[hist.length - 1], n = hist.length - 1, r = ratio();
        const dist = Math.abs(w - OPT);
        if (!isFinite(w) || Math.abs(w) > 1e6) setStatus("bad", `發散：第 ${n} 步時 w 已經大到無法顯示`);
        else if (dist < 0.005) setStatus("ok", `已到谷底：共走了 ${n} 步`);
        else if (r < -1) setStatus("bad", n ? "發散：跨過谷底後落得更遠，越彈越遠" : "這個學習率會發散，按「走一步」看看");
        else if (r === -1) setStatus("warn", "在谷底兩側來回彈跳，永遠停不下來");
        else if (r < 0) setStatus("warn", n ? "震盪：每步都跨過谷底，但越跨越短，最後仍會收斂" : "這個學習率會跨過谷底來回震盪");
        else if (r >= 0.85) setStatus("warn", `前進緩慢：第 ${n} 步，離谷底還有 ${fmt(dist)}`);
        else setStatus("info", n ? `穩定往谷底前進：第 ${n} 步，離谷底 ${fmt(dist)}` : "按「走一步」開始");
        hint.textContent = "";

        if (!isFinite(w) || Math.abs(w) > 1e6 || dist < 0.005) {
          calc.innerHTML = `<span class="calc-row"><span class="k">目前位置</span> w = ${isFinite(w) ? fmt(w, 3) : "—"}　<span class="k">損失</span> L = ${isFinite(w) ? fmt(L(w), 4) : "—"}</span>`;
        } else {
          const m = dL(w), step = -lr * m, wn = w + step;
          calc.innerHTML =
            `<span class="calc-row"><span class="k">第 ${n} 步　目前位置</span> w = <b>${fmt(w, 3)}</b>　<span class="k">損失</span> L = ${fmt(L(w), 3)}</span>` +
            `<span class="calc-row"><span class="k">斜率（梯度）</span> dL/dw = 2 × (w − 1) = 2 × ${fmtP(w - OPT, 3)} = <span class="${m < 0 ? "n" : "p"}">${fmt(m, 3)}</span></span>` +
            `<span class="calc-row"><span class="k">這一步移動</span> −η × 斜率 = −${fmt(lr, 3)} × ${fmtP(m, 3)} = <span class="hl">${fmt(step, 3, true)}</span></span>` +
            `<span class="calc-row"><span class="k">下一個位置</span> w = ${fmt(w, 3)} ${step < 0 ? "−" : "+"} ${fmt(Math.abs(step), 3)} = <b>${fmt(wn, 3)}</b></span>`;
        }
        note.innerHTML = `這條曲線上，每走一步，離谷底的距離都變成原本的 |1 − 2η| = <b class="mono">${fmt(Math.abs(r), 2)}</b> 倍${r < 0 ? "，而且換到谷底另一側" : ""}。${Math.abs(r) < 1 ? "小於 1 才會越來越近。" : "大於或等於 1，所以不會靠近。"}`;
      }

      const api = {
        presets: [["太小", 0.05], ["適中", 0.3], ["偏大", 0.9], ["太大", 1.05]],
        slider: { min: 0.01, max: 1.2, step: 0.01 },
        get lr() { return lr; },
        show() { view.replaceChildren(svg); extra.replaceChildren(miniWrap); draw(); updateText(); },
        setLr(v) { lr = v; api.reset(); },
        reset() { hist = [start]; draw(); updateText(); },
        step() {
          if (done()) return false;
          const w = hist[hist.length - 1];
          hist.push(w - lr * dL(w));
          draw(); updateText();
          return !done();
        },
        redraw() { draw(); },
      };
      return api;
    })();

    /* ===================== 2D ===================== */
    const two = (() => {
      const SURF = {
        valley: {
          f: (x, y) => (x * x + 10 * y * y) / 2,
          g: (x, y) => [x, 10 * y],
          gText: "(x, 10y)",
          box: [-6, 6, -3, 3], start: [-5, 2.4],
          presets: [["太小", 0.02], ["適中", 0.1], ["偏大", 0.185], ["太大", 0.205]],
          slider: { min: 0.005, max: 0.25, step: 0.005 }, lr: 0.185,
          formula: "L(x, y) = (x² + 10y²) ÷ 2",
          marks: [{ x: 0, y: 0, label: "谷底", kind: "global" }],
        },
        twin: {
          f: (x, y) => (x * x - 1) ** 2 + 0.3 * x + y * y,
          g: (x, y) => [4 * x * (x * x - 1) + 0.3, 2 * y],
          gText: "(4x(x² − 1) + 0.3, 2y)",
          box: [-2, 2, -1, 1], start: [1.7, 0.8],
          presets: [["太小", 0.01], ["適中", 0.05], ["偏大", 0.25], ["太大", 0.3]],
          slider: { min: 0.005, max: 0.35, step: 0.005 }, lr: 0.05,
          formula: "L(x, y) = (x² − 1)² + 0.3x + y²",
          marks: null, // 由數值求出
        },
      };
      // 求 twin 的三個臨界點：4x³ − 4x + 0.3 = 0
      (function () {
        const root = (x) => { for (let i = 0; i < 60; i++) x -= (4 * x ** 3 - 4 * x + 0.3) / (12 * x * x - 4); return x; };
        SURF.twin.marks = [
          { x: root(-1.1), y: 0, label: "大域最適解", kind: "global" },
          { x: root(1.1), y: 0, label: "局所最適解", kind: "local" },
          { x: root(0.05), y: 0, label: "鞍点", kind: "saddle" },
        ];
      })();

      let key = "valley", S = SURF[key], lr = S.lr, start = S.start.slice(), hist = [start.slice()];
      const wrap = h("div", { style: { position: "relative", width: "100%", maxWidth: "760px", margin: "0 auto" } });
      const canvas = h("canvas", { "aria-label": "損失的等高線圖與移動路徑", role: "img", style: { display: "block", width: "100%", touchAction: "manipulation", cursor: "crosshair", borderRadius: "8px", border: "1px solid var(--line)" } });
      wrap.append(canvas);
      const legend = h("div", { class: "hint" });
      let bg = null, cssW = 0, cssH = 0, dpr = 1;

      const tx = (x) => ((x - S.box[0]) / (S.box[1] - S.box[0])) * cssW;
      const ty = (y) => cssH - ((y - S.box[2]) / (S.box[3] - S.box[2])) * cssH;

      function sizeCanvas() {
        cssW = Math.max(300, Math.min(760, wrap.clientWidth || 640));
        cssH = Math.round(cssW * ((S.box[3] - S.box[2]) / (S.box[1] - S.box[0])));
        dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
        canvas.style.height = cssH + "px";
        renderBg();
      }
      function renderBg() {
        const Wp = canvas.width, Hp = canvas.height, step = 2;
        const gw = Math.ceil(Wp / step), gh = Math.ceil(Hp / step);
        const vals = new Float64Array(gw * gh);
        let lo = Infinity, hi = -Infinity;
        for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
          const x = S.box[0] + ((i + 0.5) / gw) * (S.box[1] - S.box[0]);
          const y = S.box[3] - ((j + 0.5) / gh) * (S.box[3] - S.box[2]);
          const v = S.f(x, y); vals[j * gw + i] = v; if (v < lo) lo = v; if (v > hi) hi = v;
        }
        const BANDS = 14, band = new Uint8Array(gw * gh), lg = Math.log(1 + hi - lo);
        for (let k = 0; k < vals.length; k++) band[k] = Math.max(0, Math.min(BANDS - 1, Math.floor((Math.log(1 + vals[k] - lo) / lg) * BANDS) || 0));
        const lo_c = Atlas.color("--seq-lo"), hi_c = Atlas.color("--seq-hi");
        const pal = []; for (let b = 0; b < BANDS; b++) pal.push(Atlas.mix(lo_c, hi_c, (b / (BANDS - 1)) * 0.78));
        const lineC = Atlas.mix(lo_c, hi_c, 0.95);
        const off = document.createElement("canvas"); off.width = gw; off.height = gh;
        const octx = off.getContext("2d"), img = octx.createImageData(gw, gh);
        for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
          const k = j * gw + i, b = band[k];
          const edge = (i + 1 < gw && band[k + 1] !== b) || (j + 1 < gh && band[k + gw] !== b);
          const c = edge ? lineC : pal[b];
          img.data.set([c[0], c[1], c[2], edge ? 150 : 255], k * 4);
          if (edge) { const p = pal[b]; for (let q = 0; q < 3; q++) img.data[k * 4 + q] = Math.round(p[q] * 0.45 + c[q] * 0.55); img.data[k * 4 + 3] = 255; }
        }
        octx.putImageData(img, 0, 0);
        bg = off;
        draw();
      }
      function draw() {
        if (!bg) return;
        const c = canvas.getContext("2d");
        c.setTransform(1, 0, 0, 1, 0, 0);
        c.imageSmoothingEnabled = true;
        c.drawImage(bg, 0, 0, canvas.width, canvas.height);
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        const ink = Atlas.color("--ink"), panel = Atlas.color("--panel"), hot = Atlas.color("--hot"), ok = Atlas.color("--ok"), bad = Atlas.color("--bad"), muted = Atlas.color("--muted");
        const font = getComputedStyle(document.body).fontFamily;
        // critical points
        for (const m of S.marks) {
          const X = tx(m.x), Y = ty(m.y);
          c.beginPath();
          if (m.kind === "saddle") { c.rect(X - 5, Y - 5, 10, 10); c.fillStyle = muted; }
          else { c.arc(X, Y, 6, 0, Math.PI * 2); c.fillStyle = m.kind === "global" ? ok : bad; }
          c.fill(); c.lineWidth = 2; c.strokeStyle = panel; c.stroke();
          c.font = `700 12px ${font}`; c.textAlign = "center";
          c.lineWidth = 4; c.strokeStyle = panel; c.strokeText(m.label, X, Y - 12);
          c.fillStyle = m.kind === "saddle" ? muted : m.kind === "global" ? ok : bad; c.fillText(m.label, X, Y - 12);
        }
        // path
        const pts = hist.filter((p) => isFinite(p[0]) && Math.abs(p[0]) < 1e4 && Math.abs(p[1]) < 1e4);
        c.lineWidth = 2; c.strokeStyle = hot; c.beginPath();
        pts.forEach((p, i) => (i ? c.lineTo(tx(p[0]), ty(p[1])) : c.moveTo(tx(p[0]), ty(p[1]))));
        c.stroke();
        pts.forEach((p, i) => {
          const last = i === pts.length - 1;
          c.beginPath(); c.arc(tx(p[0]), ty(p[1]), last ? 7 : 3, 0, Math.PI * 2);
          c.fillStyle = hot; c.fill();
          if (last) { c.lineWidth = 2.5; c.strokeStyle = panel; c.stroke(); }
        });
        // start ring
        c.beginPath(); c.arc(tx(start[0]), ty(start[1]), 9, 0, Math.PI * 2); c.lineWidth = 1.5; c.strokeStyle = ink; c.setLineDash([3, 3]); c.stroke(); c.setLineDash([]);
        c.font = `12px ${font}`; c.textAlign = "left"; c.fillStyle = ink; c.lineWidth = 4; c.strokeStyle = panel;
        c.strokeText(S.formula, 8, 18); c.fillText(S.formula, 8, 18);
        if (hist.length === 1) { c.textAlign = "right"; c.strokeText("點圖可改變起點", cssW - 8, cssH - 10); c.fillText("點圖可改變起點", cssW - 8, cssH - 10); }
      }
      canvas.addEventListener("click", (e) => {
        const r = canvas.getBoundingClientRect();
        const x = S.box[0] + ((e.clientX - r.left) / r.width) * (S.box[1] - S.box[0]);
        const y = S.box[3] - ((e.clientY - r.top) / r.height) * (S.box[3] - S.box[2]);
        start = [Math.round(x * 100) / 100, Math.round(y * 100) / 100];
        player.pause(); api.reset();
      });

      const out = (p) => !isFinite(p[0]) || !isFinite(p[1]) || Math.abs(p[0]) > (S.box[1] - S.box[0]) * 5 || Math.abs(p[1]) > (S.box[3] - S.box[2]) * 5;
      const gnorm = (p) => Math.hypot(...S.g(p[0], p[1]));
      const done = () => { const p = hist[hist.length - 1]; return out(p) || gnorm(p) < 1e-3 || hist.length > 400; };

      function updateText() {
        const p = hist[hist.length - 1], n = hist.length - 1;
        if (out(p)) setStatus("bad", `發散：第 ${n} 步已經衝出地圖`);
        else if (gnorm(p) < 1e-3) {
          if (key === "twin" && p[0] > 0) setStatus("warn", `停在局所最適解（第 ${n} 步）：四周都是上坡，梯度≈0，走不出去`);
          else setStatus("ok", `抵達${key === "twin" ? "大域最適解" : "谷底"}：共 ${n} 步`);
        } else if (n === 0) setStatus("info", "按「走一步」開始");
        else if (key === "valley" && n >= 2 && Math.sign(hist[n][1]) !== Math.sign(hist[n - 1][1]) && Math.abs(hist[n][1]) > 1e-3) {
          setStatus("warn", `鋸齒前進：陡的 y 方向來回彈，緩的 x 方向慢慢走（第 ${n} 步）`);
        } else setStatus("info", `第 ${n} 步，損失 ${fmt(S.f(p[0], p[1]), 3)}`);

        if (out(p) || gnorm(p) < 1e-3) {
          calc.innerHTML = out(p) ? `<span class="k">位置已超出地圖</span>` :
            `<span class="calc-row"><span class="k">停在</span> (x, y) = (${fmt(p[0], 3)}, ${fmt(p[1], 3)})　<span class="k">損失</span> ${fmt(S.f(p[0], p[1]), 4)}</span>`;
        } else {
          const [gx, gy] = S.g(p[0], p[1]);
          const nx = p[0] - lr * gx, ny = p[1] - lr * gy;
          calc.innerHTML =
            `<span class="calc-row"><span class="k">第 ${n} 步　位置</span> (x, y) = (<b>${fmt(p[0], 3)}</b>, <b>${fmt(p[1], 3)}</b>)　<span class="k">損失</span> ${fmt(S.f(p[0], p[1]), 3)}</span>` +
            `<span class="calc-row"><span class="k">梯度（各方向的坡度）</span> ${S.gText} = (${fmt(gx, 3)}, ${fmt(gy, 3)})</span>` +
            `<span class="calc-row"><span class="k">下一個位置</span> (x − η·梯度x, y − η·梯度y) = (${fmt(p[0], 3)} − ${fmt(lr, 3)}×${fmtP(gx, 3)}, ${fmt(p[1], 3)} − ${fmt(lr, 3)}×${fmtP(gy, 3)}) = (<span class="hl">${fmt(nx, 3)}</span>, <span class="hl">${fmt(ny, 3)}</span>)</span>`;
        }
        if (key === "valley") {
          note.innerHTML = `顏色越深＝損失越大，線條是等高線。這個山谷 y 方向的坡比 x 方向陡 10 倍：同一個學習率，對 y 方向可能太大（來回彈），對 x 方向卻太小（走很慢）。改良的走法見 <a href="#optim">モーメンタムと Adam</a>；每步只用部分資料估坡度，見 <a href="#sgd">確率的勾配降下法とミニバッチ</a>。`;
          legend.textContent = "";
        } else {
          note.innerHTML = `這片地形有兩個谷：左邊較深，是<span class="jp">大域最適解</span>（全域最佳解，<span class="en">global optimum</span>）；右邊較淺，是<span class="jp">局所最適解</span>（局部最佳解，<span class="en">local optimum</span>）。兩谷之間的方塊是<span class="jp">鞍点</span>（鞍點，<span class="en">saddle point</span>）：沿 x 是山頂、沿 y 是谷底，梯度同樣是 0。試著從左、右兩側分別出發。`;
        }
      }

      const api = {
        get presets() { return S.presets; },
        get slider() { return S.slider; },
        get lr() { return lr; },
        show() { view.replaceChildren(wrap); extra.replaceChildren(); sizeCanvas(); updateText(); },
        setSurface(k) { key = k; S = SURF[k]; lr = S.lr; start = S.start.slice(); player.pause(); syncControls(); api.show(); api.reset(); },
        setLr(v) { lr = v; api.reset(); },
        reset() { hist = [start.slice()]; draw(); updateText(); },
        step() {
          if (done()) return false;
          const p = hist[hist.length - 1], [gx, gy] = S.g(p[0], p[1]);
          hist.push([p[0] - lr * gx, p[1] - lr * gy]);
          draw(); updateText();
          return !done();
        },
        resize() { if (wrap.isConnected && Math.max(300, Math.min(760, wrap.clientWidth || 640)) !== cssW) sizeCanvas(); },
        redraw() { if (wrap.isConnected) renderBg(); },
      };
      return api;
    })();

    /* ===================== 切換與生命週期 ===================== */
    const cur = () => (mode === "1d" ? one : two);
    function syncControls() {
      const c = cur(), sl = c.slider, inp = bench.querySelector("#gd-lr");
      inp.min = sl.min; inp.max = sl.max; inp.step = sl.step;
      lrSlider.set(c.lr);
      setPresets(c.presets);
    }
    function switchMode() {
      player.pause();
      surfSeg.el.hidden = mode !== "2d";
      syncControls();
      cur().show();
    }
    switchMode();

    ctx.onTheme(() => { one.redraw(); two.redraw(); });
    let rT = null;
    const ro = new ResizeObserver(() => { clearTimeout(rT); rT = setTimeout(() => mode === "2d" && two.resize(), 120); });
    ro.observe(view);
    return { destroy() { ro.disconnect(); player.pause(); } };
  },
});
