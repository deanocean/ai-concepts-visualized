/* 梯度下降法與學習率（大綱 §16）
   1D：L(w) = (w − 1)²，看學習率三種情況
   2D：等高線，「細長山谷」看鋸齒、「兩個谷」看局部最佳解
   文字在 locales/zh/gd.js、locales/ja/gd.js。 */
Atlas.define("gd", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    let mode = "1d";

    /* ===================== 共用 UI ===================== */
    const modeSeg = ui.seg({
      label: t("modeLabel"), value: mode,
      options: [{ value: "1d", label: t("mode1d") }, { value: "2d", label: t("mode2d") }],
      onChange: (v) => { mode = v; switchMode(); },
    });
    const surfSeg = ui.seg({
      label: t("surfLabel"), value: "valley",
      options: [{ value: "valley", label: t("surfValley") }, { value: "twin", label: t("surfTwin") }],
      onChange: (v) => { two.setSurface(v); },
    });
    const view = h("div", { class: "scroll-x" });
    const status = h("div", { class: "status" });
    const note = h("p", { class: "note" });
    const calc = h("div", { class: "calc" });
    const extra = h("div");
    const lrSlider = ui.slider({ id: "gd-lr", label: t("lrLabel"), min: 0.01, max: 1.2, step: 0.01, value: 0.3, format: (v) => v.toFixed(3).replace(/0$/, ""), onInput: (v) => cur().setLr(v) });
    const presetBox = h("div", { class: "btn-group" });
    const player = ui.player(ctx, () => cur().step(), 520);
    const stepBtn = ui.btn(t("btnStep"), () => { player.pause(); cur().step(); }, { primary: true });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); cur().reset(); });
    const hint = h("p", { class: "hint" });

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el, surfSeg.el),
      view,
      h("div", { class: "bench__row bench__row--between" }, status, hint),
      h("div", { class: "bench__row" }, lrSlider.el, h("div", { class: "field" }, h("span", { class: "field__label" }, t("presetLabel")), presetBox)),
      h("div", { class: "bench__row" }, stepBtn, player.el, resetBtn),
      calc, note, extra,
    );

    function setPresets(list) {
      presetBox.replaceChildren(...list.map(([label, v]) =>
        ui.btn(t("presetBtn", { label: t(label), v }), () => { player.pause(); lrSlider.set(v); cur().setLr(v); }, { sm: true })));
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

      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgAria"), style: "display:block;width:100%;min-width:520px;max-width:760px;cursor:crosshair" });
      const mini = s("svg", { viewBox: `0 0 ${W} 130`, role: "img", "aria-label": t("miniLabel"), style: "display:block;width:100%;min-width:520px;max-width:760px" });
      const miniWrap = h("div", null, h("p", { class: "field__label" }, t("miniLabel")), h("div", { class: "scroll-x" }, mini));

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
        g.append(s("text", { x: W - M.r, y: H - 4, "text-anchor": "end", "font-size": 12, class: "t-muted", text: t("axisW") }));
        g.append(s("text", { x: sx(OPT), y: M.t + 12, "text-anchor": "middle", "font-size": 12.5, class: "t-muted t-halo", text: t("curveLabel") }));
        // curve
        let d = "";
        for (let i = 0; i <= 200; i++) { const w = X0 + (i / 200) * (X1 - X0); d += (i ? "L" : "M") + sx(w).toFixed(1) + "," + sy(L(w)).toFixed(1); }
        g.append(s("path", { d, fill: "none", stroke: "var(--seq-hi)", "stroke-width": 2.5 }));
        // minimum marker
        g.append(s("line", { x1: sx(OPT), x2: sx(OPT), y1: sy(0), y2: sy(0) - 8, stroke: "var(--ok)", "stroke-width": 2 }));
        g.append(s("text", { x: sx(OPT), y: sy(2.6), "text-anchor": "middle", "font-size": 11.5, fill: "var(--ok)", text: t("minLabel") }));
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
          svg.append(s("text", { x: right ? W - M.r - 4 : M.l + 4, y: M.t + 30, "text-anchor": right ? "end" : "start", "font-size": 13, "font-weight": 700, class: "t-halo", fill: "var(--bad)", text: right ? t("flyRight") : t("flyLeft") }));
        }
        if (hist.length === 1) svg.append(s("text", { x: W - M.r - 4, y: sy(4), "text-anchor": "end", "font-size": 12, class: "t-muted", text: t("clickCurve") }));
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
        mini.append(s("text", { x: MW - m.r, y: MH - 6, "text-anchor": "end", "font-size": 11, class: "t-muted", text: t("miniAxis", { n }) }));
        let d = "";
        ls.forEach((v, i) => { d += (i ? "L" : "M") + px(i).toFixed(1) + "," + py(v).toFixed(1); });
        mini.append(s("path", { d, fill: "none", stroke: "var(--hot)", "stroke-width": 2 }));
        ls.forEach((v, i) => mini.append(s("circle", { cx: px(i), cy: py(v), r: 2.6, fill: "var(--hot)" })));
      }

      function updateText() {
        const w = hist[hist.length - 1], n = hist.length - 1, r = ratio();
        const dist = Math.abs(w - OPT);
        if (!isFinite(w) || Math.abs(w) > 1e6) setStatus("bad", t("s1Overflow", { n }));
        else if (dist < 0.005) setStatus("ok", t("s1Done", { n }));
        else if (r < -1) setStatus("bad", n ? t("s1Diverge") : t("s1DivergeIdle"));
        else if (r === -1) setStatus("warn", t("s1Bounce"));
        else if (r < 0) setStatus("warn", n ? t("s1Osc") : t("s1OscIdle"));
        else if (r >= 0.85) setStatus("warn", t("s1Slow", { n, dist: fmt(dist) }));
        else setStatus("info", n ? t("s1Steady", { n, dist: fmt(dist) }) : t("statusIdle"));
        hint.textContent = "";

        if (!isFinite(w) || Math.abs(w) > 1e6 || dist < 0.005) {
          calc.innerHTML = `<span class="calc-row"><span class="k">${t("calcPosK")}</span> w = ${isFinite(w) ? fmt(w, 3) : "—"}　<span class="k">${t("calcLossK")}</span> L = ${isFinite(w) ? fmt(L(w), 4) : "—"}</span>`;
        } else {
          const m = dL(w), step = -lr * m, wn = w + step;
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcStepPosK", { n })}</span> w = <b>${fmt(w, 3)}</b>　<span class="k">${t("calcLossK")}</span> L = ${fmt(L(w), 3)}</span>` +
            `<span class="calc-row"><span class="k">${t("calcSlopeK")}</span> dL/dw = 2 × (w − 1) = 2 × ${fmtP(w - OPT, 3)} = <span class="${m < 0 ? "n" : "p"}">${fmt(m, 3)}</span></span>` +
            `<span class="calc-row"><span class="k">${t("calcMoveK")}</span> ${t("calcMoveSym")} = −${fmt(lr, 3)} × ${fmtP(m, 3)} = <span class="hl">${fmt(step, 3, true)}</span></span>` +
            `<span class="calc-row"><span class="k">${t("calcNextK")}</span> w = ${fmt(w, 3)} ${step < 0 ? "−" : "+"} ${fmt(Math.abs(step), 3)} = <b>${fmt(wn, 3)}</b></span>`;
        }
        note.innerHTML = t("note1d", { r: fmt(Math.abs(r), 2), flip: r < 0, near: Math.abs(r) < 1 });
      }

      const api = {
        presets: [["presetSmall", 0.05], ["presetGood", 0.3], ["presetBig", 0.9], ["presetHuge", 1.05]],
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
          presets: [["presetSmall", 0.02], ["presetGood", 0.1], ["presetBig", 0.185], ["presetHuge", 0.205]],
          slider: { min: 0.005, max: 0.25, step: 0.005 }, lr: 0.185,
          formula: "L(x, y) = (x² + 10y²) ÷ 2",
          marks: [{ x: 0, y: 0, label: t("markValley"), kind: "global" }],
        },
        twin: {
          f: (x, y) => (x * x - 1) ** 2 + 0.3 * x + y * y,
          g: (x, y) => [4 * x * (x * x - 1) + 0.3, 2 * y],
          gText: "(4x(x² − 1) + 0.3, 2y)",
          box: [-2, 2, -1, 1], start: [1.7, 0.8],
          presets: [["presetSmall", 0.01], ["presetGood", 0.05], ["presetBig", 0.25], ["presetHuge", 0.3]],
          slider: { min: 0.005, max: 0.35, step: 0.005 }, lr: 0.05,
          formula: "L(x, y) = (x² − 1)² + 0.3x + y²",
          marks: null, // 由數值求出
        },
      };
      // 求 twin 的三個臨界點：4x³ − 4x + 0.3 = 0
      (function () {
        const root = (x) => { for (let i = 0; i < 60; i++) x -= (4 * x ** 3 - 4 * x + 0.3) / (12 * x * x - 4); return x; };
        SURF.twin.marks = [
          { x: root(-1.1), y: 0, label: t("markGlobal"), kind: "global" },
          { x: root(1.1), y: 0, label: t("markLocal"), kind: "local" },
          { x: root(0.05), y: 0, label: t("markSaddle"), kind: "saddle" },
        ];
      })();

      let key = "valley", S = SURF[key], lr = S.lr, start = S.start.slice(), hist = [start.slice()];
      const wrap = h("div", { style: { position: "relative", width: "100%", maxWidth: "760px", margin: "0 auto" } });
      const canvas = h("canvas", { "aria-label": t("canvasAria"), role: "img", style: { display: "block", width: "100%", touchAction: "manipulation", cursor: "crosshair", borderRadius: "8px", border: "1px solid var(--line)" } });
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
        if (hist.length === 1) { c.textAlign = "right"; c.strokeText(t("clickMap"), cssW - 8, cssH - 10); c.fillText(t("clickMap"), cssW - 8, cssH - 10); }
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
        if (out(p)) setStatus("bad", t("s2Out", { n }));
        else if (gnorm(p) < 1e-3) {
          if (key === "twin" && p[0] > 0) setStatus("warn", t("s2Local", { n }));
          else setStatus("ok", t(key === "twin" ? "s2ReachGlobal" : "s2ReachValley", { n }));
        } else if (n === 0) setStatus("info", t("statusIdle"));
        else if (key === "valley" && n >= 2 && Math.sign(hist[n][1]) !== Math.sign(hist[n - 1][1]) && Math.abs(hist[n][1]) > 1e-3) {
          setStatus("warn", t("s2Zigzag", { n }));
        } else setStatus("info", t("s2Step", { n, loss: fmt(S.f(p[0], p[1]), 3) }));

        if (out(p) || gnorm(p) < 1e-3) {
          calc.innerHTML = out(p) ? `<span class="k">${t("calcOut")}</span>` :
            `<span class="calc-row"><span class="k">${t("calcStopK")}</span> (x, y) = (${fmt(p[0], 3)}, ${fmt(p[1], 3)})　<span class="k">${t("calcLossK")}</span> ${fmt(S.f(p[0], p[1]), 4)}</span>`;
        } else {
          const [gx, gy] = S.g(p[0], p[1]);
          const nx = p[0] - lr * gx, ny = p[1] - lr * gy;
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcStepPos2K", { n })}</span> (x, y) = (<b>${fmt(p[0], 3)}</b>, <b>${fmt(p[1], 3)}</b>)　<span class="k">${t("calcLossK")}</span> ${fmt(S.f(p[0], p[1]), 3)}</span>` +
            `<span class="calc-row"><span class="k">${t("calcGradK")}</span> ${S.gText} = (${fmt(gx, 3)}, ${fmt(gy, 3)})</span>` +
            `<span class="calc-row"><span class="k">${t("calcNextK")}</span> ${t("calcNext2Sym")} = (${fmt(p[0], 3)} − ${fmt(lr, 3)}×${fmtP(gx, 3)}, ${fmt(p[1], 3)} − ${fmt(lr, 3)}×${fmtP(gy, 3)}) = (<span class="hl">${fmt(nx, 3)}</span>, <span class="hl">${fmt(ny, 3)}</span>)</span>`;
        }
        if (key === "valley") {
          note.innerHTML = t("noteValley");
          legend.textContent = "";
        } else {
          note.innerHTML = t("noteTwin");
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
