/* オートエンコーダ（シラバス §24）
   資料：7×7 置中的長方形，寬 1〜7 × 高 1〜7，共 49 張（assets/nn.js 的 rects）。
   網路：49 →(ReLU)32 → 中間層 k（線形）→(ReLU)32 → 49（Sigmoid）；損失＝交差エントロピー（49 格加總）。
   頁面上即時訓練：Adam η=0.01、迷你批次 7 張、600 エポック，每 10 エポック存一份權重，可拖回去看。
   中間層 k = 1／2／4 各自訓練（固定亂數種子，「換一組初始值」換種子 1〜5）。
   文字在 locales/zh/ae.js、locales/ja/ae.js。 */
(function () {
  const { D, F } = Atlas.nn.rects();
  const EPOCHS = 600, SNAP = 10, LR = 0.01, BS = 7;

  Atlas.define("ae", {

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { MLP, rng } = nn;
      const { t } = ctx;
      let k = 2, seed = 1, sel = 30, follow = true, viewE = 0, pick = null; // pick：在中間層平面點到的位置
      const runs = {};

      bench.append(h("style", { text: `
        .ae-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; align-items: flex-start; }
        .ae-fig { flex: 1 1 300px; max-width: 400px; min-width: 0; }
        .ae-fig > svg { display: block; width: 100%; height: auto; }
        .ae-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .ae-pair { display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: nowrap; }
        .ae-pair canvas { width: 104px !important; flex: none; border-radius: 4px; }
        .ae-mid { display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 12px; color: var(--muted); min-width: 92px; text-align: center; }
        .ae-z { font-family: var(--f-mono); font-size: 13px; color: var(--ink); font-weight: 600; background: var(--hot-soft); border: 1.5px solid var(--hot); border-radius: 6px; padding: 2px 6px; line-height: 1.5; }
        .ae-dec { display: flex; align-items: center; gap: 10px; justify-content: center; font-size: 13px; color: var(--muted); margin-top: 6px; min-height: 64px; }
        .ae-dec canvas { width: 60px !important; flex: none; }
        .ae-hit { cursor: pointer; touch-action: manipulation; }
      ` }));

      /* ---------- 控制 ---------- */
      const kSeg = ui.seg({ label: t("kLabel"), value: k, options: [1, 2, 4].map((n) => ({ value: n, label: t("kOpt", { n }) })), onChange: (v) => { k = v; pick = null; ensure(); } });
      const runBtn = ui.btn(t("btnRun"), () => { const R = cur(); R.loop.running ? R.loop.pause() : (follow = true, R.loop.start()); refreshBtns(); }, { primary: true });
      const ffBtn = ui.btn(t("btnFf"), () => { follow = true; cur().loop.fastForward(); refreshBtns(); });
      const seedBtn = ui.btn(t("btnSeed"), () => { cur().loop.pause(); seed = (seed % 5) + 1; pick = null; ensure(); }, { sm: true });
      const prog = ui.slider({ id: "ae-ep", label: t("progLabel"), min: 0, max: EPOCHS, step: SNAP, value: 0, format: (v) => v + " / " + EPOCHS,
        onInput: (v) => { const R = cur(); const last = R.snaps[R.snaps.length - 1].e; if (v > last) { v = last; prog.set(v); } follow = v === last; viewE = v; show(); } });

      /* ---------- 圖 ---------- */
      const net = s("svg", { viewBox: "0 0 470 128", role: "img", "aria-label": t("netAria") });
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const cIn = nn.canvas(104, 104, t("inAria")), cOut = nn.canvas(104, 104, t("outAria"));
      const midBox = h("div", { class: "ae-mid" });
      const gal = nn.canvas(340, 352, t("galAria"));
      gal.el.classList.add("ae-hit");
      const lat = nn.canvas(330, 300, t("latAria"));
      lat.el.classList.add("ae-hit");
      const decC = nn.canvas(60, 60, t("decAria")), decTxt = h("span");
      const decBox = h("div", { class: "ae-dec" }, decC.el, decTxt);
      const lossSvg = s("svg", { role: "img", "aria-label": t("lossAria") });
      const latCap = h("div", { class: "ae-cap" });

      bench.append(
        h("div", { class: "bench__row" }, kSeg.el, seedBtn),
        h("div", { class: "scroll-x" }, h("div", { style: { minWidth: "440px", maxWidth: "560px" } }, net)),
        h("div", { class: "bench__row" }, runBtn, ffBtn, prog.el),
        status,
        h("div", { class: "ae-row" },
          h("div", { class: "ae-fig", style: { flexBasis: "320px" } },
            h("div", { class: "ae-cap" }, t("capPair")),
            h("div", { class: "ae-pair" }, cIn.el, midBox, cOut.el)),
          h("div", { class: "ae-fig", style: { maxWidth: "360px" } }, h("div", { class: "ae-cap" }, t("capGal")), gal.el)),
        calc,
        h("div", { class: "ae-row" },
          h("div", { class: "ae-fig" }, latCap, lat.el, decBox),
          h("div", { class: "ae-fig" }, h("div", { class: "ae-cap" }, t("capLoss")), lossSvg)),
        note);

      /* ---------- 訓練 ---------- */
      function snapOf(R) { return { e: R.epoch, enc: R.enc.snapshot(), dec: R.dec.snapshot() }; }
      function makeRun(kk, sd) {
        const r = rng(sd * 1009 + kk * 7);
        const R = { k: kk, enc: new MLP([49, 32, kk], ["relu", "lin"], r), dec: new MLP([kk, 32, 49], ["relu", "sigmoid"], r), r, order: D.map((_, i) => i), epoch: 0, hist: [], snaps: [] };
        R.vEnc = new MLP([49, 32, kk], ["relu", "lin"], rng(0)); R.vDec = new MLP([kk, 32, 49], ["relu", "sigmoid"], rng(0));
        R.snaps.push(snapOf(R));
        const dOut = new Float64Array(49);
        R.loop = nn.loop(ctx, {
          total: EPOCHS, perTick: 4,
          run(from, n) {
            for (let q = 0; q < n; q++) {
              const o = R.order;
              for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const tmp = o[i]; o[i] = o[j]; o[j] = tmp; }
              let tot = 0;
              for (let b = 0; b < o.length; b += BS) {
                for (let p = b; p < b + BS; p++) {
                  const x = D[o[p]];
                  const z = Float64Array.from(R.enc.forward(x)), y = R.dec.forward(z);
                  tot += nn.bce(x, y);
                  for (let i = 0; i < 49; i++) dOut[i] = y[i] - x[i]; // Sigmoid＋交叉熵：對輸出層 z 的梯度
                  R.enc.backward(R.dec.backward(dOut, true));
                }
                R.enc.adam(LR, 1 / BS); R.dec.adam(LR, 1 / BS);
              }
              R.epoch++; R.hist.push([R.epoch, tot / D.length]);
              if (R.epoch % SNAP === 0) R.snaps.push(snapOf(R));
            }
          },
          onTick() { if (R === cur()) { if (follow) show(); else { drawLoss(); drawStatus(); } refreshBtns(); } },
          onEnd() { if (R === cur()) refreshBtns(); },
        });
        return R;
      }
      const cur = () => runs[k + "-" + seed];
      function ensure() {
        Object.values(runs).forEach((R) => R !== runs[k + "-" + seed] && R.loop.pause());
        if (!runs[k + "-" + seed]) runs[k + "-" + seed] = makeRun(k, seed);
        const R = cur(); follow = true;
        show(); refreshBtns();
      }
      function refreshBtns() {
        const R = cur();
        runBtn.textContent = t(R.loop.running ? "btnPause" : R.loop.finished ? "btnDone" : R.epoch ? "btnResume" : "btnRun");
        runBtn.disabled = R.loop.finished; ffBtn.disabled = R.loop.finished;
      }

      /* ---------- 顯示 ---------- */
      let V = null; // 目前查看的那一份：codes、outs、wrong
      function computeView() {
        const R = cur();
        const snap = follow ? R.snaps[R.snaps.length - 1] : R.snaps.find((q) => q.e === viewE) || R.snaps[R.snaps.length - 1];
        viewE = snap.e; R.vEnc.restore(snap.enc); R.vDec.restore(snap.dec);
        const codes = [], outs = [], wrong = [], loss = [];
        D.forEach((x) => {
          const z = Float64Array.from(R.vEnc.forward(x)), y = Float64Array.from(R.vDec.forward(z));
          codes.push(z); outs.push(y); loss.push(nn.bce(x, y));
          let w = 0; for (let i = 0; i < 49; i++) if ((y[i] > 0.5) !== (x[i] > 0.5)) w++; wrong.push(w);
        });
        V = { R, codes, outs, wrong, loss };
      }
      function show() {
        computeView();
        prog.set(viewE);
        drawNet(); drawPair(); drawGal(); drawLat(); drawLoss(); drawStatus(); drawCalc();
      }
      const C = () => ({ off: Atlas.rgb(ctx.color("--panel-2")), on: Atlas.rgb(ctx.color("--ink")), bad: ctx.color("--bad"), hot: ctx.color("--hot"), line: ctx.color("--line-strong"), muted: ctx.color("--muted"), acc: ctx.color("--accent"), ink: ctx.color("--ink"), panel: ctx.color("--panel") });
      const font = (px, w = 400) => `${w} ${px}px ${ctx.color("--f-body") || "sans-serif"}`;
      function badCells(g, x0, y0, size, x, y) {
        const c = size / 7; g.strokeStyle = C().bad; g.lineWidth = 2;
        for (let i = 0; i < 49; i++) if ((y[i] > 0.5) !== (x[i] > 0.5)) g.strokeRect(x0 + (i % 7) * c + 1.5, y0 + Math.floor(i / 7) * c + 1.5, c - 3, c - 3);
      }

      function drawNet() {
        net.replaceChildren();
        const cols = [[30, 49, t("netIn")], [120, 32, "32"], [235, k, t("netMid", { k })], [350, 32, "32"], [440, 49, t("netOut")]];
        const hgt = (n) => Math.max(10, n * 1.6), cy = 62;
        // 梯形：編碼器、解碼器
        net.append(s("path", { d: `M42,${cy - hgt(49) / 2} L223,${cy - hgt(k) / 2} L223,${cy + hgt(k) / 2} L42,${cy + hgt(49) / 2} Z`, fill: "var(--accent-soft)", stroke: "none" }));
        net.append(s("path", { d: `M247,${cy - hgt(k) / 2} L428,${cy - hgt(49) / 2} L428,${cy + hgt(49) / 2} L247,${cy + hgt(k) / 2} Z`, fill: "var(--accent-soft)", stroke: "none" }));
        cols.forEach(([x, n, lab], i) => {
          net.append(s("rect", { x: x - 10, y: cy - hgt(n) / 2, width: 20, height: hgt(n), rx: 4, fill: i === 2 ? "var(--hot)" : "var(--panel)", stroke: i === 2 ? "var(--hot)" : "var(--line-strong)", "stroke-width": 1.5 }));
          net.append(s("text", { x, y: 122, "text-anchor": "middle", "font-size": 11.5, "font-weight": i === 2 ? 700 : 400, fill: i === 2 ? "var(--hot)" : null, class: i === 2 ? null : "t-muted", text: lab }));
        });
        net.append(s("text", { x: 132, y: 12, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--accent)", text: t("netEnc") }));
        net.append(s("text", { x: 338, y: 12, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--accent)", text: t("netDec") }));
      }

      function drawPair() {
        const c = C(), x = D[sel], y = V.outs[sel], z = V.codes[sel];
        let g = cIn.begin(); nn.drawImg(g, 0, 0, 104, x, c.off, c.on, 2);
        g = cOut.begin(); nn.drawImg(g, 0, 0, 104, y, c.off, c.on, 2); badCells(g, 0, 0, 104, x, y);
        midBox.replaceChildren(h("span", null, t("midEnc")), h("span", { class: "ae-z", html: Array.from(z).map((v) => fmt(v, 2)).join("<br>") }), h("span", null, t("midDec")));
      }

      // 圖庫：橫向寬 1〜7、縱向高 1〜7
      const GX = 26, GY = 22, GS = 40, GG = 4.5;
      function drawGal() {
        const c = C(), g = gal.begin();
        g.font = font(11); g.fillStyle = c.muted; g.textAlign = "center";
        for (let w = 1; w <= 7; w++) g.fillText(t("galW", { w }), GX + (w - 1) * (GS + GG) + GS / 2, 14);
        g.textAlign = "right";
        for (let hh = 1; hh <= 7; hh++) g.fillText(t("galH", { h: hh }), GX - 2, GY + (hh - 1) * (GS + GG) + GS / 2 + 4);
        D.forEach((x, i) => {
          const X = GX + (i % 7) * (GS + GG), Y = GY + Math.floor(i / 7) * (GS + GG);
          nn.drawImg(g, X, Y, GS, V.outs[i], c.off, c.on, 0.6);
          if (V.wrong[i]) { g.strokeStyle = c.bad; g.lineWidth = 2; g.strokeRect(X - 1, Y - 1, GS + 2, GS + 2); }
          if (i === sel) { g.strokeStyle = c.hot; g.lineWidth = 3; g.setLineDash([4, 3]); g.strokeRect(X - 3, Y - 3, GS + 6, GS + 6); g.setLineDash([]); }
        });
      }
      gal.el.addEventListener("click", (e) => {
        const [x, y] = gal.toLocal(e);
        const col = Math.floor((x - GX + GG / 2) / (GS + GG)), row = Math.floor((y - GY + GG / 2) / (GS + GG));
        if (col >= 0 && col < 7 && row >= 0 && row < 7) { sel = row * 7 + col; pick = null; drawPair(); drawGal(); drawLat(); drawCalc(); }
      });

      /* 中間層：2 個＝平面、1 個＝數線、4 個＝畫不出來 */
      let LT = null; // 座標轉換
      function drawLat() {
        const c = C(), g = lat.begin(), W = lat.w, H = lat.h;
        if (k === 4) {
          latCap.textContent = t("lat4Cap");
          g.font = font(13); g.fillStyle = c.muted; g.textAlign = "center";
          g.fillText(t("lat4Line1"), W / 2, H / 2 - 10); g.fillText(t("lat4Line2"), W / 2, H / 2 + 14);
          decBox.hidden = true; LT = null; return;
        }
        decBox.hidden = false;
        const zs = V.codes, T = 18, m = 28;
        if (k === 2) {
          latCap.textContent = t("lat2Cap");
          let x0 = Math.min(...zs.map((z) => z[0])), x1 = Math.max(...zs.map((z) => z[0])), y0 = Math.min(...zs.map((z) => z[1])), y1 = Math.max(...zs.map((z) => z[1]));
          const span = Math.max(x1 - x0, y1 - y0, 1e-6) * 1.15, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
          const sc = Math.min(W - m - 10, H - m - 10) / span;
          const ox = m + (W - m - 10) / 2, oy = (H - m) / 2;
          LT = { X: (v) => ox + (v - cx) * sc, Y: (v) => oy - (v - cy) * sc, inv: (px, py) => [cx + (px - ox) / sc, cy - (py - oy) / sc] };
          axes2(g, c, span, cx, cy, sc, ox, oy, m);
          zs.forEach((z, i) => thumb(g, c, LT.X(z[0]) - T / 2, LT.Y(z[1]) - T / 2, T, D[i], i === sel));
          if (pick) cross(g, c, LT.X(pick[0]), LT.Y(pick[1]));
        } else {
          latCap.textContent = t("lat1Cap");
          const x0 = Math.min(...zs.map((z) => z[0])), x1 = Math.max(...zs.map((z) => z[0])), span = Math.max(x1 - x0, 1e-6) * 1.1, cx = (x0 + x1) / 2;
          const sc = (W - 30) / span, ox = W / 2, base = H - 34;
          LT = { X: (v) => ox + (v - cx) * sc, inv: (px) => [cx + (px - ox) / sc] };
          g.strokeStyle = c.line; g.lineWidth = 1.5; g.beginPath(); g.moveTo(10, base); g.lineTo(W - 10, base); g.stroke();
          const st = niceStepLocal(span / 4); g.font = font(10); g.fillStyle = c.muted; g.textAlign = "center";
          for (let v = Math.ceil((cx - span / 2) / st) * st; v <= cx + span / 2; v += st) { g.fillRect(LT.X(v), base, 1, 5); g.fillText(fmt(v, st < 1 ? 1 : 0), LT.X(v), base + 17); }
          // 疊放：同一處太擠就往上一層
          const order = zs.map((z, i) => i).sort((a, b) => zs[a][0] - zs[b][0]), rows = [];
          const pos = [];
          order.forEach((i) => { const x = LT.X(zs[i][0]); let r = 0; while (rows[r] != null && x - rows[r] < T + 1) r++; rows[r] = x; pos[i] = [x, r]; });
          order.forEach((i) => { const [x, r] = pos[i]; const y = base - 8 - T - r * (T + 2); g.strokeStyle = c.line; g.lineWidth = 1; g.beginPath(); g.moveTo(x, base); g.lineTo(x, y + T); g.stroke(); });
          order.forEach((i) => { const [x, r] = pos[i]; thumb(g, c, x - T / 2, base - 8 - T - r * (T + 2), T, D[i], i === sel); });
          if (pick) { g.fillStyle = c.hot; g.beginPath(); g.moveTo(LT.X(pick[0]), base - 2); g.lineTo(LT.X(pick[0]) - 6, base + 9); g.lineTo(LT.X(pick[0]) + 6, base + 9); g.fill(); }
        }
        drawDec();
      }
      function niceStepLocal(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw || 1))), r = raw / p; return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p; }
      function axes2(g, c, span, cx, cy, sc, ox, oy, m) {
        const st = niceStepLocal(span / 4), W = lat.w, H = lat.h;
        g.font = font(10); g.fillStyle = c.muted; g.strokeStyle = c.line; g.lineWidth = 1;
        g.textAlign = "center";
        for (let v = Math.ceil((cx - span / 2) / st) * st; v <= cx + span / 2; v += st) { const x = ox + (v - cx) * sc; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(x, 4); g.lineTo(x, H - m); g.stroke(); g.globalAlpha = 1; g.fillText(fmt(v, st < 1 ? 1 : 0), x, H - m + 13); }
        g.textAlign = "right";
        for (let v = Math.ceil((cy - span / 2) / st) * st; v <= cy + span / 2; v += st) { const y = oy - (v - cy) * sc; if (y < 4 || y > H - m) continue; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(m, y); g.lineTo(W - 4, y); g.stroke(); g.globalAlpha = 1; g.fillText(fmt(v, st < 1 ? 1 : 0), m - 3, y + 3); }
        g.textAlign = "right"; g.fillText("z₁", W - 6, H - m - 4); g.textAlign = "left"; g.fillText("z₂", m + 4, 12);
      }
      function thumb(g, c, x, y, T, img, on) {
        g.fillStyle = c.panel; g.fillRect(x - 1, y - 1, T + 2, T + 2);
        nn.drawImg(g, x, y, T, img, c.off, c.on, 0);
        g.strokeStyle = on ? c.hot : c.line; g.lineWidth = on ? 3 : 1; g.strokeRect(x - (on ? 2 : 0.5), y - (on ? 2 : 0.5), T + (on ? 4 : 1), T + (on ? 4 : 1));
      }
      function cross(g, c, x, y) { g.strokeStyle = c.hot; g.lineWidth = 2.5; g.beginPath(); g.moveTo(x - 7, y); g.lineTo(x + 7, y); g.moveTo(x, y - 7); g.lineTo(x, y + 7); g.stroke(); }
      function drawDec() {
        const c = C(), g = decC.begin();
        const z = pick ? Float64Array.from(pick) : V.codes[sel];
        const y = Float64Array.from(V.R.vDec.forward(z)); nn.drawImg(g, 0, 0, 60, y, c.off, c.on, 1);
        const nb = nn.nearest(D, y), [w, hh] = F[nb.i];
        decTxt.innerHTML = t(pick ? "decPick" : "decSel", { z: Array.from(z).map((v) => fmt(v, 2)).join(", "), w, h: hh, diff: nb.d === 0 ? t("decSame") : t("decDiff", { d: nb.d }) });
      }
      lat.el.addEventListener("click", (e) => {
        if (!LT) return;
        const [px, py] = lat.toLocal(e);
        if (k === 2) {
          // 點到縮圖就選那張，否則解碼點到的位置
          const hit = V.codes.findIndex((z) => Math.abs(LT.X(z[0]) - px) <= 9 && Math.abs(LT.Y(z[1]) - py) <= 9);
          if (hit >= 0) { sel = hit; pick = null; drawPair(); drawGal(); drawCalc(); }
          else pick = LT.inv(px, py);
        } else pick = LT.inv(px);
        drawLat();
      });

      function drawLoss() {
        const R = cur();
        nn.lineChart(lossSvg, { W: 360, H: 200, xMax: EPOCHS, yMin: 0.001, yMax: 100, log: true, xLabel: t("lossAxis"), marker: viewE,
          series: [{ pts: R.hist.filter((p, i) => i % 2 === 0 || i === R.hist.length - 1), color: "var(--accent)" }] });
      }

      function drawStatus() {
        const R = cur(), perfect = V.wrong.filter((w) => w === 0).length, avg = V.loss.reduce((a, b) => a + b, 0) / 49;
        const head = t(R.loop.running ? "statusHeadRun" : "statusHead", { e: viewE, loss: fmtL(avg), n: perfect });
        let kd = "info", tail = "";
        if (viewE === 0) tail = t(R.loop.running ? "tailStartRun" : "tailStart");
        else if (perfect === 49) { kd = "ok"; tail = t("tailPerfect", { k }); }
        else if (R.loop.finished && viewE === EPOCHS && k === 1) { kd = "warn"; tail = t("tailOne"); }
        status.className = "status status--" + kd; status.textContent = head + tail;
      }
      const fmtL = (v) => (v < 0.01 ? v.toExponential(1).replace("e-", "e−") : fmt(v, 3));

      function drawCalc() {
        const x = D[sel], y = V.outs[sel], z = V.codes[sel], [w, hh] = F[sel];
        let worst = 0, wv = -1;
        for (let i = 0; i < 49; i++) { const p = Math.min(1 - 1e-7, Math.max(1e-7, y[i])), ce = -(x[i] * Math.log(p) + (1 - x[i]) * Math.log(1 - p)); if (ce > wv) { wv = ce; worst = i; } }
        const xi = x[worst], yi = y[worst];
        const row = (kk, b) => `<span class="calc-row"><span class="k">${kk}</span> ${b}</span>`;
        calc.innerHTML =
          row(t("calcInK"), t("calcIn", { w, h: hh, on: w * hh })) +
          row(t("calcEncK"), t("calcEnc", { z: Array.from(z).map((v) => fmt(v, 2)).join(", ") })) +
          row(t("calcDecK"), t("calcDec", { cls: V.wrong[sel] ? "n" : "p", n: V.wrong[sel] })) +
          row(t("calcLossK"), t("calcLoss", { v: fmtL(V.loss[sel]) })) +
          row(t("calcWorstK"), t("calcWorst", { r: Math.floor(worst / 7) + 1, c: (worst % 7) + 1, x: xi, y: fmt(yi, 3), arg: xi ? fmt(yi, 3) : "(1 − " + fmt(yi, 3) + ")", v: fmtL(wv) }));
      }

      note.innerHTML = t("note");

      ctx.onTheme(() => { drawPair(); drawGal(); drawLat(); });
      ensure();
      return { destroy() { Object.values(runs).forEach((R) => R.loop.pause()); } };
    },
  });
})();
