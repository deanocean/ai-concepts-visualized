/* 変分オートエンコーダ（VAE）（シラバス §24）
   資料：和オートエンコーダ頁相同的 49 張 7×7 長方形（寬 1〜7 × 高 1〜7）。
   エンコーダ 49 →(ReLU)32 → 4 個輸出（μ₁, μ₂, ln σ₁², ln σ₂²）；z = μ + σ⊙ε（ε〜N(0,1)，再パラメータ化トリック）；
   デコーダ 2 →(ReLU)32 → 49（Sigmoid）。損失 = 復元誤差（交差エントロピー）+ β × KL，KL = ½Σ(μ² + σ² − 1 − ln σ²)。
   頁面上即時訓練：Adam η=0.005、迷你批次 7 張、800 エポック，每 10 エポック存一份權重。β = 0／1／4 各自訓練。
   「從 N(0,1) 抽 200 個點」的統計每次重算（固定種子）。 */
(function () {
  const { D, F } = Atlas.nn.rects();
  const EPOCHS = 800, SNAP = 10, LR = 0.005, BS = 7, NS = 200;

  Atlas.define("vae", {
    lead: `<p><a href="#ae">オートエンコーダ</a>的中間層，數值散在哪裡沒有規則，隨便挑一點解碼不一定像樣，所以不能拿來產生新圖。<span class="jp">変分オートエンコーダ</span>（變分自編碼器，<span class="en">variational autoencoder, VAE</span>）讓エンコーダ輸出「一個範圍」（平均 μ 與標準差 σ）而不是一個點，並把每個範圍都拉向原點附近的標準常態分布；之後從標準常態分布抽一點交給デコーダ，就能生成像樣的圖。</p>`,
    closing: `VAE 的損失是「還原得好不好」加上「範圍離標準常態分布多遠（KL 情報量）」：後者把所有圖的位置都收進標準常態分布的範圍、排得緊密，而訓練時每張圖要從自己範圍內隨機的一點還原，デコーダ連圖與圖之間的空隙也學會解碼，所以從標準常態分布隨便抽一點都解碼得出像樣的圖。β 決定這股拉力：β = 0 時退回普通的オートエンコーダ，β 太大則還原變模糊。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { MLP, rng, gauss } = nn;
      let beta = 1, seed = 1, sel = 30, follow = true, viewE = 0, pick = null, zoom = "fix", epsSeed = 1, drawSeed = 1;
      const runs = {};

      bench.append(h("style", { text: `
        .va-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; align-items: flex-start; }
        .va-fig { flex: 1 1 300px; max-width: 400px; min-width: 0; }
        .va-fig > svg { display: block; width: 100%; height: auto; }
        .va-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .va-dec { display: flex; align-items: center; gap: 10px; justify-content: center; font-size: 13px; color: var(--muted); margin-top: 6px; min-height: 64px; }
        .va-dec canvas { width: 60px !important; flex: none; }
        .va-hit { cursor: pointer; touch-action: manipulation; }
        .va-stat { font-size: 13px; color: var(--muted); text-align: center; margin-top: 4px; }
        .va-stat b { color: var(--ink); font-family: var(--f-mono); }
      ` }));

      /* ---------- 控制 ---------- */
      const bSeg = ui.seg({ label: "β（KL 項的權重）", value: beta, options: [{ value: 0, label: "0（不拉）" }, { value: 1, label: "1（標準 VAE）" }, { value: 4, label: "4（β-VAE）" }], onChange: (v) => { beta = v; pick = null; ensure(); } });
      const runBtn = ui.btn("▶ 開始訓練", () => { const R = cur(); R.loop.running ? R.loop.pause() : (follow = true, R.loop.start()); refreshBtns(); }, { primary: true });
      const ffBtn = ui.btn("⏭ 快轉到結束", () => { follow = true; cur().loop.fastForward(); refreshBtns(); });
      const seedBtn = ui.btn("換一組初始值", () => { cur().loop.pause(); seed = (seed % 5) + 1; pick = null; ensure(); }, { sm: true });
      const prog = ui.slider({ id: "va-ep", label: "查看第幾エポック的狀態", min: 0, max: EPOCHS, step: SNAP, value: 0, format: (v) => v + " / " + EPOCHS,
        onInput: (v) => { const R = cur(); const last = R.snaps[R.snaps.length - 1].e; if (v > last) { v = last; prog.set(v); } follow = v === last; viewE = v; show(); } });
      const zSeg = ui.seg({ label: "平面顯示範圍", value: zoom, options: [{ value: "fix", label: "−3〜3" }, { value: "all", label: "全部" }], onChange: (v) => { zoom = v; drawLat(); } });
      const epsBtn = ui.btn("重抽 ε", () => { epsSeed++; drawLat(); drawCalc(); }, { sm: true });
      const drawBtn = ui.btn("再抽 16 個", () => { drawSeed++; drawSamples(); }, { sm: true });

      /* ---------- 圖 ---------- */
      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const lat = nn.canvas(330, 330, "每張圖在潛在空間的平均與範圍"); lat.el.classList.add("va-hit");
      const decC = nn.canvas(60, 60, "在潛在空間選的位置解碼出來的圖"), decTxt = h("span");
      const map = nn.canvas(330, 330, "潛在空間 −2 到 2 每隔 0.5 解碼出來的圖");
      const smp = nn.canvas(330, 84, "從標準常態分布抽 16 個點解碼出來的圖");
      const smpStat = h("div", { class: "va-stat" });
      const lossSvg = s("svg", { role: "img", "aria-label": "每個エポック的復元誤差與 KL" });

      bench.append(
        h("div", { class: "bench__row" }, bSeg.el, seedBtn),
        h("div", { class: "bench__row" }, runBtn, ffBtn, prog.el),
        status,
        h("div", { class: "va-row" },
          h("div", { class: "va-fig" },
            h("div", { class: "va-cap", html: "每張圖的平均 μ（縮圖）與範圍 ±σ（橘色＝選中的圖，× ＝這次抽到的 z）。虛線圓＝標準常態分布的 1σ、2σ。點縮圖選圖，點空白處解碼" }),
            lat.el, h("div", { class: "va-dec" }, decC.el, decTxt), h("div", { class: "bench__row", style: { justifyContent: "center" } }, zSeg.el)),
          h("div", { class: "va-fig" }, h("div", { class: "va-cap" }, "潛在空間地圖：z₁、z₂ 從 −2 到 2 每隔 0.5 交給デコーダ"), map.el)),
        h("div", { class: "bench__row" }, epsBtn, h("span", { class: "hint" }, "ε 是從標準常態分布抽的亂數；換一個 ε，同一張圖會落在範圍內的另一點")),
        calc,
        h("div", { class: "va-row" },
          h("div", { class: "va-fig" }, h("div", { class: "va-cap" }, "生成：從標準常態分布 N(0, 1) 抽點 → デコーダ"), smp.el, smpStat, h("div", { class: "bench__row", style: { justifyContent: "center", marginTop: "6px" } }, drawBtn)),
          h("div", { class: "va-fig" }, h("div", { class: "va-cap", html: `每張圖的平均損失（對數刻度）：<span style="color:var(--accent)">━ 復元誤差</span>　<span style="color:var(--hot)">━ KL</span>` }), lossSvg)),
        note);

      /* ---------- 訓練 ---------- */
      const mkEnc = (r) => new MLP([49, 32, 4], ["relu", "lin"], r), mkDec = (r) => new MLP([2, 32, 49], ["relu", "sigmoid"], r);
      function snapOf(R) { return { e: R.epoch, enc: R.enc.snapshot(), dec: R.dec.snapshot() }; }
      function makeRun(bt, sd) {
        const r = rng(sd * 2003 + bt * 13 + 5);
        const R = { beta: bt, enc: mkEnc(r), dec: mkDec(r), vEnc: mkEnc(rng(0)), vDec: mkDec(rng(0)), order: D.map((_, i) => i), epoch: 0, hist: [], snaps: [] };
        R.snaps.push(snapOf(R));
        const dOut = new Float64Array(49), dE = new Float64Array(4), z = new Float64Array(2);
        R.loop = nn.loop(ctx, {
          total: EPOCHS, perTick: 5,
          run(from, n) {
            for (let q = 0; q < n; q++) {
              const o = R.order;
              for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = o[i]; o[i] = o[j]; o[j] = t; }
              let rec = 0, kl = 0;
              for (let b = 0; b < o.length; b += BS) {
                for (let p = b; p < b + BS; p++) {
                  const x = D[o[p]], out = R.enc.forward(x);
                  const mu = [out[0], out[1]], lv = [out[2], out[3]], sd2 = [Math.exp(0.5 * lv[0]), Math.exp(0.5 * lv[1])], eps = [gauss(r), gauss(r)];
                  z[0] = mu[0] + sd2[0] * eps[0]; z[1] = mu[1] + sd2[1] * eps[1];
                  for (let j = 0; j < 2; j++) kl += 0.5 * (mu[j] * mu[j] + sd2[j] * sd2[j] - 1 - lv[j]);
                  const y = R.dec.forward(z); rec += nn.bce(x, y);
                  for (let i = 0; i < 49; i++) dOut[i] = y[i] - x[i];
                  const dz = R.dec.backward(dOut, true);
                  for (let j = 0; j < 2; j++) {
                    dE[j] = dz[j] + bt * mu[j];                                                     // ∂/∂μ
                    dE[2 + j] = dz[j] * eps[j] * 0.5 * sd2[j] + bt * 0.5 * (sd2[j] * sd2[j] - 1);    // ∂/∂(ln σ²)
                  }
                  R.enc.backward(dE);
                }
                R.enc.adam(LR, 1 / BS); R.dec.adam(LR, 1 / BS);
              }
              R.epoch++; R.hist.push([R.epoch, rec / D.length, kl / D.length]);
              if (R.epoch % SNAP === 0) R.snaps.push(snapOf(R));
            }
          },
          onTick() { if (R === cur()) { if (follow && R.snaps[R.snaps.length - 1].e !== viewE) show(); else if (!follow) { drawLoss(); drawStatus(); } refreshBtns(); } },
          onEnd() { if (R === cur()) { show(); refreshBtns(); } },
        });
        return R;
      }
      const cur = () => runs[beta + "-" + seed];
      function ensure() {
        Object.values(runs).forEach((R) => R !== runs[beta + "-" + seed] && R.loop.pause());
        if (!runs[beta + "-" + seed]) runs[beta + "-" + seed] = makeRun(beta, seed);
        const R = cur(); follow = true;
        show(); refreshBtns();
      }
      function refreshBtns() {
        const R = cur();
        runBtn.textContent = R.loop.running ? "❚❚ 暫停" : R.loop.finished ? "✓ 訓練完成" : R.epoch ? "▶ 繼續訓練" : "▶ 開始訓練";
        runBtn.disabled = R.loop.finished; ffBtn.disabled = R.loop.finished;
      }

      /* ---------- 顯示 ---------- */
      let V = null;
      function computeView() {
        const R = cur();
        const snap = follow ? R.snaps[R.snaps.length - 1] : R.snaps.find((q) => q.e === viewE) || R.snaps[R.snaps.length - 1];
        viewE = snap.e; R.vEnc.restore(snap.enc); R.vDec.restore(snap.dec);
        const mu = [], sg = [], lv = [], outs = [], wrong = [], rec = [], kl = [];
        D.forEach((x) => {
          const o = R.vEnc.forward(x), m = [o[0], o[1]], l = [o[2], o[3]];
          mu.push(m); lv.push(l); sg.push(l.map((v) => Math.exp(0.5 * v)));
          kl.push(0.5 * (m[0] ** 2 + Math.exp(l[0]) - 1 - l[0] + m[1] ** 2 + Math.exp(l[1]) - 1 - l[1]));
          const y = Float64Array.from(R.vDec.forward(Float64Array.from(m))); // 還原用平均 μ
          outs.push(y); rec.push(nn.bce(x, y));
          let w = 0; for (let i = 0; i < 49; i++) if ((y[i] > 0.5) !== (x[i] > 0.5)) w++; wrong.push(w);
        });
        // 從 N(0,1) 抽 200 點
        const r = rng(777), hit = new Set(); let far = 0;
        for (let i = 0; i < NS; i++) { const y = R.vDec.forward(Float64Array.from([gauss(r), gauss(r)])); const nb = nn.nearest(D, y); if (nb.d === 0) hit.add(nb.i); if (nb.d >= 3) far++; }
        V = { R, mu, sg, lv, outs, wrong, rec, kl, kinds: hit.size, far };
      }
      function show() { computeView(); prog.set(viewE); drawLat(); drawMap(); drawSamples(); drawLoss(); drawStatus(); drawCalc(); }
      const C = () => ({ off: Atlas.rgb(ctx.color("--panel-2")), on: Atlas.rgb(ctx.color("--ink")), hot: ctx.color("--hot"), hotSoft: ctx.color("--hot-soft"), line: ctx.color("--line-strong"), muted: ctx.color("--muted"), acc: ctx.color("--accent"), accSoft: ctx.color("--accent-soft"), ink: ctx.color("--ink"), panel: ctx.color("--panel") });
      const font = (px, w = 400) => `${w} ${px}px ${ctx.color("--f-body") || "sans-serif"}`;
      const fmtL = (v) => (v < 0.01 ? v.toExponential(1).replace("e-", "e−") : fmt(v, v >= 100 ? 1 : 3));
      const niceStep = (raw) => { const p = Math.pow(10, Math.floor(Math.log10(raw || 1))), q = raw / p; return (q < 1.5 ? 1 : q < 3.5 ? 2 : q < 7.5 ? 5 : 10) * p; };
      const epsFor = (i) => { const r = rng(epsSeed * 97 + i); return [gauss(r), gauss(r)]; };

      /* 潛在空間：平均 μ（縮圖）＋ ±σ 橢圓 */
      let LT = null;
      function drawLat() {
        const c = C(), g = lat.begin(), W = lat.w, H = lat.h, m = 26, T = 15;
        let lim = 3;
        if (zoom === "all") lim = Math.max(3, ...V.mu.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.08;
        const sc = (Math.min(W, H) - m - 8) / (2 * lim), ox = m + (W - m - 8) / 2, oy = (H - m) / 2 + 4;
        LT = { X: (v) => ox + v * sc, Y: (v) => oy - v * sc, inv: (px, py) => [(px - ox) / sc, (oy - py) / sc], lim };
        // 格線
        const st = niceStep(lim / 2); g.font = font(10); g.lineWidth = 1;
        for (let v = -Math.floor(lim / st) * st; v <= lim + 1e-9; v += st) {
          g.strokeStyle = c.line; g.globalAlpha = v === 0 ? 0.9 : 0.35;
          g.beginPath(); g.moveTo(LT.X(v), LT.Y(lim)); g.lineTo(LT.X(v), LT.Y(-lim)); g.moveTo(LT.X(-lim), LT.Y(v)); g.lineTo(LT.X(lim), LT.Y(v)); g.stroke();
          g.globalAlpha = 1; g.fillStyle = c.muted; g.textAlign = "center"; g.fillText(fmt(v, st < 1 ? 1 : 0), LT.X(v), LT.Y(-lim) + 13);
          g.textAlign = "right"; g.fillText(fmt(v, st < 1 ? 1 : 0), LT.X(-lim) - 4, LT.Y(v) + 3);
        }
        g.textAlign = "right"; g.fillText("z₁", LT.X(lim) - 2, LT.Y(0) - 5); g.textAlign = "left"; g.fillText("z₂", LT.X(0) + 5, LT.Y(lim) + 10);
        // 標準常態分布的 1σ、2σ 圓
        g.strokeStyle = c.acc; g.lineWidth = 1.5; g.setLineDash([5, 4]);
        [1, 2].forEach((rr) => { g.beginPath(); g.arc(LT.X(0), LT.Y(0), rr * sc, 0, Math.PI * 2); g.stroke(); });
        g.setLineDash([]);
        g.save(); g.beginPath(); g.rect(LT.X(-lim), LT.Y(lim), 2 * lim * sc, 2 * lim * sc); g.clip();
        // ±σ 橢圓
        V.mu.forEach((p, i) => {
          if (i === sel) return;
          g.fillStyle = c.accSoft; g.strokeStyle = c.acc; g.lineWidth = 0.8; g.globalAlpha = 0.55;
          g.beginPath(); g.ellipse(LT.X(p[0]), LT.Y(p[1]), Math.max(0.5, V.sg[i][0] * sc), Math.max(0.5, V.sg[i][1] * sc), 0, 0, Math.PI * 2); g.fill(); g.stroke(); g.globalAlpha = 1;
        });
        let out = 0;
        V.mu.forEach((p, i) => {
          if (Math.abs(p[0]) > lim || Math.abs(p[1]) > lim) { out++; return; }
          if (i !== sel) thumb(g, c, LT.X(p[0]) - T / 2, LT.Y(p[1]) - T / 2, T, D[i], false);
        });
        // 選中的圖：橢圓、μ、這次抽到的 z
        const p = V.mu[sel], sd = V.sg[sel], e = epsFor(sel), zz = [p[0] + sd[0] * e[0], p[1] + sd[1] * e[1]];
        g.fillStyle = c.hotSoft; g.strokeStyle = c.hot; g.lineWidth = 2; g.globalAlpha = 0.8;
        g.beginPath(); g.ellipse(LT.X(p[0]), LT.Y(p[1]), Math.max(1, sd[0] * sc), Math.max(1, sd[1] * sc), 0, 0, Math.PI * 2); g.fill(); g.stroke(); g.globalAlpha = 1;
        thumb(g, c, LT.X(p[0]) - T / 2, LT.Y(p[1]) - T / 2, T, D[sel], true);
        g.strokeStyle = c.hot; g.lineWidth = 2.5; const zx = LT.X(zz[0]), zy = LT.Y(zz[1]);
        g.beginPath(); g.moveTo(zx - 5, zy - 5); g.lineTo(zx + 5, zy + 5); g.moveTo(zx + 5, zy - 5); g.lineTo(zx - 5, zy + 5); g.stroke();
        if (pick) { g.strokeStyle = c.ink; g.lineWidth = 2; g.beginPath(); g.moveTo(LT.X(pick[0]) - 7, LT.Y(pick[1])); g.lineTo(LT.X(pick[0]) + 7, LT.Y(pick[1])); g.moveTo(LT.X(pick[0]), LT.Y(pick[1]) - 7); g.lineTo(LT.X(pick[0]), LT.Y(pick[1]) + 7); g.stroke(); }
        g.restore();
        if (out) { g.font = font(11.5, 700); g.fillStyle = c.hot; g.textAlign = "left"; g.fillText(`有 ${out} 張圖的平均在範圍外（改選「全部」）`, LT.X(-lim) + 4, LT.Y(lim) + 14); }
        drawDec();
      }
      function thumb(g, c, x, y, T, img, on) {
        g.fillStyle = c.panel; g.fillRect(x - 1, y - 1, T + 2, T + 2);
        nn.drawImg(g, x, y, T, img, c.off, c.on, 0);
        g.strokeStyle = on ? c.hot : c.line; g.lineWidth = on ? 3 : 1; g.strokeRect(x - (on ? 2 : 0.5), y - (on ? 2 : 0.5), T + (on ? 4 : 1), T + (on ? 4 : 1));
      }
      function drawDec() {
        const c = C(), g = decC.begin();
        const z = pick ? pick : V.mu[sel];
        const y = Float64Array.from(V.R.vDec.forward(Float64Array.from(z))); nn.drawImg(g, 0, 0, 60, y, c.off, c.on, 1);
        const nb = nn.nearest(D, y), [w, hh] = F[nb.i];
        decTxt.innerHTML = `${pick ? "你點的位置" : "選中的圖的平均 μ"} (${z.map((v) => fmt(v, 2)).join(", ")}) → デコーダ → 左圖<br>最像的訓練圖：寬 ${w}・高 ${hh}，${nb.d === 0 ? "四捨五入後完全一樣" : `差 ${nb.d} 格`}`;
      }
      lat.el.addEventListener("click", (e) => {
        if (!LT) return;
        const [px, py] = lat.toLocal(e);
        const hit = V.mu.findIndex((p) => Math.abs(p[0]) <= LT.lim && Math.abs(p[1]) <= LT.lim && Math.abs(LT.X(p[0]) - px) <= 8 && Math.abs(LT.Y(p[1]) - py) <= 8);
        if (hit >= 0) { sel = hit; pick = null; drawCalc(); } else pick = LT.inv(px, py);
        drawLat();
      });

      /* 潛在空間地圖：9×9 */
      function drawMap() {
        const c = C(), g = map.begin(), m = 26, n = 9, cell = (map.w - m - 6) / n, T = cell - 4;
        g.font = font(10); g.fillStyle = c.muted;
        for (let a = 0; a < n; a++) {
          const v = -2 + a * 0.5;
          g.textAlign = "center"; if (a % 2 === 0) g.fillText(fmt(v, 0), m + a * cell + cell / 2, map.h - 4);
          g.textAlign = "right"; if (a % 2 === 0) g.fillText(fmt(v, 0), m - 4, (n - 1 - a) * cell + cell / 2 + 4);
        }
        g.textAlign = "left"; g.fillText("z₂", 2, 10); g.textAlign = "right"; g.fillText("z₁", map.w - 2, map.h - 4);
        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
          const z = Float64Array.from([-2 + a * 0.5, -2 + b * 0.5]);
          const y = V.R.vDec.forward(z);
          nn.drawImg(g, m + a * cell + 2, (n - 1 - b) * cell + 2, T, y, c.off, c.on, 0);
        }
      }

      /* 從 N(0,1) 抽 16 個 */
      function drawSamples() {
        const c = C(), g = smp.begin(), r = rng(500 + drawSeed), cell = smp.w / 8, T = cell - 6;
        for (let i = 0; i < 16; i++) {
          const y = V.R.vDec.forward(Float64Array.from([gauss(r), gauss(r)]));
          nn.drawImg(g, (i % 8) * cell + 3, Math.floor(i / 8) * (T + 6) + 3, T, y, c.off, c.on, 0);
        }
        smpStat.innerHTML = `固定抽 200 個點統計：解碼後和某張訓練圖完全一樣的，涵蓋 <b>${V.kinds}</b> / 49 種；和每張訓練圖都差 3 格以上（四不像）的有 <b>${V.far}</b> 個`;
      }

      function drawLoss() {
        const R = cur();
        nn.lineChart(lossSvg, { W: 360, H: 200, xMax: EPOCHS, yMin: 0.01, yMax: 1000, log: true, xLabel: "エポック", marker: viewE,
          series: [{ pts: R.hist.filter((p, i) => i % 2 === 0).map((p) => [p[0], p[1]]), color: "var(--accent)" }, { pts: R.hist.filter((p, i) => i % 2 === 0).map((p) => [p[0], p[2]]), color: "var(--hot)" }] });
      }

      function drawStatus() {
        const R = cur(), perfect = V.wrong.filter((w) => w === 0).length, rec = V.rec.reduce((a, b) => a + b, 0) / 49, kl = V.kl.reduce((a, b) => a + b, 0) / 49;
        let head = `第 ${viewE} エポック${R.loop.running ? "（訓練中）" : ""}：復元誤差 ${fmtL(rec)}、KL ${fmtL(kl)}；49 張中 ${perfect} 張完全還原；從 N(0,1) 抽點涵蓋 ${V.kinds} / 49 種`;
        let kd = "info", tail = "";
        if (viewE === 0) tail = R.loop.running ? "。剛開始訓練" : "。還沒訓練。按「▶ 開始訓練」開始（約 5 秒）";
        else if (R.loop.finished && viewE === EPOCHS) {
          const far = Math.max(...V.mu.map((p) => Math.hypot(p[0], p[1]))), sAvg = V.sg.reduce((a, q) => a + q[0] + q[1], 0) / 98;
          if (beta === 0) { kd = "warn"; tail = `。沒有 KL 的拉力：平均離原點最遠到 ${fmt(far, 1)}、σ 平均只有 ${fmt(sAvg, 3)}，原點附近只對到少數幾種圖`; }
          else if (perfect < 45) { kd = "warn"; tail = "。拉力太強，圖擠在一起，還原變模糊"; }
          else if (V.kinds >= 40) { kd = "ok"; tail = "。各圖都收進標準常態分布的範圍內，從 N(0,1) 抽點幾乎每種都生成得出來"; }
        }
        status.className = "status status--" + kd; status.textContent = head + tail;
      }

      function drawCalc() {
        const mu = V.mu[sel], sg = V.sg[sel], lv = V.lv[sel], e = epsFor(sel), [w, hh] = F[sel];
        const z = [mu[0] + sg[0] * e[0], mu[1] + sg[1] * e[1]];
        const y = V.R.vDec.forward(Float64Array.from(z)), recZ = nn.bce(D[sel], y);
        const klj = [0, 1].map((j) => 0.5 * (mu[j] ** 2 + sg[j] ** 2 - 1 - lv[j]));
        const row = (kk, b) => `<span class="calc-row"><span class="k">${kk}</span> ${b}</span>`;
        const pair = (a) => "(" + a.map((v) => fmt(v, 3)).join(", ") + ")";
        calc.innerHTML =
          row(`寬 ${w} × 高 ${hh} → エンコーダ`, `μ = ${pair(mu)}，ln σ² = ${pair(lv)} → σ = ${pair(sg)}`) +
          row("抽點", `ε = ${pair(e)}，z = μ + σ × ε = <span class="hl">${pair(z)}</span>`) +
          row("復元誤差", `把 z 交給デコーダ，和輸入比：交差エントロピー = ${fmtL(recZ)}`) +
          row("KL", `½ Σ (μ² + σ² − 1 − ln σ²) = ${fmtL(klj[0])} + ${fmtL(klj[1])} = <span class="hl">${fmtL(klj[0] + klj[1])}</span>`) +
          row("這張圖的損失", `復元誤差 + β × KL = ${fmtL(recZ)} + ${beta} × ${fmtL(klj[0] + klj[1])} = <span class="hl">${fmtL(recZ + beta * (klj[0] + klj[1]))}</span>`);
      }

      note.innerHTML = `抽點本身是隨機的，沒辦法對「抽」這個動作求梯度；改寫成 z = μ + σ × ε，把隨機性全部交給 ε，μ 和 σ 就能照常用<a href="#backprop">誤差逆伝播法</a>學，這叫<span class="jp">再パラメータ化トリック</span>（重新參數化技巧，<span class="en">reparameterization trick</span>）。KL 是 <a href="#loss">KL 情報量</a>：衡量「平均 μ、標準差 σ 的常態分布」離標準常態分布多遠，μ = 0、σ = 1 時剛好是 0；KL 越小，每張圖的範圍越像標準常態分布。網路實際輸出 ln σ² 而不是 σ，是為了讓 σ 永遠是正的。<span class="jp">β-VAE</span> 把 β 調大，目的是讓潛在空間的每個軸各自對應一個因素（例如一軸管寬、一軸管高），代價是還原變模糊；在這個小例子裡，加大 β 最明顯的變化是還原變模糊。其他變形：<span class="jp">VQ-VAE</span> 讓潛在變數只能從一組學出來的向量（<span class="jp">コードブック</span>，<span class="en">codebook</span>）中挑最接近的一個，變成離散的；<span class="jp">info VAE</span> 修改目標函數，讓潛在變數保留更多關於輸入的資訊。`;

      ctx.onTheme(() => { drawLat(); drawMap(); drawSamples(); });
      ensure();
      return { destroy() { Object.values(runs).forEach((R) => R.loop.pause()); } };
    },
  });
})();
