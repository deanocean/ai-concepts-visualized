/* Diffusion Model（シラバス §30）
   真資料：半徑 2 圓周上 8 團，共 400 個點（和 GAN 頁的「八團」同一種分布，固定種子）。
   正向：T = 40 步，β_t 從 0.001 線性增加到 0.2；ᾱ_t = Π(1 − β_s)；x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε，可一步算到任一個 t。
   網路 ε̂(x_t, t)：輸入 x_t（2）＋ t 的 5 個特徴（t/T、sin・cos）→ 32 → 32 → 2（ReLU），損失 |ε̂ − ε|²。
   頁面上即時訓練：Adam η=0.005（最後 30% 降為 0.0015）、每步 64 筆、3000 步。
   逆向（DDPM）：x_{t−1} = (x_t − β_t/√(1 − ᾱ_t)·ε̂)/√(1 − β_t) + σ_t·z，σ_t² = β_t(1 − ᾱ_{t−1})/(1 − ᾱ_t)。 */
(function () {
  const T = 40, STEPS = 3000, R2 = 2;
  const BETA = Array.from({ length: T }, (_, i) => 0.001 + (0.2 - 0.001) * i / (T - 1)); // BETA[t−1] = β_t
  const AB = [1]; for (let t = 1; t <= T; t++) AB.push(AB[t - 1] * (1 - BETA[t - 1]));      // AB[t] = ᾱ_t
  const temb = (t) => { const u = t / T; return [u, Math.sin(Math.PI * u), Math.cos(Math.PI * u), Math.sin(2 * Math.PI * u), Math.cos(2 * Math.PI * u)]; };

  Atlas.define("diffusion", {
    lead: `<p><a href="#gan">GAN</a> 靠兩個網路互相較勁，訓練容易不穩，還可能只會生成其中幾種。<span class="jp">Diffusion Model</span>（擴散模型）換個做法：把資料一點一點加上雜訊，直到變成純雜訊（這一步只是固定的算式，不用學）；只訓練一個網路去猜「加進去的雜訊是什麼」。生成時從純雜訊出發，一步步減掉猜到的雜訊。</p>`,
    closing: `正向過程只是照公式加雜訊，網路要學的只有「看到加了雜訊的點，猜出雜訊」，這和一般的迴歸一樣穩定；生成時從純雜訊出發，重複「減掉猜到的雜訊」許多步，點就一步步聚回資料的形狀。代價是生成要走很多步、比較慢，換來的是訓練穩定、不容易漏掉某一團。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { MLP, rng, gauss } = nn;
      let mode = "fwd", tF = 20, tV = 4, kG = 0, sel = 0, selG = 0, seed = 1, genSeed = 1;

      bench.append(h("style", { text: `
        .df-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; align-items: flex-start; }
        .df-fig { flex: 1 1 300px; max-width: 420px; min-width: 0; }
        .df-fig > svg { display: block; width: 100%; height: auto; }
        .df-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .df-hit { cursor: pointer; touch-action: manipulation; }
        .df-key { display: flex; flex-wrap: wrap; gap: 4px 14px; justify-content: center; font-size: 12.5px; color: var(--muted); margin-top: 4px; }
        .df-key i { display: inline-block; width: 14px; height: 0; border-top: 2.5px solid; margin-right: 4px; vertical-align: 3px; }
      ` }));

      /* ---------- 資料 ---------- */
      const DATA = [], LAB = [], EPS = [];
      (() => { const r = rng(51); for (let i = 0; i < 400; i++) { const k = Math.floor(r() * 8), a = k * Math.PI / 4; DATA.push([R2 * Math.cos(a) + 0.12 * gauss(r), R2 * Math.sin(a) + 0.12 * gauss(r)]); LAB.push(k); }
        const q = rng(52); for (let i = 0; i < 400; i++) EPS.push([gauss(q), gauss(q)]); })();
      const xt = (i, t) => { const a = Math.sqrt(AB[t]), b = Math.sqrt(1 - AB[t]); return [a * DATA[i][0] + b * EPS[i][0], a * DATA[i][1] + b * EPS[i][1]]; };

      /* ---------- 網路與訓練（三個模式共用；打開頁面就開始） ---------- */
      let net, r, hist, step;
      const inp = new Float64Array(7);
      const predict = (x, t, N = net) => { inp[0] = x[0]; inp[1] = x[1]; const e = temb(t); for (let i = 0; i < 5; i++) inp[2 + i] = e[i]; const o = N.forward(inp); return [o[0], o[1]]; };
      let loop;
      function makeNet() {
        if (loop) loop.pause();
        r = rng(seed * 4001 + 7); net = new MLP([7, 32, 32, 2], ["relu", "relu", "lin"], r); hist = []; step = 0;
        let acc = 0;
        loop = nn.loop(ctx, { total: STEPS, perTick: 15,
          run(from, n) {
            for (let q = 0; q < n; q++) {
              let L = 0;
              for (let b = 0; b < 64; b++) {
                const i = Math.floor(r() * 400), t = 1 + Math.floor(r() * T), a = Math.sqrt(AB[t]), c = Math.sqrt(1 - AB[t]), e0 = gauss(r), e1 = gauss(r);
                const o = predict([a * DATA[i][0] + c * e0, a * DATA[i][1] + c * e1], t);
                L += (o[0] - e0) ** 2 + (o[1] - e1) ** 2;
                net.backward([2 * (o[0] - e0), 2 * (o[1] - e1)]);
              }
              net.adam(step >= STEPS * 0.7 ? 0.0015 : 0.005, 1 / 64);
              step++; acc += L / 64;
              if (step % 50 === 0) { hist.push([step, acc / 50]); acc = 0; }
            }
          },
          onTick() { genCache = null; if (mode !== "fwd") render(); refreshBtns(); } });
        genCache = null;
      }

      /* ---------- 控制 ---------- */
      const modeSeg = ui.seg({ label: "模式", value: mode, options: [{ value: "fwd", label: "① 正向：加雜訊" }, { value: "train", label: "② 訓練：學猜雜訊" }, { value: "gen", label: "③ 生成：一步步去雜訊" }], onChange: (v) => { mode = v; layout(); } });
      const runBtn = ui.btn("▶ 開始訓練", () => { loop.running ? loop.pause() : loop.start(); refreshBtns(); }, { primary: true });
      const ffBtn = ui.btn("⏭ 快轉到結束", () => { loop.fastForward(); refreshBtns(); });
      const seedBtn = ui.btn("換一組初始值", () => { seed = (seed % 5) + 1; makeNet(); render(); refreshBtns(); }, { sm: true });
      const trainRow = h("div", { class: "bench__row" }, runBtn, ffBtn, seedBtn);
      let vRow;
      function refreshBtns() { runBtn.textContent = loop.running ? "❚❚ 暫停訓練" : loop.finished ? "✓ 訓練完成" : step ? "▶ 繼續訓練" : "▶ 開始訓練"; runBtn.disabled = ffBtn.disabled = loop.finished; }

      // ① 正向
      const tSl = ui.slider({ id: "df-t", label: "加了幾步雜訊 t", min: 0, max: T, step: 1, value: tF, format: (v) => v + " / " + T, onInput: (v) => { pF.pause(); tF = v; render(); } });
      const pF = ui.player(ctx, () => { if (tF >= T) return false; tF++; tSl.set(tF); render(); return tF < T; }, 220);
      const fwdRow = h("div", { class: "bench__row" }, tSl.el, ui.btn("◀", () => { pF.pause(); if (tF > 0) { tF--; tSl.set(tF); render(); } }, { aria: "少一步" }), ui.btn("▶", () => { pF.pause(); if (tF < T) { tF++; tSl.set(tF); render(); } }, { aria: "多一步" }), pF.el, ui.btn("回到 t = 0", () => { pF.pause(); tF = 0; tSl.set(0); render(); }));
      // ② 訓練：查看的 t
      const vSl = ui.slider({ id: "df-tv", label: "查看哪個雜訊程度 t", min: 1, max: T, step: 1, value: tV, format: (v) => v + " / " + T, onInput: (v) => { tV = v; render(); } });
      vRow = h("div", { class: "bench__row" }, vSl.el);
      // ③ 生成
      const gSl = ui.slider({ id: "df-k", label: "生成進度（已去雜訊幾步）", min: 0, max: T, step: 1, value: kG, format: (v) => v + " / " + T, onInput: (v) => { pG.pause(); kG = v; render(); } });
      const pG = ui.player(ctx, () => { if (kG >= T) return false; kG++; gSl.set(kG); render(); return kG < T; }, 220);
      const genRow = h("div", { class: "bench__row" }, gSl.el, ui.btn("◀", () => { pG.pause(); if (kG > 0) { kG--; gSl.set(kG); render(); } }, { aria: "上一步" }), ui.btn("▶", () => { pG.pause(); if (kG < T) { kG++; gSl.set(kG); render(); } }, { aria: "下一步" }), pG.el,
        ui.btn("換一組起始雜訊", () => { pG.pause(); genSeed++; genCache = null; kG = 0; gSl.set(0); render(); }, { sm: true }));

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const plane = nn.canvas(360, 360, "平面上的點"); plane.el.classList.add("df-hit");
      const planeCap = h("div", { class: "df-cap" }), planeKey = h("div", { class: "df-key" });
      const side = s("svg", { role: "img", "aria-label": "輔助圖" }), sideCap = h("div", { class: "df-cap" });
      bench.append(
        h("div", { class: "bench__row" }, modeSeg.el),
        trainRow, fwdRow, vRow, genRow,
        status,
        h("div", { class: "df-row" },
          h("div", { class: "df-fig", style: { flexBasis: "340px" } }, planeCap, plane.el, planeKey),
          h("div", { class: "df-fig" }, sideCap, side)),
        calc, note);
      function layout() {
        trainRow.hidden = mode === "fwd"; fwdRow.hidden = mode !== "fwd"; vRow.hidden = mode !== "train"; genRow.hidden = mode !== "gen";
        pF.pause(); pG.pause(); render();
      }

      /* ---------- 生成的軌跡（網路變了就重算） ---------- */
      let genCache = null;
      function genAll() {
        if (genCache) return genCache;
        const q = rng(900 + genSeed);
        let X = Array.from({ length: 400 }, () => [gauss(q), gauss(q)]);
        const P = [X], E = [], Z = [];
        for (let t = T; t >= 1; t--) {
          const b = BETA[t - 1], ab = AB[t], sg = t > 1 ? Math.sqrt(b * (1 - AB[t - 1]) / (1 - ab)) : 0;
          const eh = X.map((x) => predict(x, t)), zz = X.map(() => (t > 1 ? [gauss(q), gauss(q)] : [0, 0]));
          X = X.map((x, i) => [0, 1].map((d) => (x[d] - (b / Math.sqrt(1 - ab)) * eh[i][d]) / Math.sqrt(1 - b) + sg * zz[i][d]));
          P.push(X); E.push(eh); Z.push(zz);
        }
        return (genCache = { P, E, Z });
      }
      function region(p) {
        const a = Math.atan2(p[1], p[0]), k = ((Math.round(a / (Math.PI / 4)) % 8) + 8) % 8;
        return { k, near: Math.hypot(p[0] - R2 * Math.cos(k * Math.PI / 4), p[1] - R2 * Math.sin(k * Math.PI / 4)) < 0.4 };
      }

      /* ---------- 畫 ---------- */
      const L = 3.5, PX = (x) => 26 + ((x + L) / (2 * L)) * 328, PY = (y) => 8 + ((L - y) / (2 * L)) * 328;
      const font = (px, w = 400) => `${w} ${px}px ${ctx.color("--f-body") || "sans-serif"}`;
      const dark = () => { const p = Atlas.rgb(ctx.color("--panel")); return p[0] + p[1] + p[2] < 300; };
      const hue = (k, a = 1) => `hsla(${k * 45},70%,${dark() ? 66 : 42}%,${a})`;
      function frame(g) {
        g.strokeStyle = ctx.color("--line"); g.lineWidth = 1; g.font = font(10); g.fillStyle = ctx.color("--muted");
        for (let v = -3; v <= 3; v++) {
          g.globalAlpha = 0.7; g.beginPath(); g.moveTo(PX(v), 8); g.lineTo(PX(v), 336); g.moveTo(26, PY(v)); g.lineTo(354, PY(v)); g.stroke(); g.globalAlpha = 1;
          g.textAlign = "center"; g.fillText(fmt(v, 0), PX(v), 350); g.textAlign = "right"; g.fillText(fmt(v, 0), 22, PY(v) + 3);
        }
        g.strokeStyle = ctx.color("--muted"); g.setLineDash([5, 4]); g.lineWidth = 1.2;
        [1, 2].forEach((rr) => { g.beginPath(); g.arc(PX(0), PY(0), (rr / (2 * L)) * 328, 0, Math.PI * 2); g.stroke(); });
        g.setLineDash([]);
      }
      function dot(g, p, fill, r = 2.4) { g.fillStyle = fill; g.beginPath(); g.arc(PX(p[0]), PY(p[1]), r, 0, Math.PI * 2); g.fill(); }
      function arrow(g, p, v, col, w = 1.6, dash = null) {
        const x1 = PX(p[0]), y1 = PY(p[1]), x2 = PX(p[0] + v[0]), y2 = PY(p[1] + v[1]), len = Math.hypot(x2 - x1, y2 - y1); if (len < 3) return;
        const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
        g.strokeStyle = col; g.fillStyle = col; g.lineWidth = w; g.setLineDash(dash || []);
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2 - ux * 4, y2 - uy * 4); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(x2, y2); g.lineTo(x2 - ux * 7 - uy * 3.8, y2 - uy * 7 + ux * 3.8); g.lineTo(x2 - ux * 7 + uy * 3.8, y2 - uy * 7 - ux * 3.8); g.fill();
      }
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      const pr = (a, d = 2) => "(" + a.map((v) => fmt(v, d)).join(", ") + ")";
      const setStatus = (k, t) => { status.className = "status status--" + k; status.textContent = t; };

      function render() {
        if (mode === "fwd") renderFwd(); else if (mode === "train") renderTrain(); else renderGen();
      }

      /* ① 正向 */
      function renderFwd() {
        const g = plane.begin(); frame(g);
        planeCap.textContent = "400 個點（顏色＝原本屬於哪一團）；點一個點，看它一路被加雜訊的軌跡";
        planeKey.innerHTML = `<span>虛線圓＝標準常態分布的 1σ、2σ</span>`;
        g.save(); g.beginPath(); g.rect(26, 8, 328, 328); g.clip();
        for (let i = 0; i < 400; i++) dot(g, xt(i, tF), hue(LAB[i], 0.85));
        // 選中的點：t = 0 到現在的軌跡
        g.strokeStyle = ctx.color("--ink"); g.lineWidth = 1.6; g.beginPath();
        for (let t = 0; t <= tF; t++) { const p = xt(sel, t); t ? g.lineTo(PX(p[0]), PY(p[1])) : g.moveTo(PX(p[0]), PY(p[1])); } g.stroke();
        const p = xt(sel, tF); g.strokeStyle = ctx.color("--ink"); g.lineWidth = 2; g.setLineDash([3, 2]); g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 8, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        g.restore();
        // 旁圖：保留的原資料 vs 雜訊
        sideCap.innerHTML = `x_t ＝ 原資料 × <span style="color:var(--accent)">━ √ᾱ_t</span> ＋ 雜訊 × <span style="color:var(--hot)">━ √(1 − ᾱ_t)</span>（兩個係數的平方加起來永遠是 1）`;
        nn.lineChart(side, { W: 360, H: 200, xMax: T, yMin: 0, yMax: 1, xLabel: "t", marker: tF,
          series: [{ pts: AB.map((a, t) => [t, Math.sqrt(a)]), color: "var(--accent)" }, { pts: AB.map((a, t) => [t, Math.sqrt(1 - a)]), color: "var(--hot)" }] });
        const a = Math.sqrt(AB[tF]), b = Math.sqrt(1 - AB[tF]);
        if (tF === 0) setStatus("info", "t = 0：原本的資料，8 團清清楚楚");
        else setStatus(tF === T ? "ok" : "info", `t = ${tF}：x_t ＝ 原資料 × ${fmt(a, 3)} ＋ 雜訊 × ${fmt(b, 3)}` + (tF === T ? "；原資料只剩 ${fmt(a, 3)} 倍，8 團已經看不出來，幾乎就是標準常態分布的純雜訊" : a < 0.5 ? "；雜訊的係數比較大，各團混在一起" : ""));
        const x0 = DATA[sel], e = EPS[sel];
        calc.innerHTML = tF === 0 ? row("t = 0", `還沒加雜訊：x₀ = ${pr(x0)}`) :
          row(`β_${tF}`, `${fmt(BETA[tF - 1], 4)}（第 ${tF} 步新加的雜訊量，從 0.001 線性增加到 0.2）`) +
          row(`ᾱ_${tF}`, `(1 − β₁)(1 − β₂)…(1 − β_${tF}) = ${fmt(AB[tF], 4)}`) +
          row("一步算到第 t 步", `x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε = ${fmt(a, 3)} × ${pr(x0)} + ${fmt(b, 3)} × ${pr(e)} = <span class="hl">${pr(xt(sel, tF))}</span>`);
      }

      /* ② 訓練 */
      function renderTrain() {
        const g = plane.begin(); frame(g);
        planeCap.textContent = `t = ${tV} 時加了雜訊的點；箭頭＝網路認為「原本的資料在哪個方向」（−ε̂）`;
        planeKey.innerHTML = `<span><i style="border-color:var(--ink)"></i>真正加進去的雜訊 ε</span><span><i style="border-color:var(--hot)"></i>網路猜的雜訊 ε̂</span>`;
        for (let i = 0; i < 400; i++) dot(g, xt(i, tV), hue(LAB[i], 0.35), 2);
        const muted = ctx.color("--accent");
        for (let a = -3; a <= 3; a += 0.5) for (let b = -3; b <= 3; b += 0.5) {
          const e = predict([a, b], tV), m = Math.hypot(e[0], e[1]) || 1, len = Math.min(0.42, 0.2 * m) / m;
          arrow(g, [a, b], [-e[0] * len, -e[1] * len], muted, 1.2);
        }
        const p = xt(sel, tV), e = EPS[sel], eh = predict(p, tV), sc = 0.5;
        arrow(g, p, [e[0] * sc, e[1] * sc], ctx.color("--ink"), 2.2, [4, 3]);
        arrow(g, p, [eh[0] * sc, eh[1] * sc], ctx.color("--hot"), 2.6);
        g.strokeStyle = ctx.color("--ink"); g.lineWidth = 2; g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 7, 0, Math.PI * 2); g.stroke();
        sideCap.textContent = "訓練損失 |ε̂ − ε|²（每 50 步平均）";
        nn.lineChart(side, { W: 360, H: 200, xMax: STEPS, yMin: 0, yMax: 2.5, xLabel: "訓練步數", series: [{ pts: hist, color: "var(--accent)" }] });
        const last = hist.length ? hist[hist.length - 1][1] : null;
        setStatus(loop.finished ? "ok" : "info", `訓練第 ${step} / ${STEPS} 步${loop.running ? "（訓練中）" : ""}` + (last != null ? `：損失 ${fmt(last, 3)}` : "：還沒訓練，箭頭是亂的。按「▶ 開始訓練」開始（約 6 秒）") + (loop.finished ? (tV <= 10 ? "。雜訊小（t 小）時，箭頭把點推回 8 團所在的圓環" : tV >= 25 ? "。雜訊大（t 大）時，看不出點原本在哪，網路只能把點大致推向中間" : "。把 t 調小或調大，比較箭頭的變化") : ""));
        const err = (eh[0] - e[0]) ** 2 + (eh[1] - e[1]) ** 2;
        calc.innerHTML =
          row("出題", `原本的點 x₀ = ${pr(DATA[sel])}，抽雜訊 ε = ${pr(e)}，t = ${tV} → x_t = ${pr(p)}`) +
          row("網路", `只看 x_t = ${pr(p)} 和 t = ${tV} → 猜 ε̂ = <span class="hl">${pr(eh)}</span>`) +
          row("損失", `|ε̂ − ε|² = (${fmt(eh[0], 2)} − ${fmt(e[0], 2)})² + (${fmt(eh[1], 2)} − ${fmt(e[1], 2)})² = <span class="hl">${fmt(err, 3)}</span>`) +
          row("為什麼不會到 0", "同一個 x_t 可能來自不同的 x₀ 加上不同的雜訊，網路只能猜「平均而言」最可能的雜訊");
      }

      /* ③ 生成 */
      function renderGen() {
        const G = genAll(), X = G.P[kG], t = T - kG, g = plane.begin(); frame(g);
        planeCap.textContent = "淡藍＝真資料；橘色方塊＝生成中的點；線＝其中 12 個點走過的路";
        planeKey.innerHTML = `<span>點一個橘色方塊，看它這一步的計算</span>`;
        const blue = Atlas.rgb(ctx.color("--pos"));
        g.fillStyle = `rgba(${blue.join(",")},0.3)`; DATA.forEach((p) => { g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 2.2, 0, Math.PI * 2); g.fill(); });
        g.strokeStyle = ctx.color("--ink"); g.lineWidth = 1.1; g.globalAlpha = 0.45;
        for (let i = 0; i < 400; i += 34) { g.beginPath(); for (let k = 0; k <= kG; k++) { const p = G.P[k][i]; k ? g.lineTo(PX(p[0]), PY(p[1])) : g.moveTo(PX(p[0]), PY(p[1])); } g.stroke(); }
        g.globalAlpha = 1;
        g.save(); g.beginPath(); g.rect(26, 8, 328, 328); g.clip();
        g.fillStyle = ctx.color("--hot"); g.strokeStyle = ctx.color("--ink"); g.lineWidth = 0.7;
        X.forEach((p) => { g.fillRect(PX(p[0]) - 2.4, PY(p[1]) - 2.4, 4.8, 4.8); g.strokeRect(PX(p[0]) - 2.4, PY(p[1]) - 2.4, 4.8, 4.8); });
        const ps = X[selG]; g.strokeStyle = ctx.color("--ink"); g.lineWidth = 2; g.setLineDash([3, 2]); g.beginPath(); g.arc(PX(ps[0]), PY(ps[1]), 8, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        g.restore();
        // 旁圖：8 個方向
        const cnt = Array(8).fill(0), near = Array(8).fill(0);
        X.forEach((p) => { const q = region(p); cnt[q.k]++; if (q.near) near[q.k]++; });
        sideCap.textContent = "400 個點落在哪個方向（深色＝落在真資料附近；虛線＝平均分配的 50 個）";
        bars(cnt, near);
        const nearAll = (near.reduce((a, b) => a + b, 0) / 4), cover = cnt.filter((n) => n >= 20).length;
        if (step === 0 && !loop.running) setStatus("warn", "網路還沒訓練，生成不出東西。先按「▶ 開始訓練」（也可以切到「② 訓練」看它學的過程）");
        else if (!loop.finished) setStatus("warn", `網路只訓練了 ${step} / ${STEPS} 步，生成結果還不可靠（可以按「快轉到結束」）`);
        else if (kG === 0) setStatus("info", "第 0 步：從標準常態分布抽 400 個點，純雜訊。按 ▶ 一步步去雜訊");
        else if (kG < T) setStatus("info", `第 ${kG} 步（t 從 ${t + 1} 到 ${t}）：${fmt(nearAll, 1)}% 的點已經落在真資料附近`);
        else setStatus(cover === 8 ? "ok" : "warn", `40 步完成：${fmt(nearAll, 1)}% 的點落在真資料附近，8 個方向中 ${cover} 個有 5% 以上的點` + (cover === 8 ? "：8 團都生成得出來" : ""));
        if (kG === T) { calc.innerHTML = row("完成", `這個點最後落在 ${pr(ps)}`) + row("每一步做的事", "x_{t−1} = ( x_t − β_t ÷ √(1 − ᾱ_t) × ε̂ ) ÷ √(1 − β_t) + σ_t × z"); return; }
        const tt = T - kG, b = BETA[tt - 1], ab = AB[tt], sg = tt > 1 ? Math.sqrt(b * (1 - AB[tt - 1]) / (1 - ab)) : 0, eh = G.E[kG][selG], z = G.Z[kG][selG], nx = G.P[kG + 1][selG];
        calc.innerHTML =
          row(`下一步（t = ${tt} → ${tt - 1}）`, `網路看 x_t = ${pr(ps)}、t = ${tt}，猜雜訊 ε̂ = ${pr(eh)}`) +
          row("減掉雜訊", `( x_t − ${fmt(b, 3)} ÷ √(1 − ${fmt(ab, 4)}) × ε̂ ) ÷ √(1 − ${fmt(b, 3)}) = ${pr([0, 1].map((d) => (ps[d] - (b / Math.sqrt(1 - ab)) * eh[d]) / Math.sqrt(1 - b)))}`) +
          row("再加一點新雜訊", tt > 1 ? `+ σ_t × z = ${fmt(sg, 3)} × ${pr(z)} → x_${tt - 1} = <span class="hl">${pr(nx)}</span>` : `最後一步不加雜訊（σ₁ = 0）→ x₀ = <span class="hl">${pr(nx)}</span>`);
      }
      function bars(cnt, near) {
        side.setAttribute("viewBox", "0 0 360 170"); side.replaceChildren();
        const names = ["右", "右上", "上", "左上", "左", "左下", "下", "右下"], W = 360, base = 136, top = 18, bw = 30, gap = (W - 30 - 8 * bw) / 7, mx = Math.max(100, ...cnt);
        side.append(s("line", { x1: 15, x2: W - 15, y1: base, y2: base, stroke: "var(--line-strong)" }));
        const yA = base - (50 / mx) * (base - top);
        side.append(s("line", { x1: 15, x2: W - 15, y1: yA, y2: yA, stroke: "var(--ink)", "stroke-dasharray": "4 3", "stroke-opacity": 0.6 }));
        cnt.forEach((n, k) => {
          const x = 15 + k * (bw + gap), hA = (n / mx) * (base - top), hN = (near[k] / mx) * (base - top);
          side.append(s("rect", { x, y: base - hA, width: bw, height: hA, fill: "var(--hot-soft)", stroke: "var(--hot)" }));
          side.append(s("rect", { x, y: base - hN, width: bw, height: hN, fill: "var(--hot)" }));
          side.append(s("text", { x: x + bw / 2, y: base - hA - 4, "text-anchor": "middle", "font-size": 10, class: "t-mono t-halo", text: n }));
          side.append(s("text", { x: x + bw / 2, y: base + 14, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: names[k] }));
        });
        side.append(s("text", { x: W / 2, y: 166, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: "方向（從原點看）" }));
      }

      plane.el.addEventListener("click", (e) => {
        const [x, y] = plane.toLocal(e);
        const pts = mode === "gen" ? genAll().P[kG] : DATA.map((_, i) => xt(i, mode === "fwd" ? tF : tV));
        let best = -1, bd = 12; pts.forEach((p, i) => { const d = Math.hypot(PX(p[0]) - x, PY(p[1]) - y); if (d < bd) { bd = d; best = i; } });
        if (best < 0) return;
        if (mode === "gen") selG = best; else sel = best;
        render();
      });

      note.innerHTML = `其實可以一步一步加雜訊（每步 x_t = √(1 − β_t)·x_{t−1} + √β_t·新的雜訊），但連乘起來剛好等於「一步算到第 t 步」的公式，所以訓練時隨便挑一個 t 直接出題就好。網路猜的是雜訊而不是原圖，這是 DDPM（<span class="en">Denoising Diffusion Probabilistic Models</span>，2020）的做法；猜雜訊和「找出往原資料的方向」其實是同一件事（ε̂ 反過來就是右圖的箭頭）。生成時每步除了減掉雜訊，還會加回一點小雜訊（σ_t × z），讓結果有變化、不會全擠到同一點。真正的影像生成把平面上的 2 個數換成整張圖的所有像素，網路換成擅長影像的大型網路，並可以加上文字等條件；Stable Diffusion 先用 <a href="#vae">VAE</a> 把影像壓到較小的潛在空間，在那裡做擴散，最後再解碼回影像。和 <a href="#gan">GAN</a> 的「八團」比較：GAN 有時會漏掉幾團（モード崩壊），這裡每一團都生成得出來。`;

      ctx.onTheme(render);
      makeNet(); layout(); refreshBtns();
      return { destroy() { loop.pause(); pF.pause(); pG.pause(); } };
    },
  });
})();
