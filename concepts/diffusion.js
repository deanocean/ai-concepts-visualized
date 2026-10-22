/* Diffusion Model（シラバス §30）
   真資料：半徑 2 圓周上 8 團，共 400 個點（和 GAN 頁的「八團」同一種分布，固定種子）。
   正向：T = 40 步，β_t 從 0.001 線性增加到 0.2；ᾱ_t = Π(1 − β_s)；x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε，可一步算到任一個 t。
   網路 ε̂(x_t, t)：輸入 x_t（2）＋ t 的 5 個特徴（t/T、sin・cos）→ 32 → 32 → 2（ReLU），損失 |ε̂ − ε|²。
   頁面上即時訓練：Adam η=0.005（最後 30% 降為 0.0015）、每步 64 筆、3000 步。
   逆向（DDPM）：x_{t−1} = (x_t − β_t/√(1 − ᾱ_t)·ε̂)/√(1 − β_t) + σ_t·z，σ_t² = β_t(1 − ᾱ_{t−1})/(1 − ᾱ_t)。
   文字在 locales/zh/diffusion.js、locales/ja/diffusion.js。 */
(function () {
  const T = 40, STEPS = 3000, R2 = 2;
  const BETA = Array.from({ length: T }, (_, i) => 0.001 + (0.2 - 0.001) * i / (T - 1)); // BETA[t−1] = β_t
  const AB = [1]; for (let t = 1; t <= T; t++) AB.push(AB[t - 1] * (1 - BETA[t - 1]));      // AB[t] = ᾱ_t
  const temb = (t) => { const u = t / T; return [u, Math.sin(Math.PI * u), Math.cos(Math.PI * u), Math.sin(2 * Math.PI * u), Math.cos(2 * Math.PI * u)]; };

  Atlas.define("diffusion", {

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { MLP, rng, gauss } = nn;
      const { t } = ctx;
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
      const xt = (i, ts) => { const a = Math.sqrt(AB[ts]), b = Math.sqrt(1 - AB[ts]); return [a * DATA[i][0] + b * EPS[i][0], a * DATA[i][1] + b * EPS[i][1]]; };

      /* ---------- 網路與訓練（三個模式共用；打開頁面就開始） ---------- */
      let net, r, hist, step;
      const inp = new Float64Array(7);
      const predict = (x, ts, N = net) => { inp[0] = x[0]; inp[1] = x[1]; const e = temb(ts); for (let i = 0; i < 5; i++) inp[2 + i] = e[i]; const o = N.forward(inp); return [o[0], o[1]]; };
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
                const i = Math.floor(r() * 400), ts = 1 + Math.floor(r() * T), a = Math.sqrt(AB[ts]), c = Math.sqrt(1 - AB[ts]), e0 = gauss(r), e1 = gauss(r);
                const o = predict([a * DATA[i][0] + c * e0, a * DATA[i][1] + c * e1], ts);
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
      const modeSeg = ui.seg({ label: t("modeLabel"), value: mode, options: [{ value: "fwd", label: t("modeFwd") }, { value: "train", label: t("modeTrain") }, { value: "gen", label: t("modeGen") }], onChange: (v) => { mode = v; layout(); } });
      const runBtn = ui.btn(t("btnRun"), () => { loop.running ? loop.pause() : loop.start(); refreshBtns(); }, { primary: true });
      const ffBtn = ui.btn(t("btnFf"), () => { loop.fastForward(); refreshBtns(); });
      const seedBtn = ui.btn(t("btnSeed"), () => { seed = (seed % 5) + 1; makeNet(); render(); refreshBtns(); }, { sm: true });
      const trainRow = h("div", { class: "bench__row" }, runBtn, ffBtn, seedBtn);
      let vRow;
      function refreshBtns() { runBtn.textContent = t(loop.running ? "btnPause" : loop.finished ? "btnDone" : step ? "btnResume" : "btnRun"); runBtn.disabled = ffBtn.disabled = loop.finished; }

      // ① 正向
      const tSl = ui.slider({ id: "df-t", label: t("tLabel"), min: 0, max: T, step: 1, value: tF, format: (v) => v + " / " + T, onInput: (v) => { pF.pause(); tF = v; render(); } });
      const pF = ui.player(ctx, () => { if (tF >= T) return false; tF++; tSl.set(tF); render(); return tF < T; }, 220);
      const fwdRow = h("div", { class: "bench__row" }, tSl.el, ui.btn("◀", () => { pF.pause(); if (tF > 0) { tF--; tSl.set(tF); render(); } }, { aria: t("ariaLess") }), ui.btn("▶", () => { pF.pause(); if (tF < T) { tF++; tSl.set(tF); render(); } }, { aria: t("ariaMore") }), pF.el, ui.btn(t("btnT0"), () => { pF.pause(); tF = 0; tSl.set(0); render(); }));
      // ② 訓練：查看的 t
      const vSl = ui.slider({ id: "df-tv", label: t("tvLabel"), min: 1, max: T, step: 1, value: tV, format: (v) => v + " / " + T, onInput: (v) => { tV = v; render(); } });
      vRow = h("div", { class: "bench__row" }, vSl.el);
      // ③ 生成
      const gSl = ui.slider({ id: "df-k", label: t("kLabel"), min: 0, max: T, step: 1, value: kG, format: (v) => v + " / " + T, onInput: (v) => { pG.pause(); kG = v; render(); } });
      const pG = ui.player(ctx, () => { if (kG >= T) return false; kG++; gSl.set(kG); render(); return kG < T; }, 220);
      const genRow = h("div", { class: "bench__row" }, gSl.el, ui.btn("◀", () => { pG.pause(); if (kG > 0) { kG--; gSl.set(kG); render(); } }, { aria: t("ariaPrev") }), ui.btn("▶", () => { pG.pause(); if (kG < T) { kG++; gSl.set(kG); render(); } }, { aria: t("ariaNext") }), pG.el,
        ui.btn(t("btnNoise"), () => { pG.pause(); genSeed++; genCache = null; kG = 0; gSl.set(0); render(); }, { sm: true }));

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const plane = nn.canvas(360, 360, t("planeAria")); plane.el.classList.add("df-hit");
      const planeCap = h("div", { class: "df-cap" }), planeKey = h("div", { class: "df-key" });
      const side = s("svg", { role: "img", "aria-label": t("sideAria") }), sideCap = h("div", { class: "df-cap" });
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
        for (let ts = T; ts >= 1; ts--) {
          const b = BETA[ts - 1], ab = AB[ts], sg = ts > 1 ? Math.sqrt(b * (1 - AB[ts - 1]) / (1 - ab)) : 0;
          const eh = X.map((x) => predict(x, ts)), zz = X.map(() => (ts > 1 ? [gauss(q), gauss(q)] : [0, 0]));
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
      const setStatus = (k, txt) => { status.className = "status status--" + k; status.textContent = txt; };

      function render() {
        if (mode === "fwd") renderFwd(); else if (mode === "train") renderTrain(); else renderGen();
      }

      /* ① 正向 */
      function renderFwd() {
        const g = plane.begin(); frame(g);
        planeCap.textContent = t("capFwd");
        planeKey.innerHTML = t("keyFwd");
        g.save(); g.beginPath(); g.rect(26, 8, 328, 328); g.clip();
        for (let i = 0; i < 400; i++) dot(g, xt(i, tF), hue(LAB[i], 0.85));
        // 選中的點：t = 0 到現在的軌跡
        g.strokeStyle = ctx.color("--ink"); g.lineWidth = 1.6; g.beginPath();
        for (let j = 0; j <= tF; j++) { const p = xt(sel, j); j ? g.lineTo(PX(p[0]), PY(p[1])) : g.moveTo(PX(p[0]), PY(p[1])); } g.stroke();
        const p = xt(sel, tF); g.strokeStyle = ctx.color("--ink"); g.lineWidth = 2; g.setLineDash([3, 2]); g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 8, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        g.restore();
        // 旁圖：保留的原資料 vs 雜訊
        sideCap.innerHTML = t("sideFwd");
        nn.lineChart(side, { W: 360, H: 200, xMax: T, yMin: 0, yMax: 1, xLabel: "t", marker: tF,
          series: [{ pts: AB.map((a, j) => [j, Math.sqrt(a)]), color: "var(--accent)" }, { pts: AB.map((a, j) => [j, Math.sqrt(1 - a)]), color: "var(--hot)" }] });
        const a = Math.sqrt(AB[tF]), b = Math.sqrt(1 - AB[tF]);
        if (tF === 0) setStatus("info", t("fwdT0"));
        else setStatus(tF === T ? "ok" : "info", t(tF === T ? "fwdTEnd" : a < 0.5 ? "fwdTMix" : "fwdT", { t: tF, a: fmt(a, 3), b: fmt(b, 3) }));
        const x0 = DATA[sel], e = EPS[sel];
        calc.innerHTML = tF === 0 ? row("t = 0", t("calcT0", { x: pr(x0) })) :
          row(`β_${tF}`, t("calcBeta", { b: fmt(BETA[tF - 1], 4), t: tF })) +
          row(`ᾱ_${tF}`, `(1 − β₁)(1 − β₂)…(1 − β_${tF}) = ${fmt(AB[tF], 4)}`) +
          row(t("calcOneK"), `x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε = ${fmt(a, 3)} × ${pr(x0)} + ${fmt(b, 3)} × ${pr(e)} = <span class="hl">${pr(xt(sel, tF))}</span>`);
      }

      /* ② 訓練 */
      function renderTrain() {
        const g = plane.begin(); frame(g);
        planeCap.textContent = t("capTrain", { t: tV });
        planeKey.innerHTML = t("keyTrain");
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
        sideCap.textContent = t("sideTrain");
        nn.lineChart(side, { W: 360, H: 200, xMax: STEPS, yMin: 0, yMax: 2.5, xLabel: t("trainAxis"), series: [{ pts: hist, color: "var(--accent)" }] });
        const last = hist.length ? hist[hist.length - 1][1] : null;
        setStatus(loop.finished ? "ok" : "info", t(loop.running ? "trainHeadRun" : "trainHead", { s: step, n: STEPS }) + (last != null ? t("trainLoss", { loss: fmt(last, 3) }) : t("trainNone")) + (loop.finished ? t(tV <= 10 ? "trainSmall" : tV >= 25 ? "trainLarge" : "trainMid") : ""));
        const err = (eh[0] - e[0]) ** 2 + (eh[1] - e[1]) ** 2;
        calc.innerHTML =
          row(t("calcQK"), t("calcQ", { x0: pr(DATA[sel]), e: pr(e), t: tV, xt: pr(p) })) +
          row(t("calcNetK"), t("calcNet", { xt: pr(p), t: tV, eh: pr(eh) })) +
          row(t("calcLossK"), `|ε̂ − ε|² = (${fmt(eh[0], 2)} − ${fmt(e[0], 2)})² + (${fmt(eh[1], 2)} − ${fmt(e[1], 2)})² = <span class="hl">${fmt(err, 3)}</span>`) +
          row(t("calcWhyK"), t("calcWhy"));
      }

      /* ③ 生成 */
      function renderGen() {
        const G = genAll(), X = G.P[kG], tc = T - kG, g = plane.begin(); frame(g);
        planeCap.textContent = t("capGen");
        planeKey.innerHTML = t("keyGen");
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
        sideCap.textContent = t("sideGen");
        bars(cnt, near);
        const nearAll = (near.reduce((a, b) => a + b, 0) / 4), cover = cnt.filter((n) => n >= 20).length;
        if (step === 0 && !loop.running) setStatus("warn", t("genUntrained"));
        else if (!loop.finished) setStatus("warn", t("genPartial", { s: step, n: STEPS }));
        else if (kG === 0) setStatus("info", t("genK0"));
        else if (kG < T) setStatus("info", t("genK", { k: kG, t1: tc + 1, t: tc, near: fmt(nearAll, 1) }));
        else setStatus(cover === 8 ? "ok" : "warn", t(cover === 8 ? "genDoneAll" : "genDone", { near: fmt(nearAll, 1), cover }));
        if (kG === T) { calc.innerHTML = row(t("calcDoneK"), t("calcDone", { p: pr(ps) })) + row(t("calcEachK"), "x_{t−1} = ( x_t − β_t ÷ √(1 − ᾱ_t) × ε̂ ) ÷ √(1 − β_t) + σ_t × z"); return; }
        const tt = T - kG, b = BETA[tt - 1], ab = AB[tt], sg = tt > 1 ? Math.sqrt(b * (1 - AB[tt - 1]) / (1 - ab)) : 0, eh = G.E[kG][selG], z = G.Z[kG][selG], nx = G.P[kG + 1][selG];
        calc.innerHTML =
          row(t("calcNextK", { t: tt, t2: tt - 1 }), t("calcNext", { p: pr(ps), t: tt, eh: pr(eh) })) +
          row(t("calcSubK"), `( x_t − ${fmt(b, 3)} ÷ √(1 − ${fmt(ab, 4)}) × ε̂ ) ÷ √(1 − ${fmt(b, 3)}) = ${pr([0, 1].map((d) => (ps[d] - (b / Math.sqrt(1 - ab)) * eh[d]) / Math.sqrt(1 - b)))}`) +
          row(t("calcAddK"), tt > 1 ? `+ σ_t × z = ${fmt(sg, 3)} × ${pr(z)} → x_${tt - 1} = <span class="hl">${pr(nx)}</span>` : t("calcAddLast", { x: pr(nx) }));
      }
      function bars(cnt, near) {
        side.setAttribute("viewBox", "0 0 360 170"); side.replaceChildren();
        const names = ["dirR", "dirUR", "dirU", "dirUL", "dirL", "dirDL", "dirD", "dirDR"].map((k) => t(k)), W = 360, base = 136, top = 18, bw = 30, gap = (W - 30 - 8 * bw) / 7, mx = Math.max(100, ...cnt);
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
        side.append(s("text", { x: W / 2, y: 166, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: t("covAxis") }));
      }

      plane.el.addEventListener("click", (e) => {
        const [x, y] = plane.toLocal(e);
        const pts = mode === "gen" ? genAll().P[kG] : DATA.map((_, i) => xt(i, mode === "fwd" ? tF : tV));
        let best = -1, bd = 12; pts.forEach((p, i) => { const d = Math.hypot(PX(p[0]) - x, PY(p[1]) - y); if (d < bd) { bd = d; best = i; } });
        if (best < 0) return;
        if (mode === "gen") selG = best; else sel = best;
        render();
      });

      note.innerHTML = t("note");

      ctx.onTheme(render);
      makeNet(); layout(); refreshBtns();
      return { destroy() { loop.pause(); pF.pause(); pG.pause(); } };
    },
  });
})();
