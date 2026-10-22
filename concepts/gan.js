/* 敵対的生成ネットワーク（GAN）（シラバス §30）
   真資料：平面上的點。「環狀」＝半徑 2 的圓（雜訊 σ=0.1）；「八團」＝半徑 2 圓周上 8 團（σ=0.12）。每批都重新抽。
   生成器 G：雜訊 z（2 維標準常態）→ 16 → 16 → 2（Leaky ReLU）；識別器 D：2 → 16 → 16 → 1（Leaky ReLU，輸出 Sigmoid）。
   一輪：先更新 D 一次（32 個真 + 32 個假），再更新 G（標準 1 次；失敗示範 10 次）。Adam η=0.002、β₁=0.5。
   G 的損失用 −ln D(G(z))。標準 3000 輪、失敗示範 300 輪（兩者 G 都更新 3000 次）。
   畫面：底色＝D 判斷「是真資料」的機率；橘色方塊＝300 個固定 z 生成的點；箭頭＝生成器被推的方向（∂ ln D / ∂x）。
   文字在 locales/zh/gan.js、locales/ja/gan.js。 */
(function () {
  const R2 = 2;
  Atlas.define("gan", {

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { MLP, rng, gauss } = nn;
      const { t } = ctx;
      let data = "circle", sched = "std", seed = 1, follow = true, viewR = 0, sel = 0, curTotal = 3000;
      const runs = {};

      bench.append(h("style", { text: `
        .gn-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; align-items: flex-start; }
        .gn-fig { flex: 1 1 300px; max-width: 420px; min-width: 0; }
        .gn-fig > svg { display: block; width: 100%; height: auto; }
        .gn-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .gn-hit { cursor: pointer; touch-action: manipulation; }
        .gn-key { display: flex; flex-wrap: wrap; gap: 4px 14px; justify-content: center; font-size: 12.5px; color: var(--muted); margin-top: 4px; }
        .gn-key i { display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: -1px; }
      ` }));

      /* ---------- 真資料 ---------- */
      const sampler = (kind, r) => kind === "circle"
        ? () => { const a = 2 * Math.PI * r(); return [R2 * Math.cos(a) + 0.1 * gauss(r), R2 * Math.sin(a) + 0.1 * gauss(r)]; }
        : () => { const a = Math.floor(r() * 8) * Math.PI / 4; return [R2 * Math.cos(a) + 0.12 * gauss(r), R2 * Math.sin(a) + 0.12 * gauss(r)]; };
      const SHOW = { circle: (() => { const f = sampler("circle", rng(31)); return Array.from({ length: 300 }, f); })(), ring8: (() => { const f = sampler("ring8", rng(32)); return Array.from({ length: 300 }, f); })() };
      const ZS = (() => { const r = rng(41); return Array.from({ length: 300 }, () => [gauss(r), gauss(r)]); })();  // 畫面上固定的 300 個 z
      const ZC = (() => { const r = rng(42); return Array.from({ length: 1000 }, () => [gauss(r), gauss(r)]); })(); // 統計用 1000 個 z
      /** 一個點屬於 8 區中的哪一區，以及是否落在真資料附近 */
      function region(p, kind) {
        const a = Math.atan2(p[1], p[0]), k = ((Math.round(a / (Math.PI / 4)) % 8) + 8) % 8;
        const near = kind === "circle" ? Math.abs(Math.hypot(p[0], p[1]) - R2) < 0.3 : Math.hypot(p[0] - R2 * Math.cos(k * Math.PI / 4), p[1] - R2 * Math.sin(k * Math.PI / 4)) < 0.4;
        return { k, near };
      }

      /* ---------- 控制 ---------- */
      const dSeg = ui.seg({ label: t("dLabel"), value: data, options: [{ value: "circle", label: t("dCircle") }, { value: "ring8", label: t("dRing8") }], onChange: (v) => { data = v; ensure(); } });
      const sSeg = ui.seg({ label: t("sLabel"), value: sched, options: [{ value: "std", label: t("sStd") }, { value: "g10", label: t("sG10") }], onChange: (v) => { sched = v; ensure(); } });
      const seedBtn = ui.btn(t("btnSeed"), () => { cur().loop.pause(); seed = (seed % 5) + 1; ensure(); }, { sm: true });
      const runBtn = ui.btn(t("btnRun"), () => { const R = cur(); R.loop.running ? R.loop.pause() : (follow = true, R.loop.start()); refreshBtns(); }, { primary: true });
      const oneBtn = ui.btn(t("btnOne"), () => { const R = cur(); R.loop.pause(); follow = true; R.step1(); show(); refreshBtns(); });
      const ffBtn = ui.btn(t("btnFf"), () => { follow = true; cur().loop.fastForward(); refreshBtns(); });
      const prog = ui.slider({ id: "gn-r", label: t("progLabel"), min: 0, max: 3000, step: 1, value: 0, format: (v) => v + " / " + curTotal,
        onInput: (v) => { const R = cur(); const sn = R.snaps.filter((q) => q.r <= v).pop(); const last = R.snaps[R.snaps.length - 1]; follow = sn === last; viewR = sn.r; show(); } });

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const plane = nn.canvas(360, 360, t("planeAria")); plane.el.classList.add("gn-hit");
      const dSvg = s("svg", { role: "img", "aria-label": t("dAria") }), dCap = h("div", { class: "gn-cap" });
      const cSvg = s("svg", { viewBox: "0 0 360 170", role: "img", "aria-label": t("covAria") });
      bench.append(
        h("div", { class: "bench__row" }, dSeg.el, sSeg.el, seedBtn),
        h("div", { class: "bench__row" }, runBtn, oneBtn, ffBtn, prog.el),
        status,
        h("div", { class: "gn-row" },
          h("div", { class: "gn-fig", style: { flexBasis: "340px" } },
            h("div", { class: "gn-cap" }, t("capPlane")), plane.el,
            h("div", { class: "gn-key", html: t("key") })),
          h("div", { class: "gn-fig" },
            dCap, dSvg,
            h("div", { class: "gn-cap", style: { marginTop: "10px" } }, t("capCov")), cSvg)),
        calc, note);

      /* ---------- 訓練 ---------- */
      function makeRun(kind, sc, sd) {
        const r = rng(sd * 3001 + (kind === "circle" ? 1 : 2) * 17 + (sc === "std" ? 0 : 5));
        const G = new MLP([2, 16, 16, 2], ["lrelu", "lrelu", "lin"], r), D = new MLP([2, 16, 16, 1], ["lrelu", "lrelu", "sigmoid"], r);
        const kG = sc === "std" ? 1 : 10, total = sc === "std" ? 3000 : 300, SN = sc === "std" ? 50 : 5, BS = 32, LR = 0.002;
        const real = sampler(kind, r), f = new Float64Array(2);
        const R = { kind, sc, total, G, D, vG: new MLP([2, 16, 16, 2], ["lrelu", "lrelu", "lin"], rng(0)), vD: new MLP([2, 16, 16, 1], ["lrelu", "lrelu", "sigmoid"], rng(0)), round: 0, hist: [], snaps: [] };
        const snap = () => R.snaps.push({ r: R.round, G: G.snapshot(), D: D.snapshot() });
        snap();
        R.step1 = () => {
          if (R.round >= total) return;
          let sr = 0, sf = 0;
          for (let b = 0; b < BS; b++) {   // 判別器：真的判 1、假的判 0
            const p = D.forward(real())[0]; sr += p; D.backward([p - 1], true);
            f.set(G.forward([gauss(r), gauss(r)])); const q = D.forward(f)[0]; sf += q; D.backward([q], true);
          }
          D.adam(LR, 1 / BS, 0.5);
          for (let k = 0; k < kG; k++) {    // 生成器：讓識別器把假的判成 1（D 的權重不動）
            for (let b = 0; b < BS; b++) { f.set(G.forward([gauss(r), gauss(r)])); const q = D.forward(f)[0]; G.backward(D.backward([q - 1], true, false)); }
            G.adam(LR, 1 / BS, 0.5);
          }
          R.round++; R.hist.push([R.round, sr / BS, sf / BS]);
          if (R.round % SN === 0) snap();
        };
        R.loop = nn.loop(ctx, { total, perTick: sc === "std" ? 12 : 1,
          run(from, n) { for (let i = 0; i < n; i++) R.step1(); },
          onTick() { if (R === cur()) { if (follow) show(); else { drawD(); drawStatus(); } refreshBtns(); } } });
        return R;
      }
      const key = () => data + "-" + sched + "-" + seed;
      const cur = () => runs[key()];
      function ensure() {
        Object.values(runs).forEach((R) => R !== runs[key()] && R.loop.pause());
        if (!runs[key()]) runs[key()] = makeRun(data, sched, seed);
        const R = cur(); follow = true;
        prog.el.querySelector("input").max = R.total; curTotal = R.total;
        show(); refreshBtns();
      }
      function refreshBtns() {
        const R = cur();
        runBtn.textContent = t(R.loop.running ? "btnPause" : R.loop.finished ? "btnDone" : R.round ? "btnResume" : "btnRun");
        runBtn.disabled = ffBtn.disabled = oneBtn.disabled = R.loop.finished;
      }

      /* ---------- 顯示 ---------- */
      let V = null;
      function computeView() {
        const R = cur();
        // 跟著最新：直接用正在訓練的網路（「前進 1 輪」也看得到）；查看過去：還原存下的權重
        let G = R.G, D = R.D;
        if (follow) viewR = R.round;
        else { const sn = R.snaps.filter((q) => q.r <= viewR).pop(); viewR = sn.r; R.vG.restore(sn.G); R.vD.restore(sn.D); G = R.vG; D = R.vD; }
        const fake = ZS.map((z) => Array.from(G.forward(z)));
        const dReal = SHOW[R.kind].map((p) => D.forward(p)[0]), dFake = fake.map((p) => D.forward(p)[0]);
        const grads = fake.map((p) => { const q = D.forward(p)[0]; return Array.from(D.backward([1 - q], true, false)); }); // ∂ ln D / ∂x
        const cnt = Array(8).fill(0), near = Array(8).fill(0);
        ZC.forEach((z) => { const g = region(G.forward(z), R.kind); cnt[g.k]++; if (g.near) near[g.k]++; });
        const N = 60, grid = new Float64Array(N * N);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) grid[j * N + i] = D.forward([-3 + (i + 0.5) * 6 / N, 3 - (j + 0.5) * 6 / N])[0];
        V = { R, G, D, fake, dReal, dFake, grads, cnt, near, grid, N };
      }
      function show() { computeView(); prog.set(viewR); drawPlane(); drawD(); drawCov(); drawStatus(); drawCalc(); }
      const col = () => ({ panel: Atlas.rgb(ctx.color("--panel")), pos: Atlas.rgb(ctx.color("--pos")), hot: Atlas.rgb(ctx.color("--hot")), posC: ctx.color("--pos"), hotC: ctx.color("--hot"), ink: ctx.color("--ink"), line: ctx.color("--line"), muted: ctx.color("--muted"), panelC: ctx.color("--panel") });
      const font = (px, w = 400) => `${w} ${px}px ${ctx.color("--f-body") || "sans-serif"}`;
      const PX = (x) => 30 + ((x + 3) / 6) * 320, PY = (y) => 10 + ((3 - y) / 6) * 320;

      function drawPlane() {
        const c = col(), g = plane.begin(), N = V.N, cs = 320 / N;
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const d = V.grid[j * N + i], tt = Math.min(1, Math.abs(d - 0.5) * 2) * 0.6, tgt = d >= 0.5 ? c.pos : c.hot;
          g.fillStyle = `rgb(${c.panel.map((v, k) => Math.round(v + (tgt[k] - v) * tt)).join(",")})`;
          g.fillRect(30 + i * cs, 10 + j * cs, cs + 0.5, cs + 0.5);
        }
        g.strokeStyle = c.line; g.lineWidth = 1; g.font = font(10); g.fillStyle = c.muted; g.textAlign = "center";
        for (let v = -3; v <= 3; v++) {
          g.globalAlpha = 0.6; g.beginPath(); g.moveTo(PX(v), 10); g.lineTo(PX(v), 330); g.moveTo(30, PY(v)); g.lineTo(350, PY(v)); g.stroke(); g.globalAlpha = 1;
          g.textAlign = "center"; g.fillText(fmt(v, 0), PX(v), 345); g.textAlign = "right"; g.fillText(fmt(v, 0), 25, PY(v) + 3);
        }
        // 真資料
        g.fillStyle = c.posC; g.strokeStyle = c.panelC; g.lineWidth = 0.8;
        SHOW[V.R.kind].forEach((p) => { g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 2.3, 0, Math.PI * 2); g.fill(); g.stroke(); });
        // 假資料與箭頭
        g.save(); g.beginPath(); g.rect(30, 10, 320, 320); g.clip();
        g.fillStyle = c.hotC; g.strokeStyle = c.ink; g.lineWidth = 0.8;
        V.fake.forEach((p) => { g.fillRect(PX(p[0]) - 2.6, PY(p[1]) - 2.6, 5.2, 5.2); g.strokeRect(PX(p[0]) - 2.6, PY(p[1]) - 2.6, 5.2, 5.2); });
        g.strokeStyle = c.ink; g.fillStyle = c.ink; g.lineWidth = 1.3;
        V.fake.forEach((p, i) => { if (i % 10 === 0 || i === sel) arrow(g, p, V.grads[i], i === sel ? 2.2 : 1.3); });
        const p = V.fake[sel];
        g.strokeStyle = c.ink; g.lineWidth = 2; g.setLineDash([3, 2]); g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 9, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
        g.restore();
      }
      function arrow(g, p, gr, w) {
        const m = Math.hypot(gr[0], gr[1]); if (m < 1e-6) return;
        const L = Math.min(0.55, 0.35 * m) / m, x1 = PX(p[0]), y1 = PY(p[1]), x2 = PX(p[0] + gr[0] * L), y2 = PY(p[1] + gr[1] * L);
        const len = Math.hypot(x2 - x1, y2 - y1); if (len < 4) return;
        const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
        g.lineWidth = w; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        g.beginPath(); g.moveTo(x2, y2); g.lineTo(x2 - ux * 6 - uy * 3.5, y2 - uy * 6 + ux * 3.5); g.lineTo(x2 - ux * 6 + uy * 3.5, y2 - uy * 6 - ux * 3.5); g.fill();
      }
      plane.el.addEventListener("click", (e) => {
        const [x, y] = plane.toLocal(e);
        let best = -1, bd = 12;
        V.fake.forEach((p, i) => { const d = Math.hypot(PX(p[0]) - x, PY(p[1]) - y); if (d < bd) { bd = d; best = i; } });
        if (best >= 0) { sel = best; drawPlane(); drawCalc(); }
      });

      function drawD() {
        const R = cur(), W = 20, sm = (j) => { const out = []; for (let i = 0; i + W <= R.hist.length; i += W) { let a = 0; for (let k = i; k < i + W; k++) a += R.hist[k][j]; out.push([R.hist[i + W - 1][0], a / W]); } return out; };
        const smR = R.sc === "std" ? sm(1) : R.hist.map((q) => [q[0], q[1]]), smF = R.sc === "std" ? sm(2) : R.hist.map((q) => [q[0], q[2]]);
        dCap.innerHTML = t(R.sc === "std" ? "dCapStd" : "dCapG10");
        nn.lineChart(dSvg, { W: 360, H: 190, xMax: R.total, yMin: 0, yMax: 1, xLabel: t("dAxis"), marker: viewR, refY: 0.5,
          series: [{ pts: smR, color: "var(--pos)" }, { pts: smF, color: "var(--hot)" }] });
      }
      function drawCov() {
        cSvg.replaceChildren();
        const names = ["dirR", "dirUR", "dirU", "dirUL", "dirL", "dirDL", "dirD", "dirDR"].map((k) => t(k)), W = 360, base = 136, top = 18, bw = 30, gap = (W - 30 - 8 * bw) / 7;
        const mx = Math.max(250, ...V.cnt);
        cSvg.append(s("line", { x1: 15, x2: W - 15, y1: base, y2: base, stroke: "var(--line-strong)" }));
        cSvg.append(s("line", { x1: 15, x2: W - 15, y1: base - (125 / mx) * (base - top), y2: base - (125 / mx) * (base - top), stroke: "var(--ink)", "stroke-dasharray": "4 3", "stroke-opacity": 0.6 }));
        V.cnt.forEach((n, k) => {
          const x = 15 + k * (bw + gap), hAll = (n / mx) * (base - top), hNear = (V.near[k] / mx) * (base - top);
          cSvg.append(s("rect", { x, y: base - hAll, width: bw, height: hAll, fill: "var(--hot-soft)", stroke: "var(--hot)" }));
          cSvg.append(s("rect", { x, y: base - hNear, width: bw, height: hNear, fill: "var(--hot)" }));
          cSvg.append(s("text", { x: x + bw / 2, y: base - hAll - 4, "text-anchor": "middle", "font-size": 10, class: "t-mono", text: n }));
          cSvg.append(s("text", { x: x + bw / 2, y: base + 14, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: names[k] }));
        });
        cSvg.append(s("text", { x: W / 2, y: 166, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: t("covAxis") }));
      }
      const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      function drawStatus() {
        const R = cur(), ar = avg(V.dReal), af = avg(V.dFake), nearAll = V.near.reduce((a, b) => a + b, 0) / 10, cover = V.cnt.filter((n) => n >= 50).length;
        const head = t(R.loop.running ? "statusHeadRun" : "statusHead", { r: viewR, ar: fmt(ar, 2), af: fmt(af, 2), near: fmt(nearAll, 1), cover });
        let kd = "info", tail = "";
        if (viewR === 0) tail = t(R.loop.running ? "tailStartRun" : "tailStart");
        else if (R.sc === "g10" && cover <= 3) { kd = "bad"; tail = t("tailCollapse"); }
        else if (R.sc === "g10" && viewR === R.total && nearAll < 30) { kd = "warn"; tail = t("tailChase"); }
        else if (viewR === R.total && Math.abs(ar - 0.5) < 0.15 && Math.abs(af - 0.5) < 0.15 && cover >= 7) { kd = "ok"; tail = t("tailOk"); }
        else if (viewR === R.total && R.sc === "std") { kd = "warn"; tail = t("tailMiss"); }
        status.className = "status status--" + kd; status.textContent = head + tail;
      }
      function drawCalc() {
        const z = ZS[sel], x = V.fake[sel], q = V.dFake[sel], gr = V.grads[sel], ar = avg(V.dReal);
        const lossD = -(avg(V.dReal.map((p) => Math.log(Math.max(1e-12, p)))) + avg(V.dFake.map((p) => Math.log(Math.max(1e-12, 1 - p)))));
        const row = (kk, b) => `<span class="calc-row"><span class="k">${kk}</span> ${b}</span>`;
        const pr = (a) => "(" + a.map((v) => fmt(v, 2)).join(", ") + ")";
        calc.innerHTML =
          row(t("calcGK"), t("calcG", { z: pr(z), x: pr(x) })) +
          row(t("calcDK"), t("calcD", { q: fmt(q, 3) })) +
          row(t("calcGLossK"), t("calcGLoss", { q: fmt(q, 3), v: fmt(-Math.log(Math.max(1e-12, q)), 3), g: pr(gr) })) +
          row(t("calcDLossK"), t("calcDLoss", { v: fmt(lossD, 3) }));
      }

      note.innerHTML = t("note");

      ctx.onTheme(() => drawPlane());
      ensure();
      return { destroy() { Object.values(runs).forEach((R) => R.loop.pause()); } };
    },
  });
})();
