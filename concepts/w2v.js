/* 単語埋め込み（word2vec）（シラバス §27）
   小語料 20 句、26 個詞。窗口 2（前後各 2 個詞），向量 8 維，完整 softmax（26 類），SGD η = 0.05，200 エポック，每エポック洗牌（固定種子）。
   スキップグラム：用中間的詞猜窗口內每個詞；CBOW：用窗口內詞向量的平均猜中間的詞。詞向量＝輸入側的權重 W_in 的那一列。
   實測種子 1〜5：スキップグラム每次「國王↔王后、貓↔狗、男人↔女人、魚↔肉」都互為最相近；三組類推（國王−男人+女人 等）兩種做法全部答對；
   CBOW 不穩定（有些詞最相近的對象會跑掉，例如魚、肉）。平面圖＝8 維投影到前兩個主成分。
   文字在 locales/zh/w2v.js、locales/ja/w2v.js。語料（corpus）也在語言檔：中文、日文各一份，句子以「/」、詞以空白分隔；
   程式要用到的 9 個詞（國王、王后、男人、女人、貓、狗、魚、肉、水）由 keyWords 依序對應到 king、queen…。
   日文語料（省略助詞的短句，語序是日文的）也實測過種子 1〜5：スキップグラム 4 對全部配成、類推全對，CBOW 一樣不穩定。 */
(function () {
  const D = 8, WIN = 2, LR = 0.05, EPOCHS = 200;
  const GCOL = { p: "var(--pos)", a: "var(--hot)", f: "var(--ok)", o: "var(--muted)" };
  const KEYS = ["king", "queen", "man", "woman", "cat", "dog", "fish", "meat", "water"];
  let SENT, VOC, ID, NV, K, GROUP, ANALOGY, SAMPLES;
  /** 依語言檔的語料建立詞彙表與訓練樣本；K：king 等 9 個 id → 這個語言的詞 */
  function setCorpus(corpus, keyWords) {
    SENT = corpus.split("/").map((l) => l.trim().split(/\s+/));
    VOC = [...new Set(SENT.flat())]; ID = Object.fromEntries(VOC.map((w, i) => [w, i])); NV = VOC.length;
    K = Object.fromEntries(keyWords.split(" ").map((w, i) => [KEYS[i], w]));
    GROUP = { [K.king]: "p", [K.queen]: "p", [K.man]: "p", [K.woman]: "p", [K.cat]: "a", [K.dog]: "a", [K.fish]: "f", [K.meat]: "f", [K.water]: "f" };
    ANALOGY = [[K.king, K.man, K.woman, K.queen], [K.man, K.king, K.queen, K.woman], [K.queen, K.woman, K.man, K.king]]; // 最後一個＝應該得到的答案
    // 訓練樣本：每個位置一筆（中間的詞、窗口內的詞）
    SAMPLES = [];
    SENT.forEach((s, si) => s.forEach((w, i) => { const ctx = []; for (let j = Math.max(0, i - WIN); j <= Math.min(s.length - 1, i + WIN); j++) if (j !== i) ctx.push(ID[s[j]]); SAMPLES.push({ si, i, c: ID[w], ctx }); }));
  }

  const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / Math.sqrt(na * nb || 1); };

  function makeModel(mode, seed) {
    const { rng, gauss } = Atlas.nn, r = rng(seed);
    const Win = Array.from({ length: NV }, () => Array.from({ length: D }, () => gauss(r) * 0.1));
    const Wout = Array.from({ length: NV }, () => Array.from({ length: D }, () => gauss(r) * 0.1));
    const M = { mode, Win, Wout, ep: 0, loss: [] };
    /** 由輸入詞（可多個，取平均）算出 26 個詞的機率 */
    M.predict = (inp) => {
      const hh = new Array(D).fill(0); inp.forEach((k) => { for (let q = 0; q < D; q++) hh[q] += Win[k][q] / inp.length; });
      const sc = Wout.map((w) => w.reduce((a, v, q) => a + v * hh[q], 0)), m = Math.max(...sc), ex = sc.map((x) => Math.exp(x - m)), z = ex.reduce((a, b) => a + b);
      return { h: hh, p: ex.map((x) => x / z) };
    };
    function learn(inp, tgt) {
      const { h: hh, p } = M.predict(inp), dh = new Array(D).fill(0);
      for (let v = 0; v < NV; v++) { const g = p[v] - (v === tgt ? 1 : 0); for (let q = 0; q < D; q++) { dh[q] += g * Wout[v][q]; Wout[v][q] -= LR * g * hh[q]; } }
      inp.forEach((k) => { for (let q = 0; q < D; q++) Win[k][q] -= (LR * dh[q]) / inp.length; });
      return -Math.log(p[tgt]);
    }
    M.epoch = () => {
      const order = SAMPLES.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      let L = 0, n = 0;
      for (const oi of order) {
        const S = SAMPLES[oi];
        if (mode === "cbow") { L += learn(S.ctx, S.c); n++; }
        else for (const o of S.ctx) { L += learn([S.c], o); n++; }
      }
      M.ep++; M.loss.push([M.ep, L / n]);
    };
    /** 目前資料上的平均損失（不更新） */
    M.evalLoss = () => {
      let L = 0, n = 0;
      for (const S of SAMPLES) {
        if (mode === "cbow") { L -= Math.log(M.predict(S.ctx).p[S.c]); n++; }
        else for (const o of S.ctx) { L -= Math.log(M.predict([S.c]).p[o]); n++; }
      }
      return L / n;
    };
    M.loss0 = M.evalLoss();
    return M;
  }
  function neighbors(W, w, k = 5) { return VOC.map((v, i) => [v, cos(W[ID[w]], W[i])]).filter((x) => x[0] !== w).sort((a, b) => b[1] - a[1]).slice(0, k); }
  function analogy(W, [a, b, c]) {
    const v = W[ID[a]].map((x, i) => x - W[ID[b]][i] + W[ID[c]][i]);
    return { v, list: VOC.map((w, i) => [w, cos(v, W[i])]).filter((x) => ![a, b, c].includes(x[0])).sort((x, y) => y[1] - x[1]).slice(0, 3) };
  }
  /** 前兩個主成分（冪次法），正負號固定成「國王在右上」 */
  function pca2(W) {
    const mu = new Array(D).fill(0); W.forEach((r) => r.forEach((v, q) => (mu[q] += v / W.length)));
    const X = W.map((r) => r.map((v, q) => v - mu[q]));
    const Cv = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => X.reduce((a, r) => a + r[i] * r[j], 0) / X.length));
    const comps = [];
    for (let k = 0; k < 2; k++) {
      let v = Array.from({ length: D }, (_, i) => Math.sin(i * 1.7 + k + 1));
      for (let it = 0; it < 200; it++) {
        let nv = Cv.map((row) => row.reduce((a, x, j) => a + x * v[j], 0));
        comps.forEach((u) => { const d = nv.reduce((a, x, j) => a + x * u[j], 0); nv = nv.map((x, j) => x - d * u[j]); });
        const nn2 = Math.hypot(...nv) || 1; v = nv.map((x) => x / nn2);
      }
      comps.push(v);
    }
    const P = X.map((r) => comps.map((u) => r.reduce((a, x, j) => a + x * u[j], 0)));
    [0, 1].forEach((k) => { if (P[ID[K.king]][k] < 0) P.forEach((p) => (p[k] = -p[k])); });
    return P;
  }

  Atlas.define("w2v", {
    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { t } = ctx;
      setCorpus(t("corpus"), t("keyWords"));
      let mode = "sg", seed = 1, sel = K.king, sample = 12, ana = 0, M = null, loop = null, P = null;

      bench.append(h("style", { text: `
        .wv-corpus { display: flex; flex-wrap: wrap; gap: 6px 10px; font-size: 14px; }
        .wv-sent { display: inline-flex; gap: 2px; padding: 2px 6px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel-2); }
        .wv-w { border: 0; background: transparent; padding: 1px 4px; border-radius: 4px; cursor: pointer; font: inherit; color: var(--ink); }
        .wv-w:hover { background: var(--accent-soft); }
        .wv-w.c { background: var(--hot); color: var(--panel); font-weight: 700; }
        .wv-w.x { background: var(--hot-soft); outline: 1.5px solid var(--hot); }
        .wv-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; align-items: flex-start; }
        .wv-fig { flex: 1 1 320px; max-width: 440px; min-width: 0; }
        .wv-fig > svg { display: block; width: 100%; height: auto; }
        .wv-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .wv-key { display: flex; flex-wrap: wrap; gap: 4px 12px; justify-content: center; font-size: 12.5px; color: var(--muted); }
        .wv-key i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
        .wv-pt { cursor: pointer; }
        .wv-lab { stroke: var(--panel-2); }
      ` }));

      const modeSeg = ui.seg({ label: t("modeLabel"), value: mode, options: [{ value: "sg", label: t("modeSg") }, { value: "cbow", label: t("modeCbow") }], onChange: (v) => { mode = v; reset(); } });
      const seedBtn = ui.btn(t("btnSeed"), () => { seed = (seed % 5) + 1; reset(); }, { sm: true });
      const runBtn = ui.btn(t("btnStart"), () => { if (loop.running) loop.pause(); else loop.start(); render(); }, { primary: true });
      const oneBtn = ui.btn(t("btnStep"), () => { loop.pause(); if (M.ep < EPOCHS) M.epoch(); render(); });
      const ffBtn = ui.btn(t("btnFF"), () => { loop.fastForward(); render(); });

      const status = h("div", { class: "status" }), corpus = h("div", { class: "wv-corpus" }), sampleCalc = h("div", { class: "calc" });
      const mapSvg = s("svg", { viewBox: "0 0 380 320", role: "img", "aria-label": t("mapAria") }), lossSvg = s("svg", { role: "img", "aria-label": t("lossAria") });
      const nbCalc = h("div", { class: "calc" }), anaCalc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const anaSeg = ui.seg({ value: ana, options: ANALOGY.map(([a, b, c], i) => ({ value: i, label: t("anaOpt", { a, b, c }) })), onChange: (v) => { ana = v; renderAna(); } });

      bench.append(
        h("div", { class: "bench__row" }, modeSeg.el, seedBtn),
        h("div", { class: "field" }, h("span", { class: "field__label" }, t("corpusLabel", { s: SENT.length, v: NV })), corpus),
        sampleCalc,
        h("div", { class: "bench__row" }, runBtn, oneBtn, ffBtn),
        status,
        h("div", { class: "wv-row" },
          h("div", { class: "wv-fig" }, h("div", { class: "wv-cap" }, t("mapCap")), mapSvg,
            h("div", { class: "wv-key", html: [["p", "groupP"], ["a", "groupA"], ["f", "groupF"], ["o", "groupO"]].map(([g, k]) => `<span><i style="background:${GCOL[g]}"></i>${t(k)}</span>`).join("") })),
          h("div", { class: "wv-fig", style: { flex: "1 1 280px" } }, h("div", { class: "wv-cap" }, t("lossCap")), lossSvg)),
        h("div", { class: "field" }, h("span", { class: "field__label", html: t("nbLabel", { w: `<b class="wv-selname"></b>` }) }), nbCalc),
        h("div", { class: "field" }, h("span", { class: "field__label" }, t("anaLabel")), anaSeg.el, anaCalc),
        note);

      /* ---------- 語料 ---------- */
      const wordBtns = [];
      SENT.forEach((snt, si) => {
        const box = h("span", { class: "wv-sent" });
        snt.forEach((w, i) => { const b = h("button", { type: "button", class: "wv-w" }, w); b.addEventListener("click", () => { sample = SAMPLES.findIndex((x) => x.si === si && x.i === i); sel = w; render(); }); wordBtns.push({ b, si, i }); box.append(b); });
        corpus.append(box);
      });

      function reset() {
        if (loop) loop.pause();
        M = makeModel(mode, seed);
        loop = nn.loop(ctx, { total: EPOCHS, perTick: 2, run(from, n) { for (let i = 0; i < n && M.ep < EPOCHS; i++) M.epoch(); if (M.ep >= EPOCHS) loop.pause(); }, onTick: render });
        render();
      }

      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      const pct = (p) => (p < 0.001 ? "<0.1" : fmt(p * 100, 1)) + "%";
      function renderSample() {
        const S = SAMPLES[sample], w = VOC[S.c], ctxW = S.ctx.map((k) => VOC[k]);
        wordBtns.forEach(({ b, si, i }) => { b.classList.toggle("c", si === S.si && i === S.i); b.classList.toggle("x", si === S.si && Math.abs(i - S.i) <= WIN && i !== S.i); });
        const guess = `<span class="k">${t("randomNote", { v: NV, p: fmt(100 / NV, 1) })}</span>`;
        if (mode === "sg") {
          const p = M.predict([S.c]).p;
          sampleCalc.innerHTML = row(t("sampleSgK"), t("sampleSg", { w, n: ctxW.length })) +
            row(t("modelK"), ctxW.map((c) => `P(${c} | ${w}) = <span class="hl">${pct(p[ID[c]])}</span>`).join("　") + "　" + guess);
        } else {
          const p = M.predict(S.ctx).p, cw = ctxW.join(t("listSep"));
          sampleCalc.innerHTML = row(t("sampleCbowK"), t("sampleCbow", { ctx: cw, w })) +
            row(t("modelK"), `P(${w} | ${cw}) = <span class="hl">${pct(p[S.c])}</span>　` + guess);
        }
      }

      function drawMap() {
        P = pca2(M.Win);
        const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
        const pad = 0.08, x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
        const sc = Math.min(330 / (x1 - x0 || 1), 270 / (y1 - y0 || 1)) * (1 - pad);
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        const X = (v) => 190 + (v - cx) * sc, Y = (v) => 160 - (v - cy) * sc;
        mapSvg.replaceChildren();
        mapSvg.append(s("rect", { x: 1, y: 1, width: 378, height: 318, rx: 8, fill: "var(--panel-2)", stroke: "var(--line)" }));
        const nb = neighbors(M.Win, sel, 3).map((x) => x[0]);
        nb.forEach((w) => mapSvg.append(s("line", { x1: X(P[ID[sel]][0]), y1: Y(P[ID[sel]][1]), x2: X(P[ID[w]][0]), y2: Y(P[ID[w]][1]), stroke: "var(--hot)", "stroke-width": 1.6, "stroke-dasharray": "4 3" })));
        // 標籤避讓：依序嘗試右、左、上、下、右上、左下，選第一個不和已放標籤重疊的位置
        const placed = [], order = VOC.map((w, i) => i).sort((a, b) => (GROUP[VOC[b]] ? 1 : 0) - (GROUP[VOC[a]] ? 1 : 0) || (VOC[b] === sel) - (VOC[a] === sel));
        const lab = {};
        order.forEach((i) => {
          const w = VOC[i], fs = GROUP[w] ? 13 : 11.5, tw = w.length * fs + 2, th = fs + 2, px = X(P[i][0]), py = Y(P[i][1]);
          const cands = [[px + 8, py + fs * 0.35, "start"], [px - 8, py + fs * 0.35, "end"], [px, py - 9, "middle"], [px, py + fs + 7, "middle"], [px + 7, py - 7, "start"], [px - 7, py + fs + 4, "end"]];
          const box = ([x, y, an]) => { const x0 = an === "start" ? x : an === "end" ? x - tw : x - tw / 2; return [x0, y - fs * 0.85, x0 + tw, y - fs * 0.85 + th]; };
          const hit = (b) => placed.some((q) => b[0] < q[2] && b[2] > q[0] && b[1] < q[3] && b[3] > q[1]) || b[0] < 3 || b[2] > 377 || b[1] < 3 || b[3] > 317;
          let pick = cands.find((c) => !hit(box(c))) || cands[0];
          placed.push(box(pick)); lab[i] = { pick, fs };
        });
        VOC.forEach((w, i) => {
          const g = s("g", { class: "wv-pt" }), isSel = w === sel, col = GCOL[GROUP[w] || "o"], { pick, fs } = lab[i];
          g.append(s("circle", { cx: X(P[i][0]), cy: Y(P[i][1]), r: isSel ? 7 : 5, fill: col, stroke: isSel ? "var(--ink)" : "var(--panel)", "stroke-width": isSel ? 2 : 1 }));
          g.append(s("text", { x: pick[0], y: pick[1], "text-anchor": pick[2], "font-size": fs, "font-weight": GROUP[w] || isSel ? 700 : 400, class: "t-halo wv-lab", fill: GROUP[w] ? null : "var(--muted)", text: w }));
          g.append(s("circle", { cx: X(P[i][0]), cy: Y(P[i][1]), r: 13, fill: "transparent" }));
          g.addEventListener("click", () => { sel = w; const k = SAMPLES.findIndex((x) => x.c === i); if (k >= 0) sample = k; render(); });
          mapSvg.append(g);
        });
      }

      function render() {
        renderSample(); drawMap();
        nn.lineChart(lossSvg, { W: 340, H: 200, xMax: EPOCHS, yMin: 0, yMax: 3.5, xLabel: t("epochAxis"), series: [{ pts: [[0, M.loss0], ...M.loss], color: "var(--hot)" }] });
        runBtn.textContent = t(loop.running ? "btnPause" : M.ep >= EPOCHS ? "btnDone" : M.ep ? "btnContinue" : "btnStart");
        runBtn.disabled = oneBtn.disabled = ffBtn.disabled = M.ep >= EPOCHS;
        const pairs = [[K.king, K.queen], [K.cat, K.dog], [K.man, K.woman], [K.fish, K.meat]];
        const okPairs = pairs.filter(([a, b]) => neighbors(M.Win, a, 1)[0][0] === b && neighbors(M.Win, b, 1)[0][0] === a);
        const curLoss = M.ep ? M.loss[M.loss.length - 1][1] : M.loss0;
        let kd = "info", txt;
        if (!M.ep) txt = t("statusStart", { loss: fmt(M.loss0, 2), v: NV, ln: fmt(Math.log(NV), 2), ep: EPOCHS });
        else {
          const found = okPairs.length ? t("statusPairs", { list: okPairs.map((p) => p.join("↔")).join(t("listSep")) }) : t("statusNoPairs");
          txt = t(loop.running ? "statusEpRunning" : "statusEp", { ep: M.ep, loss: fmt(curLoss, 2), found });
          const miss = pairs.filter((p) => !okPairs.includes(p));
          if (M.ep >= EPOCHS) { kd = miss.length ? "warn" : "ok"; if (miss.length) txt += t("statusMiss", { list: miss.map((p) => p.join(t("listSep"))).join(t("listSep")), why: t(mode === "cbow" ? "whyCbow" : "whySeed") }); }
        }
        status.className = "status status--" + kd; status.textContent = txt;
        bench.querySelector(".wv-selname").textContent = sel;
        const nbs = neighbors(M.Win, sel, 5), oh = `<span class="k">${t("oneHot", { w: sel })}</span>`;
        nbCalc.innerHTML = row(t("nbVecK"), `${t("quoted", { w: sel })} = [${M.Win[ID[sel]].map((v) => fmt(v, 2)).join(", ")}]`) +
          row(t("nbTopK"), nbs.map(([w, c], i) => `${i === 0 ? "<span class=\"hl\">" : ""}${w} ${fmt(c, 2)}${i === 0 ? "</span>" : ""}`).join("　")) + `<span class="calc-row">${oh}</span>`;
        renderAna();
      }
      function renderAna() {
        const [a, b, c, want] = ANALOGY[ana], r = analogy(M.Win, ANALOGY[ana]);
        anaCalc.innerHTML = row(t("anaRowK", { a, b, c }), `= [${r.v.map((v) => fmt(v, 2)).join(", ")}]`) +
          row(t("anaTopK"), r.list.map(([w, cc], i) => `${i === 0 ? "<span class=\"hl\">" : ""}${w} ${fmt(cc, 2)}${i === 0 ? "</span>" : ""}`).join("　")) +
          row(t("anaJudgeK"), !M.ep ? t("anaUntrained") : r.list[0][0] === want ? t("anaOk", { w: want }) : t("anaNot", { got: r.list[0][0], w: want }));
      }

      note.innerHTML = t("note");

      reset();
      ctx.onTheme(render);
      return { destroy() { if (loop) loop.pause(); } };
    },
  });
})();
