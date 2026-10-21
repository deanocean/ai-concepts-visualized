/* 単語埋め込み（word2vec）（シラバス §27）
   小語料 20 句、26 個詞。窗口 2（前後各 2 個詞），向量 8 維，完整 softmax（26 類），SGD η = 0.05，200 エポック，每エポック洗牌（固定種子）。
   スキップグラム：用中間的詞猜窗口內每個詞；CBOW：用窗口內詞向量的平均猜中間的詞。詞向量＝輸入側的權重 W_in 的那一列。
   實測種子 1〜5：スキップグラム每次「國王↔王后、貓↔狗、男人↔女人、魚↔肉」都互為最相近；三組類推（國王−男人+女人 等）兩種做法全部答對；
   CBOW 不穩定（有些詞最相近的對象會跑掉，例如魚、肉）。平面圖＝8 維投影到前兩個主成分。 */
(function () {
  const SENT = [
    "他 是 國王", "她 是 王后", "他 是 男人", "她 是 女人",
    "國王 住 城堡", "王后 住 城堡", "男人 住 村莊", "女人 住 村莊",
    "國王 戴 王冠", "王后 戴 王冠", "男人 下 田", "女人 下 田",
    "貓 吃 魚", "狗 吃 肉", "貓 喝 水", "狗 喝 水",
    "貓 在 家 睡覺", "狗 在 家 睡覺", "我 養 貓", "我 養 狗",
  ].map((l) => l.split(" "));
  const VOC = [...new Set(SENT.flat())], ID = Object.fromEntries(VOC.map((w, i) => [w, i])), NV = VOC.length;
  const D = 8, WIN = 2, LR = 0.05, EPOCHS = 200;
  const GROUP = { 國王: "p", 王后: "p", 男人: "p", 女人: "p", 貓: "a", 狗: "a", 魚: "f", 肉: "f", 水: "f" };
  const GCOL = { p: "var(--pos)", a: "var(--hot)", f: "var(--ok)", o: "var(--muted)" };
  const ANALOGY = [["國王", "男人", "女人"], ["男人", "國王", "王后"], ["王后", "女人", "男人"]];
  // 訓練樣本：每個位置一筆（中間的詞、窗口內的詞）
  const SAMPLES = [];
  SENT.forEach((s, si) => s.forEach((w, i) => { const ctx = []; for (let j = Math.max(0, i - WIN); j <= Math.min(s.length - 1, i + WIN); j++) if (j !== i) ctx.push(ID[s[j]]); SAMPLES.push({ si, i, c: ID[w], ctx }); }));

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
    [0, 1].forEach((k) => { if (P[ID["國王"]][k] < 0) P.forEach((p) => (p[k] = -p[k])); });
    return P;
  }

  Atlas.define("w2v", {
    lead: `<p>電腦處理文字前，要先把詞變成數字。最直接的做法是<span class="jp">ワンホットベクトル</span>（one-hot 向量，<span class="en">one-hot vector</span>）：詞彙有幾個就開幾維，該詞那一格是 1、其他是 0。但這樣任兩個詞都一樣「不像」，看不出「國王」和「王后」比和「水」接近。<span class="jp">単語埋め込み</span>（詞嵌入，<span class="en">word embedding</span>）改用幾維的實數向量表示一個詞，稱為<span class="jp">分散表現</span>（分散式表示，<span class="en">distributed representation</span>）；<span class="jp">word2vec</span> 的想法是：<b>出現在相似上下文的詞，意思也相近</b>，所以讓網路練習「由一個詞猜周圍的詞」，向量就會自己長出意義。</p>`,
    closing: `訓練只是在練習「猜周圍的詞」，但上下文相似的詞會被推到相近的位置：用スキップグラム訓練，每一組初始值都把國王與王后、貓與狗、男人與女人、魚與肉配成彼此最相近。連「國王 − 男人 + 女人」這種向量加減都會落在王后附近（兩種訓練方式都是），因為「男→女」的差別在不同的詞上被學成差不多的方向。CBOW 在這麼小的語料上比較不穩定，有些詞最相近的對象會跑掉。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      let mode = "sg", seed = 1, sel = "國王", sample = 12, ana = 0, M = null, loop = null, P = null;

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

      const modeSeg = ui.seg({ label: "訓練方式", value: mode, options: [{ value: "sg", label: "スキップグラム（中間 → 周圍）" }, { value: "cbow", label: "CBOW（周圍 → 中間）" }], onChange: (v) => { mode = v; reset(); } });
      const seedBtn = ui.btn("換一組初始值", () => { seed = (seed % 5) + 1; reset(); }, { sm: true });
      const runBtn = ui.btn("▶ 開始訓練", () => { if (loop.running) loop.pause(); else loop.start(); render(); }, { primary: true });
      const oneBtn = ui.btn("前進 1 エポック", () => { loop.pause(); if (M.ep < EPOCHS) M.epoch(); render(); });
      const ffBtn = ui.btn("⏭ 快轉到結束", () => { loop.fastForward(); render(); });

      const status = h("div", { class: "status" }), corpus = h("div", { class: "wv-corpus" }), sampleCalc = h("div", { class: "calc" });
      const mapSvg = s("svg", { viewBox: "0 0 380 320", role: "img", "aria-label": "詞向量投影到平面" }), lossSvg = s("svg", { role: "img", "aria-label": "損失" });
      const nbCalc = h("div", { class: "calc" }), anaCalc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const anaSeg = ui.seg({ value: ana, options: ANALOGY.map(([a, b, c], i) => ({ value: i, label: `${a} − ${b} + ${c}` })), onChange: (v) => { ana = v; renderAna(); } });

      bench.append(
        h("div", { class: "bench__row" }, modeSeg.el, seedBtn),
        h("div", { class: "field" }, h("span", { class: "field__label" }, "語料（20 句，26 個詞）：點任一個詞，看它在訓練時變成什麼題目（窗口＝前後各 2 個詞）"), corpus),
        sampleCalc,
        h("div", { class: "bench__row" }, runBtn, oneBtn, ffBtn),
        status,
        h("div", { class: "wv-row" },
          h("div", { class: "wv-fig" }, h("div", { class: "wv-cap" }, "8 維的詞向量投影到平面（點一個詞看它最相近的詞）"), mapSvg,
            h("div", { class: "wv-key", html: `<span><i style="background:var(--pos)"></i>人</span><span><i style="background:var(--hot)"></i>動物</span><span><i style="background:var(--ok)"></i>食物・飲料</span><span><i style="background:var(--muted)"></i>其他</span>` })),
          h("div", { class: "wv-fig", style: { flex: "1 1 280px" } }, h("div", { class: "wv-cap" }, "平均損失（猜周圍詞的 −ln 機率）"), lossSvg)),
        h("div", { class: "field" }, h("span", { class: "field__label", html: `和「<b class="wv-selname"></b>」最相近的詞（<span class="jp">コサイン類似度</span>，cosine similarity，1＝方向完全相同、0＝無關）` }), nbCalc),
        h("div", { class: "field" }, h("span", { class: "field__label" }, "向量的加減（類推）"), anaSeg.el, anaCalc),
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
        if (mode === "sg") {
          const p = M.predict([S.c]).p;
          sampleCalc.innerHTML = row("スキップグラム的題目", `輸入「${w}」的向量 → 要猜出窗口內的 ${ctxW.length} 個詞（每個詞各算一次損失）`) +
            row("目前的模型", ctxW.map((c) => `P(${c} | ${w}) = <span class="hl">${pct(p[ID[c]])}</span>`).join("　") + `　<span class="k">（亂猜是 1/26 ≈ 3.8%）</span>`);
        } else {
          const p = M.predict(S.ctx).p;
          sampleCalc.innerHTML = row("CBOW 的題目", `把窗口內「${ctxW.join("、")}」的向量平均 → 要猜出中間的「${w}」`) +
            row("目前的模型", `P(${w} | ${ctxW.join("、")}) = <span class="hl">${pct(p[S.c])}</span>　<span class="k">（亂猜是 1/26 ≈ 3.8%）</span>`);
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
        nn.lineChart(lossSvg, { W: 340, H: 200, xMax: EPOCHS, yMin: 0, yMax: 3.5, xLabel: "エポック", series: [{ pts: [[0, M.loss0], ...M.loss], color: "var(--hot)" }] });
        runBtn.textContent = loop.running ? "❚❚ 暫停" : M.ep >= EPOCHS ? "✓ 訓練完成" : M.ep ? "▶ 繼續訓練" : "▶ 開始訓練";
        runBtn.disabled = oneBtn.disabled = ffBtn.disabled = M.ep >= EPOCHS;
        const pairs = [["國王", "王后"], ["貓", "狗"], ["男人", "女人"], ["魚", "肉"]];
        const okPairs = pairs.filter(([a, b]) => neighbors(M.Win, a, 1)[0][0] === b && neighbors(M.Win, b, 1)[0][0] === a);
        const curLoss = M.ep ? M.loss[M.loss.length - 1][1] : M.loss0;
        let kd = "info", txt;
        if (!M.ep) txt = `還沒訓練：詞向量是隨機的小數字，平面上的位置沒有意義，平均損失 ${fmt(M.loss0, 2)}（約等於亂猜的 ln 26 ≈ 3.26）。按「▶ 開始訓練」跑 ${EPOCHS} エポック（約 3 秒）`;
        else {
          txt = `第 ${M.ep} エポック${loop.running ? "（訓練中）" : ""}：平均損失 ${fmt(curLoss, 2)}；` + (okPairs.length ? `互為最相近的詞：${okPairs.map((p) => p.join("↔")).join("、")}` : "還沒有哪一對詞互為最相近");
          const miss = pairs.filter((p) => !okPairs.includes(p));
          if (M.ep >= EPOCHS) { kd = miss.length ? "warn" : "ok"; if (miss.length) txt += `；${miss.map((p) => p.join("、")).join("、")}還沒配成一對（${mode === "cbow" ? "CBOW 在這麼小的語料上比較不穩定" : "可以換一組初始值比較"}）`; }
        }
        status.className = "status status--" + kd; status.textContent = txt;
        bench.querySelector(".wv-selname").textContent = sel;
        const nbs = neighbors(M.Win, sel, 5), oh = `<span class="k">若用ワンホットベクトル：「${sel}」和任何其他詞的コサイン類似度都是 0</span>`;
        nbCalc.innerHTML = row("向量", `「${sel}」= [${M.Win[ID[sel]].map((v) => fmt(v, 2)).join(", ")}]`) +
          row("最相近", nbs.map(([w, c], i) => `${i === 0 ? "<span class=\"hl\">" : ""}${w} ${fmt(c, 2)}${i === 0 ? "</span>" : ""}`).join("　")) + `<span class="calc-row">${oh}</span>`;
        renderAna();
      }
      function renderAna() {
        const [a, b, c] = ANALOGY[ana], r = analogy(M.Win, ANALOGY[ana]);
        const want = { 0: "王后", 1: "女人", 2: "國王" }[ana];
        anaCalc.innerHTML = row(`「${a}」−「${b}」+「${c}」`, `= [${r.v.map((v) => fmt(v, 2)).join(", ")}]`) +
          row("最接近的詞（不算題目裡的三個）", r.list.map(([w, cc], i) => `${i === 0 ? "<span class=\"hl\">" : ""}${w} ${fmt(cc, 2)}${i === 0 ? "</span>" : ""}`).join("　")) +
          row("判讀", !M.ep ? "還沒訓練，結果沒有意義" : r.list[0][0] === want ? `答對：最接近「${want}」` : `目前最接近的是「${r.list[0][0]}」，還不是「${want}」`);
      }

      note.innerHTML = `這一頁的語料、維數（8 維）都刻意做得很小；真實的 word2vec 用數十億詞的語料、100〜300 維，而且為了省計算，不對全部詞算 softmax，改用負例抽樣（negative sampling）等近似。平面圖是把 8 維投影到前兩個主成分（見 <a href="#pca">主成分分析</a>），距離只是近似，相似度一律用原本的 8 維計算。スキップグラム（<span class="en">skip-gram</span>）在小語料、稀有詞上通常較好；<span class="jp">CBOW</span>（<span class="en">continuous bag-of-words</span>）計算較快。fastText 把詞再拆成更小的字元片段來學，沒看過的詞也能組出向量；word2vec 一個詞只有一個向量，ELMo、BERT 則依上下文給同一個詞不同的向量。把整份文件看成「各詞出現次數」的 BoW（Bag-of-Words）、以及加權的 TF-IDF，都是更早的、不考慮詞義相近的表示法。`;

      reset();
      ctx.onTheme(render);
      return { destroy() { if (loop) loop.pause(); } };
    },
  });
})();
