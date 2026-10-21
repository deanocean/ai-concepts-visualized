/* RNN と勾配消失（シラバス §22）
   沿時間展開的 RNN：正向看隱藏狀態往後傳，反向（BPTT）看梯度往回傳時一路變淡；
   切換 LSTM，看梯度沿記憶單元傳回時幾乎不衰減。數值皆即時計算（一維的簡化模型）。 */
Atlas.define("rnn", {
  lead: `<p>句子要按順序讀，後面的詞常要靠前面的內容才能理解。<span class="jp">RNN</span>（循環神經網路，<span class="en">recurrent neural network</span>）每讀一個詞，就把「到目前為止讀到的內容」壓成一個<span class="jp">隠れ状態</span>（隱藏狀態，<span class="en">hidden state</span>）交給下一步。問題出在訓練：誤差要沿同一條路倒著傳回去，傳得越遠，梯度就越淡，前面的詞幾乎學不到東西。</p>`,
  closing: `RNN 每往回一步，梯度都要乘上「w<sub>h</sub> × tanh 的斜率」，這個數通常小於 1，連乘下去就趨近 0，這就是<span class="jp">勾配消失問題</span>（梯度消失，<span class="en">vanishing gradient</span>）；LSTM 讓梯度沿記憶單元只乘上接近 1 的忘記閘，所以能傳到很前面。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const TOK = ["小明", "昨天", "在", "車站", "遇到", "了", "老同學", "他", "很", "開心"];
    const X = [0.9, -0.4, 0.2, 0.7, -0.6, 0.1, 0.8, 0.5, -0.3, 0.6];
    const T = TOK.length, WX = 1.0;
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const LP = { wf: 0.5, uf: 0.5, bf: 2.5, wi: 1, ui: 0.5, bi: -0.5, wg: 1, ug: 0.5, wo: 1, uo: 0.5, bo: 0.5 };

    let mode = "rnn", wh = 0.8, pos = 0; // pos: 0..2T

    function runRNN(w) {
      const H = [], pre = []; let hp = 0;
      for (let t = 0; t < T; t++) { const z = w * hp + WX * X[t]; pre.push(z); hp = Math.tanh(z); H.push(hp); }
      const G = new Array(T); G[T - 1] = 1;
      const F = new Array(T).fill(null); // F[t] = 從 t+1 傳回 t 時乘的數
      for (let t = T - 2; t >= 0; t--) { F[t] = w * (1 - H[t + 1] ** 2); G[t] = G[t + 1] * F[t]; }
      return { H, pre, G, F };
    }
    function runLSTM() {
      const H = [], C = [], gates = []; let hp = 0, cp = 0;
      for (let t = 0; t < T; t++) {
        const x = X[t];
        const f = sig(LP.wf * x + LP.uf * hp + LP.bf), i = sig(LP.wi * x + LP.ui * hp + LP.bi);
        const g = Math.tanh(LP.wg * x + LP.ug * hp), o = sig(LP.wo * x + LP.uo * hp + LP.bo);
        const c = f * cp + i * g, hh = o * Math.tanh(c);
        gates.push({ f, i, g, o, cPrev: cp, hPrev: hp }); C.push(c); H.push(hh); cp = c; hp = hh;
      }
      const G = new Array(T); G[T - 1] = 1;
      const F = new Array(T).fill(null);
      for (let t = T - 2; t >= 0; t--) { F[t] = gates[t + 1].f; G[t] = G[t + 1] * F[t]; }
      return { H, C, gates, G, F };
    }
    const fmtG = (v) => (v >= 0.1 ? v.toFixed(2) : v >= 0.001 ? v.toFixed(3) : v === 0 ? "0" : Number(v.toPrecision(1)).toString());

    /* ---------- controls ---------- */
    const modeSeg = ui.seg({ label: "模型", value: mode, options: [{ value: "rnn", label: "RNN" }, { value: "lstm", label: "換成 LSTM" }],
      onChange: (v) => { mode = v; whSlider.el.hidden = v !== "rnn"; render(); } });
    const whSlider = ui.slider({ id: "rnn-wh", label: "循環權重 w<sub>h</sub>（上一步狀態的權重）", min: 0.1, max: 2, step: 0.05, value: wh, format: (v) => v.toFixed(2), onInput: (v) => { wh = v; render(); } });
    const player = ui.player(ctx, () => { if (pos >= 2 * T) return false; pos++; render(); return pos < 2 * T; }, 700);
    const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn("下一步 ▶", () => { player.pause(); if (pos < 2 * T) { pos++; render(); } }, { primary: true });
    const jumpBtn = ui.btn("跳到反向傳播", () => { player.pause(); pos = T + 1; render(); });
    const resetBtn = ui.btn("重設", () => { player.pause(); pos = 0; render(); });

    const phase = h("div", { class: "status" });
    const svgWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const note = h("p", { class: "note" });
    const barWrap = h("div");

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el, whSlider.el),
      phase,
      svgWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, jumpBtn, resetBtn),
      calc, note, barWrap,
    );

    /* ---------- drawing ---------- */
    const COLW = 76, LEFT = 74, W = LEFT + COLW * T + 96;
    const cx = (t) => LEFT + COLW * t + COLW / 2;

    function arrow(x1, y1, x2, y2, color, width = 1.6, dash) {
      const g = s("g");
      g.append(s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": width, "stroke-dasharray": dash || null }));
      const a = Math.atan2(y2 - y1, x2 - x1), L = 6 + width;
      g.append(s("path", { d: `M${x2},${y2} L${x2 - L * Math.cos(a - 0.45)},${y2 - L * Math.sin(a - 0.45)} L${x2 - L * Math.cos(a + 0.45)},${y2 - L * Math.sin(a + 0.45)} Z`, fill: color }));
      return g;
    }

    function render() {
      const R = runRNN(wh), Lm = mode === "lstm" ? runLSTM() : null;
      const M = Lm || R;
      const fwd = Math.min(pos, T);            // 已送入的詞數
      const back = Math.max(0, pos - T);       // 已傳回的步數（1 ＝ 剛在最後一步算出梯度 1）
      const gKnown = (t) => back > 0 && t >= T - back;
      const curT = pos === 0 ? -1 : pos <= T ? pos - 1 : T - back;

      const lstm = !!Lm;
      const H = lstm ? 250 : 210;
      const yGrad = 22, yC = 78, yH = lstm ? 150 : 100, yTok = lstm ? 222 : 182;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "沿時間展開的網路", style: `display:block;width:100%;min-width:${W * 0.78}px;max-width:${W}px` });

      // row labels
      const lab = (y, t1, t2) => { svg.append(s("text", { x: 6, y: y - 2, "font-size": 12, "font-weight": 700, text: t1 })); if (t2) svg.append(s("text", { x: 6, y: y + 13, "font-size": 10.5, class: "t-muted", text: t2 })); };
      lab(yTok, "輸入", "詞 → 數字");
      lab(yH, "隱藏狀態", "h");
      if (lstm) lab(yC, "記憶單元", "c（セル）");
      if (back) svg.append(s("text", { x: 6, y: yGrad + 4, "font-size": 11.5, "font-weight": 700, fill: "var(--hot)", text: "梯度" }));

      const hMax = 1;
      // highway for LSTM
      if (lstm) {
        svg.append(s("line", { x1: cx(0) - 34, x2: cx(T - 1) + 34, y1: yC, y2: yC, stroke: "var(--line-strong)", "stroke-width": 6, "stroke-linecap": "round" }));
      }
      for (let t = 0; t < T; t++) {
        const x = cx(t), on = t < fwd;
        // token
        svg.append(s("text", { x, y: yTok, "text-anchor": "middle", "font-size": 14, "font-weight": t === curT && pos <= T ? 700 : 400, fill: on ? "var(--ink)" : "var(--faint)", text: TOK[t] }));
        svg.append(s("text", { x, y: yTok + 16, "text-anchor": "middle", "font-size": 10.5, class: "t-mono t-muted", text: fmt(X[t], 1) }));
        // input arrow
        svg.append(arrow(x, yTok - 18, x, yH + 22, on ? "var(--muted)" : "var(--line)", 1.4));
        if (lstm) svg.append(s("line", { x1: x, y1: yH - 20, x2: x, y2: yC + 14, stroke: on ? "var(--muted)" : "var(--line)", "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
        // recurrent arrow h(t-1) -> h(t)
        if (t > 0) svg.append(arrow(cx(t - 1) + 28, yH, x - 30, yH, t < fwd ? "var(--accent)" : "var(--line)", 2));
        // c cells
        if (lstm) {
          const v = Lm.C[t];
          svg.append(s("rect", { x: x - 24, y: yC - 14, width: 48, height: 28, rx: 6, fill: on ? Atlas.divColor(v, 1.6) : "var(--panel-2)", stroke: t === curT ? "var(--hot)" : "var(--line-strong)", "stroke-width": t === curT ? 2.5 : 1 }));
          if (on) svg.append(s("text", { x, y: yC + 4, "text-anchor": "middle", "font-size": 11.5, class: "t-mono", text: fmt(v, 2) }));
          if (t > 0 && t < fwd) svg.append(s("text", { x: x - COLW / 2, y: yC + 30, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", fill: "var(--accent)", text: "×" + fmt(Lm.gates[t].f, 2) }));
        }
        // h cells
        const hv = M.H[t];
        svg.append(s("rect", { x: x - 28, y: yH - 20, width: 56, height: 40, rx: 8, fill: on ? Atlas.divColor(hv, hMax) : "var(--panel-2)", stroke: t === curT && (pos <= T || !lstm) ? "var(--hot)" : "var(--line-strong)", "stroke-width": t === curT ? 2.5 : 1 }));
        svg.append(s("text", { x, y: yH + 5, "text-anchor": "middle", "font-size": 13, class: "t-mono", fill: on ? (Math.abs(hv) > 0.6 ? "var(--panel)" : "var(--ink)") : "var(--faint)", text: on ? fmt(hv, 2) : "?" }));

        // gradient badges
        if (gKnown(t)) {
          const g = M.G[t], a = Math.max(0.12, Math.min(1, Math.sqrt(g)));
          svg.append(s("rect", { x: x - 26, y: yGrad - 13, width: 52, height: 22, rx: 11, fill: "var(--hot)", "fill-opacity": a, stroke: "var(--hot)" }));
          svg.append(s("text", { x, y: yGrad + 3, "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, class: "t-mono", fill: a > 0.55 ? "var(--panel)" : "var(--ink)", text: fmtG(g) }));
          if (t < T - 1) {
            const yA = lstm ? yC - 26 : yH - 32, wdt = 1 + 5 * Math.sqrt(M.G[t]);
            svg.append(arrow(cx(t + 1) - 22, yA, x + 22, yA, "var(--hot)", wdt));
          }
        }
      }
      // output / loss at end
      const xe = cx(T - 1) + 34;
      svg.append(arrow(xe, yH, xe + 30, yH, fwd === T ? "var(--accent)" : "var(--line)", 2));
      svg.append(s("text", { x: xe + 36, y: yH - 4, "font-size": 12, fill: fwd === T ? "var(--ink)" : "var(--faint)", text: "輸出" }));
      svg.append(s("text", { x: xe + 36, y: yH + 12, "font-size": 12, fill: fwd === T ? "var(--ink)" : "var(--faint)", text: "→ 損失" }));
      svgWrap.replaceChildren(svg);

      /* ---------- phase + calc ---------- */
      const set = (k, t) => { phase.className = "status status--" + k; phase.textContent = t; };
      if (pos === 0) {
        set("info", "① 正向傳播：按「下一步」把句子一個詞一個詞送進去");
        calc.innerHTML = `<span class="k">每個詞先轉成數字（真實模型是一串數字，這裡簡化成一個）。隱藏狀態 h 也是一個數字，代表「到目前為止讀到的內容」的摘要，一開始是 0。</span>`;
      } else if (pos <= T) {
        const t = pos - 1;
        set("info", `① 正向傳播：送入第 ${t + 1} 個詞「${TOK[t]}」`);
        const hp = t ? M.H[t - 1] : 0;
        if (!lstm) {
          calc.innerHTML =
            `<span class="calc-row"><span class="k">新的狀態 = tanh(w<sub>h</sub> × 上一步狀態 + w<sub>x</sub> × 這個詞)</span></span>` +
            `<span class="calc-row">h<sub>${t + 1}</sub> = tanh(${fmt(wh)} × ${fmtP(hp)} + ${fmt(WX, 1)} × ${fmtP(X[t], 1)}) = tanh(${fmt(R.pre[t])}) = <span class="hl">${fmt(R.H[t])}</span></span>`;
        } else {
          const g = Lm.gates[t];
          calc.innerHTML =
            `<span class="calc-row"><span class="k">忘記閘</span> f = ${fmt(g.f)}　<span class="k">輸入閘</span> i = ${fmt(g.i)}　<span class="k">候選內容</span> g = ${fmt(g.g)}　<span class="k">輸出閘</span> o = ${fmt(g.o)}</span>` +
            `<span class="calc-row"><span class="k">記憶單元</span> c<sub>${t + 1}</sub> = f × c<sub>${t}</sub> + i × g = ${fmt(g.f)} × ${fmtP(g.cPrev)} + ${fmt(g.i)} × ${fmtP(g.g)} = <span class="hl">${fmt(Lm.C[t])}</span></span>` +
            `<span class="calc-row"><span class="k">隱藏狀態</span> h<sub>${t + 1}</sub> = o × tanh(c<sub>${t + 1}</sub>) = ${fmt(g.o)} × ${fmt(Math.tanh(Lm.C[t]))} = <span class="hl">${fmt(Lm.H[t])}</span></span>`;
        }
      } else {
        const t = T - back; // 0-indexed 剛算出梯度的位置
        if (back === 1) {
          set("warn", "② 反向傳播：在最後一步算出損失，梯度從這裡開始往回傳");
          calc.innerHTML = `<span class="k">把最後一步的梯度當作 1（100%），看它傳回每個詞時還剩多少。沿時間倒著做誤差逆伝播，叫 </span><b>BPTT</b><span class="k">（通過時間的反向傳播，backpropagation through time）。</span>`;
        } else {
          set("warn", `② 反向傳播：梯度傳回第 ${t + 1} 個詞「${TOK[t]}」，剩 ${fmtG(M.G[t])}`);
          if (!lstm) {
            const hn = R.H[t + 1];
            calc.innerHTML =
              `<span class="calc-row"><span class="k">往回一步要乘上</span> w<sub>h</sub> × (1 − h<sub>${t + 2}</sub>²) = ${fmt(wh)} × (1 − ${fmtP(hn)}²) = ${fmt(wh)} × ${fmt(1 - hn * hn)} = <span class="hl">${fmt(R.F[t], 3)}</span></span>` +
              `<span class="calc-row"><span class="k">梯度</span> ${fmtG(R.G[t + 1])} × ${fmt(R.F[t], 3)} = <span class="hl">${fmtG(R.G[t])}</span></span>`;
          } else {
            calc.innerHTML =
              `<span class="calc-row"><span class="k">沿記憶單元往回一步，只乘上下一步的忘記閘</span> f<sub>${t + 2}</sub> = <span class="hl">${fmt(Lm.F[t], 3)}</span></span>` +
              `<span class="calc-row"><span class="k">梯度</span> ${fmtG(Lm.G[t + 1])} × ${fmt(Lm.F[t], 3)} = <span class="hl">${fmtG(Lm.G[t])}</span></span>`;
          }
          if (t === 0) set(lstm ? "ok" : "bad", lstm
            ? `傳回第 1 個詞時，梯度還剩 ${fmtG(Lm.G[0])}（約 ${Math.round(Lm.G[0] * 100)}%）`
            : `傳回第 1 個詞時，梯度只剩 ${fmtG(R.G[0])}（${(R.G[0] * 100).toFixed(R.G[0] < 0.01 ? 2 : 1)}%）：「${TOK[0]}」幾乎學不到`);
        }
      }

      if (!lstm) {
        note.innerHTML = `(1 − h²) 是 tanh 在那一點的斜率，最大只有 1；h 越接近 ±1，斜率越接近 0。試著把 w<sub>h</sub> 調大：狀態被 tanh 壓到 ±1 附近，梯度反而消失得更快。若換成不會壓縮的活性化関数、每步乘的數又大於 1，梯度會越傳越大，稱為<span class="jp">勾配爆発問題</span>（梯度爆炸，<span class="en">exploding gradient</span>）。同樣的連乘也發生在層數很深的網路，見 <a href="#vanish">勾配消失問題</a>。`;
      } else {
        note.innerHTML = `LSTM（長短期記憶，<span class="en">long short-term memory</span>）多了一條記憶單元 c 的直通路：每步只做「乘上<span class="jp">忘却ゲート</span>（忘記閘，<span class="en">forget gate</span>）再加上新內容」，不經過 tanh 的壓縮。忘記閘接近 1 時，梯度沿這條路往回傳幾乎不打折。各個閘門怎麼決定開關，見 <a href="#lstm">LSTM のゲート</a>。`;
      }

      /* ---------- gradient bars ---------- */
      barWrap.replaceChildren();
      if (back > 0) {
        const BW = W, BH = 150, top = 22, base = BH - 30;
        const bs = s("svg", { viewBox: `0 0 ${BW} ${BH}`, role: "img", "aria-label": "傳回各詞的梯度大小", style: `display:block;width:100%;min-width:${BW * 0.78}px;max-width:${BW}px` });
        const py = (v) => base - v * (base - top);
        [0, 0.5, 1].forEach((v) => {
          bs.append(s("line", { x1: LEFT - 8, x2: cx(T - 1) + 34, y1: py(v), y2: py(v), stroke: "var(--line)" }));
          bs.append(s("text", { x: LEFT - 12, y: py(v) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: v }));
        });
        bs.append(s("text", { x: 6, y: 14, "font-size": 12, "font-weight": 700, text: "傳回各詞的梯度（最後一步＝1）" }));
        for (let t = 0; t < T; t++) {
          if (!gKnown(t)) continue;
          const x = cx(t);
          if (lstm) {
            const rv = R.G[t], lv = Lm.G[t];
            bs.append(s("rect", { x: x - 18, y: py(rv), width: 16, height: Math.max(1, base - py(rv)), fill: "var(--faint)", rx: 2 }));
            bs.append(s("rect", { x: x + 2, y: py(lv), width: 16, height: Math.max(1, base - py(lv)), fill: "var(--accent)", rx: 2 }));
            bs.append(s("text", { x: x + 10, y: py(lv) - 4, "text-anchor": "middle", "font-size": 10, class: "t-mono", fill: "var(--accent)", text: fmtG(lv) }));
          } else {
            const v = R.G[t];
            bs.append(s("rect", { x: x - 12, y: py(v), width: 24, height: Math.max(1, base - py(v)), fill: "var(--hot)", rx: 2 }));
            bs.append(s("text", { x, y: py(v) - 4, "text-anchor": "middle", "font-size": 10.5, class: "t-mono", text: fmtG(v) }));
          }
          bs.append(s("text", { x, y: base + 16, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: TOK[t] }));
        }
        barWrap.append(h("div", { class: "scroll-x" }, bs));
        if (lstm) barWrap.append(h("p", { class: "hint", html: `<span style="color:var(--accent);font-weight:700">■ LSTM（沿記憶單元）</span>　<span style="color:var(--faint);font-weight:700">■ 同一句用 RNN（w<sub>h</sub> = ${fmt(wh)}）</span>` }));
      }

      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= 2 * T;
      jumpBtn.hidden = pos > T;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
