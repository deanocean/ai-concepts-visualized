/* 教師強制（シラバス §22）
   沿用 Seq2Seq 頁的翻譯「私は猫が好きです → I like cats」，只看解碼器。
   為了看得清楚，解碼器簡化成「只看前一個詞」：P(下一個詞 | 前一個詞) = softmax(該列的分數)。分數是設計好的示範值（訓練到一半的模型），
   7 個詞：I, like, am, cats, a, cat, 〈結束〉。這個模型自己生成會說出「I am a cat」。
   訓練時每一步的損失 = −ln P(正解詞)。比較下一步的輸入：教師強制＝餵正解；自由生成＝餵自己的預測（機率最高的詞）。
   「更新」：對用到的每一列做一次梯度下降（分數 −= η × (p − 正解的 one-hot)），η = 0.5；兩種做法各有一份模型。 */
(function () {
  const V = ["I", "like", "am", "cats", "a", "cat", "〈結束〉"], IN = ["〈開始〉", "I", "like", "am", "cats", "a", "cat"];
  const TGT = ["I", "like", "cats", "〈結束〉"], N = 4, ETA = 0.5;
  const L0 = { "〈開始〉": [2.0, 0, 0.3, 0, 0, 0, 0], I: [0, 1.0, 2.0, 0, 0, 0, 0], like: [0, 0, 0, 1.8, 0, 0.8, 0], am: [0, 0, 0, 0, 1.8, 0, 0.2], cats: [0, 0, 0, 0, 0, 0, 2.0], a: [0, 0, 0, 0, 0, 2.0, 0], cat: [0, 0, 0, 0, 0, 0, 1.5] };
  const softmax = (r) => { const e = r.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((x) => x / z); };
  const clone = (L) => Object.fromEntries(Object.entries(L).map(([k, v]) => [k, v.slice()]));

  /** 跑一次訓練用的 4 步 */
  function pass(L, mode) {
    let inp = "〈開始〉"; const steps = [];
    for (let k = 0; k < N; k++) {
      const p = softmax(L[inp]), pred = V[p.indexOf(Math.max(...p))], ti = V.indexOf(TGT[k]);
      steps.push({ k, inp, inpFrom: k === 0 ? "start" : mode, p, pred, ti, loss: -Math.log(p[ti]), ok: pred === TGT[k] });
      inp = mode === "tf" ? TGT[k] : pred;
    }
    return steps;
  }
  function generate(L) {
    let w = "〈開始〉"; const out = [];
    for (let k = 0; k < 6; k++) { if (!L[w]) break; const p = softmax(L[w]); w = V[p.indexOf(Math.max(...p))]; out.push(w); if (w === "〈結束〉") break; }
    return out;
  }

  Atlas.define("teacher", {
    lead: `<p>RNN 的解碼器一次產生一個詞，而每一步的輸入是「前一個詞」。訓練時如果把模型<b>自己的預測</b>當成下一步的輸入，前面只要錯一個詞，後面整串都建立在錯誤上，損失也就在教它錯的東西。<span class="jp">教師強制</span>（<span class="en">teacher forcing</span>）的做法是：訓練時下一步一律餵<b>正解</b>的詞，不管模型剛才預測了什麼。</p>`,
    closing: `教師強制讓每一步都在「前文正確」的情況下學習，錯一個詞不會拖累後面，所以學得快也學得穩（還能一次平行算完所有步）。代價是推論時沒有正解可餵、只能餵自己的預測，模型會遇到訓練時沒看過的錯誤前文。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      let mode = "tf", pos = 0;
      const models = { tf: { L: clone(L0), n: 0, changed: [] }, fr: { L: clone(L0), n: 0, changed: [] } };

      bench.append(h("style", { text: `
        .tf-heat { border-collapse: separate; border-spacing: 2px; font-family: var(--f-mono); font-size: 12px; }
        .tf-heat th { font-weight: 500; color: var(--muted); padding: 2px 5px; white-space: nowrap; }
        .tf-heat th.row { text-align: right; }
        .tf-heat td { min-width: 46px; height: 28px; text-align: center; border-radius: 4px; }
        .tf-heat td.used { outline: 2.5px solid var(--hot); outline-offset: -2px; font-weight: 700; }
        .tf-heat tr.chg th { color: var(--hot); }
      ` }));

      const modeSeg = ui.seg({ label: "訓練時，下一步的輸入用什麼", value: mode,
        options: [{ value: "tf", label: "教師強制（餵正解）" }, { value: "fr", label: "自由生成（餵自己的預測）" }],
        onChange: (v) => { mode = v; render(); } });
      const player = ui.player(ctx, () => { if (pos >= N) return false; pos++; render(); return pos < N; }, 900);
      const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (pos > 0) { pos--; render(); } });
      const nextBtn = ui.btn("下一步 ▶", () => { player.pause(); if (pos < N) { pos++; render(); } }, { primary: true });
      const resetBtn = ui.btn("回到第 0 步", () => { player.pause(); pos = 0; render(); });
      const upd1 = ui.btn("用這 4 步的損失更新 1 次", () => update(1));
      const upd5 = ui.btn("連續更新 5 次", () => update(5));
      const resetM = ui.btn("兩個模型都還原", () => { player.pause(); models.tf = { L: clone(L0), n: 0, changed: [] }; models.fr = { L: clone(L0), n: 0, changed: [] }; pos = 0; render(); }, { sm: true });

      const status = h("div", { class: "status" }), svgWrap = h("div", { class: "scroll-x" }), calc = h("div", { class: "calc" });
      const heatWrap = h("div", { class: "scroll-x" }), cmpWrap = h("div", { class: "scroll-x" }), note = h("p", { class: "note" });
      bench.append(
        h("div", { class: "bench__row" }, modeSeg.el),
        status, svgWrap,
        h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, resetBtn),
        calc,
        h("div", { class: "field" }, h("span", { class: "field__label" }, "訓練：用上面 4 步的損失做梯度下降（η = 0.5），只更新這個做法自己的那份模型"), h("div", { class: "bench__row" }, upd1, upd5, resetM)),
        h("div", { class: "field" }, h("span", { class: "field__label" }, "兩種做法的模型比一比（都從同一個「訓練到一半」的模型開始）"), cmpWrap),
        h("div", { class: "field" }, h("span", { class: "field__label", html: "模型本身：前一個詞 → 下一個詞的機率（橘框＝這次 4 步用到的「正解」那一格；橘色列名＝上次更新改到的列）" }), heatWrap),
        note);

      function update(times) {
        player.pause();
        const M = models[mode];
        for (let t = 0; t < times; t++) {
          const st = pass(M.L, mode), rows = [];
          st.forEach((x) => { const p = softmax(M.L[x.inp]); M.L[x.inp] = M.L[x.inp].map((v, i) => v - ETA * (p[i] - (i === x.ti ? 1 : 0))); rows.push(x.inp); });
          M.changed = [...new Set(rows)]; M.n++;
        }
        pos = N; render();
      }

      function arrow(g, d, color, op = 1) {
        g.append(s("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-opacity": op }));
      }
      function draw(st) {
        const W = 720, H = 300, x = (k) => 150 + k * 160, yT = 26, yI = 84, yB = 112, rowH = 15, yP = 236;
        const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "解碼器每一步的輸入、預測與正解", style: `display:block;width:100%;min-width:560px;max-width:${W}px` });
        svg.append(s("text", { x: 8, y: yT + 5, "font-size": 12, "font-weight": 700, text: "正解（訓練資料）" }));
        svg.append(s("text", { x: 8, y: yI + 5, "font-size": 12, "font-weight": 700, text: "這一步的輸入" }));
        svg.append(s("text", { x: 8, y: yB + 50, "font-size": 12, "font-weight": 700, text: "模型給的機率" }));
        svg.append(s("text", { x: 8, y: yB + 66, "font-size": 11, class: "t-muted", text: "（下一個詞）" }));
        svg.append(s("text", { x: 8, y: yP + 5, "font-size": 12, "font-weight": 700, text: "模型的預測" }));
        svg.append(s("text", { x: 8, y: yP + 34, "font-size": 12, "font-weight": 700, text: "這一步的損失" }));
        for (let k = 0; k < N; k++) {
          const X = x(k), on = k < pos, cur = k === pos - 1, S = st[k];
          // 正解
          svg.append(s("text", { x: X, y: yT + 5, "text-anchor": "middle", "font-size": 15, class: "t-mono", "font-weight": 700, fill: on ? "var(--ok)" : "var(--faint)", text: TGT[k] }));
          // 輸入來源的箭頭
          if (k > 0 && on) {
            const src = st[k].inpFrom;
            if (src === "tf") arrow(svg, `M${x(k - 1) + 22},${yT + 9} C${x(k - 1) + 70},${yT + 30} ${X - 60},${yI - 30} ${X - 46},${yI - 12}`, "var(--ok)");
            else arrow(svg, `M${x(k - 1) + 58},${yP} C${x(k - 1) + 92},${yP} ${X - 92},${yI} ${X - 58},${yI}`, "var(--neg)");
            const ex = src === "tf" ? X - 46 : X - 58, ey = src === "tf" ? yI - 12 : yI;
            const ang = src === "tf" ? Math.atan2(18, 14) : 0;
            svg.append(s("path", { d: `M${ex},${ey} L${ex - 8 * Math.cos(ang - 0.45)},${ey - 8 * Math.sin(ang - 0.45)} L${ex - 8 * Math.cos(ang + 0.45)},${ey - 8 * Math.sin(ang + 0.45)} Z`, fill: src === "tf" ? "var(--ok)" : "var(--neg)" }));
          }
          // 輸入
          const inpWrong = on && S.inp !== (k ? TGT[k - 1] : "〈開始〉");
          svg.append(s("rect", { x: X - 56, y: yI - 14, width: 112, height: 28, rx: 7, fill: inpWrong ? "var(--neg-soft)" : "var(--panel-2)", stroke: cur ? "var(--hot)" : inpWrong ? "var(--neg)" : "var(--line-strong)", "stroke-width": cur ? 2.5 : 1 }));
          svg.append(s("text", { x: X, y: yI + 5, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: on ? "var(--ink)" : "var(--faint)", text: on ? S.inp : "?" }));
          // 機率長條
          V.forEach((w, i) => {
            const y = yB + i * rowH, p = on ? S.p[i] : 0, isT = i === S.ti, isP = on && w === S.pred;
            svg.append(s("text", { x: X - 18, y: y + 10.5, "text-anchor": "end", "font-size": 10.5, class: "t-mono", fill: on ? (isT ? "var(--ok)" : "var(--muted)") : "var(--faint)", "font-weight": isT && on ? 700 : 400, text: w === "〈結束〉" ? "〈結〉" : w }));
            svg.append(s("rect", { x: X - 14, y: y + 2, width: 50, height: rowH - 4, fill: "var(--panel-2)" }));
            if (on) svg.append(s("rect", { x: X - 14, y: y + 2, width: 50 * p, height: rowH - 4, fill: isP ? (S.ok ? "var(--ok)" : "var(--neg)") : "var(--accent)", "fill-opacity": isP ? 1 : 0.55 }));
            if (on) svg.append(s("text", { x: X + 40, y: y + 10.5, "font-size": 10, class: "t-mono", fill: isP ? "var(--ink)" : "var(--muted)", "font-weight": isP ? 700 : 400, text: Math.round(p * 100) + "%" }));
          });
          // 預測
          svg.append(s("rect", { x: X - 56, y: yP - 14, width: 112, height: 28, rx: 7, fill: on ? (S.ok ? "var(--panel)" : "var(--neg-soft)") : "var(--panel-2)", stroke: on ? (S.ok ? "var(--ok)" : "var(--neg)") : "var(--line-strong)", "stroke-width": on ? 1.6 : 1 }));
          svg.append(s("text", { x: X, y: yP + 5, "text-anchor": "middle", "font-size": 14, class: "t-mono", "font-weight": 700, fill: on ? (S.ok ? "var(--ok)" : "var(--neg)") : "var(--faint)", text: on ? S.pred + (S.ok ? " ✓" : " ✗") : "?" }));
          if (on) svg.append(s("text", { x: X, y: yP + 34, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: S.loss > 2 ? "var(--neg)" : "var(--ink)", "font-weight": 700, text: fmt(S.loss, 2) }));
        }
        svgWrap.replaceChildren(svg);
      }

      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      function render() {
        const M = models[mode], st = pass(M.L, mode);
        draw(st);
        prevBtn.disabled = pos === 0; nextBtn.disabled = pos >= N;
        const name = mode === "tf" ? "教師強制" : "自由生成";
        let kd = "info", txt;
        if (pos === 0) txt = `還沒開始：按「下一步」看解碼器一步一步產生譯文（${name}${M.n ? `，這份模型已更新 ${M.n} 次` : "，模型是訓練到一半的狀態"}）`;
        else if (pos < N) {
          const S = st[pos - 1], nextIn = mode === "tf" ? TGT[pos - 1] : S.pred;
          txt = `第 ${pos} 步：輸入「${S.inp}」→ 模型最想說「${S.pred}」，正解是「${TGT[pos - 1]}」${S.ok ? "，答對" : "，答錯"}。下一步的輸入：${mode === "tf" ? `正解「${nextIn}」` : `自己的預測「${nextIn}」`}`;
          kd = S.ok ? "info" : mode === "tf" ? "warn" : "bad";
        } else {
          const tot = st.reduce((a, x) => a + x.loss, 0), wrong = st.filter((x) => !x.ok).length;
          const firstWrong = st.findIndex((x) => !x.ok);
          if (!wrong) { kd = "ok"; txt = `4 步全對，損失合計 ${fmt(tot, 2)}`; }
          else if (mode === "tf") { kd = "warn"; txt = `4 步中錯 ${wrong} 步，損失合計 ${fmt(tot, 2)}；第 ${firstWrong + 1} 步錯了，但後面的輸入照樣是正解，所以錯誤沒有往後傳`; }
          else { kd = "bad"; txt = `4 步中錯 ${wrong} 步，損失合計 ${fmt(tot, 2)}；第 ${firstWrong + 1} 步錯了以後，後面每一步的輸入都是錯的，損失在教「${st[firstWrong + 1] ? st[firstWrong + 1].inp : ""} 之後要接 ${TGT[firstWrong + 1] || ""}」這種本來就不對的事`; }
        }
        status.className = "status status--" + kd; status.textContent = txt;

        if (pos === 0) calc.innerHTML = row("這個簡化的解碼器", "只看前一個詞來猜下一個詞：P(下一個詞 | 前一個詞) = softmax(這個詞那一列的分數)") + row("每一步的損失", "−ln P(正解的詞)；猜對而且很有把握時接近 0");
        else {
          const S = st[pos - 1], sc = M.L[S.inp], ex = sc.map(Math.exp), z = ex.reduce((a, b) => a + b);
          calc.innerHTML = row(`第 ${pos} 步，輸入「${S.inp}」那一列的分數`, V.map((w, i) => `${w} ${fmt(sc[i], 2)}`).join("　")) +
            row("softmax", `P(${TGT[pos - 1]}) = e^${fmt(sc[S.ti], 2)} ÷ ${fmt(z, 2)} = <span class="hl">${fmt(S.p[S.ti], 3)}</span>`) +
            row("損失", `−ln ${fmt(S.p[S.ti], 3)} = <span class="hl">${fmt(S.loss, 3)}</span>`) +
            (pos === N ? row("4 步合計", `${st.map((x) => fmt(x.loss, 2)).join(" + ")} = <span class="hl">${fmt(st.reduce((a, x) => a + x.loss, 0), 2)}</span>`) : "");
        }

        // 比較表
        const info = (k) => { const L = models[k].L; const pc = pass(L, "tf").reduce((a, x) => a * x.p[x.ti], 1); return { n: models[k].n, gen: generate(L).join(" "), pc, junk: softmax(L.am)[3] }; };
        const A = info("tf"), B = info("fr");
        cmpWrap.replaceChildren(h("table", { class: "tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, ""), h("th", null, "更新次數"), h("th", { class: "l" }, "自己生成的句子"), h("th", null, "正解整句的機率"), h("th", null, "P(cats | am)"))),
          h("tbody", null, ...[["教師強制", A, "tf"], ["自由生成", B, "fr"]].map(([nm, r, k]) => h("tr", { class: k === mode ? "is-sel" : "" },
            h("td", { class: "l" }, nm), h("td", null, r.n), h("td", { class: "l" }, r.gen), h("td", null, fmt(r.pc * 100, 1) + "%"), h("td", null, fmt(r.junk * 100, 1) + "%"))))));
        // 模型表
        const used = new Set(st.slice(0, pos).map((x) => x.inp + "|" + x.ti));
        const panel = ctx.color("--panel"), acc = ctx.color("--accent");
        heatWrap.replaceChildren(h("table", { class: "tf-heat" },
          h("thead", null, h("tr", null, h("th", { class: "row" }, "前一個詞 ↓ ／ 下一個 →"), ...V.map((w) => h("th", null, w)))),
          h("tbody", null, ...IN.map((w) => { const p = softmax(M.L[w]); return h("tr", { class: M.changed.includes(w) ? "chg" : "" }, h("th", { class: "row" }, w + (M.changed.includes(w) ? " ●" : "")),
            ...p.map((v, i) => h("td", { class: used.has(w + "|" + i) ? "used" : "", style: { background: Atlas.rgbStr(Atlas.mix(panel, acc, Math.min(1, v * 1.1))), color: v > 0.5 ? "var(--panel)" : "var(--ink)" } }, Math.round(v * 100) + "%"))); }))));
      }

      note.innerHTML = `為了看得清楚，這個解碼器只看前一個詞，分數是設計好的示範值；真實的解碼器（<a href="#seq2seq">Seq2Seq</a> 的 RNN 或 <a href="#transformer">Transformer</a>）還會看原文與整段前文，數字全由訓練學到。自由生成這邊，「am 之後接 cats」的機率會被一路推高（看比較表最右欄），這就是後面的步驟在學錯的東西；而教師強制下，am 那一列根本不會被用到。訓練只看過正解前文、推論時卻要接自己的預測，兩者的落差叫 <span class="en">exposure bias</span>；實務上有時會在訓練後期逐漸改餵自己的預測來縮小這個落差。Transformer 訓練時也用教師強制：把整句正解一次餵進去，靠遮罩（mask）讓每個位置只看得到前面的詞，所以所有位置能平行計算。`;

      render();
      ctx.onTheme(render);
      return { destroy() { player.pause(); } };
    },
  });
})();
