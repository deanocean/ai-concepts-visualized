/* Self-Attention（シラバス §23）
   7 個 token、4 維的示範向量；點任一詞看注意力權重，展開 Q・K → softmax → 加權 V 的計算。 */
Atlas.define("selfatt", {
  lead: `<p>讀到「it」時，要知道它指的是誰，就得回頭看句中其他詞。<span class="jp">Self-Attention</span>（自注意力）讓每個詞自己算出「該多關注句中的每個詞」，再按這個比例，把那些詞的資訊混進自己的向量。</p>`,
  closing: `每個詞都同時對句中所有詞算一次注意力，不必像 RNN 一格格往後傳，所以距離再遠也能直接連上，而且可以平行計算；把這個機制層層疊起來，就是 Transformer 的核心。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const TOK = ["The", "cat", "sat", "because", "it", "was", "tired"];
    const DIM = ["生物", "代名詞", "動作", "狀態"];
    // 示範用詞向量（4 維，每一維代表一種性質）
    const E = [[0.2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0.2, 0.2], [0.3, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1]];
    // W_Q：代名詞、動作、狀態 → 去找「生物」；生物 → 去找「動作／狀態」
    const WQ = [[0, 0, 2, 1], [6, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0]];
    const I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    const WK = I4, WV = I4, D = 4, SQ = Math.sqrt(D);
    const mul = (v, M) => M[0].map((_, j) => v.reduce((a, x, i) => a + x * M[i][j], 0));
    const dot = (a, b) => a.reduce((s2, x, i) => s2 + x * b[i], 0);
    const Q = E.map((e) => mul(e, WQ)), Kv = E.map((e) => mul(e, WK)), V = E.map((e) => mul(e, WV));
    const SC = Q.map((q) => Kv.map((k) => dot(q, k) / SQ));
    const A = SC.map((row) => { const ex = row.map(Math.exp), t = ex.reduce((a, b) => a + b); return ex.map((x) => x / t); });
    const Z = A.map((w) => [0, 1, 2, 3].map((j) => w.reduce((a, wi, i) => a + wi * V[i][j], 0)));

    let sel = 4, showSteps = false, step = 1;
    const vec = (v, d = 2) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
    const pct = (w) => (w * 100).toFixed(0) + "%";

    bench.append(h("style", { text: `
      .sa-toks { display: flex; flex-wrap: wrap; gap: 8px; }
      .sa-tok { min-height: 40px; padding: 0 14px; border-radius: 8px; border: 1.5px solid var(--line-strong); background: var(--panel-2); font-family: var(--f-mono); font-size: 15px; cursor: pointer; }
      .sa-tok:hover { border-color: var(--accent); }
      .sa-tok[aria-pressed="true"] { background: var(--hot); border-color: var(--hot); color: var(--panel); font-weight: 600; }
      .sa-heat { border-collapse: separate; border-spacing: 3px; font-family: var(--f-mono); font-size: 12px; }
      .sa-heat th { font-weight: 500; color: var(--muted); padding: 2px 4px; white-space: nowrap; }
      .sa-heat th.row { text-align: right; }
      .sa-heat td { width: 46px; height: 32px; text-align: center; border-radius: 4px; cursor: pointer; }
      .sa-heat tr.is-sel th { color: var(--hot); font-weight: 700; }
      .sa-heat tr.is-sel td { outline: 2px solid var(--hot); outline-offset: -1px; }
      .sa-steps { border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 10px; background: var(--panel-2); }
      .sa-steps p { margin: 0; font-size: 14px; }
      .sa-mat { border-collapse: collapse; font-family: var(--f-mono); font-size: 12px; }
      .sa-mat td, .sa-mat th { border: 1px solid var(--line); padding: 2px 8px; text-align: right; }
      .sa-mat th { color: var(--muted); font-weight: 500; font-family: var(--f-body); }
      .sa-cmp { display: grid; grid-template-columns: auto repeat(4, minmax(48px, auto)); gap: 4px 10px; align-items: center; font-size: 13px; }
      .sa-cmp .mono { text-align: right; }
      .sa-cmp .hd { color: var(--muted); font-size: 12px; text-align: right; }
    ` }));

    /* ---------- elements ---------- */
    const toks = h("div", { class: "sa-toks", role: "group", "aria-label": "選擇查詢的詞" });
    const tokBtns = TOK.map((t, i) => { const b = h("button", { type: "button", class: "sa-tok", "aria-pressed": "false" }, t); b.addEventListener("click", () => select(i)); toks.append(b); return b; });
    const diagWrap = h("div", { class: "scroll-x" });
    const status = h("div", { class: "status status--info" });
    const heatWrap = h("div", { class: "scroll-x" });
    const toggle = ui.btn("展開計算步驟（Q・K → softmax → 加權 V）", () => { showSteps = !showSteps; renderSteps(); }, { id: "sa-toggle" });
    const stepsBox = h("div", { class: "sa-steps", hidden: true });

    bench.append(
      h("div", { class: "field" }, h("span", { class: "field__label" }, "點任一個詞，看它把注意力分給句中哪些詞"), toks),
      diagWrap, status,
      h("div", { class: "field" }, h("span", { class: "field__label" }, "整句的注意力權重表（每一列＝該詞分出去的注意力，總和 100%；點一列可切換）"), heatWrap),
      h("p", { class: "note", html: `例子中的向量和矩陣是為了示範而設計的小數字；真實模型有數百維，數字全是訓練學到的。這裡沒有放入詞的順序（打亂詞序結果不變），要靠 <a href="#posenc">位置エンコーディング</a> 補上；同時用好幾組權重看不同關係，見 <a href="#mha">Multi-Head Attention</a>；翻譯時從譯文回頭看原文，見 <a href="#seq2seq">Source-Target Attention</a>；這些零件怎麼組成完整的模型，見 <a href="#transformer">Transformer</a>。` }),
      h("div", null, toggle), stepsBox,
    );

    /* ---------- attention arcs ---------- */
    function renderDiag() {
      const W = 700, H = 200, base = 160, x = (i) => 50 + i * 100;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `「${TOK[sel]}」對各詞的注意力`, style: "display:block;width:100%;min-width:540px;max-width:760px" });
      const w = A[sel];
      w.forEach((wi, j) => {
        const op = 0.18 + 0.82 * wi, sw = 1 + 16 * wi;
        if (j === sel) {
          svg.append(s("path", { d: `M${x(j) - 10},${base - 22} C${x(j) - 34},${base - 74} ${x(j) + 34},${base - 74} ${x(j) + 10},${base - 22}`, fill: "none", stroke: "var(--hot)", "stroke-width": sw, "stroke-opacity": op, "stroke-linecap": "round" }));
        } else {
          const dx = Math.abs(x(j) - x(sel)), top = base - 30 - Math.min(110, 26 + dx * 0.28);
          svg.append(s("path", { d: `M${x(sel)},${base - 22} Q${(x(sel) + x(j)) / 2},${top} ${x(j)},${base - 22}`, fill: "none", stroke: "var(--hot)", "stroke-width": sw, "stroke-opacity": op, "stroke-linecap": "round" }));
        }
      });
      TOK.forEach((t, j) => {
        const isSel = j === sel;
        svg.append(s("rect", { x: x(j) - 42, y: base - 20, width: 84, height: 34, rx: 8, fill: isSel ? "var(--hot)" : "var(--panel)", stroke: isSel ? "var(--hot)" : "var(--line-strong)" }));
        svg.append(s("text", { x: x(j), y: base + 3, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: isSel ? "var(--panel)" : "var(--ink)", "font-weight": isSel ? 700 : 400, text: t }));
        svg.append(s("text", { x: x(j), y: base + 32, "text-anchor": "middle", "font-size": 13, "font-weight": 700, class: "t-mono", fill: "var(--hot)", text: pct(w[j]) }));
      });
      diagWrap.replaceChildren(svg);
      const order = w.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]);
      const [w1, j1] = order[0];
      status.textContent = w1 > 0.3
        ? `「${TOK[sel]}」最關注「${TOK[j1]}」（${pct(w1)}）`
        : `「${TOK[sel]}」的注意力分得很平均（最高是「${TOK[j1]}」的 ${pct(w1)}）`;
    }

    /* ---------- heatmap ---------- */
    function renderHeat() {
      const tbl = h("table", { class: "sa-heat" });
      const head = h("tr", null, h("th", { class: "row", html: "看的詞 ↓ ／ 被看的詞 →" }), ...TOK.map((t) => h("th", null, t)));
      tbl.append(h("thead", null, head));
      const body = h("tbody");
      A.forEach((row, i) => {
        const tr = h("tr", { class: i === sel ? "is-sel" : "" }, h("th", { class: "row" }, TOK[i]));
        row.forEach((v) => {
          const td = h("td", { style: { background: Atlas.rgbStr(Atlas.mix(Atlas.color("--panel"), Atlas.color("--hot"), Math.min(1, v * 1.3))), color: v > 0.45 ? "var(--panel)" : "var(--ink)" } }, pct(v));
          td.addEventListener("click", () => select(i));
          tr.append(td);
        });
        body.append(tr);
      });
      tbl.append(body);
      heatWrap.replaceChildren(tbl);
    }

    /* ---------- steps ---------- */
    const stepSeg = ui.seg({ value: step, options: [
      { value: 1, label: "① Q・K・V" }, { value: 2, label: "② 分數 q·k" }, { value: 3, label: "③ softmax" }, { value: 4, label: "④ 加權 V" }],
      onChange: (v) => { step = v; renderSteps(); } });
    const stepBody = h("div", { style: { display: "flex", flexDirection: "column", gap: "10px" } });
    const nextStep = ui.btn("下一步 ▶", () => { step = step % 4 + 1; stepSeg.set(step); renderSteps(); }, { primary: true, sm: true });
    stepsBox.append(h("div", { class: "bench__row" }, stepSeg.el, nextStep), stepBody);

    function nzTerms(a, b) {
      const t = a.map((x, i) => [x, b[i]]).filter(([x, y]) => x !== 0 && y !== 0);
      return t.length ? t.map(([x, y]) => `${fmt(x, 1)}×${fmt(y, 1)}`).join(" + ") : "0";
    }
    const dimHead = () => h("tr", null, h("th", { class: "l" }, ""), ...DIM.map((d) => h("th", null, d)));

    function renderSteps() {
      stepsBox.hidden = !showSteps;
      toggle.textContent = showSteps ? "收起計算步驟" : "展開計算步驟（Q・K → softmax → 加權 V）";
      if (!showSteps) return;
      stepSeg.set(step);
      const t = TOK[sel];
      const parts = [];
      if (step === 1) {
        parts.push(h("p", { html: `每個詞先是一個 4 維向量，每一維代表一種性質（${DIM.join("／")}）。它分別乘上三個矩陣，變成三個角色：<span class="jp">クエリ</span>（查詢，<span class="en">query</span>）q＝「我在找什麼」、<span class="jp">キー</span>（鍵，<span class="en">key</span>）k＝「我是什麼，給別人比對」、<span class="jp">バリュー</span>（值，<span class="en">value</span>）v＝「被選中時我提供的內容」。` }));
        const m = h("table", { class: "sa-mat" }, h("tr", null, h("th", null, "W_Q"), ...DIM.map((d) => h("th", null, "→" + d))),
          ...WQ.map((r, i) => h("tr", null, h("th", null, DIM[i]), ...r.map((v) => h("td", null, fmt(v, 0))))));
        const qTerms = [0, 1, 2, 3].map((j) => { const terms = E[sel].map((x, i) => [x, WQ[i][j]]).filter(([x, w]) => x && w); return terms.length ? `q<sub>${DIM[j]}</sub> = ${terms.map(([x, w]) => `${fmt(x, 1)}×${fmt(w, 0)}`).join(" + ")} = ${fmt(Q[sel][j], 1)}` : `q<sub>${DIM[j]}</sub> = 0`; });
        parts.push(h("div", { class: "bench__row", style: { alignItems: "flex-start" } },
          h("div", { class: "scroll-x" }, m),
          h("div", { class: "calc", style: { flex: "1 1 260px" }, html:
            `<span class="calc-row"><span class="k">「${t}」的向量</span> x = ${vec(E[sel], 1)}</span>` +
            qTerms.map((r) => `<span class="calc-row">${r}</span>`).join("") +
            `<span class="calc-row"><span class="k">クエリ</span> q = x × W_Q = <span class="hl">${vec(Q[sel], 1)}</span></span>` }),
        ));
        parts.push(h("p", { class: "note", html: `W_Q 的意思：代名詞、動作、狀態類的詞，會去「找生物」；生物類的詞，會去「找動作和狀態」。為了好讀，這個例子的 W_K、W_V 設成不改變向量（k = v = x），真實模型三個矩陣都要學。` }));
        const kv = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, "詞"), h("th", null, "キー k"), h("th", null, "バリュー v"))),
          h("tbody", null, ...TOK.map((w, i) => h("tr", { class: i === sel ? "is-sel" : "" }, h("td", { class: "l" }, w), h("td", null, vec(Kv[i], 1)), h("td", null, vec(V[i], 1))))));
        parts.push(h("div", { class: "scroll-x" }, kv));
      } else if (step === 2) {
        parts.push(h("p", { html: `拿「${t}」的 q 和每個詞的 k 做內積（對應維度相乘再加總）：性質越對得上，分數越高。再除以 √d（d＝向量維數 ${D}，√${D} = ${SQ}），避免維數一多分數就變得太大。` }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, "被看的詞"), h("th", null, "k"), h("th", { class: "l" }, `q·k（q = ${vec(Q[sel], 1)}）`), h("th", null, "q·k"), h("th", null, `÷ √${D}`))),
          h("tbody", null, ...TOK.map((w, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, w), h("td", null, vec(Kv[j], 1)), h("td", { class: "l" }, nzTerms(Q[sel], Kv[j])), h("td", null, fmt(dot(Q[sel], Kv[j]), 2)), h("td", null, fmt(SC[sel][j], 2))))));
        parts.push(h("div", { class: "scroll-x" }, tb));
      } else if (step === 3) {
        const ex = SC[sel].map(Math.exp), tot = ex.reduce((a, b) => a + b);
        parts.push(h("p", { html: `分數還不能直接當比例用（可能是負數，總和也不是 1）。<span class="jp">ソフトマックス関数</span>（softmax 函數，<span class="en">softmax function</span>）先對每個分數取 e 的次方（變成正數，而且分數高的被放大），再除以總和，得到加起來剛好 100% 的注意力權重。` }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, "被看的詞"), h("th", null, "分數 s"), h("th", null, "e^s"), h("th", null, "e^s ÷ 總和"), h("th", null, "權重"))),
          h("tbody", null, ...TOK.map((w, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, w), h("td", null, fmt(SC[sel][j], 2)), h("td", null, fmt(ex[j], 2)), h("td", null, `${fmt(ex[j], 2)} ÷ ${fmt(tot, 2)}`), h("td", null, pct(A[sel][j]))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l" }, "總和"), h("td"), h("td", null, fmt(tot, 2)), h("td"), h("td", null, "100%"))));
        parts.push(h("div", { class: "scroll-x" }, tb));
      } else {
        const w = A[sel];
        parts.push(h("p", { html: `最後把每個詞的 v 乘上權重再加總，得到「${t}」的新向量：關注得越多的詞，貢獻越大。` }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, "詞"), h("th", null, "權重"), h("th", null, "v"), ...DIM.map((d) => h("th", null, "權重×v " + d)))),
          h("tbody", null, ...TOK.map((wd, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, wd), h("td", null, pct(w[j])), h("td", null, vec(V[j], 1)), ...V[j].map((x) => h("td", null, fmt(w[j] * x, 3)))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l" }, "加總"), h("td"), h("td"), ...Z[sel].map((x) => h("td", null, fmt(x, 3))))));
        parts.push(h("div", { class: "scroll-x" }, tb));
        const cmp = h("div", { class: "sa-cmp" }, h("span"), ...DIM.map((d) => h("span", { class: "hd" }, d)),
          h("span", null, "原本的向量"), ...E[sel].map((x) => h("span", { class: "mono" }, fmt(x, 2))),
          h("span", { style: { color: "var(--hot)", fontWeight: 700 } }, "注意力之後"), ...Z[sel].map((x) => h("span", { class: "mono", style: { color: "var(--hot)", fontWeight: 700 } }, fmt(x, 2))));
        parts.push(h("div", { class: "scroll-x" }, cmp));
        const diff = Z[sel].map((z, j) => z - E[sel][j]);
        const jUp = diff.indexOf(Math.max(...diff));
        const top = w.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0])[0][1];
        parts.push(h("div", { class: "status status--info", text: diff[jUp] > 0.05
          ? `「${t}」在「${DIM[jUp]}」這一維從 ${fmt(E[sel][jUp], 2)} 變成 ${fmt(Z[sel][jUp], 2)}：它的向量混入了「${TOK[top]}」的資訊。`
          : `「${t}」的新向量是各詞內容的平均混合，沒有特別偏向哪個詞。` }));
      }
      stepBody.replaceChildren(...parts);
    }

    function select(i) {
      sel = i;
      tokBtns.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
      renderDiag(); renderHeat(); renderSteps();
    }
    select(sel);
    ctx.onTheme(() => renderHeat());
    void fmtP;
    return {};
  },
});
