/* Multi-Head Attention（シラバス §23）
   沿用 Self-Attention 頁的句子與 4 維向量（生物／代名詞／動作／狀態）。
   2 個頭、每頭 2 維：頭 1 的 K・V 只看「生物・代名詞」（找指的是誰），頭 2 的 K・V 只看「動作・狀態」（找它怎麼了）。
   對照「只用 1 個頭」：同樣的 Q・K 參數放進同一個 softmax，兩種關係得搶同一份 100%。 */
Atlas.define("mha", {
  lead: `<p>Self-Attention 讓每個詞只分配<b>一組</b>注意力權重，但一個詞常同時需要好幾種關係：「it」要知道自己指的是誰（cat），也要知道它怎麼了（tired）。一組加起來 100% 的權重只能分一次。<span class="jp">Multi-Head Attention</span>（多頭注意力）準備好幾組 Q・K・V，每一組叫一個<span class="jp">ヘッド</span>（頭，<span class="en">head</span>），各自算注意力，最後再合起來。</p>`,
  closing: `每個頭在自己較小的空間裡找一種關係，不必和別的關係搶同一份 100%；把各頭的結果串接、再乘上 W<sub>O</sub> 混合，輸出就同時帶著多種關係的資訊。Transformer 原論文用 8 個頭，把 512 維分成 8 份、每頭 64 維，總計算量和單頭差不多。`,

  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const TOK = ["The", "cat", "sat", "because", "it", "was", "tired"];
    const DIM = ["生物", "代名詞", "動作", "狀態"];
    const E = [[0.2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0.2, 0.2], [0.3, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1]];
    const mul = (v, M) => M[0].map((_, j) => v.reduce((a, x, i) => a + x * M[i][j], 0));
    const dot = (a, b) => a.reduce((t, x, i) => t + x * b[i], 0);
    const softmax = (r) => { const e = r.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); };
    const HEADS = [
      { name: "頭 1", role: "找指的是誰", sub: ["生物", "代名詞"], color: "var(--hot)",
        WQ: [[0, 0], [5, 0], [3, 0], [3, 0]], WK: [[1, 0], [0, 1], [0, 0], [0, 0]] },
      { name: "頭 2", role: "找它怎麼了", sub: ["動作", "狀態"], color: "var(--pos)",
        WQ: [[0, 3], [0, 5], [0, 0], [0, 0]], WK: [[0, 0], [0, 0], [1, 0], [0, 1]] },
    ];
    function runHead(WQ, WK, dk) {
      const Q = E.map((e) => mul(e, WQ)), K = E.map((e) => mul(e, WK)), V = K; // 此例 W_V = W_K
      const SC = Q.map((q) => K.map((k) => dot(q, k) / Math.sqrt(dk)));
      const A = SC.map(softmax);
      const Z = A.map((w) => K[0].map((_, d) => w.reduce((a, wj, j) => a + wj * V[j][d], 0)));
      return { Q, K, V, SC, A, Z };
    }
    const R = HEADS.map((hd) => runHead(hd.WQ, hd.WK, 2));
    // 單頭：把兩頭的 W_Q、W_K 並排成 4×4（同樣的參數），一次 softmax，除以 √4
    const WQs = HEADS[0].WQ.map((r, i) => r.concat(HEADS[1].WQ[i]));
    const WKs = HEADS[0].WK.map((r, i) => r.concat(HEADS[1].WK[i]));
    const S1 = runHead(WQs, WKs, 4);
    const MH = E.map((_, i) => R[0].Z[i].concat(R[1].Z[i])); // 串接（W_O＝單位矩陣）

    let mode = "multi", sel = 4, showMat = false;
    const vec = (v, d = 2) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
    const pct = (w) => Math.round(w * 100) + "%";

    bench.append(h("style", { text: `
      .mh-toks { display: flex; flex-wrap: wrap; gap: 8px; }
      .mh-tok { min-height: 40px; padding: 0 14px; border-radius: 8px; border: 1.5px solid var(--line-strong); background: var(--panel-2); font-family: var(--f-mono); font-size: 15px; cursor: pointer; }
      .mh-tok:hover { border-color: var(--accent); }
      .mh-tok[aria-pressed="true"] { background: var(--ink); border-color: var(--ink); color: var(--panel); font-weight: 600; }
      .mh-w td { min-width: 52px; }
      .mh-out td.big { font-weight: 700; color: var(--hot); }
      .mh-mats { display: flex; flex-wrap: wrap; gap: 14px; }
      .mh-mat { border-collapse: collapse; font-family: var(--f-mono); font-size: 12px; }
      .mh-mat td, .mh-mat th { border: 1px solid var(--line); padding: 2px 8px; text-align: right; }
      .mh-mat th { color: var(--muted); font-weight: 500; font-family: var(--f-body); }
      .mh-mat caption { font-size: 12.5px; font-weight: 700; text-align: left; padding-bottom: 4px; }
    ` }));

    const modeSeg = ui.seg({ label: "注意力的組數", value: mode,
      options: [{ value: "multi", label: "2 個頭（Multi-Head）" }, { value: "single", label: "只用 1 個頭（同樣的參數）" }],
      onChange: (v) => { mode = v; render(); } });
    const toks = h("div", { class: "mh-toks", role: "group", "aria-label": "選擇查詢的詞" });
    const tokBtns = TOK.map((t, i) => { const b = h("button", { type: "button", class: "mh-tok", "aria-pressed": "false" }, t); b.addEventListener("click", () => { sel = i; render(); }); toks.append(b); return b; });
    const status = h("div", { class: "status status--info" });
    const svgWrap = h("div", { class: "scroll-x" });
    const wWrap = h("div");
    const outWrap = h("div");
    const calc = h("div", { class: "calc" });
    const matBtn = ui.btn("看各頭的矩陣", () => { showMat = !showMat; render(); }, { sm: true });
    const matBox = h("div", { class: "mh-mats", hidden: true });
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el),
      h("div", { class: "field" }, h("span", { class: "field__label" }, "點任一個詞，看它的注意力怎麼分"), toks),
      status, svgWrap, wWrap, outWrap, calc,
      h("div", null, matBtn), matBox, note,
    );

    function draw() {
      const multi = mode === "multi";
      const W = 760, H = multi ? 300 : 186, yT = 150, x = (i) => 56 + i * 108;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `「${TOK[sel]}」在各頭的注意力`, style: `display:block;width:100%;min-width:${W * 0.74}px;max-width:${W}px` });
      const sets = multi
        ? [{ A: R[0].A[sel], color: HEADS[0].color, up: true, label: `頭 1：${HEADS[0].role}` }, { A: R[1].A[sel], color: HEADS[1].color, up: false, label: `頭 2：${HEADS[1].role}` }]
        : [{ A: S1.A[sel], color: "var(--muted)", up: true, label: "唯一的一個頭" }];
      sets.forEach((st) => {
        const yEdge = st.up ? yT - 46 : yT + 46, dir = st.up ? -1 : 1;
        svg.append(s("text", { x: 6, y: st.up ? 16 : H - 6, "font-size": 12, "font-weight": 700, fill: st.color, text: st.label }));
        st.A.forEach((w, j) => {
          const sw = 1 + 14 * w, op = 0.2 + 0.8 * w;
          const d = j === sel
            ? `M${x(j) - 12},${yEdge} C${x(j) - 34},${yEdge + dir * 44} ${x(j) + 34},${yEdge + dir * 44} ${x(j) + 12},${yEdge}`
            : `M${x(sel)},${yEdge} Q${(x(sel) + x(j)) / 2},${yEdge + dir * Math.min(96, 30 + Math.abs(j - sel) * 18)} ${x(j)},${yEdge}`;
          svg.append(s("path", { d, fill: "none", stroke: st.color, "stroke-width": sw, "stroke-opacity": op, "stroke-linecap": "round" }));
        });
        st.A.forEach((w, j) => {
          const yp = st.up ? yT - 34 : yT + 34;
          svg.append(s("rect", { x: x(j) - 24, y: yp - 10, width: 48, height: 20, rx: 10, fill: "var(--panel)", stroke: st.color, "stroke-width": 1.2 }));
          svg.append(s("text", { x: x(j), y: yp + 4.5, "text-anchor": "middle", "font-size": 12, "font-weight": 700, class: "t-mono", fill: st.color, text: pct(w) }));
        });
      });
      TOK.forEach((t, j) => {
        const isSel = j === sel;
        svg.append(s("rect", { x: x(j) - 46, y: yT - 18, width: 92, height: 36, rx: 8, fill: isSel ? "var(--ink)" : "var(--panel-2)", stroke: isSel ? "var(--ink)" : "var(--line-strong)" }));
        svg.append(s("text", { x: x(j), y: yT + 5, "text-anchor": "middle", "font-size": 14.5, class: "t-mono", fill: isSel ? "var(--panel)" : "var(--ink)", "font-weight": isSel ? 700 : 400, text: t }));
      });
      svgWrap.replaceChildren(svg);
    }

    function topOf(A) { const o = A.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]); return o[0]; }

    function render() {
      tokBtns.forEach((b, j) => b.setAttribute("aria-pressed", String(j === sel)));
      draw();
      const multi = mode === "multi", t = TOK[sel];
      if (multi) {
        const [w1, j1] = topOf(R[0].A[sel]), [w2, j2] = topOf(R[1].A[sel]);
        const say = (w, j) => (w > 0.3 ? `最關注「${TOK[j]}」（${pct(w)}）` : `分得很平均（最高「${TOK[j]}」${pct(w)}）`);
        status.className = "status status--ok";
        status.textContent = `「${t}」：頭 1 ${say(w1, j1)}，頭 2 ${say(w2, j2)}——兩個頭各分各的 100%`;
      } else {
        const A = S1.A[sel], [w1, j1] = topOf(A);
        const o = A.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]);
        status.className = "status status--warn";
        status.textContent = o[1][0] > 0.15
          ? `「${t}」：只有一組權重，「${TOK[o[0][1]]}」（${pct(o[0][0])}）和「${TOK[o[1][1]]}」（${pct(o[1][0])}）得分同一份 100%，兩邊都被稀釋`
          : `「${t}」：只有一組權重，最高是「${TOK[j1]}」（${pct(w1)}）`;
      }
      // weights of the selected word
      const rows = multi
        ? HEADS.map((hd, k) => h("tr", null, h("td", { class: "l", style: { color: hd.color, fontWeight: 700 } }, `${hd.name}（${hd.role}）`), ...R[k].A[sel].map((v) => h("td", null, pct(v))), h("td", null, "100%")))
        : [h("tr", null, h("td", { class: "l", style: { fontWeight: 700 } }, "唯一的頭"), ...S1.A[sel].map((v) => h("td", null, pct(v))), h("td", null, "100%"))];
      wWrap.replaceChildren(h("div", { class: "field" }, h("span", { class: "field__label" }, `「${t}」分給各詞的注意力`),
        h("div", { class: "scroll-x" }, h("table", { class: "tbl mh-w" }, h("thead", null, h("tr", null, h("th", { class: "l" }, ""), ...TOK.map((w) => h("th", null, w)), h("th", null, "合計"))), h("tbody", null, ...rows)))));
      // output comparison
      const out = multi ? MH[sel] : S1.Z[sel];
      const otherOut = multi ? S1.Z[sel] : MH[sel];
      outWrap.replaceChildren(h("div", { class: "field" }, h("span", { class: "field__label" }, `「${t}」注意力之後的向量（4 格＝${DIM.join("／")}）`),
        h("div", { class: "scroll-x" }, h("table", { class: "tbl mh-out" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, ""), ...DIM.map((d) => h("th", null, d)))),
          h("tbody", null,
            h("tr", null, h("td", { class: "l" }, "原本的向量"), ...E[sel].map((v) => h("td", null, fmt(v, 2)))),
            h("tr", { class: "is-sel" }, h("td", { class: "l" }, multi ? "2 個頭（串接）" : "1 個頭"), ...out.map((v) => h("td", null, fmt(v, 2)))),
            h("tr", null, h("td", { class: "l", style: { color: "var(--muted)" } }, multi ? "對照：1 個頭" : "對照：2 個頭"), ...otherOut.map((v) => h("td", { style: { color: "var(--muted)" } }, fmt(v, 2)))))))));
      // calc
      if (multi) {
        calc.innerHTML = HEADS.map((hd, k) => {
          const r = R[k];
          return `<span class="calc-row"><span class="k">${hd.name}（キー・バリュー只取「${hd.sub.join("・")}」2 維）</span> q = ${vec(r.Q[sel], 1)}　分數 q·k ÷ √2：${TOK.map((w, j) => `${w} ${fmt(r.SC[sel][j], 2)}`).join("　")}</span>` +
            `<span class="calc-row">　→ softmax 後加權 v：z${k + 1} = <span class="hl">${vec(r.Z[sel])}</span></span>`;
        }).join("") +
          `<span class="calc-row"><span class="k">串接</span> [z1, z2] = ${vec(MH[sel])}　<span class="k">× W<sub>O</sub>（此例設成不改變）</span> = <span class="hl">${vec(MH[sel])}</span></span>`;
      } else {
        calc.innerHTML =
          `<span class="calc-row"><span class="k">同樣的參數，Q・K 都是 4 維</span> q = ${vec(S1.Q[sel], 1)}</span>` +
          `<span class="calc-row"><span class="k">分數 q·k ÷ √4</span> ${TOK.map((w, j) => `${w} ${fmt(S1.SC[sel][j], 2)}`).join("　")}</span>` +
          `<span class="calc-row"><span class="k">一次 softmax 後加權 v</span> z = <span class="hl">${vec(S1.Z[sel])}</span></span>`;
      }
      // matrices
      matBox.hidden = !showMat; matBtn.textContent = showMat ? "收起矩陣" : "看各頭的矩陣";
      if (showMat) {
        const mat = (cap, M, cols) => h("table", { class: "mh-mat" }, h("caption", null, cap),
          h("tr", null, h("th", null, ""), ...cols.map((c) => h("th", null, "→" + c))),
          ...M.map((r, i) => h("tr", null, h("th", null, DIM[i]), ...r.map((v) => h("td", null, fmt(v, 0))))));
        matBox.replaceChildren(
          mat("頭 1 的 W_Q", HEADS[0].WQ, HEADS[0].sub), mat("頭 1 的 W_K（＝W_V）", HEADS[0].WK, HEADS[0].sub),
          mat("頭 2 的 W_Q", HEADS[1].WQ, HEADS[1].sub), mat("頭 2 的 W_K（＝W_V）", HEADS[1].WK, HEADS[1].sub));
      }
      note.innerHTML = `每個頭用自己的 W<sub>Q</sub>・W<sub>K</sub>・W<sub>V</sub> 把 4 維向量投影到 2 維：頭 1 的キー只留下「生物・代名詞」，頭 2 只留下「動作・狀態」，所以兩個頭自然看向不同的詞。「只用 1 個頭」是把同樣的矩陣並排成 4 維、放進同一個 softmax。這些矩陣是為了示範而設計的（W<sub>V</sub> 取和 W<sub>K</sub> 相同、W<sub>O</sub> 設成不改變），真實模型全由訓練學到，每個頭學到什麼關係也不是人指定的。單頭的計算見 <a href="#selfatt">Self-Attention</a>；Multi-Head Attention 在整個模型裡的位置見 <a href="#transformer">Transformer</a>。`;
    }
    render();
    ctx.onTheme(render);
    return {};
  },
});
