/* Multi-Head Attention（シラバス §23）
   沿用 Self-Attention 頁的句子與 4 維向量（生物／代名詞／動作／狀態）。
   2 個頭、每頭 2 維：頭 1 的 K・V 只看「生物・代名詞」（找指的是誰），頭 2 的 K・V 只看「動作・狀態」（找它怎麼了）。
   對照「只用 1 個頭」：同樣的 Q・K 參數放進同一個 softmax，兩種關係得搶同一份 100%。
   文字在 locales/zh/mha.js、locales/ja/mha.js（4 個維度的名稱 dims 也在語言檔，以空白分隔）。 */
Atlas.define("mha", {
  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const { t } = ctx;
    const TOK = ["The", "cat", "sat", "because", "it", "was", "tired"];
    const DIM = t("dims").split(" ");
    const E = [[0.2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0.2, 0.2], [0.3, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1]];
    const mul = (v, M) => M[0].map((_, j) => v.reduce((a, x, i) => a + x * M[i][j], 0));
    const dot = (a, b) => a.reduce((acc, x, i) => acc + x * b[i], 0);
    const softmax = (r) => { const e = r.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); };
    const HEADS = [
      { name: t("headN", { n: 1 }), role: t("role1"), sub: DIM.slice(0, 2), color: "var(--hot)",
        WQ: [[0, 0], [5, 0], [3, 0], [3, 0]], WK: [[1, 0], [0, 1], [0, 0], [0, 0]] },
      { name: t("headN", { n: 2 }), role: t("role2"), sub: DIM.slice(2, 4), color: "var(--pos)",
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

    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode,
      options: [{ value: "multi", label: t("modeMulti") }, { value: "single", label: t("modeSingle") }],
      onChange: (v) => { mode = v; render(); } });
    const toks = h("div", { class: "mh-toks", role: "group", "aria-label": t("toksAria") });
    const tokBtns = TOK.map((w, i) => { const b = h("button", { type: "button", class: "mh-tok", "aria-pressed": "false" }, w); b.addEventListener("click", () => { sel = i; render(); }); toks.append(b); return b; });
    const status = h("div", { class: "status status--info" });
    const svgWrap = h("div", { class: "scroll-x" });
    const wWrap = h("div");
    const outWrap = h("div");
    const calc = h("div", { class: "calc" });
    const matBtn = ui.btn(t("btnMat"), () => { showMat = !showMat; render(); }, { sm: true });
    const matBox = h("div", { class: "mh-mats", hidden: true });
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el),
      h("div", { class: "field" }, h("span", { class: "field__label" }, t("toksLabel")), toks),
      status, svgWrap, wWrap, outWrap, calc,
      h("div", null, matBtn), matBox, note,
    );

    function draw() {
      const multi = mode === "multi";
      const W = 760, H = multi ? 300 : 186, yT = 150, x = (i) => 56 + i * 108;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("svgAria", { w: TOK[sel] }), style: `display:block;width:100%;min-width:${W * 0.74}px;max-width:${W}px` });
      const sets = multi
        ? [{ A: R[0].A[sel], color: HEADS[0].color, up: true, label: t("headLabel", HEADS[0]) }, { A: R[1].A[sel], color: HEADS[1].color, up: false, label: t("headLabel", HEADS[1]) }]
        : [{ A: S1.A[sel], color: "var(--muted)", up: true, label: t("onlyHead") }];
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
      TOK.forEach((tk, j) => {
        const isSel = j === sel;
        svg.append(s("rect", { x: x(j) - 46, y: yT - 18, width: 92, height: 36, rx: 8, fill: isSel ? "var(--ink)" : "var(--panel-2)", stroke: isSel ? "var(--ink)" : "var(--line-strong)" }));
        svg.append(s("text", { x: x(j), y: yT + 5, "text-anchor": "middle", "font-size": 14.5, class: "t-mono", fill: isSel ? "var(--panel)" : "var(--ink)", "font-weight": isSel ? 700 : 400, text: tk }));
      });
      svgWrap.replaceChildren(svg);
    }

    function topOf(A) { const o = A.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]); return o[0]; }

    function render() {
      tokBtns.forEach((b, j) => b.setAttribute("aria-pressed", String(j === sel)));
      draw();
      const multi = mode === "multi", tk = TOK[sel];
      if (multi) {
        const [w1, j1] = topOf(R[0].A[sel]), [w2, j2] = topOf(R[1].A[sel]);
        const say = (w, j) => t(w > 0.3 ? "sayFocus" : "sayEven", { top: TOK[j], pct: pct(w) });
        status.className = "status status--ok";
        status.textContent = t("statusMulti", { w: tk, h1: say(w1, j1), h2: say(w2, j2) });
      } else {
        const A = S1.A[sel], [w1, j1] = topOf(A);
        const o = A.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]);
        status.className = "status status--warn";
        status.textContent = o[1][0] > 0.15
          ? t("statusSingleSplit", { w: tk, a: TOK[o[0][1]], pa: pct(o[0][0]), b: TOK[o[1][1]], pb: pct(o[1][0]) })
          : t("statusSingle", { w: tk, top: TOK[j1], pct: pct(w1) });
      }
      // weights of the selected word
      const rows = multi
        ? HEADS.map((hd, k) => h("tr", null, h("td", { class: "l", style: { color: hd.color, fontWeight: 700 } }, t("headRow", hd)), ...R[k].A[sel].map((v) => h("td", null, pct(v))), h("td", null, "100%")))
        : [h("tr", null, h("td", { class: "l", style: { fontWeight: 700 } }, t("onlyHeadRow")), ...S1.A[sel].map((v) => h("td", null, pct(v))), h("td", null, "100%"))];
      wWrap.replaceChildren(h("div", { class: "field" }, h("span", { class: "field__label" }, t("wLabel", { w: tk })),
        h("div", { class: "scroll-x" }, h("table", { class: "tbl mh-w" }, h("thead", null, h("tr", null, h("th", { class: "l" }, ""), ...TOK.map((w) => h("th", null, w)), h("th", null, t("thSum")))), h("tbody", null, ...rows)))));
      // output comparison
      const out = multi ? MH[sel] : S1.Z[sel];
      const otherOut = multi ? S1.Z[sel] : MH[sel];
      outWrap.replaceChildren(h("div", { class: "field" }, h("span", { class: "field__label" }, t("outLabel", { w: tk, dims: DIM.join(t("dimSep")) })),
        h("div", { class: "scroll-x" }, h("table", { class: "tbl mh-out" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, ""), ...DIM.map((d) => h("th", null, d)))),
          h("tbody", null,
            h("tr", null, h("td", { class: "l" }, t("rowOrig")), ...E[sel].map((v) => h("td", null, fmt(v, 2)))),
            h("tr", { class: "is-sel" }, h("td", { class: "l" }, t(multi ? "rowMulti" : "rowSingle")), ...out.map((v) => h("td", null, fmt(v, 2)))),
            h("tr", null, h("td", { class: "l", style: { color: "var(--muted)" } }, t(multi ? "rowCmpSingle" : "rowCmpMulti")), ...otherOut.map((v) => h("td", { style: { color: "var(--muted)" } }, fmt(v, 2)))))))));
      // calc
      if (multi) {
        calc.innerHTML = HEADS.map((hd, k) => {
          const r = R[k];
          return `<span class="calc-row"><span class="k">${t("calcHeadK", { name: hd.name, a: hd.sub[0], b: hd.sub[1] })}</span> q = ${vec(r.Q[sel], 1)}　${t("calcScore2")}${TOK.map((w, j) => `${w} ${fmt(r.SC[sel][j], 2)}`).join("　")}</span>` +
            `<span class="calc-row">${t("calcHeadZ", { k: k + 1 })} <span class="hl">${vec(r.Z[sel])}</span></span>`;
        }).join("") +
          `<span class="calc-row"><span class="k">${t("calcConcatK")}</span> [z1, z2] = ${vec(MH[sel])}　<span class="k">${t("calcWoK")}</span> = <span class="hl">${vec(MH[sel])}</span></span>`;
      } else {
        calc.innerHTML =
          `<span class="calc-row"><span class="k">${t("calcSingleK")}</span> q = ${vec(S1.Q[sel], 1)}</span>` +
          `<span class="calc-row"><span class="k">${t("calcScore4K")}</span> ${TOK.map((w, j) => `${w} ${fmt(S1.SC[sel][j], 2)}`).join("　")}</span>` +
          `<span class="calc-row"><span class="k">${t("calcSingleZK")}</span> z = <span class="hl">${vec(S1.Z[sel])}</span></span>`;
      }
      // matrices
      matBox.hidden = !showMat; matBtn.textContent = t(showMat ? "btnMatHide" : "btnMat");
      if (showMat) {
        const mat = (cap, M, cols) => h("table", { class: "mh-mat" }, h("caption", null, cap),
          h("tr", null, h("th", null, ""), ...cols.map((c) => h("th", null, "→" + c))),
          ...M.map((r, i) => h("tr", null, h("th", null, DIM[i]), ...r.map((v) => h("td", null, fmt(v, 0))))));
        matBox.replaceChildren(
          mat(t("matWQ", HEADS[0]), HEADS[0].WQ, HEADS[0].sub), mat(t("matWK", HEADS[0]), HEADS[0].WK, HEADS[0].sub),
          mat(t("matWQ", HEADS[1]), HEADS[1].WQ, HEADS[1].sub), mat(t("matWK", HEADS[1]), HEADS[1].WK, HEADS[1].sub));
      }
      note.innerHTML = t("note");
    }
    render();
    ctx.onTheme(render);
    return {};
  },
});
