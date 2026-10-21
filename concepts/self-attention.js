/* Self-Attention（シラバス §23）
   7 個 token、4 維的示範向量；點任一詞看注意力權重，展開 Q・K → softmax → 加權 V 的計算。
   文字在 locales/zh/selfatt.js、locales/ja/selfatt.js（4 個維度的名稱 dims 也在語言檔，以空白分隔）。 */
Atlas.define("selfatt", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
    const TOK = ["The", "cat", "sat", "because", "it", "was", "tired"];
    const DIM = t("dims").split(" ");
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
    const A = SC.map((row) => { const ex = row.map(Math.exp), tot = ex.reduce((a, b) => a + b); return ex.map((x) => x / tot); });
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
    const toks = h("div", { class: "sa-toks", role: "group", "aria-label": t("toksAria") });
    const tokBtns = TOK.map((w, i) => { const b = h("button", { type: "button", class: "sa-tok", "aria-pressed": "false" }, w); b.addEventListener("click", () => select(i)); toks.append(b); return b; });
    const diagWrap = h("div", { class: "scroll-x" });
    const status = h("div", { class: "status status--info" });
    const heatWrap = h("div", { class: "scroll-x" });
    const toggle = ui.btn(t("btnShowSteps"), () => { showSteps = !showSteps; renderSteps(); }, { id: "sa-toggle" });
    const stepsBox = h("div", { class: "sa-steps", hidden: true });

    bench.append(
      h("div", { class: "field" }, h("span", { class: "field__label" }, t("toksLabel")), toks),
      diagWrap, status,
      h("div", { class: "field" }, h("span", { class: "field__label" }, t("heatLabel")), heatWrap),
      h("p", { class: "note", html: t("note") }),
      h("div", null, toggle), stepsBox,
    );

    /* ---------- attention arcs ---------- */
    function renderDiag() {
      const W = 700, H = 200, base = 160, x = (i) => 50 + i * 100;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("diagAria", { w: TOK[sel] }), style: "display:block;width:100%;min-width:540px;max-width:760px" });
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
      TOK.forEach((tk, j) => {
        const isSel = j === sel;
        svg.append(s("rect", { x: x(j) - 42, y: base - 20, width: 84, height: 34, rx: 8, fill: isSel ? "var(--hot)" : "var(--panel)", stroke: isSel ? "var(--hot)" : "var(--line-strong)" }));
        svg.append(s("text", { x: x(j), y: base + 3, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: isSel ? "var(--panel)" : "var(--ink)", "font-weight": isSel ? 700 : 400, text: tk }));
        svg.append(s("text", { x: x(j), y: base + 32, "text-anchor": "middle", "font-size": 13, "font-weight": 700, class: "t-mono", fill: "var(--hot)", text: pct(w[j]) }));
      });
      diagWrap.replaceChildren(svg);
      const order = w.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0]);
      const [w1, j1] = order[0];
      status.textContent = t(w1 > 0.3 ? "statusFocus" : "statusEven", { w: TOK[sel], top: TOK[j1], pct: pct(w1) });
    }

    /* ---------- heatmap ---------- */
    function renderHeat() {
      const tbl = h("table", { class: "sa-heat" });
      const head = h("tr", null, h("th", { class: "row", html: t("heatCorner") }), ...TOK.map((w) => h("th", null, w)));
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
      { value: 1, label: t("step1") }, { value: 2, label: t("step2") }, { value: 3, label: "③ softmax" }, { value: 4, label: t("step4") }],
      onChange: (v) => { step = v; renderSteps(); } });
    const stepBody = h("div", { style: { display: "flex", flexDirection: "column", gap: "10px" } });
    const nextStep = ui.btn(t("btnNext"), () => { step = step % 4 + 1; stepSeg.set(step); renderSteps(); }, { primary: true, sm: true });
    stepsBox.append(h("div", { class: "bench__row" }, stepSeg.el, nextStep), stepBody);

    function nzTerms(a, b) {
      const pairs = a.map((x, i) => [x, b[i]]).filter(([x, y]) => x !== 0 && y !== 0);
      return pairs.length ? pairs.map(([x, y]) => `${fmt(x, 1)}×${fmt(y, 1)}`).join(" + ") : "0";
    }
    const dimHead = () => h("tr", null, h("th", { class: "l" }, ""), ...DIM.map((d) => h("th", null, d)));

    function renderSteps() {
      stepsBox.hidden = !showSteps;
      toggle.textContent = t(showSteps ? "btnHideSteps" : "btnShowSteps");
      if (!showSteps) return;
      stepSeg.set(step);
      const tk = TOK[sel];
      const parts = [];
      if (step === 1) {
        parts.push(h("p", { html: t("s1Intro", { dims: DIM.join(t("dimSep")) }) }));
        const m = h("table", { class: "sa-mat" }, h("tr", null, h("th", null, "W_Q"), ...DIM.map((d) => h("th", null, "→" + d))),
          ...WQ.map((r, i) => h("tr", null, h("th", null, DIM[i]), ...r.map((v) => h("td", null, fmt(v, 0))))));
        const qTerms = [0, 1, 2, 3].map((j) => { const terms = E[sel].map((x, i) => [x, WQ[i][j]]).filter(([x, w]) => x && w); return terms.length ? `q<sub>${DIM[j]}</sub> = ${terms.map(([x, w]) => `${fmt(x, 1)}×${fmt(w, 0)}`).join(" + ")} = ${fmt(Q[sel][j], 1)}` : `q<sub>${DIM[j]}</sub> = 0`; });
        parts.push(h("div", { class: "bench__row", style: { alignItems: "flex-start" } },
          h("div", { class: "scroll-x" }, m),
          h("div", { class: "calc", style: { flex: "1 1 260px" }, html:
            `<span class="calc-row"><span class="k">${t("s1VecK", { w: tk })}</span> x = ${vec(E[sel], 1)}</span>` +
            qTerms.map((r) => `<span class="calc-row">${r}</span>`).join("") +
            `<span class="calc-row"><span class="k">${t("queryK")}</span> q = x × W_Q = <span class="hl">${vec(Q[sel], 1)}</span></span>` }),
        ));
        parts.push(h("p", { class: "note", html: t("s1Note") }));
        const kv = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, t("thWord")), h("th", null, t("keyK") + " k"), h("th", null, t("valueK") + " v"))),
          h("tbody", null, ...TOK.map((w, i) => h("tr", { class: i === sel ? "is-sel" : "" }, h("td", { class: "l" }, w), h("td", null, vec(Kv[i], 1)), h("td", null, vec(V[i], 1))))));
        parts.push(h("div", { class: "scroll-x" }, kv));
      } else if (step === 2) {
        parts.push(h("p", { html: t("s2Intro", { w: tk, d: D, sq: SQ }) }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, t("thSeen")), h("th", null, "k"), h("th", { class: "l" }, t("thQK", { q: vec(Q[sel], 1) })), h("th", null, "q·k"), h("th", null, `÷ √${D}`))),
          h("tbody", null, ...TOK.map((w, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, w), h("td", null, vec(Kv[j], 1)), h("td", { class: "l" }, nzTerms(Q[sel], Kv[j])), h("td", null, fmt(dot(Q[sel], Kv[j]), 2)), h("td", null, fmt(SC[sel][j], 2))))));
        parts.push(h("div", { class: "scroll-x" }, tb));
      } else if (step === 3) {
        const ex = SC[sel].map(Math.exp), tot = ex.reduce((a, b) => a + b);
        parts.push(h("p", { html: t("s3Intro") }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, t("thSeen")), h("th", null, t("thScore")), h("th", null, "e^s"), h("th", null, t("thDiv")), h("th", null, t("thWeight")))),
          h("tbody", null, ...TOK.map((w, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, w), h("td", null, fmt(SC[sel][j], 2)), h("td", null, fmt(ex[j], 2)), h("td", null, `${fmt(ex[j], 2)} ÷ ${fmt(tot, 2)}`), h("td", null, pct(A[sel][j]))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l" }, t("total")), h("td"), h("td", null, fmt(tot, 2)), h("td"), h("td", null, "100%"))));
        parts.push(h("div", { class: "scroll-x" }, tb));
      } else {
        const w = A[sel];
        parts.push(h("p", { html: t("s4Intro", { w: tk }) }));
        const tb = h("table", { class: "tbl" }, h("thead", null, h("tr", null, h("th", { class: "l" }, t("thWord")), h("th", null, t("thWeight")), h("th", null, "v"), ...DIM.map((d) => h("th", null, t("thWv", { dim: d }))))),
          h("tbody", null, ...TOK.map((wd, j) => h("tr", { class: j === sel ? "is-sel" : "" },
            h("td", { class: "l" }, wd), h("td", null, pct(w[j])), h("td", null, vec(V[j], 1)), ...V[j].map((x) => h("td", null, fmt(w[j] * x, 3)))))),
          h("tfoot", null, h("tr", null, h("td", { class: "l" }, t("sum")), h("td"), h("td"), ...Z[sel].map((x) => h("td", null, fmt(x, 3))))));
        parts.push(h("div", { class: "scroll-x" }, tb));
        const cmp = h("div", { class: "sa-cmp" }, h("span"), ...DIM.map((d) => h("span", { class: "hd" }, d)),
          h("span", null, t("cmpBefore")), ...E[sel].map((x) => h("span", { class: "mono" }, fmt(x, 2))),
          h("span", { style: { color: "var(--hot)", fontWeight: 700 } }, t("cmpAfter")), ...Z[sel].map((x) => h("span", { class: "mono", style: { color: "var(--hot)", fontWeight: 700 } }, fmt(x, 2))));
        parts.push(h("div", { class: "scroll-x" }, cmp));
        const diff = Z[sel].map((z, j) => z - E[sel][j]);
        const jUp = diff.indexOf(Math.max(...diff));
        const top = w.map((v, j) => [v, j]).filter(([, j]) => j !== sel).sort((a, b) => b[0] - a[0])[0][1];
        parts.push(h("div", { class: "status status--info", text: diff[jUp] > 0.05
          ? t("s4Mixed", { w: tk, dim: DIM[jUp], a: fmt(E[sel][jUp], 2), b: fmt(Z[sel][jUp], 2), top: TOK[top] })
          : t("s4Even", { w: tk }) }));
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
