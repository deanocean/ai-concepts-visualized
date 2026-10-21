/* Seq2Seq と Source-Target Attention（シラバス §23；Seq2Seq 亦見 §27）
   日→英「私 は 猫 が 好き です」→「I like cats」。
   編碼器：h_j = x_j + 0.5 × h_{j−1}（簡化成線性 RNN），4 維＝人／動物／喜好／文法。
   原始 Seq2Seq：解碼器只拿到最後的 h₆；加上注意力：每一步用解碼器狀態當 Q、編碼器各步 h 當 K 和 V。
   文字在 locales/zh/seq2seq.js、locales/ja/seq2seq.js（原文的詞 src、4 個維度的名稱 dims 也在語言檔，以空白分隔）。 */
Atlas.define("seq2seq", {
  mount(bench, ctx) {
    const { h, s, fmt, ui } = Atlas;
    const { t } = ctx;
    const SRC = t("src").split(" ");
    const TGT = ["I", "like", "cats"];
    const DIM = t("dims").split(" ");
    const X = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]];
    const A = 0.5, D = 4, SQ = Math.sqrt(D);
    // 解碼器狀態（示範值）：要說 I 時在找「人」、like 找「喜好」、cats 找「動物」，並且避開助詞
    const Sd = [[6, 0, 0, -3], [0, 0, 6, -3], [0, 6, 0, -3]];
    const NEED = [0, 2, 1];
    const N = SRC.length, M = TGT.length;
    const Hs = []; { let hp = [0, 0, 0, 0]; for (const x of X) { hp = x.map((v, d) => v + A * hp[d]); Hs.push(hp); } }
    const dot = (a, b) => a.reduce((t, v, i) => t + v * b[i], 0);
    const SC = Sd.map((q) => Hs.map((k) => dot(q, k) / SQ));
    const W = SC.map((r) => { const e = r.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((v) => v / z); });
    const CTX = W.map((w) => [0, 1, 2, 3].map((d) => w.reduce((a, wj, j) => a + wj * Hs[j][d], 0)));
    const LAST = Hs[N - 1];
    const COEF = SRC.map((_, j) => A ** (N - 1 - j)); // h₆ 中各詞的係數

    let mode = "att", pos = 0; // 0..N+M
    const vec = (v, d = 2) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
    const pct = (w) => Math.round(w * 100) + "%";

    bench.append(h("style", { text: `
      .s2-tbl th.need { color: var(--hot); font-weight: 700; } .s2-tbl td.need { color: var(--hot); font-weight: 700; box-shadow: inset 2px 0 0 var(--hot), inset -2px 0 0 var(--hot); }
      .s2-heat { border-collapse: separate; border-spacing: 3px; font-family: var(--f-mono); font-size: 12px; }
      .s2-heat th { font-weight: 500; color: var(--muted); padding: 2px 4px; white-space: nowrap; }
      .s2-heat td { width: 50px; height: 30px; text-align: center; border-radius: 4px; }
    ` }));

    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode,
      options: [{ value: "plain", label: t("modePlain") }, { value: "att", label: t("modeAtt") }],
      onChange: (v) => { mode = v; render(); } });
    const player = ui.player(ctx, () => { if (pos >= N + M) return false; pos++; render(); return pos < N + M; }, 800);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (pos < N + M) { pos++; render(); } }, { primary: true });
    const jumpBtn = ui.btn(t("btnJump"), () => { player.pause(); pos = N + 1; render(); });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); pos = 0; render(); });

    const status = h("div", { class: "status" });
    const svgWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const cmpWrap = h("div");
    const heatWrap = h("div");
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, modeSeg.el),
      status, svgWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, jumpBtn, resetBtn),
      calc, cmpWrap, heatWrap, note,
    );

    function arrow(g, x1, y1, x2, y2, color, width = 1.6, op) {
      g.append(s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": width, "stroke-opacity": op }));
      const a = Math.atan2(y2 - y1, x2 - x1), L = 5 + width;
      g.append(s("path", { d: `M${x2},${y2} L${x2 - L * Math.cos(a - 0.45)},${y2 - L * Math.sin(a - 0.45)} L${x2 - L * Math.cos(a + 0.45)},${y2 - L * Math.sin(a + 0.45)} Z`, fill: color, "fill-opacity": op }));
    }
    // 4 格小色條：一個 4 維向量
    function strip(g, cx, y, v, on, max = 1.3) {
      const cw = 17, x0 = cx - cw * 2;
      v.forEach((val, d) => g.append(s("rect", { x: x0 + d * cw, y, width: cw - 2, height: 20, rx: 3, fill: on ? Atlas.divColor(val, max) : "var(--panel-2)", stroke: "var(--line-strong)", "stroke-width": 0.8 })));
    }

    function draw() {
      const Wd = 780, H = 384;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${H}`, role: "img", "aria-label": t("svgAria"), style: `display:block;width:100%;min-width:${Wd * 0.78}px;max-width:${Wd}px` });
      const ex = (j) => 120 + j * 92, yE = 74;
      const dx = (k) => 250 + k * 140, yD = 268;
      const encDone = Math.min(pos, N), dec = pos - N; // dec: 已完成的解碼步數
      const curDec = dec >= 1 ? dec - 1 : -1;
      const att = mode === "att";

      svg.append(s("text", { x: 8, y: yE + 4, "font-size": 12.5, "font-weight": 700, text: t("encName") }));
      svg.append(s("text", { x: 8, y: yE + 20, "font-size": 11, class: "t-muted", text: t("encSub") }));
      svg.append(s("text", { x: 8, y: yD + 4, "font-size": 12.5, "font-weight": 700, text: t("decName") }));
      svg.append(s("text", { x: 8, y: yD + 20, "font-size": 11, class: "t-muted", text: t("decSub") }));

      // decoder → encoder links (behind boxes)
      const links = s("g");
      if (curDec >= 0) {
        if (att) {
          W[curDec].forEach((w, j) => {
            const wdt = 1 + 12 * w;
            links.append(s("line", { x1: dx(curDec), y1: yD - 34, x2: ex(j), y2: yE + 34, stroke: "var(--hot)", "stroke-width": wdt, "stroke-opacity": 0.2 + 0.8 * w, "stroke-linecap": "round" }));
          });
        } else {
          // the single vector goes to every decoder step
          for (let k = 0; k <= curDec; k++) {
            links.append(s("path", { d: `M${ex(N - 1) + 70},${yE + 10} C${ex(N - 1) + 100},${yE + 150} ${dx(k) + 40},${yD - 110} ${dx(k) + 10},${yD - 36}`, fill: "none", stroke: "var(--hot)", "stroke-width": 2.2, "stroke-opacity": k === curDec ? 1 : 0.35 }));
          }
        }
      }
      svg.append(links);

      // encoder chain
      for (let j = 0; j < N; j++) {
        const x = ex(j), on = j < encDone, cur = pos === j + 1;
        svg.append(s("text", { x, y: 24, "text-anchor": "middle", "font-size": 15, "font-weight": cur ? 700 : 400, fill: on ? "var(--ink)" : "var(--faint)", text: SRC[j] }));
        arrow(svg, x, 31, x, yE - 22, on ? "var(--muted)" : "var(--line)", 1.3);
        if (j > 0) arrow(svg, ex(j - 1) + 38, yE, x - 40, yE, j < encDone ? "var(--accent)" : "var(--line)", 1.8);
        svg.append(s("rect", { x: x - 38, y: yE - 20, width: 76, height: 40, rx: 8, fill: "var(--panel)", stroke: cur ? "var(--hot)" : "var(--line-strong)", "stroke-width": cur ? 2.5 : 1 }));
        strip(svg, x, yE - 14, Hs[j], on);
        svg.append(s("text", { x: x - 34, y: yE + 34, "font-size": 10.5, class: "t-mono t-muted", text: `h${"₁₂₃₄₅₆"[j]}` }));
        if (att && curDec >= 0) {
          const w = W[curDec][j];
          svg.append(s("rect", { x: x - 4, y: yE + 23, width: 40, height: 18, rx: 4, fill: "var(--hot-soft)", stroke: "var(--hot)", "stroke-width": 0.8 }));
          svg.append(s("text", { x: x + 16, y: yE + 36, "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, class: "t-mono", fill: "var(--hot)", text: pct(w) }));
        }
      }
      // fixed-length vector (plain mode)
      if (!att) {
        const x = ex(N - 1) + 70;
        const on = encDone === N;
        svg.append(s("rect", { x: x - 8, y: yE - 24, width: 96, height: 48, rx: 8, fill: on ? "var(--hot-soft)" : "var(--panel-2)", stroke: on ? "var(--hot)" : "var(--line-strong)", "stroke-dasharray": on ? null : "4 3" }));
        svg.append(s("text", { x: x + 40, y: yE - 4, "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, text: t("onlyVec1") }));
        svg.append(s("text", { x: x + 40, y: yE + 13, "text-anchor": "middle", "font-size": 11.5, "font-weight": 700, text: t("onlyVec2") }));
        arrow(svg, ex(N - 1) + 38, yE, x - 8, yE, on ? "var(--hot)" : "var(--line)", 1.8);
      }

      // decoder
      for (let k = 0; k < M; k++) {
        const x = dx(k), on = k < dec, cur = k === curDec;
        if (k > 0) arrow(svg, dx(k - 1) + 46, yD, x - 48, yD, on ? "var(--accent)" : "var(--line)", 1.8);
        svg.append(s("rect", { x: x - 46, y: yD - 32, width: 92, height: 64, rx: 8, fill: "var(--panel)", stroke: cur ? "var(--hot)" : "var(--line-strong)", "stroke-width": cur ? 2.5 : 1 }));
        svg.append(s("text", { x, y: yD - 16, "text-anchor": "middle", "font-size": 10.5, class: "t-muted", text: att ? "context" : t("decGotH6") }));
        strip(svg, x, yD - 8, att ? CTX[k] : LAST, on);
        arrow(svg, x, yD + 32, x, yD + 56, on ? "var(--muted)" : "var(--line)", 1.3);
        svg.append(s("text", { x, y: yD + 78, "text-anchor": "middle", "font-size": 16, class: "t-mono", "font-weight": cur ? 700 : 400, fill: on ? "var(--ink)" : "var(--faint)", text: on ? TGT[k] : "?" }));
      }
      // legend for the strips
      const lg = s("g");
      const lx = 200, ly = H - 10;
      lg.append(s("text", { x: lx - 8, y: ly + 4, "text-anchor": "end", "font-size": 11, class: "t-muted", text: t("legend") }));
      DIM.forEach((d, i) => lg.append(s("text", { x: lx + i * 52, y: ly + 4, "font-size": 11, class: "t-muted", text: `${i + 1}=${d}` })));
      svg.append(lg);
      svgWrap.replaceChildren(svg);
    }

    function render() {
      draw();
      const att = mode === "att";
      const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };
      cmpWrap.replaceChildren(); heatWrap.replaceChildren();
      if (pos === 0) {
        set("info", t("statusStart"));
        calc.innerHTML = `<span class="k">${t("calcStart")}</span>`;
      } else if (pos <= N) {
        const j = pos - 1, hp = j ? Hs[j - 1] : [0, 0, 0, 0];
        set("info", t("statusEnc", { n: j + 1, w: SRC[j] }));
        calc.innerHTML =
          `<span class="calc-row"><span class="k">${t("calcEncRule")}</span></span>` +
          `<span class="calc-row">h${"₁₂₃₄₅₆"[j]} = ${vec(X[j], 0)} + 0.5 × ${vec(hp)} = <span class="hl">${vec(Hs[j])}</span></span>` +
          (j === N - 1 ? `<span class="calc-row"><span class="k">${t("calcEncShareK")}</span> h₆ = ${SRC.map((w, i) => `${fmt(COEF[i], COEF[i] < 0.1 ? 3 : 2)}×${w}`).join(" + ")}</span>` : "");
      } else {
        const k = pos - N - 1, nd = NEED[k];
        const got = att ? CTX[k][nd] : LAST[nd];
        set(att ? "ok" : (got < 0.3 ? "bad" : "warn"), att
          ? t("statusDecAtt", { tgt: TGT[k], dim: DIM[nd], src: SRC[W[k].indexOf(Math.max(...W[k]))], pct: pct(Math.max(...W[k])), got: fmt(got, 2) })
          : t(got < 0.3 ? "statusDecPlainLow" : "statusDecPlainHigh", { tgt: TGT[k], dim: DIM[nd], got: fmt(got, 2) }));
        if (att) {
          const ex = SC[k].map(Math.exp), tot = ex.reduce((a, b) => a + b);
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcQK")}</span> s = ${vec(Sd[k], 0)}　<span class="k">${t("calcKVK")}</span></span>` +
            `<span class="calc-row"><span class="k">${t("calcScoreK", { d: D })}</span> ${SRC.map((w, j) => `${w} ${fmt(SC[k][j], 2)}`).join("　")}</span>` +
            `<span class="calc-row"><span class="k">${t("calcSoftK", { tot: fmt(tot, 2) })}</span> ${SRC.map((w, j) => `${w} ${pct(W[k][j])}`).join("　")}</span>` +
            `<span class="calc-row"><span class="k">${t("calcCtxK")}</span> = <span class="hl">${vec(CTX[k])}</span></span>`;
        } else {
          calc.innerHTML =
            `<span class="calc-row"><span class="k">${t("calcSameK")}</span> h₆ = <span class="hl">${vec(LAST)}</span></span>` +
            `<span class="calc-row"><span class="k">${t("calcCellK", { dim: DIM[nd] })}</span> = ${fmt(LAST[nd], 3)}</span>`;
        }
        // need vs got table
        const tb = h("table", { class: "tbl s2-tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, t("thSay", { tgt: TGT[k] })), ...DIM.map((d, i) => h("th", { class: i === nd ? "need" : "" }, i === nd ? t("thNeed", { dim: d }) : d)))),
          h("tbody", null,
            h("tr", { class: att ? "" : "is-sel" }, h("td", { class: "l" }, t("rowPlain")), ...LAST.map((v, i) => h("td", { class: i === nd ? "need" : "" }, fmt(v, 2)))),
            h("tr", { class: att ? "is-sel" : "" }, h("td", { class: "l" }, t("rowAtt")), ...CTX[k].map((v, i) => h("td", { class: i === nd ? "need" : "" }, fmt(v, 2))))));
        cmpWrap.append(h("div", { class: "field" }, h("span", { class: "field__label" }, t("cmpLabel")), h("div", { class: "scroll-x" }, tb)));
      }
      // alignment heatmap
      if (att && pos > N) {
        const done = pos - N;
        const tbl = h("table", { class: "s2-heat" },
          h("thead", null, h("tr", null, h("th", null, t("heatCorner")), ...SRC.map((w) => h("th", null, w)))),
          h("tbody", null, ...TGT.map((w, k) => h("tr", null, h("th", { style: { textAlign: "right" } }, k < done ? w : "?"),
            ...SRC.map((_, j) => {
              const v = W[k][j], on = k < done;
              return h("td", { style: { background: on ? Atlas.rgbStr(Atlas.mix(Atlas.color("--panel"), Atlas.color("--hot"), Math.min(1, v * 1.2))) : "var(--panel-2)", color: on && v > 0.5 ? "var(--panel)" : "var(--ink)" } }, on ? pct(v) : "");
            })))));
        heatWrap.append(h("div", { class: "field" }, h("span", { class: "field__label" }, t("heatLabel")), h("div", { class: "scroll-x" }, tbl)));
      }
      note.innerHTML = t("note");
      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= N + M;
      jumpBtn.hidden = pos > N;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
