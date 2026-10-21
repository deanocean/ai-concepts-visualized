/* Transformer（シラバス §23）
   整體架構頁：把 Self-Attention、Multi-Head、位置エンコーディング、Source-Target Attention 等零件接起來。
   ① 架構圖：點方塊或用「下一步」照資料流動的順序看，每塊附角色、形狀與參數量（以原論文 base 模型計：d_model = 512、8 個頭、FFN 中間 2048、編碼器與解碼器各 6 層；參數量是真的算出來的，不含詞嵌入與輸出層）。
   ② 訓練時（教師強制＋遮罩，一次並行）vs 推論時（一次一個詞，解碼器要跑好幾次）。
   ③ 解碼器的遮罩：4 個詞 × 4 維的示範向量（詞向量＋位置エンコーディング），Q = K = V = x，分數 q·k ÷ √4，被遮住的設成 −∞；並示範 Add & Norm。
   文字在 locales/zh/transformer.js、locales/ja/transformer.js：每個方塊的名稱、角色、形狀用 <id>Name / <id>Role / <id>Shape 這類 key。 */
(function () {
  const DM = 512, H = 8, FF = 2048, NL = 6;
  const P_ATT = 4 * DM * DM + 4 * DM, P_FFN = DM * FF + FF + FF * DM + DM, P_LN = 2 * DM;
  const num = (n) => n.toLocaleString("en-US");

  /* ③ 遮罩示範的數字 */
  const E = [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]];
  const PE = (p) => [Math.sin(p), Math.cos(p), Math.sin(p / 100), Math.cos(p / 100)];
  const X = E.map((e, p) => e.map((v, i) => v + PE(p)[i]));
  const SC = X.map((a) => X.map((b) => a.reduce((s, v, i) => s + v * b[i], 0) / 2));
  function attn(mask) {
    const A = SC.map((r, i) => { const rr = r.map((v, j) => (mask && j > i ? -Infinity : v)), e = rr.map(Math.exp), z = e.reduce((a, b) => a + b); return e.map((x) => x / z); });
    const Z = A.map((w) => [0, 1, 2, 3].map((d) => w.reduce((a, wj, j) => a + wj * X[j][d], 0)));
    return { A, Z };
  }

  Atlas.define("transformer", {
    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
      const { t } = ctx;
      const V = { dm: DM, h: H, dh: DM / H, ff: FF, nl: NL };
      const SRC = t("src").split(" "), TGT = ["I", "like", "cats", t("eos")], DEC_IN = [t("bos"), "I", "like", "cats"];

      /* 架構圖上的方塊（依資料流順序）；label 有語言檔 key 的用語言檔（\n 換行），其餘是英文原名 */
      const BLOCKS = [
        { id: "encin", col: 0, label: t("encinLabel"), name: t("encinName"), role: t("encinRole"), shape: t("encinShape", V), params: null },
        { id: "encatt", col: 0, label: "Multi-Head Self-Attention", name: t("encattName"), role: t("encattRole", V), shape: t("encattShape", V),
          params: [[t("paramAtt"), `4 × 512 × 512 + 4 × 512 = ${num(P_ATT)}`]] },
        { id: "encan1", col: 0, label: "Add & Norm", name: t("anName"), role: t("encan1Role", V), shape: `6 × ${DM} → 6 × ${DM}`, params: [[t("paramLN"), `2 × 512 = ${num(P_LN)}`]] },
        { id: "encffn", col: 0, label: "Feed Forward", name: t("ffnName"), role: t("encffnRole", V), shape: `6 × ${DM} → 6 × ${FF} → 6 × ${DM}`, params: [["512 × 2048 + 2048 + 2048 × 512 + 512", num(P_FFN)]] },
        { id: "encan2", col: 0, label: "Add & Norm", name: t("anName"), role: t("encan2Role", V), shape: `6 × ${DM} → 6 × ${DM}`, params: [[t("paramGB"), num(P_LN)]] },
        { id: "decin", col: 1, label: t("decinLabel"), name: t("decinName"), role: t("decinRole"), shape: t("decinShape", V), params: null },
        { id: "decatt", col: 1, label: "Masked Self-Attention", name: t("decattName"), role: t("decattRole"), shape: `4 × ${DM} → 4 × ${DM}`, params: [[t("paramSameEnc"), num(P_ATT)]] },
        { id: "decan1", col: 1, label: "Add & Norm", name: t("anName"), role: t("decan1Role"), shape: `4 × ${DM}`, params: [[t("paramGB"), num(P_LN)]] },
        { id: "deccross", col: 1, label: "Source-Target Attention", name: t("crossName"), role: t("crossRole"), shape: t("crossShape", V), params: [[t("paramSameSize"), num(P_ATT)]] },
        { id: "decan2", col: 1, label: "Add & Norm", name: t("anName"), role: t("decan2Role"), shape: `4 × ${DM}`, params: [[t("paramGB"), num(P_LN)]] },
        { id: "decffn", col: 1, label: "Feed Forward", name: t("ffnName"), role: t("decffnRole"), shape: `4 × ${DM} → 4 × ${FF} → 4 × ${DM}`, params: [["", num(P_FFN)]] },
        { id: "decan3", col: 1, label: "Add & Norm", name: t("anName"), role: t("decan3Role", V), shape: `4 × ${DM}`, params: [[t("paramGB"), num(P_LN)]] },
        { id: "out", col: 1, label: t("outLabel"), name: t("outName"), role: t("outRole", V), shape: t("outShape", V), params: null },
      ];

      let cur = 0, run = "train", inf = 0, mask = true, rowSel = 2;

      bench.append(h("style", { text: `
        .tr-fig { max-width: 560px; margin-inline: auto; width: 100%; }
        .tr-fig > svg { display: block; width: 100%; height: auto; }
        .tr-blk { cursor: pointer; }
        .tr-blk:hover rect { stroke: var(--accent); }
        .tr-panel { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--panel-2); font-size: 14px; display: flex; flex-direction: column; gap: 6px; }
        .tr-panel h3 { margin: 0; font-size: 15px; }
        .tr-panel p { margin: 0; }
        .tr-kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 13px; }
        .tr-kv > span:nth-child(odd) { color: var(--muted); white-space: nowrap; }
        .tr-kv .mono { overflow-wrap: anywhere; }
        @media (max-width: 520px) { .tr-kv { grid-template-columns: 1fr; } .tr-kv > span:nth-child(even) { margin-bottom: 4px; } }
        .tr-toks { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-family: var(--f-mono); font-size: 14px; }
        .tr-tok { padding: 3px 9px; border-radius: 6px; border: 1.5px solid var(--line-strong); background: var(--panel); }
        .tr-tok.new { border-color: var(--hot); background: var(--hot-soft); font-weight: 700; }
        .tr-tok.dim { color: var(--faint); border-style: dashed; }
        .tr-heat { border-collapse: separate; border-spacing: 3px; font-family: var(--f-mono); font-size: 12.5px; }
        .tr-heat th { font-weight: 500; color: var(--muted); padding: 2px 5px; white-space: nowrap; }
        .tr-heat th.row { text-align: right; cursor: pointer; }
        .tr-heat td { min-width: 58px; height: 30px; text-align: center; border-radius: 4px; cursor: pointer; }
        .tr-heat tr.is-sel th { color: var(--hot); font-weight: 700; }
        .tr-heat tr.is-sel td { outline: 2px solid var(--hot); outline-offset: -1px; }
        .tr-heat td.masked { background: repeating-linear-gradient(45deg, var(--panel-2) 0 5px, var(--line) 5px 7px); color: var(--muted); }
        .tr-sec { border-top: 1px solid var(--line); padding-top: 12px; display: flex; flex-direction: column; gap: 12px; }
        .tr-sec h3 { margin: 0; font-size: 15px; }
      ` }));

      /* ---------- ① 架構圖 ---------- */
      const player = ui.player(ctx, () => { if (cur >= BLOCKS.length - 1) return false; cur++; render(); return cur < BLOCKS.length - 1; }, 1600);
      const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (cur > 0) { cur--; render(); } });
      const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (cur < BLOCKS.length - 1) { cur++; render(); } }, { primary: true });
      const status = h("div", { class: "status status--info" });
      const svgFig = h("div", { class: "tr-fig" }), panel = h("div", { class: "tr-panel" });
      const totals = h("p", { class: "note" });

      /* ---------- ② 訓練 vs 推論 ---------- */
      const runSeg = ui.seg({ value: run, options: [{ value: "train", label: t("runTrain") }, { value: "infer", label: t("runInfer") }], onChange: (v) => { run = v; inf = 0; renderRun(); } });
      const infPrev = ui.btn("◀", () => { if (inf > 0) { inf--; renderRun(); } }, { sm: true, aria: t("infPrevAria") });
      const infNext = ui.btn(t("btnInfNext"), () => { if (inf < 4) { inf++; renderRun(); } }, { sm: true });
      const runStatus = h("div", { class: "status" }), runBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "8px" } });

      /* ---------- ③ 遮罩 ---------- */
      const maskSeg = ui.seg({ value: mask, options: [{ value: true, label: t("maskOn") }, { value: false, label: t("maskOff") }], onChange: (v) => { mask = v; renderMask(); } });
      const maskStatus = h("div", { class: "status" }), heatWrap = h("div", { class: "scroll-x" }), maskCalc = h("div", { class: "calc" }), anCalc = h("div", { class: "calc" });

      bench.append(
        h("div", { class: "field" }, h("span", { class: "field__label" }, t("sec1Label")), h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el)),
        status, svgFig, panel, totals,
        h("div", { class: "tr-sec" }, h("h3", null, t("sec2Title")), h("div", { class: "bench__row" }, runSeg.el, infPrev, infNext), runStatus, runBox),
        h("div", { class: "tr-sec" }, h("h3", null, t("sec3Title")),
          h("p", { class: "hint", style: { margin: 0 }, html: t("sec3Hint") }),
          h("div", { class: "bench__row" }, maskSeg.el), maskStatus, heatWrap, maskCalc,
          h("div", { class: "field" }, h("span", { class: "field__label" }, t("anLabel")), anCalc)));

      function drawArch() {
        const W = 400, Hh = 560, colX = [12, 208], bw = 180;
        const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": t("archAria") });
        const pos = {};
        // 編碼器：由下往上
        const encY = { encin: 470, encatt: 380, encan1: 338, encffn: 274, encan2: 232 };
        const decY = { decin: 470, decatt: 400, decan1: 358, deccross: 300, decan2: 258, decffn: 200, decan3: 158, out: 70 };
        const yOf = (b) => (b.col ? decY : encY)[b.id];
        // 層的外框（×6）
        svg.append(s("rect", { x: colX[0] - 6, y: 220, width: bw + 12, height: 196, rx: 10, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "5 4" }));
        svg.append(s("rect", { x: colX[1] - 6, y: 146, width: bw + 12, height: 292, rx: 10, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "5 4" }));
        svg.append(s("text", { x: colX[0] - 4, y: 212, "font-size": 12, "font-weight": 700, text: t("encStack", V) })); // 靠左對齊：日文標籤較長，置右會被裁掉
        svg.append(s("text", { x: colX[1] + bw / 2 - 8, y: 138, "text-anchor": "end", "font-size": 12, "font-weight": 700, text: t("decStack", V) }));
        // 箭頭
        const ar = (x1, y1, x2, y2, color = "var(--muted)", dash) => {
          svg.append(s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": 1.6, "stroke-dasharray": dash || null }));
          const a = Math.atan2(y2 - y1, x2 - x1);
          svg.append(s("path", { d: `M${x2},${y2} L${x2 - 7 * Math.cos(a - 0.45)},${y2 - 7 * Math.sin(a - 0.45)} L${x2 - 7 * Math.cos(a + 0.45)},${y2 - 7 * Math.sin(a + 0.45)} Z`, fill: color }));
        };
        const order0 = ["encin", "encatt", "encan1", "encffn", "encan2"], order1 = ["decin", "decatt", "decan1", "deccross", "decan2", "decffn", "decan3", "out"];
        const bh = (id) => (id === "encin" || id === "decin" ? 40 : id.includes("an") ? 26 : id === "out" ? 40 : 34);
        [order0, order1].forEach((ord, c) => ord.forEach((id, k) => {
          if (!k) return;
          const a = ord[k - 1], cx = colX[c] + bw / 2;
          ar(cx, (c ? decY : encY)[a] - bh(a) / 2, cx, (c ? decY : encY)[id] + bh(id) / 2 + 1);
        }));
        // 殘差的捷徑（虛線弧）
        // 編碼器的捷徑畫在左邊、解碼器的畫在右邊，中間留給「原文的資訊」那條線
        const skip = (c, from, to) => {
          const ys = (c ? decY : encY), y1 = ys[from] + bh(from) / 2 - 4, y2 = ys[to];
          const d = c ? `M${colX[1] + bw - 10},${y1} C${colX[1] + bw + 12},${y1} ${colX[1] + bw + 12},${y2} ${colX[1] + bw},${y2}` : `M${colX[0] + 10},${y1} C${colX[0] - 12},${y1} ${colX[0] - 12},${y2} ${colX[0]},${y2}`;
          svg.append(s("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 1.4, "stroke-dasharray": "3 3" }));
        };
        skip(0, "encin", "encan1"); skip(0, "encan1", "encan2"); skip(1, "decin", "decan1"); skip(1, "decan1", "decan2"); skip(1, "decan2", "decan3");
        // 編碼器 → 解碼器的 Source-Target Attention
        svg.append(s("path", { d: `M${colX[0] + bw},${encY.encan2} C${colX[0] + bw + 14},${encY.encan2} ${colX[1] - 18},${decY.deccross} ${colX[1] - 1},${decY.deccross}`, fill: "none", stroke: "var(--hot)", "stroke-width": 2.2 }));
        svg.append(s("path", { d: `M${colX[1]},${decY.deccross} l-8,-4 l0,8 Z`, fill: "var(--hot)" }));
        svg.append(s("text", { x: colX[0] + 4, y: 160, "font-size": 11.5, fill: "var(--hot)", text: t("encOut1") }));
        svg.append(s("text", { x: colX[0] + 4, y: 176, "font-size": 11.5, fill: "var(--hot)", text: t("encOut2") }));
        // 最下方的詞
        svg.append(s("text", { x: colX[0] + bw / 2, y: 512, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: SRC.join(" ") }));
        svg.append(s("text", { x: colX[0] + bw / 2, y: 530, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: t("srcCap") }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 512, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: DEC_IN.join(" ") }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 530, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: t("decInCap") }));
        ar(colX[0] + bw / 2, 498, colX[0] + bw / 2, encY.encin + 21); ar(colX[1] + bw / 2, 498, colX[1] + bw / 2, decY.decin + 21);
        svg.append(s("text", { x: colX[1] + bw / 2, y: 30, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: "→ " + TGT.join(" ") }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 14, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: t("outCap") }));
        ar(colX[1] + bw / 2, decY.out - 20, colX[1] + bw / 2, 36);
        // 方塊
        BLOCKS.forEach((b, i) => {
          const y = yOf(b), hh = bh(b.id), x = colX[b.col], on = i === cur, done = i < cur;
          const g = s("g", { class: "tr-blk", role: "button", "aria-label": b.name });
          const isAN = b.id.includes("an");
          g.append(s("rect", { x, y: y - hh / 2, width: bw, height: hh, rx: 7, fill: on ? "var(--hot-soft)" : isAN ? "var(--accent-soft)" : "var(--panel)", stroke: on ? "var(--hot)" : "var(--line-strong)", "stroke-width": on ? 2.5 : 1 }));
          const lines = b.label.split("\n");
          lines.forEach((ln, k) => g.append(s("text", { x: x + bw / 2, y: y + 4.5 + (k - (lines.length - 1) / 2) * 15, "text-anchor": "middle", "font-size": isAN ? 12 : 12.5, "font-weight": on ? 700 : 500, fill: done && !on ? "var(--muted)" : null, text: ln })));
          g.addEventListener("click", () => { player.pause(); cur = i; render(); });
          svg.append(g); pos[b.id] = y;
        });
        svgFig.replaceChildren(svg);
      }

      function render() {
        drawArch();
        const b = BLOCKS[cur];
        prevBtn.disabled = cur === 0; nextBtn.disabled = cur >= BLOCKS.length - 1;
        status.textContent = t(cur === 4 ? "statusBlockEnc" : cur === BLOCKS.length - 1 ? "statusBlockOut" : "statusBlock", { i: cur + 1, n: BLOCKS.length, name: b.name });
        panel.replaceChildren(
          h("h3", null, b.label.replace(/\n/g, "")),
          h("p", { html: b.role }),
          h("div", { class: "tr-kv" },
            h("span", null, t("kvShape")), h("span", { class: "mono" }, b.shape),
            ...(b.params ? [h("span", null, t("kvParams")), h("span", { class: "mono", html: b.params.map(([k, v]) => (k ? t("paramKV", { k, v: `<b>${v}</b>` }) : `<b>${v}</b>`)).join("<br>") })] : [])));
      }
      const encL = P_ATT + P_FFN + 2 * P_LN, decL = 2 * P_ATT + P_FFN + 3 * P_LN;
      totals.innerHTML = t("totals", { ...V, enc: num(encL), dec: num(decL), encAll: num(NL * encL), decAll: num(NL * decL), all: num(NL * (encL + decL)) });

      function renderRun() {
        infPrev.hidden = infNext.hidden = run !== "infer";
        infPrev.disabled = inf === 0; infNext.disabled = inf >= 4;
        const tok = (w, cls) => h("span", { class: "tr-tok " + (cls || "") }, w);
        if (run === "train") {
          runStatus.className = "status status--info";
          runStatus.textContent = t("statusTrain");
          runBox.replaceChildren(
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, t("decInLabel")), ...DEC_IN.map((w) => tok(w))),
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, t("predLabel")), ...TGT.map((w) => tok(w, "new"))),
            h("p", { class: "note", style: { margin: 0 }, html: t("noteTrain") }));
        } else {
          const decRuns = inf, posComputed = (inf * (inf + 1)) / 2;
          runStatus.className = "status status--" + (inf >= 4 ? "ok" : "info");
          const sum = "1" + (inf > 1 ? " + " + Array.from({ length: inf - 1 }, (_, k) => k + 2).join(" + ") : "");
          runStatus.textContent = inf === 0
            ? t("statusInfer0", { bos: DEC_IN[0] })
            : t(inf >= 4 ? "statusInferEnd" : "statusInfer", { n: inf, runs: decRuns, sum, tot: posComputed, eos: TGT[3] });
          const shown = DEC_IN.slice(0, Math.max(1, inf + (inf < 4 ? 1 : 0)));
          runBox.replaceChildren(
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, t("decRunLabel", { k: Math.min(inf + 1, 4) })), ...shown.map((w, k) => tok(w, k === shown.length - 1 && inf > 0 && inf < 4 ? "new" : ""))),
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, t("genLabel")), ...(inf ? TGT.slice(0, inf).map((w, k) => tok(w, k === inf - 1 ? "new" : "")) : [tok(t("genNone"), "dim")])),
            h("p", { class: "note", style: { margin: 0 }, html: t("noteInfer") }));
        }
      }

      const vec = (v, d = 2) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      function renderMask() {
        const { A, Z } = attn(mask);
        const panelC = Atlas.color("--panel"), hot = Atlas.color("--hot");
        heatWrap.replaceChildren(h("table", { class: "tr-heat" },
          h("thead", null, h("tr", null, h("th", { class: "row" }, t("heatCorner")), ...DEC_IN.map((w) => h("th", null, w)))),
          h("tbody", null, ...A.map((r, i) => {
            const tr = h("tr", { class: i === rowSel ? "is-sel" : "" }, h("th", { class: "row" }, DEC_IN[i]),
              ...r.map((v, j) => mask && j > i
                ? h("td", { class: "masked", title: t("maskedTitle") }, "✕")
                : h("td", { style: { background: Atlas.rgbStr(Atlas.mix(panelC, hot, Math.min(1, v * 1.2))), color: v > 0.5 ? "var(--panel)" : "var(--ink)" } }, fmt(v * 100, 1) + "%")));
            tr.addEventListener("click", () => { rowSel = i; renderMask(); });
            return tr;
          }))));
        const peek = attn(false).A[0][1];
        if (mask) { maskStatus.className = "status status--ok"; maskStatus.textContent = t("statusMaskOn", { bos: DEC_IN[0] }); }
        else { maskStatus.className = "status status--bad"; maskStatus.textContent = t("statusMaskOff", { bos: DEC_IN[0], p: fmt(peek * 100, 1) }); }
        const i = rowSel, sc = SC[i], vis = sc.map((v, j) => !(mask && j > i));
        const ex = sc.map((v, j) => (vis[j] ? Math.exp(v) : 0)), tot = ex.reduce((a, b) => a + b);
        maskCalc.innerHTML =
          row(t("calcVecK", { w: DEC_IN[i], i: i + 1 }), t("calcVec", { e: vec(E[i], 0), pe: vec(PE(i)), x: vec(X[i]) })) +
          row(t("calcScoreK"), DEC_IN.map((w, j) => `${w} ${vis[j] ? fmt(sc[j], 2) : t("maskedScore")}`).join("　")) +
          row("softmax", DEC_IN.map((w, j) => `${w} ${vis[j] ? `e^${fmt(sc[j], 2)} ÷ ${fmt(tot, 2)} = ${fmt(A[i][j] * 100, 1)}%` : "e^−∞ = 0"}`).join("　")) +
          row(t("calcOutK"), `<span class="hl">${vec(Z[i])}</span>`);
        const r = X[i].map((v, d) => v + Z[i][d]), mu = r.reduce((a, b) => a + b) / 4, va = r.reduce((a, b) => a + (b - mu) ** 2, 0) / 4, eps = 1e-5;
        const nrm = r.map((v) => (v - mu) / Math.sqrt(va + eps));
        anCalc.innerHTML =
          row(t("anAddK"), `x + z = ${vec(X[i])} + ${vec(Z[i])} = ${vec(r)}`) +
          row(t("anMeanK"), `${fmt(mu, 3)}　<span class="k">${t("anVarK")}</span> ${fmt(va, 3)}`) +
          row("Norm", `(x + z − μ) ÷ √(σ² + ε) = <span class="hl">${vec(nrm)}</span>　<span class="k">${t("anNormNote")}</span>`);
      }

      render(); renderRun(); renderMask();
      ctx.onTheme(() => renderMask());
      return { destroy() { player.pause(); } };
    },
  });
})();
