/* Transformer（シラバス §23）
   整體架構頁：把 Self-Attention、Multi-Head、位置エンコーディング、Source-Target Attention 等零件接起來。
   ① 架構圖：點方塊或用「下一步」照資料流動的順序看，每塊附角色、形狀與參數量（以原論文 base 模型計：d_model = 512、8 個頭、FFN 中間 2048、編碼器與解碼器各 6 層；參數量是真的算出來的，不含詞嵌入與輸出層）。
   ② 訓練時（教師強制＋遮罩，一次並行）vs 推論時（一次一個詞，解碼器要跑好幾次）。
   ③ 解碼器的遮罩：4 個詞 × 4 維的示範向量（詞向量＋位置エンコーディング），Q = K = V = x，分數 q·k ÷ √4，被遮住的設成 −∞；並示範 Add & Norm。 */
(function () {
  const DM = 512, H = 8, FF = 2048, NL = 6;
  const P_ATT = 4 * DM * DM + 4 * DM, P_FFN = DM * FF + FF + FF * DM + DM, P_LN = 2 * DM;
  const SRC = ["私", "は", "猫", "が", "好き", "です"], TGT = ["I", "like", "cats", "〈結束〉"], DEC_IN = ["〈開始〉", "I", "like", "cats"];
  const num = (n) => n.toLocaleString("en-US");

  /* 架構圖上的方塊（依資料流順序） */
  const BLOCKS = [
    { id: "encin", col: 0, label: "入力埋め込み ＋ 位置エンコーディング", name: "編碼器的輸入",
      role: `原文每個詞先查表變成向量（<a href="#w2v">単語埋め込み</a>），再加上<a href="#posenc">位置エンコーディング</a>：注意力本身不管順序，位置要另外加進去。`,
      shape: `6 個詞 → 6 × ${DM}`, params: null },
    { id: "encatt", col: 0, label: "Multi-Head Self-Attention", name: "編碼器的 Self-Attention",
      role: `原文的每個詞同時看原文所有的詞（<a href="#selfatt">Self-Attention</a>），用 ${H} 個頭各看一種關係（<a href="#mha">Multi-Head Attention</a>）。`,
      shape: `6 × ${DM} → 6 × ${DM}（每個頭 ${DM / H} 維）`, params: [["W_Q、W_K、W_V、W_O 各 512 × 512，加上偏置", `4 × 512 × 512 + 4 × 512 = ${num(P_ATT)}`]] },
    { id: "encan1", col: 0, label: "Add & Norm", name: "殘差＋レイヤー正規化",
      role: `把這一層的輸入直接加回輸出（<a href="#resnet">スキップ結合</a>），再做<span class="jp">レイヤー正規化</span>（每個詞自己的 ${DM} 個數拉回平均 0、變異數 1，見<a href="#norm">正規化層</a>）。層疊很深也學得動。下面「③」有實際算給你看。`,
      shape: `6 × ${DM} → 6 × ${DM}`, params: [["γ、β 各 512", `2 × 512 = ${num(P_LN)}`]] },
    { id: "encffn", col: 0, label: "Feed Forward", name: "位置ごとの全結合層",
      role: `每個詞<b>各自</b>通過同一個兩層全結合網路（${DM} → ${FF}，ReLU → ${DM}）。注意力負責「詞與詞之間交換資訊」，這裡負責「每個詞自己再加工」。`,
      shape: `6 × ${DM} → 6 × ${FF} → 6 × ${DM}`, params: [["512 × 2048 + 2048 ＋ 2048 × 512 + 512", num(P_FFN)]] },
    { id: "encan2", col: 0, label: "Add & Norm", name: "殘差＋レイヤー正規化", role: `同上：把 Feed Forward 的輸入加回去再正規化。到這裡是編碼器的 1 層，整個編碼器把這一層疊 ${NL} 次（每層權重不同），最後的輸出交給解碼器的每一層。`, shape: `6 × ${DM} → 6 × ${DM}`, params: [["γ、β", num(P_LN)]] },
    { id: "decin", col: 1, label: "出力埋め込み ＋ 位置エンコーディング", name: "解碼器的輸入",
      role: `解碼器的輸入是<b>到目前為止的譯文</b>，最前面放一個〈開始〉記號（所以叫「往右挪一格」，shifted right）。訓練時一次放進整句正解（<a href="#teacher">教師強制</a>）；推論時只能放已經生成的詞。`,
      shape: `譯文 4 個詞 → 4 × ${DM}`, params: null },
    { id: "decatt", col: 1, label: "Masked Self-Attention", name: "解碼器的 Masked Self-Attention",
      role: `譯文的每個詞看譯文中<b>自己和前面</b>的詞；後面的詞用遮罩（mask）擋住，因為推論時後面的詞還不存在，訓練時也不能偷看答案。下面「③」可以把遮罩拿掉比較。`,
      shape: `4 × ${DM} → 4 × ${DM}`, params: [["和編碼器的 Self-Attention 相同", num(P_ATT)]] },
    { id: "decan1", col: 1, label: "Add & Norm", name: "殘差＋レイヤー正規化", role: `同編碼器。`, shape: `4 × ${DM}`, params: [["γ、β", num(P_LN)]] },
    { id: "deccross", col: 1, label: "Source-Target Attention", name: "Source-Target Attention",
      role: `解碼器的詞當クエリ，編碼器最後的輸出當キー・バリュー：譯文每個位置回頭到原文找需要的資訊（<a href="#seq2seq">Seq2Seq と Source-Target Attention</a>）。這是原文和譯文唯一的接點。`,
      shape: `Q：4 × ${DM}（譯文）；K・V：6 × ${DM}（原文）→ 4 × ${DM}`, params: [["和 Self-Attention 相同大小", num(P_ATT)]] },
    { id: "decan2", col: 1, label: "Add & Norm", name: "殘差＋レイヤー正規化", role: `同上。`, shape: `4 × ${DM}`, params: [["γ、β", num(P_LN)]] },
    { id: "decffn", col: 1, label: "Feed Forward", name: "位置ごとの全結合層", role: `同編碼器的 Feed Forward。`, shape: `4 × ${DM} → 4 × ${FF} → 4 × ${DM}`, params: [["", num(P_FFN)]] },
    { id: "decan3", col: 1, label: "Add & Norm", name: "殘差＋レイヤー正規化", role: `解碼器的 1 層到此為止，同樣疊 ${NL} 次。`, shape: `4 × ${DM}`, params: [["γ、β", num(P_LN)]] },
    { id: "out", col: 1, label: "線形層 → ソフトマックス", name: "輸出",
      role: `每個位置的 ${DM} 維向量乘上一個矩陣，變成「詞彙表裡每個詞」的分數，再用ソフトマックス（見<a href="#act">活性化関数</a>）變成機率：第 1 個位置預測 I、第 2 個預測 like……`,
      shape: `4 × ${DM} → 4 × 詞彙數（機率）`, params: null },
  ];

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
    lead: `<p>RNN 必須一個詞一個詞往後算，句子一長既慢又容易忘記前面。<span class="jp">Transformer</span> 完全拿掉 RNN，只用注意力：讓每個詞一次看遍所有的詞，所以距離再遠也能直接連上，而且整句可以同時平行計算。它由<span class="jp">エンコーダ</span>（讀原文）和<span class="jp">デコーダ</span>（寫譯文）兩疊相同結構的層組成；這一頁把前面看過的零件接起來看。</p>`,
    closing: `編碼器每一層＝Self-Attention（詞與詞交換資訊）＋ Feed Forward（每個詞自己加工），各自外面包一層「殘差＋レイヤー正規化」；解碼器多了「遮住後面的詞」的 Self-Attention 和回頭看原文的 Source-Target Attention。訓練時靠教師強制加遮罩，整句一次算完；推論時沒有正解，只能一個詞一個詞生成，解碼器要跑好幾次。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui } = Atlas;
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
      const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (cur > 0) { cur--; render(); } });
      const nextBtn = ui.btn("下一步 ▶", () => { player.pause(); if (cur < BLOCKS.length - 1) { cur++; render(); } }, { primary: true });
      const status = h("div", { class: "status status--info" });
      const svgFig = h("div", { class: "tr-fig" }), panel = h("div", { class: "tr-panel" });
      const totals = h("p", { class: "note" });

      /* ---------- ② 訓練 vs 推論 ---------- */
      const runSeg = ui.seg({ value: run, options: [{ value: "train", label: "訓練時" }, { value: "infer", label: "推論時" }], onChange: (v) => { run = v; inf = 0; renderRun(); } });
      const infPrev = ui.btn("◀", () => { if (inf > 0) { inf--; renderRun(); } }, { sm: true, aria: "上一次" });
      const infNext = ui.btn("生成下一個詞 ▶", () => { if (inf < 4) { inf++; renderRun(); } }, { sm: true });
      const runStatus = h("div", { class: "status" }), runBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "8px" } });

      /* ---------- ③ 遮罩 ---------- */
      const maskSeg = ui.seg({ value: mask, options: [{ value: true, label: "有遮罩（實際做法）" }, { value: false, label: "拿掉遮罩" }], onChange: (v) => { mask = v; renderMask(); } });
      const maskStatus = h("div", { class: "status" }), heatWrap = h("div", { class: "scroll-x" }), maskCalc = h("div", { class: "calc" }), anCalc = h("div", { class: "calc" });

      bench.append(
        h("div", { class: "field" }, h("span", { class: "field__label" }, "① 照資料流動的順序看每一塊（也可以直接點圖上的方塊）"), h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el)),
        status, svgFig, panel, totals,
        h("div", { class: "tr-sec" }, h("h3", null, "② 訓練時和推論時怎麼跑"), h("div", { class: "bench__row" }, runSeg.el, infPrev, infNext), runStatus, runBox),
        h("div", { class: "tr-sec" }, h("h3", null, "③ 解碼器的遮罩，與 Add & Norm"),
          h("p", { class: "hint", style: { margin: 0 }, html: "譯文 4 個詞的示範向量（4 維詞向量＋位置エンコーディング），Q = K = V = 這個向量，分數 q·k ÷ √4。點一列看那個詞的計算。" }),
          h("div", { class: "bench__row" }, maskSeg.el), maskStatus, heatWrap, maskCalc,
          h("div", { class: "field" }, h("span", { class: "field__label" }, "Add & Norm：把注意力層的輸出加回原本的向量，再做レイヤー正規化（γ = 1、β = 0）"), anCalc)));

      function drawArch() {
        const W = 400, Hh = 560, colX = [12, 208], bw = 180;
        const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Transformer 的架構" });
        const pos = {};
        // 編碼器：由下往上
        const encY = { encin: 470, encatt: 380, encan1: 338, encffn: 274, encan2: 232 };
        const decY = { decin: 470, decatt: 400, decan1: 358, deccross: 300, decan2: 258, decffn: 200, decan3: 158, out: 70 };
        const yOf = (b) => (b.col ? decY : encY)[b.id];
        // 層的外框（×6）
        svg.append(s("rect", { x: colX[0] - 6, y: 220, width: bw + 12, height: 196, rx: 10, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "5 4" }));
        svg.append(s("rect", { x: colX[1] - 6, y: 146, width: bw + 12, height: 292, rx: 10, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "5 4" }));
        svg.append(s("text", { x: colX[0] + bw / 2 - 8, y: 212, "text-anchor": "end", "font-size": 12, "font-weight": 700, text: `編碼器（×${NL} 層）` }));
        svg.append(s("text", { x: colX[1] + bw / 2 - 8, y: 138, "text-anchor": "end", "font-size": 12, "font-weight": 700, text: `解碼器（×${NL} 層）` }));
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
        svg.append(s("text", { x: colX[0] + 4, y: 160, "font-size": 11.5, fill: "var(--hot)", text: "編碼器最後的輸出" }));
        svg.append(s("text", { x: colX[0] + 4, y: 176, "font-size": 11.5, fill: "var(--hot)", text: "→ 解碼器每一層的 K・V" }));
        // 最下方的詞
        svg.append(s("text", { x: colX[0] + bw / 2, y: 512, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: SRC.join(" ") }));
        svg.append(s("text", { x: colX[0] + bw / 2, y: 530, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: "原文" }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 512, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: DEC_IN.join(" ") }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 530, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: "到目前為止的譯文（往右挪一格）" }));
        ar(colX[0] + bw / 2, 498, colX[0] + bw / 2, encY.encin + 21); ar(colX[1] + bw / 2, 498, colX[1] + bw / 2, decY.decin + 21);
        svg.append(s("text", { x: colX[1] + bw / 2, y: 30, "text-anchor": "middle", "font-size": 13, class: "t-mono", text: "→ " + TGT.join(" ") }));
        svg.append(s("text", { x: colX[1] + bw / 2, y: 14, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: "每個位置預測下一個詞" }));
        ar(colX[1] + bw / 2, decY.out - 20, colX[1] + bw / 2, 36);
        // 方塊
        BLOCKS.forEach((b, i) => {
          const y = yOf(b), hh = bh(b.id), x = colX[b.col], on = i === cur, done = i < cur;
          const g = s("g", { class: "tr-blk", role: "button", "aria-label": b.name });
          const isAN = b.id.includes("an");
          g.append(s("rect", { x, y: y - hh / 2, width: bw, height: hh, rx: 7, fill: on ? "var(--hot-soft)" : isAN ? "var(--accent-soft)" : "var(--panel)", stroke: on ? "var(--hot)" : "var(--line-strong)", "stroke-width": on ? 2.5 : 1 }));
          const lines = b.label.includes(" ＋ ") ? b.label.split(" ＋ ").map((t, k) => (k ? "＋ " + t : t)) : [b.label];
          lines.forEach((t, k) => g.append(s("text", { x: x + bw / 2, y: y + 4.5 + (k - (lines.length - 1) / 2) * 15, "text-anchor": "middle", "font-size": isAN ? 12 : 12.5, "font-weight": on ? 700 : 500, fill: done && !on ? "var(--muted)" : null, text: t })));
          g.addEventListener("click", () => { player.pause(); cur = i; render(); });
          svg.append(g); pos[b.id] = y;
        });
        svgFig.replaceChildren(svg);
      }

      function render() {
        drawArch();
        const b = BLOCKS[cur];
        prevBtn.disabled = cur === 0; nextBtn.disabled = cur >= BLOCKS.length - 1;
        status.textContent = `第 ${cur + 1} / ${BLOCKS.length} 塊：${b.name}${cur === 4 ? "（編碼器到此結束，輸出交給解碼器）" : cur === BLOCKS.length - 1 ? "（得到每個位置的下一個詞的機率）" : ""}`;
        panel.replaceChildren(
          h("h3", null, b.label.replace(" ＋ ", "＋")),
          h("p", { html: b.role }),
          h("div", { class: "tr-kv" },
            h("span", null, "形狀（原文 6 詞、譯文 4 詞）"), h("span", { class: "mono" }, b.shape),
            ...(b.params ? [h("span", null, "這一塊的參數量"), h("span", { class: "mono", html: b.params.map(([k, v]) => (k ? `${k}：` : "") + `<b>${v}</b>`).join("<br>") })] : [])));
      }
      const encL = P_ATT + P_FFN + 2 * P_LN, decL = 2 * P_ATT + P_FFN + 3 * P_LN;
      totals.innerHTML = `參數量以原論文的 base 模型計（d_model = ${DM}、${H} 個頭、Feed Forward 中間 ${FF} 維、各 ${NL} 層）：編碼器 1 層 ${num(encL)}、解碼器 1 層 ${num(decL)}，${NL} 層加起來 ${num(NL * encL)} + ${num(NL * decL)} = <b>${num(NL * (encL + decL))}</b>（不含詞嵌入與輸出層）。編碼器的 Self-Attention 是 <a href="#mha">Multi-Head</a>：8 個頭各 64 維，合起來的參數量和 1 個 512 維的頭一樣。`;

      function renderRun() {
        infPrev.hidden = infNext.hidden = run !== "infer";
        infPrev.disabled = inf === 0; infNext.disabled = inf >= 4;
        const tok = (t, cls) => h("span", { class: "tr-tok " + (cls || "") }, t);
        if (run === "train") {
          runStatus.className = "status status--info";
          runStatus.textContent = "訓練時：整句正解已經知道，解碼器一次放進全部 4 個輸入、4 個位置同時算出預測（教師強制），靠遮罩讓每個位置只看得到自己和前面。編碼器 1 次、解碼器 1 次就算完一句。";
          runBox.replaceChildren(
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, "解碼器的輸入："), ...DEC_IN.map((t) => tok(t))),
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, "各位置要預測："), ...TGT.map((t) => tok(t, "new"))),
            h("p", { class: "note", style: { margin: 0 }, html: `第 k 個位置的損失是「預測第 k 個正解詞」的交差エントロピー，4 個位置的損失一起算、一起做<a href="#backprop">誤差逆伝播</a>。` }));
        } else {
          const decRuns = inf, posComputed = (inf * (inf + 1)) / 2;
          runStatus.className = "status status--" + (inf >= 4 ? "ok" : "info");
          runStatus.textContent = inf === 0
            ? "推論時：沒有正解。編碼器先把原文算 1 次（之後一直重複使用），解碼器從〈開始〉出發，每次只產生下一個詞。按「生成下一個詞」。"
            : `已生成 ${inf} 個詞：編碼器只算 1 次，解碼器已經跑了 ${decRuns} 次${inf >= 4 ? "，生成〈結束〉就停下" : ""}。若每次都把整段輸入重算，解碼器總共處理了 1${inf > 1 ? " + " + Array.from({ length: inf - 1 }, (_, k) => k + 2).join(" + ") : ""} = ${posComputed} 個位置（訓練時同一句只要 1 次、4 個位置）`;
          const shown = DEC_IN.slice(0, Math.max(1, inf + (inf < 4 ? 1 : 0)));
          runBox.replaceChildren(
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, `解碼器第 ${Math.min(inf + 1, 4)} 次的輸入：`), ...shown.map((t, k) => tok(t, k === shown.length - 1 && inf > 0 && inf < 4 ? "new" : ""))),
            h("div", { class: "tr-toks" }, h("span", { class: "hint" }, "已生成："), ...(inf ? TGT.slice(0, inf).map((t, k) => tok(t, k === inf - 1 ? "new" : "")) : [tok("（還沒有）", "dim")])),
            h("p", { class: "note", style: { margin: 0 }, html: `這裡假設模型已經訓練好、每次都挑機率最高的詞，所以生成的剛好是正解；真實推論時錯一個詞，後面就建立在錯誤上（見<a href="#teacher">教師強制</a>）。為了不必每次重算前面的位置，實作上會把前面算過的キー・バリュー存起來重複使用。` }));
        }
      }

      const vec = (v, d = 2) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      function renderMask() {
        const { A, Z } = attn(mask);
        const panelC = Atlas.color("--panel"), hot = Atlas.color("--hot");
        heatWrap.replaceChildren(h("table", { class: "tr-heat" },
          h("thead", null, h("tr", null, h("th", { class: "row" }, "看的詞 ↓ ／ 被看的詞 →"), ...DEC_IN.map((t) => h("th", null, t)))),
          h("tbody", null, ...A.map((r, i) => {
            const tr = h("tr", { class: i === rowSel ? "is-sel" : "" }, h("th", { class: "row" }, DEC_IN[i]),
              ...r.map((v, j) => mask && j > i
                ? h("td", { class: "masked", title: "被遮罩擋住" }, "✕")
                : h("td", { style: { background: Atlas.rgbStr(Atlas.mix(panelC, hot, Math.min(1, v * 1.2))), color: v > 0.5 ? "var(--panel)" : "var(--ink)" } }, fmt(v * 100, 1) + "%")));
            tr.addEventListener("click", () => { rowSel = i; renderMask(); });
            return tr;
          }))));
        const peek = attn(false).A[0][1];
        if (mask) { maskStatus.className = "status status--ok"; maskStatus.textContent = `有遮罩：每一列只把注意力分給自己和前面的詞（右上角都是 0）。〈開始〉只能看自己（100%），它要預測的「I」完全看不到`; }
        else { maskStatus.className = "status status--bad"; maskStatus.textContent = `拿掉遮罩：〈開始〉把 ${fmt(peek * 100, 1)}% 的注意力放在「I」上，而「I」正是它要預測的答案；這等於訓練時偷看答案，推論時卻沒有這個詞可看`; }
        const i = rowSel, sc = SC[i], vis = sc.map((v, j) => !(mask && j > i));
        const ex = sc.map((v, j) => (vis[j] ? Math.exp(v) : 0)), tot = ex.reduce((a, b) => a + b);
        maskCalc.innerHTML =
          row(`「${DEC_IN[i]}」的向量（第 ${i + 1} 個位置）`, `x = 詞向量 ${vec(E[i], 0)} + 位置 ${vec(PE(i))} = ${vec(X[i])}`) +
          row("分數 q·k ÷ √4", DEC_IN.map((t, j) => `${t} ${vis[j] ? fmt(sc[j], 2) : "−∞（遮住）"}`).join("　")) +
          row("softmax", DEC_IN.map((t, j) => `${t} ${vis[j] ? `e^${fmt(sc[j], 2)} ÷ ${fmt(tot, 2)} = ${fmt(A[i][j] * 100, 1)}%` : "e^−∞ = 0"}`).join("　")) +
          row("輸出 z = Σ 權重 × v", `<span class="hl">${vec(Z[i])}</span>`);
        const r = X[i].map((v, d) => v + Z[i][d]), mu = r.reduce((a, b) => a + b) / 4, va = r.reduce((a, b) => a + (b - mu) ** 2, 0) / 4, eps = 1e-5;
        const nrm = r.map((v) => (v - mu) / Math.sqrt(va + eps));
        anCalc.innerHTML =
          row("Add（殘差）", `x + z = ${vec(X[i])} + ${vec(Z[i])} = ${vec(r)}`) +
          row("平均 μ", `${fmt(mu, 3)}　<span class="k">變異數 σ²</span> ${fmt(va, 3)}`) +
          row("Norm", `(x + z − μ) ÷ √(σ² + ε) = <span class="hl">${vec(nrm)}</span>　<span class="k">（平均 0、變異數 1；只在這個詞自己的 4 個數之間計算）</span>`);
      }

      render(); renderRun(); renderMask();
      ctx.onTheme(() => renderMask());
      return { destroy() { player.pause(); } };
    },
  });
})();
