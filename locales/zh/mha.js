/* 多頭注意力（concepts/mha.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、calc*、note 這類會以 HTML 顯示的項目
   - dims：4 個維度的名稱（以半形空白分隔，和自注意力頁相同）；dimSep：把它們連起來的分隔符號 */
Atlas.strings("zh", "mha", {
  lead: `<p>自注意力讓每個詞只分配<b>一組</b>注意力權重，但一個詞常同時需要好幾種關係：「it」要知道自己指的是誰（cat），也要知道它怎麼了（tired）。一組加起來 100% 的權重只能分一次。<span class="term">多頭注意力</span>（<span class="en">multi-head attention</span>）準備好幾組 Q・K・V，每一組叫一個<span class="term">頭</span>（<span class="en">head</span>），各自算注意力，最後再合起來。</p>`,
  closing: `每個頭在自己較小的空間裡找一種關係，不必和別的關係搶同一份 100%；把各頭的結果串接、再乘上 W<sub>O</sub> 混合，輸出就同時帶著多種關係的資訊。Transformer 原論文用 8 個頭，把 512 維分成 8 份、每頭 64 維，總計算量和單頭差不多。`,

  dims: "生物 代名詞 動作 狀態",
  dimSep: "／",
  headN: "頭 {n}",
  role1: "找指的是誰",
  role2: "找它怎麼了",
  headLabel: "{name}：{role}",
  headRow: "{name}（{role}）",

  /* 控制項 */
  modeLabel: "注意力的組數",
  modeMulti: "2 個頭（多頭）",
  modeSingle: "只用 1 個頭（同樣的參數）",
  toksAria: "選擇查詢的詞",
  toksLabel: "點任一個詞，看它的注意力怎麼分",
  btnMat: "看各頭的矩陣",
  btnMatHide: "收起矩陣",

  /* 圖與狀態列 */
  svgAria: "「{w}」在各頭的注意力",
  onlyHead: "唯一的一個頭",
  onlyHeadRow: "唯一的頭",
  sayFocus: "最關注「{top}」（{pct}）",
  sayEven: "分得很平均（最高「{top}」{pct}）",
  statusMulti: "「{w}」：頭 1 {h1}，頭 2 {h2}——兩個頭各分各的 100%",
  statusSingleSplit: "「{w}」：只有一組權重，「{a}」（{pa}）和「{b}」（{pb}）得分同一份 100%，兩邊都被稀釋",
  statusSingle: "「{w}」：只有一組權重，最高是「{top}」（{pct}）",

  /* 表格 */
  wLabel: "「{w}」分給各詞的注意力",
  thSum: "合計",
  outLabel: "「{w}」注意力之後的向量（4 格＝{dims}）",
  rowOrig: "原本的向量",
  rowMulti: "2 個頭（串接）",
  rowSingle: "1 個頭",
  rowCmpSingle: "對照：1 個頭",
  rowCmpMulti: "對照：2 個頭",

  /* 算式 */
  calcHeadK: "{name}（鍵・值只取「{a}・{b}」2 維）",
  calcScore2: "分數 q·k ÷ √2：",
  calcHeadZ: "　→ softmax 後加權 v：z{k} =",
  calcConcatK: "串接",
  calcWoK: "× W<sub>O</sub>（此例設成不改變）",
  calcSingleK: "同樣的參數，Q・K 都是 4 維",
  calcScore4K: "分數 q·k ÷ √4",
  calcSingleZK: "一次 softmax 後加權 v",
  matWQ: "{name} 的 W_Q",
  matWK: "{name} 的 W_K（＝W_V）",

  note: `每個頭用自己的 W<sub>Q</sub>・W<sub>K</sub>・W<sub>V</sub> 把 4 維向量投影到 2 維：頭 1 的鍵只留下「生物・代名詞」，頭 2 只留下「動作・狀態」，所以兩個頭自然看向不同的詞。「只用 1 個頭」是把同樣的矩陣並排成 4 維、放進同一個 softmax。這些矩陣是為了示範而設計的（W<sub>V</sub> 取和 W<sub>K</sub> 相同、W<sub>O</sub> 設成不改變），真實模型全由訓練學到，每個頭學到什麼關係也不是人指定的。單頭的計算見 <a href="#selfatt">自注意力</a>；多頭注意力在整個模型裡的位置見 <a href="#transformer">Transformer</a>。`,
});
