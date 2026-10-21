/* 深度可分離卷積（concepts/dwsep.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、lab*、cap*、calc*、note* 這類會以 HTML 顯示的項目
   - 注意：中文的「列」是橫排（row）、「行」是直排（column），日文剛好相反 */
Atlas.strings("zh", "dwsep", {
  lead: `<p>標準卷積的每個濾波器同時處理「空間上的 3×3」和「所有通道」，通道一多，權重和計算量就爆增，手機跑不動。<span class="term">深度可分離卷積</span>（<span class="en">depthwise separable convolution</span>，以下簡稱 DS）把這件事拆成兩步：先用<span class="term">逐深度卷積</span>（<span class="en">depthwise convolution</span>）讓各通道自己看空間，再用 1×1 的<span class="term">逐點卷積</span>（<span class="en">pointwise convolution</span>）把通道混合。</p>`,
  closing: `拆開後，空間的工作（逐深度）和通道的工作（逐點）各做一次就好，不必為每個輸出通道重做一遍。計算量約變成原本的 1/輸出通道數 + 1/(卷積核邊長)²，3×3 時大約少 8～9 倍；代價是能表達的濾波器種類受限。MobileNet 就是靠它把模型做小。`,

  /* 示範資料的名稱 */
  chVert: "直線",
  chHorz: "橫線",
  chDiag: "斜線",
  dVert: "垂直邊緣",
  dHorz: "水平邊緣",
  dCross: "十字加總",

  /* 控制項 */
  viewLabel: "看哪一部分",
  viewStd: "① 標準卷積（一步做完）",
  viewDs: "② 深度可分離卷積（兩步）",
  viewCalc: "③ 參數量計算機",
  oLabel: "看第幾個輸出通道",
  outN: "輸出 {o}",
  cinLabel: "輸入通道數",
  coutLabel: "輸出通道數",
  kLabel: "卷積核邊長",
  hLabel: "特徵圖邊長",
  preLabel: "快速選擇",
  preSmall: "這頁的小例子（3→4、輸出 3×3）",
  preMobile: "MobileNet 中段（256→256、14×14）",

  /* 格子（螢幕報讀用） */
  cellAria: "{name}，第 {r} 列第 {c} 行，值 {v}",
  nameIn: "輸入通道 {c}",
  nameFilt: "濾波器第 {c} 片",
  nameOut: "輸出通道 {o}",
  nameDw: "逐深度卷積的結果，通道 {c}",
  nameDwF: "逐深度卷積的濾波器 {c}",

  /* 長條圖 */
  barW: "權重數",
  barM: "乘法次數",
  barStd: "標準卷積",
  barDs: "DS",
  fMulStd: "{h}×{h} 格 × {kkc} × {cout}",
  fMulDs: "{h}×{h} 格 × ({kk}×{cin} + {cin}×{cout})",

  /* 共用 */
  capIn: "通道 {c}<br>{name}",

  /* ① 標準卷積 */
  capSlice: "第 {c} 片",
  labInStd: "輸入 5×5×3 <span>（橘框＝輸出那一格看的範圍）</span>",
  labFilt: "輸出 {o} 的濾波器 3×3×3 <span>（一次看完 3 個通道，27 個權重）</span>",
  labOut: "輸出 3×3×4 <span>（每個輸出通道各有一個這樣的濾波器）</span>",
  calcStdK: "輸出 {o}、第 {r} 列第 {c} 格",
  calcStd: `3 個通道各 9 個乘積（共 27 個）分別加總：{expr} = <span class="hl">{v}</span>`,
  calcLayerK: "整層",
  calcLayer: "4 個輸出通道 × 9 格，每格都要做 27 次乘法",
  statusStd: "標準卷積：4 個濾波器 × 27 個權重 = 108 個權重；乘法 9 格 × 4 × 27 = 972 次",
  hintStd: "點輸出的格子，換一格看；上面的分段鈕換輸出通道。",
  noteStd: "這個標準卷積的權重刻意設成「② 的逐點係數 × 逐深度濾波器」，所以兩邊的輸出完全相同，方便比較計算量。一般訓練出的標準卷積權重可以是任意組合，DS 只能表示其中這一類。",

  /* ② 深度可分離卷積 */
  capDw: "通道 {c} 的結果",
  capDwF: "通道 {c} 專用<br>{name}",
  pwHead: "1×1 係數",
  pwCh: "通道 {c}",
  pwCap: "逐點卷積的權重：4 組 × 3 個",
  labInDs: "輸入 5×5×3",
  labDw: "① 逐深度卷積：每個通道只用自己的 3×3 <span>（3 × 9 = 27 個權重）</span>",
  labPw: "② 逐點卷積：1×1 卷積，在每一格把 3 個通道加權混合 <span>（4 × 3 = 12 個權重）</span>",
  calcDwK: "① 第 {r} 列第 {c} 格",
  calcDwPart: "通道 {c}：9 個乘積加總 = {v}",
  calcDwSep: "；",
  calcPwK: "② 混成輸出 {o}",
  calcCmpK: "和 ① 標準卷積比",
  calcCmp: "輸出完全相同（{res}）",
  cmpSame: "4 個通道 9 格全部一致",
  cmpDiff: "不一致",
  statusDs: "兩步合計 27 + 12 = 39 個權重（標準 108）；乘法 9×3×9 + 9×4×3 = 351 次（標準 972）",
  hintDs: "點任一張 3×3 結果的格子換位置；上面的分段鈕換輸出通道。",
  noteDs: "逐深度卷積只看空間、不混通道；逐點卷積只混通道、不看空間。小例子只有 4 個輸出通道，省得不多；通道越多省越多，到 ③ 試試。",

  /* ③ 計算機 */
  calcStdRowK: "標準",
  calcDsRowK: "DS",
  calcWeights: "權重 {expr} = {n}",
  calcRatioK: "比例",
  calcRatio: `DS ÷ 標準 = 1/{cout} + 1/{kk}² = {a} + {b} = <span class="hl">{r}</span>　<span class="k">（乘法次數的比例也一樣）</span>`,
  statusCalc: "DS 只需要標準卷積的 {pct}%，約少 {x} 倍",
  noteCalc: "乘法次數假設步幅 1、填充讓特徵圖大小不變。輸出通道多時 1/輸出通道數 幾乎是 0，比例就由 1/(卷積核邊長)² 決定：3×3 約 1/9。",
});
