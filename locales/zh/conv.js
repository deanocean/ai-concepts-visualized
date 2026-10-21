/* 卷積（concepts/convolution.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、*Label、cap*、calc*、size、edge*、param 這類會以 HTML 顯示的項目
   - 注意：中文的「列」是橫排（row）、「行」是直排（column），日文剛好相反 */
Atlas.strings("zh", "conv", {
  lead: `<p>辨識圖像時，關鍵是局部的形狀（邊、角、線條），而同一種形狀可能出現在圖上任何位置。<span class="term">卷積</span>（<span class="en">convolution</span>）拿一個 3×3 的小數字表，也就是<span class="term">濾波器</span>（<span class="en">filter</span>），在圖上滑動，每停一處就算出「這裡和這個形狀有多像」。</p>`,
  closing: `濾波器的數字和局部圖樣越吻合，相乘再加總的值就越大，特徵圖上那一格的顏色就越深。同一組 9 個數字走遍整張圖，所以參數很少，形狀出現在哪裡都抓得到；CNN 訓練時學的，就是這些數字。`,

  /* 控制項 */
  imgLabel: "輸入圖像（點格子可塗改）",
  imgBar: "直條",
  imgRing: "方框",
  imgDiag: "斜線",
  imgEll: "L 形",
  imgBlank: "空白",
  filterLabel: "濾波器",
  fVert: "垂直邊緣",
  fHorz: "水平邊緣",
  fDiag: "斜線（＼）",
  fBlur: "平均（模糊）",
  fCustom: "自訂",
  strideLabel: `<span class="term">步幅</span>（<span class="en">stride</span>）`,
  stride1: "1 格",
  stride2: "2 格",
  padLabel: `<span class="term">填充</span>（<span class="en">padding</span>）`,
  pad0: "0（不補）",
  pad1: "1（外圍補 0）",
  btnPrev: "◀ 上一格",
  btnNext: "下一格 ▶",
  btnAll: "全部算完",
  btnClear: "清空",

  /* 格子（螢幕報讀用） */
  cellIn: "輸入第 {r} 列第 {c} 行，值 {v}",
  cellFilter: "濾波器第 {r} 列第 {c} 行，值 {v}，點一下改變",
  cellOut: "特徵圖第 {r} 列第 {c} 行",

  /* 圖說 */
  capIn: "<b>輸入</b> {n}×{n}",
  capInPad: "（虛線＝補的 0）",
  capFilter: "<b>濾波器</b> 3×3<br>點格子可改數字",
  capOut: `<b><span class="term">特徵圖</span></b>（<span class="en">feature map</span>）{n}×{n}<br>點格子可跳到那一格`,

  /* 算式 */
  calcStart: "按「下一格」：濾波器蓋住左上角 3×3，對應格相乘後加總，結果寫進特徵圖第一格。",
  calcCellK: "第 {r} 列第 {c} 格：輸入 × 濾波器（逐格相乘）",
  calcSumK: "加總",
  sizeK: "特徵圖邊長",
  size: "= (輸入 {n} + 2×填充 {p} − 濾波器 {k}) ÷ 步幅 {s} + 1 = ({a}) ÷ {s} + 1 = <b>{out}</b>",
  edgeHorz: "這個濾波器上排是 +1、下排是 −1：遇到「上面有、下面沒有」的邊得正數（藍），「上面沒有、下面有」得負數（紅）；沒有邊的平坦處正負抵消，得 0。",
  edgeVert: "這個濾波器左行是 +1、右行是 −1：遇到「左邊有、右邊沒有」的邊得正數（藍），「左邊沒有、右邊有」得負數（紅）；沒有邊的平坦處正負抵消，得 0。",
  param: `這個濾波器只有 <b class="mono">9</b> 個權重，走遍整張圖都用同一組。若改用<span class="term">全連接層</span>（<span class="en">fully connected layer</span>），讓 {outN}×{outN} 個輸出各自連到 {inN}×{inN} 個輸入，需要 {inCells} × {outCells} = <b class="mono">{big}</b> 個權重。濾波器也叫<span class="term">卷積核</span>（<span class="en">kernel</span>）。接下來常接 <a href="#pool">池化</a>；擴大看得到的範圍見 <a href="#dilated">感受野與空洞卷積</a>，減少計算量見 <a href="#dwsep">深度可分離卷積</a>。`,
});
