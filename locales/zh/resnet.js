/* 跳躍連接（concepts/resnet.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、calc*、note 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "resnet", {
  lead: `<p>網路疊得越深，<span class="term">反向傳播</span>（<span class="en">backpropagation</span>）時梯度要連乘的次數越多，前面的層就學不動，這就是<span class="term">梯度消失</span>（<span class="en">vanishing gradient</span>）。<span class="term">跳躍連接</span>（<span class="en">skip connection</span>）在層的旁邊多拉一條捷徑，把輸入直接加到輸出上，梯度也能沿這條捷徑原封不動地傳回去。</p>`,
  closing: `有了捷徑，梯度每過一層乘的是「1 + 層的斜率」，而不是只有「層的斜率」；就算層的斜率很小，那個 1 也讓梯度不會連乘到 0。層本身只需要學「在輸入之上再加多少」，也就是<span class="term">殘差</span>（<span class="en">residual</span>）。ResNet 用這個結構把網路疊到 152 層。`,

  /* 控制項 */
  nLabel: "層數",
  nFmt: "{n} 層",
  wLabel: "每層的權重 w",
  btnPrev: "◀ 上一步",
  btnNext: "往前傳一層 ▶",
  btnAll: "全部算完",
  btnReset: "重設",

  /* 網路圖 */
  svgAria: "沒有捷徑與有捷徑的網路，各層的輸出與傳回的梯度",
  lanePlain: "普通的網路（沒有捷徑）",
  laneRes: "加上跳躍連接（弧線＝捷徑：輸出 = 層的結果 + 輸入）",
  laneGrad: "梯度",
  laneIn: "輸入",
  laneOut: "輸出",
  layerK: "第 {k} 層",

  /* 狀態列與算式 */
  statusStart: "兩個網路都已算好正向（藍框＝每層的輸出）。按「往前傳一層」，把梯度從最後一層同時傳回去",
  calcPlainK: "普通",
  calcPlain: "第 k 層的輸出 = σ(w × 上一層)　例：第 1 層 σ({w} × {x}) = {a}",
  calcResK: "有捷徑",
  calcRes: "第 k 層的輸出 = 上一層 + σ(w × 上一層)　例：第 1 層 {x} + {s} = {a}",
  statusFirst: "從最後一層（第 {n} 層）開始，兩邊的梯度都設為 1",
  calcFirstK: "起點",
  calcFirst: "把最後一層收到的梯度當作 1，看它傳回每一層時還剩多少",
  calcStepPK: "普通：第 {k1} → {k} 層乘",
  calcStepP: `w × σ′(z) = {w} × σ′({z}) = {w} × {d} = <span class="hl">{f}</span>　梯度 {g}`,
  calcStepRK: "有捷徑：乘",
  calcStepR: `1（捷徑） + w × σ′(z) = 1 + {w} × {d} = <span class="hl">{f}</span>　梯度 {g}`,
  statusStep: "傳回第 {k} 層：普通 {gp}，有捷徑 {gr}",
  statusDone: "第 1 層收到的梯度：普通的網路 {gp}{vanish}；有捷徑 {gr}{kept}",
  doneVanish: "（梯度消失）",
  doneKept: "，保住了",

  /* 長條圖 */
  barAria: "各層收到的梯度：普通 vs 有捷徑（對數刻度）",
  barTitle: "各層收到的梯度（最後一層 = 1，對數刻度：每格差 10 倍）",
  legPlain: "普通",
  legRes: "有捷徑",
  barAxis: "← 層",

  note: `σ′ 是 Sigmoid 的斜率 σ(z)(1 − σ(z))，最大只有 0.25，所以 w = 1 時普通的網路每層至少縮成四分之一。示範網路每層只有一個神經元；真正的 ResNet 每個捷徑跨過兩三層卷積（中間有<span class="term">批次正規化</span>（<span class="en">batch normalization</span>）和 ReLU），但「梯度多一條乘 1 的路」的道理相同。在真正的 ResNet 裡，殘差那一路的權重若都學成 0，區塊就等於把輸入原樣傳下去，所以多疊的層至少不容易讓結果變差。同一個網路沒有捷徑時的完整情形見 <a href="#vanish">梯度消失問題</a>。`,
});
