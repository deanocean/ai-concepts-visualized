/* 梯度消失問題（concepts/vanishing.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note*、calc* 這類會以 HTML 顯示的項目
   - Sigmoid、tanh、ReLU 是函數名，寫在程式裡 */
Atlas.strings("zh", "vanish", {
  lead: `<p>層數越多，網路能表達的東西越多；但早期的深層網路，層一多就訓練不起來。原因出在反向傳播：梯度從輸出往前傳，每經過一層都要乘上一個數，傳到前面的層時可能小到幾乎是 0，那些層的權重就幾乎不會更新。這叫<span class="term">梯度消失問題</span>（<span class="en">vanishing gradient problem</span>）。</p>`,
  closing: `梯度每往前一層，就乘上「權重 × 激活函數的斜率」。這個數小於 1，乘十幾次就趨近 0（梯度消失）；大於 1，就越乘越大，稱為<span class="term">梯度爆炸</span>（<span class="en">exploding gradient</span>）。對策是讓每層乘的數保持在 1 附近：隱藏層改用 ReLU、配合適當的權重初始值，以及<span class="term">跳躍連接</span>（<span class="en">skip connection</span>）和正規化層。`,

  /* 控制項 */
  fnLabel: "隱藏層的激活函數",
  nLabel: "層數",
  nFmt: "{n} 層",
  wLabel: "每層的權重 w",
  presetLabel: "快速選擇",
  presetFmt: "{f}、w = {w}",
  btnPrev: "◀ 上一步",
  btnNext: "往前傳一層 ▶",
  btnAll: "全部算完",
  btnReset: "重設",

  /* 網路圖 */
  svgAria: "深層網路各層的輸出與傳回的梯度",
  nodeIn: "輸入",
  rowGrad: "梯度",
  layerN: "第 {k} 層",
  nodeOut: "輸出",
  nodeLoss: "→ 損失",

  /* 狀態列與算式 */
  status0: "正向傳播已算好每層的輸出（藍框）。按「往前傳一層」，把梯度從最後一層傳回去",
  calcLayerK: "每層的計算",
  calcLayer: "輸出 = f(w × 上一層的輸出)，f = {f}",
  calcExK: "例：第 1 層",
  status1: "從最後一層（第 {n} 層）開始，梯度設為 1",
  calcStartK: "起點",
  calcStart: "把最後一層收到的梯度當作 1（100%），看它傳回每一層時還剩多少",
  calcBackK: "從第 {from} 層傳回第 {to} 層，要乘上",
  reluPos: "（z<sub>{i}</sub> = {z} > 0，斜率 1）",
  reluNeg: "（z<sub>{i}</sub> = {z} ≤ 0，斜率 0）",
  calcGradK: "梯度",
  statusBack: "傳回第 {k} 層：梯度剩 {g}",
  statusVanish: "梯度消失：第 1 層只收到 {g}，約是最後一層的 {n} 分之一",
  statusExplode: "梯度爆炸：第 1 層收到 {g}，是最後一層的 {g} 倍，更新會大到失控",
  statusDecay: "第 1 層收到 {g}：有衰減，層數再多就會消失",
  statusOk: "第 1 層收到 {g}：梯度大致保住了",

  /* 長條圖 */
  barAria: "各層收到的梯度（對數刻度）",
  barTitle: "各層收到的梯度（最後一層 = 1，對數刻度：每格差 10 倍）",
  barAxis: "← 層",

  noteSigmoid: `Sigmoid 的斜率最大只有 0.25，所以 w = 1 時每往前一層，梯度至少縮成原本的四分之一；8 層就剩萬分之一以下。把 w 調大雖然能抵銷一些，但 z 也跟著變大、落進 Sigmoid 兩端的平坦區，斜率反而更小。`,
  noteTanh: `tanh 的斜率最大是 1，這個例子裡每層乘的數接近 1，衰減比 Sigmoid 慢得多；但輸出一旦接近 ±1，斜率就趨近 0，梯度同樣會消失。試著把 w 調到 2 以上。`,
  noteRelu: `ReLU 在正側斜率固定是 1，每層乘的數就等於 w：w = 1 時梯度完整傳回；w 小於 1 仍會消失，大於 1 則越乘越大，稱為<span class="term">梯度爆炸</span>（<span class="en">exploding gradient</span>）。所以 ReLU 也要搭配適當的權重初始值。`,
  noteTail: `示範網路每層只有一個神經元、沒有偏置；真實網路每層有很多神經元，但「每過一層乘一次」的道理相同。各函數的形狀見 <a href="#act">激活函數</a>；同樣的連乘沿著時間發生，就是 <a href="#rnn">RNN 與梯度消失</a>；加上捷徑後的對照見 <a href="#resnet">跳躍連接（ResNet）</a>。`,
});
