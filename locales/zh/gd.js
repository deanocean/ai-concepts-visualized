/* 梯度下降法與學習率（concepts/gradient-descent.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note*、calc* 這類會以 HTML 顯示的項目
   - note1d 依條件換句子，寫成函式 */
Atlas.strings("zh", "gd", {
  lead: `<p>訓練模型，就是找出讓<b>損失</b>（預測和正確答案差多少）最小的參數。但我們看不到整片地形，只能在腳下量出坡度，然後往下坡走一步。這個做法叫<span class="term">梯度下降法</span>（<span class="en">gradient descent</span>）；每一步跨多大，由<span class="term">學習率</span>（<span class="en">learning rate</span>）決定。</p>`,
  closing: `學習率太小要走非常多步；太大會跨過谷底來回震盪，甚至越彈越遠而發散。地形有好幾個谷時，從哪裡出發，就決定最後落進哪一個谷。`,

  /* 控制項 */
  modeLabel: "地形",
  mode1d: "1D 曲線",
  mode2d: "2D 等高線",
  surfLabel: "2D 地形",
  surfValley: "細長山谷",
  surfTwin: "兩個谷",
  lrLabel: "學習率 η",
  presetLabel: "快速選擇",
  presetBtn: "{label}（{v}）",
  presetSmall: "太小",
  presetGood: "適中",
  presetBig: "偏大",
  presetHuge: "太大",
  btnStep: "走一步",
  btnReset: "重設",
  statusIdle: "按「走一步」開始",

  /* 1D 圖 */
  svgAria: "損失曲線與目前位置",
  miniLabel: "損失隨步數的變化",
  axisW: "參數 w",
  curveLabel: "損失 L(w) = (w − 1)²",
  minLabel: "谷底 w = 1",
  flyRight: "球飛出畫面 →",
  flyLeft: "← 球飛出畫面",
  clickCurve: "點曲線可改變起點",
  miniAxis: "步數（0〜{n}）",

  /* 1D 狀態列 */
  s1Overflow: "發散：第 {n} 步時 w 已經大到無法顯示",
  s1Done: "已到谷底：共走了 {n} 步",
  s1Diverge: "發散：跨過谷底後落得更遠，越彈越遠",
  s1DivergeIdle: "這個學習率會發散，按「走一步」看看",
  s1Bounce: "在谷底兩側來回彈跳，永遠停不下來",
  s1Osc: "震盪：每步都跨過谷底，但越跨越短，最後仍會收斂",
  s1OscIdle: "這個學習率會跨過谷底來回震盪",
  s1Slow: "前進緩慢：第 {n} 步，離谷底還有 {dist}",
  s1Steady: "穩定往谷底前進：第 {n} 步，離谷底 {dist}",

  /* 1D 算式 */
  calcPosK: "目前位置",
  calcLossK: "損失",
  calcStepPosK: "第 {n} 步　目前位置",
  calcSlopeK: "斜率（梯度）",
  calcMoveK: "這一步移動",
  calcMoveSym: "−η × 斜率",
  calcNextK: "下一個位置",
  note1d: ({ r, flip, near }) => `這條曲線上，每走一步，離谷底的距離都變成原本的 |1 − 2η| = <b class="mono">${r}</b> 倍${flip ? "，而且換到谷底另一側" : ""}。${near ? "小於 1 才會越來越近。" : "大於或等於 1，所以不會靠近。"}`,

  /* 2D 圖 */
  canvasAria: "損失的等高線圖與移動路徑",
  markValley: "谷底",
  markGlobal: "全域最佳解",
  markLocal: "局部最佳解",
  markSaddle: "鞍點",
  clickMap: "點圖可改變起點",

  /* 2D 狀態列 */
  s2Out: "發散：第 {n} 步已經衝出地圖",
  s2Local: "停在局部最佳解（第 {n} 步）：四周都是上坡，梯度≈0，走不出去",
  s2ReachGlobal: "抵達全域最佳解：共 {n} 步",
  s2ReachValley: "抵達谷底：共 {n} 步",
  s2Zigzag: "鋸齒前進：陡的 y 方向來回彈，緩的 x 方向慢慢走（第 {n} 步）",
  s2Step: "第 {n} 步，損失 {loss}",

  /* 2D 算式 */
  calcOut: "位置已超出地圖",
  calcStopK: "停在",
  calcStepPos2K: "第 {n} 步　位置",
  calcGradK: "梯度（各方向的坡度）",
  calcNext2Sym: "(x − η·梯度x, y − η·梯度y)",

  noteValley: `顏色越深＝損失越大，線條是等高線。這個山谷 y 方向的坡比 x 方向陡 10 倍：同一個學習率，對 y 方向可能太大（來回彈），對 x 方向卻太小（走很慢）。改良的走法見 <a href="#optim">動量與 Adam</a>；每步只用部分資料估坡度，見 <a href="#sgd">隨機梯度下降與小批次</a>。`,
  noteTwin: `這片地形有兩個谷：左邊較深，是<span class="term">全域最佳解</span>（<span class="en">global optimum</span>）；右邊較淺，是<span class="term">局部最佳解</span>（<span class="en">local optimum</span>）。兩谷之間的方塊是<span class="term">鞍點</span>（<span class="en">saddle point</span>）：沿 x 是山頂、沿 y 是谷底，梯度同樣是 0。試著從左、右兩側分別出發。`,
});
