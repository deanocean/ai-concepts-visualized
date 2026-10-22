/* 隨機梯度下降與小批次（concepts/sgd.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、count*、cmpLeg、calc* 這類會以 HTML 顯示的項目
   - 圖例（leg*）放在固定寬度的框裡，盡量保持簡短 */
Atlas.strings("zh", "sgd", {
  lead: `<p><a href="#gd">梯度下降法</a>每走一步，都要把<b>全部</b>資料的誤差算一遍才知道坡度。資料有幾百萬筆時，算一次坡度就要很久，走一步也很久。<span class="term">隨機梯度下降法</span>（<span class="en">stochastic gradient descent, SGD</span>）改成每次只隨機拿一小部分資料來估坡度：估得沒那麼準，但同樣的計算量可以走好多步。</p>`,
  closing: `一次用全部資料（<span class="term">批次學習</span>（<span class="en">batch learning</span>））方向最準，但一個 epoch 只能走一步；一次用 1 筆，一個 epoch 走 16 步，很快靠近谷底，但方向搖擺、到了谷底附近仍停不下來。<span class="term">小批次學習</span>（<span class="en">mini-batch learning</span>）介於兩者之間，是現在訓練神經網路的標準做法。`,

  /* 控制項 */
  bLabel: "一次用幾筆資料算梯度（批次大小）",
  bAll: "全部 16 筆",
  bN: "{n} 筆",
  lrLabel: "學習率 η",
  btnStep: "更新一次",
  btnEpoch: "跑完這個 epoch",
  btnReset: "重設",

  /* 圖 */
  dataAria: "資料點與目前的直線",
  parAria: "參數平面上的損失等高線與路徑",
  capData: "資料與目前的直線 y = a·x + b",
  capPar: "參數 (a, b) 的損失地形（用全部資料算）",
  legBatch: "這一批",
  legAll: "全部用上",
  legUsed: "這 epoch 用過",
  legUnused: "還沒用",
  legErr: "誤差",
  legStepBatch: "這批估出的一步",
  legStepAll: "用全部資料的一步",
  markValley: "谷底",
  axisA: "a（斜率）",
  clickMap: "點圖可改變起點",

  /* 比較圖 */
  cmpAria: "三種做法跑 5 個 epoch 的損失比較",
  cmpLabel: "同樣算 5 個 epoch（每種做法都看過每筆資料 5 次），整份資料的損失比較",
  cmpAxisX: "epoch",
  cmpAxisY: "損失 − 谷底的損失（對數刻度）",
  cmpLeg: "━ {name}（{per} 次更新／epoch）：5 個 epoch 後多出 {extra}",
  cmpHint: "粗線是目前選的做法，圓點是你目前跑到的位置。",

  /* 計數 */
  countIter: "<span class=\"term\">迭代</span>（<span class=\"en\">iteration</span>）次數 <b>{n}</b>",
  countEpoch: "epoch（看完全部資料的次數）<b>{n}</b>",
  countGrads: "已算過的樣本梯度 <b>{n}</b> 筆",
  countLoss: "整份資料的損失 <b>{loss}</b>（谷底 {opt}）",

  /* 狀態列 */
  statusDone: "已跑完 {n} 個 epoch，按「重設」重來",
  statusFullIdle: "按「更新一次」開始：每次都用全部 16 筆資料，方向最準，但一個 epoch 只更新 1 次",
  statusFull: "第 {n} 次更新，離谷底 {dist}：方向一直很準，只是每看完一遍資料才走一步",
  statusNear: "已在谷底附近（距離 {dist}），但每批估出的方向不同，所以會一直在附近晃動",
  statusBatch: "下一步只用 {b} 筆估坡度：和用全部資料算的方向相差 {ang}°",

  /* 算式 */
  calcPosK: "目前位置",
  calcLossK: "損失",
  calcUpdK: "第 {n} 次更新（第 {e} 個 epoch 的第 {k} 批）　這批用",
  listSep: "、",
  calcErrK: "各筆誤差 a·x + b − y",
  errSep: "；",
  calcGradK: "這批估出的梯度",
  calcGradSym: "(平均(誤差 × x), 平均(誤差))",
  calcFullK: "用全部 16 筆算",
  calcNextK: "下一個位置",
  calcNextSym: "(a − η·梯度a, b − η·梯度b)",

  note: `損失用<span class="term">均方誤差</span>（<span class="en">mean squared error</span>）再除以 2（除以 2 只是讓梯度式子少一個係數 2）。每個 epoch 開始前把 16 筆資料重新洗牌，再依序切成幾批，所以每筆資料在一個 epoch 裡剛好用一次；這裡用固定的亂數，每次重設結果相同。「隨機」指的是隨機抽樣。每來一筆資料就馬上更新一次的做法也叫<span class="term">線上學習</span>（<span class="en">online learning</span>）。一批多筆的計算可以在 GPU 上同時做，所以實務上用 32〜256 筆左右的小批次；方向帶點隨機的搖擺，有時反而能幫忙離開<span class="term">局部最佳解</span>（<span class="en">local optimum</span>）或<span class="term">鞍點</span>（<span class="en">saddle point</span>）。學習率和每次走的方式怎麼改良，見 <a href="#optim">動量與 Adam</a>。`,
});
