/* 動量與 Adam（concepts/optim.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、calc* 這類會以 HTML 顯示的項目
   - SGD、AdaGrad、RMSprop、Adam 是專有名稱，寫在程式裡；只有 nameMom 需要翻譯 */
Atlas.strings("zh", "optim", {
  lead: `<p>在<a href="#gd">梯度下降法</a>的細長山谷裡，陡的方向來回彈、緩的方向走很慢，而且整個模型只有一個學習率：調大，陡的方向會發散；調小，緩的方向更慢。改良的方向有兩個：<span class="term">動量</span>（<span class="en">momentum</span>）讓移動帶有慣性；AdaGrad、RMSprop 讓<b>每個參數</b>依自己過去的梯度大小自動調步伐。<span class="term">Adam</span> 把兩者合在一起。</p>`,
  closing: `動量靠慣性：來回彈的方向互相抵消、一直同向的方向越走越快。AdaGrad、RMSprop、Adam 把每個參數的梯度除以「它過去有多大」，所以不管某個方向多陡，步伐都差不多大；把山谷調陡，它們的路徑完全不變。AdaGrad 累加全部歷史，步伐只會越來越小；RMSprop 改成只記得最近，Adam 再加上慣性，是現在最常用的預設選擇。`,

  nameMom: "動量",

  /* 控制項 */
  focusLabel: "細看哪一個（調學習率、看算式）",
  kLabel: "y 方向比 x 方向陡幾倍 k",
  lrLabel: "學習率 η（細看的那一個）",
  btnOnly: "只顯示細看的那一個",
  btnShowAll: "五個都顯示",
  btnDefaults: "學習率全部還原",
  btnStep: "走一步",
  btnReset: "重設",

  /* 圖 */
  canvasAria: "五種最佳化方法在等高線上的路徑",
  clickMap: "點圖可改變起點",
  chartAria: "離谷底的距離隨步數的變化",
  chartLabel: "離谷底的距離（對數刻度）",
  chartNear: "距離 {near}（算「到達」）",
  chartAxisX: "步數",

  /* 比較表 */
  thMethod: "方法",
  thLr: "學習率 η",
  thReach: "到達谷底附近（距離 < {near}）",
  thDist: "目前離谷底",
  cellDead: "發散（第 {n} 步）",
  cellReach: "第 {n} 步",
  cellNotYet: "還沒",

  /* 狀態列 */
  statusIdle: "按「走一步」或「自動播放」：五種方法從同一點同時出發（最多 {max} 步）",
  statusDead: "{names} 發散了：y 方向太陡，同樣的學習率一步就跨得比谷寬還遠。把 k 調回 10 試試",
  nameSep: "、",
  statusEnd: "已走完 {max} 步",
  statusSgd: "第 {n} 步：SGD 的學習率 {lr} 剛好是替這個陡度調好的，所以最快到；把 k 調大，看誰撐得住",
  statusStep: "第 {n} 步",

  /* 算式 */
  calcDead: "{name} 已經發散，位置超出計算範圍",
  calcPosK: "{name}　第 {n} 步　位置",
  calcGradK: "梯度",
  calcMove: "移動",
  calcMomNote: "（移動量就是 v）",
  calcEff: "有效學習率",
  calcCorr: "修正後",
  calcNextK: "下一個位置",

  note: `這裡的地形是固定的式子，梯度每次都用整個地形算，所以「SGD」在這裡就是<a href="#gd">梯度下降法</a>；實際訓練時這五種方法都搭配<a href="#sgd">小批次</a>使用。動量的慣性也會讓它衝過谷底再折回，所以路徑常繞一大圈才停下。各方法的學習率不同，是因為它們的「一步」定義不同：AdaGrad、RMSprop、Adam 的移動量大約就是 η 本身。β、ρ 用常見的預設值（動量 β = 0.9；RMSprop ρ = 0.9；Adam β₁ = 0.9、β₂ = 0.999），Adam 的 m̂、v̂ 是補正剛開始時 m、v 從 0 起算而偏小的部分。其他改良：<span class="term">AdaDelta</span> 和 RMSprop 一樣改用最近的梯度，並且連學習率都不用設定；<span class="term">AdaBound</span>（和它的變形 <span class="term">AMSBound</span>）替 Adam 的有效學習率加上逐漸收窄的上下限，訓練後期變得像 SGD。學習率、β 這類訓練前由人決定的值叫<span class="term">超參數</span>（<span class="en">hyperparameter</span>），尋找的方法常用<span class="term">網格搜尋</span>（<span class="en">grid search</span>），把候選值的組合全部試一遍；或用<span class="term">隨機搜尋</span>（<span class="en">random search</span>），隨機抽組合來試。k = 10 時 SGD 最快、k 一調大卻發散，這種「沒有一種方法在所有問題上都最好」的事實叫<span class="term">沒有免費午餐定理</span>（<span class="en">no free lunch theorem</span>）：方法的優劣要看問題而定。`,
});
