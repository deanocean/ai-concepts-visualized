/* 反向傳播與連鎖律（concepts/backprop.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、calc* 這類會以 HTML 顯示的項目
   - w₁、h₁、ŷ 這類符號和算式寫在程式裡 */
Atlas.strings("zh", "backprop", {
  lead: `<p>網路輸出錯了，可是要調整的權重散在每一層：每個權重該為這個誤差負多少責任、該往哪個方向調多少？這叫<span class="term">信用分配問題</span>（<span class="en">credit assignment problem</span>）。<span class="term">反向傳播</span>（<span class="en">backpropagation</span>）從損失出發往輸入方向倒著走，每經過一個節點就乘上那裡的局部斜率，走一趟就算出所有權重的梯度；背後的數學是<span class="term">連鎖律</span>（<span class="en">chain rule</span>）。</p>`,
  closing: `每個權重的梯度，都是從損失沿著連線倒著走、把途中每一段的局部斜率連乘起來。離輸出越遠的權重，要乘的項越多；如果這些項都小於 1，梯度就會越乘越小，這就是下一個問題 <a href="#vanish">梯度消失問題</a>。`,

  /* 控制項 */
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnReset: "重設",
  btnUpdate: "用梯度更新權重（η = {eta}）",
  btnUpdate10: "連續更新 10 次",

  /* 網路圖 */
  svgAria: "2 → 2 → 1 的小網路，正向值與反向梯度",
  layerIn: "輸入層",
  layerHidden: "隱藏層（Sigmoid）",
  layerOut: "輸出層",
  layerLoss: "損失",
  target: "正確答案 t = {v}",
  legend: "藍框＝正向算出的值　橘色 ∂＝損失對它的梯度（∂L/∂…）",
  clickEdge: "點任一條連線，看它的連鎖律",

  /* 狀態列與算式：① 正向傳播 */
  status0: "① 正向傳播：按「下一步」，把輸入一層層往右算",
  calcInK: "輸入",
  calcIn: "x₁ = {x1}，x₂ = {x2}；正確答案 t = {t}",
  calcWK: "權重",
  listSep: "，",
  status1: "① 正向傳播：算出隱藏層",
  calcSigK: "σ 是 Sigmoid",
  calcSig: "σ(z) = 1 ÷ (1 + e<sup>−z</sup>)，把任何數壓到 0〜1 之間",
  status2: "① 正向傳播：算出輸出",
  status3: "① 正向傳播完成：算出損失",
  statusUpdated: "已更新 {n} 次：損失 {from} → {to}",
  calcNextK: "接下來",
  calcNext: "按「下一步」，從損失倒著算每個權重的梯度",

  /* ② 反向傳播 */
  status4: "② 反向傳播：從損失開始，先算 ∂L/∂ŷ",
  calcSelfK: "L 對 L 自己的梯度",
  calcSelf: "= 1（起點）",
  calcMeanK: "意思",
  calcMean: "ŷ 增加 0.01，L 大約變化 {d}；負號＝ŷ 應該變大",
  status5: "② 反向傳播：往回一層，經過 ŷ = v₁h₁ + v₂h₂",
  status6: "② 反向傳播：穿過 Sigmoid，乘上它在那一點的斜率",
  calcSlopeK: "Sigmoid 的斜率",
  calcSlope: "σ′(z) = h × (1 − h)，最大只有 0.25",
  status7: "② 反向傳播完成：6 個權重的梯度都有了（目前看 {w}）",

  /* 連鎖律 */
  chainK: "連鎖律",
  chainTermsK: "各段的局部斜率",
  chainSubK: "代入",
  chainCheckK: "數值驗算：把 {w} 加減 0.0001 實際重算損失",
  chainUpdK: "更新",
  chainUpdSym: "{w} − η × 梯度",

  /* 損失圖 */
  lossLabel: "損失隨更新次數的變化",
  lossAxis: "更新次數（0〜{n}）",

  note: `∂L/∂w 讀作「L 對 w 的<span class="term">偏微分</span>（<span class="en">partial derivative</span>）」，意思是只讓 w 增加一點點時，損失 L 會跟著變多少（斜率）；梯度為負，表示把 w 調大能讓損失變小。示範網路省略了<span class="term">偏置</span>（<span class="en">bias</span>），只有 6 個權重；真實模型有數百萬個權重，數字是訓練學到的，但算法相同。損失用 ½(ŷ − t)²，乘 ½ 是為了讓微分結果剛好是 ŷ − t。`,
});
