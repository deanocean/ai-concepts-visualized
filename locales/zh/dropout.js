/* Dropout（concepts/dropout.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、stat*、calc* 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "dropout", {
  lead: `<p>大網路很容易<a href="#overfit">過擬合</a>：神經元之間會互相配合，一起把訓練資料背起來。訓練好幾個不同的模型再把答案平均，叫<span class="term">集成學習</span>（<span class="en">ensemble learning</span>），能減輕這個問題，但大網路訓練一個就很久。<span class="term">Dropout</span>（<span class="en">dropout</span>）在訓練的每一步隨機讓一部分神經元輸出 0，等於每次都在訓練一個不同的小網路，而且它們共用同一組權重。</p>`,
  closing: `訓練時每次關掉不同的神經元，每個神經元就不能依賴特定的夥伴，必須自己學到有用的特徵；推論時全部打開，再把輸出的比例調回訓練時的平均，結果相當於把所有子網路的答案平均起來。所以 Dropout 是一種正則化，而且只在訓練時使用。`,

  /* 控制項 */
  modeLabel: "模式",
  modeTrain: "① 訓練：每次隨機關掉",
  modeAll: "② 全部 64 個子網路",
  pLabel: "丟棄率 p（每個神經元被關掉的機率）",
  schLabel: "比例怎麼調整",
  schInv: "訓練時 ÷ (1 − p)",
  schOrig: "推論時 × (1 − p)",
  btnTrainOnce: "訓練一次（抽新的遮罩）",
  btnTrain100: "連抽 100 次",
  btnReset: "重設",
  stripLabel: "每次訓練時的輸出（一個點＝一次）",
  gridLabel: "64 種開／關組合（藍點＝開著的神經元；點一格看它的計算）",

  /* 網路圖、點狀圖、64 格 */
  netAria: "網路圖：被關掉的神經元以虛線表示",
  stripAria: "每次抽到的輸出",
  netIn: "輸入",
  netHidOn: "隱藏層（ReLU）：開著 {n} 個",
  netHidAll: "隱藏層（ReLU）：全部開著",
  netOutTrain: "訓練時的輸出",
  netOutInfer: "推論時的輸出",
  mkInfer: "推論時 {v}",
  mkRaw: "不調整就全開 {v}",
  mkMean: "平均 {v}",
  cellTitle: "機率 {p}%",

  /* 統計與狀態列 */
  statN: "已訓練 <b>{n}</b> 次",
  statMean: "輸出的平均 <b>{m}</b>",
  statInferInv: "推論時（全開） <b>{v}</b>",
  statInferOrig: "推論時（全開後 × {k}） <b>{v}</b>",
  sIdle: "按「訓練一次」：每個神經元各自以機率 p 被關掉",
  sFew: "這次關掉了 {off} 個，輸出 {z}；每次都不一樣，多抽幾次看平均",
  sClose: "抽了 {n} 次，平均 {mean}，已經很接近推論時全開的 {inf}",
  sFar: "抽了 {n} 次，平均 {mean}，和推論時全開的 {inf} 還差 {diff}；抽得越多，平均越接近",
  sAll: "64 個子網路依出現機率加權平均：{avg}；推論時全開：{inf}，完全相同",

  /* 算式 */
  calcInferK: "推論時（全開）",
  calcInferScaled: "× {k} 後 z = <b>{v}</b>",
  calcMaskK: "這次的遮罩",
  calcMaskNote: "（1＝留下，0＝關掉）",
  calcAvgK: "加權平均",
  calcAvg: "Σ 機率 × 輸出（{eq}）",
  eqHalf: "每一種的機率都是 0.5⁶ = 1/64",
  eqGen: "關掉 k 個的組合，機率是 {p}ᵏ × {k}⁽⁶⁻ᵏ⁾",
  calcSelK: "選中的這一個",
  calcSelProb: "機率 {p}%",
  calcSelOut: "輸出",
  calcPickHint: "點上面任一格，看那個子網路的計算",

  note: `兩種調整比例的方法結果相同：原始論文是推論時把輸出乘上 (1 − p)；現在的框架多半改成訓練時把留下的值除以 (1 − p)，稱為 <span class="term">Inverted Dropout</span>（<span class="en">inverted dropout</span>），推論時就什麼都不用改。如果不調整，推論時全部打開，下一層收到的訊號會比訓練時大很多（「推論時 × (1 − p)」模式下的紅線）。這個示範只有一層 Dropout、後面直接接線性的輸出，所以平均剛好完全相等；層數多、中間有非線性時只是近似，但實務上效果很好。6 個神經元就有 64 種子網路；真實網路有上千個神經元，子網路的數量多到天文數字，每一步只訓練其中一個。常見的丟棄率是隱藏層 0.5、輸入層 0.2 左右。隱藏層的值是用示範權重算出的（輸入 × 權重再過 ReLU）。`,
});
