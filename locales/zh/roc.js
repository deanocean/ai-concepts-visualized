/* ROC 曲線與 AUC（concepts/roc.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、cm*、calc*、note 這類會以 HTML 顯示的項目
   - dLevels：區分能力滑桿的 8 個刻度名稱，以「/」分隔（順序固定，由「和亂猜一樣」到「完美分開」） */
Atlas.strings("zh", "roc", {
  lead: `<p>分類模型通常輸出一個「是陽性的機率」，要再切一條<span class="term">門檻</span>（<span class="en">threshold</span>）才變成「陽性／陰性」的判定。門檻放低，真正的陽性抓得多，但誤報也多；放高則相反。只看某一個門檻的<span class="term">準確率</span>（<span class="en">accuracy</span>），看不出模型本身好不好。<span class="term">ROC 曲線</span>（<span class="en">ROC curve</span>）把所有門檻的結果畫在一起，<span class="term">AUC</span>（<span class="en">area under the curve</span>）再把它濃縮成一個數。</p>`,
  closing: `門檻決定「抓多少、誤報多少」的取捨，要看任務選（漏掉病人代價大，就把門檻放低、重視召回率）；ROC 曲線和 AUC 則和門檻無關，衡量的是模型把陽性排在陰性前面的能力：AUC 1 是完美分開，0.5 和亂猜一樣。`,

  /* 控制項 */
  dLabel: "模型的區分能力（兩類分數拉開多少）",
  dLevels: "和亂猜一樣/很弱/弱/普通/不錯/很好/優秀/完美分開",
  tLabel: "門檻（分數 ≥ 門檻就判為陽性）",
  dataLabel: "資料",
  dataBal: "陽性 20、陰性 20",
  dataRare: "陽性很少（4 對 36）",

  /* 圖 */
  stripAria: "每個人的模型分數與門檻（點圖可移動門檻）",
  stripCap: "每個點＝一個人，橫軸是模型給的陽性機率；點圖或拖動可以移動門檻",
  stripPos: "判為陽性 →",
  stripNeg: "← 判為陰性",
  rowP: "陽性",
  rowPSub: "（真的有）",
  rowN: "陰性",
  rowNSub: "（真的沒有）",
  cmCap: "混淆矩陣（這個門檻的結果）",
  rocAria: "ROC 曲線",
  rocCap: "ROC 曲線：門檻從 1 降到 0 時，點從左下走到右上",
  rocRandom: "亂猜（AUC 0.5）",
  rocX: "偽陽性率 FPR（陰性被誤報的比例）",
  rocY: "真陽性率 TPR（＝召回率）",

  /* 混淆矩陣 */
  cmPredP: "判為陽性",
  cmPredN: "判為陰性",
  cmActP: "實際陽性",
  cmActN: "實際陰性",
  cmTP: "真陽性",
  cmFN: "偽陰性",
  cmFP: "偽陽性",
  cmTN: "真陰性",
  cmTPSub: "TP",
  cmFNSub: "FN（漏掉）",
  cmFPSub: "FP（誤報）",
  cmTNSub: "TN",

  /* 狀態列 */
  statusRare: "全部判為陰性也有準確率 {acc}%，但 4 個陽性一個都沒抓到（召回率 0%）：陽性很少時，準確率會騙人",
  statusPerfect: "這個門檻剛好把兩類完全分開：沒有漏掉、也沒有誤報",
  statusLow: "門檻很低：陽性抓到 {tp} / {np}，但陰性也有 {fp} / {nn} 被誤報",
  statusHigh: "門檻很高：幾乎不誤報（{fp} / {nn}），但漏掉 {fn} / {np} 個陽性",
  statusInfo: "門檻 {thr}：抓到 {tp} / {np} 個陽性，誤報 {fp} / {nn} 個陰性。拖動門檻，看 ROC 曲線上的點怎麼移動",

  /* 算式 */
  pcNone: "—（沒有判為陽性的）",
  calcAccK: "準確率",
  calcAcc: `(TP + TN) ÷ 全部 = ({tp} + {tn}) ÷ {n} = <span class="hl">{v}</span>`,
  calcPrecK: "精確率",
  calcPrec: `TP ÷ (TP + FP) = {tp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">判為陽性的人裡，真的是陽性的比例</span>`,
  calcRecK: "召回率",
  calcRec: `TP ÷ (TP + FN) = {tp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">真正的陽性裡，被抓到的比例（＝ROC 的縱軸）</span>`,
  calcF1K: "F 值",
  calcF1None: "—（精確率或召回率是 0）",
  calcF1: `2 × 精確率 × 召回率 ÷ (精確率 + 召回率) = <span class="hl">{v}%</span>　<span class="k">兩者的調和平均</span>`,
  calcFprK: "偽陽性率",
  calcFpr: `FP ÷ (FP + TN) = {fp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">（ROC 的橫軸）</span>`,
  calcAucK: "AUC",
  calcAuc: `曲線下的面積 = <span class="hl">{auc}</span>；驗算：陽性 × 陰性共 {pairs} 對，陽性分數比較高的有 {win} 對 → {win} ÷ {pairs} = {ratio}`,

  note: `示範資料是假想的檢查結果（固定亂數）。陽性用藍色圓點、陰性用紅色方塊，判錯的點畫成半透明加黑框。門檻往左移，ROC 曲線上的點往右上走（抓得多也誤報得多）；不管門檻放哪，點都在同一條曲線上，所以曲線屬於模型、不屬於某個門檻。AUC（曲線下面積）恰好等於「隨機抽一個陽性和一個陰性，模型給陽性的分數比較高」的機率。陽性很少的資料，準確率容易很高卻毫無用處，要一起看<span class="term">精確率</span>（<span class="en">precision</span>）與<span class="term">召回率</span>（<span class="en">recall</span>）；兩者的調和平均叫 <span class="term">F 值</span>（<span class="en">F-measure</span>）。這四格的表叫<span class="term">混淆矩陣</span>（<span class="en">confusion matrix</span>）。這些指標都要用<b>沒參與訓練的資料</b>計算（<span class="term">保留驗證</span>（<span class="en">holdout validation</span>）、<span class="term">交叉驗證</span>（<span class="en">cross-validation</span>）），才代表<a href="#overfit">泛化能力</a>。輸出機率的模型怎麼訓練見<a href="#loss">損失函數</a>的交叉熵。`,
});
