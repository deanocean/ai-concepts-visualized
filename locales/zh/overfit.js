/* 過擬合與正則化（concepts/overfitting.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note*、calc* 這類會以 HTML 顯示的項目
   - note 由 noteNone／noteL2／noteL1 其中一段加上 noteCommon 組成（各自是完整的句子） */
Atlas.strings("zh", "overfit", {
  lead: `<p>模型的目的是預測<b>沒看過的資料</b>，但訓練時只看得到手上的資料。模型太靈活，就會連資料裡的雜訊一起記住：訓練資料幾乎全對，遇到新資料卻錯很多。這叫<span class="term">過擬合</span>（<span class="en">overfitting</span>）。<span class="term">正則化</span>（<span class="en">regularization</span>）在損失上加一項「權重太大就扣分」，逼模型不要為了貼合每個點而把權重拉得很大。</p>`,
  closing: `訓練誤差一直下降，不代表模型變好；要看模型沒學過的驗證資料，誤差最小的地方才是<span class="term">泛化能力</span>（<span class="en">generalization performance</span>）最好的地方。L2 正則化把所有權重一起縮小、曲線變平滑；L1 正則化會把不重要的權重直接壓成 0，等於自動挑出要用的特徵。`,

  /* 控制項 */
  degLabel: "多項式的次數（模型的靈活度）",
  degFmt: "{n} 次",
  regLabel: "正則化",
  regNone: "無",
  regL2: "L2（嶺迴歸）",
  regL1: "L1（套索迴歸）",
  lamLabel: "正則化的強度 λ",
  btnTruthShow: "顯示真正的規律",
  btnTruthHide: "隱藏真正的規律",
  presetLabel: "快速選擇",
  preSimple: "太簡單（1 次）",
  preRight: "剛好（3 次）",
  preMemo: "背答案（9 次）",
  preL2: "9 次＋L2",
  preL1: "9 次＋L1",
  coefLabel: "每一項的係數（權重）",
  curveLabel: "次數 vs. 誤差（目前的正則化設定）",

  /* 主圖 */
  svgAria: "訓練資料、驗證資料與擬合曲線",
  lgTrain: "訓練資料（10 點，拿來學）",
  lgValid: "驗證資料（20 點，訓練時看不到）",
  lgPred: "模型的預測",
  lgTruth: "真正的規律 y = sin 2πx",

  /* 狀態列 */
  sOver: "過擬合：訓練誤差 {tr} 幾乎貼合，驗證誤差卻有 {va}",
  sTooStrong: "正則化太強：訓練誤差 {tr}、驗證誤差 {va}，連訓練資料都沒貼好",
  sTooSimple: "太簡單：訓練誤差 {tr}、驗證誤差 {va}，連訓練資料都沒貼好",
  sGood: "訓練誤差 {tr}、驗證誤差 {va}：對沒看過的資料也預測得不錯",

  /* 算式 */
  calcModelK: "模型",
  calcModel: "ŷ = w₀ + w₁u + w₂u² + … + w<sub>{d}</sub>u<sup>{d}</sup>（u = 2x − 1，把 x 換到 −1〜1）",
  calcLossK: "要最小化的損失",
  calcLossNone: "訓練資料的均方誤差 = {v}",
  calcLossReg: "均方誤差 + λ × {pen}",
  calcNoPenW0: "（w₀ 不懲罰）",
  calcValidK: "驗證資料的均方誤差",
  calcValidNote: "（不參與訓練，只拿來檢查）",
  calcNzK: "不是 0 的係數",
  calcNz: "{nz} / {d} 個",

  /* 係數圖、誤差曲線 */
  coefAria: "各項係數的大小",
  curveAria: "次數與訓練、驗證誤差",
  axisDeg: "次數",
  bestValid: "驗證最小",
  lgTrainErr: "訓練誤差",
  lgValidErr: "驗證誤差（對數刻度）",

  noteNone: `次數越高，曲線能彎的地方越多：9 次多項式有 10 個係數，剛好能穿過全部 10 個訓練點，訓練誤差變成 0，但為了穿過帶雜訊的點，曲線在點與點之間劇烈擺動，係數也大到好幾百。能用簡單的模型解釋，就不要用複雜的，這個原則叫<span class="term">奧坎剃刀</span>（<span class="en">Occam's razor</span>）。`,
  noteL2: `<span class="term">L2 正則化</span>（<span class="en">L2 regularization</span>）的懲罰是權重的平方和。大的權重扣分特別重，所以每個係數都被往 0 拉、但很少剛好變成 0，曲線變平滑。用在線性迴歸上叫<span class="term">嶺迴歸</span>（<span class="en">ridge regression</span>）。λ 太大時連必要的彎曲也被壓掉，又變回太簡單。`,
  noteL1: `<span class="term">L1 正則化</span>（<span class="en">L1 regularization</span>）的懲罰是權重的絕對值和。小的權重被壓成剛好 0（長條變成灰色的 0），剩下的就是模型認為重要的項，所以適合想減少特徵數量時使用。用在線性迴歸上叫<span class="term">套索迴歸</span>（<span class="en">lasso regression</span>）。直接以「不是 0 的權重個數」當懲罰的叫 <span class="term">L0 正則化</span>（<span class="en">L0 regularization</span>），最直接但很難求解，L1 常被當成它可以計算的替代。`,
  noteCommon: `　資料由 y = sin 2πx 加上隨機雜訊產生（固定亂數，每次相同）；L2 的解用公式直接算出，L1 用座標下降法反覆逼近。神經網路常用的另一種正則化見 <a href="#dropout">Dropout</a>。訓練神經網路時常一邊訓練一邊看驗證誤差，它開始上升就停止訓練，這叫<span class="term">提前停止</span>（<span class="en">early stopping</span>）。另外，模型大到能把訓練資料完全記住之後，再繼續加大，驗證誤差有時反而會再次下降，叫<span class="term">雙重下降</span>（<span class="en">double descent</span>），在大型神經網路上觀察到（這裡的多項式最多 9 次，看不到這一段）。分類模型的評估指標見 <a href="#roc">ROC 曲線與 AUC</a>。`,
});
