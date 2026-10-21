/* k-means 分群（concepts/kmeans.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、calc*、note 這類會以 HTML 顯示的項目
   - 資料點的報讀文字由 pointAria + pointGroup + pointChanged 依序接起來（各自是一個用「，」開頭的項目） */
Atlas.strings("zh", "kmeans", {
  lead: `<p>手上只有資料、沒有正解標籤，想把相似的資料自動分成幾群，這叫<span class="term">分群</span>（<span class="en">clustering</span>），例如依購買紀錄把顧客分群。<span class="term">k-means 分群</span>（<span class="en">k-means clustering</span>）先決定要分 k 群、隨便放 k 個中心，然後只重複兩件簡單的事，直到不再變化。</p>`,
  closing: `k-means 分群只做兩件事：每個點歸給最近的中心、每個中心搬到組員的平均位置；每一步都讓組內距離平方和變小，所以一定會停下來。但停下來的地方取決於初始中心（可能卡在比較差的結果），而且 k 要由人決定，可以用不同 k 的結果比較來輔助判斷。`,

  /* 控制項 */
  kLabel: "群數 k",
  kFmt: "{k} 群",
  initLabel: "初始中心",
  initRand: "隨機挑 k 個點",
  initBad: "故意放壞的位置（k = 3）",
  btnSeed: "換一組隨機初始",
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnEnd: "跑到收斂",
  btnReset: "回到第 0 步",

  /* 圖 */
  mainAria: "資料點、各群中心與分組",
  histAria: "每一步的組內距離平方和",
  elbowAria: "不同群數收斂後的組內距離平方和",
  mainCap: "◆＝中心（第 0 步可以拖動）；點一個資料點，看它的計算",
  histCap: "每一步之後的 SSE（組內距離平方和；○ 分組後、● 移動中心後）",
  elbowCap: "群數 k 與收斂後的 SSE（每個 k 試 20 組初始取最好）",
  histX: "步",
  elbowX: "群數 k",
  elbowHint: "← 從這裡開始下降變緩",
  pointAria: "資料點 ({x}, {y})",
  pointGroup: "，第 {g} 群",
  pointChanged: "，這一步換了組",
  centerAria: "第 {j} 個中心",
  centerAriaDrag: "第 {j} 個中心，可拖動",

  /* 狀態列 */
  statusInitBad: "第 {n} 步：初始中心故意擠在左下那一團（1、2 號）。按「下一步」開始",
  statusInitDrag: "第 {n} 步：用你拖動的位置當初始中心。按「下一步」開始",
  statusInitRand: "第 {n} 步：隨機挑了 {k} 個資料點當初始中心。按「下一步」開始",
  statusAssign1: "第 {n} 步：分組：每個點歸給最近的中心",
  statusAssign: "第 {n} 步：分組：有 {nch} 個點換了組（粗黑框）",
  statusUpdate: "第 {n} 步：移動中心：每個中心搬到組員的平均位置，最多移動 {move}",
  statusStuck: "第 {n} 步：收斂了（沒有點換組），但 SSE = {sse}，比 k = {k} 時能達到的 {best} 大 {ratio} 倍：卡在比較差的結果",
  statusDone: "第 {n} 步：收斂：沒有點換組，中心不再移動。SSE = {sse}",

  /* 算式 */
  calcNextK: "下一步",
  calcNext: "選中的點 {p}：算出它到每個中心的距離，歸給最近的那一個",
  calcCenterK: "中心 {j} 的新位置",
  calcCenter: `選中的點所在的第 {j} 群有 {n} 個點，平均 = ({sx} ÷ {n}, {sy} ÷ {n}) = <span class="hl">({cx}, {cy})</span>`,
  calcSseK: "SSE",
  calcSseUpdate: "每個點到自己中心的距離² 加總 = {sse}（移動前 {prev}）",
  calcSseAssign: "每個點到自己中心的距離² 加總 = {sse}（分組前 {prev}）",
  calcSse: "每個點到自己中心的距離² 加總 = {sse}",
  calcDistK: "點 {p} 到各中心的距離²",
  calcDistItem: "中心 {j}：{v}",
  calcDistTo: "→ 歸給中心 {b}",

  note: `中心的位置就是組員的平均（mean），名稱裡的「means」由此而來；k 是要分幾群。SSE（組內距離平方和）在「分組」和「移動中心」兩步都只會變小，所以一定會停，但停的地方不一定是最好的：試「故意放壞的位置」，兩個中心擠在同一團，最後把右邊和上面兩團併成一群。實務上會換幾組初始值多跑幾次，或用 k-means++（讓初始中心彼此離遠一點）。群數越多 SSE 一定越小（每個點自成一群時是 0），所以不能只看 SSE 選 k；右下圖下降突然變緩的地方（像手肘）是常用的參考，叫<span class="term">手肘法</span>（<span class="en">elbow method</span>）。不必事先決定群數、由近到遠一步步合併成<span class="term">樹狀圖</span>（<span class="en">dendrogram</span>）的方法有<span class="term">華德法</span>（<span class="en">Ward's method</span>）等。另一種常見的<span class="term">非監督式學習</span>（<span class="en">unsupervised learning</span>）是降維，見<a href="#pca">主成分分析</a>。`,
});
