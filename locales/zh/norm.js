/* 正規化層（concepts/norm.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、desc*、headCh、calc* 這類會以 HTML 顯示的項目
   - 說明列由 descBn／descLn／descIn／descGn 其中一句，加上 descFrame（與 descAfter）組成 */
Atlas.strings("zh", "norm", {
  lead: `<p>一層的輸出裡，有的通道數值都在 50 左右，有的在 0 附近；訓練中前面的權重一改，後面每層收到的數值範圍也跟著跑。數值尺度不一，學習率很難同時適合每一處，學習就慢又不穩。<span class="term">正規化層</span>（<span class="en">normalization layer</span>）把一組數減去平均、除以標準差，拉回「平均 0、標準差 1」。各種方法的差別，只在<b>哪些數放在一起算平均</b>。</p>`,
  closing: `批次正規化每個通道跨整批資料算，所以結果會隨同一批的其他資料而變，批次太小就不穩；層正規化每筆資料自己算，不受批次影響，Transformer 和 RNN 都用它；實例正規化每筆資料的每個通道各自算；組正規化介於兩者之間，把通道分組來算，批次很小時也穩定。`,

  /* 控制項 */
  methodLabel: "正規化方法",
  mBn: "批次正規化",
  mLn: "層正規化",
  mIn: "實例正規化",
  mGn: "組正規化",
  viewLabel: "顯示",
  viewBefore: "正規化前",
  viewAfter: "正規化後",
  btnSwap: "換掉資料 2〜4（資料 1 不變）",
  btnRestore: "還原資料",

  /* 說明列 */
  descBn: `<span class="term">批次正規化</span>（<span class="en">batch normalization</span>）：同一個通道、跨整批資料的所有位置（N × H × W）一起算。`,
  descLn: `<span class="term">層正規化</span>（<span class="en">layer normalization</span>）：同一筆資料、所有通道的所有位置（C × H × W）一起算。`,
  descIn: `<span class="term">實例正規化</span>（<span class="en">instance normalization</span>）：同一筆資料、同一個通道（H × W）自己算。`,
  descGn: `<span class="term">組正規化</span>（<span class="en">group normalization</span>）：同一筆資料、同一組通道（這裡 2 個一組）一起算。`,
  descFrame: "橘框＝和選中的數一起算平均的那一組。",
  descAfter: "格子裡顯示到小數一位，精確值見下方算式。",

  /* 表格 */
  headCh: "<b>ch{c}</b><br>平均約 {m}",
  groupFmt: "第 {g} 組",
  rowFmt: "資料 {n}",
  cellAria: "資料 {n} ch{c} 第 {k} 格",

  /* 狀態列 */
  sChanged: "資料 1 完全沒變，但它正規化後的值變了（選中的那格 {a} → {b}）：同一批的其他資料會影響它",
  sSame: "資料 1 正規化後的值完全沒變（選中的那格仍是 {y}）：只用它自己的數字計算",
  sSwapped: "已換掉資料 2〜4；先點資料 1 的某一格再換，就能比較前後",
  sGroup: "這一組 {n} 個數：平均 {mu}、標準差 {sd}；正規化後平均 0、標準差 1",

  /* 算式 */
  calcSelK: "選中",
  calcSel: "資料 {n}、ch{c} 的 x = <b>{x}</b>",
  calcCountK: "同組共",
  calcCount: "{n} 個數",
  calcMeanK: "平均",
  calcVarK: "變異數",
  calcVar: "平均((x − μ)²)",
  calcSdK: "標準差",
  calcNormK: "正規化",
  calcLastK: "最後",
  calcLastNote: "（γ、β 是可學習的參數，這裡 γ = 1、β = 0）",

  note: `ε 是很小的數（這裡 0.00001），避免除以 0。正規化後再乘 γ、加 β，讓模型需要時能把尺度調回來，所以不會失去表現能力。批次正規化在推論時通常一次只來一筆資料，沒有「同一批」可算，所以改用訓練時以<span class="term">移動平均</span>（<span class="en">moving average</span>）累積的平均與變異數。原始論文說它解決的是<span class="term">內部共變量偏移</span>（<span class="en">internal covariate shift</span>）：前面的層一更新，後面的層收到的數值分布就跟著變。注意「正規化」（normalization）和<a href="#overfit">正則化</a>（regularization）是不同的東西，有些中文資料把 regularization 也叫「正規化」，閱讀時要留意；JDLA 的大綱也把這一節從「正則化層」改名為「正規化層」。實例正規化常用在<span class="term">風格轉換</span>（<span class="en">style transfer</span>），組正規化常用在批次只能放幾張大圖的<span class="term">物體偵測</span>（<span class="en">object detection</span>）。數字是示範用的整數，每張特徵圖只有 2×2。`,
});
