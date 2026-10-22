/* 自編碼器（concepts/ae.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、cap*、calc*、dec* 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "ae", {
  lead: `<p>一張 7×7 的圖有 49 個數字，但下面這 49 張長方形其實只由「寬」和「高」兩件事決定。<span class="term">自編碼器</span>（<span class="en">autoencoder</span>）把圖壓成少數幾個數、再從這幾個數還原回原圖；訓練目標只是「輸出＝輸入」，不需要正解標籤。</p>`,
  closing: `中間層比輸入小，網路只好把 49 個數字濃縮成最關鍵的幾個：這組圖只靠寬和高決定，所以中間層 2 個就能完全還原，1 個就不夠。不用標籤就學出精簡表示，可以拿來降維，也曾用來替深層網路做預訓練。`,

  /* 控制項 */
  kLabel: "中間層的數字個數",
  kOpt: "{n} 個",
  btnRun: "▶ 開始訓練",
  btnResume: "▶ 繼續訓練",
  btnPause: "❚❚ 暫停",
  btnDone: "✓ 訓練完成",
  btnFf: "⏭ 快轉到結束",
  btnSeed: "換一組初始值",
  progLabel: "查看第幾個 epoch 的狀態",

  /* 圖（螢幕報讀用） */
  netAria: "網路結構：49 個輸入壓縮到中間層，再還原成 49 個輸出",
  inAria: "選中的輸入圖",
  outAria: "選中的圖還原的結果",
  galAria: "49 張圖的還原結果，橫向寬 1 到 7、縱向高 1 到 7",
  latAria: "每張圖在中間層的數值",
  decAria: "在中間層選的位置解碼出來的圖",
  lossAria: "每個 epoch 的平均還原誤差",

  /* 圖說 */
  capPair: "選中的圖：輸入 → 中間層 → 還原（紅框＝四捨五入後錯的格子）",
  capGal: "49 張圖的還原結果（點一張來選；紅框＝有錯的格子）",
  capLoss: "平均還原誤差（對數刻度；直線＝目前查看的 epoch）",
  netIn: "輸入 49",
  netMid: "中間層 {k}",
  netOut: "輸出 49",
  netEnc: "編碼器（壓縮）",
  netDec: "解碼器（還原）",
  midEnc: "編碼器 →",
  midDec: "→ 解碼器",
  galW: "寬 {w}",
  galH: "高{h}",
  lat4Cap: "中間層的數值",
  lat4Line1: "中間層有 4 個數，畫不成平面。",
  lat4Line2: "改選「1 個」或「2 個」來看分布。",
  lat2Cap: "每張圖在中間層的位置（2 個數＝平面上一點）；點空白處，把那一點交給解碼器",
  lat1Cap: "每張圖在中間層的值（1 個數＝數線上一點）；點數線上任一處，把那個值交給解碼器",
  decPick: "你點的位置 ({z}) → 解碼器 → 左圖<br>最像的訓練圖：寬 {w}・高 {h}，{diff}",
  decSel: "選中的圖的位置 ({z}) → 解碼器 → 左圖<br>最像的訓練圖：寬 {w}・高 {h}，{diff}",
  decSame: "四捨五入後完全一樣",
  decDiff: "差 {d} 格",
  lossAxis: "epoch",

  /* 狀態列 */
  statusHead: "第 {e} 個 epoch：平均還原誤差 {loss}；49 張中 {n} 張完全還原（四捨五入後每格都對）",
  statusHeadRun: "第 {e} 個 epoch（訓練中）：平均還原誤差 {loss}；49 張中 {n} 張完全還原（四捨五入後每格都對）",
  tailStartRun: "。剛開始，輸出和輸入幾乎無關",
  tailStart: "。還沒訓練，輸出和輸入幾乎無關。按「▶ 開始訓練」開始（約 5 秒）",
  tailPerfect: "。中間層只有 {k} 個數，卻能還原全部 49 張",
  tailOne: "。只有 1 個數，裝不下「寬」和「高」兩件事",

  /* 算式 */
  calcInK: "輸入",
  calcIn: "寬 {w} × 高 {h} 的長方形：49 格中 {on} 格是 1，其餘是 0",
  calcEncK: "編碼器",
  calcEnc: `49 → 32 → <span class="hl">中間層 ({z})</span>`,
  calcDecK: "解碼器",
  calcDec: `→ 32 → 49 個輸出（每格 0〜1）；四捨五入後和輸入不同的格子：<span class="{cls}">{n} 格</span>`,
  calcLossK: "還原誤差",
  calcLoss: `−Σ [ x·ln y + (1−x)·ln(1−y) ]（49 格加總）= <span class="hl">{v}</span>`,
  calcWorstK: "誤差最大的一格",
  calcWorst: "從上數第 {r} 格、從左數第 {c} 格：x = {x}、y = {y} → −ln {arg} = {v}",

  note: `負責壓縮的前半叫<span class="term">編碼器</span>（<span class="en">encoder</span>），負責還原的後半叫<span class="term">解碼器</span>（<span class="en">decoder</span>）。誤差用的是<a href="#loss">交叉熵</a>：輸入是 1 的格子，輸出越接近 1 誤差越小；輸入是 0 的格子則相反。這裡的編碼器與解碼器各有一層 32 個神經元（ReLU），輸出層用 Sigmoid 把每格夾在 0〜1。中間層的數字就是壓縮後的特徵，可以當成<span class="term">降維</span>（<span class="en">dimensionality reduction</span>）的結果：<a href="#pca">主成分分析</a>只能找直線方向，自編碼器有激活函數，能學出彎曲的對應（若全部改成線性、改用平方誤差，學到的就和主成分分析相當）。2006 年前後，深層網路很難從頭直接訓練，於是先把自編碼器一層一層訓練、疊起來當初始權重（<span class="term">堆疊自編碼器</span>（<span class="en">stacked autoencoder</span>）），再用有標籤的資料微調，這叫<span class="term">預訓練</span>（<span class="en">pre-training</span>）。中間層的值散在哪裡沒有規則（每次訓練、每組初始值都不同），隨便挑一點解碼不一定像樣，所以不適合直接拿來產生新圖，見<a href="#vae">VAE</a>。`,
});
