/* 詞嵌入（concepts/w2v.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、nbLabel、note 這類會以 HTML 顯示的項目
   - corpus：訓練語料，句子以「/」分隔、詞以半形空白分隔（改了語料，訓練結果會跟著變，要重新確認 closing 的說法）
   - keyWords：程式要用到的 9 個詞，依序是 國王、王后、男人、女人、貓、狗、魚、肉、水（都必須出現在 corpus 裡）
   - groupP / groupA / groupF / groupO：平面圖的圖例（人／動物／食物・飲料／其他） */
Atlas.strings("zh", "w2v", {
  lead: `<p>電腦處理文字前，要先把詞變成數字。最直接的做法是<span class="term">one-hot 向量</span>（<span class="en">one-hot vector</span>）：詞彙有幾個就開幾維，該詞那一格是 1、其他是 0。但這樣任兩個詞都一樣「不像」，看不出「國王」和「王后」比和「水」接近。<span class="term">詞嵌入</span>（<span class="en">word embedding</span>）改用幾維的實數向量表示一個詞，稱為<span class="term">分散式表示</span>（<span class="en">distributed representation</span>）；<span class="term">word2vec</span> 的想法是：<b>出現在相似上下文的詞，意思也相近</b>，所以讓網路練習「由一個詞猜周圍的詞」，向量就會自己長出意義。</p>`,
  closing: `訓練只是在練習「猜周圍的詞」，但上下文相似的詞會被推到相近的位置：用 Skip-gram 訓練，每一組初始值都把國王與王后、貓與狗、男人與女人、魚與肉配成彼此最相近。連「國王 − 男人 + 女人」這種向量加減都會落在王后附近（兩種訓練方式都是），因為「男→女」的差別在不同的詞上被學成差不多的方向。CBOW 在這麼小的語料上比較不穩定，有些詞最相近的對象會跑掉。`,

  corpus: "他 是 國王 / 她 是 王后 / 他 是 男人 / 她 是 女人 / 國王 住 城堡 / 王后 住 城堡 / 男人 住 村莊 / 女人 住 村莊 / 國王 戴 王冠 / 王后 戴 王冠 / 男人 下 田 / 女人 下 田 / 貓 吃 魚 / 狗 吃 肉 / 貓 喝 水 / 狗 喝 水 / 貓 在 家 睡覺 / 狗 在 家 睡覺 / 我 養 貓 / 我 養 狗",
  keyWords: "國王 王后 男人 女人 貓 狗 魚 肉 水",
  listSep: "、",
  quoted: "「{w}」",

  /* 控制項 */
  modeLabel: "訓練方式",
  modeSg: "Skip-gram（中間 → 周圍）",
  modeCbow: "CBOW（周圍 → 中間）",
  btnSeed: "換一組初始值",
  btnStart: "▶ 開始訓練",
  btnContinue: "▶ 繼續訓練",
  btnPause: "❚❚ 暫停",
  btnDone: "✓ 訓練完成",
  btnStep: "前進 1 個 epoch",
  btnFF: "⏭ 快轉到結束",

  /* 語料與題目 */
  corpusLabel: "語料（{s} 句，{v} 個詞）：點任一個詞，看它在訓練時變成什麼題目（窗口＝前後各 2 個詞）",
  sampleSgK: "Skip-gram 的題目",
  sampleSg: "輸入「{w}」的向量 → 要猜出窗口內的 {n} 個詞（每個詞各算一次損失）",
  sampleCbowK: "CBOW 的題目",
  sampleCbow: "把窗口內「{ctx}」的向量平均 → 要猜出中間的「{w}」",
  modelK: "目前的模型",
  randomNote: "（亂猜是 1/{v} ≈ {p}%）",

  /* 圖 */
  mapAria: "詞向量投影到平面",
  mapCap: "8 維的詞向量投影到平面（點一個詞看它最相近的詞）",
  groupP: "人",
  groupA: "動物",
  groupF: "食物・飲料",
  groupO: "其他",
  lossAria: "損失",
  lossCap: "平均損失（猜周圍詞的 −ln 機率）",
  epochAxis: "epoch",

  /* 狀態列 */
  statusStart: "還沒訓練：詞向量是隨機的小數字，平面上的位置沒有意義，平均損失 {loss}（約等於亂猜的 ln {v} ≈ {ln}）。按「▶ 開始訓練」跑 {ep} 個 epoch（約 3 秒）",
  statusEp: "第 {ep} 個 epoch：平均損失 {loss}；{found}",
  statusEpRunning: "第 {ep} 個 epoch（訓練中）：平均損失 {loss}；{found}",
  statusPairs: "互為最相近的詞：{list}",
  statusNoPairs: "還沒有哪一對詞互為最相近",
  statusMiss: "；{list}還沒配成一對（{why}）",
  whyCbow: "CBOW 在這麼小的語料上比較不穩定",
  whySeed: "可以換一組初始值比較",

  /* 最相近的詞與類推 */
  nbLabel: `和「{w}」最相近的詞（<span class="term">餘弦相似度</span>（<span class="en">cosine similarity</span>）：1＝方向完全相同、0＝無關）`,
  nbVecK: "向量",
  nbTopK: "最相近",
  oneHot: "若用 one-hot 向量：「{w}」和任何其他詞的餘弦相似度都是 0",
  anaLabel: "向量的加減（類推）",
  anaOpt: "{a} − {b} + {c}",
  anaRowK: "「{a}」−「{b}」+「{c}」",
  anaTopK: "最接近的詞（不算題目裡的三個）",
  anaJudgeK: "判讀",
  anaUntrained: "還沒訓練，結果沒有意義",
  anaOk: "答對：最接近「{w}」",
  anaNot: "目前最接近的是「{got}」，還不是「{w}」",

  note: `這一頁的語料、維數（8 維）都刻意做得很小；真實的 word2vec 用數十億詞的語料、100〜300 維，而且為了省計算，不對全部詞算 softmax，改用<span class="term">負採樣</span>（<span class="en">negative sampling</span>）等近似。平面圖是把 8 維投影到前兩個主成分（見 <a href="#pca">主成分分析</a>），距離只是近似，相似度一律用原本的 8 維計算。<span class="term">Skip-gram</span>（<span class="en">skip-gram</span>）在小語料、稀有詞上通常較好；<span class="term">CBOW</span>（<span class="en">continuous bag-of-words</span>）計算較快。fastText 把詞再拆成更小的字元片段來學，沒看過的詞也能組出向量；word2vec 一個詞只有一個向量，ELMo、BERT 則依上下文給同一個詞不同的向量。把整份文件看成「各詞出現次數」的 BoW（<span class="en">bag-of-words</span>）、以及加權的 TF-IDF，都是更早的、不考慮詞義相近的表示法。`,
});
