/* 自注意力（concepts/self-attention.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、heatCorner、note、s*Intro、s1Note、s1VecK 這類會以 HTML 顯示的項目
   - dims：4 個維度的名稱（以半形空白分隔）；dimSep：在說明文裡把它們連起來的分隔符號 */
Atlas.strings("zh", "selfatt", {
  lead: `<p>讀到「it」時，要知道它指的是誰，就得回頭看句中其他詞。<span class="term">自注意力</span>（<span class="en">self-attention</span>）讓每個詞自己算出「該多關注句中的每個詞」，再按這個比例，把那些詞的資訊混進自己的向量。</p>`,
  closing: `每個詞都同時對句中所有詞算一次注意力，不必像 RNN 一格格往後傳，所以距離再遠也能直接連上，而且可以平行計算；把這個機制層層疊起來，就是 Transformer 的核心。`,

  dims: "生物 代名詞 動作 狀態",
  dimSep: "／",

  /* 上半部 */
  toksAria: "選擇查詢的詞",
  toksLabel: "點任一個詞，看它把注意力分給句中哪些詞",
  diagAria: "「{w}」對各詞的注意力",
  statusFocus: "「{w}」最關注「{top}」（{pct}）",
  statusEven: "「{w}」的注意力分得很平均（最高是「{top}」的 {pct}）",
  heatLabel: "整句的注意力權重表（每一列＝該詞分出去的注意力，總和 100%；點一列可切換）",
  heatCorner: "看的詞 ↓ ／ 被看的詞 →",
  note: `例子中的向量和矩陣是為了示範而設計的小數字；真實模型有數百維，數字全是訓練學到的。這裡沒有放入詞的順序（打亂詞序結果不變），要靠 <a href="#posenc">位置編碼</a> 補上；同時用好幾組權重看不同關係，見 <a href="#mha">多頭注意力</a>；翻譯時從譯文回頭看原文，見 <a href="#seq2seq">Source-Target 注意力</a>；這些零件怎麼組成完整的模型，見 <a href="#transformer">Transformer</a>。`,

  /* 計算步驟 */
  btnShowSteps: "展開計算步驟（Q・K → softmax → 加權 V）",
  btnHideSteps: "收起計算步驟",
  btnNext: "下一步 ▶",
  step1: "① Q・K・V",
  step2: "② 分數 q·k",
  step4: "④ 加權 V",

  s1Intro: `每個詞先是一個 4 維向量，每一維代表一種性質（{dims}）。它分別乘上三個矩陣，變成三個角色：<span class="term">查詢</span>（<span class="en">query</span>）q＝「我在找什麼」、<span class="term">鍵</span>（<span class="en">key</span>）k＝「我是什麼，給別人比對」、<span class="term">值</span>（<span class="en">value</span>）v＝「被選中時我提供的內容」。`,
  s1VecK: "「{w}」的向量",
  queryK: "查詢",
  keyK: "鍵",
  valueK: "值",
  s1Note: `W_Q 的意思：代名詞、動作、狀態類的詞，會去「找生物」；生物類的詞，會去「找動作和狀態」。為了好讀，這個例子的 W_K、W_V 設成不改變向量（k = v = x），真實模型三個矩陣都要學。`,
  thWord: "詞",

  s2Intro: `拿「{w}」的 q 和每個詞的 k 做內積（對應維度相乘再加總）：性質越對得上，分數越高。再除以 √d（d＝向量維數 {d}，√{d} = {sq}），避免維數一多分數就變得太大。`,
  thSeen: "被看的詞",
  thQK: "q·k（q = {q}）",

  s3Intro: `分數還不能直接當比例用（可能是負數，總和也不是 1）。<span class="term">softmax 函數</span>（<span class="en">softmax function</span>）先對每個分數取 e 的次方（變成正數，而且分數高的被放大），再除以總和，得到加起來剛好 100% 的注意力權重。`,
  thScore: "分數 s",
  thDiv: "e^s ÷ 總和",
  thWeight: "權重",
  total: "總和",

  s4Intro: `最後把每個詞的 v 乘上權重再加總，得到「{w}」的新向量：關注得越多的詞，貢獻越大。`,
  thWv: "權重×v {dim}",
  sum: "加總",
  cmpBefore: "原本的向量",
  cmpAfter: "注意力之後",
  s4Mixed: "「{w}」在「{dim}」這一維從 {a} 變成 {b}：它的向量混入了「{top}」的資訊。",
  s4Even: "「{w}」的新向量是各詞內容的平均混合，沒有特別偏向哪個詞。",
});
