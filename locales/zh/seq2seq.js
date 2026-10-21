/* Seq2Seq 與 Source-Target 注意力（concepts/seq2seq.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、calc*、note 這類會以 HTML 顯示的項目
   - src：原文的 6 個詞（日文例句，兩種語言共用，以半形空白分隔）；dims：4 個維度的名稱（以半形空白分隔） */
Atlas.strings("zh", "seq2seq", {
  lead: `<p>翻譯要把一整句變成另一整句，長度和語序都可能不同。<span class="term">Seq2Seq</span>（<span class="en">sequence-to-sequence</span>）先用<span class="term">編碼器</span>（<span class="en">encoder</span>）讀完原文，再用<span class="term">解碼器</span>（<span class="en">decoder</span>）一個詞一個詞生成譯文。原始做法只把編碼器最後的狀態交給解碼器，整句話都得擠進這一個固定長度的向量。</p>`,
  closing: `加上 <span class="term">Source-Target 注意力</span>（<span class="en">source-target attention</span>，又稱 <span class="en">encoder-decoder attention</span>）後，解碼器每生成一個詞，都用自己的狀態當查詢，到原文每個位置的狀態（鍵和值）去找需要的資訊：「I」看「私」、「cats」看「猫」，語序不同也對得上，前面的詞也不會被擠掉。`,

  src: "私 は 猫 が 好き です",
  dims: "人 動物 喜好 文法",

  /* 控制項 */
  modeLabel: "解碼器拿到什麼",
  modePlain: "原始 Seq2Seq（只有最後的狀態）",
  modeAtt: "加上 Source-Target 注意力",
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnJump: "跳到解碼",
  btnReset: "重設",

  /* 圖 */
  svgAria: "編碼器與解碼器",
  encName: "編碼器",
  encSub: "讀原文",
  decName: "解碼器",
  decSub: "寫譯文",
  onlyVec1: "唯一交出去的",
  onlyVec2: "向量 = h₆",
  decGotH6: "拿到的 h₆",
  legend: "色條的 4 格：",

  /* 狀態列與算式 */
  statusStart: "① 編碼：按「下一步」把原文一個詞一個詞讀進編碼器",
  calcStart: "每個詞是一個 4 維向量，4 格依序代表「人／動物／喜好／文法（助詞等）」。例：私 = [1, 0, 0, 0]、猫 = [0, 1, 0, 0]、好き = [0, 0, 1, 0]、は・が・です = [0, 0, 0, 1]。",
  statusEnc: "① 編碼：讀入第 {n} 個詞「{w}」",
  calcEncRule: "新的狀態 = 這個詞 + 0.5 × 上一步狀態",
  calcEncShareK: "展開來看，最後的狀態裡各詞的份量",
  statusDecAtt: "② 解碼：要說「{tgt}」，需要「{dim}」的資訊 → 注意力集中在「{src}」（{pct}），拿到 {got}",
  statusDecPlainLow: "② 解碼：要說「{tgt}」，需要「{dim}」的資訊，但 h₆ 裡只剩 {got}：前面的詞幾乎被擠掉了",
  statusDecPlainHigh: "② 解碼：要說「{tgt}」，需要「{dim}」的資訊，但 h₆ 裡只剩 {got}：後面讀到的詞還留得比較多",
  calcQK: "查詢＝解碼器這一步的狀態",
  calcKVK: "鍵・值＝編碼器各步的 h",
  calcScoreK: "分數 s·h ÷ √{d}",
  calcSoftK: "softmax（e^分數 ÷ {tot}）",
  calcCtxK: "context = Σ 權重 × h",
  calcSameK: "每一步拿到的都是同一個向量",
  calcCellK: "它的「{dim}」那一格",

  /* 表格 */
  thSay: "要說「{tgt}」時",
  thNeed: "{dim}（需要）",
  rowPlain: "原始：只拿到 h₆",
  rowAtt: "注意力：context",
  cmpLabel: "兩種做法拿到的資訊（橘框＝這個詞最需要的那一格）",
  heatCorner: "譯文 ↓ ／ 原文 →",
  heatLabel: "對齊表：每個譯文詞把注意力分給哪個原文詞（每列總和 100%）",

  note: `為了好讀，編碼器簡化成「這個詞 + 0.5 × 上一步」的線性 RNN，解碼器每一步的狀態 s 用設計好的示範值（真實模型由解碼器 RNN 算出，數字全由訓練學到）。分數用和 <a href="#selfatt">自注意力</a> 相同的內積再除以 √d。差別只在來源：自注意力的<span class="term">查詢</span>（<span class="en">query</span>）、<span class="term">鍵</span>（<span class="en">key</span>）、<span class="term">值</span>（<span class="en">value</span>）都來自同一句；Source-Target 注意力的查詢來自譯文（解碼器），鍵和值來自原文（編碼器）。訓練解碼器時，下一步的輸入用正解而不是自己的預測，稱為<a href="#teacher">教師強制</a>（<span class="en">teacher forcing</span>）。整個翻譯模型完全改用注意力搭建的，見 <a href="#transformer">Transformer</a>。`,
});
