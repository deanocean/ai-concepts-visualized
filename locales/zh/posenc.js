/* 位置編碼（concepts/posenc.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、calc*、note* 這類會以 HTML 顯示的項目
   - words：三個詞，依序對應程式裡的 dog、bite、man（以半形空白分隔）；sentA、sentB：兩句的整句寫法
   - 注意：中文的「列」是橫排（row）——熱圖裡一列＝一個位置 */
Atlas.strings("zh", "posenc", {
  lead: `<p>自注意力一次看整句、每個詞直接互相比較，不像 RNN 按順序一格格讀，所以「狗咬人」和「人咬狗」在它眼中是同一袋詞。<span class="term">位置編碼</span>（<span class="en">positional encoding</span>）把「第幾個」也變成一組數字，加到詞向量上，模型才分得出順序。</p>`,
  closing: `Transformer 用不同頻率的 sin・cos 當位置編號：前面的維度變得快、後面的維度變得慢，像時鐘的秒針、分針、時針，組合起來每個位置都獨一無二；而且兩個位置的相似度只由它們隔多遠決定，相鄰的位置最像。`,

  words: "狗 咬 人",
  sentA: "狗咬人",
  sentB: "人咬狗",

  /* 控制項 */
  viewLabel: "看哪一部分",
  viewWhy: "① 為什麼需要",
  viewWave: "② sin・cos 長什麼樣",
  peLabel: "位置資訊",
  peOff: "不加",
  peOn: "加上位置編碼",
  wordLabel: "比較哪個詞",
  dLabel: "向量維數 d",
  posLabel: "位置 pos",

  /* ① 為什麼需要 */
  whyAria: "兩句話中同一個詞的注意力與輸出",
  sentName: "句子 {s}",
  posPct: "位置 {j}・{pct}",
  outOf: "「{w}」的輸出",
  dimN: "第 {k} 維",
  statusSameVerb: "兩句中「{w}」的輸出完全相同：模型看不出誰在它前面、誰在後面",
  statusSame: "兩句中「{w}」的輸出完全相同：模型分不出是誰咬誰",
  statusDiff: "加上位置後，兩句的「{w}」輸出不同了（差最多的是第 {k} 維：{a} 對 {b}）",
  calcEmbK: "輸入只有詞向量",
  calcEmbNote: "注意力權重只由「詞和詞像不像」決定，跟它排在第幾個無關；換順序只是把同樣的數字換個位置排。",
  calcPeK: "位置編碼（d = 4）",
  calcSentK: "句子 {s} 的「{w}」",
  noteWhy: `為了好讀，這裡的查詢、鍵、值都直接用輸入向量（不乘權重矩陣），分數一樣是內積 ÷ √d（見 <a href="#selfatt">自注意力</a>）。位置編碼是<b>加</b>到詞向量上，不是另外接一段，所以維數不變。它在整個模型裡的位置見 <a href="#transformer">Transformer</a>。`,

  /* ② sin・cos */
  waveAria: "位置編碼熱圖",
  waveTitle: "每一列＝一個位置的 {d} 維向量（藍＝正、紅＝負）",
  posAxis: "位置↑",
  dimFast: "第 1 維（變得快）",
  dimSlow: "第 {d} 維（變得慢）",
  simAria: "和其他位置的相似度",
  simTitle: "位置 {p} 和每個位置的相似度（內積 ÷ (d/2)，自己＝1）",
  simAxis: "其他位置 →",
  statusWave: "位置 {p} 和相鄰的位置 {near} 相似度 {sn}；和隔 25 個的位置 {far} 只有 {sf}",
  calcRuleK: "偶數維用 sin、奇數維用 cos：",
  calcRule: "PE(pos, 2i) = sin(pos ÷ 10000<sup>2i/d</sup>)，PE(pos, 2i+1) = cos(pos ÷ 10000<sup>2i/d</sup>)",
  calcDim12K: "第 1・2 維（i = 0）",
  calcDim34K: "第 3・4 維（i = 1）",
  calcLastK: "最後兩維（i = {i}）",
  noteWave: `分母 10000<sup>2i/d</sup> 隨維度變大，波長從約 6.3 個位置一路拉長到上萬個位置，所以前幾維像秒針、後幾維像時針。原論文的 d 是 512。這種固定公式不用訓練；也有模型（例如 BERT）把位置向量當參數直接學。`,
});
