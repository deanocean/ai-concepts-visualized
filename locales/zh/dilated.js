/* 感受野與空洞卷積（concepts/dilated.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、*Label、calc*、note* 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "dilated", {
  lead: `<p>要判斷一個像素屬於什麼（例如這一點是路面還是車），得看它周圍夠大的範圍。一個輸出看得到的輸入範圍叫<span class="term">感受野</span>（<span class="en">receptive field</span>）。一般的 3×3 卷積每疊一層只多看 2 格；<span class="term">空洞卷積</span>（<span class="en">dilated convolution</span>）讓濾波器隔著空隙取樣，不增加權重就看得更遠。</p>`,
  closing: `每層的間隔加倍（1, 2, 4, 8…），感受野就隨層數指數成長；權重數量和一般卷積完全一樣，特徵圖也不必像池化那樣縮小、丟掉細節。所以它常用在需要逐像素判斷的<span class="term">語意分割</span>（<span class="en">semantic segmentation</span>），例如 DeepLab，以及要看很長一段聲音的 WaveNet。`,

  /* 控制項 */
  viewLabel: "看哪一部分",
  viewTower: "① 疊幾層看得多遠（一維）",
  viewGrid: "② 2D 的 3×3 長什麼樣",
  modeLabel: "卷積的種類",
  modeNormal: "一般卷積（間隔 1）",
  modeDilated: "空洞卷積（間隔 1、2、4、8）",
  layersLabel: "層數",
  layersFmt: "{n} 層",
  rateLabel: `間隔 r（<span class="en">dilation rate</span>）`,
  btnPrev: "◀ 上一步",
  btnNext: "往下追一層 ▶",
  btnAll: "全部顯示",
  btnReset: "重設",

  /* ① 一維 */
  towerAria: "多層卷積中，一個輸出用到的輸入範圍",
  layerIn: "輸入",
  layerK: "第 {l} 層",
  rateK: "間隔 {d}",
  pickAria: "選第 {l} 層第 {i} 格",
  rfLabel: "感受野 {n} 格",
  statusStart: "橘點是選中的輸出。按「往下追一層」，看它用到下一層的哪 3 格",
  statusTraced: "已追到第 {l} 層：用到 {n} 格",
  modeNameNormal: "一般卷積",
  modeNameDilated: "空洞卷積",
  statusDone: "{mode}疊 {layers} 層：一格輸出看得到輸入 {rf} 格{clip}；{other}同樣 {layers} 層是 {rfO} 格",
  statusClip: "（靠邊被截掉，完整是 {ideal} 格）",
  hintTower: "點最上層任一格，換一個輸出來追。",
  calcRfK: "感受野",
  calcRf: `1 + (卷積核 3 − 1) × ({ds}) = 1 + 2 × {sum} = <span class="hl">{rf}</span> 格`,
  calcWK: "權重",
  calcW: "每層 3 個 × {layers} 層 = <b>{w}</b> 個（兩種卷積一樣多）",
  calcNeedK: "若要一般卷積看到同樣範圍",
  calcNeed: "需要疊 ({rf} − 1) ÷ 2 = <b>{n}</b> 層",
  noteTower: `一般卷積的感受野每層只多 2 格（線性成長）；間隔每層加倍時，感受野是 2<sup>層數+1</sup> − 1（指數成長）。另一個擴大感受野的方法是池化，但它會把特徵圖縮小、丟掉位置細節，逐像素分類時不划算。示範用一維、每層 3 個權重；輸入外側視為補 0（填充），所以每層的長度不變。間隔取樣的規則見 ② 的 2D 圖。`,

  /* ② 二維 */
  gridAria: "間隔 {r} 的 3×3 卷積核取樣的位置",
  statusGrid: "間隔 {r}：9 個權重散開取樣，覆蓋 {span}×{span} = {area} 格的範圍",
  calcCoverK: "覆蓋範圍",
  calcCover: `邊長 = 2 × 間隔 {r} + 1 = {span}，面積 {span}×{span} = <span class="hl">{area}</span> 格`,
  calcW2: "仍是 3×3 = <b>9</b> 個；中間的空格不參與計算（等於一個 {span}×{span}、空格處固定為 0 的濾波器，但那些 0 不用學）",
  calcPlainK: "一般 3×3",
  calcPlain: "就是間隔 1 的特例：覆蓋 3×3 = 9 格",
  noteGrid: `空洞卷積的英文有兩種叫法：dilated convolution 和 <span class="term">atrous convolution</span>。atrous 來自法文 à trous（帶著洞），指濾波器中間的空隙。兩個名字指同一種卷積，G 檢定的大綱兩個都有列出。`,
});
