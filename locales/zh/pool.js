/* 池化（concepts/pooling.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、cap*、calc* 這類會以 HTML 顯示的項目
   - 注意：中文的「列」是橫排（row）、「行」是直排（column），日文剛好相反 */
Atlas.strings("zh", "pool", {
  lead: `<p>卷積算出的<span class="term">特徵圖</span>（<span class="en">feature map</span>）很大，而且物體只要挪動一兩個像素，整張圖的數字就跟著變。<span class="term">池化</span>（<span class="en">pooling</span>）把每個小區塊濃縮成一個數：資料量變小，小小的位置偏移也被吸收掉。</p>`,
  closing: `最大池化只留下每個區塊裡最強的反應，所以特徵在區塊內稍微移動，輸出不變，這就是<span class="term">不變性</span>（<span class="en">invariance</span>）：對小幅的位置偏移不敏感。它只是取最大值或平均，沒有任何要學習的權重。`,

  /* 控制項 */
  kindLabel: "池化的種類",
  kindMax: "最大池化",
  kindAvg: "平均池化",
  kindGap: "全域平均池化（GAP）",
  shortMax: "最大值",
  shortAvg: "平均",
  moveGroup: "移動輸入",
  moveUp: "往上移 1 格",
  moveDown: "往下移 1 格",
  moveLeft: "往左移 1 格",
  moveRight: "往右移 1 格",
  moveHome: "回到原位",
  moveLabel: "把輸入整體移動 1 格（模擬物體稍微挪動，最多 ±1）",
  btnPrev: "◀ 上一格",
  btnNext: "下一格 ▶",
  btnAll: "全部算完",
  btnClear: "清空",

  /* 格子（螢幕報讀用） */
  cellAria: "{name}第 {r} 列第 {c} 行，值 {v}",
  cellEmpty: "未計算",
  nameIn: "輸入",
  nameOut: "輸出",
  nameRef: "不移動時的輸出",

  /* 圖說 */
  capIn: "<b>輸入</b>（特徵圖）{n}×{n}",
  capInBlocks: "粗線＝2×2 的區塊",
  capInMoved: "已移動：{dirs}",
  dirLeft: "左 1 格",
  dirRight: "右 1 格",
  dirUp: "上 1 格",
  dirDown: "下 1 格",
  dirSep: "、",
  opGap: "整張平均",
  opWin: "2×2 取{short}",
  capOut: "<b>輸出</b> {size}",
  sizeOne: "1 個數",
  capOutChanged: "紅框＝和不移動時不同",
  capOutPick: "點格子可跳到那一格",
  capRef: "不移動時的輸出<br>（對照用）",

  /* 狀態列 */
  statusIdle: "36 個數縮成 9 個。算完後試著用方向鍵移動輸入，看輸出有多少格改變",
  statusGapIdle: "GAP 把整張特徵圖平均成 1 個數。試著移動輸入",
  statusMoved: "輸入有 {inChg} 格的值變了；輸出 9 格中 {outChg} 格改變。{extra}",
  statusGapMoved: "輸入有 {inChg} 格的值變了，但整張的平均 {avg} 完全不變：只要特徵還在圖裡，位置在哪都一樣",
  extraMaxKeep: "最強的反應 9 仍在輸出的同一格",
  extraMaxMove: "最強的反應 9 跨過了區塊邊界，換到另一格",
  extraAvg: "平均會把移進、移出的小數字都算進去，所以變動比較多",

  /* 算式 */
  calcStartK: "按「下一格」",
  calcStart: "橘框蓋住左上角 2×2，取{short}，寫進輸出第一格；接著往右跳 2 格（步幅 2）",
  calcStartGap: "把整張 36 格加總再除以 36",
  calcGap: `36 格加總 {sum} ÷ 36 = <span class="hl">{avg}</span>`,
  calcWhyK: "為什麼常用",
  calcWhy: "最後一層特徵圖 7×7、512 通道，要分 1000 類：攤平後接全連接層要 7×7×512×1000 = {big} 個權重；先做 GAP 變成 512 個數，只要 512×1000 = {small} 個",
  calcCellK: "輸出第 {r} 列第 {c} 格",
  calcSizeK: "尺寸",
  calcSize: "邊長 (6 − 2) ÷ 步幅 2 + 1 = 3，數量 36 → 9（剩 1/4）",

  note: `示範的 6×6 是某個濾波器算出的特徵圖（數字越大＝越像它要找的形狀）。窗的大小和<span class="term">步幅</span>（<span class="en">stride</span>）通常一樣，區塊之間不重疊。只有在同一個 2×2 區塊裡的移動會被完全吸收，跨過區塊邊界就會改變；疊好幾層卷積＋池化，能容忍的偏移就逐層變大。GAP 直接把每張特徵圖平均成一個數，常用來取代最後的<span class="term">全連接層</span>（<span class="en">fully connected layer</span>）。卷積本身見 <a href="#conv">卷積</a>。`,
});
