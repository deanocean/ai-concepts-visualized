/* RNN 與梯度消失（concepts/rnn-vanishing.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、*Label、calc*、note*、leg* 這類會以 HTML 顯示的項目
   - tokens：例句的 10 個詞，以半形空白分隔（程式依位置配上固定的數值） */
Atlas.strings("zh", "rnn", {
  lead: `<p>句子要按順序讀，後面的詞常要靠前面的內容才能理解。<span class="term">循環神經網路</span>（<span class="en">recurrent neural network</span>）也就是 RNN，每讀一個詞，就把「到目前為止讀到的內容」壓成一個<span class="term">隱藏狀態</span>（<span class="en">hidden state</span>）交給下一步。問題出在訓練：誤差要沿同一條路倒著傳回去，傳得越遠，梯度就越淡，前面的詞幾乎學不到東西。</p>`,
  closing: `RNN 每往回一步，梯度都要乘上「w<sub>h</sub> × tanh 的斜率」，這個數通常小於 1，連乘下去就趨近 0，這就是<span class="term">梯度消失</span>（<span class="en">vanishing gradient</span>）；LSTM 讓梯度沿記憶單元只乘上接近 1 的遺忘閘，所以能傳到很前面。`,

  tokens: "小明 昨天 在 車站 遇到 了 老同學 他 很 開心",

  /* 控制項 */
  modeLabel: "模型",
  modeLstm: "換成 LSTM",
  whLabel: "循環權重 w<sub>h</sub>（上一步狀態的權重）",
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnJump: "跳到反向傳播",
  btnReset: "重設",

  /* 展開圖 */
  svgAria: "沿時間展開的網路",
  rowIn: "輸入",
  rowInSub: "詞 → 數字",
  rowH: "隱藏狀態",
  rowC: "記憶單元",
  rowGrad: "梯度",
  outLabel: "輸出",
  lossLabel: "→ 損失",

  /* 狀態列與算式 */
  statusStart: "① 正向傳播：按「下一步」把句子一個詞一個詞送進去",
  calcStart: "每個詞先轉成數字（真實模型是一串數字，這裡簡化成一個）。隱藏狀態 h 也是一個數字，代表「到目前為止讀到的內容」的摘要，一開始是 0。",
  statusFwd: "① 正向傳播：送入第 {n} 個詞「{tok}」",
  calcRnnRule: "新的狀態 = tanh(w<sub>h</sub> × 上一步狀態 + w<sub>x</sub> × 這個詞)",
  gateF: "遺忘閘",
  gateI: "輸入閘",
  gateG: "候選內容",
  gateO: "輸出閘",
  statusBackStart: "② 反向傳播：在最後一步算出損失，梯度從這裡開始往回傳",
  calcBackStart: `<span class="k">把最後一步的梯度當作 1（100%），看它傳回每個詞時還剩多少。沿時間倒著做反向傳播，叫 </span><span class="term">BPTT</span><span class="k">（<span class="en">backpropagation through time</span>）。</span>`,
  statusBack: "② 反向傳播：梯度傳回第 {n} 個詞「{tok}」，剩 {g}",
  calcRnnStepK: "往回一步要乘上",
  calcLstmStepK: "沿記憶單元往回一步，只乘上下一步的遺忘閘",
  statusEndLstm: "傳回第 1 個詞時，梯度還剩 {g}（約 {pct}%）",
  statusEndRnn: "傳回第 1 個詞時，梯度只剩 {g}（{pct}%）：「{tok}」幾乎學不到",

  noteRnn: `(1 − h²) 是 tanh 在那一點的斜率，最大只有 1；h 越接近 ±1，斜率越接近 0。試著把 w<sub>h</sub> 調大：狀態被 tanh 壓到 ±1 附近，梯度反而消失得更快。若換成不會壓縮的激活函數、每步乘的數又大於 1，梯度會越傳越大，稱為<span class="term">梯度爆炸</span>（<span class="en">exploding gradient</span>）。同樣的連乘也發生在層數很深的網路，見 <a href="#vanish">梯度消失問題</a>。`,
  noteLstm: `<span class="term">LSTM</span>（<span class="en">long short-term memory</span>）多了一條記憶單元 c 的直通路：每步只做「乘上<span class="term">遺忘閘</span>（<span class="en">forget gate</span>）再加上新內容」，不經過 tanh 的壓縮。遺忘閘接近 1 時，梯度沿這條路往回傳幾乎不打折。各個閘門怎麼決定開關，見 <a href="#lstm">LSTM 的閘門</a>。`,

  /* 長條圖 */
  barAria: "傳回各詞的梯度大小",
  barTitle: "傳回各詞的梯度（最後一步＝1）",
  legLstm: "LSTM（沿記憶單元）",
  legRnn: "同一句用 RNN（w<sub>h</sub> = {wh}）",
});
