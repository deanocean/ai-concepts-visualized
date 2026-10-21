/* LSTM 的閘門（concepts/lstm.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、sentLine、calc*、fixedTo1、note 這類會以 HTML 顯示的項目
   - 例句（Mary walked to the park…）是英文，兩種語言共用，留在程式裡 */
Atlas.strings("zh", "lstm", {
  lead: `<p>普通 RNN 每一步都把隱藏狀態整個重算一次，早先的資訊很快被沖淡。<span class="term">LSTM</span>（<span class="en">long short-term memory</span>）另外準備一條<span class="term">記憶單元</span>（<span class="en">memory cell</span>）c，再用三個<span class="term">閘門</span>（<span class="en">gate</span>）——0〜1 之間的開關——決定舊記憶留多少、新內容寫多少、此刻拿出多少。</p>`,
  closing: `遺忘閘管「留」、輸入閘管「寫」、輸出閘管「拿」；三個閘都是 0〜1 的 sigmoid，所以開關的程度可以靠訓練學出來。記憶單元只做「乘上遺忘閘、再加上新內容」，資訊和梯度都能沿著它傳得很遠。`,

  /* 控制項 */
  sentLabel: "句子（空格要填 she 還是 he？）",
  sentA: "中間沒換人",
  sentB: "中間出現新人物",
  modeLabel: "閘門",
  modeN: "正常",
  modeF: "遺忘閘固定 1",
  modeI: "輸入閘固定 1",
  modeO: "輸出閘固定 1",
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnReset: "重設",
  sentLine: "句子：{s}　正解：{a}",
  lineLabel: "整句的經過（點一欄可跳到那一步）",

  /* 單元圖 */
  cellAria: "LSTM 單元這一步的計算",
  highway: "記憶單元的直通路（只有乘法和加法）",
  gateF: "遺忘閘",
  gateI: "輸入閘",
  gateO: "輸出閘",
  gateG: "候選內容",
  oldC: "舊記憶",
  oldCSub: "上一步留下的",
  newC: "新記憶",
  fixed1: "固定 1",
  outK: "輸出",
  wordNow: "這一步的詞",
  tagNameF: "人名・女性",
  tagNameM: "人名・男性",
  tagBlank: "要填代名詞",
  tagOther: "一般詞",

  /* 整句圖 */
  lineAria: "整句各步的閘門與記憶",
  rowWord: "詞",
  rowC: "記憶",

  /* 狀態列與算式 */
  statusStart: "按「下一步」逐詞送入，看記憶單元 c 怎麼記住「上一個人是女性（+）還是男性（−）」",
  calcStart: "每個詞先轉成 4 個特徵：是否人名、性別（女 +1／男 −1）、是否是要填代名詞的空格、其他內容（用一個數字代表）。三個閘門和候選內容都從這些特徵算出來。",
  calcFeatK: "特徵",
  calcFeat: "人名 {p}、性別 {sx}、空格 {q}、其他內容 {r}",
  calcF: "f = σ(−8×人名 + 5) = σ({z}) = {v}",
  calcI: "i = σ(8×人名 − 4) = σ({z}) = {v}",
  calcG: "g = tanh(2×性別 + 1.5×其他內容) = tanh({z}) = {v}",
  calcO: "o = σ(8×空格 − 4) = σ({z}) = {v}",
  fixedTo1: "固定成 1",
  calcC: `c = f × 舊記憶 + i × g = {f} × {cp} + {i} × {g} = <span class="hl">{c}</span>`,
  predK: "預測",
  calcPred: `P(she) = σ(6 × h) = σ({z}) = <span class="hl">{p}</span>，P(he) = {q}`,

  statusOutOpen: "答案 {pred}（{pct}%）沒錯，但輸出閘一直全開：句子中間每一步的輸出 h 都約 {h}，把「誰」的記憶硬塞進每個位置的輸出",
  statusFMixed: "分不出來：P(she) = {p}。遺忘閘不會關，舊的「女性」沒清掉，和新寫入的「男性」互相抵消（c = {c}）",
  statusIWrong: "答錯：預測 {pred}（{pct}%）。輸入閘一直全開，無關的詞也一路寫進記憶，c 從 {c0} 被推到 {c}",
  statusILucky: "這次碰巧答對 {pred}，但 c = {c}（正常是 {cn}）：無關的詞也寫進了記憶，結果只是剛好偏向同一邊",
  statusPredOk: "要填代名詞：輸出閘打開（{o}），把記憶拿出來 → 預測 {pred}（{pct}%），正確",
  statusPredBad: "要填代名詞：輸出閘打開（{o}），把記憶拿出來 → 預測 {pred}（{pct}%），錯誤",
  whoF: "女性",
  whoM: "男性",
  statusNameFixF: "人名「{w}」：遺忘閘被固定成 1，舊記憶清不掉，新的「{who}」直接疊上去 → c = {c}",
  statusNameF0: "人名「{w}」：遺忘閘固定 1；輸入閘打開（{i}），寫入「{who}」→ c = {c}",
  statusName: "人名「{w}」：遺忘閘幾乎關上（{f}），清掉舊記憶；輸入閘打開（{i}），寫入「{who}」→ c = {c}",
  statusOtherI: "「{w}」不是人名，但輸入閘被固定成 1：它的其他內容（g = {g}）也寫進記憶 → c 從 {cp} 變成 {c}",
  statusOtherO: "「{w}」：記憶保留得好好的，但輸出閘被固定成 1，這一步也把記憶整個輸出（h = {h}）",
  statusOther: "「{w}」不是人名：遺忘閘幾乎全開（{f}），輸入閘幾乎關上（{i}）→ 記憶幾乎不變（{cp} → {c}），輸出閘關著（h ≈ {h}）",

  note: `σ 是 <span class="term">Sigmoid 函數</span>（<span class="en">sigmoid function</span>），輸出永遠在 0〜1 之間，所以適合當「開多少」的比例；tanh 輸出在 −1〜1，用來產生可正可負的內容。為了好讀，這裡的閘門只看當下的詞，真實的 LSTM 還會一起看上一步的輸出 h，權重全由訓練學到。記憶單元這條只有乘法和加法的路，原始論文稱為 <span class="term">CEC</span>（<span class="en">constant error carousel</span>）：誤差沿它往回傳不會被 tanh 壓縮（對照 <a href="#rnn">RNN 與梯度消失</a> 的「換成 LSTM」）。把遺忘閘和輸入閘合併成一個<span class="term">更新閘</span>（<span class="en">update gate</span>）、再加上<span class="term">重設閘</span>（<span class="en">reset gate</span>）、不另設記憶單元的簡化版，叫 <span class="term">GRU</span>（<span class="en">gated recurrent unit</span>），參數較少。`,
});
