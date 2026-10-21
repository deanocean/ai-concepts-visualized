/* 教師強制（concepts/teacher.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、heatLabel、calc*、note 這類會以 HTML 顯示的項目
   - bos / eos：句首、句尾記號；eosShort 是機率長條旁的短寫 */
Atlas.strings("zh", "teacher", {
  lead: `<p>RNN 的<span class="term">解碼器</span>（<span class="en">decoder</span>）一次產生一個詞，而每一步的輸入是「前一個詞」。訓練時如果把模型<b>自己的預測</b>當成下一步的輸入，前面只要錯一個詞，後面整串都建立在錯誤上，損失也就在教它錯的東西。<span class="term">教師強制</span>（<span class="en">teacher forcing</span>）的做法是：訓練時下一步一律餵<b>正解</b>的詞，不管模型剛才預測了什麼。</p>`,
  closing: `教師強制讓每一步都在「前文正確」的情況下學習，錯一個詞不會拖累後面，所以學得快也學得穩（還能一次平行算完所有步）。代價是推論時沒有正解可餵、只能餵自己的預測，模型會遇到訓練時沒看過的錯誤前文。`,

  bos: "〈開始〉",
  eos: "〈結束〉",
  eosShort: "〈結〉",

  /* 控制項 */
  modeLabel: "訓練時，下一步的輸入用什麼",
  modeTf: "教師強制（餵正解）",
  modeFr: "自由生成（餵自己的預測）",
  btnPrev: "◀ 上一步",
  btnNext: "下一步 ▶",
  btnReset: "回到第 0 步",
  btnUpd1: "用這 4 步的損失更新 1 次",
  btnUpd5: "連續更新 5 次",
  btnResetM: "兩個模型都還原",
  trainLabel: "訓練：用上面 4 步的損失做梯度下降（η = 0.5），只更新這個做法自己的那份模型",
  cmpLabel: "兩種做法的模型比一比（都從同一個「訓練到一半」的模型開始）",
  heatLabel: "模型本身：前一個詞 → 下一個詞的機率（橘框＝這次 4 步用到的「正解」那一格；橘色列名＝上次更新改到的列）",

  /* 解碼圖 */
  svgAria: "解碼器每一步的輸入、預測與正解",
  rowTgt: "正解（訓練資料）",
  rowIn: "這一步的輸入",
  rowP: "模型給的機率",
  rowPSub: "（下一個詞）",
  rowPred: "模型的預測",
  rowLoss: "這一步的損失",

  /* 狀態列 */
  nameTf: "教師強制",
  nameFr: "自由生成",
  statusStart0: "還沒開始：按「下一步」看解碼器一步一步產生譯文（{name}，模型是訓練到一半的狀態）",
  statusStartN: "還沒開始：按「下一步」看解碼器一步一步產生譯文（{name}，這份模型已更新 {n} 次）",
  statusStepOk: "第 {k} 步：輸入「{inp}」→ 模型最想說「{pred}」，正解是「{tgt}」，答對。",
  statusStepBad: "第 {k} 步：輸入「{inp}」→ 模型最想說「{pred}」，正解是「{tgt}」，答錯。",
  statusNextTf: "下一步的輸入：正解「{w}」",
  statusNextFr: "下一步的輸入：自己的預測「{w}」",
  statusAllOk: "4 步全對，損失合計 {tot}",
  statusDoneTf: "4 步中錯 {wrong} 步，損失合計 {tot}；第 {k} 步錯了，但後面的輸入照樣是正解，所以錯誤沒有往後傳",
  statusDoneFr: "4 步中錯 {wrong} 步，損失合計 {tot}；第 {k} 步錯了以後，後面每一步的輸入都是錯的，損失在教「{inp} 之後要接 {tgt}」這種本來就不對的事",

  /* 算式 */
  calcModelK: "這個簡化的解碼器",
  calcModel: "只看前一個詞來猜下一個詞：P(下一個詞 | 前一個詞) = softmax(這個詞那一列的分數)",
  calcLossK: "每一步的損失",
  calcLoss: "−ln P(正解的詞)；猜對而且很有把握時接近 0",
  calcRowK: "第 {k} 步，輸入「{inp}」那一列的分數",
  lossK: "損失",
  sumK: "4 步合計",

  /* 表格 */
  thN: "更新次數",
  thGen: "自己生成的句子",
  thPc: "正解整句的機率",
  heatCorner: "前一個詞 ↓ ／ 下一個 →",

  note: `為了看得清楚，這個解碼器只看前一個詞，分數是設計好的示範值；真實的解碼器（<a href="#seq2seq">Seq2Seq</a> 的 RNN 或 <a href="#transformer">Transformer</a>）還會看原文與整段前文，數字全由訓練學到。自由生成這邊，「am 之後接 cats」的機率會被一路推高（看比較表最右欄），這就是後面的步驟在學錯的東西；而教師強制下，am 那一列根本不會被用到。訓練只看過正解前文、推論時卻要接自己的預測，兩者的落差叫<span class="term">曝光偏差</span>（<span class="en">exposure bias</span>）；實務上有時會在訓練後期逐漸改餵自己的預測來縮小這個落差。Transformer 訓練時也用教師強制：把整句正解一次餵進去，靠<span class="term">遮罩</span>（<span class="en">mask</span>）讓每個位置只看得到前面的詞，所以所有位置能平行計算。`,
});
