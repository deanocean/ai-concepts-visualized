/* LSTM のゲート（concepts/lstm.js）の日本語テキスト
   - である調。key は locales/zh/lstm.js と一致させる
   - 例文（Mary walked to the park…）は英語で、両言語共通のためプログラム側に置く */
Atlas.strings("ja", "lstm", {
  lead: `<p>通常の RNN は各ステップで隠れ状態をまるごと計算し直すため、前のほうの情報はすぐに薄まってしまう。<span class="term">LSTM</span>（<span class="en">long short-term memory</span>）は別に<span class="term">メモリセル</span>（<span class="en">memory cell</span>）c を用意し、さらに 3 つの<span class="term">ゲート</span>（<span class="en">gate</span>）——0〜1 の間で開き具合が決まるスイッチ——で、古い記憶をどれだけ残すか、新しい内容をどれだけ書き込むか、いまどれだけ取り出すかを決める。</p>`,
  closing: `忘却ゲートは「残す」、入力ゲートは「書き込む」、出力ゲートは「取り出す」を担う。3 つとも 0〜1 を出力するシグモイド関数なので、開き具合を学習で決められる。メモリセルで行うのは「忘却ゲートを掛けて新しい内容を足す」だけであり、情報も勾配もこれに沿って遠くまで伝わる。`,

  /* 操作 */
  sentLabel: "文（空欄に入るのは she か he か？）",
  sentA: "途中で人物が変わらない",
  sentB: "途中で新しい人物が登場",
  modeLabel: "ゲート",
  modeN: "通常",
  modeF: "忘却ゲートを 1 に固定",
  modeI: "入力ゲートを 1 に固定",
  modeO: "出力ゲートを 1 に固定",
  btnPrev: "◀ 前へ",
  btnNext: "次へ ▶",
  btnReset: "リセット",
  sentLine: "文：{s}　正解：{a}",
  lineLabel: "文全体の経過（列をクリックするとそのステップへ移動）",

  /* セルの図 */
  cellAria: "このステップにおける LSTM ユニットの計算",
  highway: "メモリセルの直通路（掛け算と足し算だけ）",
  gateF: "忘却ゲート",
  gateI: "入力ゲート",
  gateO: "出力ゲート",
  gateG: "候補",
  oldC: "古い記憶",
  oldCSub: "前のステップから",
  newC: "新しい記憶",
  fixed1: "1 に固定",
  outK: "出力",
  wordNow: "このステップの語",
  tagNameF: "人名・女性",
  tagNameM: "人名・男性",
  tagBlank: "代名詞を入れる空欄",
  tagOther: "一般の語",

  /* 文全体の図 */
  lineAria: "文の各ステップにおけるゲートと記憶",
  rowWord: "語",
  rowC: "記憶",

  /* ステータスと計算式 */
  statusStart: "「次へ」で 1 語ずつ入力し、メモリセル c が「直前の人物は女性（+）か男性（−）か」をどう覚えるかを見てみよう",
  calcStart: "各語はまず 4 つの特徴に変換される：人名かどうか、性別（女性 +1／男性 −1）、代名詞を入れる空欄かどうか、その他の内容（1 つの数値で表す）。3 つのゲートと候補は、これらの特徴から計算される。",
  calcFeatK: "特徴",
  calcFeat: "人名 {p}、性別 {sx}、空欄 {q}、その他 {r}",
  calcF: "f = σ(−8×人名 + 5) = σ({z}) = {v}",
  calcI: "i = σ(8×人名 − 4) = σ({z}) = {v}",
  calcG: "g = tanh(2×性別 + 1.5×その他) = tanh({z}) = {v}",
  calcO: "o = σ(8×空欄 − 4) = σ({z}) = {v}",
  fixedTo1: "1 に固定",
  calcC: `c = f × 古い記憶 + i × g = {f} × {cp} + {i} × {g} = <span class="hl">{c}</span>`,
  predK: "予測",
  calcPred: `P(she) = σ(6 × h) = σ({z}) = <span class="hl">{p}</span>、P(he) = {q}`,

  statusOutOpen: "答え {pred}（{pct}%）は正しいが、出力ゲートが常に全開である：文の途中の各ステップでも出力 h が約 {h} になり、「誰か」の記憶がすべての位置の出力に押し込まれている",
  statusFMixed: "判別できない：P(she) = {p}。忘却ゲートが閉じないため古い「女性」が消えず、新しく書き込んだ「男性」と打ち消し合っている（c = {c}）",
  statusIWrong: "不正解：予測は {pred}（{pct}%）。入力ゲートが常に全開なので、無関係な語までどんどん記憶に書き込まれ、c は {c0} から {c} まで押し流された",
  statusILucky: "今回はたまたま {pred} で正解したが、c = {c}（通常は {cn}）：無関係な語も記憶に書き込まれ、たまたま同じ側に偏っただけである",
  statusPredOk: "代名詞を入れる位置：出力ゲートが開き（{o}）、記憶を取り出す → 予測 {pred}（{pct}%）、正解",
  statusPredBad: "代名詞を入れる位置：出力ゲートが開き（{o}）、記憶を取り出す → 予測 {pred}（{pct}%）、不正解",
  whoF: "女性",
  whoM: "男性",
  statusNameFixF: "人名「{w}」：忘却ゲートが 1 に固定されているため古い記憶を消せず、新しい「{who}」がそのまま上に重なる → c = {c}",
  statusNameF0: "人名「{w}」：忘却ゲートは 1 に固定。入力ゲートが開き（{i}）、「{who}」を書き込む → c = {c}",
  statusName: "人名「{w}」：忘却ゲートがほぼ閉じて（{f}）古い記憶を消し、入力ゲートが開いて（{i}）「{who}」を書き込む → c = {c}",
  statusOtherI: "「{w}」は人名ではないが、入力ゲートが 1 に固定されている：その他の内容（g = {g}）も記憶に書き込まれる → c が {cp} から {c} に変わる",
  statusOtherO: "「{w}」：記憶はきちんと保たれているが、出力ゲートが 1 に固定されているため、このステップでも記憶をまるごと出力してしまう（h = {h}）",
  statusOther: "「{w}」は人名ではない：忘却ゲートはほぼ全開（{f}）、入力ゲートはほぼ閉じている（{i}）→ 記憶はほとんど変わらず（{cp} → {c}）、出力ゲートも閉じている（h ≈ {h}）",

  note: `σ は<span class="term">シグモイド関数</span>（<span class="en">sigmoid function</span>）で、出力は必ず 0〜1 の間に収まるため「どれだけ開くか」の割合に向いている。tanh の出力は −1〜1 で、正にも負にもなる内容を作るのに使う。読みやすさのため、ここのゲートは現在の語だけを見ているが、実際の LSTM は 1 つ前の出力 h も合わせて見ており、重みはすべて学習で決まる。掛け算と足し算だけでできたメモリセルの道は、元の論文で <span class="term">CEC</span>（<span class="en">constant error carousel</span>）と呼ばれる。誤差はこれに沿って戻っても tanh で圧縮されない（<a href="#rnn">RNN と勾配消失</a> の「LSTM に切り替え」と比べてみよう）。忘却ゲートと入力ゲートを 1 つの<span class="term">更新ゲート</span>（<span class="en">update gate</span>）にまとめ、<span class="term">リセットゲート</span>（<span class="en">reset gate</span>）を加え、メモリセルを別に持たない簡略版を <span class="term">GRU</span>（<span class="en">gated recurrent unit</span>）といい、パラメータが少ない。`,
});
