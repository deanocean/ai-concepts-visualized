/* RNN と勾配消失（concepts/rnn-vanishing.js）の日本語テキスト
   - である調。key は locales/zh/rnn.js と一致させる
   - tokens：例文の 10 語を半角スペースで区切る（各位置の数値はプログラム側で固定） */
Atlas.strings("ja", "rnn", {
  lead: `<p>文は順番に読む必要があり、後ろの語を理解するには前の内容が要ることが多い。<span class="term">リカレントニューラルネットワーク</span>（<span class="en">recurrent neural network</span>）、すなわち RNN は、1 語読むたびに「ここまでに読んだ内容」を 1 つの<span class="term">隠れ状態</span>（<span class="en">hidden state</span>）にまとめ、次のステップへ渡す。問題は学習にある。誤差は同じ道を逆向きにたどって伝わるが、遠くまで伝わるほど勾配は薄くなり、前のほうの語はほとんど学習されない。</p>`,
  closing: `RNN では 1 ステップ戻るたびに、勾配に「w<sub>h</sub> × tanh の傾き」が掛かる。この値はふつう 1 より小さく、掛け続けると 0 に近づく。これが<span class="term">勾配消失問題</span>（<span class="en">vanishing gradient problem</span>）である。LSTM では、勾配がメモリセルに沿って 1 に近い忘却ゲートの値だけを掛けられて戻るため、ずっと前まで届く。`,

  tokens: "太郎 は 昨日 駅 で 旧友 に 会い とても 喜んだ",

  /* 操作 */
  modeLabel: "モデル",
  modeLstm: "LSTM に切り替え",
  whLabel: "再帰の重み w<sub>h</sub>（1 つ前の状態に掛ける重み）",
  btnPrev: "◀ 前へ",
  btnNext: "次へ ▶",
  btnJump: "逆伝播へ進む",
  btnReset: "リセット",

  /* 展開図 */
  svgAria: "時間方向に展開したネットワーク",
  rowIn: "入力",
  rowInSub: "語 → 数値",
  rowH: "隠れ状態",
  rowC: "メモリセル",
  rowGrad: "勾配",
  outLabel: "出力",
  lossLabel: "→ 誤差",

  /* ステータスと計算式 */
  statusStart: "① 順伝播：「次へ」を押し、文を 1 語ずつ入力してみよう",
  calcStart: "各語はまず数値に変換される（実際のモデルでは数値の列だが、ここでは 1 つに簡略化している）。隠れ状態 h も 1 つの数値で、「ここまでに読んだ内容」の要約を表し、最初は 0 である。",
  statusFwd: "① 順伝播：{n} 番目の語「{tok}」を入力",
  calcRnnRule: "新しい状態 = tanh(w<sub>h</sub> × 1 つ前の状態 + w<sub>x</sub> × この語)",
  gateF: "忘却ゲート",
  gateI: "入力ゲート",
  gateG: "候補",
  gateO: "出力ゲート",
  statusBackStart: "② 逆伝播：最後のステップで誤差を計算し、ここから勾配を逆向きに伝える",
  calcBackStart: `<span class="k">最後のステップの勾配を 1（100%）とし、各語まで戻ったときにどれだけ残るかを見る。時間をさかのぼって誤差逆伝播法を行うことを </span><span class="term">BPTT</span><span class="k">（<span class="en">backpropagation through time</span>）という。</span>`,
  statusBack: "② 逆伝播：勾配が {n} 番目の語「{tok}」まで戻り、残りは {g}",
  calcRnnStepK: "1 ステップ戻るときに掛ける値",
  calcLstmStepK: "メモリセルに沿って 1 ステップ戻るときは、次のステップの忘却ゲートだけを掛ける",
  statusEndLstm: "1 番目の語まで戻っても、勾配はまだ {g}（約 {pct}%）残っている",
  statusEndRnn: "1 番目の語まで戻ると、勾配は {g}（{pct}%）しか残らない：「{tok}」はほとんど学習されない",

  noteRnn: `(1 − h²) はその点における tanh の傾きで、最大でも 1 である。h が ±1 に近いほど傾きは 0 に近づく。w<sub>h</sub> を大きくしてみよう。状態が tanh で ±1 付近に押しつぶされ、勾配はかえって速く消える。逆に、値を押しつぶさない活性化関数を使い、各ステップで掛ける値が 1 より大きいと、勾配は伝わるほど大きくなる。これを<span class="term">勾配爆発問題</span>（<span class="en">exploding gradient problem</span>）という。同じ掛け算の連鎖は層の深いネットワークでも起こる。<a href="#vanish">勾配消失問題</a> を参照。`,
  noteLstm: `<span class="term">LSTM</span>（<span class="en">long short-term memory</span>）には、メモリセル c をまっすぐ通る道がある。各ステップで行うのは「<span class="term">忘却ゲート</span>（<span class="en">forget gate</span>）を掛けて新しい内容を足す」だけで、tanh による圧縮を通らない。忘却ゲートが 1 に近ければ、勾配はこの道を通ってほとんど減らずに戻る。各ゲートの開閉の決まり方は <a href="#lstm">LSTM のゲート</a> を参照。`,

  /* 棒グラフ */
  barAria: "各語まで戻った勾配の大きさ",
  barTitle: "各語まで戻った勾配（最後のステップ＝1）",
  legLstm: "LSTM（メモリセル経由）",
  legRnn: "同じ文を RNN で（w<sub>h</sub> = {wh}）",
});
