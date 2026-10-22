/* 勾配降下法と学習率（concepts/gradient-descent.js）の日本語テキスト
   - である調。key は locales/zh/gd.js と一致させる
   - note1d は条件で文が変わるため関数で書く */
Atlas.strings("ja", "gd", {
  lead: `<p>モデルの訓練とは、<b>損失</b>（予測が正解からどれだけずれているか）が最小になるパラメータを探すことである。しかし地形の全体は見えず、足もとの傾きを測って下り坂へ一歩進むことしかできない。このやり方を<span class="term">勾配降下法</span>（<span class="en">gradient descent</span>）といい、一歩の大きさは<span class="term">学習率</span>（<span class="en">learning rate</span>）で決まる。</p>`,
  closing: `学習率が小さすぎると非常に多くのステップが必要になる。大きすぎると谷底を飛び越えて行き来し、さらには跳ね返るたびに遠ざかって発散する。谷がいくつもある地形では、どこから出発するかで最後にどの谷に落ちるかが決まる。`,

  /* 操作 */
  modeLabel: "地形",
  mode1d: "1D 曲線",
  mode2d: "2D 等高線",
  surfLabel: "2D 地形",
  surfValley: "細長い谷",
  surfTwin: "2 つの谷",
  lrLabel: "学習率 η",
  presetLabel: "クイック選択",
  presetBtn: "{label}（{v}）",
  presetSmall: "小さすぎ",
  presetGood: "適切",
  presetBig: "やや大きい",
  presetHuge: "大きすぎ",
  btnStep: "1 歩進む",
  btnReset: "リセット",
  statusIdle: "「1 歩進む」を押して始める",

  /* 1D の図 */
  svgAria: "損失の曲線と現在の位置",
  miniLabel: "ステップごとの損失の変化",
  axisW: "パラメータ w",
  curveLabel: "損失 L(w) = (w − 1)²",
  minLabel: "谷底 w = 1",
  flyRight: "ボールが画面外へ →",
  flyLeft: "← ボールが画面外へ",
  clickCurve: "曲線をクリックすると出発点が変わる",
  miniAxis: "ステップ数（0〜{n}）",

  /* 1D のステータス */
  s1Overflow: "発散：{n} 歩目で w が表示できないほど大きくなった",
  s1Done: "谷底に到着：全部で {n} 歩",
  s1Diverge: "発散：谷底を飛び越えるたびに遠くへ落ち、どんどん離れていく",
  s1DivergeIdle: "この学習率では発散する。「1 歩進む」を押してみよう",
  s1Bounce: "谷底の両側を行き来し続け、いつまでも止まらない",
  s1Osc: "振動：毎回谷底を飛び越えるが、幅はだんだん短くなり、最後には収束する",
  s1OscIdle: "この学習率では谷底を飛び越えて行き来する",
  s1Slow: "進みが遅い：{n} 歩目、谷底まであと {dist}",
  s1Steady: "谷底へ順調に進んでいる：{n} 歩目、谷底まで {dist}",

  /* 1D の計算式 */
  calcPosK: "現在の位置",
  calcLossK: "損失",
  calcStepPosK: "{n} 歩目　現在の位置",
  calcSlopeK: "傾き（勾配）",
  calcMoveK: "この 1 歩の移動",
  calcMoveSym: "−η × 傾き",
  calcNextK: "次の位置",
  note1d: ({ r, flip, near }) => `この曲線では、1 歩ごとに谷底までの距離が元の |1 − 2η| = <b class="mono">${r}</b> 倍になる${flip ? "（しかも谷底の反対側に移る）" : ""}。${near ? "1 より小さいときだけ近づいていく。" : "1 以上なので近づかない。"}`,

  /* 2D の図 */
  canvasAria: "損失の等高線図と移動の経路",
  markValley: "谷底",
  markGlobal: "大域最適解",
  markLocal: "局所最適解",
  markSaddle: "鞍点",
  clickMap: "図をクリックすると出発点が変わる",

  /* 2D のステータス */
  s2Out: "発散：{n} 歩目で地図の外へ飛び出した",
  s2Local: "局所最適解で停止（{n} 歩目）：周りはすべて上り坂で勾配≈0 のため、抜け出せない",
  s2ReachGlobal: "大域最適解に到着：全部で {n} 歩",
  s2ReachValley: "谷底に到着：全部で {n} 歩",
  s2Zigzag: "ジグザグに進む：急な y 方向は行き来し、緩やかな x 方向はゆっくり進む（{n} 歩目）",
  s2Step: "{n} 歩目、損失 {loss}",

  /* 2D の計算式 */
  calcOut: "位置が地図の外に出た",
  calcStopK: "停止位置",
  calcStepPos2K: "{n} 歩目　位置",
  calcGradK: "勾配（各方向の傾き）",
  calcNext2Sym: "(x − η·勾配x, y − η·勾配y)",

  noteValley: `色が濃いほど損失が大きく、線は等高線である。この谷は y 方向の傾きが x 方向の 10 倍急である。同じ学習率が、y 方向には大きすぎ（行き来する）、x 方向には小さすぎる（なかなか進まない）ことがある。改良した進み方は <a href="#optim">モーメンタムと Adam</a>、1 歩ごとにデータの一部だけで傾きを見積もる方法は <a href="#sgd">確率的勾配降下法とミニバッチ</a> を参照。`,
  noteTwin: `この地形には谷が 2 つある。左の深い谷が<span class="term">大域最適解</span>（<span class="en">global optimum</span>）、右の浅い谷が<span class="term">局所最適解</span>（<span class="en">local optimum</span>）である。2 つの谷の間の四角は<span class="term">鞍点</span>（<span class="en">saddle point</span>）で、x 方向に見ると山頂、y 方向に見ると谷底であり、やはり勾配は 0 になる。左右それぞれから出発してみよう。`,
});
