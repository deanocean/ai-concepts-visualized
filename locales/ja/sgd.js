/* 確率的勾配降下法とミニバッチ（concepts/sgd.js）の日本語テキスト
   - である調。key は locales/zh/sgd.js と一致させる
   - 凡例（leg*）は固定幅の枠に入るため短く保つ */
Atlas.strings("ja", "sgd", {
  lead: `<p><a href="#gd">勾配降下法</a>では、1 歩進むたびに<b>全部</b>のデータの誤差を計算しないと傾きがわからない。データが数百万件あると、傾きを 1 回求めるのにも、1 歩進むのにも長い時間がかかる。<span class="term">確率的勾配降下法</span>（<span class="en">stochastic gradient descent, SGD</span>）は、毎回データの一部だけをランダムに取り出して傾きを見積もる。見積もりの精度は下がるが、同じ計算量で何歩も進める。</p>`,
  closing: `毎回全データを使う<span class="term">バッチ学習</span>（<span class="en">batch learning</span>）は方向が最も正確だが、1 エポックで 1 歩しか進めない。毎回 1 件だけ使うと 1 エポックで 16 歩進み、すぐに谷底へ近づくが、方向がぶれて谷底付近でも止まらない。<span class="term">ミニバッチ学習</span>（<span class="en">mini-batch learning</span>）はその中間で、現在のニューラルネットワークの訓練の標準的な方法である。`,

  /* 操作 */
  bLabel: "勾配の計算に毎回使うデータ数（バッチサイズ）",
  bAll: "16 件すべて",
  bN: "{n} 件",
  lrLabel: "学習率 η",
  btnStep: "1 回更新",
  btnEpoch: "このエポックを最後まで",
  btnReset: "リセット",

  /* 図 */
  dataAria: "データ点と現在の直線",
  parAria: "パラメータ平面上の損失の等高線と経路",
  capData: "データと現在の直線 y = a·x + b",
  capPar: "パラメータ (a, b) の損失地形（全データで計算）",
  legBatch: "このバッチ",
  legAll: "すべて使用",
  legUsed: "使用済み",
  legUnused: "未使用",
  legErr: "誤差",
  legStepBatch: "バッチの 1 歩",
  legStepAll: "全データの 1 歩",
  markValley: "谷底",
  axisA: "a（傾き）",
  clickMap: "図をクリックすると出発点が変わる",

  /* 比較図 */
  cmpAria: "3 つのやり方で 5 エポック学習したときの損失の比較",
  cmpLabel: "同じく 5 エポック（どのやり方も各データを 5 回ずつ見る）学習したときの、全データでの損失の比較",
  cmpAxisX: "エポック",
  cmpAxisY: "損失 − 谷底の損失（対数目盛）",
  cmpLeg: "━ {name}（{per} 回更新／エポック）：5 エポック後の差 {extra}",
  cmpHint: "太線が現在選んでいるやり方、丸が現在の進み具合である。",

  /* カウンタ */
  countIter: "<span class=\"term\">イテレーション</span>（<span class=\"en\">iteration</span>）数 <b>{n}</b>",
  countEpoch: "エポック（全データを見終えた回数）<b>{n}</b>",
  countGrads: "計算したサンプルの勾配 <b>{n}</b> 件",
  countLoss: "全データでの損失 <b>{loss}</b>（谷底 {opt}）",

  /* ステータス */
  statusDone: "{n} エポックを終えた。「リセット」でやり直せる",
  statusFullIdle: "「1 回更新」を押して始める。毎回 16 件すべてを使うので方向は最も正確だが、1 エポックで 1 回しか更新しない",
  statusFull: "{n} 回目の更新、谷底まで {dist}：方向はずっと正確だが、データを一通り見るごとに 1 歩しか進まない",
  statusNear: "谷底付近にいる（距離 {dist}）が、バッチごとに見積もった方向が異なるため、付近で揺れ続ける",
  statusBatch: "次の 1 歩は {b} 件だけで傾きを見積もる。全データで計算した方向とのずれは {ang}°",

  /* 計算式 */
  calcPosK: "現在の位置",
  calcLossK: "損失",
  calcUpdK: "{n} 回目の更新（{e} エポック目の {k} 番目のバッチ）　このバッチ",
  listSep: "、",
  calcErrK: "各データの誤差 a·x + b − y",
  errSep: "；",
  calcGradK: "このバッチで見積もった勾配",
  calcGradSym: "(平均(誤差 × x), 平均(誤差))",
  calcFullK: "16 件すべてで計算",
  calcNextK: "次の位置",
  calcNextSym: "(a − η·勾配a, b − η·勾配b)",

  note: `損失には<span class="term">平均二乗誤差</span>（<span class="en">mean squared error</span>）をさらに 2 で割ったものを使う（2 で割るのは勾配の式から係数 2 を消すためだけである）。各エポックの始めに 16 件のデータをシャッフルし、順に区切ってバッチにするため、各データは 1 エポックの中でちょうど 1 回ずつ使われる。ここでは乱数を固定しているので、リセットするたびに同じ結果になる。「確率的」とはランダムに抽出することを指す。データが 1 件来るたびにすぐ更新するやり方は<span class="term">オンライン学習</span>（<span class="en">online learning</span>）とも呼ぶ。1 つのバッチの複数件の計算は GPU で同時に行えるため、実務では 32〜256 件程度のミニバッチを使う。方向にランダムなぶれがあることで、かえって<span class="term">局所最適解</span>（<span class="en">local optimum</span>）や<span class="term">鞍点</span>（<span class="en">saddle point</span>）から抜け出しやすくなることもある。学習率や進み方の改良は <a href="#optim">モーメンタムと Adam</a> を参照。`,
});
