/* モーメンタムと Adam（concepts/optim.js）の日本語テキスト
   - である調。key は locales/zh/optim.js と一致させる
   - SGD・AdaGrad・RMSprop・Adam は固有名なのでプログラム側に置く。翻訳が必要なのは nameMom だけ */
Atlas.strings("ja", "optim", {
  lead: `<p><a href="#gd">勾配降下法</a>の細長い谷では、急な方向は行き来し、緩やかな方向はなかなか進まない。しかもモデル全体で学習率は 1 つしかない。大きくすれば急な方向が発散し、小さくすれば緩やかな方向がさらに遅くなる。改良の方向は 2 つある。<span class="term">モーメンタム</span>（<span class="en">momentum</span>）は移動に慣性を持たせる。AdaGrad や RMSprop は<b>パラメータごとに</b>、過去の勾配の大きさに応じて歩幅を自動で調整する。<span class="term">Adam</span> はその両方を組み合わせたものである。</p>`,
  closing: `モーメンタムは慣性を利用する。行き来する方向の動きは打ち消し合い、ずっと同じ向きの方向はどんどん速くなる。AdaGrad・RMSprop・Adam は各パラメータの勾配を「それが過去にどれだけ大きかったか」で割るため、どの方向がどれだけ急でも歩幅はほぼ同じになる。谷を急にしても、これらの経路はまったく変わらない。AdaGrad は全履歴を累積するため歩幅は小さくなる一方である。RMSprop は直近の勾配だけを覚えるように改め、Adam はそこに慣性を加えた。Adam は現在最もよく使われる標準的な選択肢である。`,

  nameMom: "モーメンタム",

  /* 操作 */
  focusLabel: "詳しく見る手法（学習率の調整と計算式）",
  kLabel: "y 方向は x 方向の何倍急か k",
  lrLabel: "学習率 η（詳しく見る手法）",
  btnOnly: "詳しく見る手法だけ表示",
  btnShowAll: "5 つすべて表示",
  btnDefaults: "学習率をすべて元に戻す",
  btnStep: "1 歩進む",
  btnReset: "リセット",

  /* 図 */
  canvasAria: "5 つの最適化手法の等高線上の経路",
  clickMap: "図をクリックすると出発点が変わる",
  chartAria: "ステップごとの谷底までの距離の変化",
  chartLabel: "谷底までの距離（対数目盛）",
  chartNear: "距離 {near}（「到達」とみなす）",
  chartAxisX: "ステップ数",

  /* 比較表 */
  thMethod: "手法",
  thLr: "学習率 η",
  thReach: "谷底付近に到達（距離 < {near}）",
  thDist: "現在の谷底までの距離",
  cellDead: "発散（{n} 歩目）",
  cellReach: "{n} 歩目",
  cellNotYet: "まだ",

  /* ステータス */
  statusIdle: "「1 歩進む」か「自動再生」を押す。5 つの手法が同じ点から同時に出発する（最大 {max} 歩）",
  statusDead: "{names} が発散した。y 方向が急すぎて、同じ学習率では 1 歩で谷の幅より遠くへ飛んでしまう。k を 10 に戻してみよう",
  nameSep: "、",
  statusEnd: "{max} 歩を終えた",
  statusSgd: "{n} 歩目：SGD の学習率 {lr} はちょうどこの急さに合わせて調整してあるので、最も速く到達する。k を大きくして、どれが持ちこたえるか見てみよう",
  statusStep: "{n} 歩目",

  /* 計算式 */
  calcDead: "{name} はすでに発散し、位置が計算範囲を超えた",
  calcPosK: "{name}　{n} 歩目　位置",
  calcGradK: "勾配",
  calcMove: "移動",
  calcMomNote: "（移動量は v そのもの）",
  calcEff: "実効学習率",
  calcCorr: "補正後",
  calcNextK: "次の位置",

  note: `ここでの地形は固定の式で、勾配は毎回地形全体から計算するため、ここでの「SGD」は<a href="#gd">勾配降下法</a>そのものである。実際の訓練では、この 5 つの手法はどれも<a href="#sgd">ミニバッチ</a>と組み合わせて使う。モーメンタムは慣性のせいで谷底を通り過ぎてから折り返すため、大きく回り込んでから止まることが多い。手法ごとに学習率が違うのは、「1 歩」の定義が違うからである。AdaGrad・RMSprop・Adam の移動量はおおよそ η そのものになる。β と ρ は一般的な既定値を使う（モーメンタム β = 0.9、RMSprop ρ = 0.9、Adam β₁ = 0.9・β₂ = 0.999）。Adam の m̂ と v̂ は、序盤に m と v が 0 から始まるため小さくなる分を補正したものである。ほかの改良として、<span class="term">AdaDelta</span> は RMSprop と同じく直近の勾配を使い、さらに学習率の設定も不要にした。<span class="term">AdaBound</span>（とその変形の <span class="term">AMSBound</span>）は Adam の実効学習率に次第に狭まる上限と下限を設け、訓練の後半には SGD のように振る舞う。学習率や β のように訓練の前に人が決める値を<span class="term">ハイパーパラメータ</span>（<span class="en">hyperparameter</span>）という。探し方としては、候補値の組み合わせをすべて試す<span class="term">グリッドサーチ</span>（<span class="en">grid search</span>）や、組み合わせをランダムに選んで試す<span class="term">ランダムサーチ</span>（<span class="en">random search</span>）がよく使われる。k = 10 では SGD が最も速いのに、k を大きくすると発散する。このように「あらゆる問題で最良の手法は存在しない」という事実を<span class="term">ノーフリーランチ定理</span>（<span class="en">no free lunch theorem</span>）という。手法の優劣は問題によって決まる。`,
});
