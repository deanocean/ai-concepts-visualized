/* 誤差逆伝播法と連鎖律（concepts/backprop.js）の日本語テキスト
   - である調。key は locales/zh/backprop.js と一致させる
   - w₁・h₁・ŷ などの記号と数式はプログラム側に置く */
Atlas.strings("ja", "backprop", {
  lead: `<p>ネットワークの出力が間違っていても、調整すべき重みは各層に散らばっている。それぞれの重みがこの誤差にどれだけ責任を負い、どちらへどれだけ調整すべきか。これを<span class="term">信用割当問題</span>（<span class="en">credit assignment problem</span>）という。<span class="term">誤差逆伝播法</span>（<span class="en">backpropagation</span>）は損失から入力側へ逆向きにたどり、ノードを通るたびにそこでの局所的な傾きを掛けていく。1 回たどるだけで、すべての重みの勾配が求まる。その背後にある数学が<span class="term">連鎖律</span>（<span class="en">chain rule</span>）である。</p>`,
  closing: `各重みの勾配は、損失から結合をたどって逆向きに進み、途中の各区間の局所的な傾きを掛け合わせたものである。出力から遠い重みほど掛ける項が多くなる。それらの項がどれも 1 より小さければ、勾配は掛けるほど小さくなる。これが次の問題、<a href="#vanish">勾配消失問題</a>である。`,

  /* 操作 */
  btnPrev: "◀ 前へ",
  btnNext: "次へ ▶",
  btnReset: "リセット",
  btnUpdate: "勾配で重みを更新（η = {eta}）",
  btnUpdate10: "10 回続けて更新",

  /* ネットワーク図 */
  svgAria: "2 → 2 → 1 の小さなネットワーク、順伝播の値と逆伝播の勾配",
  layerIn: "入力層",
  layerHidden: "隠れ層（シグモイド）",
  layerOut: "出力層",
  layerLoss: "損失",
  target: "正解 t = {v}",
  legend: "青枠＝順伝播で求めた値　オレンジの ∂＝それに対する損失の勾配（∂L/∂…）",
  clickEdge: "結合をクリックすると連鎖律が見られる",

  /* ステータスと計算式：① 順伝播 */
  status0: "① 順伝播：「次へ」を押して、入力から 1 層ずつ右へ計算する",
  calcInK: "入力",
  calcIn: "x₁ = {x1}、x₂ = {x2}、正解 t = {t}",
  calcWK: "重み",
  listSep: "、",
  status1: "① 順伝播：隠れ層を求める",
  calcSigK: "σ はシグモイド関数",
  calcSig: "σ(z) = 1 ÷ (1 + e<sup>−z</sup>)。どんな数も 0〜1 の間に押し込める",
  status2: "① 順伝播：出力を求める",
  status3: "① 順伝播の完了：損失を求めた",
  statusUpdated: "{n} 回更新した：損失 {from} → {to}",
  calcNextK: "次は",
  calcNext: "「次へ」を押して、損失から逆向きに各重みの勾配を求める",

  /* ② 逆伝播 */
  status4: "② 逆伝播：損失から始め、まず ∂L/∂ŷ を求める",
  calcSelfK: "L の L 自身に対する勾配",
  calcSelf: "= 1（出発点）",
  calcMeanK: "意味",
  calcMean: "ŷ が 0.01 増えると L はおよそ {d} 変化する。負の符号＝ŷ は大きくなるべき",
  status5: "② 逆伝播：ŷ = v₁h₁ + v₂h₂ を通って 1 層戻る",
  status6: "② 逆伝播：シグモイドを通り抜け、その点での傾きを掛ける",
  calcSlopeK: "シグモイドの傾き",
  calcSlope: "σ′(z) = h × (1 − h)。最大でも 0.25",
  status7: "② 逆伝播の完了：6 個の重みの勾配がすべて求まった（表示中は {w}）",

  /* 連鎖律 */
  chainK: "連鎖律",
  chainTermsK: "各区間の局所的な傾き",
  chainSubK: "代入",
  chainCheckK: "数値による検算：{w} を ±0.0001 して実際に損失を計算し直す",
  chainUpdK: "更新",
  chainUpdSym: "{w} − η × 勾配",

  /* 損失の図 */
  lossLabel: "更新回数ごとの損失の変化",
  lossAxis: "更新回数（0〜{n}）",

  note: `∂L/∂w は「L の w に関する<span class="term">偏微分</span>（<span class="en">partial derivative</span>）」と読み、w だけをわずかに増やしたときに損失 L がどれだけ変わるか（傾き）を表す。勾配が負なら、w を大きくすると損失が小さくなる。例のネットワークは<span class="term">バイアス</span>（<span class="en">bias</span>）を省略しており、重みは 6 個だけである。実際のモデルには数百万個の重みがあり、値は訓練で学習したものだが、計算方法は同じである。損失に ½(ŷ − t)² を使うのは、微分の結果がちょうど ŷ − t になるように ½ を掛けているためである。`,
});
