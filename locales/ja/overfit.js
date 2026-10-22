/* 過学習と正則化（concepts/overfitting.js）の日本語テキスト
   - である調。key は locales/zh/overfit.js と一致させる
   - note は noteNone／noteL2／noteL1 のどれか 1 つに noteCommon を続けたもの（それぞれ完結した文） */
Atlas.strings("ja", "overfit", {
  lead: `<p>モデルの目的は<b>見たことのないデータ</b>を予測することだが、訓練中に見られるのは手元のデータだけである。モデルが柔軟すぎると、データに含まれるノイズまで一緒に覚えてしまう。訓練データはほぼ全問正解なのに、新しいデータでは大きく外れる。これを<span class="term">過学習</span>（<span class="en">overfitting</span>）という。<span class="term">正則化</span>（<span class="en">regularization</span>）は損失に「重みが大きすぎると減点」という項を加え、すべての点に合わせるために重みを大きくすることを防ぐ。</p>`,
  closing: `訓練誤差が下がり続けても、モデルが良くなったとは限らない。モデルが学習していない検証データを見て、その誤差が最も小さいところが<span class="term">汎化性能</span>（<span class="en">generalization performance</span>）の最も高いところである。L2 正則化はすべての重みをまとめて小さくし、曲線を滑らかにする。L1 正則化は重要でない重みをちょうど 0 にするので、使う特徴量を自動的に選ぶことになる。`,

  /* 操作 */
  degLabel: "多項式の次数（モデルの柔軟さ）",
  degFmt: "{n} 次",
  regLabel: "正則化",
  regNone: "なし",
  regL2: "L2（リッジ回帰）",
  regL1: "L1（ラッソ回帰）",
  lamLabel: "正則化の強さ λ",
  btnTruthShow: "真の規則を表示",
  btnTruthHide: "真の規則を隠す",
  presetLabel: "クイック選択",
  preSimple: "単純すぎ（1 次）",
  preRight: "ちょうどよい（3 次）",
  preMemo: "丸暗記（9 次）",
  preL2: "9 次＋L2",
  preL1: "9 次＋L1",
  coefLabel: "各項の係数（重み）",
  curveLabel: "次数 vs. 誤差（現在の正則化の設定）",

  /* メインの図 */
  svgAria: "訓練データ、検証データ、当てはめた曲線",
  lgTrain: "訓練データ（10 点、学習に使う）",
  lgValid: "検証データ（20 点、訓練中は見えない）",
  lgPred: "モデルの予測",
  lgTruth: "真の規則 y = sin 2πx",

  /* ステータス */
  sOver: "過学習：訓練誤差 {tr} とほぼぴったりなのに、検証誤差は {va} もある",
  sTooStrong: "正則化が強すぎる：訓練誤差 {tr}、検証誤差 {va}。訓練データにすら合っていない",
  sTooSimple: "単純すぎる：訓練誤差 {tr}、検証誤差 {va}。訓練データにすら合っていない",
  sGood: "訓練誤差 {tr}、検証誤差 {va}：見たことのないデータもうまく予測できている",

  /* 計算式 */
  calcModelK: "モデル",
  calcModel: "ŷ = w₀ + w₁u + w₂u² + … + w<sub>{d}</sub>u<sup>{d}</sup>（u = 2x − 1、x を −1〜1 に変換）",
  calcLossK: "最小化する損失",
  calcLossNone: "訓練データの平均二乗誤差 = {v}",
  calcLossReg: "平均二乗誤差 + λ × {pen}",
  calcNoPenW0: "（w₀ には罰則を課さない）",
  calcValidK: "検証データの平均二乗誤差",
  calcValidNote: "（訓練には使わず、確認にだけ使う）",
  calcNzK: "0 でない係数",
  calcNz: "{nz} / {d} 個",

  /* 係数の図・誤差の曲線 */
  coefAria: "各項の係数の大きさ",
  curveAria: "次数と訓練誤差・検証誤差",
  axisDeg: "次数",
  bestValid: "検証で最小",
  lgTrainErr: "訓練誤差",
  lgValidErr: "検証誤差（対数目盛）",

  noteNone: `次数が高いほど、曲線が曲がれる箇所が増える。9 次多項式には係数が 10 個あり、10 個の訓練点すべてをちょうど通れるので、訓練誤差は 0 になる。しかしノイズを含む点を通るために、曲線は点と点の間で激しく揺れ、係数も数百にまで大きくなる。単純なモデルで説明できるなら複雑なモデルは使わない、という原則を<span class="term">オッカムの剃刀</span>（<span class="en">Occam's razor</span>）という。`,
  noteL2: `<span class="term">L2 正則化</span>（<span class="en">L2 regularization</span>）の罰則は重みの 2 乗和である。大きな重みほど重く減点されるので、どの係数も 0 に向かって引き寄せられるが、ちょうど 0 になることはまれで、曲線は滑らかになる。線形回帰に使うと<span class="term">リッジ回帰</span>（<span class="en">ridge regression</span>）という。λ が大きすぎると必要な曲がりまで押しつぶされ、また単純すぎる状態に戻る。`,
  noteL1: `<span class="term">L1 正則化</span>（<span class="en">L1 regularization</span>）の罰則は重みの絶対値の和である。小さな重みはちょうど 0 に押しつぶされ（棒が灰色の 0 になる）、残ったものがモデルにとって重要な項になる。そのため特徴量の数を減らしたいときに向いている。線形回帰に使うと<span class="term">ラッソ回帰</span>（<span class="en">lasso regression</span>）という。「0 でない重みの個数」をそのまま罰則にするものを <span class="term">L0 正則化</span>（<span class="en">L0 regularization</span>）といい、最も直接的だが解くのが難しいため、計算できる代わりとして L1 がよく使われる。`,
  noteCommon: `　データは y = sin 2πx に乱数のノイズを加えて作った（乱数は固定なので毎回同じ）。L2 の解は公式で直接求め、L1 は座標降下法で繰り返し近づけている。ニューラルネットワークでよく使われる別の正則化は <a href="#dropout">ドロップアウト</a> を参照。ニューラルネットワークの訓練では、訓練しながら検証誤差を見て、それが上がり始めたら訓練を止めることが多い。これを<span class="term">早期終了</span>（<span class="en">early stopping</span>）という。また、訓練データを完全に覚えられるほどモデルが大きくなった後、さらに大きくすると、検証誤差が再び下がることがある。これを<span class="term">二重降下現象</span>（<span class="en">double descent</span>）といい、大規模なニューラルネットワークで観察されている（ここの多項式は最大 9 次なので、この部分は見えない）。分類モデルの評価指標は <a href="#roc">ROC 曲線と AUC</a> を参照。`,
});
