/* スキップ結合（concepts/resnet.js）の日本語テキスト
   - である調。key は locales/zh/resnet.js と一致させる */
Atlas.strings("ja", "resnet", {
  lead: `<p>ネットワークを深くするほど、<span class="term">誤差逆伝播法</span>（<span class="en">backpropagation</span>）で勾配を掛け合わせる回数が増え、入力側の層がほとんど学習しなくなる。これが<span class="term">勾配消失</span>（<span class="en">vanishing gradient</span>）である。<span class="term">スキップ結合</span>（<span class="en">skip connection</span>）は層の横に近道を 1 本加え、入力をそのまま出力に足す。勾配もこの近道を通って、そのまま逆向きに伝わる。</p>`,
  closing: `近道があると、勾配が 1 層通るたびに掛かる値は「層の傾き」だけでなく「1 + 層の傾き」になる。層の傾きが小さくても、この 1 があるため、勾配が掛け算で 0 に近づくことはない。層自体は「入力に対してさらにどれだけ足すか」、すなわち<span class="term">残差</span>（<span class="en">residual</span>）だけを学習すればよい。ResNet はこの構造で 152 層まで深くしている。`,

  /* 操作 */
  nLabel: "層数",
  nFmt: "{n} 層",
  wLabel: "各層の重み w",
  btnPrev: "◀ 前へ",
  btnNext: "1 層戻す ▶",
  btnAll: "すべて計算",
  btnReset: "リセット",

  /* ネットワーク図 */
  svgAria: "近道のないネットワークとあるネットワークの、各層の出力と逆向きに伝わる勾配",
  lanePlain: "通常のネットワーク（近道なし）",
  laneRes: "スキップ結合あり（弧線＝近道：出力 = 層の結果 + 入力）",
  laneGrad: "勾配",
  laneIn: "入力",
  laneOut: "出力",
  layerK: "第 {k} 層",

  /* ステータスと計算式 */
  statusStart: "どちらのネットワークも順伝播は計算済みである（青枠＝各層の出力）。「1 層戻す」を押し、最後の層から勾配を同時に逆向きに伝えてみよう",
  calcPlainK: "通常",
  calcPlain: "第 k 層の出力 = σ(w × 前の層)　例：第 1 層 σ({w} × {x}) = {a}",
  calcResK: "スキップ結合あり",
  calcRes: "第 k 層の出力 = 前の層 + σ(w × 前の層)　例：第 1 層 {x} + {s} = {a}",
  statusFirst: "最後の層（第 {n} 層）から始め、両方の勾配を 1 とする",
  calcFirstK: "出発点",
  calcFirst: "最後の層が受け取る勾配を 1 とし、各層まで戻ったときにどれだけ残るかを見る",
  calcStepPK: "通常：第 {k1} → {k} 層で掛ける値",
  calcStepP: `w × σ′(z) = {w} × σ′({z}) = {w} × {d} = <span class="hl">{f}</span>　勾配 {g}`,
  calcStepRK: "スキップ結合あり：掛ける値",
  calcStepR: `1（近道） + w × σ′(z) = 1 + {w} × {d} = <span class="hl">{f}</span>　勾配 {g}`,
  statusStep: "第 {k} 層まで戻した：通常 {gp}、スキップ結合あり {gr}",
  statusDone: "第 1 層が受け取る勾配：通常のネットワーク {gp}{vanish}、スキップ結合あり {gr}{kept}",
  doneVanish: "（勾配消失）",
  doneKept: "で、保たれている",

  /* 棒グラフ */
  barAria: "各層が受け取る勾配：通常 vs スキップ結合あり（対数目盛）",
  barTitle: "各層が受け取る勾配（最後の層 = 1、対数目盛：1 目盛で 10 倍）",
  legPlain: "通常",
  legRes: "スキップ結合あり",
  barAxis: "← 層",

  note: `σ′ は Sigmoid の傾き σ(z)(1 − σ(z)) で、最大でも 0.25 しかない。そのため w = 1 のとき、通常のネットワークでは 1 層ごとに少なくとも 4 分の 1 に縮む。例のネットワークは各層にニューロンが 1 つしかない。実際の ResNet では 1 本の近道が 2〜3 層の畳み込み（間に<span class="term">バッチ正規化</span>（<span class="en">batch normalization</span>）と ReLU を挟む）をまたぐが、「勾配に 1 を掛ける道が 1 本増える」という原理は同じである。また実際の ResNet では、残差側の重みがすべて 0 に学習されると、ブロックは入力をそのまま次へ渡すことになる。そのため層を増やしても、少なくとも結果は悪化しにくい。同じネットワークで近道がない場合の詳細は <a href="#vanish">勾配消失問題</a> を参照。`,
});
