/* 敵対的生成ネットワーク（concepts/gan.js）の日本語テキスト
   - である調。key は locales/zh/gan.js と一致させる */
Atlas.strings("ja", "gan", {
  lead: `<p>コンピュータに「本物らしい」データを作らせたいが、「本物らしさ」の採点基準を直接書くのは難しい。<span class="term">敵対的生成ネットワーク</span>（<span class="en">generative adversarial network, GAN</span>）は 2 つのネットワークを競わせる。<span class="term">生成器</span>（<span class="en">generator</span>）は乱数を偽データに変え、<span class="term">識別器</span>（<span class="en">discriminator</span>）はデータが本物か偽物かを判定する。識別器が見抜くのがうまくなるほど、生成器はより本物らしく作らざるを得なくなる。</p>`,
  closing: `識別器は本物と偽物の見分け方を学び、生成器は識別器の判断に沿って偽データを「より本物らしい」方向へ動かす。両者を交互に更新し、理想的な終着点は識別器がどこでも 0.5 としか言えなくなる状態である。両者の力は拮抗していなければならない。生成器を一度に何ステップも進めると、すべての点を識別器がその時点で最も本物だと信じる場所に集めてしまい、まとめて追い払われやすい。これが<span class="term">モード崩壊</span>（<span class="en">mode collapse</span>）である。`,

  /* 操作 */
  dLabel: "本物のデータ",
  dCircle: "円環",
  dRing8: "8 つの塊",
  sLabel: "更新のしかた",
  sStd: "交互に 1 回ずつ",
  sG10: "生成器 10 回、識別器 1 回",
  btnSeed: "初期値を変える",
  btnRun: "▶ 訓練開始",
  btnResume: "▶ 訓練再開",
  btnPause: "❚❚ 一時停止",
  btnDone: "✓ 訓練完了",
  btnOne: "1 ラウンド進める",
  btnFf: "⏭ 最後まで早送り",
  progLabel: "何ラウンド目の状態を見るか",

  /* 図（読み上げ用） */
  planeAria: "平面：本物のデータ、生成データ、識別器の判断",
  dAria: "識別器が本物のデータと生成データに与えた平均確率",
  covAria: "生成した点が 8 方向それぞれに落ちた数",

  /* 図の説明 */
  capPlane: "オレンジの四角をクリックすると、その計算が見られる",
  key: `<span><i style="background:var(--pos);border-radius:50%"></i>本物のデータ</span><span><i style="background:var(--hot)"></i>生成器が作った偽データ</span><span>→ 生成器が押される方向</span><span><i style="background:var(--pos-soft);border:1px solid var(--pos)"></i>識別器が「本物」と判断</span><span><i style="background:var(--hot-soft);border:1px solid var(--hot)"></i>「偽物」と判断</span>`,
  capCov: "1000 個の生成点がどの方向に落ちたか（濃い色＝本物のデータの近く。点線＝均等に分けた場合の 125 個）",
  dCapStd: `識別器がこのラウンドの本物 32 個・偽物 32 個に与えた平均確率（さらに 20 ラウンドごとに平均）：<span style="color:var(--pos)">━ 本物</span>　<span style="color:var(--hot)">━ 偽物</span>　点線 0.5`,
  dCapG10: `識別器がこのラウンドの本物 32 個・偽物 32 個に与えた平均確率：<span style="color:var(--pos)">━ 本物</span>　<span style="color:var(--hot)">━ 偽物</span>　点線 0.5`,
  dAxis: "ラウンド",
  dirR: "右",
  dirUR: "右上",
  dirU: "上",
  dirUL: "左上",
  dirL: "左",
  dirDL: "左下",
  dirD: "下",
  dirDR: "右下",
  covAxis: "方向（原点から見て）",

  /* ステータス */
  statusHead: "{r} ラウンド目：識別器の判断は本物に平均 {ar}、偽物に平均 {af}。生成点の {near}% が本物のデータの近くにあり、8 方向中 {cover} 方向に 5% 以上の点がある",
  statusHeadRun: "{r} ラウンド目（訓練中）：識別器の判断は本物に平均 {ar}、偽物に平均 {af}。生成点の {near}% が本物のデータの近くにあり、8 方向中 {cover} 方向に 5% 以上の点がある",
  tailStartRun: "。始まったばかりで、偽データと本物のデータの分布はまったく違う",
  tailStart: "。まだ訓練していないので、偽データと本物のデータの分布はまったく違う。「▶ 訓練開始」で連続して訓練するか、「1 ラウンド進める」で 1 ラウンドずつ見てみよう",
  tailCollapse: "。モード崩壊：点が少数の方向に集まっている",
  tailChase: "。生成点の大半が本物のデータの近くにない。生成器が識別器のその時点の判断を追いかけ続けている",
  tailOk: "。識別器は本物と偽物をほとんど見分けられず、偽データもすべての方向に広がっている",
  tailMiss: "。まだ埋まっていない方向がある。GAN の訓練結果は初期値に左右されるので、初期値を変えて比べてみよう",

  /* 計算式 */
  calcGK: "生成器",
  calcG: `ノイズ z = {z} → 生成器 → 偽データ x = <span class="hl">{x}</span>`,
  calcDK: "識別器",
  calcD: "D(x) = {q}（本物のデータだと判断する確率）",
  calcGLossK: "生成器の損失",
  calcGLoss: `−ln D(x) = −ln {q} = <span class="hl">{v}</span>。∂ln D/∂x = {g} の方向へ動くと損失が小さくなる（太い矢印）`,
  calcDLossK: "識別器の損失",
  calcDLoss: `−[ ln D(本物) の平均 + ln(1 − D(偽物)) の平均 ]（画面上の各 300 個）= <span class="hl">{v}</span>。まったく見分けられない（すべて 0.5）とき = 2 ln 2 ≈ 1.386`,

  note: `元の論文では生成器の目標を「ln(1 − D(G(z))) を小さくする」と書いているが、訓練の初期は識別器が簡単に正しく判定できるため、この項の勾配はほぼ 0 になる。そこで実務では（このページも含め）「−ln D(G(z)) を小さくする」を使う。生成器は本物のデータを一度も見ておらず、学ぶことはすべて識別器の勾配を通じて伝わってくる。このページは平面上の点だが、2 つのネットワークを<a href="#conv">畳み込み</a>層にして画像を生成するのが <span class="term">DCGAN</span>（<span class="en">deep convolutional GAN</span>）である。ある画像から対応する別の画像を生成する（例：線画 → 着色）とき、ペアになった訓練データがあれば <span class="term">Pix2Pix</span>、ペアのデータがなければ（例：馬 ↔ シマウマの写真）<span class="term">CycleGAN</span>（変換して元に戻すと元の画像になるようにする）を使う。GAN は音声生成や文章生成などにも使われる。2 つのネットワークを競わせる必要がなく、訓練がより安定した生成手法については <a href="#diffusion">拡散モデル</a> を参照。`,
});
