/* 位置エンコーディング（concepts/posenc.js）の日本語テキスト
   - である調。key は locales/zh/posenc.js と一致させる
   - words：3 語をプログラム内の dog・bite・man の順に半角スペース区切りで。sentA・sentB：2 つの文の表記
     （日本語は助詞で役割が決まり語順を入れ替えても意味が変わらないため、例文は英語にしている）
   - 注意：日本語の「行」は横（row）——ヒートマップの 1 行＝1 つの位置 */
Atlas.strings("ja", "posenc", {
  lead: `<p>Self-Attention は文全体を一度に見て各語を直接比べ合うので、RNN のように順番に 1 マスずつ読むわけではない。そのため「dog bites man」と「man bites dog」は、Self-Attention から見ると同じ語の詰め合わせになってしまう。<span class="term">位置エンコーディング</span>（<span class="en">positional encoding</span>）は「何番目か」も数値の組に変えて語のベクトルに足し、モデルが順序を区別できるようにする。</p>`,
  closing: `Transformer は周波数の異なる sin・cos を位置の番号として使う。前の次元ほど速く、後ろの次元ほどゆっくり変化し、時計の秒針・分針・時針のように組み合わさって、どの位置も唯一の値になる。しかも 2 つの位置の類似度は両者の距離だけで決まり、隣り合う位置ほどよく似ている。`,

  words: "dog bites man",
  sentA: "dog bites man",
  sentB: "man bites dog",

  /* 操作 */
  viewLabel: "表示する部分",
  viewWhy: "① なぜ必要か",
  viewWave: "② sin・cos の形",
  peLabel: "位置の情報",
  peOff: "加えない",
  peOn: "位置エンコーディングを加える",
  wordLabel: "比べる語",
  dLabel: "ベクトルの次元数 d",
  posLabel: "位置 pos",

  /* ① なぜ必要か */
  whyAria: "2 つの文における同じ語の Attention と出力",
  sentName: "文 {s}",
  posPct: "位置 {j}・{pct}",
  outOf: "「{w}」の出力",
  dimN: "第 {k} 次元",
  statusSameVerb: "2 つの文で「{w}」の出力がまったく同じ：モデルには誰が前にいて誰が後ろにいるのかわからない",
  statusSame: "2 つの文で「{w}」の出力がまったく同じ：モデルにはどちらがどちらを噛んだのか区別できない",
  statusDiff: "位置を加えると、2 つの文の「{w}」の出力が異なるようになった（差が最も大きいのは第 {k} 次元：{a} 対 {b}）",
  calcEmbK: "入力は語のベクトルだけ",
  calcEmbNote: "Attention の重みは「語と語が似ているか」だけで決まり、何番目に並んでいるかとは無関係である。順序を入れ替えても、同じ数値の並ぶ場所が変わるだけである。",
  calcPeK: "位置エンコーディング（d = 4）",
  calcSentK: "文 {s} の「{w}」",
  noteWhy: `読みやすさのため、ここではクエリ・キー・バリューに入力ベクトルをそのまま使い（重み行列を掛けない）、スコアは同じく内積 ÷ √d である（<a href="#selfatt">Self-Attention</a> を参照）。位置エンコーディングは語のベクトルに<b>足す</b>もので、別に後ろへつなげるわけではないため、次元数は変わらない。モデル全体の中での位置づけは <a href="#transformer">Transformer</a> を参照。`,

  /* ② sin・cos */
  waveAria: "位置エンコーディングのヒートマップ",
  waveTitle: "各行＝1 つの位置の {d} 次元ベクトル（青＝正、赤＝負）",
  posAxis: "位置↑",
  dimFast: "第 1 次元（速く変化）",
  dimSlow: "第 {d} 次元（ゆっくり変化）",
  simAria: "ほかの位置との類似度",
  simTitle: "位置 {p} と各位置の類似度（内積 ÷ (d/2)、自分自身＝1）",
  simAxis: "ほかの位置 →",
  statusWave: "位置 {p} と隣の位置 {near} の類似度は {sn}。25 離れた位置 {far} とは {sf} しかない",
  calcRuleK: "偶数次元は sin、奇数次元は cos：",
  calcRule: "PE(pos, 2i) = sin(pos ÷ 10000<sup>2i/d</sup>)、PE(pos, 2i+1) = cos(pos ÷ 10000<sup>2i/d</sup>)",
  calcDim12K: "第 1・2 次元（i = 0）",
  calcDim34K: "第 3・4 次元（i = 1）",
  calcLastK: "最後の 2 次元（i = {i}）",
  noteWave: `分母の 10000<sup>2i/d</sup> は次元が進むほど大きくなり、波長は約 6.3 位置分から数万位置分まで伸びていく。そのため前のほうの次元は秒針、後ろのほうの次元は時針のようにふるまう。原論文の d は 512 である。この固定の式は学習を必要としない。位置のベクトルをパラメータとして直接学習するモデル（たとえば BERT）もある。`,
});
