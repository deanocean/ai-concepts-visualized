/* Self-Attention（concepts/self-attention.js）の日本語テキスト
   - である調。key は locales/zh/selfatt.js と一致させる
   - dims：4 つの次元の名前（半角スペース区切り）。dimSep：説明文でそれらをつなぐ区切り記号 */
Atlas.strings("ja", "selfatt", {
  lead: `<p>「it」を読んだとき、それが誰を指すのかを知るには、文中のほかの語を振り返る必要がある。<span class="term">Self-Attention</span>（<span class="en">self-attention</span>）では、各語が「文中のそれぞれの語にどれだけ注目すべきか」を自分で計算し、その割合に従って、それらの語の情報を自分のベクトルに混ぜ込む。</p>`,
  closing: `各語は文中のすべての語に対して一度に Attention を計算する。RNN のように 1 マスずつ後ろへ渡す必要がないため、どれだけ離れていても直接つながり、しかも並列に計算できる。この仕組みを何層も積み重ねたものが Transformer の中核である。`,

  dims: "生物 代名詞 動作 状態",
  dimSep: "／",

  /* 上半分 */
  toksAria: "クエリにする語を選ぶ",
  toksLabel: "語をクリックし、その語が文中のどの語に注意を配分するかを見てみよう",
  diagAria: "「{w}」から各語への Attention",
  statusFocus: "「{w}」が最も注目しているのは「{top}」（{pct}）",
  statusEven: "「{w}」の注意はほぼ均等に分かれている（最大は「{top}」の {pct}）",
  heatLabel: "文全体の Attention の重みの表（各行＝その語が配分する注意、合計 100%。行をクリックすると切り替わる）",
  heatCorner: "注目する語 ↓ ／ 注目される語 →",
  note: `例のベクトルと行列は説明用に設定した小さな数値である。実際のモデルは数百次元で、数値はすべて学習で決まる。ここには語の順序が入っていない（語順を入れ替えても結果は変わらない）ため、<a href="#posenc">位置エンコーディング</a> で補う。複数組の重みで異なる関係を同時に見るのは <a href="#mha">Multi-Head Attention</a>、翻訳時に訳文から原文を振り返るのは <a href="#seq2seq">Source-Target Attention</a>、これらの部品で完全なモデルを組み立てたものは <a href="#transformer">Transformer</a> を参照。`,

  /* 計算手順 */
  btnShowSteps: "計算手順を表示（Q・K → softmax → 重み付き V）",
  btnHideSteps: "計算手順を閉じる",
  btnNext: "次へ ▶",
  step1: "① Q・K・V",
  step2: "② スコア q·k",
  step4: "④ 重み付き V",

  s1Intro: `各語はまず 4 次元ベクトルで、各次元が 1 つの性質を表す（{dims}）。これに 3 つの行列をそれぞれ掛けて、3 つの役割に変える：<span class="term">クエリ</span>（<span class="en">query</span>）q＝「自分は何を探しているか」、<span class="term">キー</span>（<span class="en">key</span>）k＝「自分は何者か（ほかの語との照合用）」、<span class="term">バリュー</span>（<span class="en">value</span>）v＝「選ばれたときに自分が渡す内容」。`,
  s1VecK: "「{w}」のベクトル",
  queryK: "クエリ",
  keyK: "キー",
  valueK: "バリュー",
  s1Note: `W_Q の意味：代名詞・動作・状態の語は「生物を探し」、生物の語は「動作と状態を探す」。読みやすさのため、この例では W_K と W_V をベクトルを変えないもの（k = v = x）にしているが、実際のモデルでは 3 つの行列すべてを学習する。`,
  thWord: "語",

  s2Intro: `「{w}」の q と各語の k の内積（対応する次元どうしを掛けて足す）をとる。性質がよく合うほどスコアが高くなる。さらに √d（d＝ベクトルの次元数 {d}、√{d} = {sq}）で割り、次元数が増えるとスコアが大きくなりすぎるのを防ぐ。`,
  thSeen: "注目される語",
  thQK: "q·k（q = {q}）",

  s3Intro: `スコアはそのままでは割合として使えない（負になることもあり、合計も 1 にならない）。<span class="term">ソフトマックス関数</span>（<span class="en">softmax function</span>）は各スコアについて e のべき乗をとり（正の数になり、高いスコアほど強調される）、合計で割って、足すとちょうど 100% になる Attention の重みを求める。`,
  thScore: "スコア s",
  thDiv: "e^s ÷ 合計",
  thWeight: "重み",
  total: "合計",

  s4Intro: `最後に各語の v に重みを掛けて足し合わせ、「{w}」の新しいベクトルを得る。よく注目された語ほど寄与が大きい。`,
  thWv: "重み×v {dim}",
  sum: "合計",
  cmpBefore: "元のベクトル",
  cmpAfter: "Attention の後",
  s4Mixed: "「{w}」の「{dim}」の次元は {a} から {b} に変わった：このベクトルに「{top}」の情報が混ざり込んでいる。",
  s4Even: "「{w}」の新しいベクトルは各語の内容を平均的に混ぜたもので、特定の語に偏っていない。",
});
