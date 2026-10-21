/* Seq2Seq と Source-Target Attention（concepts/seq2seq.js）の日本語テキスト
   - である調。key は locales/zh/seq2seq.js と一致させる
   - src：原文の 6 語（両言語共通の日本語の例文、半角スペース区切り）。dims：4 つの次元の名前（半角スペース区切り） */
Atlas.strings("ja", "seq2seq", {
  lead: `<p>翻訳では文全体を別の文全体に変換する必要があり、長さも語順も異なりうる。<span class="term">Seq2Seq</span>（<span class="en">sequence-to-sequence</span>）は、まず<span class="term">エンコーダ</span>（<span class="en">encoder</span>）で原文を読み終え、次に<span class="term">デコーダ</span>（<span class="en">decoder</span>）で訳文を 1 語ずつ生成する。元の方式ではエンコーダの最後の状態だけをデコーダに渡すため、文全体をこの 1 本の固定長ベクトルに詰め込まなければならない。</p>`,
  closing: `<span class="term">Source-Target Attention</span>（<span class="en">source-target attention</span>、<span class="en">encoder-decoder attention</span> とも呼ぶ）を加えると、デコーダは 1 語生成するたびに自分の状態をクエリとし、原文の各位置の状態（キー・バリュー）から必要な情報を探す。「I」は「私」を、「cats」は「猫」を見るので、語順が違っても対応がとれ、前のほうの語が押し出されることもない。`,

  src: "私 は 猫 が 好き です",
  dims: "人 動物 好み 文法",

  /* 操作 */
  modeLabel: "デコーダが受け取るもの",
  modePlain: "元の Seq2Seq（最後の状態だけ）",
  modeAtt: "Source-Target Attention を追加",
  btnPrev: "◀ 前へ",
  btnNext: "次へ ▶",
  btnJump: "デコードへ進む",
  btnReset: "リセット",

  /* 図 */
  svgAria: "エンコーダとデコーダ",
  encName: "エンコーダ",
  encSub: "原文を読む",
  decName: "デコーダ",
  decSub: "訳文を書く",
  onlyVec1: "唯一渡される",
  onlyVec2: "ベクトル = h₆",
  decGotH6: "受け取った h₆",
  legend: "色の帯の 4 マス：",

  /* ステータスと計算式 */
  statusStart: "① エンコード：「次へ」を押し、原文を 1 語ずつエンコーダに読み込ませてみよう",
  calcStart: "各語は 4 次元ベクトルで、4 マスは順に「人／動物／好み／文法（助詞など）」を表す。例：私 = [1, 0, 0, 0]、猫 = [0, 1, 0, 0]、好き = [0, 0, 1, 0]、は・が・です = [0, 0, 0, 1]。",
  statusEnc: "① エンコード：{n} 番目の語「{w}」を読み込む",
  calcEncRule: "新しい状態 = この語 + 0.5 × 1 つ前の状態",
  calcEncShareK: "展開すると、最後の状態に含まれる各語の割合は",
  statusDecAtt: "② デコード：「{tgt}」と言うには「{dim}」の情報が要る → Attention は「{src}」に集中し（{pct}）、{got} を受け取る",
  statusDecPlainLow: "② デコード：「{tgt}」と言うには「{dim}」の情報が要るが、h₆ には {got} しか残っていない：前のほうの語はほとんど押し出されている",
  statusDecPlainHigh: "② デコード：「{tgt}」と言うには「{dim}」の情報が要るが、h₆ には {got} しか残っていない：後で読んだ語のほうが多く残っている",
  calcQK: "クエリ＝このステップのデコーダの状態",
  calcKVK: "キー・バリュー＝エンコーダの各ステップの h",
  calcScoreK: "スコア s·h ÷ √{d}",
  calcSoftK: "softmax（e^スコア ÷ {tot}）",
  calcCtxK: "context = Σ 重み × h",
  calcSameK: "どのステップでも受け取るのは同じベクトル",
  calcCellK: "その「{dim}」のマス",

  /* 表 */
  thSay: "「{tgt}」と言うとき",
  thNeed: "{dim}（必要）",
  rowPlain: "元の方式：h₆ だけ",
  rowAtt: "Attention：context",
  cmpLabel: "2 つのやり方で受け取る情報（オレンジ枠＝この語に最も必要なマス）",
  heatCorner: "訳文 ↓ ／ 原文 →",
  heatLabel: "対応表：訳文の各語が原文のどの語に注意を配分しているか（各行の合計は 100%）",

  note: `読みやすさのため、エンコーダは「この語 + 0.5 × 1 つ前」という線形の RNN に簡略化し、デコーダの各ステップの状態 s は設定したデモ用の値を使っている（実際のモデルではデコーダの RNN が計算し、数値はすべて学習で決まる）。スコアは <a href="#selfatt">Self-Attention</a> と同じく内積を √d で割ったものである。違うのは出どころだけで、Self-Attention では<span class="term">クエリ</span>（<span class="en">query</span>）・<span class="term">キー</span>（<span class="en">key</span>）・<span class="term">バリュー</span>（<span class="en">value</span>）がすべて同じ文から来るのに対し、Source-Target Attention ではクエリが訳文（デコーダ）から、キー・バリューが原文（エンコーダ）から来る。デコーダの学習時に、次のステップの入力として自分の予測ではなく正解を使うことを<a href="#teacher">教師強制</a>（<span class="en">teacher forcing</span>）という。翻訳モデル全体を Attention だけで組み立てたものは <a href="#transformer">Transformer</a> を参照。`,
});
