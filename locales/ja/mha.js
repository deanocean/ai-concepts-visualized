/* Multi-Head Attention（concepts/mha.js）の日本語テキスト
   - である調。key は locales/zh/mha.js と一致させる
   - dims：4 つの次元の名前（半角スペース区切り、Self-Attention のページと同じ）。dimSep：それらをつなぐ区切り記号 */
Atlas.strings("ja", "mha", {
  lead: `<p>Self-Attention では各語が配分する Attention の重みは<b>1 組</b>だけだが、1 つの語には同時にいくつもの関係が必要になることが多い。「it」は自分が誰を指すか（cat）も、それがどうなったか（tired）も知る必要がある。合計 100% の重み 1 組では、1 回しか配分できない。<span class="term">Multi-Head Attention</span>（<span class="en">multi-head attention</span>）は Q・K・V を何組も用意し、1 組ずつを<span class="term">ヘッド</span>（<span class="en">head</span>）と呼んで、それぞれ別々に Attention を計算し、最後にまとめる。</p>`,
  closing: `各ヘッドは自分の小さな空間で 1 種類の関係を探すので、ほかの関係と同じ 100% を取り合わずに済む。各ヘッドの結果を連結し、さらに W<sub>O</sub> を掛けて混ぜると、出力は複数の関係の情報を同時に持つ。Transformer の原論文では 8 個のヘッドを使い、512 次元を 8 つに分けて各ヘッド 64 次元とするため、全体の計算量はヘッド 1 つの場合とほぼ同じである。`,

  dims: "生物 代名詞 動作 状態",
  dimSep: "／",
  headN: "ヘッド {n}",
  role1: "誰を指すかを探す",
  role2: "どうなったかを探す",
  headLabel: "{name}：{role}",
  headRow: "{name}（{role}）",

  /* 操作 */
  modeLabel: "Attention の組数",
  modeMulti: "ヘッド 2 個（Multi-Head）",
  modeSingle: "ヘッド 1 個だけ（同じパラメータ）",
  toksAria: "クエリにする語を選ぶ",
  toksLabel: "語をクリックし、その語の注意がどう配分されるかを見てみよう",
  btnMat: "各ヘッドの行列を見る",
  btnMatHide: "行列を閉じる",

  /* 図とステータス */
  svgAria: "各ヘッドにおける「{w}」の Attention",
  onlyHead: "ただ 1 つのヘッド",
  onlyHeadRow: "唯一のヘッド",
  sayFocus: "は「{top}」に最も注目（{pct}）",
  sayEven: "はほぼ均等（最大は「{top}」{pct}）",
  statusMulti: "「{w}」：ヘッド 1 {h1}、ヘッド 2 {h2}——2 つのヘッドがそれぞれ自分の 100% を配分している",
  statusSingleSplit: "「{w}」：重みが 1 組しかないため、「{a}」（{pa}）と「{b}」（{pb}）が同じ 100% を分け合い、どちらも薄まっている",
  statusSingle: "「{w}」：重みは 1 組だけで、最大は「{top}」（{pct}）",

  /* 表 */
  wLabel: "「{w}」が各語に配分する注意",
  thSum: "合計",
  outLabel: "Attention の後の「{w}」のベクトル（4 マス＝{dims}）",
  rowOrig: "元のベクトル",
  rowMulti: "ヘッド 2 個（連結）",
  rowSingle: "ヘッド 1 個",
  rowCmpSingle: "比較：ヘッド 1 個",
  rowCmpMulti: "比較：ヘッド 2 個",

  /* 計算式 */
  calcHeadK: "{name}（キー・バリューは「{a}・{b}」の 2 次元だけ）",
  calcScore2: "スコア q·k ÷ √2：",
  calcHeadZ: "　→ softmax の後に v を重み付け：z{k} =",
  calcConcatK: "連結",
  calcWoK: "× W<sub>O</sub>（この例では何も変えない）",
  calcSingleK: "同じパラメータで、Q・K はどちらも 4 次元",
  calcScore4K: "スコア q·k ÷ √4",
  calcSingleZK: "softmax 1 回の後に v を重み付け",
  matWQ: "{name} の W_Q",
  matWK: "{name} の W_K（＝W_V）",

  note: `各ヘッドは自分の W<sub>Q</sub>・W<sub>K</sub>・W<sub>V</sub> で 4 次元ベクトルを 2 次元に射影する。ヘッド 1 のキーには「生物・代名詞」だけ、ヘッド 2 には「動作・状態」だけが残るため、2 つのヘッドは自然と別の語に目を向ける。「ヘッド 1 個だけ」は、同じ行列を並べて 4 次元にし、1 つの softmax に入れたものである。これらの行列は説明用に設定したもので（W<sub>V</sub> は W<sub>K</sub> と同じ、W<sub>O</sub> は何も変えないもの）、実際のモデルではすべて学習で決まり、各ヘッドがどんな関係を学ぶかも人が指定するわけではない。ヘッド 1 つの計算は <a href="#selfatt">Self-Attention</a>、モデル全体における Multi-Head Attention の位置は <a href="#transformer">Transformer</a> を参照。`,
});
