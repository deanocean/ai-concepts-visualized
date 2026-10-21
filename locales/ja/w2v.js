/* 単語埋め込み（concepts/w2v.js）の日本語テキスト
   - である調。key は locales/zh/w2v.js と一致させる
   - corpus：学習用コーパス。文は「/」、語は半角スペースで区切る。日本語の語順で、助詞はほぼ省いた短文にしている
     （コーパスを変えると学習結果も変わるので、closing の記述を確認し直すこと。いまの文は種子 1〜5 で確認済み）
   - keyWords：プログラムが使う 9 語。順に 王、王妃、男、女、猫、犬、魚、肉、水（すべて corpus に含まれていること） */
Atlas.strings("ja", "w2v", {
  lead: `<p>コンピュータで文字を扱うには、まず語を数値に変える必要がある。最も素朴なのは<span class="term">ワンホットベクトル</span>（<span class="en">one-hot vector</span>）で、語彙の数だけ次元を用意し、その語のマスだけを 1、ほかを 0 にする。しかしこれではどの 2 語も同じだけ「似ていない」ことになり、「王」が「水」より「王妃」に近いことがわからない。<span class="term">単語埋め込み</span>（<span class="en">word embedding</span>）は数次元の実数ベクトルで 1 語を表す方法で、これを<span class="term">分散表現</span>（<span class="en">distributed representation</span>）という。<span class="term">word2vec</span> の発想は、<b>似た文脈に現れる語は意味も近い</b>というものである。そこでネットワークに「ある語から周りの語を当てる」練習をさせると、ベクトルにひとりでに意味が宿る。</p>`,
  closing: `学習でやっているのは「周りの語を当てる」練習だけだが、文脈の似た語は近い位置に押しやられる。スキップグラムで学習すると、どの初期値でも王と王妃、猫と犬、男と女、魚と肉が互いに最も近い語どうしになる。「王 − 男 + 女」のようなベクトルの足し引きさえ王妃の近くに落ちる（どちらの学習方法でも）。「男→女」の違いが、異なる語の上でほぼ同じ方向として学習されるからである。CBOW はこれほど小さなコーパスでは不安定で、最も近い相手がずれる語がある。`,

  corpus: "彼 は 王 / 彼女 は 王妃 / 彼 は 男 / 彼女 は 女 / 王 城 住む / 王妃 城 住む / 男 村 住む / 女 村 住む / 王 王冠 かぶる / 王妃 王冠 かぶる / 男 畑 耕す / 女 畑 耕す / 猫 魚 食べる / 犬 肉 食べる / 猫 水 飲む / 犬 水 飲む / 猫 家 で 寝る / 犬 家 で 寝る / 私 猫 飼う / 私 犬 飼う",
  keyWords: "王 王妃 男 女 猫 犬 魚 肉 水",
  listSep: "、",
  quoted: "「{w}」",

  /* 操作 */
  modeLabel: "学習方法",
  modeSg: "スキップグラム（中心 → 周り）",
  modeCbow: "CBOW（周り → 中心）",
  btnSeed: "初期値を変える",
  btnStart: "▶ 学習開始",
  btnContinue: "▶ 学習を続ける",
  btnPause: "❚❚ 一時停止",
  btnDone: "✓ 学習完了",
  btnStep: "1 エポック進める",
  btnFF: "⏭ 最後まで早送り",

  /* コーパスと問題 */
  corpusLabel: "コーパス（{s} 文、{v} 語。助詞はほぼ省略）：語をクリックし、学習でどんな問題になるかを見てみよう（ウィンドウ＝前後 2 語ずつ）",
  sampleSgK: "スキップグラムの問題",
  sampleSg: "「{w}」のベクトルを入力 → ウィンドウ内の {n} 語を当てる（1 語ごとに誤差を 1 回計算）",
  sampleCbowK: "CBOW の問題",
  sampleCbow: "ウィンドウ内の「{ctx}」のベクトルを平均 → 中心の「{w}」を当てる",
  modelK: "現在のモデル",
  randomNote: "（当てずっぽうなら 1/{v} ≈ {p}%）",

  /* 図 */
  mapAria: "平面に射影した単語ベクトル",
  mapCap: "8 次元の単語ベクトルを平面に射影（語をクリックすると最も近い語を表示）",
  groupP: "人",
  groupA: "動物",
  groupF: "食べ物・飲み物",
  groupO: "その他",
  lossAria: "誤差",
  lossCap: "平均誤差（周りの語を当てる確率の −ln）",
  epochAxis: "エポック",

  /* ステータス */
  statusStart: "まだ学習していない：単語ベクトルはランダムな小さい数値で、平面上の位置に意味はない。平均誤差は {loss}（当てずっぽうの ln {v} ≈ {ln} とほぼ同じ）。「▶ 学習開始」を押して {ep} エポック（約 3 秒）学習させてみよう",
  statusEp: "第 {ep} エポック：平均誤差 {loss}。{found}",
  statusEpRunning: "第 {ep} エポック（学習中）：平均誤差 {loss}。{found}",
  statusPairs: "互いに最も近い語どうし：{list}",
  statusNoPairs: "互いに最も近い語のペアはまだない",
  statusMiss: "。{list}はまだペアになっていない（{why}）",
  whyCbow: "CBOW はこれほど小さなコーパスでは不安定である",
  whySeed: "初期値を変えて比べてみよう",

  /* 最も近い語と類推 */
  nbLabel: `「{w}」に最も近い語（<span class="term">コサイン類似度</span>（<span class="en">cosine similarity</span>）：1＝向きが完全に同じ、0＝無関係）`,
  nbVecK: "ベクトル",
  nbTopK: "最も近い語",
  oneHot: "ワンホットベクトルなら、「{w}」とほかのどの語のコサイン類似度も 0 である",
  anaLabel: "ベクトルの足し引き（類推）",
  anaOpt: "{a} − {b} + {c}",
  anaRowK: "「{a}」−「{b}」+「{c}」",
  anaTopK: "最も近い語（問題の 3 語を除く）",
  anaJudgeK: "判定",
  anaUntrained: "まだ学習していないので、結果に意味はない",
  anaOk: "正解：最も近いのは「{w}」",
  anaNot: "いま最も近いのは「{got}」で、まだ「{w}」ではない",

  note: `このページのコーパスと次元数（8 次元）は意図的にとても小さくしている。実際の word2vec は数十億語のコーパスと 100〜300 次元を使い、計算を減らすためにすべての語で softmax を計算せず、<span class="term">ネガティブサンプリング</span>（<span class="en">negative sampling</span>）などの近似を用いる。平面図は 8 次元を第 1・第 2 主成分に射影したもので（<a href="#pca">主成分分析</a>を参照）、距離はあくまで近似であり、類似度はすべて元の 8 次元で計算している。<span class="term">スキップグラム</span>（<span class="en">skip-gram</span>）は小さなコーパスや稀な語に強いことが多く、<span class="term">CBOW</span>（<span class="en">continuous bag-of-words</span>）は計算が速い。fastText は語をさらに小さな文字単位の断片に分けて学習するため、未知の語でもベクトルを組み立てられる。word2vec では 1 語に 1 つのベクトルしかないが、ELMo や BERT は文脈に応じて同じ語に異なるベクトルを与える。文書全体を「各語の出現回数」とみなす BoW（<span class="en">bag-of-words</span>）や、それに重みを付けた TF-IDF は、語の意味の近さを考慮しない、より古い表現方法である。`,
});
