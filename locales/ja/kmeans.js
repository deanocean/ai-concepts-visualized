/* k-means 法（concepts/kmeans.js）の日本語テキスト
   - である調。key は locales/zh/kmeans.js と一致させる
   - データ点の読み上げは pointAria + pointGroup + pointChanged をこの順につなげる（それぞれ「、」で始まる項目） */
Atlas.strings("ja", "kmeans", {
  lead: `<p>手元にあるのはデータだけで正解ラベルはない。それでも似たデータを自動でいくつかのグループに分けたい。これを<span class="term">クラスタリング</span>（<span class="en">clustering</span>）といい、たとえば購買履歴から顧客をグループ分けするのに使う。<span class="term">k-means 法</span>（<span class="en">k-means clustering</span>）は、まずクラスタ数 k を決めて k 個の中心を適当に置き、あとは 2 つの簡単な操作を変化がなくなるまで繰り返すだけである。</p>`,
  closing: `k-means 法がするのは 2 つだけである。各点を最も近い中心に割り当てることと、各中心をメンバーの平均の位置へ動かすこと。どちらの操作でもクラスタ内の距離の二乗和は小さくなるので、必ずどこかで止まる。ただし止まる場所は初期の中心しだいで（よくない結果で止まることもある）、k も人が決める必要がある。k を変えた結果を比べることが判断の助けになる。`,

  /* 操作 */
  kLabel: "クラスタ数 k",
  kFmt: "{k} 個",
  initLabel: "初期の中心",
  initRand: "ランダムに k 点を選ぶ",
  initBad: "わざと悪い位置（k = 3）",
  btnSeed: "別の乱数で初期化",
  btnPrev: "◀ 前のステップ",
  btnNext: "次のステップ ▶",
  btnEnd: "収束まで進める",
  btnReset: "ステップ 0 に戻る",

  /* 図 */
  mainAria: "データ点、各クラスタの中心と割り当て",
  histAria: "各ステップのクラスタ内誤差平方和",
  elbowAria: "クラスタ数ごとの収束後のクラスタ内誤差平方和",
  mainCap: "◆＝中心（ステップ 0 ではドラッグできる）。データ点をクリックすると、その点の計算を表示",
  histCap: "各ステップ後の SSE（クラスタ内誤差平方和。○ 割り当て後、● 中心の移動後）",
  elbowCap: "クラスタ数 k と収束後の SSE（k ごとに 20 通りの初期値で最良のもの）",
  histX: "ステップ",
  elbowX: "クラスタ数 k",
  elbowHint: "← ここから減り方が緩やかになる",
  pointAria: "データ点 ({x}, {y})",
  pointGroup: "、クラスタ {g}",
  pointChanged: "、このステップで所属が変わった",
  centerAria: "{j} 番目の中心",
  centerAriaDrag: "{j} 番目の中心、ドラッグできる",

  /* ステータス */
  statusInitBad: "ステップ {n}：初期の中心をわざと左下のかたまりに寄せてある（1 番と 2 番）。「次のステップ」を押して始めよう",
  statusInitDrag: "ステップ {n}：ドラッグした位置を初期の中心にする。「次のステップ」を押して始めよう",
  statusInitRand: "ステップ {n}：データ点を {k} 個ランダムに選び、初期の中心にした。「次のステップ」を押して始めよう",
  statusAssign1: "ステップ {n}：割り当て。各点を最も近い中心のクラスタに入れる",
  statusAssign: "ステップ {n}：割り当て。{nch} 個の点の所属が変わった（太い黒枠）",
  statusUpdate: "ステップ {n}：中心の移動。各中心をメンバーの平均の位置へ動かす（最大の移動量 {move}）",
  statusStuck: "ステップ {n}：収束した（所属の変わる点がない）が、SSE = {sse} で、k = {k} で到達できる {best} の {ratio} 倍ある。よくない結果で止まっている",
  statusDone: "ステップ {n}：収束。所属の変わる点がなく、中心ももう動かない。SSE = {sse}",

  /* 計算式 */
  calcNextK: "次のステップ",
  calcNext: "選んだ点 {p} から各中心までの距離を計算し、最も近い中心に割り当てる",
  calcCenterK: "中心 {j} の新しい位置",
  calcCenter: `選んだ点が属するクラスタ {j} には {n} 個の点があり、平均 = ({sx} ÷ {n}, {sy} ÷ {n}) = <span class="hl">({cx}, {cy})</span>`,
  calcSseK: "SSE",
  calcSseUpdate: "各点から自分の中心までの距離² の合計 = {sse}（移動前 {prev}）",
  calcSseAssign: "各点から自分の中心までの距離² の合計 = {sse}（割り当て前 {prev}）",
  calcSse: "各点から自分の中心までの距離² の合計 = {sse}",
  calcDistK: "点 {p} から各中心までの距離²",
  calcDistItem: "中心 {j}：{v}",
  calcDistTo: "→ 中心 {b} に割り当て",

  note: `中心の位置はメンバーの平均（mean）であり、名前の「means」はここから来ている。k はクラスタの数である。SSE（クラスタ内誤差平方和）は「割り当て」と「中心の移動」のどちらでも小さくなる一方なので必ず止まるが、止まった場所が最良とは限らない。「わざと悪い位置」を試すと、2 つの中心が同じかたまりに入り、最終的に右と上の 2 つのかたまりが 1 つのクラスタにまとめられてしまう。実務では初期値を変えて何度か実行したり、k-means++（初期の中心どうしを離して選ぶ）を使ったりする。クラスタ数を増やせば SSE は必ず小さくなる（全点が 1 つずつのクラスタなら 0）ので、SSE だけでは k を選べない。右下の図で減り方が急に緩やかになる点（ひじのような形）がよく目安にされ、これを<span class="term">エルボー法</span>（<span class="en">elbow method</span>）という。クラスタ数を事前に決めず、近いものから順にまとめて<span class="term">デンドログラム</span>（<span class="en">dendrogram</span>）を作る方法には<span class="term">ウォード法</span>（<span class="en">Ward's method</span>）などがある。もう 1 つの代表的な<span class="term">教師なし学習</span>（<span class="en">unsupervised learning</span>）である次元削減は<a href="#pca">主成分分析</a>を参照。`,
});
