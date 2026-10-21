/* 受容野と Dilated Convolution（concepts/dilated.js）の日本語テキスト
   - である調。key は locales/zh/dilated.js と一致させる */
Atlas.strings("ja", "dilated", {
  lead: `<p>ある画素が何に属するか（たとえば路面か車か）を判断するには、その周囲の十分に広い範囲を見る必要がある。1 つの出力が見ている入力の範囲を<span class="term">受容野</span>（<span class="en">receptive field</span>）という。通常の 3×3 畳み込みは 1 層重ねるごとに 2 マスずつしか広がらない。<span class="term">Dilated Convolution</span>（<span class="en">dilated convolution</span>）はフィルタが値を取り出す位置に隙間をあけ、重みを増やさずにより遠くまで見る。</p>`,
  closing: `層ごとに間隔を倍にすると（1, 2, 4, 8…）、受容野は層数に対して指数的に広がる。重みの数は通常の畳み込みとまったく同じで、プーリングのように特徴マップを縮小して細部を失うこともない。そのため、画素ごとに判断する<span class="term">セマンティックセグメンテーション</span>（<span class="en">semantic segmentation</span>）の DeepLab や、長い音声区間を見る必要がある WaveNet でよく使われる。`,

  /* 操作 */
  viewLabel: "表示する部分",
  viewTower: "① 何層でどこまで見えるか（1 次元）",
  viewGrid: "② 2D の 3×3 の形",
  modeLabel: "畳み込みの種類",
  modeNormal: "通常の畳み込み（間隔 1）",
  modeDilated: "Dilated（間隔 1・2・4・8）",
  layersLabel: "層数",
  layersFmt: "{n} 層",
  rateLabel: `間隔 r（<span class="en">dilation rate</span>）`,
  btnPrev: "◀ 前へ",
  btnNext: "1 層下へたどる ▶",
  btnAll: "すべて表示",
  btnReset: "リセット",

  /* ① 1 次元 */
  towerAria: "多層の畳み込みで、1 つの出力が使う入力の範囲",
  layerIn: "入力",
  layerK: "第 {l} 層",
  rateK: "間隔 {d}",
  pickAria: "第 {l} 層の {i} マス目を選ぶ",
  rfLabel: "受容野 {n} マス",
  statusStart: "オレンジの点が選んだ出力である。「1 層下へたどる」を押し、下の層のどの 3 マスを使うか見てみよう",
  statusTraced: "第 {l} 層までたどった：{n} マスを使う",
  modeNameNormal: "通常の畳み込み",
  modeNameDilated: "Dilated Convolution",
  statusDone: "{mode}を {layers} 層重ねると、出力 1 マスが見る入力は {rf} マス{clip}。{other}で同じ {layers} 層なら {rfO} マスである",
  statusClip: "（端で切れている。本来は {ideal} マス）",
  hintTower: "最上段のマスをクリックすると、別の出力をたどれる。",
  calcRfK: "受容野",
  calcRf: `1 + (カーネル 3 − 1) × ({ds}) = 1 + 2 × {sum} = <span class="hl">{rf}</span> マス`,
  calcWK: "重み",
  calcW: "各層 3 個 × {layers} 層 = <b>{w}</b> 個（どちらの畳み込みも同じ）",
  calcNeedK: "通常の畳み込みで同じ範囲を見るには",
  calcNeed: "({rf} − 1) ÷ 2 = <b>{n}</b> 層重ねる必要がある",
  noteTower: `通常の畳み込みの受容野は 1 層につき 2 マスずつしか増えない（線形）。間隔を層ごとに倍にすると、受容野は 2<sup>層数+1</sup> − 1 になる（指数的）。受容野を広げるもう 1 つの方法はプーリングだが、特徴マップを縮小して位置の細部を失うため、画素ごとの分類には向かない。例は 1 次元・各層 3 個の重みで、入力の外側は 0 で埋める（パディング）ものとするため、各層の長さは変わらない。間隔をあけて値を取り出す規則は ② の 2D 図を参照。`,

  /* ② 2 次元 */
  gridAria: "間隔 {r} の 3×3 カーネルが値を取り出す位置",
  statusGrid: "間隔 {r}：9 個の重みが離れた位置から値を取り出し、{span}×{span} = {area} マスの範囲を覆う",
  calcCoverK: "覆う範囲",
  calcCover: `一辺 = 2 × 間隔 {r} + 1 = {span}、面積 {span}×{span} = <span class="hl">{area}</span> マス`,
  calcW2: "3×3 = <b>9</b> 個のままである。間の空いたマスは計算に加わらない（{span}×{span} で空きマスを 0 に固定したフィルタと同じだが、その 0 は学習しない）",
  calcPlainK: "通常の 3×3",
  calcPlain: "間隔 1 の特別な場合であり、3×3 = 9 マスを覆う",
  noteGrid: `Dilated Convolution は <span class="term">Atrous Convolution</span>（<span class="en">atrous convolution</span>）とも呼ばれる。atrous はフランス語の à trous（穴のあいた）に由来し、フィルタの間の隙間を指す。2 つの名前は同じ畳み込みを指し、シラバスには両方が載っている。`,
});
