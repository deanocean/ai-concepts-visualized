/* Depthwise Separable Convolution（concepts/dwsep.js）の日本語テキスト
   - である調。key は locales/zh/dwsep.js と一致させる
   - 注意：日本語の「行」は横（row）、「列」は縦（column）で、中国語と逆 */
Atlas.strings("ja", "dwsep", {
  lead: `<p>標準的な畳み込みでは、各フィルタが「空間方向の 3×3」と「すべてのチャンネル」を同時に処理する。チャンネルが増えると重みも計算量も急増し、スマートフォンでは動かせなくなる。<span class="term">Depthwise Separable Convolution</span>（<span class="en">depthwise separable convolution</span>、以下 DS）はこれを 2 段階に分ける。まず <span class="term">Depthwise Convolution</span>（<span class="en">depthwise convolution</span>）で各チャンネルが個別に空間方向を見て、次に 1×1 の <span class="term">Pointwise Convolution</span>（<span class="en">pointwise convolution</span>）でチャンネルを混ぜ合わせる。</p>`,
  closing: `分けることで、空間方向の処理（Depthwise）とチャンネル方向の処理（Pointwise）はそれぞれ 1 回で済み、出力チャンネルごとにやり直す必要がなくなる。計算量はおよそ元の 1/出力チャンネル数 + 1/(カーネルの一辺)² になり、3×3 なら約 8〜9 分の 1 に減る。代わりに、表現できるフィルタの種類は限られる。MobileNet はこれによってモデルを小さくしている。`,

  /* 例のデータの名前 */
  chVert: "縦線",
  chHorz: "横線",
  chDiag: "斜線",
  dVert: "縦方向のエッジ",
  dHorz: "横方向のエッジ",
  dCross: "十字の合計",

  /* 操作 */
  viewLabel: "表示する部分",
  viewStd: "① 標準の畳み込み（1 段階）",
  viewDs: "② Depthwise Separable（2 段階）",
  viewCalc: "③ パラメータ数の計算機",
  oLabel: "表示する出力チャンネル",
  outN: "出力 {o}",
  cinLabel: "入力チャンネル数",
  coutLabel: "出力チャンネル数",
  kLabel: "カーネルの一辺",
  hLabel: "特徴マップの一辺",
  preLabel: "クイック選択",
  preSmall: "このページの小さな例（3→4、出力 3×3）",
  preMobile: "MobileNet の中間層（256→256、14×14）",

  /* マス（読み上げ用） */
  cellAria: "{name}の {r} 行 {c} 列、値 {v}",
  nameIn: "入力チャンネル {c}",
  nameFilt: "フィルタの {c} 枚目",
  nameOut: "出力チャンネル {o}",
  nameDw: "Depthwise の結果、チャンネル {c}",
  nameDwF: "Depthwise のフィルタ {c}",

  /* 棒グラフ */
  barW: "重みの数",
  barM: "乗算回数",
  barStd: "標準の畳み込み",
  barDs: "DS",
  fMulStd: "{h}×{h} マス × {kkc} × {cout}",
  fMulDs: "{h}×{h} マス × ({kk}×{cin} + {cin}×{cout})",

  /* 共通 */
  capIn: "チャンネル {c}<br>{name}",

  /* ① 標準の畳み込み */
  capSlice: "{c} 枚目",
  labInStd: "入力 5×5×3 <span>（オレンジの枠＝出力のそのマスが見る範囲）</span>",
  labFilt: "出力 {o} のフィルタ 3×3×3 <span>（3 チャンネルを一度に見る。重み 27 個）</span>",
  labOut: "出力 3×3×4 <span>（出力チャンネルごとに、このようなフィルタが 1 つずつある）</span>",
  calcStdK: "出力 {o}、{r} 行 {c} 列",
  calcStd: `3 チャンネルそれぞれ 9 個の積（計 27 個）を合計：{expr} = <span class="hl">{v}</span>`,
  calcLayerK: "層全体",
  calcLayer: "出力 4 チャンネル × 9 マスで、各マスに 27 回の乗算が必要",
  statusStd: "標準の畳み込み：フィルタ 4 個 × 重み 27 個 = 108 個の重み。乗算は 9 マス × 4 × 27 = 972 回",
  hintStd: "出力のマスをクリックすると、別のマスを見られる。上のボタンで出力チャンネルを切り替える。",
  noteStd: "この標準の畳み込みの重みは、あえて「② の Pointwise の係数 × Depthwise のフィルタ」にしてある。そのため両者の出力は完全に一致し、計算量を比べやすい。通常、学習された標準の畳み込みの重みは任意の組み合わせをとれるが、DS が表現できるのはそのうちこの種類だけである。",

  /* ② Depthwise Separable */
  capDw: "チャンネル {c} の結果",
  capDwF: "チャンネル {c} 専用<br>{name}",
  pwHead: "1×1 の係数",
  pwCh: "チャンネル {c}",
  pwCap: "Pointwise の重み：4 組 × 3 個",
  labInDs: "入力 5×5×3",
  labDw: "① Depthwise：各チャンネルは自分専用の 3×3 だけを使う <span>（3 × 9 = 27 個の重み）</span>",
  labPw: "② Pointwise：1×1 の畳み込みで、マスごとに 3 チャンネルを重み付きで混ぜる <span>（4 × 3 = 12 個の重み）</span>",
  calcDwK: "① {r} 行 {c} 列",
  calcDwPart: "チャンネル {c}：9 個の積の合計 = {v}",
  calcDwSep: "、",
  calcPwK: "② 出力 {o} に混ぜる",
  calcCmpK: "① 標準の畳み込みとの比較",
  calcCmp: "出力は完全に一致（{res}）",
  cmpSame: "4 チャンネル 9 マスすべて同じ",
  cmpDiff: "不一致",
  statusDs: "2 段階の合計で重みは 27 + 12 = 39 個（標準は 108）。乗算は 9×3×9 + 9×4×3 = 351 回（標準は 972）",
  hintDs: "3×3 の結果のどれかのマスをクリックすると位置が変わる。上のボタンで出力チャンネルを切り替える。",
  noteDs: "Depthwise は空間だけを見てチャンネルを混ぜない。Pointwise はチャンネルだけを混ぜて空間を見ない。小さな例は出力チャンネルが 4 つしかないため、あまり減らない。チャンネルが多いほど大きく減るので、③ で試してみよう。",

  /* ③ 計算機 */
  calcStdRowK: "標準",
  calcDsRowK: "DS",
  calcWeights: "重み {expr} = {n}",
  calcRatioK: "比率",
  calcRatio: `DS ÷ 標準 = 1/{cout} + 1/{kk}² = {a} + {b} = <span class="hl">{r}</span>　<span class="k">（乗算回数の比率も同じ）</span>`,
  statusCalc: "DS は標準の畳み込みの {pct}% で済み、約 {x} 分の 1 になる",
  noteCalc: "乗算回数は、ストライド 1 で、パディングにより特徴マップの大きさが変わらないものとして計算している。出力チャンネルが多いと 1/出力チャンネル数 はほぼ 0 になり、比率は 1/(カーネルの一辺)² で決まる。3×3 なら約 1/9 である。",
});
