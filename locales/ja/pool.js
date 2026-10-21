/* プーリング（concepts/pooling.js）の日本語テキスト
   - である調。key は locales/zh/pool.js と一致させる
   - 注意：日本語の「行」は横（row）、「列」は縦（column）で、中国語と逆 */
Atlas.strings("ja", "pool", {
  lead: `<p>畳み込みで得られる<span class="term">特徴マップ</span>（<span class="en">feature map</span>）は大きく、物体が 1〜2 ピクセル動いただけでマップ全体の数値が変わってしまう。<span class="term">プーリング</span>（<span class="en">pooling</span>）は小さな区画ごとに値を 1 つにまとめる。データ量が減り、わずかな位置のずれも吸収される。</p>`,
  closing: `最大値プーリングは各区画で最も強い反応だけを残すため、特徴が区画内で少し動いても出力は変わらない。これが<span class="term">不変性</span>（<span class="en">invariance</span>）の獲得である。最大値や平均をとるだけなので、学習すべき重みは 1 つもない。`,

  /* 操作 */
  kindLabel: "プーリングの種類",
  kindMax: "最大値プーリング",
  kindAvg: "平均値プーリング",
  kindGap: "グローバルアベレージプーリング（GAP）",
  shortMax: "最大値",
  shortAvg: "平均値",
  moveGroup: "入力を移動",
  moveUp: "上へ 1 マス移動",
  moveDown: "下へ 1 マス移動",
  moveLeft: "左へ 1 マス移動",
  moveRight: "右へ 1 マス移動",
  moveHome: "元の位置に戻す",
  moveLabel: "入力全体を 1 マス動かす（物体のわずかな移動を再現、最大 ±1）",
  btnPrev: "◀ 前のマス",
  btnNext: "次のマス ▶",
  btnAll: "すべて計算",
  btnClear: "クリア",

  /* マス（読み上げ用） */
  cellAria: "{name}の {r} 行 {c} 列、値 {v}",
  cellEmpty: "未計算",
  nameIn: "入力",
  nameOut: "出力",
  nameRef: "移動しない場合の出力",

  /* 図の説明 */
  capIn: "<b>入力</b>（特徴マップ）{n}×{n}",
  capInBlocks: "太線＝2×2 の区画",
  capInMoved: "移動：{dirs}",
  dirLeft: "左へ 1 マス",
  dirRight: "右へ 1 マス",
  dirUp: "上へ 1 マス",
  dirDown: "下へ 1 マス",
  dirSep: "、",
  opGap: "全体の平均",
  opWin: "2×2 の{short}",
  capOut: "<b>出力</b> {size}",
  sizeOne: "値 1 つ",
  capOutChanged: "赤枠＝移動しない場合と異なる",
  capOutPick: "マスをクリックするとそこまで進む",
  capRef: "移動しない場合の出力<br>（比較用）",

  /* ステータス */
  statusIdle: "36 個の値が 9 個になる。計算し終えたら矢印ボタンで入力を動かし、出力が何マス変わるか見てみよう",
  statusGapIdle: "GAP は特徴マップ全体を平均して 1 つの値にする。入力を動かしてみよう",
  statusMoved: "入力の {inChg} マスの値が変わり、出力 9 マスのうち {outChg} マスが変化した。{extra}",
  statusGapMoved: "入力の {inChg} マスの値が変わったが、全体の平均 {avg} はまったく変わらない。特徴が画像内にある限り、位置はどこでも同じである",
  extraMaxKeep: "最も強い反応 9 は出力の同じマスに残っている",
  extraMaxMove: "最も強い反応 9 が区画の境界を越え、別のマスに移った",
  extraAvg: "平均は出入りした小さな値もすべて計算に入るため、変化が多くなる",

  /* 計算式 */
  calcStartK: "「次のマス」を押す",
  calcStart: "オレンジの枠が左上の 2×2 を覆い、その{short}を出力の最初のマスに書き込む。次は右へ 2 マス進む（ストライド 2）",
  calcStartGap: "36 マスすべてを合計して 36 で割る",
  calcGap: `36 マスの合計 {sum} ÷ 36 = <span class="hl">{avg}</span>`,
  calcWhyK: "よく使われる理由",
  calcWhy: "最終層の特徴マップが 7×7・512 チャンネルで 1000 クラスに分類する場合、平坦化して全結合層につなぐと 7×7×512×1000 = {big} 個の重みが必要になる。先に GAP で 512 個の値にすれば 512×1000 = {small} 個で済む",
  calcCellK: "出力の {r} 行 {c} 列",
  calcSizeK: "サイズ",
  calcSize: "一辺 (6 − 2) ÷ ストライド 2 + 1 = 3、個数 36 → 9（1/4 になる）",

  note: `例の 6×6 は、あるフィルタが算出した特徴マップである（値が大きいほど、そのフィルタが探す形に近い）。窓の大きさと<span class="term">ストライド</span>（<span class="en">stride</span>）は通常同じで、区画どうしは重ならない。完全に吸収されるのは同じ 2×2 区画内での移動だけで、区画の境界を越えると出力が変わる。畳み込み＋プーリングを何層も重ねると、許容できるずれは層ごとに大きくなる。GAP は各特徴マップを平均して 1 つの値にし、最後の<span class="term">全結合層</span>（<span class="en">fully connected layer</span>）の代わりによく使われる。畳み込みそのものは <a href="#conv">畳み込み</a> を参照。`,
});
