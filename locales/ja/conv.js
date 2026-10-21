/* 畳み込み（concepts/convolution.js）の日本語テキスト
   - である調。key は locales/zh/conv.js と一致させる
   - 注意：日本語の「行」は横（row）、「列」は縦（column）で、中国語と逆 */
Atlas.strings("ja", "conv", {
  lead: `<p>画像認識で鍵になるのは局所的な形（エッジ、角、線）であり、同じ形が画像のどこに現れてもおかしくない。<span class="term">畳み込み</span>（<span class="en">convolution</span>）は 3×3 の小さな数値の表、すなわち<span class="term">フィルタ</span>（<span class="en">filter</span>）を画像上で滑らせ、止まるたびに「ここがこの形にどれだけ似ているか」を計算する。</p>`,
  closing: `フィルタの数値が局所的な模様とよく一致するほど、掛けて足し合わせた値は大きくなり、特徴マップのそのマスの色が濃くなる。同じ 9 個の数値で画像全体を走査するため、パラメータが少なく、形がどこにあっても検出できる。CNN が学習するのは、まさにこれらの数値である。`,

  /* 操作 */
  imgLabel: "入力画像（マスをクリックすると塗り替えられる）",
  imgBar: "縦線",
  imgRing: "四角い枠",
  imgDiag: "斜線",
  imgEll: "L 字",
  imgBlank: "空白",
  filterLabel: "フィルタ",
  fVert: "縦方向のエッジ",
  fHorz: "横方向のエッジ",
  fDiag: "斜線（＼）",
  fBlur: "平均（ぼかし）",
  fCustom: "カスタム",
  strideLabel: `<span class="term">ストライド</span>（<span class="en">stride</span>）`,
  stride1: "1 マス",
  stride2: "2 マス",
  padLabel: `<span class="term">パディング</span>（<span class="en">padding</span>）`,
  pad0: "0（なし）",
  pad1: "1（周囲を 0 で埋める）",
  btnPrev: "◀ 前のマス",
  btnNext: "次のマス ▶",
  btnAll: "すべて計算",
  btnClear: "クリア",

  /* マス（読み上げ用） */
  cellIn: "入力の {r} 行 {c} 列、値 {v}",
  cellFilter: "フィルタの {r} 行 {c} 列、値 {v}。クリックで変更",
  cellOut: "特徴マップの {r} 行 {c} 列",

  /* 図の説明 */
  capIn: "<b>入力</b> {n}×{n}",
  capInPad: "（破線＝埋めた 0）",
  capFilter: "<b>フィルタ</b> 3×3<br>マスをクリックすると数値が変わる",
  capOut: `<b><span class="term">特徴マップ</span></b>（<span class="en">feature map</span>）{n}×{n}<br>マスをクリックするとそこまで進む`,

  /* 計算式 */
  calcStart: "「次のマス」を押す。フィルタが左上の 3×3 を覆い、対応するマスどうしを掛けて合計し、結果を特徴マップの最初のマスに書き込む。",
  calcCellK: "{r} 行 {c} 列：入力 × フィルタ（マスごとの積）",
  calcSumK: "合計",
  sizeK: "特徴マップの一辺",
  size: "= (入力 {n} + 2×パディング {p} − フィルタ {k}) ÷ ストライド {s} + 1 = ({a}) ÷ {s} + 1 = <b>{out}</b>",
  edgeHorz: "このフィルタは上の行が +1、下の行が −1 である。「上にあって下にない」エッジでは正の値（青）、「上になくて下にある」エッジでは負の値（赤）になる。エッジのない平らな所では正負が打ち消し合い、0 になる。",
  edgeVert: "このフィルタは左の列が +1、右の列が −1 である。「左にあって右にない」エッジでは正の値（青）、「左になくて右にある」エッジでは負の値（赤）になる。エッジのない平らな所では正負が打ち消し合い、0 になる。",
  param: `このフィルタの重みはわずか <b class="mono">9</b> 個で、画像全体で同じものを使い回す。<span class="term">全結合層</span>（<span class="en">fully connected layer</span>）で {outN}×{outN} 個の出力をそれぞれ {inN}×{inN} 個の入力につなぐと、{inCells} × {outCells} = <b class="mono">{big}</b> 個の重みが必要になる。フィルタは<span class="term">カーネル</span>（<span class="en">kernel</span>）とも呼ばれる。この後には <a href="#pool">プーリング</a> が続くことが多い。見える範囲を広げる方法は <a href="#dilated">受容野と Dilated Convolution</a>、計算量を減らす方法は <a href="#dwsep">Depthwise Separable Convolution</a> を参照。`,
});
