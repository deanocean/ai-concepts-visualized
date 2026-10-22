/* 正規化層（concepts/norm.js）の日本語テキスト
   - である調。key は locales/zh/norm.js と一致させる
   - 説明の行は descBn／descLn／descIn／descGn のどれか 1 文に descFrame（と descAfter）を続けたもの */
Atlas.strings("ja", "norm", {
  lead: `<p>ある層の出力のうち、値がどれも 50 前後のチャンネルもあれば、0 付近のチャンネルもある。訓練中に前の層の重みが変わると、後ろの各層が受け取る値の範囲もずれていく。値のスケールがばらばらだと、1 つの学習率をすべての箇所に合わせるのが難しく、学習は遅く不安定になる。<span class="term">正規化層</span>（<span class="en">normalization layer</span>）は、ひとまとまりの値から平均を引いて標準偏差で割り、「平均 0、標準偏差 1」にそろえる。各手法の違いは、<b>どの値をまとめて平均を計算するか</b>だけである。</p>`,
  closing: `バッチ正規化はチャンネルごとにバッチ全体で計算するので、結果が同じバッチの他のデータによって変わり、バッチが小さすぎると不安定になる。レイヤー正規化はデータごとに単独で計算するのでバッチの影響を受けず、Transformer や RNN で使われる。インスタンス正規化はデータごと・チャンネルごとに別々に計算する。グループ正規化はその中間で、チャンネルをグループに分けて計算し、バッチがとても小さくても安定する。`,

  /* 操作 */
  methodLabel: "正規化の手法",
  mBn: "バッチ正規化",
  mLn: "レイヤー正規化",
  mIn: "インスタンス正規化",
  mGn: "グループ正規化",
  viewLabel: "表示",
  viewBefore: "正規化前",
  viewAfter: "正規化後",
  btnSwap: "データ 2〜4 だけ入れ替える",
  btnRestore: "データを元に戻す",

  /* 説明の行 */
  descBn: `<span class="term">バッチ正規化</span>（<span class="en">batch normalization</span>）：同じチャンネルについて、バッチ全体のすべての位置（N × H × W）をまとめて計算する。`,
  descLn: `<span class="term">レイヤー正規化</span>（<span class="en">layer normalization</span>）：同じデータについて、すべてのチャンネルのすべての位置（C × H × W）をまとめて計算する。`,
  descIn: `<span class="term">インスタンス正規化</span>（<span class="en">instance normalization</span>）：同じデータの同じチャンネル（H × W）だけで計算する。`,
  descGn: `<span class="term">グループ正規化</span>（<span class="en">group normalization</span>）：同じデータについて、同じグループのチャンネル（ここでは 2 つで 1 組）をまとめて計算する。`,
  descFrame: "オレンジの枠＝選んだ値と一緒に平均を計算するグループ。",
  descAfter: "マスには小数第 1 位まで表示している。正確な値は下の計算式を参照。",

  /* 表 */
  headCh: "<b>ch{c}</b><br>平均 約 {m}",
  groupFmt: "グループ {g}",
  rowFmt: "データ {n}",
  cellAria: "データ {n} ch{c} の {k} マス目",

  /* ステータス */
  sChanged: "データ 1 はまったく変わっていないのに、正規化後の値が変わった（選んだマス {a} → {b}）：同じバッチの他のデータの影響を受ける",
  sSame: "データ 1 の正規化後の値はまったく変わらない（選んだマスは {y} のまま）：自分の値だけで計算している",
  sSwapped: "データ 2〜4 を入れ替えた。先にデータ 1 のマスを選んでから入れ替えると、前後を比べられる",
  sGroup: "このグループの値は {n} 個：平均 {mu}、標準偏差 {sd}。正規化後は平均 0、標準偏差 1",

  /* 計算式 */
  calcSelK: "選択",
  calcSel: "データ {n}・ch{c} の x = <b>{x}</b>",
  calcCountK: "同じグループ",
  calcCount: "全 {n} 個",
  calcMeanK: "平均",
  calcVarK: "分散",
  calcVar: "平均((x − μ)²)",
  calcSdK: "標準偏差",
  calcNormK: "正規化",
  calcLastK: "最後に",
  calcLastNote: "（γ・β は学習可能なパラメータ。ここでは γ = 1、β = 0）",

  note: `ε はごく小さな数（ここでは 0.00001）で、0 で割るのを防ぐ。正規化の後に γ を掛けて β を足すことで、必要ならモデルがスケールを元に戻せるので、表現力は失われない。バッチ正規化では推論時にデータが 1 件ずつ来ることが多く、「同じバッチ」で計算できないため、訓練中に<span class="term">移動平均</span>（<span class="en">moving average</span>）で蓄積した平均と分散を代わりに使う。元の論文は、これが<span class="term">内部共変量シフト</span>（<span class="en">internal covariate shift</span>）、つまり前の層が更新されるたびに後ろの層が受け取る値の分布が変わる問題を解決すると説明している。なお「正規化」（normalization）と<a href="#overfit">正則化</a>（regularization）は別物である。シラバスでもこの節は「正則化層」から「正規化層」に名前が変わった。インスタンス正規化は<span class="term">スタイル変換</span>（<span class="en">style transfer</span>）でよく使われ、グループ正規化は大きな画像を数枚しかバッチに入れられない<span class="term">物体検出</span>（<span class="en">object detection</span>）でよく使われる。数値は例として用意した整数で、各特徴マップは 2×2 しかない。`,
});
