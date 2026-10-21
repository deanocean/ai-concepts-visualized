/* ROC 曲線と AUC（concepts/roc.js）の日本語テキスト
   - である調。key は locales/zh/roc.js と一致させる
   - dLevels：識別能力スライダーの 8 段階の名前を「/」区切りで並べる（「ランダムと同じ」→「完全に分離」の順） */
Atlas.strings("ja", "roc", {
  lead: `<p>分類モデルは多くの場合「陽性である確率」を出力し、そこに<span class="term">しきい値</span>（<span class="en">threshold</span>）を設けて初めて「陽性／陰性」の判定になる。しきい値を下げれば本当の陽性を多く拾えるが誤検出も増え、上げればその逆になる。あるしきい値での<span class="term">正解率</span>（<span class="en">accuracy</span>）だけを見ても、モデルそのものの良し悪しはわからない。<span class="term">ROC 曲線</span>（<span class="en">ROC curve</span>）はすべてのしきい値の結果を 1 本の曲線にまとめ、<span class="term">AUC</span>（<span class="en">area under the curve</span>）はそれを 1 つの数値に要約する。</p>`,
  closing: `しきい値は「どれだけ拾い、どれだけ誤検出するか」のトレードオフを決めるもので、タスクに応じて選ぶ（病気の見逃しが致命的なら、しきい値を下げて再現率を重視する）。一方、ROC 曲線と AUC はしきい値に依存せず、モデルが陽性を陰性より上位に並べる能力を測る。AUC が 1 なら完全に分離でき、0.5 ならランダムと同じである。`,

  /* 操作 */
  dLabel: "モデルの識別能力（2 クラスのスコアの離れ具合）",
  dLevels: "ランダムと同じ/とても弱い/弱い/ふつう/良い/とても良い/優秀/完全に分離",
  tLabel: "しきい値（スコア ≥ しきい値なら陽性と判定）",
  dataLabel: "データ",
  dataBal: "陽性 20・陰性 20",
  dataRare: "陽性がまれ（4 対 36）",

  /* 図 */
  stripAria: "各データのモデルスコアとしきい値（クリックでしきい値を移動）",
  stripCap: "点 1 つ＝1 人。横軸はモデルが出した陽性の確率。図をクリック・ドラッグするとしきい値が動く",
  stripPos: "陽性と判定 →",
  stripNeg: "← 陰性と判定",
  rowP: "陽性",
  rowPSub: "（該当）",
  rowN: "陰性",
  rowNSub: "（非該当）",
  cmCap: "混同行列（このしきい値での結果）",
  rocAria: "ROC 曲線",
  rocCap: "ROC 曲線：しきい値を 1 から 0 へ下げると、点は左下から右上へ進む",
  rocRandom: "ランダム（AUC 0.5）",
  rocX: "偽陽性率 FPR（陰性を誤検出した割合）",
  rocY: "真陽性率 TPR（＝再現率）",

  /* 混同行列 */
  cmPredP: "陽性と判定",
  cmPredN: "陰性と判定",
  cmActP: "実際は陽性",
  cmActN: "実際は陰性",
  cmTP: "真陽性",
  cmFN: "偽陰性",
  cmFP: "偽陽性",
  cmTN: "真陰性",
  cmTPSub: "TP",
  cmFNSub: "FN（見逃し）",
  cmFPSub: "FP（誤検出）",
  cmTNSub: "TN",

  /* ステータス */
  statusRare: "すべて陰性と判定しても正解率は {acc}% になるが、4 人の陽性は 1 人も拾えていない（再現率 0%）。陽性がまれなデータでは、正解率は当てにならない",
  statusPerfect: "このしきい値で 2 クラスが完全に分かれた。見逃しも誤検出もない",
  statusLow: "しきい値が低い。陽性は {tp} / {np} 拾えたが、陰性も {fp} / {nn} が誤検出された",
  statusHigh: "しきい値が高い。誤検出はほとんどない（{fp} / {nn}）が、陽性を {fn} / {np} 見逃している",
  statusInfo: "しきい値 {thr}：陽性 {tp} / {np} を検出、陰性 {fp} / {nn} を誤検出。しきい値を動かして、ROC 曲線上の点の動きを見てみよう",

  /* 計算式 */
  pcNone: "—（陽性と判定したものがない）",
  calcAccK: "正解率",
  calcAcc: `(TP + TN) ÷ 全体 = ({tp} + {tn}) ÷ {n} = <span class="hl">{v}</span>`,
  calcPrecK: "適合率",
  calcPrec: `TP ÷ (TP + FP) = {tp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">陽性と判定したうち、本当に陽性だった割合</span>`,
  calcRecK: "再現率",
  calcRec: `TP ÷ (TP + FN) = {tp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">本当の陽性のうち、検出できた割合（＝ROC の縦軸）</span>`,
  calcF1K: "F値",
  calcF1None: "—（適合率か再現率が 0）",
  calcF1: `2 × 適合率 × 再現率 ÷ (適合率 + 再現率) = <span class="hl">{v}%</span>　<span class="k">両者の調和平均</span>`,
  calcFprK: "偽陽性率",
  calcFpr: `FP ÷ (FP + TN) = {fp} ÷ {d} = <span class="hl">{v}</span>　<span class="k">（ROC の横軸）</span>`,
  calcAucK: "AUC",
  calcAuc: `曲線の下の面積 = <span class="hl">{auc}</span>。検算：陽性 × 陰性の組は全部で {pairs} 組、そのうち陽性のスコアが高いのは {win} 組 → {win} ÷ {pairs} = {ratio}`,

  note: `例は架空の検査結果である（固定した乱数で生成）。陽性は青い丸、陰性は赤い四角で表し、誤判定の点は半透明・黒枠で描いている。しきい値を左へ動かすと ROC 曲線上の点は右上へ進む（多く拾うほど誤検出も増える）。しきい値をどこに置いても点は同じ曲線の上にあるので、曲線はしきい値ではなくモデルに属する。AUC（曲線下面積）は「陽性と陰性を 1 人ずつ無作為に選んだとき、陽性のほうに高いスコアを付ける確率」にちょうど等しい。陽性がまれなデータでは正解率が高くても役に立たないことがあるため、<span class="term">適合率</span>（<span class="en">precision</span>）と<span class="term">再現率</span>（<span class="en">recall</span>）もあわせて見る。両者の調和平均を <span class="term">F値</span>（<span class="en">F-measure</span>）という。この 4 マスの表が<span class="term">混同行列</span>（<span class="en">confusion matrix</span>）である。これらの指標は<b>学習に使っていないデータ</b>で計算して（<span class="term">ホールドアウト検証</span>（<span class="en">holdout validation</span>）、<span class="term">交差検証</span>（<span class="en">cross-validation</span>））初めて<a href="#overfit">汎化性能</a>を表す。確率を出力するモデルの学習は<a href="#loss">誤差関数</a>の交差エントロピーを参照。`,
});
