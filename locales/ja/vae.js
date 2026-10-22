/* 変分オートエンコーダ（concepts/vae.js）の日本語テキスト
   - である調。key は locales/zh/vae.js と一致させる */
Atlas.strings("ja", "vae", {
  lead: `<p><a href="#ae">オートエンコーダ</a>の中間層は、値がどこに散らばるかに規則がなく、適当に選んだ点をデコードしてもまともな画像になるとは限らない。そのため新しい画像の生成には使えない。<span class="term">変分オートエンコーダ</span>（<span class="en">variational autoencoder, VAE</span>）は、エンコーダに 1 点ではなく「範囲」（平均 μ と標準偏差 σ）を出力させ、どの範囲も原点付近の<span class="term">標準正規分布</span>（<span class="en">standard normal distribution</span>）へ引き寄せる。こうすると、標準正規分布から 1 点を取り出してデコーダに渡すだけで、まともな画像を生成できる。</p>`,
  closing: `VAE の損失は「どれだけうまく復元できたか」に「範囲が標準正規分布からどれだけ離れているか（KL 情報量）」を加えたものである。後者はすべての画像の位置を標準正規分布の範囲に収め、密に並べる。さらに訓練では各画像を自分の範囲内のランダムな 1 点から復元するため、デコーダは画像と画像の間のすき間もデコードできるようになる。だから標準正規分布から適当に点を取っても、まともな画像がデコードされる。この引き寄せる力を決めるのが β で、β = 0 なら普通のオートエンコーダに戻り、β が大きすぎると復元がぼやける。`,

  /* 操作 */
  bLabel: "β（KL 項の重み）",
  bOpt0: "0（引き寄せない）",
  bOpt1: "1（標準の VAE）",
  bOpt4: "4（β-VAE）",
  btnRun: "▶ 訓練開始",
  btnResume: "▶ 訓練再開",
  btnPause: "❚❚ 一時停止",
  btnDone: "✓ 訓練完了",
  btnFf: "⏭ 最後まで早送り",
  btnSeed: "初期値を変える",
  progLabel: "何エポック目の状態を見るか",
  zLabel: "平面の表示範囲",
  zAll: "全体",
  btnEps: "ε を引き直す",
  btnDraw: "もう一度 16 個引く",

  /* 図（読み上げ用） */
  latAria: "各画像の潜在空間での平均と範囲",
  decAria: "潜在空間で選んだ位置をデコードした画像",
  mapAria: "潜在空間の −2〜2 を 0.5 刻みでデコードした画像",
  smpAria: "標準正規分布から 16 点を取り出してデコードした画像",
  lossAria: "エポックごとの復元誤差と KL",

  /* 図の説明 */
  capLat: "各画像の平均 μ（サムネイル）と範囲 ±σ（オレンジ＝選んだ画像、×＝今回引いた z）。点線の円＝標準正規分布の 1σ、2σ。サムネイルをクリックで選択、空いた所をクリックでデコード",
  capMap: "潜在空間マップ：z₁、z₂ を −2〜2 の範囲で 0.5 刻みにデコーダへ渡す",
  epsHint: "ε は標準正規分布から引いた乱数である。ε を変えると、同じ画像が範囲内の別の点に落ちる",
  capSmp: "生成：標準正規分布 N(0, 1) から点を取り出す → デコーダ",
  capLoss: `画像 1 枚あたりの平均損失（対数目盛）：<span style="color:var(--accent)">━ 復元誤差</span>　<span style="color:var(--hot)">━ KL</span>`,
  latOut: "{n} 枚の画像の平均が範囲外にある（「全体」を選ぼう）",
  decPick: "クリックした位置 ({z}) → デコーダ → 左の画像<br>最も似た訓練画像：幅 {w}・高さ {h}、{diff}",
  decSel: "選んだ画像の平均 μ ({z}) → デコーダ → 左の画像<br>最も似た訓練画像：幅 {w}・高さ {h}、{diff}",
  decSame: "四捨五入すると完全に一致",
  decDiff: "{d} マス違う",
  smpStat: "固定の 200 点で集計：デコード結果がいずれかの訓練画像と完全に一致したのは <b>{kinds}</b> / 49 種類、どの訓練画像とも 3 マス以上違う（どれでもない）ものは <b>{far}</b> 個",
  lossAxis: "エポック",

  /* ステータス */
  statusHead: "{e} エポック目：復元誤差 {rec}、KL {kl}。49 枚中 {n} 枚を完全に復元。N(0,1) から点を取ると 49 種類中 {kinds} 種類を生成",
  statusHeadRun: "{e} エポック目（訓練中）：復元誤差 {rec}、KL {kl}。49 枚中 {n} 枚を完全に復元。N(0,1) から点を取ると 49 種類中 {kinds} 種類を生成",
  tailStartRun: "。訓練が始まったところである",
  tailStart: "。まだ訓練していない。「▶ 訓練開始」を押してみよう（約 5 秒）",
  tailB0: "。KL の引き寄せがないため、平均は原点から最大 {far} まで離れ、σ の平均はわずか {s} である。原点付近は数種類の画像にしか対応しない",
  tailTight: "。引き寄せが強すぎて画像が密集し、復元がぼやける",
  tailOk: "。どの画像も標準正規分布の範囲に収まり、N(0,1) から点を取るとほぼすべての種類が生成できる",

  /* 計算式 */
  calcEncK: "幅 {w} × 高さ {h} → エンコーダ",
  calcEnc: "μ = {mu}、ln σ² = {lv} → σ = {sg}",
  calcDrawK: "点を取る",
  calcDraw: `ε = {e}、z = μ + σ × ε = <span class="hl">{z}</span>`,
  calcRecK: "復元誤差",
  calcRec: "z をデコーダに渡して入力と比べる：交差エントロピー = {v}",
  calcLossK: "この画像の損失",
  calcLoss: `復元誤差 + β × KL = {rec} + {b} × {kl} = <span class="hl">{v}</span>`,

  note: `点を取り出す操作自体はランダムなので、「取り出す」ことについて勾配は求められない。z = μ + σ × ε と書き直してランダム性をすべて ε に任せれば、μ と σ は通常どおり<a href="#backprop">誤差逆伝播法</a>で学習できる。これを<span class="term">Reparameterization Trick</span>（<span class="en">reparameterization trick</span>）という。KL は <a href="#loss">KL 情報量</a>で、「平均 μ、標準偏差 σ の正規分布」が標準正規分布からどれだけ離れているかを測る。μ = 0、σ = 1 のときちょうど 0 になり、KL が小さいほど各画像の範囲は標準正規分布に近い。ネットワークが σ ではなく ln σ² を出力するのは、σ を常に正にするためである。<span class="term">β-VAE</span>（<span class="en">beta-VAE</span>）は β を大きくして、<span class="term">潜在空間</span>（<span class="en">latent space</span>）の各軸がそれぞれ 1 つの要因に対応する（例えばある軸が幅、別の軸が高さを担う）ようにすることを狙うが、代償として復元がぼやける。この小さな例では、β を大きくしたときに最も目立つ変化は復元のぼやけである。ほかの派生として、<span class="term">VQ-VAE</span>（<span class="en">vector quantized VAE</span>）は<span class="term">潜在変数</span>（<span class="en">latent variable</span>）を、学習したベクトルの集まり（<span class="term">コードブック</span>（<span class="en">codebook</span>））の中から最も近い 1 つに限定し、離散的にする。<span class="term">InfoVAE</span>（<span class="en">information maximizing VAE</span>）は目的関数を修正し、潜在変数が入力についてより多くの情報を保つようにする。`,
});
