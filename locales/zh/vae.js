/* 變分自編碼器（concepts/vae.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、cap*、calc*、dec*、smpStat 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "vae", {
  lead: `<p><a href="#ae">自編碼器</a>的中間層，數值散在哪裡沒有規則，隨便挑一點解碼不一定像樣，所以不能拿來產生新圖。<span class="term">變分自編碼器</span>（<span class="en">variational autoencoder, VAE</span>）讓編碼器輸出「一個範圍」（平均 μ 與標準差 σ）而不是一個點，並把每個範圍都拉向原點附近的<span class="term">標準常態分布</span>（<span class="en">standard normal distribution</span>）；之後從標準常態分布抽一點交給解碼器，就能生成像樣的圖。</p>`,
  closing: `VAE 的損失是「還原得好不好」加上「範圍離標準常態分布多遠（KL 散度）」：後者把所有圖的位置都收進標準常態分布的範圍、排得緊密，而訓練時每張圖要從自己範圍內隨機的一點還原，解碼器連圖與圖之間的空隙也學會解碼，所以從標準常態分布隨便抽一點都解碼得出像樣的圖。β 決定這股拉力：β = 0 時退回普通的自編碼器，β 太大則還原變模糊。`,

  /* 控制項 */
  bLabel: "β（KL 項的權重）",
  bOpt0: "0（不拉）",
  bOpt1: "1（標準 VAE）",
  bOpt4: "4（β-VAE）",
  btnRun: "▶ 開始訓練",
  btnResume: "▶ 繼續訓練",
  btnPause: "❚❚ 暫停",
  btnDone: "✓ 訓練完成",
  btnFf: "⏭ 快轉到結束",
  btnSeed: "換一組初始值",
  progLabel: "查看第幾個 epoch 的狀態",
  zLabel: "平面顯示範圍",
  zAll: "全部",
  btnEps: "重抽 ε",
  btnDraw: "再抽 16 個",

  /* 圖（螢幕報讀用） */
  latAria: "每張圖在潛在空間的平均與範圍",
  decAria: "在潛在空間選的位置解碼出來的圖",
  mapAria: "潛在空間 −2 到 2 每隔 0.5 解碼出來的圖",
  smpAria: "從標準常態分布抽 16 個點解碼出來的圖",
  lossAria: "每個 epoch 的還原誤差與 KL",

  /* 圖說 */
  capLat: "每張圖的平均 μ（縮圖）與範圍 ±σ（橘色＝選中的圖，× ＝這次抽到的 z）。虛線圓＝標準常態分布的 1σ、2σ。點縮圖選圖，點空白處解碼",
  capMap: "潛在空間地圖：z₁、z₂ 從 −2 到 2 每隔 0.5 交給解碼器",
  epsHint: "ε 是從標準常態分布抽的亂數；換一個 ε，同一張圖會落在範圍內的另一點",
  capSmp: "生成：從標準常態分布 N(0, 1) 抽點 → 解碼器",
  capLoss: `每張圖的平均損失（對數刻度）：<span style="color:var(--accent)">━ 還原誤差</span>　<span style="color:var(--hot)">━ KL</span>`,
  latOut: "有 {n} 張圖的平均在範圍外（改選「全部」）",
  decPick: "你點的位置 ({z}) → 解碼器 → 左圖<br>最像的訓練圖：寬 {w}・高 {h}，{diff}",
  decSel: "選中的圖的平均 μ ({z}) → 解碼器 → 左圖<br>最像的訓練圖：寬 {w}・高 {h}，{diff}",
  decSame: "四捨五入後完全一樣",
  decDiff: "差 {d} 格",
  smpStat: "固定抽 200 個點統計：解碼後和某張訓練圖完全一樣的，涵蓋 <b>{kinds}</b> / 49 種；和每張訓練圖都差 3 格以上（四不像）的有 <b>{far}</b> 個",
  lossAxis: "epoch",

  /* 狀態列 */
  statusHead: "第 {e} 個 epoch：還原誤差 {rec}、KL {kl}；49 張中 {n} 張完全還原；從 N(0,1) 抽點涵蓋 {kinds} / 49 種",
  statusHeadRun: "第 {e} 個 epoch（訓練中）：還原誤差 {rec}、KL {kl}；49 張中 {n} 張完全還原；從 N(0,1) 抽點涵蓋 {kinds} / 49 種",
  tailStartRun: "。剛開始訓練",
  tailStart: "。還沒訓練。按「▶ 開始訓練」開始（約 5 秒）",
  tailB0: "。沒有 KL 的拉力：平均離原點最遠到 {far}、σ 平均只有 {s}，原點附近只對到少數幾種圖",
  tailTight: "。拉力太強，圖擠在一起，還原變模糊",
  tailOk: "。各圖都收進標準常態分布的範圍內，從 N(0,1) 抽點幾乎每種都生成得出來",

  /* 算式 */
  calcEncK: "寬 {w} × 高 {h} → 編碼器",
  calcEnc: "μ = {mu}，ln σ² = {lv} → σ = {sg}",
  calcDrawK: "抽點",
  calcDraw: `ε = {e}，z = μ + σ × ε = <span class="hl">{z}</span>`,
  calcRecK: "還原誤差",
  calcRec: "把 z 交給解碼器，和輸入比：交叉熵 = {v}",
  calcLossK: "這張圖的損失",
  calcLoss: `還原誤差 + β × KL = {rec} + {b} × {kl} = <span class="hl">{v}</span>`,

  note: `抽點本身是隨機的，沒辦法對「抽」這個動作求梯度；改寫成 z = μ + σ × ε，把隨機性全部交給 ε，μ 和 σ 就能照常用<a href="#backprop">反向傳播</a>學，這叫<span class="term">重新參數化技巧</span>（<span class="en">reparameterization trick</span>）。KL 是 <a href="#loss">KL 散度</a>：衡量「平均 μ、標準差 σ 的常態分布」離標準常態分布多遠，μ = 0、σ = 1 時剛好是 0；KL 越小，每張圖的範圍越像標準常態分布。網路實際輸出 ln σ² 而不是 σ，是為了讓 σ 永遠是正的。<span class="term">β-VAE</span>（<span class="en">beta-VAE</span>）把 β 調大，目的是讓<span class="term">潛在空間</span>（<span class="en">latent space</span>）的每個軸各自對應一個因素（例如一軸管寬、一軸管高），代價是還原變模糊；在這個小例子裡，加大 β 最明顯的變化是還原變模糊。其他變形：<span class="term">VQ-VAE</span>（<span class="en">vector quantized VAE</span>）讓<span class="term">潛在變數</span>（<span class="en">latent variable</span>）只能從一組學出來的向量（<span class="term">碼本</span>（<span class="en">codebook</span>））中挑最接近的一個，變成離散的；<span class="term">InfoVAE</span>（<span class="en">information maximizing VAE</span>）修改目標函數，讓潛在變數保留更多關於輸入的資訊。`,
});
