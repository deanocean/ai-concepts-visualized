/* 生成對抗網路（concepts/gan.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、key、dCap*、calc* 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "gan", {
  lead: `<p>想讓電腦產生「看起來像真的」的資料，卻很難直接寫出「像真的」的評分標準。<span class="term">生成對抗網路</span>（<span class="en">generative adversarial network, GAN</span>）讓兩個網路互相較勁：<span class="term">生成器</span>（<span class="en">generator</span>）把亂數變成假資料，<span class="term">判別器</span>（<span class="en">discriminator</span>）判斷資料是真是假。判別器越會抓，生成器就被逼得越像。</p>`,
  closing: `判別器學著分辨真假，生成器則順著判別器的判斷，把假資料往「更像真的」方向推；兩者輪流更新，理想的終點是判別器到處都只能猜 0.5。兩邊要勢均力敵：讓生成器一次跑太多步，它容易把所有點擠到判別器當下最相信的地方，再一起被趕走，這就是<span class="term">模式崩潰</span>（<span class="en">mode collapse</span>）。`,

  /* 控制項 */
  dLabel: "真資料",
  dCircle: "環狀",
  dRing8: "八團",
  sLabel: "更新方式",
  sStd: "輪流各更新 1 次",
  sG10: "生成器 10 次、判別器 1 次",
  btnSeed: "換一組初始值",
  btnRun: "▶ 開始訓練",
  btnResume: "▶ 繼續訓練",
  btnPause: "❚❚ 暫停",
  btnDone: "✓ 訓練完成",
  btnOne: "前進 1 輪",
  btnFf: "⏭ 快轉到結束",
  progLabel: "查看第幾輪的狀態",

  /* 圖（螢幕報讀用） */
  planeAria: "平面：真資料、生成資料、判別器的判斷",
  dAria: "判別器給真資料與生成資料的平均機率",
  covAria: "生成的點落在 8 個方向的數量",

  /* 圖說 */
  capPlane: "點一個橘色方塊，看它的計算",
  key: `<span><i style="background:var(--pos);border-radius:50%"></i>真資料</span><span><i style="background:var(--hot)"></i>生成器做的假資料</span><span>→ 生成器被推的方向</span><span><i style="background:var(--pos-soft);border:1px solid var(--pos)"></i>判別器認為「真」</span><span><i style="background:var(--hot-soft);border:1px solid var(--hot)"></i>認為「假」</span>`,
  capCov: "1000 個生成點落在哪個方向（深色＝落在真資料附近；虛線＝平均分配的 125 個）",
  dCapStd: `判別器給這一輪 32 個真、32 個假資料的平均機率（每 20 輪再平均）：<span style="color:var(--pos)">━ 真資料</span>　<span style="color:var(--hot)">━ 假資料</span>　虛線 0.5`,
  dCapG10: `判別器給這一輪 32 個真、32 個假資料的平均機率：<span style="color:var(--pos)">━ 真資料</span>　<span style="color:var(--hot)">━ 假資料</span>　虛線 0.5`,
  dAxis: "輪",
  dirR: "右",
  dirUR: "右上",
  dirU: "上",
  dirUL: "左上",
  dirL: "左",
  dirDL: "左下",
  dirD: "下",
  dirDR: "右下",
  covAxis: "方向（從原點看）",

  /* 狀態列 */
  statusHead: "第 {r} 輪：判別器給真資料平均 {ar}、給假資料平均 {af}；生成點 {near}% 落在真資料附近，8 個方向中 {cover} 個方向有 5% 以上的點",
  statusHeadRun: "第 {r} 輪（訓練中）：判別器給真資料平均 {ar}、給假資料平均 {af}；生成點 {near}% 落在真資料附近，8 個方向中 {cover} 個方向有 5% 以上的點",
  tailStartRun: "。剛開始：假資料和真資料的分布完全不同",
  tailStart: "。還沒訓練：假資料和真資料的分布完全不同。按「▶ 開始訓練」連續訓練，或按「前進 1 輪」一輪一輪看",
  tailCollapse: "。模式崩潰：點都擠在少數方向",
  tailChase: "。大多數生成點不在真資料附近：生成器一直追著判別器當下的判斷跑",
  tailOk: "。判別器幾乎分不出真假，假資料也鋪滿了各個方向",
  tailMiss: "。還有方向沒鋪到：GAN 訓練的結果和初始值有關，可以換一組初始值比較",

  /* 算式 */
  calcGK: "生成器",
  calcG: `雜訊 z = {z} → 生成器 → 假資料 x = <span class="hl">{x}</span>`,
  calcDK: "判別器",
  calcD: "D(x) = {q}（認為它是真資料的機率）",
  calcGLossK: "生成器的損失",
  calcGLoss: `−ln D(x) = −ln {q} = <span class="hl">{v}</span>；往 ∂ln D/∂x = {g} 的方向移動，損失會變小（粗箭頭）`,
  calcDLossK: "判別器的損失",
  calcDLoss: `−[ ln D(真) 的平均 + ln(1 − D(假)) 的平均 ]（畫面上各 300 個）= <span class="hl">{v}</span>；完全分不出來（都是 0.5）時 = 2 ln 2 ≈ 1.386`,

  note: `原始論文把生成器的目標寫成「讓 ln(1 − D(G(z))) 變小」，但訓練初期判別器很容易判對，這一項的梯度幾乎是 0，所以實務上（也包括這一頁）改用「讓 −ln D(G(z)) 變小」。生成器從來沒看過真資料，它學到的一切都是透過判別器的梯度傳過來的。這一頁是平面上的點；把兩個網路換成<a href="#conv">卷積</a>層來生成影像的是 <span class="term">DCGAN</span>（<span class="en">deep convolutional GAN</span>）；給定一張圖、生成對應的另一張圖（例如線稿 → 上色），有成對的訓練資料時用 <span class="term">Pix2Pix</span>，沒有成對資料（例如馬 ↔ 斑馬照片）時用 <span class="term">CycleGAN</span>（轉過去再轉回來要變回原圖）。GAN 也用在語音生成、文字生成等。另一種不必兩個網路互相較勁、訓練比較穩定的生成方法，見 <a href="#diffusion">擴散模型</a>。`,
});
