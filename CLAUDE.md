# AI 抽象概念互動圖解集：給 Claude 的工作說明

純靜態網站（不需建置），中文／日文雙語。結構見 `README.md`。

## 多語系架構

- 程式與文字分開：`concepts/<file>.js` 只放邏輯，文字放在 `locales/<lang>/<id>.js`（`<id>` 是 registry 的概念 id，不一定等於概念檔名，例如 `pool` ↔ `concepts/pooling.js`）。
- 全站共用文字（目錄、按鈕、分區名 `area.<id>`、概念名 `name.<id>`、一句話說明 `desc.<id>`）在 `locales/<lang>/_core.js`。
- 語言檔是 `.js` 不是 `.json`：直接雙擊開 `index.html`（`file://`）時瀏覽器不准讀 JSON；`.js` 也能寫註解、多行字串和函式。
- 概念檔裡用 `ctx.t("key", { 變數 })` 取文字；`{name}` 是佔位符。語序差太多、無法只靠佔位符的句子，值可以寫成函式 `(vars) => "..."`。
- **一定要整句交給語言檔**，不要在程式裡把片段拼成句子（語序會亂）。
- registry 標 `i18n: true` 的概念才會載入語言檔；沒標的仍是舊寫法（文字寫在概念檔裡，只有中文），日文模式下會顯示「中国語のみ」提示。
- 缺字時退回中文；語言決定順序：網址 `?lang=` → 使用者手動選過的 → 瀏覽器語言 → 中文。

## 翻譯規則

### 共通
- 術語第一次出現時標成：`<span class="term">術語</span>（<span class="en">english term</span>）`。括號用全形 `（）`，括號裡只放英文。
- 用詞依 `docs/glossary.md`；遇到表裡沒有的術語，翻好後把它加進表裡。
- 數字與單位之間保留半形空格（`1 格`、`1 マス`、`36 個`），沿用原本的排版習慣。
- 程式識別字、數學式、`<span class="hl">` 等 HTML 原樣保留，只翻文字。
- **陷阱：中文的「列」是橫排（row）、「行」是直排（column）；日文剛好相反（行＝row、列＝column）。**

### 中文（zh）
- 術語用中文（台灣慣用），不再夾日文術語：例如「池化（pooling）」，不是「プーリング（池化，pooling）」。
- 舊概念檔裡的日文術語（入力→輸入、出力→輸出、特徴マップ→特徵圖、ストライド→步幅、全結合層→全連接層…）在拆出語言檔時一併改成中文。

### 日文（ja）
- である調（「〜である」「〜する」「〜になる」），不用です・ます。
- 請讀者操作的提示可用「〜してみよう」。
- 術語用 G検定／JDLA シラバス的寫法（例：最大値プーリング、全結合層、勾配消失問題）。

## 把一個概念拆成雙語（標準步驟）

1. 讀 `concepts/<file>.js`，找出所有顯示給使用者的文字：`lead`、`closing`、按鈕、`aria-label`、圖說、狀態列、算式說明、`note`。
2. 寫 `locales/zh/<id>.js`：照上面的中文規則改寫（日文術語換成中文）。
3. 寫 `locales/ja/<id>.js`：key 與中文檔完全一致，である調。
4. 改概念檔：刪掉 `lead` / `closing`，文字全部改成 `t("key")`；開頭加 `const { t } = ctx;`。
5. `concepts/_registry.js` 該概念加上 `i18n: true`。
6. 執行 `node tools/check-i18n.mjs`，必須顯示「✓ 語言檔一致」（會檢查：兩語言 key 相同、佔位符相同、程式裡不再有中日文字、用到的 key 都有定義）。
7. 用瀏覽器實際看過中文、日文兩種版本（含手機寬度 390px，不能出現橫向捲動），把每個互動都點過一次。

範本：`pool`（`concepts/pooling.js`、`locales/zh/pool.js`、`locales/ja/pool.js`）。

### key 命名
- 小駝峰、看名字就知道是哪段：`btnNext`、`kindMax`、`statusMoved`、`capIn`、`calcSizeK`（`K` 結尾＝算式列左邊的標籤）。
- 固定保留：`lead`、`closing`、`note`。

## 翻譯進度與工作流程

一次 session 處理一個分區，結束時開 PR，合併後下一個 session 從最新的 main 開始。

| 分區 | 概念 id（概念檔） |
|---|---|
| 機器學習基礎 | kernel (svm)、kmeans、pca、roc |
| 神經網路怎麼學習 | gd (gradient-descent)、sgd、optim、backprop、vanish (vanishing)、act (activation)、loss、overfit (overfitting)、dropout、norm |
| 卷積與影像 ✓ | ~~conv (convolution)~~、~~pool (pooling)~~、~~dilated~~、~~dwsep~~、~~resnet~~ |
| 序列與語言 | rnn (rnn-vanishing)、lstm、teacher、seq2seq、selfatt (self-attention)、mha、posenc、transformer、w2v |
| 生成模型 | ae、vae、gan、diffusion |
| 強化學習 | bandit、qlearn |

目前進度以 `node tools/check-i18n.mjs` 最後一行為準。`assets/nn.js` 裡若有顯示用文字，處理到用它的分區時一併拆出（放 `_core.js`）。
