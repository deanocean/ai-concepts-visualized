# AI 抽象概念互動圖解集

把「隨時間或隨步驟流動」的 AI 抽象概念做成可以親手操作的圖解（G検定 / JDLA シラバス2024 第1.4版）。

純靜態網站，不需建置：直接開 `index.html`，或用 GitHub Pages 發布。

## 檔案結構

- `index.html` — 頁面骨架
- `assets/atlas.css` — 全站樣式與色彩 token（深淺色）
- `assets/atlas.js` — 核心：路由、導覽、語言切換、共用 UI 元件、按需載入概念模組
- `assets/nn.js` — 生成模型、強化學習、word2vec 各頁共用的小型神經網路工具
- `concepts/_registry.js` — 概念清單（分區、シラバス節號、對應模組檔）
- `concepts/<id>.js` — 每個概念一個檔，呼叫 `Atlas.define(id, {...})`，只放程式邏輯
- `locales/<lang>/_core.js` — 全站共用文字（中文 `zh`、日文 `ja`）
- `locales/<lang>/<id>.js` — 各概念的文字
- `tools/check-i18n.mjs` — 檢查中日文語言檔是否一致：`node tools/check-i18n.mjs`
- `docs/glossary.md` — 中／日／英術語對照表
- `CLAUDE.md` — 翻譯規則與把概念拆成雙語的步驟

新增概念：寫 `concepts/<id>.js` 與 `locales/zh|ja/<id>.js`，在 `_registry.js` 該概念填上 `file` 與 `i18n: true`。

## 語言

右上角可切換中文／日本語。預設依瀏覽器語言決定，手動選過會記住；網址加上 `?lang=ja` 或 `?lang=zh` 可直接指定（方便分享）。

> 因為 `concepts/_registry.js` 以底線開頭，repo 根目錄放了 `.nojekyll`，避免 GitHub Pages 的 Jekyll 把它略過。

## 本機預覽

概念模組是用 `<script>` 動態載入，直接雙擊開檔即可；若瀏覽器有限制，可用：

```sh
python3 -m http.server 8000
```
