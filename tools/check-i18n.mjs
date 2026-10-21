#!/usr/bin/env node
/* 檢查語言檔是否一致：node tools/check-i18n.mjs
   - 每個語言檔的 key 在各語言都要有（缺的、多的都列出）
   - 同一個 key 在各語言用的 {佔位符} 要一樣
   - registry 的每個分區、概念在 _core.js 都有 area.<id>、name.<id>、desc.<id>
   - 標了 i18n: true 的概念：各語言的 locales/<lang>/<id>.js 都存在；
     概念檔裡 t("key") 用到的 key 都有定義；概念檔（註解以外）不再有中日文字
   有錯誤時結束碼為 1。 */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LANGS = ["zh", "ja"];
const BASE = "zh";
const errors = [], warnings = [];
const err = (m) => { if (!errors.includes(m)) errors.push(m); }, warn = (m) => warnings.push(m);

/* ---------- 載入 registry 與所有語言檔（用假的 Atlas 收集資料） ---------- */
const dicts = {}; // dicts[lang][ns] = dict
let areas = [], concepts = [], calls = [];
const Atlas = {
  registry(a, c) { areas = a; concepts = c; },
  strings(lang, ns, dict) { calls.push(`${lang}/${ns}`); ((dicts[lang] ??= {})[ns] ??= {}); Object.assign(dicts[lang][ns], dict); },
};
function run(file) {
  const code = fs.readFileSync(path.join(ROOT, file), "utf8");
  try { vm.runInNewContext(code, { Atlas }, { filename: file }); }
  catch (e) { err(`${file}：無法執行（${e.message}）`); }
}
run("concepts/_registry.js");
for (const lang of LANGS) {
  const dir = path.join(ROOT, "locales", lang);
  if (!fs.existsSync(dir)) { err(`缺少資料夾 locales/${lang}/`); continue; }
  for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".js")).sort()) {
    const ns = f === "_core.js" ? "core" : f.slice(0, -3);
    calls = [];
    run(`locales/${lang}/${f}`);
    if (calls.length !== 1 || calls[0] !== `${lang}/${ns}`)
      err(`locales/${lang}/${f}：應只呼叫一次 Atlas.strings("${lang}", "${ns}", {...})（實際：${calls.join("、") || "沒有呼叫"}）`);
  }
}

/* ---------- 各語言 key 與佔位符一致 ---------- */
const holders = (v) => (typeof v === "string" ? [...new Set(v.match(/\{\w+\}/g) || [])].sort().join(" ") : null);
const allNs = new Set(LANGS.flatMap((l) => Object.keys(dicts[l] || {})));
for (const ns of allNs) {
  const file = (l) => `locales/${l}/${ns === "core" ? "_core" : ns}.js`;
  const keys = new Set(LANGS.flatMap((l) => Object.keys((dicts[l] || {})[ns] || {})));
  for (const k of keys) {
    for (const l of LANGS) {
      const d = (dicts[l] || {})[ns];
      if (!d) continue;
      if (!(k in d)) err(`${file(l)}：缺少 "${k}"`);
      else if (typeof d[k] === "string" && !d[k].trim()) warn(`${file(l)}：「${k}」是空字串`);
    }
    const hs = LANGS.map((l) => ((dicts[l] || {})[ns] || {})[k]).filter((v) => v !== undefined).map(holders).filter((v) => v !== null);
    if (new Set(hs).size > 1) err(`${ns}.${k}：各語言的佔位符不同（${hs.map((h, i) => `${LANGS[i]}: ${h || "無"}`).join("；")}）`);
  }
  for (const l of LANGS) if (!(dicts[l] || {})[ns]) err(`缺少 ${file(l)}`);
}

/* ---------- registry 對應的名稱 ---------- */
for (const l of LANGS) {
  const core = (dicts[l] || {}).core || {};
  for (const a of areas) if (!(`area.${a.id}` in core)) err(`locales/${l}/_core.js：缺少 "area.${a.id}"`);
  for (const c of concepts) for (const k of ["name", "desc"]) if (!(`${k}.${c.id}` in core)) err(`locales/${l}/_core.js：缺少 "${k}.${c.id}"`);
}

/* ---------- 已拆出語言檔的概念 ---------- */
const CJK = /[぀-ヿ㐀-鿿！-～]/;
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " ")).replace(/(^|[^:"'`])\/\/.*$/gm, "$1");
for (const c of concepts.filter((c) => c.i18n)) {
  if (!c.file) { err(`${c.id}：標了 i18n 但沒有 file`); continue; }
  for (const l of LANGS) if (!((dicts[l] || {})[c.id])) err(`${c.id}：缺少 locales/${l}/${c.id}.js`);
  const src = fs.readFileSync(path.join(ROOT, c.file), "utf8");
  const base = (dicts[BASE] || {})[c.id] || {};
  for (const m of src.matchAll(/\bt\(\s*["'`](\w+)["'`]/g)) if (!(m[1] in base)) err(`${c.file}：t("${m[1]}") 沒有定義在 locales/${BASE}/${c.id}.js`);
  const literals = new Set([...src.matchAll(/["'`](\w+)["'`]/g)].map((m) => m[1]));
  for (const k of Object.keys(base)) if (k !== "lead" && k !== "closing" && !literals.has(k)) warn(`locales/${BASE}/${c.id}.js：「${k}」在 ${c.file} 裡沒有用到`);
  stripComments(src).split("\n").forEach((line, i) => {
    if (CJK.test(line)) err(`${c.file}:${i + 1}：程式裡還有中日文字（應移到語言檔）：${line.trim().slice(0, 60)}`);
  });
  if (/\b(lead|closing)\s*:\s*[`"']/.test(stripComments(src))) err(`${c.file}：lead / closing 應放在語言檔`);
}

/* ---------- 結果 ---------- */
const ready = concepts.filter((c) => c.file);
const done = ready.filter((c) => c.i18n);
for (const w of warnings) console.log("⚠ " + w);
for (const e of errors) console.log("✗ " + e);
console.log(`\n已拆出語言檔（中日文）：${done.length} / ${ready.length}${done.length ? "（" + done.map((c) => c.id).join("、") + "）" : ""}`);
console.log(errors.length ? `\n${errors.length} 個錯誤` : "\n✓ 語言檔一致");
process.exit(errors.length ? 1 : 0);
