/* =========================================================
   AI 抽象概念互動圖解集 — 核心
   - Atlas.registry(areas, concepts)  由 concepts/_registry.js 呼叫
   - Atlas.define(id, module)         由各概念檔呼叫
       module = {
         lead:    HTML 字串（這個概念解決什麼問題，一兩句）
         closing: HTML 字串（操作完後一句收束）
         mount(benchEl, ctx) → 可回傳 { destroy() }
       }
   - ctx 提供：onTheme(cb)、color(token)、every(ms, fn)（自動清除的計時器）
   - Atlas.ui / Atlas.h / Atlas.s / Atlas.fmt：共用小工具
   ========================================================= */
(function () {
  "use strict";
  const Atlas = (window.Atlas = { areas: [], concepts: [], modules: {}, ui: {} });
  const waiting = {};

  /* ---------- DOM helpers ---------- */
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    if (props) for (const k in props) {
      const v = props[k];
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
    return el;
  }
  const SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) {
      if (k === "text") el.textContent = attrs[k];
      else el.setAttribute(k, attrs[k]);
    }
    for (const c of kids.flat()) if (c) el.append(c);
    return el;
  }
  /** 數字格式：用真正的減號，固定小數位 */
  function fmt(x, d = 2, sign = false) {
    if (!isFinite(x)) return x > 0 ? "∞" : "−∞";
    let t = Math.abs(x) >= 1e5 ? Math.abs(x).toExponential(1) : Math.abs(x).toFixed(d);
    if (Number(t) === 0) return (0).toFixed(d);
    return (x < 0 ? "−" : sign ? "+" : "") + t;
  }
  /** 負數加括號，用在乘式中 */
  const fmtP = (x, d = 2) => (x < 0 ? "(" + fmt(x, d) + ")" : fmt(x, d));
  Atlas.h = h; Atlas.s = s; Atlas.fmt = fmt; Atlas.fmtP = fmtP;

  /* ---------- color helpers ---------- */
  function color(token) { return getComputedStyle(document.documentElement).getPropertyValue(token).trim(); }
  function rgb(c) {
    c = c.trim();
    if (c[0] === "#") {
      if (c.length === 4) c = "#" + [...c.slice(1)].map((x) => x + x).join("");
      return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    }
    const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
  }
  function mix(a, b, t) { const A = rgb(a), B = rgb(b); return A.map((v, i) => Math.round(v + (B[i] - v) * t)); }
  const rgbStr = (arr, alpha) => (alpha == null ? `rgb(${arr.join(",")})` : `rgb(${arr.join(",")} / ${alpha})`);
  /** 正負值上色：正＝--pos，負＝--neg，強度依 |v|/max */
  function divColor(v, max) {
    const t = Math.min(1, Math.abs(v) / (max || 1));
    return rgbStr(mix(color("--panel"), color(v >= 0 ? "--pos" : "--neg"), t * 0.85));
  }
  Atlas.color = color; Atlas.mix = mix; Atlas.rgb = rgb; Atlas.rgbStr = rgbStr; Atlas.divColor = divColor;

  /* ---------- UI widgets ---------- */
  Atlas.ui.btn = (label, onClick, opt = {}) =>
    h("button", { type: "button", class: "btn" + (opt.primary ? " btn--primary" : "") + (opt.sm ? " btn--sm" : ""), onclick: onClick, id: opt.id, "aria-label": opt.aria }, label);

  /** 分段切換鈕 */
  Atlas.ui.seg = ({ options, value, onChange, label, id }) => {
    const wrap = h("div", { class: "seg", role: "group", "aria-label": label || null, id });
    const btns = options.map((o) => {
      const b = h("button", { type: "button", "aria-pressed": String(o.value === value) }, o.label);
      b.addEventListener("click", () => { api.set(o.value); onChange && onChange(o.value); });
      wrap.append(b); return b;
    });
    const api = {
      el: wrap,
      get: () => value,
      set(v) { value = v; btns.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === v))); },
    };
    if (!label) return api;
    api.el = h("div", { class: "field" }, h("span", { class: "field__label", html: label }), wrap);
    return api;
  };

  /** 滑桿 */
  Atlas.ui.slider = ({ id, label, min, max, step, value, format = (v) => v, onInput }) => {
    const val = h("span", { class: "slider__val" }, format(value));
    const input = h("input", { type: "range", id, min, max, step, value });
    input.addEventListener("input", () => { const v = Number(input.value); val.textContent = format(v); onInput && onInput(v); });
    const el = h("label", { class: "slider", for: id }, h("span", { class: "slider__top" }, h("span", { html: label }), val), input);
    return {
      el, get: () => Number(input.value),
      set(v) { input.value = v; val.textContent = format(Number(input.value)); },
    };
  };

  /** 播放器：反覆呼叫 step()，step 回傳 false 時自動停 */
  Atlas.ui.player = (ctx, step, ms = 650) => {
    let timer = null;
    const btn = Atlas.ui.btn("▶ 自動播放", () => api.toggle());
    const api = {
      el: btn,
      get playing() { return !!timer; },
      play() {
        if (timer) return;
        if (step() === false) return;
        timer = ctx.every(ms, () => { if (step() === false) api.pause(); });
        btn.textContent = "❚❚ 暫停";
      },
      pause() { if (timer) { timer.stop(); timer = null; } btn.textContent = "▶ 自動播放"; },
      toggle() { timer ? api.pause() : api.play(); },
    };
    return api;
  };

  /* ---------- registry & modules ---------- */
  Atlas.registry = (areas, concepts) => {
    Atlas.areas = areas; Atlas.concepts = concepts;
    boot();
  };
  Atlas.define = (id, mod) => {
    Atlas.modules[id] = mod;
    if (waiting[id]) { waiting[id].forEach((r) => r(mod)); delete waiting[id]; }
  };
  /** 共用的輔助檔（registry 的 deps），每個只載入一次 */
  const depLoads = {};
  function loadDep(src) {
    return depLoads[src] || (depLoads[src] = new Promise((resolve, reject) => {
      const sc = h("script", { src });
      sc.onload = resolve;
      sc.onerror = () => { delete depLoads[src]; reject(new Error("無法載入 " + src)); };
      document.body.append(sc);
    }));
  }
  function loadModule(c) {
    if (Atlas.modules[c.id]) return Promise.resolve(Atlas.modules[c.id]);
    if (c.deps && c.deps.length) return Promise.all(c.deps.map(loadDep)).then(() => loadModuleFile(c));
    return loadModuleFile(c);
  }
  function loadModuleFile(c) {
    if (Atlas.modules[c.id]) return Promise.resolve(Atlas.modules[c.id]);
    return new Promise((resolve, reject) => {
      (waiting[c.id] = waiting[c.id] || []).push(resolve);
      if (document.querySelector(`script[data-concept="${c.id}"]`)) return;
      const sc = h("script", { src: c.file, "data-concept": c.id });
      sc.onerror = () => reject(new Error("無法載入 " + c.file));
      document.body.append(sc);
    });
  }
  const ready = (c) => !!c.file;
  const conceptById = (id) => Atlas.concepts.find((c) => c.id === id);
  const areaById = (id) => Atlas.areas.find((a) => a.id === id);

  /* ---------- theme watching ---------- */
  const themeSubs = new Set();
  function notifyTheme() { themeSubs.forEach((f) => { try { f(); } catch (e) { console.error(e); } }); }
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", notifyTheme); } catch (e) {}
  new MutationObserver(notifyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });

  /* ---------- 明暗切換（預設跟系統；手動選過就記住） ---------- */
  const THEME_KEY = "atlas-theme";
  const sysDark = () => { try { return matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) { return false; } };
  const effTheme = () => document.documentElement.getAttribute("data-theme") || (sysDark() ? "dark" : "light");
  function syncThemeSwitch() {
    const t = effTheme();
    document.querySelectorAll("#themeSw [data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === t)));
  }
  function setTheme(t) {
    // 選的剛好等於系統設定時，不寫 data-theme，讓之後系統改變仍會跟著走
    if (t === (sysDark() ? "dark" : "light")) {
      document.documentElement.removeAttribute("data-theme");
      try { localStorage.removeItem(THEME_KEY); } catch (e) {}
    } else {
      document.documentElement.setAttribute("data-theme", t);
      try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
    }
    syncThemeSwitch();
  }
  function initThemeSwitch() {
    const sw = document.getElementById("themeSw"); if (!sw) return;
    sw.addEventListener("click", (e) => { const b = e.target.closest("[data-theme-set]"); if (b) setTheme(b.dataset.themeSet); });
    try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeSwitch); } catch (e) {}
    syncThemeSwitch();
  }

  /* ---------- views ---------- */
  const $main = () => document.getElementById("main");
  let current = null; // { destroy, timers, themeFns }

  function teardown() {
    if (!current) return;
    try { current.inst && current.inst.destroy && current.inst.destroy(); } catch (e) { console.error(e); }
    current.timers.forEach((t) => clearInterval(t));
    current.themeFns.forEach((f) => themeSubs.delete(f));
    current = null;
  }

  function renderSidebar(activeId) {
    const nav = document.getElementById("sidebar");
    nav.innerHTML = "";
    nav.append(h("a", { class: "nav-home", href: "#" }, "全部概念總覽"));
    for (const a of Atlas.areas) {
      nav.append(h("div", { class: "nav-area" }, h("span", null, a.name), h("span", { class: "mono" }, a.sec)));
      for (const c of Atlas.concepts.filter((c) => c.area === a.id)) {
        if (ready(c)) {
          nav.append(h("a", { class: "nav-item", href: "#" + c.id, "aria-current": c.id === activeId ? "page" : null },
            h("span", { class: "dot", "aria-hidden": "true" }), c.jp));
        } else {
          nav.append(h("span", { class: "nav-item", title: "準備中" }, h("span", { class: "dot", "aria-hidden": "true" }), c.jp));
        }
      }
    }
  }

  function renderHome() {
    const total = Atlas.concepts.length, done = Atlas.concepts.filter(ready).length;
    const root = h("div", { class: "home" },
      h("h1", null, "AI 抽象概念互動圖解集"),
      h("p", { class: "home__intro" }, "把「隨時間或隨步驟流動」的抽象概念做成可以親手操作的圖解。每一頁的數字都是即時算出來的：調一個參數，就能看到過程怎麼變。"),
      h("div", { class: "legend" },
        h("span", null, h("span", { class: "chip chip--ready" }, "可操作"), `已完成 ${done} / ${total}`),
        h("span", null, h("span", { class: "chip" }, "準備中"), "之後陸續補上"),
        h("span", null, h("span", { class: "chip chip--added" }, "シラバス補充"), "依官方シラバス追加的項目"),
      ),
    );
    for (const a of Atlas.areas) {
      const items = Atlas.concepts.filter((c) => c.area === a.id);
      const tiles = h("div", { class: "tiles" });
      for (const c of items) {
        const chips = h("div", { class: "tile__row" },
          h("span", { class: "chip" + (ready(c) ? " chip--ready" : "") }, ready(c) ? "可操作" : "準備中"),
          h("span", { class: "chip" }, c.sec),
          c.added ? h("span", { class: "chip chip--added" }, "シラバス補充") : null);
        const inner = [h("span", { class: "tile__jp" }, c.jp), h("span", { class: "tile__sub" }, c.zh), chips];
        tiles.append(ready(c)
          ? h("a", { class: "tile", href: "#" + c.id }, ...inner, h("span", { class: "tile__go" }, "開始操作 →"))
          : h("div", { class: "tile" }, ...inner));
      }
      root.append(h("section", { class: "area" },
        h("div", { class: "area__head" }, h("h2", null, a.name), h("span", { class: "mono" }, a.sec)), tiles));
    }
    $main().replaceChildren(root);
    document.title = "AI 抽象概念互動圖解集";
  }

  function renderConcept(c) {
    const area = areaById(c.area);
    const readyList = Atlas.concepts.filter(ready);
    const idx = readyList.indexOf(c);
    const prev = readyList[idx - 1], next = readyList[idx + 1];
    const bench = h("div", { class: "bench", id: "bench-" + c.id });
    const leadBox = h("div", { class: "callout lead" }, h("span", { class: "callout__label" }, "這解決什麼問題"));
    const takeBox = h("div", { class: "callout takeaway", hidden: true }, h("span", { class: "callout__label" }, "收束"));
    const root = h("article", { class: "concept" },
      h("a", { class: "crumb", href: "#" }, "← 全部概念"),
      h("header", { class: "c-head" },
        h("div", { class: "c-eyebrow" }, h("span", null, area ? area.name : ""), h("span", { class: "mono" }, "シラバス " + c.sec)),
        h("h1", null, c.jp),
        h("div", { class: "c-sub" }, c.zh, " · ", h("span", { class: "mono" }, c.en))),
      leadBox, bench, takeBox,
      h("nav", { class: "c-pager", "aria-label": "上一個／下一個概念" },
        prev ? h("a", { href: "#" + prev.id }, h("small", null, "← 上一個"), prev.jp) : h("span"),
        next ? h("a", { class: "next", href: "#" + next.id }, h("small", null, "下一個 →"), next.jp) : null));
    $main().replaceChildren(root);
    document.title = c.jp + "｜AI 抽象概念互動圖解集";

    if (!ready(c)) { bench.replaceWith(h("div", { class: "placeholder" }, "這個概念的圖解準備中。")); return; }
    bench.append(h("p", { class: "hint" }, "載入中…"));
    const token = (current = { timers: [], themeFns: [], inst: null });
    loadModule(c).then((mod) => {
      if (current !== token) return;
      leadBox.append(h("div", { html: mod.lead || "" }));
      if (mod.closing) { takeBox.append(h("p", { html: mod.closing })); takeBox.hidden = false; }
      bench.replaceChildren();
      const ctx = {
        id: c.id,
        color,
        onTheme(fn) { themeSubs.add(fn); token.themeFns.push(fn); },
        every(ms, fn) { const t = setInterval(fn, ms); token.timers.push(t); return { stop: () => clearInterval(t) }; },
      };
      try { token.inst = mod.mount(bench, ctx) || null; }
      catch (e) { console.error(e); bench.append(h("p", { class: "err" }, "圖解發生錯誤：" + e.message)); }
    }).catch((e) => { bench.replaceChildren(h("p", { class: "err" }, e.message)); });
  }

  function route() {
    teardown();
    const id = decodeURIComponent(location.hash.replace(/^#/, ""));
    const c = id && conceptById(id);
    renderSidebar(c ? c.id : null);
    if (c) renderConcept(c); else renderHome();
    closeDrawer();
    window.scrollTo(0, 0);
  }

  /* ---------- drawer (phones) ---------- */
  function openDrawer() {
    document.getElementById("sidebar").classList.add("is-open");
    document.getElementById("scrim").hidden = false;
    document.getElementById("menuBtn").setAttribute("aria-expanded", "true");
  }
  function closeDrawer() {
    const sb = document.getElementById("sidebar"); if (!sb) return;
    sb.classList.remove("is-open");
    document.getElementById("scrim").hidden = true;
    document.getElementById("menuBtn").setAttribute("aria-expanded", "false");
  }

  function boot() {
    initThemeSwitch();
    document.getElementById("menuBtn").addEventListener("click", () =>
      document.getElementById("sidebar").classList.contains("is-open") ? closeDrawer() : openDrawer());
    document.getElementById("scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
    window.addEventListener("hashchange", route);
    route();
  }
})();
