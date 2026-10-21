/* 正規化層（シラバス §19）
   一層的輸出：4 筆資料（N）× 4 個チャンネル（C）× 每張特徴マップ 2×2（H×W），共 64 個數。
   バッチ正規化／レイヤー正規化／インスタンス正規化／グループ正規化（2 組，每組 2 個チャンネル）
   差別只在「哪些數一起算平均與變異數」。點任一個數看它屬於哪一組，算式框代入實際數字。
   「換掉其他 3 筆資料」：資料 1 不變，看它正規化後的值會不會跟著變（只有バッチ正規化會）。 */
Atlas.define("norm", {
  lead: `<p>一層的輸出裡，有的チャンネル數值都在 50 左右，有的在 0 附近；訓練中前面的權重一改，後面每層收到的數值範圍也跟著跑。數值尺度不一，學習率很難同時適合每一處，學習就慢又不穩。<span class="jp">正規化層</span>（標準化層，<span class="en">normalization layer</span>）把一組數減去平均、除以標準差，拉回「平均 0、標準差 1」。各種方法的差別，只在<b>哪些數放在一起算平均</b>。</p>`,
  closing: `<span class="jp">バッチ正規化</span>每個チャンネル跨整批資料算，所以結果會隨同一批的其他資料而變，批次太小就不穩；<span class="jp">レイヤー正規化</span>每筆資料自己算，不受批次影響，Transformer 和 RNN 都用它；<span class="jp">インスタンス正規化</span>每筆資料的每個チャンネル各自算；<span class="jp">グループ正規化</span>介於兩者之間，把チャンネル分組來算，批次很小時也穩定。`,

  mount(bench, ctx) {
    const { h, fmt, fmtP, ui } = Atlas;
    const N = 4, C = 4, HW = 4, G = 2, EPS = 1e-5;
    const CH_MEAN = [10, -2, 50, 0], CH_SD = [2, 1, 8, 3];
    let seed = 1;
    function rng(sd) { return function () { sd |= 0; sd = (sd + 0x6d2b79f5) | 0; let t = Math.imul(sd ^ (sd >>> 15), 1 | sd); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    const gauss = (r) => Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());
    /** 第 n 筆資料（每筆有自己的整體偏移，讓レイヤー正規化也有事可做） */
    function sample(r) {
      const shift = gauss(r) * 1.5;
      return Array.from({ length: C }, (_, c) => Array.from({ length: HW }, () => Math.round(CH_MEAN[c] + CH_SD[c] * (gauss(r) + shift * 0.6))));
    }
    const first = sample(rng(4242));
    let X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + i)))];
    let method = "bn", showAfter = false, sel = { n: 0, c: 1, k: 0 }, prevSel = null;

    const METHODS = {
      bn: { name: "バッチ正規化", zh: "批次標準化", en: "batch normalization", desc: "同一個チャンネル、跨整批資料的所有位置（N × H × W）一起算" },
      ln: { name: "レイヤー正規化", zh: "層標準化", en: "layer normalization", desc: "同一筆資料、所有チャンネル的所有位置（C × H × W）一起算" },
      in: { name: "インスタンス正規化", zh: "實例標準化", en: "instance normalization", desc: "同一筆資料、同一個チャンネル（H × W）自己算" },
      gn: { name: "グループ正規化", zh: "組標準化", en: "group normalization", desc: "同一筆資料、同一組チャンネル（這裡 2 個一組）一起算" },
    };
    const group = (m, n, c) => {
      const out = [];
      for (let nn = 0; nn < N; nn++) for (let cc = 0; cc < C; cc++) for (let k = 0; k < HW; k++) {
        const ok = m === "bn" ? cc === c : m === "ln" ? nn === n : m === "in" ? nn === n && cc === c : nn === n && Math.floor(cc / (C / G)) === Math.floor(c / (C / G));
        if (ok) out.push([nn, cc, k]);
      }
      return out;
    };
    function stats(m, n, c) {
      const vals = group(m, n, c).map(([a, b, k]) => X[a][b][k]);
      const mu = vals.reduce((t, v) => t + v, 0) / vals.length;
      const v2 = vals.reduce((t, v) => t + (v - mu) ** 2, 0) / vals.length;
      return { vals, mu, v2 };
    }
    const normed = (m, n, c, k) => { const st = stats(m, n, c); return (X[n][c][k] - st.mu) / Math.sqrt(st.v2 + EPS); };

    bench.append(h("style", { text: `
      .nm-wrap { display: grid; grid-template-columns: auto repeat(${C}, auto); gap: 6px 8px; align-items: center; width: max-content; margin-inline: auto; }
      .nm-head { font-size: 12px; color: var(--muted); text-align: center; line-height: 1.3; }
      .nm-head b { color: var(--ink); }
      .nm-gbar { grid-column: span 2; text-align: center; font-size: 11.5px; color: var(--hot); border-bottom: 2px solid var(--hot); padding-bottom: 1px; }
      .nm-gbar.off { visibility: hidden; }
      .nm-row { font-size: 12.5px; color: var(--muted); white-space: nowrap; text-align: right; }
      .nm-row.fixed b { color: var(--accent); }
      .nm-map { display: grid; grid-template-columns: repeat(2, var(--nc)); gap: 2px; padding: 3px; border-radius: 7px; border: 1.5px solid transparent; }
      .nm-map.grp { border-color: var(--hot); background: var(--hot-soft); }
      .nm-v { width: var(--nc); height: calc(var(--nc) * 0.8); display: grid; place-items: center; font-family: var(--f-mono); font-size: 11.5px; border: 1px solid var(--line); border-radius: 4px; background: var(--panel); color: var(--ink); padding: 0; cursor: pointer; }
      .nm-v:hover { border-color: var(--accent); }
      .nm-v.cur { outline: 2.5px solid var(--hot); outline-offset: 1px; font-weight: 700; }
      .nm { --nc: clamp(30px, 8.6vw, 40px); }
    ` }));
    bench.classList.add("nm");

    const mSeg = ui.seg({ label: "正規化方法", value: method, options: Object.entries(METHODS).map(([k, v]) => ({ value: k, label: v.name })), onChange: (v) => { method = v; prevSel = null; render(); } });
    const viewSeg = ui.seg({ label: "顯示", value: showAfter, options: [{ value: false, label: "正規化前" }, { value: true, label: "正規化後" }], onChange: (v) => { showAfter = v; render(); } });
    const desc = h("p", { class: "hint" });
    const table = h("div", { class: "nm-wrap" });
    const status = h("div", { class: "status" });
    const calc = h("div", { class: "calc" });
    const swapBtn = ui.btn("換掉資料 2〜4（資料 1 不變）", () => {
      prevSel = sel.n === 0 ? { key: `${sel.c},${sel.k},${method}`, val: normed(method, 0, sel.c, sel.k), all: X[0].map((ch, c) => ch.map((_, k) => normed(method, 0, c, k))) } : null;
      seed++;
      X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + seed * 10 + i)))];
      render(true);
    }, { primary: true });
    const note = h("p", { class: "note" });
    bench.append(
      h("div", { class: "bench__row" }, mSeg.el, viewSeg.el),
      desc,
      h("div", { class: "scroll-x" }, table),
      status,
      h("div", { class: "bench__row" }, swapBtn, ui.btn("還原資料", () => { seed = 1; X = [first, ...Array.from({ length: N - 1 }, (_, i) => sample(rng(100 + i)))]; prevSel = null; render(); })),
      calc, note);

    function render(swapped) {
      const M = METHODS[method];
      desc.innerHTML = `<span class="jp">${M.name}</span>（${M.zh}，<span class="en">${M.en}</span>）：${M.desc}。橘框＝和選中的數一起算平均的那一組。${showAfter ? "格子裡顯示到小數一位，精確值見下方算式。" : ""}`;
      const inGrp = new Set(group(method, sel.n, sel.c).map(([a, b, k]) => `${a},${b},${k}`));
      const maps = new Set(group(method, sel.n, sel.c).map(([a, b]) => `${a},${b}`));
      const kids = [h("span")];
      for (let c = 0; c < C; c++) kids.push(h("div", { class: "nm-head", html: `<b>ch${c + 1}</b><br>平均約 ${fmt(CH_MEAN[c], 0)}` }));
      // グループ正規化的分組標示
      kids.push(h("span"));
      for (let g = 0; g < G; g++) kids.push(h("div", { class: "nm-gbar" + (method === "gn" ? "" : " off") }, `グループ ${g + 1}`));
      for (let n = 0; n < N; n++) {
        kids.push(h("div", { class: "nm-row" + (n === 0 ? " fixed" : ""), html: n === 0 ? "<b>資料 1</b>" : `資料 ${n + 1}` }));
        for (let c = 0; c < C; c++) {
          const map = h("div", { class: "nm-map" + (maps.has(`${n},${c}`) ? " grp" : "") });
          for (let k = 0; k < HW; k++) {
            const v = showAfter ? normed(method, n, c, k) : X[n][c][k];
            const cur = sel.n === n && sel.c === c && sel.k === k;
            const b = h("button", { type: "button", class: "nm-v" + (cur ? " cur" : ""), "aria-label": `資料 ${n + 1} ch${c + 1} 第 ${k + 1} 格`,
              style: showAfter ? { background: Atlas.divColor(v, 2.2) } : null }, showAfter ? fmt(v, 1) : fmt(v, 0));
            b.addEventListener("click", () => { sel = { n, c, k }; prevSel = null; render(); });
            map.append(b);
          }
          kids.push(map);
        }
      }
      table.replaceChildren(...kids);

      const st = stats(method, sel.n, sel.c), x = X[sel.n][sel.c][sel.k], y = (x - st.mu) / Math.sqrt(st.v2 + EPS);
      // 狀態：換資料後，資料 1 的結果有沒有變
      if (swapped && prevSel) {
        const now = X[0].map((ch, c) => ch.map((_, k) => normed(method, 0, c, k)));
        const changed = now.flat().some((v, i) => Math.abs(v - prevSel.all.flat()[i]) > 1e-9);
        if (changed) { status.className = "status status--warn"; status.textContent = `資料 1 完全沒變，但它正規化後的值變了（選中的那格 ${fmt(prevSel.val, 3)} → ${fmt(y, 3)}）：同一批的其他資料會影響它`; }
        else { status.className = "status status--ok"; status.textContent = `資料 1 正規化後的值完全沒變（選中的那格仍是 ${fmt(y, 3)}）：只用它自己的數字計算`; }
      } else if (swapped) {
        status.className = "status status--info"; status.textContent = "已換掉資料 2〜4；先點資料 1 的某一格再換，就能比較前後";
      } else {
        status.className = "status status--info";
        status.textContent = `這一組 ${st.vals.length} 個數：平均 ${fmt(st.mu, 2)}、標準差 ${fmt(Math.sqrt(st.v2), 2)}；正規化後平均 0、標準差 1`;
      }
      const list = st.vals.map((v) => (v < 0 ? `(${fmt(v, 0)})` : String(v)));
      const shown = list.length > 16 ? list.slice(0, 16).join(" + ") + " + …" : list.join(" + ");
      calc.innerHTML =
        `<span class="calc-row"><span class="k">選中</span> 資料 ${sel.n + 1}、ch${sel.c + 1} 的 x = <b>${fmt(x, 0)}</b>　<span class="k">同組共</span> ${st.vals.length} 個數</span>` +
        `<span class="calc-row"><span class="k">平均</span> μ = (${shown}) ÷ ${st.vals.length} = <b>${fmt(st.mu, 3)}</b></span>` +
        `<span class="calc-row"><span class="k">變異數</span> σ² = 平均((x − μ)²) = <b>${fmt(st.v2, 3)}</b>　<span class="k">標準差</span> σ = ${fmt(Math.sqrt(st.v2), 3)}</span>` +
        `<span class="calc-row"><span class="k">正規化</span> (x − μ) ÷ √(σ² + ε) = (${fmt(x, 0)} − ${fmtP(st.mu, 3)}) ÷ ${fmt(Math.sqrt(st.v2 + EPS), 3)} = <span class="hl">${fmt(y, 3)}</span></span>` +
        `<span class="calc-row"><span class="k">最後</span> γ × ${fmt(y, 3)} + β　<span class="k">（γ、β 是可學習的參數，這裡 γ = 1、β = 0）</span></span>`;
    }

    note.innerHTML = `ε 是很小的數（這裡 0.00001），避免除以 0。正規化後再乘 γ、加 β，讓模型需要時能把尺度調回來，所以不會失去表現能力。バッチ正規化在推論時通常一次只來一筆資料，沒有「同一批」可算，所以改用訓練時累積的平均與變異數（移動平均）。原始論文說它解決的是<span class="jp">内部共変量シフト</span>（內部共變量偏移，<span class="en">internal covariate shift</span>）：前面的層一更新，後面的層收到的數值分布就跟著變。注意日文的「正規化」（標準化，normalization）和<a href="#overfit">正則化</a>（regularization）是不同的東西；シラバス也把這一節從「正則化層」改名為「正規化層」。インスタンス正規化常用在風格轉換，グループ正規化常用在批次只能放幾張大圖的物體偵測。數字是示範用的整數，每張特徴マップ只有 2×2。`;

    render();
    ctx.onTheme(() => render());
    return {};
  },
});
