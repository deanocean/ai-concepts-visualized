/* モーメンタムと Adam（シラバス §16）
   地形：L(x, y) = (x² + k·y²) ÷ 2（k = 10 時和勾配降下法頁的「細長山谷」相同），起點 (−5, 2.4)。
   五種方法同時走：SGD（這裡梯度用整個地形算，等於勾配降下法）、モーメンタム、AdaGrad、RMSprop、Adam。
   調 k（y 方向的陡度）時，AdaGrad／RMSprop／Adam 的路徑完全不變（每個座標的梯度同乘 k，分子分母一起放大），
   SGD 和モーメンタム則會震盪甚至發散。所有數字即時計算。 */
Atlas.define("optim", {
  lead: `<p>在<a href="#gd">勾配降下法</a>的細長山谷裡，陡的方向來回彈、緩的方向走很慢，而且整個模型只有一個學習率：調大，陡的方向會發散；調小，緩的方向更慢。改良的方向有兩個：<span class="jp">モーメンタム</span>（動量法，<span class="en">momentum</span>）讓移動帶有慣性；AdaGrad、RMSprop 讓<b>每個參數</b>依自己過去的梯度大小自動調步伐。<span class="jp">Adam</span> 把兩者合在一起。</p>`,
  closing: `モーメンタム靠慣性：來回彈的方向互相抵消、一直同向的方向越走越快。AdaGrad、RMSprop、Adam 把每個參數的梯度除以「它過去有多大」，所以不管某個方向多陡，步伐都差不多大；把山谷調陡，它們的路徑完全不變。AdaGrad 累加全部歷史，步伐只會越來越小；RMSprop 改成只記得最近，Adam 再加上慣性，是現在最常用的預設選擇。`,

  mount(bench, ctx) {
    const { h, fmt, fmtP, ui } = Atlas;
    const KS = [1, 3, 10, 30, 100];
    const MAX = 150, EPS = 1e-8, NEAR = 0.05;
    const METHODS = [
      { id: "sgd", name: "SGD", color: "--muted", eta: 0.18, range: [0.005, 0.25, 0.005] },
      { id: "mom", name: "モーメンタム", color: "--pos", eta: 0.03, range: [0.005, 0.1, 0.005], beta: 0.9 },
      { id: "ada", name: "AdaGrad", color: "--ok", eta: 1.0, range: [0.1, 3, 0.1] },
      { id: "rms", name: "RMSprop", color: "--hot", eta: 0.1, range: [0.02, 0.5, 0.02], rho: 0.9 },
      { id: "adam", name: "Adam", color: "--neg", eta: 0.3, range: [0.05, 1, 0.05], b1: 0.9, b2: 0.999 },
    ];
    let ki = 2, start = [-5, 2.4], focus = "adam", onlyFocus = false;
    const etas = Object.fromEntries(METHODS.map((m) => [m.id, m.eta]));
    const k = () => KS[ki];
    const f = (x, y) => (x * x + k() * y * y) / 2;
    const g = (x, y) => [x, k() * y];
    const BOX = [-6, 6, -3, 3];

    /* ---------- 每種方法的狀態與一步 ---------- */
    let runs = {};
    function fresh(m) { return { m, pos: [start.slice()], v: [0, 0], s: [0, 0], mm: [0, 0], t: 0, dead: false, reach: null }; }
    /** 計算下一步，回傳中間值（給算式框用）；apply = true 時才真的更新 */
    function stepOf(r, apply) {
      const m = r.m, p = r.pos[r.pos.length - 1], gr = g(p[0], p[1]), eta = etas[m.id];
      const out = { g: gr, next: [0, 0], parts: [] };
      const v = r.v.slice(), sq = r.s.slice(), mm = r.mm.slice(), t = r.t + 1;
      for (let i = 0; i < 2; i++) {
        let d;
        if (m.id === "sgd") { d = -eta * gr[i]; out.parts.push({}); }
        else if (m.id === "mom") { v[i] = m.beta * v[i] - eta * gr[i]; d = v[i]; out.parts.push({ vOld: r.v[i], v: v[i] }); }
        else if (m.id === "ada") { sq[i] += gr[i] ** 2; d = (-eta * gr[i]) / (Math.sqrt(sq[i]) + EPS); out.parts.push({ hOld: r.s[i], h: sq[i], eff: eta / (Math.sqrt(sq[i]) + EPS) }); }
        else if (m.id === "rms") { sq[i] = m.rho * sq[i] + (1 - m.rho) * gr[i] ** 2; d = (-eta * gr[i]) / (Math.sqrt(sq[i]) + EPS); out.parts.push({ hOld: r.s[i], h: sq[i], eff: eta / (Math.sqrt(sq[i]) + EPS) }); }
        else {
          mm[i] = m.b1 * mm[i] + (1 - m.b1) * gr[i]; sq[i] = m.b2 * sq[i] + (1 - m.b2) * gr[i] ** 2;
          const mh = mm[i] / (1 - m.b1 ** t), vh = sq[i] / (1 - m.b2 ** t);
          d = (-eta * mh) / (Math.sqrt(vh) + EPS);
          out.parts.push({ mOld: r.mm[i], m: mm[i], vOld: r.s[i], v: sq[i], mh, vh });
        }
        out.next[i] = p[i] + d;
        out.parts[i].d = d;
      }
      if (apply) {
        r.v = v; r.s = sq; r.mm = mm; r.t = t; r.pos.push(out.next);
        const q = out.next;
        if (!isFinite(q[0]) || !isFinite(q[1]) || Math.abs(q[0]) > 1e4 || Math.abs(q[1]) > 1e4) r.dead = true;
        else if (r.reach == null && Math.hypot(q[0], q[1]) < NEAR) r.reach = r.t;
      }
      return out;
    }
    const nStep = () => runs.sgd.pos.length - 1;
    function step() {
      if (nStep() >= MAX) return false;
      for (const m of METHODS) { const r = runs[m.id]; if (!r.dead) stepOf(r, true); else r.pos.push(r.pos[r.pos.length - 1]); }
      draw(); text();
      return nStep() < MAX;
    }
    function reset() { runs = Object.fromEntries(METHODS.map((m) => [m.id, fresh(m)])); draw(); text(); }

    /* ---------- 控制 ---------- */
    bench.append(h("style", { text: `
      .op-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
      .op-tbl td.l { font-family: var(--f-body); }
      .op-tbl tr.is-sel td { background: var(--hot-soft); }
      .op-tbl td.bad { color: var(--bad); font-weight: 700; }
      .op-tbl td.ok { color: var(--ok); font-weight: 700; }
    ` }));
    const focusSeg = ui.seg({ label: "細看哪一個（調學習率、看算式）", value: focus,
      options: METHODS.map((m) => ({ value: m.id, label: m.name })),
      onChange: (v) => { focus = v; syncEta(); draw(); text(); } });
    const kSlider = ui.slider({ id: "op-k", label: "y 方向比 x 方向陡幾倍 k", min: 0, max: KS.length - 1, step: 1, value: ki, format: (v) => `k = ${KS[v]}`, onInput: (v) => { ki = v; player.pause(); renderBg(); reset(); } });
    const etaSlider = ui.slider({ id: "op-eta", label: "学習率 η（細看的那一個）", min: 0.005, max: 1, step: 0.005, value: etas[focus], format: (v) => String(Math.round(v * 1000) / 1000), onInput: (v) => { etas[focus] = v; player.pause(); reset(); } });
    function syncEta() {
      const m = METHODS.find((x) => x.id === focus), inp = bench.querySelector("#op-eta");
      inp.min = m.range[0]; inp.max = m.range[1]; inp.step = m.range[2]; etaSlider.set(etas[focus]);
    }
    const onlyBtn = ui.btn("只顯示細看的那一個", () => { onlyFocus = !onlyFocus; onlyBtn.textContent = onlyFocus ? "五個都顯示" : "只顯示細看的那一個"; draw(); }, { sm: true });
    const defBtn = ui.btn("學習率全部還原", () => { METHODS.forEach((m) => (etas[m.id] = m.eta)); syncEta(); player.pause(); reset(); }, { sm: true });

    const wrap = h("div", { style: { position: "relative", width: "100%", maxWidth: "760px", margin: "0 auto" } });
    const canvas = h("canvas", { role: "img", "aria-label": "五種最佳化方法在等高線上的路徑", style: { display: "block", width: "100%", cursor: "crosshair", borderRadius: "8px", border: "1px solid var(--line)" } });
    wrap.append(canvas);
    const tbl = h("table", { class: "tbl op-tbl" });
    const player = ui.player(ctx, () => step(), 110);
    const status = h("div", { class: "status" });
    const calc = h("div", { class: "calc" });
    const chart = Atlas.s("svg", { viewBox: "0 0 640 200", role: "img", "aria-label": "離谷底的距離隨步數的變化", style: "display:block;width:100%;min-width:520px;max-width:760px" });
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, focusSeg.el),
      h("div", { class: "bench__row" }, kSlider.el, etaSlider.el),
      h("div", { class: "scroll-x" }, wrap),
      status,
      h("div", { class: "bench__row" }, ui.btn("走一步", () => { player.pause(); step(); }, { primary: true }), player.el, ui.btn("重設", () => { player.pause(); reset(); }), onlyBtn, defBtn),
      h("div", { class: "scroll-x" }, tbl),
      calc,
      h("div", null, h("p", { class: "field__label" }, "離谷底的距離（對數刻度）"), h("div", { class: "scroll-x" }, chart)),
      note,
    );

    /* ---------- 等高線（和勾配降下法頁同一種畫法） ---------- */
    let bg = null, cssW = 0, cssH = 0, dpr = 1;
    const tx = (x) => ((x - BOX[0]) / (BOX[1] - BOX[0])) * cssW;
    const ty = (y) => cssH - ((y - BOX[2]) / (BOX[3] - BOX[2])) * cssH;
    function sizeCanvas() {
      cssW = Math.max(300, Math.min(760, wrap.clientWidth || 640));
      cssH = Math.round(cssW / 2);
      dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
      canvas.style.height = cssH + "px";
      renderBg();
    }
    function renderBg() {
      const Wp = canvas.width, Hp = canvas.height, st = 2;
      if (!Wp) return;
      const gw = Math.ceil(Wp / st), gh = Math.ceil(Hp / st);
      const vals = new Float64Array(gw * gh); let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const x = BOX[0] + ((i + 0.5) / gw) * (BOX[1] - BOX[0]), y = BOX[3] - ((j + 0.5) / gh) * (BOX[3] - BOX[2]);
        const v = f(x, y); vals[j * gw + i] = v; if (v < lo) lo = v; if (v > hi) hi = v;
      }
      const BANDS = 14, band = new Uint8Array(gw * gh), lg = Math.log(1 + hi - lo);
      for (let q = 0; q < vals.length; q++) band[q] = Math.max(0, Math.min(BANDS - 1, Math.floor((Math.log(1 + Math.max(0, vals[q] - lo)) / lg) * BANDS) || 0));
      const lo_c = Atlas.color("--seq-lo"), hi_c = Atlas.color("--seq-hi");
      const pal = []; for (let b = 0; b < BANDS; b++) pal.push(Atlas.mix(lo_c, hi_c, (b / (BANDS - 1)) * 0.78));
      const lineC = Atlas.mix(lo_c, hi_c, 0.95);
      const off = document.createElement("canvas"); off.width = gw; off.height = gh;
      const octx = off.getContext("2d"), img = octx.createImageData(gw, gh);
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const q = j * gw + i, b = band[q], p = pal[b];
        const edge = (i + 1 < gw && band[q + 1] !== b) || (j + 1 < gh && band[q + gw] !== b);
        for (let c = 0; c < 3; c++) img.data[q * 4 + c] = edge ? Math.round(p[c] * 0.45 + lineC[c] * 0.55) : p[c];
        img.data[q * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      bg = off; draw();
    }
    function draw() {
      if (!bg || !runs.sgd) return;
      const c = canvas.getContext("2d");
      c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(bg, 0, 0, canvas.width, canvas.height);
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      const panel = Atlas.color("--panel"), ink = Atlas.color("--ink"), font = getComputedStyle(document.body).fontFamily;
      // 谷底
      c.beginPath(); c.arc(tx(0), ty(0), 6, 0, Math.PI * 2); c.fillStyle = Atlas.color("--ok"); c.fill(); c.lineWidth = 2; c.strokeStyle = panel; c.stroke();
      // 起點
      c.beginPath(); c.arc(tx(start[0]), ty(start[1]), 9, 0, Math.PI * 2); c.lineWidth = 1.5; c.strokeStyle = ink; c.setLineDash([3, 3]); c.stroke(); c.setLineDash([]);
      const order = METHODS.filter((m) => m.id !== focus).concat(METHODS.filter((m) => m.id === focus));
      const labels = [];
      for (const m of order) {
        if (onlyFocus && m.id !== focus) continue;
        const r = runs[m.id], col = Atlas.color(m.color), sel = m.id === focus;
        const pts = r.pos.filter((p) => isFinite(p[0]) && isFinite(p[1]) && Math.abs(p[0]) < 1e3 && Math.abs(p[1]) < 1e3);
        c.globalAlpha = sel || onlyFocus ? 1 : 0.85;
        c.lineWidth = sel ? 3 : 1.8; c.strokeStyle = col; c.lineJoin = "round"; c.beginPath();
        pts.forEach((p, i) => (i ? c.lineTo(tx(p[0]), ty(p[1])) : c.moveTo(tx(p[0]), ty(p[1]))));
        c.stroke();
        const last = pts[pts.length - 1];
        if (last && !r.dead) {
          c.beginPath(); c.arc(tx(last[0]), ty(last[1]), sel ? 6.5 : 4.5, 0, Math.PI * 2); c.fillStyle = col; c.fill(); c.lineWidth = 2; c.strokeStyle = panel; c.stroke();
          if (nStep() > 0) labels.push({ x: tx(last[0]), y: ty(last[1]), text: m.name, col, sel });
        }
        c.globalAlpha = 1;
      }
      // 各方法的名字：放在目前位置右上，彼此太近就往下排
      c.font = `700 12px ${font}`; c.textAlign = "left";
      const used = [];
      labels.sort((a, b) => a.y - b.y).forEach((L) => {
        let X = L.x + 9, Y = L.y - 8;
        const w = c.measureText(L.text).width;
        if (X + w > cssW - 4) X = L.x - 9 - w;
        while (used.some((u) => Math.abs(u.y - Y) < 15 && X < u.x + u.w + 4 && u.x < X + w + 4)) Y += 15;
        used.push({ x: X, y: Y, w });
        c.lineWidth = 4; c.strokeStyle = panel; c.strokeText(L.text, X, Y); c.fillStyle = L.col; c.fillText(L.text, X, Y);
      });
      c.font = `12px ${font}`; c.fillStyle = ink; c.lineWidth = 4; c.strokeStyle = panel;
      const ft = `L(x, y) = (x² + ${k()}y²) ÷ 2`;
      c.strokeText(ft, 8, cssH - 10); c.fillText(ft, 8, cssH - 10);
      if (nStep() === 0) { c.textAlign = "right"; c.strokeText("點圖可改變起點", cssW - 8, cssH - 10); c.fillText("點圖可改變起點", cssW - 8, cssH - 10); }
      drawChart();
    }
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      start = [Math.round((BOX[0] + ((e.clientX - r.left) / r.width) * (BOX[1] - BOX[0])) * 10) / 10,
               Math.round((BOX[3] - ((e.clientY - r.top) / r.height) * (BOX[3] - BOX[2])) * 10) / 10];
      player.pause(); reset();
    });

    /* ---------- 距離圖 ---------- */
    function drawChart() {
      const { s } = Atlas, W = 640, H = 200, M = { l: 46, r: 16, t: 10, b: 28 };
      const LO = -3, HI = 1.5;
      const px = (i) => M.l + (i / MAX) * (W - M.l - M.r);
      const py = (d) => M.t + ((HI - Math.log10(Math.max(1e-3, Math.min(d, 10 ** HI)))) / (HI - LO)) * (H - M.t - M.b);
      chart.replaceChildren();
      for (let e = 1; e >= -3; e--) {
        chart.append(s("line", { x1: M.l, x2: W - M.r, y1: py(10 ** e), y2: py(10 ** e), stroke: "var(--line)" }));
        chart.append(s("text", { x: M.l - 6, y: py(10 ** e) + 4, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: e >= 0 ? String(10 ** e) : "0." + "0".repeat(-e - 1) + "1" }));
      }
      chart.append(s("line", { x1: M.l, x2: W - M.r, y1: py(NEAR), y2: py(NEAR), stroke: "var(--ok)", "stroke-dasharray": "5 4" }));
      chart.append(s("text", { x: W - M.r, y: py(NEAR) - 4, "text-anchor": "end", "font-size": 11, fill: "var(--ok)", class: "t-halo", text: `距離 ${NEAR}（算「到達」）` }));
      for (let i = 0; i <= MAX; i += 25) chart.append(s("text", { x: px(i), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, class: "t-mono t-muted", text: i }));
      chart.append(s("text", { x: W - M.r, y: H - 2, "text-anchor": "end", "font-size": 11, class: "t-muted", text: "步數" }));
      for (const m of METHODS.filter((x) => x.id !== focus).concat(METHODS.filter((x) => x.id === focus))) {
        if (onlyFocus && m.id !== focus) continue;
        const r = runs[m.id];
        const d = r.pos.map((p, i) => (i ? "L" : "M") + px(i).toFixed(1) + "," + py(isFinite(p[0]) && isFinite(p[1]) ? Math.hypot(p[0], p[1]) : 1e9).toFixed(1)).join("");
        chart.append(s("path", { d, fill: "none", stroke: `var(${m.color})`, "stroke-width": m.id === focus ? 2.8 : 1.5 }));
      }
    }

    /* ---------- 文字 ---------- */
    function text() {
      const n = nStep();
      // 比較表
      tbl.replaceChildren(
        h("thead", null, h("tr", null, h("th", { class: "l" }, "方法"), h("th", null, "學習率 η"), h("th", null, `到達谷底附近（距離 < ${NEAR}）`), h("th", null, "目前離谷底"))),
        h("tbody", null, ...METHODS.map((m) => {
          const r = runs[m.id], p = r.pos[r.pos.length - 1], d = Math.hypot(p[0], p[1]);
          return h("tr", { class: m.id === focus ? "is-sel" : null },
            h("td", { class: "l" }, h("span", { class: "op-dot", style: { background: `var(${m.color})` } }), m.name),
            h("td", null, String(etas[m.id])),
            h("td", { class: r.dead ? "bad" : r.reach != null ? "ok" : null }, r.dead ? `發散（第 ${r.t} 步）` : r.reach != null ? `第 ${r.reach} 步` : n ? "還沒" : "—"),
            h("td", null, r.dead ? "—" : fmt(d, 3)));
        })));
      // 狀態
      const dead = METHODS.filter((m) => runs[m.id].dead).map((m) => m.name);
      if (n === 0) { status.className = "status status--info"; status.textContent = `按「走一步」或「自動播放」：五種方法從同一點同時出發（最多 ${MAX} 步）`; }
      else if (dead.length) { status.className = "status status--bad"; status.textContent = `${dead.join("、")} 發散了：y 方向太陡，同樣的學習率一步就跨得比谷寬還遠。把 k 調回 10 試試`; }
      else if (n >= MAX) { status.className = "status status--info"; status.textContent = `已走完 ${MAX} 步`; }
      else if (k() <= 10 && runs.sgd.reach != null && METHODS.every((m) => m.id === "sgd" || runs[m.id].reach == null || runs[m.id].reach >= runs.sgd.reach)) {
        status.className = "status status--warn"; status.textContent = `第 ${n} 步：SGD 的學習率 ${etas.sgd} 剛好是替這個陡度調好的，所以最快到；把 k 調大，看誰撐得住`;
      }
      else { status.className = "status status--info"; status.textContent = `第 ${n} 步`; }
      // 算式框
      const r = runs[focus], m = r.m, p = r.pos[r.pos.length - 1];
      if (r.dead) { calc.innerHTML = `<span class="k">${m.name} 已經發散，位置超出計算範圍</span>`; return; }
      const o = stepOf(r, false), eta = etas[focus], X = ["x", "y"];
      const rows = [`<span class="calc-row"><span class="k">${m.name}　第 ${n} 步　位置</span> (${fmt(p[0], 3)}, ${fmt(p[1], 3)})　<span class="k">梯度</span> g = (x, ${k()}y) = (${fmt(o.g[0], 3)}, ${fmt(o.g[1], 3)})</span>`];
      for (let i = 0; i < 2; i++) {
        const q = o.parts[i], gi = fmtP(o.g[i], 3), c = X[i];
        let t;
        if (m.id === "sgd") t = `${c}：移動 = −η·g = −${eta}×${gi} = <span class="hl">${fmt(q.d, 3, true)}</span>`;
        else if (m.id === "mom") t = `${c}：v = β·v − η·g = ${m.beta}×${fmtP(q.vOld, 3)} − ${eta}×${gi} = <span class="hl">${fmt(q.v, 3)}</span>（移動量就是 v）`;
        else if (m.id === "ada") t = `${c}：h = h + g² = ${fmt(q.hOld, 3)} + ${gi}² = ${fmt(q.h, 3)}　有效學習率 η÷√h = ${eta}÷${fmt(Math.sqrt(q.h), 3)} = ${fmt(q.eff, 4)}　移動 = −${fmt(q.eff, 4)}×${gi} = <span class="hl">${fmt(q.d, 3, true)}</span>`;
        else if (m.id === "rms") t = `${c}：h = ρ·h + (1 − ρ)·g² = ${m.rho}×${fmt(q.hOld, 3)} + ${fmt(1 - m.rho, 1)}×${gi}² = ${fmt(q.h, 3)}　有效學習率 η÷√h = ${fmt(q.eff, 4)}　移動 = <span class="hl">${fmt(q.d, 3, true)}</span>`;
        else t = `${c}：m = ${m.b1}·m + ${fmt(1 - m.b1, 1)}·g = ${fmt(q.m, 3)}，v = ${m.b2}·v + ${fmt(1 - m.b2, 3)}·g² = ${fmt(q.v, 4)}；修正後 m̂ = ${fmt(q.mh, 3)}，v̂ = ${fmt(q.vh, 3)}；移動 = −η·m̂÷√v̂ = −${eta}×${fmtP(q.mh, 3)}÷${fmt(Math.sqrt(q.vh), 3)} = <span class="hl">${fmt(q.d, 3, true)}</span>`;
        rows.push(`<span class="calc-row">${t}</span>`);
      }
      rows.push(`<span class="calc-row"><span class="k">下一個位置</span> (<b>${fmt(o.next[0], 3)}</b>, <b>${fmt(o.next[1], 3)}</b>)</span>`);
      calc.innerHTML = rows.join("");
    }

    note.innerHTML = `這裡的地形是固定的式子，梯度每次都用整個地形算，所以「SGD」在這裡就是<a href="#gd">勾配降下法</a>；實際訓練時這五種方法都搭配<a href="#sgd">ミニバッチ</a>使用。モーメンタム的慣性也會讓它衝過谷底再折回，所以路徑常繞一大圈才停下。各方法的學習率不同，是因為它們的「一步」定義不同：AdaGrad、RMSprop、Adam 的移動量大約就是 η 本身。β、ρ 用常見的預設值（モーメンタム β = 0.9；RMSprop ρ = 0.9；Adam β₁ = 0.9、β₂ = 0.999），Adam 的 m̂、v̂ 是補正剛開始時 m、v 從 0 起算而偏小的部分。其他改良：<span class="jp">AdaDelta</span> 和 RMSprop 一樣改用最近的梯度，並且連學習率都不用設定；<span class="jp">AdaBound</span>（和它的變形 <span class="jp">AMSBound</span>）替 Adam 的有效學習率加上逐漸收窄的上下限，訓練後期變得像 SGD。學習率、β 這類訓練前由人決定的值叫<span class="jp">ハイパーパラメータ</span>（超參數，<span class="en">hyperparameter</span>），常用<span class="jp">グリッドサーチ</span>（把候選值的組合全部試一遍）或<span class="jp">ランダムサーチ</span>（隨機抽組合來試）尋找。k = 10 時 SGD 最快、k 一調大卻發散，這種「沒有一種方法在所有問題上都最好」的事實叫<span class="jp">ノーフリーランチの定理</span>（沒有免費午餐定理，<span class="en">no free lunch theorem</span>）：方法的優劣要看問題而定。`;

    syncEta();
    reset();
    sizeCanvas();
    ctx.onTheme(() => renderBg());
    let rT = null;
    const ro = new ResizeObserver(() => { clearTimeout(rT); rT = setTimeout(() => { if (Math.max(300, Math.min(760, wrap.clientWidth || 640)) !== cssW) sizeCanvas(); }, 120); });
    ro.observe(wrap);
    return { destroy() { ro.disconnect(); player.pause(); } };
  },
});
