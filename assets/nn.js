/* =========================================================
   小型神經網路工具（生成模型各頁共用，頁面上即時訓練用）
   - Atlas.nn.rng(seed)            固定種子的亂數（0〜1）
   - Atlas.nn.gauss(r)             標準常態亂數
   - new Atlas.nn.MLP(sizes, acts, r)
       sizes = [輸入, 隱藏…, 輸出]；acts = 每層的活化函數（"relu" | "lrelu" | "tanh" | "sigmoid" | "lin"）
       forward(x) → 輸出（Float64Array，下次 forward 會被覆寫）
       backward(dOut) → 對輸入的梯度；權重梯度累加在 gW / gb
       zero()、adam(lr)、snapshot() / restore(s)
   一次只處理一筆資料：迷你批次＝對批內每筆 forward → backward 累加梯度，再 adam 一次。
   ========================================================= */
(function (root) {
  "use strict";
  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const ACT = {
    relu: [(z) => (z > 0 ? z : 0), (z, a) => (z > 0 ? 1 : 0)],
    lrelu: [(z) => (z > 0 ? z : 0.1 * z), (z, a) => (z > 0 ? 1 : 0.1)],
    tanh: [(z) => Math.tanh(z), (z, a) => 1 - a * a],
    sigmoid: [(z) => 1 / (1 + Math.exp(-z)), (z, a) => a * (1 - a)],
    lin: [(z) => z, () => 1],
  };

  class MLP {
    constructor(sizes, acts, r) {
      this.sizes = sizes; this.acts = acts; this.L = sizes.length - 1;
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      this.mW = []; this.vW = []; this.mb = []; this.vb = [];
      this.z = []; this.a = [new Float64Array(sizes[0])]; this.t = 0;
      for (let l = 0; l < this.L; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const sd = Math.sqrt((acts[l] === "relu" || acts[l] === "lrelu" ? 2 : 1) / nin);
        const W = new Float64Array(nin * nout);
        for (let i = 0; i < W.length; i++) W[i] = gauss(r) * sd;
        this.W.push(W); this.b.push(new Float64Array(nout));
        for (const k of ["gW", "mW", "vW"]) this[k].push(new Float64Array(nin * nout));
        for (const k of ["gb", "mb", "vb"]) this[k].push(new Float64Array(nout));
        this.z.push(new Float64Array(nout)); this.a.push(new Float64Array(nout));
      }
      this.d = sizes.map((n) => new Float64Array(n));
    }
    forward(x) {
      const a0 = this.a[0]; for (let i = 0; i < a0.length; i++) a0[i] = x[i];
      for (let l = 0; l < this.L; l++) {
        const nin = this.sizes[l], nout = this.sizes[l + 1], W = this.W[l], b = this.b[l], ai = this.a[l], z = this.z[l], ao = this.a[l + 1], f = ACT[this.acts[l]][0];
        for (let o = 0; o < nout; o++) {
          let s = b[o]; const off = o * nin;
          for (let i = 0; i < nin; i++) s += W[off + i] * ai[i];
          z[o] = s; ao[o] = f(s);
        }
      }
      return this.a[this.L];
    }
    /** dOut：損失對「輸出（活化後）」的梯度；若 preAct 為 true，則是對最後一層 z 的梯度（例如 sigmoid＋交差エントロピー） */
    backward(dOut, preAct = false, accum = true) {
      let d = this.d[this.L];
      for (let i = 0; i < d.length; i++) d[i] = dOut[i];
      for (let l = this.L - 1; l >= 0; l--) {
        const nin = this.sizes[l], nout = this.sizes[l + 1], W = this.W[l], gW = this.gW[l], gb = this.gb[l], ai = this.a[l], z = this.z[l], ao = this.a[l + 1];
        const df = ACT[this.acts[l]][1], dz = this.d[l + 1], dn = this.d[l];
        if (!(preAct && l === this.L - 1)) for (let o = 0; o < nout; o++) dz[o] *= df(z[o], ao[o]);
        dn.fill(0);
        for (let o = 0; o < nout; o++) {
          const g = dz[o]; if (g === 0) continue;
          const off = o * nin;
          if (accum) { gb[o] += g; for (let i = 0; i < nin; i++) { gW[off + i] += g * ai[i]; dn[i] += g * W[off + i]; } }
          else for (let i = 0; i < nin; i++) dn[i] += g * W[off + i];
        }
      }
      return this.d[0];
    }
    zero() { for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); } }
    adam(lr, scale = 1, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      this.t++;
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      for (let l = 0; l < this.L; l++) {
        for (const [P, G, M, V] of [[this.W[l], this.gW[l], this.mW[l], this.vW[l]], [this.b[l], this.gb[l], this.mb[l], this.vb[l]]]) {
          for (let i = 0; i < P.length; i++) {
            const g = G[i] * scale;
            M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g;
            P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + eps);
          }
        }
      }
      this.zero();
    }
    snapshot() { return { W: this.W.map((w) => w.slice()), b: this.b.map((w) => w.slice()) }; }
    restore(sn) { sn.W.forEach((w, l) => this.W[l].set(w)); sn.b.forEach((w, l) => this.b[l].set(w)); }
    get nParams() { return this.W.reduce((a, w) => a + w.length, 0) + this.b.reduce((a, w) => a + w.length, 0); }
  }

  /** 分段訓練：每 30ms 做一小段，畫面不會卡住。run(已完成數, 這次要做的數)；fastForward 時每段盡量做到 40ms */
  function loop(ctx, { total, perTick, run, onTick, onEnd }) {
    let done = 0, timer = null, fast = false;
    function tick() {
      const t0 = performance.now();
      do { const n = Math.min(perTick, total - done); run(done, n); done += n; }
      while (fast && done < total && performance.now() - t0 < 40);
      const end = done >= total;
      if (end) api.pause();
      onTick && onTick(done);
      if (end) onEnd && onEnd();
    }
    const api = {
      get done() { return done; }, get total() { return total; },
      get running() { return !!timer; }, get finished() { return done >= total; },
      start() { fast = false; if (!timer && done < total) timer = ctx.every(30, tick); },
      fastForward() { fast = true; if (!timer && done < total) timer = ctx.every(30, tick); },
      pause() { if (timer) { timer.stop(); timer = null; } fast = false; },
    };
    return api;
  }

  /** 高解析度 canvas：CSS 寬度 100%、高度依比例；toLocal(e) 把滑鼠位置換成畫布座標 */
  function canvas(w, h, label) {
    const dpr = Math.min(3, (typeof devicePixelRatio !== "undefined" && devicePixelRatio) || 1);
    const el = document.createElement("canvas");
    el.width = Math.round(w * dpr); el.height = Math.round(h * dpr);
    el.style.width = "100%"; el.style.height = "auto"; el.style.display = "block";
    el.setAttribute("role", "img"); if (label) el.setAttribute("aria-label", label);
    const g = el.getContext("2d");
    return {
      el, g, w, h,
      begin() { g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); return g; },
      toLocal(e) { const r = el.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h]; },
    };
  }

  /** 7×7 長方形資料：寬 1〜7 × 高 1〜7，置中，共 49 張（AE、VAE 共用）。索引 = (高−1)×7 + (寬−1) */
  function rects() {
    const D = [], F = [];
    for (let hh = 1; hh <= 7; hh++) for (let ww = 1; ww <= 7; ww++) {
      const a = new Float64Array(49), t = Math.floor((7 - hh) / 2), l = Math.floor((7 - ww) / 2);
      for (let r = t; r < t + hh; r++) for (let c = l; c < l + ww; c++) a[r * 7 + c] = 1;
      D.push(a); F.push([ww, hh]);
    }
    return { D, F };
  }

  /** 在 canvas 上畫一張 7×7 的圖（值 0〜1：0＝off 色、1＝on 色）。gap＝格子間距 */
  function drawImg(g, x, y, size, pix, off, on, gap = 0) {
    const c = size / 7;
    for (let i = 0; i < 49; i++) {
      const v = Math.max(0, Math.min(1, pix[i]));
      g.fillStyle = `rgb(${Math.round(off[0] + (on[0] - off[0]) * v)},${Math.round(off[1] + (on[1] - off[1]) * v)},${Math.round(off[2] + (on[2] - off[2]) * v)})`;
      g.fillRect(x + (i % 7) * c + gap / 2, y + Math.floor(i / 7) * c + gap / 2, c - gap, c - gap);
    }
  }

  /** 和 49 張訓練圖比對：四捨五入後差幾格，回傳最接近的一張 */
  function nearest(D, y) {
    let best = 99, bi = -1;
    for (let j = 0; j < D.length; j++) {
      let d = 0; const x = D[j];
      for (let i = 0; i < 49; i++) if ((y[i] > 0.5) !== (x[i] > 0.5)) d++;
      if (d < best) { best = d; bi = j; }
    }
    return { i: bi, d: best };
  }

  /** 交差エントロピー（49 格加總）；y 夾在 1e−7〜1−1e−7 */
  function bce(x, y) {
    let s = 0;
    for (let i = 0; i < x.length; i++) { const p = Math.min(1 - 1e-7, Math.max(1e-7, y[i])); s -= x[i] * Math.log(p) + (1 - x[i]) * Math.log(1 - p); }
    return s;
  }

  /** SVG 折線圖（損失曲線等）。series = [{ pts: [[x, y]…], color, dash, label }]；log＝對數縱軸 */
  function lineChart(svg, { W = 360, H = 190, xMax, yMin, yMax, log = false, series, xLabel = "", marker = null, refY = null }) {
    const S = root.Atlas.s, fmt = root.Atlas.fmt;
    const m = { l: 44, r: 12, t: 12, b: 30 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.replaceChildren();
    const f = log ? (v) => Math.log10(Math.max(v, yMin)) : (v) => v;
    const lo = f(yMin), hi = f(yMax);
    const px = (x) => m.l + (x / xMax) * (W - m.l - m.r), py = (v) => H - m.b - ((f(v) - lo) / (hi - lo || 1)) * (H - m.t - m.b);
    const ticks = [];
    if (log) for (let e = Math.ceil(lo - 1e-9); e <= Math.floor(hi + 1e-9); e++) ticks.push(Math.pow(10, e));
    else { const st = niceStep((yMax - yMin) / 4); for (let v = Math.ceil(yMin / st) * st; v <= yMax + 1e-9; v += st) ticks.push(+v.toFixed(10)); }
    ticks.forEach((v) => {
      svg.append(S("line", { x1: m.l, x2: W - m.r, y1: py(v), y2: py(v), stroke: "var(--line)", "stroke-dasharray": "3 3" }));
      svg.append(S("text", { x: m.l - 5, y: py(v) + 4, "text-anchor": "end", "font-size": 10, class: "t-mono t-muted", text: log ? (v >= 1 ? String(v) : v.toExponential(0).replace("e-", "e−")) : fmt(v, v % 1 ? 1 : 0) }));
    });
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--line-strong)" }));
    const xs = niceStep(xMax / 4);
    for (let x = 0; x <= xMax + 1e-9; x += xs) svg.append(S("text", { x: px(x), y: H - m.b + 14, "text-anchor": "middle", "font-size": 10, class: "t-mono t-muted", text: x }));
    svg.append(S("text", { x: W - m.r, y: H - 3, "text-anchor": "end", "font-size": 10.5, class: "t-muted", text: xLabel }));
    if (refY != null) svg.append(S("line", { x1: m.l, x2: W - m.r, y1: py(refY), y2: py(refY), stroke: "var(--ink)", "stroke-dasharray": "5 4", "stroke-opacity": 0.6 }));
    if (marker != null) svg.append(S("line", { x1: px(marker), x2: px(marker), y1: m.t, y2: H - m.b, stroke: "var(--hot)", "stroke-width": 1.5 }));
    series.forEach((sr) => {
      if (!sr.pts.length) return;
      svg.append(S("path", { d: sr.pts.map((p, i) => (i ? "L" : "M") + px(p[0]).toFixed(1) + "," + py(Math.min(Math.max(p[1], yMin), yMax)).toFixed(1)).join(""), fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-dasharray": sr.dash || null }));
    });
    return { px, py };
  }
  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw || 1))), r = raw / p;
    return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p;
  }

  const nn = { rng, gauss, MLP, loop, canvas, rects, drawImg, nearest, bce, lineChart };
  if (root.Atlas) root.Atlas.nn = nn;
  if (typeof module !== "undefined") module.exports = nn;
})(typeof window !== "undefined" ? window : globalThis);
