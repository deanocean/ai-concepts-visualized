/* バンディットと ε-greedy 方策・UCB 方策（シラバス §9）
   5 台拉霸機 A〜E，每次拉中獎（獎勵 1）的機率依序 0.20／0.40／0.55／0.30／0.75（代理人不知道）。一輪玩 2000 次。
   兩種方策都先把每台各拉 1 次（第 1〜5 次），之後：
   - ε-greedy：抽亂數 u，u < ε 就隨機挑一台（探索），否則挑估計值最高的（利用；同分隨機）
   - UCB：挑「估計值 + c×√(ln t ÷ n)」最大的一台（t＝已經玩的次數、n＝這台玩過的次數），c = 1
   估計值＝這台到目前的平均獎勵（增量更新 Q ← Q + (r − Q) ÷ n）。
   ① 一步步看：單一輪，種子 1〜20。② 比較：6 種方策各玩 200 輪（種子 1000〜1199），取平均。 */
(function () {
  const P = [0.2, 0.4, 0.55, 0.3, 0.75], NAMES = ["A", "B", "C", "D", "E"], K = 5, T = 2000, BEST = 4, C = 1;
  const EPS = [0, 0.01, 0.1, 0.3, 1];

  /** 一輪的模擬器：step() 拉一次並回傳這一步的紀錄 */
  function makeRun(strat, eps, seed) {
    const r = Atlas.nn.rng(seed);
    const R = { strat, eps, t: 0, n: Array(K).fill(0), q: Array(K).fill(0), tot: 0, bestN: 0, last: null, curve: [[0, 0]], picks: [] };
    R.step = () => {
      if (R.t >= T) return null;
      const t = R.t, rec = { t: t + 1, qBefore: R.q.slice(), nBefore: R.n.slice() };
      let a;
      if (t < K) { a = t; rec.why = "init"; }
      else if (strat === "ucb") {
        rec.bonus = R.n.map((n) => C * Math.sqrt(Math.log(t) / n));
        rec.score = R.q.map((q, i) => q + rec.bonus[i]);
        a = rec.score.indexOf(Math.max(...rec.score)); rec.why = "ucb";
      } else {
        const u = r(); rec.u = u;
        if (u < eps) { a = Math.floor(r() * K); rec.why = "explore"; }
        else {
          const m = Math.max(...R.q), cand = []; R.q.forEach((v, i) => v === m && cand.push(i));
          a = cand.length > 1 ? cand[Math.floor(r() * cand.length)] : cand[0]; rec.why = "exploit"; rec.ties = cand.length;
        }
      }
      const rew = r() < P[a] ? 1 : 0;
      R.n[a]++; R.q[a] += (rew - R.q[a]) / R.n[a];
      R.t++; R.tot += rew; if (a === BEST) R.bestN++;
      rec.a = a; rec.rew = rew; rec.qAfter = R.q[a];
      R.last = rec; R.picks.push(a);
      if (R.t % 10 === 0 || R.t === T) R.curve.push([R.t, R.bestN / R.t]);
      return rec;
    };
    return R;
  }
  /** 比較模式用：只算統計，不留紀錄 */
  function quickRun(strat, eps, seed, every) {
    const r = Atlas.nn.rng(seed), n = Array(K).fill(0), q = Array(K).fill(0), best = [];
    let tot = 0, bestN = 0, last200 = 0;
    for (let t = 0; t < T; t++) {
      let a;
      if (t < K) a = t;
      else if (strat === "ucb") { let bs = -1; for (let i = 0; i < K; i++) { const sc = q[i] + C * Math.sqrt(Math.log(t) / n[i]); if (sc > bs) { bs = sc; a = i; } } }
      else if (r() < eps) a = Math.floor(r() * K);
      else { const m = Math.max(...q), cand = []; q.forEach((v, i) => v === m && cand.push(i)); a = cand.length > 1 ? cand[Math.floor(r() * cand.length)] : cand[0]; }
      const rew = r() < P[a] ? 1 : 0; n[a]++; q[a] += (rew - q[a]) / n[a]; tot += rew;
      if (a === BEST) { bestN++; if (t >= T - 200) last200++; }
      if ((t + 1) % every === 0) best.push(bestN / (t + 1));
    }
    return { tot, bestN, last200, best };
  }

  Atlas.define("bandit", {
    lead: `<p>面前有 5 台拉霸機，各自的中獎機率不同，但你不知道哪台最好，只能邊玩邊猜。一直玩「目前看起來最好的」叫<span class="jp">活用</span>（利用，<span class="en">exploitation</span>），可能錯過真正更好的；一直換著試叫<span class="jp">探索</span>（<span class="en">exploration</span>），又會浪費次數在差的機台上。這種「邊試邊選」的問題叫<span class="jp">多腕バンディット問題</span>（多臂拉霸問題，<span class="en">multi-armed bandit</span>），解法統稱<span class="jp">バンディットアルゴリズム</span>。</p>`,
    closing: `只利用（ε = 0）的話，一開始運氣不好就可能一路困在較差的機台；ε 越大越不會困住，但永遠有 ε 的次數花在亂試上。<span class="jp">UCB 方策</span>改成替「試得少、還沒把握」的機台加分，試夠了加分自然變小，探索就跟著減少：在這組機台上，它平均拿到的總獎勵最多，200 輪裡也沒有一輪困住。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      let mode = "one", strat = "eps", eps = 0.1, seed = 1, run = null, timer = null, cmp = null;

      bench.append(h("style", { text: `
        .bd-fig { max-width: 520px; margin-inline: auto; width: 100%; }
        .bd-fig > svg { display: block; width: 100%; height: auto; }
        .bd-cap { font-size: 12.5px; color: var(--muted); text-align: center; margin-bottom: 4px; }
        .bd-key { display: flex; flex-wrap: wrap; gap: 4px 14px; justify-content: center; font-size: 12.5px; color: var(--muted); margin-top: 4px; }
        .bd-key i { display: inline-block; width: 14px; height: 10px; margin-right: 4px; vertical-align: -1px; }
        .bd-sw { display: inline-block; width: 18px; height: 0; border-top: 3px solid; vertical-align: 3px; margin-right: 4px; }
      ` }));

      /* ---------- 控制 ---------- */
      const modeSeg = ui.seg({ label: "看什麼", value: mode, options: [{ value: "one", label: "① 一步步看一輪" }, { value: "cmp", label: "② 比較 6 種方策（各 200 輪平均）" }], onChange: (v) => { mode = v; layout(); } });
      const stratSeg = ui.seg({ label: "方策", value: strat, options: [{ value: "eps", label: "ε-greedy 方策" }, { value: "ucb", label: "UCB 方策" }], onChange: (v) => { strat = v; reset(); } });
      const epsSeg = ui.seg({ label: "ε（亂試的機率）", value: eps, options: EPS.map((e) => ({ value: e, label: e === 0 ? "0（只利用）" : e === 1 ? "1（只亂試）" : String(e) })), onChange: (v) => { eps = v; reset(); } });
      const seedBtn = ui.btn("換一組亂數", () => { seed = (seed % 20) + 1; reset(); }, { sm: true });
      const playBtn = ui.btn("▶ 自動玩", () => (timer ? pause() : play()), { primary: true });
      const oneBtn = ui.btn("拉 1 次", () => { pause(); run.step(); render(); });
      const ffBtn = ui.btn("⏭ 快轉到 2000 次", () => { pause(); while (run.step()); render(); });
      const resetBtn = ui.btn("重設", () => reset());
      const cmpBtn = ui.btn("▶ 開始比較", () => startCmp(), { primary: true });

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const mSvg = s("svg", { viewBox: "0 0 366 262", role: "img", "aria-label": "5 台拉霸機的估計值" });
      const cSvg = s("svg", { role: "img", "aria-label": "選到最佳機台 E 的比例" });
      const cmpSvg = s("svg", { role: "img", "aria-label": "6 種方策選到最佳機台的比例（平均）" });
      const cmpTbl = h("div", { class: "scroll-x" });
      const keyOne = h("div", { class: "bd-key" });
      const cmpKey = h("div", { class: "bd-key" });

      const oneBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, stratSeg.el, epsSeg.el, seedBtn),
        h("div", { class: "bench__row" }, playBtn, oneBtn, ffBtn, resetBtn),
        status,
        h("div", { class: "bd-fig" }, mSvg, keyOne),
        calc,
        h("div", { class: "bd-fig" }, h("div", { class: "bd-cap" }, "到目前為止，選到最好的機台 E 的比例（虛線＝隨機亂選的 20%）"), cSvg));
      const cmpBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("div", { class: "bench__row" }, cmpBtn),
        h("div", { class: "status" }),
        h("div", { class: "bd-fig" }, h("div", { class: "bd-cap" }, "200 輪平均：到第 t 次為止，選到最好的機台 E 的比例"), cmpSvg, cmpKey),
        cmpTbl);
      const cmpStatus = cmpBox.children[1];
      bench.append(h("div", { class: "bench__row" }, modeSeg.el), oneBox, cmpBox, note);

      /* ---------- ① 一輪 ---------- */
      function pause() { if (timer) { timer.stop(); timer = null; } refreshBtns(); }
      function play() {
        if (run.t >= T) return;
        timer = ctx.every(50, () => {
          const k = run.t < 50 ? 1 : run.t < 300 ? 5 : 25;
          for (let i = 0; i < k; i++) run.step();
          if (run.t >= T) pause();
          render();
        });
        refreshBtns(); render();
      }
      function reset() { pause(); run = makeRun(strat, strat === "eps" ? eps : 0, seed); epsSeg.el.hidden = strat !== "eps"; render(); }
      function refreshBtns() {
        if (!run) return;
        playBtn.textContent = timer ? "❚❚ 暫停" : run.t >= T ? "✓ 玩完 2000 次" : "▶ 自動玩";
        playBtn.disabled = oneBtn.disabled = ffBtn.disabled = run.t >= T;
      }

      function drawMachines() {
        const R = run, L = R.last, svg = mSvg; svg.replaceChildren();
        const x0 = 32, cw = 66, top = 36, base = 200, H = base - top, y = (v) => base - Math.min(1.3, v) / 1.3 * H;
        // 刻度
        [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
          svg.append(s("line", { x1: x0, x2: x0 + cw * K, y1: y(v), y2: y(v), stroke: "var(--line)", "stroke-dasharray": v ? "3 3" : null }));
          svg.append(s("text", { x: x0 - 4, y: y(v) + 3.5, "text-anchor": "end", "font-size": 10.5, class: "t-mono t-muted", text: fmt(v, 2) }));
        });
        for (let i = 0; i < K; i++) {
          const cx = x0 + cw * i + cw / 2, bw = 28, sel = L && L.a === i;
          const tried = R.n[i] > 0;
          // UCB 加分（下一次要用的分數）
          if (strat === "ucb" && R.t >= K) {
            const bonus = C * Math.sqrt(Math.log(R.t) / R.n[i]);
            svg.append(s("rect", { x: cx - bw / 2, y: y(R.q[i] + bonus), width: bw, height: y(R.q[i]) - y(R.q[i] + bonus), fill: "var(--hot-soft)", stroke: "var(--hot)", "stroke-dasharray": "3 2" }));
          }
          if (tried) svg.append(s("rect", { x: cx - bw / 2, y: y(R.q[i]), width: bw, height: base - y(R.q[i]), fill: "var(--accent)", "fill-opacity": 0.85 }));
          // 真正的機率（代理人看不到）
          svg.append(s("line", { x1: cx - bw / 2 - 7, x2: cx + bw / 2 + 7, y1: y(P[i]), y2: y(P[i]), stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
          // 機台名
          svg.append(s("rect", { x: cx - 18, y: 4, width: 36, height: 24, rx: 6, fill: sel ? "var(--hot)" : "var(--panel-2)", stroke: sel ? "var(--hot)" : "var(--line-strong)", "stroke-width": sel ? 2 : 1 }));
          svg.append(s("text", { x: cx, y: 21, "text-anchor": "middle", "font-size": 15, "font-weight": 700, class: "t-mono", fill: sel ? "var(--panel)" : "var(--ink)", text: NAMES[i] }));
          svg.append(s("text", { x: cx, y: base + 17, "text-anchor": "middle", "font-size": 13.5, "font-weight": 600, class: "t-mono", text: tried ? fmt(R.q[i], 2) : "—" }));
          svg.append(s("text", { x: cx, y: base + 34, "text-anchor": "middle", "font-size": 12, class: "t-muted", text: `玩 ${R.n[i]} 次` }));
          if (sel) svg.append(s("text", { x: cx, y: base + 52, "text-anchor": "middle", "font-size": 12.5, "font-weight": 700, fill: L.rew ? "var(--ok)" : "var(--muted)", text: L.rew ? "中獎 +1" : "沒中" }));
        }
        keyOne.innerHTML = `<span><i style="background:var(--accent)"></i>估計值＝這台到目前的平均獎勵</span>` +
          (strat === "ucb" ? `<span><i style="background:var(--hot-soft);border:1px dashed var(--hot)"></i>UCB 加分（下一次用）</span>` : "") +
          `<span><span class="bd-sw" style="border-top-style:dashed;border-top-width:2px;color:var(--ink)"></span>真正的中獎機率（代理人不知道）</span>`;
      }

      function drawCurve() {
        nn.lineChart(cSvg, { W: 360, H: 170, xMax: T, yMin: 0, yMax: 1, xLabel: "次數", refY: 0.2,
          series: [{ pts: run.curve.slice(1), color: "var(--hot)" }] });
      }

      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      function render() {
        refreshBtns(); drawMachines(); drawCurve();
        const R = run, L = R.last, share = R.t ? R.bestN / R.t : 0;
        const head = R.t ? `第 ${R.t} 次${timer ? "（自動玩中）" : ""}：總獎勵 ${R.tot}（平均每次 ${fmt(R.tot / R.t, 3)}）；最好的機台 E 被選了 ${R.bestN} 次（${fmt(share * 100, 1)}%）` : "";
        let kd = "info", text;
        if (!R.t) text = "還沒開始：每台都還沒玩過，估計值都是空的。按「拉 1 次」一次一次看，或按「▶ 自動玩」連續玩 2000 次（約 8 秒）";
        else if (R.t < K) text = head + "。先把每台各拉 1 次，才有估計值可以比";
        else if (R.t < T) {
          const most = R.n.indexOf(Math.max(...R.n));
          text = head + `。目前玩最多的是 ${NAMES[most]}`;
        } else {
          const most = R.n.indexOf(Math.max(...R.n));
          if (strat === "eps" && eps === 1) { kd = "bad"; text = head + "。完全亂選：每台都約 20%，學到的估計值完全沒被用上"; }
          else if (share < 0.5 && most !== BEST) { kd = "bad"; text = head + `。困住了：大部分時間在玩 ${NAMES[most]}（真正機率 ${fmt(P[most], 2)}）` + (R.q[BEST] < R.q[most] ? `；E 的估計值只有 ${fmt(R.q[BEST], 2)}（玩了 ${R.n[BEST]} 次），比 ${NAMES[most]} 的 ${fmt(R.q[most], 2)} 低，所以很少再回去試` : ""); }
          else if (share < 0.5) { kd = "warn"; text = head + "。E 玩得最多，但還不到一半"; }
          else if (share >= 0.8) { kd = "ok"; text = head + "。大部分時間都在玩最好的 E"; }
          else { kd = "warn"; text = head + (strat === "eps" ? `。有找到 E，但 ε = ${eps} 的亂試一直持續到最後` : "。有找到 E，但前面花了不少次數在探索"); }
        }
        status.className = "status status--" + kd; status.textContent = text;

        if (!L) {
          calc.innerHTML = row("規則", strat === "ucb"
            ? "先每台各拉 1 次；之後每次選「估計值 + c × √(ln t ÷ n)」最大的一台（t＝已經玩的次數、n＝這台玩過的次數，c = 1）"
            : "先每台各拉 1 次；之後每次抽一個 0〜1 的亂數 u：u < ε 就隨機挑一台（探索），否則挑估計值最高的一台（利用）") +
            row("估計值", "這台到目前為止的平均獎勵；中獎得 1、沒中得 0");
          return;
        }
        const a = L.a, name = NAMES[a];
        let why;
        if (L.why === "init") why = row(`第 ${L.t} 次`, `先每台各拉 1 次 → 拉 <span class="hl">${name}</span>`);
        else if (L.why === "ucb") why = row(`第 ${L.t} 次：分數 = 估計值 + 1 × √(ln ${L.t - 1} ÷ n)`, NAMES.map((nm, i) => `${i === a ? "<span class=\"hl\">" : ""}${nm} ${fmt(L.qBefore[i], 2)}+${fmt(L.bonus[i], 2)}=${fmt(L.score[i], 2)}${i === a ? "</span>" : ""}`).join("　")) +
          row("", `分數最高 → 拉 <span class="hl">${name}</span>${L.qBefore[a] < Math.max(...L.qBefore) ? "（估計值不是最高，是加分把它推上來的：它試得比較少）" : ""}`);
        else if (L.why === "explore") why = row(`第 ${L.t} 次`, `亂數 u = ${fmt(L.u, 3)} &lt; ε = ${eps} → 探索：隨機挑到 <span class="hl">${name}</span>`);
        else why = row(`第 ${L.t} 次`, `亂數 u = ${fmt(L.u, 3)} ≥ ε = ${eps} → 利用：挑估計值最高的 <span class="hl">${name}</span>（${fmt(L.qBefore[a], 2)}${L.ties > 1 ? `，${L.ties} 台同分隨機挑` : ""}）`);
        const n = L.nBefore[a] + 1;
        const upd = L.nBefore[a] === 0
          ? row("更新估計值", `${name} 第 1 次玩，獎勵 ${L.rew} → 估計值 = <span class="hl">${fmt(L.qAfter, 2)}</span>`)
          : row("更新估計值（平均）", `${name}：${fmt(L.qBefore[a], 3)} + (${L.rew} − ${fmt(L.qBefore[a], 3)}) ÷ ${n} = <span class="hl">${fmt(L.qAfter, 3)}</span>`);
        calc.innerHTML = why + upd;
      }

      /* ---------- ② 比較 ---------- */
      const SER = [
        { key: "e0", strat: "eps", eps: 0, label: "ε = 0（只利用）", color: "var(--neg)" },
        { key: "e001", strat: "eps", eps: 0.01, label: "ε = 0.01", color: "var(--hot)", dash: "6 4" },
        { key: "e01", strat: "eps", eps: 0.1, label: "ε = 0.1", color: "var(--pos)" },
        { key: "e03", strat: "eps", eps: 0.3, label: "ε = 0.3", color: "var(--accent)", dash: "2 3" },
        { key: "e1", strat: "eps", eps: 1, label: "ε = 1（只亂試）", color: "var(--muted)", dash: "8 3 2 3" },
        { key: "ucb", strat: "ucb", eps: 0, label: "UCB 方策", color: "var(--ok)" },
      ];
      const RUNS = 200, EVERY = 20;
      function startCmp() {
        if (cmp && cmp.loop.running) return;
        cmp = { res: SER.map(() => ({ tot: 0, last200: 0, stuck: 0, best: Array(T / EVERY).fill(0), done: 0 })) };
        cmp.loop = nn.loop(ctx, { total: SER.length * RUNS, perTick: 30,
          run(from, n) {
            for (let j = from; j < from + n; j++) {
              const si = Math.floor(j / RUNS), sr = SER[si], o = quickRun(sr.strat, sr.eps, 1000 + (j % RUNS), EVERY), acc = cmp.res[si];
              acc.tot += o.tot; acc.last200 += o.last200; if (o.bestN / T < 0.5) acc.stuck++;
              o.best.forEach((v, k) => (acc.best[k] += v)); acc.done++;
            }
          },
          onTick: renderCmp });
        cmp.loop.start(); renderCmp();
      }
      function renderCmp() {
        cmpBtn.disabled = !!(cmp && !cmp.loop.finished);
        cmpBtn.textContent = !cmp ? "▶ 開始比較" : cmp.loop.finished ? "✓ 比較完成" : "計算中…";
        cmpKey.innerHTML = SER.map((sr) => `<span><span class="bd-sw" style="color:${sr.color};border-top-style:${sr.dash ? "dashed" : "solid"}"></span>${sr.label}</span>`).join("");
        if (!cmp) {
          cmpStatus.className = "status status--info";
          cmpStatus.textContent = "還沒開始：按「▶ 開始比較」，6 種方策各玩 200 輪（每輪 2000 次，種子固定），把結果平均（約 1〜2 秒）";
          nn.lineChart(cmpSvg, { W: 360, H: 210, xMax: T, yMin: 0, yMax: 1, xLabel: "次數", refY: 0.2, series: [] });
          cmpTbl.replaceChildren();
          return;
        }
        nn.lineChart(cmpSvg, { W: 360, H: 210, xMax: T, yMin: 0, yMax: 1, xLabel: "次數", refY: 0.2,
          series: SER.map((sr, i) => ({ pts: cmp.res[i].done ? cmp.res[i].best.map((v, k) => [(k + 1) * EVERY, v / cmp.res[i].done]) : [], color: sr.color, dash: sr.dash })) });
        const done = cmp.loop.finished;
        const rows = SER.map((sr, i) => { const r = cmp.res[i], d = r.done || 1; return { sr, avg: r.tot / d, last: r.last200 / d / 200, stuck: r.stuck, done: r.done }; });
        const bestAvg = Math.max(...rows.map((r) => r.avg));
        cmpTbl.replaceChildren(h("table", { class: "tbl" },
          h("thead", null, h("tr", null, h("th", { class: "l" }, "方策"), h("th", null, "平均總獎勵（2000 次）"), h("th", null, "最後 200 次選 E 的比例"), h("th", null, "困住的輪數（E 不到一半）"))),
          h("tbody", null, ...rows.map((r) => h("tr", { class: done && r.avg === bestAvg ? "is-sel" : "" },
            h("td", { class: "l" }, h("span", { class: "bd-sw", style: { color: r.sr.color, borderTopStyle: r.sr.dash ? "dashed" : "solid" } }), r.sr.label),
            h("td", null, r.done ? fmt(r.avg, 1) : "…"), h("td", null, r.done ? fmt(r.last * 100, 1) + "%" : "…"), h("td", null, r.done ? `${r.stuck} / ${r.done}` : "…")))),
          h("tfoot", null, h("tr", null, h("td", { class: "l" }, "一直玩 E 的期望值"), h("td", null, fmt(T * P[BEST], 1)), h("td", null, "100%"), h("td", null, "")))));
        if (!done) { cmpStatus.className = "status status--info"; cmpStatus.textContent = `計算中…（${cmp.loop.done} / ${SER.length * RUNS} 輪）`; return; }
        const best = rows.find((r) => r.avg === bestAvg), e0 = rows[0], e1 = rows[4];
        cmpStatus.className = "status status--ok";
        cmpStatus.textContent = `${best.sr.label} 平均總獎勵最高（${fmt(best.avg, 1)}）。ε = 0 有 ${e0.stuck} 輪困在較差的機台；ε = 1 完全亂選，只有 ${fmt(e1.avg, 1)}`;
      }

      function layout() {
        pause();
        oneBox.hidden = mode !== "one"; cmpBox.hidden = mode !== "cmp";
        if (mode === "cmp") renderCmp(); else render();
      }

      note.innerHTML = `中獎機率是設計好的示範值。「先每台各拉 1 次」是常見的起手式，這一頁兩種方策都這樣做，比較才公平。UCB 是 <span class="en">Upper Confidence Bound</span>（信賴上界）的縮寫：加分項 √(ln t ÷ n) 在一台玩得越少（n 小）時越大，所以也叫「對不確定的機台保持樂觀」；標準的 UCB1 用 c = √2，這一頁用 c = 1（這組機台上表現較好），c 越大越愛探索。拉霸機只有「選哪一台」這一個決定，選完也不會改變狀況；如果每次行動還會改變接下來的處境（例如走迷宮），就要用到 <a href="#qlearn">Q 学習</a> 這類方法。ε-greedy 方策也是 Q 学習在訓練時挑行動的標準做法。實際應用例如：網站要顯示哪個廣告、推薦哪篇文章，邊看點擊率邊調整。`;

      ctx.onTheme(() => { if (mode === "one") render(); else renderCmp(); });
      reset(); layout();
      return { destroy() { pause(); if (cmp) cmp.loop.pause(); } };
    },
  });
})();
