/* Q 学習と SARSA（シラバス §9）
   懸崖格子（cliff walking）：4 × 10 格，起點在左下、終點在右下，中間底排 8 格是懸崖。
   行動：上／右／下／左（撞牆就留在原地）。每走一步獎勵 −1；掉進懸崖 −100 並回到起點（這一回合繼續）。走到終點結束。
   兩個代理人同時學：ε-greedy（ε = 0.1）挑行動，學習率 α = 0.2，割引率 γ = 1 或 0.9，每組 800 回合。
   Q 学習：目標 = r + γ × max Q(s′, ·)；SARSA：目標 = r + γ × Q(s′, a′)（a′＝下一步實際要走的行動）。
   兩者亂數各自獨立（種子 1〜5）。原本 500 回合時，SARSA 偶爾還沒收斂（照最高 Q 值走會繞圈），改成 800 回合；
   實測 5 種子 × 2 種 γ：Q 学習的路線都是 11 步緊貼懸崖，SARSA 都是不經過懸崖正上方那一列的遠路，訓練中掉下懸崖的次數 SARSA 都少得多。
   文字在 locales/zh/qlearn.js、locales/ja/qlearn.js。 */
(function () {
  const ROWS = 4, COLS = 10, S0 = [3, 0], GOAL = [3, 9], EPS = 0.1, ALPHA = 0.2, EPISODES = 800;
  const DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]], ARW = ["↑", "→", "↓", "←"];
  const isCliff = (r, c) => r === ROWS - 1 && c > 0 && c < COLS - 1;
  const isGoal = (r, c) => r === GOAL[0] && c === GOAL[1];

  function makeAgent(method, gamma, seed) {
    const rnd = Atlas.nn.rng(seed); // 兩個代理人各自一份亂數（同一個種子）
    const Q = Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0]);
    const A = { method, gamma, Q, ep: 0, falls: 0, rets: [], ret: 0, steps: 0, pos: S0.slice(), act: null, actRand: false, last: null };
    function pick(s) {
      if (rnd() < EPS) return { a: Math.floor(rnd() * 4), rand: true };
      const q = Q[s], m = Math.max(...q), c = []; q.forEach((v, i) => v === m && c.push(i));
      return { a: c[Math.floor(rnd() * c.length)], rand: false };
    }
    /** 走一步；回合結束回傳 true */
    A.step = () => {
      if (A.ep >= EPISODES) return false;
      const [r, c] = A.pos, s = r * COLS + c;
      if (A.act == null) { const p = pick(s); A.act = p.a; A.actRand = p.rand; }
      const a = A.act;
      let nr = Math.min(ROWS - 1, Math.max(0, r + DIRS[a][0])), nc = Math.min(COLS - 1, Math.max(0, c + DIRS[a][1]));
      let rew = -1, fell = false;
      if (isCliff(nr, nc)) { rew = -100; fell = true; A.falls++; [nr, nc] = S0; }
      const done = isGoal(nr, nc), s2 = nr * COLS + nc;
      const p2 = done ? { a: 0, rand: false } : pick(s2);
      const next = done ? 0 : method === "q" ? Math.max(...Q[s2]) : Q[s2][p2.a];
      const target = rew + gamma * next, old = Q[s][a];
      Q[s][a] = old + ALPHA * (target - old);
      A.last = { r, c, a, aRand: A.actRand, rew, fell, done, nr, nc, a2: done ? null : p2.a, a2Rand: p2.rand, q2: Q[s2].slice(), next, target, old, nw: Q[s][a] };
      A.ret += rew; A.steps++;
      A.pos = [nr, nc]; A.act = done ? null : p2.a; A.actRand = p2.rand;
      if (done || A.steps >= 1000) { A.rets.push(A.ret); A.ep++; A.ret = 0; A.steps = 0; A.pos = S0.slice(); A.act = null; return true; }
      return false;
    };
    A.episode = () => { if (A.ep >= EPISODES) return; const e = A.ep; let k = 0; while (A.ep === e && k++ < 5000) A.step(); };
    return A;
  }
  /** 目前 Q 表的貪婪路線（不探索） */
  function greedyPath(Q) {
    let [r, c] = S0; const path = [[r, c]], seen = new Set([r * COLS + c]);
    for (let k = 0; k < 60; k++) {
      const q = Q[r * COLS + c], a = q.indexOf(Math.max(...q));
      const nr = Math.min(ROWS - 1, Math.max(0, r + DIRS[a][0])), nc = Math.min(COLS - 1, Math.max(0, c + DIRS[a][1]));
      if (isCliff(nr, nc)) return { path, ok: false, why: "cliff" };
      if (seen.has(nr * COLS + nc)) return { path: path.concat([[nr, nc]]), ok: false, why: "loop" };
      r = nr; c = nc; path.push([r, c]); seen.add(r * COLS + c);
      if (isGoal(r, c)) return { path, ok: true };
    }
    return { path, ok: false, why: "loop" };
  }

  Atlas.define("qlearn", {

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
      const { t } = ctx;
      const DN = ["dirUp", "dirRight", "dirDown", "dirLeft"].map((k) => t(k));
      const cellName = (r, c) => t("cellName", { r: r + 1, c: c + 1 });
      let gamma = 1, seed = 1, sel = [2, 4], agents = null, loop = null;

      bench.append(h("style", { text: `
        .ql-row { display: flex; flex-wrap: wrap; gap: 14px 22px; justify-content: center; }
        .ql-fig { flex: 1 1 330px; max-width: 440px; min-width: 0; }
        .ql-fig > svg { display: block; width: 100%; height: auto; }
        .ql-cap { font-size: 13px; text-align: center; margin-bottom: 2px; font-weight: 700; }
        .ql-cap small { font-weight: 400; color: var(--muted); font-size: 12px; }
        .ql-cell { cursor: pointer; }
        .ql-key { display: flex; flex-wrap: wrap; gap: 4px 14px; justify-content: center; font-size: 12.5px; color: var(--muted); }
        .ql-key i { display: inline-block; width: 14px; height: 10px; margin-right: 4px; vertical-align: -1px; }
        .ql-sw { display: inline-block; width: 18px; height: 0; border-top: 3px solid; vertical-align: 3px; margin-right: 4px; }
      ` }));

      const gSeg = ui.seg({ label: t("gLabel"), value: gamma, options: [{ value: 1, label: t("g1") }, { value: 0.9, label: "0.9" }], onChange: (v) => { gamma = v; reset(); } });
      const seedBtn = ui.btn(t("btnSeed"), () => { seed = (seed % 5) + 1; reset(); }, { sm: true });
      const runBtn = ui.btn(t("btnRun"), () => { if (loop.running) loop.pause(); else loop.start(); render(); }, { primary: true });
      const stepBtn = ui.btn(t("btnStep"), () => { loop.pause(); agents.forEach((A) => A.step()); render(); });
      const epBtn = ui.btn(t("btnEp"), () => { loop.pause(); agents.forEach((A) => A.episode()); render(); });
      const ffBtn = ui.btn(t("btnFf"), () => { loop.fastForward(); render(); });

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), cellCalc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const svgs = [s("svg", { viewBox: "0 0 380 190", role: "img", "aria-label": t("qAria") }), s("svg", { viewBox: "0 0 380 190", role: "img", "aria-label": t("sAria") })];
      const caps = [h("div", { class: "ql-cap" }), h("div", { class: "ql-cap" })];
      const chart = s("svg", { role: "img", "aria-label": t("chartAria") });
      bench.append(
        h("div", { class: "bench__row" }, gSeg.el, seedBtn),
        h("div", { class: "bench__row" }, runBtn, stepBtn, epBtn, ffBtn),
        status,
        h("div", { class: "ql-row" }, h("div", { class: "ql-fig" }, caps[0], svgs[0]), h("div", { class: "ql-fig" }, caps[1], svgs[1])),
        h("div", { class: "ql-key", html: t("key") }),
        calc,
        h("div", { class: "field" }, h("span", { class: "field__label" }, t("cellLabel")), cellCalc),
        h("div", { class: "ql-fig", style: { flex: "none", maxWidth: "520px", marginInline: "auto", width: "100%" } },
          h("div", { class: "ql-cap", style: { fontWeight: 400, fontSize: "12.5px", color: "var(--muted)" }, html: t("capChart") }), chart),
        note);

      function reset() {
        if (loop) loop.pause();
        agents = [makeAgent("q", gamma, seed), makeAgent("sarsa", gamma, seed)];
        loop = nn.loop(ctx, { total: EPISODES, perTick: 4,
          run(from, n) { for (let i = 0; i < n; i++) agents.forEach((A) => A.episode()); if (doneEps() >= EPISODES) loop.pause(); },
          onTick: render });
        render();
      }
      const doneEps = () => Math.min(agents[0].ep, agents[1].ep);

      /* ---------- 格子圖 ---------- */
      const CS = 34, X0 = 20, Y0 = 26;
      function drawGrid(k) {
        const A = agents[k], svg = svgs[k], Q = A.Q; svg.replaceChildren();
        const maxQ = Q.map((q) => Math.max(...q));
        let lo = 0; maxQ.forEach((v, i) => { const r = Math.floor(i / COLS), c = i % COLS; if (!isCliff(r, c)) lo = Math.min(lo, v); });
        lo = Math.min(lo, -1);
        const panel = ctx.color("--panel"), acc = ctx.color("--seq-hi"), txt = s("g", { style: "pointer-events:none" });
        for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
          const x = X0 + c * CS, y = Y0 + r * CS, i = r * COLS + c;
          const g = s("g", { class: "ql-cell" });
          if (isCliff(r, c)) g.append(s("rect", { x, y, width: CS, height: CS, fill: "var(--neg-soft)", stroke: "var(--neg)", "stroke-width": 0.8 }));
          else {
            const tv = Math.min(1, maxQ[i] / lo), fill = Atlas.rgbStr(Atlas.mix(panel, acc, tv * 0.75));
            g.append(s("rect", { x, y, width: CS, height: CS, fill, stroke: "var(--line-strong)", "stroke-width": 0.8 }));
            const dark = tv > 0.55, ink = dark ? "var(--panel)" : "var(--ink)";
            if (isGoal(r, c)) txt.append(s("text", { x: x + CS / 2, y: y + 21, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: ink, text: t("goal") }));
            else {
              const visited = Q[i].some((v) => v !== 0);
              if (visited) {
                txt.append(s("text", { x: x + CS / 2, y: y + 15, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: ink, text: ARW[Q[i].indexOf(maxQ[i])] }));
                txt.append(s("text", { x: x + CS / 2, y: y + 28, "text-anchor": "middle", "font-size": 9, class: "t-mono", fill: ink, text: fmt(maxQ[i], Math.abs(maxQ[i]) < 10 ? 1 : 0) }));
              }
              if (r === S0[0] && c === S0[1]) txt.append(s("text", { x: x + 3, y: y + 10, "font-size": 8.5, "font-weight": 700, fill: ink, text: t("start") }));
            }
          }
          g.addEventListener("click", () => { sel = [r, c]; render(); });
          svg.append(g);
        }
        svg.append(s("text", { x: X0 + CS * 5, y: Y0 + 3 * CS + 21, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "var(--neg)", text: t("cliff"), style: "pointer-events:none" }));
        // 選中的格子
        svg.append(s("rect", { x: X0 + sel[1] * CS + 1, y: Y0 + sel[0] * CS + 1, width: CS - 2, height: CS - 2, fill: "none", stroke: "var(--ink)", "stroke-width": 2, "stroke-dasharray": "3 2", style: "pointer-events:none" }));
        // 路線
        const gp = greedyPath(Q);
        if (gp.path.length > 1) svg.append(s("polyline", { points: gp.path.map(([r, c]) => `${X0 + c * CS + CS / 2},${Y0 + r * CS + CS / 2}`).join(" "), fill: "none", stroke: "var(--hot)", "stroke-width": 5, "stroke-linejoin": "round", "stroke-opacity": 0.75, style: "pointer-events:none" }));
        svg.append(txt);
        // 代理人
        const [pr, pc] = A.pos;
        svg.append(s("circle", { cx: X0 + pc * CS + CS / 2, cy: Y0 + pr * CS + CS / 2, r: 7, fill: "var(--hot)", stroke: "var(--panel)", "stroke-width": 2, style: "pointer-events:none" }));
        // 欄號
        svg.append(s("text", { x: X0, y: 16, "font-size": 11, class: "t-muted", text: t(A.steps ? "gridHeadSteps" : "gridHead", { ep: A.ep, f: A.falls, s: A.steps }) }));
        A.gp = gp;
        caps[k].innerHTML = t(k === 0 ? "capQ" : "capS");
      }

      const avgLast = (arr, n) => { const a = arr.slice(-n); return a.reduce((x, y) => x + y, 0) / (a.length || 1); };
      function drawChart() {
        const sm = (rets) => { const out = []; for (let i = 0; i < rets.length; i++) { const a = rets.slice(Math.max(0, i - 19), i + 1); out.push([i + 1, a.reduce((x, y) => x + y, 0) / a.length]); } return out; };
        nn.lineChart(chart, { W: 420, H: 190, xMax: EPISODES, yMin: -120, yMax: 0, xLabel: t("chartAxis"), series: [{ pts: sm(agents[0].rets), color: "var(--hot)" }, { pts: sm(agents[1].rets), color: "var(--pos)" }] });
      }
      /** 路線離懸崖多遠：中間各欄（第 2〜9 欄）中，路線最靠近懸崖的那一列 */
      function pathInfo(gp) {
        const mid = gp.path.filter(([r, c]) => c > 0 && c < COLS - 1);
        const nearest = mid.length ? Math.max(...mid.map(([r]) => r)) : -1;
        const minRow = mid.length ? Math.min(...mid.map(([r]) => r)) : -1;
        return { len: gp.path.length - 1, edge: nearest === ROWS - 2 && mid.every(([r]) => r === ROWS - 2), minRow };
      }

      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      function render() {
        drawGrid(0); drawGrid(1); drawChart();
        const e = doneEps(), [Aq, As] = agents, run = loop.running;
        runBtn.textContent = t(run ? "btnPause" : loop.finished || e >= EPISODES ? "btnDone" : e || Aq.steps ? "btnResume" : "btnRun");
        runBtn.disabled = stepBtn.disabled = epBtn.disabled = ffBtn.disabled = e >= EPISODES;
        let kd = "info", txt;
        if (!e && !Aq.steps && !Aq.ep) txt = t("statusIdle");
        else {
          const pq = pathInfo(Aq.gp), ps = pathInfo(As.gp);
          const desc = (gp, pi) => !gp.ok ? t(gp.why === "cliff" ? "descCliff" : "descLoop") : t(pi.edge ? "descEdge" : "descUp", { len: pi.len, row: pi.minRow + 1 });
          txt = t(run ? "statusRun" : "status", { e, q: desc(Aq.gp, pq), s: desc(As.gp, ps) });
          if (e >= EPISODES) {
            const rq = avgLast(Aq.rets, 50), rs = avgLast(As.rets, 50);
            txt += t("statusEnd", { fq: Aq.falls, fs: As.falls, rq: fmt(rq, 1), rs: fmt(rs, 1) });
            kd = Aq.gp.ok && As.gp.ok ? "ok" : "warn";
          }
        }
        status.className = "status status--" + kd; status.textContent = txt;

        // 最後一次更新
        const one = (A, nm) => {
          const L = A.last;
          if (!L) return row(nm, t("notYet"));
          const head = t(L.aRand ? "moveRand" : "move", { cell: cellName(L.r, L.c), dir: DN[L.a], res: L.fell ? t("resFell") : L.done ? t("resGoal") : t("resMove", { cell: cellName(L.nr, L.nc) }) });
          let tgt;
          if (L.done) tgt = t("tgtDone", { v: fmt(L.target, 2) });
          else if (A.method === "q") tgt = t("tgtQ", { r: L.rew, g: gamma, qs: L.q2.map((v) => fmt(v, 2)).join(", "), n: fmt(L.next, 2), v: fmt(L.target, 2) });
          else tgt = t(L.a2Rand ? "tgtSRand" : "tgtS", { dir: DN[L.a2], r: L.rew, g: gamma, n: fmt(L.next, 2), v: fmt(L.target, 2) });
          return row(nm, head) + row("", tgt) + row("", `Q(s, ${DN[L.a]}) ← ${fmt(L.old, 2)} + ${ALPHA} × (${fmt(L.target, 2)} − ${fmt(L.old, 2)}) = <span class="hl">${fmt(L.nw, 2)}</span>`);
        };
        calc.innerHTML = row(t("lastK"), t("lastV")) + one(Aq, t("nameQ")) + one(As, "SARSA");

        const [r, c] = sel, i = r * COLS + c;
        if (isCliff(r, c)) cellCalc.innerHTML = row(cellName(r, c), t("cliffCell"));
        else if (isGoal(r, c)) cellCalc.innerHTML = row(cellName(r, c), t("goalCell"));
        else cellCalc.innerHTML = row(cellName(r, c), "") + agents.map((A, k) => {
          const q = A.Q[i], m = Math.max(...q);
          return row(k ? "SARSA" : t("nameQ"), q.map((v, a) => `${v === m && q.some((x) => x !== 0) ? "<span class=\"hl\">" : ""}${DN[a]} ${fmt(v, 2)}${v === m && q.some((x) => x !== 0) ? "</span>" : ""}`).join("　"));
        }).join("") + (r === ROWS - 2 && c > 0 && c < COLS - 1 && e >= 100 && As.Q[i][1] < Aq.Q[i][1] - 1
          ? row(t("cmpK"), t("cmp", { s: fmt(As.Q[i][1], 2), q: fmt(Aq.Q[i][1], 2) }))
          : "");
      }

      note.innerHTML = t("note");

      reset();
      ctx.onTheme(() => { drawGrid(0); drawGrid(1); });
      return { destroy() { if (loop) loop.pause(); } };
    },
  });
})();
