/* Q 学習と SARSA（シラバス §9）
   懸崖格子（cliff walking）：4 × 10 格，起點在左下、終點在右下，中間底排 8 格是懸崖。
   行動：上／右／下／左（撞牆就留在原地）。每走一步獎勵 −1；掉進懸崖 −100 並回到起點（這一回合繼續）。走到終點結束。
   兩個代理人同時學：ε-greedy（ε = 0.1）挑行動，學習率 α = 0.2，割引率 γ = 1 或 0.9，每組 800 回合。
   Q 学習：目標 = r + γ × max Q(s′, ·)；SARSA：目標 = r + γ × Q(s′, a′)（a′＝下一步實際要走的行動）。
   兩者亂數各自獨立（種子 1〜5）。原本 500 回合時，SARSA 偶爾還沒收斂（照最高 Q 值走會繞圈），改成 800 回合；
   實測 5 種子 × 2 種 γ：Q 学習的路線都是 11 步緊貼懸崖，SARSA 都是不經過懸崖正上方那一列的遠路，訓練中掉下懸崖的次數 SARSA 都少得多。 */
(function () {
  const ROWS = 4, COLS = 10, S0 = [3, 0], GOAL = [3, 9], EPS = 0.1, ALPHA = 0.2, EPISODES = 800;
  const DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]], DN = ["往上", "往右", "往下", "往左"], ARW = ["↑", "→", "↓", "←"];
  const isCliff = (r, c) => r === ROWS - 1 && c > 0 && c < COLS - 1;
  const isGoal = (r, c) => r === GOAL[0] && c === GOAL[1];
  const cellName = (r, c) => `從上數第 ${r + 1} 格、從左數第 ${c + 1} 格`;

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
    lead: `<p>機器人要從起點走到終點，每走一步扣 1 分，掉下懸崖扣 100 分並回到起點，但沒有人告訴它該怎麼走，只能自己試。它替每一格的每一個行動記一個數字 Q(s, a)：「在這格這樣走，之後總共還會拿到多少獎勵」，這叫<span class="jp">行動価値関数</span>（行動價值函數，<span class="en">action-value function</span>）。每走一步，就用「拿到的獎勵 + 下一格的價值」修正剛才那一格的 Q 值。<span class="jp">Q 学習</span>（<span class="en">Q-learning</span>）和 <span class="jp">SARSA</span> 都這樣學，差別只在「下一格的價值」怎麼算。</p>`,
    closing: `兩者都用 ε-greedy 邊走邊亂試，也都用「獎勵 + 下一格的價值」更新。Q 学習取下一格<b>最好的</b>行動的值，等於假設之後都走最佳路線，所以學出貼著懸崖的最短路，但訓練時一亂走就掉下去；SARSA 取下一步<b>實際要走的</b>行動（包含亂走）的值，把「在懸崖邊偶爾會失足」也算進 Q 值，所以學出繞遠一點的安全路，訓練期間掉下去的次數也少得多。`,

    mount(bench, ctx) {
      const { h, s, fmt, ui, nn } = Atlas;
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

      const gSeg = ui.seg({ label: "割引率 γ", value: gamma, options: [{ value: 1, label: "1（不打折）" }, { value: 0.9, label: "0.9" }], onChange: (v) => { gamma = v; reset(); } });
      const seedBtn = ui.btn("換一組初始值", () => { seed = (seed % 5) + 1; reset(); }, { sm: true });
      const runBtn = ui.btn("▶ 開始訓練", () => { if (loop.running) loop.pause(); else loop.start(); render(); }, { primary: true });
      const stepBtn = ui.btn("前進 1 步", () => { loop.pause(); agents.forEach((A) => A.step()); render(); });
      const epBtn = ui.btn("跑完 1 回合", () => { loop.pause(); agents.forEach((A) => A.episode()); render(); });
      const ffBtn = ui.btn("⏭ 快轉到結束", () => { loop.fastForward(); render(); });

      const status = h("div", { class: "status" }), calc = h("div", { class: "calc" }), cellCalc = h("div", { class: "calc" }), note = h("p", { class: "note" });
      const svgs = [s("svg", { viewBox: "0 0 380 190", role: "img", "aria-label": "Q 学習的 Q 值與路線" }), s("svg", { viewBox: "0 0 380 190", role: "img", "aria-label": "SARSA 的 Q 值與路線" })];
      const caps = [h("div", { class: "ql-cap" }), h("div", { class: "ql-cap" })];
      const chart = s("svg", { role: "img", "aria-label": "每回合拿到的總獎勵" });
      bench.append(
        h("div", { class: "bench__row" }, gSeg.el, seedBtn),
        h("div", { class: "bench__row" }, runBtn, stepBtn, epBtn, ffBtn),
        status,
        h("div", { class: "ql-row" }, h("div", { class: "ql-fig" }, caps[0], svgs[0]), h("div", { class: "ql-fig" }, caps[1], svgs[1])),
        h("div", { class: "ql-key", html: `<span><i style="background:var(--neg-soft);border:1px solid var(--neg)"></i>懸崖</span><span>箭頭＝目前 Q 值最高的行動</span><span>格中數字＝最高的 Q 值（底色越藍越負）</span><span><span class="ql-sw" style="color:var(--hot)"></span>只走最高 Q 值時的路線</span><span>● 代理人現在的位置</span>` }),
        calc,
        h("div", { class: "field" }, h("span", { class: "field__label" }, "點任一格，比較兩者在那一格的 4 個 Q 值"), cellCalc),
        h("div", { class: "ql-fig", style: { flex: "none", maxWidth: "520px", marginInline: "auto", width: "100%" } },
          h("div", { class: "ql-cap", style: { fontWeight: 400, fontSize: "12.5px", color: "var(--muted)" }, html: `每回合拿到的總獎勵（最近 20 回合平均，−120 以下畫在底線）：<span style="color:var(--hot)">━ Q 学習</span>　<span style="color:var(--pos)">━ SARSA</span>` }), chart),
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
            const t = Math.min(1, maxQ[i] / lo), fill = Atlas.rgbStr(Atlas.mix(panel, acc, t * 0.75));
            g.append(s("rect", { x, y, width: CS, height: CS, fill, stroke: "var(--line-strong)", "stroke-width": 0.8 }));
            const dark = t > 0.55, ink = dark ? "var(--panel)" : "var(--ink)";
            if (isGoal(r, c)) txt.append(s("text", { x: x + CS / 2, y: y + 21, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: ink, text: "終點" }));
            else {
              const visited = Q[i].some((v) => v !== 0);
              if (visited) {
                txt.append(s("text", { x: x + CS / 2, y: y + 15, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: ink, text: ARW[Q[i].indexOf(maxQ[i])] }));
                txt.append(s("text", { x: x + CS / 2, y: y + 28, "text-anchor": "middle", "font-size": 9, class: "t-mono", fill: ink, text: fmt(maxQ[i], Math.abs(maxQ[i]) < 10 ? 1 : 0) }));
              }
              if (r === S0[0] && c === S0[1]) txt.append(s("text", { x: x + 3, y: y + 10, "font-size": 8.5, "font-weight": 700, fill: ink, text: "起點" }));
            }
          }
          g.addEventListener("click", () => { sel = [r, c]; render(); });
          svg.append(g);
        }
        svg.append(s("text", { x: X0 + CS * 5, y: Y0 + 3 * CS + 21, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "var(--neg)", text: "懸崖（−100，回到起點）", style: "pointer-events:none" }));
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
        svg.append(s("text", { x: X0, y: 16, "font-size": 11, class: "t-muted", text: `第 ${A.ep} 回合結束　掉下懸崖 ${A.falls} 次${A.steps ? `　這回合已走 ${A.steps} 步` : ""}` }));
        A.gp = gp;
        caps[k].innerHTML = k === 0 ? `Q 学習 <small>目標 = r + γ × max Q(s′, ·)</small>` : `SARSA <small>目標 = r + γ × Q(s′, a′)</small>`;
      }

      const avgLast = (arr, n) => { const a = arr.slice(-n); return a.reduce((x, y) => x + y, 0) / (a.length || 1); };
      function drawChart() {
        const sm = (rets) => { const out = []; for (let i = 0; i < rets.length; i++) { const a = rets.slice(Math.max(0, i - 19), i + 1); out.push([i + 1, a.reduce((x, y) => x + y, 0) / a.length]); } return out; };
        nn.lineChart(chart, { W: 420, H: 190, xMax: EPISODES, yMin: -120, yMax: 0, xLabel: "回合", series: [{ pts: sm(agents[0].rets), color: "var(--hot)" }, { pts: sm(agents[1].rets), color: "var(--pos)" }] });
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
        runBtn.textContent = run ? "❚❚ 暫停" : loop.finished || e >= EPISODES ? "✓ 訓練完成" : e || Aq.steps ? "▶ 繼續訓練" : "▶ 開始訓練";
        runBtn.disabled = stepBtn.disabled = epBtn.disabled = ffBtn.disabled = e >= EPISODES;
        let kd = "info", txt;
        if (!e && !Aq.steps && !Aq.ep) txt = "還沒訓練：Q 值全是 0，兩個代理人都只能亂走。按「前進 1 步」看每一步怎麼更新，或按「▶ 開始訓練」連續跑 800 回合（約 6 秒）";
        else {
          const pq = pathInfo(Aq.gp), ps = pathInfo(As.gp);
          const desc = (gp, pi) => !gp.ok ? (gp.why === "cliff" ? "照最高 Q 值走會掉下懸崖（還沒學好）" : "照最高 Q 值走會繞圈、到不了終點（還沒學好）") : `${pi.len} 步${pi.edge ? "、一路緊貼懸崖" : `、最高走到從上數第 ${pi.minRow + 1} 格`}`;
          txt = `${run ? "訓練中，" : ""}已完成 ${e} 回合：Q 学習的路線 ${desc(Aq.gp, pq)}；SARSA 的路線 ${desc(As.gp, ps)}`;
          if (e >= EPISODES) {
            const rq = avgLast(Aq.rets, 50), rs = avgLast(As.rets, 50);
            txt += `。訓練中掉下懸崖：Q 学習 ${Aq.falls} 次、SARSA ${As.falls} 次；最後 50 回合平均每回合獎勵 Q 学習 ${fmt(rq, 1)}、SARSA ${fmt(rs, 1)}`;
            kd = Aq.gp.ok && As.gp.ok ? "ok" : "warn";
          }
        }
        status.className = "status status--" + kd; status.textContent = txt;

        // 最後一次更新
        const one = (A, nm) => {
          const L = A.last;
          if (!L) return row(nm, "還沒走過");
          const head = `在${cellName(L.r, L.c)}${DN[L.a]}${L.aRand ? "（ε 亂走）" : ""} → ${L.fell ? "掉下懸崖，獎勵 −100，回到起點" : L.done ? "到達終點，獎勵 −1" : `獎勵 −1，到${cellName(L.nr, L.nc)}`}`;
          let tgt;
          if (L.done) tgt = `終點之後沒有獎勵，目標 = r = <span class="hl">${fmt(L.target, 2)}</span>`;
          else if (A.method === "q") tgt = `目標 = r + γ × max Q(s′, ·) = ${L.rew} + ${gamma} × max(${L.q2.map((v) => fmt(v, 2)).join(", ")}) = ${L.rew} + ${gamma} × ${fmt(L.next, 2)} = <span class="hl">${fmt(L.target, 2)}</span>`;
          else tgt = `下一步實際要${DN[L.a2]}${L.a2Rand ? "（這是 ε 亂走抽到的）" : ""}：目標 = r + γ × Q(s′, ${DN[L.a2]}) = ${L.rew} + ${gamma} × ${fmt(L.next, 2)} = <span class="hl">${fmt(L.target, 2)}</span>`;
          return row(nm, head) + row("", tgt) + row("", `Q(s, ${DN[L.a]}) ← ${fmt(L.old, 2)} + ${ALPHA} × (${fmt(L.target, 2)} − ${fmt(L.old, 2)}) = <span class="hl">${fmt(L.nw, 2)}</span>`);
        };
        calc.innerHTML = row("最後一步的更新", "（學習率 α = 0.2、ε = 0.1）") + one(Aq, "Q 学習") + one(As, "SARSA");

        const [r, c] = sel, i = r * COLS + c;
        if (isCliff(r, c)) cellCalc.innerHTML = row(cellName(r, c), "這是懸崖，代理人不會停在這裡，沒有 Q 值");
        else if (isGoal(r, c)) cellCalc.innerHTML = row(cellName(r, c), "這是終點，到了就結束，沒有 Q 值");
        else cellCalc.innerHTML = row(cellName(r, c), "") + agents.map((A, k) => {
          const q = A.Q[i], m = Math.max(...q);
          return row(k ? "SARSA" : "Q 学習", q.map((v, a) => `${v === m && q.some((x) => x !== 0) ? "<span class=\"hl\">" : ""}${DN[a]} ${fmt(v, 2)}${v === m && q.some((x) => x !== 0) ? "</span>" : ""}`).join("　"));
        }).join("") + (r === ROWS - 2 && c > 0 && c < COLS - 1 && e >= 100 && As.Q[i][1] < Aq.Q[i][1] - 1
          ? row("比一比", `這格在懸崖正上方：SARSA 的「往右」${fmt(As.Q[i][1], 2)} 比 Q 学習的 ${fmt(Aq.Q[i][1], 2)} 低，因為 SARSA 把「接下來有 10% 會亂走、可能往下掉」也算進去了`)
          : "");
      }

      note.innerHTML = `把世界寫成「狀態 → 行動 → 獎勵與下一個狀態」，而且下一步只取決於現在的狀態與行動，這種設定叫<span class="jp">マルコフ決定過程</span>（馬可夫決策過程，<span class="en">Markov decision process, MDP</span>）。<span class="jp">割引率</span>（折扣率，<span class="en">discount rate</span>）γ 讓越遠的獎勵算得越少：γ = 1 時起點的 Q 值約等於「到終點還要扣幾分」，γ = 0.9 時 11 步的路是 −1 − 0.9 − 0.81 − … ≈ −6.9。只看格子本身好不好（不分行動）的叫<span class="jp">状態価値関数</span>（狀態價值函數）。Q 学習和 SARSA 都是先學「價值」再照價值選行動；另一類是直接學「怎麼選行動」的機率，叫<span class="jp">方策勾配法</span>（策略梯度法，代表為 <span class="jp">REINFORCE</span>），兩者結合的是 <span class="jp">Actor-Critic</span>。格子一多就沒辦法用表格存 Q 值，改用神經網路估計 Q 值的就是 DQN（シラバス §29）。挑行動用的 ε-greedy 見 <a href="#bandit">バンディット</a>；SARSA 的名稱來自更新用到的五個量 s, a, r, s′, a′。`;

      reset();
      ctx.onTheme(() => { drawGrid(0); drawGrid(1); });
      return { destroy() { if (loop) loop.pause(); } };
    },
  });
})();
