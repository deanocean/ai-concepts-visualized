/* Q 學習與 SARSA（concepts/qlearn.js）的中文文字
   - {name} 是程式帶入的值；HTML 只用在 lead、closing、note、key、cap*、tgt*、qUpd 這類會以 HTML 顯示的項目 */
Atlas.strings("zh", "qlearn", {
  lead: `<p>機器人要從起點走到終點，每走一步扣 1 分，掉下懸崖扣 100 分並回到起點，但沒有人告訴它該怎麼走，只能自己試。它替每一格的每一個行動記一個數字 Q(s, a)：「在這格這樣走，之後總共還會拿到多少獎勵」，這叫<span class="term">行動價值函數</span>（<span class="en">action-value function</span>）。每走一步，就用「拿到的獎勵 + 下一格的價值」修正剛才那一格的 Q 值。<span class="term">Q 學習</span>（<span class="en">Q-learning</span>）和 <span class="term">SARSA</span> 都這樣學，差別只在「下一格的價值」怎麼算。</p>`,
  closing: `兩者都用 ε-greedy 邊走邊亂試，也都用「獎勵 + 下一格的價值」更新。Q 學習取下一格<b>最好的</b>行動的值，等於假設之後都走最佳路線，所以學出貼著懸崖的最短路，但訓練時一亂走就掉下去；SARSA 取下一步<b>實際要走的</b>行動（包含亂走）的值，把「在懸崖邊偶爾會失足」也算進 Q 值，所以學出繞遠一點的安全路，訓練期間掉下去的次數也少得多。`,

  /* 方向（格子裡的表格與句子共用） */
  dirUp: "往上",
  dirRight: "往右",
  dirDown: "往下",
  dirLeft: "往左",
  cellName: "從上數第 {r} 格、從左數第 {c} 格",

  /* 控制項 */
  gLabel: "折扣率 γ",
  g1: "1（不打折）",
  btnSeed: "換一組初始值",
  btnRun: "▶ 開始訓練",
  btnResume: "▶ 繼續訓練",
  btnPause: "❚❚ 暫停",
  btnDone: "✓ 訓練完成",
  btnStep: "前進 1 步",
  btnEp: "跑完 1 回合",
  btnFf: "⏭ 快轉到結束",

  /* 圖（螢幕報讀用） */
  qAria: "Q 學習的 Q 值與路線",
  sAria: "SARSA 的 Q 值與路線",
  chartAria: "每回合拿到的總獎勵",

  /* 圖說 */
  key: `<span><i style="background:var(--neg-soft);border:1px solid var(--neg)"></i>懸崖</span><span>箭頭＝目前 Q 值最高的行動</span><span>格中數字＝最高的 Q 值（底色越藍越負）</span><span><span class="ql-sw" style="color:var(--hot)"></span>只走最高 Q 值時的路線</span><span>● 代理人現在的位置</span>`,
  cellLabel: "點任一格，比較兩者在那一格的 4 個 Q 值",
  capChart: `每回合拿到的總獎勵（最近 20 回合平均，−120 以下畫在底線）：<span style="color:var(--hot)">━ Q 學習</span>　<span style="color:var(--pos)">━ SARSA</span>`,
  goal: "終點",
  start: "起點",
  cliff: "懸崖（−100，回到起點）",
  gridHead: "第 {ep} 回合結束　掉下懸崖 {f} 次",
  gridHeadSteps: "第 {ep} 回合結束　掉下懸崖 {f} 次　這回合已走 {s} 步",
  capQ: "Q 學習 <small>目標 = r + γ × max Q(s′, ·)</small>",
  capS: "SARSA <small>目標 = r + γ × Q(s′, a′)</small>",
  chartAxis: "回合",

  /* 狀態列 */
  statusIdle: "還沒訓練：Q 值全是 0，兩個代理人都只能亂走。按「前進 1 步」看每一步怎麼更新，或按「▶ 開始訓練」連續跑 800 回合（約 6 秒）",
  descCliff: "照最高 Q 值走會掉下懸崖（還沒學好）",
  descLoop: "照最高 Q 值走會繞圈、到不了終點（還沒學好）",
  descEdge: "{len} 步、一路緊貼懸崖",
  descUp: "{len} 步、最高走到從上數第 {row} 格",
  status: "已完成 {e} 回合：Q 學習的路線 {q}；SARSA 的路線 {s}",
  statusRun: "訓練中，已完成 {e} 回合：Q 學習的路線 {q}；SARSA 的路線 {s}",
  statusEnd: "。訓練中掉下懸崖：Q 學習 {fq} 次、SARSA {fs} 次；最後 50 回合平均每回合獎勵 Q 學習 {rq}、SARSA {rs}",

  /* 算式 */
  nameQ: "Q 學習",
  lastK: "最後一步的更新",
  lastV: "（學習率 α = 0.2、ε = 0.1）",
  notYet: "還沒走過",
  move: "在{cell}{dir} → {res}",
  moveRand: "在{cell}{dir}（ε 亂走） → {res}",
  resFell: "掉下懸崖，獎勵 −100，回到起點",
  resGoal: "到達終點，獎勵 −1",
  resMove: "獎勵 −1，到{cell}",
  tgtDone: `終點之後沒有獎勵，目標 = r = <span class="hl">{v}</span>`,
  tgtQ: `目標 = r + γ × max Q(s′, ·) = {r} + {g} × max({qs}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  tgtS: `下一步實際要{dir}：目標 = r + γ × Q(s′, {dir}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  tgtSRand: `下一步實際要{dir}（這是 ε 亂走抽到的）：目標 = r + γ × Q(s′, {dir}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  cliffCell: "這是懸崖，代理人不會停在這裡，沒有 Q 值",
  goalCell: "這是終點，到了就結束，沒有 Q 值",
  cmpK: "比一比",
  cmp: "這格在懸崖正上方：SARSA 的「往右」{s} 比 Q 學習的 {q} 低，因為 SARSA 把「接下來有 10% 會亂走、可能往下掉」也算進去了",

  note: `把世界寫成「狀態 → 行動 → 獎勵與下一個狀態」，而且下一步只取決於現在的狀態與行動，這種設定叫<span class="term">馬可夫決策過程</span>（<span class="en">Markov decision process, MDP</span>）。<span class="term">折扣率</span>（<span class="en">discount rate</span>）γ 讓越遠的獎勵算得越少：γ = 1 時起點的 Q 值約等於「到終點還要扣幾分」，γ = 0.9 時 11 步的路是 −1 − 0.9 − 0.81 − … ≈ −6.9。只看格子本身好不好（不分行動）的叫<span class="term">狀態價值函數</span>（<span class="en">state-value function</span>）。Q 學習和 SARSA 都是先學「價值」再照價值選行動；另一類是直接學「怎麼選行動」的機率，叫<span class="term">策略梯度法</span>（<span class="en">policy gradient method</span>，代表為 REINFORCE），兩者結合的是 <span class="term">Actor-Critic</span>。格子一多就沒辦法用表格存 Q 值，改用神經網路估計 Q 值的就是 DQN（大綱 §29）。挑行動用的 ε-greedy 見 <a href="#bandit">多臂拉霸</a>；SARSA 的名稱來自更新用到的五個量 s, a, r, s′, a′。`,
});
