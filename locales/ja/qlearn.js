/* Q 学習と SARSA（concepts/qlearn.js）の日本語テキスト
   - である調。key は locales/zh/qlearn.js と一致させる */
Atlas.strings("ja", "qlearn", {
  lead: `<p>ロボットがスタートからゴールまで歩く。1 歩進むごとに 1 点減り、崖から落ちると 100 点減ってスタートに戻る。どう歩けばよいかは誰も教えてくれず、自分で試すしかない。ロボットは各マスの各行動に Q(s, a) という数値を記録する。「このマスでこう動くと、この先合計でどれだけ報酬が得られるか」であり、これを<span class="term">行動価値関数</span>（<span class="en">action-value function</span>）という。1 歩進むたびに、「得た報酬 + 次のマスの価値」で直前のマスの Q 値を修正する。<span class="term">Q 学習</span>（<span class="en">Q-learning</span>）も <span class="term">SARSA</span> もこうして学び、違いは「次のマスの価値」の計算方法だけである。</p>`,
  closing: `どちらも ε-greedy で歩きながらでたらめに試し、どちらも「報酬 + 次のマスの価値」で更新する。Q 学習は次のマスで<b>最もよい</b>行動の値を使う。これはこの先ずっと最適な道を歩くと仮定することに等しいため、崖沿いの最短路を学ぶが、訓練中にでたらめに動くとすぐ落ちる。SARSA は次に<b>実際にとる</b>行動（でたらめな動きも含む）の値を使い、「崖のそばではたまに足を踏み外す」ことまで Q 値に織り込む。そのため少し遠回りの安全な道を学び、訓練中に落ちる回数もずっと少ない。`,

  /* 方向（マスの表と文で共用） */
  dirUp: "上",
  dirRight: "右",
  dirDown: "下",
  dirLeft: "左",
  cellName: "上から {r} マス目・左から {c} マス目",

  /* 操作 */
  gLabel: "割引率 γ",
  g1: "1（割り引かない）",
  btnSeed: "初期値を変える",
  btnRun: "▶ 訓練開始",
  btnResume: "▶ 訓練再開",
  btnPause: "❚❚ 一時停止",
  btnDone: "✓ 訓練完了",
  btnStep: "1 歩進める",
  btnEp: "1 エピソード進める",
  btnFf: "⏭ 最後まで早送り",

  /* 図（読み上げ用） */
  qAria: "Q 学習の Q 値と経路",
  sAria: "SARSA の Q 値と経路",
  chartAria: "エピソードごとの報酬の合計",

  /* 図の説明 */
  key: `<span><i style="background:var(--neg-soft);border:1px solid var(--neg)"></i>崖</span><span>矢印＝現在 Q 値が最も高い行動</span><span>マスの数値＝最大の Q 値（背景が青いほど負）</span><span><span class="ql-sw" style="color:var(--hot)"></span>最大の Q 値だけをたどった経路</span><span>● エージェントの現在位置</span>`,
  cellLabel: "マスをクリックすると、そのマスの 4 つの Q 値を 2 つの手法で比べられる",
  capChart: `エピソードごとの報酬の合計（直近 20 エピソードの平均。−120 以下は下端に描く）：<span style="color:var(--hot)">━ Q 学習</span>　<span style="color:var(--pos)">━ SARSA</span>`,
  goal: "ゴール",
  start: "出発",
  cliff: "崖（−100、スタートに戻る）",
  gridHead: "{ep} エピソード終了　崖から落ちた回数 {f}",
  gridHeadSteps: "{ep} エピソード終了　崖から落ちた回数 {f}　このエピソードで {s} 歩",
  capQ: "Q 学習 <small>目標 = r + γ × max Q(s′, ·)</small>",
  capS: "SARSA <small>目標 = r + γ × Q(s′, a′)</small>",
  chartAxis: "エピソード",

  /* ステータス */
  statusIdle: "まだ訓練していない。Q 値はすべて 0 で、2 つのエージェントはでたらめに歩くしかない。「1 歩進める」で 1 歩ごとの更新を見るか、「▶ 訓練開始」で 800 エピソード続けて実行してみよう（約 6 秒）",
  descCliff: "最大の Q 値どおりに歩くと崖から落ちる（まだ学習できていない）",
  descLoop: "最大の Q 値どおりに歩くと同じ所を回り、ゴールに着かない（まだ学習できていない）",
  descEdge: "{len} 歩で、ずっと崖沿いを進む",
  descUp: "{len} 歩で、最も上は上から {row} マス目まで",
  status: "{e} エピソード完了：Q 学習の経路は{q}。SARSA の経路は{s}",
  statusRun: "訓練中、{e} エピソード完了：Q 学習の経路は{q}。SARSA の経路は{s}",
  statusEnd: "。訓練中に崖から落ちた回数は Q 学習 {fq} 回、SARSA {fs} 回。最後の 50 エピソードの平均報酬は Q 学習 {rq}、SARSA {rs}",

  /* 計算式 */
  nameQ: "Q 学習",
  lastK: "最後の 1 歩の更新",
  lastV: "（学習率 α = 0.2、ε = 0.1）",
  notYet: "まだ歩いていない",
  move: "{cell}で{dir}へ進む → {res}",
  moveRand: "{cell}で{dir}へ進む（ε によるでたらめな動き） → {res}",
  resFell: "崖から落ちて報酬 −100、スタートに戻る",
  resGoal: "ゴールに到着、報酬 −1",
  resMove: "報酬 −1、{cell}へ",
  tgtDone: `ゴールの後に報酬はないので、目標 = r = <span class="hl">{v}</span>`,
  tgtQ: `目標 = r + γ × max Q(s′, ·) = {r} + {g} × max({qs}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  tgtS: `次に実際に{dir}へ進む：目標 = r + γ × Q(s′, {dir}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  tgtSRand: `次に実際に{dir}へ進む（ε によるでたらめな動きで選ばれた）：目標 = r + γ × Q(s′, {dir}) = {r} + {g} × {n} = <span class="hl">{v}</span>`,
  cliffCell: "ここは崖で、エージェントがとどまることはないので Q 値はない",
  goalCell: "ここはゴールで、着いたら終わりなので Q 値はない",
  cmpK: "比べてみる",
  cmp: "このマスは崖のすぐ上にある。SARSA の「右」{s} は Q 学習の {q} より低い。SARSA は「この先 10% の確率ででたらめに動き、下に落ちるかもしれない」ことまで計算に入れているからである",

  note: `世界を「状態 → 行動 → 報酬と次の状態」として表し、次の状態が現在の状態と行動だけで決まるという設定を<span class="term">マルコフ決定過程</span>（<span class="en">Markov decision process, MDP</span>）という。<span class="term">割引率</span>（<span class="en">discount rate</span>）γ は遠い将来の報酬ほど小さく数える。γ = 1 のときスタートの Q 値はおよそ「ゴールまであと何点減るか」に等しく、γ = 0.9 のとき 11 歩の道は −1 − 0.9 − 0.81 − … ≈ −6.9 になる。行動を区別せず、マスそのものの良し悪しだけを見るものを<span class="term">状態価値関数</span>（<span class="en">state-value function</span>）という。Q 学習も SARSA もまず「価値」を学び、価値に従って行動を選ぶ。もう一つの系統は「行動の選び方」の確率を直接学ぶもので、<span class="term">方策勾配法</span>（<span class="en">policy gradient method</span>、代表例は REINFORCE）という。両者を組み合わせたものが <span class="term">Actor-Critic</span> である。マスが多くなると Q 値を表に保存できなくなり、ニューラルネットワークで Q 値を推定するのが DQN である（シラバス §29）。行動の選択に使う ε-greedy は <a href="#bandit">バンディット</a> を参照。SARSA という名前は、更新に使う 5 つの量 s, a, r, s′, a′ に由来する。`,
});
