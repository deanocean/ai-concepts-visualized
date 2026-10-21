/* LSTM のゲート（シラバス §22；CEC は §27 キーワード）
   一個純量記憶單元 c，記住「上一個提到的人是女性(+)還是男性(−)」，到空格時填 she / he。
   三個閘門都是 sigmoid（0〜1），逐詞計算；可把任一閘門固定成 1，看少了它會出什麼事。
   權重是為了示範而設計的小數字（真實模型的閘門也會看上一步的 h，數字全由訓練學到）。 */
Atlas.define("lstm", {
  lead: `<p>普通 RNN 每一步都把隱藏狀態整個重算一次，早先的資訊很快被沖淡。<span class="jp">LSTM</span>（長短期記憶，<span class="en">long short-term memory</span>）另外準備一條<span class="jp">メモリセル</span>（記憶單元，<span class="en">memory cell</span>）c，再用三個<span class="jp">ゲート</span>（閘門，<span class="en">gate</span>）——0〜1 之間的開關——決定舊記憶留多少、新內容寫多少、此刻拿出多少。</p>`,
  closing: `忘記閘管「留」、輸入閘管「寫」、輸出閘管「拿」；三個閘都是 0〜1 的 sigmoid，所以開關的程度可以靠訓練學出來。記憶單元只做「乘上忘記閘、再加上新內容」，資訊和梯度都能沿著它傳得很遠。`,

  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const sig = (z) => 1 / (1 + Math.exp(-z));
    // 詞的特徵：p＝是否人名、sx＝性別（女 +1／男 −1）、q＝是否要填代名詞的空格、r＝其他內容
    const SENT = {
      A: { label: "Mary walked to the park and then ___", answer: "she", toks: [
        ["Mary", 1, 1, 0, 0], ["walked", 0, 0, 0, -0.6], ["to", 0, 0, 0, -0.2], ["the", 0, 0, 0, 0.1],
        ["park", 0, 0, 0, -0.5], ["and", 0, 0, 0, 0.1], ["then", 0, 0, 0, -0.3], ["___", 0, 0, 1, 0]] },
      B: { label: "Mary met John in the park and ___", answer: "he", toks: [
        ["Mary", 1, 1, 0, 0], ["met", 0, 0, 0, -0.6], ["John", 1, -1, 0, 0], ["in", 0, 0, 0, -0.2],
        ["the", 0, 0, 0, 0.1], ["park", 0, 0, 0, -0.5], ["and", 0, 0, 0, 0.1], ["___", 0, 0, 1, 0]] },
    };
    const GATE = { f: "忘記閘", i: "輸入閘", o: "輸出閘" };
    let sent = "A", mode = "n", pos = 0; // pos 0..T（已送入的詞數）

    function run(sk, md) {
      const out = []; let c = 0;
      for (const [w, p, sx, q, r] of SENT[sk].toks) {
        const zf = -8 * p + 5, zi = 8 * p - 4, zg = 2 * sx + 1.5 * r, zo = 8 * q - 4;
        let f = sig(zf), i = sig(zi), o = sig(zo);
        const g = Math.tanh(zg);
        if (md === "f") f = 1; if (md === "i") i = 1; if (md === "o") o = 1;
        const cPrev = c; c = f * c + i * g; const hh = o * Math.tanh(c);
        out.push({ w, p, sx, q, r, zf, zi, zg, zo, f, i, g, o, cPrev, c, h: hh });
      }
      return out;
    }
    const pShe = (hh) => sig(6 * hh);

    bench.append(h("style", { text: `
      .ls-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; color: var(--muted); }
      .ls-legend b { color: var(--ink); font-weight: 600; }
    ` }));

    /* ---------- controls ---------- */
    const sentSeg = ui.seg({ label: "句子（空格要填 she 還是 he？）", value: sent,
      options: [{ value: "A", label: "中間沒換人" }, { value: "B", label: "中間出現新人物" }],
      onChange: (v) => { sent = v; player.pause(); pos = 0; render(); } });
    const modeSeg = ui.seg({ label: "閘門", value: mode,
      options: [{ value: "n", label: "正常" }, { value: "f", label: "忘記閘固定 1" }, { value: "i", label: "輸入閘固定 1" }, { value: "o", label: "輸出閘固定 1" }],
      onChange: (v) => { mode = v; render(); } });
    const T = 8;
    const player = ui.player(ctx, () => { if (pos >= T) return false; pos++; render(); return pos < T; }, 900);
    const prevBtn = ui.btn("◀ 上一步", () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn("下一步 ▶", () => { player.pause(); if (pos < T) { pos++; render(); } }, { primary: true });
    const resetBtn = ui.btn("重設", () => { player.pause(); pos = 0; render(); });

    const sentLine = h("p", { class: "hint", style: { margin: 0 } });
    const status = h("div", { class: "status" });
    const cellWrap = h("div", { class: "scroll-x" });
    const calc = h("div", { class: "calc" });
    const lineWrap = h("div", { class: "scroll-x" });
    const note = h("p", { class: "note" });

    bench.append(
      h("div", { class: "bench__row" }, sentSeg.el, modeSeg.el),
      sentLine, status, cellWrap,
      h("div", { class: "bench__row" }, prevBtn, nextBtn, player.el, resetBtn),
      calc,
      h("div", { class: "field" }, h("span", { class: "field__label" }, "整句的經過（點一欄可跳到那一步）"), lineWrap),
      note,
    );

    /* ---------- helpers ---------- */
    function arrow(g, x1, y1, x2, y2, color, width = 1.6) {
      g.append(s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": width }));
      const a = Math.atan2(y2 - y1, x2 - x1), L = 6 + width;
      g.append(s("path", { d: `M${x2},${y2} L${x2 - L * Math.cos(a - 0.45)},${y2 - L * Math.sin(a - 0.45)} L${x2 - L * Math.cos(a + 0.45)},${y2 - L * Math.sin(a + 0.45)} Z`, fill: color }));
    }
    const fixed = (k) => mode === k;

    /* ---------- the cell at the current step ---------- */
    function drawCell(R) {
      const W = 760, H = 300;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "LSTM 單元這一步的計算", style: `display:block;width:100%;min-width:${W * 0.78}px;max-width:${W}px` });
      const t = pos - 1, st = t >= 0 ? R[t] : null;
      const on = !!st, ink = on ? "var(--ink)" : "var(--faint)", wire = on ? "var(--muted)" : "var(--line-strong)";
      const yC = 52, xF = 220, xA = 400, yMul = 112, yG = 172, xI = 335, xGc = 465, xO = 590, xOut = 680;

      // memory highway
      svg.append(s("line", { x1: 118, y1: yC, x2: 548, y2: yC, stroke: "var(--line-strong)", "stroke-width": 7, "stroke-linecap": "round", "stroke-opacity": 0.55 }));
      svg.append(s("text", { x: (xF + xA) / 2, y: 18, "text-anchor": "middle", "font-size": 11.5, class: "t-muted", text: "記憶單元的直通路（只有乘法和加法）" }));
      const box = (x, y, w, hh, val, label, sub, max = 1.2) => {
        svg.append(s("rect", { x, y, width: w, height: hh, rx: 8, fill: on && val != null ? Atlas.divColor(val, max) : "var(--panel-2)", stroke: "var(--line-strong)" }));
        svg.append(s("text", { x: x + w / 2, y: y - 6, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: label }));
        svg.append(s("text", { x: x + w / 2, y: y + hh / 2 + 5, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: on && val != null && Math.abs(val) > 0.75 * max ? "var(--panel)" : ink, text: on && val != null ? fmt(val, 2) : "?" }));
        if (sub) svg.append(s("text", { x: x + w / 2, y: y + hh + 15, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: sub }));
      };
      box(18, yC - 18, 96, 36, st ? st.cPrev : null, "舊記憶 c", "上一步留下的");
      box(552, yC - 18, 96, 36, st ? st.c : null, "新記憶 c", "");
      const op = (x, y, sym, hot) => {
        svg.append(s("circle", { cx: x, cy: y, r: 15, fill: "var(--panel)", stroke: hot ? "var(--hot)" : wire, "stroke-width": 2 }));
        svg.append(s("text", { x, y: y + 6, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: hot ? "var(--hot)" : ink, text: sym }));
      };
      // gates
      const gate = (x, key, val, z) => {
        const w = 104, hh = 54, y = yG;
        const isFix = fixed(key);
        svg.append(s("rect", { x: x - w / 2, y, width: w, height: hh, rx: 8, fill: "var(--panel)", stroke: isFix ? "var(--hot)" : "var(--line-strong)", "stroke-width": isFix ? 2 : 1, "stroke-dasharray": isFix ? "5 3" : null }));
        svg.append(s("text", { x, y: y + 17, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: GATE[key] + " " + key }));
        svg.append(s("rect", { x: x - 40, y: y + 25, width: 80, height: 9, rx: 4, fill: "var(--panel-2)", stroke: "var(--line)" }));
        if (on) svg.append(s("rect", { x: x - 40, y: y + 25, width: 80 * val, height: 9, rx: 4, fill: isFix ? "var(--hot)" : "var(--accent)" }));
        svg.append(s("text", { x, y: y + 48, "text-anchor": "middle", "font-size": 12.5, class: "t-mono", fill: ink, text: on ? (isFix ? "固定 1" : fmt(val, 3)) : "?" }));
        void z;
      };
      const g1 = s("g");
      // c_prev -> xF -> xA -> c_new
      arrow(g1, 114, yC, xF - 16, yC, wire, 2);
      arrow(g1, xF + 16, yC, xA - 16, yC, wire, 2);
      arrow(g1, xA + 16, yC, 550, yC, wire, 2);
      // f gate up into ×
      arrow(g1, xF, yG, xF, yC + 16, wire, 1.6);
      // i, g into the lower × then up into +
      arrow(g1, xI, yG, xA - 11, yMul + 11, wire, 1.6);
      arrow(g1, xGc, yG, xA + 11, yMul + 11, wire, 1.6);
      arrow(g1, xA, yMul - 15, xA, yC + 16, wire, 1.6);
      // output path: c_new -> tanh -> × -> h
      g1.append(s("polyline", { points: `648,${yC} ${xO + 70},${yC} ${xO + 70},${yMul - 17}`, fill: "none", stroke: wire, "stroke-width": 1.6 }));
      arrow(g1, xO + 70, yMul - 17, xO + 70, yMul - 14, wire, 1.6);
      svg.append(g1);
      // tanh box on output path
      svg.append(s("rect", { x: xO + 46, y: yMul - 14, width: 48, height: 26, rx: 6, fill: "var(--panel)", stroke: wire }));
      svg.append(s("text", { x: xO + 70, y: yMul + 4, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: ink, text: "tanh" }));
      const g2 = s("g");
      arrow(g2, xO + 70, yMul + 12, xO + 70, yMul + 45, wire, 1.6);
      arrow(g2, xO, yG, xO + 58, yMul + 55, wire, 1.6);
      arrow(g2, xO + 70, yMul + 75, xO + 70, 262, wire, 1.6);
      svg.append(g2);
      op(xF, yC, "×", on && st.f < 0.5);
      op(xA, yC, "+", false);
      op(xA, yMul, "×", on && st.i > 0.5);
      op(xO + 70, yMul + 60, "×", on && st.o > 0.5);
      gate(xF, "f", st ? st.f : 0);
      gate(xI, "i", st ? st.i : 0);
      // candidate box
      svg.append(s("rect", { x: xGc - 52, y: yG, width: 104, height: 54, rx: 8, fill: on ? Atlas.divColor(st.g, 1) : "var(--panel-2)", stroke: "var(--line-strong)" }));
      svg.append(s("text", { x: xGc, y: yG + 17, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: on && Math.abs(st.g) > 0.75 ? "var(--panel)" : "var(--ink)", text: "候選內容 g" }));
      svg.append(s("text", { x: xGc, y: yG + 42, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: on && Math.abs(st.g) > 0.75 ? "var(--panel)" : ink, text: on ? fmt(st.g, 3) : "?" }));
      gate(xO, "o", st ? st.o : 0);
      // h output
      const hv = st ? st.h : null;
      svg.append(s("rect", { x: xO + 26, y: 262, width: 88, height: 32, rx: 8, fill: on ? Atlas.divColor(hv, 1) : "var(--panel-2)", stroke: "var(--line-strong)" }));
      svg.append(s("text", { x: xO + 70, y: 283, "text-anchor": "middle", "font-size": 13.5, class: "t-mono", fill: on && Math.abs(hv) > 0.75 ? "var(--panel)" : ink, text: on ? fmt(hv, 2) : "?" }));
      svg.append(s("text", { x: xO + 120, y: 283, "font-size": 12, "font-weight": 700, text: "輸出 h" }));
      // input bus
      const yBus = 252;
      svg.append(s("line", { x1: 118, y1: yBus, x2: xO, y2: yBus, stroke: wire, "stroke-width": 1.4 }));
      [xF, xI, xGc, xO].forEach((x) => arrow(svg, x, yBus, x, yG + 57, wire, 1.4));
      svg.append(s("rect", { x: 10, y: yBus - 20, width: 104, height: 40, rx: 8, fill: "var(--hot-soft)", stroke: "var(--hot)" }));
      svg.append(s("text", { x: 62, y: yBus - 26, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: "這一步的詞" }));
      svg.append(s("text", { x: 62, y: yBus + 6, "text-anchor": "middle", "font-size": 15, class: "t-mono", "font-weight": 600, text: st ? st.w : "—" }));
      // tags of the word
      if (st) {
        const tag = st.p ? `人名・${st.sx > 0 ? "女性" : "男性"}` : st.q ? "要填代名詞" : "一般詞";
        svg.append(s("text", { x: 62, y: yBus + 36, "text-anchor": "middle", "font-size": 11.5, fill: "var(--hot)", text: tag }));
      }
      cellWrap.replaceChildren(svg);
    }

    /* ---------- the whole sentence ---------- */
    function drawLine(R) {
      const COLW = 78, LEFT = 82, W = LEFT + COLW * T + 8;
      const rows = [["詞", 18], ["忘記閘 f", 50], ["輸入閘 i", 78], ["輸出閘 o", 106], ["記憶 c", 142], ["輸出 h", 182]];
      const H = 206;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "整句各步的閘門與記憶", style: `display:block;width:100%;min-width:${W * 0.8}px;max-width:${W}px` });
      rows.forEach(([t, y]) => svg.append(s("text", { x: 4, y: y + 4, "font-size": 11.5, "font-weight": 700, class: t === "詞" ? "t-muted" : "", text: t })));
      R.forEach((st, t) => {
        const x = LEFT + COLW * t, cx = x + COLW / 2, on = t < pos, cur = t === pos - 1;
        const col = s("g", { style: "cursor:pointer" });
        col.append(s("rect", { x: x + 2, y: 2, width: COLW - 4, height: H - 4, rx: 8, fill: cur ? "var(--hot-soft)" : "transparent", stroke: cur ? "var(--hot)" : "none" }));
        col.append(s("text", { x: cx, y: 22, "text-anchor": "middle", "font-size": 13, class: "t-mono", "font-weight": cur ? 700 : 400, fill: on ? "var(--ink)" : "var(--faint)", text: st.w }));
        ["f", "i", "o"].forEach((k, j) => {
          const y = rows[j + 1][1];
          col.append(s("rect", { x: cx - 30, y: y - 7, width: 60, height: 10, rx: 4, fill: "var(--panel-2)", stroke: "var(--line)" }));
          if (on) col.append(s("rect", { x: cx - 30, y: y - 7, width: 60 * st[k], height: 10, rx: 4, fill: fixed(k) ? "var(--hot)" : "var(--accent)" }));
          if (on) col.append(s("text", { x: cx, y: y + 16, "text-anchor": "middle", "font-size": 9.5, class: "t-mono t-muted", text: fmt(st[k], 2) }));
        });
        const cell = (y, v) => {
          col.append(s("rect", { x: cx - 30, y: y - 14, width: 60, height: 26, rx: 6, fill: on ? Atlas.divColor(v, 1.2) : "var(--panel-2)", stroke: "var(--line-strong)" }));
          col.append(s("text", { x: cx, y: y + 4, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: on && Math.abs(v) > 0.9 ? "var(--panel)" : on ? "var(--ink)" : "var(--faint)", text: on ? fmt(v, 2) : "?" }));
        };
        cell(rows[4][1], st.c); cell(rows[5][1], st.h);
        col.addEventListener("click", () => { player.pause(); pos = t + 1; render(); });
        svg.append(col);
      });
      lineWrap.replaceChildren(svg);
    }

    /* ---------- status & calc ---------- */
    function render() {
      const R = run(sent, mode), Rn = run(sent, "n");
      const S = SENT[sent];
      sentLine.innerHTML = `句子：<b class="mono">${S.label}</b>　正解：<b class="mono">${S.answer}</b>`;
      drawCell(R); drawLine(R);
      const set = (k, t) => { status.className = "status status--" + k; status.textContent = t; };
      const t = pos - 1, st = R[t];
      if (pos === 0) {
        set("info", "按「下一步」逐詞送入，看記憶單元 c 怎麼記住「上一個人是女性（+）還是男性（−）」");
        calc.innerHTML = `<span class="k">每個詞先轉成 4 個特徵：是否人名、性別（女 +1／男 −1）、是否是要填代名詞的空格、其他內容（用一個數字代表）。三個閘門和候選內容都從這些特徵算出來。</span>`;
      } else {
        const fz = (k, v) => (fixed(k) ? `${v} → <span class="hl">固定成 1</span>` : v);
        calc.innerHTML =
          `<span class="calc-row"><span class="k">特徵</span> 人名 ${st.p}、性別 ${fmt(st.sx, 0)}、空格 ${st.q}、其他內容 ${fmt(st.r, 1)}</span>` +
          `<span class="calc-row"><span class="k">忘記閘</span> f = σ(−8×人名 + 5) = σ(${fmt(st.zf, 0)}) = ${fz("f", fmt(sig(st.zf), 3))}</span>` +
          `<span class="calc-row"><span class="k">輸入閘</span> i = σ(8×人名 − 4) = σ(${fmt(st.zi, 0)}) = ${fz("i", fmt(sig(st.zi), 3))}</span><span class="calc-row"><span class="k">候選內容</span> g = tanh(2×性別 + 1.5×其他內容) = tanh(${fmt(st.zg, 2)}) = ${fmt(st.g, 3)}</span>` +
          `<span class="calc-row"><span class="k">輸出閘</span> o = σ(8×空格 − 4) = σ(${fmt(st.zo, 0)}) = ${fz("o", fmt(sig(st.zo), 3))}</span>` +
          `<span class="calc-row"><span class="k">新記憶</span> c = f × 舊記憶 + i × g = ${fmt(st.f, 3)} × ${fmtP(st.cPrev, 3)} + ${fmt(st.i, 3)} × ${fmtP(st.g, 3)} = <span class="hl">${fmt(st.c, 3)}</span></span>` +
          `<span class="calc-row"><span class="k">輸出</span> h = o × tanh(c) = ${fmt(st.o, 3)} × ${fmt(Math.tanh(st.c), 3)} = <span class="hl">${fmt(st.h, 3)}</span></span>` +
          (st.q ? `<span class="calc-row"><span class="k">預測</span> P(she) = σ(6 × h) = σ(${fmt(6 * st.h, 2)}) = <span class="hl">${fmt(pShe(st.h), 3)}</span>，P(he) = ${fmt(1 - pShe(st.h), 3)}</span>` : "");

        if (st.q) {
          const P = pShe(st.h), pred = P >= 0.5 ? "she" : "he", conf = Math.max(P, 1 - P);
          const ok = pred === S.answer;
          const cN = Rn[t].c;
          if (mode === "o") set("warn", `答案 ${pred}（${Math.round(conf * 100)}%）沒錯，但輸出閘一直全開：句子中間每一步的輸出 h 都約 ${fmt(R[1].h, 2)}，把「誰」的記憶硬塞進每個位置的輸出`);
          else if (mode === "f" && conf < 0.65) set("bad", `分不出來：P(she) = ${fmt(P, 2)}。忘記閘不會關，舊的「女性」沒清掉，和新寫入的「男性」互相抵消（c = ${fmt(st.c, 2)}）`);
          else if (mode === "i" && !ok) set("bad", `答錯：預測 ${pred}（${Math.round(conf * 100)}%）。輸入閘一直全開，無關的詞也一路寫進記憶，c 從 ${fmt(R[0].c, 2)} 被推到 ${fmt(st.c, 2)}`);
          else if (mode === "i") set("warn", `這次碰巧答對 ${pred}，但 c = ${fmt(st.c, 2)}（正常是 ${fmt(cN, 2)}）：無關的詞也寫進了記憶，結果只是剛好偏向同一邊`);
          else set(ok ? "ok" : "bad", `要填代名詞：輸出閘打開（${fmt(st.o, 2)}），把記憶拿出來 → 預測 ${pred}（${Math.round(conf * 100)}%）${ok ? "，正確" : "，錯誤"}`);
        } else if (st.p) {
          const who = st.sx > 0 ? "女性" : "男性";
          if (mode === "f" && t > 0) set("warn", `人名「${st.w}」：忘記閘被固定成 1，舊記憶清不掉，新的「${who}」直接疊上去 → c = ${fmt(st.c, 2)}`);
          else set("info", `人名「${st.w}」：忘記閘${mode === "f" ? "固定 1" : `幾乎關上（${fmt(st.f, 2)}），清掉舊記憶`}；輸入閘打開（${fmt(st.i, 2)}），寫入「${who}」→ c = ${fmt(st.c, 2)}`);
        } else {
          if (mode === "i") set("warn", `「${st.w}」不是人名，但輸入閘被固定成 1：它的其他內容（g = ${fmt(st.g, 2)}）也寫進記憶 → c 從 ${fmt(st.cPrev, 2)} 變成 ${fmt(st.c, 2)}`);
          else if (mode === "o") set("warn", `「${st.w}」：記憶保留得好好的，但輸出閘被固定成 1，這一步也把記憶整個輸出（h = ${fmt(st.h, 2)}）`);
          else set("info", `「${st.w}」不是人名：忘記閘幾乎全開（${fmt(st.f, 3)}），輸入閘幾乎關上（${fmt(st.i, 3)}）→ 記憶幾乎不變（${fmt(st.cPrev, 2)} → ${fmt(st.c, 2)}），輸出閘關著（h ≈ ${fmt(st.h, 2)}）`);
        }
      }
      note.innerHTML = `σ 是シグモイド関数，輸出永遠在 0〜1 之間，所以適合當「開多少」的比例；tanh 輸出在 −1〜1，用來產生可正可負的內容。為了好讀，這裡的閘門只看當下的詞，真實的 LSTM 還會一起看上一步的輸出 h，權重全由訓練學到。記憶單元這條只有乘法和加法的路，原始論文稱為 <span class="jp">CEC</span>（定誤差カルーセル，<span class="en">constant error carousel</span>）：誤差沿它往回傳不會被 tanh 壓縮（對照 <a href="#rnn">RNN と勾配消失</a> 的「換成 LSTM」）。把忘記閘和輸入閘合併成一個<span class="jp">更新ゲート</span>（更新閘，<span class="en">update gate</span>）、再加上<span class="jp">リセットゲート</span>（重設閘，<span class="en">reset gate</span>）、不另設記憶單元的簡化版，叫 <span class="jp">GRU</span>（<span class="en">gated recurrent unit</span>），參數較少。`;
      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= T;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
